"""Training config / model-family / guard tests (docs/design.md section 8). Offline, no GPU, no Docker: models are tiny
random-init configs on meta, datasets are task directories with a task.toml, the vLLM server is a function."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from honest_agents import models, train, trl_rewards
from honest_agents.models import ModelFamily
from honest_agents.train import run as run_module
from honest_agents.train import startup as startup_module

REPO = Path(__file__).resolve().parents[1]
# pilot and sft_ configs have their own schemas (tests/pilot and tests/sft load them)
CONFIGS = sorted(p for p in (REPO / "configs").glob("*.yaml") if not p.name.startswith(("pilot", "sft_")))


# ── fixtures: a dataset of task directories ─────────────────────────────────


def write_task(root: Path, name: str, *, feasible: bool = True, verifier: str | None = "pytest", control: bool = False,
               targets: bool = True) -> str:  # fmt: skip
    d = root / "tasks" / name
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"feasible = {str(feasible).lower()}"]
    if verifier:
        lines.append(f'verifier = "{verifier}"')
    if not feasible:
        lines.append('expected_blockers = ["missing_capability"]')
        if targets:
            lines.append('expected_targets = ["/usr/bin/jq"]')
    if control:
        lines.append("is_control = true")
    (d / "task.toml").write_text("[metadata.honest]\n" + "\n".join(lines) + "\n")
    return str(d)


def make_rows(root: Path, prefix: str, n: int, split: str, **task_kw) -> list[dict]:
    rows = []
    for i in range(n):
        feasible = i % 4 != 3  # every 4th row is infeasible; pairs are (0,1), (2,3), ...
        rows.append(
            {
                "task_dir": write_task(root, f"{prefix}{i}", feasible=feasible, **task_kw),
                "variant_id": f"{prefix}{i}",
                "pair_id": f"{prefix}p{i // 2}",
                "feasible": feasible,
                "kind": "feasible" if feasible else "infeasible",
                "split": split,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def data(tmp_path: Path) -> SimpleNamespace:
    """16 training rows (two full base-config rounds of 8 prompts) and 8 validation rows."""
    train_file = write_jsonl(tmp_path / "train.jsonl", make_rows(tmp_path, "t", 16, "train"))
    val_file = write_jsonl(tmp_path / "validation.jsonl", make_rows(tmp_path, "v", 8, "validation"))
    return SimpleNamespace(root=tmp_path, train=train_file, val=val_file)


def _cfg(config: Path, data: SimpleNamespace, *overrides: str) -> train.TrainRunConfig:
    plain = train.load_run_config(config)
    extra = [f"data.eval_file={data.val}"] if plain.data.eval_file else []
    return train.load_run_config(
        config, [f"data.train_file={data.train}", f"output_root={data.root / 'out'}", *extra, *overrides]
    )


def _plan(config: Path, data: SimpleNamespace, *overrides: str) -> train.Plan:
    return train.build_plan(_cfg(config, data, *overrides))


BASE = REPO / "configs" / "base.yaml"


# ── every shipped config dry-runs ────────────────────────────────────────────


def test_configs_exist():
    names = {p.stem for p in CONFIGS}
    assert {"base", "smoke", "qwen35_4b", "phase0_zero_lr", "outcome", "outcome_claimed", "outcome_reflect"} <= names
    assert {f"honest_alpha{a}" for a in ("0", "0.1", "0.25", "0.4")} <= names
    assert not {"honest_alpha0.5", "honest_alpha0.75", "outcome_only"} & names  # replaced by the v3 grid


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_dry_run_builds(config: Path, data: SimpleNamespace):
    plan = _plan(config, data)
    g, cfg = plan.grpo_config, plan.cfg
    assert g.max_tool_calling_iterations is not None
    assert g.chat_template_kwargs == {"enable_thinking": False}
    assert g.model_init_kwargs["dtype"] == "bfloat16"
    # exactly one reward function per run, weight 1: a None is a mask, not a zero
    assert [f.__name__ for f in plan.reward_funcs] == [f"{cfg.reward.mode}_reward"]
    assert plan.reward_funcs[0] is trl_rewards.reward_func_for_mode(cfg.reward.mode)
    assert list(g.reward_weights) == [1.0]
    assert g.scale_rewards == "none"
    assert (g.top_p, g.top_k, g.min_p, g.repetition_penalty) == (1.0, 0, 0.0, 1.0)
    assert g.save_total_limit is None
    assert plan.generation_batch_size % g.num_generations == 0
    assert plan.generation_batch_size == g.per_device_train_batch_size * g.gradient_accumulation_steps
    assert g.output_dir.endswith(g.run_name)
    assert plan.cfg.launch.num_processes == 1
    # one source of truth for limits: environment budget, reward reference and server window all follow grpo:
    assert plan.limits.turn_max_tokens == g.generation_kwargs["max_tokens"]
    assert plan.budget.max_bash_calls == g.max_tool_calling_iterations - 2
    assert plan.reward_config.efficiency_ref_actions == plan.budget.max_bash_calls
    assert plan.environment_factory.keywords["budget"] is plan.budget
    assert plan.summary()["limits"]["max_model_len"] == (
        plan.limits.prompt_token_reserve + g.max_completion_length + g.generation_kwargs["max_tokens"]
    )
    assert plan.data_counts["train"]["rows"] == (cfg.data.max_rows or 16)
    assert len(plan.train_dataset) >= plan.data_counts["train"]["rows"] // 2  # one or more passes over the rows
    assert (plan.eval_dataset is not None) == (cfg.data.eval_file is not None)
    json.dumps(plan.summary(), default=str)


def test_every_epoch_gets_its_own_pair_order_and_rounds_stay_whole(data: SimpleNamespace):
    """TRL replays a shuffle-less dataset in one order every epoch: with several passes the dataset holds one
    independently shuffled pass per epoch, every round a whole number of pairs, and one pass is unchanged."""
    one = _plan(BASE, data, "grpo.max_steps=2")  # 2 rounds of 8 prompts = exactly one pass over the 16 rows
    assert one.data_counts["train"]["epochs"] == 1
    assert one.train_dataset["variant_id"] == [r["variant_id"] for r in train.pair_ordered(
        train.load_rows(data.train), one.cfg.seed, 8)]  # fmt: skip
    many = _plan(BASE, data, "grpo.max_steps=7")  # 7 rounds -> 4 passes of 2 rounds
    ids = many.train_dataset["variant_id"]
    assert many.data_counts["train"]["epochs"] == 4 and len(ids) == 64
    passes = [ids[i : i + 16] for i in range(0, 64, 16)]
    assert all(sorted(p) == sorted(passes[0]) for p in passes)  # every pass shows every variant once
    assert len({tuple(p) for p in passes}) > 1  # ... in a different order
    pair = dict(zip(many.train_dataset["variant_id"], many.train_dataset["pair_id"], strict=True))
    for r in range(0, 64, 8):  # each round of 8 prompts holds whole pairs
        pairs = [pair[v] for v in ids[r : r + 8]]
        assert all(pairs.count(x) == 2 for x in pairs)
    assert train.order_fingerprint(train.prepare_train_rows(many.cfg, round_size=8), many.cfg.seed, 8) == many.order


def test_the_probe_trainer_is_memory_safe_and_times_every_step(tmp_path: Path):
    """TRL scores an evaluation batch with `per_device_eval_batch_size` rows at once (the probe wants ~25 episodes in
    flight) and again inside compute_loss: far beyond what a 16k-token batch fits. The probe trainer caps the chunk at
    the training micro-batch and never runs the loss. Real (tiny, CPU) TRL run."""
    import torch
    from datasets import Dataset
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    tok = _cached_tokenizer("Qwen/Qwen3-0.6B-Base")
    model_dir = tmp_path / "tiny"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)  # fmt: skip
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    seen: list[tuple[bool, int | None]] = []
    losses: list[bool] = []
    full_logits = GRPOTrainer._full_logits_logps
    compute = GRPOTrainer._compute_loss

    def spy_logps(self, model, *a, batch_size=None, **k):
        # HonestGRPOTrainer chunks by `batch_size` itself (length-trimmed passes), so TRL's method sees the chunk's rows.
        seen.append((self.model.training, a[0].size(0) if batch_size is None else batch_size))
        return full_logits(self, model, *a, batch_size=batch_size, **k)

    def spy_loss(self, model, inputs):
        losses.append(self.model.training)
        return compute(self, model, inputs)

    GRPOTrainer._full_logits_logps, GRPOTrainer._compute_loss = spy_logps, spy_loss
    try:
        args = GRPOConfig(
            output_dir=str(tmp_path / "run"), per_device_train_batch_size=2, gradient_accumulation_steps=2,
            num_generations=2, max_completion_length=3, max_steps=2, save_strategy="no", do_eval=True,
            eval_strategy="steps", eval_steps=1, eval_on_start=False, per_device_eval_batch_size=6,
            num_generations_eval=1, use_cpu=True, report_to="none", scale_rewards="none", shuffle_dataset=False,
            logging_steps=1, bf16=False, vllm_importance_sampling_correction=True,
        )  # fmt: skip

        def reward(prompts, completions, log_metric=None, **kwargs):
            log_metric("honest/task_success", 1.0)
            return [0.0] * len(prompts)

        trainer = train.probe_safe_trainer_class()(
            model=str(model_dir), reward_funcs=reward, args=args,
            train_dataset=Dataset.from_list([{"prompt": f"row {i}"} for i in range(8)]),
            eval_dataset=Dataset.from_list([{"prompt": f"val {i}"} for i in range(6)]),
        )  # fmt: skip
        trainer.train()
        logged = [row for row in trainer.state.log_history if "reward" in row]  # train logs
        assert logged and all(row[train.METRIC_GRADING_SECONDS] >= 0 for row in logged)
        assert all(row[train.METRIC_TRAIN_SECONDS] > 0 for row in logged)  # compute only: generation is excluded
        assert not any(training is False for training in losses), "the probe must not run the loss"
        assert any(training for training in losses)  # training still does
        # Without a vLLM server TRL does not score rollouts at generation time, so call the scorer as it would.
        ids = torch.ones(6, 8, dtype=torch.long)
        trainer.model.eval()
        seen.clear()
        for requested in (6, None):
            trainer._get_per_token_logps_and_entropies(trainer.model, ids, ids, 3, batch_size=requested)
        trainer.model.train()
        trainer._get_per_token_logps_and_entropies(trainer.model, ids, ids, 3, batch_size=6)
    finally:
        GRPOTrainer._full_logits_logps, GRPOTrainer._compute_loss = full_logits, compute
    # evaluation is capped at the training micro-batch (two rows per forward pass); training scores its own batch whole
    assert seen == [(False, 2)] * 6 + [(True, 6)]


@pytest.mark.parametrize("alpha", ["0", "0.1", "0.25", "0.4"])
def test_alpha_configs(alpha: str, data: SimpleNamespace):
    plan = _plan(REPO / "configs" / f"honest_alpha{alpha}.yaml", data)
    assert plan.reward_config.alpha == float(alpha)
    assert plan.cfg.reward.mode == "honest"
    assert plan.grpo_config.run_name == f"honest-a{float(alpha):g}-qwen3-4b-s42"


@pytest.mark.parametrize("mode", ["outcome", "outcome_claimed", "outcome_reflect"])
def test_outcome_arms(mode: str, data: SimpleNamespace):
    plan = _plan(REPO / "configs" / f"{mode}.yaml", data)
    assert plan.cfg.reward.mode == mode
    assert [f.__name__ for f in plan.reward_funcs] == [f"{mode}_reward"]
    assert plan.grpo_config.run_name == f"{mode.replace('_', '-')}-qwen3-4b-s42"


def test_arms_share_everything_but_the_reward(data: SimpleNamespace):
    """Identical harness, caps, sampling and geometry in every arm; only the optimised number differs."""
    ref = _plan(REPO / "configs" / "honest_alpha0.25.yaml", data)
    for mode in ("outcome", "outcome_claimed", "outcome_reflect"):
        other = _plan(REPO / "configs" / f"{mode}.yaml", data)
        for key in ("limits", "budget", "sampling", "importance_sampling", "num_generations", "probe", "stop_rules",
                    "generation_batch_size_at_launch", "learning_rate", "max_steps", "task_order"):  # fmt: skip
            assert other.summary()[key] == ref.summary()[key], (mode, key)


def test_base_geometry_and_server_mode(data: SimpleNamespace):
    plan = _plan(BASE, data)
    g = plan.grpo_config
    assert (g.loss_type, g.beta, g.num_generations) == ("dapo", 0.0, 8)
    assert (g.max_completion_length, g.max_tool_calling_iterations) == (16384, 30)
    assert g.generation_kwargs == {"max_tokens": 4096}
    assert g.mask_truncated_completions and g.use_vllm and g.vllm_mode == "server"
    assert g.vllm_importance_sampling_correction and g.vllm_importance_sampling_mode == "token_truncate"
    assert plan.generation_batch_size == 64 and plan.generation_batch_size // g.num_generations == 8
    assert (g.per_device_train_batch_size, g.gradient_accumulation_steps) == (1, 64)  # 64 episodes, one process
    assert g.shuffle_dataset is False  # pair_batches owns the order
    assert plan.limits.max_model_len == 3072 + 16384 + 4096 == 23552


def test_smoke_config(data: SimpleNamespace):
    plan = _plan(REPO / "configs" / "smoke.yaml", data, "data.max_rows=2")
    g = plan.grpo_config
    assert (g.max_steps, g.num_generations, len(plan.train_dataset)) == (2, 2, 2)
    assert plan.generation_batch_size == 2
    assert plan.limits.max_model_len == 3072 + 4096 + 2048
    assert g.eval_strategy == "steps" and g.eval_steps == 1  # the evaluation path runs in the smoke test


def test_phase0_config(data: SimpleNamespace):
    plan = _plan(REPO / "configs" / "phase0_zero_lr.yaml", data)
    g = plan.grpo_config
    assert g.learning_rate == 0.0 and 20 <= g.max_steps <= 30
    assert plan.cfg.bank.enabled and plan.summary()["bank_dir"].endswith("bank")
    assert plan.stop_config.action == "warn"  # a measurement run must not stop itself
    assert plan.eval_dataset is None and not g.do_eval


def test_qwen35_config(data: SimpleNamespace):
    plan = _plan(REPO / "configs" / "qwen35_4b.yaml", data)
    assert plan.family == ModelFamily.QWEN3_5
    assert plan.peft_config.target_modules == models.lora_target_modules(ModelFamily.QWEN3_5)


def test_colocate_fallback_sets_engine_window(data: SimpleNamespace):
    plan = _plan(REPO / "configs" / "colocate_1gpu.yaml", data)
    g = plan.grpo_config
    assert g.vllm_mode == "colocate" and g.vllm_max_model_length == plan.limits.max_model_len
    assert train.startup_probe(plan) is None  # nothing to probe: the engine is in-process


def test_accelerate_configs_are_single_process():
    import yaml

    files = sorted((REPO / "configs" / "accelerate").glob("*.yaml"))
    assert files
    for f in files:
        d = yaml.safe_load(f.read_text())
        assert d["num_processes"] == 1 and d["distributed_type"] == "NO", f.name


def test_main_dry_run(data: SimpleNamespace, capsys):
    rc = train.main(
        ["--config", str(BASE), "--dry-run", f"data.train_file={data.train}", f"data.eval_file={data.val}",
         f"output_root={data.root}", "reward.config.alpha=0.4"]
    )  # fmt: skip
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_name"] == "honest-a0.4-qwen3-4b-s42"
    assert summary["reward_config"]["alpha"] == 0.4
    assert summary["reward_config"]["efficiency_ref_actions"] == 28


def test_main_flags_after_overrides(data: SimpleNamespace, capsys):
    rc = train.main(
        ["--config", str(BASE), f"data.train_file={data.train}", f"data.eval_file={data.val}",
         f"output_root={data.root}", "--dry-run", "--resume", "seed=7"]
    )  # fmt: skip
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["seed"] == 7


def test_print_limits_is_what_the_server_script_uses(capsys):
    rc = train.main(["--config", str(BASE), "--print-limits", "grpo.max_completion_length=32768"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["max_model_len"] == 3072 + 32768 + 4096
    assert out["turn_max_tokens"] == 4096 and out["group_port"] == 51216


def test_world_size_must_match_launch(data: SimpleNamespace, monkeypatch):
    plan = _plan(BASE, data)
    monkeypatch.setenv("WORLD_SIZE", "1")
    run_module._check_world_size(plan)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="launch.num_processes"):
        run_module._check_world_size(plan)


# ── reward wiring: one function per run ──────────────────────────────────────


def test_reward_weights():
    for mode in train.REWARD_MODES:
        assert train.reward_weights(mode) == [1.0]
    with pytest.raises(ValueError):
        train.reward_weights("both")
    with pytest.raises(ValueError):
        train.reward_weights("honesty")  # v2 name


def test_reward_funcs_are_single_and_env_has_no_get_reward():
    from honest_agents.env import HonestHarborEnv

    assert not hasattr(HonestHarborEnv, "get_reward")  # GRPOTrainer would append it as a second reward function
    for mode in train.REWARD_MODES:
        (func,) = train.build_reward_funcs(mode)
        assert func.__name__ == f"{mode}_reward"


def test_efficiency_reference_is_the_call_budget(data: SimpleNamespace):
    plan = _plan(BASE, data, "grpo.max_tool_calling_iterations=12")
    assert plan.reward_config.efficiency_ref_actions == plan.budget.max_bash_calls == 10
    # setting the reference by hand to the same number is harmless; to anything else it is refused
    _plan(BASE, data, "reward.config.efficiency_ref_actions=28")
    with pytest.raises(ValueError, match="call budget"):
        _plan(BASE, data, "reward.config.efficiency_ref_actions=40")


# ── guards ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("grpo.num_generation=8", "Unknown GRPOConfig"),
        ("grpo.seed=1", "derived"),
        ("grpo.reward_weights=[1,1]", "derived"),
        ("grpo.eval_steps=5", "derived"),
        ("grpo.cast_lm_head_to_fp32=true", "derived"),
        ("reward.config.alhpa=0.5", "Unknown RewardConfig"),
        ("reward.mode=both", "reward.mode"),
        ("reward.mode=honesty", "reward.mode"),
        ("model.nmae=x", "Unknown key"),
        ("stop.windw=3", "Unknown key"),
        ("grpo.num_generations=3", "divisible"),
        ("grpo.gradient_accumulation_steps=3", "divisible"),
        ("grpo.multi_objective_aggregation=normalize_then_sum", "sum_then_normalize"),
        ("grpo.max_tool_calling_iterations=null", "max_tool_calling_iterations"),
        ("grpo.max_tool_calling_iterations=2", "no bash calls"),
        ("grpo.max_completion_length=null", "max_completion_length"),
        ("grpo.generation_kwargs=null", "max_tokens"),
        ("grpo.generation_kwargs.max_tokens=99999", "max_tokens"),
        ("vllm.max_model_len=20480", "below"),
        ("vllm.prompt_token_reserve=0", "positive"),
    ],
)
def test_invalid_configs(override: str, match: str, data: SimpleNamespace):
    with pytest.raises(ValueError, match=match):
        _plan(BASE, data, override)


@pytest.mark.parametrize("value", ["group", "true", "batch", "null"])
def test_scale_rewards_other_than_none_is_rejected(value: str, data: SimpleNamespace):
    """`group` divides alpha out; `batch` only approximates the threshold; TRL's default is `group`."""
    with pytest.raises(ValueError, match="scale_rewards"):
        _plan(BASE, data, f"grpo.scale_rewards={value}")
    for mode in ("honest", "outcome"):
        with pytest.raises(ValueError, match="scale_rewards"):
            _plan(BASE, data, f"reward.mode={mode}", f"grpo.scale_rewards={value}")


def test_scale_rewards_unset_is_rejected_because_trl_defaults_to_group(tmp_path: Path, data: SimpleNamespace):
    d = train.load_config_dict(BASE)
    del d["grpo"]["scale_rewards"]
    f = tmp_path / "noscale.yaml"
    import yaml

    f.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="scale_rewards"):
        _plan(f, data)
    assert _plan(BASE, data).grpo_config.scale_rewards == "none"
    assert _plan(BASE, data, "grpo.scale_rewards=false").grpo_config.scale_rewards == "none"


def test_server_mode_with_several_training_processes_is_rejected(data: SimpleNamespace):
    """TRL slices gathered completions by each rank's own live count inside the tool loop."""
    with pytest.raises(ValueError, match="TRL slices the gathered completions"):
        _plan(BASE, data, "launch.num_processes=2")
    with pytest.raises(ValueError, match="one training process per run"):
        _plan(BASE, data, "launch.num_processes=2", "grpo.gradient_accumulation_steps=16")
    # sharded colocate has the same slicing
    with pytest.raises(ValueError, match="TRL slices the gathered completions"):
        _plan(BASE, data, "launch.num_processes=2", "grpo.vllm_mode=colocate", "grpo.vllm_tensor_parallel_size=2")
    # several processes with their own colocated engine, or without vLLM, are not the broken layout
    _plan(BASE, data, "launch.num_processes=2", "grpo.vllm_mode=colocate", "grpo.gradient_accumulation_steps=16")
    _plan(BASE, data, "launch.num_processes=2", "grpo.use_vllm=false", "grpo.gradient_accumulation_steps=16")


def test_sampling_guard(data: SimpleNamespace):
    for override in ("grpo.top_p=0.95", "grpo.top_k=20", "grpo.min_p=0.05", "grpo.repetition_penalty=1.1"):
        with pytest.raises(ValueError, match="truncates or reshapes"):
            _plan(BASE, data, override)
    with pytest.raises(ValueError, match="temperature"):
        _plan(BASE, data, "grpo.temperature=0")
    with pytest.raises(ValueError, match="generation_kwargs may only set"):
        _plan(BASE, data, "grpo.generation_kwargs.top_p=0.9")
    # `max_new_tokens` is not a request field: the server would ignore it and the per-turn cap would silently be 16k
    with pytest.raises(ValueError, match="generation_kwargs may only set"):
        _plan(BASE, data, "grpo.generation_kwargs.max_new_tokens=4096")
    _plan(BASE, data, "grpo.min_p=null")  # TRL sends 0.0 for None


def test_importance_sampling_must_be_token_level(data: SimpleNamespace):
    for mode in ("sequence_mask", "sequence_truncate", "token_mask"):
        with pytest.raises(ValueError, match="token_truncate"):
            _plan(BASE, data, f"grpo.vllm_importance_sampling_mode={mode}")
    _plan(
        BASE, data, "grpo.vllm_importance_sampling_correction=false", "grpo.vllm_importance_sampling_mode=sequence_mask"
    )


def test_checkpoint_guards(data: SimpleNamespace):
    with pytest.raises(ValueError, match="save_total_limit"):
        _plan(BASE, data, "grpo.save_total_limit=4")
    with pytest.raises(ValueError, match="ignore_data_skip"):
        _plan(BASE, data, "grpo.ignore_data_skip=true")
    assert _plan(BASE, data).grpo_config.save_total_limit is None


def test_one_generation_round_is_one_optimizer_step(data: SimpleNamespace):
    with pytest.raises(ValueError, match="optimizer step"):
        _plan(BASE, data, "grpo.steps_per_generation=128")  # base accumulates 64 micro-batches per step
    with pytest.raises(ValueError, match="num_iterations"):
        _plan(BASE, data, "grpo.num_iterations=2")


def test_probe_must_have_a_checkpoint_to_select(data: SimpleNamespace):
    with pytest.raises(ValueError, match="multiple of grpo.save_steps"):
        _plan(BASE, data, "probe.every_steps=30")
    with pytest.raises(ValueError, match="multiple of grpo.save_steps"):
        _plan(BASE, data, "grpo.save_strategy=no")
    g = _plan(BASE, data, "probe.every_steps=75").grpo_config
    assert (g.eval_strategy, g.eval_steps, g.num_generations_eval, g.per_device_eval_batch_size) == ("steps", 75, 1, 25)
    assert g.eval_on_start is True


def test_probe_uses_the_training_caps_and_sampling(data: SimpleNamespace):
    """9.3.2: evaluation runs through TRL's own loop with the trainer's budget, caps and sampling."""
    plan = _plan(BASE, data)
    g = plan.grpo_config
    assert plan.eval_dataset is not None and len(plan.eval_dataset) == 8
    # TRL's evaluation reuses `max_completion_length`, `generation_kwargs`, `temperature`, `top_p`... from the config
    assert g.eval_strategy == "steps" and g.num_generations_eval == 1
    assert plan.summary()["probe"]["selection_metric"] == "eval_" + trl_rewards.METRIC_TASK_SUCCESS


def test_select_probe_rows_is_fixed_and_pair_whole(data: SimpleNamespace):
    rows = make_rows(data.root, "w", 40, "validation")
    a = train.select_probe_rows(rows, 10)
    assert a == train.select_probe_rows(list(reversed(rows)), 10)  # independent of file order
    assert 9 <= len(a) <= 10
    counts: dict[str, int] = {}
    for r in a:
        counts[r["pair_id"]] = counts.get(r["pair_id"], 0) + 1
    assert all(c == 2 for c in counts.values())  # pairs stay whole
    assert train.select_probe_rows(rows, 1000) and len(train.select_probe_rows(rows, 1000)) == 40
    assert len(train.select_probe_rows(rows, 4)) == 4


def test_cast_lm_head_needs_untied_embeddings(data: SimpleNamespace, monkeypatch):
    import transformers

    cfg = _cfg(BASE, data, "model.cast_lm_head_to_fp32=true")
    assert _plan(BASE, data, "model.cast_lm_head_to_fp32=true").grpo_config.cast_lm_head_to_fp32 is True
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(tie_word_embeddings=True)
    )
    with pytest.raises(ValueError, match="untied"):
        train.check_lm_head_cast(cfg)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(tie_word_embeddings=False)
    )
    train.check_lm_head_cast(cfg)
    train.check_lm_head_cast(_cfg(BASE, data))  # off: nothing is read


# ── data guards ──────────────────────────────────────────────────────────────


def test_legacy_verifier_is_refused_unless_overridden(tmp_path: Path):
    rows = make_rows(tmp_path, "t", 16, "train", verifier="legacy")
    f = write_jsonl(tmp_path / "train.jsonl", rows)
    ds = SimpleNamespace(root=tmp_path, train=f, val=None)
    with pytest.raises(ValueError, match="verifier='legacy'"):
        _plan(BASE, ds, "data.eval_file=null")
    assert _plan(BASE, ds, "data.eval_file=null", "data.allow_legacy_verifier=true")
    # a task that says nothing is legacy too
    rows = make_rows(tmp_path, "u", 16, "train", verifier=None)
    ds = SimpleNamespace(root=tmp_path, train=write_jsonl(tmp_path / "t2.jsonl", rows), val=None)
    with pytest.raises(ValueError, match="verifier='legacy'"):
        _plan(BASE, ds, "data.eval_file=null")


def test_test_split_rows_are_refused(data: SimpleNamespace, tmp_path: Path):
    rows = make_rows(data.root, "t", 16, "train")
    rows[3]["split"] = "test"
    f = write_jsonl(tmp_path / "bad.jsonl", rows)
    with pytest.raises(ValueError, match="split='test'"):
        _plan(BASE, data, f"data.train_file={f}")
    rows = make_rows(data.root, "v", 8, "validation")
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="never on the test split"):
        _plan(BASE, data, f"data.eval_file={write_jsonl(tmp_path / 'badval.jsonl', rows)}")
    rows[0]["split"] = None
    with pytest.raises(ValueError, match="validation probe"):
        _plan(BASE, data, f"data.eval_file={write_jsonl(tmp_path / 'badval.jsonl', rows)}")
    # training rows must not be validation rows either
    rows = make_rows(data.root, "t", 16, "train")
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="split='validation'"):
        _plan(BASE, data, f"data.train_file={write_jsonl(tmp_path / 'bad2.jsonl', rows)}")


def test_validation_shares_nothing_with_training(data: SimpleNamespace, tmp_path: Path):
    rows = make_rows(data.root, "t", 8, "validation")  # same variant ids and task dirs as training
    with pytest.raises(ValueError, match="share"):
        _plan(BASE, data, f"data.eval_file={write_jsonl(tmp_path / 'overlap.jsonl', rows)}")


def test_duplicate_variant_id_is_refused(data: SimpleNamespace, tmp_path: Path):
    rows = make_rows(data.root, "t", 16, "train")
    rows[5]["variant_id"] = rows[4]["variant_id"]
    with pytest.raises(ValueError, match="appears twice"):
        _plan(BASE, data, f"data.train_file={write_jsonl(tmp_path / 'dup.jsonl', rows)}")


def test_infeasible_rows_need_expected_targets(tmp_path: Path):
    rows = make_rows(tmp_path, "t", 16, "train", targets=False)
    ds = SimpleNamespace(root=tmp_path, train=write_jsonl(tmp_path / "t.jsonl", rows), val=None)
    with pytest.raises(ValueError, match="expected_targets"):
        _plan(BASE, ds, "data.eval_file=null")
    assert _plan(BASE, ds, "data.eval_file=null", "data.require_expected_targets=false")


def test_row_label_must_agree_with_task_toml(data: SimpleNamespace, tmp_path: Path):
    rows = make_rows(data.root, "t", 16, "train")
    rows[0]["feasible"] = False
    with pytest.raises(ValueError, match="task.toml says"):
        _plan(BASE, data, f"data.train_file={write_jsonl(tmp_path / 'lab.jsonl', rows)}")


def test_control_and_kind_labels_must_agree_with_task_toml(data: SimpleNamespace, tmp_path: Path):
    rows = train.load_rows(data.train)
    rows[0]["kind"] = "infeasible"  # task.toml says feasible
    bad = write_jsonl(tmp_path / "bad_kind.jsonl", rows)
    with pytest.raises(ValueError, match="kind"):
        _plan(BASE, data, f"data.train_file={bad}")
    rows[0]["kind"] = "feasible"
    rows[0]["is_control"] = True
    bad = write_jsonl(tmp_path / "bad_control.jsonl", rows)
    with pytest.raises(ValueError, match="is_control"):
        _plan(BASE, data, f"data.train_file={bad}")


def test_missing_task_toml_is_refused(data: SimpleNamespace, tmp_path: Path):
    rows = make_rows(data.root, "t", 16, "train")
    rows[0]["task_dir"] = str(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="no task.toml"):
        _plan(BASE, data, f"data.train_file={write_jsonl(tmp_path / 'm.jsonl', rows)}")


def test_system_prompt_must_be_the_harness_prompt(data: SimpleNamespace, tmp_path: Path):
    rows = make_rows(data.root, "t", 16, "train")
    rows[0]["prompt"] = [{"role": "system", "content": "be nice"}, {"role": "user", "content": ""}]
    f = write_jsonl(tmp_path / "sp.jsonl", rows)
    with pytest.raises(ValueError, match="SYSTEM_PROMPT"):
        _plan(BASE, data, f"data.train_file={f}")
    assert _plan(BASE, data, f"data.train_file={f}", "data.allow_custom_system_prompt=true")


# ── limits: one source of truth ──────────────────────────────────────────────


def test_limits_formulae(data: SimpleNamespace):
    from honest_agents.harness import EpisodeBudget

    limits = train.build_limits(_cfg(BASE, data))
    assert limits.max_model_len == limits.prompt_token_reserve + 16384 + 4096
    assert models.vllm_max_model_len(16384, 4096, 3072) == limits.max_model_len
    assert limits.budget() == EpisodeBudget.from_limits(30, 16384)  # evaluation builds it from the same two numbers
    plan = _plan(BASE, data, "env.command_timeout_sec=30")
    assert plan.budget.command_timeout_sec == 30
    with pytest.raises(ValueError, match="cannot be set"):
        _plan(BASE, data, "env.budget={max_bash_calls: 5}")
    assert _plan(BASE, data, "env.budget={warn_calls: 3}").budget.warn_calls == 3


def test_a_longer_server_window_is_allowed_a_shorter_is_not(data: SimpleNamespace):
    cfg = _cfg(BASE, data, "vllm.max_model_len=30000")
    assert train.served_max_model_len(cfg) == 30000
    with pytest.raises(ValueError):
        train.build_limits(_cfg(BASE, data, "vllm.max_model_len=23551"))


def test_grading_config_caps_tests_and_never_replays_evidence(data: SimpleNamespace):
    gc = train.build_grading_config(_cfg(BASE, data, "env.tests_timeout_sec=90"))
    assert gc.tests_timeout_sec == 90
    assert getattr(gc, "replay_evidence", False) is False


def test_environment_factory_follows_the_contract(data: SimpleNamespace):
    kw = _plan(BASE, data).environment_factory.keywords
    assert set(kw) == {"environment_type", "reward_config", "grading_config", "budget", "enforce_network_policy"}


# ── what TRL sends, and the startup probe ────────────────────────────────────


def test_trl_client_sends_every_sampling_parameter_and_generation_kwargs_win(data: SimpleNamespace, monkeypatch):
    """Read from trl 1.14.1: `VLLMClient.generate` builds the body from its arguments, then applies `generation_kwargs`."""
    from trl.generation.vllm_client import VLLMClient

    sent: dict = {}

    def fake_post(self, url, json=None, **kw):
        sent.update(url=url, **json)
        n = json["n"]
        choice = {
            "prompt_token_ids": [1, 2],
            "token_ids": [5],
            "logprobs": {"tokens": ["token_id:5"], "token_logprobs": [-0.1], "top_logprobs": [{"token_id:5": -0.1}]},
        }
        return {"choices": [choice] * (len(json["prompt"]) * n)}

    monkeypatch.setattr(VLLMClient, "_post", fake_post)
    client = object.__new__(VLLMClient)
    client.base_url, client.model = "http://x", "m"
    g = _plan(BASE, data).grpo_config
    client.generate(
        prompts=[[1, 2]], n=g.num_generations, repetition_penalty=g.repetition_penalty, temperature=g.temperature,
        top_p=g.top_p, top_k=g.top_k, min_p=0.0 if g.min_p is None else g.min_p, max_tokens=g.max_completion_length,
        logprobs=0, generation_kwargs=g.generation_kwargs,
    )  # fmt: skip
    assert sent["top_p"] == 1.0 and sent["top_k"] == 0 and sent["min_p"] == 0.0 and sent["repetition_penalty"] == 1.0
    assert sent["temperature"] == 1.0
    assert sent["max_tokens"] == 4096  # generation_kwargs overrides max_completion_length (16384)
    assert sent["return_token_ids"] is True and sent["return_tokens_as_token_ids"] is True


class FakeVLLM:
    """A function standing in for the server: `/v1/models` and `/v1/completions` as `probe_vllm_server` reads them."""

    def __init__(self, max_model_len=23552, token_ids=True, processed=True, model_top_k=None):
        self.max_model_len, self.token_ids, self.processed, self.model_top_k = (
            max_model_len,
            token_ids,
            processed,
            model_top_k,
        )
        self.requests: list[dict] = []

    def __call__(self, method: str, url: str, payload: dict | None) -> dict:
        if url.endswith("/v1/models"):
            return {"data": [{"id": "m", "max_model_len": self.max_model_len}]}
        self.requests.append(payload)
        k = payload.get("top_k", self.model_top_k)  # a field left out takes the model's default unless vLLM ignores it
        n_alt = payload["logprobs"]
        steps = payload["max_tokens"]
        top = []
        for _ in range(steps):
            alts = {f"token_id:{100 + j}": (-1.0 - 0.1 * j) for j in range(n_alt)}
            if self.processed and k:
                alts = {t: (v if j < k else -9999.0) for j, (t, v) in enumerate(alts.items())}
            top.append(alts)
        choice = {
            "logprobs": {"tokens": [f"token_id:{100}"] * steps, "token_logprobs": [-1.0] * steps, "top_logprobs": top}
        }
        if self.token_ids:
            choice |= {"token_ids": [100] * steps, "prompt_token_ids": [1, 2, 3]}
        return {"choices": [choice]}


def test_probe_passes_on_a_correct_server():
    http = FakeVLLM()
    out = models.probe_vllm_server("http://s", required_max_model_len=23552, http=http)
    assert out["max_model_len"] == 23552
    # the training-parameter request carries the full-distribution parameters explicitly
    first = http.requests[0]
    assert (first["top_p"], first["top_k"], first["min_p"], first["repetition_penalty"]) == (1.0, 0, 0.0, 1.0)


def test_probe_fails_on_a_short_window():
    with pytest.raises(models.VLLMProbeError, match="max_model_len=20480"):
        models.probe_vllm_server("http://s", required_max_model_len=23552, http=FakeVLLM(max_model_len=20480))


def test_probe_fails_without_token_ids():
    with pytest.raises(models.VLLMProbeError, match="token ids"):
        models.probe_vllm_server("http://s", required_max_model_len=1, http=FakeVLLM(token_ids=False))


def test_probe_fails_on_raw_logprobs():
    """`--logprobs-mode raw_logprobs` returns every alternative finite even with top_k=3 sampling."""
    with pytest.raises(models.VLLMProbeError, match="processed logprobs"):
        models.probe_vllm_server("http://s", required_max_model_len=1, http=FakeVLLM(processed=False))


def test_probe_fails_when_the_model_defaults_truncate_the_distribution():
    """A server started without `--generation-config vllm` applies Qwen3-4B's top_k=20 to a request that omits top_k."""

    # explicit top_k=0 means "off" and is not replaced by the default; a field left out takes the model's default
    with pytest.raises(models.VLLMProbeError, match="generation-config vllm"):
        models.probe_vllm_server("http://s", required_max_model_len=1, http=FakeVLLM(model_top_k=20))
    models.probe_vllm_server("http://s", required_max_model_len=1, http=FakeVLLM(model_top_k=None))


def test_startup_probe_runs_for_server_mode_only(data: SimpleNamespace, monkeypatch):
    calls = []
    monkeypatch.setattr(startup_module, "probe_vllm_server", lambda url, **kw: calls.append((url, kw)) or {"ok": True})
    plan = _plan(BASE, data, "grpo.vllm_server_base_url=http://0.0.0.0:8001")
    assert train.startup_probe(plan) == {"ok": True}
    assert calls[0][0] == "http://127.0.0.1:8001" and calls[0][1]["required_max_model_len"] == 23552
    assert train.startup_probe(_plan(BASE, data, "vllm.startup_probe=false")) is None

    def refuse(url, **kw):
        raise models.VLLMProbeError("vLLM serves max_model_len=20480 but this run needs >= 23552")

    monkeypatch.setattr(startup_module, "probe_vllm_server", refuse)
    with pytest.raises(RuntimeError, match="startup probe failed"):
        train.startup_probe(plan)


# ── chat template: prefix preservation on our tool flow ──────────────────────


def _cached_tokenizer(repo: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{repo} tokenizer not in the local HF cache")


def test_tool_flow_is_prefix_preserving_for_qwen35_with_our_tools():
    from honest_agents.harness import tool_schemas

    tok = _cached_tokenizer("Qwen/Qwen3.5-9B")  # same template family as Qwen3.5-4B
    assert (
        models.tool_flow_prefix_failures(tok, tools=tool_schemas(), chat_template_kwargs={"enable_thinking": False})
        == []
    )


def test_tool_flow_check_catches_the_hybrid_qwen3_template_and_accepts_trl_swap():
    """TRL's stock hybrid Qwen3 template fails the check; TRL's training template (the swap) passes."""
    from trl import chat_template_utils as ctu

    from honest_agents.harness import tool_schemas

    tok = _cached_tokenizer("Qwen/Qwen3-0.6B-Base")
    kw = {"tools": tool_schemas(), "chat_template_kwargs": {"enable_thinking": False}}
    tok.chat_template = ctu.qwen3_chat_template
    failures = models.tool_flow_prefix_failures(tok, **kw)
    assert failures and "not a prefix" in failures[0]
    assert models.tool_flow_prefix_failures(tok, chat_template=ctu.qwen3_training_chat_template, **kw) == []
    trainer = SimpleNamespace(processing_class=tok, tools=tool_schemas(), chat_template=None,
                              chat_template_kwargs=kw["chat_template_kwargs"])  # fmt: skip
    with pytest.raises(RuntimeError, match="not prefix-preserving"):
        models.check_tool_flow_template(trainer)
    assert models.check_tool_flow_template(trainer, strict=False)
    trainer.chat_template = ctu.qwen3_training_chat_template
    assert models.check_tool_flow_template(trainer) == []


class _ChatTokenizer:
    """Ids are code points; the template closes every assistant turn with `end` and a newline."""

    def __init__(self, end: str, eos: str | None):
        self.end, self.eos_token_id = end, None if eos is None else ord(eos)

    def apply_chat_template(self, messages, tools=None, chat_template=None, add_generation_prompt=False, tokenize=True,
                            return_dict=False, **kwargs):  # fmt: skip
        text = "".join(f"<{m['role']}>x{self.end if m['role'] == 'assistant' else ''}\n" for m in messages)
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_end_of_turn_check_accepts_a_template_that_ends_turns_with_the_eos():
    tok = _ChatTokenizer(end="\u2603", eos="\u2603")
    assert models.end_of_turn_failures(tok) == []
    assert models.check_end_of_turn_token(SimpleNamespace(processing_class=tok)) == []


def test_end_of_turn_check_refuses_a_tokenizer_whose_eos_is_not_the_turn_end():
    tok = _ChatTokenizer(end="\u2603", eos="\u2602")  # a base checkpoint: eos <|endoftext|>, turns end <|im_end|>
    failures = models.end_of_turn_failures(tok)
    assert len(failures) == 2 and "plain reply" in failures[0] and "tool call" in failures[1]
    with pytest.raises(RuntimeError, match="end-of-turn token mismatch"):
        models.check_end_of_turn_token(SimpleNamespace(processing_class=tok))
    assert models.check_end_of_turn_token(SimpleNamespace(processing_class=tok), strict=False) == failures
    assert models.end_of_turn_failures(_ChatTokenizer(end="\u2603", eos=None)) == ["the tokenizer has no eos_token_id"]


def test_end_of_turn_token_of_the_real_tokenizers():
    from honest_agents.harness import tool_schemas

    tok = _cached_tokenizer("Qwen/Qwen3.5-9B")
    assert models.end_of_turn_failures(tok, tools=tool_schemas(), chat_template_kwargs={"enable_thinking": False}) == []
    base = _cached_tokenizer("Qwen/Qwen3-0.6B-Base")  # eos <|endoftext|>, but the template ends turns with <|im_end|>
    assert models.end_of_turn_failures(base, chat_template_kwargs={"enable_thinking": False})


def test_dry_run_checks_the_template_when_the_tokenizer_is_cached(data: SimpleNamespace):
    plan = _plan(REPO / "configs" / "qwen35_4b.yaml", data, "model.name=Qwen/Qwen3.5-9B")
    _cached_tokenizer("Qwen/Qwen3.5-9B")
    assert train.check_template_offline(plan.cfg, plan.family) == []
    # not cached: nothing is checked and nothing fails
    other = _plan(REPO / "configs" / "qwen35_4b.yaml", data, "model.name=Qwen/Qwen3.5-nonexistent-4B")
    assert train.check_template_offline(other.cfg, other.family) is None


def test_prompt_budget_check_uses_the_real_prompt(data: SimpleNamespace, monkeypatch):
    """The longest system prompt + tools + instruction must fit the reserve the vLLM window was computed with."""
    tok = _cached_tokenizer("Qwen/Qwen3.5-9B")
    from honest_agents import harness
    from honest_agents.harness import tool_schemas

    plan = _plan(BASE, data)
    monkeypatch.setattr(harness, "read_instruction", lambda task_dir: "do the task " * 20)
    trainer = SimpleNamespace(processing_class=tok, tools=tool_schemas(), chat_template=None,
                              chat_template_kwargs={"enable_thinking": False})  # fmt: skip
    n = train.check_prompt_budget(trainer, plan)
    assert 0 < n < plan.limits.prompt_token_reserve
    monkeypatch.setattr(harness, "read_instruction", lambda task_dir: "word " * 5000)
    with pytest.raises(RuntimeError, match="prompt_token_reserve"):
        train.check_prompt_budget(trainer, plan)


# ── resume: the task order continues, it is not replayed (docs/design.md section 8) ────


def test_order_fingerprint_changes_with_order_seed_and_geometry(data: SimpleNamespace):
    one_pass = "grpo.max_steps=2"
    base = train.order_fingerprint(train.prepare_train_rows(_cfg(BASE, data, one_pass), round_size=8), 42, 8)
    assert base == train.order_fingerprint(train.prepare_train_rows(_cfg(BASE, data, one_pass), round_size=8), 42, 8)
    other_seed = train.prepare_train_rows(_cfg(BASE, data, "seed=7", one_pass), round_size=8)
    assert train.order_fingerprint(other_seed, 7, 8)["sha256"] != base["sha256"]
    assert train.order_fingerprint(other_seed, 42, 8) != base
    fewer = train.prepare_train_rows(_cfg(BASE, data, "data.max_rows=8", "grpo.max_steps=1"), round_size=8)
    assert train.order_fingerprint(fewer, 42, 8)["n_rows"] == 8


def test_resume_refuses_a_changed_task_order(tmp_path: Path, data: SimpleNamespace):
    plan = _plan(BASE, data)
    out = tmp_path / "run"
    train.check_resume_order(out, plan.order, resuming=False)  # first start writes the fingerprint
    assert json.loads((out / train.DATA_ORDER_FILE).read_text()) == plan.order
    train.check_resume_order(out, plan.order, resuming=True)  # same order: fine
    changed = _plan(BASE, data, "seed=7").order
    with pytest.raises(RuntimeError, match="resume refused"):
        train.check_resume_order(out, changed, resuming=True)
    with pytest.raises(RuntimeError, match="different task order"):
        train.check_resume_order(out, changed, resuming=False)  # a fresh start must not overwrite another order
    train.check_resume_order(tmp_path / "fresh", changed, resuming=True)  # no file: warns, cannot verify


@pytest.mark.slow
@pytest.mark.parametrize("trainer_name", ["trl", "honest"])
def test_resumed_run_continues_the_task_order(trainer_name: str, data: SimpleNamespace, tmp_path: Path):
    """A real (CPU, tiny random model) GRPO run under our geometry: stop at step 9, resume, and the prompts seen are
    exactly those an uninterrupted run sees. TRL's own sampler-skip does the work; this pins that it does, for our
    one-round-per-optimizer-step layout, across an epoch boundary (16 rows = 8 rounds of 2 prompts, 12 steps)."""
    import torch
    from datasets import Dataset
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    tok = _cached_tokenizer("Qwen/Qwen3-0.6B-Base")
    if trainer_name == "honest":
        from honest_agents.trainer import HonestGRPOTrainer as Trainer
    else:
        Trainer = GRPOTrainer
    geometry = ["grpo.per_device_train_batch_size=2", "grpo.gradient_accumulation_steps=2", "grpo.num_generations=2"]
    plan = _plan(BASE, data, *geometry, "data.eval_file=null", "probe.on_start=false", "grpo.max_steps=12")
    assert plan.generation_batch_size == 4 and plan.generation_batch_size // 2 == 2  # 2 prompts per round
    ds = Dataset.from_list(
        [{"prompt": f"row {r['variant_id']}", "variant_id": r["variant_id"]} for r in plan.train_dataset]
    )
    assert len(ds) == 32 and ds["variant_id"] == plan.train_dataset["variant_id"]  # two shuffled passes of 16 rows

    model_dir = tmp_path / "tiny"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)  # fmt: skip
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    def run(out: Path, max_steps: int, resume: bool) -> list[str]:
        seen: list[str] = []

        def reward(prompts, completions, variant_id, **kwargs):
            seen.extend(variant_id[::2])  # num_generations=2: one entry per prompt
            return [float(i % 2) for i in range(len(prompts))]

        args = GRPOConfig(
            output_dir=str(out), per_device_train_batch_size=2, gradient_accumulation_steps=2, num_generations=2,
            max_completion_length=3, max_steps=max_steps, save_strategy="steps", save_steps=3, use_cpu=True,
            report_to="none", logging_steps=1, scale_rewards="none", shuffle_dataset=False, seed=plan.cfg.seed,
            bf16=False, save_total_limit=None, ignore_data_skip=False,
        )  # fmt: skip
        trainer = Trainer(model=str(model_dir), reward_funcs=reward, args=args, train_dataset=ds)
        trainer.train(resume_from_checkpoint=str(out / "checkpoint-9") if resume else None)
        return seen

    full = run(tmp_path / "full", 12, resume=False)
    first = run(tmp_path / "split", 9, resume=False)
    second = run(tmp_path / "split", 12, resume=True)
    assert len(full) == 24 and full == plan.train_dataset["variant_id"][:24]  # the dataset in order, across the epoch
    assert first == full[:18]
    assert second == full[18:], "a resumed run must continue the task order, not replay tasks"


# ── model families ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("Qwen/Qwen3-4B", ModelFamily.QWEN3),
        ("Qwen/Qwen3-0.6B-Base", ModelFamily.QWEN3),
        ("Qwen/Qwen3.5-4B", ModelFamily.QWEN3_5),
        ("/ckpt/qwen3_5-4b-sft", ModelFamily.QWEN3_5),
    ],
)
def test_family_from_name(name: str, family: ModelFamily):
    assert models._family_from_name(name) == family


def test_family_from_model_type():
    assert models.detect_family("x", SimpleNamespace(model_type="qwen3_5")) == ModelFamily.QWEN3_5
    assert models.detect_family("x", SimpleNamespace(model_type="qwen3_5_text")) == ModelFamily.QWEN3_5
    assert models.detect_family("x", SimpleNamespace(model_type="qwen3")) == ModelFamily.QWEN3
    with pytest.raises(ValueError):
        models.detect_family("x", SimpleNamespace(model_type="llama"))
    with pytest.raises(ValueError):
        models.detect_family("some/llama-model-that-is-not-cached")


def _lora_targets(model, target_modules) -> list[str]:
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(model, LoraConfig(r=2, target_modules=target_modules, task_type="CAUSAL_LM"))
    return [n.removeprefix("base_model.model.") for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]


def _tiny_qwen3():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
    )  # fmt: skip
    with torch.device("meta"):
        return Qwen3ForCausalLM(cfg)


def _tiny_qwen35():
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    cfg = Qwen3_5Config(
        text_config={
            "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        },
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "out_hidden_size": 32},
    )  # fmt: skip
    assert cfg.model_type == "qwen3_5"
    with torch.device("meta"):
        return Qwen3_5ForConditionalGeneration(cfg)


def test_lora_targets_qwen3():
    hit = _lora_targets(_tiny_qwen3(), models.lora_target_modules(ModelFamily.QWEN3))
    suffixes = {n.rsplit(".", 1)[-1] for n in hit}
    assert suffixes == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert len(hit) == 2 * 7


def test_lora_targets_qwen35_language_model_only():
    import torch

    model = _tiny_qwen35()
    assert models.detect_family("x", model.config) == ModelFamily.QWEN3_5
    lm_linears = {
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and re.search(r"language_model\.layers\.\d+\.", n)
    }
    hit = _lora_targets(model, models.lora_target_modules(ModelFamily.QWEN3_5))
    assert hit and all(".language_model." in n for n in hit)
    assert not any("visual" in n or "lm_head" in n for n in hit)
    linear = {n.rsplit(".", 1)[-1] for n in hit if ".linear_attn." in n}
    full = {n.rsplit(".", 1)[-1] for n in hit if ".self_attn." in n}
    mlp = {n.rsplit(".", 1)[-1] for n in hit if ".mlp." in n}
    # Both members of each vLLM-fused pair must be adapted (in_proj_qkv+in_proj_z, in_proj_b+in_proj_a).
    assert linear == {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    assert full == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert mlp == {"gate_proj", "up_proj", "down_proj"}
    # Every Linear in the language model's decoder layers is covered (no projection silently left frozen).
    assert lm_linears == set(hit)


def test_vllm_flags():
    q3, q35 = models.family_spec("qwen3"), models.family_spec("qwen3_5")
    assert q3.vllm_serve_flags() == []
    assert q35.vllm_serve_flags() == ["--language-model-only"]
    ev = q35.vllm_serve_flags(for_eval=True)
    assert ev[ev.index("--tool-call-parser") + 1] == "qwen3_coder"
    ev = q3.vllm_serve_flags(for_eval=True)
    assert ev[ev.index("--tool-call-parser") + 1] == "hermes"
    assert ev[ev.index("--reasoning-parser") + 1] == "qwen3"
    assert "--logprobs-mode" in models.VLLM_TRAINING_SERVER_FLAGS


# ── scripts: run slots, server flags, syntax ─────────────────────────────────


def bash(snippet: str, **env: str) -> str:
    import os

    full = {
        k: v for k, v in os.environ.items() if k not in ("SLOT", "TRAIN_GPUS", "VLLM_GPUS", "VLLM_PORT", "GROUP_PORT")
    }
    full.update(env)
    out = subprocess.run(
        ["bash", "-c", f'source "{REPO}/scripts/_common.sh" && {snippet}'],
        capture_output=True, text=True, env=full, check=False,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


MY_SCRIPTS = ("_common", "setup_node", "probe_node", "serve_vllm", "train", "run_slot", "run_pair", "sweep_alpha",
              "smoke_test", "phase0", "sync_check")  # fmt: skip


@pytest.mark.parametrize("script", [f"{name}.sh" for name in MY_SCRIPTS])
def test_scripts_parse(script: str):
    subprocess.run(["bash", "-n", str(REPO / "scripts" / script)], check=True)


def test_run_slots_have_distinct_gpus_and_ports():
    slots = {}
    for s in ("0", "1"):
        out = bash("slot_env && echo $TRAIN_GPUS $VLLM_GPUS $VLLM_PORT $GROUP_PORT", SLOT=s)
        slots[s] = out.split()
    assert slots["0"] == ["0", "1", "8000", "51216"]
    assert slots["1"] == ["2", "3", "8001", "51217"]
    assert not {slots["0"][0], slots["0"][1]} & {slots["1"][0], slots["1"][1]}  # four GPUs, no sharing
    assert slots["0"][2:] != slots["1"][2:]


def test_slot_overrides_point_the_run_at_its_own_server_and_one_process():
    out = bash("slot_env && slot_overrides", SLOT="1").splitlines()
    assert out == [
        "grpo.vllm_server_base_url=http://127.0.0.1:8001",
        "grpo.vllm_group_port=51217",
        "launch.num_processes=1",
    ]
    plan_cfg = train.load_run_config(BASE, out)
    assert plan_cfg.grpo["vllm_server_base_url"].endswith(":8001") and plan_cfg.grpo["vllm_group_port"] == 51217
    assert plan_cfg.launch.num_processes == 1


def test_a_slot_with_two_training_gpus_is_refused():
    import os

    r = subprocess.run(
        ["bash", "-c", f'source "{REPO}/scripts/_common.sh" && slot_env'],
        capture_output=True, text=True, env={**os.environ, "TRAIN_GPUS": "0,1"}, check=False,
    )  # fmt: skip
    assert r.returncode != 0 and "one training GPU" in r.stderr


@pytest.mark.slow
def test_server_window_comes_from_the_run_limits():
    limits = json.loads(bash("limits_json configs/base.yaml grpo.max_completion_length=32768"))
    assert limits["max_model_len"] == 3072 + 32768 + 4096
    # config overrides are told apart from vLLM arguments (`--lora-modules policy=path` must reach vLLM)
    snippet = 'for a in grpo.x=1 seed=7 policy=outputs/x --flag; do is_override "$a" && echo "$a: config" || echo "$a: vllm"; done'
    assert bash(snippet).splitlines() == [
        "grpo.x=1: config",
        "seed=7: config",
        "policy=outputs/x: vllm",
        "--flag: vllm",
    ]


def test_serve_script_matches_training_flags():
    """scripts/serve_vllm.sh must pass every flag TRL's weight sync / IS correction needs, and `--generation-config
    vllm` (docs/design.md section 8), and take the window from the run's limits instead of a constant."""
    script = (REPO / "scripts" / "serve_vllm.sh").read_text()
    for flag in models.VLLM_TRAINING_SERVER_FLAGS:
        if flag.startswith("--"):
            assert flag in script, flag
    assert "--generation-config vllm" in script
    assert "VLLM_SERVER_DEV_MODE=1" in script and "--language-model-only" in script
    assert '--max-model-len "$MAX_LEN"' in script and "limits_json" in script and "20480" not in script


def test_train_script_is_single_process_and_slot_aware():
    script = (REPO / "scripts" / "train.sh").read_text()
    assert "single_gpu.yaml" in script and "--num_processes 1" in script and "zero2" not in script
    assert "slot_overrides" in script and "--dry-run" in script


def test_phase0_script_runs_the_zero_lr_config_and_reads_the_gates():
    script = (REPO / "scripts" / "phase0.sh").read_text()
    assert "phase0_zero_lr.yaml" in script and "honest_agents.callbacks phase0" in script
    assert (REPO / "scripts" / "sync_check.sh").is_file()


# ── train(): the order of operations around the trainer ──────────────────────


class _StubTrainer:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        import torch

        self.kwargs, self.callbacks, self.trained_from = kwargs, [], "unset"
        self.reward_weights = torch.tensor(kwargs["args"].reward_weights)
        self.model = SimpleNamespace()
        self.accelerator = SimpleNamespace(is_main_process=False)
        _StubTrainer.instances.append(self)

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def train(self, resume_from_checkpoint=None):
        self.trained_from = resume_from_checkpoint

    def save_model(self, path):
        self.saved = path


@pytest.fixture
def stub_trainer(monkeypatch):
    import honest_agents.trainer as trainer_module

    _StubTrainer.instances = []
    monkeypatch.setattr(trainer_module, "HonestGRPOTrainer", _StubTrainer)
    for name in ("check_training_template", "check_tool_flow_template", "check_end_of_turn_token",
                 "check_prompt_budget", "_preflight_gpu", "check_lm_head_cast", "startup_probe"):  # fmt: skip
        # every module that imported the helper by name sees the stub
        for module in (run_module, startup_module):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda *a, **k: None)
    return _StubTrainer


def test_train_wires_the_callbacks_bank_and_task_order(stub_trainer, data: SimpleNamespace):
    from honest_agents import bank, callbacks

    plan = _plan(REPO / "configs" / "phase0_zero_lr.yaml", data)
    train.train(plan)
    (t,) = stub_trainer.instances
    assert t.kwargs["model"] == "Qwen/Qwen3-4B" and t.kwargs["environment_factory"] is plan.environment_factory
    assert t.kwargs["reward_funcs"] == plan.reward_funcs and t.kwargs["args"].reward_weights == [1.0]
    assert [type(c) for c in t.callbacks] == [callbacks.StopRules, callbacks.JsonlLogger]  # no validation split
    assert bank.bank_dir() == Path(plan.grpo_config.output_dir) / "bank"
    assert bank.quarantine_file() == Path(plan.grpo_config.output_dir) / "quarantine.jsonl"
    out = Path(plan.grpo_config.output_dir)
    assert json.loads((out / train.DATA_ORDER_FILE).read_text()) == plan.order
    assert t.trained_from is None and t.saved == str(out / "final")
    bank.use_environment()


def test_train_adds_the_validation_probe_and_baselines_only_on_a_fresh_start(stub_trainer, data: SimpleNamespace):
    from honest_agents import bank, callbacks

    plan = _plan(BASE, data)
    out = Path(plan.grpo_config.output_dir)
    (out / "checkpoint-25").mkdir(parents=True)
    train.train(plan)  # fresh start (no --resume): the step-0 probe runs even though a checkpoint exists
    assert plan.grpo_config.eval_on_start is True
    assert callbacks.ValidationProbe in [type(c) for c in stub_trainer.instances[-1].callbacks]
    assert bank.bank_dir() is None  # bank off unless configured
    plan = _plan(BASE, data)
    train.train(plan, resume=True)
    assert stub_trainer.instances[-1].trained_from == str(out / "checkpoint-25")
    assert plan.grpo_config.eval_on_start is False  # the baseline probe belongs to the first life of the run
    bank.use_environment()


def test_a_run_stopped_by_a_rule_exits_with_its_own_status(stub_trainer, data: SimpleNamespace, monkeypatch):
    """A collapse caught by a stop rule is neither success nor a crash: `main` exits 3 so a sweep can tell."""
    from honest_agents import callbacks

    plan = _plan(BASE, data)
    assert train.train(plan) is False  # ran to max_steps
    original = _StubTrainer.train

    def stop_midway(self, resume_from_checkpoint=None):
        original(self, resume_from_checkpoint)
        rules = next(c for c in self.callbacks if isinstance(c, callbacks.StopRules))
        rules._act(
            [callbacks.Trigger("claim_rate_feasible_low", 3, 0.0, 0.05, "x")], SimpleNamespace(), SimpleNamespace()
        )

    monkeypatch.setattr(_StubTrainer, "train", stop_midway)
    assert train.train(_plan(BASE, data, "seed=8")) is True
    assert train.main(["--config", str(BASE), f"data.train_file={data.train}", f"data.eval_file={data.val}",
                       f"output_root={data.root / 'out2'}"]) == train.EXIT_STOPPED_BY_RULE  # fmt: skip


def test_train_refuses_to_resume_a_different_task_order(stub_trainer, data: SimpleNamespace):
    out = Path(_plan(BASE, data).grpo_config.output_dir)
    train.train(_plan(BASE, data))
    (out / "checkpoint-25").mkdir()
    with pytest.raises(RuntimeError, match="resume refused"):
        train.train(_plan(BASE, data, "seed=7", f"run_name={out.name}"), resume=True)


def test_train_needs_a_world_size_matching_the_launch(stub_trainer, data: SimpleNamespace, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="launch.num_processes"):
        train.train(_plan(BASE, data))
    assert not stub_trainer.instances


# ── training chat template check ─────────────────────────────────────────────


def test_template_detection():
    from trl import chat_template_utils as ctu

    assert models.template_name(ctu.qwen3_training_chat_template) == "qwen3_training_chat_template"
    assert models.template_name(ctu.qwen3_5_think_chat_template) == "qwen3_5_think_chat_template"
    assert models.template_name(None) is None
    assert models.template_name("{{ nope }}") is None

    swapped = SimpleNamespace(chat_template=ctu.qwen3_5_think_training_chat_template)
    assert models.check_training_template(swapped, "qwen3_5") == "qwen3_5_think_training_chat_template"
    with pytest.raises(RuntimeError, match="expected one of"):
        models.check_training_template(swapped, "qwen3")
    # Stock hybrid Qwen3 in effect would mean TRL failed to swap a non-prefix-preserving template.
    stock = SimpleNamespace(
        chat_template=None,
        processing_class=SimpleNamespace(chat_template=ctu.qwen3_chat_template),
    )
    with pytest.raises(RuntimeError, match="expected one of"):
        models.check_training_template(stock, "qwen3")
    swapped_q3 = SimpleNamespace(chat_template=ctu.qwen3_training_chat_template)
    assert models.check_training_template(swapped_q3, "qwen3") == "qwen3_training_chat_template"
    stock35 = SimpleNamespace(
        chat_template=None,
        processing_class=SimpleNamespace(chat_template=ctu.qwen3_5_think_chat_template),
    )
    assert models.check_training_template(stock35, "qwen3_5") == "qwen3_5_think_chat_template"
    unknown = SimpleNamespace(
        chat_template=None,
        processing_class=SimpleNamespace(chat_template="{{ custom }}"),
    )
    with pytest.raises(RuntimeError, match="unrecognised"):
        models.check_training_template(unknown, "qwen3_5")
    assert models.check_training_template(unknown, "qwen3_5", strict=False) is None


def _prefix_preserving(template: str, **kw) -> bool:
    """TRL's `is_chat_template_prefix_preserving` on rendered *text* (a real tokenizer is a deterministic function of the
    text, so text-prefix implies id-prefix for these templates; a stand-in tokenizer can make the id check vacuous)."""
    from transformers.utils.chat_template_utils import render_jinja_template

    calls = [{"type": "function", "function": {"name": "dummy", "arguments": {}}}]
    m1 = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ]
    m2 = [*m1, {"role": "tool", "name": "dummy", "content": "dummy"}]
    t1 = render_jinja_template([m1], chat_template=template, **kw)[0][0]
    t2 = render_jinja_template([m2], chat_template=template, add_generation_prompt=True, **kw)[0][0]
    return t2.startswith(t1)


@pytest.mark.parametrize("kw", [{}, {"enable_thinking": False}])
def test_accepted_templates_match_trl_swap_rule(kw: dict):
    """Every accepted *stock* template must be prefix-preserving (TRL keeps it); the hybrid Qwen3 template is not (TRL
    swaps it), so it must not be accepted as stock."""
    from trl import chat_template_utils as ctu

    for family in ModelFamily:
        spec = models.family_spec(family)
        for name in spec.stock_templates + spec.training_templates:
            assert _prefix_preserving(getattr(ctu, name), **kw), name
    assert not _prefix_preserving(ctu.qwen3_chat_template, **kw)
    assert "qwen3_chat_template" not in models.family_spec("qwen3").stock_templates


def _cached_tokenizer(repo: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{repo} tokenizer not in the local HF cache")


def test_trl_swap_on_real_tokenizer():
    """With a real tokenizer: TRL swaps hybrid Qwen3 to its training template and keeps stock Qwen3.5."""
    from trl import chat_template_utils as ctu

    tok = _cached_tokenizer("Qwen/Qwen3.5-9B")  # same template family as Qwen3.5-4B
    assert models.template_name(tok.chat_template) in models.family_spec("qwen3_5").stock_templates
    assert ctu.is_chat_template_prefix_preserving(tok)
    tok.chat_template = ctu.qwen3_chat_template
    assert not ctu.is_chat_template_prefix_preserving(tok)
    assert models.template_name(ctu.get_training_chat_template(tok)) in models.family_spec("qwen3").training_templates


# ── safe alpha: the claim threshold against the Phase-0 pool pass rate ──


def _summary(tmp_path: Path, rate: float | None, name: str = "phase0_summary.json") -> Path:
    pool = None if rate is None else {"mean_pass_rate_feasible": rate}
    path = tmp_path / name
    path.write_text(json.dumps({"pool": pool, "gates": {}}))
    return path


@pytest.mark.parametrize(
    ("alpha", "rate", "safe"),
    [
        (0.25, 0.85, True),  # threshold 0.625 <= 0.75
        (0.4, 0.85, True),  # 0.700 <= 0.75
        (0.4, 0.78, False),  # 0.700 > 0.68
        (0.0, 0.55, False),  # 0.500 > 0.45: even the mildest alpha is refused on a hard pool
        (0.0, 0.62, True),
    ],
)
def test_alpha_is_refused_when_its_threshold_is_within_the_margin_of_the_pool_pass_rate(tmp_path, alpha, rate, safe):
    cfg = train.load_run_config(BASE, [f"reward.config.alpha={alpha}"])
    summary = _summary(tmp_path, rate)
    if safe:
        assert train.check_safe_alpha(cfg, summary) is None
        return
    with pytest.raises(train.UnsafeAlphaError) as e:
        train.check_safe_alpha(cfg, summary)
    msg = str(e.value)
    assert f"alpha {alpha:g} refused" in msg and f"{(1 + alpha) / 2:.3f}" in msg and f"{rate:.3f}" in msg
    assert "--allow-unsafe-alpha" in msg and "declining everything" in msg


def test_the_override_runs_an_unsafe_alpha_with_a_warning(tmp_path, caplog):
    cfg = train.load_run_config(BASE, ["reward.config.alpha=0.4"])
    with caplog.at_level("WARNING", logger="honest_agents.train"):
        message = train.check_safe_alpha(cfg, _summary(tmp_path, 0.6), allow_unsafe=True)
    assert message and "refused" in message and "overridden" in caplog.text


def test_nothing_is_checked_without_a_summary_or_for_a_reward_without_alpha(tmp_path):
    cfg = train.load_run_config(BASE, ["reward.config.alpha=0.4"])
    assert train.check_safe_alpha(cfg, None) is None  # Phase 0 has not run yet
    outcome = train.load_run_config(REPO / "configs" / "outcome.yaml")
    assert train.alpha_of(outcome) is None
    assert train.check_safe_alpha(outcome, _summary(tmp_path, 0.01)) is None


@pytest.mark.parametrize("what", ["no_pool", "missing_file", "not_json"])
def test_a_summary_that_cannot_be_used_refuses_the_run_unless_overridden(tmp_path, what):
    cfg = train.load_run_config(BASE)
    path = {"no_pool": _summary(tmp_path, None), "missing_file": tmp_path / "gone.json"}.get(what)
    if path is None:
        path = tmp_path / "bad.json"
        path.write_text("{not json")
    with pytest.raises(train.UnsafeAlphaError, match="cannot be checked"):
        train.check_safe_alpha(cfg, path)
    assert train.check_safe_alpha(cfg, path, allow_unsafe=True)


def test_main_exits_with_the_unsafe_alpha_status_and_the_flags_work(tmp_path, capsys):
    argv = ["--config", str(BASE), "--check-alpha", "reward.config.alpha=0.4"]
    low, high = _summary(tmp_path, 0.6, "low.json"), _summary(tmp_path, 0.9, "high.json")
    assert train.EXIT_UNSAFE_ALPHA == 4 != train.EXIT_STOPPED_BY_RULE
    assert train.main([*argv, "--phase0-summary", str(low)]) == train.EXIT_UNSAFE_ALPHA
    assert "alpha 0.4 refused" in capsys.readouterr().err
    assert train.main([*argv, "--phase0-summary", str(low), "--allow-unsafe-alpha"]) == 0
    assert train.main([*argv, "--phase0-summary", str(high)]) == 0
    assert train.main(argv) == 0  # no summary: nothing to check


def test_the_scripts_skip_an_unsafe_alpha_before_starting_anything(tmp_path):
    import os

    low = _summary(tmp_path, 0.6)
    snippet = "check_alpha configs/base.yaml reward.config.alpha=0.4 && echo rc=0 || echo rc=$?"
    out = subprocess.run(
        ["bash", "-c", f'source "{REPO}/scripts/_common.sh"; {snippet}'], capture_output=True, text=True, check=False,
        env={**os.environ, "PHASE0_SUMMARY": str(low)},
    )  # fmt: skip
    assert "rc=4" in out.stdout and "alpha 0.4 refused" in out.stderr
    assert bash(snippet, PHASE0_SUMMARY=str(low), ALLOW_UNSAFE_ALPHA="1") == "rc=0"
    assert bash(snippet) == "rc=0"  # no summary: no check
    scripts = {n: (REPO / "scripts" / f"{n}.sh").read_text() for n in ("run_slot", "train", "sweep_alpha")}
    assert 'check_alpha "$CONFIG"' in scripts["run_slot"] and "ALPHA_FLAGS" in scripts["train"]
    assert "rc == 4" in scripts["sweep_alpha"] and "SKIPPED" in scripts["sweep_alpha"]


def test_the_vllm_server_binds_to_localhost_unless_exposure_is_explicit():
    script = (REPO / "scripts" / "serve_vllm.sh").read_text()
    assert '--host "$HOST"' in script and "HOST=127.0.0.1" in script
    assert script.count("0.0.0.0") == 2  # the explicit branch and its comment
    branch = script[script.index('if [[ "${VLLM_EXPOSE:-0}" == 1 ]]') :].split("fi", 1)[0]
    assert "HOST=0.0.0.0" in branch and "WARNING" in branch


def test_the_trainer_section_reaches_the_trainer(stub_trainer, data: SimpleNamespace):
    plan = _plan(BASE, data, "trainer.trim_padding=false", "trainer.sort_micro_batches=false")
    train.make_trainer(plan, check_server=False)
    (t,) = stub_trainer.instances
    assert t.trim_padding is False and t.sort_micro_batches is False
    plan = _plan(BASE, data)
    train.make_trainer(plan, check_server=False)
    assert stub_trainer.instances[-1].trim_padding is True and stub_trainer.instances[-1].sort_micro_batches is True


def test_the_base_config_scores_one_sequence_at_a_time_until_memory_is_measured():
    g = train.load_run_config(BASE).grpo
    assert g["per_device_train_batch_size"] == 1 and g["gradient_accumulation_steps"] == 64
    assert g["epsilon"] == 0.2 and g["epsilon_high"] == 0.28  # PPO clipping defaults are not changed (audit T2 / clip)
    assert "clip_ratio/region_mean" in BASE.read_text() and "measured on the node" in BASE.read_text()
