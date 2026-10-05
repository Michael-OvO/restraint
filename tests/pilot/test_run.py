"""The sweep driver: config, arms, GRPO config per arm, dataset caching, the dry run. No model, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from honest_agents.pilot import run
from honest_agents.pilot.run import (
    Arm,
    PilotConfig,
    expand_arms,
    load_pilot_config,
    select_arms,
)
from tests.pilot.helpers import synthetic_rows, write_bank
from tests.pilot.sim import simulate_arm, write_arm

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "pilot.yaml"


def test_the_shipped_config_loads_and_sweeps_alpha_by_scale():
    cfg = load_pilot_config(CONFIG)
    arms = expand_arms(cfg)
    assert len(arms) == 24 and {a.scale_rewards for a in arms} == {"none", "group"}
    assert cfg.sweep.seeds == [42, 43, 44] and {a.seed for a in arms} == {42, 43, 44}  # SP5: seed variance is measured
    assert [a.alpha for a in arms[:4]] == [0.0, 0.25, 0.5, 0.75]
    assert arms[0].name == "a0_none_s42" and arms[5].name == "a0.25_group_s42"
    assert cfg.launch.num_processes == 1 and cfg.reward.mode == "honest"


def test_the_shipped_grpo_section_follows_the_spec():
    cfg = load_pilot_config(CONFIG)
    g = cfg.grpo
    assert "scale_rewards" not in g and "max_tool_calling_iterations" not in g  # the sweep and the pilot set them
    assert g["vllm_importance_sampling_mode"] == "token_truncate" and g["save_total_limit"] is None  # design.md 8
    assert (g["top_p"], g["top_k"], g["repetition_penalty"], g["temperature"]) == (
        1.0,
        0,
        1.0,
        1.0,
    )  # §9.1.2
    assert g["generation_kwargs"] == {"max_tokens": g["max_completion_length"]}
    assert g["use_vllm"] and g["vllm_mode"] == "server"
    assert run.prompts_per_round(g) == 8  # 8 prompts x 8 reports = 64 completions per round
    assert cfg.data.build["mix"] == "default"
    assert cfg.launch.vllm_max_model_len >= 16384 + g["max_completion_length"]


def test_every_arm_builds_a_valid_grpo_config():
    from honest_agents.models import ModelFamily

    cfg = load_pilot_config(CONFIG)
    cfg.grpo["bf16"] = False  # a laptop has no bf16 device; everything else is as shipped
    cfg.grpo["gradient_checkpointing"] = False
    cfg.grpo["use_vllm"] = False
    for arm in expand_arms(cfg):
        args = run.build_grpo_config(cfg, arm, ModelFamily.QWEN3_5)
        assert args.scale_rewards == arm.scale_rewards and args.max_tool_calling_iterations == 0
        assert args.reward_weights == [1.0] and args.seed == arm.seed and args.run_name == f"pilot-{arm.name}"
        assert args.output_dir == str(Path("outputs/pilot") / arm.name / "trainer")
        assert args.num_generations == 8 and args.chat_template_kwargs is not None
        assert args.generation_batch_size == 64 and args.loss_type == "dapo" and args.beta == 0.0


def test_overrides_and_validation(tmp_path):
    cfg = load_pilot_config(
        CONFIG,
        [
            "sweep.alpha=[0.5]",
            "sweep.scale_rewards=[none]",
            "sweep.seeds=[1,2]",
            "grpo.max_steps=3",
        ],
    )
    assert [a.name for a in expand_arms(cfg)] == [
        "a0.5_none_s1",
        "a0.5_none_s2",
    ] and cfg.grpo["max_steps"] == 3
    for bad, match in [
        (["reward.config.alpha=0.3"], "sweep sets it"),
        (["sweep.scale_rewards=[batch]"], "subset"),
        (["sweep.alpha=[1.5]"], "alpha"),
        (["launch.num_processes=2"], "one training process"),
        (["reward.mode=nope"], "mode"),
        (["analysis.nope=1"], "unknown analysis key"),
        (["data.nope=1"], "unknown key"),
        (["nope=1"], "unknown top-level"),
        (["data.build.max_rows=0"], "max_rows"),
        (["reward.config.false_blame=0.9"], "false_blame"),
    ]:
        with pytest.raises(ValueError, match=match):
            load_pilot_config(CONFIG, bad)
    with pytest.raises(ValueError, match="key.path=value"):
        load_pilot_config(CONFIG, ["oops"])


def test_base_inheritance_and_cycles(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"base": "b.yaml", "grpo": {"max_steps": 1}}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"grpo": {"max_steps": 9, "num_generations": 4}, "seed": 3}))
    d = run.load_config_dict(tmp_path / "a.yaml")
    assert d == {"grpo": {"max_steps": 1, "num_generations": 4}, "seed": 3}
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"base": "a.yaml"}))
    with pytest.raises(ValueError, match="cycle"):
        run.load_config_dict(tmp_path / "a.yaml")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("grpo.top_p=0.95", "truncates"),
        ("grpo.top_k=20", "truncates"),
        ("grpo.generation_kwargs={top_p: 0.9}", "generation_kwargs"),
        ("grpo.scale_rewards=group", "derives"),
        ("grpo.max_tool_calling_iterations=5", "derives"),
        ("grpo.reward_weights=[1,0]", "derives"),
        ("grpo.no_such_field=1", "unknown GRPOConfig"),
        ("grpo.num_generations=5", "multiple of"),
        ("grpo.multi_objective_aggregation=normalize_then_sum", "sum_then_normalize"),
    ],
)
def test_grpo_section_is_validated(override, match):
    cfg = load_pilot_config(CONFIG, [override])
    with pytest.raises(ValueError, match=match):
        run.validate_grpo(cfg)


def test_select_arms_by_name_filter_and_shard():
    arms = expand_arms(load_pilot_config(CONFIG))
    assert [a.name for a in select_arms(arms, ["a0.25_none_s42"])] == ["a0.25_none_s42"]
    assert len(select_arms(arms, ["scale=group"])) == 12 and len(select_arms(arms, ["alpha=0.5"])) == 6
    assert [a.name for a in select_arms(arms, ["alpha=0.25,scale=none,seed=42"])] == ["a0.25_none_s42"]
    assert select_arms(arms, ["alpha=0.3"]) == [] and select_arms(arms, ["bogus"]) == []
    shards = [select_arms(arms, shard=f"{k}/2") for k in range(2)]
    assert sorted(a.name for s in shards for a in s) == sorted(a.name for a in arms) and not set(shards[0]) & set(
        shards[1]
    )
    assert select_arms(arms, ["scale=none"], "0/2") == [
        a for i, a in enumerate(arms) if i % 2 == 0 and a.scale_rewards == "none"
    ]
    with pytest.raises(ValueError, match="K/N"):
        select_arms(arms, shard="2/2")


def test_arm_reward_config_sets_only_alpha():
    cfg = load_pilot_config(CONFIG, ["reward.config.efficiency_bonus=0.0"])
    rc = Arm(0.5, "none", 1).reward_config(cfg.reward.config)
    assert rc.alpha == 0.5 and rc.efficiency_bonus == 0.0 and rc.claim_threshold == pytest.approx(0.75)


def test_arm_metadata_states_the_nominal_threshold():
    cfg = load_pilot_config(CONFIG)
    m_none = run.arm_metadata(cfg, Arm(0.5, "none", 42), [], completed=False)
    m_group = run.arm_metadata(cfg, Arm(0.5, "group", 42), [], completed=True)
    assert m_none["claim_threshold_nominal"] == 0.75 and m_group["claim_threshold_nominal"] == 0.5
    assert m_none["prompts_per_round"] == 8 and m_group["completed"] is True


# ── dataset and dry run ──────────────────────────────────────────────────────


def _small_config(tmp_path: Path, bank: Path, **extra) -> Path:
    d = {
        "base": str(CONFIG),
        "output_root": str(tmp_path / "out"),
        "data": {
            "bank": str(bank),
            "dataset_file": str(tmp_path / "pilot.jsonl"),
            "min_rows": 8,
            "max_epochs": 1000,
            "build": {"mix": "default", "seed": 1},
        },
        "sweep": {"alpha": [0.0, 0.5], "scale_rewards": ["none", "group"], "seeds": [42]},
        **extra,
    }
    path = tmp_path / "pilot_test.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


def test_ensure_dataset_builds_once_then_reads_the_file(tmp_path):
    bank = write_bank(tmp_path / "bank", synthetic_rows(n_feasible=16, n_control=6, n_infeasible=8))
    cfg = load_pilot_config(_small_config(tmp_path, bank))
    rows, stats = run.ensure_dataset(cfg)
    assert stats is not None and Path(cfg.data.dataset_file).exists() and len(rows) >= 8
    again, stats2 = run.ensure_dataset(cfg)
    assert stats2 is None and [r["pilot_id"] for r in again] == [r["pilot_id"] for r in rows]
    cfg.data.min_rows = 10_000
    with pytest.raises(ValueError, match="min_rows"):
        run.ensure_dataset(cfg)


def test_dry_run_plans_every_arm_and_trains_nothing(tmp_path, capsys, monkeypatch):
    bank = write_bank(tmp_path / "bank", synthetic_rows(n_feasible=16, n_control=6, n_infeasible=8))
    overrides = [
        "grpo.bf16=false",
        "grpo.gradient_checkpointing=false",
        "grpo.use_vllm=false",
    ]
    monkeypatch.setattr(run, "run_arm", lambda *a, **k: pytest.fail("a dry run must not train"))
    assert run.main(["--config", str(_small_config(tmp_path, bank)), "--dry-run", *overrides]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["arms"] == [
        "a0_none_s42",
        "a0.5_none_s42",
        "a0_group_s42",
        "a0.5_group_s42",
    ]
    assert plan["prompts_per_round"] == 8 and set(plan["kinds"]) == {
        "feasible",
        "control",
        "infeasible",
    }


def test_print_model_for_the_vllm_script(capsys):
    assert run.main(["--config", str(CONFIG), "--print-model"]) == 0
    assert capsys.readouterr().out.split() == ["Qwen/Qwen3.5-4B", "qwen3_5"]


def test_the_sweep_skips_completed_arms_and_reports_the_verdict(tmp_path, capsys, monkeypatch):
    bank = write_bank(tmp_path / "bank", synthetic_rows(n_feasible=16, n_control=6, n_infeasible=8))
    cfg_path = _small_config(
        tmp_path,
        bank,
        sweep={"alpha": [0.0, 0.25, 0.5, 0.75], "scale_rewards": ["none", "group"], "seeds": [42]},
        analysis={"window_rounds": 10},
    )
    cfg = load_pilot_config(cfg_path)
    ran = []

    def fake_run_arm(cfg, arm, rows, resume=False):
        ran.append(arm.name)
        out = write_arm(
            Path(cfg.output_root),
            arm.alpha,
            arm.scale_rewards,
            simulate_arm(arm.alpha, arm.scale_rewards, rounds=120, seed=5)[1],
        )
        (out / "arm.json").write_text(json.dumps(run.arm_metadata(cfg, arm, rows, completed=True)))
        return out

    monkeypatch.setattr(run, "run_arm", fake_run_arm)
    monkeypatch.setattr(run, "validate_grpo", lambda cfg: {})
    assert run.main(["--config", str(cfg_path)]) == 0  # GO
    assert len(ran) == 8 and "PILOT GO" in capsys.readouterr().out
    ran.clear()
    assert run.main(["--config", str(cfg_path)]) == 0
    assert ran == []  # every arm is marked completed
    assert run.main(["--config", str(cfg_path), "--force", "--arm", "alpha=0.5,scale=none"]) == 0 and ran == [
        "a0.5_none_s42"
    ]
    assert run.arm_done(cfg, Arm(0.5, "none", 42))
    assert run.main(["--config", str(cfg_path), "--analyze-only"]) == 0


def test_a_failing_arm_does_not_stop_the_sweep(tmp_path, monkeypatch, capsys):
    bank = write_bank(tmp_path / "bank", synthetic_rows(n_feasible=16, n_control=6, n_infeasible=8))
    cfg_path = _small_config(tmp_path, bank)
    seen = []

    def flaky(cfg, arm, rows, resume=False):
        seen.append(arm.name)
        if arm.alpha == 0.0:
            raise RuntimeError("boom")

    monkeypatch.setattr(run, "run_arm", flaky)
    monkeypatch.setattr(run, "validate_grpo", lambda cfg: {})
    assert run.main(["--config", str(cfg_path)]) == 3
    assert len(seen) == 4 and "FAILED arms" in capsys.readouterr().err


class FakeTokenizer:
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True):
        text = json.dumps(messages) + json.dumps(tools or [])
        return {"input_ids": list(range(len(text) // 4))}


def test_the_context_window_check_renders_the_longest_prompts():
    from honest_agents.pilot.dataset import build_rows
    from honest_agents.pilot.tools import tool_schemas

    rows, _ = build_rows(synthetic_rows(n_feasible=4, n_control=0, n_infeasible=0, per_variant=1))
    worst = run.check_context_window(rows, FakeTokenizer(), tool_schemas(), budget=100_000)
    assert worst > 100
    with pytest.raises(ValueError, match="only 50 are available"):
        run.check_context_window(rows, FakeTokenizer(), tool_schemas(), budget=50)


def test_the_pilot_config_class_is_the_documented_one():
    assert PilotConfig.from_dict({}).sweep.alpha == [0.0, 0.25, 0.5, 0.75]
    assert PilotConfig.from_dict({}).sweep.seeds == [42, 43, 44]


def test_break_even_flags_the_alphas_where_blanket_claiming_loses():
    cfg = load_pilot_config(CONFIG)
    rows = [{"feasible": True, "passed": i < 6} for i in range(10)] + [{"feasible": False, "passed": False}] * 5
    be = run.break_even(cfg, rows)
    assert be["feasible_pass_share"] == 0.6 and be["alphas_where_blanket_claiming_loses"] == [0.25, 0.5, 0.75]
    assert run.break_even(cfg, [])["alphas_where_blanket_claiming_loses"] == []


def test_plan_warns_when_the_bank_was_written_under_another_call_budget(tmp_path):
    cfg = load_pilot_config(CONFIG)
    plan = run.plan_summary(cfg, [], [], {"efficiency_ref_actions": [28, 40]})
    assert plan["efficiency_ref_mismatch"] == [40]
