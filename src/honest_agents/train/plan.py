"""Environment and reward wiring, and the `Plan` that bundles everything a run needs."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import trl_rewards
from honest_agents.callbacks import StopRuleConfig
from honest_agents.models import (
    ModelFamily,
)
from honest_agents.reward import RewardConfig
from honest_agents.train.config import REWARD_MODES, Limits, TrainRunConfig
from honest_agents.train.derive import (
    build_limits,
    build_lora_config,
    build_reward_config,
    output_dir,
    resolve_family,
    served_max_model_len,
)
from honest_agents.train.grpo import (
    build_grpo_config,
    check_batch_geometry,
    check_selection,
    probe_enabled,
    probe_steps,
)
from honest_agents.train.rows import (
    _normalise_columns,
    check_rows,
    check_split_isolation,
    epochs_needed,
    load_rows,
    order_fingerprint,
    prepare_train_rows,
    select_probe_rows,
)

# ── environment and reward wiring ───────────────────────────────────────────


def build_budget(cfg: TrainRunConfig, limits: Limits):
    """`harness.EpisodeBudget` from the limits; the config may only add the fields the limits do not determine."""
    from honest_agents.harness import EpisodeBudget

    allowed = {f.name for f in dataclasses.fields(EpisodeBudget)} - {"max_bash_calls", "max_total_chars"}
    overrides = {"command_timeout_sec": cfg.env.command_timeout_sec, **cfg.env.budget}
    bad = set(overrides) - allowed
    if bad:
        raise ValueError(
            f"env.budget: {sorted(bad)} cannot be set; allowed {sorted(allowed)} (max_bash_calls and max_total_chars "
            "follow from grpo.max_tool_calling_iterations and grpo.max_completion_length)"
        )
    return limits.budget(report_turns=cfg.env.report_turns, **overrides)


def build_grading_config(cfg: TrainRunConfig):
    """`grading.GradingConfig` with the capped test timeout; evidence replay is an evaluation-only feature."""
    from honest_agents.grading import GradingConfig

    fields = {f.name for f in dataclasses.fields(GradingConfig)}
    wanted: dict[str, Any] = {"tests_timeout_sec": cfg.env.tests_timeout_sec, "replay_evidence": False}
    if cfg.env.verifier_bundle_dir is not None:
        wanted["verifier_bundle_dir"] = Path(cfg.env.verifier_bundle_dir)
    missing = {k for k in wanted if k not in fields and k != "replay_evidence"}
    if missing:
        raise RuntimeError(f"grading.GradingConfig has no field(s) {sorted(missing)}")
    return GradingConfig(**{k: v for k, v in wanted.items() if k in fields})


def build_environment_factory(
    cfg: TrainRunConfig, limits: Limits | None = None, budget: Any = None
) -> functools.partial:
    from honest_agents.env import HonestHarborEnv

    limits = limits or build_limits(cfg)
    budget = budget or build_budget(cfg, limits)
    return functools.partial(
        HonestHarborEnv,
        environment_type=cfg.env.environment_type,
        reward_config=build_reward_config(cfg, budget.max_bash_calls),
        grading_config=build_grading_config(cfg),
        budget=budget,
        enforce_network_policy=cfg.env.enforce_network_policy,
    )


def build_reward_funcs(mode: str = "honest") -> list:
    """Exactly one reward function: with two, TRL scores a `None` from one of them as 0."""
    from honest_agents.env import HonestHarborEnv

    funcs = [trl_rewards.reward_func_for_mode(mode)]
    if hasattr(HonestHarborEnv, "get_reward"):
        # GRPOTrainer would append it as an extra reward func with weight 1, silently mixing it into every mode.
        raise RuntimeError("HonestHarborEnv defines get_reward; a run registers exactly one reward function")
    return funcs


@dataclass
class Plan:
    """Everything `GRPOTrainer` needs, minus the model."""

    cfg: TrainRunConfig
    family: ModelFamily
    grpo_config: Any
    peft_config: Any
    train_dataset: Any
    eval_dataset: Any
    environment_factory: functools.partial
    reward_funcs: list
    generation_batch_size: int
    limits: Limits
    budget: Any
    stop_config: StopRuleConfig
    order: dict[str, Any]
    data_counts: dict[str, Any] = field(default_factory=dict)

    @property
    def reward_config(self) -> RewardConfig:
        return self.environment_factory.keywords["reward_config"]

    def summary(self) -> dict[str, Any]:
        g = self.grpo_config
        return {
            "run_name": g.run_name,
            "output_dir": g.output_dir,
            "model": self.cfg.model.name,
            "family": str(self.family),
            "reward_mode": self.cfg.reward.mode,
            "reward_config": dataclasses.asdict(self.reward_config),
            "reward_funcs": [f.__name__ for f in self.reward_funcs],
            "reward_weights": list(g.reward_weights),
            "scale_rewards": g.scale_rewards,
            "lora_target_modules": (
                sorted(self.peft_config.target_modules)
                if self.peft_config is not None and not isinstance(self.peft_config.target_modules, str)
                else getattr(self.peft_config, "target_modules", None)
            ),
            "chat_template_kwargs": g.chat_template_kwargs,
            "train_rows": len(self.train_dataset),
            "eval_rows": len(self.eval_dataset) if self.eval_dataset is not None else 0,
            "data_counts": self.data_counts,
            "num_generations": g.num_generations,
            "launch_num_processes": self.cfg.launch.num_processes,
            "generation_batch_size_at_launch": self.generation_batch_size,
            "prompts_per_generation_round": self.generation_batch_size // g.num_generations,
            "limits": {
                "turn_max_tokens": self.limits.turn_max_tokens,
                "max_completion_length": self.limits.max_completion_length,
                "max_tool_calling_iterations": self.limits.max_tool_calling_iterations,
                "prompt_token_reserve": self.limits.prompt_token_reserve,
                "max_model_len": served_max_model_len(self.cfg, self.limits),
            },
            "budget": dataclasses.asdict(self.budget),
            "sampling": {k: getattr(g, k) for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")},
            "importance_sampling": {
                "correction": g.vllm_importance_sampling_correction,
                "mode": g.vllm_importance_sampling_mode,
            },
            "loss_type": g.loss_type,
            "beta": g.beta,
            "use_vllm": g.use_vllm,
            "vllm_mode": g.vllm_mode,
            "vllm_server": _server_url(self.cfg) if g.use_vllm and g.vllm_mode == "server" else None,
            "probe": (
                {
                    **dataclasses.asdict(self.cfg.probe),
                    "file": self.cfg.data.probe_file or self.cfg.data.eval_file,
                    "file_sha256": self.data_counts.get("validation", {}).get("source_sha256"),
                    "selection": check_selection(self.cfg),  # how best_checkpoint.json is chosen (DEF-08)
                    **probe_composition(list(self.eval_dataset)),
                    **probe_cost(self.cfg, len(self.eval_dataset), self.generation_batch_size),
                }
                if self.eval_dataset is not None
                else None
            ),
            "stop_rules": dataclasses.asdict(self.stop_config),
            "bank_dir": str(bank_dir(self.cfg)) if self.cfg.bank.enabled else None,
            "quarantine_file": str(quarantine_file(self.cfg)),
            "cast_lm_head_to_fp32": g.cast_lm_head_to_fp32,
            "learning_rate": g.learning_rate,
            "max_steps": g.max_steps,
            "task_order": self.order,
            "max_concurrent_boots": self.cfg.env.max_concurrent_boots,
            "seed": g.seed,
        }


def bank_dir(cfg: TrainRunConfig) -> Path:
    return Path(cfg.bank.dir) if cfg.bank.dir else output_dir(cfg) / "bank"


def quarantine_file(cfg: TrainRunConfig) -> Path:
    return Path(cfg.bank.quarantine_file) if cfg.bank.quarantine_file else output_dir(cfg) / "quarantine.jsonl"


def _server_url(cfg: TrainRunConfig) -> str:
    g = cfg.grpo
    return (
        g.get("vllm_server_base_url")
        or f"http://{g.get('vllm_server_host', '0.0.0.0')}:{g.get('vllm_server_port', 8000)}"
    )


def build_stop_config(cfg: TrainRunConfig) -> StopRuleConfig:
    allowed = {f.name for f in dataclasses.fields(StopRuleConfig)}
    unknown = set(cfg.stop) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in `stop:`; expected {sorted(allowed)}")
    return StopRuleConfig(**cfg.stop)


def probe_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Impossible, feasible (plain and control) and control variants of the probe rows."""
    infeasible = sum(1 for r in rows if r.get("feasible") is False)
    control = sum(1 for r in rows if r.get("feasible") is not False and r.get("is_control"))
    return {"infeasible": infeasible, "feasible": len(rows) - infeasible, "control": control}


def load_probe_rows(cfg: TrainRunConfig, train_rows: list[dict[str, Any]], counts: dict[str, Any]) -> list[dict]:
    """The probe's rows: `data.probe_file` whole, else the fixed hash subset of `data.eval_file`.

    Both go through the validation guards (validation split only, labels agree with task.toml, no test id) and share
    nothing with training. A probe file must reach `probe.min_infeasible` / `probe.min_feasible`: it exists to make
    the probe large enough to resolve honesty changes, and a short one would look like it did. Fills `counts`.
    """
    path = cfg.data.probe_file or cfg.data.eval_file
    rows = load_rows(path, allow_custom_system_prompt=cfg.data.allow_custom_system_prompt)
    counts["validation"] = check_rows(rows, cfg, role="validation")
    check_split_isolation(train_rows, rows)
    if cfg.data.probe_file is None:
        rows = select_probe_rows(rows, cfg.probe.max_variants)
    comp = probe_composition(rows)
    if cfg.data.probe_file is not None:
        p = cfg.probe
        if comp["infeasible"] < p.min_infeasible or comp["feasible"] < p.min_feasible:
            raise ValueError(
                f"data.probe_file {path}: {comp['infeasible']} impossible and {comp['feasible']} feasible variants, "
                f"below probe.min_infeasible={p.min_infeasible} / probe.min_feasible={p.min_feasible}"
            )
    counts["validation"].update(
        source=str(path),
        # The probe list is built once and paired across probes and runs by task: its digest in the plan summary
        # (printed at every start, and by `--dry-run`) shows two runs probed the same file.
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        probe_rows=len(rows),
        probe_infeasible=comp["infeasible"],
        probe_feasible=comp["feasible"],
        probe_control=comp["control"],
    )
    return rows


def probe_cost(cfg: TrainRunConfig, n_rows: int, generation_batch_size: int) -> dict[str, Any]:
    """Episodes one probe costs and how many probes the run makes, next to the training episodes.

    Probes run at steps `every_steps, 2 x every_steps, ...` (TRL's `eval_steps`), at the last step if that is not
    one of them (transformers' DefaultFlowCallback), and at step 0 with `on_start`: run 2 (50 steps, every 20) probed
    at 0, 20, 40 and 50. With `probe.steps` only the listed steps up to `max_steps` (and step 0 with `on_start`) are
    probed (`callbacks.ProbeSchedule`). The episode count is the cost that matters: a probe episode takes about as long
    as a training one.
    """
    p = cfg.probe
    per_probe = n_rows * p.samples_per_variant
    max_steps = cfg.grpo.get("max_steps")
    listed = probe_steps(cfg)
    probes = None
    if max_steps and max_steps > 0 and listed is not None:
        probes = sum(s <= int(max_steps) for s in listed) + int(p.on_start)
    elif max_steps and max_steps > 0:
        probes = int(max_steps) // p.every_steps + int(int(max_steps) % p.every_steps != 0) + int(p.on_start)
    training = int(max_steps) * generation_batch_size if max_steps and max_steps > 0 else None
    return {
        "episodes_per_probe": per_probe,
        "probes_per_run": probes,
        "probe_episodes_per_run": per_probe * probes if probes is not None else None,
        "training_episodes_per_run": training,
    }


def build_plan(cfg: TrainRunConfig) -> Plan:
    from datasets import Dataset

    if cfg.reward.mode not in REWARD_MODES:
        raise ValueError(f"reward.mode must be one of {REWARD_MODES}, got {cfg.reward.mode!r}")
    family = resolve_family(cfg)
    limits = build_limits(cfg)
    grpo_cfg = build_grpo_config(cfg, family, limits)
    gen_bs = check_batch_geometry(grpo_cfg, cfg.launch.num_processes, cfg.grpo)
    round_size = gen_bs // grpo_cfg.num_generations
    budget = build_budget(cfg, limits)
    stop_config = build_stop_config(cfg)

    base_rows = load_rows(
        cfg.data.train_file, cfg.data.max_rows, allow_custom_system_prompt=cfg.data.allow_custom_system_prompt
    )
    counts = {"train": check_rows(base_rows, cfg, role="train")}
    train_rows = prepare_train_rows(cfg, round_size=round_size, rows=base_rows)
    counts["train"]["epochs"] = epochs_needed(cfg, len(base_rows), round_size) if cfg.data.pair_batches else None
    eval_ds = None
    if probe_enabled(cfg):
        eval_rows = load_probe_rows(cfg, train_rows, counts)
        eval_ds = Dataset.from_list(_normalise_columns(eval_rows))
    return Plan(
        cfg=cfg,
        family=family,
        grpo_config=grpo_cfg,
        peft_config=build_lora_config(cfg, family),
        train_dataset=Dataset.from_list(_normalise_columns(train_rows)),
        eval_dataset=eval_ds,
        environment_factory=build_environment_factory(cfg, limits, budget),
        reward_funcs=build_reward_funcs(cfg.reward.mode),
        generation_batch_size=gen_bs,
        limits=limits,
        budget=budget,
        stop_config=stop_config,
        order=order_fingerprint(train_rows, cfg.seed, round_size),
        data_counts=counts,
    )
