"""Objects derived from the config: token limits, reward config, alpha safety, run names, LoRA config."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from honest_agents.callbacks import alpha_is_safe
from honest_agents.models import (
    ModelFamily,
    detect_family,
    lora_target_modules,
)
from honest_agents.reward import RewardConfig
from honest_agents.train.config import ALPHA_PASS_MARGIN, REWARD_MODES, Limits, TrainRunConfig

logger = logging.getLogger("honest_agents.train")


def build_limits(cfg: TrainRunConfig) -> Limits:
    grpo = cfg.grpo
    kwargs = grpo.get("generation_kwargs")
    if not isinstance(kwargs, dict) or kwargs.get("max_tokens") is None:
        raise ValueError(
            "grpo.generation_kwargs.max_tokens (the per-turn generation cap) must be set: without it TRL asks for "
            "max_completion_length tokens on every turn, which vLLM rejects once the transcript is longer than "
            "max_model_len - max_completion_length, and one runaway turn eats the whole budget"
        )
    turn = int(kwargs["max_tokens"])
    total = grpo.get("max_completion_length")
    iterations = grpo.get("max_tool_calling_iterations")
    if total is None:
        raise ValueError("grpo.max_completion_length (the total budget) must be set")
    if iterations is None:
        raise ValueError("grpo.max_tool_calling_iterations must be set (unbounded agent loops never terminate)")
    total, iterations = int(total), int(iterations)
    if not 0 < turn <= total:
        raise ValueError(f"grpo.generation_kwargs.max_tokens={turn} must be in (0, max_completion_length={total}]")
    if iterations < 3:
        raise ValueError(
            f"grpo.max_tool_calling_iterations={iterations} leaves no bash calls (budget is iterations - 2)"
        )
    reserve = int(cfg.vllm.prompt_token_reserve)
    if reserve < 1:
        raise ValueError("vllm.prompt_token_reserve must be positive")
    limits = Limits(turn, total, iterations, reserve)
    if cfg.vllm.max_model_len is not None and int(cfg.vllm.max_model_len) < limits.max_model_len:
        raise ValueError(
            f"vllm.max_model_len={cfg.vllm.max_model_len} is below the {limits.max_model_len} this run needs "
            f"(prompt reserve {reserve} + budget {total} + turn cap {turn}); leave it unset to use the computed value"
        )
    return limits


def served_max_model_len(cfg: TrainRunConfig, limits: Limits | None = None) -> int:
    """The `--max-model-len` the server for this run is started with (`scripts/serve_vllm.sh` calls `--print-limits`)."""
    limits = limits or build_limits(cfg)
    return int(cfg.vllm.max_model_len) if cfg.vllm.max_model_len is not None else limits.max_model_len


# ── derived objects ─────────────────────────────────────────────────────────


def resolve_family(cfg: TrainRunConfig) -> ModelFamily:
    if cfg.model.family != "auto":
        return ModelFamily(cfg.model.family)
    return detect_family(cfg.model.name)


def build_reward_config(cfg: TrainRunConfig, call_budget: int | None = None) -> RewardConfig:
    """`RewardConfig` from the config. `efficiency_ref_actions` is the call budget: the efficiency bonus
    then scales against the same number of calls the environment allows. Setting it by hand is refused, because a
    different reference would make training and evaluation disagree about what a cheap solution is."""
    allowed = {f.name for f in dataclasses.fields(RewardConfig)}
    unknown = set(cfg.reward.config) - allowed
    if unknown:
        raise ValueError(f"Unknown RewardConfig field(s) {sorted(unknown)}; expected {sorted(allowed)}")
    values = dict(cfg.reward.config)
    if call_budget is None:
        call_budget = int(cfg.grpo.get("max_tool_calling_iterations", 0)) - 2
    if "efficiency_ref_actions" in values and int(values["efficiency_ref_actions"]) != call_budget:
        raise ValueError(
            f"reward.config.efficiency_ref_actions={values['efficiency_ref_actions']} must equal the call budget "
            f"({call_budget} = grpo.max_tool_calling_iterations - 2); it is set from the budget, do not set it"
        )
    values["efficiency_ref_actions"] = call_budget
    return RewardConfig(**values)


class UnsafeAlphaError(ValueError):
    """The honesty reward's claim threshold is too close to (or above) what the task pool lets the policy pass."""


def alpha_of(cfg: TrainRunConfig) -> float | None:
    """The run's alpha, or None for a reward without one (the outcome arms)."""
    if cfg.reward.mode != "honest":
        return None
    return float(build_reward_config(cfg).alpha)


def load_pool_pass_rate(summary_path: str | Path) -> float:
    """The mean pass rate on feasible tasks from `scripts/phase0.sh`'s `phase0_summary.json` (`pool` section)."""
    path = Path(summary_path)
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise UnsafeAlphaError(f"cannot read the Phase-0 summary {path}: {e}") from e
    rate = (summary.get("pool") or {}).get("mean_pass_rate_feasible")
    if not isinstance(rate, int | float) or isinstance(rate, bool):
        raise UnsafeAlphaError(
            f"{path} has no pool.mean_pass_rate_feasible (it needs the trajectory bank: run Phase 0 with bank.enabled), "
            "so the alpha cannot be checked against the pool"
        )
    return float(rate)


def check_safe_alpha(
    cfg: TrainRunConfig,
    summary_path: str | Path | None,
    *,
    allow_unsafe: bool = False,
    margin: float = ALPHA_PASS_MARGIN,
) -> str | None:
    """Refuse an alpha whose claim threshold (1 + alpha) / 2 exceeds the Phase-0 pool pass rate minus `margin`.

    Above that line quitting beats working at the pool's pass rate, and the reward drifts to refuse-everything:
    `callbacks.pool_stats` only reported it (`safe_alphas`), this makes it a hard skip. Needs a summary;
    without one nothing is checked (Phase 0 has not run: the first runs are the ones that produce it). Outcome arms have
    no alpha. `allow_unsafe` turns the refusal into a warning. Returns the message when it was overridden.
    """
    alpha = alpha_of(cfg)
    if summary_path is None or alpha is None:
        return None
    try:
        rate = load_pool_pass_rate(summary_path)
        threshold = (1 + alpha) / 2
        if alpha_is_safe(alpha, rate, margin):
            logger.info("alpha %g is safe: threshold %.3f <= pool pass rate %.3f - %g", alpha, threshold, rate, margin)
            return None
        message = (
            f"alpha {alpha:g} refused: its claim threshold (1 + alpha) / 2 = {threshold:.3f} exceeds the Phase-0 pool "
            f"pass rate {rate:.3f} minus the margin {margin:g} = {rate - margin:.3f} ({summary_path}). A policy would "
            "do better by declining everything. Pick a smaller alpha, easier tasks, or pass --allow-unsafe-alpha "
            "(ALLOW_UNSAFE_ALPHA=1 in the scripts) to run it anyway."
        )
    except UnsafeAlphaError as e:
        message = f"alpha {alpha:g} cannot be checked: {e}. Pass --allow-unsafe-alpha to run without the check."
    if not allow_unsafe:
        raise UnsafeAlphaError(message)
    logger.warning("%s (overridden by --allow-unsafe-alpha)", message)
    return message


def reward_weights(mode: str) -> list[float]:
    """One reward function per run, weight 1: a None from it is a mask, not a zero."""
    if mode not in REWARD_MODES:
        raise ValueError(f"reward.mode must be one of {REWARD_MODES}, got {mode!r}")
    return [1.0]


def default_run_name(cfg: TrainRunConfig) -> str:
    model_short = cfg.model.name.rstrip("/").split("/")[-1].lower()
    if cfg.reward.mode == "honest":
        tag = f"honest-a{float(cfg.reward.config.get('alpha', RewardConfig().alpha)):g}"
    else:
        tag = cfg.reward.mode.replace("_", "-")
    return f"{tag}-{model_short}-s{cfg.seed}"


def run_name(cfg: TrainRunConfig) -> str:
    return cfg.run_name or default_run_name(cfg)


def output_dir(cfg: TrainRunConfig) -> Path:
    return Path(cfg.output_root) / run_name(cfg)


def build_lora_config(cfg: TrainRunConfig, family: ModelFamily):
    if not cfg.lora.enabled:
        return None
    from peft import LoraConfig

    targets = cfg.lora.target_modules
    if targets == "auto":
        targets = lora_target_modules(family)
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
