"""GRPO training entrypoint: `python -m honest_agents.train --config configs/x.yaml [key.path=value ...] [--dry-run]`.

A run is fully described by one YAML file (optionally inheriting from another via `base:`) plus dotted CLI overrides,
so sweeps are just override lists (see `scripts/sweep_alpha.sh`). `--dry-run` builds and validates everything except
the model and the trainer (dataset, `GRPOConfig`, `LoraConfig`, reward wiring, environment factory, limits, guards), so
config validity is unit-testable on a laptop without GPUs, Docker or network.

What this module guarantees (SPEC_V3 §5, §9; the finding ids are in `docs/research/go-no-go-review.json`):

* **One reward function per run** (`reward.mode`: honest, outcome, outcome_claimed, outcome_reflect), registered with
  `reward_weights = [1.0]`, so a `None` is a real mask (V-1, TDC-8).
* **`scale_rewards: none` everywhere.** `group` is refused: it divides alpha out of two-valued groups (INC-1, TDC-3).
* **One training process per run** when vLLM is a separate server (or sharded): TRL slices the gathered completions by
  each rank's own count of live episodes, which mis-routes them inside the tool loop (TDC-1, M1, F9).
* **One source of truth for limits.** `grpo.generation_kwargs.max_tokens` (per turn), `grpo.max_completion_length`
  (total) and `grpo.max_tool_calling_iterations` give `harness.EpisodeBudget.from_limits(...)` for the environment, the
  efficiency reference of the reward and the vLLM `--max-model-len`, which is asserted against the running server at
  startup (TDC-2, TDC-4, M2, SPEC §9.3.2). Evaluation builds the same budget from the same numbers.
* **Sampling from the full distribution** (SPEC §9.1.2): `top_p 1`, `top_k 0`, `min_p 0`, no repetition penalty,
  checked in the config and probed on the server.
* **Resume continues the task order** (SPEC §9.3.3): the dataset order is a pure function of the config and the seed,
  its fingerprint is stored and re-checked on resume, and one generation round is exactly one optimizer step so a
  checkpoint never falls inside a generation batch.
* **Unsafe alphas are refused** when a Phase-0 summary is supplied (`--phase0-summary`): (1 + alpha) / 2 above the pool
  pass rate minus 0.1 means declining everything beats working (`--allow-unsafe-alpha` overrides; exit status 4).
* **Trainer safety checks at start**: the chat template ends every assistant turn with the tokenizer's EOS (else every
  turn counts as truncated), and the vLLM server binds to 127.0.0.1 unless `VLLM_EXPOSE=1` (`scripts/serve_vllm.sh`).
* **Data guards**: only `verifier = "pytest"` tasks (override: `data.allow_legacy_verifier`), no test-split rows in
  training, a validation probe split that shares no variant with training, and labels in the rows that agree with the
  task's own `task.toml`.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from honest_agents import trl_rewards
from honest_agents.callbacks import SAFE_ALPHA_MARGIN, StopRuleConfig, alpha_is_safe
from honest_agents.models import (
    ModelFamily,
    check_end_of_turn_token,
    check_full_distribution_sampling,
    check_tool_flow_template,
    check_training_template,
    detect_family,
    family_spec,
    flash_linear_attention_available,
    lora_target_modules,
    probe_vllm_server,
    vllm_max_model_len,
)
from honest_agents.reward import RewardConfig

logger = logging.getLogger("honest_agents.train")

REWARD_MODES = trl_rewards.MODES
SPLIT_TRAIN = "train"
VALIDATION_SPLITS = ("validation", "val")
DATA_ORDER_FILE = "data_order.json"
EXIT_STOPPED_BY_RULE = 3  # distinct from a crash, so a sweep can tell a collapse from a failure
EXIT_UNSAFE_ALPHA = 4  # the alpha was refused against the Phase-0 pool pass rate: a sweep skips it instead of failing
ALPHA_PASS_MARGIN = (
    SAFE_ALPHA_MARGIN  # (1 + alpha) / 2 must stay this far below the pool pass rate (audit2 memo, reward item 3)
)

# GRPOConfig fields derived from other sections; setting them under `grpo:` would silently disagree with the run.
_DERIVED_GRPO_FIELDS = {
    "output_dir",
    "run_name",
    "seed",
    "reward_weights",
    "model_init_kwargs",
    # from `model.cast_lm_head_to_fp32`
    "cast_lm_head_to_fp32",
    # from the `probe:` section
    "do_eval",
    "eval_strategy",
    "eval_steps",
    "eval_on_start",
    "per_device_eval_batch_size",
    "num_generations_eval",
}


# ── config schema ─────────────────────────────────────────────────────────────


@dataclass
class ModelSection:
    name: str = "Qwen/Qwen3-4B"
    family: str = "auto"  # auto | qwen3 | qwen3_5
    revision: str | None = None
    dtype: str = "bfloat16"  # TRL loads in float32 unless told otherwise
    attn_implementation: str | None = None
    strict_template: bool = True  # fail unless the rollout template is one TRL knows for the family (models.py)
    require_fla: bool = True  # Qwen3.5: fail on GPU if flash-linear-attention is missing (~20x slower otherwise)
    # Compute the LM head in float32 (ScaleRL recipe; Tmax found it "especially crucial" on Qwen3.5, L6). TRL only
    # supports it for untied embeddings; `train()` checks the model's config before loading weights.
    cast_lm_head_to_fp32: bool = False


@dataclass
class LoraSection:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str | list[str] = "auto"  # auto -> honest_agents.models.lora_target_modules(family)


@dataclass
class DataSection:
    train_file: str = "data/train.jsonl"
    eval_file: str | None = None  # the validation split the periodic probe runs on; never the test split
    max_rows: int | None = None
    # Keep feasible/infeasible variants of a pair_id in the same generation round (TRL's RepeatSampler cuts the dataset
    # into consecutive chunks of generation_batch_size / num_generations prompts) so each round contains both sides of
    # its pairs (abstention collapse is the main GRPO risk). Pairs are shuffled as units with the run seed, packed so
    # none straddles a round boundary, and TRL's own sampler shuffle is disabled.
    pair_batches: bool = True
    # Guards (SPEC_V3 §3.2, §4). Each has an explicit override so a deliberate exception is visible in the config.
    allow_legacy_verifier: bool = False  # tasks must say `[metadata.honest] verifier = "pytest"`
    check_tasks: bool = True  # read every row's task.toml (verifier, labels); off only for synthetic unit-test rows
    require_expected_targets: bool = True  # an infeasible row without expected_targets lets a blind guess earn m = 1
    allow_custom_system_prompt: bool = False  # rows must use harness.SYSTEM_PROMPT (one prompt in train and eval)
    # test_ids.txt of the dataset build (L2): a train or validation row whose pair_id is in it is refused. None = the file
    # next to `train_file` if there is one (tasks.build writes it there), else no check.
    holdout_ids: str | None = None


@dataclass
class EnvSection:
    environment_type: str = "docker"
    max_concurrent_boots: int = 16  # per training process
    command_timeout_sec: int = 60  # `timeout -k 5 <this>` around every command (SPEC §3.7); part of the budget
    tests_timeout_sec: int = 180  # hermetic test run, capped (MV-1)
    verifier_bundle_dir: str | None = None  # vendored pytest bundle; None = harness default
    enforce_network_policy: bool = True
    # Extra `harness.EpisodeBudget` fields (max_output_chars, warn_calls, ...). The two big numbers come from `grpo:`.
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardSection:
    mode: str = "honest"  # honest | outcome | outcome_claimed | outcome_reflect
    # Any `honest_agents.reward.RewardConfig` field (alpha, beta, implicit_abstain, ...). `efficiency_ref_actions` is
    # set from the call budget and may not be set here.
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaunchSection:
    """How the run is launched. Only used to validate batch divisibility on machines with a different world size."""

    num_processes: int = 1


@dataclass
class VLLMSection:
    """What the generation server must be started with so that it can serve this run's loop (TDC-2)."""

    # Tokens reserved for the system prompt, the tool schemas and the instruction. Checked against the real prompts
    # with the real tokenizer at startup.
    prompt_token_reserve: int = 3072
    max_model_len: int | None = None  # None = computed (prompt reserve + total budget + per-turn cap)
    startup_probe: bool = True  # fail at startup if the server cannot serve this run (window, token ids, logprobs)


@dataclass
class ProbeSection:
    """Periodic held-out probe (SPEC_V3 §9.3.1): TRL's own evaluation loop over a fixed validation subset."""

    every_steps: int = 50
    max_variants: int = 100  # fixed subset of `data.eval_file`, chosen by a hash of the pair id (pairs stay whole)
    batch_size: int = 25  # episodes in flight per evaluation batch
    on_start: bool = True  # evaluate the untrained policy first, so a later fall has a baseline
    selection_metric: str = f"eval_{trl_rewards.METRIC_TASK_SUCCESS}"  # checkpoints are chosen on validation only


@dataclass
class BankSection:
    """Trajectory bank (Phase 0 feeds the report-only pilot) and the quarantine list of leaky variants."""

    enabled: bool = False
    dir: str | None = None  # default: <output_dir>/bank
    quarantine_file: str | None = None  # default: <output_dir>/quarantine.jsonl


@dataclass
class TrainerSection:
    """Attributes of `honest_agents.trainer.HonestGRPOTrainer` set after construction."""

    max_tool_workers: int | None = None  # None = one thread per live episode
    trim_padding: bool = True  # length-trimmed forward chunks (audit T1); False = TRL's batch-global padding
    sort_micro_batches: bool = True  # regroup a round's micro-batches by length (dapo/cispo/vespo only)


@dataclass
class TrainRunConfig:
    model: ModelSection = field(default_factory=ModelSection)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataSection = field(default_factory=DataSection)
    env: EnvSection = field(default_factory=EnvSection)
    reward: RewardSection = field(default_factory=RewardSection)
    launch: LaunchSection = field(default_factory=LaunchSection)
    vllm: VLLMSection = field(default_factory=VLLMSection)
    probe: ProbeSection = field(default_factory=ProbeSection)
    stop: dict[str, Any] = field(default_factory=dict)  # `callbacks.StopRuleConfig` fields
    bank: BankSection = field(default_factory=BankSection)
    trainer: TrainerSection = field(default_factory=TrainerSection)
    grpo: dict[str, Any] = field(default_factory=dict)  # passed to trl.GRPOConfig (names validated)
    seed: int = 42
    output_root: str = "outputs"
    run_name: str | None = None  # default: derived from mode / alpha / model / seed
    config_path: str | None = None  # where this config came from (for provenance)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainRunConfig:
        d = dict(d)
        d.pop("base", None)
        sections = {
            "model": ModelSection,
            "lora": LoraSection,
            "data": DataSection,
            "env": EnvSection,
            "reward": RewardSection,
            "launch": LaunchSection,
            "vllm": VLLMSection,
            "probe": ProbeSection,
            "bank": BankSection,
            "trainer": TrainerSection,
        }
        kwargs: dict[str, Any] = {}
        top = {f.name for f in dataclasses.fields(cls)}
        for k, v in d.items():
            if k not in top:
                raise ValueError(f"Unknown top-level config key {k!r}; expected one of {sorted(top)}")
            if k in sections:
                kwargs[k] = _build_section(sections[k], v or {}, k)
            else:
                kwargs[k] = v
        return cls(**kwargs)


def _build_section(cls: type, values: dict[str, Any], name: str) -> Any:
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in `{name}:`; expected {sorted(allowed)}")
    return cls(**values)


# ── loading / overrides ─────────────────────────────────────────────────────


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config_dict(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML, resolving `base: other.yaml` (relative to this file) recursively; this file's keys win."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, (*_seen, path)))}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    base = data.get("base")
    if base:
        data = _deep_merge(load_config_dict(path.parent / base, (*_seen, path)), data)
    data.pop("base", None)
    return data


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides; values are parsed as YAML (`reward.config.alpha=0.5`, `grpo.report_to=[wandb]`)."""
    d = copy.deepcopy(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override {ov!r} must look like key.path=value")
        key, raw = ov.split("=", 1)
        parts = key.strip().split(".")
        node = d
        for p in parts[:-1]:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise TypeError(f"Override {ov!r}: {p!r} is not a mapping")
            node = nxt
        node[parts[-1]] = yaml.safe_load(raw)
    return d


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> TrainRunConfig:
    d = apply_overrides(load_config_dict(path), overrides or [])
    cfg = TrainRunConfig.from_dict(d)
    cfg.config_path = str(path)
    return cfg


# ── limits: one source of truth (SPEC_V3 §5.3, §9.3.2) ──────────────────────


@dataclass(frozen=True)
class Limits:
    """The three numbers that bound an episode, and what follows from them.

    They live in `grpo:` (`generation_kwargs.max_tokens`, `max_completion_length`, `max_tool_calling_iterations`)
    because TRL reads them there. Everything else derives from this object: the environment's `EpisodeBudget`, the
    reward's efficiency reference, the vLLM window.
    """

    turn_max_tokens: int  # per-turn generation cap, sent to the server as `max_tokens` on every request
    max_completion_length: int  # total budget over all turns, tool output included
    max_tool_calling_iterations: int  # cap on tool-call turns
    prompt_token_reserve: int  # system prompt + tool schemas + instruction

    @property
    def max_model_len(self) -> int:
        """vLLM rejects a request whose input exceeds `max_model_len - max_tokens`, so the window must cover the
        longest transcript (prompt + budget) plus one more turn (TDC-2)."""
        return vllm_max_model_len(self.max_completion_length, self.turn_max_tokens, self.prompt_token_reserve)

    def budget(self, **overrides: Any):
        """`harness.EpisodeBudget` for the environment: training, evaluation and the pilot build theirs the same way."""
        from honest_agents.harness import EpisodeBudget

        return EpisodeBudget.from_limits(
            max_tool_calling_iterations=self.max_tool_calling_iterations,
            max_completion_length=self.max_completion_length,
            **overrides,
        )


def build_limits(cfg: TrainRunConfig) -> Limits:
    grpo = cfg.grpo
    kwargs = grpo.get("generation_kwargs")
    if not isinstance(kwargs, dict) or kwargs.get("max_tokens") is None:
        raise ValueError(
            "grpo.generation_kwargs.max_tokens (the per-turn generation cap) must be set: without it TRL asks for "
            "max_completion_length tokens on every turn, which vLLM rejects once the transcript is longer than "
            "max_model_len - max_completion_length (TDC-2), and one runaway turn eats the whole budget"
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
    """`RewardConfig` from the config. `efficiency_ref_actions` is the call budget (SPEC §9.2.1): the efficiency bonus
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

    Above that line quitting beats working at the pool's pass rate, and the reward drifts to refuse-everything (audit2
    memo, INC-5): `callbacks.pool_stats` only reported it (`safe_alphas`), this makes it a hard skip. Needs a summary;
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
    """One reward function per run, weight 1: a None from it is a mask, not a zero (V-1)."""
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


# ── guards on the GRPO section ──────────────────────────────────────────────


def check_scale_rewards(grpo: dict[str, Any]) -> None:
    """`scale_rewards` must be `none`: group scaling divides alpha out of two-valued groups (realised claim thresholds
    0.40/0.48/0.54/0.57 for nominal 0.50/0.625/0.75/0.875, INC-1, TDC-3) and `batch` only approximates the threshold."""
    value = grpo.get("scale_rewards", "group")  # TRL's default is "group"
    if value is False or value == "none":
        return
    raise ValueError(
        f"grpo.scale_rewards={value!r} is refused: reward scaling by the group's standard deviation erases alpha, so the "
        "claim threshold (1 + alpha) / 2 is no longer what the run optimises (INC-1, TDC-3). Use scale_rewards: none."
    )


def check_launch_layout(grpo: dict[str, Any], num_processes: int) -> None:
    """Refuse the layouts that mis-route completions inside TRL's tool loop (TDC-1, M1)."""
    if num_processes < 1:
        raise ValueError(f"launch.num_processes must be >= 1, got {num_processes}")
    if num_processes == 1 or not grpo.get("use_vllm", False):
        return
    mode = grpo.get("vllm_mode", "colocate")
    if mode == "server" or int(grpo.get("vllm_tensor_parallel_size", 1)) > 1:
        raise ValueError(
            f"launch.num_processes={num_processes} with vLLM {mode} mode: TRL slices the gathered completions by each "
            "rank's own count of still-active episodes, which differs between ranks from the second tool round on, so "
            "ranks train on completions sampled for other episodes (TDC-1, M1). Use one training process per run; the "
            "other GPU of the pair serves vLLM, and the second pair of GPUs runs a second run."
        )


def check_importance_sampling(grpo: dict[str, Any]) -> None:
    """Token-level truncated IS (TDC-6): the sequence-level default drops long trajectories systematically."""
    if not grpo.get("vllm_importance_sampling_correction", True):
        return
    mode = grpo.get("vllm_importance_sampling_mode", "sequence_mask")  # TRL 1.14.1 default
    if mode != "token_truncate":
        raise ValueError(
            f"grpo.vllm_importance_sampling_mode={mode!r}: use 'token_truncate' (SPEC_V3 §5.4). Sequence-level ratios "
            "shrink with length and zero out long episodes (TDC-6); or turn the correction off."
        )


def check_checkpointing(grpo: dict[str, Any]) -> None:
    if grpo.get("save_total_limit") is not None:
        raise ValueError(
            "grpo.save_total_limit must be null: every checkpoint is kept, because the validation probe picks one "
            "afterwards and the best step is often not the last (SPEC_V3 §9.3.1, TDC-9)"
        )
    if grpo.get("ignore_data_skip"):
        raise ValueError("grpo.ignore_data_skip=true would replay the first tasks on resume (SPEC_V3 §9.3.3)")


def check_batch_geometry(grpo_cfg, num_processes: int, raw: dict[str, Any] | None = None) -> int:
    """Re-run GRPOConfig's divisibility checks for the *launch* world size (the dry-run machine has world_size 1).

    `raw` is the user's `grpo:` section: GRPOConfig has already filled `steps_per_generation` / `generation_batch_size`
    for world size 1, so which of the two the user actually set must come from there (trl GRPOConfig.__post_init__).
    One generation round must be exactly one optimizer step (`steps_per_generation == gradient_accumulation_steps`,
    `num_iterations == 1`): a checkpoint is written at an optimizer step, so with longer rounds a resume would fall
    inside a generation batch and replay or skip prompts (SPEC_V3 §9.3.3).
    Returns the generation batch size (completions per generation round) at the launch world size.
    """
    raw = raw or {}
    per_dev = grpo_cfg.per_device_train_batch_size
    global_bs = per_dev * num_processes
    if raw.get("generation_batch_size") is not None:
        gen_bs = int(raw["generation_batch_size"])
        if gen_bs % global_bs:
            raise ValueError(
                f"generation_batch_size {gen_bs} not divisible by the global batch size {per_dev}*{num_processes}"
            )
    else:
        spg = raw.get("steps_per_generation") or grpo_cfg.gradient_accumulation_steps
        gen_bs = global_bs * spg
    if gen_bs % grpo_cfg.num_generations:
        raise ValueError(
            f"generation batch {gen_bs} (at {num_processes} processes) not divisible by "
            f"num_generations={grpo_cfg.num_generations}"
        )
    if gen_bs != global_bs * grpo_cfg.gradient_accumulation_steps:
        raise ValueError(
            f"one generation round ({gen_bs} completions) must be one optimizer step "
            f"({global_bs} x gradient_accumulation_steps={grpo_cfg.gradient_accumulation_steps}); otherwise a "
            "checkpoint falls inside a generation batch and a resumed run replays or skips tasks"
        )
    if getattr(grpo_cfg, "num_iterations", 1) != 1:
        raise ValueError("grpo.num_iterations must be 1: reusing a generation batch breaks resume order (SPEC §9.3.3)")
    if grpo_cfg.do_eval and grpo_cfg.eval_strategy != "no":
        n_eval = grpo_cfg.num_generations_eval or grpo_cfg.num_generations
        if (grpo_cfg.per_device_eval_batch_size * num_processes) % n_eval:
            raise ValueError(
                f"global eval batch {grpo_cfg.per_device_eval_batch_size}*{num_processes} not divisible by {n_eval}"
            )
    return gen_bs


def probe_enabled(cfg: TrainRunConfig) -> bool:
    return cfg.data.eval_file is not None


def build_grpo_config(cfg: TrainRunConfig, family: ModelFamily, limits: Limits | None = None):
    from trl import GRPOConfig

    grpo = dict(cfg.grpo)
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = set(grpo) - known
    if unknown:
        raise ValueError(f"Unknown GRPOConfig field(s) under `grpo:`: {sorted(unknown)}")
    clash = set(grpo) & _DERIVED_GRPO_FIELDS
    if clash:
        raise ValueError(f"`grpo:` must not set derived field(s) {sorted(clash)}; use the top-level/section keys")
    if cfg.reward.mode not in REWARD_MODES:
        raise ValueError(f"reward.mode must be one of {REWARD_MODES}, got {cfg.reward.mode!r}")

    agg = grpo.get("multi_objective_aggregation", "sum_then_normalize")
    if agg != "sum_then_normalize":
        raise ValueError("reward modes assume multi_objective_aggregation='sum_then_normalize'")
    check_scale_rewards(grpo)
    grpo["scale_rewards"] = "none"
    check_full_distribution_sampling(grpo)
    check_importance_sampling(grpo)
    check_checkpointing(grpo)
    check_launch_layout(grpo, cfg.launch.num_processes)
    limits = limits or build_limits(cfg)

    spec = family_spec(family)
    grpo.setdefault("chat_template_kwargs", dict(spec.chat_template_kwargs))
    if cfg.data.pair_batches:
        grpo["shuffle_dataset"] = False  # order is set by `pair_ordered` below
    if grpo.get("use_vllm") and grpo.get("vllm_mode", "colocate") == "colocate":
        grpo["vllm_max_model_length"] = served_max_model_len(cfg, limits)  # the engine's window, same formula

    model_init_kwargs: dict[str, Any] = {"dtype": cfg.model.dtype}
    if cfg.model.revision:
        model_init_kwargs["revision"] = cfg.model.revision
    if cfg.model.attn_implementation:
        model_init_kwargs["attn_implementation"] = cfg.model.attn_implementation

    derived: dict[str, Any] = {"cast_lm_head_to_fp32": cfg.model.cast_lm_head_to_fp32}
    if probe_enabled(cfg):
        p = cfg.probe
        if p.every_steps < 1 or p.batch_size < 1 or p.max_variants < 1:
            raise ValueError("probe.every_steps, probe.batch_size and probe.max_variants must be >= 1")
        save_steps = grpo.get("save_steps", 500)
        if grpo.get("save_strategy", "steps") != "steps" or p.every_steps % int(save_steps):
            raise ValueError(
                f"probe.every_steps={p.every_steps} must be a multiple of grpo.save_steps={save_steps} "
                "(save_strategy: steps), so every probed step has a checkpoint to select"
            )
        derived.update(
            do_eval=True,
            eval_strategy="steps",
            eval_steps=p.every_steps,
            eval_on_start=p.on_start,
            per_device_eval_batch_size=p.batch_size,
            num_generations_eval=1,  # one rollout per variant: 100 variants, not 100 groups of 8
        )

    return GRPOConfig(
        output_dir=str(output_dir(cfg)),
        run_name=run_name(cfg),
        seed=cfg.seed,
        reward_weights=reward_weights(cfg.reward.mode),
        model_init_kwargs=model_init_kwargs,
        **derived,
        **grpo,
    )


# ── rows ─────────────────────────────────────────────────────────────────────


def pair_ordered(rows: list[dict[str, Any]], seed: int, round_size: int | None = None) -> list[dict[str, Any]]:
    """Shuffle rows as pair_id groups (rows without a pair_id are singleton groups); members stay adjacent.

    With `round_size` (prompts per generation round), groups are first-fit packed into rounds of exactly that many rows
    so no group straddles a round boundary. Mere adjacency is not enough: a single unpaired row shifts every later
    pair across a boundary. Rows that do not fill a whole round go last (TRL drops the final partial chunk).
    """
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for i, r in enumerate(rows):
        groups.setdefault(r.get("pair_id") or f"__row{i}", []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    # Feasible first within a pair: deterministic, and irrelevant to the loss.
    for k in keys:
        groups[k].sort(key=lambda r: not r.get("feasible", True))
    if not round_size:
        return [r for k in keys for r in groups[k]]

    rounds: list[list[dict[str, Any]]] = []
    for k in keys:
        g = groups[k]
        if len(g) > round_size:
            logger.warning("pair_id %r has %d rows > %d prompts per round; splitting it", k, len(g), round_size)
        for j in range(0, len(g), round_size):
            part = g[j : j + round_size]
            slot = next((b for b in rounds if len(b) + len(part) <= round_size), None)
            if slot is None:
                rounds.append(slot := [])
            slot.extend(part)
    full = [b for b in rounds if len(b) == round_size]
    partial = [r for b in rounds if len(b) < round_size for r in b]
    if partial:
        logger.info("%d row(s) do not fill a whole generation round and go last", len(partial))
    return [r for b in full for r in b] + partial


def load_rows(
    path: str | Path, max_rows: int | None = None, *, allow_custom_system_prompt: bool = False
) -> list[dict[str, Any]]:
    """Read the dataset builder's JSONL.

    Required per row: `task_dir` (absolute, or relative to the jsonl's directory). Optional: `prompt` (list of chat
    messages ending in a user turn; TRL appends `env.reset()`'s instruction to it); when absent we use
    `[system: SYSTEM_PROMPT, user: ""]`. A row's system message must be `harness.SYSTEM_PROMPT`: training and
    evaluation share one prompt (INTERFACES_V3 §1). Every other column is passed through to `HonestHarborEnv.reset`.
    """
    from honest_agents.harness import SYSTEM_PROMPT

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "task_dir" not in r:
                raise ValueError(f"{path}:{ln}: row has no `task_dir`")
            td = Path(r["task_dir"])
            r["task_dir"] = str(td if td.is_absolute() else (path.parent / td).resolve())
            prompt = r.get("prompt") or [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ""},
            ]
            if not isinstance(prompt, list) or not prompt or prompt[-1].get("role") != "user":
                raise ValueError(f"{path}:{ln}: `prompt` must be a list of messages ending with a user message")
            system = [m.get("content") for m in prompt if m.get("role") == "system"]
            if not allow_custom_system_prompt and any(s != SYSTEM_PROMPT for s in system):
                raise ValueError(
                    f"{path}:{ln}: the system message differs from harness.SYSTEM_PROMPT; training rows use the one "
                    "prompt evaluation uses (data.allow_custom_system_prompt to override)"
                )
            r["prompt"] = prompt
            rows.append(r)
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _constant_prompt() -> list[dict[str, str]]:
    from honest_agents.harness import SYSTEM_PROMPT

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]


def holdout_ids(cfg: TrainRunConfig) -> set[str]:
    """The held-out test ids (`data.holdout_ids`, or `test_ids.txt` beside the train file); empty if there is none."""
    from honest_agents.tasks.build import read_test_ids

    explicit = cfg.data.holdout_ids
    path = Path(explicit) if explicit else Path(cfg.data.train_file).parent / "test_ids.txt"
    if not path.is_file():
        if explicit:
            raise ValueError(f"data.holdout_ids {explicit!r} does not exist")
        return set()
    return read_test_ids(path)  # raises on a missing or wrong sha256 line


def _source_group(pair_id: str) -> str:
    """The split group of a pair id (SETA_Evolve variants go with their parent, as in `SeedTask.split_group`)."""
    return re.sub(r"__[bd]\d+$", "", pair_id) if pair_id.startswith("seta-env/") else pair_id


def check_rows(rows: list[dict[str, Any]], cfg: TrainRunConfig, *, role: str) -> dict[str, Any]:
    """Guards on one dataset file (`role` is `train` or `validation`); returns counts for the summary.

    * no row's pair_id is in the held-out `test_ids.txt` (`holdout_ids`);
    * the split column: training rows are always `train` (a missing split column is refused), probe rows always `validation`;
    * `variant_id` is unique (it is the prompt-group id the reward masks leaks by);
    * every row's task says `verifier = "pytest"` (SPEC §3.2) and its labels agree with the row's columns;
    * every infeasible row records `expected_blockers` and `expected_targets` (GR-1).
    """
    from honest_agents.spec import VERIFIER_PYTEST, HonestTaskSpec

    where = f"{role} rows"
    held_out = holdout_ids(cfg)
    seen: set[str] = set()
    counts: dict[str, int] = {"rows": len(rows)}
    specs: dict[str, HonestTaskSpec] = {}
    for i, r in enumerate(rows):
        split = r.get("split")
        if role == "train" and split != SPLIT_TRAIN:  # a missing column is refused too: an unlabelled row may be test
            raise ValueError(f"{where}: row {i} has split={split!r}; only split='train' rows may be trained on")
        if role == "validation" and split not in VALIDATION_SPLITS:
            raise ValueError(
                f"{where}: row {i} has split={split!r}; the validation probe runs on split in {VALIDATION_SPLITS} "
                "only, never on the test split"
            )
        if not cfg.data.allow_custom_system_prompt and "prompt" in r and r["prompt"] != _constant_prompt():
            raise ValueError(  # TRL appends reset()'s instruction to it: anything else here is extra, row-specific text
                f"{where}: row {i} `prompt` is not the constant [system prompt, empty user turn]; the only "
                "row-specific text the policy may see is instruction.md"
            )
        pair = r.get("pair_id")
        if pair is not None and (pair in held_out or _source_group(pair) in held_out):
            raise ValueError(
                f"{where}: row {i} pair_id {pair!r} is a held-out test id (test_ids.txt); refusing to use it"
            )
        vid = r.get("variant_id")
        if vid is not None:
            if vid in seen:
                raise ValueError(f"{where}: variant_id {vid!r} appears twice; it identifies the prompt group")
            seen.add(vid)
        if not cfg.data.check_tasks:
            continue
        td = r["task_dir"]
        if td not in specs:
            if not (Path(td) / "task.toml").is_file():
                raise ValueError(f"{where}: row {i} task_dir {td!r} has no task.toml")
            specs[td] = HonestTaskSpec.from_task_dir(td)
        spec = specs[td]
        if spec.verifier != VERIFIER_PYTEST and not cfg.data.allow_legacy_verifier:
            raise ValueError(
                f'{where}: {td} has verifier={spec.verifier!r}; training needs [metadata.honest] verifier = "pytest" '
                "(hermetic grading, SPEC_V3 §3.2). data.allow_legacy_verifier=true overrides."
            )
        if "feasible" in r and bool(r["feasible"]) != spec.feasible:
            raise ValueError(f"{where}: row {i} says feasible={r['feasible']} but {td}/task.toml says {spec.feasible}")
        for column, truth in (("is_control", spec.is_control), ("kind", spec.kind)):
            if r.get(column) is not None and r[column] != truth:  # a mislabelled control skews the mixture and metrics
                raise ValueError(f"{where}: row {i} says {column}={r[column]!r} but {td}/task.toml says {truth!r}")
        if vid is not None and spec.variant_id is not None and spec.variant_id != vid:
            raise ValueError(f"{where}: row {i} variant_id {vid!r} != task.toml variant_id {spec.variant_id!r}")
        if (
            not spec.feasible
            and cfg.data.require_expected_targets
            and not (spec.expected_blockers and spec.expected_targets)
        ):
            raise ValueError(
                f"{where}: infeasible task {td} needs both expected_blockers and expected_targets in task.toml; "
                "without a target a blind category guess earns the full decline reward (GR-1)"
            )
        counts[spec.kind] = counts.get(spec.kind, 0) + 1
    return counts


def check_split_isolation(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    """The probe split shares no variant, pair or task with training; sharing a source is only a warning."""
    for key in ("variant_id", "pair_id", "task_dir"):
        a = {r[key] for r in train_rows if r.get(key) is not None}
        b = {r[key] for r in eval_rows if r.get(key) is not None}
        if a & b:
            raise ValueError(f"train and validation rows share {len(a & b)} {key}(s), e.g. {min(a & b)!r}")
    a = {r["source"] for r in train_rows if r.get("source")}
    b = {r["source"] for r in eval_rows if r.get("source")}
    if a & b:
        logger.warning(
            "%d validation source task(s) also appear in training: the probe is not source-held-out", len(a & b)
        )


def select_probe_rows(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """A fixed validation subset: pair groups ordered by a hash of their id (not by position or run seed order), taken
    whole until `n` rows are reached, so the same ~100 variants are probed at every step of every run."""
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for i, r in enumerate(rows):
        groups.setdefault(str(r.get("pair_id") or r.get("variant_id") or f"__row{i}"), []).append(r)
    ordered = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}|{k}".encode()).hexdigest())
    for members in groups.values():
        members.sort(key=lambda r: str(r.get("variant_id") or r.get("task_dir")))
    out: list[dict[str, Any]] = []
    for k in ordered:
        if len(out) >= n:
            break
        if len(out) + len(groups[k]) <= n or not out:
            out.extend(groups[k])
    return out[:n] if len(out) > n else out


def _normalise_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arrow needs one type per column: fill missing keys with None and JSON-encode nested non-message values."""
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = []
    for r in rows:
        nr = {}
        for k in keys:
            v = r.get(k)
            if k != "prompt" and isinstance(v, dict):
                v = json.dumps(v)
            nr[k] = v
        out.append(nr)
    return out


def order_fingerprint(rows: list[dict[str, Any]], seed: int, round_size: int | None) -> dict[str, Any]:
    """Identity of the task order: what a resumed run must reproduce for TRL to continue where the run stopped."""
    ids = [str(r.get("variant_id") or r.get("task_dir")) for r in rows]
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return {"n_rows": len(rows), "seed": seed, "round_size": round_size, "sha256": digest}


def check_resume_order(out: Path, fingerprint: dict[str, Any], *, resuming: bool) -> None:
    """Write the fingerprint on a fresh start; on resume refuse if the order differs (SPEC_V3 §9.3.3)."""
    path = out / DATA_ORDER_FILE
    if resuming and path.is_file():
        stored = json.loads(path.read_text())
        if stored != fingerprint:
            raise RuntimeError(
                f"resume refused: the task order differs from the original run ({stored} vs {fingerprint}). The data "
                "file, data.max_rows, data.pair_batches, the batch geometry or the seed changed, so TRL would skip "
                "the wrong prompts."
            )
        return
    if resuming:
        logger.warning("resuming but %s is missing; cannot verify the task order", path)
    elif path.is_file() and json.loads(path.read_text()) != fingerprint:
        raise RuntimeError(f"{path} belongs to a different task order; use a new run_name or output_root")
    if int(os.environ.get("RANK", "0")) == 0:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fingerprint, indent=2) + "\n")


def build_dataset(cfg: TrainRunConfig, path: str | None = None, round_size: int | None = None):
    from datasets import Dataset

    rows = prepare_train_rows(cfg, path, round_size)
    return Dataset.from_list(_normalise_columns(rows))


def epochs_needed(cfg: TrainRunConfig, n_rows: int, round_size: int) -> int:
    """How many passes over the rows `max_steps` generation rounds take (1 when `max_steps` is unset)."""
    max_steps = cfg.grpo.get("max_steps")
    if not max_steps or max_steps < 1 or n_rows < 1:
        return 1
    return max(1, -(-int(max_steps) * round_size // n_rows))  # ceil


def prepare_train_rows(
    cfg: TrainRunConfig,
    path: str | None = None,
    round_size: int | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The training rows in the order TRL will draw them (its own shuffle is off with `pair_batches`).

    TRL replays a shuffle-less dataset in the same order every epoch, so every epoch would put the same pairs in the
    same rounds (TDC-10c). When `max_steps` takes more than one pass, the dataset is therefore the concatenation of one
    independently pair-shuffled pass per epoch, each cut to whole rounds so no round straddles two passes. The order
    stays a pure function of the config and the seed, which is what resume needs (SPEC_V3 §9.3.3).
    """
    rows = rows if rows is not None else load_rows(
        path or cfg.data.train_file, cfg.data.max_rows, allow_custom_system_prompt=cfg.data.allow_custom_system_prompt
    )  # fmt: skip
    if round_size and len(rows) < round_size:
        # RepeatSampler drops incomplete chunks: fewer rows than one round means zero training batches.
        raise ValueError(
            f"{len(rows)} training row(s) < {round_size} prompts per generation round; training would see no batches"
        )
    if not cfg.data.pair_batches:
        return list(rows)
    epochs = epochs_needed(cfg, len(rows), round_size) if round_size else 1
    out: list[dict[str, Any]] = []
    for epoch in range(epochs):
        ordered = pair_ordered(rows, cfg.seed + epoch * 1_000_003, round_size)
        if epoch < epochs - 1 and round_size:
            ordered = ordered[: len(ordered) // round_size * round_size]  # a partial tail would shift later rounds
        out.extend(ordered)
    return out


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
    return limits.budget(**overrides)


def build_grading_config(cfg: TrainRunConfig):
    """`grading.GradingConfig` with the capped test timeout; evidence replay is an evaluation-only feature (MV-1)."""
    from honest_agents.grading import GradingConfig

    fields = {f.name for f in dataclasses.fields(GradingConfig)}
    wanted: dict[str, Any] = {"tests_timeout_sec": cfg.env.tests_timeout_sec, "replay_evidence": False}
    if cfg.env.verifier_bundle_dir is not None:
        wanted["verifier_bundle_dir"] = Path(cfg.env.verifier_bundle_dir)
    missing = {k for k in wanted if k not in fields and k != "replay_evidence"}
    if missing:
        raise RuntimeError(f"grading.GradingConfig has no field(s) {sorted(missing)} (INTERFACES_V3 §4)")
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
    """Exactly one reward function: with two, TRL scores a `None` from one of them as 0 (V-1, TDC-8)."""
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
            "probe": dataclasses.asdict(self.cfg.probe) if self.eval_dataset is not None else None,
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
        eval_rows = load_rows(cfg.data.eval_file, allow_custom_system_prompt=cfg.data.allow_custom_system_prompt)
        counts["validation"] = check_rows(eval_rows, cfg, role="validation")
        check_split_isolation(train_rows, eval_rows)
        eval_rows = select_probe_rows(eval_rows, cfg.probe.max_variants)
        counts["validation"]["probe_rows"] = len(eval_rows)
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


# ── startup checks that need the model, the tokenizer or the server ─────────


def check_template_offline(
    cfg: TrainRunConfig, family: ModelFamily, tools: list[Any] | None = None
) -> list[str] | None:
    """Prefix-preservation of our tool flow with the real tokenizer, if it is cached locally (SPEC §9.1.4).

    Mirrors how GRPOTrainer picks its template: the tokenizer's own if it is prefix-preserving, else TRL's training
    template. Returns the failures (empty = fine), or None when the tokenizer is not cached (nothing was checked).
    """
    from transformers import AutoTokenizer
    from trl.chat_template_utils import (
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
    )

    from honest_agents.models import tool_flow_prefix_failures

    try:
        tok = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, or a path that is not a model
        logger.info(
            "tokenizer for %s is not cached locally; template check skipped (it runs at training start)", cfg.model.name
        )
        return None
    if tools is None:
        try:
            from honest_agents.harness import tool_schemas
        except ImportError:
            logger.warning("harness.tool_schemas is not available; template check skipped")
            return None
        tools = tool_schemas()
    tok = add_response_schema(tok)
    template = None if is_chat_template_prefix_preserving(tok) else get_training_chat_template(tok)
    kwargs = dict(family_spec(family).chat_template_kwargs) | dict(cfg.grpo.get("chat_template_kwargs") or {})
    return tool_flow_prefix_failures(tok, tools=tools, chat_template=template, chat_template_kwargs=kwargs)


def check_prompt_budget(trainer: Any, plan: Plan) -> int:
    """Longest system prompt + tool schemas + instruction, in tokens, against `vllm.prompt_token_reserve`.

    The server window is prompt reserve + budget + turn cap; a prompt longer than the reserve would shrink the room for
    the transcript and could make vLLM reject a request mid-run (TDC-2).
    """
    from honest_agents.harness import SYSTEM_PROMPT, read_instruction

    longest = 0
    datasets = [plan.train_dataset] + ([plan.eval_dataset] if plan.eval_dataset is not None else [])
    for task_dir in sorted({r["task_dir"] for ds in datasets for r in ds}):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": read_instruction(task_dir)},
        ]
        ids = trainer.processing_class.apply_chat_template(
            messages,
            tools=trainer.tools or None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=getattr(trainer, "chat_template", None),
            **trainer.chat_template_kwargs,
        )
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        longest = max(longest, len(ids))
    reserve = plan.limits.prompt_token_reserve
    if longest > reserve:
        raise RuntimeError(
            f"the longest prompt is {longest} tokens but vllm.prompt_token_reserve={reserve}; raise the reserve (the "
            "vLLM window is computed from it) and restart the server"
        )
    logger.info("longest prompt: %d tokens (reserve %d)", longest, reserve)
    return longest


def check_lm_head_cast(cfg: TrainRunConfig) -> None:
    """TRL's `cast_lm_head_to_fp32` is documented for untied embeddings only."""
    if not cfg.model.cast_lm_head_to_fp32:
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg.model.name, revision=cfg.model.revision)
    text = getattr(config, "text_config", None) or config
    if getattr(text, "tie_word_embeddings", False):
        raise ValueError(
            f"model.cast_lm_head_to_fp32 needs untied embeddings but {cfg.model.name} ties them "
            "(trl GRPOConfig.cast_lm_head_to_fp32); turn it off for this model"
        )


def startup_probe(plan: Plan, retries: int = 3, wait_sec: float = 10.0) -> dict[str, Any] | None:
    """Assert the running vLLM server can serve this run: window, token ids, processed logprobs, full distribution."""
    import time

    from honest_agents.models import VLLMProbeError

    g = plan.grpo_config
    if not (g.use_vllm and g.vllm_mode == "server" and plan.cfg.vllm.startup_probe):
        return None
    url, last = _server_url(plan.cfg).replace("0.0.0.0", "127.0.0.1"), None
    for attempt in range(retries):
        try:
            return probe_vllm_server(
                url,
                required_max_model_len=served_max_model_len(plan.cfg, plan.limits),
                temperature=float(g.temperature),
            )
        except VLLMProbeError as e:
            last = e
            if "failed:" not in str(e) or attempt == retries - 1:  # only connection errors are worth retrying
                break
            time.sleep(wait_sec)
    raise RuntimeError(f"vLLM startup probe failed for {url}: {last}") from last


# ── run ─────────────────────────────────────────────────────────────────────


def _preflight_gpu(plan: Plan) -> None:
    spec = family_spec(plan.family)
    if spec.needs_flash_linear_attention and not flash_linear_attention_available():
        msg = "flash-linear-attention (`fla`) is not installed; Qwen3.5 Gated DeltaNet falls back to slow torch code."
        if plan.cfg.model.require_fla:
            raise RuntimeError(msg + " Install it or set model.require_fla=false.")
        logger.warning(msg)


def _check_world_size(plan: Plan) -> None:
    """Batch geometry and pair packing were computed for `launch.num_processes`; a different real world size would
    silently change the prompts per round (and split pairs across rounds)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != plan.cfg.launch.num_processes:
        raise RuntimeError(
            f"WORLD_SIZE={world} but launch.num_processes={plan.cfg.launch.num_processes}; "
            "pass launch.num_processes=<n> (scripts/train.sh does this)"
        )


def _last_checkpoint(out: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(out)) if out.is_dir() else None


def add_callbacks(trainer: Any, plan: Plan) -> None:
    """Stop rules, the validation probe's checkpoint selection and the run log."""
    from honest_agents.callbacks import JsonlLogger, StopRules, ValidationProbe

    out = plan.grpo_config.output_dir
    trainer.add_callback(StopRules(plan.stop_config, reward_mode=plan.cfg.reward.mode, output_dir=out))
    trainer.add_callback(JsonlLogger(out))
    if plan.eval_dataset is not None:
        trainer.add_callback(ValidationProbe(out, plan.cfg.probe.selection_metric))


# Per-step wall-clock columns the run log carries for the Phase-0 trainer decision (SPEC_V3 §10.3). The trainer logs
# `rollout/*` (generation, tools); these three are the remaining components. TRL's `step_time` is their sum.
METRIC_GRADING_SECONDS = "timing/grading_seconds"  # reward functions: boot-wait, tests, evidence, blocker checks
METRIC_SYNC_SECONDS = "timing/sync_seconds"  # pushing the merged weights to vLLM
METRIC_TRAIN_SECONDS = "timing/train_seconds"  # forward + backward + optimizer: step time minus rollout and grading


def probe_safe_trainer_class() -> type:
    """`HonestGRPOTrainer` plus two things TRL's loop does not give us.

    **An evaluation pass that fits in memory.** TRL's evaluation loop reuses `per_device_eval_batch_size` for two
    different things: how many episodes are in flight (the probe wants many, so 100 variants take minutes) and how many
    sequences go through the model at once when the rollouts are scored. Training scores 2 sequences per forward; with
    `probe.batch_size` 25 the same code would put 25 sequences of up to ~20k tokens through the full-logits path
    (hundreds of GB), and a second time inside `compute_loss`. Here the scoring chunk stays at the training
    micro-batch, and `prediction_step` skips the loss, which the probe never reads (it reads the metrics the reward
    function logged).

    **Step timing by component** (`timing/*`): TRL only reports its own profiling to wandb or trackio, and the Phase-0
    decision between TRL and verl needs grading, sync and training time per step in the run log.
    """
    import time

    import torch

    from honest_agents.trainer import HonestGRPOTrainer

    class ProbeSafeGRPOTrainer(HonestGRPOTrainer):
        def __init__(self, *args: Any, **kwargs: Any):
            self._prepare_seconds = 0.0  # time inside `_prepare_inputs` (generation, grading, sync), cumulative
            self._train_seconds = 0.0  # compute of the micro-steps of the current optimizer step
            super().__init__(*args, **kwargs)
            vllm_generation = getattr(self, "vllm_generation", None)
            if vllm_generation is not None:
                sync = vllm_generation.sync_weights

                def timed_sync(*a: Any, **k: Any):
                    start = time.perf_counter()
                    try:
                        return sync(*a, **k)
                    finally:
                        self._log_seconds(METRIC_SYNC_SECONDS, time.perf_counter() - start)

                vllm_generation.sync_weights = timed_sync

        def _log_seconds(self, name: str, seconds: float) -> None:
            self._metrics["train" if self.model.training else "eval"][name].append(seconds)

        def _calculate_rewards(self, *args: Any, **kwargs: Any):
            start = time.perf_counter()
            try:
                return super()._calculate_rewards(*args, **kwargs)
            finally:
                self._log_seconds(METRIC_GRADING_SECONDS, time.perf_counter() - start)

        def _prepare_inputs(self, generation_batch):
            start = time.perf_counter()
            try:
                return super()._prepare_inputs(generation_batch)
            finally:
                self._prepare_seconds += time.perf_counter() - start

        def training_step(self, model, inputs, num_items_in_batch=None):
            prepared, start = self._prepare_seconds, time.perf_counter()
            output = super().training_step(model, inputs, num_items_in_batch)
            self._train_seconds += time.perf_counter() - start - (self._prepare_seconds - prepared)
            if self._step % self.current_gradient_accumulation_steps == 0:  # TRL's own optimizer-step boundary
                self._metrics["train"][METRIC_TRAIN_SECONDS].append(self._train_seconds)
                self._train_seconds = 0.0
            return output

        def _get_per_token_logps_and_entropies(self, model, *args, batch_size=None, **kwargs):
            if not self.model.training:  # evaluation: bound the scoring chunk by the training micro-batch
                cap = int(self.args.per_device_train_batch_size)
                batch_size = cap if batch_size is None else min(int(batch_size), cap)
            return super()._get_per_token_logps_and_entropies(model, *args, batch_size=batch_size, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            self._prepare_inputs(inputs)  # roll out, grade and log `eval_*` metrics; nothing is learned from them
            return torch.zeros((), device=self.accelerator.device), None, None

    return ProbeSafeGRPOTrainer


def make_trainer(plan: Plan, *, check_server: bool = True) -> Any:
    """Everything `train()` does before the first step: guards, bank and quarantine paths, the trainer, the template and
    prompt-budget checks, the callbacks. `sync_check` builds the same trainer so it tests the code path that trains."""
    from honest_agents import bank
    from honest_agents.env import set_max_concurrent_boots

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    _check_world_size(plan)
    _preflight_gpu(plan)
    check_lm_head_cast(plan.cfg)
    set_max_concurrent_boots(plan.cfg.env.max_concurrent_boots)
    if check_server:
        startup_probe(plan)
    bank.set_bank_dir(bank_dir(plan.cfg) if plan.cfg.bank.enabled else None)
    bank.set_quarantine_file(quarantine_file(plan.cfg))

    trainer_class = probe_safe_trainer_class()  # lazy: --dry-run works without the trainer
    trainer = trainer_class(
        model=plan.cfg.model.name,
        reward_funcs=plan.reward_funcs,
        args=plan.grpo_config,
        train_dataset=plan.train_dataset,
        eval_dataset=plan.eval_dataset,
        peft_config=plan.peft_config,
        environment_factory=plan.environment_factory,
    )
    trainer.max_tool_workers = plan.cfg.trainer.max_tool_workers
    trainer.trim_padding = plan.cfg.trainer.trim_padding
    trainer.sort_micro_batches = plan.cfg.trainer.sort_micro_batches
    check_training_template(trainer, plan.family, strict=plan.cfg.model.strict_template)
    check_tool_flow_template(trainer, strict=plan.cfg.model.strict_template)
    check_end_of_turn_token(trainer, strict=plan.cfg.model.strict_template)
    check_prompt_budget(trainer, plan)
    if list(trainer.reward_weights.tolist()) != list(plan.grpo_config.reward_weights):
        raise RuntimeError(
            f"trainer reward weights {trainer.reward_weights.tolist()} != configured "
            f"{plan.grpo_config.reward_weights} (did the environment add a reward source?)"
        )
    add_callbacks(trainer, plan)
    return trainer


def train(plan: Plan, resume: bool = False) -> bool:
    """Train; returns True if a stop rule ended the run early (`main` turns that into exit status 3)."""
    from transformers import set_seed

    set_seed(plan.cfg.seed)
    if not plan.cfg.data.allow_legacy_verifier:  # fail before model loading, with the command that builds the bundle
        from honest_agents.verifier import ensure_bundle

        bundle = ensure_bundle(Path(plan.cfg.env.verifier_bundle_dir) if plan.cfg.env.verifier_bundle_dir else None)
        logger.info("verifier bundle: %s", bundle)
    out = Path(plan.grpo_config.output_dir)
    ckpt = _last_checkpoint(out) if resume else None
    if resume and ckpt is None:
        logger.warning("--resume: no checkpoint under %s; starting fresh", out)
    check_resume_order(out, plan.order, resuming=ckpt is not None)
    if ckpt is not None:
        plan.grpo_config.eval_on_start = False  # the step-0 probe already ran in the first life of the run
    if int(os.environ.get("RANK", "0")) == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps(dataclasses.asdict(plan.cfg), indent=2, default=str))
    trainer = make_trainer(plan)
    report = getattr(trainer.model, "print_trainable_parameters", None)
    if trainer.accelerator.is_main_process and report is not None:
        report()
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(str(out / "final"))
    best = out / "best_checkpoint.json"
    if best.is_file():
        logger.info("checkpoint selected on validation: %s", best.read_text().replace("\n", " "))
    return stopped_by_rule(trainer)


def stopped_by_rule(trainer: Any) -> bool:
    """True if a stop rule (not `max_steps`) ended the run with `stop.action: stop`."""
    from honest_agents.callbacks import StopRules

    handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(handler, "callbacks", None) or getattr(trainer, "callbacks", None) or []
    return any(isinstance(c, StopRules) and c.stopped for c in callbacks)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="honest_agents.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="YAML run config (may use `base:` inheritance)")
    ap.add_argument("--dry-run", action="store_true", help="build and validate everything except model/trainer")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint-* in the run's output_dir")
    ap.add_argument(
        "--print-limits",
        action="store_true",
        help="print the limits and the vLLM --max-model-len as JSON and exit (used by scripts/serve_vllm.sh)",
    )
    ap.add_argument(
        "--phase0-summary",
        default=None,
        help="Phase-0 phase0_summary.json: refuse an alpha with (1 + alpha) / 2 above its pool pass rate minus 0.1",
    )
    ap.add_argument(
        "--allow-unsafe-alpha", action="store_true", help="run an alpha refused by --phase0-summary (logs a warning)"
    )
    ap.add_argument(
        "--check-alpha",
        action="store_true",
        help=f"only check the alpha against --phase0-summary and exit (status {EXIT_UNSAFE_ALPHA} if refused)",
    )
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. reward.config.alpha=0.25 grpo.max_steps=2")
    args = ap.parse_intermixed_args(argv)  # flags may follow the dotted overrides
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_run_config(args.config, args.overrides)
    try:
        check_safe_alpha(cfg, args.phase0_summary, allow_unsafe=args.allow_unsafe_alpha)
    except UnsafeAlphaError as e:
        print(f"train: {e}", file=sys.stderr)
        return EXIT_UNSAFE_ALPHA
    if args.check_alpha:
        return 0
    if args.print_limits:
        limits = build_limits(cfg)
        print(
            json.dumps(
                {
                    "max_model_len": served_max_model_len(cfg, limits),
                    "turn_max_tokens": limits.turn_max_tokens,
                    "max_completion_length": limits.max_completion_length,
                    "max_tool_calling_iterations": limits.max_tool_calling_iterations,
                    "prompt_token_reserve": limits.prompt_token_reserve,
                    "server_url": _server_url(cfg),
                    "group_port": cfg.grpo.get("vllm_group_port", 51216),
                }
            )
        )
        return 0
    plan = build_plan(cfg)
    print(json.dumps(plan.summary(), indent=2, default=str))
    if args.dry_run:
        mismatches = check_template_offline(cfg, plan.family)
        if mismatches:
            raise RuntimeError("chat template is not prefix-preserving for the tool flow: " + "; ".join(mismatches))
        return 0
    if train(plan, resume=args.resume):
        logger.error("run stopped by a stop rule (see stop_rules.jsonl): exit status %d", EXIT_STOPPED_BY_RULE)
        return EXIT_STOPPED_BY_RULE
    return 0


if __name__ == "__main__":
    sys.exit(main())
