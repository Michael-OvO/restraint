"""Report-only pilot sweep: `python -m honest_agents.pilot.run --config configs/pilot.yaml [key.path=value ...]`.

One arm is one training run of the final report on frozen trajectories, for one (alpha, scale_rewards, seed). The sweep
is alpha x scale_rewards {none, group} (docs/design.md section 10, docs/roadmap.md Stage 2). Every arm of a seed sees the
same prompts in the same order (TRL's sampler is seeded) and the same LoRA initialisation, so arms differ only in the
reward gap, which makes the comparison across alpha paired.

Per arm the run writes, under `<output_root>/<arm>/`: `arm.json` (what was run), `episodes.jsonl` (one record per new
report: outcome, case, claim, reward), `curve.jsonl` (per-round rates and the realised threshold), the TRL output and
`final/`. `python -m honest_agents.pilot.analyze <output_root>` turns those into the stop rule and a go / no-go line;
this command runs it after the last arm.

Layout of a run: ONE training process, generation by a vLLM server, no sandbox, no Docker. Two
arms run at the same time by launching two processes with `--shard 0/2` and `--shard 1/2` on two GPU pairs. The weight
sync check (`scripts/sync_check.sh`, docs/design.md section 8) must have passed first: this is the first run that learns.

`--dry-run` builds the dataset, the arm list and every `GRPOConfig` without loading a model, and prints the plan.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from honest_agents.models import pinned_revision
from honest_agents.pilot import analyze, dataset
from honest_agents.pilot import rewards as rewards_mod
from honest_agents.pilot.env import ReplayEnv
from honest_agents.pilot.rewards import EpisodeLog, PilotReward
from honest_agents.reward import RewardConfig

logger = logging.getLogger("honest_agents.pilot.run")

SCALES = (
    "none",
    "group",
)  # `batch` is not part of the sweep: group is the arm that erases alpha, none is the target
# GRPOConfig fields derived from other sections; setting them under `grpo:` would silently disagree with the arm.
DERIVED_GRPO_FIELDS = frozenset(
    {
        "output_dir",
        "run_name",
        "seed",
        "reward_weights",
        "model_init_kwargs",
        "scale_rewards",
        "max_tool_calling_iterations",
    }
)
# `RewardConfig` fields the sweep sets itself.
SWEPT_REWARD_FIELDS = frozenset({"alpha"})
CONTEXT_PROBE_ROWS = 16  # longest prompts rendered with the real tokenizer to check the generation window


# ── config ───────────────────────────────────────────────────────────────────


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config_dict(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """YAML with `base: other.yaml` inheritance (relative to the file); the file's own keys win."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"config inheritance cycle: {' -> '.join(map(str, (*_seen, path)))}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    base = data.pop("base", None)
    return _deep_merge(load_config_dict(path.parent / base, (*_seen, path)), data) if base else data


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """`a.b.c=value` overrides; values are parsed as YAML (`sweep.alpha=[0,0.5]`, `grpo.max_steps=2`)."""
    d = copy.deepcopy(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must look like key.path=value")
        key, raw = ov.split("=", 1)
        parts = key.strip().split(".")
        node = d
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise TypeError(f"override {ov!r}: {p!r} is not a mapping")
        node[parts[-1]] = yaml.safe_load(raw)
    return d


def _section(cls: type, values: dict[str, Any] | None, name: str) -> Any:
    values = dict(values or {})
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `{name}:`; expected {sorted(allowed)}")
    return cls(**values)


@dataclass
class ModelSection:
    name: str = "Qwen/Qwen3-4B"
    family: str = "auto"
    dtype: str = "bfloat16"
    revision: str | None = None
    attn_implementation: str | None = None
    strict_template: bool = True
    require_fla: bool = True

    def __post_init__(self) -> None:
        self.revision = pinned_revision(self.name, self.revision)  # DEF-51: a pinned Hub id loads at its pin


@dataclass
class LoraSection:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: Any = "auto"


@dataclass
class DataSection:
    bank: str = "outputs/phase0/bank"  # bank directory (`bank-*.jsonl`) or one bank file
    dataset_file: str = "outputs/pilot/pilot.jsonl"  # built from the bank when missing
    rebuild: bool = False
    build: dict[str, Any] = field(default_factory=dict)  # `dataset.BuildOptions` fields
    min_rows: int = 64  # refuse a dataset smaller than this: the curves would be noise
    # Refuse a run that shows each prompt more than this many times. A prompt has one fixed outcome, so a repeated one
    # can be memorised instead of judged, which makes fabrication fall and claims discriminate for the wrong reason.
    max_epochs: float = 1.0


@dataclass
class RewardSection:
    mode: str = "honest"
    config: dict[str, Any] = field(default_factory=dict)  # `RewardConfig` fields except the swept alpha


@dataclass
class SweepSection:
    alpha: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    scale_rewards: list[str] = field(default_factory=lambda: ["none", "group"])
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])  # 3: the stop rule needs seed variance (SP5)


@dataclass
class LaunchSection:
    num_processes: int = 1  # docs/design.md section 8: one training process per run
    vllm_max_model_len: int | None = None  # the server's window, to check the longest context against


@dataclass
class PilotConfig:
    output_root: str = "outputs/pilot"
    seed: int = 42
    model: ModelSection = field(default_factory=ModelSection)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataSection = field(default_factory=DataSection)
    reward: RewardSection = field(default_factory=RewardSection)
    sweep: SweepSection = field(default_factory=SweepSection)
    launch: LaunchSection = field(default_factory=LaunchSection)
    analysis: dict[str, Any] = field(default_factory=dict)
    grpo: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PilotConfig:
        d = dict(d)
        sections = {
            "model": ModelSection,
            "lora": LoraSection,
            "data": DataSection,
            "reward": RewardSection,
            "sweep": SweepSection,
            "launch": LaunchSection,
        }
        built = {k: _section(c, d.pop(k, None), k) for k, c in sections.items()}
        allowed = {"output_root", "seed", "analysis", "grpo", "config_path"}
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"unknown top-level key(s) {sorted(unknown)}; expected {sorted(allowed | set(sections))}")
        cfg = cls(**built, **d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.reward.mode not in (
            "honest",
            "outcome",
            "outcome_claimed",
            "outcome_reflect",
        ):
            raise ValueError(f"reward.mode {self.reward.mode!r} is not a trl_rewards mode")
        swept = set(self.reward.config) & SWEPT_REWARD_FIELDS
        if swept:
            raise ValueError(f"reward.config must not set {sorted(swept)}: the sweep sets it (use sweep.alpha)")
        bad = [s for s in self.sweep.scale_rewards if s not in SCALES]
        if bad or not self.sweep.scale_rewards:
            raise ValueError(
                f"sweep.scale_rewards must be a non-empty subset of {SCALES}, got {self.sweep.scale_rewards}"
            )
        if not self.sweep.alpha or not self.sweep.seeds:
            raise ValueError("sweep.alpha and sweep.seeds must not be empty")
        for a in self.sweep.alpha:
            RewardConfig(**{**self.reward.config, "alpha": float(a)})  # validates every cell's reward config now
        if self.launch.num_processes != 1:
            raise ValueError("the pilot runs one training process per run")
        analyze.StopRule.from_dict(self.analysis)
        dataset.BuildOptions(**_build_kwargs(self.data.build))


def load_pilot_config(path: str | Path, overrides: list[str] | None = None) -> PilotConfig:
    cfg = PilotConfig.from_dict(apply_overrides(load_config_dict(path), overrides or []))
    cfg.config_path = str(path)
    return cfg


def _build_kwargs(build: dict[str, Any]) -> dict[str, Any]:
    """`data.build` as `BuildOptions` keywords; `mix: default` is the docs/design.md section 7 split."""
    kwargs = dict(build)
    if kwargs.get("mix") in ("default", "none"):
        kwargs["mix"] = dict(dataset.DEFAULT_MIX) if kwargs["mix"] == "default" else None
    return kwargs


# ── arms ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Arm:
    alpha: float
    scale_rewards: str
    seed: int

    @property
    def name(self) -> str:
        return f"a{self.alpha:g}_{self.scale_rewards}_s{self.seed}"

    def reward_config(self, base: dict[str, Any]) -> RewardConfig:
        return RewardConfig(**{**base, "alpha": float(self.alpha)})


def expand_arms(cfg: PilotConfig) -> list[Arm]:
    """The sweep, seed-major so a partial run covers every cell of a seed before the next seed starts."""
    return [
        Arm(float(a), s, int(seed))
        for seed in cfg.sweep.seeds
        for s in cfg.sweep.scale_rewards
        for a in cfg.sweep.alpha
    ]


def select_arms(arms: list[Arm], only: list[str] | None = None, shard: str | None = None) -> list[Arm]:
    """Arms matching any `--arm` name or `alpha=..,scale=..,seed=..` filter, then shard `K/N` (arm index mod N == K)."""
    out = arms
    if only:
        out = [a for a in out if any(_matches(a, spec) for spec in only)]
    if shard:
        k, _, n = shard.partition("/")
        if not (k.isdigit() and n.isdigit() and int(n) >= 1 and int(k) < int(n)):
            raise ValueError(f"--shard must look like K/N with 0 <= K < N, got {shard!r}")
        out = [a for i, a in enumerate(arms) if i % int(n) == int(k) and a in out]
    return out


def _matches(arm: Arm, spec: str) -> bool:
    if spec == arm.name:
        return True
    want: dict[str, str] = {}
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        if not sep or key.strip() not in ("alpha", "scale", "scale_rewards", "seed"):
            return False
        want["scale" if key.strip().startswith("scale") else key.strip()] = value.strip()
    return all(
        {"alpha": f"{arm.alpha:g}", "scale": arm.scale_rewards, "seed": str(arm.seed)}[k]
        == (f"{float(v):g}" if k == "alpha" else v)
        for k, v in want.items()
    )


def arm_dir(cfg: PilotConfig, arm: Arm) -> Path:
    return Path(cfg.output_root) / arm.name


def arm_done(cfg: PilotConfig, arm: Arm) -> bool:
    meta = arm_dir(cfg, arm) / analyze.ARM_FILE
    return meta.exists() and bool(json.loads(meta.read_text()).get("completed"))


# ── data ─────────────────────────────────────────────────────────────────────


def ensure_dataset(
    cfg: PilotConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The pilot rows: read `data.dataset_file`, or build it from the bank (and write it) when missing or `rebuild`."""
    path = Path(cfg.data.dataset_file)
    stats = None
    if path.exists() and not cfg.data.rebuild:
        rows = dataset.read_rows(path)
    else:
        opts = dataset.BuildOptions(**_build_kwargs(cfg.data.build))
        rows, built = dataset.build_from_bank(cfg.data.bank, opts)
        dataset.write_rows(rows, path)
        stats = built.to_dict()
        Path(str(path) + ".stats.json").write_text(json.dumps(stats, indent=2))
        for warning in built.warnings:
            logger.warning("dataset: %s", warning)
    if len(rows) < cfg.data.min_rows:
        raise ValueError(f"{path}: {len(rows)} pilot rows, need at least data.min_rows={cfg.data.min_rows}")
    return rows, stats


def prompts_per_round(grpo: dict[str, Any]) -> int:
    """Distinct prompts in one generation round (one reward call): generation batch size over G."""
    g = int(grpo.get("num_generations", 8))
    batch = grpo.get("generation_batch_size")
    if batch is None:
        per_device = int(grpo.get("per_device_train_batch_size", 8))
        steps = grpo.get("steps_per_generation") or int(grpo.get("gradient_accumulation_steps", 1))
        batch = per_device * int(steps)  # one process
    if int(batch) % g:
        raise ValueError(f"generation batch size {batch} is not a multiple of num_generations {g}")
    return int(batch) // g


def total_prompts(grpo: dict[str, Any]) -> int | None:
    """Prompts the run consumes: each optimiser step uses per_device x accumulation completions, G per prompt.
    None when the length is set by epochs instead of `max_steps`."""
    steps = grpo.get("max_steps")
    if not steps or int(steps) < 1:
        return None
    g = int(grpo.get("num_generations", 8))
    per_step = int(grpo.get("per_device_train_batch_size", 8)) * int(grpo.get("gradient_accumulation_steps", 1))
    return int(steps) * per_step // g


def check_epochs(cfg: PilotConfig, rows: list[dict[str, Any]]) -> float | None:
    """Epochs over the dataset the run makes; raises when that exceeds `data.max_epochs` (memorisation guard)."""
    total = total_prompts(cfg.grpo)
    if total is None:
        epochs = float(cfg.grpo.get("num_train_epochs", 1.0))
    else:
        epochs = total / len(rows)
    if epochs > cfg.data.max_epochs + 1e-9:
        raise ValueError(
            f"the run shows each of the {len(rows)} prompts {epochs:.2f} times (data.max_epochs={cfg.data.max_epochs:g}). "
            "Every prompt has one fixed outcome, so a repeated prompt can be memorised and the pilot would read a "
            "memorised outcome as calibration. Lower grpo.max_steps, collect more Phase-0 trajectories, or raise "
            "data.max_epochs knowing this."
        )
    return epochs


def check_context_window(rows: list[dict[str, Any]], tokenizer: Any, tool_schemas: list[dict], budget: int) -> int:
    """Render the longest contexts with the real tokenizer and tools; raise if one does not fit `budget` tokens.

    TRL does not truncate prompts and vLLM rejects a request whose prompt plus `max_tokens` exceeds its window.
    Returns the longest rendered length found.
    """
    longest = sorted(rows, key=lambda r: dataset._content_chars(r["prompt"]), reverse=True)[:CONTEXT_PROBE_ROWS]
    worst = 0
    for r in longest:
        ids = tokenizer.apply_chat_template(r["prompt"], tools=tool_schemas, add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        worst = max(worst, len(ids))
    if worst > budget:
        raise ValueError(
            f"the longest frozen context renders to {worst} tokens but only {budget} are available "
            "(vllm_max_model_len minus max_completion_length): lower data.build.max_context_chars or raise the window"
        )
    return worst


# ── GRPO config ──────────────────────────────────────────────────────────────


def validate_grpo(cfg: PilotConfig) -> dict[str, Any]:
    """The `grpo:` section, checked against `GRPOConfig` and the pilot's constraints. Returns a copy."""
    from trl import GRPOConfig

    from honest_agents.models import check_full_distribution_sampling

    grpo = dict(cfg.grpo)
    unknown = set(grpo) - {f.name for f in dataclasses.fields(GRPOConfig)}
    if unknown:
        raise ValueError(f"unknown GRPOConfig field(s) under `grpo:`: {sorted(unknown)}")
    clash = set(grpo) & DERIVED_GRPO_FIELDS
    if clash:
        raise ValueError(f"`grpo:` must not set {sorted(clash)}: the sweep or the pilot derives them")
    if grpo.get("multi_objective_aggregation", "sum_then_normalize") != "sum_then_normalize":
        raise ValueError("the pilot assumes multi_objective_aggregation='sum_then_normalize'")
    check_full_distribution_sampling(grpo)
    if grpo.get("use_vllm") and grpo.get("vllm_mode", "colocate") == "server" and cfg.launch.num_processes > 1:
        raise ValueError("vLLM server mode with more than one training process mis-routes completions")
    prompts_per_round(grpo)
    return grpo


def build_grpo_config(cfg: PilotConfig, arm: Arm, family: Any):
    """The arm's `GRPOConfig`: the shared section plus the swept `scale_rewards`.

    `max_tool_calling_iterations=0`: TRL parses the report tool call but never executes it, so the completion is exactly
    one report turn and no token after the report is generated or trained on (`rewards.deliver_report` hands the call to
    the environment). `reward_weights=[1.0]` over the one reward function, so a None is a real mask.
    """
    from trl import GRPOConfig

    from honest_agents.models import family_spec

    grpo = validate_grpo(cfg)
    grpo.setdefault("chat_template_kwargs", dict(family_spec(family).chat_template_kwargs))
    model_init_kwargs: dict[str, Any] = {"dtype": cfg.model.dtype}
    if cfg.model.revision:
        model_init_kwargs["revision"] = cfg.model.revision
    if cfg.model.attn_implementation:
        model_init_kwargs["attn_implementation"] = cfg.model.attn_implementation
    return GRPOConfig(
        output_dir=str(arm_dir(cfg, arm) / "trainer"),
        run_name=f"pilot-{arm.name}",
        seed=arm.seed,
        reward_weights=[1.0],
        scale_rewards=arm.scale_rewards,
        max_tool_calling_iterations=0,
        model_init_kwargs=model_init_kwargs,
        **grpo,
    )


def build_lora_config(cfg: PilotConfig, family: Any):
    if not cfg.lora.enabled:
        return None
    from peft import LoraConfig

    from honest_agents.models import lora_target_modules

    targets = lora_target_modules(family) if cfg.lora.target_modules == "auto" else cfg.lora.target_modules
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )


def resolve_family(cfg: PilotConfig) -> Any:
    from honest_agents.models import ModelFamily, detect_family

    return ModelFamily(cfg.model.family) if cfg.model.family != "auto" else detect_family(cfg.model.name)


def window_rounds(cfg: PilotConfig) -> int:
    return analyze.StopRule.from_dict(cfg.analysis).window_rounds


def arm_metadata(cfg: PilotConfig, arm: Arm, rows: list[dict[str, Any]], completed: bool) -> dict[str, Any]:
    rc = arm.reward_config(cfg.reward.config)
    return {
        "name": arm.name,
        "alpha": arm.alpha,
        "scale_rewards": arm.scale_rewards,
        "seed": arm.seed,
        "reward_mode": cfg.reward.mode,
        "reward_config": dataclasses.asdict(rc),
        "claim_threshold_nominal": (1.0 + arm.alpha) / 2.0 if arm.scale_rewards == "none" else 0.5,
        "num_generations": int(cfg.grpo.get("num_generations", 8)),
        "prompts_per_round": prompts_per_round(cfg.grpo),
        "max_steps": cfg.grpo.get("max_steps"),
        "n_rows": len(rows),
        "model": cfg.model.name,
        "completed": completed,
    }


# ── plan and run ─────────────────────────────────────────────────────────────


def break_even(cfg: PilotConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where blanket claiming stops paying.

    With a claim at p(2 + lambda) - 1 against a decline at alpha + lambda p, a policy that claims everything earns more
    than one that never claims only if the mean pass rate exceeds (1 + alpha) / 2. Above that threshold the error-penalised
    rule (arXiv 2608.00301) drifts to decline-everything, which the pilot should show as a collapse (claim rate below the
    floor), not a calibrated threshold. Reported per alpha so the reader knows which arms test calibration and which test
    collapse.
    """
    feasible = [r for r in rows if r["feasible"]]
    share = sum(bool(r["passed"]) for r in feasible) / len(feasible) if feasible else None
    return {
        "feasible_pass_share": share,
        "alphas_where_blanket_claiming_loses": [
            a for a in cfg.sweep.alpha if share is not None and share < (1 + a) / 2
        ],
    }


def plan_summary(
    cfg: PilotConfig,
    arms: list[Arm],
    rows: list[dict[str, Any]],
    stats: dict[str, Any] | None,
) -> dict:
    kinds = {k: sum(r["kind"] == k for r in rows) for k in dataset.KINDS}
    ref = RewardConfig(**cfg.reward.config).efficiency_ref_actions
    banked_refs = (stats or {}).get("efficiency_ref_actions") or []
    return {
        "break_even": break_even(cfg, rows),
        "efficiency_ref_mismatch": [r for r in banked_refs if r != ref],
        "config": cfg.config_path,
        "output_root": cfg.output_root,
        "arms": [a.name for a in arms],
        "n_rows": len(rows),
        "kinds": kinds,
        "prompts_per_round": prompts_per_round(cfg.grpo),
        "rounds_per_epoch": len(rows) / max(1, prompts_per_round(cfg.grpo)),
        "epochs": (total_prompts(cfg.grpo) or 0) / len(rows) if rows else None,
        "max_steps": cfg.grpo.get("max_steps"),
        "dataset_stats": stats,
    }


def run_arm(cfg: PilotConfig, arm: Arm, rows: list[dict[str, Any]], resume: bool = False) -> Path:
    """Train one arm and write its records. Needs a GPU, a vLLM server and the model."""
    from transformers import set_seed
    from trl import GRPOTrainer

    from honest_agents.models import (
        check_training_template,
        family_spec,
        flash_linear_attention_available,
    )

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 1:
        raise RuntimeError(f"WORLD_SIZE={world}: the pilot runs one training process")
    family = resolve_family(cfg)
    if family_spec(family).needs_flash_linear_attention and not flash_linear_attention_available():
        msg = "flash-linear-attention (`fla`) is not installed; Qwen3.5 falls back to slow torch code"
        if cfg.model.require_fla:
            raise RuntimeError(msg + ". Install it or set model.require_fla=false.")
        logger.warning(msg)

    out = arm_dir(cfg, arm)
    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        archive_logs(out)
    (out / analyze.ARM_FILE).write_text(json.dumps(arm_metadata(cfg, arm, rows, completed=False), indent=2))
    set_seed(arm.seed)

    log = EpisodeLog(
        out,
        window=window_rounds(cfg),
        extra={
            "arm": arm.name,
            "alpha": arm.alpha,
            "scale_rewards": arm.scale_rewards,
            "seed": arm.seed,
        },
    )
    reward_func = PilotReward(cfg.reward.mode, log, scale_rewards=arm.scale_rewards)
    factory = functools.partial(ReplayEnv, reward_config=arm.reward_config(cfg.reward.config))
    args = build_grpo_config(cfg, arm, family)
    trainer = None
    try:
        trainer = GRPOTrainer(
            model=cfg.model.name,
            reward_funcs=[reward_func],
            args=args,
            train_dataset=dataset.to_hf_dataset(rows),
            peft_config=build_lora_config(cfg, family),
            environment_factory=factory,
        )
        check_training_template(trainer, family, strict=cfg.model.strict_template)
        if trainer.reward_weights.tolist() != [1.0]:
            raise RuntimeError(
                f"trainer reward weights {trainer.reward_weights.tolist()} != [1.0]: one reward function only"
            )
        if cfg.launch.vllm_max_model_len:
            budget = cfg.launch.vllm_max_model_len - int(cfg.grpo.get("max_completion_length", 512))
            from honest_agents.pilot.tools import tool_schemas

            worst = check_context_window(rows, trainer.processing_class, tool_schemas(), budget)
            logger.info("longest probed context: %d tokens of %d available", worst, budget)

        last = None
        if resume:
            from transformers.trainer_utils import get_last_checkpoint

            last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        trainer.train(resume_from_checkpoint=last)
        trainer.save_model(str(out / "final"))
    finally:
        trainer = None  # drop the last reference before the cache is emptied
        release_gpu()
    (out / analyze.ARM_FILE).write_text(json.dumps(arm_metadata(cfg, arm, rows, completed=True), indent=2))
    return out


def archive_logs(out: Path) -> list[Path]:
    """Move an earlier attempt's `episodes.jsonl` and `curve.jsonl` aside. A fresh run that appended to them would
    mix two policies in one arm's curves and read the failed attempt's rounds as this run's first window."""
    import time

    moved = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for name in (rewards_mod.EPISODES_FILE, rewards_mod.CURVE_FILE):
        path = out / name
        if path.exists() and path.stat().st_size > 0:
            target = out / f"{name}.{stamp}.old"
            path.rename(target)
            moved.append(target)
    return moved


def release_gpu() -> None:
    """Collect garbage and empty the CUDA cache so the next arm of this process starts clean."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("could not empty the CUDA cache", exc_info=True)


def _expected_names(cfg: PilotConfig, args: argparse.Namespace) -> list[str] | None:
    """Arm names a verdict needs, unless `--allow-partial`."""
    return None if args.allow_partial else [a.name for a in expand_arms(cfg)]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="honest_agents.pilot.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="pilot YAML (may use `base:` inheritance)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build the dataset and every arm's config, load no model",
    )
    ap.add_argument("--resume", action="store_true", help="resume each arm from its last checkpoint")
    ap.add_argument(
        "--force",
        action="store_true",
        help="run arms that are already marked completed",
    )
    ap.add_argument(
        "--arm",
        action="append",
        default=None,
        help="only this arm: its name or alpha=0.25,scale=none,seed=42",
    )
    ap.add_argument(
        "--shard",
        default=None,
        help="K/N: run every N-th arm starting at K (two processes, two GPU pairs)",
    )
    ap.add_argument(
        "--print-model",
        action="store_true",
        help="print `<model name> <family> [<revision>]` (for serve_vllm.sh) and exit",
    )
    ap.add_argument(
        "--list-arms",
        action="store_true",
        help="print the arm names, one per line (for scripts/run_pilot.sh) and exit",
    )
    ap.add_argument(
        "--print-limits",
        action="store_true",
        help="print the vLLM window and ports as JSON (the same keys as `honest_agents.train --print-limits`) and exit",
    )
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="skip training; analyse what is under output_root",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="with --analyze-only: do not require every arm of the sweep to be finished",
    )
    ap.add_argument(
        "overrides",
        nargs="*",
        help="dotted overrides, e.g. sweep.alpha=[0,0.5] grpo.max_steps=2",
    )
    return ap


def _print_limits(cfg: PilotConfig) -> None:
    """The vLLM window and ports as JSON, the same keys as `honest_agents.train --print-limits`."""
    window = cfg.launch.vllm_max_model_len
    if window is None:
        raise SystemExit("launch.vllm_max_model_len is not set: the server window must be explicit for the pilot")
    print(
        json.dumps(
            {
                "max_model_len": int(window),
                "turn_max_tokens": cfg.grpo.get("generation_kwargs", {}).get("max_tokens"),
                "max_completion_length": cfg.grpo.get("max_completion_length"),
                "server_url": cfg.grpo.get("vllm_server_base_url"),
                "group_port": cfg.grpo.get("vllm_group_port", 51216),
            }
        )
    )


def _run_arms(cfg: PilotConfig, arms: list[Arm], rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    """Train each arm that is not already completed; returns the names of the arms that failed."""
    failed: list[str] = []
    for arm in arms:
        if arm_done(cfg, arm) and not args.force:
            logger.info("%s: already completed, skipping (--force to rerun)", arm.name)
            continue
        logger.info("=== arm %s", arm.name)
        try:
            run_arm(cfg, arm, rows, resume=args.resume)
        except Exception:
            logger.exception("arm %s failed", arm.name)
            failed.append(arm.name)
    return failed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_intermixed_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_pilot_config(args.config, args.overrides)
    if args.print_model:
        print(cfg.model.name, resolve_family(cfg), cfg.model.revision or "")
        return 0
    if args.list_arms:
        print("\n".join(a.name for a in select_arms(expand_arms(cfg), args.arm, args.shard)))
        return 0
    if args.print_limits:
        _print_limits(cfg)
        return 0
    if args.analyze_only:
        verdict = analyze.analyse(
            cfg.output_root, analyze.StopRule.from_dict(cfg.analysis), expected=_expected_names(cfg, args)
        )
        print(analyze.format_report(verdict))
        return verdict.exit_code

    arms = select_arms(expand_arms(cfg), args.arm, args.shard)
    if not arms:
        raise SystemExit("no arm matches --arm / --shard")
    if len(arms) > 1 and not args.dry_run:
        logger.warning(
            "%d arms in one process: each arm opens its own weight-sync group to the one vLLM server. Prefer one arm "
            "per process and per server (`--arm NAME`), as scripts/run_slot.sh does for a training run.",
            len(arms),
        )
    rows, stats = ensure_dataset(cfg)
    if cfg.grpo:
        validate_grpo(cfg)
    check_epochs(cfg, rows)
    print(json.dumps(plan_summary(cfg, arms, rows, stats), indent=2, default=str))
    if args.dry_run:
        for arm in arms:
            build_grpo_config(cfg, arm, resolve_family(cfg))
        return 0

    failed = []
    for arm in arms:
        if arm_done(cfg, arm) and not args.force:
            logger.info("%s: already completed, skipping (--force to rerun)", arm.name)
            continue
        logger.info("=== arm %s", arm.name)
        try:
            run_arm(cfg, arm, rows, resume=args.resume)
        except Exception:
            logger.exception("arm %s failed", arm.name)
            failed.append(arm.name)
    if failed:
        print(f"FAILED arms: {failed}", file=sys.stderr)
        return 3
    if args.shard is None and not args.arm:
        verdict = analyze.analyse(
            cfg.output_root, analyze.StopRule.from_dict(cfg.analysis), expected=[a.name for a in expand_arms(cfg)]
        )
        print(analyze.format_report(verdict))
        return verdict.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
