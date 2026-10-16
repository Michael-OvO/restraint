"""Supervised stage before GRPO: `python -m honest_agents.sft.train_sft --config configs/sft_hindsight.yaml`.

One entrypoint for every SFT stage (`kind:` in the config picks the example builder): `hindsight` (report-only
hindsight SFT, `sft.hindsight`), reward-filtered best-of-N (`bestofn`) and the `combo` mixture of both (`sft.bestofn`),
imitation of audited teacher trajectories (`teacher`, `sft.teacher_stage`), context distillation of hinted base-model
rollouts (`selfdistill`, `sft.selfdistill`) and decision-point supervision at the model's own high_risk states
(`decision`, `sft.decision`) and judgment items (`judgment`, `sft.judgment`, iteration 15); the last six register their builders with `register_stage`.
A stage turns the fit rows and the held-out rows of a trajectory bank into `SFTExample`s; this module tokenises them
with the loss mask of `sft.data`, wires TRL's `SFTTrainer` with PEFT LoRA and writes the adapter.

The stage is meant to be light (one epoch, low learning rate, LoRA) so that sampling entropy is not collapsed before
GRPO. Two checks guard that: the held-out batch is evaluated every `train.eval_steps` steps and `eval_entropy` (mean
policy entropy over the trained tokens) is logged, `guard.entropy_floor_ratio` optionally stops the run when it falls
below that fraction of its starting value; and `--spread BANK` reports the fraction of prompt groups whose rewards still
differ, for the bank the stage started from and, once it exists, a bank sampled from the tuned policy.

`--dry-run` builds the examples, the tokenised dataset (when the tokenizer is cached) and the trainer config without
loading a model. `run_sft` also takes a ready model and tokenizer, which is how the tests fit a tiny random model on CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.models import ModelFamily, detect_family, family_spec, lora_target_modules
from honest_agents.pilot.run import apply_overrides, load_config_dict
from honest_agents.sft import data, hindsight

logger = logging.getLogger("honest_agents.sft.train_sft")


# ── config ───────────────────────────────────────────────────────────────────


@dataclass
class ModelSection:
    name: str = "Qwen/Qwen3-4B"
    family: str = "auto"
    dtype: str = "bfloat16"
    revision: str | None = None
    attn_implementation: str | None = None
    local_files_only: bool = False  # only read the tokenizer / model from the local cache


@dataclass
class LoraSection:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Any = "auto"


@dataclass
class DataSection:
    bank: str = "outputs/phase0/bank"  # bank directory (`bank-*.jsonl`) or one bank file
    holdout_fraction: float = 0.1  # of VARIANTS (or pairs, see `partition_by`): held-out entropy batch and calibration
    partition_by: str = "variant"  # `variant` | `pair`: what the held-out split keeps together (`data.PARTITION_KEYS`)
    max_length: int = 16384  # longer examples are dropped (truncating would cut the report)
    max_rows: int | None = None  # cap on fit examples, taken in a stable pseudo-random order
    min_examples: int = 16  # refuse a run with fewer fit examples
    row_filter: dict[str, Any] = field(default_factory=dict)  # `data.RowFilter` fields
    dump_examples: bool = True  # write `examples.jsonl` next to the run, for reading the labels by eye


@dataclass
class TrainSection:
    learning_rate: float = 5e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_steps: float = 0.1  # int = steps; float in [0, 1) = share of all steps
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    logging_steps: int = 1
    eval_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    report_to: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)  # further `SFTConfig` fields


@dataclass
class GuardSection:
    entropy_floor_ratio: float | None = None  # stop when eval entropy < ratio * its value at the start


@dataclass
class SftConfig:
    kind: str = "hindsight"
    seed: int = 0
    output_root: str = "outputs/sft"
    run_name: str | None = None
    model: ModelSection = field(default_factory=ModelSection)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataSection = field(default_factory=DataSection)
    hindsight: dict[str, Any] = field(default_factory=dict)  # `hindsight.HindsightOptions` fields
    bestofn: dict[str, Any] = field(default_factory=dict)  # `bestofn.BestOfNOptions` fields (kinds bestofn, combo)
    teacher: dict[str, Any] = field(default_factory=dict)  # `teacher_stage.TeacherOptions` fields (kind teacher)
    selfdistill: dict[str, Any] = field(default_factory=dict)  # `selfdistill.SelfDistillOptions` (kind selfdistill)
    decision: dict[str, Any] = field(default_factory=dict)  # `decision.DecisionOptions` fields (kind decision)
    judgment: dict[str, Any] = field(default_factory=dict)  # `judgment.JudgmentOptions` fields (kind judgment)
    splice: dict[str, Any] = field(default_factory=dict)  # `judgment_splice.SpliceOptions` fields (kind judgment_splice)
    read_rft: dict[str, Any] = field(default_factory=dict)  # `read_rft.ReadRftOptions` fields (kind read_rft)
    train: TrainSection = field(default_factory=TrainSection)
    guard: GuardSection = field(default_factory=GuardSection)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / (self.run_name or f"{self.kind}-s{self.seed}")


def _section(cls: type, values: dict[str, Any] | None, name: str) -> Any:
    values = dict(values or {})
    allowed = {f.name for f in dataclasses.fields(cls)}
    if unknown := set(values) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `{name}:`; expected {sorted(allowed)}")
    return cls(**values)


# Stage modules imported on demand: importing one registers its builder (`register_stage`).
STAGE_MODULES = {
    "bestofn": "honest_agents.sft.bestofn",
    "combo": "honest_agents.sft.bestofn",
    "teacher": "honest_agents.sft.teacher_stage",
    "selfdistill": "honest_agents.sft.selfdistill",
    "decision": "honest_agents.sft.decision",
    "judgment": "honest_agents.sft.judgment",
    "judgment_splice": "honest_agents.sft.judgment_splice",
    "read_rft": "honest_agents.sft.read_rft",
}


def _stage_known(kind: str) -> bool:
    if kind not in STAGES and kind in STAGE_MODULES:
        import importlib

        try:
            importlib.import_module(STAGE_MODULES[kind])
        except ModuleNotFoundError as e:
            if e.name != STAGE_MODULES[kind]:
                raise
    return kind in STAGES


def parse_config(raw: dict[str, Any], *, check_kind: bool = True) -> SftConfig:
    raw = dict(raw)
    sections = {
        "model": ModelSection,
        "lora": LoraSection,
        "data": DataSection,
        "train": TrainSection,
        "guard": GuardSection,
    }
    built = {name: _section(cls, raw.pop(name, None), name) for name, cls in sections.items()}
    top = {f.name for f in dataclasses.fields(SftConfig)} - set(sections)
    if unknown := set(raw) - top:
        raise ValueError(f"unknown top-level key(s) {sorted(unknown)}; expected {sorted(top | set(sections))}")
    cfg = SftConfig(**raw, **built)
    if check_kind and not _stage_known(cfg.kind):
        raise ValueError(f"unknown or not yet implemented kind {cfg.kind!r}; available: {sorted(STAGES)}")
    if cfg.kind == hindsight.KIND:
        hindsight.HindsightOptions(**cfg.hindsight)  # unknown keys and bad values fail at parse time
    if (validator := STAGE_VALIDATORS.get(cfg.kind)) is not None:
        validator(cfg)
    if not 0.0 <= cfg.data.holdout_fraction < 1.0:
        raise ValueError("data.holdout_fraction must be in [0, 1)")
    if cfg.data.partition_by not in data.PARTITION_KEYS:
        raise ValueError(f"data.partition_by must be one of {sorted(data.PARTITION_KEYS)}, got {cfg.data.partition_by!r}")
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> SftConfig:
    return parse_config(apply_overrides(load_config_dict(path), list(overrides or [])))


def resolve_family(cfg: SftConfig) -> ModelFamily:
    return ModelFamily(cfg.model.family) if cfg.model.family != "auto" else detect_family(cfg.model.name)


# ── stages ───────────────────────────────────────────────────────────────────


@dataclass
class StageData:
    train: list[data.SFTExample]
    eval: list[data.SFTExample]
    info: dict[str, Any] = field(default_factory=dict)  # goes to the plan and `sft_run.json`


# A stage turns (config, fit rows, held-out rows, scorable rows) into examples. Fit and held-out rows are the usable
# rows split by variant; scorable rows are every train-split outcome that counts (for calibration).
StageBuilder = Callable[[SftConfig, list, list, list], StageData]
STAGES: dict[str, StageBuilder] = {}
STAGE_VALIDATORS: dict[str, Callable[[SftConfig], None]] = {}  # a stage's own checks of its config keys, run at parse


def register_stage(name: str, builder: StageBuilder, validator: Callable[[SftConfig], None] | None = None) -> None:
    STAGES[name] = builder
    if validator is not None:
        STAGE_VALIDATORS[name] = validator


def _hindsight_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    opts = hindsight.HindsightOptions(**cfg.hindsight)
    held_ids = {data.variant_of(r) for r in held}
    # Calibration reads held-out variants only, and every scorable row of them (also tampered ones: they did not pass).
    table = hindsight.calibrate([r for r in scorable if data.variant_of(r) in held_ids], opts)
    train = hindsight.build_examples(fit, table, opts)
    evals = hindsight.build_examples(held, table, opts)
    info = {
        "labels": hindsight.label_summary(train),
        "confidence": opts.confidence,
        "calibration": dataclasses.asdict(table),
    }
    return StageData(train, evals, info)


register_stage(hindsight.KIND, _hindsight_stage)


def _stable_order(rows: list) -> list:
    import hashlib

    return sorted(rows, key=lambda r: hashlib.sha256(r.episode_id.encode()).hexdigest())


def build_stage_data(cfg: SftConfig) -> tuple[StageData, dict[str, Any]]:
    """Read the bank, partition it by variant and build the stage's examples. Returns the data and a summary."""
    flt = data.RowFilter(**cfg.data.row_filter)
    sel = data.read_rows(cfg.data.bank, flt)
    fit, held = data.partition(sel.usable, cfg.data.holdout_fraction, cfg.seed, data.PARTITION_KEYS[cfg.data.partition_by])
    if cfg.data.max_rows is not None:
        fit = _stable_order(fit)[: cfg.data.max_rows]
    stage = STAGES[cfg.kind](cfg, fit, held, sel.scorable)
    spread = data.reward_spread(fit)
    summary = {
        "bank": cfg.data.bank,
        "rows_usable": len(sel.usable),
        "rows_dropped": dict(sel.dropped),
        "variants_fit": len({data.variant_of(r) for r in fit}),
        "variants_heldout": len({data.variant_of(r) for r in held}),
        "examples_fit": len(stage.train),
        "examples_heldout": len(stage.eval),
        "source_reward_spread": dataclasses.asdict(spread) | {"fraction": spread.fraction},
        **stage.info,
    }
    return stage, summary


# ── trainer ──────────────────────────────────────────────────────────────────


def build_lora_config(cfg: SftConfig, family: ModelFamily):
    from peft import LoraConfig

    targets = cfg.lora.target_modules
    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=lora_target_modules(family) if targets == "auto" else targets,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: SftConfig, family: ModelFamily, *, has_eval: bool):
    """The `SFTConfig`. The dataset is tokenised here (`input_ids` + `labels`), so TRL's own preparation is skipped and
    `max_length` is not applied again: over-long examples were dropped, never truncated."""
    from trl import SFTConfig

    t = cfg.train
    init_kwargs: dict[str, Any] = {"dtype": cfg.model.dtype}
    if cfg.model.revision:
        init_kwargs["revision"] = cfg.model.revision
    if cfg.model.attn_implementation:
        init_kwargs["attn_implementation"] = cfg.model.attn_implementation
    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.output_dir),
        "seed": cfg.seed,
        "learning_rate": t.learning_rate,
        "num_train_epochs": t.num_train_epochs,
        "max_steps": t.max_steps,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "warmup_steps": t.warmup_steps,
        "lr_scheduler_type": t.lr_scheduler_type,
        "weight_decay": t.weight_decay,
        "max_grad_norm": t.max_grad_norm,
        "gradient_checkpointing": t.gradient_checkpointing,
        "bf16": t.bf16,
        "logging_steps": t.logging_steps,
        "save_strategy": "steps",
        "save_steps": t.save_steps,
        "save_total_limit": t.save_total_limit,
        "report_to": t.report_to,
        "max_length": None,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "model_init_kwargs": init_kwargs,
        "remove_unused_columns": False,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": t.eval_steps,
        "eval_on_start": has_eval,
        **t.extra,
    }
    return SFTConfig(**kwargs)


def _entropy_guard(ratio: float):
    from transformers import TrainerCallback

    class EntropyFloor(TrainerCallback):
        """Stop when the held-out entropy falls below `ratio` times its value at the first evaluation."""

        def __init__(self) -> None:
            self.start: float | None = None

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            value = (metrics or {}).get("eval_entropy")
            if value is None:
                return
            if self.start is None:
                self.start = float(value)
            elif self.start > 0 and value < ratio * self.start:
                logger.warning(
                    "eval entropy %.4f fell below %.2f x its start (%.4f): stopping", value, ratio, self.start
                )
                control.should_training_stop = True

    return EntropyFloor()


def load_tokenizer(cfg: SftConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.model.name, revision=cfg.model.revision, local_files_only=cfg.model.local_files_only
    )


def tokenize_stage(stage: StageData, cfg: SftConfig, family: ModelFamily, tokenizer: Any):
    """(train dataset, eval dataset or None, tokenisation summary)."""
    renderer = data.make_renderer(tokenizer, family)
    train_tok, _, dropped = data.tokenize_examples(stage.train, renderer, cfg.data.max_length)
    eval_tok, _, dropped_eval = data.tokenize_examples(stage.eval, renderer, cfg.data.max_length)
    if len(train_tok) < cfg.data.min_examples:
        raise ValueError(f"only {len(train_tok)} training examples (data.min_examples={cfg.data.min_examples})")
    lengths = [len(t.input_ids) for t in train_tok]
    trained = [t.n_trained for t in train_tok]
    summary = {
        "tokenized_fit": len(train_tok),
        "tokenized_heldout": len(eval_tok),
        "dropped_fit": dict(dropped),
        "dropped_heldout": dict(dropped_eval),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "mean_trained_tokens": sum(trained) / len(trained),
    }
    return data.to_hf_dataset(train_tok), (data.to_hf_dataset(eval_tok) if eval_tok else None), summary


def run_sft(cfg: SftConfig, *, model: Any = None, tokenizer: Any = None, resume: bool = False) -> dict[str, Any]:
    """Build the data, train, save the adapter to `<output_dir>/final`. `model` / `tokenizer` may be passed ready-made
    (tests); otherwise the model is loaded by TRL from `model.name` with `model.dtype`."""
    from trl import SFTTrainer

    family = resolve_family(cfg)
    stage, summary = build_stage_data(cfg)
    tokenizer = tokenizer or load_tokenizer(cfg)
    train_ds, eval_ds, tok_summary = tokenize_stage(stage, cfg, family, tokenizer)
    summary |= tok_summary
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if cfg.data.dump_examples:
        data.write_examples(stage.train + stage.eval, out / "examples.jsonl")
    sft_config = build_sft_config(cfg, family, has_eval=eval_ds is not None)
    callbacks = [_entropy_guard(cfg.guard.entropy_floor_ratio)] if cfg.guard.entropy_floor_ratio and eval_ds else []
    trainer = SFTTrainer(
        model=model if model is not None else cfg.model.name,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg, family),
        callbacks=callbacks,
    )
    last = None
    if resume:
        from transformers.trainer_utils import get_last_checkpoint

        last = get_last_checkpoint(str(out)) if out.is_dir() else None
    result = trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out / "final"))
    summary["train_loss"] = result.training_loss
    summary["log_history"] = trainer.state.log_history
    (out / "sft_run.json").write_text(
        json.dumps({"config": dataclasses.asdict(cfg), "summary": summary}, default=str, indent=2)
    )
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def dry_run(cfg: SftConfig, tokenizer: Any = None) -> dict[str, Any]:
    """Everything except the model: the examples, the tokenised dataset when the tokenizer is cached, the configs."""
    family = resolve_family(cfg)
    stage, summary = build_stage_data(cfg)
    try:
        tokenizer = tokenizer or load_tokenizer(cfg)
    except Exception as e:  # noqa: BLE001 - not cached, or no network
        summary["tokenization"] = f"skipped: tokenizer for {cfg.model.name} not available ({type(e).__name__})"
        has_eval = bool(stage.eval)
    else:
        _, eval_ds, tok_summary = tokenize_stage(stage, cfg, family, tokenizer)
        summary |= tok_summary
        has_eval = eval_ds is not None
    sft_config = build_sft_config(cfg, family, has_eval=has_eval)
    lora = build_lora_config(cfg, family)
    summary["family"] = family.value
    summary["chat_template_kwargs"] = family_spec(family).chat_template_kwargs
    summary["output_dir"] = sft_config.output_dir
    summary["lora"] = {"r": lora.r, "alpha": lora.lora_alpha, "targets": str(lora.target_modules)[:120]}
    summary["effective_batch"] = sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps
    if cfg.data.dump_examples:
        data.write_examples(stage.train + stage.eval, cfg.output_dir / "examples.jsonl")
    return summary


def spread_report(bank_path: str, group: str | None = None) -> dict[str, Any]:
    """Within-group reward spread of a bank (train-split rows only), for the before / after comparison."""
    rows = [r for r in bank_mod.iter_bank(bank_path) if r.task.get("split") == data.TRAIN_SPLIT]
    stats = data.reward_spread(rows)
    return dataclasses.asdict(stats) | {"fraction": stats.fraction, "bank": bank_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", help="YAML config (`base:` inheritance supported)")
    parser.add_argument("--dry-run", action="store_true", help="build examples and configs, load no model")
    parser.add_argument("--spread", metavar="BANK", help="print the within-group reward spread of a bank and exit")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint of the run")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.spread:
        print(json.dumps(spread_report(args.spread), indent=2))
        return 0
    if not args.config:
        parser.error("--config is required")
    cfg = load_config(args.config, args.overrides)
    summary = dry_run(cfg) if args.dry_run else run_sft(cfg, resume=args.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    # Under `python -m` this file is `__main__`, a second copy of the module: stage modules register into the imported
    # `honest_agents.sft.train_sft`, so dispatch through that one or `kind: bestofn` is never found.
    from honest_agents.sft.train_sft import main as _main

    sys.exit(_main())
