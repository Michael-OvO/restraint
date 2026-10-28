"""Train the report's confidence only: GRPO over the value of `confidence` (iteration 17, docs/iteration-17-design.md).

    uv run --extra train python -m honest_agents.train.confidence --config configs/it17_confidence.yaml [--dry-run]
        [key=value overrides, e.g. model.name=outputs/it15/g2_merged]

Each prompt is one `sft.confidence_bandit` item rendered with the served chat template and cut right before the
model's own confidence value (the raw text tokenises to exactly the template's ids, checked on all it16b items). The
trainer samples `num_generations` continuations of `max_completion_length` tokens, the reward is the Brier score of
the parsed value against the episode's graded outcome (`confidence_bandit.brier_reward`; unparsable = -1), and the
loss is on those sampled tokens only. Nothing else of the episode or the report is trained.

* **Data:** the items files; prompts over `max_prompt_tokens` are dropped (counted), never cut. A held-out split by
  source (`heldout_fraction`, stable hash) is the eval set: its mean reward is the held-out Brier, logged each
  `eval_steps` and used for no selection (the select split selects, as in it16).
* **Fixed rules** (as `train.grpo`): `scale_rewards: none`, gradient checkpointing on, W&B, LoRA from the start model
  with `beta` > 0 so the KL keeps the policy at G2 everywhere the reward does not reach.

`--dry-run` builds the dataset and the trainer config, renders and tokenises every prompt, loads no model and reports
the counts, the prompt lengths and the label mix.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from honest_agents.sft import confidence_bandit as cb

logger = logging.getLogger(__name__)


@dataclass
class ModelSection:
    name: str = "outputs/it15/g2_merged"
    family: str = "qwen3_5"
    dtype: str = "bfloat16"
    tokenizer: str | None = None  # default: `name`


@dataclass
class LoraSection:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Any = "auto"


@dataclass
class DataSection:
    items: list[str] = field(default_factory=list)
    max_prompt_tokens: int = 16384
    heldout_fraction: float = 0.1
    seed: str = "it17-confidence"


@dataclass
class ConfidenceConfig:
    run_name: str = "it17-confidence"
    output_root: str = "outputs/confidence"
    seed: int = 0
    model: ModelSection = field(default_factory=ModelSection)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataSection = field(default_factory=DataSection)
    grpo: dict[str, Any] = field(default_factory=dict)  # GRPOConfig fields

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name


FIXED = {"scale_rewards": "none", "gradient_checkpointing": True}
DEFAULT_GRPO = {
    "learning_rate": 1.0e-5,
    "num_generations": 8,
    "max_completion_length": 8,
    "temperature": 1.0,
    "beta": 0.04,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "logging_steps": 1,
    "eval_strategy": "steps",
    "eval_steps": 25,
    "save_strategy": "epoch",
    "report_to": "wandb",
    "log_completions": False,
}


def _section(cls: type, raw: dict[str, Any] | None, name: str) -> Any:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(cls)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `{name}:`; expected {sorted(allowed)}")
    return cls(**raw)


def _set(raw: dict[str, Any], dotted: str, value: str) -> None:
    *path, last = dotted.split(".")
    d = raw
    for p in path:
        d = d.setdefault(p, {})
    d[last] = yaml.safe_load(value)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ConfidenceConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for o in overrides:
        key, _, value = o.partition("=")
        _set(raw, key, value)
    top = {k: v for k, v in raw.items() if k not in ("model", "lora", "data", "grpo")}
    cfg = ConfidenceConfig(
        **top,
        model=_section(ModelSection, raw.get("model"), "model"),
        lora=_section(LoraSection, raw.get("lora"), "lora"),
        data=_section(DataSection, raw.get("data"), "data"),
        grpo={**DEFAULT_GRPO, **dict(raw.get("grpo") or {})},
    )
    if bad := {k: cfg.grpo.get(k) for k, v in FIXED.items() if k in cfg.grpo and cfg.grpo[k] != v}:
        raise ValueError(f"grpo: {bad} is refused; these are fixed to {FIXED}")
    if cfg.grpo["beta"] <= 0:
        raise ValueError("grpo.beta must be > 0: the KL keeps the policy at the start model off the reward's reach")
    if not cfg.data.items:
        raise ValueError("data.items is empty")
    return cfg


def heldout(source: str, fraction: float, seed: str) -> bool:
    h = int(hashlib.sha256(f"{seed}|{source}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < fraction


@dataclass
class Prepared:
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    dropped: Counter[str]
    lengths: list[int]


def prepare(cfg: ConfidenceConfig, tokenizer: Any) -> Prepared:
    """Rows `{prompt, y, item_id, source, cell}` for the trainer, split by source; over-long prompts dropped."""
    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec

    kwargs = dict(family_spec(cfg.model.family).chat_template_kwargs)
    tools = tool_schemas()
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    lengths: list[int] = []
    for path in cfg.data.items:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            item = cb.Item.from_json(json.loads(line))
            prompt = cb.render_prefix(item.messages, tokenizer, tools, kwargs)
            n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            if n > cfg.data.max_prompt_tokens:
                dropped["prompt_too_long"] += 1
                continue
            lengths.append(n)
            cell = f"{'feasible' if item.feasible else 'infeasible'}/{item.status}"
            row = {"prompt": prompt, "y": item.y, "item_id": item.id, "source": item.source, "cell": cell}
            (held if heldout(item.source, cfg.data.heldout_fraction, cfg.data.seed) else train).append(row)
    return Prepared(train, held, dropped, lengths)


def brier_rewards(completions: list[str], y: list[int], **_: Any) -> list[float]:
    """TRL reward function: the Brier reward of each sampled value against its item's label."""
    return [cb.brier_reward(cb.parse_confidence(c), int(label)) for c, label in zip(completions, y, strict=True)]


def build_grpo_config(cfg: ConfidenceConfig):
    from trl import GRPOConfig

    from honest_agents.train.dashboard import resolve_report_to

    grpo = {**cfg.grpo, **FIXED}
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    if unknown := set(grpo) - known:
        raise ValueError(f"unknown GRPOConfig field(s) {sorted(unknown)}")
    grpo["report_to"] = resolve_report_to(grpo.get("report_to", "none"))
    return GRPOConfig(output_dir=str(cfg.output_dir), run_name=cfg.run_name, seed=cfg.seed,
                      model_init_kwargs={"dtype": cfg.model.dtype}, **grpo)  # fmt: skip


def summarize(p: Prepared) -> dict[str, Any]:
    lengths = sorted(p.lengths)
    q = (lambda f: lengths[min(len(lengths) - 1, int(f * len(lengths)))]) if lengths else (lambda f: None)  # noqa: E731
    return {
        "train": len(p.train),
        "eval": len(p.eval),
        "dropped": dict(p.dropped),
        "train_sources": len({r["source"] for r in p.train}),
        "eval_sources": len({r["source"] for r in p.eval}),
        "cells_train": dict(sorted(Counter(f"{r['cell']}/y={r['y']}" for r in p.train).items())),
        "prompt_tokens": {"p50": q(0.5), "p95": q(0.95), "max": lengths[-1] if lengths else None},
    }


def run(cfg: ConfidenceConfig, *, dry_run: bool = False) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from honest_agents.models import pinned_revision

    name = cfg.model.tokenizer or cfg.model.name
    tok = AutoTokenizer.from_pretrained(name, revision=pinned_revision(name))
    prepared = prepare(cfg, tok)
    if set(r["source"] for r in prepared.train) & set(r["source"] for r in prepared.eval):
        raise AssertionError("a source is on both sides of the held-out split")
    out = {"config": dataclasses.asdict(cfg), "data": summarize(prepared)}
    grpo_cfg = build_grpo_config(cfg)
    if dry_run:
        return out
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOTrainer

    from honest_agents.models import lora_target_modules

    targets = cfg.lora.target_modules
    peft_cfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout, bias="none",
                          target_modules=lora_target_modules(cfg.model.family) if targets == "auto" else targets,
                          task_type="CAUSAL_LM")  # fmt: skip
    trainer = GRPOTrainer(
        model=cfg.model.name,
        reward_funcs=brier_rewards,
        args=grpo_cfg,
        train_dataset=Dataset.from_list(prepared.train),
        eval_dataset=Dataset.from_list(prepared.eval) if prepared.eval else None,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "run_summary.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final"))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.train.confidence", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value, e.g. model.name=outputs/it15/g2_merged")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    out = run(load_config(args.config, args.overrides), dry_run=args.dry_run)
    json.dump(out["data"], sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
