"""Train the report's confidence by the exact expected Brier score (iteration 17, docs/iteration-17-design.md).

    uv run --extra train python -m honest_agents.train.confidence_exact --config configs/confidence/it17.yaml \\
        [--dry-run] [key=value overrides, e.g. model.name=outputs/it15/g2_merged]

The same objective as `train.confidence` (GRPO on the Brier reward of sampled values, KL to the start model), computed
in closed form instead of estimated from samples: the value the model writes after `<parameter=confidence>\\n` is
decided by at most three tokens (Qwen3.5 writes digits one per token: `1`, `0`, `0.85` is `0 . 8 5`), so one forward
pass over `prompt + "0."` gives every probability the value needs:

* node 1, after the prompt: `1` (value 1.0, whatever follows), `0`, anything else (unparsable);
* node 2, after `0`: `\\n` (value 0.0), `.`, anything else (unparsable);
* node 3, after `0.`: the tenths digit d (value d / 10; further digits are not resolved: `0.95` counts as 0.9).

The loss per item is `E[(v - y)^2] + p_unparsable * 1 + beta * KL`, with the KL to the start model (the adapter
disabled) taken exactly over the same tree (`KL_1 + P(0) KL_2 + P(0) P(.) KL_3`, full vocabulary at each node). One
forward and backward pass per item instead of one per sample: about 8x cheaper than GRPO with 8 samples and with no
sampling variance. Evaluation (the paired and the online runs) still samples the value as the model writes it.

Data and the held-out split are `train.confidence`'s (`prepare`, `heldout`); the `exact:` section of the config holds
the optimiser settings. Every `eval_steps` optimiser steps the held-out items' mean expected Brier and the AUROC of the
expected value against failure on held-out feasible claims are logged; they select nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from honest_agents.train import confidence as tc

logger = logging.getLogger(__name__)

DIGITS = tuple(str(d) for d in range(10))
UNPARSABLE_COST = 1.0  # the worst Brier, as `confidence_bandit.brier_reward` scores an unparsable value


@dataclass(frozen=True)
class TreeTokens:
    one: int
    zero: int
    dot: int
    newline: int
    digits: tuple[int, ...]

    @classmethod
    def of(cls, tokenizer: Any) -> TreeTokens:
        def single(text: str) -> int:
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise ValueError(f"{text!r} is not one token ({ids}): the value tree assumes one token per digit")
            return ids[0]

        return cls(single("1"), single("0"), single("."), single("\n"), tuple(single(d) for d in DIGITS))


def tree_probabilities(lp1: Any, lp2: Any, lp3: Any, t: TreeTokens) -> dict[str, Any]:
    """Probabilities of each resolved value from the three nodes' log-probabilities (1-D tensors over the vocabulary):
    `one` (1.0), `zero` (0.0), `tenths` (10, value d / 10) and `other` (unparsable, 1 - the rest)."""
    import torch

    p_one = lp1[t.one].exp()
    p_zero_first = lp1[t.zero].exp()
    p_zero = (lp1[t.zero] + lp2[t.newline]).exp()
    p_tenths = (lp1[t.zero] + lp2[t.dot] + lp3[list(t.digits)]).exp()
    known = p_one + p_zero + p_tenths.sum()
    return {"one": p_one, "zero": p_zero, "tenths": p_tenths, "other": torch.clamp(1.0 - known, min=0.0),
            "reach2": p_zero_first, "reach3": (lp1[t.zero] + lp2[t.dot]).exp()}  # fmt: skip


def expected_brier(probs: dict[str, Any], y: float) -> Any:
    import torch

    values = torch.arange(10, dtype=probs["tenths"].dtype, device=probs["tenths"].device) / 10
    return (probs["one"] * (1.0 - y) ** 2 + probs["zero"] * y**2
            + (probs["tenths"] * (values - y) ** 2).sum() + probs["other"] * UNPARSABLE_COST)  # fmt: skip


def expected_value(probs: dict[str, Any]) -> Any:
    """E[v] over the parsable values, renormalised (the stated confidence the model would write on average)."""
    import torch

    values = torch.arange(10, dtype=probs["tenths"].dtype, device=probs["tenths"].device) / 10
    mass = probs["one"] + probs["zero"] + probs["tenths"].sum()
    return (probs["one"] + (probs["tenths"] * values).sum()) / torch.clamp(mass, min=1e-12)


def tree_kl(lp: Sequence[Any], ref: Sequence[Any], probs: dict[str, Any]) -> Any:
    """KL(policy || reference) over the value tree: node 1, plus node 2 weighted by reaching it, plus node 3."""
    kls = [(lp_i.exp() * (lp_i - ref_i)).sum() for lp_i, ref_i in zip(lp, ref, strict=True)]
    return kls[0] + probs["reach2"] * kls[1] + probs["reach3"] * kls[2]


@dataclass
class ExactSection:
    learning_rate: float = 1.0e-5
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 8
    beta: float = 0.04
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    eval_steps: int = 50
    max_eval_items: int = 400
    report_to: str = "wandb"


def exact_section(raw: dict[str, Any]) -> ExactSection:
    allowed = {f.name for f in dataclasses.fields(ExactSection)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `exact:`; expected {sorted(allowed)}")
    s = ExactSection(**raw)
    if s.beta <= 0:
        raise ValueError("exact.beta must be > 0: the KL keeps the policy at the start model off the reward's reach")
    return s


def load_model(name: str, dtype: str) -> Any:
    """The model as TRL's trainers load it by name (the class its config declares): on GPU 0 when CUDA is there, else on
    the CPU (never `device_map="auto"`, which picks MPS on a Mac and died there loading the weights)."""
    import torch
    from trl.trainer.utils import create_model_from_path

    return create_model_from_path(name, dtype=dtype, device_map={"": 0} if torch.cuda.is_available() else None)


def _node_logprobs(model: Any, ids: list[int], n_prompt: int) -> list[Any]:
    """Log-probabilities at the tree's three nodes. They are the sequence's last three positions (`prompt, "0", "."`),
    so only those logits are computed (`logits_to_keep`): full-vocabulary logits at every position of a 15k-token
    prompt would take about 7 GB."""
    import torch

    if len(ids) != n_prompt + 2:
        raise ValueError("an item is its prompt followed by exactly the two tokens 0 and .")
    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=3)
    logits = out.logits[0, -3:].float()
    return [torch.log_softmax(logits[i], dim=-1) for i in range(3)]


def item_loss(model: Any, ids: list[int], n_prompt: int, y: int, t: TreeTokens, beta: float) -> tuple[Any, dict]:
    import torch

    with torch.no_grad(), model.disable_adapter():
        ref = _node_logprobs(model, ids, n_prompt)
    lp = _node_logprobs(model, ids, n_prompt)
    probs = tree_probabilities(*lp, t)
    brier = expected_brier(probs, float(y))
    kl = tree_kl(lp, ref, probs)
    stats = {"brier": brier.item(), "kl": kl.item(), "p_other": probs["other"].item(),
             "value": expected_value(probs).item(), "y": y}  # fmt: skip
    return brier + beta * kl, stats


def encode(rows: Sequence[dict[str, Any]], tokenizer: Any, t: TreeTokens) -> list[tuple[list[int], int, dict]]:
    out = []
    for r in rows:
        prompt = tokenizer(r["prompt"], add_special_tokens=False).input_ids
        out.append(([*prompt, t.zero, t.dot], len(prompt), r))
    return out


def _auroc_failure(values: Sequence[float], ys: Sequence[int]) -> float | None:
    from honest_agents.eval.metrics.calibration import auroc

    return auroc([-v for v in values], [y == 0 for y in ys])


def evaluate(model: Any, items: Sequence[tuple[list[int], int, dict]], t: TreeTokens) -> dict[str, Any]:
    import torch

    model.eval()
    briers, values, ys, claims = [], [], [], []
    with torch.no_grad():
        for ids, n, row in items:
            probs = tree_probabilities(*_node_logprobs(model, ids, n), t)
            briers.append(float(expected_brier(probs, float(row["y"]))))
            v = float(expected_value(probs))
            if row["cell"] == "feasible/complete":
                values.append(v)
                ys.append(int(row["y"]))
            claims.append(v)
    model.train()
    return {"eval/expected_brier": sum(briers) / max(len(briers), 1), "eval/claims_auroc": _auroc_failure(values, ys),
            "eval/mean_value": sum(claims) / max(len(claims), 1), "eval/n": len(items)}  # fmt: skip


def run(cfg: tc.ConfidenceConfig, ex: ExactSection, *, dry_run: bool = False) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from honest_agents.models import pinned_revision

    name = cfg.model.tokenizer or cfg.model.name
    tok = AutoTokenizer.from_pretrained(name, revision=pinned_revision(name))
    t = TreeTokens.of(tok)
    prepared = tc.prepare(cfg, tok)
    if {r["source"] for r in prepared.train} & {r["source"] for r in prepared.eval}:
        raise AssertionError("a source is on both sides of the held-out split")
    summary = {"config": dataclasses.asdict(cfg), "exact": dataclasses.asdict(ex), "data": tc.summarize(prepared)}
    if dry_run:
        return summary
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    from honest_agents.models import family_spec, flash_linear_attention_available, lora_target_modules

    # as `train.run`'s preflight: Qwen3.5 is about 20x slower without flash-linear-attention
    needs_fla = family_spec(cfg.model.family).needs_flash_linear_attention
    if torch.cuda.is_available() and needs_fla and not flash_linear_attention_available():
        raise RuntimeError(
            "flash-linear-attention (`fla`) is not installed; Qwen3.5 would fall back to slow torch code"
        )
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    # As TRL's trainers load a model by name (`create_model_from_path`: the class the checkpoint declares). Qwen3.5
    # declares Qwen3_5ForConditionalGeneration, whose module names (`...language_model.layers...`) are what the LoRA
    # targets, `sft.merge` and serving expect; AutoModelForCausalLM would build the text-only class and match nothing.
    model = load_model(cfg.model.name, cfg.model.dtype)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    targets = cfg.lora.target_modules
    model = get_peft_model(model, LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=lora_target_modules(cfg.model.family) if targets == "auto" else targets))  # fmt: skip
    model.enable_input_require_grads()
    train = encode(prepared.train, tok, t)
    held = encode(prepared.eval, tok, t)
    held = rng.sample(held, min(len(held), ex.max_eval_items))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=ex.learning_rate)
    total = math.ceil(len(train) * ex.num_train_epochs / ex.gradient_accumulation_steps)
    sched = get_cosine_schedule_with_warmup(opt, int(ex.warmup_ratio * total), total)
    wb = None
    if ex.report_to == "wandb":
        import wandb

        wb = wandb.init(name=cfg.run_name, config=summary, reinit=True)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log = open(out_dir / "train_log.jsonl", "a")  # noqa: SIM115
    step, acc, window = 0, 0, []
    t0 = time.time()

    def emit(record: dict[str, Any]) -> None:
        log.write(json.dumps(record) + "\n")
        log.flush()
        if wb is not None:
            wb.log(record, step=step)

    emit({"step": 0, **evaluate(model, held, t)})
    for epoch in range(ex.num_train_epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for k in order:
            ids, n, row = train[k]
            loss, stats = item_loss(model, ids, n, int(row["y"]), t, ex.beta)
            (loss / ex.gradient_accumulation_steps).backward()
            window.append(stats)
            acc += 1
            if acc % ex.gradient_accumulation_steps:
                continue
            torch.nn.utils.clip_grad_norm_(params, ex.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "epoch": acc / len(train), "lr": sched.get_last_lr()[0],
                   "train/expected_brier": sum(s["brier"] for s in window) / len(window),
                   "train/kl": sum(s["kl"] for s in window) / len(window),
                   "train/p_unparsable": sum(s["p_other"] for s in window) / len(window),
                   "elapsed_s": time.time() - t0}  # fmt: skip
            window = []
            if step % ex.eval_steps == 0:
                rec |= evaluate(model, held, t)
            emit(rec)
        model.save_pretrained(str(out_dir / f"epoch-{epoch + 1}"))
        emit({"step": step, "epoch_end": epoch + 1, **evaluate(model, held, t)})
    model.save_pretrained(str(out_dir / "final"))
    log.close()
    if wb is not None:
        wb.finish()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.train.confidence_exact", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = tc.load_config(args.config, args.overrides)
    out = run(cfg, exact_section(cfg.exact), dry_run=args.dry_run)
    json.dump(out["data"], sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
