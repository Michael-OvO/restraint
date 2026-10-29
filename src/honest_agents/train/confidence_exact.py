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

**Action anchor** (iteration 18, `exact.anchor_beta`): iteration 17's adapter, trained on the value alone with a KL on
the value's tree only, eroded the start model's actions through the shared weights (blind deletions 4 -> 30 of 120
high_risk twins). The anchor adds `anchor_beta` x the KL from the start model to the policy over the start model's own
action tokens in each transcript (every assistant turn and the open report turn: `assistant_positions`), estimated with
k3 on up to `anchor_max_tokens` sampled positions and scaled to the transcript's action-token count. The drift monitor
logs that KL on the held-out transcripts at every evaluation, whatever `anchor_beta`.

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
    # The action anchor (iteration 18): the KL from the start model to the policy over the start model's own action
    # tokens in each transcript (every assistant turn, the open report turn up to the cut included), estimated on up
    # to `anchor_max_tokens` sampled positions. 0 switches it off (iteration 17's run, which let the actions drift).
    anchor_beta: float = 0.0
    anchor_max_tokens: int = 512
    # Stop the run (an error: the chain fails before any evaluation) when the held-out action KL per token exceeds this
    # at an evaluation; None = never. Iteration 17's adapter: 0.18 per token (134 nats per transcript, 20 items).
    abort_action_kl_per_token: float | None = None


def exact_section(raw: dict[str, Any]) -> ExactSection:
    allowed = {f.name for f in dataclasses.fields(ExactSection)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `exact:`; expected {sorted(allowed)}")
    s = ExactSection(**raw)
    if s.beta <= 0:
        raise ValueError("exact.beta must be > 0: the KL keeps the value's distribution at the start model's")
    if s.abort_action_kl_per_token is not None and s.abort_action_kl_per_token <= 0:
        raise ValueError("exact.abort_action_kl_per_token must be > 0 (or null)")
    if s.anchor_beta < 0 or s.anchor_max_tokens < 1:
        raise ValueError("exact.anchor_beta must be >= 0 and exact.anchor_max_tokens >= 1")
    return s


def load_model(name: str, dtype: str) -> Any:
    """The model as TRL's trainers load it by name (the class its config declares): on GPU 0 when CUDA is there, else on
    the CPU (never `device_map="auto"`, which picks MPS on a Mac and died there loading the weights)."""
    import torch
    from trl.trainer.utils import create_model_from_path

    return create_model_from_path(name, dtype=dtype, device_map={"": 0} if torch.cuda.is_available() else None)


@dataclass(frozen=True)
class ChatMarkers:
    """ChatML's turn markers: an assistant turn is the tokens after `<|im_start|>assistant\\n` up to and including its
    `<|im_end|>` (the end-of-turn decision is the model's too)."""

    im_start: int
    im_end: int
    header: tuple[int, ...]  # the tokens of "assistant\\n" after <|im_start|>

    @classmethod
    def of(cls, tokenizer: Any) -> ChatMarkers:
        start, end = tokenizer.convert_tokens_to_ids("<|im_start|>"), tokenizer.convert_tokens_to_ids("<|im_end|>")
        if start is None or end is None or start == tokenizer.unk_token_id or end == tokenizer.unk_token_id:
            raise ValueError("the tokenizer has no ChatML markers: the action anchor cannot find the assistant turns")
        return cls(start, end, tuple(tokenizer("assistant\n", add_special_tokens=False).input_ids))


def assistant_positions(ids: Sequence[int], m: ChatMarkers) -> list[int]:
    """Indices of every token of an assistant turn in `ids` (the last turn may be open: it runs to the end)."""
    out: list[int] = []
    h = len(m.header)
    i = 0
    while i < len(ids):
        if ids[i] == m.im_start and tuple(ids[i + 1 : i + 1 + h]) == m.header:
            j = i + 1 + h
            while j < len(ids) and ids[j] != m.im_end:
                out.append(j)
                j += 1
            if j < len(ids):
                out.append(j)  # the turn's own <|im_end|>
            i = j + 1
        else:
            i += 1
    return out


def sample_positions(positions: Sequence[int], k: int, key: str) -> list[int]:
    """At most `k` of `positions`, uniformly, with a stable seed per item (the same positions in every epoch)."""
    if len(positions) <= k:
        return list(positions)
    return sorted(random.Random(f"anchor|{key}").sample(list(positions), k))


def _forward(model: Any, ids: list[int], n_prompt: int, actions: Sequence[int]) -> tuple[list[Any], Any]:
    """One pass: the log-probabilities at the value tree's three nodes, and of each action token at its own position.
    Logits are computed only at those positions (`logits_to_keep` with an index tensor: identical to a full pass,
    checked on Qwen3.5); full-vocabulary logits at every position of a 15k-token prompt would take about 7 GB."""
    import torch

    if len(ids) != n_prompt + 2:
        raise ValueError("an item is its prompt followed by exactly the two tokens 0 and .")
    keep = [a - 1 for a in actions] + [n_prompt - 1, n_prompt, n_prompt + 1]
    idx = torch.tensor(keep, device=model.device)
    out = model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=idx)
    logits = out.logits[0].float()
    nodes = [torch.log_softmax(logits[-3 + i], dim=-1) for i in range(3)]
    if not actions:
        return nodes, logits.new_zeros(0)
    act = torch.log_softmax(logits[: len(actions)], dim=-1)
    targets = torch.tensor([ids[a] for a in actions], device=model.device)
    return nodes, act.gather(1, targets[:, None])[:, 0]


def _node_logprobs(model: Any, ids: list[int], n_prompt: int) -> list[Any]:
    return _forward(model, ids, n_prompt, ())[0]


def action_kl(lp_actions: Any, ref_actions: Any, n_total: int) -> Any:
    """KL(start model || policy) over the transcript's action tokens, from the start model's own samples: the mean of
    the non-negative unbiased estimator k3 = (r - 1) - log r, r = p_policy / p_start, times the number of action
    tokens in the transcript (`n_total`, not only the sampled ones)."""
    if lp_actions.numel() == 0:
        return lp_actions.sum()
    log_r = lp_actions - ref_actions
    return (log_r.exp() - 1 - log_r).mean() * n_total


def item_loss(model: Any, item: Item, t: TreeTokens, ex: ExactSection) -> tuple[Any, dict]:
    import torch

    ids, n_prompt, row, actions, n_actions = item
    actions = actions if ex.anchor_beta > 0 else ()
    with torch.no_grad(), model.disable_adapter():
        ref_nodes, ref_act = _forward(model, ids, n_prompt, actions)
    nodes, act = _forward(model, ids, n_prompt, actions)
    probs = tree_probabilities(*nodes, t)
    brier = expected_brier(probs, float(row["y"]))
    kl = tree_kl(nodes, ref_nodes, probs)
    akl = action_kl(act, ref_act, n_actions)
    stats = {"brier": brier.item(), "kl": kl.item(), "p_other": probs["other"].item(), "action_kl": akl.item(),
             "value": expected_value(probs).item(), "y": int(row["y"])}  # fmt: skip
    return brier + ex.beta * kl + ex.anchor_beta * akl, stats


Item = tuple[list[int], int, dict, list[int], int]  # ids, prompt length, row, sampled action positions, action tokens


def encode(rows: Sequence[dict[str, Any]], tokenizer: Any, t: TreeTokens, k: int = 512) -> list[Item]:
    m = ChatMarkers.of(tokenizer)
    out = []
    for r in rows:
        prompt = tokenizer(r["prompt"], add_special_tokens=False).input_ids
        positions = assistant_positions(prompt, m)
        out.append(
            ([*prompt, t.zero, t.dot], len(prompt), r, sample_positions(positions, k, r["item_id"]), len(positions))
        )
    return out


def _auroc_failure(values: Sequence[float], ys: Sequence[int]) -> float | None:
    from honest_agents.eval.metrics.calibration import auroc

    return auroc([-v for v in values], [y == 0 for y in ys])


def evaluate(model: Any, items: Sequence[Item], t: TreeTokens) -> dict[str, Any]:
    """Held-out expected Brier, the claims AUROC of the expected value, and the drift monitor: the action KL from the
    start model over the held-out transcripts' own actions (per token and per transcript), whatever `anchor_beta`."""
    import torch

    model.eval()
    briers, values, ys, claims, akl, akl_tok = [], [], [], [], [], []
    with torch.no_grad():
        for ids, n, row, actions, n_actions in items:
            nodes, act = _forward(model, ids, n, actions)
            with model.disable_adapter():
                _, ref_act = _forward(model, ids, n, actions)
            if actions:
                kl = action_kl(act, ref_act, n_actions)
                akl.append(float(kl))
                akl_tok.append(float(kl) / n_actions)
            probs = tree_probabilities(*nodes, t)
            briers.append(float(expected_brier(probs, float(row["y"]))))
            v = float(expected_value(probs))
            if row["cell"] == "feasible/complete":
                values.append(v)
                ys.append(int(row["y"]))
            claims.append(v)
    model.train()
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"eval/expected_brier": mean(briers), "eval/claims_auroc": _auroc_failure(values, ys),
            "eval/mean_value": mean(claims), "eval/n": len(items),
            "eval/action_kl": mean(akl), "eval/action_kl_per_token": mean(akl_tok),
            "eval/action_kl_max": max(akl) if akl else None}  # fmt: skip


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
    train = encode(prepared.train, tok, t, ex.anchor_max_tokens)
    held = encode(prepared.eval, tok, t, ex.anchor_max_tokens)
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

    class DriftAbort(RuntimeError):
        pass

    def check_drift(rec: dict[str, Any]) -> None:
        limit, got = ex.abort_action_kl_per_token, rec.get("eval/action_kl_per_token")
        if limit is not None and got is not None and got > limit:
            log.write(json.dumps({"step": step, "abort": f"action KL per token {got:.4f} > {limit}"}) + "\n")
            log.flush()
            raise DriftAbort(f"held-out action KL per token {got:.4f} exceeds {limit} at step {step}: stopped")

    first = {"step": 0, **evaluate(model, held, t)}
    emit(first)
    for epoch in range(ex.num_train_epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for k in order:
            loss, stats = item_loss(model, train[k], t, ex)
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
                   "train/action_kl": sum(s["action_kl"] for s in window) / len(window),
                   "elapsed_s": time.time() - t0}  # fmt: skip
            window = []
            if step % ex.eval_steps == 0:
                rec |= evaluate(model, held, t)
            emit(rec)
            check_drift(rec)
        model.save_pretrained(str(out_dir / f"epoch-{epoch + 1}"))
        end = {"step": step, "epoch_end": epoch + 1, **evaluate(model, held, t)}
        emit(end)
        check_drift(end)
    model.save_pretrained(str(out_dir / "final"))
    log.close()
    if wb is not None:
        wb.finish()
    return summary


def drift(cfg: tc.ConfidenceConfig, adapter: str, n_items: int, anchor_max_tokens: int = 512) -> dict[str, Any]:
    """An adapter's held-out metrics against the start model (`cfg.model.name`), with no training: the drift monitor's
    action KL and the value's expected Brier and claims AUROC, on `n_items` held-out items (stable sample)."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    from honest_agents.models import pinned_revision

    name = cfg.model.tokenizer or cfg.model.name
    tok = AutoTokenizer.from_pretrained(name, revision=pinned_revision(name))
    t = TreeTokens.of(tok)
    held = encode(tc.prepare(cfg, tok).eval, tok, t, anchor_max_tokens)
    held = random.Random(cfg.seed).sample(held, min(len(held), n_items))
    model = PeftModel.from_pretrained(load_model(cfg.model.name, cfg.model.dtype), adapter)
    return {"adapter": adapter, **evaluate(model, held, t)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.train.confidence_exact", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--drift", metavar="ADAPTER", help="measure this adapter's held-out drift instead of training")
    ap.add_argument("--drift-items", type=int, default=40)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = tc.load_config(args.config, args.overrides)
    if args.drift:
        print(json.dumps(drift(cfg, args.drift, args.drift_items), indent=2))
        return 0
    out = run(cfg, exact_section(cfg.exact), dry_run=args.dry_run)
    json.dump(out["data"], sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
