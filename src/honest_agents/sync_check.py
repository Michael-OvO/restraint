"""Weight-sync check: does vLLM serve the policy the trainer is training? (docs/design.md section 8.)

A zero-learning-rate run cannot answer this, because the adapter delta is zero and the merged weights equal the base
weights whatever the sync does. TRL 1.14.1 merges the LoRA adapter, yields the merged parameters to vLLM's weight
transfer and un-merges inside the same generator; vLLM copies the final packed chunk only after the generator is
exhausted, so that chunk can be read after the un-merge (`trainer.clone_synced_params` is the patch). A CPU
simulation says so; nobody has seen it on a GPU. This module is the GPU check, to run once before the report-only
pilot, which is the first run that learns.

What it does, against a running generation server (`scripts/serve_vllm.sh`) and the trainer built by `train.make_trainer`
(the very object that will train, so the patched sync path is the one tested):

1. Takes a fixed set of token sequences (2k to 20k tokens): real banked trajectories if a bank holds token ids, else
   synthetic transcripts rendered with the real chat template and our tools.
2. **Base**: with the adapter at zero, compares the log-probability the server assigns to each token (`prompt_logprobs`)
   with the trainer's own. The mean absolute gap is the model's numerical floor (bf16 kernels differ).
3. **Adapter**: sets a non-zero adapter, syncs it with TRL's own `sync_weights`, and compares again.
4. Passes if the mean importance ratio exp(trainer - vLLM) is within 1% of 1, the mean absolute gap is no larger than
   the base model's (within `gap_slack`), nothing is NaN, and the adapter actually moved the trainer's log-probabilities
   (otherwise the check is vacuous and the run is refused).
5. Optionally repeats with the stock (unpatched) sync to show what the patch buys, and recommends the sync mode:
   merged with the clone patch if it passes, adapter-only sync if only that passes, float32 serving or Qwen3-4B if
   neither passes on Qwen3.5.

**Tiny-adapter mode** (`--tiny-adapter`). The default adapter (std 0.01) is 20-60x larger than
what the first optimizer steps produce, so a sync that loses small deltas still passes it. A bf16 merge adds the delta
to a weight that is already bf16: a delta below half the weight's spacing (about 3e-5 for a weight of 0.02) rounds
away entirely. Tiny mode sets B to std 1e-5, the size of one Adam step at the training learning rate, and asks what
fraction of the delta survives the merge. That is decided on the weights, by simulating the merge in the trainer's
dtype and storing it in the served dtype (`weight_merge_survival`, exact, no server): it must be at least
`MIN_DELTA_SURVIVAL`. The served log-probabilities cannot decide it: once any weight moves, bf16 activations round
differently and the change of the served log-probabilities is dominated by that rounding noise (a projection of it
onto the trainer's change came out at 1.2 when 4% of the delta survived), so they are only required to move at all
(complete loss gives exactly zero change) and `served_delta_survival` is reported as information.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("honest_agents.sync_check")

RATIO_TOLERANCE = 0.01  # mean importance ratio within 1% of 1 (docs/design.md section 8)
DEFAULT_GAP_SLACK = 1.10  # the adapter gap may exceed the base gap by this factor (numerical noise on both sides)
GAP_ABS_TOLERANCE = 1e-4  # ... plus this much, so a base gap of exactly 0 (float32 on a CPU) is not an impossible bar
MIN_EFFECT_OVER_FLOOR = 5.0  # the adapter must move trainer log-probs by this multiple of the numerical floor
MIN_TOKENS, MAX_TOKENS = 2_000, 20_000
TINY_ADAPTER_SCALE = 1e-5  # std of LoRA B in tiny mode: one Adam step at learning rate 1e-5 moves each entry by ~1e-5
MIN_DELTA_SURVIVAL = 0.5  # tiny mode: at least this fraction of the delta must survive the merge (weights and server)

HttpFn = Callable[[str, str, dict | None], dict]


@dataclass
class SyncStats:
    """Agreement between the trainer's and the server's log-probabilities on the same tokens."""

    n_tokens: int
    mean_ratio: float  # mean of exp(trainer_logp - vllm_logp): the importance ratio TRL applies
    mean_gap: float  # mean of (trainer_logp - vllm_logp), signed
    mean_abs_gap: float
    max_abs_gap: float
    n_nan: int  # positions where either side gave a non-finite value (excluded from the other statistics)


def compare_logprobs(trainer: Sequence[float | None], vllm: Sequence[float | None]) -> SyncStats:
    """Statistics over aligned per-token log-probabilities. Non-finite or missing values count as NaN, not as 0."""
    if len(trainer) != len(vllm):
        raise ValueError(f"{len(trainer)} trainer log-probs but {len(vllm)} from the server: sequences are misaligned")
    gaps, nan = [], 0
    for t, v in zip(trainer, vllm, strict=True):
        if t is None or v is None or not math.isfinite(t) or not math.isfinite(v):
            nan += 1
            continue
        gaps.append(t - v)
    if not gaps:
        return SyncStats(0, math.nan, math.nan, math.nan, math.nan, nan)
    n = len(gaps)
    return SyncStats(
        n_tokens=n,
        mean_ratio=sum(math.exp(max(min(g, 50.0), -50.0)) for g in gaps) / n,
        mean_gap=sum(gaps) / n,
        mean_abs_gap=sum(abs(g) for g in gaps) / n,
        max_abs_gap=max(abs(g) for g in gaps),
        n_nan=nan,
    )


def effect_size(before: Sequence[float | None], after: Sequence[float | None]) -> float:
    """Mean absolute change of the trainer's own log-probabilities when the adapter was switched on."""
    diffs = [abs(a - b) for a, b in zip(after, before, strict=True) if a is not None and b is not None]
    return sum(diffs) / len(diffs) if diffs else math.nan


def verdict(
    adapter: SyncStats,
    base: SyncStats,
    *,
    effect: float,
    ratio_tolerance: float = RATIO_TOLERANCE,
    gap_slack: float = DEFAULT_GAP_SLACK,
    gap_abs_tolerance: float = GAP_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Pass/fail of one sync mode. Every criterion is reported so a failure says which one."""
    floor = base.mean_abs_gap
    checks = {
        "no_nan": adapter.n_nan == 0 and base.n_nan == 0,
        "mean_ratio_within_tolerance": math.isfinite(adapter.mean_ratio)
        and abs(adapter.mean_ratio - 1.0) <= ratio_tolerance,
        "gap_not_larger_than_base": math.isfinite(adapter.mean_abs_gap)
        and adapter.mean_abs_gap <= floor * gap_slack + gap_abs_tolerance,
        "adapter_moves_logprobs": math.isfinite(effect)
        and effect >= MIN_EFFECT_OVER_FLOOR * max(floor, gap_abs_tolerance),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "adapter_effect": effect,
        "base_mean_abs_gap": floor,
        "adapter_mean_abs_gap": adapter.mean_abs_gap,
        "adapter_mean_ratio": adapter.mean_ratio,
    }


def delta_survival(
    trainer_before: Sequence[float | None],
    trainer_after: Sequence[float | None],
    vllm_before: Sequence[float | None],
    vllm_after: Sequence[float | None],
) -> float:
    """Fraction of the trainer's change in log-probabilities that the server reproduces after the sync (information).

    Noise-dominated for a delta below the bf16 spacing (see the module docstring), so it is not a pass criterion. The
    projection coefficient sum(dv * dt) / sum(dt^2) of the server's per-token change `dv` onto the trainer's `dt`:
    1 when the server moved exactly as the trainer did, 0 when it did not move (the delta rounded away in the merge).
    Each side is compared with itself before and after, so its own numerical noise cancels.
    """
    num = den = 0.0
    for tb, ta, vb, va in zip(trainer_before, trainer_after, vllm_before, vllm_after, strict=True):
        if None in (tb, ta, vb, va):
            continue
        dt, dv = ta - tb, va - vb
        if math.isfinite(dt) and math.isfinite(dv):
            num += dv * dt
            den += dt * dt
    return num / den if den > 0 else math.nan


def tiny_verdict(
    *,
    weight_survival: float,
    served_moved: bool,
    effect: float,
    n_nan: int,
    served_survival: float = math.nan,
    threshold: float = MIN_DELTA_SURVIVAL,
) -> dict[str, Any]:
    """Pass/fail of tiny-adapter mode; every criterion is reported. `served_survival` is information, not a criterion."""
    checks = {
        "no_nan": n_nan == 0,
        "adapter_moves_logprobs": math.isfinite(effect) and effect > 0.0,
        "delta_survives_merge_in_weights": math.isfinite(weight_survival) and weight_survival >= threshold,
        "served_logprobs_moved": bool(served_moved),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "adapter_effect": effect,
        "weight_delta_survival": weight_survival,
        "served_delta_survival": served_survival,
        "threshold": threshold,
    }


def recommend(results: dict[str, dict[str, Any]], family: str) -> str:
    """Which sync mode to use, from the verdicts of the modes that were run (docs/design.md section 8)."""
    if results.get("clone", {}).get("passed"):
        return "merged sync with the clone patch (HonestGRPOTrainer default)"
    if results.get("stock", {}).get("passed"):
        return "stock merged sync passes but the clone patch does not: investigate before training"
    if results.get("adapter_only", {}).get("passed"):
        return "adapter-only sync (versioned adapter names, never load_inplace)"
    fallback = "float32 serving (and trainer weights) or Qwen3-4B" if family == "qwen3_5" else "float32 serving"
    return f"no mode passes: use {fallback}, or raise the LoRA learning rate so the merged delta survives bf16"


# ── sequences ────────────────────────────────────────────────────────────────

_WORDS = [
    "config",
    "build",
    "test",
    "file",
    "output",
    "error",
    "warning",
    "path",
    "module",
    "import",
    "version",
    "install",
    "package",
    "user",
    "group",
    "read",
    "write",
    "line",
    "total",
    "size",
    "owner",
    "mode",
    "date",
    "index",
    "main",
    "src",
    "lib",
    "bin",
    "tmp",
    "data",
    "log",
    "run",
    "make",
    "check",
    "result",
    "value",
    "item",
    "name",
    "list",
]


def _filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def synthetic_messages(rng: random.Random, tool_chars: int = 1500) -> list[dict]:
    """One episode in our harness (system, instruction, bash call and output, bash call and output, report)."""
    from honest_agents.models import tool_flow_messages

    msgs = tool_flow_messages(instruction=f"Fix the build in /app. {_filler(rng, 40)}")
    out = []
    for m in msgs:
        m = dict(m)
        if m["role"] == "tool" and m.get("name") == "bash":
            lines = [
                f"-rw-r--r-- 1 root root {rng.randint(1, 9999)} {_filler(rng, 3)}.txt" for _ in range(tool_chars // 40)
            ]
            m["content"] = "[exit code: 0]\n" + "\n".join(lines)
        out.append(m)
    return out


def build_sequences(
    processing_class: Any,
    *,
    tools: Sequence[Any] | None,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    target_tokens: int = 8_000,
    max_sequence_tokens: int = 4_096,
    seed: int = 0,
) -> list[list[int]]:
    """Fixed token sequences, rendered with the real chat template and tools, until `target_tokens` are reached."""
    if not MIN_TOKENS <= target_tokens <= MAX_TOKENS:
        raise ValueError(f"target_tokens must be within [{MIN_TOKENS}, {MAX_TOKENS}], got {target_tokens}")
    rng, sequences, total = random.Random(seed), [], 0
    for _ in range(1000):
        if total >= target_tokens:
            break
        ids = processing_class.apply_chat_template(
            synthetic_messages(rng),
            tools=list(tools) if tools else None,
            chat_template=chat_template,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=False,
            **(chat_template_kwargs or {}),
        )
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = list(ids)[:max_sequence_tokens]
        sequences.append(ids)
        total += len(ids) - 1
    return sequences


def sequences_from_bank(bank_path: str | Path, target_tokens: int, max_sequence_tokens: int = 4_096) -> list[list[int]]:
    """Real trajectories: prompt + completion token ids of banked rows (Phase 0), when the bank carries them."""
    from honest_agents.bank import iter_bank

    out, total = [], 0
    for row in iter_bank(bank_path):
        if total >= target_tokens:
            break
        if row.prompt_ids and row.completion_ids:
            ids = [*row.prompt_ids, *row.completion_ids][:max_sequence_tokens]
            out.append(ids)
            total += len(ids) - 1
    return out


# ── the two sides ────────────────────────────────────────────────────────────


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def vllm_logprobs(
    base_url: str, sequences: Sequence[Sequence[int]], *, http: HttpFn | None = None, model: str | None = None
) -> list[list[float | None]]:
    """The log-probability the server assigns to every token after the first, per sequence (`prompt_logprobs`).

    `prompt_logprobs` is computed from the raw logits at temperature 1 (no sampling parameter touches it), the same
    quantity the trainer's forward pass produces. Prefix caching is skipped for such requests by vLLM itself.
    """
    http = http or (lambda m, u, p: _http_json(m, u, p))
    base = base_url.rstrip("/")
    name = model or http("GET", f"{base}/v1/models", None)["data"][0]["id"]
    out: list[list[float | None]] = []
    for ids in sequences:
        payload = {
            "model": name,
            "prompt": list(ids),
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 0,
            "return_tokens_as_token_ids": True,
        }
        choice = http("POST", f"{base}/v1/completions", payload)["choices"][0]
        entries = choice.get("prompt_logprobs")
        if not entries or len(entries) != len(ids):
            raise RuntimeError(
                f"server returned {len(entries or [])} prompt_logprobs for {len(ids)} prompt tokens; is "
                "`--max-logprobs -1` set and is this a vLLM >= 0.30 server?"
            )
        lps: list[float | None] = []
        for token_id, entry in zip(ids[1:], entries[1:], strict=True):
            hit = (entry or {}).get(str(token_id))
            lps.append(None if hit is None else hit.get("logprob"))
        out.append(lps)
    return out


def trainer_logprobs(model: Any, sequences: Sequence[Sequence[int]], *, temperature: float = 1.0) -> list[list[float]]:
    """The trainer's log-probability of every token after the first, one forward pass per sequence, no gradient."""
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: list[list[float]] = []
    try:
        with torch.no_grad():
            for ids in sequences:
                x = torch.tensor([list(ids)], device=device)
                logits = model(input_ids=x).logits[:, :-1].float() / temperature
                lp = torch.log_softmax(logits, dim=-1).gather(-1, x[:, 1:, None]).squeeze(-1)
                out.append(lp[0].tolist())
                del logits, lp
    finally:
        model.train(was_training)
    return out


def set_adapter_scale(model: Any, scale: float, seed: int = 0) -> int:
    """Give every LoRA B matrix small random values (std `scale`), or zero them when `scale` is 0. Returns the count.

    PEFT initialises B to zero, so a fresh adapter changes nothing; this stands in for "an adapter after some steps".
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    n = 0
    with torch.no_grad():
        for module in model.modules():
            lora_b = getattr(module, "lora_B", None)
            if lora_b is None:
                continue
            for layer in lora_b.values():
                w = layer.weight
                if scale == 0:
                    w.zero_()
                else:
                    w.copy_((torch.randn(w.shape, generator=generator) * scale).to(device=w.device, dtype=w.dtype))
                n += 1
    return n


def lora_deltas(model: Any):
    """`(base weight, exact delta)` for every LoRA module and adapter: delta = scaling * B @ A, in float32."""
    import torch

    for module in model.modules():
        if getattr(module, "lora_A", None) is None or not hasattr(module, "get_base_layer"):
            continue
        weight = module.get_base_layer().weight
        for name in module.lora_A:
            with torch.no_grad():
                delta = module.scaling[name] * (module.lora_B[name].weight.float() @ module.lora_A[name].weight.float())
            yield weight.detach(), delta.to(weight.device)


def weight_merge_survival(model: Any, served_dtype: Any = None) -> dict[str, Any]:
    """What a merge keeps of the LoRA delta in the base weights' own dtype (no server involved).

    Merges the way PEFT does (`weight += delta`, stored in the weight's dtype) on a copy of each weight, then stores the
    result in `served_dtype` (the server's dtype; default: the weights' own) and projects the realised change onto the
    exact delta: 1 = all of it survived, 0 = the merge changed nothing. Weights that stay exactly as they were are
    counted in `unchanged_modules`.
    """
    num = den = 0.0
    modules = unchanged = 0
    for weight, delta in lora_deltas(model):
        merged = weight.clone()
        merged += delta
        served = served_dtype or weight.dtype
        realised = merged.to(served).float() - weight.to(served).float()
        num += float((realised * delta).sum())
        den += float((delta * delta).sum())
        modules += 1
        unchanged += bool(not realised.any())
    return {
        "fraction": num / den if den > 0 else math.nan,
        "n_modules": modules,
        "unchanged_modules": unchanged,
        "dtype": str(next(model.parameters()).dtype),
    }


def _use_stock_sync(vllm_generation: Any) -> None:
    """Undo `clone_synced_params`: the instance attribute it sets shadows the class method."""
    vllm_generation.__dict__.pop("_iter_named_params", None)


def run_check(
    trainer: Any,
    *,
    base_url: str,
    sequences: Sequence[Sequence[int]],
    modes: Sequence[str] = ("clone",),
    adapter_scale: float = 0.01,
    gap_slack: float = DEFAULT_GAP_SLACK,
    family: str = "qwen3",
    http: HttpFn | None = None,
    tiny_adapter: bool = False,
    served_dtype: Any = None,
) -> dict[str, Any]:
    """Run the check on a built trainer. UNVERIFIED without a GPU and a server (see the module docstring).

    Modes: `clone` is the sync `HonestGRPOTrainer` ships; `stock` is TRL's own (to see what the patch buys).
    `tiny_adapter` swaps the verdict for the tiny-adapter one (the adapter is set to `TINY_ADAPTER_SCALE` unless
    `adapter_scale` is given smaller); `served_dtype` is the dtype the server stores weights in (tiny mode only).
    """
    from honest_agents.trainer import clone_synced_params

    model, vg = trainer.model, trainer.vllm_generation
    temperature = float(getattr(trainer, "temperature", 1.0))
    n_adapters = set_adapter_scale(model, 0.0)
    if not n_adapters:
        raise RuntimeError("the model has no LoRA adapter: nothing is synced as a delta, the check would be vacuous")
    # Base: the server as started (checkpoint weights, no sync yet) against the trainer with a zero adapter.
    base_trainer = trainer_logprobs(model, sequences, temperature=temperature)
    base_vllm = vllm_logprobs(base_url, sequences, http=http)
    base = compare_logprobs(_flat(base_trainer), _flat(base_vllm))
    results: dict[str, Any] = {
        "n_tokens": base.n_tokens,
        "n_lora_modules": n_adapters,
        "base": asdict(base),
        "modes": {},
    }
    if tiny_adapter:
        adapter_scale = min(adapter_scale, TINY_ADAPTER_SCALE)
    results["adapter_scale"] = adapter_scale
    set_adapter_scale(model, adapter_scale)
    results["weight_merge"] = weight_merge_survival(model, served_dtype)
    moved_trainer = trainer_logprobs(model, sequences, temperature=temperature)
    effect = effect_size(_flat(base_trainer), _flat(moved_trainer))
    for mode in modes:
        if mode == "clone":
            clone_synced_params(vg)
        elif mode == "stock":
            _use_stock_sync(vg)
        else:
            raise ValueError(f"unknown sync mode {mode!r}; supported: clone, stock")
        vg.sync_weights()
        served = vllm_logprobs(base_url, sequences, http=http)
        stats = compare_logprobs(_flat(moved_trainer), _flat(served))
        if tiny_adapter:
            moved = effect_size(_flat(base_vllm), _flat(served))
            judged = tiny_verdict(
                weight_survival=results["weight_merge"]["fraction"],
                served_moved=math.isfinite(moved) and moved > 0.0,
                effect=effect,
                n_nan=stats.n_nan + base.n_nan,
                served_survival=delta_survival(
                    _flat(base_trainer), _flat(moved_trainer), _flat(base_vllm), _flat(served)
                ),
            )
        else:
            judged = verdict(stats, base, effect=effect, gap_slack=gap_slack)
        results["modes"][mode] = {"stats": asdict(stats), **judged}
    results["recommendation"] = recommend(results["modes"], family)
    results["passed"] = bool(results["modes"].get("clone", {}).get("passed"))
    return results


def _flat(per_sequence: Sequence[Sequence[float | None]]) -> list[float | None]:
    return [v for seq in per_sequence for v in seq]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="honest_agents.sync_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the run config whose trainer and server are checked")
    ap.add_argument("--tokens", type=int, default=8_000, help=f"tokens to compare ({MIN_TOKENS}..{MAX_TOKENS})")
    ap.add_argument("--adapter-scale", type=float, default=0.01, help="std of the random LoRA B values")
    ap.add_argument(
        "--served-dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="dtype the server stores weights in (scripts/serve_vllm.sh: bfloat16); tiny-adapter mode only",
    )
    ap.add_argument(
        "--tiny-adapter",
        action="store_true",
        help=f"std {TINY_ADAPTER_SCALE:g} (one learning step) and ask what fraction of the delta survives the merge",
    )
    ap.add_argument("--gap-slack", type=float, default=DEFAULT_GAP_SLACK)
    ap.add_argument("--bank", default=None, help="bank directory: use its token ids instead of synthetic transcripts")
    ap.add_argument("--compare-stock", action="store_true", help="also run TRL's unpatched sync")
    ap.add_argument("--output", default=None, help="write the JSON result here")
    ap.add_argument("overrides", nargs="*", help="dotted config overrides (the same ones the run gets)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    from honest_agents import train

    cfg = train.load_run_config(args.config, args.overrides)
    plan = train.build_plan(cfg)
    trainer = train.make_trainer(plan)
    url = train._server_url(cfg).replace("0.0.0.0", "127.0.0.1")
    sequences = sequences_from_bank(args.bank, args.tokens) if args.bank else []
    if not sequences:
        sequences = build_sequences(
            trainer.processing_class,
            tools=getattr(trainer, "tools", None),
            chat_template=getattr(trainer, "chat_template", None),
            chat_template_kwargs=getattr(trainer, "chat_template_kwargs", None),
            target_tokens=args.tokens,
        )
    result = run_check(
        trainer,
        base_url=url,
        sequences=sequences,
        modes=("clone", "stock") if args.compare_stock else ("clone",),
        adapter_scale=args.adapter_scale,
        gap_slack=args.gap_slack,
        family=str(plan.family),
        tiny_adapter=args.tiny_adapter,
        served_dtype=getattr(torch, args.served_dtype),
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
