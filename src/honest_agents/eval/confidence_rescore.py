"""Paired confidence scoring (iteration 17, docs/iteration-17-design.md section 5): a served model states the
confidence of archived episodes, on exactly the training prompt, so two models are compared on the same claims.

    uv run --extra eval python -m honest_agents.eval.confidence_rescore --bank DIR [--bank DIR ...] \\
        --out runs/it17/rescore_<arm>.jsonl [--base-url http://localhost:8000/v1] [--model policy] \\
        [--tokenizer Qwen/Qwen3.5-4B] [--samples 8] [--temperature 1.0] [--concurrency 16]

Every bank row that `sft.confidence_bandit.item_of` accepts (no source exclusion here: the select split is scored,
never trained on) is rendered with the served template and cut before its confidence value
(`confidence_bandit.render_prefix`) and sent to the server's completions endpoint with `n = --samples` and
`max_tokens` 8; each completion's first line is parsed (`parse_confidence`). The server's prompt token count must
equal the local tokenizer's (`PromptMismatchError` otherwise): a different tokenisation would score another prompt.
A row per item: its id, source, task, family, feasibility, status, label y, the confidence the model wrote in the
episode, the sampled values (None = unparsable) and their mean over the parsable ones. Rows are appended and an item
already in `--out` is skipped, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.sft import confidence_bandit as cb

MAX_TOKENS = 8


class PromptMismatchError(RuntimeError):
    pass


def items_of(banks: Sequence[str | Path]) -> tuple[list[cb.Item], dict[str, int]]:
    items: list[cb.Item] = []
    dropped: dict[str, int] = {}
    seen: set[str] = set()
    for path in banks:
        for row in bank_mod.iter_bank(Path(path).expanduser()):
            got = cb.item_of(row, set())
            if isinstance(got, str):
                dropped[got] = dropped.get(got, 0) + 1
            elif got.id not in seen:
                seen.add(got.id)
                items.append(got)
    return items, dropped


def row_of(item: cb.Item, texts: Sequence[str]) -> dict[str, Any]:
    values = [cb.parse_confidence(t) for t in texts]
    ok = [v for v in values if v is not None]
    return {
        "id": item.id, "source": item.source, "task": item.task, "family": item.family, "feasible": item.feasible,
        "is_control": item.is_control, "status": item.status, "y": item.y,
        "original_confidence": item.original_confidence, "samples": values,
        "mean": sum(ok) / len(ok) if ok else None, "parsed": len(ok),
    }  # fmt: skip


def append_row(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


async def _score(items: Sequence[cb.Item], args: argparse.Namespace, out: Path) -> int:
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec, pinned_revision

    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=pinned_revision(args.tokenizer))
    kwargs = dict(family_spec(args.family).chat_template_kwargs)
    tools = tool_schemas()
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(item: cb.Item) -> None:
        nonlocal done
        prompt = cb.render_prefix(item.messages, tok, tools, kwargs)
        n_local = len(tok(prompt, add_special_tokens=False).input_ids)
        async with sem:
            res = await client.completions.create(model=args.model, prompt=prompt, n=args.samples,
                                                  max_tokens=MAX_TOKENS, temperature=args.temperature, seed=args.seed)  # fmt: skip
        if res.usage is not None and res.usage.prompt_tokens != n_local:
            raise PromptMismatchError(f"{item.id}: server {res.usage.prompt_tokens} prompt tokens, local {n_local}")
        row = row_of(item, [c.text for c in sorted(res.choices, key=lambda c: c.index)])
        async with lock:
            append_row(out, row)
            done += 1

    await asyncio.gather(*(one(i) for i in items))
    return done


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.eval.confidence_rescore", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--bank", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="policy")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--family", default="qwen3_5")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args(argv)
    items, dropped = items_of(args.bank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if args.out.is_file():
        have = {json.loads(line)["id"] for line in args.out.read_text().splitlines() if line.strip()}
    todo = [i for i in items if i.id not in have]
    print(f"{len(items)} items ({len(have)} already scored), dropped {dropped}", file=sys.stderr)
    n = asyncio.run(_score(todo, args, args.out))
    print(f"scored {n} items into {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
