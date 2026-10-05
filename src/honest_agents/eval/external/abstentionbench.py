"""AbstentionBench (arXiv 2506.09038, facebookresearch/AbstentionBench) with our own inference + their LLM judge.

Their harness pins vllm==0.6.4.post1 (cannot serve Qwen3 / Qwen3.5) and the HF dataset is a loading *script* that only
`datasets<=3.6.0` executes, so the pipeline is split:

1. `abstentionbench_export_standalone.py` (separate py3.11 venv with datasets==3.6.0) → `abstentionbench.jsonl` with
   rows {dataset, index, question, reference_answers, should_abstain, metadata_json}.
2. `run`: answer every question with the policy over an OpenAI-compatible endpoint (single turn, no tools; no system
   prompt by default, as in their `use_system_prompt=False`), then classify each answer with their CoCoNot-style
   abstention judge prompt (`LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT`, read at runtime from a checkout of their repo
   — the repo is CC BY-NC 4.0, so we do not vendor it) on a judge endpoint. Their default judge is
   Llama-3.1-8B-Instruct at temperature 0, max_tokens 100.
3. `score`: abstention precision / recall / F1 with should_abstain as the positive class, per dataset and macro over
   datasets (their `analysis/tables.py`).

Purpose in this project: detect regressions in non-agentic QA abstention (and over-abstention) after RL.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROMPT_NAME = "LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT"


def load_judge_prompt(path: str | Path, name: str = PROMPT_NAME) -> str:
    """Read a module-level string constant from their `recipe/evaluation_judge_prompts.py` without importing it."""
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def format_judge_prompt(template: str, row: dict[str, Any], answer: str) -> str:
    refs = row.get("reference_answers")
    ref = "[" + "; ".join(refs) + "]" if refs else "[]"  # as LLMJudgeAbstentionDetector.detect_abstention
    return template.format(
        question=row["question"],
        ref_answer=ref,
        abstention_label=row["should_abstain"],
        model_answer=answer,
    )


def parse_judge(text: str) -> bool | None:
    t = (text or "").lower().strip(" .,\n")
    return {"yes": True, "no": False}.get(t)


async def _complete(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    sem: asyncio.Semaphore,
    **kw: Any,
) -> str:
    async with sem:
        try:
            r = await client.chat.completions.create(model=model, messages=messages, **kw)
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            return f"<<error: {type(e).__name__}: {e}>>"


async def run(
    rows: list[dict[str, Any]],
    *,
    policy: Any,
    policy_model: str,
    judge: Any,
    judge_model: str,
    judge_template: str,
    concurrency: int = 64,
    max_tokens: int = 4096,
    temperature: float | None = 0.0,
    system_prompt: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    kw: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        kw["temperature"] = temperature
    if extra_body:
        kw["extra_body"] = extra_body

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": row["question"]}
        ]
        answer = await _complete(policy, policy_model, msgs, sem, **kw)
        verdict = None
        judge_raw = ""
        if not answer.startswith("<<error"):
            judge_raw = await _complete(
                judge,
                judge_model,
                [
                    {
                        "role": "user",
                        "content": format_judge_prompt(judge_template, row, answer),
                    }
                ],
                sem,
                temperature=0.0,
                max_tokens=100,
            )
            verdict = parse_judge(judge_raw)
        return {
            **row,
            "response": answer,
            "judge_response": judge_raw,
            "is_abstention": verdict,
        }

    return await asyncio.gather(*(one(r) for r in rows))


def _prf(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    rows = [r for r in rows if r.get("is_abstention") is not None]
    tp = sum(r["should_abstain"] and r["is_abstention"] for r in rows)
    fp = sum((not r["should_abstain"]) and r["is_abstention"] for r in rows)
    fn = sum(r["should_abstain"] and not r["is_abstention"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0  # sklearn zero_division default ("warn") → 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }


def score(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    per = {d: _prf(rs) for d, rs in sorted(by_ds.items())}
    keys = ("precision", "recall", "f1")
    return {
        "overall": _prf(results),
        "macro": {k: sum(v[k] for v in per.values()) / len(per) if per else None for k in keys},
        "unjudged": sum(r.get("is_abstention") is None for r in results),
        "by_dataset": per,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="inference + judge + score")
    r.add_argument(
        "--data",
        required=True,
        help="abstentionbench.jsonl from the standalone exporter",
    )
    r.add_argument("--out", required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--api-key", default="EMPTY")
    r.add_argument(
        "--judge-base-url",
        required=True,
        help="endpoint serving the judge (e.g. Llama-3.1-8B-Instruct)",
    )
    r.add_argument("--judge-model", default="meta-llama/Llama-3.1-8B-Instruct")
    r.add_argument("--judge-api-key", default="EMPTY")
    r.add_argument(
        "--judge-prompt-file",
        required=True,
        help="AbstentionBench/recipe/evaluation_judge_prompts.py",
    )
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument(
        "--system-prompt-file",
        help="optional system prompt (default none, as in AbstentionBench)",
    )
    r.add_argument("--no-thinking", action="store_true")
    r.add_argument("--limit", type=int)
    s = sub.add_parser("score", help="re-score an existing responses.jsonl")
    s.add_argument("responses")
    args = p.parse_args(argv)

    if args.cmd == "score":
        print(json.dumps(score(_read_jsonl(args.responses)), indent=2))
        return 0

    from openai import AsyncOpenAI

    rows = _read_jsonl(args.data)[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(
        run(
            rows,
            policy=AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600, max_retries=3),
            policy_model=args.model,
            judge=AsyncOpenAI(
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                timeout=600,
                max_retries=3,
            ),
            judge_model=args.judge_model,
            judge_template=load_judge_prompt(args.judge_prompt_file),
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=Path(args.system_prompt_file).read_text() if args.system_prompt_file else None,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else None,
        )
    )
    with open(out / "responses.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in results)
    summary = score(results)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_dataset"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
