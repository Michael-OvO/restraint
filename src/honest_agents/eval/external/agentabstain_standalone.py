"""AgentAbstain (arXiv 2607.10059, github.com/AntiQuality/agentabstain) against our vLLM endpoint.

Runs INSIDE the AgentAbstain checkout's own py3.11 venv (cwd = repo root); imports only their deps, never honest_agents.

    python agentabstain_standalone.py infer --base-url http://localhost:8000/v1 --model policy \\
        --task-config src/configs/tasks_preview.yaml --workers 8
    OPENAI_API_KEY=sk-... python -m eval.runner --provider openaisdk --model policy     # GPT-5.4 judge (paid)
    python agentabstain_standalone.py summarize --model policy

Why a launcher: their `openaisdk` harness builds `agents.Agent(model=<str>)` and never selects an API, so the
openai-agents SDK uses the *Responses* API on the default client (api.openai.com). We point the SDK's default client at
our endpoint and switch it to chat completions — vLLM's best-tested tool-calling path — before their `run_inference`
starts (`set_default_openai_client`, `set_default_openai_api`, `set_tracing_disabled`; openai-agents 0.23.x).

Non-thinking mode for Qwen3 is set server-side (`vllm serve --default-chat-template-kwargs '{"enable_thinking":
false}'`, which `EVAL=1 scripts/serve_vllm.sh` does), because their runtime only forwards `temperature` to the model.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import tempfile
from pathlib import Path


def infer(args: argparse.Namespace) -> int:
    from agents import (
        set_default_openai_api,
        set_default_openai_client,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    set_default_openai_client(AsyncOpenAI(base_url=args.base_url, api_key=args.api_key), use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)

    runtime = Path(tempfile.mkdtemp()) / "runtime.yaml"
    runtime.write_text(
        "\n".join(
            [
                'provider: "openaisdk"',
                f"model: {json.dumps(args.model)}",
                f"temperature: {args.temperature}",
                f"max_turns: {args.max_turns}",
                f"results_root: {json.dumps(args.results_root)}",
            ]
        )
        + "\n"
    )
    argv = [
        "run_inference",
        "--runtime-config",
        str(runtime),
        "--task-config",
        args.task_config,
    ]
    argv += ["--workers", str(args.workers)] + (["--multi-run"] if args.multi_run else [])
    sys.argv = argv
    runpy.run_module("src.scripts.run_inference", run_name="__main__")
    return 0


def summarize(args: argparse.Namespace) -> int:
    """Leaderboard numbers as `eval/statistics/figure_leaderboard_bar.py` computes them: metric `combined`, mean over
    action types within a category, then macro-mean over the 8 categories."""
    from eval.statistics.analysis import (
        build_metric_run_frame,
        compute_summary_metrics,
        load_eval_frame,
        select_latest_runs,
    )

    summary = compute_summary_metrics(build_metric_run_frame(select_latest_runs(load_eval_frame(args.results_root))))
    if summary.empty:
        print("no evaluated runs found (run `python -m eval.runner` first)")
        return 1
    combined = summary[(summary["metric_name"] == "combined") & (summary["model"] == args.model)]
    cols = ["should_act_accuracy", "should_abstain_accuracy", "paired_accuracy", "car"]
    per_cat = combined.groupby("category")[cols].mean()
    macro = per_cat.mean()
    out = {
        "model": args.model,
        "macro": {c: float(macro[c]) for c in cols},
        "per_category": {cat: {c: float(row[c]) for c in cols} for cat, row in per_cat.iterrows()},
        "n_pairs": int(combined["num_pairs"].sum()),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


def main() -> int:
    sys.path.insert(0, str(Path.cwd()))  # their packages (src/, eval/, agent/) live at the repo root = cwd
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--base-url", required=True)
    i.add_argument("--api-key", default="EMPTY")
    i.add_argument(
        "--model",
        required=True,
        help="served model name; becomes results/openaisdk/<model>/",
    )
    i.add_argument("--temperature", type=float, default=0.0)
    i.add_argument("--max-turns", type=int, default=30)
    i.add_argument("--results-root", default="results/openaisdk")
    i.add_argument(
        "--task-config",
        default="src/configs/tasks_preview.yaml",
        help="263 pairs (tasks.yaml has 10)",
    )
    i.add_argument("--workers", type=int, default=8)
    i.add_argument("--multi-run", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--model", required=True)
    s.add_argument("--results-root", default="results")
    s.add_argument("--out")
    args = p.parse_args()
    return infer(args) if args.cmd == "infer" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
