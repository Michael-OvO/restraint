"""Command-line entry point of `python -m honest_agents.train`."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from honest_agents import train as train_package  # only its docstring is read, at call time
from honest_agents.train.config import EXIT_STOPPED_BY_RULE, EXIT_UNSAFE_ALPHA, load_run_config
from honest_agents.train.derive import UnsafeAlphaError, build_limits, check_safe_alpha, served_max_model_len
from honest_agents.train.plan import _server_url, build_plan
from honest_agents.train.run import train
from honest_agents.train.startup import check_template_offline

logger = logging.getLogger("honest_agents.train")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="honest_agents.train", description=train_package.__doc__.split("\n\n")[0])
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
