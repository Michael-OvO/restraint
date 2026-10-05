"""Held-out evaluation CLI: roll out a served policy on an eval split, grade every episode, write metrics.

    python -m honest_agents.eval.run_eval --split data/test.jsonl \\
        --base-url http://localhost:8000/v1 --model policy --out runs/eval/base

Outputs in `--out`:
  episodes.jsonl      one `EpisodeRecord` per (task, sample), appended as episodes finish (crash-safe, `--resume`)
  transcripts/        full chat transcript per episode
  metrics.json        `honest_agents.eval.metrics.compute_metrics` (with reference policies and the confidence sweep)
  summary.md          human-readable tables
  config.json         the exact rollout / reward configuration and the evaluation protocol (fixed before the run)

Limits and sampling default to the training values (`honest_agents.harness`); `--train-config` reads them from a
training YAML so a run with non-default caps is evaluated under the same caps.

Baselines without training:
  --system-prompt-file FILE / --neutral-prompt   replace the system prompt (the 2×2 of {base, trained} × {prompt})
  the confidence-threshold sweep is part of every metrics.json

Other modes:
  --metrics-only                recompute metrics.json / summary.md from an existing episodes.jsonl
  --compare DIR [DIR ...]       comparison table + Pareto front over finished runs
  --noninferiority CAND REF     non-inferiority of CAND against REF on verified success (the primary endpoint), margin
                                from the protocol; each side may be several seed runs, comma separated (a,b,c)
  --diff CAND REF --metric M    paired difference CAND − REF on any metric (H1 direction check)
  --bank-dir DIR                also write a trajectory bank (`honest_agents.bank`) for the report-only pilot
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from honest_agents import bank, harness
from honest_agents.eval import metrics as M
from honest_agents.eval.metrics import DEFAULT_ALPHAS, compute_metrics
from honest_agents.eval.protocol import EvalProtocol, endpoint_role, protocol_of_run
from honest_agents.eval.records import (
    EpisodeRecord,
    iter_records,
    read_records,
    record_line,
    write_records,
)
from honest_agents.eval.rollout import RolloutConfig, Task, load_split, run_rollouts
from honest_agents.eval.summary import render_comparison, render_summary, render_test
from honest_agents.reward import RewardConfig

NEUTRAL_PROMPT_FILE = Path(__file__).parent / "prompts" / "neutral_system_prompt.txt"
_UNSET: Any = object()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m honest_agents.eval.run_eval",
        description=__doc__.split("\n\n")[0],
    )
    p.add_argument(
        "--split",
        help="JSONL split (task_dir column) or a directory of Harbor task dirs",
    )
    p.add_argument("--out", required=False, help="output directory")
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument(
        "--model",
        help="served model name (vLLM --served-model-name / LoRA module name)",
    )
    p.add_argument("--concurrency", type=int, default=8, help="live episodes (= sandboxes) at once")
    p.add_argument("--max-concurrent-boots", type=int, default=None)
    p.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="rollouts per task (paired by sample index)",
    )
    p.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    p.add_argument(
        "--train-config",
        help="training YAML: take caps, sampling and reward config from it",
    )
    p.add_argument(
        "--max-tool-rounds",
        type=int,
        default=_UNSET,
        help="default: training max_tool_calling_iterations",
    )
    p.add_argument(
        "--max-completion-tokens",
        type=int,
        default=_UNSET,
        help="default: training max_completion_length",
    )
    p.add_argument(
        "--max-tokens-per-turn",
        type=int,
        default=_UNSET,
        help="default: training per-turn max_tokens",
    )
    p.add_argument("--temperature", type=float, default=_UNSET, help="default: 1.0 (training)")
    p.add_argument(
        "--top-p",
        type=float,
        default=_UNSET,
        help="default: 1.0 (training: full distribution)",
    )
    p.add_argument(
        "--server-sampling-defaults",
        action="store_true",
        help="send no sampling fields; the server's defaults apply (NOT what training does)",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="send chat_template_kwargs.enable_thinking=false (Qwen3 / Qwen3.5 non-thinking mode, as in training)",
    )
    p.add_argument("--system-prompt-file", help="replace the system prompt with this file's text")
    p.add_argument(
        "--neutral-prompt",
        action="store_true",
        help=f"the shipped {NEUTRAL_PROMPT_FILE.name} and neutral tool descriptions (the default tool docstrings "
        "repeat the honesty instruction)",
    )
    p.add_argument(
        "--neutral-tools",
        action="store_true",
        help="neutral tool descriptions only (implied by --neutral-prompt)",
    )
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--environment-type", default="docker")
    p.add_argument(
        "--command-timeout",
        type=int,
        default=None,
        help="default: EpisodeBudget.command_timeout_sec",
    )
    p.add_argument(
        "--tests-timeout",
        type=int,
        default=None,
        help="default: env.tests_timeout_sec of --train-config, else GradingConfig.tests_timeout_sec",
    )
    p.add_argument("--no-network-enforcement", action="store_true")
    p.add_argument(
        "--no-evidence-replay",
        action="store_true",
        help="skip replaying evidence_commands after grading",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip (task, sample) already in episodes.jsonl",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=_UNSET,
        help="α of the reward used for mean_reward",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help="α sweep for re-scoring",
    )
    p.add_argument(
        "--protocol",
        help="JSON file overriding the evaluation protocol (margin, bootstrap); rollout time",
    )
    p.add_argument("--n-boot", type=int, default=_UNSET, help="default: the protocol's n_boot")
    p.add_argument("--seed", type=int, default=_UNSET, help="default: the protocol's seed")
    p.add_argument(
        "--bank-dir",
        help="append one trajectory-bank row per episode to this directory",
    )
    p.add_argument("--quarantine-file", help="append variants that leaked to this quarantine list")
    p.add_argument(
        "--exclude-quarantined",
        help="quarantine list whose variants are void for the metrics",
    )
    p.add_argument("--metrics-only", action="store_true")
    p.add_argument(
        "--compare",
        nargs="+",
        metavar="RUN_DIR",
        help="compare finished runs instead of evaluating",
    )
    p.add_argument("--noninferiority", nargs=2, metavar=("CANDIDATE_DIR[,SEED_DIR...]", "REFERENCE_DIR[,SEED_DIR...]"))
    p.add_argument("--diff", nargs=2, metavar=("CANDIDATE_DIR[,SEED_DIR...]", "REFERENCE_DIR[,SEED_DIR...]"))
    p.add_argument(
        "--metric",
        default=None,
        help="metric for --diff / --noninferiority (default: protocol ni_metric)",
    )
    p.add_argument(
        "--ni-margin",
        type=float,
        default=None,
        help="must equal the margin both runs were started under",
    )
    return p


# ── configuration ────────────────────────────────────────────────────────


def load_train_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """A training YAML as a plain dict, with `base:` inheritance resolved (same rule as `train.load_config_dict`).

    Read here, without importing `honest_agents.train`, so evaluation does not need the training stack.
    """
    import yaml

    p = Path(path).resolve()
    if p in _seen:
        raise ValueError(f"circular `base:` chain at {p}")
    d = yaml.safe_load(p.read_text()) or {}
    base = d.pop("base", None)
    if base is None:
        return d
    merged = load_train_config(p.parent / base, (*_seen, p))

    def merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    return merge(merged, d)


def _pick(cli: Any, *fallbacks: Any) -> Any:
    """The CLI value if given, else the first fallback that is not None."""
    if cli is not _UNSET:
        return cli
    return next((f for f in fallbacks if f is not None), None)


def resolve_configs(
    args: argparse.Namespace,
) -> tuple[RolloutConfig, RewardConfig, Any]:
    """(rollout config, reward config, episode budget) from the CLI, the training YAML and the training defaults."""
    train = load_train_config(args.train_config) if args.train_config else {}
    grpo = train.get("grpo") or {}
    gen = grpo.get("generation_kwargs") or {}
    if args.system_prompt_file and args.neutral_prompt:
        raise SystemExit("--system-prompt-file and --neutral-prompt are mutually exclusive")
    prompt_file = NEUTRAL_PROMPT_FILE if args.neutral_prompt else args.system_prompt_file
    sampling: dict[str, Any] = {
        "temperature": _pick(args.temperature, grpo.get("temperature"), 1.0),
        "top_p": _pick(args.top_p, grpo.get("top_p"), gen.get("top_p"), 1.0),
        # the rest of the full-distribution settings, from the training config when it sets them
        "top_k": _pick(_UNSET, grpo.get("top_k"), 0),
        "min_p": _pick(_UNSET, grpo.get("min_p"), 0.0),
        "repetition_penalty": _pick(_UNSET, grpo.get("repetition_penalty"), 1.0),
    }
    if args.server_sampling_defaults:
        sampling = dict.fromkeys(("temperature", "top_p", "top_k", "min_p", "repetition_penalty"))
    rollout_cfg = RolloutConfig(
        model=args.model,
        max_tool_rounds=_pick(
            args.max_tool_rounds,
            grpo.get("max_tool_calling_iterations"),
            harness.DEFAULT_MAX_TOOL_ITERATIONS,
        ),
        max_completion_tokens=_pick(
            args.max_completion_tokens,
            grpo.get("max_completion_length"),
            harness.DEFAULT_MAX_COMPLETION_TOKENS,
        ),
        max_tokens_per_turn=_pick(
            args.max_tokens_per_turn,
            gen.get("max_tokens"),
            harness.DEFAULT_TURN_MAX_TOKENS,
        ),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else None,
        system_prompt=Path(prompt_file).read_text() if prompt_file else None,
        neutral_tools=bool(args.neutral_prompt or args.neutral_tools),
        **sampling,
    )
    # The environment's own limits, as `train.build_budget` applies them: the command timeout and any `env.budget`.
    env_cfg = train.get("env") or {}
    overrides: dict[str, Any] = (
        {"command_timeout_sec": env_cfg["command_timeout_sec"]} if "command_timeout_sec" in env_cfg else {}
    )
    overrides.update(env_cfg.get("budget") or {})
    if args.command_timeout:
        overrides["command_timeout_sec"] = args.command_timeout
    budget = rollout_cfg.budget(**overrides)
    known = {f.name for f in dataclasses.fields(RewardConfig)}
    reward_kwargs = {k: v for k, v in ((train.get("reward") or {}).get("config") or {}).items() if k in known}
    reward_kwargs.setdefault("efficiency_ref_actions", budget.max_bash_calls)  # as train/ sets it
    if args.alpha is not _UNSET:
        reward_kwargs["alpha"] = args.alpha
    return rollout_cfg, RewardConfig(**reward_kwargs), budget


def _protocol(args: argparse.Namespace, out: Path | None = None) -> EvalProtocol:
    """The protocol of a run: the file given now, else what the run was started under, else the defaults."""
    if args.protocol:
        proto = EvalProtocol.from_file(args.protocol)
    else:
        proto = (protocol_of_run(out) if out is not None else None) or EvalProtocol()
    changes = {k: getattr(args, k) for k in ("n_boot", "seed") if getattr(args, k) is not _UNSET}
    return dataclasses.replace(proto, **changes) if changes else proto


def _stored_reward_config(out: Path, args: argparse.Namespace) -> RewardConfig:
    """The reward config a finished run was scored under (config.json), with `--alpha` applied."""
    kwargs: dict[str, Any] = {}
    cfg_file = out / "config.json"
    if cfg_file.is_file():
        known = {f.name for f in dataclasses.fields(RewardConfig)}
        kwargs = {k: v for k, v in (json.loads(cfg_file.read_text()).get("reward") or {}).items() if k in known}
    if args.alpha is not _UNSET:
        kwargs["alpha"] = args.alpha
    return RewardConfig(**kwargs)


def _excluded(args: argparse.Namespace) -> set[str]:
    return (
        bank.quarantined_variants(args.exclude_quarantined, exclude_sources=()) if args.exclude_quarantined else set()
    )


# ── outputs ──────────────────────────────────────────────────────────────


def write_metrics(out: Path, cfg: RewardConfig, args: argparse.Namespace, title: str | None = None) -> dict:
    records = read_records(out / "episodes.jsonl")
    proto = _protocol(args, out)
    m = compute_metrics(
        records,
        cfg,
        alphas=args.alphas,
        n_boot=proto.n_boot,
        seed=proto.seed,
        exclude_variants=_excluded(args),
        min_source_tasks=proto.min_source_tasks,
        min_family_variants=proto.min_family_variants,
    )
    m["protocol"] = proto.to_dict()
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    (out / "summary.md").write_text(render_summary(m, title or f"Honest Agents eval: {out.name}"))
    if args.quarantine_file:
        write_quarantine(records, cfg, args.quarantine_file)
    return m


def write_quarantine(records: list[EpisodeRecord], cfg: RewardConfig, path: str | Path) -> int:
    """Append every variant that leaked in this run to the quarantine list (once per variant and kind)."""
    known = {(e.variant_id, e.kind) for e in bank.read_quarantine(path)}
    entries = []
    for e in M.score_records(records, cfg):
        kind = (
            bank.QUARANTINE_LEAK
            if e.br.leak
            else bank.QUARANTINE_UNCLAIMED_PASS
            if e.br.unclaimed_pass_infeasible
            else None
        )
        if kind is None or (e.rec.variant_key, kind) in known:
            continue
        known.add((e.rec.variant_key, kind))
        entries.append(
            bank.QuarantineEntry(
                variant_id=e.rec.variant_key,
                task_dir=e.rec.task_dir,
                pair_id=e.rec.pair_id,
                perturbation=e.rec.perturbation,
                family=e.rec.family,
                kind=kind,
                source="eval",
                detail=f"episode {e.rec.episode_id} (model {e.rec.model}, harness {e.rec.harness})",
            )
        )
    bank.append_quarantine(entries, path)
    return len(entries)


def bank_row(
    rec: EpisodeRecord,
    transcript: list[dict[str, Any]],
    task: Task | None,
    cfg: RewardConfig,
) -> Any:
    """One trajectory-bank row for an evaluated episode (no token ids: the chat API returns text)."""
    messages = [m for m in transcript if m.get("role") != "_reasoning"]
    n_prompt = int(rec.extra.get("n_prompt_messages") or 0)
    br = rec.score(cfg)
    row = {k: v for k, v in (task.row if task else {}).items() if k != "prompt"}
    if rec.task_dir:
        try:
            from honest_agents.spec import HonestTaskSpec

            row["spec"] = HonestTaskSpec.from_task_dir(rec.task_dir).model_dump(mode="json")
        except (OSError, ValueError):
            pass
    return bank.make_row(
        prompt=messages[:n_prompt],
        completion=messages[n_prompt:],
        outcome=rec.outcome,
        breakdown=br,
        reward_value=br.reward,
        task={**row, "task_dir": rec.task_dir, "task_id": rec.task_id},
        blocker_facts=rec.extra.get("blocker_facts"),
        reward_config=cfg,
        reward_mode="eval",
        group=rec.variant_key,
        extra={
            "episode_id": rec.episode_id,
            "model": rec.model,
            "harness": rec.harness,
            "sample": rec.sample,
        },
    )


def _client(args: argparse.Namespace):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=3,
    )


def _env_factory(args: argparse.Namespace, reward_cfg: RewardConfig, budget: Any = None):
    from honest_agents.env import HonestHarborEnv
    from honest_agents.grading import GradingConfig
    from honest_agents.harness import set_max_concurrent_boots

    if args.max_concurrent_boots:
        set_max_concurrent_boots(args.max_concurrent_boots)
    # Evidence commands are replayed in evaluation only, after every check and test. The test timeout is
    # the training one when a training config is given: a slower verifier would turn passes into failures.
    grading_kwargs: dict[str, Any] = {"replay_evidence": not args.no_evidence_replay}
    tests_timeout = args.tests_timeout
    if tests_timeout is None and args.train_config:
        tests_timeout = (load_train_config(args.train_config).get("env") or {}).get("tests_timeout_sec")
    if tests_timeout is not None:
        grading_kwargs["tests_timeout_sec"] = int(tests_timeout)
    grading_cfg = GradingConfig(**grading_kwargs)

    def make() -> HonestHarborEnv:
        return HonestHarborEnv(
            environment_type=args.environment_type,
            reward_config=reward_cfg,
            grading_config=grading_cfg,
            budget=budget,
            enforce_network_policy=not args.no_network_enforcement,
        )

    return make


def _check_resumable(out: Path, args: argparse.Namespace, rollout_cfg: RolloutConfig) -> None:
    """Refuse to mix episodes of two different policies, prompts or splits in one run directory."""
    cfg_file = out / "config.json"
    if not cfg_file.is_file():
        return
    old = json.loads(cfg_file.read_text())
    now = dataclasses.asdict(rollout_cfg)
    diffs = [
        f"{k}: {old['rollout'].get(k)!r} -> {now.get(k)!r}"
        for k in ("model", "system_prompt", "neutral_tools", "max_tool_rounds", "max_completion_tokens", "temperature")
        if k in old.get("rollout", {}) and old["rollout"][k] != now[k]
    ]
    if old.get("split") != args.split:
        diffs.append(f"split: {old.get('split')!r} -> {args.split!r}")
    if diffs:
        raise SystemExit("--resume would mix two configurations in one run directory:\n  " + "\n  ".join(diffs))


def evaluate(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    rollout_cfg, reward_cfg, budget = resolve_configs(args)
    proto = _protocol(args, out if args.resume else None)  # a resumed run keeps the protocol it started under
    tasks = load_split(args.split)[: args.limit]
    by_id = {t.task_id: t for t in tasks}
    episodes = out / "episodes.jsonl"
    skip: set[tuple[str, int]] = set()
    if args.resume:
        _check_resumable(out, args, rollout_cfg)
    if args.resume and episodes.exists():  # keep finished episodes; infra-error ones are retried
        kept = [r for r in iter_records(episodes) if r.infra_error is None]
        write_records(episodes, kept, cfg=reward_cfg)
        skip = {(r.task_id, r.sample) for r in kept}
    elif episodes.exists():
        episodes.unlink()
    (out / "config.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "base_url": args.base_url,
                "harness": "honest",
                "n_tasks": len(tasks),
                "n_samples": args.n_samples,
                "rollout": dataclasses.asdict(rollout_cfg),
                "budget": dataclasses.asdict(budget) if dataclasses.is_dataclass(budget) else str(budget),
                "reward": dataclasses.asdict(reward_cfg),
                "protocol": proto.to_dict(),
                "train_config": args.train_config,
                "system_prompt_source": (
                    "neutral" if args.neutral_prompt else args.system_prompt_file or "harness.SYSTEM_PROMPT / row"
                ),
                "environment_type": args.environment_type,
                "argv": sys.argv,
            },
            indent=2,
        )
    )
    done = 0

    def on_result(rec, transcript) -> None:
        nonlocal done
        f.write(record_line(rec, reward_cfg))
        f.flush()
        (out / "transcripts" / f"{rec.episode_id.replace('/', '_')}.json").write_text(
            json.dumps(transcript, indent=1, default=str)
        )
        if args.bank_dir and rec.infra_error is None:
            bank.append_rows(
                [bank_row(rec, transcript, by_id.get(rec.task_id), reward_cfg)],
                args.bank_dir,
            )
        done += 1
        status = rec.infra_error or rec.score(reward_cfg).case
        print(
            f"[{done}/{len(tasks) * args.n_samples - len(skip)}] {rec.task_id} s{rec.sample}: {status}",
            flush=True,
        )

    with open(episodes, "a") as f:
        asyncio.run(
            run_rollouts(
                tasks,
                rollout_cfg,
                client=_client(args),
                env_factory=_env_factory(args, reward_cfg, budget),
                concurrency=args.concurrency,
                n_samples=args.n_samples,
                skip=skip,
                on_result=on_result,
            )
        )
    return write_metrics(out, reward_cfg, args)


# ── comparisons between finished runs ────────────────────────────────────


def compare_two(args: argparse.Namespace, dirs: list[str], noninferiority: bool) -> dict[str, Any]:
    arms = [[Path(d) for d in arm.split(",") if d] for arm in dirs]  # each arm: one run per training seed
    runs = [d for arm in arms for d in arm]
    protos = [_protocol(args, d) for d in runs]
    if any(p != protos[0] for p in protos):
        raise SystemExit("the runs were started under different protocols:\n  " + "\n  ".join(map(str, protos)))
    proto = protos[0]
    if args.ni_margin is not None and abs(args.ni_margin - proto.ni_margin) > 1e-12:
        raise SystemExit(
            f"--ni-margin {args.ni_margin} differs from the margin fixed before the runs ({proto.ni_margin}); "
            "the margin cannot be changed after the fact"
        )
    cfg = _stored_reward_config(arms[0][0], args)
    recs = [[read_records(d / "episodes.jsonl") for d in arm] for arm in arms]
    cand, ref = (r if len(r) > 1 else r[0] for r in recs)  # a single run stays a flat list
    metric = args.metric or proto.ni_metric
    names = [",".join(d.name for d in arm) for arm in arms]
    if noninferiority:
        t = M.noninferiority(
            cand,
            ref,
            cfg,
            margin=proto.ni_margin,
            metric=metric,
            alpha=proto.ni_alpha,
            n_boot=proto.n_boot,
            seed=proto.seed,
            allow_single_seed=proto.ni_allow_single_seed,
        )
        title = f"Non-inferiority: {names[0]} against {names[1]}"
    else:
        t = M.paired_difference(cand, ref, metric, cfg, n_boot=proto.n_boot, seed=proto.seed, level=proto.ci_level)
        t.pop("samples")
        title = f"Paired difference: {names[0]} − {names[1]}"
    t["endpoint"] = endpoint_role(metric)
    print(render_test(t, title))
    return t


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        runs = {Path(d).name: json.loads((Path(d) / "metrics.json").read_text()) for d in args.compare}
        print(render_comparison(runs))
        return 0
    if args.noninferiority or args.diff:
        t = compare_two(args, args.noninferiority or args.diff, bool(args.noninferiority))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(t, indent=2))
        return 0 if t["difference"] is not None else 1
    if not args.out:
        raise SystemExit("--out is required")
    if args.metrics_only:
        m = write_metrics(Path(args.out), _stored_reward_config(Path(args.out), args), args)
    else:
        if not (args.split and args.model):
            raise SystemExit("--split and --model are required (or use --metrics-only / --compare)")
        m = evaluate(args)
    print((Path(args.out) / "summary.md").read_text())
    return 0 if m["counts"]["scored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
