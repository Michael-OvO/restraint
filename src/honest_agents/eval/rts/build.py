"""Build `rts/v1` (spec phases P0-P2): seeds, candidate pairs, the gate, the freeze.

  python -m honest_agents.eval.rts.build seeds [--cards humanevalfix terminal-bench ...] [--limit N] [--cache DIR]
  python -m honest_agents.eval.rts.build pairs [--seeds-index FILE] [--out data/rts/v1/candidates] [--operators ...]
  python -m honest_agents.eval.rts.build measure [--out data/rts/v1/candidates] --concurrency 2   # Docker; then pairs again
  python -m honest_agents.eval.rts.build gate --pairs data/rts/v1/candidates/pairs.jsonl --concurrency 8   # Docker
  python -m honest_agents.eval.rts.build freeze --train-tasks DIR [--train-tasks DIR ...] [--check]

`seeds` downloads the pinned Harbor datasets (network, no Docker) and writes `<cache>/index.jsonl`. `pairs` logs every
operator's applicability on every seed to `<out>/applicability.jsonl` and writes the built pairs to `<out>/pairs.jsonl`
(no Docker). `measure` runs the seed's oracle for the verdicts `pairs` logged as stage `measure` (`disk_quota`,
`mem_limit` size their pair from it; Docker); `pairs` run again builds them. `gate` runs on a Linux node: resumable, one row per pair with its rejection
reason in `<out>/gate.jsonl`. `freeze` writes `data/rts/v1/` (manifest, sha256 sidecar, task dirs, protocol, yield);
`--check` writes the report only. A run is then scored with `eval.run_eval --split data/rts/v1/tasks` and
`eval.rts.scoreboard add --manifest data/rts/v1/manifest.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from honest_agents.eval.rts import benchmarks, candidates, freeze, gate, measure
from honest_agents.eval.rts.protocol import RtsProtocol
from honest_agents.tasks.perturb.rts.base import RTSOperatorSpec

log = logging.getLogger(__name__)


def _operators(names: list[str] | None) -> dict[str, RTSOperatorSpec]:
    from honest_agents.tasks.perturb.rts.registry import RTS_OPERATORS

    if names is None:
        return dict(RTS_OPERATORS)
    if unknown := sorted(set(names) - set(RTS_OPERATORS)):
        raise SystemExit(f"unknown operators {unknown}; registered: {sorted(RTS_OPERATORS)}")
    return {n: RTS_OPERATORS[n] for n in names}


def cmd_seeds(args: argparse.Namespace) -> int:
    cards = args.cards or list(benchmarks.CARDS)
    if unknown := sorted(set(cards) - set(benchmarks.CARDS)):
        raise SystemExit(f"unknown cards {unknown}; known: {sorted(benchmarks.CARDS)}")
    cache = Path(args.cache)
    all_seeds = []
    for name in cards:
        card = benchmarks.CARDS[name]
        if not card.license_known and not args.allow_unknown_license:
            raise SystemExit(f"{name}: licence not known ({card.license}); pass --allow-unknown-license")
        seeds = benchmarks.load_seeds(card, cache, limit=args.limit, seed=args.seed)
        print(f"{name}: {len(seeds)} seeds ({sum(s.has_oracle for s in seeds)} with an oracle)")
        all_seeds += seeds
    index = Path(args.index) if args.index else cache / benchmarks.INDEX_NAME
    benchmarks.write_index(all_seeds, index)
    print(f"wrote {index}")
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    index = Path(args.seeds_index) if args.seeds_index else Path(args.cache) / benchmarks.INDEX_NAME
    seeds = benchmarks.read_index(index)
    if args.benchmarks:
        seeds = [s for s in seeds if s.benchmark in set(args.benchmarks)]
    ops = _operators(args.operators)
    if not ops:
        raise SystemExit("no RTS operator is registered (tasks/perturb/rts/<operator>.py with a SPEC)")
    out = Path(args.out)
    mpath = Path(args.measurements) if args.measurements else out / candidates.MEASUREMENTS_NAME
    pairs, rows = candidates.build_pairs(seeds, out, ops, seed=args.seed, measurements=measure.read_measurements(mpath))
    candidates.write_jsonl(rows, out / candidates.APPLICABILITY_NAME)
    candidates.write_jsonl([p.to_json() for p in pairs], out / candidates.PAIRS_NAME)
    by = Counter((r["operator"], r["stage"]) for r in rows)
    for op in sorted(ops):
        print(
            f"{op}: {by[(op, 'built')]} built, {by[(op, 'build')]} failed to build, {by[(op, 'applicable')]} not applicable"
            + (f", {by[(op, 'measure')]} awaiting or failing a measurement" if by[(op, "measure")] else "")
        )
    print(f"wrote {len(pairs)} pairs to {out / candidates.PAIRS_NAME}")
    if todo := [r for r in measure.pending(rows) if not r.get("measured")]:
        print(f"{len(todo)} verdicts need a measurement on the seed's oracle: run `measure`, then `pairs` again")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    from functools import partial

    from honest_agents.harness import set_max_concurrent_boots
    from honest_agents.tasks.validate import OracleHarborEnv

    out_dir = Path(args.out)
    rows = [json.loads(ln) for ln in (out_dir / candidates.APPLICABILITY_NAME).read_text().splitlines() if ln.strip()]
    todo = measure.pending(rows)
    if args.operators:
        todo = [r for r in todo if r["operator"] in set(args.operators)]
    if args.limit is not None:
        todo = todo[: args.limit]
    path = Path(args.measurements) if args.measurements else out_dir / candidates.MEASUREMENTS_NAME
    set_max_concurrent_boots(max(args.concurrency, 1))
    m = partial(
        measure.measure_seed,
        env_factory=partial(OracleHarborEnv, environment_type=args.environment_type),
        timeout_sec=args.oracle_timeout,
    )
    done = asyncio.run(measure.measure_many(todo, path, measure=m, concurrency=args.concurrency))
    wanted = {(r["source_task"], r["operator"]) for r in todo}
    got = [done[k] for k in wanted if k in done]
    ok = sum(r.get(r["quantity"]) is not None for r in got)
    print(f"{ok}/{len(got)} measured -> {path}; run `pairs` again to build them")
    return 0


def _print_summary(results: Mapping[str, Mapping[str, object]]) -> None:
    for op, s in gate.summary(results).items():
        print(f"{op}: {s['kept']}/{s['gated']} kept ({s['infra']} infrastructure errors)")
        for why, n in sorted(s["reasons"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {n:4d}  {why}")


def cmd_gate(args: argparse.Namespace) -> int:
    from honest_agents.harness import set_max_concurrent_boots

    pairs_path = Path(args.pairs)
    rows = candidates.read_pairs(pairs_path)
    if args.operators:
        rows = [r for r in rows if r.operator in set(args.operators)]
    out = Path(args.out) if args.out else pairs_path.parent / "gate.jsonl"
    settings = gate.default_settings(oracle_runs=args.oracle_runs, oracle_timeout_sec=args.oracle_timeout)
    validate, probe = gate.docker_checks(settings, args.environment_type)
    set_max_concurrent_boots(max(args.concurrency, 16))  # as `tasks.validate`: boots are throttled apart
    results = asyncio.run(
        gate.gate_many(
            rows,
            out,
            validate=validate,
            probe=probe,
            settings=settings,
            concurrency=args.concurrency,
            resume=not args.no_resume,
        )
    )
    _print_summary(results)
    print(f"{sum(bool(r.get('ok')) for r in results.values())}/{len(results)} pairs kept -> {out}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    pairs_path = Path(args.pairs)
    pairs = candidates.read_pairs(pairs_path)
    gate_path = Path(args.gate) if args.gate else pairs_path.parent / "gate.jsonl"
    results = gate.read_results(gate_path)
    proto = RtsProtocol.from_json(json.loads(Path(args.protocol).read_text())) if args.protocol else RtsProtocol()
    out = Path(args.out)
    if (out / freeze.MANIFEST_NAME).exists() and not args.check:
        raise SystemExit(f"{out / freeze.MANIFEST_NAME} exists: a frozen version is never rewritten")
    plan = freeze.plan_freeze(
        pairs, results, _operators(None), train_dirs=[Path(d) for d in args.train_tasks], proto=proto
    )
    report = {
        "kept_pairs": len(plan.kept),
        "dropped": plan.dropped,
        **freeze.yield_report(pairs, results, plan),
    }
    report_path = Path(args.report) if args.report else pairs_path.parent / freeze.REPORT_NAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"{len(plan.kept)} pairs pass the static checks ({len(plan.dropped)} dropped); report {report_path}")
    for p in plan.balance:
        print(f"  balance: {p}")
    if args.check:
        return 0 if plan.ok else 1
    if not plan.ok:
        print("refusing to freeze: the balance rules fail (a tier gets more sources, never a lower bar)")
        return 1
    digest = freeze.write_freeze(plan, pairs, results, out, proto)
    print(f"froze {len(plan.items)} items into {out / freeze.MANIFEST_NAME} (sha256 {digest})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m honest_agents.eval.rts.build", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seeds", help="download and normalise the benchmark cards' datasets (network, no Docker)")
    s.add_argument("--cards", nargs="+", help="card names (default: all)")
    s.add_argument("--limit", type=int, help="seeded sample of this many tasks per card")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--index", help="seeds index (default: <cache>/index.jsonl)")
    s.add_argument("--allow-unknown-license", action="store_true")
    pr = sub.add_parser("pairs", help="applicability of every operator on every seed, and the pairs (no Docker)")
    pr.add_argument("--seeds-index", help="default: <cache>/index.jsonl")
    pr.add_argument("--benchmarks", nargs="+")
    pr.add_argument("--operators", nargs="+", help="default: every registered operator")
    pr.add_argument("--out", default=str(candidates.DEFAULT_OUT))
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--measurements", help="default: measurements.jsonl in --out")
    m = sub.add_parser("measure", help="run the seed's oracle for operators that need a measured quantity (Docker)")
    m.add_argument("--out", default=str(candidates.DEFAULT_OUT), help="the `pairs` --out dir")
    m.add_argument("--measurements", help="default: measurements.jsonl in --out")
    m.add_argument("--operators", nargs="+")
    m.add_argument("--limit", type=int, help="measure at most this many verdicts")
    m.add_argument("--concurrency", type=int, default=2, help="sandboxes at once")
    m.add_argument("--oracle-timeout", type=int, default=900)
    m.add_argument("--environment-type", default="docker")
    g = sub.add_parser("gate", help="oracle, do-nothing and blocker checks of every pair (Docker; resumable)")
    g.add_argument("--pairs", default=str(candidates.DEFAULT_OUT / candidates.PAIRS_NAME))
    g.add_argument("--out", help="results jsonl (default: gate.jsonl next to --pairs; appended)")
    g.add_argument("--operators", nargs="+")
    g.add_argument("--concurrency", type=int, default=8, help="pairs at once (each runs one sandbox at a time)")
    g.add_argument("--oracle-runs", type=int, default=3)
    g.add_argument("--oracle-timeout", type=int, default=900)
    g.add_argument("--environment-type", default="docker")
    g.add_argument("--no-resume", action="store_true")
    f = sub.add_parser("freeze", help="static checks, balance rules, then the frozen manifest")
    f.add_argument("--pairs", default=str(candidates.DEFAULT_OUT / candidates.PAIRS_NAME))
    f.add_argument("--gate", help="gate results (default: gate.jsonl next to --pairs)")
    f.add_argument("--train-tasks", nargs="+", required=True, help="training task dirs to dedup against")
    f.add_argument("--out", default=str(freeze.DEFAULT_OUT))
    f.add_argument("--protocol", help="RtsProtocol JSON (default: the pre-registered defaults)")
    f.add_argument("--report", help="default: freeze_report.json next to --pairs")
    f.add_argument("--check", action="store_true", help="write the report only; exit 1 if the freeze would refuse")
    for sp in (s, pr):
        sp.add_argument("--cache", default=str(benchmarks.DEFAULT_CACHE))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    return {"seeds": cmd_seeds, "pairs": cmd_pairs, "measure": cmd_measure, "gate": cmd_gate, "freeze": cmd_freeze}[
        args.cmd
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
