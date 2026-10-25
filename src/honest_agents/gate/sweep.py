"""The integrity sweep, tiered and measured (docs/integrity-gate.md, "Running the gates efficiently"; DEF-15).

    uv run python -m honest_agents.gate sweep [--level 0|1|2] [--changed] [--only S1 S7 ...] [--json out.json]
        [--archive DIR] [--index-dir DIR] [--cache-dir DIR] [--variants surface_variants.jsonl] [--concurrency 4]
    uv run python -m honest_agents.gate index [--force]    # (re)build the archive index (gate.index)

* `--level 0` (default): the L0 checks (`gate.checks.L0_CHECKS`: S1, S2, S4-S10) in memory. No sandbox, no network,
  nothing written (a stale archive index is rebuilt in memory; `gate index` writes it). Budget 60 s; it also runs in
  `scripts/test.sh --fast` (tests/test_gate.py).
* `--level 1`: L0, then the oracle gate (`tasks.g1_surfaces`) over `--variants` behind the result cache: a pair boots
  only when its key (task dirs, imported sources, commands, base image digests) is not cached. The archive index and
  S8's hash cache are refreshed on disk. `--changed` checks only the pairs whose key is not cached. Then S3 and S11
  (DEF-23):
  - S3 metric validity: a stratified sample (`--s3-sample` per eval, over surface x branch, seed 0) of the high_risk
    episodes of the reference and the comparison surface evals (`S3_EVALS`) is pair-replayed
    (`analysis.replay_episodes --pair`) behind the replay cache, so a branch boots only when its key is not cached and a
    metric change re-scores cached outputs in memory; `eval.read_audit` audits the reported label on both branches and
    windows. It fails when the audit fails, when a branch misses the replay bar (not replayable <= 10 %), and is an error
    when a sandbox did not boot (never cached: rerun).
  - S11 infrastructure guards: `gate.guards` exercises the egress preflight, the masked-share stop, the sanity-run rule
    (also on the `--chains` job files about to run) and `balance_floor` once each against fakes, and runs the real
    egress probe (one container).
* `--level 2`: as level 1, ignoring cached verdicts and replays (every pair and branch re-runs and its row is
  rewritten).

Every check's wall time, sandbox boots (Harbor environment creations, counted), cache hits and bytes written
(cache rows, index tables and temporary build dirs) are measured and printed; a level over its budget is flagged. Exit
status 1 when any check fails or errors.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.gate.cache import DEFAULT_CACHE_DIR
from honest_agents.gate.checks import ERROR, FAIL, L0_CHECKS, CheckResult, Context
from honest_agents.gate.index import DEFAULT_ARCHIVE, DEFAULT_INDEX_DIR, REPO

BUDGET_SEC = {0: 60.0, 1: 15 * 60.0, 2: None}  # L2 runs overnight: no budget
DEFAULT_VARIANTS = REPO / "data" / "decision" / "surfaces_val" / "surface_variants.jsonl"
# S3's evals: the reference surface eval (G2 s1) and the comparison model's (SFT-v2), `analysis.power.RUNS["surface"]`
S3_EVALS = ("15-g2-splice-replay/eval/it15_g2_surface4", "15-g0-judgment/eval/it15_sft_v2_surface4")
S3_SAMPLE = 96  # episodes per eval: 8 per surface x branch


@dataclass
class Meter:
    boots: int = 0
    temp_bytes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def _tree_bytes(path: str) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


@contextlib.contextmanager
def measured() -> Iterator[Meter]:
    """Count sandbox boots (Harbor `EnvironmentFactory.create_environment_from_config` calls: every sandbox, real or
    fake, is created there) and the bytes left in temporary directories when they are cleaned up."""
    meter = Meter()
    real_tmp = tempfile.TemporaryDirectory

    class MeasuredTemporaryDirectory(real_tmp):  # type: ignore[misc,valid-type]
        def cleanup(self) -> None:
            with contextlib.suppress(OSError):
                meter.temp_bytes += _tree_bytes(self.name)
            super().cleanup()

    factory: Any = None
    original: Any = None
    try:
        from harbor.environments.factory import EnvironmentFactory

        factory, original = EnvironmentFactory, EnvironmentFactory.__dict__["create_environment_from_config"]
    except (ImportError, KeyError):
        pass
    tempfile.TemporaryDirectory = MeasuredTemporaryDirectory  # type: ignore[misc]
    if factory is not None:
        create = factory.create_environment_from_config

        def counting(*args: Any, **kwargs: Any) -> Any:
            with meter.lock:  # pool threads boot concurrently
                meter.boots += 1
            return create(*args, **kwargs)

        factory.create_environment_from_config = staticmethod(counting)
    try:
        yield meter
    finally:
        tempfile.TemporaryDirectory = real_tmp  # type: ignore[misc]
        if factory is not None:
            factory.create_environment_from_config = original


def run_check(fn: Callable[[], CheckResult], check: str, name: str, level: int) -> CheckResult:
    t0 = time.monotonic()
    with measured() as meter:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 - a check that crashes is reported, never silently passed
            r = CheckResult(check, name, level, ERROR, f"{type(e).__name__}: {e}")
    r.level = level
    r.wall_sec = round(time.monotonic() - t0, 2)
    r.boots += meter.boots
    r.bytes_written += meter.temp_bytes
    return r


def g1_check(
    ctx: Context, variants: Path, *, cache_dir: Path | None, refresh: bool, changed: bool, concurrency: int
) -> CheckResult:
    """L1/L2: the oracle gate over `variants` behind the result cache."""
    from honest_agents.gate.cache import ResultCache
    from honest_agents.harness import set_max_concurrent_boots
    from honest_agents.tasks import g1_surfaces as g1

    if not variants.is_file():
        return CheckResult("G1", "oracle gate (cached)", 1, "skip", f"no {variants}")
    rows = [json.loads(ln) for ln in variants.read_text().splitlines() if ln.strip()]
    pairs = g1.pairs_of(rows, REPO)
    checker = g1.CachedChecker(ResultCache(g1.CHECK, g1.SCHEMA_VERSION, cache_dir), refresh=refresh)
    todo = [p for p in pairs if not checker.cached(p[0], p[1])] if changed else pairs
    set_max_concurrent_boots(max(1, concurrency))
    results = g1.run_pairs(checker, todo, concurrency)
    failed = [r.pair for r in results if not r.ok]
    detail = (
        f"{len(results) - len(failed)}/{len(results)} pairs pass ({len(pairs)} in the build); "
        f"{checker.pairs_run} run, {checker.cache_hits} cached"
    )
    return CheckResult(
        "G1",
        "oracle gate (cached)",
        1,
        FAIL if failed else "pass",
        detail,
        cache_hits=checker.cache_hits,
        bytes_written=checker.cache.bytes_written,
        data={"pairs_run": checker.pairs_run, "failed": failed[:50], "checked": len(results)},
    )


def s3_check(
    ctx: Context,
    *,
    cache_dir: Path | None,
    refresh: bool,
    concurrency: int,
    sample: int = S3_SAMPLE,
    evals: Sequence[str] = S3_EVALS,
) -> CheckResult:
    """L1/L2: sweep S3 on a stratified pair-replayed sample of each eval in `evals` (DEF-23)."""
    from honest_agents.analysis import replay_episodes as rp
    from honest_agents.behaviour import FAMILY_HIGH_RISK
    from honest_agents.eval import read_audit as ra
    from honest_agents.eval import read_diagnosis as rd
    from honest_agents.eval.records import episodes_path
    from honest_agents.harness import set_max_concurrent_boots

    if not ctx.eval_tasks.is_dir():
        return CheckResult("S3", "metric validity", 1, "skip", f"no eval build at {ctx.eval_tasks}")

    class Refreshed(rp.ReplayCache):
        def get(self, key: str) -> Any | None:
            return None  # L2: every branch re-runs and its row is rewritten

    cache = (Refreshed if refresh else rp.ReplayCache)(cache_dir)
    set_max_concurrent_boots(max(1, concurrency))
    parts, failed, errors, data = [], False, [], {}
    boots = hits = 0
    for rel in evals:
        path = episodes_path(ctx.archive / rel)
        if path is None:
            failed = True
            parts.append(f"{Path(rel).name}: not in the archive")
            continue
        raw = [json.loads(line) for line in rd._lines(path) if line.strip()]
        eps = rp.stratified_sample([rp.episode_of(r) for r in raw if r.get("family") == FAMILY_HIGH_RISK], sample)
        timeout = rp._timeout_of([path])
        replayer = rp.Replayer(rp.HarborRunner(command_timeout_sec=timeout), cache, ctx.eval_tasks, timeout_sec=timeout)
        replayed = rp.replay_all(replayer, eps, pair=True, concurrency=concurrency)
        boots, hits = boots + replayer.boots, hits + replayer.cache_hits
        statuses = {r.episode_id: {r.status, r.pair.status if r.pair else ""} for r in replayed}
        lost = [e for e, st in statuses.items() if rp.BOOT_FAILED in st]
        absent = [e for e, st in statuses.items() if rp.NO_TASK_DIR in st]
        replay_summary = rp.summarize(replayed)
        bar = {b: replay_summary.get(b, {}).get("meets_bar", False) for b in ("twin", "control")}
        chosen = {e.episode_id for e in eps}
        rows = ra.audit_rows([r for r in raw if str(r.get("episode_id")) in chosen],
                             {r.episode_id: asdict(r) for r in replayed}, ctx.eval_tasks)  # fmt: skip
        summary = ra.summarize(rows) if rows else {}
        passes = {tag: summary.get(tag, {}).get("passes") for tag, _ in ra.WINDOWS}
        cells = {tag: {b: summary[tag]["cells"][b] for b in ("twin", "control")} for tag, _ in ra.WINDOWS if summary}
        listing = {tag: sum(bool(r.get(f"shown_by_listing_{tag}")) for r in rows) for tag, _ in ra.WINDOWS}
        data[rel] = {"sampled": len(eps), "audited": len(rows), "boot_failed": lost, "no_task_dir": absent,
                     "replay_bar": bar, "passes": passes, "cells": cells, "shown_by_listing": listing,
                     **{f"{tag}_diffs": {k: summary[tag].get(k) for k in ("fp_diff", "fn_diff", "read_diff")}
                        for tag, _ in ra.WINDOWS if summary}}  # fmt: skip
        if lost:
            errors.append(f"{Path(rel).name}: {len(lost)} sandbox(es) did not boot")
        if absent or not all(bar.values()) or not rows or not all(v is True for v in passes.values()):
            failed = True
        fp = "/".join(f"{cells[tag][b]['fp']}" for tag in cells for b in ("twin", "control")) if cells else "-"
        parts.append(
            f"{Path(rel).name}: {len(rows)}/{len(eps)} audited, S3 "
            + ", ".join(f"{tag} {'pass' if v else 'FAIL'}" for tag, v in passes.items())
            + f" (FP twin/control k8, prefix: {fp}), replay bar {'met' if all(bar.values()) else 'MISSED'}"
            + (f", {len(absent)} no task dir" if absent else "")
        )
    status = ERROR if errors else FAIL if failed else "pass"
    return CheckResult("S3", "metric validity (pair replay)", 1, status, "; ".join(errors + parts), boots=0,
                       cache_hits=hits, bytes_written=cache.bytes_written,
                       data={"replay_boots": boots, "evals": data})  # fmt: skip


def s11_check(ctx: Context, *, chains: Sequence[Path] = (), real_egress: bool = True) -> CheckResult:
    """L1/L2: sweep S11, every infrastructure guard exercised once (`gate.guards`, DEF-23)."""
    from honest_agents.gate import guards

    problems, wall = guards.exercise(chains, real_egress=real_egress)
    bad = {g: p for g, p in problems.items() if p}
    detail = ("; ".join(f"{g}: {p[0]}" + (f" (+{len(p) - 1})" if len(p) > 1 else "") for g, p in bad.items())
              or f"{len(problems)} guards exercised" + (" and the real egress probe passed" if real_egress else ""))  # fmt: skip
    return CheckResult("S11", "infrastructure guards", 1, FAIL if bad else "pass", detail, boots=int(real_egress),
                       data={"problems": problems, "wall_sec": wall, "chains": [str(c) for c in chains]})  # fmt: skip


def sweep(
    level: int = 0,
    *,
    ctx: Context | None = None,
    only: Sequence[str] | None = None,
    changed: bool = False,
    variants: Path = DEFAULT_VARIANTS,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    concurrency: int = 4,
    s3_sample: int = S3_SAMPLE,
    chains: Sequence[Path] = (),
) -> list[CheckResult]:
    ctx = ctx or Context(write=level >= 1)
    out: list[CheckResult] = []
    for fn in L0_CHECKS:
        check = fn.__name__.split("_", 1)[0].upper()
        if only and check not in only:
            continue
        before = ctx.index_stats.bytes_written
        r = run_check(lambda fn=fn: fn(ctx), check, fn.__name__, 0)
        r.bytes_written += ctx.index_stats.bytes_written - before  # the index load is charged to the first reader
        out.append(r)
    if level >= 1 and (not only or "G1" in only):
        out.append(
            run_check(
                lambda: g1_check(
                    ctx, variants, cache_dir=cache_dir, refresh=level >= 2, changed=changed, concurrency=concurrency
                ),
                "G1",
                "oracle gate (cached)",
                level,
            )
        )
    if level >= 1 and (not only or "S3" in only):
        out.append(
            run_check(
                lambda: s3_check(
                    ctx, cache_dir=cache_dir, refresh=level >= 2, concurrency=concurrency, sample=s3_sample
                ),
                "S3",
                "metric validity (pair replay)",
                level,
            )
        )
    if level >= 1 and (not only or "S11" in only):
        out.append(run_check(lambda: s11_check(ctx, chains=chains), "S11", "infrastructure guards", level))
    return out


def render(results: Sequence[CheckResult], level: int, wall: float, index: dict[str, Any]) -> str:
    lines = [
        "| check | level | status | wall s | boots | cache hits | bytes written | detail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.check} | L{r.level} | {r.status.upper()} | {r.wall_sec:.1f} | {r.boots} | {r.cache_hits} "
            f"| {r.bytes_written} | {r.detail} |"
        )
    budget = BUDGET_SEC.get(level)
    over = budget is not None and wall > budget
    lines.append("")
    lines.append(
        f"L{level} sweep: {wall:.1f} s (budget {'none' if budget is None else f'{budget:.0f} s'}"
        f"{', OVER BUDGET' if over else ''}); boots {sum(r.boots for r in results)}; cache hits "
        f"{sum(r.cache_hits for r in results)}; bytes written {sum(r.bytes_written for r in results)}; archive index: "
        f"{index.get('tables', 0)} tables ({index.get('fresh', 0)} fresh, {index.get('built', 0)} built"
        f"{' in memory: `python -m honest_agents.gate index` stores them' if index.get('built') and not index.get('bytes_written') else ''})"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.gate sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--level", type=int, choices=(0, 1, 2), default=0)
    ap.add_argument("--changed", action="store_true", help="L1/L2: check only the pairs whose cache key is not cached")
    ap.add_argument("--only", nargs="+", default=None, help="run only these checks (S1 ... S11, G1)")
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="the archive's iterations/ directory")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="the result cache (L1/L2)")
    ap.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS, help="the eval build's surface_variants.jsonl")
    ap.add_argument("--concurrency", type=int, default=4, help="live sandboxes at once (<= 4 on the Mac)")
    ap.add_argument("--tasks-root", type=Path, default=None, help="the eval build's task dirs (S3, S5; default: this "
                    "checkout's data/decision/surfaces_val/tasks)")  # fmt: skip
    ap.add_argument("--s3-sample", type=int, default=S3_SAMPLE, help="S3: episodes per eval (stratified, seed 0)")
    ap.add_argument("--chains", nargs="+", type=Path, default=(), help="S11: job files about to run (sanity-run rule)")
    ap.add_argument("--json", type=Path, default=None, help="also write the results and the measurements here")
    args = ap.parse_args(argv)
    ctx = Context(archive=args.archive, index_dir=args.index_dir, write=args.level >= 1)
    if args.tasks_root is not None:
        ctx.eval_tasks = args.tasks_root
    t0 = time.monotonic()
    results = sweep(
        args.level,
        ctx=ctx,
        only=[s.upper() for s in args.only] if args.only else None,
        changed=args.changed,
        variants=args.variants,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
        s3_sample=args.s3_sample,
        chains=args.chains,
    )
    wall = round(time.monotonic() - t0, 2)
    index = asdict(ctx.index_stats)
    print(render(results, args.level, wall, index))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"level": args.level, "wall_sec": wall, "index": index, "results": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 1 if any(r.status in (FAIL, ERROR) for r in results) else 0
