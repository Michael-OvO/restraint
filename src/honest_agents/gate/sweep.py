"""The integrity sweep, tiered and measured (docs/integrity-gate.md, "Running the gates efficiently"; DEF-15).

    uv run python -m honest_agents.gate sweep [--level 0|1|2] [--changed] [--only S1 S7 ...] [--json out.json]
        [--archive DIR] [--index-dir DIR] [--cache-dir DIR] [--variants surface_variants.jsonl] [--concurrency 4]
    uv run python -m honest_agents.gate index [--force]    # (re)build the archive index (gate.index)

* `--level 0` (default): the L0 checks (`gate.checks.L0_CHECKS`: S1, S2, S4-S10) in memory. No sandbox, no network,
  nothing written (a stale archive index is rebuilt in memory; `gate index` writes it). Budget 60 s; it also runs in
  `scripts/test.sh --fast` (tests/test_gate.py).
* `--level 1`: L0, then the oracle gate (`tasks.g1_surfaces`) over `--variants` behind the result cache: a pair boots
  only when its key (task dirs, imported sources, commands, base image digests) is not cached. The archive index and
  S8's hash cache are refreshed on disk. `--changed` checks only the pairs whose key is not cached.
* `--level 2`: as level 1, ignoring cached verdicts (every pair re-runs and its row is rewritten).

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


def sweep(
    level: int = 0,
    *,
    ctx: Context | None = None,
    only: Sequence[str] | None = None,
    changed: bool = False,
    variants: Path = DEFAULT_VARIANTS,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    concurrency: int = 4,
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
    ap.add_argument("--only", nargs="+", default=None, help="run only these checks (S1 ... S10, G1)")
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="the archive's iterations/ directory")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="the result cache (L1/L2)")
    ap.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS, help="the eval build's surface_variants.jsonl")
    ap.add_argument("--concurrency", type=int, default=4, help="live sandboxes at once (<= 4 on the Mac)")
    ap.add_argument("--json", type=Path, default=None, help="also write the results and the measurements here")
    args = ap.parse_args(argv)
    ctx = Context(archive=args.archive, index_dir=args.index_dir, write=args.level >= 1)
    t0 = time.monotonic()
    results = sweep(
        args.level,
        ctx=ctx,
        only=[s.upper() for s in args.only] if args.only else None,
        changed=args.changed,
        variants=args.variants,
        cache_dir=args.cache_dir,
        concurrency=args.concurrency,
    )
    wall = round(time.monotonic() - t0, 2)
    index = asdict(ctx.index_stats)
    print(render(results, args.level, wall, index))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"level": args.level, "wall_sec": wall, "index": index, "results": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 1 if any(r.status in (FAIL, ERROR) for r in results) else 0
