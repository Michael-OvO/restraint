"""Coverage report of a commit-check build (L0: the manifest only, in memory; nothing is written).

    uv run python -m honest_agents.commit_points.coverage [--manifest data/commit/build_manifest.json] \\
        [--eval-pools data/val.jsonl data/test.jsonl] [--eval-ids data/decision/val_high_risk_sources.txt] \\
        [--max-per-task N] [--max-task-share X] [--min-per-cell N] [--strict] [--json out.json]

Counts per commitment point x outcome x task (`schema`), and per cell the number of tasks and the largest task's share.
The quotas come from the manifest's `quota` block; a flag overrides one.

**Fails** (exit 1) when:
- a task holds more than `max_per_task` examples of one point x outcome cell, or more than `max_task_share` of the
  build (no task dominates);
- an example's task is an eval source (a `pair_id` of the eval pools or an id of the eval id files), or its split is not
  `train`: eval/val/test sources never enter training data;
- an example's decision is not the one its outcome licenses (`act` after `pass`, `hold` after `fail`);
- a row breaks the schema, or two rows share an id.

**Reported**, and failing only with `--strict`: cells under `min_per_cell` (a missing cell counts 0), and points whose
fail share is outside `fail_share` (the check should change the decision about half the time). Also reported: the
sources per point, the hazards of the destructive examples and, for claims, the checklist agreement rate with the
hidden tests (the data-quality metric).

`commit_coverage(ctx)` is the same report as a gate check (`gate.checks.CheckResult`): it skips when the build has no
manifest yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from honest_agents.commit_points.schema import (
    CLAIM,
    COMMIT_POINTS,
    DEFAULT_MANIFEST,
    DESTRUCTIVE,
    FAIL,
    OUTCOMES,
    TRAIN_SPLIT,
    BuildManifest,
    Quota,
    SchemaError,
    expected_decision,
    read_manifest,
)

DEFAULT_EVAL_POOLS = (Path("data/val.jsonl"), Path("data/test.jsonl"))
DEFAULT_EVAL_IDS = (Path("data/decision/val_high_risk_sources.txt"),)
_SHOW = 20  # violations listed per kind (all are counted)


def eval_task_ids(pools: Iterable[str | Path], id_files: Iterable[str | Path] = ()) -> set[str]:
    """Every source task of the eval pools (`pair_id`, and `source_id` where a row has one) and of the id files (one id
    per line). A missing file raises: the build cannot be shown leak-free against an eval set that is not there."""
    out: set[str] = set()
    for path in pools:
        for line in Path(path).expanduser().read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out.update(str(row[k]) for k in ("pair_id", "source_id") if row.get(k))
    for path in id_files:
        out.update(x.strip() for x in Path(path).expanduser().read_text().splitlines() if x.strip())
    return out


@dataclass
class Cell:
    commit_point: str
    outcome: str
    n: int
    tasks: int
    top_task: str | None
    top_task_n: int
    top_task_share: float | None
    under_quota: bool


@dataclass
class Coverage:
    examples: int
    tasks: int
    quota: dict[str, Any]
    cells: list[Cell]
    counts: dict[str, dict[str, dict[str, int]]]  # point -> outcome -> task -> n
    max_task_share: float | None
    max_task: str | None
    fail_share: dict[str, float | None]
    sources: dict[str, dict[str, int]]
    hazards: dict[str, int]
    checklist_agreement: dict[str, Any]
    cap_violations: list[str] = field(default_factory=list)
    share_violations: list[str] = field(default_factory=list)
    eval_leaks: list[str] = field(default_factory=list)
    split_violations: list[str] = field(default_factory=list)
    decision_mismatches: list[str] = field(default_factory=list)
    schema_problems: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    under_quota: list[str] = field(default_factory=list)
    unbalanced: list[str] = field(default_factory=list)
    eval_checked: bool = True

    def failures(self, strict: bool = False) -> dict[str, int]:
        kinds = {
            "cap": self.cap_violations,
            "share": self.share_violations,
            "eval_leak": self.eval_leaks,
            "split": self.split_violations,
            "decision": self.decision_mismatches,
            "schema": self.schema_problems,
            "duplicate_id": self.duplicate_ids,
        }
        if strict:
            kinds |= {"under_quota": self.under_quota, "unbalanced": self.unbalanced}
        return {k: len(v) for k, v in kinds.items() if v}

    def ok(self, strict: bool = False) -> bool:
        return not self.failures(strict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coverage(manifest: BuildManifest, eval_ids: set[str] | None) -> Coverage:
    """The report. `eval_ids` None skips the eval-source check (recorded as `eval_checked` False)."""
    q = manifest.quota
    rows = manifest.examples
    counts: dict[str, dict[str, Counter[str]]] = {p: {o: Counter() for o in OUTCOMES} for p in COMMIT_POINTS}
    by_task: Counter[str] = Counter()
    ids: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    hazards: Counter[str] = Counter()
    agree: Counter[str] = Counter()
    report = Coverage(0, 0, q.to_dict(), [], {}, None, None, {}, {}, {}, {}, eval_checked=eval_ids is not None)
    for e in rows:
        ids[e.id] += 1
        problems = e.problems()
        if problems:
            report.schema_problems.append(f"{e.id}: {'; '.join(problems)}")
            continue
        counts[e.commit_point][e.outcome][e.task_id] += 1
        by_task[e.task_id] += 1
        sources[e.commit_point][e.source] += 1
        if e.commit_point == DESTRUCTIVE and e.hazard:
            hazards[e.hazard] += 1
        if e.commit_point == CLAIM:
            agree["unknown" if e.checklist_agrees is None else "agrees" if e.checklist_agrees else "disagrees"] += 1
        if e.split != TRAIN_SPLIT:
            report.split_violations.append(f"{e.id}: split {e.split!r}")
        if eval_ids is not None and e.task_id in eval_ids:
            report.eval_leaks.append(f"{e.id}: task {e.task_id} is an eval source")
        if e.decision != expected_decision(e.outcome):
            report.decision_mismatches.append(f"{e.id}: decision {e.decision} after {e.outcome}")
    report.duplicate_ids = sorted(i for i, n in ids.items() if n > 1)
    total = sum(by_task.values())
    report.examples, report.tasks = total, len(by_task)
    for p in COMMIT_POINTS:
        n_point = sum(sum(counts[p][o].values()) for o in OUTCOMES)
        report.fail_share[p] = round(sum(counts[p][FAIL].values()) / n_point, 4) if n_point else None
        lo, hi = q.fail_share
        share = report.fail_share[p]
        if share is not None and not lo <= share <= hi:
            report.unbalanced.append(f"{p}: fail share {share:.3f} outside [{lo}, {hi}]")
        for o in OUTCOMES:
            c = counts[p][o]
            n = sum(c.values())
            top, top_n = c.most_common(1)[0] if c else (None, 0)
            report.cells.append(
                Cell(p, o, n, len(c), top, top_n, round(top_n / n, 4) if n else None, n < q.min_per_cell)
            )
            if n < q.min_per_cell:
                report.under_quota.append(f"{p}/{o}: {n} < {q.min_per_cell}")
            for task, k in sorted(c.items()):
                if k > q.max_per_task:
                    report.cap_violations.append(f"{p}/{o}/{task}: {k} > max_per_task {q.max_per_task}")
    report.counts = {p: {o: dict(sorted(counts[p][o].items())) for o in OUTCOMES} for p in COMMIT_POINTS}
    if total:
        report.max_task, top_n = by_task.most_common(1)[0]
        report.max_task_share = round(top_n / total, 4)
        report.share_violations = [
            f"{task}: {n}/{total} = {n / total:.4f} > max_task_share {q.max_task_share}"
            for task, n in sorted(by_task.items())
            if n / total > q.max_task_share
        ]
    report.sources = {p: dict(sorted(sources[p].items())) for p in COMMIT_POINTS}
    report.hazards = dict(sorted(hazards.items()))
    known = agree["agrees"] + agree["disagrees"]
    report.checklist_agreement = {**{k: agree[k] for k in ("agrees", "disagrees", "unknown")},
                                  "rate": round(agree["agrees"] / known, 4) if known else None}  # fmt: skip
    return report


def render(r: Coverage, strict: bool = False) -> str:
    lines = [
        f"commit-check coverage: {r.examples} examples, {r.tasks} tasks; quota {json.dumps(r.quota)}",
        "",
        "| point | outcome | examples | tasks | top task | top share | under quota |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in r.cells:
        share = "n/a" if c.top_task_share is None else f"{c.top_task_share:.3f}"
        lines.append(
            f"| {c.commit_point} | {c.outcome} | {c.n} | {c.tasks} | {c.top_task or '-'} ({c.top_task_n}) | {share} "
            f"| {'YES' if c.under_quota else 'no'} |"
        )
    lines += [
        "",
        f"max task share {r.max_task_share} ({r.max_task}); fail share per point {r.fail_share}",
        f"sources {r.sources}; hazards {r.hazards}; claim checklist agreement {r.checklist_agreement}",
    ]
    if not r.eval_checked:
        lines.append("eval-source check NOT run (--no-eval-check)")
    for name, items in (
        ("over the per-task cap", r.cap_violations),
        ("over the task share", r.share_violations),
        ("eval sources", r.eval_leaks),
        ("not split train", r.split_violations),
        ("decision not licensed by the outcome", r.decision_mismatches),
        ("schema problems", r.schema_problems),
        ("duplicate ids", r.duplicate_ids),
        ("cells under quota", r.under_quota),
        ("unbalanced points", r.unbalanced),
    ):
        if items:
            lines.append(f"{name} ({len(items)}): " + "; ".join(items[:_SHOW]) + (" ..." if len(items) > _SHOW else ""))
    failures = r.failures(strict)
    lines.append(
        f"verdict: {'PASS' if not failures else 'FAIL ' + json.dumps(failures)}" + (" (strict)" if strict else "")
    )
    return "\n".join(lines)


def with_overrides(quota: Quota, overrides: Mapping[str, Any]) -> Quota:
    return replace(quota, **{k: v for k, v in overrides.items() if v is not None})


def commit_coverage(ctx: Any = None, *, manifest: Path | None = None, repo: Path | None = None) -> Any:
    """The coverage report as a gate check (L0). Skips when the build has no manifest yet; fails as the CLI does."""
    from honest_agents.gate.checks import FAIL as GATE_FAIL
    from honest_agents.gate.checks import PASS as GATE_PASS
    from honest_agents.gate.checks import SKIP, CheckResult
    from honest_agents.gate.index import REPO

    root = repo or REPO
    path = manifest or root / DEFAULT_MANIFEST
    name = "commit-check coverage"
    if not path.is_file():
        return CheckResult("C1", name, 0, SKIP, f"no {path}")
    try:
        m = read_manifest(path)
    except (SchemaError, ValueError, TypeError) as e:
        return CheckResult("C1", name, 0, GATE_FAIL, f"{path}: {e}")
    missing = [p for p in (*DEFAULT_EVAL_POOLS, *DEFAULT_EVAL_IDS) if not (root / p).is_file()]
    if missing:
        return CheckResult("C1", name, 0, GATE_FAIL, f"cannot check eval sources: missing {[str(p) for p in missing]}")
    r = coverage(m, eval_task_ids([root / p for p in DEFAULT_EVAL_POOLS], [root / p for p in DEFAULT_EVAL_IDS]))
    failures = r.failures()
    detail = (
        f"{r.examples} examples, {r.tasks} tasks, max task share {r.max_task_share}; {len(r.under_quota)} cells under "
        f"quota" + (f"; FAIL {failures}" if failures else "")
    )
    return CheckResult("C1", name, 0, GATE_FAIL if failures else GATE_PASS, detail, data=r.to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--eval-pools", type=Path, nargs="*", default=list(DEFAULT_EVAL_POOLS))
    ap.add_argument("--eval-ids", type=Path, nargs="*", default=list(DEFAULT_EVAL_IDS))
    ap.add_argument("--no-eval-check", action="store_true", help="skip the eval-source check (reported as not run)")
    ap.add_argument("--min-per-cell", type=int, default=None)
    ap.add_argument("--max-per-task", type=int, default=None)
    ap.add_argument("--max-task-share", type=float, default=None)
    ap.add_argument("--strict", action="store_true", help="also fail on cells under quota and unbalanced points")
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    args = ap.parse_args(argv)
    try:
        manifest = read_manifest(args.manifest)
    except (OSError, SchemaError, ValueError, TypeError) as e:
        print(f"coverage: {args.manifest}: {e}", file=sys.stderr)
        return 2
    manifest.quota = with_overrides(
        manifest.quota,
        {"min_per_cell": args.min_per_cell, "max_per_task": args.max_per_task, "max_task_share": args.max_task_share},
    )
    try:
        eval_ids = None if args.no_eval_check else eval_task_ids(args.eval_pools, args.eval_ids)
    except OSError as e:
        print(f"coverage: cannot check eval sources ({e}); pass the eval pools, or --no-eval-check", file=sys.stderr)
        return 2
    report = coverage(manifest, eval_ids)
    print(render(report, args.strict))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return 0 if report.ok(args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
