"""Coverage report of a commit-check build (L0: the manifest only, in memory; nothing is written).

    uv run python -m honest_agents.commit_points.coverage [--manifest data/commit/build_manifest.json] [--data-dir data] \\
        [--cap-per-task N] [--max-source-share X] [--max-task-share X] [--min-per-cell N] [--max-outcome-ratio R] [--strict] [--json out.json]

Counts per commitment point x outcome x task (`schema`), and per cell the number of tasks and the largest task's share.
The quotas come from the manifest's `quota` block; a flag overrides one.

**Fails** (exit 1) when:
- a task holds more than `cap_per_task` examples of one point x outcome cell, more than `max_task_share` of the build
  (when set), or a source task more than `max_source_share` of it (no task dominates);
- an example's point is not one of the quota's `points` (the points the build trains);
- an example's source is in `tasks.select_split.excluded_from_commit_training(--data-dir)` (select, val and test
  sources, and the final eval's decision-probe sources: DEF-63), or its recorded split is not `train`: eval sources
  never enter training data;
- an example's decision is not the one its outcome licenses (committed after `pass`, not after `fail`);
- a row breaks the schema, or two rows share an id.

**Reported**, and failing only with `--strict`: cells of the quota's `points` under `min_per_cell` (a missing cell counts
0), and points whose larger outcome exceeds `max_outcome_ratio` times the smaller (the check should change the decision about half the time).
Also reported: the origins per point, the hazards of the destructive examples and, for claims, the checklist agreement
rate with the hidden tests (the data-quality metric).

`commit_coverage(ctx)` is the same report as a gate check (`gate.checks.CheckResult`, id C1): it skips when the build
has no manifest yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from honest_agents.commit_points.schema import (
    CLAIM,
    DEFAULT_MANIFEST,
    DESTRUCTIVE,
    FAIL,
    OUTCOMES,
    PASS,
    POINTS,
    TRAIN_SPLIT,
    BuildManifest,
    Quota,
    SchemaError,
    licensed,
    read_manifest,
)

_SHOW = 20  # violations listed per kind (all are counted)


@dataclass
class Cell:
    point: str
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
    sources: int
    quota: dict[str, Any]
    cells: list[Cell] = field(default_factory=list)
    counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)  # point -> outcome -> task -> n
    max_source_share: float | None = None
    max_source: str | None = None
    fail_share: dict[str, float | None] = field(default_factory=dict)
    origins: dict[str, dict[str, int]] = field(default_factory=dict)
    hazards: dict[str, int] = field(default_factory=dict)
    checklist_agreement: dict[str, Any] = field(default_factory=dict)
    cap_violations: list[str] = field(default_factory=list)
    share_violations: list[str] = field(default_factory=list)
    task_share_violations: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    excluded_sources: list[str] = field(default_factory=list)
    split_violations: list[str] = field(default_factory=list)
    decision_mismatches: list[str] = field(default_factory=list)
    schema_problems: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    under_quota: list[str] = field(default_factory=list)
    unbalanced: list[str] = field(default_factory=list)
    excluded_checked: bool = True

    def failures(self, strict: bool = False) -> dict[str, int]:
        kinds = {
            "cap": self.cap_violations,
            "share": self.share_violations,
            "task_share": self.task_share_violations,
            "out_of_scope": self.out_of_scope,
            "excluded_source": self.excluded_sources,
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


def coverage(manifest: BuildManifest, excluded: set[str] | None) -> Coverage:
    """The report. `excluded` None skips the excluded-source check (recorded as `excluded_checked` False)."""
    q = manifest.quota
    counts: dict[str, dict[str, Counter[str]]] = {p: {o: Counter() for o in OUTCOMES} for p in POINTS}
    by_source: Counter[str] = Counter()
    by_task: Counter[str] = Counter()
    tasks: set[str] = set()
    ids: Counter[str] = Counter()
    origins: dict[str, Counter[str]] = defaultdict(Counter)
    hazards: Counter[str] = Counter()
    agree: Counter[str] = Counter()
    r = Coverage(0, 0, 0, q.to_dict(), excluded_checked=excluded is not None)
    for e in manifest.examples:
        ids[e.id] += 1
        problems = e.problems()
        if problems:
            r.schema_problems.append(f"{e.id}: {'; '.join(problems)}")
            continue
        if e.point not in q.points:
            r.out_of_scope.append(f"{e.id}: point {e.point} is not one of the build's points {list(q.points)}")
        counts[e.point][e.outcome][e.task] += 1
        by_source[e.source] += 1
        by_task[e.task] += 1
        tasks.add(e.task)
        origins[e.point][e.origin] += 1
        if e.point == DESTRUCTIVE and e.hazard:
            hazards[e.hazard] += 1
        if e.point == CLAIM:
            agree["unknown" if e.checklist_agrees is None else "agrees" if e.checklist_agrees else "disagrees"] += 1
        if e.split is not None and e.split != TRAIN_SPLIT:
            r.split_violations.append(f"{e.id}: split {e.split!r}")
        if excluded is not None and e.source in excluded:
            r.excluded_sources.append(f"{e.id}: source {e.source} is excluded from training")
        if e.committed != licensed(e.outcome):
            r.decision_mismatches.append(f"{e.id}: committed={e.committed} after {e.outcome}")
    r.duplicate_ids = sorted(i for i, n in ids.items() if n > 1)
    total = sum(by_source.values())
    r.examples, r.tasks, r.sources = total, len(tasks), len(by_source)
    for p in POINTS:
        n_out = {o: sum(counts[p][o].values()) for o in OUTCOMES}
        n_point = sum(n_out.values())
        r.fail_share[p] = round(n_out[FAIL] / n_point, 4) if n_point else None
        small, large = sorted((n_out[PASS], n_out[FAIL]))
        if p in q.points and n_point and large > q.max_outcome_ratio * small:
            r.unbalanced.append(f"{p}: pass {n_out[PASS]} / fail {n_out[FAIL]} exceeds ratio {q.max_outcome_ratio}")
        for o in OUTCOMES:
            c = counts[p][o]
            n = n_out[o]
            top, top_n = c.most_common(1)[0] if c else (None, 0)
            share = round(top_n / n, 4) if n else None
            under = p in q.points and n < q.min_per_cell
            r.cells.append(Cell(p, o, n, len(c), top, top_n, share, under))
            if under:
                r.under_quota.append(f"{p}/{o}: {n} < {q.min_per_cell}")
            for task, k in sorted(c.items()):
                if k > q.cap_per_task:
                    r.cap_violations.append(f"{p}/{o}/{task}: {k} > cap_per_task {q.cap_per_task}")
    r.counts = {p: {o: dict(sorted(counts[p][o].items())) for o in OUTCOMES} for p in POINTS}
    if total:
        r.max_source, top_n = by_source.most_common(1)[0]
        r.max_source_share = round(top_n / total, 4)
        r.share_violations = [
            f"{source}: {n}/{total} = {n / total:.4f} > max_source_share {q.max_source_share}"
            for source, n in sorted(by_source.items())
            if n / total > q.max_source_share
        ]
        if q.max_task_share is not None:
            r.task_share_violations = [
                f"{task}: {n}/{total} = {n / total:.4f} > max_task_share {q.max_task_share}"
                for task, n in sorted(by_task.items())
                if n / total > q.max_task_share
            ]
    r.origins = {p: dict(sorted(origins[p].items())) for p in POINTS}
    r.hazards = dict(sorted(hazards.items()))
    known = agree["agrees"] + agree["disagrees"]
    r.checklist_agreement = {k: agree[k] for k in ("agrees", "disagrees", "unknown")}
    r.checklist_agreement["rate"] = round(agree["agrees"] / known, 4) if known else None
    return r


def render(r: Coverage, strict: bool = False) -> str:
    lines = [
        f"commit-check coverage: {r.examples} examples, {r.tasks} tasks, {r.sources} sources; quota {json.dumps(r.quota)}",
        "",
        "| point | outcome | examples | tasks | top task | top share | under quota |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in r.cells:
        share = "n/a" if c.top_task_share is None else f"{c.top_task_share:.3f}"
        lines.append(
            f"| {c.point} | {c.outcome} | {c.n} | {c.tasks} | {c.top_task or '-'} ({c.top_task_n}) | {share} "
            f"| {'YES' if c.under_quota else 'no'} |"
        )
    lines += [
        "",
        f"max source share {r.max_source_share} ({r.max_source}); fail share per point {r.fail_share}",
        f"origins {r.origins}; hazards {r.hazards}; claim checklist agreement {r.checklist_agreement}",
    ]
    if not r.excluded_checked:
        lines.append("excluded-source check NOT run (--no-excluded-check)")
    for name, items in (
        ("over the per-task cap", r.cap_violations),
        ("over the source share", r.share_violations),
        ("over the task share", r.task_share_violations),
        ("points the build does not train", r.out_of_scope),
        ("sources excluded from training", r.excluded_sources),
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
    verdict = "PASS" if not failures else "FAIL " + json.dumps(failures)
    lines.append(f"verdict: {verdict}" + (" (strict)" if strict else ""))
    return "\n".join(lines)


def with_overrides(quota: Quota, overrides: Mapping[str, Any]) -> Quota:
    return replace(quota, **{k: v for k, v in overrides.items() if v is not None})


def excluded_sources(data_dir: Path) -> set[str]:
    """`tasks.select_split.excluded_from_commit_training`: raises when the select file, the split seeds or the final
    decision probe are missing."""
    from honest_agents.tasks.select_split import excluded_from_commit_training

    return excluded_from_commit_training(data_dir)


def commit_coverage(ctx: Any = None, *, manifest: Path | None = None, repo: Path | None = None) -> Any:
    """The coverage report as a gate check (L0). Skips when the build has no manifest yet; fails as the CLI does, and
    when the excluded sources cannot be read (a manifest that cannot be shown leak-free does not pass)."""
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
    try:
        excluded = excluded_sources(root / "data")
    except OSError as e:
        return CheckResult("C1", name, 0, GATE_FAIL, f"cannot read the excluded sources: {e}")
    r = coverage(m, excluded)
    failures = r.failures()
    detail = (
        f"{r.examples} examples, {r.tasks} tasks, max source share {r.max_source_share}; {len(r.under_quota)} cells "
        f"under quota" + (f"; FAIL {failures}" if failures else "")
    )
    return CheckResult("C1", name, 0, GATE_FAIL if failures else GATE_PASS, detail, data=r.to_dict())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data-dir", type=Path, default=Path("data"), help="select_split's data dir (excluded sources)")
    ap.add_argument("--no-excluded-check", action="store_true", help="skip the excluded-source check (reported)")
    ap.add_argument("--cap-per-task", type=int, default=None)
    ap.add_argument("--max-source-share", type=float, default=None)
    ap.add_argument("--max-task-share", type=float, default=None)
    ap.add_argument("--min-per-cell", type=int, default=None)
    ap.add_argument("--max-outcome-ratio", type=float, default=None)
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
        {
            "cap_per_task": args.cap_per_task,
            "max_source_share": args.max_source_share,
            "max_task_share": args.max_task_share,
            "min_per_cell": args.min_per_cell,
            "max_outcome_ratio": args.max_outcome_ratio,
        },
    )
    try:
        excluded = None if args.no_excluded_check else excluded_sources(args.data_dir)
    except OSError as e:
        print(
            f"coverage: cannot read the excluded sources ({e}); pass --data-dir, or --no-excluded-check",
            file=sys.stderr,
        )
        return 2
    report = coverage(manifest, excluded)
    print(render(report, args.strict))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return 0 if report.ok(args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
