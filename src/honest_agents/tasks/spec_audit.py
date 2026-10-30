"""Spec audit: find tasks whose hidden tests do not follow from their instruction, and keep a reviewed verdict for each.

    python -m honest_agents.tasks.spec_audit scan [--archive DIR] [--out data/spec_audit/candidates.json]
    python -m honest_agents.tasks.spec_audit check [--candidates ...] [--reviews ...] [--ledger docs/defects.md]
    python -m honest_agents.tasks.spec_audit defective [--reviews ...]

The oracle gate cannot see this defect: the oracle is written to the test's reading, so it passes a task whose
instruction contradicts itself or never states what the test asserts (docs/iteration-19-research/badcase.md: 36 of
G2's 95 false claims on the select split, 7 sources, all oracle-valid). A static comparison of the test's literals
with the instruction was tried and rejected: test files are full of code literals (median 4 "unstated" per source over
2,492 feasible dirs) and it caught none of the contradictions (2026-10-30, docs/iteration-19-proposal.md).

The audit is behavioural, then human:

* **scan** counts, per source, every graded feasible episode in every archived eval (any model, any variant): a source
  is a candidate when nobody passes it reliably and the failures are near misses (`RULE`). That is the signature of a
  test that asks for something the instruction does not say: agents do the task and lose on one detail. On the 7 known
  defects the rule flags 5; the two it misses are contradictions that agents sometimes resolve the test's way.
* **reviews** (`data/spec_audit/reviews.jsonl`, one row per reviewed source) hold the verdict of reading the
  instruction against the tests: `defect` (with its kind, the evidence and the ledger ID) or `ok`. Sources can be
  reviewed without being candidates.
* **check** (integrity sweep S12, L0) fails while a candidate has no review, a defect names no ledger entry or one that
  is not in the ledger, a row is malformed, or the archive holds more eval runs than the candidates were scanned from.
* **defective** prints the sources reviewed as defects, for pool builders and analyses to exclude.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = Path.home() / "Documents" / "honest-agents-archive" / "iterations"
AUDIT_DIR = REPO / "data" / "spec_audit"
CANDIDATES = AUDIT_DIR / "candidates.json"
REVIEWS = AUDIT_DIR / "reviews.jsonl"
LEDGER = REPO / "docs" / "defects.md"
EPISODE_GLOBS = ("*/eval/*/episodes.jsonl.gz", "*/eval/*/episodes.jsonl")
VERDICTS = ("defect", "ok")
KINDS = ("contradiction", "hidden_requirement", "ambiguous", "other")
_SOURCE = re.compile(r"(endless-terminals)__(task_\d+_[0-9a-f]{8})")
_DEF_ID = re.compile(r"^DEF-[A-Za-z0-9-]+$")


@dataclass(frozen=True)
class Rule:
    """A candidate: at least `min_episodes` graded feasible episodes, a pass rate of at most `max_pass_rate`, and a
    median pass fraction of at least `min_median_fail_fraction` over the failing ones."""

    min_episodes: int = 8
    max_pass_rate: float = 0.2
    min_median_fail_fraction: float = 0.6


RULE = Rule()


@dataclass
class SourceStats:
    source: str
    episodes: int = 0
    passes: int = 0
    fail_fractions: list[float] = field(default_factory=list)
    runs: set[str] = field(default_factory=set)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.episodes if self.episodes else 0.0

    @property
    def median_fail_fraction(self) -> float | None:
        return statistics.median(self.fail_fractions) if self.fail_fractions else None

    def row(self) -> dict[str, Any]:
        mf = self.median_fail_fraction
        return {"source": self.source, "episodes": self.episodes, "passes": self.passes,
                "pass_rate": round(self.pass_rate, 4), "median_fail_fraction": None if mf is None else round(mf, 4),
                "runs": len(self.runs)}  # fmt: skip


def source_of(task_id: str) -> str | None:
    """`endless-terminals/task_000000_1978f062` for any variant id of that source."""
    m = _SOURCE.search(task_id)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def episode_files(archive: Path) -> list[Path]:
    return sorted({p for g in EPISODE_GLOBS for p in archive.glob(g)})


def _read(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def graded_feasible(e: dict[str, Any]) -> bool:
    o = e.get("outcome") or {}
    return (
        bool(o.get("feasible"))
        and o.get("tests_passed") is not None
        and not o.get("grading_error")
        and not (e.get("infra_error") or (e.get("extra") or {}).get("infra_error"))
    )


def collect(files: Iterable[Path]) -> dict[str, SourceStats]:
    stats: dict[str, SourceStats] = {}
    for path in files:
        run = f"{path.parent.parent.parent.name}/{path.parent.name}"
        for e in _read(path):
            src = source_of(str(e.get("task_id", "")))
            if src is None or not graded_feasible(e):
                continue
            s = stats.setdefault(src, SourceStats(src))
            s.episodes += 1
            s.runs.add(run)
            o = e["outcome"]
            if o["tests_passed"]:
                s.passes += 1
            elif o.get("pass_fraction") is not None:
                s.fail_fractions.append(float(o["pass_fraction"]))
    return stats


def is_candidate(s: SourceStats, rule: Rule = RULE) -> bool:
    mf = s.median_fail_fraction
    return (
        s.episodes >= rule.min_episodes
        and s.pass_rate <= rule.max_pass_rate
        and mf is not None
        and mf >= rule.min_median_fail_fraction
    )


def scan(archive: Path, rule: Rule = RULE) -> dict[str, Any]:
    files = episode_files(archive)
    stats = collect(files)
    cands = sorted((s for s in stats.values() if is_candidate(s, rule)), key=lambda s: (s.pass_rate, s.source))
    return {"rule": asdict(rule), "eval_runs": len(files), "sources_seen": len(stats),
            "candidates": [s.row() for s in cands]}  # fmt: skip


def load_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def review_problems(reviews: Sequence[dict[str, Any]], ledger_ids: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for i, r in enumerate(reviews, 1):
        src, verdict = r.get("source"), r.get("verdict")
        where = f"reviews line {i} ({src})"
        if not isinstance(src, str) or source_of(src.replace("/", "__")) != src:
            out.append(f"{where}: `source` must look like endless-terminals/task_000000_1978f062")
        if src in seen:
            out.append(f"{where}: a second review of the same source")
        seen.add(src)
        if verdict not in VERDICTS:
            out.append(f"{where}: verdict {verdict!r} is not one of {VERDICTS}")
        if not str(r.get("evidence") or "").strip():
            out.append(f"{where}: no evidence")
        if verdict == "defect":
            if r.get("kind") not in KINDS:
                out.append(f"{where}: kind {r.get('kind')!r} is not one of {KINDS}")
            did = r.get("def_id")
            if not isinstance(did, str) or not _DEF_ID.match(did):
                out.append(f"{where}: a defect needs `def_id` (a ledger ID)")
            elif did not in ledger_ids:
                out.append(f"{where}: {did} is not in the ledger")
    return out


def ledger_ids(text: str) -> set[str]:
    return set(re.findall(r"^###\s+(DEF-[A-Za-z0-9-]+)\s*:", text, re.MULTILINE))


def check(candidates: dict[str, Any], reviews: Sequence[dict[str, Any]], ledger_text: str,
          archive_runs: int | None = None) -> list[str]:  # fmt: skip
    """Every problem that keeps the audit open (empty means S12 passes)."""
    out = review_problems(reviews, ledger_ids(ledger_text))
    reviewed = {r.get("source") for r in reviews}
    out += [f"candidate {c['source']} has no review" for c in candidates.get("candidates", [])
            if c["source"] not in reviewed]  # fmt: skip
    if archive_runs is not None and archive_runs > int(candidates.get("eval_runs", 0)):
        out.append(f"the archive holds {archive_runs} eval runs, the candidates were scanned from "
                   f"{candidates.get('eval_runs', 0)}: re-run `spec_audit scan`")  # fmt: skip
    return out


def defective_sources(reviews_path: Path = REVIEWS) -> set[str]:
    return {r["source"] for r in load_reviews(reviews_path) if r.get("verdict") == "defect"}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.tasks.spec_audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="flag candidate sources from every archived eval")
    s.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    s.add_argument("--out", type=Path, default=CANDIDATES)
    c = sub.add_parser("check", help="exit 0 only when every candidate is reviewed and every defect is in the ledger")
    c.add_argument("--candidates", type=Path, default=CANDIDATES)
    c.add_argument("--reviews", type=Path, default=REVIEWS)
    c.add_argument("--ledger", type=Path, default=LEDGER)
    c.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="checked for runs newer than the scan")
    d = sub.add_parser("defective", help="print the sources reviewed as defects")
    d.add_argument("--reviews", type=Path, default=REVIEWS)
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        if not a.archive.is_dir():
            print(f"error: no archive at {a.archive}", file=sys.stderr)
            return 1
        doc = scan(a.archive)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{len(doc['candidates'])} candidates of {doc['sources_seen']} sources from {doc['eval_runs']} eval "
              f"runs -> {a.out}")  # fmt: skip
        return 0
    if a.cmd == "defective":
        print("\n".join(sorted(defective_sources(a.reviews))))
        return 0
    runs = len(episode_files(a.archive)) if a.archive.is_dir() else None
    problems = check(json.loads(a.candidates.read_text()), load_reviews(a.reviews), a.ledger.read_text(), runs)
    for p in problems:
        print(f"  problem: {p}", file=sys.stderr)
    print(f"spec audit: {'PASS' if not problems else f'{len(problems)} problem(s)'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
