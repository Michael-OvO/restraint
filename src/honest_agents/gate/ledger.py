"""The integrity gate's rule as a launch guard: no experiment starts while any defect in the ledger is open.

    python -m honest_agents.gate.ledger check [--ledger docs/defects.md] [--json FILE]

docs/integrity-gate.md "The rule": an open defect is any entry of docs/defects.md whose status is not `closed` or
`waived`. The ledger states each status twice, in the summary table (`| DEF-10 | stats | fixing | ... |`) and in the
entry's own `**Status:**` line (`### DEF-10: ...`). Both are read, and the check fails closed:

* an entry is open when either status does not start with `closed` or `waived` (`fixing (awaiting ...)` is open);
* an ID in the table with no `###` entry, an entry with no table row, an entry whose status cannot be found, a
  duplicate ID or a ledger with no entry at all is a problem, reported like an open entry.

Exit 0 only when every entry is closed or waived and nothing is malformed; exit 1 otherwise (the open entries and the
problems are printed). Every iteration-16 experiment job file opens with this check as its first step
(`configs/chains/it16_*.yaml`), so a chain launched while the gate is shut stops before anything is served or trained.
A placeholder ID of a new defect (`DEF-NEW-<lane>-<n>`) counts like any other.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_LEDGER = Path("docs/defects.md")
NOT_OPEN = ("closed", "waived")
_ID = r"DEF-[A-Za-z0-9-]+"
_ROW = re.compile(rf"^\|\s*({_ID})\s*\|[^|]*\|\s*([^|]*?)\s*\|")
_HEADING = re.compile(rf"^###\s+({_ID})\s*:")
_STATUS = re.compile(r"\*\*Status:\*\*\s*(.+)")


@dataclass
class LedgerVerdict:
    entries: int = 0
    open: dict[str, str] = field(default_factory=dict)  # id -> the status that keeps it open
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.entries > 0 and not self.open and not self.problems


def is_open(status: str) -> bool:
    return not status.strip().lower().startswith(NOT_OPEN)


def parse(text: str) -> tuple[dict[str, str], dict[str, str | None], list[str]]:
    """(table statuses, entry statuses, problems). An entry's status is the first `**Status:**` after its heading,
    before the next heading; None when there is none."""
    table: dict[str, str] = {}
    entries: dict[str, str | None] = {}
    problems: list[str] = []
    current: str | None = None
    for line in text.splitlines():
        if m := _ROW.match(line):
            if m.group(1) in table:
                problems.append(f"{m.group(1)}: two rows in the summary table")
            table[m.group(1)] = m.group(2)
            continue
        if line.startswith("#"):
            current = None
            if m := _HEADING.match(line):
                current = m.group(1)
                if current in entries:
                    problems.append(f"{current}: two entries")
                entries[current] = None
            continue
        if current is not None and entries[current] is None and (m := _STATUS.search(line)):
            entries[current] = m.group(1).strip()
    return table, entries, problems


def check(text: str) -> LedgerVerdict:
    table, entries, problems = parse(text)
    verdict = LedgerVerdict(entries=len(set(table) | set(entries)), problems=list(problems))
    if not verdict.entries:
        verdict.problems.append("no defect entry found: not a ledger, or an unreadable one")
    for did in sorted(set(table) | set(entries)):
        row, entry = table.get(did), entries.get(did)
        if row is None:
            verdict.problems.append(f"{did}: an entry with no row in the summary table")
        if did not in entries:
            verdict.problems.append(f"{did}: a table row with no `### {did}` entry")
        elif entry is None:
            verdict.problems.append(f"{did}: the entry has no **Status:** line")
        statuses = [s for s in (row, entry) if s is not None]
        if any(is_open(s) for s in statuses):
            verdict.open[did] = next(s for s in statuses if is_open(s))
    return verdict


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.gate.ledger", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="exit 0 only when every ledger entry is closed or waived")
    c.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    c.add_argument("--json", type=Path, default=None, help="also write the verdict as JSON")
    a = ap.parse_args(argv)
    try:
        text = a.ledger.read_text()
    except OSError as e:
        print(f"error: cannot read the ledger {a.ledger} ({e.strerror}); the gate stays shut", file=sys.stderr)
        return 1
    verdict = check(text)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({**asdict(verdict), "ok": verdict.ok, "ledger": str(a.ledger)}, indent=1))
    if verdict.ok:
        print(f"ledger {a.ledger}: all {verdict.entries} entries closed or waived; the gate is open")
        return 0
    print(
        f"ledger {a.ledger}: the integrity gate is SHUT (docs/integrity-gate.md); no experiment may start",
        file=sys.stderr,
    )
    for did, status in verdict.open.items():
        print(f"  open: {did}: {status}", file=sys.stderr)
    for p in verdict.problems:
        print(f"  problem: {p}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
