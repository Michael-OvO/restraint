"""`gate.ledger`: the integrity gate's rule (no experiment while any defect is open) as the first step of every
iteration-16 job file. It reads both statuses of each entry and fails closed on anything it cannot read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.gate import ledger

REPO = Path(__file__).resolve().parent.parent


def _ledger(rows: list[tuple[str, str]], entries: list[tuple[str, str | None]]) -> str:
    table = "\n".join(f"| {i} | stats | {s} | one line |" for i, s in rows)
    body = "\n\n".join(
        f"### {i}: a defect\n- **Area:** stats."
        + (f" **Status:** {s}. **Found:** 2026-10-23." if s else " **Found:** x.")
        for i, s in entries
    )
    return f"# Defect ledger\n\n| ID | Area | Status | One line |\n|---|---|---|---|\n{table}\n\n---\n\n{body}\n"


def test_all_closed_or_waived_for_the_experiment_opens_the_gate() -> None:
    text = _ledger([("DEF-01", "closed"), ("DEF-02", "waived")],
                   [("DEF-01", "closed 2026-10-24"), ("DEF-02", "waived 2026-10-25, owner; covers it16a")])  # fmt: skip
    v = ledger.check(text, "it16a")
    assert v.ok and v.entries == 2 and not v.open and not v.problems


@pytest.mark.parametrize("experiment", ["it16d", "it16", "it16a2", "it17a", None])
def test_a_waiver_covers_only_the_experiments_it_names(experiment: str | None) -> None:
    """integrity-gate.md: a waiver names the experiments it covers, so a defect waived for stage (a) keeps (d) shut."""
    text = _ledger([("DEF-02", "waived (covers it16a, it16b)")],
                   [("DEF-02", "waived 2026-10-25 by the owner (covers it16a, it16b)")])  # fmt: skip
    assert ledger.check(text, "it16b").ok
    v = ledger.check(text, experiment)
    assert not v.ok and list(v.open) == ["DEF-02"] and "does not name experiment" in v.open["DEF-02"]


@pytest.mark.parametrize(
    "row, entry",
    [("fixing", "fixing (tool done)"), ("open", "open"),
     ("fixing (awaiting owner decision on releasing the 24 select variants)", "fixing"),
     ("closed", "fixing (the table was updated first)"), ("fixing", "closed 2026-10-24")],
    ids=["fixing", "open", "fixing with a note", "entry still fixing", "table still fixing"],
)  # fmt: skip
def test_an_entry_is_open_when_either_status_is_not_closed(row: str, entry: str) -> None:
    v = ledger.check(_ledger([("DEF-01", "closed"), ("DEF-10", row)], [("DEF-01", "closed"), ("DEF-10", entry)]))
    assert not v.ok and list(v.open) == ["DEF-10"]


def test_a_placeholder_id_of_a_new_defect_counts() -> None:
    v = ledger.check(_ledger([("DEF-NEW-it16-jobs-1", "open")], [("DEF-NEW-it16-jobs-1", "open")]))
    assert list(v.open) == ["DEF-NEW-it16-jobs-1"]


@pytest.mark.parametrize(
    "rows, entries, problem",
    [([("DEF-01", "closed"), ("DEF-02", "closed")], [("DEF-01", "closed")], "no `### DEF-02` entry"),
     ([("DEF-01", "closed")], [("DEF-01", "closed"), ("DEF-02", "closed")], "DEF-02: an entry with no row"),
     ([("DEF-01", "closed")], [("DEF-01", None)], "no **Status:** line"),
     ([("DEF-01", "closed"), ("DEF-01", "closed")], [("DEF-01", "closed")], "two rows"),
     ([], [], "no defect entry")],
    ids=["entry missing", "row missing", "status missing", "duplicate row", "empty"],
)  # fmt: skip
def test_a_ledger_it_cannot_read_keeps_the_gate_shut(rows, entries, problem: str) -> None:
    v = ledger.check(_ledger(rows, entries))
    assert not v.ok and any(problem in p for p in v.problems), v.problems


def test_the_cli_exits_1_while_the_gate_is_shut_and_0_when_it_opens(tmp_path, capsys) -> None:
    shut = tmp_path / "shut.md"
    shut.write_text(_ledger([("DEF-10", "fixing")], [("DEF-10", "fixing (the pre-registration is a DRAFT)")]))
    assert ledger.main(["check", "--ledger", str(shut), "--json", str(tmp_path / "v.json")]) == 1
    err = capsys.readouterr().err
    assert "SHUT" in err and "open: DEF-10: fixing" in err
    assert json.loads((tmp_path / "v.json").read_text())["ok"] is False
    shut.write_text(_ledger([("DEF-10", "closed")], [("DEF-10", "closed 2026-10-27")]))
    assert ledger.main(["check", "--ledger", str(shut)]) == 0
    assert ledger.main(["check", "--ledger", str(tmp_path / "missing.md")]) == 1  # unreadable: shut
    shut.write_text(_ledger([("DEF-10", "waived (covers it16a)")], [("DEF-10", "waived 2026-10-27 (covers it16a)")]))
    assert ledger.main(["check", "--ledger", str(shut), "--experiment", "it16d"]) == 1
    assert ledger.main(["check", "--ledger", str(shut), "--experiment", "it16a"]) == 0


def test_the_real_ledger_is_readable() -> None:
    """Every entry of docs/defects.md has a table row and a status, so the gate's verdict is its open entries alone."""
    v = ledger.check((REPO / "docs" / "defects.md").read_text())
    assert v.problems == [] and v.entries >= 57
