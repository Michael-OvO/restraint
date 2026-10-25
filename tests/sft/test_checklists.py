"""Instruction-only requirement checklists (iteration 16, `sft.checklists`): the schema, the leak audit on synthetic task
dirs, and an L0 check of every committed file under data/commit_checks/checklists against its audit record (and, when
the decision surface tasks are on disk, against the instructions and hidden tests themselves)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from honest_agents.sft import checklists as cl

REPO = Path(__file__).resolve().parents[2]
CHECKLISTS = REPO / cl.DEFAULT_DIR
AUDIT = REPO / cl.DEFAULT_AUDIT
TASKS_ROOT = Path(os.environ.get("HONEST_DECISION_TASKS", REPO / cl.DEFAULT_TASKS_ROOT))

INSTRUCTION = """Create `/home/user/out/report.txt` with one line per host.

The archive is called 2024-logs.tar.gz and lives in /home/user/api\\_archives.
Remove `/srv/ember` when you are done.
"""
TESTS = """REPORT = "/home/user/out/report.txt"
SECRET = "/home/user/out/.hidden_marker"
def test_count():
    assert open(REPORT).read().count("\\n") == 17
    assert "ABORTED" in open("/home/user/api_archives/2024-logs.tar.gz", "rb").read().decode()
"""


def req(rid: str, cmd: str, quote: str = "Remove `/srv/ember` when you are done.", pass_if: str = "rc==0") -> dict:
    return {"id": rid, "quote": quote, "check": {"cmd": cmd, "pass_if": pass_if}}


def checklist(task: str, *reqs: dict) -> dict:
    return {"task": task, "source": "endless-terminals", "requirements": list(reqs), "author": "claude",
            "inputs": ["instruction.md"]}  # fmt: skip


def task_dir(root: Path, task: str, instruction: str = INSTRUCTION, tests: str = TESTS) -> Path:
    d = root / task
    (d / "tests").mkdir(parents=True)
    (d / "instruction.md").write_text(instruction)
    (d / "tests" / "test_final_state.py").write_text(tests)
    return d


# ── schema ───────────────────────────────────────────────────────────────────


def test_parse_accepts_the_format_and_checks_quotes_against_the_instruction():
    data = checklist("t", req("r1", "test ! -e /srv/ember"), req("r2", "cat /x", pass_if="stdout_contains:ok"))
    c = cl.parse_checklist(data, INSTRUCTION)
    assert [r.id for r in c.requirements] == ["r1", "r2"] and c.commands == ["test ! -e /srv/ember", "cat /x"]
    # whitespace-normalised verbatim match; without an instruction the quote is not checked
    assert cl.parse_checklist(checklist("t", req("r1", "true", quote="Remove  `/srv/ember`\nwhen")), INSTRUCTION)
    with pytest.raises(cl.ChecklistError, match="not verbatim"):
        cl.parse_checklist(checklist("t", req("r1", "true", quote="Remove /srv/ember")), INSTRUCTION)
    assert cl.parse_checklist(checklist("t", req("r1", "true", quote="anything")))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(extra=1), "keys"),
        (lambda d: d.update(author="human"), "author"),
        (lambda d: d.update(inputs=["instruction.md", "tests/test_final_state.py"]), "instruction.md only"),
        (lambda d: d.update(requirements=[]), "no requirements"),
        (lambda d: d["requirements"][0].update(destructive=True), "malformed requirement"),
        (lambda d: d["requirements"][0]["check"].update(timeout=5), "malformed check"),
        (lambda d: d["requirements"][0]["check"].update(pass_if="rc==1"), "pass_if"),
        (lambda d: d["requirements"][0]["check"].update(pass_if="stdout_contains:"), "pass_if"),
        (lambda d: d["requirements"][0]["check"].update(cmd="  "), "empty command"),
        (lambda d: d["requirements"][0]["check"].update(cmd="rm -rf /srv/ember"), "not read-only"),
        (lambda d: d["requirements"][0]["check"].update(cmd="echo x > /tmp/y"), "not read-only"),
        (lambda d: d["requirements"].append(dict(d["requirements"][0])), "duplicate"),
    ],
)
def test_parse_refuses_a_file_that_breaks_the_schema(mutate, match):
    data = checklist("t", req("r1", "test ! -e /srv/ember"))
    mutate(data)
    with pytest.raises(cl.ChecklistError, match=match):
        cl.parse_checklist(data)


def test_load_checks_the_file_name_and_the_quotes_when_the_task_is_on_disk(tmp_path):
    root, d = tmp_path / "tasks", tmp_path / "cl"
    task_dir(root, "t1")
    d.mkdir()
    (d / "t1.json").write_text(json.dumps(checklist("t1", req("r1", "true"))))
    (d / "t2.json").write_text(json.dumps(checklist("t1", req("r1", "true"))))
    (d / "t3.json").write_text("{")
    good, bad = cl.load_checklists(d, root)
    assert list(good) == ["t1"] and good["t1"].sha == hashlib.sha256((d / "t1.json").read_bytes()).hexdigest()[:16]
    assert "names task" in bad["t2.json"] and "not JSON" in bad["t3.json"]
    (d / "t1.json").write_text(json.dumps(checklist("t1", req("r1", "true", quote="not in the instruction"))))
    assert "not verbatim" in cl.load_checklists(d, root)[1]["t1.json"]
    assert "t1" in cl.load_checklists(d, None)[0]  # no task dir: schema only


def test_pass_if():
    def r(pass_if: str) -> cl.Requirement:
        return cl.Requirement("r", "q", "c", pass_if)

    assert r("rc==0").passes(0, "") and not r("rc==0").passes(1, "")
    assert r("stdout_empty").passes(1, " \n") and not r("stdout_empty").passes(None, "")
    assert r("stdout_contains:OK").passes(3, "all OK") and not r("stdout_contains:OK").passes(0, "ok")


# ── leak audit ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cmd, leaked",
    [
        ("test -f /home/user/out/.hidden_marker", ["hidden_marker"]),  # a path only the tests name
        ('test "$(wc -l < /home/user/out/report.txt)" -eq 17', ["17"]),  # a magic value only the tests hold
        ("test -f /home/user/out/report.txt", []),  # every part is in the instruction
        ("tar -tzf /home/user/api_archives/2024-logs.tar.gz", []),  # assembled from the instruction, `\_` unescaped
        ("python3 -c \"import json,sqlite3;print(json.load(open('/home/user/out/report.txt')))\"", []),  # vocabulary
        ("grep -q aborted /home/user/out/report.txt", ["aborted"]),  # the tests' `ABORTED`: case-insensitive
        ("grep -qx 'x\\nabort' /home/user/out/report.txt", []),  # escapes are blanked, `abort` is in neither
        ("grep -cE '[0-9a-f]{2}' /home/user/out/report.txt", []),  # ranges and small numbers are generic
    ],
)
def test_leaked_strings(cmd, leaked):
    r = cl.Requirement("r1", "q", cmd, "rc==0")
    assert cl.leaked_strings(r, cl.instruction_text(INSTRUCTION), TESTS.casefold()) == leaked


def test_a_number_matches_whole_and_a_word_case_insensitively():
    tests = "assert n == 170 and status == 'aborted'"
    ins = cl.instruction_text("There are 1700 rows; count the ABORTED ones.")
    assert cl.leaked_strings(cl.Requirement("r", "q", "test $n -eq 170 && grep -q aborted f", "rc==0"), ins, tests) == [
        "170"
    ]


def test_audit_drops_leaking_and_invalid_requirements_and_records_why(tmp_path):
    root, src, out = tmp_path / "tasks", tmp_path / "src", tmp_path / "out"
    task_dir(root, "t1")
    task_dir(root, "t2")
    src.mkdir()
    ok = req("r1", "test ! -e /srv/ember") | {"destructive": True}  # an authoring mark: stripped, not a failure
    leak = req("r2", "test -f /home/user/out/.hidden_marker")
    mutating = req("r3", "grep -q x /home/user/out/report.txt > /tmp/seen")
    unquoted = req("r4", "true", quote="Remove it")
    (src / "t1.json").write_text(json.dumps(checklist("t1", ok, leak, mutating, unquoted)))
    (src / "t2.json").write_text(json.dumps(checklist("t2", leak)))
    (src / "t3.json").write_text(json.dumps(checklist("t3", ok)))  # no task dir
    rec = cl.audit(src, out, root)
    assert rec["files"] == 3 and list(rec["file_failures"]) == ["t3.json"]
    assert rec["requirements"] == 5 and rec["dropped_by_reason"] == {"leak": 2, "invalid": 2}
    assert rec["leak_rate"] == 0.4 and rec["stripped_keys"] == {"destructive": 1}
    assert rec["kept_files"] == 1 and rec["kept_requirements"] == 1 and rec["dropped_files"] == ["t2"]
    assert {(d["task"], d["id"], d["reason"]) for d in rec["dropped"]} == {
        ("t1", "r2", "leak"), ("t1", "r3", "invalid"), ("t1", "r4", "invalid"), ("t2", "r2", "leak")}  # fmt: skip
    assert next(d for d in rec["dropped"] if d["id"] == "r2")["leaked"] == ["hidden_marker"]
    assert sorted(p.name for p in out.iterdir()) == ["t1.json"]
    kept = cl.load_checklist(out / "t1.json", root)
    assert [r.id for r in kept.requirements] == ["r1"]
    assert rec["instruction_sha256"] == {"t1": hashlib.sha256(INSTRUCTION.encode()).hexdigest()}


# ── the committed checklists (L0) ────────────────────────────────────────────


def _record() -> dict:
    return json.loads(AUDIT.read_text())


def test_every_committed_checklist_passes_the_schema():
    good, bad = cl.load_checklists(CHECKLISTS)
    assert not bad
    rec = _record()
    assert len(good) == rec["kept_files"] == len(list(CHECKLISTS.glob("*.json"))) and not rec["file_failures"]
    assert sum(len(c.requirements) for c in good.values()) == rec["kept_requirements"]
    assert set(good) == set(rec["instruction_sha256"])
    gone = {(d["task"], d["id"]) for d in rec["dropped"]}
    assert not {(t, r.id) for t, c in good.items() for r in c.requirements} & gone


@pytest.mark.skipif(not TASKS_ROOT.is_dir(), reason="needs the decision surface tasks (data/decision/surfaces/tasks)")
def test_committed_checklists_match_their_instructions_and_do_not_leak():
    good, bad = cl.load_checklists(CHECKLISTS, TASKS_ROOT)  # quotes verbatim
    assert not bad
    pins = _record()["instruction_sha256"]
    for task, c in good.items():
        d = TASKS_ROOT / task
        assert hashlib.sha256((d / "instruction.md").read_bytes()).hexdigest() == pins[task], (
            f"{task}: instruction changed"
        )
        ins, tests = cl.instruction_text((d / "instruction.md").read_text()), cl.tests_text(d)
        assert not [r.id for r in c.requirements if cl.leaked_strings(r, ins, tests)], task
