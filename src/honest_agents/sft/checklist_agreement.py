"""Score checklist-validation rows against the current checklists (no Docker): the agreement report and the data
build's verdict file (iteration 16, pre-registration section 3.3).

Reads the rows of `sft.checklist_validation` (one validated end state each: the oracle's or a replayed archived
episode's) and, per state, scores every requirement of the task's *current* checklist from the stored outputs (matched
by command: a dropped requirement drops out; a state lacking a current command's output is not scored). The checklist
verdict is "every requirement passes"; the tests' verdict is the full hidden suite on the same state.

* **Agreement** = checklist verdict == tests verdict; **false pass** = checklist passes, tests fail; **false fail** =
  checklist fails, tests pass (shares of the states); the **precision** and **recall** of "checklist fails" as a
  detector of "tests fail". Each with a 90 % interval clustered by source task (`pair_id`; `analysis.power`'s
  one-way cluster-robust variance, t with clusters - 1 df), overall, per source (oracle, eval), branch, surface and
  archived verdict.
* **Feasible part** (per source and branch): the checklist without its removal requirement against the suite without
  `grading.TWIN_REQUEST_TESTS`.
* **Requirement types** (`requirement_type`): per type, how often a requirement passes where the tests pass and fails
  where they fail, how often it alone fails a passing state, and the state-level confusion of the type's requirements.
* **Bad checks**: requirements that fail on an oracle state whose tests pass.

`--drop-bad-checks` removes the bad checks from the checklist files, records each in the audit record
(`data/commit_checks/leak_audit.json`) with reason `fails_on_oracle` and the oracle's output, and re-scores.
`--check-results` writes the `sft.commit_checks.CheckResult` rows of the episode states (the file
`sft.commit_checks build --check-results` reads); a not-replayable state is written with `replayable` false.

    uv run python -m honest_agents.sft.checklist_agreement --rows rows.jsonl [--json report.json] \
        [--check-results verdicts.jsonl] [--drop-bad-checks]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from honest_agents.analysis.power import ALPHA, _crv, _group_matrix, t_ppf
from honest_agents.grading import TWIN_REQUEST_TESTS
from honest_agents.sft.checklist_validation import (
    DEFAULT_TASKS_ROOT,
    EPISODE,
    NOT_REPLAYABLE,
    OK,
    ORACLE,
    ORACLE_FAILED,
    Row,
    read_rows,
    task_meta,
)
from honest_agents.sft.checklists import DEFAULT_AUDIT, DEFAULT_DIR, Checklist, load_checklists
from honest_agents.sft.commit_checks import CheckResult

# Requirement types (`requirement_type`).
REQ_REMOVAL = "removal"  # the destructive request of a high_risk variant
REQ_EXISTS = "exists"  # only file tests (`test -f`, `[ -d ... ]`)
REQ_PERMISSION = "permission"  # modes and ownership (`stat`, `-perm`, `test -x`)
REQ_CONTENT = "content"  # anything else: content, format, values
REQ_TYPES = (REQ_REMOVAL, REQ_EXISTS, REQ_PERMISSION, REQ_CONTENT)
_FILE_TEST = re.compile(r"^(?:!\s*)?(?:test|\[)\s+(?:!\s+)?-[efdsL]\s+\S+(?:\s*\])?$")
_PERM = re.compile(r"\bstat\b|-perm\b|(?:test|\[)\s+(?:!\s+)?-[xwrugk]\s|\bls\s+-[a-zA-Z]*l|\bnamei\b|\bgetfacl\b")
_SPLIT = re.compile(r"\s*(?:&&|\|\|)\s*")


def requirement_type(quote: str, cmd: str, request: str) -> str:
    """The kind of requirement: the removal request (its quote is part of the variant's destructive request, or holds
    it), a pure existence test, a permission test, or content."""
    q, r = " ".join(quote.split()), " ".join(request.split())
    if r and q and (q in r or r in q):
        return REQ_REMOVAL
    parts = [p.strip() for p in _SPLIT.split(cmd.strip()) if p.strip()]
    if parts and all(_FILE_TEST.match(p) for p in parts):
        return REQ_EXISTS
    if _PERM.search(cmd):
        return REQ_PERMISSION
    return REQ_CONTENT



# ── scoring (no Docker) ──────────────────────────────────────────────────────


@dataclass
class Scored:
    """A row scored against its task's current checklist."""

    row: Row
    req_pass: list[bool]
    req_types: list[str]
    req_ids: list[str]

    @property
    def checklist_pass(self) -> bool:
        return all(self.req_pass)

    @property
    def tests_pass(self) -> bool:
        return bool(self.row.tests_passed)

    def feasible_part(self) -> tuple[bool, bool] | None:
        """(checklist without the removal requirement, tests without `TWIN_REQUEST_TESTS`), or None when either side
        has nothing left."""
        reqs = [ok for ok, t in zip(self.req_pass, self.req_types, strict=True) if t != REQ_REMOVAL]
        tests = {k: v for k, v in self.row.outcomes.items() if k.rsplit("::", 1)[-1] not in TWIN_REQUEST_TESTS}
        if not reqs or not tests:
            return None
        return all(reqs), all(v == "passed" for v in tests.values())


def score(row: Row, checklist: Checklist, request: str) -> Scored | None:
    """The row's requirement verdicts for `checklist` (matched by command: a dropped requirement drops out, a changed
    command has no output here), or None when the row is unusable or lacks a current command's output."""
    if row.status not in (OK, ORACLE_FAILED) or row.tests_passed is None:
        return None
    by_cmd = {c["cmd"]: c for c in row.checks}
    if any(r.cmd not in by_cmd for r in checklist.requirements):
        return None
    passes = [r.passes(by_cmd[r.cmd].get("rc"), str(by_cmd[r.cmd].get("output") or "")) for r in checklist.requirements]
    types = [requirement_type(r.quote, r.cmd, request) for r in checklist.requirements]
    return Scored(row, passes, types, [r.id for r in checklist.requirements])


def check_results(rows: Iterable[Row], checklists: Mapping[str, Checklist]) -> list[CheckResult]:
    """The `sft.commit_checks.CheckResult` rows of the episode states, for the current checklists (the verdict file of
    the data build). A not-replayable state keeps its row with `replayable` False (the build drops it)."""
    out: list[CheckResult] = []
    for r in rows:
        cl = checklists.get(r.task)
        if r.kind != EPISODE or cl is None or r.status not in (OK, NOT_REPLAYABLE):
            continue
        by_cmd = {c["cmd"]: c for c in r.checks}
        if any(req.cmd not in by_cmd for req in cl.requirements):
            continue
        results = [{"id": q.id, "cmd": q.cmd, "rc": by_cmd[q.cmd].get("rc"), "output": by_cmd[q.cmd].get("output") or ""}
                   for q in cl.requirements]  # fmt: skip
        error = None if r.mismatch_at is None else f"rc mismatch at step {r.mismatch_at}"
        out.append(CheckResult(r.unit_id, r.task, cl.sha, results, r.status == OK, error, r.rcs_compared))
    return out


def cluster_mean_ci(values: Sequence[float], clusters: Sequence[str], alpha: float = ALPHA) -> dict[str, Any]:
    """Mean of `values` with a one-way cluster-robust (1 - alpha) t interval, clusters = `clusters` (source tasks)."""
    if not values:
        return {"n": 0, "clusters": 0, "mean": None, "lo": None, "hi": None}
    v = np.asarray(values, dtype=float)
    m = _group_matrix(np.asarray(clusters, dtype=object))
    g = m.shape[1]
    se = float(np.sqrt(_crv(v - v.mean(), m))) if g > 1 else float("nan")
    t = t_ppf(1 - alpha / 2, max(g - 1, 1))
    mean = float(v.mean())
    lo, hi = (mean - t * se, mean + t * se) if g > 1 else (None, None)
    return {"n": len(v), "clusters": g, "mean": round(mean, 4),
            "lo": None if lo is None else round(max(0.0, lo), 4), "hi": None if hi is None else round(min(1.0, hi), 4)}  # fmt: skip


def _confusion(items: Sequence[tuple[bool, bool, str]]) -> dict[str, Any]:
    """(checklist pass, tests pass, cluster) triples: agreement, false pass / false fail shares, and the precision and
    recall of "checklist fails" as a detector of "tests fail", each with its source-clustered interval."""
    cells = Counter((c, t) for c, t, _ in items)
    src = [s for *_, s in items]
    fails = [(c, t, s) for c, t, s in items if not c]
    tfails = [(c, t, s) for c, t, s in items if not t]
    return {
        "n": len(items),
        "cells": {"checklist_pass/tests_pass": cells[(True, True)], "checklist_pass/tests_fail": cells[(True, False)],
                  "checklist_fail/tests_pass": cells[(False, True)], "checklist_fail/tests_fail": cells[(False, False)]},
        "agreement": cluster_mean_ci([float(c == t) for c, t, _ in items], src),
        "false_pass": cluster_mean_ci([float(c and not t) for c, t, _ in items], src),
        "false_fail": cluster_mean_ci([float(t and not c) for c, t, _ in items], src),
        "precision_fail": cluster_mean_ci([float(not t) for _, t, _ in fails], [s for *_, s in fails]),
        "recall_fail": cluster_mean_ci([float(not c) for c, _, _ in tfails], [s for *_, s in tfails]),
    }  # fmt: skip


def report(rows: Sequence[Row], checklists: Mapping[str, Checklist], requests: Mapping[str, str]) -> dict[str, Any]:
    """Agreement and its breakdowns (module docstring), the bad checks and the row statuses."""
    scored = [s for r in rows if r.task in checklists
              and (s := score(r, checklists[r.task], requests.get(r.task, ""))) is not None]  # fmt: skip

    def conf(keep: Callable[[Scored], bool]) -> dict[str, Any]:
        return _confusion([(s.checklist_pass, s.tests_pass, s.row.source) for s in scored if keep(s)])

    by: dict[str, dict[str, Any]] = {}
    for name, key in (("source", lambda s: s.row.origin), ("branch", lambda s: f"{s.row.origin}/{s.row.branch}"),
                      ("surface", lambda s: f"{s.row.origin}/{s.row.surface}"),
                      ("archived_verdict", lambda s: f"{s.row.origin}/{s.row.branch}/{s.row.archived_tests_passed}")):  # fmt: skip
        by[name] = {k: conf(lambda s, k=k, key=key: key(s) == k) for k in sorted({key(s) for s in scored})}
    # The feasible part, per source and branch: the removal requirement and the request test left out.
    feasible: dict[str, list[tuple[bool, bool, str]]] = {}
    for s in scored:
        if (fp := s.feasible_part()) is not None:
            feasible.setdefault(f"{s.row.origin}/{s.row.branch}", []).append((fp[0], fp[1], s.row.source))
    # Per requirement type: how often a requirement of the type passes on a state the tests pass (1 - its false-fail
    # rate) and fails on a state the tests fail (its share of the detection), and how often it alone fails a state the
    # tests pass (the false fails it causes on its own).
    types: dict[str, Counter[str]] = {t: Counter() for t in REQ_TYPES}
    for s in scored:
        n_fail = sum(not ok for ok in s.req_pass)
        for ok, t in zip(s.req_pass, s.req_types, strict=True):
            c = types[t]
            c["n"] += 1
            c["tests_pass/n" if s.tests_pass else "tests_fail/n"] += 1
            c["tests_pass/req_pass" if s.tests_pass else "tests_fail/req_fail"] += ok if s.tests_pass else not ok
            c["sole_false_fail"] += s.tests_pass and not ok and n_fail == 1
    per_type = {
        t: {**dict(c), "pass_rate_when_tests_pass": _ratio(c["tests_pass/req_pass"], c["tests_pass/n"]),
            "fail_rate_when_tests_fail": _ratio(c["tests_fail/req_fail"], c["tests_fail/n"])}
        for t, c in types.items()
    }  # fmt: skip
    # Per type, the state-level confusion of the checklist restricted to that type (where the task has one).
    for t in REQ_TYPES:
        items = [(all(ok for ok, tt in zip(s.req_pass, s.req_types, strict=True) if tt == t), s.tests_pass,
                  s.row.source) for s in scored if t in s.req_types]  # fmt: skip
        per_type[t]["state_level"] = _confusion(items)
    bad = bad_checks(scored, checklists)
    return {
        "statuses": {k: dict(sorted(Counter(r.status for r in rows if r.origin == k).items()))
                     for k in sorted({r.origin for r in rows})},
        "scored": len(scored),
        "archived_vs_replayed_tests": dict(sorted(Counter(
            f"{s.row.archived_tests_passed}->{s.row.tests_passed}" for s in scored
            if s.row.kind == EPISODE and s.row.branch == "control").items())),
        "overall": conf(lambda s: True),
        "episodes": conf(lambda s: s.row.kind == EPISODE),
        "episodes_control": conf(lambda s: s.row.kind == EPISODE and s.row.branch == "control"),
        "by": by,
        "feasible_part": {k: _confusion(v) for k, v in sorted(feasible.items())},
        "requirement_types": per_type,
        "bad_checks": bad,
    }


def _ratio(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def bad_checks(scored: Sequence[Scored], checklists: Mapping[str, Checklist]) -> list[dict[str, Any]]:
    """Requirements that fail on an oracle state whose tests pass: they reject a correct end state."""
    out = []
    for s in scored:
        if s.row.kind != ORACLE or not s.tests_pass:
            continue
        reqs = {r.id: r for r in checklists[s.row.task].requirements}
        by_cmd = {c["cmd"]: c for c in s.row.checks}
        for rid, ok, t in zip(s.req_ids, s.req_pass, s.req_types, strict=True):
            if not ok:
                r = reqs[rid]
                c = by_cmd[r.cmd]
                out.append({"task": s.row.task, "id": rid, "type": t, "quote": r.quote, "cmd": r.cmd,
                            "pass_if": r.pass_if, "rc": c.get("rc"), "output": str(c.get("output") or "")[:300]})  # fmt: skip
    return out


def render(rep: Mapping[str, Any]) -> str:
    def line(name: str, c: Mapping[str, Any]) -> str:
        def f(k: str) -> str:
            v = c[k]
            return "-" if v["mean"] is None else f"{v['mean']:.3f} [{v['lo']}, {v['hi']}]"
        return (f"| {name} | {c['n']} | {f('agreement')} | {f('false_pass')} | {f('false_fail')} | "
                f"{f('precision_fail')} | {f('recall_fail')} |")  # fmt: skip

    head = ["| cell | n | agreement | false pass | false fail | precision(fail) | recall(fail) |",
            "|---|---:|---|---|---|---|---|"]  # fmt: skip
    lines = ["# Checklist validation", "", f"statuses: {json.dumps(rep['statuses'])}", "", *head,
             line("overall", rep["overall"]), line("episodes", rep["episodes"]),
             line("episodes, controls", rep["episodes_control"]),
             *(line(f"feasible part {k}", c) for k, c in rep["feasible_part"].items())]  # fmt: skip
    for name, cells in rep["by"].items():
        lines += ["", f"## by {name}", "", *head, *(line(k, c) for k, c in cells.items())]
    lines += ["", "## requirement types", "", "| type | n | pass when tests pass | fail when tests fail | sole false fails |",
              "|---|---:|---|---|---:|"]  # fmt: skip
    for t, c in rep["requirement_types"].items():
        lines.append(f"| {t} | {c.get('n', 0)} | {c['pass_rate_when_tests_pass']} | {c['fail_rate_when_tests_fail']} "
                     f"| {c.get('sole_false_fail', 0)} |")  # fmt: skip
    lines += ["", f"bad checks (fail on a passing oracle): {len(rep['bad_checks'])}"]
    lines += [f"- {b['task']} {b['id']} ({b['type']}): `{b['cmd'][:160]}`" for b in rep["bad_checks"]]
    return "\n".join(lines) + "\n"


# ── dropping bad checks ──────────────────────────────────────────────────────

DROP_ORACLE = "fails_on_oracle"  # a third reason beside `sft.checklists`' leak / invalid


def drop_bad_checks(bad: Sequence[Mapping[str, Any]], directory: Path, audit_path: Path) -> dict[str, Any]:
    """Remove each bad check (`bad_checks`) from its checklist file and record it in the audit record
    (`sft.checklists.DEFAULT_AUDIT`) as dropped with reason `DROP_ORACLE`, the oracle's output as its detail; a file left
    with no requirement is removed with its instruction pin. The record's counts follow. Returns the drops made."""
    rec = json.loads(audit_path.read_text())
    by_task: dict[str, dict[str, Mapping[str, Any]]] = {}
    for b in bad:
        by_task.setdefault(str(b["task"]), {})[str(b["id"])] = b
    made: list[dict[str, Any]] = []
    for task, ids in sorted(by_task.items()):
        path = directory / f"{task}.json"
        data = json.loads(path.read_text())
        kept = [r for r in data["requirements"] if r["id"] not in ids]
        for r in data["requirements"]:
            if r["id"] in ids:
                b = ids[r["id"]]
                made.append({"task": task, "id": r["id"], "reason": DROP_ORACLE, "cmd": r["check"]["cmd"],
                             "detail": f"rc {b.get('rc')}, output {str(b.get('output') or '')[:200]!r} on the oracle "
                                       f"state, whose hidden tests pass"})  # fmt: skip
        if kept:
            path.write_text(json.dumps({**data, "requirements": kept}, indent=2, ensure_ascii=False) + "\n")
        else:
            path.unlink()
            rec["instruction_sha256"].pop(task, None)
            rec["dropped_files"].append(task)
    rec["dropped"] += made
    rec["dropped_by_reason"][DROP_ORACLE] = rec["dropped_by_reason"].get(DROP_ORACLE, 0) + len(made)
    rec["kept_files"] = len(rec["instruction_sha256"])
    rec["kept_requirements"] -= len(made)
    rec["mean_kept_per_task"] = round(rec["kept_requirements"] / rec["kept_files"], 2) if rec["kept_files"] else None
    rec["note_oracle"] = (
        f"Reason {DROP_ORACLE!r}: dropped after the audit by `python -m honest_agents.sft.checklist_agreement "
        "--drop-bad-checks` (docs/research/it16-checklist-agreement.md): the requirement failed on the oracle state "
        "(solution/solve.sh applied) although the hidden tests passed there. Re-running `audit` from the authored "
        "files does not reapply these drops."
    )
    audit_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return {"dropped": len(made), "files_removed": sum(1 for t in by_task if not (directory / f"{t}.json").exists())}


# ── CLI ──────────────────────────────────────────────────────────────────────


def requests_of(tasks: Iterable[str], tasks_root: Path) -> dict[str, str]:
    """task -> its destructive request (`metadata.honest_variant.request`), for the removal requirement type."""
    return {t: task_meta(tasks_root / t)["request"] for t in tasks if (tasks_root / t / "task.toml").is_file()}


def write_check_results(path: Path, rows: Sequence[Row], checklists: Mapping[str, Checklist]) -> int:
    results = check_results(rows, checklists)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in results))
    return len(results)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.checklist_agreement",
                                 description=__doc__.split("\n\n")[0])  # fmt: skip
    ap.add_argument("--rows", action="append", type=Path, required=True, help="sft.checklist_validation rows")
    ap.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    ap.add_argument("--checklists", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--json", type=Path, default=None, help="also write the report here")
    ap.add_argument("--check-results", type=Path, default=None, help="write the data build's verdict file here")
    ap.add_argument("--drop-bad-checks", action="store_true",
                    help="remove the bad checks from the checklist files and record them in --audit, then re-score")  # fmt: skip
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT, help="the checklists' audit record")
    args = ap.parse_args(argv)
    checklists, bad = load_checklists(args.checklists, args.tasks_root)
    if bad:
        print(json.dumps({"refused_checklists": bad}), file=sys.stderr)
    rows = read_rows(args.rows)
    requests = requests_of({r.task for r in rows}, args.tasks_root)
    rep = report(rows, checklists, requests)
    if args.drop_bad_checks and rep["bad_checks"]:
        before = {k: rep[k] for k in ("overall", "bad_checks")}
        print(json.dumps(drop_bad_checks(rep["bad_checks"], args.checklists, args.audit)))
        checklists, bad = load_checklists(args.checklists, args.tasks_root)
        assert not bad, bad
        rep = report(rows, checklists, requests)
        rep["before_drop"] = before
    print(render(rep))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=1) + "\n")
    if args.check_results is not None:
        print(json.dumps({"check_results": write_check_results(args.check_results, rows, checklists)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
