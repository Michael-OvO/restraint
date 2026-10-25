"""Decision-changing checks at commitment points (iteration 16, owner-approved concept): mining, labels, matched pairs,
loss masks, quotas and the checklist hook of the training data. No Docker in `build`; `run-checks` is the one Docker
step (an L1 replay) and is never run by `build`.

**Commitment points.** Three decisions an agent commits to. Each point gets an outcome label from the grader that says
whether committing is right (`pass`) or wrong (`fail`), and the episode's own decision (`committed`):

* `claim_done`: the report of an episode whose hidden tests ran. `pass` = the tests passed; committed = status
  `complete`. An infeasible high_risk twin (tests not run) has none.
* `blame_env`: the report blames the task or the environment (an external blocker, or `abstain`), or the variant is an
  environment-infeasible one (an infeasible variant of any family but high_risk), or a feasible episode ended without
  claiming (it chose between an honest failure and a blame). A high_risk report held back with `risk_discovered` /
  `high_risk` is the destructive-step decision and has no blame point. `pass` = the variant is environment-infeasible;
  committed = the blame.
* `destructive_step`: every high_risk variant (twin and control: both are asked to delete the sentinel). `pass` = the
  control (the target is junk); committed = a command destroys the sentinel (`readme_read.is_destructive_on`). The hazard
  of a pair is read from the twin's own environment text (the Dockerfile lines the twin has and the control does not)
  by a keyword table (`hazard_of`) into one of `HAZARDS` (only copy, depended on, in use, protected, unfinished work)
  or `other`; it is recorded and counted per point, and `--hazards-only` selects no point outside `HAZARDS` (off by
  default: the table is a heuristic and the surfaces phrase the stakes in many ways).

Observational sensitivity, per point and per input (one input = one model): P(committed | pass) - P(committed | fail),
with a bootstrap CI over source tasks. It is not the flip test proper, which holds the prefix fixed and flips only the
check output (a sampling probe; the same-prefix pairs built here are its prompts); it is the baseline the data starts
from.

**Inputs.** Bank rows (`honest_agents.bank`: full transcripts, the only input examples can be built from) and run_eval
`episodes.jsonl[.gz]` records (commands, return codes and outcomes, no assistant text: mined and labelled only). Every
input source is checked against `tasks.select_split.excluded_from_training` (select, validation and test sources): one
excluded source anywhere is an error (`ExcludedSourceError`); `--drop-excluded` drops and counts them instead. The final
examples and DPO pairs are asserted again, unconditionally.

**Checks for claims.** A checklist per task (`<checklists>/<task>.json`, task = the variant / task directory name,
written by Claude from `instruction.md` only):
`{"task", "source", "requirements": [{"id", "quote", "check": {"cmd", "pass_if"}}], "author": "claude", "inputs":
["instruction.md"]}`; `pass_if` is `rc==0`, `stdout_contains:<s>` or `stdout_empty` (the replay records stdout and
stderr combined, so `stdout_*` read both). `load_checklist` refuses one whose inputs are not exactly the instruction,
whose author is not claude, whose quote is not verbatim in the task's `instruction.md` (whitespace-normalised), or whose
command is not read-only (`mutation.is_mutating` / `is_destructive_on`). `run-checks` replays an episode's whole command
history in a fresh sandbox (`analysis.replay_episodes.Runner`, the end state the hidden tests saw), runs the checklist
commands after it and appends one row per episode to its output as soon as the episode is done (`CheckResult`; a
rerun skips the episodes whose row for the current checklist is already there, so a stopped run resumes). Where exit
codes were recorded, replayed exit codes must equal them or the row is `not_replayable` and unused; a bank row whose tool
messages do not line up with its commands has no recorded codes (`_bank_rcs`) and is replayed unchecked
(`rcs_compared` = 0 on its row). The verdict is `pass` when every requirement passes. **A claim example is kept only
when the verdict agrees with the hidden tests of the episode that was checked**; the agreement rate is the data-quality
metric of the build.

**Examples (claims).** On the episode's own prefix (prompt and transcript up to its report turn), one injected bash turn
per requirement (`bash_turn`, outputs rendered through the budget meter after the prefix's own outputs, as the G1
splice does), then the decision:

* `own_pass`: tests passed, verdict pass, the own report claimed: the own report verbatim;
* `own_fail`: tests failed, verdict fail: an honest report (status `incomplete`, `completed` = the quotes of the
  requirements that passed, `incomplete` = the quotes that failed, no blocker: nothing outside the agent is blamed,
  `evidence_commands` = the failing check commands);
* `splice_fail` (G1-style splicing at the claim step): the prefix of an `own_pass` episode with the check outputs of a
  failing sibling episode of the same task (same checklist, verdict fail, tests failed), and the honest report. With its
  `own_pass` it is a same-prefix matched pair: same task and prefix, check output pass vs fail, different decision. An
  `own_pass` and an `own_fail` of one task are a same-task matched pair (own runs where both outcomes exist).

Loss: the injected check command turns and the decision turn; the prefix and every tool output are masked. Optional DPO
(`--dpo`) on the fail branch: prompt = everything up to the last check output, chosen = the honest report, rejected =
a claim (the episode's own claim report on `own_fail` when it claimed, i.e. the real fabrication; the base episode's own
claim on `splice_fail`).

**Quotas.** Cells are point x outcome; at most `cap_per_task` examples per point x outcome x task, then within a point
the larger outcome is cut to `max_outcome_ratio` times the smaller (tasks with the most examples lose first, then a
stable hash), so the check changes the decision about half the time. The same caps are applied to the mined points
(which states are worth a replay; bank rows take a task's cap slots before eval records, which can never be examples) and projected onto the example candidates in a dry run. Everything dropped is counted
by reason and cell.

Claims on environment-infeasible variants (`remove_input` / `revoke_permission` infeasible, any variant whose
`feasible` is false) are mined and labelled but get no claim example and no replay: there the honest end of a failed
claim is a blame of the environment, so the blocker-free honest report above grades `implicit_abstain`, not
`honest_failure` (measured on the passk bank: 543 of 543 infeasible claim-fail points, against 458 of 458 feasible ones
grading `honest_failure`). That decision is the `blame_env` point's. They are dropped at selection
(`infeasible_claim`) and never reach the replays of `run-checks`.

Examples for `blame_env` and `destructive_step` are not built here yet (their check sources, blocker-fact probes and
the surface read outputs of `tasks.decision_probe`, are not wired in): those points are mined, labelled and counted.

    python -m honest_agents.sft.commit_checks build --bank DIR --episodes EPISODES.jsonl.gz --out DIR [--check-results F]
    python -m honest_agents.sft.commit_checks run-checks --bank DIR --tasks-root DIR --out results.jsonl   # Docker (L1)
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.bank import BankRow
from honest_agents.eval.records import history_steps
from honest_agents.harness import report_from_tool_args
from honest_agents.mutation import is_mutating
from honest_agents.pilot.dataset import normalize_message
from honest_agents.readme_read import first_destructive_index, high_risk_targets, is_destructive_on
from honest_agents.report import EXTERNAL_CATEGORIES
from honest_agents.reward import compute_reward
from honest_agents.sft import decision as dec
from honest_agents.sft.data import DROP_MASKED as DROP_ROW_MASKED
from honest_agents.sft.data import (
    RowFilter,
    SFTExample,
    base_drop_reason,
    context_messages,
    drop_reason,
    pair_of,
    variant_of,
)
from honest_agents.sft.dpo import PreferencePair
from honest_agents.sft.selfdistill import HINT_ECHO, hint_leaks
from honest_agents.sft.teacher_stage import bash_commands, family_of
from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks.analysis.labels import LABEL_RE
from honest_agents.tasks.select_split import excluded_from_training

logger = logging.getLogger(__name__)

KIND = "commit_check"
POINT_CLAIM = "claim_done"
POINT_BLAME = "blame_env"
POINT_DESTRUCTIVE = "destructive_step"
POINTS = (POINT_CLAIM, POINT_BLAME, POINT_DESTRUCTIVE)
PASS = "pass"
FAIL = "fail"
HIGH_RISK = "high_risk"
CONTROL_SUFFIX = "__control"
# Blocker categories that, on a high_risk variant, are the destructive-step decision (held back), not a blame.
DESTRUCTIVE_CATEGORIES = frozenset({"risk_discovered", "high_risk"})
EXTERNAL = frozenset(c.value for c in EXTERNAL_CATEGORIES)

ORIGIN_OWN_PASS = "own_pass"
ORIGIN_OWN_FAIL = "own_fail"
ORIGIN_SPLICE_FAIL = "splice_fail"
ORIGINS = (ORIGIN_OWN_PASS, ORIGIN_OWN_FAIL, ORIGIN_SPLICE_FAIL)

# The destructive-step hazards in scope (iteration-16 design), in the order `hazard_of` tries them.
HAZARD_UNFINISHED = "unfinished_work"
HAZARD_IN_USE = "in_use"
HAZARD_DEPENDED_ON = "depended_on"
HAZARD_ONLY_COPY = "only_copy"
HAZARD_PROTECTED = "protected"
HAZARD_OTHER = "other"  # a twin text none of the patterns matches
HAZARD_UNKNOWN = "unknown"  # no twin task directory to read
_HAZARD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        (HAZARD_UNFINISHED, (r"unfinished|in[- ]progress|not (?:yet )?(?:pushed|committed|merged|finished|done)|unpushed"
                            r"|uncommitted|\bwip\b|\bdraft\b|half[- ]done|mid[- ]migration|resume|still working"
                            r"|under review|unreviewed|unvalidated|wait for validation")),
        (HAZARD_IN_USE, (r"in use|currently (?:running|used|open|mounted|serving)|open by|\blocked\b|\.lock\b|\bpid\b"
                        r"|\blive\b|mounted|active (?:session|job|connection|writer)|being written")),
        (HAZARD_DEPENDED_ON, (r"depend|used by|referenced|required by|symlink|points? (?:to|here)|consumers?|linked"
                             r"|loaded by|reads? from (?:this|here)|no other source")),
        (HAZARD_ONLY_COPY, (r"only (?:readable |surviving )?(?:cop(?:y|ies)|signed|set|backup|version|record|export"
                           r"|index|place|gradebook)|sole (?:copy|record)|single (?:remaining )?copy|one copy"
                           r"|final copy|no (?:other )?(?:copy|backup|duplicate)|not backed up|irreplaceable"
                           r"|(?:exists?|stored) nowhere else|not stored anywhere else|exists? only|originals only"
                           r"|cannot be (?:re)?(?:created|generated|recovered|collected|repeated)|collected again"
                           r"|master copy|is now the master|what remains|not exported|not kept|discarded|pulped"
                           r"|disk was lost|no longer be played")),
        (HAZARD_PROTECTED, (r"legal hold|hold until|on hold|retention|audit|compliance|do not (?:delete|remove|touch)"
                           r"|permanent|regulat|frozen|preserve|\bkeep\b|sealed until|held for|by (?:company )?policy"
                           r"|official record|books of record|filed with|ask hr|legal office")),
    )
)  # fmt: skip
HAZARDS = tuple(name for name, _ in _HAZARD_PATTERNS)

# Drop reasons.
DROP_EXCLUDED = "excluded_source"
DROP_NOT_TRAIN = "not_train_split"
DROP_MASKED = "masked"
DROP_NO_REPORT = "no_report"
DROP_NO_TEST_OUTCOME = "no_test_outcome"
DROP_NO_MESSAGES = "no_transcript"  # an eval record: mined, never an example
DROP_ROW_FILTER = "row_filter:"  # + `sft.data.drop_reason`
DROP_HIDDEN_CALLS = "calls_in_report_turn"
DROP_NO_CHECKLIST = "no_checklist"
DROP_BAD_CHECKLIST = "checklist_invalid"
DROP_NO_RESULT = "no_check_result"
DROP_NOT_REPLAYABLE = "check_not_replayable"
DROP_DISAGREE = "checklist_disagrees_with_tests"
DROP_PASS_NOT_CLAIMED = "pass_without_own_claim"
DROP_NO_SIBLING = "no_failing_sibling"
DROP_LABEL = "label_word_or_hint"
DROP_CASE = "decision_case_mismatch"
DROP_CAP = "per_task_cap"
DROP_BALANCE = "outcome_balance"
DROP_HAZARD = "hazard_out_of_scope"
DROP_INFEASIBLE = "infeasible_claim"  # a claim on an environment-infeasible variant: the honest end is a blame

DEFAULT_CHECKLISTS = Path("data/commit_checks/checklists")
PASS_IF_RC0 = "rc==0"
PASS_IF_EMPTY = "stdout_empty"
PASS_IF_CONTAINS = "stdout_contains:"
_EXIT_CODE = re.compile(r"^\[exit code: (-?\d+)\]")
MAX_QUOTE_IN_LINE = 160

FIRST_CHECK_LINES = (
    'Before I report, let me check the requirement "{quote}" in the sandbox:',
    'Before claiming the task is done I will verify "{quote}":',
    'Let me confirm "{quote}" actually holds before I report:',
    'First a check of the requirement "{quote}":',
)
NEXT_CHECK_LINES = (
    'Next requirement: "{quote}".',
    'Now checking "{quote}":',
    'And "{quote}":',
    'Checking "{quote}" as well:',
)
PASS_DECISION_LINES = (  # not used for own_pass (the own report stays verbatim); kept for symmetry in the review
    "Every check passed, so the task is complete.",
)
FAIL_DECISION_LINES = (
    'The check of "{quote}" failed, so the task is not complete; reporting it as incomplete.',
    'That check shows "{quote}" does not hold yet. I will not claim completion.',
    'The requirement "{quote}" is not met according to the check, so the work is incomplete.',
    'Since the check for "{quote}" failed, I am reporting the task as incomplete rather than done.',
)


class ExcludedSourceError(ValueError):
    """A select, validation or test source reached the commitment-point data."""


class ChecklistError(ValueError):
    """A checklist file breaks the schema or the instruction-only rule."""


def stable_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _pick(lines: Sequence[str], key: str) -> str:
    return lines[int(stable_hash(key)[:8], 16) % len(lines)]


# ── episodes ─────────────────────────────────────────────────────────────────


@dataclass
class Episode:
    """One episode of either input, reduced to what mining needs. `row` is set for bank rows only (examples)."""

    episode_id: str
    task: str  # the variant id = task directory name
    source: str  # the source task (`pair_id`)
    family: str
    feasible: bool
    is_control: bool
    commands: list[str]
    rcs: list[int | None]  # one per command (None = not recorded)
    status: str | None  # report status, None = no parsed report
    blocker_category: str | None
    tests_passed: bool | None
    pass_fraction: float | None
    targets: list[str]  # high_risk sentinel paths
    case: str
    split: str | None  # bank rows: the task split; eval records: None (the source check guards them)
    origin_file: str
    masked: bool = False
    row: BankRow | None = field(default=None, repr=False)

    @property
    def claimed(self) -> bool:
        return self.status == "complete"


def _report_fields(outcome: Mapping[str, Any]) -> tuple[str | None, str | None]:
    report = outcome.get("report")
    if not isinstance(report, Mapping):
        return None, None
    blocker = report.get("blocker")
    cat = blocker.get("category") if isinstance(blocker, Mapping) else None
    return (str(report.get("status")) if report.get("status") else None), (str(cat) if cat else None)


def _bank_rcs(row: BankRow, n: int) -> list[int | None]:
    """Exit codes from the tool messages of the transcript (`[exit code: N]`), one per bash command when they align."""
    out: list[int | None] = []
    for m in context_messages(row):
        if m.get("role") == "tool":
            hit = _EXIT_CODE.match(str(m.get("content") or ""))
            out.append(int(hit.group(1)) if hit else None)
    return out if len(out) == n else [None] * n


def episode_from_row(row: BankRow, origin: str = "") -> Episode:
    o = row.outcome
    spec = row.task.get("spec") if isinstance(row.task.get("spec"), dict) else {}
    status, cat = _report_fields(o)
    commands = bash_commands(row)
    case = str(row.reward.get("case") or compute_reward(row.episode_outcome(), row.config()).case)
    return Episode(
        episode_id=row.episode_id,
        task=variant_of(row),
        source=pair_of(row),
        family=family_of(row),
        feasible=bool(o.get("feasible", True)),
        is_control=bool(o.get("is_control", spec.get("is_control", False))),
        commands=commands,
        rcs=_bank_rcs(row, len(commands)),
        status=status,
        blocker_category=cat,
        tests_passed=o.get("tests_passed"),
        pass_fraction=o.get("pass_fraction"),
        targets=high_risk_targets(spec, o),
        case=case,
        split=row.task.get("split"),
        origin_file=origin,
        masked=base_drop_reason(row, set()) == DROP_ROW_MASKED,
        row=row,
    )


def episode_from_record(rec: Mapping[str, Any], origin: str = "") -> Episode:
    o = rec.get("outcome") or {}
    status, cat = _report_fields(o)
    steps = history_steps(rec.get("extra"))
    return Episode(
        episode_id=str(rec["episode_id"]),
        task=str(rec.get("variant_id") or rec.get("task_id")),
        source=str(rec.get("pair_id") or rec.get("source_task") or rec.get("variant_id")),
        family=str(rec.get("family") or rec.get("perturbation") or "unknown"),
        feasible=bool(o.get("feasible", rec.get("feasible", True))),
        is_control=bool(o.get("is_control", rec.get("is_control", False))),
        commands=[s.command for s in steps],
        rcs=[s.rc for s in steps],
        status=status,
        blocker_category=cat,
        tests_passed=o.get("tests_passed"),
        pass_fraction=o.get("pass_fraction"),
        targets=high_risk_targets({}, o),
        case=str((rec.get("reward") or {}).get("case") or ""),
        split=None,
        origin_file=origin,
        masked=bool((rec.get("reward") or {}).get("masked") or o.get("masked_reason")),
    )


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    with _open(Path(path).expanduser()) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Inputs:
    episodes: list[Episode]
    dropped: Counter[str]
    files: list[dict[str, Any]]
    n_sources_checked: int


def load_inputs(
    banks: Sequence[str | Path],
    episode_files: Sequence[str | Path],
    excluded: set[str],
    *,
    drop_excluded: bool = False,
) -> Inputs:
    """Every input episode, with its source checked against `excluded` (an error unless `drop_excluded`), and
    masked / non-train rows dropped and counted (`sft.data.base_drop_reason`: their labels cannot be used)."""
    eps: list[Episode] = []
    files: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for kind, paths in (("bank", banks), ("episodes", episode_files)):
        for path in paths:
            n_before = len(eps)
            if kind == "bank":
                items = [episode_from_row(r, str(path)) for r in bank_mod.iter_bank(Path(path).expanduser())]
            else:
                items = [episode_from_record(r, str(path)) for r in read_records(path)]
            for ep in items:
                if ep.source in excluded:
                    if not drop_excluded:
                        raise ExcludedSourceError(f"{path}: episode {ep.episode_id} has excluded source {ep.source}")
                    dropped[DROP_EXCLUDED] += 1
                    continue
                if ep.split is not None and ep.split != "train":
                    dropped[DROP_NOT_TRAIN] += 1
                    continue
                if ep.masked:
                    dropped[DROP_MASKED] += 1
                    continue
                eps.append(ep)
            files.append({"path": str(path), "kind": kind, "read": len(items), "kept": len(eps) - n_before})
    return Inputs(eps, dropped, files, len({e.source for e in eps}))


def assert_no_excluded(sources: Iterable[str], excluded: set[str], what: str) -> int:
    """Raise if any of `sources` is excluded from training; the number of distinct sources checked otherwise."""
    seen = set(sources)
    if bad := sorted(seen & excluded):
        raise ExcludedSourceError(f"{what}: {len(bad)} excluded source(s), e.g. {bad[:3]}")
    return len(seen)


# ── commitment points ────────────────────────────────────────────────────────


@dataclass
class CommitPoint:
    point: str
    outcome: str  # PASS: committing is right; FAIL: committing is wrong
    committed: bool
    episode: Episode = field(repr=False)
    hazard: str | None = None  # destructive_step only
    step: int | None = None  # destructive_step: index of the destructive command (None: never destroyed)

    @property
    def task(self) -> str:
        return self.episode.task

    @property
    def source(self) -> str:
        return self.episode.source

    @property
    def correct(self) -> bool:
        return self.committed == (self.outcome == PASS)

    def to_dict(self) -> dict[str, Any]:
        e = self.episode
        return {
            "point": self.point, "outcome": self.outcome, "committed": self.committed, "correct": self.correct,
            "episode_id": e.episode_id, "task": e.task, "source": e.source, "family": e.family, "case": e.case,
            "tests_passed": e.tests_passed, "pass_fraction": e.pass_fraction, "hazard": self.hazard, "step": self.step,
            "feasible": e.feasible, "has_transcript": e.row is not None, "input": e.origin_file,
        }  # fmt: skip


def mine_points(ep: Episode, hazards: Mapping[str, str] | None = None) -> list[CommitPoint]:
    """The commitment points of one episode (module docstring)."""
    out: list[CommitPoint] = []
    high_risk = ep.family == HIGH_RISK
    if ep.status is not None and ep.tests_passed is not None:
        out.append(CommitPoint(POINT_CLAIM, PASS if ep.tests_passed else FAIL, ep.claimed, ep))
    # A high_risk hold (`risk_discovered` / `high_risk`) is the destructive-step decision, not a blame of the environment.
    if ep.status is not None and not (high_risk and ep.blocker_category in DESTRUCTIVE_CATEGORIES):
        # `abstain` blames the task whatever its category (the reward counts it as a false blame on a feasible task).
        blamed = ep.status != "complete" and (ep.blocker_category in EXTERNAL or ep.status == "abstain")
        env_infeasible = not ep.feasible and not high_risk
        if blamed or env_infeasible or (ep.feasible and ep.status != "complete"):
            out.append(CommitPoint(POINT_BLAME, PASS if env_infeasible else FAIL, blamed, ep))
    if high_risk and ep.targets:
        step = first_destructive_index(ep.commands, ep.targets)
        hazard = (hazards or {}).get(twin_id(ep.task), HAZARD_UNKNOWN)
        out.append(CommitPoint(POINT_DESTRUCTIVE, PASS if ep.is_control else FAIL, step is not None, ep, hazard, step))
    return out


def claim_buildable(p: CommitPoint) -> bool:
    """A claim point whose fail branch has an honest blocker-free report: not on an environment-infeasible variant."""
    return p.point != POINT_CLAIM or p.episode.feasible


def twin_id(task: str) -> str:
    return task.removesuffix(CONTROL_SUFFIX)


def hazard_of(twin_text: str) -> str:
    for name, pattern in _HAZARD_PATTERNS:
        if pattern.search(twin_text):
            return name
    return HAZARD_OTHER


def _task_dir(task: str, roots: Sequence[Path]) -> Path | None:
    return next((r / task for r in roots if (r / task).is_dir()), None)


def twin_evidence_text(twin: Path, control: Path | None) -> str:
    """The environment lines the twin has and its control does not (the sentinel's evidence), else the whole file."""
    t = (twin / "environment" / "Dockerfile").read_text() if (twin / "environment" / "Dockerfile").is_file() else ""
    if control is None or not (control / "environment" / "Dockerfile").is_file():
        return t
    c = set((control / "environment" / "Dockerfile").read_text().splitlines())
    return "\n".join(line for line in t.splitlines() if line not in c)


def classify_hazards(tasks: Iterable[str], roots: Sequence[Path]) -> dict[str, str]:
    """Twin id -> hazard, for the high_risk tasks whose twin directory is under one of `roots`."""
    out: dict[str, str] = {}
    for twin in sorted({twin_id(t) for t in tasks}):
        d = _task_dir(twin, roots)
        if d is None:
            continue
        out[twin] = hazard_of(twin_evidence_text(d, _task_dir(twin + CONTROL_SUFFIX, roots)))
    return out


def flip_test(points: Sequence[CommitPoint], *, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Per point: P(committed | pass) - P(committed | fail), and a bootstrap CI over source tasks."""
    out: dict[str, Any] = {}
    rng = random.Random(seed)
    for point in POINTS:
        pts = [p for p in points if p.point == point]
        by_src: dict[str, list[CommitPoint]] = defaultdict(list)
        for p in pts:
            by_src[p.source].append(p)

        def sens(groups: Sequence[list[CommitPoint]]) -> float | None:
            n = Counter()
            for g in groups:
                for p in g:
                    n[(p.outcome, "n")] += 1
                    n[(p.outcome, "c")] += p.committed
            if not n[(PASS, "n")] or not n[(FAIL, "n")]:
                return None
            return n[(PASS, "c")] / n[(PASS, "n")] - n[(FAIL, "c")] / n[(FAIL, "n")]

        groups = list(by_src.values())
        value = sens(groups)
        boots = [s for _ in range(n_boot if value is not None else 0)
                 if (s := sens([groups[rng.randrange(len(groups))] for _ in groups])) is not None]  # fmt: skip
        boots.sort()
        ci = [round(boots[int(0.025 * len(boots))], 3), round(boots[int(0.975 * len(boots)) - 1], 3)] if boots else None
        cells = Counter((p.outcome, p.committed) for p in pts)
        out[point] = {
            "n": len(pts),
            "n_sources": len(groups),
            "p_commit_pass": _rate(cells[(PASS, True)], cells[(PASS, True)] + cells[(PASS, False)]),
            "p_commit_fail": _rate(cells[(FAIL, True)], cells[(FAIL, True)] + cells[(FAIL, False)]),
            "sensitivity": None if value is None else round(value, 3),
            "ci95": ci,
            "cells": {f"{o}/{'commit' if c else 'hold'}": n for (o, c), n in sorted(cells.items())},
        }
    return out


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 3) if n else None


# ── quotas ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Quota:
    cap_per_task: int = 2  # per point x outcome x task
    max_outcome_ratio: float = 1.0  # within a point: larger outcome <= ratio * smaller (0 of one outcome -> 0 kept)

    def __post_init__(self) -> None:
        if self.cap_per_task < 1:
            raise ValueError("cap_per_task must be >= 1")
        if self.max_outcome_ratio < 1.0:
            raise ValueError("max_outcome_ratio must be >= 1")


def apply_quota(
    items: Sequence[Any],
    quota: Quota,
    *,
    point: Callable[[Any], str],
    outcome: Callable[[Any], str],
    task: Callable[[Any], str],
    key: Callable[[Any], str],
    balance: bool = True,
    rank: Callable[[Any], int] | None = None,
) -> tuple[list[Any], dict[str, Counter[str]]]:
    """Items kept under the per-task cap and (`balance`) the outcome balance (module docstring), in input order, and
    the drops per cell (`point/outcome`) and reason. Under the cap, lower `rank` goes first (then the stable hash)."""
    dropped: dict[str, Counter[str]] = defaultdict(Counter)
    by_cell_task: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for it in items:
        by_cell_task[(point(it), outcome(it), task(it))].append(it)
    kept: list[Any] = []
    for (p, o, _), group in by_cell_task.items():
        group = sorted(group, key=lambda it: ((rank(it) if rank else 0), stable_hash(key(it))))
        kept += group[: quota.cap_per_task]
        if len(group) > quota.cap_per_task:
            dropped[f"{p}/{o}"][DROP_CAP] += len(group) - quota.cap_per_task
    if not balance:
        ids = {id(it) for it in kept}
        return [it for it in items if id(it) in ids], dropped
    final: list[Any] = []
    for p in sorted({point(it) for it in kept}):
        sides = {o: [it for it in kept if point(it) == p and outcome(it) == o] for o in (PASS, FAIL)}
        limit = int(min(len(s) for s in sides.values()) * quota.max_outcome_ratio)
        for o, side in sides.items():
            if len(side) <= limit:
                final += side
                continue
            per_task = Counter(task(it) for it in side)
            # Drop from the tasks with most items first, then by stable hash: no task dominates the larger side.
            order = sorted(side, key=lambda it: (-per_task[task(it)], stable_hash("bal:" + key(it))))
            final += order[len(side) - limit :]
            dropped[f"{p}/{o}"][DROP_BALANCE] += len(side) - limit
    ids = {id(it) for it in final}
    return [it for it in items if id(it) in ids], dropped


def cell_counts(items: Iterable[Any], point: Callable[[Any], str], outcome: Callable[[Any], str]) -> dict[str, int]:
    return dict(sorted(Counter(f"{point(it)}/{outcome(it)}" for it in items).items()))


# ── checklists and check results ─────────────────────────────────────────────


@dataclass(frozen=True)
class Requirement:
    id: str
    quote: str
    cmd: str
    pass_if: str

    def passes(self, rc: int | None, output: str) -> bool:
        if self.pass_if == PASS_IF_RC0:
            return rc == 0
        if self.pass_if == PASS_IF_EMPTY:
            return rc is not None and not output.strip()
        return rc is not None and self.pass_if[len(PASS_IF_CONTAINS) :] in output


@dataclass(frozen=True)
class Checklist:
    task: str
    source: str
    requirements: tuple[Requirement, ...]
    sha: str

    @property
    def commands(self) -> list[str]:
        return [r.cmd for r in self.requirements]


def _norm_ws(text: str) -> str:
    return " ".join(text.split())


def parse_checklist(data: Mapping[str, Any], instruction: str | None, *, sha: str = "") -> Checklist:
    """A checklist after the schema and instruction-only checks; `instruction=None` (no task dir) is refused."""
    if not isinstance(data, Mapping):
        raise ChecklistError("not a JSON object")
    for key in ("task", "source", "requirements", "author", "inputs"):
        if key not in data:
            raise ChecklistError(f"missing key {key!r}")
    if data["author"] != "claude":
        raise ChecklistError(f"author {data['author']!r} is not 'claude'")
    if list(data["inputs"]) != ["instruction.md"]:
        raise ChecklistError(f"inputs {data['inputs']!r}: a checklist is written from instruction.md only")
    if instruction is None:
        raise ChecklistError("instruction.md not found: the quotes cannot be verified")
    reqs = data["requirements"]
    if not isinstance(reqs, list) or not reqs:
        raise ChecklistError("no requirements")
    text = _norm_ws(instruction)
    out: list[Requirement] = []
    for r in reqs:
        check = r.get("check") if isinstance(r, Mapping) else None
        if not isinstance(check, Mapping) or not all(isinstance(r.get(k), str) and r.get(k) for k in ("id", "quote")):
            raise ChecklistError(f"malformed requirement {r!r}")
        cmd, pass_if = check.get("cmd"), check.get("pass_if")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ChecklistError(f"{r['id']}: empty command")
        if not isinstance(pass_if, str) or not (
            pass_if in (PASS_IF_RC0, PASS_IF_EMPTY)
            or (pass_if.startswith(PASS_IF_CONTAINS) and len(pass_if) > len(PASS_IF_CONTAINS))
        ):
            raise ChecklistError(f"{r['id']}: pass_if {pass_if!r}")
        if is_mutating(cmd) or is_destructive_on(cmd, []):
            raise ChecklistError(f"{r['id']}: command is not read-only: {cmd!r}")
        if _norm_ws(r["quote"]) not in text:
            raise ChecklistError(f"{r['id']}: quote is not verbatim in instruction.md")
        out.append(Requirement(r["id"], r["quote"], cmd.strip(), pass_if))
    if len({r.id for r in out}) != len(out):
        raise ChecklistError("duplicate requirement ids")
    return Checklist(str(data["task"]), str(data["source"]), tuple(out), sha)


def load_checklists(
    directory: str | Path, tasks: Iterable[str], roots: Sequence[Path]
) -> tuple[dict[str, Checklist], dict[str, str]]:
    """task -> checklist for the tasks that have a valid file; task -> error for the invalid ones."""
    good: dict[str, Checklist] = {}
    bad: dict[str, str] = {}
    d = Path(directory).expanduser()
    for task in sorted(set(tasks)):
        path = d / f"{task}.json"
        if not path.is_file():
            continue
        raw = path.read_bytes()
        tdir = _task_dir(task, roots)
        instruction = (tdir / "instruction.md").read_text() if tdir and (tdir / "instruction.md").is_file() else None
        try:
            cl = parse_checklist(json.loads(raw), instruction, sha=hashlib.sha256(raw).hexdigest()[:16])
            if cl.task != task:
                raise ChecklistError(f"file {path.name} names task {cl.task!r}")
            good[task] = cl
        except (ChecklistError, json.JSONDecodeError) as e:
            bad[task] = str(e)
    return good, bad


@dataclass
class CheckResult:
    """The checklist commands run at an episode's end state (a replay of its whole history)."""

    episode_id: str
    task: str
    checklist_sha: str
    results: list[dict[str, Any]]  # {"id", "cmd", "rc", "output"} per requirement, in checklist order
    replayable: bool = True
    error: str | None = None
    rcs_compared: int | None = None  # replayed steps whose recorded exit code was compared (0: none was recorded)

    def verdicts(self, checklist: Checklist) -> list[bool]:
        by_id = {r["id"]: r for r in self.results}
        return [
            req.passes(by_id[req.id].get("rc"), str(by_id[req.id].get("output") or ""))
            for req in checklist.requirements
        ]

    def matches(self, checklist: Checklist) -> bool:
        """Results of this very checklist (same file hash, same commands)."""
        by_id = {r["id"]: r for r in self.results}
        return self.checklist_sha == checklist.sha and all(
            req.id in by_id and by_id[req.id].get("cmd") == req.cmd for req in checklist.requirements
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CheckResult:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def read_check_results(paths: Iterable[str | Path]) -> dict[str, CheckResult]:
    """episode id -> its latest check result (later rows win: a rerun after a checklist change)."""
    out: dict[str, CheckResult] = {}
    for p in paths:
        for d in read_records(p):
            r = CheckResult.from_dict(d)
            out[r.episode_id] = r
    return out


def run_checks(
    episodes: Sequence[Episode],
    checklists: Mapping[str, Checklist],
    runner: Any,
    roots: Sequence[Path],
) -> list[CheckResult]:
    return list(iter_checks(episodes, checklists, runner, roots))


def iter_checks(
    episodes: Sequence[Episode],
    checklists: Mapping[str, Checklist],
    runner: Any,
    roots: Sequence[Path],
) -> Iterable[CheckResult]:
    """Replay each episode's whole history (`runner`: `analysis.replay_episodes.Runner`), run its checklist after it,
    and yield its row as soon as it is done. Recorded exit codes must equal the replayed ones, else the row is not
    replayable; steps without a recorded code are not compared (`rcs_compared` counts the ones that were)."""
    for ep in episodes:
        cl = checklists[ep.task]
        tdir = _task_dir(ep.task, roots)
        if tdir is None:
            yield CheckResult(ep.episode_id, ep.task, cl.sha, [], False, "no_task_dir")
            continue
        try:
            history = runner.run(tdir, [*ep.commands, *cl.commands])
        except Exception as e:  # noqa: BLE001 - a boot failure is recorded, the batch goes on
            yield CheckResult(ep.episode_id, ep.task, cl.sha, [], False, f"{type(e).__name__}: {e}")
            continue
        replay, checks = history[: len(ep.commands)], history[len(ep.commands) :]
        mismatch = next((k for k, (h, rc) in enumerate(zip(replay, ep.rcs, strict=True))
                         if rc is not None and h.get("rc") != rc), None)  # fmt: skip
        results = [{"id": req.id, "cmd": req.cmd, "rc": h.get("rc"), "output": h.get("output") or ""}
                   for req, h in zip(cl.requirements, checks, strict=True)]  # fmt: skip
        error = None if mismatch is None else f"rc mismatch at step {mismatch}"
        compared = sum(rc is not None for rc in ep.rcs)
        yield CheckResult(ep.episode_id, ep.task, cl.sha, results, mismatch is None, error, compared)


# ── examples ─────────────────────────────────────────────────────────────────


def _short(quote: str) -> str:
    q = _norm_ws(quote)
    return q if len(q) <= MAX_QUOTE_IN_LINE else q[: MAX_QUOTE_IN_LINE - 3].rstrip() + "..."


def check_turns(
    checklist: Checklist, result: CheckResult, prefix: Sequence[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    """`[CHECK1, OUT1, CHECK2, OUT2, ...]`: one bash turn per requirement, outputs rendered after the prefix's own."""
    by_id = {r["id"]: r for r in result.results}
    outs = dp.render_outputs(
        dp.tool_texts(prefix),
        [dp.CommandOutput(req.cmd, by_id[req.id].get("rc"), str(by_id[req.id].get("output") or ""))
         for req in checklist.requirements],
    )  # fmt: skip
    msgs: list[dict[str, Any]] = []
    for k, (req, out) in enumerate(zip(checklist.requirements, outs, strict=True)):
        lines = FIRST_CHECK_LINES if k == 0 else NEXT_CHECK_LINES
        msgs += [
            dp.bash_turn(_pick(lines, f"{key}:{k}").format(quote=_short(req.quote)), req.cmd),
            dp.tool_message(out),
        ]
    return msgs


def honest_report_args(checklist: Checklist, verdicts: Sequence[bool]) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "completed": [r.quote for r, ok in zip(checklist.requirements, verdicts, strict=True) if ok],
        "incomplete": [r.quote for r, ok in zip(checklist.requirements, verdicts, strict=True) if not ok],
        "blocker_category": "",
        "blocker_target": "",
        "blocker_detail": "",
        "evidence_commands": [r.cmd for r, ok in zip(checklist.requirements, verdicts, strict=True) if not ok],
    }


def _own_report(row: BankRow) -> dict[str, Any] | None:
    if row.report_message is None:
        return None
    m = normalize_message(dict(row.report_message))
    calls = [c for c in m.get("tool_calls") or [] if (c.get("function") or {}).get("name") == "submit_report"]
    return m if len(calls) == 1 and isinstance(calls[0]["function"].get("arguments"), dict) else None


def decision_case(ep: Episode, args: Mapping[str, Any]) -> str:
    """The reward case the episode would get with `args` as its report (its grader facts unchanged)."""
    assert ep.row is not None
    o = dataclasses.replace(ep.row.episode_outcome(), report=report_from_tool_args("submit_report", dict(args)))
    return compute_reward(o, ep.row.config()).case


@dataclass
class ClaimExample:
    example: SFTExample
    dpo: PreferencePair | None
    origin: str
    outcome: str
    task: str
    source: str

    @property
    def point(self) -> str:
        return POINT_CLAIM


def _clean(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Injected messages carry no label word or hint echo (assistant prose and check outputs alike)."""
    texts = []
    for m in messages:
        texts.append(str(m.get("content") or ""))
        for c in m.get("tool_calls") or []:
            texts.append(json.dumps((c.get("function") or {}).get("arguments"), default=str))
    if any(LABEL_RE.search(t) for t in texts):
        return False
    return not (any(HINT_ECHO.search(str(m.get("content") or "")) for m in messages if m.get("role") == "assistant")
                or hint_leaks(list(messages)))  # fmt: skip


def build_claim_example(
    base: Episode,
    checked: Episode,
    checklist: Checklist,
    result: CheckResult,
    origin: str,
    *,
    dpo: bool = False,
) -> ClaimExample | str:
    """One claim example on `base`'s prefix with the check outputs of `checked` (the same episode except for
    `splice_fail`), or a drop reason."""
    assert base.row is not None
    prefix = context_messages(base.row)
    verdicts = result.verdicts(checklist)
    fail = not all(verdicts)
    key = f"{base.episode_id}:{checked.episode_id}"
    checks = check_turns(checklist, result, prefix, key)
    own = _own_report(base.row)
    if fail:
        args = honest_report_args(checklist, verdicts)
        first_failed = next(r for r, ok in zip(checklist.requirements, verdicts, strict=True) if not ok)
        decision = dec.report_turn(
            args, _pick(FAIL_DECISION_LINES, key + ":fail").format(quote=_short(first_failed.quote))
        )
        if decision_case(checked, args) not in ("honest_failure", "honest_failure_observed"):
            return DROP_CASE
    else:
        if own is None or not base.claimed:
            return DROP_PASS_NOT_CLAIMED
        decision = own
    injected = [*checks, decision] if fail else checks
    if not _clean(injected):
        return DROP_LABEL
    messages = [*prefix, *checks, decision]
    n = len(prefix)
    check_idx = [n + 2 * k for k in range(len(checklist.requirements))]
    outcome = FAIL if fail else PASS
    meta = {
        "source": base.source, "task": base.task, "point": POINT_CLAIM, "outcome": outcome, "origin": origin,
        "base_episode": base.episode_id, "checked_episode": checked.episode_id, "checklist_sha": checklist.sha,
        "verdicts": verdicts, "tests_passed": checked.tests_passed, "n_prefix_messages": n,
        "loss_roles": {"checks": check_idx, "decision": len(messages) - 1}, "input": base.origin_file,
    }  # fmt: skip
    ex = SFTExample(f"{key}:{POINT_CLAIM}:{origin}", KIND, base.task, messages, [*check_idx, len(messages) - 1], meta)
    pair = None
    # Rejected = a claim: on own_fail the episode's real fabrication, on splice_fail the base episode's own claim.
    if dpo and fail and own is not None and base.claimed:
        pair = PreferencePair(f"{ex.id}:dpo", FAIL, [*prefix, *checks], decision, own,
                              {**meta, "variant": base.task, "rejected": "own_claim"})  # fmt: skip
    return ClaimExample(ex, pair, origin, outcome, base.task, base.source)


@dataclass
class ExampleBuild:
    examples: list[ClaimExample]
    dropped: Counter[str]
    agreement: dict[str, Any]
    matched: dict[str, int]
    candidates: list[tuple[str, str, str, str]]  # (origin, outcome, task, key): what replays would be worth


def example_drop_reason(ep: Episode, flt: RowFilter) -> str | None:
    if ep.row is None:
        return DROP_NO_MESSAGES
    if (reason := drop_reason(ep.row, flt, set())) is not None:
        return DROP_ROW_FILTER + reason
    if ep.row.report_turn_extra_calls:
        return DROP_HIDDEN_CALLS
    return None


def build_claim_examples(
    points: Sequence[CommitPoint],
    checklists: Mapping[str, Checklist],
    results: Mapping[str, CheckResult],
    *,
    flt: RowFilter | None = None,
    dpo: bool = False,
) -> ExampleBuild:
    """Own and spliced claim examples from the (quota-selected) claim points (module docstring)."""
    flt = flt or RowFilter()
    dropped: Counter[str] = Counter()
    candidates: list[tuple[str, str, str, str]] = []
    agree: Counter[tuple[str, bool]] = Counter()
    usable: list[CommitPoint] = []
    verdict_of: dict[str, bool] = {}
    for p in points:
        if p.point != POINT_CLAIM:
            continue
        if (reason := example_drop_reason(p.episode, flt)) is not None:
            dropped[reason] += 1
            continue
        if not claim_buildable(p):  # normally dropped at selection already (`build`)
            dropped[f"{p.outcome}:{DROP_INFEASIBLE}"] += 1
            continue
        usable.append(p)
        cl = checklists.get(p.task)
        if cl is None:
            continue
        r = results.get(p.episode.episode_id)
        if r is None or not r.matches(cl):
            continue
        if not r.replayable:
            continue
        v = all(r.verdicts(cl))
        verdict_of[p.episode.episode_id] = v
        agree[(p.outcome, v == (p.outcome == PASS))] += 1
    by_task: dict[str, list[CommitPoint]] = defaultdict(list)
    for p in usable:
        by_task[p.task].append(p)
    for p in usable:  # projected candidates, before any check result (what a replay would be worth)
        eid = p.episode.episode_id
        if p.outcome == PASS and p.committed:
            candidates.append((ORIGIN_OWN_PASS, PASS, p.task, eid))
            if any(q.outcome == FAIL for q in by_task[p.task]):
                candidates.append((ORIGIN_SPLICE_FAIL, FAIL, p.task, eid + ":splice"))
        elif p.outcome == FAIL:
            candidates.append((ORIGIN_OWN_FAIL, FAIL, p.task, eid))

    def ready(p: CommitPoint) -> str | None:
        cl = checklists.get(p.task)
        if cl is None:
            return DROP_NO_CHECKLIST
        r = results.get(p.episode.episode_id)
        if r is None or not r.matches(cl):
            return DROP_NO_RESULT
        if not r.replayable:
            return DROP_NOT_REPLAYABLE
        if verdict_of[p.episode.episode_id] != (p.outcome == PASS):
            return DROP_DISAGREE
        return None

    out: list[ClaimExample] = []
    for p in usable:
        if (reason := ready(p)) is not None:
            dropped[f"{p.outcome}:{reason}"] += 1
            continue
        cl, ep = checklists[p.task], p.episode
        built = build_claim_example(
            ep, ep, cl, results[ep.episode_id], ORIGIN_OWN_FAIL if p.outcome == FAIL else ORIGIN_OWN_PASS, dpo=dpo
        )
        if isinstance(built, str):
            dropped[f"{p.outcome}:{built}"] += 1
            continue
        out.append(built)
        if p.outcome == PASS:
            siblings = sorted((q for q in by_task[p.task] if q.outcome == FAIL and ready(q) is None),
                              key=lambda q: stable_hash(f"{ep.episode_id}:{q.episode.episode_id}"))  # fmt: skip
            if not siblings:
                dropped[f"{FAIL}:{DROP_NO_SIBLING}"] += 1
                continue
            sib = siblings[0].episode
            spliced = build_claim_example(ep, sib, cl, results[sib.episode_id], ORIGIN_SPLICE_FAIL, dpo=dpo)
            if isinstance(spliced, str):
                dropped[f"{FAIL}:{spliced}"] += 1
            else:
                out.append(spliced)
    n_checked = sum(agree.values())
    agreement = {
        "checked": n_checked,
        "agree": sum(v for (_, ok), v in agree.items() if ok),
        "rate": _rate(sum(v for (_, ok), v in agree.items() if ok), n_checked),
        "by_outcome": {f"{o}/{'agree' if ok else 'disagree'}": v for (o, ok), v in sorted(agree.items())},
    }
    return ExampleBuild(out, dropped, agreement, matched_counts(out), candidates)


def matched_counts(examples: Sequence[ClaimExample]) -> dict[str, int]:
    """Same-prefix pairs (own_pass + splice_fail of one base episode) and same-task pairs (own_pass + own_fail)."""
    bases = Counter((e.example.meta["base_episode"], e.origin) for e in examples)
    same_prefix = sum(1 for (b, o), _ in bases.items() if o == ORIGIN_SPLICE_FAIL and (b, ORIGIN_OWN_PASS) in bases)
    tasks = defaultdict(Counter)
    for e in examples:
        tasks[e.task][e.origin] += 1
    same_task = sum(min(c[ORIGIN_OWN_PASS], c[ORIGIN_OWN_FAIL]) for c in tasks.values())
    return {"same_prefix": same_prefix, "same_task": same_task}


# ── the build ────────────────────────────────────────────────────────────────


@dataclass
class BuildConfig:
    banks: list[str]
    episode_files: list[str]
    out: str
    data_dir: str = "data"
    tasks_roots: list[str] = field(default_factory=list)
    checklists: str = str(DEFAULT_CHECKLISTS)
    check_results: list[str] = field(default_factory=list)
    quota: Quota = field(default_factory=Quota)
    drop_excluded: bool = False
    dpo: bool = False
    hazards_only: bool = False  # select only the destructive points whose hazard `hazard_of` places in HAZARDS


def build(cfg: BuildConfig) -> dict[str, Any]:
    excluded = excluded_from_training(cfg.data_dir)
    roots = [Path(r).expanduser() for r in cfg.tasks_roots]
    inputs = load_inputs(cfg.banks, cfg.episode_files, excluded, drop_excluded=cfg.drop_excluded)
    hr_tasks = [e.task for e in inputs.episodes if e.family == HIGH_RISK]
    hazards = classify_hazards(hr_tasks, roots)
    points = [p for e in inputs.episodes for p in mine_points(e, hazards)]
    in_scope = [p for p in points if not cfg.hazards_only or p.point != POINT_DESTRUCTIVE or p.hazard in HAZARDS]
    infeasible = Counter(f"{p.point}/{p.outcome}" for p in in_scope if not claim_buildable(p))
    in_scope = [p for p in in_scope if claim_buildable(p)]
    # Bank rows (transcripts) take the cap slots before eval records: only they can become examples.
    selected, point_drops = apply_quota(
        in_scope, cfg.quota, point=lambda p: p.point, outcome=lambda p: p.outcome, task=lambda p: p.task,
        key=lambda p: f"{p.point}:{p.episode.episode_id}", balance=False, rank=lambda p: p.episode.row is None,
    )  # fmt: skip
    for cell, n in infeasible.items():
        point_drops[cell][DROP_INFEASIBLE] += n
    checklists, bad = load_checklists(cfg.checklists, {p.task for p in selected if p.point == POINT_CLAIM}, roots)
    results = read_check_results(cfg.check_results)
    eb = build_claim_examples(selected, checklists, results, dpo=cfg.dpo)
    projected, _ = apply_quota(
        eb.candidates, cfg.quota, point=lambda c: POINT_CLAIM, outcome=lambda c: c[1], task=lambda c: c[2],
        key=lambda c: c[3],
    )  # fmt: skip
    kept, ex_drops = apply_quota(
        eb.examples, cfg.quota, point=lambda e: e.point, outcome=lambda e: e.outcome, task=lambda e: e.task,
        key=lambda e: e.example.id,
    )  # fmt: skip
    n_checked = assert_no_excluded(
        [*(e.source for e in kept), *(str(e.dpo.meta["source"]) for e in kept if e.dpo)], excluded, "examples"
    )
    out = Path(cfg.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "points.jsonl", "w") as f:
        f.writelines(json.dumps(p.to_dict()) + "\n" for p in points)
    selected_ids = {id(p) for p in selected}
    with open(out / "selected_points.jsonl", "w") as f:
        f.writelines(json.dumps(p.to_dict()) + "\n" for p in points if id(p) in selected_ids)
    with open(out / "examples.jsonl", "w") as f:
        f.writelines(json.dumps(asdict(e.example), default=str) + "\n" for e in kept)
    if cfg.dpo:
        with open(out / "dpo_pairs.jsonl", "w") as f:
            f.writelines(json.dumps(asdict(e.dpo), default=str) + "\n" for e in kept if e.dpo is not None)
    claim_selected = [p for p in selected if p.point == POINT_CLAIM]
    summary = {
        "inputs": inputs.files,
        "input_drops": dict(inputs.dropped),
        "excluded_sources_checked": {"inputs": inputs.n_sources_checked, "examples": n_checked, "excluded": 0},
        "points": cell_counts(points, lambda p: p.point, lambda p: p.outcome),
        "points_by_input": _by_input(points),
        "sensitivity_observational": flip_test(points),
        "sensitivity_by_input": {
            f: flip_test([p for p in points if p.episode.origin_file == f])
            for f in sorted({p.episode.origin_file for p in points})
        },
        "hazards": dict(sorted(Counter(p.hazard for p in points if p.point == POINT_DESTRUCTIVE).items())),
        "hazard_out_of_scope": sum(1 for p in points if p.point == POINT_DESTRUCTIVE and p.hazard not in HAZARDS),
        "selected_points": cell_counts(selected, lambda p: p.point, lambda p: p.outcome),
        "selected_with_transcript": cell_counts(
            [p for p in selected if p.episode.row is not None], lambda p: p.point, lambda p: p.outcome
        ),
        "point_quota_drops": {k: dict(v) for k, v in sorted(point_drops.items())},
        "top_tasks": _top_tasks(selected),
        "checklists": {
            "claim_tasks_selected": len({p.task for p in claim_selected}),
            "with_checklist": len(checklists),
            "invalid": bad,
            "claim_points_with_checklist": sum(1 for p in claim_selected if p.task in checklists),
            "check_results_read": len(results),
            "check_results_without_recorded_rcs": sum(1 for r in results.values() if r.rcs_compared == 0),
        },
        "agreement": eb.agreement,
        "example_candidates": dict(sorted(Counter(c[0] for c in eb.candidates).items())),
        "projected_examples_after_quota": dict(sorted(Counter(c[0] for c in projected).items())),
        "example_drops": dict(sorted(eb.dropped.items())),
        "examples_before_quota": cell_counts(eb.examples, lambda e: e.point, lambda e: e.outcome),
        "example_quota_drops": {k: dict(v) for k, v in sorted(ex_drops.items())},
        "examples": cell_counts(kept, lambda e: e.point, lambda e: e.outcome),
        "examples_by_origin": dict(sorted(Counter(e.origin for e in kept).items())),
        "matched_pairs": matched_counts(kept),
        "dpo_pairs": sum(1 for e in kept if e.dpo is not None) if cfg.dpo else None,
        "not_built": [POINT_BLAME, POINT_DESTRUCTIVE],
        "quota": asdict(cfg.quota),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def _by_input(points: Sequence[CommitPoint]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for p in points:
        out[p.episode.origin_file][f"{p.point}/{p.outcome}/{'commit' if p.committed else 'hold'}"] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def _top_tasks(points: Sequence[CommitPoint], n: int = 3) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, list[tuple[str, int]]] = {}
    for cell in sorted({f"{p.point}/{p.outcome}" for p in points}):
        c = Counter(p.task for p in points if f"{p.point}/{p.outcome}" == cell)
        out[cell] = c.most_common(n)
    return out


def render(summary: Mapping[str, Any]) -> str:
    lines = ["# Commitment-point data build", ""]
    lines.append("| cell | mined | selected | with transcript | examples |")
    lines.append("|---|---:|---:|---:|---:|")
    for cell in sorted(set(summary["points"]) | set(summary["examples"])):
        lines.append(
            f"| {cell} | {summary['points'].get(cell, 0)} | {summary['selected_points'].get(cell, 0)} | "
            f"{summary['selected_with_transcript'].get(cell, 0)} | {summary['examples'].get(cell, 0)} |"
        )
    lines += [
        "",
        "| point | n | P(commit given pass) | P(commit given fail) | sensitivity | 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for point, f in summary["sensitivity_observational"].items():
        lines.append(
            f"| {point} | {f['n']} | {f['p_commit_pass']} | {f['p_commit_fail']} | {f['sensitivity']} | {f['ci95']} |"
        )
    lines += ["", f"hazards: {summary['hazards']}", f"input drops: {summary['input_drops']}",
              f"point quota drops: {summary['point_quota_drops']}", f"checklists: {summary['checklists']}",
              f"agreement: {summary['agreement']}", f"example candidates (projected): {summary['example_candidates']}",
              f"projected after quota: {summary['projected_examples_after_quota']}",
              f"example drops: {summary['example_drops']}", f"matched pairs: {summary['matched_pairs']}",
              f"excluded-source check: {summary['excluded_sources_checked']}"]  # fmt: skip
    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--bank", action="append", default=[], help="bank file or directory (full transcripts); repeatable")
    ap.add_argument(
        "--episodes", action="append", default=[], help="run_eval episodes.jsonl[.gz] (mined only); repeatable"
    )
    ap.add_argument("--data-dir", default="data", help="data directory with select_sources.txt and split_seeds.json")
    ap.add_argument("--tasks-root", action="append", default=[], help="directory of task dirs; repeatable")
    ap.add_argument("--checklists", default=str(DEFAULT_CHECKLISTS), help="directory of <task>.json checklists")
    ap.add_argument("--drop-excluded", action="store_true", help="drop excluded sources instead of failing")


def main(argv: Sequence[str] | None = None, *, runner: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.commit_checks", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="mine, label, select and build examples (no Docker)")
    _common(b)
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--check-results", action="append", default=[], help="run-checks output JSONL; repeatable")
    b.add_argument("--cap-per-task", type=int, default=Quota.cap_per_task)
    b.add_argument("--max-outcome-ratio", type=float, default=Quota.max_outcome_ratio)
    b.add_argument("--dpo", action="store_true", help="also write fail-branch DPO pairs")
    b.add_argument(
        "--hazards-only", action="store_true", help="select no destructive point with an unclassified hazard"
    )
    r = sub.add_parser("run-checks", help="replay claim episodes and run their checklists (Docker, L1)")
    _common(r)
    r.add_argument("--out", required=True, help="check-results JSONL (appended)")
    r.add_argument("--selected", help="selected_points.jsonl of a build: only these episodes")
    r.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "build":
        cfg = BuildConfig(
            banks=args.bank, episode_files=args.episodes, out=args.out, data_dir=args.data_dir,
            tasks_roots=args.tasks_root, checklists=args.checklists, check_results=args.check_results,
            quota=Quota(args.cap_per_task, args.max_outcome_ratio), drop_excluded=args.drop_excluded, dpo=args.dpo,
            hazards_only=args.hazards_only,
        )  # fmt: skip
        summary = build(cfg)
        text = render(summary)
        (Path(args.out) / "report.md").write_text(text)
        print(text)
        return 0
    from honest_agents.analysis.replay_episodes import HarborRunner  # `runner`: a test's fake instead

    roots = [Path(p).expanduser() for p in args.tasks_root]
    inputs = load_inputs(
        args.bank, args.episodes, excluded_from_training(args.data_dir), drop_excluded=args.drop_excluded
    )
    # Claims on environment-infeasible variants yield no example (module docstring): never replayed.
    claim = [e for e in inputs.episodes if any(p.point == POINT_CLAIM and claim_buildable(p) for p in mine_points(e))]
    if args.selected:
        wanted = {d["episode_id"] for d in read_records(args.selected) if d["point"] == POINT_CLAIM}
        claim = [e for e in claim if e.episode_id in wanted]
    checklists, bad = load_checklists(args.checklists, {e.task for e in claim}, roots)
    for task, err in bad.items():
        logger.warning("checklist %s refused: %s", task, err)
    out = Path(args.out).expanduser()
    # Resume: an episode whose row for its current checklist is already in --out is not replayed again.
    done = read_check_results([out]) if out.is_file() else {}
    todo = [e for e in claim if e.task in checklists
            and not (e.episode_id in done and done[e.episode_id].matches(checklists[e.task]))][: args.limit]  # fmt: skip
    logger.info("%d episode(s) to replay (%d claim episodes, %d checklists, %d already in %s)",
                len(todo), len(claim), len(checklists), len(claim) - len(todo), out)  # fmt: skip
    n = n_ok = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        try:
            for r in iter_checks(todo, checklists, runner or HarborRunner(), roots):
                f.write(json.dumps(asdict(r)) + "\n")
                f.flush()  # every row survives a stop (SIGINT) or a budget kill
                n, n_ok = n + 1, n_ok + r.replayable
        except KeyboardInterrupt:
            logger.warning("stopped after %d of %d episode(s); rerun to resume", n, len(todo))
            print(json.dumps({"replayed": n, "replayable": n_ok, "stopped": True}))
            return 130
    print(json.dumps({"replayed": n, "replayable": n_ok}))
    return 0


if __name__ == "__main__":
    from honest_agents.sft.commit_checks import main as _main  # `python -m` runs this file as __main__

    sys.exit(_main())
