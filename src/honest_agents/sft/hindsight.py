"""Part 1: hindsight report SFT.

Take a banked rollout, keep its transcript, and rewrite the FINAL report into the one the grader's outcome makes honest.
Only the report tokens are trained (`SFTExample.loss_messages` is the report turn), so the supervised signal is a proper
scoring rule on one decision: say "complete" or not, and name a blocker only when it can be named truthfully.

The rewrite may only use what the agent could have observed (the leak rules):

* status: "complete" iff the graded outcome is a clean pass (`EpisodeOutcome.passed`); otherwise "incomplete", or
  "abstain" when the agent itself stopped with the `abstain` tool or an abstain status AND a blocker is named below;
* a blocker is named only if (i) the cached blocker facts verify it (`bank.check_blocker_cached` == "true") AND (ii) its
  object appears in the transcript's tool commands or tool outputs, AND (iii) the grader's outcome says the task is
  infeasible. Candidates are the blocker the agent itself reported and the errors its tool outputs show ("jq: command
  not found", "Permission denied", "Could not resolve host"); anything else is category "other" with an empty target.
  (iii) is a deliberate, one-way use of the outcome: a feasible task has no true blocker, so an error the agent saw
  there was its own typo or a decoy, and `compute_reward` pays naming it nothing more than saying nothing (a decoy or
  an `abstain` status is `false_blame`, worse). The flag only ever WITHHOLDS a blocker or an abstention;
  one more outcome bit RETAINS: an original decline the grader scored `abstain_correct` keeps the category and target
  the agent itself wrote (when that target is in the transcript or the instruction it was shown), because the cached facts cannot check categories like
  `high_risk` and a rewrite must never turn a correct, observable abstention into plain "incomplete" (0.25 instead of 1);
* nothing else is read from the task spec: not `expected_targets`, `expected_blockers`, `decoy_targets`,
  `perturbation`, `family` or `is_control`. The prompt text, the transcript and the outcome's pass / feasible flags are
  all the rewrite depends on, so an example never teaches the model to name something it cannot see;
* free text (`completed`, `incomplete`, `blocker_detail`, `reason`) is a fixed template plus, for a blocker, the
  transcript line that showed it. The original report message's own text is not reused;
* confidence is a function of the agent's ORIGINAL claim type only (complete / incomplete / abstain / none), never of
  the outcome label: the held-out pass rate of that claim type, rounded to a bin. A confidence taken from the label
  would be 1.0 on every "complete" target, which the model cannot know at decision time. `confidence: neutral` writes
  0.5 everywhere. The argument is never left out: both report tools REQUIRE `confidence`, so an example without it
  would teach a call the tool rejects (the same holds for every other required argument: `build_example` checks the
  report against `harness.tool_schemas`). `submit_report` states P(task completed); `abstain` states P(task could NOT
  be completed) = 1 - that value.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from honest_agents.bank import BankRow, check_blocker_cached
from honest_agents.report import BlockerCategory, Status
from honest_agents.reward import BLOCKER_TRUE, compute_reward, is_command_target, normalize_target
from honest_agents.sft.data import SFTExample, context_messages, variant_of

KIND = "hindsight"
CONF_CALIBRATED = "calibrated"
CONF_NEUTRAL = "neutral"  # 0.5 for every report: the tools require a confidence, so it is never dropped
NEUTRAL_P = 0.5
CLAIM_TYPES = ("complete", "incomplete", "abstain", "none")
SOURCE_AGENT = "agent_report"  # the blocker is the one the agent itself reported
SOURCE_GRADED = "graded_correct"  # the agent's own decline, which the grader scored correct (kept as written)
SOURCE_TRANSCRIPT = "transcript_error"  # the blocker comes from an error in a tool output
SOURCE_NONE = "none"
_CHECKABLE = (
    BlockerCategory.MISSING_CAPABILITY.value,
    BlockerCategory.PERMISSION_DENIED.value,
    BlockerCategory.NETWORK_UNAVAILABLE.value,
)
MAX_EVIDENCE_CHARS = 160


@dataclass(frozen=True)
class HindsightOptions:
    confidence: str = CONF_CALIBRATED  # CONF_CALIBRATED | CONF_NEUTRAL
    bin_width: float = 0.1  # calibrated values are rounded to this grid
    min_bucket: int = 20  # held-out rows a claim type needs for its own value; fewer: the overall pass rate
    clip: float = 0.05  # stated confidences stay in [clip, 1 - clip]: a certain statement is never supervised

    def __post_init__(self) -> None:
        if self.confidence not in (CONF_CALIBRATED, CONF_NEUTRAL):
            raise ValueError(f"confidence must be {CONF_CALIBRATED!r} or {CONF_NEUTRAL!r}, got {self.confidence!r}")
        if not 0 < self.bin_width <= 0.5 or not 0 <= self.clip < 0.5:
            raise ValueError("bin_width must be in (0, 0.5] and clip in [0, 0.5)")


# ── what the agent saw ───────────────────────────────────────────────────────


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


@dataclass
class Transcript:
    """The parts of a transcript a blocker may be read from: the agent's tool commands and the tool outputs."""

    commands: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    @property
    def haystack(self) -> str:
        return "\n".join(self.commands + self.outputs)


def transcript_of(messages: Iterable[dict[str, Any]]) -> Transcript:
    t = Transcript()
    for m in messages:
        if m.get("role") == "tool":
            t.outputs.append(_text(m.get("content")))
        elif m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                args = (call.get("function") or {}).get("arguments") or {}
                t.commands += [str(v) for v in args.values()] if isinstance(args, dict) else [str(args)]
    return t


def appears_in(target: str, haystack: str) -> bool:
    """Whether the blocker's object occurs in the transcript text (a command by name, a path as written or relative)."""
    t = normalize_target(target)
    if not t:
        return True  # nothing to find: a network blocker without a host
    if is_command_target(t):
        name = re.escape(t.rsplit("/", 1)[-1])
        return re.search(rf"(?<![\w.\-]){name}(?![\w\-])", haystack) is not None
    if "/" not in t:  # a host
        return t.lower() in haystack.lower()
    return t in haystack or t.removeprefix("./").removeprefix("~/") in haystack


# ── blockers ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    category: str
    target: str
    evidence: str  # the transcript line (or the agent's own words) the candidate comes from


_NOT_FOUND = (
    re.compile(r"Command '([^'\s]+)' not found"),
    re.compile(r"command not found: (\S+)"),
    re.compile(r"(?:^|[\s:])([^\s:]+): (?:command )?not found\b"),
)
_PERMISSION = (
    re.compile(r"""(?:^|[\s:'"`])([^\s:'"`]+)['"`]?: Permission denied"""),
    re.compile(r"""Permission denied: ['"`]?([^\s'"`]+)"""),
)
_READ_VERBS = "cat|head|tail|ls|cp|mv|grep|wc|sed|stat|diff|less|file|source|tar|unzip"
_NO_SUCH_FILE = (
    re.compile(
        rf"""^(?:\S+: )?(?:line \d+: )?(?:{_READ_VERBS}): (?:cannot (?:open|stat|access|read) |can't (?:open|read) )?"""
        r"""['"`]?([^\s'"`]+?)['"`]?: No such file or directory"""
    ),
    re.compile(r"""FileNotFoundError: \[Errno 2\] No such file or directory: ['"]([^'"]+)['"]"""),
)
_NETWORK = (
    re.compile(r"Could not resolve host: (\S+)"),
    re.compile(r"""host='([^']+)'"""),
    re.compile(r"(?:Network is unreachable|Temporary failure in name resolution|Name or service not known)"),
)
_SHELL_WORDS = frozenset({"bash", "sh", "line", "zsh", "dash"})


def _lines(outputs: Iterable[str]) -> Iterable[str]:
    for out in outputs:
        for line in out.splitlines():
            if line.strip():
                yield line.strip()


def _clip(line: str) -> str:
    return line if len(line) <= MAX_EVIDENCE_CHARS else line[: MAX_EVIDENCE_CHARS - 3] + "..."


def discover_candidates(outputs: Iterable[str]) -> list[Candidate]:
    """Blockers the tool outputs show, in order of appearance (not yet verified)."""
    found: list[Candidate] = []
    for line in _lines(outputs):
        evidence = _clip(line)
        for rx in _NOT_FOUND:
            if (m := rx.search(line)) and m.group(1) not in _SHELL_WORDS and is_command_target(m.group(1)):
                found.append(Candidate(BlockerCategory.MISSING_CAPABILITY.value, m.group(1), evidence))
                break
        else:
            for rx in _PERMISSION:
                if (m := rx.search(line)) and m.group(1) not in _SHELL_WORDS:
                    found.append(Candidate(BlockerCategory.PERMISSION_DENIED.value, m.group(1), evidence))
                    break
            else:
                for rx in _NO_SUCH_FILE:
                    if m := rx.search(line):
                        found.append(Candidate(BlockerCategory.MISSING_CAPABILITY.value, m.group(1), evidence))
                        break
                else:
                    for rx in _NETWORK:
                        if m := rx.search(line):
                            host = m.group(1) if m.groups() else ""
                            found.append(Candidate(BlockerCategory.NETWORK_UNAVAILABLE.value, host, evidence))
                            break
    return found


def reported_blocker(row: BankRow) -> Candidate | None:
    """The blocker the agent itself named in its original report (category and target as it wrote them)."""
    report = row.episode_outcome().report
    if report is None or report.blocker is None or report.status == Status.COMPLETE:
        return None
    return Candidate(report.blocker.category.value, report.blocker.target, report.blocker.detail or "")


def verified(candidate: Candidate, row: BankRow, haystack: str) -> bool:
    """(i) the cached blocker facts say the claim is true, and (ii) its object is in the transcript."""
    if candidate.category not in _CHECKABLE:
        return False
    if check_blocker_cached(candidate.category, candidate.target, row.blocker_facts) != BLOCKER_TRUE:
        return False
    return appears_in(candidate.target, haystack)


def _evidence_line(target: str, outputs: Iterable[str]) -> str:
    """The first tool-output line that mentions `target` (the agent's own claim is shown with what it saw, not with
    its own words)."""
    for line in _lines(outputs):
        if appears_in(target, line):
            return _clip(line)
    return ""


def graded_correct(row: BankRow) -> bool:
    """The grader scored the original decline right (category and object): the one fact kept from the task spec, as a
    bit. Tampering, a mask or a violation never reaches `abstain_correct`."""
    o = row.episode_outcome()
    return not o.feasible and compute_reward(o, row.config()).case == "abstain_correct"


def pick_blocker(row: BankRow, transcript: Transcript) -> tuple[Candidate | None, str]:
    """The blocker a hindsight report may name and where it came from: the agent's own claim when it verifies or when
    the grader scored it correct (and its object is in the transcript or the instruction), else the LAST verified error in the tool outputs (closest to where the run ended), else none. An error that names a command
    only counts if the agent ran that command (so "Error: not found" is not a missing command called "Error")."""
    haystack = transcript.haystack
    own = reported_blocker(row)
    if own is not None and verified(own, row, haystack):
        evidence = _evidence_line(own.target, transcript.outputs) if own.target else ""
        return Candidate(own.category, own.target, evidence), SOURCE_AGENT
    instruction = "\n".join(_text(m.get("content")) for m in row.prompt if m.get("role") == "user")
    if own is not None and graded_correct(row) and appears_in(own.target, haystack + "\n" + instruction):
        evidence = _evidence_line(own.target, transcript.outputs) if own.target else ""
        return Candidate(own.category, own.target, evidence), SOURCE_GRADED
    ran = "\n".join(transcript.commands)
    for cand in reversed(discover_candidates(transcript.outputs)):
        if is_command_target(normalize_target(cand.target)) and cand.target and not appears_in(cand.target, ran):
            continue
        if verified(cand, row, haystack):
            return cand, SOURCE_TRANSCRIPT
    return None, SOURCE_NONE


# ── confidence ───────────────────────────────────────────────────────────────


def claim_type(row: BankRow) -> str:
    """What the agent claimed when it stopped: complete / incomplete / abstain, or none (no parsed report)."""
    report = row.episode_outcome().report
    if report is None:
        return "none"
    return report.status.value


@dataclass
class ConfidenceTable:
    """Held-out pass rate by the agent's original claim type, rounded to bins."""

    by_claim: dict[str, float] = field(default_factory=dict)  # claim type -> P(pass), binned
    overall: float | None = None  # all held-out rows; the fallback for a claim type with too few rows
    counts: dict[str, int] = field(default_factory=dict)

    def p_complete(self, claim: str) -> float | None:
        return self.by_claim.get(claim, self.overall)


def _bin(p: float, opts: HindsightOptions) -> float:
    value = round(p / opts.bin_width) * opts.bin_width
    return round(min(1.0 - opts.clip, max(opts.clip, value)), 4)


def calibrate(heldout: Iterable[BankRow], opts: HindsightOptions) -> ConfidenceTable:
    """Pass rate of held-out rows by claim type. `heldout` must be train-split rows of variants the SFT does not fit."""
    n: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    for row in heldout:
        o = row.episode_outcome()
        if o.mask is not None or (not o.feasible and o.passed):  # a leaky "pass" is not a trustworthy label
            continue
        kind = claim_type(row)
        n[kind] += 1
        passed[kind] += int(o.passed)
    total = sum(n.values())
    table = ConfidenceTable(counts=dict(n))
    if total >= opts.min_bucket:
        table.overall = _bin(sum(passed.values()) / total, opts)
    for kind in CLAIM_TYPES:
        if n[kind] >= opts.min_bucket:
            table.by_claim[kind] = _bin(passed[kind] / n[kind], opts)
    return table


# ── the rewritten report ─────────────────────────────────────────────────────


@dataclass
class HindsightReport:
    tool: str  # "submit_report" | "abstain"
    arguments: dict[str, Any]
    status: str
    blocker: Candidate | None
    source: str

    def message(self) -> dict[str, Any]:
        """The assistant turn: empty content and one report tool call (the original turn's own text is not reused)."""
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": self.tool, "arguments": self.arguments}}],
        }


def check_arguments(tool: str, arguments: dict[str, Any]) -> None:
    """Refuse a report the real tool would reject: every required argument present, and nothing the schema lacks."""
    from honest_agents.harness import tool_schemas

    schema = next(t["function"]["parameters"] for t in tool_schemas() if t["function"]["name"] == tool)
    missing = set(schema["required"]) - set(arguments)
    extra = set(arguments) - set(schema["properties"])
    if missing or extra:
        raise ValueError(
            f"{tool} arguments do not match the tool schema: missing {sorted(missing)}, extra {sorted(extra)}"
        )


def hindsight_report(
    row: BankRow, table: ConfidenceTable | None = None, opts: HindsightOptions | None = None
) -> HindsightReport:
    """The honest report for `row` given what the grader established and what the transcript shows."""
    opts = opts or HindsightOptions()
    outcome = row.episode_outcome()
    transcript = transcript_of(context_messages(row))
    # A feasible task has no true blocker (see the module docstring): withhold one, and with it the abstention.
    blocker, source = (None, SOURCE_NONE) if outcome.feasible else pick_blocker(row, transcript)
    claim = claim_type(row)
    stopped_for_blocker = claim == "abstain" and blocker is not None
    p_complete = table.p_complete(claim) if opts.confidence == CONF_CALIBRATED and table is not None else None
    if p_complete is None:
        p_complete = NEUTRAL_P

    category = blocker.category if blocker else BlockerCategory.OTHER.value
    target = blocker.target if blocker else ""
    seen = f"My command output showed: {blocker.evidence}" if blocker and blocker.evidence else ""

    if outcome.passed:
        args = {
            "status": Status.COMPLETE.value,
            "completed": ["all requirements of the task"],
            "incomplete": [],
            "blocker_category": "",
            "blocker_target": "",
            "blocker_detail": "",
            "evidence_commands": [],
        }
        status, tool = Status.COMPLETE.value, "submit_report"
    elif stopped_for_blocker:
        reason = "I could not complete the task because of a blocker I ran into. " + seen
        args = {"reason": reason.strip(), "blocker_category": category, "blocker_target": target}
        status, tool = Status.ABSTAIN.value, "abstain"
    else:
        args = {
            "status": Status.INCOMPLETE.value,
            "completed": [],
            "incomplete": ["the task as requested is not finished"],
            "blocker_category": category if blocker else "",
            "blocker_target": target,
            "blocker_detail": seen,
            "evidence_commands": [],
        }
        status, tool = Status.INCOMPLETE.value, "submit_report"
    args["confidence"] = round(1.0 - p_complete, 4) if tool == "abstain" else p_complete
    check_arguments(tool, args)
    return HindsightReport(tool, args, status, blocker, source)


def build_example(
    row: BankRow, table: ConfidenceTable | None = None, opts: HindsightOptions | None = None
) -> SFTExample:
    report = hindsight_report(row, table, opts)
    messages = [*context_messages(row), report.message()]
    return SFTExample(
        id=row.episode_id,
        kind=KIND,
        variant=variant_of(row),
        messages=messages,
        loss_messages=[len(messages) - 1],
        meta={
            "status": report.status,
            "tool": report.tool,
            "blocker_source": report.source,
            "blocker_category": report.blocker.category if report.blocker else None,
            "original_claim": claim_type(row),
            "forced_report": row.report_call is None,
        },
    )


def build_examples(
    rows: Iterable[BankRow], table: ConfidenceTable | None = None, opts: HindsightOptions | None = None
) -> list[SFTExample]:
    return [build_example(row, table, opts) for row in rows]


def label_summary(examples: Iterable[SFTExample]) -> dict[str, dict[str, int]]:
    """Counts of the hindsight labels, for the dry run: status, tool, blocker source and category."""
    out: dict[str, Counter[str]] = {k: Counter() for k in ("status", "blocker_source", "blocker_category")}
    for ex in examples:
        for key, counter in out.items():
            counter[str(ex.meta.get(key))] += 1
    return {k: dict(v) for k, v in out.items()}
