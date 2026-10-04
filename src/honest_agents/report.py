"""Structured self-report the agent submits at the end of an episode.

The report is the agent's *claim* about the final environment state. It is never trusted: the grader
(`honest_agents.grading`) checks it against the sandbox programmatically.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError, field_validator


class Status(StrEnum):
    COMPLETE = "complete"  # agent claims the task is fully done
    INCOMPLETE = "incomplete"  # agent attempted but reports (part of) the task as not done
    ABSTAIN = "abstain"  # agent explicitly declines to (finish) the task, with a reason


class BlockerCategory(StrEnum):
    """Why a task cannot (or should not) be completed.

    The first eight mirror AgentAbstain's scenario taxonomy; `permission_denied` and `network_unavailable` are
    terminal-specific refinements that our perturbations produce.
    """

    MISSING_PARAMETER = "missing_parameter"
    AMBIGUOUS_ACTION = "ambiguous_action"
    CONFLICTING_CONSTRAINTS = "conflicting_constraints"
    HIGH_RISK = "high_risk"
    MISSING_CAPABILITY = "missing_capability"  # required tool / dependency / file is absent
    TOOL_FAILURE = "tool_failure"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    RISK_DISCOVERED = "risk_discovered"
    PERMISSION_DENIED = "permission_denied"
    NETWORK_UNAVAILABLE = "network_unavailable"
    OTHER = "other"


class Blocker(BaseModel):
    category: BlockerCategory = BlockerCategory.OTHER
    detail: str = ""


class SelfReport(BaseModel):
    status: Status
    completed: list[str] = Field(default_factory=list)
    incomplete: list[str] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    # Shell commands that should exit 0 iff the claims hold. Replayed by the grader in a clean shell.
    evidence_commands: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("blockers", mode="before")
    @classmethod
    def _coerce_blockers(cls, v):
        return [parse_blocker(b) for b in _as_list(v)]

    @field_validator("completed", "incomplete", "evidence_commands", mode="before")
    @classmethod
    def _coerce_str_list(cls, v):
        return [str(x) for x in _as_list(v) if str(x).strip()]

    @property
    def claims_complete(self) -> bool:
        return self.status == Status.COMPLETE

    @property
    def explains_non_completion(self) -> bool:
        """An explicit, reasoned refusal/non-completion (as opposed to stopping silently)."""
        if self.status == Status.COMPLETE:
            return False
        return bool(self.blockers) or bool(self.reason.strip())

    @property
    def blocker_categories(self) -> set[str]:
        return {b.category.value for b in self.blockers}


def _as_list(v) -> list:
    """Small models emit lists as JSON strings, newline-separated strings, or scalars. Accept all of them."""
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [line.strip(" -*\t") for line in s.splitlines() if line.strip(" -*\t")]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def parse_blocker(b) -> Blocker:
    """Accept `Blocker`, a dict, or a `"category: detail"` string."""
    if isinstance(b, Blocker):
        return b
    if isinstance(b, dict):
        return Blocker(category=_coerce_category(b.get("category", "other")), detail=str(b.get("detail", "")))
    s = str(b)
    head, sep, tail = s.partition(":")
    if sep:
        cat = _coerce_category(head)
        if cat != BlockerCategory.OTHER or head.strip().lower() == "other":
            return Blocker(category=cat, detail=tail.strip())
    return Blocker(category=BlockerCategory.OTHER, detail=s.strip())


def _coerce_category(s) -> BlockerCategory:
    key = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return BlockerCategory(key)
    except ValueError:
        return BlockerCategory.OTHER


class ReportParseError(ValueError):
    pass


def parse_report(raw: str | dict) -> SelfReport:
    """Parse a report from a dict or a JSON string (tolerating markdown code fences)."""
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s.split("\n", 1)[1] if "\n" in s else s
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1:
            raise ReportParseError("no JSON object found in report")
        try:
            raw = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ReportParseError(f"invalid JSON: {e}") from e
    try:
        return SelfReport.model_validate(raw)
    except ValidationError as e:
        raise ReportParseError(str(e)) from e
