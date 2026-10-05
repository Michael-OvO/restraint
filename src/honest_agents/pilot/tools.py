"""The two report tools the pilot exposes, taken from the shared harness so the schemas cannot drift.

`honest_agents.harness.ReportTools` (docs/INTERFACES_V3.md §2) is the single definition of `submit_report` and
`abstain` for every harness of ours; `ReplayEnv` inherits it. The pilot has no `bash`: the frozen transcript already
holds every command that was run.
"""

from __future__ import annotations

import inspect
from typing import Any

from honest_agents import harness

ReportTools = harness.ReportTools

# Parameter names of the two tools, in order, as fixed by docs/SPEC_V3.md §2 and docs/INTERFACES_V3.md §2.
CONTRACT_SIGNATURES: dict[str, tuple[str, ...]] = {
    "submit_report": (
        "status",
        "completed",
        "incomplete",
        "blocker_category",
        "blocker_target",
        "blocker_detail",
        "evidence_commands",
        "confidence",
    ),
    "abstain": ("reason", "blocker_category", "blocker_target", "confidence"),
}
# JSON-schema type of every parameter, as the contract's annotations give them.
CONTRACT_TYPES: dict[str, dict[str, str]] = {
    "submit_report": {
        "status": "string",
        "completed": "array",
        "incomplete": "array",
        "blocker_category": "string",
        "blocker_target": "string",
        "blocker_detail": "string",
        "evidence_commands": "array",
        "confidence": "number",
    },
    "abstain": {
        "reason": "string",
        "blocker_category": "string",
        "blocker_target": "string",
        "confidence": "number",
    },
}
REPORT_TOOL_NAMES: tuple[str, ...] = tuple(CONTRACT_SIGNATURES)


def tool_parameters(cls: type, name: str) -> tuple[str, ...]:
    """Parameter names of tool `name` on `cls`, without `self`."""
    return tuple(p for p in inspect.signature(getattr(cls, name)).parameters if p != "self")


def tool_schemas(env: Any = None) -> list[dict]:
    """JSON schemas of the report tools as TRL builds them (`get_json_schema` on the bound methods of `env`), or, with
    no `env`, as `harness.tool_schemas(include_bash=False)` builds them."""
    if env is None:
        return harness.tool_schemas(include_bash=False)
    from transformers.utils import get_json_schema

    return [get_json_schema(getattr(env, name)) for name in REPORT_TOOL_NAMES]
