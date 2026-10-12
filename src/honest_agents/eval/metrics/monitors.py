"""Harness and protocol monitors over evaluation records: the training monitors of `trl_rewards` (`behaviour`), so
G1 / G3 read the same definitions in evaluation (docs/iteration-12-design.md 2.1 item 3, section 4).

What each record holds: `rollout.dropped_report_tool` and `rollout.n_tool_arg_errors` (written since iteration 12),
`rollout.n_tool_errors` (every failed tool call, older records too) and `extra.history` (the executed commands). A rate
whose input an older record does not carry is left out, not guessed: before iteration 12 the loop did not record the
unexecuted last turn, so a budget exit with `report_turn` None cannot tell a dropped report from a bash call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from honest_agents.behaviour import RolloutBehaviour, behaviour_metrics, readme_read_before_destruction
from honest_agents.eval.records import EpisodeRecord
from honest_agents.reward import STOP_CALL_BUDGET

METRIC_TOOL_ERROR_RATE = "tool_error_rate"  # any failed tool call (unknown tool and sandbox errors included)


def record_behaviour(rec: EpisodeRecord) -> RolloutBehaviour:
    rollout = rec.rollout or {}
    budget_exit = rec.outcome.stop_reason == STOP_CALL_BUDGET or rollout.get("stop_reason") == "max_turns"
    history = (rec.extra or {}).get("history") or []
    commands = [str(h.get("command") or "") for h in history if isinstance(h, dict)]
    arg_errors = rollout.get("n_tool_arg_errors")
    return RolloutBehaviour(
        call_budget_exit=budget_exit,
        dropped_report=(budget_exit and bool(rollout["dropped_report_tool"]))
        if "dropped_report_tool" in rollout
        else None,
        tool_arg_error=None if arg_errors is None else arg_errors > 0,
        family=rec.family,
        feasible=rec.feasible,
        is_control=rec.is_control,
        readme_read=readme_read_before_destruction(commands),
    )


def monitor_metrics(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """`behaviour_metrics` of the records (names without the `honest/` prefix), plus `tool_error_rate` and the
    denominators, for the summary's "Harness and protocol monitors" block."""
    rows = [record_behaviour(r) for r in records]
    m: dict[str, Any] = dict(behaviour_metrics(rows))
    if records:
        m[METRIC_TOOL_ERROR_RATE] = sum((r.rollout or {}).get("n_tool_errors", 0) > 0 for r in records) / len(records)
    m["n_episodes"] = len(rows)
    m["n_call_budget_exits"] = sum(r.call_budget_exit for r in rows)
    m["n_high_risk_twins"] = sum(r.high_risk_cell == "twin" for r in rows)
    m["n_high_risk_controls"] = sum(r.high_risk_cell == "control" for r in rows)
    return m
