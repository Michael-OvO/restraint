"""The OpenEnv wrapper must expose the tools of the training harness argument for argument, and the same prompt."""

from __future__ import annotations

import inspect

import pytest

from honest_agents import harness
from honest_agents.openenv_env import session


def _params(fn) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


def test_prompt_and_budget_come_from_the_shared_harness() -> None:
    assert session.SYSTEM_PROMPT is harness.SYSTEM_PROMPT
    assert session.EpisodeBudget is harness.EpisodeBudget
    cfg = session.SessionConfig()
    assert (cfg.max_tool_iterations, cfg.max_completion_tokens) == (
        harness.DEFAULT_MAX_TOOL_ITERATIONS,
        harness.DEFAULT_MAX_COMPLETION_TOKENS,
    )


def test_action_fields_match_the_report_tools() -> None:
    pytest.importorskip("openenv")
    from honest_agents.openenv_env.client import HonestEnvClient
    from honest_agents.openenv_env.models import REPORT_FIELDS, HonestAction

    assert tuple(REPORT_FIELDS) == session.REPORT_TOOLS
    for tool, fields in REPORT_FIELDS.items():
        want = _params(getattr(harness.ReportTools, tool))
        assert list(fields) == want, tool
        assert set(fields) <= set(HonestAction.model_fields), tool
        assert _params(getattr(HonestEnvClient, tool)) == want, tool
    assert "blockers" not in HonestAction.model_fields  # one blocker with a target, not a list


def test_tools_match_the_trl_env() -> None:
    """Runs where trl is installed (the training venv)."""
    env_mod = pytest.importorskip("honest_agents.env", exc_type=ImportError)
    cls = env_mod.HonestHarborEnv
    for tool in session.REPORT_TOOLS:
        assert _params(getattr(cls, tool)) == _params(getattr(harness.ReportTools, tool))
    assert _params(cls.bash) == ["command"]
