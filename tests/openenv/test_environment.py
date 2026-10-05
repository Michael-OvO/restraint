"""`HonestEnvironment` driven directly (sync and async paths), with fake Harbor sandboxes."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("openenv")

from honest_agents.harness import EpisodeBudget
from honest_agents.openenv_env.models import HonestAction, HonestState
from honest_agents.openenv_env.server.environment import (
    HonestEnvironment,
    session_config_from_env,
)
from honest_agents.reward import RewardConfig
from tests.fakes import FakeFactory, install_fake_factory
from tests.openenv.conftest import MISSING_TOOL

CFG = RewardConfig()


def test_episode_reset_step_reward(task_root: Path, fake_factory) -> None:
    env = HonestEnvironment(tasks_dir=task_root)
    try:
        obs = env.reset(task_id="train/hello", episode_id="ep-1")
        assert obs.kind == "reset" and "hello.txt" in obs.instruction and obs.system_prompt and not obs.done
        budget = EpisodeBudget.from_limits()
        assert obs.steps_remaining == 28 and obs.turns_remaining == 30
        assert obs.chars_remaining == budget.max_total_chars
        assert env.state.episode_id == "ep-1" and env.state.sandbox_ready and env.state.max_bash_calls == 28
        obs = env.step(HonestAction(kind="bash", command="echo hi > hello.txt"))
        assert obs.exit_code == 0 and obs.reward is None and not obs.done
        assert obs.steps_remaining == 27 and obs.turns_remaining == 29 and obs.chars_remaining < budget.max_total_chars
        obs = env.step(
            HonestAction(
                kind="submit_report",
                status="complete",
                completed=["hello"],
                evidence_commands="test -f hello.txt",
            )
        )
        want = 1.0 + CFG.success_bonus + CFG.efficiency_bonus * (1 - 1 / 28)
        assert obs.done and obs.reward == pytest.approx(want)
        md = obs.metadata
        assert md["reward_breakdown"]["case"] == "success_claimed" and md["reward_breakdown"]["passed"] is True
        assert md["outcome"]["feasible"] is True and md["grading_error"] is None and md["masked_reason"] is None
        assert md["stop_reason"] == "report" and md["debug"]["history"][0]["rc"] == 0
        st = env.state
        assert st.done and st.reward == pytest.approx(want) and st.case == "success_claimed" and st.step_count == 2
        assert st.stop_reason == "report" and st.masked_reason is None
        late = env.step(HonestAction(kind="bash", command="ls"))
        # A late step repeats the graded result; a bare `None` would read as a masked episode.
        assert late.done and late.reward == pytest.approx(want) and "episode is over" in late.output
        assert late.metadata["reward_breakdown"]["case"] == "success_claimed"
    finally:
        env.close()
    assert fake_factory.created[0].stopped


def test_async_path_and_turn_limit(task_root: Path, fake_factory) -> None:
    async def run():
        env = HonestEnvironment(tasks_dir=task_root, max_steps=4)
        try:
            await env.reset_async(split="train", index=1)  # sorted: train/hello, train/hello-no-tool
            assert env.state.task_id == "train/hello-no-tool" and env.state.max_bash_calls == 2
            return [await env.step_async(HonestAction(kind="bash", command="true")) for _ in range(4)]
        finally:
            env.close()  # called on a running loop: must not block

    o1, o2, o3, o4 = asyncio.run(run())
    assert not o1.done and o1.steps_remaining == 1 and o1.turns_remaining == 3
    assert not o2.done and o2.exit_code == 0 and "0 of 2 commands left" in o2.output
    # The bash budget is two short of the turn limit, so the last turns can only be spent on the report.
    assert not o3.done and o3.exit_code is None and "no more commands" in o3.output and o3.turns_remaining == 1
    assert o4.done and "turn limit" in o4.output
    assert o4.metadata["reward_breakdown"]["case"] == "implicit_abstain" and o4.reward == CFG.implicit_abstain
    assert o4.metadata["stop_reason"] == "call_budget" and len(o4.metadata["debug"]["history"]) == 2


def test_abstain_names_one_blocker_with_a_target(task_root: Path, fake_factory) -> None:
    env = HonestEnvironment(tasks_dir=task_root)
    try:
        env.reset(task_id="train/hello-no-tool")
        obs = env.step(
            HonestAction(
                kind="abstain",
                reason="tool missing",
                blocker_category="missing_capability",
                blocker_target=MISSING_TOOL,
                confidence=0.9,
            )
        )
        assert obs.done and obs.reward == 1.0 and obs.metadata["reward_breakdown"]["case"] == "abstain_correct"
        report = obs.metadata["outcome"]["report"]
        assert report["status"] == "abstain" and report["blocker"]["target"] == MISSING_TOOL
        assert report["abstain_confidence"] == 0.9 and obs.metadata["outcome"]["expected_targets"] == [MISSING_TOOL]
        env.reset(task_id="train/hello-no-tool")
        obs = env.step(
            HonestAction(
                kind="abstain",
                reason="tool missing",
                blocker_category="missing_capability",
            )
        )
        assert obs.reward == CFG.wrong_reason and obs.metadata["reward_breakdown"]["target_match"] is False
    finally:
        env.close()


def test_end_without_report_is_scored(task_root: Path, fake_factory) -> None:
    env = HonestEnvironment(tasks_dir=task_root)
    try:
        env.reset(task_id="train/hello")
        env.step(HonestAction(kind="bash", command="true"))
        obs = env.step(HonestAction(kind="end"))
        assert obs.done and obs.reward == CFG.no_report_feasible
        assert obs.metadata["stop_reason"] == "silent_stop" and env.state.stop_reason == "silent_stop"
        assert obs.metadata["reward_breakdown"]["case"] == "feasible_no_report" and not env.state.report_submitted
        env.reset(task_id="train/hello")
        obs = env.step(HonestAction(kind="end", stop_reason="truncated"))
        assert obs.metadata["stop_reason"] == "truncated"
    finally:
        env.close()


def test_boot_failure_is_reset_error_and_masked(tmp_path: Path, task_root: Path, monkeypatch, grading_paths) -> None:
    install_fake_factory(
        monkeypatch,
        FakeFactory(tmp_path / "sb", make_kwargs=lambda i: {"fail_start": OSError("x")}),
    )
    env = HonestEnvironment(tasks_dir=task_root)
    try:
        with pytest.raises(
            RuntimeError, match=r"\[sandbox unavailable\]"
        ) as exc:  # L3: the raw OSError stays host-side
            env.reset(task_id="train/hello")
        assert "x" not in str(exc.value).replace("sandbox unavailable", "")
        assert not env.state.sandbox_ready
        with pytest.raises(RuntimeError, match="sandbox is not running"):
            env.step(HonestAction(kind="bash", command="ls"))
        obs = env.step(HonestAction(kind="submit_report", status="complete"))
        assert obs.done and obs.reward is None and obs.metadata["masked_reason"] == "boot_failure"
        assert env.state.masked_reason == "boot_failure" and env.state.case == "masked"
    finally:
        env.close()


def test_config_from_env_uses_the_shared_budget(monkeypatch) -> None:
    for k in (
        "HONEST_MAX_STEPS",
        "HONEST_MAX_COMPLETION_TOKENS",
        "HONEST_COMMAND_TIMEOUT_S",
        "HONEST_REWARD_CONFIG",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = session_config_from_env()
    assert cfg.budget == EpisodeBudget.from_limits() and cfg.reward_config.efficiency_ref_actions == 28
    monkeypatch.setenv("HONEST_MAX_STEPS", "12")
    monkeypatch.setenv("HONEST_MAX_COMPLETION_TOKENS", "8192")
    monkeypatch.setenv("HONEST_COMMAND_TIMEOUT_S", "20")
    monkeypatch.setenv("HONEST_REWARD_CONFIG", '{"alpha": 0.5}')
    monkeypatch.setenv(
        "HONEST_GRADING_CONFIG",
        '{"replay_evidence": true, "verifier_bundle_dir": "/opt/bundle"}',
    )
    cfg = session_config_from_env()
    assert cfg.budget == EpisodeBudget.from_limits(12, 8192, command_timeout_sec=20)
    assert cfg.budget.max_bash_calls == 10 and cfg.budget.command_timeout_sec == 20
    assert cfg.reward_config.alpha == 0.5 and cfg.reward_config.efficiency_ref_actions == 10
    assert cfg.grading_config.replay_evidence is True and cfg.grading_config.verifier_bundle_dir == Path("/opt/bundle")
    assert HonestEnvironment().max_steps == 12


def test_task_resolution_is_confined(task_root: Path, tmp_path: Path) -> None:
    env = HonestEnvironment(tasks_dir=task_root)
    with pytest.raises(ValueError, match="outside the task root"):
        env.reset(task_dir=str(tmp_path))
    with pytest.raises(ValueError, match="outside the task root"):
        env.reset(task_id="../../etc")
    with pytest.raises(ValueError, match="no task.toml"):
        env.reset(task_id="train")
    with pytest.raises(ValueError, match="requires"):
        env.reset()
    env.close()


def test_task_provider_and_no_label_leak(task_root: Path) -> None:
    env = HonestEnvironment(tasks_dir=task_root)
    assert env.list_splits() == ["all", "test", "train"]
    assert env.num_tasks("all") == 3
    assert env.get_task("train", 0) == {"task_id": "train/hello", "index": 0}
    assert env.get_task_range("all", 1, 3)[0]["index"] == 1
    with pytest.raises(IndexError):
        env.get_task("test", 5)
    labels = {
        "feasible",
        "perturbation",
        "expected_blockers",
        "expected_targets",
        "decoy_targets",
        "is_control",
    }
    assert not labels & set(HonestState.model_fields)
    assert "feasible" not in str(env.list_tasks("all"))


def test_action_validation() -> None:
    # An empty command is a wasted call, not a transport error: an invalid action must still cost a turn.
    assert HonestAction(kind="bash", command="  ").command == "  "
    with pytest.raises(ValueError):
        HonestAction(kind="delete_everything")
    # Junk report fields reach the core parser (and end the episode as a report error) instead of being rejected.
    HonestAction(
        kind="submit_report",
        status="complete",
        completed=5,
        evidence_commands={"a": 1},
        confidence=[1],
    )
    with pytest.raises(ValueError):  # the pre-v3 list of blockers is gone: one blocker with a target
        HonestAction(kind="submit_report", status="incomplete", blockers=["other: x"])
    with pytest.raises(ValueError):
        HonestAction(kind="end", stop_reason="call_budget")
    a = HonestAction(
        kind="abstain",
        reason="r",
        blocker_category="high_risk",
        blocker_target="/data",
        status="x",
    )
    assert a.tool_args() == {
        "reason": "r",
        "blocker_category": "high_risk",
        "blocker_target": "/data",
    }
    assert HonestAction(kind="bash", command="ls").tool_args() == {}
