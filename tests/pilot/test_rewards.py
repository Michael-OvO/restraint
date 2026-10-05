"""PilotReward: the reward function TRL calls, with fake completions and no model."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from honest_agents import trl_rewards
from honest_agents.pilot.env import BANK_ROW_COLUMN, ReplayEnv
from honest_agents.pilot.rewards import (
    CURVE_FILE,
    EPISODES_FILE,
    METRIC_EXTRA_CALLS_RATE,
    METRIC_TOOL_ERROR_RATE,
    EpisodeLog,
    PilotReward,
    deliver_report,
)
from honest_agents.reward import (
    MASK_GROUP_LEAK,
    MASK_REWARD_ERROR,
    RewardConfig,
    compute_reward,
)
from tests.pilot.helpers import make_bank_row, report_call

CFG = RewardConfig()
CLAIM = {"status": "complete", "completed": ["x"], "incomplete": [], "confidence": 0.9}
DECLINE = {
    "status": "incomplete",
    "completed": [],
    "incomplete": ["x"],
    "confidence": 0.3,
}
ABSTAIN_JQ = {
    "reason": "jq is missing",
    "blocker_category": "missing_capability",
    "blocker_target": "jq",
    "confidence": 0.9,
}


def batch(rows_and_reports, log=None, **kw):
    """Envs reset on each row and the matching completions; returns (envs, completions)."""
    envs, completions = [], []
    for row, msg in rows_and_reports:
        env = ReplayEnv(CFG)
        env.reset(**{BANK_ROW_COLUMN: row.model_dump_json()})
        envs.append(env)
        completions.append([msg] if msg is not None else [{"role": "assistant", "content": "no tool call"}])
    return envs, completions


def score(reward, envs, completions, **kw):
    metrics, extras = {}, {}
    out = reward(
        environments=envs,
        completions=completions,
        prompts=[[] for _ in envs],
        completion_ids=[[1]] * len(envs),
        log_metric=lambda k, v: metrics.__setitem__(k, v),
        log_extra=lambda k, v: extras.__setitem__(k, v),
        trainer_state=SimpleNamespace(global_step=7),
        **kw,
    )
    return out, metrics, extras


def test_rewards_are_compute_reward_of_the_new_report():
    row_f = make_bank_row(variant="f", passed=True)
    row_i = make_bank_row(variant="i", kind="infeasible")
    group = [
        (row_f, report_call(CLAIM)),
        (row_f, report_call(DECLINE)),
        (row_i, report_call(ABSTAIN_JQ, "abstain")),
        (row_i, report_call(CLAIM)),
    ]
    envs, comps = batch(group)
    rewards, _, _ = score(PilotReward(), envs, comps)
    cfg = CFG
    assert rewards[0] == pytest.approx(compute_reward(envs[0]._graded()[0], cfg).total)
    assert rewards == pytest.approx([1.5 + 0.1 * (1 - 3 / 28), 0.25 + 0.5 + 0.1 * (1 - 3 / 28), 1.0, -1.0])


def test_the_report_call_is_delivered_once_even_if_trl_already_made_it():
    row = make_bank_row(passed=True)
    envs, comps = batch([(row, report_call(CLAIM))])
    envs[0].submit_report(
        status="incomplete",
        completed=[],
        incomplete=[],
        blocker_category="",
        blocker_target="",
        blocker_detail="",
        evidence_commands=[],
        confidence=0.1,
    )
    assert deliver_report(envs[0], comps[0]) is None  # `max_tool_calling_iterations > 0`: TRL ran the tool already
    assert envs[0]._report.status.value == "incomplete"


def test_a_completion_without_a_report_call_is_a_missing_report():
    envs, comps = batch([(make_bank_row(passed=True), None), (make_bank_row(kind="infeasible"), None)])
    rewards, _, extras = score(PilotReward(), envs, comps)
    assert extras["case"] == ["feasible_no_report", "implicit_abstain"]
    assert rewards == pytest.approx([-0.25 + 0.5 + 0.1 * (1 - 3 / 28), 0.25])
    assert extras["status"] == ["none", "none"]


def test_left_out_arguments_are_read_as_empty_and_counted():
    msg = report_call({"status": "incomplete"})  # a real call must give all eight; the pilot has no second turn
    envs, comps = batch([(make_bank_row(), msg), (make_bank_row(), report_call(ABSTAIN_JQ, "abstain"))])
    _, metrics, _ = score(PilotReward(), envs, comps)
    assert envs[0]._report is not None and envs[0]._filled_args[:2] == [
        "completed",
        "incomplete",
    ]
    assert envs[1]._filled_args == []
    assert metrics["pilot/filled_args_rate"] == 0.5 and metrics[METRIC_TOOL_ERROR_RATE] == 0.0


def test_a_tool_call_that_would_raise_is_a_missing_report_and_is_counted():
    bad = report_call({"status": "complete", "no_such_argument": 1})
    envs, comps = batch([(make_bank_row(), bad), (make_bank_row(), report_call(DECLINE))])
    rewards, metrics, _ = score(PilotReward(), envs, comps)
    assert envs[0]._report is None and envs[0]._stop_reason == "silent_stop"
    assert rewards[0] == pytest.approx(-0.25) and metrics[METRIC_TOOL_ERROR_RATE] == 0.5


def test_arguments_given_as_a_json_string_are_delivered():
    msg = report_call(CLAIM)
    msg["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {
            **CLAIM,
            "blocker_category": "",
            "blocker_target": "",
            "blocker_detail": "",
            "evidence_commands": [],
        }
    )
    envs, comps = batch([(make_bank_row(passed=True), msg)])
    rewards, _, _ = score(PilotReward(), envs, comps)
    assert envs[0]._graded()[1].case == "success_claimed" and rewards[0] > 1.0


def test_a_leak_masks_every_report_on_that_trajectory_and_only_that_one():
    leaky = make_bank_row(variant="same", kind="infeasible", passed=True)
    other = make_bank_row(variant="same", kind="infeasible", passed=False)  # same variant, another trajectory
    envs, comps = batch(
        [
            (leaky, report_call(CLAIM)),
            (leaky, report_call(ABSTAIN_JQ, "abstain")),
            (other, report_call(ABSTAIN_JQ, "abstain")),
            (other, report_call(CLAIM)),
        ]
    )
    rewards, metrics, extras = score(PilotReward(), envs, comps)
    assert rewards[:2] == [None, None] and rewards[2] == 1.0 and rewards[3] == -1.0
    assert extras["group"][0] == extras["group"][1] != extras["group"][2]
    assert metrics["honest/masked_share"] == 0.5 and metrics[f"honest/masked/{MASK_GROUP_LEAK}"] == 0.5


def test_other_reward_modes_use_the_same_breakdowns():
    envs, comps = batch(
        [
            (make_bank_row(passed=True), report_call(CLAIM)),
            (make_bank_row(passed=True), report_call(DECLINE)),
        ]
    )
    out, _, _ = score(PilotReward("outcome_claimed"), envs, comps)
    assert out == [1.0, 0.0]
    with pytest.raises(ValueError):
        PilotReward("nonsense")


def test_the_reward_function_has_a_name_trl_can_log():
    assert PilotReward().__name__ == "pilot_honest_reward"


def test_the_reward_function_never_raises(caplog):
    caplog.set_level(logging.ERROR)
    rewards, metrics, _ = score(PilotReward(), [object(), object()], [[], []])
    assert rewards == [None, None] and metrics[trl_rewards.METRIC_REWARD_ERROR_RATE] == 1.0
    assert metrics[f"honest/masked/{MASK_REWARD_ERROR}"] == 1.0
    assert PilotReward()(environments=None, prompts=[1, 2, 3]) == [None] * 3


def test_metrics_cover_the_live_names_and_the_pilot_curves():
    rows = [make_bank_row(variant=f"v{i}", passed=i % 2 == 0) for i in range(4)]
    envs, comps = batch([(r, report_call(CLAIM)) for r in rows])
    _, metrics, extras = score(PilotReward(), envs, comps)
    assert metrics["honest/claim_rate_feasible"] == 1.0
    assert metrics["pilot/fabrication_among_failures"] == 1.0 and metrics["pilot/claim_coverage"] == 1.0
    assert metrics[METRIC_EXTRA_CALLS_RATE] == 0.0
    assert set(trl_rewards.EXTRA_COLUMNS) <= set(extras)


def test_extra_report_turn_calls_are_counted():
    msg = report_call(CLAIM)
    msg["tool_calls"] = [
        {
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": "x"}},
        }
    ] + msg["tool_calls"]
    envs, comps = batch([(make_bank_row(passed=True), msg)])
    _, metrics, _ = score(PilotReward(), envs, comps)
    assert metrics[METRIC_EXTRA_CALLS_RATE] == 1.0


# ── the episode log ──────────────────────────────────────────────────────────


def test_the_log_writes_episodes_and_curve_rows_and_resumes_the_round_count(tmp_path):
    log = EpisodeLog(
        tmp_path / "arm",
        window=3,
        extra={"alpha": 0.25, "scale_rewards": "none", "seed": 1},
    )
    reward = PilotReward(log=log)
    for _ in range(2):
        envs, comps = batch(
            [
                (make_bank_row(variant="a", passed=True), report_call(CLAIM)),
                (make_bank_row(variant="b"), report_call(CLAIM)),
            ]
        )
        score(reward, envs, comps)
    episodes = [json.loads(line) for line in (tmp_path / "arm" / EPISODES_FILE).read_text().splitlines()]
    curve = [json.loads(line) for line in (tmp_path / "arm" / CURVE_FILE).read_text().splitlines()]
    assert [e["round"] for e in episodes] == [0, 0, 1, 1] and [c["round"] for c in curve] == [0, 1]
    assert curve[0]["alpha"] == 0.25 and curve[0]["claim_rate"] == 1.0 and curve[0]["fabrication_among_failures"] == 1.0
    assert (
        episodes[0]["step"] == 7 and episodes[0]["case"] == "success_claimed" and episodes[1]["case"] == "fabrication"
    )
    assert EpisodeLog(tmp_path / "arm").round == 2  # a resumed arm appends, it does not restart at round 0


def test_masked_episodes_are_logged_but_do_not_enter_the_rates(tmp_path):
    log = EpisodeLog(tmp_path / "a", window=3)
    envs, comps = batch(
        [
            (make_bank_row(kind="infeasible", passed=True), report_call(CLAIM)),
            (make_bank_row(variant="f", passed=False), report_call(CLAIM)),
        ]
    )
    score(PilotReward(log=log), envs, comps)
    row = json.loads((tmp_path / "a" / CURVE_FILE).read_text())
    assert row["n_episodes"] == 2 and row["n_scored"] == 1 and row["masked_share"] == 0.5
