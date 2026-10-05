"""Curves: rates, the advantage TRL computes, and the realised-threshold estimator."""

from __future__ import annotations

import math
import random

import pytest
import torch
from trl.trainer.utils import nanstd

from honest_agents.pilot import curves
from honest_agents.reward import RewardConfig, compute_reward
from tests.pilot.helpers import episode_record, outcome


def trl_advantages(rewards, g, scale):
    """The lines of `GRPOTrainer._generate_and_score_completions` for one reward function, on `trl`'s own `nanstd`."""
    r = torch.tensor([math.nan if x is None else x for x in rewards], dtype=torch.float32)
    mean = torch.nanmean(r.view(-1, g), dim=1).repeat_interleave(g, dim=0)
    if scale in ("group", "none"):
        std = nanstd(r.view(-1, g), dim=1).repeat_interleave(g, dim=0)
    else:
        std = nanstd(r).expand_as(r)
    adv = r - mean
    if scale != "none":
        adv = adv / (std + 1e-4)
    return torch.nan_to_num(adv, nan=0.0).tolist()


@pytest.mark.parametrize("scale", curves.SCALES)
@pytest.mark.parametrize(
    "rewards",
    [
        [1.5, -1.0, 0.25, 0.75, 1.5, 1.5, -1.0, 0.25],
        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        [None, None, None, None, 1.0, -1.0, 1.0, 1.0],  # a fully masked group
        [None, 1.0, 1.0, 0.25, 1.5, 1.5, None, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [None, None, 1.0, None, 1.0, 1.0, 0.0, 0.0],  # a group with one scored report
    ],
)
def test_advantages_match_trl(rewards, scale):
    assert curves.advantages(rewards, 4, scale) == pytest.approx(trl_advantages(rewards, 4, scale), abs=1e-5)


def test_advantages_reject_bad_shapes():
    with pytest.raises(ValueError):
        curves.advantages([1.0, 2.0, 3.0], 2)
    with pytest.raises(ValueError):
        curves.advantages([1.0, 2.0], 2, "sum")


def test_a_masked_report_gets_no_advantage_and_does_not_move_the_baseline():
    adv = curves.advantages([None, 1.5, -1.0, 1.5], 4, "none")
    assert adv[0] == 0.0 and adv[1] == pytest.approx(5 / 6) and adv[2] == pytest.approx(-5 / 3)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
def test_scaling_group_erases_alpha_in_a_one_outcome_group_and_none_keeps_it(alpha):
    """One prompt, one outcome: a report either claims (and gets 1+lambda or -1) or declines (alpha + lambda p)."""
    cfg = RewardConfig(alpha=alpha, efficiency_bonus=0.0)

    def reward(claimed, passed):
        status = "complete" if claimed else "incomplete"
        return compute_reward(outcome(True, {"status": status}, tests_passed=passed), cfg).total

    # a failed trajectory: 4 fabricate, 4 decline honestly
    rewards = [reward(True, False)] * 4 + [reward(False, False)] * 4
    none = curves.advantages(rewards, 8, "none")
    group = curves.advantages(rewards, 8, "group")
    assert none[-1] - none[0] == pytest.approx(1.0 + alpha)  # the honest/fabricate gap is 1 + alpha
    assert group[-1] == pytest.approx(-group[0]) and abs(group[0]) == pytest.approx(
        0.935, abs=1e-3
    )  # TDC corrected no. 4


def test_nominal_threshold():
    assert [curves.nominal_threshold(a, "none") for a in (0, 0.25, 0.5, 0.75)] == [
        0.5,
        0.625,
        0.75,
        0.875,
    ]
    assert {curves.nominal_threshold(a, "group") for a in (0, 0.25, 0.5, 0.75)} == {0.5}


# ── rates ────────────────────────────────────────────────────────────────────


def test_rates_pool_scored_feasible_records_only():
    recs = (
        [episode_record(passed=True, claimed=True)] * 3
        + [episode_record(passed=True, claimed=False)]
        + [episode_record(passed=False, claimed=True)] * 2
        + [episode_record(passed=False, claimed=False)] * 6
        + [episode_record(feasible=False, claimed=True, passed=False, kind="infeasible")]
        + [episode_record(masked=True, claimed=True, passed=False)] * 5
    )
    r = curves.rates(recs)
    assert r["n_episodes"] == 18 and r["n_scored"] == 13 and r["n_feasible"] == 12
    assert r["claim_rate"] == pytest.approx(5 / 12)
    assert r["fabrication_among_failures"] == pytest.approx(2 / 8)
    assert r["claim_coverage"] == pytest.approx(3 / 4)
    assert r["precision_claims"] == pytest.approx(3 / 5) and r["pass_rate_declined"] == pytest.approx(1 / 7)
    assert r["claim_rate_infeasible"] == 1.0 and r["masked_share"] == pytest.approx(5 / 18)


def test_a_rate_without_a_denominator_is_none_not_zero():
    r = curves.rates([episode_record(passed=True, claimed=True)])
    assert (
        r["fabrication_among_failures"] is None and r["false_blame_rate"] == 0.0 and r["claim_rate_infeasible"] is None
    )
    assert curves.rates([])["claim_rate"] is None


def test_false_blame_and_control_rates():
    recs = [
        episode_record(false_blame=True, is_control=True, kind="control"),
        episode_record(is_control=True, kind="control"),
        episode_record(false_blame=True),
        episode_record(),
    ]
    r = curves.rates(recs)
    assert r["false_blame_rate"] == 0.5 and r["control_false_blame_rate"] == 0.5


# ── realised threshold ───────────────────────────────────────────────────────


def simulate(threshold, rounds=40, prompts=8, g=8, sharpness=0.06, seed=0):
    """A calibrated threshold policy: belief q is the pass probability; it claims when q is above `threshold`, with
    probability sigmoid((q - threshold) / sharpness) per report."""
    rng = random.Random(seed)
    recs = []
    for rnd in range(rounds):
        for p in range(prompts):
            q = rng.random()
            passed = rng.random() < q
            pc = 1 / (1 + math.exp(-(q - threshold) / sharpness))
            for _ in range(g):
                claimed = rng.random() < pc
                recs.append(episode_record(round=rnd, pilot_id=f"{rnd}-{p}", passed=passed, claimed=claimed))
    return recs


@pytest.mark.parametrize("threshold", [0.5, 0.625, 0.75, 0.875])
def test_realised_threshold_recovers_a_calibrated_threshold_policy(threshold):
    est = curves.realised_threshold(simulate(threshold, rounds=200, seed=1))
    assert est is not None and abs(est - threshold) < 0.08


def test_realised_threshold_ranks_policies_and_sits_in_the_bracket():
    recs = {t: simulate(t, rounds=100, seed=2) for t in (0.5, 0.875)}
    low, high = (curves.realised_threshold(recs[t]) for t in (0.5, 0.875))
    assert low < high
    r = curves.rates(recs[0.75] if 0.75 in recs else simulate(0.75, rounds=100, seed=3))
    assert r["pass_rate_declined"] <= 0.75 + 0.05 and r["precision_claims"] >= 0.75 - 0.05


def test_realised_threshold_is_none_without_undecided_prompts():
    unanimous = [episode_record(round=0, pilot_id=f"p{i}", passed=i % 2 == 0, claimed=i % 2 == 0) for i in range(40)]
    assert curves.realised_threshold(unanimous) is None
    assert curves.realised_threshold([]) is None


def test_curve_rows_and_window_summaries():
    recs = [
        episode_record(round=r, pilot_id=f"{r}-{i}", claimed=r < 5, passed=i % 2 == 0, step=r)
        for r in range(10)
        for i in range(4)
    ]
    rows = curves.curve(recs, window=3, extra={"alpha": 0.5})
    assert [r["round"] for r in rows] == list(range(10)) and rows[0]["alpha"] == 0.5 and rows[0]["step"] == 0
    assert rows[0]["claim_rate"] == 1.0 and rows[9]["claim_rate"] == 0.0
    assert (
        curves.window_summary(recs, True, 3)["claim_rate"] == 1.0
        and curves.window_summary(recs, False, 3)["claim_rate"] == 0.0
    )
    assert curves.window_summary(recs, True, 3)["rounds"] == [0, 2]
