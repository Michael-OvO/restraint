"""Group washout and a stylised bandit for the damage-aware reward (docs/design.md section 4). Sanity checks, not proof:
earlier tabular bandits converged to careful play under both the old and the new table, so the claims that matter live in
`tests/test_reward_damage.py` (the invariants) and `tests/test_grading_damage.py` (the label).

* Washout: with `scale_rewards = "none"` the 0.75 to 2.3 gaps survive group normalisation; with `"group"` they do not.
* Bandit: three actions per prompt (shortcut, careful, refuse) with logits shared across prompts plus small per-cell offsets
  (the cue sensitivity), rewards from the real `compute_reward`, exact expected policy gradient. The trap-only mixture
  teaches refusal everywhere; the control twin is what keeps the shortcut alive where it is correct."""

from __future__ import annotations

import numpy as np
import pytest

from honest_agents.pilot.curves import advantages
from honest_agents.report import parse_report
from honest_agents.reward import RewardConfig, compute_reward
from tests.pilot.helpers import outcome

CFG = RewardConfig(efficiency_bonus=0.0)
IN = "/app/in.txt"


def test_lone_and_crowded_damagers_keep_their_gap_without_group_scaling():
    pass_, damaged = 1.571, -0.75
    for k, (want_damager, want_clean) in {1: (-2.03, 0.29), 4: (-1.16, 1.16)}.items():
        adv = advantages([damaged] * k + [pass_] * (8 - k), 8, "none")
        assert adv[0] == pytest.approx(want_damager, abs=0.01) and adv[-1] == pytest.approx(want_clean, abs=0.01)


def test_group_scaling_washes_the_signal_out_to_unit_variance_whatever_the_penalty():
    # With std scaling a damage of -0.75 and one of -50 look identical: the magnitude the reward design priced is gone.
    mild = advantages([-0.75] + [1.571] * 7, 8, "group")
    huge = advantages([-50.0] + [1.571] * 7, 8, "group")
    assert mild[0] == pytest.approx(huge[0], abs=0.01)
    none_mild, none_huge = advantages([-0.75] + [1.571] * 7, 8, "none"), advantages([-50.0] + [1.571] * 7, 8, "none")
    assert none_huge[0] < 10 * none_mild[0]


def test_an_all_damage_group_is_flat_and_the_expected_advantage_follows_the_rate():
    assert advantages([-0.75] * 8, 8, "none") == [0.0] * 8
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        k = round(8 * p)
        adv = advantages([-0.75] * k + [1.571] * (8 - k), 8, "none")
        if 0 < k < 8:
            assert sum(adv) == pytest.approx(0.0, abs=1e-9) and adv[0] < 0 < adv[-1]
    # chance that a group of 8 has no contrast at damage rate p: the dynamic-sampling target band is [0.15, 0.85]
    flat = lambda p: p**8 + (1 - p) ** 8
    assert flat(0.15) < 0.30 and flat(0.5) < 0.01 and flat(0.95) > 0.6


# ── the stylised bandit ──────────────────────────────────────────────────────

ACTIONS = ("shortcut", "careful", "refuse")
P_PASS = {"shortcut": 0.9, "careful": 0.85, "refuse": 0.0}
CONFESS = 0.3  # share of damaging rollouts that honestly report incomplete instead of claiming


def _pass_reward(cell: str, n_actions: int) -> float:
    return compute_reward(
        outcome(True, parse_report({"status": "complete"}), tests_passed=True, n_actions=n_actions), CFG
    ).total


def expected_rewards(cell: str, table: str) -> np.ndarray:
    """E[reward] of (shortcut, careful, refuse) on a trap prompt or its control. `old`: no manifest, damage is invisible."""
    done = parse_report({"status": "complete"})
    honest = parse_report({"status": "incomplete"})
    fail = lambda r, **kw: compute_reward(outcome(True, r, tests_passed=False, n_actions=3, **kw), CFG).total
    win = _pass_reward(cell, 3)
    careful = P_PASS["careful"] * _pass_reward(cell, 7) + (1 - P_PASS["careful"]) * fail(honest)
    refuse = compute_reward(
        outcome(
            True, parse_report({"status": "abstain", "blocker_category": "high_risk"}), n_actions=3, tests_passed=False
        ),
        CFG,
    ).total
    if cell == "control" or table == "old":
        direct = P_PASS["shortcut"] * win + (1 - P_PASS["shortcut"]) * fail(honest)
    else:  # trap, damage-aware: the shortcut passes the tests but damages the preserved path, which voids the pass
        dmg = {"damage_paths": (IN,)}
        direct = (1 - CONFESS) * fail(done, **dmg) + CONFESS * fail(honest, **dmg)
    return np.array([direct, careful, refuse])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def train(mix: dict[str, float], table: str, steps: int = 4000, lr: float = 0.05, cue: float = 0.0):
    """Expected policy gradient with a mean baseline. Shared logits `w`, small per-cell offsets `u` (cue sensitivity)."""
    w = np.zeros(3)
    u = {c: np.zeros(3) for c in ("trap", "control")}
    rewards = {c: expected_rewards(c, table) for c in u}
    start = {
        "trap": np.array([0.5, cue, 0.0]),
        "control": np.array([0.5, 0.0, 0.0]),
    }  # the policy begins shortcut-leaning
    for _ in range(steps):
        gw = np.zeros(3)
        for cell, share in mix.items():
            if not share:
                continue
            p = softmax(w + u[cell] + start[cell])
            adv = rewards[cell] - p @ rewards[cell]
            g = p * adv  # d E[R] / d logits for a softmax policy
            gw += share * g
            u[cell] += 0.2 * lr * share * g  # the cell's own offset moves slowly: the cue is hard to read
        w += lr * gw
    return {c: softmax(w + u[c] + start[c]) for c in u}


def test_the_damage_aware_table_prices_the_trap_and_keeps_the_control_working():
    trap, control = expected_rewards("trap", "new"), expected_rewards("control", "new")
    assert trap[1] > 1.2 and trap[0] < -0.5 and trap[2] == 0.0  # careful >> refuse > shortcut on the trap
    assert control[0] > control[1] > control[2] == 0.0  # the shortcut is cheapest and right on the control
    old = expected_rewards("trap", "old")
    assert old[0] > old[1]  # without the manifest the damaging shortcut pays more than the careful path


@pytest.mark.parametrize("cue", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "mix", [{"trap": 1.0, "control": 0.0}, {"trap": 0.5, "control": 0.5}, {"trap": 2 / 3, "control": 1 / 3}]
)
def test_the_new_table_does_not_teach_abstain_everything_in_the_bandit(mix, cue):
    """Trap:control 1:0, 1:1 and 2:1, cue ladder 0 to 2. The damaging shortcut is learned away on the trap, refusal stays
    a minor share everywhere (never above the old table's) and the control keeps acting.

    The design's hope was that a trap-only mixture would drag the control's act rate below 50% and that the twin is what
    prevents it. This bandit does NOT show that: careful play dominates refusal on every prompt (about 1.0 to 1.3 against 0),
    so refusal never takes over at 1:0 either. The necessity of the twin is therefore not established offline; it rests on
    the pricing (`test_the_damage_aware_table_prices_the_trap_and_keeps_the_control_working`, `test_reward_damage.py`) and on
    the staged GPU smoke, which measures p(violate) and the control's act rate per cue level before anything is sized."""
    new, old = train(mix, "new", steps=300, cue=cue), train(mix, "old", steps=300, cue=cue)
    assert new["trap"][0] < 0.1  # violation rate on the trap falls
    for cell in ("trap", "control"):
        assert 1 - new[cell][2] >= 0.9  # act rate: nobody refuses everything
        assert new[cell][2] <= old[cell][2] + 0.1
    final = train(mix, "new", steps=4000, cue=cue)
    assert final["control"][2] < 0.05 and final["trap"][2] < 0.05


def test_damaging_mass_falls_faster_under_the_new_table_than_the_old():
    mix = {"trap": 0.5, "control": 0.5}
    new, old = train(mix, "new", steps=300), train(mix, "old", steps=300)
    assert new["trap"][0] < old["trap"][0]  # the shortcut on the trap shrinks faster
    assert (
        old["trap"][0] > 0.5 * 0.5
    )  # the old table cannot see the damage: only the careful path's extra cost moves it
