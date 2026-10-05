"""The stop rule on simulated sweeps: false-GO rate with no effect, power for a real effect, and seed noise (audit2 SP5).

The sweeps come from `sim.schedule_sweep`: a calibrated threshold policy whose threshold moves from 0 (claim everything,
all fabrication) to `effect` times the nominal (1 + alpha) / 2. Verdicts are computed on arms in memory, no disk. The
rates printed by the slow test are the ones quoted in docs/EVAL.md.
"""

from __future__ import annotations

from collections import Counter

import pytest

from honest_agents.pilot import analyze
from honest_agents.pilot.analyze import GO, NO_GO, UNDECIDED, ArmData, StopRule
from tests.pilot.sim import schedule_sweep

SCENARIOS = {
    "no effect": {"effect": 0.0},
    "no effect, seed noise 0.2": {"effect": 0.0, "seed_sd": 0.2},
    "learns, alpha does nothing": {"effect": 1.0, "alpha_dependent": False},
    "learns, alpha does nothing, seed noise 0.2": {"effect": 1.0, "alpha_dependent": False, "seed_sd": 0.2},
    "real effect": {"effect": 1.0},
    "real effect, seed noise 0.1": {"effect": 1.0, "seed_sd": 0.1},
    "real effect, seed noise 0.2": {"effect": 1.0, "seed_sd": 0.2},
    "half the effect": {"effect": 0.5},
}


def verdict_counts(reps: int, **scenario) -> Counter:
    counts: Counter = Counter()
    for r in range(reps):
        arms = [ArmData(f"a{a}_s{s}", a, "none", s, rec) for a, s, rec in schedule_sweep(rng_seed=r, **scenario)]
        counts[analyze.evaluate(arms, StopRule()).verdict] += 1
    return counts


def test_no_effect_is_never_a_go_and_a_real_effect_is_a_go():
    reps = 25
    nothing = verdict_counts(reps, **SCENARIOS["no effect"])
    assert nothing[GO] == 0 and nothing[NO_GO] >= 0.8 * reps, nothing
    real = verdict_counts(reps, **SCENARIOS["real effect"])
    assert real[GO] >= 0.9 * reps and real[NO_GO] == 0, real


def test_learning_without_an_alpha_trend_is_not_a_go():
    """Fabrication falls in every arm but alpha does nothing (the `group` failure mode): the trend check holds the GO."""
    counts = verdict_counts(25, **SCENARIOS["learns, alpha does nothing"])
    assert counts[GO] <= 1 and counts[UNDECIDED] + counts[NO_GO] >= 24, counts


def test_seed_noise_makes_the_rule_inconclusive_not_wrongly_confident():
    counts = verdict_counts(25, **SCENARIOS["real effect, seed noise 0.2"])
    # a real effect is never called absent; with three seeds this noisy most sweeps are honestly undecided
    assert counts[NO_GO] == 0 and counts[GO] + counts[UNDECIDED] == 25, counts


@pytest.mark.slow
def test_false_go_rate_and_power_of_the_stop_rule():
    """200 sweeps per scenario (run with `-s` to see the table). The false-GO rows must stay at or below 5%, the
    one-sided level of every interval."""
    reps, rows = 200, {}
    for name, scenario in SCENARIOS.items():
        c = verdict_counts(reps, **scenario)
        rows[name] = c
        print(
            f"{name:<46} GO {c[GO] / reps:5.1%}  NO-GO {c[NO_GO] / reps:5.1%}  INCONCLUSIVE {c[UNDECIDED] / reps:5.1%}"
        )
    for name in ("no effect", "no effect, seed noise 0.2", "learns, alpha does nothing"):
        assert rows[name][GO] / reps <= 0.05, (name, rows[name])
    assert rows["learns, alpha does nothing, seed noise 0.2"][GO] / reps <= 0.10, rows
    assert rows["real effect"][GO] / reps >= 0.95 and rows["real effect, seed noise 0.1"][GO] / reps >= 0.8, rows
    assert rows["real effect, seed noise 0.2"][NO_GO] == 0, rows
