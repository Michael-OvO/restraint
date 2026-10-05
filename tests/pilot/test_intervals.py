"""Prompt-cluster counts, the seed widening, and the alpha-trend statistic behind the pilot stop rule."""

from __future__ import annotations

import numpy as np
import pytest

from honest_agents.eval.protocol import seed_inflation
from honest_agents.pilot import intervals
from tests.pilot.helpers import episode_record


def test_the_unit_is_the_prompt_not_the_episode():
    recs = [episode_record(pilot_id="p0", passed=False, claimed=i < 6, declined=i >= 6) for i in range(8)]
    recs += [episode_record(pilot_id="p1", passed=True, claimed=True, declined=False) for _ in range(8)]
    counts = intervals.prompt_counts(recs)
    assert counts.shape == (2, len(intervals.COLUMNS))  # 16 episodes, 2 prompts
    m = intervals.metrics_of(counts.sum(axis=0))
    assert m["fabrication"] == pytest.approx(6 / 8) and m["claim_rate"] == pytest.approx(14 / 16)
    assert intervals.n_failed_prompts(counts) == 1
    masked = [episode_record(pilot_id="p2", masked=True)]
    assert intervals.prompt_counts(masked).shape[0] == 0


def test_an_interval_needs_nearly_every_draw_defined():
    draws = np.linspace(0, 1, 101)
    i = intervals.interval(0.5, draws, 0.9)
    assert i.lo == pytest.approx(0.05) and i.hi == pytest.approx(0.95)
    assert intervals.interval(0.5, np.where(draws < 0.5, np.nan, draws), 0.9).lo is None
    assert intervals.interval(None, draws, 0.9).point is None


def test_seed_inflation_widens_few_seeds_and_vanishes_with_one():
    assert seed_inflation(1) == 1.0
    assert seed_inflation(2) > seed_inflation(3) > seed_inflation(5) > seed_inflation(10) > 1.0
    assert seed_inflation(3) == pytest.approx(2.17, abs=0.02)


def test_spread_over_alpha_is_positive_when_the_claim_rate_falls():
    alphas = [0.0, 0.25, 0.5, 0.75]
    assert intervals.spread_over_alpha(alphas, np.array([0.8, 0.6, 0.4, 0.2])) == pytest.approx(0.6)
    assert intervals.spread_over_alpha(alphas, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)
    assert intervals.spread_over_alpha(alphas, np.array([[0.2, 0.4, 0.6, 0.8], [0.5, 0.5, 0.5, 0.5]])).tolist() == (
        pytest.approx([-0.6, 0.0])
    )
