"""RTS scores: constant policies score 0, the two-level macro, RTS-C weighting, the verdict and the ladder."""

from __future__ import annotations

import dataclasses

import pytest

from honest_agents.eval.metrics import score_records
from honest_agents.eval.rts.protocol import RtsProtocol
from honest_agents.eval.rts.score import VERDICT_CAPABILITY_LOSS, endpoint, render_endpoint, score_model, tier_ladder
from honest_agents.eval.rts.units import build_units
from tests.eval.rts.helpers import episodes, pairs

FAST = RtsProtocol(n_boot=300)


def _units(items, **kw):
    return build_units(score_records(episodes(items, **kw)), items)


@pytest.mark.parametrize(
    ("hits", "false_refusals"),
    [(20, 20), (0, 0), (10, 10)],  # refuse everything, refuse nothing, refuse the same half of both
)
def test_blind_policies_score_zero(hits, false_refusals):
    items = pairs("tb2", "mid", "remove_tool", 20)
    rts = score_model(_units(items, hits=hits, false_refusals=false_refusals), FAST)["headline"]["rts"]
    assert rts["value"] == pytest.approx(0.0)


def test_perfect_policy_scores_one():
    items = pairs("tb2", "mid", "remove_tool", 20)
    assert score_model(_units(items, hits=20, false_refusals=0), FAST)["headline"]["rts"]["value"] == pytest.approx(1.0)


def test_headline_is_mean_of_tiers_of_mean_of_operators():
    near_a, near_b = pairs("tb2", "near", "op_a", 10), pairs("tb2", "near", "op_b", 30)
    mid, far = pairs("swe", "mid", "op_c", 10), pairs("quix", "far", "op_d", 10)
    units = [
        *_units(near_a, hits=10, false_refusals=0),  # J 1.0
        *_units(near_b, hits=0, false_refusals=0),  # J 0.0 -> near = 0.5 (pooled would be 0.25)
        *_units(mid, hits=8, false_refusals=2),  # J 0.6
        *_units(far, hits=0, false_refusals=0),  # J 0.0
    ]
    s = score_model(units, FAST)
    assert s["tiers"]["near"]["rts"]["value"] == pytest.approx(0.5)
    assert s["tiers"]["mid"]["rts"]["value"] == pytest.approx(0.6)
    assert s["headline"]["rts"]["value"] == pytest.approx((0.5 + 0.6 + 0.0) / 3)
    assert s["operators"]["op_a"]["value"] == pytest.approx(1.0)
    lo, hi = s["headline"]["rts"]["lo"], s["headline"]["rts"]["hi"]
    assert lo <= s["headline"]["rts"]["value"] <= hi


def test_capability_weights_benchmarks_equally():
    x, y = pairs("quix", "mid", "op_a", 10), pairs("swe", "mid", "op_b", 30)
    units = [*_units(x, hits=0, false_refusals=0), *_units(y, hits=0, false_refusals=0, control_pass=False)]
    s = score_model(units, FAST)
    assert s["headline"]["rts_c"]["value"] == pytest.approx(0.5)  # pooled would be 0.25
    assert s["benchmarks"]["quix"]["rts_c"]["value"] == pytest.approx(1.0)
    assert s["benchmarks"]["swe"]["rts_c"]["value"] == pytest.approx(0.0)


def test_endpoint_transfer():
    items = pairs("tb2", "mid", "remove_tool", 20)
    rep = endpoint(_units(items, hits=2, false_refusals=2), _units(items, hits=18, false_refusals=2), FAST)
    assert rep["verdict"] == "transfer"
    j = next(r for r in rep["endpoints"] if r["metric"] == "j")
    assert j["delta"] == pytest.approx(0.8)
    assert "verdict: **transfer**" in render_endpoint(rep)


def test_endpoint_capability_loss_blocks_transfer():
    items = pairs("tb2", "mid", "remove_tool", 20)
    base = _units(items, hits=2, false_refusals=2)
    trained = _units(items, hits=18, false_refusals=2, control_pass=False)
    assert endpoint(base, trained, FAST)["verdict"] == VERDICT_CAPABILITY_LOSS


def test_endpoint_over_refusal():
    items = pairs("tb2", "mid", "remove_tool", 20)
    rep = endpoint(_units(items, hits=2, false_refusals=2), _units(items, hits=20, false_refusals=10), FAST)
    assert rep["verdict"] == "gain_with_possible_over_refusal"


def test_ladder_orders_tiers():
    near, mid, far = pairs("tb2", "near", "op_a", 10), pairs("swe", "mid", "op_b", 10), pairs("quix", "far", "op_c", 10)
    base = [
        *_units(near, hits=0, false_refusals=0),
        *_units(mid, hits=0, false_refusals=0),
        *_units(far, hits=0, false_refusals=0),
    ]
    trained = [
        *_units(near, hits=10, false_refusals=0),
        *_units(mid, hits=5, false_refusals=0),
        *_units(far, hits=0, false_refusals=0),
    ]
    lad = tier_ladder(base, trained, FAST)
    assert lad["tiers"]["near"]["delta"] == pytest.approx(1.0)
    assert lad["tiers"]["mid"]["delta"] == pytest.approx(0.5)
    assert lad["tiers"]["far"]["delta"] == pytest.approx(0.0)
    assert lad["near_minus_far"]["lo"] > 0
    assert lad["ordered_share"] == pytest.approx(1.0)


def test_ladder_with_a_missing_tier():
    near = pairs("tb2", "near", "op_a", 10)
    lad = tier_ladder(_units(near, hits=0, false_refusals=0), _units(near, hits=10, false_refusals=0), FAST)
    assert lad["tiers"]["far"]["delta"] is None
    assert lad["ordered_share"] is None
    assert lad["near_minus_far"] is None


def test_ladder_pairs_within_tier():
    near = pairs("tb2", "near", "op_a", 40)
    mid = [
        dataclasses.replace(i, item_id=i.item_id.replace("/near/", "/mid/") + "m", pair_id=i.pair_id + "m", tier="mid")
        for i in near
    ]
    items = [*near, *mid]
    base = build_units(
        score_records([*episodes(near, hits=40, false_refusals=0), *episodes(mid, hits=20, false_refusals=0)]), items
    )
    lost = {i.source_task for i in near[: 2 * 20]}
    recs = [
        dataclasses.replace(r, outcome=dataclasses.replace(r.outcome, masked_reason="boot_failure"))
        if r.source_task in lost and "/mid/" in r.variant_key
        else r
        for r in [*episodes(near, hits=40, false_refusals=0), *episodes(mid, hits=20, false_refusals=0)]
    ]
    trained = build_units(score_records(recs), items)
    lad = tier_ladder(base, trained, FAST)
    assert lad["tiers"]["mid"]["delta"] == pytest.approx(0.0)
    assert lad["tiers"]["mid"]["lo"] <= 0 <= lad["tiers"]["mid"]["hi"]
    rep = endpoint(base, trained, FAST)
    assert next(r for r in rep["endpoints"] if r["metric"] == "j")["delta"] == pytest.approx(0.0)
