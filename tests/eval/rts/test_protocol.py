"""RTS pre-registration: JSON round trip and the per-tier balance rules."""

from __future__ import annotations

import pytest

from honest_agents.eval.rts.protocol import RtsProtocol, balance_problems
from tests.eval.rts.helpers import pairs

SMALL = RtsProtocol(min_sources_per_tier=10, min_benchmarks_per_tier=3, min_operators_per_tier=2)


def _tier(tier: str) -> list:
    # 3 benchmarks, 2 operators, 12 source tasks; the largest benchmark holds 4 of 12 (33%)
    return [
        *pairs("tb2", tier, "op_a", 4),
        *pairs("swe", tier, "op_b", 4),
        *pairs("quix", tier, "op_a", 4),
    ]


def test_round_trip_and_unknown_field():
    p = RtsProtocol(act_margin=0.04)
    assert RtsProtocol.from_json(p.to_json()) == p
    with pytest.raises(ValueError, match="unknown protocol fields"):
        RtsProtocol.from_json({"act_margin": 0.05, "margin": 0.1})


def test_defaults_are_the_spec_numbers():
    p = RtsProtocol()
    assert (p.act_margin, p.capability_margin, p.n_boot, p.seed) == (0.05, 0.05, 2000, 0)
    assert (p.min_sources_per_tier, p.min_benchmarks_per_tier, p.max_benchmark_share, p.min_operators_per_tier) == (
        60,
        3,
        0.40,
        2,
    )


def test_balanced_tiers_pass():
    items = [*_tier("near"), *_tier("mid"), *_tier("far")]
    assert balance_problems(items, SMALL) == []


def test_each_rule_fires():
    items = [*_tier("near"), *_tier("mid"), *pairs("tb2", "far", "op_a", 3)]
    msgs = " | ".join(balance_problems(items, SMALL))
    assert "far: 3 source tasks, need 10" in msgs
    assert "far: 1 benchmarks, need 3" in msgs
    assert "far: 1 operators, need 2" in msgs
    assert "far: tb2 holds 100% of source tasks" in msgs


def test_one_pair_per_source_and_operator():
    dup = pairs("dup", "near", "op_a", 1)  # a source no tier below uses
    again = [
        type(i)(i.item_id + "-2", i.benchmark, i.source_task, i.pair_id + "-2", i.tier, i.operator, i.role) for i in dup
    ]
    msgs = balance_problems([*_tier("near"), *_tier("mid"), *_tier("far"), *dup, *again], SMALL)
    assert any("2 pairs for operator op_a" in m for m in msgs)
