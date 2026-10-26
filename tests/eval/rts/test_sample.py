"""`build sample` and the freeze's tier cap on fakes: deterministic, balance-aware selection of candidate pairs."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import replace

import pytest

from honest_agents.eval.rts import build, freeze, sample
from honest_agents.eval.rts.candidates import PairRow, pair_id_of
from honest_agents.eval.rts.protocol import RtsProtocol
from honest_agents.eval.rts.sample import Candidate, select_balanced, select_tier

PROTO = RtsProtocol()  # the pre-registered balance rules: 60 sources, 3 benchmarks, 40 %, 2 operators


def _cands(tier: str, spec: dict[tuple[str, str], int], *, stage: str = "built") -> list[Candidate]:
    """`spec`: (operator, benchmark) -> number of source tasks, source names shared across operators of a benchmark."""
    out = []
    for (op, bench), n in spec.items():
        for i in range(n):
            src = f"{bench}/t{i}"
            out.append(Candidate(pair_id_of(src, op), src, op, bench, tier, stage))
    return out


def _sources_by_bench(picked):
    by: dict[str, set[str]] = {}
    for c in picked:
        by.setdefault(c.benchmark, set()).add(c.source_task)
    return {b: len(s) for b, s in by.items()}


def test_selection_is_deterministic_and_order_independent():
    cands = _cands("near", {("a", "x"): 50, ("b", "y"): 50, ("a", "z"): 50})
    one, d1 = select_balanced(cands, 40, max_share=0.4, seed=0)
    shuffled = list(cands)
    random.Random(7).shuffle(shuffled)
    two, d2 = select_balanced(shuffled, 40, max_share=0.4, seed=0)
    assert one == two and d1 == d2 and len(one) == 40
    other, _ = select_balanced(cands, 40, max_share=0.4, seed=1)
    assert {c.pair_id for c in other} != {c.pair_id for c in one}  # the seed matters
    # adding candidates of another tier or extra candidates never reorders the rank of these
    assert sample.rank("x/t1::a", 0) == sample.rank("x/t1::a", 0) != sample.rank("x/t1::a", 1)


def test_at_most_one_pair_per_source_and_operator():
    c = Candidate("x/t0::a", "x/t0", "a", "x", "mid")
    dup = replace(c, pair_id="x/t0::a#2")
    picked, dropped = select_balanced([c, dup], 5, max_share=1.0)
    assert len(picked) == 1 and "at most one per source and operator" in next(iter(dropped.values()))


def test_operators_are_balanced_first_then_new_sources():
    # operator a has 100 candidates, b 100 (on the same sources), c only 5
    cands = _cands("near", {("a", "x"): 100, ("b", "x"): 100, ("c", "x"): 5})
    picked, _ = select_balanced(cands, 30, max_share=1.0)
    ops = Counter(c.operator for c in picked)
    assert ops == {"a": 13, "b": 12, "c": 5} or ops == {"a": 12, "b": 13, "c": 5}
    assert len({c.source_task for c in picked}) == 30  # a source not yet held is preferred to a second operator on it


def test_benchmark_share_cap_holds_and_minorities_are_kept_whole():
    cands = _cands("far", {("o", "big"): 200, ("p", "big"): 200, ("o", "b2"): 30, ("p", "b3"): 25})
    picked, dropped = select_balanced(cands, 160, max_share=0.4)
    by = _sources_by_bench(picked)
    total = len({c.source_task for c in picked})
    assert all(n <= 0.4 * total for n in by.values()), by
    assert by["b2"] == 30 and by["b3"] == 25  # the small benchmarks are not trimmed
    assert any(w.startswith("benchmark cap: big") for w in dropped.values())
    assert set(dropped) | {c.pair_id for c in picked} == {c.pair_id for c in cands}


def test_supply_below_the_target_selects_everything_that_balances():
    cands = _cands("mid", {("o", "x"): 10, ("o", "y"): 10, ("p", "z"): 10})
    picked, dropped = select_balanced(cands, 160, max_share=0.4)
    assert len(picked) == 30 and dropped == {}


def test_a_tier_short_of_sources_is_selected_without_the_share_cap():
    # two benchmarks can never satisfy a 40 % cap: the capped selection is empty, so the tier is taken as it is
    cands = _cands("mid", {("o", "x"): 30, ("p", "y"): 5})
    capped, _ = select_balanced(cands, 160, max_share=0.4)
    assert capped == []
    picked, _, note = select_tier(cands, 160, PROTO)
    assert len(picked) == 35 and note and "below the 60 minimum" in note
    # with enough sources the cap stays
    many = _cands("near", {("o", "x"): 200, ("p", "y"): 60, ("o", "z"): 60})
    picked, _, note = select_tier(many, 160, PROTO)
    assert note is None and max(_sources_by_bench(picked).values()) <= 0.4 * len({c.source_task for c in picked})


def _rows(cands, *, failed_measurement=()):
    rows = []
    for c in cands:
        r = {"benchmark": c.benchmark, "source_task": c.source_task, "operator": c.operator, "tier": c.tier}
        if c.stage == "measure":
            measured = c.pair_id in failed_measurement
            rows.append({**r, "stage": "measure", "ok": False, "reason": "oracle exited 1", "measured": measured})
        else:
            rows.append({**r, "stage": "built", "ok": True, "reason": "ok"})
    rows.append({**r, "stage": "applicable", "ok": False, "reason": "not applicable", "operator": "zz"})
    return rows


def test_sample_document_per_tier_with_oversample_and_reasons():
    near = _cands("near", {("a", "x"): 120, ("b", "y"): 120, ("a", "z"): 120})
    mid = _cands("mid", {("m", "x"): 40, ("m", "y"): 40}, stage="measure") + _cands("mid", {("s", "z"): 30})
    far = _cands("far", {("f", "x"): 10})
    failed = {mid[0].pair_id}
    doc = sample.sample(_rows(near + mid + far, failed_measurement=failed), target=100, oversample=1.6)
    assert doc["per_tier"] == math.ceil(100 * 1.6) == 160
    t = doc["tiers"]
    assert t["near"]["pairs"] == 160 and t["near"]["projected_problems"] == []
    assert t["mid"]["candidates"] == 109 and t["mid"]["awaiting_measurement"] > 0
    assert doc["dropped"][mid[0].pair_id].startswith("measurement failed")
    assert t["far"]["pairs"] == 10 and any("1 benchmarks, need 3" in p for p in t["far"]["projected_problems"])
    ids = {c["pair_id"] for c in doc["selected"]}
    assert not ids & set(doc["dropped"]) and len(ids) == sum(x["pairs"] for x in t.values())
    with pytest.raises(ValueError):
        sample.sample([], target=100, oversample=0.5)


def test_cli_sample_then_gate_and_measure_take_the_selection(tmp_path, monkeypatch, capsys):
    from honest_agents.eval.rts import gate, measure

    out = tmp_path / "cand"
    out.mkdir()
    cands = _cands("near", {("a", "x"): 5, ("b", "y"): 5}) + _cands("mid", {("m", "x"): 4}, stage="measure")
    (out / "applicability.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(cands)))
    assert build.main(["sample", "--out", str(out), "--target", "3", "--oversample", "1.5"]) == 0
    doc = json.loads((out / "selection.json").read_text())
    assert doc["tiers"]["near"]["pairs"] == 5 and doc["tiers"]["mid"]["pairs"] == 4  # ceil(3 * 1.5) = 5 per tier
    assert "near: 5/10 candidates selected" in capsys.readouterr().out
    chosen = sample.read_selection(out / "selection.json")

    built = [c for c in cands if c.stage == "built"]
    rows = [
        PairRow(
            c.pair_id, c.benchmark, c.source_task, c.operator, c.tier, True, ("x",), ("t",), "s", "t", "c", "ti", "ci"
        )
        for c in built
    ]
    (out / "pairs.jsonl").write_text("".join(json.dumps(r.to_json()) + "\n" for r in rows))
    seen: list[str] = []

    async def fake_gate_many(rows, out_path, **kw):
        seen.extend(r.pair_id for r in rows)
        return {}

    monkeypatch.setattr(gate, "gate_many", fake_gate_many)
    monkeypatch.setattr(gate, "docker_checks", lambda settings, env_type: (None, None))
    args = ["gate", "--pairs", str(out / "pairs.jsonl"), "--selection", str(out / "selection.json")]
    assert build.main(args) == 0
    assert set(seen) == chosen & {c.pair_id for c in built} and len(seen) == 5
    assert "4 selected candidates are not built" in capsys.readouterr().out

    measured: list[str] = []

    async def fake_measure_many(todo, path, **kw):
        measured.extend(pair_id_of(r["source_task"], r["operator"]) for r in todo)
        return {}

    monkeypatch.setattr(measure, "measure_many", fake_measure_many)
    assert build.main(["measure", "--out", str(out), "--selection", str(out / "selection.json")]) == 0
    assert set(measured) == {c.pair_id for c in cands if c.stage == "measure"} & chosen


def _pair_rows(spec: dict[tuple[str, str, str], int]) -> list[PairRow]:
    out = []
    for (tier, op, bench), n in spec.items():
        for i in range(n):
            src = f"{bench}/t{i}"
            pid = pair_id_of(src, op)
            out.append(PairRow(pid, bench, src, op, tier, False, ("x",), ("t",), "s", "t", "c", pid, pid + "_c"))
    return out


def test_freeze_caps_each_tier_at_the_target_and_logs_why():
    rows = _pair_rows({("near", "a", "x"): 80, ("near", "b", "y"): 80, ("near", "a", "z"): 80, ("far", "f", "x"): 7})
    kept, why, notes = freeze.cap_tiers(rows, 100, PROTO, 0)
    per_tier = Counter(r.tier for r in kept)
    assert per_tier == {"near": 100, "far": 7}
    assert all(w.startswith("tier cap (100 pairs per tier)") for w in why.values()) and len(why) == 140
    assert "far" in notes and "near" not in notes  # far is short of sources: taken whole, the freeze then refuses it
    again, _, _ = freeze.cap_tiers(list(reversed(rows)), 100, PROTO, 0)
    assert sorted(r.pair_id for r in again) == sorted(r.pair_id for r in kept)
    assert freeze.cap_tiers(rows, None, PROTO, 0)[0] == rows
