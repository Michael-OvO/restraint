"""The training-time probe list: honesty-weighted, family-balanced, pair-whole, validation only, fixed."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from honest_agents.tasks import probe_set


def pool(n_pairs: dict[str, int], split: str = "val") -> list[dict]:
    """Per family, `n` source tasks with a plain twin, one impossible variant and its near-miss control."""
    rows = []
    for fam, n in n_pairs.items():
        for i in range(n):
            pid = f"{fam}/s{i}"
            common = {"pair_id": pid, "split": split, "prompt": None}
            rows += [
                {**common, "variant_id": f"{pid}/plain", "kind": "feasible", "family": "feasible", "feasible": True,
                 "is_control": False},
                {**common, "variant_id": f"{pid}/{fam}", "kind": "infeasible", "family": fam, "feasible": False,
                 "is_control": False},
                {**common, "variant_id": f"{pid}/{fam}/control", "kind": "control", "family": fam, "feasible": True,
                 "is_control": True},
            ]  # fmt: skip
    return rows


ROWS = pool({"high_risk": 60, "revoke_permission": 30, "remove_input": 12})


def test_impossible_variants_are_drawn_round_robin_with_their_control_and_plain_twin():
    chosen = probe_set.select_probe_set(ROWS, n_infeasible=64, n_feasible=64)
    kinds = Counter(r["kind"] for r in chosen)
    assert kinds == {"infeasible": 64, "control": 64, "feasible": 64}
    fams = Counter(r["family"] for r in chosen if r["kind"] == "infeasible")
    # the small family is taken whole before high_risk fills the rest
    assert fams == {"remove_input": 12, "revoke_permission": 26, "high_risk": 26}
    pairs = {r["pair_id"] for r in chosen if r["kind"] == "infeasible"}
    assert {r["pair_id"] for r in chosen} == pairs  # every row belongs to a chosen pair: both sides of each source
    assert [r["variant_id"] for r in chosen] == [r["variant_id"] for r in ROWS if r in chosen]  # file order


def test_the_list_is_fixed_by_the_seed_not_the_file_order():
    a = probe_set.select_probe_set(ROWS, n_infeasible=30, n_feasible=30)
    b = probe_set.select_probe_set(list(reversed(ROWS)), n_infeasible=30, n_feasible=30)
    assert sorted(r["variant_id"] for r in a) == sorted(r["variant_id"] for r in b)
    c = probe_set.select_probe_set(ROWS, n_infeasible=30, n_feasible=30, seed=1)
    assert {r["variant_id"] for r in a} != {r["variant_id"] for r in c}


def test_feasible_rows_are_topped_up_from_other_sources():
    chosen = probe_set.select_probe_set(ROWS, n_infeasible=10, n_feasible=40, with_plain_twins=False)
    kinds = Counter(r["kind"] for r in chosen)
    assert kinds["infeasible"] == 10 and kinds["control"] == 10 and kinds["control"] + kinds["feasible"] == 40


def test_too_few_impossible_variants_or_a_test_row_are_refused():
    with pytest.raises(ValueError, match="only 102 impossible variants"):
        probe_set.select_probe_set(ROWS, n_infeasible=200)
    with pytest.raises(ValueError, match="validation rows only"):
        probe_set.select_probe_set(pool({"high_risk": 70}, split="test"))


def test_quarantined_variants_and_unwanted_families_are_left_out():
    leaky = {"remove_input/s0/remove_input"}
    chosen = probe_set.select_probe_set(ROWS, n_infeasible=60, n_feasible=60, exclude=leaky)
    ids = {r["variant_id"] for r in chosen}
    assert not ids & leaky and "remove_input/s0/remove_input/control" not in ids  # its control goes with it
    only = probe_set.select_probe_set(ROWS, n_infeasible=20, n_feasible=20, families=["revoke_permission"])
    assert {r["family"] for r in only if r["kind"] == "infeasible"} == {"revoke_permission"}


def test_cli_writes_the_rows_and_prints_the_cost(tmp_path: Path, capsys):
    src = tmp_path / "val_pool.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    out = tmp_path / "probe.jsonl"
    assert probe_set.main(["--rows", str(src), "--out", str(out), "--samples-per-variant", "4"]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 192 and all(r in ROWS for r in rows)  # written unchanged
    summary = json.loads(capsys.readouterr().out)
    assert summary["by_kind"]["infeasible"] == 64 and summary["episodes_per_probe"] == 768
