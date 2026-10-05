"""Difficulty gate from a Phase-0 bank (SPEC_V3 §4.8, INC-5, LV-9): 1-7 of 8 kept, zero-pass share capped, infeasible
leaks reported, alpha safety from the pool's mean pass rate."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from honest_agents.bank import QUARANTINE_LEAK, append_rows, make_row, read_quarantine
from honest_agents.report import SelfReport
from honest_agents.reward import MASK_BOOT_FAILURE, EpisodeOutcome
from honest_agents.tasks import difficulty
from honest_agents.tasks.difficulty import (
    DifficultyGate,
    MissingSplitError,
    auroc,
    claim_recall,
    confidence_pairs,
    d_prime,
    difficulty_report,
    quarantine_entries,
    variant_stats,
)


def bank_row(
    variant_id: str,
    pair_id: str,
    kind: str,
    passed: bool,
    *,
    claim: bool = False,
    confidence: float | None = None,
    family: str | None = None,
    masked: str | None = None,
    split: str | None = "train",
):
    report = SelfReport(status="complete" if claim else "incomplete", confidence=confidence)
    outcome = EpisodeOutcome(
        feasible=kind != "infeasible",
        report=report,
        tests_passed=passed,
        n_actions=3,
        agent_acted=True,
        is_control=kind == "control",
        masked_reason=masked,
    )
    task = {"variant_id": variant_id, "pair_id": pair_id, "kind": kind, "family": family, "spec": {}}
    if split is not None:  # the dataset row's split column, which TRL hands to the environment (L6)
        task["split"] = split
    return make_row(prompt=[], completion=[], outcome=outcome, task=task)


def group(variant_id: str, pair_id: str, kind: str, passes: int, n: int = 8, **kw) -> list:
    return [bank_row(variant_id, pair_id, kind, i < passes, **kw) for i in range(n)]


def plain_pool(passes_by_pair: dict[str, int], n: int = 8) -> list:
    rows = []
    for pair, passes in passes_by_pair.items():
        rows += group(f"{pair}__feasible", pair, "feasible", passes, n)
    return rows


def test_keeps_one_to_seven_of_eight():
    rows = plain_pool({f"p{k}": k for k in range(9)})
    r = difficulty_report(rows, zero_share=0.0)
    assert r.keep_pairs == [f"p{k}" for k in range(1, 8)]
    assert r.drop_pairs == {"p0": "too_hard", "p8": "too_easy"}
    assert r.feasible["histogram"] == {str(k): 1 for k in range(9)}
    assert r.feasible["mean_pass_rate_kept"] == pytest.approx(sum(range(1, 8)) / 7 / 8)


def test_band_and_rollouts_are_configurable():
    rows = plain_pool({"a": 1, "b": 4, "c": 7})
    assert difficulty_report(rows, band=(4, 7), zero_share=0.0).keep_pairs == ["b", "c"]
    four = plain_pool({"a": 0, "b": 2, "c": 4}, n=4)
    r = difficulty_report(four, n_rollouts=4, band=(1, 3), zero_share=0.0)
    assert r.keep_pairs == ["b"] and r.drop_pairs == {"a": "too_hard", "c": "too_easy"}
    with pytest.raises(ValueError):
        difficulty_report(rows, band=(0, 9))


def test_a_capped_share_of_zero_pass_tasks_stays():
    rows = plain_pool({**{f"mid{i}": 3 for i in range(17)}, **{f"zero{i}": 0 for i in range(10)}})
    r = difficulty_report(rows, zero_share=0.15)
    zero_kept = [p for p, s in r.pair_status.items() if s["status"] == "zero_kept"]
    assert len(zero_kept) == 3  # 3 / (17 + 3) = 15% of the final pool
    assert len(r.keep_pairs) == 20 and r.feasible["n_zero_kept"] == 3 and r.feasible["n_too_hard"] == 7
    # deterministic and seed-dependent
    assert difficulty_report(rows, zero_share=0.15).keep_pairs == r.keep_pairs
    assert difficulty_report(rows, zero_share=0.15, seed=1).keep_pairs != r.keep_pairs
    assert difficulty_report(rows, zero_share=0.0).feasible["n_zero_kept"] == 0


def test_unmeasured_tasks_are_reported_not_dropped():
    rows = plain_pool({"a": 3}) + group("b__feasible", "b", "feasible", 1, n=2)
    r = difficulty_report(rows, zero_share=0.0)
    assert r.unmeasured_pairs == ["b"] and r.keep_pairs == ["a"]
    gate = DifficultyGate(
        keep=frozenset(r.keep_pairs), drop=frozenset(r.drop_pairs), unmeasured=frozenset(r.unmeasured_pairs)
    )
    assert gate.admits("b") and gate.admits("a") and gate.admits("never-seen")


def test_infrastructure_failures_are_not_episodes():
    rows = group("a__feasible", "a", "feasible", 4)
    rows += [bank_row("a__feasible", "a", "feasible", False, masked=MASK_BOOT_FAILURE) for _ in range(5)]
    (stat,) = variant_stats(rows).values()
    assert (stat.n, stat.passes, stat.n_infra) == (8, 4, 5)


def test_controls_follow_their_pair_and_are_summarised_separately():
    rows = (
        plain_pool({"a": 3})
        + group("a__x__control", "a", "control", 6, family="x")
        + group("b__x__control", "b", "control", 0, family="x")
    )
    r = difficulty_report(rows, zero_share=0.0)
    assert r.keep_pairs == ["a"]  # a control's own pass rate does not decide the pair
    assert r.controls["n_variants"] == 2 and r.controls["mean_pass_rate"] == pytest.approx(0.375)
    assert r.controls["by_family"]["x"]["n"] == 2


def test_infeasible_leaks_are_reported_and_quarantined(tmp_path):
    rows = plain_pool({"a": 3})
    rows += group("a__remove_input", "a", "infeasible", 0, family="remove_input")
    rows += group("a__remove_tool", "a", "infeasible", 2, family="remove_tool")
    rows += [
        bank_row("a__remove_tool", "a", "infeasible", True, claim=True, family="remove_tool")
    ]  # a claimed clean pass
    r = difficulty_report(rows, zero_share=0.0)
    inf = r.infeasible
    assert inf["n_variants"] == 2 and inf["n_with_clean_pass"] == 1 and inf["leak_rate"] == 0.5
    (leak,) = inf["leaks"]
    assert leak["variant_id"] == "a__remove_tool" and leak["passes"] == 3 and leak["claimed"] == 1 and leak["n"] == 9
    assert (
        inf["by_family"]["remove_tool"]["with_clean_pass"] == 1
        and inf["by_family"]["remove_input"]["with_clean_pass"] == 0
    )
    entries = quarantine_entries(r)
    assert [(e.variant_id, e.kind, e.source) for e in entries] == [("a__remove_tool", QUARANTINE_LEAK, "phase0")]

    bank_dir = tmp_path / "bank"
    append_rows(rows, bank_dir)
    qfile = tmp_path / "q.jsonl"
    difficulty.main(["--bank", str(bank_dir), "--out", str(tmp_path / "d.json"), "--write-quarantine", str(qfile)])
    assert [e.variant_id for e in read_quarantine(qfile)] == ["a__remove_tool"]
    report = json.loads((tmp_path / "d.json").read_text())
    assert report["keep_pairs"] == ["a"] and report["band"] == [1, 7]
    assert DifficultyGate.load(tmp_path / "d.json").keep == {"a"}


# ── alpha safety ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "s, alpha, expected",
    [(0.3, 0.5, 0.07), (0.5, 0.5, 0.27), (0.7, 0.5, 0.60), (0.3, 0.75, 0.01), (0.5, 0.75, 0.07), (0.7, 0.75, 0.27)],
)
def test_claim_recall_reproduces_the_incentives_review(s, alpha, expected):
    """INC-5: calibrated-optimal share of true successes claimed at d' = 1 (AUROC 0.76)."""
    assert claim_recall(s, alpha, 1.0) == pytest.approx(expected, abs=0.015)


def test_claim_recall_limits():
    assert d_prime(0.5) == pytest.approx(0.0, abs=1e-9)
    assert d_prime(0.76) == pytest.approx(1.0, abs=0.02)
    assert claim_recall(0.7, 0.25, 0.0) == 1.0 and claim_recall(0.5, 0.25, 0.0) == 0.0  # no signal: all or nothing
    assert claim_recall(0.5, 1.0, 2.0) == 0.0
    assert claim_recall(0.4, 0.25, 2.0) > claim_recall(0.4, 0.25, 1.0)  # better separation, more recall
    assert claim_recall(0.4, 0.5, 1.0) < claim_recall(0.4, 0.25, 1.0)  # higher alpha, less recall


def test_alpha_table_needs_the_break_even_below_the_mean_pass_rate():
    easy = plain_pool({f"p{i}": 6 for i in range(10)})  # mean pass rate 0.75
    r = difficulty_report(easy, zero_share=0.0, alphas=(0.0, 0.25, 0.5, 0.75))
    table = {row["alpha"]: row for row in r.alpha["table"]}
    assert table[0.0]["clears_break_even"] and table[0.25]["clears_break_even"]
    assert not table[0.75]["clears_break_even"] and not table[0.75]["safe"]  # (1 + 0.75) / 2 = 0.875 > 0.75
    assert r.alpha["auroc_assumed"] and r.alpha["auroc"] == pytest.approx(difficulty.ASSUMED_AUROC)
    assert table[0.0]["safe"] and 0.0 in r.safe_alphas and 0.75 not in r.safe_alphas

    hard = plain_pool({f"p{i}": 2 for i in range(10)})  # mean pass rate 0.25: even alpha = 0 (break-even 0.5) is unsafe
    r = difficulty_report(hard, zero_share=0.0)
    assert r.safe_alphas == [] and r.alpha["mean_pass_rate"] == pytest.approx(0.25)


def test_alpha_table_suggests_a_tighter_band():
    rows = plain_pool({**{f"lo{i}": 1 for i in range(10)}, **{f"hi{i}": 6 for i in range(10)}})
    r = difficulty_report(rows, zero_share=0.0, alphas=(0.25,))
    (row,) = r.alpha["table"]
    assert not row["clears_break_even"]  # mean 0.4375 < 0.625
    assert row["suggested_min_pass"] is None or row["suggested_min_pass"] > 1
    r2 = difficulty_report(rows, zero_share=0.0, alphas=(0.0,))
    assert r2.alpha["table"][0]["suggested_min_pass"] == 2  # drop the 1-of-8 tasks and the pool clears 0.5


def test_auroc_is_measured_from_the_bank_when_there_are_enough_reports():
    rng = random.Random(0)
    rows = []
    for i in range(80):
        passed = i % 2 == 0
        conf = min(1.0, max(0.0, rng.gauss(0.7 if passed else 0.4, 0.15)))
        rows.append(bank_row(f"v{i}", f"p{i}", "feasible", passed, confidence=conf))
    r = difficulty_report(rows, zero_share=0.0)
    assert not r.alpha["auroc_assumed"] and r.alpha["n_confidence_reports"] == 80
    assert 0.85 < r.alpha["auroc"] < 0.99 and r.alpha["d_prime"] > 1.5


def test_auroc_function():
    pairs = [(0.9, True), (0.8, True), (0.3, False), (0.2, False)] * 4
    assert auroc(pairs) == 1.0
    assert auroc([(0.5, True), (0.5, False)] * 10) == 0.5  # ties count one half
    assert auroc([(0.9, True)] * 3 + [(0.1, False)] * 20) is None  # too few of one class
    assert math.isclose(auroc([(0.9, False), (0.1, True)] * 10), 0.0)


def test_cli_prints_a_summary(tmp_path, capsys):
    bank_dir = tmp_path / "bank"
    append_rows(plain_pool({f"p{i}": 4 for i in range(5)}), bank_dir)
    difficulty.main(["--bank", str(bank_dir), "--out", str(tmp_path / "out" / "d.json"), "--zero-share", "0"])
    out = capsys.readouterr().out
    assert "feasible: 5/5 kept" in out and "safe alphas" in out
    assert Path(tmp_path / "out" / "d.json").exists()


def test_an_unclaimed_clean_pass_on_an_infeasible_variant_is_still_a_leak():
    """The label is wrong whenever the tests pass cleanly, claimed or not: a feasible route exists."""
    rows = plain_pool({"a": 3}) + group("a__remove_input", "a", "infeasible", 1, family="remove_input")
    r = difficulty_report(rows, zero_share=0.0)
    (leak,) = r.infeasible["leaks"]
    assert leak["variant_id"] == "a__remove_input" and leak["passes"] == 1 and leak["claimed"] == 0


# ── L6: the gate only reads split == "train" rows ────────────────────────────


def test_rows_without_a_split_column_are_refused():
    rows = group("a__feasible", "a", "feasible", 4, split=None)
    with pytest.raises(MissingSplitError, match="no split column"):
        variant_stats(rows)
    with pytest.raises(MissingSplitError):
        difficulty_report(rows)


def test_validation_and_test_rows_never_shape_the_gate():
    rows = plain_pool({"a": 4}) + group("v__feasible", "v", "feasible", 8, split="val")
    rows += group("t__feasible", "t", "feasible", 0, split="test")
    skipped: Counter = Counter()
    stats = variant_stats(rows, skipped=skipped)
    assert set(stats) == {"a__feasible"} and skipped == {"val": 8, "test": 8}
    report = difficulty_report(rows)
    assert report.keep_pairs == ["a"] and "v" not in report.drop_pairs and "t" not in report.drop_pairs
    assert report.non_train_rows_skipped == {"val": 8, "test": 8}
    assert report.to_dict()["non_train_rows_skipped"] == {"val": 8, "test": 8}


def test_confidence_pairs_ignore_non_train_rows_too():
    train = [bank_row("a__feasible", "a", "feasible", True, claim=True, confidence=0.9)]
    other = [bank_row("b__feasible", "b", "feasible", False, claim=True, confidence=0.9, split="test")]
    assert confidence_pairs(train + other) == [(0.9, True)]
    with pytest.raises(MissingSplitError):
        confidence_pairs([bank_row("c__feasible", "c", "feasible", True, claim=True, confidence=0.5, split=None)])
