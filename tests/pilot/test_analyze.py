"""The stop rule and the go / no-go line, on simulated arms and on hand-made curves."""

from __future__ import annotations

import json

import pytest

from honest_agents.pilot import analyze
from honest_agents.pilot.analyze import GO, NO_GO, UNDECIDED, StopRule
from tests.pilot.helpers import episode_record
from tests.pilot.sim import simulate_arm, write_arm

ALPHAS = (0.0, 0.25, 0.5, 0.75)
RULE = StopRule(window_rounds=10)


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    """The full sweep, simulated once: alpha x scale_rewards, one seed."""
    root = tmp_path_factory.mktemp("pilot")
    thresholds = {}
    for scale in ("none", "group"):
        for alpha in ALPHAS:
            t, records = simulate_arm(alpha, scale, rounds=120, seed=5)
            thresholds[(alpha, scale)] = t
            write_arm(root, alpha, scale, records)
    return root, thresholds


def test_the_simulated_sweep_learns_the_threshold_the_algebra_predicts(learned):
    _, t = learned
    for alpha in ALPHAS:
        assert t[(alpha, "none")] == pytest.approx((1 + alpha) / 2, abs=0.1)
        assert t[(alpha, "group")] == pytest.approx(0.5, abs=0.1)
    assert t[(0.0, "none")] < t[(0.25, "none")] < t[(0.5, "none")] < t[(0.75, "none")]


def test_a_sweep_that_behaves_is_a_go(learned):
    root, _ = learned
    v = analyze.analyse(root, RULE, write=False)
    assert v.verdict == GO and v.exit_code == 0 and v.line().startswith("PILOT GO")
    assert [c.status for c in v.checks[:3]] == ["pass", "pass", "pass"]
    group_check = next(c for c in v.checks if c.name.startswith("group scaling"))
    assert "as predicted" in group_check.detail
    text = analyze.format_report(v)
    assert text.splitlines()[-1] == v.line() and "none" in text and "group" in text


def test_analyse_writes_the_verdict_and_the_curves(learned, tmp_path):
    root, _ = learned
    import shutil

    copy = tmp_path / "run"
    shutil.copytree(root, copy)
    analyze.analyse(copy, RULE)
    data = json.loads((copy / analyze.ANALYSIS_FILE).read_text())
    assert data["verdict"] == "GO" and len(data["cells"]) == 8
    header, *lines = (copy / analyze.CURVES_CSV).read_text().splitlines()
    assert header.startswith("arm,alpha,scale_rewards,seed,round") and len(lines) == 8 * 120


def write_seeds(root, scale, seeds=(1, 2, 3), **sim):
    """Every alpha, several training seeds, each arm from `simulate_arm(**sim)`."""
    for seed in seeds:
        for alpha in ALPHAS:
            kw = {k: v(alpha) if callable(v) else v for k, v in sim.items()}
            write_arm(root, alpha, scale, simulate_arm(alpha, "none", rounds=120, seed=seed, **kw)[1], seed=seed)


def test_alpha_that_does_not_move_the_claim_rate_is_a_no_go(tmp_path):
    for seed in (1, 2, 3, 4, 5):  # the scale_rewards=none arms behave like group arms: alpha is erased
        for alpha in ALPHAS:
            write_arm(tmp_path, alpha, "none", simulate_arm(alpha, "group", rounds=120, seed=seed)[1], seed=seed)
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == NO_GO and v.exit_code == 1
    assert "does not fall with alpha" in v.line()


def test_one_seed_that_cannot_tell_is_inconclusive_not_a_no_go(tmp_path):
    """The same data from one seed: the trend interval includes 0 and also the smallest effect, so no verdict. (Three
    seeds are not enough to rule an effect out either: the t-widened seed component keeps the interval wide.)"""
    for alpha in ALPHAS:
        write_arm(tmp_path, alpha, "none", simulate_arm(alpha, "group", rounds=120, seed=5)[1])
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == UNDECIDED and "trend interval includes 0" in v.line()


def test_a_claim_rate_that_rises_with_alpha_is_a_no_go(tmp_path):
    write_seeds(tmp_path, "none", learn=False, start=lambda alpha: 0.2 + 0.6 * (1 - alpha))  # the wrong sign
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == NO_GO and "does not fall with alpha" in v.line()


def test_fabrication_that_does_not_fall_is_a_no_go(tmp_path):
    # fixed policies: claim rate ordered by alpha but nothing is learned over time
    write_seeds(tmp_path, "none", learn=False, start=lambda alpha: 0.2 + 0.6 * alpha)
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == NO_GO and "fabrication among failures falls" in v.line()


def test_the_verdict_is_deterministic_and_reports_the_intervals(learned):
    root, _ = learned
    a, b = (analyze.analyse(root, RULE, write=False) for _ in range(2))
    assert a.to_dict() == b.to_dict()
    cell = next(c for c in a.cells if c["scale_rewards"] == "none" and c["alpha"] == 0.0)
    point, lo, hi = cell["intervals"]["fabrication_drop"]
    assert lo < point < hi and lo > 0.0  # a real drop: the interval excludes 0
    assert "drop" in next(c for c in a.checks if c.name.startswith("fabrication among")).detail


def hand_arm(root, alpha, first, last, coverage=(0.9, 0.9), report_rate_last=1.0, rounds=30, n=40):
    """An arm whose first and last windows have the given (claim rate, fabrication) and coverage."""
    records = []
    for rnd in range(rounds):
        claim_fail, claim_pass = (
            (first if rnd < rounds // 2 else last),
            (coverage[0] if rnd < rounds // 2 else coverage[1]),
        )
        for i in range(n):
            passed = i % 2 == 0
            p = claim_pass if passed else claim_fail
            claimed = (i // 2) % 10 < round(p * 10)
            records.append(
                episode_record(
                    round=rnd,
                    pilot_id=f"{rnd}-{i}",
                    passed=passed,
                    claimed=claimed,
                    has_report=(i % 20) / 20 < report_rate_last if rnd >= rounds - 10 else True,
                )
            )
    return write_arm(root, alpha, "none", records)


def test_fabrication_falling_by_quitting_is_a_no_go(tmp_path):
    for k, alpha in enumerate(ALPHAS):
        hand_arm(tmp_path, alpha, first=0.9, last=0.0 + 0.0 * k, coverage=(0.9, 0.0))
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == NO_GO
    check = next(c for c in v.checks if "quitting" in c.name)
    assert check.status == "fail" and "collapse" in check.detail


def test_claiming_less_at_random_is_not_honesty(tmp_path):
    for (
        alpha
    ) in ALPHAS:  # fabrication falls from 0.9 to 0.4, but so does coverage by the same amount: no discrimination
        hand_arm(
            tmp_path,
            alpha,
            first=0.9,
            last=0.7 - 0.2 * alpha * 4 / 3,
            coverage=(0.9, 0.7 - 0.2 * alpha * 4 / 3),
        )
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == NO_GO and "no longer tell success apart" in v.line()


def test_report_rate_collapse_is_a_no_go(tmp_path):
    for alpha in ALPHAS:
        hand_arm(
            tmp_path,
            alpha,
            first=0.9,
            last=0.3 + 0.1 * (0.75 - alpha) * 4,
            coverage=(0.9, 0.8),
            report_rate_last=0.5,
        )
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == NO_GO and "report rate" in v.line()


def test_too_little_data_is_inconclusive_not_a_pass(tmp_path):
    for alpha in ALPHAS:
        _, records = simulate_arm(alpha, "none", rounds=12, seed=1)  # less than two windows of 10 rounds
        write_arm(tmp_path, alpha, "none", records)
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict == UNDECIDED and v.exit_code == 2 and "INCONCLUSIVE" in v.line()


def test_nothing_to_reduce_is_inconclusive(tmp_path):
    for alpha in ALPHAS:  # the base policy never fabricates
        hand_arm(tmp_path, alpha, first=0.0, last=0.0, coverage=(0.9, 0.9))
    v = analyze.analyse(tmp_path, RULE, write=False)
    assert v.verdict in (UNDECIDED, NO_GO)
    assert next(c for c in v.checks if c.name.startswith("fabrication among")).status == "inconclusive"


def test_only_group_arms_is_inconclusive(tmp_path):
    for alpha in ALPHAS[:2]:
        write_arm(tmp_path, alpha, "group", simulate_arm(alpha, "group", rounds=30)[1])
    assert analyze.analyse(tmp_path, RULE, write=False).verdict == UNDECIDED


def test_seeds_are_pooled_by_the_mean(tmp_path):
    for seed in (1, 2):
        for alpha in ALPHAS:
            write_arm(
                tmp_path,
                alpha,
                "none",
                simulate_arm(alpha, "none", rounds=120, seed=seed)[1],
                seed=seed,
            )
    arms = analyze.load_arms(tmp_path)
    assert len(arms) == 8
    cells = analyze.pool_cells([analyze.summarise_arm(a, RULE) for a in arms])
    assert len(cells) == 4 and all(c["seeds"] == [1, 2] for c in cells)
    assert analyze.evaluate(arms, RULE).verdict == GO


def test_arm_names_and_missing_metadata(tmp_path):
    d = write_arm(tmp_path, 0.25, "none", [episode_record()])
    (d / "arm.json").unlink()
    assert analyze.load_arm(d).alpha == 0.25  # recovered from `a0.25_none_s42`
    bad = tmp_path / "weird"
    bad.mkdir()
    (bad / "episodes.jsonl").write_text("")
    with pytest.raises(ValueError, match="no arm.json"):
        analyze.load_arm(bad)
    assert analyze.load_arm(tmp_path / "empty") is None
    with pytest.raises(FileNotFoundError):
        analyze.load_arms(tmp_path / "nope")
    with pytest.raises(ValueError, match="no arm has written"):
        analyze.analyse(tmp_path / "..", RULE, write=False) if False else analyze.analyse(_empty(tmp_path), RULE)


def _empty(tmp_path):
    d = tmp_path / "emptyroot"
    d.mkdir()
    return d


def test_stop_rule_validation():
    assert StopRule.from_dict({"window_rounds": 5}).window_rounds == 5
    with pytest.raises(ValueError, match="unknown analysis key"):
        StopRule.from_dict({"nope": 1})
    with pytest.raises(ValueError):
        StopRule(min_claim_rate=1.5)
    with pytest.raises(ValueError):
        StopRule(window_rounds=0)


def test_cli_prints_the_line_and_exits_with_the_verdict(learned, tmp_path, capsys):
    root, _ = learned
    code = analyze.main([str(root), "--window-rounds", "10", "--no-write"])
    out = capsys.readouterr().out
    assert code == 0 and out.strip().splitlines()[-1].startswith("PILOT GO")
    assert analyze.main([str(root), "--no-write", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "GO"
