"""`analysis.calibration` on a synthetic run: claim loading, the out-of-fold feature model and the power arithmetic."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from honest_agents.analysis import calibration as cal


def _episode(task: str, source: str, sample: int, *, failed: bool, n_cmds: int, conf: float | None = 1.0) -> dict:
    report = {"status": "complete", "completed": ["x"], **({} if conf is None else {"confidence": conf})}
    hist = [{"command": "ls", "rc": 0}] * n_cmds
    return {"task_id": task, "pair_id": source, "sample": sample, "task_dir": f"/x/{task}",
            "outcome": {"feasible": True, "tests_passed": not failed, "report": report},
            "extra": {"history": hist}}  # fmt: skip


def _run(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    with gzip.open(d / "episodes.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return d


def test_claims_and_a_feature_that_separates_failures(tmp_path: Path):
    rows = []
    for i in range(40):
        failed = i % 2 == 0
        for s in range(2):
            rows.append(_episode(f"t{i}", f"src/{i}", s, failed=failed, n_cmds=20 if failed else 5))
    rows.append({**_episode("tx", "src/x", 0, failed=False, n_cmds=3), "outcome": {"feasible": False}})  # not a claim
    claims = cal.load_claims({"r": _run(tmp_path, rows)}, [tmp_path])
    assert len(claims) == 80 and sum(c.failed for c in claims) == 40
    out = cal.scope(claims)
    assert out["features_each"]["n_cmds"] == 1.0 and out["features_oof"] > 0.9
    assert out["by_run"]["r"]["confidence"] == 0.5 and out["by_run"]["r"]["other_sample"] == 1.0
    p = cal.power(claims, [2, 8], n_boot=200)["r"]
    s2, s8 = p["designs"]["S2"], p["designs"]["S8"]
    assert math.isclose(s8["se_auroc"], s2["se_auroc"] / 2) and math.isclose(
        s2["se_difference"], s2["se_auroc"] * 2**0.5
    )


def test_paired_compares_two_arms_on_the_same_claims():
    rows_a, rows_b = [], []
    for i in range(60):
        passed = i % 3 != 0
        base = {"id": f"e{i}", "source": f"src/{i // 2}", "feasible": True, "status": "complete", "y": int(passed)}
        rows_a.append({**base, "mean": 1.0})  # constant: no discrimination
        rows_b.append({**base, "mean": 0.9 if passed else 0.3})  # perfect separation
    rows_a.append({"id": "x", "source": "s", "feasible": True, "status": "incomplete", "y": 0, "mean": 0.1})
    claims = cal.paired_claims(rows_a, rows_b)
    assert len(claims) == 60
    out = cal.paired(claims, n_boot=300)
    assert out["a"]["auroc"] == 0.5 and out["b"]["auroc"] == 1.0 and math.isclose(out["b_minus_a"]["auroc"], 0.5)
    lo, hi = out["ci90"]["auroc"]
    assert lo > 0.3 and hi <= 0.5 + 1e-9
    assert out["b"]["confident_precision"] == 1.0 and out["a"]["confident_share"] == 1.0
    bad = [{**rows_b[0], "y": 1 - rows_b[0]["y"]}]
    import pytest

    with pytest.raises(ValueError, match="disagree"):
        cal.paired_claims(rows_a[:1], bad)


def test_parts_pool_under_one_name_and_the_heldout_filter_hits_one_part(tmp_path: Path):
    rows = [_episode(f"t{i}", f"src/{i}", 0, failed=i % 2 == 0, n_cmds=3) for i in range(20)]
    run = _run(tmp_path, rows)
    keep = lambda name, src: name != "dev:pool" or src in {"src/0", "src/1"}
    claims = cal.load_claims([("dev:select", run), ("dev:pool", run)], [tmp_path], keep)
    assert {c.run for c in claims} == {"dev"} and len(claims) == 20 + 2


ARCHIVE = Path("~/Documents/honest-agents-archive/iterations").expanduser()
G2_SELECT = ARCHIVE / "16-it16c-g2-select" / "eval" / "it16_g2_select"


def test_the_guard_margins_reproduce_it16s_select_margin():
    """The verified-success margin at the select split's K and S is it16's `ni_margin` (configs/protocols/it16_select.json,
    0.077, computed independently from it14_base_select), so the DEF-66 sources are included as they were then."""
    import pytest

    if not G2_SELECT.is_dir():
        pytest.skip("archive not present")
    out = cal.guard_margins(G2_SELECT, ("verified_success", "mean_commands"), n_sim=500, spec_defects="include")
    assert out["verified_success"]["sources"] == 60 and out["verified_success"]["samples"] == 2
    assert abs(out["verified_success"]["margin"] - 0.077) < 0.005
    assert not out["mean_commands"]["higher_is_better"] and out["mean_commands"]["margin"] > 0.5


STAGING = Path("~/Documents/honest-agents-archive/staging/runs/eval").expanduser()


def test_the_read_guard_does_not_pass_it17s_adapter():
    """Backtest: it17's confidence-only adapter eroded read-before-destruction (blind deletions 4 -> 30 of 120 twins);
    the guard iteration 18 adds must not pass it on either branch."""
    import pytest

    ref, cand = STAGING / "it17_g2_select", STAGING / "it17_cal_select"
    tasks = Path(__file__).parents[1] / "data" / "tasks"
    if not (ref.is_dir() and cand.is_dir() and tasks.is_dir()):
        pytest.skip("it17 select runs or task dirs not present")
    out = cal.read_guard(ref, cand, tasks, n_sim=300)
    for name in cal.READ_METRICS:
        assert not out[name]["pass"] and out[name]["diff"] < -0.1, (name, out[name])
