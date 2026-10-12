"""Iteration-12 distillation pool: disjoint from the RL pool, leaks out, plain rows from the pass@4 band, per-count files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.bank import BankRow
from honest_agents.reward import EpisodeOutcome, outcome_to_dict
from honest_agents.tasks import distill_pool as dp
from honest_agents.tasks.rl_pool import EXCL_LEAK_BANK, EXCL_LEAK_VALIDATION, Exclusions

FAMILIES = ("high_risk", "remove_input", "revoke_permission")


def row(seed: str, kind: str, family: str | None = None) -> dict:
    suffix = {"feasible": "feasible", "control": f"{family}__control", "infeasible": family}[kind]
    return {
        "variant_id": f"src__{seed}__{suffix}",
        "pair_id": f"src/{seed}",
        "kind": kind,
        "family": family or "feasible",
        "is_control": kind == "control",
        "feasible": kind != "infeasible",
        "split": "train",
        "task_dir": f"data/tasks/src__{seed}__{suffix}",
    }


def pool(n: int = 12) -> list[dict]:
    rows = []
    for i in range(n):
        seed = f"s{i:02d}"
        rows.append(row(seed, "feasible"))
        fam = FAMILIES[i % 3]
        rows += [row(seed, "control", fam), row(seed, "infeasible", fam)]
    return rows


def plan(rows, rl=frozenset(), ex=None, passes=None, **opts):
    passes = passes if passes is not None else {r["variant_id"]: (4, 2) for r in rows}
    return dp.plan(rows, set(rl), ex or Exclusions(), passes, dp.DistillOptions(**opts))


def test_rl_pool_sources_are_left_out_whole():
    rows = pool(6)
    built = plan(rows, rl={"src/s00", "src/s01"})
    assert {r["pair_id"] for r in built.rows}.isdisjoint({"src/s00", "src/s01"})
    assert len(built.excluded[dp.EXCL_RL_POOL]) == 6  # three rows per source


def test_leaky_twins_are_excluded_but_their_control_stays():
    rows = pool(3)
    ex = Exclusions(infeasible_variants={"src__s00__high_risk": EXCL_LEAK_VALIDATION})
    built = plan(rows, ex=ex)
    ids = {r["variant_id"] for r in built.rows}
    assert "src__s00__high_risk" not in ids and "src__s00__high_risk__control" in ids
    assert built.excluded[EXCL_LEAK_VALIDATION] == ["src__s00__high_risk"]


def test_samples_per_cell():
    built = plan(pool(6))
    n = {r["variant_id"]: r["n_samples"] for r in built.rows}
    assert n["src__s00__high_risk"] == 4 and n["src__s01__remove_input"] == 3 and n["src__s02__revoke_permission"] == 2
    assert n["src__s00__high_risk__control"] == 2 and n["src__s00__feasible"] == 2
    assert dp.samples_for({"kind": "infeasible", "family": "corrupt_input"}, dp.DEFAULT_SAMPLES) == 2


def test_plain_rows_come_from_the_band_and_fill_from_easy_rows_only():
    rows = pool(12)
    passes = {r["variant_id"]: (4, 2) for r in rows if r["kind"] != "feasible"}
    plain = [r["variant_id"] for r in rows if r["kind"] == "feasible"]
    # 3 in the band, 4 too easy, 3 never pass, 2 unmeasured
    for vid, p in zip(plain[:10], [1, 2, 3, 4, 4, 4, 4, 0, 0, 0], strict=True):
        passes[vid] = (4, p)
    built = plan(rows, passes=passes, plain_n=5)
    drawn = [r["variant_id"] for r in built.rows if r["kind"] == "feasible"]
    assert set(plain[:3]) <= set(drawn) and len(drawn) == 5
    assert set(drawn) <= set(plain[:7])  # never a 0/4 or unmeasured row
    assert (built.plain_band_rows, built.plain_fill_rows, built.plain_shortfall) == (3, 2, 0)
    assert len(built.excluded[dp.EXCL_PLAIN_NEVER_PASSES]) == 3
    assert len(built.excluded[dp.EXCL_PLAIN_UNMEASURED]) == 2
    assert len(built.excluded[dp.EXCL_PLAIN_NOT_DRAWN]) == 2
    # no fill: the band only, and the shortfall is reported
    strict = plan(rows, passes=passes, plain_n=5, plain_fill=dp.PLAIN_FILL_NONE)
    assert (strict.plain_band_rows, strict.plain_fill_rows, strict.plain_shortfall) == (3, 0, 2)
    # stable: the same seed draws the same rows
    assert plan(rows, passes=passes, plain_n=5).rows == built.rows
    # twins and controls are never thinned
    assert sum(r["kind"] != "feasible" for r in built.rows) == 24


def test_options_are_checked():
    with pytest.raises(ValueError, match="plain_fill"):
        dp.DistillOptions(plain_fill="hard")
    with pytest.raises(ValueError, match="plain_band"):
        dp.DistillOptions(plain_band=(3, 1))


def _bank_line(r: dict, passed: bool) -> str:
    outcome = EpisodeOutcome(feasible=r["feasible"], report=None, tests_passed=passed, n_actions=3, agent_acted=True)
    return BankRow(task=r, group=r["variant_id"], outcome=outcome_to_dict(outcome)).model_dump_json() + "\n"


def test_cli_writes_one_file_per_sample_count_and_reports(tmp_path: Path, capsys):
    rows = pool(12)
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "rl.jsonl").write_text(json.dumps(row("s11", "feasible")) + "\n")
    (tmp_path / "validation.jsonl").write_text(json.dumps({"variant_id": "src__s03__high_risk", "leak": True}) + "\n")
    lines = []
    for r in rows:
        for k in range(4):
            if r["kind"] == "feasible":
                passed = k < (1 if r["pair_id"] < "src/s06" else 4)  # s00-s05 in the band, s06-s10 easy
            else:
                passed = r["kind"] == "control" or (r["variant_id"] == "src__s01__remove_input" and k == 0)
            lines.append(_bank_line(r, passed))
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "bank-1.jsonl").write_text("".join(lines))
    out = tmp_path / "out" / "distill_pool.jsonl"
    rc = dp.main(
        [
            "--pool", str(tmp_path / "pool.jsonl"), "--rl-pool", str(tmp_path / "rl.jsonl"),
            "--validation", str(tmp_path / "validation.jsonl"), "--bank", str(bank),
            "--out", str(out), "--report", str(tmp_path / "report.json"), "--plain-n", "8",
        ]
    )  # fmt: skip
    assert rc == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["excluded"][dp.EXCL_RL_POOL] == 3
    assert report["excluded"][EXCL_LEAK_VALIDATION] == 1 and report["excluded"][EXCL_LEAK_BANK] == 1
    assert report["plain"] | {"passes_of_drawn": None} == {
        "requested": 8, "band": [1, 3], "from_band": 6, "filled_above_band": 2, "shortfall": 0, "passes_of_drawn": None,
    }  # fmt: skip
    written = {Path(p).name: n for p, n in report["files"].items()}
    assert written == {"distill_pool_x4.jsonl": 3, "distill_pool_x3.jsonl": 3, "distill_pool_x2.jsonl": 8 + 11 + 3}
    whole = [json.loads(line) for line in out.read_text().splitlines()]
    assert sum(r["n_samples"] for r in whole) == report["summary"]["episodes"] == 3 * 4 + 3 * 3 + 22 * 2
    for name, n in (("distill_pool_x4.jsonl", 4), ("distill_pool_x2.jsonl", 2)):
        part = [json.loads(line) for line in (out.parent / name).read_text().splitlines()]
        assert {r["n_samples"] for r in part} == {n}
    assert any("--n-samples 4" in line and "hint_tier1.txt" in line for line in report["run_eval"])
    assert json.loads(capsys.readouterr().out)["summary"]["rows"] == len(whole)


def test_cli_refuses_a_missing_bank(tmp_path: Path, capsys):
    (tmp_path / "pool.jsonl").write_text(json.dumps(row("a", "feasible")) + "\n")
    (tmp_path / "rl.jsonl").write_text("")
    (tmp_path / "bank").mkdir()
    rc = dp.main(
        ["--rl-pool", str(tmp_path / "rl.jsonl"), "--pool", str(tmp_path / "pool.jsonl"),
         "--bank", str(tmp_path / "bank"), "--out", str(tmp_path / "o.jsonl")]
    )  # fmt: skip
    assert rc == 2 and "no train-split rows" in capsys.readouterr().err
