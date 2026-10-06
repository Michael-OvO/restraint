from __future__ import annotations

import json

from honest_agents.tasks.rl_pool import build_pool, main, summarize


def row(seed: str, kind: str, family: str | None = None) -> dict:
    fam = family or "none"
    suffix = {"feasible": "feasible", "control": f"{fam}__control", "infeasible": fam}[kind]
    return {
        "variant_id": f"src__{seed}__{suffix}",
        "pair_id": f"src/{seed}",
        "kind": kind,
        "family": family,
        "is_control": kind == "control",
        "feasible": kind != "infeasible",
    }


def pool() -> list[dict]:
    rows = []
    for seed in ("a", "b", "c", "d"):  # a, b kept by the difficulty gate; c, d too easy
        rows.append(row(seed, "feasible"))
        for fam in ("high_risk", "revoke_permission"):
            rows += [row(seed, "control", fam), row(seed, "infeasible", fam)]
    return rows


def test_feasible_side_is_gated_but_impossible_variants_of_dropped_seeds_stay():
    out = build_pool(pool(), {"src/a", "src/b"}, max_infeasible=None)
    ids = {r["variant_id"] for r in out}
    assert "src__c__feasible" not in ids and "src__d__feasible" not in ids  # too easy: dropped
    assert "src__c__high_risk" in ids and "src__d__revoke_permission" in ids  # their impossible twins stay
    assert "src__c__high_risk__control" in ids  # ... with the near-miss control that guards against over-refusal
    assert sum(r["kind"] == "infeasible" for r in out) == 8


def test_infeasible_sample_is_balanced_over_families_deterministic_and_keeps_pool_order():
    rows = pool() + [row(f"x{i}", "infeasible", "remove_input") for i in range(10)]  # a big family
    out = build_pool(rows, set(), max_infeasible=6, seed=3)
    fams = [r["family"] for r in out if r["kind"] == "infeasible"]
    assert sorted(fams) == ["high_risk"] * 2 + ["remove_input"] * 2 + ["revoke_permission"] * 2
    assert out == build_pool(rows, set(), max_infeasible=6, seed=3)
    order = {r["variant_id"]: i for i, r in enumerate(rows)}
    assert [order[r["variant_id"]] for r in out] == sorted(order[r["variant_id"]] for r in out)
    for r in out:
        if r["kind"] == "infeasible":  # every selected impossible variant brings its control
            assert any(c["pair_id"] == r["pair_id"] and c["kind"] == "control" and c["family"] == r["family"] for c in out) or (
                not any(c["pair_id"] == r["pair_id"] and c["kind"] == "control" and c["family"] == r["family"] for c in rows)
            )


def test_cli_writes_rows_unchanged(tmp_path, capsys):
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool()))
    (tmp_path / "diff.json").write_text(json.dumps({"keep_pairs": ["src/a"]}))
    main(["--pool", str(tmp_path / "pool.jsonl"), "--difficulty", str(tmp_path / "diff.json"), "--out", str(tmp_path / "rl.jsonl")])
    written = [json.loads(line) for line in (tmp_path / "rl.jsonl").read_text().splitlines()]
    assert written and all(r in pool() for r in written)
    assert json.loads(capsys.readouterr().out) == summarize(written)
