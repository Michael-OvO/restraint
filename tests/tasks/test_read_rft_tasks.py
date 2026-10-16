"""The G4 sampling task list: surfaces, pairing, source rules, determinism and the leak assertions, on a synthetic tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.tasks import read_rft_tasks as rt


def write_task(root: Path, pair: str, surface: str, control: bool, family: str = "high_risk") -> None:
    name = f"{pair}__{family}__{surface}" + ("__control" if control else "")
    d = root / name
    d.mkdir(parents=True)
    feasible = "true" if control else "false"
    (d / "task.toml").write_text(
        f'[metadata.provenance]\nsource = "s"\n[metadata.honest]\nfeasible = {feasible}\nis_control = {str(control).lower()}\n'
        f'family = "{family}"\npair_id = "{pair}"\n[metadata.honest.detail]\nsurface = "{surface}"\n'
    )


def pool(path: Path, *pairs: str) -> str:
    path.write_text("".join(json.dumps({"pair_id": p, "variant_id": f"{p}__v"}) + "\n" for p in pairs))
    return str(path)


@pytest.fixture
def tree(tmp_path: Path):
    tasks, val = tmp_path / "tasks", tmp_path / "val"
    val.mkdir()
    for i in range(12):
        for s in rt.SURFACES_TRAIN:
            write_task(tasks, f"p{i}", s, False)
            write_task(tasks, f"p{i}", s, True)
    write_task(tasks, "p0", "empty_backup", False)  # a held-out surface: never listed
    write_task(tasks, "p1", "manifest", False, family="other")
    (tasks / "p3__high_risk__manifest__control" / "task.toml").unlink()  # p3 has no manifest control
    write_task(val, "p5", "inbound_link", False)  # p5 is a source of the held-out surface set
    pools = [
        pool(tmp_path / "rl.jsonl", *[f"p{i}" for i in range(8)]),
        pool(tmp_path / "dp.jsonl", "p8", "p9", "p10", "p11"),
    ]
    forbidden = [pool(tmp_path / "sel.jsonl", "p2"), pool(tmp_path / "val.jsonl", "p9")]
    return tasks, val, pools, forbidden


def build(tree, **kw):
    tasks, val, pools, forbidden = tree
    return rt.build_rows(tasks, pools=pools, forbidden=forbidden, val_root=val, **{"pairs_per_surface": 5, **kw})


def test_balanced_pairs_of_training_surfaces_only(tree):
    rows, info = build(tree)
    s = rt.summarize(rows)
    assert s["tasks"] == 40 and set(s["by_surface_side"].values()) == {5}
    assert {r["detail"]["surface"] for r in rows} == set(rt.SURFACES_TRAIN)
    assert all(r["split"] == "train" and r["family"] == "high_risk" for r in rows)
    assert {r["is_control"] for r in rows} == {True, False} and all(
        r["task_dir"].startswith("/root/restraint/") for r in rows
    )
    assert info["skipped"]["select_val_test_probe_source"] >= 1 and info["skipped"]["missing_side"] >= 1


def test_forbidden_val_surface_and_unpooled_sources_never_listed(tree):
    rows, _ = build(tree, pairs_per_surface=7)
    pairs = {r["pair_id"] for r in rows}
    assert pairs.isdisjoint({"p2", "p9", "p5"})
    assert pairs <= {f"p{i}" for i in range(12)}


def test_too_few_eligible_pairs_fails(tree):
    with pytest.raises(ValueError, match="eligible pairs"):
        build(tree, pairs_per_surface=11)


def test_deterministic_and_seeded(tree):
    a, _ = build(tree)
    b, _ = build(tree)
    c, _ = build(tree, seed="other")
    assert a == b and a != c


def test_assert_clean_catches_leaks(tree):
    _, val, pools, forbidden = tree
    rows, _ = build(tree)
    rt.assert_clean(rows, forbidden=forbidden, val_root=val, pools=pools)
    leak = [*rows, {**rows[0], "pair_id": "p2", "variant_id": "x"}]
    with pytest.raises(AssertionError, match="select / val / test"):
        rt.assert_clean(leak, forbidden=forbidden, val_root=val, pools=pools)
    with pytest.raises(AssertionError, match="surfaces_val"):
        rt.assert_clean([{**rows[0], "pair_id": "p5", "variant_id": "y"}], forbidden=[], val_root=val, pools=pools)
    with pytest.raises(AssertionError, match="outside"):
        rt.assert_clean([{**rows[0], "pair_id": "zz", "variant_id": "z"}], forbidden=[], val_root=val, pools=pools)
    with pytest.raises(AssertionError, match="not training surfaces"):
        rt.assert_clean([{**rows[0], "detail": {"surface": "empty_backup"}}], forbidden=[], val_root=val, pools=pools)


def test_cli_build_and_verify_roundtrip(tree, tmp_path: Path, capsys):
    tasks, val, pools, forbidden = tree
    out = tmp_path / "list.jsonl"
    args = ["--tasks-root", str(tasks), "--val-root", str(val), "--out", str(out), "--pools", *pools]
    assert rt.main(["build", *args, "--forbidden", *forbidden, "--pairs-per-surface", "3"]) == 0
    assert len(rt.read_jsonl(out)) == 24
    assert rt.main(["verify", *args, "--forbidden", *forbidden]) == 0 and "PASS" in capsys.readouterr().out
    with pytest.raises(AssertionError):  # the verify catches a source that became forbidden after the build
        rt.main(
            ["verify", *args, "--forbidden", pool(tmp_path / "late.jsonl", *{r["pair_id"] for r in rt.read_jsonl(out)})]
        )
