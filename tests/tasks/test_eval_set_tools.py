"""Mac-side evaluation-set helpers: `surfaces_cli ids` / `complete`, `g1_surfaces.prune_failed`,
`validate.write_quarantine_list`."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from honest_agents.bank import read_quarantine
from honest_agents.tasks import g1_surfaces, surfaces_cli, validate


def test_ids_command(tmp_path: Path) -> None:
    split = tmp_path / "val.jsonl"
    rows = [
        {"pair_id": "a/1", "family": "high_risk", "is_control": False},
        {"pair_id": "a/1", "family": "high_risk", "is_control": True},
        {"pair_id": "a/2", "family": "feasible", "is_control": False},
        {"pair_id": "a/3", "family": "high_risk", "is_control": False},
    ]
    split.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "ids.txt"
    assert surfaces_cli.main(["ids", "--split-file", str(split), "--out", str(out), "--family", "high_risk", "--twins-only"]) == 0
    assert out.read_text().split() == ["a/1", "a/3"]


def test_complete_command(tmp_path: Path, capsys) -> None:
    """DEF-05 (sweep S5): every source has both dirs of every surface, and every high_risk episode resolves to one."""
    tasks = tmp_path / "tasks"
    for v in ("org__a__high_risk__readme", "org__a__high_risk__readme__control", "org__b__high_risk__readme"):
        (tasks / v).mkdir(parents=True)
        (tasks / v / "task.toml").write_text("")
    eps = tmp_path / "episodes.jsonl.gz"
    rows = [{"family": "high_risk", "task_id": "org__a__high_risk__readme"}, {"family": "feasible", "task_id": "x"}]
    eps.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    base = ["complete", "--tasks", str(tasks), "--surface", "readme"]
    assert surfaces_cli.main([*base, "--ids", "org/a", "--episodes", str(eps)]) == 0
    assert surfaces_cli.main([*base, "--ids", "org/a", "org/b"]) == 1
    assert "MISSING dir org__b__high_risk__readme__control" in capsys.readouterr().out
    rows.append({"family": "high_risk", "task_id": "org__c__high_risk__readme"})
    eps.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode()))
    assert surfaces_cli.main([*base, "--ids", "org/a", "--episodes", str(eps)]) == 1
    assert "MISSING episode task org__c__high_risk__readme" in capsys.readouterr().out


VAL_TASKS = Path(__file__).resolve().parents[2] / "data" / "decision" / "surfaces_val" / "tasks"
VAL_IDS = VAL_TASKS.parent.parent / "val_high_risk_sources.txt"


@pytest.mark.skipif(
    not VAL_TASKS.is_dir() or not VAL_IDS.is_file(), reason="no local surfaces_val build (data/ is untracked)"
)
def test_the_local_surfaces_val_build_has_every_source() -> None:
    """DEF-05: 17a590fc, one of the 20 val sources the it14/it15 surface evals ran on, had no task dirs (228 of 240)."""
    assert surfaces_cli.missing_dirs(VAL_TASKS, surfaces_cli._ids(SimpleNamespace(ids=None, ids_file=VAL_IDS)),
                                     surfaces_cli._surfaces(None)) == []  # fmt: skip


def test_prune_failed(tmp_path: Path) -> None:
    rows = []
    for v in ("p1", "p2"):
        for suffix in ("", "__control"):
            d = tmp_path / "tasks" / (v + suffix)
            d.mkdir(parents=True)
            rows.append({"variant_id": v + suffix, "task_dir": str(d), "control": bool(suffix)})
    f = tmp_path / "surface_variants.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    results = [g1_surfaces.PairCheck("p1", "readme"), g1_surfaces.PairCheck("p2", "readme", problems=["x"])]
    assert g1_surfaces.prune_failed(f, rows, results) == 1
    assert sorted(p.name for p in (tmp_path / "tasks").iterdir()) == ["p1", "p1__control"]
    assert [json.loads(ln)["variant_id"] for ln in f.read_text().splitlines()] == ["p1", "p1__control"]


def test_write_quarantine_list(tmp_path: Path) -> None:
    results = {
        "good": SimpleNamespace(ok=True, leak=False, relabel_to_control=False, reason=""),
        "bad": SimpleNamespace(ok=False, leak=False, relabel_to_control=False, reason="oracle failed"),
        "leaky": SimpleNamespace(ok=False, leak=True, relabel_to_control=False, reason="x"),
    }
    p = tmp_path / "q.txt"
    assert validate.write_quarantine_list(results, p) == 2
    es = read_quarantine(p)
    assert {e.variant_id for e in es} == {"bad", "leaky"} and all(e.kind == "leak" for e in es)


def test_prune_from_earlier_results(tmp_path: Path) -> None:
    from dataclasses import asdict

    d = tmp_path / "tasks" / "p1"
    d.mkdir(parents=True)
    f = tmp_path / "v.jsonl"
    f.write_text(json.dumps({"variant_id": "p1", "task_dir": str(d), "control": False}) + "\n")
    g1 = tmp_path / "g1.jsonl"
    g1.write_text(json.dumps(asdict(g1_surfaces.PairCheck("p1", "readme", problems=["x"]))) + "\n")
    assert g1_surfaces.main(["--variants", str(f), "--prune-from", str(g1)]) == 0
    assert not d.exists() and f.read_text() == ""
