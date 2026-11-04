"""The data bundle round-trips byte-for-byte with mtimes, refuses tampered or clashing inputs, and splits the held-out
files from the rest: train has none of them, final has only them, and the two restore together without a clash."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pytest

from honest_agents import data_bundle as db

TRAIN_FILES = {
    "data/tasks/et__task_1__feasible/task.toml": "a",
    "data/tasks/et__task_1__feasible/environment/Dockerfile": "FROM x",
    "data/decision/surfaces/tasks/s1/task.toml": "b",
    "data/weights/it14/sft-v2-e3-final/adapter_config.json": "{}",
    "data/weights/it15/g2-s1-final/adapter_config.json": "{}",
    "data/split_seeds.json": "{}",
    "data/spec_audit/reviews.jsonl": "{}\n",
    "data/test_ids.txt": "# held-out sources\net/task_9\n",
}
HELD_OUT_FILES = {
    "data/tasks/et__task_9__feasible/task.toml": "held",
    "data/tasks/et__task_9__high_risk__control/tests/test_outputs.py": "hidden test",
    "data/audit/seeds/et/task_9/task.toml": "seed",
}
FINAL_ONLY = {"data/def74/affected.jsonl": "{}\n"}
UNBUNDLED = {"data/weights/other/unused.bin": "skip", "data/rts/v1/big.bin": "excluded"}


def make_repo(root: Path) -> None:
    for rel, text in {**TRAIN_FILES, **HELD_OUT_FILES, **FINAL_ONLY, **UNBUNDLED}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    os.utime(root / "data/tasks/et__task_1__feasible/task.toml", (1_700_000_000, 1_700_000_000))


def test_pack_restore_round_trip_keeps_bytes_and_mtimes_and_skips_rts(tmp_path):
    src, bundle, dest = tmp_path / "src", tmp_path / "bundle", tmp_path / "dest"
    make_repo(src)
    manifest = db.pack(bundle, repo=src)
    assert set(manifest["parts"]) == {"tasks", "decision", "weights", "data"}
    assert db.verify(bundle) == []
    assert db.restore(bundle, dest) == len(TRAIN_FILES)
    assert (dest / "data/spec_audit/reviews.jsonl").read_text() == "{}\n"
    assert not (dest / "data/rts").exists() and not (dest / "data/weights/other").exists()
    assert int((dest / "data/tasks/et__task_1__feasible/task.toml").stat().st_mtime) == 1_700_000_000


def test_train_holds_no_held_out_file_and_final_only_those_and_together_they_are_complete(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_repo(src)
    db.pack(tmp_path / "train", repo=src, split="train")
    db.pack(tmp_path / "final", repo=src, split="final")
    db.restore(tmp_path / "train", dest)
    train = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert train == set(TRAIN_FILES)
    assert db.restore(tmp_path / "final", dest) == len(HELD_OUT_FILES) + len(FINAL_ONLY)  # no clash
    got = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert got == set(TRAIN_FILES) | set(HELD_OUT_FILES) | set(FINAL_ONLY)


def test_names_held_out_matches_variant_dirs_and_seed_paths_only():
    s = frozenset({"et/task_9"})
    assert db.names_held_out("data/tasks/et__task_9__revoke_permission__control/task.toml", s)
    assert db.names_held_out("data/audit_run1/raw/et/task_9/instruction.md", s)
    assert not db.names_held_out("data/tasks/et__task_99__feasible/task.toml", s)
    assert not db.names_held_out("data/audit/raw/et/task_99/task.toml", s)


def test_pack_fails_closed_without_the_held_out_list(tmp_path):
    make_repo(tmp_path / "src")
    (tmp_path / "src/data/test_ids.txt").unlink()
    with pytest.raises(FileNotFoundError, match="held-out"):
        db.pack(tmp_path / "bundle", repo=tmp_path / "src")


def test_a_tampered_part_or_member_fails_verification(tmp_path):
    src, bundle = tmp_path / "src", tmp_path / "bundle"
    make_repo(src)
    db.pack(bundle, repo=src)
    m = json.loads((bundle / db.MANIFEST).read_text())
    m["parts"]["data"]["files"]["data/split_seeds.json"] = "0" * 64
    (bundle / db.MANIFEST).write_text(json.dumps(m))
    assert any("split_seeds" in p for p in db.verify(bundle))
    with pytest.raises(ValueError, match="verification"):
        db.restore(bundle, tmp_path / "dest")
    with tarfile.open(bundle / "tasks.tar.gz", "w:gz"):
        pass
    assert any("tasks" in p for p in db.verify(bundle))


def test_restore_refuses_to_overwrite_without_force(tmp_path):
    src, bundle, dest = tmp_path / "src", tmp_path / "bundle", tmp_path / "dest"
    make_repo(src)
    db.pack(bundle, repo=src)
    (dest / "data").mkdir(parents=True)
    (dest / "data/split_seeds.json").write_text("local")
    with pytest.raises(FileExistsError):
        db.restore(bundle, dest)
    assert db.restore(bundle, dest, force=True) == len(TRAIN_FILES)


def test_archive_parts_split_by_root_and_restore_under_archive(tmp_path, monkeypatch):
    src, arch, dest = tmp_path / "src", tmp_path / "arch", tmp_path / "dest"
    make_repo(src)
    monkeypatch.setattr(db, "archived_builds", lambda archive: ("iterations/x/run/r/examples.jsonl",))
    for rel in (*db.TRAIN_ARCHIVE, *db.FINAL_ARCHIVE):
        (arch / rel).mkdir(parents=True, exist_ok=True)
        (arch / rel / "f.txt").write_text(rel)
    (arch / "staging/analysis/it20_design/et__task_9__feasible.json").write_text("held")
    (arch / "iterations/x/run/r").mkdir(parents=True)
    (arch / "iterations/x/run/r/examples.jsonl").write_text("{}\n")
    db.pack(tmp_path / "train", repo=src, archive=arch, split="train")
    db.pack(tmp_path / "final", repo=src, archive=arch, split="final")
    db.restore(tmp_path / "train", dest)
    assert (dest / "archive/staging/analysis/it20_design/f.txt").read_text() == "staging/analysis/it20_design"
    assert (dest / "archive/iterations/x/run/r/examples.jsonl").is_file()
    assert not (dest / "archive/benchmarks").exists()
    assert not (dest / "archive/staging/analysis/it20_design/et__task_9__feasible.json").exists()
    db.restore(tmp_path / "final", dest)
    assert (dest / "archive/benchmarks/f.txt").read_text() == "benchmarks"
    assert (dest / "archive/staging/analysis/it20_design/et__task_9__feasible.json").read_text() == "held"
