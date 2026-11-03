"""The data bundle round-trips byte-for-byte with mtimes, and refuses tampered or clashing inputs."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pytest

from honest_agents import data_bundle as db


def make_repo(root: Path) -> None:
    for rel, text in {
        "data/tasks/t1/task.toml": "a",
        "data/tasks/t1/environment/Dockerfile": "FROM x",
        "data/decision/surfaces/tasks/s1/task.toml": "b",
        "data/weights/it14/sft-v2-e3-final/adapter_config.json": "{}",
        "data/weights/it15/g2-s1-final/adapter_config.json": "{}",
        "data/weights/other/unused.bin": "skip",
        "data/split_seeds.json": "{}",
        "data/spec_audit/reviews.jsonl": "{}\n",
        "data/rts/v1/big.bin": "excluded",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    os.utime(root / "data/tasks/t1/task.toml", (1_700_000_000, 1_700_000_000))


def test_pack_restore_round_trip_keeps_bytes_and_mtimes_and_skips_rts(tmp_path):
    src, bundle, dest = tmp_path / "src", tmp_path / "bundle", tmp_path / "dest"
    make_repo(src)
    manifest = db.pack(bundle, repo=src)
    assert set(manifest["parts"]) == {"tasks", "decision", "weights", "data"}
    assert db.verify(bundle) == []
    n = db.restore(bundle, dest)
    assert n == 7  # rts and the unused weights are not bundled
    assert (dest / "data/spec_audit/reviews.jsonl").read_text() == "{}\n"
    assert not (dest / "data/rts").exists() and not (dest / "data/weights/other").exists()
    assert int((dest / "data/tasks/t1/task.toml").stat().st_mtime) == 1_700_000_000


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
    assert db.restore(bundle, dest, force=True) == 7


def test_archive_part_restores_under_archive(tmp_path):
    src, arch, bundle, dest = tmp_path / "src", tmp_path / "arch", tmp_path / "bundle", tmp_path / "dest"
    make_repo(src)
    for rel in db.ARCHIVE_EXTRAS:
        (arch / rel).mkdir(parents=True, exist_ok=True)
        (arch / rel / "f.txt").write_text(rel)
    db.pack(bundle, repo=src, archive=arch)
    db.restore(bundle, dest)
    assert (dest / "archive/staging/analysis/f.txt").read_text() == "staging/analysis"
