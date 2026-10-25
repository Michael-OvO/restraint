"""The RTS manifest: round trip, digest sidecar, pair validation."""

from __future__ import annotations

import pytest

from honest_agents.eval.rts.manifest import ManifestItem, load_manifest, pair_problems, write_manifest
from tests.eval.rts.helpers import pairs


def test_round_trip_and_digest(tmp_path):
    items = pairs("tb2", "mid", "remove_tool", 3)
    path = tmp_path / "v1" / "manifest.jsonl"
    digest = write_manifest(reversed(items), path)
    assert (tmp_path / "v1" / "manifest.jsonl.sha256").read_text().strip() == digest
    loaded = load_manifest(path)
    assert loaded == sorted(items, key=lambda i: i.item_id)
    twin = next(i for i in loaded if i.role == "twin")
    assert twin.expected_categories == ("missing_capability",)


def test_edited_manifest_is_refused(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(pairs("tb2", "mid", "remove_tool", 2), path)
    path.write_text(path.read_text().replace("remove_tool", "remove_tooX"))
    with pytest.raises(ValueError, match="never edited"):
        load_manifest(path)


def test_missing_sidecar_is_refused(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest(pairs("tb2", "mid", "remove_tool", 1), path)
    (tmp_path / "manifest.jsonl.sha256").unlink()
    with pytest.raises(ValueError, match="digest"):
        load_manifest(path)


def test_pair_problems():
    items = pairs("tb2", "mid", "remove_tool", 2)
    assert pair_problems(items) == []
    lone_twin = [i for i in items if not (i.role == "control" and i.pair_id == items[0].pair_id)]
    assert any("one twin and one control" in p for p in pair_problems(lone_twin))
    twin, ctrl = items[0], items[1]
    moved = ManifestItem(ctrl.item_id, ctrl.benchmark, ctrl.source_task, ctrl.pair_id, "far", ctrl.operator, "control")
    assert any("differ in tier" in p for p in pair_problems([twin, moved]))
    assert any("duplicate item ids" in p for p in pair_problems([twin, twin, ctrl]))


def test_write_refuses_invalid_pairs(tmp_path):
    with pytest.raises(ValueError, match="invalid manifest"):
        write_manifest(pairs("tb2", "mid", "remove_tool", 1)[:1], tmp_path / "m.jsonl")


def test_unknown_tier_or_role_is_refused():
    with pytest.raises(ValueError, match="tier"):
        ManifestItem("a", "tb2", "s", "p", "medium", "op", "twin")
    with pytest.raises(ValueError, match="role"):
        ManifestItem("a", "tb2", "s", "p", "mid", "op", "impossible")
