"""DEF-07: recorded pair assignments keep every existing variant byte-identical when a pool grows (sweep S7)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from honest_agents.tasks import pool_build
from honest_agents.tasks.perturb.risk_surfaces import (
    ASSIGNMENTS_FILE,
    SURFACES,
    Assignment,
    AssignmentsFile,
    PoolError,
    build_varied,
    load_pool,
    pool_dir,
)
from honest_agents.tasks.validate import task_hash
from tests.tasks.synthetic import rich_task

REPO = Path(__file__).resolve().parents[2]
IDS = [f"synthetic/grow-{i}" for i in range(10)]


@pytest.fixture
def rich(tmp_path) -> Path:
    return rich_task(tmp_path / "src" / "rich", verifier="pytest")


@pytest.fixture
def pools(tmp_path) -> Path:
    """A copy of the repo pools at `<tmp>/pools/surfaces`, with no assignments next to it."""
    d = tmp_path / "pools" / "surfaces"
    shutil.copytree(pool_dir(), d)
    return d


def _build(src: Path, out: Path, pools: Path, split: str = "test") -> dict[str, tuple[str, str, str]]:
    """variant id -> (task hash, pool pair, context) for IDS x SURFACES."""
    got = {}
    for s in SURFACES:
        pool = load_pool(s, directory=pools)
        for pid in IDS:
            built = build_varied(src, out, pair_id=pid, surface=s, pool=pool, split=split)
            assert built is not None
            for v in built:
                got[v.variant_id] = (task_hash(v.task_dir), v.detail["pool_pair"], v.detail["context"])
    return got


def _record(pools: Path, built: dict[str, tuple[str, str, str]], split: str = "test") -> None:
    variants = {
        vid: Assignment(surface=vid.split("__high_risk__")[1], pair=pair, context=ctx, split=split)
        for vid, (_, pair, ctx) in built.items()
        if not vid.endswith("__control")
    }
    (pools.parent / ASSIGNMENTS_FILE).write_text(json.dumps(AssignmentsFile(variants=variants).model_dump()))


def _grow(pools: Path, n: int = 3) -> None:
    """Append `n` held-out pairs (copies of an existing held-out pair under new ids) to every pool."""
    for s in SURFACES:
        f = pools / f"{s}.json"
        raw = json.loads(f.read_text())
        tpl = next(p for p in raw["pairs"] if p.get("heldout"))
        raw["pairs"] += [{**tpl, "id": f"{tpl['id']}-new{k}", "heldout": True} for k in range(n)]
        f.write_text(json.dumps(raw))


def test_growing_a_pool_leaves_every_recorded_variant_byte_identical(rich, pools, tmp_path):
    before = _build(rich, tmp_path / "a", pools)
    _record(pools, before)
    assert _build(rich, tmp_path / "b", pools) == before  # honouring the record reproduces the build
    _grow(pools)
    assert _build(rich, tmp_path / "c", pools) == before


def test_without_the_record_growth_moves_variants(rich, pools, tmp_path):
    """The old behaviour (and the sensitivity of the test above): with no assignments, appending pairs re-draws."""
    before = _build(rich, tmp_path / "a", pools)
    _grow(pools)
    after = _build(rich, tmp_path / "b", pools)
    assert sum(after[v][0] != before[v][0] for v in before) > 0


def test_a_new_variant_draws_as_if_nothing_were_recorded(rich, pools, tmp_path):
    """An unrecorded variant draws over the whole split. Drawing only from unheld pairs piled the new sources of one
    build onto the few pairs left (48 of the 68-source eval's inbound_link twins on one pair)."""
    new = [f"synthetic/new-{i}" for i in range(8)]

    def draws() -> list[tuple[str, str]]:
        out = []
        for s in SURFACES:
            pool = load_pool(s, directory=pools)
            for pid in new:
                twin, _ = build_varied(rich, tmp_path / "n", pair_id=pid, surface=s, pool=pool, split="test")
                out.append((twin.detail["pool_pair"], twin.detail["context"]))
        return out

    free = draws()
    _record(pools, _build(rich, tmp_path / "a", pools))
    assert draws() == free


def test_an_assigned_pair_cannot_leave_the_pool_or_cross_the_split(rich, pools, tmp_path):
    before = _build(rich, tmp_path / "a", pools)
    _record(pools, before)
    vid = next(v for v in before if v.endswith("__high_risk__readme"))
    with pytest.raises(PoolError, match="split"):
        build_varied(rich, tmp_path / "x", pair_id=IDS[0], surface="readme", pool=load_pool("readme", directory=pools))
    f = pools / "readme.json"
    raw = json.loads(f.read_text())
    raw["pairs"] = [p for p in raw["pairs"] if p["id"] != before[vid][1]]
    f.write_text(json.dumps(raw))
    with pytest.raises(PoolError, match="not in the pool"):
        load_pool("readme", directory=pools)


def test_strict_requires_an_explicit_heldout_flag(pools):
    for s in SURFACES:  # the repo pools carry the flag on every pair
        assert all(p.heldout is not None for p in load_pool(s, strict=True).pairs)
    f = pools / "manifest.json"
    raw = json.loads(f.read_text())
    raw["pairs"][0].pop("heldout")
    f.write_text(json.dumps(raw))
    load_pool("manifest", directory=pools)  # non-strict: still the hash fallback
    with pytest.raises(PoolError, match="explicit `heldout`"):
        load_pool("manifest", directory=pools, strict=True)


def test_compare_lists_changed_missing_and_new():
    r = pool_build.BuildResult(
        [pool_build.Built("a", "readme", False, "h1", "p", ""), pool_build.Built("c", "readme", False, "h3", "p", "")]
    )
    assert pool_build.compare({"a": "h0", "b": "h2"}, r) == (["a"], ["b"], ["c"])


@pytest.mark.skipif(not (REPO / pool_build.DEFAULT_BANK).is_dir(), reason="needs the local task bank (data/tasks)")
def test_committed_manifest_matches_a_rebuild(monkeypatch, tmp_path):
    """Sweep S7 at L0: the surface eval rebuilds byte-identical to `data/pools/build_manifest.json`. The manifest
    carries its sources, so the (gitignored) ids file is not needed: the check runs with it pointed at a missing path."""
    monkeypatch.chdir(REPO)
    assert pool_build.main(["check"]) == 0
    raw = json.loads(pool_build.manifest_path().read_text())
    pools = tmp_path / "pools" / "surfaces"
    shutil.copytree(pool_dir(), pools)
    shutil.copy(REPO / "data" / "pools" / ASSIGNMENTS_FILE, pools.parent / ASSIGNMENTS_FILE)
    no_ids = {**raw, "build": {**raw["build"], "ids_file": str(tmp_path / "absent.txt")}}
    pool_build.manifest_path(pools).write_text(json.dumps(no_ids))
    assert pool_build.main(["--pools", str(pools), "check"]) == 0
    (tmp_path / "absent.txt").write_text("\n".join(raw["sources"][:-1]) + "\n")  # the ids file drifted
    assert pool_build.main(["--pools", str(pools), "check"]) == 1
