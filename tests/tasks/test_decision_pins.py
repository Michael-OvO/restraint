"""DEF-18: the decision probe / SFT surface builds keep every recorded twin byte-identical when a pool grows (sweep S7)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from honest_agents.tasks import decision_surfaces as dsurf
from honest_agents.tasks import pool_build
from honest_agents.tasks.perturb import OPERATORS, make_feasible
from honest_agents.tasks.perturb.risk_surfaces import POOL_ENV, SURFACES, PoolError, _cached_pool, pool_dir
from honest_agents.tasks.validate import task_hash
from tests.tasks.synthetic import rich_task

REPO = Path(__file__).resolve().parents[2]
SFT = [f"synthetic/sft-{i}" for i in range(4)]
PROBE = [f"synthetic/probe-{i}" for i in range(4)]
HELD = [f"synthetic/held-{i}" for i in range(2)]
LEFTOVER = "synthetic/leftover-0"  # built on readme into the root, but not in surface_pairs.jsonl
ROOT = Path("data/decision/surfaces/tasks")
PAIRS = Path("data/decision/surface_pairs.jsonl")


@pytest.fixture
def work(tmp_path, monkeypatch) -> Iterator[Path]:
    """`tmp_path` as a repo root: a synthetic bank at data/tasks, the repo configs, and a copy of the repo pools (with no
    assignments next to it) that `pool_dir()` points at."""
    rich = rich_task(tmp_path / "src" / "rich", verifier="pytest")
    bank = tmp_path / "data" / "tasks"
    for pid in (*SFT, *PROBE, *HELD, LEFTOVER):
        make_feasible(rich, bank, pair_id=pid)
        assert OPERATORS["high_risk"](rich, bank, pair_id=pid) is not None
    pools = tmp_path / "pools" / "surfaces"
    shutil.copytree(pool_dir(), pools)
    (tmp_path / "configs").symlink_to(REPO / "configs")
    monkeypatch.setenv(POOL_ENV, str(pools))
    monkeypatch.chdir(tmp_path)
    _cached_pool.cache_clear()
    yield pools
    _cached_pool.cache_clear()


def _build(out: Path) -> list[dsurf.SurfacePairRow]:
    """The SFT rows, then the probe rows with "seen" from the SFT rows (as `decision_surfaces build` runs them)."""
    bank = Path("data/tasks")
    sft, _ = dsurf.build_sft_rows(SFT, bank_tasks=bank, out_root=out)
    probe = dsurf.ProbeSources(PROBE, HELD, 0, 0)
    rows, _ = dsurf.build_probe_rows(probe, dsurf.sft_pool_pairs(sft), bank_tasks=bank, out_root=out)
    assert len(sft) == 4 * len(SFT) and len(rows) == 4 * len(PROBE) + len(HELD) + len(PROBE) + len(HELD)
    return sft + rows


def _hashes(rows: list[dsurf.SurfacePairRow]) -> dict[str, tuple[str, str]]:
    return {Path(d).name: (task_hash(Path(d)), r.pool_pair) for r in rows for d in (r.twin_task_dir, r.control_task_dir)}


def _grow(pools: Path, n: int = 3) -> None:
    """Append `n` training and `n` held-out pairs (copies under new ids) to every pool."""
    for s in SURFACES:
        f = pools / f"{s}.json"
        raw = json.loads(f.read_text())
        for held in (True, False):
            tpl = next(p for p in raw["pairs"] if bool(p.get("heldout")) == held)
            raw["pairs"] += [{**tpl, "id": f"{tpl['id']}-new{k}-{held}", "heldout": held} for k in range(n)]
        f.write_text(json.dumps(raw))
    _cached_pool.cache_clear()  # `get_pool` caches by directory


def _live_root() -> None:
    """The live root: the rows of `_build` plus one unlisted leftover dir, and their surface_pairs.jsonl."""
    rows = _build(ROOT)
    dsurf.write_manifest(rows, PAIRS)
    pool = dsurf.get_pool("readme")
    pair = pool.pairs_for("train")[0].id
    assert dsurf.build_row("sft", LEFTOVER, "readme", pool, pair, bank_tasks=Path("data/tasks"), out_root=ROOT, split="train")


def _root_hashes() -> dict[str, str]:
    return {d.name: task_hash(d) for d in ROOT.iterdir()}


def test_recorded_probe_and_sft_builds_survive_pool_growth(work):
    """The closure test: record the root, grow every pool, rebuild in a temporary dir: every hash is unchanged."""
    _live_root()
    live = _root_hashes()
    assert pool_build.main(["record-decision"]) == 0
    manifest = json.loads(pool_build.decision_manifest_path().read_text())
    assert manifest["variants"] == live and len(live) == 2 * (len(_build(Path("tmp-a"))) + 1)
    assert manifest["sources"]["unlisted"] == {dsurf.twin_variant_id(LEFTOVER, "readme"): LEFTOVER}
    assert pool_build.main(["check-decision"]) == 0
    _grow(work)
    assert pool_build.main(["check-decision"]) == 0
    assert {k: h for k, (h, _) in _hashes(_build(Path("tmp-b"))).items()} == {
        k: v for k, v in live.items() if "leftover" not in k
    }  # the default build (`decision_surfaces build`) honours the pins too


def test_without_the_pins_growth_moves_probe_and_sft_variants(work):
    """The old behaviour (and the sensitivity of the test above): the round robin, the probe's pair rank and the forced
    path's `{context}` draw all follow the pool size."""
    _live_root()
    assert pool_build.main(["record-decision"]) == 0
    dsurf.pins_path().unlink()
    assert pool_build.main(["check-decision"]) == 1  # the leftover cannot be rebuilt without its pin
    before = _hashes(_build(Path("tmp-a")))
    _grow(work)
    after = _hashes(_build(Path("tmp-b")))
    assert sum(after[v] != before[v] for v in before) > 0


def test_a_new_source_draws_as_if_nothing_were_pinned(work):
    _live_root()
    free = _hashes(_build(Path("tmp-a")))
    assert pool_build.main(["record-decision"]) == 0
    assert _hashes(_build(Path("tmp-b"))) == free  # also: pinned sources do not change the unpinned sources' draw


def test_a_pinned_pair_cannot_leave_the_pool_or_cross_the_split(work):
    _live_root()
    assert pool_build.main(["record-decision"]) == 0
    pins = dsurf.load_pins()
    vid, pin = next((v, a) for v, a in pins.roles["sft"].items() if a.surface == "readme")
    source = next(s for s in SFT if dsurf.twin_variant_id(s, "readme") == vid)
    pool = dsurf.get_pool("readme")
    with pytest.raises(PoolError, match="split"):
        dsurf.build_row("probe", source, "readme", pool, pin.pair, bank_tasks=Path("data/tasks"), out_root=Path("x"),
                        split="test", pin=pin)  # fmt: skip
    f = work / "readme.json"
    raw = json.loads(f.read_text())
    raw["pairs"] = [p for p in raw["pairs"] if p["id"] != pin.pair]
    f.write_text(json.dumps(raw))
    with pytest.raises(PoolError, match="no longer in the readme pool"):
        dsurf.build_sft_rows([source], bank_tasks=Path("data/tasks"), out_root=Path("y"), surfaces=["readme"],
                             pools={"readme": pool_build.load_pool("readme", directory=work)})  # fmt: skip


def test_record_refuses_a_root_it_cannot_reproduce(work):
    _live_root()
    victim = next(d for d in ROOT.iterdir() if d.name.endswith("__control"))
    (victim / "instruction.md").write_text("edited by hand\n")
    assert pool_build.main(["record-decision"]) == 1
    assert not dsurf.pins_path().exists() and not pool_build.decision_manifest_path().exists()


@pytest.mark.skipif(
    not (REPO / "data" / "tasks").is_dir() or not (REPO / "data" / "decision" / "surfaces" / "tasks").is_dir(),
    reason="needs the local task bank (data/tasks) and the probe / SFT root",
)
def test_committed_decision_manifest_matches_a_rebuild(monkeypatch):
    """Sweep S7 at L0 for the probe / SFT root: it rebuilds byte-identical to `data/pools/decision_build_manifest.json`."""
    monkeypatch.chdir(REPO)
    assert pool_build.main(["check-decision"]) == 0
