"""`sft.commit_build`: the merge of point builds (whole pairs trimmed to the source share, coverage --strict) and the
training stage's loading checks. No GPU, no Docker."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from honest_agents.commit_points.schema import (
    CLAIM,
    DESTRUCTIVE,
    FAIL,
    PASS,
    BuildManifest,
    ExampleRef,
    Quota,
    write_manifest,
)
from honest_agents.sft import commit_build as cb
from honest_agents.sft import commit_checks as cc
from honest_agents.sft.data import SFTExample

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]


def _example(point: str, outcome: str, source: str, k: int) -> tuple[SFTExample, ExampleRef]:
    task = f"{source.replace('/', '__')}__high_risk__readme__control" + ("" if outcome == PASS else "_t")
    key = "source_episode" if point == DESTRUCTIVE else "base_episode"
    ex_id = f"{point}:{source}:{k}:{outcome}"
    meta = {"source": source, "task": task, "point": point, "outcome": outcome, key: f"{source}-ep{k}"}
    hazard = "copy" if point == DESTRUCTIVE else None
    ref = ExampleRef(ex_id, task, source, point, outcome, outcome == PASS, "o", split="train", hazard=hazard)
    return SFTExample(ex_id, cc.KIND, task, MESSAGES, [2], meta), ref


def _write_build(d: Path, point: str, sources: dict[str, int]) -> Path:
    """`sources`: source -> number of matched pairs (pass + fail on one prefix episode each)."""
    pairs = [_example(point, o, s, k) for s, n in sources.items() for k in range(n) for o in (PASS, FAIL)]
    d.mkdir(parents=True)
    (d / cb.EXAMPLES_FILE).write_text("".join(json.dumps(dataclasses.asdict(e)) + "\n" for e, _ in pairs))
    write_manifest(BuildManifest.of([r for _, r in pairs]), d / cc.MANIFEST_FILE)
    return d


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    (d / "decision").mkdir(parents=True)
    (d / "select_sources.txt").write_text("sel/task_x\n")
    (d / "split_seeds.json").write_text(json.dumps({"val/task_y": "val"}))
    (d / "decision" / "probe_prompts_surfaces.jsonl").write_text(json.dumps({"pair_id": "probe/task_p"}) + "\n")
    return d


def test_merge_trims_whole_pairs_to_the_source_share_and_passes_coverage(tmp_path: Path, data_dir: Path):
    small = {f"src/task_{i:03d}": 1 for i in range(30)}
    claims = _write_build(tmp_path / "claims", CLAIM, {**small, "src/big": 6})
    destructive = _write_build(tmp_path / "destructive", DESTRUCTIVE, {**small, "src/big": 6})
    quota = Quota(cap_per_task=20, max_outcome_ratio=1.5, min_per_cell=10, max_source_share=0.05,
                  max_task_share=0.2, points=(CLAIM, DESTRUCTIVE))  # fmt: skip
    summary = cb.merge([claims, destructive], tmp_path / "out", quota, data_dir)
    kept = cb.read_examples(tmp_path / "out" / cb.EXAMPLES_FILE)
    big = [e for e in kept if e.meta["source"] == "src/big"]
    assert len(big) <= 0.05 * len(kept) and summary["dropped"]["source_share"] == 24 - len(big)
    assert len(big) % 2 == 0 and {e.meta["outcome"] for e in big} == {PASS, FAIL}  # whole pairs only
    assert summary["coverage"]["failures"] == {}
    manifest = cb.PointBuild.read(tmp_path / "out")  # the manifest lists exactly the kept examples
    assert len(manifest.examples) == len(kept)
    assert manifest.manifest.build["inputs"][0]["examples_sha256"] == cb.sha256_file(claims / cb.EXAMPLES_FILE)


def test_merge_refuses_a_build_that_fails_coverage_or_holds_an_excluded_source(tmp_path: Path, data_dir: Path):
    claims = _write_build(tmp_path / "claims", CLAIM, {f"src/task_{i:03d}": 1 for i in range(5)})
    with pytest.raises(ValueError, match="under_quota"):
        cb.merge([claims], tmp_path / "out", Quota(min_per_cell=10, max_source_share=1.0, points=(CLAIM,)), data_dir)
    probe = _write_build(tmp_path / "probe", DESTRUCTIVE, {"probe/task_p": 1})
    with pytest.raises(cc.ExcludedSourceError):
        cb.merge([probe], tmp_path / "out2", Quota(min_per_cell=1, max_source_share=1.0), data_dir)


def test_the_training_stage_loads_only_the_frozen_build(tmp_path: Path, data_dir: Path):
    claims = _write_build(tmp_path / "claims", CLAIM, {f"src/task_{i:03d}": 1 for i in range(3)})
    digest = cb.sha256_file(claims / cc.MANIFEST_FILE)
    opts = cb.CommitBuildOptions(str(claims), digest, str(data_dir))
    assert len(cb.load_build(opts)) == 6
    with pytest.raises(ValueError, match="not the frozen"):
        cb.load_build(dataclasses.replace(opts, manifest_sha256="0" * 64))
    with pytest.raises(ValueError, match="unknown key"):
        cb.parse_options({"build": "x", "bild": "y"})
    lines = (claims / cb.EXAMPLES_FILE).read_text().splitlines()
    (claims / cb.EXAMPLES_FILE).write_text("\n".join(lines[:-1]) + "\n")  # an example the manifest lists is gone
    with pytest.raises(ValueError, match="the manifest lists"):
        cb.load_build(dataclasses.replace(opts, manifest_sha256=None))
