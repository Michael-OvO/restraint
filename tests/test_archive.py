from __future__ import annotations

import gzip
import json
import os
import shutil

import pytest

from honest_agents import archive
from honest_agents.archive import (
    Source,
    check,
    duplicate_groups,
    gzip_large_text,
    hash_archive,
    hash_episodes,
    main,
    main_check,
    save,
    select_eval_files,
    select_run_files,
)

RUN = [
    "README.md", "run_config.json", "train_log.jsonl", "validation_probe.jsonl", "best_checkpoint.json",
    "completions/completions_00001.parquet",
    "checkpoint-10/adapter_model.safetensors", "checkpoint-10/optimizer.pt", "checkpoint-10/adapter_config.json",
    "checkpoint-50/adapter_model.safetensors", "checkpoint-50/optimizer.pt", "checkpoint-50/adapter_config.json",
    "final/adapter_model.safetensors", "final/adapter_config.json", "final/tokenizer.json", "final/README.md",
]  # fmt: skip


def test_a_run_keeps_logs_completions_and_final_adapter_never_optimizer_states():
    keep = select_run_files(RUN, best_checkpoint="outputs/x/checkpoint-50")
    assert "train_log.jsonl" in keep and "completions/completions_00001.parquet" in keep
    assert "final/adapter_model.safetensors" in keep and "final/tokenizer.json" not in keep
    assert not any("optimizer" in p for p in keep)
    assert not any(p.startswith("checkpoint-") for p in keep)  # best == last checkpoint == final: no duplicate


def test_an_earlier_best_checkpoint_keeps_its_adapter_and_no_weights_keeps_none():
    keep = select_run_files(RUN, best_checkpoint="outputs/x/checkpoint-10")
    assert "checkpoint-10/adapter_model.safetensors" in keep and "checkpoint-10/optimizer.pt" not in keep
    light = select_run_files(RUN, best_checkpoint="outputs/x/checkpoint-10", weights=False)
    assert not any(p.endswith(".safetensors") for p in light) and "train_log.jsonl" in light


def test_an_evaluation_keeps_results_not_transcripts():
    paths = ["summary.md", "metrics.json", "config.json", "episodes.jsonl", "transcripts/a.json", "bank/bank-1.jsonl"]
    assert select_eval_files(paths) == ["config.json", "episodes.jsonl", "metrics.json", "summary.md"]
    assert "bank/bank-1.jsonl" in select_eval_files(paths, keep_bank=True)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_save_from_a_local_mirror_writes_a_small_self_describing_folder(tmp_path):
    src = tmp_path / "mirror"
    run = src / "outputs" / "grpo" / "r1"
    for rel in RUN:
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * (2_000_000 if rel == "train_log.jsonl" else 10))
    (run / "best_checkpoint.json").write_text(json.dumps({"checkpoint": "outputs/grpo/r1/checkpoint-10"}))
    ev = src / "runs" / "eval" / "val"
    (ev / "transcripts").mkdir(parents=True)
    for rel in ("summary.md", "metrics.json", "config.json", "episodes.jsonl", "transcripts/t.json"):
        (ev / rel).write_text("{}")
    folder = save(Source(str(src)), tmp_path / "iterations", 9, "grpo-run2", runs=["outputs/grpo/r1"],
                  evals=["runs/eval/val"], note="test", hash_cache=None)  # fmt: skip
    assert folder.name == "09-grpo-run2"
    kept = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    assert "run/r1/train_log.jsonl.gz" in kept  # over 1 MB: gzipped
    assert gzip.decompress((folder / "run/r1/train_log.jsonl.gz").read_bytes()) == b"x" * 2_000_000
    assert "run/r1/checkpoint-10/adapter_model.safetensors" in kept and "run/r1/final/adapter_model.safetensors" in kept
    assert not any("optimizer" in k or "tokenizer" in k or "transcripts" in k for k in kept)
    assert {"eval/val/summary.md", "ITERATION.md", "MANIFEST.sha256"} <= kept
    assert "Iteration 09: grpo-run2" in (folder / "ITERATION.md").read_text()


# ── content hashes and duplicate episode sets (DEF-12, sweep S8) ─────────────


def _episodes(ids: list[str]) -> bytes:
    return b"".join(json.dumps({"episode_id": i, "reward": 1.0}).encode() + b"\n" for i in ids)


def _write_eval(root, rel: str, data: bytes, mtime: int = 0):
    p = root / rel / "episodes.jsonl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(data, mtime=mtime))
    return p


def test_a_copy_regzipped_with_another_header_is_still_a_duplicate(tmp_path):
    data = _episodes(["t1__s0__aaaaaa", "t2__s0__bbbbbb"])
    a = hash_episodes(_write_eval(tmp_path, "15-a/eval/x", data, mtime=1), tmp_path)
    b = hash_episodes(_write_eval(tmp_path, "15-b/eval/x", data, mtime=2), tmp_path)
    assert a.file_sha256 != b.file_sha256  # the archived it15 pair: gz bytes differ, content does not
    assert (a.content_sha256, a.ids_sha256, a.n_episodes) == (b.content_sha256, b.ids_sha256, 2)
    assert [sorted(h.path for h in g) for g in duplicate_groups([a, b])] == [[a.path, b.path]]


def test_a_reordered_copy_shares_its_sorted_ids_and_two_runs_share_nothing(tmp_path):
    a = hash_episodes(_write_eval(tmp_path, "a/eval/x", _episodes(["e1", "e2"])), tmp_path)
    b = hash_episodes(_write_eval(tmp_path, "b/eval/x", _episodes(["e2", "e1"])), tmp_path)
    c = hash_episodes(_write_eval(tmp_path, "c/eval/x", _episodes(["e3", "e4"])), tmp_path)
    assert a.content_sha256 != b.content_sha256 and a.ids_sha256 == b.ids_sha256
    assert [sorted(h.path for h in g) for g in duplicate_groups([a, b, c])] == [[a.path, b.path]]


def test_check_fails_on_an_undeclared_copy_and_passes_once_it_is_declared(tmp_path, capsys):
    data = _episodes(["e1", "e2", "e3"])
    _write_eval(tmp_path, "15-g0-judgment/eval/sft", data, mtime=1)
    _write_eval(tmp_path, "15-g2-splice-replay/eval/sft", data, mtime=2)
    _write_eval(tmp_path, "15-g2-splice-replay/eval/g2", _episodes(["f1"]))
    cache = tmp_path / "cache.jsonl"
    assert check(tmp_path, cache) == 1
    assert main_check(["--dest", str(tmp_path), "--cache", str(cache)]) == 1
    assert "DUPLICATE (undeclared)" in capsys.readouterr().out
    # a copy declared the other way round, or of a path outside the group, does not count
    (tmp_path / "15-g2-splice-replay" / "ITERATION.md").write_text(
        "- duplicate: `eval/sft` is a copy of `15-g2-splice-replay/eval/g2`\n"
    )
    assert check(tmp_path, cache) == 1
    (tmp_path / "15-g2-splice-replay" / "ITERATION.md").write_text(
        "- duplicate: `eval/sft` is a copy of `15-g0-judgment/eval/sft` (DEF-12)\n"
    )
    assert check(tmp_path, cache) == 0
    assert "duplicate (declared copy)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_:
        main(["check", "--dest", str(tmp_path), "--no-cache"])
    assert exit_.value.code == 0


def test_the_hash_cache_skips_unchanged_files_and_rehashes_changed_ones(tmp_path, monkeypatch):
    root, cache = tmp_path / "it", tmp_path / "cache.jsonl"
    p = _write_eval(root, "a/eval/x", _episodes(["e1"]))
    first = hash_archive(root, cache)
    monkeypatch.setattr(archive, "hash_episodes", lambda *a: pytest.fail("re-hashed an unchanged file"))
    assert hash_archive(root, cache) == first
    monkeypatch.undo()
    p.write_bytes(gzip.compress(_episodes(["e1", "e2"])))
    os.utime(p, ns=(1, 1))
    assert [h.n_episodes for h in hash_archive(root, cache)] == [2]
    cache.write_text(cache.read_text() + "not json\n" + json.dumps({"schema": 0, "path": "a/eval/x/episodes.jsonl.gz"}))
    assert [h.n_episodes for h in hash_archive(root, cache)] == [2]  # corrupt and old-schema rows are ignored


def test_gzip_large_text_is_deterministic(tmp_path):
    outs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "episodes.jsonl").write_bytes(_episodes([f"e{i}" for i in range(50_000)]))
        os.utime(tmp_path / name / "episodes.jsonl", (100 + len(outs), 100 + len(outs)))
        gzip_large_text(tmp_path / name)
        outs.append((tmp_path / name / "episodes.jsonl.gz").read_bytes())
    assert outs[0] == outs[1]


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_save_records_episode_hashes_and_warns_when_an_eval_is_already_archived(tmp_path, capsys):
    src = tmp_path / "mirror"
    ev = src / "runs" / "eval" / "sft_v2_surface4"
    ev.mkdir(parents=True)
    (ev / "episodes.jsonl").write_bytes(_episodes(["e1", "e2"]))
    dest, cache = tmp_path / "iterations", tmp_path / "cache.jsonl"
    first = save(Source(str(src)), dest, 15, "g0-judgment", evals=["runs/eval/sft_v2_surface4"], hash_cache=cache)
    assert "WARNING" not in capsys.readouterr().err
    text = (first / "ITERATION.md").read_text()
    assert '"n_episodes": 2' in text and '"episode_ids_sha256"' in text and "duplicate" not in text
    second = save(Source(str(src)), dest, 15, "g2-splice-replay", evals=["runs/eval/sft_v2_surface4"], hash_cache=cache)
    assert "15-g0-judgment/eval/sft_v2_surface4" in capsys.readouterr().err
    assert (
        "- duplicate: `15-g2-splice-replay/eval/sft_v2_surface4` has the same episodes as"
        in (second / "ITERATION.md").read_text()
    )
    assert check(dest, cache) == 1
