from __future__ import annotations

import gzip
import json
import shutil

import pytest

from honest_agents.archive import Source, save, select_eval_files, select_run_files

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
                  evals=["runs/eval/val"], note="test")  # fmt: skip
    assert folder.name == "09-grpo-run2"
    kept = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    assert "run/r1/train_log.jsonl.gz" in kept  # over 1 MB: gzipped
    assert gzip.decompress((folder / "run/r1/train_log.jsonl.gz").read_bytes()) == b"x" * 2_000_000
    assert "run/r1/checkpoint-10/adapter_model.safetensors" in kept and "run/r1/final/adapter_model.safetensors" in kept
    assert not any("optimizer" in k or "tokenizer" in k or "transcripts" in k for k in kept)
    assert {"eval/val/summary.md", "ITERATION.md", "MANIFEST.sha256"} <= kept
    assert "Iteration 09: grpo-run2" in (folder / "ITERATION.md").read_text()
