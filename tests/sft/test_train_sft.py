"""Config, dry run and a tiny CPU fit of the SFT entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.sft import train_sft
from tests.sft.helpers import bank_dir, make_row, tiny_tokenizer

ROOT = Path(__file__).resolve().parents[2]
NOT_FOUND = [("jq . in.json", "bash: line 1: jq: command not found")]


def synthetic_bank(tmp_path: Path, variants: int = 24) -> Path:
    rows = []
    for i in range(variants):
        for j in range(2):
            passed = (i + j) % 2 == 0
            rows.append(
                make_row(
                    f"v{i}",
                    passed=passed,
                    original={"status": "complete" if (i + j) % 3 else "incomplete"},
                    turns=[("ls", "in.txt")] if passed else NOT_FOUND,
                )
            )
    return bank_dir(tmp_path, rows)


def tiny_config(tmp_path: Path, **over: str) -> train_sft.SftConfig:
    overrides = [
        f"data.bank={synthetic_bank(tmp_path)}",
        f"output_root={tmp_path / 'out'}",
        "model.name=tiny",
        "model.family=qwen3",
        "model.dtype=float32",
        "data.min_examples=2",
        "data.holdout_fraction=0.25",
        "data.max_length=4096",
        "lora.r=4",
        "lora.alpha=8",
        "train.bf16=false",
        "train.gradient_checkpointing=false",
        "train.gradient_accumulation_steps=1",
        "train.per_device_train_batch_size=2",
        "train.per_device_eval_batch_size=2",
        "train.max_steps=3",
        "train.eval_steps=2",
        "train.save_steps=100",
        "train.logging_steps=1",
        "train.extra={use_cpu: true}",
        *(f"{k}={v}" for k, v in over.items()),
    ]
    return train_sft.load_config(ROOT / "configs" / "sft_hindsight.yaml", overrides)


def test_shipped_configs_parse():
    cfg = train_sft.load_config(ROOT / "configs" / "sft_hindsight.yaml")
    assert cfg.kind == "hindsight" and cfg.train.num_train_epochs == 1 and cfg.lora.r == 16
    raw = train_sft.apply_overrides(train_sft.load_config_dict(ROOT / "configs" / "sft_bestofn.yaml"), [])
    best = train_sft.parse_config(raw, check_kind=False)
    assert best.kind == "bestofn" and best.data.max_length > cfg.data.max_length and best.bestofn["top_k"] == 1


def test_unknown_keys_and_kinds_are_refused():
    with pytest.raises(ValueError, match="unknown key"):
        train_sft.parse_config({"train": {"learning_rat": 1e-5}})
    with pytest.raises(ValueError, match="top-level"):
        train_sft.parse_config({"epochs": 3})
    with pytest.raises(ValueError, match="kind"):
        train_sft.parse_config({"kind": "nope"})
    with pytest.raises(ValueError, match="confidence"):
        train_sft.parse_config({"hindsight": {"confidence": "maybe"}})


def test_dry_run_builds_examples_without_a_model(tmp_path):
    cfg = tiny_config(tmp_path)
    summary = train_sft.dry_run(cfg, tokenizer=tiny_tokenizer())
    assert summary["examples_fit"] > 0 and summary["examples_heldout"] > 0
    assert summary["tokenized_fit"] == summary["examples_fit"] and summary["mean_trained_tokens"] < 400
    assert set(summary["labels"]["status"]) <= {"complete", "incomplete", "abstain"}
    assert summary["labels"]["blocker_category"].get("missing_capability", 0) > 0
    assert 0.0 <= summary["source_reward_spread"]["fraction"] <= 1.0
    lines = (cfg.output_dir / "examples.jsonl").read_text().splitlines()
    assert len(lines) == summary["examples_fit"] + summary["examples_heldout"]
    assert json.loads(lines[0])["loss_messages"]


def test_dry_run_without_a_cached_tokenizer_still_reports(tmp_path):
    cfg = tiny_config(tmp_path, **{"model.local_files_only": "true"})
    summary = train_sft.dry_run(cfg)
    assert str(summary["tokenization"]).startswith("skipped") and summary["examples_fit"] > 0


def test_max_rows_and_heldout_variants_are_respected(tmp_path):
    stage, _ = train_sft.build_stage_data(tiny_config(tmp_path, **{"data.max_rows": "6"}))
    assert len(stage.train) == 6
    assert {e.variant for e in stage.train}.isdisjoint({e.variant for e in stage.eval})


def test_spread_report(tmp_path):
    report = train_sft.spread_report(str(synthetic_bank(tmp_path)))
    assert report["n_groups"] == 24 and 0.0 <= report["fraction"] <= 1.0


def test_cli_dry_run_and_spread(tmp_path, capsys):
    cfg_bank = synthetic_bank(tmp_path)
    assert train_sft.main(["--spread", str(cfg_bank)]) == 0
    assert json.loads(capsys.readouterr().out)["n_groups"] == 24


def test_tiny_cpu_fit_trains_only_report_tokens_and_saves_an_adapter(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = tiny_tokenizer()
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8192,
        )
    )
    cfg = tiny_config(tmp_path)
    summary = train_sft.run_sft(cfg, model=model, tokenizer=tok)
    assert summary["train_loss"] > 0
    assert (cfg.output_dir / "final" / "adapter_config.json").exists()
    assert (cfg.output_dir / "sft_run.json").exists()
    evals = [h for h in summary["log_history"] if "eval_entropy" in h]
    assert evals and all(h["eval_entropy"] >= 0 for h in evals)


def test_entropy_guard_stops_only_when_entropy_falls_below_the_ratio():
    from types import SimpleNamespace

    guard = train_sft._entropy_guard(0.5)
    control = SimpleNamespace(should_training_stop=False)
    guard.on_evaluate(None, None, control, metrics={"eval_entropy": 2.0})  # the start value
    guard.on_evaluate(None, None, control, metrics={"eval_entropy": 1.2})
    guard.on_evaluate(None, None, control, metrics={"eval_loss": 0.3})  # no entropy: ignored
    assert not control.should_training_stop
    guard.on_evaluate(None, None, control, metrics={"eval_entropy": 0.9})
    assert control.should_training_stop
