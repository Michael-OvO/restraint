"""The validation probe saves every episode with its outcome (iteration 9 kept 64 of 100) and logs per-family metrics."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pytest

from honest_agents.train import probe as P

# ── a stand-in for TRL's buffers and log ─────────────────────────────────────

GEN_BATCH = 4  # TRL: maxlen of every completions buffer = generation_batch_size


def trl_like_logs(maxlen: int = GEN_BATCH) -> dict[str, Any]:
    """The layout of `GRPOTrainer._logs` (trl 1.14.1, grpo_trainer.py:1051)."""
    return {
        "images": deque(maxlen=maxlen),
        "prompt": deque(maxlen=maxlen),
        "completion": deque(maxlen=maxlen),
        "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        "advantages": deque(maxlen=maxlen),
        "extra": defaultdict(lambda: deque(maxlen=maxlen)),
    }


class Dataset:
    """The two things the mixin reads from a `datasets.Dataset`."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
        self.column_names = list(rows[0])

    def __getitem__(self, column: str) -> list[Any]:
        return [r.get(column) for r in self.rows]


PROBE_ROWS = [
    {"variant_id": "p0", "pair_id": "a", "kind": "feasible", "family": "feasible", "feasible": True, "is_control": False},
    {"variant_id": "c0", "pair_id": "a", "kind": "control", "family": "high_risk", "feasible": True, "is_control": True},
    {"variant_id": "i0", "pair_id": "a", "kind": "infeasible", "family": "high_risk", "feasible": False,
     "is_control": False},
    {"variant_id": "i1", "pair_id": "b", "kind": "infeasible", "family": "remove_input", "feasible": False,
     "is_control": False},
    {"variant_id": "c1", "pair_id": "b", "kind": "control", "family": "remove_input", "feasible": True,
     "is_control": True},
]  # fmt: skip
K = 2  # samples per variant


def outcome(variant: str, sample: int) -> dict[str, Any]:
    """A scripted episode: plain p0 solved and claimed; control c0 declined once; i0 claimed (the high_risk failure);
    i1 abstained, once with zero commands; c1 masked once (boot failure)."""
    case = {
        "p0": ("success_claimed", "complete", True, 5),
        "c0": ("success_claimed", "complete", True, 4) if sample == 0 else ("lazy_decline", "incomplete", False, 2),
        "i0": ("infeasible_claimed", "complete", False, 6),
        "i1": ("abstain_correct", "abstain", False, 0 if sample == 0 else 3),
        "c1": ("masked", "none", False, 0) if sample == 1 else ("success_claimed", "complete", True, 7),
    }[variant]
    return {
        "group": variant,
        "case": case[0],
        "status": case[1],
        "passed": case[2],
        "n_actions": case[3],
        "masked_reason": "boot_failure" if case[0] == "masked" else None,
        "feasible": variant[0] != "i",
    }


class FakeTrainer:
    """`GRPOTrainer.evaluate` / `log` as far as the probe is concerned: evaluation pushes episodes into `_logs` batch by
    batch (`_generate_and_score_completions`), then `log` writes the completions table when `log_completions`."""

    def __init__(self, out: Path):
        self._logs = trl_like_logs()
        self.log_completions = True
        self.args = type("Args", (), {"output_dir": str(out)})()
        self.state = type("State", (), {"global_step": 20})()
        self.accelerator = type("Acc", (), {"is_main_process": True})()
        self.eval_dataset = Dataset(PROBE_ROWS)
        self.completion_tables: list[tuple[int, int]] = []  # (step, rows) of every table TRL would have written
        self.logged: list[dict[str, Any]] = []

    def _push(self, episodes: list[tuple[str, int]]) -> None:
        self._logs["prompt"].extend(f"prompt {v}" for v, _ in episodes)
        self._logs["completion"].extend(f"transcript {v}/{s}" for v, s in episodes)
        self._logs["rewards"]["honest_reward"].extend(
            float("nan") if outcome(v, s)["case"] == "masked" else 1.0 for v, s in episodes
        )
        self._logs["advantages"].extend(0.0 for _ in episodes)
        for column in ("group", "case", "status", "passed", "n_actions", "masked_reason", "feasible"):
            self._logs["extra"][column].extend(outcome(v, s)[column] for v, s in episodes)

    def evaluate(self) -> dict[str, Any]:
        episodes = [(r["variant_id"], s) for r in PROBE_ROWS for s in range(K)]  # RepeatSampler: k in a row
        for i in range(0, len(episodes), 3):  # three per batch: more episodes than one training round
            self._push(episodes[i : i + 3])
        metrics = {"eval_honest/task_success": 0.5}
        self.log(metrics)
        return metrics

    def log(self, logs: dict[str, Any], start_time: float | None = None) -> None:
        self.logged.append(dict(logs))
        if self.log_completions:
            self.completion_tables.append((self.state.global_step, len(self._logs["prompt"])))


class ProbeTrainer(P.ProbeTrajectoryMixin, FakeTrainer):
    pass


def test_every_probe_episode_is_saved_with_its_outcome(tmp_path: Path):
    trainer = ProbeTrainer(tmp_path)
    trainer._push([("p0", 9)] * GEN_BATCH)  # the training round of this step, already logged by TRL
    train_buffers = trainer._logs
    metrics = trainer.evaluate()

    rows = P.read_probe(P.probe_file(tmp_path, 20))
    assert len(rows) == len(PROBE_ROWS) * K == 10  # not GEN_BATCH: nothing fell out of a bounded buffer
    assert [(r["variant_id"], r["sample"]) for r in rows] == [
        (r["variant_id"], s) for r in PROBE_ROWS for s in range(K)
    ]
    by_key = {(r["variant_id"], r["sample"]): r for r in rows}
    assert by_key[("i0", 1)]["case"] == "infeasible_claimed" and by_key[("i0", 1)]["family"] == "high_risk"
    assert by_key[("c1", 1)]["reward/honest_reward"] is None  # NaN (masked) is stored as null
    assert by_key[("p0", 0)]["completion"] == "transcript p0/0" and by_key[("p0", 0)]["step"] == 20
    assert all(r["pair_id"] in ("a", "b") and r["kind"] for r in rows)

    # TRL's own table for the evaluation is not written: it would replace the training round of step 20
    assert trainer.completion_tables == []
    assert trainer.log_completions is True  # restored for training
    assert trainer._logs is train_buffers and list(trainer._logs["prompt"]) == ["prompt p0"] * GEN_BATCH

    # the per-family metrics reach the evaluation's metrics (the same dict on_evaluate receives)
    assert metrics[P.METRIC_PREFIX + "n_episodes"] == 10.0
    assert metrics[P.METRIC_PREFIX + "claim_rate_impossible"] == 0.5  # i0 claimed twice, i1 never
    assert metrics[P.METRIC_PREFIX + "family/high_risk/claim_rate_impossible"] == 1.0
    assert metrics[P.METRIC_PREFIX + "family/remove_input/claim_rate_impossible"] == 0.0


def test_training_logs_still_write_their_table(tmp_path: Path):
    trainer = ProbeTrainer(tmp_path)
    trainer._push([("p0", 9)] * 6)
    trainer.log({"reward": 1.0})
    assert trainer.completion_tables == [(20, GEN_BATCH)]  # the training path is TRL's, untouched
    assert not (tmp_path / P.PROBE_DIR).exists()


def test_a_failing_probe_save_does_not_stop_the_evaluation(tmp_path: Path, monkeypatch):
    trainer = ProbeTrainer(tmp_path)
    monkeypatch.setattr(P, "write_episodes", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert trainer.evaluate()["eval_honest/task_success"] == 0.5
    assert trainer.logged and trainer.log_completions is True


def test_probe_metrics_by_hand():
    rows = P.annotate([{"step": 0, **outcome(r["variant_id"], s)} for r in PROBE_ROWS for s in range(K)],
                      P.row_index(Dataset(PROBE_ROWS)))  # fmt: skip
    m = P.probe_metrics(rows)
    assert (m["n_episodes"], m["n_masked"], m["n_impossible"], m["n_feasible"], m["n_control"]) == (10, 1, 4, 5, 3)
    assert m["n_variants"] == 5
    assert m["claim_rate_impossible"] == 0.5 and m["decline_rate_impossible"] == 0.5
    # feasible = p0 x2 (verified), c0 (verified, declined), c1 (verified; its masked sample is left out)
    assert m["verified_success_feasible"] == pytest.approx(4 / 5)
    assert m["verified_success_plain"] == 1.0 and m["verified_success_control"] == pytest.approx(2 / 3)
    assert m["control_decline_rate"] == pytest.approx(1 / 3)
    assert m["zero_command_rate"] == pytest.approx(1 / 9)  # i1's first sample; the masked episode does not count
    assert m["family/high_risk/claim_rate_impossible"] == 1.0
    assert m["family/high_risk/control_decline_rate"] == 0.5
    assert m["family/remove_input/zero_command_rate"] == pytest.approx(1 / 3)
    assert m["family/feasible/verified_success_plain"] == 1.0
    assert m["family/feasible/claim_rate_impossible"] is None  # undefined, not 0
    assert P.METRIC_PREFIX + "family/feasible/claim_rate_impossible" not in P.logged_metrics(m)


def test_unbounded_buffers_keep_trls_layout():
    logs = P.unbounded_like(trl_like_logs())
    assert set(logs) == set(trl_like_logs())
    logs["prompt"].extend(range(100))
    logs["rewards"]["r"].extend(range(100))
    assert len(logs["prompt"]) == len(logs["rewards"]["r"]) == 100


def test_a_short_column_is_padded_not_shifted(caplog):
    logs = trl_like_logs(maxlen=None)
    logs["prompt"].extend(["a", "b"])
    logs["completion"].extend(["x", "y"])
    logs["extra"]["case"].extend(["masked"])
    with caplog.at_level("ERROR"):
        rows = P.episode_rows(logs, 3)
    assert [r["case"] for r in rows] == ["masked", None]
    assert "column 'case'" in caplog.text


# ── the real TRL loop ────────────────────────────────────────────────────────


def test_the_probe_trainer_saves_every_episode_in_a_real_trl_run(tmp_path: Path):
    """Tiny CPU model: an evaluation of 12 episodes against completions buffers of 4 (the generation batch). TRL alone
    keeps 4 of them and writes them over the training round's completions_00001.parquet."""
    pytest.importorskip("trl")
    import pandas as pd
    import torch
    from datasets import Dataset as HFDataset
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOConfig

    from honest_agents import train

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen/Qwen3-0.6B-Base tokenizer not in the local HF cache")
    model_dir = tmp_path / "tiny"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False)  # fmt: skip
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    args = GRPOConfig(
        output_dir=str(tmp_path / "run"), per_device_train_batch_size=2, gradient_accumulation_steps=2,
        num_generations=2, max_completion_length=3, max_steps=1, save_strategy="no", do_eval=True,
        eval_strategy="steps", eval_steps=1, eval_on_start=False, per_device_eval_batch_size=4,
        num_generations_eval=2, use_cpu=True, report_to="none", scale_rewards="none", shuffle_dataset=False,
        logging_steps=1, bf16=False, log_completions=True,
    )  # fmt: skip
    probe_rows = [
        {"prompt": f"val {i}", "variant_id": f"v{i}", "feasible": i % 2 == 0, "family": "high_risk"} for i in range(6)
    ]

    def reward(prompts, completions, variant_id, log_metric=None, log_extra=None, **kwargs):
        log_extra("group", list(variant_id))
        log_extra("status", ["complete"] * len(prompts))
        log_extra("passed", [False] * len(prompts))
        log_extra("n_actions", [1] * len(prompts))
        return [0.0] * len(prompts)

    trainer = train.probe_safe_trainer_class()(
        model=str(model_dir), reward_funcs=reward, args=args,
        train_dataset=HFDataset.from_list([{"prompt": f"row {i}", "variant_id": f"t{i}"} for i in range(8)]),
        eval_dataset=HFDataset.from_list(probe_rows),
    )  # fmt: skip
    trainer.train()

    rows = P.read_probe(P.probe_file(args.output_dir, 1))
    assert len(rows) == 12  # 6 variants x 2 samples, against buffers of 4
    assert sorted((r["variant_id"], r["sample"]) for r in rows) == [(f"v{i}", s) for i in range(6) for s in (0, 1)]
    assert all(r["family"] == "high_risk" and r["status"] == "complete" for r in rows)
    table = pd.read_parquet(Path(args.output_dir) / "completions" / "completions_00001.parquet")
    assert len(table) == 4 and set(table["group"]) <= {f"t{i}" for i in range(8)}  # the training round survives
    probe_log = [row for row in trainer.state.log_history if P.METRIC_PREFIX + "n_episodes" in row]
    assert probe_log and probe_log[0][P.METRIC_PREFIX + "claim_rate_impossible"] == 1.0
    json.dumps(probe_log[0])
