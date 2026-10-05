"""Stop rules, the validation probe and the Phase-0 summary (SPEC_V3 5.6, 9.2.2, 9.3.1, 9.3.4). No GPU, no Docker.

The rules are tested on synthetic log dictionaries carrying the metric names `trl_rewards.py` logs, and the probe
wiring on a real (CPU, tiny random model) TRL run, so the `eval_` prefixing and the callback order are TRL's own.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import TrainerControl, TrainerState

from honest_agents import callbacks, trl_rewards
from honest_agents.callbacks import (
    StopRuleConfig,
    StopRules,
    ValidationProbe,
    is_train_log,
    phase0_summary,
    pool_stats,
)

CLAIM = trl_rewards.METRIC_CLAIM_RATE_FEASIBLE
SUCCESS = trl_rewards.METRIC_TASK_SUCCESS
ZERO_VAR = trl_rewards.METRIC_ZERO_VARIANCE_GROUPS
DECLINE = trl_rewards.METRIC_DECLINE_RATE_FEASIBLE
MASKED = trl_rewards.METRIC_MASKED_SHARE
REPORT = trl_rewards.METRIC_REPORT_RATE
W = 4  # window used by the synthetic tests


def healthy(**over: float) -> dict[str, float]:
    logs = {CLAIM: 0.4, SUCCESS: 0.5, ZERO_VAR: 0.2, DECLINE: 0.2, MASKED: 0.0, REPORT: 0.9, "reward": 0.3}
    logs.update(over)
    return logs


def rules(**cfg) -> StopRules:
    return StopRules(StopRuleConfig(window=W, **cfg), reward_mode="honest")


def feed(stop: StopRules, logs: list[dict], start: int = 1) -> dict[int, list[str]]:
    """Observe a series of logs; returns {step: [rules fired]} for the steps where something fired."""
    fired: dict[int, list[str]] = {}
    for i, log in enumerate(logs):
        triggers = stop.observe(start + i, log)
        if triggers:
            fired[start + i] = [t.rule for t in triggers]
    return fired


# ── metric names ─────────────────────────────────────────────────────────────


def test_rules_read_the_metric_names_trl_rewards_logs():
    """Never invent metric names: every tracked series is a constant `trl_rewards` exports and logs."""
    tracked = {name for name in StopRules()._tracked if name != callbacks.TRL_REWARD}
    assert tracked <= set(trl_rewards.REQUIRED_METRICS) | {trl_rewards.METRIC_NO_CONTRAST}
    assert all(name.startswith("honest/") for name in tracked)


def test_is_train_log_separates_training_evaluation_and_summaries():
    assert is_train_log({"reward": 0.1, "loss": 0.0})
    assert is_train_log({"honest/task_success": 0.5})
    assert not is_train_log({"eval_reward": 0.1, "eval_honest/task_success": 0.5})
    assert not is_train_log({"train_runtime": 12.0, "train_loss": 0.0})


# ── threshold rules ──────────────────────────────────────────────────────────


def test_claim_rate_below_five_percent_fires_after_a_full_window():
    stop = rules()
    fired = feed(stop, [healthy(**{CLAIM: 0.02})] * 6)
    assert min(fired) == W  # not before a window of evidence
    assert fired[W] == [callbacks.RULE_CLAIM_RATE]


def test_claim_rate_at_the_threshold_does_not_fire():
    assert feed(rules(), [healthy(**{CLAIM: 0.05})] * 8) == {}
    assert feed(rules(), [healthy(**{CLAIM: 0.06})] * 8) == {}


def test_zero_variance_share_above_half_fires():
    fired = feed(rules(), [healthy(**{ZERO_VAR: 0.7})] * 5)
    assert fired[W] == [callbacks.RULE_ZERO_VARIANCE]
    assert feed(rules(), [healthy(**{ZERO_VAR: 0.5})] * 8) == {}


def test_decline_rate_ceiling_fires():
    fired = feed(rules(max_decline_rate_feasible=0.6), [healthy(**{DECLINE: 0.8})] * 5)
    assert fired[W] == [callbacks.RULE_DECLINE_RATE]
    assert feed(rules(max_decline_rate_feasible=0.9), [healthy(**{DECLINE: 0.8})] * 8) == {}


def test_masked_share_above_fifteen_percent_fires():
    fired = feed(rules(), [healthy(**{MASKED: 0.2})] * 5)
    assert fired[W] == [callbacks.RULE_MASKED_SHARE]
    assert feed(rules(), [healthy(**{MASKED: 0.15})] * 8) == {}


def test_a_single_bad_step_is_averaged_out():
    logs = [healthy()] * 3 + [healthy(**{CLAIM: 0.0})] + [healthy()] * 4
    assert feed(rules(), logs) == {}


def test_a_metric_that_was_not_logged_is_skipped_not_zero():
    """A ratio with an empty denominator is not logged (no feasible episode): it must not read as 0 and fire."""
    logs = [{k: v for k, v in healthy().items() if k != CLAIM}] * 8
    assert feed(rules(), logs) == {}
    # half the window missing is still enough evidence from the other half
    mixed = [healthy(**{CLAIM: 0.0}), {k: v for k, v in healthy().items() if k != CLAIM}] * 4
    assert callbacks.RULE_CLAIM_RATE in feed(rules(), mixed)[W]
    # None (TRL logs None for a metric with no valid batch) and NaN are missing values too
    assert feed(rules(), [healthy(**{CLAIM: None})] * 8) == {}
    assert feed(rules(), [healthy(**{CLAIM: float("nan")})] * 8) == {}


# ── "falling while reward rises" rules ───────────────────────────────────────


def _rise_then_collapse(success_after: float, reward_after: float, n_after: int = 2 * W) -> list[dict]:
    before = [healthy(**{SUCCESS: 0.6, "reward": 0.2})] * (2 * W)
    after = [healthy(**{SUCCESS: success_after, "reward": reward_after})] * n_after
    return before + after


def test_task_success_falling_while_reward_rises_fires():
    """The signature of reward hacking by quitting: training reward up, success down (arXiv 2608.00301)."""
    fired = feed(rules(), _rise_then_collapse(success_after=0.3, reward_after=0.5))
    assert any(callbacks.RULE_SUCCESS_VS_REWARD in rules_ for rules_ in fired.values())


def test_task_success_falling_with_reward_falling_does_not_fire():
    fired = feed(rules(), _rise_then_collapse(success_after=0.3, reward_after=0.1))
    assert not any(callbacks.RULE_SUCCESS_VS_REWARD in rules_ for rules_ in fired.values())


def test_small_success_drop_does_not_fire():
    fired = feed(rules(), _rise_then_collapse(success_after=0.55, reward_after=0.5))
    assert not any(callbacks.RULE_SUCCESS_VS_REWARD in rules_ for rules_ in fired.values())


def test_share_of_episodes_ending_with_a_report_falling_fires():
    """9.2.2: the multi-harness guide saw it fall from 69% to 41% while training reward looked healthy."""
    logs = [healthy(**{REPORT: 0.69})] * (2 * W) + [healthy(**{REPORT: 0.41})] * (2 * W)
    fired = feed(rules(), logs)
    assert any(callbacks.RULE_REPORT_RATE in rules_ for rules_ in fired.values())
    assert feed(rules(), [healthy(**{REPORT: 0.69})] * (2 * W) + [healthy(**{REPORT: 0.62})] * (2 * W)) == {}


def test_gradual_report_rate_decay_is_caught_by_comparing_with_the_best_window():
    logs = [healthy(**{REPORT: 0.9 - 0.02 * i}) for i in range(40)]  # 0.90 -> 0.12, no single step jumps
    fired = feed(rules(), logs)
    assert any(callbacks.RULE_REPORT_RATE in rules_ for rules_ in fired.values())


# ── actions, files, resume ───────────────────────────────────────────────────


def _handler_inputs(step: int):
    return SimpleNamespace(), TrainerState(global_step=step), TrainerControl()


def test_stop_action_stops_training_and_records_the_rule_once(tmp_path: Path):
    stop = StopRules(StopRuleConfig(window=W), "honest", output_dir=tmp_path)
    for step in range(1, 9):
        args, state, control = _handler_inputs(step)
        stop.on_log(args, state, control, logs=healthy(**{CLAIM: 0.0}))
    assert control.should_training_stop is True
    lines = [json.loads(line) for line in (tmp_path / callbacks.STOP_RULES_FILE).read_text().splitlines()]
    assert [line["rule"] for line in lines] == [callbacks.RULE_CLAIM_RATE]  # first firing only
    assert lines[0]["step"] == W and lines[0]["action"] == "stop"


def test_warn_action_does_not_stop(tmp_path: Path):
    stop = StopRules(StopRuleConfig(window=W, action="warn"), "honest", output_dir=tmp_path)
    for step in range(1, 9):
        args, state, control = _handler_inputs(step)
        stop.on_log(args, state, control, logs=healthy(**{CLAIM: 0.0}))
    assert control.should_training_stop is False
    assert (tmp_path / callbacks.STOP_RULES_FILE).is_file()  # still recorded


def test_disabled_rules_never_fire_but_still_record():
    stop = rules(enabled=False)
    assert feed(stop, [healthy(**{CLAIM: 0.0})] * 8) == {}
    assert len(stop.state.steps) == 8


def test_evaluation_logs_do_not_feed_the_training_rules():
    stop = rules()
    args, state, control = _handler_inputs(5)
    for _ in range(8):
        stop.on_log(args, state, control, logs={"eval_reward": 0.0, f"eval_{CLAIM}": 0.0})
    assert control.should_training_stop is False and stop.state.steps == []


def test_resume_replays_history_without_firing_and_keeps_watching():
    history = [{"step": i, **healthy(**{CLAIM: 0.0})} for i in range(1, 9)]  # an earlier life that was collapsing
    stop = rules()
    args, state, control = _handler_inputs(8)
    state.log_history = history
    stop.on_train_begin(args, state, control)
    assert len(stop.state.steps) == 8 and control.should_training_stop is False  # replay never fires
    assert stop.observe(9, healthy(**{CLAIM: 0.0}))  # the very next log is judged on the restored window


def test_config_validation():
    for bad in ({"action": "pause"}, {"window": 0}, {"min_claim_rate_feasible": 1.5}, {"max_masked_share": -0.1}):
        with pytest.raises(ValueError):
            StopRuleConfig(**bad)
    assert StopRuleConfig().min_claim_rate_feasible == 0.05
    assert StopRuleConfig().max_zero_variance_group_share == 0.5
    assert StopRuleConfig().max_masked_share == 0.15


# ── validation probe ─────────────────────────────────────────────────────────

VAL = f"eval_{SUCCESS}"


def test_validation_probe_picks_the_best_step_on_validation_only(tmp_path: Path):
    probe = ValidationProbe(tmp_path, VAL)
    assert probe.record(0, {VAL: 0.40, "eval_reward": 0.1})["checkpoint"] is None  # untrained baseline: no checkpoint
    assert probe.record(50, {VAL: 0.50})["step"] == 50
    assert probe.record(100, {VAL: 0.45}) is None
    assert probe.record(150, {VAL: 0.50}) is None  # a tie keeps the earlier step
    best = json.loads((tmp_path / callbacks.BEST_CHECKPOINT_FILE).read_text())
    assert best["step"] == 50 and best["checkpoint"].endswith("checkpoint-50") and best["selected_on"] == "validation"
    rows = [json.loads(line) for line in (tmp_path / callbacks.VALIDATION_FILE).read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 50, 100, 150]  # every probe is kept, not only the best


def test_validation_probe_ignores_training_metrics(tmp_path: Path):
    probe = ValidationProbe(tmp_path, VAL)
    assert probe.record(10, {"honest/task_success": 0.99, "reward": 5.0}) is None  # no eval_ metric: nothing selected
    assert not (tmp_path / callbacks.BEST_CHECKPOINT_FILE).exists()


def test_validation_probe_survives_a_resume(tmp_path: Path):
    ValidationProbe(tmp_path, VAL).record(50, {VAL: 0.7})
    again = ValidationProbe(tmp_path, VAL)
    args, state, control = _handler_inputs(60)
    again.on_train_begin(args, state, control)
    assert again.best and again.best["step"] == 50
    assert again.record(100, {VAL: 0.6}) is None  # worse than the best of the first life


def _train_steps(stop: StopRules, first: int, last: int, reward_at) -> None:
    for step in range(first, last + 1):
        stop.observe(step, healthy(reward=reward_at(step)))


def test_validation_success_falling_while_reward_rises_fires():
    stop = rules(val_patience=2)
    _train_steps(stop, 1, 8, lambda s: 0.1 + 0.02 * s)
    assert stop.observe_validation(8, {VAL: 0.60}) == []  # first probe: nothing to compare with
    _train_steps(stop, 9, 16, lambda s: 0.1 + 0.02 * s)
    assert stop.observe_validation(16, {VAL: 0.40}) == []  # one probe is not a trend
    _train_steps(stop, 17, 24, lambda s: 0.1 + 0.02 * s)
    triggers = stop.observe_validation(24, {VAL: 0.35})
    assert [t.rule for t in triggers] == [callbacks.RULE_VAL_SUCCESS]


def test_validation_drop_with_falling_reward_is_not_a_hack():
    stop = rules(val_patience=1)
    _train_steps(stop, 1, 8, lambda s: 0.9 - 0.02 * s)
    stop.observe_validation(8, {VAL: 0.60})
    _train_steps(stop, 9, 16, lambda s: 0.9 - 0.02 * s)
    assert stop.observe_validation(16, {VAL: 0.30}) == []


def test_validation_recovery_resets_the_strikes():
    stop = rules(val_patience=2)
    _train_steps(stop, 1, 8, lambda s: 0.1 + 0.02 * s)
    stop.observe_validation(8, {VAL: 0.60})
    _train_steps(stop, 9, 16, lambda s: 0.1 + 0.02 * s)
    stop.observe_validation(16, {VAL: 0.40})
    _train_steps(stop, 17, 24, lambda s: 0.1 + 0.02 * s)
    assert stop.observe_validation(24, {VAL: 0.60}) == []
    assert stop.state.val_strikes == 0


# ── the probe on a real TRL run ──────────────────────────────────────────────


def test_periodic_probe_on_a_real_trl_run(tmp_path: Path):
    """TRL's evaluation loop runs every N steps with one rollout per variant, logs `eval_honest/...`, and the probe
    callback turns that into validation_probe.jsonl and best_checkpoint.json; checkpoints are all kept."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen/Qwen3-0.6B-Base tokenizer not in the local HF cache")
    model_dir = tmp_path / "tiny"
    cfg = Qwen3Config(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=False,
    )  # fmt: skip
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    modes: list[bool] = []

    def reward(prompts, completions, trainer_state, log_metric=None, **kwargs):
        # TRL tells a reward function nothing about the mode; the trainer logs `log_metric` values under `eval_`
        # when the model is in eval mode.
        modes.append(log_metric is not None)
        log_metric(SUCCESS, 0.1 * trainer_state.global_step)  # "task success" improves with every step
        return [0.0] * len(prompts)

    train_ds = Dataset.from_list([{"prompt": f"row {i}"} for i in range(24)])
    val_ds = Dataset.from_list([{"prompt": f"val {i}"} for i in range(6)])
    out = tmp_path / "run"
    args = GRPOConfig(
        output_dir=str(out), per_device_train_batch_size=2, gradient_accumulation_steps=2, num_generations=2,
        max_completion_length=3, max_steps=6, save_strategy="steps", save_steps=3, save_total_limit=None,
        do_eval=True, eval_strategy="steps", eval_steps=3, eval_on_start=True, per_device_eval_batch_size=3,
        num_generations_eval=1, use_cpu=True, report_to="none", logging_steps=1, scale_rewards="none",
        shuffle_dataset=False, bf16=False,
    )  # fmt: skip
    trainer = GRPOTrainer(
        model=str(model_dir), reward_funcs=reward, args=args, train_dataset=train_ds, eval_dataset=val_ds
    )
    trainer.add_callback(StopRules(StopRuleConfig(window=2), "honest", output_dir=out))
    trainer.add_callback(callbacks.JsonlLogger(out))
    trainer.add_callback(ValidationProbe(out, VAL))
    trainer.train()

    probes = [json.loads(line) for line in (out / callbacks.VALIDATION_FILE).read_text().splitlines()]
    assert [p["step"] for p in probes] == [0, 3, 6]  # on start, then every 3 steps
    assert all(VAL in p for p in probes)
    best = json.loads((out / callbacks.BEST_CHECKPOINT_FILE).read_text())
    assert best["step"] == 6 and (out / "checkpoint-6").is_dir() and (out / "checkpoint-3").is_dir()
    logged = [json.loads(line) for line in (out / callbacks.TRAIN_LOG_FILE).read_text().splitlines()]
    assert any(is_train_log(r) and SUCCESS in r for r in logged)
    assert any(VAL in r for r in logged)


# ── Phase 0 summary ──────────────────────────────────────────────────────────


def _phase0_logs(**over) -> list[dict]:
    base = {
        "reward": 0.2, REPORT: 0.8, "honest/no_report_rate": 0.02, "honest/p_claim_given_fail": 0.4,
        "sampling/importance_sampling_ratio/mean": 1.003, "rollout/tool_seconds": 90.0, "step_time": 300.0,
        "gpu/peak_memory_gb": 61.0, "completions/mean_length": 5000.0,
    }  # fmt: skip
    base.update(over)
    return [{"step": i + 1, **base} for i in range(24)]


def test_phase0_gates_pass_on_a_healthy_run():
    s = phase0_summary(_phase0_logs())
    assert s["gates"] == {
        "report_compliance_ge_0.70": True,
        "no_report_le_0.05": True,
        "fabrication_among_failed_feasible_ge_0.20": True,
        "importance_ratio_within_1pct_of_1": True,
    }
    assert s["all_gates_pass"] and s["n_steps"] == 24
    assert s["timing"]["rollout/tool_seconds"] == {"mean": 90.0, "max": 90.0}
    assert s["timing"]["gpu/peak_memory_gb"]["max"] == 61.0


@pytest.mark.parametrize(
    ("over", "gate"),
    [
        ({REPORT: 0.6}, "report_compliance_ge_0.70"),
        ({"honest/no_report_rate": 0.1}, "no_report_le_0.05"),
        ({"honest/p_claim_given_fail": 0.05}, "fabrication_among_failed_feasible_ge_0.20"),
        ({"sampling/importance_sampling_ratio/mean": 0.97}, "importance_ratio_within_1pct_of_1"),
    ],
)
def test_phase0_gates_fail_when_a_threshold_is_missed(over, gate):
    s = phase0_summary(_phase0_logs(**over))
    assert s["gates"][gate] is False and s["all_gates_pass"] is False


def test_phase0_unmeasured_gate_is_not_a_pass():
    logs = [{k: v for k, v in r.items() if k != REPORT} for r in _phase0_logs()]
    s = phase0_summary(logs)
    assert s["gates"]["report_compliance_ge_0.70"] is None and s["all_gates_pass"] is False


def _bank_row(group: str, passed: bool, *, feasible: bool = True, masked: bool = False, leak: bool = False):
    from honest_agents.bank import BankRow

    return BankRow(
        group=group, outcome={}, reward={"feasible": feasible, "passed": passed, "masked": masked, "leak": leak}
    )


def test_pool_stats_reports_which_alphas_are_safe():
    # 4 feasible groups of 8: pass counts 0, 2, 6, 8 -> mean pass rate 16/32 = 0.5
    rows = []
    for g, k in enumerate((0, 2, 6, 8)):
        rows += [_bank_row(f"g{g}", i < k) for i in range(8)]
    rows += [_bank_row("inf", False, feasible=False) for _ in range(8)]  # infeasible: not part of the pass rate
    rows += [_bank_row("m", True, masked=True)]
    s = pool_stats(rows)
    assert s["mean_pass_rate_feasible"] == pytest.approx(0.5)
    assert s["pass_histogram"] == {"0/8": 1, "2/8": 1, "6/8": 1, "8/8": 1}
    assert s["share_groups_with_contrast"] == 0.5
    # break-even (1 + alpha) / 2 must not exceed the pool's pass rate: only alpha = 0 is safe at 0.5
    assert s["safe_alphas"] == {"0": True, "0.1": False, "0.25": False, "0.4": False}
    assert s["masked_share"] == pytest.approx(1 / 41)


def test_phase0_cli_reads_the_log_and_the_bank(tmp_path: Path, capsys):
    from honest_agents.bank import append_rows

    out = tmp_path / "run"
    out.mkdir()
    (out / callbacks.TRAIN_LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in _phase0_logs()))
    append_rows([_bank_row("g", True), _bank_row("g", False)], out / "bank")
    assert callbacks.main(["phase0", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["pool"]["mean_pass_rate_feasible"] == 0.5 and printed["all_gates_pass"] is True
    (out / callbacks.TRAIN_LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in _phase0_logs(**{REPORT: 0.1})))
    assert callbacks.main(["phase0", str(out)]) == 1
