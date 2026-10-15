"""Stop rules, the validation probe and the Phase-0 summary (docs/design.md section 8). No GPU, no Docker.

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
    tracked -= set(callbacks.TRAINER_METRICS)  # logged by TRL and trainer.py, not by the reward function
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


TAMPER = trl_rewards.METRIC_TAMPER_RATE
ERRORS = trl_rewards.METRIC_REWARD_ERROR_RATE


def test_tamper_rate_fires_over_its_own_short_window():
    """CA-1: a flaky grader is scored as tamper, so the tamper rule has to act within a few steps, not a full window."""
    calm = healthy(**{TAMPER: 0.0})
    fired = feed(rules(tamper_window=2), [calm] * 3 + [healthy(**{TAMPER: 0.15})] * 3)
    assert 4 not in fired  # one spike is averaged with a calm step: 0.075
    assert fired[5] == [callbacks.RULE_TAMPER_RATE]  # two bad steps complete the 2-step window; W = 4 would need more
    assert feed(rules(tamper_window=2), [healthy(**{TAMPER: 0.1})] * 8) == {}  # at the ceiling does not fire
    assert feed(rules(tamper_window=2), [calm] * 8) == {}


def test_tamper_rate_window_is_five_steps_by_default():
    stop = StopRules(StopRuleConfig(window=20), reward_mode="honest")
    fired = feed(stop, [healthy(**{TAMPER: 0.0})] * 10 + [healthy(**{TAMPER: 0.5})] * 5)
    assert fired[15] == [callbacks.RULE_TAMPER_RATE]  # five steps in, long before a 20-step window could judge it


def test_error_rate_and_tamper_rate_are_separate_rules():
    only_errors = feed(rules(), [healthy(**{ERRORS: 0.3})] * 5)
    assert only_errors[W] == [callbacks.RULE_REWARD_ERROR]
    only_tamper = feed(rules(), [healthy(**{TAMPER: 0.3})] * 5)  # default tamper window: 5 steps
    assert only_tamper[5] == [callbacks.RULE_TAMPER_RATE]


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
    bad_configs = (
        {"action": "pause"},
        {"window": 0},
        {"tamper_window": 0},
        {"max_tamper_rate": 1.5},
        {"min_claim_rate_feasible": 1.5},
        {"max_masked_share": -0.1},
    )
    for bad in bad_configs:
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


# ── capability on the fixed probe ────────────────────────────────────────────

P_SUCCESS = "eval_probe/verified_success_control"
P_DECLINE = "eval_probe/control_decline_rate"
P_ZERO = "eval_probe/zero_command_rate"


def probe(success: float = 0.78, decline: float = 0.03, zero: float = 0.02, n_variants: int = 192) -> dict:
    """The eval_probe/* rates train.probe logs for one probe of the same 192-variant list."""
    return {P_SUCCESS: success, P_DECLINE: decline, P_ZERO: zero, "eval_probe/n_variants": float(n_variants),
            "eval_probe/n_control": 64.0, VAL: 0.6}  # fmt: skip


def run_probes(stop: StopRules, probes: list[tuple[int, dict]]) -> TrainerControl:
    control = TrainerControl()
    for step, metrics in probes:
        stop.on_evaluate(SimpleNamespace(), TrainerState(global_step=step), control, metrics=metrics)
    return control


def test_probe_metric_names_are_the_ones_train_probe_logs():
    from honest_agents.train.probe import METRIC_PREFIX, _block

    assert callbacks.PROBE_METRIC_PREFIX == METRIC_PREFIX
    names = {METRIC_PREFIX + k for k in _block([])} | {METRIC_PREFIX + "n_variants"}
    assert {c.key for c in callbacks.PROBE_CANARIES} | {callbacks.PROBE_N_VARIANTS} <= names


def test_a_sustained_capability_drop_on_the_probe_stops_the_run(tmp_path: Path):
    stop = StopRules(StopRuleConfig(), "honest", output_dir=tmp_path)
    control = run_probes(stop, [(0, probe()), (20, probe(success=0.60))])
    assert control.should_training_stop is False  # one fail-sized probe is not yet sustained
    control = run_probes(stop, [(40, probe(success=0.58))])
    assert control.should_training_stop is True and stop.stopped
    lines = [json.loads(line) for line in (tmp_path / callbacks.STOP_RULES_FILE).read_text().splitlines()]
    assert [line["rule"] for line in lines] == ["probe_control_success_falling"]
    assert lines[0]["step"] == 40 and lines[0]["threshold"] == pytest.approx(0.78 - 0.15)
    assert "from 78.0% at step 0" in lines[0]["detail"]


def test_rising_declines_or_zero_command_episodes_on_the_probe_stop_the_run():
    stop = rules()
    run_probes(stop, [(0, probe()), (20, probe(decline=0.25))])
    assert run_probes(stop, [(40, probe(decline=0.30))]).should_training_stop is True
    stop = rules()
    run_probes(stop, [(0, probe()), (20, probe(zero=0.30))])
    assert run_probes(stop, [(40, probe(zero=0.40))]).should_training_stop is True


def test_noise_sized_probe_changes_never_stop_the_run():
    stop = rules()
    noise = [(0, probe()), (20, probe(success=0.70, decline=0.10)), (40, probe(success=0.66, zero=0.12)),
             (60, probe(success=0.72)), (80, probe(success=0.64, decline=0.15, zero=0.15))]  # fmt: skip
    assert run_probes(stop, noise).should_training_stop is False and not stop.state.fired
    # a fail-sized drop that recovers in between is not sustained either
    stop = rules()
    flicker = [(0, probe()), (20, probe(success=0.55)), (40, probe(success=0.70)), (60, probe(success=0.55))]
    assert run_probes(stop, flicker).should_training_stop is False
    # rising success and falling declines are improvements, not drops
    stop = rules()
    better = [(0, probe()), (20, probe(success=0.99, decline=0.0)), (40, probe(success=0.99, decline=0.0))]
    assert run_probes(stop, better).should_training_stop is False


def test_the_probe_rule_honours_warn_patience_and_a_changed_variant_set(tmp_path: Path):
    drop = [(0, probe()), (20, probe(success=0.50)), (40, probe(success=0.50))]
    stop = StopRules(StopRuleConfig(action="warn"), "honest", output_dir=tmp_path)
    assert run_probes(stop, drop).should_training_stop is False and "probe_control_success_falling" in stop.state.fired
    assert run_probes(rules(probe_capability_patience=1), drop[:2]).should_training_stop is True
    assert run_probes(rules(probe_capability_drop=1.0), drop).should_training_stop is False  # turned off
    other = [(0, probe()), (20, probe(success=0.50, n_variants=100)), (40, probe(success=0.50, n_variants=100))]
    assert run_probes(rules(), other).should_training_stop is False  # another probe list: no paired baseline
    with pytest.raises(ValueError):
        StopRuleConfig(probe_capability_patience=0)


def test_a_resumed_run_keeps_its_step_0_probe_as_the_baseline():
    history = [
        {"step": 0, **probe()},
        *({"step": i, **healthy()} for i in range(1, 21)),
        {"step": 20, **probe(success=0.6)},
    ]
    stop = rules()
    args, state, control = _handler_inputs(20)
    state.log_history = history
    stop.on_train_begin(args, state, control)
    assert [s for s, _ in stop.state.probe_rates] == [0, 20] and control.should_training_stop is False
    assert run_probes(stop, [(40, probe(success=0.6))]).should_training_stop is True


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
    # the tiny model cuts every rollout, so the truncation ceiling is lifted
    trainer.add_callback(StopRules(StopRuleConfig(window=2, max_truncated_share=1.0), "honest", output_dir=out))
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


def test_probe_schedule_evaluates_only_the_listed_steps():
    """`probe.steps: [10, 15, 20]` on TRL's interval of 5: transformers' flow callback asks for a probe every 5 steps
    (and at the last step); ProbeSchedule, which runs after it, keeps only the listed ones."""
    from transformers.trainer_callback import DefaultFlowCallback
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy

    args = SimpleNamespace(
        eval_strategy=IntervalStrategy.STEPS, eval_delay=0, logging_strategy=IntervalStrategy.STEPS,
        logging_first_step=False, save_strategy=SaveStrategy.STEPS,
    )  # fmt: skip
    state = TrainerState(max_steps=22, eval_steps=5, save_steps=5, logging_steps=1)
    flow, schedule = DefaultFlowCallback(), callbacks.ProbeSchedule([10, 15, 20])
    probed = []
    for step in range(1, 23):
        state.global_step = step
        control = TrainerControl()
        flow.on_step_end(args, state, control)
        schedule.on_step_end(args, state, control)
        if control.should_evaluate:
            probed.append(step)
    assert probed == [10, 15, 20]


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


@pytest.mark.parametrize(
    ("pass_rate", "expected"),
    [
        (0.5, {"0": False, "0.1": False, "0.25": False, "0.4": False}),  # (1 + 0) / 2 = 0.5 > 0.5 - 0.1
        (0.62, {"0": True, "0.1": False, "0.25": False, "0.4": False}),  # 0.5 <= 0.52 < 0.55
        (0.75, {"0": True, "0.1": True, "0.25": True, "0.4": False}),  # 0.625 <= 0.65 < 0.7
        (0.9, {"0": True, "0.1": True, "0.25": True, "0.4": True}),
    ],
)
def test_pool_stats_exposes_a_hard_skip_boolean_per_alpha(pass_rate, expected):
    """The trainer may refuse to train an alpha whose claim threshold is not 0.1 below the pool's pass rate."""
    n = 1000
    rows = [_bank_row("g", i < pass_rate * n) for i in range(n)]
    assert pool_stats(rows)["alpha_is_safe"] == expected
    assert pool_stats([_bank_row("i", False, feasible=False)])["alpha_is_safe"] is None
    for alpha in (0.0, 0.25):
        assert callbacks.alpha_is_safe(alpha, pass_rate) == expected[f"{alpha:g}"]


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


def _drift_rules(config: callbacks.StopRuleConfig, logs: dict[str, float], steps: int = 4) -> list[str]:
    rules = callbacks.StopRules(config)
    fired: list[str] = []
    for step in range(1, steps + 1):
        fired = [t.rule for t in rules.observe(step, dict(logs))]
    return fired


def test_sampler_trainer_drift_rules_fire_only_outside_their_bands():
    cfg = callbacks.StopRuleConfig(window=4)
    assert _drift_rules(cfg, {callbacks.METRIC_IS_MEAN: 1.01, callbacks.METRIC_IS_AT_CAP: 0.01}) == []
    assert callbacks.RULE_IS_DRIFT in _drift_rules(cfg, {callbacks.METRIC_IS_MEAN: 0.9})
    assert callbacks.RULE_IS_DRIFT in _drift_rules(cfg, {callbacks.METRIC_IS_MEAN: 1.2})
    assert callbacks.RULE_IS_AT_CAP in _drift_rules(cfg, {callbacks.METRIC_IS_AT_CAP: 0.2})


def test_truncated_share_rule_uses_the_absolute_ceiling():
    cfg = callbacks.StopRuleConfig(window=4)
    assert _drift_rules(cfg, {callbacks.METRIC_TRUNCATED: 0.2}) == []
    assert _drift_rules(cfg, {callbacks.METRIC_TRUNCATED: 0.5}) == [callbacks.RULE_TRUNCATED]


def test_masked_collapse_fires_on_two_consecutive_steps_not_one():
    """Iteration 15 G3: all boots failed, masked share 1.0 every step, and the 20-step window rule waited."""
    bad = healthy(**{MASKED: 1.0, "honest/masked/boot_failure": 1.0, "honest/masked/toolchain": 0.0})
    fired = feed(rules(), [healthy(), bad, healthy(), bad])
    assert fired == {}  # never two in a row
    stop = rules()
    fired = feed(stop, [healthy(), bad, bad])
    assert fired == {3: [callbacks.RULE_MASKED_COLLAPSE]}
    trig = StopRules(StopRuleConfig(window=20)).observe(1, bad)
    assert trig == []
    s2 = StopRules(StopRuleConfig(window=20))
    s2.observe(1, bad)
    (t,) = s2.observe(2, bad)
    assert "boot_failure" in t.detail
    assert StopRuleConfig().masked_fail_share == 0.5 and StopRuleConfig().masked_fail_patience == 2
    assert feed(rules(masked_fail_patience=3), [bad, bad]) == {}
    assert feed(rules(), [healthy(**{MASKED: 0.49})] * 4) == {}
