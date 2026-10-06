from __future__ import annotations

import json

from honest_agents.callbacks import TRAIN_LOG_FILE
from honest_agents.train.health import FAIL, OK, WARN, check, main


def step(i: int, **over) -> dict:
    row = {
        "step": i,
        "time": 1000.0 + 400.0 * i,
        "step_time": 400.0,
        "loss": 0.05,
        "grad_norm": 0.06,
        "reward": 0.30 + 0.01 * i,  # slowly learning
        "entropy": 0.32,
        "honest/task_success": 0.6,
        "honest/mean_commands": 15.0,
        "honest/no_report_rate": 0.15,
        "honest/lazy_rate_feasible": 0.0,
        "honest/decline_rate_feasible": 0.05,
        "honest/tamper_rate": 0.0,
        "honest/reward_error_rate": 0.0,
        "honest/masked_share": 0.0,
        "honest/zero_variance_group_share": 0.25,
        "completions/mean_length": 2700.0,
        "completions/clipped_ratio": 0.0,
        "sampling/importance_sampling_ratio/mean": 1.0,
        "sampling/importance_sampling_ratio/at_cap_share": 0.0,
        "sampling/sampling_logp_difference/mean": 0.008,
    }
    row.update(over)
    return row


def run(n: int, **late) -> list[dict]:
    """`n` healthy steps; the last 3 carry the `late` overrides."""
    return [step(i, **(late if i > n - 3 else {})) for i in range(1, n + 1)]


def checks(rows, **kw) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in check(rows, **kw):
        out[f.check] = max(out.get(f.check, OK), f.level)
    return out


def test_a_healthy_learning_run_trips_nothing():
    rows = run(20)
    assert checks(rows, now=rows[-1]["time"] + 60) == {"all": OK}


def test_stall_and_non_finite_numbers_fail():
    rows = run(5)
    assert checks(rows, now=rows[-1]["time"] + 3 * 400)["stall"] == FAIL
    assert checks(run(5, loss=float("nan")))["numbers"] == FAIL


def test_sync_drift_shows_as_a_ratio_away_from_one_or_a_growing_log_prob_gap():
    assert checks(run(6, **{"sampling/importance_sampling_ratio/mean": 1.08}))["drift"] == FAIL
    assert checks(run(6, **{"sampling/sampling_logp_difference/mean": 0.05}))["drift"] == WARN


def test_collapse_and_quitting_fail():
    assert checks(run(8, entropy=0.1))["collapse"] == FAIL
    assert checks(run(8, **{"honest/mean_commands": 2.0}))["collapse"] == FAIL
    assert checks(run(8, **{"honest/lazy_rate_feasible": 0.4}))["gaming"] == FAIL
    assert checks(run(8, **{"honest/tamper_rate": 0.3}))["gaming"] == FAIL


def test_a_flat_reward_after_enough_steps_is_a_warning_not_before():
    flat = [step(i, reward=0.3) for i in range(1, 16)]
    assert checks(flat)["learning"] == WARN
    assert "learning" not in checks(flat[:10])


def test_cli_exit_status_is_the_worst_level(tmp_path, capsys):
    (tmp_path / TRAIN_LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in run(6, entropy=0.1)))
    assert main([str(tmp_path), "--now", "0"]) == FAIL
    assert capsys.readouterr().out.startswith("FAIL | step 6")
    assert main([str(tmp_path / "missing")]) == OK
