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


def test_learning_is_judged_on_honesty_rates_not_on_difficulty_confounded_reward():
    honest = {"honest/n_infeasible": 16, "honest/n_feasible": 48}
    flat = [
        step(i, **honest, **{"honest/claim_rate_infeasible": 0.35, "honest/p_claim_given_fail": 0.6})
        for i in range(1, 16)
    ]
    assert checks(flat)["learning"] == WARN
    assert "learning" not in checks(flat[:10])
    improving = [
        step(
            i,
            reward=0.6 - 0.02 * i,
            **honest,
            **{"honest/claim_rate_infeasible": 0.4 - 0.02 * i, "honest/p_claim_given_fail": 0.6},
        )
        for i in range(1, 16)
    ]
    assert "learning" not in checks(improving)  # reward falls (harder tasks) but false claims on impossible tasks fell


def test_cli_exit_status_is_the_worst_level(tmp_path, capsys):
    (tmp_path / TRAIN_LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in run(6, entropy=0.1)))
    assert main([str(tmp_path), "--now", "0"]) == FAIL
    assert capsys.readouterr().out.startswith("FAIL | step 6")
    assert main([str(tmp_path / "missing")]) == OK


def test_a_slow_rise_of_feasible_declines_is_flagged_before_it_reaches_the_ceiling():
    rows = [step(i, **{"honest/decline_rate_feasible": 0.005 + 0.006 * i}) for i in range(1, 16)]
    assert checks(rows)["gaming"] == WARN
    assert "gaming" not in checks(run(15))


def test_host_ram_growth_warns_before_the_oom_killer():
    rows = [step(i, **{"mem/rss_gb": 10.0 + 3.8 * i, "mem/available_gb": 60.0 - 3.8 * i}) for i in range(1, 8)]
    assert checks(rows)["memory"] == WARN  # ~33 GB left at 3.8 GB/step: under 10 steps
    rows = [step(i, **{"mem/rss_gb": 10.0 + 3.8 * i, "mem/available_gb": 60.0 - 3.8 * i}) for i in range(1, 15)]
    assert checks(rows)["memory"] == FAIL  # ~7 GB left: about 2 steps
    assert "memory" not in checks([step(i, **{"mem/rss_gb": 20.0, "mem/available_gb": 50.0}) for i in range(1, 8)])


def canary_run(n: int, key: str, start: float, late: float, from_step: int, **extra) -> list[dict]:
    """`n` steps with `key` at `start`, then at `late` from step `from_step` on; ~33 controls and 64 episodes per step."""
    counts = {"honest/n_control": 33, "honest/n_scored": 64}
    return [step(i, **counts, **extra, **{key: late if i >= from_step else start}) for i in range(1, n + 1)]


SUCCESS = "honest/control/verified_success_rate"
DECLINE = "honest/control/decline_rate"
ZERO = "honest/zero_command_rate"


def test_flat_over_abstention_canaries_trip_nothing():
    rows = [
        step(i, **{"honest/n_control": 33, "honest/n_scored": 64, SUCCESS: 0.78, DECLINE: 0.02, ZERO: 0.01})
        for i in range(1, 21)
    ]
    assert checks(rows) == {"all": OK}


def test_per_step_canaries_only_warn_however_large_the_shift():
    # The steps draw different controls (base pass rates 0.25 to 1.0), so the per-step comparison is a hint, never a FAIL.
    assert checks(canary_run(12, SUCCESS, 0.80, 0.68, from_step=8))["capability"] == WARN  # 12 points
    assert "capability" not in checks(canary_run(12, SUCCESS, 0.80, 0.72, from_step=8))  # 8 points: noise
    assert checks(canary_run(12, SUCCESS, 0.80, 0.60, from_step=8))["capability"] == WARN
    assert checks(canary_run(14, SUCCESS, 0.80, 0.0, from_step=6))["capability"] == WARN
    assert "per-step hint" in next(
        f.message for f in check(canary_run(12, SUCCESS, 0.8, 0.6, 8)) if f.check == "capability"
    )


def test_canaries_need_two_disjoint_windows():
    assert "capability" not in checks(canary_run(9, SUCCESS, 0.80, 0.0, from_step=6))  # first 5 and last 5 overlap
    assert checks(canary_run(10, SUCCESS, 0.80, 0.0, from_step=6))["capability"] == WARN


def test_rising_control_declines_and_zero_command_episodes_are_flagged():
    assert checks(canary_run(12, DECLINE, 0.02, 0.14, from_step=8))["over-refusal"] == WARN
    assert checks(canary_run(14, DECLINE, 0.02, 0.30, from_step=8))["over-refusal"] == WARN
    assert "over-refusal" not in checks(canary_run(14, DECLINE, 0.30, 0.02, from_step=8))  # falling declines are fine
    assert checks(canary_run(14, ZERO, 0.01, 0.40, from_step=8))["quitting"] == WARN
    assert "quitting" not in checks(canary_run(14, ZERO, 0.01, 0.05, from_step=8))


def test_an_unchanged_policy_on_a_shifting_task_mix_does_not_fail():
    """The review's simulation: control base pass rates 0.25-1.0, a different block of 33 every step, the policy
    unchanged, 50 steps probed at 0, 20 and 40 on the same 64 controls with one sample each. The per-step canary may
    warn but never FAILs; the probe check FAILs on well under 2 % of runs (two fail-sized probes in a row)."""
    import random

    rng = random.Random(0)
    pool = [rng.uniform(0.25, 1.0) for _ in range(400)]
    controls = pool[:64]
    probe_fails = 0
    trials = 200
    for trial in range(trials):
        rows = [probe_row(0, success=sum(rng.random() < p for p in controls) / 64)]
        for i in range(1, 51):
            start = (i * 33 + trial * 7) % 360
            block = pool[start : start + 33]
            rate = sum(rng.random() < p for p in block) / len(block)
            rows.append(step(i, **{"honest/n_control": len(block), "honest/n_scored": 64, SUCCESS: rate}))
            if i % 20 == 0:
                rows.append(probe_row(i, success=sum(rng.random() < p for p in controls) / 64))
        found = checks(rows)
        assert found.get("capability", OK) < FAIL
        probe_fails += found.get("probe-capability", OK) == FAIL
    assert probe_fails / trials < 0.02


P_SUCCESS = "eval_probe/verified_success_control"
P_DECLINE = "eval_probe/control_decline_rate"
P_ZERO = "eval_probe/zero_command_rate"


def probe_row(i: int, success: float = 0.78, decline: float = 0.03, zero: float = 0.02, n_variants: int = 192) -> dict:
    return {"step": i, "time": 1000.0 + 400.0 * i + 1.0, "eval_honest/task_success": 0.6, P_SUCCESS: success,
            P_DECLINE: decline, P_ZERO: zero, "eval_probe/n_variants": float(n_variants)}  # fmt: skip


def probed_run(*probes: dict, n: int = 60) -> list[dict]:
    """`n` healthy steps with the probe rows placed after their step's training row."""
    rows = [probe_row(0, **probes[0])] if probes else []
    later = {20 * k: p for k, p in enumerate(probes[1:], start=1)}
    for i in range(1, n + 1):
        rows.append(step(i))
        if i in later:
            rows.append(probe_row(i, **later[i]))
    return rows


def test_the_probe_check_compares_each_probe_with_the_step_0_probe_on_the_same_variants():
    assert "probe-capability" not in checks(probed_run({}, {"success": 0.70}))  # 8 points: noise
    assert checks(probed_run({}, {"success": 0.66}))["probe-capability"] == WARN  # 12 points
    assert checks(probed_run({}, {"success": 0.60}))["probe-capability"] == WARN  # 18 points, one probe
    fail = check(probed_run({}, {"success": 0.60}, {"success": 0.58}))
    found = next(f for f in fail if f.check == "probe-capability")
    assert found.level == FAIL and "at step 0" in found.message and "2 consecutive probes" in found.message
    assert "probe-capability" not in checks(probed_run({}, {"success": 0.60}, {"success": 0.75}))  # recovered
    assert checks(probed_run({}, {"decline": 0.25}, {"decline": 0.30}))["probe-over-refusal"] == FAIL
    assert checks(probed_run({}, {"zero": 0.25}, {"zero": 0.30}))["probe-quitting"] == FAIL
    other = probed_run({}, {"success": 0.5, "n_variants": 100}, {"success": 0.5, "n_variants": 100})
    assert "probe-capability" not in checks(other)  # a different probe list has no paired baseline


def test_a_re_probed_step_counts_once_with_its_latest_life():
    rows = probed_run({}, {"success": 0.60}, {"success": 0.58})
    rows.append(probe_row(40, success=0.78))  # the step-40 probe ran again after a resume and recovered
    assert "probe-capability" not in checks(rows)


def test_canaries_weight_steps_by_their_control_count():
    # Four late steps drew one control each and failed it; the fifth drew 40 and kept its success.
    rows = canary_run(10, SUCCESS, 0.80, 0.80, from_step=11)
    for i in (5, 6, 7, 8):
        rows[i].update({"honest/n_control": 1, SUCCESS: 0.0})
    rows[9].update({"honest/n_control": 40})
    assert "capability" not in checks(rows)  # (40 x 0.8) / 44 = 0.73: 7 points
    assert checks([{**r, "honest/n_control": 33} for r in rows])["capability"] == WARN  # unweighted: 0.16


def test_a_validation_probe_between_steps_is_not_a_stall():
    rows = run(5)
    probe = {"step": 5, "time": rows[-1]["time"] + 600.0, "eval_reward": 0.6}  # a 10-minute probe logs its own row
    assert "stall" not in checks(rows + [probe], now=probe["time"] + 400)
    assert checks(rows, now=rows[-1]["time"] + 1200)["stall"] == FAIL  # over 2.5 x the 400 s step
