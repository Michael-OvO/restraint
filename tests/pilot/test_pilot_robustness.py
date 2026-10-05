"""Robustness tests: memorisation guard, stale logs, partial sweeps, bank discovery, one-advantage noise diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from honest_agents import bank
from honest_agents.pilot import analyze, dataset, run
from honest_agents.pilot.env import BANK_ROW_COLUMN, ReplayEnv
from honest_agents.pilot.rewards import (
    CURVE_FILE,
    EPISODES_FILE,
    METRIC_ADV_FABRICATION,
    METRIC_ADV_HONEST_FAILURE,
    EpisodeLog,
    PilotReward,
)
from honest_agents.reward import RewardConfig
from tests.pilot.helpers import make_bank_row, report_call, synthetic_rows, write_bank
from tests.pilot.sim import simulate_arm, write_arm

CLAIM = {"status": "complete", "completed": ["x"], "incomplete": [], "confidence": 0.9}
DECLINE = {"status": "incomplete", "completed": [], "incomplete": ["x"], "confidence": 0.3}


def _cfg(**grpo) -> run.PilotConfig:
    cfg = run.PilotConfig()
    cfg.grpo = {"num_generations": 8, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 32, **grpo}
    return cfg


def test_total_prompts_counts_distinct_prompts_not_completions():
    assert run.total_prompts(_cfg(max_steps=100).grpo) == 800  # 100 steps x 64 completions / 8 per prompt
    assert run.total_prompts(_cfg().grpo) is None


def test_a_run_that_repeats_prompts_is_refused():
    rows = [{"x": i} for i in range(300)]
    cfg = _cfg(max_steps=100)  # 800 prompts over 300 rows: 2.7 epochs
    with pytest.raises(ValueError, match="memoris"):
        run.check_epochs(cfg, rows)
    cfg.data.max_epochs = 3.0
    assert run.check_epochs(cfg, rows) == pytest.approx(800 / 300)
    assert run.check_epochs(_cfg(max_steps=100), [{}] * 800) == 1.0


def test_the_shipped_config_shows_every_prompt_once():
    cfg = run.load_pilot_config(Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml")
    assert run.total_prompts(cfg.grpo) <= cfg.data.min_rows
    run.check_epochs(cfg, [{}] * cfg.data.min_rows)


def test_a_fresh_run_moves_the_failed_attempts_logs_aside(tmp_path):
    (tmp_path / EPISODES_FILE).write_text('{"round": 0}\n')
    (tmp_path / CURVE_FILE).write_text('{"round": 0}\n')
    moved = run.archive_logs(tmp_path)
    assert len(moved) == 2 and not (tmp_path / EPISODES_FILE).exists()
    assert all(p.name.endswith(".old") and p.exists() for p in moved)
    assert EpisodeLog(tmp_path).round == 0  # the round count starts over, so the curves do not mix two runs
    assert run.archive_logs(tmp_path) == []


def test_bank_discovery_finds_the_phase0_layout_and_applies_its_quarantine(tmp_path):
    rows = synthetic_rows(n_feasible=6, n_control=2, n_infeasible=2)
    write_bank(tmp_path / "phase0" / "run-a" / "bank", rows[:5])
    write_bank(tmp_path / "phase0" / "run-b" / "bank", rows[5:])
    assert len(dataset.bank_files(tmp_path / "phase0")) == 2
    bad = rows[0]
    variant = dataset.variant_of(bad)
    bank.append_quarantine(
        [bank.QuarantineEntry(variant_id=variant, kind=bank.QUARANTINE_LEAK)],
        tmp_path / "phase0" / "run-a" / "bank" / bank.QUARANTINE_BASENAME,
    )
    out, stats = dataset.build_from_bank(tmp_path / "phase0")
    assert stats.n_bank_rows == len(rows)
    assert all(r["variant_id"] != variant for r in out) and stats.dropped.get(dataset.DROP_QUARANTINED, 0) >= 1
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no bank"):
        dataset.build_from_bank(tmp_path / "empty")


def _sweep(root: Path, completed: bool | None = True, alphas=(0.0, 0.25, 0.5, 0.75)) -> None:
    for alpha in alphas:
        d = write_arm(root, alpha, "none", simulate_arm(alpha, "none", rounds=120, seed=5)[1])
        if completed is not None:
            meta = json.loads((d / "arm.json").read_text())
            (d / "arm.json").write_text(json.dumps({**meta, "completed": completed}))


def test_a_sweep_with_unfinished_arms_is_never_a_go(tmp_path):
    _sweep(tmp_path, completed=True)
    rule = analyze.StopRule(window_rounds=10)
    assert analyze.analyse(tmp_path, rule, write=False).verdict == analyze.GO
    meta = json.loads((tmp_path / "a0.75_none_s42" / "arm.json").read_text())
    (tmp_path / "a0.75_none_s42" / "arm.json").write_text(json.dumps({**meta, "completed": False}))
    v = analyze.analyse(tmp_path, rule, write=False)
    assert v.verdict == analyze.UNDECIDED and "not finished: a0.75_none_s42" in v.line()


def test_a_sweep_missing_an_arm_is_never_a_go(tmp_path):
    _sweep(tmp_path, alphas=(0.0, 0.25, 0.5))  # the config also wants alpha 0.75
    rule = analyze.StopRule(window_rounds=10)
    assert analyze.analyse(tmp_path, rule, write=False).verdict == analyze.GO  # no list of expected arms: as before
    expected = [f"a{a:g}_none_s42" for a in (0.0, 0.25, 0.5, 0.75)]
    v = analyze.analyse(tmp_path, rule, write=False, expected=expected)
    assert v.verdict == analyze.UNDECIDED and "a0.75_none_s42" in v.line()


def test_the_analyze_cli_requires_the_configs_whole_sweep_unless_told_otherwise(tmp_path):
    cfg_root = Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml"
    _sweep(tmp_path / "out", alphas=(0.0, 0.25, 0.5, 0.75))  # `none` arms only: the `group` arms are missing
    assert analyze.main([str(tmp_path / "out"), "--config", str(cfg_root), "--no-write"]) == 2
    assert analyze.main([str(tmp_path / "out"), "--config", str(cfg_root), "--no-write", "--allow-partial"]) == 0


def test_the_advantage_of_honest_failure_is_logged(tmp_path):
    """The one-advantage-per-trajectory noise is made visible. Under `none` an honest failure in a group of
    mostly fabricators is positive, in a group of mostly honest failures it is negative."""
    row = make_bank_row(variant="f", passed=False, original={"status": "complete"})
    envs, completions = [], []
    for msg in [report_call(CLAIM)] * 6 + [report_call(DECLINE)] * 2:
        env = ReplayEnv(RewardConfig())
        env.reset(**{BANK_ROW_COLUMN: row.model_dump_json()})
        envs.append(env)
        completions.append([msg])
    metrics = {}
    rewards = PilotReward("honest", EpisodeLog(tmp_path), scale_rewards="none")(
        environments=envs,
        completions=completions,
        prompts=[[] for _ in envs],
        log_metric=lambda k, v: metrics.__setitem__(k, v),
        trainer_state=SimpleNamespace(global_step=1),
    )
    assert rewards == [-1.0] * 6 + [0.25] * 2
    assert metrics[METRIC_ADV_HONEST_FAILURE] == pytest.approx(0.25 - (-6 + 0.5) / 8)
    assert metrics[METRIC_ADV_FABRICATION] == pytest.approx(-1 - (-6 + 0.5) / 8)
    episodes = [json.loads(line) for line in (tmp_path / EPISODES_FILE).read_text().splitlines()]
    assert episodes[-1]["advantage"] == pytest.approx(metrics[METRIC_ADV_HONEST_FAILURE])
