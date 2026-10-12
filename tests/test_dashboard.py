"""train.dashboard against a stand-in for the wandb module: no network, no wandb install needed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from honest_agents import memory
from honest_agents.callbacks import TRAIN_LOG_FILE, JsonlLogger, ValidationProbe
from honest_agents.train import dashboard, health
from honest_agents.train.config import TrainRunConfig
from honest_agents.train.dashboard import STEP_KEY, WandbMirror

SECRET = "0123456789abcdef-not-a-real-key"


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.alerts: list[tuple[str, str]] = []
        self.finished: int | None = None

    def log(self, data: dict) -> None:
        self.logged.append(dict(data))

    def alert(self, title: str, text: str) -> None:
        self.alerts.append((title, text))

    def finish(self, exit_code: int = 0) -> None:
        self.finished = exit_code


class FakeWandb:
    def __init__(self) -> None:
        self.run: FakeRun | None = None
        self.init_kwargs: dict | None = None

    def init(self, **kwargs) -> FakeRun:
        self.init_kwargs = kwargs
        self.run = FakeRun()
        return self.run


def state(step: int) -> SimpleNamespace:
    return SimpleNamespace(global_step=step, is_world_process_zero=True, log_history=[])


def train_logs(i: int, **over) -> dict:
    logs = {"loss": 0.05, "reward": 0.4, "honest/task_success": 0.6, "honest/n_control": 33, "honest/n_scored": 64,
            "honest/control/verified_success_rate": 0.8, "step_time": 400.0}  # fmt: skip
    logs.update(over)
    return logs


@pytest.fixture
def fake_memory(monkeypatch):
    monkeypatch.setattr(memory, "rss_gb", lambda *a, **k: 12.5)
    monkeypatch.setattr(memory, "mem_available_gb", lambda *a, **k: 40.0)


def test_wandb_mode_disabled_drops_wandb_from_report_to():
    off, on = {"WANDB_MODE": "disabled"}, {}
    assert dashboard.resolve_report_to("wandb", off) == "none"
    assert dashboard.resolve_report_to(["wandb", "tensorboard"], off) == ["tensorboard"]
    assert dashboard.resolve_report_to("wandb", on) == "wandb"
    assert dashboard.resolve_report_to("none", off) == "none"
    assert dashboard.wandb_enabled(["wandb"], on) and not dashboard.wandb_enabled(["wandb"], off)
    assert not dashboard.wandb_enabled("none", on) and not dashboard.wandb_enabled([], on)


def test_the_run_id_survives_a_resume_but_not_a_fresh_start(tmp_path: Path):
    first = dashboard.run_id(tmp_path, resuming=False)
    assert dashboard.run_id(tmp_path, resuming=True) == first
    assert dashboard.run_id(tmp_path, resuming=False) != first


def test_init_names_the_run_after_the_iteration_and_takes_the_entity_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "michael_ovo")
    monkeypatch.setenv("WANDB_API_KEY", SECRET)
    cfg = TrainRunConfig.from_dict({"wandb": {"name": "it11-inspect-first", "group": "it11", "tags": ["colocate"]}})
    fake = FakeWandb()
    run = dashboard.init_run(cfg, tmp_path, resuming=False, wandb_module=fake)
    kw = fake.init_kwargs
    assert run is fake.run
    assert (kw["project"], kw["entity"], kw["name"], kw["group"], kw["tags"]) == (
        "honest-agents", "michael_ovo", "it11-inspect-first", "it11", ["colocate"])  # fmt: skip
    assert kw["id"] == (tmp_path / dashboard.RUN_ID_FILE).read_text().strip() and kw["resume"] == "allow"
    assert SECRET not in json.dumps(kw, default=str)  # the key is wandb's to read, never ours to pass or store
    monkeypatch.delenv("WANDB_ENTITY")
    assert dashboard.init_run(TrainRunConfig(), tmp_path, resuming=True, wandb_module=fake) is fake.run
    assert fake.init_kwargs["entity"] is None and fake.init_kwargs["name"] == "honest-a0.25-qwen3-4b-s42"  # run_name


def test_the_mirror_sends_only_what_trl_does_not_log(tmp_path, fake_memory):
    fake = FakeWandb()
    fake.init()
    run_log = JsonlLogger(tmp_path)
    mirror = WandbMirror(run_log, wandb_module=fake)
    for cb in (run_log, mirror):
        cb.on_train_begin(None, state(0), None)
    for i in (1, 2):
        logs = train_logs(i)
        run_log.on_log(None, state(i), None, logs=logs)
        mirror.on_log(None, state(i), None, logs=logs)
    _, second = fake.run.logged
    assert not set(train_logs(1)) & set(second)  # TRL's WandbCallback already logs these
    assert second["mem/rss_gb"] == 12.5 and second["mem/available_gb"] == 40.0 and "wall_sec_since_last_log" in second
    assert "step" not in second and "time" not in second and second[STEP_KEY] == 2
    assert {f"health/{c}" for c in health.CHECKS} | {"health/verdict"} <= set(second)
    assert second["health/verdict"] == 0.0 and fake.run.summary["health/findings"] == "OK"
    mirror.on_log(None, state(2), None, logs={"eval_honest/task_success": 0.7})  # an evaluation log: TRL has it
    assert len(fake.run.logged) == 2


def test_a_sustained_control_drop_reaches_the_dashboard_as_a_fail_and_one_alert(tmp_path, fake_memory):
    fake = FakeWandb()
    fake.init()
    run_log = JsonlLogger(tmp_path)
    mirror = WandbMirror(run_log, wandb_module=fake)
    for cb in (run_log, mirror):
        cb.on_train_begin(None, state(0), None)
    for i in range(1, 15):
        logs = train_logs(i, **({"honest/control/verified_success_rate": 0.5} if i >= 8 else {}))
        run_log.on_log(None, state(i), None, logs=logs)
        mirror.on_log(None, state(i), None, logs=logs)
    last = fake.run.logged[-1]
    assert last["health/capability"] == health.FAIL and last["health/verdict"] == health.FAIL
    assert len(fake.run.alerts) == 1 and fake.run.alerts[0][0].startswith("health FAIL: capability")
    assert "verified success on controls fell" in fake.run.summary["health/findings"]


def test_a_resumed_run_continues_its_health_history_from_the_run_log(tmp_path):
    rows = [{"step": i, "time": 1.0 * i, **train_logs(i)} for i in range(1, 6)]
    (tmp_path / TRAIN_LOG_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    mirror = WandbMirror(JsonlLogger(tmp_path), wandb_module=FakeWandb())
    mirror.on_train_begin(None, state(5), None)
    assert [r["step"] for r in mirror.rows] == [1, 2, 3, 4, 5]


def test_the_probe_choice_is_logged_and_a_dashboard_error_never_stops_the_run(tmp_path, fake_memory, caplog):
    fake = FakeWandb()
    fake.init()
    probe = ValidationProbe(tmp_path, "eval_honest/task_success")
    mirror = WandbMirror(JsonlLogger(tmp_path), probe, wandb_module=fake)
    probe.on_evaluate(None, state(20), None, metrics={"eval_honest/task_success": 0.81})
    mirror.on_evaluate(None, state(20), None, metrics={"eval_honest/task_success": 0.81})
    assert fake.run.logged[-1] == {"probe/best_step": 20.0, "probe/best_value": 0.81, STEP_KEY: 20}
    assert fake.run.summary["probe/best_checkpoint"].endswith("checkpoint-20")

    def broken(data):
        raise ConnectionError("W&B unreachable")

    fake.run.log = broken
    mirror.on_log(None, state(21), None, logs=train_logs(21))  # logs a warning, does not raise
    assert "could not log step 21" in caplog.text


def test_no_dashboard_without_an_open_run(tmp_path, fake_memory):
    fake = FakeWandb()  # wandb.init never ran (e.g. not rank 0)
    mirror = WandbMirror(JsonlLogger(tmp_path), wandb_module=fake)
    mirror.on_log(None, state(1), None, logs=train_logs(1))
    assert fake.run is None and len(mirror.rows) == 1
