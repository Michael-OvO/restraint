"""Concurrency/lifecycle hazards that only bite on a long-running server with real Docker: per-session event loops,
teardown racing in-flight work, and Harbor's loop-bound image-build lock."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("openenv")

from honest_agents.openenv_env.models import HonestAction
from honest_agents.openenv_env.server.environment import HonestEnvironment
from tests.openenv.conftest import seed as _seed
from tests.sandbox_fakes import FakeFactory, install_fake_factory


def _in_thread(fn):
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_close_closes_loop_and_refuses_reuse(task_root: Path, fake_factory) -> None:
    env = HonestEnvironment(tasks_dir=task_root)
    env.reset(task_id="train/hello")
    loop, thread = env._loop, env._thread
    env.close()
    thread.join(timeout=10)
    assert not thread.is_alive() and loop.is_closed()  # no fd/thread leak per session
    assert fake_factory.created[0].stopped
    with pytest.raises(RuntimeError, match="closed"):
        env.step(HonestAction(kind="bash", command="ls"))


def test_close_waits_for_inflight_reset(
    tmp_path: Path, task_root: Path, monkeypatch, grading_paths, sandbox_path: str
) -> None:
    gate = threading.Event()
    factory = install_fake_factory(
        monkeypatch,
        FakeFactory(
            tmp_path / "sb",
            make_kwargs=lambda i: {
                "start_gate": gate,
                "on_start": _seed,
                "env": {"PATH": sandbox_path},
            },
        ),
    )
    env = HonestEnvironment(tasks_dir=task_root)
    try:
        t_reset, res = _in_thread(lambda: env.reset(task_id="train/hello"))
        time.sleep(0.2)
        t_close, _ = _in_thread(env.close)  # e.g. idle reaper / disconnect while the boot is still running
        time.sleep(0.2)
        assert not factory.created[0].stopped  # teardown must not run underneath the boot
    finally:
        gate.set()
    t_reset.join(10)
    t_close.join(10)
    assert "error" not in res and res["value"].kind == "reset"
    assert factory.created[0].stopped  # ...and still happens afterwards


def test_same_task_image_builds_are_serialised(
    tmp_path: Path, task_root: Path, monkeypatch, grading_paths, sandbox_path: str
) -> None:
    gate = threading.Event()
    factory = install_fake_factory(
        monkeypatch,
        FakeFactory(
            tmp_path / "sb",
            make_kwargs=lambda i: (
                {"on_start": _seed, "env": {"PATH": sandbox_path}} | ({"start_gate": gate} if i == 0 else {})
            ),
        ),
    )
    a, b = (
        HonestEnvironment(tasks_dir=task_root),
        HonestEnvironment(tasks_dir=task_root),
    )
    try:
        ta, ra = _in_thread(lambda: a.reset(task_id="train/hello"))
        time.sleep(0.2)
        tb, rb = _in_thread(lambda: b.reset(task_id="train/hello"))
        time.sleep(0.3)
        # No docker_image: Harbor would build under its loop-bound asyncio.Lock, so the second boot must wait for ours.
        assert len(factory.created) == 2 and not factory.created[1].started
        gate.set()
        ta.join(10)
        tb.join(10)
        assert "error" not in ra and "error" not in rb
        assert all(e.started for e in factory.created)
    finally:
        gate.set()
        a.close()
        b.close()
