"""Shared pytest configuration for tests/eval."""

import os

import pytest


@pytest.fixture(autouse=True)
def gnu_timeout_on_path(monkeypatch: pytest.MonkeyPatch, shim_bin) -> None:
    """The eval agents wrap every command in GNU `timeout -k` (as `HonestHarborEnv.bash` does), which macOS lacks."""
    monkeypatch.setenv("PATH", f"{shim_bin}{os.pathsep}{os.environ.get('PATH', '')}")


@pytest.fixture(autouse=True)
def no_docker_network_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_eval.evaluate` checks that Docker can enforce network policies before the first boot (a probe container);
    these tests run over host sandboxes on a laptop that may have no Docker. `evaluate` imports it at call time."""
    import honest_agents.env as env_module

    monkeypatch.setattr(env_module, "preflight_network_policy", lambda *a, **k: False)
