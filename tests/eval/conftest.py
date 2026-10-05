"""Shared pytest configuration for tests/eval."""

import os

import pytest


@pytest.fixture(autouse=True)
def gnu_timeout_on_path(monkeypatch: pytest.MonkeyPatch, shim_bin) -> None:
    """The eval agents wrap every command in GNU `timeout -k` (as `HonestHarborEnv.bash` does), which macOS lacks."""
    monkeypatch.setenv("PATH", f"{shim_bin}{os.pathsep}{os.environ.get('PATH', '')}")
