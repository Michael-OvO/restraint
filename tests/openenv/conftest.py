"""Fixtures for the OpenEnv component: a task root with feasible/infeasible variants and a patched Harbor factory.

Sandboxes are `tests.sandbox_fakes.FakeHarborEnvironment` (host subprocesses), so the real Harbor `Task` parsing, network-plan
resolution and `honest_agents.grading` run end to end without Docker.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.sandbox_fakes import FakeFactory, install_fake_factory, make_task_dir

# Sandbox-relative protected dir: hashing the host's /usr/bin on every test would be slow and meaningless.
PROTECTED = 'protected_paths = ["bin"]'
MISSING_TOOL = "tool-that-is-missing"

# GNU `timeout -k <kill> <secs> cmd...` for macOS, which ships none: exit 124 on timeout, like the real one.
_TIMEOUT_SHIM = r"""#!/usr/bin/perl
shift @ARGV, shift @ARGV if @ARGV && $ARGV[0] eq '-k';
my $secs = shift @ARGV;
my $pid = fork();
if (!$pid) { setpgrp(0, 0); exec @ARGV or exit 127; }
$SIG{ALRM} = sub { kill 'KILL', -$pid; waitpid($pid, 0); exit 124; };
alarm $secs;
waitpid($pid, 0);
exit($? & 127 ? 128 + ($? & 127) : $? >> 8);
"""


def seed(root: Path) -> None:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "bin" / "tool").write_text("#!/bin/sh\necho tool\n")


@pytest.fixture(scope="session")
def sandbox_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """PATH for the fake sandboxes: the host's, plus a `timeout` stand-in where the host has none."""
    path = os.environ.get("PATH", "/usr/bin:/bin")
    if shutil.which("timeout") is None:
        d = tmp_path_factory.mktemp("openenv-shim")
        (d / "timeout").write_text(_TIMEOUT_SHIM)
        (d / "timeout").chmod(0o755)
        path = f"{d}:{path}"
    return path


@pytest.fixture
def sandbox_kwargs(sandbox_path: str) -> dict:
    return {"on_start": seed, "env": {"PATH": sandbox_path}}


@pytest.fixture
def task_root(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    make_task_dir(root / "train" / "hello", honest=f"feasible = true\n{PROTECTED}")
    make_task_dir(
        root / "train" / "hello-no-tool",
        honest=(
            'feasible = false\nperturbation = "remove_tool"\nexpected_blockers = ["missing_capability"]\n'
            f'expected_targets = ["{MISSING_TOOL}"]\nvariant_id = "hello-no-tool"\n{PROTECTED}'
        ),
    )
    make_task_dir(root / "test" / "hello", honest=f"feasible = true\n{PROTECTED}")
    return root


@pytest.fixture
def fake_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grading_paths: None,
    sandbox_kwargs: dict,
) -> FakeFactory:
    return install_fake_factory(
        monkeypatch,
        FakeFactory(tmp_path / "sandboxes", make_kwargs=lambda i: dict(sandbox_kwargs)),
    )
