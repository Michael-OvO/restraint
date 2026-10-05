"""Shared fixtures. Everything here runs on a macOS laptop without Docker: sandboxes are host subprocesses."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from honest_agents import grading, harness
from honest_agents.verifier import runner as verifier_runner
from tests.fakes import (
    REPORT_VAR,
    SCRATCH_VAR,
    TESTS_VAR,
    VERIFIER_VAR,
    LocalSandbox,
    make_test_bundle,
)

# GNU `timeout [-k DUR] [-s SIG] DURATION CMD...` for macOS, which ships none: exit 124 on timeout, like the real one.
_TIMEOUT_SHIM = r"""#!/usr/bin/perl
while (@ARGV && $ARGV[0] =~ /^-/) { my $o = shift @ARGV; shift @ARGV if $o eq '-k' || $o eq '-s'; }
my $secs = shift @ARGV;
my $pid = fork();
if (!$pid) { setpgrp(0, 0); exec @ARGV or exit 127; }
$SIG{ALRM} = sub { kill 'KILL', -$pid; waitpid($pid, 0); exit 124; };
alarm $secs;
waitpid($pid, 0);
exit($? & 127 ? 128 + ($? & 127) : $? >> 8);
"""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "docker: needs a Docker daemon (Linux training node)")
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU")


@pytest.fixture(scope="session")
def shim_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory of stand-ins for GNU tools the grader relies on but macOS lacks (currently only `timeout`)."""
    d = tmp_path_factory.mktemp("shim-bin")
    if shutil.which("timeout", path=grading.CLEAN_PATH) is None and shutil.which("timeout") is None:
        (d / "timeout").write_text(_TIMEOUT_SHIM)
        (d / "timeout").chmod(0o755)
    return d


@pytest.fixture(scope="session")
def verifier_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A pytest bundle for the hermetic verifier, built offline from this venv's pytest (see `make_test_bundle`)."""
    return make_test_bundle(tmp_path_factory.mktemp("verifier-bundle"))


@pytest.fixture
def grading_paths(monkeypatch: pytest.MonkeyPatch, shim_bin: Path, verifier_bundle: Path) -> None:
    """Point the grader's absolute /tests, /logs/verifier, scratch dir and report file at per-sandbox shell variables.

    Each `LocalSandbox` binds the variables to directories under its own root, so concurrently graded sandboxes (a TRL
    batch) never share a tests/verifier dir, exactly like separate containers. It also makes `timeout` available on
    macOS, points the hermetic verifier at the offline test bundle and at this venv's interpreter (a laptop has no
    `python3` under /usr/bin that is guaranteed to work).
    """
    monkeypatch.setattr(grading, "TESTS_DIR", f"${TESTS_VAR}")
    monkeypatch.setattr(grading, "VERIFIER_DIR", f"${VERIFIER_VAR}")
    monkeypatch.setattr(grading, "SCRATCH_DIR", f"${SCRATCH_VAR}")
    monkeypatch.setattr(grading, "REPORT_FILE", f"${REPORT_VAR}")
    monkeypatch.setattr(grading, "CLEAN_PATH", f"{shim_bin}:{grading.CLEAN_PATH}")
    monkeypatch.setattr(verifier_runner, "TESTS_DIR", f"${TESTS_VAR}")
    monkeypatch.setattr(verifier_runner, "VERIFIER_DIR", f"${VERIFIER_VAR}")
    monkeypatch.setattr(verifier_runner, "SCRATCH_DIR", f"${SCRATCH_VAR}")
    monkeypatch.setenv("PATH", f"{shim_bin}{os.pathsep}{os.environ.get('PATH', '')}")  # LocalSandbox copies os.environ
    monkeypatch.setenv("HONEST_VERIFIER_BUNDLE", str(verifier_bundle))

    async def venv_python(sb) -> str:
        return sys.executable

    monkeypatch.setattr(grading, "resolve_python", venv_python)
    monkeypatch.setattr(verifier_runner, "resolve_python", venv_python)

    # `stop_background` kills every process but init and itself: against a sandbox that is a plain host subprocess that
    # would be the developer's own session. Tests of the kill script aim `grading.kill_script` at a fake /proc instead.
    async def no_kill(sb, cfg):
        return None

    monkeypatch.setattr(grading, "stop_background", no_kill)


@pytest.fixture
def sandbox(tmp_path: Path, grading_paths: None) -> LocalSandbox:
    return LocalSandbox(tmp_path / "sandbox")


@pytest.fixture(autouse=True)
def _cold_image_cache():
    """Every test starts with no task image marked as built (`harness._WARM` is process-wide)."""
    harness._WARM.clear()
    yield
    harness._WARM.clear()


# ── a private copy of the interpreter (hermetic-verifier toolchain tests) ───────────────────────────────────────


@pytest.fixture(scope="session")
def interpreter_template(tmp_path_factory) -> Path | None:
    """`<t>/bin/python3` and `<t>/lib/pythonX.Y/` copied from this machine, or None if the copy cannot run on its own."""
    t = tmp_path_factory.mktemp("interp-template")
    exe = Path(os.path.realpath(sys.executable))
    stdlib = Path(sysconfig.get_path("stdlib"))
    tag = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    (t / "bin").mkdir()
    shutil.copy2(exe, t / "bin" / "python3")
    shutil.copytree(
        stdlib,
        t / "lib" / tag,
        symlinks=True,
        ignore=shutil.ignore_patterns("site-packages", "__pycache__", "test*"),
    )
    probe = subprocess.run(
        [
            str(t / "bin" / "python3"),
            "-I",
            "-S",
            "-c",
            "import encodings, json, hashlib, subprocess; print('ok')",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return t if probe.stdout.strip() == "ok" else None


@pytest.fixture
def private_python(interpreter_template, tmp_path) -> Path:
    """A fresh copy of the template per test: a test may edit its stdlib or plant files next to it."""
    if interpreter_template is None:
        pytest.skip("this interpreter cannot run from a copy (dynamic libpython outside the prefix)")
    dest = tmp_path / "pyhome"
    shutil.copytree(interpreter_template, dest, symlinks=True)
    return dest
