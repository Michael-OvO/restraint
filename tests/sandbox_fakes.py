"""Docker-free stand-ins for Harbor sandboxes.

`LocalSandbox` satisfies `honest_agents.grading.AsyncSandbox` by running each command as a host subprocess inside a
private root directory. `FakeHarborEnvironment` adds the slice of `harbor.environments.base.BaseEnvironment` that
`HonestHarborEnv` drives (start/stop/healthcheck/network policy/upload_file), so the real `_boot_episode` /
`_grade_episode` code paths run end-to-end once `EnvironmentFactory.create_environment_from_config` is patched.

Differences from a container that tests must respect: `user` is recorded but ignored (no setuid on a laptop), and the
grader's absolute `/tests` and `/logs/verifier` are redirected through the `$HA_TESTS_DIR` / `$HA_VERIFIER_DIR` shell
variables (see the `grading_paths` fixture), which every sandbox binds to directories under its own root.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import shutil
import signal
import string
import textwrap
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TESTS_VAR = "HA_TESTS_DIR"
VERIFIER_VAR = "HA_VERIFIER_DIR"
SCRATCH_VAR = "HA_SCRATCH_DIR"  # parent of the hermetic verifier's random per-run directory
REPORT_VAR = "HA_REPORT_FILE"  # stands in for harness.REPORT_FILE
# The grader's unpatched cleanup would `rm -rf` real host directories (any flag spelling, `--`, or later operand).
_HOST_DANGER = re.compile(r"\brm\b[^;&|\n]*\s/(tests|logs)(/|\s|$)")


@dataclass
class LocalExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int


@dataclass
class ExecCall:
    command: str
    cwd: str | None
    env: dict[str, str] | None
    timeout_sec: int | None
    user: str | int | None


class LocalSandbox:
    """Runs commands with `<shell> -c` on the host, cwd = `root` unless given.

    Mirrors Harbor's Docker backend where it matters to the grader: env vars are merged over a persistent base env, and a
    timeout kills the whole process group and raises `RuntimeError` (as `DockerEnvironment._run_docker_compose_command`
    does). Background processes started by a command outlive it, as they do inside a running container.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        env: dict[str, str] | None = None,
        shell: str = "/bin/bash",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tests_dir = self.root / "tests"
        self.verifier_dir = self.root / "logs" / "verifier"
        self.scratch_dir = self.root / "scratch"
        self.report_file = self.root / "logs" / "agent" / "honest_report.json"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.shell = shell
        self.env: dict[str, str] = {
            **os.environ,
            TESTS_VAR: str(self.tests_dir),
            VERIFIER_VAR: str(self.verifier_dir),
            SCRATCH_VAR: str(self.scratch_dir),
            REPORT_VAR: str(self.report_file),
            **(env or {}),
        }
        self.calls: list[ExecCall] = []

    def expand(self, path: str) -> str:
        """Resolve `$HA_*` variables and sandbox-relative paths the way the sandbox shell would."""
        p = string.Template(str(path)).safe_substitute(self.env)
        return p if os.path.isabs(p) else str(self.root / p)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> LocalExecResult:
        if _HOST_DANGER.search(command):
            raise AssertionError(f"refusing to run on the host (grading_paths fixture not applied?): {command}")
        self.calls.append(ExecCall(command, cwd, env, timeout_sec, user))
        proc = await asyncio.create_subprocess_exec(
            self.shell,
            "-c",
            command,
            cwd=self.expand(cwd) if cwd else str(self.root),
            env={**self.env, **(env or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        except TimeoutError:
            _killpg(proc.pid)
            await proc.wait()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
        return LocalExecResult(out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        shutil.copytree(source_dir, self.expand(target_dir), dirs_exist_ok=True)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        dst = Path(self.expand(target_path))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dst)


def _killpg(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class FakeHarborEnvironment(LocalSandbox):
    """The `BaseEnvironment` surface `HonestHarborEnv` uses, on top of `LocalSandbox`."""

    def __init__(
        self,
        root: Path | str,
        *,
        factory_kwargs: dict[str, Any] | None = None,
        on_start: Callable[[Path], None] | None = None,
        start_gate: threading.Event | None = None,
        fail_start: Exception | None = None,
        fail_healthcheck: Exception | None = None,
        dynamic_network_policy: bool = True,
        **kw: Any,
    ):
        super().__init__(root, **kw)
        self.factory_kwargs = factory_kwargs or {}
        self.network_policy = self.factory_kwargs.get("network_policy")
        self.network_policy_calls: list[Any] = []
        self.on_start = on_start
        self.start_gate = start_gate
        self.fail_start = fail_start
        self.fail_healthcheck = fail_healthcheck
        # Docker only supports switching policies at runtime when the egress sidecar is enabled.
        self.dynamic_network_policy = dynamic_network_policy
        self.started = False
        self.healthchecks = 0
        self.stop_calls: list[bool] = []

    @property
    def os(self):
        from harbor.models.task.config import TaskOS

        return TaskOS.LINUX

    async def start(self, force_build: bool) -> None:
        if self.start_gate is not None:
            await asyncio.to_thread(self.start_gate.wait)
        if self.fail_start is not None:
            raise self.fail_start
        if self.on_start is not None:
            self.on_start(self.root)
        self.started = True

    async def stop(self, delete: bool) -> None:
        self.stop_calls.append(delete)

    async def run_healthcheck(self, healthcheck=None) -> None:
        self.healthchecks += 1
        if self.fail_healthcheck is not None:
            raise self.fail_healthcheck

    async def set_network_policy(self, network_policy) -> None:
        """Same contract as `BaseEnvironment.set_network_policy`: no-op if unchanged, else needs dynamic support."""
        self.network_policy_calls.append(network_policy)
        if network_policy == self.network_policy:
            return
        if not self.dynamic_network_policy:
            raise ValueError("fake environment cannot change network policy after start.")
        self.network_policy = network_policy

    @property
    def stopped(self) -> bool:
        return bool(self.stop_calls)


@dataclass
class FakeFactory:
    """Replacement for `EnvironmentFactory.create_environment_from_config`; records every environment it creates."""

    base: Path
    make_kwargs: Callable[[int], dict[str, Any]] = lambda i: {}
    # Per-task overrides keyed by the task directory name (parent of `environment_dir`). Boots run concurrently, so creation
    # order — and hence the `make_kwargs` index — is not the order in which envs were reset.
    by_name: dict[str, dict[str, Any]] = field(default_factory=dict)
    created: list[FakeHarborEnvironment] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __call__(self, **kwargs: Any) -> FakeHarborEnvironment:
        with self._lock:
            i = len(self.created)
            opts = self.make_kwargs(i) | self.by_name.get(_task_name(kwargs), {})
            env = FakeHarborEnvironment(self.base / f"sandbox-{i}", factory_kwargs=kwargs, **opts)
            self.created.append(env)
        return env

    def named(self, name: str) -> list[FakeHarborEnvironment]:
        return [e for e in self.created if _task_name(e.factory_kwargs) == name]


def _task_name(kwargs: dict[str, Any]) -> str | None:
    """The task's directory name. Harbor environment names are opaque (L3), so find the task by its environment_dir."""
    env_dir = kwargs.get("environment_dir")
    return Path(env_dir).parent.name if env_dir else None


def install_fake_factory(monkeypatch, factory: FakeFactory) -> FakeFactory:
    """Patch Harbor's factory. Every call is still bound against the *real* signatures, so a kwarg Harbor would reject
    (e.g. after a version bump) fails here instead of only on the Docker node."""
    from harbor.environments.base import BaseEnvironment
    from harbor.environments.factory import EnvironmentFactory

    real = inspect.signature(EnvironmentFactory.create_environment_from_config)
    env_init = inspect.signature(BaseEnvironment.__init__)
    named_init = {
        n for n, p in env_init.parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and n != "self"
    }

    def checked(**kwargs: Any) -> FakeHarborEnvironment:
        real.bind(**kwargs)
        extra = set(kwargs) - set(real.parameters) - named_init
        assert not extra, f"kwargs no BaseEnvironment accepts by name: {sorted(extra)}"
        return factory(**kwargs)

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", staticmethod(checked))
    return factory


# A test.sh that also runs unmodified under real Harbor (falls back to /logs/verifier there).
PASS_IF_HELLO = """\
#!/bin/bash
V="${HA_VERIFIER_DIR:-/logs/verifier}"
if [ "$(cat hello.txt 2>/dev/null)" = "${EXPECTED:-hi}" ]; then echo 1 > "$V/reward.txt"; else echo 0 > "$V/reward.txt"; fi
"""


def make_task_dir(
    path: Path,
    *,
    honest: str = "",
    test_sh: str = PASS_IF_HELLO,
    instruction: str = "Write `hi` to hello.txt.\n",
    extra_toml: str = "",
) -> Path:
    """Minimal Harbor 0.24 task directory (`TaskConfig` schema 1.4) carrying a `[metadata.honest]` table."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "task.toml").write_text(
        textwrap.dedent(
            """\
            schema_version = "1.4"

            [task]
            name = "honest-tests/{name}"

            [metadata.honest]
            {honest}

            [verifier]
            timeout_sec = 60.0

            [agent]
            timeout_sec = 60.0

            [environment]
            build_timeout_sec = 60.0
            """
        ).format(name=path.name, honest=textwrap.dedent(honest).strip())
        + textwrap.dedent(extra_toml)
    )
    (path / "instruction.md").write_text(instruction)
    (path / "environment").mkdir(exist_ok=True)
    (path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
    (path / "tests").mkdir(exist_ok=True)
    (path / "tests" / "test.sh").write_text(test_sh)
    (path / "tests" / "test.sh").chmod(0o755)
    return path


# A pytest-verifier task (docs/design.md section 3): `verifier = "pytest"`, tests that need only the standard library and pytest,
# and a `test.sh` that stays valid for `harbor run` (our grader never runs it).
PYTEST_VERIFIER_TOML = 'verifier = "pytest"'
PASS_IF_HELLO_PY = """\
from pathlib import Path


def test_hello_written():
    assert Path("hello.txt").read_text().strip() == "hi"
"""


def make_pytest_task_dir(
    path: Path,
    *,
    honest: str = "",
    test_py: str = PASS_IF_HELLO_PY,
    instruction: str = "Write `hi` to hello.txt.\n",
    extra_toml: str = "",
) -> Path:
    """Like `make_task_dir`, with `verifier = "pytest"` and `tests/test_task.py` instead of a reward-writing script."""
    honest = (textwrap.dedent(honest).strip() + "\n" + PYTEST_VERIFIER_TOML).strip()
    d = make_task_dir(path, honest=honest, instruction=instruction, extra_toml=extra_toml)
    (d / "tests" / "test_task.py").write_text(test_py)
    return d


_BUNDLE_PACKAGES = ("_pytest", "pytest", "pluggy", "iniconfig", "packaging", "pygments")


def make_test_bundle(dest: Path) -> Path:
    """A verifier bundle for unit tests, copied from the pure-Python pytest in this venv (no network, no build step).

    The real bundle (`scripts/build_verifier_bundle.sh`, pytest 8.4.1 for Python 3.9) is exercised by a test that skips
    when it has not been built. Same layout and digest as the real one.
    """
    import importlib.util
    import json

    from honest_agents.verifier._hv_run import tree_digest

    site = dest / "site"
    site.mkdir(parents=True, exist_ok=True)
    for name in _BUNDLE_PACKAGES:
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin, f"{name} is not installed in this environment"
        shutil.copytree(
            Path(spec.origin).parent,
            site / name,
            ignore=shutil.ignore_patterns("__pycache__"),
            dirs_exist_ok=True,
        )
    py_shim = importlib.util.find_spec("py")
    if py_shim is not None and py_shim.origin and py_shim.origin.endswith("py.py"):
        shutil.copy2(py_shim.origin, site / "py.py")
    (dest / "BUNDLE.json").write_text(json.dumps({"pytest": "test-copy", "digest": tree_digest(str(site))}))
    return dest


def fake_proc(root: Path, pids: dict[int, int]) -> Path:
    """A directory shaped like /proc: `<pid>/status` with a `PPid:` line (value: parent pid)."""
    root.mkdir(parents=True, exist_ok=True)
    for pid, ppid in pids.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / "status").write_text(
            f"Name:\tx\nUmask:\t0022\nState:\tS (sleeping)\nPPid:\t{ppid}\nUid:\t0\n"
        )
    (root / "1").mkdir(exist_ok=True)
    (root / "1" / "comm").write_text("sleep\n")
    return root


def bare_docker_env(cls: type) -> Any:
    """A Harbor `DockerEnvironment` (or a subclass) with no compose project: Harbor's own `exec` / `service_exec` /
    `ensure_dirs` / upload code runs down to `_run_docker_compose_command`, which the test overrides or patches to record
    the compose argv instead of running it (DEF-16 / DEF-19 umask pin)."""
    import contextvars
    import logging
    from types import SimpleNamespace

    from harbor.environments.docker.docker_unix import UnixOps

    env = object.__new__(cls)
    env._platform = UnixOps(env)
    env._is_windows_container = False
    env.task_env_config = SimpleNamespace(workdir=None)
    env.default_user = None
    env._persistent_env = {}
    env._output_callbacks = contextvars.ContextVar("output_callbacks", default=())
    env._exec_env_overlays = contextvars.ContextVar("exec_env_overlays", default=())
    env.logger = logging.getLogger("bare_docker_env")
    return env
