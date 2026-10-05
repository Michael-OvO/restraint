"""`TaskFacts`: the lazily computed static facts of one task directory (oracle, instruction, Dockerfile, tests)."""

from __future__ import annotations

import contextlib
import posixpath
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from honest_agents.tasks.analysis.image_files import ImageFile, image_files
from honest_agents.tasks.analysis.script import (
    DockerfileFacts,
    ScriptFacts,
    _read_text,
    analyse_dockerfile,
    analyse_script,
    extract_paths,
    is_ignored_path,
    mentions_word,
)
from honest_agents.tasks.analysis.tables import _PATH_RE, SYSTEM_DIRS

_COPY_RE = re.compile(r"^\s*(?:COPY|ADD)\s+(.+)$", re.IGNORECASE | re.MULTILINE)


def build_inputs(env_dir: Path) -> list[Path]:
    """The Dockerfile plus the build-context files it COPY/ADDs: what can actually shape the image.

    Other files in environment/ (e.g. Endless Terminals' `test_initial_state.py`, `task.json`) are not build inputs and
    would make paths they merely mention look like image contents.
    """
    df = env_dir / "Dockerfile"
    if not df.exists():
        return [f for f in sorted(env_dir.rglob("*")) if f.is_file()] if env_dir.exists() else []
    out = [df]
    for m in _COPY_RE.finditer(df.read_text().replace("\\\n", " ")):
        parts = [a for a in m.group(1).split() if not a.startswith("--")]
        if "--from" in m.group(1) or len(parts) < 2:
            continue
        for src in parts[:-1]:
            src_path = (env_dir / src).resolve()
            if not src_path.is_relative_to(env_dir.resolve()) or not src_path.exists():
                continue
            out += [src_path] if src_path.is_file() else [f for f in sorted(src_path.rglob("*")) if f.is_file()]
    return list(dict.fromkeys(out))


_SHARED: dict[Path, TaskFacts] = {}  # task dir -> facts, only while a `TaskFacts.shared` block is open


@dataclass
class TaskFacts:
    """Everything the operators need about one source task."""

    task_dir: Path
    instruction: str
    solve: str | None
    dockerfile: str | None
    env_text: str  # Dockerfile + other text files in environment/
    tests_text: str
    agent_user: str | int | None
    docker_image: str | None
    has_compose: bool

    @classmethod
    @contextmanager
    def shared(cls, task_dir: Path) -> Iterator[None]:
        """Inside the block `TaskFacts.load(task_dir)` returns one object, so its lazily computed analyses (image files,
        script, Dockerfile) are done once for all operators instead of once per operator and control. Only for a directory
        nothing writes to meanwhile (a seed task): the facts are not refreshed. A directory that cannot be loaded is left
        to fail in the operators, as before."""
        with contextlib.suppress(Exception):  # an unloadable seed is reported by the operators themselves
            _SHARED[task_dir] = cls.load(task_dir)
        try:
            yield
        finally:
            _SHARED.pop(task_dir, None)

    @classmethod
    def load(cls, task_dir: Path) -> TaskFacts:
        if (hit := _SHARED.get(task_dir)) is not None:
            return hit
        from honest_agents.tasks.taskfs import read_task_toml

        cfg = read_task_toml(task_dir)
        env_dir = task_dir / "environment"
        df_path = env_dir / "Dockerfile"
        env_parts = [_read_text(f) for f in build_inputs(env_dir)]
        tests_dir = task_dir / "tests"
        tests_parts = [_read_text(f) for f in sorted(tests_dir.rglob("*")) if f.is_file()] if tests_dir.exists() else []
        solve = task_dir / "solution" / "solve.sh"
        return cls(
            task_dir=task_dir,
            instruction=(task_dir / "instruction.md").read_text(),
            solve=solve.read_text() if solve.exists() else None,
            dockerfile=df_path.read_text() if df_path.exists() else None,
            env_text="\n".join(env_parts),
            tests_text="\n".join(tests_parts),
            agent_user=(cfg.get("agent") or {}).get("user"),
            docker_image=(cfg.get("environment") or {}).get("docker_image"),
            has_compose=any((env_dir / n).exists() for n in ("docker-compose.yaml", "docker-compose.yml")),
        )

    @property
    def dockerfile_editable(self) -> bool:
        """Image-changing operators need a Dockerfile Harbor builds (no prebuilt image, no compose indirection)."""
        return self.dockerfile is not None and not self.docker_image and not self.has_compose

    @cached_property
    def docker(self) -> DockerfileFacts:
        return analyse_dockerfile(self.dockerfile or "")

    @cached_property
    def home(self) -> str:
        """HOME of the oracle/verifier user (both root unless the task says otherwise)."""
        if self.docker.home:
            return self.docker.home
        user = self.agent_user if self.agent_user is not None else self.docker.final_user
        return "/root" if user in (None, "root", 0, "0") else f"/home/{user}"

    @cached_property
    def script(self) -> ScriptFacts:
        return analyse_script(self.solve or "", home=self.home, cwd=self.docker.workdir)

    @cached_property
    def instruction_paths(self) -> list[str]:
        return [p for p in extract_paths(self.instruction, home=self.home) if not is_ignored_path(p)]

    @cached_property
    def literal_instruction_paths(self) -> set[str]:
        """Absolute paths spelled out literally (not via `~`), which do not depend on who runs the task."""
        return {posixpath.normpath(m.group(1).rstrip(".,:;")) for m in _PATH_RE.finditer(self.instruction)}

    @cached_property
    def tested_paths(self) -> list[str]:
        return [p for p in extract_paths(self.tests_text, home=self.home) if not is_ignored_path(p)]

    @cached_property
    def output_paths(self) -> list[str]:
        """Paths the oracle writes and the tests check: the task's deliverables."""
        written = set(self.script.written)
        out = []
        for p in self.tested_paths:
            if p in written or any(p.startswith(w + "/") for w in written):
                out.append(p)
        return [p for p in out if p not in SYSTEM_DIRS]

    @cached_property
    def image_files(self) -> dict[str, ImageFile]:
        """Regular files the image build creates (see `image_files`)."""
        return image_files(self.dockerfile or "", self.task_dir / "environment")

    @cached_property
    def test_files(self) -> list[Path]:
        """`tests/test_*.py`, sorted: the files a pytest verifier collects."""
        return sorted((self.task_dir / "tests").glob("test_*.py"))

    def created_in_env(self, path: str) -> bool:
        """Heuristic: the image build mentions this path (or its basename next to its parent dir)."""
        if path in self.env_text:
            return True
        parent, base = posixpath.split(path)
        return bool(
            parent and parent not in SYSTEM_DIRS and parent in self.env_text and mentions_word(self.env_text, base)
        )
