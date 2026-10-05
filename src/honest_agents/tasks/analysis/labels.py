"""Label strings (words such as `impossible`) in what the container can see."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from honest_agents.tasks.analysis.script import _read_text

# ── label strings in what the container can see ─────────────────────────────────────────────────────────────────

LABEL_RE = re.compile(
    r"(?<![A-Za-z])(?:abstain|control|twin|decoy|impossible|infeasible|perturb|honest|near[-_ ]?miss)", re.IGNORECASE
)
# task.toml tables the harness hands to the container or to Docker (everything but `[metadata]`, which stays host-side).
_CONTAINER_TOML_TABLES = ("task", "agent", "environment", "verifier", "solution")


class LabelLeak(NamedTuple):
    path: str  # relative to the task dir
    line: int
    text: str


def _label_lines(rel: str, text: str, ignore: frozenset[str] = frozenset()) -> list[LabelLeak]:
    return [
        LabelLeak(rel, i, ln.strip()[:200])
        for i, ln in enumerate(text.splitlines(), 1)
        if LABEL_RE.search(ln) and ln.strip() not in ignore
    ]


def label_leaks(task_dir: Path, source_dir: Path | None = None) -> list[LabelLeak]:
    """Lines of an emitted task that carry a label word and can reach the container, comments and file names included.

    Scans every text file under `environment/` (the Docker build context: Dockerfile, comments, scripts, file names)
    and the container-facing tables of task.toml. With `source_dir` (the seed task), lines the seed already had are
    not counted, so a source whose own Dockerfile says "version control" does not block the build; what the operators
    add is what is checked. Labels live only in `[metadata.honest]`, which this scan skips by design.
    """
    import tomllib

    out: list[LabelLeak] = []
    env = task_dir / "environment"
    src_env = source_dir / "environment" if source_dir is not None else None
    for f in sorted(env.rglob("*")) if env.is_dir() else []:
        rel = f.relative_to(task_dir).as_posix()
        out += _label_lines(rel + " (name)", "\n".join(rel.split("/")[1:]))
        if not f.is_file():
            continue
        text = _read_text(f, limit=2_000_000)
        seen: frozenset[str] = frozenset()
        if src_env is not None and (src_env / f.relative_to(env)).is_file():
            seen = frozenset(
                ln.strip() for ln in _read_text(src_env / f.relative_to(env), limit=2_000_000).splitlines()
            )
        out += _label_lines(rel, text, seen)
    toml_path = task_dir / "task.toml"
    if toml_path.is_file():
        try:
            cfg = tomllib.loads(toml_path.read_text())
        except tomllib.TOMLDecodeError:
            cfg = {}
        for table in _CONTAINER_TOML_TABLES:
            if table in cfg:
                out += _label_lines(f"task.toml [{table}]", json.dumps(cfg[table], indent=0, default=str))
    return out
