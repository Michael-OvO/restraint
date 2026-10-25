"""The gate's content-addressed result cache (docs/integrity-gate.md, "the result cache"; DEF-15).

One append-only JSONL file per check and schema version under `~/.cache/honest-agents/gate/` (a node:
`/root/.cache/honest-agents/gate/`), outside the repo and the archive. A row is `{check, schema, key, <field>}`: the
verdict and the small outputs the check needs, never images, containers or full episodes. The key is the sha256 of
the task dir contents, of the source files the check imports, of the command list and of the base image digest
(`cache_key`), so a change to one task re-runs that task only and a change to shared code re-runs everything. A
corrupt, torn or old-version row is ignored, never trusted; a later row for the same key wins.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import subprocess
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "honest-agents" / "gate"
SRC = Path(__file__).resolve().parents[1]  # src/honest_agents


def sha256_tree(root: Path) -> str:
    """sha256 over the relative paths and bytes of every file under `root` (sorted, so order-independent)."""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _module_file(name: str) -> Path | None:
    """The file of an `honest_agents` module (a package's `__init__.py`), or None for any other module."""
    parts = name.split(".")
    if parts[0] != SRC.name:
        return None
    p = SRC.joinpath(*parts[1:])
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    return p.with_suffix(".py") if p.with_suffix(".py").is_file() else None


def _imported_modules(path: Path) -> list[str]:
    """Every module name `path` imports, at the top level or inside a function (a lazy import still shapes a verdict),
    with each `from m import x` also tried as the submodule `m.x` and every parent package of a name."""
    pkg = [SRC.name, *path.relative_to(SRC).with_suffix("").parts][:-1]  # the package a relative import starts from
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = pkg[: len(pkg) - node.level + 1] if node.level else []
            mod = ".".join([*base, *([node.module] if node.module else [])])
            names += [mod, *(f"{mod}.{a.name}" for a in node.names)]
    return [".".join(n.split(".")[: i + 1]) for n in names for i in range(len(n.split(".")))]


def import_closure(*entries: Path) -> tuple[Path, ...]:
    """The `honest_agents` source files `entries` import, directly or through each other (sorted, entries included):
    what a cache key's source hash covers, so no imported module can change under a cached verdict."""
    seen: set[Path] = set()
    stack = [Path(e).resolve() for e in entries]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack += [q for q in map(_module_file, _imported_modules(p)) if q is not None and q not in seen]
    return tuple(sorted(seen))


def library_versions(*dists: str) -> dict[str, str]:
    """Installed versions of third-party distributions a verdict depends on (Harbor, TRL); '' when not installed."""
    out = {}
    for d in dists:
        try:
            out[d] = importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError:
            out[d] = ""
    return out


def base_image(task_dir: Path) -> str:
    """The `FROM` image of the task's Dockerfile ('' when there is none)."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return ""
    for line in dockerfile.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "FROM":
            return next((x for x in parts[1:] if not x.startswith("--")), "")
    return ""


_DIGESTS: dict[str, str] = {}


def docker_image_digest(image: str) -> str:
    """Local image id of `image` (`docker image inspect`), memoised per process; 'unknown' when Docker cannot say."""
    if not image:
        return ""
    if image not in _DIGESTS:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            _DIGESTS[image] = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"
        except (OSError, subprocess.SubprocessError):
            _DIGESTS[image] = "unknown"
    return _DIGESTS[image]


def cache_key(check: str, schema: int, **parts: Any) -> str:
    """sha256 of the check, its schema version and its key parts (task hash, source hash, commands, image digest...)."""
    payload = json.dumps({"check": check, "schema": schema, **parts}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class ResultCache:
    """Append-only JSONL of `{check, schema, key, <field>}` rows: `<cache_dir>/<check>.v<schema>.jsonl`. With
    `cache_dir=None` it is an in-memory cache that writes nothing (`--no-cache`)."""

    def __init__(self, check: str, schema: int, cache_dir: Path | None, *, field: str = "value"):
        self.check, self.schema, self.field = check, schema, field
        self.path = None if cache_dir is None else cache_dir / f"{check}.v{schema}.jsonl"
        self._rows: dict[str, Any] = {}
        self._lock = threading.Lock()  # one appended row at a time: pool threads write rows larger than a buffer
        self.bytes_written = 0
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn or corrupt row is never trusted
                if (
                    isinstance(row, dict)
                    and row.get("check") == check
                    and row.get("schema") == schema
                    and isinstance(row.get("key"), str)
                    and field in row
                    and self._valid(row[field])
                ):
                    self._rows[row["key"]] = row[field]

    def _valid(self, value: Any) -> bool:
        """Subclasses narrow the type of a stored value; a row that fails is ignored."""
        return True

    def get(self, key: str) -> Any | None:
        return self._rows.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._rows

    def put(self, key: str, value: Any) -> None:
        line = json.dumps({"check": self.check, "schema": self.schema, "key": key, self.field: value}) + "\n"
        with self._lock:
            self._rows[key] = value
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line)
            self.bytes_written += len(line.encode())
