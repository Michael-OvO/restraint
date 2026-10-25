"""The gate's content-addressed result cache (docs/integrity-gate.md, "the result cache"; DEF-15).

One append-only JSONL file per check and schema version under `~/.cache/honest-agents/gate/` (a node:
`/root/.cache/honest-agents/gate/`), outside the repo and the archive. A row is `{check, schema, key, <field>}`: the
verdict and the small outputs the check needs, never images, containers or full episodes. The key is the sha256 of
the task dir contents, of the source files the check imports, of the command list and of the base image digest
(`cache_key`), so a change to one task re-runs that task only and a change to shared code re-runs everything. A
corrupt, torn or old-version row is ignored, never trusted; a later row for the same key wins.
"""

from __future__ import annotations

import hashlib
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
