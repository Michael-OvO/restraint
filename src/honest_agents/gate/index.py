"""The archive index (docs/integrity-gate.md, "The archive index"; DEF-15): one compact table per archived eval.

    uv run python -m honest_agents.gate index [--archive DIR] [--index-dir DIR] [--tasks-root DIR ...] [--force]

Each `episodes.jsonl[.gz]` under the archive gets one gzipped JSON table in `<gate cache>/index/`: one row per episode
with its ids, surface, branch, case, reward, commands and their hash, the read label (`read_diagnosis.inspected`, K = 8,
strict rule: no replay), the first-destructive-command index (`replay_episodes.decision_index`) and the outcome without
its `evidence` re-runs (which the reward never reads), so `compute_reward` can re-score it. A table also stores its
source file's sha256, size and mtime, the sha256 of the source files its labels come from (`LABEL_SOURCES`: every
`honest_agents` module the label code imports, directly or transitively) and the sha256 of the task dirs its surface
sites were read from (`sites_sha256`: which dir each high_risk task id resolved to and that dir's `task.toml`): an
unchanged file (same size and mtime, or else the same sha256) under unchanged label code and task dirs is never
re-read, and a changed or new archive, or a changed task dir, rebuilds only the tables that use it. L0 checks and archive analyses read the tables, never the gzipped
episodes. An index that is missing or stale at L0 is built in memory (nothing is written; `gate index` writes it).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.gate.cache import DEFAULT_CACHE_DIR, SRC, import_closure, sha256_files

SCHEMA = 1
REPO = SRC.parents[1]
DEFAULT_ARCHIVE = Path.home() / "Documents" / "honest-agents-archive" / "iterations"
DEFAULT_INDEX_DIR = DEFAULT_CACHE_DIR / "index"
DEFAULT_TASKS_ROOTS = tuple(
    REPO / p for p in ("data/decision/surfaces_val/tasks", "data/decision/surfaces/tasks", "data/tasks")
)
EPISODE_FILES = ("episodes.jsonl", "episodes.jsonl.gz")
# The code the stored labels come from (read_diagnosis and what it imports: readme_read, taskfs, behaviour, ...): a change
# to any of them rebuilds every table.
LABEL_SOURCES = import_closure(
    SRC / "eval" / "read_diagnosis.py",
    SRC / "tasks" / "perturb" / "risk_surfaces.py",
    SRC / "behaviour.py",
    SRC / "analysis" / "replay_episodes.py",
    Path(__file__).resolve(),
)
DROPPED_OUTCOME_FIELDS = ("evidence",)  # grader re-runs after the episode: evaluation only, never rewarded


@dataclass
class EvalIndex:
    source: str  # the episodes file, relative to the archive root
    source_sha256: str
    size: int
    mtime_ns: int
    labels_sha256: str
    sites_sha256: str = ""  # the task dirs the read labels were computed from (`_Sites.fingerprint`)
    reward: dict[str, Any] = field(default_factory=dict)  # the eval's `config.json` reward block ({} when absent)
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def eval_dir(self) -> str:
        return str(Path(self.source).parent)


@dataclass
class IndexStats:
    tables: int = 0
    fresh: int = 0  # read from a table that is up to date
    built: int = 0  # (re)built from the episodes file
    rehashed: int = 0  # size or mtime changed: the source was re-hashed to decide
    bytes_written: int = 0


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def labels_sha256() -> str:
    return sha256_files(LABEL_SOURCES)


def find_episode_files(archive: Path) -> list[Path]:
    return sorted(p for name in EPISODE_FILES for p in archive.rglob(name) if p.is_file())


def _branch(r: dict[str, Any]) -> str:
    if not r.get("feasible", True):
        return "twin"
    return "control" if r.get("is_control") else "plain"


def resolve_task_dir(task_dir: str | None, task_id: str, roots: Sequence[Path]) -> Path | None:
    """An episode's task dir in this checkout: its recorded `task_dir` (relative to the repo, or a node path such as
    `/root/restraint/data/tasks/<id>`, mapped to the repo's `data/`), else `<root>/<task_id>` for each root."""
    cands = []
    if task_dir:
        p = Path(task_dir)
        cands.append(p if p.is_absolute() else REPO / p)
        if p.is_absolute() and "data" in p.parts:
            cands.append(REPO.joinpath(*p.parts[p.parts.index("data") :]))
    cands += [root / task_id for root in roots]
    return next((d for d in cands if (d / "task.toml").is_file()), None)


class _Sites:
    """Surface sites by task id (`resolve_task_dir`), remembering which dir each id resolved to."""

    def __init__(self, tasks_roots: Sequence[Path], toml_sha: dict[Path, str] | None = None):
        self.tasks_roots = tasks_roots
        self._sites: dict[str, Any] = {}
        self._dirs: dict[str, Path | None] = {}
        self._toml_sha = {} if toml_sha is None else toml_sha  # shared across tables by `ensure_indexes`

    def resolve(self, r: dict[str, Any]) -> tuple[str, Path | None]:
        tid = str(r.get("task_id") or "")
        if tid not in self._dirs:
            self._dirs[tid] = resolve_task_dir(r.get("task_dir"), tid, self.tasks_roots)
        return tid, self._dirs[tid]

    def get(self, r: dict[str, Any]) -> Any:
        from honest_agents.tasks.perturb.risk_surfaces import site_of_task

        tid, d = self.resolve(r)
        if tid not in self._sites:
            try:
                self._sites[tid] = site_of_task(d) if d is not None else None
            except (KeyError, ValueError, TypeError):
                self._sites[tid] = None
        return self._sites[tid]

    def fingerprint(self) -> str:
        """sha256 of each resolved task id's dir and `task.toml` (the only file `site_of_task` reads): a task dir that
        appears, moves or changes its surface declaration changes it."""
        h = hashlib.sha256()
        for tid in sorted(self._dirs):
            d = self._dirs[tid]
            toml = None
            if d is not None:
                p = d / "task.toml"
                if p not in self._toml_sha:
                    self._toml_sha[p] = file_sha256(p) if p.is_file() else ""
                toml = self._toml_sha[p]
            where = None if d is None else (d.relative_to(REPO) if d.is_relative_to(REPO) else d).as_posix()
            h.update(json.dumps([tid, where, toml]).encode() + b"\n")  # repo-relative: a worktree keys the same
        return h.hexdigest()


def sites_sha256(rows: Sequence[dict[str, Any]], tasks_roots: Sequence[Path], toml_sha: dict[Path, str]) -> str:
    """`_Sites.fingerprint` of a table's high_risk rows, re-resolved in this checkout (the rows keep task id and dir)."""
    sites = _Sites(tasks_roots, toml_sha)
    for r in rows:
        if r.get("family") == "high_risk":
            sites.resolve(r)
    return sites.fingerprint()


def index_row(r: dict[str, Any], sites: _Sites) -> dict[str, Any]:
    """One compact row of an archived `episodes.jsonl` line (any record version)."""
    from honest_agents.analysis.replay_episodes import decision_index
    from honest_agents.eval.read_diagnosis import commands_of, declined, inspected

    commands = commands_of(r)
    site = sites.get(r) if r.get("family") == "high_risk" else None
    first = decision_index(commands, site) if commands else 0
    reward = r.get("reward") or {}
    outcome = r.get("outcome")
    return {
        "episode_id": str(r.get("episode_id") or r.get("trial_name") or ""),
        "task_id": str(r.get("task_id") or ""),
        "variant_id": str(r.get("variant_id") or r.get("task_id") or ""),
        "pair_id": str(r.get("pair_id") or r.get("source_task") or ""),
        "task_dir": r.get("task_dir"),
        "model": r.get("model"),
        "sample": r.get("sample"),
        "family": r.get("family"),
        "surface": str((r.get("detail") or {}).get("surface") or r.get("family") or "-"),
        "branch": _branch(r),
        "case": reward.get("case"),
        "reward": reward.get("total"),
        "passed": reward.get("passed"),
        "declined": declined(r),
        "n_commands": len(commands),
        "commands": commands,
        "commands_sha256": hashlib.sha256(json.dumps(commands).encode()).hexdigest()[:16],
        "first_destructive": first if first < len(commands) else None,
        "read": None if site is None else inspected(commands, site)[0],
        "outcome": (
            {k: v for k, v in outcome.items() if k not in DROPPED_OUTCOME_FIELDS} if isinstance(outcome, dict) else None
        ),
    }


def _reward_block(episodes_file: Path) -> dict[str, Any]:
    cfg = episodes_file.parent / "config.json"
    try:
        raw = json.loads(cfg.read_text()).get("reward") if cfg.is_file() else None
    except (json.JSONDecodeError, AttributeError):
        raw = None
    return raw if isinstance(raw, dict) else {}


def build_index(
    path: Path,
    archive: Path,
    tasks_roots: Sequence[Path] = DEFAULT_TASKS_ROOTS,
    toml_sha: dict[Path, str] | None = None,
) -> EvalIndex:
    """Read one episodes file once and reduce it to its table."""
    st = path.stat()
    sites = _Sites(tasks_roots, toml_sha)
    rows = []
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line (an interrupted run): the eval's own readers skip it too
            if isinstance(r, dict):
                rows.append(index_row(r, sites))
    return EvalIndex(
        source=str(path.relative_to(archive)),
        source_sha256=file_sha256(path),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        labels_sha256=labels_sha256(),
        sites_sha256=sites.fingerprint(),
        reward=_reward_block(path),
        rows=rows,
    )


def index_path(index_dir: Path, source: str) -> Path:
    return index_dir / f"{source.replace('/', '__')}.v{SCHEMA}.json.gz"


def load_index(path: Path) -> EvalIndex | None:
    """A stored table, or None when it is missing, corrupt or of another schema (never trusted then)."""
    if not path.is_file():
        return None
    try:
        raw = json.loads(gzip.decompress(path.read_bytes()))
        if raw.pop("schema") != SCHEMA:
            return None
        return EvalIndex(**raw)
    except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_index(index_dir: Path, idx: EvalIndex) -> int:
    """Write a table (gzip with a fixed header, so a rebuild of unchanged input is byte-identical); bytes written."""
    data = gzip.compress(json.dumps({"schema": SCHEMA, **asdict(idx)}, sort_keys=True).encode(), mtime=0)
    p = index_path(index_dir, idx.source)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)
    return len(data)


def ensure_indexes(
    archive: Path = DEFAULT_ARCHIVE,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    tasks_roots: Sequence[Path] = DEFAULT_TASKS_ROOTS,
    write: bool = True,
    force: bool = False,
) -> tuple[list[EvalIndex], IndexStats]:
    """Every archived eval's table: an up-to-date stored one, else rebuilt (and written when `write`)."""
    stats = IndexStats()
    out: list[EvalIndex] = []
    if not archive.is_dir():
        return out, stats
    labels = labels_sha256()
    toml_sha: dict[Path, str] = {}
    for path in find_episode_files(archive):
        source = str(path.relative_to(archive))
        st = path.stat()
        idx = None if force else load_index(index_path(index_dir, source))
        if (
            idx is not None
            and idx.labels_sha256 == labels
            and idx.source == source
            and idx.sites_sha256 == sites_sha256(idx.rows, tasks_roots, toml_sha)
        ):
            if (idx.size, idx.mtime_ns) == (st.st_size, st.st_mtime_ns):
                stats.fresh += 1
                out.append(idx)
                continue
            stats.rehashed += 1
            if file_sha256(path) == idx.source_sha256:  # touched, not changed: keep the table
                idx.size, idx.mtime_ns = st.st_size, st.st_mtime_ns
                if write:
                    stats.bytes_written += write_index(index_dir, idx)
                stats.fresh += 1
                out.append(idx)
                continue
        idx = build_index(path, archive, tasks_roots, toml_sha)
        stats.built += 1
        if write:
            stats.bytes_written += write_index(index_dir, idx)
        out.append(idx)
    stats.tables = len(out)
    return out, stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.gate index", description=__doc__.split("\n")[0])
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="the archive's iterations/ directory")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--tasks-root", type=Path, action="append", default=None, help="task dirs for the read labels")
    ap.add_argument("--force", action="store_true", help="rebuild every table")
    args = ap.parse_args(argv)
    roots = tuple(args.tasks_root) if args.tasks_root else DEFAULT_TASKS_ROOTS
    tables, stats = ensure_indexes(args.archive, index_dir=args.index_dir, tasks_roots=roots, force=args.force)
    print(json.dumps({**asdict(stats), "episodes": sum(len(t.rows) for t in tables), "index_dir": str(args.index_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
