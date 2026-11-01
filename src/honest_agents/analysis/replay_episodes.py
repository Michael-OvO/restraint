"""Replay archived episodes' commands in fresh sandboxes to recover the tool outputs the records never kept (DEF-06).

    uv run python -m honest_agents.analysis.replay_episodes EPISODES.jsonl[.gz] [...] --out replay.jsonl \
        [--tasks-root data/decision/surfaces_val/tasks] [--pair] [--sample 24 --seed 0] [--concurrency 4]
        [--regrade --surface inbound_link --branch control]

Records written before DEF-06 hold only `{command, rc}` per step (`extra.history`), and `outcome.evidence` holds grader
re-runs made after the episode, often after the `rm`. Neither shows what the agent saw. For each episode this tool:

* rebuilds the task from its task directory (`<tasks-root>/<task_id>`, the deterministic build the eval ran on; a
  missing directory is reported as `no_task_dir`, never skipped silently);
* replays the **decision prefix**: the commands before the first destructive command (`behaviour.is_destructive`) that
  names the surface's target or a protected path (full path or basename), or the whole history when there is none (the
  episode ended in a report or `abstain`). `--full` replays the whole history instead. Each command runs through the
  real `HonestHarborEnv.bash` (same wrapper, user, network policy and `timeout`), with a budget that never refuses;
* records each output as `harness.history_entry` does for new episodes (capped at 4,000 chars, cut recorded);
* checks fidelity: the replayed exit codes must equal the archived ones, else the episode is `not_replayable` with the
  first mismatching step;
* with `--pair`, replays the same prefix in the other branch of the twin/control pair (`<id>` <-> `<id>__control`).
  Its exit codes may differ by design; they are recorded, not checked;
* with `--regrade`, replays the **whole** history, hands the env the archived report and grades the episode against the
  task dir as it is now (a fixed test or sentinel), with the eval's reward config (`config.json`): the row carries the
  archived and the new case, so a grading fix re-scores the archive ("re-scored, post-fix"). Regrades are not cached.

Results are cached content-addressed (docs/integrity-gate.md, "the result cache"; `gate.cache`, shared with
`tasks.g1_surfaces`): the key is the sha256 of the task dir
contents, of the source files that shape a replay (`env.py`, `harness.py`, `grading.py` and everything this module
imports, `gate.cache.import_closure`), of the Harbor and TRL versions, of the command list, of the command timeout and
of the base image digest. A cache hit boots nothing. A branch whose exec failed mid-way
(`[sandbox unavailable]`) is `boot_failed` and never cached. Rows are append-only and carry a schema
version; a corrupt or old row is ignored. The summary gives, per surface and branch, the share not replayable (bar:
<= 10 % per branch, docs/iteration-16-direction.md S0.2).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import logging
import posixpath
import re
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from honest_agents.behaviour import FAMILY_HIGH_RISK, is_destructive
from honest_agents.eval.records import history_steps
from honest_agents.gate import cache as gate_cache
from honest_agents.gate.cache import (
    base_image,
    docker_image_digest,
    import_closure,
    library_versions,
    sha256_files,
    sha256_tree,
)
from honest_agents.harness import SANDBOX_UNAVAILABLE
from honest_agents.report import SelfReport
from honest_agents.reward import EpisodeOutcome, RewardBreakdown, RewardConfig, outcome_from_dict
from honest_agents.tasks.perturb.risk_surfaces import SurfaceSite, site_of_task

log = logging.getLogger(__name__)

CHECK = "replay_episodes"
SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = gate_cache.DEFAULT_CACHE_DIR
DEFAULT_TASKS_ROOT = Path("data/decision/surfaces_val/tasks")
DEFAULT_TIMEOUT_SEC = 60
MAX_NOT_REPLAYABLE = 0.10  # S0.2 bar, per branch
CONTROL_SUFFIX = "__control"
# The source files a replayed output depends on: every `honest_agents` module this one imports, directly or through
# another (the bash wrapper and output capture in env and harness, the boot-time snapshot and self-check in grading,
# and what they import: bank, mutation, spec, report, reward, verifier), and the libraries the env is built on. The
# integrity gate keys every cached check on the files it imports (DEF-24).
SOURCE_FILES = import_closure(Path(__file__))
LIBRARIES = ("harbor", "trl")

# Statuses of one replayed episode.
REPLAYABLE = "replayable"  # (a regraded episode: replayed and graded, exit codes matched)
NOT_REPLAYABLE = "not_replayable"
NO_TASK_DIR = "no_task_dir"
BOOT_FAILED = "boot_failed"


# ── archived episodes ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchivedEpisode:
    episode_id: str
    task_id: str
    surface: str
    twin: bool  # infeasible branch of a twin/control pair (`feasible = false`)
    commands: tuple[str, ...]
    rcs: tuple[int | None, ...]
    family: str | None = None
    # the archived `outcome` and `reward.case` (`--regrade` hands the report back and compares the case)
    outcome: dict[str, Any] | None = field(default=None, compare=False, hash=False, repr=False)
    case: str = ""

    @property
    def branch(self) -> str:
        return "twin" if self.twin else "control"


def surface_label(row: dict[str, Any]) -> str:
    """A row's surface: the canonical `read_diagnosis.surface_of` (`readme` when `detail.surface` is absent) on a
    high_risk row (DEF-31), the family on any other."""
    if row.get("family") != FAMILY_HIGH_RISK:
        return str(row.get("family") or "-")
    from honest_agents.eval.read_diagnosis import surface_of  # read_diagnosis imports this module

    return surface_of(row)


def episode_of(row: dict[str, Any]) -> ArchivedEpisode:
    """One `episodes.jsonl` row (any record version)."""
    steps = history_steps(row.get("extra"))
    return ArchivedEpisode(
        episode_id=str(row["episode_id"]),
        task_id=str(row["task_id"]),
        surface=surface_label(row),
        twin=not bool(row.get("feasible", True)),
        commands=tuple(s.command for s in steps),
        rcs=tuple(s.rc for s in steps),
        family=row.get("family"),
        outcome=row.get("outcome") if isinstance(row.get("outcome"), dict) else None,
        case=str((row.get("reward") or {}).get("case") or ""),
    )


def load_episodes(path: Path) -> list[ArchivedEpisode]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [episode_of(json.loads(line)) for line in f if line.strip()]


def partner_id(task_id: str) -> str:
    """The other branch of a twin/control pair: `<id>` <-> `<id>__control`."""
    return task_id.removesuffix(CONTROL_SUFFIX) if task_id.endswith(CONTROL_SUFFIX) else task_id + CONTROL_SUFFIX


def _names(command: str, path: str) -> bool:
    """`command` names `path` (or its basename, longer than 3 characters) as a whole word: `/a/comet` and `comet/x`
    count, `comet_old` does not."""
    path = posixpath.normpath(path)
    base = posixpath.basename(path)
    words = [path] + ([base] if len(base) > 3 else [])
    return any(re.search(rf"(?<![\w.-]){re.escape(w)}(?![\w.-])", command) for w in words)


def decision_index(commands: Sequence[str], site: SurfaceSite | None) -> int:
    """Index of the first destructive command on the decision's target (the end of the replayed prefix).

    With a surface site: a destructive command naming the target, a protected path or the real directory of a link. A
    task with no site: the first destructive command. None found: `len(commands)` (the episode ended in a report)."""
    paths = [] if site is None else [p for p in (site.target, *site.protected, site.real) if p]
    for i, command in enumerate(commands):
        if is_destructive(command) and (not paths or any(_names(command, p) for p in paths)):
            return i
    return len(commands)


def stratified_sample(episodes: Sequence[ArchivedEpisode], n: int, seed: int = 0) -> list[ArchivedEpisode]:
    """`n` episodes spread evenly over (surface, branch) strata, each stratum in a seeded hash order (deterministic and
    independent of file order)."""
    strata: dict[tuple[str, str], list[ArchivedEpisode]] = defaultdict(list)
    for ep in episodes:
        strata[(ep.surface, ep.branch)].append(ep)
    for group in strata.values():
        group.sort(key=lambda e: hashlib.sha256(f"{seed}:{e.episode_id}".encode()).hexdigest())
    out: list[ArchivedEpisode] = []
    keys = sorted(strata)
    depth = 0
    while len(out) < n and any(depth < len(strata[k]) for k in keys):
        for k in keys:
            if depth < len(strata[k]) and len(out) < n:
                out.append(strata[k][depth])
        depth += 1
    return out


# ── sandbox runner ────────────────────────────────────────────────────────────


class BootError(RuntimeError):
    """The sandbox did not come up, so no command ran."""


class Runner(Protocol):
    def run(self, task_dir: Path, commands: Sequence[str]) -> list[dict[str, Any]]:
        """Boot `task_dir`, run `commands` in order, tear down; one `harness.history_entry` per command."""
        ...


class HarborRunner:
    """Runs commands through a real `HonestHarborEnv` (Docker), with a budget that never refuses and no grading."""

    def __init__(self, command_timeout_sec: int = DEFAULT_TIMEOUT_SEC, enforce_network_policy: bool = True):
        self.command_timeout_sec = command_timeout_sec
        self.enforce_network_policy = enforce_network_policy

    def run(self, task_dir: Path, commands: Sequence[str]) -> list[dict[str, Any]]:
        from honest_agents.env import HonestHarborEnv
        from honest_agents.harness import EpisodeBudget

        budget = EpisodeBudget(
            max_bash_calls=len(commands) + 1,
            max_total_chars=10**9,
            command_timeout_sec=self.command_timeout_sec,
        )
        env = HonestHarborEnv(budget=budget, enforce_network_policy=self.enforce_network_policy)
        try:
            env.reset(task_dir=str(task_dir))
            try:
                env._wait_boot()
            except Exception as e:
                raise BootError(f"{type(e).__name__}: {e}") from e
            for command in commands:
                env.bash(command)
            history = list(env._history)
        finally:
            try:
                if env._boot is not None:
                    concurrent.futures.wait([env._boot])
                if env._env is not None:
                    env._run(env._teardown())
            finally:
                env._loop.call_soon_threadsafe(env._loop.stop)
        if len(history) != len(commands):
            raise BootError(f"{len(history)} of {len(commands)} commands ran")
        return history

    def regrade(
        self, task_dir: Path, commands: Sequence[str], report: SelfReport | None, reward_config: RewardConfig
    ) -> tuple[list[dict[str, Any]], EpisodeOutcome, RewardBreakdown]:
        """Boot `task_dir`, run `commands`, submit `report` (None: the episode ended without one) and grade, as the eval
        would have; the env tears the sandbox down after grading."""
        from honest_agents.env import HonestHarborEnv
        from honest_agents.harness import EpisodeBudget

        budget = EpisodeBudget(
            max_bash_calls=len(commands) + 1,
            max_total_chars=10**9,
            command_timeout_sec=self.command_timeout_sec,
        )
        env = HonestHarborEnv(
            budget=budget, reward_config=reward_config, enforce_network_policy=self.enforce_network_policy
        )
        try:
            env.reset(task_dir=str(task_dir))
            try:
                env._wait_boot()
            except Exception as e:
                raise BootError(f"{type(e).__name__}: {e}") from e
            for command in commands:
                env.bash(command)
            env._set_external_report(report, None)
            outcome, breakdown, _ = env._graded()
            history = list(env._history)
        finally:
            try:
                if env._boot is not None:
                    concurrent.futures.wait([env._boot])
                if env._env is not None and env._result is None:
                    env._run(env._teardown())
            finally:
                env._loop.call_soon_threadsafe(env._loop.stop)
        if len(history) != len(commands):
            raise BootError(f"{len(history)} of {len(commands)} commands ran")
        return history, outcome, breakdown


# ── result cache ──────────────────────────────────────────────────────────────


def cache_key(
    task_hash: str,
    source_hash: str,
    commands: Sequence[str],
    image_digest: str,
    timeout_sec: int,
    libraries: dict[str, str],
) -> str:
    return gate_cache.cache_key(
        CHECK,
        SCHEMA_VERSION,
        task=task_hash,
        sources=source_hash,
        libraries=libraries,
        commands=list(commands),
        image=image_digest,
        timeout=timeout_sec,
    )


class ReplayCache(gate_cache.ResultCache):
    """The gate's result cache (`gate.cache.ResultCache`) for replayed history entries: `{check, schema, key, entries}`
    rows under the gate cache directory."""

    def __init__(self, cache_dir: Path | None):
        super().__init__(CHECK, SCHEMA_VERSION, cache_dir, field="entries")

    def _valid(self, value: Any) -> bool:
        return isinstance(value, list)


# ── replay ────────────────────────────────────────────────────────────────────


@dataclass
class BranchReplay:
    task_id: str
    status: str  # REPLAYABLE | NOT_REPLAYABLE | NO_TASK_DIR | BOOT_FAILED (the pair branch: REPLAYABLE once it ran)
    rcs: list[int | None] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    cached: bool = False
    error: str = ""


@dataclass
class EpisodeReplay:
    episode_id: str
    task_id: str
    surface: str
    branch: str
    n_commands: int  # archived history length
    prefix_end: int  # commands replayed: history[:prefix_end]
    stop: str  # "destructive" (prefix ends before it) | "report" (whole history) | "full" (--full)
    archived_rcs: list[int | None]
    status: str
    mismatch_at: int | None = None
    own: BranchReplay | None = None
    pair: BranchReplay | None = None


@dataclass
class EpisodeRegrade:
    episode_id: str
    task_id: str
    surface: str
    branch: str
    n_commands: int
    status: str  # REPLAYABLE (replayed, exit codes matched, graded) | NOT_REPLAYABLE | NO_TASK_DIR | BOOT_FAILED
    archived_case: str
    case: str = ""  # the new case (also set when NOT_REPLAYABLE: the grade of a diverged replay, not to be used)
    mismatch_at: int | None = None
    archived_tests_passed: bool | None = None
    tests_passed: bool | None = None
    violation_details: list[str] = field(default_factory=list)
    error: str = ""


def _sandbox_lost(entry: dict[str, Any]) -> bool:
    """A step whose exec raised (`HonestHarborEnv.bash` records rc None and `SANDBOX_UNAVAILABLE`)."""
    return entry.get("rc") is None and entry.get("output") == SANDBOX_UNAVAILABLE


@dataclass
class Replayer:
    runner: Runner
    cache: ReplayCache
    tasks_root: Path
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    image_digest: Callable[[str], str] = docker_image_digest
    source_hash: str = field(default_factory=lambda: sha256_files(SOURCE_FILES))
    libraries: dict[str, str] = field(default_factory=lambda: library_versions(*LIBRARIES))
    boots: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def _count(self, name: str) -> None:
        with self._lock:  # `replay_all` calls `run_branch` from pool threads
            setattr(self, name, getattr(self, name) + 1)

    def run_branch(self, task_id: str, commands: Sequence[str]) -> BranchReplay:
        task_dir = self.tasks_root / task_id
        if not (task_dir / "task.toml").is_file():
            return BranchReplay(task_id, NO_TASK_DIR)
        if not commands:
            return BranchReplay(task_id, REPLAYABLE)
        key = cache_key(
            sha256_tree(task_dir),
            self.source_hash,
            commands,
            self.image_digest(base_image(task_dir)),
            self.timeout_sec,
            self.libraries,
        )
        entries = self.cache.get(key)
        cached = entries is not None
        if entries is None:
            self._count("boots")
            try:
                entries = self.runner.run(task_dir, commands)
            except BootError as e:
                return BranchReplay(task_id, BOOT_FAILED, error=str(e)[:500])
            lost = next((i for i, e in enumerate(entries) if _sandbox_lost(e)), None)
            if lost is not None:
                # The exec itself failed (docker gone, daemon restart): an infra failure, not a result. Never cached,
                # so a rerun tries again instead of reporting the episode not replayable until a source file changes.
                return BranchReplay(task_id, BOOT_FAILED, error=f"sandbox unavailable at step {lost}")
            self.cache.put(key, entries)
        else:
            self._count("cache_hits")
        return BranchReplay(task_id, REPLAYABLE, [e.get("rc") for e in entries], entries, cached)

    def replay(self, ep: ArchivedEpisode, *, pair: bool = False, full: bool = False) -> EpisodeReplay:
        task_dir = self.tasks_root / ep.task_id
        site = site_of_task(task_dir) if (task_dir / "task.toml").is_file() else None
        end = len(ep.commands) if full else decision_index(ep.commands, site)
        stop = "full" if full else ("destructive" if end < len(ep.commands) else "report")
        prefix = list(ep.commands[:end])
        archived = list(ep.rcs[:end])
        own = self.run_branch(ep.task_id, prefix)
        row = EpisodeReplay(
            episode_id=ep.episode_id,
            task_id=ep.task_id,
            surface=ep.surface,
            branch=ep.branch,
            n_commands=len(ep.commands),
            prefix_end=end,
            stop=stop,
            archived_rcs=archived,
            status=own.status,
            own=own,
        )
        if own.status == REPLAYABLE:
            row.mismatch_at = next((i for i, (a, b) in enumerate(zip(archived, own.rcs, strict=True)) if a != b), None)
            if row.mismatch_at is not None:
                row.status = own.status = NOT_REPLAYABLE
        if pair:
            row.pair = self.run_branch(partner_id(ep.task_id), prefix)
        return row

    def regrade(self, ep: ArchivedEpisode, reward_config: RewardConfig) -> EpisodeRegrade:
        """Replay the whole history, submit the archived report and grade against the task dir as it is now."""
        row = EpisodeRegrade(
            episode_id=ep.episode_id,
            task_id=ep.task_id,
            surface=ep.surface,
            branch=ep.branch,
            n_commands=len(ep.commands),
            status=REPLAYABLE,
            archived_case=ep.case,
            archived_tests_passed=(ep.outcome or {}).get("tests_passed"),
        )
        task_dir = self.tasks_root / ep.task_id
        if not (task_dir / "task.toml").is_file():
            row.status = NO_TASK_DIR
            return row
        report = outcome_from_dict(ep.outcome).report if ep.outcome else None
        self._count("boots")
        try:
            entries, outcome, breakdown = self.runner.regrade(task_dir, ep.commands, report, reward_config)  # type: ignore[attr-defined]
        except BootError as e:
            row.status, row.error = BOOT_FAILED, str(e)[:500]
            return row
        lost = next((i for i, e in enumerate(entries) if _sandbox_lost(e)), None)
        if lost is not None or outcome.grading_error is not None:
            row.status = BOOT_FAILED
            row.error = f"sandbox unavailable at step {lost}" if lost is not None else str(outcome.grading_error)[:500]
            return row
        row.case, row.tests_passed, row.violation_details = (
            breakdown.case,
            outcome.tests_passed,
            outcome.violation_details,
        )
        rcs = [e.get("rc") for e in entries]
        row.mismatch_at = next((i for i, (a, b) in enumerate(zip(ep.rcs, rcs, strict=True)) if a != b), None)
        if row.mismatch_at is not None:
            row.status = NOT_REPLAYABLE
        return row


def regrade_all(
    replayer: Replayer,
    episodes: Sequence[ArchivedEpisode],
    reward_config: RewardConfig,
    *,
    concurrency: int = 4,
    on_result: Callable[[EpisodeRegrade], None] | None = None,
) -> list[EpisodeRegrade]:
    """`Replayer.regrade` over `episodes` with at most `concurrency` live sandboxes; results in input order."""
    results: dict[int, EpisodeRegrade] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(replayer.regrade, ep, reward_config): i for i, ep in enumerate(episodes)}
        try:
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                results[futures[f]] = r
                if on_result is not None:
                    on_result(r)
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise
    return [results[i] for i in sorted(results)]


def regrade_summary(rows: Sequence[EpisodeRegrade]) -> dict[str, Any]:
    """Per (surface, branch): status counts, and archived vs new case counts over the replayable episodes."""
    out: dict[str, Any] = {}
    for r in rows:
        c = out.setdefault(f"{r.surface}/{r.branch}", {"n": 0, "status": {}, "archived": {}, "new": {}, "changed": 0})
        c["n"] += 1
        c["status"][r.status] = c["status"].get(r.status, 0) + 1
        if r.status == REPLAYABLE:
            c["archived"][r.archived_case] = c["archived"].get(r.archived_case, 0) + 1
            c["new"][r.case] = c["new"].get(r.case, 0) + 1
            c["changed"] += r.case != r.archived_case
    return dict(sorted(out.items()))


def replay_all(
    replayer: Replayer,
    episodes: Sequence[ArchivedEpisode],
    *,
    pair: bool = False,
    full: bool = False,
    concurrency: int = 4,
    on_result: Callable[[EpisodeReplay], None] | None = None,
) -> list[EpisodeReplay]:
    """Replay every episode with at most `concurrency` live sandboxes; results in input order. Ctrl-C cancels what has
    not started and lets running replays tear their sandboxes down."""
    results: dict[int, EpisodeReplay] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(replayer.replay, ep, pair=pair, full=full): i for i, ep in enumerate(episodes)}
        try:
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                results[futures[f]] = r
                if on_result is not None:
                    on_result(r)
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise
    return [results[i] for i in sorted(results)]


# ── summary ───────────────────────────────────────────────────────────────────


def summarize(rows: Sequence[EpisodeReplay]) -> dict[str, Any]:
    """Counts per status for every (surface, branch) cell and per branch, the not-replayable share among episodes that
    could be attempted (task dir present, sandbox booted) and whether each branch meets the S0.2 bar."""
    cells: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for name in (f"{r.surface}/{r.branch}", r.branch, "all"):
            cells[name][r.status] += 1
            cells[name]["n"] += 1
            cells[name]["empty_prefix"] += r.prefix_end == 0
            if r.pair is not None:
                cells[name][f"pair_{r.pair.status}"] += 1
    out: dict[str, Any] = {}
    for name, c in sorted(cells.items(), key=lambda kv: (kv[0] in ("twin", "control", "all"), kv[0])):
        tried = c[REPLAYABLE] + c[NOT_REPLAYABLE]
        share = c[NOT_REPLAYABLE] / tried if tried else None
        out[name] = {**dict(c), "not_replayable_share": share}
        if name in ("twin", "control"):
            out[name]["meets_bar"] = share is not None and share <= MAX_NOT_REPLAYABLE
    return out


def render(summary: dict[str, Any]) -> str:
    lines = [
        "| cell | n | replayable | not replayable | no task dir | boot failed | empty prefix | not-replayable share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in summary.items():
        share = c["not_replayable_share"]
        lines.append(
            f"| {name} | {c.get('n', 0)} | {c.get(REPLAYABLE, 0)} | {c.get(NOT_REPLAYABLE, 0)} | {c.get(NO_TASK_DIR, 0)} "
            f"| {c.get(BOOT_FAILED, 0)} | {c.get('empty_prefix', 0)} | {'-' if share is None else f'{100 * share:.1f}%'} |"
        )
    return "\n".join(lines) + "\n"


def _reward_of(paths: Sequence[Path]) -> RewardConfig:
    """The eval's reward config from a `config.json` beside the episode file, else the default."""
    for p in paths:
        cfg = p.parent / "config.json"
        if cfg.is_file() and isinstance(r := json.loads(cfg.read_text()).get("reward"), dict):
            return RewardConfig(**r)
    return RewardConfig()


def _timeout_of(paths: Sequence[Path]) -> int:
    """The eval's `budget.command_timeout_sec` from a `config.json` beside the episode file, else the default."""
    for p in paths:
        cfg = p.parent / "config.json"
        if cfg.is_file():
            t = (json.loads(cfg.read_text()).get("budget") or {}).get("command_timeout_sec")
            if isinstance(t, int):
                return t
    return DEFAULT_TIMEOUT_SEC


def read_episode_ids(path: Path) -> set[str]:
    """Episode ids from a JSONL of rows with `episode_id`, or from plain lines of ids."""
    out = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.add(str(json.loads(line)["episode_id"]) if line.startswith("{") else line)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.analysis.replay_episodes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("episodes", nargs="+", type=Path, help="archived episodes.jsonl[.gz] files")
    ap.add_argument("--out", type=Path, required=True, help="one JSON row per replayed episode")
    ap.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT, help="task dirs, one per task_id")
    ap.add_argument("--pair", action="store_true", help="also replay the prefix in the other branch of the pair")
    ap.add_argument("--full", action="store_true", help="replay the whole history, not only the decision prefix")
    ap.add_argument(
        "--regrade",
        action="store_true",
        help="replay the whole history, submit the archived report and grade against the task dir as it is now",
    )
    ap.add_argument("--surface", default=None, help="only episodes of this surface")
    ap.add_argument("--branch", choices=("twin", "control"), default=None, help="only episodes of this branch")
    ap.add_argument("--family", default="high_risk", help="only episodes of this family ('' for all)")
    ap.add_argument(
        "--episode-ids",
        type=Path,
        default=None,
        help="only these episodes: a JSONL with `episode_id` (e.g. data/def74/affected.jsonl) or one id per line",
    )
    ap.add_argument("--sample", type=int, default=None, help="a stratified sample of this many episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4, help="live sandboxes at once (<= 4 on the Mac)")
    ap.add_argument("--timeout", type=int, default=None, help="per-command timeout (default: the eval's config.json)")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the result cache")
    ap.add_argument("--json", type=Path, default=None, help="also write the summary and run stats here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    episodes = [e for p in args.episodes for e in load_episodes(p) if not args.family or e.family == args.family]
    episodes = [
        e
        for e in episodes
        if (args.surface is None or e.surface == args.surface) and (args.branch is None or e.branch == args.branch)
    ]
    if args.episode_ids is not None:
        wanted = read_episode_ids(args.episode_ids)
        episodes = [e for e in episodes if e.episode_id in wanted]
    if args.sample is not None:
        episodes = stratified_sample(episodes, args.sample, args.seed)
    timeout = args.timeout if args.timeout is not None else _timeout_of(args.episodes)
    from honest_agents.harness import set_max_concurrent_boots

    set_max_concurrent_boots(max(1, args.concurrency))
    replayer = Replayer(
        runner=HarborRunner(command_timeout_sec=timeout),
        cache=ReplayCache(None if args.no_cache else args.cache_dir),
        tasks_root=args.tasks_root,
        timeout_sec=timeout,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.regrade:
        with open(args.out, "w") as out:

            def write_regrade(r: EpisodeRegrade) -> None:
                out.write(json.dumps(asdict(r)) + "\n")
                out.flush()
                log.info(
                    "%-15s %-7s %-14s %s -> %s %s", r.surface, r.branch, r.status, r.archived_case, r.case, r.episode_id
                )

            regraded = regrade_all(
                replayer, episodes, _reward_of(args.episodes), concurrency=args.concurrency, on_result=write_regrade
            )
        result = {
            "summary": regrade_summary(regraded),
            "stats": {
                "episodes": len(regraded),
                "wall_time_sec": round(time.monotonic() - t0, 1),
                "timeout_sec": timeout,
            },
        }
        print(json.dumps(result, indent=2))
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    with open(args.out, "w") as out:

        def write(r: EpisodeReplay) -> None:
            out.write(json.dumps(asdict(r)) + "\n")
            out.flush()
            log.info("%-15s %-7s %-14s %s", r.surface, r.branch, r.status, r.episode_id)

        rows = replay_all(
            replayer, episodes, pair=args.pair, full=args.full, concurrency=args.concurrency, on_result=write
        )
    summary = summarize(rows)
    stats = {
        "episodes": len(rows),
        "wall_time_sec": round(time.monotonic() - t0, 1),
        "sandbox_boots": replayer.boots,
        "cache_hits": replayer.cache_hits,
        "out_bytes": args.out.stat().st_size,
        "timeout_sec": timeout,
    }
    print(render(summary))
    print(json.dumps(stats))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "stats": stats}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
