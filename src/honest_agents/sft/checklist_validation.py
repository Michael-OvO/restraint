"""Validate the requirement checklists against the hidden tests, in Docker (iteration 16, pre-registration section 3.3).

For each training-pool task with a checklist (`data/commit_checks/checklists/<task>.json`) the checklist commands run on
end states whose hidden-test verdict is known, and the checklist verdict (every requirement passes) is compared with the
tests' verdict on the same state:

* **oracle**: `solution/solve.sh` applied (`tasks.validate.OracleHarborEnv._run_oracle`, as the oracle gate runs it).
  A requirement that fails where the tests pass is a **bad check** (it rejects a correct end state);
* **episode**: an archived own episode's whole command history replayed (`analysis.replay_episodes`' fidelity rule:
  every recorded exit code must be replayed, else the state is `not_replayable` and unused). These are the near-miss
  states: the end states the model's own claims were made on.

In one sandbox: boot, run the oracle or the history, run the checklist commands through the env's `bash` (the agent
user and wrapper the data build's `run-checks` uses), then grade with the hidden tests. A high_risk twin's tests do not
run in an episode (`run_tests_when_infeasible = false`); here they are forced on, so every state has a full-suite
verdict, and the twin's *feasible part* (every test but `grading.TWIN_REQUEST_TESTS`, against every requirement but the
removal one) is compared as well. Checks are read-only by the checklist schema, so running them before the tests does
not change what the tests see.

Results are cached content-addressed (docs/integrity-gate.md, "the result cache"; `gate.cache`): the key is the sha256
of the task dir, of the source files this module imports (`import_closure`: env, harness, grading, verifier, validate,
replay_episodes...), of the Harbor and TRL versions, of the state's commands (or the oracle marker), of the checklist
commands, of the command timeout and of the base image digest. A state whose sandbox failed (boot, lost exec, grading
error) is never cached. The verdicts are derived in memory from the stored outputs, so a change to a `pass_if`, or a
dropped requirement, re-scores without a boot; a changed command re-runs only the states of its task.

`--out` gets one row per state (`Row`: the requirement outputs, the test verdict and the per-test outcomes). Scoring,
the agreement report and the data build's verdict file are `sft.checklist_agreement` (no Docker): it is kept out of
this module's import closure, so changing the report does not invalidate the cache.

    uv run python -m honest_agents.sft.checklist_validation --oracle --episodes EPISODES.jsonl.gz --out rows.jsonl \\
        [--sample-per-cell N] [--budget-min 170] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import logging
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
from honest_agents.sft.checklists import DEFAULT_DIR as DEFAULT_CHECKLISTS
from honest_agents.sft.checklists import Checklist, load_checklists
from honest_agents.tasks.taskfs import read_task_toml

log = logging.getLogger(__name__)

CHECK = "checklist_validation"
SCHEMA_VERSION = 1
SOURCE_FILES = import_closure(Path(__file__))
LIBRARIES = ("harbor", "trl")
DEFAULT_TASKS_ROOT = Path("data/decision/surfaces/tasks")
DEFAULT_TIMEOUT_SEC = 60
ORACLE_TIMEOUT_SEC = 900
ORACLE = "oracle"
EPISODE = "episode"
CONTROL_SUFFIX = "__control"

# Row statuses.
OK = "ok"
NOT_REPLAYABLE = "not_replayable"  # an archived exit code was not replayed: the state is not the archived one
NO_TASK_DIR = "no_task_dir"
NO_CHECKLIST = "no_checklist"
BOOT_FAILED = "boot_failed"  # boot, a lost exec or a grading error: infrastructure, never cached
ORACLE_FAILED = "oracle_failed"  # solve.sh exited non-zero (the state is kept; the tests decide)

class BootError(RuntimeError):
    """The sandbox did not come up, so no state was made."""


# ── states ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class State:
    """An end state with a known hidden-test verdict: the oracle's, or an archived episode's whole history."""

    unit_id: str  # `oracle:<task>` or the episode id
    kind: str  # ORACLE | EPISODE
    task: str
    origin: str  # ORACLE, or the eval the episode comes from
    commands: tuple[str, ...] = ()
    rcs: tuple[int | None, ...] = ()
    archived_tests_passed: bool | None = None
    archived_case: str = ""

    @property
    def branch(self) -> str:
        return "control" if self.task.endswith(CONTROL_SUFFIX) else "twin"


def oracle_states(tasks: Iterable[str]) -> list[State]:
    return [State(f"{ORACLE}:{t}", ORACLE, t, ORACLE) for t in sorted(set(tasks))]


def episode_states(path: Path) -> list[State]:
    """Every episode of a run_eval `episodes.jsonl[.gz]`, named by the eval directory it comes from."""
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[State] = []
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            steps = history_steps(rec.get("extra"))
            out.append(State(
                unit_id=str(rec["episode_id"]), kind=EPISODE, task=str(rec.get("variant_id") or rec["task_id"]),
                origin=path.parent.name, commands=tuple(s.command for s in steps), rcs=tuple(s.rc for s in steps),
                archived_tests_passed=(rec.get("outcome") or {}).get("tests_passed"),
                archived_case=str((rec.get("reward") or {}).get("case") or ""),
            ))  # fmt: skip
    return out


def stratified(states: Sequence[State], per_cell: int | None, seed: int = 0) -> list[State]:
    """At most `per_cell` episode states per (branch, archived test verdict) cell, each cell in a seeded hash order,
    interleaved round-robin over the cells. Oracle states are always kept and go first. `per_cell` None keeps every state (still interleaved, so a budget stop leaves a balanced set)."""
    oracles = [s for s in states if s.kind == ORACLE]
    cells: dict[tuple[str, str], list[State]] = defaultdict(list)
    for s in states:
        if s.kind == EPISODE:
            cells[(s.branch, str(s.archived_tests_passed))].append(s)
    for group in cells.values():
        group.sort(key=lambda s: hashlib.sha256(f"{seed}:{s.unit_id}".encode()).hexdigest())
        del group[per_cell if per_cell is not None else len(group) :]
    out: list[State] = []
    keys = sorted(cells)
    depth = 0
    while any(depth < len(cells[k]) for k in keys):
        out += [cells[k][depth] for k in keys if depth < len(cells[k])]
        depth += 1
    return [*oracles, *out]


# ── the sandbox ──────────────────────────────────────────────────────────────


class Runner(Protocol):
    def run(self, task_dir: Path, state: State, checks: Sequence[str]) -> dict[str, Any]:
        """Boot `task_dir`, make `state` (the oracle or the commands), run `checks`, grade with the tests forced on.
        Returns {"rcs": [...], "checks": [{"cmd", "rc", "output"}], "oracle_rc", "tests_passed", "outcomes",
        "exit_code", "timed_out", "no_result", "grading_error"}; raises `BootError` when nothing could run."""
        ...


class HarborRunner:
    """A real `OracleHarborEnv` (Docker), with a budget that never refuses."""

    def __init__(self, command_timeout_sec: int = DEFAULT_TIMEOUT_SEC, oracle_timeout_sec: int = ORACLE_TIMEOUT_SEC):
        self.command_timeout_sec = command_timeout_sec
        self.oracle_timeout_sec = oracle_timeout_sec

    def run(self, task_dir: Path, state: State, checks: Sequence[str]) -> dict[str, Any]:
        from honest_agents.harness import EpisodeBudget
        from honest_agents.tasks.validate import ORACLE_REPORT, OracleHarborEnv

        budget = EpisodeBudget(
            max_bash_calls=len(state.commands) + len(checks) + 1,
            max_total_chars=10**9,
            command_timeout_sec=self.command_timeout_sec,
        )
        env = OracleHarborEnv(budget=budget)
        out: dict[str, Any] = {"oracle_rc": None}
        try:
            env.reset(task_dir=str(task_dir))
            # Every state gets a full-suite verdict: a high_risk twin's tests are forced on (grading only).
            env._spec = env._spec.model_copy(update={"run_tests_when_infeasible": True})
            try:
                env._wait_boot()
            except Exception as e:
                raise BootError(f"{type(e).__name__}: {e}") from e
            if state.kind == ORACLE:
                oracle = env._run(env._run_oracle(timeout_sec=self.oracle_timeout_sec, destructive=[]))
                out["oracle_rc"] = oracle.get("oracle_rc")
                out["oracle_tail"] = str(oracle.get("oracle_tail") or "")[-500:]
            for command in state.commands:
                env.bash(command)
            n = len(state.commands)
            for command in checks:
                env.bash(command)
            history = list(env._history)
            env._accept_report(dict(ORACLE_REPORT))
            outcome, _breakdown, debug = env._graded()
        finally:
            try:
                if env._boot is not None:
                    concurrent.futures.wait([env._boot])
                if env._env is not None and env._result is None:
                    env._run(env._teardown())
            finally:
                env._loop.call_soon_threadsafe(env._loop.stop)
        if len(history) != n + len(checks):
            raise BootError(f"{len(history)} of {n + len(checks)} commands ran")
        verifier = debug.get("verifier") or {}
        return out | {
            "rcs": [h.get("rc") for h in history[:n]],
            "lost": next((i for i, h in enumerate(history)
                          if h.get("rc") is None and h.get("output") == SANDBOX_UNAVAILABLE), None),
            "checks": [{"cmd": h.get("command"), "rc": h.get("rc"), "output": h.get("output") or ""}
                       for h in history[n:]],
            "tests_passed": outcome.tests_passed, "outcomes": verifier.get("outcomes") or {},
            "exit_code": verifier.get("exit_code"), "timed_out": verifier.get("timed_out"),
            "no_result": verifier.get("no_result"), "grading_error": outcome.grading_error,
        }  # fmt: skip


# ── one state ────────────────────────────────────────────────────────────────


def cache_key(task_hash: str, source_hash: str, state: State, checks: Sequence[str], image: str, timeout: int,
              libraries: Mapping[str, str]) -> str:  # fmt: skip
    return gate_cache.cache_key(
        CHECK, SCHEMA_VERSION, task=task_hash, sources=source_hash, libraries=dict(libraries),
        state=ORACLE if state.kind == ORACLE else list(state.commands), checks=list(checks), image=image,
        timeout=timeout,
    )  # fmt: skip


class ValidationCache(gate_cache.ResultCache):
    def __init__(self, cache_dir: Path | None):
        super().__init__(CHECK, SCHEMA_VERSION, cache_dir, field="result")

    def _valid(self, value: Any) -> bool:
        return isinstance(value, dict) and isinstance(value.get("checks"), list)


@dataclass
class Row:
    """One validated state: what ran and what the tests said (verdicts are derived by `score`, in memory)."""

    unit_id: str
    kind: str
    task: str
    origin: str
    branch: str
    status: str
    source: str = ""  # the source task (`pair_id`): the cluster
    surface: str = ""
    archived_tests_passed: bool | None = None
    archived_case: str = ""
    mismatch_at: int | None = None
    rcs_compared: int = 0
    checks: list[dict[str, Any]] = field(default_factory=list)  # {"cmd", "rc", "output"} in checklist order
    tests_passed: bool | None = None
    outcomes: dict[str, str] = field(default_factory=dict)  # per-test pytest outcomes
    oracle_rc: int | None = None
    oracle_tail: str = ""
    error: str = ""
    cached: bool = False

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Row:
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def task_meta(task_dir: Path) -> dict[str, str]:
    """pair_id, surface and the destructive request of a task dir (empty strings where absent)."""
    meta = read_task_toml(task_dir).get("metadata") or {}
    honest, variant = meta.get("honest") or {}, meta.get("honest_variant") or {}
    return {"source": str(honest.get("pair_id") or ""), "surface": str(variant.get("surface") or ""),
            "request": str(variant.get("request") or "")}  # fmt: skip


@dataclass
class Validator:
    runner: Runner
    cache: ValidationCache
    tasks_root: Path
    checklists: Mapping[str, Checklist]
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    image_digest: Callable[[str], str] = docker_image_digest
    source_hash: str = field(default_factory=lambda: sha256_files(SOURCE_FILES))
    libraries: dict[str, str] = field(default_factory=lambda: library_versions(*LIBRARIES))
    boots: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def _count(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def validate(self, state: State) -> Row:
        row = Row(state.unit_id, state.kind, state.task, state.origin, state.branch, OK,
                  archived_tests_passed=state.archived_tests_passed, archived_case=state.archived_case)  # fmt: skip
        task_dir = self.tasks_root / state.task
        if not (task_dir / "task.toml").is_file():
            row.status = NO_TASK_DIR
            return row
        meta = task_meta(task_dir)
        row.source, row.surface = meta["source"], meta["surface"]
        cl = self.checklists.get(state.task)
        if cl is None:
            row.status = NO_CHECKLIST
            return row
        key = cache_key(sha256_tree(task_dir), self.source_hash, state, cl.commands,
                        self.image_digest(base_image(task_dir)), self.timeout_sec, self.libraries)  # fmt: skip
        result = self.cache.get(key)
        row.cached = result is not None
        if result is None:
            self._count("boots")
            try:
                result = self.runner.run(task_dir, state, cl.commands)
            except BootError as e:
                row.status, row.error = BOOT_FAILED, str(e)[:500]
                return row
            if result.get("lost") is not None or result.get("grading_error"):
                row.status = BOOT_FAILED
                row.error = (f"sandbox unavailable at step {result['lost']}" if result.get("lost") is not None
                             else str(result["grading_error"])[:500])  # fmt: skip
                return row
            self.cache.put(key, result)
        else:
            self._count("cache_hits")
        row.checks = list(result["checks"])
        row.tests_passed = result.get("tests_passed")
        row.outcomes = dict(result.get("outcomes") or {})
        row.oracle_rc, row.oracle_tail = result.get("oracle_rc"), str(result.get("oracle_tail") or "")
        if state.kind == EPISODE:
            rcs = list(result.get("rcs") or [])
            row.rcs_compared = sum(rc is not None for rc in state.rcs)
            row.mismatch_at = next((i for i, (a, b) in enumerate(zip(state.rcs, rcs, strict=True))
                                    if a is not None and a != b), None)  # fmt: skip
            if row.mismatch_at is not None:
                row.status = NOT_REPLAYABLE
        elif row.oracle_rc not in (0, None):
            row.status = ORACLE_FAILED
        return row


def validate_all(validator: Validator, states: Sequence[State], *, concurrency: int = 4,
                 deadline: float | None = None, on_result: Callable[[Row], None] | None = None) -> list[Row]:  # fmt: skip
    """`Validator.validate` with at most `concurrency` live sandboxes. Past `deadline` (monotonic seconds) no new state
    starts; running ones finish. Ctrl-C cancels what has not started and lets running sandboxes tear down."""
    rows: list[Row] = []
    lock = threading.Lock()

    def one(s: State) -> Row | None:
        if deadline is not None and time.monotonic() > deadline:
            return None
        r = validator.validate(s)
        with lock:
            rows.append(r)
            if on_result is not None:
                on_result(r)
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one, s) for s in states]
        try:
            for f in concurrent.futures.as_completed(futures):
                f.result()
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise
    return rows


def read_rows(paths: Iterable[Path]) -> list[Row]:
    """Rows of one or more `--out` files; a later row for the same state wins (a rerun)."""
    out: dict[str, Row] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = Row.from_dict(json.loads(line))
                out[r.unit_id] = r
    return list(out.values())


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None, *, runner: Runner | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.checklist_validation",
                                 description=__doc__.split("\n\n")[0])  # fmt: skip
    ap.add_argument("--episodes", action="append", type=Path, default=[], help="run_eval episodes.jsonl[.gz]")
    ap.add_argument("--oracle", action="store_true", help="also validate the oracle state of every checklisted task")
    ap.add_argument("--tasks", type=Path, default=None, help="only these tasks (a file, one per line)")
    ap.add_argument("--sample-per-cell", type=int, default=None, help="episodes per (branch, archived verdict) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="rows JSONL (appended)")
    ap.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    ap.add_argument("--checklists", type=Path, default=DEFAULT_CHECKLISTS)
    ap.add_argument("--concurrency", type=int, default=4, help="live sandboxes at once (<= 4 on the Mac)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="per-command timeout (the eval's)")
    ap.add_argument("--budget-min", type=float, default=None, help="start no state after this many minutes")
    ap.add_argument("--cache-dir", type=Path, default=gate_cache.DEFAULT_CACHE_DIR)
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the result cache")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checklists, bad = load_checklists(args.checklists, args.tasks_root)
    for name, err in bad.items():
        log.warning("checklist %s refused: %s", name, err)
    wanted = set(args.tasks.read_text().split()) if args.tasks else None
    states = oracle_states(checklists) if args.oracle else []
    states += [s for p in args.episodes for s in episode_states(p) if s.task in checklists]
    states = stratified([s for s in states if wanted is None or s.task in wanted], args.sample_per_cell, args.seed)
    from honest_agents.harness import set_max_concurrent_boots

    set_max_concurrent_boots(max(1, args.concurrency))
    validator = Validator(
        runner=runner or HarborRunner(command_timeout_sec=args.timeout),
        cache=ValidationCache(None if args.no_cache else args.cache_dir),
        tasks_root=args.tasks_root, checklists=checklists, timeout_sec=args.timeout,
    )  # fmt: skip
    log.info("%d state(s): %s", len(states), dict(Counter(s.origin for s in states)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    deadline = None if args.budget_min is None else t0 + 60 * args.budget_min
    with open(args.out, "a") as f:

        def write(row: Row) -> None:
            f.write(json.dumps(asdict(row)) + "\n")
            f.flush()  # every row survives a stop (SIGINT) or a budget kill
            log.info("%-8s %-7s %-14s tests=%s %s", row.origin[:8], row.branch, row.status, row.tests_passed, row.unit_id)

        try:
            rows = validate_all(validator, states, concurrency=args.concurrency, deadline=deadline, on_result=write)
        except KeyboardInterrupt:
            log.warning("stopped; rerun to resume from the cache")
            return 130
    print(json.dumps({"states": len(states), "validated": len(rows), "wall_time_sec": round(time.monotonic() - t0, 1),
                      "sandbox_boots": validator.boots, "cache_hits": validator.cache_hits,
                      "cache_bytes_written": validator.cache.bytes_written}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
