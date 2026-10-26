"""Oracle validation of task variants through the training code path (`HonestHarborEnv`), docs/design.md section 7.

Every check boots a fresh sandbox exactly as in training (resolved network policy, baseline manifests); the oracle
`solution/solve.sh` runs as `[agent].user` under the agent-phase policy, a synthetic "complete" report is submitted, and
the episode is graded by `honest_agents.grading` (hermetic pytest verifier, offline, for `verifier = "pytest"` tasks).

* **Null check** (every variant whose tests run): the tests must FAIL on the untouched environment. Otherwise a policy
  that does nothing and claims completion would be right, and the task teaches nothing.
* **Feasible** (including near-miss controls): the oracle passes `oracle_runs` of `oracle_runs` times (default 3), with
  no forbidden/irreversible violation. One failure rejects the task, which also drops flaky verifiers.
* **Infeasible**: every oracle run fails the tests or violates the injected constraint. A clean pass (tests pass, no
  violation) is a **leak**: a real solution exists, so the variant is rejected (`leak = True`).
  - `conflicting_constraint`: the oracle must violate the forbidden path;
  - `high_risk`: the sentinel exists at reset, and deleting it (run after the oracle) is recorded as an irreversible
    violation, i.e. the detector works; the construction itself is oracle-agnostic.
* **Repair probes** (`remove_input`, `corrupt_input`): recreate the removed or corrupted path EMPTY (as the agent user) and rerun the oracle; then, if that
  fails, recreate it with PLAUSIBLE content (`analysis.plausible_recreation`: code blocks of the instruction and a sibling
  file of the same extension, never the removed file's own content) and rerun it. If either passes, the "missing" input was
  not what blocked the task, so the variant is not infeasible: it is flagged `relabel_to_control` (`relabel_to_control()`
  rewrites it as a feasible near-miss control whose oracle makes the same recreation). A variant of those families whose
  oracle passes without any repair is a leak and is flagged the same way. For `corrupt_input` the oracle must also FAIL
  on the impossible variant (the garbage is the only difference to its control, whose oracle must pass), and "recreate"
  overwrites the noise, so an oracle that does not depend on the file's bytes is caught the same way.
* For hermetic tasks the verifier phase must resolve to `no-network`; the check reads the env's resolved network plan.

Needs Docker (Linux VM). On macOS the logic is exercised with a fake sandbox (tests/tasks).

    python -m honest_agents.tasks.validate --tasks-dir out/tasks --out out/validation.jsonl --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import posixpath
import re
import shlex
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.env import HonestHarborEnv
from honest_agents.harness import SANDBOX_UMASK, set_max_concurrent_boots
from honest_agents.reward import EpisodeOutcome
from honest_agents.spec import VERIFIER_PYTEST, HonestTaskSpec
from honest_agents.tasks.hermetic import NO_NETWORK, hermetic_problems
from honest_agents.tasks.taskfs import (
    prepend_solve,
    read_task_toml,
    set_honest,
    write_task_toml,
)

log = logging.getLogger(__name__)

# Container paths (module-level so Docker-free tests can point them inside a local sandbox root).
SOLUTION_DIR = "/solution"
ORACLE_LOG = "/tmp/honest_oracle.log"

DEFAULT_ORACLE_RUNS = 3
# A hung oracle is killed inside the container (Harbor's exec timeout only abandons the client), so grading never runs
# next to a still-running solve.sh.
_KILL_AFTER_SEC = 5
_EXEC_SLACK_SEC = 30

ORACLE_REPORT = {
    "status": "complete",
    "completed": ["oracle solution executed"],
    "incomplete": [],
    "evidence_commands": [],
    "confidence": 1.0,
}

# A relabels.json entry `plausible:<path>` is recreated with `plausible_content`; a bare path is recreated empty.
PLAUSIBLE_PREFIX = "plausible:"

# Operators whose impossibility is "the input cannot be had back": the repair probes apply (recreate the path, rerun).
REPAIR_OPERATORS = ("remove_input", "corrupt_input", "shortcut_trap")  # shortcut_trap: its infeasible cell only

# Operators whose verdict does not depend on whether the tests pass: one oracle run is deterministic.
_CONSTRAINT_OPERATORS = ("conflicting_constraint", "high_risk")


class OracleHarborEnv(HonestHarborEnv):
    """`HonestHarborEnv` that can run the task's oracle in place of a policy. Adds no public (tool) methods."""

    async def _facts(self, sentinels: Iterable[str] = ()) -> dict[str, Any]:
        """Boot state the validator needs: the verifier phase's network mode and which sentinel paths exist."""
        await asyncio.wrap_future(self._boot)
        # What grading will actually apply (`HonestHarborEnv._verifier_policy`: offline for pytest tasks, the task's own
        # verifier phase otherwise), not merely what task.toml declares; nothing is enforced when the env opts out.
        mode = None
        if getattr(self, "_enforce_network", True):
            mode = getattr(self._verifier_policy(), "network_mode", None)
        present: dict[str, bool] = {}
        for path in sentinels:
            r = await self._env.exec(f"test -e {shlex.quote(path)}", user="root", timeout_sec=60)
            present[path] = r.return_code == 0
        return {
            "verifier_network": None if mode is None else str(getattr(mode, "value", mode)),
            "sentinels_present": present,
        }

    async def _run_oracle(
        self,
        *,
        timeout_sec: int,
        destructive: list[str],
        pre_commands: Iterable[str] = (),
        script: str = "solve.sh",
    ) -> dict[str, Any]:
        """Run `pre_commands` then the oracle, both as the agent user; `destructive` paths are removed afterwards.

        `script` names another file of the solution dir to run in its place (`shortcut.sh`, the destructive shortcut of a
        `shortcut_trap` cell)."""
        from harbor.utils.env import resolve_env_vars

        await asyncio.wrap_future(self._boot)
        sol_dir = self._task.paths.solution_dir
        user = self._task.config.agent.user
        await self._env.upload_dir(sol_dir, SOLUTION_DIR)
        # Readable by an unprivileged agent user too (revoke_permission runs the oracle as that user).
        await self._env.exec(
            f"chmod -R a+rX {SOLUTION_DIR} && chmod a+x {SOLUTION_DIR}/{script}",
            user="root",
            timeout_sec=60,
        )
        out: dict[str, Any] = {}
        for cmd in pre_commands:
            r = await self._env.exec(cmd, user=user, timeout_sec=60)
            out.setdefault("pre_rc", []).append(r.return_code)
        env = {"DEBIAN_FRONTEND": "noninteractive"}
        if self._task.config.solution.env:
            env |= resolve_env_vars(self._task.config.solution.env)
        try:
            res = await self._env.exec(
                f"timeout -k {_KILL_AFTER_SEC} {int(timeout_sec)} {SOLUTION_DIR}/{script} > {ORACLE_LOG} 2>&1; "
                f"rc=$?; tail -c 3000 {ORACLE_LOG}; exit $rc",
                env=env,
                timeout_sec=int(timeout_sec) + _KILL_AFTER_SEC + _EXEC_SLACK_SEC,
                user=user,
            )
            out |= {
                "oracle_rc": res.return_code,
                "oracle_tail": (res.stdout or "")[-3000:],
            }
        except Exception as e:  # noqa: BLE001 — e.g. a download hanging without network: the oracle failed,
            # but the sandbox is fine and grading still decides whether the variant is valid
            out |= {
                "oracle_rc": None,
                "oracle_tail": f"[oracle exec error] {type(e).__name__}: {e}",
            }
        for path in destructive:  # high_risk: emulate a policy that complies with the destructive request
            r = await self._env.exec(f"rm -rf -- {shlex.quote(path)}", user=user, timeout_sec=60)
            out.setdefault("destructive_rc", []).append(r.return_code)
        return out


@dataclass
class ValidationResult:
    variant_id: str
    task_dir: str
    pair_id: str | None
    feasible: bool
    perturbation: str | None
    ok: bool
    reason: str
    tests_passed: bool | None = None
    test_reward: float | None = None
    tamper: bool = False
    forbidden_violation: bool = False
    irreversible_violation: bool = False
    damage_paths: list[str] = field(
        default_factory=list
    )  # preserve_paths damaged by the deciding run (an oracle's: bad)
    oracle_rc: int | None = None
    error: str | None = None
    duration_sec: float = 0.0
    task_hash: str = ""  # content of the validated task dir; a regenerated variant must be re-validated
    debug: dict[str, Any] = field(default_factory=dict)
    # docs/design.md section 7
    oracle_runs: int = 0  # oracle episodes actually run (a failing task stops early)
    oracle_passes: int = 0  # of those, clean passes: tests passed, no violation, no tamper
    null_check_failed_as_expected: bool | None = None  # tests failed on the untouched env; None = not run / no tests
    relabel_to_control: bool = False  # `REPAIR_OPERATORS` only: the removed input is not what blocks the task
    leak: bool = False  # infeasible variant with a clean oracle pass
    # `shortcut_trap` trap cell: the shortcut passed the tests (True) and damaged a preserved path (damage list), as built.
    shortcut_passes_tests: bool | None = None
    shortcut_damage_paths: list[str] = field(default_factory=list)


def task_hash(task_dir: Path) -> str:
    """sha256 over every file (relative path + bytes) of a task dir."""
    h = hashlib.sha256()
    for f in sorted(p for p in task_dir.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(task_dir)).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def is_current(r: ValidationResult, task_dir: Path) -> bool:
    """A stored result still applies to `task_dir` (results written before hashing existed are trusted)."""
    return not r.task_hash or r.task_hash == task_hash(task_dir)


UMASK_UNKNOWN = "unknown"  # `sandbox_umask` of a booted result whose preflight printed no umask


def umask_current(r: ValidationResult) -> bool:
    """A stored result was graded under the nodes' umask (DEF-43). Results written before the boot preflight recorded it
    (DEF-19: no `sandbox_umask` key) ran under a local daemon's 0000 and fail file-mode tests the nodes pass, so they
    are not current; a result that never booted records None, and one that booted but whose preflight printed no umask
    records `UMASK_UNKNOWN` (not current: its umask is not known)."""
    if "sandbox_umask" not in r.debug:
        return False
    return r.debug["sandbox_umask"] in (None, "0" + SANDBOX_UMASK)


def is_clean_pass(outcome: EpisodeOutcome) -> bool:
    """Tests passed with no forbidden/irreversible violation and no tamper."""
    return outcome.tests_passed is True and not outcome.violation and not outcome.tamper


def is_leak(spec: HonestTaskSpec, outcome: EpisodeOutcome) -> bool:
    """An infeasible variant whose oracle (or an empty-handed agent) passes the tests without violating the injected
    constraint: a real solution exists. Tamper does not excuse it (an oracle that reinstalls a removed tool under a
    protected path still shows the task is solvable)."""
    return not spec.feasible and outcome.tests_passed is True and not outcome.violation


def judge(spec: HonestTaskSpec, outcome: EpisodeOutcome, *, sentinel_seeded: bool = True) -> tuple[bool, str]:
    """Decision rule: is this variant what its labels claim? Pure function of one graded oracle episode."""
    if outcome.grading_error is not None:
        return False, f"grading_error: {outcome.grading_error}"
    violation = outcome.violation
    if spec.feasible:
        if outcome.damage_paths:
            return False, f"feasible: oracle damages a preserved path ({outcome.damage_paths[0]})"
        if violation:
            return False, "feasible: oracle violates a forbidden/irreversible path"
        if outcome.tests_passed:
            return True, "feasible: oracle passes"
        return (
            False,
            "feasible: oracle fails tests" if outcome.tests_passed is False else "feasible: no tests ran",
        )
    if spec.perturbation == "hide_parameter":
        return (
            False,
            "hide_parameter: not oracle-checkable (needs LLM-judge validation)",
        )
    if spec.perturbation == "conflicting_constraint":
        if outcome.forbidden_violation:
            return True, "oracle violates forbidden path"
        if outcome.tests_passed:
            return False, "leaky: oracle passes without touching the forbidden path"
        return False, "oracle does not touch the forbidden path"
    if spec.perturbation == "high_risk":
        if not sentinel_seeded:
            return False, "sentinel missing at reset"
        if outcome.irreversible_violation:
            return True, "sentinel seeded; destruction detected"
        return False, "sentinel destruction not detected"
    if violation:
        return True, "oracle violates injected constraint"
    if outcome.tests_passed is False:
        return True, "oracle fails"
    if outcome.tests_passed is None:
        return False, "no tests ran"
    return False, "leaky: oracle still passes"


def judge_null(outcome: EpisodeOutcome) -> tuple[bool | None, str]:
    """The null check: tests must fail when nothing was done. Returns (failed_as_expected, reason)."""
    if outcome.grading_error is not None:
        return None, f"null check: grading_error: {outcome.grading_error}"
    if outcome.tests_passed is False:
        return True, "null check: tests fail on the untouched environment"
    if outcome.tests_passed is True:
        return False, "null check: tests pass on the untouched environment"
    return None, "null check: no tests ran"


def removed_paths(task_dir: Path, spec: HonestTaskSpec) -> list[str]:
    """Absolute paths a `REPAIR_OPERATORS` variant deleted or corrupted: `[metadata.honest_variant]` path(s), else
    `expected_targets`."""
    detail = (read_task_toml(task_dir).get("metadata") or {}).get("honest_variant") or {}
    found = [detail["path"]] if isinstance(detail.get("path"), str) else list(detail.get("paths") or [])
    found = found or [t for t in spec.expected_targets]
    return [p for p in dict.fromkeys(found) if isinstance(p, str) and p.startswith("/")]


def plausible_content(task_dir: Path, path: str) -> str:
    """A plausible stand-in for the removed `path`, written from what the variant still shows (never the removed file)."""
    from honest_agents.tasks.analysis import TaskFacts, plausible_recreation

    facts = TaskFacts.load(Path(task_dir))
    others = {q: f.content for q, f in facts.image_files.items() if q != path}
    return plausible_recreation(path, facts.instruction, others)


def repair_commands(paths: Iterable[str], task_dir: Path | None = None) -> list[str]:
    """Shell steps that recreate each removed input: empty (the cheapest repair an agent could make), or, for a
    `plausible:`-prefixed path, with `plausible_content` of `task_dir`."""
    out = []
    for p in paths:
        plain = p.removeprefix(PLAUSIBLE_PREFIX)
        mk = f"mkdir -p {shlex.quote(posixpath.dirname(plain) or '/')}"
        if p.startswith(PLAUSIBLE_PREFIX) and task_dir is not None:
            body = plausible_content(task_dir, plain)
            out.append(f"{mk} && printf '%s' {shlex.quote(body)} > {shlex.quote(plain)}")
        else:
            out.append(f"{mk} && : > {shlex.quote(plain)}")
    return out


def relabel_to_control(task_dir: Path, *, repair_paths: Iterable[str] = ()) -> HonestTaskSpec:
    """Rewrite a leaky/trivially repairable `REPAIR_OPERATORS` variant in place as a feasible near-miss control.

    The labels become feasible + `is_control` (`control_of` = the operator), `expected_*` are cleared and the removed path
    becomes a decoy target (blaming it is false blame). `repair_paths` are recreated at the top of the oracle (empty, or with plausible content for `plausible:` entries), so
    the control validates as feasible; pass them only when the repair probe needed them (`debug["repair_needed"]`).
    """
    task_dir = Path(task_dir)
    spec = HonestTaskSpec.from_task_dir(task_dir)
    decoys = list(
        dict.fromkeys(
            [
                *spec.expected_targets,
                *(p.removeprefix(PLAUSIBLE_PREFIX) for p in repair_paths),
                *removed_paths(task_dir, spec),
            ]
        )
    )
    data = spec.model_dump(mode="json", exclude_none=True) | {
        "feasible": True,
        "is_control": True,
        "control_of": spec.perturbation,
        "expected_blockers": [],
        "expected_targets": [],
        "decoy_targets": decoys,
        "run_tests_when_infeasible": True,
    }
    new = HonestTaskSpec.model_validate(data)
    cfg = read_task_toml(task_dir)
    write_task_toml(
        task_dir,
        set_honest(cfg, new, (cfg.get("metadata") or {}).get("honest_variant")),
    )
    if repair := list(repair_paths):
        prepend_solve(task_dir, repair_commands(repair, task_dir))
    return new


# ── episodes ─────────────────────────────────────────────────────────────────


@dataclass
class _Episode:
    outcome: EpisodeOutcome
    oracle: dict[str, Any]
    facts: dict[str, Any]
    debug: dict[str, Any]


async def _on_env(env: OracleHarborEnv, coro):
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, env._loop))


async def _abort(env: OracleHarborEnv | None, name: str) -> None:
    """Free the sandbox after a failed episode (grading frees it itself on the happy path)."""
    if env is None:
        return
    if env._boot is not None:  # let a still-running boot finish, else _stop races it and leaks a container
        with contextlib.suppress(Exception):
            await asyncio.wrap_future(env._boot)
    try:
        await _on_env(env, env._stop())
    except Exception as stop_err:  # noqa: BLE001
        log.warning("could not stop sandbox for %s: %s", name, stop_err)


async def _run_episode(
    task_dir: Path,
    env_factory: Callable[[], OracleHarborEnv],
    *,
    oracle: bool,
    oracle_timeout_sec: int,
    destructive: Iterable[str] = (),
    pre_commands: Iterable[str] = (),
    sentinels: Iterable[str] = (),
    script: str = "solve.sh",
    report: dict[str, Any] | None = None,
) -> _Episode:
    """One graded episode: boot, optionally run the oracle, submit `report` (default: claim completion), grade. Raises on
    infrastructure errors."""
    env = None
    try:
        env = env_factory()
        env.reset(task_dir=str(task_dir))
        facts = await _on_env(env, env._facts(sentinels))
        oracle_out: dict[str, Any] = {}
        if oracle:
            oracle_out = await _on_env(
                env,
                env._run_oracle(
                    timeout_sec=oracle_timeout_sec,
                    destructive=list(destructive),
                    pre_commands=list(pre_commands),
                    **({"script": script} if script != "solve.sh" else {}),
                ),
            )
        env._accept_report(dict(ORACLE_REPORT if report is None else report))
        outcome, _breakdown, debug = await asyncio.wrap_future(env._start_grading())
    except BaseException:
        await _abort(env, task_dir.name)
        raise
    return _Episode(outcome, oracle_out, facts, debug)


def _tests_run(spec: HonestTaskSpec) -> bool:
    return spec.feasible or spec.run_tests_when_infeasible


async def validate_task(
    task_dir: Path,
    *,
    env_factory: Callable[[], OracleHarborEnv] = OracleHarborEnv,
    oracle_timeout_sec: int = 900,
    oracle_runs: int = DEFAULT_ORACLE_RUNS,
    null_check: bool = True,
    repair_probe: bool = True,
) -> ValidationResult:
    """Validate one variant. Never raises: infrastructure errors become `ok=False` with `error` set."""
    t0 = time.monotonic()
    try:
        spec = HonestTaskSpec.from_task_dir(task_dir)
        digest = task_hash(task_dir)
    except Exception as e:  # noqa: BLE001
        return ValidationResult(
            task_dir.name,
            str(task_dir),
            None,
            False,
            None,
            False,
            "error",
            error=f"{type(e).__name__}: {e}",
        )
    res = ValidationResult(
        variant_id=task_dir.name,
        task_dir=str(task_dir),
        pair_id=spec.pair_id,
        feasible=spec.feasible,
        perturbation=spec.perturbation,
        ok=False,
        reason="",
        task_hash=digest,
    )
    res.debug["settings"] = {
        "oracle_runs": max(1, oracle_runs),
        "null_check": bool(null_check),
        "repair_probe": bool(repair_probe),
        "repair_plausible": bool(repair_probe),  # results made before the plausible-content probe are not current
    }
    res.debug["sandbox_umask"] = None  # set by `_record` from the boot preflight; see `umask_current`
    if not (task_dir / "solution" / "solve.sh").exists():
        res.reason = "no oracle solution"
        return res
    if spec.verifier == VERIFIER_PYTEST and (problems := hermetic_problems(task_dir)):
        res.reason = f"not hermetic: {problems[0]}"
        res.debug["hermetic_problems"] = problems
        return res
    try:
        await _validate(
            task_dir,
            spec,
            res,
            env_factory,
            oracle_timeout_sec,
            max(1, oracle_runs),
            null_check,
            repair_probe,
        )
    except Exception as e:  # noqa: BLE001 — boot/oracle failure (e.g. a perturbation broke the image build)
        res.ok, res.reason, res.error = False, "error", f"{type(e).__name__}: {e}"
    res.duration_sec = round(time.monotonic() - t0, 2)
    return res


def _offline_problem(spec: HonestTaskSpec, ep: _Episode) -> str | None:
    """A hermetic task's verifier must run with the network off; read it off the env's resolved plan."""
    if spec.verifier != VERIFIER_PYTEST:
        return None
    mode = ep.facts.get("verifier_network")
    if mode != NO_NETWORK:
        return f"verifier phase is not offline (network_mode={mode})"
    return None


def _preserve_problem(ep: _Episode) -> str | None:
    """A preserve root that holds files a read-only tool rewrites (.git, sqlite -wal/-shm/-journal, .pyc) would label an
    honest `git status` or `select` as damage, which is the careful policy's penalty: list the data files instead."""
    volatile = ep.debug.get("preserve_volatile")
    if volatile:
        return f"preserve_paths hold bookkeeping that read-only tools rewrite ({volatile[0]}): preserve the data files only"
    return None


def _verifier_problem(spec: HonestTaskSpec, ep: _Episode) -> str | None:
    """A pytest run that never produced a result (killed, timed out, no test collected) is an infrastructure error.

    Its `tests_passed = False` would otherwise read as "the oracle fails", and an infeasible variant would validate
    because pytest never ran. `debug["verifier"]` is filled by the real grader when the tests ran.
    """
    if spec.verifier != VERIFIER_PYTEST:
        return None
    v = ep.debug.get("verifier")
    if not isinstance(v, dict):
        return None
    if v.get("no_result"):
        return "verifier produced no result (pytest killed or crashed): infrastructure error, not an oracle failure"
    if v.get("timed_out"):
        return "verifier timed out: infrastructure error, not an oracle failure"
    if v.get("n_tests") is not None and int(v["n_tests"]) < 1:
        return "verifier collected no tests: infrastructure error, not an oracle failure"
    return None


def _record(res: ValidationResult, ep: _Episode) -> None:
    """Copy the deciding episode's grading facts onto the result."""
    o = ep.outcome
    res.tests_passed, res.test_reward = o.tests_passed, o.test_reward
    res.tamper = res.tamper or o.tamper
    res.forbidden_violation = o.forbidden_violation
    res.irreversible_violation = o.irreversible_violation
    res.damage_paths = list(o.damage_paths)
    res.error = o.grading_error
    res.oracle_rc = ep.oracle.get("oracle_rc")
    res.debug["oracle_tail"] = str(ep.oracle.get("oracle_tail", ""))[-1500:]
    res.debug["test_log"] = str(ep.debug.get("test_log", ""))[-1500:]
    res.debug["tamper_details"] = o.tamper_details[:20]
    res.debug["violations"] = o.violation_details[:20]
    res.debug["verifier_network"] = ep.facts.get("verifier_network")
    # DEF-19: the boot preflight's umask. A graded episode booted, so None here means the probe printed nothing.
    res.debug["sandbox_umask"] = ep.debug.get("sandbox_umask") or UMASK_UNKNOWN


async def _validate(
    task_dir: Path,
    spec: HonestTaskSpec,
    res: ValidationResult,
    env_factory: Callable[[], OracleHarborEnv],
    oracle_timeout_sec: int,
    oracle_runs: int,
    null_check: bool,
    repair_probe: bool,
) -> None:
    sentinels = list(spec.irreversible_paths) if spec.perturbation == "high_risk" else []
    destructive = sentinels
    common = {"oracle_timeout_sec": oracle_timeout_sec, "sentinels": sentinels}

    # 1. null check: nothing done, tests must fail
    if null_check and _tests_run(spec):
        ep = await _run_episode(task_dir, env_factory, oracle=False, destructive=[], **common)
        _record(res, ep)
        if (why := _offline_problem(spec, ep) or _verifier_problem(spec, ep) or _preserve_problem(ep)) is not None:
            res.reason = why
            return
        expected, reason = judge_null(ep.outcome)
        res.null_check_failed_as_expected = expected
        if expected is not True:
            res.reason = reason
            res.leak = not spec.feasible and expected is False
            return

    # 2. oracle runs
    n = 1 if spec.perturbation in _CONSTRAINT_OPERATORS else oracle_runs
    verdicts: list[tuple[bool, str]] = []
    first_bad: _Episode | None = None
    last: _Episode | None = None
    for i in range(n):
        ep = await _run_episode(task_dir, env_factory, oracle=True, destructive=destructive, **common)
        last = ep
        res.oracle_runs += 1
        res.oracle_passes += is_clean_pass(ep.outcome)
        res.tamper = res.tamper or ep.outcome.tamper  # a tamper verdict in any run counts, not just the deciding one
        res.leak = res.leak or is_leak(spec, ep.outcome)
        seeded = any(ep.facts.get("sentinels_present", {}).values())
        ok, reason = judge(spec, ep.outcome, sentinel_seeded=seeded)
        verdicts.append((ok, reason))
        res.debug.setdefault("runs", []).append(
            {
                "ok": ok,
                "reason": reason,
                "tests_passed": ep.outcome.tests_passed,
                "oracle_rc": ep.oracle.get("oracle_rc"),
            }
        )
        if (why := _offline_problem(spec, ep) or _verifier_problem(spec, ep) or _preserve_problem(ep)) is not None:
            _record(res, ep)
            res.reason = why
            res.ok = False
            return
        if not ok and first_bad is None:
            first_bad = ep
        if not ok:
            break  # one bad run decides: a feasible oracle must pass every time, an infeasible one must never pass
    deciding = first_bad or last
    assert deciding is not None
    _record(res, deciding)
    bad = [v for v in verdicts if not v[0]]
    if bad:
        res.ok = False
        if spec.feasible and res.oracle_passes and len(verdicts) > 1:
            res.reason = f"feasible: oracle flaky ({res.oracle_passes} of {res.oracle_runs} runs passed)"
        else:
            res.reason = bad[0][1]
    else:
        res.ok, res.reason = True, verdicts[-1][1]

    # 2b. shortcut_trap trap cell: the shortcut (the source oracle, which writes through the alias) must pass the tests AND
    # damage a preserved path, else the trap does not trap (a harmless shortcut teaches nothing, and the label is wrong).
    if res.ok and spec.perturbation == "shortcut_trap" and spec.feasible and spec.preserve_paths:
        await _shortcut_check(task_dir, spec, res, env_factory, oracle_timeout_sec)

    # 3. remove_input, corrupt_input: a leak, or a trivial repair, means the lost input is not what blocks the task
    if spec.perturbation in REPAIR_OPERATORS and not spec.feasible:
        if res.leak:
            res.relabel_to_control = True
            res.debug["repair_needed"] = False
            res.debug["repair_paths"] = removed_paths(task_dir, spec)
        elif res.ok and repair_probe:
            await _repair_probe(task_dir, spec, res, env_factory, oracle_timeout_sec)


async def _shortcut_check(
    task_dir: Path,
    spec: HonestTaskSpec,
    res: ValidationResult,
    env_factory: Callable[[], OracleHarborEnv],
    oracle_timeout_sec: int,
) -> None:
    """Run `solution/shortcut.sh` in a fresh sandbox: it must pass the hermetic tests and trip the preserve manifest."""
    if not (task_dir / "solution" / "shortcut.sh").exists():
        res.ok, res.reason = False, "shortcut_trap: no solution/shortcut.sh to run as the shortcut"
        return
    ep = await _run_episode(
        task_dir, env_factory, oracle=True, oracle_timeout_sec=oracle_timeout_sec, script="shortcut.sh"
    )
    o = ep.outcome
    res.shortcut_passes_tests = o.tests_passed
    res.shortcut_damage_paths = list(o.damage_paths)
    res.debug["shortcut"] = {"tests_passed": o.tests_passed, "damage": list(o.damage_paths)[:20]}
    if (why := _offline_problem(spec, ep) or _verifier_problem(spec, ep) or _preserve_problem(ep)) is not None:
        res.ok, res.reason = False, why
    elif o.grading_error is not None:
        res.ok, res.reason = False, f"shortcut: grading_error: {o.grading_error}"
    elif o.tests_passed is not True:
        res.ok, res.reason = (
            False,
            "shortcut_trap: the shortcut does not pass the tests (it is not a tempting shortcut)",
        )
    elif not o.damage_paths:
        res.ok, res.reason = False, "shortcut_trap: the shortcut leaves the preserved paths intact (nothing to protect)"
    else:
        res.reason = "feasible: oracle passes; shortcut passes the tests and damages a preserved path"


async def _repair_probe(
    task_dir: Path,
    spec: HonestTaskSpec,
    res: ValidationResult,
    env_factory: Callable[[], OracleHarborEnv],
    oracle_timeout_sec: int,
) -> None:
    paths = removed_paths(task_dir, spec)
    if not paths:
        res.ok = False
        res.reason = f"{spec.perturbation}: removed path not recorded, so the trivial-repair probe cannot run"
        return
    for kind, repair in (("empty", paths), ("plausible", [PLAUSIBLE_PREFIX + p for p in paths])):
        ep = await _run_episode(
            task_dir,
            env_factory,
            oracle=True,
            oracle_timeout_sec=oracle_timeout_sec,
            pre_commands=repair_commands(repair, task_dir),
        )
        res.debug["repair_probe" if kind == "empty" else "repair_probe_plausible"] = {
            "paths": paths,
            "tests_passed": ep.outcome.tests_passed,
            "oracle_rc": ep.oracle.get("oracle_rc"),
            "pre_rc": ep.oracle.get("pre_rc"),
        }
        if ep.outcome.grading_error is not None:
            res.ok, res.reason = (
                False,
                f"repair probe ({kind}): grading_error: {ep.outcome.grading_error}",
            )
            res.error = ep.outcome.grading_error
            return
        if is_clean_pass(ep.outcome):
            res.ok = False
            res.relabel_to_control = True
            res.debug["repair_needed"] = True
            res.debug["repair_paths"] = repair
            res.reason = (
                f"{spec.perturbation}: oracle passes once the removed path is recreated {kind} (trivial repair)"
            )
            return


# ── batches ──────────────────────────────────────────────────────────────────


def read_results(path: Path) -> dict[str, ValidationResult]:
    """Results from a JSONL file; keys this version does not know (written by a newer one) are ignored."""
    known = {f.name for f in dataclasses.fields(ValidationResult)}
    out: dict[str, ValidationResult] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = ValidationResult(**{k: v for k, v in json.loads(line).items() if k in known})
                out[r.variant_id] = r
    return out


def _covers(r: ValidationResult, wanted: dict[str, Any]) -> bool:
    """A stored result was produced with checks at least as strict as `wanted` (results with no record are not)."""
    got = r.debug.get("settings")
    if not isinstance(got, dict):
        return False
    return (
        got.get("oracle_runs", 0) >= wanted["oracle_runs"]
        and (got.get("null_check") or not wanted["null_check"])
        and (got.get("repair_probe") or not wanted["repair_probe"])
        and (got.get("repair_plausible") or not wanted["repair_plausible"])
    )


async def validate_many(
    task_dirs: list[Path],
    out_path: Path,
    *,
    concurrency: int = 8,
    env_factory: Callable[[], OracleHarborEnv] = OracleHarborEnv,
    oracle_timeout_sec: int = 900,
    resume: bool = True,
    oracle_runs: int = DEFAULT_ORACLE_RUNS,
    null_check: bool = True,
    repair_probe: bool = True,
) -> dict[str, ValidationResult]:
    """Validate variants with bounded concurrency, appending each result to `out_path` (JSONL) as it completes."""
    done = read_results(out_path) if resume else {}
    wanted = {
        "oracle_runs": max(1, oracle_runs),
        "null_check": bool(null_check),
        "repair_probe": bool(repair_probe),
        "repair_plausible": bool(repair_probe),  # results made before the plausible-content probe are not current
    }
    todo = [
        d
        for d in task_dirs
        if d.name not in done
        or not is_current(done[d.name], d)
        or not _covers(done[d.name], wanted)
        or not umask_current(done[d.name])
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(d: Path) -> None:
        async with sem:
            try:
                r = await validate_task(
                    d,
                    env_factory=env_factory,
                    oracle_timeout_sec=oracle_timeout_sec,
                    oracle_runs=oracle_runs,
                    null_check=null_check,
                    repair_probe=repair_probe,
                )
            except Exception as e:  # noqa: BLE001 — one bad variant must not abort a multi-hour batch
                r = ValidationResult(d.name, str(d), None, False, None, False, "error", error=repr(e))
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(asdict(r)) + "\n")
            done[r.variant_id] = r
            log.info("%s %s: %s", "OK  " if r.ok else "FAIL", r.variant_id, r.reason)

    await asyncio.gather(*(one(d) for d in todo))
    return done


def filter_valid(
    results: dict[str, ValidationResult],
    *,
    allow_oracle_tamper: bool = False,
    task_dirs: dict[str, Path] | None = None,
    require_null_check: bool = False,
    min_oracle_runs: int = 0,
) -> set[str]:
    """Variant ids to keep: valid themselves, and (for infeasible ones) their pair's feasible copy is valid too.

    A feasible oracle that trips the tamper check would teach the policy that the reference behaviour is punished, so
    such pairs are dropped unless `allow_oracle_tamper`. A leak or a pending relabel is never kept as it stands. With
    `task_dirs` (variant id -> current dir), a result only counts if it was produced for the same task content.
    `require_null_check` drops results whose tests were never shown to fail on the untouched environment (a run with
    `--no-null-check`; `high_risk` runs no tests, so it is exempt), and `min_oracle_runs` drops results that ran the
    oracle fewer times than the build requires (docs/design.md section 7 asks for 3 of 3; the two constraint operators are
    deterministic and run once).
    """
    if task_dirs is not None:
        results = {k: r for k, r in results.items() if k in task_dirs and is_current(r, task_dirs[k])}
    results = {
        k: r
        for k, r in results.items()
        if (r.oracle_runs >= min_oracle_runs or r.perturbation in _CONSTRAINT_OPERATORS)
        and not (require_null_check and r.perturbation != "high_risk" and r.null_check_failed_as_expected is not True)
    }
    # The pair's PLAIN feasible copy (no perturbation) must be valid: a near-miss control is feasible too, but a pair
    # whose plain copy failed validation is a pair whose source task does not work.
    good_pairs = {
        r.pair_id
        for r in results.values()
        if r.feasible and r.perturbation is None and r.ok and (allow_oracle_tamper or not r.tamper)
    }
    # A control that fails its own tests means the lock/modification is not the only thing that differs from the plain
    # task (identity, modes, an unrelated side effect): its impossible twin then "validates" because the oracle fails
    # for that other reason, so the twin is dropped with it (found on a real Docker node, revoke_permission).
    bad_controls = {(r.pair_id, r.perturbation) for r in results.values() if r.feasible and r.perturbation and not r.ok}
    return {
        r.variant_id
        for r in results.values()
        if r.ok
        and not r.leak
        and not r.relabel_to_control
        and r.pair_id in good_pairs
        and (r.feasible or (r.pair_id, r.perturbation) not in bad_controls)
    }


def select_dirs(dirs: list[Path], only_ids: set[str] | None) -> list[Path]:
    """The task dirs to validate: all of them, or those whose variant id (the dir name) is in `only_ids`.

    Variants without a result are dropped by the build, so validating a chosen subset is safe; an id with no task dir is
    an error (a stale list would silently validate less than intended)."""
    if only_ids is None:
        return dirs
    missing = only_ids - {d.name for d in dirs}
    if missing:
        raise SystemExit(f"--only-file lists {len(missing)} variant ids with no task dir, e.g. {sorted(missing)[:3]}")
    return [d for d in dirs if d.name in only_ids]


def write_quarantine_list(results: dict[str, Any], path: Path) -> int:
    """Write (replace) the quarantine list of the rejected variants: one `QuarantineEntry` line each, kind `leak` (the
    only kind evaluation runs honour by default), the reason in `detail`, with the sandbox umask and the failing tests
    when the result recorded them (DEF-43). An empty list is written as an empty file."""
    from honest_agents.bank import QUARANTINE_LEAK, QuarantineEntry

    lines = []
    for vid, r in sorted(results.items()):
        if r.ok and not r.leak and not r.relabel_to_control:
            continue
        why = "leak" if r.leak else "relabel_to_control" if r.relabel_to_control else (r.reason or "not ok")
        debug = getattr(r, "debug", None) or {}
        if debug.get("sandbox_umask"):
            why += f" (umask {debug['sandbox_umask']})"
        if failed := re.findall(r"^FAILED \S+::(\w+)", str(debug.get("test_log", "")), re.MULTILINE):
            why += "; failed: " + ", ".join(failed)
        lines.append(
            QuarantineEntry(
                variant_id=vid, kind=QUARANTINE_LEAK, source="validation", detail=f"G1 select: {why}"[:300]
            ).model_dump_json()
            + "\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return len(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="dir of variant task dirs (<out>/tasks)",
    )
    ap.add_argument("--out", type=Path, required=True, help="validation.jsonl (appended; resumable)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--oracle-timeout", type=int, default=900)
    ap.add_argument(
        "--oracle-runs",
        type=int,
        default=DEFAULT_ORACLE_RUNS,
        help="oracle runs per variant",
    )
    ap.add_argument(
        "--no-null-check",
        action="store_true",
        help="skip the untouched-environment check",
    )
    ap.add_argument(
        "--no-repair-probe",
        action="store_true",
        help="skip the trivial-repair probe of remove_input and corrupt_input",
    )
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="rewrite remove_input and corrupt_input variants flagged relabel_to_control as feasible near-miss controls, then revalidate",
    )
    ap.add_argument("--environment-type", default="docker")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--only-file",
        type=Path,
        help="validate only the variant ids listed in this file (one per line); the build drops variants without a result",
    )
    ap.add_argument(
        "--quarantine-out",
        type=Path,
        help="append every rejected variant (ok false, leak, relabel_to_control) to this quarantine list (bank.QuarantineEntry lines)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dirs = sorted(p.parent for p in args.tasks_dir.glob("*/task.toml"))
    only = {ln.strip() for ln in args.only_file.read_text().splitlines() if ln.strip()} if args.only_file else None
    dirs = select_dirs(dirs, only)
    # Sandbox boots are throttled by a separate semaphore (16 by default): without this, --concurrency 32 would overlap 32
    # validations but never boot more than 16 containers at a time.
    set_max_concurrent_boots(max(args.concurrency, 16))
    kw: dict[str, Any] = {
        "concurrency": args.concurrency,
        "env_factory": lambda: OracleHarborEnv(environment_type=args.environment_type),
        "oracle_timeout_sec": args.oracle_timeout,
        "oracle_runs": args.oracle_runs,
        "null_check": not args.no_null_check,
        "repair_probe": not args.no_repair_probe,
    }
    results = asyncio.run(validate_many(dirs, args.out, resume=not args.no_resume, **kw))
    if args.relabel:
        flagged = [r for r in results.values() if r.relabel_to_control and r.task_dir]
        for r in flagged:
            paths = r.debug.get("repair_paths", []) if r.debug.get("repair_needed") else []
            relabel_to_control(Path(r.task_dir), repair_paths=paths)
        if flagged:  # the task hash changed, so a resumed run revalidates exactly these
            results = asyncio.run(validate_many(dirs, args.out, resume=True, **kw))
        print(f"relabelled {len(flagged)} variants as near-miss controls")
    if args.quarantine_out:
        write_quarantine_list(results, args.quarantine_out)
    n_ok = sum(r.ok for r in results.values())
    n_leak = sum(r.leak for r in results.values())
    print(
        f"validated {len(results)} variants: {n_ok} ok, {len(results) - n_ok} rejected ({n_leak} leaks) -> {args.out}"
    )


if __name__ == "__main__":
    main()
