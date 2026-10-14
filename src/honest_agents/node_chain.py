"""A chain of node jobs: dependency order, GPU-aware parallelism, done markers, one state file, a final archive step.

    python -m honest_agents.node_chain check JOBFILE            # validate, print the plan (Mac or node)
    python -m honest_agents.node_chain run JOBFILE [--poll 5]   # node side; start it with `honest_agents.node chain`

Why. Iteration 13 lost ~$9 because a GRPO run finished with no watcher armed and the node idled 5.5 h. Every node job
therefore lives in one job file; the runner starts each step when its dependencies are done and its GPUs are free, records
what happens in `STATE_DIR/chain_state.json`, and when the chain ends (all done, a failure, a STOP request) runs the final
archive-to-staging command and writes `STATE_DIR/ALL_DONE`. The Mac side (`honest_agents.node_watch`) reads those two
files over ssh, pulls the results and stops the instance.

Job file (YAML or JSON), paths relative to the repo root on the node:

    name: it14
    state_dir: outputs/chain/it14      # chain_state.json, ALL_DONE, STOP, logs/<step>.log
    sequential: true                   # a step without `after` waits for the previous step; false: starts at once
    instance_id: 12345678              # Vast instance (the watcher stops it); balance_floor: 8 (USD)
    steps:
      - name: smoke
        command: scripts/run_slot.sh configs/smoke.yaml
        gpus: [0, 1]                   # steps whose GPU sets overlap never run together
        done: outputs/smoke/DONE       # file whose existence means done; checked again after exit 0
        after: []                      # explicit [] = no dependency; omitted = previous step if `sequential`
        timeout_min: 90                # SIGINT, then SIGTERM, then SIGKILL of the step's process group
        on_failure: stop               # stop: start nothing new; continue: independent steps still run
        env: {SLOT: "0"}
        progress: {pattern: '^\\[(\\d+)/712\\]', total: 712}   # last match in the log tail, for the watcher
    final:
      command: uv run python -m honest_agents.archive --source /root/restraint --dest /root/staging ...
    watch:                             # used by the watcher only
      pull_paths: [/root/staging]
      archives: [{iteration: 14, name: it14-session, run: [outputs/x/y], no_weights: true}]

Design notes. Steps run as `bash -lc COMMAND` in their own session, so a timeout or abort signals the whole process group
(PIDs and groups are tracked; nothing is found with `pgrep -f`, whose pattern also matches its own command line). A stopped step
gets SIGINT first so the evaluation's sandbox teardown runs; leaked `ep*` containers created after the step started are then
removed by name (deferred until no other step is running, so a parallel step's sandboxes are not touched). Every step's
environment carries `HA_CHAIN_STEP=<step>.<nonce>` (and `HA_CHAIN=<chain>`); when the step ends, whatever still carries its
tag is killed (SIGTERM, then SIGKILL), found through /proc rather than by command-line pattern: this reaches a vLLM server the
step started with `setsid` (scripts/_common.sh start_vllm) and did not stop, which would otherwise hold its GPU and port
and fail the next step on that GPU. A step whose done marker already exists is not run again (resume after a crash: start the
same chain file again); delete stale markers of an earlier attempt first. A restarted runner refuses to start while
processes of an earlier run of the same chain are alive (status `refused`), so a step is never run twice at once. The final
step always runs, so a failed or stopped chain still archives what exists.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

STATE_FILE = "chain_state.json"
ALL_DONE_FILE = "ALL_DONE"
STOP_FILE = "STOP"
LOCK_FILE = "chain.lock"
FAILURE_POLICIES = ("stop", "continue")
SANDBOX_PREFIX = "ep"
PROGRESS_TAIL_BYTES = 65536
HEARTBEAT_SECONDS = 20.0
STEP_TAG = "HA_CHAIN_STEP"  # in every step's environment: <step>.<nonce>; found again in /proc when the step ends
CHAIN_TAG = "HA_CHAIN"  # the chain's name; a restarted runner looks for it before starting anything
PROC = Path("/proc")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_STEP_KEYS = {"name", "command", "gpus", "done", "after", "timeout_min", "on_failure", "env", "progress", "stall_min"}
_TOP_KEYS = {"name", "state_dir", "sequential", "instance_id", "balance_floor", "steps", "final", "watch"}
_FINAL_KEYS = {"command", "timeout_min", "env"}


class ChainError(ValueError):
    """The job file is invalid."""


# ── job file ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Progress:
    pattern: str
    total: int

    def last(self, text: str) -> int | None:
        found = re.findall(self.pattern, text, flags=re.MULTILINE)
        if not found:
            return None
        last = found[-1]
        last = last[0] if isinstance(last, tuple) else last
        digits = re.search(r"\d+", str(last))
        return int(digits.group()) if digits else None


@dataclass(frozen=True)
class Step:
    name: str
    command: str
    done: str
    gpus: tuple[int, ...] = ()
    after: tuple[str, ...] = ()
    timeout_min: float | None = None
    on_failure: str = "stop"
    env: Mapping[str, str] = field(default_factory=dict)
    progress: Progress | None = None
    stall_min: float = 20.0


@dataclass(frozen=True)
class Final:
    command: str
    timeout_min: float | None = 60.0
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chain:
    name: str
    state_dir: str
    steps: tuple[Step, ...]
    final: Final | None = None
    instance_id: int | None = None
    balance_floor: float | None = None
    watch: Mapping[str, Any] = field(default_factory=dict)

    def step(self, name: str) -> Step:
        return next(s for s in self.steps if s.name == name)


def _unknown(where: str, got: Mapping[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(got) - allowed)
    if extra:
        raise ChainError(f"{where}: unknown key(s) {extra}; allowed {sorted(allowed)}")


def _str_map(where: str, raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ChainError(f"{where}: must be a mapping")
    return {str(k): str(v) for k, v in raw.items()}


def parse_chain(raw: Mapping[str, Any]) -> Chain:
    """Validate a decoded job file: unique names, known dependencies, no cycles, sane GPUs and policies."""
    _unknown("job file", raw, _TOP_KEYS)
    name = str(raw.get("name", ""))
    if not _NAME.match(name):
        raise ChainError(f"name {name!r}: letters, digits, '.', '_' and '-' only")
    state_dir = str(raw.get("state_dir") or f"outputs/chain/{name}")
    sequential = bool(raw.get("sequential", True))
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ChainError("steps: give a non-empty list")
    steps: list[Step] = []
    for i, rs in enumerate(raw_steps):
        if not isinstance(rs, Mapping):
            raise ChainError(f"steps[{i}]: must be a mapping")
        _unknown(f"steps[{i}]", rs, _STEP_KEYS)
        sname = str(rs.get("name", ""))
        if not _NAME.match(sname):
            raise ChainError(f"steps[{i}].name {sname!r}: letters, digits, '.', '_' and '-' only")
        if not rs.get("command"):
            raise ChainError(f"step {sname}: command is required")
        if not rs.get("done"):
            raise ChainError(f"step {sname}: done (a marker file path) is required")
        policy = str(rs.get("on_failure", "stop"))
        if policy not in FAILURE_POLICIES:
            raise ChainError(f"step {sname}: on_failure must be one of {FAILURE_POLICIES}")
        gpus = rs.get("gpus") or []
        if not isinstance(gpus, list) or any(not isinstance(g, int) or isinstance(g, bool) or g < 0 for g in gpus):
            raise ChainError(f"step {sname}: gpus must be a list of non-negative integers")
        if "after" in rs and rs["after"] is not None:
            after = tuple(str(a) for a in rs["after"])
        else:
            after = (steps[-1].name,) if sequential and steps else ()
        timeout = rs.get("timeout_min")
        if timeout is not None and float(timeout) <= 0:
            raise ChainError(f"step {sname}: timeout_min must be positive")
        prog = rs.get("progress")
        progress = None
        if prog:
            if not isinstance(prog, Mapping) or "pattern" not in prog or "total" not in prog:
                raise ChainError(f"step {sname}: progress needs pattern and total")
            try:
                re.compile(str(prog["pattern"]))
            except re.error as e:
                raise ChainError(f"step {sname}: progress pattern: {e}") from None
            progress = Progress(str(prog["pattern"]), int(prog["total"]))
        steps.append(Step(sname, str(rs["command"]), str(rs["done"]), tuple(sorted(set(gpus))), after,
                          float(timeout) if timeout is not None else None, policy, _str_map(f"step {sname}.env", rs.get("env")),
                          progress, float(rs.get("stall_min", 20.0))))  # fmt: skip
    names = [s.name for s in steps]
    if len(set(names)) != len(names):
        raise ChainError(f"duplicate step names: {sorted({n for n in names if names.count(n) > 1})}")
    for s in steps:
        for a in s.after:
            if a not in names:
                raise ChainError(f"step {s.name}: after {a!r} is not a step")
            if a == s.name:
                raise ChainError(f"step {s.name}: depends on itself")
    _check_acyclic(steps)
    final = None
    if raw.get("final"):
        rf = raw["final"]
        if not isinstance(rf, Mapping) or not rf.get("command"):
            raise ChainError("final: needs a command")
        _unknown("final", rf, _FINAL_KEYS)
        final = Final(str(rf["command"]), float(rf["timeout_min"]) if rf.get("timeout_min") is not None else 60.0,
                      _str_map("final.env", rf.get("env")))  # fmt: skip
    watch = raw.get("watch") or {}
    if not isinstance(watch, Mapping):
        raise ChainError("watch: must be a mapping")
    iid = raw.get("instance_id")
    floor = raw.get("balance_floor")
    return Chain(name, state_dir, tuple(steps), final, int(iid) if iid is not None else None,
                 float(floor) if floor is not None else None, dict(watch))  # fmt: skip


def _check_acyclic(steps: Sequence[Step]) -> None:
    after = {s.name: s.after for s in steps}
    state: dict[str, int] = {}

    def visit(n: str, trail: tuple[str, ...]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise ChainError("dependency cycle: " + " -> ".join([*trail, n]))
        state[n] = 1
        for a in after[n]:
            visit(a, (*trail, n))
        state[n] = 2

    for s in steps:
        visit(s.name, ())


def load_chain(path: Path) -> Chain:
    text = Path(path).expanduser().read_text()
    try:
        if str(path).endswith(".json"):
            raw = json.loads(text)
        else:
            import yaml  # lazy: JSON job files need nothing but the standard library

            raw = yaml.safe_load(text)
    except Exception as e:  # noqa: BLE001  # the parser's message can quote the file
        raise ChainError(f"{path}: cannot parse ({type(e).__name__})") from None
    if not isinstance(raw, Mapping):
        raise ChainError(f"{path}: top level must be a mapping")
    return parse_chain(raw)


# ── sandbox cleanup (pure builders + parser) ─────────────────────────────────

_CREATED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([+-]\d{4})")


def sandbox_list_argv(prefix: str = SANDBOX_PREFIX) -> list[str]:
    """`docker ps -a` for containers whose name contains `prefix` (the filter is a substring regex; the exact
    name prefix is applied in `leaked_sandboxes`)."""
    return ["docker", "ps", "-a", "--no-trunc", "--filter", f"name={prefix}",
            "--format", "{{.ID}}\t{{.Names}}\t{{.CreatedAt}}"]  # fmt: skip


def parse_created(text: str) -> float | None:
    """`2026-10-13 10:00:00 +0000 UTC` (docker's CreatedAt) -> epoch seconds."""
    m = _CREATED.match(text.strip())
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S %z").timestamp()


def leaked_sandboxes(listing: str, since: float, prefix: str = SANDBOX_PREFIX, slack: float = 2.0) -> list[str]:
    """Container ids from `sandbox_list_argv` output whose name starts with `prefix` and that were created at or after
    `since` (minus `slack`: docker prints whole seconds)."""
    ids = []
    for line in listing.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        cid, cname, created = parts[0], parts[1].lstrip("/"), parse_created(parts[2])
        if cname.startswith(prefix) and created is not None and created >= since - slack:
            ids.append(cid)
    return ids


def sandbox_rm_argv(ids: Sequence[str]) -> list[str]:
    return ["docker", "rm", "-f", *ids]


def cleanup_sandboxes(since: float, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[str]:
    """Remove `ep*` containers created after `since`. Returns the ids removed; docker missing or failing removes none."""
    try:
        listed = run(sandbox_list_argv(), capture_output=True, text=True, timeout=60, check=False,
                     stdin=subprocess.DEVNULL)  # fmt: skip
        if listed.returncode != 0:
            return []
        ids = leaked_sandboxes(listed.stdout, since)
        if ids:
            run(sandbox_rm_argv(ids), capture_output=True, text=True, timeout=300, check=False,
                stdin=subprocess.DEVNULL)  # fmt: skip
        return ids
    except (OSError, subprocess.SubprocessError):
        return []


# ── stragglers: processes that left the step's process group (`setsid ... &`) ─


def tagged_pids(var: str, value: str, proc_root: Path = PROC) -> list[int]:
    """PIDs whose environment holds exactly `var=value` (Linux /proc; elsewhere none). A step's children inherit the tag
    even through `setsid`, so a vLLM server the step started and did not stop is found; the runner itself never is."""
    needle = f"{var}={value}".encode()
    me = os.getpid()
    out: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return out
    for p in entries:
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            env = (p / "environ").read_bytes()  # empty for a zombie, unreadable for another user's process
        except OSError:
            continue
        if needle in env.split(b"\0"):
            out.append(int(p.name))
    return sorted(out)


def kill_tagged(var: str, value: str, wait: float = 30.0, proc_root: Path = PROC) -> list[int]:
    """SIGTERM every tagged process, wait up to `wait` seconds for them to go, SIGKILL the rest. Returns the pids found."""
    found = tagged_pids(var, value, proc_root)
    for sig, limit in ((signal.SIGTERM, wait), (signal.SIGKILL, 5.0)):
        alive = tagged_pids(var, value, proc_root)
        if not alive:
            break
        for pid in alive:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline and tagged_pids(var, value, proc_root):
            time.sleep(0.2)
    return found


def sweep_step(token: str) -> list[int]:
    return kill_tagged(STEP_TAG, token)


def find_chain_processes(name: str) -> list[int]:
    return tagged_pids(CHAIN_TAG, name)


# ── runner ───────────────────────────────────────────────────────────────────

PENDING, RUNNING, DONE, FAILED, TIMEOUT, SKIPPED = "pending", "running", "done", "failed", "timeout", "skipped"
FINISHED_BAD = (FAILED, TIMEOUT)


def now() -> float:
    return time.time()


def _iso(t: float | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else None


@dataclass
class _Live:
    proc: subprocess.Popen[bytes]
    started: float
    log_path: Path
    token: str = ""  # the step's HA_CHAIN_STEP value
    killed: str | None = None  # "timeout" | "abort"


class Runner:
    """Executes a `Chain`. `cleanup` (docker), `sweep` and `find` (/proc), `clock` and `sleep` are injectable for tests."""

    def __init__(self, chain: Chain, cwd: Path, poll: float = 5.0, grace: tuple[float, float] = (60.0, 20.0),
                 cleanup: Callable[[float], list[str]] = cleanup_sandboxes, clock: Callable[[], float] = now,
                 sleep: Callable[[float], None] = time.sleep, sweep: Callable[[str], list[int]] = sweep_step,
                 find: Callable[[str], list[int]] = find_chain_processes) -> None:  # fmt: skip
        self.chain, self.cwd, self.poll, self.grace = chain, Path(cwd), poll, grace
        self.cleanup, self.clock, self.sleep, self.sweep, self.find = cleanup, clock, sleep, sweep, find
        self.state_dir = self.cwd / chain.state_dir
        self.status: dict[str, dict[str, Any]] = {
            s.name: {"status": PENDING, "start": None, "end": None, "exit_code": None, "log": str(self._log(s.name)),
                     "gpus": list(s.gpus), "note": "", "progress": None, "total": s.progress.total if s.progress else None,
                     "log_mtime": None}
            for s in chain.steps
        }  # fmt: skip
        self.final_status: dict[str, Any] = {"status": PENDING, "start": None, "end": None, "exit_code": None}
        self.live: dict[str, _Live] = {}
        self.chain_status = "running"
        self.started = self.clock()
        self.abort = False
        self.stop_reason = ""
        self.cleanup_since: float | None = None
        self.cleaned: list[str] = []
        self._last_beat = 0.0

    # paths
    def _log(self, name: str) -> Path:
        return self.state_dir / "logs" / f"{name}.log"

    def _marker(self, step: Step) -> Path:
        return self.cwd / step.done

    # state file
    def write_state(self) -> None:
        state = {
            "chain": self.chain.name, "status": self.chain_status, "pid": os.getpid(), "started": _iso(self.started),
            "updated": _iso(self.clock()), "updated_epoch": self.clock(), "stop_reason": self.stop_reason,
            "steps": self.status, "final": self.final_status, "cleaned_sandboxes": len(self.cleaned),
            "all_done": (self.state_dir / ALL_DONE_FILE).exists(),
        }  # fmt: skip
        tmp = self.state_dir / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, self.state_dir / STATE_FILE)  # atomic: the watcher never reads half a file
        self._last_beat = self.clock()

    # scheduling
    def stop_requested(self) -> bool:
        return self.abort or (self.state_dir / STOP_FILE).exists()

    def _blocked_by(self, step: Step) -> str | None:
        for a in step.after:
            st = self.status[a]["status"]
            if st in (FAILED, TIMEOUT, SKIPPED):
                return a
        return None

    def ready(self) -> list[Step]:
        """Pending steps whose dependencies are done and whose GPUs are free, in file order."""
        busy = {g for n in self.live for g in self.chain.step(n).gpus}
        out = []
        for s in self.chain.steps:
            if self.status[s.name]["status"] != PENDING:
                continue
            if any(self.status[a]["status"] != DONE for a in s.after):
                continue
            if busy & set(s.gpus):
                continue
            out.append(s)
            busy |= set(s.gpus)
        return out

    def _spawn(self, step: Step) -> tuple[subprocess.Popen[bytes], str]:
        self._log(step.name).parent.mkdir(parents=True, exist_ok=True)
        token = f"{step.name}.{uuid.uuid4().hex[:12]}"
        env = {**os.environ, **step.env, CHAIN_TAG: self.chain.name, STEP_TAG: token}
        with self._log(step.name).open("ab") as log:
            log.write(f"== {_iso(self.clock())} start: {step.command}\n".encode())
            log.flush()
            proc = subprocess.Popen(["bash", "-lc", step.command], cwd=self.cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        return proc, token

    def _start(self, step: Step) -> None:
        st = self.status[step.name]
        if self._marker(step).exists():
            st.update(
                status=DONE, start=self.clock(), end=self.clock(), exit_code=0, note="done marker existed; not run"
            )
            return
        proc, token = self._spawn(step)
        self.live[step.name] = _Live(proc, self.clock(), self._log(step.name), token)
        st.update(status=RUNNING, start=self.clock(), pid=proc.pid)

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _sweep(self, lv: _Live) -> list[int]:
        """After the step's leader exited: SIGTERM, then SIGKILL, to what is left of its process group (a `&` child),
        then the processes that carry the step's tag but left the group (`setsid`): a vLLM server left running would
        hold its GPU and port, and the next step on that GPU would fail."""
        if self._group_alive(lv.proc.pid):
            self._signal_group(lv.proc, signal.SIGTERM)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and self._group_alive(lv.proc.pid):
                time.sleep(0.2)
            self._signal_group(lv.proc, signal.SIGKILL)
        return self.sweep(lv.token)

    def _signal_group(self, proc: subprocess.Popen[bytes], sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)  # the step was started in its own session: pid == pgid
        except (ProcessLookupError, PermissionError):
            pass

    def terminate(self, proc: subprocess.Popen[bytes]) -> None:
        """SIGINT (so an evaluation's teardown runs), then SIGTERM, then SIGKILL, each after its grace period."""
        for sig, wait in ((signal.SIGINT, self.grace[0]), (signal.SIGTERM, self.grace[1]), (signal.SIGKILL, 10.0)):
            self._signal_group(proc, sig)
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    self._signal_group(proc, signal.SIGKILL)  # stragglers of the group
                    return
                time.sleep(0.05)
        proc.poll()

    def _tail(self, path: Path) -> str:
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - PROGRESS_TAIL_BYTES))
                return f.read().decode(errors="replace")
        except OSError:
            return ""

    def refresh_live(self) -> None:
        for name, lv in self.live.items():
            st, step = self.status[name], self.chain.step(name)
            try:
                st["log_mtime"] = lv.log_path.stat().st_mtime
            except OSError:
                pass
            if step.progress:
                p = step.progress.last(self._tail(lv.log_path))
                if p is not None:
                    st["progress"] = p

    def reap(self) -> None:
        for name in list(self.live):
            lv, step = self.live[name], self.chain.step(name)
            rc = lv.proc.poll()
            if rc is None and step.timeout_min and self.clock() - lv.started > step.timeout_min * 60 and not lv.killed:
                lv.killed = "timeout"
                self.terminate(lv.proc)
                rc = lv.proc.poll()
            if rc is None and self.abort and not lv.killed:
                lv.killed = "abort"
                self.terminate(lv.proc)
                rc = lv.proc.poll()
            if rc is None:
                continue
            del self.live[name]
            stragglers = self._sweep(lv)
            st = self.status[name]
            st.update(end=self.clock(), exit_code=rc)
            if lv.killed == "timeout":
                st.update(status=TIMEOUT, note=f"killed after {step.timeout_min} min")
            elif lv.killed == "abort":
                st.update(status=FAILED, note="aborted")
            elif rc != 0:
                st.update(status=FAILED, note=f"exit {rc}")
            elif not self._marker(step).exists():
                st.update(status=FAILED, note=f"exit 0 but done marker {step.done} missing")
            else:
                st.update(status=DONE, note="")
            if stragglers:
                st["stragglers"] = len(stragglers)
                st["note"] = (
                    f"{st['note']}; " if st["note"] else ""
                ) + f"{len(stragglers)} leftover process(es) killed (a server the step did not stop?)"
            if st["status"] != DONE:
                self.cleanup_since = min(self.cleanup_since or lv.started, lv.started)
                if step.on_failure == "stop" and not self.stop_reason:
                    self.stop_reason = f"step {name} {st['status']}"

    def _maybe_cleanup(self) -> None:
        if self.cleanup_since is not None and not self.live:  # a parallel step's sandboxes must survive
            self.cleaned += self.cleanup(self.cleanup_since)
            self.cleanup_since = None

    def _skip_unreachable(self) -> None:
        stopping = self.stop_requested() or bool(self.stop_reason)
        for s in self.chain.steps:
            st = self.status[s.name]
            if st["status"] != PENDING:
                continue
            blocker = self._blocked_by(s)
            if blocker:
                st.update(status=SKIPPED, note=f"dependency {blocker} did not finish")
            elif stopping:
                st.update(status=SKIPPED, note=self.stop_reason or "stop requested")

    def run_final(self) -> None:
        fin = self.chain.final
        if fin is None:
            self.final_status["status"] = SKIPPED
            return
        log = self._log("final")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.final_status.update(status=RUNNING, start=self.clock())
        self.write_state()
        with log.open("ab") as f:
            f.write(f"== {_iso(self.clock())} start: {fin.command}\n".encode())
            f.flush()
            env = {**os.environ, **fin.env, CHAIN_TAG: self.chain.name}
            proc = subprocess.Popen(["bash", "-lc", fin.command], cwd=self.cwd, env=env, stdout=f,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
            deadline = self.clock() + (fin.timeout_min or 1e9) * 60
            while proc.poll() is None:
                if self.clock() > deadline:
                    self.terminate(proc)
                    break
                self.sleep(min(self.poll, 1.0))
                if self.clock() - self._last_beat > HEARTBEAT_SECONDS:
                    self.write_state()
        rc = proc.poll()
        self.final_status.update(status=DONE if rc == 0 else FAILED, end=self.clock(), exit_code=rc)

    def verdict(self) -> str:
        if self.abort:
            return "aborted"
        statuses = [s["status"] for s in self.status.values()]
        if any(x in FINISHED_BAD for x in statuses):
            return "failed"
        if any(x == SKIPPED for x in statuses):
            return "stopped"
        return "ok"

    def write_all_done(self) -> None:
        payload = {"status": self.chain_status, "final_exit": self.final_status["exit_code"], "time": _iso(self.clock()),
                   "chain": self.chain.name}  # fmt: skip
        tmp = self.state_dir / (ALL_DONE_FILE + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.state_dir / ALL_DONE_FILE)

    def run(self) -> str:
        """Run the chain to the end; returns the verdict (ok | failed | stopped | aborted), or `refused` when processes
        of an earlier run of this chain are still alive (nothing is started, no ALL_DONE is written)."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ALL_DONE_FILE).unlink(missing_ok=True)
        left = self.find(self.chain.name)
        if left:
            self.chain_status = "refused"
            self.stop_reason = (f"processes of an earlier run of chain {self.chain.name} are still alive (pids "
                                f"{left[:8]}): wait for them or stop them (kill -TERM PID), then start the chain again")  # fmt: skip
            self.write_state()
            return self.chain_status
        self.write_state()
        while True:
            self.reap()
            self._maybe_cleanup()
            if self.stop_requested() and not self.stop_reason:
                self.stop_reason = "abort signal" if self.abort else "STOP file"
            if not (self.stop_reason or self.stop_requested()):
                for step in self.ready():
                    self._start(step)
            self._skip_unreachable()
            self.refresh_live()
            pending = any(s["status"] == PENDING for s in self.status.values())
            if not self.live and not pending:
                break
            if not self.live and pending and not self.ready() and not self.stop_reason:
                # nothing runs and nothing can start: only an unsatisfiable GPU/dependency state; fail loudly
                for s in self.status.values():
                    if s["status"] == PENDING:
                        s.update(status=SKIPPED, note="unreachable")
                break
            self.write_state()  # every pass: also the heartbeat the watcher uses to tell a dead runner from a quiet one
            self.sleep(self.poll)
        self._maybe_cleanup()
        self.chain_status = self.verdict()
        self.run_final()
        self.write_state()
        self.write_all_done()
        self.write_state()  # all_done: true
        return self.chain_status


def _acquire_lock(state_dir: Path) -> Any:
    state_dir.mkdir(parents=True, exist_ok=True)
    f = (state_dir / LOCK_FILE).open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another runner holds {state_dir / LOCK_FILE}; one chain per state_dir") from None
    return f


_PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_]*(?::[^<>\n]*)?>")


def placeholders(chain: Chain) -> list[str]:
    """`<UPPER_CASE>` tokens left in commands, done markers or the final command (a template not yet filled in)."""
    texts = {s.name: f"{s.command} {s.done}" for s in chain.steps}
    if chain.final:
        texts["final"] = chain.final.command
    return [f"{n}: {m}" for n, t in texts.items() for m in _PLACEHOLDER.findall(t)]


def plan_lines(chain: Chain) -> list[str]:
    lines = [f"chain {chain.name}: {len(chain.steps)} steps, state in {chain.state_dir}"]
    for s in chain.steps:
        lines.append(f"  {s.name}: gpus {list(s.gpus) or '-'} after {list(s.after) or '-'} done {s.done} "
                     f"timeout {s.timeout_min or '-'} min, on failure {s.on_failure}")  # fmt: skip
    lines.append("  final: " + (textwrap.shorten(chain.final.command, 100) if chain.final else "none"))
    left = placeholders(chain)
    if left:
        lines.append(f"  WARNING: {len(left)} placeholder(s) left: " + "; ".join(left[:6]))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.node_chain", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate a job file and print the plan")
    c.add_argument("jobfile", type=Path)
    r = sub.add_parser("run", help="run the chain (node side)")
    r.add_argument("jobfile", type=Path)
    r.add_argument("--poll", type=float, default=5.0, help="seconds between scheduler passes")
    r.add_argument("--allow-placeholders", action="store_true", help="run although <PLACEHOLDER> tokens are left")
    r.add_argument("--cwd", type=Path, default=Path.cwd(), help="repo root (default: the current directory)")
    a = ap.parse_args(argv)
    try:
        chain = load_chain(a.jobfile)
    except ChainError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.cmd == "check":
        print("\n".join(plan_lines(chain)))
        return 0
    left = placeholders(chain)
    if left and not a.allow_placeholders:
        print("error: placeholders left in the job file: " + "; ".join(left), file=sys.stderr)
        return 2
    cwd = a.cwd.expanduser().resolve()
    lock = _acquire_lock(cwd / chain.state_dir)  # noqa: F841  # held for the process lifetime
    runner = Runner(chain, cwd, poll=a.poll)

    def on_signal(signum: int, _frame: object) -> None:
        runner.abort = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    verdict = runner.run()
    print(f"chain {chain.name}: {verdict}", flush=True)
    return 0 if verdict == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
