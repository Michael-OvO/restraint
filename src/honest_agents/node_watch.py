"""Mac-side watcher for a node chain (`honest_agents.node_chain`): one terse line per event, then pull and stop.

    python -m honest_agents.node watch --host HOST --port PORT --instance-id ID configs/chains/it14_session.yaml

Meant to run under a Claude Code Monitor: each stdout line is one notification, so it prints only on events
(START, PROGRESS at 25 % steps, DONE, FAILED/TIMEOUT, SKIPPED, STALL, RUNNER (no heartbeat), NODE (ssh down/back),
BALANCE, ALL_DONE, PULL, INSTANCE). One ssh call per poll reads `chain_state.json` and `ALL_DONE`; an unreachable node
alerts only after N consecutive failures (ssh blips are normal on rented boxes).

When `ALL_DONE` exists the watcher pulls (the state directory, the job file's `watch.pull_paths`, then one
`honest_agents.archive` call per `watch.archives` entry) and then **stops** the Vast instance
(`vastai stop instance ID --raw`, then `vastai show instance ID --raw` until `intended_status` is `stopped`; with `--raw`
the CLI exits 0 even when the stop failed and prints `false`, so the exit code alone proves nothing). It never destroys: a
stop keeps the disk, so even a failed pull loses nothing.
Balance floor: below `balance_floor` (`vastai show user --raw`, field `credit`; the `balance` field is something else and
is 0 on this account) it writes the chain's `STOP` file, so no new step starts; the running steps finish, the final archive
step runs, `ALL_DONE` appears and the normal path follows. `hard_floor` (optional) skips the wait: pull what exists and stop
at once.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import node as nodemod
from honest_agents.node import Node
from honest_agents.node_chain import (
    ALL_DONE_FILE,
    DONE,
    FAILED,
    RUNNING,
    SKIPPED,
    STATE_FILE,
    STOP_FILE,
    TIMEOUT,
    Chain,
)

SEP = "@@@"
ARCHIVE_LIST_KEYS = ("run", "eval", "file", "log")
ARCHIVE_FLAGS = (("no_weights", "--no-weights"), ("keep_bank", "--keep-bank"))
VASTAI = "vastai"
RSYNC_STALL_SECONDS = 300  # rsync --timeout: a pull over a dead ssh connection fails instead of hanging for an hour


@dataclass(frozen=True)
class WatchConfig:
    chain: Chain
    node: Node
    instance_id: int | None
    balance_floor: float | None = None
    hard_floor: float | None = None
    poll: float = 60.0
    ssh_fail_limit: int = 3
    balance_every: int = 5  # polls between balance checks
    beat_stale: float = 300.0  # seconds without a heartbeat from the runner while it claims to be running
    no_state_timeout: float = 900.0  # seconds without any state file after the watcher started
    stop_instance: bool = True


@dataclass
class Snapshot:
    state: dict[str, Any] | None
    all_done: dict[str, Any] | None
    remote_now: float | None


# ── remote reads (pure builders / parsers) ───────────────────────────────────


def read_command(node: Node, state_dir: str) -> str:
    """One command, one ssh round trip: the state file, the remote clock, the ALL_DONE file. No secrets involved."""
    d = shlex.quote(f"{node.dest.rstrip('/')}/{state_dir.strip('/')}")
    return f"cat {d}/{STATE_FILE} 2>/dev/null; echo; echo {SEP} $(date +%s); cat {d}/{ALL_DONE_FILE} 2>/dev/null; echo"


def stop_request_command(node: Node, state_dir: str) -> str:
    d = shlex.quote(f"{node.dest.rstrip('/')}/{state_dir.strip('/')}")
    return f"mkdir -p {d} && touch {d}/{STOP_FILE}"


def parse_snapshot(out: str) -> Snapshot:
    head, _, tail = out.partition(SEP)
    clock, _, done_text = tail.partition("\n")

    def js(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        try:
            val = json.loads(text)
        except json.JSONDecodeError:
            return None  # a torn read; the next poll sees the whole file
        return val if isinstance(val, dict) else None

    try:
        remote_now: float | None = float(clock.strip())
    except ValueError:
        remote_now = None
    return Snapshot(js(head), js(done_text), remote_now)


def parse_credit(raw: str) -> float | None:
    """`vastai show user --raw` -> the `credit` field (USD). Only that field: the output also carries `balance`, which is
    0 on this account while `credit` is the spendable amount, so a fallback to it would stop a healthy chain."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None
    v = data.get("credit")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def stop_accepted(stdout: str) -> bool:
    """`vastai stop instance ID --raw` prints the CLI's return value as JSON on its last line: `true` when the API
    accepted the stop, `false` (after the API's message) when it did not; the exit code is 0 either way."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return bool(lines) and lines[-1] == "true"


def instance_stopped(raw: str) -> bool | None:
    """`vastai show instance ID --raw`: True when the instance is stopped or told to stop, False when it is meant to run,
    None when the answer cannot be read (not JSON, `{"instances": null}` for an unknown id, fields missing)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("id"):
        return None
    intended = str(data.get("intended_status") or "").lower()
    actual = str(data.get("actual_status") or "").lower()
    if intended == "stopped" or actual in ("stopped", "exited"):
        return True
    return False if (intended or actual) else None


def archive_argv(entry: dict[str, Any], node: Node, staging_dest: Path | None = None) -> list[str]:
    """`honest_agents.archive` call for one `watch.archives` entry, reading from the node."""
    if "iteration" not in entry or "name" not in entry:
        raise ValueError("archives entry needs iteration and name")
    argv = ["uv", "run", "--extra", "train", "python", "-m", "honest_agents.archive", "--iteration", str(entry["iteration"]),
            "--name", str(entry["name"]), "--source", f"{node.target}:{node.dest}", "--port", str(node.port),
            "--key", node.key_path]  # fmt: skip
    for key in ARCHIVE_LIST_KEYS:
        for v in entry.get(key, []) or []:
            argv += [f"--{key}", str(v)]
    for key, flag in ARCHIVE_FLAGS:
        if entry.get(key):
            argv.append(flag)
    if entry.get("note"):
        argv += ["--note", str(entry["note"])]
    if staging_dest:
        argv += ["--dest", str(staging_dest)]
    return argv


def pull_commands(chain: Chain, node: Node, staging: Path = nodemod.DEFAULT_STAGING) -> list[list[str]]:
    """Every local command the final pull runs, in order: node pulls (state directory first, with the step logs),
    then the archive calls."""
    cmds: list[list[str]] = []
    for remote, local_dir in nodemod.pull_plan(node, pull_paths(chain), staging):
        argv = nodemod.pull_argv(node, remote, local_dir)
        cmds.append([argv[0], f"--timeout={RSYNC_STALL_SECONDS}", *argv[1:]])
    for entry in chain.watch.get("archives") or []:
        cmds.append(archive_argv(entry, node))
    return cmds


def pull_paths(chain: Chain) -> list[str]:
    return [chain.state_dir, *(chain.watch.get("pull_paths") or [])]


def vast_stop_argv(instance_id: int, binary: str = VASTAI) -> list[str]:
    return [binary, "stop", "instance", str(instance_id), "--raw"]


def vast_show_argv(instance_id: int, binary: str = VASTAI) -> list[str]:
    return [binary, "show", "instance", str(instance_id), "--raw"]


def vast_balance_argv(binary: str = VASTAI) -> list[str]:
    return [binary, "show", "user", "--raw"]


# ── effects (injectable) ─────────────────────────────────────────────────────


@dataclass
class Effects:
    """Everything the watcher does to the outside world; tests replace these with fakes."""

    fetch: Callable[[], str | None]  # raw read_command output, None when ssh failed
    request_stop: Callable[[], bool]
    balance: Callable[[], float | None]
    pull: Callable[[], list[str]]  # descriptions of failed pulls (empty: all ok)
    stop_instance: Callable[[], bool]
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep


def ssh_options(node: Node) -> list[str]:
    return [*node.ssh_options(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/ha-watch-%C",
            "-o", "ControlPersist=300"]  # fmt: skip


def real_effects(cfg: WatchConfig, staging: Path = nodemod.DEFAULT_STAGING, vastai: str = VASTAI,
                 repo: Path = nodemod.REPO) -> Effects:  # fmt: skip
    node, chain = cfg.node, cfg.chain

    def ssh(command: str, timeout: float = 60) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(["ssh", *ssh_options(node), node.target, command], capture_output=True, text=True,
                                  timeout=timeout, check=False, stdin=subprocess.DEVNULL)  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None

    def fetch() -> str | None:
        r = ssh(read_command(node, chain.state_dir))
        return r.stdout if r is not None and r.returncode == 0 else None

    def request_stop() -> bool:
        r = ssh(stop_request_command(node, chain.state_dir))
        return r is not None and r.returncode == 0

    def balance() -> float | None:
        try:
            r = subprocess.run(vast_balance_argv(vastai), capture_output=True, text=True, timeout=60, check=False,
                               stdin=subprocess.DEVNULL)  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return None
        return parse_credit(r.stdout) if r.returncode == 0 else None

    def pull() -> list[str]:
        failed = []
        for _, local_dir in nodemod.pull_plan(node, pull_paths(chain), staging):
            local_dir.mkdir(parents=True, exist_ok=True)  # the Mac's openrsync creates no missing parent directories
        for argv in pull_commands(chain, node, staging):
            for attempt in range(2):
                try:
                    rc = subprocess.run(argv, cwd=repo, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, check=False, timeout=3600).returncode  # fmt: skip
                except (OSError, subprocess.SubprocessError):
                    rc = 1
                if rc == 0:
                    break
                time.sleep(5)
            else:
                failed.append(" ".join(argv[:6])[:80])
        return failed

    def vast(argv: list[str]) -> str | None:
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    def confirmed_stopped(instance_id: int) -> bool:
        """`show instance` until it says stopped (the API flips `intended_status` at once). Unreadable answers count
        as confirmation only when every poll was unreadable: the stop call itself already said `true`."""
        unreadable = 0
        for i in range(6):
            out = vast(vast_show_argv(instance_id, vastai))
            state = instance_stopped(out) if out is not None else None
            if state:
                return True
            unreadable += state is None
            if i < 5:
                time.sleep(10)
        return unreadable == 6

    def stop_instance() -> bool:
        if cfg.instance_id is None:
            return False
        for attempt in range(3):
            out = vast(vast_stop_argv(cfg.instance_id, vastai))
            if out is not None and stop_accepted(out) and confirmed_stopped(cfg.instance_id):
                return True
            if attempt < 2:
                time.sleep(10)
        return False

    return Effects(fetch, request_stop, balance, pull, stop_instance)


# ── the watcher ──────────────────────────────────────────────────────────────


def _mins(seconds: float) -> str:
    return f"{int(seconds // 60)}m"


@dataclass
class Watcher:
    cfg: WatchConfig
    fx: Effects
    polls: int = 0
    ssh_fails: int = 0
    node_down: bool = False
    started: float = 0.0
    seen: dict[str, str] = field(default_factory=dict)
    quarter: dict[str, int] = field(default_factory=dict)
    stalled: set[str] = field(default_factory=set)
    runner_stale: bool = False
    no_state_said: bool = False
    stop_sent: bool = False
    low_said: bool = False
    balance_fails: int = 0
    finished: bool = False
    verdict: str = ""
    chain_seen: str = ""

    def __post_init__(self) -> None:
        self.started = self.fx.clock()

    def poll_once(self) -> list[str]:
        """One poll: returns the event lines (empty most of the time)."""
        self.polls += 1
        ev: list[str] = []
        out = self.fx.fetch()
        if out is None:
            self.ssh_fails += 1
            if self.ssh_fails == self.cfg.ssh_fail_limit:
                self.node_down = True
                ev.append(f"NODE ssh failing {self.ssh_fails} polls in a row")
        else:
            if self.node_down:
                ev.append("NODE ssh back")
            self.ssh_fails, self.node_down = 0, False
            snap = parse_snapshot(out)
            ev += self._on_snapshot(snap)
        ev += self._balance()
        return ev

    # state transitions
    def _on_snapshot(self, snap: Snapshot) -> list[str]:
        ev: list[str] = []
        now = self.fx.clock()
        st = snap.state
        if st is None:
            if not self.no_state_said and now - self.started > self.cfg.no_state_timeout and snap.all_done is None:
                self.no_state_said = True
                ev.append(f"NODE no chain state after {_mins(now - self.started)}: is the runner started?")
        else:
            self.no_state_said = False
            ev += self._steps(st, snap)
            ev += self._chain(st, snap)
            ev += self._beat(st, snap)
        if snap.all_done is not None:
            self.verdict = str(snap.all_done.get("status", "?"))
            ev.append(f"ALL_DONE {snap.all_done.get('status', '?')} (final exit {snap.all_done.get('final_exit')})")
            ev += self._finalize()
        return ev

    def _steps(self, st: dict[str, Any], snap: Snapshot) -> list[str]:
        ev: list[str] = []
        remote_now = snap.remote_now or self.fx.clock()
        for name, s in (st.get("steps") or {}).items():
            status = s.get("status")
            if status != self.seen.get(name):
                self.seen[name] = status
                dur = f" {_mins(s['end'] - s['start'])}" if s.get("end") and s.get("start") else ""
                if status == RUNNING:
                    ev.append(f"START {name}")
                elif status == DONE:
                    ev.append(f"DONE {name}{dur}")
                elif status in (FAILED, TIMEOUT):
                    ev.append(f"{status.upper()} {name} exit {s.get('exit_code')}: {s.get('note', '')}")
                elif status == SKIPPED:
                    ev.append(f"SKIPPED {name}: {s.get('note', '')}")
            if status == RUNNING:
                total, prog = s.get("total"), s.get("progress")
                if total and prog is not None:
                    q = min(4, prog * 4 // total)
                    if q > self.quarter.get(name, 0):
                        self.quarter[name] = q
                        ev.append(f"PROGRESS {name} {q * 25}% ({prog}/{total})")
                mtime = s.get("log_mtime")
                limit = self._stall_limit(name)
                quiet = remote_now - mtime if mtime else 0.0
                if mtime and quiet > limit * 60 and name not in self.stalled:
                    self.stalled.add(name)
                    ev.append(f"STALL {name}: log quiet {_mins(quiet)}")
                elif name in self.stalled and quiet <= limit * 60:
                    self.stalled.discard(name)
        return ev

    def _stall_limit(self, name: str) -> float:
        try:
            return self.cfg.chain.step(name).stall_min
        except StopIteration:
            return 20.0

    def _chain(self, st: dict[str, Any], snap: Snapshot) -> list[str]:
        """The chain's own status once it leaves `running`: the verdict while the final archive still runs (ALL_DONE
        follows), or `refused` (a restarted runner found an earlier run's processes alive and started nothing)."""
        status = str(st.get("status") or "")
        if status == self.chain_seen or status in ("", "running") or snap.all_done is not None:
            self.chain_seen = status or self.chain_seen
            return []
        self.chain_seen = status
        reason = str(st.get("stop_reason") or "")
        tail = "" if status == "refused" else " (final archive runs next, then ALL_DONE)"
        return [f"CHAIN {status}" + (f": {reason}" if reason else "") + tail]

    def _beat(self, st: dict[str, Any], snap: Snapshot) -> list[str]:
        """The runner rewrites the state file every scheduler pass (and every 20 s while the final archive runs); a
        stale one before ALL_DONE means the runner died (or the node rebooted) and nothing will ever write ALL_DONE.
        `refused` is written once by a runner that then exits on purpose."""
        if (
            st.get("all_done")
            or st.get("status") == "refused"
            or snap.remote_now is None
            or st.get("updated_epoch") is None
        ):
            self.runner_stale = False
            return []
        age = snap.remote_now - float(st["updated_epoch"])
        if age > self.cfg.beat_stale and not self.runner_stale:
            self.runner_stale = True
            return [f"RUNNER no heartbeat for {_mins(age)}: process dead? restart the chain (done markers resume it)"]
        if age <= self.cfg.beat_stale:
            self.runner_stale = False
        return []

    def _balance(self) -> list[str]:
        floor = self.cfg.balance_floor
        every = max(1, self.cfg.balance_every)
        if floor is None or self.polls % every != 1 % every or self.finished:
            return []
        credit = self.fx.balance()
        if credit is None:
            self.balance_fails += 1
            return ["BALANCE unreadable 3 times (vastai show user)"] if self.balance_fails == 3 else []
        self.balance_fails = 0
        ev: list[str] = []
        if self.cfg.hard_floor is not None and credit < self.cfg.hard_floor:
            ev.append(f"BALANCE ${credit:.2f} below hard floor ${self.cfg.hard_floor:.2f}: pulling and stopping now")
            return ev + self._finalize()
        if credit < floor:
            if not self.low_said:
                self.low_said = True
                ev.append(f"BALANCE ${credit:.2f} below floor ${floor:.2f}: stopping the chain after the current step")
            if not self.stop_sent:
                self.stop_sent = self.fx.request_stop()
                if not self.stop_sent:
                    ev.append("BALANCE could not write the STOP file (ssh); will retry")
        return ev

    def _finalize(self) -> list[str]:
        if self.finished:
            return []
        self.finished = True
        ev: list[str] = []
        failed = self.fx.pull()
        ev.append("PULL ok" if not failed else f"PULL FAILED {len(failed)}: {'; '.join(failed)[:200]}")
        if not self.cfg.stop_instance:
            ev.append("INSTANCE left running (--no-stop)")
        elif self.cfg.instance_id is None:
            ev.append("INSTANCE not stopped: no instance id given")
        elif self.fx.stop_instance():
            ev.append(f"INSTANCE {self.cfg.instance_id} stopped (not destroyed)")
        else:
            ev.append(f"INSTANCE {self.cfg.instance_id} STOP FAILED: run `vastai stop instance {self.cfg.instance_id}`")
        return ev


def watch(cfg: WatchConfig, fx: Effects, out: Callable[[str], None] | None = None) -> int:
    """Poll until the chain is finished and the instance handled. Returns 0 when the chain verdict was ok (1: failed, stopped, or a balance stop)."""
    w = Watcher(cfg, fx)
    emit = out or (lambda line: print(line, flush=True))
    while not w.finished:
        for line in w.poll_once():
            emit(time.strftime("%H:%M ", time.localtime(fx.clock())) + line)
        if w.finished:
            break
        fx.sleep(cfg.poll)
    return 0 if w.verdict == "ok" else 1


if __name__ == "__main__":
    sys.exit(nodemod.main(["watch", *sys.argv[1:]]))
