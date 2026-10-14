"""honest_agents.node_watch: event lines, ssh blips, stalls, balance floor, ALL_DONE -> pull -> stop (never destroy).
The ssh/vastai/rsync layer is replaced by `Effects` fakes; no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from honest_agents import node as nodemod
from honest_agents.node import Node
from honest_agents.node_chain import parse_chain
from honest_agents.node_watch import (
    Effects,
    WatchConfig,
    Watcher,
    archive_argv,
    instance_stopped,
    parse_credit,
    parse_snapshot,
    pull_commands,
    read_command,
    stop_accepted,
    stop_request_command,
    vast_balance_argv,
    vast_show_argv,
    vast_stop_argv,
    watch,
)

NODE = Node("1.2.3.4", 2222, "~/.ssh/k")
CHAIN = parse_chain(
    {
        "name": "t",
        "state_dir": "outputs/chain/t",
        "steps": [
            {"name": "a", "command": "x", "done": "m/a", "stall_min": 10, "progress": {"pattern": "x", "total": 100}},
            {"name": "b", "command": "y", "done": "m/b"},
        ],
        "watch": {
            "pull_paths": ["/root/staging"],
            "archives": [{"iteration": 14, "name": "s", "eval": ["runs/eval/e"], "no_weights": True, "note": "n"}],
        },
    }
)


def snap(steps: dict[str, dict[str, Any]], status: str = "running", now: float = 1000.0, updated: float | None = None,
         done: dict[str, Any] | None = None) -> str:  # fmt: skip
    state = {"status": status, "updated_epoch": now if updated is None else updated, "steps": steps}
    return json.dumps(state) + f"\n@@@ {int(now)}\n" + (json.dumps(done) if done else "") + "\n"


def S(status: str, **kw: Any) -> dict[str, Any]:
    return {"status": status, "start": 100.0, "end": None, "exit_code": None, "note": "", **kw}


class Fake:
    def __init__(self, outputs: list[str | None], credits: list[float | None] | None = None) -> None:
        self.outputs, self.credits = list(outputs), list(credits or [])
        self.t = 1000.0
        self.log: list[str] = []
        self.pull_failures: list[str] = []
        self.stop_ok = True

    def effects(self) -> Effects:
        def fetch() -> str | None:
            return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]

        def balance() -> float | None:
            self.log.append("balance")
            return self.credits.pop(0) if self.credits else 100.0

        def request_stop() -> bool:
            self.log.append("request_stop")
            return True

        def pull() -> list[str]:
            self.log.append("pull")
            return self.pull_failures

        def stop() -> bool:
            self.log.append("stop")
            return self.stop_ok

        return Effects(fetch, request_stop, balance, pull, stop, clock=lambda: self.t, sleep=lambda d: None)


def cfg(**kw: Any) -> WatchConfig:
    return WatchConfig(CHAIN, NODE, kw.pop("instance_id", 77), **kw)


def poll(w: Watcher, fake: Fake, n: int = 1, dt: float = 60.0) -> list[str]:
    ev: list[str] = []
    for _ in range(n):
        ev += w.poll_once()
        fake.t += dt
    return ev


# ── parsing and commands ─────────────────────────────────────────────────────


def test_parse_snapshot_variants() -> None:
    s = parse_snapshot(snap({"a": S("running")}, done={"status": "ok"}))
    assert s.state and s.all_done == {"status": "ok"} and s.remote_now == 1000.0
    empty = parse_snapshot("\n@@@ 5\n\n")
    assert empty.state is None and empty.all_done is None and empty.remote_now == 5.0
    torn = parse_snapshot('{"status": "run\n@@@ 5\n\n')
    assert torn.state is None


def test_parse_credit() -> None:
    assert parse_credit('{"credit": 12.5, "id": 1}') == 12.5
    assert parse_credit('[{"credit": 3}]') == 3.0
    # the real `vastai show user --raw` carries both: credit is the spendable amount, balance is 0 on this account
    assert parse_credit('{"balance": 0, "credit": 102.68}') == 102.68
    assert parse_credit('{"balance": 4.25}') is None  # never a fallback: it would stop a healthy chain
    assert parse_credit("not json") is None and parse_credit('{"credit": "x"}') is None


def test_stop_output_and_instance_state_parsers() -> None:
    # `vastai stop instance ID --raw` exits 0 either way and prints the CLI's return value as its last line
    assert stop_accepted("stopping instance 77.\ntrue\n")
    assert not stop_accepted("Instance 77 not found\nfalse\n") and not stop_accepted("")
    assert instance_stopped('{"id": 77, "intended_status": "stopped", "actual_status": "running"}') is True
    assert instance_stopped('{"id": 77, "intended_status": "running", "actual_status": "running"}') is False
    assert instance_stopped('{"id": 77, "actual_status": "exited"}') is True
    assert instance_stopped('{"instances": null}') is None and instance_stopped("junk") is None


def test_command_builders() -> None:
    assert vast_stop_argv(77) == ["vastai", "stop", "instance", "77", "--raw"]
    assert vast_show_argv(77) == ["vastai", "show", "instance", "77", "--raw"]
    assert vast_balance_argv() == ["vastai", "show", "user", "--raw"]
    assert "destroy" not in " ".join(vast_stop_argv(77))
    cmd = read_command(NODE, "outputs/chain/t")
    assert "/root/restraint/outputs/chain/t/chain_state.json" in cmd and "ALL_DONE" in cmd
    assert stop_request_command(NODE, "outputs/chain/t").endswith("/root/restraint/outputs/chain/t/STOP")


def test_pull_commands_cover_state_dir_paths_and_archives() -> None:
    cmds = pull_commands(CHAIN, NODE, Path("/stg"))
    assert cmds[0][0] == "rsync" and cmds[0][-2].endswith(":/root/restraint/outputs/chain/t")
    assert "--timeout=300" in cmds[0] and "--timeout=300" in cmds[1]  # a dead connection fails instead of hanging
    assert cmds[1][-2].endswith(":/root/staging") and cmds[1][-1] == "/stg/root/"
    arch = cmds[2]
    assert arch[:8] == ["uv", "run", "--extra", "train", "python", "-m", "honest_agents.archive", "--iteration"]
    assert ["--source", "root@1.2.3.4:/root/restraint", "--port", "2222"] == arch[
        arch.index("--source") : arch.index("--source") + 4
    ]
    assert "--eval" in arch and "--no-weights" in arch and "--note" in arch
    with pytest.raises(ValueError):
        archive_argv({"name": "x"}, NODE)


# ── events ───────────────────────────────────────────────────────────────────


def test_start_progress_done_events_once_each() -> None:
    f = Fake([snap({"a": S("running", progress=10, total=100), "b": S("pending")}),
              snap({"a": S("running", progress=30, total=100), "b": S("pending")}),
              snap({"a": S("running", progress=31, total=100), "b": S("pending")}),
              snap({"a": S("done", end=700.0, exit_code=0, progress=100, total=100), "b": S("running")})])  # fmt: skip
    w = Watcher(cfg(), f.effects())
    ev = poll(w, f, 4)
    assert ev == ["START a", "PROGRESS a 25% (30/100)", "DONE a 10m", "START b"]
    assert poll(w, f, 3) == []  # steady state: silence


def test_failed_timeout_and_skipped() -> None:
    f = Fake([snap({"a": S("failed", exit_code=7, note="exit 7"), "b": S("skipped", note="dependency a did not finish")},
                   status="failed")])  # fmt: skip
    w = Watcher(cfg(), f.effects())
    ev = poll(w, f, 3)
    assert ev == ["FAILED a exit 7: exit 7", "SKIPPED b: dependency a did not finish",
                  "CHAIN failed (final archive runs next, then ALL_DONE)"]  # fmt: skip


def test_refused_chain_is_announced_once_without_heartbeat_alarm() -> None:
    reason = "processes of an earlier run of chain t are still alive (pids [4242]): wait for them"
    stale = json.dumps({"status": "refused", "stop_reason": reason, "updated_epoch": 100.0, "steps": {}})
    f = Fake([stale + "\n@@@ 5000\n\n"])
    ev = poll(Watcher(cfg(), f.effects()), f, 3)
    assert ev == [f"CHAIN refused: {reason}"]


def test_ssh_blips_alert_only_after_the_limit_then_recover() -> None:
    ok = snap({"a": S("running")})
    f = Fake([ok, None, None, ok, None, None, None, ok])
    w = Watcher(cfg(), f.effects())
    ev = poll(w, f, 8)
    assert ev == ["START a", "NODE ssh failing 3 polls in a row", "NODE ssh back"]


def test_stall_alert_and_recovery() -> None:
    # step a: stall_min 10 -> alert when the log has been quiet more than 600 s on the node's clock
    s_ok = snap({"a": S("running", log_mtime=900.0)}, now=1000.0)
    s_quiet = snap({"a": S("running", log_mtime=100.0)}, now=1000.0)
    f = Fake([s_ok, s_quiet, s_quiet, s_ok])
    w = Watcher(cfg(), f.effects())
    ev = poll(w, f, 4)
    assert ev == ["START a", "STALL a: log quiet 15m"]
    assert "a" not in w.stalled


def test_runner_heartbeat_loss() -> None:
    f = Fake([snap({"a": S("running")}, now=2000.0, updated=1000.0)])
    ev = poll(Watcher(cfg(), f.effects()), f, 3)
    assert len([e for e in ev if e.startswith("RUNNER")]) == 1
    fresh = Fake([snap({"a": S("running")}, now=2000.0, updated=1990.0)])
    assert not [e for e in poll(Watcher(cfg(), fresh.effects()), fresh) if e.startswith("RUNNER")]
    # the verdict is written before the final archive runs: a runner that dies then must still be noticed
    final = Fake([snap({"a": S("done", end=150.0)}, status="failed", now=2000.0, updated=1000.0)])
    ev = poll(Watcher(cfg(), final.effects()), final, 2)
    assert len([e for e in ev if e.startswith("RUNNER")]) == 1 and any(e.startswith("CHAIN failed") for e in ev)


def test_missing_state_alerts_after_timeout() -> None:
    f = Fake(["\n@@@ 5\n\n"])
    w = Watcher(cfg(no_state_timeout=300), f.effects())
    ev = poll(w, f, 8)
    assert len(ev) == 1 and ev[0].startswith("NODE no chain state")


# ── ALL_DONE: pull then stop ─────────────────────────────────────────────────


def test_all_done_pulls_then_stops_the_instance() -> None:
    f = Fake([snap({"a": S("done", end=200.0, exit_code=0)}, status="ok", done={"status": "ok", "final_exit": 0})])
    out: list[str] = []
    rc = watch(cfg(), f.effects(), out.append)
    assert rc == 0 and f.log == ["pull", "stop"]
    assert [l.split(" ", 1)[1] for l in out] == [
        "DONE a 1m",
        "ALL_DONE ok (final exit 0)",
        "PULL ok",
        "INSTANCE 77 stopped (not destroyed)",
    ]


def test_failed_pull_still_stops_and_says_so() -> None:
    f = Fake([snap({}, status="failed", done={"status": "failed", "final_exit": 1})])
    f.pull_failures = ["rsync -az x"]
    out: list[str] = []
    rc = watch(cfg(), f.effects(), out.append)
    assert rc == 1 and f.log == ["pull", "stop"]
    assert any("PULL FAILED 1" in l for l in out)


def test_stop_failure_is_reported_with_the_manual_command() -> None:
    f = Fake([snap({}, status="ok", done={"status": "ok"})])
    f.stop_ok = False
    out: list[str] = []
    watch(cfg(), f.effects(), out.append)
    assert "STOP FAILED: run `vastai stop instance 77`" in out[-1]


def test_no_stop_flag_and_missing_instance_id() -> None:
    done = snap({}, status="ok", done={"status": "ok"})
    f = Fake([done])
    watch(cfg(stop_instance=False), f.effects(), lambda l: None)
    assert f.log == ["pull"]
    g = Fake([done])
    out: list[str] = []
    watch(cfg(instance_id=None), g.effects(), out.append)  # type: ignore[arg-type]
    assert g.log == ["pull"] and "no instance id" in out[-1]


# ── balance floor ────────────────────────────────────────────────────────────


def test_balance_below_floor_requests_stop_once_and_chain_end_follows() -> None:
    running = snap({"a": S("running")})
    f = Fake([running], credits=[20.0, 7.0, 6.0, 5.0])
    w = Watcher(cfg(balance_floor=8.0, balance_every=2), f.effects())
    ev = poll(w, f, 8)
    assert [e for e in ev if e.startswith("BALANCE")] == [
        "BALANCE $7.00 below floor $8.00: stopping the chain after the current step"
    ]
    assert f.log.count("request_stop") == 1
    assert "stop" not in f.log  # the instance stops only after ALL_DONE
    f.outputs = [snap({"a": S("done", end=300.0)}, status="stopped", done={"status": "stopped", "final_exit": 0})]
    ev = poll(w, f)
    assert f.log[-2:] == ["pull", "stop"] and any(e.startswith("INSTANCE 77 stopped") for e in ev)


def test_hard_floor_pulls_and_stops_immediately() -> None:
    f = Fake([snap({"a": S("running")})], credits=[2.0])
    w = Watcher(cfg(balance_floor=8.0, hard_floor=3.0), f.effects())
    ev = poll(w, f)
    assert f.log == ["balance", "pull", "stop"] and w.finished
    assert any("hard floor" in e for e in ev)


def test_unreadable_balance_alerts_after_three_failures() -> None:
    f = Fake([snap({"a": S("running")})], credits=[None, None, None, None])
    w = Watcher(cfg(balance_floor=8.0, balance_every=1), f.effects())
    ev = [e for e in poll(w, f, 4) if e.startswith("BALANCE")]
    assert ev == ["BALANCE unreadable 3 times (vastai show user)"]


# ── CLI plumbing ─────────────────────────────────────────────────────────────


def test_watch_dry_run_prints_pulls_and_stop(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    job = tmp_path / "c.json"
    job.write_text(json.dumps({"name": "t", "instance_id": 5, "steps": [{"name": "a", "command": "x", "done": "m/a"}]}))
    assert nodemod.main(["watch", "--host", "1.2.3.4", "--port", "2222", "--dry-run", str(job)]) == 0
    out = capsys.readouterr().out
    assert "rsync" in out and "vastai stop instance 5 --raw" in out and "destroy" not in out


def test_chain_command_refuses_placeholders_and_passes_no_secret_in_argv(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    unfilled = tmp_path / "unfilled.json"
    unfilled.write_text(json.dumps({"name": "t", "steps": [{"name": "a", "command": "<CMD: x>", "done": "m/a"}]}))
    assert nodemod.main(["chain", "--host", "h", "--no-secrets", str(unfilled)]) == 2
    template = nodemod.REPO / "configs" / "chains" / "it14_session.yaml"
    assert nodemod.main(["chain", "--host", "h", "--no-secrets", "--dry-run", str(template)]) == 0
    out = capsys.readouterr().out
    assert "honest_agents.node_chain run configs/chains/it14_session.yaml" in out and "secrets over stdin: none" in out
    assert (
        "rm -f outputs/chain/it14/ALL_DONE &&" in out
    )  # a stale ALL_DONE must not trigger the watcher's pull-and-stop
