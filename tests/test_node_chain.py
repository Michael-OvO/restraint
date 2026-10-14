"""honest_agents.node_chain: job file validation, dependency order, GPU-aware parallelism, done markers, failure policy,
state file, ALL_DONE, sandbox cleanup. Steps are real `bash -lc` processes in a temporary directory; docker is faked."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from honest_agents import node_chain as nc
from honest_agents.node_chain import (
    ALL_DONE_FILE,
    STATE_FILE,
    STOP_FILE,
    Chain,
    ChainError,
    Runner,
    cleanup_sandboxes,
    leaked_sandboxes,
    parse_chain,
    placeholders,
    sandbox_list_argv,
    sandbox_rm_argv,
)


def step(name: str, command: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"name": name, "command": command or f"touch m/{name}", "done": f"m/{name}", **kw}


def chain_of(steps: list[dict[str, Any]], **top: Any) -> Chain:
    return parse_chain({"name": "t", "state_dir": "state", "steps": steps, **top})


def run_chain(tmp_path: Path, chain: Chain, cleanup: Any = None, **kw: Any) -> tuple[Runner, str]:
    (tmp_path / "m").mkdir(exist_ok=True)
    calls: list[float] = []
    r = Runner(chain, tmp_path, poll=0.02, grace=(0.3, 0.3), cleanup=cleanup or (lambda since: calls.append(since) or []),
               **kw)  # fmt: skip
    r.calls = calls  # type: ignore[attr-defined]
    return r, r.run()


def state(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "state" / STATE_FILE).read_text())


# ── job file ─────────────────────────────────────────────────────────────────


def test_sequential_default_and_explicit_after() -> None:
    c = chain_of([step("a"), step("b"), step("c", after=[]), step("d", after=["a", "b"])])
    assert [s.after for s in c.steps] == [(), ("a",), (), ("a", "b")]
    parallel = chain_of([step("a"), step("b")], sequential=False)
    assert parallel.steps[1].after == ()


@pytest.mark.parametrize(
    "steps, match",
    [
        ([step("a", after=["b"]), step("b", after=["a"])], "cycle"),
        ([step("a", after=["zzz"])], "not a step"),
        ([step("a"), step("a")], "duplicate"),
        ([{"name": "a", "command": "true"}], "done"),
        ([step("a", on_failure="explode")], "on_failure"),
        ([step("a", gpus=[-1])], "gpus"),
        ([step("a", timeout_min=0)], "timeout"),
        ([step("a", surprise=1)], "unknown key"),
        ([step("bad name")], "letters"),
    ],
)
def test_invalid_job_files(steps: list[dict[str, Any]], match: str) -> None:
    with pytest.raises(ChainError, match=match):
        chain_of(steps)


def test_load_yaml_and_json(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text("name: x\nsteps:\n  - {name: a, command: 'true', done: m/a, gpus: [1, 0, 1]}\n")
    c = nc.load_chain(tmp_path / "c.yaml")
    assert c.steps[0].gpus == (0, 1) and c.state_dir == "outputs/chain/x"
    (tmp_path / "c.json").write_text(json.dumps({"name": "x", "steps": [step("a")]}))
    assert nc.load_chain(tmp_path / "c.json").steps[0].name == "a"
    (tmp_path / "bad.yaml").write_text("name: [unclosed\n")
    with pytest.raises(ChainError, match="cannot parse"):
        nc.load_chain(tmp_path / "bad.yaml")


def test_template_chain_parses_and_reports_placeholders() -> None:
    c = nc.load_chain(Path(__file__).resolve().parents[1] / "configs" / "chains" / "it14_session.yaml")
    assert c.final is not None and c.balance_floor == 8
    assert placeholders(c), "the template must refuse to start until filled in"
    assert any("<CMD" in p for p in placeholders(c))


def test_progress_last_match() -> None:
    p = nc.Progress(r"^\[(\d+)/712\]", 712)
    assert p.last("noise\n[3/712] a\n[40/712] b\nx") == 40
    assert p.last("nothing") is None


# ── scheduling ───────────────────────────────────────────────────────────────


def test_dependency_order(tmp_path: Path) -> None:
    order = tmp_path / "order.txt"
    steps = [step(n, f"echo {n} >> {order}; touch m/{n}", after=a) for n, a in [("c", ["b"]), ("b", ["a"]), ("a", [])]]
    _, verdict = run_chain(tmp_path, chain_of(steps))
    assert verdict == "ok"
    assert order.read_text().split() == ["a", "b", "c"]


def test_ready_respects_gpus_and_dependencies(tmp_path: Path) -> None:
    c = chain_of(
        [
            step("a", gpus=[0], after=[]),
            step("b", gpus=[0, 1], after=[]),
            step("c", gpus=[1], after=[]),
            step("d", gpus=[], after=["a"]),
        ]
    )
    r = Runner(c, tmp_path)
    assert [s.name for s in r.ready()] == ["a", "c"]  # b overlaps a on GPU 0; d waits for a; c is disjoint from a


def test_parallel_when_gpus_disjoint_serial_when_shared(tmp_path: Path) -> None:
    def timed(name: str) -> str:
        return f"date +%s.%N > m/{name}.start; sleep 0.6; date +%s.%N > m/{name}.end; touch m/{name}"

    steps = [step("a", timed("a"), gpus=[0], after=[]), step("b", timed("b"), gpus=[1], after=[]),
             step("c", timed("c"), gpus=[0], after=[])]  # fmt: skip
    _, verdict = run_chain(tmp_path, chain_of(steps))
    assert verdict == "ok"
    t = {f"{n}.{k}": float((tmp_path / "m" / f"{n}.{k}").read_text()) for n in "abc" for k in ("start", "end")}
    assert t["b.start"] < t["a.end"] and t["a.start"] < t["b.end"]  # a and b overlapped
    assert t["c.start"] >= t["a.end"] - 0.01  # c shares GPU 0 with a
    assert state(tmp_path)["steps"]["c"]["status"] == "done"


# ── done markers ─────────────────────────────────────────────────────────────


def test_exit_zero_without_marker_fails(tmp_path: Path) -> None:
    _, verdict = run_chain(tmp_path, chain_of([step("a", "true")]))
    assert verdict == "failed"
    s = state(tmp_path)["steps"]["a"]
    assert s["status"] == "failed" and "marker" in s["note"] and s["exit_code"] == 0


def test_existing_marker_skips_the_run(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a").write_text("")
    ran = tmp_path / "ran.txt"
    _, verdict = run_chain(tmp_path, chain_of([step("a", f"touch {ran}; touch m/a"), step("b")]))
    assert verdict == "ok" and not ran.exists()
    assert "not run" in state(tmp_path)["steps"]["a"]["note"]


# ── failure policy ───────────────────────────────────────────────────────────


def test_stop_policy_starts_nothing_new_and_skips_dependents(tmp_path: Path) -> None:
    steps = [step("bad", "exit 3", after=[], gpus=[0]), step("other", after=["bad"]), step("indep", after=[], gpus=[1])]
    # `indep` starts together with `bad` (disjoint GPUs, no dependency); `other` is blocked and skipped
    _, verdict = run_chain(tmp_path, chain_of(steps))
    st = state(tmp_path)["steps"]
    assert verdict == "failed"
    assert st["bad"]["status"] == "failed" and st["bad"]["exit_code"] == 3
    assert st["other"]["status"] == "skipped"
    assert state(tmp_path)["stop_reason"] == "step bad failed"


def test_stop_policy_blocks_later_independent_steps(tmp_path: Path) -> None:
    steps = [step("bad", "exit 1", gpus=[0], after=[]), step("later", gpus=[0], after=[])]  # later waits for GPU 0
    run_chain(tmp_path, chain_of(steps))
    st = state(tmp_path)["steps"]
    assert st["later"]["status"] == "skipped" and not (tmp_path / "m" / "later").exists()


def test_continue_policy_runs_independent_steps_only(tmp_path: Path) -> None:
    steps = [step("bad", "exit 1", gpus=[0], after=[], on_failure="continue"), step("dep", after=["bad"]),
             step("later", gpus=[0], after=[])]  # fmt: skip
    _, verdict = run_chain(tmp_path, chain_of(steps))
    st = state(tmp_path)["steps"]
    assert (st["bad"]["status"], st["dep"]["status"], st["later"]["status"]) == ("failed", "skipped", "done")
    assert verdict == "failed"  # a failure still shows in the verdict


# ── state file, ALL_DONE, final ──────────────────────────────────────────────


def test_state_file_and_all_done(tmp_path: Path) -> None:
    final = {"command": f"echo archived > {tmp_path}/final.txt"}
    _, verdict = run_chain(tmp_path, chain_of([step("a"), step("b")], final=final))
    s = state(tmp_path)
    assert verdict == "ok" and s["status"] == "ok" and s["all_done"] is True and s["pid"] == os.getpid()
    for name in ("a", "b"):
        e = s["steps"][name]
        assert e["status"] == "done" and e["exit_code"] == 0 and e["start"] <= e["end"]
        assert Path(e["log"]).is_file() and "start:" in Path(e["log"]).read_text()
    assert s["final"]["status"] == "done"
    assert (tmp_path / "final.txt").read_text().strip() == "archived"
    done = json.loads((tmp_path / "state" / ALL_DONE_FILE).read_text())
    assert done["status"] == "ok" and done["final_exit"] == 0
    assert not list((tmp_path / "state").glob("*.tmp"))


def test_final_runs_after_failure_and_failed_final_is_recorded(tmp_path: Path) -> None:
    _, verdict = run_chain(tmp_path, chain_of([step("bad", "exit 2")], final={"command": "exit 5"}))
    done = json.loads((tmp_path / "state" / ALL_DONE_FILE).read_text())
    assert verdict == "failed" and done["status"] == "failed" and done["final_exit"] == 5


def test_stop_file_prevents_new_steps_but_final_runs(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / STOP_FILE).touch()
    _, verdict = run_chain(tmp_path, chain_of([step("a"), step("b")], final={"command": "true"}))
    s = state(tmp_path)
    assert verdict == "stopped" and s["stop_reason"] == "STOP file"
    assert {e["status"] for e in s["steps"].values()} == {"skipped"}
    assert (tmp_path / "state" / ALL_DONE_FILE).exists()


def test_stop_file_during_a_step_lets_it_finish(tmp_path: Path) -> None:
    steps = [step("a", f"touch {tmp_path}/state/{STOP_FILE}; sleep 0.3; touch m/a"), step("b")]
    _, verdict = run_chain(tmp_path, chain_of(steps))
    st = state(tmp_path)["steps"]
    assert (st["a"]["status"], st["b"]["status"]) == ("done", "skipped") and verdict == "stopped"


def test_progress_is_recorded_for_running_steps(tmp_path: Path) -> None:
    cmd = "echo '[7/20] x'; sleep 0.4; touch m/a"
    c = chain_of([step("a", cmd, progress={"pattern": r"^\[(\d+)/20\]", "total": 20})])
    (tmp_path / "m").mkdir()
    seen: list[Any] = []
    r = Runner(
        c,
        tmp_path,
        poll=0.02,
        cleanup=lambda s: [],
        sleep=lambda d: (seen.append(r.status["a"]["progress"]), time.sleep(d))[1],
    )
    r.run()
    assert 7 in seen


# ── timeout, process groups, cleanup ─────────────────────────────────────────


def test_timeout_kills_the_process_group_and_cleans_up(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    cmd = f"sleep 60 & echo $! > {pidfile}; wait"
    t0 = time.time()
    r, verdict = run_chain(tmp_path, chain_of([step("slow", cmd, timeout_min=0.01)]))
    st = state(tmp_path)["steps"]["slow"]
    assert st["status"] == "timeout" and verdict == "failed" and "killed" in st["note"]
    pid = int(pidfile.read_text())
    for _ in range(50):  # the sleep child died with its group (a zombie of init may need a moment to be reaped)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the step's child process survived the group kill")
    assert len(r.calls) == 1 and t0 - 5 <= r.calls[0] <= time.time()  # type: ignore[attr-defined]


def test_cleanup_waits_for_parallel_steps(tmp_path: Path) -> None:
    steps = [step("bad", "exit 1", gpus=[0], after=[], on_failure="continue"),
             step("long", "sleep 0.8; touch m/long", gpus=[1], after=[])]  # fmt: skip
    events: list[float] = []
    (tmp_path / "m").mkdir()
    r = Runner(chain_of(steps), tmp_path, poll=0.02, cleanup=lambda s: (events.append(time.time()), [])[1])
    r.run()
    long = state(tmp_path)["steps"]["long"]
    assert len(events) == 1 and long["status"] == "done"
    assert events[0] >= long["end"]  # cleanup waited until `long` (whose sandboxes it would have killed) was over


def test_abort_signal_terminates_steps_and_still_archives(tmp_path: Path) -> None:
    c = chain_of([step("a", "sleep 60", after=[])], final={"command": f"touch {tmp_path}/final"})
    (tmp_path / "m").mkdir()
    r = Runner(c, tmp_path, poll=0.02, grace=(0.3, 0.3), cleanup=lambda s: [])
    orig = r.sleep
    r.sleep = lambda d: (setattr(r, "abort", True), orig(d))[1]  # type: ignore[method-assign]
    assert r.run() == "aborted"
    assert (tmp_path / "final").exists()
    assert json.loads((tmp_path / "state" / ALL_DONE_FILE).read_text())["status"] == "aborted"


def test_tagged_pids_match_the_exact_environment_entry(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    entries = {100: b"A=1\0HA_CHAIN_STEP=a.1\0", 101: b"HA_CHAIN_STEP=a.12\0", 102: b"X=HA_CHAIN_STEP=a.1\0",
               os.getpid(): b"HA_CHAIN_STEP=a.1\0"}  # fmt: skip
    for pid, env in entries.items():
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "environ").write_bytes(env)
    (proc / "self").mkdir()
    (proc / "103").mkdir()  # no readable environ (gone, or another user's)
    assert nc.tagged_pids("HA_CHAIN_STEP", "a.1", proc) == [100]  # not the prefix match, not the value match, not us
    assert nc.tagged_pids("HA_CHAIN_STEP", "a.1", tmp_path / "no-proc") == []  # macOS, or a missing /proc: harmless


def test_steps_are_tagged_and_swept_when_they_end(tmp_path: Path) -> None:
    seen = tmp_path / "env.txt"
    swept: list[str] = []
    (tmp_path / "m").mkdir()
    c = chain_of([step("a", f'echo "$HA_CHAIN_STEP $HA_CHAIN" > {seen}; touch m/a')])
    r = Runner(c, tmp_path, poll=0.02, cleanup=lambda s: [], sweep=lambda tok: (swept.append(tok), [11, 12])[1])
    assert r.run() == "ok"
    token, chain_name = seen.read_text().split()
    assert token.startswith("a.") and chain_name == "t" and swept == [token]  # the step's own token, once
    s = state(tmp_path)["steps"]["a"]
    assert s["status"] == "done" and s["stragglers"] == 2 and "leftover process(es) killed" in s["note"]


def test_background_child_in_the_group_is_killed_after_a_normal_exit(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    _, verdict = run_chain(tmp_path, chain_of([step("a", f"sleep 60 & echo $! > {pidfile}; touch m/a")]))
    assert verdict == "ok"
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("a `&` child of the step survived the step's end")


def test_runner_refuses_while_an_earlier_run_is_alive(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / ALL_DONE_FILE).write_text("{}")  # stale, from the earlier run
    r = Runner(chain_of([step("a")], final={"command": "true"}), tmp_path, poll=0.02, cleanup=lambda s: [],
               find=lambda name: [4242] if name == "t" else [])  # fmt: skip
    assert r.run() == "refused"
    s = state(tmp_path)
    assert s["status"] == "refused" and "4242" in s["stop_reason"] and s["steps"]["a"]["status"] == "pending"
    assert not (tmp_path / "state" / ALL_DONE_FILE).exists() and not (tmp_path / "m" / "a").exists()


def test_second_runner_is_refused(tmp_path: Path) -> None:
    held = nc._acquire_lock(tmp_path / "state")
    with pytest.raises(SystemExit, match="another runner"):
        nc._acquire_lock(tmp_path / "state")
    held.close()
    nc._acquire_lock(tmp_path / "state").close()


# ── sandbox cleanup commands ─────────────────────────────────────────────────

LISTING = "\n".join(  # noqa: FLY002
    [
        "aaa\tep0-main-1\t2026-10-13 10:00:30 +0000 UTC",  # after the step started
        "bbb\t/ep3-sidecar-1\t2026-10-13 10:05:00 +0000 UTC",
        "ccc\tep9-old\t2026-10-13 09:00:00 +0000 UTC",  # before
        "ddd\tkeep-ep-db\t2026-10-13 10:30:00 +0000 UTC",  # contains ep but does not start with it
        "eee\tep1\tgarbage",
        "short line",
    ]
)


def test_cleanup_command_construction() -> None:
    argv = sandbox_list_argv()
    assert argv[:4] == ["docker", "ps", "-a", "--no-trunc"] and "name=ep" in argv
    assert sandbox_rm_argv(["a", "b"]) == ["docker", "rm", "-f", "a", "b"]
    assert nc.parse_created("2026-10-13 10:00:00 +0000 UTC") == 1791885600.0
    assert nc.parse_created("2026-10-13 10:00:00 +0200 CEST") == 1791878400.0
    assert nc.parse_created("nope") is None
    since = nc.parse_created("2026-10-13 10:00:00 +0000 UTC")
    assert since is not None
    assert leaked_sandboxes(LISTING, since) == ["aaa", "bbb"]
    assert leaked_sandboxes(LISTING, since + 120) == ["bbb"]


def test_cleanup_runs_docker_rm_with_the_matching_ids() -> None:
    calls: list[list[str]] = []

    def fake(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=LISTING if argv[1] == "ps" else "", stderr="")

    since = nc.parse_created("2026-10-13 10:00:00 +0000 UTC")
    assert since is not None
    assert cleanup_sandboxes(since, run=fake) == ["aaa", "bbb"]
    assert calls[1] == ["docker", "rm", "-f", "aaa", "bbb"]
    assert cleanup_sandboxes(since + 3600, run=fake) == []
    assert len(calls) == 3  # no rm when nothing matches


def test_cleanup_without_docker_is_harmless() -> None:
    def broken(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    assert cleanup_sandboxes(0.0, run=broken) == []
