"""The glue scripts of the iteration-14 chain (docs/iteration-14-runbook.md section 3a), on a Mac with no GPU: the `--`
split and the environment of scripts/serve_and_run.sh, its server teardown on every exit path (a stub server in a
temporary copy of scripts/), the epoch mapping of scripts/merge_epochs.sh and the fallback of scripts/select_epoch.sh."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from honest_agents.sft import epochs

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

STUB_SERVE = r"""#!/usr/bin/env bash
# stand-in for scripts/serve_vllm.sh: /health on VLLM_PORT, plus a worker child like vLLM's
echo "$$" > "$STUB_DIR/server.pid"
echo "$*" > "$STUB_DIR/server.args"
sleep 300 &
echo "$!" > "$STUB_DIR/worker.pid"
exec python3 -c '
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()' "$VLLM_PORT"
"""
# macOS has no setsid(1); this one starts the command in a new session like util-linux's
SETSID_SHIM = '#!/usr/bin/env bash\nexec python3 -c \'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])\' "$@"\n'


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(cond, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


@pytest.fixture
def sandbox(tmp_path: Path):
    """A copy of scripts/_common.sh and serve_and_run.sh next to a stub serve_vllm.sh, with a setsid shim on PATH."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("_common.sh", "serve_and_run.sh"):
        shutil.copy(SCRIPTS / name, scripts / name)
    (scripts / "serve_vllm.sh").write_text(STUB_SERVE)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "setsid").write_text(SETSID_SHIM)
    for p in (*scripts.glob("*.sh"), bin_dir / "setsid"):
        p.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_DIR": str(tmp_path),
        "VLLM_PORT": str(_free_port()),
        "VLLM_GPUS": "0",
        "WAIT_CHILD_SEC": "3",
    }
    env.pop("HA_CHAIN_STEP", None)
    return tmp_path, env


def _serve(tmp: Path, env, *args: str) -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", str(tmp / "scripts" / "serve_and_run.sh"), "configs/x.yaml", *args],
        env=env, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip


def _server_stopped(tmp: Path) -> bool:
    pids = [int((tmp / f).read_text()) for f in ("server.pid", "worker.pid")]
    return _wait_for(lambda: not any(_alive(p) for p in pids))


def test_serve_and_run_splits_groups_exports_the_environment_and_stops_the_server(sandbox):
    tmp, env = sandbox
    probe = "echo \"$MODEL $BASE_URL $*\" >> out.txt"
    p = _serve(tmp, env, "model.name=outputs/m", "--", "bash", "-c", probe, "_", "one", "--", "bash", "-c", probe, "_", "two", "--flag")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    port = env["VLLM_PORT"]
    assert (tmp / "out.txt").read_text().splitlines() == [
        f"policy http://127.0.0.1:{port}/v1 one",
        f"policy http://127.0.0.1:{port}/v1 two --flag",
    ]
    args = (tmp / "server.args").read_text()
    assert "configs/x.yaml model.name=outputs/m --served-model-name policy" in args
    assert _server_stopped(tmp)
    assert (tmp / "outputs" / "logs" / f"serve_port{port}.log").is_file()


def test_serve_and_run_stops_at_the_first_failure_with_its_status_and_stops_the_server(sandbox):
    tmp, env = sandbox
    p = _serve(tmp, env, "--", "bash", "-c", "exit 7", "--", "touch", "never.txt")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 7, out
    assert not (tmp / "never.txt").exists()
    assert _server_stopped(tmp)


def test_serve_and_run_names_its_log_after_the_chain_step(sandbox):
    tmp, env = sandbox
    env["HA_CHAIN_STEP"] = "base_select.abc123"
    out, _ = (p := _serve(tmp, env, "--", "true")).communicate(timeout=60)
    assert p.returncode == 0, out
    assert (tmp / "outputs" / "logs" / "serve_base_select.log").is_file()


@pytest.mark.parametrize(("sig", "status"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_serve_and_run_stops_the_server_on_sigint_and_sigterm(sandbox, sig, status):
    tmp, env = sandbox
    p = _serve(tmp, env, "--", "bash", "-c", "echo $$ > cmd.pid; sleep 120")
    assert _wait_for(lambda: (tmp / "cmd.pid").exists()), "the command never started"
    cmd_pid = int((tmp / "cmd.pid").read_text())
    p.send_signal(sig)  # to the script alone, as a runner that signals the leader does
    out, _ = p.communicate(timeout=60)
    assert p.returncode == status, out
    assert _server_stopped(tmp)
    assert _wait_for(lambda: not _alive(cmd_pid)), "the running command was left behind"


def test_serve_and_run_stops_the_server_on_a_signal_while_it_waits_for_health(sandbox):
    tmp, env = sandbox
    env["VLLM_PORT"] = str(_free_port())
    (tmp / "scripts" / "serve_vllm.sh").write_text(  # never becomes healthy
        '#!/usr/bin/env bash\necho $$ > "$STUB_DIR/server.pid"\nsleep 300 &\necho $! > "$STUB_DIR/worker.pid"\nwait\n'
    )
    p = _serve(tmp, env, "--", "true")
    assert _wait_for(lambda: (tmp / "worker.pid").exists())
    time.sleep(1)
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=60)
    assert p.returncode == 143
    assert _server_stopped(tmp)


def test_serve_and_run_dry_run_prints_the_plan_and_starts_nothing(sandbox):
    tmp, env = sandbox
    env["DRY_RUN"] = "1"
    p = _serve(tmp, env, "model.name=outputs/m", "--", "scripts/eval_heldout.sh", "a", "b", "--n", "2", "--", "echo", "hi")
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, out
    assert "command 1: scripts/eval_heldout.sh a b --n 2" in out and "command 2: echo hi" in out
    assert "model.name=outputs/m --served-model-name policy" in out and "MODEL=policy" in out
    assert not (tmp / "server.pid").exists()


@pytest.mark.parametrize("args", [[], ["model.name=x"], ["--"]])
def test_serve_and_run_refuses_a_missing_command(sandbox, args):
    tmp, env = sandbox
    p = _serve(tmp, env, *args)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 2 and not (tmp / "server.pid").exists(), out


# ── epochs: merge_epochs.sh and select_epoch.sh ──────────────────────────────


def _run_dir(root: Path, steps_epochs: dict[int, float]) -> Path:
    run = root / "run"
    for step, epoch in steps_epochs.items():
        ck = run / f"checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "adapter_config.json").write_text("{}")
        (ck / "trainer_state.json").write_text(json.dumps({"epoch": epoch, "global_step": step}))
    return run


def test_epochs_come_from_the_trainer_state_not_the_listing_order(tmp_path):
    # lexical order would put checkpoint-100 first; the epoch of a checkpoint is what its trainer state says
    run = _run_dir(tmp_path, {100: 3.0, 9: 1.0, 55: 2.0})
    assert [(k, p.name) for k, p in epochs.checkpoint_epochs(run, 3)] == [
        (1, "checkpoint-9"), (2, "checkpoint-55"), (3, "checkpoint-100"),
    ]  # fmt: skip
    # a run stopped after epoch 2, whose checkpoints are 1 and 2, is not "epochs 1..3"
    short = _run_dir(tmp_path / "s", {10: 1.0, 20: 2.0})
    with pytest.raises(SystemExit, match="expected epoch checkpoints 1..3"):
        epochs.checkpoint_epochs(short, 3)
    assert [k for k, _ in epochs.checkpoint_epochs(short)] == [1, 2]


@pytest.mark.parametrize(
    "states",
    [{10: 1.0, 20: 1.0}, {10: 1.5}, {10: 0.4}],
    ids=["duplicate epoch", "mid-epoch save", "no full epoch"],
)
def test_epochs_refuse_an_ambiguous_run(tmp_path, states):
    with pytest.raises(SystemExit):
        epochs.checkpoint_epochs(_run_dir(tmp_path, states))


def test_epochs_refuse_a_checkpoint_without_state_or_adapter(tmp_path):
    run = _run_dir(tmp_path, {10: 1.0})
    (run / "checkpoint-10" / "trainer_state.json").unlink()
    with pytest.raises(SystemExit, match="trainer_state"):
        epochs.checkpoint_epochs(run)
    (run / "checkpoint-10" / "adapter_config.json").unlink()
    with pytest.raises(SystemExit, match="adapter"):
        epochs.checkpoint_epochs(run)


def _selection(path: Path, selected: str | None) -> Path:
    path.write_text(json.dumps({"selected": selected}))
    return path


def test_decide_takes_the_selected_epoch_or_the_pre_registered_fallback(tmp_path):
    out = tmp_path / "o"
    sel = _selection(tmp_path / "selection.json", "it14_sft_v2_select_e2")
    assert epochs.decide(sel, out, fallback=3, status=0, prefix="sft_v2") == (2, "selected")
    assert (out / "selected_epoch.txt").read_text() == "2 selected\n"
    assert os.readlink(out / "sft_v2_selected_merged") == "sft_v2_e2_merged"
    sel = _selection(tmp_path / "selection.json", None)  # nobody passes the gate: underpowered, so epoch 3
    assert epochs.decide(sel, out, fallback=3, status=1, prefix="sft_v2") == (3, "fallback")
    assert (out / "selected_epoch.txt").read_text() == "3 fallback\n"
    assert os.readlink(out / "sft_v2_selected_merged") == "sft_v2_e3_merged"  # the old link was replaced


def test_decide_does_not_mistake_a_refused_comparison_for_a_failed_gate(tmp_path):
    out = tmp_path / "o"
    with pytest.raises(SystemExit, match="refused the comparison"):  # status 1 and no selection.json: SystemExit(message)
        epochs.decide(tmp_path / "selection.json", out, fallback=3, status=1, prefix="sft_v2")
    with pytest.raises(SystemExit, match="no fallback"):
        epochs.decide(_selection(tmp_path / "s.json", None), out, fallback=3, status=2, prefix="sft_v2")
    with pytest.raises(SystemExit, match="contradicts"):
        epochs.decide(_selection(tmp_path / "s.json", "x_e1"), out, fallback=3, status=1, prefix="sft_v2")
    assert not (out / "selected_epoch.txt").exists()


FAKE_UV = r"""#!/usr/bin/env bash
# `env -u VIRTUAL_ENV uv run --frozen --extra train python ARGS` -> the test interpreter; merge and select are recorded
while [[ $# -gt 0 && "$1" != python ]]; do shift; done
shift
case "$*" in
  *honest_agents.sft.merge*) echo "$*" >> "$FAKE_LOG" ;;
  *honest_agents.eval.select_checkpoint*)
    echo "$*" >> "$FAKE_LOG"
    out=""; while [[ $# -gt 0 ]]; do [[ "$1" == --out ]] && out="$2"; shift; done
    if [[ "$FAKE_SELECTED" != refuse ]]; then mkdir -p "$(dirname "$out")"; echo "{\"selected\": $FAKE_SELECTED}" > "$out"; fi
    exit "$FAKE_RC" ;;
  *) exec "$REAL_PY" "$@" ;;
esac
"""


@pytest.fixture
def fake_uv(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text(FAKE_UV)
    (bin_dir / "uv").chmod(0o755)
    env = {
        **os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REAL_PY": sys.executable, "FAKE_LOG": str(tmp_path / "calls.log"),
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")]),
    }  # fmt: skip
    return tmp_path, env


def _sh(script: str, env, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPTS / script), *args], env=env, cwd=REPO, capture_output=True, text=True)


def test_merge_epochs_merges_each_checkpoint_into_its_epoch_directory(fake_uv):
    tmp, env = fake_uv
    run = _run_dir(tmp, {200: 3.0, 20: 1.0, 110: 2.0})
    done = _sh("merge_epochs.sh", env, str(run), str(tmp / "out" / "sft_v2"), "Qwen/Qwen3.5-4B")
    assert done.returncode == 0, done.stderr
    calls = (tmp / "calls.log").read_text().splitlines()
    assert len(calls) == 3
    for k, step, call in zip((1, 2, 3), (20, 110, 200), calls, strict=True):
        assert f"--adapter {run}/checkpoint-{step} " in call and call.endswith(f"--out {tmp}/out/sft_v2_e{k}_merged")
        assert "--base Qwen/Qwen3.5-4B" in call


def test_merge_epochs_refuses_other_than_three_checkpoints_and_dry_run_merges_nothing(fake_uv):
    tmp, env = fake_uv
    run = _run_dir(tmp, {20: 1.0, 110: 2.0})
    done = _sh("merge_epochs.sh", env, str(run), str(tmp / "o" / "p"), "B")
    assert done.returncode != 0 and "expected epoch checkpoints 1..3" in done.stderr
    assert not (tmp / "calls.log").exists()
    run3 = _run_dir(tmp / "r3", {20: 1.0, 110: 2.0, 200: 3.0})
    done = _sh("merge_epochs.sh", {**env, "DRY_RUN": "1"}, str(run3), str(tmp / "o" / "p"), "B")
    assert done.returncode == 0 and done.stdout.count("->") == 3 and not (tmp / "calls.log").exists()


def _select(tmp: Path, env, *, rc: int, selected: str, fallback: str = "3"):
    env = {**env, "FAKE_RC": str(rc), "FAKE_SELECTED": selected}
    return _sh("select_epoch.sh", env, "runs/eval/base", str(tmp / "out"), fallback, "runs/eval/e1", "runs/eval/e2", "runs/eval/e3")


def test_select_epoch_passes_the_runbook_arguments_and_links_the_selected_epoch(fake_uv):
    tmp, env = fake_uv
    done = _select(tmp, env, rc=0, selected='"e2"')
    assert done.returncode == 0, done.stderr
    call = (tmp / "calls.log").read_text()
    assert "-m honest_agents.eval.select_checkpoint --base runs/eval/base runs/eval/e1 runs/eval/e2 runs/eval/e3" in call
    assert f"--require-split select --out {tmp}/out/selection.json" in call
    assert (tmp / "out" / "selected_epoch.txt").read_text() == "2 selected\n"
    assert os.readlink(tmp / "out" / "sft_v2_selected_merged") == "sft_v2_e2_merged"


def test_select_epoch_falls_back_to_epoch_three_when_no_epoch_passes(fake_uv):
    tmp, env = fake_uv
    done = _select(tmp, env, rc=1, selected="null")
    assert done.returncode == 0, done.stderr
    assert (tmp / "out" / "selected_epoch.txt").read_text() == "3 fallback\n"
    assert os.readlink(tmp / "out" / "sft_v2_selected_merged") == "sft_v2_e3_merged"
    assert json.loads((tmp / "out" / "selection.json").read_text())["selected"] is None


def test_select_epoch_fails_when_the_comparison_is_refused_even_with_a_stale_selection(fake_uv):
    tmp, env = fake_uv
    (tmp / "out").mkdir()
    _selection(tmp / "out" / "selection.json", "stale_e1")
    done = _select(tmp, env, rc=1, selected="refuse")
    assert done.returncode != 0 and "refused the comparison" in done.stderr
    assert not (tmp / "out" / "selected_epoch.txt").exists()
    assert _select(tmp, env, rc=2, selected="null").returncode != 0
