"""honest_agents.node builds the right rsync/ssh commands and never puts a secret in argv, a script or its output.
No network: subprocess.run is replaced, and the secrets come from temporary files (never the real ~/.netrc)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from honest_agents import node as nd
from honest_agents.node import (
    Node,
    Secrets,
    pull_argv,
    pull_plan,
    read_secrets,
    remote_script,
    run_argv,
    sync_argv,
)

WANDB = "wandb-test-key-0123456789abcdef"
HF = "hf_testtoken_ABCDEF"


@pytest.fixture
def secret_files(tmp_path: Path) -> tuple[Path, Path]:
    netrc = tmp_path / "netrc"
    netrc.write_text(
        f"machine example.com\n  login u\n  password other\nmachine api.wandb.ai\n  login user\n  password {WANDB}\n"
    )
    netrc.chmod(0o600)  # netrc refuses world-readable files holding passwords
    hf = tmp_path / "hf_token"
    hf.write_text(HF + "\n")
    return netrc, hf


# ── sync / pull ──────────────────────────────────────────────────────────────


def test_sync_sends_the_tree_without_data_or_outputs_over_the_vast_key():
    argv = sync_argv(Node("1.2.3.4", 4242), src=Path("/repo"))
    assert argv[:2] == ["rsync", "-az"]
    assert (
        argv[argv.index("-e") + 1] == f"ssh -i {Path('~/.ssh/vast_ed25519').expanduser()} -p 4242 -o IdentitiesOnly=yes"
    )
    assert argv[-2:] == ["/repo/", "root@1.2.3.4:/root/restraint/"]
    excl = {a.split("=", 1)[1] for a in argv if a.startswith("--exclude=")}
    assert {
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "/outputs",
        "/runs",
        "/wandb",
        "/.claude",
        "*.log",
        "/data",
        ".env",
    } <= excl
    assert ".git" not in excl  # the node has no GitHub key: .git travels so git log and archive can name the commit
    assert "--dry-run" not in argv and "--delete" not in argv


def test_sync_with_data_and_dry_run():
    argv = sync_argv(Node("me@h", 22, key="/k k/id", dest="/srv/r/"), src=Path("/repo"), with_data=True, dry_run=True)
    assert "--exclude=/data" not in argv
    assert "--dry-run" in argv and "--itemize-changes" in argv
    assert "'/k k/id'" in argv[argv.index("-e") + 1]  # quoted for rsync's own word splitting
    assert argv[-1] == "me@h:/srv/r/"


def test_the_documented_probe_lists_travel_with_data():
    """The probe lists' documented `--out` is under data/ (sent by --with-data), never under a path sync never sends."""
    import re

    from honest_agents.eval import probe_set as eval_probe_set
    from honest_agents.tasks import probe_set as tasks_probe_set

    always = {e for e in nd.EXCLUDES if e.startswith("/")}
    for module in (eval_probe_set, tasks_probe_set):
        out = re.search(r"--out (\S+)", module.__doc__ or "")
        assert out, module.__name__
        top = "/" + Path(out.group(1)).parts[0]
        assert top == nd.DATA_EXCLUDE and top not in always, (module.__name__, out.group(1))


def test_pull_keeps_the_repo_relative_layout_in_staging(tmp_path):
    plan = pull_plan(Node("h", 7), ["runs/eval/val/", "/root/grpo.log"], staging=tmp_path)
    assert plan == [("/root/restraint/runs/eval/val", tmp_path / "runs/eval"), ("/root/grpo.log", tmp_path / "root")]
    argv = pull_argv(Node("h", 7), *plan[0], dry_run=True)
    assert argv[-2:] == ["root@h:/root/restraint/runs/eval/val", f"{tmp_path}/runs/eval/"]
    assert "-R" not in "".join(a for a in argv if a.startswith("-a"))  # openrsync ignores `/./`; no -R
    for bad in ([], ["../secrets"], ["runs/../../etc"], ["/"]):
        with pytest.raises(ValueError):
            pull_plan(Node("h"), bad)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_pull_layout_with_the_local_rsync(tmp_path):
    """The plan's (source, parent dir) pairs give `staging/runs/eval/val/...` with whatever rsync the Mac has."""
    node_root = tmp_path / "node"
    (node_root / "runs/eval/val").mkdir(parents=True)
    (node_root / "runs/eval/val/summary.json").write_text("{}")
    staging = tmp_path / "staging"
    for remote, local_dir in pull_plan(Node("h", dest=str(node_root)), ["runs/eval/val"], staging):
        local_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-az", remote, f"{local_dir}/"], check=True, capture_output=True)
    assert (staging / "runs/eval/val/summary.json").is_file()


# ── secrets ──────────────────────────────────────────────────────────────────


def test_secrets_are_read_from_the_given_files_and_never_shown(secret_files, tmp_path):
    s = read_secrets(*secret_files)
    assert (s.wandb_api_key, s.hf_token) == (WANDB, HF)
    assert WANDB not in repr(s) and HF not in repr(s)
    assert s.names() == ["WANDB_API_KEY", "HF_TOKEN"]
    assert s.stdin_payload() == f"{WANDB}\n{HF}\n"
    none = read_secrets(tmp_path / "missing", tmp_path / "missing2")
    assert none.names() == [] and none.stdin_payload() == "\n\n"
    with pytest.raises(ValueError):
        Secrets("a\nb").stdin_payload()


def test_a_broken_netrc_is_reported_without_its_contents(tmp_path):
    bad = tmp_path / "netrc"
    bad.write_text(f"machine api.wandb.ai login u password {WANDB} bogus {WANDB}x\n")
    bad.chmod(0o600)
    with pytest.raises(SystemExit) as e:
        read_secrets(bad, tmp_path / "missing")
    assert WANDB not in str(e.value) and e.value.__cause__ is None and e.value.__suppress_context__


# ── run ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", nd.MODES)
def test_remote_script_holds_names_not_values(mode, secret_files):
    script = remote_script("/root/restraint", "scripts/run_grpo.sh configs/x.yaml", mode, "grpo3")
    assert "read -r WANDB_API_KEY" in script and "read -r HF_TOKEN" in script
    assert "cd /root/restraint" in script and "scripts/run_grpo.sh configs/x.yaml" in script
    assert WANDB not in script and HF not in script
    if mode == "tmux":
        assert "tmux -L grpo3 has-session" in script and "/root/grpo3.log" in script
    if mode == "nohup":
        assert "setsid nohup" in script and ">> /root/grpo3.log" in script


def test_remote_script_rejects_unsafe_names_and_modes():
    with pytest.raises(ValueError):
        remote_script("/r", "true", "nohup", "x; rm -rf /")
    with pytest.raises(ValueError):
        remote_script("/r", "true", "screen", "x")


def test_run_main_sends_secrets_only_on_stdin(secret_files, monkeypatch, capsys):
    netrc, hf = secret_files
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(nd.subprocess, "run", fake_run)
    rc = nd.main(["run", "--host", "1.2.3.4", "--port", "4242", "--netrc", str(netrc), "--hf-token-file", str(hf),
                  "--name", "grpo3", "--mode", "tmux", "--", "scripts/run_grpo.sh", "configs/x.yaml"])  # fmt: skip
    assert rc == 0 and len(calls) == 1
    argv, kw = calls[0]
    assert argv[:2] == ["ssh", "-i"] and "root@1.2.3.4" in argv
    assert not any(WANDB in a or HF in a for a in argv)
    assert kw["input"] == f"{WANDB}\n{HF}\n"
    out = capsys.readouterr()
    assert WANDB not in out.out + out.err and HF not in out.out + out.err
    assert "WANDB_API_KEY, HF_TOKEN" in out.out


def test_run_dry_run_sends_nothing(secret_files, monkeypatch, capsys):
    monkeypatch.setattr(nd.subprocess, "run", lambda *a, **k: pytest.fail("dry run must not execute"))
    rc = nd.main(["run", "--host", "h", "--netrc", str(secret_files[0]), "--hf-token-file", str(secret_files[1]),
                  "--dry-run", "--", "nvidia-smi"])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "--- remote script ---" in out and WANDB not in out and HF not in out


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not installed")
def test_the_remote_script_exports_what_stdin_carries(tmp_path):
    """Run the fg script locally, as the node's `bash -c` would: the job sees the values, an empty line stays unset."""
    script = remote_script(str(tmp_path), 'printf "%s|%s|%s" "$WANDB_API_KEY" "${HF_TOKEN-unset}" "$PWD"', "fg", "t")
    done = subprocess.run(["bash", "-c", script], input=Secrets(WANDB).stdin_payload(), capture_output=True,
                          text=True, check=True, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})  # fmt: skip
    assert done.stdout.endswith(f"{WANDB}|unset|{tmp_path.resolve()}") or done.stdout.endswith(
        f"{WANDB}|unset|{tmp_path}"
    )


def test_a_custom_log_path_is_quoted_everywhere():
    for mode in ("nohup", "tmux"):
        script = remote_script("/r", "true", mode, "j", log="/tmp/a b$(x).log")
        assert "$(x)" not in script.replace("'/tmp/a b$(x).log'", "")


def test_run_argv_wraps_the_script_for_the_remote_shell():
    argv = run_argv(Node("h", 22), "echo hi\n")
    assert argv[-1].startswith("bash -c ") and argv[-2] == "root@h"
