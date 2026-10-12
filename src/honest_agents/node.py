"""Work on a rented node that has no GitHub key: push the tree, pull results, start jobs with secrets that never touch disk.

    python -m honest_agents.node sync --host HOST --port PORT [--with-data] [--dry-run]
    python -m honest_agents.node run  --host HOST --port PORT --name grpo3 -- scripts/run_grpo.sh configs/x.yaml
    python -m honest_agents.node pull --host HOST --port PORT runs/eval/val_step50 outputs/grpo_v3/run1/train_log.jsonl

Why this exists. The node is a short-lived rental (docs/node-runbook.md): giving it a GitHub deploy key means one more
credential to create, scope and revoke for every box, so the code goes over rsync from the Mac instead, `.git` included
(so `git log` on the node and `honest_agents.archive` can still name the commit). Datasets are gitignored and large, so
they travel only when asked (`--with-data`; the probe lists of `tasks.probe_set` / `eval.probe_set` live there too, so the
first sync of a node, which needs `data/` anyway, carries them). Training needs the W&B key (and the Hugging Face token for gated weights);
writing them into a file on the node or onto a command line (which every user on the box sees in `ps`, and which shell
history keeps) leaks them to whoever rents the machine next. `run` therefore reads them on the Mac at call time
(`~/.netrc`, machine `api.wandb.ai`; `~/.cache/huggingface/token`) and sends them through ssh's stdin to a small remote
script that `read`s them into exported variables and then starts the job; ssh's argv and the remote script contain only
variable names. The job then sees `WANDB_API_KEY` and `HF_TOKEN` in its environment and nowhere else (they are in the
process environment, which only root and the job's own user can read).

Session modes for `run`: `nohup` (default; `setsid nohup`, survives the ssh disconnect, output appended to the log),
`tmux` (a dedicated tmux server per job, `tmux -L NAME`, so the server is started by this call and inherits the secrets;
an already running server would not pass them on, so a name in use is refused), and `fg` (attached; for short checks).

`pull` copies node paths into a staging folder of the archive, keeping their repo-relative layout; the curated,
self-describing save of an iteration remains `python -m honest_agents.archive`.
"""

from __future__ import annotations

import argparse
import netrc
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_KEY = "~/.ssh/vast_ed25519"
DEFAULT_DEST = "/root/restraint"
DEFAULT_STAGING = Path.home() / "Documents" / "honest-agents-archive" / "staging"
DEFAULT_NETRC = Path.home() / ".netrc"
DEFAULT_HF_TOKEN = Path.home() / ".cache" / "huggingface" / "token"
WANDB_MACHINE = "api.wandb.ai"
MODES = ("nohup", "tmux", "fg")
# Never pushed: local environments and caches, and everything a run writes on the node (pushing an empty or stale
# `outputs/` over the node's would be worse than useless). Leading `/` anchors a pattern at the repo root.
# `.env` (gitignored) may hold keys and must never land on a box the next renter gets.
EXCLUDES = (
    ".venv", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".DS_Store", "*.log",
    ".env", ".env.*", ".coverage", "htmlcov", "*.egg-info",
    "/outputs", "/runs", "/wandb", "/.claude",
)  # fmt: skip
DATA_EXCLUDE = "/data"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


# ── the node ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Node:
    """One ssh target. `host` may carry a user (`root@1.2.3.4`); without one the user is root, as on Vast."""

    host: str
    port: int = 22
    key: str = DEFAULT_KEY
    dest: str = DEFAULT_DEST

    @property
    def target(self) -> str:
        return self.host if "@" in self.host else f"root@{self.host}"

    @property
    def key_path(self) -> str:
        return str(Path(self.key).expanduser())

    def ssh_options(self) -> list[str]:
        # IdentitiesOnly: with several keys in the agent, ssh may offer the wrong ones first and hit MaxAuthTries
        return ["-i", self.key_path, "-p", str(self.port), "-o", "IdentitiesOnly=yes"]

    def ssh_argv(self, remote_command: str) -> list[str]:
        return ["ssh", *self.ssh_options(), self.target, remote_command]

    def rsync_shell(self) -> str:
        """The `rsync -e` value; rsync splits it like a shell, so a key path with spaces is quoted."""
        return " ".join(["ssh", *(shlex.quote(o) for o in self.ssh_options())])


# ── sync and pull (pure argv builders, unit-tested) ──────────────────────────


def sync_argv(node: Node, src: Path = REPO, with_data: bool = False, dry_run: bool = False,
              delete: bool = False) -> list[str]:  # fmt: skip
    """rsync of the working tree to `node.dest`. `--delete` is opt-in: it removes node files the Mac does not have
    (never the excluded ones, which rsync protects unless told otherwise)."""
    argv = ["rsync", "-az", "--human-readable", "-e", node.rsync_shell()]
    if dry_run:
        argv += ["--dry-run", "--itemize-changes"]
    if delete:
        argv.append("--delete")
    excludes = [*EXCLUDES] + ([] if with_data else [DATA_EXCLUDE])
    argv += [f"--exclude={e}" for e in excludes]
    argv += [f"{src}/", f"{node.target}:{node.dest.rstrip('/')}/"]
    return argv


def pull_plan(node: Node, paths: Sequence[str], staging: Path = DEFAULT_STAGING) -> list[tuple[str, Path]]:
    """(node path, local directory) per path, keeping each path's layout below the repo root: `runs/eval/x` lands in
    `staging/runs/eval/x`; an absolute path (a log in /root) keeps its full path under `staging`.

    One rsync per path into its parent directory rather than `rsync -R` with a `/./` marker: the Mac's stock rsync is
    openrsync, which ignores the marker and would recreate `/root/restraint/...` under `staging`."""
    if not paths:
        raise ValueError("pull needs at least one path")
    plan = []
    for p in paths:
        p = p.rstrip("/")
        if not p or ".." in Path(p).parts:
            raise ValueError(f"pull path {p!r}: give a path below --dest, or an absolute one, without '..'")
        remote = p if p.startswith("/") else f"{node.dest.rstrip('/')}/{p}"
        plan.append((remote, staging / Path(p.lstrip("/")).parent))
    return plan


def pull_argv(node: Node, remote: str, local_dir: Path, dry_run: bool = False) -> list[str]:
    """rsync of one node path into `local_dir` (no trailing slash on the source, so a directory arrives as itself)."""
    argv = ["rsync", "-az", "--human-readable", "-e", node.rsync_shell()]
    if dry_run:
        argv += ["--dry-run", "--itemize-changes"]
    return [*argv, f"{node.target}:{remote}", f"{local_dir}/"]


# ── secrets ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Secrets:
    """Values read on the Mac at call time. `repr` never shows them, so a stray print or a traceback cannot leak them."""

    wandb_api_key: str | None = field(default=None, repr=False)
    hf_token: str | None = field(default=None, repr=False)

    def names(self) -> list[str]:
        """Which secrets will be sent (names only; this is what the CLI prints)."""
        return [n for n, v in (("WANDB_API_KEY", self.wandb_api_key), ("HF_TOKEN", self.hf_token)) if v]

    def stdin_payload(self) -> str:
        """Two lines, in the order the remote script reads them; an empty line means absent."""
        values = [self.wandb_api_key or "", self.hf_token or ""]
        if any("\n" in v or "\r" in v for v in values):
            raise ValueError("a secret contains a newline")
        return "".join(v + "\n" for v in values)


def read_secrets(netrc_path: Path = DEFAULT_NETRC, hf_token_path: Path = DEFAULT_HF_TOKEN) -> Secrets:
    """W&B key from the netrc entry `machine api.wandb.ai` (what `wandb login` writes); HF token from the cache file
    (what `hf auth login` writes). Either may be missing: the job then runs without it."""
    wandb = None
    if netrc_path.is_file():
        try:
            auth = netrc.netrc(str(netrc_path)).authenticators(WANDB_MACHINE)
        except netrc.NetrcParseError as e:  # its message can quote a token from the file, so it is not passed on
            raise SystemExit(f"{netrc_path}: cannot parse (line {e.lineno}); fix it or pass --no-secrets") from None
        wandb = (auth[2] or None) if auth else None
    hf = (hf_token_path.read_text().strip() or None) if hf_token_path.is_file() else None
    return Secrets(wandb, hf)


# ── run ──────────────────────────────────────────────────────────────────────


def default_name() -> str:
    return time.strftime("job-%Y%m%d-%H%M%S", time.gmtime())


def remote_script(dest: str, command: str, mode: str = "nohup", name: str = "job", log: str | None = None) -> str:
    """The bash program the node runs: read the two secret lines from stdin, export the non-empty ones, then start
    `command` (a shell command line, run by a login shell in `dest` so the node's PATH, with uv, applies).
    It contains variable names only; the values arrive on stdin."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not _NAME.match(name):
        raise ValueError(f"session name {name!r}: letters, digits, '.', '_' and '-' only")
    log = log or f"/root/{name}.log"
    q = shlex.quote
    head = [
        "set -eu",
        "IFS= read -r WANDB_API_KEY || true",
        "IFS= read -r HF_TOKEN || true",
        'if [ -n "$WANDB_API_KEY" ]; then export WANDB_API_KEY; else unset WANDB_API_KEY; fi',
        'if [ -n "$HF_TOKEN" ]; then export HF_TOKEN; else unset HF_TOKEN; fi',
        f"cd {q(dest)}",
    ]
    if mode == "fg":
        return "\n".join([*head, f"exec bash -lc {q(command)}"]) + "\n"
    if mode == "nohup":
        body = [
            f'echo "== $(date -u +%FT%TZ) start: "{q(command)} >> {q(log)}',
            f"setsid nohup bash -lc {q(command)} >> {q(log)} 2>&1 < /dev/null &",
            f'echo "started {name}: pid $! log "{q(log)}',
        ]
        return "\n".join([*head, *body]) + "\n"
    # tmux: a server of its own (-L NAME), started right here, so it inherits the exported secrets
    # the subshell makes the whole command line (`a && b`) write to the log, not only its last part
    inner = f'( {command}\n) 2>&1 | tee -a {q(log)}; echo "== exit ${{PIPESTATUS[0]}}" | tee -a {q(log)}'
    body = [
        f"if tmux -L {q(name)} has-session 2>/dev/null; then echo 'tmux server {name} is running; pick another --name' >&2; exit 1; fi",
        f"tmux -L {q(name)} new-session -d -s {q(name)} {q('bash -lc ' + q(inner))}",
        f'echo "started {name}: tmux -L {name} attach -t {name}; log "{q(log)}',
    ]  # fmt: skip
    return "\n".join([*head, *body]) + "\n"


def run_argv(node: Node, script: str) -> list[str]:
    """ssh argv; the remote side runs `bash -c SCRIPT`, which reads the secrets from the ssh channel's stdin."""
    return node.ssh_argv(f"bash -c {shlex.quote(script)}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _print_argv(argv: Sequence[str]) -> None:
    print("$ " + shlex.join(argv), flush=True)


def _node_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--host", required=True, help="node address, optionally user@ (default user root)")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--key", default=DEFAULT_KEY, help=f"ssh private key (default {DEFAULT_KEY})")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"repo root on the node (default {DEFAULT_DEST})")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.node", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="rsync this working tree to the node")
    _node_args(s)
    s.add_argument("--src", type=Path, default=REPO, help="local repo root (default: this checkout)")
    s.add_argument("--with-data", action="store_true", help="also send data/ (datasets and task directories)")
    s.add_argument("--delete", action="store_true", help="delete node files the Mac does not have (excluded ones kept)")
    s.add_argument("--dry-run", action="store_true", help="list what would be transferred, change nothing")

    p = sub.add_parser("pull", help="rsync node paths into the archive staging folder")
    _node_args(p)
    p.add_argument("paths", nargs="+", help="paths relative to --dest on the node, or absolute (logs)")
    p.add_argument("--staging", type=Path, default=DEFAULT_STAGING, help=f"local target (default {DEFAULT_STAGING})")
    p.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("run", help="start a command on the node with W&B/HF secrets passed over stdin")
    _node_args(r)
    r.add_argument("--mode", choices=MODES, default="nohup")
    r.add_argument("--name", default=None, help="session name; default job-<UTC time>; log /root/<name>.log")
    r.add_argument("--log", default=None, help="log path on the node (nohup and tmux modes)")
    r.add_argument("--netrc", type=Path, default=DEFAULT_NETRC, help="where the W&B key is read (machine api.wandb.ai)")
    r.add_argument("--hf-token-file", type=Path, default=DEFAULT_HF_TOKEN)
    r.add_argument("--no-secrets", action="store_true", help="send no secrets (empty lines)")
    r.add_argument("--dry-run", action="store_true", help="print the ssh command and remote script, send nothing")
    r.add_argument("command", nargs=argparse.REMAINDER, help="after `--`: the command line to run in --dest")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    node = Node(a.host, a.port, a.key, a.dest)
    if a.cmd == "sync":
        src = a.src.expanduser()
        if (src / ".git").is_file():  # a git worktree: .git is a pointer to a gitdir on the Mac, useless on the node
            print(f"warning: {src} is a git worktree; the node gets no usable .git (sync from the main checkout)",
                  file=sys.stderr)  # fmt: skip
        cmd = sync_argv(node, src, with_data=a.with_data, dry_run=a.dry_run, delete=a.delete)
        _print_argv(cmd)
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, check=False).returncode
    if a.cmd == "pull":
        rc = 0
        for remote, local_dir in pull_plan(node, a.paths, a.staging.expanduser()):
            cmd = pull_argv(node, remote, local_dir, dry_run=a.dry_run)
            _print_argv(cmd)
            if not a.dry_run:
                local_dir.mkdir(parents=True, exist_ok=True)
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, check=False).returncode or rc
        return rc
    words = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not words:
        raise SystemExit("run: give the command after `--`")
    # one word is taken as a shell command line (pipes, &&); several are joined as a single quoted command
    command = words[0] if len(words) == 1 else shlex.join(words)
    script = remote_script(node.dest, command, a.mode, a.name or default_name(), a.log)
    cmd = run_argv(node, script)
    secrets = Secrets() if a.no_secrets else read_secrets(a.netrc.expanduser(), a.hf_token_file.expanduser())
    _print_argv(cmd)
    print(f"secrets over stdin: {', '.join(secrets.names()) or 'none'}", flush=True)
    if a.dry_run:
        print("--- remote script ---\n" + script, end="")
        return 0
    return subprocess.run(cmd, input=secrets.stdin_payload(), text=True, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
