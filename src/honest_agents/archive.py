"""Save the key data of one iteration (a training run, an evaluation, a dataset build) into a small, self-describing folder.

    python -m honest_agents.archive --iteration 9 --name grpo-run2 \\
        --source root@HOST:/root/restraint --port PORT --key ~/.ssh/vast_ed25519 \\
        --run outputs/grpo_v2/honest-a0.1-qwen3.5-4b-s42 --eval runs/eval/val_base --log /root/grpo_v2.log
    python -m honest_agents.archive ... --source ~/Documents/honest-agents-archive/2026-10-06-node1/repo-files   # a local mirror

What is kept (docs/training-log.md "Saving an iteration"):
* a run directory (GRPO or SFT): every top-level file (configs, `train_log.jsonl`, probe results, `best_checkpoint.json`,
  SFT `examples.jsonl`), `completions/`, and the adapter weights of `final/` and of the selected checkpoint when it is not
  the last one. Optimizer states, intermediate checkpoints and tokenizer copies are skipped (a 50-step GRPO run: ~4.8 GB
  on the node, ~275 MB here; `--no-weights` keeps only the ~15 MB of logs);
* an evaluation directory: `summary.md`, `metrics.json`, `config.json`, `episodes.jsonl` (not the transcripts; the
  trajectory bank only with `--keep-bank`);
* `--file`: single files relative to the source root (dataset splits, configs); `--log`: absolute log paths.
Text files over 1 MB are gzipped. `ITERATION.md` records what was saved from where (and the source's git commit when the
source is the node), `MANIFEST.sha256` the checksums.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DEST = Path.home() / "Documents" / "honest-agents-archive" / "iterations"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "README.md")
EVAL_FILES = ("summary.md", "metrics.json", "config.json", "episodes.jsonl")
GZIP_SUFFIXES = (".jsonl", ".json", ".log", ".md", ".txt", ".csv")
GZIP_MIN_BYTES = 1_000_000
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


# ── selection (pure, unit-tested) ────────────────────────────────────────────


def select_run_files(paths: Sequence[str], best_checkpoint: str | None = None, weights: bool = True) -> list[str]:
    """Files of a run directory to keep, from its relative file paths (see the module docstring)."""
    keep = []
    checkpoints = sorted({int(m.group(1)) for p in paths if (m := _CHECKPOINT.match(p.split("/")[0]))})
    has_final = "final/adapter_model.safetensors" in paths
    best = best_checkpoint.rstrip("/").split("/")[-1] if best_checkpoint else None
    best_is_last = bool(best and checkpoints and best == f"checkpoint-{checkpoints[-1]}" and has_final)
    for p in paths:
        top, _, rest = p.partition("/")
        if not rest:
            keep.append(p)  # top-level file
        elif (
            top == "completions"
            or weights
            and top == "final"
            and rest in ADAPTER_FILES
            or weights
            and best
            and top == best
            and not best_is_last
            and rest in ADAPTER_FILES
        ):
            keep.append(p)
    return sorted(keep)


def select_eval_files(paths: Sequence[str], keep_bank: bool = False) -> list[str]:
    return sorted(p for p in paths if p in EVAL_FILES or (keep_bank and p.startswith("bank/")))


# ── sources ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """A repo root on the node (`user@host:/path`) or a local directory (a mirror)."""

    spec: str
    port: int | None = None
    key: str | None = None

    @property
    def remote(self) -> bool:
        return ":" in self.spec and not Path(self.spec).exists()

    @property
    def host(self) -> str:
        return self.spec.split(":", 1)[0]

    @property
    def root(self) -> str:
        return self.spec.split(":", 1)[1] if self.remote else str(Path(self.spec).expanduser())

    def _ssh_opts(self) -> list[str]:
        opts = ["-o", "IdentitiesOnly=yes"] if self.key else []
        if self.key:
            opts += ["-i", str(Path(self.key).expanduser())]
        if self.port:
            opts += ["-p", str(self.port)]
        return opts

    def ssh(self, command: str) -> str:
        res = subprocess.run(["ssh", *self._ssh_opts(), self.host, command], capture_output=True, text=True, check=True,
                             stdin=subprocess.DEVNULL)  # fmt: skip
        return res.stdout

    def list_files(self, rel_dir: str) -> list[str]:
        """Relative paths of the regular files under `rel_dir` (empty if it does not exist)."""
        if self.remote:
            out = self.ssh(f"cd {self.root}/{rel_dir} 2>/dev/null && find . -type f | sed 's#^\\./##'")
            return sorted(line for line in out.splitlines() if line)
        base = Path(self.root) / rel_dir
        if not base.is_dir():
            return []
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())

    def read_text(self, rel_path: str) -> str | None:
        if self.remote:
            try:
                return self.ssh(f"cat {self.root}/{rel_path}")
            except subprocess.CalledProcessError:
                return None
        p = Path(self.root) / rel_path
        return p.read_text() if p.is_file() else None

    def copy(self, rel_dir: str, files: Sequence[str], dest: Path) -> None:
        """Copy `files` (relative to `rel_dir`) into `dest`, keeping their sub-paths."""
        if not files:
            return
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(files) + "\n")
            listing = f.name
        try:
            src = f"{self.host}:{self.root}/{rel_dir}/" if self.remote else f"{self.root}/{rel_dir}/"
            cmd = ["rsync", "-a", f"--files-from={listing}", src, f"{dest}/"]
            if self.remote:
                cmd[1:1] = ["-e", " ".join(["ssh", *self._ssh_opts()])]
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        finally:
            os.unlink(listing)

    def copy_absolute(self, path: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        src = f"{self.host}:{path}" if self.remote else str(Path(path).expanduser())
        cmd = ["rsync", "-a", src, f"{dest}/"]
        if self.remote:
            cmd[1:1] = ["-e", " ".join(["ssh", *self._ssh_opts()])]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)

    def git_commit(self) -> str | None:
        if not self.remote:
            return None
        try:
            return self.ssh(f"git -C {self.root} log -1 --format='%h %s'").strip() or None
        except subprocess.CalledProcessError:
            return None


# ── post-processing ──────────────────────────────────────────────────────────


def gzip_large_text(folder: Path, min_bytes: int = GZIP_MIN_BYTES) -> list[Path]:
    done = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix in GZIP_SUFFIXES and p.stat().st_size >= min_bytes:
            with p.open("rb") as src, gzip.open(p.with_name(p.name + ".gz"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            p.unlink()
            done.append(p)
    return done


def write_manifest(folder: Path) -> int:
    lines = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder)}")
    (folder / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return len(lines)


def folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


# ── save ─────────────────────────────────────────────────────────────────────


def save(
    source: Source,
    dest_root: Path,
    iteration: int,
    name: str,
    runs: Sequence[str] = (),
    evals: Sequence[str] = (),
    files: Sequence[str] = (),
    logs: Sequence[str] = (),
    weights: bool = True,
    keep_bank: bool = False,
    note: str = "",
) -> Path:
    folder = dest_root / f"{iteration:02d}-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {"runs": {}, "evals": {}, "files": [], "logs": []}
    for rel in runs:
        paths = source.list_files(rel)
        best_raw = source.read_text(f"{rel}/best_checkpoint.json")
        best = json.loads(best_raw).get("checkpoint") if best_raw else None
        keep = select_run_files(paths, best, weights=weights)
        source.copy(rel, keep, folder / "run" / Path(rel).name)
        record["runs"][rel] = {"files_on_source": len(paths), "kept": len(keep), "best_checkpoint": best}  # type: ignore[index]
    for rel in evals:
        paths = source.list_files(rel)
        keep = select_eval_files(paths, keep_bank=keep_bank)
        source.copy(rel, keep, folder / "eval" / Path(rel).name)
        record["evals"][rel] = {"files_on_source": len(paths), "kept": len(keep)}  # type: ignore[index]
    for rel in files:
        source.copy(str(Path(rel).parent), [Path(rel).name], folder / "files" / Path(rel).parent)
        record["files"].append(rel)  # type: ignore[union-attr]
    for path in logs:
        source.copy_absolute(path, folder / "logs")
        record["logs"].append(path)  # type: ignore[union-attr]
    zipped = gzip_large_text(folder)
    commit = source.git_commit()
    size = folder_bytes(folder)
    lines = [
        f"# Iteration {iteration:02d}: {name}",
        "",
        f"- saved: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"- source: `{source.spec}`" + (f" (git `{commit}`)" if commit else ""),
        f"- size: {size / 1e6:.1f} MB; weights kept: {weights}; gzipped: {len(zipped)} file(s)",
        f"- write-up: docs/training-log.md, entry {iteration}",
    ]
    if note:
        lines += ["", note]
    lines += ["", "```json", json.dumps(record, indent=2), "```", ""]
    (folder / "ITERATION.md").write_text("\n".join(lines))
    write_manifest(folder)
    return folder


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--name", required=True, help="short slug, e.g. grpo-run2")
    ap.add_argument("--source", required=True, help="user@host:/repo/root on the node, or a local directory")
    ap.add_argument("--port", type=int)
    ap.add_argument("--key", help="ssh key for a remote source")
    ap.add_argument("--run", action="append", default=[], help="run directory relative to the source root (repeatable)")
    ap.add_argument("--eval", action="append", default=[], help="evaluation directory relative to the source root")
    ap.add_argument("--file", action="append", default=[], help="single file relative to the source root")
    ap.add_argument("--log", action="append", default=[], help="absolute path of a log file on the source")
    ap.add_argument("--no-weights", action="store_true", help="skip adapter weights (logs and results only)")
    ap.add_argument("--keep-bank", action="store_true", help="also keep an evaluation's trajectory bank")
    ap.add_argument("--note", default="", help="one line for ITERATION.md")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    a = ap.parse_args(argv)
    folder = save(Source(a.source, a.port, a.key), a.dest.expanduser(), a.iteration, a.name, a.run, a.eval, a.file,
                  a.log, weights=not a.no_weights, keep_bank=a.keep_bank, note=a.note)  # fmt: skip
    print(f"{folder}  {folder_bytes(folder) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
