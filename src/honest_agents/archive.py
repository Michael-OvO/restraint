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
Text files over 1 MB are gzipped (with a fixed header, so the same file always gzips to the same bytes). `ITERATION.md`
records what was saved from where (and the source's git commit when the source is the node) and, per eval, the content
hashes of its episodes; `MANIFEST.sha256` the checksums. Saving an eval whose episodes are already archived under
another name prints a warning and notes it in `ITERATION.md` (DEF-12).

Check the whole archive for duplicate episode sets (sweep S8; a cached hash per file, re-hashed only when its size or
mtime changes):

    python -m honest_agents.archive check [--dest DIR]

A duplicate is two episode files with the same raw bytes, the same unpacked content, or the same sorted episode ids
(ids carry a random suffix per rollout, so two runs never share them). One that is a known copy is declared in the
copy's `ITERATION.md` with a line holding `` `eval/<name>` is a copy of `<folder>/eval/<name>` ``; the check exits 1
only on undeclared duplicates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DEST = Path.home() / "Documents" / "honest-agents-archive" / "iterations"
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "README.md")
EVAL_FILES = ("summary.md", "metrics.json", "config.json", "episodes.jsonl")
GZIP_SUFFIXES = (".jsonl", ".json", ".log", ".md", ".txt", ".csv")
GZIP_MIN_BYTES = 1_000_000
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")
EPISODE_FILES = ("episodes.jsonl", "episodes.jsonl.gz")
HASH_CACHE = Path.home() / ".cache" / "honest-agents" / "archive-index" / "episode_hashes.jsonl"
HASH_SCHEMA = 1
_EPISODE_ID = re.compile(r'^\{"episode_id": "([^"\\]*)"')
_DECLARED_COPY = re.compile(r"`(eval/[^`]+)` is a copy of `([^`]+)`")


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
            # mtime 0 and no name in the header: the same content gzips to the same bytes, so a copy hashes the same
            with (
                p.open("rb") as src,
                p.with_name(p.name + ".gz").open("wb") as raw,
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as dst,
            ):
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


# ── content hashes and duplicates (DEF-12, sweep S8) ─────────────────────────


@dataclass(frozen=True)
class EpisodeHashes:
    """Content hashes of one episodes file: its raw bytes, its unpacked bytes, and its sorted episode ids."""

    path: str  # relative to the archive root
    size: int
    mtime_ns: int
    file_sha256: str
    content_sha256: str
    ids_sha256: str
    n_episodes: int


def _episode_id(line: bytes) -> str:
    m = _EPISODE_ID.match(line.decode("utf-8", "replace"))
    if m:
        return m.group(1)
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        row = None
    if isinstance(row, dict) and (eid := row.get("episode_id") or row.get("trial_name")):
        return str(eid)
    return "sha256:" + hashlib.sha256(line).hexdigest()  # no id: the line itself stands for the episode


def hash_episodes(path: Path, root: Path) -> EpisodeHashes:
    st = path.stat()
    raw = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            raw.update(block)
    content = hashlib.sha256()
    ids = []
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as f:
        for line in f:
            content.update(line)
            if line.strip():
                ids.append(_episode_id(line.strip()))
    ids_hash = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    return EpisodeHashes(str(path.relative_to(root)), st.st_size, st.st_mtime_ns, raw.hexdigest(), content.hexdigest(),
                         ids_hash, len(ids))  # fmt: skip


def find_episode_files(root: Path) -> list[Path]:
    return sorted(p for name in EPISODE_FILES for p in root.rglob(name) if p.is_file())


def _load_hash_cache(cache: Path | None) -> dict[str, EpisodeHashes]:
    out: dict[str, EpisodeHashes] = {}
    if cache is None or not cache.is_file():
        return out
    for line in cache.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.pop("schema") != HASH_SCHEMA:
                continue
            root = row.pop("root")
            h = EpisodeHashes(**row)
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue  # a corrupt or old-schema row is ignored, never trusted
        out[f"{root}\0{h.path}"] = h  # a later row for the same file wins
    return out


def hash_archive(root: Path, cache: Path | None = HASH_CACHE, jobs: int = 4) -> list[EpisodeHashes]:
    """Hashes of every episodes file under `root`; files whose size and mtime match the cache are not re-read."""
    root = root.resolve()
    cached = _load_hash_cache(cache)
    found, todo = [], []
    for p in find_episode_files(root):
        st = p.stat()
        h = cached.get(f"{root}\0{p.relative_to(root)}")
        if h and h.size == st.st_size and h.mtime_ns == st.st_mtime_ns:
            found.append(h)
        else:
            todo.append(p)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:  # hashlib and zlib release the GIL
        fresh = list(pool.map(lambda p: hash_episodes(p, root), todo))
    if cache is not None and fresh:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a") as f:
            for h in fresh:
                f.write(json.dumps({"schema": HASH_SCHEMA, "root": str(root), **asdict(h)}) + "\n")
    return sorted(found + fresh, key=lambda h: h.path)


def duplicate_groups(hashes: Iterable[EpisodeHashes]) -> list[list[EpisodeHashes]]:
    """Groups of episode files that share their raw bytes, unpacked content or sorted episode ids (transitively)."""
    hashes = list(hashes)
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in ("file_sha256", "content_sha256", "ids_sha256"):
        first: dict[str, int] = {}
        for i, h in enumerate(hashes):
            if h.n_episodes == 0 and key == "ids_sha256":
                continue  # empty files are not duplicates of each other
            j = first.setdefault(getattr(h, key), i)
            parent[find(i)] = find(j)
    groups: dict[int, list[EpisodeHashes]] = {}
    for i, h in enumerate(hashes):
        groups.setdefault(find(i), []).append(h)
    return [g for g in groups.values() if len(g) > 1]


def declared_copies(root: Path) -> set[tuple[str, str]]:
    """(copy eval dir, original eval dir) pairs declared in the iterations' `ITERATION.md` files, relative to `root`."""
    out = set()
    for doc in sorted(root.glob("*/ITERATION.md")):
        for copy, original in _DECLARED_COPY.findall(doc.read_text()):
            out.add((f"{doc.parent.name}/{copy}", original.rstrip("/")))
    return out


def undeclared(group: Sequence[EpisodeHashes], declared: set[tuple[str, str]]) -> list[str]:
    """Eval dirs of a duplicate group beyond one original that are not declared copies of another member."""
    dirs = sorted({str(Path(h.path).parent) for h in group})
    copies = {c for c, o in declared if c in dirs and o in dirs and c != o}
    rest = [d for d in dirs if d not in copies]
    return rest[1:]  # one undeclared member is the original


def check(root: Path, cache: Path | None = HASH_CACHE, jobs: int = 4) -> int:
    """Print the archive's duplicate episode sets; return the number of undeclared ones (sweep S8)."""
    hashes = hash_archive(root, cache, jobs)
    declared = declared_copies(root)
    bad = 0
    for group in duplicate_groups(hashes):
        extra = undeclared(group, declared)
        bad += len(extra)
        print(
            ("DUPLICATE (undeclared)" if extra else "duplicate (declared copy)") + f": {group[0].n_episodes} episodes"
        )
        for h in group:
            same = [k for k in ("file_sha256", "content_sha256", "ids_sha256")
                    if all(getattr(o, k) == getattr(h, k) for o in group)]  # fmt: skip
            print(f"  {h.path}  ({', '.join(same) or 'linked through another member'} equal)")
    print(f"{len(hashes)} episode file(s) under {root}; {bad} undeclared duplicate(s)")
    return bad


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
    hash_cache: Path | None = HASH_CACHE,
) -> Path:
    dest_root = dest_root.resolve()
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
    duplicates = []
    new: dict[str, EpisodeHashes] = {}
    for rel in evals:
        d = folder / "eval" / Path(rel).name
        if ep := next((d / n for n in EPISODE_FILES if (d / n).is_file()), None):
            new[rel] = h = hash_episodes(ep, dest_root)
            record["evals"][rel].update(  # type: ignore[index]
                {"episodes_sha256": h.content_sha256, "episode_ids_sha256": h.ids_sha256, "n_episodes": h.n_episodes}
            )
    if new:  # compare with everything already archived (cached hashes) and with each other
        others = [h for h in hash_archive(dest_root, cache=hash_cache) if not h.path.startswith(folder.name + "/")]
        for group in duplicate_groups([*new.values(), *others]):
            for h in (h for h in group if h in new.values()):
                dup = ", ".join(f"`{Path(o.path).parent}`" for o in group if o is not h)
                duplicates.append(f"- duplicate: `{Path(h.path).parent}` has the same episodes as {dup} (DEF-12)")
                print(f"WARNING: {Path(h.path).parent} has the same episodes as {dup}", file=sys.stderr)
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
    if duplicates:
        lines += ["", *duplicates]
    lines += ["", "```json", json.dumps(record, indent=2), "```", ""]
    (folder / "ITERATION.md").write_text("\n".join(lines))
    write_manifest(folder)
    return folder


def main_check(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.archive check",
                                 description="List duplicate episode sets in the archive (sweep S8, DEF-12).")  # fmt: skip
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="archive root (the iterations folder)")
    ap.add_argument("--cache", type=Path, default=HASH_CACHE, help="hash cache (JSONL), outside the archive")
    ap.add_argument("--no-cache", action="store_true", help="re-hash every file and write no cache")
    ap.add_argument("--jobs", type=int, default=4, help="files hashed in parallel")
    a = ap.parse_args(argv)
    return 1 if check(a.dest.expanduser(), None if a.no_cache else a.cache.expanduser(), a.jobs) else 0


def main(argv: Sequence[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["check"]:
        raise SystemExit(main_check(argv[1:]))
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
