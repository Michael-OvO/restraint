"""Pack, publish and restore the data an executor needs that git does not hold (data/ is gitignored).

    python -m honest_agents.data_bundle pack --out BUNDLE_DIR [--archive ~/Documents/honest-agents-archive]
    python -m honest_agents.data_bundle push BUNDLE_DIR --repo USER/restraint-data          # private HF dataset
    python -m honest_agents.data_bundle fetch --repo USER/restraint-data [--dest .] [--force]
    python -m honest_agents.data_bundle verify BUNDLE_DIR

Why this exists. Task dirs, surface tasks, the G2 adapters and the split files live in data/ on the owner's Mac and on
nodes synced from it (`node sync --with-data`); an executor that only clones the repository has none of them, and one
missing file (data/spec_audit/reviews.jsonl) used to make the DEF-66 exclusion silently empty (DEF-79). A bundle is a
few large tar.gz parts (fast to download, resumable on the Hub) plus `MANIFEST.json`: per part its sha256 and the
sha256 of every file inside it. `fetch` verifies both before it writes anything into the checkout, keeps file mtimes
(DEF-25 depends on them) and refuses to overwrite existing files unless `--force`.

Parts: `tasks` (data/tasks), `decision` (data/decision), `weights` (the SFT-v2 and G2 adapters G2 is merged from),
`data` (the rest of data/ except data/rts, which no iteration-20 step reads), `runs` (the result runs in runs/ that
the iteration-20 analyses read) and, with `--archive`, `archive` (the archived analyses and result runs the iteration-20
design cites, restored under `archive/`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "MANIFEST.json"
EXCLUDED_DATA = ("data/rts",)  # benchmark-scale data no iteration-20 step reads (2.5 GB)
WEIGHTS = ("data/weights/it14/sft-v2-e3-final", "data/weights/it15/g2-s1-final")
# Archive paths (relative to the archive root) that docs/iteration-20-*.md cite; restored under archive/.
ARCHIVE_EXTRAS = (
    "staging/analysis",
    "staging/runs/final/it19a2",
    "staging/runs/harbor/it19a2_g2_tb2_def76_invalid",
    "staging/runs/eval/it19c_ablation_select",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/crgrpo_state.jsonl",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/completions",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/checkpoint-30/adapter_config.json",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/checkpoint-30/adapter_model.safetensors",
    "iterations/19-it19a-baseline",
)
# Result runs kept in this checkout's runs/ (gitignored) that the iteration-20 analyses read; restored in place.
RUNS = (
    "runs/def74",
    "runs/eval/it19c_g2_select",
    "runs/eval/it19c_method_c5_select",
    "runs/eval/it19c_ablation_select",
    "runs/final/it19a2",
)


@dataclass(frozen=True)
class Part:
    name: str
    roots: tuple[str, ...]  # paths relative to `base`
    base: Path
    prefix: str = ""  # restored under this directory of the checkout ("" = repository root)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(base: Path, roots: Sequence[str], exclude: Sequence[str] = ()) -> list[Path]:
    out: list[Path] = []
    for r in roots:
        p = base / r
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
        else:
            raise FileNotFoundError(f"{p}: not found (bundle input)")
    skip = tuple(base / e for e in exclude)
    return sorted({f for f in out if not any(f.is_relative_to(s) for s in skip)})


def parts(repo: Path = REPO, archive: Path | None = None) -> list[Part]:
    data_rest = sorted(
        str(p.relative_to(repo))
        for p in (repo / "data").iterdir()
        if str(p.relative_to(repo)) not in ("data/tasks", "data/decision", "data/weights", *EXCLUDED_DATA)
    )
    out = [
        Part("tasks", ("data/tasks",), repo),
        Part("decision", ("data/decision",), repo),
        Part("weights", WEIGHTS, repo),
        Part("data", tuple(data_rest), repo),
    ]
    runs = tuple(r for r in RUNS if (repo / r).exists())
    if runs:
        out.append(Part("runs", runs, repo))
    if archive is not None:
        out.append(Part("archive", ARCHIVE_EXTRAS, archive, prefix="archive"))
    return out


def pack(out_dir: Path, repo: Path = REPO, archive: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"format": 1, "parts": {}}
    for part in parts(repo, archive):
        files = _files(part.base, part.roots)
        tar_path = out_dir / f"{part.name}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in files:
                tar.add(f, arcname=str(f.relative_to(part.base)), recursive=False)
        manifest["parts"][part.name] = {
            "archive": tar_path.name,
            "sha256": sha256_file(tar_path),
            "prefix": part.prefix,
            "files": {str(f.relative_to(part.base)): sha256_file(f) for f in files},
        }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def verify(bundle: Path) -> list[str]:
    """Problems with a bundle directory: a missing or corrupted part, or a member that differs from the manifest."""
    manifest = json.loads((bundle / MANIFEST).read_text())
    problems: list[str] = []
    for name, info in manifest["parts"].items():
        tar_path = bundle / info["archive"]
        if not tar_path.is_file():
            problems.append(f"{name}: {tar_path.name} missing")
            continue
        if sha256_file(tar_path) != info["sha256"]:
            problems.append(f"{name}: {tar_path.name} sha256 differs from the manifest")
            continue
        with tarfile.open(tar_path, "r:gz") as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            if set(members) != set(info["files"]):
                problems.append(f"{name}: member list differs from the manifest")
                continue
            for arc, want in info["files"].items():
                fobj = tar.extractfile(members[arc])
                if fobj is None or hashlib.sha256(fobj.read()).hexdigest() != want:
                    problems.append(f"{name}: {arc} sha256 differs from the manifest")
    return problems


def restore(bundle: Path, dest: Path, *, force: bool = False) -> int:
    """Extract every part into `dest` after `verify`; returns the number of files written. Refuses to overwrite an
    existing file unless `force`; keeps mtimes."""
    problems = verify(bundle)
    if problems:
        raise ValueError("bundle failed verification: " + "; ".join(problems[:5]))
    manifest = json.loads((bundle / MANIFEST).read_text())
    written = 0
    for info in manifest["parts"].values():
        root = dest / info["prefix"] if info["prefix"] else dest
        clashes = [a for a in info["files"] if (root / a).exists()]
        if clashes and not force:
            raise FileExistsError(f"{len(clashes)} file(s) already exist, e.g. {root / clashes[0]}; use --force")
        with tarfile.open(bundle / info["archive"], "r:gz") as tar:
            tar.extractall(root, members=[m for m in tar.getmembers() if m.isfile()], filter="tar")
        written += len(info["files"])
    return written


def push(bundle: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi

    problems = verify(bundle)
    if problems:
        raise ValueError("refusing to push a bundle that fails verification: " + "; ".join(problems[:5]))
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(bundle), repo_id=repo_id, repo_type="dataset",
                      commit_message="restraint data bundle")  # fmt: skip
    return f"https://huggingface.co/datasets/{repo_id}"


def fetch(repo_id: str, dest: Path, *, force: bool = False, staging: Path | None = None) -> int:
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo_id, repo_type="dataset", local_dir=str(staging) if staging else None))
    return restore(local, dest, force=force)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.data_bundle", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="build the bundle from this checkout's data/ (and archive extras)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path, default=None, help="archive root; adds the `archive` part")
    v = sub.add_parser("verify", help="check a bundle directory against its manifest")
    v.add_argument("bundle", type=Path)
    u = sub.add_parser("push", help="verify, then upload a bundle to a private Hugging Face dataset repo")
    u.add_argument("bundle", type=Path)
    u.add_argument("--repo", required=True)
    f = sub.add_parser("fetch", help="download, verify and restore a bundle into a checkout")
    f.add_argument("--repo", required=True)
    f.add_argument("--dest", type=Path, default=REPO)
    f.add_argument("--staging", type=Path, default=None, help="where the download is kept (default: the HF cache)")
    f.add_argument("--force", action="store_true", help="overwrite files that already exist")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        m = pack(a.out, archive=a.archive.expanduser() if a.archive else None)
        for name, info in m["parts"].items():
            print(f"{name}: {len(info['files'])} files, {(a.out / info['archive']).stat().st_size / 1e6:.0f} MB")
        return 0
    if a.cmd == "verify":
        problems = verify(a.bundle)
        print("\n".join(problems) if problems else "ok")
        return 1 if problems else 0
    if a.cmd == "push":
        print(push(a.bundle, a.repo))
        return 0
    print(f"restored {fetch(a.repo, a.dest, force=a.force, staging=a.staging)} files into {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
