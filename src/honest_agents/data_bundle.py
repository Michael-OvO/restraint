"""Pack, publish and restore the data an executor needs that git does not hold (data/ is gitignored).

    python -m honest_agents.data_bundle pack --split train|final --out BUNDLE_DIR [--archive ~/Documents/honest-agents-archive]
    python -m honest_agents.data_bundle push BUNDLE_DIR --repo ORG/NAME          # private HF dataset
    python -m honest_agents.data_bundle fetch --repo ORG/NAME [--dest .] [--force]
    python -m honest_agents.data_bundle verify BUNDLE_DIR

Why this exists. Task dirs, surface tasks, the G2 adapters and the split files live in data/ on the owner's Mac and on
nodes synced from it (`node sync --with-data`); an executor that only clones the repository has none of them, and one
missing file (data/spec_audit/reviews.jsonl) used to make the DEF-66 exclusion silently empty (DEF-79). A bundle is a
few large tar.gz parts (fast to download, resumable on the Hub) plus `MANIFEST.json`: per part its sha256 and the
sha256 of every file inside it. `fetch` verifies both before it writes anything into the checkout, keeps file mtimes
(DEF-25 depends on them) and refuses to overwrite existing files unless `--force`.

Two splits, so that a method can be trained by someone who cannot see the final benchmark.
- `train` (for the executor): `tasks` (data/tasks), `decision` (data/decision), `weights` (the SFT-v2 and G2 adapters
  G2 is merged from), `data` (the rest of data/ except data/rts, which no iteration-20 step reads), `runs` (the select
  runs the iteration-20 analyses read) and, with `--archive`, `archive` (the archived analyses and runs the design
  cites) and `builds` (every archived training build `final_benchmark never-trained` scans), both under `archive/`.
  Every file whose path names a held-out source (data/test_ids.txt) is left out, whatever part it would be in.
- `final` (for the evaluator only): exactly those held-out files (the test-split task dirs and their audit seeds),
  the held-out result runs (DEF-74 re-grade, it19a, it19a2) and, with `--archive`, the pinned benchmark downloads and
  texts `final_benchmark check --archive` and `never-trained --archive` read, plus the analyses made on held-out
  episodes. Restored over a `train` checkout it clashes with nothing; the two together are the full data.
The held-out ids and prompts themselves (data/test_ids.txt, data/test.jsonl, data/benchmarks/) stay in `train`: the
legality rule needs them, and data/benchmarks/internal_heldout.jsonl is tracked in git anyway. What `final` withholds
is the environments and hidden tests, and every episode run on them.
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
from typing import Literal

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "MANIFEST.json"
EXCLUDED_DATA = ("data/rts",)  # benchmark-scale data no iteration-20 step reads (2.5 GB)
HELD_OUT_IDS = "data/test_ids.txt"  # the held-out sources (split groups); the final benchmark's internal side
WEIGHTS = ("data/weights/it14/sft-v2-e3-final", "data/weights/it15/g2-s1-final")
FINAL_DATA = ("data/def74",)  # the DEF-74 re-grade of held-out episodes
# Archive paths (relative to the archive root) that docs/iteration-20-*.md cite; restored under archive/.
TRAIN_ARCHIVE = (
    "staging/analysis/it20_design",
    "staging/runs/eval/it19c_ablation_select",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/crgrpo_state.jsonl",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/completions",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/checkpoint-30/adapter_config.json",
    "staging/outputs/it19/it19-crgrpo-ablation-r2/checkpoint-30/adapter_model.safetensors",
)
FINAL_ARCHIVE = (
    "benchmarks",  # the pinned ImpossibleBench and MASK downloads (final_benchmark.ARCHIVE_IB, ARCHIVE_MASK)
    "2026-10-06-node1/repo-files/benchmarks",  # final_benchmark.ARCHIVE_TEXTS
    "staging/analysis/evidence_reward_offline_check",  # run on it19a held-out episodes
    "staging/runs/final/it19a2",
    "staging/runs/harbor/it19a2_g2_tb2_def76_invalid",
    "iterations/19-it19a-baseline",
)
# Result runs kept in this checkout's runs/ (gitignored); restored in place.
TRAIN_RUNS = ("runs/eval/it19c_g2_select", "runs/eval/it19c_method_c5_select", "runs/eval/it19c_ablation_select")
FINAL_RUNS = ("runs/def74", "runs/final/it19a2")
Split = Literal["train", "final"]


@dataclass(frozen=True)
class Part:
    name: str
    roots: tuple[str, ...]  # paths relative to `base`
    base: Path
    prefix: str = ""  # restored under this directory of the checkout ("" = repository root)
    keep: Literal["all", "train", "held_out"] = "all"  # filter on whether a path names a held-out source
    exclude: tuple[str, ...] = ()  # paths relative to `base` left out


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


def held_out_sources(repo: Path = REPO) -> frozenset[str]:
    """The held-out sources (`dataset/task`). Fails closed: without the list nothing could be withheld (cf. DEF-79)."""
    path = repo / HELD_OUT_IDS
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the held-out source list is needed to split the bundle")
    return frozenset(ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#"))


def names_held_out(rel: str, sources: frozenset[str]) -> bool:
    """Whether a path names a held-out source: a `dataset/task` pair of components (audit seeds) or a variant dir
    `dataset__task__kind[...]` (data/tasks)."""
    comps = rel.split("/")
    for i, c in enumerate(comps):
        if "__" in c and "/".join(c.split("__")[:2]) in sources:
            return True
        if i and f"{comps[i - 1]}/{c}" in sources:
            return True
    return False


def archived_builds(archive: Path) -> tuple[str, ...]:
    """Archive-relative paths of the training builds `final_benchmark never-trained --archive` scans."""
    from honest_agents.eval.final_benchmark import training_builds

    return tuple(sorted({str(p.relative_to(archive)) for _, p in training_builds(archive=archive)
                         if p.is_relative_to(archive)}))  # fmt: skip


def parts(repo: Path = REPO, archive: Path | None = None, split: Split = "train") -> list[Part]:
    data_rest = tuple(sorted(
        str(p.relative_to(repo))
        for p in (repo / "data").iterdir()
        if str(p.relative_to(repo)) not in ("data/tasks", "data/decision", "data/weights", *EXCLUDED_DATA, *FINAL_DATA)
    ))  # fmt: skip
    if split == "train":
        out = [
            Part("tasks", ("data/tasks",), repo, keep="train"),
            Part("decision", ("data/decision",), repo, keep="train"),
            Part("weights", WEIGHTS, repo, keep="train"),
            Part("data", data_rest, repo, keep="train"),
        ]
        if runs := tuple(r for r in TRAIN_RUNS if (repo / r).exists()):
            out.append(Part("runs", runs, repo, keep="train"))
        if archive is not None:
            out.append(Part("archive", TRAIN_ARCHIVE, archive, prefix="archive", keep="train"))
            out.append(Part("builds", archived_builds(archive), archive, prefix="archive", keep="train"))
        return out
    train_roots = ("data/tasks", "data/decision", *data_rest, *(r for r in TRAIN_RUNS if (repo / r).exists()))
    out = [Part("held_out", train_roots, repo, keep="held_out"), Part("data", FINAL_DATA, repo)]
    if runs := tuple(r for r in FINAL_RUNS if (repo / r).exists()):
        out.append(Part("runs", runs, repo))
    if archive is not None:
        out.append(Part("archive", FINAL_ARCHIVE, archive, prefix="archive"))
        # files of the train archive roots that name a held-out source belong here
        out.append(Part("archive_held_out", (*TRAIN_ARCHIVE, *archived_builds(archive)), archive, prefix="archive",
                        keep="held_out"))  # fmt: skip
    return out


def part_files(part: Part, sources: frozenset[str]) -> list[Path]:
    files = _files(part.base, part.roots, part.exclude)
    if part.keep == "all":
        return files
    want = part.keep == "held_out"
    return [f for f in files if names_held_out(str(f.relative_to(part.base)), sources) == want]


def pack(out_dir: Path, repo: Path = REPO, archive: Path | None = None, split: Split = "train") -> dict:
    sources = held_out_sources(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"format": 1, "split": split, "parts": {}}
    for part in parts(repo, archive, split):
        files = part_files(part, sources)
        if not files:
            continue
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
    p.add_argument("--split", choices=("train", "final"), required=True,
                   help="train: everything but the held-out files; final: only those (evaluator)")  # fmt: skip
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
        m = pack(a.out, archive=a.archive.expanduser() if a.archive else None, split=a.split)
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
