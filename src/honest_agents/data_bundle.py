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
  runs and G2's it17 training-pool bank with its DEF-74 re-grade) and, with `--archive`, `archive` (the design's
  analyses and every archived eval run on train, select, probe or surface tasks) and `builds` (every archived training
  build `final_benchmark never-trained` scans), both under `archive/`. Every file whose path names a held-out source
  (data/test_ids.txt) is left out, whatever part it would be in.
- `final` (for the evaluator only): exactly those held-out files (the test-split task dirs and their audit seeds),
  the held-out runs (every archived eval run whose episodes name a held-out source: the iteration-2 test baseline, the
  iteration-12 transfer runs, it19a, it19a2, the DEF-74 re-grade of it19a) and, with `--archive`, the pinned benchmark
  downloads and texts `final_benchmark check --archive` and `never-trained --archive` read, plus the analysis made on
  held-out episodes. Restored over a `train` checkout it shares no file; the two together are the full data. The Harbor
  job dirs (TB2 and ImpossibleBench trials) are in neither.
The held-out ids and prompts themselves (data/test_ids.txt, data/test.jsonl, data/benchmarks/) stay in `train`: the
legality rule needs them, and data/benchmarks/internal_heldout.jsonl is tracked in git anyway. What `final` withholds
is the environments and hidden tests, and every episode run on them.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
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
# data/def74 and runs/def74 hold the DEF-74 re-grade of two banks: G2's it17 training-pool rollouts (train) and the
# it19a held-out runs (final). affected.jsonl lists both, and is split by row (MIXED_FILES).
TRAIN_DATA = ("data/def74/sources/it17_rollouts_feasible", "data/def74/affected.jsonl")
FINAL_DATA = (
    "data/ext",  # the pinned ImpossibleBench copy the evaluation chains materialize from
    "data/def74/sources/it19a_base_heldout",
    "data/def74/sources/it19a_g2_heldout",
)
# Archive paths (relative to the archive root) that docs/iteration-20-*.md cite; restored under archive/.
TRAIN_ARCHIVE = (
    "staging/analysis/it20_design",
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
    "staging/runs/final",
    "staging/runs/harbor/it19a2_g2_tb2_def76_invalid",
    "iterations/19-it19a-baseline",  # its two sanity runs (train tasks) go to train: see `parts`
)
# Archived eval runs, each sorted by its contents (`archived_runs`): one that ran any held-out source goes to final.
# The Harbor job dirs (staging/runs/harbor, 1.2 GB of TB2 / ImpossibleBench trials) are in neither bundle.
ARCHIVE_RUNS = ("iterations/*/eval/*", "staging/runs/eval/*", "staging/runs/def74/*", "staging/runs/it1[0-9]/*")
# Result runs kept in this checkout's runs/ (gitignored); restored in place.
TRAIN_RUNS = (
    "runs/eval/it19c_g2_select",
    "runs/eval/it19c_method_c5_select",
    "runs/eval/it19c_ablation_select",
    "runs/def74/it17_rollouts_feasible",
    "runs/def74/regrade_it17_rollouts_feasible.jsonl",
    "runs/def74/regrade_it17_rollouts_feasible.summary.json",
)
FINAL_RUNS = (
    "runs/def74/it19a_base_heldout",
    "runs/def74/it19a_g2_heldout",
    "runs/def74/regrade_it19a_base_heldout.jsonl",
    "runs/def74/regrade_it19a_base_heldout.summary.json",
    "runs/def74/regrade_it19a_g2_heldout.jsonl",
    "runs/def74/regrade_it19a_g2_heldout.summary.json",
    "runs/final/it19a2",
)
# Files that mix held-out rows into otherwise train-side data, with answer-level content in those rows (oracle
# output, test logs, protected paths, blocker targets): the train bundle gets them without the held-out rows, the final
# bundle gets only those rows, as `held_out_name(path)`.
MIXED_FILES = {"data/validation.jsonl": "jsonl", "data/relabels.json": "dict", "data/def74/affected.jsonl": "jsonl"}
_TASK_HASH = re.compile(r"task_\d+_[0-9a-f]{8}")
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


def _mentions_held_out(path: Path, hashes: frozenset[str]) -> bool:
    """Whether a run (file or dir) ran a held-out source: any task hash in its episodes (or, without an episodes file,
    in any of its files) is a held-out one."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("episodes.jsonl*")) or sorted(f for f in path.rglob("*") if f.is_file())
    for f in files:
        opener = gzip.open if f.suffix == ".gz" else open
        try:
            with opener(f, "rt", errors="replace") as fh:
                if any(m in hashes for line in fh for m in _TASK_HASH.findall(line)):
                    return True
        except (OSError, EOFError):  # a binary file (an adapter, a parquet) names no task
            continue
    return False


def archived_runs(archive: Path, sources: frozenset[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The archived eval runs (`ARCHIVE_RUNS`), split by contents: (train, final)."""
    hashes = frozenset(s.rsplit("/", 1)[-1] for s in sources)
    train: list[str] = []
    final: list[str] = []
    for pattern in ARCHIVE_RUNS:
        for p in sorted(archive.glob(pattern)):
            (final if _mentions_held_out(p, hashes) else train).append(str(p.relative_to(archive)))
    return tuple(train), tuple(final)


def parts(repo: Path = REPO, archive: Path | None = None, split: Split = "train",
          sources: frozenset[str] = frozenset()) -> list[Part]:  # fmt: skip
    data_rest = tuple(sorted(
        str(p.relative_to(repo))
        for p in (repo / "data").iterdir()
        if str(p.relative_to(repo)) not in ("data/tasks", "data/decision", "data/weights", "data/def74", "data/ext", *EXCLUDED_DATA)
    ))  # fmt: skip
    present = lambda paths, base=repo: tuple(r for r in paths if (base / r).exists())
    train_repo = ("data/tasks", "data/decision", *WEIGHTS, *data_rest, *present(TRAIN_DATA), *present(TRAIN_RUNS))
    if archive is not None:
        builds = archived_builds(archive)
        runs_train, runs_final = archived_runs(archive, sources)
        train_archive = (*TRAIN_ARCHIVE, *runs_train)
    if split == "train":
        out = [
            Part("tasks", ("data/tasks",), repo, keep="train"),
            Part("decision", ("data/decision",), repo, keep="train"),
            Part("weights", WEIGHTS, repo, keep="train"),
            Part("data", (*data_rest, *present(TRAIN_DATA)), repo, keep="train"),
            Part("runs", present(TRAIN_RUNS), repo, keep="train"),
        ]
        if archive is not None:
            out.append(Part("archive", train_archive, archive, prefix="archive", keep="train", exclude=builds))
            out.append(Part("builds", builds, archive, prefix="archive", keep="train"))
        return out
    out = [
        Part("held_out", train_repo, repo, keep="held_out"),  # files of the train roots that name a held-out source
        Part("data", present(FINAL_DATA), repo),
        Part("runs", present(FINAL_RUNS), repo),
    ]
    if archive is not None:
        out.append(Part("archive", (*FINAL_ARCHIVE, *runs_final), archive, prefix="archive", exclude=runs_train))
        out.append(Part("archive_held_out", (*train_archive, *builds), archive, prefix="archive", keep="held_out"))
    return out


def part_files(part: Part, sources: frozenset[str]) -> list[Path]:
    files = _files(part.base, part.roots, part.exclude)
    if part.keep == "all":
        return files
    want = part.keep == "held_out"
    return [f for f in files if names_held_out(str(f.relative_to(part.base)), sources) == want]


def held_out_name(rel: str) -> str:
    """Where a mixed file's held-out rows go in the final bundle: data/validation.jsonl -> data/validation.held_out.jsonl."""
    p = Path(rel)
    return str(p.with_name(f"{p.stem}.held_out{p.suffix}"))


def split_rows(path: Path, rel: str, sources: frozenset[str]) -> tuple[bytes, bytes]:
    """A mixed file (`MIXED_FILES`) as (its rows without held-out sources, its held-out rows), each as file bytes."""
    if MIXED_FILES[rel] == "jsonl":
        keep: list[str] = []
        held: list[str] = []
        for line in path.read_text().splitlines(keepends=True):
            if line.strip():
                row = json.loads(line)
                is_held = row.get("pair_id") in sources or any(
                    names_held_out(row.get(k) or "", sources) for k in ("variant_id", "task_id")
                )
                (held if is_held else keep).append(line)
        return "".join(keep).encode(), "".join(held).encode()
    data = json.loads(path.read_text())  # a dict keyed by variant id
    keep_d = {k: v for k, v in data.items() if not names_held_out(k, sources)}
    held_d = {k: v for k, v in data.items() if names_held_out(k, sources)}
    return (json.dumps(keep_d, indent=1).encode() + b"\n", json.dumps(held_d, indent=1).encode() + b"\n")


def pack(out_dir: Path, repo: Path = REPO, archive: Path | None = None, split: Split = "train") -> dict:
    sources = held_out_sources(repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"format": 1, "split": split, "parts": {}}
    plan = [(part, part_files(part, sources)) for part in parts(repo, archive, split, sources)]
    if split == "final":  # the held-out rows of the mixed files, under their own names
        plan.append(
            (Part("held_out_rows", tuple(MIXED_FILES), repo), [repo / r for r in MIXED_FILES if (repo / r).is_file()])
        )
    for part, files in plan:
        if not files:
            continue
        tar_path = out_dir / f"{part.name}.tar.gz"
        hashes: dict[str, str] = {}
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in files:
                arc = str(f.relative_to(part.base))
                if part.base == repo and arc in MIXED_FILES:
                    keep, held = split_rows(f, arc, sources)
                    body, arc = (keep, arc) if split == "train" else (held, held_out_name(arc))
                    info = tar.gettarinfo(str(f), arcname=arc)
                    info.size = len(body)
                    tar.addfile(info, io.BytesIO(body))
                    hashes[arc] = hashlib.sha256(body).hexdigest()
                    continue
                tar.add(f, arcname=arc, recursive=False)
                hashes[arc] = sha256_file(f)
        manifest["parts"][part.name] = {
            "archive": tar_path.name,
            "sha256": sha256_file(tar_path),
            "prefix": part.prefix,
            "files": hashes,
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
    """Extract every part into `dest` after `verify`; returns the number of files written. A file that already exists
    with the same bytes is left alone (a fresh clone tracks some of data/ in git); one with different bytes is a clash,
    and any clash refuses the whole restore before anything is written, unless `force`. Keeps mtimes."""
    problems = verify(bundle)
    if problems:
        raise ValueError("bundle failed verification: " + "; ".join(problems[:5]))
    manifest = json.loads((bundle / MANIFEST).read_text())
    todo: list[tuple[Path, dict, set[str]]] = []
    clashes: list[Path] = []
    for info in manifest["parts"].values():
        root = dest / info["prefix"] if info["prefix"] else dest
        same = set()
        for arc, want in info["files"].items():
            if (root / arc).exists():
                if sha256_file(root / arc) == want:
                    same.add(arc)
                else:
                    clashes.append(root / arc)
        todo.append((root, info, same))
    if clashes and not force:
        raise FileExistsError(f"{len(clashes)} file(s) exist with different contents, e.g. {clashes[0]}; use --force")
    written = 0
    for root, info, same in todo:
        with tarfile.open(bundle / info["archive"], "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name not in same]
            tar.extractall(root, members=members, filter="tar")
        written += len(members)
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


NODE_ROOT = "/root/restraint"  # the checkout path every split file's `task_dir` was written with (node.DEFAULT_DEST)
PREFLIGHT_SPLITS = ("data/select.jsonl", "data/probe_v1*.jsonl", "data/train_rl*.jsonl",
                    "data/benchmarks/internal_heldout.jsonl")  # fmt: skip


def preflight(repo: Path = REPO) -> tuple[list[str], list[str]]:
    """Whether this checkout can run what its bundles hold: per split file, how many `task_dir`s resolve. Returns
    (report lines, problems). The split files store absolute NODE_ROOT paths, so each row is also tried rebased onto
    this checkout: a row found only that way means the checkout is in the wrong place, one found neither way means
    missing data. A held-out row may be missing (a train-bundle checkout); any other missing row is a problem."""
    sources = held_out_sources(repo)
    lines: list[str] = []
    problems: list[str] = []
    held = {"present": 0, "missing": 0}
    misplaced = 0
    for pattern in PREFLIGHT_SPLITS:
        for split in sorted(repo.glob(pattern)):
            n = missing = 0
            for line in split.read_text().splitlines():
                if not line.strip():
                    continue
                td = Path(json.loads(line)["task_dir"])
                td = td if td.is_absolute() else split.parent / td
                ok = (td / "task.toml").is_file()
                if (
                    not ok
                    and td.is_relative_to(NODE_ROOT)
                    and (repo / td.relative_to(NODE_ROOT) / "task.toml").is_file()
                ):
                    misplaced += 1
                    ok = True
                if names_held_out(td.name, sources):
                    held["present" if ok else "missing"] += 1
                    continue
                n += 1
                missing += not ok
            lines.append(f"{split.relative_to(repo)}: {n - missing} of {n} task dirs present (held-out rows aside)")
            if missing:
                problems.append(f"{split.relative_to(repo)}: {missing} of {n} non-held-out task dirs missing")
    if misplaced:
        problems.append(f"{misplaced} task dirs exist only under this checkout ({repo.resolve()}), and the split files "
                        f"name them as {NODE_ROOT}/...: clone to {NODE_ROOT}, or `ln -s {repo.resolve()} {NODE_ROOT}`")  # fmt: skip
    if held["present"] and held["missing"]:
        problems.append(
            f"held-out task dirs: {held['present']} present, {held['missing']} missing (partial final bundle)"
        )
    lines.append(
        "held-out task dirs: "
        + ("all present (evaluator checkout)" if not held["missing"] else "absent (train bundle only)")
    )
    for w in WEIGHTS:
        if not (repo / w / "adapter_model.safetensors").is_file():
            problems.append(f"{w}: adapter missing (train bundle, part weights)")
    return lines, problems


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
    c = sub.add_parser("preflight", help="check that this checkout's split files resolve (run after fetch)")
    c.add_argument("--repo", type=Path, default=REPO)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        lines, problems = preflight(a.repo)
        print("\n".join([*lines, *(f"PROBLEM: {p}" for p in problems)] or ["ok"]))
        return 1 if problems else 0
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
