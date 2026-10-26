"""RTS source benchmarks (spec sections 3.2 and 4, item 1): one `BenchmarkCard` per Harbor adapter dataset, and the loader
that turns a pinned dataset into feasible seed tasks (`RTSSeed`) our harness and grader run.

Every candidate is a standard Harbor task dir (task.toml, instruction.md, environment/, tests/test.sh, solution/solve.sh),
so one normaliser serves them all; what differs per benchmark is a named fix-up (`FIXUPS`) for an oracle that ships
inside the agent's image or a deliverable path the instruction misstates. Layouts were read off the downloaded datasets
on 2026-10-26 (`harbor datasets download <name>@<version>`, harbor 0.24.0; BIRD from the Hub tree of one task):

* humanevalfix, aider-polyglot, deveval and terminal-bench stamp the Terminal-Bench canary line into most files; the
  others carry none. The canary is a comment line and is stripped (a canary outside a comment rejects the task): an eval
  task must not tell the policy it is a benchmark, and the freeze checks that none is left.
* aider-polyglot ships its oracle as `environment/workspace/.oracle/solution.enc` (AES under the fixed password
  `tb-secret-please-change`), i.e. inside the image; deveval copies the full source repository (the solution) to
  `/deveval_source_repo` in the image. Both are moved into `solution/`, which only the oracle sees.
* ds-1000 and terminal-bench name a prebuilt `docker_image`; Harbor then ignores the Dockerfile and image-changing
  operators cannot apply (`TaskFacts.dockerfile_editable`), so the image is built from the Dockerfile instead (the
  dropped name is kept in provenance). ds-1000's instruction says `solution/solution.py` and "run bash tests/test.sh"
  while its test reads `/solution/solution.py` and the tests are not in the image; both are corrected.
* Grading keeps each benchmark's own `tests/test.sh` (the legacy verifier, evaluation only) unless the card asks for the
  offline pytest verifier and the task converts (`tasks.hermetic`); a task that does not convert keeps its own.

Network yes (the download), Docker no.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.spec import VERIFIER_LEGACY, HonestTaskSpec
from honest_agents.tasks.dedup import CANARY_RE
from honest_agents.tasks.sources import SeedReject, _migrate_config, sample_ids, seed_leak_problems
from honest_agents.tasks.taskfs import (
    copy_task,
    normalize_heredocs,
    read_task_toml,
    scrub_environment,
    set_honest,
    set_task_name,
    slug,
    write_task_toml,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE = Path("~/.cache/honest-agents/rts/seeds").expanduser()
INDEX_NAME = "index.jsonl"
REJECTED_NAME = "_rejected.jsonl"
_DOWNLOADED = ".downloaded"  # marker in a dataset's raw dir: the download finished (a partial one is redone)
_UNKNOWN = "unknown"


@dataclass(frozen=True)
class BenchmarkCard:
    name: str
    harbor_dataset: str  # pinned `name@version`
    domain: str
    license: str  # "unknown: ..." when no licence is stated for the data
    license_basis: str  # where the licence was read, so a reader can check it
    cpu_only: bool
    note: str = ""
    registry_path: str | None = None  # repo-relative registry.json to resolve the dataset from (default: Harbor's)
    hermetic: bool = False  # convert to the offline pytest verifier where the task allows it
    fixups: tuple[str, ...] = ()  # `FIXUPS` names applied after the copy

    @property
    def license_known(self) -> bool:
        return not self.license.lower().startswith(_UNKNOWN)

    @property
    def dataset_name(self) -> str:
        return self.harbor_dataset.split("@", 1)[0]


_READ = "read 2026-10-26"
CARDS: dict[str, BenchmarkCard] = {
    c.name: c
    for c in (
        BenchmarkCard(
            "humanevalfix",
            "humanevalfix@1.0",
            "bug fixing (Python)",
            "MIT",
            f"HF dataset card bigcode/humanevalpack (`license: mit`), {_READ}; Harbor adapter code Apache-2.0",
            cpu_only=True,
            note="164 tasks, python:3.12-bookworm; the buggy function is /workspace/Python__<n>.py.",
        ),
        BenchmarkCard(
            "aider-polyglot",
            "aider-polyglot@1.0",
            "bug fixing / code editing (6 languages)",
            "MIT",
            "Harbor adapter README (adapters/aider_polyglot: 'Aider Polyglot Benchmark repository (MIT License) ... "
            f"inherit upstream licensing from Exercism (MIT)'), {_READ}; the GitHub API reports no licence file on "
            "Aider-AI/polyglot-benchmark, exercism tracks are MIT",
            cpu_only=True,
            note="225 tasks; verifiers install packages at test time (npm, gradle, cargo), so the verifier needs the "
            "network. Oracle moved out of the image (`aider_oracle`).",
            fixups=("aider_oracle",),
        ),
        BenchmarkCard(
            "bigcodebench-hard-complete",
            "bigcodebench-hard-complete@1.0.0",
            "library use (Python)",
            "Apache-2.0",
            "Harbor adapter README (adapters/bigcodebench_hard, 'BigCodeBench dataset is licensed under Apache 2.0') and "
            f"HF dataset card bigcode/bigcodebench (`license: apache-2.0`), {_READ}",
            cpu_only=True,
            note="145 tasks, python:3.10-slim with a large pinned library stack (mandated libraries).",
        ),
        BenchmarkCard(
            "ds-1000",
            "ds-1000@head",
            "library use (data science, Python)",
            "CC-BY-SA-4.0",
            f"GitHub API licence of xlang-ai/DS-1000 and HF dataset card xlangai/DS-1000 (`cc-by-sa-4.0`), {_READ}",
            cpu_only=True,
            note="1000 tasks on one image (CPU torch and tensorflow-cpu: a large build). `@head` is what the registry "
            "lists; its tasks are pinned to harbor-datasets commit c3fe2cf. Deliverable path and prebuilt image fixed "
            "(`ds1000_paths`).",
            fixups=("ds1000_paths",),
        ),
        BenchmarkCard(
            "bird-bench",
            "bird-bench@parity",
            "SQL (SQLite)",
            "CC-BY-SA-4.0",
            f"HF dataset card birdsql/bird_sql_dev_20251106 (`cc-by-sa-4.0`), {_READ}; the Harbor re-upload "
            "harborframework/harbor-datasets is apache-2.0 (code)",
            cpu_only=True,
            note="150-task parity subset, one SQLite file per task in the image (about 20 GB in all, git LFS from the "
            "Hub); layout verified on one task from the Hub tree, not on a full download.",
        ),
        BenchmarkCard(
            "deveval",
            "deveval@1.0",
            "repository development (Python, C++, Java, JavaScript)",
            "Apache-2.0",
            f"GitHub API licence of open-compass/DevEval (the adapter README's provenance), {_READ}",
            cpu_only=True,
            note="63 tasks on ghcr.io/laude-institute/t-bench/deveval:latest (an unpinned tag); TextCNN (GPU) is "
            "excluded upstream. Source repository moved out of the image (`deveval_oracle`).",
            fixups=("deveval_oracle",),
        ),
        BenchmarkCard(
            "compilebench",
            "compilebench@1.0",
            "build (C/C++ toolchains)",
            "MIT",
            f"GitHub API licence of QuesmaOrg/CompileBench, {_READ}",
            cpu_only=True,
            note="15 tasks, x86 images (`--platform=linux/amd64`); verifiers apt-get at test time.",
        ),
        BenchmarkCard(
            "terminal-bench",
            "terminal-bench@2.0",
            "shell, ops",
            "Apache-2.0",
            f"GitHub API licence of laude-institute/terminal-bench-2, {_READ}",
            cpu_only=True,
            note="89 tasks at terminal-bench-2 commit 69671fb (data/benchmarks/registry_tb2.json); x86 images. Tests "
            "install uv and pytest at test time, so the offline pytest verifier is tried first.",
            registry_path="data/benchmarks/registry_tb2.json",
            hermetic=True,
        ),
    )
}


@dataclass(frozen=True)
class RTSSeed:
    benchmark: str
    source_task: str  # the cluster id, `<benchmark>/<task name>`
    task_dir: Path  # normalised, feasible, with its oracle when `has_oracle`
    has_oracle: bool

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "task_dir": str(self.task_dir)}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RTSSeed:
        return cls(d["benchmark"], d["source_task"], Path(d["task_dir"]), bool(d["has_oracle"]))


# ── fix-ups: the per-benchmark code ─────────────────────────────────────────


class FixupError(ValueError):
    """The task does not have the layout its benchmark's fix-up was written for: it is rejected, never half-fixed."""


def _replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise FixupError(f"{path.name}: expected {old!r}")
    path.write_text(text.replace(old, new))


def _aider_oracle(task_dir: Path) -> None:
    enc = task_dir / "environment" / "workspace" / ".oracle" / "solution.enc"
    if not enc.is_file():
        raise FixupError("no environment/workspace/.oracle/solution.enc")
    (task_dir / "solution").mkdir(exist_ok=True)
    shutil.move(enc, task_dir / "solution" / "solution.enc")
    shutil.rmtree(enc.parent)
    _replace_once(
        task_dir / "solution" / "solve.sh",
        'ENC_FILE="$ORACLE_DIR/solution.enc"',
        'ENC_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/solution.enc"',
    )


_DEVEVAL_COPY = re.compile(r'^\s*COPY\s+"?deveval_source_repo/"?\s+"?/deveval_source_repo/"?\s*$\n?', re.MULTILINE)


def _deveval_oracle(task_dir: Path) -> None:
    df = task_dir / "environment" / "Dockerfile"
    repo = task_dir / "environment" / "deveval_source_repo"
    text = df.read_text() if df.is_file() else ""
    if not repo.is_dir() or not _DEVEVAL_COPY.search(text):
        raise FixupError("no deveval_source_repo COPY to /deveval_source_repo")
    df.write_text(_DEVEVAL_COPY.sub("", text))  # /app still gets its copy, minus the files the task removes
    shutil.copytree(repo, task_dir / "solution" / "deveval_source_repo", symlinks=True)
    _replace_once(task_dir / "solution" / "solve.sh", "/deveval_source_repo/", "/solution/deveval_source_repo/")


_DS1000_TEST_LINE = re.compile(r"^.*bash tests/test\.sh.*$\n?", re.MULTILINE)


def _ds1000_paths(task_dir: Path) -> None:
    inst = task_dir / "instruction.md"
    _replace_once(inst, "`solution/solution.py`", "`/solution/solution.py`")
    inst.write_text(_DS1000_TEST_LINE.sub("", inst.read_text()))


FIXUPS: dict[str, Callable[[Path], None]] = {
    "aider_oracle": _aider_oracle,
    "deveval_oracle": _deveval_oracle,
    "ds1000_paths": _ds1000_paths,
}


# ── canary ───────────────────────────────────────────────────────────────────

_COMMENT = re.compile(r"^\s*(?:#|//|--|;|<!--|/\*|\*|rem\b)", re.IGNORECASE)


def strip_canary(task_dir: Path, *, max_bytes: int = 4_000_000) -> int:
    """Delete the Terminal-Bench canary comment lines from every text file; returns how many went. Raises `FixupError`
    if a canary sits outside a comment (deleting it would change what the file means)."""
    n = 0
    for f in sorted(task_dir.rglob("*")):
        if f.is_symlink() or not f.is_file() or f.stat().st_size > max_bytes:
            continue
        raw = f.read_bytes()
        if b"\0" in raw:
            continue
        text = raw.decode("utf-8", errors="surrogateescape")
        if not CANARY_RE.search(text):
            continue
        kept = []
        for line in text.splitlines(keepends=True):
            if CANARY_RE.search(line):
                if not _COMMENT.match(line) and f.name != "instruction.md":
                    raise FixupError(f"{f.relative_to(task_dir)}: canary outside a comment line")
                n += 1
                continue
            kept.append(line)
        f.write_bytes("".join(kept).encode("utf-8", errors="surrogateescape"))
    return n


# ── normalisation ────────────────────────────────────────────────────────────


def normalize_seed(
    raw_task_dir: Path, out_dir: Path, card: BenchmarkCard, task: str
) -> tuple[RTSSeed | None, str | None]:
    """Copy one downloaded task into a clean, loadable task dir. Returns (seed, None), or (None, why it was dropped)."""
    from harbor.models.task.task import Task

    if not (raw_task_dir / "task.toml").is_file() or not (raw_task_dir / "instruction.md").is_file():
        return None, "missing task.toml or instruction.md"
    source_task = f"{card.name}/{task}"
    dst = copy_task(raw_task_dir, out_dir)
    try:
        return _normalize(dst, card, task, source_task, Task), None
    except (FixupError, _Reject) as e:
        shutil.rmtree(dst, ignore_errors=True)
        return None, str(e)


class _Reject(ValueError):
    pass


def _normalize(dst: Path, card: BenchmarkCard, task: str, source_task: str, task_cls: Any) -> RTSSeed:
    n_canary = strip_canary(dst)
    for name in card.fixups:
        FIXUPS[name](dst)
    if problems := seed_leak_problems(dst):
        raise _Reject("leak screen: " + "; ".join(problems))
    scrub_environment(dst)
    df = dst / "environment" / "Dockerfile"
    if df.is_file():
        df.write_text(normalize_heredocs(df.read_text()))
    try:
        cfg = _migrate_config(read_task_toml(dst))
    except Exception as e:
        raise _Reject(f"unreadable task.toml ({e})") from e
    env = dict(cfg.get("environment") or {})
    dropped_image = env.pop("docker_image", None) if df.is_file() else None
    cfg["environment"] = env
    # Harbor's `[task]` table is the package name; adapters that use it for other keys (ds-1000) would not load.
    old_task = dict(cfg.get("task") or {})
    if isinstance(old_task.get("max_time_seconds"), int | float) and "timeout_sec" not in (cfg.get("agent") or {}):
        cfg["agent"] = {**dict(cfg.get("agent") or {}), "timeout_sec": float(old_task["max_time_seconds"])}
    cfg = set_task_name({k: v for k, v in cfg.items() if k != "task"}, f"rts-{slug(card.name)}/{slug(task)}")
    md = dict(cfg.get("metadata") or {})
    md["provenance"] = {
        "source": card.name,
        "task_id": task,
        "pair_id": source_task,
        "harbor_dataset": card.harbor_dataset,
        "license": card.license,
        "license_basis": card.license_basis,
        "canary_lines_stripped": n_canary,
        "fixups": list(card.fixups),
        **({"docker_image_dropped": dropped_image} if dropped_image else {}),
    }
    cfg["metadata"] = md
    spec = HonestTaskSpec(feasible=True, pair_id=source_task, source_task=source_task, verifier=VERIFIER_LEGACY)
    write_task_toml(dst, set_honest(cfg, spec))
    try:
        task_cls(dst)
    except Exception as e:
        raise _Reject(f"not a valid Harbor task ({e})") from e
    if card.hermetic:
        _try_hermetic(dst, task_cls)
    return RTSSeed(card.name, source_task, dst, (dst / "solution" / "solve.sh").is_file())


def _try_hermetic(dst: Path, task_cls: Any) -> None:
    """Convert to the offline pytest verifier on a scratch copy; keep the task's own test.sh if that fails."""
    from honest_agents.tasks.hermetic import to_pytest_verifier

    with tempfile.TemporaryDirectory(prefix=".hermetic-", dir=dst.parent) as tmp:
        trial = copy_task(dst, Path(tmp) / dst.name)
        rep = to_pytest_verifier(trial)
        why = rep.reason if not rep.ok else None
        if why is None:
            try:
                task_cls(trial)
            except Exception as e:  # noqa: BLE001
                why = f"invalid after conversion: {e}"
        if why is None:
            shutil.rmtree(dst)
            shutil.move(trial, dst)
            return
    cfg = read_task_toml(dst)
    md = dict(cfg.get("metadata") or {})
    md["provenance"] = {**dict(md.get("provenance") or {}), "hermetic_refused": str(why)[:300]}
    write_task_toml(dst, {**cfg, "metadata": md})


# ── download and load ────────────────────────────────────────────────────────


def _harbor_cmd() -> list[str]:
    exe = Path(sys.executable).parent / "harbor"
    return [str(exe)] if exe.exists() else ["harbor"]


def harbor_download(card: BenchmarkCard, raw_dir: Path) -> None:
    """`harbor datasets download <pinned>` into `raw_dir` (export layout), once; a finished download is reused."""
    if (raw_dir / _DOWNLOADED).is_file():
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*_harbor_cmd(), "datasets", "download", card.harbor_dataset, "--output-dir", str(raw_dir), "--overwrite"]
    if card.registry_path:
        cmd += ["--registry-path", str(REPO_ROOT / card.registry_path)]
    log.info("downloading %s: %s", card.harbor_dataset, " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=6 * 3600)
    (raw_dir / _DOWNLOADED).write_text(card.harbor_dataset + "\n")


def raw_tasks(raw_dir: Path) -> dict[str, Path]:
    """Task name (its dir name) -> raw task dir, for every task.toml under the download."""
    out: dict[str, Path] = {}
    for toml_path in sorted(raw_dir.rglob("task.toml")):
        d = toml_path.parent
        if d.name in out:
            raise ValueError(f"{raw_dir}: two tasks named {d.name!r} ({out[d.name]}, {d})")
        out[d.name] = d
    return out


def load_seeds(
    card: BenchmarkCard,
    cache_root: Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    download: Callable[[BenchmarkCard, Path], None] | None = None,  # default: `harbor_download`
    rejected: list[SeedReject] | None = None,
) -> list[RTSSeed]:
    """Download (once) and normalise a pinned dataset into `cache_root/<benchmark>/<task>/`.

    `limit` draws a seeded sample of task names. Dropped tasks are appended to `rejected` and written, with their
    reasons, to `cache_root/<benchmark>/_rejected.jsonl`."""
    raw_dir = cache_root / "raw" / slug(card.harbor_dataset)
    (download or harbor_download)(card, raw_dir)
    tasks = raw_tasks(raw_dir)
    names = sample_ids(sorted(tasks), limit, seed)
    out_root = cache_root / slug(card.name)
    seeds: list[RTSSeed] = []
    rejects: list[SeedReject] = []
    for name in names:
        s, why = normalize_seed(tasks[name], out_root / slug(name), card, name)
        if s is None:
            log.warning("skip %s/%s: %s", card.name, name, why)
            rejects.append(SeedReject(card.name, name, str(why)))
        else:
            seeds.append(s)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / REJECTED_NAME).write_text("".join(json.dumps(asdict(r)) + "\n" for r in rejects))
    if rejected is not None:
        rejected.extend(rejects)
    return seeds


def write_index(seeds: list[RTSSeed], path: Path) -> None:
    """Rewrite the seeds index: rows of the benchmarks in `seeds` are replaced, other benchmarks' rows are kept."""
    fresh = {s.benchmark for s in seeds}
    kept = [s for s in read_index(path) if s.benchmark not in fresh] if path.is_file() else []
    rows = sorted([*kept, *seeds], key=lambda s: s.source_task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(s.to_json(), sort_keys=True) + "\n" for s in rows))


def read_index(path: Path) -> list[RTSSeed]:
    return [RTSSeed.from_json(json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()]
