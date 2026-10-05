"""Feasible seed-task sources: download from the Hugging Face Hub and normalise into Harbor 0.24 task dirs.

Repo layouts (inspected 2026-10-04 via the Hub tree API):

* `obiwan96/endless-terminals` — one Harbor-style dir per task at the repo root (`task_000000_0033979a/`), with
  `solution/solve.sh`, `tests/test.sh` and model trajectories (`solution/*_summary.json`, dropped). task.toml uses the
  legacy `version = "0.1"` key. Measured on 40 random tasks (2026-10-04): every `test.sh` is the same boilerplate that
  needs the internet (`apt-get`, the uv installer, `uvx --python 3.12 --with pytest==8.4.1`), every image is
  `ubuntu:22.04` with python3 and a pip-installed pytest, every task has one `tests/test_final_state.py` (about 5 tests)
  importing only the standard library and pytest. This is the first source and is converted to the hermetic pytest
  verifier on normalisation (`honest_agents.tasks.hermetic`); the other sources stay available but are not hermetic by
  default.
* `camel-ai/SETA-Env` — `SETA_Synth/<task>/` (3255) and `SETA_Evolve/<task>__{b1,d1}/` (1312, evolved variants of
  Synth tasks: same split group). Harbor-style, with `solution/solve.sh`; legacy `memory = "2G"` fields; `.DS_Store`.
* `TMaxxx/TMax-15K-Harbor` — a single `tasks.zip` (14,601 Harbor dirs, no solutions, `allow_internet = true`) plus a
  metadata parquet. Members are read with ranged HTTP requests, so sampling a few tasks does not fetch the archive.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import zipfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.tasks.dedup import canary_files
from honest_agents.tasks.hermetic import is_hermetic, to_pytest_verifier
from honest_agents.tasks.taskfs import (
    copy_task,
    forbidden_context_files,
    forbidden_copy_sources,
    normalize_heredocs,
    read_task_toml,
    scrub_environment,
    set_task_name,
    slug,
    write_task_toml,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceInfo:
    name: str
    repo_id: str
    license: str
    has_solutions: bool
    note: str = ""
    hermetic_default: bool = False  # convert to the offline pytest verifier on normalisation
    enabled: bool = (
        False  # selectable without `--allow-disabled-sources` (a source must clear the decontamination audit)
    )
    license_known: bool = True  # False: no licence is stated for the data; a build refuses it without a flag
    license_basis: str = ""  # where the licence was read, so a reader can check it
    code_license: str = ""  # licence of the generating code repository, when it differs from the data's


SOURCES: dict[str, SourceInfo] = {
    "endless-terminals": SourceInfo(
        "endless-terminals",
        "obiwan96/endless-terminals",
        "MIT",
        True,
        "The Hub API returns license None because the README front-matter (`license: mit`) sits after the body; the "
        "tasks are LLM-generated, so the generating model's terms also apply.",
        hermetic_default=True,
        enabled=True,
        license_basis="dataset card README front-matter (`license: mit`), read 2026-10-05; Hub API cardData is None",
        code_license="Apache-2.0 (github.com/kanishkg/endless-terminals, the paper's generator)",
    ),
    "seta-env": SourceInfo(
        "seta-env",
        "camel-ai/SETA-Env",
        "Apache-2.0",
        True,
        "DISABLED by default (leak audit L4): some SETA_Synth files carry the terminal-bench canary GUID; such seeds are "
        "rejected, and the source needs `--allow-disabled-sources` plus a clean benchmark overlap report.",
        license_basis="dataset card (not re-read for this record)",
    ),
    "tmax-15k": SourceInfo(
        "tmax-15k",
        "TMaxxx/TMax-15K-Harbor",
        "unspecified (upstream allenai/TMax-15K is ODC-BY-1.0)",
        False,
        "DISABLED by default (leak audit L4): a task ships `_perturbation.json` (the expected output) into its image via "
        "COPY, which the seed screen now rejects. No oracle solutions: feasible-only pool, usable only after a pass@k "
        "filter (allowlist).",
        license_known=False,
        license_basis="none stated on the Harbor re-upload; the upstream card says ODC-BY-1.0, which a re-upload may not carry",
    ),
}

_TMAX_ZIP = "tasks.zip"
_SETA_SUBSETS = ("SETA_Synth", "SETA_Evolve")


@dataclass(frozen=True)
class SeedTask:
    source: str
    task_id: str  # id within the source (e.g. `SETA_Synth/ask_ubuntu__0`)
    task_dir: Path  # normalised Harbor task dir
    has_solution: bool
    license: str
    hermetic: bool = False  # graded by the offline pytest verifier (`[metadata.honest] verifier = "pytest"`)

    @property
    def pair_id(self) -> str:
        return f"{self.source}/{self.task_id}"

    @property
    def split_group(self) -> str:
        """Tasks that must land in the same split: SETA_Evolve variants go with their SETA_Synth parent."""
        if self.source == "seta-env":
            return f"seta-env/{re.sub(r'__[bd]\d+$', '', self.task_id.split('/')[-1])}"
        return self.pair_id


# ── listing & download ──────────────────────────────────────────────────────


def list_task_ids(source: str) -> list[str]:
    """All task ids of a source, from the Hub tree API (no file downloads)."""
    from huggingface_hub import HfApi, RepoFolder

    info = SOURCES[source]
    api = HfApi()
    if source == "endless-terminals":
        tree = api.list_repo_tree(info.repo_id, repo_type="dataset", recursive=False)
        return sorted(f.path for f in tree if isinstance(f, RepoFolder) and f.path.startswith("task_"))
    if source == "seta-env":
        ids: list[str] = []
        for subset in _SETA_SUBSETS:
            tree = api.list_repo_tree(info.repo_id, path_in_repo=subset, repo_type="dataset", recursive=False)
            ids += [f.path for f in tree if isinstance(f, RepoFolder)]
        return sorted(ids)
    if source == "tmax-15k":
        with _open_tmax_zip() as z:
            return sorted({n.split("/")[0] for n in z.namelist() if "/" in n})
    raise KeyError(source)


def _open_tmax_zip() -> zipfile.ZipFile:
    from huggingface_hub import HfFileSystem

    f = HfFileSystem().open(f"datasets/{SOURCES['tmax-15k'].repo_id}/{_TMAX_ZIP}", "rb", block_size=1 << 16)
    return zipfile.ZipFile(f)


def sample_ids(ids: list[str], limit: int | None, seed: int) -> list[str]:
    if limit is None or limit >= len(ids):
        return list(ids)
    return sorted(random.Random(seed).sample(ids, limit))


REVISION_FILE = ".revision"  # in a source's raw dir: the dataset git revision it was fetched at, when known
# A pool this large is fetched with one shallow clone: the per-file Hub API rate-limits at about 20k requests.
_GIT_MIN_TASKS = 300


def _git_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _is_lfs_pointer(path: Path) -> bool:
    return path.read_bytes()[:40].startswith(b"version https://git-lfs.github.com/spec")


def _download_git(repo_id: str, raw_dir: Path, todo: list[str]) -> bool:
    """Fetch `todo` from one shallow `git clone` of the dataset repository (about 5 s for all 2,492 Endless Terminals tasks,
    against many minutes of rate-limited per-file requests). Model trajectories and `.DS_Store` files are dropped, as in
    the per-file path. Returns False, leaving `raw_dir` as it was, when git is missing or the clone fails, so the caller
    falls back to the Hub API."""
    import subprocess
    import tempfile

    if shutil.which("git") is None:
        return False
    with tempfile.TemporaryDirectory(prefix=".clone-", dir=raw_dir) as tmp:
        clone = Path(tmp) / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", _git_url(repo_id), str(clone)],
                check=True,
                capture_output=True,
                env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
                timeout=1800,
            )
        except (subprocess.SubprocessError, OSError) as e:
            log.warning("git clone of %s failed (%s): falling back to the Hub API", repo_id, e)
            return False
        rev = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        log.info("fetched %d tasks of %s from one git clone (revision %s)", len(todo), repo_id, rev or "unknown")
        if rev:
            (raw_dir / REVISION_FILE).write_text(rev + "\n")  # the manifest records which dataset revision was built
        # GIT_LFS_SKIP_SMUDGE leaves a large file as a three-line pointer: a task holding one would be built from the pointer
        # text. The Hub API path resolves it, so fall back to it rather than copy a corrupt task.
        for t in todo:
            for f in (clone / t).rglob("*") if (clone / t).is_dir() else ():
                if f.is_file() and not f.is_symlink() and f.stat().st_size < 1024 and _is_lfs_pointer(f):
                    log.warning("git clone of %s holds a Git LFS pointer (%s): falling back to the Hub API", repo_id, f)
                    return False
        junk = shutil.ignore_patterns("*_summary.json", ".DS_Store")
        for t in todo:
            if (clone / t / "task.toml").exists():
                shutil.copytree(clone / t, raw_dir / t, ignore=junk, dirs_exist_ok=True, symlinks=True)
    return True


def download(source: str, raw_dir: Path, task_ids: list[str], *, chunk: int = 200) -> None:
    """Fetch only the given tasks into `raw_dir/<task_id>/` (skipping ones already present)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in task_ids if not (raw_dir / t / "task.toml").exists()]
    if not todo:
        return
    info = SOURCES[source]
    if source == "tmax-15k":
        wanted = set(todo)
        with _open_tmax_zip() as z:
            members = [n for n in z.namelist() if n.split("/")[0] in wanted and not n.endswith("/")]
            for n in members:
                target = (raw_dir / n).resolve()
                if not target.is_relative_to(raw_dir.resolve()):
                    raise ValueError(f"unsafe zip member {n}")
                z.extract(n, raw_dir)
        return
    if len(todo) >= _GIT_MIN_TASKS and source != "seta-env" and _download_git(info.repo_id, raw_dir, todo):
        return
    from huggingface_hub import snapshot_download

    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        snapshot_download(
            info.repo_id,
            repo_type="dataset",
            local_dir=raw_dir,
            allow_patterns=[f"{t}/**" for t in batch],
            ignore_patterns=["*/.DS_Store", "*_summary.json"],
        )


# ── normalisation ────────────────────────────────────────────────────────────


def _migrate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Rewrite deprecated Harbor fields so the task loads without deprecation warnings (same semantics)."""
    cfg = dict(cfg)
    if "version" in cfg and "schema_version" not in cfg:
        cfg["schema_version"] = str(cfg.pop("version"))
    env = dict(cfg.get("environment") or {})
    for old, new in (("memory", "memory_mb"), ("storage", "storage_mb")):
        if old in env:
            mb = _size_to_mb(env[old]) if isinstance(env[old], str) else None
            if mb is None:
                continue  # leave it for Harbor to accept or reject (a rejected seed is logged and dropped)
            env.pop(old)
            env.setdefault(new, mb)
    if "allow_internet" in env:
        allow = env.pop("allow_internet")
        if allow is not None and "network_mode" not in env:
            env["network_mode"] = "public" if allow else "no-network"
    cfg["environment"] = env
    return cfg


def _size_to_mb(s: str) -> int | None:
    """Harbor's own `1G`/`512M`/`64K` rule; None for anything it would not parse (e.g. a bare `2048`)."""
    from harbor.models.task.config import EnvironmentConfig

    try:
        return EnvironmentConfig._parse_size_to_mb(s)
    except ValueError:
        return None


@dataclass(frozen=True)
class SeedReject:
    """A seed task that was dropped, and why (hermetic conversion refusals are listed, never silently skipped)."""

    source: str
    task_id: str
    reason: str


def seed_leak_problems(raw_task_dir: Path) -> list[str]:
    """Why a raw task must not become a seed: benchmark canary text anywhere in it, or a Dockerfile that COPY/ADDs tests,
    expected files, solutions or task.json into the image (the agent could read them). Empty = clean."""
    problems = [f"terminal-bench canary in {f}" for f in canary_files(raw_task_dir)]
    env = raw_task_dir / "environment"
    for df in sorted(env.rglob("Dockerfile*")) if env.is_dir() else []:
        bad = forbidden_copy_sources(df.read_text(errors="replace"))
        if bad:
            problems.append(f"{df.relative_to(raw_task_dir)} COPY/ADDs {sorted(set(bad))}")
        if ctx := forbidden_context_files(env, df.read_text(errors="replace")):
            problems.append(f"{df.relative_to(raw_task_dir)} copies the whole context, which holds {ctx[:5]}")
    return problems


def normalize_task(
    raw_task_dir: Path,
    out_dir: Path,
    *,
    source: str,
    task_id: str,
    hermetic: bool | None = None,
    rejected: list[SeedReject] | None = None,
) -> SeedTask | None:
    """Copy one raw task into a clean Harbor task dir; returns None (logged) if it is not a usable task.

    `hermetic` converts the task to the offline pytest verifier (default: the source's `hermetic_default`). A task that
    cannot be converted is dropped, because a non-hermetic seed must not reach training; the reason is appended to
    `rejected`.
    """
    from harbor.models.task.task import Task, strip_canary

    info = SOURCES.get(source)
    if not (raw_task_dir / "task.toml").exists() or not (raw_task_dir / "instruction.md").exists():
        log.warning("skip %s/%s: missing task.toml or instruction.md", source, task_id)
        return None
    if problems := seed_leak_problems(raw_task_dir):
        log.warning("skip %s/%s: %s", source, task_id, "; ".join(problems))
        if rejected is not None:
            rejected.append(SeedReject(source, task_id, "leak screen: " + "; ".join(problems)))
        return None
    dst = copy_task(raw_task_dir, out_dir)
    scrub_environment(dst)  # the expected state must never sit in the docker build context
    df = dst / "environment" / "Dockerfile"
    if df.exists():
        df.write_text(normalize_heredocs(df.read_text()))
    for junk in (dst / "solution").glob("*_summary.json"):  # Endless Terminals model trajectories
        junk.unlink()
    # Harbor strips leading canary lines only via `Task.instruction`; `HonestHarborEnv.reset` reads the file directly.
    inst = dst / "instruction.md"
    raw_inst = inst.read_text()
    if (clean := strip_canary(raw_inst)) != raw_inst:
        inst.write_text(clean)
    try:
        cfg = _migrate_config(read_task_toml(dst))
    except Exception as e:  # noqa: BLE001 — malformed upstream task.toml
        log.warning("skip %s/%s: unreadable task.toml (%s)", source, task_id, e)
        shutil.rmtree(dst, ignore_errors=True)
        return None
    pair_id = f"{source}/{task_id}"
    cfg = set_task_name(cfg, f"{slug(source)}/{slug(task_id)}")
    md = dict(cfg.get("metadata") or {})
    md["provenance"] = {
        "source": source,
        "task_id": task_id,
        "pair_id": pair_id,
        "repo_id": info.repo_id if info else None,
        "license": info.license if info else "unknown",
    }
    cfg["metadata"] = md
    write_task_toml(dst, cfg)
    try:
        Task(dst)
    except Exception as e:  # noqa: BLE001 — record and drop malformed upstream tasks
        log.warning("skip %s: not a valid Harbor task (%s)", pair_id, e)
        shutil.rmtree(dst, ignore_errors=True)
        return None
    if hermetic is None:
        hermetic = bool(info and info.hermetic_default)
    if hermetic:
        rep = to_pytest_verifier(dst)
        if not rep.ok:
            log.warning(
                "skip %s: not convertible to the hermetic verifier (%s)",
                pair_id,
                rep.reason,
            )
            if rejected is not None:
                rejected.append(SeedReject(source, task_id, rep.reason))
            shutil.rmtree(dst, ignore_errors=True)
            return None
        try:
            Task(dst)  # the conversion adds `[verifier] network_mode`: make sure Harbor still loads it
        except Exception as e:  # noqa: BLE001
            log.warning("skip %s: converted task is not a valid Harbor task (%s)", pair_id, e)
            if rejected is not None:
                rejected.append(SeedReject(source, task_id, f"invalid after conversion: {e}"))
            shutil.rmtree(dst, ignore_errors=True)
            return None
    has_solution = (dst / "solution" / "solve.sh").exists()
    return SeedTask(
        source,
        task_id,
        dst,
        has_solution,
        info.license if info else "unknown",
        hermetic,
    )


def prepare_source(
    source: str,
    raw_root: Path,
    seeds_root: Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    task_ids: list[str] | None = None,
    download_missing: bool = True,
    hermetic: bool | None = None,
    rejected: list[SeedReject] | None = None,
    allow_disabled: bool = False,
) -> list[SeedTask]:
    """List (or reuse) -> sample -> download -> normalise one Hub source into `seeds_root/<source>/<slug>/`.

    A source whose `SourceInfo.enabled` is False (SETA, tmax) is refused unless `allow_disabled`.

    Tasks dropped by the hermetic conversion are appended to `rejected` and written to
    `seeds_root/<source>/_rejected.jsonl`.
    """
    if not SOURCES[source].enabled and not allow_disabled:
        raise ValueError(
            f"source {source!r} is disabled by default (see SOURCES[{source!r}].note); pass allow_disabled"
        )
    raw_dir = raw_root / source
    if task_ids is None:
        if download_missing:
            task_ids = sample_ids(list_task_ids(source), limit, seed)
        else:
            task_ids = sample_ids(_local_ids(source, raw_dir), limit, seed)
    if download_missing:
        download(source, raw_dir, task_ids)
    seeds = []
    rejects = rejected if rejected is not None else []
    first = len(rejects)
    for tid in task_ids:
        st = normalize_task(
            raw_dir / tid,
            seeds_root / source / slug(tid),
            source=source,
            task_id=tid,
            hermetic=hermetic,
            rejected=rejects,
        )
        if st is not None:
            seeds.append(st)
    out = seeds_root / source / "_rejected.jsonl"
    if len(rejects) > first:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(asdict(r)) + "\n" for r in rejects[first:]))
    else:  # a re-run that rejects nothing must not leave the previous run's list behind
        out.unlink(missing_ok=True)
    return seeds


def _local_ids(source: str, raw_dir: Path) -> list[str]:
    if not raw_dir.exists():
        return []
    return sorted(str(p.parent.relative_to(raw_dir)) for p in raw_dir.rglob("task.toml"))


def local_seeds(root: Path, *, source: str = "local", rejected: list[SeedReject] | None = None) -> list[SeedTask]:
    """Already-Harbor task dirs under `root` (any depth) used as-is; ids are their paths relative to `root`.

    They get the same leak screen as downloaded sources (canary text, COPY/ADD of tests) but are not rewritten."""
    seeds = []
    for toml_path in sorted(root.rglob("task.toml")):
        d = toml_path.parent
        if problems := seed_leak_problems(d):
            log.warning("skip local task %s: %s", d, "; ".join(problems))
            if rejected is not None:
                rejected.append(SeedReject(source, str(d.relative_to(root)), "leak screen: " + "; ".join(problems)))
            continue
        tid = str(d.relative_to(root)) if d != root else d.name
        has_solution = (d / "solution" / "solve.sh").exists()
        prov = (read_task_toml(d).get("metadata") or {}).get("provenance") or {}
        seeds.append(
            SeedTask(
                prov.get("source", source),
                prov.get("task_id", tid),
                d,
                has_solution,
                prov.get("license", "unknown"),
                is_hermetic(d),
            )
        )
    return seeds


def source_revisions(sources: Iterable[str], raw_root: Path) -> dict[str, str | None]:
    """Dataset git revision each registered source was fetched at (None: fetched file by file, or a local directory)."""
    out: dict[str, str | None] = {}
    for src in sources:
        f = raw_root / src / REVISION_FILE
        out[src] = f.read_text().strip() if f.is_file() else None
    return out


def unknown_license_sources(sources: Iterable[str]) -> list[str]:
    """The registered sources among `sources` whose data licence is not stated (a `local` dir is the owner's own)."""
    return [s for s in sources if s in SOURCES and not SOURCES[s].license_known]


def license_record(licenses: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Source -> licence facts for manifest.json: licence, whether it is known, where it was read, the code licence."""
    out: dict[str, dict[str, Any]] = {}
    for src, lic in sorted(licenses.items()):
        info = SOURCES.get(src)
        if info is None:  # a local task dir: whatever its provenance says, otherwise unrecorded
            out[src] = {"license": lic, "known": lic != "unknown", "basis": "task.toml [metadata.provenance]"}
        else:
            out[src] = {
                "license": info.license,
                "known": info.license_known,
                "basis": info.license_basis,
                "code_license": info.code_license,
                "note": info.note,
            }
    return out
