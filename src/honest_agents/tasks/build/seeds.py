"""Collecting seeds: sources, licences, decontamination, near-duplicate removal and the hermetic filter."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from honest_agents.tasks.build.config import BuildConfig
from honest_agents.tasks.dedup import Deduplicator, NearDupReport, load_benchmark_tasks
from honest_agents.tasks.sources import (
    SOURCES,
    SeedReject,
    SeedTask,
    local_seeds,
    prepare_source,
    unknown_license_sources,
)

log = logging.getLogger(__name__)


# ── seeds ────────────────────────────────────────────────────────────────────


def collect_seeds(cfg: BuildConfig, rejected: list[SeedReject] | None = None) -> list[SeedTask]:
    raw = cfg.raw_dir or cfg.out / "raw"
    seeds: list[SeedTask] = []
    for src in cfg.sources:
        if src == "local":
            if cfg.seeds_dir is None:
                raise ValueError("--sources local needs --seeds-dir")
            seeds += local_seeds(cfg.seeds_dir, rejected=rejected)
        elif src in SOURCES:
            seeds += prepare_source(
                src,
                raw,
                cfg.out / "seeds",
                limit=cfg.limit,
                seed=cfg.seed,
                download_missing=cfg.download,
                hermetic=False if cfg.allow_legacy_verifier else None,
                rejected=rejected,
                allow_disabled=cfg.allow_disabled_sources,
            )
        else:
            raise ValueError(f"unknown source {src!r}; known: {sorted(SOURCES)} + ['local']")
    return seeds


def check_benchmark_config(cfg: BuildConfig) -> None:
    """Decontamination is mandatory: fail before any download when there is nothing to deduplicate against."""
    if cfg.i_know_no_benchmark:
        return
    if not cfg.benchmark_dirs:
        raise ValueError(
            "--benchmark-dir is required: run scripts/fetch_benchmarks.sh and pass its output dir "
            "(--i-know-no-benchmark builds without decontamination; tests only, never training data)"
        )
    missing = [str(d) for d in cfg.benchmark_dirs if not d.is_dir()]
    if missing:
        raise ValueError(f"--benchmark-dir does not exist: {missing} (run scripts/fetch_benchmarks.sh)")


def check_licenses(cfg: BuildConfig) -> None:
    """Refuse a source whose data licence is not stated, before any download (`--allow-unknown-license` overrides)."""
    unknown = unknown_license_sources(cfg.sources)
    if unknown and not cfg.allow_unknown_license:
        raise ValueError(
            f"sources with an unknown licence: {unknown} ({'; '.join(SOURCES[s].license_basis for s in unknown)}); "
            "check the terms and pass --allow-unknown-license to build anyway (docs/data.md, licences)"
        )
    if unknown:
        log.warning("building from sources with an unknown licence %s (--allow-unknown-license)", unknown)


def dedup_seeds_report(
    cfg: BuildConfig, seeds: list[SeedTask]
) -> tuple[list[SeedTask], list[dict[str, Any]], dict[str, Any]]:
    """Drop seeds that duplicate a benchmark task; also returns overlap statistics for the manifest."""
    check_benchmark_config(cfg)
    if not cfg.benchmark_dirs:
        log.warning("--i-know-no-benchmark: seeds are NOT deduplicated against the eval benchmarks")
        return seeds, [], {"checked": False}
    bench = [b for d in cfg.benchmark_dirs for b in load_benchmark_tasks(d)]
    if not bench and not cfg.i_know_no_benchmark:
        raise ValueError(f"no benchmark tasks (instruction.md) found under {[str(d) for d in cfg.benchmark_dirs]}")
    dd = Deduplicator(bench, threshold=cfg.dedup_threshold)
    kept, dropped, best = [], [], []
    for s in seeds:
        text = (s.task_dir / "instruction.md").read_text()
        _, j = dd.best_overlap(text)
        best.append(j)
        hit = dd.match(s.task_id.split("/")[-1], text)
        if hit is None:
            kept.append(s)
        else:
            dropped.append({"pair_id": s.pair_id, "benchmark": hit[0].benchmark, "match": hit[0].name, "why": hit[1]})
    log.info("dedup: dropped %d/%d seeds against %d benchmark tasks", len(dropped), len(seeds), len(bench))
    stats = {
        "checked": True,
        "n_benchmark_tasks": len(bench),
        "benchmarks": dict(Counter(b.benchmark for b in bench)),
        "threshold": cfg.dedup_threshold,
        "n_seeds_checked": len(seeds),
        "n_dropped": len(dropped),
        "max_8gram_jaccard": round(max(best, default=0.0), 4),
        "n_seeds_with_any_8gram_overlap": sum(j > 0 for j in best),
    }
    return kept, dropped, stats


def dedup_seeds(cfg: BuildConfig, seeds: list[SeedTask]) -> tuple[list[SeedTask], list[dict[str, Any]]]:
    kept, dropped, _ = dedup_seeds_report(cfg, seeds)
    return kept, dropped


def seed_documents(seeds: Iterable[SeedTask]) -> dict[str, tuple[str, str]]:
    """Split group -> (instruction, tests text), for near-duplicate clustering. Members of a group are concatenated."""
    docs: dict[str, tuple[str, str]] = {}
    for s in seeds:
        inst = (s.task_dir / "instruction.md").read_text(errors="replace")
        tests = "\n".join(f.read_text(errors="replace") for f in sorted((s.task_dir / "tests").rglob("*.py")))
        a, b = docs.get(s.split_group, ("", ""))
        docs[s.split_group] = (a + "\n" + inst, b + "\n" + tests)
    return docs


def neardup_stats(report: NearDupReport | None, before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    """How many test (and val) groups have a near-twin in train, per Jaccard threshold, before and after clustering.

    `before` is the assignment without clusters, `after` the one used; a pair counts when its Jaccard is at least the
    threshold on either the instruction (3-grams) or the tests (5-grams)."""
    from honest_agents.tasks.dedup import NEARDUP_REPORT_THRESHOLDS

    if report is None:
        return {"enabled": False}

    def twins(assign: dict[str, str], thr: float, held: str) -> int:
        hit = set()
        for a, b, j, _ in report.pairs:
            if j >= thr and a in assign and b in assign:
                for x, y in ((a, b), (b, a)):
                    if assign[x] == held and assign[y] == "train":
                        hit.add(x)
        return len(hit)

    return {
        "enabled": True,
        "threshold": report.threshold,
        "n_groups": len(report.clusters),
        "n_groups_in_multi_clusters": report.n_clustered(),
        "n_clusters": len(set(report.clusters.values())),
        "max_pair_jaccard": round(max((p[2] for p in report.pairs), default=0.0), 4),
        "test_groups_with_train_twin_before": {str(t): twins(before, t, "test") for t in NEARDUP_REPORT_THRESHOLDS},
        "test_groups_with_train_twin_after": {str(t): twins(after, t, "test") for t in NEARDUP_REPORT_THRESHOLDS},
        "val_groups_with_train_twin_after": {str(t): twins(after, t, "val") for t in NEARDUP_REPORT_THRESHOLDS},
    }


def filter_hermetic(seeds: list[SeedTask], *, allow_legacy: bool) -> tuple[list[SeedTask], list[SeedTask]]:
    """Seeds graded by the offline pytest verifier; the rest are dropped unless legacy verifiers are allowed."""
    if allow_legacy:
        return list(seeds), []
    kept = [s for s in seeds if s.hermetic]
    dropped = [s for s in seeds if not s.hermetic]
    if dropped:
        log.warning("dropped %d/%d seeds that are not graded by the hermetic pytest verifier", len(dropped), len(seeds))
    return kept, dropped


def read_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
