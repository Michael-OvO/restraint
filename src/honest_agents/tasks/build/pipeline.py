"""The build itself: staging, every stage in order, atomic publication of the dataset files."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.bank import (
    DEFAULT_IGNORED_SOURCES,
    QUARANTINE_LEAK,
    QuarantineEntry,
    append_quarantine,
    quarantined_variants,
    read_quarantine,
)
from honest_agents.tasks.analysis import TaskFacts
from honest_agents.tasks.build.config import (
    COLUMNS,
    DEFAULT_TEST_SEEDS,
    MIN_FAMILY_TEST_ROWS,
    MIN_TRAIN_PAIRS,
    RELABELS_FILE,
    SPLIT_SALT,
    SPLIT_SEEDS_FILE,
    SPLITS,
    TEST_IDS_FILE,
    BuildConfig,
    Row,
    heldout_problems,
    make_prompt,
)
from honest_agents.tasks.build.mixture import balance_mixture
from honest_agents.tasks.build.pairs import Pair, apply_validation, enforce_pairing, read_relabels
from honest_agents.tasks.build.reports import (
    CUE_BAND,
    HELDOUT_CAVEAT,
    cue_metrics,
    emitted_label_leaks,
    held_out_coverage,
    pairing_stats,
    training_report,
    yield_report,
)
from honest_agents.tasks.build.seeds import (
    check_benchmark_config,
    check_licenses,
    collect_seeds,
    dedup_seeds_report,
    filter_hermetic,
    neardup_stats,
    read_allowlist,
    seed_documents,
)
from honest_agents.tasks.build.splits import (
    assign_seed_splits,
    check_no_test_leak,
    phrasing_split,
    read_test_ids,
    write_test_ids,
)
from honest_agents.tasks.dedup import NearDupReport, near_duplicate_clusters
from honest_agents.tasks.difficulty import DifficultyGate
from honest_agents.tasks.perturb import (
    LEAK_DROPPED,
    deliverable_names,
    generate_variants,
    set_name_bank,
)
from honest_agents.tasks.sources import (
    SeedReject,
    SeedTask,
    license_record,
    source_revisions,
)

log = logging.getLogger(__name__)


def _publish(stage: Path, targets: dict[str, Path]) -> None:
    """Move the staged files to their final paths, manifest last, so an interrupted publish has no manifest.json."""
    names = sorted(targets, key=lambda n: n == "manifest.json")
    for n in names:
        if n == "manifest.json":
            targets[n].unlink(missing_ok=True)  # an old manifest must not describe the new files
    for n in names:
        targets[n].parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / n, targets[n])


@contextlib.contextmanager
def staging(out: Path) -> Iterator[Path]:
    """A fresh temporary directory next to `out` (same filesystem, so the final renames are atomic); always removed."""
    stage = out.with_name(f".{out.name}.build-{os.getpid()}")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        yield stage
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def build(cfg: BuildConfig) -> dict[str, Any]:
    """Build the dataset into `cfg.out` atomically: nothing but caches is left behind when a gate or a check fails."""
    check_benchmark_config(cfg)  # before any download: decontamination is not optional
    check_licenses(cfg)
    cfg.out.mkdir(parents=True, exist_ok=True)
    with staging(cfg.out) as stage:
        manifest, targets = _build(cfg, stage)
        _publish(stage, targets)
    return manifest


class _Staged:
    """Dataset files written into the staging directory, each with the path it is published to."""

    def __init__(self, stage: Path, out: Path):
        self.stage = stage
        self.out = out
        self.targets: dict[str, Path] = {}

    def __call__(self, name: str, target: Path | None = None) -> Path:
        """Where to write `name` now; `target` (default `<out>/<name>`) is where `_publish` moves it."""
        self.targets[name] = target or self.out / name
        return self.stage / name


@dataclass
class _SplitPlan:
    """How the seed groups are divided, fixed before any variant is built (the phrasing pool depends on the split)."""

    seed_split: dict[str, str]  # split group -> split
    split_before: dict[str, str]  # the same without near-duplicate clustering, for the manifest
    neardup: NearDupReport | None
    groups: list[str]
    frozen_test: set[str]  # test ids of earlier builds
    all_test: set[str]
    test_ids_path: Path
    test_ids_sha: str


def _plan_splits(cfg: BuildConfig, seeds: list[SeedTask], staged: _Staged) -> _SplitPlan:
    """Assign each seed group to a split and stage `split_seeds.json` and the frozen test-id list."""
    split_path = cfg.split_seeds_file or cfg.out / SPLIT_SEEDS_FILE
    previous = json.loads(split_path.read_text()) if split_path.exists() else {}
    test_ids_path = cfg.test_ids_file or cfg.out / TEST_IDS_FILE
    frozen_test = read_test_ids(test_ids_path) if test_ids_path.exists() else set()
    if cfg.test_ids_file is not None and not test_ids_path.exists():
        log.warning(
            "--test-ids-file %s does not exist: starting a new held-out list (hash cutoffs still apply)", test_ids_path
        )
    groups = sorted({s.split_group for s in seeds})
    neardup: NearDupReport | None = None
    if cfg.neardup_threshold > 0 and seeds:
        neardup = near_duplicate_clusters(seed_documents(seeds), threshold=cfg.neardup_threshold)
    split_kw: dict[str, Any] = {
        "n_test": cfg.test_seeds,
        "n_val": cfg.val_seeds,
        "test_bp": cfg.test_bp,
        "val_bp": cfg.val_bp,
        "seed": cfg.split_seed,
        "previous": previous,
        "frozen_test": frozen_test,
    }
    if cfg.test_seeds is not None or cfg.val_seeds is not None:
        log.warning(
            "--test-seeds/--val-seeds rank the current pool: the split is only stable through %s", test_ids_path
        )
    seed_split = assign_seed_splits(groups, clusters=neardup.clusters if neardup else None, **split_kw)
    split_before = assign_seed_splits(groups, **split_kw) if neardup else seed_split
    merged = {**previous, **seed_split}
    staged(SPLIT_SEEDS_FILE, split_path).write_text(json.dumps(merged, indent=2, sort_keys=True))
    all_test = frozen_test | {g for g, sp in merged.items() if sp == "test"}
    if all_test != frozen_test:
        log.info("test ids: %d -> %d (%s)", len(frozen_test), len(all_test), test_ids_path)
    ids_sha = write_test_ids(
        staged(TEST_IDS_FILE, test_ids_path), all_test
    )  # same bytes as before when nothing changed
    if test_ids_path.resolve() != (cfg.out / TEST_IDS_FILE).resolve():  # the output dir stays self-contained
        shutil.copyfile(staged.stage / TEST_IDS_FILE, staged("test_ids.out.txt", cfg.out / TEST_IDS_FILE))
    if not any(sp == "train" for sp in seed_split.values()):
        log.warning("no training seeds: the test/val cutoffs use the whole pool")
    n_test_groups = sum(sp == "test" for sp in seed_split.values())
    if cfg.test_seeds is None and n_test_groups < DEFAULT_TEST_SEEDS:
        log.warning(
            "only %d test seed groups (< %d): raise --limit or --test-bp; changing them later is safe, no source moves",
            n_test_groups,
            DEFAULT_TEST_SEEDS,
        )
    return _SplitPlan(seed_split, split_before, neardup, groups, frozen_test, all_test, test_ids_path, ids_sha)


def _perturb_seeds(
    cfg: BuildConfig, seeds: list[SeedTask], plan: _SplitPlan, tasks_root: Path
) -> tuple[list[Pair], int]:
    """Generate every variant of every seed; returns the `(seed, variant)` pairs and the seeds skipped for lacking an
    oracle solution (those only join a feasible-only pool when `--feasible-allowlist` names them)."""
    bank: list[tuple[str, str, str]] = []  # D' of a conflicting_constraint control: other tasks' deliverable names
    for s in seeds:
        if s.has_solution:
            split = phrasing_split(plan.seed_split[s.split_group])
            bank += [(split, s.pair_id, n) for n in deliverable_names(TaskFacts.load(s.task_dir))]
    set_name_bank(bank)
    allow = read_allowlist(cfg.feasible_allowlist)
    variants: list[Pair] = []
    skipped_no_solution = 0
    for s in seeds:
        split = phrasing_split(plan.seed_split[s.split_group])
        if s.has_solution:
            for v in generate_variants(
                s.task_dir,
                tasks_root,
                pair_id=s.pair_id,
                operators=cfg.operators,
                seed=cfg.seed,
                split=split,
                require_controls=cfg.require_controls,
            ):
                variants.append((s, v))
        elif s.pair_id in allow:  # feasible-only pool (no oracle => no infeasible variants)
            variants += [(s, v) for v in generate_variants(s.task_dir, tasks_root, pair_id=s.pair_id, operators=[])]
        else:
            skipped_no_solution += 1
    if skipped_no_solution:
        log.warning("skipped %d seeds without oracle solution (not in --feasible-allowlist)", skipped_no_solution)
    return variants, skipped_no_solution


def _drop_quarantined(cfg: BuildConfig, variants: list[Pair]) -> tuple[list[Pair], int, int]:
    """Remove quarantined variants and re-enforce pairing; also returns how many were removed and how many
    quarantine entries from eval sources were ignored."""
    quarantine = quarantined_variants(
        cfg.quarantine_file,
        kinds=cfg.quarantine_kinds,
        exclude_sources=() if cfg.quarantine_include_eval else DEFAULT_IGNORED_SOURCES,
    )
    n_eval_ignored = (
        0
        if cfg.quarantine_include_eval
        else len(quarantined_variants(cfg.quarantine_file, kinds=cfg.quarantine_kinds, exclude_sources=()) - quarantine)
    )
    n_quarantined = sum(v.variant_id in quarantine for _, v in variants)
    kept = enforce_pairing(
        [(s, v) for s, v in variants if v.variant_id not in quarantine], require_controls=cfg.require_controls
    )
    return kept, n_quarantined, n_eval_ignored


def _validate_and_relabel(cfg: BuildConfig, variants: list[Pair], staged: _Staged) -> tuple[list[Pair], dict[str, Any]]:
    """Oracle-validate the variants (when asked), apply validation results and relabels, quarantine any leak found."""
    validation_path = cfg.validation_jsonl or cfg.out / "validation.jsonl"
    info: dict[str, Any] = {"relabelled": [], "leaks": [], "dropped": 0}
    if cfg.validate:
        from honest_agents.tasks.validate import validate_many

        to_check = [v.task_dir for s, v in variants if s.has_solution]
        asyncio.run(validate_many(to_check, validation_path, concurrency=cfg.concurrency))
    relabels_path = cfg.out / RELABELS_FILE
    relabels = read_relabels(relabels_path)
    validating = bool(cfg.validate or cfg.validation_jsonl)
    if validating or relabels:
        from honest_agents.tasks.validate import read_results

        before = len(variants)
        results = read_results(validation_path) if validating else {}
        variants, info = apply_validation(
            variants,
            results,
            relabels=relabels,
            allow_oracle_tamper=cfg.allow_oracle_tamper,
            require_valid=validating,
            require_null_check=cfg.require_null_check,
            min_oracle_runs=cfg.min_oracle_runs,
        )
        staged(RELABELS_FILE).write_text(json.dumps(relabels, indent=2, sort_keys=True))
        variants = enforce_pairing(variants, require_controls=cfg.require_controls)
        log.info("validation kept %d/%d variants (%d relabelled)", len(variants), before, len(info["relabelled"]))
        if info["leaks"]:
            _quarantine_leaks(info["leaks"], cfg=cfg)
    return variants, info


def _make_rows(cfg: BuildConfig, variants: list[Pair], seed_split: dict[str, str], tasks_root: Path) -> list[Row]:
    prompt = make_prompt()
    root = cfg.task_dir_root.rstrip("/") if cfg.task_dir_root else str(tasks_root.resolve())
    return [
        Row(
            prompt=prompt,
            task_dir=f"{root}/{v.variant_id}",
            variant_id=v.variant_id,
            pair_id=s.pair_id,
            feasible=v.feasible,
            is_control=v.is_control,
            kind=v.kind,
            family=v.spec.family or "feasible",
            perturbation=v.perturbation,
            source=s.source,
            split=seed_split[s.split_group],
            split_group=s.split_group,
        )
        for s, v in variants
    ]


def _select_split_rows(
    cfg: BuildConfig, rows: list[Row], all_test: set[str], staged: _Staged
) -> tuple[dict[str, list[Row]], list[Row], int]:
    """Per-split rows after the held-out-family rule, the difficulty gate and the mixture balance, written to staging.

    Returns the splits, the unfiltered training pool and the number of rows the difficulty gate dropped."""
    heldout = set(cfg.heldout_families)
    splits: dict[str, list[Row]] = {k: [] for k in SPLITS}
    for r in rows:
        if r.family in heldout and r.split != "test":
            continue  # held-out families are only ever seen at test time
        splits[r.split].append(r)

    check_no_test_leak((r for name, rs in splits.items() if name != "test" for r in rs), all_test)
    splits_pool = list(splits["train"])
    # Phase 0 measures a sample of the pool, so every train pair is listed here before the gate and the mixture.
    write_jsonl(staged("train_pool.jsonl"), [row_dict(r) for r in splits["train"]])
    gate = DifficultyGate.load(cfg.difficulty_json) if cfg.difficulty_json else None
    n_difficulty_dropped = 0
    if gate is not None:  # the training split only: test metrics stay unfiltered
        kept = [r for r in splits["train"] if gate.admits(r.pair_id, strict=cfg.difficulty_strict)]
        n_difficulty_dropped = len(splits["train"]) - len(kept)
        splits["train"] = kept
    for name in ("train", "val"):  # test keeps every variant: it is the held-out evaluation set
        splits[name] = balance_mixture(splits[name], cfg.mixture, cfg.seed, paired=cfg.pair_mixture)
    for name, rs in splits.items():
        write_jsonl(staged(f"{name}.jsonl"), [row_dict(r) for r in rs])
    return splits, splits_pool, n_difficulty_dropped


def _enforce_cue_gate(
    cfg: BuildConfig, variants: list[Pair], splits: dict[str, list[Row]], heldout: set[str]
) -> dict[str, Any]:
    """Existence-cue metrics of the kept variants; a cue outside `CUE_BAND` in a training family fails the build."""
    kept_ids = {name: {r.variant_id for r in rs} for name, rs in splits.items()}
    cue = cue_metrics(
        {name: [v for _, v in variants if v.variant_id in ids] for name, ids in kept_ids.items()}, heldout=heldout
    )
    bad = [f"{name}: {m}" for name, ms in cue["violations"].items() for m in ms]  # training families only
    for m in bad:
        log.warning("existence cue outside %s in a TRAINING family: %s", CUE_BAND, m)
    for name, ms in cue.get("heldout_violations", {}).items():
        for m in ms:
            log.warning(
                "cue outside %s in a HELD-OUT family (reported, not enforced; %s): %s", CUE_BAND, HELDOUT_CAVEAT, m
            )
    for name, ms in cue.get("watch", {}).items():
        for m in ms:
            log.warning("cue worth a look (not enforced) in %s: %s", name, m)
    errors = [f"{name}: {m['error']}" for name, m in cue.items() if isinstance(m, dict) and "error" in m]
    if cfg.enforce_cue_band and (bad or errors or not cue["available"]):  # a metric that crashed is not a clean one
        raise ValueError(
            f"existence AUC outside {CUE_BAND}: {bad or errors or 'analysis.existence_auc is not available'}"
        )
    return cue


def _split_summary(splits: dict[str, list[Row]]) -> dict[str, dict[str, Any]]:
    return {
        k: {
            "rows": len(rs),
            "feasible": sum(r.feasible for r in rs),
            "kinds": dict(Counter(r.kind for r in rs)),
            "families": dict(Counter(r.family for r in rs)),
            "pairs": len({r.pair_id for r in rs}),
        }
        for k, rs in splits.items()
    }


def _build(cfg: BuildConfig, stage: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Everything `build` does, writing each dataset file into `stage`; returns the manifest and staged name -> target."""
    tasks_root = cfg.out / "tasks"
    staged = _Staged(stage, cfg.out)

    for problem in heldout_problems(cfg.heldout_families):
        log.warning("held-out families %s: %s", cfg.heldout_families, problem)

    rejected: list[SeedReject] = []
    seeds = collect_seeds(cfg, rejected)
    n_collected = len(seeds)
    seeds, dropped, benchmark_stats = dedup_seeds_report(cfg, seeds)
    write_jsonl(staged("dedup.jsonl"), dropped)
    seeds, not_hermetic = filter_hermetic(seeds, allow_legacy=cfg.allow_legacy_verifier)
    if not seeds:  # checked after every filter: a wrong --raw-dir, a decontamination that removed everything or a pool
        # the hermetic conversion refused must not look like a finished, empty build (no manifest, no dataset file)
        raise ValueError(
            f"no seed tasks found: {n_collected} collected from {cfg.sources} (raw dir {cfg.raw_dir or cfg.out / 'raw'}, "
            f"{len(rejected)} refused on normalisation), {len(dropped)} dropped as benchmark duplicates, "
            f"{len(not_hermetic)} not graded by the hermetic verifier"
        )

    plan = _plan_splits(cfg, seeds, staged)
    leak_dropped_before = Counter(LEAK_DROPPED)
    variants, skipped_no_solution = _perturb_seeds(cfg, seeds, plan, tasks_root)
    variants, n_quarantined, n_quarantine_eval_ignored = _drop_quarantined(cfg, variants)
    variants, info = _validate_and_relabel(cfg, variants, staged)

    label_leaks_found = emitted_label_leaks(variants)
    if label_leaks_found:
        raise ValueError(f"{len(label_leaks_found)} emitted files carry label strings, e.g. {label_leaks_found[:3]}")
    n_leak_dropped = dict(Counter(LEAK_DROPPED) - leak_dropped_before)

    rows = _make_rows(cfg, variants, plan.seed_split, tasks_root)
    heldout = set(cfg.heldout_families)
    splits, splits_pool, n_difficulty_dropped = _select_split_rows(cfg, rows, plan.all_test, staged)

    test_coverage = held_out_coverage(splits["test"], cfg.operators)
    for fam, n in test_coverage["short_families"].items():
        log.warning(
            "test split has %d infeasible %s variants (< %d): report this family pooled or not at all",
            n,
            fam,
            MIN_FAMILY_TEST_ROWS,
        )
    cue = _enforce_cue_gate(cfg, variants, splits, heldout)
    training = training_report(splits, [f for f in cfg.operators if f not in heldout], pool=splits_pool)
    for fam, n in training["thin_families"].items():
        log.warning(
            "training family %s has only %d pairs in train (< %d): its rate is too noisy to learn or report",
            fam,
            n,
            MIN_TRAIN_PAIRS,
        )
    neardup = plan.neardup
    yields = yield_report(rows, plan.seed_split, neardup.clusters if neardup else {}, splits, heldout, cfg.operators)
    manifest = {
        "test_coverage": test_coverage,
        "training": training,
        "yield": yields,
        "split_stability": {
            "mode": "count (pool-dependent)" if cfg.test_seeds is not None or cfg.val_seeds is not None else "hash",
            "salt": f"{SPLIT_SALT}|{cfg.split_seed}",
            "test_bp": cfg.test_bp,
            "val_bp": cfg.val_bp,
            "test_ids_file": str(plan.test_ids_path),
            "test_ids_n": len(plan.all_test),
            "test_ids_sha256": plan.test_ids_sha,
            "n_test_ids_added_this_build": len(plan.all_test - plan.frozen_test),
            "test_ids_forced_from_file": len(plan.frozen_test & set(plan.groups)),
            "quarantine_eval_entries_ignored": n_quarantine_eval_ignored,
        },
        "benchmark_overlap": benchmark_stats,
        "neardup": neardup_stats(neardup, plan.split_before, plan.seed_split),
        "pairing": pairing_stats(splits),
        "cue_metrics": cue,
        "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
        "licenses": license_record({x.source: x.license for x in seeds}),
        "source_revisions": source_revisions(cfg.sources, cfg.raw_dir or cfg.out / "raw"),
        "n_seeds": len(seeds),
        "n_dedup_dropped": len(dropped),
        "n_not_hermetic": len(not_hermetic),
        "n_hermetic_rejected": len(rejected),
        "n_skipped_no_solution": skipped_no_solution,
        "label_leaks": {"emitted": 0, "variants_dropped_for_a_label_string": n_leak_dropped},
        "n_quarantined": n_quarantined,
        "n_relabelled": len(info["relabelled"]),
        "n_leaks_found": len(info["leaks"]),
        "n_difficulty_dropped": n_difficulty_dropped,
        "heldout_families": sorted(heldout),
        "seed_splits": dict(Counter(plan.seed_split.values())),
        "splits": _split_summary(splits),
    }
    staged("manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest, staged.targets


def _quarantine_leaks(leaks: list[Any], *, cfg: BuildConfig) -> None:
    """Append validation leaks to the quarantine list (once per variant) when a list is configured."""
    known = {e.variant_id for e in read_quarantine(cfg.quarantine_file)}
    entries = [
        QuarantineEntry(
            variant_id=r.variant_id,
            task_dir=r.task_dir,
            pair_id=r.pair_id,
            perturbation=r.perturbation,
            family=r.perturbation,
            kind=QUARANTINE_LEAK,
            source="validation",
            detail=str(getattr(r, "reason", "")),
        )
        for r in leaks
        if r.variant_id not in known
    ]
    if entries and append_quarantine(entries, cfg.quarantine_file) is None:
        log.warning("%d leaking variants found but no quarantine file is configured", len(entries))


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def row_dict(r: Row) -> dict[str, Any]:
    d = asdict(r)
    d.pop("split_group")
    assert set(d) == set(COLUMNS), set(d) ^ set(COLUMNS)
    return d


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
