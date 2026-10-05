"""Command-line entry point of `python -m honest_agents.tasks.build`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from honest_agents.tasks import build as build_package  # only its docstring is read, at call time
from honest_agents.tasks.build.config import (
    DEFAULT_HELDOUT_FAMILIES,
    DEFAULT_MIXTURE,
    DEFAULT_NEARDUP_THRESHOLD,
    DEFAULT_TEST_BP,
    DEFAULT_VAL_BP,
    BuildConfig,
)
from honest_agents.tasks.build.pipeline import build
from honest_agents.tasks.build.reports import format_yield
from honest_agents.tasks.perturb import (
    DEFAULT_OPERATORS,
    OPERATORS,
)


def parse_args(argv: list[str] | None = None) -> BuildConfig:
    ap = argparse.ArgumentParser(
        description=build_package.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=["endless-terminals"], help="+ seta-env, tmax-15k, local")
    ap.add_argument("--raw-dir", type=Path, help="download cache (default <out>/raw)")
    ap.add_argument("--seeds-dir", type=Path, help="Harbor task dirs for --sources local")
    ap.add_argument("--limit", type=int, help="max seed tasks per source (seeded sample)")
    ap.add_argument("--no-download", action="store_true", help="only use tasks already in --raw-dir")
    ap.add_argument("--operators", nargs="+", default=list(DEFAULT_OPERATORS), choices=list(OPERATORS))
    ap.add_argument(
        "--heldout-families",
        default=",".join(DEFAULT_HELDOUT_FAMILIES),
        help="comma-separated test-only families (default conflicting_constraint,no_network: lexical cue / too few pairs)",
    )
    ap.add_argument(
        "--test-bp",
        type=int,
        default=DEFAULT_TEST_BP,
        help="test share of the split hash space, in 1/10000 (fixed cutoff: independent of the pool; default 2200)",
    )
    ap.add_argument("--val-bp", type=int, default=DEFAULT_VAL_BP, help="validation share, in 1/10000 (default 2200)")
    ap.add_argument(
        "--test-seeds",
        type=int,
        help="LEGACY, dev pools only: hold out exactly this many source tasks by ranking the current pool",
    )
    ap.add_argument("--val-seeds", type=int, help="LEGACY: validation source tasks, ranked within the pool")
    ap.add_argument(
        "--split-seed", type=int, default=0, help="salt of the split hash (never change it after a release)"
    )
    ap.add_argument(
        "--test-ids-file",
        type=Path,
        help="held-out ids with a sha256 line (default <out>/test_ids.txt); read if present, extended, never shrunk",
    )
    ap.add_argument(
        "--neardup-threshold",
        type=float,
        default=DEFAULT_NEARDUP_THRESHOLD,
        help="word 3-gram Jaccard above which two seeds' instructions form one split unit (0 = off)",
    )
    ap.add_argument(
        "--i-know-no-benchmark",
        action="store_true",
        help="build without --benchmark-dir (tests only: the data would not be decontaminated)",
    )
    ap.add_argument(
        "--allow-disabled-sources", action="store_true", help="allow the seta-env / tmax-15k sources (leak audit L4)"
    )
    ap.add_argument(
        "--allow-unknown-license",
        action="store_true",
        help="build from a source whose data licence is not stated (check its terms first; docs/data.md)",
    )
    ap.add_argument(
        "--no-pair-mixture", action="store_true", help="draw infeasible and control rows independently (old behaviour)"
    )
    ap.add_argument(
        "--quarantine-include-eval",
        action="store_true",
        help="honour quarantine entries found while evaluating (source=eval): they change test.jsonl",
    )
    ap.add_argument(
        "--enforce-cue-band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail the build when a family's existence AUC leaves [0.4, 0.6] (L1; default on, --no-enforce-cue-band for dev)",
    )
    ap.add_argument(
        "--mixture",
        default=",".join(str(x) for x in DEFAULT_MIXTURE),
        help="plain feasible,control,infeasible shares of train and val (default 0.55,0.20,0.25); test is kept whole",
    )
    ap.add_argument("--benchmark-dir", type=Path, action="append", default=[], help="eval benchmark task dirs")
    ap.add_argument("--dedup-threshold", type=float, default=0.5)
    ap.add_argument("--feasible-allowlist", type=Path, help="pair_ids of solution-less tasks to keep (pass@k)")
    ap.add_argument("--validate", action="store_true", help="oracle-validate variants (needs Docker)")
    ap.add_argument("--validation-jsonl", type=Path, help="filter with an existing validation.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--allow-oracle-tamper", action="store_true")
    ap.add_argument(
        "--allow-weak-validation",
        action="store_true",
        help="accept validation results without the null check or with fewer than 3 oracle runs (never for training)",
    )
    ap.add_argument("--task-dir-root", help="prefix for task_dir in rows (where tasks/ lives on the training node)")
    ap.add_argument(
        "--quarantine-file", type=Path, help="variants listed here (kind leak) are excluded; leaks found are appended"
    )
    ap.add_argument(
        "--split-seeds-file",
        "--split-seeds",
        type=Path,
        help="seed -> split assignment to keep across builds (default <out>/split_seeds.json); reuse it when "
        "building into a fresh --out so test seeds stay out of training",
    )
    ap.add_argument(
        "--difficulty-json", type=Path, help="Phase-0 difficulty gate (tasks.difficulty); filters train only"
    )
    ap.add_argument(
        "--difficulty-strict",
        action="store_true",
        help="with --difficulty-json: keep only source tasks the Phase-0 run measured and kept (drop unmeasured ones)",
    )
    ap.add_argument(
        "--allow-legacy-verifier", action="store_true", help="keep seeds that are not hermetic (eval sets, tests)"
    )
    ap.add_argument(
        "--allow-missing-controls", action="store_true", help="keep infeasible variants that have no control"
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return BuildConfig(
        out=a.out,
        sources=a.sources,
        raw_dir=a.raw_dir,
        seeds_dir=a.seeds_dir,
        limit=a.limit,
        download=not a.no_download,
        operators=a.operators,
        heldout_families=[f for f in a.heldout_families.split(",") if f],
        test_bp=a.test_bp,
        val_bp=a.val_bp,
        test_seeds=a.test_seeds,
        val_seeds=a.val_seeds,
        split_seed=a.split_seed,
        test_ids_file=a.test_ids_file,
        neardup_threshold=a.neardup_threshold,
        i_know_no_benchmark=a.i_know_no_benchmark,
        allow_disabled_sources=a.allow_disabled_sources,
        allow_unknown_license=a.allow_unknown_license,
        pair_mixture=not a.no_pair_mixture,
        quarantine_include_eval=a.quarantine_include_eval,
        enforce_cue_band=a.enforce_cue_band,
        mixture=tuple(float(x) for x in a.mixture.split(",")),  # type: ignore[arg-type]
        benchmark_dirs=a.benchmark_dir,
        dedup_threshold=a.dedup_threshold,
        feasible_allowlist=a.feasible_allowlist,
        validate=a.validate,
        validation_jsonl=a.validation_jsonl,
        concurrency=a.concurrency,
        allow_oracle_tamper=a.allow_oracle_tamper,
        task_dir_root=a.task_dir_root,
        quarantine_file=a.quarantine_file,
        difficulty_json=a.difficulty_json,
        difficulty_strict=a.difficulty_strict,
        split_seeds_file=a.split_seeds_file,
        allow_legacy_verifier=a.allow_legacy_verifier,
        require_controls=not a.allow_missing_controls,
        require_null_check=not a.allow_weak_validation,
        min_oracle_runs=0 if a.allow_weak_validation else 3,
        seed=a.seed,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = build(parse_args(argv))
    print(json.dumps(manifest["splits"], indent=2))
    print(format_yield(manifest["yield"]))
    print("training pairs per family:", json.dumps(manifest["training"]["pairs_per_family"]))
