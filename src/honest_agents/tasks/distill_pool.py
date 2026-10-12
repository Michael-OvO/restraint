"""Iteration-12 distillation pool: the train rows the base model is sampled on WITH the hint (docs/iteration-12-design.md
section 2.3), split by how many samples each row gets.

    python -m honest_agents.tasks.distill_pool --pool data/train_pool.jsonl \\
        --rl-pool ~/Documents/honest-agents-archive/iterations/11-sft-grpo/files/data/train_rl.jsonl \\
        --validation data/validation.jsonl \\
        --bank ~/Documents/honest-agents-archive/iterations/03-passk-bank/eval/passk/bank \\
        --out data/distill_pool.jsonl [--report data/distill_pool.report.json] [--plain-n 50] [--seed 0]

**Which rows.** Every train row whose source (`pair_id`) is outside the iteration-11 RL pool, so the SFT data and the
GRPO pool stay disjoint (as in iteration 11) and GRPO tests generalisation to twins the distilled policy never saw. The
RL pool is the iteration-11 file (351 rows, 108 sources; archived under 11-sft-grpo/files/data/train_rl.jsonl, and
data/train_rl_candidates.jsonl in the working data). NOT `data/train_rl.jsonl` of the working data: that is still the
iteration-9 pool (360 rows, 132 sources). The teacher's 121 sources are included on purpose: their teacher
trajectories are no longer used. Leaky impossible twins are left out (as `tasks.rl_pool` does: the oracle's
`validation.jsonl` leak flags, quarantine-list leaks, and twins a base rollout of the pass@4 bank passed cleanly); a
leak flag on a row the oracle relabelled as a control does not remove it (it is a valid feasible control now).

**Plain rows.** Only `plain_n` (50) plain rows, drawn from the base pass@4 band `plain_band` (1-3 passes of 4) in a
stable hash order, so they are neither trivial nor never-passing. The 2026-10 data has only 17 such rows outside the RL
pool (the RL pool's difficulty gate took the band-2/3 sources; 108 of the 145 are 4/4, 18 are 0/4), so the shortfall is
filled from the 4/4 rows (`plain_fill: easy`, the default; `none` keeps the band rows only). A plain example is there to
keep the base model's ordinary competence in the mixture, which an easy task also does; a never-passing row (0/4) or one
without bank rows is never used. The report states the shortfall.

**Samples per row** (temperature 1.0, the hinted prompt): high_risk twins 4, remove_input twins 3, revoke_permission
twins 2, controls 2, plain 2 (design: 364 + 126 + 58 + 324 + 100 = 972 episodes before leak exclusions). `run_eval
--n-samples` applies to a whole run, so the rows are written to one file per sample count next to `--out`
(`<stem>_x4.jsonl`, `<stem>_x3.jsonl`, `<stem>_x2.jsonl`), plus `--out` itself with every row and its `n_samples`
column (the record of the whole pool). Three runs into ONE bank directory (each `run_eval` process writes its own
`bank-<pid>.jsonl`), high_risk first so G2's early check (read rate at the first ~200 episodes) can run on it:

    for n in 4 3 2; do
      python -m honest_agents.eval.run_eval --split data/distill_pool_x$n.jsonl --n-samples $n \\
        --system-prompt-file src/honest_agents/eval/prompts/hint_tier1.txt --train-config configs/grpo_it12.yaml \\
        --bank-dir data/distill/bank --out runs/eval/it12_a1_x$n --concurrency 96 \\
        --base-url http://localhost:8000/v1 --model policy
    done

(`report["run_eval"]` prints these lines for the files actually written.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.tasks.rl_pool import (
    EXCL_LEAK_BANK,
    EXCL_LEAK_QUARANTINE,
    EXCL_LEAK_VALIDATION,
    Exclusions,
    read_bank_stats,
    read_jsonl,
    validation_leaks,
)

Row = dict[str, Any]

# Samples per row, by "<kind>/<family>" for twins and by kind otherwise (docs/iteration-12-design.md section 2.3).
DEFAULT_SAMPLES: dict[str, int] = {
    "infeasible/high_risk": 4,
    "infeasible/remove_input": 3,
    "infeasible/revoke_permission": 2,
    "control": 2,
    "feasible": 2,
}
DEFAULT_TWIN_SAMPLES = 2  # a twin family the table does not list
PLAIN_FILL_EASY = "easy"
PLAIN_FILL_NONE = "none"

EXCL_RL_POOL = "rl_pool_source"
EXCL_PLAIN_NOT_DRAWN = "plain_not_drawn"
EXCL_PLAIN_NEVER_PASSES = "plain_never_passes"
EXCL_PLAIN_UNMEASURED = "plain_unmeasured"


@dataclass
class DistillOptions:
    pool: str = "data/train_pool.jsonl"
    rl_pool: str | None = None  # iteration-11 train_rl.jsonl (required: see the module docstring)
    validation: str | None = None
    quarantine: list[str] = field(default_factory=list)
    bank: list[str] = field(default_factory=list)  # base pass@4 bank: plain band and twin leaks
    out: str | None = None
    report: str | None = None
    seed: int = 0
    plain_n: int = 50
    plain_band: tuple[int, int] = (1, 3)
    plain_fill: str = PLAIN_FILL_EASY
    min_bank_samples: int = 4
    samples: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SAMPLES))

    def __post_init__(self) -> None:
        lo, hi = self.plain_band
        if not 0 <= lo <= hi:
            raise ValueError(f"plain_band {self.plain_band} is not an interval of pass counts")
        if self.plain_fill not in (PLAIN_FILL_EASY, PLAIN_FILL_NONE):
            raise ValueError(f"plain_fill must be {PLAIN_FILL_EASY!r} or {PLAIN_FILL_NONE!r}")
        if self.plain_n < 0 or any(n < 1 for n in self.samples.values()):
            raise ValueError("plain_n must be >= 0 and every sample count >= 1")


def _source(r: Row) -> str:
    return r.get("pair_id") or r["variant_id"]


def samples_for(r: Row, table: Mapping[str, int]) -> int:
    if r["kind"] == "infeasible":
        return table.get(f"infeasible/{r.get('family')}", DEFAULT_TWIN_SAMPLES)
    return table[r["kind"]]


def _hash(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


@dataclass
class DistillBuild:
    rows: list[Row]  # pool order, each with its `n_samples` column
    excluded: dict[str, list[str]]  # reason -> variant ids
    plain_band_rows: int  # plain rows drawn from the band
    plain_fill_rows: int  # plain rows filled from above the band
    plain_shortfall: int  # plain_n minus the plain rows drawn


def plan(
    rows: Sequence[Row],
    rl_sources: set[str],
    exclusions: Exclusions,
    passes: Mapping[str, tuple[int, int]],
    opts: DistillOptions,
) -> DistillBuild:
    """The distillation pool. `passes`: variant id -> (scored episodes, clean passes) from the base pass@k bank."""
    excluded: dict[str, set[str]] = {}

    def drop(r: Row, why: str) -> None:
        excluded.setdefault(why, set()).add(r["variant_id"])

    live: list[Row] = []
    for r in rows:
        if _source(r) in rl_sources:
            drop(r, EXCL_RL_POOL)
        elif (why := exclusions.reason(r)) is not None:
            drop(r, why)
        else:
            live.append(r)

    lo, hi = opts.plain_band
    band: list[Row] = []
    easy: list[Row] = []
    for r in live:
        if r["kind"] != "feasible":
            continue
        n, p = passes.get(r["variant_id"], (0, 0))
        if n < opts.min_bank_samples:
            drop(r, EXCL_PLAIN_UNMEASURED)
        elif p == 0:
            drop(r, EXCL_PLAIN_NEVER_PASSES)
        elif lo <= p <= hi:
            band.append(r)
        elif p > hi:
            easy.append(r)
    band.sort(key=lambda r: _hash(r["variant_id"], opts.seed))
    easy.sort(key=lambda r: _hash(r["variant_id"], opts.seed))
    chosen = band[: opts.plain_n]
    fill = easy[: opts.plain_n - len(chosen)] if opts.plain_fill == PLAIN_FILL_EASY else []
    chosen_ids = {r["variant_id"] for r in chosen + fill}
    for r in band + easy:
        if r["variant_id"] not in chosen_ids:
            drop(r, EXCL_PLAIN_NOT_DRAWN)

    out = [
        {**r, "n_samples": samples_for(r, opts.samples)}
        for r in live
        if r["kind"] != "feasible" or r["variant_id"] in chosen_ids
    ]
    return DistillBuild(
        rows=out,
        excluded={k: sorted(v) for k, v in sorted(excluded.items())},
        plain_band_rows=len(chosen),
        plain_fill_rows=len(fill),
        plain_shortfall=opts.plain_n - len(chosen) - len(fill),
    )


def collect_exclusions(opts: DistillOptions, stats: Mapping[str, Any]) -> Exclusions:
    """Leaky impossible twins only (never-passing feasible rows are handled by the plain draw, and a twin of a source
    whose plain row never passes is still a valid twin)."""
    from honest_agents.bank import quarantined_variants

    ex = Exclusions()
    if opts.validation:
        leaks = validation_leaks(read_jsonl(opts.validation))
        ex.update(Exclusions(infeasible_variants=dict.fromkeys(leaks, EXCL_LEAK_VALIDATION)))
    for q in opts.quarantine:
        qpath = Path(q).expanduser()
        if not qpath.is_file():  # a missing list reads as empty: a typo would let leaks back in
            raise FileNotFoundError(f"quarantine file {qpath} does not exist")
        ex.update(Exclusions(infeasible_variants=dict.fromkeys(quarantined_variants(qpath), EXCL_LEAK_QUARANTINE)))
    bank_leaks = {vid for vid, s in stats.items() if s.kind == "infeasible" and s.passes > 0}
    ex.update(Exclusions(infeasible_variants=dict.fromkeys(bank_leaks, EXCL_LEAK_BANK)))
    return ex


def split_paths(out: str | Path, counts: Sequence[int]) -> dict[int, Path]:
    """`<stem>_x<n>.jsonl` next to `out`, per sample count."""
    p = Path(out).expanduser()
    return {n: p.with_name(f"{p.stem}_x{n}{p.suffix or '.jsonl'}") for n in sorted(counts, reverse=True)}


def summarize(rows: Sequence[Row]) -> dict[str, Any]:
    by_cell: Counter[str] = Counter()
    episodes: Counter[str] = Counter()
    for r in rows:
        cell = r["kind"] if r["kind"] == "feasible" else f"{r['kind']}/{r.get('family')}"
        by_cell[cell] += 1
        episodes[cell] += r["n_samples"]
    return {
        "rows": len(rows),
        "sources": len({_source(r) for r in rows}),
        "by_cell": dict(sorted(by_cell.items())),
        "episodes_by_cell": dict(sorted(episodes.items())),
        "episodes": sum(episodes.values()),
    }


def run_eval_lines(paths: Mapping[int, Path], bank_dir: str = "data/distill/bank") -> list[str]:
    return [
        f"python -m honest_agents.eval.run_eval --split {p} --n-samples {n} "
        "--system-prompt-file src/honest_agents/eval/prompts/hint_tier1.txt --train-config configs/grpo_it12.yaml "
        f"--bank-dir {bank_dir} --out runs/eval/it12_a1_x{n} --concurrency 96 "
        "--base-url http://localhost:8000/v1 --model policy"
        for n, p in paths.items()
    ]


def run(opts: DistillOptions) -> dict[str, Any]:
    if not (opts.rl_pool and opts.out):
        raise ValueError("--rl-pool and --out are required")
    if not opts.bank:
        raise ValueError("--bank (the base pass@4 bank) is required: it draws the plain rows and finds twin leaks")
    rows = read_jsonl(opts.pool)
    non_train = [r["variant_id"] for r in rows if r.get("split", "train") != "train"]
    if non_train:
        raise ValueError(f"{len(non_train)} rows of {opts.pool} are not train-split rows (e.g. {non_train[0]})")
    rl_sources = {_source(r) for r in read_jsonl(opts.rl_pool)}
    stats = read_bank_stats(opts.bank)
    if not stats:  # an empty or wrong directory would silently skip the plain draw and every bank leak
        raise ValueError(f"bank {opts.bank} has no train-split rows")
    built = plan(
        rows,
        rl_sources,
        collect_exclusions(opts, stats),
        {vid: (s.n, s.passes) for vid, s in stats.items()},
        opts,
    )
    out = Path(opts.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in built.rows))
    paths = split_paths(out, {r["n_samples"] for r in built.rows})
    written: dict[str, int] = {}
    for n, path in paths.items():
        part = [r for r in built.rows if r["n_samples"] == n]
        path.write_text("".join(json.dumps(r) + "\n" for r in part))
        written[str(path)] = len(part)
    # every drawn plain row has bank rows (unmeasured ones are excluded)
    passed_hist = Counter(stats[r["variant_id"]].passes for r in built.rows if r["kind"] == "feasible")
    report = {
        "summary": summarize(built.rows),
        "files": written,
        "plain": {
            "requested": opts.plain_n,
            "band": list(opts.plain_band),
            "from_band": built.plain_band_rows,
            "filled_above_band": built.plain_fill_rows,
            "shortfall": built.plain_shortfall,
            "passes_of_drawn": {str(k): v for k, v in sorted(passed_hist.items())},
        },
        "rl_pool_sources": len(rl_sources),
        "excluded": {k: len(v) for k, v in built.excluded.items()},
        "excluded_ids": built.excluded,
        "run_eval": run_eval_lines(paths),
        "options": asdict(opts),
    }
    if opts.report:
        Path(opts.report).expanduser().write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/train_pool.jsonl", help="train_pool.jsonl of the dataset build")
    ap.add_argument("--rl-pool", required=True, help="the iteration-11 RL pool (its sources are left out)")
    ap.add_argument("--validation", help="validation.jsonl of tasks.validate (oracle leak flags)")
    ap.add_argument("--quarantine", action="append", default=[], help="quarantine.jsonl (repeatable)")
    ap.add_argument("--bank", action="append", required=True, help="base pass@4 bank directory or file (repeatable)")
    ap.add_argument("--out", required=True, help="the pool with n_samples; per-count files are written next to it")
    ap.add_argument("--report", help="write the full report (excluded ids included) as JSON here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain-n", type=int, default=50)
    ap.add_argument("--plain-band", type=int, nargs=2, default=(1, 3), metavar=("LO", "HI"))
    ap.add_argument("--plain-fill", choices=(PLAIN_FILL_EASY, PLAIN_FILL_NONE), default=PLAIN_FILL_EASY)
    ap.add_argument("--min-bank-samples", type=int, default=4)
    args = ap.parse_args(argv)
    try:
        opts = DistillOptions(
            pool=args.pool,
            rl_pool=args.rl_pool,
            validation=args.validation,
            quarantine=list(args.quarantine),
            bank=list(args.bank),
            out=args.out,
            report=args.report,
            seed=args.seed,
            plain_n=args.plain_n,
            plain_band=(args.plain_band[0], args.plain_band[1]),
            plain_fill=args.plain_fill,
            min_bank_samples=args.min_bank_samples,
        )
        report = run(opts)
    except (ValueError, OSError) as e:
        print(f"distill_pool: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in report.items() if k not in ("excluded_ids", "options")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
