"""Difficulty gate from a Phase-0 trajectory bank (SPEC_V3 §4.8; INC-5, LV-9).

    python -m honest_agents.tasks.difficulty --bank runs/phase0/bank --out data/difficulty.json

Phase 0 (`configs/phase0_zero_lr.yaml`) runs the real trainer at learning rate 0, so every prompt group is `n` rollouts of
the base policy on one variant, written to the trajectory bank. This module reads that bank and

* **keeps feasible tasks with 1 to 7 passes of 8**: an all-pass group has no contrast and an all-fail group teaches
  nothing about success. LV-9 adjustments: the filter applies to the **training split only**, a capped share
  (`zero_share`, default 15%) of 0-of-8 tasks stays because they are where honest reporting matters most, and the
  verdict is per source task (`pair_id`): the plain feasible variant decides, its controls and infeasible variants stay;
* **measures the infeasible rows of the same run** and reports the leaks: variants whose tests pass cleanly although
  they are labelled infeasible (a leak is the label being wrong, GR-7, LV-1), optionally appended to the quarantine list;
* **reports which alpha values are safe for the pool's mean pass rate** (INC-5). A claim pays 2 + lambda on success and
  -1 on failure while a non-claim pays alpha, so a calibrated policy claims iff its success probability exceeds
  `(1 + alpha) / 2`. If the pool's mean pass rate is below that break-even, blanket quitting wins in expectation and the
  policy learns it within a few steps (arXiv 2608.00301). For each alpha the table gives the break-even, whether the
  mean pass rate clears it, the predicted claim recall of a calibrated policy whose confidence has the measured AUROC
  (equal-variance Gaussian model: successes ~ N(d', 1), failures ~ N(0, 1)), and the smallest lower band edge that
  would make the pool clear the break-even. "Safe" needs both: the break-even below the mean pass rate and a predicted
  recall of at least 30% of true successes.

The AUROC of stated confidence against pass is measured from the bank when it holds enough reports, otherwise assumed
(d' = 1, AUROC 0.76) and flagged as assumed in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

from honest_agents.bank import (
    QUARANTINE_LEAK,
    BankRow,
    QuarantineEntry,
    append_quarantine,
    iter_bank,
)
from honest_agents.reward import MASK_BOOT_FAILURE, MASK_REWARD_ERROR, MASK_TOOLCHAIN

log = logging.getLogger(__name__)

DEFAULT_ROLLOUTS = 8
DEFAULT_BAND = (1, 7)
DEFAULT_ZERO_SHARE = 0.15
DEFAULT_ALPHAS = (0.0, 0.1, 0.25, 0.4, 0.5, 0.75)
MIN_RECALL = 0.30  # INC-5: run an alpha only if a calibrated policy would still claim about 30% of its true successes
ASSUMED_AUROC = 0.76  # d' = 1: the benign case INC-5 simulates when confidence has not been measured
MIN_AUROC_REPORTS = 40  # fewer scored reports than this and the AUROC is too noisy to use
MIN_AUROC_PER_CLASS = 8
_INFRA_MASKS = frozenset({MASK_BOOT_FAILURE, MASK_TOOLCHAIN, MASK_REWARD_ERROR})

STATUS_KEEP = "keep"
STATUS_TOO_EASY = "too_easy"  # every rollout passed
STATUS_TOO_HARD = "too_hard"  # no rollout passed and the zero-pass quota is used up
STATUS_ZERO_KEPT = "zero_kept"  # no rollout passed but kept as part of the capped zero-pass share
STATUS_UNMEASURED = "unmeasured"  # too few rollouts to tell


@dataclass
class VariantStat:
    variant_id: str
    pair_id: str
    kind: str  # feasible | control | infeasible
    family: str | None
    n: int = 0  # scored episodes (infrastructure failures excluded)
    passes: int = 0  # clean passes: tests passed, no violation, no tamper
    claimed_passes: int = 0  # passes that ended in a completion claim (infeasible: the reward's `leak`)
    n_infra: int = 0

    @property
    def rate(self) -> float:
        return self.passes / self.n if self.n else 0.0


@dataclass
class AlphaRow:
    alpha: float
    break_even: float  # (1 + alpha) / 2
    clears_break_even: bool  # pool mean pass rate > break-even
    predicted_claim_recall: float  # share of true successes a calibrated policy claims, at the measured d'
    safe: bool
    suggested_min_pass: int | None  # smallest lower band edge (of n) whose pool clears the break-even; None = none does


@dataclass
class DifficultyReport:
    n_rollouts: int
    band: tuple[int, int]
    zero_share: float
    feasible: dict[str, Any] = field(default_factory=dict)
    infeasible: dict[str, Any] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)
    alpha: dict[str, Any] = field(default_factory=dict)
    keep_pairs: list[str] = field(default_factory=list)
    drop_pairs: dict[str, str] = field(default_factory=dict)  # pair_id -> status
    unmeasured_pairs: list[str] = field(default_factory=list)
    pair_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    non_train_rows_skipped: dict[str, int] = field(default_factory=dict)  # split -> rows ignored (L6)

    @property
    def safe_alphas(self) -> list[float]:
        return [r["alpha"] for r in self.alpha.get("table", []) if r["safe"]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["band"] = list(self.band)
        return d


# ── reading the bank ─────────────────────────────────────────────────────────


def _kind(task: dict[str, Any], spec: dict[str, Any], feasible: bool) -> str:
    kind = task.get("kind")
    if kind in ("feasible", "control", "infeasible"):
        return kind
    if not feasible:
        return "infeasible"
    return "control" if task.get("is_control", spec.get("is_control", False)) else "feasible"


def _row_split(row: BankRow) -> str:
    task = row.task or {}
    split = task.get("split")
    if split is None:
        raise MissingSplitError(
            f"bank row for {task.get('variant_id') or row.group!r} has no split column; the difficulty gate only "
            "accepts rows from train_pool.jsonl (split == 'train')"
        )
    return str(split)


class MissingSplitError(ValueError):
    """A bank row whose task has no `split` column: the gate cannot tell training rows from validation or test rows."""


def variant_stats(rows: Iterable[BankRow], *, skipped: Counter[str] | None = None) -> dict[str, VariantStat]:
    """One `VariantStat` per variant: episodes, clean passes, passes that were claimed. Infra failures are not episodes.

    Only `split == "train"` rows count (L6): the gate decides what the training split may use, so a validation or test
    row in a bank must not shape it. A row with no split column is an operator mistake (a bank not made from
    `train_pool.jsonl`) and raises `MissingSplitError`. Skipped non-train rows are tallied per split in `skipped`."""
    stats: dict[str, VariantStat] = {}
    for row in rows:
        o = row.episode_outcome()
        task = row.task or {}
        spec = task.get("spec") or {}
        split = _row_split(row)
        if split != "train":
            if skipped is not None:
                skipped[str(split)] += 1
            continue
        vid = task.get("variant_id") or spec.get("variant_id") or row.group or task.get("task_dir")
        if not vid:
            continue
        stat = stats.get(vid)
        if stat is None:
            stat = stats[vid] = VariantStat(
                variant_id=vid,
                pair_id=task.get("pair_id") or spec.get("pair_id") or vid,
                kind=_kind(task, spec, o.feasible),
                family=task.get("family") or spec.get("family"),
            )
        if o.mask in _INFRA_MASKS or (o.grading_error is not None and not o.acted):
            stat.n_infra += 1
            continue
        stat.n += 1
        if o.passed:
            stat.passes += 1
            if o.report is not None and o.report.claims_complete:
                stat.claimed_passes += 1
    return stats


def confidence_pairs(rows: Iterable[BankRow]) -> list[tuple[float, bool]]:
    """(stated confidence, passed) of every feasible episode whose report gave a confidence."""
    out = []
    for row in rows:
        if _row_split(row) != "train":
            continue
        o = row.episode_outcome()
        if not o.feasible or o.report is None or o.report.confidence is None:
            continue
        if o.mask in _INFRA_MASKS:
            continue
        out.append((float(o.report.confidence), bool(o.passed)))
    return out


def auroc(pairs: Sequence[tuple[float, bool]]) -> float | None:
    """P(confidence of a pass > confidence of a fail), ties counting one half. None when a class is too small."""
    pos = [c for c, y in pairs if y]
    neg = [c for c, y in pairs if not y]
    if len(pos) < MIN_AUROC_PER_CLASS or len(neg) < MIN_AUROC_PER_CLASS:
        return None
    ranks: dict[float, list[int]] = defaultdict(list)
    for i, (c, _) in enumerate(sorted(pairs, key=lambda p: p[0])):
        ranks[c].append(i + 1)
    mean_rank = {c: sum(r) / len(r) for c, r in ranks.items()}
    u = sum(mean_rank[c] for c in pos) - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# ── alpha safety (INC-5) ─────────────────────────────────────────────────────

_NORM = NormalDist()


def d_prime(auc: float) -> float:
    """Separation of two equal-variance Gaussians with this AUROC."""
    a = min(max(auc, 1e-6), 1 - 1e-6)
    return math.sqrt(2) * _NORM.inv_cdf(a)


def claim_recall(success_rate: float, alpha: float, d: float) -> float:
    """Share of true successes a calibrated policy claims.

    The policy claims when its posterior success probability exceeds `t = (1 + alpha) / 2`. With signal x ~ N(d, 1) on
    successes and N(0, 1) on failures the posterior exceeds `t` iff `x > d/2 + ln((1-s)/s * t/(1-t)) / d`, and recall
    is the success-conditional mass above that point. Checked against INC-5: d' = 1, alpha = 0.5, s = 0.3 gives 7%.
    """
    s = min(max(success_rate, 1e-9), 1 - 1e-9)
    t = (1 + alpha) / 2
    if t >= 1:
        return 0.0
    if d <= 1e-6:  # no signal: the posterior is the prior, so claim everything or nothing
        return 1.0 if s > t else 0.0
    x_star = d / 2 + math.log((1 - s) / s * t / (1 - t)) / d
    return 1.0 - _NORM.cdf(x_star - d)


def alpha_table(
    stats: Sequence[VariantStat],
    kept: Sequence[VariantStat],
    *,
    d: float,
    n_rollouts: int,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    min_recall: float = MIN_RECALL,
) -> list[AlphaRow]:
    mean = sum(v.rate for v in kept) / len(kept) if kept else 0.0
    rows = []
    for a in alphas:
        t = (1 + a) / 2
        recall = claim_recall(mean, a, d) if kept else 0.0
        suggested = None
        for lo in range(1, n_rollouts):
            band = [v for v in stats if v.n and lo <= round(v.rate * n_rollouts) <= n_rollouts - 1]
            if band and sum(v.rate for v in band) / len(band) > t:
                suggested = lo
                break
        rows.append(AlphaRow(a, t, mean > t, recall, mean > t and recall >= min_recall, suggested))
    return rows


# ── the gate ─────────────────────────────────────────────────────────────────


def _hash_order(key: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:12], 16)


def difficulty_report(
    rows: Iterable[BankRow],
    *,
    n_rollouts: int = DEFAULT_ROLLOUTS,
    band: tuple[int, int] = DEFAULT_BAND,
    zero_share: float = DEFAULT_ZERO_SHARE,
    min_episodes: int | None = None,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seed: int = 0,
) -> DifficultyReport:
    """Classify every source task of a Phase-0 bank (see the module docstring)."""
    rows = list(rows)
    lo, hi = band
    if not 0 <= lo <= hi <= n_rollouts:
        raise ValueError(f"band {band} does not fit in {n_rollouts} rollouts")
    if not 0 <= zero_share < 1:
        raise ValueError("zero_share must be in [0, 1)")
    min_episodes = min_episodes if min_episodes is not None else max(1, n_rollouts // 2)
    skipped: Counter[str] = Counter()
    stats = variant_stats(rows, skipped=skipped)
    report = DifficultyReport(n_rollouts=n_rollouts, band=band, zero_share=zero_share)
    report.non_train_rows_skipped = dict(skipped)

    def scaled(v: VariantStat) -> float:
        """Passes out of `n_rollouts`, for a variant measured with a different number of episodes."""
        return v.rate * n_rollouts

    plain = [v for v in stats.values() if v.kind == "feasible"]
    status: dict[str, str] = {}
    zero_pairs = []
    for v in sorted(plain, key=lambda v: v.pair_id):
        if v.n < min_episodes:
            status[v.pair_id] = STATUS_UNMEASURED
        elif v.passes == 0:
            zero_pairs.append(v)
        elif scaled(v) >= n_rollouts - 1e-9 or v.passes == v.n:
            status[v.pair_id] = STATUS_TOO_EASY
        elif lo - 1e-9 <= scaled(v) <= hi + 1e-9:
            status[v.pair_id] = STATUS_KEEP
        else:
            status[v.pair_id] = STATUS_TOO_EASY if scaled(v) > hi else STATUS_TOO_HARD
    # Capped share of zero-pass tasks: z of the final pool, so z / (1 - z) of the band-kept count, in hash order.
    n_band = sum(s == STATUS_KEEP for s in status.values())
    quota = int(zero_share / (1 - zero_share) * n_band + 1e-9) if zero_share else 0
    zero_pairs.sort(key=lambda v: _hash_order(v.pair_id, seed))
    for i, v in enumerate(zero_pairs):
        status[v.pair_id] = STATUS_ZERO_KEPT if i < quota else STATUS_TOO_HARD

    kept = [v for v in plain if status.get(v.pair_id) in (STATUS_KEEP, STATUS_ZERO_KEPT)]
    measured = [v for v in plain if status.get(v.pair_id) != STATUS_UNMEASURED]
    report.keep_pairs = sorted(v.pair_id for v in kept)
    report.unmeasured_pairs = sorted(v.pair_id for v in plain if status[v.pair_id] == STATUS_UNMEASURED)
    report.drop_pairs = {
        v.pair_id: status[v.pair_id] for v in plain if status[v.pair_id] in (STATUS_TOO_EASY, STATUS_TOO_HARD)
    }
    report.pair_status = {
        v.pair_id: {"status": status[v.pair_id], "passes": v.passes, "n": v.n, "variant_id": v.variant_id}
        for v in plain
    }
    histogram = Counter(round(scaled(v)) for v in measured)
    report.feasible = {
        "n_variants": len(plain),
        "n_measured": len(measured),
        "n_kept": len(kept),
        "n_zero_kept": sum(status[v.pair_id] == STATUS_ZERO_KEPT for v in plain),
        "n_too_easy": sum(status[v.pair_id] == STATUS_TOO_EASY for v in plain),
        "n_too_hard": sum(status[v.pair_id] == STATUS_TOO_HARD for v in plain),
        "n_unmeasured": len(report.unmeasured_pairs),
        "mean_pass_rate_all": _mean([v.rate for v in measured]),
        "mean_pass_rate_kept": _mean([v.rate for v in kept]),
        "histogram": {str(k): histogram.get(k, 0) for k in range(n_rollouts + 1)},
    }

    ctrl = [v for v in stats.values() if v.kind == "control"]
    report.controls = {
        "n_variants": len(ctrl),
        "mean_pass_rate": _mean([v.rate for v in ctrl if v.n]),
        "by_family": _by_family(ctrl),
    }

    inf = [v for v in stats.values() if v.kind == "infeasible"]
    leaks = [v for v in inf if v.passes > 0]
    report.infeasible = {
        "n_variants": len(inf),
        "n_measured": sum(v.n > 0 for v in inf),
        "n_with_clean_pass": len(leaks),
        "leak_rate": len(leaks) / len(inf) if inf else None,
        "leaks": [
            {
                "variant_id": v.variant_id,
                "pair_id": v.pair_id,
                "family": v.family,
                "passes": v.passes,
                "n": v.n,
                "claimed": v.claimed_passes,
            }
            for v in sorted(leaks, key=lambda v: v.variant_id)
        ],
        "by_family": _by_family(inf),
    }

    pairs = confidence_pairs(rows)
    measured_auc = auroc(pairs) if len(pairs) >= MIN_AUROC_REPORTS else None
    use_auc = measured_auc if measured_auc is not None else ASSUMED_AUROC
    d = d_prime(use_auc)
    table = alpha_table(measured, kept, d=d, n_rollouts=n_rollouts, alphas=alphas)
    report.alpha = {
        "auroc": use_auc,
        "auroc_assumed": measured_auc is None,
        "n_confidence_reports": len(pairs),
        "d_prime": d,
        "mean_pass_rate": _mean([v.rate for v in kept]),
        "min_recall": MIN_RECALL,
        "table": [asdict(r) for r in table],
        "safe_alphas": [r.alpha for r in table if r.safe],
    }
    return report


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _by_family(vs: Sequence[VariantStat]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for fam in sorted({v.family or "unknown" for v in vs}):
        mine = [v for v in vs if (v.family or "unknown") == fam]
        out[fam] = {
            "n": len(mine),
            "with_clean_pass": sum(v.passes > 0 for v in mine),
            "mean_pass_rate": _mean([v.rate for v in mine if v.n]),
        }
    return out


def quarantine_entries(report: DifficultyReport, *, source: str = "phase0") -> list[QuarantineEntry]:
    """Quarantine entries for the infeasible variants whose tests passed cleanly in the measured run."""
    return [
        QuarantineEntry(
            variant_id=leak["variant_id"],
            pair_id=leak["pair_id"],
            family=leak["family"],
            perturbation=leak["family"],
            kind=QUARANTINE_LEAK,
            source=source,
            detail=f"{leak['passes']}/{leak['n']} clean passes ({leak['claimed']} claimed) in the difficulty run",
        )
        for leak in report.infeasible.get("leaks", [])
    ]


# ── consuming the gate (build.py) ────────────────────────────────────────────


@dataclass(frozen=True)
class DifficultyGate:
    """Which source tasks the training split may use."""

    keep: frozenset[str]
    drop: frozenset[str]
    unmeasured: frozenset[str] = frozenset()

    def admits(self, pair_id: str, *, strict: bool = False) -> bool:
        """Dropped tasks are out; kept and not-measured tasks stay (no evidence against them).

        With `strict` only tasks the Phase-0 run measured and kept are admitted. Phase 0 sees a sample of the pool, so a
        task it never measured has no 1-7-of-8 verdict and may be exactly the all-pass or all-fail kind LV-9 removes.
        """
        if strict:
            return pair_id in self.keep
        return pair_id not in self.drop

    @classmethod
    def load(cls, path: str | Path) -> DifficultyGate:
        d = json.loads(Path(path).read_text())
        return cls(
            keep=frozenset(d.get("keep_pairs", [])),
            drop=frozenset(d.get("drop_pairs", {})),
            unmeasured=frozenset(d.get("unmeasured_pairs", [])),
        )


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", type=Path, nargs="+", required=True, help="bank directories or bank-*.jsonl files")
    ap.add_argument(
        "--out", type=Path, required=True, help="difficulty.json (consumed by tasks.build --difficulty-json)"
    )
    ap.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS, help="episodes per prompt group (default 8)")
    ap.add_argument("--min-pass", type=int, default=DEFAULT_BAND[0])
    ap.add_argument("--max-pass", type=int, default=DEFAULT_BAND[1])
    ap.add_argument(
        "--zero-share", type=float, default=DEFAULT_ZERO_SHARE, help="share of the final pool that may be 0-of-n tasks"
    )
    ap.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write-quarantine", type=Path, help="append infeasible variants with a clean pass to this quarantine file"
    )
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = (row for b in a.bank for row in iter_bank(b))
    report = difficulty_report(
        rows,
        n_rollouts=a.rollouts,
        band=(a.min_pass, a.max_pass),
        zero_share=a.zero_share,
        alphas=a.alphas,
        seed=a.seed,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report.to_dict(), indent=2))
    if a.write_quarantine:
        entries = quarantine_entries(report)
        append_quarantine(entries, a.write_quarantine)
        print(f"quarantined {len(entries)} leaking infeasible variants -> {a.write_quarantine}")
    f, i, al = report.feasible, report.infeasible, report.alpha
    print(
        f"feasible: {f['n_kept']}/{f['n_variants']} kept ({f['n_zero_kept']} zero-pass, {f['n_too_easy']} too easy, "
        f"{f['n_too_hard']} too hard, {f['n_unmeasured']} unmeasured); mean pass rate kept {f['mean_pass_rate_kept']}"
    )
    print(f"infeasible: {i['n_with_clean_pass']}/{i['n_variants']} variants with a clean pass (leaks)")
    print(
        f"AUROC {al['auroc']:.2f} ({'assumed' if al['auroc_assumed'] else 'measured'}); "
        f"safe alphas for the pool: {al['safe_alphas'] or 'none'}"
    )
    for r in al["table"]:
        print(
            f"  alpha {r['alpha']:<5} break-even {r['break_even']:.3f}  clears {r['clears_break_even']!s:<5} "
            f"recall {r['predicted_claim_recall']:.2f}  safe {r['safe']}  min_pass>= {r['suggested_min_pass']}"
        )


if __name__ == "__main__":
    main()
