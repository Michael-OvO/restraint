"""Cluster bootstrap intervals and the comparison of two runs on the same split."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from honest_agents.eval.metrics.calibration import _confidence_rows, claim_threshold
from honest_agents.eval.metrics.episode import _TERM_KEYS, METRIC_NAMES, _add, episode_terms, finalize
from honest_agents.eval.metrics.paired import PAIRED_NAMES, finalize_paired, pair_units
from honest_agents.eval.metrics.scored import NON_ADDITIVE, Scored, Terms, leaky_variants, score_records
from honest_agents.eval.protocol import DEFAULT_NI_MARGIN, seed_inflation
from honest_agents.eval.records import DEFAULT_REWARD_CONFIG, EpisodeRecord
from honest_agents.reward import (
    RewardConfig,
)

# ── cluster bootstrap ────────────────────────────────────────────────────


def cluster_terms(eps: Sequence[Scored]) -> dict[str, Terms]:
    """Summed episode and pair-unit terms per bootstrap cluster (the source task)."""
    out: dict[str, Terms] = defaultdict(dict)
    for e in eps:
        _add(out[e.rec.cluster], episode_terms(e))
    for u in pair_units(eps):
        _add(out[u.cluster], u.terms())
    return dict(out)


def _finalize_all(t: Terms) -> dict[str, float | None]:
    return {**finalize(t), **finalize_paired(t)}


def _quantiles(xs: list[float], level: float, n_boot: int) -> tuple[float, float] | None:
    """Percentile interval, reported only if the metric is defined in (almost) every resample."""
    import numpy as np

    if len(xs) < 0.9 * n_boot or not xs:
        return None
    a = (1 - level) / 2
    return float(np.quantile(xs, a)), float(np.quantile(xs, 1 - a))


def _cluster_weights(n_clusters: int, n_boot: int, seed: int):
    """n_boot × clusters multinomial counts: how often each source task is drawn in each resample."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)


def _boot_auroc(rows: list[tuple[str, float, bool, bool]], index: dict[str, int], weights) -> list[float]:
    """AUROC of stated confidence in every resample.

    With cluster weights w the statistic is wᵀKw / ((w·P)(w·N)), where K[c, d] sums the pairwise kernel between the
    positives of cluster c and the negatives of cluster d, so no resample needs the episodes again.
    """
    import numpy as np

    c_of = np.array([index[r[0]] for r in rows], dtype=int)
    s = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=bool)
    n_c = len(index)
    if not y.any() or y.all():
        return []
    pos_s, pos_c, neg_s, neg_c = s[y], c_of[y], s[~y], c_of[~y]
    neg_onehot = np.zeros((len(neg_s), n_c), dtype=np.float32)
    neg_onehot[np.arange(len(neg_s)), neg_c] = 1.0
    kernel = np.zeros((n_c, n_c))
    for start in range(0, len(pos_s), 1024):  # chunked: the pairwise matrix is positives × negatives
        chunk = pos_s[start : start + 1024, None]
        k = (chunk > neg_s[None, :]).astype(np.float32) + 0.5 * (chunk == neg_s[None, :]).astype(np.float32)
        np.add.at(kernel, pos_c[start : start + 1024], k @ neg_onehot)
    n_pos = weights @ np.bincount(pos_c, minlength=n_c)
    n_neg = weights @ np.bincount(neg_c, minlength=n_c)
    num = np.einsum("bc,cd,bd->b", weights, kernel, weights)
    ok = (n_pos > 0) & (n_neg > 0)
    return [float(x) for x in num[ok] / (n_pos[ok] * n_neg[ok])]


def bootstrap(
    eps: Sequence[Scored],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float] | None]:
    """Percentile CIs for every metric, resampling source tasks with replacement."""
    import numpy as np

    clusters = cluster_terms(eps)
    names = sorted(clusters)
    if not names or n_boot <= 0:
        return {}
    index = {c: i for i, c in enumerate(names)}
    keys = sorted({k for t in clusters.values() for k in t})
    mat = np.array([[clusters[c].get(k, 0.0) for k in keys] for c in names])  # clusters × terms
    weights = _cluster_weights(len(names), n_boot, seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for row in weights @ mat:
        for k, v in _finalize_all(dict(zip(keys, row, strict=True))).items():
            if v is not None:
                samples[k].append(float(v))
    feas = _confidence_rows(eps, feasible_only=True)
    every = _confidence_rows(eps, feasible_only=False)
    samples["auroc_confidence"] = _boot_auroc(feas, index, weights) if feas else []
    samples["auroc_confidence_all"] = _boot_auroc(every, index, weights) if every else []
    if every:
        c_of = np.array([index[r[0]] for r in every], dtype=int)
        conf, claimed = [r[1] for r in every], [r[3] for r in every]
        for w in weights:
            cut = claim_threshold(conf, claimed, w[c_of])
            if cut is not None:
                samples["diagnostic_confidence_cut"].append(cut[0])
                samples["diagnostic_confidence_cut_agreement"].append(cut[1])
    return {k: _quantiles(samples.get(k, []), level, n_boot) for k in (*METRIC_NAMES, *PAIRED_NAMES, *NON_ADDITIVE)}


# ── comparing two runs on the same split ─────────────────────────────────


def _as_runs(x: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]]) -> list[list[EpisodeRecord]]:
    """One arm as a list of seed runs: a flat list of records is a single seed."""
    if not x or isinstance(x[0], EpisodeRecord):
        return [list(x)]  # type: ignore[arg-type]
    return [list(r) for r in x]  # type: ignore[union-attr]


def _seed_sd(values: list[float]) -> float | None:
    import numpy as np

    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def paired_difference(
    candidate: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]],
    reference: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]],
    metric: str = "verified_success_rate",
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """candidate − reference on `metric`, with a bootstrap over the source tasks all runs evaluated.

    Each arm is one run (a flat record list) or several training seeds (a list of record lists). The two arms are
    resampled with the *same* source-task draws in every run, so task difficulty cancels. With several seeds the seeds
    of each arm are also resampled inside every draw and the arm value is the mean over the drawn seeds, so the
    interval covers which tasks were drawn and which seeds were run (SP2). The seed component is widened by
    `protocol.seed_inflation` (a bootstrap of 2-3 seeds understates seed variance); read `seed_sd_*` and size the
    margin on them. Variants
    that leaked in any run are void in all, so the rates share their denominators' population. `metric` is any
    additive, derived or paired metric (not the AUROC of confidence or the diagnostic confidence cut).
    """
    import numpy as np

    arms = [_as_runs(candidate), _as_runs(reference)]
    scored = [[score_records(r, cfg) for r in runs] for runs in arms]
    void = set().union(*(leaky_variants(e) for runs in scored for e in runs))
    terms = [[cluster_terms(score_records(r, cfg, exclude_variants=void)) for r in runs] for runs in arms]
    per_arm = [set().union(*map(set, runs)) for runs in terms]
    shared = sorted(set.intersection(*(set(t) for runs in terms for t in runs)))
    out: dict[str, Any] = {
        "metric": metric,
        "n_source_tasks": len(shared),
        "n_dropped_candidate_only": len(per_arm[0] - per_arm[1]),
        "n_dropped_reference_only": len(per_arm[1] - per_arm[0]),
        "n_seeds_candidate": len(terms[0]),
        "n_seeds_reference": len(terms[1]),
        "void_variants": sorted(void),
        "candidate": None,
        "reference": None,
        "seed_values_candidate": [],
        "seed_values_reference": [],
        "seed_sd_candidate": None,
        "seed_sd_reference": None,
        "difference": None,
        "ci": None,
        "level": level,
        "samples": [],
    }
    if not shared:
        return out
    mats, keysets = [], []  # one (clusters × terms) matrix per run, arms flattened candidate first
    for run in (t for runs in terms for t in runs):
        keys = sorted({k for c in shared for k in run[c]})
        keysets.append(keys)
        mats.append(np.array([[run[c].get(k, 0.0) for k in keys] for c in shared]))
    n_c = len(terms[0])

    def value(row, keys) -> float | None:
        v = _finalize_all(dict(zip(keys, row, strict=True)))
        if metric not in v:
            raise KeyError(f"unknown or non-additive metric {metric!r}")
        return v[metric]

    point = [value(m.sum(axis=0), k) for m, k in zip(mats, keysets, strict=True)]
    if any(p is None for p in point):
        return out
    out["seed_values_candidate"], out["seed_values_reference"] = point[:n_c], point[n_c:]
    out["seed_sd_candidate"], out["seed_sd_reference"] = _seed_sd(point[:n_c]), _seed_sd(point[n_c:])
    out["candidate"], out["reference"] = float(np.mean(point[:n_c])), float(np.mean(point[n_c:]))
    out["difference"] = out["candidate"] - out["reference"]
    if n_boot <= 0:
        return out
    weights = _cluster_weights(len(shared), n_boot, seed)
    nan = float("nan")

    def draw_values(resampled: np.ndarray, keys: list[str]) -> np.ndarray:
        """`value` of every resampled term row (NaN where undefined)."""
        if metric in _TERM_KEYS:
            # a plain ratio of two summed terms: one vectorised division (IEEE, so the same floats as `_div` per row)
            col = {k: i for i, k in enumerate(keys)}
            num_key, den_key = _TERM_KEYS[metric]
            zeros = np.zeros(len(resampled))
            num = resampled[:, col[num_key]] if num_key in col else zeros
            den = resampled[:, col[den_key]] if den_key in col else zeros
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(den != 0, num / den, nan)
        return np.array([nan if (v := value(row, keys)) is None else v for row in resampled])

    draws = np.array([draw_values(weights @ m, k) for m, k in zip(mats, keysets, strict=True)]).T  # n_boot × runs
    seed_rng = np.random.default_rng(seed + 1)  # its own stream: one-seed results do not depend on it

    def arm_value(cols: np.ndarray, pts: list[float]) -> np.ndarray:
        if cols.shape[1] == 1:
            return cols[:, 0]
        # task noise: the mean of the runs' task-bootstrap values; seed noise: a resample of the runs' point values,
        # widened for few seeds and taken from the points so task noise is not counted twice
        picked = seed_rng.integers(0, len(pts), size=cols.shape)
        return cols.mean(axis=1) + seed_inflation(len(pts)) * (np.asarray(pts)[picked].mean(axis=1) - np.mean(pts))

    diffs = arm_value(draws[:, :n_c], point[:n_c]) - arm_value(draws[:, n_c:], point[n_c:])
    diffs = diffs[~np.isnan(diffs)]
    out["samples"] = [float(d) for d in diffs]
    ci = _quantiles(out["samples"], level, n_boot)
    out["ci"] = list(ci) if ci else None
    return out


def noninferiority(
    candidate: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]],
    reference: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]],
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    *,
    margin: float = DEFAULT_NI_MARGIN,
    metric: str = "verified_success_rate",
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: int = 0,
    allow_single_seed: bool = False,
) -> dict[str, Any]:
    """One-sided non-inferiority test of `candidate` against `reference` on `metric` (higher is better).

    H0: candidate − reference ≤ −margin. Rejected (non-inferior) when the one-sided (1 − alpha) lower bound of the
    difference, the alpha-quantile of the paired bootstrap (source tasks, and training seeds when each arm has
    several; see `paired_difference`), is above −margin. The margin belongs to the evaluation protocol and is fixed
    before the run (`honest_agents.eval.protocol`); the primary metric is verified success, not task success (SP1).

    Non-inferiority is **not declared from a single seed** unless `allow_single_seed` (the protocol's explicit
    `ni_allow_single_seed`): the interval of one run per arm leaves seed variance out, so `non_inferior` is None and
    `refusal` says why, while `lower_bound` stays the task-sampling-only bound.
    """
    import numpy as np

    d = paired_difference(candidate, reference, metric, cfg, n_boot=n_boot, seed=seed, level=1 - 2 * alpha)
    samples = d.pop("samples")
    lower = float(np.quantile(samples, alpha)) if len(samples) >= 0.9 * n_boot and samples else None
    single = min(d["n_seeds_candidate"], d["n_seeds_reference"]) < 2
    refusal = (
        "one training seed per arm leaves seed variance out of the interval: supply several seeds per arm, or fix "
        "`ni_allow_single_seed` in the protocol before the run"
        if single and not allow_single_seed
        else None
    )
    return {
        **d,
        "margin": margin,
        "alpha": alpha,
        "lower_bound": lower,
        "single_seed": single,
        "refusal": refusal,
        "non_inferior": None if lower is None or refusal else bool(lower > -margin),
        "p_value": float(np.mean(np.asarray(samples) <= -margin)) if samples else None,
    }
