"""Cluster-bootstrap intervals for the pilot stop rule.

The unit of resampling is the distinct prompt, not the episode: the G generations of one prompt share one outcome, so
episodes are not independent and counting them overstates the data eight-fold. With several training seeds per cell
the seeds are resampled too, so an interval covers which prompts were drawn and which seeds were run. Everything is
a matrix product over per-prompt count rows, so a draw costs a few microseconds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from honest_agents.eval.protocol import seed_inflation
from honest_agents.pilot import curves

# Per-prompt counts (one row per distinct prompt); every metric below is a ratio of column sums.
COLUMNS = (
    "feasible",
    "claimed",
    "failed",
    "failed_claimed",
    "claim_passed",
    "declined",
    "decline_passed",
    "scored",
    "has_report",
)
_COL = {name: i for i, name in enumerate(COLUMNS)}
MIN_FINITE = 0.9  # an interval is reported only if the metric is defined in this share of the draws


def prompt_counts(records: Iterable[dict[str, Any]]) -> np.ndarray:
    """One row of `COLUMNS` per distinct prompt (`pilot_id`; a record without one is its own prompt)."""
    rows: dict[Any, list[float]] = {}
    for i, r in enumerate(records):
        if r.get("masked"):
            continue
        row = rows.setdefault(r.get("pilot_id") or r.get("variant_id") or ("record", i), [0.0] * len(COLUMNS))
        row[_COL["scored"]] += 1
        row[_COL["has_report"]] += bool(r.get("has_report"))
        if not r.get("feasible"):
            continue
        claimed, passed = bool(r.get("claimed")), bool(r.get("passed"))
        row[_COL["feasible"]] += 1
        if claimed:
            row[_COL["claimed"]] += 1
            row[_COL["claim_passed"]] += passed
        else:
            row[_COL["declined"]] += 1
            row[_COL["decline_passed"]] += passed
        if not passed:
            row[_COL["failed"]] += 1
            row[_COL["failed_claimed"]] += claimed
    return np.array(list(rows.values()), dtype=float).reshape(-1, len(COLUMNS))


def window_counts(records: Sequence[dict[str, Any]], first: bool, n_rounds: int) -> np.ndarray:
    """Prompt counts over the first or the last `n_rounds` rounds (the same windows as `curves.window_summary`)."""
    rounds = curves.by_round(records)
    keys = list(rounds)
    chosen = keys[:n_rounds] if first else keys[-n_rounds:]
    return prompt_counts(r for k in chosen for r in rounds[k])


def n_failed_prompts(counts: np.ndarray) -> int:
    """Distinct prompts with at least one failed feasible episode: what a fabrication rate is estimated from."""
    return int((counts[:, _COL["failed"]] > 0).sum()) if len(counts) else 0


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def metrics_of(totals: np.ndarray) -> dict[str, np.ndarray]:
    """The four stop-rule rates from column sums (last axis = `COLUMNS`); NaN where a denominator is zero."""
    c = {name: totals[..., i] for name, i in _COL.items()}
    return {
        "fabrication": _ratio(c["failed_claimed"], c["failed"]),
        "claim_rate": _ratio(c["claimed"], c["feasible"]),
        "discrimination": _ratio(c["claim_passed"], c["claimed"]) - _ratio(c["decline_passed"], c["declined"]),
        "report_rate": _ratio(c["has_report"], c["scored"]),
    }


@dataclass(frozen=True)
class Interval:
    point: float | None
    lo: float | None
    hi: float | None

    def to_list(self) -> list[float | None]:
        return [self.point, self.lo, self.hi]


def interval(point: float | None, draws: np.ndarray, level: float) -> Interval:
    """Percentile interval of `draws`, undefined (None bounds) unless nearly every draw is defined."""
    finite = draws[np.isfinite(draws)]
    if point is None or not np.isfinite(point) or len(finite) < MIN_FINITE * len(draws) or not len(finite):
        return Interval(None if point is None or not np.isfinite(point) else float(point), None, None)
    a = (1.0 - level) / 2.0
    return Interval(float(point), float(np.quantile(finite, a)), float(np.quantile(finite, 1.0 - a)))


def _multinomial(rng: np.random.Generator, n_prompts: int, n_boot: int) -> np.ndarray:
    return rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=n_boot)


def _seed_mean(points: Sequence[float], cols: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per draw, the seed mean: the mean of the per-seed prompt-bootstrap draws (`cols`, draws × seeds: prompt noise)
    plus the deviation of a resample of the seeds' point values from their mean, widened by `seed_inflation`
    (seed noise; taken from the point values so the prompt noise inside each draw is not counted twice). One seed:
    its own draws."""
    if cols.shape[1] == 1:
        return cols[:, 0]
    pts = np.asarray(points, dtype=float)
    picked = rng.integers(0, len(pts), size=cols.shape)
    return cols.mean(axis=1) + seed_inflation(len(pts)) * (pts[picked].mean(axis=1) - pts.mean())


def cell_draws(
    seeds: Sequence[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[dict[str, float | None], dict[str, np.ndarray]]:
    """Point estimates and bootstrap draws for one (scale, alpha) cell.

    `seeds` holds (first-window, last-window) prompt counts per training seed. Per draw: resample the prompts of every
    seed's two windows (independently: the windows hold different prompts), compute the seed's rates, then resample the
    seeds and average. Returns the cell's `fabrication_first`, `fabrication_last`, `fabrication_drop` and the last
    window's `claim_rate`, `discrimination`, `report_rate`, each as (point, draws of length `n_boot`).
    """
    names = ("fabrication_first", "fabrication_last", "claim_rate", "discrimination", "report_rate")
    point_cols: dict[str, list[float]] = {k: [] for k in names}
    draw_cols: dict[str, list[np.ndarray]] = {k: [] for k in names}
    for first, last in seeds:
        m_first, m_last = metrics_of(first.sum(axis=0)), metrics_of(last.sum(axis=0))
        b_first = metrics_of(_multinomial(rng, len(first), n_boot) @ first) if len(first) else None
        b_last = metrics_of(_multinomial(rng, len(last), n_boot) @ last) if len(last) else None
        nan = np.full(n_boot, np.nan)
        for k, (pt, dr) in {
            "fabrication_first": (m_first["fabrication"], None if b_first is None else b_first["fabrication"]),
            "fabrication_last": (m_last["fabrication"], None if b_last is None else b_last["fabrication"]),
            "claim_rate": (m_last["claim_rate"], None if b_last is None else b_last["claim_rate"]),
            "discrimination": (m_last["discrimination"], None if b_last is None else b_last["discrimination"]),
            "report_rate": (m_last["report_rate"], None if b_last is None else b_last["report_rate"]),
        }.items():
            point_cols[k].append(float(pt))
            draw_cols[k].append(nan if dr is None else dr)
    points = {k: float(np.mean(v)) if np.isfinite(v).all() else None for k, v in point_cols.items()}
    stacked = {k: np.stack(v, axis=1) for k, v in draw_cols.items()}  # n_boot × seeds
    draws = {k: _seed_mean(point_cols[k], v, rng) for k, v in stacked.items()}
    # the drop is formed per seed (same seed in both windows) before the seeds are averaged
    drop_cols = stacked["fabrication_first"] - stacked["fabrication_last"]
    pts = [a - b for a, b in zip(point_cols["fabrication_first"], point_cols["fabrication_last"], strict=True)]
    draws["fabrication_drop"] = _seed_mean(pts, drop_cols, rng)
    points["fabrication_drop"] = float(np.mean(pts)) if np.isfinite(pts).all() else None
    return points, draws


def spread_over_alpha(alphas: Sequence[float], rates: np.ndarray) -> np.ndarray:
    """Fall of the claim rate across the alpha range, from the least-squares slope over all cells:
    -slope * (max alpha - min alpha). `rates` has the cells on the last axis; positive means claims fall with alpha."""
    a = np.asarray(alphas, dtype=float)
    centred = a - a.mean()
    slope = (rates * centred).sum(axis=-1) / (centred**2).sum()
    return -slope * (a.max() - a.min())
