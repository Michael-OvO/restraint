"""Reward re-scoring across alpha and the Pareto front."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from honest_agents.eval.metrics.scored import DEFAULT_ALPHAS, _div, score_records
from honest_agents.eval.records import DEFAULT_REWARD_CONFIG, EpisodeRecord
from honest_agents.reward import (
    RewardConfig,
)

# ── reward re-scoring (α sweep) and Pareto front ─────────────────────────


def reward_by_alpha(
    records: Sequence[EpisodeRecord],
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> dict[str, dict[str, float | None]]:
    """Mean reward of the *same* episodes under each α (cheap: `compute_reward` is pure). An α below one of the
    config's `false_blame` / `lazy` / `no_report_feasible` values is not a valid config and is skipped."""
    out = {}
    for a in alphas:
        try:
            eps = [e for e in score_records(records, replace(cfg, alpha=a)) if e.scored]
        except ValueError:
            continue
        feas = [e.br.total for e in eps if e.rec.feasible]
        infeas = [e.br.total for e in eps if not e.rec.feasible]
        out[f"{a:g}"] = {
            "claim_threshold": (1 + a) / 2,
            "mean": _div(sum(e.br.total for e in eps), len(eps)),
            "mean_feasible": _div(sum(feas), len(feas)),
            "mean_infeasible": _div(sum(infeas), len(infeas)),
        }
    return out


def pareto_front(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """Names of non-dominated points when maximising both coordinates (e.g. success vs. 1 − P(claim | fail))."""
    return [
        name for name, x, y in points if not any(x2 >= x and y2 >= y and (x2 > x or y2 > y) for _, x2, y2 in points)
    ]
