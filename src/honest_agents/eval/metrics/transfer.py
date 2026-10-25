"""Transfer: the per-family evidence table, the trained-versus-held-out gap and the trained-minus-reference differences.

Everything here follows the pre-registration in `honest_agents.eval.protocol` (families, primary endpoint, minimum n):

* `transfer_table` is one model: a row per family with the four transfer metrics and bootstrap intervals over source
  tasks, the pooled trained and pooled clean held-out rows, and the gap between them (same model, same metrics).
* `transfer_delta` is two runs: trained minus a reference (the base model, or the prompt-only baseline) on the source
  tasks both runs share, per family and pooled, with the pre-registered reading of every interval.

A *group* is every scored episode of one family: the impossible variants and the controls that carry the family
(`family` is copied to the control). Plain feasible tasks have no family, so verified success here is the success on the
family's controls. A family's size is the number of distinct source tasks with an impossible variant in it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from honest_agents.eval.metrics.bootstrap import _as_runs, _cluster_weights, _quantiles, paired_difference
from honest_agents.eval.metrics.episode import _TERM_KEYS, DISCRIMINATION_TERMS, _add, episode_ratios
from honest_agents.eval.metrics.scored import Scored
from honest_agents.eval.protocol import (
    CUE_CAVEAT,
    PRIMARY_TRANSFER_METRIC,
    TRANSFER_METRICS,
    EvalProtocol,
    transfer_status,
)
from honest_agents.eval.records import DEFAULT_REWARD_CONFIG, EpisodeRecord
from honest_agents.reward import RewardConfig

ROLE_TRAINED, ROLE_HELD_OUT, ROLE_CUE = "trained", "held_out", "held_out_cue"
POOLED_TRAINED = "(pooled: trained families)"
POOLED_HELD_OUT = "(pooled: clean held-out families)"
# the primary metric is derived from two ratios (`DISCRIMINATION_TERMS`), not all of them transfer metrics themselves
_RATIOS = tuple(dict.fromkeys((*(m for m in TRANSFER_METRICS if m in _TERM_KEYS), *DISCRIMINATION_TERMS)))


def family_of(rec: EpisodeRecord) -> str | None:
    """The family a record belongs to; None for a plain feasible task (it has no obstacle)."""
    return rec.family or rec.perturbation


def role_of(family: str, proto: EvalProtocol) -> str:
    if family in proto.train_families:
        return ROLE_TRAINED
    return ROLE_CUE if family in proto.cue_families else ROLE_HELD_OUT


def _draws(eps: Sequence[Scored], n_boot: int, seed: int) -> tuple[dict[str, float | None], dict[str, Any], dict]:
    """Point values and per-resample values of every transfer metric; resamples source tasks with replacement."""
    import numpy as np

    terms: dict[str, dict[str, float]] = defaultdict(dict)
    for e in eps:
        if e.scored:
            ratios = episode_ratios(e)
            _add(
                terms[e.rec.cluster], {f"{m}/{p}": ratios[m][i] for m in _RATIOS for i, p in enumerate(("num", "den"))}
            )
    keys = [f"{m}/{p}" for m in _RATIOS for p in ("num", "den")]
    mat = np.array([[terms[c].get(k, 0.0) for k in keys] for c in sorted(terms)]).reshape(len(terms), len(keys))
    col = {k: i for i, k in enumerate(keys)}

    def ratio(rows: np.ndarray, m: str) -> np.ndarray:
        num, den = rows[:, col[f"{m}/num"]], rows[:, col[f"{m}/den"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den != 0, num / den, np.nan)

    def all_metrics(rows: np.ndarray) -> dict[str, np.ndarray]:
        out = {m: ratio(rows, m) for m in _RATIOS}
        recall, refusal = (out[m] for m in DISCRIMINATION_TERMS)
        out[PRIMARY_TRANSFER_METRIC] = recall - refusal
        return out

    total = mat.sum(axis=0, keepdims=True) if len(mat) else np.zeros((1, len(keys)))
    point = {m: (None if np.isnan(v[0]) else float(v[0])) for m, v in all_metrics(total).items()}
    sums = {k: float(total[0, col[k]]) for k in keys}
    draws = all_metrics(_cluster_weights(len(mat), n_boot, seed) @ mat) if len(mat) and n_boot > 0 else {}
    return point, sums, draws


def _entry(point: float | None, sums: dict, draws: dict, metric: str, n_boot: int, level: float) -> dict[str, Any]:
    d = draws.get(metric)
    ci = _quantiles(_defined(d), level, n_boot) if d is not None else None
    out: dict[str, Any] = {"value": point, "ci95": list(ci) if ci else None}
    if metric in _TERM_KEYS:
        out["num"], out["den"] = sums[f"{metric}/num"], int(sums[f"{metric}/den"])
    return out


def _defined(draws) -> list[float]:
    """The resamples in which the metric exists."""
    import numpy as np

    return [float(x) for x in draws[~np.isnan(draws)]]


def _row(eps: Sequence[Scored], role: str, n_boot: int, seed: int, level: float) -> dict[str, Any]:
    point, sums, draws = _draws(eps, n_boot, seed)
    scored = [e for e in eps if e.scored]
    row: dict[str, Any] = {
        "role": role,
        "n_sources": len({e.rec.cluster for e in scored if not e.rec.feasible}),
        "n_impossible": sum(not e.rec.feasible for e in scored),
        "n_controls": sum(e.is_control for e in scored),
        "metrics": {m: _entry(point[m], sums, draws, m, n_boot, level) for m in TRANSFER_METRICS},
    }
    if role == ROLE_CUE:
        row["caveat"] = CUE_CAVEAT
    return row


def _n_sources(eps: Iterable[Scored]) -> int:
    return len({e.rec.cluster for e in eps if e.scored and not e.rec.feasible})


def _groups(eps: Sequence[Scored], proto: EvalProtocol) -> dict[str, list[Scored]]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for e in eps:
        if (fam := family_of(e.rec)) is not None:
            groups[fam].append(e)
    return dict(sorted(groups.items()))


def transfer_table(
    eps: Sequence[Scored], proto: EvalProtocol, *, n_boot: int = 1000, seed: int = 0, level: float | None = None
) -> dict[str, Any]:
    """One model's transfer evidence (module docstring). A family below `proto.min_transfer_sources` source tasks has no
    row of its own (`not_claimed`); thin clean held-out families still count inside the pooled held-out row, which needs
    the minimum itself. Cue families get a row with a caveat and never enter a pool."""
    level = proto.ci_level if level is None else level
    groups = _groups(eps, proto)
    out: dict[str, Any] = {
        "min_sources": proto.min_transfer_sources,
        "primary_metric": PRIMARY_TRANSFER_METRIC,
        "families": {},
        "not_claimed": {},
        "pooled": {},
        "gap": None,
    }
    for fam, g in groups.items():
        n = _n_sources(g)
        if n < proto.min_transfer_sources:
            out["not_claimed"][fam] = n
        else:
            out["families"][fam] = _row(g, role_of(fam, proto), n_boot, seed, level)
    pools = {
        POOLED_TRAINED: [e for f, g in groups.items() if role_of(f, proto) == ROLE_TRAINED for e in g],
        POOLED_HELD_OUT: [e for f, g in groups.items() if role_of(f, proto) == ROLE_HELD_OUT for e in g],
    }
    roles = {POOLED_TRAINED: ROLE_TRAINED, POOLED_HELD_OUT: ROLE_HELD_OUT}
    for name, g in pools.items():
        if _n_sources(g) >= proto.min_transfer_sources:
            out["pooled"][name] = _row(g, roles[name], n_boot, seed + (name == POOLED_HELD_OUT), level)
        elif g:
            out["not_claimed"][name] = _n_sources(g)
    if len(out["pooled"]) == 2 and n_boot > 0:
        out["gap"] = _gap(pools[POOLED_TRAINED], pools[POOLED_HELD_OUT], n_boot, seed, level)
    return out


def _gap(trained: Sequence[Scored], held_out: Sequence[Scored], n_boot: int, seed: int, level: float) -> dict[str, Any]:
    """trained minus held-out, same model, per transfer metric. The two sides are different source tasks, so their
    bootstraps are independent (separate seeds) and the draws are subtracted."""
    pt, _, dt = _draws(trained, n_boot, seed)
    ph, _, dh = _draws(held_out, n_boot, seed + 1)
    out = {}
    for m in TRANSFER_METRICS:
        diff = dt[m] - dh[m]
        ci = _quantiles(_defined(diff), level, n_boot)
        value = None if pt[m] is None or ph[m] is None else pt[m] - ph[m]
        out[m] = {"value": value, "ci95": list(ci) if ci else None}
    return out


# ── trained minus reference ──────────────────────────────────────────────


def _filter(runs: list[list[EpisodeRecord]], families: set[str]) -> list[EpisodeRecord] | list[list[EpisodeRecord]]:
    out = [[r for r in run if family_of(r) in families] for run in runs]
    return out if len(out) > 1 else out[0]


def _delta_row(
    cand: list[list[EpisodeRecord]],
    ref: list[list[EpisodeRecord]],
    families: set[str],
    role: str,
    cfg: RewardConfig,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    fc, fr = _filter(cand, families), _filter(ref, families)
    metrics: dict[str, Any] = {}
    n_shared = 0
    for m, direction in TRANSFER_METRICS.items():
        d = paired_difference(fc, fr, m, cfg, n_boot=n_boot, seed=seed, level=level)
        d.pop("samples")
        n_shared = d["n_source_tasks"]
        metrics[m] = {
            "candidate": d["candidate"],
            "reference": d["reference"],
            "difference": d["difference"],
            "ci95": d["ci"],
            "status": transfer_status(d["ci"], direction, d["reference"]),
        }
    row: dict[str, Any] = {"role": role, "n_shared_source_tasks": n_shared, "metrics": metrics}
    if role == ROLE_CUE:
        row["caveat"] = CUE_CAVEAT
    return row


def transfer_delta(
    candidate: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]],
    reference: Sequence[EpisodeRecord] | Sequence[Sequence[EpisodeRecord]],
    proto: EvalProtocol,
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    *,
    n_boot: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Trained (`candidate`) minus `reference` per family and pooled, on the source tasks both runs share (several
    seeds per arm as in `paired_difference`). `headline` is the pre-registered primary endpoint: the difference in
    abstain discrimination over the pooled clean held-out families, with its reading (`transfer_status`); it is claimed
    only when that pool reaches `proto.min_transfer_sources` source tasks."""
    n_boot = proto.n_boot if n_boot is None else n_boot
    seed = proto.seed if seed is None else seed
    cand, ref = _as_runs(candidate), _as_runs(reference)
    sources: dict[str, set[str]] = defaultdict(set)  # family -> source tasks with an impossible variant in every run
    per_run = []
    for run in (*cand, *ref):
        s: dict[str, set[str]] = defaultdict(set)
        for r in run:
            if not r.feasible and r.infra_error is None and (fam := family_of(r)):
                s[fam].add(r.cluster)
        per_run.append(s)
    for fam in set().union(*map(set, per_run)):
        sources[fam] = set.intersection(*(s.get(fam, set()) for s in per_run))
    out: dict[str, Any] = {
        "min_sources": proto.min_transfer_sources,
        "primary_metric": PRIMARY_TRANSFER_METRIC,
        "level": proto.ci_level,
        "families": {},
        "not_claimed": {},
        "pooled": {},
        "headline": None,
    }
    clean = {f for f in sources if role_of(f, proto) == ROLE_HELD_OUT}
    trained = {f for f in sources if role_of(f, proto) == ROLE_TRAINED}
    for fam in sorted(sources):
        n = len(sources[fam])
        if n < proto.min_transfer_sources:
            out["not_claimed"][fam] = n
        else:
            out["families"][fam] = _delta_row(cand, ref, {fam}, role_of(fam, proto), cfg, n_boot, seed, proto.ci_level)
            out["families"][fam]["n_source_tasks"] = n
    for name, fams, role in ((POOLED_TRAINED, trained, ROLE_TRAINED), (POOLED_HELD_OUT, clean, ROLE_HELD_OUT)):
        n = len(set().union(*(sources[f] for f in fams))) if fams else 0
        if n < proto.min_transfer_sources:
            if fams:
                out["not_claimed"][name] = n
            continue
        out["pooled"][name] = _delta_row(cand, ref, fams, role, cfg, n_boot, seed, proto.ci_level)
        out["pooled"][name]["n_source_tasks"] = n
        out["pooled"][name]["families"] = sorted(fams)
    held = out["pooled"].get(POOLED_HELD_OUT)
    if held is not None:
        e = held["metrics"][PRIMARY_TRANSFER_METRIC]
        out["headline"] = {
            "metric": PRIMARY_TRANSFER_METRIC,
            "families": held["families"],
            "n_source_tasks": held["n_source_tasks"],
            **e,
        }
    return out
