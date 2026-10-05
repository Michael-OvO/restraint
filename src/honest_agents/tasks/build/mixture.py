"""Balancing the feasible / control / infeasible mixture of a split."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Callable, Sequence

from honest_agents.tasks.build.config import DEFAULT_MIXTURE, KINDS, Row, normalise_mixture

log = logging.getLogger(__name__)


# ── mixture ──────────────────────────────────────────────────────────────────


def _round_robin(rows: list[Row], group: Callable[[Row], str], n: int, rng: random.Random) -> list[Row]:
    """Up to `n` rows, one group at a time in turn (groups in name order, rows within a group in random order)."""
    queues = defaultdict(list)
    for r in rows:
        queues[group(r)].append(r)
    qs = [rng.sample(queues[k], len(queues[k])) for k in sorted(queues)]
    out: list[Row] = []
    while len(out) < n and any(qs):
        for q in qs:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def balance_mixture(
    rows: list[Row], mixture: Sequence[float] = DEFAULT_MIXTURE, seed: int = 0, *, paired: bool = True
) -> list[Row]:
    """Down-sample to the requested plain-feasible / control / infeasible mixture.

    Paired (default): an infeasible variant and its control (same source, same operator) are drawn together, round-robin
    over families, so inside a family the two kinds have the same count and the same sources and a family-level cue
    cannot separate them by prior or by source. The control and infeasible shares are averaged to make that possible.
    Controls without an infeasible twin (relabelled ones) are not drawn. Plain feasible rows prefer the sources that
    contributed a pair. Unpaired: infeasible rows round-robin, a control for each chosen variant first, then others.
    A stratum with no rows leaves the split as it is (the ratio is unattainable).
    """
    share = dict(zip(KINDS, normalise_mixture(mixture), strict=True))
    by_kind = {k: [r for r in rows if r.kind == k] for k in KINDS}
    needed = [k for k in KINDS if share[k] > 0]
    if any(not by_kind[k] for k in needed):
        log.warning(
            "mixture: %s, left as is",
            ", ".join(f"{len(by_kind[k])} {k}" for k in KINDS),
        )
        return list(rows)
    rng = random.Random(seed)
    if paired and share["control"] > 0 and share["infeasible"] > 0:
        return _balance_paired(rows, by_kind, share, rng)
    total = min(len(by_kind[k]) / share[k] for k in needed)
    want = {k: min(len(by_kind[k]), round(total * share[k])) for k in KINDS}

    chosen_inf = _round_robin(by_kind["infeasible"], lambda r: r.family, want["infeasible"], rng)
    twins = {(r.pair_id, r.perturbation) for r in chosen_inf}
    ctrl_twin = [r for r in by_kind["control"] if (r.pair_id, r.perturbation) in twins]
    ctrl_rest = [r for r in by_kind["control"] if (r.pair_id, r.perturbation) not in twins]
    chosen_ctl = _round_robin(ctrl_twin, lambda r: r.family, want["control"], rng)
    chosen_ctl += _round_robin(ctrl_rest, lambda r: r.family, want["control"] - len(chosen_ctl), rng)
    chosen_feas = _choose_feasible(
        by_kind["feasible"], {r.pair_id for r in chosen_inf + chosen_ctl}, want["feasible"], rng
    )
    keep = {id(r) for r in chosen_inf + chosen_ctl + chosen_feas}
    return [r for r in rows if id(r) in keep]


def _choose_feasible(feas: list[Row], paired: set[str], n: int, rng: random.Random) -> list[Row]:
    chosen = _round_robin([r for r in feas if r.pair_id in paired], lambda r: r.pair_id, n, rng)
    return chosen + _round_robin(
        [r for r in feas if r.pair_id not in paired], lambda r: r.pair_id, n - len(chosen), rng
    )


def _balance_paired(
    rows: list[Row], by_kind: dict[str, list[Row]], share: dict[str, float], rng: random.Random
) -> list[Row]:
    pair_share = (share["control"] + share["infeasible"]) / 2
    ctl = {}
    for r in by_kind["control"]:
        ctl.setdefault((r.pair_id, r.perturbation), r)
    pairs = [(r, ctl[(r.pair_id, r.perturbation)]) for r in by_kind["infeasible"] if (r.pair_id, r.perturbation) in ctl]
    if not pairs:
        log.warning("mixture: no infeasible variant has a control twin, left as is")
        return list(rows)
    n_feas = len(by_kind["feasible"])
    total = (
        min(n_feas / share["feasible"], len(pairs) / pair_share) if share["feasible"] > 0 else len(pairs) / pair_share
    )
    want_pairs = min(len(pairs), round(total * pair_share))
    by_inf = {id(inf): (inf, c) for inf, c in pairs}
    chosen_inf = _round_robin([inf for inf, _ in pairs], lambda r: r.family, want_pairs, rng)
    chosen_ctl = [by_inf[id(r)][1] for r in chosen_inf]
    want_feas = min(n_feas, round(total * share["feasible"]))
    chosen_feas = _choose_feasible(by_kind["feasible"], {r.pair_id for r in chosen_inf}, want_feas, rng)
    keep = {id(r) for r in chosen_inf + chosen_ctl + chosen_feas}
    return [r for r in rows if id(r) in keep]
