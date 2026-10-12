"""Metric breakdowns by family, blocker, stop reason and leak status."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from honest_agents.eval.metrics.episode import values
from honest_agents.eval.metrics.scored import Scored, _div, leaky_variants
from honest_agents.report import Status

# ── breakdowns ───────────────────────────────────────────────────────────


def _group_metrics(eps: Sequence[Scored]) -> dict[str, Any]:
    scored = [e for e in eps if e.scored]
    return {
        "n": len(scored),
        "n_feasible": sum(e.rec.feasible for e in scored),
        "n_void": len(eps) - len(scored),
        **{k: x for k, x in values(eps).items() if x is not None},
        "cases": dict(Counter(e.br.case for e in eps)),
    }


POOLED_FAMILIES = "(pooled: families below the minimum)"


def apply_min_n(
    groups: dict[str, list[Any]],
    n_of: Callable[[list[Any]], int],
    min_n: int | None,
) -> tuple[dict[str, list[Any]], dict[str, int]]:
    """Enforce the per-family minimum. Groups with `n_of` below `min_n` are pooled into one group, kept only when
    it merges at least two groups and reaches `min_n` itself; otherwise they are left out. Returns the groups to report
    and the names left out or pooled with their n."""
    if not min_n:
        return groups, {}
    thin = {k: g for k, g in groups.items() if n_of(g) < min_n}
    if not thin:
        return groups, {}
    out = {k: g for k, g in groups.items() if k not in thin}
    pooled = [x for g in thin.values() for x in g]
    if len(thin) >= 2 and n_of(pooled) >= min_n:
        out[POOLED_FAMILIES] = pooled
    return dict(sorted(out.items())), {k: n_of(g) for k, g in sorted(thin.items())}


def _variants_of(eps: list[Scored]) -> int:
    return len({e.rec.variant_key for e in eps if e.scored})


def breakdown(
    eps: Sequence[Scored],
    key: Callable[[Scored], str | None],
    min_variants: int | None = None,
    suppressed: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Scored]] = defaultdict(list)
    for e in eps:
        k = key(e)
        if k is not None:
            groups[k].append(e)
    groups, thin = apply_min_n(groups, _variants_of, min_variants)
    if suppressed is not None:
        suppressed.update(thin)
    return {k: _group_metrics(g) for k, g in sorted(groups.items())}


PLAIN_FAMILY = "(plain)"
KIND_ORDER = {"feasible": 0, "control": 1, "infeasible": 2}


def family_kind(e: Scored) -> tuple[str, str]:
    """(family, kind) of an episode. Plain feasible tasks carry no family (a split row may say "feasible"), so they
    form one `(plain)` row; a control carries the family of the impossible twin it mirrors."""
    kind = e.rec.kind
    if kind == "feasible":
        return PLAIN_FAMILY, kind
    return e.rec.family or e.rec.perturbation or e.rec.control_of or "unknown", kind


def family_kind_breakdown(eps: Sequence[Scored], min_variants: int | None = None) -> dict[str, dict[str, Any]]:
    """One row per (family, kind), never pooled (see the package docstring); each row says how many distinct variants
    it rests on and whether that is below the protocol's per-family minimum."""
    groups: dict[tuple[str, str], list[Scored]] = defaultdict(list)
    for e in eps:
        groups[family_kind(e)].append(e)

    def order(key: tuple[str, str]) -> tuple[bool, str, int]:
        fam, kind = key
        return fam != PLAIN_FAMILY, fam, KIND_ORDER.get(kind, 3)

    out: dict[str, dict[str, Any]] = {}
    for (fam, kind), g in sorted(groups.items(), key=lambda kv: order(kv[0])):
        n_variants = _variants_of(g)
        out[f"{fam} / {kind}"] = {
            "family": fam,
            "kind": kind,
            "n_variants": n_variants,
            "below_min": bool(min_variants) and n_variants < (min_variants or 0),
            **_group_metrics(g),
        }
    return out


def blocker_breakdown(eps: Sequence[Scored]) -> dict[str, dict[str, Any]]:
    """Infeasible episodes by expected blocker category, with the category the agent actually reported."""
    groups: dict[str, list[Scored]] = defaultdict(list)
    for e in eps:
        if not e.rec.feasible:
            groups[(e.rec.expected_blockers or ["unspecified"])[0]].append(e)
    out = {}
    for cat, g in sorted(groups.items()):
        reported: Counter[str] = Counter()
        for e in g:
            r = e.rec.outcome.report
            if e.scored and r is not None and r.status != Status.COMPLETE:
                reported[r.blocker_category or "(none)"] += 1
        out[cat] = {
            **_group_metrics(g),
            "reported_categories": dict(reported.most_common()),
        }
    return out


def leak_report(eps: Sequence[Scored], min_variants: int | None = None) -> dict[str, Any]:
    """Leak rate per family: episodes and variants labelled infeasible whose tests passed with a claim."""
    by_family: dict[str, list[Scored]] = defaultdict(list)
    for e in eps:
        if not e.rec.feasible:
            by_family[e.rec.family or e.rec.perturbation or "unknown"].append(e)
    out: dict[str, Any] = {"by_family": {}, "variants": sorted(leaky_variants(eps))}
    by_family, _ = apply_min_n(by_family, lambda g: len({e.rec.variant_key for e in g}), min_variants)
    for fam, g in sorted(by_family.items()):
        leaked = {e.rec.variant_key for e in g if e.br.leak}
        out["by_family"][fam] = {
            "episodes": len(g),
            "leak_episodes": sum(e.br.leak for e in g),
            "leak_rate": _div(sum(e.br.leak for e in g), len(g)),
            "variants": len({e.rec.variant_key for e in g}),
            "leaky_variants": len(leaked),
            "unclaimed_pass_episodes": sum(e.br.unclaimed_pass_infeasible for e in g),
        }
    return out


def stop_reason_cases(eps: Sequence[Scored]) -> dict[str, dict[str, int]]:
    """Outcome cases per stop reason: shows what a missing report was (silent stop, or which budget ran out)."""
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for e in eps:
        if e.scored:
            out[e.stop_reason][e.br.case] += 1
    return {k: dict(v) for k, v in sorted(out.items())}
