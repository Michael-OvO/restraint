"""Manifest sections: cue metrics, pairing, training and yield reports."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from honest_agents.tasks.build.config import MIN_FAMILY_TEST_ROWS, MIN_TRAIN_PAIRS, SPLITS, Row
from honest_agents.tasks.build.pairs import Pair
from honest_agents.tasks.perturb import (
    OPERATORS,
)

CUE_BAND = (0.4, 0.6)  # L1: a family's existence cue must not separate infeasible from control outside this band


def emitted_label_leaks(variants: Iterable[Pair]) -> list[str]:
    """Every variant, the plain feasible copy included, is scanned for label strings in what ships under environment/."""
    from honest_agents.tasks.analysis import label_leaks

    return [
        f"{v.variant_id}: {leak.path}:{leak.line}" for s, v in variants for leak in label_leaks(v.task_dir, s.task_dir)
    ]


def pairing_stats(splits: dict[str, list[Row]]) -> dict[str, Any]:
    """Per split: infeasible/control counts per family, controls without a twin, feasible rows from paired sources."""
    out: dict[str, Any] = {}
    for name, rs in splits.items():
        inf = Counter((r.family) for r in rs if r.kind == "infeasible")
        ctl = Counter((r.family) for r in rs if r.kind == "control")
        twins = {(r.pair_id, r.perturbation) for r in rs if r.kind == "infeasible"}
        paired_src = {p for p, _ in twins}
        feas = [r for r in rs if r.kind == "feasible"]
        out[name] = {
            "infeasible_per_family": dict(inf),
            "control_per_family": dict(ctl),
            "infeasible_share_per_family": {f: round(inf[f] / (inf[f] + ctl[f]), 3) for f in inf.keys() | ctl.keys()},
            "controls_without_twin": sum(r.kind == "control" and (r.pair_id, r.perturbation) not in twins for r in rs),
            "feasible_from_paired_sources": sum(r.pair_id in paired_src for r in feas),
            "feasible": len(feas),
        }
    return out


HELDOUT_CAVEAT = "zero-shot test family: a lexical cue the generator cannot remove (or too few pairs); keep it out of headline numbers"


def cue_metrics(variants_by_split: dict[str, list[Any]], heldout: Iterable[str] = ()) -> dict[str, Any]:
    """Per split and family: the existence AUC of the forbidden/target object, from `analysis.existence_auc`.

    `variants_by_split` holds the emitted `Variant`s. Violations are split in two: `violations` (the families that
    train, which the build enforces) and `heldout_violations` (families in `heldout`, only reported: a test-only family
    can never block a build, and its cue is the documented construct-validity limit). The import is guarded so an older
    `analysis` (no `existence_auc`) yields `available: false` instead of breaking the build."""
    heldout = set(heldout)
    try:
        from honest_agents.tasks.analysis import cue_classifiers, existence_auc, existence_rows, existence_violations
    except ImportError:
        return {"available": False, "band": list(CUE_BAND), "violations": {}}
    out: dict[str, Any] = {
        "available": True,
        "band": list(CUE_BAND),
        "heldout_families": sorted(heldout),
        "heldout_caveat": HELDOUT_CAVEAT,
        "violations": {},
        "heldout_violations": {},
    }
    for name, vs in variants_by_split.items():
        try:
            rows = existence_rows(vs)
            report = existence_auc(rows)
            for fam, cv in cue_classifiers(rows).items():
                report[fam]["cv"] = cv
            out[name] = report
            trained = {f: e for f, e in report.items() if f not in heldout}
            out["violations"][name] = existence_violations(trained, low=CUE_BAND[0], high=CUE_BAND[1])
            out["heldout_violations"][name] = existence_violations(
                {f: e for f, e in report.items() if f in heldout}, low=CUE_BAND[0], high=CUE_BAND[1]
            )
            out.setdefault("watch", {})[name] = cue_watch(report)
            # A thin family is not judged (too few rows for an AUC): list it so it never reads as "clean".
            out.setdefault("unjudged", {})[name] = sorted(
                f for f, e in report.items() if min(e.get("n_infeasible", 0), e.get("n_control", 0)) < 10
            )
        except Exception as e:  # noqa: BLE001 — a metric must never break the build (enforcement is explicit)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def cue_watch(report: dict[str, Any], *, min_per_class: int = 20, z: float = 2.0) -> list[str]:
    """Features outside the enforced set that still separate infeasible from control: AUC outside the band and more than
    `z` standard errors from 0.5. They are listed in the manifest (never fatal), because a cue the enforced list does not
    name (for example `mentions_in_instruction`) is the same shortcut in another form."""
    out = []
    for fam, entry in sorted(report.items()):
        for name, f in sorted((entry.get("features") or {}).items()):
            auc, se = f.get("auc"), f.get("auc_se")
            if auc is None or se is None or min(f["n_infeasible"], f["n_control"]) < min_per_class:
                continue
            if not CUE_BAND[0] <= auc <= CUE_BAND[1] and abs(auc - 0.5) > z * se:
                out.append(f"{fam}: {name} AUC {auc:.2f} (+-{se:.2f}, n {f['n_infeasible']}/{f['n_control']})")
    return out


def training_report(
    splits: dict[str, list[Row]], families: Sequence[str], *, pool: Sequence[Row] = ()
) -> dict[str, Any]:
    """Infeasible/control pairs per training family in the train split (and in the pool before the mixture), and the
    families with fewer than `MIN_TRAIN_PAIRS` (a rate learned from fewer is mostly noise)."""
    train = Counter(r.family for r in splits["train"] if r.kind == "infeasible")
    pooled = Counter(r.family for r in pool if r.kind == "infeasible")
    return {
        "families": list(families),
        "min_pairs": MIN_TRAIN_PAIRS,
        "pairs_per_family": {f: train.get(f, 0) for f in families},
        "pool_pairs_per_family": {f: pooled.get(f, 0) for f in families},
        "thin_families": {f: train.get(f, 0) for f in families if train.get(f, 0) < MIN_TRAIN_PAIRS},
    }


def yield_report(
    rows: Sequence[Row],
    seed_split: dict[str, str],
    clusters: dict[str, str],
    emitted: dict[str, list[Row]],
    heldout: Iterable[str] = (),
) -> dict[str, Any]:
    """Per split and family: how many seeds yield an infeasible/control pair, and how many independent source tasks
    stand behind them (near-duplicate clusters count once). `pool` is every variant built for the split before the
    held-out filter and the mixture; `emitted` is what the split file holds. A family with a few source tasks gives a
    rate nobody should quote, however many rows the mixture drew from it."""
    heldout = set(heldout)

    def unit(r: Row) -> str:
        return clusters.get(r.split_group, r.split_group)

    out: dict[str, Any] = {"per": 150}
    for split in SPLITS:
        groups = [g for g, sp in seed_split.items() if sp == split]
        n_units = len({clusters.get(g, g) for g in groups})
        fams: dict[str, Any] = {}
        for fam in OPERATORS:
            inf = [r for r in rows if r.split == split and r.family == fam and r.kind == "infeasible"]
            ctl = [r for r in rows if r.split == split and r.family == fam and r.kind == "control"]
            kept = [r for r in emitted[split] if r.family == fam and r.kind == "infeasible"]
            seeds = {r.split_group for r in inf}
            fams[fam] = {
                "role": "held-out (test only)" if fam in heldout else "train",
                "pool_infeasible": len(inf),
                "pool_controls": len(ctl),
                "pool_source_tasks": len({unit(r) for r in inf}),
                "pool_seeds_with_pair": len(seeds),
                "pool_yield_per_150_seeds": round(150 * len(seeds) / len(groups), 1) if groups else 0.0,
                "emitted_pairs": len(kept),
                "emitted_source_tasks": len({unit(r) for r in kept}),
            }
        out[split] = {"seeds": len(groups), "source_tasks": n_units, "families": fams}
    return out


def format_yield(report: dict[str, Any]) -> str:
    """The yield report as a table, for the console at the end of a build."""
    lines = []
    for split in SPLITS:
        block = report[split]
        lines.append(f"{split}: {block['seeds']} seeds, {block['source_tasks']} independent source tasks")
        lines.append(f"  {'family':24s}{'role':22s}{'pool':>6s}{'src':>6s}{'/150':>7s}{'emit':>6s}{'src':>6s}")
        for fam, e in block["families"].items():
            lines.append(
                f"  {fam:24s}{e['role']:22s}{e['pool_infeasible']:6d}{e['pool_source_tasks']:6d}"
                f"{e['pool_yield_per_150_seeds']:7.1f}{e['emitted_pairs']:6d}{e['emitted_source_tasks']:6d}"
            )
    return "\n".join(lines)


def held_out_coverage(rows: list[Row]) -> dict[str, Any]:
    """Infeasible variants per family in the test split, and the families below `MIN_FAMILY_TEST_ROWS`."""
    per = Counter(r.family for r in rows if r.kind == "infeasible")
    return {
        "infeasible_per_family": dict(per),
        "min_family_rows": MIN_FAMILY_TEST_ROWS,
        "short_families": {f: per.get(f, 0) for f in OPERATORS if per.get(f, 0) < MIN_FAMILY_TEST_ROWS},
        "pairs": len({r.pair_id for r in rows}),
    }
