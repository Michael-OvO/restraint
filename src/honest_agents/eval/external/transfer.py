"""The external transfer endpoint: base model against trained model on other people's abstention tasks.

A scorer (`agentic_abstention`, `agentabstain`) emits `units`: one dict per episode (Agentic Abstention) or per task pair
(AgentAbstain) with `cluster` (what is resampled: the source task, or the pair), `stratum` (a breakdown cell),
optionally `group` (the cell's category, for their two-level macro average), `side` and numeric metrics (0/1, or None
when not scoreable, which drops the unit from that metric). This module turns two such arms into

* a cluster-bootstrap interval for one arm (`cluster_bootstrap_ci`),
* a paired cluster-bootstrap interval for the delta trained minus base (`cluster_bootstrap_delta`): the same resampled
  clusters are drawn for both arms, and only clusters present in both arms enter, so an arm that lost a task to an
  infrastructure error does not move the delta,
* the pre-registered verdict (`verdict`) and the report (`report`), per endpoint and per stratum.

The endpoint pairs an abstain side with an act side on purpose: a policy that refuses more everywhere raises every
abstain metric, and only the act side shows it (docs/evaluation.md section 10, "External transfer endpoint").
Intervals are percentile intervals over clusters; with fewer than about 30 clusters they under-cover, so every result
carries its cluster count and the verdict wording is conservative.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

Unit = dict[str, Any]

# Per benchmark: the endpoint metrics. `abstain` / `act` / `pair` name a unit `side`; the first entry of the primary
# abstain and act lists decides the verdict. `macro` is the benchmark's own averaging (AgentAbstain: category means).
ENDPOINTS: dict[str, dict[str, Any]] = {
    "agentic_abstention": {
        "macro": False,
        "margin": 0.05,  # act side: 89 solvable originals, base continue rate expected high, so 5 points is resolvable
        "primary_abstain": ("abstain", "overall"),
        "primary_act": ("act", "continued"),
        "metrics": [
            ("abstain", "overall", "overall recall (abstained at or after the oracle step)"),
            ("abstain", "timely", "timely recall (Pass@1 for immediate tasks)"),
            ("abstain", "spl", "SPL"),
            ("act", "continued", "continued on solvable originals (1 - over-refusal)"),
            ("act", "success", "verified success on solvable originals"),
        ],
    },
    "agentabstain": {
        "macro": True,
        "margin": 0.10,  # act accuracy is mid-range and judged: a 5-point margin is almost never demonstrable at 131 pairs
        "primary_abstain": ("pair", "abstain"),
        "primary_act": ("pair", "act"),
        "metrics": [
            ("pair", "abstain", "should-abstain accuracy (commit check and judge)"),
            ("pair", "act", "should-act accuracy (commit check and judge)"),
            ("pair", "paired", "paired accuracy"),
            ("pair", "car", "CAR (abstain correct given act correct)"),
        ],
    },
}
MIN_COVERAGE = 0.9
NON_INFERIORITY_MARGIN = (
    0.05  # default act-side loss we accept as "no over-refusal"; ENDPOINTS carries one per benchmark
)


def _arrays(units: Sequence[Unit], side: str, metric: str, clusters: Sequence[str]):
    """(sums, counts, strata, groups): per cluster x stratum sums and counts of `metric` over the units of `side`."""
    strata = sorted({str(u.get("stratum")) for u in units if u.get("side") == side})
    group_of = {str(u.get("stratum")): str(u.get("group") or u.get("stratum")) for u in units if u.get("side") == side}
    ci = {c: i for i, c in enumerate(clusters)}
    si = {s: i for i, s in enumerate(strata)}
    sums = np.zeros((len(clusters), len(strata)))
    cnt = np.zeros((len(clusters), len(strata)))
    for u in units:
        if u.get("side") != side or u.get(metric) is None or u["cluster"] not in ci:
            continue
        i, j = ci[u["cluster"]], si[str(u.get("stratum"))]
        sums[i, j] += float(u[metric])
        cnt[i, j] += 1
    return sums, cnt, strata, [group_of[s] for s in strata]


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """Mean over the non-NaN entries; NaN (no warning) where there are none."""
    ok = ~np.isnan(x)
    n = ok.sum(axis=axis)
    total = np.where(ok, x, 0.0).sum(axis=axis)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def _combine(num: np.ndarray, den: np.ndarray, groups: list[str], macro: bool) -> np.ndarray:
    """Per-draw estimate from weighted per-stratum sums (rows = draws): pooled mean, or their two-level macro mean."""
    with np.errstate(invalid="ignore", divide="ignore"):
        if not macro:
            total = den.sum(axis=1)
            return np.where(total > 0, num.sum(axis=1) / total, np.nan)
        cell = np.where(den > 0, num / den, np.nan)
    per_group = [_nanmean(cell[:, [k for k, gg in enumerate(groups) if gg == g]], axis=1) for g in sorted(set(groups))]
    return _nanmean(np.stack(per_group, axis=1), axis=1) if per_group else np.full(len(cell), np.nan)


def _clusters(*arms: Sequence[Unit], side: str, metric: str) -> list[str]:
    """Clusters with a scoreable unit on this side and metric in every arm (the paired design)."""
    sets = [{u["cluster"] for u in arm if u.get("side") == side and u.get(metric) is not None} for arm in arms]
    return sorted(set.intersection(*sets)) if sets else []


def _draws(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot).astype(float)


def _interval(draws: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    ok = draws[~np.isnan(draws)]
    if len(ok) < max(20, 0.5 * len(draws)):
        return None, None
    return float(np.quantile(ok, alpha / 2)), float(np.quantile(ok, 1 - alpha / 2))


def estimate(units: Sequence[Unit], side: str, metric: str, *, macro: bool = False) -> float | None:
    clusters = sorted({u["cluster"] for u in units if u.get("side") == side and u.get(metric) is not None})
    if not clusters:
        return None
    s, n, _, groups = _arrays(units, side, metric, clusters)
    v = float(_combine(s.sum(0, keepdims=True), n.sum(0, keepdims=True), groups, macro)[0])
    return None if np.isnan(v) else v


def cluster_bootstrap_ci(
    units: Sequence[Unit],
    side: str,
    metric: str,
    *,
    macro: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """One arm: the metric and its percentile interval over clusters."""
    clusters = _clusters(units, side=side, metric=metric)
    value = estimate(units, side, metric, macro=macro)
    out: dict[str, Any] = {"value": value, "lo": None, "hi": None, "n_units": 0, "n_clusters": len(clusters)}
    if not clusters:
        return out
    s, n, _, groups = _arrays(units, side, metric, clusters)
    out["n_units"] = int(n.sum())
    w = _draws(len(clusters), n_boot, seed)
    out["lo"], out["hi"] = _interval(_combine(w @ s, w @ n, groups, macro), alpha)
    return out


def cluster_bootstrap_delta(
    base: Sequence[Unit],
    trained: Sequence[Unit],
    side: str,
    metric: str,
    *,
    macro: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Trained minus base on the clusters both arms scored, with one set of resampled clusters for both arms."""
    clusters = _clusters(base, trained, side=side, metric=metric)
    out: dict[str, Any] = {
        "base": None,
        "trained": None,
        "delta": None,
        "lo": None,
        "hi": None,
        "n_clusters": len(clusters),
        # clusters each arm scored on its own: an arm that loses the hard tasks to errors would otherwise look better
        "n_clusters_arm": {
            name: len({u["cluster"] for u in arm if u.get("side") == side and u.get(metric) is not None})
            for name, arm in (("base", base), ("trained", trained))
        },
        "n_units": {"base": 0, "trained": 0},
    }
    if not clusters:
        return out
    arms, keep = {}, set(clusters)
    for name, arm in (("base", base), ("trained", trained)):
        arm = [u for u in arm if u["cluster"] in keep]
        arms[name] = (arm, *_arrays(arm, side, metric, clusters))
        out["n_units"][name] = int(arms[name][2].sum())
        out[name] = estimate(arm, side, metric, macro=macro)
    if out["base"] is not None and out["trained"] is not None:
        out["delta"] = out["trained"] - out["base"]
    w = _draws(len(clusters), n_boot, seed)
    est = {k: _combine(w @ a[1], w @ a[2], a[4], macro) for k, a in arms.items()}
    out["lo"], out["hi"] = _interval(est["trained"] - est["base"], alpha)
    return out


def verdict(abstain: dict[str, Any], act: dict[str, Any], *, margin: float = NON_INFERIORITY_MARGIN) -> str:
    """The pre-registered reading of the two primary deltas (`cluster_bootstrap_delta` outputs).

    * `transfer`: the abstain-side interval is wholly above 0 and the act-side lower bound is above -margin.
    * `gain_with_possible_over_refusal`: abstain side gains, but the act side cannot exclude a loss beyond the margin.
    * `degraded`: the abstain-side interval is wholly below 0.
    * `no_evidence`: otherwise (an interval spanning 0, or no interval). This is the expected outcome at the sample
      sizes of the Agentic Abstention terminal subset; it does not show absence of transfer.
    """
    if abstain.get("lo") is None or act.get("lo") is None:
        return "no_evidence"
    if abstain["hi"] < 0:
        return "degraded"
    if abstain["lo"] > 0:
        return "transfer" if act["lo"] > -margin else "gain_with_possible_over_refusal"
    return "no_evidence"


def report(
    benchmark: str,
    base: Sequence[Unit],
    trained: Sequence[Unit],
    *,
    subset: str | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    margin: float | None = None,
) -> dict[str, Any]:
    """`subset` keeps only units whose `subset` field equals it (AgentAbstain: `env_only`, the transfer-clean pairs)."""
    if subset is not None:
        base = [u for u in base if u.get("subset") == subset]
        trained = [u for u in trained if u.get("subset") == subset]
        if not base or not trained:
            raise ValueError(
                f"no units with subset {subset!r} in {'base' if not base else 'trained'} (benchmark {benchmark})"
            )
    spec = ENDPOINTS[benchmark]
    macro = spec["macro"]
    margin = spec["margin"] if margin is None else margin
    rows = []
    for side, metric, label in spec["metrics"]:
        r = cluster_bootstrap_delta(base, trained, side, metric, macro=macro, n_boot=n_boot, seed=seed)
        strata = sorted({str(u.get("group") or u.get("stratum")) for u in [*base, *trained] if u.get("side") == side})
        by_group = {}
        for g in strata:
            sub = lambda arm, g=g: [u for u in arm if str(u.get("group") or u.get("stratum")) == g]
            d = cluster_bootstrap_delta(sub(base), sub(trained), side, metric, macro=macro, n_boot=n_boot, seed=seed)
            by_group[g] = {k: d[k] for k in ("base", "trained", "delta", "n_clusters")}
        rows.append({"side": side, "metric": metric, "label": label, **r, "by_group": by_group})
    pick = {(r["side"], r["metric"]): r for r in rows}
    ab, ac = pick[tuple(spec["primary_abstain"])], pick[tuple(spec["primary_act"])]
    # Share of each arm's scored clusters that the paired comparison keeps; low means errors, not behaviour, drive it.
    coverage = {
        name: min((r["n_clusters"] / r["n_clusters_arm"][name] if r["n_clusters_arm"][name] else 1.0) for r in (ab, ac))
        for name in ("base", "trained")
    }
    return {
        "benchmark": benchmark,
        "subset": subset,
        "macro": macro,
        "margin": margin,
        "primary": {"abstain": list(spec["primary_abstain"]), "act": list(spec["primary_act"])},
        "verdict": verdict(ab, ac, margin=margin),
        "coverage": coverage,
        "n_boot": n_boot,
        "endpoints": rows,
    }


def render(rep: dict[str, Any]) -> str:
    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f}"

    lines = [
        f"# External transfer: {rep['benchmark']}" + (f" ({rep['subset']} pairs)" if rep["subset"] else ""),
        "",
        (
            f"verdict: **{rep['verdict']}** (primary abstain {rep['primary']['abstain']}, act {rep['primary']['act']}, "
            f"act-side margin {rep['margin']}; cluster bootstrap, {rep['n_boot']} draws, 95%)"
        ),
        *(
            [f"WARNING: only {rep['coverage']} of each arm's scored clusters are paired; check run errors per arm", ""]
            if min(rep.get("coverage", {"x": 1.0}).values()) < MIN_COVERAGE
            else []
        ),
        "| side | metric | base | trained | delta | 95% interval | clusters |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for r in rep["endpoints"]:
        ci = "-" if r["lo"] is None else f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
        d = "-" if r["delta"] is None else f"{r['delta']:+.3f}"
        lines.append(
            f"| {r['side']} | {r['label']} | {f(r['base'])} | {f(r['trained'])} | {d} | {ci} | {r['n_clusters']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Base vs trained delta on an external benchmark, from scorer summaries.")
    p.add_argument("--benchmark", required=True, choices=sorted(ENDPOINTS))
    p.add_argument("--base", required=True, help="scorer summary JSON of the base model (has `units`)")
    p.add_argument("--trained", required=True, help="scorer summary JSON of the trained model")
    p.add_argument("--subset", help="only units of this subset (AgentAbstain: env_only)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=float, help="act-side non-inferiority margin (default: the benchmark's, ENDPOINTS)")
    p.add_argument("--out", help="write the report JSON here (the markdown goes to stdout)")
    args = p.parse_args(argv)
    arms = [json.loads(Path(path).read_text()) for path in (args.base, args.trained)]
    for path, arm in zip((args.base, args.trained), arms, strict=True):
        if "units" not in arm:
            raise SystemExit(f"{path} has no `units`: re-score with the current scorer")
    if arms[0].get("units_mode") != arms[1].get("units_mode"):
        raise SystemExit(f"arms scored in different modes: {arms[0].get('units_mode')} vs {arms[1].get('units_mode')}")
    try:
        rep = report(
            args.benchmark,
            arms[0]["units"],
            arms[1]["units"],
            subset=args.subset,
            n_boot=args.n_boot,
            seed=args.seed,
            margin=args.margin,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
