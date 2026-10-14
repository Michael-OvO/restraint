"""Top level: every table of one evaluation report."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from honest_agents.eval.metrics.bootstrap import bootstrap
from honest_agents.eval.metrics.breakdown import (
    blocker_breakdown,
    breakdown,
    family_kind_breakdown,
    leak_report,
    phrasing_breakdown,
    pooled_surface_breakdown,
    stop_reason_cases,
    surface_breakdown,
)
from honest_agents.eval.metrics.calibration import calibration
from honest_agents.eval.metrics.episode import ratios, values
from honest_agents.eval.metrics.monitors import monitor_metrics
from honest_agents.eval.metrics.paired import PAIRED_NAMES, paired_metrics
from honest_agents.eval.metrics.scored import DEFAULT_ALPHAS, INFRA_ERROR_WARN, _div, leaky_variants, score_records
from honest_agents.eval.metrics.sweep import reward_by_alpha
from honest_agents.eval.metrics.transfer import transfer_table
from honest_agents.eval.protocol import EvalProtocol
from honest_agents.eval.records import DEFAULT_REWARD_CONFIG, EpisodeRecord
from honest_agents.reward import (
    RewardConfig,
)

# ── top level ────────────────────────────────────────────────────────────


def compute_metrics(
    records: Sequence[EpisodeRecord],
    cfg: RewardConfig = DEFAULT_REWARD_CONFIG,
    *,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    n_boot: int = 1000,
    seed: int = 0,
    exclude_variants: Iterable[str] = (),
    references: bool = True,
    thresholds: Sequence[float] | None = None,
    min_source_tasks: int | None = None,
    min_family_variants: int | None = None,
    protocol: EvalProtocol | None = None,
    forced_attempt: bool = False,
) -> dict[str, Any]:
    """Everything `summary.render_summary` prints. `protocol` names the training and held-out families for the
    `transfer` table (default: `EvalProtocol()`). `references` adds the constant-policy reference rows and the
    confidence-threshold sweep (`honest_agents.eval.baselines`), computed with `compute_reward` on the same episodes.
    `forced_attempt` marks a capability run (`run_eval --forced-attempt`) and adds the `capability` block.
    """
    from honest_agents.eval import baselines

    exclude = set(exclude_variants)
    eps = score_records(records, cfg, exclude_variants=exclude)
    scored = [e for e in eps if e.scored]
    errors = [r for r in records if r.infra_error is not None]
    ratio = ratios(eps)
    headline = {**values(eps), **calibration(eps)}
    paired = paired_metrics(eps, min_family_variants)
    ci = bootstrap(eps, n_boot=n_boot, seed=seed)
    harnesses = sorted({e.rec.harness for e in eps})

    def entry(k: str) -> dict[str, Any]:
        c = ci.get(k)
        d: dict[str, Any] = {"value": headline[k], "ci95": list(c) if c else None}
        if k in ratio:
            d["num"], d["den"] = ratio[k]
        return d

    warnings = (
        [f"episodes from several harnesses are pooled ({', '.join(harnesses)}): read `by_harness`"]
        if len(harnesses) > 1
        else []
    )
    if records and len(errors) / len(records) > INFRA_ERROR_WARN:
        warnings.append(
            f"{len(errors)} of {len(records)} episodes ({100 * len(errors) / len(records):.0f}%) are infra errors and "
            "are excluded: a policy that provokes server or sandbox errors would be flattered, so check them "
            "(`infra_error_examples`) and re-run before quoting the numbers"
        )
    source_tasks = len({e.rec.cluster for e in scored})
    if min_source_tasks is not None and source_tasks < min_source_tasks:
        warnings.append(
            f"only {source_tasks} source tasks were scored (the protocol asks for {min_source_tasks}): intervals are "
            "wide and per-family numbers rest on a handful of tasks"
        )
    if min_family_variants is not None:
        small = {
            fam: len({e.rec.variant_key for e in scored if (e.rec.family or e.rec.perturbation) == fam})
            for fam in {e.rec.family or e.rec.perturbation for e in scored if not e.rec.feasible} - {None}
        }
        thin = {f: n for f, n in sorted(small.items()) if n < min_family_variants}
        if thin:
            warnings.append(
                f"families with fewer than {min_family_variants} variants are pooled or left out of the per-family "
                "tables (`suppressed_families`): " + ", ".join(f"{f}={n}" for f, n in thin.items())
            )
    suppressed: dict[str, dict[str, int]] = {"by_family": {}, "by_perturbation": {}}
    out: dict[str, Any] = {
        "reward_config": dict(cfg.__dict__),
        "nominal_claim_threshold": cfg.claim_threshold,
        "harnesses": harnesses,
        "warnings": warnings,
        "counts": {
            "episodes": len(records),
            "scored": len(scored),
            "infra_errors": len(errors),
            "infra_error_rate": _div(len(errors), len(records)),
            "void_leak_episodes": len(eps) - len(scored),
            "leaky_variants": len(leaky_variants(eps)),
            "excluded_variants": len(exclude),
            "feasible": sum(e.rec.feasible for e in scored),
            "controls": sum(e.is_control for e in scored),
            "infeasible": sum(not e.rec.feasible for e in scored),
            "tasks": len({e.rec.task_id for e in scored}),
            "pairs": len({e.rec.pair_id for e in scored if e.rec.pair_id}),
            "source_tasks": source_tasks,
        },
        "metrics": {k: entry(k) for k in headline},
        "paired": {
            **{k: v for k, v in paired.items() if k.startswith("n_")},
            **{k: {"value": paired[k], "ci95": list(ci[k]) if ci.get(k) else None} for k in PAIRED_NAMES},
            "by_family": paired["by_family"],
        },
        "cases": dict(Counter(e.br.case for e in eps)),
        "stop_reasons": dict(Counter(e.stop_reason for e in scored)),
        "stop_reason_cases": stop_reason_cases(eps),
        "leaks": leak_report(eps, min_family_variants),
        "by_kind": breakdown(eps, lambda e: e.rec.kind),
        "by_family_kind": family_kind_breakdown(eps, min_family_variants),
        "by_family": breakdown(
            eps, lambda e: e.rec.family or e.rec.perturbation or "(none)", min_family_variants, suppressed["by_family"]
        ),
        "by_perturbation": breakdown(
            eps, lambda e: e.rec.perturbation or "(original)", min_family_variants, suppressed["by_perturbation"]
        ),
        "suppressed_families": suppressed,
        "by_surface": surface_breakdown(eps),
        "by_surface_pooled": pooled_surface_breakdown(eps),
        "by_phrasing": phrasing_breakdown(eps),
        "transfer": transfer_table(eps, protocol or EvalProtocol(), n_boot=n_boot, seed=seed),
        "by_harness": breakdown(eps, lambda e: e.rec.harness),
        "by_blocker": blocker_breakdown(eps),
        "reward_by_alpha": reward_by_alpha(records, cfg, alphas),
        "infra_error_examples": [{"episode_id": r.episode_id, "error": r.infra_error} for r in errors[:20]],
        "monitors": monitor_metrics([e.rec for e in scored]),
    }
    out["mode"] = "forced_attempt" if forced_attempt else "normal"
    if forced_attempt:
        n_abstain = sum(e.br.abstained for e in scored)
        out["capability"] = {
            "verified_pass_rate": entry("task_success"),
            "verified_success_rate": entry("verified_success_rate"),
            "decline_rate_feasible": entry("decline_rate_feasible"),
            "n_infeasible_episodes": sum(not e.rec.feasible for e in scored),
            "n_abstain_reports": n_abstain,
        }
        if n_abstain:
            warnings.append(
                f"{n_abstain} episodes report status abstain in a forced-attempt run (through submit_report): the "
                "prompt did not hold for them; they count as declines and as non-passes where the tests failed"
            )
    if references:
        # The same split as the policy's own row: a variant that leaked under the observed reports is void in every
        # reference row too (a rewritten report would otherwise turn the leak into an ordinary scored episode).
        void = exclude | leaky_variants(eps)
        out["reference_policies"] = baselines.reference_rows(records, cfg, exclude_variants=void)
        out["confidence_sweep"] = baselines.confidence_sweep(records, cfg, thresholds, exclude_variants=void)
    return out
