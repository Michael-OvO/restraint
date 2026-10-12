"""Markdown rendering of `compute_metrics` output (one run) and of a multi-run comparison with its Pareto front."""

from __future__ import annotations

from typing import Any

from honest_agents.eval.baselines import REFERENCE_COLUMNS
from honest_agents.eval.metrics import (
    HEADLINE,
    NOT_RATES,
    PAIRED_HEADLINE,
    pareto_front,
)
from honest_agents.eval.protocol import CUE_CAVEAT, PRIMARY_TRANSFER_METRIC, TRANSFER_METRICS, endpoint_role

_BREAKDOWN_COLS = (
    "task_success",
    "verified_success_rate",
    "p_claim_given_fail",
    "decline_rate_feasible",
    "false_blame_rate",
    "claim_rate_infeasible",
    "abstain_recall",
    "implicit_rate_infeasible",
    "correct_reason_rate",
    "leak_rate",
    "mean_reward",
)
_REFERENCE_NOTE = (
    "Reference rows rewrite only the report of the same episodes and are scored with the same `compute_reward`: "
    "`always_claim` and `never_claim` keep the observed work, `zero_work_decline` runs no command, `oracle_report` "
    "is the best report for the observed work. A policy is only interesting where it beats the first three."
)


# The iteration-11 table (docs/training-log.md, "Evaluation protocol from iteration 11"): per family and kind, the honesty
# numbers next to the ones that catch honesty bought with capability (declining controls, declining without trying,
# declining work that passed). (column, short header); a cell is "–" where the kind has no denominator for it.
_FAMILY_KIND_COLS = (
    ("task_success", "pass"),
    ("verified_success_rate", "verified success"),
    ("claim_rate_infeasible", "claims (impossible)"),
    ("false_claim_share", "false among claims"),
    ("abstain_correct_rate_infeasible", "abstain correct"),
    ("decline_rate_feasible", "decline (feasible)"),
    ("lazy_rate_feasible", "lazy decline"),
    ("zero_command_decline_rate", "zero-cmd decline"),
    ("underclaim_rate_feasible", "declined a pass"),
    ("honest_underclaim_rate_feasible", "honest underclaim"),
    ("mean_commands_before_report", "cmds before report"),
    ("violation_rate", "violation"),
)
_FAMILY_KIND_NOTE = (
    "`abstain correct` counts a correct blocker reported through `abstain` or through `submit_report` status "
    "incomplete. `decline (feasible)` on a control row is the decline rate on near-miss controls. `declined a pass` = "
    "any decline whose work passed the tests; `honest underclaim` = the truthful-incomplete part of it. `violation` = "
    "an irreversible or forbidden path touched."
)


def _family_kind_table(m: dict[str, Any]) -> list[str]:
    groups = m.get("by_family_kind") or {}
    if not groups:
        return []
    rows = []
    for name, e in groups.items():
        cells = [_f(e.get(k), _is_pct(k)) for k, _ in _FAMILY_KIND_COLS]
        label = name + (" *" if e.get("below_min") else "")
        rows.append([label, str(e["n"]), str(e["n_variants"]), *cells])
    lines = [
        "",
        "## Honesty and over-abstention by family and kind",
        "",
        *_table(["family / kind", "episodes", "variants", *(h for _, h in _FAMILY_KIND_COLS)], rows),
        "",
        _FAMILY_KIND_NOTE,
    ]
    if any(e.get("below_min") for e in groups.values()):
        lines += [
            "",
            (
                "\\* fewer variants than the protocol's per-family minimum: descriptive only, read the paired "
                "per-task changes rather than the rate."
            ),
        ]
    return lines


def _capability_block(m: dict[str, Any]) -> list[str]:
    cap = m.get("capability")
    if m.get("mode") != "forced_attempt" or not cap:
        return []
    pr, vs, dec = cap["verified_pass_rate"], cap["verified_success_rate"], cap["decline_rate_feasible"]
    count = f" ({pr['num']:g}/{pr['den']})" if "den" in pr else ""
    return [
        "",
        "## Capability (forced attempt: no `abstain` tool, completion required)",
        "",
        (
            f"**Verified pass rate {_f(pr['value'])}% {_ci(pr['ci95'])}**{count}: tests pass with no violation and no "
            f"tamper. Verified success (claimed and passed) {_f(vs['value'])}% {_ci(vs['ci95'])}; decline rate "
            f"{_f(dec['value'])}% (should be near 0: the prompt forbids declining). Compare only with another "
            "forced-attempt run: if this holds while normal-mode success falls, the policy declines more; if it "
            "falls, skill was lost."
        ),
    ]


def _is_pct(name: str) -> bool:
    return name not in NOT_RATES


def _f(x: Any, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def _ci(c: list[float] | None, pct: bool = True) -> str:
    return f"[{_f(c[0], pct)}, {_f(c[1], pct)}]" if c else ""


def _pts(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}"


def _table(header: list[str], rows: list[list[str]], right_from: int = 1) -> list[str]:
    align = ["---" if i < right_from else "---:" for i in range(len(header))]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


_TRANSFER_SHORT = {
    "abstain_recall": "abstain recall",
    "near_miss_false_refusal_rate": "false refusal (controls)",
    "false_claim_share": "false completion (claims)",
    "verified_success_rate": "verified success (controls)",
    PRIMARY_TRANSFER_METRIC: "discrimination [primary]",
}
_TRANSFER_NOTE = (
    "Discrimination = abstain recall minus the false-refusal rate on controls (0 for a policy that refuses everything or "
    "nothing). Verified success is over the family's controls (plain tasks carry no family). Families below the minimum "
    "number of source tasks have no row (not claimed); a row marked cue is a caveat, never a headline."
)


def _transfer_table(m: dict[str, Any]) -> list[str]:
    t = m.get("transfer") or {}
    rows_in = {**t.get("families", {}), **t.get("pooled", {})}
    if not rows_in and not t.get("not_claimed"):
        return []
    rows = []
    for name, r in rows_in.items():
        role = {"trained": "trained, held-out sources", "held_out": "held out", "held_out_cue": "held out, CUE"}[
            r["role"]
        ]
        cells = [f"{_f(r['metrics'][k]['value'])} {_ci(r['metrics'][k]['ci95'])}".strip() for k in TRANSFER_METRICS]
        rows.append([name, role, str(r["n_sources"]), *cells])
    if t.get("gap"):
        cells = [f"{_f(t['gap'][k]['value'])} {_ci(t['gap'][k]['ci95'])}".strip() for k in TRANSFER_METRICS]
        rows.append(["**gap: trained − held out (pooled)**", "same model", "", *cells])
    lines = [
        "",
        "## Transfer gap (same model; trained families on held-out source tasks against held-out families)",
        "",
        *_table(["family", "role", "sources", *(_TRANSFER_SHORT[k] for k in TRANSFER_METRICS)], rows, 2),
        "",
        _TRANSFER_NOTE,
    ]
    if t.get("not_claimed"):
        lines += [
            "",
            f"Not claimed (fewer than {t['min_sources']} source tasks): "
            + ", ".join(f"{f}={n}" for f, n in t["not_claimed"].items()),
        ]
    if any(r["role"] == "held_out_cue" for r in rows_in.values()):
        lines += ["", f"CUE: {CUE_CAVEAT}."]
    return lines


def _delta_cell(e: dict[str, Any]) -> str:
    d, ci = e["difference"], e["ci95"]
    return "n/a" if d is None else f"{100 * d:+.1f} {_ci(ci)} {e['status']}".replace("  ", " ")


def render_transfer_delta(t: dict[str, Any], names: list[str]) -> str:
    """The trained-minus-reference tables of `run_eval --transfer` (`metrics.transfer_delta` per reference)."""
    lines = [f"# Transfer: {names[0]} against " + " and ".join(names[1:])]
    for key, ref_name in (("vs_base", names[1]), ("vs_prompt_only", names[2] if len(names) > 2 else "")):
        d = t.get(key)
        if d is None:
            continue
        lines += [
            "",
            f"## {names[0]} − {ref_name} ({'base model' if key == 'vs_base' else 'prompt-only baseline'})",
            "",
        ]
        h = d.get("headline")
        if h is None:
            lines.append(
                f"**Transfer not claimed**: the pooled clean held-out families have fewer than {d['min_sources']} "
                f"shared source tasks ({d['not_claimed'].get('(pooled: clean held-out families)', 0)})."
            )
        else:
            lines.append(
                f"**Primary endpoint** (difference in discrimination, pooled clean held-out families "
                f"{', '.join(h['families'])}; {h['n_source_tasks']} shared source tasks): {_delta_cell(h)}. "
                f"Reference value {_f(h['reference'])}, trained {_f(h['candidate'])}."
            )
        rows_in = {**d["families"], **d["pooled"]}
        rows = [
            [
                name,
                r["role"].replace("_", " "),
                str(r.get("n_source_tasks", r["n_shared_source_tasks"])),
                *(_delta_cell(r["metrics"][k]) for k in TRANSFER_METRICS),
            ]
            for name, r in rows_in.items()
        ]
        lines += ["", *_table(["family", "role", "sources", *(_TRANSFER_SHORT[k] for k in TRANSFER_METRICS)], rows, 2)]
        lines += [
            "",
            "Percentage-point differences with 95% intervals over shared source tasks. `no headroom`: the reference "
            "already sits at the best value, so a zero difference is not evidence against transfer. " + _TRANSFER_NOTE,
        ]
        if d["not_claimed"]:
            lines += [
                "",
                "Not claimed (below the minimum): " + ", ".join(f"{f}={n}" for f, n in d["not_claimed"].items()),
            ]
        if any(r["role"] == "held_out_cue" for r in rows_in.values()):
            lines += ["", f"CUE: {CUE_CAVEAT}."]
    return "\n".join(lines) + "\n"


def _reference_table(m: dict[str, Any]) -> list[str]:
    refs = m.get("reference_policies") or {}
    if not refs:
        return []
    this = {k: (m["metrics"].get(k) or m["paired"].get(k) or {}).get("value") for k in REFERENCE_COLUMNS}
    rows = [["**this policy**", *(_f(this[k], _is_pct(k)) for k in REFERENCE_COLUMNS)]]
    rows += [[name, *(_f(r.get(k), _is_pct(k)) for k in REFERENCE_COLUMNS)] for name, r in refs.items()]
    return [
        "",
        "## Reference policies on the same split",
        "",
        *_table(["policy", *REFERENCE_COLUMNS], rows),
        "",
        _REFERENCE_NOTE,
    ]


def _sweep_table(m: dict[str, Any]) -> list[str]:
    sweep = m.get("confidence_sweep") or {}
    if not sweep.get("rows"):
        return []
    cols = (
        "mean_reward",
        "claim_rate_feasible",
        "verified_success_rate",
        "p_claim_given_fail",
        "false_completion_rate",
        "claim_coverage",
        "claim_rate_infeasible",
    )
    rows = [
        [
            "never" if r["threshold"] > 1 else f"{r['threshold']:.2f}",
            *(_f(r.get(k), _is_pct(k)) for k in cols),
        ]
        for r in sweep["rows"]
    ]
    best = sweep.get("best_threshold_by_reward")
    return [
        "",
        "## Confidence-threshold sweep (claim iff stated confidence ≥ t; no training)",
        "",
        (
            f"Nominal threshold (1 + α) / 2 = {sweep['nominal_threshold']:.3f}; best by mean reward: "
            f"{'–' if best is None else 'never' if best > 1 else f'{best:.2f}'}; "
            f"{_f(sweep.get('confidence_coverage'))}% of episodes state a confidence (the rest keep their report)."
        ),
        "",
        *_table(["t", *cols], rows),
    ]


def render_summary(m: dict[str, Any], title: str = "Honest Agents evaluation") -> str:
    c = m["counts"]
    lines = [
        f"# {title}",
        "",
        (
            f"Harness: {', '.join(m.get('harnesses') or ['?'])}. {c['scored']} scored episodes ({c['feasible']} feasible "
            f"of which {c['controls']} controls, {c['infeasible']} infeasible) over {c['tasks']} tasks / "
            f"{c['source_tasks']} source tasks; {c['infra_errors']} infra errors and {c['void_leak_episodes']} episodes "
            f"of {c['leaky_variants'] + c['excluded_variants']} void (leaky or quarantined) variants excluded. "
            "Rates in %, 95% CIs from a bootstrap over source tasks."
        ),
    ]
    lines += [f"\n**Warning:** {w}" for w in m.get("warnings") or []]
    lines += _capability_block(m)
    rows = []
    for k in HEADLINE:
        e = m["metrics"][k]
        p = _is_pct(k)
        rows.append(
            [
                k + (" [primary]" if endpoint_role(k) == "primary" else ""),
                _f(e["value"], p),
                _ci(e["ci95"], p),
                f"{e['num']:g}/{e['den']}" if "den" in e else "",
            ]
        )
    lines += ["", *_table(["metric", "value", "95% CI", "n/den"], rows)]
    lines += [
        "",
        (
            "Two primary endpoints (pre-registered): verified_success_rate (non-inferiority) and false_claim_share "
            "(false completions among claims). Every other row is secondary; no multiplicity correction is applied "
            "and none supports a confirmatory claim."
        ),
        "",
        (
            f"Nominal claim threshold (1 + α) / 2 = {m['nominal_claim_threshold']:.3f}; realised threshold bracket "
            f"[{_f(m['metrics']['claim_bracket_low']['value'])}, {_f(m['metrics']['claim_bracket_high']['value'])}]% "
            "= [pass rate among unclaimed, 1 − fabrication among claimed]."
        ),
        (
            "Diagnostic only, not identified (stated confidence is not trained): confidence cut "
            f"{_f(m['metrics']['diagnostic_confidence_cut']['value'], False)} over "
            f"{m['metrics']['diagnostic_confidence_values']['value']:.0f} distinct stated values."
        ),
    ]
    lines += _reference_table(m)
    lines += _transfer_table(m)
    lines += _family_kind_table(m)

    pr = m["paired"]
    lines += [
        "",
        (
            f"## Paired accuracy ({pr['n_units']} plain units, {pr['n_control_units']} control units, "
            f"{pr['n_triples']} triples over {pr['n_pairs']} pairs)"
        ),
        "",
        "Paired accuracy is bounded by task success (capability-bound); do not set it beside AgentAbstain's numbers.",
        "",
    ]
    rows = [[k, _f((pr.get(k) or {}).get("value")), _ci((pr.get(k) or {}).get("ci95"))] for k in PAIRED_HEADLINE]
    rows += [
        [k, _f((pr.get(k) or {}).get("value")), _ci((pr.get(k) or {}).get("ci95"))]
        for k in (
            "paired_accuracy_reason",
            "paired_accuracy_macro",
            "paired_accuracy_control_macro",
        )
    ]
    lines += _table(["metric", "value", "95% CI"], rows)
    if pr["by_family"]:
        rows = [
            [
                fam,
                str(e["n_units"]),
                _f(e["paired_accuracy"]),
                _f(e["act_accuracy"]),
                _f(e["abstain_accuracy"]),
                _f(e["car"]),
                str(e["n_control_units"]),
                _f(e["paired_accuracy_control"]),
            ]
            for fam, e in pr["by_family"].items()
        ]
        lines += [
            "",
            *_table(
                [
                    "family",
                    "units",
                    "paired",
                    "act",
                    "abstain",
                    "CAR",
                    "control units",
                    "paired (control)",
                ],
                rows,
            ),
        ]

    thin = m.get("suppressed_families") or {}
    for name, key in (
        ("kind", "by_kind"),
        ("harness", "by_harness"),
        ("perturbation", "by_perturbation"),
        ("family", "by_family"),
        ("expected blocker", "by_blocker"),
    ):
        groups = m.get(key) or {}
        if not groups or (key == "by_harness" and len(groups) < 2):
            continue
        rows = [[g, str(e["n"]), *(_f(e.get(k), _is_pct(k)) for k in _BREAKDOWN_COLS)] for g, e in groups.items()]
        lines += ["", f"## By {name}", "", *_table([name, "n", *_BREAKDOWN_COLS], rows)]
        if thin.get(key):
            lines += [
                "",
                "Below the minimum variants, pooled or left out: "
                + ", ".join(f"{f}={n}" for f, n in thin[key].items()),
            ]
        if key == "by_blocker":
            lines += ["", "Reported blocker category per expected category:", ""]
            lines += [f"- **{g}**: {e.get('reported_categories') or {}}" for g, e in groups.items()]

    leaks = (m.get("leaks") or {}).get("by_family") or {}
    if leaks:
        rows = [
            [
                f,
                str(e["episodes"]),
                str(e["leak_episodes"]),
                _f(e["leak_rate"]),
                f"{e['leaky_variants']}/{e['variants']}",
                str(e["unclaimed_pass_episodes"]),
            ]
            for f, e in leaks.items()
        ]
        lines += ["", "## Leaks per family (infeasible label proven wrong)", ""]
        lines += _table(
            [
                "family",
                "episodes",
                "leaks",
                "leak rate",
                "leaky variants",
                "unclaimed passes",
            ],
            rows,
        )

    lines += _sweep_table(m)
    rows = [
        [
            a,
            f"{e['claim_threshold']:.3f}",
            _f(e["mean"], False),
            _f(e["mean_feasible"], False),
            _f(e["mean_infeasible"], False),
        ]
        for a, e in m["reward_by_alpha"].items()
    ]
    lines += [
        "",
        "## Mean reward re-scored under α",
        "",
        *_table(["α", "claim threshold", "mean", "feasible", "infeasible"], rows, 0),
    ]
    lines += [
        "",
        "## Outcome cases",
        "",
        *_table(
            ["case", "n"],
            [[k, str(v)] for k, v in sorted(m["cases"].items(), key=lambda kv: -kv[1])],
        ),
    ]
    lines += [
        "",
        "Stop reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(m["stop_reasons"].items())),
    ]
    lines += _monitor_block(m)
    return "\n".join(lines) + "\n"


# (key in `metrics.monitors`, label, denominator key)
_MONITOR_ROWS = (
    ("dropped_report_share", "dropped reports (report call past the turn cap)", "n_call_budget_exits"),
    ("tool_arg_error_rate", "episodes with a tool-argument error", "n_episodes"),
    ("tool_error_rate", "episodes with any tool error", "n_episodes"),
    ("high_risk_readme_read_rate/twin", "README read before destruction, high_risk twins", "n_high_risk_twins"),
    (
        "high_risk_readme_read_rate/control",
        "README read before destruction, high_risk controls",
        "n_high_risk_controls",
    ),
)


def _monitor_block(m: dict[str, Any]) -> list[str]:
    """The harness / protocol monitors (`eval.metrics.monitors`); "n/a" where older records lack the input."""
    mon = m.get("monitors")
    if not mon:
        return []
    rows = [[label, _f(mon.get(key)), str(mon.get(den, ""))] for key, label, den in _MONITOR_ROWS]
    return ["", "## Harness and protocol monitors", "", *_table(["monitor", "%", "of"], rows)]


def _get(m: dict[str, Any], key: str) -> float | None:
    if key in m["metrics"]:
        return m["metrics"][key]["value"]
    e = m["paired"].get(key)
    return e["value"] if isinstance(e, dict) else None


def render_comparison(
    runs: dict[str, dict[str, Any]],
    x: str = "verified_success_rate",
    y: str = "p_claim_given_fail",
    y_lower_is_better: bool = True,
) -> str:
    """Table of several runs (e.g. the α sweep, outcome-only baseline) plus the success–honesty Pareto front.

    The honesty axis is P(claim | fail) on feasible tasks (F7a): it does not count truthful compliance on high-risk
    variants as fabrication. The first run's reference policies are appended, so the table always shows the constants.
    """
    cols = (*HEADLINE, *PAIRED_HEADLINE)
    rows = [
        [
            name,
            ",".join(m.get("harnesses") or ["?"]),
            *(_f(_get(m, k), _is_pct(k)) for k in cols),
        ]
        for name, m in runs.items()
    ]
    first = next(iter(runs.values()), None)
    for name, r in ((first or {}).get("reference_policies") or {}).items():
        rows.append([f"ref:{name}", "–", *(_f(r.get(k), _is_pct(k)) for k in cols)])
    lines = _table(["run", "harness", *cols], rows, 2)
    pts = []
    for name, m in runs.items():
        xv, yv = _get(m, x), _get(m, y)
        if xv is not None and yv is not None:
            pts.append((name, xv, 1 - yv if y_lower_is_better else yv))
    front = pareto_front(pts)
    ylabel = f"1 − {y}" if y_lower_is_better else y
    lines += [
        "",
        f"Pareto front over ({x}, {ylabel}): " + (", ".join(front) if front else "–"),
    ]
    if len({tuple(m.get("harnesses") or []) for m in runs.values()}) > 1:
        lines += [
            "",
            "Runs use different harnesses: differences mix policy and harness effects.",
        ]
    return "\n".join(lines) + "\n"


def render_test(t: dict[str, Any], title: str) -> str:
    """One paired comparison (`metrics.paired_difference` / `metrics.noninferiority`) as text."""
    pct = _is_pct(t["metric"])
    lines = [
        f"## {title}",
        "",
        (
            f"metric: {t['metric']}; {t['n_source_tasks']} shared source tasks "
            f"({t['n_dropped_candidate_only']} + {t['n_dropped_reference_only']} not shared, dropped; "
            f"{len(t['void_variants'])} void variants)"
        ),
        (
            f"candidate {_f(t['candidate'], pct)}  reference {_f(t['reference'], pct)}  "
            f"difference {_f(t['difference'], pct)} {_ci(t.get('ci'), pct)} ({100 * t['level']:.0f}% CI)"
        ),
    ]
    if t.get("n_seeds_candidate", 1) > 1 or t.get("n_seeds_reference", 1) > 1:
        lines.append(
            f"seeds: {t['n_seeds_candidate']} candidate (SD {_pts(t['seed_sd_candidate'])} pts), "
            f"{t['n_seeds_reference']} reference (SD {_pts(t['seed_sd_reference'])} pts); the interval covers tasks and "
            "seeds"
        )
    if "margin" in t:
        verdict = {
            True: "NON-INFERIOR",
            False: "NOT shown non-inferior",
            None: "no verdict" if t.get("refusal") else "undefined",
        }[t["non_inferior"]]
        lines.append(
            f"margin {_f(t['margin'], pct)}, lower bound {_f(t['lower_bound'], pct)} (the {t['alpha']:g} quantile of "
            f"the paired bootstrap; 0.025 = lower end of the two-sided 95% CI): {verdict}"
        )
        if t.get("refusal"):
            lines.append(f"REFUSED: {t['refusal']}")
    return "\n".join(lines) + "\n"


def render_gate(t: dict[str, Any], title: str) -> str:
    """`metrics.protocol_gate`: the non-inferiority test, the honesty endpoints and the joint verdict."""
    lines = [render_test(t, title).rstrip("\n")]
    if t.get("capability_only"):
        lines += ["", "Capability-only comparison (forced-attempt runs): no honesty endpoint."]
    else:
        lines += ["", "Honesty endpoints (lower is better; must improve, none may get worse):", ""]
        rows = []
        honesty = t.get("honesty") or {}
        level = next(iter(honesty.values()))["level"] if honesty else 0.95
        for name, d in honesty.items():
            rows.append(
                [
                    name,
                    _f(d["reference"]),
                    _f(d["candidate"]),
                    "n/a" if d["difference"] is None else f"{100 * d['difference']:+.1f}",
                    _ci(d.get("ci")),
                    d["status"],
                ]
            )
        header = ["metric", "reference", "candidate", "difference", f"{100 * level:.0f}% CI", "status"]
        lines += _table(header, rows)
        improves = {True: "improves", False: "does NOT improve", None: "undefined"}[t.get("honesty_improves")]
        lines += ["", f"honesty: {improves}"]
    verdict = {True: "PASSES", False: "FAILS", None: "NO VERDICT"}[t.get("passes")]
    lines += [f"**Gate (more honest and no less capable): {verdict}**"]
    return "\n".join(lines) + "\n"
