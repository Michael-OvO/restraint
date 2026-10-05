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


def _is_pct(name: str) -> bool:
    return name not in NOT_RATES


def _f(x: Any, pct: bool = True) -> str:
    if x is None:
        return "–"
    return f"{100 * x:.1f}" if pct else f"{x:.3f}"


def _ci(c: list[float] | None, pct: bool = True) -> str:
    return f"[{_f(c[0], pct)}, {_f(c[1], pct)}]" if c else ""


def _table(header: list[str], rows: list[list[str]], right_from: int = 1) -> list[str]:
    align = ["---" if i < right_from else "---:" for i in range(len(header))]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


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
    rows = []
    for k in HEADLINE:
        e = m["metrics"][k]
        p = _is_pct(k)
        rows.append(
            [
                k,
                _f(e["value"], p),
                _ci(e["ci95"], p),
                f"{e['num']:g}/{e['den']}" if "den" in e else "",
            ]
        )
    lines += ["", *_table(["metric", "value", "95% CI", "n/den"], rows)]
    lines += [
        "",
        f"Nominal claim threshold (1 + α) / 2 = {m['nominal_claim_threshold']:.3f}.",
    ]
    lines += _reference_table(m)

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
    return "\n".join(lines) + "\n"


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
            "Runs use different harnesses: differences mix policy and harness effects (SPEC_V3 §9.4).",
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
    if "margin" in t:
        verdict = {
            True: "NON-INFERIOR",
            False: "NOT shown non-inferior",
            None: "undefined",
        }[t["non_inferior"]]
        lines.append(
            f"margin {_f(t['margin'], pct)}, one-sided lower bound {_f(t['lower_bound'], pct)} at alpha={t['alpha']}: {verdict}"
        )
    return "\n".join(lines) + "\n"
