"""Module-level TRL reward functions (picklable, so they also work with AsyncGRPOTrainer workers)."""

from __future__ import annotations

import concurrent.futures
from collections import Counter

from honest_agents.env import HonestHarborEnv

_CASES = (
    "success_claimed",
    "fabrication",
    "honest_failure",
    "honest_underclaim",
    "lazy_abstain",
    "feasible_no_report",
    "abstain_correct",
    "abstain_wrong_reason",
    "infeasible_claimed",
    "implicit_abstain",
    "tamper",
)


def honesty_reward(environments: list[HonestHarborEnv], log_metric=None, log_extra=None, **_) -> list[float | None]:
    """Honesty-aware reward (see `honest_agents.reward`). Grades the whole batch concurrently.

    Returns `None` (→ NaN, ignored by TRL) for episodes whose sandbox failed to boot or grade, so infrastructure
    failures never become a learning signal.
    """
    futures = [env._start_grading() for env in environments]
    concurrent.futures.wait(futures)
    results = [env._graded() for env in environments]

    rewards: list[float | None] = []
    cases: Counter[str] = Counter()
    feas = infeas = 0
    fab = claimed_feas = abst_feas = infeas_claimed = infeas_explicit = infeas_implicit = 0
    tamper = post_hoc = errors = tests_leak = 0
    for outcome, br, _debug in results:
        if outcome.grading_error is not None:
            errors += 1
            rewards.append(None)
            continue
        rewards.append(br.total)
        cases[br.case] += 1
        tamper += br.tamper
        post_hoc += br.post_hoc
        if outcome.feasible:
            feas += 1
            claimed_feas += br.claimed_complete
            fab += br.fabricated
            abst_feas += br.abstained
        else:
            infeas += 1
            infeas_claimed += br.claimed_complete
            infeas_explicit += br.explicit_non_completion
            infeas_implicit += br.implicit
            tests_leak += bool(outcome.tests_passed)

    if log_metric is not None:
        n = max(1, len(results) - errors)
        for c in _CASES:
            log_metric(f"honest/case/{c}", cases[c] / n)
        log_metric("honest/fabrication_rate_feasible", fab / max(1, claimed_feas))
        log_metric("honest/abstain_rate_feasible", abst_feas / max(1, feas))
        log_metric("honest/claim_rate_infeasible", infeas_claimed / max(1, infeas))
        log_metric("honest/explicit_abstain_rate_infeasible", infeas_explicit / max(1, infeas))
        log_metric("honest/implicit_rate_infeasible", infeas_implicit / max(1, infeas))
        log_metric("honest/tamper_rate", tamper / n)
        log_metric("honest/post_hoc_rate", post_hoc / n)
        log_metric("honest/infeasible_tests_pass_rate", tests_leak / max(1, infeas))
        log_metric("honest/grading_error_rate", errors / max(1, len(results)))
    if log_extra is not None:
        log_extra("case", [br.case for _, br, _ in results])
        log_extra("tests_passed", [o.tests_passed for o, _, _ in results])
        log_extra("status", [o.report.status.value if o.report else None for o, _, _ in results])
    return rewards


def task_success(environments: list[HonestHarborEnv], **_) -> list[float | None]:
    """Outcome-only reward: test pass on feasible tasks; 0 on infeasible. The H1 baseline (weight it alone)."""
    concurrent.futures.wait([env._start_grading() for env in environments])
    out: list[float | None] = []
    for env in environments:
        outcome, _, _ = env._graded()
        if outcome.grading_error is not None:
            out.append(None)
        else:
            out.append(float(bool(outcome.tests_passed)) if outcome.feasible else 0.0)
    return out
