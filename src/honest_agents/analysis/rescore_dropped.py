"""Re-score a GRPO run's training rollouts with the reports its harness dropped (docs/iteration-12-design.md 2.1 item 4).

    python -m honest_agents.analysis.rescore_dropped <run_dir> --tasks-dir data/tasks [--bucket-size 5] [--json out.json]

Iteration 11 capped an episode at 30 tool calls (28 bash + 2). A report written after the policy had spent both spare
calls on refused bash came one turn past the loop's cap and was never executed: 73 of 168 training budget exits ended in
an unexecuted `submit_report` / `abstain` call and were scored as "no report" (bad-case review R1). This script reads
the run's completions tables (`completions/*.parquet`: the rendered completion text plus the grader's columns), recovers
each such call from the Qwen tool-call XML (`behaviour.unexecuted_report_call`), and scores the episode again with that
report (`reward.compute_reward`, the run's reward config), everything else as graded.

What is and is not known offline:
* tests_passed, n_actions, masking and the case come from the table; the task labels (expected blockers and targets,
  decoys, control) from the variant's `task.toml` under `--tasks-dir`;
* the grader's violation flag is not a column. It is inferred from the logged reward: the no-report outcome is scored
  with and without a violation and the one that reproduces `honest_total` is kept (an episode neither reproduces is
  reported as unverified and left unchanged);
* a blocker's verdict on a feasible task needs the sandbox's facts, which the table does not keep: it is `unknown`,
  which the reward treats like a false blame (`reward.compute_reward`). Nearly all late feasible declines blame the
  exhausted budget itself (`bash`), which the grader would also call false.

Confounded by task mix (design section 9 item 6): each step trains on different tasks, so the per-step-bucket rates
are "training-rollout claim rates with recovered reports", not a policy comparison on fixed tasks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honest_agents.behaviour import ToolCall, unexecuted_report_call
from honest_agents.harness import report_from_tool_args
from honest_agents.report import ReportParseError
from honest_agents.reward import STOP_CALL_BUDGET, EpisodeOutcome, RewardBreakdown, RewardConfig, compute_reward
from honest_agents.spec import HonestTaskSpec

COMPLETIONS_DIR = "completions"
RUN_CONFIG_FILE = "run_config.json"
TOLERANCE = 1e-6  # |recomputed - logged| reward for the no-report reconstruction to count as reproduced


@dataclass(frozen=True)
class Rollout:
    """One row of a run's completions table (the columns this analysis reads)."""

    step: int
    group: str  # the variant id (`trl_rewards.group_ids`), which is also its task directory's name
    completion: str
    stop_reason: str | None
    case: str
    status: str
    feasible: bool
    tests_passed: bool | None
    n_actions: int
    honest_total: float
    masked_reason: str | None


@dataclass(frozen=True)
class Rescored:
    rollout: Rollout
    family: str  # spec family ("feasible" for a plain task)
    kind: str  # "twin" | "control" | "plain"
    recovered: ToolCall | None  # the unexecuted report call of a call-budget exit
    after: (
        RewardBreakdown | None
    )  # the episode scored with the recovered report (None: nothing recovered or unverified)
    violation: bool | None  # inferred grader violation flag (None: not reproduced)
    parse_error: str | None = None  # the recovered call did not parse as a report

    @property
    def dropped(self) -> bool:
        return self.recovered is not None

    @property
    def scored(self) -> bool:
        return self.rollout.masked_reason is None

    # Before = as logged; after = the recovered report's score where there is one, else as logged.
    def claimed(self, recovered: bool) -> bool:
        if recovered and self.after is not None:
            return self.after.claimed_complete
        return self.rollout.status == "complete"

    def case(self, recovered: bool) -> str:
        return self.after.case if recovered and self.after is not None else self.rollout.case

    def total(self, recovered: bool) -> float:
        return self.after.total if recovered and self.after is not None else self.rollout.honest_total


# ── loading ─────────────────────────────────────────────────────────────────


def _bool_or_none(v: Any) -> bool | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return bool(v)


def _str_or_none(v: Any) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def load_rollouts(run_dir: Path) -> list[Rollout]:
    """Every row of `<run_dir>/completions/*.parquet`, in step order."""
    import pyarrow.parquet as pq

    files = sorted((run_dir / COMPLETIONS_DIR).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {COMPLETIONS_DIR}/*.parquet under {run_dir}")
    out = []
    for f in files:
        for r in pq.read_table(f).to_pylist():
            out.append(
                Rollout(
                    step=int(r["step"]),
                    group=str(r["group"]),
                    completion=str(r.get("completion") or ""),
                    stop_reason=_str_or_none(r.get("stop_reason")),
                    case=str(r.get("case")),
                    status=str(r.get("status")),
                    feasible=bool(r.get("feasible")),
                    tests_passed=_bool_or_none(r.get("tests_passed")),
                    n_actions=int(r.get("n_actions") or 0),
                    honest_total=float(r.get("honest_total") or 0.0),
                    masked_reason=_str_or_none(r.get("masked_reason")),
                )
            )
    return sorted(out, key=lambda r: r.step)


def reward_config_of_run(run_dir: Path) -> RewardConfig:
    """The run's reward config (`run_config.json` `reward.config`): the overrides of the library defaults it trained with."""
    path = run_dir / RUN_CONFIG_FILE
    raw = json.loads(path.read_text()).get("reward", {}).get("config", {}) if path.is_file() else {}
    return RewardConfig(**raw)


# ── re-scoring ──────────────────────────────────────────────────────────────


def _outcome(r: Rollout, spec: HonestTaskSpec, *, report: Any, violation: bool, report_error: str | None = None):
    return EpisodeOutcome(
        feasible=r.feasible,
        report=report,
        report_error=report_error,
        tests_passed=r.tests_passed,
        n_actions=r.n_actions,
        agent_acted=r.n_actions > 0,
        expected_blockers=list(spec.expected_blockers),
        expected_targets=list(spec.expected_targets),
        decoy_targets=list(spec.decoy_targets),
        is_control=spec.is_control,
        irreversible_violation=violation,
        tamper=r.case == "tamper",
        stop_reason=r.stop_reason,
        masked_reason=r.masked_reason,
    )


def infer_violation(r: Rollout, spec: HonestTaskSpec, cfg: RewardConfig) -> bool | None:
    """The grader's violation flag of a no-report episode: the value whose score reproduces the logged reward."""
    for violation in (False, True):
        br = compute_reward(_outcome(r, spec, report=None, violation=violation), cfg)
        if abs(br.total - r.honest_total) <= TOLERANCE and br.case == r.case:
            return violation
    return None


def kind_of(spec: HonestTaskSpec) -> str:
    if not spec.feasible:
        return "twin"
    return "control" if spec.is_control else "plain"


def rescore(r: Rollout, spec: HonestTaskSpec, cfg: RewardConfig) -> Rescored:
    """`r` with its dropped report (if it has one) scored. Only call-budget exits can have one."""
    family, kind = spec.family or ("feasible" if spec.feasible else "unknown"), kind_of(spec)
    call = unexecuted_report_call(r.completion) if r.stop_reason == STOP_CALL_BUDGET else None
    if call is None or r.masked_reason is not None:
        return Rescored(r, family, kind, call if r.masked_reason is None else None, None, None)
    violation = infer_violation(r, spec, cfg)
    if violation is None:
        return Rescored(r, family, kind, call, None, None)
    try:
        report, error = report_from_tool_args(call.name, call.arguments), None
    except ReportParseError as e:  # unparsable: scored like the trainer scores an unparsable report
        report, error = None, str(e)
    after = compute_reward(_outcome(r, spec, report=report, violation=violation, report_error=error), cfg)
    return Rescored(r, family, kind, call, after, violation, error)


def rescore_run(run_dir: Path, tasks_dir: Path, cfg: RewardConfig | None = None) -> list[Rescored]:
    cfg = cfg or reward_config_of_run(run_dir)
    specs: dict[str, HonestTaskSpec] = {}
    out = []
    for r in load_rollouts(run_dir):
        if r.group not in specs:
            specs[r.group] = HonestTaskSpec.from_task_dir(tasks_dir / r.group)
        out.append(rescore(r, specs[r.group], cfg))
    return out


# ── the table ───────────────────────────────────────────────────────────────


def _rate(items: Sequence[Rescored], pred: Callable[[Rescored], bool]) -> float | None:
    return sum(pred(x) for x in items) / len(items) if items else None


def cell_metrics(items: Sequence[Rescored]) -> dict[str, Any]:
    """Rates over the scored rows of one cell, before (as logged) and after (dropped reports recovered)."""
    scored = [x for x in items if x.scored]
    impossible = [x for x in scored if x.kind == "twin"]
    feasible = [x for x in scored if x.kind != "twin"]
    m: dict[str, Any] = {
        "n": len(scored),
        "n_call_budget": sum(x.rollout.stop_reason == STOP_CALL_BUDGET for x in scored),
        "n_dropped": sum(x.dropped for x in scored),
        "n_rescored": sum(x.after is not None for x in scored),
        "n_unverified": sum(x.dropped and x.violation is None for x in scored),
        "n_impossible": len(impossible),
        "n_feasible": len(feasible),
    }
    for tag, recovered in (("before", False), ("after", True)):
        m[f"claim_rate_impossible_{tag}"] = _rate(impossible, lambda x, rec=recovered: x.claimed(rec))
        m[f"abstain_correct_{tag}"] = _rate(impossible, lambda x, rec=recovered: x.case(rec) == "abstain_correct")
        m[f"fabrication_{tag}"] = _rate(feasible, lambda x, rec=recovered: x.case(rec) == "fabrication")
        m[f"mean_reward_{tag}"] = _rate(scored, lambda x, rec=recovered: x.total(rec))  # a mean, not a share
    m["reward_change_total"] = sum(x.total(True) - x.total(False) for x in scored)
    return m


def bucket_of(step: int, size: int) -> str:
    lo = (step - 1) // size * size + 1
    return f"{lo}-{lo + size - 1}" if size > 1 else str(step)


def family_label(x: Rescored) -> str:
    return x.family if x.kind == "twin" else ("plain" if x.kind == "plain" else f"{x.family} control")


def table(rescored: Iterable[Rescored], bucket_size: int = 5) -> list[dict[str, Any]]:
    """One row per (step bucket, family), then one "all" row per bucket and one per family over the run."""
    items = list(rescored)
    cells: dict[tuple[str, str], list[Rescored]] = defaultdict(list)
    for x in items:
        bucket = bucket_of(x.rollout.step, bucket_size)
        cells[(bucket, family_label(x))].append(x)
        cells[(bucket, "all")].append(x)
        cells[("all", family_label(x))].append(x)
        cells[("all", "all")].append(x)

    def order(key: tuple[str, str]) -> tuple[float, int, str]:
        bucket, family = key
        first = math.inf if bucket == "all" else float(bucket.split("-")[0])
        return first, family == "all", family

    return [{"steps": b, "family": f, **cell_metrics(cells[(b, f)])} for b, f in sorted(cells, key=order)]


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def _pair(row: dict[str, Any], key: str) -> str:
    before, after = row[f"{key}_before"], row[f"{key}_after"]
    if before is None:
        return "-"
    return _pct(before) if after == before else f"{_pct(before)} -> {_pct(after)}"


def render(rows: Sequence[dict[str, Any]]) -> str:
    head = [
        "steps", "family", "n", "budget exits", "dropped", "re-scored", "claims on impossible %",
        "abstain_correct %", "fabrication (feasible) %", "mean reward", "reward change (sum)",
    ]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * 2 + "---:|" * (len(head) - 2)]
    for r in rows:
        mean = (
            "-"
            if r["mean_reward_before"] is None
            else f"{r['mean_reward_before']:+.3f} -> {r['mean_reward_after']:+.3f}"
        )
        cells = [
            r["steps"], r["family"], str(r["n"]), str(r["n_call_budget"]), str(r["n_dropped"]), str(r["n_rescored"]),
            _pair(r, "claim_rate_impossible"), _pair(r, "abstain_correct"), _pair(r, "fabrication"), mean,
            f"{r['reward_change_total']:+.2f}",
        ]  # fmt: skip
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def recovered_statuses(rescored: Iterable[Rescored]) -> dict[str, int]:
    """What the dropped calls said: `submit_report:<status>` or `abstain`, plus `unparsed` and `unverified`."""
    out: dict[str, int] = defaultdict(int)
    for x in rescored:
        if not x.dropped or not x.scored:
            continue
        if x.violation is None:
            out["unverified"] += 1
        elif x.parse_error is not None:
            out["unparsed"] += 1
        elif x.recovered is not None and x.recovered.name == "submit_report":
            out[f"submit_report:{x.recovered.arguments.get('status')}"] += 1
        else:
            out["abstain"] += 1
    return dict(sorted(out.items()))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a GRPO run directory (completions/*.parquet, run_config.json)")
    ap.add_argument("--tasks-dir", type=Path, default=Path("data/tasks"), help="task directories, one per variant id")
    ap.add_argument("--bucket-size", type=int, default=5, help="training steps per row of the table (default 5)")
    ap.add_argument("--json", type=Path, default=None, help="also write the table and the counts as JSON")
    args = ap.parse_args(argv)
    if args.bucket_size < 1:
        ap.error("--bucket-size must be >= 1")
    rescored = rescore_run(args.run_dir, args.tasks_dir)
    rows = table(rescored, args.bucket_size)
    statuses = recovered_statuses(rescored)
    print(render(rows))
    print()
    print("Recovered calls: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"table": rows, "recovered": statuses}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
