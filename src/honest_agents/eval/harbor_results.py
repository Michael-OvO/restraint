"""Parse a Harbor job directory into `EpisodeRecord`s, so `harbor run` results share our metrics and reward.

Layout (harbor 0.23.0, `harbor/job.py` + `harbor/models/trial/paths.py`):

    <job_dir>/result.json                 JobResult
    <job_dir>/<trial_name>/result.json    TrialResult (task_name, config.task.path, verifier_result.rewards,
                                          agent_info, agent_result, exception_info)
    <job_dir>/<trial_name>/agent/         the agent's logs dir: `HonestPolicyAgent` writes honest_result.json here;
                                          a third-party harness leaves the policy-written honest_report.json and
                                          usually an ATIF trajectory.json
    <job_dir>/<trial_name>/artifacts/     `--artifact` downloads, mirrored by absolute source path

Where the report comes from, per trial:

1. `agent/honest_result.json`, only when Harbor's `agent_info.name` is our agent: our harness (`harness = "honest"`),
   report from the tool call (or its file fallback);
2. else the report file of the harness-agnostic protocol (`honest_agents.eval.report_file`), under the name of the
   Harbor agent that ran (`agent_info.name`, e.g. `terminus-2`);
3. else no report channel at all (a stock run without the instruction suffix): only task success is meaningful.

Every record carries `harness`, and metrics are reported per harness (SPEC_V3 §9.4).

The JSON is read leniently (plain dicts, not `TrialResult.model_validate`) so results from neighbouring Harbor versions
still parse. What Harbor's verifier does not check is recorded as absent: no tamper, no path violation, no evidence
replay and no blocker verification. A trial without a verifier reward was never graded and is an infra error.

Task labels: `[metadata.honest]` if the task dir is available locally; else Agentic Abstention's materialised
`metadata.expected_abstain`; else the task is assumed feasible (TB Lite / TB 2.0).
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

from honest_agents.eval import report_file
from honest_agents.eval.harbor_agent import RESULT_FILE
from honest_agents.eval.records import HARNESS_HONEST, EpisodeRecord, write_records
from honest_agents.report import SelfReport
from honest_agents.reward import (
    MASK_REWARD_ERROR,
    STOP_REPORT,
    EpisodeOutcome,
    RewardConfig,
)
from honest_agents.spec import HonestTaskSpec

OUR_AGENT_NAME = "honest-policy"  # `HonestPolicyAgent.name()`: the only agent whose `honest_result.json` is trusted


def trial_dirs(job_dir: str | Path) -> list[Path]:
    return sorted(d for d in Path(job_dir).iterdir() if d.is_dir() and (d / "result.json").is_file())


def verifier_reward(result: dict[str, Any]) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if "reward" in rewards and isinstance(rewards["reward"], (int, float)):
        return float(rewards["reward"])
    nums = [v for v in rewards.values() if isinstance(v, (int, float))]
    return float(nums[0]) if nums else None


def _task_path(result: dict[str, Any]) -> Path | None:
    p = ((result.get("config") or {}).get("task") or {}).get("path")
    if p and (Path(p) / "task.toml").is_file():
        return Path(p)
    p = (result.get("task_id") or {}).get("path")
    return Path(p) if p and (Path(p) / "task.toml").is_file() else None


def task_labels(task_dir: Path | None) -> tuple[dict[str, Any], HonestTaskSpec | None]:
    """(record labels, the task's honesty spec if it has one)."""
    if task_dir is None:
        return {"feasible": True}, None
    meta = tomllib.loads((task_dir / "task.toml").read_text()).get("metadata", {})
    if "honest" in meta:
        spec = HonestTaskSpec.model_validate(meta["honest"])
        return EpisodeRecord.labels_from_spec(spec), spec
    if "expected_abstain" in meta:  # Agentic Abstention materialised task
        return {
            "feasible": not meta["expected_abstain"],
            "perturbation": meta.get("instruction_level_category") or meta.get("variant"),
            "family": meta.get("instruction_level_category"),
            "pair_id": meta.get("source_task_name"),
            "source_task": meta.get("source_task_name"),
        }, None
    return {"feasible": True}, None


def _our_payload(trial_dir: Path, agent_name: str | None) -> dict[str, Any] | None:
    """The result `HonestPolicyAgent` wrote, or None when another harness ran the trial.

    Only trusted when Harbor says our agent ran the trial. `/logs/agent` is writable by whatever policy a third-party
    harness runs, so a file of that name from another agent could carry a forged command count or stop reason; for
    those trials the report is read through the report-file protocol like any other policy-written text.
    """
    if agent_name not in (None, "", OUR_AGENT_NAME):
        return None
    path = trial_dir / "agent" / RESULT_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def load_trial(trial_dir: Path, *, success_threshold: float = 1.0, sample: int = 0) -> EpisodeRecord:
    result = json.loads((trial_dir / "result.json").read_text())
    task_dir = _task_path(result)
    labels, spec = task_labels(task_dir)
    agent_info = result.get("agent_info") or {}
    meta = (result.get("agent_result") or {}).get("metadata") or {}
    extra: dict[str, Any] = {}

    ours = _our_payload(trial_dir, agent_info.get("name"))
    if (
        ours is None and "honest_report" in meta and agent_info.get("name") in (None, "", OUR_AGENT_NAME)
    ):  # our agent ran but its result file was lost: use the trial result
        ours = {
            "report": meta.get("honest_report"),
            "report_error": meta.get("honest_report_error"),
            "n_actions": meta.get("n_actions"),
            "stop_reason": meta.get("stop_reason"),
        }
    if ours is not None:
        harness_name = ours.get("harness") or HARNESS_HONEST
        raw = ours.get("report")
        report = SelfReport.model_validate(raw) if raw else None
        report_error = ours.get("report_error")
        channel = ours.get("report_channel") or ("tool" if report is not None or report_error else "none")
        n_actions = int(ours.get("n_actions") or 0)
        acted = bool(ours.get("agent_acted", n_actions > 0))
        stop_reason = ours.get("stop_reason")
        rollout = dict(ours.get("rollout") or {})
        model = ours.get("model")
        has_channel = True
    else:
        harness_name = agent_info.get("name") or "unknown"
        report, report_error, path = report_file.read_trial_report(trial_dir)
        wrote = path is not None
        channel = "file" if wrote else "none"
        n_actions, known = report_file.actions_before_report(report_file.trajectory_tool_calls(trial_dir), wrote)
        acted = n_actions > 0 or wrote
        stop_reason = STOP_REPORT if wrote else None
        rollout, model = {}, None
        has_channel = wrote
        extra["n_actions_known"] = known
        extra["report_file"] = str(path) if path else None

    reward = verifier_reward(result)
    exc = result.get("exception_info") or {}
    masked_reason = None
    if reward is None:
        # No verifier reward = the trial never got graded (environment or verifier failure): infra, not policy.
        masked_reason = MASK_REWARD_ERROR
        extra["infra_error"] = (
            f"harbor: {exc.get('exception_type', 'no verifier result')}: {exc.get('exception_message', '')}"
        )
    outcome = EpisodeOutcome(
        feasible=labels.get("feasible", True),
        report=report,
        report_error=report_error if report is None else None,
        tests_passed=None if reward is None else reward >= success_threshold,
        test_reward=reward,
        n_actions=n_actions,
        agent_acted=acted,
        expected_blockers=list(labels.get("expected_blockers", [])),
        expected_targets=list(spec.expected_targets) if spec else [],
        decoy_targets=list(spec.decoy_targets) if spec else [],
        is_control=bool(labels.get("is_control", False)),
        stop_reason=stop_reason,
        masked_reason=masked_reason,
        n_blockers_submitted=report.n_blockers_submitted if report is not None else 0,
    )
    if exc:
        rollout["exception"] = exc.get("exception_type")
    return EpisodeRecord(
        episode_id=result.get("trial_name") or trial_dir.name,
        task_id=result.get("task_name") or trial_dir.name,
        task_dir=str(task_dir) if task_dir else None,
        sample=sample,
        model=model or ((agent_info.get("model_info") or {}).get("name")),
        source="harbor",
        harness=harness_name,
        report_channel=channel,
        outcome=outcome,
        rollout=rollout,
        extra={
            **extra,
            "trial_dir": str(trial_dir),
            "task_path": ((result.get("config") or {}).get("task") or {}).get("path"),
            "agent": agent_info.get("name"),
            "has_self_report_channel": has_channel,
            "decision": meta.get("decision"),
            "n_interaction_rounds": meta.get("n_interaction_rounds"),
        },
        **labels,
    )


def load_harbor_job(job_dir: str | Path, *, success_threshold: float = 1.0) -> list[EpisodeRecord]:
    """One record per trial; repeated attempts of a task (`-k`) get consecutive sample indices."""
    seen: dict[str, int] = defaultdict(int)
    records = []
    for d in trial_dirs(job_dir):
        rec = load_trial(d, success_threshold=success_threshold)
        rec.sample = seen[rec.task_id]
        seen[rec.task_id] += 1
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> int:
    from honest_agents.eval.metrics import compute_metrics
    from honest_agents.eval.summary import render_summary

    p = argparse.ArgumentParser(description="Convert a Harbor job dir into episodes.jsonl + metrics.json + summary.md")
    p.add_argument("job_dir")
    p.add_argument("--out", help="output dir (default: <job_dir>/honest_eval)")
    p.add_argument("--success-threshold", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=RewardConfig().alpha)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not Path(args.job_dir).is_dir():
        raise SystemExit(f"no such Harbor job dir: {args.job_dir}")
    out = Path(args.out or Path(args.job_dir) / "honest_eval")
    out.mkdir(parents=True, exist_ok=True)
    cfg = RewardConfig(alpha=args.alpha)
    records = load_harbor_job(args.job_dir, success_threshold=args.success_threshold)
    write_records(out / "episodes.jsonl", records, cfg=cfg)
    m = compute_metrics(records, cfg, n_boot=args.n_boot, seed=args.seed)
    no_channel = sum(not r.extra.get("has_self_report_channel", True) for r in records)
    if no_channel:
        m["warnings"].append(
            f"{no_channel} of {len(records)} trials have no report channel (no report file): for them only task "
            "success is meaningful; a missing report is scored as no report"
        )
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    summary = render_summary(m, f"Harbor job {Path(args.job_dir).name}")
    (out / "summary.md").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
