"""The pilot's TRL reward function: re-score each new report against its trajectory's cached outcome.

`PilotReward(mode)` is registered alone, with `reward_weights=[1.0]`, like the live reward functions, so a None is a
real mask. It reuses `honest_agents.trl_rewards` for everything that must match a live run (`mode_reward`, the
group-level leak mask, `batch_metrics`, the completions-table columns) and never writes to the trajectory bank or the
quarantine list: the pilot reads the bank, and a "leak" here would only repeat what the bank already says.

Where the report comes from. With `max_tool_calling_iterations: 0` (the pilot's default) TRL parses the tool call but
never executes it, so the completion is exactly one report turn and no token after the report is generated or trained
on. The reward function then delivers the parsed call to the environment itself, the same way TRL's tool loop would
(`tool(**arguments)`; a call that raises leaves the episode without a report). With one or more iterations TRL has
already called the tool and nothing is delivered twice.

Like the live reward functions, this one never raises into the training loop: a failure masks the whole batch.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from honest_agents import bank, trl_rewards
from honest_agents.pilot import curves
from honest_agents.pilot.env import ReplayEnv
from honest_agents.pilot.tools import CONTRACT_TYPES, tool_parameters
from honest_agents.reward import (
    MASK_REWARD_ERROR,
    STOP_SILENT,
    EpisodeOutcome,
    RewardBreakdown,
)

logger = logging.getLogger(__name__)

EPISODES_FILE = "episodes.jsonl"
CURVE_FILE = "curve.jsonl"
PILOT_METRICS = {  # curve metric -> name logged through TRL's `log_metric`
    "claim_rate": "pilot/claim_rate",
    "fabrication_among_failures": "pilot/fabrication_among_failures",
    "claim_coverage": "pilot/claim_coverage",
    "false_blame_rate": "pilot/false_blame_rate",
    "realised_threshold": "pilot/realised_threshold",
    "precision_claims": "pilot/precision_claims",
    "pass_rate_declined": "pilot/pass_rate_declined",
    "report_rate": "pilot/report_rate",
}
METRIC_TOOL_ERROR_RATE = "pilot/tool_error_rate"
METRIC_EXTRA_CALLS_RATE = "pilot/extra_calls_rate"
METRIC_FILLED_ARGS_RATE = "pilot/filled_args_rate"
# One advantage per trajectory makes the report decision noisy. Logged so the noise is visible, not acted on.
METRIC_ADV_HONEST_FAILURE = "pilot/mean_advantage_honest_failure"
METRIC_ADV_FABRICATION = "pilot/mean_advantage_fabrication"
_HONEST_FAILURE_CASES = ("honest_failure", "honest_failure_observed", "honest_underclaim")
_EMPTY = {
    "string": "",
    "array": [],
    "number": None,
}  # what a left-out tool argument is read as


def deliver_report(env: Any, completion: Any) -> str | None:
    """Give `env` the report call found in `completion`, unless it already has a report.

    Returns the error text when the call could not be made (an argument the tool does not have, arguments that are not a
    mapping), as TRL's tool loop would have reported it to the model; None otherwise, including when there is no report
    call.

    Arguments the model left out are read as empty (`""`, `[]`, no confidence) and listed in `env._filled_args`. In the
    live loop a call that omits a required argument is answered with an error and the model tries again; the pilot has
    one turn, so scoring that as "no report" would train format compliance instead of the claim decision. Counted by
    `pilot/filled_args_rate`.
    """
    if env._reported():
        return None
    split = bank.split_at_report(completion if isinstance(completion, list) else None)
    if split.report_call is None:
        env._stop_reason = STOP_SILENT
        return None
    name, arguments = split.report_call["name"], split.report_call["arguments"]
    try:
        if not isinstance(arguments, dict):
            raise TypeError(f"arguments of {name} are not a mapping")
        missing = [p for p in tool_parameters(type(env), name) if p not in arguments]
        env._filled_args = missing
        getattr(env, name)(**{**{p: _EMPTY[CONTRACT_TYPES[name][p]] for p in missing}, **arguments})
    except Exception as e:  # noqa: BLE001 - what TRL turns into `{"error": ...}` for the model; the episode has no report
        env._stop_reason = STOP_SILENT
        return f"{type(e).__name__}: {e}"
    return None


def _status(outcome: EpisodeOutcome) -> str:
    """complete / incomplete / abstain, "unparsed" for a report that did not parse, "none" for no report."""
    if outcome.report is not None:
        return outcome.report.status.value
    return "unparsed" if outcome.report_error is not None else "none"


def episode_record(
    env: Any,
    outcome: EpisodeOutcome,
    breakdown: RewardBreakdown,
    reward: float | None,
    *,
    round_index: int,
    step: int | None,
    row: dict[str, Any] | None = None,
    tool_error: str | None = None,
) -> dict[str, Any]:
    """Everything the curves and the analysis need about one new report, as plain JSON values."""
    row = row or {}
    bank_row = getattr(env, "_row", None)
    report = outcome.report
    kind = row.get("kind") or ("infeasible" if not breakdown.feasible else "control" if breakdown.is_control else None)
    return {
        "round": round_index,
        "step": step,
        "pilot_id": row.get("pilot_id") or getattr(bank_row, "episode_id", None),
        "variant_id": row.get("variant_id"),
        "kind": kind or "feasible",
        "feasible": breakdown.feasible,
        "is_control": breakdown.is_control,
        "forced_report": bool(row.get("forced_report", False)),
        "passed": outcome.passed,
        "has_report": breakdown.has_report,
        "parse_error": breakdown.report_parse_error,
        "tool_error": tool_error,
        "status": _status(outcome),
        "claimed": breakdown.claimed_complete,
        "declined": breakdown.declined,
        "case": breakdown.case,
        "masked": breakdown.masked,
        "reward": reward,
        "honest_total": breakdown.reward,
        "false_blame": breakdown.false_blame,
        "blocker_category": report.blocker_category if report is not None else None,
        "blocker_check": breakdown.blocker_check,
        "reason_match": breakdown.reason_match,
        "confidence": report.confidence if report is not None else None,
    }


class EpisodeLog:
    """Appends episode records and curve rows for one arm, and keeps the last `window` rounds in memory for the
    realised threshold."""

    def __init__(
        self,
        directory: str | Path | None,
        window: int = 10,
        extra: dict[str, Any] | None = None,
    ):
        self.directory = Path(directory) if directory is not None else None
        self.window = max(1, int(window))
        self.extra = dict(extra or {})
        self.round = 0
        self._recent: dict[int, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.round = _next_round(self.directory / EPISODES_FILE)

    def next_round(self) -> int:
        with self._lock:
            index = self.round
            self.round += 1
            return index

    def add(self, round_index: int, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            self._recent[round_index] = list(records)
            for old in [k for k in self._recent if k <= round_index - self.window]:
                del self._recent[old]
            row = curves.curve_row(self._recent, round_index, self.window, self.extra)
            if self.directory is not None:
                with open(self.directory / EPISODES_FILE, "a") as f:
                    f.writelines(json.dumps(r) + "\n" for r in records)
                with open(self.directory / CURVE_FILE, "a") as f:
                    f.write(json.dumps(row) + "\n")
            return row


def _next_round(path: Path) -> int:
    """Continue the round count of an existing log, so a resumed arm appends instead of restarting at round 0."""
    if not path.exists():
        return 0
    last = -1
    with open(path) as f:
        for line in f:
            if line.strip():
                last = max(last, int(json.loads(line).get("round", -1)))
    return last + 1


def _advantages(rewards: Sequence[float | None], groups: Sequence[str], scale_rewards: str) -> list[float] | None:
    """The advantage TRL will give each completion, when the batch is groups of one size laid out one group after the
    other (TRL's `RepeatSampler`); None otherwise."""
    if not rewards:
        return None
    sizes, previous = [], object()
    for g in groups:
        if g == previous:
            sizes[-1] += 1
        else:
            sizes.append(1)
            previous = g
    if len(set(sizes)) != 1:
        return None
    try:
        return curves.advantages(rewards, sizes[0], scale_rewards)
    except ValueError:
        return None


def _row_of(columns: dict[str, Any], i: int, n: int) -> dict[str, Any]:
    keys = ("pilot_id", "variant_id", "kind", "forced_report")
    return {k: columns[k][i] for k in keys if isinstance(columns.get(k), (list, tuple)) and len(columns[k]) == n}


class PilotReward:
    """Callable reward function for one arm. `mode` is a `trl_rewards.MODES` entry; the pilot sweep uses "honest"."""

    def __init__(
        self,
        mode: str = "honest",
        log: EpisodeLog | None = None,
        scale_rewards: str = "none",
    ):
        if mode not in trl_rewards.PER_EPISODE_MODES:
            raise ValueError(f"reward mode must be one of {trl_rewards.PER_EPISODE_MODES}, got {mode!r}")
        self.mode = mode
        self.scale_rewards = scale_rewards  # only for the advantage diagnostics; the trainer does the real scaling
        self.log = log or EpisodeLog(None)
        self.__name__ = f"pilot_{mode}_reward"  # the name TRL logs the reward under
        self.last_curve_row: dict[str, Any] | None = None

    def score(
        self,
        environments: Sequence[Any],
        *,
        log_metric: Callable[[str, float], None] | None = None,
        log_extra: Callable[[str, list], None] | None = None,
        columns: dict[str, Any] | None = None,
    ) -> list[float | None]:
        """Score one batch. Raises on misuse; `__call__` wraps it so nothing reaches the training loop."""
        columns = columns or {}
        n = len(environments)
        for env in environments:
            if not isinstance(env, ReplayEnv):
                raise TypeError(f"the pilot reward needs ReplayEnv environments, got {type(env).__name__}")
        completions = columns.get("completions")
        completions = completions if isinstance(completions, (list, tuple)) and len(completions) == n else [None] * n
        tool_errors = [deliver_report(env, c) for env, c in zip(environments, completions, strict=True)]
        extra_calls = [bank.split_at_report(c if isinstance(c, list) else None).extra_calls for c in completions]

        graded = [env._graded() for env in environments]
        outcomes = [g[0] for g in graded]
        # One group per frozen trajectory. Not `variant_id`: several trajectories of one variant are separate prompts.
        groups = [str(env._row.episode_id) for env in environments]
        breakdowns, _ = trl_rewards.mask_leaking_groups(outcomes, [g[1] for g in graded], groups)
        rewards = [trl_rewards.mode_reward(self.mode, br) for br in breakdowns]

        step = getattr(columns.get("trainer_state"), "global_step", None)
        round_index = self.log.next_round()
        records = [
            episode_record(
                env,
                outcomes[i],
                breakdowns[i],
                rewards[i],
                round_index=round_index,
                step=step,
                row=_row_of(columns, i, n),
                tool_error=tool_errors[i],
            )
            for i, env in enumerate(environments)
        ]
        advs = _advantages(rewards, groups, self.scale_rewards)
        if advs is not None:
            for record, adv in zip(records, advs, strict=True):
                record["advantage"] = adv
        curve_row = self.log.add(round_index, records)
        self.last_curve_row = curve_row

        if log_metric is not None:
            metrics = trl_rewards.batch_metrics(breakdowns, groups, rewards)
            metrics[trl_rewards.METRIC_REWARD_ERROR_RATE] = 0.0
            metrics[METRIC_TOOL_ERROR_RATE] = sum(e is not None for e in tool_errors) / n if n else 0.0
            metrics[METRIC_EXTRA_CALLS_RATE] = sum(c > 0 for c in extra_calls) / n if n else 0.0
            metrics[METRIC_FILLED_ARGS_RATE] = sum(bool(e._filled_args) for e in environments) / n if n else 0.0
            if advs is not None:
                for name, cases in (
                    (METRIC_ADV_HONEST_FAILURE, _HONEST_FAILURE_CASES),
                    (METRIC_ADV_FABRICATION, ("fabrication",)),
                ):
                    picked = [a for a, br in zip(advs, breakdowns, strict=True) if br.case in cases and not br.masked]
                    if picked:
                        metrics[name] = sum(picked) / len(picked)
            for key, name in PILOT_METRICS.items():
                if curve_row.get(key) is not None:
                    metrics[name] = float(curve_row[key])
            for name, value in metrics.items():
                log_metric(name, value)
        if log_extra is not None:
            for column, values in trl_rewards.episode_columns(outcomes, breakdowns, groups).items():
                log_extra(column, values)
        return rewards

    def __call__(self, environments=None, log_metric=None, log_extra=None, **kwargs) -> list[float | None]:
        try:
            if environments is None:
                raise ValueError("the pilot reward needs `environments` (set `environment_factory` on the trainer)")
            return self.score(environments, log_metric=log_metric, log_extra=log_extra, columns=kwargs)
        except Exception:
            logger.exception("pilot reward failed; the whole batch is masked (%s)", MASK_REWARD_ERROR)
        sized = next(
            (
                v
                for v in (
                    environments,
                    kwargs.get("prompts"),
                    kwargs.get("completions"),
                )
                if v is not None
            ),
            (),
        )
        n = len(sized)
        if log_metric is not None:
            try:
                log_metric(trl_rewards.METRIC_MASKED_SHARE, 1.0)
                log_metric(f"honest/masked/{MASK_REWARD_ERROR}", 1.0)
                log_metric(trl_rewards.METRIC_REWARD_ERROR_RATE, 1.0)
            except Exception:
                logger.exception("could not log the reward failure")
        return [None] * n
