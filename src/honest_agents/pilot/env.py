"""`ReplayEnv`: a TRL `environment_factory` environment with no sandbox (SPEC_V3 §6).

One episode is one new report on a frozen trajectory. `reset(**row)` loads the trajectory's bank row and returns
nothing, because the dataset prompt already holds the whole transcript up to the point where the agent must report.
The only tools are `submit_report` and `abstain`, inherited from the shared harness. Grading is a pure function call:
the trajectory's cached outcome with the new report in place of the old one, scored by `compute_reward`
(`honest_agents.bank.rescored_outcome`, the function behind `bank.rescore`).

The class keeps the private names `honest_agents.trl_rewards` reads from an environment (`_start_grading`, `_graded`,
`_reward_config`, `_spec`, `_task_dir`, `_history`, `_report`, `_n_actions_at_report`, `_stop_reason`), so the same
metric code scores a pilot batch and a live batch. It must not grow other public methods: TRL exposes every public
method except `reset` as a tool.
"""

from __future__ import annotations

import concurrent.futures
import json
from typing import Any

from honest_agents import bank, harness
from honest_agents.bank import BankRow
from honest_agents.pilot.tools import ReportTools
from honest_agents.report import ReportParseError, SelfReport
from honest_agents.reward import (
    STOP_REPORT,
    STOP_SILENT,
    EpisodeOutcome,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
)
from honest_agents.spec import HonestTaskSpec

BANK_ROW_COLUMN = "bank_row"  # dataset column holding the bank row as a JSON string
REPORT_RECEIVED = harness.REPORT_RECEIVED
REPORT_DUPLICATE = harness.REPORT_DUPLICATE
# The live environment's lenient coercion (clamps an out-of-range confidence), so a report parses here as it does there.
_coerce_report = harness.coerce_report


def load_bank_row(value: Any) -> BankRow:
    """A `BankRow` from itself, its dict, or its JSON string (the form a dataset column carries)."""
    if isinstance(value, BankRow):
        return value
    if isinstance(value, (str, bytes)):
        return BankRow.model_validate_json(value)
    if isinstance(value, dict):
        return BankRow.model_validate(value)
    raise TypeError(f"`{BANK_ROW_COLUMN}` must be a BankRow, a dict or a JSON string, got {type(value).__name__}")


class ReplayEnv(ReportTools):  # type: ignore[misc, valid-type]
    def __init__(self, reward_config: RewardConfig | None = None):
        self._reward_config = reward_config or RewardConfig()
        self._row: BankRow | None = None
        self._spec: HonestTaskSpec | None = None
        self._task_dir: str | None = None
        self._clear_episode()

    def _clear_episode(self) -> None:
        self._history: list[dict[str, Any]] = []  # no command ever runs here; kept for the reward-side contract
        self._report: SelfReport | None = None
        self._report_error: str | None = None
        self._raw_report: dict | None = None
        self._filled_args: list[str] = []  # report arguments the model left out (`rewards.deliver_report`)
        self._n_actions_at_report: int | None = None
        self._stop_reason: str | None = None
        self._result: tuple[EpisodeOutcome, RewardBreakdown, dict] | None = None

    # ── TRL contract ─────────────────────────────────────────────────────

    def reset(self, bank_row: Any = None, **row) -> None:
        """Load the frozen trajectory. Returns None: TRL then leaves the prompt, which already ends at the report
        point, untouched."""
        if bank_row is None:
            raise ValueError(f"ReplayEnv.reset requires `{BANK_ROW_COLUMN}` (a column of the pilot dataset).")
        self._clear_episode()
        self._row = load_bank_row(bank_row)
        self._task_dir = self._row.task.get("task_dir")
        self._spec = _spec_of(self._row)
        self._n_actions_at_report = int(self._row.outcome.get("n_actions") or 0)

    # ── internals (the tools themselves come from `ReportTools`) ─────────

    def _accept_report(self, raw: dict) -> str:
        if self._row is None:
            raise RuntimeError("reset() was not called")
        if self._report is not None or self._report_error is not None:
            return REPORT_DUPLICATE
        self._raw_report = dict(raw)
        self._stop_reason = STOP_REPORT
        self._result = None
        try:
            self._report = _coerce_report(raw)
        except ReportParseError as e:
            self._report_error = str(e)
            return f"Report could not be parsed ({e}). The episode is over."
        return REPORT_RECEIVED

    def _reported(self) -> bool:
        return self._report is not None or self._report_error is not None

    def _grade(self) -> tuple[EpisodeOutcome, RewardBreakdown, dict]:
        if self._row is None:
            raise RuntimeError("reset() was not called")
        outcome = bank.rescored_outcome(self._row, self._report, report_error=self._report_error)
        if not self._reported():
            # The new episode ended without a report. Whatever ended the banked trajectory no longer applies.
            outcome.stop_reason = self._stop_reason or STOP_SILENT
        breakdown = compute_reward(outcome, self._reward_config)
        debug = {
            "history": [],
            "blocker_facts": self._row.blocker_facts.model_dump(),
            "stop_reason": outcome.stop_reason,
            "test_log": "",
            "episode_id": self._row.episode_id,
            "replay": True,
        }
        return outcome, breakdown, debug

    def _start_grading(self) -> concurrent.futures.Future:
        """A finished future, to satisfy the environment contract: there is nothing to wait for without a sandbox."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(self._graded())
        except Exception as e:  # noqa: BLE001 - the caller sees it when it asks for the result, as with a live environment
            future.set_exception(e)
        return future

    def _graded(self) -> tuple[EpisodeOutcome, RewardBreakdown, dict]:
        if self._result is None:
            self._result = self._grade()
        return self._result


def _spec_of(row: BankRow) -> HonestTaskSpec | None:
    """The task's `[metadata.honest]` as stored in the bank row, when it still validates."""
    spec = row.task.get("spec")
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError:
            return None
    if not isinstance(spec, dict):
        return None
    try:
        return HonestTaskSpec.model_validate(spec)
    except ValueError:
        return None
