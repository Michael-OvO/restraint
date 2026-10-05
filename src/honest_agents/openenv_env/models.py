"""Wire models for the honest-agents OpenEnv environment.

One action type with a `kind` discriminator mirrors the three tools of the training harness (`bash`, `submit_report`,
`abstain`) argument for argument (`honest_agents.harness.ReportTools`; `tests/openenv/test_parity.py` checks it), so a
policy prompted for training can be pointed at this server unchanged. A report names at most ONE blocker: its category,
the specific object (`blocker_target`) and a detail. Report fields are typed `Any` (a list, a string or junk) because the core
parser (`honest_agents.report`) deliberately tolerates what small models emit; a malformed report must end the episode
as a report error, not be rejected at the transport layer.

`kind="end"` is not a tool. It is how a client says the model stopped without reporting, which the TRL loop sees for
itself; the episode is then graded as it stands.
"""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

ActionKind = Literal["bash", "submit_report", "abstain", "end"]
# Tool argument names per report kind, in the order of the tool signatures.
REPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "submit_report": (
        "status",
        "completed",
        "incomplete",
        "blocker_category",
        "blocker_target",
        "blocker_detail",
        "evidence_commands",
        "confidence",
    ),
    "abstain": ("reason", "blocker_category", "blocker_target", "confidence"),
}


class HonestAction(Action):
    kind: ActionKind = Field(description="Which tool to call: bash | submit_report | abstain; or end (no report)")
    # bash
    command: str = Field(default="", description="bash: shell command to run (fresh shell per call)")
    # submit_report
    status: Any = Field(default=None, description='submit_report: "complete" or "incomplete"')
    completed: Any = Field(default=None, description="submit_report: requirements completed")
    incomplete: Any = Field(default=None, description="submit_report: requirements not completed")
    blocker_detail: Any = Field(
        default=None,
        description="submit_report: one sentence on what the blocker prevents",
    )
    evidence_commands: Any = Field(
        default=None,
        description="submit_report: commands that exit 0 only if the completed claims are true",
    )
    # abstain
    reason: Any = Field(default=None, description="abstain: why the task cannot or should not be done")
    # submit_report and abstain: the single blocker
    blocker_category: Any = Field(default=None, description="the ONE blocker category that applies; empty if none")
    blocker_target: Any = Field(
        default=None,
        description="the specific object blocked on: a command name, a path or a host; empty if none",
    )
    # submit_report: P(task fully done); abstain: P(task could not have been done)
    confidence: Any = Field(default=None, description="probability in [0, 1]")
    # end
    stop_reason: Literal["silent_stop", "truncated"] | None = Field(
        default=None,
        description="end: why the model stopped without a report (default silent_stop)",
    )

    def tool_args(self) -> dict[str, Any]:
        """The report tool's arguments as the model gave them: only the fields that were set, like a tool call."""
        return {k: v for k in REPORT_FIELDS.get(self.kind, ()) if (v := getattr(self, k)) is not None}


class HonestObservation(Observation):
    """`output` is what the policy sees as tool output; everything else is for the harness.

    On the terminal observation `reward` is the episode's honesty reward (reward v3, `honest_agents.reward`), or None
    when the episode is masked: the sandbox or the verifier toolchain failed before the agent ran a command, or an
    "infeasible" variant was solved (a leak, so the label is wrong). `metadata` then carries `reward_breakdown`,
    `outcome` (including the task's labels and the blocker verdict), `masked_reason`, `stop_reason`, `grading_error`
    and `debug`.
    """

    output: str = Field(default="", description="Tool output (the instruction on reset)")
    kind: str = Field(
        default="",
        description="Action kind that produced this observation ('reset' on reset)",
    )
    exit_code: int | None = Field(default=None, description="bash exit code; None if the command did not run")
    instruction: str = Field(default="", description="Task instruction (reset only)")
    system_prompt: str = Field(default="", description="Harness system prompt used in training (reset only)")
    task_id: str = Field(default="")
    steps_remaining: int | None = Field(default=None, description="bash calls left; bash refuses to run at 0")
    chars_remaining: int | None = Field(
        default=None,
        description="tool-output characters left in the episode budget; bash refuses to run at 0",
    )
    turns_remaining: int | None = Field(
        default=None,
        description="actions left before the episode is cut off and graded without a report",
    )


class HonestState(State):
    """Session state. Never exposes the task's honesty labels (feasible, perturbation, expected blockers or targets)."""

    task_id: str = ""
    sandbox_ready: bool = False
    n_commands: int = 0
    max_steps: int | None = None  # tool-call turns per episode (TRL's `max_tool_calling_iterations`)
    max_bash_calls: int | None = None
    report_submitted: bool = False
    done: bool = False
    reward: float | None = None
    case: str | None = None
    stop_reason: str | None = None
    masked_reason: str | None = None
