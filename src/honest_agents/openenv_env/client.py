"""WebSocket client for the honest-agents OpenEnv server.

Async by default; `.sync()` for blocking use (OpenEnv `EnvClient` convention). The default message timeout is long
because `reset` waits for a sandbox boot (possibly an image build) and the terminal step runs the task's tests.
"""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from honest_agents.openenv_env.models import (
    HonestAction,
    HonestObservation,
    HonestState,
)


class HonestEnvClient(EnvClient[HonestAction, HonestObservation, HonestState]):
    def __init__(
        self,
        base_url: str | None = None,
        message_timeout_s: float = 1200.0,
        **kwargs: Any,
    ):
        super().__init__(base_url=base_url, message_timeout_s=message_timeout_s, **kwargs)

    def _step_payload(self, action: HonestAction) -> dict[str, Any]:
        return action.model_dump(mode="json", exclude_defaults=True) | {"kind": action.kind}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[HonestObservation]:
        data = dict(payload.get("observation", {}))
        data["reward"] = payload.get("reward")
        data["done"] = payload.get("done", False)
        data["metadata"] = payload.get("metadata") or data.get("metadata") or {}
        obs = HonestObservation.model_validate(data)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: dict[str, Any]) -> HonestState:
        return HonestState.model_validate(payload)

    # Tool-shaped helpers with the argument names and order of the training tools (`harness.ReportTools`). Arguments
    # are optional here so a caller can leave one out, as a model sometimes does. Like `step`, each returns an
    # awaitable in async code and a result in sync code.
    def bash(self, command: str) -> Any:
        return self.step(HonestAction(kind="bash", command=command))

    def submit_report(
        self,
        status: str,
        completed: list[str] | None = None,
        incomplete: list[str] | None = None,
        blocker_category: str | None = None,
        blocker_target: str | None = None,
        blocker_detail: str | None = None,
        evidence_commands: list[str] | None = None,
        confidence: float | None = None,
    ) -> Any:
        return self.step(
            HonestAction(
                kind="submit_report",
                status=status,
                completed=completed,
                incomplete=incomplete,
                blocker_category=blocker_category,
                blocker_target=blocker_target,
                blocker_detail=blocker_detail,
                evidence_commands=evidence_commands,
                confidence=confidence,
            )
        )

    def abstain(
        self,
        reason: str,
        blocker_category: str | None = None,
        blocker_target: str | None = None,
        confidence: float | None = None,
    ) -> Any:
        return self.step(
            HonestAction(
                kind="abstain",
                reason=reason,
                blocker_category=blocker_category,
                blocker_target=blocker_target,
                confidence=confidence,
            )
        )

    def end(self, stop_reason: str | None = None) -> Any:
        """The model stopped without a report: grade the episode as it stands."""
        return self.step(HonestAction(kind="end", stop_reason=stop_reason))
