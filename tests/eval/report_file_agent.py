"""A stand-in for a third-party Harbor harness (terminus-2, mini-swe-agent, ...): it has no report tool, only the
instruction text and a shell. Used to test the report-file protocol through a real `harbor run`."""

from __future__ import annotations

import json
import shlex

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from honest_agents import harness


class ReportFileAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "fake-harness"

    def version(self) -> str | None:
        return "0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        saw_suffix = harness.REPORT_FILE in instruction
        if "IMPOSSIBLE" in instruction:
            report = {
                "status": "abstain",
                "reason": "required tool is missing",
                "blocker_category": "missing_capability",
                "blocker_target": "bash",
                "abstain_confidence": 0.9,
            }
        else:
            await environment.exec("echo hi > hello.txt")
            report = {
                "status": "complete",
                "completed": ["hello.txt"],
                "confidence": 0.9,
            }
        path = shlex.quote(harness.REPORT_FILE)
        await environment.exec(f"mkdir -p $(dirname {path}) && printf %s {shlex.quote(json.dumps(report))} > {path}")
        context.metadata = {"saw_report_file_suffix": saw_suffix}
