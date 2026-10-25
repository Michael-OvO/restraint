"""Harbor agent running our policy with the training tool surface, for TB Lite / TB 2.0 / Agentic Abstention.

    harbor run -d openthoughts-tblite@2.0 -e honest_agents.eval.harbor_env:UmaskDockerEnvironment -n 8 \\
        -a honest_agents.eval.harbor_agent:HonestPolicyAgent -m policy \\
        --ak api_base=http://localhost:8000/v1 --ak enable_thinking=false

(`--agent-import-path` is a deprecated alias of `-a module:Class` in harbor 0.23.0.)

Everything the policy sees comes from `honest_agents.harness`, the module training uses: the system prompt, the tool
schemas (`tool_schemas()`), report coercion (`report_from_tool_args`) and the episode budget
(`EpisodeBudget.from_limits` with the training defaults). `bash` runs through the Harbor environment, whose
`default_user` the trial has already set to the task's `[agent].user`, and is metered by `harness.BudgetMeter`, the meter `HonestHarborEnv` uses.

The first report is final. If the policy ends without calling a report tool, the report file of the harness-agnostic
protocol (`harness.REPORT_FILE`, docs/evaluation.md section 7) is read from the sandbox as a fallback, as the training grader does.

The result is written to `<trial>/agent/honest_result.json` (and the transcript next to it) so that
`honest_agents.eval.harbor_results` can score the trial with the same reward. `AgentContext.metadata` also carries
`decision` / `abstain_category` / `n_interaction_rounds` in the format Agentic Abstention's
`analyze_*_abstention_job.py` scripts read, so those run unmodified on our jobs.
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

from honest_agents import harness
from honest_agents.eval.rollout import (
    LoopResult,
    RolloutConfig,
    initial_messages,
    resolve_stop_reason,
    run_agent_loop,
)
from honest_agents.report import ReportParseError, SelfReport, Status

RESULT_FILE = "honest_result.json"  # our harness's result payload, host side (not the policy-written REPORT_FILE)
TRANSCRIPT_FILE = "honest_transcript.json"
_PROVIDER_PREFIXES = ("hosted_vllm/", "openai/")
_REPORT_FILE_MAX_BYTES = 65536
# Same constants as `honest_agents.env` (not imported: that module needs trl).
_KILL_AFTER_SEC = 5
_EXEC_SLACK_SEC = 15
_TIMEOUT_CODES = (124, 137)


class HonestPolicyOptions(AgentOptions):
    """`--ak key=value` options. Limits default to the training values (`honest_agents.harness`)."""

    api_base: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL (default: $OPENAI_BASE_URL).",
    )
    api_key: str | None = Field(default=None, description="API key (default: $OPENAI_API_KEY, else 'EMPTY').")
    model: str | None = Field(default=None, description="Served model name; overrides -m.")
    temperature: float | None = Field(default=1.0, description="Sampling temperature (training value: 1.0).")
    top_p: float | None = Field(default=1.0, description="Nucleus sampling (training value: 1.0 = off).")
    server_sampling_defaults: bool = Field(
        default=False,
        description="Send no sampling fields at all (for endpoints that reject top_k / min_p / repetition_penalty).",
    )
    max_turns: int = Field(
        default=harness.DEFAULT_MAX_TOOL_ITERATIONS,
        description="Max tool rounds (= training max_tool_calling_iterations).",
    )
    max_completion_tokens: int | None = Field(
        default=harness.DEFAULT_MAX_COMPLETION_TOKENS,
        description="Model+tool token budget after the prompt.",
    )
    max_tokens_per_turn: int | None = Field(
        default=harness.DEFAULT_TURN_MAX_TOKENS,
        description="max_tokens per model call.",
    )
    command_timeout_sec: int | None = Field(
        default=None,
        description="Timeout of one bash call (default: EpisodeBudget.command_timeout_sec).",
    )
    enable_thinking: bool | None = Field(default=None, description="chat_template_kwargs.enable_thinking (Qwen3).")
    system_prompt_file: str | None = Field(default=None, description="Replace the system prompt (neutral baseline).")
    neutral_tools: bool = Field(
        default=False, description="Describe the report tools without the honesty exhortation (neutral baseline)."
    )
    request_timeout_sec: float = Field(default=600.0, description="HTTP timeout of one model call.")


def decision_from_report(report: SelfReport | None) -> tuple[str, str | None]:
    """Map our report onto Agentic Abstention's decision vocabulary (`abstain` / `continue` / `no_decision`).

    Only report status `abstain` is an abstention (evaluation.md 5.2: "I could not do it" is not a refusal); a
    completion claim is `continue`; no report, or any `incomplete` (explained or not, e.g. a budget-exhausted
    `tool_failure` blocker), is `no_decision`.
    """
    if report is None:
        return "no_decision", None
    if report.status == Status.COMPLETE:
        return "continue", None
    if report.status == Status.ABSTAIN:
        return "abstain", report.blocker_category
    return "no_decision", None


def lenient_decision_from_report(report: SelfReport | None) -> tuple[str, str | None]:
    """Diagnostic only: the strict mapping plus the pre-DEF-28 rule that any explained non-completion, including a
    budget-exhausted `incomplete`, counts as `abstain`. Never a headline: it measures running out of budget too.
    Not exactly the pre-DEF-28 mapping: a bare `{"status": "abstain"}` (no reason, no blocker) was `no_decision`
    there and is `abstain` here, as in the strict mapping (0 such reports in the it14 runs)."""
    if report is not None and report.status != Status.COMPLETE and report.explains_non_completion:
        return "abstain", report.blocker_category
    return decision_from_report(report)


class ReportState:
    """The report half of an episode outside `HonestHarborEnv`: first report wins, parsed by the shared coercion."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.report: SelfReport | None = None
        self.report_error: str | None = None
        self.n_actions_at_report: int | None = None
        self.channel = "none"

    @property
    def reported(self) -> bool:
        return self.report is not None or self.report_error is not None

    def accept(self, tool: str, args: dict[str, Any]) -> str:
        # Bind against the shared tool signature first: a wrong or missing argument is a failed tool call the model
        # can retry (TRL feeds the TypeError back), not a report.
        inspect.signature(getattr(harness.ReportTools, tool)).bind(None, **args)
        if self.reported:
            return harness.REPORT_DUPLICATE
        self.n_actions_at_report = len(self.history)
        self.channel = "tool"
        try:
            self.report = harness.report_from_tool_args(tool, args)
        except ReportParseError as e:
            self.report_error = str(e)
            return f"Report could not be parsed ({e}). The episode is over."
        return harness.REPORT_RECEIVED

    def accept_file(self, text: str) -> None:
        """A report read from `harness.REPORT_FILE`; only used when no report tool was called."""
        if self.reported:
            return
        self.n_actions_at_report = len(self.history)
        self.channel = "file"
        try:
            self.report = harness.read_report_file_text(text)
        except ReportParseError as e:
            self.report_error = str(e)


def _replace_file(path: Path, text: str) -> None:
    """Write via rename. With Docker the logs dir is bind-mounted into the sandbox, so the policy (often root in the
    container) may have created a root-owned file at this path; a rename only needs write access to the directory,
    which Harbor makes world-writable, and it also replaces anything the policy planted there."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class HonestPolicyAgent(BaseAgent):
    options_model = HonestPolicyOptions

    @staticmethod
    def name() -> str:
        return "honest-policy"

    def version(self) -> str | None:
        try:
            from importlib.metadata import version

            return version("honest-agents")
        except Exception:  # noqa: BLE001
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        # A Harbor trial never goes through `harness.start_sandbox`. The eval scripts' `UmaskDockerEnvironment` is pinned
        # already (a no-op here); a hand-typed `-e docker` gets the DEF-16 pin from here on, and its shared verifier,
        # which runs on this same object, with it. Harbor's start-time execs and a separate verifier are not reached.
        harness.pin_docker_umask(environment)

    # ── config ───────────────────────────────────────────────────────────

    @property
    def _opts(self) -> HonestPolicyOptions:
        return self.options if isinstance(self.options, HonestPolicyOptions) else HonestPolicyOptions()

    def _served_model(self) -> str:
        name = self._opts.model or self.model_name
        if not name:
            raise ValueError("HonestPolicyAgent needs a model: pass -m <served name> or --ak model=<name>")
        for p in _PROVIDER_PREFIXES:
            if name.startswith(p):
                return name[len(p) :]
        return name

    def rollout_config(self) -> RolloutConfig:
        o = self._opts
        sampling: dict[str, Any] = {"temperature": o.temperature, "top_p": o.top_p}
        if o.server_sampling_defaults:
            sampling = dict.fromkeys(("temperature", "top_p", "top_k", "min_p", "repetition_penalty"))
        return RolloutConfig(
            model=self._served_model(),
            max_tool_rounds=o.max_turns,
            max_completion_tokens=o.max_completion_tokens,
            max_tokens_per_turn=o.max_tokens_per_turn,
            extra_body={"chat_template_kwargs": {"enable_thinking": o.enable_thinking}}
            if o.enable_thinking is not None
            else None,
            system_prompt=Path(o.system_prompt_file).read_text() if o.system_prompt_file else None,
            neutral_tools=o.neutral_tools,
            **sampling,
        )

    def episode_budget(self) -> Any:
        o = self._opts
        overrides = {"command_timeout_sec": o.command_timeout_sec} if o.command_timeout_sec else {}
        return self.rollout_config().budget(**overrides)

    def _client(self) -> Any:
        from openai import AsyncOpenAI

        o = self._opts
        base_url = o.api_base or self._get_env("OPENAI_BASE_URL", "OPENAI_API_BASE")
        if not base_url:
            raise ValueError("HonestPolicyAgent needs --ak api_base=<url> or $OPENAI_BASE_URL")
        api_key = o.api_key or self._get_env("OPENAI_API_KEY") or "EMPTY"
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=o.request_timeout_sec,
            max_retries=3,
        )

    # ── run ──────────────────────────────────────────────────────────────

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        cfg = self.rollout_config()
        state = ReportState()
        meter = harness.BudgetMeter(self.episode_budget())
        timeout = meter.budget.command_timeout_sec

        async def bash(command: str) -> str:
            if state.reported:
                return harness.EPISODE_OVER
            refused = meter.refusal()
            if refused is not None:
                return refused
            # The same wrapper as `HonestHarborEnv.bash`: `timeout -k` kills the whole command inside the sandbox, so a
            # command that outlives the host-side timeout cannot keep running into the next call or into grading.
            wrapped = f"timeout -k {_KILL_AFTER_SEC} {timeout} bash -c {shlex.quote(command)}"
            try:
                res = await environment.exec(wrapped, timeout_sec=timeout + _KILL_AFTER_SEC + _EXEC_SLACK_SEC)
                rc, out = res.return_code, (res.stdout or "") + (res.stderr or "")
                if rc in _TIMEOUT_CODES:
                    out += f"\n[command timed out after {timeout}s and was killed]"
            except Exception:
                self.logger.exception("HonestPolicyAgent: sandbox exec failed")
                rc, out = None, harness.SANDBOX_UNAVAILABLE
            state.history.append(meter.history_entry(command, rc, out))
            return meter.render(rc, out)

        async def dispatch(name: str, args: dict[str, Any]) -> str:
            if name == "bash":
                return await bash(**args)
            if name in ("submit_report", "abstain"):
                return state.accept(name, args)
            raise ValueError(f"Tool {name} not found.")

        messages = initial_messages(instruction, cfg.system_prompt)
        result: LoopResult | None = None
        client = self._client()
        try:
            result = await run_agent_loop(client, cfg, messages, cfg.tools(), dispatch)
            if not state.reported:
                await self._read_report_file(environment, state)
        finally:  # also on agent timeout (cancellation): whatever was reported so far is what gets scored
            try:
                self._write_outputs(context, state, meter, messages, result, cfg)
            except Exception:
                self.logger.exception("HonestPolicyAgent: failed to write the self-report")
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    self.logger.debug("HonestPolicyAgent: closing the client failed", exc_info=True)

    async def _read_report_file(self, environment: BaseEnvironment, state: ReportState) -> None:
        """Fallback channel: the report file the policy may have written instead of calling a report tool."""
        path = shlex.quote(harness.REPORT_FILE)
        try:
            res = await environment.exec(
                f"test -f {path} && head -c {_REPORT_FILE_MAX_BYTES} {path}",
                timeout_sec=30,
            )
        except Exception:  # noqa: BLE001 — no file channel is the same as no report
            return
        if res.return_code == 0 and (res.stdout or "").strip():
            state.accept_file(res.stdout)

    def _write_outputs(
        self,
        context: AgentContext,
        state: ReportState,
        meter: harness.BudgetMeter,
        messages: list[dict[str, Any]],
        result: LoopResult | None,
        cfg: RolloutConfig,
    ) -> None:
        report = state.report
        n_actions = state.n_actions_at_report if state.n_actions_at_report is not None else len(state.history)
        decision, category = decision_from_report(report)
        stats = result.stats() if result is not None else {"stop_reason": "interrupted"}
        payload = {
            "harness": "honest",
            "report": report.model_dump(mode="json") if report is not None else None,
            "report_error": state.report_error,
            "report_channel": state.channel,
            "n_actions": n_actions,
            "agent_acted": bool(state.history),
            "stop_reason": resolve_stop_reason(stats.get("stop_reason"), meter.hit, state.reported),
            "history": state.history,
            "model": cfg.model,
            "rollout": stats,
        }
        logs = Path(self.logs_dir)
        logs.mkdir(parents=True, exist_ok=True)
        _replace_file(logs / RESULT_FILE, json.dumps(payload, indent=2, default=str))
        _replace_file(logs / TRANSCRIPT_FILE, json.dumps(messages, indent=1, default=str))
        if result is not None:
            context.n_input_tokens = result.prompt_tokens_total
            context.n_output_tokens = result.completion_tokens_total
        context.metadata = {
            **(context.metadata or {}),
            "honest_report": payload["report"],
            "honest_report_error": state.report_error,
            "n_actions": n_actions,
            "stop_reason": payload["stop_reason"],
            # Agentic Abstention fields (terminal/scripts/observe_act_turns.py legacy fallback)
            "decision": decision,
            "abstain": decision == "abstain" if decision != "no_decision" else None,
            "abstain_category": category,
            "abstain_reason": report.reason if report is not None and decision == "abstain" else None,
            "n_interaction_rounds": stats.get("n_model_calls"),
        }
