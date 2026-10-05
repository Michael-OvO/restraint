"""`HonestSession`: one honest-agents episode over a Harbor sandbox, with no TRL (and no OpenEnv) dependency.

The OpenEnv server cannot import `honest_agents.env` (it needs trl, whose training stack conflicts with openenv), so
this module drives the same episode from the trl-free core, through the contract in `docs/INTERFACES_V3.md`:

* budgets come from `harness.EpisodeBudget.from_limits`, built from the same two numbers as training;
* reports are parsed by `harness.report_from_tool_args` (one blocker with a target) or read from the report file;
* boot runs `grading.snapshot` and the verifier toolchain `grading.self_check` before the agent acts;
* `grading.grade` runs the state checks, the blocker check and the verifier, and `reward.compute_reward` scores it.

Only what trl forces us to restate lives here: the boot sequence and the `bash` exec wrapper; the budget is `harness.BudgetMeter`.

All coroutines of one session must run on a single event loop: Harbor's sandbox clients are bound to the loop that
created them.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import grading, harness
from honest_agents.grading import GradingConfig
from honest_agents.harness import (  # noqa: F401 — re-exported for the server and the tests
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    EPISODE_OVER,
    REPORT_DUPLICATE,
    REPORT_FILE,
    REPORT_RECEIVED,
    SANDBOX_UNAVAILABLE,
    SYSTEM_PROMPT,
    BudgetMeter,
    EpisodeBudget,
    format_bash_output,
    harbor_names,
    read_instruction,
    report_from_tool_args,
    set_max_concurrent_boots,
    start_sandbox,
)
from honest_agents.report import ReportParseError, SelfReport
from honest_agents.reward import (
    MASK_BOOT_FAILURE,
    MASK_REWARD_ERROR,
    MASK_TOOLCHAIN,
    STOP_CALL_BUDGET,
    STOP_CHAR_BUDGET,
    STOP_REPORT,
    STOP_SILENT,
    STOP_TRUNCATED,
    EpisodeOutcome,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
    outcome_to_dict,
)
from honest_agents.spec import HonestTaskSpec

_log = logging.getLogger(__name__)

REPORT_TOOLS = ("submit_report", "abstain")
_KILL_AFTER_SEC = 5  # `timeout -k`: SIGKILL this long after SIGTERM
_EXEC_SLACK_SEC = 15  # the host-side exec timeout is only a backstop behind `timeout` inside the sandbox
_TIMEOUT_CODES = (124, 137)  # GNU timeout: 124 = timed out, 137 = had to SIGKILL
_ABS_PATH = re.compile(r"(?<![\w.~$])/[\w.+\-]+(?:/[\w.+\-]+)*")
_MAX_FACT_PATHS = 64


@dataclass
class SessionConfig:
    environment_type: str = "docker"
    enforce_network_policy: bool = True
    # The same two numbers training and evaluation build their budget from (SPEC_V3 §9.3.2).
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    budget: EpisodeBudget | None = None  # None = `EpisodeBudget.from_limits` of the two numbers above
    reward_config: RewardConfig | None = None  # None = defaults, with the efficiency reference set to the call budget
    grading_config: GradingConfig = field(default_factory=GradingConfig)
    # Where per-episode Harbor `TrialPaths` live. None = system temp dir. Must be a host-identical path when the server
    # runs in a container that drives the host's Docker daemon through a mounted socket.
    trials_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.budget is None:
            self.budget = EpisodeBudget.from_limits(self.max_tool_iterations, self.max_completion_tokens)
        if self.reward_config is None:
            self.reward_config = RewardConfig(efficiency_ref_actions=max(1, self.budget.max_bash_calls))


@dataclass
class GradeResult:
    outcome: EpisodeOutcome
    breakdown: RewardBreakdown
    debug: dict[str, Any]

    @property
    def reward(self) -> float | None:
        """None only when the episode is masked (SPEC_V3 §1): a boot or toolchain failure before the agent acted, or
        a leak. Everything the policy can cause is scored."""
        return self.breakdown.reward

    def to_metadata(self) -> dict[str, Any]:
        """JSON-native summary (labels included — only ever revealed after the episode is over)."""
        return {
            "reward_breakdown": _jsonable(self.breakdown.to_dict()),
            "outcome": _jsonable(outcome_to_dict(self.outcome)),
            "masked_reason": self.breakdown.masked_reason,
            "stop_reason": self.outcome.stop_reason,
            "grading_error": self.outcome.grading_error,
            "debug": _jsonable(self.debug),
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def format_output(rc: int | None, out: str, limit: int) -> str:
    """Tool-output text (the shared harness format), kept here under its old name for the tests."""
    return format_bash_output(rc, out, limit)


class HonestSession:
    """Drives one episode at a time; `start` again (or `stop`) to tear the previous sandbox down."""

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        self.budget: EpisodeBudget = self.config.budget
        self.spec: HonestTaskSpec | None = None
        self.task_dir: Path | None = None
        self.instruction: str = ""
        self._env = None  # harbor BaseEnvironment
        self._task = None
        self._paths = None
        self._plan = None
        self._baseline = None
        self._clear_episode()

    def _clear_episode(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.report: SelfReport | None = None
        self.report_error: str | None = None
        self._n_actions_at_report: int | None = None
        self._meter = BudgetMeter(self.budget)  # the shared budget rules (harness), one meter per episode
        self._ended: str | None = None  # stop reason of an episode that ended without a report tool call
        self._boot_error: str | None = None
        self._boot_mask: str | None = None
        self._result: GradeResult | None = None

    @property
    def report_submitted(self) -> bool:
        return self.report is not None or self.report_error is not None

    @property
    def n_commands(self) -> int:
        return len(self.history)

    @property
    def calls_remaining(self) -> int:
        return self._meter.calls_left

    @property
    def chars_remaining(self) -> int:
        return self._meter.chars_left

    @property
    def budget_exhausted(self) -> str | None:
        """The stop reason of the budget that is used up, if any (`bash` refuses to run from then on)."""
        return self._meter.exhausted

    @property
    def stop_reason(self) -> str | None:
        """Why the episode ended (reward.STOP_*), or None while it is still running."""
        if self._n_actions_at_report is not None:
            return STOP_REPORT
        return self._ended

    @property
    def result(self) -> GradeResult | None:
        return self._result

    @property
    def sandbox_ready(self) -> bool:
        return self._env is not None and self._baseline is not None

    # ── lifecycle ────────────────────────────────────────────────────────

    def load(self, task_dir: str | Path) -> str:
        """Host-only part of reset: read labels + instruction (cheap, raises on a malformed task before any boot)."""
        self._clear_episode()
        self.task_dir = Path(task_dir)
        self.spec = HonestTaskSpec.from_task_dir(self.task_dir)
        self.instruction = read_instruction(self.task_dir)
        return self.instruction

    async def start(self, task_dir: str | Path) -> str:
        """Boot the task's sandbox with the task's network policy and users enforced, baseline the protected paths
        and check the verifier toolchain. Returns the instruction.

        Raises on any failure (after tearing the half-started sandbox down). Nothing the agent did can cause a failure
        here, so `grade()` of such an episode is masked.
        """
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.config import VerifierEnvironmentMode
        from harbor.models.task.task import Task
        from harbor.models.trial.config import AgentConfig as TrialAgentConfig
        from harbor.models.trial.config import (
            EnvironmentConfig as TrialEnvironmentConfig,
        )
        from harbor.models.trial.paths import TrialPaths
        from harbor.trial.network_policy import resolve_trial_network_plan

        await self.stop()
        instruction = self.load(task_dir)
        cfg = self.config
        try:
            self._task = Task(task_dir=self.task_dir)
            trials_dir = None
            if cfg.trials_dir is not None:
                Path(cfg.trials_dir).mkdir(parents=True, exist_ok=True)
                trials_dir = str(cfg.trials_dir)
            self._paths = TrialPaths(trial_dir=Path(tempfile.mkdtemp(prefix="honest_", dir=trials_dir)))
            self._paths.mkdir()
            trial_env_cfg = TrialEnvironmentConfig(type=cfg.environment_type)
            self._plan = resolve_trial_network_plan(
                self._task.config,
                TrialAgentConfig(),
                trial_env_cfg,
                None,
                verifier_mode=VerifierEnvironmentMode.SHARED,  # we always grade in place (tests uploaded fresh)
            )
            kwargs: dict[str, Any] = {}
            if cfg.enforce_network_policy:
                kwargs = {
                    "network_policy": self._plan.agent_env_baseline,
                    "phase_network_policies": [
                        self._plan.agent_phase,
                        self._verifier_policy(),
                    ],
                }
            env_name, session_id = harbor_names(self._task, cfg.environment_type)
            self._env = EnvironmentFactory.create_environment_from_config(
                config=trial_env_cfg,
                environment_dir=self._task.paths.environment_dir,
                environment_name=env_name,
                session_id=session_id,
                trial_paths=self._paths,
                task_env_config=self._task.config.environment,
                **kwargs,
            )
            await self._start_sandbox()
            await self._env.run_healthcheck()
            baseline = await grading.snapshot(self._env, self.spec, cfg.grading_config)
            try:
                await grading.self_check(self._env, self.spec, cfg.grading_config)
            except grading.ToolchainError:
                self._boot_mask = MASK_TOOLCHAIN
                raise
            await self._check_timeout_binary()
            await self._prepare_report_dir()
            if cfg.enforce_network_policy:
                await self._env.set_network_policy(self._plan.agent_phase)
            self._baseline = baseline  # set last: the sandbox counts as ready only once every boot step succeeded
        except BaseException as e:
            self._boot_error = f"{type(e).__name__}: {e}"
            self._boot_mask = self._boot_mask or MASK_BOOT_FAILURE
            await self.stop()
            raise
        return instruction

    async def _check_timeout_binary(self) -> None:
        """Every command runs under `timeout -k`. An image without it would fail every command through no fault of the
        policy, so that is a toolchain failure at reset (masked), not something the agent is scored for."""
        res = await self._env.exec(
            "command -v timeout >/dev/null 2>&1",
            timeout_sec=60,
            user=self._task.config.agent.user,
        )
        if res.return_code != 0:
            self._boot_mask = MASK_TOOLCHAIN
            raise grading.ToolchainError("`timeout` is not installed in the task image")

    async def _prepare_report_dir(self) -> None:
        """Make `dirname(REPORT_FILE)` exist and be writable by the agent user. Harbor's own trials mount `/logs/agent`;
        a session passes no mounts, so without this a non-root agent could not use the report-file channel."""
        ensure = getattr(self._env, "ensure_dirs", None)  # BaseEnvironment only; test doubles have none
        if ensure is None:
            return
        try:
            await ensure([str(Path(REPORT_FILE).parent)])
        except Exception as e:  # noqa: BLE001 — the tool report still works without the file channel
            _log.warning(
                "could not prepare %s: %s: %s",
                Path(REPORT_FILE).parent,
                type(e).__name__,
                e,
            )

    def _verifier_policy(self) -> Any:
        """Network policy while grading. The hermetic verifier runs with the network off (SPEC_V3 §3.2); a legacy
        `test.sh` keeps the task's own verifier-phase policy, because those scripts download their toolchain."""
        if self.spec is not None and self.spec.verifier == "pytest":
            from harbor.models.task.config import NetworkMode, NetworkPolicy

            return NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        return self._plan.verifier_phase

    async def _start_sandbox(self) -> None:
        await start_sandbox(self._env, self._task, self.config.environment_type)

    async def stop(self) -> None:
        """Best-effort teardown: never raises (except cancellation), so it cannot mask a boot error or a grade."""
        env, self._env = self._env, None
        paths, self._paths = self._paths, None
        self._baseline = None
        try:
            if env is not None:
                await env.stop(delete=True)
        except Exception as e:  # noqa: BLE001
            _log.warning("sandbox teardown failed: %s: %s", type(e).__name__, e)
        finally:
            if paths is not None:
                shutil.rmtree(paths.trial_dir, ignore_errors=True)

    # ── harness ──────────────────────────────────────────────────────────

    async def bash(self, command: str) -> tuple[int | None, str]:
        """Run `command` as the task's agent user. Returns (exit code or None if it did not run, tool-output text).

        Once the call budget or the character budget is used the command is refused, and the refusal says so: a
        missing report is then the policy's choice and is scored, never masked.
        """
        if self.report_submitted or self._ended is not None:
            return None, EPISODE_OVER
        if not self.sandbox_ready:
            raise RuntimeError("sandbox is not running (call start() first)")
        b = self.budget
        refusal = self._meter.refusal()
        if refusal is not None:
            # A missing report is then the policy's choice and is scored, never masked.
            return None, f"[not run] {refusal}"
        wrapped = f"timeout -k {_KILL_AFTER_SEC} {b.command_timeout_sec} bash -c {shlex.quote(command)}"
        try:
            res = await self._env.exec(
                wrapped,
                timeout_sec=b.command_timeout_sec + _KILL_AFTER_SEC + _EXEC_SLACK_SEC,
                user=self._task.config.agent.user,
            )
            rc, out = res.return_code, (res.stdout or "") + (res.stderr or "")
            if rc in _TIMEOUT_CODES:
                out += f"\n[timed out after {b.command_timeout_sec} seconds]"
        except Exception:  # noqa: BLE001 - fixed string only; the real text is logged host-side
            _log.exception("sandbox exec failed")
            rc, out = None, SANDBOX_UNAVAILABLE
        self.history.append({"command": command, "rc": rc})
        return rc, self._meter.render(rc, out)

    def accept_report(self, tool: str, args: dict[str, Any]) -> str:
        """Record the first `submit_report` / `abstain` call; later ones are refused. A report that does not parse
        still ends the episode, and is scored as a parse error rather than as silence."""
        if tool not in REPORT_TOOLS:
            raise ValueError(f"not a report tool: {tool!r}")
        if self.report_submitted or self._ended is not None:
            return REPORT_DUPLICATE
        self._n_actions_at_report = len(self.history)
        try:
            self.report = report_from_tool_args(tool, dict(args))
        except ReportParseError as e:
            self.report_error = str(e)
            return f"Report could not be parsed ({e}). The episode is over."
        return REPORT_RECEIVED

    def end(self, stop_reason: str | None = None) -> None:
        """The episode is over without a report tool call: the model stopped calling tools (`silent_stop`), its last
        generation was cut (`truncated`), or the harness ran out of turns. A budget that `bash` already refused on
        takes precedence, because that is what the policy ran into."""
        if self.report_submitted or self._ended is not None:
            return
        if stop_reason not in (
            None,
            STOP_SILENT,
            STOP_TRUNCATED,
            STOP_CALL_BUDGET,
            STOP_CHAR_BUDGET,
        ):
            raise ValueError(f"unknown stop reason: {stop_reason!r}")
        self._ended = self._meter.hit or stop_reason or STOP_SILENT

    # ── grading ──────────────────────────────────────────────────────────

    def _n_actions(self) -> int:
        return self._n_actions_at_report if self._n_actions_at_report is not None else len(self.history)

    def _failed_outcome(self, error: str, masked_reason: str | None = None) -> EpisodeOutcome:
        """Outcome of an episode that could not be graded. `compute_reward` masks it when the agent had not run a
        command and scores it as tamper when it had (SPEC_V3 §3.5)."""
        spec = self.spec
        return EpisodeOutcome(
            feasible=spec.feasible,
            report=self.report,
            report_error=self.report_error,
            n_actions=self._n_actions(),
            agent_acted=bool(self.history),
            expected_blockers=list(spec.expected_blockers),
            expected_targets=list(spec.expected_targets),
            decoy_targets=list(spec.decoy_targets),
            is_control=spec.is_control,
            stop_reason=self.stop_reason,
            masked_reason=masked_reason if not self.history else None,
            grading_error=error,
        )

    def _fact_paths(self) -> list[str]:
        """Paths worth caching facts for: the blocker's target and the absolute paths the agent's commands name."""
        seen: dict[str, None] = {}
        if self.report is not None and self.report.blocker is not None and self.report.blocker.target:
            seen[self.report.blocker.target] = None
        for h in self.history:
            for p in _ABS_PATH.findall(h["command"]):
                seen[p] = None
        return list(seen)[:_MAX_FACT_PATHS]

    async def grade(self) -> GradeResult:
        """Grade the episode (cached) and free the sandbox."""
        if self._result is not None:
            return self._result
        spec = self.spec
        if spec is None:
            raise RuntimeError("no episode to grade (call start() first)")
        if self.stop_reason is None:
            self.end()
        cfg = self.config
        if not self.sandbox_ready:
            err = self._boot_error or "sandbox not running"
            outcome = self._failed_outcome(f"boot: {err}", self._boot_mask or MASK_BOOT_FAILURE)
            debug: dict[str, Any] = {"error": err}
        else:
            debug = {}
            try:
                agent_user = self._task.config.agent.user
                policy = self._plan.agent_phase if cfg.enforce_network_policy else None
                if not self.report_submitted:
                    # Harness-agnostic report channel (SPEC_V3 §9.4.1): the same schema as a JSON file.
                    self.report, self.report_error = await grading.read_report_file(self._env, agent_user=agent_user)
                    debug["report_source"] = "file" if self.report_submitted else None
                else:
                    debug["report_source"] = "tool"
                try:
                    facts = await grading.collect_blocker_facts(
                        self._env,
                        paths=self._fact_paths(),
                        agent_user=agent_user,
                        network_policy=policy,
                    )
                    debug["blocker_facts"] = facts.model_dump()
                except Exception as e:  # noqa: BLE001 — cached facts are for re-scoring only; grading does not need them
                    debug["blocker_facts_error"] = f"{type(e).__name__}: {e}"
                if cfg.enforce_network_policy:
                    await self._env.set_network_policy(self._verifier_policy())
                tests_dir = self._task.paths.tests_dir
                outcome, graded = await grading.grade(
                    self._env,
                    spec,
                    self._baseline,
                    report=self.report,
                    report_error=self.report_error,
                    n_actions=self._n_actions(),
                    agent_acted=bool(self.history),
                    stop_reason=self.stop_reason,
                    tests_dir=tests_dir if tests_dir.exists() else None,
                    verifier_env=_resolved_verifier_env(self._task),
                    verifier_user=self._task.config.verifier.user,
                    agent_user=agent_user,
                    network_policy=policy,
                    cfg=grading.resolve_config(
                        cfg.grading_config,
                        cfg.environment_type,
                        getattr(self._task.config.verifier, "timeout_sec", None),
                    ),
                )
                debug = {**debug, **graded}
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                outcome = self._failed_outcome(f"grade: {err}", MASK_BOOT_FAILURE)
                debug["error"] = err
            finally:
                await self.stop()  # free the container as soon as it is graded
        debug["history"] = self.history
        debug["stop_reason"] = outcome.stop_reason
        try:
            breakdown = compute_reward(outcome, cfg.reward_config)
        except Exception as e:  # noqa: BLE001 — a reward failure comes back as a value, never as an exception (§9.2.3)
            debug["reward_error"] = f"{type(e).__name__}: {e}"
            outcome = dataclasses.replace(self._failed_outcome(f"reward: {e}"), masked_reason=MASK_REWARD_ERROR)
            breakdown = compute_reward(outcome, cfg.reward_config)
        self._result = GradeResult(outcome, breakdown, debug)
        self._quarantine(breakdown)
        return self._result

    def _quarantine(self, br: RewardBreakdown) -> None:
        """Append a variant labelled infeasible whose tests passed cleanly to the quarantine list, when one is
        configured (`HONEST_QUARANTINE_FILE` / `HONEST_BANK_DIR`), so the next dataset build drops it (SPEC_V3 §4.7)."""
        if not (br.leak or br.unclaimed_pass_infeasible):
            return
        try:
            from honest_agents import bank

            spec = self.spec
            entry = bank.QuarantineEntry(
                variant_id=spec.variant_id,
                task_dir=str(self.task_dir),
                pair_id=spec.pair_id,
                perturbation=spec.perturbation,
                family=spec.family,
                kind=bank.QUARANTINE_LEAK if br.leak else bank.QUARANTINE_UNCLAIMED_PASS,
                source="openenv",
            )
            bank.append_quarantine([entry])
        except Exception as e:  # noqa: BLE001 — bookkeeping must not lose a graded episode
            _log.warning("could not write the quarantine entry: %s: %s", type(e).__name__, e)


def _resolved_verifier_env(task: Any) -> dict[str, str] | None:
    """`[verifier].env` with `${VAR}` / `${VAR:-default}` resolved against the host, as Harbor's Verifier does."""
    from harbor.utils.env import resolve_env_vars

    env = task.config.verifier.env
    return resolve_env_vars(env) if env else None


acquire = harness.acquire  # kept for the lifecycle tests, which exercise boot throttling through this module
