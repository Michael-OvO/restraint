"""`HonestHarborEnv`: TRL `environment_factory` environment over Harbor sandboxes.

Extends `trl.experimental.harbor.HarborEnv` with what honesty training needs and the stock class lacks:

* **Enforcement Harbor's `Trial` normally does** (and which the stock env silently skips): the task's resolved
  per-phase network policy and `[agent].user` / `[verifier].user`. Our infeasible-by-construction perturbations
  (revoked permissions, no network) are meaningless without these.
* **Non-blocking reset**: TRL calls `reset()` serially on the main thread. The instruction is on the host, so `reset()`
  returns immediately and the sandbox boots in the background on this env's event-loop thread; the first tool call
  waits for it. All of a batch's sandboxes therefore boot concurrently.
* **Harness**: `bash`, `submit_report`, `abstain` tools. The episode's claim is the first report submitted. The
  environment owns the budget (`harness.EpisodeBudget`): `bash` counts calls and returned characters, warns, shrinks its
  output and refuses once either is used up, so an episode ends with a report by choice; commands run under
  `timeout -k`; the stop reason is recorded.
* **Grading**: baseline manifests and the verifier toolchain self-check at reset (only a failure there masks the
  episode), `honest_agents.grading.grade` after the rollout, cached per episode. After the agent has acted, a grading
  failure is scored as tamper, never masked, and is counted as `honest/reward_error_rate` so a flaky
  grader is visible; `grade` retries a transport error once, and a teardown failure never changes a grade.

Prompt, tool-output format and report coercion live in the trl-free `honest_agents.harness`, shared with the OpenEnv
session and the eval agents.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")  # trl.experimental warns on every import otherwise

from trl.experimental.harbor import HarborEnv

from honest_agents import grading
from honest_agents.bank import BlockerFacts
from honest_agents.grading import (
    Baseline,
    GradingConfig,
    ToolchainError,
    grade,
    resolve_config,
    snapshot,
)
from honest_agents.harness import (
    BASH_DOC,
    EPISODE_OVER,
    REPORT_DUPLICATE,
    REPORT_RECEIVED,
    SANDBOX_UNAVAILABLE,
    BudgetMeter,
    EpisodeBudget,
    ReportTools,
    coerce_report,
    harbor_names,
    read_instruction,
    start_sandbox,
)
from honest_agents.mutation import first_mutation_index
from honest_agents.report import ReportParseError, SelfReport
from honest_agents.reward import (
    MASK_BOOT_FAILURE,
    MASK_REWARD_ERROR,
    MASK_TOOLCHAIN,
    STOP_REPORT,
    STOP_SILENT,
    STOP_TRUNCATED,
    EpisodeOutcome,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
)
from honest_agents.spec import VERIFIER_PYTEST, HonestTaskSpec

logger = logging.getLogger(__name__)

_DEL_TIMEOUT_SEC = 30
_STOP_TIMEOUT_SEC = 120  # `compose down` under 64 concurrent sandboxes is slow; a hang must not hold the batch
_KILL_AFTER_SEC = 5  # `timeout -k`: SIGKILL this long after SIGTERM
_EXEC_SLACK_SEC = 15  # the host-side exec timeout is only a backstop behind `timeout` inside the sandbox
_TIMEOUT_CODES = (124, 137)  # GNU timeout: 124 = timed out, 137 = had to SIGKILL
_ABS_PATH = re.compile(r"(?<![\w.~$])/[\w.+\-]+(?:/[\w.+\-]+)*")
_MAX_FACT_PATHS = 40
DEFAULT_PIDS_LIMIT = 1024  # fork-bomb guard; Harbor's Docker backend sets only cpus and memory
# Largest single file a sandbox process may write (RLIMIT_FSIZE, hard, so the agent cannot raise it without
# CAP_SYS_RESOURCE). Seen on a real node: one episode's script appended to a log in a loop and wrote 63 GB in minutes,
# filling the host disk and killing every run on it. ext4 has no per-container quota, so a single-file cap is what is
# available; the base images' tasks write well under this.
DEFAULT_MAX_FILE_BYTES = 1 << 30


class HonestHarborEnv(ReportTools, HarborEnv):
    PROMPT_SUFFIX = ""

    def __init__(
        self,
        environment_type: str = "docker",
        reward_config: RewardConfig | None = None,
        grading_config: GradingConfig | None = None,
        budget: EpisodeBudget | None = None,
        enforce_network_policy: bool = True,
        command_timeout_sec: int | None = None,
        pids_limit: int | None = DEFAULT_PIDS_LIMIT,
        max_file_bytes: int | None = DEFAULT_MAX_FILE_BYTES,
    ):
        super().__init__(environment_type=environment_type)
        self._reward_config = reward_config or RewardConfig()
        self._grading_config = grading_config or GradingConfig()
        self._budget = budget or EpisodeBudget()
        if command_timeout_sec is not None:  # back-compat spelling of `budget.command_timeout_sec`
            self._budget = EpisodeBudget(**{**self._budget.__dict__, "command_timeout_sec": command_timeout_sec})
        self._enforce_network = enforce_network_policy
        self._pids_limit = pids_limit
        self._max_file_bytes = max_file_bytes
        self._boot: concurrent.futures.Future | None = None
        self._spec: HonestTaskSpec | None = None
        self._task_dir: Path | None = None
        self._plan = None
        self._baseline: Baseline | None = None
        self._verifier_env: dict[str, str] | None = None
        self._use_timeout = True  # whether the image has a `timeout` that takes `-k` (probed at boot)
        self._clear_episode()

    def _clear_episode(self) -> None:
        self._history: list[dict[str, Any]] = []
        self._report: SelfReport | None = None
        self._report_error: str | None = None
        self._n_actions_at_report: int | None = None
        self._result: tuple[EpisodeOutcome, RewardBreakdown, dict] | None = None
        self._grade_future: concurrent.futures.Future | None = None
        self._meter = BudgetMeter(self._budget)
        self._stop_override: str | None = None
        self._boot_mask: str | None = None

    # ── TRL contract ─────────────────────────────────────────────────────

    def reset(self, task_dir: str | None = None, **row) -> str:
        if task_dir is None:
            raise ValueError("HonestHarborEnv.reset requires `task_dir` (a dataset column).")
        if self._boot is not None:  # previous episode's boot must finish before we tear it down
            with contextlib.suppress(Exception):
                self._boot.result()
        self._clear_episode()
        self._task_dir = Path(task_dir)
        self._spec = HonestTaskSpec.from_task_dir(self._task_dir)
        instruction = read_instruction(self._task_dir)
        self._boot = asyncio.run_coroutine_threadsafe(self._boot_episode(), self._loop)
        return instruction + self.PROMPT_SUFFIX

    @property
    def reward(self) -> float:
        # Never triggers grading: TRL evaluates properties when it introspects env members with `inspect.getmembers`.
        return self._result[1].total if self._result is not None else 0.0

    @property
    def _stop_reason(self) -> str | None:
        """Why the rollout ended: a report, the call or character budget, or the model simply stopped. A trainer that
        sees TRL cut the generation off may set it to "truncated". None before `reset()`."""
        if self._stop_override is not None:
            return self._stop_override
        if self._report is not None or self._report_error is not None:
            return STOP_REPORT
        if self._boot is None:
            return None
        return self._meter.hit or self._meter.exhausted or STOP_SILENT

    @_stop_reason.setter
    def _stop_reason(self, value: str | None) -> None:
        self._stop_override = value

    def _set_loop_exit(self, reason: str) -> None:
        """Trainer hook (`trainer.ENV_EXIT_HOOK`): only the trainer sees a generation cut off by the length limit.

        The other exit reasons are derived from the environment's own state, so only "truncated" is taken from it,
        and never over a report that was actually recorded.
        """
        if reason == STOP_TRUNCATED and self._report is None and self._report_error is None:
            self._stop_override = STOP_TRUNCATED

    # ── tools (every public method is exposed to the model) ──────────────

    # Tool methods carry no return annotation on purpose: `get_json_schema` would add a `function.return` key that TRL
    # renders into the training prompt but OpenAI-compatible servers (eval) drop, so train and eval prompts would differ.
    # `submit_report` and `abstain` come from `harness.ReportTools`; `bash` shares its docstring through `BASH_DOC`.

    def bash(self, command: str):
        if self._report is not None or self._report_error is not None:
            return EPISODE_OVER
        try:
            self._wait_boot()
        except Exception:
            logger.exception("sandbox boot failed")
            return SANDBOX_UNAVAILABLE
        refused = self._meter.refusal()
        if refused is not None:
            return refused
        timeout = self._budget.command_timeout_sec
        # `timeout -k` kills the command inside the container; Harbor's own timeout only abandons the host-side client and
        # leaves the process running. The host-side one stays as a backstop.
        wrapped = (
            f"timeout -k {_KILL_AFTER_SEC} {timeout} bash -c {shlex.quote(command)}" if self._use_timeout else command
        )
        try:
            res = self._run(
                self._env.exec(
                    wrapped,
                    timeout_sec=timeout + _KILL_AFTER_SEC + _EXEC_SLACK_SEC,
                    user=self._task.config.agent.user,
                )
            )
            rc, out = res.return_code, (res.stdout or "") + (res.stderr or "")
            if rc in _TIMEOUT_CODES:
                out += f"\n[command timed out after {timeout}s and was killed]"
        except Exception:
            logger.exception("sandbox exec failed")
            rc, out = None, SANDBOX_UNAVAILABLE
        self._history.append({"command": command, "rc": rc})
        return self._meter.render(rc, out)

    # ── internals ────────────────────────────────────────────────────────

    def _accept_report(self, raw: dict) -> str:
        if self._report is not None or self._report_error is not None:
            return REPORT_DUPLICATE
        self._n_actions_at_report = len(self._history)
        try:
            self._report = coerce_report(raw)
        except ReportParseError as e:
            self._report_error = str(e)
            return f"Report could not be parsed ({e}). The episode is over."
        return REPORT_RECEIVED

    def _set_external_report(self, report: SelfReport | None, error: str | None) -> None:
        """Inject a report that did not come through the tools (read from `harness.REPORT_FILE` by a third-party
        harness's evaluation). The first report still wins, so a tool report is never replaced."""
        if self._report is not None or self._report_error is not None:
            return
        if report is None and error is None:
            return
        self._n_actions_at_report = len(self._history)
        self._report, self._report_error = report, error

    def _wait_boot(self) -> None:
        if self._boot is None:
            raise RuntimeError("reset() was not called")
        self._boot.result()  # re-raises boot errors into the tool call

    def _verifier_policy(self):
        """Network policy while grading. The hermetic verifier runs with the network off; a legacy
        `test.sh` keeps the task's own verifier-phase policy, because those scripts download their toolchain."""
        if self._spec is not None and self._spec.verifier == VERIFIER_PYTEST:
            from harbor.models.task.config import NetworkMode, NetworkPolicy

            return NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
        return self._plan.verifier_phase

    def _write_pids_override(self) -> Path | None:
        """Compose overlay with a pids limit and a single-file size cap for the main service (Harbor's Docker backend sets
        neither), when either is configured."""
        if (not self._pids_limit and not self._max_file_bytes) or self._environment_type != "docker":
            return None
        lines = ["services:", "  main:"]
        if self._pids_limit:
            lines.append(f"    pids_limit: {int(self._pids_limit)}")
        if self._max_file_bytes:
            size = int(self._max_file_bytes)
            lines += ["    ulimits:", "      fsize:", f"        soft: {size}", f"        hard: {size}"]
        path = self._paths.trial_dir / "docker-compose.pids.yaml"
        path.write_text("\n".join(lines) + "\n")
        return path

    async def _boot_episode(self) -> None:
        from harbor.environments.factory import EnvironmentFactory
        from harbor.models.task.config import VerifierEnvironmentMode
        from harbor.models.task.task import Task
        from harbor.models.trial.config import AgentConfig as TrialAgentConfig
        from harbor.models.trial.config import (
            EnvironmentConfig as TrialEnvironmentConfig,
        )
        from harbor.models.trial.paths import TrialPaths
        from harbor.trial.network_policy import resolve_trial_network_plan

        await self._teardown()
        self._task = Task(task_dir=self._task_dir)
        # Resolved here, before the agent acts: a missing host variable is an infrastructure failure (masked), not
        # something to blame on the policy once it has run commands.
        self._verifier_env = _resolved_verifier_env(self._task)
        self._paths = TrialPaths(trial_dir=Path(tempfile.mkdtemp(prefix="honest_")))
        self._paths.mkdir()
        pids = self._write_pids_override()
        trial_env_cfg = TrialEnvironmentConfig(
            type=self._environment_type, extra_docker_compose=[pids] if pids is not None else []
        )
        self._plan = resolve_trial_network_plan(
            self._task.config,
            TrialAgentConfig(),
            trial_env_cfg,
            None,
            verifier_mode=VerifierEnvironmentMode.SHARED,  # we always grade in place (tests uploaded fresh)
        )
        kwargs: dict[str, Any] = {}
        if self._enforce_network:
            kwargs = {
                "network_policy": self._plan.agent_env_baseline,
                "phase_network_policies": [self._plan.agent_phase, self._verifier_policy()],
            }
        env_name, session_id = harbor_names(self._task, self._environment_type)
        self._env = EnvironmentFactory.create_environment_from_config(
            config=trial_env_cfg,
            environment_dir=self._task.paths.environment_dir,
            environment_name=env_name,
            session_id=session_id,
            trial_paths=self._paths,
            task_env_config=self._task.config.environment,
            **kwargs,
        )
        await start_sandbox(self._env, self._task, self._environment_type)
        await self._env.run_healthcheck()
        probe = await self._env.exec("timeout -k 1 5 true", timeout_sec=60, user=self._task.config.agent.user)
        self._use_timeout = probe.return_code == 0
        try:
            baseline = await snapshot(self._env, self._spec, self._grading_config)
            await grading.self_check(self._env, self._spec, self._grading_config)
        except ToolchainError:
            self._boot_mask = MASK_TOOLCHAIN  # nothing the agent did can cause this: masked
            raise
        if self._enforce_network:
            await self._env.set_network_policy(self._plan.agent_phase)
        self._baseline = baseline  # set last: the episode is gradeable only once every boot step succeeded

    def _grading_cfg_for_task(self) -> GradingConfig:
        declared = getattr(self._task.config.verifier, "timeout_sec", None)
        return resolve_config(self._grading_config, self._environment_type, declared)

    def _fact_paths(self) -> list[str]:
        """Paths the transcript names, whose existence and permissions the bank caches for blocker re-checks."""
        seen: dict[str, None] = {}
        report = self._report
        if report is not None and report.blocker is not None and report.blocker.target:
            seen[report.blocker.target] = None
        spec = self._spec
        if spec is not None:  # replay of a new blocker can name a labelled target the commands never mentioned
            for target in [*spec.expected_targets, *spec.decoy_targets]:
                if target.startswith("/"):
                    seen[target] = None
        for h in self._history:
            for m in _ABS_PATH.findall(h.get("command") or ""):
                seen[m] = None
        return list(seen)[:_MAX_FACT_PATHS]

    def _failed_outcome(self, error: str, *, acted: bool, masked_reason: str | None = None) -> EpisodeOutcome:
        """Outcome for a grading failure: a tamper verdict if the agent had acted, else a mask."""
        spec = self._spec
        return EpisodeOutcome(
            feasible=spec.feasible,
            report=self._report,
            report_error=self._report_error,
            n_actions=self._n_actions_at_report if self._n_actions_at_report is not None else len(self._history),
            agent_acted=acted,
            expected_blockers=list(spec.expected_blockers),
            expected_targets=list(spec.expected_targets),
            decoy_targets=list(spec.decoy_targets),
            is_control=spec.is_control,
            tamper=acted,
            tamper_details=[f"grading_error:{error[:200]}"] if acted else [],
            stop_reason=self._stop_reason,
            masked_reason=None if acted else (masked_reason or MASK_BOOT_FAILURE),
            n_blockers_submitted=self._report.n_blockers_submitted if self._report is not None else 0,
            grading_error=error,
        )

    async def _grade_episode(self) -> tuple[EpisodeOutcome, RewardBreakdown, dict]:
        outcome, debug = await self._grade_outcome()
        # Metrics and best-of-N tiebreaks only; the reward never reads it.
        outcome.first_mutation_idx = first_mutation_index([str(h.get("command") or "") for h in self._history])
        debug["history"] = self._history
        debug["stop_reason"] = outcome.stop_reason
        # The contract promises these keys on every episode, graded or not.
        debug.setdefault("test_log", "")
        debug.setdefault("blocker_facts", BlockerFacts().model_dump())
        try:
            breakdown = compute_reward(outcome, self._reward_config)
        except Exception as e:  # noqa: BLE001 — a reward failure comes back as a value, never as an exception
            debug["reward_error"] = f"{type(e).__name__}: {e}"
            outcome = self._failed_outcome(f"reward: {e}", acted=False, masked_reason=MASK_REWARD_ERROR)
            breakdown = compute_reward(outcome, self._reward_config)
        return outcome, breakdown, debug

    async def _grade_outcome(self) -> tuple[EpisodeOutcome, dict[str, Any]]:
        acted = bool(self._history)
        try:
            await asyncio.wrap_future(self._boot)
        except Exception as e:  # noqa: BLE001 — sandbox never came up: not the policy's fault, mask from loss
            await self._teardown()  # start() may have succeeded before healthcheck/snapshot failed
            return (
                self._failed_outcome(f"boot: {e}", acted=False, masked_reason=self._boot_mask),
                {"error": str(e)},
            )
        debug: dict[str, Any] = {}
        agent_user = self._task.config.agent.user
        try:
            agent_policy = self._plan.agent_phase if self._enforce_network else None
            try:  # cached facts serve the report-only pilot; grading does not need them
                facts = await grading.collect_blocker_facts(
                    self._env, paths=self._fact_paths(), agent_user=agent_user, network_policy=agent_policy
                )
                debug["blocker_facts"] = facts.model_dump()
            except Exception as e:  # noqa: BLE001
                debug["blocker_facts_error"] = f"{type(e).__name__}: {e}"
            if self._enforce_network:
                await self._env.set_network_policy(self._verifier_policy())
            n_actions = self._n_actions_at_report if self._n_actions_at_report is not None else len(self._history)
            tests_dir = self._task.paths.tests_dir
            outcome, graded = await grade(
                self._env,
                self._spec,
                self._baseline,
                report=self._report,
                report_error=self._report_error,
                n_actions=n_actions,
                agent_acted=acted,
                stop_reason=self._stop_reason,
                tests_dir=tests_dir if tests_dir.exists() else None,
                verifier_env=self._verifier_env,
                verifier_user=self._task.config.verifier.user,
                agent_user=agent_user,
                network_policy=agent_policy,
                cfg=self._grading_cfg_for_task(),
            )
            debug.update(graded)
        except Exception as e:  # noqa: BLE001 — `grade` itself never raises; this covers the network switch and the like
            err = f"{type(e).__name__}: {e}"
            outcome = self._failed_outcome(f"grade: {err}", acted=acted)
            debug["error"] = err
        finally:
            await self._teardown()  # free the container as soon as it is graded
        return outcome, debug

    async def _teardown(self) -> None:
        """Stop the sandbox; a failure is logged and never reaches the reward.

        A `compose down` error or timeout after a good grade would otherwise replace the computed outcome and score an
        honest verified success as tamper (-1.5). The worst a failed teardown can cost is a leaked container, which
        the next boot's own cleanup and the VM's reaper deal with."""
        try:
            await asyncio.wait_for(self._stop(), _STOP_TIMEOUT_SEC)
        except Exception as e:  # noqa: BLE001 - TimeoutError included
            logger.warning("sandbox teardown failed (%s: %s); the grade is kept", type(e).__name__, e)

    def _start_grading(self) -> concurrent.futures.Future:
        """Kick off grading on this env's loop without blocking (lets a reward func grade a whole batch concurrently)."""
        if self._grade_future is None:
            self._grade_future = asyncio.run_coroutine_threadsafe(self._grade_episode(), self._loop)
        return self._grade_future

    def _graded(self) -> tuple[EpisodeOutcome, RewardBreakdown, dict]:
        if self._result is None:
            self._result = self._start_grading().result()
        return self._result

    def __del__(self):
        # TRL's HarborEnv.__del__ waits on teardown with no timeout; at interpreter exit the daemon loop thread is
        # already frozen, so that wait hangs the process (trainer / pytest exit).
        loop = getattr(self, "_loop", None)
        if loop is None or loop.is_closed():
            return
        if getattr(self, "_env", None) is not None and loop.is_running():
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._teardown(), loop).result(timeout=_DEL_TIMEOUT_SEC)
        with contextlib.suppress(Exception):
            loop.call_soon_threadsafe(loop.stop)


HonestHarborEnv.bash.__doc__ = BASH_DOC


def _resolved_verifier_env(task: Any) -> dict[str, str] | None:
    """`[verifier].env` with `${VAR}` / `${VAR:-default}` resolved against the host, as Harbor's Verifier does.

    A missing host variable raises ValueError; it is resolved at boot, so it masks the episode.
    """
    from harbor.utils.env import resolve_env_vars

    env = task.config.verifier.env
    return resolve_env_vars(env) if env else None
