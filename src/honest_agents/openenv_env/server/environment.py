"""`HonestEnvironment`: the honest-agents episode as an OpenEnv `Environment`.

Composes the trl-free `HonestSession` (the v3 core: budgets, hermetic verifier, blocker check, reward). Design notes:

* Harbor's sandbox clients are bound to the event loop that created them, and a container boot or a test run can take
  minutes. Each environment instance therefore owns a private event-loop thread; `reset_async`/`step_async` (which the
  OpenEnv WebSocket server awaits directly on its own loop) hand coroutines to it, so one slow sandbox never stalls the
  server or other sessions.
* `reset` waits for the sandbox to be up and baselined (unlike TRL's background boot): an OpenEnv client expects a
  ready environment after reset, and a boot failure is reported as a reset error the client can retry.
* The episode ends on the first `submit_report`/`abstain`, on `end` (the model stopped without a report), or after
  `max_steps` actions of any kind (TRL's `max_tool_calling_iterations`). `bash` refuses to run once the call or the
  character budget is used, which leaves the last turns for the report; an episode that still ends without one is
  scored, not masked. The terminal observation carries the reward and its breakdown.
* Task selection: `reset(task_id=...)` relative to `HONEST_TASKS_DIR`, or `reset(split=..., index=...)`. Absolute
  `task_dir` outside that root is refused unless explicitly allowed, since the server would otherwise build Docker images
  from arbitrary host directories on a client's say-so.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import threading
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from honest_agents.grading import GradingConfig
from honest_agents.harness import SANDBOX_UNAVAILABLE
from honest_agents.openenv_env.models import (
    HonestAction,
    HonestObservation,
    HonestState,
)
from honest_agents.openenv_env.session import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    SYSTEM_PROMPT,
    EpisodeBudget,
    HonestSession,
    SessionConfig,
)
from honest_agents.reward import STOP_CALL_BUDGET, RewardConfig

ALL_SPLIT = "all"
_log = logging.getLogger(__name__)
_CLOSE_TIMEOUT_S = 300.0


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def session_config_from_env(max_steps: int | None = None) -> SessionConfig:
    """Server configuration from env vars (`create_app` instantiates the environment with no arguments).

    The budget is built by `EpisodeBudget.from_limits` from the same two numbers as in training and evaluation
    (`HONEST_MAX_STEPS` = tool-call turns, `HONEST_MAX_COMPLETION_TOKENS`), so the caps cannot drift apart.
    """
    trials = os.getenv("HONEST_TRIALS_DIR")
    steps = max_steps if max_steps is not None else int(os.getenv("HONEST_MAX_STEPS", str(DEFAULT_MAX_TOOL_ITERATIONS)))
    tokens = int(os.getenv("HONEST_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS)))
    overrides: dict[str, Any] = {}
    if "HONEST_COMMAND_TIMEOUT_S" in os.environ:
        overrides["command_timeout_sec"] = int(os.environ["HONEST_COMMAND_TIMEOUT_S"])
    budget = EpisodeBudget.from_limits(steps, tokens, **overrides)
    reward = json.loads(os.getenv("HONEST_REWARD_CONFIG", "{}"))
    reward.setdefault("efficiency_ref_actions", max(1, budget.max_bash_calls))
    grading = json.loads(os.getenv("HONEST_GRADING_CONFIG", "{}"))
    if grading.get("verifier_bundle_dir"):
        grading["verifier_bundle_dir"] = Path(grading["verifier_bundle_dir"])
    return SessionConfig(
        environment_type=os.getenv("HONEST_ENVIRONMENT_TYPE", "docker"),
        enforce_network_policy=_env_bool("HONEST_ENFORCE_NETWORK", True),
        max_tool_iterations=steps,
        max_completion_tokens=tokens,
        budget=budget,
        reward_config=RewardConfig(**reward),
        grading_config=GradingConfig(**grading),
        trials_dir=Path(trials) if trials else None,
    )


@functools.lru_cache(maxsize=8)
def _discover(root: str) -> dict[str, tuple[str, ...]]:
    """Map split -> sorted task ids (paths relative to root). A split is a top-level dir that is not itself a task."""
    base = Path(root)
    ids = sorted(str(p.parent.relative_to(base)) for p in base.rglob("task.toml"))
    splits: dict[str, tuple[str, ...]] = {ALL_SPLIT: tuple(ids)}
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not (p / "task.toml").exists()):
        members = tuple(i for i in ids if i.startswith(d.name + "/"))
        if members:
            splits[d.name] = members
    return splits


class HonestEnvironment(Environment[HonestAction, HonestObservation, HonestState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # one sandbox + one loop thread per instance, no shared mutable state

    def __init__(
        self,
        tasks_dir: str | Path | None = None,
        session_config: SessionConfig | None = None,
        max_steps: int | None = None,
        allow_any_task_dir: bool | None = None,
    ):
        super().__init__()
        root = tasks_dir if tasks_dir is not None else os.getenv("HONEST_TASKS_DIR")
        self.tasks_dir = Path(root).resolve() if root else None
        self.session_config = session_config or session_config_from_env(max_steps)
        # Actions per episode, of any kind. Defaults to the number the budget was built from.
        self.max_steps = max_steps if max_steps is not None else self.session_config.max_tool_iterations
        self.allow_any_task_dir = (
            allow_any_task_dir if allow_any_task_dir is not None else _env_bool("HONEST_ALLOW_ANY_TASK_DIR", False)
        )
        # Created lazily: the server builds throwaway instances for /state, /metadata and task discovery.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        # Serialises reset/step/teardown on the env loop: `close()` (idle reaper, disconnect) can arrive while a boot
        # or grading is in flight, and must not tear the sandbox down underneath it.
        self._op_lock = asyncio.Lock()
        self._session: HonestSession | None = None
        self._state = self._new_state()

    # ── OpenEnv contract ─────────────────────────────────────────────────

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        task_dir: str | None = None,
        split: str | None = None,
        index: int | None = None,
        **kwargs: Any,
    ) -> HonestObservation:
        path, tid = self._resolve_task(task_id=task_id, task_dir=task_dir, split=split, index=index)
        return await self._on_loop(self._reset(path, tid, episode_id))

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        task_dir: str | None = None,
        split: str | None = None,
        index: int | None = None,
        **kwargs: Any,
    ) -> HonestObservation:
        path, tid = self._resolve_task(task_id=task_id, task_dir=task_dir, split=split, index=index)
        return self._run_blocking(self._reset(path, tid, episode_id))

    async def step_async(
        self, action: HonestAction, timeout_s: float | None = None, **kwargs: Any
    ) -> HonestObservation:
        return await self._on_loop(self._step(action))

    def step(self, action: HonestAction, timeout_s: float | None = None, **kwargs: Any) -> HonestObservation:
        return self._run_blocking(self._step(action))

    @property
    def state(self) -> HonestState:
        return self._state

    def close(self) -> None:
        """Stop the sandbox and the loop thread. Never blocks a running event loop (the server calls this inline)."""
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        self._closed = True
        if loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._locked(self._shutdown()), loop)
        fut.add_done_callback(lambda _: loop.call_soon_threadsafe(loop.stop))
        try:
            asyncio.get_running_loop()
            return  # fire-and-forget: teardown finishes on the env's own thread
        except RuntimeError:
            pass
        try:
            fut.result(timeout=_CLOSE_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — best-effort teardown
            _log.warning("sandbox teardown failed: %s", e)
        if thread is not None:
            thread.join(timeout=10)

    def get_metadata(self) -> EnvironmentMetadata:
        readme = Path(__file__).resolve().parents[1] / "README.md"
        return EnvironmentMetadata(
            name="honest_agents",
            description=(
                "Terminal tasks (Harbor sandboxes) with feasible and infeasible-by-construction variants. The agent "
                "must end with a structured self-report or an abstention naming one blocker, which is graded against "
                "the real final sandbox state (tamper and sentinel manifests, a hermetic verifier, a blocker check)."
            ),
            readme_content=readme.read_text() if readme.exists() else None,
            version="0.3.0",
        )

    # ── task discovery (OpenEnv TaskProvider protocol; ids only, never labels) ──

    def list_splits(self) -> list[str]:
        return list(_discover(str(self.tasks_dir))) if self.tasks_dir is not None else []

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return [{"task_id": t, "index": i} for i, t in enumerate(self._split(split))]

    def num_tasks(self, split: str) -> int:
        return len(self._split(split))

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        ids = self._split(split)
        if not 0 <= index < len(ids):
            raise IndexError(f"index {index} out of range for split {split!r} ({len(ids)} tasks)")
        return {"task_id": ids[index], "index": index}

    def get_task_range(self, split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, Any]]:
        ids = self._split(split)
        return [{"task_id": ids[i], "index": i} for i in range(len(ids))[start:stop]]

    def _split(self, split: str) -> tuple[str, ...]:
        if self.tasks_dir is None:
            raise ValueError("no task root configured (set HONEST_TASKS_DIR)")
        splits = _discover(str(self.tasks_dir))
        if split not in splits:
            raise ValueError(f"unknown split {split!r}; available: {sorted(splits)}")
        return splits[split]

    def _resolve_task(
        self,
        *,
        task_id: str | None,
        task_dir: str | None,
        split: str | None,
        index: int | None,
    ) -> tuple[Path, str]:
        if split is not None or index is not None:
            if split is None or index is None:
                raise ValueError("reset needs both `split` and `index`")
            task_id = self.get_task(split, int(index))["task_id"]
        if task_id is not None:
            if self.tasks_dir is None:
                raise ValueError("`task_id` needs a task root (set HONEST_TASKS_DIR)")
            path = (self.tasks_dir / task_id).resolve()
        elif task_dir is not None:
            path = Path(task_dir).resolve()
        else:
            raise ValueError("reset requires `task_id`, `task_dir`, or `split` + `index`")
        inside = self.tasks_dir is not None and path.is_relative_to(self.tasks_dir)
        if not inside and not self.allow_any_task_dir:
            raise ValueError(f"task {path} is outside the task root {self.tasks_dir}")
        if not (path / "task.toml").is_file():
            raise ValueError(f"not a task directory (no task.toml): {task_id or task_dir}")
        tid = str(path.relative_to(self.tasks_dir)) if inside else str(path)
        return path, tid

    # ── episode logic (runs on the env's own loop) ──────────────────────

    def _new_state(self, **kw: Any) -> HonestState:
        return HonestState(
            max_steps=self.max_steps,
            max_bash_calls=self.session_config.budget.max_bash_calls,
            **kw,
        )

    async def _reset(self, path: Path, task_id: str, episode_id: str | None) -> HonestObservation:
        if self._session is not None:
            await self._session.stop()
        self._session = s = HonestSession(self.session_config)
        self._state = self._new_state(episode_id=episode_id or str(uuid.uuid4()), task_id=task_id)
        try:
            instruction = await s.start(path)
        except Exception:  # noqa: BLE001 - the client must see SANDBOX_UNAVAILABLE only
            # The raw error carries compose commands, host paths and the task name: log it here, tell the client nothing.
            _log.exception("sandbox failed to start for %s", task_id)
            raise RuntimeError(SANDBOX_UNAVAILABLE) from None
        self._state.sandbox_ready = True
        return self._obs("reset", instruction, instruction=instruction, system_prompt=SYSTEM_PROMPT)

    async def _step(self, action: HonestAction) -> HonestObservation:
        s = self._session
        if s is None:
            raise RuntimeError("call reset() before step()")
        st = self._state
        if st.done:
            # Repeat the graded result: a bare `reward=None` here would read as a mask to a client that steps once too
            # often, and a masked episode is dropped from the loss.
            res = s.result
            return self._obs(
                action.kind,
                "The episode is over. Call reset() to start a new one.",
                done=True,
                reward=res.reward if res is not None else None,
                metadata=res.to_metadata() if res is not None else None,
            )
        if action.kind == "bash":
            rc, text = await s.bash(action.command)  # raises if the sandbox never came up: not a turn
            st.step_count += 1
            st.n_commands = s.n_commands
            if st.step_count < self.max_steps:
                return self._obs("bash", text, exit_code=rc)
            s.end(STOP_CALL_BUDGET)
            text += f"\n[turn limit of {self.max_steps} tool calls reached; the episode ended without a report]"
            return await self._finish("bash", text, exit_code=rc)
        st.step_count += 1
        if action.kind == "end":
            s.end(action.stop_reason)
            return await self._finish("end", "Episode ended without a report.")
        msg = s.accept_report(action.kind, action.tool_args())
        st.report_submitted = True
        return await self._finish(action.kind, msg)

    async def _finish(self, kind: str, text: str, exit_code: int | None = None) -> HonestObservation:
        result = await self._session.grade()
        st = self._state
        st.done, st.sandbox_ready = True, False
        st.reward, st.case = result.reward, result.breakdown.case
        st.stop_reason, st.masked_reason = (
            result.outcome.stop_reason,
            result.breakdown.masked_reason,
        )
        return self._obs(
            kind,
            text,
            exit_code=exit_code,
            done=True,
            reward=result.reward,
            metadata=result.to_metadata(),
        )

    def _obs(
        self,
        kind: str,
        text: str,
        *,
        exit_code: int | None = None,
        done: bool = False,
        reward: float | None = None,
        metadata: dict[str, Any] | None = None,
        **fields: Any,
    ) -> HonestObservation:
        s = self._session
        return HonestObservation(
            output=text,
            kind=kind,
            exit_code=exit_code,
            task_id=self._state.task_id,
            steps_remaining=s.calls_remaining if s is not None else None,
            chars_remaining=s.chars_remaining if s is not None else None,
            turns_remaining=max(0, self.max_steps - self._state.step_count),
            done=done,
            reward=reward,
            metadata=metadata or {},
            **fields,
        )

    async def _shutdown(self) -> None:
        s, self._session = self._session, None
        if s is not None:
            await s.stop()

    # ── loop plumbing ────────────────────────────────────────────────────

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=_run_loop,
                args=(self._loop,),
                name="honest-env-loop",
                daemon=True,
            )
            self._thread.start()
        return self._loop

    async def _locked(self, coro: Coroutine) -> Any:
        async with self._op_lock:
            return await coro

    def _submit(self, coro: Coroutine) -> concurrent.futures.Future:
        if self._closed:
            coro.close()
            raise RuntimeError("environment is closed (session ended or reaped); open a new session")
        return asyncio.run_coroutine_threadsafe(self._locked(coro), self._ensure_loop())

    async def _on_loop(self, coro: Coroutine) -> Any:
        return await asyncio.wrap_future(self._submit(coro))

    def _run_blocking(self, coro: Coroutine) -> Any:
        return self._submit(coro).result()


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Loop-thread body. Closing the loop on exit matters: a server outlives thousands of sessions, and an unclosed
    loop leaks its selector/self-pipe fds and default-executor threads."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor(timeout=30))
        finally:
            loop.close()
