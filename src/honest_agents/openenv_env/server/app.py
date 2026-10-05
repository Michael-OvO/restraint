"""FastAPI app serving `HonestEnvironment` (OpenEnv HTTP + WebSocket protocol).

Use the WebSocket endpoint (`/ws`, i.e. `HonestEnvClient`): episodes are stateful. The stateless HTTP `/reset` and
`/step` routes OpenEnv registers in simulation mode build a fresh environment per request, so `/reset` boots a sandbox
only to tear it down and `/step` always fails.

Env vars: HONEST_TASKS_DIR, HONEST_MAX_CONCURRENT_ENVS (64), HONEST_MAX_CONCURRENT_BOOTS (16), HONEST_SESSION_TIMEOUT_S
(1800; idle sessions are reaped so leaked sandboxes get stopped), plus those read by `session_config_from_env`.

    uvicorn honest_agents.openenv_env.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import ConcurrencyConfig

from honest_agents.openenv_env.models import (
    HonestAction,
    HonestObservation,
    HonestState,
)
from honest_agents.openenv_env.server.environment import HonestEnvironment
from honest_agents.openenv_env.session import set_max_concurrent_boots


def build_app(
    env_factory: Callable[[], HonestEnvironment] = HonestEnvironment,
    *,
    max_concurrent_envs: int | None = None,
    session_timeout_s: float | None = None,
) -> FastAPI:
    if max_concurrent_envs is None:
        max_concurrent_envs = int(os.getenv("HONEST_MAX_CONCURRENT_ENVS", "64"))
    if session_timeout_s is None:
        session_timeout_s = float(os.getenv("HONEST_SESSION_TIMEOUT_S", "1800"))
    return create_app(
        env_factory,
        HonestAction,
        HonestObservation,
        env_name="honest_agents",
        concurrency_config=ConcurrencyConfig(
            max_concurrent_envs=max_concurrent_envs,
            session_timeout=session_timeout_s or None,  # 0 = never reap
        ),
        state_cls=HonestState,
    )


if "HONEST_MAX_CONCURRENT_BOOTS" in os.environ:
    set_max_concurrent_boots(int(os.environ["HONEST_MAX_CONCURRENT_BOOTS"]))

app = build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(args.host, args.port)
