"""Harbor's Docker environment with every main-container docker exec under `harness.SANDBOX_UMASK` (DEF-16, DEF-19).

`harbor run -e docker` builds a stock `DockerEnvironment`, whose `docker exec` takes the daemon's umask (0000 under the
Mac's OrbStack), whatever the agent. Our eval scripts pass this class instead:

    harbor run -p TASKS -e honest_agents.eval.harbor_env:UmaskDockerEnvironment -a terminus-2 ...

Harbor builds every environment of a trial from that one config (the agent's and a separate verifier's, `trial.py`).
The pin sits in `_run_docker_compose_command`, which every local docker exec passes (`harness.umask_compose_command`):
the agent's commands under any harness (terminus-2, mini-swe-agent), the verifier, Harbor's start-time execs and the
raw execs of its upload fallback all run under 022. On the nodes it is a no-op (they already run under 0022).
"""

from __future__ import annotations

from typing import Any

from harbor.environments.docker.docker import DockerEnvironment

from honest_agents.harness import umask_compose_command


class UmaskDockerEnvironment(DockerEnvironment):
    _umask_pinned = True  # `harness.pin_docker_umask` leaves it alone: no second wrapper

    async def _run_docker_compose_command(self, command: list[str], *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "_is_windows_container", False):
            command = umask_compose_command(command)
        return await super()._run_docker_compose_command(command, *args, **kwargs)
