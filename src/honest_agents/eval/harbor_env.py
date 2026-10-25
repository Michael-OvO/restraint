"""Harbor's Docker environment with every main-container command under `harness.SANDBOX_UMASK` (DEF-16).

`harbor run -e docker` builds a stock `DockerEnvironment`, whose `docker exec` takes the daemon's umask (0000 under the
Mac's OrbStack), whatever the agent. Our eval scripts pass this class instead:

    harbor run -p TASKS -e honest_agents.eval.harbor_env:UmaskDockerEnvironment -a terminus-2 ...

Harbor builds every environment of a trial from that one config (the agent's and a separate verifier's, `trial.py`), and
its own main-container execs (the readiness probe, `service_exec` on the main service) go through `exec`, so stock
harnesses (terminus-2, mini-swe-agent), the verifier and Harbor's start-time execs all run under 022. On the nodes it
is a no-op (they already run under 0022).
"""

from __future__ import annotations

from typing import Any

from harbor.environments.docker.docker import DockerEnvironment

from honest_agents.harness import SANDBOX_UMASK


class UmaskDockerEnvironment(DockerEnvironment):
    _umask_pinned = True  # `harness.pin_exec_umask` leaves it alone: no second prefix

    async def exec(self, command: str, *args: Any, **kwargs: Any) -> Any:
        if not getattr(self, "_is_windows_container", False):
            command = f"umask {SANDBOX_UMASK}; {command}"
        return await super().exec(command, *args, **kwargs)
