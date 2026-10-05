"""Honest-agents environment exposed through OpenEnv 0.7.0 (`openenv` on PyPI, not `openenv-core`).

`session` needs neither TRL nor OpenEnv; everything else needs the `openenv` extra, which conflicts with the training
extra (vllm's `mcp` pin), so names are imported lazily and this package imports in either environment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY = {
    "HonestAction": "honest_agents.openenv_env.models",
    "HonestObservation": "honest_agents.openenv_env.models",
    "HonestState": "honest_agents.openenv_env.models",
    "HonestEnvClient": "honest_agents.openenv_env.client",
    "HonestEnvironment": "honest_agents.openenv_env.server.environment",
    "HonestSession": "honest_agents.openenv_env.session",
    "SessionConfig": "honest_agents.openenv_env.session",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
