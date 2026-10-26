"""The contract every RTS operator module meets (docs/superpowers/plans/2026-10-26-rts-build.md)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from honest_agents.tasks.perturb.core import Variant


@dataclass(frozen=True)
class Applicability:
    ok: bool
    reason: str  # why not, or what was chosen
    target: dict[str, Any] = field(default_factory=dict)  # what the operator acts on (file, tool, service, ...)


@dataclass(frozen=True)
class RTSOperatorSpec:
    name: str
    tier: Literal["near", "mid", "far"]
    instruction_identical: bool
    expected_category: tuple[str, ...]  # BlockerCategory values
    applicable: Callable[[Path], Applicability]  # task dir of the seed -> applicability, static (no Docker)
    make_pair: Callable[..., tuple[Variant, Variant]]
    # make_pair(src: Path, out_root: Path, *, pair_id: str, target: dict, seed: int = 0) -> (twin, control)
