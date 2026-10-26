"""The contract every RTS operator module meets: a static applicability test and a pair builder, declared in one `SPEC`.

`applicable(seed_task_dir)` reads the seed (no Docker) and says whether the operator can act on it and on what
(`Applicability.target`: the file, tool, service, ... it chose). `make_pair(src, out_root, *, pair_id, target, seed=0)`
writes the impossible twin and its solvable near-miss control as `tasks.perturb.core.Variant`s whose `HonestTaskSpec`s
the grader reads (feasibility, `perturbation` = the operator name, `is_control`, expected blockers and targets), as the
existing operator/control pairs do. The build CLI (`eval.rts.build`) and the scorer read tier, `instruction_identical`
and `expected_category` from here and hard-code no operator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from honest_agents.report import EXTERNAL_CATEGORIES
from honest_agents.tasks.perturb.core import Variant

Tier = Literal["near", "mid", "far"]
TIERS: tuple[Tier, ...] = ("near", "mid", "far")
ENV_TEXT_CHANNELS = ("readme", "files", "lock")  # `tasks.analysis.envtext.environment_text`


@dataclass(frozen=True)
class Applicability:
    ok: bool
    reason: str  # why not, or what was chosen
    target: dict[str, Any] = field(default_factory=dict)  # what the operator acts on (file, tool, service, ...)


@dataclass(frozen=True)
class RTSOperatorSpec:
    name: str
    tier: Tier
    instruction_identical: bool
    expected_category: tuple[str, ...]  # BlockerCategory values
    applicable: Callable[[Path], Applicability]  # task dir of the seed -> applicability, static (no Docker)
    make_pair: Callable[..., tuple[Variant, Variant]]
    # make_pair(src: Path, out_root: Path, *, pair_id: str, target: dict, seed: int = 0) -> (twin, control)
    # Env-text channels whose words ARE the mechanism (the planted secret, the prod config): the freeze reports their
    # cue AUC and does not enforce it, as `envtext.CARRIER_CHANNELS` does for the training families.
    carrier_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"operator {self.name}: tier {self.tier!r} not in {TIERS}")
        allowed = {c.value for c in EXTERNAL_CATEGORIES}
        if not self.expected_category or any(c not in allowed for c in self.expected_category):
            raise ValueError(
                f"operator {self.name}: expected_category {self.expected_category} must be external blocker "
                f"categories {sorted(allowed)}"
            )
        if bad := [c for c in self.carrier_channels if c not in ENV_TEXT_CHANNELS]:
            raise ValueError(f"operator {self.name}: carrier_channels {bad} not in {ENV_TEXT_CHANNELS}")
