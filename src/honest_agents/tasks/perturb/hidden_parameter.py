"""`hide_parameter`: an LLM-assisted operator that is never built by default."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

from honest_agents.tasks.analysis import (
    TaskFacts,
)
from honest_agents.tasks.perturb.core import Variant, _rng
from honest_agents.tasks.perturb.emit import _emit
from honest_agents.tasks.perturb.high_risk import HiddenParameter


class ParameterHider(Protocol):
    """LLM-backed rewriter for `hide_parameter`. Not shipped: plug in a client-backed implementation offline.

    Contract: remove one value the task cannot be completed without (a port, a filename, a threshold) such that it is
    not recoverable from the environment, or return None. Variants must then be checked by an LLM judge plus a
    multi-sample check (a strong model given the rewritten task should ask for / abstain on the missing value); the
    oracle cannot validate them because `solve.sh` hard-codes the value.
    """

    def hide(self, instruction: str, *, rng: random.Random) -> HiddenParameter | None: ...


def hide_parameter(
    src: Path,
    out_root: Path,
    *,
    pair_id: str,
    seed: int = 0,
    split: str = "train",
    hider: ParameterHider | None = None,
) -> Variant | None:
    """Optional operator: needs a `ParameterHider` (LLM client); without one it is never applicable. No control."""
    if hider is None:
        return None
    facts = TaskFacts.load(src)
    hidden = hider.hide(facts.instruction, rng=_rng(pair_id, "hide_parameter", seed))
    if hidden is None or hidden.instruction.strip() == facts.instruction.strip():
        return None
    return _emit(
        src,
        out_root,
        pair_id=pair_id,
        op="hide_parameter",
        detail={"hidden_value": hidden.hidden_value, "rationale": hidden.rationale, "judge_verified": False},
        spec_overrides={"expected_targets": [hidden.hidden_value] if hidden.hidden_value else []},
        edit=lambda d: (d / "instruction.md").write_text(hidden.instruction),
    )
