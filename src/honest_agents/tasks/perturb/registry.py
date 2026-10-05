"""Operator registries and `generate_variants`, which also drops any variant that would leak its label."""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from honest_agents.tasks.analysis import (
    label_leaks,
)
from honest_agents.tasks.perturb.conflicting_constraint import conflicting_constraint, conflicting_constraint_control
from honest_agents.tasks.perturb.core import ControlOperator, Operator, Variant
from honest_agents.tasks.perturb.emit import make_feasible
from honest_agents.tasks.perturb.high_risk import high_risk, high_risk_control
from honest_agents.tasks.perturb.no_network import no_network, no_network_control
from honest_agents.tasks.perturb.remove_input import remove_input, remove_input_control
from honest_agents.tasks.perturb.remove_tool import remove_tool, remove_tool_control
from honest_agents.tasks.perturb.revoke_permission import revoke_permission, revoke_permission_control

log = logging.getLogger(__name__)


OPERATORS: dict[str, Operator] = {
    "remove_tool": remove_tool,
    "remove_input": remove_input,
    "revoke_permission": revoke_permission,
    "no_network": no_network,
    "conflicting_constraint": conflicting_constraint,
    "high_risk": high_risk,
}
CONTROL_OPERATORS: dict[str, ControlOperator] = {
    "remove_tool": remove_tool_control,
    "remove_input": remove_input_control,
    "revoke_permission": revoke_permission_control,
    "no_network": no_network_control,
    "conflicting_constraint": conflicting_constraint_control,
    "high_risk": high_risk_control,
}

# Variants `generate_variants` dropped because an emitted file carried a label string, per operator; `tasks.build`
# writes the delta of a build to manifest.json.
LEAK_DROPPED: Counter[str] = Counter()


def generate_variants(
    src: Path,
    out_root: Path,
    *,
    pair_id: str,
    operators: list[str] | None = None,
    seed: int = 0,
    include_feasible: bool = True,
    split: str = "train",
    controls: bool = True,
    require_controls: bool = True,
) -> list[Variant]:
    """Feasible copy, then for every applicable operator its infeasible variant followed by its near-miss control.

    An operator that crashes is logged and skipped. With `require_controls` (the default) an infeasible variant whose
    control cannot be built is dropped: a cue with no harmless counterpart is exactly what teaches "cue -> abstain".
    """
    out: list[Variant] = []
    if include_feasible:
        try:
            out.append(make_feasible(src, out_root, pair_id=pair_id, seed=seed, split=split))
        except Exception as e:  # noqa: BLE001 — a seed Harbor cannot load yields no pair at all
            log.warning("seed %s is not a loadable Harbor task, skipped: %s", pair_id, e)
            return []
    for name in operators if operators is not None else list(OPERATORS):
        try:
            v = OPERATORS[name](src, out_root, pair_id=pair_id, seed=seed, split=split)
        except Exception as e:  # noqa: BLE001 — one malformed source task must not stop a 7k-task build
            log.warning("operator %s failed on %s: %s", name, pair_id, e)
            continue
        if v is None:
            continue
        ctrl = None
        if controls:
            try:
                ctrl = CONTROL_OPERATORS[name](src, out_root, pair_id=pair_id, twin=v, seed=seed, split=split)
            except Exception as e:  # noqa: BLE001
                log.warning("control of %s failed on %s: %s", name, pair_id, e)
            if ctrl is None and require_controls:
                log.info("%s on %s has no near-miss control: variant dropped", name, pair_id)
                shutil.rmtree(v.task_dir, ignore_errors=True)
                continue
        leaked = _leaks(src, [v] + ([ctrl] if ctrl is not None else []))
        if leaked:
            log.error("%s on %s dropped, label strings in the emitted task: %s", name, pair_id, leaked[:3])
            LEAK_DROPPED[name] += 1
            for x in (v, ctrl):
                if x is None:
                    continue
                shutil.rmtree(x.task_dir, ignore_errors=True)
            continue
        out.append(v)
        if ctrl is None:
            continue
        out.append(ctrl)
    return out


def _leaks(src: Path, variants: Sequence[Variant]) -> list[tuple[str, int, str]]:
    """Label-bearing strings (`analysis.label_leaks`) in what the operators added to the container-facing files."""
    return [tuple(leak) for v in variants for leak in label_leaks(v.task_dir, src)]
