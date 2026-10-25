"""Operator registries and `generate_variants`, which also drops any variant that would leak its label."""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from honest_agents.tasks.analysis import (
    TaskFacts,
    label_leaks,
)
from honest_agents.tasks.perturb.conflicting_constraint import conflicting_constraint, conflicting_constraint_control
from honest_agents.tasks.perturb.core import ControlOperator, Operator, Variant
from honest_agents.tasks.perturb.corrupt_input import corrupt_input, corrupt_input_control
from honest_agents.tasks.perturb.emit import make_feasible
from honest_agents.tasks.perturb.high_risk import high_risk, high_risk_control
from honest_agents.tasks.perturb.no_network import no_network, no_network_control
from honest_agents.tasks.perturb.remove_input import remove_input, remove_input_control
from honest_agents.tasks.perturb.remove_tool import remove_tool, remove_tool_control
from honest_agents.tasks.perturb.revoke_permission import revoke_permission, revoke_permission_control
from honest_agents.tasks.perturb.risk_surfaces import SURFACES, surface_operator
from honest_agents.tasks.perturb.shortcut_trap import shortcut_trap, shortcut_trap_control, shortcut_trap_feasible

log = logging.getLogger(__name__)


OPERATORS: dict[str, Operator] = {
    "remove_tool": remove_tool,
    "remove_input": remove_input,
    "corrupt_input": corrupt_input,
    "revoke_permission": revoke_permission,
    "no_network": no_network,
    "conflicting_constraint": conflicting_constraint,
    "high_risk": high_risk,
    "shortcut_trap": shortcut_trap,
}
# `remove_tool` is registered and tested but not built by default: on the full Endless Terminals pool (2,482 convertible
# seeds) one seed gives a candidate, because every image is ubuntu + python3 + pytest, so a tool is either essential, one
# python can stand in for, or used by the verifier (docs/data.md section 2). A source whose images ship real tools can
# turn it on with `--operators`.
# `shortcut_trap` is registered and tested the same way: its cells are validated on a Linux VM (the shortcut must trip the
# manifest, the oracle must not) and the staged smoke of docs/design.md section 4 sizes its mixture before it is built.
# `conflicting_constraint` is not built by default since DEF-27: its twins are passable (37 of 46 base twin episodes of the
# iteration-2 test baseline pass the tests by writing the forbidden deliverable), so the "impossible" label does not hold
# and the family left the transfer headline in iteration 13; built by default it still put 23 pairs into test.jsonl.
# `--operators ... conflicting_constraint` builds it, still held out (`build.DEFAULT_HELDOUT_FAMILIES`) and still a cue
# family of the transfer protocol (`eval.protocol.CUE_FAMILIES`: never pooled into the headline).
NOT_BUILT_BY_DEFAULT = ("remove_tool", "shortcut_trap", "conflicting_constraint")
CONTROL_OPERATORS: dict[str, ControlOperator] = {
    "remove_tool": remove_tool_control,
    "remove_input": remove_input_control,
    "corrupt_input": corrupt_input_control,
    "revoke_permission": revoke_permission_control,
    "no_network": no_network_control,
    "conflicting_constraint": conflicting_constraint_control,
    "high_risk": high_risk_control,
    "shortcut_trap": shortcut_trap_control,
}
# A family with a third cell next to its infeasible variant and its control (the solvable trap): built after them, and
# if it cannot be built the whole family is dropped, so a pair never carries a cue without the cells that price it.
SIBLING_OPERATORS: dict[str, ControlOperator] = {"shortcut_trap": shortcut_trap_feasible}

# One `high_risk` operator per evidence surface (iteration 14, `risk_surfaces`): `generate_variants(operators=
# ["high_risk_file_header"])`. Same family and blockers, `detail.surface` says where the evidence is. Kept apart from
# OPERATORS so the default build, `--operators` choices and every iteration-13 task are unchanged.
SURFACE_OPERATORS: dict[str, Operator] = {}
SURFACE_CONTROL_OPERATORS: dict[str, ControlOperator] = {}
for _surface in SURFACES:
    SURFACE_OPERATORS[f"high_risk_{_surface}"], SURFACE_CONTROL_OPERATORS[f"high_risk_{_surface}"] = surface_operator(
        _surface
    )

DEFAULT_OPERATORS = tuple(n for n in OPERATORS if n not in NOT_BUILT_BY_DEFAULT)

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
    with TaskFacts.shared(src):  # the source is read-only here: analyse it once for every operator and control
        return _generate(src, out_root, pair_id, operators, seed, include_feasible, split, controls, require_controls)


def _generate(
    src: Path,
    out_root: Path,
    pair_id: str,
    operators: list[str] | None,
    seed: int,
    include_feasible: bool,
    split: str,
    controls: bool,
    require_controls: bool,
) -> list[Variant]:
    out: list[Variant] = []
    if include_feasible:
        try:
            out.append(make_feasible(src, out_root, pair_id=pair_id, seed=seed, split=split))
        except Exception as e:  # noqa: BLE001 — a seed Harbor cannot load yields no pair at all
            log.warning("seed %s is not a loadable Harbor task, skipped: %s", pair_id, e)
            return []
    for name in operators if operators is not None else list(OPERATORS):
        try:
            v = {**OPERATORS, **SURFACE_OPERATORS}[name](src, out_root, pair_id=pair_id, seed=seed, split=split)
        except Exception as e:  # noqa: BLE001 — one malformed source task must not stop a 7k-task build
            log.warning("operator %s failed on %s: %s", name, pair_id, e)
            continue
        if v is None:
            continue
        ctrl = None
        if controls:
            try:
                ctrl = {**CONTROL_OPERATORS, **SURFACE_CONTROL_OPERATORS}[name](
                    src, out_root, pair_id=pair_id, twin=v, seed=seed, split=split
                )
            except Exception as e:  # noqa: BLE001
                log.warning("control of %s failed on %s: %s", name, pair_id, e)
            if ctrl is None and require_controls:
                log.info("%s on %s has no near-miss control: variant dropped", name, pair_id)
                shutil.rmtree(v.task_dir, ignore_errors=True)
                continue
        sibling = None
        if name in SIBLING_OPERATORS:
            try:
                sibling = SIBLING_OPERATORS[name](src, out_root, pair_id=pair_id, twin=v, seed=seed, split=split)
            except Exception as e:  # noqa: BLE001
                log.warning("sibling of %s failed on %s: %s", name, pair_id, e)
            if sibling is None:
                log.info("%s on %s has no solvable trap cell: family dropped for this pair", name, pair_id)
                for x in (v, ctrl):
                    if x is not None:
                        shutil.rmtree(x.task_dir, ignore_errors=True)
                continue
        leaked = _leaks(src, [v] + ([ctrl] if ctrl is not None else []) + ([sibling] if sibling is not None else []))
        if leaked:
            log.error("%s on %s dropped, label strings in the emitted task: %s", name, pair_id, leaked[:3])
            LEAK_DROPPED[name] += 1
            for x in (v, ctrl, sibling):
                if x is None:
                    continue
                shutil.rmtree(x.task_dir, ignore_errors=True)
            continue
        out.append(v)
        if sibling is not None:
            out.append(sibling)
        if ctrl is None:
            continue
        out.append(ctrl)
    return out


def _leaks(src: Path, variants: Sequence[Variant]) -> list[tuple[str, int, str]]:
    """Label-bearing strings (`analysis.label_leaks`) in what the operators added to the container-facing files."""
    return [tuple(leak) for v in variants for leak in label_leaks(v.task_dir, src)]
