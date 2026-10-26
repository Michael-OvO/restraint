"""Helpers the mid-tier RTS operators share: emitting a twin and its control, compose overlays, gate measurements.

No `SPEC` here, so the registry's discovery skips this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks.perturb.core import Variant
from honest_agents.tasks.perturb.emit import _emit
from honest_agents.tasks.taskfs import read_task_toml, set_honest, write_task_toml

COMPOSE_NAME = "docker-compose.yaml"
# The key a target carries when the pair cannot be built from static facts alone: the gate measures the named quantity
# on the seed (oracle run) and passes it back in the target under that name.
NEEDS = "needs_measurement"


def emit_pair(
    src: Path,
    out_root: Path,
    *,
    pair_id: str,
    op: str,
    expected: Sequence[str],
    targets: Sequence[str],
    decoys: Sequence[str],
    detail: dict[str, Any],
    twin_edit: Callable[[Path], None] | None = None,
    control_edit: Callable[[Path], None] | None = None,
    twin_agent: dict[str, Any] | None = None,
    control_agent: dict[str, Any] | None = None,
) -> tuple[Variant, Variant]:
    """Twin (infeasible, `expected` blockers on `targets`) and control (feasible, `decoys`) from one seed.

    Both go through `_emit`, so the instruction is normalised the same way on both sides and stays byte-identical; only
    the edits and agent overrides differ.
    """
    twin = _emit(
        src,
        out_root,
        pair_id=pair_id,
        op=op,
        detail=detail,
        spec_overrides={"expected_targets": list(targets)},
        agent_overrides=twin_agent,
        edit=twin_edit,
    )
    twin = _set_blockers(twin, expected)
    control = _emit(
        src,
        out_root,
        pair_id=pair_id,
        op=op,
        control=True,
        detail=detail,
        spec_overrides={"decoy_targets": list(decoys)},
        agent_overrides=control_agent,
        edit=control_edit,
    )
    return twin, control


def _set_blockers(v: Variant, expected: Sequence[str]) -> Variant:
    """`_emit` takes the blockers from the training table (`core.EXPECTED_BLOCKERS`); an RTS twin uses its spec's."""
    spec = HonestTaskSpec.model_validate({**v.spec.model_dump(), "expected_blockers": list(expected)})
    cfg = read_task_toml(v.task_dir)
    write_task_toml(v.task_dir, set_honest(cfg, spec, (cfg.get("metadata") or {}).get("honest_variant")))
    return Variant(variant_id=v.variant_id, task_dir=v.task_dir, spec=spec, detail=v.detail)


def write_main_compose(task_dir: Path, main: dict[str, Any]) -> Path:
    """`environment/docker-compose.yaml` with only `services.main` settings, which Harbor merges over its own compose
    files (the harness boots it like any task-owned compose file)."""
    path = task_dir / "environment" / COMPOSE_NAME
    path.write_text(yaml.safe_dump({"services": {"main": main}}, sort_keys=True))
    return path


def measured(target: dict[str, Any], key: str) -> float:
    """The gate's measurement `key` from `target`; a clear error when the pair builder is called without it."""
    value = target.get(key)
    if value is None:
        raise ValueError(f"target needs `{key}` (measured on the seed's oracle by the gate) before the pair is built")
    value = float(value)
    if value <= 0:
        raise ValueError(f"`{key}` must be positive, got {value}")
    return value
