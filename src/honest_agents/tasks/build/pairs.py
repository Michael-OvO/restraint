"""Validation results, relabelling and the pairing invariant over `(seed, variant)` pairs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks.perturb import (
    Variant,
)
from honest_agents.tasks.sources import (
    SeedTask,
)
from honest_agents.tasks.taskfs import read_task_toml, write_task_toml

# ── validation, relabelling, pairing ─────────────────────────────────────────

Pair = tuple[SeedTask, Variant]


def _pair_key(v: Variant) -> tuple[str | None, str | None]:
    return v.spec.pair_id, v.spec.perturbation


def read_relabels(path: Path) -> dict[str, list[str]]:
    """variant id -> the paths its oracle must recreate (empty list = none needed)."""
    return json.loads(path.read_text()) if path.exists() else {}


def relabel_variant(task_dir: Path, repair_paths: Sequence[str]) -> None:
    """Rewrite an infeasible `remove_input` variant in place as a near-miss control (`validate.relabel_to_control`)."""
    from honest_agents.tasks.validate import relabel_to_control

    relabel_to_control(task_dir, repair_paths=list(repair_paths))
    cfg = read_task_toml(task_dir)
    md = dict(cfg.get("metadata") or {})
    if isinstance(md.get("honest_variant"), dict):
        md["honest_variant"] = {**md["honest_variant"], "control": True, "relabelled": True}
        write_task_toml(task_dir, {**cfg, "metadata": md})


def apply_validation(
    variants: list[Pair],
    results: dict[str, Any],
    *,
    relabels: dict[str, list[str]],
    allow_oracle_tamper: bool = False,
    require_valid: bool = True,
    require_null_check: bool = True,
    min_oracle_runs: int = 3,
) -> tuple[list[Pair], dict[str, Any]]:
    """Fold validation results into the variant list; returns the kept variants and what changed.

    1. A `remove_input` variant already in `relabels`, or whose current result says `relabel_to_control`, is rewritten in
       place as a control (and recorded in `relabels`, which the caller persists). It needs validating again, as a
       control, so it is kept only once a result for its new content exists. It replaces the control the operator
       generated for the same pair.
    2. With `require_valid`: variants without a current, ok, non-leaking result are dropped; so is every variant of a pair
       whose plain feasible copy is not valid (a broken source task, or an oracle that trips the tamper check).

    Leak results are returned in `info["leaks"]` for quarantine. Pairing is enforced by the caller afterwards.
    """
    from honest_agents.tasks.validate import filter_valid, is_current

    info: dict[str, Any] = {"relabelled": [], "leaks": [], "dropped": 0}
    relabelled_pairs: set[tuple] = set()
    out: list[Pair] = []
    for s, v in variants:
        r = results.get(v.variant_id)
        current = r is not None and is_current(r, v.task_dir)
        flagged = current and getattr(r, "relabel_to_control", False)
        if v.kind == "infeasible" and v.perturbation == "remove_input" and (v.variant_id in relabels or flagged):
            if v.variant_id not in relabels:
                relabels[v.variant_id] = list((getattr(r, "debug", None) or {}).get("repair_paths") or [])
            relabel_variant(v.task_dir, relabels[v.variant_id])
            spec = HonestTaskSpec.from_task_dir(v.task_dir)
            v = Variant(v.variant_id, v.task_dir, spec, {**v.detail, "relabelled": True})
            relabelled_pairs.add(_pair_key(v))
            info["relabelled"].append(v.variant_id)
        out.append((s, v))
    out = [
        (s, v)
        for s, v in out
        if not (v.kind == "control" and not v.detail.get("relabelled") and _pair_key(v) in relabelled_pairs)
    ]
    info["leaks"] = [
        results[v.variant_id]
        for _, v in out
        if getattr(results.get(v.variant_id), "leak", False) and is_current(results[v.variant_id], v.task_dir)
    ]
    if not require_valid:
        return out, info
    valid = filter_valid(
        results,
        allow_oracle_tamper=allow_oracle_tamper,
        task_dirs={v.variant_id: v.task_dir for s, v in out if s.has_solution},
        require_null_check=require_null_check,
        min_oracle_runs=min_oracle_runs,
    )

    def ok(s: SeedTask, v: Variant) -> bool:
        return v.variant_id in valid or not s.has_solution

    plain_ok = {v.spec.pair_id for s, v in out if v.kind == "feasible" and ok(s, v)}
    kept = [(s, v) for s, v in out if ok(s, v) and v.spec.pair_id in plain_ok]
    info["dropped"] = len(variants) - len(kept)
    return kept, info


def enforce_pairing(variants: list[Pair], *, require_controls: bool = True) -> list[Pair]:
    """Drop an infeasible variant that has no control and a control (not a relabelled one) that lost its twin."""
    infeasible = {_pair_key(v) for _, v in variants if v.kind == "infeasible"}
    controls = {_pair_key(v) for _, v in variants if v.kind == "control"}
    out = []
    for s, v in variants:
        if v.kind == "infeasible" and require_controls and _pair_key(v) not in controls:
            continue
        if v.kind == "control" and not v.detail.get("relabelled") and _pair_key(v) not in infeasible:
            continue
        out.append((s, v))
    return out


# ── build ────────────────────────────────────────────────────────────────────
