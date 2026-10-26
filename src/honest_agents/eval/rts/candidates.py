"""Candidate pairs (spec section 4, items 2-3): every operator's applicability on every seed, and the pairs it builds.

Applicability is computed per seed by the operator, never hand-picked, and every verdict is logged with its reason. At
most one pair is built per (source task, operator). A built pair is checked statically before the gate sees it: the
labels the grader reads, byte-identical instructions where the operator declares them, and no label word in what the
container can see (`analysis.label_leaks`). No Docker.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.eval.rts.benchmarks import RTSSeed
from honest_agents.tasks.analysis import label_leaks
from honest_agents.tasks.perturb.core import Variant
from honest_agents.tasks.perturb.rts.base import RTSOperatorSpec
from honest_agents.tasks.taskfs import slug
from honest_agents.tasks.validate import task_hash

log = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/rts/v1/candidates")
PAIRS_NAME = "pairs.jsonl"
APPLICABILITY_NAME = "applicability.jsonl"
MEASUREMENTS_NAME = (
    "measurements.jsonl"  # `eval.rts.measure` (Docker): quantities an operator needs from the seed's oracle
)
# The target key an operator sets when its pair needs a quantity measured on the seed's oracle (the mid tier's
# `disk_quota` and `mem_limit`); the value names the quantity, which the measurement puts back in the target.
NEEDS = "needs_measurement"


@dataclass(frozen=True)
class PairRow:
    pair_id: str  # manifest pair id, `<source task>::<operator>`
    benchmark: str
    source_task: str
    operator: str
    tier: str
    instruction_identical: bool
    expected_category: tuple[str, ...]
    expected_targets: tuple[str, ...]
    seed_dir: str
    twin_dir: str
    control_dir: str
    twin_id: str  # the variant ids: the manifest's item ids and the episodes' `variant_key`
    control_id: str
    target: dict[str, Any] = field(default_factory=dict)
    # the expected targets the gate probes (`RTSOperatorSpec.targets_to_probe`); None (a row written before the field
    # existed) probes them all
    probe_targets: tuple[str, ...] | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_category"] = list(self.expected_category)
        d["expected_targets"] = list(self.expected_targets)
        if self.probe_targets is not None:
            d["probe_targets"] = list(self.probe_targets)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> PairRow:
        d = dict(d)
        d["expected_category"] = tuple(d["expected_category"])
        d["expected_targets"] = tuple(d["expected_targets"])
        if d.get("probe_targets") is not None:
            d["probe_targets"] = tuple(d["probe_targets"])
        return cls(**d)


def pair_id_of(source_task: str, operator: str) -> str:
    return f"{source_task}::{operator}"


def pair_problems(spec: RTSOperatorSpec, seed: RTSSeed, twin: Variant, control: Variant) -> list[str]:
    """Static checks of a built pair; empty when it may go to the gate."""
    out = []
    ts, cs = twin.spec, control.spec
    if ts.feasible:
        out.append("twin is labelled feasible")
    if not (cs.feasible and cs.is_control):
        out.append("control is not a feasible near-miss control")
    if ts.perturbation != spec.name or cs.perturbation != spec.name or cs.control_of != spec.name:
        out.append(f"labels name another operator (twin {ts.perturbation}, control {cs.perturbation}/{cs.control_of})")
    for v in (twin, control):
        if not v.spec.variant_id or slug(v.spec.variant_id) != v.spec.variant_id:
            out.append(f"variant id {v.spec.variant_id!r} is empty or not path-safe")
    if twin.spec.variant_id == control.spec.variant_id:
        out.append("twin and control share a variant id")
    if not ts.expected_targets:
        out.append("twin names no expected target (a blind category guess would earn the hit)")
    if bad := [c for c in ts.expected_blockers if c not in spec.expected_category]:
        out.append(f"twin expects blockers {bad} outside the operator's expected_category {spec.expected_category}")
    ti, ci = (Path(v.task_dir) / "instruction.md" for v in (twin, control))
    if spec.instruction_identical and ti.read_bytes() != ci.read_bytes():
        out.append("instruction_identical, but the twin and control instructions differ")
    for role, v in (("twin", twin), ("control", control)):
        if leaks := label_leaks(Path(v.task_dir), seed.task_dir):
            out.append(f"{role}: label word in {leaks[0].path}:{leaks[0].line} ({leaks[0].text[:80]})")
    return out


def build_pairs(
    seeds: Iterable[RTSSeed],
    out_root: Path,
    operators: Mapping[str, RTSOperatorSpec],
    *,
    seed: int = 0,
    measurements: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> tuple[list[PairRow], list[dict[str, Any]]]:
    """Applicability of every operator on every seed, and the pairs built under `out_root/tasks/`.

    Returns (pairs, log rows). A log row records one (seed, operator) verdict: `stage` is `applicable` (the operator
    declined, or raised), `measure` (applicable, but the target names a quantity (`NEEDS`) that `measurements`, keyed by
    (source task, operator), does not yet hold, or holds for a seed whose task hash has changed since: `eval.rts.measure`
    runs the seed's oracle for it), `build` (make_pair
    raised or the static checks failed; its dirs are removed) or `built`."""
    measurements = measurements or {}
    tasks_root = out_root / "tasks"
    pairs: list[PairRow] = []
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    seen_sources: set[str] = set()
    for s in sorted(seeds, key=lambda x: x.source_task):
        if s.source_task in seen_sources:
            continue  # one pair per (source task, operator), however often the index lists the seed
        seen_sources.add(s.source_task)
        for name, spec in sorted(operators.items()):
            base = {"benchmark": s.benchmark, "source_task": s.source_task, "operator": name, "tier": spec.tier}
            if not s.has_oracle:
                rows.append({**base, "stage": "applicable", "ok": False, "reason": "seed has no oracle solution"})
                continue
            try:
                app = spec.applicable(s.task_dir)
            except Exception as e:  # noqa: BLE001 — one malformed seed must not stop the build
                rows.append({**base, "stage": "applicable", "ok": False, "reason": f"raised {type(e).__name__}: {e}"})
                continue
            if not app.ok:
                rows.append({**base, "stage": "applicable", "ok": False, "reason": app.reason, "target": app.target})
                continue
            target = dict(app.target)
            if need := target.get(NEEDS):
                got = measurements.get((s.source_task, name)) or {}
                stale = bool(got) and got.get("seed_hash") != task_hash(s.task_dir)
                if stale:
                    # measured on another version of the seed (a fix-up or normalisation changed it): the size or
                    # limit would be the old seed's, so it is measured again (`measure` reruns rows whose hash differs)
                    got = {"error": "seed changed since measured (build measure)"}
                if got.get(need) is None:
                    why = got.get("error") or f"needs {need} measured on the seed's oracle (build measure)"
                    rows.append(
                        {
                            **base,
                            "stage": "measure",
                            "ok": False,
                            "reason": str(why),
                            "target": target,
                            "seed_dir": str(s.task_dir),
                            "measured": bool(got) and not stale,
                        }
                    )
                    continue
                target[need] = got[need]
            try:
                twin, control = spec.make_pair(s.task_dir, tasks_root, pair_id=s.source_task, target=target, seed=seed)
            except Exception as e:  # noqa: BLE001
                rows.append(
                    {
                        **base,
                        "stage": "build",
                        "ok": False,
                        "reason": f"make_pair raised {type(e).__name__}: {e}",
                        "target": target,
                    }
                )
                continue
            try:
                probe = spec.targets_to_probe(target, list(twin.spec.expected_targets))
            except ValueError as e:
                for v in (twin, control):
                    shutil.rmtree(v.task_dir, ignore_errors=True)
                rows.append({**base, "stage": "build", "ok": False, "reason": str(e), "target": target})
                continue
            ids = {str(twin.spec.variant_id), str(control.spec.variant_id)}
            if ids & used:  # the second pair has overwritten the first one's dirs: an id scheme bug, not a seed's
                raise ValueError(f"{name} on {s.source_task}: variant ids {sorted(ids & used)} already used")
            if problems := pair_problems(spec, s, twin, control):
                for v in (twin, control):
                    shutil.rmtree(v.task_dir, ignore_errors=True)
                rows.append({**base, "stage": "build", "ok": False, "reason": "; ".join(problems)})
                continue
            used |= ids
            row = PairRow(
                pair_id=pair_id_of(s.source_task, name),
                benchmark=s.benchmark,
                source_task=s.source_task,
                operator=name,
                tier=spec.tier,
                instruction_identical=spec.instruction_identical,
                expected_category=tuple(spec.expected_category),
                expected_targets=tuple(twin.spec.expected_targets),
                seed_dir=str(s.task_dir),
                twin_dir=str(twin.task_dir),
                control_dir=str(control.task_dir),
                twin_id=str(twin.spec.variant_id),
                control_id=str(control.spec.variant_id),
                target=target,
                probe_targets=probe,
            )
            pairs.append(row)
            rows.append({**base, "stage": "built", "ok": True, "reason": app.reason, "target": target})
    return pairs, rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(r), sort_keys=True, default=str) + "\n" for r in rows))


def read_pairs(path: Path) -> list[PairRow]:
    return [PairRow.from_json(json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()]
