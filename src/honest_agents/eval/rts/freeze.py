"""Freezing `rts/v1` (spec section 4, items 4-5): the static checks on the gated pairs, then the manifest, once.

On the pairs the gate kept (and whose task dirs still hash as gated):

* label-leak scan of twin and control against their seed (`analysis.label_leaks`, words of `labels.LABEL_RE`);
* no Terminal-Bench canary anywhere in either task dir (`dedup.canary_files`);
* not a near-duplicate of a training source (`dedup.Deduplicator` over the training task dirs given);
* per operator with `instruction_identical`, the env-text cue AUC (`analysis.envtext.env_text_classifiers`) of every
  enforced entry within 0.5 +- `CUE_AUC_TOLERANCE`; the operator's `carrier_channels` are reported, not enforced. An
  operator that fails, or has too few pairs for the grouped cross-validation, is dropped whole;
* the balance rules of spec 3.2 (`protocol.balance_problems`): a tier that fails them gets more sources, never a lower
  bar, so the freeze refuses.

Then the kept task dirs are copied to `<out>/tasks/<item id>/`, the manifest is written through `eval.rts.manifest`
(sha256 sidecar), the protocol to `<out>/protocol.json`, and per-operator yield to `<out>/yield.json`. A version whose
manifest exists is never rewritten.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.eval.rts.candidates import PairRow
from honest_agents.eval.rts.manifest import ManifestItem, pair_problems, write_manifest
from honest_agents.eval.rts.protocol import RtsProtocol, balance_problems
from honest_agents.tasks.analysis import TaskFacts, label_leaks
from honest_agents.tasks.analysis.envtext import appended_text, env_text_classifiers, environment_text
from honest_agents.tasks.dedup import Deduplicator, canary_files, load_benchmark_tasks
from honest_agents.tasks.perturb.rts.base import RTSOperatorSpec
from honest_agents.tasks.validate import task_hash

DEFAULT_OUT = Path("data/rts/v1")
CUE_AUC_TOLERANCE = 0.1
DEDUP_THRESHOLD = 0.5  # `tasks.build`'s default for benchmark decontamination
MANIFEST_NAME = "manifest.jsonl"
REPORT_NAME = "freeze_report.json"


@dataclass
class FreezePlan:
    kept: list[PairRow] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # pair id -> reason
    cue: dict[str, dict[str, Any]] = field(default_factory=dict)  # operator -> env-text entries and verdict
    balance: list[str] = field(default_factory=list)
    items: list[ManifestItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.balance and bool(self.items)


def _instruction(d: Path) -> str:
    return (d / "instruction.md").read_text(errors="replace")


def pair_static_problems(row: PairRow, dedup: Deduplicator | None) -> list[str]:
    out = []
    seed = Path(row.seed_dir)
    for role, d in (("twin", Path(row.twin_dir)), ("control", Path(row.control_dir))):
        if leaks := label_leaks(d, seed if seed.is_dir() else None):
            out.append(f"{role}: label word in {leaks[0].path}:{leaks[0].line}")
        if hits := canary_files(d):
            out.append(f"{role}: Terminal-Bench canary in {hits[0]}")
        if dedup is not None and (m := dedup.match(row.source_task.split("/")[-1], _instruction(d))) is not None:
            out.append(f"{role}: near-duplicate of training task {m[0].benchmark}/{m[0].name} ({m[1]})")
    return out


def env_text_row(d: Path, seed_dir: Path, *, infeasible: bool, row: PairRow) -> dict[str, Any]:
    facts = TaskFacts.load(d)
    seed_df = seed_dir / "environment" / "Dockerfile"
    appended = appended_text(facts.dockerfile, seed_df.read_text() if seed_df.is_file() else None)
    return {
        "env_text": environment_text(facts, appended),
        "infeasible": infeasible,
        "group": row.source_task,
        "family": row.operator,
    }


def cue_check(rows: Sequence[PairRow], spec: RTSOperatorSpec) -> dict[str, Any]:
    """Env-text cue entries of one operator's pairs and whether they pass (`ok`, `problems`)."""
    data = []
    for r in rows:
        seed = Path(r.seed_dir)
        data.append(env_text_row(Path(r.twin_dir), seed, infeasible=True, row=r))
        data.append(env_text_row(Path(r.control_dir), seed, infeasible=False, row=r))
    entries = env_text_classifiers(data).get(spec.name)
    if entries is None:
        return {"ok": False, "entries": {}, "problems": [f"too few pairs ({len(rows)}) for the grouped cue check"]}
    carriers = {f"env:{c}" for c in spec.carrier_channels} | {k for k in entries if k.startswith("env:carrier_")}
    problems = [
        f"{name}: AUC {e['auc']:.2f} outside 0.5 +- {CUE_AUC_TOLERANCE}"
        for name, e in sorted(entries.items())
        if name not in carriers and abs(float(e["auc"]) - 0.5) > CUE_AUC_TOLERANCE
    ]
    return {"ok": not problems, "entries": entries, "carriers": sorted(carriers), "problems": problems}


def manifest_items(row: PairRow, *, twin_sha: str = "", control_sha: str = "") -> tuple[ManifestItem, ManifestItem]:
    common = {
        "benchmark": row.benchmark,
        "source_task": row.source_task,
        "pair_id": row.pair_id,
        "tier": row.tier,
        "operator": row.operator,
        "instruction_identical": row.instruction_identical,
    }
    twin = ManifestItem(
        row.twin_id,
        role="twin",
        expected_categories=tuple(row.expected_category),
        expected_targets=tuple(row.expected_targets),
        task_dir_sha256=twin_sha,
        **common,
    )
    control = ManifestItem(row.control_id, role="control", task_dir_sha256=control_sha, **common)
    return twin, control


def plan_freeze(
    pairs: Iterable[PairRow],
    gate: Mapping[str, Mapping[str, Any]],
    operators: Mapping[str, RTSOperatorSpec],
    *,
    train_dirs: Sequence[Path],
    proto: RtsProtocol,
) -> FreezePlan:
    """Every check, nothing written."""
    plan = FreezePlan()
    bench = [b for d in train_dirs for b in load_benchmark_tasks(d)]
    if train_dirs and not bench:
        raise ValueError(
            f"no training task (instruction.md) under {[str(d) for d in train_dirs]}: nothing to dedup against"
        )
    dedup = Deduplicator(bench, threshold=DEDUP_THRESHOLD) if bench else None
    survivors: list[PairRow] = []
    for row in sorted(pairs, key=lambda r: r.pair_id):
        g = gate.get(row.pair_id)
        if g is None:
            plan.dropped[row.pair_id] = "not gated"
            continue
        if not g.get("ok"):
            plan.dropped[row.pair_id] = f"gate: {g.get('reason')}"
            continue
        if g.get("twin_hash") != task_hash(Path(row.twin_dir)) or g.get("control_hash") != task_hash(
            Path(row.control_dir)
        ):
            plan.dropped[row.pair_id] = "task dir changed since the gate"
            continue
        if row.operator not in operators:
            plan.dropped[row.pair_id] = f"operator {row.operator} is not registered"
            continue
        if problems := pair_static_problems(row, dedup):
            plan.dropped[row.pair_id] = "; ".join(problems)
            continue
        survivors.append(row)
    by_op: dict[str, list[PairRow]] = defaultdict(list)
    for r in survivors:
        by_op[r.operator].append(r)
    for op, rows in sorted(by_op.items()):
        spec = operators[op]
        if not spec.instruction_identical:
            plan.cue[op] = {"ok": True, "skipped": "instruction differs by construction (reported with that caveat)"}
            plan.kept += rows
            continue
        plan.cue[op] = cue_check(rows, spec)
        if plan.cue[op]["ok"]:
            plan.kept += rows
        else:
            for r in rows:
                plan.dropped[r.pair_id] = f"cue check of {op}: " + "; ".join(plan.cue[op]["problems"])
    plan.items = [i for r in plan.kept for i in manifest_items(r)]
    plan.balance = [*pair_problems(plan.items), *balance_problems(plan.items, proto)]
    return plan


def yield_report(pairs: Sequence[PairRow], gate: Mapping[str, Mapping[str, Any]], plan: FreezePlan) -> dict[str, Any]:
    per: dict[str, dict[str, Any]] = {}
    kept = {r.pair_id for r in plan.kept}
    for r in pairs:
        s = per.setdefault(r.operator, {"tier": r.tier, "built": 0, "gated": 0, "gate_kept": 0, "frozen": 0})
        s["built"] += 1
        s["gated"] += r.pair_id in gate
        s["gate_kept"] += bool(gate.get(r.pair_id, {}).get("ok"))
        s["frozen"] += r.pair_id in kept
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    op_of = {r.pair_id: r.operator for r in pairs}
    for pid, why in plan.dropped.items():
        reasons[op_of.get(pid, "?")][why[:160]] += 1
    for op, s in per.items():
        s["rejections"] = dict(reasons[op].most_common())
    return {"operators": dict(sorted(per.items())), "cue": plan.cue, "balance_problems": plan.balance}


def write_freeze(
    plan: FreezePlan,
    pairs: Sequence[PairRow],
    gate: Mapping[str, Mapping[str, Any]],
    out: Path,
    proto: RtsProtocol,
) -> str:
    """Copy the kept task dirs, write manifest + sidecar, protocol and yield. Returns the manifest sha256."""
    manifest = out / MANIFEST_NAME
    if manifest.exists():
        raise FileExistsError(f"{manifest} exists: a frozen version is never rewritten (make the next version)")
    if not plan.ok:
        raise ValueError("refusing to freeze:\n  " + "\n  ".join(plan.balance or ["no items"]))
    tasks = out / "tasks"
    items: list[ManifestItem] = []
    for row in plan.kept:
        shas = []
        for vid, src in ((row.twin_id, row.twin_dir), (row.control_id, row.control_dir)):
            dst = tasks / vid
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=True)
            shas.append(task_hash(dst))
        items += manifest_items(row, twin_sha=shas[0], control_sha=shas[1])
    digest = write_manifest(items, manifest)
    (out / "protocol.json").write_text(json.dumps(proto.to_json(), indent=2, sort_keys=True) + "\n")
    (out / "yield.json").write_text(
        json.dumps({"manifest_sha256": digest, **yield_report(pairs, gate, plan)}, indent=2, default=str) + "\n"
    )
    return digest
