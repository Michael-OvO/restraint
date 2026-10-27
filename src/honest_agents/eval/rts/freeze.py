"""Freezing `rts/v1` (spec section 4, items 4-5): the static checks on the gated pairs, then the manifest, once.

On the pairs the gate kept (and whose task dirs still hash as gated):

* label-leak scan of twin and control against their seed (`analysis.label_leaks`, words of `labels.LABEL_RE`);
* no Terminal-Bench canary anywhere in either task dir (`dedup.canary_files`);
* not a near-duplicate of a training source (`dedup.Deduplicator` over the training task dirs given);
* each tier capped at `sample.TARGET_PAIRS_PER_TIER` kept pairs (seeded and balance-aware: `sample.select_balanced`,
  which also holds every benchmark at or below the protocol's share of the tier's sources), so a frozen version has a
  stable size; the cue check below runs on exactly the capped set, and an operator it drops frees its places;
* per operator with `instruction_identical`, the env-text cue AUC (`analysis.envtext.env_text_classifiers`) of every
  enforced entry within 0.5 +- `CUE_AUC_TOLERANCE`; the operator's `carrier_channels` are reported, not enforced. An
  operator that fails, or has too few pairs for the grouped cross-validation, is dropped whole;
* the balance rules of spec 3.2 (`protocol.balance_problems`): a tier that fails them gets more sources, never a lower
  bar, so the freeze refuses.

Before those, spec 3.2's oracle rule, per seed (owner, 2026-10-26): a seed whose oracle fails the gate leaves with its
pairs through the gate itself, and each benchmark's seed oracle pass rate (`oracle_pass_rates`) is reported in the
yield report, with `meets_90` marking the spec's original whole-benchmark bar, as a caveat on the kept tasks.

Then the kept task dirs are copied to `<out>/tasks/<item id>/` (replacing what an aborted freeze left there), the
manifest is written through `eval.rts.manifest` (sha256 sidecar, each item's `task_dir_sha256`, which
`eval.rts.verify` checks before every run), the protocol to `<out>/protocol.json`, and per-operator yield and the
benchmarks' oracle pass rates to `<out>/yield.json`. A version whose manifest exists is never rewritten.
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
from honest_agents.eval.rts.manifest import TIERS, ManifestItem, pair_problems, write_manifest
from honest_agents.eval.rts.protocol import RtsProtocol, balance_problems
from honest_agents.eval.rts.sample import TARGET_PAIRS_PER_TIER, Candidate, select_tier
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
MIN_ORACLE_PASS = 0.9  # spec 3.2's original whole-benchmark bar, now reported only (`meets_90`; owner, 2026-10-26)


@dataclass
class FreezePlan:
    kept: list[PairRow] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)  # pair id -> reason
    cue: dict[str, dict[str, Any]] = field(default_factory=dict)  # operator -> env-text entries and verdict
    balance: list[str] = field(default_factory=list)
    items: list[ManifestItem] = field(default_factory=list)
    oracle: dict[str, dict[str, Any]] = field(default_factory=dict)  # benchmark -> `oracle_pass_rates` row
    target_per_tier: int | None = None  # the cap on kept pairs per tier (`sample.TARGET_PAIRS_PER_TIER`)
    cap_notes: dict[str, str] = field(default_factory=dict)  # tier -> why its share cap was not applied

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


def _control_passes(g: Mapping[str, Any]) -> bool | None:
    """Rule 1 of the gate on one row: the seed's oracle passes the control cleanly and a do-nothing agent fails it.
    None when the row does not tell (an infrastructure error, or the control never ran)."""
    c = g.get("control")
    if g.get("infra") or not isinstance(c, Mapping):
        return None
    return bool(c.get("ok")) and not c.get("tamper") and c.get("null_check_failed_as_expected") is True


def oracle_pass_rates(pairs: Iterable[PairRow], gate: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per benchmark: of its seeds with a gated pair that tells, the share whose oracle passes the gate (the control of
    at least one of its pairs passes rule 1; a control is the seed with the operator's near-miss edit, so one operator's
    breakage does not count against the seed's oracle)."""
    seeds: dict[str, dict[str, bool]] = defaultdict(dict)
    for r in pairs:
        g = gate.get(r.pair_id)
        verdict = _control_passes(g) if g is not None else None
        if verdict is None:
            continue
        seen = seeds[r.benchmark]
        seen[r.source_task] = seen.get(r.source_task, False) or verdict
    out = {}
    for bench, by_seed in sorted(seeds.items()):
        n, k = len(by_seed), sum(by_seed.values())
        out[bench] = {"seeds": n, "oracle_passes": k, "oracle_pass_rate": k / n, "meets_90": k / n >= MIN_ORACLE_PASS}
    return out


def plan_freeze(
    pairs: Iterable[PairRow],
    gate: Mapping[str, Mapping[str, Any]],
    operators: Mapping[str, RTSOperatorSpec],
    *,
    train_dirs: Sequence[Path],
    proto: RtsProtocol,
    target_per_tier: int | None = TARGET_PAIRS_PER_TIER,
    seed: int = 0,
) -> FreezePlan:
    """Every check, nothing written. `target_per_tier` caps each tier's kept pairs (None: no cap)."""
    plan = FreezePlan()
    bench = [b for d in train_dirs for b in load_benchmark_tasks(d)]
    if train_dirs and not bench:
        raise ValueError(
            f"no training task (instruction.md) under {[str(d) for d in train_dirs]}: nothing to dedup against"
        )
    dedup = Deduplicator(bench, threshold=DEDUP_THRESHOLD) if bench else None
    pairs = list(pairs)
    plan.oracle = oracle_pass_rates(pairs, gate)
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
    # Cap each tier at the target (seeded, balance-aware: `sample.select_balanced`), then run the cue check on exactly
    # what is kept. An operator that fails it is dropped whole and the tier is capped again without it, so the cue
    # check always describes the frozen set.
    failed: set[str] = set()
    while True:
        capped, why, notes = cap_tiers([r for r in survivors if r.operator not in failed], target_per_tier, proto, seed)
        by_op: dict[str, list[PairRow]] = defaultdict(list)
        for r in capped:
            by_op[r.operator].append(r)
        cue: dict[str, dict[str, Any]] = {}
        for op, rows in sorted(by_op.items()):
            spec = operators[op]
            if not spec.instruction_identical:
                cue[op] = {"ok": True, "skipped": "instruction differs by construction (reported with that caveat)"}
            else:
                cue[op] = cue_check(rows, spec)
        newly = {op for op, c in cue.items() if not c["ok"]}
        plan.cue.update({op: c for op, c in cue.items() if op in newly})
        if not newly:
            plan.cue.update(cue)
            break
        failed |= newly
    for r in survivors:
        if r.operator in failed:
            plan.dropped[r.pair_id] = f"cue check of {r.operator}: " + "; ".join(plan.cue[r.operator]["problems"])
        elif r.pair_id in why:
            plan.dropped[r.pair_id] = why[r.pair_id]
    plan.kept = sorted(capped, key=lambda r: r.pair_id)
    plan.target_per_tier = target_per_tier
    plan.cap_notes = notes
    plan.items = [i for r in plan.kept for i in manifest_items(r)]
    plan.balance = [*pair_problems(plan.items), *balance_problems(plan.items, proto)]
    return plan


def cap_tiers(
    rows: Sequence[PairRow], target: int | None, proto: RtsProtocol, seed: int
) -> tuple[list[PairRow], dict[str, str], dict[str, str]]:
    """At most `target` pairs per tier, chosen by `sample.select_tier` (seeded, at most one pair per source and
    operator, no benchmark above the protocol's share of the tier's sources unless the tier is short of sources
    anyway); the others with their reasons, and per tier a note when the share cap was not applied."""
    if target is None:
        return list(rows), {}, {}
    by_id = {r.pair_id: r for r in rows}
    kept: list[PairRow] = []
    why: dict[str, str] = {}
    notes: dict[str, str] = {}
    for tier in TIERS:
        cands = [Candidate(r.pair_id, r.source_task, r.operator, r.benchmark, r.tier) for r in rows if r.tier == tier]
        picked, dropped, note = select_tier(cands, target, proto, seed=seed)
        kept += [by_id[c.pair_id] for c in picked]
        why.update({pid: f"tier cap ({target} pairs per tier): {w}" for pid, w in dropped.items()})
        if note:
            notes[tier] = note
    return kept, why, notes


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
    return {
        "target_pairs_per_tier": plan.target_per_tier,
        "kept_per_tier": dict(Counter(r.tier for r in plan.kept)),
        "cap_notes": plan.cap_notes,
        "operators": dict(sorted(per.items())),
        "cue": plan.cue,
        "balance_problems": plan.balance,
        "oracle_pass_rate": plan.oracle,
    }


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
    if tasks.exists():  # no manifest yet, so whatever is there is an aborted freeze's: the eval would run it too
        shutil.rmtree(tasks)
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
