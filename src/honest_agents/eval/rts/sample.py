"""Balance-aware sampling of candidate pairs (spec 3.2): which pairs `measure` and `gate` run, and which the freeze keeps.

`pairs` yields far more candidates than an eval arm can afford (spec 9: about 400-600 episodes per arm, two per pair),
so `build sample` picks, per tier, `ceil(TARGET_PAIRS_PER_TIER * OVERSAMPLE)` candidates (the oversample covers the
pairs the gate rejects), and the freeze picks `TARGET_PAIRS_PER_TIER` of the gate's survivors with the same rule, so a
frozen version has a stable size. Both use `select_balanced`:

* deterministic and seeded: candidates are ranked by `sha256(seed, pair id)`, so adding or removing other candidates
  never reorders these, and the same inputs always give the same selection;
* at most one pair per (source task, operator) (spec 3.2);
* greedy, one pair at a time, preferring in this order: the operator with the fewest pairs so far (a tier's J is the mean
  of its operators'), a source task the tier does not hold yet (more clusters), the benchmark with the fewest pairs so
  far, then the rank;
* no benchmark above the tier's share cap (`RtsProtocol.share_cap`) of the tier's selected source tasks: a benchmark's
  sources are capped at `floor(share * S)`, with S the selection's source count, iterated down from the target until
  it holds.

A candidate is a `pairs` verdict that is `built`, or `measure` (applicable, waiting for its oracle measurement, which
`measure --selection` then runs only for the selected ones). A verdict whose measurement already failed is no
candidate. The minimum source and benchmark counts are not selection rules (sampling cannot create supply); the
selection reports them as projected problems, and the freeze enforces them (`protocol.balance_problems`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.eval.rts.manifest import TIERS

TARGET_PAIRS_PER_TIER = 100  # kept pairs per tier in a frozen version (spec 3.2; spec 9's episode budget)
OVERSAMPLE = 1.6  # sampled candidates per kept pair: room for the gate's rejections
SELECTION_NAME = "selection.json"


@dataclass(frozen=True)
class Candidate:
    pair_id: str
    source_task: str
    operator: str
    benchmark: str
    tier: str
    stage: str = "built"  # `built`, or `measure` (applicable, its measurement pending)


def rank(pair_id: str, seed: int) -> str:
    return hashlib.sha256(f"rts-select\0{seed}\0{pair_id}".encode()).hexdigest()


def candidates_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Candidate], dict[str, str]]:
    """Candidates from `pairs`' applicability log, and the verdicts that are not candidates (pair id -> why)."""
    from honest_agents.eval.rts.candidates import pair_id_of

    out: list[Candidate] = []
    dropped: dict[str, str] = {}
    for r in rows:
        pid = pair_id_of(r["source_task"], r["operator"])
        if r.get("stage") == "built":
            out.append(Candidate(pid, r["source_task"], r["operator"], r["benchmark"], r["tier"], "built"))
        elif r.get("stage") == "measure":
            if r.get("measured"):  # measured on this seed, and still no value: the measurement failed
                dropped[pid] = f"measurement failed: {str(r.get('reason'))[:160]}"
            else:
                out.append(Candidate(pid, r["source_task"], r["operator"], r["benchmark"], r["tier"], "measure"))
    return out, dropped


def _greedy(cands: Sequence[Candidate], n: int, cap: int | None) -> list[Candidate]:
    picked: list[Candidate] = []
    op_n: Counter[str] = Counter()
    bench_n: Counter[str] = Counter()
    bench_src: dict[str, set[str]] = defaultdict(set)
    used: set[str] = set()
    left = list(cands)  # already in rank order
    while len(picked) < n and left:
        best_i, best_key = -1, None
        for i, c in enumerate(left):
            if cap is not None and c.source_task not in bench_src[c.benchmark] and len(bench_src[c.benchmark]) >= cap:
                continue
            key = (op_n[c.operator], c.source_task in used, bench_n[c.benchmark], i)
            if best_key is None or key < best_key:
                best_i, best_key = i, key
        if best_i < 0:
            break
        c = left.pop(best_i)
        picked.append(c)
        op_n[c.operator] += 1
        bench_n[c.benchmark] += 1
        bench_src[c.benchmark].add(c.source_task)
        used.add(c.source_task)
    return picked


def _shares_ok(picked: Sequence[Candidate], max_share: float) -> bool:
    by: dict[str, set[str]] = defaultdict(set)
    for c in picked:
        by[c.benchmark].add(c.source_task)
    total = len({c.source_task for c in picked})
    return all(len(s) <= max_share * total + 1e-9 for s in by.values())


def select_balanced(
    cands: Iterable[Candidate], n: int, *, max_share: float, seed: int = 0
) -> tuple[list[Candidate], dict[str, str]]:
    """At most `n` of one tier's candidates (see the module doc), and why each other one was left out."""
    dropped: dict[str, str] = {}
    unique: dict[tuple[str, str], Candidate] = {}
    for c in sorted(cands, key=lambda c: (rank(c.pair_id, seed), c.pair_id)):
        key = (c.source_task, c.operator)
        if key in unique:
            dropped[c.pair_id] = (
                f"another pair of {c.source_task} for {c.operator} (at most one per source and operator)"
            )
        else:
            unique[key] = c
    ordered = list(unique.values())
    if max_share >= 1.0:
        picked = _greedy(ordered, n, None)
    else:
        sources = n
        while True:
            picked = _greedy(ordered, n, math.floor(max_share * sources))
            got = len({c.source_task for c in picked})
            if got >= sources or _shares_ok(picked, max_share):
                break
            sources = got
    chosen = {c.pair_id for c in picked}
    by: dict[str, set[str]] = defaultdict(set)
    for c in picked:
        by[c.benchmark].add(c.source_task)
    total = len({c.source_task for c in picked})
    for c in ordered:
        if c.pair_id in chosen:
            continue
        held = len(by[c.benchmark])
        if max_share < 1.0 and c.source_task not in by[c.benchmark] and held + 1 > max_share * (total + 1):
            dropped[c.pair_id] = (
                f"benchmark cap: {c.benchmark} holds {held} of the tier's {total} selected sources "
                f"(at most {max_share:.0%})"
            )
        else:
            dropped[c.pair_id] = f"tier full ({len(picked)} of {n} selected)"
    return picked, dropped


def select_tier(
    pool: Iterable[Candidate], n: int, proto: Any, *, seed: int = 0
) -> tuple[list[Candidate], dict[str, str], str | None]:
    """`select_balanced` under the protocol's share cap, unless the capped selection already has fewer source tasks
    than the protocol's minimum: the tier then fails the balance rules whatever is trimmed, so it is selected without
    the cap (still at most `n`), and the freeze's balance check reports what is really there. Returns (picked,
    dropped, a note when the cap was not applied)."""
    pool = list(pool)
    tiers = {c.tier for c in pool}
    cap = proto.share_cap(tiers.pop()) if len(tiers) == 1 else proto.max_benchmark_share
    picked, dropped = select_balanced(pool, n, max_share=cap, seed=seed)
    got = len({c.source_task for c in picked})
    if cap >= 1.0 or got >= proto.min_sources_per_tier:
        return picked, dropped, None
    loose, why = select_balanced(pool, n, max_share=1.0, seed=seed)
    note = (
        f"the {cap:.0%} benchmark cap would leave {got} source tasks, below the "
        f"{proto.min_sources_per_tier} minimum: selected without the cap (the tier fails the balance rules as it stands)"
    )
    return loose, why, note


def tier_summary(picked: Sequence[Candidate]) -> dict[str, Any]:
    by_bench: dict[str, set[str]] = defaultdict(set)
    for c in picked:
        by_bench[c.benchmark].add(c.source_task)
    cells: Counter[str] = Counter(f"{c.operator} / {c.benchmark}" for c in picked)
    return {
        "pairs": len(picked),
        "sources": len({c.source_task for c in picked}),
        "operators": dict(sorted(Counter(c.operator for c in picked).items())),
        "benchmark_sources": {b: len(s) for b, s in sorted(by_bench.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
        "operator_benchmark_pairs": dict(sorted(cells.items())),
        "awaiting_measurement": sum(c.stage == "measure" for c in picked),
    }


def projected_problems(summary: Mapping[str, Any], tier: str, proto: Any, *, oversample: float) -> list[str]:
    """What the tier would fail at the freeze if the gate kept `1 / oversample` of the selection, evenly."""
    out = []
    kept_sources = summary["sources"] / oversample
    if kept_sources < proto.min_sources_per_tier:
        out.append(f"{tier}: about {kept_sources:.0f} source tasks after the gate, need {proto.min_sources_per_tier}")
    if len(summary["benchmark_sources"]) < proto.min_benchmarks_per_tier:
        out.append(f"{tier}: {len(summary['benchmark_sources'])} benchmarks, need {proto.min_benchmarks_per_tier}")
    if len(summary["operators"]) < proto.min_operators_per_tier:
        out.append(f"{tier}: {len(summary['operators'])} operators, need {proto.min_operators_per_tier}")
    return out


def sample(
    rows: Iterable[Mapping[str, Any]],
    *,
    target: int = TARGET_PAIRS_PER_TIER,
    oversample: float = OVERSAMPLE,
    seed: int = 0,
    proto: Any = None,
    built_only: bool = False,
) -> dict[str, Any]:
    """The selection document `build sample` writes: parameters, per-tier summaries, the selected candidates and every
    dropped one with its reason. `built_only` (the re-sample after `measure`) leaves out verdicts still awaiting a
    measurement, so the selection holds only pairs the gate can run."""
    from honest_agents.eval.rts.protocol import PREREGISTERED

    proto = proto or PREREGISTERED
    if target < 1 or oversample < 1.0:
        raise ValueError(f"target must be at least 1 and oversample at least 1.0 (got {target}, {oversample})")
    cands, dropped = candidates_from_rows(rows)
    if built_only:
        dropped.update(
            {c.pair_id: "awaiting a measurement (built-only re-sample)" for c in cands if c.stage != "built"}
        )
        cands = [c for c in cands if c.stage == "built"]
    n = math.ceil(target * oversample)
    selected: list[Candidate] = []
    tiers: dict[str, Any] = {}
    for tier in TIERS:
        pool = [c for c in cands if c.tier == tier]
        picked, why, note = select_tier(pool, n, proto, seed=seed)
        selected += picked
        dropped.update(why)
        s = tier_summary(picked)
        problems = projected_problems(s, tier, proto, oversample=oversample)
        tiers[tier] = {"candidates": len(pool), **s, "projected_problems": [*([note] if note else []), *problems]}
    return {
        "target_pairs_per_tier": target,
        "oversample": oversample,
        "per_tier": n,
        "seed": seed,
        "max_benchmark_share": {t: proto.share_cap(t) for t in TIERS},
        "tiers": tiers,
        "selected": [asdict(c) for c in sorted(selected, key=lambda c: c.pair_id)],
        "dropped": dict(sorted(dropped.items())),
    }


def write_selection(doc: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def read_selection(path: Path) -> set[str]:
    """The selected pair ids of a `build sample` document."""
    doc = json.loads(path.read_text())
    return {str(c["pair_id"]) for c in doc["selected"]}
