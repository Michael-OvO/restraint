"""Fixed paired probe set for the iteration-11 protocol: at least 60 impossible and 60 feasible variants from the val split.

    python -m honest_agents.eval.probe_set --split data/val.jsonl --out runs/probe/probe_v1.jsonl \\
        [--infeasible 60] [--feasible 60] [--seed 0] [--test-ids configs/test_ids.txt] [--allow-short]

Why (docs/training-log.md, iteration 10 and "Evaluation protocol from iteration 11"): a 100-variant probe with about 21
impossible tasks cannot resolve changes under ~10 points, and per-family results need every family on the same tasks for
both models. The set is built once, before the run, and both models are evaluated on it (`run_eval --split <out>`,
4 samples per task), so per-task changes can be read.

Selection, deterministic for a given split and seed (ranks are sha256 of seed and variant id, not the file order):

* **impossible variants**: round-robin over families, each family in rank order, until `--infeasible` are chosen or
  the split runs out; so a small family is never drowned and the families stay balanced while each has variants left;
* **feasible variants, paired**: the near-miss control of every chosen impossible variant (same pair and family) and the
  plain feasible twin of its pair, so every impossible task is read against the feasible tasks that look like it;
* if that is still fewer than `--feasible`, plain feasible variants of other pairs, in rank order.

Only a validation split is accepted: a row whose `split` is `test`, or whose pair (source task) is listed in the frozen
`configs/test_ids.txt`, stops the build. Rows are written unchanged and in split order, with `<out>.manifest.json`
next to them (source digest, seed, counts per kind and family, any shortfall). If the split cannot supply the targets
the build stops, unless `--allow-short`, which writes what exists and records the shortfall in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, Any]

MIN_PROBE_INFEASIBLE = 60  # the protocol's minimum per side (docs/training-log.md, evaluation protocol point 4)
MIN_PROBE_FEASIBLE = 60
# The frozen held-out ids of the repository (not of the working directory, so a run from elsewhere still checks them).
DEFAULT_TEST_IDS = Path(__file__).resolve().parents[3] / "configs" / "test_ids.txt"
KINDS = ("feasible", "control", "infeasible")


class ProbeSetError(ValueError):
    """The split cannot be used (a test row, a malformed row) or cannot supply the targets."""


@dataclass
class ProbeSet:
    rows: list[Row]
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeTargets:
    infeasible: int = MIN_PROBE_INFEASIBLE
    feasible: int = MIN_PROBE_FEASIBLE  # plain feasible tasks and near-miss controls together
    seed: int = 0


def _rank(seed: int, variant_id: str) -> str:
    return hashlib.sha256(f"{seed}:{variant_id}".encode()).hexdigest()


def _family(row: Row) -> str:
    return str(row.get("family") or row.get("perturbation") or "unknown")


def check_rows(rows: Sequence[Row], test_ids: set[str] | None = None) -> None:
    """Refuse anything that is not a usable validation split: test rows, test source tasks, missing labels, duplicates."""
    seen: set[str] = set()
    for i, r in enumerate(rows):
        missing = [k for k in ("variant_id", "pair_id", "kind") if not r.get(k)]
        if missing:
            raise ProbeSetError(f"row {i} has no {', '.join(missing)}")
        if r["kind"] not in KINDS:
            raise ProbeSetError(f"row {i} ({r['variant_id']}): unknown kind {r['kind']!r}")
        if r.get("split") == "test":
            raise ProbeSetError(f"row {i} ({r['variant_id']}) belongs to the test split: build probes from val only")
        if test_ids and r["pair_id"] in test_ids:
            raise ProbeSetError(f"row {i} ({r['variant_id']}): source task {r['pair_id']} is a held-out test id")
        if r["variant_id"] in seen:
            raise ProbeSetError(f"variant {r['variant_id']} appears twice")
        seen.add(r["variant_id"])


def select(rows: Sequence[Row], targets: ProbeTargets | None = None) -> list[Row]:
    """The probe rows (module docstring), in split order. Never fails on a shortfall; `build` checks the counts."""
    targets = targets or ProbeTargets()
    by_family: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r["kind"] == "infeasible":
            by_family[_family(r)].append(r)
    seed = targets.seed
    # each family's variants, highest rank first, so `pop()` takes the lowest rank
    queues = [sorted(by_family[f], key=lambda r: _rank(seed, r["variant_id"]), reverse=True) for f in sorted(by_family)]
    chosen: set[str] = set()
    while queues and len(chosen) < targets.infeasible:
        for q in list(queues):
            if len(chosen) >= targets.infeasible:
                break
            chosen.add(q.pop()["variant_id"])
            if not q:
                queues.remove(q)
    twins = {(r["pair_id"], _family(r)) for r in rows if r["variant_id"] in chosen}
    pairs = {p for p, _ in twins}
    for r in rows:
        if (r["kind"] == "control" and (r["pair_id"], _family(r)) in twins) or (
            r["kind"] == "feasible" and r["pair_id"] in pairs
        ):
            chosen.add(r["variant_id"])
    n_feasible = sum(r["kind"] != "infeasible" and r["variant_id"] in chosen for r in rows)
    spare = sorted(
        (r for r in rows if r["kind"] == "feasible" and r["variant_id"] not in chosen),
        key=lambda r: _rank(seed, r["variant_id"]),
    )
    for r in spare[: max(0, targets.feasible - n_feasible)]:
        chosen.add(r["variant_id"])
    return [r for r in rows if r["variant_id"] in chosen]


def summarize(rows: Sequence[Row]) -> dict[str, Any]:
    kinds = Counter(r["kind"] for r in rows)
    by_family = {
        k: dict(sorted(Counter(_family(r) for r in rows if r["kind"] == k).items())) for k in ("infeasible", "control")
    }
    infeasible_pairs = {r["pair_id"] for r in rows if r["kind"] == "infeasible"}
    return {
        "rows": len(rows),
        "by_kind": {k: kinds.get(k, 0) for k in KINDS},
        "n_infeasible": kinds.get("infeasible", 0),
        "n_feasible": kinds.get("feasible", 0) + kinds.get("control", 0),
        "infeasible_by_family": by_family["infeasible"],
        "control_by_family": by_family["control"],
        "pairs": len({r["pair_id"] for r in rows}),
        "pairs_with_impossible_twin": len(infeasible_pairs),
        "impossible_with_control": sum(
            any(c["kind"] == "control" and c["pair_id"] == r["pair_id"] and _family(c) == _family(r) for c in rows)
            for r in rows
            if r["kind"] == "infeasible"
        ),
        "impossible_with_plain_twin": sum(
            any(c["kind"] == "feasible" and c["pair_id"] == r["pair_id"] for c in rows)
            for r in rows
            if r["kind"] == "infeasible"
        ),
    }


def build(
    rows: Sequence[Row],
    targets: ProbeTargets | None = None,
    *,
    test_ids: set[str] | None = None,
    allow_short: bool = False,
    source: dict[str, Any] | None = None,
) -> ProbeSet:
    """Check the split, select the probe and describe it. Raises `ProbeSetError` on a test row, or on a shortfall
    unless `allow_short`."""
    targets = targets or ProbeTargets()
    check_rows(rows, test_ids)
    picked = select(rows, targets)
    counts = summarize(picked)
    available = summarize(rows)
    shortfall = {
        k: {"target": t, "selected": counts[f"n_{k}"], "available": available[f"n_{k}"]}
        for k, t in (("infeasible", targets.infeasible), ("feasible", targets.feasible))
        if counts[f"n_{k}"] < t
    }
    if shortfall and not allow_short:
        raise ProbeSetError(
            "the split cannot supply the probe targets: "
            + "; ".join(f"{k}: {v['available']} available, {v['target']} wanted" for k, v in shortfall.items())
            + " (validate more variants of the val source tasks or rebuild with a larger val split; --allow-short "
            "writes what exists and records the shortfall)"
        )
    ids = sorted(r["variant_id"] for r in picked)
    manifest = {
        "targets": asdict(targets),
        "below_protocol_minimum": targets.infeasible < MIN_PROBE_INFEASIBLE or targets.feasible < MIN_PROBE_FEASIBLE,
        "counts": counts,
        "available": available,
        "shortfall": shortfall,
        "variant_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "test_ids_checked": test_ids is not None,
        **(source or {}),
    }
    return ProbeSet(rows=list(picked), manifest=manifest)


def read_rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.eval.probe_set",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--split", type=Path, required=True, help="the val split JSONL of a dataset build (never test)")
    ap.add_argument("--out", type=Path, required=True, help="probe JSONL to write; the manifest goes next to it")
    ap.add_argument("--infeasible", type=int, default=MIN_PROBE_INFEASIBLE, help="impossible variants to select")
    ap.add_argument("--feasible", type=int, default=MIN_PROBE_FEASIBLE, help="feasible variants incl. controls (min)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--test-ids",
        type=Path,
        default=None,
        help="frozen test_ids.txt; a split row of a listed source task stops the build (default: the repository's "
        "configs/test_ids.txt; a path given here must exist)",
    )
    ap.add_argument("--allow-short", action="store_true", help="write what exists when the split is too small")
    args = ap.parse_args(argv)
    if args.out.resolve() == args.split.resolve():
        raise SystemExit("--out would overwrite --split")
    from honest_agents.tasks.build.splits import read_test_ids

    # An explicit file that is missing is an error, not a silent skip: the check is what keeps test tasks out.
    if args.test_ids is not None and not args.test_ids.is_file():
        raise SystemExit(f"probe_set: --test-ids {args.test_ids} does not exist")
    ids_file = args.test_ids or (DEFAULT_TEST_IDS if DEFAULT_TEST_IDS.is_file() else None)
    if ids_file is None:
        print(f"WARNING: {DEFAULT_TEST_IDS} not found: only rows marked split=test are refused")
    test_ids = read_test_ids(ids_file) if ids_file is not None else None
    raw = args.split.read_bytes()
    source = {
        "source": str(args.split),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "test_ids_file": str(ids_file) if ids_file is not None else None,
    }
    try:
        probe = build(
            read_rows(args.split),
            ProbeTargets(args.infeasible, args.feasible, args.seed),
            test_ids=test_ids,
            allow_short=args.allow_short,
            source=source,
        )
    except ProbeSetError as e:
        raise SystemExit(f"probe_set: {e}") from e
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in probe.rows))
    manifest_path = args.out.with_name(args.out.name + ".manifest.json")
    manifest_path.write_text(json.dumps(probe.manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "manifest": str(manifest_path), **probe.manifest["counts"]}, indent=2))
    if probe.manifest["shortfall"]:
        print(f"WARNING: short of the targets: {json.dumps(probe.manifest['shortfall'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
