"""The training-time probe list: a fixed, honesty-weighted set of validation variants (`data.probe_file`).

    python -m honest_agents.tasks.probe_set --rows data/val_pool.jsonl --out data/probe.jsonl \\
        [--infeasible 64] [--feasible 64] [--quarantine runs/quarantine.jsonl] [--seed 0]

The list lives under `data/` (`data.probe_file: data/probe.jsonl`): `node sync --with-data` sends `data/` to a node, and
`runs/` / `outputs/` are never pushed. The plan records the file's sha256 next to its path (`train --dry-run` prints it).

Why: iteration 9 probed 100 hash-chosen rows of `val.jsonl`, about 21 of them impossible, which cannot resolve an
honesty change under ~10 points. The probe list is built once, committed to the run's config and read whole at every
probe of every run, so probe steps and runs are paired by task. Selection:

* impossible variants round-robin over the families present (or `--families`), each family in a fixed hash order of
  the variant id, until `--infeasible` are chosen; so `high_risk` cannot crowd out the smaller families;
* with each impossible variant, its near-miss control (same pair, same family) and the plain feasible twin of the pair,
  so over-refusal is measured on the tasks built to catch it and every source task carries both sides;
* plain feasible rows of other pairs, in hash order, until `--feasible` feasible rows (plain and controls) are reached.

Rows are written unchanged. Only validation-split rows are accepted (never test); variants on the quarantine list are
dropped with their pair's control. The validation set after the build's mixture balance (`val.jsonl`) has too few
impossible variants for the default 64; build from `val_pool.jsonl` (every validation row before the mixture). Cost:
one probe is (rows x `probe.samples_per_variant`) episodes, so 64 impossible + 64 controls + 64 plain twins at 4
samples each is 768 episodes per probe; see docs/evaluation.md section 8.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Row = dict[str, Any]
VALIDATION_SPLITS = ("validation", "val")  # train.config.VALIDATION_SPLITS, without importing the trainer stack


def _kind(r: Row) -> str:
    if r.get("kind"):
        return str(r["kind"])
    if r.get("feasible") is False:
        return "infeasible"
    return "control" if r.get("is_control") else "feasible"


def _family(r: Row) -> str:
    return str(r.get("family") or r.get("perturbation") or "unknown")


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def select_probe_set(
    rows: Sequence[Row],
    *,
    n_infeasible: int = 64,
    n_feasible: int = 64,
    families: Sequence[str] | None = None,
    exclude: set[str] | frozenset[str] = frozenset(),
    seed: int = 0,
    with_plain_twins: bool = True,
) -> list[Row]:
    """The probe rows (see the module docstring). Raises if the rows hold fewer than `n_infeasible` impossible or
    `n_feasible` feasible variants after the exclusions: a short probe must not pass as the requested one."""
    for i, r in enumerate(rows):
        if r.get("split") not in VALIDATION_SPLITS:
            raise ValueError(f"row {i} has split={r.get('split')!r}; the probe takes validation rows only")
    by_pair: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_pair[str(r.get("pair_id") or r["variant_id"])].append(r)
    wanted = set(families) if families else None
    queues: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if _kind(r) == "infeasible" and r["variant_id"] not in exclude and (wanted is None or _family(r) in wanted):
            queues[_family(r)].append(r)
    for queue in queues.values():
        queue.sort(key=lambda r: _rank(seed, str(r["variant_id"])), reverse=True)  # pop() takes the lowest rank

    chosen: dict[str, Row] = {}

    def add(r: Row) -> None:
        chosen.setdefault(str(r["variant_id"]), r)

    n_chosen_infeasible = 0
    order = sorted(queues)
    while n_chosen_infeasible < n_infeasible and any(queues[f] for f in order):
        for fam in order:
            if n_chosen_infeasible >= n_infeasible or not queues[fam]:
                continue
            v = queues[fam].pop()
            add(v)
            n_chosen_infeasible += 1
            pair = by_pair[str(v.get("pair_id") or v["variant_id"])]
            for r in pair:
                its_control = _kind(r) == "control" and _family(r) == fam
                plain_twin = _kind(r) == "feasible" and with_plain_twins
                if (its_control or plain_twin) and r["variant_id"] not in exclude:
                    add(r)
    if n_chosen_infeasible < n_infeasible:
        raise ValueError(
            f"only {n_chosen_infeasible} impossible variants available (families {order or 'none'}), "
            f"{n_infeasible} requested; build from val_pool.jsonl or lower --infeasible"
        )
    extra = sorted(
        (
            r
            for r in rows
            if _kind(r) == "feasible" and r["variant_id"] not in exclude and r["variant_id"] not in chosen
        ),
        key=lambda r: _rank(seed, str(r["variant_id"])),
    )
    for r in extra:
        if sum(_kind(c) != "infeasible" for c in chosen.values()) >= n_feasible:
            break
        add(r)
    n_feas = sum(_kind(c) != "infeasible" for c in chosen.values())
    if n_feas < n_feasible:
        raise ValueError(f"only {n_feas} feasible variants available, {n_feasible} requested")
    position = {id(r): i for i, r in enumerate(rows)}
    return sorted(chosen.values(), key=lambda r: position[id(r)])  # file order, so the list diffs cleanly


def summarize(rows: Sequence[Row], samples_per_variant: int = 1) -> dict[str, Any]:
    kinds = Counter(_kind(r) for r in rows)
    return {
        "rows": len(rows),
        "pairs": len({r.get("pair_id") for r in rows}),
        "by_kind": dict(kinds),
        "infeasible_by_family": dict(Counter(_family(r) for r in rows if _kind(r) == "infeasible")),
        "control_by_family": dict(Counter(_family(r) for r in rows if _kind(r) == "control")),
        "episodes_per_probe": len(rows) * samples_per_variant,
        "samples_per_variant": samples_per_variant,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, nargs="+", required=True, help="validation rows (val_pool.jsonl of a build)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--infeasible", type=int, default=64, help="impossible variants (round-robin over families)")
    ap.add_argument("--feasible", type=int, default=64, help="minimum feasible variants, plain and controls")
    ap.add_argument("--families", nargs="*", default=None, help="impossible families to draw from (default: all)")
    ap.add_argument("--no-plain-twins", action="store_true", help="do not add the plain twin of each chosen pair")
    ap.add_argument("--quarantine", type=Path, default=None, help="quarantine.jsonl whose leaks are left out")
    ap.add_argument(
        "--samples-per-variant", type=int, default=4, help="for the printed cost only (probe config sets it)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for path in args.rows for line in path.read_text().splitlines() if line.strip()]
    exclude: set[str] = set()
    if args.quarantine is not None:
        from honest_agents.bank import quarantined_variants

        exclude = quarantined_variants(args.quarantine, exclude_sources=())
    chosen = select_probe_set(
        rows,
        n_infeasible=args.infeasible,
        n_feasible=args.feasible,
        families=args.families,
        exclude=exclude,
        seed=args.seed,
        with_plain_twins=not args.no_plain_twins,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in chosen))
    print(json.dumps(summarize(chosen, args.samples_per_variant), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
