"""RL training pool: gate the FEASIBLE side by difficulty, keep a balanced sample of the impossible variants.

    python -m honest_agents.tasks.rl_pool --pool data/train_pool.jsonl --difficulty data/difficulty_2_3.json \\
        --out data/train_rl.jsonl [--max-infeasible 100] [--seed 0]

`tasks.build --difficulty-json` drops a source task's impossible variants together with its plain copy when the plain copy is
too easy for the base model. An impossible variant has no pass rate, though: claiming it done versus declining is a real
contrast whatever the plain twin does (the base model claims about half of them), so dropping them leaves too few examples
to learn abstention from (32 rows on the measured pool). This builder gates only the feasible side:

* plain feasible rows and near-miss controls of the source tasks the difficulty gate kept (`keep_pairs`);
* a sample of `max_infeasible` impossible rows, round-robin over families so a small family is not drowned, with the
  near-miss control of every selected impossible variant added, so over-refusal stays penalised.

Rows are written unchanged and in pool order.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def build_pool(
    rows: Sequence[Row], keep_pairs: set[str], *, max_infeasible: int | None = 100, seed: int = 0
) -> list[Row]:
    """The RL pool from `train_pool` rows and the difficulty gate's kept source tasks (see the module docstring)."""
    rng = random.Random(seed)
    by_family: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r["kind"] == "infeasible":
            by_family[r["family"]].append(r)
    for family_rows in by_family.values():
        rng.shuffle(family_rows)
    chosen: list[Row] = []
    queues = [by_family[f] for f in sorted(by_family)]
    while queues and (max_infeasible is None or len(chosen) < max_infeasible):
        for q in list(queues):
            if max_infeasible is not None and len(chosen) >= max_infeasible:
                break
            chosen.append(q.pop())
            if not q:
                queues.remove(q)
    chosen_twins = {(r["pair_id"], r["family"]) for r in chosen}
    chosen_ids = {r["variant_id"] for r in chosen}
    out = []
    for r in rows:
        if r["kind"] == "infeasible":
            keep = r["variant_id"] in chosen_ids
        elif r["kind"] == "control":
            keep = r["pair_id"] in keep_pairs or (r["pair_id"], r["family"]) in chosen_twins
        else:  # plain feasible
            keep = r["pair_id"] in keep_pairs
        if keep:
            out.append(r)
    return out


def summarize(rows: Sequence[Row]) -> dict[str, Any]:
    kinds: dict[str, int] = defaultdict(int)
    families: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
        if r["kind"] == "infeasible":
            families[r["family"]] += 1
    return {"rows": len(rows), "by_kind": dict(kinds), "infeasible_by_family": dict(families)}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="train_pool.jsonl of a dataset build")
    ap.add_argument("--difficulty", type=Path, required=True, help="difficulty.json of tasks.difficulty")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-infeasible", type=int, default=100, help="impossible rows to keep (round-robin over families)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in args.pool.read_text().splitlines() if line.strip()]
    keep_pairs = set(json.loads(args.difficulty.read_text())["keep_pairs"])
    pool = build_pool(rows, keep_pairs, max_infeasible=args.max_infeasible, seed=args.seed)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in pool))
    print(json.dumps(summarize(pool)))


if __name__ == "__main__":
    main()
