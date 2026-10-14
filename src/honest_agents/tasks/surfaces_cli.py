"""CLI of the `high_risk` evidence-surface pools and builder (`perturb.risk_surfaces`).

    uv run python -m honest_agents.tasks.surfaces_cli check [--surface S ...] [--strict]
    uv run python -m honest_agents.tasks.surfaces_cli auc [--surface S ...] [--json out.json]
    uv run python -m honest_agents.tasks.surfaces_cli schema
    uv run python -m honest_agents.tasks.surfaces_cli build --bank data/tasks --out out/surfaces \\
        --surface file_header manifest --mode same_site --ids-file ids.txt

`check` validates every pool (schema, shape match, LABEL_RE scan, placeholders) and exits 1 when anything is dropped with
`--strict`. `auc` prints, per surface, the held-out AUC of bag-of-words and style classifiers over the evidence text, the
single-token rules and the pairs a non-semantic token separates (rewrite candidates). `build` writes twin/control task
dirs under `--out/tasks` and one row per variant to `--out/surface_variants.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks.perturb.core import Variant
from honest_agents.tasks.perturb.risk_surfaces import (
    SURFACES,
    PoolError,
    auc_report,
    bank_base,
    build_same_site,
    build_varied,
    get_pool,
    load_pool,
    pool_dir,
    pool_json_schema,
    shape_stats,
)


def _surfaces(names: Sequence[str] | None) -> list[str]:
    return list(names) if names else list(SURFACES)


def cmd_check(args: argparse.Namespace) -> int:
    bad = 0
    for s in _surfaces(args.surface):
        try:
            pool = load_pool(s, directory=args.pools)
        except PoolError as e:
            print(f"{s}: ERROR {e}")
            bad += 1
            continue
        stats = shape_stats(pool)
        print(
            f"{s}: {len(pool.pairs)} pairs ({pool.n_hard_negatives} hard negatives, "
            f"{len(pool.pairs_for('test'))} held out), {len(pool.phrasings)} phrasings, {len(pool.contexts)} contexts"
            f"{f', {len(pool.backup_notes)} backup notes' if pool.backup_notes else ''}; "
            f"length ratio danger/benign {stats['min_ratio']}..{stats['max_ratio']}"
        )
        for ident, why in pool.dropped:
            print(f"  DROPPED {ident}: {why}")
            bad += 1
    return 1 if bad and args.strict else 0


def cmd_auc(args: argparse.Namespace) -> int:
    reports = []
    for s in _surfaces(args.surface):
        r = auc_report(load_pool(s, directory=args.pools), min_accuracy=args.min_accuracy)
        reports.append(r)
        print(f"== {s}: {r['n_pairs']} pairs, {r['n_hard_negatives']} hard negatives")
        for name, e in r["classifiers"].items():
            print(f"  {name:22s} auc {e['auc']:.2f} (se {e['auc_se']:.2f}) acc {e['accuracy']:.2f}")
        for rule in r["single_token_rules"][: args.top]:
            tag = "NON-SEMANTIC" if rule["non_semantic"] else "semantic"
            print(
                f"  token {rule['token']!r:14} -> {rule['polarity']:6s} acc {rule['accuracy']:.2f} "
                f"({rule['in_danger']} danger / {rule['in_benign']} benign) {tag}"
            )
        if r["flagged_pairs"]:
            print(f"  FLAGGED (separable by a non-semantic token): {r['flagged_pairs']}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1))
    return 0


def cmd_schema(_: argparse.Namespace) -> int:
    print(json.dumps(pool_json_schema(), indent=1))
    return 0


def _ids(args: argparse.Namespace) -> list[str]:
    if args.ids:
        return list(args.ids)
    if args.ids_file:
        return [ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    twins = sorted(
        p.name for p in args.bank.glob("*__high_risk") if (p / "task.toml").is_file() and "__high_risk__" not in p.name
    )
    return twins[args.offset :: args.stride][: args.limit] if args.limit else twins[args.offset :: args.stride]


def cmd_build(args: argparse.Namespace) -> int:
    tasks = args.out / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    n_skipped = 0
    for ident in _ids(args):
        for s in _surfaces(args.surface):
            pool = get_pool(s) if args.pools is None else load_pool(s, directory=args.pools)
            try:
                if args.mode == "same_site":
                    built = build_same_site(
                        args.bank,
                        tasks,
                        ident,
                        s,
                        pool=pool,
                        seed=args.seed,
                        split=args.split,
                        pool_pair=args.pool_pair,
                    )
                else:
                    src = args.bank / (bank_base(ident) + "__feasible")
                    built = build_varied(
                        src,
                        tasks,
                        pair_id=HonestTaskSpec.from_task_dir(src).pair_id or ident,
                        surface=s,
                        pool=pool,
                        seed=args.seed,
                        split=args.split,
                        pool_pair=args.pool_pair,
                    )
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"skip {ident} {s}: {e}", file=sys.stderr)
                built = None
            if built is None:
                n_skipped += 1
                continue
            for v in built:
                rows.append(_row(v, ident))
    with (args.out / "surface_variants.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"built {len(rows)} variants ({n_skipped} source/surface combinations skipped) under {tasks}")
    return 0


def _row(v: Variant, bank_id: str) -> dict[str, object]:
    d = v.detail
    return {
        "variant_id": v.variant_id,
        "task_dir": str(v.task_dir),
        "pair_id": v.spec.pair_id,
        "bank_id": bank_id,
        "surface": d["surface"],
        "control": v.is_control,
        "pool_pair": d["pool_pair"],
        "hard_negative": d["hard_negative"],
        "mode": d["build_mode"],
        "path": d["path"],
        "detail": dict(v.spec.detail),  # surface, phrasing_heldout: what EpisodeRecord.detail is filled from
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", type=Path, default=None, help=f"pool directory (default {pool_dir()})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate and scan the pools")
    c.add_argument("--surface", nargs="+", choices=SURFACES)
    c.add_argument("--strict", action="store_true", help="exit 1 when any pair or sentence is dropped")
    c.set_defaults(fn=cmd_check)

    a = sub.add_parser("auc", help="per-surface separability of the evidence text")
    a.add_argument("--surface", nargs="+", choices=SURFACES)
    a.add_argument("--min-accuracy", type=float, default=0.75)
    a.add_argument("--top", type=int, default=8, help="single-token rules to print per surface")
    a.add_argument("--json", type=Path)
    a.set_defaults(fn=cmd_auc)

    s = sub.add_parser("schema", help="print the JSON schema of the pool files")
    s.set_defaults(fn=cmd_schema)

    b = sub.add_parser("build", help="build twin/control pairs for surfaces")
    b.add_argument("--bank", type=Path, required=True, help="dir of the bank's task dirs (data/tasks)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--surface", nargs="+", choices=SURFACES)
    b.add_argument("--mode", choices=("same_site", "varied"), default="same_site")
    b.add_argument("--split", choices=("train", "test"), default="train", help="which pool pairs and phrasings")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--ids", nargs="+", help="bank variant ids, source ids or slugs")
    b.add_argument("--ids-file", type=Path)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--offset", type=int, default=0)
    b.add_argument("--stride", type=int, default=1)
    b.add_argument("--pool-pair", help="force one pool pair id (default: drawn from the source id)")
    b.set_defaults(fn=cmd_build)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
