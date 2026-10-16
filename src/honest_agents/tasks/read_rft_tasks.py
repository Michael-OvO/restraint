"""The sampling task list of iteration 15 G4 (docs/iteration-15-g4-read-rft.md): training-surface twin / control pairs.

    python -m honest_agents.tasks.read_rft_tasks build  [--pairs-per-surface 50] [--out data/decision/read_rft_tasks.jsonl]
    python -m honest_agents.tasks.read_rft_tasks verify [--out data/decision/read_rft_tasks.jsonl]

`build` picks, per training surface (readme, file_header, config_comment, manifest), the same number of sources, in a
stable hash order, whose twin AND control variant both exist under `--tasks-root`, and writes one split row per variant
(`run_eval --split` takes it: JSONL with a `task_dir` column). Sources must be in the RL pools (`data/train_rl.jsonl`,
`data/train_rl_v2.jsonl`) or the distillation pool (`data/distill_pool.jsonl`); a source of `data/select.jsonl`,
`data/val.jsonl`, `data/test.jsonl`, `data/probe_v1.jsonl` or of a task under `--val-root` (the held-out surface set)
is refused (asserted again after the selection, on the written rows). `task_dir` is the node path (`--node-root` +
`data/decision/surfaces/tasks/<variant>`), as in every other split file; the rows carry `split: train` so the trajectory
bank rows of the sampling run pass the SFT row filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from honest_agents.tasks.perturb.risk_surfaces import SURFACES_TRAIN

DEFAULT_TASKS_ROOT = Path("data/decision/surfaces/tasks")
DEFAULT_VAL_ROOT = Path("data/decision/surfaces_val/tasks")
DEFAULT_OUT = Path("data/decision/read_rft_tasks.jsonl")
SOURCE_POOLS = ("data/train_rl.jsonl", "data/train_rl_v2.jsonl", "data/distill_pool.jsonl")
FORBIDDEN_FILES = ("data/select.jsonl", "data/val.jsonl", "data/test.jsonl", "data/probe_v1.jsonl")
NODE_ROOT = "/root/restraint"
SEED = "it15-g4"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def pair_ids(paths: Iterable[str | Path]) -> set[str]:
    return {str(r["pair_id"]) for p in paths for r in read_jsonl(p) if r.get("pair_id")}


def variant_ids(paths: Iterable[str | Path]) -> set[str]:
    return {str(r["variant_id"]) for p in paths for r in read_jsonl(p) if r.get("variant_id")}


def _toml(task_dir: Path) -> dict[str, Any]:
    return tomllib.loads((task_dir / "task.toml").read_text())


def dir_pair_ids(root: Path) -> set[str]:
    """Source (`pair_id`) of every task directory under `root` (empty when `root` does not exist)."""
    if not root.is_dir():
        return set()
    out: set[str] = set()
    for d in root.iterdir():
        if (d / "task.toml").is_file():
            out.add(str(_toml(d)["metadata"]["honest"]["pair_id"]))
    return out


def surface_variants(
    tasks_root: Path, surfaces: Sequence[str] = SURFACES_TRAIN
) -> dict[str, dict[str, dict[str, Any]]]:
    """{surface: {pair_id: {"twin": row, "control": row}}} for the high_risk variants of the training surfaces."""
    out: dict[str, dict[str, dict[str, Any]]] = {s: {} for s in surfaces}
    for d in sorted(tasks_root.iterdir()):
        if not (d / "task.toml").is_file():
            continue
        cfg = _toml(d)
        honest = cfg["metadata"]["honest"]
        detail = cfg["metadata"].get("honest_variant") or {}
        surface = (honest.get("detail") or {}).get("surface") or detail.get("surface")
        if surface not in out or honest.get("family") != "high_risk":
            continue
        side = "control" if honest.get("is_control") else "twin"
        out[surface].setdefault(str(honest["pair_id"]), {})[side] = {
            "variant_id": d.name,
            "pair_id": str(honest["pair_id"]),
            "feasible": bool(honest["feasible"]),
            "is_control": bool(honest.get("is_control")),
            "kind": "control" if honest.get("is_control") else "infeasible",
            "family": "high_risk",
            "perturbation": "high_risk",
            "source": str(cfg["metadata"].get("provenance", {}).get("source") or ""),
            "split": "train",
            "detail": {"surface": surface},
        }
    return out


def _rank(seed: str, surface: str, pair: str) -> str:
    return hashlib.sha256(f"{seed}|{surface}|{pair}".encode()).hexdigest()


def build_rows(
    tasks_root: Path,
    *,
    pools: Sequence[str | Path] = SOURCE_POOLS,
    forbidden: Sequence[str | Path] = FORBIDDEN_FILES,
    val_root: Path = DEFAULT_VAL_ROOT,
    pairs_per_surface: int = 50,
    seed: str = SEED,
    node_root: str = NODE_ROOT,
    surfaces: Sequence[str] = SURFACES_TRAIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    allowed = pair_ids(pools)
    banned = pair_ids(forbidden) | dir_pair_ids(val_root)
    banned_variants = variant_ids(forbidden)
    by_surface = surface_variants(tasks_root, surfaces)
    rows: list[dict[str, Any]] = []
    info: dict[str, Any] = {"pairs": {}, "skipped": Counter()}
    for surface in surfaces:
        eligible = []
        for pair, sides in by_surface[surface].items():
            if set(sides) != {"twin", "control"}:
                info["skipped"]["missing_side"] += 1
            elif pair in banned or any(s["variant_id"] in banned_variants for s in sides.values()):
                info["skipped"]["select_val_test_probe_source"] += 1
            elif pair not in allowed:
                info["skipped"]["source_not_in_pools"] += 1
            else:
                eligible.append(pair)
        chosen = sorted(eligible, key=lambda p: _rank(seed, surface, p))[:pairs_per_surface]
        if len(chosen) < pairs_per_surface:
            raise ValueError(f"{surface}: {len(chosen)} eligible pairs, {pairs_per_surface} requested")
        info["pairs"][surface] = len(chosen)
        for pair in sorted(chosen):
            for side in ("twin", "control"):
                row = dict(by_surface[surface][pair][side])
                row["task_dir"] = f"{node_root}/{tasks_root.as_posix()}/{row['variant_id']}"
                rows.append(row)
    info["skipped"] = dict(info["skipped"])
    assert_clean(rows, forbidden=forbidden, val_root=val_root, pools=pools)
    return rows, info


def assert_clean(
    rows: Sequence[dict[str, Any]],
    *,
    forbidden: Sequence[str | Path] = FORBIDDEN_FILES,
    val_root: Path = DEFAULT_VAL_ROOT,
    pools: Sequence[str | Path] = SOURCE_POOLS,
) -> None:
    """Raises when a row's source is outside the pools, or in select / val / test / probe_v1 / the held-out surface set."""
    pairs = {r["pair_id"] for r in rows}
    variants = {r["variant_id"] for r in rows}
    for name, bad in (
        ("select / val / test / probe_v1 source", pairs & pair_ids(forbidden)),
        ("select / val / test / probe_v1 variant", variants & variant_ids(forbidden)),
        ("held-out surface (surfaces_val) source", pairs & dir_pair_ids(val_root)),
        ("source outside the RL and distillation pools", pairs - pair_ids(pools)),
    ):
        if bad:
            raise AssertionError(f"{len(bad)} {name}(s) in the task list, e.g. {min(bad)}")
    surfaces = {(r.get("detail") or {}).get("surface") for r in rows}
    if not surfaces <= set(SURFACES_TRAIN):
        raise AssertionError(f"surfaces {sorted(surfaces - set(SURFACES_TRAIN), key=str)} are not training surfaces")
    if len(variants) != len(rows):
        raise AssertionError("duplicate variant in the task list")


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(((r.get("detail") or {}).get("surface"), "control" if r["is_control"] else "twin") for r in rows)
    return {"tasks": len(rows), "by_surface_side": {f"{s}/{k}": n for (s, k), n in sorted(c.items(), key=str)}}


def write_rows(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.tasks.read_rft_tasks", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("command", choices=("build", "verify"))
    ap.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    ap.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--pairs-per-surface", type=int, default=50, help="twin/control pairs per surface (50 -> 400 tasks)"
    )
    ap.add_argument("--seed", default=SEED)
    ap.add_argument("--pools", nargs="+", default=list(SOURCE_POOLS), help="sources must be in one of these pool files")
    ap.add_argument("--forbidden", nargs="+", default=list(FORBIDDEN_FILES), help="sources of these files are refused")
    ap.add_argument("--node-root", default=NODE_ROOT, help="repository root on the node, prefix of `task_dir`")
    args = ap.parse_args(argv)
    if args.command == "build":
        rows, info = build_rows(
            args.tasks_root, pools=args.pools, forbidden=args.forbidden, val_root=args.val_root,
            pairs_per_surface=args.pairs_per_surface, seed=args.seed, node_root=args.node_root,
        )  # fmt: skip
        write_rows(rows, args.out)
        print(json.dumps({**summarize(rows), **info, "out": str(args.out)}, indent=2))
    else:
        rows = read_jsonl(args.out)
        assert_clean(rows, forbidden=args.forbidden, val_root=args.val_root, pools=args.pools)
        print(json.dumps({**summarize(rows), "verdict": "PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
