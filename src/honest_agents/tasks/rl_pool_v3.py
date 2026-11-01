"""RL pool v3 (iteration 15 G3, docs/iteration-15-g3-rl-read.md): surface pairs mixed into pool-v2 step blocks.

    python -m honest_agents.tasks.rl_pool_v3 --config configs/tasks/rl_pool_v3.yaml [key=value ...]

Each of `steps` blocks (one optimizer step) holds `surface_pairs_per_step` surface twin + control pairs (built tasks in
`surfaces_dir`, one surface each, spread evenly), `minor_pairs_per_step` revoke_permission / remove_input pairs and
`plain_per_step` plain rows, the last two from pool v2. One unit per source per step; here every source is used once in
the whole pool. Rows have exactly the pool-v2 fields: `prompt` is the pool-v2 prompt (system prompt + empty user
turn: the environment reads `instruction.md` from `task_dir` at reset, as for every pool-v2 row) and `task_dir` is the
node path. The surface comes from the task's `task.toml` at reset, so no extra column is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.tasks.rl_pool import Row, check_paired_blocks, read_jsonl

CONTROL_SUFFIX = "__control"


@dataclass
class V3Options:
    v2_pool: str = "data/train_rl_v2.jsonl"
    rl_pools: list[str] = field(default_factory=lambda: ["data/train_rl.jsonl", "data/train_rl_v2.jsonl"])
    surfaces_dir: str = "data/decision/surfaces/tasks"
    node_root: str = "/root/restraint"
    out: str = "data/train_rl_v3.jsonl"
    report: str | None = None
    seed: int = 42
    steps: int = 15
    surfaces: list[str] = field(default_factory=lambda: ["readme", "file_header", "config_comment", "manifest"])
    surface_pairs_per_step: int = 2
    minor_pairs_per_step: int = 1
    minor_families: list[str] = field(default_factory=lambda: ["revoke_permission", "remove_input"])
    plain_per_step: int = 2
    exclude_pools: list[str] = field(default_factory=list)
    # Pool v4 (iteration 19 CR-GRPO): leave out every source reviewed as a spec defect (DEF-66, `tasks.spec_audit`), and
    # draw the plain rows by their pass count in a G2 bank (`plain_bank`, run_eval episodes): variants inside
    # `plain_pass_band` (passes out of the bank's samples) first, then unmeasured, then above the band; never below.
    exclude_spec_defects: bool = False
    plain_bank: str | None = None
    plain_pass_band: list[int] = field(default_factory=lambda: [1, 3])

    @property
    def block_size(self) -> int:
        return 2 * (self.surface_pairs_per_step + self.minor_pairs_per_step) + self.plain_per_step

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> V3Options:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown option(s): {sorted(unknown)}")
        return cls(**d)


HELD_OUT_SURFACES = ("empty_backup", "inbound_link")


def _source_key(pair_id: str) -> str:
    """pair_id `endless-terminals/task_000000_x` -> the task-directory stem `endless-terminals__task_000000_x`."""
    return pair_id.replace("/", "__", 1)


def surface_row(
    task_name: str, surfaces_dir: Path, node_root: str, template: Row, *, control: bool, surface: str
) -> Row:
    """A pool row for one built surface task, with the fields read back from its task.toml."""
    toml_path = surfaces_dir / task_name / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"surface task missing: {toml_path}")
    meta = tomllib.loads(toml_path.read_text())
    honest = meta["metadata"]["honest"]
    if honest["variant_id"] != task_name:
        raise ValueError(f"{toml_path}: variant_id {honest['variant_id']!r} != directory {task_name!r}")
    if honest["detail"]["surface"] != surface:
        raise ValueError(f"{toml_path}: surface {honest['detail']['surface']!r}, expected {surface!r}")
    if bool(honest["is_control"]) != control or bool(honest["feasible"]) != control:
        raise ValueError(f"{toml_path}: is_control/feasible do not match a {'control' if control else 'twin'}")
    node_dir = f"{node_root.rstrip('/')}/{surfaces_dir.as_posix()}/{task_name}"
    return {
        "prompt": template["prompt"],
        "task_dir": node_dir,
        "variant_id": task_name,
        "pair_id": honest["pair_id"],
        "feasible": control,
        "is_control": control,
        "kind": "control" if control else "infeasible",
        "family": honest["family"],
        "perturbation": honest["perturbation"],
        "source": meta["metadata"]["provenance"]["source"],
        "split": "train",
    }


def surface_sources(surfaces_dir: Path, surface: str, allowed: set[str]) -> list[str]:
    """pair_ids with a built twin and control of `surface`, restricted to `allowed`, sorted."""
    out = []
    for p in sorted(surfaces_dir.iterdir()):
        suffix = f"__high_risk__{surface}"
        if p.is_dir() and p.name.endswith(suffix) and (surfaces_dir / (p.name + CONTROL_SUFFIX)).is_dir():
            pair = p.name[: -len(suffix)].replace("__", "/", 1)
            if pair in allowed:
                out.append(pair)
    return out


def excluded_sources(opts: V3Options) -> set[str]:
    """Sources the pool must not use: every source of `exclude_pools`, and with `exclude_spec_defects` every source
    reviewed as a spec defect."""
    out = {r["pair_id"] for p in opts.exclude_pools for r in read_jsonl(p)}
    if opts.exclude_spec_defects:
        from honest_agents.tasks.spec_audit import defective_sources

        out |= defective_sources()
    return out


def bank_pass_counts(path: str) -> dict[str, tuple[int, int]]:
    """variant id -> (clean test passes, episodes) in a run_eval episodes file (.jsonl or .jsonl.gz)."""
    import gzip

    path = os.path.expanduser(path)
    opener = gzip.open if path.endswith(".gz") else open
    out: dict[str, list[int]] = {}
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                c = out.setdefault(str(r["task_id"]).split("/")[-1], [0, 0])
                c[0] += bool((r.get("outcome") or {}).get("tests_passed"))
                c[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def plains_by_pass_band(plains: Sequence[str], counts: dict[str, tuple[int, int]], band: Sequence[int]) -> list[str]:
    """The plain variants in draw order: those whose bank pass count lies in `band` (inclusive) first, then the
    unmeasured ones, then those above the band (they still carry the report contrast and the success anchor); one below
    the band (never passing: suspect tests, no contrast) is dropped. Order within each group is kept (the caller's
    shuffle)."""
    lo, hi = band
    mixed = [v for v in plains if v in counts and lo <= counts[v][0] <= hi]
    unmeasured = [v for v in plains if v not in counts]
    above = [v for v in plains if v in counts and counts[v][0] > hi]
    return mixed + unmeasured + above


def build(opts: V3Options) -> tuple[list[Row], dict[str, Any]]:
    bad = [s for s in opts.surfaces if s in HELD_OUT_SURFACES]
    if bad:
        raise ValueError(f"held-out surface(s) in the pool: {bad}")
    rng = random.Random(opts.seed)
    v2 = read_jsonl(opts.v2_pool)
    rl_sources = {r["pair_id"] for p in opts.rl_pools for r in read_jsonl(p)}
    excluded = excluded_sources(opts)
    allowed = rl_sources - excluded
    surfaces_dir = Path(opts.surfaces_dir)
    template = v2[0]
    if any(r["prompt"] != template["prompt"] for r in v2):
        raise ValueError("pool v2 rows carry different prompts; cannot reuse one")
    v2 = [r for r in v2 if r["pair_id"] not in excluded]

    # Surface slots: each surface gets floor/ceil(slots / n_surfaces) of them; a step takes the surfaces with the most
    # slots left (random tie-break), all different within a step, so the pairings vary from step to step.
    n_slots = opts.steps * opts.surface_pairs_per_step
    if opts.surface_pairs_per_step > len(opts.surfaces):
        raise ValueError("more surface pairs per step than surfaces")
    left = {s: n_slots // len(opts.surfaces) + (i < n_slots % len(opts.surfaces)) for i, s in enumerate(opts.surfaces)}
    slot_surface: list[str] = []
    for _ in range(opts.steps):
        order = sorted(left, key=lambda s: (-left[s], rng.random()))[: opts.surface_pairs_per_step]
        for s in order:
            left[s] -= 1
        rng.shuffle(order)
        slot_surface += order
    fam_cycle = [
        opts.minor_families[i % len(opts.minor_families)] for i in range(opts.steps * opts.minor_pairs_per_step)
    ]
    rng.shuffle(fam_cycle)
    pools = {s: surface_sources(surfaces_dir, s, allowed) for s in opts.surfaces}
    for lst in pools.values():
        rng.shuffle(lst)
    used: set[str] = set()
    slot_source: list[str] = []
    for s in slot_surface:
        pick = next((p for p in pools[s] if p not in used), None)
        if pick is None:
            raise ValueError(f"not enough unused sources for surface {s}")
        used.add(pick)
        slot_source.append(pick)

    twins = {r["pair_id"]: r for r in v2 if r["kind"] == "infeasible" and r["family"] in opts.minor_families}
    ctrls = {(r["pair_id"], r["family"]): r for r in v2 if r["kind"] == "control"}
    minor_by_fam = {f: sorted(p for p, r in twins.items() if r["family"] == f) for f in opts.minor_families}
    for lst in minor_by_fam.values():
        rng.shuffle(lst)
    plains = sorted(r["variant_id"] for r in v2 if r["kind"] == "feasible")
    plain_by_id = {r["variant_id"]: r for r in v2 if r["kind"] == "feasible"}
    rng.shuffle(plains)
    if opts.plain_bank:
        plains = plains_by_pass_band(plains, bank_pass_counts(opts.plain_bank), opts.plain_pass_band)

    rows: list[Row] = []
    used_minor: set[str] = set()
    for k in range(opts.steps):
        units: list[list[Row]] = []
        block_sources: set[str] = set()
        for j in range(opts.surface_pairs_per_step):
            slot = k * opts.surface_pairs_per_step + j
            surf, pair = slot_surface[slot], slot_source[slot]
            stem = f"{_source_key(pair)}__high_risk__{surf}"
            units.append(
                [
                    surface_row(stem, surfaces_dir, opts.node_root, template, control=False, surface=surf),
                    surface_row(
                        stem + CONTROL_SUFFIX, surfaces_dir, opts.node_root, template, control=True, surface=surf
                    ),
                ]
            )
            block_sources.add(pair)
        for j in range(opts.minor_pairs_per_step):
            fam = fam_cycle[k * opts.minor_pairs_per_step + j]
            pair = next((p for p in minor_by_fam[fam] if p not in used_minor and p not in block_sources), None)
            if pair is None or (pair, fam) not in ctrls:
                raise ValueError(f"no unused {fam} pair (with control) for step {k}")
            used_minor.add(pair)
            block_sources.add(pair)
            units.append([twins[pair], ctrls[(pair, fam)]])
        n_plain = 0
        for vid in plains:
            if n_plain == opts.plain_per_step:
                break
            r = plain_by_id[vid]
            if r["pair_id"] in block_sources or r["pair_id"] in used_minor:
                continue
            block_sources.add(r["pair_id"])
            units.append([r])
            n_plain += 1
        if n_plain != opts.plain_per_step:
            raise ValueError(f"no plain rows left for step {k}")
        plains = [v for v in plains if plain_by_id[v]["pair_id"] not in block_sources]
        rng.shuffle(units)
        rows += [r for u in units for r in u]
    return rows, {"allowed_sources": len(allowed), "surface_sources_available": {s: len(p) for s, p in pools.items()}}


def validate(rows: Sequence[Row], opts: V3Options, surfaces_dir: Path | None = None) -> None:
    """Raise if the pool breaks a rule: pairing, exclusions, held-out surfaces, unique rows, task dirs on disk."""
    problems = [p for p in check_paired_blocks(rows, opts.block_size) if p.startswith(("split", "repeat"))]
    if len(rows) != opts.steps * opts.block_size:
        problems.append(f"{len(rows)} rows, expected {opts.steps * opts.block_size}")
    if len({r["variant_id"] for r in rows}) != len(rows):
        problems.append("duplicate variant_id")
    excluded = excluded_sources(opts)
    rl = {r["pair_id"] for p in opts.rl_pools for r in read_jsonl(p)}
    for r in rows:
        if r["pair_id"] in excluded:
            problems.append(f"excluded source in pool: {r['variant_id']}")
        if r["pair_id"] not in rl:
            problems.append(f"source not in the RL pool: {r['variant_id']}")
        if any(f"__{h}" in r["variant_id"] for h in HELD_OUT_SURFACES):
            problems.append(f"held-out surface: {r['variant_id']}")
        if not r["task_dir"].startswith(opts.node_root + "/"):
            problems.append(f"task_dir not a node path: {r['task_dir']}")
        local = Path(r["task_dir"].removeprefix(opts.node_root + "/"))
        if not (local / "task.toml").is_file():
            problems.append(f"task dir missing locally: {local}")
    if problems:
        raise RuntimeError("pool v3 is invalid:\n" + "\n".join(problems[:30]))


def _surface_of(r: Row) -> str | None:
    vid = r["variant_id"].removesuffix(CONTROL_SUFFIX)
    return vid.split("__high_risk__", 1)[1] if "__high_risk__" in vid else None


def summarize(rows: Sequence[Row], block_size: int) -> dict[str, Any]:
    steps = []
    for b in range(len(rows) // block_size):
        block = rows[b * block_size : (b + 1) * block_size]
        steps.append(
            {
                "step": b + 1,
                "surfaces": sorted(s for r in block if r["kind"] == "infeasible" and (s := _surface_of(r))),
                "families": dict(
                    sorted(
                        Counter(f"{r['kind']}:{'surface' if _surface_of(r) else r['family']}" for r in block).items()
                    )
                ),
            }
        )
    pairs = Counter(s for r in rows if r["kind"] == "infeasible" and (s := _surface_of(r)))
    return {
        "rows": len(rows),
        "by_kind": dict(Counter(r["kind"] for r in rows)),
        "pairs_per_surface": dict(sorted(pairs.items())),
        "pairs_per_family": dict(
            Counter(r["family"] for r in rows if r["kind"] == "infeasible" and not _surface_of(r))
        ),
        "sources": len({r["pair_id"] for r in rows}),
        "per_step": steps,
    }


def run(opts: V3Options) -> dict[str, Any]:
    rows, info = build(opts)
    validate(rows, opts)
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {**info, "summary": summarize(rows, opts.block_size), "options": asdict(opts)}
    if opts.report:
        Path(opts.report).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    from honest_agents.train.config import apply_overrides, load_config_dict

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML of V3Options keys (configs/tasks/rl_pool_v3.yaml)")
    ap.add_argument("overrides", nargs="*", help="key=value overrides of config keys (values parsed as YAML)")
    args = ap.parse_args(argv)
    d = apply_overrides(load_config_dict(args.config) if args.config else {}, list(args.overrides))
    try:
        report = run(V3Options.from_dict(d))
    except (ValueError, RuntimeError, OSError) as e:
        print(f"rl_pool_v3: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    s = report["summary"]
    print(json.dumps({k: s[k] for k in ("rows", "by_kind", "pairs_per_surface", "pairs_per_family", "sources")}))
    for st in s["per_step"]:
        print(f"step {st['step']:2d} {st['surfaces']} {st['families']}", file=sys.stderr)


if __name__ == "__main__":
    main()
