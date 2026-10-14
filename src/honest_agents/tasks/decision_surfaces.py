"""Evidence-surface variants for the decision SFT and the decision probe (docs/iteration-14-design.md sections 1-2).

The decision dataset of iteration 13 spliced the base model's own control prefixes with the real outputs of `ls -la <dir>`
and `cat <dir>/README`. Iteration 14 repeats the splice for the new evidence surfaces (`perturb.risk_surfaces`): for a
source of the iteration-13 decision set, `build_same_site` writes a twin and a control with the bank twin's instruction
byte for byte, differing from each other (and from the bank's README variants) in the evidence only, so the base model's
own prefix up to the deletion stays valid. Each variant is booted on the Mac to collect the output of its
`inspection_commands(site)` (2-4 commands), cached by variant id *and task signature* (`decision_probe.task_signature`):
a pair that the pool authors replace changes the Dockerfile, so only that variant is booted again.

    python -m honest_agents.tasks.decision_surfaces build --role sft --bank <pass@4 bank> [--limit-sources 10]
    python -m honest_agents.tasks.decision_surfaces outputs [--parallel 6]      # boots every variant of the manifest
    python -m honest_agents.tasks.decision_surfaces build --role probe          # RL-pool / select sources, never SFT
    python -m honest_agents.tasks.decision_surfaces prompts --bank <bank>       # the six-surface probe prompt set
    python -m honest_agents.tasks.decision_surfaces cleanup

Manifest `data/decision/surface_pairs.jsonl`, one row per (role, source, surface): the pool pair, both variants' task
directories, and the flags the dataset builder asserts on. **SFT rows** (`role: sft`) are built on distill-pool sources
only, never on an RL-pool source or a source of `select_split.excluded_from_training` (select, validation, test), with a
non-held-out pool pair (`Pair.heldout` / `is_heldout`) of a training surface; pool pairs are spread evenly over the
sources by a stable hash (`assign_pool_pairs`). **Probe rows** (`role: probe`) use RL-pool and select sources; a training
surface gets, alternately by source hash, an SFT-used pool pair (seen) or a held-out one (unseen); the held-out surfaces
(`empty_backup`, `inbound_link`) are built on select sources only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks.decision_probe import VariantRef
from honest_agents.tasks.perturb.risk_surfaces import (
    SURFACES,
    SURFACES_HELDOUT,
    SURFACES_TRAIN,
    SurfacePool,
    build_same_site,
    get_pool,
    is_heldout,
)
from honest_agents.tasks.select_split import excluded_from_training, load_select_sources
from honest_agents.tasks.taskfs import slug

logger = logging.getLogger(__name__)

ROLE_SFT = "sft"
ROLE_PROBE = "probe"
SURFACE_DIR = dp.DECISION_DIR / "surfaces"
SURFACE_TASKS = SURFACE_DIR / "tasks"
SURFACE_PAIRS = dp.DECISION_DIR / "surface_pairs.jsonl"
SURFACE_OUTPUTS = dp.DECISION_DIR / "sandbox_outputs_surfaces.jsonl"
SURFACE_PROMPTS = dp.DECISION_DIR / "probe_prompts_surfaces.jsonl"
DEFAULT_SALT = "it14"
DEFAULT_PARALLEL = 6  # the Mac runs at most six boots at once
DEFAULT_PROBE_SOURCES = 24  # per surface


def _rank(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


@dataclass(frozen=True)
class SurfacePairRow:
    role: str
    source: str  # the bank source id (`pair_id` of the pools)
    surface: str
    pool_pair: str
    pool_pair_heldout: bool
    hard_negative: bool
    request_heldout: bool  # the request sentence is a test-pool phrasing
    twin_variant: str
    control_variant: str
    twin_task_dir: str
    control_task_dir: str
    instruction_equal: bool  # twin and control instruction are the bank twin's, byte for byte
    seen: bool | None = None  # probe rows: the pool pair is one the SFT used

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.role, self.source, self.surface)

    def refs(self) -> tuple[VariantRef, VariantRef]:
        """(twin, control); `pair_id` is the source, so a per-surface call of `pair_refs` pairs them."""
        return (
            VariantRef(self.twin_variant, self.source, "infeasible", self.twin_task_dir, "train", self.surface),
            VariantRef(self.control_variant, self.source, "control", self.control_task_dir, "train", self.surface),
        )


def read_manifest(path: str | Path = SURFACE_PAIRS) -> list[SurfacePairRow]:
    p = Path(path).expanduser()
    if not p.exists():
        return []
    return [SurfacePairRow(**json.loads(line)) for line in p.read_text().splitlines() if line.strip()]


def write_manifest(rows: Iterable[SurfacePairRow], path: str | Path = SURFACE_PAIRS) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(r)) + "\n" for r in sorted(rows, key=lambda r: r.key)))
    return p


def merge_manifest(new: Sequence[SurfacePairRow], path: str | Path = SURFACE_PAIRS, *, replace_roles: Sequence[str] = ()) -> list[SurfacePairRow]:
    """The manifest with `new` rows replacing rows of the same key; rows of `replace_roles` that `new` did not rebuild
    are dropped (a full rebuild must not leave a stale row behind)."""
    keep = {r.key: r for r in read_manifest(path) if r.role not in replace_roles}
    keep.update({r.key: r for r in new})
    rows = list(keep.values())
    write_manifest(rows, path)
    return rows


def refs_of(rows: Iterable[SurfacePairRow]) -> list[VariantRef]:
    return [ref for r in rows for ref in r.refs()]


def sft_pool_pairs(rows: Iterable[SurfacePairRow]) -> dict[str, set[str]]:
    """surface -> the pool pair ids the SFT rows use (what a probe row calls "seen")."""
    out: dict[str, set[str]] = {}
    for r in rows:
        if r.role == ROLE_SFT:
            out.setdefault(r.surface, set()).add(r.pool_pair)
    return out


# ── choosing sources and pool pairs ──────────────────────────────────────────


def assign_pool_pairs(sources: Sequence[str], pool: SurfacePool, salt: str = DEFAULT_SALT, *, split: str = "train") -> dict[str, str]:
    """source -> pool pair id: the sources in a stable hash order take the pool's pairs of `split` ("train": never a
    held-out pair) in a stable hash order, round robin, so every pair is used `floor` or `ceil` of `sources / pairs`
    times (an even spread)."""
    pairs = sorted(pool.pairs_for(split), key=lambda p: _rank(salt, pool.surface, "pair", p.id))
    if not pairs:
        raise ValueError(f"the {pool.surface} pool has no pairs for split {split!r}")
    ordered = sorted(sources, key=lambda s: _rank(salt, pool.surface, "source", s))
    return {s: pairs[i % len(pairs)].id for i, s in enumerate(ordered)}


def rl_pool_sources(path: str | Path = "data/train_rl.jsonl") -> set[str]:
    from honest_agents.sft.decision import pool_pair_ids

    return pool_pair_ids([path]) if Path(path).expanduser().is_file() else set()


def assert_sft_sources(sources: Iterable[str], distill: Iterable[str], rl: Iterable[str], data_dir: str | Path = "data") -> None:
    """The SFT sources are distill-pool sources, no RL-pool source and no select / validation / test source."""
    s, d = set(sources), set(distill)
    if outside := s - d:
        raise ValueError(f"{len(outside)} SFT surface source(s) are not distill-pool sources (e.g. {min(outside)})")
    if overlap := s & set(rl):
        raise ValueError(f"{len(overlap)} SFT surface source(s) are RL-pool sources (e.g. {min(overlap)})")
    if bad := s & excluded_from_training(data_dir):
        raise ValueError(f"{len(bad)} SFT surface source(s) are select / validation / test sources (e.g. {min(bad)})")


# ── building ─────────────────────────────────────────────────────────────────


def _bank_twin_instruction(bank_tasks: Path, source: str) -> str:
    return (bank_tasks / f"{slug(source)}__high_risk" / "instruction.md").read_text()


def build_row(
    role: str,
    source: str,
    surface: str,
    pool: SurfacePool,
    pool_pair: str,
    *,
    bank_tasks: Path,
    out_root: Path,
    split: str,
    seen: bool | None = None,
    seed: int = 0,
) -> SurfacePairRow | None:
    """Build one (source, surface) twin/control pair on the bank's task directories; None when it cannot be built."""
    try:
        built = build_same_site(bank_tasks, out_root, source, surface, pool=pool, seed=seed, split=split, pool_pair=pool_pair)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning("%s %s: %s", source, surface, e)
        return None
    if built is None:
        return None
    twin, control = built
    want = _bank_twin_instruction(bank_tasks, source)
    # `empty_backup` appends the backup sentence by construction; every other surface keeps the instruction as it was.
    equal = all((v.task_dir / "instruction.md").read_text() == want for v in (twin, control))
    if surface != "empty_backup" and not equal:
        logger.warning("%s %s: instruction differs from the bank twin's", source, surface)
    return SurfacePairRow(
        role=role,
        source=source,
        surface=surface,
        pool_pair=pool_pair,
        pool_pair_heldout=is_heldout(surface, pool.by_id(pool_pair)),
        hard_negative=bool(twin.detail.get("hard_negative")),
        request_heldout=bool(twin.spec.detail.get("phrasing_heldout")),
        twin_variant=twin.variant_id,
        control_variant=control.variant_id,
        twin_task_dir=str(twin.task_dir),
        control_task_dir=str(control.task_dir),
        instruction_equal=equal,
        seen=seen,
    )


def build_sft_rows(
    sources: Sequence[str],
    *,
    bank_tasks: Path,
    out_root: Path = SURFACE_TASKS,
    surfaces: Sequence[str] = SURFACES_TRAIN,
    salt: str = DEFAULT_SALT,
) -> tuple[list[SurfacePairRow], Counter[str]]:
    rows: list[SurfacePairRow] = []
    counts: Counter[str] = Counter()
    for surface in surfaces:
        if surface not in SURFACES_TRAIN:
            raise ValueError(f"{surface} is a held-out surface: it never enters SFT")
        pool = get_pool(surface)
        assignment = assign_pool_pairs(sources, pool, salt)
        for source in sorted(sources):
            row = build_row(ROLE_SFT, source, surface, pool, assignment[source], bank_tasks=bank_tasks, out_root=out_root, split="train")
            if row is None:
                counts[f"{surface}_not_built"] += 1
                continue
            rows.append(row)
            counts[surface] += 1
    return rows, counts


def probe_sources(data_dir: str | Path, rl_pool: str | Path, n: int, salt: str = DEFAULT_SALT) -> tuple[list[str], list[str]]:
    """(training-surface probe sources, held-out-surface probe sources): RL-pool and select sources in a stable hash
    order, `n` per surface; the held-out surfaces use select sources only."""
    select = sorted(load_select_sources(Path(data_dir)))
    rl = sorted(rl_pool_sources(rl_pool))
    mixed = sorted({*select, *rl}, key=lambda s: _rank(salt, "probe", s))
    return mixed[:n], sorted(select, key=lambda s: _rank(salt, "probe-held", s))[:n]


def build_probe_rows(
    train_sources: Sequence[str],
    heldout_sources: Sequence[str],
    sft_pairs: Mapping[str, set[str]],
    *,
    bank_tasks: Path,
    out_root: Path = SURFACE_TASKS,
    salt: str = DEFAULT_SALT,
) -> tuple[list[SurfacePairRow], Counter[str]]:
    rows: list[SurfacePairRow] = []
    counts: Counter[str] = Counter()
    for surface in SURFACES:
        pool = get_pool(surface)
        trained = surface in SURFACES_TRAIN
        sources = sorted(train_sources if trained else heldout_sources)
        if trained:
            used = sorted(pool.by_id(i).id for i in sft_pairs.get(surface, ()))
            test_pairs = [p.id for p in pool.pairs_for("test")]
            for k, source in enumerate(sorted(sources, key=lambda s: _rank(salt, surface, "probe-seen", s))):
                seen = k % 2 == 0 and bool(used)
                ids = used if seen else test_pairs
                if not ids:
                    counts[f"{surface}_no_pairs"] += 1
                    continue
                pair = sorted(ids, key=lambda i: _rank(salt, surface, source, i))[0]
                row = build_row(ROLE_PROBE, source, surface, pool, pair, bank_tasks=bank_tasks, out_root=out_root,
                                split="train" if seen else "test", seen=seen)  # fmt: skip
                _record(rows, counts, surface, row)
        else:
            assignment = assign_pool_pairs(sources, pool, salt)
            for source in sources:
                row = build_row(ROLE_PROBE, source, surface, pool, assignment[source], bank_tasks=bank_tasks, out_root=out_root,
                                split="train", seen=False)  # fmt: skip
                _record(rows, counts, surface, row)
    return rows, counts


def _record(rows: list[SurfacePairRow], counts: Counter[str], surface: str, row: SurfacePairRow | None) -> None:
    if row is None:
        counts[f"{surface}_not_built"] += 1
    else:
        rows.append(row)
        counts[surface] += 1


# ── probe prompt sets ────────────────────────────────────────────────────────


def surface_prompts(
    bank_rows: Sequence[Any], rows: Sequence[SurfacePairRow], outputs: Mapping[str, dp.SandboxOutputs], *, per_variant: int = 1
) -> tuple[list[dp.DecisionPrompt], Counter[str]]:
    """Own-prefix and zero-prefix prompts of the probe rows, one call per surface (a source has one pair per surface)."""
    counts: Counter[str] = Counter()
    split = dp.PhrasingSplit.build()
    sft = sft_pool_pairs(read_manifest())
    probe_rows = [r for r in rows if r.role == ROLE_PROBE]
    prompts: list[dp.DecisionPrompt] = []
    for surface in SURFACES:
        refs = refs_of(r for r in probe_rows if r.surface == surface)
        if not refs:
            continue
        before = len(prompts)
        prompts += dp.own_prefix_prompts(bank_rows, refs, outputs, split, counts, per_variant=per_variant, sft_pairs=sft)
        prompts += dp.zero_prefix_prompts(refs, outputs, split, counts, sft_pairs=sft)
        counts[f"prompts_{surface}"] = len(prompts) - before
    return prompts, counts


# ── CLI ──────────────────────────────────────────────────────────────────────


def _legacy_sources(args: argparse.Namespace) -> tuple[list[str], set[str]]:
    """(sources the iteration-13 decision set built pairs on, distill-pool sources)."""
    from honest_agents.sft import data
    from honest_agents.sft import decision as dec
    from honest_agents.sft.train_sft import load_config

    cfg = load_config(args.config, [f"data.bank={args.bank}"])
    opts = dec.parse_options({**cfg.decision, "surfaces": None, "anchors": None})
    sel = data.read_rows(cfg.data.bank, data.RowFilter(**cfg.data.row_filter))
    ds = dec.build_dataset(sel.usable, opts)
    distill = dec.pool_pair_ids([opts.distill_pool])
    used = sorted({p.state.pair_id for p in ds.pairs})
    # the legacy build drops a source whose README phrasing is held out; every other distill source with a built pair
    # stays, which is what "used by the iteration-13 decision set" means
    return used, distill


def _cmd_build(args: argparse.Namespace) -> int:
    bank_tasks = Path(args.tasks)
    out_root = Path(args.out)
    if args.role == ROLE_SFT:
        used, distill = _legacy_sources(args)
        assert_sft_sources(used, distill, rl_pool_sources(args.rl_pool), args.data_dir)
        sources = sorted(used, key=lambda s: _rank(args.salt, "subset", s))
        if args.limit_sources:
            sources = sources[: args.limit_sources]
        rows, counts = build_sft_rows(sources, bank_tasks=bank_tasks, out_root=out_root, surfaces=args.surface or SURFACES_TRAIN, salt=args.salt)
        assert all(not r.pool_pair_heldout for r in rows), "a held-out pool pair reached the SFT rows"
    else:
        train_src, held_src = probe_sources(args.data_dir, args.rl_pool, args.limit_sources or DEFAULT_PROBE_SOURCES, args.salt)
        sft = sft_pool_pairs(read_manifest(args.manifest))
        rows, counts = build_probe_rows(train_src, held_src, sft, bank_tasks=bank_tasks, out_root=out_root, salt=args.salt)
        sft_sources = {r.source for r in read_manifest(args.manifest) if r.role == ROLE_SFT}
        if overlap := sft_sources & {r.source for r in rows}:
            raise ValueError(f"{len(overlap)} probe source(s) are SFT sources (e.g. {min(overlap)})")
    merged = merge_manifest(rows, args.manifest, replace_roles=[args.role] if not args.surface else [])
    print(json.dumps({"role": args.role, "built": dict(counts), "manifest_rows": len(merged),
                      "instruction_unequal": sum(not r.instruction_equal for r in rows if r.surface != "empty_backup"),
                      "manifest": str(args.manifest)}, indent=2))  # fmt: skip
    return 0


def _cmd_outputs(args: argparse.Namespace) -> int:
    rows = [r for r in read_manifest(args.manifest) if args.role in (None, r.role)]
    refs = refs_of(rows)
    summary = dp.collect_outputs(refs, args.out, parallel=args.parallel, environment_type=args.environment_type, host=args.host, limit=args.limit)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["missing"] else 1


def _cmd_prompts(args: argparse.Namespace) -> int:
    from honest_agents.bank import iter_bank

    rows = read_manifest(args.manifest)
    wanted = {r.source for r in rows if r.role == ROLE_PROBE}
    bank_rows = [
        r for r in iter_bank(Path(args.bank).expanduser()) if r.task.get("family") == dp.HIGH_RISK and r.task.get("pair_id") in wanted
    ]
    outputs = dp.read_outputs([args.sft_outputs, args.out_cache])
    prompts, counts = surface_prompts(bank_rows, rows, outputs, per_variant=args.per_variant)
    manifest = dp.prompt_manifest(prompts, counts, dp.PhrasingSplit.build())
    dp.write_prompts(prompts, args.out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(SURFACE_PAIRS))
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build twin/control task directories and write the manifest")
    b.add_argument("--role", choices=(ROLE_SFT, ROLE_PROBE), required=True)
    b.add_argument("--bank", default="~/Documents/honest-agents-archive/iterations/03-passk-bank/eval/passk/bank")
    b.add_argument("--config", default="configs/sft_decision_qwen35.yaml", help="iteration-13 decision config (sources)")
    b.add_argument("--tasks", default="data/tasks", help="the iteration-3 task directories (`__high_risk` / `__feasible`)")
    b.add_argument("--out", default=str(SURFACE_TASKS))
    b.add_argument("--data-dir", default="data")
    b.add_argument("--rl-pool", default="data/train_rl.jsonl")
    b.add_argument("--surface", nargs="+", choices=SURFACES, help="only these surfaces (rows of the others are kept)")
    b.add_argument("--limit-sources", type=int, default=0, help="a subset: the first N sources in hash order")
    b.add_argument("--salt", default=DEFAULT_SALT)
    b.set_defaults(fn=_cmd_build)
    o = sub.add_parser("outputs", help="boot every manifest variant not yet cached and store its inspection outputs")
    o.add_argument("--role", choices=(ROLE_SFT, ROLE_PROBE), default=None)
    o.add_argument("--out", default=str(SURFACE_OUTPUTS))
    o.add_argument("--parallel", type=int, default=DEFAULT_PARALLEL)
    o.add_argument("--environment-type", default="docker")
    o.add_argument("--host", default="mac-orbstack")
    o.add_argument("--limit", type=int, default=None)
    o.set_defaults(fn=_cmd_outputs)
    p = sub.add_parser("prompts", help="assemble the six-surface decision probe prompt set")
    p.add_argument("--bank", required=True)
    p.add_argument("--out", default=str(SURFACE_PROMPTS))
    p.add_argument("--sft-outputs", default=str(dp.SANDBOX_OUTPUTS))
    p.add_argument("--out-cache", default=str(SURFACE_OUTPUTS))
    p.add_argument("--per-variant", type=int, default=1)
    p.set_defaults(fn=_cmd_prompts)
    c = sub.add_parser("cleanup", help="remove leaked sandbox containers and networks (name-filtered)")
    c.set_defaults(fn=lambda a: dp._cmd_cleanup(argparse.Namespace(dry_run=False)))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
