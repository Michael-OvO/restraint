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

Pinned builds (DEF-18): a twin recorded in `data/pools/decision_pair_assignments.json` (role -> twin variant id -> pair
and `{context}`, written by `python -m honest_agents.tasks.pool_build record-decision`) is built with that pair and
context whatever is added to a pool; the round robin and the probe's pair rank above only choose for unrecorded twins.
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

from pydantic import BaseModel, ConfigDict

from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks.decision_probe import VariantRef
from honest_agents.tasks.perturb.core import variant_id
from honest_agents.tasks.perturb.risk_surfaces import (
    FAMILY,
    SURFACES,
    SURFACES_HELDOUT,
    SURFACES_TRAIN,
    Assignment,
    PoolError,
    SurfacePool,
    build_same_site,
    get_pool,
    is_heldout,
    pool_dir,
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
DEFAULT_PROBE_SOURCES = 48  # per training surface: RL-pool sources with a usable bank decision state (~100 own-prefix states at 2 per source)
DEFAULT_HELDOUT_SELECT_SOURCES = 24  # select sources added to the held-out surfaces (zero-prefix prompts: no bank rows)
DEFAULT_RL_POOLS = ("data/train_rl.jsonl", "data/train_rl_v2.jsonl")
OWN_PREFIX_SURFACES = tuple(s for s in SURFACES if s != "empty_backup")  # empty_backup changes the instruction: its own prefix would not match
DEFAULT_SFT_CONFIG = "configs/sft_decision_v2_qwen35.yaml"  # its `max_pairs_per_surface` cap defines "seen" pool pairs


def _rank(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


# ── pinned pairs (DEF-18) ────────────────────────────────────────────────────

PINS_FILE = "decision_pair_assignments.json"  # next to the pool directory: data/pools/decision_pair_assignments.json
ROLE_UNLISTED = "unlisted"  # a twin dir of the root that surface_pairs.jsonl does not list (an earlier build's leftover)


class PinsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = 1
    note: str = ""
    roles: dict[str, dict[str, Assignment]] = {}  # role -> twin variant id (`<source>__high_risk__<surface>`) -> pin


def pins_path(directory: Path | None = None) -> Path:
    """`decision_pair_assignments.json` of a pool directory: its parent's (`data/pools/` for `data/pools/surfaces/`)."""
    return (directory or pool_dir()).parent / PINS_FILE


def load_pins(path: Path | None = None) -> PinsFile:
    """The recorded pins, or none when the file does not exist (a test pool in a temporary dir draws as before)."""
    p = path or pins_path()
    if not p.is_file():
        return PinsFile()
    try:
        return PinsFile.model_validate_json(p.read_text())
    except ValueError as e:
        raise PoolError(f"{p}: {e}") from e


def twin_variant_id(source: str, surface: str) -> str:
    return variant_id(source, f"{FAMILY}__{surface}")


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


def cap_rows(rows: Iterable[SurfacePairRow], max_per_surface: int | None) -> list[SurfacePairRow]:
    """The first `max_per_surface` rows of each surface in a stable hash order of the source: the cap
    `decision.surfaces.max_pairs_per_surface` of `sft.decision` applies to the manifest's SFT rows (the manifest itself
    stays uncapped). None = every row."""
    rows = list(rows)
    if max_per_surface is None:
        return rows
    out: list[SurfacePairRow] = []
    for surface in sorted({r.surface for r in rows}):
        group = sorted((r for r in rows if r.surface == surface), key=lambda r: _rank("cap", surface, r.source))
        out += group[:max_per_surface]
    return out


def sft_pool_pairs(rows: Iterable[SurfacePairRow], max_per_surface: int | None = None) -> dict[str, set[str]]:
    """surface -> the pool pair ids the SFT rows use (what a probe row calls "seen"). With `max_per_surface` (the
    training config's `decision.surfaces.max_pairs_per_surface`) only the rows the cap keeps count: a pool pair
    assigned to a source the cap drops was never trained on."""
    out: dict[str, set[str]] = {}
    for r in cap_rows((r for r in rows if r.role == ROLE_SFT), max_per_surface):
        out.setdefault(r.surface, set()).add(r.pool_pair)
    return out


def trained_pool_pairs(rows: Iterable[SurfacePairRow], sft_config: str | Path | None) -> dict[str, set[str]]:
    """`sft_pool_pairs` under the cap of the SFT-v2 config (`decision.surfaces.max_pairs_per_surface`); no config = no cap."""
    cap = None
    if sft_config:
        from honest_agents.sft import decision as dec
        from honest_agents.sft.train_sft import load_config

        so = dec.parse_options(load_config(str(sft_config)).decision).surfaces
        cap = so.max_pairs_per_surface if so is not None else None
    return sft_pool_pairs(rows, cap)


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


def rl_pool_sources(paths: str | Path | Iterable[str | Path] = "data/train_rl.jsonl") -> set[str]:
    """Sources of the RL pool file(s); a missing file contributes nothing."""
    from honest_agents.sft.decision import pool_pair_ids

    files = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return pool_pair_ids([p for p in files if Path(p).expanduser().is_file()])


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
    pin: Assignment | None = None,
) -> SurfacePairRow | None:
    """Build one (source, surface) twin/control pair on the bank's task directories; None when it cannot be built. A
    `pin` (recorded with the same seed) replaces `pool_pair` and the drawn `{context}` (DEF-18)."""
    context = None
    if pin is not None and pin.seed == seed:
        # checked before the build, whose errors only drop the row: a pin that cannot be honoured must stop the build
        vid = twin_variant_id(source, surface)
        if pin.surface != surface or pin.split != split:
            raise PoolError(f"{vid} is pinned to {pin.surface} {pin.pair!r} for split={pin.split!r}, not {split!r}")
        if pin.pair not in {p.id for p in pool.pairs}:
            raise PoolError(f"{vid} is pinned to {pin.pair!r}, which is no longer in the {surface} pool")
        if is_heldout(surface, pool.by_id(pin.pair)) != (split == "test"):
            raise PoolError(f"{vid}: pinned pair {pin.pair!r} changed its `heldout` flag")
        pool_pair, context = pin.pair, pin.context
    try:
        built = build_same_site(bank_tasks, out_root, source, surface, pool=pool, seed=seed, split=split,
                                pool_pair=pool_pair, pool_context=context)  # fmt: skip
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
    pins: Mapping[str, Assignment] | None = None,
    pools: Mapping[str, SurfacePool] | None = None,
) -> tuple[list[SurfacePairRow], Counter[str]]:
    """`pins`: twin variant id -> recorded pair (default: the SFT role of `load_pins()`); `pools`: surface -> pool
    (default `get_pool`)."""
    pins = load_pins().roles.get(ROLE_SFT, {}) if pins is None else pins
    rows: list[SurfacePairRow] = []
    counts: Counter[str] = Counter()
    for surface in surfaces:
        if surface not in SURFACES_TRAIN:
            raise ValueError(f"{surface} is a held-out surface: it never enters SFT")
        pool = pools[surface] if pools is not None else get_pool(surface)
        assignment = assign_pool_pairs(sources, pool, salt)
        for source in sorted(sources):
            row = build_row(ROLE_SFT, source, surface, pool, assignment[source], bank_tasks=bank_tasks, out_root=out_root,
                            split="train", pin=pins.get(twin_variant_id(source, surface)))  # fmt: skip
            if row is None:
                counts[f"{surface}_not_built"] += 1
                continue
            rows.append(row)
            counts[surface] += 1
    return rows, counts


@dataclass(frozen=True)
class ProbeSources:
    """Which sources the probe builds each surface on."""

    train: list[str]  # training surfaces and inbound_link: RL-pool sources with a usable bank decision state
    heldout_only: list[str]  # select sources: zero-prefix prompts of the held-out surfaces (no bank rows)
    rl_pool_usable: int  # RL-pool sources with a usable state before the cap to `n`
    rl_pool_total: int

    def for_surface(self, surface: str) -> list[str]:
        if surface in SURFACES_TRAIN:
            return sorted(self.train)
        if surface == "empty_backup":  # the instruction changes, so there is no own prefix to continue: zero-prefix only
            return sorted(self.heldout_only)
        return sorted({*self.train, *self.heldout_only})  # inbound_link keeps the instruction: own prefix + zero-prefix


def probe_sources(
    data_dir: str | Path,
    rl_pools: str | Path | Iterable[str | Path],
    bank_rows: Iterable[Any],
    n: int,
    *,
    salt: str = DEFAULT_SALT,
    exclude: Iterable[str] = (),
    n_select: int = DEFAULT_HELDOUT_SELECT_SOURCES,
) -> ProbeSources:
    """`n` RL-pool sources in a stable hash order that have a usable decision state in `bank_rows` (the own-prefix probe
    needs one), never a source of `exclude` (SFT, anchor and distill-pool sources) nor a select / validation / test
    source. RL-pool sources are trained on by GRPO in iteration 13 but not by the SFT model the probe reads. The
    held-out surfaces also get `n_select` select sources for zero-prefix prompts."""
    from honest_agents.sft.decision import select_states

    rl = rl_pool_sources(rl_pools)
    banned = {*exclude, *excluded_from_training(data_dir)}
    states, _ = select_states(bank_rows, pair_ids=rl - banned, per_variant=1)
    usable = sorted({st.pair_id for st in states}, key=lambda s: _rank(salt, "probe", s))
    select = sorted(load_select_sources(Path(data_dir)), key=lambda s: _rank(salt, "probe-held", s))
    return ProbeSources(usable[:n], select[:n_select], len(usable), len(rl))


def build_probe_rows(
    probe: ProbeSources,
    sft_pairs: Mapping[str, set[str]],
    *,
    bank_tasks: Path,
    out_root: Path = SURFACE_TASKS,
    salt: str = DEFAULT_SALT,
    pins: Mapping[str, Assignment] | None = None,
    pools: Mapping[str, SurfacePool] | None = None,
) -> tuple[list[SurfacePairRow], Counter[str]]:
    """`pins` and `pools` as in `build_sft_rows` (default pins: the probe role of `load_pins()`)."""
    pins = load_pins().roles.get(ROLE_PROBE, {}) if pins is None else pins
    rows: list[SurfacePairRow] = []
    counts: Counter[str] = Counter()
    for surface in SURFACES:
        pool = pools[surface] if pools is not None else get_pool(surface)
        trained = surface in SURFACES_TRAIN
        sources = probe.for_surface(surface)
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
                                split="train" if seen else "test", seen=seen,
                                pin=pins.get(twin_variant_id(source, surface)))  # fmt: skip
                _record(rows, counts, surface, row)
        else:
            assignment = assign_pool_pairs(sources, pool, salt)
            for source in sources:
                row = build_row(ROLE_PROBE, source, surface, pool, assignment[source], bank_tasks=bank_tasks, out_root=out_root,
                                split="train", seen=False, pin=pins.get(twin_variant_id(source, surface)))  # fmt: skip
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
    bank_rows: Sequence[Any],
    rows: Sequence[SurfacePairRow],
    outputs: Mapping[str, dp.SandboxOutputs],
    *,
    per_variant: int = 1,
    sft_pairs: Mapping[str, set[str]] | None = None,
) -> tuple[list[dp.DecisionPrompt], Counter[str]]:
    """Own-prefix and zero-prefix prompts of the probe rows, one call per surface (a source has one pair per surface).
    `sft_pairs` (surface -> pool pairs the SFT trained on, `trained_pool_pairs`) decides a prompt's `phrasing_seen`;
    None = the manifest's SFT rows uncapped."""
    counts: Counter[str] = Counter()
    split = dp.PhrasingSplit.build()
    sft = dict(sft_pairs) if sft_pairs is not None else sft_pool_pairs(rows)
    probe_rows = [r for r in rows if r.role == ROLE_PROBE]
    prompts: list[dp.DecisionPrompt] = []
    for surface in SURFACES:
        refs = refs_of(r for r in probe_rows if r.surface == surface)
        if not refs:
            continue
        before = len(prompts)
        if surface in OWN_PREFIX_SURFACES:
            prompts += dp.own_prefix_prompts(bank_rows, refs, outputs, split, counts, per_variant=per_variant, sft_pairs=sft)
        prompts += dp.zero_prefix_prompts(refs, outputs, split, counts, sft_pairs=sft)
        counts[f"prompts_{surface}"] = len(prompts) - before
    return prompts, counts


def surface_counts(prompts: Sequence[dp.DecisionPrompt]) -> dict[str, dict[str, int]]:
    """Per surface: own-prefix sources / states (an arm-pair is one state), their seen / unseen split, zero-prefix pairs."""
    out: dict[str, dict[str, int]] = {}
    for surface in SURFACES:
        mine = [p for p in prompts if p.surface == surface and p.arm == dp.ARM_WARNING]
        own = [p for p in mine if p.set == dp.SET_OWN_PREFIX]
        zero = [p for p in mine if p.set == dp.SET_ZERO_PREFIX]
        out[surface] = {
            "own_prefix_sources": len({p.pair_id for p in own}),
            "own_prefix_states": len(own),
            "own_prefix_seen": sum(p.phrasing_seen is True for p in own),
            "own_prefix_unseen": sum(p.phrasing_seen is False for p in own),
            "zero_prefix_pairs": len(zero),
            "zero_prefix_seen": sum(p.phrasing_seen is True for p in zero),
            "zero_prefix_unseen": sum(p.phrasing_seen is False for p in zero),
        }
    return out


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
        from honest_agents.bank import iter_bank
        from honest_agents.sft.decision import pool_pair_ids

        sft_sources = {r.source for r in read_manifest(args.manifest) if r.role == ROLE_SFT}
        distill = pool_pair_ids([args.distill_pool])
        bank_rows = [r for r in iter_bank(Path(args.bank).expanduser()) if r.task.get("family") == dp.HIGH_RISK]
        probe = probe_sources(
            args.data_dir, args.rl_pool, bank_rows, args.limit_sources or DEFAULT_PROBE_SOURCES, salt=args.salt, exclude=sft_sources | distill
        )
        sft = trained_pool_pairs(read_manifest(args.manifest), args.sft_config)  # "seen" = trained under the cap
        rows, counts = build_probe_rows(probe, sft, bank_tasks=bank_tasks, out_root=out_root, salt=args.salt)
        counts["rl_pool_sources_with_usable_state"] = probe.rl_pool_usable
        counts["rl_pool_sources"] = probe.rl_pool_total
        if overlap := (sft_sources | distill) & {r.source for r in rows if r.source not in set(probe.heldout_only)}:
            raise ValueError(f"{len(overlap)} probe source(s) are SFT / distill-pool sources (e.g. {min(overlap)})")
        if bad := excluded_from_training(args.data_dir) & {r.source for r in rows if r.source not in set(probe.heldout_only)}:
            raise ValueError(f"{len(bad)} RL probe source(s) are select / validation / test sources (e.g. {min(bad)})")
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
    sft = trained_pool_pairs(rows, args.sft_config)
    prompts, counts = surface_prompts(bank_rows, rows, outputs, per_variant=args.per_variant, sft_pairs=sft)
    manifest = dp.prompt_manifest(prompts, counts, dp.PhrasingSplit.build())
    manifest["surface_counts"] = surface_counts(prompts)
    manifest["probe_sources"] = {
        "own_prefix": "RL-pool sources (data/train_rl.jsonl, data/train_rl_v2.jsonl) with a usable bank decision state; never an "
        "SFT / anchor / distill-pool, select, validation or test source. The RL pool is trained on by GRPO in iteration 13 "
        "(not by SFT-v2), which is acceptable for a probe of the SFT model.",
        "empty_backup": "zero-prefix only (the instruction changes, so no own prefix continues it); select sources",
        "per_variant": args.per_variant,
    }
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
    b.add_argument("--rl-pool", nargs="+", default=list(DEFAULT_RL_POOLS), help="RL pool files: SFT never uses their sources, the probe draws from them")
    b.add_argument("--distill-pool", default="data/distill_pool.jsonl", help="probe: the SFT / anchor source pool, excluded from the probe")
    b.add_argument("--surface", nargs="+", choices=SURFACES, help="only these surfaces (rows of the others are kept)")
    b.add_argument("--limit-sources", type=int, default=0, help="a subset: the first N sources in hash order")
    b.add_argument("--salt", default=DEFAULT_SALT)
    b.add_argument("--sft-config", default=DEFAULT_SFT_CONFIG, help="probe: the training config whose cap decides which pool pairs count as seen ('' = uncapped)")
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
    p.add_argument("--sft-config", default=DEFAULT_SFT_CONFIG, help="the training config whose cap decides which pool pairs count as seen ('' = uncapped)")
    p.set_defaults(fn=_cmd_prompts)
    c = sub.add_parser("cleanup", help="remove leaked sandbox containers and networks (name-filtered)")
    c.set_defaults(fn=lambda a: dp._cmd_cleanup(argparse.Namespace(dry_run=False)))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
