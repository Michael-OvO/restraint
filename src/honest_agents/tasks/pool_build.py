"""Pinned pair assignments and the build manifests of the surface eval (DEF-07) and of the decision probe / SFT surface
builds (DEF-18), integrity-gate sweep S7.

    uv run python -m honest_agents.tasks.pool_build record [--bank data/tasks] \\
        [--ids-file data/decision/val_high_risk_sources.txt] [--split test] [--seed 0]
    uv run python -m honest_agents.tasks.pool_build check
    uv run python -m honest_agents.tasks.pool_build record-decision
    uv run python -m honest_agents.tasks.pool_build check-decision

`record` builds every source of `--ids-file` x surface (varied mode, as `surfaces_cli build --mode varied`) into a
temporary directory, adds the pair and `{context}` of every twin not yet in `data/pools/pair_assignments.json` (recorded
ones are honoured by the build and kept as they are), and writes `data/pools/build_manifest.json`: the build parameters,
the source ids and the `validate.task_hash` of every variant dir. Run it once after a build that is to stay fixed; a
later pool growth then leaves those variants byte-identical. Unrecorded sources draw over the whole split, as before.

`check` (S7, L0) rebuilds the manifest's sources the same way (the ids file is not needed) and exits 1 unless every
variant's hash matches; it lists the variants that changed, are missing or are new, and fails when the ids file exists
and no longer lists the recorded sources. Nothing is kept on disk: the temporary build is deleted after hashing.

`record-decision` pins every twin of `data/decision/surfaces/tasks` (the rows of `surface_pairs.jsonl` and the root's
unlisted leftovers) to the pair and `{context}` it was built with, in `data/pools/decision_pair_assignments.json`
(recorded pins are kept), rebuilds the root through the same selection as `decision_surfaces build` (round robin, the
probe's pair rank, the SFT config's cap for "seen") into a temporary directory, refuses unless every live dir is
reproduced byte for byte, and writes `data/pools/decision_build_manifest.json` (parameters, sources per role, hashes).
`check-decision` rebuilds from the manifest alone and compares the hashes, as `check` does.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks import decision_surfaces as dsurf
from honest_agents.tasks.perturb.risk_surfaces import (
    SURFACES,
    Assignment,
    AssignmentsFile,
    assignments_path,
    bank_base,
    build_varied,
    is_heldout,
    load_assignments,
    load_pool,
    pool_dir,
)
from honest_agents.tasks.taskfs import read_task_toml
from honest_agents.tasks.validate import task_hash

MANIFEST_FILE = "build_manifest.json"
DEFAULT_BANK = "data/tasks"
DEFAULT_IDS_FILE = "data/decision/val_high_risk_sources.txt"


@dataclass(frozen=True)
class BuildSpec:
    """What `surfaces_cli build --mode varied` was run with (paths relative to the repo root)."""

    bank: str = DEFAULT_BANK
    ids_file: str = DEFAULT_IDS_FILE
    split: str = "test"
    seed: int = 0
    surfaces: tuple[str, ...] = SURFACES
    mode: str = "varied"


@dataclass(frozen=True)
class Built:
    """One variant of a build: its dir hash and the pair and context it drew."""

    variant_id: str
    surface: str
    control: bool
    task_hash: str
    pool_pair: str
    context: str


@dataclass
class BuildResult:
    variants: list[Built] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # "<source> <surface>: why"


def manifest_path(pools: Path | None = None) -> Path:
    return (pools or pool_dir()).parent / MANIFEST_FILE


def read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def build(
    spec: BuildSpec, *, pools: Path | None = None, root: Path = Path(), sources: Sequence[str] | None = None
) -> BuildResult:
    """Build `spec` (its ids file, or `sources`) into a temporary directory and hash every variant dir; the directory
    is deleted afterwards."""
    pools = pools or pool_dir()
    loaded = {s: load_pool(s, directory=pools) for s in spec.surfaces}
    out = BuildResult()
    bank = root / spec.bank
    with tempfile.TemporaryDirectory(prefix="pool_build_") as td:
        for ident in read_ids(root / spec.ids_file) if sources is None else sources:
            src = bank / (bank_base(ident) + "__feasible")
            if not (src / "task.toml").is_file():
                out.skipped.extend(f"{ident} {s}: no {src}" for s in spec.surfaces)
                continue
            pair_id = HonestTaskSpec.from_task_dir(src).pair_id or ident
            for s in spec.surfaces:
                built = build_varied(
                    src, Path(td), pair_id=pair_id, surface=s, pool=loaded[s], seed=spec.seed, split=spec.split
                )
                if built is None:
                    out.skipped.append(f"{ident} {s}: no plan (no usable site, pair or phrasing)")
                    continue
                for v in built:
                    d = v.detail
                    out.variants.append(
                        Built(
                            v.variant_id,
                            s,
                            v.spec.is_control,
                            task_hash(v.task_dir),
                            str(d["pool_pair"]),
                            str(d.get("context", "")),
                        )
                    )
    return out


def new_assignments(result: BuildResult, spec: BuildSpec, recorded: AssignmentsFile) -> dict[str, Assignment]:
    """The twins of `result` that `recorded` does not hold yet, as assignments."""
    return {
        b.variant_id: Assignment(
            surface=b.surface, pair=b.pool_pair, context=b.context, split=spec.split, seed=spec.seed
        )
        for b in result.variants
        if not b.control and b.variant_id not in recorded.variants
    }


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def cmd_record(args: argparse.Namespace) -> int:
    pools = args.pools or pool_dir()
    spec = BuildSpec(bank=args.bank, ids_file=args.ids_file, split=args.split, seed=args.seed)
    sources = read_ids(Path(spec.ids_file))
    result = build(spec, pools=pools, sources=sources)
    for s in result.skipped:
        print(f"skip {s}", file=sys.stderr)
    if result.skipped and not args.allow_skipped:
        print("refusing to record a build with skipped sources (--allow-skipped to override)", file=sys.stderr)
        return 1
    apath = assignments_path(pools)
    recorded = load_assignments(apath)
    added = new_assignments(result, spec, recorded)
    merged = AssignmentsFile(
        version=recorded.version,
        note=recorded.note
        or "twin variant id -> the pool pair and {context} it was built with; honoured by risk_surfaces._pick_pair "
        "(DEF-07). Written by `python -m honest_agents.tasks.pool_build record`; never edit an entry by hand.",
        variants={**recorded.variants, **added},
    )
    _write_json(apath, merged.model_dump())
    manifest = {
        "version": 1,
        "build": {**asdict(spec), "surfaces": list(spec.surfaces)},
        "sources": sources,  # the ids file's contents, so `check` needs only the bank
        "variants": {b.variant_id: b.task_hash for b in sorted(result.variants, key=lambda b: b.variant_id)},
    }
    _write_json(manifest_path(pools), manifest)
    print(
        f"{len(result.variants)} variants hashed -> {manifest_path(pools)}; "
        f"{len(added)} new assignment(s), {len(merged.variants)} in {apath}"
    )
    return 0


def compare(manifest: dict[str, str], result: BuildResult) -> tuple[list[str], list[str], list[str]]:
    """(changed, missing, new) variant ids of a rebuild against the manifest's hashes."""
    now = {b.variant_id: b.task_hash for b in result.variants}
    changed = sorted(v for v in manifest if v in now and now[v] != manifest[v])
    missing = sorted(set(manifest) - set(now))
    new = sorted(set(now) - set(manifest))
    return changed, missing, new


def cmd_check(args: argparse.Namespace) -> int:
    pools = args.pools or pool_dir()
    raw = json.loads(manifest_path(pools).read_text())
    b = raw["build"]
    spec = BuildSpec(
        bank=b["bank"],
        ids_file=b["ids_file"],
        split=b["split"],
        seed=int(b["seed"]),
        surfaces=tuple(b["surfaces"]),
        mode=b["mode"],
    )
    sources = list(raw["sources"])
    result = build(spec, pools=pools, sources=sources)
    changed, missing, new = compare(raw["variants"], result)
    for label, vids in (("CHANGED", changed), ("MISSING", missing), ("NEW", new)):
        for v in vids:
            print(f"{label} {v}")
    n = len(raw["variants"])
    print(f"S7: {n - len(changed) - len(missing)}/{n} variants byte-identical; {len(new)} not in the manifest")
    ids_file = Path(spec.ids_file)
    drifted = ids_file.is_file() and read_ids(ids_file) != sources
    if drifted:
        print(f"S7: {ids_file} no longer lists the {len(sources)} recorded sources; re-run `record`")
    return 1 if changed or missing or new or drifted else 0


# ── the decision probe / SFT surface builds (DEF-18) ─────────────────────────

DECISION_MANIFEST_FILE = "decision_build_manifest.json"


@dataclass(frozen=True)
class DecisionSpec:
    """What `decision_surfaces build` was run with (paths relative to the repo root)."""

    bank: str = "data/tasks"
    pairs: str = "data/decision/surface_pairs.jsonl"
    root: str = "data/decision/surfaces/tasks"
    salt: str = dsurf.DEFAULT_SALT
    sft_config: str = dsurf.DEFAULT_SFT_CONFIG
    seed: int = 0


@dataclass(frozen=True)
class DecisionSources:
    sft: dict[str, list[str]]  # surface -> SFT sources
    probe_train: list[str]  # `ProbeSources.train`
    probe_heldout_only: list[str]  # `ProbeSources.heldout_only`
    unlisted: dict[str, str]  # twin variant id -> source, for the root's dirs that no manifest row lists


def decision_manifest_path(pools: Path | None = None) -> Path:
    return (pools or pool_dir()).parent / DECISION_MANIFEST_FILE


def decision_sources(spec: DecisionSpec, root: Path = Path()) -> DecisionSources:
    """The sources of the recorded build, read from `surface_pairs.jsonl` and the root's dirs."""
    rows = dsurf.read_manifest(root / spec.pairs)
    if odd := [r.twin_variant for r in rows if Path(r.twin_task_dir).name != dsurf.twin_variant_id(r.source, r.surface)]:
        raise ValueError(f"{len(odd)} row(s) whose twin dir is not named by its source and surface (e.g. {odd[0]})")
    sft: dict[str, list[str]] = {}
    for r in rows:
        if r.role == dsurf.ROLE_SFT:
            sft.setdefault(r.surface, []).append(r.source)
    probe = [r for r in rows if r.role == dsurf.ROLE_PROBE]
    train = sorted({r.source for r in probe if r.surface in dsurf.SURFACES_TRAIN})
    held = sorted({r.source for r in probe if r.surface == "empty_backup"})
    ps = dsurf.ProbeSources(train, held, 0, 0)
    for s in dsurf.SURFACES:  # the probe's sources are a function of these two lists
        if (got := sorted({r.source for r in probe if r.surface == s})) and got != ps.for_surface(s):
            raise ValueError(f"probe {s}: sources are not ProbeSources.for_surface of the recorded lists")
    listed = {Path(r.twin_task_dir).name for r in rows}
    unlisted = {}
    for d in sorted((root / spec.root).iterdir()) if (root / spec.root).is_dir() else []:
        if d.name.endswith("__control") or d.name in listed or not (d / "task.toml").is_file():
            continue
        unlisted[d.name] = str(HonestTaskSpec.from_task_dir(d).pair_id)
    return DecisionSources({k: sorted(v) for k, v in sft.items()}, train, held, unlisted)


def build_decision(
    spec: DecisionSpec,
    sources: DecisionSources,
    pins: dsurf.PinsFile,
    *,
    pools: Path | None = None,
    root: Path = Path(),
) -> BuildResult:
    """Rebuild the probe / SFT root the way `decision_surfaces build` does (with `pins`) into a temporary directory and
    hash every variant dir; the directory is deleted afterwards."""
    pools = pools or pool_dir()
    loaded = {s: load_pool(s, directory=pools) for s in SURFACES}
    roles = pins.roles
    bank = root / spec.bank
    out = BuildResult()
    with tempfile.TemporaryDirectory(prefix="decision_build_") as td:
        rows: list[dsurf.SurfacePairRow] = []
        for surface, srcs in sorted(sources.sft.items()):
            got, _ = dsurf.build_sft_rows(srcs, bank_tasks=bank, out_root=Path(td), surfaces=[surface], salt=spec.salt,
                                          pins=roles.get(dsurf.ROLE_SFT, {}), pools=loaded)  # fmt: skip
            rows += got
        cfg = str(root / spec.sft_config) if spec.sft_config else None
        seen = dsurf.trained_pool_pairs(rows, cfg)
        probe = dsurf.ProbeSources(sources.probe_train, sources.probe_heldout_only, 0, 0)
        got, _ = dsurf.build_probe_rows(probe, seen, bank_tasks=bank, out_root=Path(td), salt=spec.salt,
                                        pins=roles.get(dsurf.ROLE_PROBE, {}), pools=loaded)  # fmt: skip
        rows += got
        for vid, source in sorted(sources.unlisted.items()):
            pin = roles.get(dsurf.ROLE_UNLISTED, {}).get(vid)
            if pin is None:
                out.skipped.append(f"{vid}: unlisted and not pinned")
                continue
            row = dsurf.build_row(dsurf.ROLE_UNLISTED, source, pin.surface, loaded[pin.surface], pin.pair, bank_tasks=bank,
                                  out_root=Path(td), split=pin.split, seed=spec.seed, pin=pin)  # fmt: skip
            if row is not None:
                rows.append(row)
        for r in rows:
            for d in (Path(r.twin_task_dir), Path(r.control_task_dir)):
                detail = read_task_toml(d)["metadata"]["honest_variant"]
                out.variants.append(
                    Built(
                        d.name,
                        r.surface,
                        d.name.endswith("__control"),
                        task_hash(d),
                        r.pool_pair,
                        str(detail.get("context", "")),
                    )
                )
    return out


def pins_of_root(
    spec: DecisionSpec, sources: DecisionSources, pools: Path, root: Path = Path()
) -> dict[str, dict[str, Assignment]]:
    """role -> twin variant id -> the pair and `{context}` the live twin dir was built with."""
    loaded = {s: load_pool(s, directory=pools) for s in SURFACES}
    out: dict[str, dict[str, Assignment]] = {}
    roles = {Path(r.twin_task_dir).name: r.role for r in dsurf.read_manifest(root / spec.pairs)}
    roles.update(dict.fromkeys(sources.unlisted, dsurf.ROLE_UNLISTED))
    for vid, role in sorted(roles.items()):
        d = read_task_toml(root / spec.root / vid)["metadata"]["honest_variant"]
        surface, pair = str(d["surface"]), str(d["pool_pair"])
        split = "test" if is_heldout(surface, loaded[surface].by_id(pair)) else "train"
        out.setdefault(role, {})[vid] = Assignment(
            surface=surface, pair=pair, context=str(d.get("context", "")), split=split, seed=spec.seed
        )
    return out


def cmd_record_decision(args: argparse.Namespace) -> int:
    pools = args.pools or pool_dir()
    spec = DecisionSpec()
    sources = decision_sources(spec)
    recorded = dsurf.load_pins(dsurf.pins_path(pools))
    live = pins_of_root(spec, sources, pools)
    merged = {role: {**live.get(role, {}), **recorded.roles.get(role, {})} for role in {*live, *recorded.roles}}
    pins = dsurf.PinsFile(
        version=recorded.version,
        note=recorded.note
        or "role -> twin variant id -> the pool pair and {context} of the decision probe / SFT surface build "
        "(data/decision/surfaces/tasks); honoured by decision_surfaces.build_row (DEF-18). Written by "
        "`python -m honest_agents.tasks.pool_build record-decision`; never edit an entry by hand.",
        roles=merged,
    )
    result = build_decision(spec, sources, pins, pools=pools)
    now = {b.variant_id: b.task_hash for b in result.variants}
    want = {d.name: task_hash(d) for d in Path(spec.root).iterdir() if (d / "task.toml").is_file()}
    changed, missing, new = compare(want, result)
    for label, vids in (("CHANGED", changed), ("MISSING", missing), ("NEW", new), ("SKIPPED", result.skipped)):
        for v in vids:
            print(f"{label} {v}", file=sys.stderr)
    if changed or missing or new or result.skipped:
        print(
            f"refusing to record: the rebuild does not reproduce {spec.root} ({len(now)} built, {len(want)} live)",
            file=sys.stderr,
        )
        return 1
    _write_json(dsurf.pins_path(pools), pins.model_dump())
    manifest = {
        "version": 1,
        "build": asdict(spec),
        "sources": asdict(sources),
        "variants": dict(sorted(now.items())),
    }
    _write_json(decision_manifest_path(pools), manifest)
    n_pins = sum(len(v) for v in merged.values())
    print(
        f"{len(now)}/{len(want)} live dirs reproduced -> {decision_manifest_path(pools)}; {n_pins} pins in {dsurf.pins_path(pools)}"
    )
    return 0


def cmd_check_decision(args: argparse.Namespace) -> int:
    pools = args.pools or pool_dir()
    raw = json.loads(decision_manifest_path(pools).read_text())
    spec = DecisionSpec(**raw["build"])
    sources = DecisionSources(**raw["sources"])
    result = build_decision(spec, sources, dsurf.load_pins(dsurf.pins_path(pools)), pools=pools)
    changed, missing, new = compare(raw["variants"], result)
    for label, vids in (("CHANGED", changed), ("MISSING", missing), ("NEW", new), ("SKIPPED", result.skipped)):
        for v in vids:
            print(f"{label} {v}")
    n = len(raw["variants"])
    print(
        f"S7 (decision): {n - len(changed) - len(missing)}/{n} variants byte-identical; {len(new)} not in the manifest"
    )
    return 1 if changed or missing or new or result.skipped else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.tasks.pool_build", description=__doc__.split("\n")[0])
    ap.add_argument("--pools", type=Path, default=None, help=f"pool directory (default {pool_dir()})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="pin the pairs of a build and write its manifest of variant hashes")
    r.add_argument("--bank", default=DEFAULT_BANK, help="dir of the bank's task dirs (the sources' __feasible copies)")
    r.add_argument("--ids-file", default=DEFAULT_IDS_FILE)
    r.add_argument("--split", choices=("train", "test"), default="test")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--allow-skipped", action="store_true", help="record even when a source x surface did not build")
    r.set_defaults(fn=cmd_record)
    c = sub.add_parser("check", help="S7: rebuild the manifest's build and compare variant hashes")
    c.set_defaults(fn=cmd_check)
    rd = sub.add_parser("record-decision", help="pin the probe / SFT surface build and write its manifest (DEF-18)")
    rd.set_defaults(fn=cmd_record_decision)
    cd = sub.add_parser("check-decision", help="S7: rebuild the probe / SFT surface build and compare variant hashes")
    cd.set_defaults(fn=cmd_check_decision)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
