"""Pinned pair assignments and the build manifest of the surface eval (DEF-07, integrity-gate sweep S7).

    uv run python -m honest_agents.tasks.pool_build record [--bank data/tasks] \\
        [--ids-file data/decision/val_high_risk_sources.txt] [--split test] [--seed 0]
    uv run python -m honest_agents.tasks.pool_build check

`record` builds every source of `--ids-file` x surface (varied mode, as `surfaces_cli build --mode varied`) into a
temporary directory, adds the pair and `{context}` of every twin not yet in `data/pools/pair_assignments.json` (recorded
ones are honoured by the build and kept as they are), and writes `data/pools/build_manifest.json`: the build parameters,
the source ids and the `validate.task_hash` of every variant dir. Run it once after a build that is to stay fixed; a
later pool growth then leaves those variants byte-identical. Unrecorded sources draw over the whole split, as before.

`check` (S7, L0) rebuilds the manifest's sources the same way (the ids file is not needed) and exits 1 unless every
variant's hash matches; it lists the variants that changed, are missing or are new, and fails when the ids file exists
and no longer lists the recorded sources. Nothing is kept on disk: the temporary build is deleted after hashing.
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
from honest_agents.tasks.perturb.risk_surfaces import (
    SURFACES,
    Assignment,
    AssignmentsFile,
    assignments_path,
    bank_base,
    build_varied,
    load_assignments,
    load_pool,
    pool_dir,
)
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
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
