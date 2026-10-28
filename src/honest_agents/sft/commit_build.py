"""The frozen commit-data build of iteration 16 (prereg section 3): merge the point builds, and train on the result.

**Merge** (`merge`, no GPU, no Docker). The claim build (`sft.commit_checks build`) and the destructive-step build
(`sft.destructive_splice`) are concatenated into one build with one quota (`commit_points.schema.Quota`: the points it
trains, the per-task cap, the outcome ratio, the cell minimum, the task and source share caps). The source share is the
only quota a point build cannot hold alone (it is a share of the whole build): whole matched pairs are dropped from the
source over the cap with the most examples, in stable hash order, until no source is over it. A pair is the examples
that share a prefix episode (`meta.source_episode` of a destructive pair, `meta.base_episode` of a claim: an own pass and
its spliced fail, or an own fail alone). The merged build is then held to `commit_points.coverage --strict` (the gate of
stage (c)'s `data_gates`) and refused if it fails. Output (`--out`): `examples.jsonl`, `build_manifest.json` (its
`build` block holds every input's and output's sha256), `summary.json`.

    uv run python -m honest_agents.sft.commit_build merge --build CLAIM_DIR --build DESTRUCTIVE_DIR --out DIR \\
        --points claim_done destructive_step --max-outcome-ratio 1.5 --max-source-share 0.04 --max-task-share 0.015

**Training** (`kind: commit_build` in `sft.train_sft`): the merged build's examples as they are (their loss turns are
the build's: the check commands and the decision; tool outputs masked) plus the G2 own-success anchors
(`decision.anchors`, built as `sft.read_rft` builds them). The stage refuses a build whose manifest does not list
exactly its examples, whose `build_manifest.json` sha256 is not `commit_build.manifest_sha256` (when set), or any
example or anchor whose source is in `tasks.select_split.excluded_from_commit_training` (DEF-63).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honest_agents.commit_points import coverage
from honest_agents.commit_points.schema import POINTS, BuildManifest, Quota, read_manifest, write_manifest
from honest_agents.sft import commit_checks as cc
from honest_agents.sft.data import SFTExample, pair_of
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage
from honest_agents.tasks.select_split import excluded_from_commit_training

KIND = "commit_build"
EXAMPLES_FILE = "examples.jsonl"
DROP_SOURCE_SHARE = "source_share"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_examples(path: str | Path) -> list[SFTExample]:
    with open(path) as f:
        return [SFTExample(**json.loads(line)) for line in f if line.strip()]


def group_of(ex: SFTExample) -> str:
    """The prefix episode the example shares with its matched partner (module docstring)."""
    return str(ex.meta.get("source_episode") or ex.meta.get("base_episode") or ex.id)


@dataclass
class PointBuild:
    path: Path
    examples: list[SFTExample]
    manifest: BuildManifest

    @classmethod
    def read(cls, directory: str | Path) -> PointBuild:
        d = Path(directory).expanduser()
        examples = read_examples(d / EXAMPLES_FILE)
        manifest = read_manifest(d / cc.MANIFEST_FILE)
        if (a := sorted(e.id for e in examples)) != (b := sorted(r.id for r in manifest.examples)):
            raise ValueError(
                f"{d}: the manifest lists {len(b)} examples, the examples file holds {len(a)} (or other ids)"
            )
        return cls(d, examples, manifest)


def trim_source_share(examples: Sequence[SFTExample], max_share: float, dropped: Counter[str]) -> list[SFTExample]:
    """Drop whole pairs from the largest source over `max_share` (stable hash order) until none is over it."""
    groups: dict[str, list[SFTExample]] = {}
    for ex in examples:
        groups.setdefault(group_of(ex), []).append(ex)
    by_source: dict[str, list[str]] = {}
    for key, members in groups.items():
        by_source.setdefault(str(members[0].meta["source"]), []).append(key)
    for keys in by_source.values():
        keys.sort(key=lambda k: cc.stable_hash(f"share:{k}"))
    size = {s: sum(len(groups[k]) for k in keys) for s, keys in by_source.items()}
    total = sum(size.values())
    gone: set[str] = set()
    while total and (over := [s for s in size if size[s] > max_share * total and by_source[s]]):
        source = max(over, key=lambda s: (size[s], s))
        key = by_source[source].pop()
        gone.add(key)
        size[source] -= len(groups[key])
        total -= len(groups[key])
        dropped[DROP_SOURCE_SHARE] += len(groups[key])
    return [ex for ex in examples if group_of(ex) not in gone]


def merge(builds: Sequence[str | Path], out: str | Path, quota: Quota, data_dir: str | Path = "data") -> dict[str, Any]:
    parts = [PointBuild.read(b) for b in builds]
    examples = [e for p in parts for e in p.examples]
    if dup := [i for i, n in Counter(e.id for e in examples).items() if n > 1]:
        raise ValueError(f"{len(dup)} example id(s) in more than one build, e.g. {dup[0]}")
    refs = {r.id: r for p in parts for r in p.manifest.examples}
    dropped: Counter[str] = Counter()
    kept = trim_source_share(examples, quota.max_source_share, dropped)
    excluded = excluded_from_commit_training(data_dir)
    n_checked = cc.assert_no_excluded([str(e.meta["source"]) for e in kept], excluded, "examples")
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / EXAMPLES_FILE, "w") as f:
        f.writelines(json.dumps(dataclasses.asdict(e), default=str) + "\n" for e in kept)
    counts = dict(sorted(Counter(f"{refs[e.id].point}/{refs[e.id].outcome}" for e in kept).items()))
    build = {
        "inputs": [
            {"path": str(p.path), "examples": len(p.examples), "examples_sha256": sha256_file(p.path / EXAMPLES_FILE),
             "manifest_sha256": sha256_file(p.path / cc.MANIFEST_FILE), "build": p.manifest.build}
            for p in parts
        ],
        "examples_sha256": sha256_file(out / EXAMPLES_FILE),
        "examples": counts,
        "dropped": dict(dropped),
    }  # fmt: skip
    manifest = BuildManifest.of([refs[e.id] for e in kept], quota, build)
    write_manifest(manifest, out / cc.MANIFEST_FILE)
    report = coverage.coverage(manifest, excluded)
    summary = {
        "examples": counts,
        "n_examples": len(kept),
        "sources": len({e.meta["source"] for e in kept}),
        "dropped": dict(dropped),
        "excluded_sources_checked": {"examples": n_checked, "excluded": 0},
        "manifest_sha256": sha256_file(out / cc.MANIFEST_FILE),
        "coverage": {"failures": report.failures(strict=True), "max_source_share": report.max_source_share,
                     "fail_share": report.fail_share},
        "quota": quota.to_dict(),
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if failures := report.failures(strict=True):
        raise ValueError(f"the merged build fails coverage --strict: {failures}\n{coverage.render(report, True)}")
    return summary


# ── training stage ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CommitBuildOptions:
    build: str = ""  # the merged build directory (examples.jsonl, build_manifest.json)
    manifest_sha256: str | None = None  # the frozen build's manifest digest (prereg: frozen before any training)
    data_dir: str = "data"  # `excluded_from_commit_training`

    def __post_init__(self) -> None:
        if not self.build:
            raise ValueError("commit_build.build is the merged build directory")


def parse_options(raw: Mapping[str, Any] | None) -> CommitBuildOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(CommitBuildOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `commit_build:`; expected {sorted(allowed)}")
    return CommitBuildOptions(**raw)


def validate(cfg: SftConfig) -> None:
    from honest_agents.sft import decision as dec

    parse_options(cfg.commit_build)
    dec.validate(cfg)
    if dec.parse_options(cfg.decision).anchors is None:
        raise ValueError("commit_build needs `decision.anchors` (the G2 anchors)")


def load_build(opts: CommitBuildOptions) -> list[SFTExample]:
    """The frozen build's examples, checked against its manifest and digest and the excluded sources."""
    build = PointBuild.read(opts.build)
    digest = sha256_file(build.path / cc.MANIFEST_FILE)
    if opts.manifest_sha256 is not None and digest != opts.manifest_sha256:
        raise ValueError(f"{build.path}: build_manifest.json sha256 {digest} is not the frozen {opts.manifest_sha256}")
    excluded = excluded_from_commit_training(opts.data_dir)
    cc.assert_no_excluded([str(e.meta["source"]) for e in build.examples], excluded, "commit-build examples")
    return build.examples


def _stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    from honest_agents.sft import read_rft

    opts = parse_options(cfg.commit_build)
    examples = load_build(opts)
    anchors = read_rft.build_anchors([*fit, *held], cfg)
    excluded = excluded_from_commit_training(opts.data_dir)
    rows = {r.episode_id: r for r in [*fit, *held]}
    anchor_sources = [pair_of(rows[str(a.meta["source_episode"])]) for a in anchors.examples]
    cc.assert_no_excluded(anchor_sources, excluded, "anchors")
    train = [*examples, *anchors.examples]
    info = {
        KIND: {
            "build": opts.build,
            "manifest_sha256": sha256_file(Path(opts.build).expanduser() / cc.MANIFEST_FILE),
            "examples": len(examples),
            "by_point": dict(sorted(Counter(f"{e.meta['point']}/{e.meta['outcome']}" for e in examples).items())),
            "anchors": anchors.info,
            "examples_train": len(train),
        }
    }
    return StageData(train, [], info)


register_stage(KIND, _stage, validate)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.commit_build", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge", help="merge point builds into the frozen build (coverage --strict must pass)")
    m.add_argument("--build", action="append", required=True, help="a point build directory (repeatable)")
    m.add_argument("--out", required=True)
    m.add_argument("--data-dir", default="data")
    m.add_argument("--points", nargs="+", choices=POINTS, default=list(POINTS))
    m.add_argument("--cap-per-task", type=int, default=Quota.cap_per_task)
    m.add_argument("--max-outcome-ratio", type=float, default=Quota.max_outcome_ratio)
    m.add_argument("--min-per-cell", type=int, default=Quota.min_per_cell)
    m.add_argument("--max-source-share", type=float, default=Quota.max_source_share)
    m.add_argument("--max-task-share", type=float, default=None)
    args = ap.parse_args(argv)
    quota = Quota(args.cap_per_task, args.max_outcome_ratio, args.min_per_cell, args.max_source_share,
                  args.max_task_share, tuple(args.points))  # fmt: skip
    summary = merge(args.build, args.out, quota, args.data_dir)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
