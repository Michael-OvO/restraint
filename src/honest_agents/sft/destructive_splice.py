"""Destructive-step matched pairs from the model's own surface episodes (iteration 16, commitment point (c); no Docker).

The commit-data build (`sft.commit_checks build`) mines and labels the `destructive_step` points of the it16b rollouts
(G2 on the 400 training-surface tasks of `data/decision/read_rft_tasks.jsonl`) but builds no example for them. This
module builds them with the iteration-14 surface splice (`sft.decision.build_pair`), on G2's own control episodes:

* **States.** A high_risk *control* episode that passes `teacher_stage.programmatic_drop_reason` (verified success,
  claimed) and destroys its sentinel (`readme_read.first_destructive_index` over the spec's targets: the directory and
  its evidence file). The decision state is cut where the episode first touches the target, so everything before it is
  identical on the twin's sandbox:
  - the episode never names the target before the deletion: the cut is the deletion turn (the iteration-13 rule);
  - it reads the target first (G2 does: SFT-v2 trained the reads): the cut is the first command that names the target
    (`decision.command_names_target`, any target). Every command from there up to the deletion must be non-mutating,
    whole turns (the read turns start with that command and hold only those commands; the deletion is the first call
    of its turn), and the read turns are dropped (`DROP_*` otherwise). The pair then reads `inspection_commands(site)`
    in their place on both sides. The budget lines of later own tool outputs are not re-rendered: they are off by the
    difference in step count (the dropped reads against the 2-4 injected ones).
* **Pair** (`decision.build_pair`, its consistency rules A-C): the prefix, the injected reads with the real outputs of
  each variant's own sandbox (`--sandbox-outputs`, `tasks.decision_probe`; current for the task directory), then the
  decision. Control (`outcome` pass: no hazard found, committing is right): the own deletion turn, own tail and own
  verified-success report. Twin (`fail`: the hazard is found): the deletion stripped or skipped, the tail cut at the
  target, the rule-based `risk_discovered` report quoting the evidence it read. Loss on the reads, the decision and the
  report; the prefix and every tool output are masked. Kind `commit_check` (`sft.commit_checks.KIND`).
* **Labels** (as `sft.decision verify`): every twin report grades `abstain_correct` at reward 1.0 under the twin's own
  spec, every control row `success_claimed`, no hint leak or echo, no twin drops a mutating turn. A pair that fails is
  dropped and counted (`label_*`); the summary records that none did.
* **Held out.** A pair on a held-out pool pair or a surface outside `SURFACES_TRAIN` is dropped (the surface-val eval's
  phrasings). Every source is checked against `tasks.select_split.excluded_from_commit_training` (select, val, test and
  the final decision probe's sources: DEF-63; an error unless `--drop-excluded`), and the examples are asserted again.
* **Quota.** At most `--cap-per-task` pairs per control variant (stable hash), so each `destructive_step` x outcome x
  task cell holds at most the cap (the twin variant is the fail side's task). The two outcomes are equal by
  construction. The source share is the merge's (`sft.commit_build merge`), over the whole build.
* **Hazard** of a pair: `commit_checks.hazard_of` on the twin's own environment text, in the schema's names
  (`commit_points.schema.HAZARDS`, `other` when no pattern matches).

Output (`--out`): `examples.jsonl` (`sft.data.SFTExample`), `build_manifest.json` (`commit_points.schema.BuildManifest`),
`summary.json` and `report.md`.

    uv run python -m honest_agents.sft.destructive_splice --bank BANK_DIR --tasks-root data/decision/surfaces/tasks \\
        --sandbox-outputs data/decision/sandbox_outputs_surfaces.jsonl --out DIR [--drop-excluded] [--cap-per-task 2]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.bank import BankRow
from honest_agents.commit_points.schema import DESTRUCTIVE, FAIL, PASS, BuildManifest, ExampleRef, write_manifest
from honest_agents.commit_points.schema import HAZARD_OTHER as SCHEMA_HAZARD_OTHER
from honest_agents.commit_points.schema import Quota as ManifestQuota
from honest_agents.mutation import is_mutating
from honest_agents.readme_read import first_destructive_index, high_risk_targets
from honest_agents.reward import RewardConfig
from honest_agents.sft import commit_checks as cc
from honest_agents.sft import decision as D
from honest_agents.sft.data import SFTExample
from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks.perturb.risk_surfaces import SURFACES_TRAIN, get_pool, is_heldout
from honest_agents.tasks.select_split import excluded_from_commit_training

ORIGIN_CONTROL = "surface_splice_control"
ORIGIN_TWIN = "surface_splice_twin"
# `commit_checks` hazard names -> the schema's (`commit_points.schema.HAZARDS`).
SCHEMA_HAZARD = {
    cc.HAZARD_ONLY_COPY: "copy",
    cc.HAZARD_DEPENDED_ON: "depended_on",
    cc.HAZARD_IN_USE: "in_use",
    cc.HAZARD_PROTECTED: "protected",
    cc.HAZARD_UNFINISHED: "unfinished",
}

# Drop reasons (the `sft.decision` ones are counted under their own names).
DROP_NOT_CONTROL = "not_high_risk_control"
DROP_NO_DELETION = D.DROP_NO_DELETION
DROP_MALFORMED = D.DROP_MALFORMED
DROP_MUTATION_BETWEEN = "mutation_between_read_and_deletion"
DROP_READ_MID_TURN = "read_mid_turn"
DROP_OTHER_IN_READ_TURNS = "other_command_in_read_turns"
DROP_DELETION_MID_TURN = "deletion_mid_turn"
DROP_NO_TASK_DIR = "no_task_dir"
DROP_NO_OUTPUTS = D.DROP_NO_OUTPUTS
DROP_HELD_OUT = "held_out_pool_pair_or_surface"
DROP_LABEL = "label:"  # + the failed label check
DROP_CAP = cc.DROP_CAP


@dataclass
class Config:
    banks: list[str]
    tasks_root: str
    sandbox_outputs: list[str]
    out: str
    data_dir: str = "data"
    cap_per_task: int = 2
    drop_excluded: bool = False


@dataclass
class Pair:
    built: D.BuiltPair
    hazard: str
    cut: str  # "deletion" (never named before it) | "first_read" (the own reads replaced)

    @property
    def source(self) -> str:
        return self.built.state.pair_id

    @property
    def control_task(self) -> str:
        return self.built.state.variant_id

    @property
    def twin_task(self) -> str:
        return cc.twin_id(self.built.state.variant_id)


@dataclass
class Build:
    pairs: list[Pair]
    dropped: Counter[str] = field(default_factory=Counter)
    cuts: Counter[str] = field(default_factory=Counter)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    n_sources_checked: int = 0


def surface_state(row: BankRow, target: str) -> tuple[D.DecisionState, str] | str:
    """The decision state of a surface control episode and where it was cut, or a drop reason (module docstring)."""
    targets = high_risk_targets(D._spec(row), row.outcome)
    commands = D.bash_commands(row)
    i = first_destructive_index(commands, targets)
    if i is None:
        return DROP_NO_DELETION
    names = [target, *(t for t in targets if t != target)]
    n = next((k for k in range(i) if any(D.command_names_target(commands[k], t) for t in names)), i)
    context = D.context_messages(row)
    positions = D._command_positions(context)
    if len(positions) != len(commands):
        return DROP_MALFORMED
    if n == i:
        turn, call = positions[i]
        last = all(not is_mutating(c) for c in commands[i + 1 :])
        return D.DecisionState(row, target, commands, i, turn, call, last, context), "deletion"
    if any(is_mutating(c) for c in commands[n:i]):
        return DROP_MUTATION_BETWEEN
    (read_turn, read_call), (del_turn, del_call) = positions[n], positions[i]
    if read_call != 0:
        return DROP_READ_MID_TURN
    if del_call != 0:
        return DROP_DELETION_MID_TURN
    if any(not n <= k < i for k, (t, _) in enumerate(positions) if read_turn <= t < del_turn):
        return DROP_OTHER_IN_READ_TURNS
    kept_context = [*context[:read_turn], *context[del_turn:]]
    kept_commands = [*commands[:n], *commands[i:]]
    last = all(not is_mutating(c) for c in kept_commands[n + 1 :])
    return D.DecisionState(row, target, kept_commands, n, read_turn, 0, last, kept_context), "first_read"


def held_out(pair: D.BuiltPair) -> bool:
    if pair.surface not in SURFACES_TRAIN:
        return True
    pool_pair = pair.twin.meta.get("pool_pair")
    return pool_pair is not None and is_heldout(pair.surface, get_pool(pair.surface).by_id(pool_pair))


def label_problems(pair: D.BuiltPair, reward: RewardConfig) -> list[str]:
    """The `sft.decision.verify_labels` checks that fail on this one pair."""
    return [c.name for c in D.verify_labels([pair], reward) if not c.passed]


def collect(cfg: Config) -> Build:
    excluded = excluded_from_commit_training(cfg.data_dir)
    root = Path(cfg.tasks_root).expanduser()
    outputs = dp.read_outputs([Path(p).expanduser() for p in cfg.sandbox_outputs])
    reward = RewardConfig()
    b = Build([])
    sources: set[str] = set()
    for path in cfg.banks:
        rows = list(bank_mod.iter_bank(Path(path).expanduser()))
        n_before = len(b.pairs)
        for row in rows:
            ep = cc.episode_from_row(row, str(path))
            if ep.source in excluded:
                if not cfg.drop_excluded:
                    raise cc.ExcludedSourceError(f"{path}: episode {ep.episode_id} has excluded source {ep.source}")
                b.dropped[cc.DROP_EXCLUDED] += 1
                continue
            if ep.split is not None and ep.split != "train":
                b.dropped[cc.DROP_NOT_TRAIN] += 1
                continue
            if ep.masked:
                b.dropped[cc.DROP_MASKED] += 1
                continue
            sources.add(ep.source)
            if ep.family != cc.HIGH_RISK or not ep.is_control:
                b.dropped[DROP_NOT_CONTROL] += 1
                continue
            if (why := D.programmatic_drop_reason(row)) is not None:
                b.dropped[why] += 1
                continue
            control_dir, twin_dir = root / ep.task, root / cc.twin_id(ep.task)
            if not (control_dir / "task.toml").is_file() or not (twin_dir / "task.toml").is_file():
                b.dropped[DROP_NO_TASK_DIR] += 1
                continue
            control_meta, twin_meta = dp.read_variant_meta(control_dir), dp.read_variant_meta(twin_dir)
            state = surface_state(row, control_meta.target)
            if isinstance(state, str):
                b.dropped[state] += 1
                continue
            t_out, c_out = outputs.get(twin_meta.variant_id), outputs.get(control_meta.variant_id)
            if not (t_out and c_out and t_out.ok and c_out.ok and t_out.current_for(twin_dir)
                    and c_out.current_for(control_dir)):  # fmt: skip
                b.dropped[DROP_NO_OUTPUTS] += 1
                continue
            built = D.build_pair(
                state[0], twin_meta, control_meta, t_out, c_out, phrasing_seen=None, twin_task_dir=str(twin_dir)
            )
            if isinstance(built, str):
                b.dropped[built] += 1
                continue
            if held_out(built):
                b.dropped[DROP_HELD_OUT] += 1
                continue
            if problems := label_problems(built, reward):
                b.dropped[DROP_LABEL + problems[0]] += 1
                continue
            hazard = cc.hazard_of(cc.twin_evidence_text(twin_dir, control_dir))
            b.pairs.append(Pair(built, SCHEMA_HAZARD.get(hazard, SCHEMA_HAZARD_OTHER), state[1]))
            b.cuts[state[1]] += 1
        b.inputs.append({"path": str(path), "kind": "bank", "read": len(rows), "pairs": len(b.pairs) - n_before})
    b.n_sources_checked = len(sources)
    return b


def cap(pairs: Sequence[Pair], cap_per_task: int, dropped: Counter[str]) -> list[Pair]:
    """At most `cap_per_task` pairs per control variant (stable hash of the episode), in input order."""
    by_task: dict[str, list[Pair]] = {}
    for p in pairs:
        by_task.setdefault(p.control_task, []).append(p)
    keep: set[str] = set()
    for group in by_task.values():
        ranked = sorted(group, key=lambda p: cc.stable_hash(f"destructive:{p.built.state.episode_id}"))
        keep |= {p.built.state.episode_id for p in ranked[:cap_per_task]}
        dropped[DROP_CAP] += max(0, len(group) - cap_per_task)
    return [p for p in pairs if p.built.state.episode_id in keep]


def as_commit_example(ex: SFTExample, pair: Pair, outcome: str) -> SFTExample:
    """A pair side as a commit-data example: kind `commit_check`, with the point, outcome and the pair's keys."""
    task = pair.control_task if outcome == PASS else pair.twin_task
    meta = {
        **ex.meta,
        "source": pair.source,
        "task": task,
        "point": DESTRUCTIVE,
        "outcome": outcome,
        "origin": ORIGIN_CONTROL if outcome == PASS else ORIGIN_TWIN,
        "hazard": pair.hazard,
        "cut": pair.cut,
        "targets": [pair.built.state.target],
    }
    return dataclasses.replace(ex, kind=cc.KIND, meta=meta)


def examples_of(pairs: Sequence[Pair]) -> list[SFTExample]:
    out: list[SFTExample] = []
    for p in pairs:
        out += [as_commit_example(p.built.control, p, PASS), as_commit_example(p.built.twin, p, FAIL)]
    return out


def manifest_of(examples: Sequence[SFTExample], cfg: Config, summary: dict[str, Any]) -> BuildManifest:
    """The coverage report's manifest: the control side commits (deletes) after `pass`, the twin holds after `fail`."""
    refs = [
        ExampleRef(e.id, e.meta["task"], e.meta["source"], DESTRUCTIVE, e.meta["outcome"], e.meta["outcome"] == PASS,
                   e.meta["origin"], split="train", hazard=e.meta["hazard"])
        for e in examples
    ]  # fmt: skip
    build = {"inputs": summary["inputs"], "sandbox_outputs": cfg.sandbox_outputs, "examples": summary["examples"]}
    return BuildManifest.of(refs, ManifestQuota(cap_per_task=cfg.cap_per_task), build)


def build(cfg: Config) -> dict[str, Any]:
    collected = collect(cfg)
    kept = cap(collected.pairs, cfg.cap_per_task, collected.dropped)
    examples = examples_of(kept)
    excluded = excluded_from_commit_training(cfg.data_dir)
    n_checked = cc.assert_no_excluded([str(e.meta["source"]) for e in examples], excluded, "examples")
    shapes = Counter(p.built.twin_shape for p in kept)
    summary: dict[str, Any] = {
        "inputs": collected.inputs,
        "excluded_sources_checked": {"inputs": collected.n_sources_checked, "examples": n_checked, "excluded": 0},
        "dropped": dict(sorted(collected.dropped.items())),
        "pairs_before_cap": len(collected.pairs),
        "pairs": len(kept),
        "examples": cc.cell_counts(examples, lambda e: DESTRUCTIVE, lambda e: e.meta["outcome"]),
        "sources": len({p.source for p in kept}),
        "control_variants": len({p.control_task for p in kept}),
        "cuts": dict(sorted(Counter(p.cut for p in kept).items())),
        "twin_shapes": dict(sorted(shapes.items())),
        "surfaces": dict(sorted(Counter(p.built.surface for p in kept).items())),
        "hazards": dict(sorted(Counter(p.hazard for p in kept).items())),
        "labels": {"checked": len(collected.pairs), "failed": sum(
            n for k, n in collected.dropped.items() if k.startswith(DROP_LABEL))},
        "cap_per_task": cfg.cap_per_task,
    }  # fmt: skip
    out = Path(cfg.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "examples.jsonl", "w") as f:
        f.writelines(json.dumps(dataclasses.asdict(e), default=str) + "\n" for e in examples)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_manifest(manifest_of(examples, cfg, summary), out / cc.MANIFEST_FILE)
    (out / "report.md").write_text(render(summary) + "\n")
    return summary


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Destructive-step pairs (sft.destructive_splice)",
        "",
        f"pairs {summary['pairs']} (before the per-task cap {summary['pairs_before_cap']}), examples {summary['examples']}",
        f"sources {summary['sources']}, control variants {summary['control_variants']}",
        f"cuts {summary['cuts']}; twin shapes {summary['twin_shapes']}",
        f"surfaces {summary['surfaces']}; hazards {summary['hazards']}",
        f"labels {summary['labels']}; excluded-source check {summary['excluded_sources_checked']}",
        f"dropped {summary['dropped']}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.sft.destructive_splice", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--bank", action="append", required=True, help="a bank directory (repeatable)")
    ap.add_argument("--tasks-root", required=True, help="the variant task directories of the banked tasks")
    ap.add_argument("--sandbox-outputs", action="append", required=True, help="tasks.decision_probe outputs cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data", help="select_sources.txt, split_seeds.json, the final probe")
    ap.add_argument("--cap-per-task", type=int, default=2)
    ap.add_argument("--drop-excluded", action="store_true", help="drop excluded sources instead of failing")
    args = ap.parse_args(argv)
    cfg = Config(args.bank, args.tasks_root, args.sandbox_outputs, args.out, args.data_dir, args.cap_per_task,
                 args.drop_excluded)  # fmt: skip
    print(render(build(cfg)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
