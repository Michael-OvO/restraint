"""Rejection-sampling fine-tuning on read-first decisions (`kind: read_rft`, iteration 15 G4,
docs/iteration-15-g4-read-rft.md).

The model (G2) is sampled on the training-surface twin / control tasks (`tasks.read_rft_tasks`, `run_eval ... --bank-dir`),
and its OWN episodes are kept only when they show the behaviour wanted, not only the right outcome:

* (a) the evidence was read before the first destructive command on the target or the report
  (`eval.read_diagnosis.read_before_acting`, the rule of the read-rate tables);
* (b) the outcome is right under the eval reward: a twin `abstain_correct` (blocker target matched), a control
  `success_claimed` with the tests passed and no violation; the bank row must also pass the teacher-stage safety filters
  (`teacher_stage.programmatic_drop_reason`: not masked / tampered / damaging, a parsed report, no calls hidden in the
  report turn, non-mutating orientation first) and the hint / malformed-report filters of the decision anchors.

Per task the shortest kept episode stays (fewest assistant turns, then fewest characters as the token proxy, then the
episode id). Then, until stable: `readme` holds at most `readme_max_share` of the kept set (the habit is a README habit,
the other surfaces must carry it), and the larger of twin / control is cut to within `balance_tolerance` of the smaller
(stable hash order, from the surface that has most of that side). Examples are whole own trajectories with loss on every
assistant turn (the conversion of the own-success anchors, `decision.anchor_example`). The G2 anchors are added
(`decision.anchors`, built from `data.bank` as in `judgment_splice`). Fewer than `min_kept` kept examples is an error: stop
rather than train on too little.

    python -m honest_agents.sft.read_rft verify --config configs/sft_read_rft_g4_qwen35.yaml [--runs DIR ...] [--no-anchors]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank
from honest_agents.bank import BankRow
from honest_agents.eval.read_diagnosis import read_before_acting
from honest_agents.reward import compute_reward
from honest_agents.sft import decision as dec
from honest_agents.sft.data import SFTExample, variant_of
from honest_agents.sft.selfdistill import hint_leaks
from honest_agents.sft.teacher_stage import bash_commands, programmatic_drop_reason
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage
from honest_agents.tasks import read_rft_tasks as tasks_list
from honest_agents.tasks.perturb.risk_surfaces import SURFACES_TRAIN, SurfaceSite, site_of_task

logger = logging.getLogger(__name__)

KIND = "read_rft"
SIDE_TWIN = "twin"
SIDE_CONTROL = "control"
CASE_TWIN_OK = "abstain_correct"
CASE_CONTROL_OK = "success_claimed"

DROP_NO_TASK = "task_dir_missing"
DROP_SURFACE = "surface_not_trained"
DROP_WRONG_TWIN = "twin_not_abstain_correct"
DROP_WRONG_CONTROL = "control_not_verified_success"
DROP_NOT_READ = "evidence_not_read_before_acting"
DROP_HINT = "hint_leak_or_echo"
DROP_MALFORMED = "own_report_malformed"
DROP_NOT_SHORTEST = "not_shortest_of_task"
DROP_README_CAP = "readme_cap"
DROP_BALANCE = "side_balance"
DEFAULT_REVIEW = Path("data/decision/review_g4_read_rft.md")


@dataclass(frozen=True)
class ReadRftOptions:
    runs: list[str] = field(
        default_factory=lambda: ["runs/eval/it15_g4_sample/bank"]
    )  # trajectory banks of the sampling run(s)
    tasks_root: str = str(tasks_list.DEFAULT_TASKS_ROOT)  # local task dirs (the bank's task_dir is a node path)
    readme_max_share: float = 0.25  # readme examples / kept examples
    balance_tolerance: float = 0.10  # larger side <= (1 + tolerance) x smaller side
    min_kept: int = 200  # fewer kept examples: refuse (the anchors do not count)
    review_per_group: int = 1  # verify: examples per surface and side in the review file
    forbidden: list[str] = field(
        default_factory=lambda: list(tasks_list.FORBIDDEN_FILES)
    )  # sources that must never appear
    pools: list[str] = field(default_factory=lambda: list(tasks_list.SOURCE_POOLS))  # sources must be in one of these

    def __post_init__(self) -> None:
        if not 0 < self.readme_max_share <= 1:
            raise ValueError("read_rft.readme_max_share must be in (0, 1]")
        if self.balance_tolerance < 0:
            raise ValueError("read_rft.balance_tolerance must be >= 0")
        if self.min_kept < 1:
            raise ValueError("read_rft.min_kept must be >= 1")
        if not self.runs:
            raise ValueError("read_rft.runs needs at least one sampling-run bank")


def parse_options(raw: Mapping[str, Any] | None) -> ReadRftOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(ReadRftOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `read_rft:`; expected {sorted(allowed)}")
    return ReadRftOptions(**raw)


def validate(cfg: SftConfig) -> None:
    parse_options(cfg.read_rft)
    dec.validate(cfg)
    if dec.parse_options(cfg.decision).anchors is None:
        raise ValueError("read_rft needs `decision.anchors` (the G2 anchors)")


# ── selection ────────────────────────────────────────────────────────────────


@dataclass
class Kept:
    row: BankRow
    side: str
    surface: str
    example: SFTExample
    n_turns: int
    chars: int


@dataclass
class Selection:
    kept: list[Kept]
    dropped: Counter[str]
    seen: Counter[tuple[str, str]]  # (surface, side) of every episode that reached the outcome check

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for k in self.kept:
            out.setdefault(k.surface, {SIDE_TWIN: 0, SIDE_CONTROL: 0})[k.side] += 1
        return out


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def task_name(row: BankRow) -> str:
    return Path(str(row.task.get("task_dir") or row.task.get("task_id") or "")).name


def case_of(row: BankRow) -> str:
    return str(row.reward.get("case") or compute_reward(row.episode_outcome(), row.config()).case)


def outcome_ok(row: BankRow, side: str) -> bool:
    o = row.episode_outcome()
    rw = row.reward
    if side == SIDE_TWIN:
        return case_of(row) == CASE_TWIN_OK and rw.get("target_match") is not False
    tests = rw.get("tests_passed", o.tests_passed)
    return case_of(row) == CASE_CONTROL_OK and bool(tests) and not o.violation


def rft_example(row: BankRow, side: str, surface: str) -> SFTExample:
    """The own trajectory, every assistant turn trained: the conversion of the own-success anchors, relabelled."""
    ex = dec.anchor_example(row, dec.ANCHOR_HIGH_RISK)
    meta = {**ex.meta, "side": side, "surface": surface, "task": task_name(row), "sample": row.extra.get("sample")}
    meta.pop("anchor_kind", None)
    return dataclasses.replace(ex, id=f"{row.episode_id}:{KIND}", kind=KIND, meta=meta)


def n_assistant_turns(ex: SFTExample) -> int:
    return sum(m.get("role") == "assistant" for m in ex.messages)


def chars_of(ex: SFTExample) -> int:
    return len(json.dumps(ex.messages, default=str))


def episode_filter(
    row: BankRow, site_of: Callable[[str], SurfaceSite | None], seen: Counter[tuple[str, str]]
) -> tuple[Kept | None, str | None]:
    """(kept episode, None) or (None, the first drop reason)."""
    site = site_of(task_name(row))
    if site is None:
        return None, DROP_NO_TASK
    if site.surface not in SURFACES_TRAIN:
        return None, DROP_SURFACE
    side = SIDE_CONTROL if row.episode_outcome().feasible else SIDE_TWIN
    seen[(site.surface, side)] += 1
    if not outcome_ok(row, side):
        return None, DROP_WRONG_TWIN if side == SIDE_TWIN else DROP_WRONG_CONTROL
    if (reason := programmatic_drop_reason(row)) is not None:
        return None, reason
    if not read_before_acting(bash_commands(row), site):
        return None, DROP_NOT_READ
    ex = rft_example(row, side, site.surface)
    if dec.report_malformed_fields(ex.messages[-1]):
        return None, DROP_MALFORMED
    if hint_leaks(ex.messages) or dec.echoes_hint(ex):
        return None, DROP_HINT
    return Kept(row, side, site.surface, ex, n_assistant_turns(ex), chars_of(ex)), None


def shortest_per_task(kept: Iterable[Kept], dropped: Counter[str]) -> list[Kept]:
    by_task: dict[str, list[Kept]] = defaultdict(list)
    for k in kept:
        by_task[variant_of(k.row)].append(k)
    out = []
    for _, group in sorted(by_task.items()):
        group.sort(key=lambda k: (k.n_turns, k.chars, _hash(k.row.episode_id)))
        out.append(group[0])
        dropped[DROP_NOT_SHORTEST] += len(group) - 1
    return out


def rebalance(kept: list[Kept], opts: ReadRftOptions, dropped: Counter[str]) -> list[Kept]:
    """Cut, one example at a time, until readme <= share and the sides are within the tolerance."""
    kept = list(kept)

    def drop(side: str, surface: str | None, reason: str) -> None:
        pool = [k for k in kept if k.side == side and (surface is None or k.surface == surface)]
        victim = max(pool, key=lambda k: _hash(f"cut|{k.row.episode_id}"))
        kept.remove(victim)
        dropped[reason] += 1

    while kept:
        n = {s: sum(k.side == s for k in kept) for s in (SIDE_TWIN, SIDE_CONTROL)}
        readme = [k for k in kept if k.surface == "readme"]
        larger = max(n, key=lambda s: (n[s], s == SIDE_TWIN))
        smaller = SIDE_CONTROL if larger == SIDE_TWIN else SIDE_TWIN
        if len(readme) > opts.readme_max_share * len(kept):
            side = max((SIDE_TWIN, SIDE_CONTROL), key=lambda s: sum(k.side == s for k in readme))
            drop(side, "readme", DROP_README_CAP)
        elif n[larger] > (1 + opts.balance_tolerance) * n[smaller]:
            by_surface = Counter(k.surface for k in kept if k.side == larger)
            surface = max(by_surface, key=lambda s: (by_surface[s], s))
            drop(larger, surface, DROP_BALANCE)
        else:
            break
    return kept


def select(rows: Iterable[BankRow], opts: ReadRftOptions, site_of: Callable[[str], SurfaceSite | None]) -> Selection:
    dropped: Counter[str] = Counter()
    seen: Counter[tuple[str, str]] = Counter()
    candidates = []
    for row in rows:
        k, reason = episode_filter(row, site_of, seen)
        if k is not None:
            candidates.append(k)
        elif reason is not None:
            dropped[reason] += 1
    kept = rebalance(shortest_per_task(candidates, dropped), opts, dropped)
    kept.sort(key=lambda k: k.row.episode_id)
    return Selection(kept, dropped, seen)


def check_sources(rows: Sequence[BankRow], opts: ReadRftOptions) -> None:
    """The sampled sources are in the RL / distillation pools and never in select / val / test / probe_v1."""
    pairs = {dec.pair_of(r) for r in rows}
    if bad := pairs & tasks_list.pair_ids(opts.forbidden):
        raise ValueError(f"{len(bad)} sampled source(s) are select / val / test / probe sources (e.g. {min(bad)})")
    if bad := pairs - tasks_list.pair_ids(opts.pools):
        raise ValueError(f"{len(bad)} sampled source(s) are in no RL / distillation pool (e.g. {min(bad)})")


def site_lookup(tasks_root: str | Path) -> Callable[[str], SurfaceSite | None]:
    root = Path(tasks_root)

    def site_of(name: str) -> SurfaceSite | None:
        d = root / name
        return site_of_task(d) if (d / "task.toml").is_file() else None

    return site_of


def load_rows(opts: ReadRftOptions, runs: Sequence[str] | None = None) -> list[BankRow]:
    rows: list[BankRow] = []
    for run in runs or opts.runs:
        path = Path(run).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"read_rft sampling bank {path} is missing: run the sample step of the chain first")
        rows += list(bank.iter_bank(path))
    return rows


def build_selection(opts: ReadRftOptions, runs: Sequence[str] | None = None) -> Selection:
    rows = load_rows(opts, runs)
    check_sources(rows, opts)
    return select(rows, opts, site_lookup(opts.tasks_root))


def require_min(sel: Selection, opts: ReadRftOptions) -> None:
    if len(sel.kept) < opts.min_kept:
        raise ValueError(
            f"read_rft kept {len(sel.kept)} examples, fewer than read_rft.min_kept={opts.min_kept}: stop rather than "
            f"train on too little (dropped: {dict(sorted(sel.dropped.items()))})"
        )


# ── stage ────────────────────────────────────────────────────────────────────


def build_anchors(rows: Sequence[BankRow], cfg: SftConfig) -> dec.AnchorSet:
    """The G2 anchors, exactly as `judgment_splice` builds them: over the bank rows, matched to the length of the
    iteration-13/14 decision states (legacy README pairs)."""
    dopts = dec.parse_options(cfg.decision)
    base = dec.build_dataset(rows, dataclasses.replace(dopts, surfaces=None))
    return dec.build_anchors(rows, [], dopts, base.pairs)


def _stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    opts = parse_options(cfg.read_rft)
    sel = build_selection(opts)
    require_min(sel, opts)
    anchors = build_anchors([*fit, *held], cfg)
    train = [k.example for k in sel.kept] + anchors.examples
    info = {
        "read_rft": {
            "kept": len(sel.kept),
            "by_surface": sel.counts(),
            "dropped": dict(sorted(sel.dropped.items())),
            "anchors": anchors.info,
            "examples_train": len(train),
        }
    }
    return StageData(train, [], info)


register_stage(KIND, _stage, validate)


# ── verify ───────────────────────────────────────────────────────────────────


def render_turn(m: Mapping[str, Any]) -> str:
    lines = [str(m["content"])] if m.get("content") else []
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        lines.append(f"<{fn.get('name')}> " + json.dumps(fn.get("arguments"), ensure_ascii=False))
    return "\n".join(lines) or "(empty)"


def write_review(sel: Selection, path: str | Path, per_group: int = 1) -> Path:
    groups: dict[tuple[str, str], list[Kept]] = defaultdict(list)
    for k in sel.kept:
        groups[(k.surface, k.side)].append(k)
    out = ["# G4 read-RFT review: kept own episodes", ""]
    for (surface, side), ks in sorted(groups.items()):
        for k in sorted(ks, key=lambda k: _hash(k.row.episode_id))[:per_group]:
            out += [
                f"## {surface} / {side} / {k.row.episode_id} (task {task_name(k.row)}, {k.n_turns} assistant turns)",
                "",
            ]
            for i, m in enumerate(k.example.messages):
                if m.get("role") == "system":
                    continue
                tag = "TRAINED " if i in k.example.loss_messages else ""
                out += [f"**{tag}{m.get('role')}**", "```", render_turn(m)[:1500], "```", ""]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p


def verify(
    cfg: SftConfig, *, runs: Sequence[str] | None = None, review: str | Path | None = None, anchors: bool = True
) -> dict[str, Any]:
    from honest_agents.sft import data

    opts = parse_options(cfg.read_rft)
    sel = build_selection(opts, runs)
    checks = [
        {"name": f"at least {opts.min_kept} kept", "passed": len(sel.kept) >= opts.min_kept, "detail": len(sel.kept)},
    ]
    counts = sel.counts()
    n = {s: sum(c[s] for c in counts.values()) for s in (SIDE_TWIN, SIDE_CONTROL)}
    lo, hi = sorted(n.values())
    checks.append(
        {
            "name": f"twin / control within {opts.balance_tolerance:.0%}",
            "passed": hi <= (1 + opts.balance_tolerance) * lo,
            "detail": n,
        }
    )
    readme = sum(counts.get("readme", {}).values())
    checks.append(
        {
            "name": f"readme <= {opts.readme_max_share:.0%}",
            "passed": readme <= opts.readme_max_share * max(1, len(sel.kept)),
            "detail": readme,
        }
    )
    info: dict[str, Any] = {}
    if anchors:
        sel_rows = data.read_rows(cfg.data.bank, data.RowFilter(**cfg.data.row_filter))
        fit, held = data.partition(
            sel_rows.usable, cfg.data.holdout_fraction, cfg.seed, data.PARTITION_KEYS[cfg.data.partition_by]
        )
        a = build_anchors([*fit, *held], cfg)
        info["anchors"] = a.info
        want = dec.parse_options(cfg.decision).anchors.count  # type: ignore[union-attr]
        checks.append({"name": f"{want} anchors", "passed": len(a.examples) >= (want or 0), "detail": len(a.examples)})
    review_path = write_review(sel, review, opts.review_per_group) if review else None
    return {
        "verdict": "PASS" if all(c["passed"] for c in checks) else "FAIL",
        "checks": checks,
        "kept": len(sel.kept),
        "by_surface": counts,
        "seen_by_surface_side": {f"{s}/{k}": v for (s, k), v in sorted(sel.seen.items())},
        "dropped": dict(sorted(sel.dropped.items())),
        "review": str(review_path) if review_path else None,
        **info,
    }


def render_verify(report: Mapping[str, Any]) -> str:
    lines = [f"read_rft verify: {report['verdict']}  (kept {report['kept']})", ""]
    lines.append("kept by surface (twin / control):")
    for s, c in sorted(report["by_surface"].items()):
        lines.append(f"  {s:15s} {c[SIDE_TWIN]:4d} / {c[SIDE_CONTROL]:4d}")
    lines.append("episodes seen by surface/side: " + json.dumps(report["seen_by_surface_side"]))
    lines.append("drop reasons:")
    lines += [f"  {k:45s} {v}" for k, v in sorted(report["dropped"].items(), key=lambda kv: -kv[1])]
    lines.append("checks:")
    lines += [f"  [{'ok' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in report["checks"]]
    if report.get("review"):
        lines.append(f"review: {report['review']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from honest_agents.sft.train_sft import load_config

    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.read_rft", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("verify",), nargs="?", default="verify")
    ap.add_argument("--config", required=True, help="YAML config with kind: read_rft")
    ap.add_argument("--runs", nargs="+", help="sampling-run bank dirs (default: read_rft.runs)")
    ap.add_argument("--review", default=str(DEFAULT_REVIEW), help="review file")
    ap.add_argument("--no-anchors", action="store_true", help="skip the anchor build (needs data.bank)")
    ap.add_argument("--json", help="also write the report here")
    ap.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    report = verify(cfg, runs=args.runs, review=args.review, anchors=not args.no_anchors)
    print(render_verify(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from honest_agents.sft.read_rft import (
        main as _main,  # `python -m` runs this file as __main__; use the imported copy
    )

    sys.exit(_main())
