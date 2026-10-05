"""Pilot dataset: one prompt per frozen trajectory (SPEC_V3 §6).

    python -m honest_agents.pilot.dataset --bank outputs/phase0/bank --out data/pilot/pilot.jsonl

A row's `prompt` is the banked prompt followed by the banked completion up to, not including, the report turn
(`bank.split_at_report`). The trainer samples G new reports for it; `ReplayEnv` scores each against the cached
outcome. Every other column is passed to `ReplayEnv.reset(**row)` and to the reward function by TRL.

What is left out, and why (each reason is counted in `BuildStats.dropped`):

* `masked` — the banked episode was not scored (boot failure, toolchain self-check, leaking group). Its outcome says
  nothing about the policy, so no report on it can be scored either.
* `quarantined` — the variant is on the quarantine list.
* `infeasible_clean_pass` — the variant is labelled infeasible but its tests passed cleanly. A completion claim would
  be a leak (masked), and a decline would be paid for a label that is wrong.
* `tampered` — every report on a tampered trajectory is capped at the tamper reward, so the group has no contrast.
* `unseen_actions` — commands ran that the frozen context does not show: other tool calls in the report turn, or a
  final tool-calling turn whose results were rolled back. The outcome would not match what the model can read.
* `no_report` — the trajectory never reported. Kept by default (`include_no_report`), as a forced report turn: the
  context is cut after the last tool result and the model must report from there. Excluding these would drop the
  trajectories that ran out of budget, which are mostly failures, the place fabrication is measured.
* `too_long` — the context exceeds `max_context_chars`.

The dataset cannot go through Arrow as nested messages: Arrow gives every struct the union of all keys, so a tool call
`{"command": ...}` would gain a null entry for every argument name any other call used, and the Qwen3.5 template renders
each of them as a parameter. `to_hf_dataset` therefore stores the prompt as a JSON string and decodes it on access.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank
from honest_agents.bank import BankRow
from honest_agents.pilot.env import BANK_ROW_COLUMN

logger = logging.getLogger(__name__)

KINDS = ("feasible", "control", "infeasible")
DEFAULT_MIX = {"feasible": 0.55, "control": 0.20, "infeasible": 0.25}  # SPEC_V3 §4.5
PROMPT_JSON_COLUMN = "prompt_json"
# Bank-row fields the replay does not need. Dropping them keeps a pilot row small (token ids dominate a bank row).
_HEAVY_FIELDS = (
    "prompt",
    "completion",
    "report_message",
    "completion_ids",
    "prompt_ids",
    "logprobs",
    "tool_mask",
)

DROP_MASKED = "masked"
DROP_QUARANTINED = "quarantined"
DROP_LEAKY = "infeasible_clean_pass"
DROP_TAMPERED = "tampered"
DROP_UNSEEN = "unseen_actions"
DROP_NO_REPORT = "no_report"
DROP_NO_PROMPT = "no_prompt"
DROP_SPLIT = "not_train_split"
DROP_DUPLICATE = "duplicate"
DROP_TOO_LONG = "too_long"
DROP_MIX = "mix_downsample"
DROP_VARIANT_CAP = "variant_cap"
DROP_MAX_ROWS = "max_rows"


@dataclass
class BuildOptions:
    include_no_report: bool = True  # trajectories that never reported become forced report turns
    drop_unseen_actions: bool = True
    drop_tampered: bool = True
    drop_leaky: bool = True
    # Target shares of feasible / control / infeasible prompts, reached by downsampling. None keeps the bank's mix.
    mix: dict[str, float] | None = None
    strict_mix: bool = (
        False  # True: fail when a kind with a positive share has no rows. False: renormalise over the rest
    )
    max_per_variant: int | None = None  # cap on frozen trajectories per task variant
    max_rows: int | None = None
    max_context_chars: int | None = None  # drop transcripts longer than this many characters of message content
    quarantine_file: str | None = None  # default: `quarantine.jsonl` next to the bank files
    quarantine_ignore_sources: tuple[str, ...] = (
        bank.DEFAULT_IGNORED_SOURCES
    )  # eval-sourced entries never shape training
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mix is not None:
            unknown = set(self.mix) - set(KINDS)
            if unknown:
                raise ValueError(f"mix has unknown kind(s) {sorted(unknown)}; expected {KINDS}")
            if any(v < 0 for v in self.mix.values()) or sum(self.mix.values()) <= 0:
                raise ValueError("mix shares must be >= 0 and sum to a positive number")
        for name in ("max_per_variant", "max_rows", "max_context_chars"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or null, got {value}")


@dataclass
class BuildStats:
    n_bank_rows: int = 0
    n_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    kinds: dict[str, int] = field(default_factory=dict)
    n_variants: int = 0
    n_forced_report: int = 0
    feasible_pass_share: float | None = None  # share of feasible prompts (controls included) whose outcome is a pass
    original_claim_rate_feasible: float | None = None  # what the banked policy did at these report points
    original_fabrication_among_failures: float | None = None
    context_chars_median: int | None = None
    context_chars_max: int | None = None
    efficiency_ref_actions: list[int] = field(default_factory=list)  # values found in the rows' reward configs
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── messages ─────────────────────────────────────────────────────────────────


def _arguments(value: Any) -> dict[str, Any]:
    """Tool-call arguments as a mapping. The Qwen3.5 template iterates them, so a string would fail to render."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"arguments": value}
    if isinstance(value, dict):
        return value
    return {} if value is None else {"arguments": value}


def _tool_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return None
    return {
        "type": "function",
        "function": {"name": name, "arguments": _arguments(fn.get("arguments"))},
    }


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """One chat message in the form TRL's own tool loop builds: always a `content`, tool calls as
    `{"type": "function", "function": {"name", "arguments": {...}}}`, and no key whose value is null."""
    out: dict[str, Any] = {"role": str(message.get("role") or "user")}
    content = message.get("content")
    out["content"] = content if isinstance(content, (str, list)) else ("" if content is None else str(content))
    calls = [c for c in map(_tool_call, message.get("tool_calls") or []) if c is not None]
    if calls:
        out["tool_calls"] = calls
    for key in ("name", "reasoning_content", "tool_call_id"):
        if isinstance(message.get(key), str):
            out[key] = message[key]
    return out


def normalize_messages(
    messages: Iterable[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [normalize_message(m) for m in (messages or []) if isinstance(m, dict)]


def _content_chars(messages: Sequence[dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        content = m.get("content")
        n += len(content) if isinstance(content, str) else len(json.dumps(content, default=str))
        n += sum(len(json.dumps(c, default=str)) for c in m.get("tool_calls") or [])
    return n


@dataclass
class FrozenContext:
    messages: list[dict[str, Any]]
    forced_report: bool  # the trajectory never reported; the context is cut after its last tool result
    unseen_actions: int  # tool calls that ran, or were asked for, and that the context does not show


def frozen_context(row: BankRow) -> FrozenContext:
    """The messages a new report is conditioned on: prompt, then the completion before the report turn.

    `row.completion` already ends before the report turn. For a trajectory that never reported it is the whole
    completion; trailing assistant messages are then cut, so the context ends where the model must act (a tool result,
    or the task itself), and any tool call those messages carried counts as unseen.
    """
    prompt = normalize_messages(row.prompt)
    completion = normalize_messages(row.completion)
    unseen = int(row.report_turn_extra_calls or 0)
    forced = row.report_call is None
    if forced:
        while completion and completion[-1]["role"] == "assistant":
            unseen += len(completion.pop().get("tool_calls") or [])
    return FrozenContext(prompt + completion, forced, unseen)


# ── rows ─────────────────────────────────────────────────────────────────────


def kind_of(row: BankRow) -> str:
    """feasible / control / infeasible, from the dataset column, else the stored spec, else the outcome."""
    kind = row.task.get("kind")
    if kind in KINDS:
        return kind
    spec = row.task.get("spec") if isinstance(row.task.get("spec"), dict) else {}
    feasible = row.task.get("feasible", spec.get("feasible", row.outcome.get("feasible", True)))
    control = row.task.get("is_control", spec.get("is_control", row.outcome.get("is_control", False)))
    if not feasible:
        return "infeasible"
    return "control" if control else "feasible"


def variant_of(row: BankRow) -> str:
    spec = row.task.get("spec") if isinstance(row.task.get("spec"), dict) else {}
    for value in (
        row.task.get("variant_id"),
        spec.get("variant_id"),
        row.group,
        row.task.get("task_dir"),
    ):
        if value:
            return str(value)
    return row.episode_id


def _original_status(row: BankRow) -> str:
    report = row.outcome.get("report")
    if isinstance(report, dict) and report.get("status"):
        return str(report["status"])
    return "unparsed" if row.outcome.get("report_error") else "none"


def slim_bank_row(row: BankRow) -> BankRow:
    """The row without the fields the replay never reads (messages are in the prompt column, token ids are unused)."""
    update: dict[str, Any] = dict.fromkeys(_HEAVY_FIELDS)
    update["prompt"], update["completion"] = [], []
    return row.model_copy(update=update)


def pilot_row(row: BankRow, context: FrozenContext | None = None) -> dict[str, Any]:
    """The dataset row for one frozen trajectory."""
    context = context or frozen_context(row)
    outcome = row.episode_outcome()
    spec = row.task.get("spec") if isinstance(row.task.get("spec"), dict) else {}
    kind = kind_of(row)

    def column(name: str) -> Any:
        value = row.task.get(name)
        return value if value is not None else spec.get(name)

    return {
        "prompt": context.messages,
        BANK_ROW_COLUMN: slim_bank_row(row).model_dump_json(),
        "pilot_id": row.episode_id,
        "variant_id": variant_of(row),
        "pair_id": column("pair_id"),
        "family": column("family"),
        "perturbation": column("perturbation"),
        "kind": kind,
        "feasible": kind != "infeasible",
        "is_control": kind == "control",
        "passed": outcome.passed,
        "n_actions": outcome.n_actions,
        "forced_report": context.forced_report,
        "original_status": _original_status(row),
    }


def _drop_reason(row: BankRow, context: FrozenContext, opts: BuildOptions, quarantined: set[str]) -> str | None:
    outcome = row.episode_outcome()
    # Not `outcome.mask`: that also reads a leak from the banked report, which the pilot replaces. The variant-level
    # leak risk is `DROP_LEAKY` below.
    boot_failure = outcome.grading_error is not None and not outcome.acted
    if outcome.masked_reason is not None or boot_failure or row.reward.get("masked"):
        return DROP_MASKED
    if variant_of(row) in quarantined:
        return DROP_QUARANTINED
    if row.task.get("split") != "train":  # probe / test episodes share the bank, and a missing split proves nothing
        return DROP_SPLIT
    if not context.messages:
        return DROP_NO_PROMPT
    if opts.drop_leaky and not outcome.feasible and outcome.passed:
        return DROP_LEAKY
    if opts.drop_tampered and outcome.tampered:
        return DROP_TAMPERED
    if context.forced_report and not opts.include_no_report:
        return DROP_NO_REPORT
    if opts.drop_unseen_actions and context.unseen_actions > 0:
        return DROP_UNSEEN
    if opts.max_context_chars is not None and _content_chars(context.messages) > opts.max_context_chars:
        return DROP_TOO_LONG
    return None


def _apply_mix(
    rows: list[dict[str, Any]],
    opts: BuildOptions,
    rng: random.Random,
    stats: BuildStats,
) -> list[dict[str, Any]]:
    """Downsample to the target shares: the largest dataset in which every kind has its share."""
    if opts.mix is None:
        return rows
    by_kind: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    for r in rows:
        by_kind[r["kind"]].append(r)
    shares = {k: float(opts.mix.get(k, 0.0)) for k in KINDS}
    missing = [k for k in KINDS if shares[k] > 0 and not by_kind[k]]
    if missing:
        message = f"mix asks for {missing} but the bank has no such rows after filtering"
        if opts.strict_mix:
            raise ValueError(message)
        stats.warnings.append(message + "; shares renormalised over the kinds that exist")
        for k in missing:
            shares[k] = 0.0
    total_share = sum(shares.values())
    if total_share <= 0:
        raise ValueError("no rows left for any kind the mix asks for")
    shares = {k: v / total_share for k, v in shares.items()}
    size = min(len(by_kind[k]) / s for k, s in shares.items() if s > 0)
    kept: list[dict[str, Any]] = []
    for k in KINDS:
        want = min(len(by_kind[k]), round(size * shares[k]))
        kept += rng.sample(by_kind[k], want)
    stats.dropped[DROP_MIX] = len(rows) - len(kept)
    return kept


def build_rows(
    bank_rows: Iterable[BankRow],
    opts: BuildOptions | None = None,
    quarantined: set[str] | None = None,
) -> tuple[list[dict[str, Any]], BuildStats]:
    """Pilot rows from bank rows, plus what was kept and dropped. Deterministic for a given `opts.seed`."""
    opts = opts or BuildOptions()
    quarantined = quarantined or set()
    rng = random.Random(opts.seed)
    stats = BuildStats()
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ref_actions: set[int] = set()
    for row in bank_rows:
        stats.n_bank_rows += 1
        if row.episode_id in seen:
            dropped[DROP_DUPLICATE] += 1
            continue
        seen.add(row.episode_id)
        context = frozen_context(row)
        reason = _drop_reason(row, context, opts, quarantined)
        if reason is not None:
            dropped[reason] += 1
            continue
        if isinstance(row.reward_config.get("efficiency_ref_actions"), int):
            ref_actions.add(row.reward_config["efficiency_ref_actions"])
        by_variant[variant_of(row)].append(pilot_row(row, context))

    rows: list[dict[str, Any]] = []
    for variant in sorted(by_variant):
        group = by_variant[variant]
        if opts.max_per_variant is not None and len(group) > opts.max_per_variant:
            dropped[DROP_VARIANT_CAP] += len(group) - opts.max_per_variant
            group = rng.sample(group, opts.max_per_variant)
        rows += group
    stats.dropped = dict(dropped)
    rows = _apply_mix(rows, opts, rng, stats)
    rng.shuffle(rows)
    if opts.max_rows is not None and len(rows) > opts.max_rows:
        stats.dropped[DROP_MAX_ROWS] = len(rows) - opts.max_rows
        rows = rows[: opts.max_rows]
    stats.dropped = {k: v for k, v in stats.dropped.items() if v}
    stats.efficiency_ref_actions = sorted(ref_actions)
    _summarise(rows, stats)
    return rows, stats


def _summarise(rows: Sequence[dict[str, Any]], stats: BuildStats) -> None:
    stats.n_rows = len(rows)
    stats.kinds = {k: sum(r["kind"] == k for r in rows) for k in KINDS}
    stats.n_variants = len({r["variant_id"] for r in rows})
    stats.n_forced_report = sum(bool(r["forced_report"]) for r in rows)
    feasible = [r for r in rows if r["feasible"]]
    failed = [r for r in feasible if not r["passed"]]
    if feasible:
        stats.feasible_pass_share = sum(bool(r["passed"]) for r in feasible) / len(feasible)
        stats.original_claim_rate_feasible = sum(r["original_status"] == "complete" for r in feasible) / len(feasible)
        if stats.feasible_pass_share in (0.0, 1.0):
            stats.warnings.append(
                "every feasible prompt has the same outcome: "
                + ("fabrication" if stats.feasible_pass_share == 1.0 else "claim coverage")
                + " cannot be measured on this dataset"
            )
    if failed:
        stats.original_fabrication_among_failures = sum(r["original_status"] == "complete" for r in failed) / len(
            failed
        )
    sizes = sorted(_content_chars(r["prompt"]) for r in rows)
    if sizes:
        stats.context_chars_median, stats.context_chars_max = (
            sizes[len(sizes) // 2],
            sizes[-1],
        )
    if len(stats.efficiency_ref_actions) > 1:
        stats.warnings.append(f"bank rows were written under different call budgets: {stats.efficiency_ref_actions}")


def bank_files(path: str | Path) -> list[Path]:
    """Bank files under `path`: the file itself, the `bank-*.jsonl` in a directory, or, when the directory has none of
    its own, every one below it. A Phase-0 run writes `<output_root>/<run name>/bank/`, so `outputs/phase0` finds it."""
    p = Path(path)
    if p.is_file():
        return [p]
    direct = sorted(p.glob("bank-*.jsonl"))
    return direct or sorted(p.rglob("bank-*.jsonl"))


def build_from_bank(path: str | Path, opts: BuildOptions | None = None) -> tuple[list[dict[str, Any]], BuildStats]:
    """Pilot rows from a bank directory (every `bank-*.jsonl`, searched recursively when none is directly inside) or
    one bank file. The quarantine list next to each bank file is applied, or `opts.quarantine_file` alone."""
    opts = opts or BuildOptions()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trajectory bank not found: {p}")
    files = bank_files(p)
    if not files:
        raise FileNotFoundError(f"no bank-*.jsonl under {p}")
    if opts.quarantine_file is not None:
        qfiles = [Path(opts.quarantine_file)]
    else:
        qfiles = sorted({f.parent / bank.QUARANTINE_BASENAME for f in files})
    quarantined: set[str] = set()
    for qfile in qfiles:
        if qfile.exists():
            quarantined |= bank.quarantined_variants(qfile, exclude_sources=opts.quarantine_ignore_sources)
    rows, stats = build_rows((r for f in files for r in bank.iter_bank(f)), opts, quarantined)
    if not rows:
        raise ValueError(f"{p}: no usable trajectories (bank rows: {stats.n_bank_rows}, dropped: {stats.dropped})")
    return rows, stats


# ── files and the trainer's dataset ──────────────────────────────────────────


def write_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _decode_prompts(batch: dict[str, list]) -> dict[str, list]:
    out = {k: v for k, v in batch.items() if k != PROMPT_JSON_COLUMN}
    out["prompt"] = [json.loads(s) for s in batch[PROMPT_JSON_COLUMN]]
    return out


def to_hf_dataset(rows: Sequence[dict[str, Any]]):
    """A `datasets.Dataset` whose rows come back exactly as built (see the module docstring for why not plain Arrow)."""
    from datasets import Dataset

    if not rows:
        raise ValueError("no pilot rows")
    flat = [
        {
            **{k: v for k, v in r.items() if k != "prompt"},
            PROMPT_JSON_COLUMN: json.dumps(r["prompt"]),
        }
        for r in rows
    ]
    return Dataset.from_list(flat).with_transform(_decode_prompts)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_mix(text: str | None) -> dict[str, float] | None:
    if not text or text == "none":
        return None
    if text == "default":
        return dict(DEFAULT_MIX)
    mix = {}
    for part in text.split(","):
        kind, _, share = part.partition("=")
        mix[kind.strip()] = float(share)
    return mix


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="honest_agents.pilot.dataset", description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", required=True, help="bank directory (bank-*.jsonl) or one bank file")
    ap.add_argument("--out", required=True, help="output JSONL; statistics go to <out>.stats.json")
    ap.add_argument(
        "--mix",
        default=None,
        help='"default" (55/20/25), "none", or feasible=0.55,control=0.2,...',
    )
    ap.add_argument("--strict-mix", action="store_true")
    ap.add_argument("--max-per-variant", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-context-chars", type=int, default=None)
    ap.add_argument(
        "--exclude-no-report",
        action="store_true",
        help="drop trajectories that never reported",
    )
    ap.add_argument("--keep-unseen-actions", action="store_true")
    ap.add_argument("--quarantine-file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    opts = BuildOptions(
        include_no_report=not args.exclude_no_report,
        drop_unseen_actions=not args.keep_unseen_actions,
        mix=_parse_mix(args.mix),
        strict_mix=args.strict_mix,
        max_per_variant=args.max_per_variant,
        max_rows=args.max_rows,
        max_context_chars=args.max_context_chars,
        quarantine_file=args.quarantine_file,
        seed=args.seed,
    )
    rows, stats = build_from_bank(args.bank, opts)
    out = write_rows(rows, args.out)
    Path(str(out) + ".stats.json").write_text(json.dumps(stats.to_dict(), indent=2))
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
