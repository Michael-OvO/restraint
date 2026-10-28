"""The confidence-only bandit of iteration 17 (docs/iteration-17-design.md section 3): data, prompts and reward.

An item is one archived episode of the model's own, cut inside its own report at the value of `confidence` (the
report's probability that the task is fully and correctly done). The trainer samples only that value and scores it
against the episode's graded outcome; nothing else of the episode or the report is trained.

* **Items** (`item_of`): a bank row whose report is one `submit_report` call (no extra calls in the report turn; an
  `abstain` report states the complementary probability and is left out, counted), on a training-pool source
  (`tasks.select_split.excluded_from_commit_training`: select, val, test and the final decision probe's sources,
  DEF-63; an error unless `drop_excluded`), split `train`, not masked. Label y = `passed` of
  `reward.compute_reward` under the row's own reward config (hidden tests pass, no violation, no tamper; always 0 on
  an infeasible variant).
* **Prompt** (`prefix_messages`, `render_prefix`): the episode's messages and its report message with the report's
  arguments in their own order, `confidence` moved last (where the model writes it in most reports; the number moved
  is recorded per item) and its value replaced by a sentinel. The conversation is rendered with the chat template the
  model is served with (tools and template kwargs of `sft.data.make_renderer`), and the text is cut right before the
  sentinel, so the template's own markup for a tool-call argument precedes the sampled value. Exactly one sentinel
  must be found, or the item is refused.
* **Completion** (`parse_confidence`): the sampled text up to the template's end of the argument (the first newline
  for Qwen3.5's `<parameter=...>` blocks) must be a number in [0, 1]; anything else is unparsable.
* **Reward** (`brier_reward`): `-(c - y)^2`, and `-1` (the worst Brier) for an unparsable completion.

    uv run python -m honest_agents.sft.confidence_bandit build --bank DIR [--bank DIR ...] --out items.jsonl \\
        [--data-dir data] [--drop-excluded]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.bank import BankRow
from honest_agents.pilot.dataset import normalize_message
from honest_agents.reward import compute_reward
from honest_agents.sft.data import pair_of, variant_of
from honest_agents.tasks.select_split import excluded_from_commit_training

REPORT_TOOL = "submit_report"
ABSTAIN_TOOL = "abstain"
CONFIDENCE_KEY = "confidence"
SENTINEL = "⁣CONFIDENCE⁣"  # never in a report: invisible separators around a word
WORST_BRIER = 1.0

DROP_EXCLUDED = "excluded_source"
DROP_NOT_TRAIN = "not_train_split"
DROP_MASKED = "masked"
DROP_NO_REPORT = "no_report_call"
DROP_ABSTAIN = "abstain_report"
DROP_EXTRA_CALLS = "calls_in_report_turn"
DROP_UNPARSED_ARGS = "report_arguments_not_a_mapping"

_NUMBER = re.compile(r"\s*([01](?:\.\d+)?|\.\d+)\s*$")


class ExcludedSourceError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str  # the run's episode id
    task: str
    source: str
    family: str
    feasible: bool
    is_control: bool
    status: str | None  # the report's own status
    y: int  # 1 = passed
    original_confidence: float | None  # what the model wrote (None: left out)
    confidence_moved: bool  # `confidence` was not the last argument and was moved there
    messages: list[dict[str, Any]] = field(repr=False)  # prompt + transcript + the report message with the sentinel

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Item:
        return cls(**d)


def _report_args(row: BankRow) -> dict[str, Any] | None:
    args = (row.report_call or {}).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return dict(args) if isinstance(args, dict) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            return _number(float(value))
        except ValueError:
            return None
    return None


def prefix_messages(row: BankRow, args: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """The episode's messages with its report message rewritten: `confidence` last, its value the sentinel."""
    keys = list(args)
    moved = CONFIDENCE_KEY in keys and keys[-1] != CONFIDENCE_KEY
    new_args = {k: v for k, v in args.items() if k != CONFIDENCE_KEY} | {CONFIDENCE_KEY: SENTINEL}
    report = normalize_message(dict(row.report_message or {}))
    call = dict((report.get("tool_calls") or [])[0])
    call["function"] = {**dict(call.get("function") or {}), "name": REPORT_TOOL, "arguments": new_args}
    report["tool_calls"] = [call]
    messages = [normalize_message(dict(m)) for m in [*row.prompt, *row.completion]]
    return [*messages, report], moved


def item_of(row: BankRow, excluded: set[str], *, drop_excluded: bool = False) -> Item | str:
    """The bandit item of one bank row, or the reason it has none."""
    source = pair_of(row)
    if source in excluded:
        if not drop_excluded:
            raise ExcludedSourceError(f"episode {row.episode_id} has excluded source {source}")
        return DROP_EXCLUDED
    if (row.task or {}).get("split") not in (None, "train"):
        return DROP_NOT_TRAIN
    o = row.episode_outcome()
    if o.masked_reason:
        return DROP_MASKED
    name = (row.report_call or {}).get("name")
    if name == ABSTAIN_TOOL:
        return DROP_ABSTAIN
    if name != REPORT_TOOL or row.report_message is None:
        return DROP_NO_REPORT
    if row.report_turn_extra_calls or len(row.report_message.get("tool_calls") or []) != 1:
        return DROP_EXTRA_CALLS
    args = _report_args(row)
    if args is None:
        return DROP_UNPARSED_ARGS
    messages, moved = prefix_messages(row, args)
    passed = compute_reward(o, row.config()).passed
    return Item(
        id=str(row.extra.get("episode_id") or row.episode_id),
        task=variant_of(row),
        source=source,
        family=str((row.task or {}).get("family") or ""),
        feasible=bool(o.feasible),
        is_control=bool(o.is_control),
        status=None if args.get("status") is None else str(args["status"]),
        y=int(bool(passed)),
        original_confidence=_number(args.get(CONFIDENCE_KEY)),
        confidence_moved=moved,
        messages=messages,
    )


def render_prefix(messages: Sequence[dict[str, Any]], tokenizer: Any, tools: Any, kwargs: dict[str, Any]) -> str:
    """The prompt text: the conversation as the served template renders it, cut right before the sentinel."""
    text = tokenizer.apply_chat_template(list(messages), tools=tools, tokenize=False, **kwargs)
    if text.count(SENTINEL) != 1:
        raise ValueError(f"the sentinel occurs {text.count(SENTINEL)} times in the rendered prompt")
    return text[: text.index(SENTINEL)]


def parse_confidence(completion: str) -> float | None:
    """The sampled value: the text up to the end of the argument (first newline), a number in [0, 1]."""
    head = completion.split("\n", 1)[0]
    m = _NUMBER.match(head)
    if m is None:
        return None
    value = float(m.group(1))
    return value if 0.0 <= value <= 1.0 else None


def brier_reward(confidence: float | None, y: int) -> float:
    return -WORST_BRIER if confidence is None else -((confidence - y) ** 2)


@dataclass
class Build:
    items: list[Item]
    dropped: Counter[str]
    inputs: list[dict[str, Any]]


def build(banks: Iterable[str | Path], data_dir: str | Path = "data", *, drop_excluded: bool = False) -> Build:
    excluded = excluded_from_commit_training(data_dir)
    items: list[Item] = []
    dropped: Counter[str] = Counter()
    inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in banks:
        n_before = len(items)
        rows = list(bank_mod.iter_bank(Path(path).expanduser()))
        for row in rows:
            got = item_of(row, excluded, drop_excluded=drop_excluded)
            if isinstance(got, str):
                dropped[got] += 1
            elif got.id in seen:
                dropped["duplicate_episode"] += 1
            else:
                seen.add(got.id)
                items.append(got)
        inputs.append({"path": str(path), "rows": len(rows), "items": len(items) - n_before})
    return Build(items, dropped, inputs)


def summary(b: Build) -> dict[str, Any]:
    claims = [i for i in b.items if i.feasible and i.status == "complete"]
    return {
        "inputs": b.inputs,
        "items": len(b.items),
        "sources": len({i.source for i in b.items}),
        "dropped": dict(sorted(b.dropped.items())),
        "by_cell": dict(
            sorted(Counter(f"{'feasible' if i.feasible else 'infeasible'}/{i.status}/y={i.y}" for i in b.items).items())
        ),
        "feasible_claims": {"n": len(claims), "pass": sum(i.y for i in claims), "fail": sum(1 - i.y for i in claims)},
        "original_confidence": dict(Counter(str(i.original_confidence) for i in b.items).most_common(8)),
        "confidence_moved": sum(i.confidence_moved for i in b.items),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.sft.confidence_bandit", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="bandit items from banks")
    b.add_argument("--bank", action="append", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--data-dir", default="data")
    b.add_argument("--drop-excluded", action="store_true")
    args = ap.parse_args(argv)
    got = build(args.bank, args.data_dir, drop_excluded=args.drop_excluded)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(i.to_json(), default=str) + "\n" for i in got.items)
    s = summary(got)
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
