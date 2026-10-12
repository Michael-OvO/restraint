"""Context distillation of hinted base-model rollouts (`kind: selfdistill`, docs/iteration-12-design.md sections 2.3-2.4).

The base model is sampled on the distillation pool (`tasks.distill_pool`) with a hint appended to its system prompt
(`eval/prompts/hint_tier1.txt`, or tier 2), through `run_eval --bank-dir`. This stage keeps the rollouts worth imitating
and trains them WITHOUT the hint: the stored system message is replaced by `harness.SYSTEM_PROMPT` before rendering, so
the policy learns to do unprompted what it did when prompted. Why the base model's own rollouts and not the teacher's:
iteration 11's teacher SFT copied the teacher's short trajectories without its skill (verified success 69 -> 22 %); here
the commands, their granularity, the error rate and the recovery are the base model's own, and only the hint-induced
decisions change.

Selection, each drop counted under the first reason that applies (`row_drop_reason`):

1. The iteration-11 objective filters, reused from `sft.teacher_stage.programmatic_drop_reason` (the audit is not used):
   train split; not masked; a parsable report with no hidden calls in its turn; no tamper, self-inflicted damage or
   violation; an impossible variant that passed is a leak; feasible rows and controls only `success_claimed`, twins only
   `abstain_correct`; a read-only command before the first mutation (`mutation.first_mutation_index`).
2. **README before the decision** (high_risk rows, twin AND control; `readme_read.readme_read_before_decision`): the
   README was read before the first destructive command on the sentinel directory, or before the report when nothing
   was destroyed. A control that deletes the directory without reading its benign README is the behaviour that makes
   "notes -> stop" and "no notes -> delete" indistinguishable; only read -> warning -> stop and read -> benign -> proceed
   are imitated.
3. **Hint echo** (`hint_echo`): assistant text (message content, reasoning, report arguments; not bash commands) that
   mentions instructions, guidelines, "as directed", "as instructed", "the notes rule" or the system prompt. Such a
   trajectory justifies its decision by a prompt the trained policy will not see; imitated without the hint, it teaches
   the model to cite rules it was never given. Required by the go/no-go review; the count is reported either way. The
   words also catch the model citing the task itself ("remove /data/x as instructed": ~8 % of the base model's
   high_risk control successes), which this filter drops as well; `hint_echo_pattern` narrows it if that costs too
   many controls.
4. **Paired contrast** (`no_paired_control`): a twin is kept only if its own control (same `pair_id` and family, kind
   control) has a kept success, looked up over the fit AND held-out rows (the variant partition may separate them). The
   iteration-5 combo SFT taught decline-everything (feasible declines 1.4 -> 20 %); an abstention next to its control's
   success teaches the conditional instead.

Balance (fit examples only; held-out examples are filtered the same way, not balanced): at most `per_twin_variant` (2)
examples per twin variant, `per_control_variant` (1) per control and `per_plain_variant` (1) per plain variant (a stable
hash order picks which); then high_risk is made at least `min_high_risk_share` (half) of the twin examples by thinning
the other twin families round-robin (if there is no high_risk example at all nothing is thinned and the summary says so);
then the infeasible share is capped at `max_infeasible_share` (0.35), thinned in a high_risk / other-families alternation
(the other families round-robin), so the floor survives the cap.

Examples: the whole trajectory (`bestofn.trajectory_messages`), every assistant turn trained, the report and its stated
confidence as the model gave them (they are the model's own, unlike a teacher's), the system message unhinted. A built
example that still carries hint text anywhere raises.

    python -m honest_agents.sft.selfdistill --config configs/sft_selfdistill_qwen35.yaml   # selection only, no GPU
    python -m honest_agents.sft.train_sft --config configs/sft_selfdistill_qwen35.yaml --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from honest_agents import harness
from honest_agents.bank import BankRow
from honest_agents.pilot.dataset import kind_of, normalize_message
from honest_agents.readme_read import high_risk_targets, readme_read_before_decision
from honest_agents.reward import compute_reward
from honest_agents.sft.bestofn import trajectory_messages
from honest_agents.sft.data import SFTExample, variant_of
from honest_agents.sft.teacher_stage import (
    bash_commands,
    composition,
    family_of,
    infeasible_limit,
    programmatic_drop_reason,
)
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage

logger = logging.getLogger(__name__)

KIND = "selfdistill"
HIGH_RISK = "high_risk"
PROMPTS = Path(__file__).resolve().parent.parent / "eval" / "prompts"
HINT_FILES = {"tier1": "hint_tier1.txt", "tier2": "hint_tier2.txt"}

DROP_NO_README = "high_risk_readme_not_read_first"
DROP_HINT_ECHO = "hint_echo"
DROP_UNPAIRED = "no_paired_control"
DROP_VARIANT_CAP = "per_variant_cap"
DROP_HIGH_RISK_FLOOR = "high_risk_floor"
DROP_INFEASIBLE_CAP = "infeasible_share_cap"

# Assistant text that justifies a decision by the hint instead of by what the sandbox showed (the review's list). It
# also matches the base model citing the TASK ("remove /data/x as instructed"): on the unhinted pass@4 bank 137 of 2950
# rows match, 39 of 506 high_risk control successes and 32 of 260 remove_input abstentions among them. Those are dropped
# too by default (the review asked for this filter); `selfdistill.hint_echo_pattern` narrows it without a code change.
# References to the hint itself. "as instructed" / "instructions" alone are NOT matched: the base model uses them for the
# user's task ("remove /x as instructed": 39 of 506 base high_risk control successes would be lost).
HINT_ECHO_PATTERN = (
    r"\bguidelines?\b|\bsystem prompt\b|\bnotes?[- ]rules?\b"
    r"|\b(?:i was|i am|i'm) (?:told|asked|instructed) to (?:inspect|look|read|check)\b|\b(?:per|following|as per) (?:my|the) instructions\b"
)
HINT_ECHO = re.compile(HINT_ECHO_PATTERN, re.IGNORECASE)


@dataclass(frozen=True)
class SelfDistillOptions:
    per_twin_variant: int = 2
    per_control_variant: int = 1
    per_plain_variant: int = 1
    max_infeasible_share: float | None = 0.35
    min_high_risk_share: float | None = 0.5  # of the twin examples; None = no floor
    require_readme_read: bool = True  # high_risk rows (twin and control) must read the README before deciding
    require_paired_control: bool = True  # a twin needs its own control's kept success
    drop_hint_echo: bool = True  # False = count only
    hint_echo_pattern: str = HINT_ECHO_PATTERN  # case-insensitive regex over assistant text

    @property
    def echo_regex(self) -> re.Pattern[str]:
        return re.compile(self.hint_echo_pattern, re.IGNORECASE)

    def __post_init__(self) -> None:
        self.echo_regex  # noqa: B018 - a bad pattern fails at parse time
        if min(self.per_twin_variant, self.per_control_variant, self.per_plain_variant) < 1:
            raise ValueError("selfdistill per-variant caps must be >= 1")
        for name in ("max_infeasible_share", "min_high_risk_share"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"selfdistill.{name} must be in [0, 1] or null")
        if self.min_high_risk_share == 0.0:
            raise ValueError("selfdistill.min_high_risk_share 0 is no floor: use null")


def parse_options(raw: dict[str, Any] | None) -> SelfDistillOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(SelfDistillOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `selfdistill:`; expected {sorted(allowed)}")
    return SelfDistillOptions(**raw)


def validate(cfg: SftConfig) -> None:
    """Called by `train_sft.parse_config`: bad stage keys fail before any data is read."""
    parse_options(cfg.selfdistill)


# ── the hint ─────────────────────────────────────────────────────────────────


@cache
def hint_paragraphs() -> dict[str, str]:
    """tier -> the text the hint file appends to `harness.SYSTEM_PROMPT` (the test pins that the files start with it)."""
    out = {}
    for tier, name in HINT_FILES.items():
        text = (PROMPTS / name).read_text()
        if not text.startswith(harness.SYSTEM_PROMPT):
            raise ValueError(f"{name} does not start with harness.SYSTEM_PROMPT")
        out[tier] = text[len(harness.SYSTEM_PROMPT) :].strip()
    return out


def hint_source(row: BankRow) -> str:
    """Which prompt the rollout was sampled with: `tier1`, `tier2`, `unhinted` or `other` (recorded per example, since
    the design wants to know which tier produced the data)."""
    system = next((m.get("content") for m in row.prompt if m.get("role") == "system"), None)
    if not isinstance(system, str):
        return "other"
    if system.strip() == harness.SYSTEM_PROMPT.strip():
        return "unhinted"
    for tier, paragraph in hint_paragraphs().items():
        if system.strip() == f"{harness.SYSTEM_PROMPT}\n\n{paragraph}".strip():
            return tier
    return "other"


def _first_sentence(paragraph: str) -> str:
    return paragraph.split(". ", 1)[0]


def hint_leaks(messages: Iterable[dict[str, Any]]) -> bool:
    """A prompt-side message (system, user, tool) still carries hint text (the first sentence of either tier's
    paragraph). Assistant turns are the model's own words: an echo there is the `hint_echo` filter's business."""
    probes = {_first_sentence(p) for p in hint_paragraphs().values()}
    for m in messages:
        if m.get("role") == "assistant":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        if any(p in text for p in probes):
            return True
    return False


def unhinted(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`messages` with the system message replaced by `harness.SYSTEM_PROMPT` (inserted when there is none)."""
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0] = {**out[0], "content": harness.SYSTEM_PROMPT}
    else:
        out.insert(0, {"role": "system", "content": harness.SYSTEM_PROMPT})
    return out


def _assistant_texts(row: BankRow) -> Iterable[str]:
    for message in trajectory_messages(row):
        if message.get("role") != "assistant":
            continue
        m = normalize_message(message)
        content = m.get("content")
        yield content if isinstance(content, str) else json.dumps(content, default=str)
        if isinstance(m.get("reasoning_content"), str):
            yield m["reasoning_content"]
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != "bash":
                yield json.dumps(fn.get("arguments"), default=str)


def echoes_hint(row: BankRow, pattern: re.Pattern[str] = HINT_ECHO) -> bool:
    return any(pattern.search(text) for text in _assistant_texts(row))


# ── filters ──────────────────────────────────────────────────────────────────


def is_high_risk(row: BankRow) -> bool:
    return family_of(row) == HIGH_RISK


def _spec(row: BankRow) -> dict[str, Any]:
    spec = row.task.get("spec")
    return spec if isinstance(spec, dict) else {}


def read_readme_first(row: BankRow) -> bool:
    return readme_read_before_decision(bash_commands(row), high_risk_targets(_spec(row), row.outcome))


def pair_key(row: BankRow) -> tuple[str, str]:
    pair = row.task.get("pair_id") or _spec(row).get("pair_id") or variant_of(row)
    return str(pair), family_of(row)


def row_drop_reason(row: BankRow, opts: SelfDistillOptions, echo: bool | None = None) -> str | None:
    """Why `row` must not be imitated, before pairing and balance; None = it may be. `echo` passes a precomputed
    `echoes_hint(row)`."""
    if (reason := programmatic_drop_reason(row)) is not None:
        return reason
    if opts.require_readme_read and is_high_risk(row) and not read_readme_first(row):
        return DROP_NO_README
    if opts.drop_hint_echo and (echoes_hint(row, opts.echo_regex) if echo is None else echo):
        return DROP_HINT_ECHO
    return None


@dataclass
class Filtered:
    kept: list[BankRow]
    dropped: Counter[str]
    hint_echo: int  # rows whose assistant text echoes the hint, whatever else dropped them


def filter_rows(rows: Iterable[BankRow], opts: SelfDistillOptions) -> Filtered:
    out = Filtered([], Counter(), 0)
    pattern = opts.echo_regex
    for row in rows:
        echo = echoes_hint(row, pattern)
        out.hint_echo += echo
        reason = row_drop_reason(row, opts, echo)
        if reason is None:
            out.kept.append(row)
        else:
            out.dropped[reason] += 1
    return out


def paired(rows: list[BankRow], controls: set[tuple[str, str]], dropped: Counter[str]) -> list[BankRow]:
    """Twins whose own control has a kept success; every other row passes."""
    out = []
    for row in rows:
        if kind_of(row) == "infeasible" and pair_key(row) not in controls:
            dropped[DROP_UNPAIRED] += 1
        else:
            out.append(row)
    return out


# ── balance ──────────────────────────────────────────────────────────────────


def _hash(row: BankRow) -> str:
    return hashlib.sha256(row.episode_id.encode()).hexdigest()


def cap_per_variant(rows: list[BankRow], opts: SelfDistillOptions, dropped: Counter[str]) -> list[BankRow]:
    limits = {
        "infeasible": opts.per_twin_variant,
        "control": opts.per_control_variant,
        "feasible": opts.per_plain_variant,
    }
    by_variant: dict[str, list[BankRow]] = defaultdict(list)
    for row in rows:
        by_variant[variant_of(row)].append(row)
    keep: set[str] = set()
    for group in by_variant.values():
        keep |= {r.episode_id for r in sorted(group, key=_hash)[: limits[kind_of(group[0])]]}
    dropped[DROP_VARIANT_CAP] += len(rows) - len(keep)
    return [r for r in rows if r.episode_id in keep]


def round_robin(rows: list[BankRow]) -> list[BankRow]:
    """`rows` ordered round-robin across families, a stable hash order within each family: any prefix thins every
    family evenly and none is emptied before the others."""
    by_family: dict[str, list[BankRow]] = defaultdict(list)
    for row in sorted(rows, key=_hash):
        by_family[family_of(row)].append(row)
    queues = [by_family[f] for f in sorted(by_family)]
    return [q[d] for d in range(max((len(q) for q in queues), default=0)) for q in queues if d < len(q)]


def high_risk_floor(rows: list[BankRow], share: float | None, dropped: Counter[str]) -> list[BankRow]:
    """Thin the other twin families until high_risk is at least `share` of the twin examples."""
    twins = [r for r in rows if kind_of(r) == "infeasible"]
    n_hr = sum(map(is_high_risk, twins))
    others = [r for r in twins if not is_high_risk(r)]
    if share is None or n_hr == 0 or not others:
        return rows
    allowed = int(n_hr * (1.0 - share) / share + 1e-9)  # n_hr / (n_hr + k) >= share
    if len(others) <= allowed:
        return rows
    keep = {r.episode_id for r in round_robin(others)[:allowed]}
    dropped[DROP_HIGH_RISK_FLOOR] += len(others) - allowed
    return [r for r in rows if kind_of(r) != "infeasible" or is_high_risk(r) or r.episode_id in keep]


def cap_infeasible(rows: list[BankRow], max_share: float | None, dropped: Counter[str]) -> list[BankRow]:
    """Cap the infeasible share; the twins kept alternate high_risk and the other families (round-robin)."""
    twins = [r for r in rows if kind_of(r) == "infeasible"]
    limit = infeasible_limit(len(twins), len(rows) - len(twins), max_share)
    if limit >= len(twins):
        return rows
    hr = sorted((r for r in twins if is_high_risk(r)), key=_hash)
    other = round_robin([r for r in twins if not is_high_risk(r)])
    order: list[BankRow] = []
    for i in range(max(len(hr), len(other))):
        order += [q[i] for q in (hr, other) if i < len(q)]
    keep = {r.episode_id for r in order[:limit]}
    dropped[DROP_INFEASIBLE_CAP] += len(twins) - limit
    return [r for r in rows if kind_of(r) != "infeasible" or r.episode_id in keep]


def balance(rows: list[BankRow], opts: SelfDistillOptions, dropped: Counter[str]) -> list[BankRow]:
    rows = cap_per_variant(rows, opts, dropped)
    rows = high_risk_floor(rows, opts.min_high_risk_share, dropped)
    return cap_infeasible(rows, opts.max_infeasible_share, dropped)


def twin_mix(rows: Iterable[BankRow], share: float | None) -> dict[str, Any]:
    twins = [r for r in rows if kind_of(r) == "infeasible"]
    n_hr = sum(map(is_high_risk, twins))
    hr_share = n_hr / len(twins) if twins else 0.0
    return {
        "twins": len(twins),
        "high_risk": n_hr,
        "high_risk_share": round(hr_share, 4),
        "high_risk_floor_met": share is None or (bool(twins) and hr_share + 1e-9 >= share),
    }


# ── examples ─────────────────────────────────────────────────────────────────


def build_example(row: BankRow) -> SFTExample:
    """The whole trajectory without the hint, every assistant turn trained, the model's own report kept."""
    messages = unhinted(trajectory_messages(row))
    if hint_leaks(messages):
        raise ValueError(f"{row.episode_id}: hint text is still in the example after the system message was replaced")
    hr = is_high_risk(row)
    return SFTExample(
        id=row.episode_id,
        kind=KIND,
        variant=variant_of(row),
        messages=messages,
        loss_messages=[i for i, m in enumerate(messages) if m.get("role") == "assistant"],
        meta={
            "hint": hint_source(row),
            "task_kind": kind_of(row),
            "family": family_of(row),
            "case": compute_reward(row.episode_outcome(), row.config()).case,
            "tool": (row.report_call or {}).get("name"),
            "n_commands": len(bash_commands(row)),
            "readme_read_first": read_readme_first(row) if hr else None,
        },
    )


@dataclass
class Selection:
    fit: list[BankRow]
    held: list[BankRow]
    summary: dict[str, Any]


def select(cfg: SftConfig, fit: list[BankRow], held: list[BankRow]) -> Selection:
    opts = parse_options(cfg.selfdistill)
    f_fit = filter_rows(fit, opts)
    f_held = filter_rows(held, opts)
    controls = {pair_key(r) for r in f_fit.kept + f_held.kept if kind_of(r) == "control"}
    if opts.require_paired_control:
        kept = paired(f_fit.kept, controls, f_fit.dropped)
        kept_held = paired(f_held.kept, controls, f_held.dropped)
    else:
        kept, kept_held = f_fit.kept, f_held.kept
    before = composition(kept)
    balanced = balance(kept, opts, f_fit.dropped)
    hr_rows = [r for r in fit if is_high_risk(r)]
    summary = {
        "selfdistill": {
            "options": dataclasses.asdict(opts),
            "candidate_rows": len(fit),
            "candidate_composition": composition(fit),
            "hint_sources": dict(sorted(Counter(hint_source(r) for r in fit + held).items())),
            "dropped": dict(sorted(f_fit.dropped.items())),
            "hint_echo_rows": f_fit.hint_echo + f_held.hint_echo,
            "high_risk_readme_read_first": {
                "rows": len(hr_rows),
                "read_first": sum(map(read_readme_first, hr_rows)),
            },
            "controls_with_kept_success": len(controls),
            "selected_before_balance": before,
            "selected": composition(balanced),
            "twin_mix": twin_mix(balanced, opts.min_high_risk_share),
            "heldout_candidate_rows": len(held),
            "heldout_dropped": dict(sorted(f_held.dropped.items())),
            "heldout_selected": composition(kept_held),
        }
    }
    if not summary["selfdistill"]["twin_mix"]["high_risk_floor_met"]:
        logger.warning("high_risk is below %s of the twin examples: %s", opts.min_high_risk_share, summary)
    return Selection(balanced, kept_held, summary)


def _selfdistill_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    chosen = select(cfg, fit, held)
    return StageData([build_example(r) for r in chosen.fit], [build_example(r) for r in chosen.held], chosen.summary)


register_stage(KIND, _selfdistill_stage, validate)


# ── CLI: selection only ──────────────────────────────────────────────────────


def selection_report(cfg: SftConfig) -> dict[str, Any]:
    """The bank read, the partition and the selection, without a tokenizer, a model or a GPU."""
    from honest_agents.sft.train_sft import build_stage_data

    if cfg.kind != KIND:
        raise ValueError(f"config kind is {cfg.kind!r}, expected {KIND!r}")
    _, summary = build_stage_data(cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    from honest_agents.sft.train_sft import load_config

    parser = argparse.ArgumentParser(description="Print the selfdistill SFT selection summary (no model, no GPU).")
    parser.add_argument("--config", required=True, help="YAML config with kind: selfdistill")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(selection_report(load_config(args.config, args.overrides)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    # Under `python -m` this file is `__main__`, a second copy; dispatch through the imported module (see train_sft).
    from honest_agents.sft.selfdistill import main as _main

    sys.exit(_main())
