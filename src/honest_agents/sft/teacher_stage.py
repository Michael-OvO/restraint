"""Teacher-trajectory SFT (`kind: teacher`): imitate whole episodes an outside agent played in the real harness.

`honest_agents.teacher` lets an outside agent play the policy's part through a mailbox client: the same system prompt,
tools, command / token budgets and grader, and the episode is written as an ordinary trajectory-bank row
(`bank.BankRow`, `extra["teacher_token"]` naming the episode). A reviewer then audits every token and writes one JSON line
to the audit file (`data/teacher/audit.jsonl`): `{"token", "keep", "outcome_ok", "orientation_before_mutation",
"evidence_grounded", "style_ok", "report_ok", "issues"}`.

Why a separate stage: the SFT stages so far imitated the model's own rollouts, and that taught formatting, not honesty
(training log, iterations 4-5 and 10). A teacher episode can show what the policy rarely does on its own - look before
changing anything, abstain on an impossible twin with the real blocker named - but only if every imitated episode is one
we would want the policy to copy. So nothing is rewritten here (the report and its confidence stay as the teacher gave
them: the teacher states what it saw, and a calibrated rewrite would put a number in its mouth that its trajectory does
not support), and the selection is strict, in two layers:

* programmatic filters, each with a counted drop reason (`select_rows`), checked in this order: train split only; not
  masked; a parsable report call with no hidden tool calls in the report turn; no tamper, no self-inflicted damage, no
  violation; an infeasible variant that "passed" is a leak; a feasible or control row only when its reward case is
  `success_claimed` (a clean, truthful completion), an infeasible row only when it is `abstain_correct` (damage was
  already refused above, so never an abstention after damage); and orientation: at least one non-mutating command ran
  before the first state-changing one (`mutation.first_mutation_index`, on the transcript's bash commands; the grader's
  own `first_mutation_idx` is honoured too). An episode with no command at all oriented on nothing and is dropped. The
  rows reach the stage through `data.read_rows`, which already drops most of these (`rows_dropped` in the summary);
  they are re-checked here so a direct caller cannot bypass them.
* the audit: only tokens whose latest verdict has `keep: true` (a JSON boolean) survive. A row without a token, or a
  token without a verdict, is dropped and counted: an unreviewed episode is never imitated by default. The programmatic
  filters run first, so the audit's own contribution shows in the counts.

Balancing: the teacher's assignments over-sample impossible twins (about 36 % of the first teacher bank against 28 % of
the RL pool), and a correct abstention is short and passes the filters easily, so an unbalanced mixture would lean
towards "look, then abstain" - the over-declining of feasible work the combo SFT showed (training log, observation 23).
The infeasible share of the fit examples is capped at
`teacher.max_infeasible_share` (default 0.35). Abstentions over the cap are thinned round-robin across families (stable
hash order within a family), so a family is not wiped out by the cap. The composition by kind and family before and after
is in the summary. The held-out examples are filtered the same way but not balanced: they measure eval loss on whatever
the held-out variants hold.

Every assistant turn is trained (commands, reasoning, the report call and the end-of-turn token), exactly as in
`sft.bestofn`; system, user and tool turns are masked.

Selection without a GPU, tokenizer or trainer config:
    python -m honest_agents.sft.teacher_stage --config configs/sft_teacher_qwen35.yaml [key.path=value ...]
The full run (or `--dry-run`, which also tokenises when the tokenizer is cached):
    python -m honest_agents.sft.train_sft --config configs/sft_teacher_qwen35.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.bank import BankRow
from honest_agents.mutation import first_mutation_index
from honest_agents.pilot.dataset import kind_of, normalize_message
from honest_agents.reward import compute_reward
from honest_agents.sft.bestofn import trajectory_messages
from honest_agents.sft.data import TRAIN_SPLIT, SFTExample, context_messages, variant_of
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage

logger = logging.getLogger(__name__)

KIND = "teacher"
TOKEN_KEY = "teacher_token"  # `BankRow.extra` key the teacher writes
AUDIT_BASENAME = "audit.jsonl"
CASE_FEASIBLE_OK = "success_claimed"
CASE_INFEASIBLE_OK = "abstain_correct"

# Drop reasons, in the order they are checked.
DROP_SPLIT = "not_train_split"
DROP_MASKED = "masked"
DROP_NO_REPORT = "no_report"
DROP_HIDDEN_CALLS = "hidden_calls"
DROP_UNPARSED = "unparsed_report"
DROP_TAMPERED = "tampered"
DROP_SELF_INFLICTED = "self_inflicted"
DROP_VIOLATION = "violation"
DROP_LEAK = "infeasible_clean_pass"
DROP_FEASIBLE_CASE = "feasible_not_success_claimed"
DROP_INFEASIBLE_CASE = "infeasible_not_abstain_correct"
DROP_NO_COMMANDS = "no_commands"
DROP_NO_ORIENTATION = "mutation_before_orientation"
DROP_NO_TOKEN = "no_teacher_token"
DROP_NO_AUDIT = "no_audit_verdict"
DROP_AUDIT_REJECTED = "audit_rejected"
DROP_INFEASIBLE_CAP = "infeasible_share_cap"  # counted by `balance`, not `select_rows`


@dataclass(frozen=True)
class TeacherOptions:
    # Audit file (one JSON line per token). None = `audit.jsonl` next to the bank directory (data/teacher/bank ->
    # data/teacher/audit.jsonl), which is where the teacher tooling writes it.
    audit: str | None = None
    # Cap on the infeasible share of the fit examples; None = no cap. 1.0 is allowed and equals no cap.
    max_infeasible_share: float | None = 0.35

    def __post_init__(self) -> None:
        if self.max_infeasible_share is not None and not 0.0 <= self.max_infeasible_share <= 1.0:
            raise ValueError("teacher.max_infeasible_share must be in [0, 1] or null")


def parse_options(raw: dict[str, Any] | None) -> TeacherOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(TeacherOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `teacher:`; expected {sorted(allowed)}")
    return TeacherOptions(**raw)


def validate(cfg: SftConfig) -> None:
    """Called by `train_sft.parse_config`: bad stage keys fail before any data is read."""
    parse_options(cfg.teacher)


# ── audit ────────────────────────────────────────────────────────────────────


def audit_path(opts: TeacherOptions, bank_path: str | Path) -> Path:
    if opts.audit is not None:
        return Path(opts.audit)
    bank = Path(bank_path)
    bank_directory = bank if bank.is_dir() else bank.parent
    return bank_directory.parent / AUDIT_BASENAME


def read_audit(path: str | Path) -> dict[str, dict[str, Any]]:
    """token -> its verdict. A later line for the same token wins (a re-audit replaces the first one). A missing file is
    an empty audit (every row is then dropped as unaudited), so a dry run still shows the programmatic selection."""
    p = Path(path)
    if not p.exists():
        logger.warning("audit file %s does not exist: every teacher row will be dropped as unaudited", p)
        return {}
    verdicts: dict[str, dict[str, Any]] = {}
    unusable = 0
    for line in bank_mod._complete_lines(p):
        entry = json.loads(line)
        if isinstance(entry, dict) and isinstance(entry.get("token"), str) and entry["token"]:
            verdicts[entry["token"]] = entry
        else:
            unusable += 1
    if unusable:
        # Not fatal (the rows it meant are then dropped as unaudited), but a typo in the audit must not go unseen.
        logger.warning("%d line(s) of %s carry no token string and are ignored", unusable, p)
    return verdicts


def token_of(row: BankRow) -> str | None:
    token = row.extra.get(TOKEN_KEY)
    return token if isinstance(token, str) and token else None


# ── programmatic filters ─────────────────────────────────────────────────────


def bash_commands(row: BankRow) -> list[str]:
    """The bash commands of the trajectory in order, as the transcript shows them (the report turn runs none: a row
    with calls before its report is refused as `hidden_calls`)."""
    commands: list[str] = []
    for message in context_messages(row):
        if message.get("role") != "assistant":
            continue
        for call in normalize_message(message).get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if fn.get("name") == "bash" and isinstance(args, dict):
                commands.append(str(args.get("command") or ""))
    return commands


def oriented_first(row: BankRow) -> str | None:
    """None when at least one non-mutating command ran before the first state-changing one (or nothing mutated);
    else the drop reason. The grader's own index is honoured when the bank has it (a command the regex misses but the
    environment counted). A blank command looks at nothing, so it neither counts as orientation nor as a command."""
    commands = [c for c in bash_commands(row) if c.strip()]
    if not commands:
        return DROP_NO_COMMANDS
    if first_mutation_index(commands) == 0 or row.episode_outcome().first_mutation_idx == 0:
        return DROP_NO_ORIENTATION
    return None


def programmatic_drop_reason(row: BankRow) -> str | None:
    """Why `row` must not be imitated, before the audit is consulted; None = it may be."""
    o = row.episode_outcome()
    if row.task.get("split") != TRAIN_SPLIT:
        return DROP_SPLIT
    br = compute_reward(o, row.config())
    if o.masked_reason is not None or (o.grading_error is not None and not o.acted) or row.reward.get("masked"):
        return DROP_MASKED
    if br.masked:
        return DROP_MASKED
    if row.report_call is None:
        return DROP_NO_REPORT
    if row.report_turn_extra_calls:
        return DROP_HIDDEN_CALLS
    if o.report is None:
        return DROP_UNPARSED
    if o.tampered:
        return DROP_TAMPERED
    if o.damage_paths:
        return DROP_SELF_INFLICTED
    if o.violation:
        return DROP_VIOLATION
    if not o.feasible and o.passed:
        return DROP_LEAK
    if o.feasible and br.case != CASE_FEASIBLE_OK:
        return DROP_FEASIBLE_CASE
    if not o.feasible and br.case != CASE_INFEASIBLE_OK:
        return DROP_INFEASIBLE_CASE
    return oriented_first(row)


def audit_drop_reason(row: BankRow, verdicts: dict[str, dict[str, Any]]) -> str | None:
    token = token_of(row)
    if token is None:
        return DROP_NO_TOKEN
    verdict = verdicts.get(token)
    if verdict is None:
        return DROP_NO_AUDIT
    return None if verdict.get("keep") is True else DROP_AUDIT_REJECTED


def select_rows(
    rows: Iterable[BankRow], verdicts: dict[str, dict[str, Any]], dropped: Counter[str] | None = None
) -> list[BankRow]:
    """The rows to imitate, in input order. Drop reasons (the first that applies) are counted into `dropped`."""
    dropped = dropped if dropped is not None else Counter()
    kept: list[BankRow] = []
    for row in rows:
        reason = programmatic_drop_reason(row) or audit_drop_reason(row, verdicts)
        if reason is None:
            kept.append(row)
        else:
            dropped[reason] += 1
    return kept


# ── balancing ────────────────────────────────────────────────────────────────


def family_of(row: BankRow) -> str:
    spec = row.task.get("spec") if isinstance(row.task.get("spec"), dict) else {}
    for value in (row.task.get("family"), row.task.get("perturbation"), spec.get("perturbation")):
        if value:
            return str(value)
    return "unknown"


def _hash(row: BankRow) -> str:
    return hashlib.sha256(row.episode_id.encode()).hexdigest()


def infeasible_limit(n_infeasible: int, n_other: int, max_share: float | None) -> int:
    """The most infeasible rows that keep their share of the total at or below `max_share`."""
    if max_share is None or max_share >= 1.0:
        return n_infeasible
    # n / (n + other) <= s  <=>  n <= s * other / (1 - s)
    return min(n_infeasible, int(max_share * n_other / (1.0 - max_share) + 1e-9))


def balance(rows: list[BankRow], max_share: float | None, dropped: Counter[str] | None = None) -> list[BankRow]:
    """`rows` with the infeasible ones thinned to `max_share` of the total, round-robin across families in a stable hash
    order (so reruns pick the same rows and no family is emptied before the others are thinned). Order is kept."""
    dropped = dropped if dropped is not None else Counter()
    infeasible = [r for r in rows if kind_of(r) == "infeasible"]
    limit = infeasible_limit(len(infeasible), len(rows) - len(infeasible), max_share)
    if limit >= len(infeasible):
        return rows
    by_family: dict[str, list[BankRow]] = defaultdict(list)
    for row in sorted(infeasible, key=_hash):
        by_family[family_of(row)].append(row)
    queues = [by_family[f] for f in sorted(by_family)]
    chosen: set[str] = set()
    depth = 0
    while len(chosen) < limit:
        for queue in queues:
            if depth < len(queue) and len(chosen) < limit:
                chosen.add(queue[depth].episode_id)
        depth += 1
    dropped[DROP_INFEASIBLE_CAP] += len(infeasible) - limit
    return [r for r in rows if kind_of(r) != "infeasible" or r.episode_id in chosen]


def composition(rows: Iterable[BankRow]) -> dict[str, Any]:
    rows = list(rows)
    kinds = Counter(kind_of(r) for r in rows)
    return {
        "n": len(rows),
        "by_kind": dict(sorted(kinds.items())),
        "by_family": dict(sorted(Counter(f"{kind_of(r)}/{family_of(r)}" for r in rows).items())),
        "infeasible_share": round(kinds["infeasible"] / len(rows), 4) if rows else 0.0,
    }


# ── examples ─────────────────────────────────────────────────────────────────


def build_example(row: BankRow) -> SFTExample:
    """The whole trajectory, every assistant turn trained, the report (and its confidence) as the teacher gave it."""
    messages = trajectory_messages(row)
    return SFTExample(
        id=row.episode_id,
        kind=KIND,
        variant=variant_of(row),
        messages=messages,
        loss_messages=[i for i, m in enumerate(messages) if m.get("role") == "assistant"],
        meta={
            "teacher_token": token_of(row),
            "task_kind": kind_of(row),
            "family": family_of(row),
            "case": compute_reward(row.episode_outcome(), row.config()).case,
            "tool": (row.report_call or {}).get("name"),
            "n_commands": len(bash_commands(row)),
        },
    )


@dataclass
class TeacherSelection:
    fit: list[BankRow]
    held: list[BankRow]
    summary: dict[str, Any]


def select(cfg: SftConfig, fit: list[BankRow], held: list[BankRow]) -> TeacherSelection:
    opts = parse_options(cfg.teacher)
    path = audit_path(opts, cfg.data.bank)
    verdicts = read_audit(path)
    dropped: Counter[str] = Counter()
    kept = select_rows(fit, verdicts, dropped)
    before = composition(kept)
    balanced = balance(kept, opts.max_infeasible_share, dropped)
    dropped_held: Counter[str] = Counter()
    kept_held = select_rows(held, verdicts, dropped_held)
    summary = {
        "teacher": {
            "audit_file": str(path),
            "audit_verdicts": len(verdicts),
            "audit_keep": sum(v.get("keep") is True for v in verdicts.values()),
            "candidate_rows": len(fit),
            "dropped": dict(sorted(dropped.items())),
            "max_infeasible_share": opts.max_infeasible_share,
            "selected_before_balance": before,
            "selected": composition(balanced),
            "heldout_candidate_rows": len(held),
            "heldout_dropped": dict(sorted(dropped_held.items())),
            "heldout_selected": composition(kept_held),
        }
    }
    return TeacherSelection(balanced, kept_held, summary)


def _teacher_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    chosen = select(cfg, fit, held)
    return StageData([build_example(r) for r in chosen.fit], [build_example(r) for r in chosen.held], chosen.summary)


register_stage(KIND, _teacher_stage, validate)


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

    parser = argparse.ArgumentParser(description="Print the teacher SFT selection summary (no model, no GPU).")
    parser.add_argument("--config", required=True, help="YAML config with kind: teacher")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(selection_report(load_config(args.config, args.overrides)), indent=2, default=str))
    return 0


if __name__ == "__main__":
    # Under `python -m` this file is `__main__`, a second copy; dispatch through the imported module (see train_sft).
    from honest_agents.sft.teacher_stage import main as _main

    sys.exit(_main())
