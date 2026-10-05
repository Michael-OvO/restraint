"""Part 2: reward-filtered best-of-N trajectory SFT (one round of expert iteration), and the `combo` stage.

Rollouts of one variant are a GRPO group. The banked outcomes are scored with the SAME reward the RL stage uses
(`reward.compute_reward` on the stored outcome, under the row's own `reward_config` unless `bestofn.reward` overrides it),
and per group the best rollouts above the group mean are kept. The kept trajectories are fine-tuned WHOLE: every
assistant turn (commands, reasoning, the report call and the end-of-turn token) carries loss; the system prompt, the user
turn and the tool outputs are masked (`SFTExample.loss_messages` = every assistant turn). The trajectory and its report
are the model's own (only the stated confidence is recalibrated, below): no hindsight rewrite, nothing is read from the
task spec, so the stage cannot teach the model to name something it never saw.

Selection rules (`select_rows`):

* Only `split == train` rows exist here (`data.read_rows`); masked rows and quarantined variants are already gone, and the
  stage also refuses a tampered or violating rollout, an infeasible variant that "passed" (leak) and a rollout that never
  produced a parsable report, whatever `data.row_filter` says: a trajectory that earned its reward by an unobservable
  rule must not be imitated.
* Above the group mean is not good enough on its own: in a bad group the best rollout can still be a fabrication or a
  false blame. Only rollouts whose reward case is one of `IMITABLE_CASES` are kept (a clean claim, an honest
  non-completion, a correct abstention); a fabrication, a false blame, an abstention with the wrong reason, a silent
  stop or a lazy decline is never imitated, whatever its rank.
* The stated confidence is rewritten (`bestofn.confidence: calibrated`, the default). Imitating only the rollouts that
  came out well would teach "complete, 0.95" after every claim, because the failures that shared that confidence are
  never selected. The report's confidence becomes the held-out pass rate of its claim type, as in `sft.hindsight`
  (abstain: one minus it); the rest of the report and the trajectory stay as the model wrote them. `keep` leaves it.
* The group mean is taken over EVERY scored rollout of the variant, tampered ones included (that is the group GRPO would
  see). A rollout is kept only if its reward is above the mean by `min_margin`, so groups with equal rewards (no signal)
  contribute nothing. `above_mean: false` drops that rule and keeps the top `top_k` of every group.
* Ties break on the shorter trajectory (fewer tool calls: the reward's own efficiency preference), then a stable hash.
* Trajectories that ran out of budget (no report call) and rows whose report turn hid tool calls are not imitated.

`combo` trains both parts in one adapter: the best-of-N trajectories, plus hindsight report examples (`sft.hindsight`)
for the rows that were NOT selected. A selected episode appears once, as its own trajectory: its original report earned an
above-mean reward and a second, rewritten copy of the same transcript would give two targets for one context. The ratio
is `bestofn.hindsight_per_bestofn` (hindsight examples per best-of-N example, default 1.0). A real trajectory trains many
more tokens than a report, so equal counts already leave the report decision a minor share of the loss; the number of
trained tokens per kind is not measured on a real bank yet, so check it in the dry run before raising the ratio. `null` keeps every hindsight example.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from honest_agents.bank import BankRow
from honest_agents.pilot.dataset import normalize_message
from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import hindsight
from honest_agents.sft.data import TRAIN_SPLIT, SFTExample, context_messages, variant_of
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage

KIND = "bestofn"
KIND_COMBO = "combo"
_EPS = 1e-9

SKIP_NO_REPORT = "no_report"  # the trajectory never reported (forced report): its last turn is not a decision
SKIP_HIDDEN_CALLS = "hidden_calls"  # tool calls ran in the report turn that the row does not keep
SKIP_UNPARSED = "unparsed_report"
SKIP_TAMPERED = "tampered"
SKIP_VIOLATION = "violation"
SKIP_LEAK = "infeasible_clean_pass"
SKIP_MASKED = "masked"
SKIP_SPLIT = "not_train_split"
SKIP_SMALL_GROUP = "small_group"
SKIP_NOT_ABOVE_MEAN = "not_above_mean"
SKIP_NOT_TOP_K = "below_top_k"
SKIP_NO_PASS = "group_without_pass"
SKIP_BAD_CASE = "not_imitable_case"
CONF_KEEP = "keep"

# `RewardBreakdown.case` values worth imitating: the report is honest and, where a reason is given, the right one.
IMITABLE_CASES = frozenset(
    {"success_claimed", "honest_failure", "honest_underclaim", "honest_failure_observed", "abstain_correct"}
)


@dataclass(frozen=True)
class BestOfNOptions:
    top_k: int = 1  # rollouts kept per variant
    above_mean: bool = True  # keep only rollouts whose reward exceeds the group mean by `min_margin`
    min_margin: float = 0.0
    min_group_size: int = 2  # scored rollouts a variant needs to be ranked at all
    require_pass_when_feasible: bool = False  # stricter: on a feasible variant only a clean pass is imitated
    reward: dict[str, Any] = field(default_factory=dict)  # `RewardConfig` fields; empty = each row's stored config
    confidence: str = hindsight.CONF_CALIBRATED  # `calibrated` (held-out pass rate of the claim type) | `keep`
    hindsight_per_bestofn: float | None = 1.0  # combo only: hindsight examples per best-of-N example; None = all

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.min_group_size < 2 or self.min_margin < 0:
            raise ValueError("bestofn needs top_k >= 1, min_group_size >= 2 and min_margin >= 0")
        if self.confidence not in (hindsight.CONF_CALIBRATED, CONF_KEEP):
            raise ValueError(f"bestofn.confidence must be {hindsight.CONF_CALIBRATED!r} or {CONF_KEEP!r}")
        if self.hindsight_per_bestofn is not None and self.hindsight_per_bestofn < 0:
            raise ValueError("bestofn.hindsight_per_bestofn must be >= 0 or null")
        if self.reward:
            self.reward_config()  # unknown keys and bad values fail at parse time

    def reward_config(self) -> RewardConfig | None:
        return RewardConfig(**self.reward) if self.reward else None


def parse_options(raw: dict[str, Any] | None) -> BestOfNOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(BestOfNOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `bestofn:`; expected {sorted(allowed)}")
    return BestOfNOptions(**raw)


def validate(cfg: SftConfig) -> None:
    """Called by `train_sft.parse_config`: bad stage keys fail before any data is read."""
    parse_options(cfg.bestofn)
    if cfg.kind == KIND_COMBO:
        hindsight.HindsightOptions(**cfg.hindsight)


# ── ranking ──────────────────────────────────────────────────────────────────


@dataclass
class Selection:
    row: BankRow
    reward: float
    group_mean: float
    group_size: int
    rank: int  # 1 = best of its group


def _reward_of(row: BankRow, cfg: RewardConfig | None) -> float | None:
    return compute_reward(row.episode_outcome(), cfg or row.config()).reward


def skip_reason(row: BankRow) -> str | None:
    """Why a rollout must not be imitated, whatever its reward."""
    o = row.episode_outcome()
    if row.task.get("split") != TRAIN_SPLIT:  # defence in depth: the stage reads train rows, a direct caller may not
        return SKIP_SPLIT
    if row.report_call is None:
        return SKIP_NO_REPORT
    if row.report_turn_extra_calls:
        return SKIP_HIDDEN_CALLS
    if o.report is None:
        return SKIP_UNPARSED
    if o.tampered:
        return SKIP_TAMPERED
    if o.violation:
        return SKIP_VIOLATION
    if not o.feasible and o.passed:
        return SKIP_LEAK
    return None


def _tiebreak(row: BankRow) -> tuple[int, str]:
    return len(row.completion), hashlib.sha256(row.episode_id.encode()).hexdigest()


def select_rows(
    rows: Iterable[BankRow],
    scorable: Iterable[BankRow],
    opts: BestOfNOptions,
    skipped: Counter[str] | None = None,
) -> list[Selection]:
    """The rollouts to imitate. `rows` are the candidates (already train-split, usable); `scorable` supplies the group
    means (every scored rollout of the same variants). Skip reasons are counted into `skipped`."""
    skipped = skipped if skipped is not None else Counter()
    cfg = opts.reward_config()
    candidates: dict[str, list[BankRow]] = defaultdict(list)
    for row in rows:
        candidates[variant_of(row)].append(row)
    rewards: dict[str, list[float]] = defaultdict(list)
    for row in scorable:
        if (v := variant_of(row)) in candidates and (r := _reward_of(row, cfg)) is not None:
            rewards[v].append(r)

    chosen: list[Selection] = []
    for variant in sorted(candidates):
        pool = rewards[variant]
        if len(pool) < opts.min_group_size:
            skipped[SKIP_SMALL_GROUP] += len(candidates[variant])
            continue
        mean = sum(pool) / len(pool)
        ranked: list[tuple[float, BankRow]] = []
        for row in candidates[variant]:
            if (reason := skip_reason(row)) is not None:
                skipped[reason] += 1
            elif (br := compute_reward(row.episode_outcome(), cfg or row.config())).masked:
                skipped[SKIP_MASKED] += 1
            elif br.case not in IMITABLE_CASES:
                skipped[SKIP_BAD_CASE] += 1
            else:
                ranked.append((br.total, row))
        ranked.sort(key=lambda p: (-p[0], _tiebreak(p[1])))
        best = ranked[0][1].episode_outcome() if ranked else None
        if opts.require_pass_when_feasible and best is not None and best.feasible and not best.passed:
            skipped[SKIP_NO_PASS] += len(ranked)
            continue
        for rank, (r, row) in enumerate(ranked, 1):
            if rank > opts.top_k:
                skipped[SKIP_NOT_TOP_K] += 1
            elif opts.above_mean and r <= mean + opts.min_margin + _EPS:
                skipped[SKIP_NOT_ABOVE_MEAN] += 1
            elif (
                opts.require_pass_when_feasible and row.episode_outcome().feasible and not row.episode_outcome().passed
            ):
                skipped[SKIP_NO_PASS] += 1
            else:
                chosen.append(Selection(row, r, mean, len(pool), rank))
    return chosen


# ── examples ─────────────────────────────────────────────────────────────────


def trajectory_messages(row: BankRow) -> list[dict[str, Any]]:
    """The whole banked trajectory: prompt, the completion before the report, and the report turn as it was generated,
    cut after the report call (later calls in that message never ran)."""
    final = normalize_message(row.report_message or {})
    final["tool_calls"] = final.get("tool_calls", [])[: row.report_turn_extra_calls + 1]
    return [*context_messages(row), final]


def with_confidence(messages: list[dict[str, Any]], p_complete: float) -> list[dict[str, Any]]:
    """`messages` with the final report call's confidence set to `p_complete` (abstain: its complement). Copies the
    report turn; the bank row is not touched."""
    final = {**messages[-1], "tool_calls": [dict(c) for c in messages[-1]["tool_calls"]]}
    call = final["tool_calls"][-1]
    name = (call.get("function") or {}).get("name")
    if name in ("submit_report", "abstain"):
        arguments = dict((call["function"].get("arguments")) or {})
        arguments["confidence"] = round(1.0 - p_complete, 4) if name == "abstain" else p_complete
        call["function"] = {**call["function"], "arguments": arguments}
    return [*messages[:-1], final]


def build_example(
    sel: Selection, table: hindsight.ConfidenceTable | None = None, keep_confidence: bool = True
) -> SFTExample:
    """The kept trajectory as an example. `keep_confidence=False` states the claim type's calibrated pass rate (`table`;
    0.5 where it has no value) instead of the confidence the model gave."""
    row = sel.row
    messages = trajectory_messages(row)
    if not keep_confidence:
        claim = hindsight.claim_type(row)
        p = table.p_complete(claim) if table is not None else None
        messages = with_confidence(messages, hindsight.NEUTRAL_P if p is None else p)
    return SFTExample(
        id=row.episode_id,
        kind=KIND,
        variant=variant_of(row),
        messages=messages,
        loss_messages=[i for i, m in enumerate(messages) if m.get("role") == "assistant"],
        meta={
            "reward": round(sel.reward, 4),
            "group_mean": round(sel.group_mean, 4),
            "group_size": sel.group_size,
            "rank": sel.rank,
            "passed": row.episode_outcome().passed,
            "tool": (row.report_call or {}).get("name"),
            "confidence_rewritten": not keep_confidence,
        },
    )


def selection_summary(chosen: list[Selection], skipped: Counter[str], n_rows: int) -> dict[str, Any]:
    gains = [s.reward - s.group_mean for s in chosen]
    return {
        "bestofn": {
            "candidate_rows": n_rows,
            "selected": len(chosen),
            "variants_selected": len({variant_of(s.row) for s in chosen}),
            "skipped": dict(skipped),
            "selected_passed": sum(s.row.episode_outcome().passed for s in chosen),
            "selected_reports": dict(Counter((s.row.report_call or {}).get("name") for s in chosen)),
            "mean_reward_gain": round(sum(gains) / len(gains), 4) if gains else 0.0,
        }
    }


# ── stages ───────────────────────────────────────────────────────────────────


def _build_part(rows: list[BankRow], scorable: list[BankRow], opts: BestOfNOptions) -> tuple[list[Selection], Counter]:
    skipped: Counter[str] = Counter()
    return select_rows(rows, scorable, opts, skipped), skipped


def _scorable_for(rows: list[BankRow], scorable: list[BankRow]) -> list[BankRow]:
    variants = {variant_of(r) for r in rows}
    return [r for r in scorable if variant_of(r) in variants]


def _confidence_table(cfg: SftConfig, held: list, scorable: list) -> hindsight.ConfidenceTable:
    """Pass rate by claim type over the held-out variants (never fitted), as `sft.hindsight` calibrates it."""
    held_ids = {variant_of(r) for r in held}
    return hindsight.calibrate(
        [r for r in scorable if variant_of(r) in held_ids], hindsight.HindsightOptions(**cfg.hindsight)
    )


def _bestofn_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    opts = parse_options(cfg.bestofn)
    chosen, skipped = _build_part(fit, _scorable_for(fit, scorable), opts)
    chosen_held, _ = _build_part(held, _scorable_for(held, scorable), opts)
    keep = opts.confidence == CONF_KEEP
    table = None if keep else _confidence_table(cfg, held, scorable)
    train = [build_example(s, table, keep) for s in chosen]
    evals = [build_example(s, table, keep) for s in chosen_held]
    return StageData(train, evals, selection_summary(chosen, skipped, len(fit)))


def _cap(examples: list[SFTExample], n: int) -> list[SFTExample]:
    """A stable pseudo-random subset of n examples (by id hash), so reruns and resumes see the same mixture."""
    if len(examples) <= n:
        return examples
    return sorted(examples, key=lambda e: hashlib.sha256(e.id.encode()).hexdigest())[:n]


def _combo_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    """Best-of-N trajectories plus hindsight report examples for the other rows, in one example list."""
    from honest_agents.sft.train_sft import _hindsight_stage

    opts = parse_options(cfg.bestofn)
    chosen, skipped = _build_part(fit, _scorable_for(fit, scorable), opts)
    chosen_held, _ = _build_part(held, _scorable_for(held, scorable), opts)
    taken = {s.row.episode_id for s in chosen}
    taken_held = {s.row.episode_id for s in chosen_held}
    # Calibration and labels come from the hindsight stage; the rows already imitated are not relabelled.
    report = _hindsight_stage(cfg, [r for r in fit if r.episode_id not in taken], held, scorable)
    report_held = [e for e in report.eval if e.id not in taken_held]
    hind = report.train
    n_best = len(chosen)
    if opts.hindsight_per_bestofn is not None:
        hind = _cap(hind, round(opts.hindsight_per_bestofn * n_best))
    keep = opts.confidence == CONF_KEEP
    table = None if keep else _confidence_table(cfg, held, scorable)
    train = [build_example(s, table, keep) for s in chosen] + hind
    evals = [build_example(s, table, keep) for s in chosen_held] + report_held
    info = {
        **report.info,
        **selection_summary(chosen, skipped, len(fit)),
        "mixture": {
            "bestofn_examples": n_best,
            "hindsight_examples": len(hind),
            "hindsight_available": len(report.train),
            "hindsight_per_bestofn": opts.hindsight_per_bestofn,
        },
    }
    return StageData(train, evals, info)


register_stage(KIND, _bestofn_stage, validate)
register_stage(KIND_COMBO, _combo_stage, validate)
