"""Bank rows to supervised examples, and the loss mask (shared by every SFT stage).

Two stages read the trajectory bank (`honest_agents.bank`) and train a policy with TRL's `SFTTrainer`:

* hindsight report SFT (`sft.hindsight`): the transcript stays as it was, its final report is rewritten from what the
  grader established, and ONLY the report tokens are trained;
* reward-filtered best-of-N (a later `sft.bestofn`): the best rollouts of each variant, every assistant turn trained.

What they share lives here: the row filter (train split only, no masked / tampered rows), the held-out variant
partition, the `SFTExample` schema, the renderer that turns a conversation into token ids plus an assistant-token mask,
per-message loss selection, a `datasets` dataset of `input_ids` / `labels`, and the within-group reward-spread check.

Mask conventions
================
* An `SFTExample` is a full conversation `messages` (system, user, then assistant / tool turns, the last one an
  assistant turn) and `loss_messages`, the indices into `messages` of the assistant turns that carry loss. Everything
  else (system, user, tool outputs, assistant turns not listed) has label -100.
* Tokenisation uses the chat template's own `{% generation %}` markers (`return_assistant_tokens_mask`): TRL's training
  template for Qwen3 / Qwen3.5 when the stock template has none. The k-th contiguous run of marked tokens is the k-th
  assistant message; a template that does not give one run per assistant turn is refused, never guessed at.
* Non-thinking mode renders an empty `<think></think>` block at the start of an assistant turn. At rollout time that block
  is part of the generation PROMPT, so it is taken out of the loss (the policy is never asked to emit it).
* The end-of-turn token of a trained assistant turn is inside its run (the model learns to stop after the report).
* Sequences longer than `max_length` are dropped (a report at the end of a truncated transcript would lose its loss).

The dataset holds only `input_ids` and `labels`: nested messages cannot go through Arrow (see `pilot.dataset`), and
pre-tokenised data bypasses TRL's own tokenisation, so the mask built here is the one used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import bank as bank_mod
from honest_agents.bank import BankRow
from honest_agents.models import ModelFamily, family_spec
from honest_agents.pilot.dataset import frozen_context, variant_of
from honest_agents.reward import RewardConfig, compute_reward

logger = logging.getLogger(__name__)

TRAIN_SPLIT = "train"  # the only split a supervised example may come from (validation and test are never read)
IGNORE_INDEX = -100
EMPTY_THINK = "<think>\n\n</think>\n\n"

# Reasons a bank row is not used, in the order they are checked.
DROP_SPLIT = "not_train_split"
DROP_MASKED = "masked"
DROP_QUARANTINED = "quarantined"
DROP_NO_PROMPT = "no_prompt"
DROP_LEAKY = "infeasible_clean_pass"
DROP_TAMPERED = "tampered"
DROP_VIOLATION = "violation"
DROP_NO_REPORT = "no_report"
DROP_UNSEEN = "unseen_actions"
DROP_TOO_LONG_CHARS = "too_long_chars"
DROP_TOO_LONG = "too_long_tokens"


# ── bank rows ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RowFilter:
    """Which bank rows may become examples. `split == "train"` and "not masked" are not options."""

    drop_tampered: bool = True  # the agent cannot see that a file was protected, so a label from it is unobservable
    drop_violations: bool = True  # same for an irreversible / forbidden action the agent did not know was forbidden
    drop_leaky: bool = True  # an infeasible variant whose tests passed cleanly: the variant label is wrong
    include_no_report: bool = True  # trajectories that ran out of budget (the context is cut after the last result)
    drop_unseen_actions: bool = True  # commands ran that the transcript does not show: the outcome is not inferable
    max_context_chars: int | None = None
    quarantine_file: str | None = None  # default: `quarantine.jsonl` inside the bank directory


def _quarantined(bank_path: str | Path, flt: RowFilter) -> set[str]:
    path = flt.quarantine_file
    if path is None and Path(bank_path).is_dir():
        path = str(Path(bank_path) / bank_mod.QUARANTINE_BASENAME)
    return bank_mod.quarantined_variants(path) if path is not None else set()


def _content_chars(messages: Sequence[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m, default=str)) for m in messages)


def base_drop_reason(row: BankRow, quarantined: set[str]) -> str | None:
    """Why the outcome of `row` cannot be used at all: not the train split, masked, or a quarantined variant."""
    o = row.episode_outcome()
    if row.task.get("split") != TRAIN_SPLIT:  # a missing split proves nothing
        return DROP_SPLIT
    if o.masked_reason is not None or (o.grading_error is not None and not o.acted) or row.reward.get("masked"):
        return DROP_MASKED
    if variant_of(row) in quarantined:
        return DROP_QUARANTINED
    return None


def drop_reason(row: BankRow, flt: RowFilter, quarantined: set[str]) -> str | None:
    """Why `row` must not become an example, or None. Never selects on anything but the train split."""
    if (reason := base_drop_reason(row, quarantined)) is not None:
        return reason
    o = row.episode_outcome()
    context = frozen_context(row)
    if not context.messages:
        return DROP_NO_PROMPT
    if flt.drop_leaky and not o.feasible and o.passed:
        return DROP_LEAKY
    if flt.drop_tampered and o.tampered:
        return DROP_TAMPERED
    if flt.drop_violations and o.violation:
        return DROP_VIOLATION
    if context.forced_report and not flt.include_no_report:
        return DROP_NO_REPORT
    if flt.drop_unseen_actions and context.unseen_actions > 0:
        return DROP_UNSEEN
    if flt.max_context_chars is not None and _content_chars(context.messages) > flt.max_context_chars:
        return DROP_TOO_LONG_CHARS
    return None


@dataclass
class BankSelection:
    usable: list[BankRow]  # pass `RowFilter`: may become examples
    scorable: list[BankRow]  # train split, not masked, not quarantined: their outcomes may be counted (calibration)
    dropped: Counter[str]  # reasons for the rows that are not `usable`


def read_rows(bank_path: str | Path, flt: RowFilter | None = None) -> BankSelection:
    """Rows of a bank (file or directory) split into usable and scorable, with a count of every drop reason."""
    flt = flt or RowFilter()
    quarantined = _quarantined(bank_path, flt)
    sel = BankSelection([], [], Counter())
    for row in bank_mod.iter_bank(bank_path):
        reason = drop_reason(row, flt, quarantined)
        if reason is None:
            sel.usable.append(row)
        else:
            sel.dropped[reason] += 1
        if base_drop_reason(row, quarantined) is None:
            sel.scorable.append(row)
    return sel


def variant_fraction(variant: str, seed: int) -> float:
    """A stable number in [0, 1) for a variant id: the same variant lands in the same partition in every run."""
    digest = hashlib.sha256(f"{seed}:{variant}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_heldout(row: BankRow, fraction: float, seed: int) -> bool:
    return fraction > 0 and variant_fraction(variant_of(row), seed) < fraction


def partition(rows: Iterable[BankRow], fraction: float, seed: int) -> tuple[list[BankRow], list[BankRow]]:
    """(fit rows, held-out rows), split by VARIANT so no variant is on both sides (rollouts of one variant share a
    prompt and would leak into the held-out entropy / calibration otherwise)."""
    fit: list[BankRow] = []
    held: list[BankRow] = []
    for row in rows:
        (held if is_heldout(row, fraction, seed) else fit).append(row)
    return fit, held


# ── examples ─────────────────────────────────────────────────────────────────


@dataclass
class SFTExample:
    id: str
    kind: str  # "hindsight" | "bestofn"
    variant: str
    messages: list[dict[str, Any]]  # normalised chat messages; the last one is an assistant turn
    loss_messages: list[int]  # indices into `messages` of the assistant turns that are trained
    meta: dict[str, Any] = field(default_factory=dict)  # audit only; never rendered

    def __post_init__(self) -> None:
        if not self.loss_messages:
            raise ValueError("an SFT example needs at least one trained message")
        for i in self.loss_messages:
            if not 0 <= i < len(self.messages) or self.messages[i].get("role") != "assistant":
                raise ValueError(f"loss message {i} of {self.id} is not an assistant turn")


def context_messages(row: BankRow) -> list[dict[str, Any]]:
    """The prompt and the transcript up to (not including) the report turn, in the form the policy was shown."""
    return frozen_context(row).messages


def write_examples(examples: Iterable[SFTExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(asdict(ex), default=str) + "\n" for ex in examples)
    return path


def read_examples(path: str | Path) -> list[SFTExample]:
    with open(path) as f:
        return [SFTExample(**json.loads(line)) for line in f if line.strip()]


# ── rendering and masks ──────────────────────────────────────────────────────


@dataclass
class ChatRenderer:
    """A tokenizer plus the template, template kwargs and tools used to render every example the same way."""

    tokenizer: Any
    chat_template: str | None  # a template with `{% generation %}` markers; None = the tokenizer's own
    kwargs: dict[str, Any]
    tools: list[dict[str, Any]] | None

    def render(self, messages: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
        """(token ids, assistant mask) of the conversation."""
        out = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            chat_template=self.chat_template,
            **self.kwargs,
        )
        ids, mask = list(out["input_ids"]), list(out["assistant_masks"])
        if len(ids) != len(mask):
            raise ValueError("assistant mask and token ids differ in length")
        return ids, mask


def make_renderer(
    tokenizer: Any,
    family: ModelFamily | str,
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    strict: bool = True,
) -> ChatRenderer:
    """The renderer for a model family. `tools=None` uses the shared bash + report tool schemas (`harness`), so the
    system block is the one the rollouts were conditioned on. A stock template without generation markers is swapped for
    TRL's training template, as `SFTTrainer(assistant_only_loss=True)` would do; the end-of-turn token must be trained."""
    from trl.chat_template_utils import (
        get_training_chat_template,
        has_generation_markers,
        is_chat_template_stop_token_trained,
    )

    if tools is None:
        from honest_agents.harness import tool_schemas

        tools = tool_schemas()
    template = tokenizer.chat_template
    if not has_generation_markers(template or ""):
        template = get_training_chat_template(tokenizer)
    if not is_chat_template_stop_token_trained(tokenizer, chat_template=template):
        msg = "the chat template leaves the end-of-turn token out of the assistant mask: the model would not learn to stop"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    kwargs = dict(family_spec(family).chat_template_kwargs) | dict(chat_template_kwargs or {})
    return ChatRenderer(tokenizer, template, kwargs, tools)


def mask_runs(mask: Sequence[int]) -> list[tuple[int, int]]:
    """(start, end) of every contiguous run of 1s."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, bit in enumerate(mask):
        if bit and start is None:
            start = i
        elif not bit and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _empty_think_ids(tokenizer: Any) -> list[int]:
    return list(tokenizer.encode(EMPTY_THINK, add_special_tokens=False))


@dataclass
class Tokenized:
    input_ids: list[int]
    labels: list[int]  # token id where trained, IGNORE_INDEX elsewhere

    @property
    def n_trained(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


def tokenize_example(ex: SFTExample, renderer: ChatRenderer, max_length: int | None = None) -> Tokenized | str:
    """Token ids and labels of one example, or a drop reason (`DROP_TOO_LONG`). Raises when the template does not give
    one assistant run per assistant message, or the selected turns come out empty."""
    ids, mask = renderer.render(ex.messages)
    if max_length is not None and len(ids) > max_length:
        return DROP_TOO_LONG
    runs = mask_runs(mask)
    assistant_index = [i for i, m in enumerate(ex.messages) if m.get("role") == "assistant"]
    if len(runs) != len(assistant_index):
        raise ValueError(
            f"{ex.id}: the template marks {len(runs)} assistant runs for {len(assistant_index)} assistant messages"
        )
    run_of = dict(zip(assistant_index, runs, strict=True))
    think = _empty_think_ids(renderer.tokenizer) if not renderer.kwargs.get("enable_thinking", True) else []
    labels = [IGNORE_INDEX] * len(ids)
    for i in ex.loss_messages:
        start, end = run_of[i]
        if think and ids[start : start + len(think)] == think:
            start += len(think)
        labels[start:end] = ids[start:end]
    out = Tokenized(ids, labels)
    if out.n_trained == 0:
        raise ValueError(f"{ex.id}: no trained tokens")
    return out


def tokenize_examples(
    examples: Iterable[SFTExample], renderer: ChatRenderer, max_length: int | None = None
) -> tuple[list[Tokenized], list[SFTExample], Counter[str]]:
    """(tokenised, the examples they came from, drop counts). Examples that are too long are dropped, not truncated."""
    tokenized: list[Tokenized] = []
    kept: list[SFTExample] = []
    dropped: Counter[str] = Counter()
    for ex in examples:
        t = tokenize_example(ex, renderer, max_length)
        if isinstance(t, str):
            dropped[t] += 1
        else:
            tokenized.append(t)
            kept.append(ex)
    return tokenized, kept, dropped


def to_hf_dataset(tokenized: Sequence[Tokenized]):
    """A `datasets.Dataset` of `input_ids` and `labels`: `SFTTrainer` uses `labels` as they are."""
    from datasets import Dataset

    if not tokenized:
        raise ValueError("no examples to train on")
    return Dataset.from_list([{"input_ids": t.input_ids, "labels": t.labels} for t in tokenized])


# ── reward spread ────────────────────────────────────────────────────────────


@dataclass
class SpreadStats:
    n_rows: int  # scored rows (masked ones are skipped)
    n_groups: int  # groups with at least `min_group_size` scored rows
    n_with_variance: int
    mean_std: float  # mean within-group standard deviation of the reward

    @property
    def fraction(self) -> float:
        """Share of groups whose rewards are not all equal: GRPO learns nothing from the others."""
        return self.n_with_variance / self.n_groups if self.n_groups else 0.0


def reward_spread(
    rows: Iterable[BankRow],
    cfg: RewardConfig | None = None,
    *,
    min_group_size: int = 2,
    group_of: Callable[[BankRow], str] = variant_of,
    eps: float = 1e-9,
) -> SpreadStats:
    """Within-group reward variance of a bank, scored with `reward.compute_reward` from each stored outcome (under `cfg`,
    default: the row's own config). Run it on a bank sampled after the SFT stage: if the fraction collapses compared to
    the bank the SFT started from, the SFT narrowed the policy too much for GRPO."""
    groups: dict[str, list[float]] = defaultdict(list)
    n_rows = 0
    for row in rows:
        value = compute_reward(row.episode_outcome(), cfg or row.config()).reward
        if value is None:
            continue
        groups[group_of(row)].append(value)
        n_rows += 1
    sized = [v for v in groups.values() if len(v) >= min_group_size]
    stds = [statistics.pstdev(v) for v in sized]
    return SpreadStats(n_rows, len(sized), sum(s > eps for s in stds), statistics.fmean(stds) if stds else 0.0)
