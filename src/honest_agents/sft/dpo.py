"""Decision-state DPO (iteration 14, docs/iteration-14-design.md section 3): preference pairs at the spliced decision
states of `sft.decision`, trained with TRL's `DPOTrainer` on the merged SFT policy.

A decision state is the prefix of a base-model high_risk control episode plus the injected read turns with the real
sandbox outputs of ONE variant: `ls -la` and `cat README` for an iteration-13 README pair, the 2-4
`inspection_commands(site)` of an authored-surface pair (`decision.surfaces`, SFT-v2). The state ends where the example's
own `loss_roles["decision"]` says the deletion turn starts, so the number of read turns is derived from the example, not
assumed. Each built pair of `sft.decision` (a control and a twin of
the same source episode) gives two preference pairs, one per side of the evidence:

* **benign** (the control's README, junk contents): chosen = the model's own destructive turn (proceed with the
  deletion), rejected = a decline report built the way `decision.twin_report_args` builds the twin report but with the
  control's facts (target, README text, what; for an authored surface the surface's report template quoting the
  control's evidence lines and citing the command that printed them);
* **danger** (the twin's README, the only copy): chosen = the twin decline report (the twin's surface report), rejected = the model's own destructive
  turn (its tool call verbatim; the content is a plain "removing it as requested" line, because the content written
  after the benign README would argue from facts the twin's README contradicts).

The prompt of a pair is the whole shared conversation up to the last tool message; chosen and rejected are the final
assistant message only, so the two differ in nothing else. The decline report lists the own `completed` items, so a
state is used only when the deletion was the LAST state change of the episode (`DecisionState.deletion_last_state_change`;
otherwise the report at the decision state would claim work the own tail did later). Rendering is done here, not by TRL: the prompt goes through
`sft.data.ChatRenderer` (the deployed template, kwargs and tools; TRL's conversational path would re-template without
the tools), and `prompt` / `chosen` / `rejected` reach `DPOTrainer` as plain strings. TRL appends the tokenizer's EOS
to a string completion unless it already ends with it, so the template's trailing newline after `<|im_end|>` is cut here
(`render_pair`); `check_tokenization` asserts that prompt + chosen tokenises to exactly what `sft.data.tokenize_example`
gives for the same message list (up to that newline) and that the trained span is exactly the chosen ids.

Training: `peft` LoRA on `model.name`, which is the MERGED SFT-v2 policy (`sft.merge`); the reference is the same model
with the adapter disabled, its log-probs precomputed (`precompute_ref_log_probs`) so only one copy of the weights lives
on the GPU. `max_length` is applied by TRL's collator with `truncation_mode="keep_end"`, which would cut the system
block and tools off the front of an over-long sequence, so `build_pairs` DROPS pairs longer than `dpo.max_length`
instead and the truncation never fires (it stays configured as the last-resort setting the design asks for).

Loss: `sigmoid` (summed completion log-probs) plus `sft` (mean NLL of the chosen completion) by default. Each pair is
length-asymmetric (deletion turn ~60 tokens, decline report ~200-250) but the asymmetry is mirrored across the two sides,
so there is no global length preference to learn; the `sft` term anchors the chosen completion against the likelihood
displacement plain DPO shows on small pair sets (both completions' log-probs falling, the rejected faster). The
length-normalised variants (`sigmoid_norm`, `ipo`) are not drop-in replacements at beta 0.1: per-token averages make
beta * delta ~0.01-0.05 (near-zero gradient) and IPO's target margin 1/(2 beta) = 5 nats per token.

    python -m honest_agents.sft.dpo --config configs/sft_dpo_decision_qwen35.yaml --dry-run \
        --review data/decision/dpo_review.md model.name=<merged SFT-v2 dir>
    python -m honest_agents.sft.dpo --config configs/sft_dpo_decision_qwen35.yaml model.name=<merged SFT-v2 dir>   # GPU
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.harness import report_from_tool_args
from honest_agents.pilot.run import apply_overrides, load_config_dict
from honest_agents.sft import data
from honest_agents.sft import decision as dec
from honest_agents.sft.data import ChatRenderer, SFTExample, tokenize_example
from honest_agents.sft.train_sft import SftConfig, build_lora_config, parse_config, resolve_family
from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks import decision_surfaces as dsurf
from honest_agents.tasks.decision_probe import VariantMeta

logger = logging.getLogger("honest_agents.sft.dpo")

SIDE_BENIGN = "benign"
SIDE_DANGER = "danger"
SIDES = (SIDE_BENIGN, SIDE_DANGER)
MIN_MAX_LENGTH = 12288  # design [R-DPO]: >= 12k; the built pairs are 2.5-9.1k tokens, truncation never fires
TRUNCATION_MODES = ("keep_end", "keep_start")
# TRL 1.14.1 `DPOTrainer._compute_loss`; an unknown name there fails only at the first training step, after the model
# load and the reference pass, so it is checked at parse time.
LOSS_TYPES = (
    "sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot", "aot_unpaired",
    "apo_zero", "apo_down", "discopop", "sft", "sigmoid_norm",
)  # fmt: skip
DROP_NOT_LAST_STATE_CHANGE = "deletion_not_last_state_change"
DROP_NO_EVIDENCE = "surface_no_evidence_lines"

# The rejected destructive turn on danger evidence: the failure mode is deleting without heeding the README, so the
# prose does not mention it (rule A of `sft.decision`: a line written after the benign README does not belong here).
DESTRUCT_LINES = (
    "Now let me remove `{dir}` as requested:",
    "Removing `{dir}` now:",
    "Next, deleting the `{dir}` directory tree:",
    "Let me clear out `{dir}`:",
)
assert len(DESTRUCT_LINES) == dec.N_TEMPLATES


# ── config ───────────────────────────────────────────────────────────────────


@dataclass
class DpoSection:
    """The `dpo:` section: what DPOTrainer needs beyond the shared `train:` section. Names follow TRL 1.14.1 `DPOConfig`."""

    beta: float = 0.1
    loss_type: list[str] = field(default_factory=lambda: ["sigmoid", "sft"])
    loss_weights: list[float] | None = None
    max_length: int = 16384  # TRL `max_length`; pairs longer than this are dropped before training
    truncation_mode: str = "keep_end"
    precompute_ref_log_probs: bool = True
    precompute_ref_batch_size: int | None = None
    sides: list[str] = field(default_factory=lambda: list(SIDES))  # which preference sides to build

    def __post_init__(self) -> None:
        if self.max_length < MIN_MAX_LENGTH:
            raise ValueError(f"dpo.max_length must be >= {MIN_MAX_LENGTH} (design [R-DPO]: no pair is ever truncated)")
        if self.truncation_mode not in TRUNCATION_MODES:
            raise ValueError(f"dpo.truncation_mode must be one of {TRUNCATION_MODES}")
        if self.beta <= 0:
            raise ValueError("dpo.beta must be > 0")
        if isinstance(self.loss_type, str):
            self.loss_type = [self.loss_type]
        if not self.loss_type:
            raise ValueError("dpo.loss_type is empty")
        if unknown := set(self.loss_type) - set(LOSS_TYPES):
            raise ValueError(f"unknown dpo.loss_type {sorted(unknown)}; TRL 1.14.1 knows {list(LOSS_TYPES)}")
        if unknown := set(self.sides) - set(SIDES):
            raise ValueError(f"unknown dpo.sides {sorted(unknown)}; expected {list(SIDES)}")
        if not self.sides:
            raise ValueError("dpo.sides is empty")
        if self.loss_weights is not None and len(self.loss_weights) != len(self.loss_type):
            raise ValueError("dpo.loss_weights must have one weight per dpo.loss_type")


@dataclass
class DpoConfig:
    """An `SftConfig` of `kind: decision` (model, lora, data, decision options, train) plus the `dpo:` section."""

    sft: SftConfig
    dpo: DpoSection

    @property
    def output_dir(self) -> Path:
        return self.sft.output_dir


def parse_dpo_config(raw: dict[str, Any]) -> DpoConfig:
    raw = dict(raw)
    dpo_raw = dict(raw.pop("dpo", None) or {})
    allowed = {f.name for f in dataclasses.fields(DpoSection)}
    if unknown := set(dpo_raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `dpo:`; expected {sorted(allowed)}")
    sft = parse_config(raw)
    if sft.kind != dec.KIND:
        raise ValueError(f"the DPO stage builds on the decision dataset: kind must be {dec.KIND!r}, got {sft.kind!r}")
    return DpoConfig(sft, DpoSection(**dpo_raw))


def load_dpo_config(path: str | Path, overrides: list[str] | None = None) -> DpoConfig:
    return parse_dpo_config(apply_overrides(load_config_dict(path), list(overrides or [])))


# ── preference pairs ─────────────────────────────────────────────────────────


@dataclass
class PreferencePair:
    id: str
    side: str  # SIDE_BENIGN | SIDE_DANGER
    prompt: list[dict[str, Any]]  # the shared conversation, ending with the last tool message
    chosen: dict[str, Any]  # the final assistant message
    rejected: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def chosen_messages(self) -> list[dict[str, Any]]:
        return [*self.prompt, self.chosen]

    def rejected_messages(self) -> list[dict[str, Any]]:
        return [*self.prompt, self.rejected]

    def example(self, which: str = "chosen") -> SFTExample:
        """The pair's chosen (or rejected) side as an SFT example with loss on the final turn only."""
        messages = self.chosen_messages() if which == "chosen" else self.rejected_messages()
        return SFTExample(f"{self.id}:{which}", dec.KIND, str(self.meta.get("variant", "")), messages, [len(messages) - 1], self.meta)


def _prompt_commands(prompt: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every bash command the policy issued in the prompt (rule C: evidence commands may only come from these)."""
    return [c for turn in dec.turns_of(prompt[dec.completion_start(prompt) :]) for c in turn.commands()]


class NoEvidenceError(ValueError):
    """The decision state's last tool output holds no evidence line to quote (an authored-surface decline report)."""


def decline_turn(own_args: Mapping[str, Any], meta: VariantMeta, template: int, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The `submit_report` decline turn at this decision state: the twin report of `sft.decision` with `meta`'s facts and
    evidence commands taken from the prompt only. For an authored-surface variant it is the surface's report template,
    quoting the evidence lines of the prompt's last tool output (the output of the last inspection command, as in
    `decision.build_pair`) and citing that command."""
    commands = _prompt_commands(prompt)
    if not meta.from_pool:
        args = dec.twin_report_args(own_args, meta, commands, template)
        content = dec._fill(dec.REPORT_TEMPLATES[template].content, meta)
    else:
        lines = dec.evidence_lines(str(prompt[-1].get("content") or ""))
        if not lines:
            raise NoEvidenceError(f"{meta.variant_id}: the last tool output of the prompt has no evidence line")
        quote = dec.quote_of(lines)
        args = dec.twin_report_args(own_args, meta, commands, template, quote=quote, read_command=meta.commands[-1])
        content = dec._fill_surface(dec.SURFACE_REPORT_TEMPLATES[template].content, meta, quote)
    report_from_tool_args("submit_report", args)  # must parse
    return dec.report_turn(args, content)


def destructive_turn(turn: Mapping[str, Any], meta: VariantMeta, template: int) -> dict[str, Any]:
    """The own deletion turn, tool call verbatim, with a content line that does not argue from a benign README."""
    return {**turn, "content": DESTRUCT_LINES[template].format(dir=meta.target)}


def decision_boundary(example: SFTExample) -> int:
    """Index of the deletion turn in `example.messages` = the length of the shared decision-state prompt, from the roles
    `sft.decision` recorded: the prefix, then one assistant turn and one tool output per injected read, then the decision."""
    roles = example.meta["loss_roles"]
    n = roles["decision"]
    reads = roles["reads"]
    n_prefix = int(example.meta["n_prefix_messages"])
    if not isinstance(n, int) or not reads or n != n_prefix + 2 * len(reads) or reads != [n_prefix + 2 * k for k in range(len(reads))]:
        raise ValueError(f"{example.id}: loss_roles {roles} do not describe {n_prefix} prefix messages and consecutive read turns")
    return n


def pairs_of(
    built: dec.BuiltPair, twin_meta: VariantMeta, control_meta: VariantMeta, sides: Sequence[str] = SIDES
) -> list[PreferencePair]:
    """The preference pairs of one built decision pair (see the module docstring). Prompt = the messages up to the last
    injected tool output; chosen / rejected = the final assistant message."""
    n = decision_boundary(built.control)
    n_prefix = int(built.control.meta["n_prefix_messages"])
    if built.twin.meta["loss_roles"]["reads"] != built.control.meta["loss_roles"]["reads"]:
        raise ValueError(f"{built.state.episode_id}: twin and control read a different number of times")
    template = int(built.control.meta["template"])
    row = built.state.row
    own_args = (row.report_call or {}).get("arguments") or {}
    control_prompt = [dict(m) for m in built.control.messages[:n]]
    twin_prompt = [dict(m) for m in built.twin.messages[:n]]
    if control_prompt[:n_prefix] != twin_prompt[:n_prefix]:
        raise ValueError(f"{built.state.episode_id}: twin and control prompts differ before the injected reads")
    own_deletion = built.control.messages[n]
    if own_deletion.get("role") != "assistant":
        raise ValueError(f"{built.state.episode_id}: message {n} of the control is not the deletion turn")
    pair_key = built.control.id.removesuffix(dec.SIDE_CONTROL)  # `<episode>:` or `<episode>:<surface>:`
    surface = dec.surface_key(built)
    base = {
        "pair_id": built.state.pair_id,
        "source_episode": built.state.episode_id,
        "surface": surface,
        "template": template,
        "phrasing_index": built.control.meta.get("phrasing_index"),
        "n_prompt_messages": n,
        "n_reads": len(built.control.meta["loss_roles"]["reads"]),
    }
    out: list[PreferencePair] = []
    if SIDE_BENIGN in sides:
        out.append(PreferencePair(
            f"{pair_key}{SIDE_BENIGN}", SIDE_BENIGN, control_prompt,
            chosen=dict(own_deletion),
            rejected=decline_turn(own_args, control_meta, template, control_prompt),
            meta={**base, "side": SIDE_BENIGN, "variant": control_meta.variant_id, "readme": control_meta.readme},
        ))  # fmt: skip
    if SIDE_DANGER in sides:
        out.append(PreferencePair(
            f"{pair_key}{SIDE_DANGER}", SIDE_DANGER, twin_prompt,
            chosen=decline_turn(own_args, twin_meta, template, twin_prompt),
            rejected=destructive_turn(own_deletion, twin_meta, template),
            meta={**base, "side": SIDE_DANGER, "variant": twin_meta.variant_id, "readme": twin_meta.readme},
        ))  # fmt: skip
    return out


@dataclass
class PairSet:
    fit: list[PreferencePair]
    held: list[PreferencePair]
    info: dict[str, Any]

    @property
    def all(self) -> list[PreferencePair]:
        return [*self.fit, *self.held]


def _variant_refs(opts: dec.DecisionOptions) -> dict[str, dp.VariantRef]:
    """variant id -> task directory reference of every variant a built pair can use: the pool pairs the decision dataset
    draws its sources from (the same merge as `decision.build_dataset`) and, with `decision.surfaces`, the manifest's
    SFT surface variants."""
    sources = dec._pool_pairs(opts.distill_pool, opts.tasks_root)
    if opts.include_rl_pool_sources:
        sources = {**dec._pool_pairs(opts.rl_pool, opts.tasks_root), **sources}
    refs = {ref.variant_id: ref for pair in sources.values() for ref in pair.values()}
    if opts.surfaces is not None:
        for row in dsurf.read_manifest(opts.surfaces.manifest):
            if row.role == dsurf.ROLE_SFT:
                refs |= {ref.variant_id: ref for ref in row.refs()}
    return refs


def pairs_from_dataset(
    ds: dec.Dataset, opts: dec.DecisionOptions, sides: Sequence[str] = SIDES
) -> tuple[list[PreferencePair], Counter[str]]:
    """(preference pairs, dropped states by reason). A state whose deletion was not the episode's last state change is
    dropped: its decline report would list `completed` work the own tail did only after the deletion (the SFT twin keeps
    that tail before its report; the DPO completion follows the decision state directly). A surface state whose last
    inspection output has no evidence line to quote is dropped too."""
    refs = _variant_refs(opts)
    metas: dict[str, VariantMeta] = {}

    def meta_of(variant_id: str) -> VariantMeta:
        if variant_id not in metas:
            metas[variant_id] = dp.read_variant_meta(refs[variant_id].task_dir)
        return metas[variant_id]

    out: list[PreferencePair] = []
    dropped: Counter[str] = Counter()
    for built in ds.pairs:
        if not built.state.deletion_last_state_change:
            dropped[DROP_NOT_LAST_STATE_CHANGE] += 1
            continue
        try:
            out += pairs_of(built, meta_of(built.twin.meta["variant_id"]), meta_of(built.control.meta["variant_id"]), sides)
        except NoEvidenceError:
            dropped[f"{DROP_NO_EVIDENCE}:{dec.surface_key(built)}"] += 1
    return out, dropped


def build_pairs(cfg: DpoConfig) -> PairSet:
    """Read the bank, partition it as the decision stage does (`data.partition_by`), build the decision datasets and
    turn every built pair into its preference pairs."""
    sft = cfg.sft
    opts = dec.parse_options(sft.decision)
    sel = data.read_rows(sft.data.bank, data.RowFilter(**sft.data.row_filter))
    fit_rows, held_rows = data.partition(
        sel.usable, sft.data.holdout_fraction, sft.seed, data.PARTITION_KEYS[sft.data.partition_by]
    )
    fit_ds, held_ds = dec.build_dataset(fit_rows, opts), dec.build_dataset(held_rows, opts)
    fit, dropped_fit = pairs_from_dataset(fit_ds, opts, cfg.dpo.sides)
    held, dropped_held = pairs_from_dataset(held_ds, opts, cfg.dpo.sides)
    info = {
        "state_pairs_fit": len(fit_ds.pairs),
        "state_pairs_heldout": len(held_ds.pairs),
        "dropped_fit": dict(sorted(fit_ds.dropped.items())),
        "dropped_states": dict(sorted((dropped_fit + dropped_held).items())),  # built states this stage left out
        "sides": list(cfg.dpo.sides),
    }
    return PairSet(fit, held, info)


# ── rendering ────────────────────────────────────────────────────────────────


@dataclass
class RenderedPair:
    prompt: str
    chosen: str
    rejected: str


def _template_text(renderer: ChatRenderer, messages: list[dict[str, Any]], *, generation_prompt: bool) -> str:
    return renderer.tokenizer.apply_chat_template(
        messages,
        tools=renderer.tools,
        tokenize=False,
        add_generation_prompt=generation_prompt,
        chat_template=renderer.chat_template,
        **renderer.kwargs,
    )


def render_pair(pair: PreferencePair, renderer: ChatRenderer) -> RenderedPair:
    """`prompt` / `chosen` / `rejected` strings for TRL's standard format. The prompt ends with the generation prompt
    (including the empty think block in non-thinking mode), exactly the text the policy is sampled from; each
    completion is the rest of the full conversation's rendering, cut after the end-of-turn token (TRL adds the EOS
    itself to a string that does not already end with it, and the template's newline after `<|im_end|>` is never
    sampled)."""
    eos = renderer.tokenizer.eos_token
    prompt = _template_text(renderer, pair.prompt, generation_prompt=True)
    texts = []
    for completion in (pair.chosen, pair.rejected):
        full = _template_text(renderer, [*pair.prompt, completion], generation_prompt=False)
        if not full.startswith(prompt):
            raise ValueError(f"{pair.id}: the rendered prompt is not a prefix of the rendered conversation")
        text = full[len(prompt) :].rstrip()
        if not text.endswith(eos):
            raise ValueError(f"{pair.id}: a rendered completion does not end with the end-of-turn token {eos!r}")
        texts.append(text)
    return RenderedPair(prompt, texts[0], texts[1])


def trl_ids(renderer: ChatRenderer, rendered: RenderedPair) -> tuple[list[int], list[int], list[int]]:
    """(prompt ids, chosen ids, rejected ids) as `DPOTrainer._prepare_dataset` computes them for string data."""
    tok = renderer.tokenizer

    def ids(text: str) -> list[int]:
        return list(tok(text=text)["input_ids"])

    prompt = ids(rendered.prompt)
    out = []
    for completion in (rendered.chosen, rendered.rejected):
        full = ids(rendered.prompt + completion)
        if full[: len(prompt)] != prompt:
            raise ValueError("the tokenised prompt is not a prefix of the tokenised prompt + completion")
        out.append(full[len(prompt) :])
    return prompt, out[0], out[1]


def check_tokenization(pair: PreferencePair, renderer: ChatRenderer, rendered: RenderedPair | None = None) -> list[str]:
    """Reasons the TRL path would differ from `tokenize_example` of the same message list (empty = identical). Compared
    for the chosen and the rejected side: the ids (up to the newline after the end-of-turn token) and the trained span."""
    rendered = rendered or render_pair(pair, renderer)
    prompt_ids, chosen_ids, rejected_ids = trl_ids(renderer, rendered)
    problems: list[str] = []
    for which, completion_ids in (("chosen", chosen_ids), ("rejected", rejected_ids)):
        tokenized = tokenize_example(pair.example(which), renderer)
        if isinstance(tokenized, str):
            problems.append(f"{which}: {tokenized}")
            continue
        ids = list(tokenized.input_ids)
        while len(ids) > 1 and not renderer.tokenizer.decode([ids[-1]]).strip():
            ids.pop()  # the newline after <|im_end|>
        if prompt_ids + completion_ids != ids:
            problems.append(f"{which}: TRL ids differ from tokenize_example ({len(prompt_ids) + len(completion_ids)} vs {len(ids)})")
        trained = [t for t in tokenized.labels if t != data.IGNORE_INDEX]
        if trained != completion_ids:
            problems.append(f"{which}: the trained span of tokenize_example differs from the completion ids")
    return problems


@dataclass
class DpoData:
    train: list[dict[str, str]]
    eval: list[dict[str, str]]
    stats: dict[str, Any]
    kept: list[PreferencePair]
    rendered: list[RenderedPair]


def tokenize_pairs(pairs: PairSet, renderer: ChatRenderer, max_length: int, *, verify: bool = True) -> DpoData:
    """Render every pair, drop those longer than `max_length` (prompt + the longer completion), optionally verify the
    tokenisation, and collect length statistics."""
    train: list[dict[str, str]] = []
    evals: list[dict[str, str]] = []
    kept: list[PreferencePair] = []
    rendered_all: list[RenderedPair] = []
    dropped: Counter[str] = Counter()
    lengths: dict[str, list[int]] = {"prompt": [], "chosen": [], "rejected": [], "total": []}
    by_side: Counter[str] = Counter()
    by_surface: Counter[str] = Counter()
    held_ids = {p.id for p in pairs.held}
    for pair in pairs.all:
        rendered = render_pair(pair, renderer)
        if verify and (problems := check_tokenization(pair, renderer, rendered)):
            raise ValueError(f"{pair.id}: {'; '.join(problems)}")
        prompt_ids, chosen_ids, rejected_ids = trl_ids(renderer, rendered)
        total = len(prompt_ids) + max(len(chosen_ids), len(rejected_ids))
        if total > max_length:
            dropped["too_long"] += 1
            continue
        for key, n in (("prompt", len(prompt_ids)), ("chosen", len(chosen_ids)), ("rejected", len(rejected_ids)), ("total", total)):
            lengths[key].append(n)
        row = {"prompt": rendered.prompt, "chosen": rendered.chosen, "rejected": rendered.rejected}
        (evals if pair.id in held_ids else train).append(row)
        kept.append(pair)
        rendered_all.append(rendered)
        by_side[pair.side] += 1
        by_surface[str(pair.meta["surface"])] += 1
    stats = {
        "pairs_train": len(train),
        "pairs_eval": len(evals),
        "by_side": dict(by_side),
        "by_surface": dict(sorted(by_surface.items())),
        "dropped": dict(dropped),
        "tokens": {k: {"min": min(v), "mean": round(statistics.fmean(v), 1), "max": max(v)} for k, v in lengths.items() if v},
    }
    return DpoData(train, evals, stats, kept, rendered_all)


def to_hf_dataset(rows: Sequence[dict[str, str]]):
    from datasets import Dataset

    if not rows:
        raise ValueError("no preference pairs to train on")
    return Dataset.from_list(list(rows))


# ── review file ──────────────────────────────────────────────────────────────


def _render_message(m: Mapping[str, Any]) -> str:
    lines = []
    if m.get("content"):
        lines.append(str(m["content"]))
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        lines.append(f"<{fn.get('name')}> " + json.dumps(fn.get("arguments"), ensure_ascii=False))
    return "\n".join(lines) or "(empty)"


def write_review(pairs: Sequence[PreferencePair], path: str | Path, stats: Mapping[str, Any], n_per_side: int = 3) -> Path:
    """Counts, token lengths and `n_per_side` pairs of each side (stable hash order): the last shared turns, the chosen
    and the rejected completion."""
    out = ["# Decision-state DPO review", "", "```json", json.dumps(stats, indent=2, default=str), "```", ""]
    for side in SIDES:
        out.append(f"## {side} evidence")
        chosen = sorted((p for p in pairs if p.side == side), key=lambda p: dec._hash(p.id))[:n_per_side]
        for p in chosen:
            head = (
                f"### {p.id}  (surface {p.meta['surface']}, template {p.meta['template']}, phrasing {p.meta['phrasing_index']}, "
                f"{p.meta['n_prompt_messages']} prompt messages, {p.meta['n_reads']} reads)"
            )
            out += ["", head, "", f"README: {p.meta['readme']}", ""]
            out += ["shared prompt, the injected reads and their outputs:", "```"]
            out += [f"[{m['role']}] {_render_message(m)}" for m in p.prompt[-2 * p.meta["n_reads"] :]]
            out += ["```", "", "**chosen**:", "```", _render_message(p.chosen), "```", "",
                    "**rejected**:", "```", _render_message(p.rejected), "```"]  # fmt: skip
        out.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    return path


# ── trainer ──────────────────────────────────────────────────────────────────


def build_dpo_config(cfg: DpoConfig, *, has_eval: bool):
    """TRL 1.14.1 `DPOConfig`. Argument names checked against `trl/trainer/dpo_config.py`: `beta`, `loss_type` (list),
    `loss_weights`, `max_length`, `truncation_mode`, `precompute_ref_log_probs`, `precompute_ref_batch_size`,
    `model_init_kwargs`; the rest are `TrainingArguments`."""
    from trl import DPOConfig

    sft, d, t = cfg.sft, cfg.dpo, cfg.sft.train
    init_kwargs: dict[str, Any] = {"dtype": sft.model.dtype}
    if sft.model.revision:
        init_kwargs["revision"] = sft.model.revision
    if sft.model.attn_implementation:
        init_kwargs["attn_implementation"] = sft.model.attn_implementation
    kwargs: dict[str, Any] = {
        "output_dir": str(sft.output_dir),
        "seed": sft.seed,
        "learning_rate": t.learning_rate,
        "num_train_epochs": t.num_train_epochs,
        "max_steps": t.max_steps,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "warmup_steps": t.warmup_steps,
        "lr_scheduler_type": t.lr_scheduler_type,
        "weight_decay": t.weight_decay,
        "max_grad_norm": t.max_grad_norm,
        "gradient_checkpointing": t.gradient_checkpointing,
        "bf16": t.bf16,
        "logging_steps": t.logging_steps,
        "save_strategy": "steps",
        "save_steps": t.save_steps,
        "save_total_limit": t.save_total_limit,
        "report_to": t.report_to,
        "model_init_kwargs": init_kwargs,
        "remove_unused_columns": False,
        "eval_strategy": "epoch" if has_eval else "no",
        "beta": d.beta,
        "loss_type": list(d.loss_type),
        "loss_weights": d.loss_weights,
        "max_length": d.max_length,
        "truncation_mode": d.truncation_mode,
        "precompute_ref_log_probs": d.precompute_ref_log_probs,
        "precompute_ref_batch_size": d.precompute_ref_batch_size,
        **t.extra,
    }
    return DPOConfig(**kwargs)


def load_tokenizer(cfg: DpoConfig):
    from honest_agents.sft.train_sft import load_tokenizer as _load

    return _load(cfg.sft)


def prepare(cfg: DpoConfig, tokenizer: Any, *, verify: bool = True) -> tuple[PairSet, DpoData, ChatRenderer]:
    pairs = build_pairs(cfg)
    renderer = data.make_renderer(tokenizer, resolve_family(cfg.sft))
    return pairs, tokenize_pairs(pairs, renderer, cfg.dpo.max_length, verify=verify), renderer


def run_dpo(cfg: DpoConfig, *, resume: bool = False) -> dict[str, Any]:  # pragma: no cover - needs a GPU and the merged model
    """GPU path: train and save the adapter to `<output_dir>/final` (and one checkpoint per epoch)."""
    from trl import DPOTrainer

    tokenizer = load_tokenizer(cfg)
    pairs, dpo_data, _ = prepare(cfg, tokenizer)
    if len(dpo_data.train) < cfg.sft.data.min_examples:
        raise ValueError(f"only {len(dpo_data.train)} preference pairs (data.min_examples={cfg.sft.data.min_examples})")
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    eval_ds = to_hf_dataset(dpo_data.eval) if dpo_data.eval else None
    trainer = DPOTrainer(
        model=cfg.sft.model.name,  # the merged SFT-v2 policy; the reference is this model with the adapter disabled
        args=build_dpo_config(cfg, has_eval=eval_ds is not None),
        train_dataset=to_hf_dataset(dpo_data.train),
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg.sft, resolve_family(cfg.sft)),
    )
    last = None
    if resume:
        from transformers.trainer_utils import get_last_checkpoint

        last = get_last_checkpoint(str(out)) if out.is_dir() else None
    result = trainer.train(resume_from_checkpoint=last)
    trainer.save_model(str(out / "final"))
    summary = {**pairs.info, **dpo_data.stats, "train_loss": result.training_loss, "log_history": trainer.state.log_history}
    (out / "dpo_run.json").write_text(json.dumps({"config": dataclasses.asdict(cfg), "summary": summary}, default=str, indent=2))
    return summary


def dry_run(cfg: DpoConfig, tokenizer: Any = None, review: str | Path | None = None) -> dict[str, Any]:
    """Everything except the model: pairs, rendering, the tokenisation check, the `DPOConfig` and the LoRA config."""
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)
    pairs, dpo_data, _ = prepare(cfg, tokenizer)
    dpo_config = build_dpo_config(cfg, has_eval=bool(dpo_data.eval))
    lora = build_lora_config(cfg.sft, resolve_family(cfg.sft))
    summary: dict[str, Any] = {
        **pairs.info,
        **dpo_data.stats,
        "output_dir": dpo_config.output_dir,
        "base_model": cfg.sft.model.name,
        "lora": {"r": lora.r, "alpha": lora.lora_alpha},
        "dpo": {
            "beta": dpo_config.beta,
            "loss_type": dpo_config.loss_type,
            "loss_weights": dpo_config.loss_weights,
            "max_length": dpo_config.max_length,
            "truncation_mode": dpo_config.truncation_mode,
            "precompute_ref_log_probs": dpo_config.precompute_ref_log_probs,
        },
        "effective_batch": dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps,
        "tokenization_verified": True,
    }
    if review:
        summary["review"] = str(write_review(dpo_data.kept, review, summary))
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="YAML config (`base:` inheritance supported)")
    parser.add_argument("--dry-run", action="store_true", help="build and render the pairs, load no model")
    parser.add_argument("--review", help="dry run: write counts, lengths and rendered pairs to this markdown file")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint of the run")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_dpo_config(args.config, args.overrides)
    summary = dry_run(cfg, review=args.review) if args.dry_run else run_dpo(cfg, resume=args.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    # Under `python -m` this file is `__main__`, a second copy of the module (see train_sft); dispatch through the import.
    from honest_agents.sft.dpo import main as _main

    sys.exit(_main())
