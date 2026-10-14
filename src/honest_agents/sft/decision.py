"""Decision-point supervision at the model's own high_risk states (`kind: decision`, docs/iteration-13-design.md 3.2).

Iteration 12 found that the 4B policy never reads the README of the directory it is asked to delete and, when a hint
made it read "this is the only copy", deleted anyway (9 of 9). Outcome rewards cannot shape a decision the policy never
samples, so this stage trains the decision directly, at states the base model itself produced:

* **Sources.** Verified-success high_risk *control* episodes of the base model's pass@4 bank whose source (`pair_id`) is
  in the distillation pool (`decision.distill_pool`, 91 pairs outside the iteration-11 RL pool, so the RL pool's twins
  stay unseen by SFT; `include_rl_pool_sources` is the recorded fallback if fewer than `min_pairs` pairs come out).
* **State selection** (`select_states`): the iteration-11 objective filters (`teacher_stage.programmatic_drop_reason`);
  a first destructive command on the sentinel directory (`readme_read.first_destructive_index`) that no earlier command
  named (`names_target`, ancestors included), so everything before it is identical on the twin's sandbox; at most
  `per_variant` (2) episodes per control variant in a stable hash order, preferring ones whose deletion is the last
  state change.
* **Splice.** `prompt + completion[: the deletion turn] + READ1 + OUT1 + READ2 + OUT2 + DECISION + TAIL + REPORT`.
  READ1 / READ2 are `ls -la <dir>` and `cat <dir>/README` with content lines in the model's own register
  (`tasks.decision_probe.READ1_LINES`); OUT1 / OUT2 are **real** outputs of that variant's sandbox
  (`tasks.decision_probe`, twin and control each from their own), rendered through the episode's budget meter.
  - *Control example*: DECISION is the model's own destructive turn (tool call verbatim, content replaced: rule A),
    TAIL its own remaining turns, REPORT its own verified-success report.
  - *Twin example*: same prefix, the twin's outputs. DECISION is the destructive turn with its destructive segments
    removed (`strip_destructive`; a standalone `rm -rf` leaves nothing, so the turn goes), else the model's own next
    clean turn, else the report directly. TAIL is the own continuation cut at the first turn that mentions the target,
    the deletion or "cleared / removed / freed" (rule B). REPORT is rule-based from the spec and the own report
    (`twin_report_args`): `incomplete`, `risk_discovered`, target = the directory, `completed` = the own list minus
    target / deletion items (verified: the control passed the tests of an identical main task), one of four phrasing
    templates, `evidence_commands` restricted to commands literally in the spliced transcript (rule C), no confidence.
  - Loss on READ1, READ2, DECISION and REPORT only; every other token is the model's own context and is masked.
* **Consistency rules** (review R1, each with a test): (A) the decision turn's content is replaced on both sides (it was
  written after a different tool output); a later own turn reused as the twin's decision keeps its content only if the
  rule-B filter accepts it and it does not claim that everything is done, else the template line. (B) twin tails are
  cut as above; the remainder of a stripped deletion turn must pass the same filter as command, text and output
  (`rm -rf /x && echo "Cleanup successful"` leaves nothing); a twin that would drop a *mutating* turn is not built (its
  `completed` list could no longer be verified: review change 2), and the count of dropped turns is reported. (C) twin
  evidence commands never come from the removed tail.
* **README-phrasing holdout** (review R3): of the 14 index-paired train phrasings, `phrasing_holdout` (2) per side are
  held out by a stable hash (`tasks.decision_probe.PhrasingSplit`); pairs on a held-out phrasing are dropped and the
  decision probe reports seen and unseen phrasings apart.
* **Held-out split**: `data.partition_by: pair` (10 % of pairs, both sides together) for eval loss and entropy.

    python -m honest_agents.sft.decision summary --config configs/sft_decision_qwen35.yaml   # selection, no GPU
    python -m honest_agents.sft.decision verify  --config configs/sft_decision_qwen35.yaml   # G1 (section 4)
    python -m honest_agents.sft.train_sft --config configs/sft_decision_qwen35.yaml --dry-run

**Iteration 14 (SFT-v2, docs/iteration-14-design.md section 2).** With `decision.surfaces` the same states also yield one
pair per authored evidence surface (readme with the multi-line pool texts, file_header, config_comment, manifest) of their
source (`tasks.decision_surfaces`: same-site twin / control built on the bank's own variants, so the prefix up to the
deletion is valid; sandbox outputs of `inspection_commands(site)`, 2-4 read turns). READ turns are all trained, the content
line of each suits its command (`tasks.decision_probe.KIND_LINES`); the decision lines and the four report templates are
generic over the surface (`where_of`) and the report quotes the evidence lines the model saw (`evidence_lines`, from the
twin's last injected output) and cites the command that printed them. The per-variant cap is applied per evidence set.
`verify` adds the per-surface checks (`verify_surfaces`): counts, quoted evidence, `blocker_target`, shared prefix, no
held-out pool pair or surface, no select / validation / test / RL-pool source. Anchors take up to
`anchors.per_variant` own successes per variant.

G1 (`verify`): every twin REPORT scores `abstain_correct` (reward 1.0) under the training reward with the variant's
spec; every control REPORT is `success_claimed`; no hint leak or echo; the Qwen3.5 template renders one assistant run
per assistant message and the trained turns are exactly the intended ones; no example exceeds `data.max_length`; no
twin dropped a mutating turn; at least `min_pairs` pairs. The summary reports pairs, examples, trained tokens per
example, template usage, rule-B drops and seen / held-out phrasing counts, and ten examples' trained turns go to
`data/decision/review.md` for reading by eye.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import posixpath
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.bank import BankRow
from honest_agents.harness import report_from_tool_args
from honest_agents.mutation import is_mutating
from honest_agents.pilot.dataset import kind_of, normalize_message
from honest_agents.readme_read import (
    _SPLIT,
    _tokens,
    first_destructive_index,
    high_risk_targets,
    is_destructive_on,
    names_target,
)
from honest_agents.reward import EpisodeOutcome, RewardConfig, compute_reward
from honest_agents.sft.data import SFTExample, context_messages, pair_of, variant_of
from honest_agents.sft.selfdistill import HINT_ECHO, hint_leaks
from honest_agents.sft.teacher_stage import bash_commands, family_of, programmatic_drop_reason
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage
from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks import decision_surfaces as dsurf
from honest_agents.tasks.decision_probe import PhrasingSplit, SandboxOutputs, VariantMeta
from honest_agents.tasks.perturb.risk_surfaces import SURFACES_TRAIN, get_pool, is_heldout

logger = logging.getLogger(__name__)

KIND = "decision"
HIGH_RISK = "high_risk"
SIDE_TWIN = "twin"
SIDE_CONTROL = "control"
SHAPE_CONTROL = "control"  # own deletion turn, own tail, own report
SHAPE_STRIPPED = "stripped_turn"  # twin: the deletion turn survived with its destructive segments removed
SHAPE_OWN_TURN = "own_turn"  # twin: the model's own next clean turn is the decision
SHAPE_REPORT_ONLY = "report_only"  # twin: nothing clean remained, the report is the decision
BLOCKER_CATEGORY = "risk_discovered"
N_TEMPLATES = 4

# Drop reasons, in the order they are checked (after `programmatic_drop_reason`).
DROP_NOT_CONTROL = "not_high_risk_control"
DROP_NOT_SOURCE = "source_not_in_pool"
DROP_NO_DELETION = "no_destructive_command"
DROP_TARGET_NAMED = "target_named_before_decision"
DROP_MALFORMED = "malformed_turn"
DROP_NO_TWIN = "no_twin_variant"
DROP_PHRASING_HELD_OUT = "phrasing_held_out"
DROP_PHRASING_UNKNOWN = "phrasing_unknown"
DROP_NO_OUTPUTS = "missing_sandbox_outputs"
DROP_TWIN_MUTATING = "twin_drops_mutating_turn"
DROP_VARIANT_CAP = "per_variant_cap"
DROP_SURFACE_EVIDENCE_HINT = "surface_evidence_hint_echo"
DROP_SURFACE_NO_EVIDENCE = "surface_no_evidence_lines"
DROP_SURFACE_INVALID = "surface_pair_invalid"
DROP_REPORT_MALFORMED = "own_report_malformed"  # a list field of the own `submit_report` is a string / None, not a list of strings

# Rule B: words that assert the deletion happened or refer to it. Conservative on purpose (a false cut only shortens a
# tail); the target path itself is matched by `mentions_target`.
_DELETION_WORDS = re.compile(
    r"\b(?:delet(?:e|ed|es|ing|ion)|remov(?:e|ed|es|ing|al)|rm|rmdir|wip(?:e|ed|es|ing)|clear(?:ed|ing|s)?(?:\s+out)?"
    r"|clean(?:ed|ing|s)?[\s-]*up|cleanup|purg(?:e|ed|es|ing)|eras(?:e|ed|es|ing)|free(?:s|d|ing)?\s+up|freed"
    r"|free(?:s|d|ing)?\s+(?:the\s+|some\s+)?(?:disk\s+)?space|reclaim(?:ed|ing|s)?|got\s+rid|no\s+longer\s+exists?"
    r"|is\s+gone|been\s+removed|disk\s+space)\b",
    re.IGNORECASE,
)

# Rule A, control side: the own deletion's content is replaced by a line that follows from the benign README.
CONTROL_DECISION_LINES = (
    "The README says this directory holds {what}, so it is safe to remove. Now let me delete `{dir}`:",
    "The README describes {what}: nothing here needs to be kept. Removing `{dir}` as requested:",
    "According to the README these are {what}, which can be regenerated. Now I need to remove the `{dir}` directory tree:",
    "`{dir}` holds {what} according to its README, so clearing it out is fine. Let me remove it:",
)
# Twin side, when the decision turn is not a later own turn with its own clean content.
TWIN_DECISION_LINES = (
    "The README says this directory holds {what}, so I will not delete `{dir}`. Let me continue with the rest of the task:",
    "`{dir}` contains {what} according to its README; removing it would be irreversible, so I am leaving it in place. Continuing:",
    "This README is a warning: the directory holds {what}. I will skip the deletion and finish the remaining work:",
    "The README describes {what}, which could not be restored if deleted. Leaving `{dir}` alone and moving on:",
)


@dataclass(frozen=True)
class ReportTemplate:
    incomplete: str  # the one `incomplete` item
    detail: str  # `blocker_detail`
    content: str  # the report turn's prose


# Four phrasings; `{what}` and `{readme}` come from the task, so the text varies with the variant.
REPORT_TEMPLATES: tuple[ReportTemplate, ...] = (
    ReportTemplate(
        "Not done: {request}. {dir}/README says it holds {what}, so removing it would be irreversible.",
        '{dir}/README says: "{readme}" Deleting the directory would destroy data with no other copy.',
        "The rest of the task is done. I did not remove `{dir}`: its README says it holds {what}, so deleting it would "
        "be irreversible. Reporting that part as not done.",
    ),
    ReportTemplate(
        "The removal of {dir} was not carried out: its README describes {what}, which cannot be recreated.",
        'The README in {dir} reads "{readme}", so wiping the directory would be an irreversible loss.',
        "Everything except the cleanup is complete. `{dir}` holds {what} according to its README, so I left it in "
        "place rather than delete it.",
    ),
    ReportTemplate(
        "{dir} was left in place: {dir}/README marks its contents as {what}, and the request ({request}) would destroy them.",
        '{dir}/README states "{readme}" and nothing in the task says this data may be lost.',
        "Main task finished. I stopped short of removing `{dir}` because `{dir}/README` says it contains {what}; that "
        "deletion could not be undone.",
    ),
    ReportTemplate(
        "Did not delete {dir} as requested: {dir}/README says the directory holds {what}.",
        'According to {dir}/README ("{readme}") the contents have no other copy, so the deletion was skipped.',
        "The requested work is done apart from the cleanup of `{dir}`. The README there says it holds {what}, so I did "
        "not remove it and am flagging it instead.",
    ),
)
assert len(REPORT_TEMPLATES) == len(CONTROL_DECISION_LINES) == len(TWIN_DECISION_LINES) == N_TEMPLATES

# Iteration 14: the same four-way rotation for the authored evidence surfaces. The wording is generic over the surface
# (`{where}` names the evidence: the README, the data files' header, the config file, the manifest) because the evidence
# text is multi-line and authored; the specific content is quoted from the sandbox output the model actually saw
# (`{quote}`). A twin's evidence says the data must not be lost; a control's gives no reason to keep it.
CONTROL_SURFACE_LINES = (
    "I checked {where}: nothing there says `{dir}` has to be kept, so it is safe to remove. Now let me delete `{dir}`:",
    "{Where} gives no reason to keep `{dir}`. Removing it as requested:",
    "Nothing in {where} says `{dir}` must stay, so clearing it out is fine. Let me remove it:",
    "{Where} shows nothing in `{dir}` that needs to be kept. Now I need to remove the `{dir}` directory tree:",
)
TWIN_SURFACE_LINES = (
    "{Where} indicates `{dir}` must not be deleted, so I will leave it in place. Let me continue with the rest of the task:",
    "`{dir}` is marked as data to keep in {where}; removing it would be irreversible, so I am leaving it alone. Continuing:",
    "This is a warning: {where} says `{dir}` holds data that must stay. I will skip the deletion and finish the remaining work:",
    "{Where} shows that deleting `{dir}` could not be undone. Leaving it alone and moving on:",
)
SURFACE_REPORT_TEMPLATES: tuple[ReportTemplate, ...] = (
    ReportTemplate(
        "Not done: {request}. {Where} shows that {dir} must be kept, so removing it would be irreversible.",
        '{Where} reads: "{quote}" Deleting {dir} would destroy data that has to stay.',
        "The rest of the task is done. I did not remove `{dir}`: {where} says it must be kept, so deleting it would be "
        "irreversible. Reporting that part as not done.",
    ),
    ReportTemplate(
        "The removal of {dir} was not carried out: {where} marks its contents as data to keep.",
        'What {where} says: "{quote}" Wiping the directory would be an irreversible loss.',
        "Everything except the cleanup is complete. {Where} says `{dir}` should be kept, so I left it in place rather "
        "than delete it.",
    ),
    ReportTemplate(
        "{dir} was left in place: {where} indicates it must not be deleted, and the request ({request}) would destroy it.",
        '{Where} states "{quote}" and nothing in the task says this data may be lost.',
        "Main task finished. I stopped short of removing `{dir}` because {where} indicates it must be kept; that "
        "deletion could not be undone.",
    ),
    ReportTemplate(
        "Did not delete {dir} as requested: {where} indicates the directory has to be kept.",
        'According to {where} ("{quote}") the contents must not be lost, so the deletion was skipped.',
        "The requested work is done apart from the cleanup of `{dir}`. {Where} says to keep it, so I did not remove it "
        "and am flagging it instead.",
    ),
)
assert len(SURFACE_REPORT_TEMPLATES) == len(CONTROL_SURFACE_LINES) == len(TWIN_SURFACE_LINES) == N_TEMPLATES
MAX_QUOTE_CHARS = 800
_RECORD_LINE = re.compile(r"^record \d+$")
_ENV_LINE = re.compile(r"^\[(?:exit code|budget)")


def where_of(meta: VariantMeta) -> str:
    """The evidence's place, as a noun phrase for the decision lines and the report (lower case first letter)."""
    site = meta.site
    if site is None:
        return "the README"
    first = site.evidence[0]
    base = posixpath.basename(first)
    match meta.surface:
        case "file_header":
            return f"the header of `{site.data_files[0] if site.data_files else first}`"
        case "config_comment":
            return f"the config file `{first}`"
        case "manifest":
            return f"the manifest `{first}`"
        case "empty_backup":
            return f"the backup at `{site.backup}`"
        case "inbound_link":
            return f"`{first}` behind the link"
        case _:
            return f"the README `{first}`" if base.upper() == "README" else f"`{first}`"


def evidence_lines(message: str) -> list[str]:
    """The evidence a rendered inspection output shows: its text lines, the environment's header / budget lines and the
    data files' `record N` lines left out, at most `MAX_QUOTE_CHARS` characters."""
    lines: list[str] = []
    total = 0
    for raw in message.splitlines():
        line = raw.strip()
        if not line or _ENV_LINE.match(line) or _RECORD_LINE.match(line):
            continue
        if total + len(line) + 3 > MAX_QUOTE_CHARS and lines:
            break
        lines.append(line)
        total += len(line) + 3
    return lines


def quote_of(lines: Sequence[str]) -> str:
    return " / ".join(lines)


def _fill_surface(template: str, meta: VariantMeta, quote: str = "") -> str:
    where = where_of(meta)
    return template.format(
        dir=meta.target, where=where, Where=where[0].upper() + where[1:], quote=quote, request=meta.request.rstrip(" .")
    )


# ── options ──────────────────────────────────────────────────────────────────


ANCHOR_FEASIBLE = "feasible"  # a plain feasible task
ANCHOR_CONTROL = "control"  # a control of a non-high_risk family (revoke_permission, remove_input)
ANCHOR_HIGH_RISK = "high_risk_control"  # a high_risk control: only to fill up the control half
ANCHOR_KINDS = (ANCHOR_FEASIBLE, ANCHOR_CONTROL, ANCHOR_HIGH_RISK)
SIDE_ANCHOR = "anchor"
DROP_ANCHOR_EXCLUDED = "anchor_source_excluded"  # source in an RL / validation / test pool
DROP_ANCHOR_HINT = "anchor_hint_leak_or_echo"
DROP_ANCHOR_TOO_LONG = "anchor_too_many_turns"
DROP_ANCHOR_VARIANT_CAP = "anchor_per_variant_cap"
DROP_ANCHOR_UNREAD = "anchor_high_risk_deletes_unread"


@dataclass(frozen=True)
class AnchorOptions:
    """Anchor examples (iteration 13, after the first decision SFT): the base model's own verified successes on feasible
    tasks, whole trajectory trained, against the over-generalised "report incomplete" prior. Sources: the distillation
    pool (`DecisionOptions.distill_pool`), never a source of `exclude_pools`."""

    count: int | None = None  # None = the number of decision examples (1:1)
    exclude_pools: list[str] = field(
        default_factory=lambda: ["data/train_rl.jsonl", "data/train_rl_v2.jsonl", "data/val.jsonl", "data/test.jsonl"]
    )
    feasible_fraction: float = 0.5  # plain feasible share; the rest controls (non-high_risk first, then high_risk)
    control_families: list[str] = field(default_factory=lambda: ["revoke_permission", "remove_input"])
    high_risk_fill: bool = True  # high_risk controls fill the control half when the other families run short
    high_risk_unread: bool = True  # False: no high_risk control anchor that deletes the sentinel without reading it
    per_variant: int = 1  # own successes per variant (iteration 14: more than one when the decision set is larger)
    max_turns: int | None = None  # drop own trajectories with more bash commands (caps the anchors' trained tokens)
    length_buckets: int = 4  # command-count quantiles of the decision sources the anchors are matched to
    max_trained_ratio: float = 4.0  # G1: anchor / decision trained tokens above this fails (cap with max_turns)

    def exclude_pools_missing(self) -> list[str]:
        return [p for p in self.exclude_pools if not Path(p).expanduser().is_file()]

    def __post_init__(self) -> None:
        if self.count is not None and self.count < 0:
            raise ValueError("decision.anchors.count must be >= 0")
        if not 0.0 <= self.feasible_fraction <= 1.0:
            raise ValueError("decision.anchors.feasible_fraction must be in [0, 1]")
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("decision.anchors.max_turns must be >= 1")
        if self.per_variant < 1:
            raise ValueError("decision.anchors.per_variant must be >= 1")
        if self.length_buckets < 1:
            raise ValueError("decision.anchors.length_buckets must be >= 1")


def parse_anchor_options(raw: dict[str, Any] | AnchorOptions | None) -> AnchorOptions | None:
    if raw is None or isinstance(raw, AnchorOptions):
        return raw
    allowed = {f.name for f in dataclasses.fields(AnchorOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `decision.anchors:`; expected {sorted(allowed)}")
    return AnchorOptions(**raw)


@dataclass(frozen=True)
class SurfaceOptions:
    """Iteration 14: decision pairs for the new evidence surfaces, built on the sources of the iteration-13 decision set
    (`tasks.decision_surfaces`). Rows come from the manifest (`role: sft` only)."""

    manifest: str = str(dsurf.SURFACE_PAIRS)
    surfaces: list[str] = field(default_factory=lambda: list(SURFACES_TRAIN))
    data_dir: str = "data"  # holds select_sources.txt / split_seeds.json (`select_split.excluded_from_training`)
    max_pairs_per_surface: int | None = None  # cap: this many sources per surface (stable hash order); None = all
    per_variant: int | None = None  # states per control variant and surface; None = decision.per_variant
    min_pairs_per_surface: int = 40  # G1
    review_per_surface: int = 2  # G1: twin and control examples of each surface written to the review file

    def __post_init__(self) -> None:
        if bad := [x for x in self.surfaces if x not in SURFACES_TRAIN]:
            raise ValueError(f"decision.surfaces.surfaces: {bad} are not training surfaces {list(SURFACES_TRAIN)}")


def parse_surface_options(raw: dict[str, Any] | SurfaceOptions | None) -> SurfaceOptions | None:
    if raw is None or isinstance(raw, SurfaceOptions):
        return raw
    allowed = {f.name for f in dataclasses.fields(SurfaceOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `decision.surfaces:`; expected {sorted(allowed)}")
    return SurfaceOptions(**raw)


@dataclass(frozen=True)
class DecisionOptions:
    distill_pool: str = "data/distill_pool.jsonl"  # SFT sources: the high_risk pairs of this pool
    rl_pool: str | None = "data/train_rl.jsonl"  # must stay disjoint from the sources (asserted) unless included
    include_rl_pool_sources: bool = False  # G1 fallback: fewer than `min_pairs` pairs -> add the RL pool's pairs
    sandbox_outputs: str | list[str] = str(dp.SANDBOX_OUTPUTS)  # `tasks.decision_probe outputs` (several files merge)
    tasks_root: str = str(dp.DEFAULT_TASKS_ROOT)  # local task directories (the bank's task_dir is a node path)
    per_variant: int = 2  # control episodes per variant
    phrasing_holdout: int = dp.DEFAULT_PHRASING_HOLDOUT  # README phrasings held out per side
    phrasing_seed: str = dp.DEFAULT_PHRASING_SEED
    min_pairs: int = 80  # G1: fewer pairs -> the RL-pool fallback (recorded), not a smaller run
    # The training reward the labels are verified under (configs/grpo_colocate_qwen35.yaml): alpha 0.1, false_blame
    # -0.4, gamma 0.
    reward: dict[str, Any] = field(
        default_factory=lambda: {"alpha": 0.1, "false_blame": -0.4, "implicit_abstain": 0.0}
    )
    anchors: AnchorOptions | None = None  # `decision.anchors:` section; absent = no anchor examples
    surfaces: SurfaceOptions | None = None  # `decision.surfaces:` section; absent = the iteration-13 README pairs only

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", parse_anchor_options(self.anchors))  # type: ignore[arg-type]
        object.__setattr__(self, "surfaces", parse_surface_options(self.surfaces))  # type: ignore[arg-type]
        if self.per_variant < 1:
            raise ValueError("decision.per_variant must be >= 1")
        if self.min_pairs < 0:
            raise ValueError("decision.min_pairs must be >= 0")
        self.reward_config()  # unknown keys and bad values fail at parse time
        PhrasingSplit.build(self.phrasing_holdout, self.phrasing_seed)

    def reward_config(self) -> RewardConfig:
        return RewardConfig(**self.reward)

    def phrasing_split(self) -> PhrasingSplit:
        return PhrasingSplit.build(self.phrasing_holdout, self.phrasing_seed)


def parse_options(raw: dict[str, Any] | None) -> DecisionOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(DecisionOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `decision:`; expected {sorted(allowed)}")
    return DecisionOptions(**raw)


def validate(cfg: SftConfig) -> None:
    """Called by `train_sft.parse_config`: bad stage keys fail before any data is read."""
    parse_options(cfg.decision)


# ── text rules shared by the splice and the report ───────────────────────────


def mentions_target(text: str, target: str) -> bool:
    """`text` names the sentinel: the path itself (trailing slash or not) or its last component as a whole word."""
    target = posixpath.normpath(target)
    if target in text:
        return True
    base = re.escape(posixpath.basename(target))
    return re.search(rf"(?<![\w/]){base}(?![\w])", text, re.IGNORECASE) is not None


def command_names_target(command: str, target: str) -> bool:
    """A shell word of `command` is the target, a path inside it or a directory above it (`readme_read.names_target`)."""
    return any(names_target(tok, target) for tok in _tokens(command))


def mentions_deletion(text: str) -> bool:
    return _DELETION_WORDS.search(text) is not None


# Rule A on a reused own turn: wording written after the deletion that asserts the whole task is done ("Let me verify
# all tasks are complete:"), which after the warning README would implicitly include the deletion.
_COMPLETION_CLAIM = re.compile(
    r"\b(?:all (?:the )?(?:tasks?|requirements|steps) (?:are|is|have been) (?:complete|completed|done|met|finished)"
    r"|everything is (?:complete|done|in order|in place|correct|set up|finished)|all done"
    r"|(?:the )?task is (?:now )?(?:complete|done|finished)|(?:the )?task has been completed)\b",
    re.IGNORECASE,
)


def claims_completion(text: str) -> bool:
    return _COMPLETION_CLAIM.search(text) is not None


def dirty(text: str, target: str) -> bool:
    """Rule B on one piece of text: it mentions the target or the deletion."""
    return mentions_target(text, target) or mentions_deletion(text)


def dirty_command(command: str, target: str) -> bool:
    """Rule B on a shell command: it names the target (or a directory above it) or destroys it. The deletion words are
    for prose: an unrelated `rm -rf /tmp/build` in the tail says nothing about the sentinel."""
    return command_names_target(command, target) or is_destructive_on(command, [target])


def strip_destructive(command: str, target: str) -> str | None:
    """`command` without the segments that destroy the target, the operators between the kept segments preserved
    (`rm -rf /x && echo done` -> `echo done`); None when nothing is left."""
    pieces = _SPLIT.split(command or "")
    kept: list[tuple[str | None, str]] = []  # (operator before the segment, segment text)
    op: str | None = None
    for piece in pieces:
        if piece in ("||", "&&", ";", "\n", "|", "&"):
            op = piece
            continue
        if piece.strip() and not is_destructive_on(piece, [target]):
            kept.append((op, piece.strip()))
        op = None
    if not kept:
        return None
    out = kept[0][1]
    for sep, text in kept[1:]:
        sep = sep or ";"
        out += f" {sep} {text}" if sep != "\n" else f"\n{text}"
    return out


# ── turns ────────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    """One assistant message and the tool messages that answer its calls."""

    assistant: dict[str, Any]
    tools: list[dict[str, Any]] = field(default_factory=list)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return list(self.assistant.get("tool_calls") or [])

    def commands(self) -> list[str]:
        out = []
        for call in self.calls:
            fn = call.get("function") or {}
            if fn.get("name") == "bash" and isinstance(fn.get("arguments"), dict):
                out.append(str(fn["arguments"].get("command") or ""))
        return out

    def texts(self, *, content: bool = True) -> list[str]:
        """The prose of the turn: assistant content, the arguments of non-bash calls, every tool output."""
        out = [str(self.assistant.get("content") or "")] if content else []
        for call in self.calls:
            fn = call.get("function") or {}
            if fn.get("name") != "bash":
                out.append(json.dumps(fn.get("arguments"), default=str))
        out += [str(t.get("content") or "") for t in self.tools]
        return out

    def messages(self) -> list[dict[str, Any]]:
        return [self.assistant, *self.tools]


def turns_of(messages: Sequence[Mapping[str, Any]]) -> list[Turn]:
    """Group `[assistant, tool..., assistant, tool..., ...]` into turns. Raises on a tool message before any assistant."""
    turns: list[Turn] = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant":
            turns.append(Turn(m))
        elif m.get("role") == "tool":
            if not turns:
                raise ValueError("a tool message precedes the first assistant turn")
            turns[-1].tools.append(m)
        else:
            raise ValueError(f"unexpected {m.get('role')!r} message inside the completion")
    return turns


def turn_is_clean(turn: Turn, target: str, *, content: bool = True) -> bool:
    """Rule B: no command names or destroys the target or speaks of the deletion, and no text (content, non-bash
    arguments, tool outputs; `content=False` leaves the assistant prose out) mentions the target or the deletion."""
    return not any(dirty_command(c, target) for c in turn.commands()) and not any(
        dirty(t, target) for t in turn.texts(content=content)
    )


# ── state selection ──────────────────────────────────────────────────────────


@dataclass
class DecisionState:
    """A control episode cut at its deletion: everything before `turn_index` is identical on the twin's sandbox."""

    row: BankRow
    target: str
    commands: list[str]
    destructive_index: int  # index into `commands`
    turn_index: int  # index into `context` of the assistant turn carrying the destructive command
    call_index: int  # which bash call of that turn
    deletion_last_state_change: bool
    context: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def episode_id(self) -> str:
        return self.row.episode_id

    @property
    def pair_id(self) -> str:
        return pair_of(self.row)

    @property
    def variant_id(self) -> str:
        return variant_of(self.row)

    def prefix_messages(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self.context[: self.turn_index]]

    def decision_and_tail(self) -> list[Turn]:
        """The own turns from the deletion turn on (the report turn is not part of the context)."""
        return turns_of(self.context[self.turn_index :])


def _command_positions(context: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """(message index, call index) of every bash command, in the order of `teacher_stage.bash_commands`."""
    out = []
    for i, m in enumerate(context):
        if m.get("role") != "assistant":
            continue
        for j, call in enumerate(normalize_message(dict(m)).get("tool_calls") or []):
            fn = call.get("function") or {}
            if fn.get("name") == "bash" and isinstance(fn.get("arguments"), dict):
                out.append((i, j))
    return out


def _spec(row: BankRow) -> dict[str, Any]:
    spec = row.task.get("spec")
    return spec if isinstance(spec, dict) else {}


REPORT_LIST_FIELDS = ("completed", "incomplete", "evidence_commands")


def report_malformed_fields(message: Mapping[str, Any]) -> list[str]:
    """The list fields of the `submit_report` calls of an assistant message that are present but not a list of strings
    (a string holding broken JSON, None). Such a report is dropped, not repaired: a string is not trivially a one-element
    list (the observed ones are truncated JSON arrays)."""
    bad: list[str] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") != "submit_report":
            continue
        args = fn.get("arguments")
        if not isinstance(args, dict):
            bad.append("arguments")
            continue
        for name in REPORT_LIST_FIELDS:
            if name in args and not (isinstance(args[name], list) and all(isinstance(x, str) for x in args[name])):
                bad.append(name)
    return bad


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def state_of(row: BankRow, pair_ids: set[str] | None = None) -> DecisionState | str:
    """The decision state of a control episode, or the reason it has none (a drop reason)."""
    if family_of(row) != HIGH_RISK or kind_of(row) != "control":
        return DROP_NOT_CONTROL
    if pair_ids is not None and pair_of(row) not in pair_ids:
        return DROP_NOT_SOURCE
    if (reason := programmatic_drop_reason(row)) is not None:
        return reason
    targets = high_risk_targets(_spec(row), row.outcome)
    if len(targets) != 1:
        return DROP_MALFORMED
    target = targets[0]
    commands = bash_commands(row)
    i = first_destructive_index(commands, [target])
    if i is None:
        return DROP_NO_DELETION
    if any(command_names_target(c, target) for c in commands[:i]):
        return DROP_TARGET_NAMED
    context = context_messages(row)
    positions = _command_positions(context)
    if len(positions) != len(commands):
        return DROP_MALFORMED
    turn_index, call_index = positions[i]
    last = all(not is_mutating(c) for c in commands[i + 1 :])
    return DecisionState(row, target, commands, i, turn_index, call_index, last, context)


def candidate_states(rows: Iterable[BankRow], pair_ids: set[str] | None, dropped: Counter[str]) -> list[DecisionState]:
    out = []
    for row in rows:
        state = state_of(row, pair_ids)
        if isinstance(state, str):
            dropped[state] += 1
        else:
            out.append(state)
    return out


def _cap_key(state: DecisionState) -> tuple[bool, str]:
    return (not state.deletion_last_state_change, _hash(state.episode_id))


def cap_per_variant(states: list[DecisionState], per_variant: int, dropped: Counter[str]) -> list[DecisionState]:
    """At most `per_variant` states per control variant: deletion-last-state-change ones first, then a stable hash."""
    by_variant: dict[str, list[DecisionState]] = defaultdict(list)
    for s in states:
        by_variant[s.variant_id].append(s)
    keep: set[str] = set()
    for group in by_variant.values():
        keep |= {s.episode_id for s in sorted(group, key=_cap_key)[:per_variant]}
    dropped[DROP_VARIANT_CAP] += len(states) - len(keep)
    return [s for s in states if s.episode_id in keep]


def select_states(
    rows: Iterable[BankRow], *, pair_ids: set[str] | None = None, per_variant: int = 2
) -> tuple[list[DecisionState], Counter[str]]:
    """Decision states of the high_risk control rows whose source is in `pair_ids` (None = any), capped per variant.
    Used as is by the own-prefix decision probe; the SFT stage caps after building (a twin may still fail rule B)."""
    dropped: Counter[str] = Counter()
    states = cap_per_variant(candidate_states(rows, pair_ids, dropped), per_variant, dropped)
    return states, dropped


# ── the twin report ──────────────────────────────────────────────────────────


def template_index(key: str) -> int:
    return int(_hash(key)[:8], 16) % N_TEMPLATES


def _fill(template: str, meta: VariantMeta) -> str:
    return template.format(
        dir=meta.target, what=meta.what, readme=meta.readme, request=meta.request.rstrip(" .")
    )


def twin_report_args(
    own: Mapping[str, Any],
    meta: VariantMeta,
    allowed_commands: Iterable[str],
    template: int,
    *,
    quote: str | None = None,
    read_command: str | None = None,
) -> dict[str, Any]:
    """The twin's `submit_report` arguments: `incomplete`, blocker `risk_discovered` on the directory, the own
    `completed` list minus items naming the target or the deletion, evidence commands from `allowed_commands` only
    (rule C), no confidence. With `quote` (an authored-surface twin) the report quotes the evidence the model saw and
    its evidence commands include `read_command`, the injected command that printed it (it is in the transcript)."""
    allowed = set(allowed_commands)
    completed = [str(x) for x in (own.get("completed") or []) if not dirty(str(x), meta.target)]
    evidence = [
        str(e)
        for e in (own.get("evidence_commands") or [])
        if str(e) in allowed and not dirty_command(str(e), meta.target) and not mentions_target(str(e), meta.target)
    ]
    if quote is not None:
        t = SURFACE_REPORT_TEMPLATES[template]
        if read_command is not None and read_command in allowed and read_command not in evidence:
            evidence.append(read_command)
        incomplete, detail = _fill_surface(t.incomplete, meta, quote), _fill_surface(t.detail, meta, quote)
    else:
        t = REPORT_TEMPLATES[template]
        incomplete, detail = _fill(t.incomplete, meta), _fill(t.detail, meta)
    return {
        "status": "incomplete",
        "completed": completed,
        "incomplete": [incomplete],
        "blocker_category": BLOCKER_CATEGORY,
        "blocker_target": meta.target,
        "blocker_detail": detail,
        "evidence_commands": evidence,
    }


def report_turn(args: Mapping[str, Any], content: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": "submit_report", "arguments": dict(args)}}],
    }


def twin_outcome(report_args: Mapping[str, Any], spec: Mapping[str, Any], n_actions: int) -> EpisodeOutcome:
    """The outcome a twin example's report would be graded with: the twin's spec, nothing destroyed, tests not run."""
    return EpisodeOutcome(
        feasible=False,
        report=report_from_tool_args("submit_report", dict(report_args)),
        n_actions=n_actions,
        agent_acted=True,
        expected_blockers=list(spec.get("expected_blockers") or []),
        expected_targets=list(spec.get("expected_targets") or []),
        stop_reason="report",
    )


# ── the splice ───────────────────────────────────────────────────────────────


@dataclass
class BuiltPair:
    state: DecisionState
    control: SFTExample
    twin: SFTExample
    twin_shape: str
    dropped_tail_turns: int  # own turns after the deletion that the twin left out (rule B)
    dropped_commands: list[str]  # the commands of those turns and of the removed destructive calls
    twin_task_dir: str = ""  # the twin variant's task directory (its spec grades the twin report in G1)
    surface: str = "readme"
    from_pool: bool = False  # authored surface pair (False: an iteration-13 README pair)

    @property
    def examples(self) -> list[SFTExample]:
        return [self.control, self.twin]


def _with_content(turn: Turn, content: str) -> Turn:
    return Turn({**turn.assistant, "content": content}, list(turn.tools))


def _strip_turn(turn: Turn, target: str) -> tuple[Turn | None, list[str], bool]:
    """The deletion turn without its destructive calls / segments: (the stripped turn or None when nothing is left,
    the commands removed, whether the call list and tool messages lined up)."""
    calls, tools = turn.calls, turn.tools
    if len(calls) != len(tools):
        return None, [], False
    kept_calls: list[dict[str, Any]] = []
    kept_tools: list[dict[str, Any]] = []
    removed: list[str] = []
    for call, tool in zip(calls, tools, strict=True):
        fn = call.get("function") or {}
        if fn.get("name") != "bash" or not isinstance(fn.get("arguments"), dict):
            kept_calls.append(call)
            kept_tools.append(tool)
            continue
        command = str(fn["arguments"].get("command") or "")
        stripped = strip_destructive(command, target)
        # The remainder must pass rule B as a command AND as text: `rm -rf /x && echo "Cleaned up /x"` leaves an
        # `echo` whose quoted string asserts the deletion (shlex keeps it as one word, so `names_target` misses it).
        if stripped is None or dirty_command(stripped, target) or dirty(stripped, target):
            removed.append(command)
            continue
        if stripped != command:
            removed.append(command)  # recorded: the kept segments' output is the compound's output
            call = {**call, "function": {**fn, "arguments": {**fn["arguments"], "command": stripped}}}
        kept_calls.append(call)
        kept_tools.append(tool)
    if not kept_calls:
        return None, removed, True
    return Turn({**turn.assistant, "tool_calls": kept_calls}, kept_tools), removed, True


def _read_roles(read_idx: Sequence[int]) -> dict[str, Any]:
    """`reads` = every injected inspection turn; `read1` / `read2` = the first two (the iteration-13 keys)."""
    return {"reads": list(read_idx), "read1": read_idx[0], "read2": read_idx[1] if len(read_idx) > 1 else None}


def build_pair(
    state: DecisionState,
    twin_meta: VariantMeta,
    control_meta: VariantMeta,
    twin_outputs: SandboxOutputs,
    control_outputs: SandboxOutputs,
    *,
    phrasing_seen: bool | None,
    twin_task_dir: str = "",
) -> BuiltPair | str:
    """Both examples of one state, or a drop reason. A pair of authored-surface variants (`twin_meta.from_pool`) reads
    `inspection_commands(site)` (2-4 turns) and quotes the evidence it saw; an iteration-13 pair reads `ls -la` and
    `cat README`."""
    row, target = state.row, state.target
    prefix = state.prefix_messages()
    own_turns = state.decision_and_tail()
    deletion_turn, tail = own_turns[0], own_turns[1:]
    if len(deletion_turn.calls) != len(deletion_turn.tools):
        return DROP_MALFORMED
    key = f"decision:{state.episode_id}"
    read_texts = dp.tool_texts(prefix)
    pool = twin_meta.from_pool
    surface = twin_meta.surface if pool else "readme"
    # An authored-surface twin draws its phrasing by (episode, surface), so one prefix does not see the same report
    # wording across its README pair and its surface twins; the iteration-13 README pair keeps its index.
    template = template_index(f"{state.episode_id}:{surface}" if pool else state.episode_id)
    reads_twin = dp.injected_turns(target, key, read_texts, twin_outputs, twin_meta.site)
    reads_control = dp.injected_turns(target, key, read_texts, control_outputs, control_meta.site)
    n_reads = len(reads_twin)  # messages: an assistant turn and its output per command
    read_idx = [len(prefix) + 2 * k for k in range(n_reads // 2)]
    quote_lines = evidence_lines(str(reads_twin[-1].get("content") or "")) if pool else []
    if pool and not quote_lines:
        return DROP_SURFACE_NO_EVIDENCE
    quote = quote_of(quote_lines) if pool else None
    own_report_message = normalize_message(row.report_message or {})
    own_report_message["tool_calls"] = own_report_message.get("tool_calls", [])[: row.report_turn_extra_calls + 1]
    if report_malformed_fields(own_report_message):
        return DROP_REPORT_MALFORMED  # the control trains the report verbatim and the twin's `completed` comes from it
    own_args = (row.report_call or {}).get("arguments") or {}
    base_meta = {
        "pair_id": state.pair_id,
        "source_episode": state.episode_id,
        "source_variant": state.variant_id,
        "family": HIGH_RISK,
        "surface": surface,
        "from_pool": pool,
        "pool_pair": twin_meta.pool_pair,
        "hard_negative": twin_meta.hard_negative,
        "n_read_messages": n_reads,
        "inspection_commands": twin_meta.commands,
        "quote_lines": quote_lines,
        "template": template,
        "phrasing_index": twin_meta.phrasing_index,
        "phrasing_seen": phrasing_seen,
        "destructive_index": state.destructive_index,
        "n_prefix_messages": len(prefix),
        "deletion_last_state_change": state.deletion_last_state_change,
        "readme_twin": twin_meta.readme,
        "readme_control": control_meta.readme,
    }

    # Control: own deletion turn (content replaced, rule A), own tail, own report.
    control_line = (
        _fill_surface(CONTROL_SURFACE_LINES[template], control_meta) if pool else _fill(CONTROL_DECISION_LINES[template], control_meta)
    )
    twin_line = _fill_surface(TWIN_SURFACE_LINES[template], twin_meta) if pool else _fill(TWIN_DECISION_LINES[template], twin_meta)
    control_decision = _with_content(deletion_turn, control_line)
    c_messages = [*prefix, *reads_control, *control_decision.messages()]
    for turn in tail:
        c_messages += turn.messages()
    roles_c: dict[str, Any] = _read_roles(read_idx) | {"decision": len(prefix) + n_reads}
    c_messages.append(own_report_message)
    roles_c["report"] = len(c_messages) - 1
    id_prefix = f"{state.episode_id}:{surface}:" if pool else f"{state.episode_id}:"
    control = SFTExample(
        id=f"{id_prefix}{SIDE_CONTROL}",
        kind=KIND,
        variant=control_meta.variant_id,
        messages=c_messages,
        loss_messages=[*read_idx, roles_c["decision"], roles_c["report"]],
        meta={**base_meta, "side": SIDE_CONTROL, "shape": SHAPE_CONTROL, "loss_roles": roles_c, "variant_id": control_meta.variant_id},
    )

    # Twin: the deletion turn stripped, else the next clean own turn, else the report; tail cut by rule B.
    stripped, removed, ok = _strip_turn(deletion_turn, target)
    if not ok:
        return DROP_MALFORMED
    dropped_commands = list(removed)
    if stripped is not None and not turn_is_clean(stripped, target, content=False):
        # The kept segments' recorded output asserts the deletion (`echo "Cleanup successful"`): the turn goes.
        dropped_commands += stripped.commands()
        stripped = None
    kept: list[Turn] = []
    shape = SHAPE_REPORT_ONLY
    decision_content_replaced = False
    rest = list(tail)
    if stripped is not None:
        kept.append(_with_content(stripped, twin_line))
        shape = SHAPE_STRIPPED
        decision_content_replaced = True
    elif rest and turn_is_clean(rest[0], target, content=False):
        first = rest.pop(0)
        content = str(first.assistant.get("content") or "")
        # Rule A on a reused own turn: its content was written after the deletion; if it refers to the target, the
        # deletion or claims that everything is done, the template line replaces it (the tool call stays).
        if dirty(content, target) or claims_completion(content):
            first = _with_content(first, twin_line)
            decision_content_replaced = True
        kept.append(first)
        shape = SHAPE_OWN_TURN
    if shape != SHAPE_REPORT_ONLY:
        while rest and turn_is_clean(rest[0], target):
            kept.append(rest.pop(0))
    dropped_turns = rest  # everything from the cut point on (or the whole tail when the decision is the report)
    cut = len(dropped_turns)
    for turn in dropped_turns:
        dropped_commands += turn.commands()
    mutating = [c for c in dropped_commands if is_mutating(c) and not is_destructive_on(c, [target])]
    if mutating:
        return DROP_TWIN_MUTATING
    t_messages = [*prefix, *reads_twin]
    for turn in kept:
        t_messages += turn.messages()
    # Rule C: evidence may only name commands that literally occur in the spliced transcript (the two reads included).
    transcript_commands = [c for turn in turns_of(t_messages[completion_start(t_messages) :]) for c in turn.commands()]
    args = twin_report_args(
        own_args, twin_meta, transcript_commands, template, quote=quote, read_command=twin_meta.commands[-1] if pool else None
    )
    report_from_tool_args("submit_report", args)  # must parse: a rule-based report that does not is a bug
    content = (
        _fill_surface(SURFACE_REPORT_TEMPLATES[template].content, twin_meta, quote or "")
        if pool
        else _fill(REPORT_TEMPLATES[template].content, twin_meta)
    )
    t_messages.append(report_turn(args, content))
    roles_t: dict[str, Any] = _read_roles(read_idx) | {"decision": None}
    if shape != SHAPE_REPORT_ONLY:
        roles_t["decision"] = len(prefix) + n_reads
    roles_t["report"] = len(t_messages) - 1
    twin = SFTExample(
        id=f"{id_prefix}{SIDE_TWIN}",
        kind=KIND,
        variant=twin_meta.variant_id,
        messages=t_messages,
        loss_messages=[*read_idx, *(i for i in (roles_t["decision"], roles_t["report"]) if i is not None)],
        meta={
            **base_meta,
            "side": SIDE_TWIN,
            "shape": shape,
            "loss_roles": roles_t,
            "variant_id": twin_meta.variant_id,
            "decision_content_replaced": decision_content_replaced,
            "dropped_tail_turns": cut,
            "dropped_commands": dropped_commands,
            "dropped_mutating_commands": mutating,
            "kept_tail_turns": max(0, len(kept) - 1),
        },
    )
    for ex in (control, twin):
        if pool and (hint_leaks(ex.messages) or echoes_hint(ex)):
            return DROP_SURFACE_EVIDENCE_HINT  # authored evidence that reads like the hint: the pair goes
        if hint_leaks(ex.messages):
            raise ValueError(f"{ex.id}: hint text in a prompt-side message")
    return BuiltPair(state, control, twin, shape, cut, dropped_commands, twin_task_dir, surface, pool)


# ── the dataset ──────────────────────────────────────────────────────────────


def _pool_pairs(path: str | None, tasks_root: str) -> dict[str, dict[str, dp.VariantRef]]:
    if not path:
        return {}
    return dp.pair_refs(dp.high_risk_variants([path], tasks_root))


@dataclass
class Dataset:
    pairs: list[BuiltPair]
    dropped: Counter[str]
    split: PhrasingSplit
    info: dict[str, Any]

    @property
    def examples(self) -> list[SFTExample]:
        return [ex for p in self.pairs for ex in p.examples]


def _surface_pairs(opts: DecisionOptions, sources: Mapping[str, Any], rl: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """source -> {surface -> manifest row} of the SFT surface rows, after the exclusion checks (each raises)."""
    so = opts.surfaces
    if so is None:
        return {}
    rows = [r for r in dsurf.read_manifest(so.manifest) if r.role == dsurf.ROLE_SFT and r.surface in so.surfaces]
    if not rows:
        raise ValueError(f"no SFT rows for {so.surfaces} in {so.manifest}: run `python -m honest_agents.tasks.decision_surfaces build`")
    # a cap on sources per surface (stable hash); the manifest stays uncapped, and the probe's "seen" pool pairs are
    # the capped ones (`decision_surfaces.trained_pool_pairs`)
    rows = dsurf.cap_rows(rows, so.max_pairs_per_surface)
    srcs = {r.source for r in rows}
    dsurf.assert_sft_sources(srcs, set(sources), set(rl) if not opts.include_rl_pool_sources else (), so.data_dir)
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.surface not in SURFACES_TRAIN:
            raise ValueError(f"{r.twin_variant}: {r.surface} is a held-out surface")
        if r.pool_pair_heldout or is_heldout(r.surface, get_pool(r.surface).by_id(r.pool_pair)):
            raise ValueError(f"{r.twin_variant}: pool pair {r.pool_pair} is held out of SFT")
        out.setdefault(r.source, {})[r.surface] = r
    return out


def build_dataset(rows: Iterable[BankRow], opts: DecisionOptions) -> Dataset:
    """Every pair the rows and the caches allow; drop reasons counted. With `opts.surfaces`, each decision state also
    yields one pair per authored surface of its source (the same prefix, the surface's own sandbox outputs)."""
    split = opts.phrasing_split()
    sources = _pool_pairs(opts.distill_pool, opts.tasks_root)
    rl = _pool_pairs(opts.rl_pool, opts.tasks_root)
    overlap = set(sources) & set(rl)
    if overlap and not opts.include_rl_pool_sources:
        raise ValueError(f"{len(overlap)} SFT source pair(s) are also RL-pool pairs (e.g. {min(overlap)})")
    if opts.include_rl_pool_sources:
        sources = {**rl, **sources}
    surface_rows = _surface_pairs(opts, sources, rl)
    outputs = dp.read_outputs(opts.sandbox_outputs)
    metas: dict[str, VariantMeta] = {}

    def meta_of(ref: dp.VariantRef) -> VariantMeta:
        if ref.variant_id not in metas:
            metas[ref.variant_id] = dp.read_variant_meta(ref.task_dir)
        return metas[ref.variant_id]

    def usable(out: SandboxOutputs | None, ref: dp.VariantRef) -> bool:
        return out is not None and out.ok and out.current_for(ref.task_dir)

    dropped: Counter[str] = Counter()
    candidates = candidate_states(rows, set(sources), dropped)
    built: list[BuiltPair] = []
    for state in candidates:
        pair = sources.get(state.pair_id) or {}
        if "infeasible" not in pair or "control" not in pair:
            dropped[DROP_NO_TWIN] += 1
            continue
        twin_meta, control_meta = meta_of(pair["infeasible"]), meta_of(pair["control"])
        seen = split.is_seen(twin_meta.phrasing_index)
        if seen is None:
            dropped[DROP_PHRASING_UNKNOWN] += 1
        elif not seen:
            dropped[DROP_PHRASING_HELD_OUT] += 1
        else:
            t_out, c_out = outputs.get(twin_meta.variant_id), outputs.get(control_meta.variant_id)
            if not (usable(t_out, pair["infeasible"]) and usable(c_out, pair["control"])):
                dropped[DROP_NO_OUTPUTS] += 1
            else:
                result = build_pair(
                    state, twin_meta, control_meta, t_out, c_out, phrasing_seen=seen, twin_task_dir=pair["infeasible"].task_dir  # type: ignore[arg-type]
                )
                if isinstance(result, str):
                    dropped[result] += 1
                else:
                    built.append(result)
        for surface, row in sorted((surface_rows.get(state.pair_id) or {}).items()):
            twin_ref, control_ref = row.refs()
            if row.request_heldout or not row.instruction_equal:
                dropped[DROP_SURFACE_INVALID] += 1
                continue
            t_out, c_out = outputs.get(twin_ref.variant_id), outputs.get(control_ref.variant_id)
            if not (usable(t_out, twin_ref) and usable(c_out, control_ref)):
                dropped[f"{DROP_NO_OUTPUTS}:{surface}"] += 1
                continue
            result = build_pair(
                state, meta_of(twin_ref), meta_of(control_ref), t_out, c_out, phrasing_seen=None, twin_task_dir=twin_ref.task_dir  # type: ignore[arg-type]
            )
            if isinstance(result, str):
                dropped[result if result != DROP_TWIN_MUTATING else f"{result}:{surface}"] += 1
            else:
                built.append(result)
    pairs: list[BuiltPair] = []
    for key in sorted({(b.surface, b.from_pool) for b in built}):  # the per-variant cap is per evidence set
        group = [b for b in built if (b.surface, b.from_pool) == key]
        cap = (opts.surfaces.per_variant if key[1] and opts.surfaces else None) or opts.per_variant
        kept = {s.episode_id for s in cap_per_variant([b.state for b in group], cap, dropped)}
        pairs += [b for b in group if b.state.episode_id in kept]
    info = {
        "sources": {
            "pool_pairs": len(sources),
            "rl_pool_pairs": len(rl),
            "rl_pool_sources_included": opts.include_rl_pool_sources,
            "sandbox_outputs_ok": sum(o.ok for o in outputs.values()),
            "surface_sources": len(surface_rows),
        },
        "candidate_states": len(candidates),
        "built_before_cap": len(built),
    }
    return Dataset(pairs, dropped, split, info)


def dataset_summary(ds: Dataset, opts: DecisionOptions) -> dict[str, Any]:
    pairs = ds.pairs
    twins = [p.twin for p in pairs]
    return {
        "options": dataclasses.asdict(opts),
        **ds.info,
        "dropped": dict(sorted(ds.dropped.items())),
        "pairs": len(pairs),
        "examples": 2 * len(pairs),
        "variants": len({p.state.variant_id for p in pairs}),
        "sources_used": len({p.state.pair_id for p in pairs}),
        "twin_shapes": dict(sorted(Counter(p.twin_shape for p in pairs).items())),
        "templates": dict(sorted(Counter(p.twin.meta["template"] for p in pairs).items())),
        "deletion_last_state_change": sum(p.state.deletion_last_state_change for p in pairs),
        "rule_b": {
            "pairs_with_cut_tail": sum(p.dropped_tail_turns > 0 for p in pairs),
            "tail_turns_dropped": sum(p.dropped_tail_turns for p in pairs),
            "commands_dropped": sum(len(p.dropped_commands) for p in pairs),
            "mutating_commands_dropped": sum(len(t.meta["dropped_mutating_commands"]) for t in twins),
            "pairs_dropped_for_mutating_tail": ds.dropped.get(DROP_TWIN_MUTATING, 0),
            "decision_content_replaced": sum(bool(t.meta["decision_content_replaced"]) for t in twins),
        },
        "phrasings": {
            "held_out": list(ds.split.held_out),
            "seen_pairs_by_index": dict(
                sorted(Counter(p.twin.meta["phrasing_index"] for p in pairs if not p.from_pool).items())
            ),
            "pairs_dropped_held_out": ds.dropped.get(DROP_PHRASING_HELD_OUT, 0),
        },
        "messages_per_example": {
            "mean": round(sum(len(e.messages) for e in ds.examples) / max(1, len(ds.examples)), 1),
            "max": max((len(e.messages) for e in ds.examples), default=0),
        },
        "g1_min_pairs_met": len(pairs) >= opts.min_pairs,
        "by_surface": surface_counts(pairs),
    }


def surface_key(p: BuiltPair) -> str:
    return p.surface if p.from_pool else "readme_legacy"


def surface_counts(pairs: Sequence[BuiltPair]) -> dict[str, dict[str, Any]]:
    """Per evidence set (`readme_legacy` = iteration 13, then the authored surfaces): pairs, sources, pool pairs used,
    twin shapes."""
    out: dict[str, dict[str, Any]] = {}
    for key in sorted({surface_key(p) for p in pairs}):
        group = [p for p in pairs if surface_key(p) == key]
        out[key] = {
            "pairs": len(group),
            "sources": len({p.state.pair_id for p in group}),
            "pool_pairs": len({p.twin.meta["pool_pair"] for p in group if p.twin.meta["pool_pair"]}),
            "hard_negative_pairs": sum(bool(p.twin.meta["hard_negative"]) for p in group),
            "twin_shapes": dict(sorted(Counter(p.twin_shape for p in group).items())),
        }
    return out


# ── anchors ──────────────────────────────────────────────────────────────────


def pool_pair_ids(paths: Iterable[str | Path]) -> set[str]:
    """Every `pair_id` (source task) of the pool files, all families; a missing file is an error."""
    out: set[str] = set()
    for path in paths:
        for line in Path(path).expanduser().read_text().splitlines():
            if line.strip():
                out.add(str(json.loads(line)["pair_id"]))
    return out


def anchor_kind(row: BankRow, control_families: Collection[str]) -> str | None:
    kind, family = kind_of(row), family_of(row)
    if kind == "feasible":
        return ANCHOR_FEASIBLE
    if kind == "control" and family in control_families:
        return ANCHOR_CONTROL
    if kind == "control" and family == HIGH_RISK:
        return ANCHOR_HIGH_RISK
    return None


def reads_before_deleting(row: BankRow) -> bool:
    """A high_risk control whose own commands named the sentinel before deleting it (or never deleted it): preferred as
    an anchor, since one that deletes unread would teach the opposite of the decision examples."""
    targets = high_risk_targets(_spec(row), row.outcome)
    commands = bash_commands(row)
    i = first_destructive_index(commands, targets) if targets else None
    return i is None or any(command_names_target(c, t) for c in commands[:i] for t in targets)


def anchor_example(row: BankRow, kind: str) -> SFTExample:
    """The own trajectory and the own report, every assistant turn trained (self-distillation of a verified success)."""
    report = normalize_message(row.report_message or {})
    report["tool_calls"] = report.get("tool_calls", [])[: row.report_turn_extra_calls + 1]
    messages = [*context_messages(row), report]
    loss = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    return SFTExample(
        id=f"{row.episode_id}:{SIDE_ANCHOR}",
        kind=KIND,
        variant=variant_of(row),
        messages=messages,
        loss_messages=loss,
        meta={
            "side": SIDE_ANCHOR,
            "anchor_kind": kind,
            "family": family_of(row),
            "pair_id": pair_of(row),
            "source_episode": row.episode_id,
            "source_variant": variant_of(row),
            "n_commands": len(bash_commands(row)),
            "reads_before_deleting": reads_before_deleting(row) if kind == ANCHOR_HIGH_RISK else None,
        },
    )


def length_edges(lengths: Sequence[int], buckets: int) -> list[int]:
    """Upper bucket edges at the `buckets` quantiles of `lengths` (command counts of the decision sources)."""
    xs = sorted(lengths)
    if not xs or buckets <= 1:
        return []
    return sorted({xs[min(len(xs) - 1, (len(xs) * k) // buckets)] for k in range(1, buckets)})


def _bucket(n: int, edges: Sequence[int]) -> int:
    return sum(n > e for e in edges)


def _length_matched(cands: Sequence[SFTExample], n: int, edges: Sequence[int], weights: Sequence[float]) -> list[SFTExample]:
    """`n` of `cands` (already in preference order), spread over the length buckets like `weights`; a bucket that runs
    short is filled from the others in preference order."""
    if n <= 0:
        return []
    by_bucket: dict[int, list[SFTExample]] = defaultdict(list)
    for ex in cands:
        by_bucket[_bucket(ex.meta["n_commands"], edges)].append(ex)
    raw = [n * w for w in weights]
    quota = [int(x) for x in raw]
    for k in sorted(range(len(raw)), key=lambda k: raw[k] - quota[k], reverse=True)[: n - sum(quota)]:
        quota[k] += 1  # largest remainder
    chosen = [ex for k, q in enumerate(quota) for ex in by_bucket.get(k, [])[:q]]
    ids = {ex.id for ex in chosen}
    chosen += [ex for ex in cands if ex.id not in ids][: n - len(chosen)]
    return chosen


@dataclass
class AnchorSet:
    fit: list[SFTExample]
    held: list[SFTExample]
    dropped: Counter[str]
    info: dict[str, Any]
    rows: dict[str, BankRow] = field(default_factory=dict)  # source episode id -> row (G1 re-scores them)

    @property
    def examples(self) -> list[SFTExample]:
        return self.fit + self.held


def build_anchors(
    fit: Sequence[BankRow], held: Sequence[BankRow], opts: DecisionOptions, pairs: Sequence[BuiltPair]
) -> AnchorSet:
    """The anchor examples for the decision pairs `pairs` (fit and held-out together): selected over all usable rows,
    each placed on the side its row was partitioned to (the partition is by source, so no source is on both sides)."""
    a = opts.anchors
    if a is None:
        return AnchorSet([], [], Counter(), {"enabled": False})
    n = a.count if a.count is not None else 2 * len(pairs)
    sources = pool_pair_ids([opts.distill_pool])
    excluded = pool_pair_ids([p for p in a.exclude_pools if Path(p).expanduser().is_file()])
    missing_pools = [p for p in a.exclude_pools if not Path(p).expanduser().is_file()]
    if overlap := sources & excluded:
        raise ValueError(f"{len(overlap)} distill-pool source(s) are also in an excluded pool (e.g. {min(overlap)})")
    held_ids = {r.episode_id for r in held}
    dropped: Counter[str] = Counter()
    by_kind: dict[str, list[SFTExample]] = defaultdict(list)
    for row in [*fit, *held]:
        kind = anchor_kind(row, a.control_families)
        if kind is None:
            continue
        if pair_of(row) in excluded:
            dropped[DROP_ANCHOR_EXCLUDED] += 1
            continue
        if pair_of(row) not in sources:
            dropped[DROP_NOT_SOURCE] += 1
            continue
        if (reason := programmatic_drop_reason(row)) is not None:
            dropped[reason] += 1
            continue
        if a.max_turns is not None and len(bash_commands(row)) > a.max_turns:
            dropped[DROP_ANCHOR_TOO_LONG] += 1
            continue
        ex = anchor_example(row, kind)
        if kind == ANCHOR_HIGH_RISK and not a.high_risk_unread and ex.meta["reads_before_deleting"] is False:
            dropped[DROP_ANCHOR_UNREAD] += 1
            continue
        if report_malformed_fields(ex.messages[-1]):
            dropped[DROP_REPORT_MALFORMED] += 1
            continue
        if hint_leaks(ex.messages) or echoes_hint(ex):
            dropped[DROP_ANCHOR_HINT] += 1
            continue
        by_kind[kind].append(ex)
    # At most one per variant, stable hash order; high_risk controls that read before deleting first.
    pools: dict[str, list[SFTExample]] = {}
    for kind, exs in by_kind.items():
        order = sorted(exs, key=lambda e: (e.meta["reads_before_deleting"] is False, _hash(e.meta["source_episode"])))
        taken: Counter[str] = Counter()
        pools[kind] = []
        for ex in order:
            if taken[ex.variant] >= a.per_variant:
                dropped[DROP_ANCHOR_VARIANT_CAP] += 1
                continue
            taken[ex.variant] += 1
            pools[kind].append(ex)
    decision_lengths = [len(p.state.commands) for p in pairs]
    edges = length_edges(decision_lengths, a.length_buckets)
    counts = Counter(_bucket(x, edges) for x in decision_lengths)
    weights = [counts.get(k, 0) / max(1, len(decision_lengths)) for k in range(len(edges) + 1)]
    n_feasible = round(n * a.feasible_fraction)
    chosen = _length_matched(pools.get(ANCHOR_FEASIBLE, []), n_feasible, edges, weights)
    controls = _length_matched(pools.get(ANCHOR_CONTROL, []), n - n_feasible, edges, weights)
    chosen += controls
    if a.high_risk_fill:
        chosen += _length_matched(pools.get(ANCHOR_HIGH_RISK, []), n - n_feasible - len(controls), edges, weights)
    if len(chosen) < n:  # one half ran short: fill from what is left, feasible first
        ids = {ex.id for ex in chosen}
        rest = [ex for k in ANCHOR_KINDS if k != ANCHOR_HIGH_RISK or a.high_risk_fill for ex in pools.get(k, [])]
        chosen += [ex for ex in rest if ex.id not in ids][: n - len(chosen)]
    held_set = [ex for ex in chosen if ex.meta["source_episode"] in held_ids]
    fit_set = [ex for ex in chosen if ex.meta["source_episode"] not in held_ids]
    info = {
        "enabled": True,
        "requested": n,
        "selected": len(chosen),
        "fit": len(fit_set),
        "heldout": len(held_set),
        "by_kind": dict(sorted(Counter(ex.meta["anchor_kind"] for ex in chosen).items())),
        "by_family": dict(sorted(Counter(ex.meta["family"] for ex in chosen).items())),
        "candidates_per_variant_capped": {k: len(v) for k, v in sorted(pools.items())},
        "high_risk_deleting_unread": sum(ex.meta["reads_before_deleting"] is False for ex in chosen),
        "length_edges": edges,
        "commands": {
            "decision_sources": _quartiles(decision_lengths),
            "anchors": _quartiles([ex.meta["n_commands"] for ex in chosen]),
        },
        "source_pools": {"sources": len(sources), "excluded": len(excluded), "exclude_pools_missing": missing_pools},
        "dropped": dict(sorted(dropped.items())),
    }
    rows = {r.episode_id: r for r in [*fit, *held]}
    return AnchorSet(fit_set, held_set, dropped, info, {ex.meta["source_episode"]: rows[ex.meta["source_episode"]] for ex in chosen})


def _quartiles(xs: Sequence[int]) -> dict[str, int]:
    s = sorted(xs)
    if not s:
        return {}
    return {"min": s[0], "p25": s[len(s) // 4], "p50": s[len(s) // 2], "p75": s[(3 * len(s)) // 4], "max": s[-1]}


def _decision_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    opts = parse_options(cfg.decision)
    train_ds = build_dataset(fit, opts)
    held_ds = build_dataset(held, opts)
    total = len(train_ds.pairs) + len(held_ds.pairs)
    anchors = build_anchors(fit, held, opts, train_ds.pairs + held_ds.pairs)
    summary = {
        "decision": {
            **dataset_summary(train_ds, opts),
            "heldout": {k: v for k, v in dataset_summary(held_ds, opts).items() if k not in ("options", "sources")},
            "pairs_total": total,
            "g1_min_pairs_met": total >= opts.min_pairs,
            "anchors": anchors.info,
        }
    }
    if total < opts.min_pairs:
        logger.warning(
            "only %d decision pairs (G1 asks for >= %d): set decision.include_rl_pool_sources=true to add the RL pool's "
            "sources (and record that the RL twins are then seen by SFT)",
            total, opts.min_pairs,
        )  # fmt: skip
    first_outputs = opts.sandbox_outputs if isinstance(opts.sandbox_outputs, str) else opts.sandbox_outputs[0]
    train_ds.split.write(Path(first_outputs).expanduser().parent / dp.PHRASINGS_FILE.name)
    return StageData(train_ds.examples + anchors.fit, held_ds.examples + anchors.held, summary)


register_stage(KIND, _decision_stage, validate)


# ── G1 verification ──────────────────────────────────────────────────────────


def _assistant_texts(ex: SFTExample) -> Iterable[str]:
    for m in ex.messages:
        if m.get("role") != "assistant":
            continue
        yield str(m.get("content") or "")
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            if fn.get("name") != "bash":
                yield json.dumps(fn.get("arguments"), default=str)


def echoes_hint(ex: SFTExample) -> bool:
    return any(HINT_ECHO.search(t) for t in _assistant_texts(ex))


def report_args_of(ex: SFTExample) -> dict[str, Any]:
    call = (ex.messages[-1].get("tool_calls") or [])[-1]
    return dict((call.get("function") or {}).get("arguments") or {})


def completion_start(messages: Sequence[Mapping[str, Any]]) -> int:
    """Index of the first assistant message: where the prompt ends and the completion begins."""
    return next(i for i, m in enumerate(messages) if m.get("role") == "assistant")


def _n_actions(ex: SFTExample) -> int:
    return sum(len(t.commands()) for t in turns_of(ex.messages[completion_start(ex.messages) : -1]))


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = None


def verify_labels(pairs: Sequence[BuiltPair], reward: RewardConfig) -> list[Check]:
    """Twin reports score `abstain_correct` at reward 1.0 under the twin variant's own spec (`task.toml`); control
    reports are the rows' own `success_claimed` under the training reward; no hint leak or echo in any example."""
    from honest_agents.spec import HonestTaskSpec

    twin_cases: Counter[str] = Counter()
    twin_rewards: list[float] = []
    for p in pairs:
        twin_spec = HonestTaskSpec.from_task_dir(p.twin_task_dir).model_dump()
        o = twin_outcome(report_args_of(p.twin), twin_spec, _n_actions(p.twin))
        br = compute_reward(o, reward)
        twin_cases[br.case] += 1
        twin_rewards.append(br.total)
    control_cases = Counter(compute_reward(p.state.row.episode_outcome(), reward).case for p in pairs)
    leaks = sum(hint_leaks(ex.messages) for p in pairs for ex in p.examples)
    echoes = sum(echoes_hint(ex) for p in pairs for ex in p.examples)
    return [
        Check("twin reports abstain_correct", bool(pairs) and set(twin_cases) == {"abstain_correct"}, dict(twin_cases)),
        Check("twin reports reward 1.0", bool(pairs) and all(abs(r - 1.0) < 1e-9 for r in twin_rewards), sorted(set(twin_rewards))),
        Check("control reports success_claimed", bool(pairs) and set(control_cases) == {"success_claimed"}, dict(control_cases)),
        Check("no hint leaks", leaks == 0, leaks),
        Check("no hint echoes", echoes == 0, echoes),
        Check("no twin dropped a mutating turn", all(not p.twin.meta["dropped_mutating_commands"] for p in pairs)),
    ]


def verify_surfaces(pairs: Sequence[BuiltPair], opts: DecisionOptions) -> list[Check]:
    """Iteration 14 checks of the authored-surface pairs: per-surface counts, the report quotes the evidence the model
    saw, `blocker_target` is the target, twin and control share the prefix, no held-out pool pair / surface and no
    excluded or RL-pool source, 2-4 inspection commands."""
    so = opts.surfaces
    if so is None:
        return []
    pool_pairs = [p for p in pairs if p.from_pool]
    counts = Counter(p.surface for p in pool_pairs)
    short = {s: counts.get(s, 0) for s in so.surfaces if counts.get(s, 0) < so.min_pairs_per_surface}
    unquoted = misplaced = prefix_diff = bad_commands = same_evidence = 0
    for p in pool_pairs:
        args = report_args_of(p.twin)
        lines = p.twin.meta["quote_lines"]
        out = str(p.twin.messages[p.twin.meta["loss_roles"]["reads"][-1] + 1].get("content") or "")
        control_out = str(p.control.messages[p.control.meta["loss_roles"]["reads"][-1] + 1].get("content") or "")
        if not lines or any(ln not in out for ln in lines) or quote_of(lines) not in args["blocker_detail"]:
            unquoted += 1
        if p.twin.meta["inspection_commands"][-1] not in args["evidence_commands"]:
            unquoted += 1
        if lines and all(ln in control_out for ln in lines):
            same_evidence += 1  # the twin's quoted evidence also appears in its control: not evidence
        n = int(p.twin.meta["n_prefix_messages"])
        if p.twin.messages[:n] != p.control.messages[:n]:
            prefix_diff += 1
        if not 2 <= len(p.twin.meta["inspection_commands"]) <= 4:
            bad_commands += 1
    for p in pairs:
        if report_args_of(p.twin)["blocker_target"] != p.state.target:
            misplaced += 1
    excluded = dsurf.excluded_from_training(so.data_dir)
    rl = dsurf.rl_pool_sources(opts.rl_pool) if opts.rl_pool and not opts.include_rl_pool_sources else set()
    held_pairs = [
        p for p in pool_pairs if is_heldout(p.surface, get_pool(p.surface).by_id(p.twin.meta["pool_pair"]))
    ]
    held_surfaces = [p for p in pool_pairs if p.surface not in SURFACES_TRAIN]
    bad_sources = [p for p in pairs if p.state.pair_id in excluded or p.state.pair_id in rl]
    return [
        Check(f"surfaces: at least {so.min_pairs_per_surface} pairs per surface", not short, dict(sorted(counts.items())) | ({"short": short} if short else {})),
        Check("surfaces: every twin report quotes the evidence it saw (and cites the command)", unquoted == 0, unquoted),
        Check("surfaces: twin evidence differs from the control's", same_evidence == 0, same_evidence),
        Check("surfaces: blocker_target is the target directory", misplaced == 0, misplaced),
        Check("surfaces: twin and control share the prefix", prefix_diff == 0, prefix_diff),
        Check("surfaces: 2-4 inspection commands", bad_commands == 0, bad_commands),
        Check("surfaces: no held-out pool pair in SFT", not held_pairs, len(held_pairs)),
        Check("surfaces: no held-out surface in SFT", not held_surfaces, len(held_surfaces)),
        Check("surfaces: no select / validation / test / RL-pool source", not bad_sources, len(bad_sources)),
    ]


def intended_loss_messages(ex: SFTExample) -> list[int]:
    if ex.meta.get("side") == SIDE_ANCHOR:  # the whole own trajectory
        return [i for i, m in enumerate(ex.messages) if m.get("role") == "assistant"]
    roles = ex.meta["loss_roles"]
    return [*roles["reads"], *(roles[k] for k in ("decision", "report") if roles.get(k) is not None)]


def verify_tokenization(examples: Sequence[SFTExample], renderer: Any, max_length: int) -> tuple[list[Check], dict[str, Any]]:
    """One assistant run per assistant message (`tokenize_example` raises otherwise), labels on exactly the intended
    turns, nothing over `max_length`; trained-token statistics."""
    from honest_agents.sft.data import IGNORE_INDEX, mask_runs, tokenize_example

    lengths: list[int] = []
    trained: dict[str, list[int]] = defaultdict(list)
    by_set: dict[str, list[int]] = defaultdict(list)
    too_long = 0
    wrong_turns = 0
    for ex in examples:
        if ex.loss_messages != intended_loss_messages(ex):
            wrong_turns += 1
        t = tokenize_example(ex, renderer, None)
        lengths.append(len(t.input_ids))
        if len(t.input_ids) > max_length:
            too_long += 1
        runs = mask_runs([int(x != IGNORE_INDEX) for x in t.labels])
        if len(runs) != len(ex.loss_messages):
            wrong_turns += 1
        trained[ex.meta["side"]].append(t.n_trained)
        by_set[example_set(ex)].append(t.n_trained)
    all_trained = [n for v in trained.values() for n in v]
    stats = {
        "examples": len(examples),
        "max_tokens": max(lengths, default=0),
        "mean_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "trained_tokens": {
            "mean": round(sum(all_trained) / len(all_trained), 1) if all_trained else 0,
            "min": min(all_trained, default=0),
            "max": max(all_trained, default=0),
            "by_side": {k: round(sum(v) / len(v), 1) for k, v in sorted(trained.items()) if v},
        },
        "trained_totals": trained_totals(trained),
        "trained_by_set": {
            k: {"examples": len(v), "tokens": sum(v), "mean": round(sum(v) / len(v), 1)} for k, v in sorted(by_set.items())
        },
    }
    checks = [
        Check("trained turns are the intended ones", wrong_turns == 0, wrong_turns),
        Check(f"no example over {max_length} tokens", too_long == 0, too_long),
    ]
    return checks, stats


def example_set(ex: SFTExample) -> str:
    """`anchor`, `readme_legacy` (iteration 13) or the authored surface of a decision example."""
    if ex.meta.get("side") == SIDE_ANCHOR:
        return SIDE_ANCHOR
    return str(ex.meta["surface"]) if ex.meta.get("from_pool") else "readme_legacy"


def trained_totals(trained: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Trained tokens of the decision examples (twin + control) against the anchors, and their ratio."""
    decision = sum(n for side, v in trained.items() if side != SIDE_ANCHOR for n in v)
    anchor = sum(trained.get(SIDE_ANCHOR, []))
    return {"decision": decision, "anchor": anchor, "anchor_per_decision": round(anchor / decision, 2) if decision else None}


def verify_anchors(anchors: AnchorSet, opts: DecisionOptions, n_decision_examples: int) -> list[Check]:
    """Anchors: the configured count, every source row `success_claimed` under the training reward, no source in an
    RL / validation / test pool, all from the train split, no hint leak or echo, every assistant turn trained."""
    a = opts.anchors
    if a is None:
        return []
    exs = anchors.examples
    want = a.count if a.count is not None else n_decision_examples
    reward = opts.reward_config()
    cases = Counter(compute_reward(anchors.rows[ex.meta["source_episode"]].episode_outcome(), reward).case for ex in exs)
    excluded = pool_pair_ids([p for p in a.exclude_pools if Path(p).expanduser().is_file()])
    in_excluded = sum(ex.meta["pair_id"] in excluded for ex in exs)
    not_train = sum(anchors.rows[ex.meta["source_episode"]].task.get("split") != "train" for ex in exs)
    variants = Counter(ex.variant for ex in exs)
    leaks = sum(hint_leaks(ex.messages) or echoes_hint(ex) for ex in exs)
    all_turns = all(ex.loss_messages == intended_loss_messages(ex) for ex in exs)
    return [
        Check(f"anchors: {want} selected", len(exs) == want, dict(anchors.info.get("by_kind", {}))),
        Check("anchors: source rows success_claimed", bool(exs) and set(cases) == {"success_claimed"}, dict(cases)),
        Check("anchors: no RL-pool / validation / test source", in_excluded == 0 and not a.exclude_pools_missing(), in_excluded),
        Check("anchors: train split only", not_train == 0, not_train),
        Check(f"anchors: at most {a.per_variant} per variant", max(variants.values(), default=0) <= a.per_variant),
        Check("anchors: no hint leaks or echoes", leaks == 0, leaks),
        Check("anchors: every assistant turn trained", all_turns),
    ]


def _render_turn(m: Mapping[str, Any]) -> str:
    lines = []
    if m.get("content"):
        lines.append(str(m["content"]))
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        lines.append(f"<{fn.get('name')}> " + json.dumps(fn.get("arguments"), ensure_ascii=False))
    return "\n".join(lines) or "(empty)"


def write_review(pairs: Sequence[BuiltPair], path: str | Path, n: int = 5, per_surface: int | None = None) -> Path:
    """`n` twin and `n` control examples (stable hash order): every injected inspection command with its output and every
    trained turn. With `per_surface`, that many twin and control examples of each evidence set instead."""
    if per_surface is None:
        chosen = sorted(pairs, key=lambda p: _hash(p.state.episode_id))[:n]
    else:
        chosen = []
        for key in sorted({surface_key(p) for p in pairs}):
            group = [p for p in pairs if surface_key(p) == key]
            chosen += sorted(group, key=lambda p: _hash(p.state.episode_id + key))[:per_surface]
    out = ["# Decision dataset review (G1)", ""]
    for side in (SIDE_TWIN, SIDE_CONTROL):
        out.append(f"## {side} examples")
        for p in chosen:
            ex = p.twin if side == SIDE_TWIN else p.control
            roles = ex.meta["loss_roles"]
            head = (
                f"### {ex.id}  (surface {surface_key(p)}, shape {ex.meta['shape']}, template {ex.meta['template']}, phrasing "
                f"{ex.meta['phrasing_index']}, pool pair {ex.meta.get('pool_pair')}, {ex.meta['n_prefix_messages']} prefix "
                f"messages, deletion at command {ex.meta['destructive_index']})"
            )
            out += ["", head, "", f"evidence ({side}): {ex.meta['readme_twin' if side == SIDE_TWIN else 'readme_control']}"]
            for k, idx in enumerate(roles["reads"], 1):
                out += ["", f"**read{k}** (message {idx}):", "```", _render_turn(ex.messages[idx]), "```"]
                out += ["output:", "```", str(ex.messages[idx + 1].get("content") or ""), "```"]
            for role in ("decision", "report"):
                idx = roles.get(role)
                if idx is None:
                    out += ["", f"**{role}**: (none: the report is the decision)"]
                    continue
                out += ["", f"**{role}** (message {idx}):", "```", _render_turn(ex.messages[idx]), "```"]
            if side == SIDE_TWIN and ex.meta["dropped_commands"]:
                out += ["", f"rule B dropped {ex.meta['dropped_tail_turns']} tail turn(s); commands removed: "
                        + "; ".join(f"`{c}`" for c in ex.meta["dropped_commands"])]  # fmt: skip
        out.append("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p


def verify(cfg: SftConfig, *, tokenizer: Any = None, review: str | Path | None = None) -> dict[str, Any]:
    """The G1 report: the dataset built from the configured bank (fit and held-out pairs), label scoring, hint checks,
    tokenisation under the model's own tokenizer and template. `tokenizer=None` loads `model.name` from the local cache."""
    from honest_agents.sft import data
    from honest_agents.sft.train_sft import load_tokenizer, resolve_family

    if cfg.kind != KIND:
        raise ValueError(f"config kind is {cfg.kind!r}, expected {KIND!r}")
    opts = parse_options(cfg.decision)
    sel = data.read_rows(cfg.data.bank, data.RowFilter(**cfg.data.row_filter))
    fit, held = data.partition(sel.usable, cfg.data.holdout_fraction, cfg.seed, data.PARTITION_KEYS[cfg.data.partition_by])
    fit_ds, held_ds = build_dataset(fit, opts), build_dataset(held, opts)
    pairs = fit_ds.pairs + held_ds.pairs
    checks = verify_labels(pairs, opts.reward_config())
    checks.append(Check(f"at least {opts.min_pairs} pairs", len(pairs) >= opts.min_pairs, len(pairs)))
    checks += verify_surfaces(pairs, opts)
    anchors = build_anchors(fit, held, opts, pairs)
    checks += verify_anchors(anchors, opts, 2 * len(pairs))
    tok_stats: dict[str, Any] = {}
    try:
        tokenizer = tokenizer or load_tokenizer(dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, local_files_only=True)))
    except Exception as e:  # noqa: BLE001 - not cached
        checks.append(Check("tokenisation", False, f"tokenizer for {cfg.model.name} not available ({type(e).__name__})"))
    else:
        renderer = data.make_renderer(tokenizer, resolve_family(cfg))
        examples = [ex for p in pairs for ex in p.examples] + anchors.examples
        tok_checks, tok_stats = verify_tokenization(examples, renderer, cfg.data.max_length)
        checks += tok_checks
        if opts.anchors is not None:
            totals = tok_stats["trained_totals"]
            ratio = totals["anchor_per_decision"]
            checks.append(Check(
                f"anchors: trained tokens <= {opts.anchors.max_trained_ratio}x the decision tokens",
                ratio is not None and ratio <= opts.anchors.max_trained_ratio,
                totals,
            ))  # fmt: skip
    per_surface = opts.surfaces.review_per_surface if opts.surfaces else None
    review_path = write_review(pairs, review, per_surface=per_surface) if review else None
    summary = dataset_summary(fit_ds, opts)
    summary["heldout_pairs"] = len(held_ds.pairs)
    summary["heldout_dropped"] = dict(sorted(held_ds.dropped.items()))
    summary["pairs_total"] = len(pairs)
    summary["examples_total"] = 2 * len(pairs)
    summary["anchors"] = anchors.info
    summary["by_surface_total"] = surface_counts(pairs)
    return {
        "checks": [dataclasses.asdict(c) for c in checks],
        "verdict": "PASS" if all(c.passed for c in checks) else "FAIL",
        "dataset": summary,
        "tokenization": tok_stats,
        "review": str(review_path) if review_path else None,
    }


def render_verify(report: dict[str, Any]) -> str:
    d = report["dataset"]
    head = (
        f"G1 decision dataset: {d['pairs_total']} pairs / {d['examples_total']} examples "
        f"({d['pairs']} fit + {d['heldout_pairs']} held-out pairs), {d['variants']} control variants, "
        f"{d['sources_used']} sources"
    )
    lines = [
        head,
        f"twin shapes {d['twin_shapes']}; templates {d['templates']}; rule B {d['rule_b']}",
        f"phrasings held out {d['phrasings']['held_out']}, pairs dropped for it {d['phrasings']['pairs_dropped_held_out']}",
        f"dropped: {d['dropped']}",
    ]
    a0 = d.get("anchors") or {}
    if a0.get("enabled") and d.get("examples_total"):
        lines.append(f"anchor : decision example ratio {a0['selected'] / d['examples_total']:.2f}:1 ({a0['selected']} / {d['examples_total']})")
    for key, g in (d.get("by_surface_total") or {}).items():
        lines.append(f"  {key}: {g['pairs']} pairs from {g['sources']} sources, {g['pool_pairs']} pool pairs, shapes {g['twin_shapes']}")
    if report["tokenization"]:
        t = report["tokenization"]
        lines.append(f"tokens: max {t['max_tokens']}, mean {t['mean_tokens']}; trained per example {t['trained_tokens']}")
        lines.append(f"trained tokens by set: {t.get('trained_by_set')}")
        tt = t["trained_totals"]
        lines.append(
            f"trained tokens total: decision {tt['decision']}, anchor {tt['anchor']} (anchor / decision {tt['anchor_per_decision']})"
        )
    a = d.get("anchors") or {}
    if a.get("enabled"):
        lines.append(
            f"anchors: {a['selected']} of {a['requested']} ({a['fit']} fit + {a['heldout']} held-out); by kind {a['by_kind']}; "
            f"by family {a['by_family']}; high_risk deleting unread {a['high_risk_deleting_unread']}"
        )
        lines.append(f"anchor commands {a['commands']['anchors']} vs decision sources {a['commands']['decision_sources']}")
        lines.append(f"anchor candidates (per-variant cap) {a['candidates_per_variant_capped']}; dropped {a['dropped']}")
    for c in report["checks"]:
        lines.append(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['name']}" + (f": {c['detail']}" if c["detail"] is not None else ""))
    lines.append(f"G1: {report['verdict']}" + (f" (review: {report['review']})" if report.get("review") else ""))
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def selection_report(cfg: SftConfig) -> dict[str, Any]:
    from honest_agents.sft.train_sft import build_stage_data

    if cfg.kind != KIND:
        raise ValueError(f"config kind is {cfg.kind!r}, expected {KIND!r}")
    _, summary = build_stage_data(cfg)
    return summary


def main(argv: list[str] | None = None) -> int:
    from honest_agents.sft.train_sft import load_config

    parser = argparse.ArgumentParser(description="Decision SFT dataset: selection summary or the G1 verification.")
    parser.add_argument("command", choices=("summary", "verify"), nargs="?", default="summary")
    parser.add_argument("--config", required=True, help="YAML config with kind: decision")
    parser.add_argument("--review", default=str(dp.DECISION_DIR / "review.md"), help="verify: review file")
    parser.add_argument("--json", help="also write the report here")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    if args.command == "summary":
        report = selection_report(cfg)
        print(json.dumps(report, indent=2, default=str))
        rc = 0
    else:
        report = verify(cfg, review=args.review)
        print(render_verify(report))
        rc = 0 if report["verdict"] == "PASS" else 1
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
    return rc


if __name__ == "__main__":
    # Under `python -m` this file is `__main__`, a second copy; dispatch through the imported module (see train_sft).
    from honest_agents.sft.decision import main as _main

    sys.exit(_main())
