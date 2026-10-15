"""The G0 judgment items spliced into the model's own long runs (`kind: judgment_splice`, iteration 15 G1,
docs/iteration-15-g1-spliced-judgment.md).

G0 (`sft.judgment`) trained the judgment on short items (5-7 messages) and did not transfer into long agentic contexts
(~26 messages, ~9 own commands). G1 keeps the items and moves them into the iteration-13/14 decision pipeline's own
prefixes (`sft.decision`):

* **States.** Exactly the SFT-v2 decision states: verified-success high_risk control episodes of the pass@4 bank whose
  source is in the distillation pool, cut just before the deletion of their sentinel target T, phrasing holdout and
  per-variant cap as in `decision.build_dataset` (the iteration-13 README pairs; the evidence-surface pairs sit on a
  subset of the same states). Sources are asserted to be distill-pool sources, never RL-pool, select, validation or test
  (`tasks.select_split.excluded_from_training`).
* **Items.** `pairs_per_format` (75) G0 pairs per trained format, never a held-out pair (`judgment.heldout_pair_ids`),
  taken in a stable hash order; each pair is assigned to one state (round robin over the states in a stable hash order,
  so the spread is even to within one) and rewritten to that state's T (`rewrite_item`): the item's target path becomes
  T in commands, outputs and rationale, its parent directory becomes T's parent, and the item's directory name where it
  stands alone becomes T's. An item that cannot be rewritten cleanly (volume-name target, the target inside a longer
  name, an ambiguous generic directory name left over, a listing that shows T itself as a regular file, a read that
  mutates, a label word, a decisive line the budget meter cut) is dropped with a counted reason and the next pair of
  that format takes its place. A pair whose reads (commands and outputs, both sides) never name T, a path inside it or
  a directory above it is dropped too (`reads_never_name_target`): the item's own task text, which carried that link
  ("remove the datadir of the `an` instance"), is not part of the spliced example, so such evidence would teach
  "alarming text anywhere -> refuse" rather than a judgment about T.
* **Examples** (per pair, two, on the SAME own prefix). *Benign*: the pair's benign reads (injected assistant turns and
  tool outputs rendered through the episode's budget meter), then the control's own deletion turn with its content
  replaced by the benign rationale (rule A), the control's own tail and its own verified-success report. *Danger*: the
  danger reads, then the decision pipeline's twin construction (`BuiltPair.twin`: stripped deletion turn / next clean own
  turn / straight to the report, tail cut by rule B), then the twin report: the control's verified `completed`,
  `incomplete` = the deletion not done, `risk_discovered`, `blocker_target` = T, `blocker_detail` quoting the decisive
  output line (`judgment.decisive_line`), `evidence_commands` = the injected read commands (rule C); the report turn's
  content is the danger rationale. Loss on the injected read turns, the decision turn and the report (as SFT-v2);
  everything else is the model's own context and is masked. `splice.train_reads: false` masks the read turns as well
  (loss on the decision and the report only, as G0): the read commands are the item authors' (`cat T/ROLLUP`,
  `cat /etc/ca/ca.cnf`), files the bank's sandboxes do not have, so training them teaches inspection commands that
  cannot succeed at deployment.
* **Anchors.** `decision.anchors.count` (400) own-success anchors, `decision.build_anchors`.
* **Eval.** `eval_pairs_per_format` held-out G0 pairs per format, spliced the same way (never trained on).

    python -m honest_agents.sft.judgment_splice verify --config configs/sft_judgment_splice_g1_qwen35.yaml
    python -m honest_agents.sft.train_sft --config configs/sft_judgment_splice_g1_qwen35.yaml --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import posixpath
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.bank import BankRow
from honest_agents.harness import report_from_tool_args
from honest_agents.mutation import is_mutating
from honest_agents.readme_read import is_destructive_on
from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import decision as dec
from honest_agents.sft import judgment as jd
from honest_agents.sft.data import SFTExample, mask_runs, tokenize_example
from honest_agents.sft.selfdistill import HINT_ECHO, hint_leaks
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage
from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks.analysis.labels import LABEL_RE
from honest_agents.tasks.select_split import excluded_from_training

logger = logging.getLogger(__name__)

KIND = "judgment_splice"
SIDE_DANGER = jd.SIDE_DANGER
SIDE_BENIGN = jd.SIDE_BENIGN
SIDE_ANCHOR = dec.SIDE_ANCHOR
SOURCE = "judgment_splice"
DEFAULT_REVIEW = Path("data/judgment/review_g1.md")

# Drop reasons of a pair (the first one that applies).
DROP_VOLUME_TARGET = "target_is_a_volume_name"  # no path to map onto T
DROP_TARGET_IN_LONGER_NAME = (
    "target_inside_a_longer_name"  # `<target>.txt`, `<target>-new`, `/mnt<target>`: not a clean rewrite
)
DROP_GENERIC_BASENAME = "generic_directory_name_left_over"  # the bare name (`cache`, `2024`) cannot be mapped safely
DROP_TARGET_AS_FILE = "target_listed_as_a_regular_file"  # an `ls -l` line shows T itself as `-rw-...`: T is a directory
DROP_UNANCHORED = "reads_never_name_target"  # no read (command or output, either side) names T, a subpath or a parent
DROP_READ_MUTATING = "read_mutates_after_rewrite"
DROP_LABEL_WORD = "label_word_after_rewrite"
DROP_NO_QUOTE = "decisive_line_not_in_rendered_output"
DROP_HINT = "hint_leak_or_echo"
DROP_MALFORMED = "decisive_line_missing"

# The danger decision turn when the twin keeps a turn after the reads and its own content cannot stay (rule A). Generic
# over the evidence: the specific facts are in the reads above and in the report.
DANGER_DECISION_LINES = (
    "What that output shows means `{dir}` has to stay, so I will not delete it. Let me continue with the rest of the task:",
    "Given that output, removing `{dir}` would destroy something that matters; I am leaving it in place. Continuing:",
    "Because of what I just read I will skip the deletion of `{dir}` and finish the remaining work:",
    "That output is a reason not to remove `{dir}`. Leaving it alone and moving on:",
)


# ── options ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SpliceOptions:
    pairs_per_format: int = 75  # training pairs per format (2 examples each)
    min_pairs_per_format: int = 75  # verify: fewer (after drops) fails
    seed: str = "it15-g1"  # stable hash seed of the pair order and the state order
    eval_pairs_per_format: int = 4  # held-out G0 pairs per format spliced into the eval set
    max_state_spread: int = 1  # verify: pairs per state differ by at most this
    review_per_format: int = 1  # verify: pairs per format in the review file
    # Loss on the injected read turns (content line + the item's command) as SFT-v2 did with real sandbox commands.
    # False: the reads stay in the context but are masked; only the decision turn and the report are trained (as G0).
    train_reads: bool = True

    def __post_init__(self) -> None:
        if self.pairs_per_format < 1:
            raise ValueError("splice.pairs_per_format must be >= 1")
        if not 0 <= self.min_pairs_per_format <= self.pairs_per_format:
            raise ValueError("splice.min_pairs_per_format must be in [0, pairs_per_format]")
        if self.eval_pairs_per_format < 0:
            raise ValueError("splice.eval_pairs_per_format must be >= 0")


def parse_options(raw: dict[str, Any] | None) -> SpliceOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(SpliceOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `splice:`; expected {sorted(allowed)}")
    return SpliceOptions(**raw)


def validate(cfg: SftConfig) -> None:
    """Called by `train_sft.parse_config`: bad stage keys fail before any data is read."""
    parse_options(cfg.splice)
    jd.parse_options(cfg.judgment)
    dec.validate(cfg)
    if dec.parse_options(cfg.decision).anchors is None:
        raise ValueError("judgment_splice needs `decision.anchors` (the own-success anchors)")


# ── rewriting an item to the state's target ──────────────────────────────────


def is_distinctive(name: str) -> bool:
    """A directory name that is safe to replace where it stands alone: it has letters, at least four characters and a
    separator / digit (`ledger-q3`, `pg_old`, `2024Q4`) or is long. Plain words (`cache`, `uploads`) and bare numbers
    (`2024`, which is also a year in a date) are ambiguous."""
    if len(name) < 4 or not re.search(r"[A-Za-z]", name):
        return False
    return bool(re.search(r"[-_.]|\d", name)) or len(name) >= 10


# A file extension: a dot, then 1-11 letters / digits / hyphens with at least one letter (`.psd`, `.sqlite3`, `.parquet`,
# `.db-journal`, `.crdownload`); `.1` (a rotated log) is not one, and a name without a dot is a directory.
_FILE_EXT = re.compile(r"\.(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9-]{0,10}$")
# An `ls -l` line of a regular file (`-rw-r--r-- 1 ...`).
_LS_REGULAR_FILE = re.compile(r"^\s*-[rwxsStT-]{9}[.+@]?\s")


class PathRewrite:
    """Maps an item's target path `old` to the state's `new` in free text: the path itself (also as a prefix of a
    subpath), then its parent directory, then the bare directory name. Boundaries keep `/mnt/srv/x` and `/srv/xylophone`
    out of the rewrite of `/srv/x`. A target with a file extension (`/home/gl/art/hero.psd`) is a file: it becomes
    `new/hero.psd` and its parent directory becomes `new` itself, so `exiftool` and `head` still read a file and
    deleting `new` deletes it; its bare name stays valid."""

    def __init__(self, old: str, new: str):
        self.old = old.rstrip("/")
        self.new = new.rstrip("/")
        self.old_base = posixpath.basename(self.old)
        self.new_base = posixpath.basename(self.new)
        self.file_like = _FILE_EXT.search(self.old_base) is not None
        self.new_path = f"{self.new}/{self.old_base}" if self.file_like else self.new
        old_parent = posixpath.dirname(self.old)
        self.parent_to = self.new if self.file_like else posixpath.dirname(self.new)
        parts = [rf"(?P<t>{re.escape(self.old)})(?![\w-]|\.\w)"]
        if old_parent not in ("", "/") and self.parent_to not in ("", "/") and old_parent != self.parent_to:
            parts.append(rf"(?P<p>{re.escape(old_parent)})(?=/|(?![\w.-]))")
        self._paths = re.compile(r"(?<![\w.~-])(?:" + "|".join(parts) + ")")
        self._sibling = re.compile(rf"(?<![\w.~-]){re.escape(self.old)}(?=[\w-]|\.\w)")
        self._base = re.compile(rf"(?<![\w.~-]){re.escape(self.old_base)}(?![\w-]|\.\w)")

    def paths(self, text: str) -> str:
        return self._paths.sub(lambda m: self.new_path if m.group("t") else self.parent_to, text)

    def has_sibling(self, text: str) -> bool:
        """`text` names a path that merely starts with the target (`<target>.bak`, `<target>-new`)."""
        return self._sibling.search(text) is not None

    def has_base(self, text: str) -> bool:
        return not self.file_like and self.old_base != self.new_base and self._base.search(text) is not None

    def base(self, text: str) -> str:
        return self._base.sub(self.new_base, text) if self.old_base != self.new_base else text


def lists_target_as_file(output: str, target: str) -> bool:
    """An `ls -l`-style line of `output` shows `target` itself (by path or bare name) as a regular file."""
    target = target.rstrip("/")
    base = posixpath.basename(target)
    for line in output.splitlines():
        s = line.rstrip()
        if _LS_REGULAR_FILE.match(s) and s.endswith((" " + target, " " + base, "/" + base)):
            return True
    return False


def _ancestors(target: str) -> list[str]:
    out: list[str] = []
    p = posixpath.dirname(posixpath.normpath(target))
    while p not in ("", "/"):
        out.append(p)
        p = posixpath.dirname(p)
    return out


def names_target_text(text: str, target: str) -> bool:
    """`text` names `target`: the path (or a path inside it), its last component as a whole word, or a directory above
    it as a whole path token (`Where: /mnt/volume` for `/mnt/volume/juniper`)."""
    if dec.mentions_target(text, target):
        return True
    return any(re.search(rf"(?<![\w/.~-]){re.escape(a)}(?![\w/.-])", text) for a in _ancestors(target))


def reads_name_target(item: jd.Item, target: str) -> bool:
    """Some read of the (rewritten) item names `target`, in its command (shell word, ancestors included) or its output."""
    return any(
        dec.command_names_target(r.command, target) or names_target_text(r.command, target) or names_target_text(r.output, target)
        for r in item.reads
    )  # fmt: skip


@dataclass(frozen=True)
class Rewritten:
    item: jd.Item
    original_target: str
    base_replaced: bool  # the bare directory name was replaced somewhere


def rewrite_item(item: jd.Item, new_target: str) -> Rewritten | str:
    """`item` with its target mapped to `new_target`, or the drop reason. The task text is left alone (the splice does
    not use it: the model's own prompt stands)."""
    if not item.target.startswith("/"):
        return DROP_VOLUME_TARGET
    rw = PathRewrite(item.target, new_target)
    raw = [item.rationale, *(t for r in item.reads for t in (r.command, r.output))]
    if any(rw.has_sibling(t) for t in raw):
        return DROP_TARGET_IN_LONGER_NAME
    fields = {
        "rationale": rw.paths(item.rationale),
        "decisive": rw.paths(item.decisive) if item.decisive else None,
        "reads": [(rw.paths(r.command), rw.paths(r.output)) for r in item.reads],
    }
    flat = [fields["rationale"], *(t for pair in fields["reads"] for t in pair)]
    if any(rw.old in t for t in flat):
        return DROP_TARGET_IN_LONGER_NAME
    replaced = False
    if any(rw.has_base(t) for t in flat):
        if not is_distinctive(rw.old_base):
            return DROP_GENERIC_BASENAME
        replaced = True
        fields["rationale"] = rw.base(fields["rationale"])
        fields["decisive"] = rw.base(fields["decisive"]) if fields["decisive"] else None
        fields["reads"] = [(rw.base(c), rw.base(o)) for c, o in fields["reads"]]
    reads = tuple(jd.Read(c.strip(), o.rstrip("\n")) for c, o in fields["reads"])
    out = dataclasses.replace(
        item, target=new_target, rationale=fields["rationale"], decisive=fields["decisive"], reads=reads
    )
    if not rw.file_like and any(lists_target_as_file(r.output, new_target) for r in reads):
        return (
            DROP_TARGET_AS_FILE  # the bank task deletes a directory; a listing that makes T a plain file contradicts it
        )
    if any(is_destructive_on(r.command, [new_target]) or is_mutating(r.command) for r in reads):
        return DROP_READ_MUTATING
    if any(LABEL_RE.search(t) for t in [out.rationale, *(r.command + "\n" + r.output for r in reads)]):
        return DROP_LABEL_WORD
    return Rewritten(out, item.target, replaced)


# ── the splice ───────────────────────────────────────────────────────────────


def _pick(lines: Sequence[str], key: str) -> str:
    return lines[int(dec.stable_hash(key)[:8], 16) % len(lines)]


def _bash_commands(message: Mapping[str, Any]) -> list[str]:
    return [
        str(((c.get("function") or {}).get("arguments") or {}).get("command") or "")
        for c in message.get("tool_calls") or []
        if (c.get("function") or {}).get("name") == "bash"
    ]


@dataclass
class SplicedPair:
    base: dec.BuiltPair  # the decision pair of the state (prefix, twin shape, task directory of the twin)
    pair_id: str  # the G0 pair
    format: str
    hard: bool
    original_target: str
    base_replaced: bool
    danger: SFTExample
    benign: SFTExample
    heldout: bool = False

    @property
    def state(self) -> dec.DecisionState:
        return self.base.state

    @property
    def examples(self) -> list[SFTExample]:
        return [self.benign, self.danger]


def injected_reads(item: jd.Item, key: str, prefix_texts: Sequence[str]) -> list[dict[str, Any]]:
    """`[READ1, OUT1, READ2, OUT2, ...]` of an item: one assistant turn per command, content line a stable function of
    `key` (the same for both sides of a pair), outputs rendered after the tool outputs already in the transcript."""
    outs = dp.render_outputs(prefix_texts, [dp.CommandOutput(r.command, 0, r.output) for r in item.reads])
    msgs: list[dict[str, Any]] = []
    for k, (read, out) in enumerate(zip(item.reads, outs, strict=True)):
        line = (
            _pick(jd.FIRST_READ_LINES, key).format(dir=item.target)
            if k == 0
            else _pick(jd.NEXT_READ_LINES, f"{key}:{k}")
        )
        msgs += [dp.bash_turn(line, read.command), dp.tool_message(out)]
    return msgs


def _injected_messages(ex: SFTExample) -> list[dict[str, Any]]:
    """The messages of an example the splice wrote: the reads, the decision turn and, on the danger side, the report."""
    roles = ex.meta["loss_roles"]
    first = roles["reads"][0]
    idx = list(range(first, first + 2 * len(roles["reads"])))
    if roles.get("decision") is not None:
        idx.append(roles["decision"])
    if ex.meta["side"] == SIDE_DANGER:
        idx.append(roles["report"])
    return [ex.messages[i] for i in idx]


def _message_texts(m: Mapping[str, Any], *, reports: bool = True) -> list[str]:
    """The text of a message: content and tool-call arguments. The report's `completed` list is the model's own words
    (the control's verified list) and is left out; `reports=False` leaves the report arguments out altogether (the
    item-derived text in them is the quote, which is a line of an output, and the fixed templates)."""
    out = [str(m.get("content") or "")]
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if fn.get("name") == "submit_report":
            if not reports or not isinstance(args, dict):
                continue
            args = {k: v for k, v in args.items() if k != "completed"}
        out.append(json.dumps(args, default=str, ensure_ascii=False))
    return out


def build_spliced(
    base: dec.BuiltPair, danger: Rewritten, benign: Rewritten, *, heldout: bool = False, train_reads: bool = True
) -> SplicedPair | str:
    """Both examples of one G0 pair on the state of `base`, or a drop reason. `train_reads=False` leaves the injected
    read turns out of the loss (they stay in the context)."""
    state, target = base.state, base.state.target
    d_item, b_item = danger.item, benign.item
    n_prefix = int(base.twin.meta["n_prefix_messages"])
    n_old_reads = int(base.twin.meta["n_read_messages"])
    prefix = base.twin.messages[:n_prefix]
    if base.control.messages[:n_prefix] != prefix:
        raise ValueError(f"{state.episode_id}: twin and control of the base pair do not share the prefix")
    prefix_texts = dp.tool_texts(prefix)
    key = f"{state.episode_id}:{d_item.pair_id}"
    try:
        k_quote, quote = jd.decisive_line(d_item, b_item)
    except ValueError:
        return DROP_MALFORMED
    reads_d = injected_reads(d_item, key, prefix_texts)
    reads_b = injected_reads(b_item, key, prefix_texts)
    if quote not in str(reads_d[2 * k_quote + 1].get("content") or ""):
        return DROP_NO_QUOTE
    idx_d = [n_prefix + 2 * k for k in range(len(d_item.reads))]
    idx_b = [n_prefix + 2 * k for k in range(len(b_item.reads))]
    common = {
        "source": SOURCE,
        "format": d_item.format,
        "pair_id": d_item.pair_id,
        "hard": d_item.hard,
        "heldout": heldout,
        "target": target,
        "original_target": danger.original_target,
        "base_replaced": danger.base_replaced or benign.base_replaced,
        "state_pair": state.pair_id,
        "state_episode": state.episode_id,
        "state_variant": state.variant_id,
        "n_prefix_messages": n_prefix,
        "destructive_index": state.destructive_index,
        "surface": d_item.format,  # `decision.example_set` groups by it
        "from_pool": True,
        "reads_trained": train_reads,
    }
    trained_reads_d = idx_d if train_reads else []
    trained_reads_b = idx_b if train_reads else []

    # Benign: the reads, the control's own deletion turn (rule A: the benign rationale), its tail and its report.
    deletion, tail = state.decision_and_tail()[0], state.decision_and_tail()[1:]
    c_messages = [*prefix, *reads_b, *dec.with_content(deletion, b_item.rationale).messages()]
    for turn in tail:
        c_messages += turn.messages()
    c_messages.append(base.control.messages[-1])  # the control's own verified-success report, verbatim
    roles_b = {"reads": idx_b, "decision": n_prefix + len(reads_b), "report": len(c_messages) - 1}
    benign_ex = SFTExample(
        id=f"{state.episode_id}:{d_item.pair_id}:{SIDE_BENIGN}",
        kind=KIND,
        variant=state.variant_id,
        messages=c_messages,
        loss_messages=[*trained_reads_b, roles_b["decision"], roles_b["report"]],
        meta={**common, "side": SIDE_BENIGN, "shape": dec.SHAPE_CONTROL, "loss_roles": roles_b, "n_reads": len(idx_b)},
    )

    # Danger: the reads, the twin construction (rule B tail), the twin report (verified `completed` of the control).
    kept = dec.turns_of(base.twin.messages[n_prefix + n_old_reads : -1])
    shape = base.twin_shape
    if kept and base.twin.meta["decision_content_replaced"]:
        line = _pick(DANGER_DECISION_LINES, f"{key}:decision").format(dir=target)
        kept[0] = dec.with_content(kept[0], line)
    t_messages = [*prefix, *reads_d]
    for turn in kept:
        t_messages += turn.messages()
    args = jd.danger_report_args(d_item, b_item)
    args["completed"] = list(dec.report_args_of(base.twin).get("completed") or [])
    report_from_tool_args("submit_report", args)  # must parse
    t_messages.append(dec.report_turn(args, d_item.rationale))
    roles_d: dict[str, Any] = {"reads": idx_d, "decision": n_prefix + len(reads_d) if kept else None}
    roles_d["report"] = len(t_messages) - 1
    danger_ex = SFTExample(
        id=f"{state.episode_id}:{d_item.pair_id}:{SIDE_DANGER}",
        kind=KIND,
        variant=state.variant_id,
        messages=t_messages,
        loss_messages=[*trained_reads_d, *(i for i in (roles_d["decision"], roles_d["report"]) if i is not None)],
        meta={
            **common,
            "side": SIDE_DANGER,
            "shape": shape,
            "loss_roles": roles_d,
            "n_reads": len(idx_d),
            "quote": quote,
            "dropped_tail_turns": base.dropped_tail_turns,
        },
    )
    for ex in (benign_ex, danger_ex):
        injected = _injected_messages(ex)
        texts = [t for m in injected for t in _message_texts(m)]
        if any(
            HINT_ECHO.search(t) for m in injected if m.get("role") == "assistant" for t in _message_texts(m)
        ) or hint_leaks(injected):
            return DROP_HINT
        if any(LABEL_RE.search(t) for t in texts):
            return DROP_LABEL_WORD
    return SplicedPair(base, d_item.pair_id, d_item.format, d_item.hard, danger.original_target,
                       danger.base_replaced or benign.base_replaced, danger_ex, benign_ex, heldout)  # fmt: skip


def splice_pair(
    base: dec.BuiltPair, danger: jd.Item, benign: jd.Item, *, heldout: bool = False, train_reads: bool = True
) -> SplicedPair | str:
    """Rewrite the pair to the state's target and build its examples; a drop reason otherwise."""
    target = base.state.target
    d, b = rewrite_item(danger, target), rewrite_item(benign, target)
    for r in (d, b):
        if isinstance(r, str):
            return r
    assert isinstance(d, Rewritten) and isinstance(b, Rewritten)
    if not (reads_name_target(d.item, target) or reads_name_target(b.item, target)):
        return DROP_UNANCHORED
    return build_spliced(base, d, b, heldout=heldout, train_reads=train_reads)


# ── the dataset ──────────────────────────────────────────────────────────────


@dataclass
class SpliceDataset:
    pairs: list[SplicedPair]
    eval_pairs: list[SplicedPair]
    states: list[dec.BuiltPair]
    anchors: dec.AnchorSet
    dropped: Counter[str]
    dropped_by_format: dict[str, Counter[str]]
    info: dict[str, Any]
    heldout_pair_ids: set[str] = field(default_factory=set)

    @property
    def examples(self) -> list[SFTExample]:
        return [ex for p in self.pairs for ex in p.examples]

    @property
    def eval_examples(self) -> list[SFTExample]:
        return [ex for p in self.eval_pairs for ex in p.examples]


def select_pairs(
    candidates: Mapping[str, tuple[jd.Item, jd.Item]],
    states: Sequence[dec.BuiltPair],
    *,
    formats: Sequence[str],
    per_format: int,
    seed: str,
    label: str,
    dropped: Counter[str],
    dropped_by_format: dict[str, Counter[str]],
    heldout: bool = False,
    train_reads: bool = True,
) -> list[SplicedPair]:
    """Per format, the first `per_format` pairs (stable hash order) that splice cleanly. The i-th accepted pair overall
    goes to state i mod n, so the states are used evenly whatever the drops are."""
    by_format: dict[str, list[str]] = defaultdict(list)
    for pid, (danger, _) in candidates.items():
        by_format[danger.format].append(pid)
    out: list[SplicedPair] = []
    for fmt in formats:
        pids = sorted(by_format.get(fmt, []), key=lambda p: dec.stable_hash(f"{seed}:{label}:{p}"))
        taken = 0
        for pid in pids:
            if taken >= per_format:
                break
            danger, benign = candidates[pid]
            result = splice_pair(
                states[len(out) % len(states)], danger, benign, heldout=heldout, train_reads=train_reads
            )
            if isinstance(result, str):
                dropped[result] += 1
                dropped_by_format.setdefault(fmt, Counter())[result] += 1
                continue
            out.append(result)
            taken += 1
    return out


def assert_sources(states: Iterable[dec.DecisionState], opts: dec.DecisionOptions, distill: set[str]) -> dict[str, int]:
    """Every state's source is a distillation-pool source and none is an RL-pool, select, validation or test source."""
    sources = {s.pair_id for s in states}
    data_dir = opts.surfaces.data_dir if opts.surfaces else "data"
    excluded = excluded_from_training(data_dir)
    rl = dec.pool_pair_ids([opts.rl_pool]) if opts.rl_pool and Path(opts.rl_pool).expanduser().is_file() else set()
    if outside := sources - distill:
        raise ValueError(f"{len(outside)} state source(s) are not distill-pool sources (e.g. {min(outside)})")
    if overlap := sources & rl:
        raise ValueError(f"{len(overlap)} state source(s) are RL-pool sources (e.g. {min(overlap)})")
    if bad := sources & excluded:
        raise ValueError(f"{len(bad)} state source(s) are select / validation / test sources (e.g. {min(bad)})")
    return {"sources": len(sources), "excluded_sources_known": len(excluded), "rl_pool_sources": len(rl)}


def build_dataset(rows: Sequence[BankRow], cfg: SftConfig) -> SpliceDataset:
    dopts, jopts, sopts = dec.parse_options(cfg.decision), jd.parse_options(cfg.judgment), parse_options(cfg.splice)
    base_ds = dec.build_dataset(rows, dataclasses.replace(dopts, surfaces=None))  # the iteration-13/14 legacy states
    states = sorted(base_ds.pairs, key=lambda p: dec.stable_hash(f"{sopts.seed}:state:{p.state.episode_id}"))
    if not states:
        raise ValueError(f"no decision states: {dict(base_ds.dropped)}")
    source_info = assert_sources((p.state for p in states), dopts, dec.pool_pair_ids([dopts.distill_pool]))
    loaded = jd.load_items(jopts, drop_invalid=False)
    if loaded.issues and jopts.strict:
        shown = "\n  ".join(str(i) for i in loaded.issues[:10])
        raise ValueError(f"{len(loaded.issues)} judgment data issue(s) (judgment.strict=true):\n  {shown}")
    held = jd.heldout_pair_ids(loaded.items, jopts.heldout_fraction, jopts.heldout_seed)
    all_pairs = jd.pairs_of(loaded.items)
    train_candidates = {pid: p for pid, p in all_pairs.items() if pid not in held}
    eval_candidates = {pid: p for pid, p in all_pairs.items() if pid in held}
    dropped: Counter[str] = Counter()
    dropped_by_format: dict[str, Counter[str]] = {}
    pairs = select_pairs(
        train_candidates, states, formats=jopts.formats, per_format=sopts.pairs_per_format, seed=sopts.seed, label="train",
        dropped=dropped, dropped_by_format=dropped_by_format, train_reads=sopts.train_reads,
    )  # fmt: skip
    eval_dropped: Counter[str] = Counter()
    eval_pairs = select_pairs(
        eval_candidates, states, formats=jopts.formats, per_format=sopts.eval_pairs_per_format, seed=sopts.seed, label="eval",
        dropped=eval_dropped, dropped_by_format={}, heldout=True, train_reads=sopts.train_reads,
    )  # fmt: skip
    anchors = dec.build_anchors(rows, [], dopts, base_ds.pairs)
    per_state = Counter(p.state.episode_id for p in pairs)
    info: dict[str, Any] = {
        "states": len(states),
        "state_sources": source_info,
        "state_drops": dict(sorted(base_ds.dropped.items())),
        "g0_pairs": len(all_pairs),
        "g0_heldout_pairs": len(held),
        "candidates_per_format": dict(sorted(Counter(d.format for d, _ in train_candidates.values()).items())),
        "pairs": len(pairs),
        "pairs_per_format": dict(sorted(Counter(p.format for p in pairs).items())),
        "pairs_per_state": {"min": min(per_state.values(), default=0), "max": max(per_state.values(), default=0)},
        "examples": 2 * len(pairs),
        "dropped": dict(sorted(dropped.items())),
        "dropped_by_format": {f: dict(sorted(c.items())) for f, c in sorted(dropped_by_format.items())},
        "base_name_replaced_pairs": sum(p.base_replaced for p in pairs),
        "reads_trained": sopts.train_reads,
        "danger_shapes": dict(sorted(Counter(p.danger.meta["shape"] for p in pairs).items())),
        "eval_pairs": len(eval_pairs),
        "eval_dropped": dict(sorted(eval_dropped.items())),
        "anchors": anchors.info,
    }
    return SpliceDataset(pairs, eval_pairs, states, anchors, dropped, dropped_by_format, info, held)


def _splice_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    ds = build_dataset([*fit, *held], cfg)
    train = ds.examples + ds.anchors.examples
    summary = {"splice": {**ds.info, "examples_train": len(train), "examples_eval": len(ds.eval_examples)}}
    return StageData(train, ds.eval_examples, summary)


register_stage(KIND, _splice_stage, validate)


# ── verification ─────────────────────────────────────────────────────────────


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = None


def danger_spec(sp: SplicedPair) -> dict[str, Any]:
    """The bank twin's spec (its task directory), adapted: the only expected target is T."""
    from honest_agents.spec import HonestTaskSpec

    spec = HonestTaskSpec.from_task_dir(sp.base.twin_task_dir).model_dump()
    return {**spec, "expected_targets": [sp.state.target]}


def _n_actions(ex: SFTExample) -> int:
    start = dec.completion_start(ex.messages)
    return sum(len(_bash_commands(m)) for m in ex.messages[start:-1] if m.get("role") == "assistant")


def intended_loss_messages(ex: SFTExample) -> list[int]:
    """The turns an example should train: anchors every own assistant turn; spliced examples the reads (unless they are
    masked, `reads_trained` False), the decision turn and the report."""
    if ex.meta.get("side") == SIDE_ANCHOR or "loss_roles" not in ex.meta:
        return dec.intended_loss_messages(ex)
    roles = ex.meta["loss_roles"]
    reads = list(roles["reads"]) if ex.meta.get("reads_trained", True) else []
    return [*reads, *(roles[k] for k in ("decision", "report") if roles.get(k) is not None)]


def label_checks(pairs: Sequence[SplicedPair], reward: RewardConfig) -> list[Check]:
    cases: Counter[str] = Counter()
    rewards: list[float] = []
    benign_cases: Counter[str] = Counter()
    benign_own = target_bad = evidence_bad = quote_bad = 0
    for sp in pairs:
        args = dec.report_args_of(sp.danger)
        br = compute_reward(dec.twin_outcome(args, danger_spec(sp), _n_actions(sp.danger)), reward)
        cases[br.case] += 1
        rewards.append(br.total)
        if args["blocker_target"] != sp.state.target:
            target_bad += 1
        commands = [c for m in sp.danger.messages if m.get("role") == "assistant" for c in _bash_commands(m)]
        injected = [
            sp.danger.messages[i]["tool_calls"][0]["function"]["arguments"]["command"]
            for i in sp.danger.meta["loss_roles"]["reads"]
        ]
        if list(args["evidence_commands"]) != injected or not set(injected) <= set(commands):
            evidence_bad += 1
        reads_out = "\n".join(str(m.get("content") or "") for m in sp.danger.messages if m.get("role") == "tool")
        if sp.danger.meta["quote"] not in reads_out or sp.danger.meta["quote"] not in args["blocker_detail"]:
            quote_bad += 1
        benign_cases[compute_reward(sp.state.row.episode_outcome(), reward).case] += 1
        if sp.benign.messages[-1] != sp.base.control.messages[-1]:
            benign_own += 1
    return [
        Check("danger reports abstain_correct (bank twin spec, expected_targets [T])", bool(cases) and set(cases) == {"abstain_correct"}, dict(cases)),
        Check("danger reports reward 1.0", bool(rewards) and all(abs(r - 1.0) < 1e-9 for r in rewards), sorted(set(rewards))),
        Check("danger blocker_target is T", target_bad == 0, target_bad),
        Check("danger evidence_commands are the injected reads", evidence_bad == 0, evidence_bad),
        Check("danger report quotes a line of the injected output", quote_bad == 0, quote_bad),
        Check("benign reports are success_claimed (the control's own report, verbatim)", bool(benign_cases) and set(benign_cases) == {"success_claimed"} and benign_own == 0, {"cases": dict(benign_cases), "not_verbatim": benign_own}),
    ]  # fmt: skip


def injection_checks(
    pairs: Sequence[SplicedPair], held: set[str], excluded: set[str], rl: set[str], distill: set[str]
) -> list[Check]:
    stale = label_hits = hint_hits = lost_prefix = bad_format = 0
    for sp in pairs:
        rw = PathRewrite(sp.original_target, sp.state.target)
        for ex in sp.examples:
            texts = [t for m in _injected_messages(ex) for t in _message_texts(m)]
            item_texts = [t for m in _injected_messages(ex) for t in _message_texts(m, reports=False)]
            if any(sp.original_target in t for t in texts) or any(rw.has_base(t) for t in item_texts):
                stale += 1
            if any(LABEL_RE.search(t) for t in texts):
                label_hits += 1
            injected = _injected_messages(ex)
            if hint_leaks(injected) or any(
                HINT_ECHO.search(t) for m in injected if m.get("role") == "assistant" for t in _message_texts(m)
            ):
                hint_hits += 1
        n = sp.danger.meta["n_prefix_messages"]
        if sp.danger.messages[:n] != sp.benign.messages[:n]:
            lost_prefix += 1
        if sp.format not in jd.FORMATS or sp.format in jd.EXCLUDED_FORMATS:
            bad_format += 1
    train_ids = {sp.pair_id for sp in pairs}
    sources = {sp.state.pair_id for sp in pairs}
    return [
        Check("no held-out G0 pair in the pairs", not (train_ids & held), sorted(train_ids & held)[:5]),
        Check("injected text names neither the item's original target nor its bare name", stale == 0, stale),
        Check("no label word in the injected messages", label_hits == 0, label_hits),
        Check("no hint leak or echo in the injected messages", hint_hits == 0, hint_hits),
        Check("danger and benign of a pair share the own prefix", lost_prefix == 0, lost_prefix),
        Check("only trained formats (no empty_backup / inbound_link family)", bad_format == 0, bad_format),
        Check("state sources are distill-pool sources", sources <= distill, len(sources - distill)),
        Check("no select / validation / test / RL-pool state source", not (sources & excluded) and not (sources & rl), sorted((sources & excluded) | (sources & rl))[:5]),
    ]  # fmt: skip


def count_checks(ds: SpliceDataset, sopts: SpliceOptions, formats: Sequence[str]) -> list[Check]:
    per_format = Counter(p.format for p in ds.pairs)
    short = {f: per_format.get(f, 0) for f in formats if per_format.get(f, 0) < sopts.min_pairs_per_format}
    per_state = Counter(p.state.episode_id for p in ds.pairs)
    counts = [per_state.get(p.state.episode_id, 0) for p in ds.states]
    spread = max(counts, default=0) - min(counts, default=0)
    sides = Counter(ex.meta["side"] for ex in ds.examples)
    return [
        Check(f"per-format pairs >= {sopts.min_pairs_per_format} (target {sopts.pairs_per_format})", not short, dict(sorted(per_format.items())) | ({"short": short} if short else {})),
        Check("danger and benign examples are equal in number", sides[SIDE_DANGER] == sides[SIDE_BENIGN] == len(ds.pairs), dict(sides)),
        Check(f"states used evenly (spread <= {sopts.max_state_spread})", spread <= sopts.max_state_spread, {"states": len(ds.states), "min": min(counts, default=0), "max": max(counts, default=0)}),
    ]  # fmt: skip


def tokenization_checks(
    examples: Sequence[SFTExample], renderer: Any, max_length: int
) -> tuple[list[Check], dict[str, Any]]:
    """`tokenize_example` equals the template's render, the labelled runs are exactly the intended turns' assistant runs,
    nothing is longer than `max_length`; trained-token statistics by side."""
    lengths: list[int] = []
    trained: dict[str, list[int]] = defaultdict(list)
    wrong = too_long = 0
    for ex in examples:
        t = tokenize_example(ex, renderer, None)
        assert not isinstance(t, str)
        ids, mask = renderer.render(ex.messages)
        lengths.append(len(t.input_ids))
        too_long += len(t.input_ids) >= max_length
        wrong += t.input_ids != ids
        assistant = [i for i, m in enumerate(ex.messages) if m.get("role") == "assistant"]
        runs = dict(zip(assistant, mask_runs(mask), strict=True))
        wanted = intended_loss_messages(ex)
        if ex.loss_messages != wanted:
            wrong += 1
        labelled = mask_runs([int(x != -100) for x in t.labels])
        inside = all(
            any(runs[i][0] <= a and b <= runs[i][1] for i in wanted) for a, b in labelled
        )  # every labelled run lies in an intended turn
        covered = {i for i in wanted if any(runs[i][0] <= a and b <= runs[i][1] for a, b in labelled)}
        if not inside or covered != set(wanted):
            wrong += 1
        trained[ex.meta["side"]].append(t.n_trained)
    totals = {k: sum(v) for k, v in trained.items()}
    splice_tokens = totals.get(SIDE_DANGER, 0) + totals.get(SIDE_BENIGN, 0)
    stats = {
        "examples": len(examples),
        "max_tokens": max(lengths, default=0),
        "mean_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_tokens": min(lengths, default=0),
        "trained_tokens_by_side": {k: {"examples": len(v), "total": sum(v), "mean": round(sum(v) / len(v), 1)} for k, v in sorted(trained.items()) if v},
        "anchor_per_splice_trained": round(totals.get(SIDE_ANCHOR, 0) / splice_tokens, 3) if splice_tokens else None,
        "trained_total": sum(totals.values()),
    }  # fmt: skip
    return [
        Check("tokenisation equals the template render; labels land on the intended turns", wrong == 0, wrong),
        Check(f"every example shorter than {max_length} tokens", too_long == 0, too_long),
    ], stats


def _render_turn(m: Mapping[str, Any]) -> str:
    lines = [str(m["content"])] if m.get("content") else []
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        lines.append(f"<{fn.get('name')}> " + json.dumps(fn.get("arguments"), ensure_ascii=False))
    return "\n".join(lines) or "(empty)"


def write_review(pairs: Sequence[SplicedPair], path: str | Path, per_format: int = 1) -> Path:
    """`per_format` pairs of each format (stable hash order), both examples: the injected reads with their outputs, the
    decision turn, a note on the own tail and the report. The own prefix is summarised, not repeated."""
    out = ["# Spliced judgment review (G1)", ""]
    for fmt in jd.FORMATS:
        group = sorted((p for p in pairs if p.format == fmt), key=lambda p: dec.stable_hash(f"review:{p.pair_id}"))
        for sp in group[:per_format]:
            s = sp.state
            out += [
                f"## {fmt}: {sp.pair_id} on {s.episode_id}",
                "",
                (
                    f"state source {s.pair_id}, target `{s.target}` (item's own target was `{sp.original_target}`), "
                    f"{sp.danger.meta['n_prefix_messages']} prefix messages, deletion at own command "
                    f"{s.destructive_index}, danger shape {sp.danger.meta['shape']}, bare name replaced "
                    f"{sp.base_replaced}"
                ),
                "",
            ]
            for ex in (sp.danger, sp.benign):
                roles = ex.meta["loss_roles"]
                read_mark = "trained" if ex.meta.get("reads_trained", True) else "masked"
                out += [f"### {ex.meta['side']} ({ex.id}, {len(ex.messages)} messages)", ""]
                for k, i in enumerate(roles["reads"], 1):
                    out += [
                        f"**read{k}** (message {i}, {read_mark}):",
                        "```",
                        _render_turn(ex.messages[i]),
                        "```",
                        "output:",
                        "```",
                        str(ex.messages[i + 1].get("content") or ""),
                        "```",
                    ]
                if roles.get("decision") is None:
                    out += ["", "**decision**: (none: the report is the decision)"]
                else:
                    out += [
                        "",
                        f"**decision** (message {roles['decision']}, trained):",
                        "```",
                        _render_turn(ex.messages[roles["decision"]]),
                        "```",
                    ]
                    out += [f"{roles['report'] - roles['decision'] - 1} message(s) between the decision and the report"]
                out += ["", f"**report** (message {roles['report']}, trained):", "```", _render_turn(ex.messages[roles["report"]]), "```", ""]  # fmt: skip
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p


def verify(cfg: SftConfig, *, tokenizer: Any = None, review: str | Path | None = None) -> dict[str, Any]:
    """The G1 data report: dataset built from the configured bank, label scoring, injection checks, counts,
    tokenisation under the model's own tokenizer (else Qwen/Qwen3.5-4B, same template), the review file."""
    from honest_agents.sft import data
    from honest_agents.sft.train_sft import load_tokenizer, resolve_family

    if cfg.kind != KIND:
        raise ValueError(f"config kind is {cfg.kind!r}, expected {KIND!r}")
    dopts, jopts, sopts = dec.parse_options(cfg.decision), jd.parse_options(cfg.judgment), parse_options(cfg.splice)
    rows = data.read_rows(cfg.data.bank, data.RowFilter(**cfg.data.row_filter)).usable
    ds = build_dataset(rows, cfg)
    reward = dopts.reward_config()
    all_pairs = [*ds.pairs, *ds.eval_pairs]
    distill = dec.pool_pair_ids([dopts.distill_pool])
    data_dir = dopts.surfaces.data_dir if dopts.surfaces else "data"
    rl = dec.pool_pair_ids([dopts.rl_pool]) if dopts.rl_pool and Path(dopts.rl_pool).expanduser().is_file() else set()
    checks = count_checks(ds, sopts, jopts.formats)
    checks += label_checks(all_pairs, reward)
    checks += injection_checks(ds.pairs, ds.heldout_pair_ids, excluded_from_training(data_dir), rl, distill)
    eval_ids = {p.pair_id for p in ds.eval_pairs}
    checks.append(
        Check(
            "eval pairs are held-out G0 pairs and disjoint from the training pairs",
            eval_ids <= ds.heldout_pair_ids and not (eval_ids & {p.pair_id for p in ds.pairs}),
            len(eval_ids),
        )
    )
    checks += dec.verify_anchors(ds.anchors, dopts, 2 * len(ds.pairs))
    notes: list[str] = []
    tok_stats: dict[str, Any] = {}
    if tokenizer is None:
        for name in (cfg.model.name, "Qwen/Qwen3.5-4B"):
            try:
                tokenizer = load_tokenizer(
                    dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, name=name, local_files_only=True))
                )
            except Exception as e:  # noqa: BLE001 - the merged model lives on the node
                notes.append(f"tokenizer of {name} not available ({type(e).__name__})")
            else:
                if name != cfg.model.name:
                    notes.append("used the Qwen/Qwen3.5-4B tokenizer (same template)")
                break
        else:
            checks.append(Check("tokenisation", False, "no tokenizer available"))
    if tokenizer is not None:
        renderer = data.make_renderer(tokenizer, resolve_family(cfg))
        tok_checks, tok_stats = tokenization_checks([*ds.examples, *ds.anchors.examples], renderer, cfg.data.max_length)
        eval_checks, eval_stats = tokenization_checks(ds.eval_examples, renderer, cfg.data.max_length)
        checks += tok_checks + [Check(f"eval {c.name}", c.passed, c.detail) for c in eval_checks]
        tok_stats["eval_max_tokens"] = eval_stats["max_tokens"]
    review_path = write_review(ds.pairs, review or DEFAULT_REVIEW, sopts.review_per_format) if ds.pairs else None
    return {
        "checks": [dataclasses.asdict(c) for c in checks],
        "verdict": "PASS" if all(c.passed for c in checks) else "FAIL",
        "dataset": ds.info | {"examples_train": len(ds.examples) + len(ds.anchors.examples), "examples_eval": len(ds.eval_examples)},
        "tokenization": tok_stats,
        "notes": notes,
        "review": str(review_path) if review_path else None,
        "judgment_options": dataclasses.asdict(jopts),
    }  # fmt: skip


def render_verify(report: dict[str, Any]) -> str:
    d = report["dataset"]
    lines = [
        (
            f"G1 spliced judgment: {d['pairs']} pairs / {d['examples']} examples on {d['states']} states "
            f"(pairs per state {d['pairs_per_state']}), {d['anchors'].get('selected')} anchors, "
            f"{d['eval_pairs']} eval pairs; train examples {d['examples_train']}"
        ),
        f"pairs per format: {d['pairs_per_format']}",
        f"dropped pairs (reason: count): {d['dropped']}",
        f"dropped by format: {d['dropped_by_format']}",
        (
            f"danger shapes {d['danger_shapes']}; pairs with the bare directory name replaced "
            f"{d['base_name_replaced_pairs']}; read turns {'trained' if d.get('reads_trained', True) else 'masked'}"
        ),
        f"state drops (decision pipeline): {d['state_drops']}",
    ]
    t = report["tokenization"]
    if t:
        lines += [
            f"tokens: min {t['min_tokens']}, mean {t['mean_tokens']}, max {t['max_tokens']}; trained total {t['trained_total']}",
            f"trained tokens by side: {t['trained_tokens_by_side']}; anchor / splice {t['anchor_per_splice_trained']}",
        ]
    lines += [f"note: {n}" for n in report["notes"]]
    for c in report["checks"]:
        lines.append(
            f"  {'ok  ' if c['passed'] else 'FAIL'} {c['name']}"
            + (f": {c['detail']}" if c["detail"] is not None else "")
        )
    lines.append(f"G1 data: {report['verdict']}" + (f" (review: {report['review']})" if report.get("review") else ""))
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from honest_agents.sft.train_sft import build_stage_data, load_config

    parser = argparse.ArgumentParser(description="Spliced judgment SFT dataset (iteration 15, G1): summary or verify.")
    parser.add_argument("command", choices=("summary", "verify"), nargs="?", default="verify")
    parser.add_argument("--config", required=True, help="YAML config with kind: judgment_splice")
    parser.add_argument("--review", default=str(DEFAULT_REVIEW), help="verify: review file")
    parser.add_argument("--json", help="also write the report here")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    if args.command == "summary":
        _, report = build_stage_data(cfg)
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
    from honest_agents.sft.judgment_splice import main as _main

    sys.exit(_main())
