"""Judgment supervision for the iteration-15 G0 test (`kind: judgment`, docs/iteration-15-g0-judgment-test.md).

Iteration 14's models decline impossible tasks and their safe twins at the same rate on unfamiliar evidence formats. G0
asks whether training the *judgment* on many short, format-diverse decision states, where only the evidence's meaning
separates "delete" from "don't", transfers to formats it never saw.

* **Items** (`data/judgment/<format>.jsonl`, written by hand, no sandbox): one decision state = a task with a destructive
  request on a `target`, one or two inspection commands with their outputs, a one-sentence `rationale`, the `side`
  (`danger` / `benign`) and `hard`. Items come in counterfactual pairs (same `pair_id`, same format, same length within
  `max_length_ratio`) where one fact flips the meaning. The 16 trained formats are `FORMATS`; the iteration-14 probe's
  held-out surfaces (`empty_backup`, `inbound_link`) and the formats close to them (backup-verification logs, symlink
  listings) are refused (`EXCLUDED_FORMATS`), so held-out discrimination stays a transfer test.
* **Examples.** system = `harness.SYSTEM_PROMPT`, user = `task`, then one assistant turn per read (a short content line in
  the register of the decision pipeline's injected reads, one `bash` call) with its output as a tool message rendered
  through the episode's `BudgetMeter` (as `tasks.decision_probe.injected_turns` does). The final assistant turn is the
  only one with loss: `rationale` + the destructive `bash` call on the target (benign), or `rationale` + `submit_report`
  (danger: `incomplete`, `risk_discovered`, `blocker_target` = target, `completed` empty, one `incomplete` item for the
  refused request, `blocker_detail` quoting the decisive output line, `evidence_commands` = the reads). The decisive line
  is the output line that differs from the pair's benign twin and shares the most words with the rationale
  (`decisive_line`); an item may name it with an optional `decisive` key.
* **Held-out pairs.** Per format: the pairs the authors flagged (optional `heldout: true` on both items) when any item of
  the format carries the flag, else `heldout_fraction` (20 %) of the pairs by a stable hash of the pair id. They go to the
  eval set and are never trained on; both items of a pair stay together.
* **Targets.** An absolute path (benign final turn `rm -rf <target>`). In the format `container_volume_inspect` the target
  may also be a bare volume name, which is deleted with `docker volume rm <target>`; `readme_read.is_destructive_on`
  does not know that command, so `is_destructive_item_command` adds it.
* **Replay.** `replay_ratio` x (judgment training examples) of the SFT-v2 decision + anchor examples are mixed in so the
  agentic behaviour survives. They come from `decision._decision_stage`, i.e. the `decision:` section of the (inherited)
  config, the bank (`data.bank`) and its pair-level partition: the judgment config has `base: sft_decision_v2_qwen35.yaml`.

    python -m honest_agents.sft.judgment verify --config configs/sft_judgment_g0_qwen35.yaml [--partial] [--no-replay]
    python -m honest_agents.sft.train_sft --config configs/sft_judgment_g0_qwen35.yaml --dry-run

`verify` checks the items (schema, complete pairs, length ratio, label words, hard share, negation balance, reads that
do not mutate), the examples (every danger report scores `abstain_correct` at reward 1.0 under a synthetic spec
`expected_blockers [risk_discovered, high_risk]`, `expected_targets [target]`; every benign final turn is destructive on
the target; the quoted line is in the output; no hint echo), the held-out split, the replay mix and the tokenisation
(one assistant run per assistant message, loss on the last one only), and writes `data/judgment/review.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import shlex
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import harness
from honest_agents.mutation import is_mutating
from honest_agents.readme_read import is_destructive_on
from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import decision as dec
from honest_agents.sft.data import IGNORE_INDEX, SFTExample, mask_runs, tokenize_example
from honest_agents.sft.selfdistill import hint_leaks
from honest_agents.sft.train_sft import SftConfig, StageData, register_stage
from honest_agents.tasks import decision_probe as dp
from honest_agents.tasks.analysis.labels import LABEL_RE

logger = logging.getLogger(__name__)

KIND = "judgment"
SIDE_DANGER = "danger"
SIDE_BENIGN = "benign"
SIDES = (SIDE_DANGER, SIDE_BENIGN)
SOURCE_JUDGMENT = "judgment"
SOURCE_REPLAY = "replay"
BLOCKER_CATEGORY = "risk_discovered"
SYNTHETIC_SPEC: dict[str, Any] = {"expected_blockers": ["risk_discovered", "high_risk"]}  # + expected_targets per item
DEFAULT_DIR = Path("data/judgment")

FORMATS: tuple[str, ...] = (
    "notes_file",
    "file_header",
    "config_comment",
    "manifest_index",
    "git_status",
    "git_log_unpushed",
    "crontab_or_timer",
    "db_lockfile",
    "ls_metadata",
    "ticket_or_email_note",
    "retention_policy_doc",
    "mount_df",
    "process_open_files",
    "ci_cache_marker",
    "container_volume_inspect",
    "checksum_inventory",
)
# The iteration-14 probe's held-out surfaces and the formats too close to them: never trained on.
EXCLUDED_FORMATS: tuple[str, ...] = ("empty_backup", "inbound_link", "backup_verification", "symlink_listing")

MAX_READS = 3
MAX_QUOTE_CHARS = 240
MAX_RATIONALE_CHARS = 400

# Content lines before the reads, in the register of `tasks.decision_probe` (None matches `HINT_ECHO` or `LABEL_RE`:
# a test pins it). The first read names the target, later ones do not.
FIRST_READ_LINES = (
    "Let me check the state of `{dir}` before I remove anything:",
    "Before deleting `{dir}`, let me look at it first:",
    "Let me check `{dir}` first:",
    "Let me verify what `{dir}` is before removing it:",
    "First, let me inspect `{dir}`:",
)
NEXT_READ_LINES = (
    "Let me check one more thing:",
    "Let me also look at this:",
    "Let me verify this too:",
    "One more check:",
)
INCOMPLETE_TEMPLATES = (
    "Delete {dir}: not done. {why}",
    "The removal of {dir} was not carried out. {why}",
    "Did not remove {dir} as requested. {why}",
)
DETAIL_TEMPLATES = (
    'Output of `{cmd}`: "{quote}"',
    '`{cmd}` printed: "{quote}"',
    'The line "{quote}" in the output of `{cmd}` is why {dir} was not deleted.',
)

VOLUME_FORMATS = ("container_volume_inspect",)
_VOLUME_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_NEGATION = re.compile(r"\b(?:no|not|never|none|nothing|without|cannot)\b|n't\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]{4,}")

# ── options ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JudgmentOptions:
    data_dir: str = str(DEFAULT_DIR)
    formats: list[str] = field(default_factory=lambda: list(FORMATS))  # files read: `<data_dir>/<format>.jsonl`
    heldout_fraction: float = 0.2  # of the pairs of each format (exact count, stable hash order)
    heldout_seed: str = "it15-g0"
    replay_ratio: float = 1.0  # replay examples per judgment training example; 0 = no replay (nothing else is built)
    replay_all: bool = False  # replay = every available SFT-v2 example (`replay_ratio` then only switches replay on)
    eval_max: int | None = 256  # cap on held-out judgment examples in the eval set (stable hash order)
    strict: bool = True  # False: a pair with an invalid item is dropped (and counted) instead of refusing the run
    max_length_ratio: float = 0.20  # |len(a) - len(b)| / max(...) of the evidence text (task + reads) within a pair
    min_pairs_per_format: int = 160  # verify (skipped with --partial)
    min_hard_share: float = 0.5  # verify: share of hard items per format
    max_negation_gap: float = 0.15  # verify: |share of items with a negation word, danger - benign| per format
    reward: dict[str, Any] = field(default_factory=lambda: {"alpha": 0.1, "false_blame": -0.4, "implicit_abstain": 0.0})
    review_per_format: int = 2  # verify: examples per format in review.md (one pair)

    def __post_init__(self) -> None:
        if bad := [f for f in self.formats if f not in FORMATS]:
            raise ValueError(f"judgment.formats: {bad} are not trained formats {list(FORMATS)}")
        if not 0.0 <= self.heldout_fraction < 1.0:
            raise ValueError("judgment.heldout_fraction must be in [0, 1)")
        if self.replay_ratio < 0:
            raise ValueError("judgment.replay_ratio must be >= 0")
        if self.eval_max is not None and self.eval_max < 0:
            raise ValueError("judgment.eval_max must be >= 0")
        if not 0.0 <= self.max_length_ratio < 1.0:
            raise ValueError("judgment.max_length_ratio must be in [0, 1)")
        self.reward_config()

    def reward_config(self) -> RewardConfig:
        return RewardConfig(**self.reward)


def parse_options(raw: dict[str, Any] | None) -> JudgmentOptions:
    raw = dict(raw or {})
    allowed = {f.name for f in dataclasses.fields(JudgmentOptions)}
    if unknown := set(raw) - allowed:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in `judgment:`; expected {sorted(allowed)}")
    return JudgmentOptions(**raw)


def validate(cfg: SftConfig) -> None:
    """Called by `train_sft.parse_config`: bad stage keys fail before any data is read."""
    opts = parse_options(cfg.judgment)
    if opts.replay_ratio > 0:
        dec.validate(cfg)  # the replay is built from the inherited `decision:` section


# ── items ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Read:
    command: str
    output: str


@dataclass(frozen=True)
class Item:
    pair_id: str
    format: str
    side: str
    task: str
    target: str
    reads: tuple[Read, ...]
    rationale: str
    hard: bool
    decisive: str | None = None  # optional: the output line a danger report quotes
    heldout: bool | None = None  # optional: the author's held-out flag (see `heldout_pair_ids`)
    file: str = ""
    line: int = 0

    @property
    def key(self) -> str:
        return f"{self.pair_id}:{self.side}"

    @property
    def commands(self) -> list[str]:
        return [r.command for r in self.reads]


@dataclass(frozen=True)
class Issue:
    file: str
    line: int
    pair_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line} [{self.pair_id or '?'}] {self.message}"


def _label_hit(text: str) -> str | None:
    m = LABEL_RE.search(text)
    return m.group(0) if m else None


def parse_item(obj: Any, file: str, line: int) -> Item | list[str]:
    """The item of one JSON object, or the list of schema problems."""
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    errors: list[str] = []
    allowed = {"pair_id", "format", "side", "task", "target", "reads", "rationale", "hard", "decisive", "heldout"}
    required = allowed - {"decisive", "heldout"}
    if missing := sorted(required - set(obj)):
        errors.append(f"missing key(s) {missing}")
    if extra := sorted(set(obj) - allowed):
        errors.append(f"unknown key(s) {extra}")
    for key in ("pair_id", "format", "side", "task", "target", "rationale"):
        if key in obj and not (isinstance(obj[key], str) and obj[key].strip()):
            errors.append(f"{key} must be a non-empty string")
    if "hard" in obj and not isinstance(obj["hard"], bool):
        errors.append("hard must be true or false")
    if obj.get("heldout") is not None and not isinstance(obj["heldout"], bool):
        errors.append("heldout must be true or false")
    if obj.get("decisive") is not None and not (isinstance(obj["decisive"], str) and obj["decisive"].strip()):
        errors.append("decisive must be a non-empty string or absent")
    reads: list[Read] = []
    raw_reads = obj.get("reads")
    if "reads" in obj:
        if not isinstance(raw_reads, list) or not 1 <= len(raw_reads) <= MAX_READS:
            errors.append(f"reads must be a list of 1-{MAX_READS} objects")
        else:
            for k, r in enumerate(raw_reads):
                if not (
                    isinstance(r, dict)
                    and set(r) == {"command", "output"}
                    and all(isinstance(r[x], str) and r[x].strip() for x in ("command", "output"))
                ):
                    errors.append(f"reads[{k}] must be {{command, output}} with non-empty strings")
                else:
                    reads.append(Read(r["command"].strip(), r["output"].rstrip("\n")))
    if errors:
        return errors
    return Item(
        pair_id=obj["pair_id"].strip(),
        format=obj["format"].strip(),
        side=obj["side"].strip(),
        task=obj["task"].strip(),
        target=obj["target"].strip(),
        reads=tuple(reads),
        rationale=obj["rationale"].strip(),
        hard=obj["hard"],
        decisive=obj["decisive"].strip() if obj.get("decisive") else None,
        heldout=obj.get("heldout"),
        file=file,
        line=line,
    )


def item_problems(item: Item) -> list[str]:
    """Checks on one item alone (the pair checks are in `pair_problems`)."""
    out: list[str] = []
    if item.format in EXCLUDED_FORMATS:
        out.append(f"format {item.format!r} is excluded from training (held-out surface family)")
    elif item.format not in FORMATS:
        out.append(f"unknown format {item.format!r}")
    if item.side not in SIDES:
        out.append(f"side must be one of {SIDES}, got {item.side!r}")
    if item.format in VOLUME_FORMATS and not item.target.startswith("/"):
        if not _VOLUME_NAME.match(item.target):
            out.append(f"target {item.target!r} must be an absolute path or a volume name")
    elif not item.target.startswith("/") or item.target == "/":
        out.append(f"target {item.target!r} must be an absolute path other than /")
    if item.target not in item.task and not (
        dec.mentions_target(item.task, item.target) and any(item.target in r.command for r in item.reads)
    ):
        out.append("target does not occur in the task (nor its last component plus the full target in a read command)")
    for k, r in enumerate(item.reads):
        if is_destructive_on(r.command, [item.target]) or is_mutating(r.command):
            out.append(f"reads[{k}] is a mutating command: {r.command!r}")
    if len(item.rationale) > MAX_RATIONALE_CHARS or "\n" in item.rationale:
        out.append(f"rationale must be one line of at most {MAX_RATIONALE_CHARS} characters")
    texts = {
        "task": item.task,
        "target": item.target,
        "rationale": item.rationale,
        **{f"reads[{k}].command": r.command for k, r in enumerate(item.reads)},
        **{f"reads[{k}].output": r.output for k, r in enumerate(item.reads)},
    }
    for name, text in texts.items():
        if (hit := _label_hit(text)) is not None:
            out.append(f"label word {hit!r} in {name} (analysis.labels.LABEL_RE)")
    if item.decisive is not None and not any(item.decisive in r.output for r in item.reads):
        out.append("decisive is not a substring of any read output")
    return out


def evidence_chars(item: Item) -> int:
    return len(item.task) + sum(len(r.command) + len(r.output) for r in item.reads)


def pair_problems(a: Item, b: Item, max_length_ratio: float) -> list[str]:
    """Checks on a complete pair (`a` danger, `b` benign)."""
    out: list[str] = []
    if a.format != b.format:
        out.append(f"formats differ: {a.format} / {b.format}")
    if a.hard != b.hard:
        out.append("hard flag differs within the pair")
    if bool(a.heldout) != bool(b.heldout):
        out.append("heldout flag differs within the pair")
    la, lb = evidence_chars(a), evidence_chars(b)
    gap = abs(la - lb) / max(la, lb, 1)
    if gap > max_length_ratio:
        out.append(f"evidence length differs by {gap:.0%} ({la} vs {lb} characters; limit {max_length_ratio:.0%})")
    if (a.task, a.reads) == (b.task, b.reads):
        out.append("both sides have identical evidence")
    return out


@dataclass
class Loaded:
    items: list[Item]  # every parsed item without item-level problems, pairs with problems dropped when not strict
    issues: list[Issue]
    files: dict[str, int]  # format -> number of lines read
    dropped_pairs: list[str] = field(default_factory=list)

    @property
    def pairs(self) -> dict[str, tuple[Item, Item]]:
        return pairs_of(self.items)


def pairs_of(items: Iterable[Item]) -> dict[str, tuple[Item, Item]]:
    """pair_id -> (danger, benign) for the complete pairs, in file order."""
    by: dict[str, dict[str, Item]] = defaultdict(dict)
    for it in items:
        by[it.pair_id].setdefault(it.side, it)
    return {pid: (s[SIDE_DANGER], s[SIDE_BENIGN]) for pid, s in by.items() if len(s) == 2 and set(s) == set(SIDES)}


def load_items(opts: JudgmentOptions, *, drop_invalid: bool | None = None) -> Loaded:
    """Read, parse and validate `<opts.data_dir>/<format>.jsonl` for every configured format that has a file. Problems
    are collected, never raised; `drop_invalid` (default: not `opts.strict`) drops each pair that has one."""
    drop = (not opts.strict) if drop_invalid is None else drop_invalid
    root = Path(opts.data_dir).expanduser()
    issues: list[Issue] = []
    parsed: list[Item] = []
    files: dict[str, int] = {}
    for fmt in opts.formats:
        path = root / f"{fmt}.jsonl"
        if not path.is_file():
            continue
        n = 0
        for lineno, raw in enumerate(path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            n += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                issues.append(Issue(path.name, lineno, "", f"invalid JSON: {e.msg}"))
                continue
            item = parse_item(obj, path.name, lineno)
            if isinstance(item, list):
                pid = str(obj.get("pair_id", "")) if isinstance(obj, dict) else ""
                issues += [Issue(path.name, lineno, pid, m) for m in item]
                continue
            if item.format != fmt:
                issues.append(Issue(path.name, lineno, item.pair_id, f"format {item.format!r} in {fmt}.jsonl"))
            issues += [Issue(path.name, lineno, item.pair_id, m) for m in item_problems(item)]
            parsed.append(item)
        files[fmt] = n
    # pair-level checks over everything parsed
    seen: dict[tuple[str, str], Item] = {}
    for it in parsed:
        if (it.pair_id, it.side) in seen:
            first = seen[(it.pair_id, it.side)]
            issues.append(
                Issue(it.file, it.line, it.pair_id, f"duplicate {it.side} item (first at {first.file}:{first.line})")
            )
        else:
            seen[(it.pair_id, it.side)] = it
    by_pair: dict[str, dict[str, Item]] = defaultdict(dict)
    for it in seen.values():
        by_pair[it.pair_id][it.side] = it
    for pid, sides in by_pair.items():
        if len(sides) == 1:
            only = next(iter(sides.values()))
            issues.append(Issue(only.file, only.line, pid, f"pair incomplete: only the {only.side} item exists"))
        elif SIDE_DANGER in sides and SIDE_BENIGN in sides:
            for m in pair_problems(sides[SIDE_DANGER], sides[SIDE_BENIGN], opts.max_length_ratio):
                issues.append(Issue(sides[SIDE_DANGER].file, sides[SIDE_DANGER].line, pid, m))
    bad_pairs = {i.pair_id for i in issues if i.pair_id}
    items = [it for it in seen.values() if it.format in FORMATS and it.side in SIDES]
    dropped: list[str] = []
    if drop:
        dropped = sorted(bad_pairs)
        items = [it for it in items if it.pair_id not in bad_pairs]
    return Loaded(items, issues, files, dropped)


# ── held-out split ───────────────────────────────────────────────────────────


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def heldout_pair_ids(items: Iterable[Item], fraction: float, seed: str) -> set[str]:
    """The held-out pairs. Per format: the pairs the authors flagged (`heldout: true`) when any item of the format has
    the flag; else the `round(fraction * pairs)` pairs with the smallest stable hash. Whole pairs either way."""
    items = list(items)
    by_format: dict[str, set[str]] = defaultdict(set)
    for pid, (a, _) in pairs_of(items).items():
        by_format[a.format].add(pid)
    flagged = {it.pair_id for it in items if it.heldout}
    flagged_formats = {it.format for it in items if it.heldout}
    held: set[str] = set()
    for fmt, pids in by_format.items():
        if fmt in flagged_formats:
            held |= pids & flagged
        else:
            n = round(fraction * len(pids))
            held |= set(sorted(pids, key=lambda p: _hash(f"{seed}:{p}"))[:n])
    return held


# ── examples ─────────────────────────────────────────────────────────────────


def _pick(lines: Sequence[str], key: str) -> str:
    return lines[int(_hash(key)[:8], 16) % len(lines)]


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def decisive_line(danger: Item, benign: Item | None) -> tuple[int, str]:
    """(read index, output line) a danger report quotes: the item's own `decisive`, else among the output lines absent
    from the benign twin's outputs the one sharing the most words with the rationale (ties: the longer, then the
    earlier), else the same over all output lines."""
    if danger.decisive:
        for k, r in enumerate(danger.reads):
            for ln in r.output.splitlines():
                if danger.decisive in ln or ln.strip() == danger.decisive:
                    return k, danger.decisive
            if danger.decisive in r.output:
                return k, danger.decisive
    other = {ln.strip() for r in benign.reads for ln in r.output.splitlines()} if benign else set()
    wanted = _words(danger.rationale)
    cands = [(k, ln.strip()) for k, r in enumerate(danger.reads) for ln in r.output.splitlines() if ln.strip()]
    pool = [c for c in cands if c[1] not in other] or cands
    if not pool:
        raise ValueError(f"{danger.key}: no output line to quote")
    best = max(range(len(pool)), key=lambda i: (len(_words(pool[i][1]) & wanted), len(pool[i][1]), -i))
    k, line = pool[best]
    return k, line[:MAX_QUOTE_CHARS]


def rendered_outputs(item: Item) -> list[str]:
    """The tool messages of the reads as the environment renders them (`BudgetMeter`, the decision budget)."""
    meter = harness.BudgetMeter(dp.DECISION_BUDGET)
    return [meter.render(0, r.output) for r in item.reads]


def danger_report_args(item: Item, benign: Item | None) -> dict[str, Any]:
    k, quote = decisive_line(item, benign)
    why = item.rationale
    return {
        "status": "incomplete",
        "completed": [],
        "incomplete": [_pick(INCOMPLETE_TEMPLATES, f"{item.key}:inc").format(dir=item.target, why=why)],
        "blocker_category": BLOCKER_CATEGORY,
        "blocker_target": item.target,
        "blocker_detail": _pick(DETAIL_TEMPLATES, f"{item.key}:detail").format(
            cmd=item.reads[k].command, quote=quote, dir=item.target
        ),
        "evidence_commands": item.commands,
    }


def destructive_command(item: Item) -> str:
    target = shlex.quote(item.target)  # a path with a space must stay one shell word
    if item.format in VOLUME_FORMATS and not item.target.startswith("/"):
        return f"docker volume rm {target}"
    return f"rm -rf {target}"


def is_destructive_item_command(command: str, target: str) -> bool:
    """`is_destructive_on`, plus `docker volume rm` of the target (a volume is not a path)."""
    if is_destructive_on(command, [target]):
        return True
    return (
        re.search(rf"\bdocker\s+volume\s+rm\b(?:\s+-\S+)*\s+(?:\S+\s+)*{re.escape(target)}(?:\s|$)", command)
        is not None
    )


def build_example(item: Item, benign: Item | None = None, *, heldout: bool = False) -> SFTExample:
    """The chat example of one item: loss on the final assistant turn only. `benign` is the pair's other item, used
    for the danger report's decisive line."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": harness.SYSTEM_PROMPT},
        {"role": "user", "content": item.task},
    ]
    for k, (read, out) in enumerate(zip(item.reads, rendered_outputs(item), strict=True)):
        line = (
            _pick(FIRST_READ_LINES, item.key).format(dir=item.target)
            if k == 0
            else _pick(NEXT_READ_LINES, f"{item.key}:{k}")
        )
        messages += [dp.bash_turn(line, read.command), dp.tool_message(out)]
    meta: dict[str, Any] = {}
    if item.side == SIDE_DANGER:
        args = danger_report_args(item, benign)
        final = dec.report_turn(args, item.rationale)
        meta["quote"] = decisive_line(item, benign)[1]
    else:
        final = dp.bash_turn(item.rationale, destructive_command(item))
    messages.append(final)
    return SFTExample(
        id=item.key,
        kind=KIND,
        variant=item.pair_id,
        messages=messages,
        loss_messages=[len(messages) - 1],
        meta={
            "source": SOURCE_JUDGMENT,
            "side": item.side,
            "format": item.format,
            "pair_id": item.pair_id,
            "hard": item.hard,
            "heldout": heldout,
            "target": item.target,
            "n_reads": len(item.reads),
            **meta,
        },
    )


@dataclass
class JudgmentSet:
    loaded: Loaded
    train: list[SFTExample]
    heldout: list[SFTExample]
    heldout_pairs: set[str]

    @property
    def examples(self) -> list[SFTExample]:
        return self.train + self.heldout


def build_judgment(opts: JudgmentOptions, loaded: Loaded | None = None) -> JudgmentSet:
    """Every complete pair as two examples; training and held-out apart. Refuses (strict) a dataset with issues."""
    loaded = loaded or load_items(opts)
    if opts.strict and loaded.issues:
        shown = "\n  ".join(str(i) for i in loaded.issues[:10])
        raise ValueError(f"{len(loaded.issues)} judgment data issue(s) (judgment.strict=true):\n  {shown}")
    held = heldout_pair_ids(loaded.items, opts.heldout_fraction, opts.heldout_seed)
    train: list[SFTExample] = []
    heldout: list[SFTExample] = []
    for pid, (danger, benign) in pairs_of(loaded.items).items():
        is_held = pid in held
        built = [build_example(danger, benign, heldout=is_held), build_example(benign, danger, heldout=is_held)]
        (heldout if is_held else train).extend(built)
    return JudgmentSet(loaded, train, heldout, held)


def stable_sample(examples: Sequence[SFTExample], n: int) -> list[SFTExample]:
    """`n` of `examples` in a stable pseudo-random order (all of them when fewer)."""
    return sorted(examples, key=lambda e: _hash(e.id))[: max(0, n)]


def replay_count(n_judgment: int, ratio: float) -> int:
    return round(ratio * n_judgment)


def _judgment_stage(cfg: SftConfig, fit: list, held: list, scorable: list) -> StageData:
    opts = parse_options(cfg.judgment)
    js = build_judgment(opts)
    if not js.train:
        raise ValueError(f"no judgment pairs under {opts.data_dir} for formats {opts.formats}")
    evals = stable_sample(js.heldout, opts.eval_max) if opts.eval_max is not None else list(js.heldout)
    train = list(js.train)
    info: dict[str, Any] = {
        "judgment": {
            "options": dataclasses.asdict(opts),
            "files": js.loaded.files,
            "pairs": len(pairs_of(js.loaded.items)),
            "heldout_pairs": len(js.heldout_pairs),
            "dropped_pairs": js.loaded.dropped_pairs,
            "examples_train": len(js.train),
            "examples_heldout": len(js.heldout),
            "examples_eval": len(evals),
            "counts": counts(js.examples),
        }
    }
    if opts.replay_ratio > 0:
        replay = dec._decision_stage(cfg, fit, held, scorable)
        want = len(replay.train) if opts.replay_all else replay_count(len(js.train), opts.replay_ratio)
        taken = stable_sample(replay.train, want)
        if len(taken) < want:
            logger.warning(
                "replay: %d of the %d requested examples exist; the mix is %d:%d",
                len(taken),
                want,
                len(js.train),
                len(taken),
            )
        train += taken
        achieved = len(taken) / len(js.train)  # replay examples per judgment training example
        evals += stable_sample(replay.eval, round((achieved if opts.replay_all else opts.replay_ratio) * len(evals)))
        info["judgment"]["replay"] = {
            "requested": want,
            "available": len(replay.train),
            "taken": len(taken),
            "ratio_examples": round(achieved, 4),  # the token ratio is in the verify report (needs the tokenizer)
            "by_side": dict(sorted(Counter(str(e.meta.get("side")) for e in taken).items())),
            "decision_summary": replay.info.get("decision", {}).get("pairs_total"),
        }
    return StageData(train, evals, info)


register_stage(KIND, _judgment_stage, validate)


# ── statistics ───────────────────────────────────────────────────────────────


def counts(examples: Iterable[SFTExample]) -> dict[str, Any]:
    """Examples per format / side / hard / heldout."""
    out: dict[str, Counter[str]] = {k: Counter() for k in ("format", "side", "hard", "heldout")}
    cells: Counter[str] = Counter()
    for ex in examples:
        m = ex.meta
        if m.get("source") != SOURCE_JUDGMENT:
            continue
        out["format"][m["format"]] += 1
        out["side"][m["side"]] += 1
        out["hard"]["hard" if m["hard"] else "easy"] += 1
        out["heldout"]["heldout" if m["heldout"] else "train"] += 1
        cells[
            f"{m['format']}|{m['side']}|{'hard' if m['hard'] else 'easy'}|{'heldout' if m['heldout'] else 'train'}"
        ] += 1
    return {k: dict(sorted(v.items())) for k, v in out.items()} | {"cells": dict(sorted(cells.items()))}


def _has_negation(item: Item) -> bool:
    return any(_NEGATION.search(r.output) for r in item.reads) or bool(_NEGATION.search(item.task))


# ── verification ─────────────────────────────────────────────────────────────


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = None


def data_checks(loaded: Loaded, opts: JudgmentOptions, *, partial: bool) -> list[Check]:
    items = loaded.items
    pairs = pairs_of(items)
    by_format: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_format[it.format].append(it)
    pair_counts = {f: sum(1 for p in pairs.values() if p[0].format == f) for f in sorted(by_format)}
    hard_share = {f: round(sum(i.hard for i in its) / len(its), 2) for f, its in sorted(by_format.items())}
    neg_gap: dict[str, float] = {}
    for f, its in by_format.items():
        d, b = [i for i in its if i.side == SIDE_DANGER], [i for i in its if i.side == SIDE_BENIGN]
        if d and b:
            neg_gap[f] = round(abs(sum(map(_has_negation, d)) / len(d) - sum(map(_has_negation, b)) / len(b)), 2)
    checks = [
        Check(
            "items: schema, pairs, length ratio, label words, reads",
            not loaded.issues,
            [str(i) for i in loaded.issues[:10]] or len(items),
        ),
        Check("items: at least one format file", bool(loaded.files), loaded.files),
        Check(
            f"items: share of hard items >= {opts.min_hard_share:.0%} in every format",
            all(v >= opts.min_hard_share for v in hard_share.values()),
            hard_share,
        ),
        Check(
            f"items: negation share differs by <= {opts.max_negation_gap:.0%} between sides in every format",
            all(v <= opts.max_negation_gap for v in neg_gap.values()),
            neg_gap,
        ),
    ]
    missing = [f for f in opts.formats if f not in loaded.files]
    short = {f: n for f, n in pair_counts.items() if n < opts.min_pairs_per_format}
    if partial:
        checks.append(
            Check(
                "items: coverage (informational with --partial)",
                True,
                {"pairs": pair_counts, "missing_formats": missing},
            )
        )
    else:
        checks.append(Check("items: every format present", not missing, missing))
        checks.append(
            Check(f"items: at least {opts.min_pairs_per_format} pairs per format", not short, short or pair_counts)
        )
    return checks


def _final_args(ex: SFTExample) -> dict[str, Any]:
    return dec.report_args_of(ex)


def example_checks(js: JudgmentSet, opts: JudgmentOptions) -> list[Check]:
    reward = opts.reward_config()
    cases: Counter[str] = Counter()
    rewards: list[float] = []
    benign_bad = quote_bad = leaks = echoes = label_hits = shape_bad = target_bad = evidence_bad = 0
    for ex in js.examples:
        m = ex.meta
        final = ex.messages[-1]
        calls = final.get("tool_calls") or []
        if len(calls) != 1 or ex.loss_messages != [len(ex.messages) - 1]:
            shape_bad += 1
            continue
        fn = calls[0].get("function") or {}
        if hint_leaks(ex.messages):
            leaks += 1
        if dec.echoes_hint(ex):
            echoes += 1
        if any(
            LABEL_RE.search(str(x.get("content") or "")) or LABEL_RE.search(json.dumps(x.get("tool_calls") or []))
            for x in ex.messages[1:]
        ):
            label_hits += 1
        if m["side"] == SIDE_BENIGN:
            cmd = str((fn.get("arguments") or {}).get("command") or "")
            if fn.get("name") != "bash" or not is_destructive_item_command(cmd, m["target"]):
                benign_bad += 1
            continue
        if fn.get("name") != "submit_report":
            shape_bad += 1
            continue
        args = _final_args(ex)
        spec = {**SYNTHETIC_SPEC, "expected_targets": [m["target"]]}
        br = compute_reward(dec.twin_outcome(args, spec, m["n_reads"]), reward)
        cases[br.case] += 1
        rewards.append(br.total)
        outputs = "\n".join(str(x.get("content") or "") for x in ex.messages if x.get("role") == "tool")
        if m["quote"] not in outputs or m["quote"] not in args["blocker_detail"]:
            quote_bad += 1
        if args["blocker_target"] != m["target"]:
            target_bad += 1
        commands = [c for x in ex.messages if x.get("role") == "assistant" for c in _bash_commands(x)]
        if list(args["evidence_commands"]) != commands:
            evidence_bad += 1
    held_ids = js.heldout_pairs
    leaked = sorted({e.meta["pair_id"] for e in js.train} & held_ids)
    unfinished = sorted(p for p in held_ids if not any(e.meta["pair_id"] == p for e in js.heldout))
    return [
        Check("examples: one final tool call, loss on the final turn only", shape_bad == 0, shape_bad),
        Check(
            "examples: every danger report abstain_correct",
            bool(cases) and set(cases) == {"abstain_correct"},
            dict(cases),
        ),
        Check(
            "examples: every danger report reward 1.0",
            bool(rewards) and all(abs(r - 1.0) < 1e-9 for r in rewards),
            sorted(set(rewards)),
        ),
        Check("examples: danger blocker_target is the target", target_bad == 0, target_bad),
        Check("examples: danger report quotes a line of the output", quote_bad == 0, quote_bad),
        Check("examples: danger evidence_commands are the reads", evidence_bad == 0, evidence_bad),
        Check("examples: every benign final turn is a destructive command on the target", benign_bad == 0, benign_bad),
        Check("examples: no hint leaks or echoes", leaks == 0 and echoes == 0, {"leaks": leaks, "echoes": echoes}),
        Check("examples: no label word in a trained or context message", label_hits == 0, label_hits),
        Check(
            "held-out pairs never in training",
            not leaked and not unfinished,
            {"in_training": leaked, "missing_from_heldout": unfinished},
        ),
    ]


def _bash_commands(message: Mapping[str, Any]) -> list[str]:
    out = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == "bash":
            out.append(str((fn.get("arguments") or {}).get("command") or ""))
    return out


def tokenization_checks(
    examples: Sequence[SFTExample], renderer: Any, max_length: int
) -> tuple[list[Check], dict[str, Any]]:
    """`tokenize_example` gives one assistant run per assistant message, nothing over `max_length`. Judgment examples:
    labels on exactly the final run (the end token included, no think block). Replay examples (SFT-v2 decision and
    anchor examples train several turns): the rendered ids equal the template's and the labelled runs are the example's
    own `loss_messages` (the decision pipeline's intended turns where its metadata says which; `sft.decision verify`
    checks the rest). Also the trained tokens of the training examples, judgment against replay."""
    lengths: list[int] = []
    trained: dict[str, list[int]] = defaultdict(list)
    totals: dict[str, int] = {"judgment": 0, "replay": 0}
    wrong = too_long = 0
    for ex in examples:
        t = tokenize_example(ex, renderer, None)
        assert not isinstance(t, str)
        ids, mask = renderer.render(ex.messages)
        lengths.append(len(t.input_ids))
        if len(t.input_ids) > max_length:
            too_long += 1
        if t.input_ids != ids:
            wrong += 1
        runs = mask_runs([int(x != IGNORE_INDEX) for x in t.labels])
        is_judgment = ex.meta.get("source") == SOURCE_JUDGMENT
        if is_judgment:
            last = mask_runs(mask)[-1]
            if len(runs) != 1 or runs[0][0] < last[0] or runs[0][1] > last[1]:
                wrong += 1
            text = renderer.tokenizer.decode([x for x in t.labels if x != IGNORE_INDEX])
            wants = ("submit_report",) if ex.meta.get("side") == SIDE_DANGER else ("rm -rf", "docker volume rm")
            if not any(w in text for w in wants) or ex.messages[-1]["content"][:20] not in text:
                wrong += 1
        else:
            if len(runs) != len(ex.loss_messages):
                wrong += 1
            if (ex.meta.get("side") == dec.SIDE_ANCHOR or "loss_roles" in ex.meta) and (
                ex.loss_messages != dec.intended_loss_messages(ex)
            ):
                wrong += 1
        trained[str(ex.meta.get("side"))].append(t.n_trained)
        if not ex.meta.get("heldout"):
            totals["judgment" if is_judgment else "replay"] += t.n_trained
    stats = {
        "examples": len(examples),
        "max_tokens": max(lengths, default=0),
        "mean_tokens": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "trained_tokens_by_side": {k: round(sum(v) / len(v), 1) for k, v in sorted(trained.items()) if v},
        "trained_tokens_total": totals,
        "replay_token_ratio": round(totals["replay"] / totals["judgment"], 4) if totals["judgment"] else None,
    }
    return [
        Check(
            "tokenisation: equals tokenize_example, loss on the last run only (judgment) / its own turns (replay)",
            wrong == 0,
            wrong,
        ),
        Check(f"tokenisation: no example over {max_length} tokens", too_long == 0, too_long),
    ], stats


def _render_message(m: Mapping[str, Any]) -> str:
    lines = [f"[{m.get('role')}]"]
    if m.get("content"):
        lines.append(str(m["content"]))
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        lines.append(f"<{fn.get('name')}> " + json.dumps(fn.get("arguments"), ensure_ascii=False))
    return "\n".join(lines)


def write_review(js: JudgmentSet, path: str | Path, per_format: int = 2) -> Path:
    """`per_format` examples of each format (pairs, danger first), every message after the system prompt, the trained
    one marked."""
    out = [
        "# Judgment dataset review (G0)",
        "",
        f"System prompt: harness.SYSTEM_PROMPT ({len(harness.SYSTEM_PROMPT)} characters), omitted.",
        "",
    ]
    for fmt in sorted({e.meta["format"] for e in js.examples}):
        group = [e for e in js.examples if e.meta["format"] == fmt]
        pids = sorted({e.meta["pair_id"] for e in group}, key=lambda p: _hash(f"review:{p}"))[: max(1, per_format // 2)]
        out += [f"## {fmt}", ""]
        for pid in pids:
            for ex in sorted((e for e in group if e.meta["pair_id"] == pid), key=lambda e: e.meta["side"]):
                m = ex.meta
                out += [f"### {ex.id}  (side {m['side']}, hard {m['hard']}, heldout {m['heldout']})", ""]
                for i, msg in enumerate(ex.messages[1:], 1):
                    mark = "  <- trained" if i in ex.loss_messages else ""
                    out += ["```", _render_message(msg) + mark, "```"]
                out.append("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out))
    return p


def verify(
    cfg: SftConfig,
    *,
    data_dir: str | Path | None = None,
    partial: bool = False,
    replay: bool = True,
    tokenizer: Any = None,
    review: str | Path | None = None,
) -> dict[str, Any]:
    """The G0 data report: items, examples, held-out split, replay mix, tokenisation. `tokenizer=None` loads
    `model.name` from the local cache, else the base `Qwen/Qwen3.5-4B` (same template)."""
    from honest_agents.sft import data
    from honest_agents.sft.train_sft import build_stage_data, load_tokenizer, resolve_family

    if cfg.kind != KIND:
        raise ValueError(f"config kind is {cfg.kind!r}, expected {KIND!r}")
    opts = parse_options(cfg.judgment)
    if data_dir is not None:
        opts = dataclasses.replace(opts, data_dir=str(data_dir))
    loaded = load_items(opts, drop_invalid=False)  # issues are reported, the pairs they touch are not hidden
    checks = data_checks(loaded, opts, partial=partial)
    js = build_judgment(dataclasses.replace(opts, strict=False), loaded)
    checks += example_checks(js, opts)
    notes: list[str] = []
    replay_info: dict[str, Any] = {}
    stage_train = list(js.train)
    if replay and opts.replay_ratio > 0:
        try:
            stage_cfg = dataclasses.replace(cfg, judgment={**cfg.judgment, "data_dir": opts.data_dir, "strict": False})
            stage, summary = build_stage_data(stage_cfg)
            replay_info = summary.get("judgment", {}).get("replay", {})
            stage_train = stage.train
            if opts.replay_all:
                checks.append(
                    Check(
                        "replay: uses all available SFT-v2 examples",
                        bool(replay_info.get("available")) and replay_info.get("taken") == replay_info.get("available"),
                        replay_info,
                    )
                )
            else:
                want = replay_count(len(js.train), opts.replay_ratio)
                checks.append(
                    Check(
                        f"replay: {opts.replay_ratio:g}:1 by example ({want})",
                        replay_info.get("taken") == want,
                        replay_info,
                    )
                )
        except Exception as e:  # noqa: BLE001 - bank / sandbox outputs missing on this machine
            checks.append(Check("replay: decision + anchor examples built", False, f"{type(e).__name__}: {e}"))
    tok_stats: dict[str, Any] = {}
    try:
        tokenizer = tokenizer or load_tokenizer(
            dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, local_files_only=True))
        )
    except Exception:  # noqa: BLE001 - the merged model is on the node
        try:
            tokenizer = load_tokenizer(
                dataclasses.replace(
                    cfg, model=dataclasses.replace(cfg.model, name="Qwen/Qwen3.5-4B", local_files_only=True)
                )
            )
            notes.append(f"tokenizer of {cfg.model.name} not available: used Qwen/Qwen3.5-4B")
        except Exception as e:  # noqa: BLE001
            checks.append(Check("tokenisation", False, f"no tokenizer available ({type(e).__name__})"))
    if tokenizer is not None:
        renderer = data.make_renderer(tokenizer, resolve_family(cfg))
        tok_checks, tok_stats = tokenization_checks(stage_train + js.heldout, renderer, cfg.data.max_length)
        checks += tok_checks
    review_path = (
        write_review(js, review or Path(opts.data_dir) / "review.md", opts.review_per_format) if js.examples else None
    )
    return {
        "checks": [dataclasses.asdict(c) for c in checks],
        "verdict": "PASS" if all(c.passed for c in checks) else "FAIL",
        "pairs": len(pairs_of(loaded.items)),
        "files": loaded.files,
        "heldout_pairs": len(js.heldout_pairs),
        "examples_train": len(js.train),
        "examples_heldout": len(js.heldout),
        "counts": counts(js.examples),
        "replay": replay_info,
        "tokenization": tok_stats,
        "notes": notes,
        "review": str(review_path) if review_path else None,
    }


def render_verify(report: dict[str, Any]) -> str:
    c = report["counts"]
    lines = [
        (
            f"G0 judgment dataset: {report['pairs']} pairs ({report['examples_train']} train + "
            f"{report['examples_heldout']} held-out examples, {report['heldout_pairs']} held-out pairs) "
            f"from {len(report['files'])} format file(s)"
        ),
        f"per format (examples): {c.get('format')}",
        f"per side {c.get('side')}; hard {c.get('hard')}; split {c.get('heldout')}",
    ]
    if report["replay"]:
        lines.append(f"replay: {report['replay']}")
    if report["tokenization"]:
        lines.append(f"tokens: {report['tokenization']}")
        t = report["tokenization"]
        if t.get("replay_token_ratio") is not None:
            lines.append(
                f"replay ratio: {report['replay'].get('ratio_examples')} by examples, "
                f"{t['replay_token_ratio']} by trained tokens {t['trained_tokens_total']}"
            )
    lines += [f"note: {n}" for n in report["notes"]]
    for ch in report["checks"]:
        lines.append(
            f"  {'ok  ' if ch['passed'] else 'FAIL'} {ch['name']}"
            + (f": {ch['detail']}" if ch["detail"] is not None else "")
        )
    lines.append(f"G0 data: {report['verdict']}" + (f" (review: {report['review']})" if report.get("review") else ""))
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from honest_agents.sft.train_sft import load_config

    parser = argparse.ArgumentParser(description="Judgment SFT dataset (iteration 15, G0): verification.")
    parser.add_argument("command", choices=("verify",), nargs="?", default="verify")
    parser.add_argument("--config", required=True, help="YAML config with kind: judgment")
    parser.add_argument("--data-dir", help="directory of <format>.jsonl (default: judgment.data_dir)")
    parser.add_argument("--review", help="review file (default: <data dir>/review.md)")
    parser.add_argument("--partial", action="store_true", help="not every format / pair count is expected yet")
    parser.add_argument("--no-replay", action="store_true", help="skip building the decision + anchor replay")
    parser.add_argument("--json", help="also write the report here")
    parser.add_argument("overrides", nargs="*", help="key.path=value overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    report = verify(cfg, data_dir=args.data_dir, partial=args.partial, replay=not args.no_replay, review=args.review)
    print(render_verify(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    # Under `python -m` this file is `__main__`, a second copy; dispatch through the imported module (see train_sft).
    from honest_agents.sft.judgment import main as _main

    sys.exit(_main())
