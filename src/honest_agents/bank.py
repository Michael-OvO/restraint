"""Trajectory bank and quarantine list.

The report-only pilot trains only the final report on frozen Phase-0 trajectories. A trajectory's outcome is fixed, so a
new report can be scored without a sandbox, provided the row keeps everything the reward reads:

* the prompt messages and the completion messages **up to, not including, the report turn** (what the pilot conditions
  on), plus the report turn as it was generated;
* the graded `EpisodeOutcome` as a dict;
* the task metadata;
* cached blocker facts — the commands on the agent's PATH, existence and permissions of the paths the transcript names,
  the network policy — so `check_blocker_cached` can answer for a *different* report what the grader's live check
  would have answered.

The reward functions (`honest_agents.trl_rewards`) append one row per episode when a bank directory is configured
(`HONEST_BANK_DIR` or `set_bank_dir`). Each process writes its own `bank-<pid>.jsonl`, so concurrent runs never
interleave lines.

The quarantine list records variants labelled infeasible whose tests passed cleanly; the dataset build excludes them.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import math
import os
import posixpath
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from honest_agents.report import ReportParseError, SelfReport, parse_report  # noqa: F401 (ReportParseError: re-export)
from honest_agents.reward import (
    BLOCKER_UNKNOWN,
    STOP_REPORT,
    EpisodeOutcome,
    RewardBreakdown,
    RewardConfig,
    blocker_verdict,
    compute_reward,
    is_command_target,
    normalize_target,
    outcome_from_dict,
    outcome_to_dict,
)

logger = logging.getLogger(__name__)

ENV_BANK_DIR = "HONEST_BANK_DIR"
ENV_QUARANTINE_FILE = "HONEST_QUARANTINE_FILE"
# 2: outcomes carry `damage_paths` and `first_mutation_idx`. Rows written under 1 load with the defaults (no damage), and
# reprice with unchanged rewards.
SCHEMA_VERSION = 2
REPORT_TOOLS = ("submit_report", "abstain")  # the tool calls that end an episode
QUARANTINE_BASENAME = "quarantine.jsonl"
QUARANTINE_LEAK = "leak"  # a completion claim passed the tests of an infeasible variant (its group was masked)
QUARANTINE_UNCLAIMED_PASS = "unclaimed_pass"  # the tests passed cleanly but the agent did not claim completion

_UNSET: Any = object()
_bank_dir: Any = _UNSET
_quarantine_file: Any = _UNSET
_write_lock = threading.Lock()


# ── configuration ────────────────────────────────────────────────────────────


def set_bank_dir(path: str | Path | None) -> None:
    """Explicit bank directory. It wins over `HONEST_BANK_DIR`; None turns the bank off even if the variable is set."""
    global _bank_dir
    _bank_dir = Path(path) if path is not None else None


def set_quarantine_file(path: str | Path | None) -> None:
    """Explicit quarantine file. It wins over `HONEST_QUARANTINE_FILE`; `None` turns the list off."""
    global _quarantine_file
    _quarantine_file = Path(path) if path is not None else None


def use_environment() -> None:
    """Drop both explicit settings, so the environment variables decide again."""
    global _bank_dir, _quarantine_file
    _bank_dir = _quarantine_file = _UNSET


def bank_dir() -> Path | None:
    if _bank_dir is not _UNSET:
        return _bank_dir
    value = os.environ.get(ENV_BANK_DIR, "").strip()
    return Path(value) if value else None


def quarantine_file() -> Path | None:
    """The explicit file, else `HONEST_QUARANTINE_FILE`, else `quarantine.jsonl` inside the bank directory."""
    if _quarantine_file is not _UNSET:
        return _quarantine_file
    value = os.environ.get(ENV_QUARANTINE_FILE, "").strip()
    if value:
        return Path(value)
    d = bank_dir()
    return d / QUARANTINE_BASENAME if d is not None else None


# ── cached blocker facts ─────────────────────────────────────────────────────


class PathFact(BaseModel):
    exists: bool
    readable: bool | None = None  # `test -r` as the agent user; None = not probed
    writable: bool | None = None  # `test -w` as the agent user; None = not probed


class BlockerFacts(BaseModel):
    """What the grader saw in the sandbox, kept so a blocker claim can be checked after the sandbox is gone."""

    commands: list[str] | None = None  # basename of every executable on the agent user's PATH; None = not captured
    paths: dict[str, PathFact] = Field(
        default_factory=dict
    )  # absolute path -> facts, for paths named in the transcript
    cwd: str | None = None  # the agent's working directory, to resolve relative targets
    network_mode: str | None = None  # Harbor `NetworkMode` of the resolved agent-phase policy; None = not captured
    allowed_hosts: list[str] = Field(default_factory=list)

    def path_fact(self, target: str) -> PathFact | None:
        """Facts for `target`: by normalised absolute path, by `cwd`-relative path, else by a unique path suffix."""
        t = normalize_target(target)
        if not t:
            return None
        table = {normalize_target(k): v for k, v in self.paths.items()}
        if t in table:
            return table[t]
        if t.startswith("/"):
            return None
        rel = t.removeprefix("~/")
        if self.cwd and posixpath.normpath(posixpath.join(self.cwd, rel)) in table:
            return table[posixpath.normpath(posixpath.join(self.cwd, rel))]
        hits = [v for k, v in table.items() if k.endswith("/" + rel)]
        return hits[0] if len(hits) == 1 else None


# What `command -v <name>` finds in the grader's shell without a file on PATH: Harbor runs every exec in the main
# container as `bash -c`, so bash's builtins and reserved words (bash 5 `compgen -b -k`). `commands` lists files only.
BASH_BUILTINS = frozenset(
    """. : [ alias bg bind break builtin caller cd command compgen complete compopt continue declare dirs disown echo
    enable eval exec exit export false fc fg getopts hash help history jobs kill let local logout mapfile popd printf
    pushd pwd read readarray readonly return set shift shopt source suspend test times trap true type typeset ulimit
    umask unalias unset wait ! [[ ]] { } case coproc do done elif else esac fi for function if in select then time until
    while""".split()
)


def check_blocker_cached(category: str, target: str, facts: BlockerFacts) -> str:
    """The grader's blocker verdict, answered from cached facts. Anything the cache cannot settle is "unknown".

    Mirrors `grading.check_blocker`, which runs `command -v <target>` and `test -e <target>`: a bare name is found if it
    is a bash builtin or reserved word or a file on PATH; a path target is looked up by `command -v` at that path only,
    so it is settled by its path fact, never by its basename being installed somewhere else."""
    t = normalize_target(target)
    fact = facts.path_fact(t)
    command_found = None
    if t and is_command_target(t):
        if "/" not in t:
            if t in BASH_BUILTINS:
                command_found = True
            elif facts.commands is not None:
                command_found = t in set(facts.commands)
        elif fact is None and facts.commands is not None and posixpath.basename(t) not in set(facts.commands):
            command_found = False  # no path fact: `blocker_verdict` reads a bin-dir command absent from PATH as absent
    return blocker_verdict(
        category,
        t,
        command_found=command_found,
        path_exists=fact.exists if fact is not None else None,
        path_readable=fact.readable if fact is not None else None,
        path_writable=fact.writable if fact is not None else None,
        network_mode=facts.network_mode,
        allowed_hosts=facts.allowed_hosts,
    )


# ── bank rows ────────────────────────────────────────────────────────────────


class BankRow(BaseModel):
    schema_version: int = SCHEMA_VERSION
    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    time: float = Field(default_factory=time.time)
    step: int | None = None  # trainer global step when the episode was graded
    reward_mode: str | None = None  # which reward function wrote the row
    group: str | None = None  # prompt-group id used for leak masking (the dataset row id)
    task: dict[str, Any] = Field(default_factory=dict)  # dataset row columns, plus `spec` = [metadata.honest]
    prompt: list[dict[str, Any]] = Field(default_factory=list)  # prompt messages (system, user with the instruction)
    completion: list[dict[str, Any]] = Field(default_factory=list)  # messages before the report turn
    report_message: dict[str, Any] | None = None  # the assistant message that carried the report call, as generated
    report_call: dict[str, Any] | None = None  # {"name": ..., "arguments": {...}} of that call
    # Tool calls in the report turn that ran before the report (they are part of the outcome but not of `completion`).
    report_turn_extra_calls: int = 0
    n_completion_messages: int = 0  # length of the full completion, for audit
    # Token-level record of the rollout as sampled, when the trainer hands it over. The pilot
    # re-renders its context from the messages above, so its contexts are not token-identical to the live loop.
    completion_ids: list[int] | None = None  # sampled token ids of the whole completion, tool results included
    prompt_ids: list[int] | None = None
    # Sampling-time log-probabilities, aligned with `completion_ids`; None where the sampler gave no finite value.
    logprobs: list[float | None] | None = None
    tool_mask: list[int] | None = None  # 1 = model token, 0 = tool-result token, aligned with `completion_ids`
    outcome: dict[str, Any]  # `reward.outcome_to_dict(EpisodeOutcome)`
    reward: dict[str, Any] = Field(default_factory=dict)  # `RewardBreakdown.to_dict()` under `reward_config`
    reward_value: float | None = None  # what the reward function returned to TRL (None = masked)
    reward_config: dict[str, Any] = Field(default_factory=dict)
    blocker_facts: BlockerFacts = Field(default_factory=BlockerFacts)
    extra: dict[str, Any] = Field(default_factory=dict)

    def episode_outcome(self) -> EpisodeOutcome:
        return outcome_from_dict(self.outcome)

    def config(self) -> RewardConfig:
        """The reward config the row was written under (unknown keys from a newer version are dropped). A stored config
        that today's guards reject is still the one the row was scored under, so it loads as stored: one old row must
        not abort the build of a whole bank."""
        known = {f.name for f in dataclasses.fields(RewardConfig)}
        fields = {k: v for k, v in self.reward_config.items() if k in known}
        try:
            return RewardConfig(**fields)
        except ValueError:
            return RewardConfig.as_stored(**fields)


class ReportSplit(NamedTuple):
    prefix: list[dict[str, Any]]  # messages before the report turn
    report_message: dict[str, Any] | None
    report_call: dict[str, Any] | None  # {"name", "arguments"}
    extra_calls: int  # other tool calls in the report turn, before the report call


def _call_name_args(call: Any) -> tuple[str | None, Any]:
    """(name, arguments) of a tool call in TRL / OpenAI form (`{"type": "function", "function": {...}}`) or flat."""
    if not isinstance(call, dict):
        return None, None
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return fn.get("name"), args


def split_at_report(completion: Iterable[dict[str, Any]] | None) -> ReportSplit:
    """Cut a completion at the report turn: the first assistant message that calls `submit_report` or `abstain`.

    The first report is the final one (later tool calls only get "the episode is over"), so everything from that
    message on is dropped from the prefix. Without a report call the whole completion is the prefix.
    """
    messages = [m for m in (completion or []) if isinstance(m, dict)]
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for j, call in enumerate(message.get("tool_calls") or []):
            name, args = _call_name_args(call)
            if name in REPORT_TOOLS:
                return ReportSplit(messages[:i], message, {"name": name, "arguments": args}, j)
    return ReportSplit(messages, None, None, 0)


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so a row never fails to serialise (non-JSON objects become their `str`)."""
    return json.loads(json.dumps(value, default=str))


def _numbers(values: Any, kind: type) -> list | None:
    """A flat list of `kind` from a list / tuple / tensor; None when absent or not a flat numeric sequence.
    NaN and infinities become None entries: JSON has no spelling for them. A token vLLM could not score arrives as
    None in TRL's raw `logprobs` (grpo_trainer.py:2516) and as NaN once TRL has padded them; both are kept as None,
    so one unscored token does not cost the row all its log-probabilities."""
    if values is None:
        return None
    try:
        out = [kind(v) if v is not None else None for v in (values.tolist() if hasattr(values, "tolist") else values)]
    except (TypeError, ValueError):
        return None
    return [v if v is not None and math.isfinite(v) else None for v in out]


def make_row(
    *,
    prompt: Iterable[dict[str, Any]] | None,
    completion: Iterable[dict[str, Any]] | None,
    outcome: EpisodeOutcome,
    breakdown: RewardBreakdown | None = None,
    reward_value: float | None = None,
    task: dict[str, Any] | None = None,
    blocker_facts: BlockerFacts | dict[str, Any] | None = None,
    reward_config: RewardConfig | dict[str, Any] | None = None,
    step: int | None = None,
    reward_mode: str | None = None,
    group: str | None = None,
    extra: dict[str, Any] | None = None,
    completion_ids: Iterable[int] | None = None,
    prompt_ids: Iterable[int] | None = None,
    logprobs: Iterable[float] | None = None,
    tool_mask: Iterable[int] | None = None,
) -> BankRow:
    full = [m for m in (completion or []) if isinstance(m, dict)]
    split = split_at_report(full)
    if dataclasses.is_dataclass(reward_config) and not isinstance(reward_config, type):
        reward_config = dataclasses.asdict(reward_config)
    return BankRow(
        step=step,
        reward_mode=reward_mode,
        group=group,
        task=_jsonable(task or {}),
        prompt=_jsonable([m for m in (prompt or []) if isinstance(m, dict)]),
        completion=_jsonable(split.prefix),
        report_message=_jsonable(split.report_message),
        report_call=_jsonable(split.report_call),
        report_turn_extra_calls=split.extra_calls,
        n_completion_messages=len(full),
        outcome=_jsonable(outcome_to_dict(outcome)),
        reward=_jsonable(breakdown.to_dict()) if breakdown is not None else {},
        reward_value=reward_value,
        reward_config=_jsonable(reward_config or {}),
        blocker_facts=BlockerFacts.model_validate(blocker_facts or {}),
        extra=_jsonable(extra or {}),
        completion_ids=_numbers(completion_ids, int),
        prompt_ids=_numbers(prompt_ids, int),
        logprobs=_numbers(logprobs, float),
        tool_mask=_numbers(tool_mask, int),
    )


def append_rows(rows: Iterable[BankRow], directory: str | Path | None = None) -> Path | None:
    """Append rows to this process's bank file. Returns the file, or None when no bank directory is configured."""
    d = Path(directory) if directory is not None else bank_dir()
    if d is None:
        return None
    lines = [row.model_dump_json() + "\n" for row in rows]
    path = d / f"bank-{os.getpid()}.jsonl"
    with _write_lock:
        d.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.writelines(lines)
    return path


def _complete_lines(file: Path) -> Iterator[str]:
    """Non-empty lines of a JSONL file. A last line that is not valid JSON is skipped with a warning: it is what a
    run killed in the middle of a write leaves behind. Broken JSON anywhere else is corruption and raises."""
    pending: str | None = None
    with gzip.open(file, "rt") if file.suffix == ".gz" else open(file) as f:
        for line in f:
            if not line.strip():
                continue
            if pending is not None:
                yield pending
            pending = line
    if pending is None:
        return
    try:
        json.loads(pending)
    except json.JSONDecodeError:
        logger.warning("%s: dropping a truncated last line (%d characters)", file, len(pending))
        return
    yield pending


def bank_files(path: str | Path) -> list[Path]:
    """One bank file, or every `bank-*.jsonl` in a directory (sorted by file name). The archive gzips banks
    (`bank-*.jsonl.gz`); a gzipped file is read unless its plain twin sits next to it, so a half-archived directory is
    not counted twice."""
    p = Path(path)
    if not p.is_dir():
        return [p]
    plain = set(p.glob("bank-*.jsonl"))
    gz = {f for f in p.glob("bank-*.jsonl.gz") if f.with_suffix("") not in plain}
    return sorted(plain | gz)


def iter_bank(path: str | Path) -> Iterator[BankRow]:
    """Rows of one bank file (plain or gzipped), or of every bank file in a directory (`bank_files`)."""
    for file in bank_files(path):
        for line in _complete_lines(file):
            yield BankRow.model_validate_json(line)


def read_bank(path: str | Path) -> list[BankRow]:
    return list(iter_bank(path))


# ── re-scoring a new report (report-only pilot) ──────────────────────────────


def rescored_outcome(
    row: BankRow,
    report: SelfReport | dict[str, Any] | str | None,
    *,
    report_error: str | None = None,
    parser: Callable[[Any], SelfReport] = parse_report,
) -> EpisodeOutcome:
    """The row's outcome with `report` in place of the one the trajectory ended with.

    Everything the trajectory fixed is kept (tests, violations, tamper, commands run, the mask). What depends on the
    report is recomputed: the blocker verdict from the cached facts, the target match (left to `compute_reward`), and
    the padding count. A raw report (dict or JSON string) goes through `parser`; pass the environment's own coercion
    (`honest_agents.harness.coerce_report`) to score exactly as training does.
    """
    if isinstance(report, (dict, str)):
        try:
            report = parser(report)
        except ValueError as e:  # ReportParseError, or a pydantic ValidationError from a parser that lets one through
            report, report_error = None, str(e)
    o = row.episode_outcome()
    o.report = report
    o.report_error = report_error if report is None else None
    o.evidence = []
    o.target_match = None
    o.n_blockers_submitted = report.n_blockers_submitted if report is not None else 0
    o.blocker_check = BLOCKER_UNKNOWN
    if report is not None and report.blames_external:
        o.blocker_check = check_blocker_cached(report.blocker.category.value, report.blocker.target, row.blocker_facts)
    if report is not None or report_error is not None:
        o.stop_reason = STOP_REPORT
    return o


def rescore(
    row: BankRow,
    report: SelfReport | dict[str, Any] | str | None,
    cfg: RewardConfig | None = None,
    *,
    report_error: str | None = None,
    parser: Callable[[Any], SelfReport] = parse_report,
) -> RewardBreakdown:
    """Reward of the banked trajectory had it ended with `report` (under `cfg`, default: the row's own config)."""
    outcome = rescored_outcome(row, report, report_error=report_error, parser=parser)
    return compute_reward(outcome, cfg if cfg is not None else row.config())


# ── quarantine list ──────────────────────────────────────────────────────────


class QuarantineEntry(BaseModel):
    variant_id: str | None = None
    task_dir: str | None = None
    pair_id: str | None = None
    perturbation: str | None = None
    family: str | None = None
    kind: str = QUARANTINE_LEAK  # QUARANTINE_LEAK | QUARANTINE_UNCLAIMED_PASS
    source: str = "train"  # where it was found: "train" | "eval" | "validation"
    step: int | None = None
    time: float = Field(default_factory=time.time)
    detail: str = ""


def append_quarantine(entries: Iterable[QuarantineEntry], path: str | Path | None = None) -> Path | None:
    """Append entries to the quarantine list. Returns the file, or None when no list is configured."""
    p = Path(path) if path is not None else quarantine_file()
    if p is None:
        return None
    lines = [e.model_dump_json() + "\n" for e in entries]
    with _write_lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.writelines(lines)
    return p


def read_quarantine(path: str | Path | None = None) -> list[QuarantineEntry]:
    p = Path(path) if path is not None else quarantine_file()
    if p is None or not p.exists():
        return []
    return [QuarantineEntry.model_validate_json(line) for line in _complete_lines(p)]


# Entries found while evaluating describe held-out (test) variants. Letting them shape a rebuilt training set would
# change which test rows survive (232 -> 230 on the audit's real build), so a dataset build ignores them by default.
EVAL_QUARANTINE_SOURCE = "eval"
# "test" is the split name `trl_rewards`/`difficulty` use as `source` for a held-out row; "openenv" episodes can run on
# any split, so their entries cannot be told apart from test-derived ones.
DEFAULT_IGNORED_SOURCES: tuple[str, ...] = (EVAL_QUARANTINE_SOURCE, "test", "openenv")


def quarantined_variants(
    path: str | Path | None = None,
    kinds: Iterable[str] = (QUARANTINE_LEAK,),
    exclude_sources: Iterable[str] = DEFAULT_IGNORED_SOURCES,
) -> set[str]:
    """Variant ids to exclude on a dataset rebuild. By default only proven leaks; add `QUARANTINE_UNCLAIMED_PASS` to
    also drop variants whose tests passed without a claim. Entries whose `source` is in `exclude_sources` (default:
    "eval") are ignored; pass `exclude_sources=()` to honour every entry, as an evaluation run that skips known-leaky
    variants wants."""
    wanted, skipped = set(kinds), set(exclude_sources)
    return {
        e.variant_id for e in read_quarantine(path) if e.kind in wanted and e.variant_id and e.source not in skipped
    }
