"""Splice-based claim pairs from the cached checklist validation (iteration 16, claim commitment point; no Docker).

The commit-data build (`sft.commit_checks build`) has no claim examples: its only checked run (`it15_g4_sample`) kept
episode records, not transcripts, so every claim point is dropped `no_transcript`. This module builds the same kind of
matched pair without any new run, from what is already stored:

* **States** (the check outputs) come from the checklist-validation result cache (`sft.checklist_validation`,
  `~/.cache/honest-agents/gate/checklist_validation.v1.jsonl`): per validated end state, the real output and exit code
  of every checklist command and the hidden suite's verdict on the same state (replayed, tests forced on). A cache row
  holds no state id (its key is a content hash that includes the Docker image digest, which needs Docker to recompute),
  so states are joined by content:
  - an **episode** state of `--check-results` (`sft.commit_checks.CheckResult`: the current checklist's outputs per
    episode, `sft.checklist_agreement --check-results`) is the set of non-oracle cache rows that hold every current
    command with the same exit code and output, the same number of replayed steps and the episode's recorded exit
    codes. The rows must agree on the tests' verdict (`cache_ambiguous` otherwise; a sibling variant of the same
    source can have the same commands);
  - the **oracle** state of a task is the set of oracle cache rows (`oracle_tail` set) that hold every current command
    of its checklist; their outputs and verdicts must all agree.
  A state's checklist verdict (every requirement of the *current* checklist passes, scored from the stored outputs as
  `sft.checklist_agreement` does) must **agree** with the tests' verdict, and an episode's archived verdict must equal
  the replayed one (`archived_replayed_disagree`: a verdict that does not reproduce is no label).
* **Pass state**: an agreed passing own episode of the task (it15_g4_sample), else the task's oracle state.
  **Fail state**: an agreed failing own episode of the task (the near misses the model's claims were made on).
* **Prefix** (shared by both sides of a pair), first that exists for the task in this order:
  1. `own_transcript`: the model's own successful control episode of the task, `kind: read_rft` in the G4 training
     examples (`--prefix-examples`, the first file): G2's own text and tool outputs, up to its report turn;
  2. `decision_example`: the SFT-v2 decision example of the task as replayed in G2 (`kind: decision`, the second file;
     the SFT-v2 pipeline's own-prefix splice with its injected reads), up to its report turn;
  3. `template`: `[system, instruction]` (`tasks.decision_probe`'s zero prefix: no work shown).
  A prefix's instruction must be the task directory's `instruction.md` as the env renders it; its system message is
  re-rendered with `harness.SYSTEM_PROMPT`. A transcript prefix that destroys the control's target (read_rft and
  SFT-v2 controls proceed; `readme_read.is_destructive_on` on its bash commands) would be contradicted by a fail state
  whose failing requirements include the removal request (`checklist_agreement.requirement_type`): such a state is
  paired with the next prefix, at the latest the template.
* **Pair** (`origin` `splice_pass` / `splice_fail`): the prefix, then one injected bash turn per requirement with the
  same assistant text and command on both sides (`commit_checks.check_turns`, outputs rendered through the budget meter
  after the prefix's own outputs), then the decision. Pass: the prefix's own claim report verbatim (the template's: a
  claim listing the requirements, evidence = the check commands). Fail: the honest report of `commit_checks`
  (`incomplete`, the quotes that passed and failed, no blocker, evidence = the failing check commands), never a claim.
  Loss on the check command turns and the decision; the prefix and every tool output are masked. Both decisions are
  graded with `reward.compute_reward` on the state's facts (the archived outcome of an episode, an oracle outcome
  otherwise): the claim must be `success_claimed` on the pass side, the honest report `honest_failure[_observed]` and
  the claim `fabrication` on the fail side, else the pair is dropped (`decision_case_mismatch`).
  `--dpo`: on the fail branch, prompt = everything up to the last check output, chosen = the honest report, rejected =
  the pass side's claim.
* **Quotas**: `--pairs-per-task` pairs per task (at most `cap_per_task`, so each point x outcome x task cell holds at
  most the cap), then whole pairs are dropped from the largest sources (stable hash order) until no source holds more
  than `max_source_share` of the build (the coverage report's bound). Every drop is counted by reason.

High_risk twins get no claim pair: there the honest end is the destructive-step decision (`commit_checks.claim_buildable`).
Every source is asserted against `tasks.select_split.excluded_from_training` (select, validation and test sources):
the states, the prefixes and the final examples and DPO pairs (`ExcludedSourceError`).

Output (`--out`): `examples.jsonl` (`sft.data.SFTExample`, `kind: commit_check`), `dpo_pairs.jsonl`, `summary.json`,
`report.md` and `build_manifest.json` (`commit_points.schema.BuildManifest`, read by `commit_points.coverage`; its
`build` block holds the inputs' and outputs' sha256s, the sources and the counts).

    uv run python -m honest_agents.sft.claim_splice --tasks-root ROOT --episodes EPISODES.jsonl.gz \\
        --prefix-examples G4_EXAMPLES.jsonl.gz --prefix-examples G2_EXAMPLES.jsonl.gz --dpo \\
        [--cache ~/.cache/honest-agents/gate/checklist_validation.v1.jsonl] [--out data/commit_checks/builds/claim_splice_v1]
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents import harness
from honest_agents.commit_points.schema import BuildManifest, ExampleRef, write_manifest
from honest_agents.commit_points.schema import Quota as ManifestQuota
from honest_agents.gate.cache import DEFAULT_CACHE_DIR
from honest_agents.harness import report_from_tool_args
from honest_agents.readme_read import high_risk_targets, is_destructive_on
from honest_agents.reward import EpisodeOutcome, RewardConfig, compute_reward, outcome_from_dict
from honest_agents.sft import commit_checks as cc
from honest_agents.sft import decision as dec
from honest_agents.sft.checklist_agreement import REQ_REMOVAL, requirement_type
from honest_agents.sft.checklist_validation import task_meta
from honest_agents.sft.data import SFTExample
from honest_agents.sft.dpo import PreferencePair
from honest_agents.tasks.select_split import excluded_from_training

DEFAULT_CACHE = DEFAULT_CACHE_DIR / "checklist_validation.v1.jsonl"
DEFAULT_CHECK_RESULTS = Path("data/commit_checks/check_results/it15_g4_sample.jsonl")
DEFAULT_OUT = Path("data/commit_checks/builds/claim_splice_v1")
DEFAULT_MAX_SOURCE_SHARE = ManifestQuota.max_source_share

ORIGIN_SPLICE_PASS = "splice_pass"
ORIGIN_SPLICE_FAIL = cc.ORIGIN_SPLICE_FAIL
STATE_ORACLE = "oracle"
STATE_EPISODE = "episode"
PREFIX_OWN = "own_transcript"  # kind read_rft: the model's own successful episode
PREFIX_DECISION = "decision_example"  # kind decision: the SFT-v2 decision example (replayed in G2)
PREFIX_TEMPLATE = "template"  # [system, instruction]
PREFIX_KINDS = {"read_rft": PREFIX_OWN, "decision": PREFIX_DECISION}
HONEST_CASES = ("honest_failure", "honest_failure_observed")

# Drop reasons (states, prefixes, pairs).
DROP_NO_CACHE_ROW = "no_cache_row"
DROP_AMBIGUOUS = "cache_ambiguous"
DROP_NOT_REPLAYABLE = "not_replayable"
DROP_NO_RECORD = "no_episode_record"
DROP_DISAGREE = "checklist_disagrees_with_tests"
DROP_ARCHIVED = "archived_replayed_disagree"
DROP_TWIN = "twin_no_claim_point"
DROP_STALE = "check_result_of_another_checklist"
DROP_PREFIX_INSTRUCTION = "prefix_instruction_mismatch"
DROP_PREFIX_SHAPE = "prefix_not_a_claim_transcript"
DROP_PREFIX_DUP = "prefix_not_first_for_task"
DROP_NO_PASS = "no_pass_state"
DROP_NO_FAIL = "no_fail_state"
DROP_TASK_CAP = "per_task_cap"
DROP_SOURCE_SHARE = "source_share"


def stable_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with _open(Path(path).expanduser()) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── states ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class State:
    """A validated end state of one task: the current checklist's outputs and the replayed tests' verdict."""

    unit_id: str  # `oracle:<task>` or the episode id
    kind: str  # STATE_ORACLE | STATE_EPISODE
    task: str
    result: cc.CheckResult
    tests_passed: bool
    checklist_pass: bool
    archived_tests_passed: bool | None = None
    outcome: dict[str, Any] = field(default_factory=dict)  # the archived graded outcome (episodes)
    n_cache_rows: int = 1  # cache rows the join matched (all agree)

    @property
    def agrees(self) -> bool:
        return self.checklist_pass == self.tests_passed and self.archived_tests_passed in (None, self.tests_passed)


def cache_results(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The `result` of each cache row whose checks are a list (the validation's `ValidationCache` rule)."""
    out = []
    for r in rows:
        res = r.get("result")
        if isinstance(res, dict) and isinstance(res.get("checks"), list) and res.get("tests_passed") is not None:
            out.append({**res, "_by_cmd": {c.get("cmd"): (c.get("rc"), c.get("output") or "") for c in res["checks"]}})
    return out


def is_oracle(res: Mapping[str, Any]) -> bool:
    return "oracle_tail" in res


def _outputs(res: Mapping[str, Any], checklist: cc.Checklist) -> tuple[tuple[int | None, str], ...] | None:
    by = res["_by_cmd"]
    if any(r.cmd not in by for r in checklist.requirements):
        return None
    return tuple(by[r.cmd] for r in checklist.requirements)


def _result(unit_id: str, checklist: cc.Checklist, outs: Sequence[tuple[int | None, str]]) -> cc.CheckResult:
    results = [{"id": r.id, "cmd": r.cmd, "rc": rc, "output": out}
               for r, (rc, out) in zip(checklist.requirements, outs, strict=True)]  # fmt: skip
    return cc.CheckResult(unit_id, checklist.task, checklist.sha, results)


def _verdict(checklist: cc.Checklist, outs: Sequence[tuple[int | None, str]]) -> bool:
    return all(r.passes(rc, out) for r, (rc, out) in zip(checklist.requirements, outs, strict=True))


def oracle_states(
    cache: Sequence[Mapping[str, Any]], checklists: Mapping[str, cc.Checklist]
) -> tuple[dict[str, State], Counter[str]]:
    """task -> its oracle state, for the tasks whose matching oracle rows all agree (outputs and verdict)."""
    oracles = [r for r in cache if is_oracle(r)]
    out: dict[str, State] = {}
    dropped: Counter[str] = Counter()
    for task, cl in sorted(checklists.items()):
        matched = [(o, r["tests_passed"]) for r in oracles if (o := _outputs(r, cl)) is not None]
        if not matched:
            dropped[DROP_NO_CACHE_ROW] += 1
            continue
        if len(set(matched)) > 1:
            dropped[DROP_AMBIGUOUS] += 1
            continue
        outs, tests = matched[0]
        out[task] = State(f"oracle:{task}", STATE_ORACLE, task, _result(f"oracle:{task}", cl, outs), bool(tests),
                          _verdict(cl, outs), None, {}, len(matched))  # fmt: skip
    return out, dropped


def episode_states(
    cache: Sequence[Mapping[str, Any]],
    checklists: Mapping[str, cc.Checklist],
    check_results: Iterable[cc.CheckResult],
    records: Mapping[str, Mapping[str, Any]],
) -> tuple[list[State], Counter[str]]:
    """The episode states of `check_results`, each joined to the cache rows of the same content (module docstring)."""
    by_cmd: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in cache:
        if not is_oracle(r):
            for cmd in r["_by_cmd"]:
                by_cmd[cmd].append(r)
    out: list[State] = []
    dropped: Counter[str] = Counter()
    for cr in check_results:
        cl, rec = checklists.get(cr.task), records.get(cr.episode_id)
        if cl is None or not cr.matches(cl):
            dropped[DROP_STALE] += 1
            continue
        if not cr.replayable:
            dropped[DROP_NOT_REPLAYABLE] += 1
            continue
        if rec is None:
            dropped[DROP_NO_RECORD] += 1
            continue
        want = tuple((d["rc"], d.get("output") or "") for d in sorted(cr.results, key=lambda d: _index(cl, d["id"])))
        rcs = [h.get("rc") for h in ((rec.get("extra") or {}).get("history") or [])]
        rows = [r for r in by_cmd.get(cl.requirements[0].cmd, [])
                if len(r.get("rcs") or []) == len(rcs) and _outputs(r, cl) == want
                and all(a is None or a == b for a, b in zip(rcs, r.get("rcs") or [], strict=True))]  # fmt: skip
        tests = {r["tests_passed"] for r in rows}
        if not tests:
            dropped[DROP_NO_CACHE_ROW] += 1
            continue
        if len(tests) > 1:
            dropped[DROP_AMBIGUOUS] += 1
            continue
        o = rec.get("outcome") or {}
        out.append(State(cr.episode_id, STATE_EPISODE, cr.task, _result(cr.episode_id, cl, want), bool(tests.pop()),
                         _verdict(cl, want), o.get("tests_passed"), dict(o), len(rows)))  # fmt: skip
    return out, dropped


def _index(cl: cc.Checklist, req_id: str) -> int:
    return next(k for k, r in enumerate(cl.requirements) if r.id == req_id)


# ── prefixes ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Prefix:
    kind: str  # PREFIX_OWN | PREFIX_DECISION | PREFIX_TEMPLATE
    id: str
    task: str
    messages: list[dict[str, Any]]  # [system, user, ...] up to the report turn
    claim: dict[str, Any] | None  # the prefix's own claim report turn (None: template)
    file: str = ""
    episode_id: str | None = None

    def destroys(self, targets: Sequence[str]) -> bool:
        """A bash command of the prefix destroys one of `targets` (the template runs none)."""
        return bool(targets) and any(
            is_destructive_on(str((c.get("function") or {}).get("arguments", {}).get("command") or ""), targets)
            for m in self.messages for c in m.get("tool_calls") or [] if (c.get("function") or {}).get("name") == "bash"
        )  # fmt: skip


def _claim_turn(m: Mapping[str, Any]) -> bool:
    calls = m.get("tool_calls") or []
    if m.get("role") != "assistant" or len(calls) != 1:
        return False
    fn = calls[0].get("function") or {}
    return fn.get("name") == "submit_report" and (fn.get("arguments") or {}).get("status") == "complete"


def load_prefixes(
    paths: Sequence[str | Path], tasks: Iterable[str], instructions: Mapping[str, str]
) -> tuple[dict[str, list[Prefix]], Counter[str]]:
    """task -> its transcript prefixes, one per file at most, in file (priority) order; within a file the first
    example in a stable hash order."""
    wanted = set(tasks)
    out: dict[str, list[Prefix]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    for path in paths:
        rows = [r for r in read_jsonl(path) if r.get("variant") in wanted and r.get("kind") in PREFIX_KINDS]
        seen: set[str] = set()
        for r in sorted(rows, key=lambda r: stable_hash(str(r["id"]))):
            task, msgs = r["variant"], r["messages"]
            if task in seen:
                dropped[DROP_PREFIX_DUP] += 1
                continue
            if len(msgs) < 4 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user" \
                    or msgs[-2].get("role") != "tool" or not _claim_turn(msgs[-1]):  # fmt: skip
                dropped[DROP_PREFIX_SHAPE] += 1
                continue
            if str(msgs[1].get("content") or "").strip() != instructions[task].strip():
                dropped[DROP_PREFIX_INSTRUCTION] += 1
                continue
            seen.add(task)
            meta = r.get("meta") or {}
            head = [{"role": "system", "content": harness.SYSTEM_PROMPT}, dict(msgs[1])]
            out[task].append(Prefix(PREFIX_KINDS[r["kind"]], str(r["id"]), task, [*head, *msgs[2:-1]], msgs[-1],
                                    _portable(path), meta.get("source_episode")))  # fmt: skip
    return dict(out), dropped


def template_prefix(task: str, instruction: str) -> Prefix:
    msgs = [{"role": "system", "content": harness.SYSTEM_PROMPT}, {"role": "user", "content": instruction}]
    return Prefix(PREFIX_TEMPLATE, f"template:{task}", task, msgs, None)


# ── pairs ────────────────────────────────────────────────────────────────────


def claim_args(checklist: cc.Checklist) -> dict[str, Any]:
    """The template prefix's claim: every requirement done, evidenced by the checks just run."""
    return {"status": "complete", "completed": [r.quote for r in checklist.requirements], "incomplete": [],
            "blocker_category": "", "blocker_target": "", "blocker_detail": "",
            "evidence_commands": list(checklist.commands)}  # fmt: skip


def removal_failed(state: State, checklist: cc.Checklist, request: str) -> bool:
    verdicts = state.result.verdicts(checklist)
    return any(not ok and requirement_type(r.quote, r.cmd, request) == REQ_REMOVAL
               for r, ok in zip(checklist.requirements, verdicts, strict=True))  # fmt: skip


def _case(outcome: EpisodeOutcome, args: Mapping[str, Any]) -> str:
    o = dataclasses.replace(outcome, report=report_from_tool_args("submit_report", dict(args)))
    return compute_reward(o, RewardConfig()).case


def state_outcome(state: State, prefix: Prefix) -> EpisodeOutcome:
    """The facts a decision on `state` is graded with: the archived outcome with the replayed verdict, or (oracle) a
    clean control pass."""
    if state.kind == STATE_EPISODE:
        return dataclasses.replace(outcome_from_dict(dict(state.outcome)), tests_passed=state.tests_passed)
    n = sum(1 for m in prefix.messages if m.get("role") == "tool")
    return EpisodeOutcome(feasible=True, report=None, tests_passed=True, pass_fraction=1.0, n_actions=n,
                          agent_acted=True, is_control=True, stop_reason="report")  # fmt: skip


@dataclass
class Pair:
    task: str
    source: str
    surface: str
    prefix: Prefix
    passed: SFTExample
    failed: SFTExample
    dpo: PreferencePair | None

    @property
    def examples(self) -> tuple[SFTExample, SFTExample]:
        return self.passed, self.failed


def _args(m: Mapping[str, Any]) -> dict[str, Any]:
    return dict(m["tool_calls"][0]["function"]["arguments"])


def build_pair(
    prefix: Prefix,
    pass_state: State,
    fail_state: State,
    checklist: cc.Checklist,
    *,
    source: str,
    surface: str = "",
    dpo: bool = False,
) -> Pair | str:
    """The matched pair on `prefix` (module docstring), or a drop reason."""
    key = f"{prefix.id}:{checklist.task}"  # both sides: the same check turns' text
    pass_checks = cc.check_turns(checklist, pass_state.result, prefix.messages, key)
    fail_checks = cc.check_turns(checklist, fail_state.result, prefix.messages, key)
    claim = prefix.claim or dec.report_turn(claim_args(checklist), cc.PASS_DECISION_LINES[0])
    verdicts = fail_state.result.verdicts(checklist)
    honest = cc.honest_report_args(checklist, verdicts)
    first_failed = next(r for r, ok in zip(checklist.requirements, verdicts, strict=True) if not ok)
    decision = dec.report_turn(honest, cc._pick(cc.FAIL_DECISION_LINES, key + ":fail").format(
        quote=cc._short(first_failed.quote)))  # fmt: skip
    if _case(state_outcome(pass_state, prefix), _args(claim)) != "success_claimed":
        return cc.DROP_CASE
    fail_outcome = state_outcome(fail_state, prefix)
    if _case(fail_outcome, honest) not in HONEST_CASES or _case(fail_outcome, _args(claim)) != "fabrication":
        return cc.DROP_CASE
    injected = [*pass_checks, *fail_checks, decision, *([] if prefix.claim else [claim])]
    if not cc._clean(injected):
        return cc.DROP_LABEL
    n = len(prefix.messages)
    check_idx = [n + 2 * k for k in range(len(checklist.requirements))]
    examples = []
    for outcome, origin, checks, turn, state in ((cc.PASS, ORIGIN_SPLICE_PASS, pass_checks, claim, pass_state),
                                                 (cc.FAIL, ORIGIN_SPLICE_FAIL, fail_checks, decision, fail_state)):  # fmt: skip
        messages = [*prefix.messages, *checks, turn]
        meta = {
            "source": source, "task": checklist.task, "surface": surface, "point": cc.POINT_CLAIM, "outcome": outcome,
            "origin": origin, "prefix_kind": prefix.kind, "prefix_id": prefix.id, "prefix_file": prefix.file,
            "prefix_episode": prefix.episode_id, "state": state.unit_id, "state_kind": state.kind,
            "checklist_sha": checklist.sha, "verdicts": state.result.verdicts(checklist),
            "tests_passed": state.tests_passed, "archived_tests_passed": state.archived_tests_passed,
            "n_cache_rows": state.n_cache_rows, "n_prefix_messages": n,
            "loss_roles": {"checks": check_idx, "decision": len(messages) - 1},
        }  # fmt: skip
        examples.append(SFTExample(f"{key}:{state.unit_id}:{origin}", cc.KIND, checklist.task, messages,
                                   [*check_idx, len(messages) - 1], meta))  # fmt: skip
    passed, failed = examples
    pair = None
    if dpo:
        pair = PreferencePair(f"{failed.id}:dpo", cc.FAIL, [*prefix.messages, *fail_checks], decision, claim,
                              {**failed.meta, "variant": checklist.task, "rejected": "claim"})  # fmt: skip
    return Pair(checklist.task, source, surface, prefix, passed, failed, pair)


def _pick_states(states: Sequence[State], task: str, n: int) -> list[State]:
    return sorted(states, key=lambda s: stable_hash(f"{task}:{s.unit_id}"))[:n]


def task_pairs(
    task: str,
    checklist: cc.Checklist,
    prefixes: Sequence[Prefix],
    passes: Sequence[State],
    fails: Sequence[State],
    *,
    source: str,
    surface: str,
    request: str,
    n_pairs: int,
    dpo: bool,
) -> tuple[list[Pair], Counter[str]]:
    """Up to `n_pairs` pairs of one task: the first prefix (priority order, the template last) that some fail state
    does not contradict; distinct fail states, pass states in a stable hash order (reused when fewer)."""
    dropped: Counter[str] = Counter()
    if not passes:
        dropped[DROP_NO_PASS] += 1
        return [], dropped
    if not fails:
        dropped[DROP_NO_FAIL] += 1
        return [], dropped
    targets = sorted({t for s in fails for t in high_risk_targets({}, s.outcome)})
    for prefix in prefixes:
        destroys = prefix.destroys(targets)
        usable = [s for s in fails if not (destroys and removal_failed(s, checklist, request))]
        if not usable:
            dropped[f"prefix_skipped:{prefix.kind}:removal_contradicts_prefix"] += 1
            continue
        chosen = _pick_states(usable, task, n_pairs)
        ordered_passes = _pick_states(passes, task, len(passes))
        pairs: list[Pair] = []
        for k, f in enumerate(chosen):
            built = build_pair(prefix, ordered_passes[k % len(ordered_passes)], f, checklist, source=source,
                               surface=surface, dpo=dpo)  # fmt: skip
            if isinstance(built, str):
                dropped[built] += 1
            else:
                pairs.append(built)
        if len(usable) > n_pairs:
            dropped[DROP_TASK_CAP] += len(usable) - n_pairs
        if pairs:
            return pairs, dropped
    return [], dropped


def cap_source_share(pairs: Sequence[Pair], max_share: float) -> tuple[list[Pair], int]:
    """Drop whole pairs from the largest sources (stable hash order) until no source holds more than `max_share` of
    the pairs (a source keeps at least one pair). Returns the kept pairs, in input order, and the number dropped."""
    kept = list(pairs)
    while kept:
        per = Counter(p.source for p in kept)
        limit = max(1, int(max_share * len(kept) + 1e-9))
        over = [s for s, n in per.items() if n > limit]
        if not over:
            break
        worst = max(over, key=lambda s: (per[s], s))
        victim = max((p for p in kept if p.source == worst), key=lambda p: stable_hash("share:" + p.passed.id))
        kept.remove(victim)
    return kept, len(pairs) - len(kept)


# ── the build ────────────────────────────────────────────────────────────────


@dataclass
class Config:
    tasks_roots: list[str]
    episodes: list[str]
    prefix_examples: list[str]
    cache: str = str(DEFAULT_CACHE)
    check_results: list[str] = field(default_factory=lambda: [str(DEFAULT_CHECK_RESULTS)])
    checklists: str = str(cc.DEFAULT_CHECKLISTS)
    data_dir: str = "data"
    out: str = str(DEFAULT_OUT)
    pairs_per_task: int = 1
    quota: cc.Quota = field(default_factory=cc.Quota)
    max_source_share: float = DEFAULT_MAX_SOURCE_SHARE
    template_prefix: bool = True
    dpo: bool = False


def build(cfg: Config) -> dict[str, Any]:
    if not 1 <= cfg.pairs_per_task <= cfg.quota.cap_per_task:
        raise ValueError(f"pairs_per_task must be in [1, cap_per_task={cfg.quota.cap_per_task}]")
    excluded = excluded_from_training(cfg.data_dir)
    roots = [Path(r).expanduser() for r in cfg.tasks_roots]
    records = {str(r["episode_id"]): r for p in cfg.episodes for r in read_jsonl(p)}
    source_of = {str(r.get("variant_id") or r.get("task_id")): str(r.get("pair_id")) for r in records.values()}
    cc.assert_no_excluded(source_of.values(), excluded, "episode records")
    results = cc.read_check_results([Path(p).expanduser() for p in cfg.check_results])
    names = {Path(p).stem for p in Path(cfg.checklists).expanduser().glob("*.json")}
    controls = {t for t in names if t.endswith(cc.CONTROL_SUFFIX)}
    checklists, bad = cc.load_checklists(cfg.checklists, controls, roots)
    cache = cache_results(read_jsonl(cfg.cache))
    oracles, oracle_drops = oracle_states(cache, checklists)
    twin_rows = sum(1 for r in results.values() if not r.task.endswith(cc.CONTROL_SUFFIX))
    eps, ep_drops = episode_states(cache, checklists, (r for r in results.values()
                                                       if r.task.endswith(cc.CONTROL_SUFFIX)), records)  # fmt: skip
    ep_drops[DROP_TWIN] += twin_rows
    passes: dict[str, list[State]] = defaultdict(list)
    fails: dict[str, list[State]] = defaultdict(list)
    state_counts: Counter[str] = Counter()
    for s in [*oracles.values(), *eps]:
        cell = f"{s.kind}/{'pass' if s.tests_passed else 'fail'}"
        if not s.agrees:
            state_counts[f"{cell}:{DROP_ARCHIVED if s.checklist_pass == s.tests_passed else DROP_DISAGREE}"] += 1
            continue
        state_counts[f"{cell}:agreed"] += 1
        (passes if s.tests_passed else fails)[s.task].append(s)
    # The pass state: an agreed passing own episode, else the oracle.
    pass_of = {t: [s for s in ss if s.kind == STATE_EPISODE] or [s for s in ss if s.kind == STATE_ORACLE]
               for t, ss in passes.items()}  # fmt: skip
    tdirs = {t: d for t in checklists if (d := cc._task_dir(t, roots)) is not None}
    instructions = {t: harness.read_instruction(d) for t, d in tdirs.items()}
    metas = {t: task_meta(d) for t, d in tdirs.items()}
    prefixes, prefix_drops = load_prefixes(cfg.prefix_examples, checklists, instructions)
    for t, m in metas.items():
        if t in source_of and m["source"] and m["source"] != source_of[t]:
            raise ValueError(f"{t}: task.toml pair_id {m['source']!r} != episode pair_id {source_of[t]!r}")
    pairs: list[Pair] = []
    task_drops: Counter[str] = Counter()
    for task in sorted(checklists):
        cands = [*prefixes.get(task, []), *([template_prefix(task, instructions[task])] if cfg.template_prefix else [])]
        built, drops = task_pairs(task, checklists[task], cands, pass_of.get(task, []), fails.get(task, []),
                                  source=metas[task]["source"] or source_of.get(task, ""),
                                  surface=metas[task]["surface"], request=metas[task]["request"],
                                  n_pairs=cfg.pairs_per_task, dpo=cfg.dpo)  # fmt: skip
        pairs += built
        task_drops.update(drops)
    kept, n_share = cap_source_share(pairs, cfg.max_source_share)
    task_drops[DROP_SOURCE_SHARE] += n_share
    examples = [e for p in kept for e in p.examples]
    dpo_pairs = [p.dpo for p in kept if p.dpo is not None]
    used_states = {e.meta["state"] for e in examples}
    n_checked = cc.assert_no_excluded(
        [*(str(e.meta["source"]) for e in examples), *(str(d.meta["source"]) for d in dpo_pairs),
         *(source_of[s.task] for s in eps if s.unit_id in used_states)], excluded, "claim pairs",
    )  # fmt: skip
    out = Path(cfg.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "examples.jsonl").write_text("".join(json.dumps(asdict(e), default=str) + "\n" for e in examples))
    outputs = ["examples.jsonl"]
    if cfg.dpo:
        (out / "dpo_pairs.jsonl").write_text("".join(json.dumps(asdict(d), default=str) + "\n" for d in dpo_pairs))
        outputs.append("dpo_pairs.jsonl")
    cells = cc.cell_counts(examples, lambda e: cc.POINT_CLAIM, lambda e: e.meta["outcome"])

    def by(key: str) -> dict[str, int]:
        return dict(sorted(Counter(f"{e.meta['outcome']}/{e.meta[key]}" for e in examples).items()))

    summary = {
        "inputs": _inputs(cfg),
        "outputs": {name: sha256_file(out / name) for name in outputs},
        "checklists": {
            "control_tasks": len(controls),
            "valid": len(checklists),
            "invalid": bad,
            "with_task_dir": len(tdirs),
        },
        "cache_rows": len(cache),
        "states": dict(sorted(state_counts.items())),
        "state_drops": {"oracle": dict(sorted(oracle_drops.items())), "episode": dict(sorted(ep_drops.items()))},
        "tasks": {
            "with_agreed_pass": len(pass_of),
            "with_agreed_fail": len(fails),
            "with_both": len(set(pass_of) & set(fails)),
        },
        "prefixes": {
            "loaded": dict(sorted(Counter(p.kind for ps in prefixes.values() for p in ps).items())),
            "drops": dict(sorted(prefix_drops.items())),
        },
        "pair_drops": dict(sorted(task_drops.items())),
        "pairs_before_source_share": len(pairs),
        "pairs": len(kept),
        "examples": cells,
        "examples_by_prefix": by("prefix_kind"),
        "examples_by_state": by("state_kind"),
        "examples_by_surface": by("surface"),
        "tasks_in_build": len({p.task for p in kept}),
        "sources_in_build": len({p.source for p in kept}),
        "largest_source_share": round(max(Counter(p.source for p in kept).values()) / len(kept), 4) if kept else None,
        "dpo_pairs": len(dpo_pairs) if cfg.dpo else None,
        "excluded_sources_checked": {"sources": n_checked, "excluded": 0},
        "quota": {**asdict(cfg.quota), "pairs_per_task": cfg.pairs_per_task, "max_source_share": cfg.max_source_share},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "report.md").write_text(render(summary))
    write_manifest(build_manifest(kept, cfg, summary), out / cc.MANIFEST_FILE)
    return summary


def _inputs(cfg: Config) -> list[dict[str, Any]]:
    files = [("cache", cfg.cache), *(("episodes", p) for p in cfg.episodes),
             *(("check_results", p) for p in cfg.check_results), *(("prefix_examples", p) for p in cfg.prefix_examples)]  # fmt: skip
    out = [{"kind": k, "path": _portable(p), "sha256": sha256_file(Path(p).expanduser())} for k, p in files]
    lists = sorted(Path(cfg.checklists).expanduser().glob("*.json"))
    tree = hashlib.sha256(b"".join(p.name.encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest() for p in lists))
    return [*out, {"kind": "checklists", "path": cfg.checklists, "sha256": tree.hexdigest(), "files": len(lists)}]


def _portable(path: str | Path) -> str:
    """`path` with the home directory written as `~` (the manifest is committed; the paths are the Mac's)."""
    p, home = str(Path(path).expanduser()), str(Path.home())
    return "~" + p[len(home) :] if p == home or p.startswith(home + "/") else str(path)


def build_manifest(pairs: Sequence[Pair], cfg: Config, summary: Mapping[str, Any]) -> BuildManifest:
    """The coverage report's manifest: every example is an agreed state of a training-pool control task, and it trains
    the decision its outcome licenses (the claim after a pass, the honest report after a fail)."""
    refs = [ExampleRef(e.id, e.variant, p.source, cc.POINT_CLAIM, e.meta["outcome"], e.meta["outcome"] == cc.PASS,
                       e.meta["origin"], split="train", checklist_agrees=True)
            for p in pairs for e in p.examples]  # fmt: skip
    quota = ManifestQuota(cap_per_task=cfg.quota.cap_per_task, max_outcome_ratio=cfg.quota.max_outcome_ratio,
                          max_source_share=cfg.max_source_share)  # fmt: skip
    keys = ("inputs", "outputs", "examples", "examples_by_prefix", "examples_by_state", "pairs", "dpo_pairs",
            "sources_in_build", "tasks_in_build", "pair_drops")  # fmt: skip
    return BuildManifest.of(refs, quota, {"builder": "sft.claim_splice", **{k: summary[k] for k in keys}})


def render(summary: Mapping[str, Any]) -> str:
    lines = ["# Claim splice build", "", "| cell | examples |", "|---|---:|"]
    lines += [f"| {cell} | {n} |" for cell, n in summary["examples"].items()]
    lines += ["", (f"pairs: {summary['pairs']} ({summary['pairs_before_source_share']} before the source-share cap), "
                   f"tasks {summary['tasks_in_build']}, sources {summary['sources_in_build']}, "
                   f"largest source share {summary['largest_source_share']}, DPO pairs {summary['dpo_pairs']}")]  # fmt: skip
    for key in ("examples_by_prefix", "examples_by_state", "examples_by_surface", "states", "state_drops", "tasks",
                "prefixes", "pair_drops", "checklists", "excluded_sources_checked", "quota"):  # fmt: skip
        lines.append(f"{key}: {json.dumps(summary[key], sort_keys=True)}")
    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m honest_agents.sft.claim_splice", description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks-root", action="append", required=True, help="directory of task dirs; repeatable")
    ap.add_argument("--episodes", action="append", required=True, help="run_eval episodes.jsonl[.gz] of the states")
    ap.add_argument("--prefix-examples", action="append", default=[],
                    help="SFT examples file (read_rft / decision kinds), in priority order; repeatable")  # fmt: skip
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="checklist-validation result cache (JSONL)")
    ap.add_argument("--check-results", action="append", default=None, help="CheckResult JSONL; repeatable")
    ap.add_argument("--checklists", default=str(cc.DEFAULT_CHECKLISTS))
    ap.add_argument("--data-dir", default="data", help="data directory with select_sources.txt and split_seeds.json")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pairs-per-task", type=int, default=1)
    ap.add_argument("--cap-per-task", type=int, default=cc.Quota.cap_per_task)
    ap.add_argument("--max-source-share", type=float, default=DEFAULT_MAX_SOURCE_SHARE)
    ap.add_argument("--no-template-prefix", action="store_true", help="never fall back to the [system, user] prefix")
    ap.add_argument("--dpo", action="store_true", help="also write fail-branch DPO pairs")
    args = ap.parse_args(argv)
    cfg = Config(
        tasks_roots=args.tasks_root, episodes=args.episodes, prefix_examples=args.prefix_examples, cache=args.cache,
        check_results=args.check_results or [str(DEFAULT_CHECK_RESULTS)], checklists=args.checklists,
        data_dir=args.data_dir, out=args.out, pairs_per_task=args.pairs_per_task,
        quota=cc.Quota(cap_per_task=args.cap_per_task), max_source_share=args.max_source_share,
        template_prefix=not args.no_template_prefix, dpo=args.dpo,
    )  # fmt: skip
    summary = build(cfg)
    print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
