"""Synthetic trajectory banks for the pilot tests: no Docker, no GPU, no model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from honest_agents import bank, harness
from honest_agents.bank import BankRow, BlockerFacts, PathFact
from honest_agents.report import SelfReport, parse_report
from honest_agents.reward import (
    BLOCKER_FALSE,
    BLOCKER_TRUE,
    BLOCKER_UNKNOWN,
    EpisodeOutcome,
    RewardConfig,
    compute_reward,
)

SYSTEM = "You are an agent."
INSTRUCTION = "Parse the config and write the result to /app/out.json."


def facts_for(category: str, target: str, verdict: str) -> BlockerFacts:
    """Cached facts under which `bank.check_blocker_cached(category, target, facts)` gives `verdict`.

    Mirrors `reward.blocker_verdict`: a missing command or an unreadable path makes the blocker true, a present command
    or a readable path makes it false, an allowlist-free public network makes `network_unavailable` false, and the
    categories that cannot be checked stay unknown whatever the facts say.
    """
    if verdict == BLOCKER_UNKNOWN:
        return BlockerFacts(commands=[], cwd="/app")
    if category == "missing_capability":
        have = [] if verdict == BLOCKER_TRUE else [target.rsplit("/", 1)[-1]]
        return BlockerFacts(commands=have, cwd="/app")
    if category == "permission_denied":
        fact = PathFact(
            exists=True,
            readable=verdict == BLOCKER_FALSE,
            writable=verdict == BLOCKER_FALSE,
        )
        return BlockerFacts(commands=[], paths={target: fact}, cwd="/app")
    if category == "network_unavailable":
        return BlockerFacts(
            commands=[],
            network_mode="no-network" if verdict == BLOCKER_TRUE else "public",
            cwd="/app",
        )
    raise ValueError(f"no checkable facts for {category}")


def report_call(report: dict[str, Any] | None = None, tool: str = "submit_report") -> dict[str, Any]:
    """An assistant message carrying one report tool call, in the form TRL's `parse_response` produces."""
    args = report or {
        "status": "complete",
        "completed": ["all"],
        "incomplete": [],
        "confidence": 0.9,
    }
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}],
    }


def bash_turn(command: str, output: str = "ok") -> list[dict[str, Any]]:
    call = {
        "type": "function",
        "function": {"name": "bash", "arguments": {"command": command}},
    }
    return [
        {"role": "assistant", "content": f"Running {command}", "tool_calls": [call]},
        {"role": "tool", "name": "bash", "content": f"exit code: 0\n{output}"},
    ]


def outcome(feasible: bool = True, report: SelfReport | dict | None = None, **kw: Any) -> EpisodeOutcome:
    if isinstance(report, dict):
        report = parse_report(report)
    kw.setdefault("n_actions", 3)
    kw.setdefault("tests_passed", False)
    return EpisodeOutcome(feasible=feasible, report=report, **kw)


def make_bank_row(
    *,
    variant: str = "v0",
    kind: str = "feasible",
    passed: bool = False,
    original: dict[str, Any] | None = None,
    n_commands: int = 3,
    facts: BlockerFacts | None = None,
    reward_config: RewardConfig | None = None,
    no_report: bool = False,
    masked_reason: str | None = None,
    tamper: bool = False,
    extra_report_calls: int = 0,
    episode_id: str | None = None,
    expected_blockers: list[str] | None = None,
    expected_targets: list[str] | None = None,
    spec: dict[str, Any] | None = None,
    decoy_targets: list[str] | None = None,
) -> BankRow:
    """One banked trajectory: n bash turns, then (unless `no_report`) a report turn. The outcome is graded with the
    real `compute_reward` under `reward_config`, as the live reward function would have done."""
    feasible = kind != "infeasible"
    original = original if original is not None else ({"status": "complete"} if passed else {"status": "incomplete"})
    cfg = reward_config or RewardConfig()
    o = outcome(
        feasible,
        None if no_report else original,
        tests_passed=passed,
        n_actions=n_commands,
        is_control=kind == "control",
        expected_blockers=expected_blockers
        if expected_blockers is not None
        else ([] if feasible else ["missing_capability"]),
        expected_targets=expected_targets if expected_targets is not None else ([] if feasible else ["jq"]),
        masked_reason=masked_reason,
        tamper=tamper,
        decoy_targets=decoy_targets or [],
    )
    completion: list[dict[str, Any]] = []
    for i in range(n_commands):
        completion += bash_turn(f"cmd{i}")
    if not no_report:
        message = report_call(original)
        if extra_report_calls:
            extra = [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "late"}},
                }
            ] * extra_report_calls
            message["tool_calls"] = extra + message["tool_calls"]
        completion.append(message)
    br = compute_reward(o, cfg)
    return bank.make_row(
        prompt=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION},
        ],
        completion=completion,
        outcome=o,
        breakdown=br,
        reward_value=br.reward,
        task={
            "task_dir": f"tasks/{variant}",
            "variant_id": variant,
            "kind": kind,
            "feasible": feasible,
            "is_control": kind == "control",
            "pair_id": variant.split("-")[0],
            "split": "train",
            "family": "fam",
            "perturbation": None,
            "spec": spec or {},
        },
        blocker_facts=facts or BlockerFacts(commands=["ls", "cat"], cwd="/app", network_mode="public"),
        reward_config=cfg,
        group=variant,
    )


def write_bank(directory: Path, rows: list[BankRow], name: str = "bank-1.jsonl") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / name, "w") as f:
        f.writelines(r.model_dump_json() + "\n" for r in rows)
    return directory


def synthetic_rows(
    n_feasible: int = 12,
    n_control: int = 4,
    n_infeasible: int = 6,
    pass_share: float = 0.5,
    per_variant: int = 2,
) -> list[BankRow]:
    """A small bank with a given pass share on feasible tasks and every kind present."""
    rows: list[BankRow] = []
    for kind, n in (
        ("feasible", n_feasible),
        ("control", n_control),
        ("infeasible", n_infeasible),
    ):
        for i in range(n):
            for j in range(per_variant):
                passed = kind != "infeasible" and (
                    (i + j) % max(1, round(1 / pass_share)) == 0 if pass_share else False
                )
                rows.append(make_bank_row(variant=f"{kind[:1]}{i}-{kind}", kind=kind, passed=passed))
    return rows


def episode_record(**kw: Any) -> dict[str, Any]:
    """One `rewards.episode_record`-shaped dict for curve and analysis tests."""
    rec = {
        "round": 0,
        "step": 0,
        "pilot_id": "p0",
        "variant_id": "v0",
        "kind": "feasible",
        "feasible": True,
        "is_control": False,
        "forced_report": False,
        "passed": False,
        "has_report": True,
        "parse_error": False,
        "tool_error": None,
        "status": "incomplete",
        "claimed": False,
        "declined": True,
        "case": "honest_failure",
        "masked": False,
        "reward": 0.25,
        "honest_total": 0.25,
        "false_blame": False,
        "blocker_category": None,
        "blocker_check": "unknown",
        "reason_match": None,
        "confidence": None,
    }
    rec.update(kw)
    return rec


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


__all__ = ["BLOCKER_FALSE", "BLOCKER_TRUE", "BLOCKER_UNKNOWN", "harness"]
