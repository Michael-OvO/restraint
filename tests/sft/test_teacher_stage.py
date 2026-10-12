"""Teacher-trajectory SFT: every drop reason, the audit gate, the infeasible cap, the loss mask and the selection CLI."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from honest_agents.bank import BankRow
from honest_agents.reward import compute_reward
from honest_agents.sft import teacher_stage as ts
from honest_agents.sft import train_sft
from tests.sft.helpers import bank_dir, make_row
from tests.sft.test_train_sft import ROOT

NOT_FOUND = [("cat /app/in.json", "{}"), ("jq . /app/in.json", "bash: line 1: jq: command not found")]
LOOK_THEN_WRITE = [("ls /app", "in.txt"), ("cp /app/in.txt /app/out.txt", "")]


def teacher(row: BankRow, token: str | None, *, family: str = "fam", **task) -> BankRow:
    """`row` as the teacher writes it: a token in `extra`, the kind/family columns in `task`."""
    extra = {**row.extra, **({ts.TOKEN_KEY: token} if token is not None else {})}
    return row.model_copy(update={"extra": extra, "task": {**row.task, "family": family, **task}})


def feasible_ok(token: str, variant: str | None = None, **kw) -> BankRow:
    turns = kw.pop("turns", LOOK_THEN_WRITE)
    family = kw.pop("family", "feasible")
    row = make_row(variant or f"f-{token}", passed=True, original={"status": "complete"}, turns=turns, **kw)
    return teacher(row, token, family=family, kind="control" if kw.get("is_control") else "feasible")


def abstain_ok(token: str, variant: str | None = None, family: str = "remove_input", **kw) -> BankRow:
    row = make_row(
        variant or f"i-{token}",
        feasible=False,
        tool="abstain",
        original={"reason": "no jq", "blocker_category": "missing_capability", "blocker_target": "jq"},
        expected_blockers=["missing_capability"],
        expected_targets=["jq"],
        turns=kw.pop("turns", NOT_FOUND),
        **kw,
    )
    return teacher(row, token, family=family, kind="infeasible")


def outcome_with(row: BankRow, **fields) -> BankRow:
    return row.model_copy(update={"outcome": {**row.outcome, **fields}})


def keep_all(rows: list[BankRow], keep: bool = True) -> dict[str, dict]:
    return {ts.token_of(r): {"token": ts.token_of(r), "keep": keep} for r in rows if ts.token_of(r)}


def reasons(rows: list[BankRow], verdicts: dict | None = None) -> tuple[list[BankRow], Counter]:
    dropped: Counter = Counter()
    kept = ts.select_rows(rows, keep_all(rows) if verdicts is None else verdicts, dropped)
    return kept, dropped


# ── programmatic filters ─────────────────────────────────────────────────────


def test_clean_success_and_correct_abstention_are_kept_with_their_cases():
    good = [feasible_ok("a"), feasible_ok("c", is_control=True), abstain_ok("b")]
    assert [compute_reward(r.episode_outcome(), r.config()).case for r in good] == [
        "success_claimed",
        "success_claimed",
        "abstain_correct",
    ]
    kept, dropped = reasons(good)
    assert kept == good and not dropped


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (lambda: feasible_ok("t", split="validation"), ts.DROP_SPLIT),
        (lambda: feasible_ok("t", masked_reason="grading_error"), ts.DROP_MASKED),
        (lambda: teacher(make_row("v", passed=True, no_report=True), "t"), ts.DROP_NO_REPORT),
        (lambda: feasible_ok("t").model_copy(update={"report_turn_extra_calls": 1}), ts.DROP_HIDDEN_CALLS),
        (lambda: outcome_with(feasible_ok("t"), report=None, report_error="bad json"), ts.DROP_UNPARSED),
        (lambda: feasible_ok("t", tamper=True), ts.DROP_TAMPERED),
        (lambda: abstain_ok("t", damage_paths=("/app/in.json",)), ts.DROP_SELF_INFLICTED),
        (lambda: outcome_with(feasible_ok("t"), irreversible_violation=True), ts.DROP_VIOLATION),
        (lambda: outcome_with(abstain_ok("t"), tests_passed=True), ts.DROP_LEAK),
        # a fabrication: claimed complete, tests failed
        (
            lambda: teacher(make_row("v", passed=False, original={"status": "complete"}), "t", kind="feasible"),
            ts.DROP_FEASIBLE_CASE,
        ),
        # an honest non-completion on a feasible task is honest but not what the teacher set is for
        (lambda: teacher(make_row("v", passed=False), "t", kind="feasible"), ts.DROP_FEASIBLE_CASE),
        # infeasible, but an implicit give-up instead of an abstention with the blocker
        (
            lambda: teacher(make_row("v", feasible=False, turns=NOT_FOUND), "t", kind="infeasible"),
            ts.DROP_INFEASIBLE_CASE,
        ),
        (lambda: feasible_ok("t", turns=[]), ts.DROP_NO_COMMANDS),
        (lambda: feasible_ok("t", turns=[("rm -rf /app/cache && ls /app", "")]), ts.DROP_NO_ORIENTATION),
        (lambda: outcome_with(feasible_ok("t"), first_mutation_idx=0), ts.DROP_NO_ORIENTATION),
    ],
)
def test_every_programmatic_drop_reason_is_counted(row, reason):
    kept, dropped = reasons([row()])
    assert kept == [] and dropped == Counter({reason: 1})


def test_infeasible_pass_is_refused_even_if_the_grader_did_not_mask_it():
    row = outcome_with(abstain_ok("t"), tests_passed=True)
    assert ts.programmatic_drop_reason(row) in (ts.DROP_LEAK, ts.DROP_MASKED)


def test_orientation_reads_bash_commands_with_string_arguments():
    # Real teacher rows carry arguments as a JSON string (the OpenAI form); the check must still see the commands.
    row = feasible_ok("t")
    completion = [dict(m) for m in row.completion]
    for m in completion:
        for call in m.get("tool_calls") or []:
            call["function"] = {**call["function"], "arguments": json.dumps(call["function"]["arguments"])}
    row = row.model_copy(update={"completion": completion})
    assert ts.bash_commands(row) == [c for c, _ in LOOK_THEN_WRITE]
    assert ts.oriented_first(row) is None


def test_no_mutation_at_all_is_oriented():
    assert ts.oriented_first(abstain_ok("t")) is None


def test_a_blank_command_is_not_orientation():
    # an empty call before the first write looked at nothing: it must not satisfy "look before you change"
    row = feasible_ok("t", turns=[("", ""), ("cp /app/in.txt /app/out.txt", "")])
    assert ts.oriented_first(row) == ts.DROP_NO_ORIENTATION
    assert ts.oriented_first(feasible_ok("t", turns=[("  ", "")])) == ts.DROP_NO_COMMANDS


# ── audit ────────────────────────────────────────────────────────────────────


def test_audit_gate_counts_missing_token_missing_verdict_and_rejection():
    rows = [
        feasible_ok("yes"),
        feasible_ok("no"),
        feasible_ok("unseen"),
        feasible_ok("x").model_copy(update={"extra": {}}),
    ]
    verdicts = {"yes": {"token": "yes", "keep": True}, "no": {"token": "no", "keep": False}}
    kept, dropped = reasons(rows, verdicts)
    assert [ts.token_of(r) for r in kept] == ["yes"]
    assert dropped == Counter({ts.DROP_AUDIT_REJECTED: 1, ts.DROP_NO_AUDIT: 1, ts.DROP_NO_TOKEN: 1})


def test_keep_must_be_a_json_true():
    kept, dropped = reasons([feasible_ok("s")], {"s": {"token": "s", "keep": "true"}})
    assert kept == [] and dropped[ts.DROP_AUDIT_REJECTED] == 1


def test_programmatic_filters_run_before_the_audit():
    row = feasible_ok("t", tamper=True)
    _, dropped = reasons([row], {"t": {"token": "t", "keep": True}})
    assert dropped == Counter({ts.DROP_TAMPERED: 1})


def test_read_audit_last_verdict_wins_and_missing_file_is_empty(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps({"token": "a", "keep": True})
        + "\n\n"
        + json.dumps({"token": "a", "keep": False, "issues": ["re-audit"]})
        + "\n"
        + json.dumps({"token": "b", "keep": True})
        + "\n"
        + json.dumps({"tok": "c", "keep": True})  # a typo: ignored, never a verdict for anything
        + "\n"
        + json.dumps({"token": "", "keep": True})
        + "\n"
    )
    verdicts = ts.read_audit(path)
    assert verdicts["a"]["keep"] is False and verdicts["b"]["keep"] is True
    assert set(verdicts) == {"a", "b"}
    assert ts.read_audit(tmp_path / "missing.jsonl") == {}


def test_default_audit_path_sits_next_to_the_bank_directory(tmp_path: Path):
    bank = tmp_path / "teacher" / "bank"
    bank.mkdir(parents=True)
    assert ts.audit_path(ts.TeacherOptions(), bank) == tmp_path / "teacher" / "audit.jsonl"
    assert ts.audit_path(ts.TeacherOptions(), bank / "bank-1.jsonl") == tmp_path / "teacher" / "audit.jsonl"
    assert ts.audit_path(ts.TeacherOptions(audit="x.jsonl"), bank) == Path("x.jsonl")


# ── balancing ────────────────────────────────────────────────────────────────


def test_infeasible_limit():
    assert ts.infeasible_limit(10, 13, 0.35) == 7  # 7 / 20 = 0.35
    assert ts.infeasible_limit(3, 13, 0.35) == 3
    assert ts.infeasible_limit(10, 0, 0.35) == 0
    assert ts.infeasible_limit(10, 5, None) == 10 and ts.infeasible_limit(10, 5, 1.0) == 10
    assert ts.infeasible_limit(10, 5, 0.0) == 0


def test_balance_caps_the_infeasible_share_round_robin_across_families():
    feas = [feasible_ok(f"f{i}") for i in range(13)]
    inf = [abstain_ok(f"r{i}", family="remove_input") for i in range(8)] + [
        abstain_ok(f"h{i}", family="high_risk") for i in range(2)
    ]
    dropped: Counter = Counter()
    out = ts.balance(feas + inf, 0.35, dropped)
    comp = ts.composition(out)
    assert comp["by_kind"] == {"feasible": 13, "infeasible": 7} and comp["infeasible_share"] <= 0.35
    # both high_risk rows survive: the cap thins the big family first
    assert comp["by_family"]["infeasible/high_risk"] == 2 and comp["by_family"]["infeasible/remove_input"] == 5
    assert dropped == Counter({ts.DROP_INFEASIBLE_CAP: 3})
    assert ts.balance(feas + inf, 0.35) == out  # stable


def test_balance_under_the_cap_changes_nothing():
    rows = [feasible_ok("a"), feasible_ok("b"), abstain_ok("c")]
    dropped: Counter = Counter()
    assert ts.balance(rows, 0.35, dropped) == rows and not dropped


# ── examples ─────────────────────────────────────────────────────────────────


def test_example_trains_every_assistant_turn_and_keeps_the_teacher_confidence():
    row = abstain_ok("t")
    row = row.model_copy(
        update={
            "report_call": {**row.report_call, "arguments": {**row.report_call["arguments"], "confidence": 0.83}},
            "report_message": {
                **row.report_message,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "abstain",
                            "arguments": {**row.report_call["arguments"], "confidence": 0.83},
                        },
                    }
                ],
            },
        }
    )
    ex = ts.build_example(row)
    assert ex.kind == ts.KIND and ex.meta["teacher_token"] == "t" and ex.meta["case"] == "abstain_correct"
    assistants = [i for i, m in enumerate(ex.messages) if m["role"] == "assistant"]
    assert ex.loss_messages == assistants and len(assistants) == len(NOT_FOUND) + 1
    assert ex.messages[-1]["tool_calls"][-1]["function"]["arguments"]["confidence"] == 0.83


# ── config, stage and CLI ────────────────────────────────────────────────────


def test_shipped_config_parses_and_registers_the_stage():
    cfg = train_sft.load_config(ROOT / "configs" / "sft_teacher_qwen35.yaml")
    assert cfg.kind == "teacher" and cfg.lora.r == 32 and cfg.model.name == "Qwen/Qwen3.5-4B"
    assert cfg.teacher["max_infeasible_share"] == 0.35 and cfg.train.num_train_epochs >= 1
    assert "teacher" in train_sft.STAGES
    with pytest.raises(ValueError, match="unknown key"):
        train_sft.parse_config({"kind": "teacher", "teacher": {"nope": 1}})
    with pytest.raises(ValueError, match="max_infeasible_share"):
        train_sft.parse_config({"kind": "teacher", "teacher": {"max_infeasible_share": 1.5}})


def _teacher_bank(tmp_path: Path) -> tuple[Path, Path]:
    rows = [feasible_ok(f"f{i}") for i in range(12)]
    rows += [abstain_ok(f"i{i}", family="high_risk" if i % 2 else "remove_input") for i in range(12)]
    rows += [feasible_ok("tampered", tamper=True), feasible_ok("rejected"), feasible_ok("unaudited")]
    bank = bank_dir(tmp_path / "teacher", rows)
    verdicts = [{"token": ts.token_of(r), "keep": ts.token_of(r) != "rejected"} for r in rows]
    audit = tmp_path / "teacher" / "audit.jsonl"
    audit.write_text("".join(json.dumps(v) + "\n" for v in verdicts if v["token"] != "unaudited"))
    return bank, audit


def test_selection_cli_prints_the_summary_without_a_model(tmp_path: Path, capsys):
    bank, _ = _teacher_bank(tmp_path)
    rc = ts.main(
        [
            "--config",
            str(ROOT / "configs" / "sft_teacher_qwen35.yaml"),
            f"data.bank={bank}",
            "teacher.audit=null",  # default: audit.jsonl next to the bank directory
            "data.holdout_fraction=0.0",
            f"output_root={tmp_path / 'out'}",
        ]
    )
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    t = summary["teacher"]
    assert t["audit_file"] == str(tmp_path / "teacher" / "audit.jsonl")
    # the tampered row never reaches the stage (data.read_rows drops it); the stage's own drops are audit and cap
    assert summary["rows_dropped"] == {"tampered": 1}
    assert t["dropped"] == {ts.DROP_AUDIT_REJECTED: 1, ts.DROP_NO_AUDIT: 1, ts.DROP_INFEASIBLE_CAP: 6}
    assert t["selected_before_balance"]["by_kind"] == {"feasible": 12, "infeasible": 12}
    assert t["selected"]["by_kind"] == {"feasible": 12, "infeasible": 6}
    assert t["selected"]["by_family"] == {
        "feasible/feasible": 12,
        "infeasible/high_risk": 3,
        "infeasible/remove_input": 3,
    }
    assert summary["examples_fit"] == 18 and not (tmp_path / "out").exists()


def test_stage_builds_fit_and_heldout_examples(tmp_path: Path):
    bank, audit = _teacher_bank(tmp_path)
    cfg = train_sft.load_config(
        ROOT / "configs" / "sft_teacher_qwen35.yaml",
        [f"data.bank={bank}", f"teacher.audit={audit}", "data.holdout_fraction=0.3"],
    )
    stage, summary = train_sft.build_stage_data(cfg)
    assert stage.train and stage.eval
    assert {e.variant for e in stage.train}.isdisjoint({e.variant for e in stage.eval})
    assert all(e.kind == ts.KIND and e.loss_messages for e in stage.train + stage.eval)
    assert summary["teacher"]["selected"]["infeasible_share"] <= 0.35
    with pytest.raises(ValueError, match="expected 'teacher'"):
        ts.selection_report(train_sft.load_config(ROOT / "configs" / "sft_bestofn.yaml"))
