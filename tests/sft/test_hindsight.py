"""Hindsight report rules: what the rewritten report may say, and what it must never contain."""

from __future__ import annotations

import json

import pytest

from honest_agents.bank import BankRow
from honest_agents.report import parse_report
from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import hindsight
from honest_agents.sft.hindsight import HindsightOptions, build_example, hindsight_report
from tests.sft.helpers import denied_facts, make_row, missing_command_facts

NOT_FOUND = [("jq . in.json", "bash: line 1: jq: command not found")]


def report_of(row: BankRow, table=None, opts=None):
    return hindsight_report(row, table, opts)


def test_clean_pass_is_complete_and_names_no_blocker():
    r = report_of(make_row(passed=True, original={"status": "incomplete"}))
    assert (r.status, r.tool) == ("complete", "submit_report")
    assert r.arguments["blocker_category"] == "" and r.arguments["blocker_target"] == ""


def test_failure_is_incomplete_even_when_the_agent_claimed_complete():
    r = report_of(make_row(passed=False, original={"status": "complete"}))
    assert r.status == "incomplete"
    assert r.arguments["blocker_category"] == ""  # nothing verified, nothing named


def test_verified_error_in_transcript_is_named_with_its_evidence_line():
    row = make_row(turns=NOT_FOUND, facts=missing_command_facts())
    r = report_of(row)
    assert (r.arguments["blocker_category"], r.arguments["blocker_target"]) == ("missing_capability", "jq")
    assert "jq: command not found" in r.arguments["blocker_detail"]
    assert r.source == hindsight.SOURCE_TRANSCRIPT


def test_blocker_the_facts_do_not_verify_is_not_named():
    # `jq` is on the PATH according to the cached facts: "command not found" cannot be a true blocker.
    r = report_of(make_row(turns=NOT_FOUND, facts=missing_command_facts("jq")))
    assert r.arguments["blocker_category"] == "" and r.blocker is None


def test_unknown_facts_name_nothing():
    row = make_row(turns=NOT_FOUND)
    row = row.model_copy(update={"blocker_facts": type(row.blocker_facts)()})  # nothing cached
    assert report_of(row).blocker is None


def test_error_for_a_command_the_agent_never_ran_is_not_named():
    row = make_row(turns=[("ls", "Error: not found")], facts=missing_command_facts())
    assert report_of(row).blocker is None


def test_agent_claim_needs_its_object_in_the_transcript():
    claim = {"status": "incomplete", "blocker_category": "missing_capability", "blocker_target": "secret_tool"}
    row = make_row(original=claim, facts=missing_command_facts())
    assert report_of(row).blocker is None  # verified true by the facts, but never seen in a command or output
    seen = make_row(
        original=claim,
        turns=[("secret_tool --go", "bash: secret_tool: command not found")],
        facts=missing_command_facts(),
    )
    r = report_of(seen)
    assert (r.blocker.target, r.source) == ("secret_tool", hindsight.SOURCE_AGENT)


def test_permission_denied_path_from_output():
    row = make_row(turns=[("cat /data/x", "cat: /data/x: Permission denied")], facts=denied_facts("/data/x"))
    r = report_of(row)
    assert (r.arguments["blocker_category"], r.arguments["blocker_target"]) == ("permission_denied", "/data/x")


def test_network_blocker_needs_a_no_network_policy():
    out = [("curl https://pypi.org/x", "curl: (6) Could not resolve host: pypi.org")]
    off = missing_command_facts().model_copy(update={"network_mode": "no-network"})
    assert report_of(make_row(turns=out, facts=off)).arguments["blocker_target"] == "pypi.org"
    assert report_of(make_row(turns=out, facts=missing_command_facts())).blocker is None  # public network


def test_abstain_only_when_the_agent_stopped_for_a_named_blocker():
    abstain = {"reason": "no jq", "blocker_category": "missing_capability", "blocker_target": "jq"}
    stopped = make_row(original=abstain, tool="abstain", turns=NOT_FOUND, facts=missing_command_facts())
    r = report_of(stopped)
    assert (r.tool, r.status) == ("abstain", "abstain") and set(r.arguments) >= {"reason", "blocker_target"}
    # the same abstain without a verifiable blocker is an honest "incomplete", not an abstention
    plain = make_row(original=abstain, tool="abstain", turns=[("ls", "in.txt")], facts=missing_command_facts())
    assert (report_of(plain).tool, report_of(plain).status) == ("submit_report", "incomplete")


def test_task_spec_labels_never_reach_the_example():
    """expected_targets / perturbation / family are verified-true facts here, but they are not in the transcript."""
    row = make_row(
        feasible=False,
        expected_targets=["hidden_target_xyz"],
        expected_blockers=["missing_capability"],
        facts=missing_command_facts(),
        turns=[("ls", "in.txt")],
    )
    ex = build_example(row)
    blob = json.dumps(ex.messages) + json.dumps(ex.meta)
    for forbidden in ("hidden_target_xyz", "secret_perturbation", "missing_capability", "expected"):
        assert forbidden not in blob
    assert ex.messages[-1]["tool_calls"][0]["function"]["arguments"]["blocker_category"] == ""


def test_example_shape_loss_on_the_report_only():
    row = make_row(turns=NOT_FOUND, facts=missing_command_facts())
    ex = build_example(row)
    assert ex.loss_messages == [len(ex.messages) - 1]
    assert ex.messages[-1]["role"] == "assistant" and ex.messages[-2]["role"] == "tool"
    assert ex.messages[0]["role"] == "system"
    # the rewritten report parses and is exactly what the harness tool would accept
    args = ex.messages[-1]["tool_calls"][0]["function"]["arguments"]
    assert parse_report(args).status.value == "incomplete"


def test_hindsight_report_scores_as_the_honest_report():
    cfg = RewardConfig()
    for passed, original in ((True, {"status": "incomplete"}), (False, {"status": "complete"})):
        row = make_row(passed=passed, original=original)
        r = report_of(row)
        outcome = row.episode_outcome()
        outcome.report = parse_report(r.arguments)
        reward = compute_reward(outcome, cfg)
        assert not reward.fabricated and not reward.false_blame


# ── confidence ───────────────────────────────────────────────────────────────


def heldout_rows(n_complete_pass: int, n_complete_fail: int, n_inc: int):
    rows = [make_row(f"h{i}", passed=True, original={"status": "complete"}) for i in range(n_complete_pass)]
    rows += [make_row(f"hf{i}", passed=False, original={"status": "complete"}) for i in range(n_complete_fail)]
    rows += [make_row(f"hi{i}", passed=False, original={"status": "incomplete"}) for i in range(n_inc)]
    return rows


def test_calibration_is_by_original_claim_type_and_binned():
    opts = HindsightOptions(min_bucket=5)
    table = hindsight.calibrate(heldout_rows(7, 3, 10), opts)
    assert table.by_claim["complete"] == pytest.approx(0.7)
    assert table.by_claim["incomplete"] == pytest.approx(0.05)  # clipped away from 0
    assert "abstain" not in table.by_claim and table.overall == pytest.approx(
        0.3
    )  # 7 / 20 = 0.35, binned to the 0.1 grid


def test_confidence_does_not_depend_on_the_outcome_label():
    table = hindsight.calibrate(heldout_rows(7, 3, 10), HindsightOptions(min_bucket=5))
    won = report_of(make_row(passed=True, original={"status": "complete"}), table)
    lost = report_of(make_row(passed=False, original={"status": "complete"}), table)
    assert won.arguments["confidence"] == lost.arguments["confidence"] == pytest.approx(0.7)
    assert (won.status, lost.status) == ("complete", "incomplete")


def test_abstain_confidence_is_p_not_completable():
    table = hindsight.ConfidenceTable(by_claim={"abstain": 0.2})
    abstain = {"reason": "no jq", "blocker_category": "missing_capability", "blocker_target": "jq"}
    row = make_row(original=abstain, tool="abstain", turns=NOT_FOUND, facts=missing_command_facts())
    assert report_of(row, table).arguments["confidence"] == pytest.approx(0.8)


def test_omit_and_empty_tables_drop_the_confidence():
    row = make_row(passed=True)
    table = hindsight.calibrate(heldout_rows(7, 3, 10), HindsightOptions(min_bucket=5))
    assert "confidence" not in report_of(row, table, HindsightOptions(confidence="omit")).arguments
    assert "confidence" not in report_of(row, None).arguments
    assert "confidence" not in report_of(row, hindsight.calibrate([], HindsightOptions())).arguments


def test_calibration_skips_masked_rows():
    rows = [make_row("m", passed=True, original={"status": "complete"}, masked_reason="boot_failure")] * 30
    assert hindsight.calibrate(rows, HindsightOptions(min_bucket=5)).overall is None


# ── adversarial leakage and reward consistency ───────────────────────────────

SENTINELS = {
    "target": "zz_sentinel_target_91",
    "family": "zz_sentinel_family",
    "perturbation": "zz_sentinel_perturbation",
    "kind": "zz_sentinel_kind",
    "decoy": "zz_sentinel_decoy_77",
}


def _spec_row(*, claim_target: str, turns: list[tuple[str, str]], facts, **kwargs) -> BankRow:
    """A row whose task spec carries sentinels everywhere, and whose agent claims the sentinel target as its blocker."""
    row = make_row(
        feasible=False,
        expected_targets=[SENTINELS["target"]],
        expected_blockers=["missing_capability"],
        original={
            "status": "abstain",
            "reason": "blocked",
            "blocker_category": "missing_capability",
            "blocker_target": claim_target,
        },
        tool="abstain",
        turns=turns,
        facts=facts,
        **kwargs,
    )
    task = {
        **row.task,
        "family": SENTINELS["family"],
        "perturbation": SENTINELS["perturbation"],
        "kind": SENTINELS["kind"],
        "is_control": True,
        "decoy_targets": [SENTINELS["decoy"]],
    }
    return row.model_copy(update={"task": task})


def test_spec_sentinels_cannot_enter_an_example_unless_the_transcript_shows_them():
    # The facts verify the sentinel as a missing command, and the agent claimed it, but it is nowhere in the transcript.
    row = _spec_row(claim_target=SENTINELS["target"], turns=[("ls", "in.txt")], facts=missing_command_facts())
    ex = build_example(row)
    blob = json.dumps(ex.messages) + json.dumps(ex.meta)
    for value in SENTINELS.values():
        assert value not in blob
    args = ex.messages[-1]["tool_calls"][0]["function"]["arguments"]
    assert ex.messages[-1]["tool_calls"][0]["function"]["name"] == "submit_report"  # not an abstain: unverifiable
    assert (args["blocker_category"], args["blocker_target"]) == ("", "")


def test_a_spec_sentinel_is_named_only_when_the_agent_saw_it():
    target = SENTINELS["target"]
    seen = [(f"{target} --go", f"bash: {target}: command not found")]
    ex = build_example(_spec_row(claim_target=target, turns=seen, facts=missing_command_facts()))
    args = ex.messages[-1]["tool_calls"][0]["function"]["arguments"]
    assert args["blocker_target"] == target  # it is in the transcript the model saw, so naming it is learnable
    assert target in json.dumps(ex.messages[:-1])
    for other in ("family", "perturbation", "kind", "decoy"):
        assert SENTINELS[other] not in json.dumps(ex.messages) + json.dumps(ex.meta)


def test_complete_is_emitted_only_for_clean_passes():
    for kwargs in ({"tamper": True}, {"passed": False}):
        row = make_row(**{"passed": True, **kwargs}, original={"status": "complete"})
        assert not row.episode_outcome().passed
        assert report_of(row).status != "complete"
    assert report_of(make_row(passed=True, original={"status": "abstain", "reason": "x"}, tool="abstain")).status == (
        "complete"
    )


@pytest.mark.parametrize("feasible", [True, False])
@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("claim", ["complete", "incomplete"])
def test_relabelled_report_never_scores_below_the_original_without_a_blocker(feasible, passed, claim):
    """Same outcome, relabelled report: at least the original reward (blocker-free rows, so no checked claim differs)."""
    if passed and not feasible:
        pytest.skip("an infeasible clean pass is dropped as leaky")
    cfg = RewardConfig()
    row = make_row(passed=passed, original={"status": claim}, feasible=feasible)
    outcome = row.episode_outcome()
    before = compute_reward(outcome, cfg).reward
    outcome.report = parse_report(report_of(row).arguments)
    assert compute_reward(outcome, cfg).reward >= before - 1e-9
