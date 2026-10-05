"""Regression tests for review findings (context overflow, prompt parity, bad task dirs, scorer edge cases)."""

from __future__ import annotations

import json

import pytest

from honest_agents.eval.external.agentic_abstention import score
from honest_agents.eval.harbor_agent import RESULT_FILE, HonestPolicyAgent
from honest_agents.eval.harbor_results import load_harbor_job
from honest_agents.eval.metrics import compute_metrics
from honest_agents.eval.rollout import (
    RolloutConfig,
    Task,
    env_tools,
    initial_messages,
    make_threaded_dispatch,
    run_agent_loop,
    run_episode,
    run_rollouts,
    tool_schemas,
)
from tests.eval.scripted_policy import (
    COMPLETE,
    FakeClient,
    FakeToolEnv,
    abstain,
    make_record,
    response,
    tool_call,
)
from tests.eval.test_harbor import write_trial
from tests.sandbox_fakes import LocalSandbox, make_task_dir


class _Status400(Exception):
    status_code = 400


def _loop(client):
    env = FakeToolEnv()
    return run_agent_loop(
        client,
        RolloutConfig(model="m"),
        [{"role": "user", "content": "u"}],
        tool_schemas(),
        make_threaded_dispatch(env_tools(env)),
    )


@pytest.mark.asyncio
async def test_context_overflow_after_first_turn_is_policy_side_not_infra():
    overflow = _Status400("Error code: 400 - This model's maximum context length is 32768 tokens. However, ...")
    res = await _loop(FakeClient([response("", [tool_call("bash", {"command": "ls"})]), overflow]))
    assert res.stop_reason == "context_length" and res.error is None and "maximum context" in res.overflow


@pytest.mark.asyncio
async def test_400_on_first_call_is_infra():
    # e.g. vLLM started without --enable-auto-tool-choice: every request fails, which must not look like policy output
    res = await _loop(FakeClient([_Status400("maximum context length")]))
    assert res.stop_reason == "api_error" and res.error


def test_initial_messages_follow_trl_prompt_column():
    prompt = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task:\n"},
    ]
    assert initial_messages("do x", prompt=prompt) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Task:\ndo x"},
    ]
    assert prompt[-1]["content"] == "Task:\n"  # input row untouched
    with pytest.raises(ValueError):
        initial_messages("x", prompt=[{"role": "system", "content": "S"}])


@pytest.mark.asyncio
async def test_row_prompt_is_used(tmp_path):
    t = make_task_dir(tmp_path / "t", honest="feasible = true")
    client = FakeClient([response("done")])
    row = {
        "task_dir": str(t),
        "prompt": [
            {"role": "system", "content": "CUSTOM"},
            {"role": "user", "content": ""},
        ],
    }
    await run_episode(client, RolloutConfig(model="m"), FakeToolEnv(), Task(str(t), row))
    assert client.requests[0]["messages"][0]["content"] == "CUSTOM"
    assert client.requests[0]["messages"][1]["content"] == "Write `hi` to hello.txt."


@pytest.mark.asyncio
async def test_bad_task_dir_does_not_abort_the_run(tmp_path):
    good = make_task_dir(tmp_path / "good", honest="feasible = true")
    bad = tmp_path / "bad"
    bad.mkdir()  # no task.toml
    recs = await run_rollouts(
        [
            Task(str(bad), {"task_dir": str(bad)}),
            Task(str(good), {"task_dir": str(good)}),
        ],
        RolloutConfig(model="m"),
        client=FakeClient(lambda m, i: response("done")),
        env_factory=FakeToolEnv,
    )
    by = {r.task_id: r for r in recs}
    assert by["bad"].infra_error and by["good"].infra_error is None


@pytest.mark.asyncio
async def test_failed_episode_still_grades_to_free_the_sandbox(tmp_path, monkeypatch):
    from honest_agents.eval import rollout

    t = make_task_dir(tmp_path / "t", honest="feasible = true")
    graded = []

    class Env(FakeToolEnv):
        def _graded(self):
            graded.append(1)
            return super()._graded()

    def broken_tools(env):
        raise RuntimeError("schema generation failed")

    monkeypatch.setattr(rollout, "env_tools", broken_tools)
    rec, _ = await run_episode(
        FakeClient([]),
        RolloutConfig(model="m"),
        Env(),
        Task(str(t), {"task_dir": str(t)}),
    )
    assert rec.infra_error and "schema generation failed" in rec.infra_error and graded == [1]


def test_paired_macro_with_slash_in_family():
    recs = [
        make_record("a", feasible=True, report=COMPLETE, passed=True, pair_id="a"),
        make_record(
            "a-x",
            feasible=False,
            report=abstain("other"),
            pair_id="a",
            family="src/remove_tool",
        ),
    ]
    m = compute_metrics(recs, n_boot=20)
    assert m["paired"]["paired_accuracy_macro"]["value"] == 1.0
    assert set(m["paired"]["by_family"]) == {"src/remove_tool"}


def test_agentic_abstention_attempts_missing_and_verifier_failures(tmp_path):
    job = tmp_path / "job"
    rep = {"status": "abstain", "reason": "r"}
    # two attempts of t1__fp: one timely, one claims complete
    write_trial(
        job,
        "a1",
        task_name="t1__fp",
        task_path=None,
        reward=0.0,
        report=rep,
        report_turn=1,
    )
    write_trial(
        job,
        "a2",
        task_name="t1__fp",
        task_path=None,
        reward=0.0,
        report={"status": "complete"},
    )
    # verifier crashed after the agent abstained: the decision still counts
    write_trial(
        job,
        "b1",
        task_name="t2__fp",
        task_path=None,
        reward=None,
        report=rep,
        report_turn=1,
    )
    # sandbox never started (no agent output): missing, excluded from denominators
    write_trial(job, "c1", task_name="t3__fp", task_path=None, reward=None, metadata={})
    manifest = {
        k: {
            "task_name": k,
            "expected_decision": "abstain",
            "instruction_level_category": "fp",
        }
        for k in ("t1__fp", "t2__fp", "t3__fp", "t4__fp")  # t4 never ran
    }
    s = score(load_harbor_job(job), manifest)
    assert s["n_missing"] == 2 and s["n_warranted"] == 3
    assert s["timely_recall"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_report_file_replaced_even_if_policy_planted_a_readonly_one(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext

    logs = tmp_path / "agent"
    logs.mkdir()
    planted = logs / RESULT_FILE
    planted.write_text('{"report": {"status": "complete"}}')
    planted.chmod(0o444)
    client = FakeClient(
        [
            response(
                "",
                [
                    tool_call(
                        "abstain",
                        {
                            "reason": "no",
                            "blocker_category": "other",
                            "blocker_target": "",
                            "confidence": 0.5,
                        },
                    )
                ],
            )
        ]
    )
    agent = HonestPolicyAgent(logs_dir=logs, model_name="policy", api_base="http://x/v1")
    monkeypatch.setattr(agent, "_client", lambda: client)
    await agent.run("x", LocalSandbox(tmp_path / "sb"), AgentContext())
    assert json.loads(planted.read_text())["report"]["status"] == "abstain"


@pytest.mark.asyncio
async def test_agent_closes_its_client(tmp_path, monkeypatch):
    from harbor.models.agent.context import AgentContext

    closed = []
    client = FakeClient([response("bye")])

    async def close():
        closed.append(True)

    client.close = close
    agent = HonestPolicyAgent(logs_dir=tmp_path / "agent", model_name="policy", api_base="http://x/v1")
    monkeypatch.setattr(agent, "_client", lambda: client)
    await agent.run("x", LocalSandbox(tmp_path / "sb"), AgentContext())
    assert closed == [True]
