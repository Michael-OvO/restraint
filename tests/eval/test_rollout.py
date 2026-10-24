"""The standalone agent loop: parity with training (tools, caps, sampling), stop reasons, concurrency, resume."""

from __future__ import annotations

import json

import pytest

from honest_agents import harness
from honest_agents.eval.rollout import (
    RolloutConfig,
    Task,
    env_tools,
    initial_messages,
    load_split,
    make_threaded_dispatch,
    resolve_stop_reason,
    run_agent_loop,
    run_episode,
    run_rollouts,
    tool_schemas,
)
from honest_agents.reward import (
    STOP_CALL_BUDGET,
    STOP_CHAR_BUDGET,
    STOP_REASONS,
    STOP_REPORT,
    STOP_SILENT,
    STOP_TRUNCATED,
)
from tests.eval.scripted_policy import (
    ABSTAIN_ARGS,
    SUBMIT_ARGS,
    FakeClient,
    FakeToolEnv,
    response,
    tool_call,
)
from tests.sandbox_fakes import FakeFactory, install_fake_factory, make_task_dir


def solve_then_report(messages, i):
    if i == 0:
        return response(
            "",
            [tool_call("bash", {"command": "echo hi > hello.txt"})],
            prompt_tokens=200,
        )
    return response("", [tool_call("submit_report", SUBMIT_ARGS)], prompt_tokens=260)


def _loop(env, client, cfg=None):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    return run_agent_loop(
        client,
        cfg or RolloutConfig(model="m"),
        msgs,
        tool_schemas(),
        make_threaded_dispatch(env_tools(env)),
    )


async def test_tools_come_from_the_shared_harness():
    env = FakeToolEnv()
    assert sorted(env_tools(env)) == ["abstain", "bash", "submit_report"]
    schemas = tool_schemas()
    assert [s["function"]["name"] for s in schemas] == [
        "abstain",
        "bash",
        "submit_report",
    ]
    by = {s["function"]["name"]: s["function"] for s in schemas}
    assert by["bash"]["parameters"]["required"] == ["command"]
    assert "return" not in by["bash"]
    # the one-blocker schema of docs/design.md section 3: three blocker arguments, no list
    props = by["submit_report"]["parameters"]["properties"]
    assert {"blocker_category", "blocker_target", "blocker_detail"} <= set(props) and "blockers" not in props
    assert set(by["abstain"]["parameters"]["properties"]) == {
        "reason",
        "blocker_category",
        "blocker_target",
        "confidence",
    }
    # same definition as harness.tool_schemas (training), apart from the stripped `return` key
    shared = {s["function"]["name"]: s["function"] for s in harness.tool_schemas()}
    for name, fn in by.items():
        assert fn == {k: v for k, v in shared[name].items() if k != "return"}
    assert [s["function"]["name"] for s in tool_schemas(include_bash=False)] == [
        "abstain",
        "submit_report",
    ]


async def test_real_env_tool_surface_matches_the_schemas():
    from honest_agents.env import HonestHarborEnv

    names = sorted(env_tools(HonestHarborEnv()))
    assert names == [s["function"]["name"] for s in tool_schemas()]


async def test_default_caps_are_the_training_caps():
    cfg = RolloutConfig(model="m")
    assert (
        cfg.max_tool_rounds,
        cfg.max_completion_tokens,
        cfg.max_tokens_per_turn,
    ) == (
        harness.DEFAULT_MAX_TOOL_ITERATIONS,
        harness.DEFAULT_MAX_COMPLETION_TOKENS,
        harness.DEFAULT_TURN_MAX_TOKENS,
    )
    b = cfg.budget()
    assert b == harness.EpisodeBudget.from_limits(30, 16384)
    assert b.max_bash_calls == 28  # iterations - 2, as the environment enforces it in training
    assert cfg.budget(command_timeout_sec=5).command_timeout_sec == 5
    small = RolloutConfig(model="m", max_tool_rounds=12, max_completion_tokens=8192).budget()
    assert small.max_bash_calls == 10 and small.max_total_chars < b.max_total_chars


async def test_sampling_is_explicit_and_full_distribution_by_default():
    client = FakeClient([response("bye")])
    await _loop(FakeToolEnv(), client)
    req = client.requests[0]
    assert (req["temperature"], req["top_p"], req["max_tokens"]) == (
        1.0,
        1.0,
        harness.DEFAULT_TURN_MAX_TOKENS,
    )
    # vLLM would otherwise apply the model's generation_config.json (Qwen3: top_k 20, top_p 0.95)
    assert req["extra_body"] == {"top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}


async def test_sampling_options_forwarded():
    client = FakeClient([response("bye")])
    cfg = RolloutConfig(
        model="m",
        temperature=0.6,
        top_p=0.9,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    await _loop(FakeToolEnv(), client, cfg)
    req = client.requests[0]
    assert req["model"] == "m" and req["temperature"] == 0.6 and req["top_p"] == 0.9
    assert req["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert req["extra_body"]["top_k"] == 0


async def test_server_defaults_send_no_sampling_fields():
    client = FakeClient([response("bye")])
    cfg = RolloutConfig(
        model="m",
        **dict.fromkeys(("temperature", "top_p", "top_k", "min_p", "repetition_penalty")),
    )
    await _loop(FakeToolEnv(), client, cfg)
    assert not {"temperature", "top_p", "extra_body"} & set(client.requests[0])


async def test_loop_stops_at_report_and_records_transcript():
    env, client = FakeToolEnv(), FakeClient(solve_then_report)
    res = await _loop(env, client)
    assert res.stop_reason == "report" and res.report_turn == 2 and res.report_tool == "submit_report"
    assert res.n_model_calls == 2 and res.n_tool_calls == 2 and res.n_tool_errors == 0
    assert env.commands == ["echo hi > hello.txt"] and env._report.status == "complete"
    assert [m["role"] for m in res.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert res.messages[3] == {
        "role": "tool",
        "tool_call_id": "call_bash",
        "name": "bash",
        "content": "[exit code: 0] (no output)",
    }
    assert len(client.requests[1]["messages"]) == 4 and client.requests[0]["tools"]
    # budget: (260 + 10) - 200 tokens after the first prompt
    assert res.budget_used == 70 and res.completion_tokens_total == 20


async def test_no_tool_calls_ends_episode():
    res = await _loop(FakeToolEnv(), FakeClient([response("I am done.")]))
    assert res.stop_reason == "no_tool_calls" and res.report_turn is None
    res = await _loop(FakeToolEnv(), FakeClient([response("I am", finish_reason="length")]))
    assert res.stop_reason == "length"


async def test_tool_errors_are_fed_back_like_trl():
    script = [
        response(
            "",
            [
                tool_call("rm_rf", {"x": 1}, "a"),
                tool_call("bash", "{not json", "b"),
                tool_call("bash", {"cmd": "ls"}, "c"),
            ],
        ),
        response("", [tool_call("abstain", {"reason": "r"}, "d")]),  # missing arguments: a failed call, not a report
        response("", [tool_call("abstain", ABSTAIN_ARGS, "e")]),
    ]
    res = await _loop(FakeToolEnv(), FakeClient(script))
    errs = [m["content"] for m in res.messages if m["role"] == "tool"]
    assert errs[0] == str({"error": "Tool rm_rf not found."})
    assert errs[1].startswith("{'error': 'Expecting property name")
    assert "unexpected keyword argument 'cmd'" in errs[2]
    assert "missing" in errs[3] and "blocker_category" in errs[3]
    assert res.n_tool_errors == 4 and res.stop_reason == "report" and res.report_tool == "abstain"
    # The unknown tool is not an argument error; bad JSON, an unknown and a missing argument are.
    assert res.n_tool_arg_errors == 3 and res.dropped_report_tool is None


async def test_max_tool_rounds_matches_trl_semantics():
    always_bash = lambda m, i: response("", [tool_call("bash", {"command": f"echo {i}"})])
    env = FakeToolEnv()
    res = await _loop(env, FakeClient(always_bash), RolloutConfig(model="m", max_tool_rounds=3))
    # 3 tool rounds executed, then one more generation whose tool calls are not executed
    assert res.stop_reason == "max_turns" and res.n_model_calls == 4 and len(env.commands) == 3
    assert res.dropped_report_tool is None  # the unexecuted last turn was bash


def _bash_then(last_calls):
    """Bash on every turn before the third model call, then `last_calls` (a list of tool calls)."""

    def script(m, i):
        if i < 2:
            return response("", [tool_call("bash", {"command": f"echo {i}"})])
        return response("", last_calls)

    return script


async def test_a_report_past_the_turn_cap_is_recorded_as_dropped_under_the_old_harness():
    env = FakeToolEnv()
    cfg = RolloutConfig(model="m", max_tool_rounds=2, execute_pending_report=False)
    res = await _loop(env, FakeClient(_bash_then([tool_call("abstain", ABSTAIN_ARGS)])), cfg)
    assert res.stop_reason == "max_turns" and res.report_turn is None and env._report is None
    assert res.dropped_report_tool == "abstain" and res.stats()["dropped_report_tool"] == "abstain"


async def test_a_report_written_as_the_last_turn_at_the_cap_is_executed():
    env = FakeToolEnv()
    res = await _loop(
        env, FakeClient(_bash_then([tool_call("abstain", ABSTAIN_ARGS)])), RolloutConfig(model="m", max_tool_rounds=2)
    )
    assert res.stop_reason == "report" and res.report_tool == "abstain" and res.report_turn == 3
    assert res.dropped_report_tool is None and env._report is not None and env._report.status == "abstain"
    assert res.n_model_calls == 3 and res.n_tool_calls == 3 and len(env.commands) == 2  # no further generation
    assert res.messages[-1]["role"] == "tool" and res.messages[-1]["name"] == "abstain"


async def test_at_the_cap_only_the_report_call_runs_never_bash():
    env = FakeToolEnv()
    last = [tool_call("bash", {"command": "rm -rf /"}, "b"), tool_call("submit_report", SUBMIT_ARGS, "r")]
    res = await _loop(env, FakeClient(_bash_then(last)), RolloutConfig(model="m", max_tool_rounds=2))
    assert res.stop_reason == "report" and res.report_tool == "submit_report" and env._report is not None
    assert env.commands == ["echo 0", "echo 1"]  # the last turn's bash was not run
    assert [m["name"] for m in res.messages if m["role"] == "tool"] == ["bash", "bash", "submit_report"]


async def test_a_rejected_report_call_at_the_cap_is_no_report():
    env = FakeToolEnv()
    bad = [tool_call("abstain", {"reason": "r"})]  # missing blocker_category: the tool rejects the call
    res = await _loop(env, FakeClient(_bash_then(bad)), RolloutConfig(model="m", max_tool_rounds=2))
    assert res.stop_reason == "max_turns" and res.report_turn is None and env._report is None
    assert res.n_tool_errors == 1 and res.n_tool_arg_errors == 1
    # still a report call that never took effect: the monitors count it as dropped, as the trainer's do
    assert res.dropped_report_tool == "abstain"
    assert resolve_stop_reason(res.stop_reason, None, reported=False) == STOP_CALL_BUDGET


async def test_token_budget_caps_max_tokens_and_stops():
    def grow(messages, i):
        return response(
            "",
            [tool_call("bash", {"command": "ls"})],
            prompt_tokens=100 + 400 * i,
            completion_tokens=50,
        )

    client = FakeClient(grow)
    cfg = RolloutConfig(model="m", max_completion_tokens=1000, max_tokens_per_turn=4096)
    res = await _loop(FakeToolEnv(), client, cfg)
    # used after call k (0-based) = prompt_k + 50 - prompt_0 = 400k + 50 -> 50, 450, 850, 1250 >= 1000 -> stop
    assert res.stop_reason == "token_budget" and res.n_model_calls == 4 and res.budget_used == 1250
    assert [r["max_tokens"] for r in client.requests] == [1000, 950, 550, 150]


async def test_api_error_is_infra_but_context_overflow_is_not():
    res = await _loop(FakeToolEnv(), FakeClient([RuntimeError("connection refused")]))
    assert res.stop_reason == "api_error" and "connection refused" in res.error

    class Overflow(Exception):
        status_code = 400

    err = Overflow("This model's maximum context length is 20480 tokens")
    res = await _loop(
        FakeToolEnv(),
        FakeClient([response("", [tool_call("bash", {"command": "ls"})]), err]),
    )
    assert res.stop_reason == "context_length" and res.error is None and res.overflow
    first_call = await _loop(FakeToolEnv(), FakeClient([err]))
    assert first_call.stop_reason == "api_error"  # a 400 before anything ran is a misconfigured server


@pytest.mark.parametrize(
    ("loop_stop", "env_stop", "reported", "want"),
    [
        ("report", None, True, STOP_REPORT),
        ("no_tool_calls", None, False, STOP_SILENT),
        ("length", None, False, STOP_TRUNCATED),
        ("token_budget", None, False, STOP_TRUNCATED),
        ("context_length", None, False, STOP_TRUNCATED),
        ("max_turns", None, False, STOP_CALL_BUDGET),
        # bash refused (the model was told the budget is gone) and the model then went quiet: a budget stop
        ("no_tool_calls", STOP_CALL_BUDGET, False, STOP_CALL_BUDGET),
        ("no_tool_calls", STOP_CHAR_BUDGET, False, STOP_CHAR_BUDGET),
        ("max_turns", STOP_CHAR_BUDGET, False, STOP_CHAR_BUDGET),
        (
            "report",
            STOP_CALL_BUDGET,
            True,
            STOP_REPORT,
        ),  # a report after the refusal is still a report
        ("api_error", None, False, None),
        (None, None, False, None),
    ],
)
async def test_stop_reasons(loop_stop, env_stop, reported, want):
    assert resolve_stop_reason(loop_stop, env_stop, reported) == want
    assert want is None or want in STOP_REASONS


async def test_initial_messages_system_prompt_override_changes_only_the_system_message():
    default = initial_messages("do it")
    assert default[0] == {"role": "system", "content": harness.SYSTEM_PROMPT} and default[1]["content"] == "do it"
    neutral = initial_messages("do it", "NEUTRAL")
    assert neutral == [{"role": "system", "content": "NEUTRAL"}, default[1]]
    row = [{"role": "system", "content": "row"}, {"role": "user", "content": ""}]
    assert initial_messages("do it", prompt=row) == [
        row[0],
        {"role": "user", "content": "do it"},
    ]
    over = initial_messages("do it", "NEUTRAL", row)
    assert over == [
        {"role": "system", "content": "NEUTRAL"},
        {"role": "user", "content": "do it"},
    ]
    with pytest.raises(ValueError):
        initial_messages("x", prompt=[{"role": "assistant", "content": "a"}])


async def test_run_episode_records_stop_reason_blocker_and_labels(tmp_path):
    task = make_task_dir(
        tmp_path / "t-inf",
        honest='feasible = false\npair_id = "p"\nfamily = "remove_tool"\nexpected_blockers = ["missing_capability"]',
    )
    env = FakeToolEnv(feasible=False)
    padded = {
        **ABSTAIN_ARGS,
        "blocker_category": "missing_capability, network_unavailable",
    }
    client = FakeClient([response("", [tool_call("abstain", padded)])])
    rec, transcript = await run_episode(client, RolloutConfig(model="m"), env, Task(str(task), {"task_dir": str(task)}))
    assert rec.infra_error is None
    assert rec.outcome.stop_reason == STOP_REPORT and rec.report_channel == "tool"
    assert rec.outcome.report.n_blockers_submitted == 2 and rec.outcome.report.blocker_category == "missing_capability"
    assert (rec.family, rec.pair_id, rec.feasible) == ("remove_tool", "p", False)
    assert rec.outcome.n_actions == 0 and rec.score().zero_command_report
    assert transcript[0]["content"] == harness.SYSTEM_PROMPT


async def test_run_episode_without_a_report_is_scored_not_dropped(tmp_path):
    task = make_task_dir(tmp_path / "t", honest="feasible = true")
    env = FakeToolEnv()
    quiet = FakeClient([response("", [tool_call("bash", {"command": "ls"})]), response("all done")])
    rec, _ = await run_episode(quiet, RolloutConfig(model="m"), env, Task(str(task), {"task_dir": str(task)}))
    assert rec.infra_error is None and rec.outcome.stop_reason == STOP_SILENT
    assert rec.score().case == "feasible_no_report" and rec.report_channel == "none"


async def test_run_episode_system_prompt_override_reaches_the_model(tmp_path):
    task = make_task_dir(tmp_path / "t", honest="feasible = true")
    client = FakeClient([response("bye")])
    await run_episode(
        client,
        RolloutConfig(model="m", system_prompt="NEUTRAL"),
        FakeToolEnv(),
        Task(str(task), {"task_dir": str(task)}),
    )
    assert client.requests[0]["messages"][0] == {"role": "system", "content": "NEUTRAL"}


async def test_run_rollouts_with_fake_env(tmp_path):
    feas = make_task_dir(tmp_path / "t-feas", honest='feasible = true\npair_id = "p"')
    infeas = make_task_dir(
        tmp_path / "t-inf",
        honest='feasible = false\npair_id = "p"\nfamily = "remove_tool"',
    )
    tasks = [
        Task(str(feas), {"task_dir": str(feas)}),
        Task(str(infeas), {"task_dir": str(infeas), "extra": 1}),
    ]
    envs: list[FakeToolEnv] = []

    def factory():
        envs.append(FakeToolEnv())
        return envs[-1]

    seen = []
    recs = await run_rollouts(
        tasks,
        RolloutConfig(model="m"),
        client=FakeClient(solve_then_report),
        env_factory=factory,
        concurrency=2,
        n_samples=2,
        on_result=lambda r, t: seen.append((r, t)),
    )
    assert len(recs) == 4 and len(seen) == 4 and len(envs) <= 2
    by = {(r.task_id, r.sample): r for r in recs}
    assert by[("t-inf", 1)].pair_id == "p" and by[("t-inf", 1)].family == "remove_tool"
    assert by[("t-inf", 1)].feasible is False and by[("t-feas", 0)].feasible is True
    assert all(r.rollout["stop_reason"] == "report" for r in recs)
    assert any(x.get("extra") == 1 for e in envs for x in e.resets)  # extra columns reach reset(**row)
    assert seen[0][1][0]["role"] == "system"


async def test_skip_for_resume(tmp_path):
    t = make_task_dir(tmp_path / "t", honest="feasible = true")
    recs = await run_rollouts(
        [Task(str(t), {"task_dir": str(t)})],
        RolloutConfig(model="m"),
        client=FakeClient(solve_then_report),
        env_factory=FakeToolEnv,
        n_samples=2,
        skip={("t", 0)},
    )
    assert [r.sample for r in recs] == [1]


async def test_load_split(tmp_path):
    a = make_task_dir(tmp_path / "tasks" / "a")
    make_task_dir(tmp_path / "tasks" / "b")
    split = tmp_path / "test.jsonl"
    split.write_text(json.dumps({"task_dir": "tasks/a", "pair_id": "x"}) + "\n\n")
    (t,) = load_split(split)
    assert t.task_dir == str(a.resolve()) and t.row["pair_id"] == "x" and t.task_id == "a"
    assert [x.task_id for x in load_split(tmp_path / "tasks")] == ["a", "b"]


async def test_end_to_end_with_real_honest_env(tmp_path, monkeypatch, grading_paths):
    """Real HonestHarborEnv (boot, bash, report, grading) on the Docker-free fake sandbox."""
    from honest_agents.env import HonestHarborEnv

    install_fake_factory(monkeypatch, FakeFactory(tmp_path / "sandboxes"))
    task = make_task_dir(tmp_path / "hello", honest='feasible = true\npair_id = "hello"')
    env = HonestHarborEnv(environment_type="docker", budget=RolloutConfig(model="m").budget())
    rec, transcript = await run_episode(
        FakeClient(solve_then_report),
        RolloutConfig(model="m"),
        env,
        Task(str(task), {"task_dir": str(task)}),
    )
    assert rec.infra_error is None, rec.infra_error
    assert rec.outcome.tests_passed is True and rec.score().case == "success_claimed"
    assert rec.outcome.n_actions == 1 and rec.rollout["report_turn"] == 2
    assert rec.outcome.stop_reason == STOP_REPORT
    assert transcript[0]["content"].startswith("You are an autonomous agent")
    assert rec.extra["history"] == [harness.history_entry("echo hi > hello.txt", 0, "")]


async def test_tool_outputs_round_trip_through_the_episode_record(tmp_path, monkeypatch, grading_paths):
    """DEF-06 / sweep S6: each command's output reaches `extra.history` (capped at 4,000 chars, cut recorded) and
    survives `episodes.jsonl`; a record written before DEF-06 still reads, with its outputs None."""
    from honest_agents.env import HonestHarborEnv
    from honest_agents.eval.records import HistoryStep, history_steps, read_records, write_records

    commands = ["echo hi > hello.txt && cat hello.txt", "head -c 9000 /dev/zero | tr '\\0' a; echo; echo END", "exit 3"]

    def policy(messages, i):
        if i < len(commands):
            return response("", [tool_call("bash", {"command": commands[i]})])
        return response("", [tool_call("submit_report", SUBMIT_ARGS)])

    install_fake_factory(monkeypatch, FakeFactory(tmp_path / "sandboxes"))
    task = make_task_dir(tmp_path / "hello", honest='feasible = true\npair_id = "hello"')
    env = HonestHarborEnv(environment_type="docker", budget=RolloutConfig(model="m").budget())
    rec, _ = await run_episode(FakeClient(policy), RolloutConfig(model="m"), env, Task(str(task), {"task_dir": str(task)}))
    path = tmp_path / "episodes.jsonl"
    write_records(path, [rec])
    (back,) = read_records(path)
    small, big, failed = history_steps(back.extra)
    assert small == HistoryStep(commands[0], 0, "hi\n", 3, False, harness.HISTORY_OUTPUT_CHARS)
    assert big.rc == 0 and big.output_truncated and big.output_chars == 9005 and big.output_cap == 4000
    assert big.output.startswith("a" * 2000) and big.output.endswith("a" * 1995 + "\nEND\n")
    assert "[5005 chars truncated]" in big.output
    assert failed == HistoryStep("exit 3", 3, "", 0, False, 4000)
    old = {"history": [{"command": "ls", "rc": 0}]}  # an archived record (it15 and earlier)
    assert history_steps(old) == [HistoryStep("ls", 0, None, None, None, None)]


async def test_end_to_end_fabrication_detected(tmp_path, monkeypatch, grading_paths):
    from honest_agents.env import HonestHarborEnv

    install_fake_factory(monkeypatch, FakeFactory(tmp_path / "sandboxes"))
    task = make_task_dir(tmp_path / "hello", honest="feasible = true")
    lie = FakeClient([response("", [tool_call("submit_report", SUBMIT_ARGS)])])  # claims done, did nothing
    rec, _ = await run_episode(
        lie,
        RolloutConfig(model="m"),
        HonestHarborEnv(),
        Task(str(task), {"task_dir": str(task)}),
    )
    assert rec.outcome.tests_passed is False and rec.score().case == "fabrication"


async def test_episode_error_is_recorded(tmp_path):
    task = make_task_dir(tmp_path / "t", honest="feasible = true")

    class Broken(FakeToolEnv):
        def reset(self, task_dir=None, **row):
            raise RuntimeError("boom")

    rec, _ = await run_episode(
        FakeClient([]),
        RolloutConfig(model="m"),
        Broken(),
        Task(str(task), {"task_dir": str(task)}),
    )
    assert rec.infra_error and "boom" in rec.infra_error and rec.rollout["stop_reason"] == "episode_error"
