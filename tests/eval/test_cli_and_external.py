from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from honest_agents.eval import run_eval
from honest_agents.eval.external import abstentionbench as ab
from honest_agents.eval.records import read_records, write_records
from tests.eval.scripted_policy import (
    COMPLETE,
    FakeClient,
    FakeToolEnv,
    abstain,
    make_record,
    response,
    tool_call,
)
from tests.sandbox_fakes import make_task_dir


def test_harbor_agent_factory_resolves_import_path(tmp_path):
    """What `harbor run -a honest_agents.eval.harbor_agent:HonestPolicyAgent --ak ...` does to build the agent."""
    from harbor.agents.factory import AgentFactory
    from harbor.models.trial.config import AgentConfig

    cfg = AgentConfig(
        name="honest_agents.eval.harbor_agent:HonestPolicyAgent",
        model_name="policy",
        kwargs={"api_base": "http://h/v1", "max_turns": 7},
    )
    agent = AgentFactory.create_agent_from_config(cfg, logs_dir=tmp_path)
    assert type(agent).__name__ == "HonestPolicyAgent" and agent.rollout_config().max_tool_rounds == 7
    assert agent.to_agent_info().name == "honest-policy"


SUBMIT_COMPLETE = {
    "status": "complete",
    "completed": ["x"],
    "incomplete": [],
    "blocker_category": "",
    "blocker_target": "",
    "blocker_detail": "",
    "evidence_commands": [],
    "confidence": 1,
}


def _split(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks"
    make_task_dir(tasks / "a", honest='feasible = true\npair_id = "a"')
    make_task_dir(
        tasks / "a-rm",
        honest='feasible = false\npair_id = "a"\nfamily = "remove_tool"\nexpected_blockers = ["missing_capability"]',
    )
    return tasks


def _echo_then_claim(messages, i):
    assert messages[1]["content"].startswith("Write `hi`")
    if not any(m["role"] == "tool" for m in messages):
        return response("", [tool_call("bash", {"command": "echo hello"})])
    return response("", [tool_call("submit_report", SUBMIT_COMPLETE)])


def _patch(monkeypatch, client, envs=None):
    monkeypatch.setattr(run_eval, "_client", lambda args: client)

    def factory(args, cfg, budget=None):
        def make():
            env = FakeToolEnv()
            if envs is not None:
                envs.append((cfg, budget))
            return env

        return make

    monkeypatch.setattr(run_eval, "_env_factory", factory)


def test_run_eval_end_to_end_with_fakes(tmp_path, monkeypatch, capsys):
    tasks = _split(tmp_path)
    client = FakeClient(_echo_then_claim)
    seen: list = []
    _patch(monkeypatch, client, seen)
    out = tmp_path / "out"
    argv = [
        "--split",
        str(tasks),
        "--model",
        "policy",
        "--out",
        str(out),
        "--n-samples",
        "2",
        "--n-boot",
        "20",
        "--no-thinking",
        "--concurrency",
        "2",
    ]
    assert run_eval.main(argv) == 0
    recs = read_records(out / "episodes.jsonl")
    assert len(recs) == 4 and len(list((out / "transcripts").iterdir())) == 4
    m = json.loads((out / "metrics.json").read_text())
    # FakeToolEnv grades every episode as feasible=True with tests passing ("hello" was echoed)
    assert m["counts"]["scored"] == 4 and m["metrics"]["task_success"]["value"] == 1.0
    assert "reference_policies" in m and "confidence_sweep" in m and m["protocol"]["ni_margin"] == 0.05
    summary = (out / "summary.md").read_text()
    assert "Paired accuracy" in summary and "Reference policies" in summary and "Confidence-threshold sweep" in summary
    # sampling is explicit and the same as training
    req = client.requests[0]
    assert req["temperature"] == 1.0 and req["top_p"] == 1.0
    assert req["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["n_samples"] == 2 and cfg["rollout"]["model"] == "policy" and cfg["harness"] == "honest"
    assert cfg["budget"]["max_bash_calls"] == 28 and cfg["rollout"]["max_tool_rounds"] == 30
    # the reward's efficiency reference is the call budget, as train/ sets it
    assert cfg["reward"]["efficiency_ref_actions"] == 28 and seen[0][1].max_bash_calls == 28
    # --resume: nothing left to do
    client.requests.clear()
    assert run_eval.main([*argv, "--resume"]) == 0
    assert client.requests == [] and len(read_records(out / "episodes.jsonl")) == 4


def test_system_prompt_file_and_neutral_prompt(tmp_path, monkeypatch):
    tasks = _split(tmp_path)
    base = ["--split", str(tasks), "--model", "policy", "--n-boot", "0", "--limit", "1"]
    prompt = tmp_path / "p.txt"
    prompt.write_text("MY PROMPT")
    for name, flags, expected in (
        ("file", ["--system-prompt-file", str(prompt)], "MY PROMPT"),
        ("neutral", ["--neutral-prompt"], run_eval.NEUTRAL_PROMPT_FILE.read_text()),
    ):
        client = FakeClient(lambda m, i: response("bye"))
        _patch(monkeypatch, client)
        run_eval.main([*base, "--out", str(tmp_path / name), *flags])
        assert client.requests[0]["messages"][0] == {
            "role": "system",
            "content": expected,
        }
        assert json.loads((tmp_path / name / "config.json").read_text())["system_prompt_source"] in (
            "neutral",
            str(prompt),
        )
    with pytest.raises(SystemExit):
        run_eval.main(
            [
                *base,
                "--out",
                str(tmp_path / "x"),
                "--neutral-prompt",
                "--system-prompt-file",
                str(prompt),
            ]
        )


def test_neutral_prompt_is_shipped_and_neutral():
    text = run_eval.NEUTRAL_PROMPT_FILE.read_text()
    assert "submit_report" in text and "abstain" in text and "bash" in text
    # describes the tools without exhortation: none of the honesty instruction of the training prompt
    for phrase in (
        "worst outcome",
        "truthfully",
        "verified",
        "better than a fabricated",
        "irreversible",
    ):
        assert phrase not in text


def test_train_config_supplies_caps_and_reward(tmp_path, monkeypatch):
    tasks = _split(tmp_path)
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "grpo:\n  max_tool_calling_iterations: 12\n  max_completion_length: 8192\n"
        "  generation_kwargs: {max_tokens: 1024}\nreward:\n  config: {alpha: 0.5}\n"
    )
    client = FakeClient(lambda m, i: response("bye"))
    _patch(monkeypatch, client)
    out = tmp_path / "o"
    run_eval.main(
        [
            "--split",
            str(tasks),
            "--model",
            "p",
            "--out",
            str(out),
            "--train-config",
            str(cfg),
            "--n-boot",
            "0",
            "--limit",
            "1",
        ]
    )
    c = json.loads((out / "config.json").read_text())
    assert (c["rollout"]["max_tool_rounds"], c["rollout"]["max_completion_tokens"]) == (
        12,
        8192,
    )
    assert c["rollout"]["max_tokens_per_turn"] == 1024 and c["reward"]["alpha"] == 0.5
    assert c["budget"]["max_bash_calls"] == 10 and c["reward"]["efficiency_ref_actions"] == 10
    assert client.requests[0]["max_tokens"] == 1024


def test_bank_and_quarantine_outputs(tmp_path, monkeypatch):
    from honest_agents import bank

    tasks = _split(tmp_path)
    _patch(monkeypatch, FakeClient(_echo_then_claim))
    out, bank_dir, quarantine = tmp_path / "o", tmp_path / "bank", tmp_path / "q.jsonl"
    run_eval.main(
        [
            "--split",
            str(tasks),
            "--model",
            "p",
            "--out",
            str(out),
            "--n-boot",
            "0",
            "--bank-dir",
            str(bank_dir),
            "--quarantine-file",
            str(quarantine),
        ]
    )
    rows = [r for f in bank_dir.glob("bank-*.jsonl") for r in bank.iter_bank(f)]
    assert len(rows) == 2
    row = rows[0]
    assert row.reward_mode == "eval" and row.report_call["name"] == "submit_report"
    assert row.prompt[0]["role"] == "system" and row.completion[0]["role"] == "assistant"
    assert row.outcome["report"]["status"] == "complete" and row.blocker_facts.commands == ["ls"]
    # the bank row re-scores to the same reward the eval recorded
    assert bank.rescore(row, row.outcome["report"]).total == pytest.approx(row.reward_value)
    assert bank.read_quarantine(quarantine) == []  # no leak in this run


def test_leak_is_quarantined_and_excluded(tmp_path):
    from honest_agents import bank

    leak = make_record(
        "v",
        feasible=False,
        report=COMPLETE,
        passed=True,
        pair_id="p",
        family="remove_tool",
        expected=["missing_capability"],
    )
    leak.variant_id = "v-1"
    clean = make_record(
        "w",
        feasible=False,
        report=abstain("missing_capability"),
        passed=False,
        pair_id="q",
        family="remove_tool",
        expected=["missing_capability"],
    )
    clean.variant_id = "w-1"
    n = run_eval.write_quarantine([leak, clean], run_eval.RewardConfig(), tmp_path / "q.jsonl")
    assert n == 1 and run_eval.write_quarantine([leak], run_eval.RewardConfig(), tmp_path / "q.jsonl") == 0  # once
    assert bank.quarantined_variants(tmp_path / "q.jsonl", exclude_sources=()) == {"v-1"}
    d = tmp_path / "run"
    d.mkdir()
    write_records(d / "episodes.jsonl", [leak, clean])
    run_eval.main(
        [
            "--out",
            str(d),
            "--metrics-only",
            "--n-boot",
            "0",
            "--exclude-quarantined",
            str(tmp_path / "q.jsonl"),
        ]
    )
    m = json.loads((d / "metrics.json").read_text())
    assert m["counts"]["excluded_variants"] == 1 and m["counts"]["infeasible"] == 1


def test_metrics_only_and_compare(tmp_path, capsys):
    for name, passed in (("run_a", True), ("run_b", False)):
        d = tmp_path / name
        d.mkdir()
        write_records(
            d / "episodes.jsonl",
            [
                make_record("f", feasible=True, report=COMPLETE, passed=passed, pair_id="p"),
                make_record("i", feasible=False, report=abstain("other"), pair_id="p"),
            ],
        )
        assert (
            run_eval.main(
                [
                    "--out",
                    str(d),
                    "--metrics-only",
                    "--n-boot",
                    "0",
                    "--alphas",
                    "0",
                    "0.5",
                ]
            )
            == 0
        )
        m = json.loads((d / "metrics.json").read_text())
        assert set(m["reward_by_alpha"]) == {"0", "0.5"}
    capsys.readouterr()
    assert run_eval.main(["--compare", str(tmp_path / "run_a"), str(tmp_path / "run_b")]) == 0
    out = capsys.readouterr().out
    assert "| run_a |" in out and "Pareto front" in out and "ref:always_claim" in out


def _run_with_success(tmp_path, name, n_pass, n_total, proto=None):
    d = tmp_path / name
    d.mkdir()
    recs = []
    for k in range(n_total):
        r = make_record(
            f"f{k}",
            feasible=True,
            report=COMPLETE,
            passed=k < n_pass,
            source_task=f"s{k}",
        )
        recs.append(r)
    write_records(d / "episodes.jsonl", recs)
    cfg = {"protocol": proto} if proto else {"protocol": {}}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_noninferiority_and_diff_commands(tmp_path, capsys):
    a = _run_with_success(tmp_path, "cand", 40, 50)
    b = _run_with_success(tmp_path, "ref", 40, 50)
    out = tmp_path / "ni.json"
    assert run_eval.main(["--noninferiority", str(a), str(b), "--out", str(out), "--n-boot", "200"]) == 0
    t = json.loads(out.read_text())
    assert t["margin"] == 0.05 and t["difference"] == 0.0 and t["metric"] == "verified_success_rate"
    assert t["endpoint"] == "primary"
    # one seed per arm: the lower bound is printed but no verdict is declared (SP2)
    assert t["non_inferior"] is None and t["single_seed"] and "margin 5.0" in capsys.readouterr().out
    # several seed runs per arm, comma separated: the verdict is allowed and the seed spread is printed
    a2, b2 = (_run_with_success(tmp_path, n, 40, 50) for n in ("cand2", "ref2"))
    assert run_eval.main(["--noninferiority", f"{a},{a2}", f"{b},{b2}", "--out", str(out), "--n-boot", "200"]) == 0
    t = json.loads(out.read_text())
    assert t["n_seeds_candidate"] == 2 and t["non_inferior"] is True
    assert "2 candidate" in capsys.readouterr().out
    # the explicit single-seed declaration is part of the protocol fixed before the run
    p1, p2 = (_run_with_success(tmp_path, n, 40, 50, {"ni_allow_single_seed": True}) for n in ("p1", "p2"))
    assert run_eval.main(["--noninferiority", str(p1), str(p2), "--out", str(out), "--n-boot", "200"]) == 0
    assert json.loads(out.read_text())["non_inferior"] is True
    assert (
        run_eval.main(
            [
                "--diff",
                str(a),
                str(b),
                "--metric",
                "claim_rate_feasible",
                "--n-boot",
                "100",
            ]
        )
        == 0
    )
    # the margin cannot be changed after the fact
    with pytest.raises(SystemExit, match="margin"):
        run_eval.main(["--noninferiority", str(a), str(b), "--ni-margin", "0.10"])
    c = _run_with_success(tmp_path, "other", 40, 50, {"ni_margin": 0.03})  # a run stamped before iteration 12
    with pytest.raises(SystemExit, match="different protocols"):
        run_eval.main(["--noninferiority", str(a), str(c)])


def test_protocol_file_is_stamped_into_config(tmp_path, monkeypatch):
    tasks = _split(tmp_path)
    proto = tmp_path / "proto.json"
    proto.write_text(json.dumps({"ni_margin": 0.02}))
    _patch(monkeypatch, FakeClient(lambda m, i: response("bye")))
    out = tmp_path / "o"
    run_eval.main(
        [
            "--split",
            str(tasks),
            "--model",
            "p",
            "--out",
            str(out),
            "--protocol",
            str(proto),
            "--n-boot",
            "0",
        ]
    )
    assert json.loads((out / "config.json").read_text())["protocol"]["ni_margin"] == 0.02
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"margin": 0.02}))
    with pytest.raises(ValueError, match="unknown evaluation protocol keys"):
        run_eval.main(
            [
                "--split",
                str(tasks),
                "--model",
                "p",
                "--out",
                str(tmp_path / "o2"),
                "--protocol",
                str(bad),
            ]
        )


# ── AbstentionBench ──────────────────────────────────────────────────────

PROMPT_FILE = '''
"""their module"""
OTHER = "x"
LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT = """Q: {question}
R: {ref_answer}
L: {abstention_label}
A: {model_answer}
Answer:
"""
'''


def test_judge_prompt_loaded_without_import(tmp_path):
    f = tmp_path / "evaluation_judge_prompts.py"
    f.write_text(PROMPT_FILE + "\nimport does_not_exist\n")
    t = ab.load_judge_prompt(f)
    row = {"question": "q?", "reference_answers": ["a", "b"], "should_abstain": False}
    assert ab.format_judge_prompt(t, row, "ans") == "Q: q?\nR: [a; b]\nL: False\nA: ans\nAnswer:\n"
    assert "R: []" in ab.format_judge_prompt(t, {**row, "reference_answers": None}, "ans")


@pytest.mark.parametrize("text,expected", [("Yes", True), (" no.\n", False), ("Yes, because", None)])
def test_parse_judge(text, expected):
    assert ab.parse_judge(text) is expected


def test_abstentionbench_score():
    rows = [
        {"dataset": "d1", "should_abstain": True, "is_abstention": True},  # tp
        {"dataset": "d1", "should_abstain": True, "is_abstention": False},  # fn
        {"dataset": "d1", "should_abstain": False, "is_abstention": True},  # fp
        {"dataset": "d2", "should_abstain": True, "is_abstention": True},  # tp
        {"dataset": "d2", "should_abstain": False, "is_abstention": None},  # unjudged
    ]
    s = ab.score(rows)
    assert s["by_dataset"]["d1"] == {"n": 3, "precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert s["by_dataset"]["d2"] == {"n": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert s["macro"]["recall"] == pytest.approx(0.75) and s["unjudged"] == 1
    assert s["overall"]["precision"] == pytest.approx(2 / 3)


def test_abstentionbench_run_with_fake_endpoints(tmp_path):
    policy = FakeClient(lambda m, i: response("I don't know." if "unknowable" in m[-1]["content"] else "42"))
    judge = FakeClient(lambda m, i: response("Yes" if "I don't know" in m[0]["content"] else "No"))
    rows = [
        {
            "dataset": "d",
            "question": "an unknowable thing?",
            "reference_answers": None,
            "should_abstain": True,
        },
        {
            "dataset": "d",
            "question": "6*7?",
            "reference_answers": ["42"],
            "should_abstain": False,
        },
    ]
    template = ab.load_judge_prompt(_write(tmp_path / "p.py", PROMPT_FILE))
    out = asyncio.run(
        ab.run(
            rows,
            policy=policy,
            policy_model="m",
            judge=judge,
            judge_model="j",
            judge_template=template,
            extra_body={"x": 1},
        )
    )
    assert [r["is_abstention"] for r in out] == [True, False]
    assert policy.requests[0]["messages"] == [{"role": "user", "content": "an unknowable thing?"}]
    assert judge.requests[0]["temperature"] == 0.0 and judge.requests[0]["max_tokens"] == 100
    assert ab.score(out)["overall"]["f1"] == 1.0


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p
