"""honest_agents.teacher: the mailbox stand-in model, token counting, selection and the client protocol (no sandbox)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from honest_agents import teacher


def _rows():
    return [
        {"variant_id": "a__feasible", "pair_id": "src/a"},
        {"variant_id": "a__high_risk", "pair_id": "src/a"},
        {"variant_id": "b__feasible", "pair_id": "src/b"},
        {"variant_id": "c__feasible", "pair_id": "src/c"},
    ]


def test_select_variants_keeps_validated_rows_outside_the_excluded_sources():
    validation = [
        {"variant_id": "a__feasible", "ok": True},
        {"variant_id": "a__high_risk", "ok": True, "leak": True},
        {"variant_id": "b__feasible", "ok": True},
        {"variant_id": "c__feasible", "ok": False},
    ]
    got = teacher.select_variants(_rows(), validation, exclude_sources={"src/b"})
    assert [r["variant_id"] for r in got] == ["a__feasible"]


def test_tokens_are_opaque_and_salted():
    t1 = teacher.make_token("x__high_risk", "s1")
    assert "high_risk" not in t1 and t1 != teacher.make_token("x__high_risk", "s2")


def test_template_messages_parse_string_arguments():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
             "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}]  # fmt: skip
    out = teacher._template_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], str)  # the input is not mutated


def test_mailbox_client_round_trip(tmp_path: Path):
    client = teacher.MailboxClient(tmp_path, counter=None, act_timeout=10)

    def teacher_side():
        while not (tmp_path / "obs-0.json").exists():
            time.sleep(0.01)
        (tmp_path / "act-0.json").write_text(
            json.dumps({"content": "look", "tool": "bash", "arguments": {"command": "ls"}})
        )

    threading.Thread(target=teacher_side).start()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    resp = asyncio.run(client.chat.completions.create(messages=messages, tools=[]))
    msg = resp.choices[0].message
    assert msg.content == "look" and resp.choices[0].finish_reason == "tool_calls"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"command": "ls"}
    assert json.loads((tmp_path / "obs-0.json").read_text())["messages"] == messages


def test_mailbox_client_times_out_when_the_teacher_goes_quiet(tmp_path: Path):
    client = teacher.MailboxClient(tmp_path, counter=None, act_timeout=0.2)
    with pytest.raises(TimeoutError):
        asyncio.run(client.chat.completions.create(messages=[{"role": "user", "content": "x"}]))


def test_step_sends_exactly_one_action_per_turn(tmp_path: Path):
    lay = teacher.Layout(tmp_path)
    box = lay.mailbox("t1")
    box.mkdir(parents=True)
    (box / "obs-0.json").write_text(json.dumps({"seq": 0, "messages": []}))
    (box / "act-0.json").write_text("{}")  # another client already acted on this turn
    out = teacher._step(lay, "t1", {"content": "", "tool": "bash", "arguments": {"command": "ls"}}, timeout=0.1)
    assert out.startswith("error: an action for this turn is already pending")
    assert (box / "act-0.json").read_text() == "{}"


def test_token_counter_counts_real_tokens():
    try:
        counter = teacher.TokenCounter()
    except Exception as e:  # noqa: BLE001 - the tokenizer is not cached on this machine
        pytest.skip(f"tokenizer unavailable: {e}")
    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.harness import SYSTEM_PROMPT

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "List the files."}]
    n = counter.count(msgs, tool_schemas(), generation_prompt=True)
    assert n > 500  # the system prompt and tool schemas alone are well over this; 2 was the BatchEncoding key count
