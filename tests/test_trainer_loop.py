"""Differential tests of `HonestGRPOTrainer._tool_call_loop` against TRL's own serial `_tool_call_loop`.

What is stubbed, and nothing else:

* the trainer object is `object.__new__(HonestGRPOTrainer)` with only the attributes the loop reads set by hand
  (`max_tool_calling_iterations`, `max_completion_length`, `use_vllm`, `vllm_mode`, `model.config`, `_is_vlm`,
  `tools`, `_tokenizer`, `environments`, `_sync_tool_dicts`, `_async_tool_dicts`, `async_loop`). No model, tokenizer,
  accelerator or vLLM is loaded and `GRPOTrainer.__init__` is not run;
* `_generate_single_turn` is a scripted fake generator (token ids and logprobs are a pure function of the episode and
  its turn number, so the result does not depend on who else is in the batch);
* `_get_tool_suffix_ids` is a deterministic function of the tool messages;
* `parse_response` (patched in `trl.trainer.grpo_trainer` and in `honest_agents.trainer`) looks the message up by
  the turn's first token id.

Both loops run on the same stubbed object: TRL's through `GRPOTrainer._tool_call_loop(trainer, ...)`, ours through
the subclass. Everything else is the real code: the loop bodies, the tool dispatch, the rollback, the truncation and
the counters.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("trl")

import torch
from trl import GRPOTrainer
from trl.trainer.utils import split_tensor_dict

import honest_agents.trainer as trainer_mod
from honest_agents.reward import (
    STOP_CALL_BUDGET,
    STOP_CHAR_BUDGET,
    STOP_REPORT,
    STOP_SILENT,
    STOP_TRUNCATED,
)
from honest_agents.trainer import HonestGRPOTrainer

EOS, PAD = 2, 0
MAX_POSITIONS = 10**9


# ── fakes ──────────────────────────────────────────────────────────────────


def call(name: str, **arguments: Any) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


@dataclass
class Turn:
    n_tokens: int = 4
    calls: list[dict] | None = None
    truncated: bool = False  # the turn ends without EOS (cut by the length limit)


class FakeEnv:
    """The slice of an environment the loop touches: public methods are tools, `_report` marks the final report."""

    def __init__(
        self,
        ident: int,
        *,
        delay: float = 0.0,
        barrier: threading.Barrier | None = None,
    ):
        self.ident = ident
        self.delay = delay
        self.barrier = barrier
        self._report: dict | None = None
        self._report_error: str | None = None
        self.log: list[tuple] = []
        self.thread_names: set[str] = set()
        self.exit_reason: str | None = None

    def bash(self, command: str):
        self.thread_names.add(threading.current_thread().name)
        self.log.append(("bash", command))
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        if self.delay:
            time.sleep(self.delay)
        if command == "boom":
            raise ValueError("boom: command failed")
        if command.startswith("out:"):
            return "x" * int(command[4:])
        return f"ran {command!r} in env {self.ident}"

    async def aread(self, path: str):
        self.log.append(("aread", path))
        await asyncio.sleep(0)
        if path == "missing":
            raise FileNotFoundError(path)
        return f"contents of {path}"

    def submit_report(self, status: str):
        self.log.append(("submit_report", status))
        if self._report is not None:
            return "report already recorded"
        self._report = {"status": status}
        return "report recorded"

    def _set_loop_exit(self, reason: str) -> None:
        self.exit_reason = reason

    def reset(self, **row):  # excluded from the tool list, as in TRL
        return None


def first_token(episode: int, turn: int) -> int:
    return 1000 + episode * 100 + turn


class FakeGenerator:
    """Scripted `_generate_single_turn`. Turn t of episode e is a function of (e, t) only."""

    def __init__(self, scripts: dict[int, list[Turn]]):
        self.scripts = scripts
        self.messages: dict[int, dict] = {}
        self.rounds: list[list[int]] = []  # episodes generated for, per call, in call order
        self.seen_turns: list[tuple[int, int]] = []

    def turn_tokens(self, episode: int, turn: int) -> tuple[list[int], list[float]]:
        spec = self.scripts[episode][turn]
        ids = [first_token(episode, turn)] + [10 + (i * 7 + turn) % 30 for i in range(spec.n_tokens - 1)]
        if not spec.truncated:
            ids.append(EOS)
        logprobs = [-0.013 * (i + 1) - 0.1 * episode - 0.001 * turn for i in range(len(ids))]
        self.messages[first_token(episode, turn)] = {
            "role": "assistant",
            "content": f"e{episode}t{turn}",
            **({"tool_calls": copy.deepcopy(spec.calls)} if spec.calls else {}),
        }
        return ids, logprobs

    def initial(self, episodes: list[int]):
        ids, lps = zip(*(self.turn_tokens(e, 0) for e in episodes), strict=True)
        return [list(i) for i in ids], [list(lp) for lp in lps]

    def __call__(
        self,
        prompt_ids,
        images,
        multimodal_fields,
        num_generations,
        has_tool_images=False,
    ):
        assert num_generations == 1
        out_ids, out_lps, episode_ids = [], [], []
        for pid in prompt_ids:
            episode = pid[0] - 100
            turn = sum(1 for t in pid if first_token(episode, 0) <= t < first_token(episode, 0) + 100)
            self.seen_turns.append((episode, turn))
            ids, lps = self.turn_tokens(episode, turn)
            out_ids.append(ids)
            out_lps.append(lps)
            episode_ids.append(episode)
        self.rounds.append(episode_ids)
        return out_ids, out_lps


def fake_parse_response(tokenizer, ids, prefix=None):
    return copy.deepcopy(tokenizer.generator.messages[ids[0]])


def fake_suffix_ids(tool_messages):
    return [900] + [901] * (sum(len(str(m["content"])) for m in tool_messages) // 8) + [902]


@dataclass
class Result:
    output: tuple
    prompts: list
    envs: list[FakeEnv]
    generator: FakeGenerator
    trainer: HonestGRPOTrainer


@dataclass
class Scenario:
    scripts: dict[int, list[Turn]]
    max_iterations: int = 30
    max_completion_length: int = 10_000
    delay: float = 0.0


_async_loop: asyncio.AbstractEventLoop | None = None


def async_loop() -> asyncio.AbstractEventLoop:
    global _async_loop
    if _async_loop is None:
        _async_loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(_async_loop)
            _async_loop.call_soon(ready.set)
            _async_loop.run_forever()

        threading.Thread(target=run, daemon=True, name="test-async-loop").start()
        ready.wait()
    return _async_loop


def make_trainer(generator: FakeGenerator, envs: list[FakeEnv], scenario: Scenario) -> HonestGRPOTrainer:
    """The smallest stub that runs the loop: see the module docstring."""
    t = object.__new__(HonestGRPOTrainer)
    t.max_tool_calling_iterations = scenario.max_iterations
    t.max_completion_length = scenario.max_completion_length
    t.use_vllm = False
    t.vllm_mode = "server"
    t.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=MAX_POSITIONS))
    t._is_vlm = False
    t.tools = [lambda: None]  # truthy: the loop is only entered when `self.tools`
    t._tokenizer = SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD, generator=generator)
    t.async_loop = async_loop()
    t.environments = envs
    t._sync_tool_dicts, t._async_tool_dicts = [], []
    import inspect

    for env in envs:  # the same dicts TRL's `_generate_and_score_completions` builds
        methods = [
            m
            for n, m in inspect.getmembers(env, predicate=inspect.ismethod)
            if n not in ("reset", "get_reward") and not n.startswith("_")
        ]
        sync, asyn = {}, {}
        for tool in methods:
            (asyn if inspect.iscoroutinefunction(tool) else sync)[tool.__name__] = tool
        t._sync_tool_dicts.append(sync)
        t._async_tool_dicts.append(asyn)
    t._generate_single_turn = generator
    t._get_tool_suffix_ids = fake_suffix_ids
    return t


@pytest.fixture(autouse=True)
def _patch_parse(monkeypatch, request):
    if request.node.name.startswith("test_real_"):  # the end-to-end tests use the real tokenizer's parser
        return
    monkeypatch.setattr("trl.trainer.grpo_trainer.parse_response", fake_parse_response)
    monkeypatch.setattr(trainer_mod, "parse_response", fake_parse_response)


def run_loop(
    scenario: Scenario,
    *,
    serial: bool,
    stop_at_report: bool = True,
    max_workers: int | None = None,
    barrier: threading.Barrier | None = None,
) -> Result:
    episodes = sorted(scenario.scripts)
    generator = FakeGenerator(scenario.scripts)
    envs = [FakeEnv(e, delay=scenario.delay, barrier=barrier) for e in episodes]
    trainer = make_trainer(generator, envs, scenario)
    trainer.stop_at_report = stop_at_report
    trainer.max_tool_workers = max_workers
    prompt_ids = [[100 + e, 5, 6] for e in episodes]
    prompts = [[{"role": "system", "content": "sys"}, {"role": "user", "content": f"task {e}"}] for e in episodes]
    completion_ids, logprobs = generator.initial(episodes)
    generator.rounds.clear()
    completions = [[copy.deepcopy(generator.messages[first_token(e, 0)])] for e in episodes]
    args = (prompts, prompt_ids, completion_ids, completions, logprobs, None, {})
    if serial:
        output = GRPOTrainer._tool_call_loop(trainer, *args)
    else:
        output = trainer._tool_call_loop(*args)
    return Result(output, prompts, envs, generator, trainer)


FIELDS = (
    "tool_mask",
    "completions",
    "completion_ids",
    "logprobs",
    "tool_call_count",
    "tool_failure_count",
    "images",
)


def assert_same(ours: Result, ref: Result, rows: list[int] | None = None) -> None:
    for name, a, b in zip(FIELDS, ours.output, ref.output, strict=True):
        if rows is not None and isinstance(a, list) and name not in ("tool_call_count", "tool_failure_count"):
            a, b = [a[i] for i in rows], [b[i] for i in rows]
        assert a == b, f"{name} differs"
    if rows is None:
        assert ours.prompts == ref.prompts, "prompts (mutated in place by the loop) differ"
    for i, (ea, eb) in enumerate(zip(ours.envs, ref.envs, strict=True)):
        if rows is None or i in rows:
            assert ea.log == eb.log, f"env {i} saw different tool calls"


# ── scenarios ──────────────────────────────────────────────────────────────

BASH = lambda c: call("bash", command=c)


def mixed_scenario() -> Scenario:
    """Unequal lengths, a tool error, an unknown tool, an unsupported call type, async tools, a cut last turn."""
    return Scenario(
        {
            0: [Turn(5)],  # stops silently on turn 0
            1: [Turn(4, [BASH("ls")]), Turn(6)],  # one tool turn then stops
            2: [  # three tool turns, two calls in one (sync + async + sync: results are ordered sync-first)
                Turn(3, [BASH("a")]),
                Turn(7, [call("aread", path="/x"), BASH("b"), BASH("c")]),
                Turn(2, [BASH("d")]),
                Turn(8),
            ],
            3: [
                Turn(4, [BASH("boom")]),
                Turn(5, [call("aread", path="missing")]),
                Turn(3),
            ],
            4: [Turn(4, [call("nope", q=1)]), Turn(3)],
            5: [Turn(4, [{"type": "mystery", "name": "weird"}]), Turn(3)],
            6: [
                Turn(4, [BASH("x")]),
                Turn(9, truncated=True),
            ],  # last turn cut by the length limit
            7: [Turn(3, [BASH(f"c{i}")]) for i in range(40)],  # still calling tools when the cap is reached
            8: [
                Turn(4, [BASH("bad-args")]),
                Turn(2, [call("bash", wrong=1)]),
                Turn(3),
            ],  # TypeError in the tool
        },
        max_iterations=5,
    )


@pytest.mark.parametrize("stop_at_report", [True, False])
@pytest.mark.parametrize("workers", [None, 1, 3])
def test_identical_to_trl_serial_loop_unequal_episodes(stop_at_report, workers, monkeypatch):
    # Deliberately changed (L3): our loop shows the model a fixed error string where TRL's shows `str(e)`. This test is
    # about scheduling / masking equivalence, so the text is aligned here; the fixed strings have their own tests.
    monkeypatch.setattr(trainer_mod, "_tool_error_text", str)
    scenario = mixed_scenario()
    ref = run_loop(scenario, serial=True)
    ours = run_loop(scenario, serial=False, stop_at_report=stop_at_report, max_workers=workers)
    assert_same(ours, ref)
    mask, completions, ids, lps, calls, failures, _ = ours.output
    # the scenario really exercises what it claims to
    assert len({len(m) for m in mask}) > 4
    assert any(0 in m for m in mask) and any(0.0 in lp for lp in lps)
    assert failures >= 5  # boom, missing file, unknown tool, unsupported type, bad arguments
    assert calls == ref.output[4]
    assert all(len(m) == len(i) == len(lp) for m, i, lp in zip(mask, ids, lps, strict=True))
    # the capped episode left with its tool calls unexecuted
    assert len(ours.envs[7].log) == 5
    # tool error text reaches the model unchanged
    tool_texts = [m["content"] for m in completions[3] if m["role"] == "tool"]
    assert any("boom: command failed" in t for t in tool_texts)


def test_tool_message_order_matches_trl_sync_then_async():
    scenario = mixed_scenario()
    ours = run_loop(scenario, serial=False)
    tools = [m["name"] for m in ours.output[1][2] if m["role"] == "tool"]
    assert tools == ["bash", "bash", "bash", "aread", "bash"]


def test_ids_and_logprobs_are_the_generators_own_objects():
    """Sampled ids and logprobs are concatenated and sliced, never rebuilt: the values are bit-identical."""
    scenario = Scenario({0: [Turn(4, [BASH("a")]), Turn(5, [BASH("b")]), Turn(6)]})
    ours = run_loop(scenario, serial=False)
    gen = FakeGenerator(scenario.scripts)
    t0, l0 = gen.turn_tokens(0, 0)
    t1, l1 = gen.turn_tokens(0, 1)
    t2, l2 = gen.turn_tokens(0, 2)
    suffix0 = fake_suffix_ids([{"content": "ran 'a' in env 0"}])
    suffix1 = fake_suffix_ids([{"content": "ran 'b' in env 0"}])
    ids = t0 + suffix0 + t1 + suffix1 + t2
    mask = [1] * len(t0) + [0] * len(suffix0) + [1] * len(t1) + [0] * len(suffix1) + [1] * len(t2)
    lps = l0 + [0.0] * len(suffix0) + l1 + [0.0] * len(suffix1) + l2
    out_mask, _, out_ids, out_lps, n_calls, n_fail, _ = ours.output
    assert (out_ids[0], out_mask[0], out_lps[0]) == (ids, mask, lps)
    assert all(type(x) is float for x in out_lps[0])
    assert (n_calls, n_fail) == (2, 0)


def test_overlong_tool_result_is_rolled_back_like_trl():
    scenario = Scenario(
        {
            0: [Turn(4, [BASH("out:50")]), Turn(5)],
            1: [
                Turn(4, [BASH("out:4000")]),
                Turn(5),
            ],  # result does not fit max_completion_length
            2: [
                Turn(4, [BASH("out:40")]),
                Turn(5, [BASH("out:4000")]),
                Turn(5),
            ],  # fits first, not second
        },
        max_completion_length=300,
    )
    ref = run_loop(scenario, serial=True)
    ours = run_loop(scenario, serial=False)
    assert_same(ours, ref)
    # the rolled-back rollout keeps its completion as sampled and has no trailing tool message
    assert ours.output[2][1] == ref.output[2][1]
    assert ours.output[1][1][-1]["role"] == "assistant"
    assert ours.trainer._loop_record.exits == [
        STOP_SILENT,
        STOP_CHAR_BUDGET,
        STOP_CHAR_BUDGET,
    ]


def test_post_tool_turn_truncated_to_the_completion_budget_like_trl():
    scenario = Scenario(
        {
            0: [Turn(10, [BASH("a")]), Turn(400)],  # the second turn alone overflows
            1: [Turn(10, [BASH("a")]), Turn(20)],
        },
        max_completion_length=120,
    )
    ref = run_loop(scenario, serial=True)
    ours = run_loop(scenario, serial=False)
    assert_same(ours, ref)
    assert max(len(i) for i in ours.output[2]) <= 120
    assert len(ours.output[2][0]) <= 120
    assert ours.trainer._loop_record.exits[0] == STOP_TRUNCATED  # cut turn has no EOS


def test_iteration_cap_leaves_pending_tool_calls_unexecuted():
    scenario = Scenario({0: [Turn(3, [BASH(f"c{i}")]) for i in range(10)]}, max_iterations=3)
    ref = run_loop(scenario, serial=True)
    ours = run_loop(scenario, serial=False)
    assert_same(ours, ref)
    assert ours.envs[0].log == [("bash", "c0"), ("bash", "c1"), ("bash", "c2")]
    assert ours.trainer._loop_record.exits == [STOP_CALL_BUDGET]


def test_no_tool_calls_at_all():
    scenario = Scenario({0: [Turn(3)], 1: [Turn(5)]})
    assert_same(run_loop(scenario, serial=False), run_loop(scenario, serial=True))


def test_stop_at_report_false_with_reports_equals_trl():
    scenario = Scenario(
        {
            0: [Turn(4, [call("submit_report", status="complete")]), Turn(5)],
            1: [Turn(3)],
        }
    )
    ref = run_loop(scenario, serial=True)
    ours = run_loop(scenario, serial=False, stop_at_report=False)
    assert_same(ours, ref)


# ── an episode that has reported is not regenerated ─────────────────────────


def report_scenario() -> Scenario:
    return Scenario(
        {
            0: [
                Turn(4, [BASH("a")]),
                Turn(5, [call("submit_report", status="complete")]),
                Turn(6),
            ],  # reports turn 1
            1: [
                Turn(4, [BASH("a")]),
                Turn(5, [BASH("b")]),
                Turn(5, [BASH("c")]),
                Turn(7),
            ],  # never reports
            2: [
                Turn(4, [call("submit_report", status="incomplete")]),
                Turn(5),
            ],  # reports on turn 0
            3: [
                Turn(3, [BASH("a")]),
                Turn(3, [BASH("b")]),
                Turn(3, [call("submit_report", status="x")]),
                Turn(2),
            ],
            4: [Turn(5)],  # silent
        }
    )


def expected_up_to_report(scenario: Scenario, episode: int, report_turn: int):
    gen = FakeGenerator(scenario.scripts)
    ids, mask, lps = [], [], []
    for t in range(report_turn + 1):
        tid, tlp = gen.turn_tokens(episode, t)
        ids += tid
        mask += [1] * len(tid)
        lps += tlp
        if t < report_turn:
            content = [m for m in scenario.scripts[episode][t].calls]
            # suffix length depends on the tool message text the env returned
            text = f"ran {content[0]['function']['arguments']['command']!r} in env {episode}"
            suffix = fake_suffix_ids([{"content": text}])
            ids += suffix
            mask += [0] * len(suffix)
            lps += [0.0] * len(suffix)
    return ids, mask, lps


def test_reported_episode_is_not_regenerated_and_ends_at_the_report_turn():
    scenario = report_scenario()
    ref = run_loop(scenario, serial=True)
    ours = run_loop(scenario, serial=False)
    # the generator is asked for turn t+1 after the report in TRL's loop, never in ours
    assert (0, 2) in ref.generator.seen_turns and (2, 1) in ref.generator.seen_turns
    assert (3, 3) in ref.generator.seen_turns
    for episode, report_turn in ((0, 1), (2, 0), (3, 2)):
        assert not any(e == episode and t > report_turn for e, t in ours.generator.seen_turns)
    # fewer tokens sampled: that is the point
    assert sum(map(len, ours.generator.rounds)) < sum(map(len, ref.generator.rounds))

    mask, completions, ids, lps, calls, failures, _ = ours.output
    for episode, report_turn in ((0, 1), (2, 0), (3, 2)):
        want_ids, want_mask, want_lps = expected_up_to_report(scenario, episode, report_turn)
        assert (ids[episode], mask[episode], lps[episode]) == (
            want_ids,
            want_mask,
            want_lps,
        )
        assert ids[episode][-1] == EOS and mask[episode][-1] == 1
        last = completions[episode][-1]
        assert last["role"] == "assistant" and last["tool_calls"][0]["function"]["name"] == "submit_report"
        assert ours.envs[episode]._report is not None  # the report was executed (the env has it) ...
        assert sum(1 for m in completions[episode] if m["role"] == "tool") == report_turn  # ... but not appended
    # episodes that did not report are untouched
    assert_same(ours, ref, rows=[1, 4])
    # the report calls still count as tool calls: 2 + 3 + 1 + 3 + 0
    assert calls == 9 and failures == 0
    assert ours.trainer._loop_record.exits == [
        STOP_REPORT,
        STOP_SILENT,
        STOP_REPORT,
        STOP_REPORT,
        STOP_SILENT,
    ]


def test_prompts_are_rolled_back_for_reported_episodes():
    scenario = report_scenario()
    ours = run_loop(scenario, serial=False)
    # episode 2 reported on its first turn: the loop's own prompt copy ends before the assistant message was added
    assert ours.prompts[2] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task 2"},
    ]


def test_report_error_also_ends_the_episode():
    class ErrEnv(FakeEnv):
        def submit_report(self, status: str):
            self._report_error = "unparseable"
            return "error"

    scenario = Scenario({0: [Turn(4, [call("submit_report", status="???")]), Turn(5)]})
    generator = FakeGenerator(scenario.scripts)
    envs = [ErrEnv(0)]
    trainer = make_trainer(generator, envs, scenario)
    gen_ids, gen_lps = generator.initial([0])
    out = trainer._tool_call_loop(
        [[{"role": "user", "content": "t"}]],
        [[100, 5, 6]],
        gen_ids,
        [[copy.deepcopy(generator.messages[first_token(0, 0)])]],
        gen_lps,
        None,
        {},
    )
    assert generator.rounds == []
    assert out[2] == gen_ids and trainer._loop_record.exits == [STOP_REPORT]


def test_batch_without_environments_runs_tools_serially_like_trl():
    scenario = Scenario({0: [Turn(3, [BASH("a")]), Turn(3)], 1: [Turn(3)]})
    generator = FakeGenerator(scenario.scripts)
    envs = [FakeEnv(0), FakeEnv(1)]
    trainer = make_trainer(generator, envs, scenario)
    trainer.environments = None  # standalone tools only
    shared = []
    trainer._sync_tool_dicts = [
        {"bash": lambda command: shared.append(command) or "ok"},
        {},
    ]
    trainer._async_tool_dicts = [{}, {}]
    ids, lps = generator.initial([0, 1])
    completions = [[copy.deepcopy(generator.messages[first_token(e, 0)])] for e in (0, 1)]
    prompts = [[{"role": "user", "content": "t"}] for _ in range(2)]
    out = trainer._tool_call_loop(prompts, [[100, 5, 6], [101, 5, 6]], ids, completions, lps, None, {})
    assert shared == ["a"] and out[4] == 1 and [m["role"] for m in out[1][0]] == ["assistant", "tool", "assistant"]


# ── tool calls really overlap ──────────────────────────────────────────────


def overlap_scenario(n: int) -> Scenario:
    return Scenario({e: [Turn(3, [BASH("go")]), Turn(3)] for e in range(n)})


def test_tool_calls_of_different_episodes_overlap_in_time():
    n = 8
    # every bash call waits for all n callers: only possible if the n calls are in flight at the same time
    barrier = threading.Barrier(n)
    start = time.perf_counter()
    ours = run_loop(overlap_scenario(n), serial=False, barrier=barrier)
    assert not barrier.broken
    assert ours.output[5] == 0  # no tool failed (a broken barrier would have raised inside a tool)
    assert len({name for env in ours.envs for name in env.thread_names}) == n
    assert time.perf_counter() - start < 4


def test_trl_serial_loop_cannot_pass_the_overlap_test():
    """Control: the same barrier test fails on TRL's loop, so the test above does detect serial execution."""
    n = 3
    barrier = threading.Barrier(n)
    barrier_short = threading.Barrier(n, timeout=0.2)
    barrier_short.wait = lambda timeout=None, b=barrier_short: threading.Barrier.wait(b, timeout=0.2)  # type: ignore
    ref = run_loop(overlap_scenario(n), serial=True, barrier=barrier_short)
    assert ref.output[5] == n  # every call failed with BrokenBarrierError
    del barrier


def test_wall_clock_is_the_slowest_call_not_the_sum():
    n, delay = 8, 0.25
    scenario = overlap_scenario(n)
    scenario.delay = delay
    start = time.perf_counter()
    ours = run_loop(scenario, serial=False)
    elapsed = time.perf_counter() - start
    assert elapsed < delay * n * 0.5, f"{elapsed:.2f}s looks serial"
    record = ours.trainer._loop_record
    assert record.tool_seconds < delay * n * 0.5
    assert all(s >= delay for s in record.episode_tool_seconds)
    assert record.slowest_tool_turn >= delay


def test_max_tool_workers_caps_concurrency():
    n = 6
    active, peak, lock = 0, 0, threading.Lock()

    class CountingEnv(FakeEnv):
        def bash(self, command: str):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return "ok"

    scenario = overlap_scenario(n)
    generator = FakeGenerator(scenario.scripts)
    envs = [CountingEnv(e) for e in range(n)]
    trainer = make_trainer(generator, envs, scenario)
    trainer.max_tool_workers = 2
    ids, lps = generator.initial(list(range(n)))
    completions = [[copy.deepcopy(generator.messages[first_token(e, 0)])] for e in range(n)]
    trainer._tool_call_loop(
        [[{"role": "user", "content": "t"}] for _ in range(n)],
        [[100 + e, 5, 6] for e in range(n)],
        ids,
        completions,
        lps,
        None,
        {},
    )
    assert peak == 2


def test_calls_within_one_episode_stay_ordered():
    scenario = Scenario(
        {
            0: [Turn(3, [BASH("1"), BASH("2"), BASH("3")]), Turn(3)],
            1: [Turn(3, [BASH("z")]), Turn(3)],
        }
    )
    ours = run_loop(scenario, serial=False)
    assert [c for n, c in ours.envs[0].log] == ["1", "2", "3"]
    assert [m["content"] for m in ours.output[1][0] if m["role"] == "tool"] == [
        "ran '1' in env 0",
        "ran '2' in env 0",
        "ran '3' in env 0",
    ]


def test_a_crashing_tool_does_not_abort_other_episodes_and_is_reported_to_the_model():
    class Crash(FakeEnv):
        def bash(self, command: str):
            raise RuntimeError("sandbox died")

    scenario = Scenario({0: [Turn(3, [BASH("a")]), Turn(3)], 1: [Turn(3, [BASH("a")]), Turn(3)]})
    generator = FakeGenerator(scenario.scripts)
    envs = [Crash(0), FakeEnv(1)]
    trainer = make_trainer(generator, envs, scenario)
    ids, lps = generator.initial([0, 1])
    completions = [[copy.deepcopy(generator.messages[first_token(e, 0)])] for e in range(2)]
    out = trainer._tool_call_loop(
        [[{"role": "user", "content": "t"}] for _ in range(2)],
        [[100, 5, 6], [101, 5, 6]],
        ids,
        completions,
        lps,
        None,
        {},
    )
    assert out[5] == 1
    # deliberately changed (L3): the model gets a fixed string, never the exception text
    assert out[1][0][1]["content"] == str({"error": "[sandbox unavailable]"})
    assert "sandbox died" not in out[1][0][1]["content"]
    assert out[1][1][1]["content"] == "ran 'a' in env 1"


# ── guards and bookkeeping ─────────────────────────────────────────────────


def test_loop_exits_cover_every_reason_with_the_right_precedence():
    scenario = Scenario(
        {
            0: [Turn(3, [call("submit_report", status="complete")]), Turn(3)],  # report
            1: [Turn(3)],  # silent
            2: [Turn(3, [BASH("a")]), Turn(5, truncated=True)],  # truncated
            3: [Turn(3, [BASH(f"c{i}")]) for i in range(10)],  # call budget
            4: [Turn(3, [BASH("out:9000")]), Turn(3)],  # char budget
        },
        max_iterations=4,
        max_completion_length=200,
    )
    ours = run_loop(scenario, serial=False)
    assert ours.trainer._loop_record.exits == [
        STOP_REPORT,
        STOP_SILENT,
        STOP_TRUNCATED,
        STOP_CALL_BUDGET,
        STOP_CHAR_BUDGET,
    ]


def test_rejects_multi_process_vllm_server_with_tools():
    t = object.__new__(HonestGRPOTrainer)
    t.use_vllm, t.vllm_mode, t.tools = True, "server", [lambda: None]
    t.max_completion_length = 16384
    t.args = SimpleNamespace(generation_kwargs={"max_tokens": 4096})
    t.accelerator = SimpleNamespace(num_processes=2)
    with pytest.raises(ValueError, match="one process per run"):
        t._check_generation_layout()
    t.accelerator = SimpleNamespace(num_processes=1)
    t._check_generation_layout()
    t.vllm_mode = "colocate"
    t.accelerator = SimpleNamespace(num_processes=2)
    t.vllm_tensor_parallel_size = 1
    t._check_generation_layout()  # colocated engines per rank are independent


@pytest.mark.parametrize(
    "generation_kwargs", [None, {}, {"max_tokens": None}, {"max_tokens": 0}, {"max_tokens": 20000}]
)
def test_vllm_server_with_tools_requires_a_per_turn_cap(generation_kwargs):
    """TDC-2: without `generation_kwargs.max_tokens` TRL asks the server for the whole budget on every turn."""
    t = object.__new__(HonestGRPOTrainer)
    t.use_vllm, t.vllm_mode, t.tools, t.max_completion_length = True, "server", [lambda: None], 16384
    t.accelerator = SimpleNamespace(num_processes=1)
    t.args = SimpleNamespace(generation_kwargs=generation_kwargs)
    with pytest.raises(ValueError, match="per-turn cap"):
        t._check_generation_layout()
    t.args = SimpleNamespace(generation_kwargs={"max_tokens": 4096})
    t._check_generation_layout()
    t.vllm_mode = "colocate"  # the offline engine truncates at max_model_len instead of rejecting the request
    t.args = SimpleNamespace(generation_kwargs=None)
    t._check_generation_layout()


def test_upstream_fingerprints_match_the_installed_trl():
    trainer_mod.check_upstream()
    assert trainer_mod.upstream_fingerprints() == trainer_mod.UPSTREAM_FINGERPRINTS


def test_check_upstream_fails_loudly_when_trl_changes(monkeypatch):
    monkeypatch.setitem(trainer_mod.UPSTREAM_FINGERPRINTS, "_tool_call_loop", "0" * 64)
    with pytest.raises(RuntimeError, match="_tool_call_loop"):
        trainer_mod.check_upstream()


def test_exit_reason_is_offered_to_the_environments():
    scenario = Scenario({0: [Turn(3)], 1: [Turn(3, [BASH("a")]), Turn(4, truncated=True)]})
    ours = run_loop(scenario, serial=False)
    ours.trainer._tell_environments(ours.trainer._loop_record.exits)
    assert [e.exit_reason for e in ours.envs] == [STOP_SILENT, STOP_TRUNCATED]


def test_calculate_rewards_adds_token_kwargs_per_row(monkeypatch):
    seen = {}

    def fake_super(self, inputs, prompts, completions, completion_ids_list):
        seen["inputs"] = inputs
        return "rewards"

    monkeypatch.setattr(GRPOTrainer, "_calculate_rewards", fake_super)
    t = object.__new__(HonestGRPOTrainer)
    t._rollout_tokens = {
        "prompt_ids": [[1], [2]],
        "logprobs": [[-1.0], [-2.0]],
        "tool_mask": [[1], [1]],
    }
    rows = [{"prompt": "p", "variant_id": "a"}, {"prompt": "p", "variant_id": "b"}]
    assert t._calculate_rewards(rows, [], [], []) == "rewards"
    assert [list(r["prompt_ids"]) for r in seen["inputs"]] == [[1], [2]]
    assert [list(r["logprobs"]) for r in seen["inputs"]] == [[-1.0], [-2.0]]
    assert "prompt_ids" not in rows[0]  # the dataset rows are not mutated
    assert repr(seen["inputs"][0]["prompt_ids"]) == "<1 values>"
    assert "_rollout_tokens" not in t.__dict__


# ── weight sync (SPEC_V3 §10.1) ────────────────────────────────────────────


@dataclass
class _Dist:
    is_fsdp: bool = False
    fsdp_version: int = 0

    @contextlib.contextmanager
    def gather_params(self, params):
        yield


def _packed_consumer(named_params, chunk: int = 2):
    """Models vLLM's packed transfer: tensors of a chunk are held by reference and copied when the chunk is flushed;
    the last chunk is flushed only after the producer generator is exhausted."""
    received, pending = {}, []

    def flush():
        for name, tensor in pending:
            received[name] = tensor.clone()
        pending.clear()

    for name, tensor in named_params:
        pending.append((name, tensor))
        if len(pending) == chunk:
            flush()
    flush()  # after exhaustion: the un-merge has already run
    return received


def _lora_vllm_generation():
    peft = pytest.importorskip("peft")
    from trl.generation.vllm_generation import VLLMGeneration

    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2, self.fc3 = (torch.nn.Linear(4, 4) for _ in range(3))

        def forward(self, x):
            return self.fc3(self.fc2(self.fc1(x)))

    model = peft.get_peft_model(
        Tiny(),
        peft.LoraConfig(r=2, target_modules=["fc1", "fc2", "fc3"], init_lora_weights=False),
    )
    merged = copy.deepcopy(model)
    merged.merge_adapter()
    expected = {
        n.removeprefix("base_model.model.").replace(".base_layer", ""): p.detach().clone()
        for n, p in merged.named_parameters()
        if "lora_" not in n
    }
    vg = object.__new__(VLLMGeneration)
    vg.model, vg._dist = model, _Dist()
    return vg, expected


def test_unpatched_trl_sync_sends_unmerged_final_chunk():
    """Reproduces the TRL 1.14.1 bug (§10.1): the last packed chunk is read after the adapter was un-merged."""
    vg, expected = _lora_vllm_generation()
    received = _packed_consumer(vg._iter_named_params(), chunk=4)
    assert set(received) == set(expected)
    wrong = [n for n in expected if not torch.equal(received[n], expected[n])]
    assert wrong, "the bug did not reproduce: the stock path already sends merged weights"
    assert all("fc3" in n for n in wrong)  # the final chunk holds the last parameters


def test_cloned_sync_sends_merged_weights_for_every_chunk():
    vg, expected = _lora_vllm_generation()
    trainer_mod.clone_synced_params(vg)
    trainer_mod.clone_synced_params(vg)  # idempotent: no double wrapping
    received = _packed_consumer(vg._iter_named_params(), chunk=4)
    assert set(received) == set(expected)
    assert all(torch.equal(received[n], expected[n]) for n in expected)
    # and the model is left un-merged for training
    assert not any(
        torch.equal(p, expected[n.removeprefix("base_model.model.").replace(".base_layer", "")])
        for n, p in vg.model.named_parameters()
        if n.endswith("base_layer.weight")
    )


def test_trainer_installs_the_clone_patch_after_super_init(monkeypatch):
    vg, _ = _lora_vllm_generation()

    def fake_init(self, *a, **k):
        self.vllm_generation = vg
        self.use_vllm = False
        self.accelerator = SimpleNamespace(num_processes=1)

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_init)
    HonestGRPOTrainer()
    assert getattr(vg._iter_named_params, "_honest_clones", False)


# ── padding-free scoring (audit T1) ────────────────────────────────────────
#
# A real tiny Qwen3 and the real `GRPOTrainer._compute_loss` (dapo, token-level IS) on a stub trainer: the loss and the
# gradients with TRL's batch-global padding must equal those with trimmed, length-sorted chunks and with regrouped
# micro-batches.


def _tiny_policy():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, tie_word_embeddings=False,
    )  # fmt: skip
    return Qwen3ForCausalLM(config)


def _loss_stub(model, *, trim: bool, sort: bool, loss_type: str = "dapo", accumulation: int = 4, per_round: int = 4):
    from collections import defaultdict

    stub = object.__new__(HonestGRPOTrainer)
    gather = lambda x: x
    stub.__dict__.update(
        model=model,
        trim_padding=trim,
        sort_micro_batches=sort,
        loss_type=loss_type,
        current_gradient_accumulation_steps=accumulation,
        args=SimpleNamespace(
            gradient_accumulation_steps=accumulation,
            steps_per_generation=per_round,
            delta=None,
            use_bias_correction_kl=False,
            vllm_importance_sampling_clip_max=3.0,
            per_device_train_batch_size=2,
        ),
        accelerator=SimpleNamespace(
            num_processes=1,
            is_main_process=True,
            gather=gather,
            reduce=lambda x, reduction="sum": x,
            sync_gradients=True,
        ),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        _step=0,
        temperature=1.0,
        use_liger_kernel=False,
        model_kwarg_keys={"logits_to_keep"},
        top_entropy_quantile=1.0,
        off_policy_mask_threshold=None,
        importance_sampling_level="token",
        beta=0.0,
        epsilon_low=0.2,
        epsilon_high=0.28,
        use_vllm=True,
        vllm_importance_sampling_correction=True,
        _entropy_bonus_enabled=False,
        aux_loss_enabled=False,
        max_completion_length=16,
        top_k=0,
    )
    return stub


def _round(rows=8, prompt_width=14, completion_width=12, seed=0):
    """One generation round, padded like TRL: prompts on the left, completions on the right, unequal lengths."""
    generator = torch.Generator().manual_seed(seed)
    prompt_lengths = torch.randint(3, prompt_width + 1, (rows,), generator=generator)
    completion_lengths = torch.randint(2, completion_width + 1, (rows,), generator=generator)
    prompt_lengths[0], completion_lengths[1] = prompt_width, completion_width  # the batch-global longest
    prompt_mask = (torch.arange(prompt_width)[None] >= (prompt_width - prompt_lengths)[:, None]).long()
    completion_mask = (torch.arange(completion_width)[None] < completion_lengths[:, None]).long()
    return {
        "prompt_ids": torch.randint(1, 64, (rows, prompt_width), generator=generator) * prompt_mask,
        "prompt_mask": prompt_mask,
        "completion_ids": torch.randint(1, 64, (rows, completion_width), generator=generator) * completion_mask,
        "completion_mask": completion_mask,
        "tool_mask": torch.randint(0, 4, (rows, completion_width), generator=generator).clamp(max=1),
        "advantages": torch.randn(rows, generator=generator),
        "old_per_token_logps": -torch.rand(rows, completion_width, generator=generator) * 3,
        "importance_sampling_ratio": 0.5 + 3 * torch.rand(rows, completion_width, generator=generator),
        "num_items_in_batch": torch.tensor(float(rows * completion_width)),
    }


def _round_loss_and_grads(stub, model, round_, micro: int = 2):
    """Sum of `_compute_loss` over a round's micro-batches (as the accumulation window does) and the gradients."""
    model.zero_grad()
    batches = split_tensor_dict(dict(round_), len(round_["advantages"]) // micro)
    if stub.sort_micro_batches:
        batches = trainer_mod.sort_micro_batches(batches)
    total = 0.0
    for batch in batches:
        loss = stub._compute_loss(model, batch)
        loss.backward()
        total += float(loss.detach())
    return total, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_trimmed_sorted_loss_and_gradients_equal_trls_padded_ones():
    model = _tiny_policy()
    round_ = _round()
    stock_loss, stock_grads = _round_loss_and_grads(_loss_stub(model, trim=False, sort=False), model, round_)
    trimmed_stub = _loss_stub(model, trim=True, sort=True)
    new_loss, new_grads = _round_loss_and_grads(trimmed_stub, model, round_)
    metrics = trimmed_stub._metrics["train"]
    assert sum(metrics[trainer_mod.METRIC_PAD_TRIMMED]) < sum(metrics[trainer_mod.METRIC_PAD_STOCK])  # it did trim
    assert stock_loss == pytest.approx(new_loss, rel=1e-5, abs=1e-6)
    assert stock_grads.keys() == new_grads.keys() and stock_grads
    for name, grad in stock_grads.items():
        torch.testing.assert_close(new_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)
    assert max(float(g.abs().max()) for g in stock_grads.values()) > 1e-4  # a real signal, not equal zeros


def test_trimming_alone_keeps_the_loss_and_gradients_for_each_micro_batch_layout():
    model = _tiny_policy()
    round_ = _round(seed=3)
    for micro in (1, 2, 4):
        stock = _round_loss_and_grads(_loss_stub(model, trim=False, sort=False), model, round_, micro)
        trimmed = _round_loss_and_grads(_loss_stub(model, trim=True, sort=False), model, round_, micro)
        assert stock[0] == pytest.approx(trimmed[0], rel=1e-5, abs=1e-6)
        for name, grad in stock[1].items():
            torch.testing.assert_close(trimmed[1][name], grad, rtol=1e-4, atol=1e-6, msg=f"{name} micro={micro}")


def test_rows_with_a_fully_masked_completion_keep_the_loss_and_gradients():
    """`mask_truncated_completions` zeroes a truncated rollout's completion mask: its attended span is the prompt only."""
    model = _tiny_policy()
    round_ = _round(seed=7)
    round_["completion_mask"][[1, 4, 6]] = 0  # includes the row that held the batch-global longest completion
    stock = _round_loss_and_grads(_loss_stub(model, trim=False, sort=False), model, round_)
    trimmed = _round_loss_and_grads(_loss_stub(model, trim=True, sort=True), model, round_)
    assert stock[0] == pytest.approx(trimmed[0], rel=1e-5, abs=1e-6)
    for name, grad in stock[1].items():
        torch.testing.assert_close(trimmed[1][name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_scoring_a_whole_round_in_sorted_chunks_matches_the_padded_forward():
    """The old-logprob pass runs the whole generation batch in `batch_size` chunks; entropies come back in row order."""
    model = _tiny_policy().eval()
    round_ = _round(rows=8, seed=5)
    input_ids = torch.cat([round_["prompt_ids"], round_["completion_ids"]], dim=1)
    attention_mask = torch.cat([round_["prompt_mask"], round_["completion_mask"]], dim=1)
    keep = round_["completion_ids"].size(1)
    with torch.no_grad():
        want = _loss_stub(model, trim=False, sort=False)._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, keep, batch_size=3, compute_entropy=True
        )
        got = _loss_stub(model, trim=True, sort=True)._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, keep, batch_size=3, compute_entropy=True
        )
    valid = round_["completion_mask"].bool()
    torch.testing.assert_close(got[0][valid], want[0][valid], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[1][valid], want[1][valid], rtol=1e-4, atol=1e-5)
    assert got[0].shape == want[0].shape == round_["completion_ids"].shape and got[2] is None


def test_padded_slots_fall_on_an_unequal_length_batch():
    round_ = _round(rows=16, prompt_width=64, completion_width=48, seed=1)
    mask = torch.cat([round_["prompt_mask"], round_["completion_mask"]], dim=1)
    keep = round_["completion_mask"].size(1)
    real = int(mask.sum())
    stock = trainer_mod.padded_slots(mask)
    trimmed_only = trainer_mod.padded_slots(mask, trainer_mod.plan_chunks(mask, keep, 2, sort=False))
    sorted_ = trainer_mod.padded_slots(mask, trainer_mod.plan_chunks(mask, keep, 2, sort=True))
    assert real <= sorted_ < trimmed_only < stock
    assert stock / real > 1.5 > sorted_ / real  # the audit's 1.7-3.5x becomes close to 1


def test_regrouping_keeps_every_row_once_and_orders_micro_batches_longest_first():
    round_ = _round(rows=8, seed=2)
    original = split_tensor_dict(dict(round_), 4)
    regrouped = trainer_mod.sort_micro_batches(original)
    assert len(regrouped) == 4 and regrouped[0]["num_items_in_batch"] == round_["num_items_in_batch"]
    key = lambda b: sorted(map(tuple, torch.cat([b["prompt_ids"], b["completion_ids"]], 1).tolist()))
    assert sorted(itertools.chain(*map(key, original))) == sorted(itertools.chain(*map(key, regrouped)))
    lengths = [int((b["prompt_mask"].sum(1) + b["completion_mask"].sum(1)).sum()) for b in regrouped]
    assert lengths == sorted(lengths, reverse=True)
    for batch in regrouped:  # rows stay aligned across the keys
        rebuilt = {tuple(r) for r in torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1).tolist()}
        for i, row in enumerate(torch.cat([batch["prompt_ids"], batch["completion_ids"]], 1).tolist()):
            j = [
                k for k in range(8) if torch.cat([round_["prompt_ids"][k], round_["completion_ids"][k]]).tolist() == row
            ]
            assert len(j) == 1 and torch.equal(batch["advantages"][i], round_["advantages"][j[0]])
        assert rebuilt


def test_regrouping_leaves_multimodal_rounds_alone():
    batches = [{"x": torch.zeros(2, 3), "pixel_values": [torch.zeros(1)]}, {"x": torch.ones(2, 3), "pixel_values": []}]
    assert trainer_mod.sort_micro_batches(batches) is batches


def _prepare_stub(loss_type: str, accumulation: int, per_round: int, sort: bool = True):
    stub = _loss_stub(
        _tiny_policy(), trim=True, sort=sort, loss_type=loss_type, accumulation=accumulation, per_round=per_round
    )
    stub.model.train()
    round_ = _round(rows=8, seed=4)
    stub._buffered_inputs = None
    unsorted = split_tensor_dict(dict(round_), per_round)

    def fake_prepare(self, generation_batch):  # TRL's job: build the round once, hand out a slice per step
        if self._step % per_round == 0 or self._buffered_inputs is None:
            self._buffered_inputs = unsorted
        return self._buffered_inputs[self._step % per_round]

    return stub, unsorted, fake_prepare


@pytest.mark.parametrize(
    ("loss_type", "accumulation", "per_round", "regrouped"),
    [("dapo", 4, 4, True), ("dapo", 8, 4, True), ("bnpo", 4, 4, False), ("grpo", 4, 4, False), ("dapo", 4, 8, False)],
)
def test_micro_batches_are_regrouped_only_where_the_loss_is_globally_normalised(
    monkeypatch, loss_type, accumulation, per_round, regrouped
):
    stub, unsorted, fake_prepare = _prepare_stub(loss_type, accumulation, per_round)
    monkeypatch.setattr(GRPOTrainer, "_prepare_inputs", fake_prepare)
    first = stub._prepare_inputs({})
    assert (stub._buffered_inputs is not unsorted) is regrouped
    assert first is stub._buffered_inputs[0]
    stub._step = 1  # the second step of the round reuses the (regrouped) buffer
    assert stub._prepare_inputs({}) is stub._buffered_inputs[1]


def test_regrouping_can_be_switched_off(monkeypatch):
    stub, unsorted, fake_prepare = _prepare_stub("dapo", 4, 4, sort=False)
    monkeypatch.setattr(GRPOTrainer, "_prepare_inputs", fake_prepare)
    stub._prepare_inputs({})
    assert stub._buffered_inputs is unsorted


def test_drift_metrics_count_loss_tokens_at_the_cap_and_outside_the_band():
    stub = _loss_stub(_tiny_policy(), trim=True, sort=True)
    stub.model.train()
    ratio = torch.tensor([[3.0, 1.0, 1.2, 3.0], [0.9, 1.01, 3.0, 3.0]])
    inputs = {
        "importance_sampling_ratio": ratio,
        "completion_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "tool_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
    }
    stub._log_sampling_drift(inputs)  # loss tokens: 3 + 3 = 6 of 8
    metrics = stub._metrics["train"]
    assert metrics[trainer_mod.METRIC_IS_AT_CAP] == [
        pytest.approx(2 / 6)
    ]  # (0,0) and (1,2); (0,3) and (1,3) are masked
    assert metrics[trainer_mod.METRIC_IS_OUT_OF_BAND] == [pytest.approx(4 / 6)]  # 3.0, 1.2, 0.9, 3.0
    stub._log_sampling_drift({**inputs, "importance_sampling_ratio": ratio[:, :1]})  # sequence-level: nothing to log
    assert len(metrics[trainer_mod.METRIC_IS_AT_CAP]) == 1


# ── the real GRPOTrainer, end to end on CPU ────────────────────────────────
#
# Everything is real (`GRPOTrainer.__init__`, `environment_factory`, the tokenizer's chat template and
# `parse_response`, `_get_tool_suffix_ids`, `_generate`, `_calculate_rewards`, the loss and the optimiser step) except
# the weights (a 4-layer random Qwen3.5 text model) and the sampler, which is scripted so the episode is known.

_QWEN35 = "Qwen/Qwen3.5-9B"  # only its tokenizer and config are read, from the local Hugging Face cache


def _tiny_qwen35():
    pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        path = snapshot_download(_QWEN35, local_files_only=True)
    except Exception:  # noqa: BLE001 — not cached on this machine
        pytest.skip(f"{_QWEN35} tokenizer is not in the local Hugging Face cache")
    tokenizer = AutoTokenizer.from_pretrained(path)
    text = AutoConfig.from_pretrained(path).text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 4
    text.layer_types = text.layer_types[:4]
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
    text.linear_num_key_heads, text.linear_num_value_heads = 2, 4
    text.linear_key_head_dim = text.linear_value_head_dim = 16
    text.mtp_num_hidden_layers = 0
    torch.manual_seed(0)
    return tokenizer, AutoModelForCausalLM.from_config(text, dtype=torch.float32)


class _RealEnv:
    """The environment protocol TRL needs plus the two members the trainer reads (`_report`, `_report_error`)."""

    def __init__(self):
        self._report = self._report_error = None
        self.calls: list[tuple[str, str]] = []

    def reset(self, **row):
        self._report = self._report_error = None
        self.calls = []
        return f"do the task ep{row['task']}"

    def bash(self, command: str):
        """Run a shell command.

        Args:
            command: The command to run.
        """
        self.calls.append(("bash", command))
        return f"ran {command}"

    def submit_report(self, status: str, confidence: float):
        """Submit the final report.

        Args:
            status: complete or incomplete.
            confidence: Probability that the task is done.
        """
        self._report = (status, confidence)
        self.calls.append(("report", status))
        return "received"


def _turn_text(name: str | None = None, **arguments: Any) -> str:
    """What the model samples after the generation prompt `...<|im_start|>assistant\n<think>\n` (Qwen3.5 format)."""
    if name is None:
        return "\n</think>\n\nall done<|im_end|>"
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in arguments.items())
    return f"\n</think>\n\n<tool_call>\n<function={name}>\n{params}</function>\n</tool_call><|im_end|>"


REAL_SUFFIX = "\n<|im_start|>user\n<tool_response>\n{out}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"


@pytest.mark.parametrize("stop_at_report", [True, False])
def test_real_grpo_trainer_end_to_end_keeps_sampled_ids_and_stops_at_the_report(monkeypatch, tmp_path, stop_at_report):
    from datasets import Dataset
    from trl import GRPOConfig

    monkeypatch.setenv("TRL_EXPERIMENTAL_SILENCE", "1")
    tokenizer, model = _tiny_qwen35()
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    report = ("submit_report", {"status": "complete", "confidence": 0.9})
    # task 0: bash, report. task 1: bash, bash, then a plain-text stop (no report).
    scripts = {
        0: [_turn_text("bash", command="ls"), _turn_text(report[0], **report[1]), _turn_text()],
        1: [_turn_text("bash", command="pwd"), _turn_text("bash", command="id"), _turn_text()],
    }
    samples = {"prompts": 0}

    def scripted(self, prompt_ids, images, multimodal_fields, num_generations, has_tool_images=False):
        samples["prompts"] += len(prompt_ids)
        ids, logprobs = [], []
        for prompt in prompt_ids:
            text = tokenizer.decode(prompt)
            episode = 0 if "ep0" in text else 1
            turn = text.count("<tool_response>")
            sampled = encode(scripts[episode][turn])
            ids.append(sampled)
            logprobs.append([-0.25 - 0.01 * i for i in range(len(sampled))])
        return ids, logprobs

    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", scripted)
    seen = {}

    def reward(prompts, completions, environments, **kwargs):
        seen.update(kwargs, completions=completions, calls=[list(e.calls) for e in environments])
        return [1.0 if e._report else 0.0 for e in environments]

    rows = [
        {"prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": ""}], "task": i} for i in range(2)
    ]
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        num_generations=2,
        max_completion_length=600,
        max_tool_calling_iterations=5,
        use_cpu=True,
        report_to="none",
        bf16=False,
        fp16=False,
        max_steps=1,
        save_strategy="no",
        scale_rewards="none",
        beta=0.0,
        mask_truncated_completions=True,
    )
    trainer = HonestGRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        environment_factory=_RealEnv,
    )
    trainer.stop_at_report = stop_at_report
    trainer.train()
    assert trainer.state.global_step == 1  # the loss and the optimiser step ran on the loop's output

    # Independent oracle: the sampled ids of each turn and the tool-result ids, concatenated, never re-encoded.
    def suffix(out):
        return encode(REAL_SUFFIX.format(out=out))

    t0 = [encode(s) for s in scripts[0]]
    t1 = [encode(s) for s in scripts[1]]
    ep0 = t0[0] + suffix("ran ls") + t0[1]
    if not stop_at_report:  # TRL's loop executes the report, appends its result and samples one more turn
        ep0 += suffix("received") + t0[2]
    ep1 = t1[0] + suffix("ran pwd") + t1[1] + suffix("ran id") + t1[2]
    assert [list(i) for i in seen["completion_ids"]] == [ep0, ep0, ep1, ep1]
    # 4 first turns, 4 after the first tool round; then the two reported episodes are not sampled again
    assert samples["prompts"] == (10 if stop_at_report else 12)

    def mask_and_logprobs(turns, suffixes):
        mask, lps = [], []
        for i, turn in enumerate(turns):
            if i:
                mask += [0] * len(suffixes[i - 1])
                lps += [0.0] * len(suffixes[i - 1])
            mask += [1] * len(turn)
            lps += [-0.25 - 0.01 * j for j in range(len(turn))]
        return mask, lps

    m0, l0 = mask_and_logprobs(t0[: 2 if stop_at_report else 3], [suffix("ran ls"), suffix("received")])
    m1, l1 = mask_and_logprobs(t1, [suffix("ran pwd"), suffix("ran id")])
    assert [list(m) for m in seen["tool_mask"]] == [m0, m0, m1, m1]
    assert [list(x) for x in seen["logprobs"]] == [pytest.approx(l0)] * 2 + [pytest.approx(l1)] * 2
    assert len(seen["prompt_ids"][0]) == len(seen["prompt_ids"][1]) and all(
        isinstance(p, list) for p in seen["prompt_ids"]
    )
    roles = lambda c: [m["role"] for m in c]
    assert roles(seen["completions"][0]) == (
        ["assistant", "tool", "assistant"]
        if stop_at_report
        else ["assistant", "tool", "assistant", "tool", "assistant"]
    )
    assert roles(seen["completions"][2]) == ["assistant", "tool", "assistant", "tool", "assistant"]
    assert seen["calls"][0] == [("bash", "ls"), ("report", "complete")]
    assert seen["calls"][2] == [("bash", "pwd"), ("bash", "id")]
    assert trainer.last_loop_exits == [STOP_REPORT, STOP_REPORT, STOP_SILENT, STOP_SILENT]


@pytest.mark.gpu
def test_loop_with_real_model_and_vllm_is_identical_to_trl():  # pragma: no cover
    pytest.skip("needs a GPU and a vLLM server: run on the training node (see unverified in the report)")


@pytest.mark.docker
def test_loop_with_real_sandboxes():  # pragma: no cover
    pytest.skip("needs Docker: run on the training node (see unverified in the report)")
