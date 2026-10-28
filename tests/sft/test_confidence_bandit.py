"""`sft.confidence_bandit`: items, the prompt cut inside the model's own report, parsing and the Brier reward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.sft import confidence_bandit as cb
from tests.sft.helpers import make_row, write_bank

SOURCE = "src/task_001"
REPORT = {"status": "complete", "completed": ["did it"], "incomplete": [], "evidence_commands": ["ls"], "confidence": 1}


def row(*, passed: bool = True, report: dict | None = None, source: str = SOURCE, episode: str = "e1", **kw):
    r = make_row(f"{source.replace('/', '__')}__feasible", passed=passed, original=dict(report or REPORT),
                 episode_id=episode, spec={"pair_id": source}, **kw)  # fmt: skip
    return r.model_copy(update={"task": {**r.task, "pair_id": source}})


def _args(item: cb.Item) -> dict:
    return item.messages[-1]["tool_calls"][0]["function"]["arguments"]


def test_an_item_keeps_the_report_and_puts_the_sentinel_where_the_confidence_goes():
    item = cb.item_of(row(), set())
    assert isinstance(item, cb.Item) and item.y == 1 and item.status == "complete" and item.original_confidence == 1.0
    args = _args(item)
    assert list(args)[-1] == "confidence" and args["confidence"] == cb.SENTINEL and not item.confidence_moved
    assert {k: v for k, v in args.items() if k != "confidence"} == {
        k: v for k, v in REPORT.items() if k != "confidence"
    }
    # a report without a confidence gets one, last; one that wrote it first has it moved and recorded
    bare = cb.item_of(row(report={k: v for k, v in REPORT.items() if k != "confidence"}), set())
    assert list(_args(bare))[-1] == "confidence" and bare.original_confidence is None and not bare.confidence_moved
    first = cb.item_of(row(report={"confidence": 0.4, **{k: v for k, v in REPORT.items() if k != "confidence"}}), set())
    assert list(_args(first))[-1] == "confidence" and first.confidence_moved and first.original_confidence == 0.4


def test_the_label_is_the_graded_pass_and_drops_are_counted(tmp_path: Path):
    assert cb.item_of(row(passed=False), set()).y == 0
    assert (
        cb.item_of(row(tool="abstain", report={"reason": "x", "blocker_category": "other"}), set()) == cb.DROP_ABSTAIN
    )
    assert cb.item_of(row(no_report=True), set()) == cb.DROP_NO_REPORT
    assert cb.item_of(row(split="val"), set()) == cb.DROP_NOT_TRAIN
    assert isinstance(cb.item_of(row(split="select"), set(), train_only=False), cb.Item)  # the paired evaluation
    with pytest.raises(cb.ExcludedSourceError):
        cb.item_of(row(), {SOURCE})
    assert cb.item_of(row(), {SOURCE}, drop_excluded=True) == cb.DROP_EXCLUDED
    data = tmp_path / "data"
    (data / "decision").mkdir(parents=True)
    (data / "select_sources.txt").write_text("sel/task_x\n")
    (data / "split_seeds.json").write_text(json.dumps({SOURCE: "train"}))
    (data / "decision" / "probe_prompts_surfaces.jsonl").write_text(json.dumps({"pair_id": "probe/task_p"}) + "\n")
    bank = write_bank(tmp_path / "bank", [row(episode="a"), row(episode="b", passed=False), row(episode="a")])
    b = cb.build([bank], data)
    assert [i.id for i in b.items] == ["a", "b"] and b.dropped == {"duplicate_episode": 1}
    s = cb.summary(b)
    assert s["feasible_claims"] == {"n": 2, "pass": 1, "fail": 1}


def test_parse_and_brier():
    assert cb.parse_confidence("0.85\n</parameter>") == 0.85
    assert cb.parse_confidence("1\n</parameter>") == 1.0 and cb.parse_confidence(" .3\n") == 0.3
    for bad in ("1.5\n", "-0.1\n", "high\n", "", "0.8 because\n"):
        assert cb.parse_confidence(bad) is None, bad
    assert cb.brier_reward(0.8, 1) == pytest.approx(-0.04) and cb.brier_reward(0.8, 0) == pytest.approx(-0.64)
    assert cb.brier_reward(None, 1) == -1.0


def test_the_qwen35_prompt_ends_where_a_real_confidence_value_is_written():
    """Rendered with the served template, the prompt is the full report's text up to the value: the prefix plus the
    value reproduces the rendering of the same report with that value."""
    from transformers import AutoTokenizer

    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")
    kwargs = dict(family_spec("qwen3_5").chat_template_kwargs)
    item = cb.item_of(row(), set())
    prefix = cb.render_prefix(item.messages, tok, tool_schemas(), kwargs)
    assert prefix.endswith("<parameter=confidence>\n")
    real = [*item.messages[:-1], json.loads(json.dumps(item.messages[-1]).replace(json.dumps(cb.SENTINEL), "0.7"))]
    full = tok.apply_chat_template(real, tools=tool_schemas(), tokenize=False, **kwargs)
    assert full.startswith(prefix) and full[len(prefix) :].startswith("0.7\n</parameter>")
    assert cb.parse_confidence(full[len(prefix) :]) == 0.7


def test_every_rollout_of_a_leaky_variant_is_dropped(tmp_path: Path):
    """An infeasible variant that one rollout passed: its label is wrong for all of them (the eval voids it)."""
    data = tmp_path / "data"
    (data / "decision").mkdir(parents=True)
    (data / "select_sources.txt").write_text("sel/task_x\n")
    (data / "split_seeds.json").write_text(json.dumps({SOURCE: "train", "src/task_002": "train"}))
    (data / "decision" / "probe_prompts_surfaces.jsonl").write_text(json.dumps({"pair_id": "probe/task_p"}) + "\n")
    leak = [row(episode=f"l{k}", passed=k == 0, feasible=False) for k in range(3)]
    bank = write_bank(tmp_path / "bank", [*leak, row(episode="ok", source="src/task_002")])
    b = cb.build([bank], data)
    assert [i.id for i in b.items] == ["ok"] and b.dropped == {cb.DROP_LEAKY: 3}
