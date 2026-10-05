"""Row filter, held-out partition, loss mask and reward spread."""

from __future__ import annotations

import pytest

from honest_agents.sft import data, hindsight
from honest_agents.sft.data import RowFilter, SFTExample
from tests.sft.helpers import bank_dir, bash_turn, make_row, report_call, tiny_tokenizer


def test_only_train_rows_are_used(tmp_path):
    rows = [make_row(f"t{i}") for i in range(3)] + [make_row("v", split="validation"), make_row("x", split="test")]
    rows.append(make_row("n").model_copy(update={"task": {"variant_id": "n"}}))  # no split at all
    sel = data.read_rows(bank_dir(tmp_path, rows))
    assert len(sel.usable) == 3 and len(sel.scorable) == 3
    assert sel.dropped[data.DROP_SPLIT] == 3


def test_masked_tampered_and_leaky_rows_are_dropped(tmp_path):
    rows = [
        make_row("ok"),
        make_row("masked", masked_reason="boot_failure"),
        make_row("tamper", tamper=True),
        make_row("leaky", feasible=False, passed=True, original={"status": "incomplete"}),
    ]
    sel = data.read_rows(bank_dir(tmp_path, rows))
    assert [data.variant_of(r) for r in sel.usable] == ["ok"]
    assert sel.dropped == {data.DROP_MASKED: 1, data.DROP_TAMPERED: 1, data.DROP_LEAKY: 1}
    assert len(sel.scorable) == 3  # tampered and leaky outcomes still count for calibration, masked ones do not
    keep = data.read_rows(bank_dir(tmp_path, rows), RowFilter(drop_tampered=False))
    assert len(keep.usable) == 2


def test_unseen_actions_and_no_report_options(tmp_path):
    unseen = make_row("u")
    unseen = unseen.model_copy(update={"report_turn_extra_calls": 1})
    forced = make_row("f", no_report=True)
    d = bank_dir(tmp_path, [unseen, forced])
    assert [data.variant_of(r) for r in data.read_rows(d).usable] == ["f"]
    assert not data.read_rows(d, RowFilter(include_no_report=False)).usable
    assert len(data.read_rows(d, RowFilter(drop_unseen_actions=False)).usable) == 2


def test_quarantined_variants_are_dropped(tmp_path):
    from honest_agents import bank

    d = bank_dir(tmp_path, [make_row("q"), make_row("ok")])
    bank.append_quarantine([bank.QuarantineEntry(variant_id="q", source="train")], d / bank.QUARANTINE_BASENAME)
    assert [data.variant_of(r) for r in data.read_rows(d).usable] == ["ok"]


def test_partition_is_by_variant_and_stable():
    rows = [make_row(f"v{i}") for i in range(40) for _ in range(2)]
    fit, held = data.partition(rows, 0.25, seed=3)
    assert {data.variant_of(r) for r in fit}.isdisjoint({data.variant_of(r) for r in held})
    assert 0 < len(held) < len(rows) / 2
    assert [r.episode_id for r in data.partition(rows, 0.25, seed=3)[1]] == [r.episode_id for r in held]
    assert not data.partition(rows, 0.0, seed=3)[1]


# ── reward spread ────────────────────────────────────────────────────────────


def test_reward_spread_counts_groups_with_different_rewards():
    rows = [
        make_row("a", passed=True, original={"status": "complete"}),
        make_row("a", passed=False, original={"status": "complete"}),  # fabrication vs success: spread
        make_row("b", passed=False),
        make_row("b", passed=False),  # identical rewards: no spread
        make_row("c", passed=True),  # a group of one is not counted
        make_row("m", masked_reason="boot_failure"),
        make_row("m", masked_reason="boot_failure"),
    ]
    s = data.reward_spread(rows)
    assert (s.n_groups, s.n_with_variance, s.n_rows) == (2, 1, 5)
    assert s.fraction == 0.5 and s.mean_std > 0
    assert data.reward_spread([]).fraction == 0.0


# ── masks ────────────────────────────────────────────────────────────────────


def multi_turn_example(n_tools: int = 2) -> SFTExample:
    messages = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "Do the task."}]
    for i in range(n_tools):
        messages += bash_turn(f"cmd{i}", f"out{i}")
    messages.append(report_call({"status": "complete", "completed": ["x"], "confidence": 0.7}))
    return SFTExample("e", "test", "v", messages, [len(messages) - 1])


def trained_text(tok, t: data.Tokenized) -> str:
    return tok.decode([i for i, label in zip(t.input_ids, t.labels, strict=True) if label != data.IGNORE_INDEX])


@pytest.fixture(scope="module")
def renderer():
    return data.make_renderer(tiny_tokenizer(), "qwen3")


def test_only_the_selected_assistant_turn_is_trained(renderer):
    ex = multi_turn_example()
    t = data.tokenize_example(ex, renderer)
    text = trained_text(renderer.tokenizer, t)
    assert text.startswith("<tool_call>") and "submit_report" in text and text.rstrip().endswith("<|im_end|>")
    for masked in ("cmd0", "out0", "You are an agent", "Do the task", "<think>"):
        assert masked not in text
    assert t.n_trained < len(t.input_ids) // 4  # the system block with the tool schemas dominates and is masked


def test_all_assistant_turns_can_be_trained(renderer):
    ex = multi_turn_example()
    ex.loss_messages = [i for i, m in enumerate(ex.messages) if m["role"] == "assistant"]
    text = trained_text(renderer.tokenizer, data.tokenize_example(ex, renderer))
    assert "cmd0" in text and "cmd1" in text and "submit_report" in text
    assert "out0" not in text and "Do the task" not in text  # tool outputs and user turns stay masked


def test_too_long_examples_are_dropped_not_truncated(renderer):
    ex = multi_turn_example()
    assert data.tokenize_example(ex, renderer, max_length=50) == data.DROP_TOO_LONG
    tokenized, _, dropped = data.tokenize_examples([ex, ex], renderer, max_length=50)
    assert not tokenized and dropped[data.DROP_TOO_LONG] == 2


def test_example_validation():
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="at least one"):
        SFTExample("e", "k", "v", messages, [])
    with pytest.raises(ValueError, match="not an assistant"):
        SFTExample("e", "k", "v", messages, [0])


def test_hindsight_examples_tokenize_with_report_only_loss(renderer, tmp_path):
    rows = [make_row("a", passed=True), make_row("b", turns=[("jq x", "bash: jq: command not found")])]
    for row in rows:
        ex = hindsight.build_example(row)
        text = trained_text(renderer.tokenizer, data.tokenize_example(ex, renderer))
        assert "submit_report" in text and "ls /app" not in text
    assert "jq" in text
    ds = data.to_hf_dataset(data.tokenize_examples([hindsight.build_example(r) for r in rows], renderer)[0])
    assert set(ds.column_names) == {"input_ids", "labels"} and len(ds) == 2


def test_qwen35_template_masks_the_same_way():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")
    r = data.make_renderer(tok, "qwen3_5")
    t = data.tokenize_example(multi_turn_example(), r)
    text = trained_text(tok, t)
    assert "submit_report" in text and "cmd0" not in text and "out0" not in text and "<think>" not in text


def test_local_fallback_tokenizer_masks_the_same_way():
    from trl import chat_template_utils as ctu

    from tests.sft.helpers import _local_tokenizer

    tok = _local_tokenizer()
    tok.chat_template = ctu.qwen3_chat_template
    r = data.make_renderer(tok, "qwen3")
    text = trained_text(tok, data.tokenize_example(multi_turn_example(), r))
    assert text.startswith("<tool_call>") and "cmd0" not in text and "out0" not in text
