"""The pilot dataset builder: bank directory in, one prompt per frozen trajectory out."""

from __future__ import annotations

import json

import pytest

from honest_agents import bank
from honest_agents.bank import QuarantineEntry
from honest_agents.pilot import dataset
from honest_agents.pilot.dataset import BuildOptions, build_from_bank, build_rows
from honest_agents.pilot.env import BANK_ROW_COLUMN, ReplayEnv, load_bank_row
from honest_agents.reward import MASK_BOOT_FAILURE
from tests.pilot.helpers import (
    INSTRUCTION,
    SYSTEM,
    write_bank,
)
from tests.pilot.helpers import make_bank_row as _make_bank_row
from tests.pilot.helpers import synthetic_rows as _synthetic_rows


def _with_split(row, split):
    return row.model_copy(update={"task": {**row.task, "split": split}})


def make_bank_row(*, split="train", **kw):
    """The shared helper's rows carry no split column; the pilot now requires `split == "train"` (L6)."""
    row = _make_bank_row(**kw)
    return row if split is None else _with_split(row, split)


def synthetic_rows(*a, split="train", **kw):
    return [_with_split(r, split) for r in _synthetic_rows(*a, **kw)]


def test_a_row_is_the_frozen_transcript_up_to_the_report_point():
    row = make_bank_row(passed=True, n_commands=2, original={"status": "complete"})
    (r,), stats = build_rows([row])
    roles = [m["role"] for m in r["prompt"]]
    assert roles == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]  # ends on a tool result, not a report
    assert r["prompt"][0]["content"] == SYSTEM and r["prompt"][1]["content"] == INSTRUCTION
    assert not any(
        c["function"]["name"] in ("submit_report", "abstain") for m in r["prompt"] for c in m.get("tool_calls", [])
    )
    assert r["prompt"][2]["tool_calls"][0]["function"]["arguments"] == {"command": "cmd0"}
    assert (
        r["kind"],
        r["feasible"],
        r["is_control"],
        r["passed"],
        r["n_actions"],
        r["forced_report"],
    ) == ("feasible", True, False, True, 2, False)
    assert r["original_status"] == "complete" and stats.n_rows == 1


def test_the_bank_row_column_replays_and_drops_the_heavy_fields():
    row = make_bank_row(passed=True)
    row = row.model_copy(update={"completion_ids": list(range(1000)), "logprobs": [0.0] * 1000})
    (r,), _ = build_rows([row])
    slim = load_bank_row(r[BANK_ROW_COLUMN])
    assert slim.completion == [] and slim.prompt == [] and slim.completion_ids is None and slim.logprobs is None
    assert slim.outcome == row.outcome and slim.blocker_facts == row.blocker_facts and slim.episode_id == row.episode_id
    env = ReplayEnv()
    env.reset(**r)  # a dataset row is the reset kwargs, prompt included
    assert env._n_actions_at_report == 3


def test_masked_quarantined_leaky_tampered_and_unseen_rows_are_dropped_and_counted():
    rows = [
        make_bank_row(variant="ok"),
        make_bank_row(variant="m", masked_reason=MASK_BOOT_FAILURE),
        make_bank_row(variant="q"),
        make_bank_row(variant="leaked", kind="infeasible", passed=True),  # claimed: the live run masked it
        make_bank_row(
            variant="unclaimed",
            kind="infeasible",
            passed=True,
            original={"status": "incomplete"},
        ),
        make_bank_row(variant="t", tamper=True),
        make_bank_row(variant="u", extra_report_calls=2),
    ]
    out, stats = build_rows(rows, quarantined={"q"})
    assert [r["variant_id"] for r in out] == ["ok"]
    assert stats.dropped == {
        "masked": 2,
        "quarantined": 1,
        "infeasible_clean_pass": 1,
        "tampered": 1,
        "unseen_actions": 1,
    }
    kept, _ = build_rows(
        rows,
        BuildOptions(drop_unseen_actions=False, drop_tampered=False, drop_leaky=False),
        quarantined={"q"},
    )
    assert {r["variant_id"] for r in kept} == {"ok", "unclaimed", "t", "u"}


def test_a_trajectory_that_never_reported_becomes_a_forced_report_turn():
    row = make_bank_row(no_report=True, n_commands=2)
    row.completion.append(
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "last"}},
                }
            ],
        }
    )
    (r,), stats = build_rows([row], BuildOptions(drop_unseen_actions=False))
    assert r["forced_report"] and r["prompt"][-1]["role"] == "tool" and stats.n_forced_report == 1
    assert build_rows([row])[0] == []  # the cut-off tool call is an action the context does not show
    out, st = build_rows([make_bank_row(no_report=True)], BuildOptions(include_no_report=False))
    assert out == [] and st.dropped == {"no_report": 1}


def test_unseen_actions_from_a_final_tool_turn_are_counted_for_a_forced_report():
    row = make_bank_row(no_report=True, n_commands=1)
    row.completion.append(
        {
            "role": "assistant",
            "content": "x",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"command": "z"}},
                }
            ],
        }
    )
    out, stats = build_rows([row])
    assert out == [] and stats.dropped == {"unseen_actions": 1}


def test_duplicates_and_long_contexts_are_dropped():
    row = make_bank_row()
    out, stats = build_rows([row, row, make_bank_row(variant="long")], BuildOptions(max_context_chars=40))
    assert out == [] and stats.dropped == {"duplicate": 1, "too_long": 2}


def test_mix_downsamples_to_the_target_shares():
    rows = synthetic_rows(n_feasible=40, n_control=10, n_infeasible=10, per_variant=1)
    out, stats = build_rows(
        rows,
        BuildOptions(mix={"feasible": 0.5, "control": 0.25, "infeasible": 0.25}, seed=3),
    )
    assert stats.kinds == {
        "feasible": 20,
        "control": 10,
        "infeasible": 10,
    }  # limited by the scarcest kind
    assert stats.dropped["mix_downsample"] == 20
    same, _ = build_rows(
        rows,
        BuildOptions(mix={"feasible": 0.5, "control": 0.25, "infeasible": 0.25}, seed=3),
    )
    assert [r["pilot_id"] for r in same] == [r["pilot_id"] for r in out]  # deterministic for a seed
    other, _ = build_rows(
        rows,
        BuildOptions(mix={"feasible": 0.5, "control": 0.25, "infeasible": 0.25}, seed=4),
    )
    assert {r["pilot_id"] for r in other} != {r["pilot_id"] for r in out}


def test_a_kind_missing_from_the_bank_renormalises_or_fails_when_strict():
    rows = synthetic_rows(n_feasible=6, n_control=0, n_infeasible=0)
    out, stats = build_rows(rows, BuildOptions(mix=dataset.DEFAULT_MIX))
    assert len(out) == len(rows) and any("renormalised" in w for w in stats.warnings)
    with pytest.raises(ValueError, match="no such rows"):
        build_rows(rows, BuildOptions(mix=dataset.DEFAULT_MIX, strict_mix=True))


def test_per_variant_and_row_caps():
    rows = [make_bank_row(variant="a") for _ in range(5)] + [make_bank_row(variant="b") for _ in range(5)]
    out, stats = build_rows(rows, BuildOptions(max_per_variant=2, max_rows=3))
    assert len(out) == 3 and stats.dropped == {"variant_cap": 6, "max_rows": 1}


def test_build_options_validate():
    with pytest.raises(ValueError, match="unknown kind"):
        BuildOptions(mix={"nope": 1})
    with pytest.raises(ValueError, match="max_rows"):
        BuildOptions(max_rows=0)


def test_stats_report_what_the_pilot_can_measure():
    rows = [make_bank_row(variant=f"v{i}", passed=i < 3, original={"status": "complete"}) for i in range(10)]
    _, stats = build_rows(rows)
    assert stats.feasible_pass_share == 0.3 and stats.original_claim_rate_feasible == 1.0
    assert stats.original_fabrication_among_failures == 1.0 and stats.efficiency_ref_actions == [28]
    _, only = build_rows([make_bank_row(passed=True)])
    assert any("cannot be measured" in w for w in only.warnings)


def test_build_from_a_bank_directory_honours_the_quarantine_list(tmp_path):
    rows = synthetic_rows(n_feasible=4, n_control=0, n_infeasible=0, per_variant=1)
    write_bank(tmp_path, rows[:2], "bank-1.jsonl")
    write_bank(tmp_path, rows[2:], "bank-2.jsonl")
    bank.append_quarantine(
        [QuarantineEntry(variant_id="f0-feasible", kind="leak")],
        tmp_path / "quarantine.jsonl",
    )
    out, stats = build_from_bank(tmp_path)
    assert stats.n_bank_rows == 4 and len(out) == 3 and stats.dropped == {"quarantined": 1}
    with pytest.raises(FileNotFoundError):
        build_from_bank(tmp_path / "missing")
    with pytest.raises(ValueError, match="no usable trajectories"):
        build_from_bank(tmp_path, BuildOptions(max_context_chars=1))


def test_hf_dataset_returns_rows_as_built_without_arrow_null_arguments():
    rows, _ = build_rows(synthetic_rows(n_feasible=2, n_control=0, n_infeasible=1, per_variant=1))
    ds = dataset.to_hf_dataset(rows)
    assert len(ds) == len(rows)
    first = ds[0]
    assert first["prompt"] == rows[0]["prompt"] and first[BANK_ROW_COLUMN] == rows[0][BANK_ROW_COLUMN]
    call = first["prompt"][2]["tool_calls"][0]["function"]
    assert call["arguments"] == {"command": "cmd0"}  # Arrow would have added a null for every other argument name
    batch = ds[:2]
    assert batch["prompt"][1] == rows[1]["prompt"] and dataset.PROMPT_JSON_COLUMN not in batch
    with pytest.raises(ValueError):
        dataset.to_hf_dataset([])


def test_cli_writes_rows_and_stats(tmp_path, capsys):
    write_bank(tmp_path / "bank", synthetic_rows())
    out = tmp_path / "out" / "pilot.jsonl"
    assert (
        dataset.main(
            [
                "--bank",
                str(tmp_path / "bank"),
                "--out",
                str(out),
                "--mix",
                "default",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    rows = dataset.read_rows(out)
    stats = json.loads((tmp_path / "out" / "pilot.jsonl.stats.json").read_text())
    assert stats["n_rows"] == len(rows) and set(stats["kinds"]) == {
        "feasible",
        "control",
        "infeasible",
    }
    assert dataset._parse_mix("feasible=1,control=0.5") == {
        "feasible": 1.0,
        "control": 0.5,
    }


@pytest.mark.parametrize("split", [None, "test", "validation", "val", ""])
def test_only_train_split_rows_are_built_and_a_missing_split_is_refused(split):
    """L6: a bank row without a split column proves nothing about where it came from, so it is dropped as well."""
    good, bad = make_bank_row(variant="g"), make_bank_row(variant="b", split=split)
    if split is None:
        bad = bad.model_copy(update={"task": {k: v for k, v in bad.task.items() if k != "split"}})
    rows, stats = build_rows([good, bad])
    assert [r["variant_id"] for r in rows] == ["g"]
    assert stats.dropped[dataset.DROP_SPLIT] == 1


def test_eval_sourced_quarantine_entries_do_not_shape_the_pilot_set(tmp_path):
    rows = [make_bank_row(variant=f"v{i}") for i in range(3)]
    bank_dir = write_bank(tmp_path / "bank", rows)
    bank.append_quarantine(
        [QuarantineEntry(variant_id="v0", source="eval"), QuarantineEntry(variant_id="v1", source="train")],
        bank_dir / bank.QUARANTINE_BASENAME,
    )
    built, _ = build_from_bank(bank_dir)
    assert sorted(r["variant_id"] for r in built) == ["v0", "v2"]
