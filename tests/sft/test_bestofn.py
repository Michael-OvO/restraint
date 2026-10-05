"""Best-of-N trajectory SFT and the combo stage: ranking, filters, loss mask, mixture and a tiny CPU fit."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import bestofn, data, train_sft
from honest_agents.sft.bestofn import BestOfNOptions, select_rows
from tests.sft.helpers import bank_dir, make_row, tiny_tokenizer
from tests.sft.test_train_sft import ROOT, tiny_config

NOT_FOUND = [("jq . in.json", "bash: line 1: jq: command not found")]
OPTS = BestOfNOptions()


def pair(variant: str, *, good_id: str | None = None, **good):
    """A group: one clean pass (reward ~1.5) and one honest failure (0.25)."""
    return [
        make_row(variant, passed=True, original={"status": "complete"}, episode_id=good_id, **good),
        make_row(variant, passed=False, original={"status": "incomplete"}, turns=NOT_FOUND),
    ]


def select(rows, opts=OPTS, **kw):
    skipped: Counter = Counter()
    return select_rows(rows, rows, opts, skipped), skipped


def test_best_rollout_above_the_group_mean_is_kept():
    rows = pair("v", good_id="good")
    chosen, skipped = select(rows)
    assert [s.row.episode_id for s in chosen] == ["good"]
    assert chosen[0].rank == 1 and chosen[0].reward > chosen[0].group_mean
    assert skipped[bestofn.SKIP_NOT_TOP_K] == 1


def test_it_uses_the_rl_reward_not_the_pass_flag():
    # An infeasible variant: the honest, verified-blocker abstain is the best rollout, ahead of an implicit give-up.
    best = make_row(
        "inf",
        feasible=False,
        tool="abstain",
        original={"reason": "no jq", "blocker_category": "missing_capability", "blocker_target": "jq"},
        expected_blockers=["missing_capability"],
        expected_targets=["jq"],
        turns=NOT_FOUND,
        episode_id="best",
    )
    worse = make_row("inf", feasible=False, original={"status": "incomplete"}, turns=NOT_FOUND, episode_id="worse")
    chosen, _ = select([best, worse])
    assert [s.row.episode_id for s in chosen] == ["best"]
    assert chosen[0].reward == compute_reward(best.episode_outcome(), best.config()).reward


def test_groups_without_a_spread_and_tiny_groups_contribute_nothing():
    equal = [make_row("eq", episode_id=f"e{i}") for i in range(3)]
    lone = [make_row("lone", passed=True, original={"status": "complete"})]
    chosen, skipped = select(equal + lone)
    assert chosen == []
    assert (
        skipped[bestofn.SKIP_NOT_ABOVE_MEAN] == 1
        and skipped[bestofn.SKIP_NOT_TOP_K] == 2
        and skipped[bestofn.SKIP_SMALL_GROUP] == 1
    )
    kept, _ = select(equal, BestOfNOptions(above_mean=False))
    assert len(kept) == 1  # the opt-out keeps the top of every group


def test_tampered_and_violating_rollouts_are_never_imitated_even_with_the_row_filter_off(tmp_path):
    tamper = make_row("t", passed=True, original={"status": "complete"}, tamper=True, episode_id="tamper")
    honest = make_row("t", passed=False, original={"status": "incomplete"}, turns=NOT_FOUND, episode_id="honest")
    loose = data.RowFilter(drop_tampered=False, drop_violations=False)
    sel = data.read_rows(bank_dir(tmp_path, [tamper, honest]), loose)
    assert len(sel.usable) == 2
    skipped: Counter = Counter()
    chosen = select_rows(sel.usable, sel.scorable, OPTS, skipped)
    assert skipped[bestofn.SKIP_TAMPERED] == 1
    # The group mean still counts the tampered episode (it is in the RL group); the honest failure beats it.
    assert [s.row.episode_id for s in chosen] == ["honest"]


def test_reports_that_never_happened_are_not_imitated():
    forced = make_row("f", no_report=True, episode_id="forced")
    hidden = make_row("f", passed=True, original={"status": "complete"}, episode_id="hidden")
    hidden = hidden.model_copy(update={"report_turn_extra_calls": 1})
    fine = make_row("f", passed=True, original={"status": "complete"}, episode_id="fine")
    chosen, skipped = select([forced, hidden, fine])
    assert [s.row.episode_id for s in chosen] == ["fine"]
    assert skipped[bestofn.SKIP_NO_REPORT] == 1 and skipped[bestofn.SKIP_HIDDEN_CALLS] == 1


def test_top_k_ties_prefer_the_shorter_trajectory_and_are_deterministic():
    short = make_row("k", passed=True, original={"status": "complete"}, turns=[("ls", "a")], episode_id="short")
    long = make_row("k", passed=True, original={"status": "complete"}, turns=[("ls", "a")] * 4, episode_id="long")
    bad = make_row("k", turns=NOT_FOUND, episode_id="bad")
    chosen, _ = select([long, bad, short], BestOfNOptions(top_k=2))
    assert [s.row.episode_id for s in chosen] == ["short", "long"]
    again, _ = select([short, bad, long], BestOfNOptions(top_k=2))
    assert [s.row.episode_id for s in again] == [s.row.episode_id for s in chosen]


def test_reward_override_changes_the_ranking_config():
    rows = pair("v", good_id="good")
    # With the success bonus and efficiency removed a clean "complete" is still above an honest failure: base reward.
    strict = BestOfNOptions(reward={"success_bonus": 0.0, "efficiency_bonus": 0.0})
    chosen, _ = select(rows, strict)
    assert chosen[0].reward == compute_reward(rows[0].episode_outcome(), RewardConfig(**strict.reward)).reward


def test_require_pass_when_feasible_drops_groups_whose_best_did_not_pass():
    rows = [
        make_row("f", original={"status": "incomplete"}, turns=NOT_FOUND, episode_id="a"),
        make_row("f", original={"status": "complete"}, turns=NOT_FOUND, episode_id="b"),
    ]
    assert select(rows)[0]
    chosen, skipped = select(rows, BestOfNOptions(require_pass_when_feasible=True))
    # the fabrication is never imitated (not an imitable case); the honest failure is what the stricter rule drops
    assert chosen == [] and skipped[bestofn.SKIP_NO_PASS] == 1 and skipped[bestofn.SKIP_BAD_CASE] == 1


def test_the_best_of_a_bad_group_is_not_imitated_when_it_is_still_bad():
    """Above the group mean is relative: with a tamper (-1.5) in the group a fabrication (-1) would rank first."""
    rows = [
        make_row("bad", passed=False, original={"status": "complete"}, episode_id="fabricates"),
        make_row("bad", passed=False, original={"status": "complete"}, tamper=True, episode_id="tampers"),
    ]
    chosen, skipped = select(rows)
    assert chosen == [] and skipped[bestofn.SKIP_TAMPERED] == 1 and skipped[bestofn.SKIP_BAD_CASE] == 1
    abstain = {"reason": "no jq", "blocker_category": "missing_capability", "blocker_target": "jq"}
    feasible = [  # an abstention on a feasible task is `false_blame` (0), above two fabrications, and still wrong
        make_row("f", tool="abstain", original=abstain, turns=NOT_FOUND, episode_id="abstains"),
        make_row("f", original={"status": "complete"}, episode_id="x"),
        make_row("f", original={"status": "complete"}, episode_id="y"),
    ]
    assert select(feasible)[0] == []
    wrong = [  # infeasible: a named blocker that is not the expected one is `abstain_wrong_reason` (0.5)
        make_row("w", feasible=False, tool="abstain", original=abstain, expected_blockers=["permission_denied"],
                 turns=NOT_FOUND, episode_id="wrong"),
        make_row("w", feasible=False, original={"status": "complete"}, episode_id="claims"),
    ]  # fmt: skip
    assert select(wrong)[0] == []


def test_validation_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown key"):
        bestofn.parse_options({"topk": 2})
    with pytest.raises(ValueError, match="top_k"):
        BestOfNOptions(top_k=0)
    with pytest.raises(ValueError, match="alpha"):
        BestOfNOptions(reward={"alpha": 7.0})
    with pytest.raises(ValueError, match="unknown key"):
        train_sft.parse_config({"kind": "bestofn", "bestofn": {"nope": 1}})
    with pytest.raises(ValueError, match="confidence"):
        train_sft.parse_config({"kind": "combo", "hindsight": {"confidence": "maybe"}})


def test_shipped_config_registers_both_stages():
    cfg = train_sft.load_config(ROOT / "configs" / "sft_bestofn.yaml")
    assert cfg.kind == "bestofn" and cfg.bestofn["top_k"] == 1
    assert train_sft.load_config(ROOT / "configs" / "sft_bestofn.yaml", ["kind=combo"]).kind == "combo"
    assert {"bestofn", "combo"} <= set(train_sft.STAGES)


# ── examples and the loss mask ───────────────────────────────────────────────


def test_example_is_the_unedited_trajectory_with_every_assistant_turn_trained():
    row = make_row(
        "v",
        passed=True,
        original={"status": "complete", "completed": ["wrote out.json"], "confidence": 0.8},
        turns=[("ls", "in.txt"), ("cat in.txt", "hello")],
    )
    ex = bestofn.build_example(bestofn.Selection(row, 1.2, 0.7, 2, 1))
    roles = [m["role"] for m in ex.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert ex.loss_messages == [2, 4, 6]
    final = ex.messages[-1]["tool_calls"][0]["function"]
    assert final["arguments"]["completed"] == ["wrote out.json"] and final["arguments"]["confidence"] == 0.8


def test_spec_labels_never_enter_an_example():
    row = make_row(
        "v",
        passed=False,
        turns=NOT_FOUND,
        expected_targets=["SENTINEL_TARGET"],
        expected_blockers=["SENTINEL_BLOCKER"],
    )
    ex = bestofn.build_example(bestofn.Selection(row, 0.25, 0.1, 2, 1))
    text = json.dumps([ex.messages, ex.meta])
    for banned in ("SENTINEL_TARGET", "SENTINEL_BLOCKER", "secret_perturbation", "expected", "decoy", "family"):
        assert banned not in text


def test_labels_cover_assistant_tokens_only():
    tok = tiny_tokenizer()
    row = make_row("v", passed=True, original={"status": "complete"}, turns=[("ls", "OUTPUT_MARK"), ("cat x", "more")])
    ex = bestofn.build_example(bestofn.Selection(row, 1.2, 0.7, 2, 1))
    renderer = data.make_renderer(tok, "qwen3")
    t = data.tokenize_example(ex, renderer, 4096)
    assert not isinstance(t, str)
    trained = tok.decode([i for i in t.labels if i != data.IGNORE_INDEX])
    assert "ls" in trained and "cat x" in trained and "submit_report" in trained and "<|im_end|>" in trained
    for masked in ("OUTPUT_MARK", "more", "You are an agent", "Parse the config", "tool_response"):
        assert masked not in trained
    runs = data.mask_runs([int(x != data.IGNORE_INDEX) for x in t.labels])
    assert len(runs) == 3  # one run per assistant turn


# ── stages on a bank ─────────────────────────────────────────────────────────


def mixed_bank(tmp_path, variants: int = 24):
    rows = []
    for i in range(variants):
        rows += pair(f"v{i}")
        rows.append(make_row(f"v{i}", split="test", passed=True, original={"status": "complete"}))  # never read
    return bank_dir(tmp_path, rows)


def config(tmp_path, kind: str, **over: str):
    return tiny_config(tmp_path, **{"kind": kind, "data.bank": mixed_bank(tmp_path / "m"), **over})


def _raw(cfg):
    return dataclasses.asdict(cfg)


def test_bestofn_stage_reads_train_rows_and_splits_held_out_variants(tmp_path):
    stage, summary = train_sft.build_stage_data(config(tmp_path, "bestofn"))
    assert stage.train and stage.eval
    assert {e.variant for e in stage.train}.isdisjoint({e.variant for e in stage.eval})
    assert all(e.kind == "bestofn" and e.meta["passed"] for e in stage.train)  # the clean pass of each group
    assert summary["bestofn"]["selected"] == len(stage.train) and summary["bestofn"]["selected_passed"] == len(
        stage.train
    )
    assert summary["rows_dropped"]["not_train_split"] == 24  # the test rows were dropped, never ranked


def test_combo_mixes_without_duplicating_an_episode(tmp_path):
    cfg = config(tmp_path, "combo")
    stage, summary = train_sft.build_stage_data(cfg)
    best = [e for e in stage.train if e.kind == "bestofn"]
    hind = [e for e in stage.train if e.kind == "hindsight"]
    assert best and hind and len(hind) <= len(best)  # ratio 1.0
    ids = [e.id for e in stage.train]
    assert len(ids) == len(set(ids))
    assert summary["mixture"]["bestofn_examples"] == len(best)
    full = train_sft.parse_config({**_raw(cfg), "bestofn": {"hindsight_per_bestofn": None}})
    everything, _ = train_sft.build_stage_data(full)
    assert len([e for e in everything.train if e.kind == "hindsight"]) == summary["mixture"]["hindsight_available"]
    assert all(len(e.loss_messages) == 1 for e in hind)  # report-only: one trained turn


def test_dry_run_tokenises_both_stages(tmp_path):
    for kind in ("bestofn", "combo"):
        cfg = config(tmp_path, kind, **{"data.min_examples": "2"})
        summary = train_sft.dry_run(cfg, tokenizer=tiny_tokenizer())
        assert summary["tokenized_fit"] == summary["examples_fit"] > 0
        assert (cfg.output_dir / "examples.jsonl").exists()
    # Whole trajectories train more tokens per example than a report alone.
    assert summary["mean_trained_tokens"] > 0


def test_tiny_cpu_fit_of_the_combo_stage(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = tiny_tokenizer()
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=8192,
        )
    )
    cfg = config(tmp_path, "combo")
    summary = train_sft.run_sft(cfg, model=model, tokenizer=tok)
    assert summary["train_loss"] > 0 and (cfg.output_dir / "final" / "adapter_config.json").exists()
    assert any("eval_entropy" in h for h in summary["log_history"])


# ── review additions ─────────────────────────────────────────────────────────


def test_rows_outside_the_train_split_are_never_selected_even_when_passed_in_directly():
    rows = [
        make_row("s", split=sp, passed=True, original={"status": "complete"}, episode_id=sp) for sp in ("val", "test")
    ]
    rows.append(make_row("s", turns=NOT_FOUND, episode_id="train_fail"))
    chosen, skipped = select(rows)
    assert not [s for s in chosen if s.row.episode_id in ("val", "test")]
    assert skipped[bestofn.SKIP_SPLIT] == 2


def test_each_row_is_ranked_under_its_stored_reward_config():
    # Same outcome, two stored configs: the group mean and the ranking use what each row was scored with.
    hi = make_row(
        "c",
        passed=True,
        original={"status": "complete"},
        episode_id="hi",
        reward_config=RewardConfig(success_bonus=2.0),
    )
    lo = make_row("c", passed=True, original={"status": "complete"}, episode_id="lo")
    chosen, _ = select([lo, hi])
    assert [s.row.episode_id for s in chosen] == ["hi"]
    assert chosen[0].reward == compute_reward(hi.episode_outcome(), hi.config()).reward


def test_masked_rows_and_quarantined_variants_are_never_selected(tmp_path):
    from honest_agents import bank

    rows = [*pair("q"), *pair("m"), *pair("ok")]
    rows[2] = make_row("m", passed=True, original={"status": "complete"}, masked_reason="leak", episode_id="masked")
    d = bank_dir(tmp_path, rows)
    bank.append_quarantine([bank.QuarantineEntry(variant_id="q", source="train")], d / bank.QUARANTINE_BASENAME)
    sel = data.read_rows(d)
    chosen = select_rows(sel.usable, sel.scorable, OPTS)
    assert {s.row.task.get("variant_id", data.variant_of(s.row)) for s in chosen} == {"ok"}
    assert all(data.variant_of(s.row) not in {"q"} and s.row.episode_id != "masked" for s in chosen)


# ── stated confidence ────────────────────────────────────────────────────────


def test_kept_confidence_is_recalibrated_not_the_selected_rollouts_own():
    """Only the successes are kept, so their own "0.95" would be taught after every claim: the stated confidence
    becomes the held-out pass rate of the claim type (abstain: the complement), and the bank row is untouched."""
    row = make_row("v", passed=True, original={"status": "complete", "confidence": 0.95}, turns=[("ls", "in.txt")])
    sel = bestofn.Selection(row, 1.2, 0.7, 2, 1)
    table = bestofn.hindsight.ConfidenceTable(by_claim={"complete": 0.4, "abstain": 0.1})
    ex = bestofn.build_example(sel, table, keep_confidence=False)
    assert ex.messages[-1]["tool_calls"][0]["function"]["arguments"]["confidence"] == 0.4
    assert ex.meta["confidence_rewritten"] and ex.loss_messages[-1] == len(ex.messages) - 1
    assert row.report_call["arguments"]["confidence"] == 0.95  # not mutated
    assert bestofn.build_example(sel).messages[-1]["tool_calls"][0]["function"]["arguments"]["confidence"] == 0.95
    no_table = bestofn.build_example(sel, None, keep_confidence=False)  # nothing to calibrate with: neutral
    assert no_table.messages[-1]["tool_calls"][0]["function"]["arguments"]["confidence"] == 0.5
    abstain = {"reason": "no jq", "blocker_category": "missing_capability", "blocker_target": "jq", "confidence": 0.99}
    arow = make_row("a", feasible=False, tool="abstain", original=abstain, expected_blockers=["missing_capability"],
                    turns=NOT_FOUND)  # fmt: skip
    ex = bestofn.build_example(bestofn.Selection(arow, 1.0, 0.5, 2, 1), table, keep_confidence=False)
    assert ex.messages[-1]["tool_calls"][0]["function"]["arguments"]["confidence"] == pytest.approx(0.9)
    assert ex.messages[-1]["tool_calls"][0]["function"]["name"] == "abstain"


def test_stage_rewrites_confidence_by_default_and_keep_leaves_it(tmp_path):
    def stated(stage):
        return {e.messages[-1]["tool_calls"][0]["function"]["arguments"].get("confidence") for e in stage.train}

    rows = []
    for i in range(24):
        rows.append(make_row(f"v{i}", passed=True, original={"status": "complete", "confidence": 0.97}))
        rows.append(
            make_row(f"v{i}", passed=False, original={"status": "incomplete", "confidence": 0.3}, turns=NOT_FOUND)
        )
    bank = bank_dir(tmp_path / "b", rows)
    calibrated, _ = train_sft.build_stage_data(tiny_config(tmp_path, **{"kind": "bestofn", "data.bank": bank}))
    kept, _ = train_sft.build_stage_data(
        tiny_config(tmp_path, **{"kind": "bestofn", "data.bank": bank, "bestofn": "{confidence: keep}"})
    )
    assert stated(kept) == {0.97}
    assert stated(calibrated) and 0.97 not in stated(calibrated)
    with pytest.raises(ValueError, match="confidence"):
        BestOfNOptions(confidence="maybe")
