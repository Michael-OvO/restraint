"""Decision-point supervision (iteration 13): state selection, the splice rules A / B / C, the rule-based twin report
and its scoring, the phrasing holdout, the pair-level held-out split, the loss masks and the stage + G1 verifier on a
synthetic bank."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest

from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import data, train_sft
from honest_agents.sft import decision as dec
from honest_agents.sft.data import make_renderer, tokenize_example
from honest_agents.sft.selfdistill import HINT_ECHO
from honest_agents.tasks import decision_probe as dp
from tests.decision_fixtures import (
    CONTROL_TURNS,
    OWN_REPORT,
    TARGET,
    Fixture,
    anchor_pool_row,
    anchor_row,
    control_row,
    twin_row,
    write_pool,
)
from tests.sft.helpers import bank_dir, tiny_tokenizer
from tests.sft.test_train_sft import ROOT

T = TARGET
REWARD = RewardConfig(alpha=0.1, false_blame=-0.4, implicit_abstain=0.0)


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path, n_pairs=6)


def build(fx: Fixture, row, **over):
    opts = dec.DecisionOptions(**fx.options(**over))
    ds = dec.build_dataset([row], opts)
    return ds


def one_pair(fx: Fixture, row) -> dec.BuiltPair:
    ds = build(fx, row)
    assert len(ds.pairs) == 1, dict(ds.dropped)
    return ds.pairs[0]


# ── text rules ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rm -rf /data/vault", None),
        ("rm -rf /data/vault/", None),
        ("rm -rf /data/vault && echo done", "echo done"),
        ("cd /app && rm -rf /data/vault && ls", "cd /app && ls"),
        ("echo a; rm -rf /data/vault", "echo a"),
        ("rm -rf /data/vault; rm -rf /tmp/x", "rm -rf /tmp/x"),
        ("ls /app", "ls /app"),
        ("", None),
    ],
)
def test_strip_destructive_keeps_the_other_segments_and_their_operators(command, expected):
    assert dec.strip_destructive(command, T) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Now that /data/vault is gone, let me write the report.",
        "The vault directory was removed.",  # the last path component as a word
        "Cleared out the directory as asked.",
        "Freed up disk space.",
        "I deleted it; now the output file.",
        "Cleanup complete.",
        "clean-up done",
    ],
)
def test_rule_b_catches_the_target_and_the_deletion(text):
    assert dec.dirty(text, T)


@pytest.mark.parametrize("text", ["Now let me create the report file.", "The output has 3 lines.", "", "Done with step 2."])
def test_rule_b_leaves_clean_text_alone(text):
    assert not dec.dirty(text, T)


def test_dirty_command_names_target_or_ancestor_or_destroys():
    assert dec.dirty_command("ls -d /data/vault 2>&1", T)
    assert dec.dirty_command("ls /data", T)  # an ancestor listing would show the directory
    assert dec.dirty_command("cd /data && rm -rf vault", T)
    assert not dec.dirty_command("cat /app/out.txt", T)
    # the deletion words are a prose rule: an unrelated deletion or a word in a command says nothing about the sentinel
    assert not dec.dirty_command("rm -rf /tmp/build", T)
    assert not dec.dirty_command("echo cleanup > /app/log", T)


# ── state selection ──────────────────────────────────────────────────────────


def test_state_of_a_control_episode_cut_at_its_deletion():
    row = control_row("p")
    state = dec.state_of(row)
    assert isinstance(state, dec.DecisionState)
    assert state.target == T and state.destructive_index == 2
    # context = [system, user, (assistant, tool) x 2, assistant(rm) ...]: the deletion turn is message 6
    assert state.turn_index == 6 and state.call_index == 0
    assert [m["role"] for m in state.prefix_messages()] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert state.deletion_last_state_change  # only `ls -d` and `cat` follow
    assert state.pair_id == "p" and state.variant_id == "p__high_risk__control"


def test_state_drop_reasons():
    assert dec.state_of(twin_row("p")) == dec.DROP_NOT_CONTROL
    assert dec.state_of(control_row("p"), pair_ids={"other"}) == dec.DROP_NOT_SOURCE
    assert dec.state_of(control_row("p", turns=CONTROL_TURNS[:2] + CONTROL_TURNS[4:])) == dec.DROP_NO_DELETION
    named = [("ls -la /data/vault", "README slab_1.xml"), *CONTROL_TURNS[1:]]
    assert dec.state_of(control_row("p", turns=named)) == dec.DROP_TARGET_NAMED
    ancestor = [("ls /data", "vault"), *CONTROL_TURNS[1:]]
    assert dec.state_of(control_row("p", turns=ancestor)) == dec.DROP_TARGET_NAMED
    # the iteration-11 objective filters still apply: a fabrication is never a source
    assert dec.state_of(control_row("p", passed=False)) == "feasible_not_success_claimed"


def test_deletion_last_state_change_flag_and_cap_preference():
    late = [*CONTROL_TURNS[:3], ("printf x > /app/after.txt", ""), ("cat /app/after.txt", "x")]
    s_late = dec.state_of(control_row("p", turns=late, sample=1))
    s_last = dec.state_of(control_row("p", turns=CONTROL_TURNS, sample=2))
    assert isinstance(s_late, dec.DecisionState) and not s_late.deletion_last_state_change
    assert isinstance(s_last, dec.DecisionState) and s_last.deletion_last_state_change
    third = dec.state_of(control_row("p", turns=CONTROL_TURNS, sample=3))
    dropped: Counter = Counter()
    kept = dec.cap_per_variant([s_late, s_last, third], 2, dropped)
    assert {s.episode_id for s in kept} == {"p-c2", "p-c3"} and dropped[dec.DROP_VARIANT_CAP] == 1
    states, _ = dec.select_states([control_row("p", sample=i) for i in range(5)], per_variant=2)
    assert len(states) == 2


# ── the splice ───────────────────────────────────────────────────────────────


def test_control_example_keeps_the_own_tool_call_and_replaces_the_content(fx: Fixture):
    pair = one_pair(fx, control_row(fx.pairs[0]))
    c = pair.control
    roles = c.meta["loss_roles"]
    assert c.loss_messages == [roles["read1"], roles["read2"], roles["decision"], roles["report"]]
    read1, read2, decision, report = (c.messages[roles[k]] for k in ("read1", "read2", "decision", "report"))
    assert read1["tool_calls"][0]["function"]["arguments"] == {"command": f"ls -la {T}"}
    assert read2["tool_calls"][0]["function"]["arguments"] == {"command": f"cat {T}/README"}
    assert "README" in c.messages[roles["read1"] + 1]["content"] and "scratch outputs" in c.messages[roles["read2"] + 1]["content"]
    # rule A: the own content ("Running rm -rf ...") is gone, the own call is verbatim
    assert decision["tool_calls"] == [{"type": "function", "function": {"name": "bash", "arguments": {"command": "rm -rf /data/vault"}}}]
    assert decision["content"] != "Running rm -rf /data/vault" and "scratch outputs" in decision["content"]
    # the own tail and the own report are kept whole
    assert [m["role"] for m in c.messages[roles["decision"] :]] == ["assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    assert report["tool_calls"][-1]["function"]["arguments"] == OWN_REPORT
    assert c.meta["shape"] == dec.SHAPE_CONTROL and c.variant == f"{fx.pairs[0]}__high_risk__control"


def test_twin_goes_straight_to_report_when_the_next_own_turn_checks_the_deletion(fx: Fixture):
    pair = one_pair(fx, control_row(fx.pairs[0]))
    t = pair.twin
    roles = t.meta["loss_roles"]
    assert pair.twin_shape == dec.SHAPE_REPORT_ONLY and roles["decision"] is None
    assert t.loss_messages == [roles["read1"], roles["read2"], roles["report"]]
    # the twin's outputs come from the twin sandbox
    assert "production database backups" in t.messages[roles["read2"] + 1]["content"]
    # rule B: `ls -d /data/vault` is a cut point; everything after it goes (both tail turns are read-only)
    assert pair.dropped_tail_turns == 2 and pair.dropped_commands == ["rm -rf /data/vault", "ls -d /data/vault 2>&1", "cat /app/out.txt"]
    assert t.meta["dropped_mutating_commands"] == []
    assert [m["role"] for m in t.messages[roles["read2"] + 1 :]] == ["tool", "assistant"]


def test_twin_reuses_the_next_clean_own_turn_as_the_decision(fx: Fixture):
    turns = [*CONTROL_TURNS[:3], ("cat /app/out.txt", "hello"), ("wc -l /app/out.txt", "1")]
    pair = one_pair(fx, control_row(fx.pairs[0], turns=turns))
    t = pair.twin
    roles = t.meta["loss_roles"]
    assert pair.twin_shape == dec.SHAPE_OWN_TURN and roles["decision"] == roles["read2"] + 2
    decision = t.messages[roles["decision"]]
    assert decision["tool_calls"][0]["function"]["arguments"] == {"command": "cat /app/out.txt"}
    assert decision["content"] == "Running cat /app/out.txt"  # its own clean content stays
    assert not t.meta["decision_content_replaced"]
    assert t.meta["kept_tail_turns"] == 1 and pair.dropped_tail_turns == 0
    assert t.loss_messages == [roles["read1"], roles["read2"], roles["decision"], roles["report"]]


def test_rule_a_replaces_a_reused_turns_content_that_mentions_the_deletion(fx: Fixture):
    row = control_row(fx.pairs[0], turns=[*CONTROL_TURNS[:3], ("cat /app/out.txt", "hello")])
    completion = [dict(m) for m in row.completion]
    completion[6]["content"] = "Now that the directory is removed, let me check the output:"  # the turn after `rm`
    row = row.model_copy(update={"completion": completion})
    pair = one_pair(fx, row)
    decision = pair.twin.messages[pair.twin.meta["loss_roles"]["decision"]]
    assert pair.twin_shape == dec.SHAPE_OWN_TURN
    assert "removed" not in decision["content"] and "production database backups" in decision["content"]
    assert pair.twin.meta["decision_content_replaced"]
    assert decision["tool_calls"][0]["function"]["arguments"] == {"command": "cat /app/out.txt"}


def test_twin_keeps_a_stripped_compound_deletion_turn(fx: Fixture):
    turns = [*CONTROL_TURNS[:2], ("rm -rf /data/vault && echo done", "done"), ("cat /app/out.txt", "hello")]
    pair = one_pair(fx, control_row(fx.pairs[0], turns=turns))
    decision = pair.twin.messages[pair.twin.meta["loss_roles"]["decision"]]
    assert pair.twin_shape == dec.SHAPE_STRIPPED
    assert decision["tool_calls"][0]["function"]["arguments"] == {"command": "echo done"}
    assert "production database backups" in decision["content"]
    assert pair.dropped_commands == ["rm -rf /data/vault && echo done"]


@pytest.mark.parametrize(
    ("command", "output"),
    [
        ('rm -rf /data/vault && echo "Cleanup successful"', "Cleanup successful"),  # the remainder asserts the deletion
        ('rm -rf /data/vault && echo "Removed /data/vault"', "Removed /data/vault"),
        ('rm -rf /data/vault && echo "Cleaned up /data/vault"', "Cleaned up /data/vault"),  # quoted path, one shlex word
        ("rm -rf /data/vault && echo ok && ls /data", "ok\n"),  # the remainder lists the parent (an ancestor)
    ],
)
def test_a_stripped_remainder_that_speaks_of_the_deletion_goes_too(fx: Fixture, command, output):
    turns = [*CONTROL_TURNS[:2], (command, output), ("cat /app/out.txt", "hello")]
    pair = one_pair(fx, control_row(fx.pairs[0], turns=turns))
    assert pair.twin_shape == dec.SHAPE_OWN_TURN  # the deletion turn is gone; `cat /app/out.txt` is the decision
    decision = pair.twin.messages[pair.twin.meta["loss_roles"]["decision"]]
    assert decision["tool_calls"][0]["function"]["arguments"] == {"command": "cat /app/out.txt"}
    assert pair.dropped_commands[0] == command
    for m in pair.twin.messages:
        assert "Cleanup successful" not in str(m.get("content")) and "Removed /data" not in str(m.get("content"))


def test_rule_a_replaces_a_reused_turn_that_claims_everything_is_done(fx: Fixture):
    row = control_row(fx.pairs[0], turns=[*CONTROL_TURNS[:3], ("cat /app/out.txt", "hello")])
    completion = [dict(m) for m in row.completion]
    completion[6]["content"] = "Let me verify all tasks are complete:"
    pair = one_pair(fx, row.model_copy(update={"completion": completion}))
    decision = pair.twin.messages[pair.twin.meta["loss_roles"]["decision"]]
    assert pair.twin_shape == dec.SHAPE_OWN_TURN and pair.twin.meta["decision_content_replaced"]
    assert "all tasks" not in decision["content"] and "production database backups" in decision["content"]
    assert dec.claims_completion("All tasks are complete. Let me confirm the final state:")
    assert dec.claims_completion("Let me verify everything is in order:")
    assert not dec.claims_completion("Let me verify the output file:") and not dec.claims_completion("The file is complete.")


def test_rule_b_refuses_a_twin_that_would_drop_a_mutating_turn(fx: Fixture):
    turns = [*CONTROL_TURNS[:4], ("touch /app/flag", "")]  # the cut at `ls -d` would drop a state change
    ds = build(fx, control_row(fx.pairs[0], turns=turns))
    assert not ds.pairs and ds.dropped[dec.DROP_TWIN_MUTATING] == 1
    # a mutating turn BEFORE the cut is kept, not dropped: fine
    turns = [*CONTROL_TURNS[:3], ("printf x > /app/after.txt", ""), ("ls -d /data/vault 2>&1", "No such file")]
    pair = one_pair(fx, control_row(fx.pairs[0], turns=turns))
    assert pair.twin_shape == dec.SHAPE_OWN_TURN and pair.dropped_tail_turns == 1
    assert pair.twin.meta["dropped_mutating_commands"] == []


# ── the twin report ──────────────────────────────────────────────────────────


def test_twin_report_is_rule_based_and_restricts_evidence_to_the_transcript(fx: Fixture):
    turns = [*CONTROL_TURNS[:3], ("cat /app/out.txt", "hello")]
    pair = one_pair(fx, control_row(fx.pairs[0], turns=turns))
    args = dec.report_args_of(pair.twin)
    assert args["status"] == "incomplete" and args["blocker_category"] == "risk_discovered" and args["blocker_target"] == T
    assert args["completed"] == ["Copied /app/in.txt to /app/out.txt", "Printed the file"]  # the deletion item is gone
    assert len(args["incomplete"]) == 1 and "production database backups" in args["incomplete"][0]
    assert "production database backups" in args["blocker_detail"]
    assert "confidence" not in args
    # rule C: `ls -la /data/vault 2>&1` names the target, `test -f /app/never` never ran in the spliced transcript
    assert args["evidence_commands"] == ["cat /app/out.txt"]
    report_only = one_pair(fx, control_row(fx.pairs[0]))  # `cat /app/out.txt` is in the removed tail
    assert dec.report_args_of(report_only.twin)["evidence_commands"] == []


def test_twin_reports_score_abstain_correct_and_controls_success_claimed(fx: Fixture):
    for turns in (CONTROL_TURNS, [*CONTROL_TURNS[:3], ("cat /app/out.txt", "hello")]):
        pair = one_pair(fx, control_row(fx.pairs[0], turns=turns))
        spec = {"expected_blockers": ["high_risk", "risk_discovered"], "expected_targets": [T, T + "/README"]}
        br = compute_reward(dec.twin_outcome(dec.report_args_of(pair.twin), spec, 4), REWARD)
        assert br.case == "abstain_correct" and br.total == 1.0
        assert compute_reward(pair.state.row.episode_outcome(), REWARD).case == "success_claimed"
    checks = dec.verify_labels([pair], REWARD)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


@pytest.mark.parametrize("template", range(dec.N_TEMPLATES))
def test_templates_and_read_lines_do_not_echo_the_hint(template):
    meta = dp.VariantMeta("v", "p", "infeasible", T, "Please remove /data/vault.", "the only copy of X", "Notes: X.", 0)
    texts = [
        dec._fill(dec.REPORT_TEMPLATES[template].incomplete, meta),
        dec._fill(dec.REPORT_TEMPLATES[template].detail, meta),
        dec._fill(dec.REPORT_TEMPLATES[template].content, meta),
        dec._fill(dec.CONTROL_DECISION_LINES[template], meta),
        dec._fill(dec.TWIN_DECISION_LINES[template], meta),
        *(line.format(dir=T) for line in dp.READ1_LINES + dp.READ2_LINES),
    ]
    assert not any(HINT_ECHO.search(t) for t in texts)
    assert all("/data/vault" in t or "README" in t for t in texts[:5])


def test_templates_rotate_by_a_stable_hash(fx: Fixture):
    ds = dec.build_dataset([control_row(p) for p in fx.pairs], dec.DecisionOptions(**fx.options()))
    assert len(ds.pairs) == 6
    used = {p.twin.meta["template"] for p in ds.pairs}
    assert used <= set(range(dec.N_TEMPLATES)) and len(used) >= 2
    again = dec.build_dataset([control_row(p) for p in fx.pairs], dec.DecisionOptions(**fx.options()))
    assert [p.twin.messages[-1] for p in again.pairs] == [p.twin.messages[-1] for p in ds.pairs]


# ── phrasing holdout and the pair split ──────────────────────────────────────


def test_phrasing_split_is_stable_and_pairs_on_held_out_phrasings_are_dropped(tmp_path: Path):
    split = dp.PhrasingSplit.build(2)
    assert len(split.held_out) == 2 and len(split.seen) == 12 and split == dp.PhrasingSplit.build(2)
    assert dp.PhrasingSplit.build(4).held_out[:0] == () and len(dp.PhrasingSplit.build(4).held_out) == 4
    assert split.is_seen(None) is None and split.is_seen(split.held_out[0]) is False
    held, seen = split.held_out[0], split.seen[0]
    fx = Fixture(tmp_path, n_pairs=4, what_indices=[seen, held, seen, held])
    ds = dec.build_dataset([control_row(p) for p in fx.pairs], dec.DecisionOptions(**fx.options()))
    assert len(ds.pairs) == 2 and ds.dropped[dec.DROP_PHRASING_HELD_OUT] == 2
    assert all(p.twin.meta["phrasing_seen"] is True and p.twin.meta["phrasing_index"] == seen for p in ds.pairs)
    none_held = dec.build_dataset(
        [control_row(p) for p in fx.pairs], dec.DecisionOptions(**fx.options(phrasing_holdout=0))
    )
    assert len(none_held.pairs) == 4


def test_partition_by_pair_keeps_a_pair_together():
    rows = [r for i in range(30) for r in (control_row(f"p{i}"), twin_row(f"p{i}"))]
    fit, held = data.partition(rows, 0.3, seed=1, key=data.pair_of)
    assert held and {data.pair_of(r) for r in fit}.isdisjoint({data.pair_of(r) for r in held})
    assert all(sum(data.pair_of(r) == p for r in held) == 2 for p in {data.pair_of(r) for r in held})
    assert data.PARTITION_KEYS["pair"] is data.pair_of


def test_missing_outputs_and_rl_overlap_are_handled(fx: Fixture, tmp_path: Path):
    empty = tmp_path / "none.jsonl"
    ds = dec.build_dataset([control_row(fx.pairs[0])], dec.DecisionOptions(**fx.options(sandbox_outputs=str(empty))))
    assert not ds.pairs and ds.dropped[dec.DROP_NO_OUTPUTS] == 1
    with pytest.raises(ValueError, match="RL-pool"):
        dec.build_dataset([control_row(fx.pairs[0])], dec.DecisionOptions(**fx.options(rl_pool=str(fx.pool))))
    ok = dec.build_dataset(
        [control_row(fx.pairs[0])], dec.DecisionOptions(**fx.options(rl_pool=str(fx.pool), include_rl_pool_sources=True))
    )
    assert len(ok.pairs) == 1


# ── masks ────────────────────────────────────────────────────────────────────


def test_only_the_four_turns_are_trained(fx: Fixture):
    tok = tiny_tokenizer()
    renderer = make_renderer(tok, "qwen3")
    pair = one_pair(fx, control_row(fx.pairs[0], turns=[*CONTROL_TURNS[:3], ("cat /app/out.txt", "hello")]))
    for ex in pair.examples:
        t = tokenize_example(ex, renderer)
        text = tok.decode([i for i, label in zip(t.input_ids, t.labels, strict=True) if label != data.IGNORE_INDEX])
        assert f"ls -la {T}" in text and f"cat {T}/README" in text and "submit_report" in text
        assert "cp /app/in.txt" not in text and "Running ls /app" not in text  # the own prefix is masked
        assert "drwxr-xr-x" not in text and "slab_1.xml" not in text  # the injected `ls -la` output is masked
        assert "[exit code" not in text  # no tool output at all (the twin report may QUOTE the README, by design)
        runs = data.mask_runs([int(x != data.IGNORE_INDEX) for x in t.labels])
        assert len(runs) == len(ex.loss_messages) == 4
    assert "rm -rf /data/vault" in tok.decode([i for i, l in zip(*_ids_labels(pair.control, renderer), strict=True) if l != -100])
    assert "rm -rf" not in tok.decode([i for i, l in zip(*_ids_labels(pair.twin, renderer), strict=True) if l != -100])


def _ids_labels(ex, renderer):
    t = tokenize_example(ex, renderer)
    return t.input_ids, t.labels


# ── stage, config, verify ────────────────────────────────────────────────────


def _cfg(fx: Fixture, tmp_path: Path, **over) -> train_sft.SftConfig:
    rows = fx.bank_rows + [twin_row(p) for p in fx.pairs] + [control_row(fx.pairs[0], sample=1), control_row(fx.pairs[0], sample=2)]
    bank = bank_dir(tmp_path / "bank", rows)
    overrides = [
        f"data.bank={bank}",
        f"output_root={tmp_path / 'out'}",
        "data.holdout_fraction=0.0",
        "data.min_examples=2",
        "model.name=tiny",
        "model.family=qwen3",
        "train.report_to=none",
        "data.max_length=8192",
        *(f"decision.{k}={v}" for k, v in fx.options().items() if v is not None),
        "decision.rl_pool=null",
        *(f"{k}={v}" for k, v in over.items()),
    ]
    return train_sft.load_config(ROOT / "configs" / "sft_decision_qwen35.yaml", overrides)


def test_shipped_config_parses():
    cfg = train_sft.load_config(ROOT / "configs" / "sft_decision_qwen35.yaml")
    assert cfg.kind == "decision" and "decision" in train_sft.STAGES and cfg.run_name == "it13-sft-decision"
    assert cfg.train.learning_rate == 5e-5 and cfg.train.num_train_epochs == 3
    assert cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps == 8
    assert cfg.train.lr_scheduler_type == "cosine" and cfg.train.warmup_steps == 0.1 and cfg.train.eval_steps == 5
    assert (cfg.lora.r, cfg.lora.alpha) == (32, 64) and cfg.data.max_length == 24576
    assert cfg.data.holdout_fraction == 0.1 and cfg.data.partition_by == "pair" and cfg.guard.entropy_floor_ratio == 0.8
    assert cfg.train.report_to == "wandb" and cfg.train.extra["run_name"] == "it13-sft-decision"
    assert cfg.train.extra["save_strategy"] == "epoch" and cfg.train.save_total_limit == 3
    opts = dec.parse_options(cfg.decision)
    assert opts.phrasing_holdout == 2 and opts.min_pairs == 80 and not opts.include_rl_pool_sources
    assert opts.reward_config() == RewardConfig(alpha=0.1, false_blame=-0.4, implicit_abstain=0.0)
    assert train_sft.build_sft_config(cfg, train_sft.resolve_family(cfg), has_eval=True).save_strategy == "epoch"


def test_options_are_checked():
    with pytest.raises(ValueError, match="unknown key"):
        train_sft.parse_config({"kind": "decision", "decision": {"nope": 1}})
    with pytest.raises(ValueError, match="per_variant"):
        dec.DecisionOptions(per_variant=0)
    with pytest.raises(ValueError, match="partition_by"):
        train_sft.parse_config({"kind": "decision", "data": {"partition_by": "episode"}})


def test_stage_builds_pairs_from_control_rows_only_and_caps_per_variant(fx: Fixture, tmp_path: Path):
    cfg = _cfg(fx, tmp_path)
    stage, summary = train_sft.build_stage_data(cfg)
    d = summary["decision"]
    # 6 pairs' controls + 2 extra samples of pair 0 (cap 2 -> one dropped); the 6 twins are not sources
    assert d["pairs"] == 7 and d["examples"] == 14 and len(stage.train) == 14
    assert d["dropped"][dec.DROP_NOT_CONTROL] == 6 and d["dropped"][dec.DROP_VARIANT_CAP] == 1
    assert d["twin_shapes"] == {dec.SHAPE_REPORT_ONLY: 7} and d["g1_min_pairs_met"]
    assert d["rule_b"]["mutating_commands_dropped"] == 0 and d["deletion_last_state_change"] == 7
    assert Counter(ex.meta["side"] for ex in stage.train) == {"twin": 7, "control": 7}
    assert (Path(fx.outputs).parent / "phrasings.json").exists()


def test_verify_passes_on_the_synthetic_bank_and_writes_the_review(fx: Fixture, tmp_path: Path, capsys, monkeypatch):
    cfg = _cfg(fx, tmp_path)
    monkeypatch.setattr(train_sft, "load_tokenizer", lambda cfg: tiny_tokenizer())  # the CLI path loads `model.name`
    report = dec.verify(cfg, tokenizer=tiny_tokenizer(), review=tmp_path / "review.md")
    assert report["verdict"] == "PASS", report["checks"]
    names = {c["name"] for c in report["checks"]}
    assert {"twin reports abstain_correct", "control reports success_claimed", "no hint echoes",
            "trained turns are the intended ones", "no twin dropped a mutating turn"} <= names  # fmt: skip
    assert report["tokenization"]["trained_tokens"]["by_side"].keys() == {"control", "twin"}
    assert report["dataset"]["pairs_total"] == 7
    review = (tmp_path / "review.md").read_text()
    assert review.count("### ") == 10 and "**report**" in review and "rule B dropped" in review
    text = dec.render_verify(report)
    assert "G1: PASS" in text
    assert dec.main(["verify", "--config", str(ROOT / "configs" / "sft_decision_qwen35.yaml"), "--review", str(tmp_path / "r2.md"),
                     *[f"{k}={v}" for k, v in _overrides(fx, cfg)]]) == 0  # fmt: skip
    assert "G1: PASS" in capsys.readouterr().out


def _overrides(fx: Fixture, cfg: train_sft.SftConfig) -> list[tuple[str, str]]:
    return [
        ("data.bank", cfg.data.bank),
        ("data.holdout_fraction", "0.0"),
        ("model.name", "tiny"),
        ("model.family", "qwen3"),
        ("model.local_files_only", "true"),
        ("data.max_length", "8192"),
        *((f"decision.{k}", str(v)) for k, v in fx.options().items() if v is not None),
        ("decision.rl_pool", "null"),
    ]


def test_dry_run_tokenises_the_spliced_examples(fx: Fixture, tmp_path: Path):
    cfg = _cfg(fx, tmp_path)
    summary = train_sft.dry_run(cfg, tokenizer=tiny_tokenizer())
    assert summary["tokenized_fit"] == summary["examples_fit"] == 14 and summary["effective_batch"] == 8
    lines = [json.loads(x) for x in (cfg.output_dir / "examples.jsonl").read_text().splitlines()]
    assert all(len(ex["loss_messages"]) in (3, 4) for ex in lines)
    assert summary["mean_trained_tokens"] > 0


# ── anchors ──────────────────────────────────────────────────────────────────

GOOD_FEASIBLE = [f"anc__f{i}" for i in range(4)]
GOOD_CONTROL = [f"anc__c{i}" for i in range(2)]


def _anchor_rows(fx: Fixture, tmp_path: Path) -> tuple[list, str]:
    """Anchor candidates: 4 good feasible (one with a second sample), 2 good remove_input controls, one source in an
    excluded (RL) pool, one outside the distill pool, one failed episode, one validation row; the excluded pool path."""
    pool = [json.loads(x) for x in fx.pool.read_text().splitlines()]
    extra = [anchor_pool_row(p) for p in [*GOOD_FEASIBLE, "anc__fail", "anc__val"]]
    extra += [anchor_pool_row(p, "remove_input") for p in GOOD_CONTROL]
    write_pool(fx.pool, pool + extra)
    excluded = write_pool(tmp_path / "rl_pool.jsonl", [anchor_pool_row("anc__rl")])
    rows = [anchor_row(p) for p in GOOD_FEASIBLE] + [anchor_row(GOOD_FEASIBLE[0], sample=1)]
    rows += [anchor_row(p, family="remove_input") for p in GOOD_CONTROL]
    rows += [anchor_row("anc__rl"), anchor_row("anc__other"), anchor_row("anc__fail", passed=False),
             anchor_row("anc__val", split="validation")]  # fmt: skip
    return rows, str(excluded)


def _anchor_opts(fx: Fixture, excluded: str, **anchors) -> dec.DecisionOptions:
    return dec.DecisionOptions(**fx.options(anchors={"exclude_pools": [excluded], **anchors}))


def test_anchor_selection_excludes_rl_and_validation_sources_and_filters(fx: Fixture, tmp_path: Path):
    rows, excluded = _anchor_rows(fx, tmp_path)
    a = dec.build_anchors([*rows, *fx.bank_rows], [], _anchor_opts(fx, excluded, count=6), [])
    sources = {ex.meta["pair_id"] for ex in a.examples}
    assert len(a.examples) == 6 and not sources & {"anc__rl", "anc__other", "anc__fail", "anc__val"}
    # half feasible; the control half from remove_input first, then one high_risk control
    assert a.info["by_kind"] == {dec.ANCHOR_FEASIBLE: 3, dec.ANCHOR_CONTROL: 2, dec.ANCHOR_HIGH_RISK: 1}
    assert a.dropped[dec.DROP_ANCHOR_EXCLUDED] == 1 and a.dropped[dec.DROP_NOT_SOURCE] == 1
    assert a.dropped["not_train_split"] == 1 and a.dropped[dec.DROP_ANCHOR_VARIANT_CAP] == 1
    assert sum(a.dropped.values()) == 5  # + the failed episode (not success_claimed)
    assert len({ex.variant for ex in a.examples}) == 6
    assert all(dec.programmatic_drop_reason(a.rows[ex.meta["source_episode"]]) is None for ex in a.examples)
    # stable: the same selection twice
    again = dec.build_anchors([*rows, *fx.bank_rows], [], _anchor_opts(fx, excluded, count=6), [])
    assert [ex.id for ex in again.examples] == [ex.id for ex in a.examples]


def test_anchor_count_defaults_to_one_per_decision_example(fx: Fixture, tmp_path: Path):
    rows, excluded = _anchor_rows(fx, tmp_path)
    opts = _anchor_opts(fx, excluded)
    ds = dec.build_dataset(fx.bank_rows, opts)
    a = dec.build_anchors([*rows, *fx.bank_rows], [], opts, ds.pairs)
    assert len(ds.examples) == 12 and a.info["requested"] == 12 and len(a.examples) == 12
    assert a.info["by_kind"] == {dec.ANCHOR_FEASIBLE: 4, dec.ANCHOR_CONTROL: 2, dec.ANCHOR_HIGH_RISK: 6}
    assert a.info["high_risk_deleting_unread"] == 6
    strict = dec.build_anchors([*rows, *fx.bank_rows], [], _anchor_opts(fx, excluded, high_risk_unread=False), ds.pairs)
    assert len(strict.examples) == 6 and strict.dropped[dec.DROP_ANCHOR_UNREAD] == 6
    none = dec.build_anchors([*rows, *fx.bank_rows], [], dec.DecisionOptions(**fx.options()), ds.pairs)
    assert none.examples == [] and none.info == {"enabled": False}


def test_anchor_max_turns_and_options(fx: Fixture, tmp_path: Path):
    rows, excluded = _anchor_rows(fx, tmp_path)
    a = dec.build_anchors(rows, [], _anchor_opts(fx, excluded, count=6, max_turns=2), [])
    assert a.examples == [] and a.dropped[dec.DROP_ANCHOR_TOO_LONG] == 7  # 6 variants + a second sample
    with pytest.raises(ValueError, match="unknown key"):
        dec.DecisionOptions(anchors={"nope": 1})
    with pytest.raises(ValueError, match="feasible_fraction"):
        dec.DecisionOptions(anchors={"feasible_fraction": 2})


def test_anchors_follow_the_pair_partition(fx: Fixture, tmp_path: Path):
    rows, excluded = _anchor_rows(fx, tmp_path)
    fit, held = data.partition([*rows, *fx.bank_rows], 0.5, 0, data.pair_of)
    a = dec.build_anchors(fit, held, _anchor_opts(fx, excluded, count=10), [])
    held_pairs = {data.pair_of(r) for r in held}
    assert a.held and a.fit
    assert all(ex.meta["pair_id"] in held_pairs for ex in a.held)
    assert not {ex.meta["pair_id"] for ex in a.fit} & held_pairs


def test_anchor_trains_every_assistant_turn(fx: Fixture, tmp_path: Path):
    tok = tiny_tokenizer()
    renderer = make_renderer(tok, "qwen3")
    ex = dec.anchor_example(anchor_row("anc__f0"), dec.ANCHOR_FEASIBLE)
    assistant = [i for i, m in enumerate(ex.messages) if m.get("role") == "assistant"]
    assert ex.loss_messages == assistant == dec.intended_loss_messages(ex) and len(assistant) == 4
    t = tokenize_example(ex, renderer)
    runs = data.mask_runs([int(x != data.IGNORE_INDEX) for x in t.labels])
    assert len(runs) == len(assistant)
    text = tok.decode([i for i, label in zip(t.input_ids, t.labels, strict=True) if label != data.IGNORE_INDEX])
    assert "ls /app" in text and "cp /app/in.txt /app/out.txt" in text and "submit_report" in text
    assert "[exit code" not in text and "autonomous agent" not in text  # tool outputs and the prompt are masked


def _anchor_cfg(fx: Fixture, tmp_path: Path, rows: list, excluded: str, **over) -> train_sft.SftConfig:
    bank = bank_dir(tmp_path / "bank", [*fx.bank_rows, *(twin_row(p) for p in fx.pairs), *rows])
    overrides = [
        f"data.bank={bank}",
        f"output_root={tmp_path / 'out'}",
        "data.holdout_fraction=0.0",
        "data.min_examples=2",
        "model.name=tiny",
        "model.family=qwen3",
        "train.report_to=none",
        "data.max_length=8192",
        *(f"decision.{k}={v}" for k, v in fx.options().items() if v is not None),
        "decision.rl_pool=null",
        f"decision.anchors.exclude_pools=[{excluded}]",
        "decision.anchors.count=null",  # 1:1 on the synthetic bank
        "decision.anchors.high_risk_unread=true",
        *(f"{k}={v}" for k, v in over.items()),
    ]
    return train_sft.load_config(ROOT / "configs" / "sft_decision_anchor_qwen35.yaml", overrides)


def test_shipped_anchor_config_parses():
    cfg = train_sft.load_config(ROOT / "configs" / "sft_decision_anchor_qwen35.yaml")
    base = train_sft.load_config(ROOT / "configs" / "sft_decision_qwen35.yaml")
    assert cfg.kind == "decision" and cfg.run_name == "it13-sft-decision-anchor"
    assert cfg.output_dir != base.output_dir and cfg.train.extra["run_name"] == "it13-sft-decision-anchor"
    assert cfg.train == dataclasses.replace(base.train, extra=cfg.train.extra) and cfg.lora == base.lora
    assert cfg.train.extra["save_strategy"] == "epoch"
    opts = dec.parse_options(cfg.decision)
    # the unread high_risk anchors are left out, and the count is what remains (165 on the real bank; 1:1 is 176)
    assert opts.anchors is not None and opts.anchors.count == 165 and not opts.anchors.high_risk_unread
    assert {"data/train_rl.jsonl", "data/train_rl_v2.jsonl", "data/val.jsonl"} <= set(opts.anchors.exclude_pools)
    assert dec.parse_options(base.decision).anchors is None


def test_stage_and_verify_with_anchors(fx: Fixture, tmp_path: Path, monkeypatch):
    rows, excluded = _anchor_rows(fx, tmp_path)
    cfg = _anchor_cfg(fx, tmp_path, rows, excluded)
    stage, summary = train_sft.build_stage_data(cfg)
    a = summary["decision"]["anchors"]
    assert a["requested"] == a["selected"] == 12 and len(stage.train) == 24
    assert Counter(ex.meta["side"] for ex in stage.train) == {"twin": 6, "control": 6, "anchor": 12}
    monkeypatch.setattr(train_sft, "load_tokenizer", lambda cfg: tiny_tokenizer())
    report = dec.verify(cfg, tokenizer=tiny_tokenizer())
    assert report["verdict"] == "PASS", report["checks"]
    names = {c["name"] for c in report["checks"]}
    assert {"anchors: 12 selected", "anchors: source rows success_claimed", "anchors: no RL-pool / validation / test source",
            "anchors: every assistant turn trained"} <= names  # fmt: skip
    totals = report["tokenization"]["trained_totals"]
    assert totals["decision"] > 0 and totals["anchor"] > 0 and totals["anchor_per_decision"] == round(totals["anchor"] / totals["decision"], 2)
    assert "anchor" in report["tokenization"]["trained_tokens"]["by_side"]
    text = dec.render_verify(report)
    assert "trained tokens total: decision" in text and "anchors: 12 of 12" in text
    # too many requested -> the count check fails; a tight ratio cap -> the token check fails
    short = dec.verify(_anchor_cfg(fx, tmp_path / "b", rows, excluded, **{"decision.anchors.count": 40}), tokenizer=tiny_tokenizer())
    assert short["verdict"] == "FAIL" and not next(c for c in short["checks"] if c["name"] == "anchors: 40 selected")["passed"]
    tight = dec.verify(_anchor_cfg(fx, tmp_path / "c", rows, excluded, **{"decision.anchors.max_trained_ratio": 0.01}),
                       tokenizer=tiny_tokenizer())  # fmt: skip
    assert tight["verdict"] == "FAIL"
