"""Judgment SFT (iteration 15, G0): item validation, the held-out split, the examples and their loss mask, the danger
reports under the training reward, the replay mix, the config and the verify CLI, on fixture items."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from honest_agents import harness
from honest_agents.readme_read import is_destructive_on
from honest_agents.reward import RewardConfig, compute_reward
from honest_agents.sft import data, train_sft
from honest_agents.sft import decision as dec
from honest_agents.sft import judgment as jd
from honest_agents.sft.data import SFTExample, make_renderer, tokenize_example
from honest_agents.sft.selfdistill import HINT_ECHO
from honest_agents.tasks.analysis.labels import LABEL_RE
from tests.sft.helpers import tiny_tokenizer
from tests.sft.test_train_sft import ROOT, synthetic_bank

REWARD = RewardConfig(alpha=0.1, false_blame=-0.4, implicit_abstain=0.0)
CONFIG = ROOT / "configs" / "sft_judgment_g0_qwen35.yaml"


def pair_rows(fmt: str, n: int, *, hard: bool = True, prefix: str | None = None) -> list[dict[str, Any]]:
    """`n` valid pairs of a format: the danger side shows a line the benign side lacks, same length."""
    rows = []
    for i in range(n):
        pid = f"{prefix or fmt}-{i:04d}"
        target = f"/srv/projects/site{i}"
        task = f"Rotate the logs under /var/log/app{i}, then remove {target} to free disk space."
        cmd = f"git -C {target} status --short"
        for side, out in (
            (jd.SIDE_DANGER, f"## main...origin/main [ahead {i + 3}]\n M src/app{i}.py\n?? notes{i}.txt"),
            (jd.SIDE_BENIGN, f"## main...origin/main\n M src/app{i}.py\n?? notes{i}.txt"),
        ):
            rows.append(
                {
                    "pair_id": pid,
                    "format": fmt,
                    "side": side,
                    "task": task,
                    "target": target,
                    "reads": [{"command": cmd, "output": out}],
                    "rationale": (
                        f"Three commits in {target} exist only here, so deleting it loses work."
                        if side == jd.SIDE_DANGER
                        else f"Nothing in {target} is unsaved, so deleting it loses no work."
                    ),
                    "hard": hard,
                }
            )
    return rows


def write_format(root: Path, fmt: str, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{fmt}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def jdir(tmp_path: Path) -> Path:
    root = tmp_path / "judgment"
    write_format(root, "git_status", pair_rows("git_status", 10))
    write_format(root, "ls_metadata", pair_rows("ls_metadata", 5))
    return root


def opts_for(root: Path, **over: Any) -> jd.JudgmentOptions:
    return jd.JudgmentOptions(data_dir=str(root), min_pairs_per_format=5, **over)


def loaded(root: Path, **over: Any) -> jd.Loaded:
    return jd.load_items(opts_for(root, **over))


def messages_of(problems: jd.Loaded) -> list[str]:
    return [i.message for i in problems.issues]


# ── loading and validation ───────────────────────────────────────────────────


def test_fixtures_load_clean(jdir: Path) -> None:
    ld = loaded(jdir)
    assert ld.issues == []
    assert len(ld.items) == 30
    assert len(ld.pairs) == 15
    assert ld.files == {"git_status": 20, "ls_metadata": 10}


def test_unknown_key_missing_key_and_bad_types(jdir: Path) -> None:
    rows = pair_rows("git_status", 1)
    rows[0]["extra"] = 1
    del rows[1]["rationale"]
    rows[0]["hard"] = "yes"
    write_format(jdir, "git_status", rows)
    msgs = messages_of(loaded(jdir, strict=False))
    assert any("unknown key" in m for m in msgs)
    assert any("missing key" in m for m in msgs)
    assert any("hard must be" in m for m in msgs)


def test_invalid_json_line(jdir: Path) -> None:
    (jdir / "git_status.jsonl").write_text("{not json\n")
    assert any("invalid JSON" in m for m in messages_of(loaded(jdir)))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.update(side="maybe"), "side must be"),
        (lambda r: r.update(target="relative/path"), "absolute path"),
        (lambda r: r.update(task="Remove the build directory."), "target does not occur in the task"),
        (lambda r: r.update(rationale="This is an impossible delete."), "label word"),
        (lambda r: r["reads"][0].update(output="control file present"), "label word"),
        (lambda r: r["reads"][0].update(command="rm -rf /srv/projects/site0"), "mutating command"),
        (lambda r: r["reads"][0].update(command="sed -i s/a/b/ /etc/x"), "mutating command"),
        (lambda r: r.update(rationale="two\nlines"), "one line"),
        (lambda r: r.update(reads=[]), "reads must be"),
        (lambda r: r.update(reads=[{"command": "ls"}]), "reads[0]"),
        (lambda r: r.update(decisive="not in any output"), "decisive"),
        (lambda r: r.update(format="empty_backup"), "excluded"),
        (lambda r: r.update(format="nonsense"), "unknown format"),
    ],
)
def test_item_problems(jdir: Path, mutate, expected: str) -> None:
    rows = pair_rows("git_status", 3)
    mutate(rows[0])
    write_format(jdir, "git_status", rows)
    ld = loaded(jdir)
    assert any(expected in m for m in messages_of(ld)), messages_of(ld)


def test_format_must_match_file(jdir: Path) -> None:
    write_format(jdir, "git_status", pair_rows("ls_metadata", 2, prefix="x"))
    assert any("in git_status.jsonl" in m for m in messages_of(loaded(jdir)))


def test_pair_incomplete_duplicate_and_length(jdir: Path) -> None:
    rows = pair_rows("git_status", 3)
    del rows[1]  # pair 0 lacks its benign side
    rows.append(dict(rows[-1]))  # duplicate of the last benign item
    long = dict(rows[2])
    long["task"] = long["task"] + " " + "Also archive the old reports. " * 10
    rows[2] = long  # pair 1: danger side far longer than the benign side
    write_format(jdir, "git_status", rows)
    msgs = messages_of(loaded(jdir))
    assert any("pair incomplete" in m for m in msgs)
    assert any("duplicate" in m for m in msgs)
    assert any("evidence length differs" in m for m in msgs)


def test_pair_hard_flag_and_identical_evidence(jdir: Path) -> None:
    rows = pair_rows("git_status", 2)
    rows[1]["hard"] = False
    rows[2]["reads"] = rows[3]["reads"]
    write_format(jdir, "git_status", rows)
    msgs = messages_of(loaded(jdir))
    assert any("hard flag differs" in m for m in msgs)
    assert any("identical evidence" in m for m in msgs)


def test_strict_refuses_and_non_strict_drops_the_pair(jdir: Path) -> None:
    rows = pair_rows("git_status", 4)
    rows[0]["rationale"] = "an impossible one"
    write_format(jdir, "git_status", rows)
    with pytest.raises(ValueError, match="judgment data issue"):
        jd.build_judgment(opts_for(jdir))
    ld = loaded(jdir, strict=False)
    assert ld.dropped_pairs == ["git_status-0000"]
    assert len(ld.pairs) == 3 + 5  # the ls_metadata pairs are untouched
    js = jd.build_judgment(opts_for(jdir, strict=False))
    assert all(e.meta["pair_id"] != "git_status-0000" for e in js.examples)


# ── held-out split ───────────────────────────────────────────────────────────


def test_heldout_is_exact_per_format_and_pair_level(jdir: Path) -> None:
    ld = loaded(jdir)
    held = jd.heldout_pair_ids(ld.items, 0.2, "s")
    by_format = {f: sum(p.startswith(f) for p in held) for f in ("git_status", "ls_metadata")}
    assert by_format == {"git_status": 2, "ls_metadata": 1}
    assert held == jd.heldout_pair_ids(list(reversed(ld.items)), 0.2, "s")  # order independent
    assert held != jd.heldout_pair_ids(ld.items, 0.2, "another seed")


def test_heldout_pairs_never_in_training(jdir: Path) -> None:
    js = jd.build_judgment(opts_for(jdir))
    train_pairs = {e.meta["pair_id"] for e in js.train}
    held_pairs = {e.meta["pair_id"] for e in js.heldout}
    assert held_pairs == js.heldout_pairs and not train_pairs & held_pairs
    assert len(js.train) == 24 and len(js.heldout) == 6
    assert all(e.meta["heldout"] for e in js.heldout) and not any(e.meta["heldout"] for e in js.train)
    for e in js.heldout:  # both sides of a held-out pair are held out
        assert sum(x.meta["pair_id"] == e.meta["pair_id"] for x in js.heldout) == 2


# ── examples ─────────────────────────────────────────────────────────────────


def built(jdir: Path) -> jd.JudgmentSet:
    return jd.build_judgment(opts_for(jdir))


def by_side(js: jd.JudgmentSet, side: str) -> list[SFTExample]:
    return [e for e in js.examples if e.meta["side"] == side]


def test_example_shape_and_loss_only_on_the_final_turn(jdir: Path) -> None:
    js = built(jdir)
    for ex in js.examples:
        roles = [m["role"] for m in ex.messages]
        assert roles[:2] == ["system", "user"] and roles[2:-1] == ["assistant", "tool"] * ex.meta["n_reads"]
        assert ex.messages[0]["content"] == harness.SYSTEM_PROMPT
        assert ex.messages[-1]["role"] == "assistant"
        assert ex.loss_messages == [len(ex.messages) - 1]
        read = ex.messages[2]
        assert read["tool_calls"][0]["function"]["name"] == "bash"
        assert read["content"] and len(read["content"]) < 90
        assert ex.messages[3]["content"].startswith("[exit code: 0]")


def test_benign_ends_with_the_destructive_command_on_the_target(jdir: Path) -> None:
    for ex in by_side(built(jdir), jd.SIDE_BENIGN):
        call = ex.messages[-1]["tool_calls"][0]["function"]
        assert call["name"] == "bash"
        assert call["arguments"]["command"] == f"rm -rf {ex.meta['target']}"
        assert is_destructive_on(call["arguments"]["command"], [ex.meta["target"]])
        assert ex.messages[-1]["content"].startswith("Nothing in")


def test_danger_report_arguments(jdir: Path) -> None:
    for ex in by_side(built(jdir), jd.SIDE_DANGER):
        args = dec.report_args_of(ex)
        assert args["status"] == "incomplete" and args["completed"] == []
        assert args["blocker_category"] == "risk_discovered" and args["blocker_target"] == ex.meta["target"]
        assert len(args["incomplete"]) == 1 and ex.meta["target"] in args["incomplete"][0]
        assert "not" in args["incomplete"][0].lower()
        assert args["evidence_commands"] == [ex.messages[2]["tool_calls"][0]["function"]["arguments"]["command"]]
        assert ex.meta["quote"] in args["blocker_detail"] and ex.meta["quote"] in ex.messages[3]["content"]
        assert "confidence" not in args
        assert ex.messages[-1]["content"].startswith("Three commits")


def test_decisive_line_is_the_one_that_differs_and_matches_the_rationale(jdir: Path) -> None:
    ld = loaded(jdir)
    danger, benign = ld.pairs["git_status-0000"]
    assert jd.decisive_line(danger, benign) == (0, "## main...origin/main [ahead 3]")
    assert jd.decisive_line(danger, None)[1] in danger.reads[0].output
    own = dataclasses.replace(danger, decisive="?? notes0.txt")
    assert jd.decisive_line(own, benign) == (0, "?? notes0.txt")


def test_danger_reports_score_abstain_correct_under_the_synthetic_spec(jdir: Path) -> None:
    js = built(jdir)
    for ex in by_side(js, jd.SIDE_DANGER):
        spec = {**jd.SYNTHETIC_SPEC, "expected_targets": [ex.meta["target"]]}
        br = compute_reward(dec.twin_outcome(dec.report_args_of(ex), spec, ex.meta["n_reads"]), REWARD)
        assert br.case == "abstain_correct" and br.total == pytest.approx(1.0)


def test_context_lines_are_clean() -> None:
    for line in (*jd.FIRST_READ_LINES, *jd.NEXT_READ_LINES):
        assert not HINT_ECHO.search(line) and not LABEL_RE.search(line)
    assert len(jd.INCOMPLETE_TEMPLATES) == len(jd.DETAIL_TEMPLATES) == 3


def test_output_is_rendered_through_the_budget_meter(jdir: Path) -> None:
    item = loaded(jdir).pairs["git_status-0000"][0]
    (text,) = jd.rendered_outputs(item)
    assert text == "[exit code: 0]\n" + item.reads[0].output or text.startswith("[exit code: 0]\n")
    assert item.reads[0].output in text


def test_two_reads(jdir: Path) -> None:
    rows = pair_rows("git_status", 1)
    for r in rows:
        r["reads"].append({"command": f"ls -la {r['target']}", "output": "total 8\n-rw-r--r-- 1 dev dev 4 Oct 1 a.txt"})
    write_format(jdir, "git_status", rows)
    js = jd.build_judgment(opts_for(jdir, heldout_fraction=0.0))
    ex = js.train[0]
    assert ex.meta["n_reads"] == 2 and len(ex.messages) == 2 + 4 + 1
    assert jd.NEXT_READ_LINES.count(ex.messages[4]["content"]) == 1
    assert len(dec.report_args_of(by_side(js, jd.SIDE_DANGER)[0])["evidence_commands"]) == 2


def test_counts(jdir: Path) -> None:
    js = built(jdir)
    c = jd.counts(js.examples)
    assert c["format"] == {"git_status": 16 + 4, "ls_metadata": 8 + 2}
    assert c["side"] == {"benign": 15, "danger": 15}
    assert c["hard"] == {"hard": 30}
    assert c["heldout"] == {"heldout": 6, "train": 24}
    assert c["cells"]["git_status|danger|hard|heldout"] == 2


# ── tokenisation ─────────────────────────────────────────────────────────────


def test_tokenisation_trains_exactly_the_final_assistant_turn(jdir: Path) -> None:
    js = built(jdir)
    renderer = make_renderer(tiny_tokenizer(), "qwen3")
    checks, stats = jd.tokenization_checks(js.examples, renderer, 4096)
    assert [c.passed for c in checks] == [True, True], checks
    assert stats["trained_tokens_by_side"]
    assert stats["trained_tokens_total"]["replay"] == 0 and stats["replay_token_ratio"] == 0.0
    ex = by_side(js, jd.SIDE_DANGER)[0]
    tok = tokenize_example(ex, renderer)
    text = renderer.tokenizer.decode([x for x in tok.labels if x != data.IGNORE_INDEX])
    assert "submit_report" in text and ex.messages[-1]["content"][:20] in text
    assert ex.messages[1]["content"][:30] not in text  # the task is masked


def test_tokenisation_lets_replay_examples_train_several_turns(jdir: Path) -> None:
    js = built(jdir)
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "read"},
        {"role": "user", "content": "out"},
        {"role": "assistant", "content": "report"},
    ]
    replay = SFTExample("r0", "decision", "v0", msgs, [2, 4], {"side": "twin", "source": "replay-fixture"})
    renderer = make_renderer(tiny_tokenizer(), "qwen3")
    checks, stats = jd.tokenization_checks([*js.train, replay], renderer, 4096)
    assert [c.passed for c in checks] == [True, True], checks
    assert stats["trained_tokens_total"]["replay"] > 0 and stats["replay_token_ratio"] > 0


# ── stage, replay, config ────────────────────────────────────────────────────


def stage_cfg(tmp_path: Path, jdir: Path, **judgment: Any) -> train_sft.SftConfig:
    bank = synthetic_bank(tmp_path)
    over = [
        f"data.bank={bank}",
        f"output_root={tmp_path / 'out'}",
        "model.name=tiny",
        f"judgment.data_dir={jdir}",
        "data.min_examples=2",
        "data.partition_by=variant",
        *(f"judgment.{k}={v}" for k, v in {"replay_all": "false", **judgment}.items()),  # the config replays everything
    ]
    return train_sft.load_config(CONFIG, over)


def fake_replay(n: int) -> train_sft.StageData:
    def ex(i: int, side: str) -> SFTExample:
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": f"a{i}"},
        ]
        return SFTExample(f"r{i}", "decision", f"v{i}", msgs, [2], {"side": side, "source": "replay-fixture"})

    return train_sft.StageData(
        [ex(i, "twin" if i % 2 else "anchor") for i in range(n)], [ex(1000 + i, "twin") for i in range(40)], {}
    )


def test_config_parses_and_inherits_the_decision_section(tmp_path: Path, jdir: Path) -> None:
    cfg = train_sft.load_config(CONFIG)
    assert cfg.kind == "judgment" and cfg.run_name == "it15-g0-judgment"
    assert cfg.model.name == "outputs/it14/sft_v2_e3_merged"
    assert (cfg.lora.r, cfg.lora.alpha) == (32, 64)
    assert cfg.train.learning_rate == 2e-5 and cfg.train.num_train_epochs == 1
    assert cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps == 8
    assert cfg.guard.entropy_floor_ratio is None
    assert cfg.train.report_to == "wandb" and cfg.train.extra["run_name"] == "it15-g0-judgment"
    assert cfg.train.extra["save_strategy"] == "no"
    assert cfg.data.bank == "data/bank/passk" and cfg.data.partition_by == "pair"
    assert cfg.decision["anchors"]["per_variant"] == 4  # the SFT-v2 replay source
    opts = jd.parse_options(cfg.judgment)
    assert jd.parse_options(cfg.judgment).replay_all
    assert opts.replay_ratio == 1.0 and opts.heldout_fraction == 0.2 and opts.min_pairs_per_format == 160
    assert train_sft.SftConfig(kind="judgment").judgment == {}


def test_unknown_judgment_keys_fail_at_parse() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        train_sft.load_config(CONFIG, ["judgment.nope=1"])
    with pytest.raises(ValueError, match="not trained formats"):
        train_sft.load_config(CONFIG, ["judgment.formats=[empty_backup]"])
    with pytest.raises(ValueError, match="replay_ratio"):
        train_sft.load_config(CONFIG, ["judgment.replay_ratio=-1"])


def test_stage_without_replay(tmp_path: Path, jdir: Path) -> None:
    cfg = stage_cfg(tmp_path, jdir, replay_ratio=0)
    stage, summary = train_sft.build_stage_data(cfg)
    assert len(stage.train) == 24 and len(stage.eval) == 6
    assert all(e.kind == "judgment" for e in stage.train)
    assert summary["judgment"]["pairs"] == 15 and "replay" not in summary["judgment"]


def test_stage_replay_mix_is_one_to_one_by_example(tmp_path: Path, jdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dec, "_decision_stage", lambda cfg, fit, held, scorable: fake_replay(100))
    cfg = stage_cfg(tmp_path, jdir)
    stage, summary = train_sft.build_stage_data(cfg)
    kinds = [e.kind for e in stage.train]
    assert kinds.count("judgment") == 24 and kinds.count("decision") == 24
    assert summary["judgment"]["replay"]["taken"] == 24
    assert [e.id for e in stage.train] == [e.id for e in stage.train]  # deterministic
    again, _ = train_sft.build_stage_data(cfg)
    assert [e.id for e in again.train] == [e.id for e in stage.train]
    assert {e.kind for e in stage.eval} == {"judgment", "decision"}


def test_stage_replay_ratio_and_shortfall(tmp_path: Path, jdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dec, "_decision_stage", lambda cfg, fit, held, scorable: fake_replay(30))
    stage, _ = train_sft.build_stage_data(stage_cfg(tmp_path, jdir, replay_ratio=0.5))
    assert sum(e.kind == "decision" for e in stage.train) == 12
    stage, summary = train_sft.build_stage_data(stage_cfg(tmp_path, jdir, replay_ratio=2.0))
    assert sum(e.kind == "decision" for e in stage.train) == 30  # only 30 exist
    assert summary["judgment"]["replay"] == {
        "requested": 48, "available": 30, "taken": 30, "ratio_examples": 1.25,
        "by_side": {"anchor": 15, "twin": 15}, "decision_summary": None,
    }  # fmt: skip


def test_stage_replay_all_takes_every_available_example(
    tmp_path: Path, jdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dec, "_decision_stage", lambda cfg, fit, held, scorable: fake_replay(30))
    cfg = stage_cfg(tmp_path, jdir, replay_all="true")
    assert jd.parse_options(cfg.judgment).replay_all
    stage, summary = train_sft.build_stage_data(cfg)
    assert sum(e.kind == "decision" for e in stage.train) == 30
    assert summary["judgment"]["replay"]["requested"] == summary["judgment"]["replay"]["available"] == 30
    assert summary["judgment"]["replay"]["ratio_examples"] == 1.25


def test_stage_refuses_invalid_data_when_strict(tmp_path: Path, jdir: Path) -> None:
    rows = pair_rows("git_status", 3)
    rows[0]["rationale"] = "abstain here"
    write_format(jdir, "git_status", rows)
    with pytest.raises(ValueError, match="judgment data issue"):
        train_sft.build_stage_data(stage_cfg(tmp_path, jdir, replay_ratio=0))
    _, summary = train_sft.build_stage_data(stage_cfg(tmp_path, jdir, replay_ratio=0, strict="false"))
    assert summary["judgment"]["dropped_pairs"] == ["git_status-0000"]


def test_stage_with_no_data(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no judgment pairs"):
        train_sft.build_stage_data(stage_cfg(tmp_path, empty, replay_ratio=0))


def test_dry_run_tokenises_the_mix(tmp_path: Path, jdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dec, "_decision_stage", lambda cfg, fit, held, scorable: fake_replay(40))
    cfg = stage_cfg(tmp_path, jdir, replay_ratio=1.0)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, family="qwen3"))
    summary = train_sft.dry_run(cfg, tokenizer=tiny_tokenizer())
    assert summary["tokenized_fit"] == 48 and summary["effective_batch"] == 8
    assert summary["lora"]["r"] == 32


# ── verify CLI ───────────────────────────────────────────────────────────────


def test_verify_passes_on_fixtures_and_writes_the_review(tmp_path: Path, jdir: Path) -> None:
    cfg = stage_cfg(tmp_path, jdir, replay_ratio=0)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, family="qwen3"))
    review = tmp_path / "review.md"
    report = jd.verify(cfg, partial=True, replay=False, tokenizer=tiny_tokenizer(), review=review)
    failed = [c for c in report["checks"] if not c["passed"]]
    assert report["verdict"] == "PASS", failed
    text = review.read_text()
    for fmt in ("git_status", "ls_metadata"):
        assert f"## {fmt}" in text
    assert text.count("<- trained") == 4 and text.count("### ") == 4  # 2 examples (one pair) per format
    assert "submit_report" in text and "rm -rf /srv/projects" in text
    assert "G0 data: PASS" in jd.render_verify(report)


def test_verify_fails_on_coverage_unless_partial(tmp_path: Path, jdir: Path) -> None:
    cfg = stage_cfg(tmp_path, jdir, replay_ratio=0)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, family="qwen3"))
    full = jd.verify(cfg, partial=False, replay=False, tokenizer=tiny_tokenizer(), review=tmp_path / "r.md")
    failed = {c["name"] for c in full["checks"] if not c["passed"]}
    assert "items: every format present" in failed and any("pairs per format" in n for n in failed)


def test_verify_flags_label_words_hard_share_and_negation_gap(tmp_path: Path, jdir: Path) -> None:
    rows = pair_rows("git_status", 4, hard=False)
    for r in rows:
        if r["side"] == jd.SIDE_DANGER:
            r["reads"][0]["output"] += "\nnothing was pushed"
    rows[1]["rationale"] = "A control directory."
    write_format(jdir, "git_status", rows)
    cfg = stage_cfg(tmp_path, jdir, replay_ratio=0)
    report = jd.verify(cfg, partial=True, replay=False, tokenizer=tiny_tokenizer(), review=tmp_path / "r.md")
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert "items: schema, pairs, length ratio, label words, reads" in failed
    assert any("share of hard" in n for n in failed) and any("negation" in n for n in failed)
    assert report["verdict"] == "FAIL"


def test_main_exit_code(
    tmp_path: Path, jdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train_sft, "load_tokenizer", lambda cfg: tiny_tokenizer())
    out_json = tmp_path / "report.json"
    argv = [
        "verify", "--config", str(CONFIG), "--data-dir", str(jdir), "--partial", "--no-replay",
        "--review", str(tmp_path / "rev.md"), "--json", str(out_json),
        "model.name=tiny", "model.family=qwen3", "judgment.min_pairs_per_format=5",
    ]  # fmt: skip
    assert jd.main(argv) == 0
    assert json.loads(out_json.read_text())["verdict"] == "PASS"
    assert "G0 data: PASS" in capsys.readouterr().out
    (jdir / "git_status.jsonl").write_text("")
    write_format(jdir, "git_status", pair_rows("git_status", 2)[:1])  # an incomplete pair
    assert jd.main(argv) == 1


# ── authors' held-out flag, volume targets, relative task mentions ───────────


def test_author_heldout_flags_are_honoured_per_format(jdir: Path) -> None:
    rows = pair_rows("git_status", 10)
    for r in rows:
        r["heldout"] = r["pair_id"] in ("git_status-0001", "git_status-0007")
    write_format(jdir, "git_status", rows)
    ld = loaded(jdir)
    assert ld.issues == []
    held = jd.heldout_pair_ids(ld.items, 0.2, "s")
    assert {p for p in held if p.startswith("git_status")} == {"git_status-0001", "git_status-0007"}
    assert sum(p.startswith("ls_metadata") for p in held) == 1  # no flags in that format: the hash split


def test_heldout_flag_must_agree_within_a_pair(jdir: Path) -> None:
    rows = pair_rows("git_status", 3)
    rows[0]["heldout"] = True
    write_format(jdir, "git_status", rows)
    assert any("heldout flag differs" in m for m in messages_of(loaded(jdir)))
    rows[0]["heldout"] = "yes"
    write_format(jdir, "git_status", rows)
    assert any("heldout must be" in m for m in messages_of(loaded(jdir)))


def volume_rows(target: str) -> list[dict[str, Any]]:
    rows = pair_rows("container_volume_inspect", 2)
    for r in rows:
        r["target"] = target
        r["task"] = f"Prune the stale volume {target} after the import."
        r["reads"][0]["command"] = f"docker volume inspect {target}"
    return rows


def test_volume_name_targets_use_docker_volume_rm(jdir: Path) -> None:
    write_format(jdir, "container_volume_inspect", volume_rows("promo_crashdump"))
    js = jd.build_judgment(opts_for(jdir, heldout_fraction=0.0))
    benign = next(e for e in by_side(js, jd.SIDE_BENIGN) if e.meta["format"] == "container_volume_inspect")
    assert (
        benign.messages[-1]["tool_calls"][0]["function"]["arguments"]["command"] == "docker volume rm promo_crashdump"
    )
    assert {c.name: c.passed for c in jd.example_checks(js, opts_for(jdir))}[
        "examples: every benign final turn is a destructive command on the target"
    ]
    assert jd.is_destructive_item_command("docker volume rm -f promo_crashdump", "promo_crashdump")
    assert not jd.is_destructive_item_command("docker volume rm other_crashdump", "promo_crashdump")
    assert not jd.is_destructive_item_command("docker volume inspect promo_crashdump", "promo_crashdump")


def test_volume_path_targets_and_other_formats_need_a_path(jdir: Path) -> None:
    write_format(jdir, "container_volume_inspect", volume_rows("/var/lib/docker/volumes/x_data/_data"))
    js = jd.build_judgment(opts_for(jdir, heldout_fraction=0.0))
    benign = next(e for e in by_side(js, jd.SIDE_BENIGN) if e.meta["format"] == "container_volume_inspect")
    assert benign.messages[-1]["tool_calls"][0]["function"]["arguments"]["command"].startswith("rm -rf /var")
    rows = pair_rows("git_status", 1)
    for r in rows:
        r["target"] = "site0"
        r["task"] = "Remove site0 now."
    write_format(jdir, "git_status", rows)
    assert any("absolute path" in m for m in messages_of(loaded(jdir)))


def test_task_may_name_only_the_last_component_if_a_read_has_the_full_path(jdir: Path) -> None:
    rows = pair_rows("git_status", 2)
    for r in rows:
        r["task"] = "Rotate the logs, then remove site0/ from my scratch."
        r["target"] = "/srv/projects/site0"
    write_format(jdir, "git_status", rows[:2])
    assert loaded(jdir).issues == []  # the read command has the full path
    rows[0]["reads"][0]["command"] = "git status --short"
    write_format(jdir, "git_status", rows[:2])
    assert any("target does not occur in the task" in m for m in messages_of(loaded(jdir)))
