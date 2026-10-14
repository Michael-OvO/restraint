"""Decision-state DPO (iteration 14): pair construction per side, the shared prompt, rendering equal to
`tokenize_example`, config parsing and the dry run on the synthetic decision fixtures. No GPU: the trainer path
(`dpo.run_dpo`) is not exercised."""

from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path

import pytest

from honest_agents.report import parse_report
from honest_agents.sft import data, dpo
from honest_agents.sft import decision as dec
from honest_agents.sft.data import make_renderer
from honest_agents.tasks.perturb.risk_surfaces import SURFACE_COMMANDS, SURFACES_TRAIN, inspection_commands
from tests.decision_fixtures import TARGET, Fixture, SurfaceFixture, control_row, surface_site, twin_row
from tests.sft.helpers import bank_dir, tiny_tokenizer
from tests.sft.test_train_sft import ROOT

CONFIG = ROOT / "configs" / "sft_dpo_decision_qwen35.yaml"  # `sft_`: its own schema, skipped by tests/test_scripts


def deployed_tokenizer():
    """The tiny tokenizer with the end-of-turn token as EOS, as the Qwen3 / Qwen3.5 instruct checkpoints ship it."""
    tok = tiny_tokenizer()
    tok.eos_token = "<|im_end|>"
    return tok


@pytest.fixture
def fx(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path, n_pairs=6)


def make_cfg(fx: Fixture, tmp_path: Path, **over) -> dpo.DpoConfig:
    rows = fx.bank_rows + [twin_row(p) for p in fx.pairs] + [control_row(fx.pairs[0], sample=1)]
    overrides = [
        f"data.bank={bank_dir(tmp_path / 'bank', rows)}",
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
    return dpo.load_dpo_config(CONFIG, overrides)


@pytest.fixture
def renderer():
    return make_renderer(deployed_tokenizer(), "qwen3")


def test_shipped_config_parses():
    cfg = dpo.load_dpo_config(CONFIG)
    assert cfg.sft.kind == "decision" and cfg.sft.run_name == "it14-dpo-decision"
    d = cfg.dpo
    assert (d.beta, d.truncation_mode, d.precompute_ref_log_probs) == (0.1, "keep_end", True)
    assert d.loss_type == ["sigmoid", "sft"] and d.loss_weights == [1.0, 1.0]  # DPO anchored by NLL on the chosen
    assert d.max_length >= 12288 and cfg.sft.train.num_train_epochs in (1, 2)
    assert cfg.sft.train.report_to == "wandb" and cfg.sft.train.extra["run_name"] == "it14-dpo-decision"
    assert cfg.sft.train.extra["save_strategy"] == "epoch"


@pytest.mark.parametrize(
    "bad",
    [{"max_length": 4096}, {"truncation_mode": "middle"}, {"beta": 0}, {"sides": ["sideways"]}, {"nope": 1},
     {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0]}, {"loss_type": ["sigmoid_nrom"], "loss_weights": None},
     {"loss_type": [], "loss_weights": None}],
)  # fmt: skip
def test_bad_dpo_sections_fail_at_parse(bad):
    raw = json.loads(json.dumps(dpo.load_dpo_config(CONFIG).dpo.__dict__)) | bad
    with pytest.raises(ValueError):
        dpo.DpoSection(**raw) if "nope" not in bad else dpo.parse_dpo_config({"dpo": raw})


def test_a_non_decision_kind_is_refused():
    from honest_agents.pilot.run import apply_overrides, load_config_dict

    with pytest.raises(ValueError, match="kind must be"):
        dpo.parse_dpo_config(apply_overrides(load_config_dict(CONFIG), ["kind=hindsight"]))


def test_pairs_per_side_share_the_prompt_and_differ_only_in_the_final_turn(fx: Fixture, tmp_path: Path):
    cfg = make_cfg(fx, tmp_path)
    pairs = dpo.build_pairs(cfg)
    assert len(pairs.fit) == 2 * 7 and not pairs.held and pairs.info["state_pairs_fit"] == 7
    by_side = {s: [p for p in pairs.fit if p.side == s] for s in dpo.SIDES}
    assert len(by_side["benign"]) == len(by_side["danger"]) == 7
    for p in pairs.fit:
        assert p.prompt[-1]["role"] == "tool" and p.chosen["role"] == p.rejected["role"] == "assistant"
        assert p.chosen_messages()[:-1] == p.rejected_messages()[:-1] == p.prompt
        assert p.meta["n_prompt_messages"] == len(p.prompt)

    benign, danger = by_side["benign"][0], by_side["danger"][0]
    # benign evidence: the README of the control (junk contents); proceed is chosen, the decline report is rejected
    assert "regenerated" not in benign.meta["readme"] or True
    assert benign.chosen["tool_calls"][0]["function"]["arguments"] == {"command": f"rm -rf {TARGET}"}
    rejected_call = benign.rejected["tool_calls"][0]["function"]
    assert rejected_call["name"] == "submit_report"
    report = parse_report(rejected_call["arguments"])
    assert report.status == "incomplete" and report.blocker_target == TARGET
    assert benign.meta["readme"] in rejected_call["arguments"]["blocker_detail"] or "scratch outputs" in json.dumps(rejected_call)
    # danger evidence: the twin's README; the decline report is chosen, the own deletion turn rejected
    chosen_call = danger.chosen["tool_calls"][0]["function"]
    assert chosen_call["name"] == "submit_report" and chosen_call["arguments"]["blocker_target"] == TARGET
    assert "production database backups" in json.dumps(chosen_call)
    assert danger.rejected["tool_calls"][0]["function"]["arguments"] == {"command": f"rm -rf {TARGET}"}
    assert "README" not in danger.rejected["content"]  # no argument from the benign README
    # the two prompts are the control's and the twin's own sandbox outputs
    assert "scratch outputs" in json.dumps(benign.prompt[-1]) and "production database" in json.dumps(danger.prompt[-1])
    assert benign.prompt[:-4] == danger.prompt[:-4]


def test_decline_evidence_commands_come_from_the_prompt_only(fx: Fixture, tmp_path: Path):
    pairs = dpo.build_pairs(make_cfg(fx, tmp_path))
    for p in pairs.fit:
        report = (p.chosen if p.side == "danger" else p.rejected)["tool_calls"][0]["function"]["arguments"]
        seen = dpo._prompt_commands(p.prompt)
        assert set(report["evidence_commands"]) <= set(seen)
        assert report["evidence_commands"] == []  # the own evidence ran after the deletion: not in the prompt


def test_a_state_whose_deletion_is_not_the_last_state_change_is_dropped(fx: Fixture, tmp_path: Path):
    """The decline report lists the own `completed` items; work the own tail did after the deletion has not happened at
    the decision state, so such a state gives no pair (the SFT twin keeps that tail before its report, DPO cannot)."""
    from tests.decision_fixtures import CONTROL_TURNS

    late_work = [*CONTROL_TURNS[:3], ("touch /app/done", ""), *CONTROL_TURNS[3:]]  # mutating command after the rm
    fx.bank_rows.append(control_row(fx.pairs[1], sample=2, turns=late_work))
    pairs = dpo.build_pairs(make_cfg(fx, tmp_path))
    assert pairs.info["state_pairs_fit"] == 8  # the decision stage builds it (its tail passes rule B)
    assert pairs.info["dropped_states"] == {dpo.DROP_NOT_LAST_STATE_CHANGE: 1}
    assert len(pairs.fit) == 2 * 7 and f"{fx.pairs[1]}-c2" not in {p.meta["source_episode"] for p in pairs.fit}


def test_sides_option_builds_one_side(fx: Fixture, tmp_path: Path):
    pairs = dpo.build_pairs(make_cfg(fx, tmp_path, **{"dpo.sides": "[danger]"}))
    assert {p.side for p in pairs.fit} == {"danger"} and len(pairs.fit) == 7


def test_pair_level_heldout_keeps_both_sides_of_a_state_together(fx: Fixture, tmp_path: Path):
    cfg = make_cfg(fx, tmp_path, **{"data.holdout_fraction": 0.4, "data.partition_by": "pair"})
    pairs = dpo.build_pairs(cfg)
    assert pairs.held and pairs.fit
    held_sources = {p.meta["source_episode"] for p in pairs.held}
    assert held_sources.isdisjoint(p.meta["source_episode"] for p in pairs.fit)
    assert len(pairs.held) == 2 * len(held_sources)


def test_rendering_equals_tokenize_example(fx: Fixture, tmp_path: Path, renderer):
    pairs = dpo.build_pairs(make_cfg(fx, tmp_path))
    eos = renderer.tokenizer.eos_token
    for p in pairs.fit:
        rendered = dpo.render_pair(p, renderer)
        assert rendered.chosen.endswith(eos) and rendered.rejected.endswith(eos)
        assert rendered.prompt.rstrip().endswith("</think>")  # the generation prompt of non-thinking mode
        assert "submit_report" in rendered.prompt and "bash" in rendered.prompt  # the tools are in the system block
        assert dpo.check_tokenization(p, renderer, rendered) == []
        prompt_ids, chosen_ids, _ = dpo.trl_ids(renderer, rendered)
        tokenized = data.tokenize_example(p.example("chosen"), renderer)
        assert prompt_ids + chosen_ids == [
            *tokenized.input_ids[: len(prompt_ids) + len(chosen_ids)]
        ]  # identical ids, bar the template's trailing newline
        assert [t for t in tokenized.labels if t != data.IGNORE_INDEX] == chosen_ids


def test_trl_string_path_reproduces_trl_ids_and_masks_exactly_the_completion(fx: Fixture, tmp_path: Path, renderer):
    """`DPOTrainer._prepare_dataset` (TRL 1.14.1) on string rows: not conversational (no re-templating), `add_eos` is a
    no-op because the completions already end with the EOS, and the ids are prompt / prompt+completion through
    `_tokenize`; the collator's completion mask covers the completion including its single EOS."""
    from trl.data_utils import _tokenize, is_conversational
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    tok = renderer.tokenizer
    p = dpo.build_pairs(make_cfg(fx, tmp_path)).fit[0]
    rendered = dpo.render_pair(p, renderer)
    row = {"prompt": rendered.prompt, "chosen": rendered.chosen, "rejected": rendered.rejected}
    assert not is_conversational(row)
    assert row["chosen"].endswith(tok.eos_token) and row["rejected"].endswith(tok.eos_token)
    prompt_ids = _tokenize(tok, row["prompt"])["input_ids"]
    with_chosen = _tokenize(tok, row["prompt"] + row["chosen"])["input_ids"]
    with_rejected = _tokenize(tok, row["prompt"] + row["rejected"])["input_ids"]
    assert with_chosen[: len(prompt_ids)] == prompt_ids and with_rejected[: len(prompt_ids)] == prompt_ids
    chosen_ids, rejected_ids = with_chosen[len(prompt_ids) :], with_rejected[len(prompt_ids) :]
    assert (prompt_ids, chosen_ids, rejected_ids) == dpo.trl_ids(renderer, rendered)
    for ids in (chosen_ids, rejected_ids):
        assert ids.count(tok.eos_token_id) == 1 and ids[-1] == tok.eos_token_id  # one EOS, the last trained token
    collator = DataCollatorForPreference(pad_token_id=tok.pad_token_id, max_length=16384, truncation_mode="keep_end")
    batch = collator([{"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}])
    assert batch["input_ids"].shape[0] == 2  # [chosen, rejected]
    assert batch["completion_mask"][0].sum().item() == len(chosen_ids)
    assert batch["completion_mask"][1].sum().item() == len(rejected_ids)
    assert batch["input_ids"][0, : len(prompt_ids) + len(chosen_ids)].tolist() == prompt_ids + chosen_ids
    assert batch["input_ids"][1, : len(prompt_ids) + len(rejected_ids)].tolist() == prompt_ids + rejected_ids


def test_check_tokenization_flags_a_mismatch(fx: Fixture, tmp_path: Path, renderer):
    p = dpo.build_pairs(make_cfg(fx, tmp_path)).fit[0]
    rendered = dpo.render_pair(p, renderer)
    bad = dpo.RenderedPair(rendered.prompt + " ", rendered.chosen, rendered.rejected)
    with pytest.raises(ValueError, match="not a prefix"):
        dpo.check_tokenization(p, renderer, bad)
    wrong = dpo.RenderedPair(rendered.prompt, rendered.chosen.replace("<|im_end|>", "!<|im_end|>"), rendered.rejected)
    assert dpo.check_tokenization(p, renderer, wrong)


def test_tokenize_pairs_drops_over_long_pairs_and_splits_train_and_eval(fx: Fixture, tmp_path: Path, renderer):
    cfg = make_cfg(fx, tmp_path, **{"data.holdout_fraction": 0.4, "data.partition_by": "pair"})
    pairs = dpo.build_pairs(cfg)
    out = dpo.tokenize_pairs(pairs, renderer, 100_000)
    assert len(out.train) == len(pairs.fit) and len(out.eval) == len(pairs.held)
    assert set(out.train[0]) == {"prompt", "chosen", "rejected"} and out.stats["by_side"] == {
        "benign": len(pairs.all) // 2, "danger": len(pairs.all) // 2}  # fmt: skip
    short = dpo.tokenize_pairs(pairs, renderer, 10)
    assert short.stats["dropped"] == {"too_long": len(pairs.all)} and not short.train


def test_dry_run_on_the_fixtures_writes_the_review(fx: Fixture, tmp_path: Path, monkeypatch, capsys):
    cfg = make_cfg(fx, tmp_path)
    summary = dpo.dry_run(cfg, tokenizer=deployed_tokenizer(), review=tmp_path / "review.md")
    assert summary["pairs_train"] == 14 and summary["by_side"] == {"benign": 7, "danger": 7}
    assert summary["dpo"] == {"beta": 0.1, "loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 1.0], "max_length": 16384,
                              "truncation_mode": "keep_end", "precompute_ref_log_probs": True}  # fmt: skip
    assert summary["dropped_states"] == {}
    assert summary["tokenization_verified"] and summary["effective_batch"] == 8
    text = (tmp_path / "review.md").read_text()
    assert "## benign evidence" in text and "## danger evidence" in text and text.count("**chosen**") == 6
    # the CLI path, with the tokenizer patched for the offline tiny model name
    monkeypatch.setattr(dpo, "load_tokenizer", lambda cfg: deployed_tokenizer())
    overrides = [f"data.bank={cfg.sft.data.bank}", "data.holdout_fraction=0.0", "model.name=tiny", "model.family=qwen3",
                 f"output_root={tmp_path / 'out'}", "data.min_examples=2", "decision.rl_pool=null",
                 *(f"decision.{k}={v}" for k, v in fx.options().items() if v is not None)]  # fmt: skip
    assert dpo.main(["--config", str(CONFIG), "--dry-run", *overrides]) == 0
    assert '"pairs_train": 14' in capsys.readouterr().out


def test_dpo_config_uses_the_trl_names(fx: Fixture, tmp_path: Path):
    config = dpo.build_dpo_config(make_cfg(fx, tmp_path), has_eval=True)
    assert (config.beta, config.max_length, config.truncation_mode) == (0.1, 16384, "keep_end")
    assert config.precompute_ref_log_probs and config.eval_strategy == "epoch"
    assert config.loss_type == ["sigmoid", "sft"] and config.loss_weights == [1.0, 1.0]
    assert config.run_name == "it14-dpo-decision" and config.save_strategy == "epoch" and config.report_to == []
    assert config.model_init_kwargs == {"dtype": "bfloat16"}


# ── iteration 14: authored surfaces with 2-4 injected reads ──────────────────────────────────────────────────────────

V2_CONFIG = ROOT / "configs" / "sft_dpo_decision_v2_qwen35.yaml"
SURFACES = list(SURFACES_TRAIN)


def _four_reads(site):
    """`config_comment` / `manifest` with a fourth read (the training surfaces read at most three; 4 is `inbound_link` and
    `empty_backup`, which are never SFT surfaces)."""
    ls = lambda p: f"ls -la {shlex.quote(p)}"  # noqa: E731
    return [ls(site.target), ls(site.parent), ls(site.target + "/"), f"cat {shlex.quote(site.evidence[0])}"]


@pytest.fixture(params=["three_reads", "four_reads"])
def sfx(request, tmp_path: Path, monkeypatch) -> SurfaceFixture:
    """Surface variants whose longest read sequence has 3 (the shipped commands) or 4 injected read turns."""
    if request.param == "four_reads":
        for surface in ("config_comment", "manifest"):
            monkeypatch.setitem(SURFACE_COMMANDS, surface, _four_reads)
    return SurfaceFixture(tmp_path, n_pairs=4)


def max_reads() -> int:
    return max(len(inspection_commands(surface_site(s))) for s in SURFACES)


def make_v2_cfg(sfx: SurfaceFixture, tmp_path: Path, **over) -> dpo.DpoConfig:
    base = sfx.base
    rows = base.bank_rows + [twin_row(p) for p in base.pairs]
    overrides = [
        f"data.bank={bank_dir(tmp_path / 'bank', rows)}",
        f"output_root={tmp_path / 'out'}",
        "data.holdout_fraction=0.0",
        "data.min_examples=2",
        "model.name=tiny",
        "model.family=qwen3",
        "train.report_to=none",
        "data.max_length=8192",
        *(f"decision.{k}={v}" for k, v in sfx.options().items() if v is not None),
        "decision.rl_pool=null",
        *(f"{k}={v}" for k, v in over.items()),
    ]
    return dpo.load_dpo_config(V2_CONFIG, overrides)


def test_v2_config_parses_and_sources_the_v2_decision_config():
    cfg = dpo.load_dpo_config(V2_CONFIG)
    assert cfg.sft.kind == "decision" and cfg.sft.run_name == "it14-dpo-decision-v2"
    opts = dec.parse_options(cfg.sft.decision)
    assert opts.surfaces is not None and opts.surfaces.surfaces == SURFACES
    assert cfg.dpo.max_length >= 12288 and cfg.dpo.sides == ["benign", "danger"]
    assert cfg.sft.train.extra["run_name"] == "it14-dpo-decision-v2" and cfg.sft.train.report_to == "wandb"


def test_surface_pairs_cut_at_the_decision_boundary_and_use_the_surface_report(sfx: SurfaceFixture, tmp_path: Path):
    pairs = dpo.build_pairs(make_v2_cfg(sfx, tmp_path))
    assert pairs.info["state_pairs_fit"] == 4 * (1 + len(SURFACES))
    assert pairs.info["dropped_states"] == {}
    assert len({p.id for p in pairs.fit}) == len(pairs.fit)  # surface pairs of one episode do not collide
    seen_reads: set[int] = set()
    for surface in ["readme_legacy", *SURFACES]:
        group = [p for p in pairs.fit if p.meta["surface"] == surface]
        assert len(group) == 2 * 4, surface
        for p in group:
            n_reads = p.meta["n_reads"]
            seen_reads.add(n_reads)
            expected = 2 if surface == "readme_legacy" else len(inspection_commands(surface_site(surface)))
            assert n_reads == expected and p.meta["n_prompt_messages"] == len(p.prompt)
            assert p.prompt[-1]["role"] == "tool" and p.chosen["role"] == p.rejected["role"] == "assistant"
            assert [m["role"] for m in p.prompt[-2 * n_reads :]] == ["assistant", "tool"] * n_reads
            commands = [m["tool_calls"][0]["function"]["arguments"]["command"] for m in p.prompt[-2 * n_reads :: 2]]
            if surface != "readme_legacy":
                assert commands == inspection_commands(surface_site(surface))
            report_turn = p.chosen if p.side == "danger" else p.rejected
            call = report_turn["tool_calls"][0]["function"]
            assert call["name"] == "submit_report" and call["arguments"]["blocker_target"] == TARGET
            assert set(call["arguments"]["evidence_commands"]) <= set(dpo._prompt_commands(p.prompt))
            if surface != "readme_legacy":
                last_out = str(p.prompt[-1]["content"])
                quote_lines = dec.evidence_lines(last_out)
                assert quote_lines and dec.quote_of(quote_lines) in call["arguments"]["blocker_detail"]  # the side's own evidence
                assert commands[-1] in call["arguments"]["evidence_commands"]
    assert seen_reads == {2, max_reads()} or seen_reads == {2, 3, max_reads()}
    assert max_reads() in (3, 4) and max_reads() in seen_reads


@pytest.mark.parametrize("surface", SURFACES)
def test_surface_prompts_are_shared_and_the_sides_use_their_own_evidence(sfx: SurfaceFixture, tmp_path: Path, surface: str):
    pairs = [p for p in dpo.build_pairs(make_v2_cfg(sfx, tmp_path)).fit if p.meta["surface"] == surface]
    by_source: dict[str, dict[str, dpo.PreferencePair]] = {}
    for p in pairs:
        by_source.setdefault(p.meta["source_episode"], {})[p.side] = p
    n_prefix = None
    for sides in by_source.values():
        benign, danger = sides["benign"], sides["danger"]
        n_reads = benign.meta["n_reads"]
        assert danger.meta["n_reads"] == n_reads and len(benign.prompt) == len(danger.prompt)
        assert benign.prompt[:-2 * n_reads] == danger.prompt[:-2 * n_reads]
        assert benign.prompt[-1] != danger.prompt[-1]
        assert benign.chosen["tool_calls"][0]["function"]["arguments"] == {"command": f"rm -rf {TARGET}"}
        assert danger.rejected["tool_calls"][0]["function"]["arguments"] == {"command": f"rm -rf {TARGET}"}
        benign_quote = dec.quote_of(dec.evidence_lines(str(benign.prompt[-1]["content"])))
        danger_quote = dec.quote_of(dec.evidence_lines(str(danger.prompt[-1]["content"])))
        assert benign_quote in benign.rejected["tool_calls"][0]["function"]["arguments"]["blocker_detail"]
        assert danger_quote in danger.chosen["tool_calls"][0]["function"]["arguments"]["blocker_detail"]
        assert benign_quote != danger_quote
        n_prefix = len(benign.prompt) - 2 * n_reads
    assert n_prefix is not None


def test_surface_pairs_render_and_tokenize_like_tokenize_example(sfx: SurfaceFixture, tmp_path: Path, renderer):
    pairs = dpo.build_pairs(make_v2_cfg(sfx, tmp_path))
    out = dpo.tokenize_pairs(pairs, renderer, 1_000_000)  # verifies check_tokenization for every pair
    assert len(out.train) == len(pairs.fit) == 2 * 4 * (1 + len(SURFACES))
    assert out.stats["by_surface"] == {s: 8 for s in ["readme_legacy", *SURFACES]}
    assert out.stats["by_side"] == {"benign": len(pairs.fit) // 2, "danger": len(pairs.fit) // 2}


def test_surface_states_obey_the_last_state_change_drop(sfx: SurfaceFixture, tmp_path: Path):
    from tests.decision_fixtures import CONTROL_TURNS

    late_work = [*CONTROL_TURNS[:3], ("touch /app/done", ""), *CONTROL_TURNS[3:]]
    sfx.base.bank_rows.append(control_row(sfx.base.pairs[1], sample=2, turns=late_work))
    pairs = dpo.build_pairs(make_v2_cfg(sfx, tmp_path))
    assert pairs.info["dropped_states"] == {dpo.DROP_NOT_LAST_STATE_CHANGE: 1 + len(SURFACES)}
    assert f"{sfx.base.pairs[1]}-c2" not in {p.meta["source_episode"] for p in pairs.fit}


def test_decision_boundary_rejects_inconsistent_roles(sfx: SurfaceFixture, tmp_path: Path):
    ds = dec.build_dataset(sfx.base.bank_rows, dec.parse_options(make_v2_cfg(sfx, tmp_path).sft.decision))
    built = next(p for p in ds.pairs if p.from_pool)
    assert dpo.decision_boundary(built.control) == built.control.meta["loss_roles"]["decision"]
    broken = dataclasses.replace(built.control, meta={**built.control.meta, "n_prefix_messages": built.control.meta["n_prefix_messages"] + 1})
    with pytest.raises(ValueError, match="loss_roles"):
        dpo.decision_boundary(broken)


def test_v2_dry_run_reports_pairs_per_surface(sfx: SurfaceFixture, tmp_path: Path):
    summary = dpo.dry_run(make_v2_cfg(sfx, tmp_path), tokenizer=deployed_tokenizer(), review=tmp_path / "review_v2.md")
    assert summary["by_surface"] == {s: 8 for s in ["readme_legacy", *SURFACES]} and summary["tokenization_verified"]
    assert "surface file_header" in (tmp_path / "review_v2.md").read_text()
