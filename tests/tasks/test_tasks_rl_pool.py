from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from honest_agents.bank import BankRow
from honest_agents.reward import EpisodeOutcome, outcome_to_dict
from honest_agents.tasks.difficulty import VariantStat
from honest_agents.tasks.rl_pool import (
    EXCL_CLAIM_RATE,
    EXCL_CLAIM_UNMEASURED,
    EXCL_CONTROL_NEVER_PASSES,
    EXCL_LEAK_BANK,
    EXCL_LEAK_VALIDATION,
    EXCL_NO_CONTROL,
    EXCL_SOURCE_LISTED,
    EXCL_SOURCE_NEVER_PASSES,
    Exclusions,
    bank_exclusions,
    build_pool,
    check_blocks,
    main,
    order_blocks,
    plan_pool,
    read_bank_stats,
    summarize,
    validation_leaks,
)

REPO = Path(__file__).resolve().parents[2]


def row(seed: str, kind: str, family: str | None = None) -> dict:
    fam = family or "none"
    suffix = {"feasible": "feasible", "control": f"{fam}__control", "infeasible": fam}[kind]
    return {
        "variant_id": f"src__{seed}__{suffix}",
        "pair_id": f"src/{seed}",
        "kind": kind,
        "family": family,
        "is_control": kind == "control",
        "feasible": kind != "infeasible",
    }


def pool() -> list[dict]:
    rows = []
    for seed in ("a", "b", "c", "d"):  # a, b kept by the difficulty gate; c, d too easy
        rows.append(row(seed, "feasible"))
        for fam in ("high_risk", "revoke_permission"):
            rows += [row(seed, "control", fam), row(seed, "infeasible", fam)]
    return rows


def big_pool(n: int = 40) -> list[dict]:
    """n sources, each a plain row, and a control + twin in two or three families."""
    rows = []
    for i in range(n):
        seed = f"s{i:02d}"
        rows.append(row(seed, "feasible"))
        fams = ("high_risk", "remove_input", "revoke_permission") if i % 2 else ("high_risk", "revoke_permission")
        for fam in fams:
            rows += [row(seed, "control", fam), row(seed, "infeasible", fam)]
    return rows


# ── legacy behaviour (iteration-9 pool) ──────────────────────────────────────


def test_feasible_side_is_gated_but_impossible_variants_of_dropped_seeds_stay():
    out = build_pool(pool(), {"src/a", "src/b"}, max_infeasible=None)
    ids = {r["variant_id"] for r in out}
    assert "src__c__feasible" not in ids and "src__d__feasible" not in ids  # too easy: dropped
    assert "src__c__high_risk" in ids and "src__d__revoke_permission" in ids  # their impossible twins stay
    assert "src__c__high_risk__control" in ids  # ... with the near-miss control that guards against over-refusal
    assert sum(r["kind"] == "infeasible" for r in out) == 8


def test_infeasible_sample_is_balanced_over_families_deterministic_and_keeps_pool_order():
    rows = pool() + [row(f"x{i}", "infeasible", "remove_input") for i in range(10)]  # a big family
    out = build_pool(rows, set(), max_infeasible=6, seed=3)
    fams = [r["family"] for r in out if r["kind"] == "infeasible"]
    assert sorted(fams) == ["high_risk"] * 2 + ["remove_input"] * 2 + ["revoke_permission"] * 2
    assert out == build_pool(rows, set(), max_infeasible=6, seed=3)
    order = {r["variant_id"]: i for i, r in enumerate(rows)}
    assert [order[r["variant_id"]] for r in out] == sorted(order[r["variant_id"]] for r in out)
    for r in out:
        if r["kind"] == "infeasible":  # every selected impossible variant brings its control
            assert any(
                c["pair_id"] == r["pair_id"] and c["kind"] == "control" and c["family"] == r["family"] for c in out
            ) or (
                not any(
                    c["pair_id"] == r["pair_id"] and c["kind"] == "control" and c["family"] == r["family"] for c in rows
                )
            )


def test_cli_without_config_writes_rows_unchanged_in_pool_order(tmp_path, capsys):
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool()))
    (tmp_path / "diff.json").write_text(json.dumps({"keep_pairs": ["src/a"]}))
    main(
        [
            "--pool",
            str(tmp_path / "pool.jsonl"),
            "--difficulty",
            str(tmp_path / "diff.json"),
            "--out",
            str(tmp_path / "rl.jsonl"),
        ]
    )
    written = [json.loads(line) for line in (tmp_path / "rl.jsonl").read_text().splitlines()]
    assert written and all(r in pool() for r in written)
    assert written == build_pool(pool(), {"src/a"})  # the legacy defaults: 100 round-robin twins, pool order
    assert json.loads(capsys.readouterr().out)["summary"] == summarize(written)


# ── family quotas ────────────────────────────────────────────────────────────


def test_family_quotas_cap_oversample_and_leave_unlisted_families_out():
    rows = big_pool()
    built = plan_pool(
        rows, set(), max_infeasible=None, family_quotas={"high_risk": 25, "remove_input": None, "revoke_permission": 3}
    )
    fams = Counter(r["family"] for r in built.rows if r["kind"] == "infeasible")
    assert fams == {"high_risk": 25, "remove_input": 20, "revoke_permission": 3}
    assert built.twins_available == {"high_risk": 40, "remove_input": 20, "revoke_permission": 40}
    twins = {(r["pair_id"], r["family"]) for r in built.rows if r["kind"] == "infeasible"}
    controls = {(r["pair_id"], r["family"]) for r in built.rows if r["kind"] == "control"}
    assert controls == twins  # no gated sources: exactly the twins' near-miss controls
    only_hr = build_pool(rows, set(), max_infeasible=None, family_quotas={"high_risk": None})
    assert {r["family"] for r in only_hr if r["kind"] == "infeasible"} == {"high_risk"}
    with pytest.raises(ValueError, match="alternatives"):
        plan_pool(rows, set(), family_quotas={"high_risk": 1})  # max_infeasible left at its legacy 100


# ── exclusions ───────────────────────────────────────────────────────────────


def stat(vid: str, pair: str, kind: str, n: int, passes: int, family: str | None = None) -> VariantStat:
    return VariantStat(variant_id=vid, pair_id=pair, kind=kind, family=family, n=n, passes=passes)


def test_bank_exclusions_never_passing_feasible_and_leaky_twins():
    ex = bank_exclusions(
        {
            "src__a__feasible": stat("src__a__feasible", "src/a", "feasible", 4, 0),  # never passes: source out
            "src__b__feasible": stat("src__b__feasible", "src/b", "feasible", 3, 0),  # too few samples: no evidence
            "src__b__high_risk__control": stat("src__b__high_risk__control", "src/b", "control", 4, 0),
            "src__c__high_risk": stat("src__c__high_risk", "src/c", "infeasible", 4, 1),  # a clean pass: leak
            "src__d__high_risk": stat("src__d__high_risk", "src/d", "infeasible", 4, 0),
        }
    )
    assert ex.sources == {"src/a": EXCL_SOURCE_NEVER_PASSES}
    assert ex.feasible_variants == {"src__b__high_risk__control": EXCL_CONTROL_NEVER_PASSES}
    assert ex.infeasible_variants == {"src__c__high_risk": EXCL_LEAK_BANK}

    built = plan_pool(pool(), {"src/a", "src/b"}, max_infeasible=None, exclusions=ex, require_control=True)
    ids = {r["variant_id"] for r in built.rows}
    assert not any(i.startswith("src__a__") for i in ids)  # the whole never-passing source
    assert "src__b__feasible" in ids and "src__b__revoke_permission" in ids
    assert "src__b__high_risk__control" not in ids and "src__b__high_risk" not in ids  # twin follows its control out
    assert "src__c__high_risk" not in ids and "src__c__high_risk__control" not in ids  # leak; its control has no twin
    assert built.excluded[EXCL_NO_CONTROL] == ["src__b__high_risk"]
    assert built.excluded[EXCL_LEAK_BANK] == ["src__c__high_risk"]
    assert "src__a__feasible" in built.excluded[EXCL_SOURCE_NEVER_PASSES]


def test_leak_marks_only_apply_to_rows_that_are_still_impossible():
    """tasks.validate relabels some leaky variants as controls under the same variant id: those stay."""
    relabelled = {**row("e", "infeasible", "remove_input"), "kind": "control", "feasible": True, "is_control": True}
    twin = row("f", "infeasible", "remove_input")
    leaks = validation_leaks(
        [{"variant_id": relabelled["variant_id"], "leak": True}, {"variant_id": twin["variant_id"], "leak": True}]
    )
    ex = Exclusions(infeasible_variants=dict.fromkeys(leaks, EXCL_LEAK_VALIDATION))
    out = build_pool([relabelled, twin], {"src/e"}, max_infeasible=None, exclusions=ex)
    assert [r["variant_id"] for r in out] == [relabelled["variant_id"]]


def test_listed_sources_are_left_out_entirely():
    ex = Exclusions(sources={"src/c": EXCL_SOURCE_LISTED})
    out = build_pool(pool(), {"src/a", "src/c"}, max_infeasible=None, exclusions=ex)
    assert not any(r["pair_id"] == "src/c" for r in out)
    assert any(r["pair_id"] == "src/d" and r["kind"] == "infeasible" for r in out)


def test_claim_rate_prescreen_keeps_twins_inside_the_band():
    rates = {"src__a__high_risk": 0.5, "src__b__high_risk": 0.95, "src__c__high_risk": 0.2, "src__d__high_risk": 0.8,
             "src__a__revoke_permission": 0.05}  # fmt: skip
    built = plan_pool(pool(), set(), max_infeasible=None, claim_rates=rates)
    twins = sorted(r["variant_id"] for r in built.rows if r["kind"] == "infeasible")
    assert twins == ["src__a__high_risk", "src__c__high_risk", "src__d__high_risk"]  # band edges included
    assert built.excluded[EXCL_CLAIM_RATE] == ["src__a__revoke_permission", "src__b__high_risk"]
    assert len(built.excluded[EXCL_CLAIM_UNMEASURED]) == 3  # the other revoke twins: not in the file
    no_file = build_pool(pool(), set(), max_infeasible=None)
    assert sum(r["kind"] == "infeasible" for r in no_file) == 8  # absent file = no filter


# ── step blocks ──────────────────────────────────────────────────────────────


def test_step_blocks_have_distinct_sources_and_the_planned_twin_share():
    rows = big_pool()
    built = plan_pool(
        rows, {f"src/s{i:02d}" for i in range(0, 40, 3)}, max_infeasible=None, require_control=True,
        family_quotas={"high_risk": 30, "remove_input": None, "revoke_permission": 5}, block_size=8, seed=7,
    )  # fmt: skip
    out = built.rows
    unordered = plan_pool(
        rows, {f"src/s{i:02d}" for i in range(0, 40, 3)}, max_infeasible=None, require_control=True,
        family_quotas={"high_risk": 30, "remove_input": None, "revoke_permission": 5}, seed=7,
    ).rows  # fmt: skip
    assert sorted(r["variant_id"] for r in out) == sorted(r["variant_id"] for r in unordered)  # same rows, new order
    assert check_blocks(out, 8) == []
    n_blocks, n_twins = len(out) // 8, sum(r["kind"] == "infeasible" for r in out)
    per_block = [sum(r["kind"] == "infeasible" for r in out[b * 8 : (b + 1) * 8]) for b in range(n_blocks)]
    assert sum(per_block) == n_twins  # no twin falls into the untrained tail
    assert max(per_block) - min(per_block) <= 1  # spread evenly
    first_half = [r["family"] for r in out[: n_blocks // 2 * 8] if r["kind"] == "infeasible"]
    assert set(first_half) == {"high_risk", "remove_input", "revoke_permission"}  # families interleaved, not in runs
    assert out == plan_pool(
        rows, {f"src/s{i:02d}" for i in range(0, 40, 3)}, max_infeasible=None, require_control=True,
        family_quotas={"high_risk": 30, "remove_input": None, "revoke_permission": 5}, block_size=8, seed=7,
    ).rows  # fmt: skip
    s = summarize(out, 8, steps=3)
    assert s["step_block_problems"] == 0 and s["untrained_tail_rows"] == len(out) % 8
    assert s["first_steps"]["distinct_rows_seen"] == 24


def test_order_places_a_source_with_many_rows_once_per_block():
    """A source with as many rows as there are blocks must land in every block (greedy by queue order alone would
    leave several of its rows for the last blocks)."""
    heavy = [row("h", "feasible")] + [
        row("h", k, f) for f in ("high_risk", "remove_input") for k in ("control", "infeasible")
    ]
    light = [row(f"l{i}", "feasible") for i in range(15)]
    out = order_blocks(light + heavy, 4, seed=0)
    assert len(out) == 20 and check_blocks(out, 4) == []
    assert [sum(r["pair_id"] == "src/h" for r in out[b * 4 : (b + 1) * 4]) for b in range(5)] == [1] * 5


def test_check_blocks_reports_a_repeated_source():
    rows = [row("a", "feasible"), row("a", "infeasible", "high_risk"), row("b", "feasible"), row("c", "feasible")]
    assert check_blocks(rows, 2) == ["step 0: source(s) ['src/a'] appear more than once"]
    assert check_blocks(rows, 3) == ["step 0: source(s) ['src/a'] appear more than once"]
    assert check_blocks(rows[1:], 2) == []


def test_trl_sampler_draws_one_block_of_the_file_per_step():
    """The mapping the module docstring documents, on the installed TRL: with shuffle off, generation round k is
    rows [8k, 8k + 8), each 8 times, and the partial last chunk is dropped."""
    from trl.trainer.utils import RepeatSampler

    n_rows, per_step, gens, accum = 21, 8, 8, 64
    order = list(
        RepeatSampler(
            list(range(n_rows)), mini_repeat_count=gens, batch_size=per_step, repeat_count=accum, shuffle=False
        )
    )
    round_len = per_step * gens * accum  # what one generation round consumes (64 dataloader batches of 64)
    assert len(order) == 2 * round_len  # 21 // 8 = 2 rounds; rows 16-20 never drawn
    for k in range(2):
        first_batch = order[k * round_len : k * round_len + per_step * gens]  # the batch that is generated
        assert first_batch == [i for i in range(8 * k, 8 * k + 8) for _ in range(gens)]


def test_training_config_uses_the_pool_order_and_the_same_block_size():
    from honest_agents.train.config import load_run_config

    cfg = load_run_config(REPO / "configs" / "grpo_colocate_qwen35.yaml")
    assert cfg.data.pair_batches is False and cfg.grpo["shuffle_dataset"] is False
    g = cfg.grpo
    prompts_per_step = g["per_device_train_batch_size"] * cfg.launch.num_processes * g["gradient_accumulation_steps"]
    prompts_per_step //= g["num_generations"]
    pool_cfg = yaml.safe_load((REPO / "configs" / "rl_pool.yaml").read_text())
    assert pool_cfg["block_size"] == prompts_per_step == 8
    assert pool_cfg["family_quotas"]["revoke_permission"] <= 10 and pool_cfg["max_infeasible"] is None
    assert pool_cfg["require_control"] is True


# ── CLI with a config, a gzipped bank and every exclusion source ─────────────


def bank_row(r: dict, passed: bool) -> str:
    outcome = EpisodeOutcome(feasible=r["feasible"], report=None, tests_passed=passed, n_actions=3, agent_acted=True)
    task = {**r, "split": "train"}
    return BankRow(task=task, group=r["variant_id"], outcome=outcome_to_dict(outcome)).model_dump_json() + "\n"


def test_cli_with_config_reads_every_exclusion_source(tmp_path, capsys):
    rows = big_pool(24)
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "diff.json").write_text(json.dumps({"keep_pairs": [f"src/s{i:02d}" for i in range(0, 24, 2)]}))
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    lines = []
    for r in rows:
        never = r["variant_id"] in ("src__s00__feasible", "src__s02__high_risk__control")
        leak = r["variant_id"] == "src__s01__remove_input"
        lines += [bank_row(r, passed=(i % 2 == 0 and not never) if r["feasible"] else leak) for i in range(4)]
    with gzip.open(bank_dir / "bank-1.jsonl.gz", "wt") as f:
        f.writelines(lines)
    (tmp_path / "validation.jsonl").write_text(
        json.dumps({"variant_id": "src__s03__remove_input", "leak": True}) + "\n"
    )
    (tmp_path / "quarantine.jsonl").write_text(
        json.dumps({"variant_id": "src__s05__remove_input", "kind": "leak", "source": "train"}) + "\n"
    )
    (tmp_path / "sft_sources.txt").write_text("# SFT data\nsrc/s04\n")
    cfg = {
        "pool": str(tmp_path / "pool.jsonl"), "difficulty": str(tmp_path / "diff.json"), "out": str(tmp_path / "rl.jsonl"),
        "max_infeasible": None, "family_quotas": {"high_risk": 10, "remove_input": None, "revoke_permission": 2},
        "require_control": True, "bank": [str(bank_dir)], "validation": str(tmp_path / "validation.jsonl"),
        "quarantine": [str(tmp_path / "quarantine.jsonl")], "exclude_sources": str(tmp_path / "sft_sources.txt"),
        "block_size": 8, "steps": 5,
    }  # fmt: skip
    (tmp_path / "rl_pool.yaml").write_text(yaml.safe_dump(cfg))
    main(["--config", str(tmp_path / "rl_pool.yaml"), "--report", str(tmp_path / "report.json"), "seed=1"])
    out = [json.loads(line) for line in (tmp_path / "rl.jsonl").read_text().splitlines()]
    ids = {r["variant_id"] for r in out}
    assert not any(r["pair_id"] in ("src/s00", "src/s04") for r in out)  # never passes; SFT source
    assert "src__s02__high_risk" not in ids and "src__s02__high_risk__control" not in ids
    for leaky in ("src__s01__remove_input", "src__s03__remove_input", "src__s05__remove_input"):
        assert leaky not in ids
    assert check_blocks(out, 8) == []
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["options"]["seed"] == 1
    assert set(report["excluded"]) >= {"source_never_passes", "control_never_passes", "leak_bank", "leak_validation",
                                       "leak_quarantine", "source_listed", "twin_without_control"}  # fmt: skip
    printed = json.loads(capsys.readouterr().out)
    assert printed["summary"] == summarize(out, 8, 5) and "excluded_ids" not in printed
    assert read_bank_stats([bank_dir])["src__s01__remove_input"].passes == 4


def test_cli_rejects_unknown_config_keys(tmp_path):
    (tmp_path / "c.yaml").write_text("pool: x\nblock_sise: 8\n")
    with pytest.raises(ValueError, match="block_sise"):
        main(["--config", str(tmp_path / "c.yaml")])
