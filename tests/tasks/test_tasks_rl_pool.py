from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from honest_agents.bank import BankRow
from honest_agents.report import parse_report
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
    PoolOptions,
    bank_exclusions,
    build_pool,
    check_blocks,
    check_paired_blocks,
    main,
    order_blocks,
    order_paired_blocks,
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


def iteration9_sample(rows: list[dict], max_infeasible: int, seed: int) -> set[str]:
    """The iteration-9 builder's twin sample (git 361de8f): families shuffled in first-appearance order, one rng."""
    import random

    rng = random.Random(seed)
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        if r["kind"] == "infeasible":
            by_family.setdefault(r["family"], []).append(r)
    for family_rows in by_family.values():
        rng.shuffle(family_rows)
    chosen: list[dict] = []
    queues = [by_family[f] for f in sorted(by_family)]
    while queues and len(chosen) < max_infeasible:
        for q in list(queues):
            if len(chosen) >= max_infeasible:
                break
            chosen.append(q.pop())
            if not q:
                queues.remove(q)
    return {r["variant_id"] for r in chosen}


def test_legacy_defaults_draw_the_iteration9_sample_when_families_are_not_in_sorted_order():
    """data/train_rl.jsonl of iteration 9 must be rebuildable: the pool lists revoke_permission before remove_input, so
    shuffling families in sorted order would draw a different sample from the same seed."""
    rows = []
    for i in range(12):
        for fam in ("revoke_permission", "high_risk", "remove_input"):
            rows += [row(f"s{i:02d}", "control", fam), row(f"s{i:02d}", "infeasible", fam)]
    for seed in range(5):
        twins = {
            r["variant_id"] for r in build_pool(rows, set(), max_infeasible=7, seed=seed) if r["kind"] == "infeasible"
        }
        assert twins == iteration9_sample(rows, 7, seed)


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
    pool_cfg = yaml.safe_load((REPO / "configs" / "tasks" / "rl_pool.yaml").read_text())
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


def test_cli_refuses_a_missing_quarantine_file_or_an_empty_bank(tmp_path, capsys):
    """Both would silently drop exclusions: the quarantine reader treats a missing list as empty, and an empty bank
    directory yields no statistics."""
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool()))
    (tmp_path / "diff.json").write_text(json.dumps({"keep_pairs": ["src/a"]}))
    (tmp_path / "bank").mkdir()
    base = ["--pool", str(tmp_path / "pool.jsonl"), "--difficulty", str(tmp_path / "diff.json"),
            "--out", str(tmp_path / "rl.jsonl")]  # fmt: skip
    for extra, message in (
        (["--quarantine", str(tmp_path / "nope.jsonl")], "does not exist"),
        (["--bank", str(tmp_path / "bank")], "no train-split rows"),
    ):
        with pytest.raises(SystemExit):
            main(base + extra)
        assert message in capsys.readouterr().err
    assert not (tmp_path / "rl.jsonl").exists()


def test_cli_rejects_unknown_config_keys(tmp_path):
    (tmp_path / "c.yaml").write_text("pool: x\nblock_sise: 8\n")
    with pytest.raises(ValueError, match="block_sise"):
        main(["--config", str(tmp_path / "c.yaml")])


# ── paired strata (iteration 13 pool v2) ─────────────────────────────────────


def strata_pool(n: int = 40) -> list[dict]:
    """n sources: a plain row and a high_risk control + twin each; revoke_permission on every third source,
    remove_input on every fourth (offset 1)."""
    rows = []
    for i in range(n):
        seed = f"s{i:02d}"
        rows.append(row(seed, "feasible"))
        fams = ["high_risk"] + (["revoke_permission"] if i % 3 == 0 else []) + (["remove_input"] if i % 4 == 1 else [])
        for fam in fams:
            rows += [row(seed, "control", fam), row(seed, "infeasible", fam)]
    return rows


def paired_plan(seed: int = 42, **kw) -> list[dict]:
    rows = strata_pool()
    kw.setdefault("family_quotas", {"high_risk": 30, "revoke_permission": None, "remove_input": None})
    return plan_pool(
        rows, {f"src/s{i:02d}" for i in range(40)}, max_infeasible=None, require_control=True, block_size=8,
        seed=seed, order="paired_strata", **kw,
    ).rows  # fmt: skip


def blocks_of(rows: list[dict], size: int = 8) -> list[list[dict]]:
    return [rows[b * size : (b + 1) * size] for b in range(len(rows) // size)]


def test_paired_strata_every_block_holds_a_high_risk_pair_and_another_pair():
    out = paired_plan()
    unordered = paired_plan(family_quotas={"high_risk": 30, "revoke_permission": None, "remove_input": None})
    assert sorted(r["variant_id"] for r in out) == sorted(r["variant_id"] for r in unordered)
    assert len(out) == 60 + 48 + 40 + 10  # hr 30 + revoke 14 + remove 10 pairs, 40 plain, 10 lone hr controls
    assert check_paired_blocks(out, 8) == []  # no split pair, no repeated source, no stratum shortfall
    for block in blocks_of(out):
        pairs = [
            t for t in block if t["kind"] == "infeasible"
            and any(c["kind"] == "control" and (c["pair_id"], c["family"]) == (t["pair_id"], t["family"]) for c in block)
        ]  # fmt: skip
        assert any(t["family"] == "high_risk" for t in pairs)
        assert any(t["family"] in ("revoke_permission", "remove_input") for t in pairs)
        twins = [t for t in block if t["kind"] == "infeasible"]
        assert pairs == twins  # every twin of the block has its control in the same block
        # one unit per source: the plain copy never shares a step with its own twin
        plain_sources = {r["pair_id"] for r in block if r["kind"] == "feasible"}
        assert not plain_sources & {t["pair_id"] for t in twins}
    s = summarize(out, 8, steps=5, order="paired_strata")
    assert s["step_block_problems"] == 0 and s["untrained_tail_rows"] == len(out) % 8
    assert s["blocks_without_plain"] == [] and s["strata_shortfalls"] == []  # a plain capability anchor in every step
    twins_in_blocks = sum(r["kind"] == "infeasible" for b in blocks_of(out) for r in b)
    assert twins_in_blocks == 54  # no twin falls into the untrained tail
    hr_per_block = [sum(r["kind"] == "infeasible" and r["family"] == "high_risk" for r in b) for b in blocks_of(out)]
    assert max(hr_per_block) - min(hr_per_block) <= 2  # pairs spread over the blocks, not in runs


def test_paired_strata_order_is_deterministic_given_the_seed():
    assert paired_plan(seed=42) == paired_plan(seed=42)
    rows = paired_plan(seed=42)[::-1]  # one fixed pool, ordered with two seeds
    first, again, other = (order_paired_blocks(rows, 8, seed=s) for s in (42, 42, 7))
    assert first == again
    assert first != other and sorted(map(str, first)) == sorted(map(str, other))  # same rows, another order
    assert check_paired_blocks(first, 8) == check_paired_blocks(other, 8) == []


def test_paired_strata_reports_a_stratum_it_cannot_fill():
    """Fewer non-high_risk pairs than blocks: every block still keeps its pairs together; the blocks without a
    revoke / remove_input pair are reported as soft problems, not hidden."""
    out = paired_plan(family_quotas={"high_risk": 30, "revoke_permission": 3, "remove_input": 0})
    problems = check_paired_blocks(out, 8)
    assert problems and all(p.startswith("no_minor") for p in problems)
    assert sum(1 for b in blocks_of(out) if any(r["family"] == "revoke_permission" and r["kind"] == "infeasible"
                                                for r in b)) == 3  # fmt: skip


def test_check_paired_blocks_flags_a_split_pair_and_a_repeated_source():
    a_twin, a_ctrl = row("a", "infeasible", "high_risk"), row("a", "control", "high_risk")
    b_twin, b_ctrl = row("b", "infeasible", "revoke_permission"), row("b", "control", "revoke_permission")
    ok = [a_twin, a_ctrl, b_twin, b_ctrl]
    assert check_paired_blocks(ok, 4) == []
    split = check_paired_blocks([a_twin, b_twin, b_ctrl, row("c", "feasible"), a_ctrl, row("d", "feasible")], 4)
    assert any(p.startswith("split step 0") for p in split)
    repeat = check_paired_blocks([a_twin, a_ctrl, row("a", "feasible"), b_twin, b_ctrl, row("x", "feasible")], 6)
    assert any(p.startswith("repeat step 0") for p in repeat)


def test_order_paired_blocks_keeps_pairs_together_in_the_tail():
    rows = [
        r for i in range(5) for r in (row(f"h{i}", "infeasible", "high_risk"), row(f"h{i}", "control", "high_risk"))
    ]
    out = order_paired_blocks(rows, 4, seed=0)  # 10 rows, two pairs per block, 2 tail rows = one whole pair
    assert len(out) == 10
    tail = out[8:]
    assert {r["pair_id"] for r in tail} == {tail[0]["pair_id"]} and {r["kind"] for r in tail} == {
        "infeasible",
        "control",
    }


def test_prefer_twins_fill_the_quota_first_without_changing_the_rest_of_the_sample():
    rows = big_pool()
    remove_ids = [r["variant_id"] for r in rows if r["kind"] == "infeasible" and r["family"] == "remove_input"]
    preferred = set(remove_ids[:3])
    quotas = {"high_risk": 10, "remove_input": 5, "revoke_permission": 2}
    built = plan_pool(rows, set(), max_infeasible=None, family_quotas=quotas, prefer_twins=preferred, seed=3)
    chosen = {r["variant_id"] for r in built.rows if r["kind"] == "infeasible" and r["family"] == "remove_input"}
    assert preferred <= chosen and len(chosen) == 5
    assert built.preferred_chosen == {"remove_input": 3}
    plain = plan_pool(rows, set(), max_infeasible=None, family_quotas=quotas, seed=3)
    others = {r["variant_id"] for r in built.rows if r["family"] != "remove_input"}
    assert others == {r["variant_id"] for r in plain.rows if r["family"] != "remove_input"}  # other families: same draw


def claim_bank_row(r: dict, claimed: bool) -> str:
    report = parse_report({"status": "complete" if claimed else "abstain", "blocker_category": "high_risk",
                           "blocker_target": "/data"})  # fmt: skip
    outcome = EpisodeOutcome(feasible=r["feasible"], report=report, tests_passed=False, n_actions=3, agent_acted=True)
    return (
        BankRow(task={**r, "split": "train"}, group=r["variant_id"], outcome=outcome_to_dict(outcome)).model_dump_json()
        + "\n"
    )


def test_cli_paired_strata_with_claimed_first(tmp_path, capsys):
    rows = strata_pool()
    (tmp_path / "pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "diff.json").write_text(json.dumps({"keep_pairs": [f"src/s{i:02d}" for i in range(40)]}))
    remove = [r for r in rows if r["kind"] == "infeasible" and r["family"] == "remove_input"]
    claimers = {r["variant_id"] for r in remove[:2]}
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    lines = [claim_bank_row(r, claimed=r["variant_id"] in claimers and i == 0) for r in remove for i in range(4)]
    with gzip.open(bank_dir / "bank-1.jsonl.gz", "wt") as f:
        f.writelines(lines)
    (tmp_path / "sft_pool.jsonl").write_text(json.dumps(row("s39", "feasible")) + "\n")  # an SFT source
    cfg = {
        "pool": str(tmp_path / "pool.jsonl"), "difficulty": str(tmp_path / "diff.json"), "out": str(tmp_path / "rl.jsonl"),
        "max_infeasible": None, "family_quotas": {"high_risk": 30, "revoke_permission": None, "remove_input": 4},
        "require_control": True, "bank": [str(bank_dir)], "claimed_first": ["remove_input"],
        "exclude_pools": [str(tmp_path / "sft_pool.jsonl")], "block_size": 8, "order": "paired_strata", "steps": 5,
        "seed": 42,
    }  # fmt: skip
    (tmp_path / "rl_pool.yaml").write_text(yaml.safe_dump(cfg))
    main(["--config", str(tmp_path / "rl_pool.yaml"), "--report", str(tmp_path / "report.json")])
    out = [json.loads(line) for line in (tmp_path / "rl.jsonl").read_text().splitlines()]
    assert not any(r["pair_id"] == "src/s39" for r in out)
    chosen_remove = {r["variant_id"] for r in out if r["kind"] == "infeasible" and r["family"] == "remove_input"}
    assert claimers <= chosen_remove and len(chosen_remove) == 4
    assert [p for p in check_paired_blocks(out, 8) if p.startswith(("split", "repeat"))] == []
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["claimed_first"] == {"claimed_in_bank": 2, "chosen": {"remove_input": 2}}
    assert len(report["blocks"]) == len(out) // 8 and report["summary"]["windows"]
    captured = capsys.readouterr()
    assert json.loads(captured.out)["summary"] == summarize(out, 8, 5, "paired_strata")
    assert "step   0:" in captured.err  # per-step composition on stderr
    assert read_bank_stats([bank_dir])[remove[0]["variant_id"]].claims == 1


def test_pool_v2_config_matches_the_it13_training_config():
    from honest_agents.train.config import load_run_config

    cfg = load_run_config(REPO / "configs" / "grpo_it13.yaml")
    pool_cfg = yaml.safe_load((REPO / "configs" / "tasks" / "rl_pool_v2.yaml").read_text())
    g = cfg.grpo
    prompts_per_step = g["per_device_train_batch_size"] * cfg.launch.num_processes * g["gradient_accumulation_steps"]
    assert pool_cfg["block_size"] == prompts_per_step // g["num_generations"] == 8
    assert cfg.data.train_file == pool_cfg["out"] == "data/train_rl_v2.jsonl"
    assert cfg.data.pair_batches is False and g["shuffle_dataset"] is False
    assert pool_cfg["order"] == "paired_strata" and pool_cfg["require_control"] is True
    assert pool_cfg["family_quotas"] == {"high_risk": 80, "revoke_permission": 40, "remove_input": 25}
    assert g["max_steps"] == 15 and pool_cfg["steps"] == 15
    PoolOptions.from_dict(pool_cfg)  # every key is known


def test_paired_strata_puts_all_pair_blocks_last_and_anchors_the_rest_with_a_plain_row():
    """More pairs than blocks of 3 pairs can hold: the all-pair blocks (no plain row) go to the end of the pass, and
    every other block holds a plain row even when lone controls are first in the singles queue."""
    rows = [
        r for i in range(14) for r in (row(f"h{i}", "infeasible", "high_risk"), row(f"h{i}", "control", "high_risk"))
    ]
    rows += [r for i in range(6) for r in (row(f"r{i}", "infeasible", "revoke_permission"),
                                            row(f"r{i}", "control", "revoke_permission"))]  # fmt: skip
    rows += [row(f"c{i}", "control", "high_risk") for i in range(4)] + [row(f"p{i}", "feasible") for i in range(4)]
    out = order_paired_blocks(rows, 8, seed=1)  # 48 rows: 6 blocks, 20 pairs = 4 blocks of 3 + 2 blocks of 4
    s = summarize(out, 8, order="paired_strata")
    assert s["step_block_problems"] == 0 and s["blocks_without_plain"] == [4, 5]
