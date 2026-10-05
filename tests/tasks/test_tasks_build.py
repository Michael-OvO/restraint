"""Dataset build: columns, splits by absolute seed count, held-out families, pools per split, mixture, controls,
quarantine, relabelling, difficulty gate and the hermetic filter (SPEC_V3 §4)."""

from __future__ import annotations

import json
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pytest

from honest_agents.bank import (
    QUARANTINE_UNCLAIMED_PASS,
    QuarantineEntry,
    append_quarantine,
    read_quarantine,
)
from honest_agents.harness import SYSTEM_PROMPT
from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks import perturb
from honest_agents.tasks.build import (
    COLUMNS,
    DEFAULT_HELDOUT_FAMILIES,
    DEFAULT_MIXTURE,
    Row,
    assign_seed_splits,
    balance_mixture,
    build,
    check_no_test_leak,
    enforce_pairing,
    filter_hermetic,
    heldout_problems,
    ids_digest,
    main,
    normalise_mixture,
    parse_args,
    phrasing_split,
    read_test_ids,
    split_bucket,
    write_test_ids,
)
from honest_agents.tasks.perturb import OPERATORS
from honest_agents.tasks.sources import SeedTask
from honest_agents.tasks.validate import ValidationResult
from tests.tasks.synthetic import plain_task, rich_task, write_task

N_RICH = 14


@pytest.fixture
def seeds(tmp_path) -> Path:
    root = tmp_path / "seeds"
    for i in range(N_RICH):
        rich_task(root / f"rich-{i:02d}", verifier="pytest")
    plain_task(root / "plain")
    write_task(  # no oracle: feasible-only, needs the allowlist
        root / "nosol",
        instruction="x",
        dockerfile="FROM ubuntu\n",
        solve=None,
        test_sh="#!/bin/bash\n",
    )
    return root


def _args(out: Path, seeds: Path, *extra: str) -> list[str]:
    return [
        "--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier", "--i-know-no-benchmark", "--no-enforce-cue-band", "--neardup-threshold", "0",
        "--test-seeds", "5", "--val-seeds", "3", *extra,
    ]  # fmt: skip


def _rows(out: Path) -> dict[str, list[dict]]:
    return {s: [json.loads(x) for x in (out / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "val", "test")}


def _honest_variant(row: dict) -> dict:
    return tomllib.loads((Path(row["task_dir"]) / "task.toml").read_text())["metadata"].get("honest_variant", {})


# ── end to end ───────────────────────────────────────────────────────────────


def test_build_end_to_end(tmp_path, seeds):
    out = tmp_path / "out"
    manifest = build(parse_args(_args(out, seeds)))
    rows = _rows(out)
    assert all(rows.values())
    pair_split: dict[str, set[str]] = defaultdict(set)
    for split, rs in rows.items():
        for r in rs:
            assert set(r) == set(COLUMNS)  # exactly the INTERFACES_V3 §6 columns
            assert r["split"] == split
            assert r["prompt"] == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}]
            spec = HonestTaskSpec.from_task_dir(r["task_dir"])
            assert (spec.feasible, spec.is_control, spec.perturbation, spec.pair_id) == (
                r["feasible"], r["is_control"], r["perturbation"], r["pair_id"],
            )  # fmt: skip
            assert (
                r["kind"]
                == spec.kind
                == ("infeasible" if not r["feasible"] else "control" if r["is_control"] else "feasible")
            )
            assert r["family"] == (r["perturbation"] or "feasible")
            assert Path(r["task_dir"]).name == r["variant_id"] == spec.variant_id
            assert spec.verifier == ("legacy" if r["pair_id"] == "local/plain" else "pytest")  # carried from the seed
            pair_split[r["pair_id"]].add(split)
            if r["family"] in DEFAULT_HELDOUT_FAMILIES:
                assert split == "test"  # held-out families, controls included, are only ever seen at test time
    assert all(len(s) == 1 for s in pair_split.values())  # no pair crosses splits
    assert "local/nosol" not in pair_split  # solution-less seed skipped without allowlist
    train_families = {r["family"] for r in rows["train"]}
    allowed = {"feasible", *OPERATORS} - set(DEFAULT_HELDOUT_FAMILIES)
    assert {"feasible"} < train_families <= allowed  # the mixture samples families; held-out ones never appear
    test_families = {r["family"] for r in rows["test"]}
    assert set(DEFAULT_HELDOUT_FAMILIES) <= test_families  # one environment mechanism + high_risk
    assert manifest["n_skipped_no_solution"] == 1
    assert manifest["seed_splits"]["test"] == 5 and manifest["seed_splits"]["val"] == 3  # absolute counts
    assert len({r["pair_id"] for r in rows["test"]}) == 5
    assert len({r["pair_id"] for r in rows["val"]}) <= 3
    # test keeps every variant of its seeds: plain + (variant, control) per operator
    per_pair = Counter(r["pair_id"] for r in rows["test"])
    assert all(n == (2 if p == "local/plain" else 1 + 2 * len(OPERATORS)) for p, n in per_pair.items())


def test_test_split_size_is_an_absolute_count_not_a_fraction(tmp_path, seeds):
    for n_test in (2, 7):
        out = tmp_path / f"out{n_test}"
        build(parse_args(["--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier", "--i-know-no-benchmark", "--no-enforce-cue-band", "--neardup-threshold", "0",
                          "--test-seeds", str(n_test), "--val-seeds", "0"]))  # fmt: skip
        assert len({r["pair_id"] for r in _rows(out)["test"]}) == n_test


def test_mixture_default_is_55_20_25(tmp_path, seeds):
    assert DEFAULT_MIXTURE == (0.55, 0.20, 0.25)
    out = tmp_path / "out"
    build(parse_args(_args(out, seeds)))
    train = _rows(out)["train"]
    kinds = Counter(r["kind"] for r in train)
    n = len(train)
    for kind, share in zip(("feasible", "control", "infeasible"), DEFAULT_MIXTURE, strict=True):
        assert abs(kinds[kind] / n - share) < 0.06, (kinds, share)
    # paired draw (default): inside every family the infeasible and control rows are the same sources, same count
    per = {k: Counter((r["family"], r["pair_id"]) for r in train if r["kind"] == k) for k in ("infeasible", "control")}
    assert per["infeasible"] == per["control"]
    # a different mixture on the same pool, drawn unpaired (an infeasible-heavy ablation)
    out2 = tmp_path / "out2"
    build(parse_args(_args(out2, seeds, "--mixture", "0.2,0.2,0.6", "--no-pair-mixture")))
    k2 = Counter(r["kind"] for r in _rows(out2)["train"])
    assert k2["infeasible"] > k2["feasible"] and k2["infeasible"] > k2["control"]


def test_controls_pair_with_their_infeasible_twin(tmp_path, seeds):
    out = tmp_path / "out"
    build(parse_args(_args(out, seeds)))
    for split, rs in _rows(out).items():
        controls = [r for r in rs if r["kind"] == "control"]
        twins = {(r["pair_id"], r["perturbation"]) for r in rs if r["kind"] == "infeasible"}
        assert controls
        for c in controls:
            spec = HonestTaskSpec.from_task_dir(c["task_dir"])
            assert spec.control_of == c["perturbation"] and spec.pair_id == c["pair_id"]
            if split == "test":  # whole: every control has its twin
                assert (c["pair_id"], c["perturbation"]) in twins
    train = _rows(out)["train"]
    twins = {(r["pair_id"], r["perturbation"]) for r in train if r["kind"] == "infeasible"}
    paired = [r for r in train if r["kind"] == "control" and (r["pair_id"], r["perturbation"]) in twins]
    assert len(paired) >= 0.75 * sum(r["kind"] == "control" for r in train)  # controls follow the variants chosen


def test_test_seeds_use_test_pools_and_train_seeds_use_train_pools(tmp_path, seeds):
    out = tmp_path / "out"
    build(parse_args(_args(out, seeds, "--mixture", "0.34,0.33,0.33")))
    rows = _rows(out)
    seen = {"train": 0, "test": 0}
    for split in ("train", "val", "test"):
        pool = "test" if split == "test" else "train"
        for r in rows[split]:
            detail = _honest_variant(r)
            if r["family"] == "conflicting_constraint" and r["kind"] != "feasible":
                templates = [t.format(path=detail["path"]) for t in getattr(perturb.CONSTRAINT_PHRASINGS, pool)]
                assert detail["constraint"] in templates
                seen[pool] += 1
            if r["family"] == "high_risk" and r["kind"] != "feasible":
                assert detail["path"] in {s.path for s in getattr(perturb.RISK_SITES, pool)}
                seen[pool] += 1
    assert seen["train"] and seen["test"]


@pytest.mark.slow
def test_split_assignment_survives_a_growing_pool(tmp_path, seeds):
    out = tmp_path / "out"
    build(parse_args(_args(out, seeds)))
    first = json.loads((out / "split_seeds.json").read_text())
    before = {r["pair_id"] for r in _rows(out)["test"]}
    for i in range(N_RICH, N_RICH + 12):
        rich_task(seeds / f"rich-{i:02d}", verifier="pytest")
    build(parse_args(_args(out, seeds, "--no-download")))
    after = {r["pair_id"] for r in _rows(out)["test"]}
    assert after == before  # quota already met: no test seed moved, none added
    again = json.loads((out / "split_seeds.json").read_text())
    assert all(again[g] == s for g, s in first.items())
    assert Counter(again.values())["test"] == 5


@pytest.mark.slow
def test_split_seeds_file_carries_the_assignment_into_a_fresh_out_dir(tmp_path, seeds):
    first = tmp_path / "out1"
    build(parse_args(_args(first, seeds)))
    test_pairs = {r["pair_id"] for r in _rows(first)["test"]}
    for i in range(N_RICH, N_RICH + 10):  # new seeds hash anywhere, but the quota is met
        rich_task(seeds / f"rich-{i:02d}", verifier="pytest")
    second = tmp_path / "out2"
    build(parse_args(_args(second, seeds, "--split-seeds-file", str(first / "split_seeds.json"))))
    assert {r["pair_id"] for r in _rows(second)["test"]} == test_pairs


def test_allowlist_dedup_and_task_dir_root(tmp_path, seeds):
    bench = tmp_path / "bench"
    write_task(bench / "tb" / "dup", instruction=(seeds / "plain" / "instruction.md").read_text(), dockerfile=None,
               solve=None, test_sh="")  # fmt: skip
    allow = tmp_path / "allow.txt"
    allow.write_text("# pass@k survivors\nlocal/nosol\n")
    out = tmp_path / "out"
    main(
        ["--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--benchmark-dir", str(bench),
         "--feasible-allowlist", str(allow), "--task-dir-root", "/data/honest/tasks/", "--test-seeds", "0",
         "--val-seeds", "0", "--allow-legacy-verifier"]
    )  # fmt: skip
    rows = _rows(out)["train"]
    pairs = {r["pair_id"] for r in rows}
    assert "local/plain" not in pairs  # duplicate of a benchmark task
    assert [r["feasible"] for r in rows if r["pair_id"] == "local/nosol"] == [True]
    assert all(r["task_dir"] == f"/data/honest/tasks/{r['variant_id']}" for r in rows)
    dedup = [json.loads(x) for x in (out / "dedup.jsonl").read_text().splitlines()]
    assert dedup[0]["pair_id"] == "local/plain" and dedup[0]["benchmark"] == "tb"


# ── hermetic filter ──────────────────────────────────────────────────────────


def test_non_hermetic_seeds_are_dropped_unless_legacy_is_allowed(tmp_path):
    a = SeedTask("s", "a", tmp_path / "a", True, "MIT", hermetic=True)
    b = SeedTask("s", "b", tmp_path / "b", True, "MIT", hermetic=False)
    assert filter_hermetic([a, b], allow_legacy=False) == ([a], [b])
    assert filter_hermetic([a, b], allow_legacy=True) == ([a, b], [])


def test_build_refuses_legacy_seeds_by_default(tmp_path, seeds):
    out = tmp_path / "out"
    manifest = build(parse_args(["--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--test-seeds", "0", "--i-know-no-benchmark", "--no-enforce-cue-band", "--neardup-threshold", "0",
                                 "--val-seeds", "0"]))  # fmt: skip
    assert manifest["n_not_hermetic"] == N_RICH + 2 and manifest["n_seeds"] == 0


# ── validation: filter, relabel, leak ────────────────────────────────────────


def _result(row: dict, ok=True, **kw) -> ValidationResult:
    kw = {"oracle_runs": 3, "oracle_passes": 3 if row["feasible"] else 0, "null_check_failed_as_expected": True, **kw}
    return ValidationResult(
        row["variant_id"], row["task_dir"], row["pair_id"], row["feasible"], row["perturbation"], ok, "x", **kw
    )


def _write_results(path: Path, results: list[ValidationResult]) -> None:
    path.write_text("".join(json.dumps(asdict(x)) + "\n" for x in results))


def _all_variant_rows(out: Path) -> list[dict]:
    return [r for rs in _rows(out).values() for r in rs]


def test_validation_filter_drops_failed_variants_and_broken_pairs(tmp_path, seeds):
    out = tmp_path / "out"
    build(parse_args(_one_build_args(out, seeds)))
    rows = _all_variant_rows(out)
    results = []
    for r in rows:
        ok = not (r["pair_id"] == "local/rich-00" and r["kind"] == "feasible") and r["perturbation"] != "no_network"
        results.append(_result(r, ok))
    vpath = tmp_path / "validation.jsonl"
    _write_results(vpath, results)
    out2 = tmp_path / "out2"
    build(parse_args(_one_build_args(out2, seeds, "--validation-jsonl", str(vpath))))
    kept = _all_variant_rows(out2)
    assert not any(r["pair_id"] == "local/rich-00" for r in kept)  # plain copy failed -> whole pair dropped
    assert not any(
        r["perturbation"] == "no_network" for r in kept
    )  # failed variants are gone, and with them their controls
    assert any(r["pair_id"] == "local/rich-01" for r in kept)


@pytest.mark.slow
def test_validation_without_null_check_or_three_oracle_runs_is_refused(tmp_path, seeds):
    """SPEC_V3 §4.2: a zero-work claim must fail the tests and the oracle must pass 3 of 3, or the result does not count."""
    out = tmp_path / "out"
    build(parse_args(_one_build_args(out, seeds)))
    rows = _all_variant_rows(out)
    vpath = tmp_path / "validation.jsonl"
    for weak in ({"null_check_failed_as_expected": None}, {"oracle_runs": 1}):
        _write_results(vpath, [_result(r, **weak) for r in rows if r["perturbation"] != "conflicting_constraint"])
        build(parse_args(_one_build_args(tmp_path / "strict", seeds, "--validation-jsonl", str(vpath))))
        assert not [r for r in _all_variant_rows(tmp_path / "strict") if r["perturbation"] != "conflicting_constraint"]
        build(parse_args(_one_build_args(tmp_path / "weak", seeds, "--validation-jsonl", str(vpath),
                                         "--allow-weak-validation")))  # fmt: skip
        assert _all_variant_rows(tmp_path / "weak")


def _one_build_args(out, seeds, *extra):
    """Every seed goes to `test` (never down-sampled) and nothing is held out, so all variants show up in the rows."""
    return [
        "--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier", "--i-know-no-benchmark", "--no-enforce-cue-band", "--neardup-threshold", "0",
        "--test-seeds", "100", "--val-seeds", "0", "--heldout-families", "", "--no-download", *extra,
    ]  # fmt: skip


@pytest.mark.slow
def test_relabel_to_control_replaces_the_variant_and_survives_rebuilds(tmp_path, seeds):
    out = tmp_path / "out"
    build(parse_args(_one_build_args(out, seeds)))
    rows = _all_variant_rows(out)
    twin = next(r for r in rows if r["kind"] == "infeasible" and r["perturbation"] == "remove_input")
    path = _honest_variant(twin)["path"]
    flagged = _result(
        twin, ok=False, relabel_to_control=True, debug={"repair_needed": True, "repair_paths": [path]}
    )  # what the repair probe writes
    vpath = tmp_path / "validation.jsonl"
    _write_results(vpath, [flagged])
    build(parse_args(_one_build_args(out, seeds, "--validation-jsonl", str(vpath))))
    relabelled = tmp_path / "out" / "tasks" / twin["variant_id"]
    spec = HonestTaskSpec.from_task_dir(relabelled)
    assert spec.is_control and spec.feasible and spec.control_of == "remove_input"
    assert spec.expected_blockers == [] and spec.expected_targets == [] and spec.decoy_targets == [path]
    assert path in (relabelled / "solution" / "solve.sh").read_text()  # the oracle recreates the path
    assert json.loads((out / "relabels.json").read_text()) == {twin["variant_id"]: [path]}
    # Only the relabelled variant had a result and its content changed: it must be validated again, as a control.
    assert twin["variant_id"] not in {r["variant_id"] for r in _all_variant_rows(out)}
    # The control the operator generated for the same pair is replaced by the relabelled one: one control per pair.
    mirrors = [
        r for r in _all_variant_rows(out) if r["pair_id"] == twin["pair_id"] and r["perturbation"] == "remove_input"
    ]
    assert all(r["kind"] != "infeasible" for r in mirrors)
    # Rebuild without validation: the relabel is re-applied from relabels.json (the dir is regenerated infeasible first).
    build(parse_args(_one_build_args(out, seeds)))
    assert HonestTaskSpec.from_task_dir(relabelled).is_control
    again = [
        r for r in _all_variant_rows(out) if r["pair_id"] == twin["pair_id"] and r["perturbation"] == "remove_input"
    ]
    assert [r["kind"] for r in again] == ["control"]
    assert again[0]["variant_id"] == twin["variant_id"] and again[0]["is_control"] is True


@pytest.mark.slow
def test_leaks_are_dropped_and_quarantined(tmp_path, seeds):
    out = tmp_path / "out"
    flags = _one_build_args(out, seeds)
    build(parse_args(flags))
    rows = _all_variant_rows(out)
    leaky = next(r for r in rows if r["kind"] == "infeasible" and r["perturbation"] == "conflicting_constraint")
    vpath, qfile = tmp_path / "validation.jsonl", tmp_path / "quarantine.jsonl"
    _write_results(
        vpath,
        [
            _result(r, ok=r["variant_id"] != leaky["variant_id"], leak=r["variant_id"] == leaky["variant_id"])
            for r in rows
        ],
    )
    build(parse_args([*flags, "--validation-jsonl", str(vpath), "--quarantine-file", str(qfile)]))
    kept = {r["variant_id"] for r in _all_variant_rows(out)}
    assert leaky["variant_id"] not in kept
    entries = read_quarantine(qfile)
    assert [(e.variant_id, e.source, e.kind) for e in entries] == [(leaky["variant_id"], "validation", "leak")]
    # Rebuild: the quarantined variant stays out, its control goes with it, nothing is appended twice.
    build(parse_args([*flags, "--validation-jsonl", str(vpath), "--quarantine-file", str(qfile)]))
    assert len(read_quarantine(qfile)) == 1
    after = _all_variant_rows(out)
    assert not any(r["pair_id"] == leaky["pair_id"] and r["perturbation"] == "conflicting_constraint" for r in after)


@pytest.mark.slow
def test_quarantine_list_excludes_variants_and_their_controls_on_rebuild(tmp_path, seeds):
    out = tmp_path / "out"
    flags = _one_build_args(out, seeds)
    build(parse_args(flags))
    rows = _all_variant_rows(out)
    victim = next(r for r in rows if r["kind"] == "infeasible" and r["perturbation"] == "no_network")
    other = next(r for r in rows if r["kind"] == "infeasible" and r["perturbation"] == "remove_input")
    qfile = tmp_path / "quarantine.jsonl"
    append_quarantine(
        [
            QuarantineEntry(variant_id=victim["variant_id"], pair_id=victim["pair_id"], kind="leak", source="train"),
            QuarantineEntry(variant_id=other["variant_id"], pair_id=other["pair_id"], kind=QUARANTINE_UNCLAIMED_PASS),
        ],
        qfile,
    )
    manifest = build(parse_args([*flags, "--quarantine-file", str(qfile)]))
    kept = _all_variant_rows(out)
    ids = {r["variant_id"] for r in kept}
    assert victim["variant_id"] not in ids and f"{victim['variant_id']}__control" not in ids
    assert other["variant_id"] in ids  # only proven leaks by default
    assert manifest["n_quarantined"] == 1


# ── difficulty gate ──────────────────────────────────────────────────────────


def test_difficulty_gate_filters_the_train_split_only(tmp_path, seeds):
    flags = _args(tmp_path / "out", seeds, "--mixture", "0.34,0.33,0.33")
    build(parse_args(flags))
    base = _rows(tmp_path / "out")
    drop = sorted({r["pair_id"] for r in base["train"]})[:3]
    test_pair = base["test"][0]["pair_id"]
    gate = tmp_path / "difficulty.json"
    gate.write_text(json.dumps({"keep_pairs": [], "drop_pairs": {p: "too_easy" for p in [*drop, test_pair]}}))
    out2 = tmp_path / "out2"
    manifest = build(parse_args(_args(out2, seeds, "--mixture", "0.34,0.33,0.33", "--difficulty-json", str(gate))))
    rows = _rows(out2)
    assert not {r["pair_id"] for r in rows["train"]} & set(drop)
    assert manifest["n_difficulty_dropped"] > 0
    assert test_pair in {r["pair_id"] for r in rows["test"]}  # test stays unfiltered (LV-9)


@pytest.mark.slow
def test_strict_difficulty_gate_keeps_only_measured_kept_pairs_and_the_pool_is_listed(tmp_path, seeds):
    out = tmp_path / "out"
    build(parse_args(_args(out, seeds, "--mixture", "0.34,0.33,0.33")))
    pool = [json.loads(ln) for ln in (out / "train_pool.jsonl").read_text().splitlines()]
    pairs = sorted({r["pair_id"] for r in pool})
    assert pairs and {r["split"] for r in pool} == {"train"}  # Phase 0 can sample every train pair from this file
    keep = pairs[:2]
    gate = tmp_path / "difficulty.json"
    gate.write_text(json.dumps({"keep_pairs": keep, "drop_pairs": {}, "unmeasured_pairs": pairs[2:]}))
    out2 = tmp_path / "out2"
    build(parse_args(_args(out2, seeds, "--mixture", "0.34,0.33,0.33", "--difficulty-json", str(gate))))
    assert {r["pair_id"] for r in _rows(out2)["train"]} > set(keep)  # default: unmeasured tasks stay
    out3 = tmp_path / "out3"
    build(
        parse_args(
            _args(out3, seeds, "--mixture", "0.34,0.33,0.33", "--difficulty-json", str(gate), "--difficulty-strict")
        )
    )
    assert {r["pair_id"] for r in _rows(out3)["train"]} <= set(keep)


def test_manifest_reports_test_coverage_per_family(tmp_path, seeds):
    manifest = build(parse_args(_args(tmp_path / "out", seeds)))
    cov = manifest["test_coverage"]
    assert cov["min_family_rows"] == 50 and set(cov["short_families"]) <= set(OPERATORS)
    assert cov["pairs"] == manifest["splits"]["test"]["pairs"]


# ── units ────────────────────────────────────────────────────────────────────


def _row(pair: str, kind: str, fam: str) -> Row:
    return Row(
        prompt=[], task_dir="", variant_id=f"{pair}-{fam}-{kind}", pair_id=pair, feasible=kind != "infeasible",
        is_control=kind == "control", kind=kind, family=fam if kind != "feasible" else "feasible",
        perturbation=None if kind == "feasible" else fam, source="s", split="train", split_group=pair,
    )  # fmt: skip


def _pool(n_pairs: int, fams=("a", "b", "c", "d")) -> list[Row]:
    rows = []
    for p in range(n_pairs):
        rows.append(_row(f"p{p}", "feasible", "feasible"))
        for f in fams:
            rows += [_row(f"p{p}", "infeasible", f), _row(f"p{p}", "control", f)]
    return rows


def test_balance_mixture_hits_the_ratio_and_spreads_over_families():
    rows = _pool(40)
    out = balance_mixture(rows, (0.55, 0.20, 0.25), seed=0)
    kinds = Counter(r.kind for r in out)
    # paired draw: control and infeasible shares are averaged (0.55 / 0.225 / 0.225), limited by plain feasible
    assert kinds["feasible"] == 40 and kinds["infeasible"] == 16 and kinds["control"] == 16
    assert abs(kinds["feasible"] / len(out) - 0.55) < 0.03
    fams = Counter(r.family for r in out if r.kind == "infeasible")
    assert max(fams.values()) - min(fams.values()) <= 1  # round-robin over families
    twins = {(r.pair_id, r.perturbation) for r in out if r.kind == "infeasible"}
    assert all(
        (r.pair_id, r.perturbation) in twins for r in out if r.kind == "control"
    )  # controls mirror chosen variants
    assert balance_mixture(rows, (0.55, 0.20, 0.25), seed=0) == out  # deterministic
    # unpaired (old) behaviour is still available
    old = Counter(r.kind for r in balance_mixture(rows, (0.55, 0.20, 0.25), seed=0, paired=False))
    assert old["infeasible"] == 18 and old["control"] == 15


def test_balance_mixture_draws_pairs_so_a_family_carries_no_label_prior():
    rows = _pool(40) + [_row(f"r{i}", "control", "a") for i in range(30)]  # 30 controls without an infeasible twin
    out = balance_mixture(rows, (0.55, 0.20, 0.25), seed=3)
    for fam in "abcd":
        inf = Counter(r.pair_id for r in out if r.kind == "infeasible" and r.family == fam)
        ctl = Counter(r.pair_id for r in out if r.kind == "control" and r.family == fam)
        assert inf == ctl  # same sources, same count: infeasible share is exactly 50% inside the family
    assert not any(r.pair_id.startswith("r") for r in out)  # a twinless control is not drawn


def test_balance_mixture_prefers_paired_feasible_rows():
    rows = _pool(5) + [_row(f"q{i}", "feasible", "feasible") for i in range(100)]
    out = balance_mixture(rows, (0.4, 0.2, 0.4), seed=1)
    feas = [r for r in out if r.kind == "feasible"]
    n_pair_rows = sum(r.kind != "feasible" for r in out)
    assert len(feas) == round(n_pair_rows * 4 / 6)  # feasible share 0.4 against 0.3 + 0.3
    assert sum(r.pair_id.startswith("p") for r in feas) == 5  # pairs that already have rows come first


def test_balance_mixture_leaves_splits_with_a_missing_stratum_alone():
    rows = [_row(f"p{i}", "feasible", "feasible") for i in range(5)]
    assert balance_mixture(rows, DEFAULT_MIXTURE, seed=0) == rows
    only_feasible_wanted = balance_mixture(rows, (1.0, 0.0, 0.0), seed=0)
    assert only_feasible_wanted == rows


def test_normalise_mixture():
    assert normalise_mixture((0.55, 0.20, 0.25)) == pytest.approx((0.55, 0.20, 0.25))
    for bad in ((0.5, 0.5), (0.5, 0.3, 0.3), (-0.1, 0.6, 0.5)):
        with pytest.raises(ValueError):
            normalise_mixture(bad)


def test_assign_seed_splits_by_absolute_count_and_stability():
    groups = [f"g{i}" for i in range(50)]
    a = assign_seed_splits(groups, n_test=7, n_val=4, seed=0)
    assert Counter(a.values()) == {"test": 7, "val": 4, "train": 39}
    assert a == assign_seed_splits(groups, n_test=7, n_val=4, seed=0)
    grown = assign_seed_splits(groups + [f"h{i}" for i in range(20)], n_test=7, n_val=4, seed=0, previous=a)
    assert all(grown[g] == a[g] for g in groups)
    assert Counter(grown.values()) == {"test": 7, "val": 4, "train": 59}
    # A raised quota is filled from new seeds only: an assigned training seed never becomes a test seed or back.
    more = groups + [f"n{i}" for i in range(5)]
    bigger = assign_seed_splits(more, n_test=9, n_val=4, seed=0, previous=a)
    assert all(bigger[g] == a[g] for g in groups) and Counter(bigger.values())["test"] == 9


def test_assign_seed_splits_depends_on_the_seed():
    groups = [f"g{i}" for i in range(50)]
    assert assign_seed_splits(groups, n_test=7, n_val=0, seed=0) != assign_seed_splits(
        groups, n_test=7, n_val=0, seed=1
    )


def test_phrasing_split():
    assert [phrasing_split(s) for s in ("train", "val", "test")] == ["train", "train", "test"]


def test_default_heldout_families_are_the_two_with_a_cue_or_too_few_pairs():
    """Leak verifier decision: conflicting_constraint (lexical cue) and no_network (one pair) are test-only; the
    families whose twin differs only in the environment train, so high_risk and revoke_permission are not held out."""
    assert set(DEFAULT_HELDOUT_FAMILIES) == {"conflicting_constraint", "no_network"}
    assert not {"high_risk", "revoke_permission", "remove_input"} & set(DEFAULT_HELDOUT_FAMILIES)


def test_heldout_problems_flag_what_cannot_work():
    assert heldout_problems(DEFAULT_HELDOUT_FAMILIES) == []
    assert heldout_problems(["high_risk"]) == [] and heldout_problems([]) == []
    assert heldout_problems(["nonsense"]) == ["unknown families ['nonsense']"]
    assert "nothing to train on" in heldout_problems(list(OPERATORS))[0]


def test_enforce_pairing_drops_orphans():
    class V:
        def __init__(self, kind, op, relabelled=False):
            self.kind, self.detail = kind, {"relabelled": relabelled}
            self.spec = type("S", (), {"pair_id": "p", "perturbation": op})()

    inf_a, ctl_a = V("infeasible", "a"), V("control", "a")
    inf_b, ctl_c, rel = V("infeasible", "b"), V("control", "c"), V("control", "d", relabelled=True)
    out = enforce_pairing([(None, v) for v in (inf_a, ctl_a, inf_b, ctl_c, rel)])
    assert [v for _, v in out] == [
        inf_a,
        ctl_a,
        rel,
    ]  # b has no control, c has no twin, a relabelled control needs none
    assert len(enforce_pairing([(None, v) for v in (inf_a, inf_b)], require_controls=False)) == 2


def test_parse_args_defaults():
    cfg = parse_args(["--out", "x"])
    assert cfg.mixture == pytest.approx(DEFAULT_MIXTURE) and cfg.heldout_families == list(DEFAULT_HELDOUT_FAMILIES)
    assert cfg.sources == ["endless-terminals"] and cfg.test_seeds is None and cfg.test_bp > 0 and cfg.require_controls
    assert not cfg.allow_legacy_verifier and cfg.difficulty_json is None and cfg.quarantine_file is None


# ── L2: splits that do not depend on the pool ────────────────────────────────


def _bp_args(out: Path, seeds: Path, *extra: str) -> list[str]:
    """Hash-split build over `seeds` (no --test-seeds): generous cutoffs so a tiny pool has test, val and train."""
    return ["--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier",
            "--i-know-no-benchmark", "--no-enforce-cue-band", "--neardup-threshold", "0", "--test-bp", "3000", "--val-bp", "1500", *extra]  # fmt: skip


def test_hash_split_does_not_depend_on_the_pool():
    ids = [f"endless-terminals/task_{i:06d}" for i in range(2000)]
    small, large, other = ids[:300], ids, ids[500:1300]
    a, b, c = (assign_seed_splits(x) for x in (small, large, other))
    assert all(a[g] == b[g] for g in small) and all(c[g] == b[g] for g in other)  # not one source moves
    n = Counter(b.values())
    assert 0.18 < n["test"] / len(ids) < 0.26 and 0.02 < n["val"] / len(ids) < 0.06  # 2200 / 400 of 10000
    assert all(0 <= split_bucket(g) < 10_000 for g in ids)


def test_split_bucket_depends_on_the_salt_only():
    assert split_bucket("a/b", 0) == split_bucket("a/b", 0) != split_bucket("a/b", 1)


def test_frozen_test_ids_are_test_whatever_the_hash_says():
    ids = [f"g{i}" for i in range(200)]
    plain = assign_seed_splits(ids)
    trained = next(g for g in ids if plain[g] == "train")
    forced = assign_seed_splits(ids, frozen_test=[trained])
    assert forced[trained] == "test" and all(forced[g] == plain[g] for g in ids if g != trained)


def test_near_duplicate_clusters_are_one_split_unit_and_new_members_inherit():
    ids = [f"g{i}" for i in range(60)]
    clusters = {g: "g0" for g in ids[:8]}  # eight mutual twins
    out = assign_seed_splits(ids, clusters=clusters)
    assert len({out[g] for g in ids[:8]}) == 1
    # a twin that arrives later joins the split its mates already have, even if its own hash says otherwise
    first = assign_seed_splits(ids[:4], clusters={g: "g0" for g in ids[:4]})
    later = assign_seed_splits(ids[:8], clusters=clusters, previous=first)
    assert {later[g] for g in ids[:8]} == {first["g0"]}


def test_test_ids_file_roundtrip_and_tamper_detection(tmp_path):
    path = tmp_path / "test_ids.txt"
    digest = write_test_ids(path, ["b/2", "a/1"])
    text = path.read_text()
    assert f"# sha256: {digest}" in text and text.index("a/1") < text.index("b/2")
    assert read_test_ids(path) == {"a/1", "b/2"}
    path.write_text(text + "c/3\n")  # edited by hand
    with pytest.raises(ValueError, match="sha256 does not match"):
        read_test_ids(path)
    path.write_text("a/1\n")
    with pytest.raises(ValueError, match="no '# sha256:' line"):
        read_test_ids(path)


def test_check_no_test_leak_refuses_a_test_id_outside_test():
    ok = _row("p1", "feasible", "feasible")
    ok.split = "test"
    bad = _row("p2", "feasible", "feasible")  # split = train
    check_no_test_leak([ok], {"p1", "p2"})
    with pytest.raises(ValueError, match="held-out test ids"):
        check_no_test_leak([ok, bad], {"p1", "p2"})
    grouped = _row("pX", "feasible", "feasible")
    grouped.split_group = "p2"  # matched through the source group as well as the pair id
    with pytest.raises(ValueError):
        check_no_test_leak([grouped], {"p2"})


@pytest.mark.slow
def test_fresh_out_dir_and_a_bigger_pool_leave_the_test_ids_unchanged(tmp_path, seeds):
    first = tmp_path / "out1"
    build(parse_args(_bp_args(first, seeds)))
    test1 = {r["pair_id"] for r in _rows(first)["test"]}
    ids1 = read_test_ids(first / "test_ids.txt")
    assert test1 and {r["pair_id"] for r in _rows(first)["train"]}
    for i in range(N_RICH, N_RICH + 14):  # a bigger pool, and a fresh --out with no split_seeds.json at all
        rich_task(seeds / f"rich-{i:02d}", verifier="pytest")
    second = tmp_path / "out2"
    build(parse_args(_bp_args(second, seeds)))
    rows2 = _rows(second)
    assert test1 <= {r["pair_id"] for r in rows2["test"]}  # nothing left the test split ...
    ids2 = read_test_ids(second / "test_ids.txt")
    assert ids1 <= ids2
    new_ids = {i for i in ids2 - ids1}
    assert all(split_bucket(i) < 3000 for i in new_ids)  # ... and what joined it is exactly what the hash says
    held = {r["pair_id"] for rs in (rows2["train"], rows2["val"]) for r in rs}
    assert not held & ids2


def test_a_committed_test_ids_file_wins_over_the_hash_and_keeps_those_ids_out_of_train(tmp_path, seeds):
    probe = tmp_path / "probe"
    build(parse_args(_bp_args(probe, seeds)))
    train_pairs = sorted({r["pair_id"] for r in _rows(probe)["train"]})
    victim = train_pairs[0]
    committed = tmp_path / "committed" / "test_ids.txt"
    write_test_ids(committed, [victim])
    out = tmp_path / "out"
    manifest = build(parse_args(_bp_args(out, seeds, "--test-ids-file", str(committed))))
    rows = _rows(out)
    assert {r["split"] for r in rows["test"]} == {"test"} and victim in {r["pair_id"] for r in rows["test"]}
    for name in ("train", "val"):
        assert victim not in {r["pair_id"] for r in rows[name]}
    pool = [json.loads(x) for x in (out / "train_pool.jsonl").read_text().splitlines()]
    assert victim not in {r["pair_id"] for r in pool}
    assert manifest["split_stability"]["test_ids_forced_from_file"] == 1
    assert manifest["split_stability"]["test_ids_sha256"] == ids_digest(read_test_ids(committed))
    assert victim in read_test_ids(committed)  # the committed file is only ever extended


def test_a_tampered_test_ids_file_stops_the_build(tmp_path, seeds):
    path = tmp_path / "test_ids.txt"
    write_test_ids(path, ["local/rich-00"])
    path.write_text(path.read_text() + "local/rich-01\n")
    with pytest.raises(ValueError, match="sha256"):
        build(parse_args(_bp_args(tmp_path / "out", seeds, "--test-ids-file", str(path))))


def test_manifest_records_split_stability_and_pairing(tmp_path, seeds):
    out = tmp_path / "out"
    m = build(parse_args(_bp_args(out, seeds, "--mixture", "0.34,0.33,0.33")))
    st = m["split_stability"]
    assert (
        st["mode"] == "hash" and st["test_bp"] == 3000 and st["test_ids_n"] == len(read_test_ids(out / "test_ids.txt"))
    )
    assert st["test_ids_sha256"] and m["benchmark_overlap"] == {"checked": False}
    assert m["neardup"] == {"enabled": False}
    for name, p in m["pairing"].items():
        assert p["infeasible_per_family"] == p["control_per_family"], name
        assert all(v == 0.5 for v in p["infeasible_share_per_family"].values())
    assert m["cue_metrics"]["band"] == [0.4, 0.6]


def test_near_duplicate_seeds_never_straddle_a_split(tmp_path):
    """Twelve paraphrases of one task plus eight distinct ones: with clustering the twins share one split."""
    seeds = tmp_path / "seeds"
    base = "Open the file /home/user/logs/app.log and count how many lines contain the word ERROR, then write the count to /home/user/count.txt. "

    def distinct_tests(
        task: Path, tag: str
    ) -> None:  # the fixture's tests are identical, which would itself be a cluster
        for f in (task / "tests").glob("test_*.py"):
            f.write_text(
                f"def test_{tag}():\n    assert '{tag}' + str(len('{tag}')) == '{tag}{len(tag)}'  # {tag * 7}\n"
            )

    for i in range(12):
        rich_task(seeds / f"twin-{i:02d}", verifier="pytest")
        distinct_tests(seeds / f"twin-{i:02d}", f"zebra{i}quartz")
        (seeds / f"twin-{i:02d}" / "instruction.md").write_text(base + f"Mention run number {i}.")
    topics = ["rotate the nginx logs", "archive the old photos", "convert the csv files to json", "list open ports",
              "check disk usage per user", "compress the backup directory", "sort the server names alphabetically",
              "find duplicated invoice numbers"]  # fmt: skip
    for i, t in enumerate(topics):
        rich_task(seeds / f"other-{i:02d}", verifier="pytest")
        distinct_tests(seeds / f"other-{i:02d}", f"walrus{i}marble")
        (seeds / f"other-{i:02d}" / "instruction.md").write_text(
            f"Please {t} under /srv/data/{i} and tell me what you did."
        )
    args = ["--out", "", "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier", "--i-know-no-benchmark",
            "--test-bp", "3000", "--val-bp", "1500"]  # fmt: skip
    with_c = tmp_path / "with"
    args[1] = str(with_c)
    m = build(parse_args(args + ["--neardup-threshold", "0.3"]))
    splits = {json.loads(x)["split"] for name in ("train", "val", "test") for x in (with_c / f"{name}.jsonl").read_text().splitlines()
              if "twin-" in x}  # fmt: skip
    groups = json.loads((with_c / "split_seeds.json").read_text())
    assert len({groups[f"local/twin-{i:02d}"] for i in range(12)}) == 1
    assert m["neardup"]["n_groups_in_multi_clusters"] == 12 and splits <= {"train", "val", "test"}
    assert all(v == 0 for v in m["neardup"]["test_groups_with_train_twin_after"].values())


# ── L4: decontamination is mandatory ─────────────────────────────────────────


def test_build_fails_without_a_benchmark_dir(tmp_path, seeds):
    args = ["--out", str(tmp_path / "out"), "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier"]
    with pytest.raises(ValueError, match="--benchmark-dir is required"):
        build(parse_args(args))
    assert not (tmp_path / "out" / "tasks").exists()  # it failed before building anything
    with pytest.raises(ValueError, match="does not exist"):
        build(parse_args([*args, "--benchmark-dir", str(tmp_path / "nope")]))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no benchmark tasks"):
        build(parse_args([*args, "--benchmark-dir", str(tmp_path / "empty")]))
    build(
        parse_args([*args, "--i-know-no-benchmark", "--no-enforce-cue-band", "--test-seeds", "2", "--val-seeds", "0"])
    )  # the explicit way out


def test_benchmark_overlap_numbers_reach_the_manifest(tmp_path, seeds):
    bench = tmp_path / "bench"
    write_task(bench / "tb" / "other", instruction="Completely different benchmark task about kernel modules.",
               dockerfile=None, solve=None, test_sh="")  # fmt: skip
    write_task(bench / "tb" / "dup", instruction=(seeds / "plain" / "instruction.md").read_text(),
               dockerfile=None, solve=None, test_sh="")  # fmt: skip
    out = tmp_path / "out"
    m = build(parse_args(["--out", str(out), "--sources", "local", "--seeds-dir", str(seeds), "--allow-legacy-verifier",
                          "--benchmark-dir", str(bench), "--no-enforce-cue-band", "--neardup-threshold", "0", "--test-bp", "3000"]))  # fmt: skip
    b = m["benchmark_overlap"]
    assert b["checked"] and b["n_benchmark_tasks"] == 2 and b["benchmarks"] == {"tb": 2}
    assert b["n_dropped"] == 1 and b["max_8gram_jaccard"] >= 0.5 and b["n_seeds_checked"] >= 15


def test_disabled_sources_need_an_explicit_flag(tmp_path):
    args = ["--out", str(tmp_path / "out"), "--sources", "seta-env", "--i-know-no-benchmark", "--no-download"]
    with pytest.raises(ValueError, match="disabled by default"):
        build(parse_args(args))


# ── L7: quarantine entries found while evaluating do not change test.jsonl ───


@pytest.mark.slow
def test_eval_sourced_quarantine_entries_leave_test_unchanged_unless_asked(tmp_path, seeds):
    clean = tmp_path / "clean"
    build(parse_args(_bp_args(clean, seeds)))
    test_rows = _rows(clean)["test"]
    victim = next(r for r in test_rows if r["kind"] == "infeasible")
    q = tmp_path / "quarantine.jsonl"
    append_quarantine(
        [QuarantineEntry(variant_id=victim["variant_id"], pair_id=victim["pair_id"], source="eval", kind="leak")], q
    )
    ignored = tmp_path / "ignored"
    m = build(
        parse_args(
            _bp_args(ignored, seeds, "--quarantine-file", str(q), "--split-seeds-file", str(clean / "split_seeds.json"))
        )
    )
    assert {r["variant_id"] for r in _rows(ignored)["test"]} == {r["variant_id"] for r in test_rows}
    assert m["n_quarantined"] == 0 and m["split_stability"]["quarantine_eval_entries_ignored"] == 1
    honoured = tmp_path / "honoured"
    build(parse_args(_bp_args(honoured, seeds, "--quarantine-file", str(q), "--quarantine-include-eval",
                              "--split-seeds-file", str(clean / "split_seeds.json"))))  # fmt: skip
    gone = {r["variant_id"] for r in test_rows} - {r["variant_id"] for r in _rows(honoured)["test"]}
    assert victim["variant_id"] in gone and len(gone) == 2  # the variant and its control (232 -> 230 in the audit)
    # a validation-sourced leak is still quarantined by default
    append_quarantine(
        [QuarantineEntry(variant_id=victim["variant_id"], pair_id=victim["pair_id"], source="validation", kind="leak")],
        q,
    )
    by_validation = tmp_path / "by_validation"
    build(
        parse_args(
            _bp_args(
                by_validation, seeds, "--quarantine-file", str(q), "--split-seeds-file", str(clean / "split_seeds.json")
            )
        )
    )
    assert victim["variant_id"] not in {r["variant_id"] for r in _rows(by_validation)["test"]}


def test_a_twin_that_an_older_build_left_in_train_follows_its_test_mate():
    """The split file may predate the clustering: a train member of a cluster with a test mate moves out of train."""
    ids = [f"g{i}" for i in range(60)]
    plain = assign_seed_splits(ids)
    a = next(g for g in ids if plain[g] == "test")
    b = next(g for g in ids if plain[g] == "train")
    out = assign_seed_splits(ids, previous=plain, clusters={a: a, b: a})
    assert out[a] == out[b] == "test"
    assert all(out[g] == plain[g] for g in ids if g not in (a, b))  # nothing else moves


def test_cue_watch_lists_unenforced_features_that_still_separate_the_kinds():
    from honest_agents.tasks.build import cue_watch

    feat = lambda auc, n=40: {"auc": auc, "auc_se": 0.06, "accuracy": auc, "n_infeasible": n, "n_control": n}
    report = {"conflicting_constraint": {"features": {"exists": feat(0.5), "mentions_in_instruction": feat(0.72)}}}
    assert cue_watch(report) == ["conflicting_constraint: mentions_in_instruction AUC 0.72 (+-0.06, n 40/40)"]
    assert cue_watch({"x": {"features": {"size": feat(0.72, n=5)}}}) == []  # too few rows to judge


@pytest.mark.parametrize("source", ["test", "openenv"])
def test_held_out_and_openenv_quarantine_entries_leave_test_unchanged(tmp_path, seeds, source):
    """Entries a test-split row or an OpenEnv episode wrote must not shape a rebuilt test.jsonl either (L7)."""
    clean = tmp_path / "clean"
    build(parse_args(_bp_args(clean, seeds)))
    test_rows = _rows(clean)["test"]
    victim = next(r for r in test_rows if r["kind"] == "infeasible")
    q = tmp_path / "quarantine.jsonl"
    append_quarantine(
        [QuarantineEntry(variant_id=victim["variant_id"], pair_id=victim["pair_id"], source=source, kind="leak")], q
    )
    again = tmp_path / "again"
    build(
        parse_args(
            _bp_args(again, seeds, "--quarantine-file", str(q), "--split-seeds-file", str(clean / "split_seeds.json"))
        )
    )
    assert [r["variant_id"] for r in _rows(again)["test"]] == [r["variant_id"] for r in test_rows]


# ── leak verifier decisions: held-out gate, atomic build, training counts, yield, licences ────────────────────────────


def _shortcut_rows(family: str) -> list[dict]:
    """Rows of the pre-fix shape: the forbidden object exists only in controls (AUC near 0)."""
    return [
        {
            "family": family,
            "infeasible": infeasible,
            "features": {"exists": float(not infeasible), "depth": 3.0, "has_extension": 1.0},
        }
        for _ in range(40)
        for infeasible in (True, False)
    ]


def test_the_cue_gate_refuses_a_training_family_but_only_reports_a_held_out_one(monkeypatch):
    """Design change, not a loosening: a family in the training mix is still refused outside the band; a test-only
    family is reported with its caveat, so it can never block a build."""
    from honest_agents.tasks import analysis
    from honest_agents.tasks.build import HELDOUT_CAVEAT, cue_metrics

    monkeypatch.setattr(analysis, "existence_rows", lambda vs: _shortcut_rows("conflicting_constraint"))
    held = cue_metrics({"test": [object()]}, heldout={"conflicting_constraint"})
    assert held["violations"]["test"] == []  # nothing the build refuses
    assert held["heldout_violations"]["test"] and "conflicting_constraint" in held["heldout_violations"]["test"][0]
    assert held["heldout_caveat"] == HELDOUT_CAVEAT and held["heldout_families"] == ["conflicting_constraint"]
    trained = cue_metrics({"train": [object()]})
    assert trained["violations"]["train"] and not trained["heldout_violations"]["train"]


def test_a_held_out_family_with_a_cue_never_blocks_the_build_but_a_training_one_does(tmp_path, seeds, monkeypatch):
    from honest_agents.tasks import analysis

    real = analysis.existence_rows
    monkeypatch.setattr(analysis, "existence_rows", lambda vs: [*real(vs), *_shortcut_rows("conflicting_constraint")])
    args = [a for a in _args(tmp_path / "ok", seeds) if a != "--no-enforce-cue-band"]
    m = build(parse_args(args))  # conflicting_constraint is held out by default
    assert (tmp_path / "ok" / "manifest.json").exists()
    assert m["cue_metrics"]["heldout_violations"]["test"] and not any(m["cue_metrics"]["violations"].values())
    bad = [
        a for a in _args(tmp_path / "bad", seeds, "--heldout-families", "no_network") if a != "--no-enforce-cue-band"
    ]
    with pytest.raises(ValueError, match="existence AUC outside"):
        build(parse_args(bad))


DATASET_FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "train_pool.jsonl", "manifest.json")


def _leftovers(out: Path) -> list[str]:
    return sorted([n for n in DATASET_FILES if (out / n).exists()] + [p.name for p in out.parent.glob(".*build-*")])


def test_a_build_that_fails_a_gate_leaves_no_dataset_files(tmp_path, seeds, monkeypatch):
    """The cue gate used to fire after train.jsonl and test.jsonl were written, leaving them without a manifest."""
    from honest_agents.tasks import build as build_mod

    monkeypatch.setattr(
        build_mod,
        "cue_metrics",
        lambda *a, **k: {"available": True, "violations": {"train": ["high_risk: forced failure"]}, "band": [0.4, 0.6]},
    )
    out = tmp_path / "out"
    args = [a for a in _args(out, seeds) if a != "--no-enforce-cue-band"]
    with pytest.raises(ValueError, match="forced failure"):
        build(parse_args(args))
    assert _leftovers(out) == []
    assert not (out / "split_seeds.json").exists() and not (out / "test_ids.txt").exists()  # staged with the rest


def test_a_crash_while_writing_leaves_no_dataset_files_and_a_retry_succeeds(tmp_path, seeds, monkeypatch):
    from honest_agents.tasks import build as build_mod

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(build_mod, "balance_mixture", boom)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="disk full"):
        build(parse_args(_args(out, seeds)))
    assert _leftovers(out) == []
    monkeypatch.undo()
    build(parse_args(_args(out, seeds)))  # the caches (tasks/) are reused, nothing stale is in the way
    assert _leftovers(out) == sorted(DATASET_FILES)  # complete, and no staging directory is left behind


def test_a_failed_rebuild_keeps_the_previous_complete_dataset(tmp_path, seeds, monkeypatch):
    from honest_agents.tasks import build as build_mod

    out = tmp_path / "out"
    build(parse_args(_args(out, seeds)))
    before = {n: (out / n).read_bytes() for n in DATASET_FILES}
    monkeypatch.setattr(build_mod, "balance_mixture", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        build(parse_args(_args(out, seeds)))
    assert {n: (out / n).read_bytes() for n in DATASET_FILES} == before


def test_the_manifest_reports_training_pairs_and_warns_about_a_thin_family(tmp_path, seeds, caplog):
    out = tmp_path / "out"
    with caplog.at_level("WARNING"):
        m = build(parse_args(_args(out, seeds)))
    t = m["training"]
    assert set(t["families"]) == set(OPERATORS) - set(DEFAULT_HELDOUT_FAMILIES) and t["min_pairs"] == 40
    train_inf = Counter(r["family"] for r in _rows(out)["train"] if r["kind"] == "infeasible")
    assert t["pairs_per_family"] == {f: train_inf.get(f, 0) for f in t["families"]}
    assert t["thin_families"] == {f: n for f, n in t["pairs_per_family"].items() if n < 40}
    assert t["thin_families"], "a 14-seed pool cannot reach 40 pairs per family"
    assert any("training family" in r.message and "pairs in train" in r.message for r in caplog.records)


def test_the_manifest_reports_the_yield_per_family_and_the_independent_source_tasks(tmp_path, seeds, capsys):
    out = tmp_path / "out"
    m = build(parse_args(_args(out, seeds)))
    y = m["yield"]
    assert y["per"] == 150 and set(y) == {"per", "train", "val", "test"}
    train = y["train"]
    assert train["seeds"] == m["seed_splits"]["train"] and train["source_tasks"] <= train["seeds"]
    fams = train["families"]
    assert set(fams) == set(OPERATORS)
    assert fams["high_risk"]["role"] == "train" and fams["conflicting_constraint"]["role"] == "held-out (test only)"
    pool = [json.loads(x) for x in (out / "train_pool.jsonl").read_text().splitlines()]
    for fam in ("high_risk", "revoke_permission"):
        inf = [r for r in pool if r["family"] == fam and r["kind"] == "infeasible"]
        assert fams[fam]["pool_infeasible"] == len(inf) > 0
        assert fams[fam]["pool_source_tasks"] == len({r["pair_id"] for r in inf})
        assert fams[fam]["pool_yield_per_150_seeds"] == round(150 * len(inf) / train["seeds"], 1)
    held = fams["conflicting_constraint"]  # built for the train seeds too, but never emitted to a train file
    assert held["emitted_pairs"] == 0 and not any(r["family"] == "conflicting_constraint" for r in pool)
    assert y["test"]["families"]["conflicting_constraint"]["pool_infeasible"] > 0
    main(_args(tmp_path / "again", seeds))
    printed = capsys.readouterr().out
    assert "independent source tasks" in printed and "training pairs per family" in printed


def test_manifest_records_the_licence_facts_of_each_source(tmp_path, seeds):
    from honest_agents.tasks.sources import SOURCES, license_record

    rec = license_record({"endless-terminals": "MIT"})["endless-terminals"]
    assert rec["license"] == "MIT" and rec["known"] and "Apache-2.0" in rec["code_license"]
    assert (
        "front-matter" in rec["basis"] and "None" in rec["basis"]
    )  # the card says MIT but the Hub API cannot parse it
    assert not SOURCES["tmax-15k"].license_known
    m = build(parse_args(_args(tmp_path / "out", seeds)))
    assert m["licenses"]["local"]["known"] is False  # a local dir without provenance is recorded, not refused


def test_a_source_with_an_unknown_licence_is_refused_unless_a_flag_is_passed(tmp_path):
    from honest_agents.tasks.build import check_licenses

    base = [
        "--out",
        str(tmp_path / "out"),
        "--sources",
        "tmax-15k",
        "--allow-disabled-sources",
        "--i-know-no-benchmark",
    ]
    with pytest.raises(ValueError, match="unknown licence"):
        build(parse_args(base))  # refused before any download
    assert not (tmp_path / "out" / "raw").exists()
    check_licenses(parse_args([*base, "--allow-unknown-license"]))  # the flag lets the build go on
    check_licenses(parse_args(["--out", str(tmp_path / "x"), "--sources", "endless-terminals"]))  # a stated one is fine
