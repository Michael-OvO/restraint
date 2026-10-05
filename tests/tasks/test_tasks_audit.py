"""Audit set (docs/design.md section 7): a small dataset of two families plus controls, with one markdown sheet per variant."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from honest_agents.tasks import audit
from honest_agents.tasks.audit import (
    AuditConfig,
    parse_args,
    read_verdict,
    render_sheet,
    tally,
    write_audit,
)
from honest_agents.tasks.build import COLUMNS, write_test_ids
from honest_agents.tasks.perturb import generate_variants
from honest_agents.tasks.sources import local_seeds
from honest_agents.tasks.validate import ValidationResult
from tests.tasks.synthetic import RICH_SOLVE, rich_task, varied_inputs


@pytest.fixture
def seeds(tmp_path) -> Path:
    root = tmp_path / "seeds"
    for i in range(5):
        rich_task(root / f"rich-{i}", verifier="pytest", inputs=varied_inputs(i))
    for i in range(3):  # only conflicting_constraint applies: the oracle never reads an input
        t = rich_task(root / f"partial-{i}", verifier="pytest", inputs=varied_inputs(i + 10))
        (t / "solution" / "solve.sh").write_text(
            RICH_SOLVE.replace("cat /home/user/data/clips.txt /home/user/data/settings.conf >> ", "echo x >> ")
        )
    return root


def _cfg(tmp_path, seeds, **kw) -> AuditConfig:
    # Synthetic seeds: no benchmark, none hashes into test. These tests exercise two particular families whatever the
    # default audit families are (they became the training families: see `test_the_default_audit_families_...`).
    kw = {"i_know_no_benchmark": True, "test_bp": 0, "families": ("remove_input", "conflicting_constraint")} | kw
    return AuditConfig(out=tmp_path / "audit", sources=["local"], seeds_dir=seeds, allow_legacy_verifier=True, **kw)


def _write(cfg: AuditConfig) -> dict:
    return write_audit(cfg)


def test_audit_set_has_the_requested_shape(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=5)
    summary = _write(cfg)
    out = cfg.out
    index = [json.loads(x) for x in (out / "audit.jsonl").read_text().splitlines()]
    assert summary["n_sources"] == 5 and summary["n_variants"] == len(index)
    # Two families, each with variant + control, plus the plain copy: 5 rows per source that takes both.
    assert summary["kinds"] == {"feasible": 5, "infeasible": 10, "control": 10}
    assert set(summary["families"]) == {"feasible", "remove_input", "conflicting_constraint"}
    # Sources that take every family come first.
    assert {r["pair_id"] for r in index} == {f"local/rich-{i}" for i in range(5)}
    # A sheet for every variant, an index linking them, rows in the dataset format.
    for r in index:
        assert (out / r["sheet"]).exists() and r["sheet"] == f"sheets/{r['variant_id']}.md"
        assert Path(out / "tasks" / r["variant_id"] / "task.toml").exists()
    index_md = (out / "INDEX.md").read_text()
    assert all(f"({r['sheet']})" in index_md for r in index)
    rows = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert len(rows) == len(index) and all(set(r) == set(COLUMNS) for r in rows)
    assert (out / "val.jsonl").read_text() == "" and (out / "test.jsonl").read_text() == ""
    assert {r["kind"] for r in rows} == {"feasible", "control", "infeasible"}
    assert all(r["task_dir"] == str(out.resolve() / "tasks" / r["variant_id"]) for r in rows)


def test_partial_sources_fill_up_when_too_few_take_every_family(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=7)
    summary = _write(cfg)
    assert summary["n_sources"] == 7
    pairs = {json.loads(x)["pair_id"] for x in (cfg.out / "audit.jsonl").read_text().splitlines()}
    assert {f"local/rich-{i}" for i in range(5)} <= pairs  # full sources always chosen first
    assert len([p for p in pairs if "partial" in p]) == 2


def test_unselected_variants_are_removed(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=2)
    _write(cfg)
    dirs = {p.name for p in (cfg.out / "tasks").iterdir()}
    index = [json.loads(x) for x in (cfg.out / "audit.jsonl").read_text().splitlines()]
    assert dirs == {r["variant_id"] for r in index}


def test_sheet_contents(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=1)
    _write(cfg)
    sheets = {p.stem: p.read_text() for p in (cfg.out / "sheets").glob("*.md")}
    assert len(sheets) == 5
    twin = next(t for k, t in sheets.items() if k.endswith("__conflicting_constraint"))
    ctrl = next(t for k, t in sheets.items() if k.endswith("__conflicting_constraint__control"))
    # labels, perturbed object, diffs, validation, verdict line
    for needle in ("## Labels", "expected_blockers: `['conflicting_constraints']`", "expected_targets: `['/home/user/out/", "## Perturbed object",
                   "## Instruction diff", "```diff", "+++ variant/instruction.md", "## Environment diff", "## Oracle diff", "## Tests diff",
                   "## Validation", "Not validated", "Verdict: ", "Paired variant:"):  # fmt: skip
        assert needle in twin, needle
    # the control forbids an input the task names, not a deliverable
    assert "decoy_targets: `['/home/user/data/" in ctrl and "near-miss control of `conflicting_constraint`" in ctrl
    assert "Is the obstacle really harmless" in ctrl and "Is there ANY legitimate route" in twin
    inst_diff = twin.split("## Instruction diff")[1].split("## Environment diff")[0]
    assert "+Per the security review" in inst_diff or "+Do not create" in inst_diff or inst_diff.count("\n+") >= 2
    env_sheet = next(t for k, t in sheets.items() if k.endswith("__remove_input"))
    assert "Instruction unchanged: the obstacle is in the environment." in env_sheet
    assert "rm -f --" in env_sheet.split("## Environment diff")[1]  # the Dockerfile diff shows the deletion
    plain = next(t for k, t in sheets.items() if k.endswith("__feasible"))
    assert "Instruction unchanged" in plain and "Dockerfile unchanged." in plain and "Paired variant" not in plain


def test_sheets_show_validation_results(tmp_path, seeds):
    first = _write(_cfg(tmp_path, seeds, n_sources=1))
    index = [json.loads(x) for x in (tmp_path / "audit" / "audit.jsonl").read_text().splitlines()]
    results = []
    for r in index:
        ok = r["kind"] != "infeasible" or r["family"] != "remove_input"
        results.append(
            ValidationResult(
                r["variant_id"], "", r["pair_id"], r["kind"] != "infeasible", r["family"], ok,
                "oracle fails" if ok else "leaky: oracle still passes", oracle_runs=3, oracle_passes=0 if ok else 3,
                null_check_failed_as_expected=True, leak=not ok,
            )
        )  # fmt: skip
    vpath = tmp_path / "validation.jsonl"
    vpath.write_text("".join(json.dumps(asdict(x)) + "\n" for x in results))
    summary = write_audit(_cfg(tmp_path, seeds, n_sources=1, validation_jsonl=vpath))
    assert summary["n_variants"] == first["n_variants"]
    bad = next(r for r in index if r["family"] == "remove_input" and r["kind"] == "infeasible")
    sheet = (tmp_path / "audit" / "sheets" / f"{bad['variant_id']}.md").read_text()
    assert (
        "**FAILED**: leaky: oracle still passes" in sheet
        and "oracle clean passes: 3 of 3" in sheet
        and "leak (tests pass cleanly" in sheet
    )
    assert "Not validated" not in sheet
    new_index = [json.loads(x) for x in (tmp_path / "audit" / "audit.jsonl").read_text().splitlines()]
    assert {r["variant_id"]: r["validated"] for r in new_index}[bad["variant_id"]] is False
    assert "FAILED" in (tmp_path / "audit" / "INDEX.md").read_text()


def test_stale_validation_is_flagged(tmp_path, seeds):
    _write(_cfg(tmp_path, seeds, n_sources=1))
    index = [json.loads(x) for x in (tmp_path / "audit" / "audit.jsonl").read_text().splitlines()]
    r = index[0]
    stale = ValidationResult(r["variant_id"], "", r["pair_id"], True, None, True, "x", task_hash="0" * 64)
    seed = next(s for s in local_seeds(seeds) if s.pair_id == r["pair_id"])
    v = next(
        v
        for v in generate_variants(seed.task_dir, tmp_path / "scratch", pair_id=seed.pair_id, operators=[])
        if v.variant_id == r["variant_id"]
    )
    assert "WARNING: the task changed" in render_sheet(seed, v, twin=None, result=stale)


def test_rebuild_is_deterministic(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=3)
    _write(cfg)
    first = (cfg.out / "audit.jsonl").read_text()
    sheet = (cfg.out / "INDEX.md").read_text()
    _write(cfg)
    assert (cfg.out / "audit.jsonl").read_text() == first and (cfg.out / "INDEX.md").read_text() == sheet


# ── tally ────────────────────────────────────────────────────────────────────


def test_verdict_parsing(tmp_path):
    p = tmp_path / "s.md"
    for text, expected in (
        ("## Verdict\n\nVerdict: ok\n", "ok"),
        ("Verdict:   WRONG - leak\n", "wrong"),
        ("Verdict: unsure\n", "unsure"),
        ("Verdict: \n", None),
        ("Verdict: maybe\n", None),
        ("no verdict line\n", None),
    ):
        p.write_text(text)
        assert read_verdict(p) == expected, text


def test_tally_counts_verdicts_and_the_residual_leak_rate(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=2)
    _write(cfg)
    index = [json.loads(x) for x in (cfg.out / "audit.jsonl").read_text().splitlines()]
    assert tally(cfg.out)["residual_leak_rate"] is None  # nothing reviewed yet
    infeasible = [r for r in index if r["kind"] == "infeasible"]
    for i, r in enumerate(index):
        sheet = cfg.out / r["sheet"]
        verdict = "wrong" if (r in infeasible[:1]) else "ok" if i % 5 else "unsure"
        sheet.write_text(sheet.read_text().replace("Verdict: \n", f"Verdict: {verdict}\n", 1))
    t = tally(cfg.out)
    assert t["by_kind"]["infeasible"]["wrong"] == 1 and t["by_kind"]["infeasible"]["n"] == len(infeasible)
    assert t["residual_leak_rate"] == pytest.approx(1 / len(infeasible))
    assert t["by_kind"]["control"]["wrong"] == 0 and t["by_kind"]["feasible"]["wrong_rate"] == 0.0
    wrong_family = infeasible[0]["family"]
    row = next(r for r in t["by_family_kind"] if r["kind"] == "infeasible" and r["family"] == wrong_family)
    assert row["wrong"] == 1 and row["pending"] == 0


def test_tally_cli_and_arg_parsing(tmp_path, seeds, capsys):
    cfg = _cfg(tmp_path, seeds, n_sources=1)
    _write(cfg)
    audit.main(["--tally", str(cfg.out)])
    assert json.loads(capsys.readouterr().out)["by_kind"]["infeasible"]["pending"] == 2
    parsed, tally_dir = parse_args(["--out", "x", "--n-sources", "25", "--families", "high_risk", "remove_tool"])
    assert tally_dir is None and parsed.n_sources == 25 and parsed.families == ("high_risk", "remove_tool")
    assert parse_args(["--tally", "y"]) == (None, Path("y"))
    with pytest.raises(ValueError):
        AuditConfig(out=Path("x"), families=("nonsense",))
    with pytest.raises(SystemExit):
        parse_args([])


def test_audit_never_picks_test_sources(tmp_path, seeds):
    """The audit's readers tune operators on what they see: sources in test_ids.txt or split_seeds.json are skipped."""
    ids = tmp_path / "test_ids.txt"
    write_test_ids(ids, ["local/rich-0", "local/rich-1"])
    splits = tmp_path / "split_seeds.json"
    splits.write_text(json.dumps({"local/rich-2": "test", "local/rich-3": "train"}))
    cfg = _cfg(tmp_path, seeds, n_sources=5, test_ids_file=ids, split_seeds=splits)
    _write(cfg)
    index = [json.loads(x) for x in (cfg.out / "audit.jsonl").read_text().splitlines()]
    picked = {r["pair_id"] for r in index}
    assert picked and not picked & {"local/rich-0", "local/rich-1", "local/rich-2"}


def test_audit_skips_sources_that_hash_into_the_test_split(tmp_path, seeds):
    """With no id file at all, the pool-independent hash rule keeps test sources out of the audit."""
    everything = {f"local/rich-{i}" for i in range(5)} | {f"local/partial-{i}" for i in range(3)}
    cfg = _cfg(tmp_path, seeds, n_sources=8, test_bp=5000)
    held = audit.held_out_groups(cfg, sorted(everything))
    assert held and held < everything
    _write(cfg)
    index = [json.loads(x) for x in (cfg.out / "audit.jsonl").read_text().splitlines()]
    assert not {r["pair_id"] for r in index} & held


def test_audit_parses_the_split_flags():
    parsed, _ = parse_args(
        [
            "--out",
            "x",
            "--i-know-no-benchmark",
            "--split-seeds",
            "s.json",
            "--test-ids-file",
            "t.txt",
            "--test-bp",
            "100",
        ]
    )
    assert parsed.i_know_no_benchmark and parsed.split_seeds == Path("s.json")
    assert parsed.test_ids_file == Path("t.txt") and parsed.test_bp == 100


def test_the_default_audit_families_are_the_training_families():
    from honest_agents.tasks.build import DEFAULT_HELDOUT_FAMILIES

    assert set(audit.DEFAULT_FAMILIES) == {"remove_input", "revoke_permission", "high_risk"}
    assert not set(audit.DEFAULT_FAMILIES) & set(DEFAULT_HELDOUT_FAMILIES)
    cfg, _ = parse_args(["--out", "x"])
    assert cfg is not None and cfg.families == audit.DEFAULT_FAMILIES and cfg.show_tests


def test_sheets_show_the_tests_and_the_oracle_command(tmp_path, seeds):
    """Without the tests a human cannot judge whether a variant is truly impossible."""
    cfg = _cfg(tmp_path, seeds, n_sources=1)
    _write(cfg)
    sheets = {p.stem: p.read_text() for p in (cfg.out / "sheets").glob("*.md")}
    twin = next(t for k, t in sheets.items() if k.endswith("__remove_input"))
    section = twin.split("## Tests (what the grader checks)")[1].split("##")[0]
    assert "`test_outputs.py`" in section and "`test.sh`" in section  # the file names
    assert "`test_outputs.py::test_" in section and "assert" in section  # what each test asserts
    assert "Oracle command (`solution/solve.sh`)" in section and "ffmpeg" in section  # the oracle, comments dropped
    assert "#!/bin/bash" not in section
    assert all(len(line) <= 260 for line in section.splitlines() if line.startswith("- `test_outputs"))  # cut at 200
    # a checklist line per variant, kind-specific, ahead of the verdict line
    assert twin.index("- [ ] the tests above cannot pass") < twin.index("Verdict: ")
    assert "- [ ] the removed file is not rebuildable" in twin
    ctrl = next(t for k, t in sheets.items() if k.endswith("__conflicting_constraint__control"))
    assert "- [ ] the obstacle is surmountable" in ctrl
    plain = next(t for k, t in sheets.items() if k.endswith("__feasible"))
    assert "- [ ] the tests above check what the instruction asks" in plain


def test_no_show_tests_hides_the_tests_section(tmp_path, seeds):
    cfg = _cfg(tmp_path, seeds, n_sources=1, show_tests=False)
    _write(cfg)
    assert all("## Tests (what the grader checks)" not in p.read_text() for p in (cfg.out / "sheets").glob("*.md"))
    cfg, _ = parse_args(["--out", "x", "--no-show-tests"])
    assert cfg is not None and not cfg.show_tests


def test_test_assertions_cut_each_test_at_200_characters(tmp_path):
    t = tmp_path / "t"
    (t / "tests").mkdir(parents=True)
    (t / "tests" / "test_x.py").write_text(
        "def test_long():\n    assert 'a' == '" + "a" * 500 + "'\n\ndef helper():\n    pass\n"
    )
    (rows,) = audit.test_assertions(t)
    assert rows[:2] == ("test_x.py", "test_long") and len(rows[2]) == 200
