"""`sft.destructive_splice` on synthetic surface variants: where a state is cut, the pairs, labels, quota and the
excluded sources (the final decision probe's included). No Docker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honest_agents.bank import BankRow
from honest_agents.commit_points import coverage
from honest_agents.commit_points.schema import read_manifest
from honest_agents.sft import commit_checks as cc
from honest_agents.sft import destructive_splice as ds
from tests.decision_fixtures import OWN_REPORT, TARGET, SurfaceFixture, surface_site
from tests.sft.helpers import make_row, write_bank

MAIN = [("ls /app", "in.txt"), ("cp /app/in.txt /app/out.txt", "")]
READS = [(f"ls -la {TARGET}", "chunk_1.dat\nNOTES.txt"), (f"cat {TARGET}/NOTES.txt", "scratch outputs")]
DELETE = [(f"rm -rf {TARGET}", "")]
AFTER = [("cat /app/out.txt", "hello")]


def surface_control_row(
    source: str, turns: list[tuple[str, str]], *, sample: int = 0, surface: str = "readme"
) -> BankRow:
    """G2's own verified-success control episode of a surface variant, tagged like an it16b bank row."""
    variant = f"{source}__high_risk__{surface}__control"
    targets = list(surface_site(surface).expected_targets)
    row = make_row(
        variant,
        passed=True,
        original=dict(OWN_REPORT),
        turns=turns,
        is_control=True,
        spec={"decoy_targets": targets, "pair_id": source, "is_control": True, "feasible": True},
        episode_id=f"{source}-{surface}-c{sample}",
    )
    task = {**row.task, "kind": "control", "family": "high_risk", "perturbation": "high_risk", "pair_id": source,
            "variant_id": variant, "task_dir": f"/root/restraint/data/decision/surfaces/tasks/{variant}"}  # fmt: skip
    return row.model_copy(update={"task": task})


@pytest.fixture
def fx(tmp_path: Path) -> SurfaceFixture:
    f = SurfaceFixture(tmp_path, n_pairs=3, surfaces=("readme",))
    (f.data_dir / "decision").mkdir()
    (f.data_dir / "decision" / "probe_prompts_surfaces.jsonl").write_text(
        json.dumps({"pair_id": "probe/task_p"}) + "\n"
    )
    return f


def _cfg(fx: SurfaceFixture, bank: Path, out: Path, **kw) -> ds.Config:
    return ds.Config([str(bank)], str(fx.tasks), [str(fx.outputs)], str(out), str(fx.data_dir), **kw)


def test_a_state_is_cut_at_the_deletion_or_before_the_own_reads_and_the_reads_are_dropped(fx: SurfaceFixture):
    source = fx.base.pairs[0]
    state, cut = ds.surface_state(surface_control_row(source, MAIN + DELETE + AFTER), TARGET)
    assert cut == "deletion" and state.commands[state.destructive_index] == f"rm -rf {TARGET}"
    state, cut = ds.surface_state(surface_control_row(source, MAIN + READS + DELETE + AFTER), TARGET)
    assert cut == "first_read"
    assert state.commands == [c for c, _ in MAIN + DELETE + AFTER] and state.destructive_index == len(MAIN)
    assert not any(TARGET in str(m.get("content")) for m in state.prefix_messages())  # the prefix never saw the target
    assert state.decision_and_tail()[0].commands() == [f"rm -rf {TARGET}"]


def test_a_state_whose_reads_are_mixed_with_other_work_is_dropped(fx: SurfaceFixture):
    source = fx.base.pairs[0]
    mutating = MAIN + READS[:1] + [("cp /app/in.txt /app/b.txt", "")] + READS[1:] + DELETE
    assert ds.surface_state(surface_control_row(source, mutating), TARGET) == ds.DROP_MUTATION_BETWEEN
    assert ds.surface_state(surface_control_row(source, MAIN + AFTER), TARGET) == ds.DROP_NO_DELETION


def test_build_writes_matched_pairs_that_grade_as_licensed_and_pass_coverage(fx: SurfaceFixture, tmp_path: Path):
    rows = [surface_control_row(s, MAIN + READS + DELETE + AFTER, sample=k) for s in fx.base.pairs for k in range(3)]
    bank = write_bank(tmp_path / "bank", rows)
    summary = ds.build(_cfg(fx, bank, tmp_path / "out", cap_per_task=2))
    assert summary["pairs_before_cap"] == 9 and summary["pairs"] == 6 and summary["dropped"][ds.DROP_CAP] == 3
    assert summary["labels"] == {"checked": 9, "failed": 0} and summary["cuts"] == {"first_read": 6}
    examples = [json.loads(line) for line in (tmp_path / "out" / "examples.jsonl").read_text().splitlines()]
    assert {e["kind"] for e in examples} == {cc.KIND}
    by_episode: dict[str, list[dict]] = {}
    for e in examples:
        by_episode.setdefault(e["meta"]["source_episode"], []).append(e)
    for pair in by_episode.values():
        control, twin = sorted(pair, key=lambda e: e["meta"]["outcome"] != "pass")
        assert control["meta"]["outcome"] == "pass" and twin["meta"]["outcome"] == "fail"
        n = control["meta"]["n_prefix_messages"]
        assert control["messages"][:n] == twin["messages"][:n]  # same prefix, then the check outputs differ
        assert twin["meta"]["task"] == cc.twin_id(control["meta"]["task"])
        last = twin["messages"][-1]["tool_calls"][-1]["function"]["arguments"]
        assert last["status"] == "incomplete" and last["blocker_category"] == "risk_discovered"
    m = read_manifest(tmp_path / "out" / cc.MANIFEST_FILE)
    r = coverage.coverage(m, coverage.excluded_sources(fx.data_dir))
    # three sources hold a third each: the source share is the merge's, over the whole build
    assert r.failures() == {"share": 3}
    assert {e.hazard for e in m.examples} <= {"copy", "depended_on", "in_use", "protected", "unfinished", "other"}
    assert all(e.committed == (e.outcome == "pass") for e in m.examples)


def test_a_source_of_the_final_decision_probe_is_refused_or_dropped(fx: SurfaceFixture, tmp_path: Path):
    probe = fx.base.pairs[0]
    (fx.data_dir / "decision" / "probe_prompts_surfaces.jsonl").write_text(json.dumps({"pair_id": probe}) + "\n")
    rows = [surface_control_row(s, MAIN + READS + DELETE + AFTER) for s in fx.base.pairs]
    bank = write_bank(tmp_path / "bank", rows)
    with pytest.raises(cc.ExcludedSourceError, match=probe):
        ds.build(_cfg(fx, bank, tmp_path / "out"))
    summary = ds.build(_cfg(fx, bank, tmp_path / "out", drop_excluded=True))
    assert summary["dropped"][cc.DROP_EXCLUDED] == 1 and summary["pairs"] == 2
    examples = (tmp_path / "out" / "examples.jsonl").read_text()
    assert probe not in examples
