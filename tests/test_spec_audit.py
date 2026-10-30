"""The spec audit: the near-miss rule, the review rows and the S12 check (tasks.spec_audit)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from honest_agents.tasks import spec_audit as sa

LEDGER = "| DEF-70 | data | closed | x |\n\n### DEF-70: a task defect\n\n- **Status:** closed\n"
SRC = "endless-terminals/task_000000_aaaaaaaa"


def _episode(src_hex: str, passed: bool, fraction: float, *, feasible: bool = True, infra: bool = False) -> dict:
    return {"task_id": f"endless-terminals__task_000000_{src_hex}__feasible__s0__x",
            "outcome": {"feasible": feasible, "tests_passed": passed, "pass_fraction": fraction},
            "extra": {"infra_error": infra}}  # fmt: skip


def _archive(tmp_path: Path, runs: dict[str, list[dict]]) -> Path:
    for name, eps in runs.items():
        d = tmp_path / "it" / "eval" / name
        d.mkdir(parents=True)
        with gzip.open(d / "episodes.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(e) + "\n" for e in eps)
    return tmp_path


def test_a_source_nobody_passes_with_near_miss_failures_is_a_candidate(tmp_path) -> None:
    near = [_episode("aaaaaaaa", False, 0.8) for _ in range(8)]
    hard = [_episode("bbbbbbbb", False, 0.2) for _ in range(8)]  # fails grossly: skill, not the spec
    easy = [_episode("cccccccc", i < 5, 0.8) for i in range(8)]  # passed often
    few = [_episode("dddddddd", False, 0.9) for _ in range(3)]  # too few episodes to say
    doc = sa.scan(_archive(tmp_path, {"r1": near[:4] + hard + easy, "r2": near[4:] + few}))
    assert [c["source"] for c in doc["candidates"]] == [SRC]
    assert doc["eval_runs"] == 2 and doc["candidates"][0]["runs"] == 2


def test_infeasible_ungraded_and_infra_episodes_do_not_count(tmp_path) -> None:
    eps = [_episode("aaaaaaaa", False, 0.8) for _ in range(7)]
    eps += [_episode("aaaaaaaa", False, 0.8, feasible=False), _episode("aaaaaaaa", False, 0.8, infra=True)]
    eps.append({"task_id": "endless-terminals__task_000000_aaaaaaaa__feasible", "outcome": {"feasible": True}})
    assert sa.scan(_archive(tmp_path, {"r": eps}))["candidates"] == []


def test_check_needs_a_review_for_every_candidate_and_a_ledger_id_for_every_defect() -> None:
    cands = {"eval_runs": 3, "candidates": [{"source": SRC}]}
    assert sa.check(cands, [], LEDGER) == [f"candidate {SRC} has no review"]
    good = {"source": SRC, "verdict": "defect", "kind": "contradiction", "evidence": "instruction l3 vs test l9",
            "def_id": "DEF-70", "second_review": "confirmed"}  # fmt: skip
    assert sa.check(cands, [good], LEDGER) == []
    assert any("not in the ledger" in p for p in sa.check(cands, [{**good, "def_id": "DEF-99"}], LEDGER))
    assert any("needs `def_id`" in p for p in sa.check(cands, [{**good, "def_id": None}], LEDGER))
    assert any("kind" in p for p in sa.check(cands, [{**good, "kind": "typo"}], LEDGER))
    assert any("second_review" in p for p in sa.check(cands, [{**good, "second_review": None}], LEDGER))
    assert any("no evidence" in p for p in sa.check(cands, [{**good, "evidence": ""}], LEDGER))
    ok = {"source": SRC, "verdict": "ok", "evidence": "the test asserts what the instruction states"}
    assert sa.check(cands, [ok], LEDGER) == []
    assert any("second review" in p for p in sa.check(cands, [ok, ok], LEDGER))


def test_a_scan_older_than_the_archive_fails() -> None:
    cands = {"eval_runs": 3, "candidates": []}
    assert sa.check(cands, [], LEDGER, archive_runs=3) == []
    assert any("re-run" in p for p in sa.check(cands, [], LEDGER, archive_runs=4))


def test_the_committed_audit_is_consistent() -> None:
    """S12 on the committed files, without the archive (that part runs in the sweep)."""
    cands = json.loads(sa.CANDIDATES.read_text())
    assert sa.check(cands, sa.load_reviews(sa.REVIEWS), sa.LEDGER.read_text()) == []
    assert cands["rule"] == {"min_episodes": 8, "max_pass_rate": 0.2, "min_median_fail_fraction": 0.6}


def test_training_refuses_a_defective_source_unless_allowed(tmp_path) -> None:
    import pytest

    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps({"source": SRC, "verdict": "defect", "kind": "contradiction", "evidence": "x",
                                   "def_id": "DEF-70"}) + "\n")  # fmt: skip
    ids = ["endless-terminals__task_000000_aaaaaaaa__high_risk__control", "endless-terminals/task_000000_bbbbbbbb"]
    with pytest.raises(ValueError, match="spec defects"):
        sa.refuse_defective(ids, where="rows", reviews=reviews)
    assert sa.refuse_defective(ids, where="rows", allow=True, reviews=reviews) == {SRC}
    assert sa.refuse_defective(ids[1:], where="rows", reviews=reviews) == set()


def test_the_grpo_row_check_applies_the_guard(monkeypatch) -> None:
    import pytest

    from honest_agents.train import rows as tr
    from honest_agents.train.config import TrainRunConfig

    monkeypatch.setattr(sa, "defective_sources", lambda reviews=sa.REVIEWS: {SRC})
    cfg = TrainRunConfig()
    cfg.data.check_tasks = False
    row = {"split": "train", "task_dir": "/x/endless-terminals__task_000000_aaaaaaaa__feasible", "pair_id": SRC}
    with pytest.raises(ValueError, match="spec defects"):
        tr.check_rows([row], cfg, role="train")
    cfg.data.allow_spec_defects = True
    assert tr.check_rows([row], cfg, role="train")["rows"] == 1
    cfg.data.allow_spec_defects = False
    assert tr.check_rows([{**row, "split": "val"}], cfg, role="validation")["rows"] == 1  # the probe: any source
