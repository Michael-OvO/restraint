"""analysis.apply_regrades: a re-graded copy of a run changes exactly the replayable regrades' grades."""

from __future__ import annotations

import json

from honest_agents.analysis import apply_regrades as ar


def test_only_replayable_regrades_change_the_grade(tmp_path):
    rows = [{"episode_id": e, "outcome": {"tests_passed": False, "pass_fraction": 0.8}} for e in ("a", "b", "c")]
    reg = tmp_path / "r.jsonl"
    reg.write_text("\n".join(json.dumps(x) for x in (
        {"episode_id": "a", "status": "replayable", "tests_passed": True, "case": "success_claimed", "archived_case": "fabrication"},
        {"episode_id": "b", "status": "not_replayable", "tests_passed": True},
        {"episode_id": "z", "status": "replayable", "tests_passed": True},
    )))  # fmt: skip
    out, counts = ar.apply(rows, ar.read_regrades([reg]))
    assert out[0]["outcome"] == {"tests_passed": True, "pass_fraction": 1.0}
    assert out[0]["extra"]["regraded"]["archived_case"] == "fabrication"
    assert out[1] == rows[1] and out[2] == rows[2] and rows[0]["outcome"]["tests_passed"] is False  # input untouched
    assert counts == {"applied": 1, "changed": 1, "not_replayable": 1, "regrades_without_episode": 1}
