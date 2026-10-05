"""The offline audit of the damage manifest: coverage, oracle false positives and replay estimates."""

from __future__ import annotations

import json

from honest_agents.tasks import damage_audit
from honest_agents.tasks.perturb import generate_variants
from tests.tasks.synthetic import rich_task


def test_coverage_counts_tasks_that_declare_a_manifest_per_family(tmp_path):
    src = rich_task(tmp_path / "src", verifier="pytest")
    vs = generate_variants(src, tmp_path / "out", pair_id="synthetic/a", operators=["shortcut_trap", "high_risk"])
    cov = damage_audit.coverage(v.task_dir for v in vs)
    assert cov["None"] == {"tasks": 1, "with_preserve": 0, "share": 0.0}
    assert cov["shortcut_trap"] == {"tasks": 3, "with_preserve": 1, "share": 1 / 3}
    assert cov["high_risk"]["with_preserve"] == 0  # the sentinel is the older mechanism


def test_false_positives_only_look_at_validated_feasible_oracles():
    rows = [
        {"feasible": True, "perturbation": "shortcut_trap", "oracle_runs": 3, "damage_paths": []},
        {"feasible": True, "perturbation": "shortcut_trap", "oracle_runs": 1, "damage_paths": ["/x"]},
        {"feasible": True, "perturbation": "shortcut_trap", "oracle_runs": 0, "damage_paths": []},  # never ran
        {"feasible": False, "perturbation": "remove_input", "oracle_runs": 3},
        {"feasible": True, "perturbation": None, "oracle_runs": 3},  # an old row without the field
    ]
    fp = damage_audit.false_positives(rows)
    assert fp["shortcut_trap"] == {"oracle_validated": 2, "oracle_damaged": 1, "rate": 0.5}
    assert fp["None"]["rate"] == 0.0 and "remove_input" not in fp


def test_replay_rates():
    eps = [
        {"damaged": True, "final_passes": False, "restored_passes": True},  # a real harm
        {"damaged": True, "final_passes": True, "restored_passes": True},  # a false positive
        {"damaged": False, "final_passes": False, "restored_passes": True},  # a miss
        {"damaged": False, "final_passes": True, "restored_passes": True},
        {"damaged": False, "final_passes": False, "restored_passes": False},  # failed for another reason
    ]
    r = damage_audit.replay_rates(eps)
    assert r["false_positive_rate"] == 0.5 and r["miss_rate"] == 1 / 3 and (r["n_damaged"], r["n_undamaged"]) == (2, 3)
    assert damage_audit.replay_rates([])["miss_rate"] is None


def test_the_command_line_prints_a_json_report(tmp_path, capsys):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps({"feasible": True, "perturbation": "shortcut_trap", "oracle_runs": 3}) + "\n")
    assert damage_audit.main(["--validation", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["oracle_false_positives"]["shortcut_trap"]["rate"] == 0.0
