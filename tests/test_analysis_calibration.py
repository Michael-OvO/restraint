"""`analysis.calibration` on a synthetic run: claim loading, the out-of-fold feature model and the power arithmetic."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from honest_agents.analysis import calibration as cal


def _episode(task: str, source: str, sample: int, *, failed: bool, n_cmds: int, conf: float | None = 1.0) -> dict:
    report = {"status": "complete", "completed": ["x"], **({} if conf is None else {"confidence": conf})}
    hist = [{"command": "ls", "rc": 0}] * n_cmds
    return {"task_id": task, "pair_id": source, "sample": sample, "task_dir": f"/x/{task}",
            "outcome": {"feasible": True, "tests_passed": not failed, "report": report},
            "extra": {"history": hist}}  # fmt: skip


def _run(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    with gzip.open(d / "episodes.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return d


def test_claims_and_a_feature_that_separates_failures(tmp_path: Path):
    rows = []
    for i in range(40):
        failed = i % 2 == 0
        for s in range(2):
            rows.append(_episode(f"t{i}", f"src/{i}", s, failed=failed, n_cmds=20 if failed else 5))
    rows.append({**_episode("tx", "src/x", 0, failed=False, n_cmds=3), "outcome": {"feasible": False}})  # not a claim
    claims = cal.load_claims({"r": _run(tmp_path, rows)}, [tmp_path])
    assert len(claims) == 80 and sum(c.failed for c in claims) == 40
    out = cal.scope(claims)
    assert out["features_each"]["n_cmds"] == 1.0 and out["features_oof"] > 0.9
    assert out["by_run"]["r"]["confidence"] == 0.5 and out["by_run"]["r"]["other_sample"] == 1.0
    p = cal.power(claims, [2, 8], n_boot=200)["r"]
    s2, s8 = p["designs"]["S2"], p["designs"]["S8"]
    assert math.isclose(s8["se_auroc"], s2["se_auroc"] / 2) and math.isclose(
        s2["se_difference"], s2["se_auroc"] * 2**0.5
    )
