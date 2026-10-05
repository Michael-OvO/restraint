"""Offline audit of the damage manifest (`spec.preserve_paths`, docs/design.md section 4). Never part of grading or reward.

* **Coverage**: the share of tasks, per family, that declare `preserve_paths` (the manifest measures nothing elsewhere).
* **False positives on oracle runs**: a feasible task whose oracle damaged a preserved path (the validator rejects it, so on
  a built dataset this must be 0) and, over all validated tasks, how many were flagged.
* **Replay estimates** from episodes collected on a VM: replay `solve.sh` from the episode's final state and from a restored
  state. A damaged episode whose final-state replay still passes is a false positive (the damage did not make the task
  harder); an undamaged one that fails from the final state but passes from the restored one is a miss (indirect corruption
  or damage outside `preserve_paths`). The replay itself needs the sandbox and is not run here.

    python -m honest_agents.tasks.damage_audit --tasks-dir out/tasks --validation out/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from honest_agents.spec import HonestTaskSpec


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def coverage(task_dirs: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """`{family: {tasks, with_preserve, share}}` over task directories (`None` = the plain feasible copies)."""
    seen: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for d in task_dirs:
        spec = HonestTaskSpec.from_task_dir(d)
        row = seen[str(spec.family)]
        row[0] += 1
        row[1] += bool(spec.preserve_paths)
    return {fam: {"tasks": n, "with_preserve": k, "share": _share(k, n)} for fam, (n, k) in sorted(seen.items())}


def false_positives(results: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per family, from validation rows (`ValidationResult` as dicts): feasible tasks that declare a manifest and the ones
    whose oracle damaged it. A row without `damage_paths` (written before the field) counts as clean."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        if not r.get("feasible") or not r.get("oracle_runs"):
            continue
        row = out[str(r.get("perturbation"))]
        row[0] += 1
        row[1] += bool(r.get("damage_paths"))
    return {
        fam: {"oracle_validated": n, "oracle_damaged": k, "rate": _share(k, n)} for fam, (n, k) in sorted(out.items())
    }


def replay_rates(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """False-positive and miss rates from replay records `{damaged, final_passes, restored_passes}` (booleans).

    * false positive: damaged and the replay from the final state still passes (the task was not made harder);
    * miss: not damaged, the replay from the final state fails and the one from the restored state passes."""
    damaged = [e for e in episodes if e.get("damaged")]
    clean = [e for e in episodes if not e.get("damaged")]
    fp = sum(bool(e.get("final_passes")) for e in damaged)
    miss = sum(not e.get("final_passes") and bool(e.get("restored_passes")) for e in clean)
    return {
        "n_damaged": len(damaged),
        "n_undamaged": len(clean),
        "false_positive_rate": _share(fp, len(damaged)),
        "miss_rate": _share(miss, len(clean)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks-dir", type=Path, help="directory of task directories (coverage)")
    ap.add_argument(
        "--validation", type=Path, help="validation.jsonl written by tasks.validate (oracle false positives)"
    )
    ap.add_argument("--replays", type=Path, help="jsonl of replay records (false-positive and miss rates)")
    a = ap.parse_args(argv)
    report: dict[str, Any] = {}
    if a.tasks_dir:
        report["coverage"] = coverage(sorted(p.parent for p in a.tasks_dir.glob("*/task.toml")))
    if a.validation:
        report["oracle_false_positives"] = false_positives(
            json.loads(x) for x in a.validation.read_text().splitlines() if x
        )
    if a.replays:
        report["replay"] = replay_rates(json.loads(x) for x in a.replays.read_text().splitlines() if x)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
