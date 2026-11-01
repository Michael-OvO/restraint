"""Write a re-graded copy of a run_eval directory: each episode with a replayable `replay_episodes --regrade` row gets
that row's grade (DEF-74: the archive re-scored after a grader fix, "re-scored, post-fix").

    uv run python -m honest_agents.analysis.apply_regrades RUN_DIR REGRADES.jsonl [...] --out OUT_DIR

The copy has the run's `config.json` and an `episodes.jsonl` in which a re-graded episode's `outcome.tests_passed`
is the regrade's (and `pass_fraction` 1.0 when it now passes), with `extra.regraded` = {archived tests_passed, case,
new case, source file}. A regrade row that is not `replayable` (the replay diverged, the sandbox failed, no task dir)
changes nothing and is counted. The reward is recomputed from the outcome whenever a run is read, so nothing else
needs rewriting.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPLAYABLE = "replayable"


def read_regrades(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[str(row["episode_id"])] = {**row, "_file": str(p)}
    return out


def apply(rows: Sequence[dict[str, Any]], regrades: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    """The re-graded rows and counts (applied, changed, not replayable, regrade rows for no episode of the run)."""
    seen: set[str] = set()
    counts = {"applied": 0, "changed": 0, "not_replayable": 0}
    out = []
    for r in rows:
        rg = regrades.get(str(r.get("episode_id")))
        if rg is None:
            out.append(r)
            continue
        seen.add(str(r["episode_id"]))
        if rg.get("status") != REPLAYABLE:
            counts["not_replayable"] += 1
            out.append(r)
            continue
        new = json.loads(json.dumps(r))
        o = new["outcome"]
        before = o.get("tests_passed")
        o["tests_passed"] = rg.get("tests_passed")
        if rg.get("tests_passed"):
            o["pass_fraction"] = 1.0
        new.setdefault("extra", {})["regraded"] = {"archived_tests_passed": before, "archived_case": rg.get("archived_case"),
                                                   "case": rg.get("case"), "file": rg["_file"]}  # fmt: skip
        counts["applied"] += 1
        counts["changed"] += int(before != rg.get("tests_passed"))
        out.append(new)
    counts["regrades_without_episode"] = len(set(regrades) - seen)
    return out, counts


def _episodes(run_dir: Path) -> Path:
    return next(p for p in (run_dir / "episodes.jsonl.gz", run_dir / "episodes.jsonl") if p.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.analysis.apply_regrades", description=__doc__.split("\n")[0]
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("regrades", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    src = _episodes(args.run_dir)
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    new, counts = apply(rows, read_regrades(args.regrades))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "config.json").is_file():
        shutil.copy2(args.run_dir / "config.json", args.out / "config.json")
    with open(args.out / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in new)
    print(json.dumps({"run": str(args.run_dir), "episodes": len(new), **counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
