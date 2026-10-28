"""Iteration 17 scoping and power for calibrated report confidence (docs/iteration-17-design.md), from archived evals.

    uv run python -m honest_agents.analysis.calibration scope --run NAME=RUN_DIR [--run NAME=RUN_DIR ...] \\
        [--tasks-root data/tasks] [--json out.json]
    uv run python -m honest_agents.analysis.calibration power --run NAME=RUN_DIR [--samples 2 4] [--n-boot 2000] \\
        [--json out.json]

**Claims.** The unit is a feasible claim (status `complete`, hidden tests run): the near misses whose confidence the
iteration trains. y = 1 when the claim fails its tests (the event a calibrated confidence should flag).

**scope:** how well each predictor of failure separates the claims (AUROC, failure as the positive class):
- `confidence`: the reported confidence (inverted: low confidence predicts failure);
- `features`: a ridge logistic model on report-time transcript features (`FEATURES`: commands, failed commands,
  commands after the last change, mutations, instruction length and format words, ...), out-of-fold over 5 folds
  grouped by source, so no source is scored by a model that saw it; each feature's own AUROC is reported too;
- `other_sample`: the same run's other samples of the task failed (S >= 2), a task-difficulty signal the model could
  learn; `other_runs`: the other runs' claims on the same task failed, an upper reference.

**power:** the source-cluster bootstrap SE of the claims AUROC of the `features` predictor (a stand-in for a trained
confidence with real spread; the start model's own is near-constant), and from it the minimum detectable difference
of two arms' AUROCs at 80 % power with the pre-registered one-sided 0.05 bar (90 % two-sided CI, LB > 0), at the
run's S and at other S. The SE is scaled by (S_run / S)^e with e measured on the run itself (the SE from sample 0
alone against all samples, e = log2 of their ratio): failure is mostly a property of the task, so more samples of the
same tasks shrink the SE far less than sqrt(S) (e = 0.5); measured on the it16 select runs, e = 0.05-0.34. Only more
sources buy power.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from honest_agents.eval.metrics.calibration import auroc
from honest_agents.mutation import is_mutating

FORMAT_WORDS = re.compile(
    r"exact|exactly|must|only|format|no extra|trailing|newline|sorted|order|precisely|case[- ]sensitive", re.IGNORECASE
)
READS_AFTER = re.compile(r"\b(cat|head|tail|wc|diff)\b")
FEATURES = (
    "n_cmds",
    "n_fail_rc",
    "frac_fail_rc",
    "fail_after_last_change",
    "n_after_last_change",
    "n_mutations",
    "last_change_frac",
    "near_budget",
    "instr_kchars",
    "instr_format_words",
    "n_completed_items",
    "reads_after_last_change",
)
N_FOLDS = 5
RIDGE = 1.0
Z_POWER = 0.8416  # 80 % power
Z_ALPHA = 1.6449  # one-sided 0.05 (the 90 % two-sided CI's lower bound)


@dataclass(frozen=True)
class Claim:
    run: str
    task: str
    source: str
    sample: int
    failed: int
    confidence: float | None
    features: tuple[float, ...]


def _episodes(run_dir: Path) -> list[dict[str, Any]]:
    path = next(p for p in (run_dir / "episodes.jsonl.gz", run_dir / "episodes.jsonl") if p.is_file())
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _instruction(task_dir: str, roots: Sequence[Path]) -> str:
    for root in roots:
        p = root / Path(task_dir).name / "instruction.md"
        if p.is_file():
            return p.read_text()
    return ""


def features_of(e: dict[str, Any], instruction: str) -> tuple[float, ...]:
    o = e["outcome"]
    r = o.get("report") or {}
    hist = [x for x in (e.get("extra") or {}).get("history") or [] if isinstance(x, dict)]
    cmds = [str(x.get("command") or "") for x in hist]
    rcs = [x.get("rc") for x in hist]
    last = max((i for i, c in enumerate(cmds) if is_mutating(c)), default=-1)
    failed = [x not in (0, None) for x in rcs]
    n = len(cmds)
    return (
        float(n),
        float(sum(failed)),
        sum(failed) / max(n, 1),
        float(sum(failed[last + 1 :])),
        float(n - last - 1),
        float(sum(is_mutating(c) for c in cmds)),
        (last + 1) / max(n, 1),
        float(n >= 25),
        len(instruction) / 1000,
        float(len(FORMAT_WORDS.findall(instruction))),
        float(len(r.get("completed") or [])),
        float(sum(1 for c in cmds[last + 1 :] if READS_AFTER.search(c))),
    )


def load_claims(runs: dict[str, Path], roots: Sequence[Path]) -> list[Claim]:
    out = []
    for name, run_dir in runs.items():
        for e in _episodes(run_dir):
            o = e["outcome"]
            r = o.get("report") or {}
            if not o.get("feasible") or r.get("status") != "complete" or o.get("tests_passed") is None:
                continue
            if e.get("infra_error") or o.get("masked_reason"):
                continue
            conf = r.get("confidence")
            out.append(Claim(name, e["task_id"], e.get("pair_id") or e.get("source_task") or e["task_id"],
                             int(e.get("sample") or 0), int(not o["tests_passed"]),
                             None if conf is None else float(conf),
                             features_of(e, _instruction(e.get("task_dir") or "", roots))))  # fmt: skip
    return out


def _fold(source: str) -> int:
    return int(hashlib.sha256(source.encode()).hexdigest()[:8], 16) % N_FOLDS


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ridge-penalised logistic regression by Newton steps (standardised inputs)."""
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        g_w = x.T @ (p - y) + RIDGE * w
        g_b = float(np.sum(p - y))
        h = (x.T * (p * (1 - p))) @ x + RIDGE * np.eye(x.shape[1])
        step = np.linalg.solve(h, g_w)
        w -= step
        b -= g_b / max(float(np.sum(p * (1 - p))), 1e-9)
    return w, b


def feature_scores(claims: Sequence[Claim]) -> np.ndarray:
    """Out-of-fold failure scores of the features model, folds by source."""
    x = np.array([c.features for c in claims], dtype=float)
    y = np.array([c.failed for c in claims], dtype=float)
    folds = np.array([_fold(c.source) for c in claims])
    scores = np.zeros(len(claims))
    for k in range(N_FOLDS):
        test = folds == k
        if not test.any() or test.all():
            continue
        mu, sd = x[~test].mean(0), x[~test].std(0) + 1e-9
        w, b = _fit((x[~test] - mu) / sd, y[~test])
        scores[test] = ((x[test] - mu) / sd) @ w + b
    return scores


def _auc(scores: Sequence[float], failed: Sequence[int]) -> float | None:
    return auroc(list(scores), [bool(f) for f in failed])


def scope(claims: Sequence[Claim]) -> dict[str, Any]:
    out: dict[str, Any] = {"claims": len(claims), "failures": sum(c.failed for c in claims), "by_run": {}}
    by_task_run: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for c in claims:
        by_task_run[(c.run, c.task)].append(c)
    for run in sorted({c.run for c in claims}):
        cs = [c for c in claims if c.run == run]
        stated = [c for c in cs if c.confidence is not None]
        other_s = [(sum(o.failed for o in by_task_run[(run, c.task)] if o is not c), c.failed)
                   for c in cs if len(by_task_run[(run, c.task)]) > 1]  # fmt: skip
        other_r = []
        for c in cs:
            others = [o.failed for (r, t), group in by_task_run.items() if t == c.task and r != run for o in group]
            if others:
                other_r.append((float(np.mean(others)), c.failed))
        out["by_run"][run] = {
            "claims": len(cs),
            "failures": sum(c.failed for c in cs),
            "confidence": _auc([-c.confidence for c in stated], [c.failed for c in stated]) if stated else None,
            "confidence_coverage": len(stated) / len(cs) if cs else None,
            "other_sample": _auc([s for s, _ in other_s], [f for _, f in other_s]) if other_s else None,
            "other_runs": _auc([s for s, _ in other_r], [f for _, f in other_r]) if other_r else None,
        }
    s = feature_scores(claims)
    out["features_oof"] = _auc(s, [c.failed for c in claims])
    out["features_each"] = {
        name: _auc([c.features[i] for c in claims], [c.failed for c in claims]) for i, name in enumerate(FEATURES)
    }
    return out


def bootstrap_se(scores: np.ndarray, claims: Sequence[Claim], n_boot: int, seed: int = 0) -> float | None:
    """Source-cluster bootstrap SE of the AUROC of `scores` (failure positive)."""
    sources = sorted({c.source for c in claims})
    idx = defaultdict(list)
    for i, c in enumerate(claims):
        idx[c.source].append(i)
    rng = np.random.default_rng(seed)
    y = np.array([c.failed for c in claims])
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(sources), size=len(sources), replace=True)
        rows = [i for k in pick for i in idx[sources[k]]]
        a = _auc(scores[rows], y[rows])
        if a is not None:
            vals.append(a)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else None


def power(claims: Sequence[Claim], samples: Sequence[int], n_boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in sorted({c.run for c in claims}):
        cs = [c for c in claims if c.run == run]
        s_run = max(c.sample for c in cs) + 1
        scores = feature_scores(cs)
        se = bootstrap_se(scores, cs, n_boot)
        if se is None:
            out[run] = None
            continue
        exponent = 0.5
        se_one = None
        if s_run >= 2:
            one = [c for c in cs if c.sample == 0]
            se_one = bootstrap_se(feature_scores(one), one, n_boot)
            if se_one:
                exponent = max(0.0, math.log(se_one / se) / math.log(s_run))
        rows = {}
        for s in samples:
            se_s = se * (s_run / s) ** exponent
            se_diff = math.sqrt(2) * se_s  # two arms, independent samples on the same tasks (conservative)
            rows[f"S{s}"] = {"se_auroc": se_s, "se_difference": se_diff, "mde": (Z_ALPHA + Z_POWER) * se_diff}
        out[run] = {"claims": len(cs), "failures": sum(c.failed for c in cs), "sources": len({c.source for c in cs}),
                    "samples_run": s_run, "features_auroc": _auc(scores, [c.failed for c in cs]),
                    "se_sample0": se_one, "scaling_exponent": exponent, "designs": rows}  # fmt: skip
    return out


def _runs(pairs: Sequence[str]) -> dict[str, Path]:
    out = {}
    for p in pairs:
        name, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"--run takes NAME=DIR, got {p!r}")
        out[name] = Path(path).expanduser()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m honest_agents.analysis.calibration", description=__doc__.split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scope", "power"):
        p = sub.add_parser(name)
        p.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (a run_eval directory)")
        p.add_argument("--tasks-root", action="append", default=None, help="task dirs (instruction.md); repeatable")
        p.add_argument("--json", type=Path, default=None)
        if name == "power":
            p.add_argument("--samples", type=int, nargs="+", default=[2, 4])
            p.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    roots = [Path(r) for r in (args.tasks_root or ["data/tasks"])]
    claims = load_claims(_runs(args.run), roots)
    result = scope(claims) if args.cmd == "scope" else power(claims, args.samples, args.n_boot)
    text = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
