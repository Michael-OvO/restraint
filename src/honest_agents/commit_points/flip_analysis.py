"""Flip-test sensitivity from a probe result: P(commit | pass) - P(commit | fail) per point, with a source-clustered CI.

    uv run python -m honest_agents.commit_points.flip_analysis --prompts data/commit/flip_prompts.jsonl \\
        --samples runs/probe/<run>/samples.jsonl [--n-boot 2000] [--seed 0] [--json out.json]

`--prompts` is a `flip_probe` file (its manifest holds the pairs); `--samples` the `samples.jsonl` that
`eval.decision_probe` wrote for it. Each sample's class is the commitment at its point or not (`schema.COMMIT_CLASS`:
claim_done = `report_complete`, blame_env = `report_blocker`, destructive_step = `destroy`).

Per pair: p_pass and p_fail, the commit share of each arm's samples, and d = p_pass - p_fail. A pair counts only when both
arms have samples (the others are counted as `incomplete`). Per point, the estimate is the mean of d over pairs (each
pair weighs the same), with P(commit | pass) and P(commit | fail) as the means of p_pass and p_fail. The 95 % interval
is a percentile bootstrap that resamples **source tasks** (`pair_id`) with replacement (all pairs of a drawn source come
along), so several prefixes or variants of one source do not count as independent evidence. Seeded; <= 2,000 resamples
at L0 (integrity-gate.md). The same readout is given per pair origin (e.g. `own_pass+splice_fail`). Nothing here
samples a model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.commit_points.flip_probe import FlipPair, read_pairs
from honest_agents.commit_points.schema import COMMIT_CLASS, FAIL, PASS, POINTS

N_BOOT = 2000
ALPHA = 0.05


@dataclass(frozen=True)
class PairValue:
    pair_id: str
    point: str
    source: str  # the CI cluster
    origin: str
    p_pass: float
    p_fail: float
    n_pass: int
    n_fail: int

    @property
    def d(self) -> float:
        return self.p_pass - self.p_fail


def pair_values(pairs: Sequence[FlipPair], rows: Sequence[Mapping[str, Any]], counts: Counter[str]) -> list[PairValue]:
    """Join the probe rows (`eval.decision_probe.ProbeRow` dicts) to the pairs by pair id and arm."""
    samples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        if r.get("error") is not None:
            counts["rows_failed"] += 1
            continue
        samples[(str(r["pair_id"]), str(r["arm"]))] += [str(s["cls"]) for s in r.get("samples") or []]
    known = {p.pair_id for p in pairs}
    unknown = sum(1 for r in rows if str(r.get("pair_id")) not in known)
    if unknown:
        counts["rows_unknown_pair"] += unknown
    out: list[PairValue] = []
    for p in pairs:
        act = COMMIT_CLASS[p.point]
        got = {arm: samples.get((p.pair_id, arm), []) for arm in (PASS, FAIL)}
        if not all(got.values()):
            counts[f"incomplete_{p.point}"] += 1
            continue
        rate = {arm: sum(c == act for c in cls) / len(cls) for arm, cls in got.items()}
        out.append(PairValue(p.pair_id, p.point, p.source, p.origin, rate[PASS], rate[FAIL], len(got[PASS]),
                             len(got[FAIL])))  # fmt: skip
    return out


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def cluster_bootstrap(
    values: Sequence[PairValue], stat: str, *, n_boot: int = N_BOOT, seed: int = 0, alpha: float = ALPHA
) -> list[float] | None:
    """Percentile interval of the pair mean of `stat` ("d", "p_pass" or "p_fail") over source-resampled draws. None
    with fewer than 2 sources."""
    by: dict[str, list[float]] = defaultdict(list)
    for v in values:
        by[v.source].append(getattr(v, stat))
    clusters = [(sum(xs), len(xs)) for _, xs in sorted(by.items())]
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    k = len(clusters)
    means = []
    for _ in range(n_boot):
        total = n = 0
        for _ in range(k):
            s, c = clusters[rng.randrange(k)]
            total += s
            n += c
        means.append(total / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


@dataclass
class Readout:
    pairs: int
    sources: int
    samples: int
    p_commit_pass: float | None
    p_commit_fail: float | None
    sensitivity: float | None
    sensitivity_ci95: list[float] | None
    p_commit_pass_ci95: list[float] | None
    p_commit_fail_ci95: list[float] | None


def readout(values: Sequence[PairValue], *, n_boot: int = N_BOOT, seed: int = 0) -> Readout:
    def r(x: float | None) -> float | None:
        return None if x is None else round(x, 4)

    def ci(stat: str) -> list[float] | None:
        return cluster_bootstrap(values, stat, n_boot=n_boot, seed=seed)

    return Readout(
        pairs=len(values),
        sources=len({v.source for v in values}),
        samples=sum(v.n_pass + v.n_fail for v in values),
        p_commit_pass=r(_mean([v.p_pass for v in values])),
        p_commit_fail=r(_mean([v.p_fail for v in values])),
        sensitivity=r(_mean([v.d for v in values])),
        sensitivity_ci95=ci("d"),
        p_commit_pass_ci95=ci("p_pass"),
        p_commit_fail_ci95=ci("p_fail"),
    )


def analyse(
    pairs: Sequence[FlipPair], rows: Sequence[Mapping[str, Any]], *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    values = pair_values(pairs, rows, counts)
    points: dict[str, Any] = {}
    for point in POINTS:
        vs = [v for v in values if v.point == point]
        if not vs and not any(p.point == point for p in pairs):
            continue
        points[point] = {
            "commit_class": COMMIT_CLASS[point],
            **asdict(readout(vs, n_boot=n_boot, seed=seed)),
            "by_origin": {
                s: asdict(readout([v for v in vs if v.origin == s], n_boot=n_boot, seed=seed))
                for s in sorted({v.origin for v in vs})
            },
        }
    return {"n_boot": n_boot, "seed": seed, "alpha": ALPHA, "points": points, "counts": dict(sorted(counts.items()))}


def _f(x: float | None, signed: bool = False) -> str:
    return "n/a" if x is None else (f"{x:+.3f}" if signed else f"{x:.3f}")


def _ci(c: list[float] | None) -> str:
    return "n/a" if c is None else f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def render(result: Mapping[str, Any]) -> str:
    lines = [
        "| point | commit | pairs | sources | P(commit/pass) | P(commit/fail) | sensitivity | 95 % CI (source-clustered) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for point, r in result["points"].items():
        rows = [(point, r)] + [(f"{point} / {s}", x) for s, x in r["by_origin"].items()]
        for label, x in rows:
            lines.append(
                f"| {label} | {r['commit_class']} | {x['pairs']} | {x['sources']} | {_f(x['p_commit_pass'])} | "
                f"{_f(x['p_commit_fail'])} | {_f(x['sensitivity'], True)} | {_ci(x['sensitivity_ci95'])} |"
            )
    lines.append(f"counts {result['counts']}; {result['n_boot']} resamples, seed {result['seed']}")
    return "\n".join(lines)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).expanduser().read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, required=True, help="the flip_probe prompts file (its manifest is read)")
    ap.add_argument("--samples", type=Path, required=True, help="eval.decision_probe samples.jsonl for those prompts")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    result = analyse(read_pairs(args.prompts), read_rows(args.samples), n_boot=args.n_boot, seed=args.seed)
    print(render(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
