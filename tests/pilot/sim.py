"""A one-parameter bandit stand-in for the policy, trained with the pilot's real reward and TRL's advantage formulas.

The policy claims a feasible prompt with probability sigmoid((q - t) / s), where q is the prompt's true pass probability
(a calibrated belief) and t a learned threshold. REINFORCE with the group advantage (`curves.advantages`, which matches
TRL) moves t. With `scale_rewards: none` it settles near (1 + alpha) / 2; with `group` near 0.5 whatever alpha is.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from honest_agents.pilot import curves
from honest_agents.reward import RewardConfig, compute_reward
from tests.pilot.helpers import episode_record, outcome

ALPHAS = (0.0, 0.25, 0.5, 0.75)


def _reward(cfg: RewardConfig, claimed: bool, passed: bool) -> float:
    status = "complete" if claimed else "incomplete"
    return compute_reward(outcome(True, {"status": status}, tests_passed=passed), cfg).total


def simulate_arm(
    alpha: float,
    scale: str,
    rounds: int = 120,
    prompts: int = 8,
    g: int = 8,
    sharpness: float = 0.08,
    lr: float = 0.02,
    seed: int = 0,
    learn: bool = True,
    start: float = 0.0,
) -> tuple[float, list[dict[str, Any]]]:
    rng = random.Random(seed)
    cfg = RewardConfig(alpha=alpha)
    t, records = start, []
    for rnd in range(rounds):
        rewards, grads = [], []
        for p in range(prompts):
            q = rng.random()
            passed = rng.random() < q
            pc = 1 / (1 + math.exp(-(q - t) / sharpness))
            for _ in range(g):
                claimed = rng.random() < pc
                rewards.append(_reward(cfg, claimed, passed))
                grads.append(-(1 - pc) / sharpness if claimed else pc / sharpness)
                records.append(
                    episode_record(
                        round=rnd,
                        pilot_id=f"{rnd}-{p}",
                        passed=passed,
                        claimed=claimed,
                        declined=not claimed,
                        reward=rewards[-1],
                    )
                )
        if learn:
            adv = curves.advantages(rewards, g, scale)
            t += lr * sum(a * b for a, b in zip(adv, grads, strict=True)) / len(adv)
    return t, records


def write_arm(root: Path, alpha: float, scale: str, records: list[dict[str, Any]], seed: int = 42) -> Path:
    d = root / f"a{alpha:g}_{scale}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "arm.json").write_text(json.dumps({"name": d.name, "alpha": alpha, "scale_rewards": scale, "seed": seed}))
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return d


# ── stop-rule power simulation (no training: the threshold follows a schedule) ─────────────────────────────────────


def schedule_arm(
    t_end: float,
    rounds: int = 40,
    prompts: int = 8,
    g: int = 8,
    sharpness: float = 0.08,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """One arm whose claim threshold moves linearly from 0 (claims everything) to `t_end` over the first half of the
    rounds. Same policy family as `simulate_arm`, without the optimiser, so thousands of arms are cheap."""
    import numpy as np

    rng = np.random.default_rng(seed)
    records = []
    for rnd in range(rounds):
        t = t_end * min(1.0, rnd / max(1, rounds // 2))
        q = rng.random(prompts)
        passed = rng.random(prompts) < q
        pc = 1.0 / (1.0 + np.exp(-(q - t) / sharpness))
        claimed = rng.random((prompts, g)) < pc[:, None]
        for p in range(prompts):
            for c in claimed[p]:
                records.append(
                    episode_record(
                        round=rnd,
                        pilot_id=f"{rnd}-{p}",
                        passed=bool(passed[p]),
                        claimed=bool(c),
                        declined=not c,
                    )
                )
    return records


def schedule_sweep(
    effect: float,
    seed_sd: float = 0.0,
    seeds: int = 3,
    alphas: tuple[float, ...] = ALPHAS,
    alpha_dependent: bool = True,
    rng_seed: int = 0,
    **arm: Any,
) -> list[tuple[float, int, list[dict[str, Any]]]]:
    """(alpha, seed, records) for a sweep whose final threshold is `effect` times the nominal (1 + alpha) / 2 (`effect`
    0: nothing is learned; 1: the full effect), or `effect` times 0.5 whatever alpha is (`alpha_dependent` False: the
    policy learns to claim less but alpha does nothing). `seed_sd` is the SD of a per-run offset of the final threshold,
    the training-seed noise the stop rule must survive."""
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    out = []
    for alpha in alphas:
        nominal = (1 + alpha) / 2 if alpha_dependent else 0.5
        for s in range(seeds):
            t_end = float(np.clip(effect * nominal + rng.normal(0.0, seed_sd), 0.0, 1.0))
            out.append((alpha, s, schedule_arm(t_end, seed=int(rng.integers(1 << 31)), **arm)))
    return out


def write_sweep(root: Path, sweep: list[tuple[float, int, list[dict[str, Any]]]]) -> None:
    for alpha, s, records in sweep:
        write_arm(root, alpha, "none", records, seed=100 + s)
