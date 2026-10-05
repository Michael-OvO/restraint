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
