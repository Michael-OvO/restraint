"""`train.confidence`: the config's fixed rules, the held-out split by source and the reward function. The CPU training
smoke run is not here (it loads a model); see the module docstring."""

from __future__ import annotations

from pathlib import Path

import pytest

from honest_agents.train import confidence as tc


def _cfg(tmp_path: Path, extra: str = "") -> Path:
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  items: [x.jsonl]\n" + extra)
    return p


def test_the_fixed_rules_are_enforced_and_overrides_apply(tmp_path: Path):
    cfg = tc.load_config(_cfg(tmp_path), ["grpo.learning_rate=2e-5", "model.name=m"])
    assert cfg.grpo["learning_rate"] == 2e-5 and cfg.model.name == "m" and cfg.grpo["num_generations"] == 8
    for bad in ("grpo:\n  scale_rewards: group\n", "grpo:\n  gradient_checkpointing: false\n", "grpo:\n  beta: 0\n"):
        with pytest.raises(ValueError):
            tc.load_config(_cfg(tmp_path, bad))
    with pytest.raises(ValueError, match="unknown key"):
        tc.load_config(_cfg(tmp_path, "lora:\n  rank: 8\n"))
    (tmp_path / "e.yaml").write_text("data:\n  items: []\n")
    with pytest.raises(ValueError, match="empty"):
        tc.load_config(tmp_path / "e.yaml")


def test_override_values_keep_their_type():
    assert tc.parse_value("2e-5") == 2e-5 and isinstance(tc.parse_value("2e-5"), float)
    assert tc.parse_value("no") == "no" and tc.parse_value("epoch") == "epoch" and tc.parse_value("false") is False
    assert tc.parse_value("8") == 8 and tc.parse_value("[a.jsonl, b.jsonl]") == ["a.jsonl", "b.jsonl"]


def test_the_heldout_split_is_by_source_and_stable():
    sources = [f"src/task_{i:03d}" for i in range(400)]
    held = {s for s in sources if tc.heldout(s, 0.1, "seed")}
    assert 20 <= len(held) <= 60 and held == {s for s in sources if tc.heldout(s, 0.1, "seed")}
    assert held != {s for s in sources if tc.heldout(s, 0.1, "other")}
    assert not any(tc.heldout(s, 0.0, "seed") for s in sources)


def test_the_reward_is_the_brier_of_the_first_line():
    got = tc.brier_rewards(["0.9\n</parameter>", "0.2\n<", "maybe", "1"], [1, 0, 1, 0])
    assert got == pytest.approx([-0.01, -0.04, -1.0, -1.0])


def test_the_it17_config_loads():
    cfg = tc.load_config(Path(__file__).parents[1] / "configs" / "confidence" / "it17.yaml")
    built = tc.build_grpo_config(cfg)
    assert built.scale_rewards == "none" and built.gradient_checkpointing and built.beta == cfg.grpo["beta"]
    assert cfg.data.heldout_fraction == 0.3 and cfg.grpo["beta"] > 0 and cfg.model.family == "qwen3_5"
