"""`train.confidence_exact`: the value tree's probabilities, the exact expected Brier against brute force and against
sampling, and the tree KL. No model."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from honest_agents.sft.confidence_bandit import brier_reward, parse_confidence
from honest_agents.train import confidence_exact as ce

V = 20  # toy vocabulary: ids 0-9 the digits (1 and 0 are the tokens `1` and `0`), 10 `.`, 12 a newline, the rest other
# "1" and "0" are digit tokens, as in Qwen3.5
T = ce.TreeTokens(one=1, zero=0, dot=10, newline=12, digits=tuple(range(10)))
TEXT = {**{d: str(d) for d in range(10)}, 10: ".", 11: "x", 12: "\n", **{i: "x" for i in range(13, V)}}


def _logprobs(seed: int):
    g = torch.Generator().manual_seed(seed)
    return [torch.log_softmax(torch.randn(V, generator=g) * 2, dim=-1) for _ in range(3)]


def test_the_tree_partitions_the_probability_and_matches_brute_force():
    lp = _logprobs(0)
    probs = ce.tree_probabilities(*lp, T)
    total = probs["one"] + probs["zero"] + probs["tenths"].sum() + probs["other"]
    assert float(total) == pytest.approx(1.0, abs=1e-6)
    p = [x.exp() for x in lp]
    brute = {"one": p[0][1], "zero": p[0][0] * p[1][12]}
    assert float(probs["one"]) == pytest.approx(float(brute["one"]))
    assert float(probs["zero"]) == pytest.approx(float(brute["zero"]))
    for d in range(10):
        assert float(probs["tenths"][d]) == pytest.approx(float(p[0][0] * p[1][10] * p[2][d]), rel=1e-4)
    for y in (0, 1):
        manual = (float(probs["one"]) * (1 - y) ** 2 + float(probs["zero"]) * y**2
                  + sum(float(probs["tenths"][d]) * (d / 10 - y) ** 2 for d in range(10)) + float(probs["other"]))  # fmt: skip
        assert float(ce.expected_brier(probs, y)) == pytest.approx(manual, rel=1e-6)


def test_the_exact_objective_is_the_mean_reward_of_sampled_values():
    """Sample token by token from the same three distributions, parse as the evaluator does (first line; a value whose
    first token is 1 counts as 1.0, digits after the tenths are not drawn): the mean Brier cost converges to the exact
    expectation."""
    lp = _logprobs(3)
    p = [x.exp().tolist() for x in lp]
    probs = ce.tree_probabilities(*lp, T)
    rng = random.Random(0)
    for y in (0, 1):
        costs = []
        for _ in range(40000):
            a = rng.choices(range(V), p[0])[0]
            if a == 1:
                text = "1\n"
            elif a == 0:
                b = rng.choices(range(V), p[1])[0]
                text = (
                    "0\n"
                    if b == 12
                    else ("0." + TEXT[rng.choices(range(V), p[2])[0]] + "\n" if b == 10 else "0" + TEXT[b])
                )
            else:
                text = TEXT[a]
            costs.append(-brier_reward(parse_confidence(text), y))
        assert sum(costs) / len(costs) == pytest.approx(float(ce.expected_brier(probs, y)), abs=0.01)


def test_the_tree_kl_is_zero_at_the_reference_and_positive_away_from_it():
    lp = _logprobs(5)
    probs = ce.tree_probabilities(*lp, T)
    assert float(ce.tree_kl(lp, lp, probs)) == pytest.approx(0.0, abs=1e-7)
    other = _logprobs(6)
    assert float(ce.tree_kl(lp, other, probs)) > 0
    v = float(ce.expected_value(probs))
    assert 0.0 <= v <= 1.0


def test_the_exact_section_refuses_a_zero_kl_and_unknown_keys():
    assert ce.exact_section({}).beta > 0
    with pytest.raises(ValueError):
        ce.exact_section({"beta": 0})
    with pytest.raises(ValueError, match="unknown"):
        ce.exact_section({"lr": 1})


def test_the_lora_targets_match_the_class_qwen35_declares_and_three_logits_are_exact():
    """Regression (it17b's first launch): loaded as AutoModelForCausalLM, Qwen3.5 is the text-only class and the LoRA
    targets match nothing; the checkpoint declares the conditional-generation class, which TRL (and so SFT, merge and
    serving) load. On a tiny random model of that class: the targets match and `logits_to_keep=3` is exact."""
    import re

    import transformers
    from transformers import AutoConfig

    from honest_agents.models import lora_target_modules

    try:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 config not cached")
    tc = cfg.text_config
    for k, v in {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4, "num_attention_heads": 4,
                 "num_key_value_heads": 2, "head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                 "linear_key_head_dim": 16, "linear_value_head_dim": 16}.items():  # fmt: skip
        setattr(tc, k, v)
    tc.layer_types = tc.layer_types[:4]
    for k, v in {"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64}.items():
        if hasattr(cfg.vision_config, k):
            setattr(cfg.vision_config, k, v)
    model = getattr(transformers, cfg.architectures[0])._from_config(cfg).eval()
    rx = re.compile(lora_target_modules("qwen3_5"))
    assert any(rx.fullmatch(n) for n, _ in model.named_modules())
    ids = torch.randint(0, 1000, (1, 30))
    with torch.no_grad():
        full = model(input_ids=ids).logits[0, -3:]
        kept = model(input_ids=ids, logits_to_keep=3).logits[0]
    assert float((full - kept).abs().max()) == 0.0
