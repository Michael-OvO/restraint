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


def test_assistant_positions_cover_the_models_turns_and_nothing_else():
    """On a real it17 item rendered with the served Qwen3.5 template: every anchored token is inside an assistant turn
    (the commands and the open report turn), none is system, user or tool text."""
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec
    from honest_agents.sft import confidence_bandit as cb

    items = Path(__file__).parents[1] / "data" / "it17" / "items_it16b.jsonl"
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")
    if not items.is_file():
        pytest.skip("it17 items not built")
    item = cb.Item.from_json(json.loads(items.read_text().splitlines()[0]))
    prompt = cb.render_prefix(item.messages, tok, tool_schemas(), dict(family_spec("qwen3_5").chat_template_kwargs))
    ids = tok(prompt, add_special_tokens=False).input_ids
    pos = ce.assistant_positions(ids, ce.ChatMarkers.of(tok))
    anchored = tok.decode([ids[p] for p in pos])
    commands = [c["function"]["arguments"]["command"] for m in item.messages if m.get("role") == "assistant"
                for c in m.get("tool_calls") or [] if c["function"]["name"] == "bash"]  # fmt: skip
    assert commands and all(c.split()[0] in anchored for c in commands)
    assert "submit_report" in anchored and "<parameter=status>" in anchored  # the open report turn, up to the cut
    tool_texts = [str(m["content"]) for m in item.messages if m.get("role") == "tool" and len(str(m["content"])) > 40]
    assert tool_texts and not any(t[:40] in anchored for t in tool_texts)
    assert item.messages[1]["content"][:60] not in anchored  # the instruction
    assert pos[-1] == len(ids) - 1  # the open report turn runs to the cut


def test_the_action_kl_is_zero_at_the_reference_nonnegative_and_scales_to_the_transcript():
    ref = torch.log(torch.tensor([0.9, 0.5, 0.99, 0.2]))
    assert float(ce.action_kl(ref, ref, 100)) == 0.0
    moved = torch.log(torch.tensor([0.6, 0.5, 0.5, 0.25]))
    k = float(ce.action_kl(moved, ref, 100))
    assert k > 0 and float(ce.action_kl(moved, ref, 200)) == pytest.approx(2 * k)
    assert float(ce.action_kl(torch.zeros(0), torch.zeros(0), 10)) == 0.0


def test_sampled_positions_are_bounded_and_stable():
    pos = list(range(5000))
    a, b = ce.sample_positions(pos, 512, "item-1"), ce.sample_positions(pos, 512, "item-1")
    assert a == b and len(a) == 512 and a == sorted(a) and ce.sample_positions(pos, 512, "item-2") != a
    assert ce.sample_positions(pos[:10], 512, "x") == pos[:10]


def test_one_forward_gives_the_node_and_action_logprobs_of_a_full_pass():
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True, dtype=torch.float32)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3-0.6B not cached")
    model.eval()
    ids = list(range(100, 140)) + [15, 13]  # a prompt of 40, then two tokens
    actions = [5, 17, 30, 39]
    with torch.no_grad():
        nodes, act = ce._forward(model, ids, 40, actions)
        full = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), dim=-1)
    for i, pos in enumerate((39, 40, 41)):
        assert torch.allclose(nodes[i], full[pos], atol=1e-4)
    assert torch.allclose(act, torch.stack([full[a - 1, ids[a]] for a in actions]), atol=1e-4)


def test_the_drift_abort_and_anchor_settings_are_validated():
    assert ce.exact_section({}).anchor_beta == 0.0 and ce.exact_section({}).abort_action_kl_per_token is None
    assert ce.exact_section({"anchor_beta": 1.0, "abort_action_kl_per_token": 0.02}).anchor_beta == 1.0
    for bad in ({"anchor_beta": -1}, {"anchor_max_tokens": 0}, {"abort_action_kl_per_token": 0}):
        with pytest.raises(ValueError):
            ce.exact_section(bad)


# ── iteration 19, K2a: the status target ──────────────────────────────────────────────────────────────────────────────


def test_the_status_prompt_ends_where_the_claim_decision_is_made():
    """On a real it17 item with the served template: cut at the status, the next token is `complete` or the first token
    of `incomplete`, and nothing of the report's other fields is in the prompt."""
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from honest_agents.eval.rollout import tool_schemas
    from honest_agents.models import family_spec
    from honest_agents.sft import confidence_bandit as cb

    items = Path(__file__).parents[1] / "data" / "it17" / "items_it16b.jsonl"
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3.5 tokenizer not cached")
    if not items.is_file():
        pytest.skip("it17 items not built")
    t = ce.StatusTokens.of(tok)
    assert (t.complete, t.incomplete) == (14311, 258)
    item = cb.Item.from_json(json.loads(items.read_text().splitlines()[0]))
    kwargs = dict(family_spec("qwen3_5").chat_template_kwargs)
    prompt = cb.render_prefix(cb.status_messages(item.messages), tok, tool_schemas(), kwargs)
    assert prompt.endswith("<parameter=status>\n") and "<parameter=confidence>" not in prompt[-400:]
    for value, first in (("complete", t.complete), ("incomplete", t.incomplete)):
        assert tok(prompt + value, add_special_tokens=False).input_ids[len(tok(prompt).input_ids)] == first


def test_status_deltas_follow_the_crgrpo_table():
    from honest_agents import crgrpo

    cfg = crgrpo.CrgrpoConfig()
    rows = [{"item_id": "c1", "is_control": True, "feasible": True, "cell": "feasible/complete", "y": 1, "family": "high_risk"},
            {"item_id": "c2", "is_control": True, "feasible": True, "cell": "feasible/complete", "y": 0, "family": "high_risk"},
            {"item_id": "p1", "is_control": False, "feasible": True, "cell": "feasible/complete", "y": 0, "family": "feasible"},
            {"item_id": "t1", "is_control": False, "feasible": False, "cell": "infeasible/complete", "y": 0, "family": "high_risk"}]  # fmt: skip
    d = ce.status_deltas(rows)
    v = (1.0 - cfg.mu_fc) / 2  # the high_risk controls: one pass, one fail
    assert d["c1"] == pytest.approx(1 - cfg.base.alpha) and d["p1"] == pytest.approx(-(cfg.mu_fc + cfg.base.alpha))
    assert d["t1"] == pytest.approx(-cfg.k(v) * cfg.mu_tw)


def test_a_status_step_moves_the_claim_decision_in_the_payoff_direction():
    """End to end on a small real model: one optimiser step on a failing claim (ΔL < 0) lowers π(claim); on a
    passing one (ΔL > 0) raises it; the KL starts at 0."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    try:
        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True, dtype=torch.float32)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3-0.6B not cached")
    t = ce.StatusTokens(complete=11463, incomplete=258)
    ex = ce.ExactSection(target="status", learning_rate=1e-3, beta=0.04, report_to="none")
    for delta in (-0.386, 0.9):
        torch.manual_seed(0)
        model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
        item = (list(range(200, 260)), 60, {"y": int(delta > 0), "delta": delta}, [], 0)
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        loss, stats = ce.status_item_loss(model, item, t, ex)
        assert stats["kl"] == pytest.approx(0.0, abs=1e-6)
        loss.backward()
        opt.step()
        _, after = ce.status_item_loss(model, item, t, ex)
        assert (after["pi_claim"] - stats["pi_claim"]) * delta > 0
        base = model.unload()


def test_the_status_scorer_writes_paired_rows_for_g2_and_the_candidate(tmp_path, monkeypatch):
    """`--status-score` on three real dev items with the small cached model and an untrained adapter: both files hold
    one row per item in the rescore format, and an untrained adapter scores exactly as the start model."""
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from honest_agents.eval import confidence_rescore as cr
    from honest_agents.train import confidence as tc

    bank = Path(__file__).parents[1] / "data" / "it17" / "dev_banks" / "select"
    try:
        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True, dtype=torch.float32)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen3-0.6B not cached")
    if not bank.is_dir():
        pytest.skip("dev banks not present")
    adapter = tmp_path / "adapter"
    get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])).save_pretrained(str(adapter))
    real = cr.items_of
    monkeypatch.setattr(cr, "items_of", lambda *a, **k: (real(*a, **k)[0][:3], {}))
    cfg = tc.ConfidenceConfig()
    cfg.model.name = cfg.model.tokenizer = "Qwen/Qwen3-0.6B-Base"
    cfg.model.family = "qwen3"
    cfg.model.dtype = "float32"
    monkeypatch.setattr(ce.StatusTokens, "of", staticmethod(_qwen3_status))  # Qwen3's ids, not Qwen3.5's
    g2, cand = ce.status_score(cfg, str(adapter), [str(bank)], [])
    assert len(g2) == len(cand) == 3 and all(0.0 <= r["mean"] <= 1.0 for r in g2 + cand)
    assert all(abs(a["mean"] - b["mean"]) < 1e-5 for a, b in zip(g2, cand, strict=True))  # LoRA B starts at zero


def _qwen3_status(tok):
    c = tok("complete", add_special_tokens=False).input_ids
    i = tok("incomplete", add_special_tokens=False).input_ids
    return ce.StatusTokens(c[0], i[0])
