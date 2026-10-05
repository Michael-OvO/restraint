"""Weight-sync check (SPEC_V3 10.1, 10.2): the statistics, the verdict, the request format and, on CPU, the whole
procedure against TRL's real `_iter_named_params` and a simulated packed weight transfer.

The simulation reproduces the probable TRL bug on a tiny model: the adapter is merged, parameters are yielded, and the
un-merge runs inside the same generator, while the consumer (vLLM's packed transfer) reads the final chunk only after the
generator is exhausted. Stock TRL then hands the server the un-merged values for that chunk; `clone_synced_params` fixes it.
What cannot be tested here is a real vLLM server and a GPU (`scripts/sync_check.sh`).
"""

from __future__ import annotations

import contextlib
import copy
import math
import random
from types import SimpleNamespace

import pytest

from honest_agents import sync_check as sc

# ── statistics and verdict ───────────────────────────────────────────────────


def test_compare_identical_logprobs():
    s = sc.compare_logprobs([-1.0, -2.0, -0.5], [-1.0, -2.0, -0.5])
    assert (s.n_tokens, s.n_nan, s.mean_gap, s.mean_abs_gap, s.max_abs_gap) == (3, 0, 0.0, 0.0, 0.0)
    assert s.mean_ratio == pytest.approx(1.0)


def test_compare_measures_ratio_and_gap():
    s = sc.compare_logprobs([-1.0, -1.0], [-1.1, -0.9])  # gaps +0.1 and -0.1
    assert s.mean_gap == pytest.approx(0.0, abs=1e-12) and s.mean_abs_gap == pytest.approx(0.1)
    assert s.mean_ratio == pytest.approx((math.exp(0.1) + math.exp(-0.1)) / 2)


def test_non_finite_and_missing_values_are_nan_not_zero():
    s = sc.compare_logprobs([-1.0, float("nan"), -1.0, None], [-1.0, -1.0, float("-inf"), -1.0])
    assert s.n_nan == 3 and s.n_tokens == 1
    only_nan = sc.compare_logprobs([float("nan")], [-1.0])
    assert only_nan.n_tokens == 0 and math.isnan(only_nan.mean_ratio)
    assert not sc.verdict(only_nan, sc.compare_logprobs([-1.0], [-1.0]), effect=1.0)["passed"]


def test_misaligned_sequences_are_an_error():
    with pytest.raises(ValueError, match="misaligned"):
        sc.compare_logprobs([-1.0], [-1.0, -2.0])


def _stats(ratio=1.0, abs_gap=0.001, nan=0):
    return sc.SyncStats(1000, ratio, 0.0, abs_gap, abs_gap * 10, nan)


def test_verdict_passes_a_good_sync():
    v = sc.verdict(_stats(1.002, 0.0011), _stats(1.0, 0.001), effect=0.05)
    assert v["passed"] and all(v["checks"].values())


@pytest.mark.parametrize(
    ("adapter", "effect", "failing"),
    [
        (_stats(ratio=1.02), 0.05, "mean_ratio_within_tolerance"),
        (_stats(ratio=0.98), 0.05, "mean_ratio_within_tolerance"),
        (_stats(abs_gap=0.01), 0.05, "gap_not_larger_than_base"),
        (_stats(nan=1), 0.05, "no_nan"),
        (_stats(), 0.002, "adapter_moves_logprobs"),  # an adapter that changed nothing proves nothing
        (_stats(), float("nan"), "adapter_moves_logprobs"),
    ],
)
def test_verdict_names_the_failing_criterion(adapter, effect, failing):
    v = sc.verdict(adapter, _stats(1.0, 0.001), effect=effect)
    assert v["passed"] is False and v["checks"][failing] is False


def test_verdict_with_a_zero_base_gap_is_not_an_impossible_bar():
    assert sc.verdict(_stats(1.0, 5e-5), _stats(1.0, 0.0), effect=0.05)["passed"]
    assert not sc.verdict(_stats(1.0, 5e-3), _stats(1.0, 0.0), effect=0.05)["passed"]


def test_recommendation_follows_the_spec_decision_rule():
    ok, bad = {"passed": True}, {"passed": False}
    assert "clone patch" in sc.recommend({"clone": ok, "stock": bad}, "qwen3_5")
    assert "adapter-only" in sc.recommend({"clone": bad, "stock": bad, "adapter_only": ok}, "qwen3_5")
    assert "float32" in sc.recommend({"clone": bad, "stock": bad}, "qwen3_5") and "Qwen3-4B" in sc.recommend(
        {"clone": bad}, "qwen3_5"
    )
    assert "Qwen3-4B" not in sc.recommend({"clone": bad}, "qwen3")


# ── sequences and the server request ─────────────────────────────────────────


def _tokenizer(repo: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{repo} tokenizer not in the local HF cache")


def test_build_sequences_uses_the_real_template_and_reaches_the_target():
    from honest_agents.harness import tool_schemas

    tok = _tokenizer("Qwen/Qwen3.5-9B")
    seqs = sc.build_sequences(
        tok, tools=tool_schemas(), chat_template_kwargs={"enable_thinking": False}, target_tokens=2_500, seed=1
    )
    assert sum(len(s) - 1 for s in seqs) >= 2_500 and all(len(s) <= 4_096 for s in seqs)
    assert seqs == sc.build_sequences(
        tok, tools=tool_schemas(), chat_template_kwargs={"enable_thinking": False}, target_tokens=2_500, seed=1
    )  # fixed: the same tokens every time
    text = tok.decode(seqs[0])
    assert "submit_report" in text and "<tool_call>" in text
    with pytest.raises(ValueError, match="target_tokens"):
        sc.build_sequences(tok, tools=None, target_tokens=100)


def test_vllm_request_asks_for_prompt_logprobs_and_aligns_tokens():
    requests = []

    def http(method, url, payload):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "m"}]}
        requests.append(payload)
        ids = payload["prompt"]
        entries = [None] + [{str(t): {"logprob": -0.1 * t, "rank": 1}, "999": {"logprob": -9.0}} for t in ids[1:]]
        return {"choices": [{"prompt_logprobs": entries}]}

    out = sc.vllm_logprobs("http://s", [[5, 1, 2, 3], [7, 4]], http=http)
    assert out == [pytest.approx([-0.1, -0.2, -0.3]), pytest.approx([-0.4])]
    assert requests[0]["prompt_logprobs"] == 0 and requests[0]["max_tokens"] == 1 and requests[0]["temperature"] == 1.0


def test_vllm_missing_prompt_logprobs_is_an_error():
    def http(method, url, payload):
        return {"data": [{"id": "m"}]} if url.endswith("/v1/models") else {"choices": [{"prompt_logprobs": None}]}

    with pytest.raises(RuntimeError, match="max-logprobs"):
        sc.vllm_logprobs("http://s", [[1, 2, 3]], http=http)


# ── the whole procedure on CPU, with TRL's real parameter iteration ──────────


def _tiny():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, tie_word_embeddings=False,
    )  # fmt: skip
    return Qwen3ForCausalLM(cfg)


class FakeVLLM:
    """A server holding its own copy of the base weights, fed by a packed-transfer-like consumer of `_iter_named_params`."""

    LAST_CHUNK = 12  # parameters vLLM's packed producer reads only once the generator is exhausted

    def __init__(self, base_model, trainer_model):
        from trl.generation.vllm_generation import VLLMGeneration

        self.server = copy.deepcopy(base_model).eval()
        self.vg = object.__new__(VLLMGeneration)
        self.vg.model = trainer_model
        self.vg._dist = SimpleNamespace(is_fsdp=False, gather_params=lambda params: contextlib.nullcontext())
        self.vg.sync_weights = self.sync_weights

    def sync_weights(self):
        import torch

        params = dict(self.server.named_parameters())
        pending = []
        for i, (name, tensor) in enumerate(self.vg._iter_named_params()):
            pending.append((name, tensor))  # a view of the live weights unless the iterator clones
            if len(pending) > self.LAST_CHUNK:
                n, t = pending.pop(0)
                with torch.no_grad():
                    params[n].copy_(t)
        with torch.no_grad():  # the generator is exhausted (adapter un-merged): the final chunk is read now
            for n, t in pending:
                params[n].copy_(t)

    def http(self, method, url, payload):
        import torch

        if url.endswith("/v1/models"):
            return {"data": [{"id": "m"}]}
        ids = payload["prompt"]
        with torch.no_grad():
            lp = torch.log_softmax(self.server(input_ids=torch.tensor([ids])).logits[0].float(), -1)
        entries = [None] + [{str(t): {"logprob": lp[i, t].item()}} for i, t in enumerate(ids[1:])]
        return {"choices": [{"prompt_logprobs": entries}]}


def _peft_trainer():
    from peft import LoraConfig, get_peft_model

    base = _tiny()
    model = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   task_type="CAUSAL_LM"),
    )  # fmt: skip
    fake = FakeVLLM(base, model)
    trainer = SimpleNamespace(model=model, vllm_generation=fake.vg, temperature=1.0)
    return trainer, fake


def _sequences(n=6, length=40, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(1, 64) for _ in range(length)] for _ in range(n)]


def test_trainer_logprobs_match_a_manual_forward():
    import torch

    model = _tiny().eval()
    seq = _sequences(1)[0]
    got = sc.trainer_logprobs(model, [seq])[0]
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=torch.tensor([seq])).logits[0, :-1].float(), -1)
    want = [lp[i, t].item() for i, t in enumerate(seq[1:])]
    assert got == pytest.approx(want, abs=1e-5)


def test_set_adapter_scale_makes_the_adapter_nonzero_and_zeroes_it_again():
    trainer, _ = _peft_trainer()
    seq = _sequences(1)
    base = sc.trainer_logprobs(trainer.model, seq)
    assert sc.set_adapter_scale(trainer.model, 0.05) > 0
    moved = sc.trainer_logprobs(trainer.model, seq)
    assert sc.effect_size(sc._flat(base), sc._flat(moved)) > 1e-3
    sc.set_adapter_scale(trainer.model, 0.0)
    assert sc._flat(sc.trainer_logprobs(trainer.model, seq)) == pytest.approx(sc._flat(base), abs=1e-6)


def test_patched_sync_passes_and_stock_sync_is_caught():
    """SPEC_V3 10.1.1: the final chunk of the stock path is read after the un-merge, so the server serves base values
    for those parameters; the check sees it as a log-probability gap, and the clone patch removes it."""
    trainer, fake = _peft_trainer()
    result = sc.run_check(
        trainer,
        base_url="http://x",
        sequences=_sequences(),
        modes=("stock", "clone"),
        adapter_scale=0.05,
        http=fake.http,
    )
    assert result["base"]["mean_abs_gap"] < 1e-5  # the server and the trainer agree before any adapter is synced
    stock, clone = result["modes"]["stock"], result["modes"]["clone"]
    assert stock["passed"] is False and stock["checks"]["gap_not_larger_than_base"] is False
    assert clone["passed"] is True, clone
    assert result["passed"] is True
    assert "clone patch" in result["recommendation"]


def test_with_the_patch_only_the_modes_that_run_are_reported():
    trainer, fake = _peft_trainer()
    result = sc.run_check(trainer, base_url="http://x", sequences=_sequences(), adapter_scale=0.05, http=fake.http)
    assert set(result["modes"]) == {"clone"} and result["passed"]


def test_a_model_without_an_adapter_is_refused():
    base = _tiny()
    fake = FakeVLLM(base, base)
    trainer = SimpleNamespace(model=base, vllm_generation=fake.vg, temperature=1.0)
    with pytest.raises(RuntimeError, match="no LoRA adapter"):
        sc.run_check(trainer, base_url="http://x", sequences=_sequences(1), http=fake.http)


def test_unknown_mode_is_refused():
    trainer, fake = _peft_trainer()
    with pytest.raises(ValueError, match="unknown sync mode"):
        sc.run_check(trainer, base_url="http://x", sequences=_sequences(1), modes=("adapter_only",), http=fake.http)
