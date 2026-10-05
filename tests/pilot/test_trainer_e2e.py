"""The stock `trl.GRPOTrainer` with `ReplayEnv`, `PilotReward` and `max_tool_calling_iterations=0`, end to end on CPU.

Real: `GRPOTrainer.__init__` and its `environment_factory` probe, the Qwen3.5 chat template rendering the frozen
transcript, `parse_response` on the sampled report turn, `_calculate_rewards`, the advantage code, the loss and the
optimiser step. Fake: the weights (a 4-layer random Qwen3.5 text model) and the sampler, which is scripted so the report
each completion makes is known. Needs the Qwen3.5 tokenizer in the local Hugging Face cache, else it skips.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from honest_agents.pilot import dataset
from honest_agents.pilot.env import ReplayEnv
from honest_agents.pilot.rewards import (
    CURVE_FILE,
    EPISODES_FILE,
    EpisodeLog,
    PilotReward,
)
from honest_agents.reward import RewardConfig
from tests.pilot.helpers import make_bank_row

_QWEN35 = "Qwen/Qwen3.5-9B"  # only its tokenizer and config are read, from the local Hugging Face cache


def _tiny_qwen35():
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        path = snapshot_download(_QWEN35, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached on this machine
        pytest.skip(f"{_QWEN35} tokenizer is not in the local Hugging Face cache")
    tokenizer = AutoTokenizer.from_pretrained(path)
    text = AutoConfig.from_pretrained(path).text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 4
    text.layer_types = text.layer_types[:4]
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
    text.linear_num_key_heads, text.linear_num_value_heads = 2, 4
    text.linear_key_head_dim = text.linear_value_head_dim = 16
    text.mtp_num_hidden_layers = 0
    torch.manual_seed(0)
    return tokenizer, AutoModelForCausalLM.from_config(text, dtype=torch.float32)


def _report_turn(status: str, **arguments: Any) -> str:
    """What the model samples after `...<|im_start|>assistant\n<think>\n` (Qwen3.5 tool-call format)."""
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in {"status": status, **arguments}.items())
    return f"\n</think>\n\n<tool_call>\n<function=submit_report>\n{params}</function>\n</tool_call><|im_end|>"


@pytest.mark.parametrize("scale", ["none", "group"])
def test_stock_grpo_trainer_trains_the_report_on_a_frozen_transcript(monkeypatch, tmp_path, scale):
    from datasets import Dataset  # noqa: F401
    from trl import GRPOConfig, GRPOTrainer

    monkeypatch.setenv("TRL_EXPERIMENTAL_SILENCE", "1")
    tokenizer, model = _tiny_qwen35()
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]

    rows, _ = dataset.build_rows(
        [
            make_bank_row(variant="pass", passed=True, original={"status": "complete"}),
            make_bank_row(variant="fail", passed=False, original={"status": "complete"}),
        ]
    )
    by_variant = {r["variant_id"]: r for r in rows}
    ds = dataset.to_hf_dataset([by_variant["pass"], by_variant["fail"]])
    prompts_seen: list[str] = []

    def scripted(
        self,
        prompt_ids,
        images,
        multimodal_fields,
        num_generations,
        has_tool_images=False,
    ):
        """Per prompt, two reports: the first claims completion, the second declines with a reason."""
        ids, logprobs = [], []
        for i, prompt in enumerate(prompt_ids):
            prompts_seen.append(tokenizer.decode(prompt))
            text = (
                _report_turn("complete", confidence=0.9) if i % 2 == 0 else _report_turn("incomplete", confidence=0.2)
            )
            sampled = encode(text)
            ids.append(sampled)
            logprobs.append([-0.25] * len(sampled))
        return ids, logprobs

    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", scripted)
    log = EpisodeLog(tmp_path / "arm", window=3, extra={"alpha": 0.25})
    reward = PilotReward("honest", log)
    cfg = RewardConfig(alpha=0.25)
    args = GRPOConfig(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=4,
        num_generations=2,
        max_completion_length=600,
        max_tool_calling_iterations=0,
        use_cpu=True,
        report_to="none",
        bf16=False,
        fp16=False,
        max_steps=1,
        save_strategy="no",
        scale_rewards=scale,
        beta=0.0,
        reward_weights=[1.0],
        mask_truncated_completions=True,
        shuffle_dataset=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward],
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        environment_factory=functools.partial(ReplayEnv, reward_config=cfg),
    )
    # TRL found exactly the two report tools on the environment (no bash, no get_reward reward source)
    assert sorted(t.__name__ for t in trainer.tools) == ["abstain", "submit_report"]
    assert trainer.reward_weights.tolist() == [1.0] and trainer.max_tool_calling_iterations == 0
    trainer.train()
    assert trainer.state.global_step == 1

    # the prompt TRL rendered is the frozen transcript: the instruction, the bash calls and their results, no report
    first = prompts_seen[0]
    assert "Parse the config" in first and "cmd0" in first and "cmd2" in first and "exit code: 0" in first
    assert "submit_report" in first  # listed as a tool in the system block, not called in the transcript
    assert first.rstrip().endswith("<think>")  # generation prompt: the model writes the report next

    episodes = [json.loads(line) for line in (tmp_path / "arm" / EPISODES_FILE).read_text().splitlines()]
    assert len(episodes) == 4 and {e["round"] for e in episodes} == {0}
    # dataset order is fixed: [pass, fail] x 2 reports each; the first report of each claims, the second declines
    by = {(e["variant_id"], e["claimed"]): e for e in episodes}
    assert by[("pass", True)]["case"] == "success_claimed" and by[("pass", False)]["case"] == "honest_underclaim"
    assert by[("fail", True)]["case"] == "fabrication" and by[("fail", False)]["case"] == "honest_failure"
    assert all(e["status"] in ("complete", "incomplete") and not e["tool_error"] for e in episodes)
    curve = json.loads((tmp_path / "arm" / CURVE_FILE).read_text())
    assert curve["claim_rate"] == 0.5 and curve["fabrication_among_failures"] == 0.5 and curve["claim_coverage"] == 0.5


def test_context_window_check_works_with_the_real_chat_template():
    from honest_agents.pilot import run
    from honest_agents.pilot.tools import tool_schemas

    tokenizer, _ = _tiny_qwen35()
    rows, _ = dataset.build_rows(
        [
            make_bank_row(variant="a", n_commands=6),
            make_bank_row(variant="b", n_commands=1),
        ]
    )
    worst = run.check_context_window(rows, tokenizer, tool_schemas(), budget=100_000)
    assert 200 < worst < 5000
    with pytest.raises(ValueError, match="available"):
        run.check_context_window(rows, tokenizer, tool_schemas(), budget=100)
