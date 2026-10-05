"""Synthetic bank rows and a small tokenizer for the SFT tests: no Docker, no GPU, no model download."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from honest_agents import bank
from honest_agents.bank import BankRow, BlockerFacts, PathFact
from honest_agents.report import parse_report
from honest_agents.reward import EpisodeOutcome, RewardConfig, compute_reward
from tests.pilot.helpers import INSTRUCTION, SYSTEM, bash_turn, report_call, write_bank

__all__ = ["bash_turn", "make_row", "report_call", "tiny_tokenizer", "write_bank"]

CACHED_TOKENIZER = "Qwen/Qwen3-0.6B-Base"  # a Qwen2 BPE tokenizer; the Qwen3 template is set on it below


def make_row(
    variant: str = "v0",
    *,
    passed: bool = False,
    original: dict[str, Any] | None = None,
    tool: str = "submit_report",
    turns: list[tuple[str, str]] | None = None,
    facts: BlockerFacts | None = None,
    split: str = "train",
    feasible: bool = True,
    expected_targets: list[str] | None = None,
    expected_blockers: list[str] | None = None,
    tamper: bool = False,
    masked_reason: str | None = None,
    no_report: bool = False,
    episode_id: str | None = None,
    reward_config: RewardConfig | None = None,
    damage_paths: tuple[str, ...] = (),
    spec: dict[str, Any] | None = None,
    is_control: bool = False,
    blocker_check: str = "unknown",
) -> BankRow:
    """A banked trajectory: `turns` = [(command, output)], then a report turn (`original` args via `tool`)."""
    turns = turns if turns is not None else [("ls /app", "in.txt")]
    original = original if original is not None else ({"status": "complete"} if passed else {"status": "incomplete"})
    completion: list[dict[str, Any]] = []
    for command, output in turns:
        completion += bash_turn(command, output)
    if not no_report:
        completion.append(report_call(original, tool))
    o = EpisodeOutcome(
        feasible=feasible,
        report=None
        if no_report
        else parse_report({**original} if tool == "submit_report" else _abstain_args(original)),
        tests_passed=passed,
        n_actions=len(turns),
        expected_blockers=expected_blockers or [],
        expected_targets=expected_targets or [],
        tamper=tamper,
        masked_reason=masked_reason,
        damage_paths=damage_paths,
        is_control=is_control,
        blocker_check=blocker_check,
    )
    cfg = reward_config or RewardConfig()
    br = compute_reward(o, cfg)
    row = bank.make_row(
        prompt=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": INSTRUCTION}],
        completion=completion,
        outcome=o,
        breakdown=br,
        reward_value=br.reward,
        task={
            "variant_id": variant,
            "split": split,
            "task_dir": f"tasks/{variant}",
            "perturbation": "secret_perturbation",
            **({"spec": spec} if spec is not None else {}),
        },
        blocker_facts=facts or BlockerFacts(commands=["ls", "cat", "python3"], cwd="/app", network_mode="public"),
        reward_config=cfg,
        group=variant,
    )
    return row.model_copy(update={"episode_id": episode_id}) if episode_id else row


def _abstain_args(a: dict[str, Any]) -> dict[str, Any]:
    return {"status": "abstain", **a}


def missing_command_facts(*present: str) -> BlockerFacts:
    return BlockerFacts(commands=["ls", "cat", *present], cwd="/app", network_mode="public")


def denied_facts(path: str) -> BlockerFacts:
    return BlockerFacts(
        commands=["ls", "cat"], paths={path: PathFact(exists=True, readable=False, writable=False)}, cwd="/app"
    )


def tiny_tokenizer():
    """The cached Qwen tokenizer with the Qwen3 chat template, else a small local byte-level BPE with the same special
    tokens (so the template renders and the generation markers work either way)."""
    from transformers import AutoTokenizer
    from trl import chat_template_utils as ctu

    try:
        tok = AutoTokenizer.from_pretrained(CACHED_TOKENIZER, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached
        tok = _local_tokenizer()
    tok.chat_template = ctu.qwen3_chat_template
    return tok


def _local_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = ['abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 {}[]":,.-_/\n' * 4]
    tk.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")


def bank_dir(tmp_path: Path, rows: list[BankRow]) -> Path:
    return write_bank(tmp_path / "bank", rows)
