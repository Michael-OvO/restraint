"""Startup checks that need the model, the tokenizer or the vLLM server."""

from __future__ import annotations

import logging
from typing import Any

from honest_agents.models import (
    ModelFamily,
    family_spec,
    probe_vllm_server,
)
from honest_agents.train.config import TrainRunConfig
from honest_agents.train.derive import served_max_model_len
from honest_agents.train.plan import Plan, _server_url

logger = logging.getLogger("honest_agents.train")


# ── startup checks that need the model, the tokenizer or the server ─────────


def check_template_offline(
    cfg: TrainRunConfig, family: ModelFamily, tools: list[Any] | None = None
) -> list[str] | None:
    """Prefix-preservation of our tool flow with the real tokenizer, if it is cached locally.

    Mirrors how GRPOTrainer picks its template: the tokenizer's own if it is prefix-preserving, else TRL's training
    template. Returns the failures (empty = fine), or None when the tokenizer is not cached (nothing was checked).
    """
    from transformers import AutoTokenizer
    from trl.chat_template_utils import (
        add_response_schema,
        get_training_chat_template,
        is_chat_template_prefix_preserving,
    )

    from honest_agents.models import tool_flow_prefix_failures

    try:
        tok = AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, or a path that is not a model
        logger.info(
            "tokenizer for %s is not cached locally; template check skipped (it runs at training start)", cfg.model.name
        )
        return None
    if tools is None:
        try:
            from honest_agents.harness import tool_schemas
        except ImportError:
            logger.warning("harness.tool_schemas is not available; template check skipped")
            return None
        tools = tool_schemas()
    tok = add_response_schema(tok)
    template = None if is_chat_template_prefix_preserving(tok) else get_training_chat_template(tok)
    kwargs = dict(family_spec(family).chat_template_kwargs) | dict(cfg.grpo.get("chat_template_kwargs") or {})
    return tool_flow_prefix_failures(tok, tools=tools, chat_template=template, chat_template_kwargs=kwargs)


def check_prompt_budget(trainer: Any, plan: Plan) -> int:
    """Longest system prompt + tool schemas + instruction, in tokens, against `vllm.prompt_token_reserve`.

    The server window is prompt reserve + budget + turn cap; a prompt longer than the reserve would shrink the room for
    the transcript and could make vLLM reject a request mid-run.
    """
    from honest_agents.harness import SYSTEM_PROMPT, read_instruction

    longest = 0
    datasets = [plan.train_dataset] + ([plan.eval_dataset] if plan.eval_dataset is not None else [])
    for task_dir in sorted({r["task_dir"] for ds in datasets for r in ds}):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": read_instruction(task_dir)},
        ]
        ids = trainer.processing_class.apply_chat_template(
            messages,
            tools=trainer.tools or None,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            chat_template=getattr(trainer, "chat_template", None),
            **trainer.chat_template_kwargs,
        )
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        longest = max(longest, len(ids))
    reserve = plan.limits.prompt_token_reserve
    if longest > reserve:
        raise RuntimeError(
            f"the longest prompt is {longest} tokens but vllm.prompt_token_reserve={reserve}; raise the reserve (the "
            "vLLM window is computed from it) and restart the server"
        )
    logger.info("longest prompt: %d tokens (reserve %d)", longest, reserve)
    return longest


def check_lm_head_cast(cfg: TrainRunConfig) -> None:
    """TRL's `cast_lm_head_to_fp32` is documented for untied embeddings only."""
    if not cfg.model.cast_lm_head_to_fp32:
        return
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg.model.name, revision=cfg.model.revision)
    text = getattr(config, "text_config", None) or config
    if getattr(text, "tie_word_embeddings", False):
        raise ValueError(
            f"model.cast_lm_head_to_fp32 needs untied embeddings but {cfg.model.name} ties them "
            "(trl GRPOConfig.cast_lm_head_to_fp32); turn it off for this model"
        )


def startup_probe(plan: Plan, retries: int = 3, wait_sec: float = 10.0) -> dict[str, Any] | None:
    """Assert the running vLLM server can serve this run: window, token ids, processed logprobs, full distribution."""
    import time

    from honest_agents.models import VLLMProbeError

    g = plan.grpo_config
    if not (g.use_vllm and g.vllm_mode == "server" and plan.cfg.vllm.startup_probe):
        return None
    url, last = _server_url(plan.cfg).replace("0.0.0.0", "127.0.0.1"), None
    for attempt in range(retries):
        try:
            return probe_vllm_server(
                url,
                required_max_model_len=served_max_model_len(plan.cfg, plan.limits),
                temperature=float(g.temperature),
            )
        except VLLMProbeError as e:
            last = e
            if "failed:" not in str(e) or attempt == retries - 1:  # only connection errors are worth retrying
                break
            time.sleep(wait_sec)
    raise RuntimeError(f"vLLM startup probe failed for {url}: {last}") from last
