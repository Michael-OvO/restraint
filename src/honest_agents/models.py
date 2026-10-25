"""Model-family specifics: Qwen3 (bring-up) vs Qwen3.5 (headline).

Everything here was checked against the installed sources (transformers 5.18 `models/qwen3_5`, trl 1.14.1
`chat_template_utils.py` / `scripts/vllm_serve.py`, vllm 0.30.0 `engine/arg_utils.py`, `tool_parsers/__init__.py`,
`reasoning/__init__.py`, `entrypoints/launchers/cli_args.py`). Things that can only be confirmed on the GPU node are
marked UNVERIFIED.

Why the two families differ:

* Qwen3.5 checkpoints are `Qwen3_5ForConditionalGeneration` (model_type `qwen3_5`): a VLM wrapper whose text tower lives
  at `model.language_model` and mixes full attention (`self_attn.{q,k,v,o}_proj`) with Gated DeltaNet linear attention
  (`linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`). LoRA must be restricted to the language model
  (the vision tower is never exercised by our text-only prompts) and must cover both members of each pair vLLM fuses
  (`in_proj_qkv`+`in_proj_z`, `in_proj_b`+`in_proj_a`), otherwise merged-weight sync to vLLM drifts.
* Tool-call formats differ: Qwen3 emits Hermes JSON `<tool_call>{...}</tool_call>`; Qwen3.5 emits Qwen3-Coder XML
  `<tool_call><function=…><parameter=…>`. During training TRL parses tool calls itself from token ids
  (`add_response_schema` → `qwen3_template` / `qwen3_5_template`), so vLLM's parsers only matter for the
  OpenAI-compatible chat endpoint used by evaluation.
"""

from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class ModelFamily(StrEnum):
    QWEN3 = "qwen3"
    QWEN3_5 = "qwen3_5"


# Hub revisions of the base models (DEF-51): the one place a Hub model id is pinned. Every config section that names one
# of these ids resolves `model.revision` to its pin (`pinned_revision`); the serve scripts, `sft.merge` and
# `setup_node.sh` read it from here (`python -m honest_agents.models revision NAME`). Qwen3.5-4B: the snapshot the node
# log `2026-10-06-node1/node-logs/hf_qwen35.log` downloaded (archive); the Hub's last commit (2026-03-02) predates
# every archived run. Qwen3-4B (bring-up only, no archived run): the Mac's cached `refs/main` of 2026-10-05.
QWEN3_5_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN3_4B_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PINNED_REVISIONS: dict[str, str] = {
    "Qwen/Qwen3.5-4B": QWEN3_5_4B_REVISION,
    "Qwen/Qwen3-4B": QWEN3_4B_REVISION,
}


def pinned_revision(name: str, revision: str | None = None) -> str | None:
    """The revision to load `name` at: its pin for a pinned Hub id, else `revision` as given (a local directory, such
    as a merged model, has none). A pinned id with a different explicit revision is refused: one pin per model."""
    pin = PINNED_REVISIONS.get(name)
    if pin is None:
        return revision
    if revision is not None and revision != pin:
        raise ValueError(f"model.revision {revision!r} for {name} differs from its pin {pin} (models.PINNED_REVISIONS)")
    return pin


_QWEN3_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# PEFT full-matches a string `target_modules` against each module's qualified name (e.g.
# `model.language_model.layers.3.linear_attn.in_proj_qkv`). The vision tower (`model.visual.*`: `attn.qkv`, `attn.proj`,
# `mlp.linear_fc{1,2}`) and `lm_head` are excluded by construction.
_QWEN3_5_LORA_REGEX = (
    r".*language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))"
)

# Flags TRL's (deprecated) `trl vllm-serve` imposes on `vllm serve` for weight sync + IS correction
# (trl/scripts/vllm_serve.py::build_command). `--enable-scale-out` is added by TRL only for vllm >= 0.30.0 (our pin).
VLLM_TRAINING_SERVER_FLAGS: tuple[str, ...] = (
    "--weight-transfer-config",
    '{"backend":"nccl"}',
    "--logprobs-mode",
    "processed_logprobs",
    "--max-logprobs",
    "-1",
    "--enable-scale-out",
    # Sample from the model's own distribution. Without this vLLM applies the checkpoint's `generation_config.json`
    # (Qwen3-4B: temperature 0.6, top_k 20, top_p 0.95) to every request field the client leaves unset
    # (vllm 0.30.0 `ModelConfig.get_diff_sampling_param`, `--generation-config` in `engine/arg_utils.py`).
    "--generation-config",
    "vllm",
)


# Endpoints the trainer needs (weight transfer, prefix-cache reset) live behind vLLM's dev mode.
@dataclass(frozen=True)
class FamilySpec:
    family: ModelFamily
    lora_target_modules: str | list[str]
    # Non-thinking mode: long `<think>` blocks would eat the multi-turn `max_completion_length` budget and push the
    # self-report out of the window.
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    # Templates (names of `trl.chat_template_utils` attributes) GRPOTrainer may legitimately render multi-turn rollouts
    # with: a stock hub template that passes TRL's prefix-preservation check (so TRL keeps it), or TRL's training swap.
    # A stock template that is NOT prefix-preserving must never appear here: if it were in effect, TRL's tool-suffix
    # slicing would be wrong.
    stock_templates: tuple[str, ...] = ()
    training_templates: tuple[str, ...] = ()
    vllm_model_flags: tuple[str, ...] = ()  # needed by the training server
    vllm_tool_call_parser: str = "hermes"  # eval-only (OpenAI chat endpoint)
    vllm_reasoning_parser: str = "qwen3"  # eval-only
    needs_flash_linear_attention: bool = False

    def vllm_serve_flags(self, *, for_eval: bool = False) -> list[str]:
        """Model-specific `vllm serve` flags. Training needs `VLLM_TRAINING_SERVER_FLAGS` in addition."""
        flags = list(self.vllm_model_flags)
        if for_eval:
            flags += [
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                self.vllm_tool_call_parser,
                "--reasoning-parser",
                self.vllm_reasoning_parser,
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ]
        return flags


FAMILIES: dict[ModelFamily, FamilySpec] = {
    ModelFamily.QWEN3: FamilySpec(
        family=ModelFamily.QWEN3,
        lora_target_modules=list(_QWEN3_LORA_TARGETS),
        # Hybrid Qwen3 (`qwen3_chat_template`, e.g. Qwen3-4B) is not prefix-preserving (the last assistant turn gets an
        # empty `<think></think>` block that disappears once a tool message follows), so TRL swaps it for
        # `qwen3_training_chat_template`. Only the 2507 instruct template survives unswapped.
        stock_templates=("qwen3_instruct_2507_chat_template",),
        training_templates=(
            "qwen3_training_chat_template",
            "qwen3_instruct_2507_training_chat_template",
        ),
        vllm_tool_call_parser="hermes",
        vllm_reasoning_parser="qwen3",
    ),
    ModelFamily.QWEN3_5: FamilySpec(
        family=ModelFamily.QWEN3_5,
        lora_target_modules=_QWEN3_5_LORA_REGEX,
        # Qwen3.5-4B ships the "think" template (thinking on by default); <=2B ship "nothink". We pass
        # enable_thinking=False either way (trl/chat_templates/README.md).
        stock_templates=(
            "qwen3_5_think_chat_template",
            "qwen3_5_nothink_chat_template",
        ),
        training_templates=(
            "qwen3_5_think_training_chat_template",
            "qwen3_5_nothink_training_chat_template",
        ),
        # Skip the vision tower in vLLM (weights + encoder-cache memory). UNVERIFIED: that vLLM silently ignores the
        # `model.visual.*` tensors TRL still pushes during weight sync in this mode — check in the smoke test.
        vllm_model_flags=("--language-model-only",),
        vllm_tool_call_parser="qwen3_coder",
        vllm_reasoning_parser="qwen3",
        needs_flash_linear_attention=True,
    ),
}

_MODEL_TYPES = {
    "qwen3": ModelFamily.QWEN3,
    "qwen3_5": ModelFamily.QWEN3_5,
    "qwen3_5_text": ModelFamily.QWEN3_5,
}


def family_from_model_type(model_type: str) -> ModelFamily:
    try:
        return _MODEL_TYPES[model_type]
    except KeyError:
        raise ValueError(f"Unsupported model_type {model_type!r}; expected one of {sorted(_MODEL_TYPES)}") from None


def _family_from_name(name: str) -> ModelFamily | None:
    n = name.lower()
    if re.search(r"qwen3[._]5(?!\d)", n):
        return ModelFamily.QWEN3_5
    if re.search(r"qwen3(?![._]?\d)", n):
        return ModelFamily.QWEN3
    return None


def detect_family(model: str, config: Any = None, *, allow_download: bool = False) -> ModelFamily:
    """Resolve the family from a config's `model_type`, else the locally cached config, else the model name.

    Never touches the network unless `allow_download` (so `--dry-run` works offline).
    """
    if config is None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model, revision=pinned_revision(model), local_files_only=not allow_download
            )
        except Exception:  # noqa: BLE001 — not cached / offline: fall back to the name
            config = None
    if config is not None:
        return family_from_model_type(config.model_type)
    fam = _family_from_name(model)
    if fam is None:
        raise ValueError(f"Cannot infer model family for {model!r}; set model.family explicitly.")
    return fam


def family_spec(family: ModelFamily | str) -> FamilySpec:
    return FAMILIES[ModelFamily(family)]


def lora_target_modules(family: ModelFamily | str) -> str | list[str]:
    t = family_spec(family).lora_target_modules
    return list(t) if isinstance(t, list) else t


def flash_linear_attention_available() -> bool:
    """Gated DeltaNet without `fla` falls back to a pure-torch path (~20x slower training per TRL docs)."""
    import importlib.util

    return importlib.util.find_spec("fla") is not None


def template_name(chat_template: str | None) -> str | None:
    """Name of the `trl.chat_template_utils.*_chat_template` constant equal to `chat_template` (None if unknown)."""
    if chat_template is None:
        return None
    from trl import chat_template_utils as ctu

    for name in dir(ctu):
        if name.endswith("_chat_template") and getattr(ctu, name) == chat_template:
            return name
    return None


def check_training_template(trainer: Any, family: ModelFamily | str, *, strict: bool = True) -> str | None:
    """Log (and, if `strict`, assert) which chat template GRPOTrainer renders multi-turn rollouts with.

    With tools, GRPOTrainer swaps in TRL's training template at init only if the tokenizer's template fails TRL's
    prefix-preservation check (`trainer.chat_template`, else None = tokenizer template used as-is). In trl 1.14.1
    (checked on real tokenizers): hybrid Qwen3 (Qwen3-4B) fails the check and is swapped to
    `qwen3_training_chat_template`; the Qwen3.5 think/nothink templates pass it and are kept. What must hold is that
    the effective template is one TRL knows for this family: its response parser (`add_response_schema`) is chosen by
    exact template match, and an unknown hub template means tool calls may silently fail to parse.
    """
    spec = family_spec(family)
    swapped = getattr(trainer, "chat_template", None)
    effective = (
        swapped if swapped is not None else getattr(getattr(trainer, "processing_class", None), "chat_template", None)
    )
    name = template_name(effective)
    accepted = spec.training_templates + spec.stock_templates
    if name in accepted:
        how = "TRL training swap" if swapped is not None else "stock template, prefix-preserving, no swap"
        logger.info("[%s] rollout chat template: %s (%s)", spec.family, name, how)
        return name
    msg = f"[{spec.family}] rollout chat template is {name or 'unrecognised'}; expected one of {accepted}."
    if strict:
        raise RuntimeError(msg + " Set model.strict_template=false to override.")
    logger.warning(msg)
    return name


# ── sampling fidelity ────────────────────────────────────────

# What a policy-gradient rollout must be sampled with: the full distribution at the trainer's temperature. These are
# the `trl.GRPOConfig` field names; trl 1.14.1 `VLLMClient.generate` sends every one of them in each request
# (`repetition_penalty`, `temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`, `logprobs`) and then applies
# `generation_kwargs` on top, so a stray key there silently wins.
FULL_DISTRIBUTION_SAMPLING: dict[str, Any] = {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
# The only `generation_kwargs` key a training config may set: the per-turn cap. It must be `max_tokens`: that is the
# name of the field in the request body (and of `vllm.SamplingParams` in colocate mode), so it overrides TRL's own
# `max_tokens = max_completion_length`; a `max_new_tokens` key would be ignored by the server and leave the cap at 16k.
GENERATION_KWARGS_ALLOWED: frozenset[str] = frozenset({"max_tokens"})
# vLLM replaces -inf with this in JSON responses (vllm 0.30.0 `entrypoints/openai/completion/serving.py`).
VLLM_MASKED_LOGPROB = -9999.0

HttpFn = Callable[[str, str, dict | None], dict]


class VLLMProbeError(RuntimeError):
    """The generation server would give the trainer rollouts it cannot learn from; refuse to start."""


def check_full_distribution_sampling(grpo: dict[str, Any]) -> None:
    """Reject a `grpo:` section that truncates or reshapes the sampling distribution.

    The loss recomputes log-probabilities under the untruncated policy; rollouts drawn from a truncated one make the
    importance ratio wrong for every token that survives truncation (the guide measured 0.985–0.993 with top_p 0.95
    against 0.9984–0.9999 at 1.0).
    """
    for name, want in FULL_DISTRIBUTION_SAMPLING.items():
        got = grpo.get(name, want)
        if got is None and name == "min_p":  # TRL sends 0.0 for None
            continue
        if got is None or float(got) != float(want):
            raise ValueError(
                f"grpo.{name}={got!r} truncates or reshapes the sampling distribution; training requires "
                f"{name}={want!r} (docs/design.md section 8)"
            )
    temperature = grpo.get("temperature", 1.0)
    if not isinstance(temperature, int | float) or temperature <= 0:
        raise ValueError(f"grpo.temperature must be > 0 for sampled rollouts, got {temperature!r}")
    extra = set(grpo.get("generation_kwargs") or {}) - GENERATION_KWARGS_ALLOWED
    if extra:
        raise ValueError(
            f"grpo.generation_kwargs may only set the per-turn cap {sorted(GENERATION_KWARGS_ALLOWED)}; "
            f"{sorted(extra)} would override the sampling parameters TRL sends (docs/design.md section 8)"
        )


def vllm_max_model_len(max_completion_length: int, turn_max_tokens: int, prompt_token_reserve: int) -> int:
    """Context window the generation server needs.

    Each tool-loop request carries prompt + transcript so far (at most `max_completion_length`, tool output included)
    and asks for `turn_max_tokens` more; vLLM rejects a request whose input exceeds `max_model_len - max_tokens`, and TRL's client raises on any non-200.
    """
    return int(prompt_token_reserve) + int(max_completion_length) + int(turn_max_tokens)


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        raise VLLMProbeError(f"{method} {url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise VLLMProbeError(f"{method} {url} failed: {e}") from e


def _finite_top_logprobs(choice: dict) -> list[int]:
    """Per generated position: how many returned alternatives the sampler could actually have drawn."""
    top = (choice.get("logprobs") or {}).get("top_logprobs") or []
    return [sum(1 for v in (pos or {}).values() if v is not None and v > VLLM_MASKED_LOGPROB + 1.0) for pos in top]


def probe_vllm_server(
    base_url: str,
    *,
    required_max_model_len: int,
    temperature: float = 1.0,
    probe_logprobs: int = 40,
    probe_tokens: int = 4,
    prompt: str = "The following is a list of common Linux shell commands and what they do:\n",
    http: HttpFn | None = None,
) -> dict[str, Any]:
    """Startup checks against the running generation server; raises `VLLMProbeError`.

    1. `/v1/models` reports a context window of at least `required_max_model_len`.
    2. A completion sent exactly as TRL sends it returns `prompt_token_ids`, `token_ids` and per-token logprobs keyed
       by `token_id:<id>`.
    3. The logprobs are *processed* ones: with an explicit `top_k = 3` the 4th and later alternatives must come back
       masked. A server in `raw_logprobs` mode returns them all finite, and TRL's importance ratio would then compare
       the wrong quantity.
    4. The distribution is not truncated: with the training parameters (`top_p 1, top_k 0, min_p 0`), and again with
       those fields left out so that the server's defaults apply, all `probe_logprobs` alternatives are finite. A
       server started without `--generation-config vllm` fails the second request for a model whose
       `generation_config.json` sets `top_k` or `top_p` (Qwen3-4B does).

    Checked against vllm 0.30.0 sources, not against a running server (UNVERIFIED on GPU). `http(method, url, payload)`
    is injectable for tests.
    """
    http = http or _http_json
    base = base_url.rstrip("/")
    models = http("GET", f"{base}/v1/models", None).get("data") or []
    if not models:
        raise VLLMProbeError(f"{base}/v1/models lists no model")
    card = models[0]
    served = card.get("max_model_len")
    if served is None:
        raise VLLMProbeError(f"{base}/v1/models does not report max_model_len; cannot check the context window")
    if int(served) < int(required_max_model_len):
        raise VLLMProbeError(
            f"vLLM serves max_model_len={served} but this run needs >= {required_max_model_len} "
            "(prompt reserve + max_completion_length + per-turn max_tokens). Restart scripts/serve_vllm.sh with the "
            "same config, or lower the budget."
        )

    def complete(**sampling: Any) -> dict:
        payload = {
            "model": card["id"],
            "prompt": [prompt],
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "n": 1,
            "max_tokens": probe_tokens,
            "temperature": temperature,
            "logprobs": probe_logprobs,
            **sampling,
        }
        choices = http("POST", f"{base}/v1/completions", payload).get("choices") or []
        if not choices:
            raise VLLMProbeError("vLLM returned no choices for the probe completion")
        return choices[0]

    full = dict(FULL_DISTRIBUTION_SAMPLING)
    explicit = complete(**full)
    ids, prompt_ids = explicit.get("token_ids"), explicit.get("prompt_token_ids")
    if not ids or not prompt_ids:
        raise VLLMProbeError(
            "vLLM did not return token ids (`token_ids` / `prompt_token_ids`); the trainer learns from sampled ids, "
            "never from re-tokenised text"
        )
    logprobs = explicit.get("logprobs") or {}
    tokens, token_logprobs = logprobs.get("tokens") or [], logprobs.get("token_logprobs") or []
    if len(tokens) != len(ids) or len(token_logprobs) != len(ids):
        raise VLLMProbeError("vLLM did not return one logprob per sampled token")
    if any(not str(t).startswith("token_id:") for t in tokens):
        raise VLLMProbeError("vLLM logprob tokens are not `token_id:<id>` strings (return_tokens_as_token_ids ignored)")
    if any(
        lp is None or not math.isfinite(lp) or lp <= VLLM_MASKED_LOGPROB + 1.0 or lp > 1e-6 for lp in token_logprobs
    ):
        raise VLLMProbeError(f"vLLM returned invalid sampled-token logprobs: {token_logprobs}")

    truncated = _finite_top_logprobs(complete(**{**full, "top_k": 3}))
    if not truncated or max(truncated) > 3:
        raise VLLMProbeError(
            f"with top_k=3 the server still reports {max(truncated, default=0)} finite alternatives per token: it is "
            "not returning processed logprobs. Start it with `--logprobs-mode processed_logprobs --max-logprobs -1`."
        )

    def check_full(choice: dict, what: str, hint: str) -> list[int]:
        finite = _finite_top_logprobs(choice)
        if not finite:
            raise VLLMProbeError(f"vLLM returned no top logprobs ({what}); is `--max-logprobs -1` set?")
        if min(finite) < probe_logprobs:
            raise VLLMProbeError(
                f"the server truncates the sampling distribution ({what}): only {min(finite)} of {probe_logprobs} "
                f"alternatives are sampleable at some position. {hint}"
            )
        return finite

    check_full(explicit, "training parameters", "Check for a logits processor or a proxy rewriting the request.")
    check_full(
        complete(),
        "server defaults",
        "Start the server with `--generation-config vllm` so the checkpoint's generation_config.json is not applied.",
    )
    result = {"model": card["id"], "max_model_len": int(served), "required_max_model_len": int(required_max_model_len)}
    logger.info("vLLM probe passed: %s", result)
    return result


# ── chat template: prefix preservation on our tool flow ─────


def _flow_ids(processing_class: Any, messages: list[dict], *, tools: Any, chat_template: str | None,
              add_generation_prompt: bool, chat_template_kwargs: dict[str, Any]) -> list[int]:  # fmt: skip
    ids = processing_class.apply_chat_template(
        messages,
        tools=tools or None,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
        **chat_template_kwargs,
    )
    if ids and isinstance(ids[0], list):  # processors return a batch of one
        ids = ids[0]
    return list(ids)


def tool_flow_messages(system_prompt: str = "You are an agent.", instruction: str = "List the files.") -> list[dict]:
    """A whole episode in our harness: bash call, its output, the final report call, the report acknowledgement."""
    report = {
        "status": "incomplete",
        "completed": ["listed the directory"],
        "incomplete": ["nothing else was asked"],
        "blocker_category": "missing_capability",
        "blocker_target": "/usr/bin/jq",
        "blocker_detail": "jq is not installed",
        "evidence_commands": ["test -d /app"],
        "confidence": 0.5,
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls -la /app"}}}]},
        {"role": "tool", "name": "bash", "content": "[exit code: 0]\ntotal 0\n-rw-r--r-- 1 root root 0 a.txt"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "submit_report", "arguments": report}}]},
        {"role": "tool", "name": "submit_report", "content": "Report received. The episode is over; stop calling tools."},
    ]  # fmt: skip


def tool_flow_prefix_failures(
    processing_class: Any,
    *,
    tools: Sequence[Any] | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    messages: list[dict] | None = None,
) -> list[str]:
    """Where the template re-renders earlier turns differently once a tool result follows (empty list = preserving).

    TRL builds each tool-loop prompt by concatenating token ids: prompt + sampled completion + a tool suffix cut from a
    rendering of `[.., assistant(tool_calls), tool]` against `[.., assistant(tool_calls)]`. That is only the text the
    model would see if the template renders a conversation ending in an assistant tool call as an exact prefix of the
    same conversation with the tool result (and generation prompt) appended. TRL checks this once with dummy messages
    and no tools; this checks it with our tools, system prompt, argument shapes and both tool-result positions.
    """
    kw = {
        "tools": list(tools) if tools else None,
        "chat_template": chat_template,
        "chat_template_kwargs": dict(chat_template_kwargs or {}),
    }
    messages = messages or tool_flow_messages()
    failures = []
    for end in range(len(messages)):
        if messages[end]["role"] != "tool":
            continue
        before = _flow_ids(processing_class, messages[:end], add_generation_prompt=False, **kw)
        after = _flow_ids(processing_class, messages[: end + 1], add_generation_prompt=True, **kw)
        if after[: len(before)] != before:
            n = next(
                (i for i, (a, b) in enumerate(zip(before, after, strict=False)) if a != b), min(len(before), len(after))
            )
            failures.append(
                f"turn ending at message {end - 1} ({messages[end - 1]['role']}) is not a prefix of the conversation "
                f"with the `{messages[end].get('name')}` result appended (first difference at token {n} of {len(before)})"
            )
    return failures


def check_tool_flow_template(trainer: Any, *, strict: bool = True) -> list[str]:
    """Assert, with the trainer's real tokenizer, tools and effective template, that our tool flow is prefix-preserving."""
    failures = tool_flow_prefix_failures(
        trainer.processing_class,
        tools=getattr(trainer, "tools", None),
        chat_template=getattr(trainer, "chat_template", None),
        chat_template_kwargs=getattr(trainer, "chat_template_kwargs", None),
    )
    if failures:
        msg = "chat template is not prefix-preserving for the tool flow: " + "; ".join(failures)
        if strict:
            raise RuntimeError(msg + " (model.strict_template=false to override)")
        logger.warning(msg)
    else:
        logger.info("chat template is prefix-preserving for the bash -> submit_report tool flow")
    return failures


# ── chat template: the turn's end token is the tokenizer's EOS ─────


def end_of_turn_failures(
    processing_class: Any,
    *,
    tools: Sequence[Any] | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Where the token that closes an assistant turn in the template is not the tokenizer's `eos_token_id`.

    TRL decides that a rollout was cut by the length limit when its last sampled id is neither the tokenizer's EOS nor
    its pad token, masks such rollouts from the loss (`mask_truncated_completions`) and our loop labels them
    `truncated`. A model that ends turns with `<|im_end|>` while the tokenizer's EOS is `<|endoftext|>` (a base
    checkpoint) would have every healthy turn counted as truncated and the whole run masked, so this is asserted at
    start with the real tokenizer and template, for a plain reply and for a tool call.
    """
    kw = {
        "tools": list(tools) if tools else None,
        "chat_template": chat_template,
        "add_generation_prompt": False,
        "chat_template_kwargs": dict(chat_template_kwargs or {}),
    }
    tokenizer = getattr(
        processing_class, "tokenizer", processing_class
    )  # a multimodal processor keeps eos on its tokenizer
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return ["the tokenizer has no eos_token_id"]
    flow = tool_flow_messages()
    cases = {
        "a plain reply": [*flow[:2], {"role": "assistant", "content": "done"}],
        "a tool call": flow[:3],
    }
    failures = []
    for label, messages in cases.items():
        ids = _flow_ids(processing_class, messages, **kw)
        while ids and not tokenizer.decode([ids[-1]]).strip():  # the newline the template adds after the end token
            ids.pop()
        if not ids or ids[-1] != eos:
            last = tokenizer.decode(ids[-1:]) if ids else "(nothing)"
            failures.append(
                f"the template ends {label} with {last!r} but the tokenizer's eos_token is "
                f"{tokenizer.decode([eos])!r} (id {eos})"
            )
    return failures


def check_end_of_turn_token(trainer: Any, *, strict: bool = True) -> list[str]:
    """Assert, with the trainer's real tokenizer and effective template, that a turn ends on the tokenizer's EOS."""
    failures = end_of_turn_failures(
        trainer.processing_class,
        tools=getattr(trainer, "tools", None),
        chat_template=getattr(trainer, "chat_template", None),
        chat_template_kwargs=getattr(trainer, "chat_template_kwargs", None),
    )
    if failures:
        msg = "end-of-turn token mismatch: " + "; ".join(failures)
        if strict:
            raise RuntimeError(msg + " (model.strict_template=false to override)")
        logger.warning(msg)
    else:
        logger.info("the chat template ends every assistant turn with the tokenizer's eos token")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m honest_agents.models revision NAME`: print NAME's pinned Hub revision (empty line if none)."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m honest_agents.models")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rev = sub.add_parser("revision", help="print the pinned Hub revision of a model id (empty for a local directory)")
    rev.add_argument("name")
    args = parser.parse_args(argv)
    print(pinned_revision(args.name) or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
