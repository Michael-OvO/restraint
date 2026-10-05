"""`HonestGRPOTrainer`: TRL's GRPO trainer with a batch-concurrent tool loop that stops at the report.

A drop-in subclass of `trl.GRPOTrainer` (same constructor). It changes two things in the multi-turn rollout and
nothing in the loss value (docs/design.md section 8):

* **Tool calls run concurrently across the batch.** TRL 1.14.1 executes one rollout's tool calls after another on
  the main thread, so 64 episodes of `docker compose exec` are paid for serially. Every environment owns its own
  event-loop thread, so calls on different environments are independent: each rollout's calls for a turn run on a
  worker thread, in that rollout's own order, and the results are written back in batch order on the main thread.
* **An episode that has reported is not regenerated.** The first report is final, so the turn TRL would sample after
  it is wasted compute and puts tokens in the loss that no reward depends on. A reported episode leaves the loop
  through the same rollback TRL uses for a tool result that does not fit: its token ids, logprobs and tool mask stay
  exactly as they were when the report turn was sampled, ending on that turn's end-of-turn token.

Everything else is TRL's `_tool_call_loop`, ported line for line: the tool mask, token accounting, the rollback of
tool results that exceed the budget, the truncation of an over-long last turn, and both counters. Sampled token ids
and sampling-time logprobs are only ever concatenated and sliced, never decoded and re-encoded.
`tests/test_trainer_loop.py` drives TRL's own loop and this one with the same fake generator and environments and
asserts the outputs are equal.

The port is only valid for the TRL source it was made from, so construction fails if the installed `_tool_call_loop`,
`_generate` or `_calculate_rewards` differ from the fingerprints below.

Also here, because only the trainer knows them:

* why each episode left the loop (`last_loop_exits`, the `loop_exit` completions column, `rollout/exit/<reason>`);
* where the rollout's wall-clock went (`rollout/*_seconds`), so the slowest tool call is visible in the first hour;
* the per-rollout `prompt_ids`, `logprobs` and `tool_mask`, handed to the reward function for the trajectory bank;
* the merged-LoRA weight-sync fix (`clone_synced_params`, docs/design.md section 8): the parameters TRL streams to vLLM are
  cloned before the adapter is un-merged;
* **padding-free scoring**: TRL pads every sequence of a generation batch to the batch-global longest
  (prompts on the left, completions on the right), so padded slots were 1.7-3.5x the real tokens. Every forward pass
  here (scoring the rollouts, and the loss) runs rows sorted by length, in chunks trimmed to their own longest row, and
  the results are scattered back to TRL's padded shape. Padded positions carry no attention and no loss, and RoPE
  depends only on relative positions, so the loss value and its gradients are unchanged up to float rounding
  (`tests/test_trainer_loop.py`). The micro-batches of a generation round are also regrouped by length, longest
  first, which is exact for the token-sum losses (dapo, cispo, vespo: one global normaliser) and is skipped for the
  others;
* sampler-trainer drift metrics TRL does not log (`sampling/importance_sampling_ratio/at_cap_share`, `outside_band_share`)
  and the padding ratios (`padding/*`).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from trl import GRPOTrainer
from trl.chat_template_utils import parse_response
from trl.trainer.utils import split_pixel_values_by_grid, split_tensor_dict, unsplit_pixel_values_by_grid

from honest_agents.harness import SANDBOX_UNAVAILABLE
from honest_agents.reward import (
    STOP_CALL_BUDGET,
    STOP_CHAR_BUDGET,
    STOP_REASONS,
    STOP_REPORT,
    STOP_SILENT,
    STOP_TRUNCATED,
)

logger = logging.getLogger(__name__)


class _UnknownTool(ValueError):
    """The model named a tool that does not exist; its own text is safe to show back (it only contains the name)."""


INVALID_TOOL_ARGS = "[invalid tool arguments]"


def _tool_error_text(exc: BaseException) -> str:
    """What the model sees for a tool call that raised. Only a fixed string, except for an unknown tool name: the real
    exception can carry host paths, compose commands or the task / session name, so it is logged host-side instead."""
    if isinstance(exc, _UnknownTool):
        return str(exc)
    logger.error("tool call failed", exc_info=exc)
    return INVALID_TOOL_ARGS if isinstance(exc, TypeError) else SANDBOX_UNAVAILABLE


# sha256 of the TRL 1.14.1 sources this module ports or relies on (`inspect.getsource` of the undecorated method).
UPSTREAM_FINGERPRINTS: dict[str, str] = {
    "_tool_call_loop": "acc9b1db6fed3d92f75e9068effc4d2042af71a6807537b9c3ce243c130138b7",
    "_generate": "1a491d69449de2fa291227f93ba3f2e4e673f9f3197d7b63a10a860414e52f57",
    "_calculate_rewards": "736ae29b083840b5da2638b55ad1258659c644589991fe6a494131354952efef",
    # Not ported, but the padding trimmer calls it with sliced inputs and relies on its argument semantics.
    "_get_per_token_logps_and_entropies": "dc17fee20840b7ebfc65648b3464f007f41dd846a472c7ae36a864c342e264c5",
}

# Why a rollout left the tool loop, as the trainer saw it. The values are `reward.STOP_REASONS`:
#   report       the environment recorded a report (stop_at_report), or had one when the loop ended
#   truncated    the last sampled turn does not end on EOS; TRL's `mask_truncated_completions` drops it from the loss
#   char_budget  the tool result did not fit the completion budget and was rolled back
#   call_budget  `max_tool_calling_iterations` was reached with tool calls still pending (they are not executed)
#   silent_stop  the last turn ended normally without a tool call
LOOP_EXITS: tuple[str, ...] = STOP_REASONS
COLUMN_LOOP_EXIT = "loop_exit"  # per-episode column in TRL's completions table
METRIC_EXIT_PREFIX = "rollout/exit/"  # + reason: share of the batch's episodes
METRIC_ROLLOUT_SECONDS = "rollout/seconds"  # the whole rollout: every generation turn plus every tool call
METRIC_GENERATION_SECONDS = "rollout/generation_seconds"  # all turns, the first one included
METRIC_TOOL_SECONDS = "rollout/tool_seconds"  # wall-clock spent waiting for tools, summed over turns
METRIC_SLOWEST_TOOL_TURN = "rollout/slowest_tool_turn_seconds"  # the slowest single turn of any one episode
METRIC_SLOWEST_EPISODE_TOOLS = "rollout/slowest_episode_tool_seconds"  # the largest per-episode total
METRIC_TOOL_TURNS = "rollout/tool_turns"  # rounds of tool execution in this batch
METRIC_REGENERATED = "rollout/regenerated_turns"  # turns sampled inside the loop, summed over episodes

ENV_EXIT_HOOK = "_set_loop_exit"  # optional environment method: `_set_loop_exit(reason: str) -> None`

# Metrics of the padding trimmer and of sampler-trainer drift. TRL already logs `sampling/importance_sampling_ratio/
# {min,mean,max}`, `completions/clipped_ratio` (generations cut by the length limit) and `clip_ratio/*` (PPO clipping);
# these are the ones it does not.
METRIC_PAD_STOCK = "padding/stock_slots_per_token"  # token slots TRL's batch-global padding would run / real tokens
METRIC_PAD_TRIMMED = "padding/trimmed_slots_per_token"  # token slots actually run / real tokens
METRIC_IS_AT_CAP = "sampling/importance_sampling_ratio/at_cap_share"  # loss tokens whose ratio hit the truncation cap
METRIC_IS_OUT_OF_BAND = "sampling/importance_sampling_ratio/outside_band_share"  # loss tokens with |ratio - 1| > band
IS_BAND = 0.05  # the drift band for the batch-mean ratio, reused per token as a coarse share
_LOSS_NORMALISED_GLOBALLY = frozenset(
    {"dapo", "cispo", "vespo"}
)  # `num_items_in_batch` normaliser: regrouping is exact


def row_extents(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row, the first attended column and one past the last (0, 0 for a row with no attended token), on the CPU."""
    attended = attention_mask.detach().cpu() != 0
    width = attended.size(1)
    has_any = attended.any(dim=1)
    first = attended.int().argmax(dim=1)
    end = width - attended.flip(1).int().argmax(dim=1)
    return torch.where(has_any, first, 0), torch.where(has_any, end, 0)


def plan_chunks(
    attention_mask: torch.Tensor, logits_to_keep: int, batch_size: int | None = None, sort: bool = True
) -> list[tuple[torch.Tensor, int, int]]:
    """How to run a padded batch as forward passes of `batch_size` rows: `(rows, lo, hi)` per pass, columns `[lo, hi)`.

    TRL's layout is `[left-padded prompt | right-padded completion]`, so the window of a chunk is from its earliest
    attended token to its latest, and it always keeps at least one completion column (`logits_to_keep >= 1`). Sorting
    puts the longest rows first (an out-of-memory error then shows up on the first chunk, not in the middle of a
    step) and makes chunk-mates similar in length.
    """
    first, end = row_extents(attention_mask)
    rows = attention_mask.size(0)
    width = attention_mask.size(1)
    order = torch.argsort(first - end, stable=True) if sort else torch.arange(rows)  # most attended tokens first
    size = batch_size or rows
    completion_start = width - logits_to_keep
    chunks = []
    for start in range(0, rows, size):
        picked = order[start : start + size]
        lo = int(first[picked].min())
        hi = max(int(end[picked].max()), completion_start + 1)
        chunks.append((picked, min(lo, completion_start), hi))
    return chunks


def padded_slots(attention_mask: torch.Tensor, chunks: list[tuple[torch.Tensor, int, int]] | None = None) -> int:
    """Token positions a plan runs (rows x window per chunk); `None` is TRL's batch-global padding (one window)."""
    if chunks is None:
        return attention_mask.size(0) * attention_mask.size(1)
    return sum(len(rows) * (hi - lo) for rows, lo, hi in chunks)


def sort_micro_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of a generation round regrouped into the same number of micro-batches, longest first.

    Returns the input unchanged when it holds anything but row-aligned tensors and scalars (multimodal lists).
    """
    if len(batches) < 2:
        return batches
    keys = batches[0].keys()
    for batch in batches:
        for value in batch.values():
            if value is not None and not isinstance(value, torch.Tensor):
                return batches
    merged: dict[str, Any] = {}
    for key in keys:
        values = [b[key] for b in batches]
        if any(v is None for v in values):
            merged[key] = None
        elif values[0].ndim == 0:
            merged[key] = values[0]
        else:
            merged[key] = torch.cat(values, dim=0)
    first, end = row_extents(torch.cat([merged["prompt_mask"], merged["completion_mask"]], dim=1))
    order = torch.argsort(first - end, stable=True)
    sorted_rows = {
        key: (value if value is None or value.ndim == 0 else value[order.to(value.device)])
        for key, value in merged.items()
    }
    return split_tensor_dict(sorted_rows, len(batches))


def upstream_fingerprints() -> dict[str, str]:
    """sha256 of the installed TRL methods named in `UPSTREAM_FINGERPRINTS`."""
    out = {}
    for name in UPSTREAM_FINGERPRINTS:
        source = inspect.getsource(inspect.unwrap(getattr(GRPOTrainer, name)))
        out[name] = hashlib.sha256(source.encode()).hexdigest()
    return out


def check_upstream() -> None:
    """Refuse to run against a TRL whose rollout code is not the one this module was ported from.

    The tool loop is a copy of TRL's. If TRL changes its loop, or what `_generate` does with the loop's output, this
    copy would keep the old behaviour without any error, which is the kind of silent divergence that wastes a run.
    """
    try:
        found = upstream_fingerprints()
    except (OSError, TypeError) as e:  # no source files: fall back to the version pin
        import trl

        if trl.__version__ != "1.14.1":
            raise RuntimeError(
                f"HonestGRPOTrainer ports the tool loop of trl 1.14.1; found {trl.__version__} without sources ({e})"
            ) from e
        return
    changed = sorted(name for name, digest in UPSTREAM_FINGERPRINTS.items() if found[name] != digest)
    if changed:
        import trl

        raise RuntimeError(
            f"trl {trl.__version__}: GRPOTrainer.{', GRPOTrainer.'.join(changed)} differ from the trl 1.14.1 source "
            "that honest_agents.trainer ports or wraps. Re-port `_tool_call_loop` (or re-check the padding trimmer "
            "against the changed scoring code), rerun tests/test_trainer_loop.py, then update UPSTREAM_FINGERPRINTS."
        )


def clone_synced_params(vllm_generation: Any) -> None:
    """Make `vllm_generation._iter_named_params` yield clones.

    TRL merges the LoRA adapter, yields views of the merged weights and un-merges inside the same generator. vLLM's
    packed weight transfer copies the last chunk only after the generator is exhausted, by which time those views hold
    the un-merged base weights. A clone taken at the `yield` keeps the merged value. Idempotent.
    """
    original = vllm_generation._iter_named_params
    if getattr(original, "_honest_clones", False):
        return

    def _iter_cloned_params():
        for name, tensor in original():
            yield name, tensor.detach().clone()

    _iter_cloned_params._honest_clones = True  # type: ignore[attr-defined]
    vllm_generation._iter_named_params = _iter_cloned_params


class _Tokens(list):
    """A per-rollout token list with a short repr.

    TRL prints every reward keyword argument of a masked row in a warning; a 16k-token episode would add three lines
    of several hundred kilobytes to the log each time an episode is masked.
    """

    def __repr__(self) -> str:
        return f"<{len(self)} values>"


def _unreported(values: list, reported: list[bool]) -> list:
    return [value for value, done in zip(values, reported, strict=True) if not done]


@dataclass
class LoopRecord:
    """What one call of the tool loop did, for logging. Lists are in batch order."""

    exits: list[str] = field(default_factory=list)
    tool_seconds: float = 0.0
    slowest_tool_turn: float = 0.0
    episode_tool_seconds: list[float] = field(default_factory=list)
    tool_turns: int = 0
    regenerated_turns: int = 0


class HonestGRPOTrainer(GRPOTrainer):
    """`trl.GRPOTrainer` with concurrent tool execution and no generation after an episode's report.

    Settings are attributes, so the constructor stays TRL's. Set them on the instance after construction:

    * `stop_at_report` (default True): an episode whose environment has recorded a report leaves the loop without
      another generation. False reproduces TRL's loop exactly (kept for the differential test).
    * `max_tool_workers` (default None = one thread per active episode): cap on concurrent tool-calling threads.
    * `trim_padding`, `sort_micro_batches` (default True): the padding-free scoring described in the module docstring.

    An environment "has reported" when its `_report` or `_report_error` is set. A report tool call
    that failed before reaching the environment (bad arguments) therefore does not end the episode: the model sees
    the error and can report again.

    Tools not bound to the rollout's own environment (TRL's `tools=` argument) may be shared between rollouts, so
    calls to them are serialised with a lock, in no fixed order across rollouts. Without any environment the loop
    runs tools one rollout after another, exactly as TRL does.
    """

    stop_at_report: bool = True
    max_tool_workers: int | None = None
    clone_synced_params: bool = True  # docs/design.md section 8; set False before construction to reproduce stock TRL
    trim_padding: bool = True  # run every forward in length-trimmed chunks; False = TRL's padded layout
    sort_micro_batches: bool = True  # also regroup a generation round's micro-batches by length (token-sum losses only)

    def __init__(self, *args: Any, **kwargs: Any):
        check_upstream()
        super().__init__(*args, **kwargs)
        self._check_generation_layout()
        vllm_generation = getattr(self, "vllm_generation", None)
        if vllm_generation is not None and self.clone_synced_params:
            clone_synced_params(vllm_generation)
        self.last_loop_exits: list[str] | None = None

    # ── guards ───────────────────────────────────────────────────────────────

    def _check_generation_layout(self) -> None:
        """Reject layouts in which vLLM hands a rank the wrong completions.

        In server mode, and in colocate mode with tensor parallelism, `VLLMGeneration.generate` gathers every rank's
        prompts and gives rank r the slice `[r * n_local, (r + 1) * n_local]`. Inside the tool loop each rank sends
        only its still-active episodes, so `n_local` differs between ranks from the second turn on and a rank
        receives completions sampled for another rank's contexts. One training process per run is the supported
        layout.
        """
        if not getattr(self, "use_vllm", False):
            return
        self._check_turn_cap()
        if self.accelerator.num_processes <= 1:
            return
        shared = self.vllm_mode == "server" or getattr(self, "vllm_tensor_parallel_size", 1) > 1
        if shared and self.tools:
            raise ValueError(
                f"{self.accelerator.num_processes} training processes share one vLLM ({self.vllm_mode} mode): TRL "
                "slices the gathered completions by each rank's own count of active episodes, which mis-routes them "
                "inside the tool loop. Train with one process per run (launch.num_processes: 1)."
            )

    def _check_turn_cap(self) -> None:
        """Require a per-turn `max_tokens` when a vLLM server is used with tools.

        TRL asks the server for `max_tokens = max_completion_length` on every turn. vLLM's OpenAI route rejects a
        request whose prompt is longer than `max_model_len - max_tokens`, so with the whole episode budget as the cap
        the first continuation past `max_model_len - max_completion_length` tokens fails the step. A per-turn cap in
        `generation_kwargs` overrides it. `train/` sets and sizes it; this stops a hand-built config from missing it.
        """
        if getattr(self, "vllm_mode", None) != "server" or not self.tools:
            return
        turn = (getattr(self.args, "generation_kwargs", None) or {}).get("max_tokens")
        if turn is None or not 0 < int(turn) <= self.max_completion_length:
            raise ValueError(
                f"vLLM server mode with tools needs generation_kwargs['max_tokens'] in (0, {self.max_completion_length}]"
                f" (the per-turn cap); got {turn!r}. Without it every turn requests the whole episode budget and the "
                "server rejects the request once the transcript passes max_model_len - max_completion_length."
            )

    # ── rollout bookkeeping around TRL's `_generate` ─────────────────────────

    def _generate_single_turn(self, *args: Any, **kwargs: Any):
        start = time.perf_counter()
        try:
            return super()._generate_single_turn(*args, **kwargs)
        finally:
            self._generation_seconds = getattr(self, "_generation_seconds", 0.0) + time.perf_counter() - start

    def _generate(self, prompts: list):
        self._loop_record: LoopRecord | None = None
        self._generation_seconds = 0.0
        start = time.perf_counter()
        output = super()._generate(prompts)
        elapsed = time.perf_counter() - start
        prompt_ids, _, tool_mask, _, logprobs = output[:5]
        # Kept for `_calculate_rewards`, which TRL calls next with the same batch.
        self._rollout_tokens = {
            "prompt_ids": prompt_ids,
            "logprobs": logprobs,
            "tool_mask": tool_mask,
        }

        metrics = self._metrics["train" if self.model.training else "eval"]
        metrics[METRIC_ROLLOUT_SECONDS].append(elapsed)
        metrics[METRIC_GENERATION_SECONDS].append(self._generation_seconds)
        record = self._loop_record
        self.last_loop_exits = list(record.exits) if record is not None else None
        if record is not None:
            metrics[METRIC_TOOL_SECONDS].append(record.tool_seconds)
            metrics[METRIC_SLOWEST_TOOL_TURN].append(record.slowest_tool_turn)
            metrics[METRIC_SLOWEST_EPISODE_TOOLS].append(max(record.episode_tool_seconds, default=0.0))
            metrics[METRIC_TOOL_TURNS].append(float(record.tool_turns))
            metrics[METRIC_REGENERATED].append(float(record.regenerated_turns))
            n = len(record.exits)
            for reason in LOOP_EXITS:
                metrics[METRIC_EXIT_PREFIX + reason].append(record.exits.count(reason) / n if n else 0.0)
            self._log_completion_extra(COLUMN_LOOP_EXIT, list(record.exits))
            self._tell_environments(record.exits)
        return output

    def _tell_environments(self, exits: list[str]) -> None:
        """Offer each environment the trainer-side exit reason before it is graded.

        Only the trainer can tell a generation cut by the length limit from a silent stop. An environment that wants
        to know defines `_set_loop_exit(reason)`; a failure there must not stop training.
        """
        for environment, reason in zip(self.environments or [], exits, strict=False):
            hook = getattr(environment, ENV_EXIT_HOOK, None)
            if not callable(hook):
                continue
            try:
                hook(reason)
            except Exception:
                logger.exception("environment %s(%r) failed", ENV_EXIT_HOOK, reason)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """TRL's reward call, with the rollout's `prompt_ids`, `logprobs` and `tool_mask` added per row.

        TRL builds the reward keyword arguments from the keys of the dataset rows, so the token lists are added to
        copies of the rows. They are the lists the loss is computed from, unmodified.
        """
        tokens = self.__dict__.pop("_rollout_tokens", None) or {}  # one use: they belong to this batch only
        extra = {
            key: values
            for key, values in tokens.items()
            if isinstance(values, list) and len(values) == len(inputs) and key not in inputs[0]
        }
        if extra:
            inputs = [
                {**row, **{key: _Tokens(values[i]) for key, values in extra.items()}} for i, row in enumerate(inputs)
            ]
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    # ── padding-free scoring ──────────────────────────────────────

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, *args, batch_size=None, compute_aux_loss=False, **kwargs
    ):
        """TRL's scoring on length-sorted chunks trimmed to their own window; same outputs, padded to TRL's shape.

        Positions outside a row's attended span come back as 0 (TRL's are finite junk); every consumer masks them.
        Multimodal inputs, the MoE auxiliary loss and `trim_padding = False` take TRL's path unchanged.
        """
        extras = {k: v for k, v in kwargs.items() if k != "compute_entropy" and v is not None}
        if not self.trim_padding or compute_aux_loss or extras or input_ids.size(0) == 0:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                *args,
                batch_size=batch_size,
                compute_aux_loss=compute_aux_loss,
                **kwargs,
            )
        chunks = plan_chunks(attention_mask, logits_to_keep, batch_size)
        order = torch.cat([rows for rows, _, _ in chunks])
        want_entropy = bool(kwargs.get("compute_entropy"))
        logps, entropies = [], []
        for rows, lo, hi in chunks:
            index = rows.to(input_ids.device)
            keep = hi - (input_ids.size(1) - logits_to_keep)  # completion columns inside the window: a prefix of them
            chunk_logps, chunk_entropies, _ = super()._get_per_token_logps_and_entropies(
                model,
                input_ids[index, lo:hi],
                attention_mask[index, lo:hi],
                keep,
                *args,
                batch_size=None,
                compute_aux_loss=False,
                **kwargs,
            )
            logps.append(torch.nn.functional.pad(chunk_logps, (0, logits_to_keep - keep)))
            if want_entropy:
                entropies.append(torch.nn.functional.pad(chunk_entropies, (0, logits_to_keep - keep)))
        undo = torch.argsort(order).to(input_ids.device)
        self._log_padding(attention_mask, chunks)
        return torch.cat(logps)[undo], (torch.cat(entropies)[undo] if want_entropy else None), None

    def _log_padding(self, attention_mask: torch.Tensor, chunks: list[tuple[torch.Tensor, int, int]]) -> None:
        real = max(1, int((attention_mask != 0).sum()))
        metrics = self._metrics["train" if self.model.training else "eval"]
        metrics[METRIC_PAD_STOCK].append(padded_slots(attention_mask) / real)
        metrics[METRIC_PAD_TRIMMED].append(padded_slots(attention_mask, chunks) / real)

    def _prepare_inputs(self, generation_batch):
        """TRL's `_prepare_inputs`; when it has just built a new round, regroup the round's micro-batches by length.

        A micro-batch of 2 then pairs rows of similar length instead of a random pair. Exact only where the loss has
        one normaliser for the whole round (`_LOSS_NORMALISED_GLOBALLY`) and an optimizer step covers whole rounds.
        """
        before = getattr(self, "_buffered_inputs", None)
        inputs = super()._prepare_inputs(generation_batch)
        buffered = getattr(self, "_buffered_inputs", None)
        if not (self.sort_micro_batches and self.model.training and buffered is not None and buffered is not before):
            return inputs
        accumulation = int(self.args.gradient_accumulation_steps)
        per_round = int(self.args.steps_per_generation)
        if self.loss_type not in _LOSS_NORMALISED_GLOBALLY or accumulation % per_round:
            return inputs
        self._buffered_inputs = sort_micro_batches(buffered)
        return self._buffered_inputs[self._step % per_round]

    def _compute_loss(self, model, inputs):
        self._log_sampling_drift(inputs)
        return super()._compute_loss(model, inputs)

    def _log_sampling_drift(self, inputs: dict[str, Any]) -> None:
        """Token shares of the vLLM importance ratio TRL does not log: at the truncation cap, and far from 1."""
        ratio = inputs.get("importance_sampling_ratio")
        if not self.model.training or ratio is None or ratio.ndim != 2 or ratio.size(1) == 1:
            return
        mask = (
            inputs["completion_mask"] if "tool_mask" not in inputs else inputs["completion_mask"] * inputs["tool_mask"]
        )
        count = mask.sum().clamp(min=1)
        cap = float(getattr(self.args, "vllm_importance_sampling_clip_max", float("inf")))
        metrics = self._metrics["train"]
        metrics[METRIC_IS_AT_CAP].append(((ratio >= cap - 1e-6) * mask).sum().div(count).item())
        metrics[METRIC_IS_OUT_OF_BAND].append((((ratio - 1).abs() > IS_BAND) * mask).sum().div(count).item())

    # ── tool execution ───────────────────────────────────────────────────────

    @staticmethod
    def _episode_reported(environment: Any) -> bool:
        """True once the environment has recorded its final report, parsed or not."""
        return (
            getattr(environment, "_report", None) is not None or getattr(environment, "_report_error", None) is not None
        )

    def _shared_tool_lock(self) -> threading.Lock:
        lock = getattr(self, "_shared_lock", None)
        if lock is None:
            lock = self._shared_lock = threading.Lock()
        return lock

    def _tool_pool(self, workers: int) -> concurrent.futures.ThreadPoolExecutor:
        """One pool for the trainer's lifetime, regrown if a batch needs more threads than it has."""
        pool = getattr(self, "_tool_executor", None)
        if pool is None or getattr(self, "_tool_executor_size", 0) < workers:
            if pool is not None:
                pool.shutdown(wait=False)
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="honest-tool")
            self._tool_executor, self._tool_executor_size = pool, workers
        return pool

    def _call_tools(self, tool_call_list, sync_tool_dict, async_tool_dict, environment, shared_lock):
        """One rollout's tool calls for one turn, in order. Returns (results, calls, failures, seconds).

        TRL's per-sample body unchanged: synchronous tools are called inline, asynchronous ones are gathered on the
        trainer's event loop and their results follow the synchronous ones, and a failure becomes `{"error": ...}`.
        The only addition is the lock around tools that do not belong to this rollout's environment.
        """
        start = time.perf_counter()
        failures = 0
        async_coros = []
        tool_call_results = []
        for tool_call in tool_call_list:
            if tool_call["type"] == "function":
                function = tool_call["function"]
                name = function["name"]
                try:
                    if name in sync_tool_dict:
                        tool = sync_tool_dict[name]
                        if shared_lock is not None and getattr(tool, "__self__", None) is not environment:
                            with shared_lock:
                                result = tool(**function["arguments"])
                        else:
                            result = tool(**function["arguments"])
                        tool_call_results.append((name, result))
                    elif name in async_tool_dict:
                        async_coros.append((name, async_tool_dict[name](**function["arguments"])))
                    else:
                        raise _UnknownTool(f"Tool {name} not found.")
                except Exception as e:  # noqa: BLE001 — as TRL: a failing tool is shown to the model, not raised
                    failures += 1
                    result = {"error": _tool_error_text(e)}
                    tool_call_results.append((name, result))
            else:
                failures += 1
                name = tool_call.get("name", "unknown")
                tool_call_results.append(
                    (
                        name,
                        {"error": f"Unsupported tool call type: {tool_call['type']}"},
                    )
                )

        if async_coros:

            async def _run_async_tools(async_coros):
                coros = [coro for _, coro in async_coros]
                results = await asyncio.gather(*coros, return_exceptions=True)
                return [(name, result) for (name, _), result in zip(async_coros, results, strict=False)]

            async_results = asyncio.run_coroutine_threadsafe(_run_async_tools(async_coros), self.async_loop).result()

            for name, result in async_results:
                if isinstance(result, Exception):
                    failures += 1
                    tool_call_results.append((name, {"error": _tool_error_text(result)}))
                else:
                    tool_call_results.append((name, result))

        return (
            tool_call_results,
            len(tool_call_list),
            failures,
            time.perf_counter() - start,
        )

    def _execute_tool_calls(self, idxs_with_tool: list[int], tool_calls: list) -> list[tuple]:
        """Run one turn's tool calls for every active rollout; results are in the order of `idxs_with_tool`."""
        environments = getattr(self, "environments", None) or []
        concurrent_run = bool(environments) and len(idxs_with_tool) > 1
        workers = min(len(idxs_with_tool), self.max_tool_workers or len(idxs_with_tool))
        concurrent_run = concurrent_run and workers > 1
        jobs = [
            (
                tool_calls[idx],
                self._sync_tool_dicts[i],
                self._async_tool_dicts[i],
                environments[i] if environments else None,
                self._shared_tool_lock() if concurrent_run else None,
            )
            for idx, i in enumerate(idxs_with_tool)
        ]
        if not concurrent_run:
            return [self._call_tools(*job) for job in jobs]
        futures = [self._tool_pool(workers).submit(self._call_tools, *job) for job in jobs]
        concurrent.futures.wait(futures)  # let every rollout finish its calls before any error is raised
        return [future.result() for future in futures]

    # ── the loop ─────────────────────────────────────────────────────────────

    def _tool_call_loop(
        self,
        prompts,
        prompt_ids,
        completion_ids,
        completions,
        logprobs,
        images,
        multimodal_fields,
    ):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt.
        # A port of trl 1.14.1 `GRPOTrainer._tool_call_loop`; comments marked HONEST are the only differences.
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [[1] * len(ids) for ids in completion_ids]  # 0 for tool result tokens, 1 elsewhere
        # Collect images from multimodal tool responses for the forward pass
        tool_images = [[] for _ in completion_ids]
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        # HONEST: bookkeeping only; nothing below reads it to decide what is generated.
        environments = getattr(self, "environments", None) or []
        record = LoopRecord(episode_tool_seconds=[0.0] * len(completion_ids))
        left: dict[int, str] = {}  # rollout index -> why it left the loop after its tools ran

        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [prompts[i] for i in idxs_with_tool]  # select only prompts that need tool calls
            # Snapshot state so we can rollback tool results that would exceed max_completion_length
            completions_len_before = [len(completions[i]) for i in idxs_with_tool]
            tool_images_len_before = [len(tool_images[i]) for i in idxs_with_tool]
            prompts_len_before = [len(prompts[i]) for i in idxs_with_tool]

            # HONEST: all rollouts' tools run at once; TRL calls them one rollout after another.
            started = time.perf_counter()
            executed = self._execute_tool_calls(idxs_with_tool, tool_calls)
            record.tool_seconds += time.perf_counter() - started
            record.tool_turns += 1

            # Build the new prompt for generation, in batch order
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                tool_call_results, calls, failures, seconds = executed[idx]
                tool_call_count += calls
                tool_failure_count += failures
                record.episode_tool_seconds[idx_with_tool] += seconds
                record.slowest_tool_turn = max(record.slowest_tool_turn, seconds)
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])

                for name, result in tool_call_results:
                    # Support multimodal tool responses: if the tool returns a list of content blocks
                    # (e.g., [{"type": "image", "image": ...}, {"type": "text", "text": "..."}]),
                    # pass them through directly so _tokenize_prompts can extract images for VLMs.
                    content = result if isinstance(result, list) else str(result)
                    tool_message = {"role": "tool", "name": name, "content": content}
                    # Collect images from multimodal tool responses
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                tool_images[idx_with_tool].append(part["image"])
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # HONEST: an episode whose environment now holds a report is finished. It leaves the loop the way TRL's
            # over-budget samples do below: this turn's tool messages and images are rolled back, so `completions`,
            # `tool_images`, `completion_ids`, `tool_mask` and `logprobs` all end at the report turn as sampled.
            if self.stop_at_report and environments:
                reported = [self._episode_reported(environments[i]) for i in idxs_with_tool]
                if any(reported):
                    for idx in range(len(idxs_with_tool)):
                        if reported[idx]:
                            idx_with_tool = idxs_with_tool[idx]
                            del completions[idx_with_tool][completions_len_before[idx] :]
                            del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                            del prompts[idx_with_tool][prompts_len_before[idx] :]
                            left[idx_with_tool] = STOP_REPORT

                    idxs_with_tool = _unreported(idxs_with_tool, reported)
                    completions_len_before = _unreported(completions_len_before, reported)
                    tool_images_len_before = _unreported(tool_images_len_before, reported)
                    prompts_len_before = _unreported(prompts_len_before, reported)
                    if not idxs_with_tool:
                        break  # every active episode has reported

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool] + completion_ids[idx_with_tool] + suffix_ids
                )

            # Drop tool results whose addition would push the sequence past max_completion_length (the completion
            # budget) or past the backend context ceiling (vLLM and transformers will error out on inputs longer than
            # the model's max length). The sample exits the loop with its completion as-is, and the tool
            # messages/images appended this iteration are rolled back so completions and tool_images stay consistent
            # with completion_ids.
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = self.vllm_generation.llm.llm_engine.model_config.max_model_len
            else:
                config = self.model.config.text_config if self._is_vlm else self.model.config
                max_model_len = config.max_position_embeddings
            overlong = [
                len(pct) - len(prompt_ids[i]) > self.max_completion_length or len(pct) >= max_model_len
                for i, pct in zip(idxs_with_tool, prompt_completion_tool_ids, strict=True)
            ]
            for idx in range(len(idxs_with_tool)):
                if overlong[idx]:
                    idx_with_tool = idxs_with_tool[idx]
                    del completions[idx_with_tool][completions_len_before[idx] :]
                    del tool_images[idx_with_tool][tool_images_len_before[idx] :]
                    del prompts[idx_with_tool][prompts_len_before[idx] :]
                    left[idx_with_tool] = STOP_CHAR_BUDGET  # HONEST: bookkeeping
            # Keep only non-overlong items for further processing
            idxs_with_tool = [idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o]
            prompt_completion_tool_ids = [
                pct for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True) if not o
            ]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Filter images and multimodal fields to match the current subset (index into full batch).
            # Merge tool response images so the model can see visual feedback during generation.
            merged_images = images
            if any(imgs for imgs in tool_images):
                if merged_images is None:
                    merged_images = [imgs if imgs else None for imgs in tool_images]
                else:
                    merged_images = [
                        (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                    ]
            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
            if self._is_vlm and self.tools and any(imgs for imgs in tool_images):
                flat_loop_images = [img for img_list in loop_images if img_list for img in img_list]
                if flat_loop_images:
                    image_inputs = self.processing_class.image_processor(images=flat_loop_images, return_tensors="pt")
                    # TRL writes `super()._prepare_inputs` inside GRPOTrainer: the transformers `Trainer` method.
                    image_inputs = super(GRPOTrainer, self)._prepare_inputs(image_inputs)
                    loop_multimodal_fields = dict(image_inputs)
                else:
                    loop_multimodal_fields = {}
            elif multimodal_fields:
                if "num_images" not in multimodal_fields and images is not None:
                    multimodal_fields = {
                        **multimodal_fields,
                        "num_images": [len(img_list) if img_list else 0 for img_list in images],
                    }
                split_fields = split_pixel_values_by_grid(multimodal_fields)
                loop_multimodal_fields = {}
                for k, v in split_fields.items():
                    selected = [v[i] for i in idxs_with_tool]
                    # Per-token type-id fields may arrive as tensors (e.g. via the padding workaround);
                    # convert them to lists so they follow the same left-padding path as list-valued
                    # per-token fields below, keeping them aligned with the left-padded input_ids.
                    if k in ("token_type_ids", "mm_token_type_ids") and isinstance(selected[0], torch.Tensor):
                        selected = [s.tolist() for s in selected]
                    # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                    if isinstance(selected[0], list):
                        selected = [
                            s + [0] * (len(pct) - len(s))
                            for s, pct in zip(selected, prompt_completion_tool_ids, strict=True)
                        ]
                    elif isinstance(v, torch.Tensor):
                        selected = torch.stack(selected)
                    loop_multimodal_fields[k] = selected
                loop_multimodal_fields = unsplit_pixel_values_by_grid(loop_multimodal_fields)
                loop_multimodal_fields.pop("num_images", None)
                if "pixel_values" in loop_multimodal_fields and loop_multimodal_fields["pixel_values"].numel() == 0:
                    for key in [
                        "pixel_values",
                        "image_grid_thw",
                        "pixel_values_videos",
                        "video_grid_thw",
                    ]:
                        loop_multimodal_fields.pop(key, None)
            else:
                loop_multimodal_fields = {}

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            post_tool_ids, post_tool_logprobs = self._generate_single_turn(
                prompt_completion_tool_ids,
                loop_images,
                loop_multimodal_fields,
                num_generations=1,  # each sample has its own history, so generate one completion per sample
                has_tool_images=any(imgs for imgs in tool_images),
            )
            record.regenerated_turns += len(idxs_with_tool)  # HONEST: bookkeeping

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length.
            # The pre-regen check guarantees len(completion_tool_ids) <= max_completion_length, so any
            # excess can only come from post_tool_ids. post_tool_ids is model-generated text and never
            # contains image tokens, so a plain slice is safe.
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                completion_tool_length = len(prompt_completion_tool_ids[idx]) - len(prompt_ids[idx_with_tool])
                excess_length = completion_tool_length + len(post_tool_ids[idx]) - self.max_completion_length
                if excess_length > 0:
                    new_len = len(post_tool_ids[idx]) - excess_length
                    post_tool_ids[idx] = post_tool_ids[idx][:new_len]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][:new_len]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = prompt_completion_tool_length - prompt_length - completion_length
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[idx]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions
            post_tool_completions = [
                parse_response(self._tokenizer, ids, prefix=prompt_completion_tool_ids[idx]) if ids else {}
                for idx, ids in enumerate(post_tool_ids)
            ]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[idx]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [completion.get("tool_calls") for completion in post_tool_completions]
            idxs_with_tool = [idx for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True) if tool_call]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1

        # HONEST: record why each rollout ended. `idxs_with_tool` now holds the rollouts cut off by the iteration cap.
        record.exits = self._loop_exits(completion_ids, environments, left, set(idxs_with_tool))
        self._loop_record = record

        return (
            tool_mask,
            completions,
            completion_ids,
            logprobs,
            tool_call_count,
            tool_failure_count,
            tool_images,
        )

    def _loop_exits(
        self,
        completion_ids: list[list[int]],
        environments: list,
        left: dict[int, str],
        capped: set[int],
    ) -> list[str]:
        """One of `LOOP_EXITS` per rollout. A report wins; then a cut last turn, because TRL masks it from the loss."""
        eos_and_pad = (self._tokenizer.eos_token_id, self._tokenizer.pad_token_id)
        exits = []
        for i, ids in enumerate(completion_ids):
            if left.get(i) == STOP_REPORT or (i < len(environments) and self._episode_reported(environments[i])):
                exits.append(STOP_REPORT)
            elif not ids or ids[-1] not in eos_and_pad:
                exits.append(STOP_TRUNCATED)
            elif i in left:
                exits.append(left[i])
            elif i in capped:
                exits.append(STOP_CALL_BUDGET)
            else:
                exits.append(STOP_SILENT)
        return exits
