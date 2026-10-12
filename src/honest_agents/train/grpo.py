"""Guards on the `grpo:` section and construction of the `GRPOConfig`."""

from __future__ import annotations

import dataclasses
from typing import Any

from honest_agents.models import (
    ModelFamily,
    check_full_distribution_sampling,
    family_spec,
)
from honest_agents.train.config import _DERIVED_GRPO_FIELDS, REWARD_MODES, Limits, TrainRunConfig
from honest_agents.train.dashboard import resolve_report_to
from honest_agents.train.derive import build_limits, output_dir, reward_weights, run_name, served_max_model_len

# ── guards on the GRPO section ──────────────────────────────────────────────


def check_scale_rewards(grpo: dict[str, Any]) -> None:
    """`scale_rewards` must be `none`: group scaling divides alpha out of two-valued groups (realised claim thresholds
    0.40/0.48/0.54/0.57 for nominal 0.50/0.625/0.75/0.875) and `batch` only approximates the threshold."""
    value = grpo.get("scale_rewards", "group")  # TRL's default is "group"
    if value is False or value == "none":
        return
    raise ValueError(
        f"grpo.scale_rewards={value!r} is refused: reward scaling by the group's standard deviation erases alpha, so the "
        "claim threshold (1 + alpha) / 2 is no longer what the run optimises. Use scale_rewards: none."
    )


def check_launch_layout(grpo: dict[str, Any], num_processes: int) -> None:
    """Refuse the layouts that mis-route completions inside TRL's tool loop."""
    if num_processes < 1:
        raise ValueError(f"launch.num_processes must be >= 1, got {num_processes}")
    if num_processes == 1 or not grpo.get("use_vllm", False):
        return
    mode = grpo.get("vllm_mode", "colocate")
    if mode == "server" or int(grpo.get("vllm_tensor_parallel_size", 1)) > 1:
        raise ValueError(
            f"launch.num_processes={num_processes} with vLLM {mode} mode: TRL slices the gathered completions by each "
            "rank's own count of still-active episodes, which differs between ranks from the second tool round on, so "
            "ranks train on completions sampled for other episodes. Use one training process per run; the "
            "other GPU of the pair serves vLLM, and the second pair of GPUs runs a second run."
        )


def check_importance_sampling(grpo: dict[str, Any]) -> None:
    """Token-level truncated IS: the sequence-level default drops long trajectories systematically."""
    if not grpo.get("vllm_importance_sampling_correction", True):
        return
    mode = grpo.get("vllm_importance_sampling_mode", "sequence_mask")  # TRL 1.14.1 default
    if mode != "token_truncate":
        raise ValueError(
            f"grpo.vllm_importance_sampling_mode={mode!r}: use 'token_truncate'. Sequence-level ratios "
            "shrink with length and zero out long episodes; or turn the correction off."
        )


def check_checkpointing(grpo: dict[str, Any]) -> None:
    if grpo.get("save_total_limit") is not None:
        raise ValueError(
            "grpo.save_total_limit must be null: every checkpoint is kept, because the validation probe picks one "
            "afterwards and the best step is often not the last"
        )
    if grpo.get("ignore_data_skip"):
        raise ValueError("grpo.ignore_data_skip=true would replay the first tasks on resume")


def check_batch_geometry(grpo_cfg, num_processes: int, raw: dict[str, Any] | None = None) -> int:
    """Re-run GRPOConfig's divisibility checks for the *launch* world size (the dry-run machine has world_size 1).

    `raw` is the user's `grpo:` section: GRPOConfig has already filled `steps_per_generation` / `generation_batch_size`
    for world size 1, so which of the two the user actually set must come from there (trl GRPOConfig.__post_init__).
    One generation round must be exactly one optimizer step (`steps_per_generation == gradient_accumulation_steps`,
    `num_iterations == 1`): a checkpoint is written at an optimizer step, so with longer rounds a resume would fall
    inside a generation batch and replay or skip prompts.
    Returns the generation batch size (completions per generation round) at the launch world size.
    """
    raw = raw or {}
    per_dev = grpo_cfg.per_device_train_batch_size
    global_bs = per_dev * num_processes
    if raw.get("generation_batch_size") is not None:
        gen_bs = int(raw["generation_batch_size"])
        if gen_bs % global_bs:
            raise ValueError(
                f"generation_batch_size {gen_bs} not divisible by the global batch size {per_dev}*{num_processes}"
            )
    else:
        spg = raw.get("steps_per_generation") or grpo_cfg.gradient_accumulation_steps
        gen_bs = global_bs * spg
    if gen_bs % grpo_cfg.num_generations:
        raise ValueError(
            f"generation batch {gen_bs} (at {num_processes} processes) not divisible by "
            f"num_generations={grpo_cfg.num_generations}"
        )
    if gen_bs != global_bs * grpo_cfg.gradient_accumulation_steps:
        raise ValueError(
            f"one generation round ({gen_bs} completions) must be one optimizer step "
            f"({global_bs} x gradient_accumulation_steps={grpo_cfg.gradient_accumulation_steps}); otherwise a "
            "checkpoint falls inside a generation batch and a resumed run replays or skips tasks"
        )
    if getattr(grpo_cfg, "num_iterations", 1) != 1:
        raise ValueError("grpo.num_iterations must be 1: reusing a generation batch breaks resume order")
    if grpo_cfg.do_eval and grpo_cfg.eval_strategy != "no":
        n_eval = grpo_cfg.num_generations_eval or grpo_cfg.num_generations
        if (grpo_cfg.per_device_eval_batch_size * num_processes) % n_eval:
            raise ValueError(
                f"global eval batch {grpo_cfg.per_device_eval_batch_size}*{num_processes} not divisible by {n_eval}"
            )
    return gen_bs


def probe_enabled(cfg: TrainRunConfig) -> bool:
    return cfg.data.eval_file is not None or cfg.data.probe_file is not None


def build_grpo_config(cfg: TrainRunConfig, family: ModelFamily, limits: Limits | None = None):
    from trl import GRPOConfig

    grpo = dict(cfg.grpo)
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = set(grpo) - known
    if unknown:
        raise ValueError(f"Unknown GRPOConfig field(s) under `grpo:`: {sorted(unknown)}")
    clash = set(grpo) & _DERIVED_GRPO_FIELDS
    if clash:
        raise ValueError(f"`grpo:` must not set derived field(s) {sorted(clash)}; use the top-level/section keys")
    if cfg.reward.mode not in REWARD_MODES:
        raise ValueError(f"reward.mode must be one of {REWARD_MODES}, got {cfg.reward.mode!r}")

    agg = grpo.get("multi_objective_aggregation", "sum_then_normalize")
    if agg != "sum_then_normalize":
        raise ValueError("reward modes assume multi_objective_aggregation='sum_then_normalize'")
    check_scale_rewards(grpo)
    grpo["scale_rewards"] = "none"
    check_full_distribution_sampling(grpo)
    check_importance_sampling(grpo)
    check_checkpointing(grpo)
    check_launch_layout(grpo, cfg.launch.num_processes)
    limits = limits or build_limits(cfg)
    if "report_to" in grpo:  # WANDB_MODE=disabled drops wandb here, before transformers would import it
        grpo["report_to"] = resolve_report_to(grpo["report_to"])

    spec = family_spec(family)
    grpo.setdefault("chat_template_kwargs", dict(spec.chat_template_kwargs))
    if cfg.data.pair_batches:
        grpo["shuffle_dataset"] = False  # order is set by `pair_ordered` below
    if grpo.get("use_vllm") and grpo.get("vllm_mode", "colocate") == "colocate":
        grpo["vllm_max_model_length"] = served_max_model_len(cfg, limits)  # the engine's window, same formula

    model_init_kwargs: dict[str, Any] = {"dtype": cfg.model.dtype}
    if cfg.model.revision:
        model_init_kwargs["revision"] = cfg.model.revision
    if cfg.model.attn_implementation:
        model_init_kwargs["attn_implementation"] = cfg.model.attn_implementation

    derived: dict[str, Any] = {"cast_lm_head_to_fp32": cfg.model.cast_lm_head_to_fp32}
    if probe_enabled(cfg):
        p = cfg.probe
        if p.every_steps < 1 or p.batch_size < 1 or p.max_variants < 1 or p.samples_per_variant < 1:
            raise ValueError(
                "probe.every_steps, probe.batch_size, probe.max_variants and probe.samples_per_variant must be >= 1"
            )
        if p.batch_size % p.samples_per_variant:
            # TRL repeats each probe row samples_per_variant times inside one evaluation batch and refuses a batch that
            # would cut a row's samples apart; say so here rather than in its divisibility error.
            raise ValueError(
                f"probe.batch_size={p.batch_size} must be a multiple of probe.samples_per_variant="
                f"{p.samples_per_variant}"
            )
        if p.min_infeasible < 0 or p.min_feasible < 0:
            raise ValueError("probe.min_infeasible and probe.min_feasible must be >= 0")
        save_steps = grpo.get("save_steps", 500)
        if grpo.get("save_strategy", "steps") != "steps" or p.every_steps % int(save_steps):
            raise ValueError(
                f"probe.every_steps={p.every_steps} must be a multiple of grpo.save_steps={save_steps} "
                "(save_strategy: steps), so every probed step has a checkpoint to select"
            )
        derived.update(
            do_eval=True,
            eval_strategy="steps",
            eval_steps=p.every_steps,
            eval_on_start=p.on_start,
            per_device_eval_batch_size=p.batch_size,
            num_generations_eval=p.samples_per_variant,  # k rollouts per variant, not a training group of 8
        )

    return GRPOConfig(
        output_dir=str(output_dir(cfg)),
        run_name=run_name(cfg),
        seed=cfg.seed,
        reward_weights=reward_weights(cfg.reward.mode),
        model_init_kwargs=model_init_kwargs,
        **derived,
        **grpo,
    )
