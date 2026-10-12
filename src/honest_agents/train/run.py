"""Trainer construction and the training loop entry points."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

from honest_agents.models import (
    check_end_of_turn_token,
    check_tool_flow_template,
    check_training_template,
    family_spec,
    flash_linear_attention_available,
)
from honest_agents.train.dashboard import WandbMirror, init_run, wandb_enabled
from honest_agents.train.plan import Plan, bank_dir, quarantine_file
from honest_agents.train.rows import check_resume_order
from honest_agents.train.startup import check_lm_head_cast, check_prompt_budget, startup_probe

logger = logging.getLogger("honest_agents.train")


# ── run ─────────────────────────────────────────────────────────────────────


def _preflight_gpu(plan: Plan) -> None:
    spec = family_spec(plan.family)
    if spec.needs_flash_linear_attention and not flash_linear_attention_available():
        msg = "flash-linear-attention (`fla`) is not installed; Qwen3.5 Gated DeltaNet falls back to slow torch code."
        if plan.cfg.model.require_fla:
            raise RuntimeError(msg + " Install it or set model.require_fla=false.")
        logger.warning(msg)


def _check_world_size(plan: Plan) -> None:
    """Batch geometry and pair packing were computed for `launch.num_processes`; a different real world size would
    silently change the prompts per round (and split pairs across rounds)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != plan.cfg.launch.num_processes:
        raise RuntimeError(
            f"WORLD_SIZE={world} but launch.num_processes={plan.cfg.launch.num_processes}; "
            "pass launch.num_processes=<n> (scripts/train.sh does this)"
        )


def _last_checkpoint(out: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(out)) if out.is_dir() else None


def add_callbacks(trainer: Any, plan: Plan) -> None:
    """Stop rules, the run log, the validation probe's checkpoint selection and, with W&B on, what the dashboard
    needs beyond TRL's own logs (`train.dashboard`; it reads the run log and the probe, so it comes after both)."""
    from honest_agents.callbacks import JsonlLogger, StopRules, ValidationProbe

    out = plan.grpo_config.output_dir
    trainer.add_callback(StopRules(plan.stop_config, reward_mode=plan.cfg.reward.mode, output_dir=out))
    run_log = JsonlLogger(out)
    trainer.add_callback(run_log)
    probe = None
    if plan.eval_dataset is not None:
        probe = ValidationProbe(out, plan.cfg.probe.selection_metric)
        trainer.add_callback(probe)
    if wandb_enabled(plan.grpo_config.report_to):
        trainer.add_callback(WandbMirror(run_log, probe, health_checks=plan.cfg.wandb.health))
    from honest_agents.memory import MemoryDebug, ReleaseFreedMemory, mem_debug_enabled

    trainer.add_callback(ReleaseFreedMemory())
    if mem_debug_enabled():
        trainer.add_callback(MemoryDebug(out))


# Per-step wall-clock columns the run log carries for the Phase-0 trainer decision. The trainer logs
# `rollout/*` (generation, tools); these three are the remaining components. TRL's `step_time` is their sum.
METRIC_GRADING_SECONDS = "timing/grading_seconds"  # reward functions: boot-wait, tests, evidence, blocker checks
METRIC_SYNC_SECONDS = "timing/sync_seconds"  # pushing the merged weights to vLLM
METRIC_TRAIN_SECONDS = "timing/train_seconds"  # forward + backward + optimizer: step time minus rollout and grading


def probe_safe_trainer_class() -> type:
    """`HonestGRPOTrainer` plus two things TRL's loop does not give us.

    **An evaluation pass that fits in memory.** TRL's evaluation loop reuses `per_device_eval_batch_size` for two
    different things: how many episodes are in flight (the probe wants many, so 100 variants take minutes) and how many
    sequences go through the model at once when the rollouts are scored. Training scores 2 sequences per forward; with
    `probe.batch_size` 25 the same code would put 25 sequences of up to ~20k tokens through the full-logits path
    (hundreds of GB), and a second time inside `compute_loss`. Here the scoring chunk stays at the training
    micro-batch, and `prediction_step` skips the loss, which the probe never reads (it reads the metrics the reward
    function logged).

    **Step timing by component** (`timing/*`): TRL only reports its own profiling to wandb or trackio, and the Phase-0
    decision between TRL and verl needs grading, sync and training time per step in the run log.

    **Every probe episode saved** (`train.probe.ProbeTrajectoryMixin`): TRL's completions buffers hold one training
    round (64 rows), so a 100-episode probe kept 64, and its table overwrote the training round of the same step.
    """
    import time

    import torch

    from honest_agents.train.probe import ProbeTrajectoryMixin
    from honest_agents.trainer import HonestGRPOTrainer

    class ProbeSafeGRPOTrainer(ProbeTrajectoryMixin, HonestGRPOTrainer):
        def __init__(self, *args: Any, **kwargs: Any):
            self._prepare_seconds = 0.0  # time inside `_prepare_inputs` (generation, grading, sync), cumulative
            self._train_seconds = 0.0  # compute of the micro-steps of the current optimizer step
            super().__init__(*args, **kwargs)
            vllm_generation = getattr(self, "vllm_generation", None)
            if vllm_generation is not None:
                sync = vllm_generation.sync_weights

                def timed_sync(*a: Any, **k: Any):
                    start = time.perf_counter()
                    try:
                        return sync(*a, **k)
                    finally:
                        self._log_seconds(METRIC_SYNC_SECONDS, time.perf_counter() - start)

                vllm_generation.sync_weights = timed_sync

        def _log_seconds(self, name: str, seconds: float) -> None:
            self._metrics["train" if self.model.training else "eval"][name].append(seconds)

        def _calculate_rewards(self, *args: Any, **kwargs: Any):
            start = time.perf_counter()
            try:
                return super()._calculate_rewards(*args, **kwargs)
            finally:
                self._log_seconds(METRIC_GRADING_SECONDS, time.perf_counter() - start)

        def _prepare_inputs(self, generation_batch):
            start = time.perf_counter()
            try:
                return super()._prepare_inputs(generation_batch)
            finally:
                self._prepare_seconds += time.perf_counter() - start

        def training_step(self, model, inputs, num_items_in_batch=None):
            prepared, start = self._prepare_seconds, time.perf_counter()
            output = super().training_step(model, inputs, num_items_in_batch)
            self._train_seconds += time.perf_counter() - start - (self._prepare_seconds - prepared)
            if self._step % self.current_gradient_accumulation_steps == 0:  # TRL's own optimizer-step boundary
                self._metrics["train"][METRIC_TRAIN_SECONDS].append(self._train_seconds)
                self._train_seconds = 0.0
            return output

        def _get_per_token_logps_and_entropies(self, model, *args, batch_size=None, **kwargs):
            if not self.model.training:  # evaluation: bound the scoring chunk by the training micro-batch
                cap = int(self.args.per_device_train_batch_size)
                batch_size = cap if batch_size is None else min(int(batch_size), cap)
            return super()._get_per_token_logps_and_entropies(model, *args, batch_size=batch_size, **kwargs)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            self._prepare_inputs(inputs)  # roll out, grade and log `eval_*` metrics; nothing is learned from them
            return torch.zeros((), device=self.accelerator.device), None, None

    return ProbeSafeGRPOTrainer


def make_trainer(plan: Plan, *, check_server: bool = True) -> Any:
    """Everything `train()` does before the first step: guards, bank and quarantine paths, the trainer, the template and
    prompt-budget checks, the callbacks. `sync_check` builds the same trainer so it tests the code path that trains."""
    from honest_agents import bank
    from honest_agents.harness import set_max_concurrent_boots

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    _check_world_size(plan)
    _preflight_gpu(plan)
    check_lm_head_cast(plan.cfg)
    set_max_concurrent_boots(plan.cfg.env.max_concurrent_boots)
    if check_server:
        startup_probe(plan)
    bank.set_bank_dir(bank_dir(plan.cfg) if plan.cfg.bank.enabled else None)
    bank.set_quarantine_file(quarantine_file(plan.cfg))

    trainer_class = probe_safe_trainer_class()  # lazy: --dry-run works without the trainer
    trainer = trainer_class(
        model=plan.cfg.model.name,
        reward_funcs=plan.reward_funcs,
        args=plan.grpo_config,
        train_dataset=plan.train_dataset,
        eval_dataset=plan.eval_dataset,
        peft_config=plan.peft_config,
        environment_factory=plan.environment_factory,
    )
    trainer.max_tool_workers = plan.cfg.trainer.max_tool_workers
    trainer.trim_padding = plan.cfg.trainer.trim_padding
    trainer.sort_micro_batches = plan.cfg.trainer.sort_micro_batches
    check_training_template(trainer, plan.family, strict=plan.cfg.model.strict_template)
    check_tool_flow_template(trainer, strict=plan.cfg.model.strict_template)
    check_end_of_turn_token(trainer, strict=plan.cfg.model.strict_template)
    check_prompt_budget(trainer, plan)
    if list(trainer.reward_weights.tolist()) != list(plan.grpo_config.reward_weights):
        raise RuntimeError(
            f"trainer reward weights {trainer.reward_weights.tolist()} != configured "
            f"{plan.grpo_config.reward_weights} (did the environment add a reward source?)"
        )
    add_callbacks(trainer, plan)
    return trainer


def train(plan: Plan, resume: bool = False) -> bool:
    """Train; returns True if a stop rule ended the run early (`main` turns that into exit status 3)."""
    from transformers import set_seed

    set_seed(plan.cfg.seed)
    if not plan.cfg.data.allow_legacy_verifier:  # fail before model loading, with the command that builds the bundle
        from honest_agents.verifier import ensure_bundle

        bundle = ensure_bundle(Path(plan.cfg.env.verifier_bundle_dir) if plan.cfg.env.verifier_bundle_dir else None)
        logger.info("verifier bundle: %s", bundle)
    out = Path(plan.grpo_config.output_dir)
    ckpt = _last_checkpoint(out) if resume else None
    if resume and ckpt is None:
        logger.warning("--resume: no checkpoint under %s; starting fresh", out)
    check_resume_order(out, plan.order, resuming=ckpt is not None)
    if ckpt is not None:
        plan.grpo_config.eval_on_start = False  # the step-0 probe already ran in the first life of the run
    if int(os.environ.get("RANK", "0")) == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps(dataclasses.asdict(plan.cfg), indent=2, default=str))
    # The dashboard run opens before the model loads: a missing key or an unreachable W&B fails in seconds, not after
    # minutes of loading, and transformers' WandbCallback later finds this run instead of starting its own.
    dashboard = None
    if wandb_enabled(plan.grpo_config.report_to) and int(os.environ.get("RANK", "0")) == 0:
        dashboard = init_run(plan.cfg, out, resuming=ckpt is not None)
    try:
        trainer = make_trainer(plan)
        report = getattr(trainer.model, "print_trainable_parameters", None)
        if trainer.accelerator.is_main_process and report is not None:
            report()
        trainer.train(resume_from_checkpoint=ckpt)
    except BaseException:
        if dashboard is not None:
            dashboard.finish(exit_code=1)
        raise
    if dashboard is not None:
        dashboard.summary["stopped_by_rule"] = stopped_by_rule(trainer)
        dashboard.finish()
    trainer.save_model(str(out / "final"))
    best = out / "best_checkpoint.json"
    if best.is_file():
        logger.info("checkpoint selected on validation: %s", best.read_text().replace("\n", " "))
    return stopped_by_rule(trainer)


def stopped_by_rule(trainer: Any) -> bool:
    """True if a stop rule (not `max_steps`) ended the run with `stop.action: stop`."""
    from honest_agents.callbacks import StopRules

    handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(handler, "callbacks", None) or getattr(trainer, "callbacks", None) or []
    return any(isinstance(c, StopRules) and c.stopped for c in callbacks)
