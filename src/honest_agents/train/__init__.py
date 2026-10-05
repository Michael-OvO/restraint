"""GRPO training entrypoint: `python -m honest_agents.train --config configs/x.yaml [key.path=value ...] [--dry-run]`.

A run is fully described by one YAML file (optionally inheriting from another via `base:`) plus dotted CLI overrides,
so sweeps are just override lists (see `scripts/sweep_alpha.sh`). `--dry-run` builds and validates everything except
the model and the trainer (dataset, `GRPOConfig`, `LoraConfig`, reward wiring, environment factory, limits, guards), so
config validity is unit-testable on a laptop without GPUs, Docker or network.

What this module guarantees (docs/design.md section 8):

* **One reward function per run** (`reward.mode`: honest, outcome, outcome_claimed, outcome_reflect), registered with
  `reward_weights = [1.0]`, so a `None` is a real mask.
* **`scale_rewards: none` everywhere.** `group` is refused: it divides alpha out of two-valued groups.
* **One training process per run** when vLLM is a separate server (or sharded): TRL slices the gathered completions by
  each rank's own count of live episodes, which mis-routes them inside the tool loop.
* **One source of truth for limits.** `grpo.generation_kwargs.max_tokens` (per turn), `grpo.max_completion_length`
  (total) and `grpo.max_tool_calling_iterations` give `harness.EpisodeBudget.from_limits(...)` for the environment, the
  efficiency reference of the reward and the vLLM `--max-model-len`, which is asserted against the running server at
  startup. Evaluation builds the same budget from the same numbers.
* **Sampling from the full distribution**: `top_p 1`, `top_k 0`, `min_p 0`, no repetition penalty,
  checked in the config and probed on the server.
* **Resume continues the task order**: the dataset order is a pure function of the config and the seed,
  its fingerprint is stored and re-checked on resume, and one generation round is exactly one optimizer step so a
  checkpoint never falls inside a generation batch.
* **Unsafe alphas are refused** when a Phase-0 summary is supplied (`--phase0-summary`): (1 + alpha) / 2 above the pool
  pass rate minus 0.1 means declining everything beats working (`--allow-unsafe-alpha` overrides; exit status 4).
* **Trainer safety checks at start**: the chat template ends every assistant turn with the tokenizer's EOS (else every
  turn counts as truncated), and the vLLM server binds to 127.0.0.1 unless `VLLM_EXPOSE=1` (`scripts/serve_vllm.sh`).
* **Data guards**: only `verifier = "pytest"` tasks (override: `data.allow_legacy_verifier`), no test-split rows in
  training, a validation probe split that shares no variant with training, and labels in the rows that agree with the
  task's own `task.toml`.
"""

from honest_agents.train.cli import main
from honest_agents.train.config import (
    DATA_ORDER_FILE,
    EXIT_STOPPED_BY_RULE,
    EXIT_UNSAFE_ALPHA,
    REWARD_MODES,
    Limits,
    TrainRunConfig,
    apply_overrides,
    load_config_dict,
    load_run_config,
)
from honest_agents.train.derive import (
    UnsafeAlphaError,
    alpha_of,
    build_limits,
    build_lora_config,
    build_reward_config,
    check_safe_alpha,
    default_run_name,
    output_dir,
    resolve_family,
    reward_weights,
    run_name,
    served_max_model_len,
)
from honest_agents.train.grpo import (
    build_grpo_config,
    check_batch_geometry,
    check_checkpointing,
    check_importance_sampling,
    check_launch_layout,
    check_scale_rewards,
)
from honest_agents.train.plan import (
    Plan,
    build_budget,
    build_environment_factory,
    build_grading_config,
    build_plan,
    build_reward_funcs,
)
from honest_agents.train.rows import (
    build_dataset,
    check_resume_order,
    check_rows,
    check_split_isolation,
    holdout_ids,
    load_rows,
    order_fingerprint,
    pair_ordered,
    prepare_train_rows,
    select_probe_rows,
)
from honest_agents.train.run import (
    METRIC_GRADING_SECONDS,
    METRIC_TRAIN_SECONDS,
    make_trainer,
    probe_safe_trainer_class,
    train,
)
from honest_agents.train.startup import check_lm_head_cast, check_prompt_budget, check_template_offline, startup_probe

__all__ = [
    "DATA_ORDER_FILE",
    "EXIT_STOPPED_BY_RULE",
    "EXIT_UNSAFE_ALPHA",
    "METRIC_GRADING_SECONDS",
    "METRIC_TRAIN_SECONDS",
    "REWARD_MODES",
    "Limits",
    "Plan",
    "TrainRunConfig",
    "UnsafeAlphaError",
    "alpha_of",
    "apply_overrides",
    "build_budget",
    "build_dataset",
    "build_environment_factory",
    "build_grading_config",
    "build_grpo_config",
    "build_limits",
    "build_lora_config",
    "build_plan",
    "build_reward_config",
    "build_reward_funcs",
    "check_batch_geometry",
    "check_checkpointing",
    "check_importance_sampling",
    "check_launch_layout",
    "check_lm_head_cast",
    "check_prompt_budget",
    "check_resume_order",
    "check_rows",
    "check_safe_alpha",
    "check_scale_rewards",
    "check_split_isolation",
    "check_template_offline",
    "default_run_name",
    "holdout_ids",
    "load_config_dict",
    "load_rows",
    "load_run_config",
    "main",
    "make_trainer",
    "order_fingerprint",
    "output_dir",
    "pair_ordered",
    "prepare_train_rows",
    "probe_safe_trainer_class",
    "resolve_family",
    "reward_weights",
    "run_name",
    "select_probe_rows",
    "served_max_model_len",
    "startup_probe",
    "train",
]
