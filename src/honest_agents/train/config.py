"""The run configuration: dataclass sections, YAML loading with `extends` and dotted-path overrides."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from honest_agents import trl_rewards
from honest_agents.callbacks import SAFE_ALPHA_MARGIN
from honest_agents.models import (
    pinned_revision,
    vllm_max_model_len,
)

REWARD_MODES = trl_rewards.MODES
SPLIT_TRAIN = "train"
VALIDATION_SPLITS = ("validation", "val")
DATA_ORDER_FILE = "data_order.json"
EXIT_STOPPED_BY_RULE = 3  # distinct from a crash, so a sweep can tell a collapse from a failure
EXIT_UNSAFE_ALPHA = 4  # the alpha was refused against the Phase-0 pool pass rate: a sweep skips it instead of failing
ALPHA_PASS_MARGIN = SAFE_ALPHA_MARGIN  # (1 + alpha) / 2 must stay this far below the pool pass rate

# GRPOConfig fields derived from other sections; setting them under `grpo:` would silently disagree with the run.
_DERIVED_GRPO_FIELDS = {
    "output_dir",
    "run_name",
    "seed",
    "reward_weights",
    "model_init_kwargs",
    # from `model.cast_lm_head_to_fp32`
    "cast_lm_head_to_fp32",
    # from the `probe:` section
    "do_eval",
    "eval_strategy",
    "eval_steps",
    "eval_on_start",
    "per_device_eval_batch_size",
    "num_generations_eval",
}


# ── config schema ─────────────────────────────────────────────────────────────


@dataclass
class ModelSection:
    name: str = "Qwen/Qwen3-4B"
    family: str = "auto"  # auto | qwen3 | qwen3_5
    revision: str | None = None
    dtype: str = "bfloat16"  # TRL loads in float32 unless told otherwise
    attn_implementation: str | None = None
    strict_template: bool = True  # fail unless the rollout template is one TRL knows for the family (models.py)
    require_fla: bool = True  # Qwen3.5: fail on GPU if flash-linear-attention is missing (~20x slower otherwise)
    # Compute the LM head in float32 (ScaleRL recipe; Tmax found it "especially crucial" on Qwen3.5). TRL only
    # supports it for untied embeddings; `train()` checks the model's config before loading weights.
    cast_lm_head_to_fp32: bool = False

    def __post_init__(self) -> None:
        self.revision = pinned_revision(self.name, self.revision)  # DEF-51: a pinned Hub id loads at its pin


@dataclass
class LoraSection:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str | list[str] = "auto"  # auto -> honest_agents.models.lora_target_modules(family)


@dataclass
class DataSection:
    train_file: str = "data/train.jsonl"
    eval_file: str | None = None  # the validation split the periodic probe runs on; never the test split
    # A fixed probe variant list (`python -m honest_agents.tasks.probe_set`), used whole and in preference to the hash
    # subset of `eval_file`. Validation rows only, like `eval_file`; it must hold `probe.min_infeasible` impossible and
    # `probe.min_feasible` feasible variants.
    probe_file: str | None = None
    max_rows: int | None = None
    # Keep feasible/infeasible variants of a pair_id in the same generation round (TRL's RepeatSampler cuts the dataset
    # into consecutive chunks of generation_batch_size / num_generations prompts) so each round contains both sides of
    # its pairs (abstention collapse is the main GRPO risk). Pairs are shuffled as units with the run seed, packed so
    # none straddles a round boundary, and TRL's own sampler shuffle is disabled.
    pair_batches: bool = True
    # Guards. Each has an explicit override so a deliberate exception is visible in the config.
    allow_legacy_verifier: bool = False  # tasks must say `[metadata.honest] verifier = "pytest"`
    check_tasks: bool = True  # read every row's task.toml (verifier, labels); off only for synthetic unit-test rows
    require_expected_targets: bool = True  # an infeasible row without expected_targets lets a blind guess earn m = 1
    allow_custom_system_prompt: bool = False  # rows must use harness.SYSTEM_PROMPT (one prompt in train and eval)
    # training rows from a source reviewed as a spec defect (DEF-66, `tasks.spec_audit`) are refused; true only to
    # reproduce a run that predates the audit
    allow_spec_defects: bool = False
    # test_ids.txt of the dataset build: a train or validation row whose pair_id is in it is refused. None = the file
    # next to `train_file` if there is one (tasks.build writes it there), else no check.
    holdout_ids: str | None = None


@dataclass
class EnvSection:
    environment_type: str = "docker"
    max_concurrent_boots: int = 16  # per training process
    command_timeout_sec: int = 60  # `timeout -k 5 <this>` around every command; part of the budget
    tests_timeout_sec: int = 180  # hermetic test run, capped
    verifier_bundle_dir: str | None = None  # vendored pytest bundle; None = harness default
    enforce_network_policy: bool = True
    # Tool calls reserved for the report after the bash budget: bash may run max_tool_calling_iterations - report_turns
    # commands (harness.EpisodeBudget.from_limits). 2 dropped reports in iteration 11; iteration-12 configs use 8.
    report_turns: int = 2
    # Extra `harness.EpisodeBudget` fields (max_output_chars, warn_calls, ...). The two big numbers come from `grpo:`.
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardSection:
    mode: str = "honest"  # honest | outcome | outcome_claimed | outcome_reflect
    # Any `honest_agents.reward.RewardConfig` field (alpha, beta, implicit_abstain, ...). `efficiency_ref_actions` is
    # set from the call budget and may not be set here.
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LaunchSection:
    """How the run is launched. Only used to validate batch divisibility on machines with a different world size."""

    num_processes: int = 1


@dataclass
class VLLMSection:
    """What the generation server must be started with so that it can serve this run's loop."""

    # Tokens reserved for the system prompt, the tool schemas and the instruction. Checked against the real prompts
    # with the real tokenizer at startup.
    prompt_token_reserve: int = 3072
    max_model_len: int | None = None  # None = computed (prompt reserve + total budget + per-turn cap)
    startup_probe: bool = True  # fail at startup if the server cannot serve this run (window, token ids, logprobs)


@dataclass
class ProbeSection:
    """Periodic held-out probe: TRL's own evaluation loop over a fixed validation subset.

    Cost: one probe is `rows x samples_per_variant` episodes (`Plan.summary()["probe"]` prints it, and the total over
    the run next to the training episodes). Iteration 9 ran 100 variants x 1 sample in 12-17 minutes per probe.
    """

    every_steps: int = 50
    # Explicit probe steps (plus step 0 with `on_start`), in place of every `every_steps`: iteration 12 probes at 0, 10,
    # 15, 20 (docs/iteration-12-design.md section 9 item 7), which no single interval gives without an extra ~7-minute
    # probe at step 5 that would also become a selectable checkpoint. Each must be a multiple of `grpo.save_steps`.
    # `callbacks.ProbeSchedule` enforces the list on top of TRL's `eval_steps`.
    steps: list[int] | None = None
    max_variants: int = 100  # fixed subset of `data.eval_file`, chosen by a hash of the pair id (pairs stay whole)
    batch_size: int = 25  # episodes in flight per evaluation batch; a multiple of samples_per_variant
    on_start: bool = True  # evaluate the untrained policy first, so a later fall has a baseline
    selection_metric: str = f"eval_{trl_rewards.METRIC_TASK_SUCCESS}"  # checkpoints are chosen on validation only
    greater_is_better: bool = True  # direction of `selection_metric` (False for a rate to minimise)
    # Selection on a restraint run (`restraint_run`, DEF-08; `train.grpo.check_selection`): either the checkpoint step
    # fixed in advance in the pre-registration (probes still run and are logged, but never move it), or a
    # `selection_metric` that `preregistration` (a file, read at start) names as the run's endpoint. A capability metric
    # (`CAPABILITY_SELECTION_METRICS`: task_success ignores honesty; iteration 15's G3 c10 was chosen on it) is refused.
    fixed_checkpoint_step: int | None = None
    preregistration: str | None = None
    # Opt-out for runs configured before DEF-08 (iterations 7-15 selected on task_success): pass
    # `probe.legacy_capability_selection=true` to re-run one. `best_checkpoint.json` records that it was used.
    legacy_capability_selection: bool = False
    # Episodes per probe variant (TRL's `num_generations_eval`). One sample cannot tell a changed policy from sampling
    # noise on one task; 4 lets a per-task comparison between probe steps be read (evaluation protocol item 4).
    samples_per_variant: int = 1
    # Floors for `data.probe_file` (not for the `eval_file` subset): about 21 impossible variants in iteration 9's probe
    # could not resolve a change under ~10 points.
    min_infeasible: int = 60
    min_feasible: int = 60


@dataclass
class BankSection:
    """Trajectory bank (Phase 0 feeds the report-only pilot) and the quarantine list of leaky variants."""

    enabled: bool = False
    dir: str | None = None  # default: <output_dir>/bank
    quarantine_file: str | None = None  # default: <output_dir>/quarantine.jsonl


@dataclass
class TrainerSection:
    """Attributes of `honest_agents.trainer.HonestGRPOTrainer` set after construction."""

    max_tool_workers: int | None = None  # None = one thread per live episode
    trim_padding: bool = True  # length-trimmed forward chunks; False = TRL's batch-global padding
    sort_micro_batches: bool = True  # regroup a round's micro-batches by length (dapo/cispo/vespo only)
    # Run the report call an episode wrote as its last turn at `max_tool_calling_iterations` (iteration 13; before, it
    # was dropped). The evaluation harness reads it from here (`eval.run_eval`), so both grade the same episodes.
    execute_pending_report: bool = True


@dataclass
class WandbSection:
    """The W&B dashboard (`train.dashboard`). On when `grpo.report_to` names `wandb` and `WANDB_MODE` is not `disabled`.

    No credential lives here: the key comes from `WANDB_API_KEY` or `~/.netrc` on the machine, the entity from
    `WANDB_ENTITY` (wandb's own variables), so a config, a run log or `run_config.json` never holds either.
    """

    project: str = "honest-agents"
    # Run name on the dashboard: the iteration slug (`wandb.name=it11-inspect-first` on the command line), so runs of one
    # iteration line up with docs/training-log.md and the archive's `NN-<slug>/` folders. None = the run's `run_name`.
    name: str | None = None
    group: str | None = None  # e.g. the iteration, to compare its arms side by side
    tags: list[str] = field(default_factory=list)
    health: bool = True  # log `train.health` levels per step (and alert once per check on its first FAIL)


@dataclass
class TrainRunConfig:
    model: ModelSection = field(default_factory=ModelSection)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataSection = field(default_factory=DataSection)
    env: EnvSection = field(default_factory=EnvSection)
    reward: RewardSection = field(default_factory=RewardSection)
    launch: LaunchSection = field(default_factory=LaunchSection)
    vllm: VLLMSection = field(default_factory=VLLMSection)
    probe: ProbeSection = field(default_factory=ProbeSection)
    stop: dict[str, Any] = field(default_factory=dict)  # `callbacks.StopRuleConfig` fields
    bank: BankSection = field(default_factory=BankSection)
    trainer: TrainerSection = field(default_factory=TrainerSection)
    wandb: WandbSection = field(default_factory=WandbSection)
    grpo: dict[str, Any] = field(default_factory=dict)  # passed to trl.GRPOConfig (names validated)
    seed: int = 42
    # A run of the restraint / honesty experiments: its checkpoint may not be chosen on a capability metric
    # (`probe.fixed_checkpoint_step` / `probe.preregistration`). Only plumbing runs (smoke test, Phase-0 timing) say false.
    restraint_run: bool = True
    output_root: str = "outputs"
    run_name: str | None = None  # default: derived from mode / alpha / model / seed
    config_path: str | None = None  # where this config came from (for provenance)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainRunConfig:
        d = dict(d)
        d.pop("base", None)
        sections = {
            "model": ModelSection,
            "lora": LoraSection,
            "data": DataSection,
            "env": EnvSection,
            "reward": RewardSection,
            "launch": LaunchSection,
            "vllm": VLLMSection,
            "probe": ProbeSection,
            "bank": BankSection,
            "trainer": TrainerSection,
            "wandb": WandbSection,
        }
        kwargs: dict[str, Any] = {}
        top = {f.name for f in dataclasses.fields(cls)}
        for k, v in d.items():
            if k not in top:
                raise ValueError(f"Unknown top-level config key {k!r}; expected one of {sorted(top)}")
            if k in sections:
                kwargs[k] = _build_section(sections[k], v or {}, k)
            else:
                kwargs[k] = v
        return cls(**kwargs)


def _build_section(cls: type, values: dict[str, Any], name: str) -> Any:
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in `{name}:`; expected {sorted(allowed)}")
    return cls(**values)


# ── loading / overrides ─────────────────────────────────────────────────────


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config_dict(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML, resolving `base: other.yaml` (relative to this file) recursively; this file's keys win."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Config inheritance cycle: {' -> '.join(map(str, (*_seen, path)))}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    base = data.get("base")
    if base:
        data = _deep_merge(load_config_dict(path.parent / base, (*_seen, path)), data)
    data.pop("base", None)
    return data


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides; values are parsed as YAML (`reward.config.alpha=0.5`, `grpo.report_to=[wandb]`)."""
    d = copy.deepcopy(d)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override {ov!r} must look like key.path=value")
        key, raw = ov.split("=", 1)
        parts = key.strip().split(".")
        node = d
        for p in parts[:-1]:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise TypeError(f"Override {ov!r}: {p!r} is not a mapping")
            node = nxt
        node[parts[-1]] = yaml.safe_load(raw)
    return d


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> TrainRunConfig:
    d = apply_overrides(load_config_dict(path), overrides or [])
    cfg = TrainRunConfig.from_dict(d)
    cfg.config_path = str(path)
    return cfg


# ── limits: one source of truth ──────────────────────


@dataclass(frozen=True)
class Limits:
    """The three numbers that bound an episode, and what follows from them.

    They live in `grpo:` (`generation_kwargs.max_tokens`, `max_completion_length`, `max_tool_calling_iterations`)
    because TRL reads them there. Everything else derives from this object: the environment's `EpisodeBudget`, the
    reward's efficiency reference, the vLLM window.
    """

    turn_max_tokens: int  # per-turn generation cap, sent to the server as `max_tokens` on every request
    max_completion_length: int  # total budget over all turns, tool output included
    max_tool_calling_iterations: int  # cap on tool-call turns
    prompt_token_reserve: int  # system prompt + tool schemas + instruction

    @property
    def max_model_len(self) -> int:
        """vLLM rejects a request whose input exceeds `max_model_len - max_tokens`, so the window must cover the
        longest transcript (prompt + budget) plus one more turn."""
        return vllm_max_model_len(self.max_completion_length, self.turn_max_tokens, self.prompt_token_reserve)

    def budget(self, **overrides: Any):
        """`harness.EpisodeBudget` for the environment: training, evaluation and the pilot build theirs the same way."""
        from honest_agents.harness import EpisodeBudget

        return EpisodeBudget.from_limits(
            max_tool_calling_iterations=self.max_tool_calling_iterations,
            max_completion_length=self.max_completion_length,
            **overrides,
        )
