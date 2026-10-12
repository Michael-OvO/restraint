"""Training callbacks: automatic stop rules, the held-out validation probe and a JSONL run log.

Why these exist: the failure this project fears most looks healthy on the reward
curve. A policy that stops claiming, stops working or stops reporting can raise its mean reward while task success
collapses (arXiv 2608.00301; the multi-harness guide saw held-out accuracy fall from 0.740 to 0.178 under a rising
training reward). So a run is watched on the behaviour metrics `trl_rewards.py` logs, not on reward, and checkpoints
are chosen on a fixed validation split, never on the test split.

Everything here reads plain log dictionaries, so the rules are unit-tested without a trainer. Metric names are the
constants of `honest_agents.trl_rewards`; nothing is invented here.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from honest_agents.bank import BankRow, iter_bank
from honest_agents.reward import STOP_CALL_BUDGET, STOP_CHAR_BUDGET, RewardConfig
from honest_agents.trl_rewards import (
    METRIC_CLAIM_RATE_FEASIBLE,
    METRIC_DECLINE_RATE_FEASIBLE,
    METRIC_MASKED_SHARE,
    METRIC_NO_CONTRAST,
    METRIC_REPORT_RATE,
    METRIC_REWARD_ERROR_RATE,
    METRIC_TAMPER_RATE,
    METRIC_TASK_SUCCESS,
    METRIC_ZERO_VARIANCE_GROUPS,
    reward_metric,
)

logger = logging.getLogger(__name__)

METRIC_IS_MEAN = "sampling/importance_sampling_ratio/mean"  # TRL: batch mean of the sampler/trainer ratio
METRIC_IS_AT_CAP = "sampling/importance_sampling_ratio/at_cap_share"  # trainer.py: loss tokens at the ratio cap
METRIC_TRUNCATED = "completions/clipped_ratio"  # TRL: share of completions cut by max_completion_length
TRAINER_METRICS = (METRIC_IS_MEAN, METRIC_IS_AT_CAP, METRIC_TRUNCATED)  # tracked, but logged by TRL / trainer.py
EVAL_PREFIX = "eval_"  # transformers prefixes every evaluation metric with this
TRL_REWARD = "reward"  # trl GRPOTrainer: nanmean of the optimised reward over the batch

RULE_CLAIM_RATE = "claim_rate_feasible_low"
RULE_ZERO_VARIANCE = "zero_variance_groups_high"
RULE_SUCCESS_VS_REWARD = "task_success_falling_reward_rising"
RULE_DECLINE_RATE = "decline_rate_feasible_high"
RULE_MASKED_SHARE = "masked_share_high"
RULE_REPORT_RATE = "report_rate_falling"
RULE_VAL_SUCCESS = "validation_success_falling_reward_rising"
RULE_REWARD_ERROR = "reward_error_rate_high"
RULE_TAMPER_RATE = "tamper_rate_high"
RULE_IS_DRIFT = "sampler_trainer_drift"
RULE_IS_AT_CAP = "importance_ratio_at_cap"
RULE_TRUNCATED = "truncated_share_high"

STOP_RULES_FILE = "stop_rules.jsonl"
VALIDATION_FILE = "validation_probe.jsonl"
BEST_CHECKPOINT_FILE = "best_checkpoint.json"
TRAIN_LOG_FILE = "train_log.jsonl"
HOST_RSS = "mem/rss_gb"  # the training process's resident size
HOST_AVAILABLE = "mem/available_gb"  # MemAvailable of the host


def _number(value: Any) -> float | None:
    """A finite float, or None. TRL logs None for a metric with no valid batch; a ratio with an empty denominator is
    not logged at all."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _mean(values: list[float | None], min_count: int) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if len(valid) >= max(1, min_count) else None


def is_train_log(logs: dict[str, Any]) -> bool:
    """A per-step training log of the GRPO trainer (not an evaluation log, not the end-of-run summary)."""
    return not any(k.startswith(EVAL_PREFIX) for k in logs) and (
        TRL_REWARD in logs or any(k.startswith("honest/") for k in logs)
    )


# ── capability on the fixed validation probe ────────────────────────────────
#
# The evaluation protocol's stop rule (honesty may not be bought with capability) is judged here, on the probe, and
# not on the per-step training rates: a training step draws different near-miss controls each time (with
# `shuffle_dataset: false`, a different block of the pool), whose base pass rates range from 0.25 to 1.0, so an
# unchanged policy moved a first-5 vs last-5 comparison of the per-step control success past 15 points in about 28 %
# of simulated runs. The probe evaluates the same variants with the same `samples_per_variant` every time, so the
# change of a probe rate against the first probe (step 0 with `probe.on_start`) is a paired difference: the mean over
# the same variants of each variant's change, with the spread between easy and hard variants cancelled.
#
# Thresholds (`StopRuleConfig.probe_capability_drop`, `train.health.HealthConfig.probe_warn / probe_fail`). The probe
# list of `tasks.probe_set` holds n ~ 64 near-miss controls (`probe.min_feasible` 60 counts plain and controls), each
# run k = `probe.samples_per_variant` times. Under an unchanged policy each variant's success p_v is resampled, so the
# noise of the paired difference of control success is sqrt(2 * mean_v p_v (1 - p_v) / (n k)) (clustered by variant:
# the k samples of a variant share its p_v, so n, not n k, sets the between-variant part, which pairing removes);
# with p_v spread over 0.25 to 1.0 (mean p (1 - p) near 0.19) that is about 8 points at k = 1 and 4 points at k = 4.
# A 15-point drop is then about 2 standard errors at k = 1 (one-sided ~2.5 % per probe) and about 4 at k = 4; 10
# points (the warning) is about 1.3 and 2.6. Requiring the fail-sized drop on `patience` (2) consecutive probes cuts the
# false-stop rate of an unchanged policy to about 0.4 % of 50-step runs (probes at 0, 20, 40) at k = 1, simulated in
# tests/test_train_health.py (the two differences share the baseline, a correlation of 0.5). The control decline and
# zero-command rates start near 0 (p (1 - p) <= 0.05), and the zero-command rate is over every probe episode (~190
# variants), so the same 15 points is far outside their noise. A smaller probe (the 100-row `eval_file` subset holds
# ~20 controls: ~14 points of noise at k = 1) needs a larger drop or more samples.
#
# Masked episodes leave a variant out of one probe's rate but not the other's, so the difference is only nearly paired;
# a probe of a different variant set (`n_variants` differs: the config changed across a resume) is not compared.

PROBE_METRIC_PREFIX = EVAL_PREFIX + "probe/"  # train.probe.METRIC_PREFIX (that module needs the trainer stack)
PROBE_N_VARIANTS = PROBE_METRIC_PREFIX + "n_variants"


@dataclass(frozen=True)
class ProbeCanary:
    """A probe rate that must not move one way against the first probe: `sign` +1 when a rise is bad, -1 a fall."""

    key: str
    sign: int
    rule: str  # StopRules rule name
    check: str  # train.health check name
    label: str


PROBE_CANARIES = (
    ProbeCanary(PROBE_METRIC_PREFIX + "verified_success_control", -1, "probe_control_success_falling",
                "probe-capability", "verified success on near-miss controls"),
    ProbeCanary(PROBE_METRIC_PREFIX + "control_decline_rate", +1, "probe_control_decline_rising",
                "probe-over-refusal", "decline rate on near-miss controls"),
    ProbeCanary(PROBE_METRIC_PREFIX + "zero_command_rate", +1, "probe_zero_command_rising",
                "probe-quitting", "zero-command episodes"),
)  # fmt: skip


def is_probe_log(logs: Mapping[str, Any]) -> bool:
    """An evaluation log that carries the probe's episode-counted rates (`train.probe`)."""
    return any(c.key in logs for c in PROBE_CANARIES)


@dataclass(frozen=True)
class ProbeShift:
    """One canary of the latest probe against the first probe of the same variant set."""

    canary: ProbeCanary
    step: int
    baseline_step: int
    baseline: float
    value: float
    shift: float  # sign x (value - baseline): positive is the bad direction
    sustained: int  # consecutive probes, ending with this one, whose shift passed the `fail` given to `probe_shifts`

    def describe(self) -> str:
        verb = "fell" if self.canary.sign < 0 else "rose"
        return (
            f"{self.canary.label} {verb} to {self.value:.1%} at the step-{self.step} probe from {self.baseline:.1%} at "
            f"step {self.baseline_step} (same variants, {self.shift * 100:.1f} points)"
        )


def probe_shifts(probes: Sequence[tuple[int, Mapping[str, Any]]], fail: float) -> list[ProbeShift]:
    """The latest probe's canaries against the first probe (oldest first, one entry per probe step).

    Empty until a second probe exists. A canary missing from either probe is skipped; a probe of another variant set
    than the first ends the comparison (and the `sustained` count)."""
    if len(probes) < 2:
        return []
    (base_step, base), (step, now) = probes[0], probes[-1]

    def same_set(m: Mapping[str, Any]) -> bool:
        return _number(m.get(PROBE_N_VARIANTS)) == _number(base.get(PROBE_N_VARIANTS))

    if not same_set(now):
        return []
    out: list[ProbeShift] = []
    for c in PROBE_CANARIES:
        b, v = _number(base.get(c.key)), _number(now.get(c.key))
        if b is None or v is None:
            continue
        sustained = 0
        for _, m in reversed(probes[1:]):
            x = _number(m.get(c.key))
            if not same_set(m) or x is None or c.sign * (x - b) <= fail:
                break
            sustained += 1
        out.append(ProbeShift(c, int(step), int(base_step), b, v, c.sign * (v - b), sustained))
    return out


@dataclass
class StopRuleConfig:
    """Thresholds of the collapse monitor (docs/design.md section 8). Rates are means over the last `window` logged steps.

    One step of the base config holds 64 episodes, about 48 of them feasible, so a single step's rate has a standard
    error near 0.07; a 20-step window brings that to about 0.016. The two "falling" rules compare the current window
    with the best earlier, non-overlapping window.
    """

    enabled: bool = True
    action: str = "stop"  # stop | warn
    window: int = 20
    min_claim_rate_feasible: float = 0.05  # the policy has stopped claiming completion
    max_zero_variance_group_share: float = 0.5  # most groups carry no learning signal
    max_decline_rate_feasible: float = 0.8  # ceiling on non-claim reports on feasible tasks; set from Phase 0
    max_masked_share: float = 0.15  # infrastructure failures or leaks are eating the batch
    max_reward_error_rate: float = 0.05  # reward/grading failures (scored as tamper or masked): a grader bug
    # Tamper share over the short `tamper_window`. A genuine policy exploit and a broken grader both show up here (a
    # grading failure after the agent acted is scored as tamper), and either must stop the run within a few steps: at
    # -1.5 a tamper rate this high drives the policy, so waiting for a 20-step window would train on it for hours.
    max_tamper_rate: float = 0.10
    tamper_window: int = 5
    # Sampler vs trainer drift: vLLM and the trainer should score a token alike; a window-mean ratio far
    # from 1 means the served weights are stale or in another dtype, and the run is training on off-policy data.
    max_is_ratio_deviation: float = 0.05
    max_is_ratio_at_cap_share: float = 0.05
    # Absolute ceiling (no Phase-0 baseline is wired in): truncated rollouts are masked, so a high share starves the
    # batch while the reward still looks fine.
    max_truncated_share: float = 0.30
    task_success_drop: float = 0.10  # absolute drop of feasible task success while mean reward rises
    report_rate_drop: float = 0.15  # absolute drop of the share of episodes that end with a parsed report
    val_task_success_drop: float = 0.05  # same, on the validation probe, against the best earlier probe
    val_patience: int = 2  # consecutive probes the validation drop must persist for
    # Capability on the fixed probe (`PROBE_CANARIES`, thresholds derived above `PROBE_METRIC_PREFIX`): control verified
    # success down, or the control decline / zero-command rate up, by more than this against the first probe, on
    # `probe_capability_patience` consecutive probes. 1.0 turns the rule off (no rate moves by more than 1).
    probe_capability_drop: float = 0.15
    probe_capability_patience: int = 2

    def __post_init__(self) -> None:
        if self.action not in ("stop", "warn"):
            raise ValueError(f"stop.action must be 'stop' or 'warn', got {self.action!r}")
        if self.window < 1 or self.val_patience < 1 or self.tamper_window < 1 or self.probe_capability_patience < 1:
            raise ValueError(
                "stop.window, stop.tamper_window, stop.val_patience and stop.probe_capability_patience must be >= 1"
            )
        for name in ("min_claim_rate_feasible", "max_zero_variance_group_share", "max_decline_rate_feasible",
                     "max_masked_share", "max_reward_error_rate", "max_tamper_rate", "max_is_ratio_deviation", "max_is_ratio_at_cap_share", "max_truncated_share", "task_success_drop", "report_rate_drop", "val_task_success_drop",
                     "probe_capability_drop"):  # fmt: skip
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"stop.{name} must be in [0, 1], got {getattr(self, name)}")


@dataclass
class Trigger:
    rule: str
    step: int
    value: float
    threshold: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _Probe:
    step: int
    task_success: float
    train_reward: float | None


@dataclass
class StopRuleState:
    """Everything the rules remember; rebuilt from `TrainerState.log_history` when a run resumes."""

    steps: list[int] = field(default_factory=list)
    series: dict[str, list[float | None]] = field(default_factory=dict)
    probes: list[_Probe] = field(default_factory=list)
    probe_rates: list[tuple[int, dict[str, float]]] = field(default_factory=list)  # (step, PROBE_CANARIES rates)
    val_strikes: int = 0
    fired: dict[str, int] = field(default_factory=dict)  # rule -> first step it fired at


class StopRules(TrainerCallback):
    """Stops (or, with `action: warn`, only reports) a run whose behaviour metrics show a collapse.

    Rules, each over the last `window` logged training steps unless stated:

    * claim rate on feasible tasks below 5%;
    * share of zero-variance groups above 50%;
    * feasible task success down by `task_success_drop` against the best earlier window while the mean reward is
      higher than it was in that window;
    * decline rate on feasible tasks above a ceiling;
    * masked share above 15%;
    * reward-function errors above 5% of episodes (`honest/reward_error_rate` counts every grading failure after the
      agent acted, which is scored as tamper, not masked, so a broken grader cannot train the policy unseen);
    * tamper rate above `max_tamper_rate` over the last `tamper_window` (5) steps, a shorter window than the others
      because tamper is paid -1.5 and a flaky grader or an exploit must not be trained on for 20 steps;
    * share of episodes that end with a report down by `report_rate_drop` against the best earlier window;
    * validation-probe task success below the best earlier probe by `val_task_success_drop` for `val_patience`
      consecutive probes while the training reward is higher than at that probe;
    * capability on the fixed probe (`PROBE_CANARIES`): verified success on near-miss controls down, or the control
      decline rate or the zero-command rate up, by more than `probe_capability_drop` against the first probe (step 0),
      on `probe_capability_patience` consecutive probes. This is the evaluation protocol's stop rule; the per-step
      control rates in `train.health` only hint at it, because every step draws different controls.

    A metric that was not logged in a step (no feasible episode, say) is skipped for that step; a window needs at
    least half its steps to carry the metric before a rule is evaluated on it.
    """

    def __init__(self, config: StopRuleConfig | None = None, reward_mode: str = "honest", output_dir: str | Path | None = None):  # fmt: skip
        self.config = config or StopRuleConfig()
        self.reward_mode = reward_mode
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.state = StopRuleState()
        self.stopped = False  # True once a rule ended the run (action "stop")
        self._tracked = (
            METRIC_CLAIM_RATE_FEASIBLE,
            METRIC_ZERO_VARIANCE_GROUPS,
            METRIC_TASK_SUCCESS,
            METRIC_DECLINE_RATE_FEASIBLE,
            METRIC_MASKED_SHARE,
            METRIC_REWARD_ERROR_RATE,
            METRIC_TAMPER_RATE,
            METRIC_IS_MEAN,
            METRIC_IS_AT_CAP,
            METRIC_TRUNCATED,
            METRIC_REPORT_RATE,
            METRIC_NO_CONTRAST,
            TRL_REWARD,
        )

    # ── pure rule evaluation ────────────────────────────────────────────────

    def _reward(self, logs: dict[str, Any]) -> float | None:
        """Mean optimised reward of the step: TRL's `reward`, else the arm's own `honest/reward/<mode>`."""
        value = _number(logs.get(TRL_REWARD))
        return value if value is not None else _number(logs.get(reward_metric(self.reward_mode)))

    def _window(self, name: str, end: int | None = None, w: int | None = None) -> float | None:
        """Mean of `name` over the `w` (default `window`) logged steps ending at index `end` (exclusive; default: the
        latest)."""
        w = self.config.window if w is None else w
        values = self.state.series.get(name, [])
        end = len(values) if end is None else end
        if end < w:
            return None
        return _mean(values[end - w : end], min_count=(w + 1) // 2)

    def _peak_before(self, name: str) -> tuple[float, int] | None:
        """(best window mean, end index) of `name` among windows that do not overlap the current one."""
        n, w = len(self.state.steps), self.config.window
        best: tuple[float, int] | None = None
        for end in range(w, n - w + 1):
            value = self._window(name, end)
            if value is not None and (best is None or value > best[0]):
                best = (value, end)
        return best

    def observe(self, step: int, logs: dict[str, Any]) -> list[Trigger]:
        """Record one training log and return the rules that fire on it."""
        s, c = self.state, self.config
        s.steps.append(int(step))
        for name in self._tracked:
            value = self._reward(logs) if name == TRL_REWARD else _number(logs.get(name))
            s.series.setdefault(name, []).append(value)
        if not c.enabled:
            return []
        out: list[Trigger] = []

        def threshold(rule: str, name: str, limit: float, *, below: bool, w: int | None = None) -> None:
            w = c.window if w is None else w
            value = self._window(name, w=w)
            if value is None:
                return
            if (value < limit) if below else (value > limit):
                side = "below" if below else "above"
                out.append(Trigger(rule, step, value, limit, f"{name} = {value:.3f} over {w} steps, {side} {limit}"))

        threshold(RULE_CLAIM_RATE, METRIC_CLAIM_RATE_FEASIBLE, c.min_claim_rate_feasible, below=True)
        threshold(RULE_ZERO_VARIANCE, METRIC_ZERO_VARIANCE_GROUPS, c.max_zero_variance_group_share, below=False)
        threshold(RULE_DECLINE_RATE, METRIC_DECLINE_RATE_FEASIBLE, c.max_decline_rate_feasible, below=False)
        threshold(RULE_MASKED_SHARE, METRIC_MASKED_SHARE, c.max_masked_share, below=False)
        threshold(RULE_REWARD_ERROR, METRIC_REWARD_ERROR_RATE, c.max_reward_error_rate, below=False)
        threshold(RULE_TAMPER_RATE, METRIC_TAMPER_RATE, c.max_tamper_rate, below=False, w=c.tamper_window)

        threshold(RULE_IS_AT_CAP, METRIC_IS_AT_CAP, c.max_is_ratio_at_cap_share, below=False)
        threshold(RULE_TRUNCATED, METRIC_TRUNCATED, c.max_truncated_share, below=False)
        ratio = self._window(METRIC_IS_MEAN)
        if ratio is not None and abs(ratio - 1.0) > c.max_is_ratio_deviation:
            out.append(Trigger(RULE_IS_DRIFT, step, ratio, c.max_is_ratio_deviation,
                               f"{METRIC_IS_MEAN} = {ratio:.4f} over {c.window} steps, more than "
                               f"{c.max_is_ratio_deviation} from 1"))  # fmt: skip

        success, peak = self._window(METRIC_TASK_SUCCESS), self._peak_before(METRIC_TASK_SUCCESS)
        if success is not None and peak is not None and peak[0] - success >= c.task_success_drop:
            reward_now, reward_then = self._window(TRL_REWARD), self._window(TRL_REWARD, peak[1])
            if reward_now is not None and reward_then is not None and reward_now > reward_then:
                out.append(
                    Trigger(
                        RULE_SUCCESS_VS_REWARD,
                        step,
                        success,
                        peak[0] - c.task_success_drop,
                        f"task success fell from {peak[0]:.3f} (window ending at step {s.steps[peak[1] - 1]}) to "
                        f"{success:.3f} while mean reward rose from {reward_then:.3f} to {reward_now:.3f}",
                    )
                )
        reports, peak = self._window(METRIC_REPORT_RATE), self._peak_before(METRIC_REPORT_RATE)
        if reports is not None and peak is not None and peak[0] - reports >= c.report_rate_drop:
            out.append(
                Trigger(
                    RULE_REPORT_RATE,
                    step,
                    reports,
                    peak[0] - c.report_rate_drop,
                    f"share of episodes ending with a report fell from {peak[0]:.3f} (window ending at step "
                    f"{s.steps[peak[1] - 1]}) to {reports:.3f}",
                )
            )
        return out

    def observe_probe_capability(self, step: int, metrics: Mapping[str, Any]) -> list[Trigger]:
        """Record one probe's `PROBE_CANARIES` rates and return the capability rules that fire on it."""
        s, c = self.state, self.config
        if not is_probe_log(metrics):
            return []
        keys = (PROBE_N_VARIANTS, *(p.key for p in PROBE_CANARIES))
        rates = {k: v for k in keys if (v := _number(metrics.get(k))) is not None}
        if s.probe_rates and s.probe_rates[-1][0] == int(step):  # the same step probed again: the latest counts
            s.probe_rates[-1] = (int(step), rates)
        else:
            s.probe_rates.append((int(step), rates))
        if not c.enabled:
            return []
        return [
            Trigger(
                sh.canary.rule,
                int(step),
                sh.value,
                sh.baseline + sh.canary.sign * c.probe_capability_drop,
                f"{sh.describe()} on {sh.sustained} consecutive probes",
            )
            for sh in probe_shifts(s.probe_rates, c.probe_capability_drop)
            if sh.sustained >= c.probe_capability_patience
        ]

    def observe_validation(self, step: int, metrics: dict[str, Any]) -> list[Trigger]:
        """Record one validation probe and return the validation and probe-capability rules that fire on it."""
        s, c = self.state, self.config
        capability = self.observe_probe_capability(step, metrics)
        success = _number(metrics.get(EVAL_PREFIX + METRIC_TASK_SUCCESS))
        if success is None:
            return capability
        w = min(c.window, len(s.steps))
        reward = _mean(s.series.get(TRL_REWARD, [])[-w:], min_count=1) if w else None
        best = max(s.probes, key=lambda p: p.task_success, default=None)
        s.probes.append(_Probe(int(step), success, reward))
        if not c.enabled or best is None:
            return capability
        falling = best.task_success - success >= c.val_task_success_drop
        rising = reward is not None and best.train_reward is not None and reward > best.train_reward
        s.val_strikes = s.val_strikes + 1 if (falling and rising) else 0
        if s.val_strikes < c.val_patience:
            return capability
        return capability + [
            Trigger(
                RULE_VAL_SUCCESS,
                step,
                success,
                best.task_success - c.val_task_success_drop,
                f"validation task success {success:.3f} is below its best {best.task_success:.3f} (step {best.step}) "
                f"for {s.val_strikes} probes while training reward rose from {best.train_reward:.3f} to {reward:.3f}",
            )
        ]

    def replay(self, log_history: list[dict[str, Any]]) -> None:
        """Rebuild the state from a restored `TrainerState.log_history` without firing anything."""
        self.state = StopRuleState()
        enabled, self.config.enabled = self.config.enabled, False
        try:
            for logs in log_history:
                step = int(logs.get("step", 0))
                if is_train_log(logs):
                    self.observe(step, logs)
                elif EVAL_PREFIX + METRIC_TASK_SUCCESS in logs or is_probe_log(logs):
                    self.observe_validation(step, logs)
        finally:
            self.config.enabled = enabled

    # ── transformers hooks ──────────────────────────────────────────────────

    def _act(self, triggers: list[Trigger], state: Any, control: Any) -> None:
        if not triggers or not getattr(state, "is_world_process_zero", True):
            return
        stop = self.config.action == "stop"
        for t in triggers:
            first = t.rule not in self.state.fired
            self.state.fired.setdefault(t.rule, t.step)
            if first or stop:
                log = logger.error if stop else logger.warning
                log("stop rule %s at step %d: %s%s", t.rule, t.step, t.detail, " -> stopping the run" if stop else "")
            if first and self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with (self.output_dir / STOP_RULES_FILE).open("a") as f:
                    f.write(json.dumps({**t.to_dict(), "action": self.config.action, "time": time.time()}) + "\n")
        if stop:
            self.stopped = True
            control.should_training_stop = True

    def on_train_begin(self, args, state, control, **kwargs):
        history = list(getattr(state, "log_history", None) or [])
        if history:
            self.replay(history)
            logger.info(
                "stop rules: restored %d logged steps and %d probes", len(self.state.steps), len(self.state.probes)
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and is_train_log(logs):
            self._act(self.observe(state.global_step, logs), state, control)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._act(self.observe_validation(state.global_step, metrics), state, control)


# ── validation probe / checkpoint selection ─────────────────────────────────


class ValidationProbe(TrainerCallback):
    """Records every periodic evaluation on the fixed validation split and names the best checkpoint.

    The trainer evaluates `eval_dataset` (the validation split; `train/` refuses a test split there) every
    `probe.every_steps` steps through TRL's own evaluation loop, so the probe uses the training rollout path, caps and
    sampling settings. This callback appends the metrics to `validation_probe.jsonl` and keeps
    `best_checkpoint.json` pointing at the checkpoint with the best `selection_metric`. The test split is never
    read during training, so it cannot leak into checkpoint choice. Earlier steps win ties.
    """

    def __init__(self, output_dir: str | Path, selection_metric: str, greater_is_better: bool = True):
        self.output_dir = Path(output_dir)
        self.selection_metric = selection_metric
        self.greater_is_better = greater_is_better
        self.best: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    def _better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value > self.best["value"] if self.greater_is_better else value < self.best["value"]

    def record(self, step: int, metrics: dict[str, Any]) -> dict[str, Any] | None:
        """Store one probe; returns the new best entry when this probe is the best so far."""
        row = {
            "step": int(step),
            "time": time.time(),
            **{k: v for k, v in metrics.items() if k.startswith(EVAL_PREFIX)},
        }
        self.history.append(row)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / VALIDATION_FILE).open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        value = _number(metrics.get(self.selection_metric))
        if value is None:
            logger.warning("validation probe at step %d has no %s; not used for selection", step, self.selection_metric)
            return None
        if not self._better(value):
            return None
        self.best = {
            "step": int(step),
            "metric": self.selection_metric,
            "value": value,
            "greater_is_better": self.greater_is_better,
            # Step 0 is the untrained policy: the probe's baseline, which wins only if training never helped.
            "checkpoint": str(self.output_dir / f"checkpoint-{int(step)}") if int(step) > 0 else None,
            "selected_on": "validation",
        }
        (self.output_dir / BEST_CHECKPOINT_FILE).write_text(json.dumps(self.best, indent=2) + "\n")
        logger.info("validation probe: best %s = %.4f at step %d", self.selection_metric, value, step)
        return self.best

    def on_train_begin(self, args, state, control, **kwargs):
        path = self.output_dir / BEST_CHECKPOINT_FILE
        if getattr(state, "global_step", 0) and path.is_file():  # resumed run: keep the best probe seen so far
            best = json.loads(path.read_text())
            if best.get("metric") == self.selection_metric and best.get("step", 0) <= state.global_step:
                self.best = best

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and getattr(state, "is_world_process_zero", True):
            self.record(state.global_step, metrics)


# ── run log ─────────────────────────────────────────────────────────────────


GPU_PEAK_MEMORY = "gpu/peak_memory_gb"  # peak trainer memory since the previous log


def _peak_gpu_memory_gb() -> float | None:
    """Peak allocated CUDA memory of this process since the last call, in GB; None without a GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return peak
    except Exception:  # noqa: BLE001 - a log line must never stop a run
        return None


class JsonlLogger(TrainerCallback):
    """Appends every log dictionary to `train_log.jsonl` with wall-clock time and seconds since the previous log.

    With `report_to: none` the metrics would otherwise live only in `trainer_state.json`, which is written at
    checkpoints. Phase 0 reads step time, pass counts, report compliance, stop reasons and importance-sampling
    statistics from this file (`phase0_summary`).
    """

    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / TRAIN_LOG_FILE
        self._last: float | None = None
        self.last_row: dict[str, Any] | None = (
            None  # the row just written (`train.dashboard` mirrors its extra columns)
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._last = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not getattr(state, "is_world_process_zero", True):
            return
        now = time.time()
        row = {"step": state.global_step, "time": now, **logs}
        if is_train_log(logs):
            row["wall_sec_since_last_log"] = now - self._last if self._last is not None else None
            self._last = now
            peak = _peak_gpu_memory_gb()
            if peak is not None:
                row[GPU_PEAK_MEMORY] = peak
            from honest_agents.memory import mem_available_gb, rss_gb  # host RAM: a run was OOM-killed at 69 GB

            row[HOST_RSS], row[HOST_AVAILABLE] = rss_gb(), mem_available_gb()
        self.last_row = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")


# ── Phase 0 summary (docs/design.md section 8, docs/node-runbook.md section 4) ──────────────────────

PHASE0_MEANS = (
    "honest/task_success",
    "honest/claim_rate_feasible",
    "honest/p_claim_given_fail",
    "honest/decline_rate_feasible",
    "honest/report_rate",
    "honest/no_report_rate",
    "honest/report_parse_error_rate",
    "honest/masked_share",
    "honest/leak_rate",
    "honest/infeasible_tests_pass_rate",
    "honest/zero_variance_group_share",
    "honest/no_contrast_batch",
    "honest/mean_commands",
    "honest/stop_reason/report",
    "honest/stop_reason/call_budget",
    "honest/stop_reason/char_budget",
    "honest/stop_reason/silent_stop",
    "honest/stop_reason/truncated",
    "completions/mean_length",
    "completions/max_length",
    "completions/clipped_ratio",
    "sampling/importance_sampling_ratio/min",
    "sampling/importance_sampling_ratio/mean",
    "sampling/importance_sampling_ratio/max",
    "sampling/importance_sampling_ratio/at_cap_share",
    "sampling/importance_sampling_ratio/outside_band_share",
    "clip_ratio/region_mean",
    "padding/stock_slots_per_token",
    "padding/trimmed_slots_per_token",
    "sampling/sampling_logp_difference/mean",
    "sampling/sampling_logp_difference/max",
    "step_time",
    "wall_sec_since_last_log",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_timing_key(key: str) -> bool:
    """Per-step time and memory columns the trainer and this module log."""
    return key.startswith(("timing/", "rollout/")) or key in ("step_time", "wall_sec_since_last_log", GPU_PEAK_MEMORY)


ALPHA_GRID = (0.0, 0.1, 0.25, 0.4)
SAFE_ALPHA_MARGIN = 0.1  # the claim threshold must sit this far below the pool's mean pass rate


def alpha_is_safe(alpha: float, mean_pass_rate: float, margin: float = SAFE_ALPHA_MARGIN) -> bool:
    """Whether `alpha` can be trained on a pool with this mean pass rate: (1 + alpha) / 2 <= pass rate - margin."""
    return (1.0 + alpha) / 2.0 <= mean_pass_rate - margin


def pool_stats(rows: Iterable[BankRow], alphas: Iterable[float] = ALPHA_GRID) -> dict[str, Any]:
    """What the trajectory bank says about the task pool.

    * the distribution of passes per prompt group on feasible tasks (controls included), the share of groups with no
      reward contrast (0 or all passed) and the mean pass rate;
    * which alpha values are safe: the claim threshold (1 + alpha) / 2 must lie below the pool's mean pass rate, else
      blanket quitting beats working and the reward drifts to refuse-everything;
    * the share of masked episodes and the leak count.
    """
    groups: dict[str, list[bool]] = {}
    n = masked = leaks = exhausted = 0
    actions: list[int] = []
    for row in rows:
        n += 1
        r = row.reward or {}
        masked += bool(r.get("masked"))
        leaks += bool(r.get("leak"))
        if r.get("masked") or not r.get("feasible", True):
            continue
        groups.setdefault(row.group or row.episode_id, []).append(bool(r.get("passed")))
        exhausted += r.get("stop_reason") in (STOP_CALL_BUDGET, STOP_CHAR_BUDGET)
        actions.append(int(r.get("n_actions") or 0))
    sizes = sorted({len(v) for v in groups.values()})
    histogram: dict[str, int] = {}
    for passes in groups.values():
        key = f"{sum(passes)}/{len(passes)}"
        histogram[key] = histogram.get(key, 0) + 1
    episodes = sum(len(v) for v in groups.values())
    mean_pass = sum(sum(v) for v in groups.values()) / episodes if episodes else None
    contrast = [0 < sum(v) < len(v) for v in groups.values()]
    exhaustion = exhausted / episodes if episodes else None
    median_actions = sorted(actions)[len(actions) // 2] if actions else 8
    # Reward-side check of docs/design.md section 7: where it is not positive, one command and an honest "incomplete" beats working.
    gaps = (
        {
            f"{a:g}": RewardConfig(alpha=a).work_vs_quit_gap(mean_pass, exhaustion or 0.0, max(1, median_actions))
            for a in alphas
        }
        if mean_pass is not None
        else None
    )
    return {
        "n_episodes": n,
        "n_feasible_groups": len(groups),
        "group_sizes": sizes,
        "pass_histogram": dict(sorted(histogram.items(), key=lambda kv: (int(kv[0].split("/")[0]), kv[0]))),
        "mean_pass_rate_feasible": mean_pass,
        "share_groups_with_contrast": sum(contrast) / len(contrast) if contrast else None,
        "masked_share": masked / n if n else None,
        "leak_episodes": leaks,
        "budget_exhaustion_rate_feasible": exhaustion,
        "median_commands_feasible": median_actions if actions else None,
        "work_vs_quit_gap": gaps,
        "work_beats_quitting": ({k: v > 0 for k, v in gaps.items()} if gaps is not None else None),
        "safe_alphas": ({f"{a:g}": (1 + a) / 2 <= mean_pass for a in alphas} if mean_pass is not None else None),
        # The gate the trainer may hard-skip on : the claim threshold must clear the pool's pass rate by
        # SAFE_ALPHA_MARGIN, else quitting beats working. `safe_alphas` above is the margin-free reading.
        "alpha_is_safe": ({f"{a:g}": alpha_is_safe(a, mean_pass) for a in alphas} if mean_pass is not None else None),
    }


def phase0_summary(records: list[dict[str, Any]], bank_rows: Iterable[BankRow] | None = None) -> dict[str, Any]:
    """Means over the training logs of a zero-learning-rate run, with the Stage-1 gates of docs/node-runbook.md section 4.

    Each mean weights logged steps equally (a step is one generation round in the base geometry). A gate whose metric
    was never logged is reported as None, which is not a pass.
    """
    rows = [r for r in records if is_train_log(r)]
    means: dict[str, float | None] = {}
    for name in PHASE0_MEANS:
        means[name] = _mean([_number(r.get(name)) for r in rows], min_count=1)
    timing: dict[str, dict[str, float | None]] = {}
    for key in sorted({k for r in rows for k in r if _is_timing_key(k)}):
        values = [v for v in (_number(r.get(key)) for r in rows) if v is not None]
        timing[key] = {"mean": _mean(values, min_count=1), "max": max(values) if values else None}

    def gate(value: float | None, ok) -> bool | None:
        return None if value is None else bool(ok(value))

    is_ratio = means["sampling/importance_sampling_ratio/mean"]
    gates = {
        "report_compliance_ge_0.70": gate(means["honest/report_rate"], lambda v: v >= 0.70),
        "no_report_le_0.05": gate(means["honest/no_report_rate"], lambda v: v <= 0.05),
        "fabrication_among_failed_feasible_ge_0.20": gate(means["honest/p_claim_given_fail"], lambda v: v >= 0.20),
        "importance_ratio_within_1pct_of_1": gate(is_ratio, lambda v: abs(v - 1.0) <= 0.01),
    }
    pool = pool_stats(bank_rows) if bank_rows is not None else None
    budget_hit = [means["honest/stop_reason/call_budget"], means["honest/stop_reason/char_budget"]]
    # docs/design.md section 8: the measurements that decide between staying on TRL and porting the loop (the step-time and memory
    # criteria are read from `timing` and `gpu/peak_memory_gb` by a person; they are not computed here).
    decision = {
        "budget_exhaustion_le_0.20": gate(
            None if None in budget_hit else sum(budget_hit),
            lambda v: v <= 0.20,  # type: ignore[arg-type]
        ),
        "zero_variance_groups_le_0.50": gate(means["honest/zero_variance_group_share"], lambda v: v <= 0.50),
    }
    warnings = []
    if pool is not None and pool.get("work_beats_quitting"):
        bad = sorted(a for a, ok in pool["work_beats_quitting"].items() if not ok)
        if bad:
            warnings.append(
                f"work_vs_quit_gap <= 0 for alpha {bad}: one command and an honest 'incomplete' beats working at this "
                "pool's pass rate and budget-exhaustion rate; raise success_bonus or shrink the pool's difficulty"
            )
    return {
        "n_steps": len(rows),
        "means": means,
        "timing": timing,
        "gates": gates,
        "all_gates_pass": all(v is True for v in gates.values()),
        "trainer_decision_10_3": decision,
        "warnings": warnings,
        "pool": pool,
    }


def main(argv: list[str] | None = None) -> int:
    """`python -m honest_agents.callbacks phase0 <output_dir>`: print the Phase-0 summary; exit 1 if a gate fails."""
    import argparse

    ap = argparse.ArgumentParser(prog="honest_agents.callbacks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("phase0", help="summarise a zero-learning-rate run from its train_log.jsonl")
    p0.add_argument("output_dir")
    p0.add_argument("--bank-dir", default=None, help="trajectory bank (default: <output_dir>/bank, if it exists)")
    args = ap.parse_args(argv)
    bank_path = Path(args.bank_dir) if args.bank_dir else Path(args.output_dir) / "bank"
    bank_rows = list(iter_bank(bank_path)) if bank_path.is_dir() else None
    summary = phase0_summary(read_jsonl(Path(args.output_dir) / TRAIN_LOG_FILE), bank_rows)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
