"""Is a GRPO run healthy? Reads a run's `train_log.jsonl` and flags concrete failures, not just progress.

    python -m honest_agents.train.health outputs/<root>/<run> [--now UNIX_TIME] [--json]

Exit status: 0 healthy, 1 a warning, 2 a failure (stop and look). The trainer's own stop rules (`callbacks.StopRules`)
act on 20-step windows to avoid noise; this check is the operator's view after every step, so it uses short windows
(the first and the last `window` steps, 3 by default) and reports what it sees:

* stall: no logged step for `stall_factor` x the median step time (the process may be hung or dead);
* numbers: a non-finite loss or gradient norm, or a gradient norm 10x its running median;
* sampler vs trainer drift: importance-ratio mean away from 1, too much of it at the cap, or the sampler/trainer
  log-probability gap growing well past its first-step level (stale or partial weight sync looks like this);
* collapse: entropy, commands per episode or completion length falling to a fraction of their starting level;
* gaming: lazy declines, feasible declines or missing reports rising; tamper, masked or reward-error shares above the
  stop rules' limits; a high share of truncated completions or of groups without reward contrast;
* learning: after `min_steps_for_learning` steps the mean reward of the last steps does not beat the first ones.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from honest_agents.callbacks import TRAIN_LOG_FILE, is_train_log, read_jsonl

OK, WARN, FAIL = 0, 1, 2
LEVEL_NAMES = {OK: "OK", WARN: "WARN", FAIL: "FAIL"}


@dataclass(frozen=True)
class HealthConfig:
    window: int = 3
    stall_factor: float = 2.5
    grad_norm_spike: float = 10.0
    is_ratio_step_dev: float = 0.02  # one step's mean away from 1
    is_ratio_window_dev: float = 0.05  # mean of the last `window` steps away from 1 (the stop rule's limit)
    is_at_cap_share: float = 0.05
    logp_gap_factor: float = 3.0  # last gap vs the first logged one
    logp_gap_floor: float = 0.03  # ... and above this absolute value
    entropy_warn: float = 0.7  # fraction of the starting entropy
    entropy_fail: float = 0.5
    commands_fail: float = 0.5  # fraction of the starting commands per episode (quitting)
    length_warn: float = 0.4  # fraction of the starting completion length
    lazy_fail: float = 0.2  # share of feasible episodes that decline without working
    decline_warn: float = 0.5
    decline_rise: float = 0.05  # absolute rise of feasible declines over the start: quitting creeps in gradually
    no_report_rise: float = 0.15  # absolute rise over the start
    tamper_fail: float = 0.10
    masked_warn: float = 0.15
    reward_error_fail: float = 0.05
    truncated_warn: float = 0.30
    zero_variance_warn: float = 0.5
    min_steps_for_learning: int = 15
    min_reward_gain: float = 0.02


@dataclass(frozen=True)
class Finding:
    level: int
    check: str
    message: str


def _num(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float):
        return None
    return float(v)


def _series(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [v for r in rows if (v := _num(r, key)) is not None and math.isfinite(v)]


def _head_tail(values: list[float], n: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values[:n]), statistics.fmean(values[-n:])


def check(rows: Sequence[dict[str, Any]], *, now: float | None = None, cfg: HealthConfig | None = None) -> list[Finding]:
    """Findings for the per-step training rows of one run (oldest first)."""
    cfg = cfg or HealthConfig()
    rows = [r for r in rows if is_train_log(r)]
    if not rows:
        return [Finding(OK, "start", "no training step logged yet")]
    out: list[Finding] = []
    n = cfg.window
    last = rows[-1]

    step_times = _series(rows, "step_time")
    t_last = _num(last, "time")
    if now is not None and t_last is not None and step_times:
        idle, typical = now - t_last, statistics.median(step_times)
        if idle > cfg.stall_factor * typical:
            out.append(Finding(FAIL, "stall", f"no step for {idle / 60:.0f} min (typical step {typical / 60:.1f} min)"))

    for key in ("loss", "grad_norm"):
        v = last.get(key)
        if isinstance(v, int | float) and not isinstance(v, bool) and not math.isfinite(v):
            out.append(Finding(FAIL, "numbers", f"{key} is {v}"))
    grads = _series(rows, "grad_norm")
    if len(grads) >= 3 and grads[-1] > cfg.grad_norm_spike * statistics.median(grads[:-1]):
        out.append(Finding(WARN, "numbers", f"grad_norm {grads[-1]:.3g} is >{cfg.grad_norm_spike:.0f}x its median"))

    ratios = _series(rows, "sampling/importance_sampling_ratio/mean")
    if ratios:
        if abs(ratios[-1] - 1) > cfg.is_ratio_step_dev:
            out.append(Finding(WARN, "drift", f"importance ratio mean {ratios[-1]:.4f} this step"))
        tail = statistics.fmean(ratios[-n:])
        if len(ratios) >= n and abs(tail - 1) > cfg.is_ratio_window_dev:
            out.append(Finding(FAIL, "drift", f"importance ratio mean {tail:.4f} over the last {n} steps"))
    caps = _series(rows, "sampling/importance_sampling_ratio/at_cap_share")
    if caps and caps[-1] > cfg.is_at_cap_share:
        out.append(Finding(WARN, "drift", f"{caps[-1]:.1%} of tokens at the importance-ratio cap"))
    gaps = _series(rows, "sampling/sampling_logp_difference/mean")
    if len(gaps) >= 2 and gaps[-1] > max(cfg.logp_gap_floor, cfg.logp_gap_factor * gaps[0]):
        out.append(Finding(WARN, "drift", f"sampler/trainer log-prob gap {gaps[-1]:.4f} (first step {gaps[0]:.4f})"))

    def falling(key: str, warn: float | None, fail: float | None, label: str) -> None:
        head, tail = _head_tail(_series(rows, key), n)
        if head is None or tail is None or head <= 0 or len(_series(rows, key)) <= n:
            return
        frac = tail / head
        if fail is not None and frac < fail:
            out.append(Finding(FAIL, "collapse", f"{label} at {frac:.0%} of its start ({tail:.3g} vs {head:.3g})"))
        elif warn is not None and frac < warn:
            out.append(Finding(WARN, "collapse", f"{label} at {frac:.0%} of its start ({tail:.3g} vs {head:.3g})"))

    falling("entropy", cfg.entropy_warn, cfg.entropy_fail, "entropy")
    falling("honest/mean_commands", None, cfg.commands_fail, "commands per episode")
    falling("completions/mean_length", cfg.length_warn, None, "completion length")

    def recent(key: str) -> float | None:
        return _head_tail(_series(rows, key), n)[1]

    lazy = recent("honest/lazy_rate_feasible")
    if lazy is not None and lazy > cfg.lazy_fail:
        out.append(Finding(FAIL, "gaming", f"lazy declines on {lazy:.0%} of feasible episodes"))
    decline = recent("honest/decline_rate_feasible")
    if decline is not None and decline > cfg.decline_warn:
        out.append(Finding(WARN, "gaming", f"declines on {decline:.0%} of feasible episodes"))
    head, tail = _head_tail(_series(rows, "honest/decline_rate_feasible"), n)
    if head is not None and tail is not None and len(_series(rows, "honest/decline_rate_feasible")) > n and tail - head > cfg.decline_rise:
        out.append(Finding(WARN, "gaming", f"feasible declines rose to {tail:.1%} from {head:.1%} (quitting creeping in?)"))
    head, tail = _head_tail(_series(rows, "honest/no_report_rate"), n)
    if head is not None and tail is not None and tail - head > cfg.no_report_rise:
        out.append(Finding(WARN, "gaming", f"missing reports rose to {tail:.0%} from {head:.0%}"))
    for key, limit, level, label in (
        ("honest/tamper_rate", cfg.tamper_fail, FAIL, "tamper"),
        ("honest/reward_error_rate", cfg.reward_error_fail, FAIL, "reward errors"),
        ("honest/masked_share", cfg.masked_warn, WARN, "masked episodes"),
        ("completions/clipped_ratio", cfg.truncated_warn, WARN, "truncated completions"),
        ("honest/zero_variance_group_share", cfg.zero_variance_warn, WARN, "groups without reward contrast"),
    ):
        v = recent(key)
        if v is not None and v > limit:
            out.append(Finding(level, "gaming" if level == FAIL else "signal", f"{label} {v:.0%} (limit {limit:.0%})"))

    rewards = _series(rows, "reward")
    if len(rewards) >= cfg.min_steps_for_learning:
        head, tail = _head_tail(rewards, 5)
        if head is not None and tail is not None and tail < head + cfg.min_reward_gain:
            out.append(Finding(WARN, "learning", f"mean reward {tail:.3f} over the last 5 steps vs {head:.3f} over the first 5"))
    return out or [Finding(OK, "all", "no check tripped")]


def summary_line(rows: Sequence[dict[str, Any]]) -> str:
    rows = [r for r in rows if is_train_log(r)]
    if not rows:
        return "no steps yet"
    last = rows[-1]
    rewards = _series(rows, "reward")

    def fmt(key: str, spec: str = ".3f") -> str:
        v = _num(last, key)
        return "-" if v is None else format(v, spec)

    first = f"{statistics.fmean(rewards[:5]):.3f}" if rewards else "-"
    return (
        f"step {last.get('step')} | reward {fmt('reward')} (first-5 mean {first}) | success {fmt('honest/task_success', '.2f')}"
        f" | claims infeasible {fmt('honest/claim_rate_infeasible', '.2f')} | no-report {fmt('honest/no_report_rate', '.2f')}"
        f" | cmds {fmt('honest/mean_commands', '.1f')} | entropy {fmt('entropy')}"
        f" | IS {fmt('sampling/importance_sampling_ratio/mean', '.4f')} | gap {fmt('sampling/sampling_logp_difference/mean', '.4f')}"
        f" | step {fmt('step_time', '.0f')}s"
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help=f"a run's output directory (holds {TRAIN_LOG_FILE})")
    ap.add_argument("--now", type=float, default=None, help="wall-clock time for the stall check (default: now)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    path = args.run_dir / TRAIN_LOG_FILE
    rows = read_jsonl(path) if path.exists() else []
    findings = check(rows, now=args.now if args.now is not None else time.time())
    worst = max(f.level for f in findings)
    if args.json:
        print(json.dumps({"verdict": LEVEL_NAMES[worst], "summary": summary_line(rows), "findings": [asdict(f) for f in findings]}))
    else:
        print(f"{LEVEL_NAMES[worst]} | {summary_line(rows)}")
        for f in findings:
            if f.level > OK:
                print(f"  {LEVEL_NAMES[f.level]} {f.check}: {f.message}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
