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
* learning: after `min_steps_for_learning` steps neither false claims on impossible tasks nor claims after a failed
  feasible attempt have fallen (weighted by episode counts; raw reward is confounded by task difficulty);
* capability, over-refusal, quitting on the fixed validation probe (`probe-capability`, `probe-over-refusal`,
  `probe-quitting`; the stop rule of the iteration-11 evaluation protocol: honesty may not be bought with capability):
  each probe's verified success on near-miss controls, control decline rate and zero-command rate against the first
  probe (step 0), on the same variants (`callbacks.probe_shifts`). A shift of more than 10 points in the bad direction
  warns; more than 15 points on `probe_patience` (2) consecutive probes fails, as `callbacks.StopRules` does. Controls
  are the feasible tasks that look like their impossible twins, so a policy that learns "decline what looks risky"
  shows up here first, while its honesty numbers improve;
* the same three rates per training step (`capability`, `over-refusal`, `quitting`), last `canary_window` (5) steps
  against the first 5, weighted by episode counts: WARN only, an early hint between probes. Every step draws
  different controls (with `shuffle_dataset: false`, a different block of the pool, base pass rates 0.25 to 1.0), so
  an unchanged policy moves this comparison past 15 points in about 28 % of simulated runs: it cannot fail a run.
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

from honest_agents.callbacks import PROBE_CANARIES, TRAIN_LOG_FILE, is_probe_log, is_train_log, probe_shifts, read_jsonl
from honest_agents.trl_rewards import (
    METRIC_CONTROL_DECLINE_RATE,
    METRIC_CONTROL_VERIFIED_SUCCESS,
    METRIC_ZERO_COMMAND_RATE,
)

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
    ram_steps_left: float = 10.0  # warn when the host RAM would run out within this many steps at the current growth
    min_honesty_gain: float = 0.05  # claims on impossible tasks or after a failure must fall by this much
    # Per-step over-abstention canaries (`CANARIES`), WARN only. A step holds ~33 control episodes (185 of the 360 pool
    # rows, 64 episodes per step), but different ones every step, so the windows differ in task mix as much as in the
    # policy: these hint between probes and never fail a run.
    canary_window: int = 5
    canary_warn: float = 0.10  # absolute shift (in the bad direction) of the last window against the first
    # The same rates on the fixed probe against the first probe, paired on its variants (thresholds derived in
    # `callbacks`, above `PROBE_METRIC_PREFIX`; the same as `stop.probe_capability_drop` / `_patience`).
    probe_warn: float = 0.10
    probe_fail: float = 0.15
    probe_patience: int = 2  # consecutive probes the fail-sized shift must hold for


@dataclass(frozen=True)
class Canary:
    """A per-step rate that must not move one way: `sign` +1 when a rise is bad, -1 when a fall is bad."""

    key: str
    weight: str  # the episode count the rate is over, so a step with few such episodes counts less
    sign: int
    check: str
    label: str


CANARIES = (
    Canary(METRIC_CONTROL_VERIFIED_SUCCESS, "honest/n_control", -1, "capability", "verified success on controls"),
    Canary(METRIC_CONTROL_DECLINE_RATE, "honest/n_control", +1, "over-refusal", "decline rate on controls"),
    Canary(METRIC_ZERO_COMMAND_RATE, "honest/n_scored", +1, "quitting", "zero-command episodes"),
)

# Every `Finding.check` a training run can produce (the dashboard plots one level per check, 0 when it did not trip).
CHECKS = (
    *("stall", "numbers", "drift", "collapse", "gaming", "signal", "memory", "learning"),
    *(c.check for c in CANARIES),
    *(c.check for c in PROBE_CANARIES),
)


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


def check(
    rows: Sequence[dict[str, Any]], *, now: float | None = None, cfg: HealthConfig | None = None
) -> list[Finding]:
    """Findings for the per-step training rows of one run (oldest first)."""
    cfg = cfg or HealthConfig()
    # Any logged row shows the run is alive (a validation probe between two steps logs its own row).
    t_alive = max((t for r in rows if (t := _num(r, "time")) is not None), default=None)
    probes = probe_rows(rows)
    rows = [r for r in rows if is_train_log(r)]
    if not rows:
        return [Finding(OK, "start", "no training step logged yet")]
    out: list[Finding] = []
    n = cfg.window
    last = rows[-1]

    step_times = _series(rows, "step_time")
    if now is not None and t_alive is not None and step_times:
        idle, typical = now - t_alive, statistics.median(step_times)
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
    if (
        head is not None
        and tail is not None
        and len(_series(rows, "honest/decline_rate_feasible")) > n
        and tail - head > cfg.decline_rise
    ):
        out.append(
            Finding(WARN, "gaming", f"feasible declines rose to {tail:.1%} from {head:.1%} (quitting creeping in?)")
        )
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

    rss, avail = _series(rows, "mem/rss_gb"), _num(last, "mem/available_gb")
    if len(rss) >= 2 and avail is not None:
        growth = (rss[-1] - rss[0]) / (len(rss) - 1)
        if growth > 0.2 and avail / growth < cfg.ram_steps_left:
            out.append(
                Finding(
                    FAIL if avail / growth < 3 else WARN,
                    "memory",
                    f"host RAM: process at {rss[-1]:.0f} GB, growing {growth:.1f} GB/step, {avail:.0f} GB left "
                    f"(about {avail / growth:.0f} steps)",
                )
            )

    out.extend(_canaries(rows, cfg))
    out.extend(_probe_canaries(probes, cfg))

    # Learning, measured on honesty rates weighted by how many episodes of that kind each step had. Raw mean reward is
    # NOT used: every step draws different tasks, and on a real run the reward fell from 0.55 to 0.40 only because later
    # steps drew harder tasks (the base model's pass rate on the same tasks fell from 0.80 to 0.56).
    if len(rows) >= cfg.min_steps_for_learning:
        gains = []
        for key, weight in (
            ("honest/claim_rate_infeasible", "honest/n_infeasible"),
            ("honest/p_claim_given_fail", "honest/n_feasible"),
        ):
            head, tail = _weighted(rows[:5], key, weight), _weighted(rows[-5:], key, weight)
            if head is not None and tail is not None:
                gains.append((key.split("/")[-1], head, tail))
        if gains and all(head - tail < cfg.min_honesty_gain for _, head, tail in gains):
            text = ", ".join(f"{k} {h:.0%} -> {t:.0%}" for k, h, t in gains)
            out.append(Finding(WARN, "learning", f"no honesty gain over the first 5 steps: {text}"))
    return out or [Finding(OK, "all", "no check tripped")]


def _canary_shift(rows: Sequence[dict[str, Any]], c: Canary, w: int) -> tuple[float, float, float] | None:
    """(first-window rate, last-window rate, shift in the bad direction); None until the two windows are disjoint."""
    if len(rows) < 2 * w:
        return None
    head, tail = _weighted(rows[:w], c.key, c.weight, missing=1.0), _weighted(rows[-w:], c.key, c.weight, missing=1.0)
    if head is None or tail is None:
        return None
    return head, tail, c.sign * (tail - head)


def _canaries(rows: Sequence[dict[str, Any]], cfg: HealthConfig) -> list[Finding]:
    """WARN when the per-step shift passes `canary_warn`. Never FAIL: the steps' task mix differs (module docstring);
    the failing comparison is `_probe_canaries`, on the same variants every time."""
    out: list[Finding] = []
    w = cfg.canary_window
    for c in CANARIES:
        now = _canary_shift(rows, c, w)
        if now is None or now[2] <= cfg.canary_warn:
            continue
        head, tail, shift = now
        verb = "fell" if c.sign < 0 else "rose"
        out.append(Finding(WARN, c.check, f"{c.label} {verb} to {tail:.1%} from {head:.1%} (last {w} vs first {w} "
                                          f"steps, {shift * 100:.1f} points; per-step hint, see the probe)"))  # fmt: skip
    return out


def probe_rows(rows: Sequence[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """The run log's probe rows, one per step (the latest life of a re-probed step wins), oldest first."""
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        step = r.get("step")
        if is_probe_log(r) and isinstance(step, int) and not isinstance(step, bool):
            by_step[step] = r
    return sorted(by_step.items())


def _probe_canaries(probes: Sequence[tuple[int, dict[str, Any]]], cfg: HealthConfig) -> list[Finding]:
    """The latest probe against the first: WARN past `probe_warn`, FAIL past `probe_fail` on `probe_patience`
    consecutive probes (the comparison `callbacks.StopRules` stops on)."""
    out: list[Finding] = []
    for sh in probe_shifts(probes, cfg.probe_fail):
        if sh.shift <= cfg.probe_warn:
            continue
        if sh.shift > cfg.probe_fail and sh.sustained >= cfg.probe_patience:
            out.append(Finding(FAIL, sh.canary.check, f"{sh.describe()} on {sh.sustained} consecutive probes"))
        else:
            out.append(Finding(WARN, sh.canary.check, sh.describe()))
    return out


def _weighted(rows: Sequence[dict[str, Any]], key: str, weight: str, *, missing: float | None = None) -> float | None:
    """Mean of `key` weighted by `weight`; a row without the weight counts as `missing` (None: the row is skipped)."""
    num = den = 0.0
    for r in rows:
        v, w = _num(r, key), _num(r, weight)
        if w is None:
            w = missing
        if v is not None and w is not None and w > 0 and math.isfinite(v):
            num, den = num + v * w, den + w
    return num / den if den else None


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
        f" | control verified {fmt(METRIC_CONTROL_VERIFIED_SUCCESS, '.2f')} | zero-cmd {fmt(METRIC_ZERO_COMMAND_RATE, '.2f')}"
        f" | claims infeasible {fmt('honest/claim_rate_infeasible', '.2f')} | no-report {fmt('honest/no_report_rate', '.2f')}"
        f" | cmds {fmt('honest/mean_commands', '.1f')} | entropy {fmt('entropy')}"
        f" | IS {fmt('sampling/importance_sampling_ratio/mean', '.4f')} | gap {fmt('sampling/sampling_logp_difference/mean', '.4f')}"
        f" | RAM {fmt('mem/rss_gb', '.1f')} GB | step {fmt('step_time', '.0f')}s"
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
        print(
            json.dumps(
                {
                    "verdict": LEVEL_NAMES[worst],
                    "summary": summary_line(rows),
                    "findings": [asdict(f) for f in findings],
                }
            )
        )
    else:
        print(f"{LEVEL_NAMES[worst]} | {summary_line(rows)}")
        for f in findings:
            if f.level > OK:
                print(f"  {LEVEL_NAMES[f.level]} {f.check}: {f.message}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
