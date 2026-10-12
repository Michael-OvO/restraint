"""The W&B dashboard of a GRPO run: what TRL does not already send there.

With `grpo.report_to: wandb`, transformers' `WandbCallback` logs every trainer log: TRL's own numbers and the
`honest/*` behaviour metrics (as `train/...`), and every validation probe (as `eval/...`). This module adds only what
lives outside those logs, so no number reaches the dashboard twice:

* the run log's own columns (`callbacks.JsonlLogger`): host RAM `mem/*` (a run was OOM-killed at 69 GB),
  `gpu/peak_memory_gb` and `wall_sec_since_last_log`;
* `train.health` after every step and every validation probe: the worst level as `health/verdict` and one level per
  check as `health/<check>` (0 OK, 1 WARN, 2 FAIL), the findings' text in the run summary, and one W&B alert the first
  time a check FAILs, so an unattended run that is quietly getting worse reaches the phone, not only the log (the
  run itself is stopped by `callbacks.StopRules`, which judges the probe's capability the same way);
* the validation probe's checkpoint choice (`probe/best_step`, `probe/best_value`), which only lives in
  `best_checkpoint.json` otherwise.

The run is started here, not by `WandbCallback`, so the project, the name (the iteration slug, `wandb.name`) and the
id are ours; the callback finds the open run and logs into it. The id is kept in `<output_dir>/wandb_run_id`, so
`--resume` continues the same dashboard run instead of starting a second one.

Credentials are wandb's business: it reads `WANDB_API_KEY` or `~/.netrc`, and `WANDB_ENTITY`. Nothing here reads, logs
or stores a key. `WANDB_MODE=disabled` turns everything off (tests, offline work) without importing wandb.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from honest_agents.callbacks import JsonlLogger, ValidationProbe, is_probe_log, is_train_log, read_jsonl
from honest_agents.train import health
from honest_agents.train.config import TrainRunConfig
from honest_agents.train.derive import run_name

logger = logging.getLogger("honest_agents.train")

WANDB = "wandb"
RUN_ID_FILE = "wandb_run_id"
STEP_KEY = "train/global_step"  # the x-axis transformers' WandbCallback defines for every metric
_RUN_LOG_ONLY = ("step", "time")  # JsonlLogger bookkeeping, not metrics


# ── on or off ──────────────────────────────────────────────────────────────


def _report_list(report_to: str | Sequence[str] | None) -> list[str]:
    if report_to is None:
        return []
    items = [report_to] if isinstance(report_to, str) else list(report_to)
    return [r for r in items if r != "none"]


def wandb_disabled(env: Mapping[str, str] | None = None) -> bool:
    """`WANDB_MODE=disabled`: the off switch for tests and offline work."""
    env = os.environ if env is None else env
    return env.get("WANDB_MODE", "").strip().lower() == "disabled"


def resolve_report_to(report_to: Any, env: Mapping[str, str] | None = None) -> Any:
    """`grpo.report_to` with wandb removed when it is disabled, so a W&B config dry-runs and trains offline (and on a
    machine without wandb installed: transformers imports it as soon as `report_to` names it)."""
    items = _report_list(report_to)
    if WANDB not in items or not wandb_disabled(env):
        return report_to
    rest = [r for r in items if r != WANDB]
    return rest or "none"


def wandb_enabled(report_to: Any, env: Mapping[str, str] | None = None) -> bool:
    return WANDB in _report_list(report_to) and not wandb_disabled(env)


# ── the run ────────────────────────────────────────────────────────────────


def _import_wandb() -> Any:
    try:
        import wandb
    except ImportError as e:  # the train extra pins it; a node set up before it was added lacks it
        raise RuntimeError(
            "grpo.report_to names wandb but wandb is not installed: `uv sync --extra train` on the node, or set "
            "WANDB_MODE=disabled to train without the dashboard"
        ) from e
    return wandb


def run_id(output_dir: Path, *, resuming: bool) -> str:
    """The dashboard run id of this output directory: kept across `--resume`, new for a fresh start (a fresh start
    into an old directory is a new run and must not be drawn on top of the old one's charts)."""
    path = output_dir / RUN_ID_FILE
    if resuming and path.is_file() and (kept := path.read_text().strip()):
        return kept
    new = secrets.token_hex(8)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(new + "\n")
    return new


def init_run(cfg: TrainRunConfig, output_dir: Path, *, resuming: bool, wandb_module: Any = None) -> Any:
    """Start (or, on resume, continue) the dashboard run. Returns the wandb run."""
    wandb = wandb_module if wandb_module is not None else _import_wandb()
    w = cfg.wandb
    return wandb.init(
        project=w.project,
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=w.name or run_name(cfg),
        group=w.group,
        tags=list(w.tags) or None,
        id=run_id(output_dir, resuming=resuming),
        resume="allow",
        dir=str(output_dir),
        job_type="train",
        # The run config as loaded (it holds no credential: see `WandbSection`). transformers adds the GRPOConfig.
        config={"honest_run": dataclasses.asdict(cfg)},
    )


# ── per-step additions ─────────────────────────────────────────────────────


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def run_log_extras(row: Mapping[str, Any], logs: Mapping[str, Any]) -> dict[str, float]:
    """The numbers the run log added to a trainer log (`mem/*`, `gpu/*`, wall time): the rest is already on W&B."""
    return {k: float(v) for k, v in row.items() if k not in logs and k not in _RUN_LOG_ONLY and _finite(v)}


def health_metrics(findings: Sequence[health.Finding]) -> dict[str, float]:
    """One level per check (0 when it did not trip, so every panel exists from the first step) and the worst level."""
    levels = dict.fromkeys(health.CHECKS, health.OK)
    for f in findings:
        if f.check in levels:
            levels[f.check] = max(levels[f.check], f.level)
    out = {f"health/{check}": float(level) for check, level in levels.items()}
    out["health/verdict"] = float(max((f.level for f in findings), default=health.OK))
    return out


def findings_text(findings: Sequence[health.Finding]) -> str:
    tripped = [f"{health.LEVEL_NAMES[f.level]} {f.check}: {f.message}" for f in findings if f.level > health.OK]
    return "; ".join(tripped) or "OK"


def history(path: Path, resumed_at: int) -> list[dict[str, Any]]:
    """The per-step rows and probe rows a run resumed at step `resumed_at` already has, one of each per step, oldest
    first (a step's probe after its training row).

    The run log is append-only, so after a crash it also holds the steps past the last checkpoint, which the resumed run
    trains again and logs a second time; keeping them would count those steps twice in the health windows. The step-0
    probe is kept: it is the baseline of the probe checks. A fresh start (step 0) into a directory with an old run log
    starts an empty history: the canaries' first window and the probes' baseline must be this run's.
    """
    if resumed_at <= 0 or not path.is_file():
        return []
    by_step: dict[tuple[int, int], dict[str, Any]] = {}
    for r in read_jsonl(path):
        step = r.get("step")
        if not isinstance(step, int) or step > resumed_at:
            continue
        if is_train_log(r) and step > 0:
            by_step[(step, 0)] = r  # the latest life of a step wins
        elif is_probe_log(r) and step >= 0:
            by_step[(step, 1)] = r
    return [by_step[k] for k in sorted(by_step)]


class WandbMirror(TrainerCallback):
    """Sends the run log's extra columns, `train.health` and the probe's checkpoint choice to the open W&B run.

    Must be added after the `JsonlLogger` it reads (its row for the step must exist) and after the `ValidationProbe`
    (its best checkpoint must be updated). A dashboard failure is logged and never stops the run.
    """

    def __init__(
        self,
        run_log: JsonlLogger,
        probe: ValidationProbe | None = None,
        *,
        health_checks: bool = True,
        health_config: health.HealthConfig | None = None,
        wandb_module: Any = None,
    ):
        self.run_log = run_log
        self.probe = probe
        self.health_checks = health_checks
        self.health_config = health_config
        self._wandb = wandb_module
        self.rows: list[dict[str, Any]] = []  # per-step training rows, oldest first: what `train.health` reads
        self.alerted: set[str] = set()  # checks already alerted on (one alert per check per process)

    def _run(self) -> Any:
        if self._wandb is None:
            self._wandb = _import_wandb()
        return self._wandb.run

    def on_train_begin(self, args, state, control, **kwargs):
        self.rows = history(self.run_log.path, int(getattr(state, "global_step", 0) or 0))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not getattr(state, "is_world_process_zero", True):
            return
        train, probe = is_train_log(logs), is_probe_log(logs)
        if not (train or probe):
            return
        try:
            self._log_step(int(state.global_step), logs, probe=probe and not train)
        except Exception:  # the dashboard must never stop a run
            logger.warning("W&B: could not log step %s", state.global_step, exc_info=True)

    def _log_step(self, step: int, logs: Mapping[str, Any], *, probe: bool = False) -> None:
        """A training step's extras and health levels; for a probe log (its numbers are TRL's `eval/...` already) only
        the health levels, so a capability drop on the probe shows at once, not one step later."""
        last = self.run_log.last_row
        row = dict(last) if last is not None and last.get("step") == step else {"step": step, **logs}
        self.rows.append(row)
        run = self._run()
        if run is None:
            return
        payload = {} if probe else run_log_extras(row, logs)
        alerts: list[health.Finding] = []
        if self.health_checks:
            findings = health.check(self.rows, cfg=self.health_config)  # no `now`: a stall is seen from outside
            payload.update(health_metrics(findings))
            run.summary["health/findings"] = findings_text(findings)
            alerts = [f for f in findings if f.level == health.FAIL and f.check not in self.alerted]
        if payload:
            run.log({**payload, STEP_KEY: step})  # first: an alert that fails must not lose the step's numbers
        for f in alerts:
            self.alerted.add(f.check)
            try:
                run.alert(title=f"health FAIL: {f.check} (step {step})", text=f.message)
            except Exception:
                logger.warning("W&B: could not send the alert for %s at step %d", f.check, step, exc_info=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.probe is None or self.probe.best is None or not getattr(state, "is_world_process_zero", True):
            return
        try:
            run = self._run()
            if run is None:
                return
            best = self.probe.best
            run.log({"probe/best_step": float(best["step"]), "probe/best_value": float(best["value"]),
                     STEP_KEY: int(state.global_step)})  # fmt: skip
            run.summary["probe/best_checkpoint"] = best.get("checkpoint") or "step 0 (untrained)"
            run.summary["probe/selection_metric"] = best["metric"]
        except Exception:
            logger.warning("W&B: could not log the validation probe", exc_info=True)
