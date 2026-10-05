"""Per-episode evaluation records: the single on-disk format every eval source is converted to.

Our own rollouts (`honest_agents.eval.rollout`) and Harbor jobs (`honest_agents.eval.harbor_results`) both produce
`EpisodeRecord`s, so one metrics implementation covers held-out variants, TB Lite / TB 2.0, Agentic Abstention, and our
own tasks run under a third-party Harbor harness with the report-file protocol (SPEC_V3 §9.4).

A record stores the full `EpisodeOutcome` (not just the reward) so metrics can re-score it under any `RewardConfig`,
and so the constant-policy reference rows and the confidence-threshold sweep can rewrite the report and score the same
trajectory again (`honest_agents.eval.baselines`).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.reward import (
    MASK_BOOT_FAILURE,
    MASK_REWARD_ERROR,
    MASK_TOOLCHAIN,
    EpisodeOutcome,
    RewardBreakdown,
    RewardConfig,
    compute_reward,
    outcome_from_dict,
    outcome_to_dict,
)
from honest_agents.spec import HonestTaskSpec

__all__ = [
    "DEFAULT_REWARD_CONFIG",
    "HARNESS_HONEST",
    "INFRA_MASKS",
    "EpisodeRecord",
    "iter_records",
    "outcome_from_dict",
    "outcome_to_dict",
    "read_records",
    "record_line",
    "write_records",
]

DEFAULT_REWARD_CONFIG = RewardConfig()  # frozen; the training default
HARNESS_HONEST = "honest"  # our own tool surface (bash / submit_report / abstain)
# Masks that mean "the sandbox or the grader failed, not the policy". A leak mask is not one of them: it says the
# variant's label is wrong, which the metrics report (leak rate) instead of hiding.
INFRA_MASKS = frozenset({MASK_BOOT_FAILURE, MASK_TOOLCHAIN, MASK_REWARD_ERROR})


@dataclass
class EpisodeRecord:
    episode_id: str
    task_id: str
    outcome: EpisodeOutcome
    task_dir: str | None = None
    sample: int = 0
    model: str | None = None
    source: str = "honest"  # "honest" (our rollout loop) | "harbor" (parsed Harbor job)
    # Agent harness that produced the episode: "honest" (our tools), or a Harbor agent name such as "terminus-2".
    # Results are reported per harness (SPEC_V3 §9.4): behaviour learned in one harness transfers poorly to others.
    harness: str = HARNESS_HONEST
    report_channel: str = "tool"  # how the report reached us: "tool" | "file" (harness.REPORT_FILE) | "none"
    # task labels (from [metadata.honest]); copied so metrics never need the task tree
    feasible: bool = True
    is_control: bool = False  # near-miss control: feasible, carries the surface cue of an infeasible twin
    control_of: str | None = None  # operator of the infeasible twin the control mirrors
    perturbation: str | None = None
    family: str | None = None
    pair_id: str | None = None
    source_task: str | None = None
    variant_id: str | None = None
    expected_blockers: list[str] = field(default_factory=list)
    constrained: bool = False  # task declares irreversible / forbidden paths (post-hoc abstention is possible)
    # rollout stats: stop_reason, n_model_calls, report_turn, token counts, wall time, error, ...
    rollout: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """`"feasible"`, `"control"` or `"infeasible"` (the mixture stratum of SPEC_V3 §4.5)."""
        if not self.feasible:
            return "infeasible"
        return "control" if self.is_control else "feasible"

    @property
    def infra_error(self) -> str | None:
        """Failures that are not the policy's fault: the model endpoint failed, or the sandbox / grader failed before
        the agent ran a command. Excluded from every metric.

        A grading failure *after* the agent acted is not infra: SPEC_V3 §3.5 attributes it to the policy and
        `compute_reward` scores it as tamper, so it stays in the metrics.
        """
        if self.rollout.get("error"):
            return str(self.rollout["error"])
        mask = self.outcome.mask
        if mask in INFRA_MASKS:
            return self.outcome.grading_error or self.extra.get("infra_error") or mask
        return None

    @property
    def cluster(self) -> str:
        """Bootstrap resampling unit: the source task. Variants of one source (the plain task, its controls and its
        infeasible twins) and repeated samples are correlated, so they are resampled together."""
        return self.source_task or self.pair_id or self.task_id

    @property
    def variant_key(self) -> str:
        """Prompt-group id: every sample of one task variant. A leak in one sample voids the label for all of them."""
        return self.variant_id or self.task_id

    def score(self, cfg: RewardConfig = DEFAULT_REWARD_CONFIG) -> RewardBreakdown:
        return compute_reward(self.outcome, cfg)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "outcome"}
        d["outcome"] = outcome_to_dict(self.outcome)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeRecord:
        """Inverse of `to_dict`. Unknown keys are dropped, so files written by another version still load."""
        d = dict(d)
        outcome = outcome_from_dict(d.pop("outcome"))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(
            outcome=outcome,
            **{k: v for k, v in d.items() if k in known and k != "outcome"},
        )

    @classmethod
    def labels_from_spec(cls, spec: HonestTaskSpec) -> dict[str, Any]:
        return {
            "feasible": spec.feasible,
            "is_control": spec.is_control,
            "control_of": spec.control_of,
            "perturbation": spec.perturbation,
            "family": spec.family,
            "pair_id": spec.pair_id,
            "source_task": spec.source_task,
            "variant_id": spec.variant_id,
            "expected_blockers": list(spec.expected_blockers),
            "constrained": bool(spec.irreversible_paths or spec.forbidden_paths),
        }


def write_records(
    path: str | Path,
    records: Iterable[EpisodeRecord],
    *,
    cfg: RewardConfig | None = None,
) -> None:
    with open(path, "w") as f:
        f.writelines(record_line(r, cfg) for r in records)


def record_line(r: EpisodeRecord, cfg: RewardConfig | None = None) -> str:
    """One JSONL line. The reward under `cfg` is included for human inspection only (it is recomputed on load)."""
    d = r.to_dict()
    if cfg is not None:
        d["reward"] = r.score(cfg).to_dict()
    return json.dumps(d, default=str) + "\n"


def iter_records(path: str | Path) -> Iterator[EpisodeRecord]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield EpisodeRecord.from_dict(json.loads(line))


def read_records(path: str | Path) -> list[EpisodeRecord]:
    return list(iter_records(path))
