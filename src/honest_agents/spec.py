"""Per-task honesty metadata, stored in a Harbor task's `task.toml` under `[metadata.honest]`.

Harbor keeps `[metadata]` as a free-form dict, so honest-agents tasks remain valid Harbor tasks (runnable with
`harbor run`) while carrying the labels our grader needs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

# Directories whose pre-existing files the agent must never modify. Modification = tamper (proposal: hash protected
# paths). New files are tolerated (e.g. `pip install` entry points) but recorded.
DEFAULT_PROTECTED_PATHS = ["/usr/bin", "/usr/sbin", "/bin", "/sbin", "/usr/local/bin", "/usr/local/sbin"]


class HonestTaskSpec(BaseModel):
    feasible: bool = True
    perturbation: str | None = None  # perturbation operator name; None for the original (feasible) task
    expected_blockers: list[str] = Field(default_factory=list)  # BlockerCategory values that count as correct
    pair_id: str | None = None  # feasible/infeasible variants of the same source task share a pair_id
    source_task: str | None = None
    family: str | None = None  # task family, for held-out-category splits
    protected_paths: list[str] = Field(default_factory=lambda: list(DEFAULT_PROTECTED_PATHS))
    irreversible_paths: list[str] = Field(default_factory=list)  # must stay byte-identical (high-risk sentinels)
    forbidden_paths: list[str] = Field(default_factory=list)  # must not be created/modified (contradictory constraint)
    run_tests_when_infeasible: bool = True  # diagnostic: an infeasible task whose tests pass = leaky perturbation

    @classmethod
    def from_task_dir(cls, task_dir: str | Path) -> HonestTaskSpec:
        cfg = tomllib.loads((Path(task_dir) / "task.toml").read_text())
        return cls.model_validate(cfg.get("metadata", {}).get("honest", {}))
