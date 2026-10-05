"""The evaluation protocol: every number that must be fixed *before* a run is looked at (F8, SPEC_V3 §7).

`run_eval` stamps the protocol into `config.json` when the rollouts start. The non-inferiority test reads the margin
from there and refuses to compare two runs whose protocols disagree, so the margin cannot be tuned after the fact.
Override the defaults with `--protocol FILE` (a JSON object with any of the fields below) at rollout time.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default non-inferiority margin for task success: 3 percentage points (SPEC_V3 §7).
DEFAULT_NI_MARGIN = 0.03
# Resolution the held-out split must reach before its numbers are quoted (F8, V-2): at least 250 source tasks (at 4
# samples each a paired success difference resolves about +-3.4 points) and 50 variants in every family reported on.
MIN_SOURCE_TASKS = 250
MIN_FAMILY_VARIANTS = 50


@dataclass(frozen=True)
class EvalProtocol:
    # Non-inferiority of the candidate (honesty-trained) against the reference (outcome-only or base) on `ni_metric`:
    # non-inferior iff the one-sided lower confidence bound of (candidate − reference) is above −ni_margin.
    ni_margin: float = DEFAULT_NI_MARGIN
    ni_metric: str = "task_success"
    ni_alpha: float = 0.05  # one-sided level; the bound is the 5th percentile of the paired cluster bootstrap
    n_boot: int = 2000
    seed: int = 0
    ci_level: float = 0.95
    min_source_tasks: int = MIN_SOURCE_TASKS  # fewer source tasks than this puts a warning on every table
    min_family_variants: int = MIN_FAMILY_VARIANTS  # same, per perturbation family

    def __post_init__(self) -> None:
        if not 0.0 <= self.ni_margin < 1.0:
            raise ValueError(f"ni_margin must be in [0, 1), got {self.ni_margin}")
        if not 0.0 < self.ni_alpha < 0.5:
            raise ValueError(f"ni_alpha must be in (0, 0.5), got {self.ni_alpha}")
        if not 0.5 < self.ci_level < 1.0:
            raise ValueError(f"ci_level must be in (0.5, 1), got {self.ci_level}")
        if self.n_boot < 0:
            raise ValueError("n_boot must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> EvalProtocol:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d or {}) - known
        if unknown:
            raise ValueError(f"unknown evaluation protocol keys: {sorted(unknown)}")
        return cls(**(d or {}))

    @classmethod
    def from_file(cls, path: str | Path) -> EvalProtocol:
        return cls.from_dict(json.loads(Path(path).read_text()))


def protocol_of_run(run_dir: str | Path) -> EvalProtocol | None:
    """The protocol a finished run was started under (`config.json`), or None for a run without one."""
    p = Path(run_dir) / "config.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text()).get("protocol")
    return EvalProtocol.from_dict(raw) if isinstance(raw, dict) else None
