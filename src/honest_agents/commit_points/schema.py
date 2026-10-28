"""The commit-check build manifest: one schema for the data build, the coverage report and the flip-test probe.

A *commit-check example* is a decision state at one commitment point of a training-pool task. The names follow the
data build (`sft.commit_checks`, branch fix/commit-data): `point`, `outcome`, `committed`, `task`, `source`, `origin`.

* `claim_done`: the agent is about to claim the task done. The check runs the requirement checklist (written from the
  instruction only, never the hidden tests).
* `blame_env`: the agent is about to report that the environment stopped it. The check reproduces the fault.
* `destructive_step`: the agent is about to take a destructive step. The check looks for the hazards (`HAZARDS`).

**`pass` always means committing is right**: the checklist passes, the fault is reproduced, no hazard is found. So the
licensed decision is `committed` after `pass` and not after `fail`, and the flip-test sensitivity
P(commit | pass) - P(commit | fail) is positive for a policy that follows its check, at every point.

Two files:

* the **examples file** (JSONL, one `CommitExample` per line): the prefix messages up to the checks, and each check's
  command and raw output. The flip probe reads it.
* the **build manifest** (JSON, `BuildManifest`): the schema version, the quotas, the build's inputs and one light
  `ExampleRef` per example (no messages). The coverage report reads only this, in memory (L0).

`task` is the variant (task directory name): the per-task cap's unit. `source` is its source task (`pair_id`, e.g.
`endless-terminals/task_000000_ae4f1833`): the share cap's unit, the CI cluster and what is checked against
`tasks.select_split.excluded_from_training` (select, val and test sources never enter the build). `origin` says how the
example was made (`own_pass`, `own_fail`, `splice_fail`, ...: free text, counted).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

CLAIM, BLAME, DESTRUCTIVE = "claim_done", "blame_env", "destructive_step"
POINTS = (CLAIM, BLAME, DESTRUCTIVE)
PASS, FAIL = "pass", "fail"
OUTCOMES = (PASS, FAIL)
# The only destructive-step hazards: the target is the only copy, something depends on it, it is in use, it is
# protected, or it holds unfinished work (`other`: the build could not tell which).
HAZARDS = ("copy", "depended_on", "in_use", "protected", "unfinished")
HAZARD_OTHER = "other"
TRAIN_SPLIT = "train"

# The `eval.decision_probe` sample class that is the commitment at each point.
COMMIT_CLASS = {CLAIM: "report_complete", BLAME: "report_blocker", DESTRUCTIVE: "destroy"}

DEFAULT_MANIFEST = Path("data/commit/build_manifest.json")
DEFAULT_EXAMPLES = Path("data/commit/examples.jsonl")


class SchemaError(ValueError):
    """A manifest or example row that does not follow the schema."""


def licensed(outcome: str) -> bool:
    """Whether the check's outcome licenses committing."""
    return outcome == PASS


@dataclass(frozen=True)
class ExampleRef:
    """What the coverage report needs of one example."""

    id: str
    task: str
    source: str
    point: str
    outcome: str
    committed: bool  # the decision the example trains
    origin: str
    split: str | None = None  # the pool split of the task, where the build knows it
    hazard: str | None = None  # destructive_step: the hazard the check looks for
    checklist_agrees: bool | None = None  # claim_done: the checklist verdict agrees with the hidden tests

    def problems(self) -> list[str]:
        out = []
        if not self.id or not self.task or not self.source or not self.origin:
            out.append("empty id, task, source or origin")
        if self.point not in POINTS:
            out.append(f"point {self.point!r} not in {POINTS}")
        if self.outcome not in OUTCOMES:
            out.append(f"outcome {self.outcome!r} not in {OUTCOMES}")
        if not isinstance(self.committed, bool):
            out.append(f"committed {self.committed!r} is not a bool")
        if self.hazard is not None and self.hazard not in (*HAZARDS, HAZARD_OTHER):
            out.append(f"hazard {self.hazard!r} not in {(*HAZARDS, HAZARD_OTHER)}")
        if self.point == DESTRUCTIVE and self.hazard is None:
            out.append("a destructive_step example names its hazard")
        if self.checklist_agrees is not None and self.point != CLAIM:
            out.append("checklist_agrees is for claim_done examples only")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExampleRef:
        names = {f.name for f in fields(ExampleRef)}
        return ExampleRef(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class Check:
    """One check the agent runs at the point (a claim runs one per requirement)."""

    command: str
    output: str  # raw combined stdout+stderr, before the environment's rendering
    rc: int | None
    content: str = ""  # the assistant text before the call ("" = a stock line is chosen)


@dataclass(frozen=True)
class CommitExample(ExampleRef):
    """One decision state with its checks (a line of the examples file)."""

    prefix_id: str = ""  # the shared prefix: the pass and the fail example of a matched pair carry the same one
    messages: list[dict[str, Any]] = field(default_factory=list)  # [system, user, ...] up to the first check turn
    checks: list[Check] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)  # destructive_step: the paths the step would destroy
    episode_id: str | None = None  # the archived episode the prefix comes from

    def problems(self) -> list[str]:
        out = super().problems()
        if not self.prefix_id:
            out.append("empty prefix_id")
        if not self.checks or any(not c.command.strip() for c in self.checks):
            out.append("no check, or a check without a command")
        if not self.messages or self.messages[0].get("role") != "system":
            out.append("messages must start with the system message")
        if self.point == DESTRUCTIVE and not self.targets:
            out.append("a destructive_step example names its targets")
        return out

    def ref(self) -> ExampleRef:
        return ExampleRef.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CommitExample:
        names = {f.name for f in fields(cls)}
        row = {k: v for k, v in d.items() if k in names}
        row["checks"] = [c if isinstance(c, Check) else Check(**c) for c in row.get("checks") or []]
        return cls(**row)


@dataclass(frozen=True)
class Quota:
    """Balance and coverage quotas per commitment point x outcome x task. `cap_per_task` and `max_outcome_ratio` are
    the data build's (`sft.commit_checks.Quota`), the others the coverage report's."""

    cap_per_task: int = 2  # examples of one task in one point x outcome cell
    max_outcome_ratio: float = 1.0  # within a point: the larger outcome <= ratio x the smaller
    min_per_cell: int = 50  # examples per point x outcome cell
    max_source_share: float = 0.02  # share of the whole build one source task may hold: no task dominates
    max_task_share: float | None = None  # share of the whole build one task (variant) may hold; None = not checked
    points: tuple[str, ...] = POINTS  # the points the build trains: `min_per_cell` and the balance apply to these

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"points": list(self.points)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> Quota:
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise SchemaError(f"unknown quota keys {sorted(unknown)}")
        if "points" in d:
            if bad := set(d["points"]) - set(POINTS):
                raise SchemaError(f"unknown points {sorted(bad)}")
            d["points"] = tuple(d["points"])
        return cls(**d)


@dataclass
class BuildManifest:
    examples: list[ExampleRef]
    quota: Quota = field(default_factory=Quota)
    build: dict[str, Any] = field(default_factory=dict)  # inputs, commit, counts: free-form, recorded as given
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "quota": self.quota.to_dict(),
            "build": self.build,
            "examples": [e.to_dict() for e in self.examples],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BuildManifest:
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaError(f"manifest schema_version {version!r}, this code reads {SCHEMA_VERSION}")
        return cls(
            examples=[ExampleRef.from_dict(e) for e in d.get("examples") or []],
            quota=Quota.from_dict(d.get("quota")),
            build=dict(d.get("build") or {}),
            schema_version=version,
        )

    @classmethod
    def of(
        cls, examples: Iterable[ExampleRef], quota: Quota | None = None, build: Mapping[str, Any] | None = None
    ) -> BuildManifest:
        refs = [e.ref() if isinstance(e, CommitExample) else e for e in examples]
        return cls(refs, quota or Quota(), dict(build or {}))


def read_manifest(path: str | Path) -> BuildManifest:
    return BuildManifest.from_dict(json.loads(Path(path).expanduser().read_text()))


def write_manifest(manifest: BuildManifest, path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return p


def read_examples(path: str | Path) -> list[CommitExample]:
    lines = Path(path).expanduser().read_text().splitlines()
    return [CommitExample.from_dict(json.loads(line)) for line in lines if line.strip()]


def write_examples(examples: Sequence[CommitExample], path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in examples))
    return p
