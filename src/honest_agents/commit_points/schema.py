"""The commit-check build manifest: one schema for the data build, the coverage report and the flip-test probe.

A *commit-check example* is a decision state at one commitment point of a training-pool task:

* `claim`: the agent is about to claim the task done. The check runs the requirement checklist (written from the
  instruction only, never the hidden tests).
* `blame`: the agent is about to report that the environment stopped it. The check reproduces the fault.
* `destructive`: the agent is about to take a destructive step. The check looks for the hazards (`HAZARDS`).

`outcome` is what the check printed, and **`pass` always means the check licenses the act**: the checklist passes, the
fault is reproduced, no hazard is found. So the expected decision is `act` after `pass` and `hold` after `fail`, and the
flip-test sensitivity P(act | pass) - P(act | fail) is positive for a policy that follows its check, at every point.

Two files:

* the **examples file** (JSONL, one `CommitExample` per line): the prefix messages up to the check, the check command
  and its raw output. The flip probe reads it.
* the **build manifest** (JSON, `BuildManifest`): the schema version, the quotas, the build's inputs and one light
  `ExampleRef` per example (no messages). The coverage report reads only this, in memory (L0).

`task_id` is the source task (`pair_id` of the pool rows, e.g. `endless-terminals/task_000000_ae4f1833`): the quota
unit and the CI cluster. `split` must be `train` (`TRAIN_SPLIT`); val/test sources never enter the build
(`coverage` asserts it against the eval pools).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

CLAIM, BLAME, DESTRUCTIVE = "claim", "blame", "destructive"
COMMIT_POINTS = (CLAIM, BLAME, DESTRUCTIVE)
PASS, FAIL = "pass", "fail"
OUTCOMES = (PASS, FAIL)
ACT, HOLD = "act", "hold"  # the post-check decision: take the committed step, or not (keep working / decline)
DECISIONS = (ACT, HOLD)
SOURCE_OWN_RUN = "own_run"  # the model's own episode on a training-pool task, cut at the commitment point
SOURCE_SPLICE = "splice"  # a G1-style splice of the check into a prefix at the claim step
SOURCES = (SOURCE_OWN_RUN, SOURCE_SPLICE)
# The only destructive-step hazards: the target is the only copy, something depends on it, it is in use, it is
# protected, or it holds unfinished work.
HAZARDS = ("copy", "depended_on", "in_use", "protected", "unfinished")
TRAIN_SPLIT = "train"
EVAL_SPLITS = ("val", "test")

# The `eval.decision_probe` sample class that is the act at each point.
ACT_CLASS = {CLAIM: "report_complete", BLAME: "report_blocker", DESTRUCTIVE: "destroy"}

DEFAULT_MANIFEST = Path("data/commit/build_manifest.json")
DEFAULT_EXAMPLES = Path("data/commit/examples.jsonl")


class SchemaError(ValueError):
    """A manifest or example row that does not follow the schema."""


def expected_decision(outcome: str) -> str:
    return ACT if outcome == PASS else HOLD


@dataclass(frozen=True)
class ExampleRef:
    """What the coverage report needs of one example."""

    id: str
    task_id: str
    commit_point: str
    outcome: str
    decision: str
    source: str
    split: str
    hazard: str | None = None  # destructive: the hazard the check looks for
    checklist_agrees: bool | None = None  # claim: the checklist verdict agrees with the hidden tests (None = not run)

    def problems(self) -> list[str]:
        out = []
        if not self.id or not self.task_id:
            out.append("empty id or task_id")
        for name, value, allowed in (
            ("commit_point", self.commit_point, COMMIT_POINTS),
            ("outcome", self.outcome, OUTCOMES),
            ("decision", self.decision, DECISIONS),
            ("source", self.source, SOURCES),
        ):
            if value not in allowed:
                out.append(f"{name} {value!r} not in {allowed}")
        if self.hazard is not None and self.hazard not in HAZARDS:
            out.append(f"hazard {self.hazard!r} not in {HAZARDS}")
        if self.commit_point == DESTRUCTIVE and self.hazard is None:
            out.append("a destructive example names its hazard")
        if self.checklist_agrees is not None and self.commit_point != CLAIM:
            out.append("checklist_agrees is for claim examples only")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExampleRef:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class CommitExample(ExampleRef):
    """One decision state with its check (a line of the examples file)."""

    prefix_id: str = ""  # the shared prefix: the pass and the fail example of a matched pair carry the same one
    messages: list[dict[str, Any]] = field(default_factory=list)  # [system, user, ...] up to the check turn
    check_command: str = ""
    check_output: str = ""  # raw combined stdout+stderr, before the environment's rendering
    check_rc: int | None = None
    check_content: str = ""  # the assistant text before the check call ("" = a stock line is chosen)
    targets: list[str] = field(default_factory=list)  # destructive: the paths the act would destroy
    episode_id: str | None = None  # own_run: the archived episode the prefix comes from

    def problems(self) -> list[str]:
        out = super().problems()
        if not self.prefix_id:
            out.append("empty prefix_id")
        if not self.check_command.strip():
            out.append("empty check_command")
        if not self.messages or self.messages[0].get("role") != "system":
            out.append("messages must start with the system message")
        if self.commit_point == DESTRUCTIVE and not self.targets:
            out.append("a destructive example names its targets")
        return out

    def ref(self) -> ExampleRef:
        return ExampleRef.from_dict(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CommitExample:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class Quota:
    """Balance and coverage quotas per commitment point x outcome x task."""

    min_per_cell: int = 50  # examples per (point, outcome) cell
    max_per_task: int = 4  # examples of one task in one (point, outcome) cell: the per-task cap
    max_task_share: float = 0.02  # share of the whole build one task may hold
    fail_share: tuple[float, float] = (0.35, 0.65)  # per point: the check changes the decision about half the time

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "fail_share": list(self.fail_share)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> Quota:
        d = dict(d or {})
        if "fail_share" in d:
            d["fail_share"] = tuple(d["fail_share"])
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise SchemaError(f"unknown quota keys {sorted(unknown)}")
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
