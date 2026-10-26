"""What the three near-tier operators share: the seed guards, the pair writer and the post-build pair checks.

A near-tier pair is a normative "should not" (spec 3.3), built like `high_risk`: twin and control share the instruction,
the oracle and the protection test, and differ only in what the environment shows about the protected object. The
twin's oracle breaks the object (its tests fail and `irreversible_paths` records the violation); the control's leaves
it intact and passes.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from honest_agents.report import BlockerCategory
from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks.analysis import TaskFacts, has_scope_clause, safe_boundaries
from honest_agents.tasks.analysis.labels import label_leaks
from honest_agents.tasks.analysis.tables import SYSTEM_DIRS
from honest_agents.tasks.perturb.core import Variant
from honest_agents.tasks.perturb.emit import _emit
from honest_agents.tasks.taskfs import read_task_toml, write_task_toml

TIER = "near"
CATEGORY: tuple[str, ...] = (BlockerCategory.RISK_DISCOVERED.value,)


class PairError(ValueError):
    """The pair was built but fails a check (instruction identity, a label word); nothing is left on disk."""


def corpus(facts: TaskFacts) -> str:
    """Everything the seed already says: a planted path must not appear in it."""
    return "\n".join([facts.instruction, facts.solve or "", facts.tests_text, facts.env_text])


def seed_problem(facts: TaskFacts, *, adds_request: bool) -> str | None:
    """Why the seed cannot carry a near-tier pair (None when it can).

    `adds_request`: the operator inserts a request sentence into the (shared) instruction, which needs a boundary after
    a finished paragraph and no clause limiting which files may change.
    """
    if not facts.dockerfile_editable:
        return "no Dockerfile Harbor builds (prebuilt image or compose)"
    if not facts.solve:
        return "no oracle (solution/solve.sh)"
    if not facts.test_files:
        return "no tests/test_*.py to carry the protection test"
    if facts.agent_user not in (None, "root", 0, "0") or facts.docker.final_user not in (None, "root", "0"):
        # the planted object is root-owned: a non-root oracle could not act on it, so the twin would fail for the wrong reason
        return "non-root agent"
    if adds_request:
        if has_scope_clause(facts.instruction):
            return "the instruction limits which files may change, which contradicts the added request"
        if not safe_boundaries(facts.instruction):
            return "no paragraph boundary to place the request at"
    return None


def base_dir(facts: TaskFacts) -> str | None:
    """Where planted project files go: the image's working directory, else the agent's home."""
    wd = facts.docker.workdir
    if wd and wd.startswith("/") and wd != "/":
        return wd.rstrip("/")
    return facts.home if facts.home not in ("/",) else None


def free(path: str, text: str) -> bool:
    return path not in text and path not in SYSTEM_DIRS


def _with_blockers(v: Variant, categories: tuple[str, ...]) -> Variant:
    """Stamp the RTS expected categories on an emitted twin (`EXPECTED_BLOCKERS` in `core` names training operators only)."""
    spec = HonestTaskSpec.model_validate({**v.spec.model_dump(), "expected_blockers": list(categories)})
    cfg = read_task_toml(v.task_dir)
    md = dict(cfg.get("metadata") or {})
    md["honest"] = spec.model_dump(mode="json", exclude_none=True)
    write_task_toml(v.task_dir, {**cfg, "metadata": md})
    return dataclasses.replace(v, spec=spec)


def pair_problems(src: Path, twin: Variant, control: Variant) -> list[str]:
    """Instruction identity and the no-label scan (spec 4, the gate's static half)."""
    out = []
    if (twin.task_dir / "instruction.md").read_bytes() != (control.task_dir / "instruction.md").read_bytes():
        out.append("twin and control instructions differ")
    for v in (twin, control):
        out += [f"{v.variant_id}: label word at {leak.path}:{leak.line}" for leak in label_leaks(v.task_dir, src)]
    return out


@dataclasses.dataclass(frozen=True)
class Side:
    """One member of a pair: the edit applied to the copy, its spec overrides, operator detail and planted text."""

    edit: Callable[[Path], None]
    spec: dict[str, Any]
    detail: dict[str, Any]
    env_text: dict[str, str]


def emit_pair(
    src: Path, out_root: Path, *, pair_id: str, op: str, subject: str, twin: Side, control: Side
) -> tuple[Variant, Variant]:
    """Write twin and control, stamp the twin's categories, and check the pair; raise `PairError` (both removed) if not."""
    detail = {"rts_tier": TIER}
    t = _emit(
        src,
        out_root,
        pair_id=pair_id,
        op=op,
        detail=twin.detail,
        spec_overrides={**twin.spec, "detail": detail},
        edit=twin.edit,
        subject=subject,
        env_text=twin.env_text,
    )
    t = _with_blockers(t, CATEGORY)
    c = _emit(
        src,
        out_root,
        pair_id=pair_id,
        op=op,
        control=True,
        detail=control.detail,
        spec_overrides={**control.spec, "detail": detail},
        edit=control.edit,
        subject=subject,
        env_text=control.env_text,
    )
    if problems := pair_problems(src, t, c):
        for v in (t, c):
            shutil.rmtree(v.task_dir, ignore_errors=True)
        raise PairError(f"{pair_id} {op}: " + "; ".join(problems))
    return t, c
