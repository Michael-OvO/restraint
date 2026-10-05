"""Writing a variant to disk: copy the source task, apply an edit, scrub labels, check that it still loads."""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from honest_agents.spec import VERIFIER_LEGACY, HonestTaskSpec
from honest_agents.tasks.analysis import (
    TaskFacts,
    appended_text,
    environment_text,
    subject_features,
)
from honest_agents.tasks.perturb.core import EXPECTED_BLOCKERS, Variant, task_name, variant_id
from honest_agents.tasks.taskfs import (
    copy_task,
    normalise_instruction,
    read_task_toml,
    scrub_environment,
    set_honest,
    set_task_name,
    write_task_toml,
)

# ── emitting variants ────────────────────────────────────────────────────────


def _source_honest(src: Path) -> dict[str, Any]:
    """`[metadata.honest]` of the source: carries the verifier (`pytest`) the data pipeline stamped on the seed."""
    return dict(((read_task_toml(src).get("metadata") or {}).get("honest")) or {})


def _emit(
    src: Path,
    out_root: Path,
    *,
    pair_id: str,
    op: str | None,
    control: bool = False,
    detail: dict[str, Any] | None = None,
    spec_overrides: dict[str, Any] | None = None,
    agent_overrides: dict[str, Any] | None = None,
    edit: Callable[[Path], None] | None = None,
    subject: str | None = None,
) -> Variant:
    """Copy `src`, apply `edit` to the copy, and stamp task.toml (unique name, agent overrides, honest spec).

    `op=None` is the plain feasible copy, `op` alone the infeasible variant, `op` with `control=True` its near-miss
    control (feasible, `is_control`, `control_of=op`).
    """
    vid = variant_id(pair_id, op, control)
    dst = copy_task(src, out_root / vid)
    scrub_environment(dst)  # the expected state never ships, whatever the seed's source did
    if edit is not None:
        edit(dst)
    normalise_instruction(dst)  # plain and perturbed instructions share one format
    scrub_labels(dst)
    detail = dict(detail or {})
    if subject:
        # What `existence_auc` compares between impossible variants and controls, read off the emitted image.
        facts = TaskFacts.load(dst)
        detail["subject"] = subject
        detail["subject_features"] = subject_features(
            facts,
            subject,
            (dst / "instruction.md").read_text(),
            detail.get("position"),
            original_instruction=(src / "instruction.md").read_text(),
        )
        # What `envtext` compares: the README, the other planted files and the lock words this operator added.
        source_dockerfile = src / "environment" / "Dockerfile"
        detail["env_text"] = environment_text(
            facts,
            appended_text(facts.dockerfile, source_dockerfile.read_text() if source_dockerfile.exists() else None),
        )
    infeasible = op is not None and not control
    base = _source_honest(src)
    spec = HonestTaskSpec(
        feasible=not infeasible,
        perturbation=op,
        expected_blockers=list(EXPECTED_BLOCKERS.get(op, [])) if infeasible else [],
        pair_id=pair_id,
        source_task=pair_id,
        family=op,
        variant_id=vid,
        is_control=control,
        control_of=op if control else None,
        verifier=base.get("verifier", VERIFIER_LEGACY),
        verifier_site=bool(base.get("verifier_site", False)),
        **(spec_overrides or {}),
    )
    cfg = read_task_toml(dst)
    cfg = set_task_name(cfg, task_name(vid))
    if agent_overrides:
        agent = {**(cfg.get("agent") or {}), **agent_overrides}
        if agent.get("network_mode") != "allowlist":
            agent.pop("allowed_hosts", None)  # Harbor rejects allowed_hosts unless network_mode = "allowlist"
        cfg["agent"] = agent
    variant_detail = None
    if op:
        variant_detail = {"operator": op, "control": control, **(detail or {})}
    cfg = set_honest(cfg, spec, variant_detail)
    write_task_toml(dst, cfg)
    _check_loads(dst)
    return Variant(variant_id=vid, task_dir=dst, spec=spec, detail=detail)


# What `append_dockerfile` / `prepend_solve` write to keep a perturbation auditable. Useful in a diff, harmful in an
# image: "# honest-agents: high_risk abstain twin /srv/x" names the label. The role lives in `[metadata.honest]`.
_MARKER_LINE = re.compile(r"^[ \t]*#[ \t]*honest-agents\b.*(?:\n|$)", re.MULTILINE)


def scrub_labels(task_dir: Path) -> None:
    """Remove the operator comments (`# honest-agents ...`) from the emitted Dockerfile and oracle."""
    for rel in ("environment/Dockerfile", "solution/solve.sh"):
        f = task_dir / rel
        if not f.is_file():
            continue
        text = f.read_text()
        stripped = _MARKER_LINE.sub("", text)
        if stripped != text:
            f.write_text(stripped)


def _check_loads(task_dir: Path) -> None:
    """Fail at build time (operator skipped and logged) rather than at sandbox boot on the training node."""
    from harbor.models.task.task import Task

    try:
        Task(task_dir)
    except Exception:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise


def make_feasible(src: Path, out_root: Path, *, pair_id: str, seed: int = 0, split: str = "train") -> Variant:
    """The unperturbed member of the pair: identical task, labelled feasible."""
    return _emit(src, out_root, pair_id=pair_id, op=None)
