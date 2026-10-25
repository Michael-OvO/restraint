"""Builders for RTS tests: `n` pairs of one (benchmark, tier, operator), one source task per pair (tier in its id)."""

from __future__ import annotations

from typing import Any

from honest_agents.eval.records import EpisodeRecord
from honest_agents.eval.rts.manifest import ManifestItem
from tests.eval.scripted_policy import COMPLETE, abstain, make_record

BLOCK = abstain("missing_capability", "/x")


def pairs(
    benchmark: str, tier: str, operator: str, n: int, *, categories: tuple[str, ...] = ("missing_capability",)
) -> list[ManifestItem]:
    out = []
    for k in range(n):
        src = f"{benchmark}/{tier}/{operator}-{k}"
        pid = f"{src}::{operator}"
        out.append(
            ManifestItem(f"{pid}__twin", benchmark, src, pid, tier, operator, "twin", expected_categories=categories)
        )
        out.append(ManifestItem(f"{pid}__control", benchmark, src, pid, tier, operator, "control"))
    return out


def episodes(
    items: list[ManifestItem],
    *,
    hits: int,
    false_refusals: int,
    control_pass: bool = True,
    twin_report: dict[str, Any] | None = None,
) -> list[EpisodeRecord]:
    """One episode per item, in pair order: the first `hits` twins decline (`BLOCK`, or `twin_report`), the rest claim
    and fail; the first `false_refusals` controls are refused with `BLOCK`, the rest claim complete and pass iff
    `control_pass`."""
    twins = [i for i in items if i.role == "twin"]
    controls = [i for i in items if i.role == "control"]
    recs = [
        make_record(
            i.item_id,
            feasible=False,
            report=(twin_report or BLOCK) if k < hits else COMPLETE,
            passed=False,
            pair_id=i.pair_id,
            source_task=i.source_task,
        )
        for k, i in enumerate(twins)
    ]
    for k, i in enumerate(controls):
        refused = k < false_refusals
        recs.append(
            make_record(
                i.item_id,
                feasible=True,
                is_control=True,
                report=BLOCK if refused else COMPLETE,
                passed=(not refused) and control_pass,
                pair_id=i.pair_id,
                source_task=i.source_task,
            )
        )
    return recs
