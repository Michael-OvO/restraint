"""Builders for RTS tests: `n` pairs of one (benchmark, tier, operator), one source task per pair (tier in its id)."""

from __future__ import annotations

from honest_agents.eval.rts.manifest import ManifestItem


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
