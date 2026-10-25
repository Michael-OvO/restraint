"""The RTS manifest (spec section 4, item 5): one JSON line per item, an impossible twin or its near-miss control.

The manifest *is* the benchmark. It is written once with a sha256 sidecar (`<manifest>.sha256`) and never edited; a
change makes a new version directory. `item_id` equals the `EpisodeRecord.variant_key` of the item's episodes, which is
how a run is joined to it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TIERS = ("near", "mid", "far")
ROLES = ("twin", "control")
_PAIR_KEYS = ("benchmark", "source_task", "tier", "operator")


@dataclass(frozen=True)
class ManifestItem:
    item_id: str
    benchmark: str
    source_task: str  # the cluster: what the bootstrap resamples
    pair_id: str
    tier: str
    operator: str
    role: str
    expected_categories: tuple[str, ...] = ()  # blocker categories a strict hit must name (twins)
    expected_targets: tuple[str, ...] = ()
    instruction_identical: bool = True  # twin and control instructions are byte-identical
    task_dir_sha256: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"{self.item_id}: tier {self.tier!r} not in {TIERS}")
        if self.role not in ROLES:
            raise ValueError(f"{self.item_id}: role {self.role!r} not in {ROLES}")

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_categories"] = list(self.expected_categories)
        d["expected_targets"] = list(self.expected_targets)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ManifestItem:
        d = dict(d)
        d["expected_categories"] = tuple(d.get("expected_categories", ()))
        d["expected_targets"] = tuple(d.get("expected_targets", ()))
        return cls(**d)


def pair_problems(items: Sequence[ManifestItem]) -> list[str]:
    """Every pair holds one twin and one control that agree on benchmark, source task, tier and operator."""
    problems = []
    dup = sorted(k for k, n in Counter(i.item_id for i in items).items() if n > 1)
    if dup:
        problems.append(f"duplicate item ids: {dup}")
    by_pair: dict[str, list[ManifestItem]] = defaultdict(list)
    for i in items:
        by_pair[i.pair_id].append(i)
    for pid, members in sorted(by_pair.items()):
        roles = sorted(m.role for m in members)
        if roles != ["control", "twin"]:
            problems.append(f"pair {pid}: roles {roles}, want one twin and one control")
            continue
        for k in _PAIR_KEYS:
            if len({getattr(m, k) for m in members}) > 1:
                problems.append(f"pair {pid}: twin and control differ in {k}")
    return problems


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_manifest(items: Iterable[ManifestItem], path: str | Path) -> str:
    """Write the manifest (sorted by item id) and its digest sidecar; refuse an invalid set. Returns the digest."""
    path = Path(path)
    items = sorted(items, key=lambda i: i.item_id)
    if problems := pair_problems(items):
        raise ValueError("invalid manifest:\n  " + "\n  ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i.to_json(), sort_keys=True) + "\n" for i in items))
    digest = sha256_of(path)
    _sidecar(path).write_text(digest + "\n")
    return digest


def load_manifest(path: str | Path) -> list[ManifestItem]:
    """Read a manifest; refuse it without its digest sidecar, if it no longer matches it, or if a pair is broken."""
    path = Path(path)
    side = _sidecar(path)
    if not side.is_file():
        raise ValueError(f"{path}: no {side.name}; a manifest is only read with its digest")
    want, got = side.read_text().strip(), sha256_of(path)
    if want != got:
        raise ValueError(f"{path}: sha256 {got} does not match {side.name} ({want}); manifests are never edited")
    items = [ManifestItem.from_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if problems := pair_problems(items):
        raise ValueError(f"{path}: invalid manifest:\n  " + "\n  ".join(problems))
    return items
