"""Seed-group splits: hash buckets, near-duplicate-aware assignment and the frozen test-id list."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from honest_agents.tasks.build.config import BUCKETS, DEFAULT_TEST_BP, DEFAULT_VAL_BP, SPLIT_SALT, SPLITS, Row

# ── splits ───────────────────────────────────────────────────────────────────


def hash_key(seed: int, group: str) -> int:
    return int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()[:12], 16)


def split_bucket(group: str, seed: int = 0) -> int:
    """Where a source group sits in [0, 10000): a function of the group id and the salt only, never of the pool."""
    return int(hashlib.sha256(f"{SPLIT_SALT}|{seed}|{group}".encode()).hexdigest()[:12], 16) % BUCKETS


def split_for_bucket(bucket: int, test_bp: int = DEFAULT_TEST_BP, val_bp: int = DEFAULT_VAL_BP) -> str:
    return "test" if bucket < test_bp else "val" if bucket < test_bp + val_bp else "train"


_HELDOUT_ORDER = {"test": 0, "val": 1, "train": 2}


def assign_seed_splits(
    groups: Iterable[str],
    *,
    n_test: int | None = None,
    n_val: int | None = None,
    test_bp: int = DEFAULT_TEST_BP,
    val_bp: int = DEFAULT_VAL_BP,
    seed: int = 0,
    previous: dict[str, str] | None = None,
    frozen_test: Iterable[str] = (),
    clusters: dict[str, str] | None = None,
) -> dict[str, str]:
    """Split group -> `train` / `val` / `test`.

    Default (`n_test` and `n_val` None): the group's hash bucket against the fixed cutoffs, so the answer for a group
    does not depend on which other groups are in the pool. With `n_test` / `n_val` (legacy, dev pools): the unit
    hashes first fill the test quota, then the validation quota, the rest train; that depends on the pool.

    `frozen_test` groups (the committed test ids) are test whatever the hash says; groups in `previous` keep their
    split. `clusters` (group -> cluster id) makes a near-duplicate cluster one unit: a new member joins the most
    held-out split among its already-assigned mates, otherwise the cluster takes the split of its smallest member id.
    A member that `previous` (a build made before the clustering) left in train moves to the most held-out split of its
    mates: moving a source out of train never leaks. Deterministic in `seed`.
    """
    groups = sorted(set(groups))
    frozen = set(frozen_test)
    out = {g: "test" for g in groups if g in frozen}
    out.update({g: previous[g] for g in groups if g not in out and previous and previous.get(g) in SPLITS})
    members: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        members[(clusters or {}).get(g, g)].append(g)
    count_mode = n_test is not None or n_val is not None
    have = Counter(out.values())
    order = sorted(members, key=(lambda u: hash_key(seed, u)) if count_mode else (lambda u: u))
    for unit in order:
        todo = [g for g in members[unit] if g not in out]
        if not todo:
            continue
        mates = [out[g] for g in members[unit] if g in out]
        if mates:
            split = min(mates, key=_HELDOUT_ORDER.__getitem__)
        elif count_mode:
            split = "test" if have["test"] < (n_test or 0) else "val" if have["val"] < (n_val or 0) else "train"
        else:
            split = split_for_bucket(split_bucket(min(members[unit]), seed), test_bp, val_bp)
        for g in todo:
            out[g] = split
            have[split] += 1
    for unit, gs in members.items():  # a twin that an earlier build already put in train follows its held-out mates
        if len(gs) < 2:
            continue
        top = min((out[g] for g in gs), key=_HELDOUT_ORDER.__getitem__)
        for g in gs:
            if out[g] == "train" and top != "train":
                out[g] = top
    return out


# ── test ids: the frozen held-out set ────────────────────────────────────────


def ids_digest(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def write_test_ids(path: Path, ids: Iterable[str]) -> str:
    """test_ids.txt: comment header, a `# sha256:` line over the sorted ids, then one id per line. Returns the digest."""
    ids = sorted(set(ids))
    digest = ids_digest(ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# honest-agents held-out test ids (split groups, i.e. source/task). Commit this file.\n"
        "# A build refuses to emit a train, val or train_pool row for any of them.\n"
        f"# count: {len(ids)}\n# sha256: {digest}\n" + "".join(f"{i}\n" for i in ids)
    )
    return digest


def read_test_ids(path: Path) -> set[str]:
    """The ids in `path`; raises if the sha256 line is missing or does not match (a hand-edited or truncated file)."""
    ids, digest = set(), None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("# sha256:"):
            digest = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#"):
            ids.add(line)
    if digest is None:
        raise ValueError(f"{path} has no '# sha256:' line")
    if ids_digest(ids) != digest:
        raise ValueError(
            f"{path}: sha256 does not match its ids (edited or truncated); restore it from version control"
        )
    return ids


def check_no_test_leak(rows: Iterable[Row], test_ids: set[str]) -> None:
    """Refuse to emit a non-test row whose pair_id or source group is a held-out test id."""
    bad = sorted(
        {r.pair_id for r in rows if r.split != "test" and (r.pair_id in test_ids or r.split_group in test_ids)}
    )
    if bad:
        raise ValueError(f"{len(bad)} held-out test ids would be emitted outside the test split, e.g. {bad[:3]}")


def phrasing_split(split: str) -> str:
    """Which operator pool a seed's variants draw from: test seeds use the test pools, everything else the train pools."""
    return "test" if split == "test" else "train"
