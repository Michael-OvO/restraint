"""Build the honest-agents task dataset: sources -> dedup -> split -> variants -> (validate) -> mixture -> JSONL.

    python -m honest_agents.tasks.build --out data --sources endless-terminals --limit 600 \\
        --test-seeds 300 --val-seeds 60 --mixture 0.55,0.20,0.25

Layout written under --out:
    seeds/<source>/<task>/          normalised feasible seed tasks
    tasks/<variant_id>/             plain feasible copies, infeasible variants and their near-miss controls
    train_pool.jsonl                every train row before the difficulty gate and the mixture (what Phase 0 can sample)
    {train,val,test}.jsonl          rows for TRL / the pilot / evaluation (INTERFACES_V3 §6 columns)
    split_seeds.json                seed group -> split, kept across rebuilds so a test seed never reaches training
                                    (`--split-seeds-file` / `--split-seeds` reuses one when building into a fresh --out)
    test_ids.txt                    every held-out seed group plus a sha256 line; commit it. A build refuses to emit a
                                    train, val or train_pool row for any pair_id or source in it (`--test-ids-file`)
    relabels.json                   variants validation relabelled as controls (re-applied on every rebuild)
    dedup.jsonl, validation.jsonl, manifest.json

SPEC_V3 §4 rules implemented here:

* **Pool-independent splits** (L2): a source group goes to `test` / `val` / `train` by `sha256(salt|group) mod 10000`
  against fixed cutoffs (`--test-bp 2200 --val-bp 400`, about 330 and 60 of a 1,500-task pool), so changing `--limit`,
  the sample seed or the pool size never moves a source. The legacy `--test-seeds N --val-seeds M` ranks the pool
  instead (only for tiny dev pools; it warns). Either way `split_seeds.json` and `test_ids.txt` persist the assignment
  and the build refuses to put a test id into any other split. A pair never crosses splits.
* **Near-duplicate clusters** (`--neardup-threshold`): sources whose instructions (word 3-gram Jaccard) or tests (5-gram)
  nearly coincide are split as one unit, so a paraphrase of a test task never trains.
* **Decontamination is mandatory** (L4): `--benchmark-dir` must be given and non-empty (`scripts/fetch_benchmarks.sh`),
  unless `--i-know-no-benchmark`. Seeds with a Terminal-Bench canary, or whose Dockerfile COPY/ADDs tests or expected
  files, are rejected; `environment/{task.json,test_initial_state.py,container.def}` are deleted (see `sources.py`).
* **Two held-out families** appear only in `test`: `high_risk` and one environment-mechanism family
  (default `revoke_permission`). Their controls are held out with them.
* **Phrasing and name pools**: variants of test seeds are built from the test pools in `perturb.POOLS`, everything else
  from the train pools, so no test phrasing or sentinel path is ever seen in training.
* **Mixture** (`--mixture`, default 55% plain feasible / 20% near-miss controls / 25% infeasible) is applied to `train` and
  `val`; `test` keeps every variant. Infeasible variants and their controls are drawn as PAIRS (same source, same
  operator), round-robin over families, so within a family the two kinds come from the same sources in the same
  proportion and the family carries no label prior (`--no-pair-mixture` restores the unpaired draw). In paired mode
  the control and infeasible shares are averaged (0.55/0.20/0.25 becomes 0.55/0.225/0.225).
* **Controls 1:1** per infeasible variant at generation time, sharing the `pair_id`; an infeasible variant without a
  control is dropped (and a control whose twin was dropped goes with it).
* **Quarantine**: variants on the quarantine list (`bank.quarantined_variants`) are excluded on every rebuild, together
  with their controls. Leaks found by validation are appended to the list. Entries found while evaluating
  (`source = "eval"`, held-out variants) are ignored unless `--quarantine-include-eval`, so they cannot change test.jsonl.
* **Validation**: Docker needed, so a typical flow is: build, run `python -m honest_agents.tasks.validate` on the Linux
  node, rebuild with `--validation-jsonl`. A variant the repair probe relabelled (`relabel_to_control`) becomes a control
  at the next build and is re-validated as one; a leak is quarantined.
* **Difficulty gate** (`--difficulty-json` from `tasks.difficulty`) filters the *training* split only.
* **Hermetic verifier**: seeds that are not graded by the offline pytest verifier are dropped unless
  `--allow-legacy-verifier` (SPEC_V3 §3.2: `train.py` refuses them anyway).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honest_agents.bank import (
    DEFAULT_IGNORED_SOURCES,
    QUARANTINE_LEAK,
    QuarantineEntry,
    append_quarantine,
    quarantined_variants,
    read_quarantine,
)
from honest_agents.spec import HonestTaskSpec
from honest_agents.tasks.analysis import TaskFacts
from honest_agents.tasks.dedup import Deduplicator, NearDupReport, load_benchmark_tasks, near_duplicate_clusters
from honest_agents.tasks.difficulty import DifficultyGate
from honest_agents.tasks.perturb import (
    LEAK_DROPPED,
    OPERATORS,
    Variant,
    deliverable_names,
    generate_variants,
    set_name_bank,
)
from honest_agents.tasks.sources import (
    SOURCES,
    SeedReject,
    SeedTask,
    local_seeds,
    prepare_source,
)
from honest_agents.tasks.taskfs import read_task_toml, write_task_toml

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
KINDS = ("feasible", "control", "infeasible")  # the mixture strata, in the order of `DEFAULT_MIXTURE`
DEFAULT_MIXTURE = (0.55, 0.20, 0.25)  # SPEC_V3 §4.5 (L7)
ENV_FAMILIES = ("remove_tool", "remove_input", "revoke_permission", "no_network")  # environment-mechanism operators
DEFAULT_HELDOUT_FAMILIES = ("high_risk", "revoke_permission")  # SPEC_V3 §4.6: high_risk + one environment mechanism
DEFAULT_TEST_SEEDS = (
    300  # V-2: at least 300 test pairs, so that every reported family has at least MIN_FAMILY_TEST_ROWS
)
MIN_FAMILY_TEST_ROWS = (
    50  # V-2: infeasible variants per reported family in the test split (fewer: pool or drop the family)
)
DEFAULT_VAL_SEEDS = 60  # about 100 variants after the mixture and the held-out families (SPEC_V3 §9.3 probe)
DEFAULT_TEST_BP = 2200  # test share of the hash space in basis points: about 330 of a 1,500-task pool (V-2 wants 300)
DEFAULT_VAL_BP = 400  # about 60 of 1,500
SPLIT_SALT = "honest-agents-split-v1"
BUCKETS = 10_000
SPLIT_SEEDS_FILE = "split_seeds.json"
TEST_IDS_FILE = "test_ids.txt"
DEFAULT_NEARDUP_THRESHOLD = 0.1
RELABELS_FILE = "relabels.json"
COLUMNS = (
    "prompt",
    "task_dir",
    "variant_id",
    "pair_id",
    "feasible",
    "is_control",
    "kind",
    "family",
    "perturbation",
    "source",
    "split",
)  # INTERFACES_V3 §6: exactly these


@dataclass
class BuildConfig:
    out: Path
    sources: list[str] = field(default_factory=lambda: ["endless-terminals"])
    raw_dir: Path | None = None
    seeds_dir: Path | None = None  # for the "local" source: existing Harbor task dirs
    limit: int | None = None
    download: bool = True
    operators: list[str] = field(default_factory=lambda: list(OPERATORS))
    heldout_families: list[str] = field(default_factory=lambda: list(DEFAULT_HELDOUT_FAMILIES))
    test_bp: int = DEFAULT_TEST_BP  # fixed hash cutoffs (basis points of [0, 10000)): pool-independent (L2)
    val_bp: int = DEFAULT_VAL_BP
    test_seeds: int | None = None  # legacy: rank the pool and hold out exactly this many (pool-dependent; dev only)
    val_seeds: int | None = None
    split_seed: int = 0  # salt of the split hash; never changes with --seed or the pool
    test_ids_file: Path | None = None  # held-out ids to keep (default <out>/test_ids.txt)
    neardup_threshold: float = DEFAULT_NEARDUP_THRESHOLD  # word 3-gram Jaccard of instructions; 0 = no clustering
    i_know_no_benchmark: bool = False  # build without --benchmark-dir (tests only, never for training data)
    allow_disabled_sources: bool = False  # SETA / tmax (see sources.SOURCES)
    pair_mixture: bool = True  # draw infeasible and control rows as pairs
    quarantine_include_eval: bool = False  # honour quarantine entries whose source is "eval" (they change test.jsonl)
    enforce_cue_band: bool = True  # fail the build when a family's existence AUC leaves [0.4, 0.6]
    mixture: tuple[float, float, float] = DEFAULT_MIXTURE  # plain feasible, controls, infeasible
    benchmark_dirs: list[Path] = field(default_factory=list)
    dedup_threshold: float = 0.5
    feasible_allowlist: Path | None = None  # pair_ids of no-solution tasks that passed a pass@k filter
    validate: bool = False
    validation_jsonl: Path | None = None
    concurrency: int = 8
    allow_oracle_tamper: bool = False
    task_dir_root: str | None = None  # path prefix written into rows instead of <out>/tasks
    quarantine_file: Path | None = None  # default: the bank module's configured list, if any
    quarantine_kinds: tuple[str, ...] = (QUARANTINE_LEAK,)
    difficulty_json: Path | None = None  # output of `tasks.difficulty`; filters the train split
    difficulty_strict: bool = False  # keep only source tasks Phase 0 measured and kept (not the unmeasured ones too)
    split_seeds_file: Path | None = None  # seed -> split assignment to keep (default <out>/split_seeds.json)
    allow_legacy_verifier: bool = False  # keep seeds that are not hermetic (evaluation sets, tests)
    require_controls: bool = True
    # SPEC_V3 §4.2: the oracle must pass 3 of 3 and the null check must have shown the tests fail on the untouched
    # environment. A validation.jsonl made with --no-null-check or fewer oracle runs is refused unless relaxed here.
    require_null_check: bool = True
    min_oracle_runs: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        self.mixture = normalise_mixture(self.mixture)
        for name, n in (
            ("test_seeds", self.test_seeds),
            ("val_seeds", self.val_seeds),
            ("test_bp", self.test_bp),
            ("val_bp", self.val_bp),
        ):
            if n is not None and n < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.test_bp + self.val_bp > BUCKETS:
            raise ValueError("test_bp + val_bp must not exceed 10000")


@dataclass
class Row:
    prompt: list[dict[str, str]]
    task_dir: str
    variant_id: str
    pair_id: str
    feasible: bool
    is_control: bool
    kind: str
    family: str
    perturbation: str | None
    source: str
    split: str = "train"
    split_group: str = field(default="", repr=False)


def normalise_mixture(mixture: Sequence[float]) -> tuple[float, float, float]:
    m = tuple(float(x) for x in mixture)
    if len(m) != len(KINDS) or any(x < 0 for x in m) or sum(m) <= 0:
        raise ValueError(
            f"mixture needs {len(KINDS)} non-negative shares (feasible, control, infeasible), got {mixture}"
        )
    if abs(sum(m) - 1.0) > 0.01:
        raise ValueError(f"mixture shares must sum to 1, got {sum(m):.3f}")
    total = sum(m)
    return tuple(x / total for x in m)  # type: ignore[return-value]


def heldout_problems(families: Sequence[str]) -> list[str]:
    """Ways a held-out family list departs from SPEC_V3 §4.6 (empty = conforms). A deviation is allowed but logged."""
    problems = []
    if families and "high_risk" not in families:
        problems.append("high_risk is not held out")
    if families and not any(f in ENV_FAMILIES for f in families):
        problems.append("no environment-mechanism family is held out")
    unknown = [f for f in families if f not in OPERATORS]
    if unknown:
        problems.append(f"unknown families {unknown}")
    return problems


def system_prompt() -> str:
    from honest_agents.harness import SYSTEM_PROMPT

    return SYSTEM_PROMPT


def make_prompt() -> list[dict[str, str]]:
    """System prompt + empty user turn: TRL appends `reset()`'s return (the instruction) to the last message."""
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": ""}]


# ── splits ───────────────────────────────────────────────────────────────────


def _hash_key(seed: int, group: str) -> int:
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
    does not depend on which other groups are in the pool (L2). With `n_test` / `n_val` (legacy, dev pools): the unit
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
    order = sorted(members, key=(lambda u: _hash_key(seed, u)) if count_mode else (lambda u: u))
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


# ── mixture ──────────────────────────────────────────────────────────────────


def _round_robin(rows: list[Row], group: Callable[[Row], str], n: int, rng: random.Random) -> list[Row]:
    """Up to `n` rows, one group at a time in turn (groups in name order, rows within a group in random order)."""
    queues = defaultdict(list)
    for r in rows:
        queues[group(r)].append(r)
    qs = [rng.sample(queues[k], len(queues[k])) for k in sorted(queues)]
    out: list[Row] = []
    while len(out) < n and any(qs):
        for q in qs:
            if q and len(out) < n:
                out.append(q.pop())
    return out


def balance_mixture(
    rows: list[Row], mixture: Sequence[float] = DEFAULT_MIXTURE, seed: int = 0, *, paired: bool = True
) -> list[Row]:
    """Down-sample to the requested plain-feasible / control / infeasible mixture (SPEC_V3 §4.5).

    Paired (default): an infeasible variant and its control (same source, same operator) are drawn together, round-robin
    over families, so inside a family the two kinds have the same count and the same sources and a family-level cue
    cannot separate them by prior or by source. The control and infeasible shares are averaged to make that possible.
    Controls without an infeasible twin (relabelled ones) are not drawn. Plain feasible rows prefer the sources that
    contributed a pair. Unpaired: infeasible rows round-robin, a control for each chosen variant first, then others.
    A stratum with no rows leaves the split as it is (the ratio is unattainable).
    """
    share = dict(zip(KINDS, normalise_mixture(mixture), strict=True))
    by_kind = {k: [r for r in rows if r.kind == k] for k in KINDS}
    needed = [k for k in KINDS if share[k] > 0]
    if any(not by_kind[k] for k in needed):
        log.warning(
            "mixture: %s, left as is",
            ", ".join(f"{len(by_kind[k])} {k}" for k in KINDS),
        )
        return list(rows)
    rng = random.Random(seed)
    if paired and share["control"] > 0 and share["infeasible"] > 0:
        return _balance_paired(rows, by_kind, share, rng)
    total = min(len(by_kind[k]) / share[k] for k in needed)
    want = {k: min(len(by_kind[k]), round(total * share[k])) for k in KINDS}

    chosen_inf = _round_robin(by_kind["infeasible"], lambda r: r.family, want["infeasible"], rng)
    twins = {(r.pair_id, r.perturbation) for r in chosen_inf}
    ctrl_twin = [r for r in by_kind["control"] if (r.pair_id, r.perturbation) in twins]
    ctrl_rest = [r for r in by_kind["control"] if (r.pair_id, r.perturbation) not in twins]
    chosen_ctl = _round_robin(ctrl_twin, lambda r: r.family, want["control"], rng)
    chosen_ctl += _round_robin(ctrl_rest, lambda r: r.family, want["control"] - len(chosen_ctl), rng)
    chosen_feas = _choose_feasible(
        by_kind["feasible"], {r.pair_id for r in chosen_inf + chosen_ctl}, want["feasible"], rng
    )
    keep = {id(r) for r in chosen_inf + chosen_ctl + chosen_feas}
    return [r for r in rows if id(r) in keep]


def _choose_feasible(feas: list[Row], paired: set[str], n: int, rng: random.Random) -> list[Row]:
    chosen = _round_robin([r for r in feas if r.pair_id in paired], lambda r: r.pair_id, n, rng)
    return chosen + _round_robin(
        [r for r in feas if r.pair_id not in paired], lambda r: r.pair_id, n - len(chosen), rng
    )


def _balance_paired(
    rows: list[Row], by_kind: dict[str, list[Row]], share: dict[str, float], rng: random.Random
) -> list[Row]:
    pair_share = (share["control"] + share["infeasible"]) / 2
    ctl = {}
    for r in by_kind["control"]:
        ctl.setdefault((r.pair_id, r.perturbation), r)
    pairs = [(r, ctl[(r.pair_id, r.perturbation)]) for r in by_kind["infeasible"] if (r.pair_id, r.perturbation) in ctl]
    if not pairs:
        log.warning("mixture: no infeasible variant has a control twin, left as is")
        return list(rows)
    n_feas = len(by_kind["feasible"])
    total = (
        min(n_feas / share["feasible"], len(pairs) / pair_share) if share["feasible"] > 0 else len(pairs) / pair_share
    )
    want_pairs = min(len(pairs), round(total * pair_share))
    by_inf = {id(inf): (inf, c) for inf, c in pairs}
    chosen_inf = _round_robin([inf for inf, _ in pairs], lambda r: r.family, want_pairs, rng)
    chosen_ctl = [by_inf[id(r)][1] for r in chosen_inf]
    want_feas = min(n_feas, round(total * share["feasible"]))
    chosen_feas = _choose_feasible(by_kind["feasible"], {r.pair_id for r in chosen_inf}, want_feas, rng)
    keep = {id(r) for r in chosen_inf + chosen_ctl + chosen_feas}
    return [r for r in rows if id(r) in keep]


# ── seeds ────────────────────────────────────────────────────────────────────


def collect_seeds(cfg: BuildConfig, rejected: list[SeedReject] | None = None) -> list[SeedTask]:
    raw = cfg.raw_dir or cfg.out / "raw"
    seeds: list[SeedTask] = []
    for src in cfg.sources:
        if src == "local":
            if cfg.seeds_dir is None:
                raise ValueError("--sources local needs --seeds-dir")
            seeds += local_seeds(cfg.seeds_dir, rejected=rejected)
        elif src in SOURCES:
            seeds += prepare_source(
                src,
                raw,
                cfg.out / "seeds",
                limit=cfg.limit,
                seed=cfg.seed,
                download_missing=cfg.download,
                hermetic=False if cfg.allow_legacy_verifier else None,
                rejected=rejected,
                allow_disabled=cfg.allow_disabled_sources,
            )
        else:
            raise ValueError(f"unknown source {src!r}; known: {sorted(SOURCES)} + ['local']")
    return seeds


def check_benchmark_config(cfg: BuildConfig) -> None:
    """Decontamination is mandatory (L4): fail before any download when there is nothing to deduplicate against."""
    if cfg.i_know_no_benchmark:
        return
    if not cfg.benchmark_dirs:
        raise ValueError(
            "--benchmark-dir is required: run scripts/fetch_benchmarks.sh and pass its output dir "
            "(--i-know-no-benchmark builds without decontamination; tests only, never training data)"
        )
    missing = [str(d) for d in cfg.benchmark_dirs if not d.is_dir()]
    if missing:
        raise ValueError(f"--benchmark-dir does not exist: {missing} (run scripts/fetch_benchmarks.sh)")


def dedup_seeds_report(
    cfg: BuildConfig, seeds: list[SeedTask]
) -> tuple[list[SeedTask], list[dict[str, Any]], dict[str, Any]]:
    """Drop seeds that duplicate a benchmark task; also returns overlap statistics for the manifest."""
    check_benchmark_config(cfg)
    if not cfg.benchmark_dirs:
        log.warning("--i-know-no-benchmark: seeds are NOT deduplicated against the eval benchmarks")
        return seeds, [], {"checked": False}
    bench = [b for d in cfg.benchmark_dirs for b in load_benchmark_tasks(d)]
    if not bench and not cfg.i_know_no_benchmark:
        raise ValueError(f"no benchmark tasks (instruction.md) found under {[str(d) for d in cfg.benchmark_dirs]}")
    dd = Deduplicator(bench, threshold=cfg.dedup_threshold)
    kept, dropped, best = [], [], []
    for s in seeds:
        text = (s.task_dir / "instruction.md").read_text()
        _, j = dd.best_overlap(text)
        best.append(j)
        hit = dd.match(s.task_id.split("/")[-1], text)
        if hit is None:
            kept.append(s)
        else:
            dropped.append({"pair_id": s.pair_id, "benchmark": hit[0].benchmark, "match": hit[0].name, "why": hit[1]})
    log.info("dedup: dropped %d/%d seeds against %d benchmark tasks", len(dropped), len(seeds), len(bench))
    stats = {
        "checked": True,
        "n_benchmark_tasks": len(bench),
        "benchmarks": dict(Counter(b.benchmark for b in bench)),
        "threshold": cfg.dedup_threshold,
        "n_seeds_checked": len(seeds),
        "n_dropped": len(dropped),
        "max_8gram_jaccard": round(max(best, default=0.0), 4),
        "n_seeds_with_any_8gram_overlap": sum(j > 0 for j in best),
    }
    return kept, dropped, stats


def dedup_seeds(cfg: BuildConfig, seeds: list[SeedTask]) -> tuple[list[SeedTask], list[dict[str, Any]]]:
    kept, dropped, _ = dedup_seeds_report(cfg, seeds)
    return kept, dropped


def seed_documents(seeds: Iterable[SeedTask]) -> dict[str, tuple[str, str]]:
    """Split group -> (instruction, tests text), for near-duplicate clustering. Members of a group are concatenated."""
    docs: dict[str, tuple[str, str]] = {}
    for s in seeds:
        inst = (s.task_dir / "instruction.md").read_text(errors="replace")
        tests = "\n".join(f.read_text(errors="replace") for f in sorted((s.task_dir / "tests").rglob("*.py")))
        a, b = docs.get(s.split_group, ("", ""))
        docs[s.split_group] = (a + "\n" + inst, b + "\n" + tests)
    return docs


def neardup_stats(report: NearDupReport | None, before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    """How many test (and val) groups have a near-twin in train, per Jaccard threshold, before and after clustering.

    `before` is the assignment without clusters, `after` the one used; a pair counts when its Jaccard is at least the
    threshold on either the instruction (3-grams) or the tests (5-grams)."""
    from honest_agents.tasks.dedup import NEARDUP_REPORT_THRESHOLDS

    if report is None:
        return {"enabled": False}

    def twins(assign: dict[str, str], thr: float, held: str) -> int:
        hit = set()
        for a, b, j, _ in report.pairs:
            if j >= thr and a in assign and b in assign:
                for x, y in ((a, b), (b, a)):
                    if assign[x] == held and assign[y] == "train":
                        hit.add(x)
        return len(hit)

    return {
        "enabled": True,
        "threshold": report.threshold,
        "n_groups": len(report.clusters),
        "n_groups_in_multi_clusters": report.n_clustered(),
        "n_clusters": len(set(report.clusters.values())),
        "max_pair_jaccard": round(max((p[2] for p in report.pairs), default=0.0), 4),
        "test_groups_with_train_twin_before": {str(t): twins(before, t, "test") for t in NEARDUP_REPORT_THRESHOLDS},
        "test_groups_with_train_twin_after": {str(t): twins(after, t, "test") for t in NEARDUP_REPORT_THRESHOLDS},
        "val_groups_with_train_twin_after": {str(t): twins(after, t, "val") for t in NEARDUP_REPORT_THRESHOLDS},
    }


def filter_hermetic(seeds: list[SeedTask], *, allow_legacy: bool) -> tuple[list[SeedTask], list[SeedTask]]:
    """Seeds graded by the offline pytest verifier; the rest are dropped unless legacy verifiers are allowed."""
    if allow_legacy:
        return list(seeds), []
    kept = [s for s in seeds if s.hermetic]
    dropped = [s for s in seeds if not s.hermetic]
    if dropped:
        log.warning("dropped %d/%d seeds that are not graded by the hermetic pytest verifier", len(dropped), len(seeds))
    return kept, dropped


def _read_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


# ── validation, relabelling, pairing ─────────────────────────────────────────

Pair = tuple[SeedTask, Variant]


def _pair_key(v: Variant) -> tuple[str | None, str | None]:
    return v.spec.pair_id, v.spec.perturbation


def read_relabels(path: Path) -> dict[str, list[str]]:
    """variant id -> the paths its oracle must recreate (empty list = none needed)."""
    return json.loads(path.read_text()) if path.exists() else {}


def relabel_variant(task_dir: Path, repair_paths: Sequence[str]) -> None:
    """Rewrite an infeasible `remove_input` variant in place as a near-miss control (`validate.relabel_to_control`)."""
    from honest_agents.tasks.validate import relabel_to_control

    relabel_to_control(task_dir, repair_paths=list(repair_paths))
    cfg = read_task_toml(task_dir)
    md = dict(cfg.get("metadata") or {})
    if isinstance(md.get("honest_variant"), dict):
        md["honest_variant"] = {**md["honest_variant"], "control": True, "relabelled": True}
        write_task_toml(task_dir, {**cfg, "metadata": md})


def apply_validation(
    variants: list[Pair],
    results: dict[str, Any],
    *,
    relabels: dict[str, list[str]],
    allow_oracle_tamper: bool = False,
    require_valid: bool = True,
    require_null_check: bool = True,
    min_oracle_runs: int = 3,
) -> tuple[list[Pair], dict[str, Any]]:
    """Fold validation results into the variant list; returns the kept variants and what changed.

    1. A `remove_input` variant already in `relabels`, or whose current result says `relabel_to_control`, is rewritten in
       place as a control (and recorded in `relabels`, which the caller persists). It needs validating again, as a
       control, so it is kept only once a result for its new content exists. It replaces the control the operator
       generated for the same pair.
    2. With `require_valid`: variants without a current, ok, non-leaking result are dropped; so is every variant of a pair
       whose plain feasible copy is not valid (a broken source task, or an oracle that trips the tamper check).

    Leak results are returned in `info["leaks"]` for quarantine. Pairing is enforced by the caller afterwards.
    """
    from honest_agents.tasks.validate import filter_valid, is_current

    info: dict[str, Any] = {"relabelled": [], "leaks": [], "dropped": 0}
    relabelled_pairs: set[tuple] = set()
    out: list[Pair] = []
    for s, v in variants:
        r = results.get(v.variant_id)
        current = r is not None and is_current(r, v.task_dir)
        flagged = current and getattr(r, "relabel_to_control", False)
        if v.kind == "infeasible" and v.perturbation == "remove_input" and (v.variant_id in relabels or flagged):
            if v.variant_id not in relabels:
                relabels[v.variant_id] = list((getattr(r, "debug", None) or {}).get("repair_paths") or [])
            relabel_variant(v.task_dir, relabels[v.variant_id])
            spec = HonestTaskSpec.from_task_dir(v.task_dir)
            v = Variant(v.variant_id, v.task_dir, spec, {**v.detail, "relabelled": True})
            relabelled_pairs.add(_pair_key(v))
            info["relabelled"].append(v.variant_id)
        out.append((s, v))
    out = [
        (s, v)
        for s, v in out
        if not (v.kind == "control" and not v.detail.get("relabelled") and _pair_key(v) in relabelled_pairs)
    ]
    info["leaks"] = [
        results[v.variant_id]
        for _, v in out
        if getattr(results.get(v.variant_id), "leak", False) and is_current(results[v.variant_id], v.task_dir)
    ]
    if not require_valid:
        return out, info
    valid = filter_valid(
        results,
        allow_oracle_tamper=allow_oracle_tamper,
        task_dirs={v.variant_id: v.task_dir for s, v in out if s.has_solution},
        require_null_check=require_null_check,
        min_oracle_runs=min_oracle_runs,
    )

    def ok(s: SeedTask, v: Variant) -> bool:
        return v.variant_id in valid or not s.has_solution

    plain_ok = {v.spec.pair_id for s, v in out if v.kind == "feasible" and ok(s, v)}
    kept = [(s, v) for s, v in out if ok(s, v) and v.spec.pair_id in plain_ok]
    info["dropped"] = len(variants) - len(kept)
    return kept, info


def enforce_pairing(variants: list[Pair], *, require_controls: bool = True) -> list[Pair]:
    """Drop an infeasible variant that has no control and a control (not a relabelled one) that lost its twin."""
    infeasible = {_pair_key(v) for _, v in variants if v.kind == "infeasible"}
    controls = {_pair_key(v) for _, v in variants if v.kind == "control"}
    out = []
    for s, v in variants:
        if v.kind == "infeasible" and require_controls and _pair_key(v) not in controls:
            continue
        if v.kind == "control" and not v.detail.get("relabelled") and _pair_key(v) not in infeasible:
            continue
        out.append((s, v))
    return out


# ── build ────────────────────────────────────────────────────────────────────


def build(cfg: BuildConfig) -> dict[str, Any]:
    check_benchmark_config(cfg)  # before any download: decontamination is not optional
    cfg.out.mkdir(parents=True, exist_ok=True)
    tasks_root = cfg.out / "tasks"
    for problem in heldout_problems(cfg.heldout_families):
        log.warning("held-out families %s deviate from SPEC_V3 §4.6: %s", cfg.heldout_families, problem)

    rejected: list[SeedReject] = []
    seeds = collect_seeds(cfg, rejected)
    seeds, dropped, benchmark_stats = dedup_seeds_report(cfg, seeds)
    _write_jsonl(cfg.out / "dedup.jsonl", dropped)
    seeds, not_hermetic = filter_hermetic(seeds, allow_legacy=cfg.allow_legacy_verifier)

    # Splits are fixed per seed group before any variant is built, because the phrasing pool depends on the split.
    split_path = cfg.split_seeds_file or cfg.out / SPLIT_SEEDS_FILE
    previous = json.loads(split_path.read_text()) if split_path.exists() else {}
    test_ids_path = cfg.test_ids_file or cfg.out / TEST_IDS_FILE
    frozen_test = read_test_ids(test_ids_path) if test_ids_path.exists() else set()
    if cfg.test_ids_file is not None and not test_ids_path.exists():
        log.warning(
            "--test-ids-file %s does not exist: starting a new held-out list (hash cutoffs still apply)", test_ids_path
        )
    groups = sorted({s.split_group for s in seeds})
    neardup: NearDupReport | None = None
    if cfg.neardup_threshold > 0 and seeds:
        neardup = near_duplicate_clusters(seed_documents(seeds), threshold=cfg.neardup_threshold)
    split_kw: dict[str, Any] = {
        "n_test": cfg.test_seeds,
        "n_val": cfg.val_seeds,
        "test_bp": cfg.test_bp,
        "val_bp": cfg.val_bp,
        "seed": cfg.split_seed,
        "previous": previous,
        "frozen_test": frozen_test,
    }
    if cfg.test_seeds is not None or cfg.val_seeds is not None:
        log.warning(
            "--test-seeds/--val-seeds rank the current pool: the split is only stable through %s", test_ids_path
        )
    seed_split = assign_seed_splits(groups, clusters=neardup.clusters if neardup else None, **split_kw)
    split_before = assign_seed_splits(groups, **split_kw) if neardup else seed_split
    merged = {**previous, **seed_split}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    all_test = frozen_test | {g for g, sp in merged.items() if sp == "test"}
    if all_test != frozen_test:
        log.info("test ids: %d -> %d (%s)", len(frozen_test), len(all_test), test_ids_path)
    ids_sha = (
        write_test_ids(test_ids_path, all_test)
        if all_test != frozen_test or not test_ids_path.exists()
        else ids_digest(frozen_test)
    )
    if test_ids_path.resolve() != (cfg.out / TEST_IDS_FILE).resolve():  # the output dir stays self-contained
        shutil.copyfile(test_ids_path, cfg.out / TEST_IDS_FILE)
    if not any(sp == "train" for sp in seed_split.values()):
        log.warning("no training seeds: the test/val cutoffs use the whole pool")
    n_test_groups = sum(sp == "test" for sp in seed_split.values())
    if cfg.test_seeds is None and n_test_groups < DEFAULT_TEST_SEEDS:
        log.warning(
            "only %d test seed groups (< %d, V-2): raise --limit or --test-bp; changing them later is safe, no source moves",
            n_test_groups,
            DEFAULT_TEST_SEEDS,
        )

    bank: list[tuple[str, str, str]] = []  # D' of a conflicting_constraint control: other tasks' deliverable names
    for s in seeds:
        if s.has_solution:
            split = phrasing_split(seed_split[s.split_group])
            bank += [(split, s.pair_id, n) for n in deliverable_names(TaskFacts.load(s.task_dir))]
    set_name_bank(bank)
    allow = _read_allowlist(cfg.feasible_allowlist)
    leak_dropped_before = Counter(LEAK_DROPPED)
    variants: list[Pair] = []
    skipped_no_solution = 0
    for s in seeds:
        split = phrasing_split(seed_split[s.split_group])
        if s.has_solution:
            for v in generate_variants(
                s.task_dir,
                tasks_root,
                pair_id=s.pair_id,
                operators=cfg.operators,
                seed=cfg.seed,
                split=split,
                require_controls=cfg.require_controls,
            ):
                variants.append((s, v))
        elif s.pair_id in allow:  # feasible-only pool (no oracle => no infeasible variants)
            variants += [(s, v) for v in generate_variants(s.task_dir, tasks_root, pair_id=s.pair_id, operators=[])]
        else:
            skipped_no_solution += 1
    if skipped_no_solution:
        log.warning("skipped %d seeds without oracle solution (not in --feasible-allowlist)", skipped_no_solution)

    quarantine = quarantined_variants(
        cfg.quarantine_file,
        kinds=cfg.quarantine_kinds,
        exclude_sources=() if cfg.quarantine_include_eval else DEFAULT_IGNORED_SOURCES,
    )
    n_quarantine_eval_ignored = (
        0
        if cfg.quarantine_include_eval
        else len(quarantined_variants(cfg.quarantine_file, kinds=cfg.quarantine_kinds, exclude_sources=()) - quarantine)
    )
    n_quarantined = sum(v.variant_id in quarantine for _, v in variants)
    variants = enforce_pairing(
        [(s, v) for s, v in variants if v.variant_id not in quarantine], require_controls=cfg.require_controls
    )

    validation_path = cfg.validation_jsonl or cfg.out / "validation.jsonl"
    info: dict[str, Any] = {"relabelled": [], "leaks": [], "dropped": 0}
    if cfg.validate:
        from honest_agents.tasks.validate import validate_many

        to_check = [v.task_dir for s, v in variants if s.has_solution]
        asyncio.run(validate_many(to_check, validation_path, concurrency=cfg.concurrency))
    relabels_path = cfg.out / RELABELS_FILE
    relabels = read_relabels(relabels_path)
    validating = bool(cfg.validate or cfg.validation_jsonl)
    if validating or relabels:
        from honest_agents.tasks.validate import read_results

        before = len(variants)
        results = read_results(validation_path) if validating else {}
        variants, info = apply_validation(
            variants,
            results,
            relabels=relabels,
            allow_oracle_tamper=cfg.allow_oracle_tamper,
            require_valid=validating,
            require_null_check=cfg.require_null_check,
            min_oracle_runs=cfg.min_oracle_runs,
        )
        relabels_path.write_text(json.dumps(relabels, indent=2, sort_keys=True))
        variants = enforce_pairing(variants, require_controls=cfg.require_controls)
        log.info("validation kept %d/%d variants (%d relabelled)", len(variants), before, len(info["relabelled"]))
        if info["leaks"]:
            _quarantine_leaks(info["leaks"], cfg=cfg)

    label_leaks_found = emitted_label_leaks(variants)
    if label_leaks_found:
        raise ValueError(
            f"{len(label_leaks_found)} emitted files carry label strings (L5), e.g. {label_leaks_found[:3]}"
        )
    n_leak_dropped = dict(Counter(LEAK_DROPPED) - leak_dropped_before)

    prompt = make_prompt()
    root = cfg.task_dir_root.rstrip("/") if cfg.task_dir_root else str(tasks_root.resolve())
    rows = [
        Row(
            prompt=prompt,
            task_dir=f"{root}/{v.variant_id}",
            variant_id=v.variant_id,
            pair_id=s.pair_id,
            feasible=v.feasible,
            is_control=v.is_control,
            kind=v.kind,
            family=v.spec.family or "feasible",
            perturbation=v.perturbation,
            source=s.source,
            split=seed_split[s.split_group],
            split_group=s.split_group,
        )
        for s, v in variants
    ]
    heldout = set(cfg.heldout_families)
    splits: dict[str, list[Row]] = {k: [] for k in SPLITS}
    for r in rows:
        if r.family in heldout and r.split != "test":
            continue  # held-out families are only ever seen at test time
        splits[r.split].append(r)

    check_no_test_leak((r for name, rs in splits.items() if name != "test" for r in rs), all_test)
    # Phase 0 measures a sample of the pool, so every train pair is listed here before the gate and the mixture.
    _write_jsonl(cfg.out / "train_pool.jsonl", [_row_dict(r) for r in splits["train"]])
    gate = DifficultyGate.load(cfg.difficulty_json) if cfg.difficulty_json else None
    n_difficulty_dropped = 0
    if gate is not None:  # the training split only (LV-9): test metrics stay unfiltered
        kept = [r for r in splits["train"] if gate.admits(r.pair_id, strict=cfg.difficulty_strict)]
        n_difficulty_dropped = len(splits["train"]) - len(kept)
        splits["train"] = kept
    for name in ("train", "val"):  # test keeps every variant: it is the held-out evaluation set
        splits[name] = balance_mixture(splits[name], cfg.mixture, cfg.seed, paired=cfg.pair_mixture)
    for name, rs in splits.items():
        _write_jsonl(cfg.out / f"{name}.jsonl", [_row_dict(r) for r in rs])

    test_coverage = _test_coverage(splits["test"])
    for fam, n in test_coverage["short_families"].items():
        log.warning(
            "test split has %d infeasible %s variants (< %d): report this family pooled or not at all",
            n,
            fam,
            MIN_FAMILY_TEST_ROWS,
        )
    kept_ids = {name: {r.variant_id for r in rs} for name, rs in splits.items()}
    cue = cue_metrics({name: [v for _, v in variants if v.variant_id in ids] for name, ids in kept_ids.items()})
    bad = [f"{name}: {m}" for name, ms in cue["violations"].items() for m in ms]
    for m in bad:
        log.warning("existence cue outside %s: %s", CUE_BAND, m)
    for name, ms in cue.get("watch", {}).items():
        for m in ms:
            log.warning("cue worth a look (not enforced) in %s: %s", name, m)
    if cfg.enforce_cue_band and (bad or not cue["available"]):
        raise ValueError(f"existence AUC outside {CUE_BAND} (L1): {bad or 'analysis.existence_auc is not available'}")
    manifest = {
        "test_coverage": test_coverage,
        "split_stability": {
            "mode": "count (pool-dependent)" if cfg.test_seeds is not None or cfg.val_seeds is not None else "hash",
            "salt": f"{SPLIT_SALT}|{cfg.split_seed}",
            "test_bp": cfg.test_bp,
            "val_bp": cfg.val_bp,
            "test_ids_file": str(test_ids_path),
            "test_ids_n": len(all_test),
            "test_ids_sha256": ids_sha,
            "n_test_ids_added_this_build": len(all_test - frozen_test),
            "test_ids_forced_from_file": len(frozen_test & set(groups)),
            "quarantine_eval_entries_ignored": n_quarantine_eval_ignored,
        },
        "benchmark_overlap": benchmark_stats,
        "neardup": neardup_stats(neardup, split_before, seed_split),
        "pairing": pairing_stats(splits),
        "cue_metrics": cue,
        "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
        "licenses": {s: SOURCES[s].license for s in {x.source for x in seeds} if s in SOURCES},
        "n_seeds": len(seeds),
        "n_dedup_dropped": len(dropped),
        "n_not_hermetic": len(not_hermetic),
        "n_hermetic_rejected": len(rejected),
        "n_skipped_no_solution": skipped_no_solution,
        "label_leaks": {"emitted": 0, "variants_dropped_for_a_label_string": n_leak_dropped},
        "n_quarantined": n_quarantined,
        "n_relabelled": len(info["relabelled"]),
        "n_leaks_found": len(info["leaks"]),
        "n_difficulty_dropped": n_difficulty_dropped,
        "heldout_families": sorted(heldout),
        "seed_splits": dict(Counter(seed_split.values())),
        "splits": {
            k: {
                "rows": len(rs),
                "feasible": sum(r.feasible for r in rs),
                "kinds": dict(Counter(r.kind for r in rs)),
                "families": dict(Counter(r.family for r in rs)),
                "pairs": len({r.pair_id for r in rs}),
            }
            for k, rs in splits.items()
        },
    }
    (cfg.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


CUE_BAND = (0.4, 0.6)  # L1: a family's existence cue must not separate infeasible from control outside this band


def emitted_label_leaks(variants: Iterable[Pair]) -> list[str]:
    """Every variant, the plain feasible copy included, is scanned for label strings in what ships under environment/ (L5)."""
    from honest_agents.tasks.analysis import label_leaks

    return [
        f"{v.variant_id}: {leak.path}:{leak.line}" for s, v in variants for leak in label_leaks(v.task_dir, s.task_dir)
    ]


def pairing_stats(splits: dict[str, list[Row]]) -> dict[str, Any]:
    """Per split: infeasible/control counts per family, controls without a twin, feasible rows from paired sources."""
    out: dict[str, Any] = {}
    for name, rs in splits.items():
        inf = Counter((r.family) for r in rs if r.kind == "infeasible")
        ctl = Counter((r.family) for r in rs if r.kind == "control")
        twins = {(r.pair_id, r.perturbation) for r in rs if r.kind == "infeasible"}
        paired_src = {p for p, _ in twins}
        feas = [r for r in rs if r.kind == "feasible"]
        out[name] = {
            "infeasible_per_family": dict(inf),
            "control_per_family": dict(ctl),
            "infeasible_share_per_family": {f: round(inf[f] / (inf[f] + ctl[f]), 3) for f in inf.keys() | ctl.keys()},
            "controls_without_twin": sum(r.kind == "control" and (r.pair_id, r.perturbation) not in twins for r in rs),
            "feasible_from_paired_sources": sum(r.pair_id in paired_src for r in feas),
            "feasible": len(feas),
        }
    return out


def cue_metrics(variants_by_split: dict[str, list[Any]]) -> dict[str, Any]:
    """Per split and family: the existence AUC of the forbidden/target object (L1), from `analysis.existence_auc`.

    `variants_by_split` holds the emitted `Variant`s. The import is guarded so an older `analysis` (no
    `existence_auc`) yields `available: false` instead of breaking the build."""
    try:
        from honest_agents.tasks.analysis import cue_classifiers, existence_auc, existence_rows, existence_violations
    except ImportError:
        return {"available": False, "band": list(CUE_BAND), "violations": {}}
    out: dict[str, Any] = {"available": True, "band": list(CUE_BAND), "violations": {}}
    for name, vs in variants_by_split.items():
        try:
            rows = existence_rows(vs)
            report = existence_auc(rows)
            for fam, cv in cue_classifiers(rows).items():
                report[fam]["cv"] = cv
            out[name] = report
            out["violations"][name] = existence_violations(report, low=CUE_BAND[0], high=CUE_BAND[1])
            out.setdefault("watch", {})[name] = cue_watch(report)
            # A thin family is not judged (too few rows for an AUC): list it so it never reads as "clean".
            out.setdefault("unjudged", {})[name] = sorted(
                f for f, e in report.items() if min(e.get("n_infeasible", 0), e.get("n_control", 0)) < 10
            )
        except Exception as e:  # noqa: BLE001 — a metric must never break the build (enforcement is explicit)
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def cue_watch(report: dict[str, Any], *, min_per_class: int = 20, z: float = 2.0) -> list[str]:
    """Features outside the enforced set that still separate infeasible from control: AUC outside the band and more than
    `z` standard errors from 0.5. They are listed in the manifest (never fatal), because a cue the enforced list does not
    name (for example `mentions_in_instruction`) is the same shortcut in another form."""
    out = []
    for fam, entry in sorted(report.items()):
        for name, f in sorted((entry.get("features") or {}).items()):
            auc, se = f.get("auc"), f.get("auc_se")
            if auc is None or se is None or min(f["n_infeasible"], f["n_control"]) < min_per_class:
                continue
            if not CUE_BAND[0] <= auc <= CUE_BAND[1] and abs(auc - 0.5) > z * se:
                out.append(f"{fam}: {name} AUC {auc:.2f} (+-{se:.2f}, n {f['n_infeasible']}/{f['n_control']})")
    return out


def _test_coverage(rows: list[Row]) -> dict[str, Any]:
    """V-2: infeasible variants per family in the test split, and the families below `MIN_FAMILY_TEST_ROWS`."""
    per = Counter(r.family for r in rows if r.kind == "infeasible")
    return {
        "infeasible_per_family": dict(per),
        "min_family_rows": MIN_FAMILY_TEST_ROWS,
        "short_families": {f: per.get(f, 0) for f in OPERATORS if per.get(f, 0) < MIN_FAMILY_TEST_ROWS},
        "pairs": len({r.pair_id for r in rows}),
    }


def _quarantine_leaks(leaks: list[Any], *, cfg: BuildConfig) -> None:
    """Append validation leaks to the quarantine list (once per variant) when a list is configured."""
    known = {e.variant_id for e in read_quarantine(cfg.quarantine_file)}
    entries = [
        QuarantineEntry(
            variant_id=r.variant_id,
            task_dir=r.task_dir,
            pair_id=r.pair_id,
            perturbation=r.perturbation,
            family=r.perturbation,
            kind=QUARANTINE_LEAK,
            source="validation",
            detail=str(getattr(r, "reason", "")),
        )
        for r in leaks
        if r.variant_id not in known
    ]
    if entries and append_quarantine(entries, cfg.quarantine_file) is None:
        log.warning("%d leaking variants found but no quarantine file is configured", len(entries))


def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _row_dict(r: Row) -> dict[str, Any]:
    d = asdict(r)
    d.pop("split_group")
    assert set(d) == set(COLUMNS), set(d) ^ set(COLUMNS)
    return d


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def parse_args(argv: list[str] | None = None) -> BuildConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=["endless-terminals"], help="+ seta-env, tmax-15k, local")
    ap.add_argument("--raw-dir", type=Path, help="download cache (default <out>/raw)")
    ap.add_argument("--seeds-dir", type=Path, help="Harbor task dirs for --sources local")
    ap.add_argument("--limit", type=int, help="max seed tasks per source (seeded sample)")
    ap.add_argument("--no-download", action="store_true", help="only use tasks already in --raw-dir")
    ap.add_argument("--operators", nargs="+", default=list(OPERATORS), choices=list(OPERATORS))
    ap.add_argument(
        "--heldout-families",
        default=",".join(DEFAULT_HELDOUT_FAMILIES),
        help="comma-separated families kept out of train/val (SPEC_V3: high_risk + one environment mechanism)",
    )
    ap.add_argument(
        "--test-bp",
        type=int,
        default=DEFAULT_TEST_BP,
        help="test share of the split hash space, in 1/10000 (fixed cutoff: independent of the pool; default 2200)",
    )
    ap.add_argument("--val-bp", type=int, default=DEFAULT_VAL_BP, help="validation share, in 1/10000 (default 400)")
    ap.add_argument(
        "--test-seeds",
        type=int,
        help="LEGACY, dev pools only: hold out exactly this many source tasks by ranking the current pool",
    )
    ap.add_argument("--val-seeds", type=int, help="LEGACY: validation source tasks, ranked within the pool")
    ap.add_argument(
        "--split-seed", type=int, default=0, help="salt of the split hash (never change it after a release)"
    )
    ap.add_argument(
        "--test-ids-file",
        type=Path,
        help="held-out ids with a sha256 line (default <out>/test_ids.txt); read if present, extended, never shrunk",
    )
    ap.add_argument(
        "--neardup-threshold",
        type=float,
        default=DEFAULT_NEARDUP_THRESHOLD,
        help="word 3-gram Jaccard above which two seeds' instructions form one split unit (0 = off)",
    )
    ap.add_argument(
        "--i-know-no-benchmark",
        action="store_true",
        help="build without --benchmark-dir (tests only: the data would not be decontaminated)",
    )
    ap.add_argument(
        "--allow-disabled-sources", action="store_true", help="allow the seta-env / tmax-15k sources (leak audit L4)"
    )
    ap.add_argument(
        "--no-pair-mixture", action="store_true", help="draw infeasible and control rows independently (old behaviour)"
    )
    ap.add_argument(
        "--quarantine-include-eval",
        action="store_true",
        help="honour quarantine entries found while evaluating (source=eval): they change test.jsonl",
    )
    ap.add_argument(
        "--enforce-cue-band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail the build when a family's existence AUC leaves [0.4, 0.6] (L1; default on, --no-enforce-cue-band for dev)",
    )
    ap.add_argument(
        "--mixture",
        default=",".join(str(x) for x in DEFAULT_MIXTURE),
        help="plain feasible,control,infeasible shares of train and val (default 0.55,0.20,0.25); test is kept whole",
    )
    ap.add_argument("--benchmark-dir", type=Path, action="append", default=[], help="eval benchmark task dirs")
    ap.add_argument("--dedup-threshold", type=float, default=0.5)
    ap.add_argument("--feasible-allowlist", type=Path, help="pair_ids of solution-less tasks to keep (pass@k)")
    ap.add_argument("--validate", action="store_true", help="oracle-validate variants (needs Docker)")
    ap.add_argument("--validation-jsonl", type=Path, help="filter with an existing validation.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--allow-oracle-tamper", action="store_true")
    ap.add_argument(
        "--allow-weak-validation",
        action="store_true",
        help="accept validation results without the null check or with fewer than 3 oracle runs (never for training)",
    )
    ap.add_argument("--task-dir-root", help="prefix for task_dir in rows (where tasks/ lives on the training node)")
    ap.add_argument(
        "--quarantine-file", type=Path, help="variants listed here (kind leak) are excluded; leaks found are appended"
    )
    ap.add_argument(
        "--split-seeds-file",
        "--split-seeds",
        type=Path,
        help="seed -> split assignment to keep across builds (default <out>/split_seeds.json); reuse it when "
        "building into a fresh --out so test seeds stay out of training",
    )
    ap.add_argument(
        "--difficulty-json", type=Path, help="Phase-0 difficulty gate (tasks.difficulty); filters train only"
    )
    ap.add_argument(
        "--difficulty-strict",
        action="store_true",
        help="with --difficulty-json: keep only source tasks the Phase-0 run measured and kept (drop unmeasured ones)",
    )
    ap.add_argument(
        "--allow-legacy-verifier", action="store_true", help="keep seeds that are not hermetic (eval sets, tests)"
    )
    ap.add_argument(
        "--allow-missing-controls", action="store_true", help="keep infeasible variants that have no control"
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return BuildConfig(
        out=a.out,
        sources=a.sources,
        raw_dir=a.raw_dir,
        seeds_dir=a.seeds_dir,
        limit=a.limit,
        download=not a.no_download,
        operators=a.operators,
        heldout_families=[f for f in a.heldout_families.split(",") if f],
        test_bp=a.test_bp,
        val_bp=a.val_bp,
        test_seeds=a.test_seeds,
        val_seeds=a.val_seeds,
        split_seed=a.split_seed,
        test_ids_file=a.test_ids_file,
        neardup_threshold=a.neardup_threshold,
        i_know_no_benchmark=a.i_know_no_benchmark,
        allow_disabled_sources=a.allow_disabled_sources,
        pair_mixture=not a.no_pair_mixture,
        quarantine_include_eval=a.quarantine_include_eval,
        enforce_cue_band=a.enforce_cue_band,
        mixture=tuple(float(x) for x in a.mixture.split(",")),  # type: ignore[arg-type]
        benchmark_dirs=a.benchmark_dir,
        dedup_threshold=a.dedup_threshold,
        feasible_allowlist=a.feasible_allowlist,
        validate=a.validate,
        validation_jsonl=a.validation_jsonl,
        concurrency=a.concurrency,
        allow_oracle_tamper=a.allow_oracle_tamper,
        task_dir_root=a.task_dir_root,
        quarantine_file=a.quarantine_file,
        difficulty_json=a.difficulty_json,
        difficulty_strict=a.difficulty_strict,
        split_seeds_file=a.split_seeds_file,
        allow_legacy_verifier=a.allow_legacy_verifier,
        require_controls=not a.allow_missing_controls,
        require_null_check=not a.allow_weak_validation,
        min_oracle_runs=0 if a.allow_weak_validation else 3,
        seed=a.seed,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = build(parse_args(argv))
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
