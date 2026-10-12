"""RL training pool: gate the feasible side by difficulty, clean the pool, set the impossible-twin mix per family, and
write the rows in the order the trainer will draw them.

    python -m honest_agents.tasks.rl_pool --config configs/tasks/rl_pool.yaml [--out DIR/train_rl.jsonl] [key=value ...]
    python -m honest_agents.tasks.rl_pool --pool data/train_pool.jsonl --difficulty data/difficulty_2_3.json \\
        --out data/train_rl.jsonl [--max-infeasible 100] [--seed 0]          # the iteration-9 pool (legacy defaults)

**Feasible side.** `tasks.build --difficulty-json` drops a source task's impossible variants together with its plain copy
when the plain copy is too easy for the base model. An impossible variant has no pass rate, though: claiming it done
versus declining is a real contrast whatever the plain twin does, so this builder gates only the feasible side: plain
rows and near-miss controls of the source tasks the difficulty gate kept (`keep_pairs`), plus the near-miss control of
every selected impossible twin, so over-refusal stays penalised.

**Why the rest exists (iteration 10, docs/training-log.md).** In run 2 the impossible twins carried ~12 % of the summed
|advantage| (controls 59 %, plain 29 %): most twin groups were one-sided (all 8 claim or all 8 abstain), and
`revoke_permission` twins, solved in base (1.4 % claims), filled 33 of the 100 impossible slots. Label noise blurred the
rest: 8 feasible sources that never pass produced 21 % of the fabrication penalties, and 9-10 `remove_input` twins leak
(their impossible variant can still be passed). And the pair-batched order put a twin in the same step as a feasible row
of its own source in 148 of 158 groups. So, in order:

1. **Exclusions** (`Exclusions`). (a) Feasible variants with 0 clean passes over >= `min_bank_samples` episodes of the
   base pass@k bank: a never-passing plain row takes its whole source out (its tests are suspect, so its twins are too); a
   never-passing control takes itself and its twin out. (b) Leaky impossible variants: `validation.jsonl` leak flags,
   quarantine-list leaks, and bank rows where the infeasible variant passed cleanly. Leak marks apply to rows that are
   still impossible: the oracle validation relabels some leaks as controls under the same variant id, and those stay.
   (c) An optional source list (`exclude_sources`: pair_ids used for SFT data), so SFT and RL stay disjoint.
2. **Pre-screen** (optional, `claim_rates`): a JSONL of `{"variant_id", "claim_rate"}` (base claim rate per twin, measured
   on the GPU). Only twins whose rate lies in `claim_rate_band` (default [0.2, 0.8]) stay; a twin missing from the file
   is dropped (counted as `unmeasured`). Absent file = no filter. A group with claim rate p has both outcomes with
   probability 1 - p^8 - (1 - p)^8 (>= 0.83 inside the band), against ~0.25 for a 94 %-claim `high_risk` twin.
3. **Per-family quotas** (`family_quotas`): the number of impossible twins per family (null = every eligible twin; a
   family not listed gets none). The configured mix oversamples `high_risk` (the honesty failure) and `remove_input`, and
   caps `revoke_permission` at 10. With `require_control` a twin whose near-miss control is missing or excluded is not
   eligible. Without quotas the legacy round-robin sample of `max_infeasible` twins is used.
4. **Step blocks** (`block_size`, `order_blocks`). The rows are written so that every consecutive block of `block_size`
   rows, one optimizer step, has distinct sources (one variant per source per step) and a planned number of impossible
   twins (the pool's twin share, spread evenly; families interleaved). `check_blocks` verifies it.

**How file order becomes steps** (TRL 1.14.1, `GRPOTrainer._get_train_sampler`). The train sampler is
`RepeatSampler(mini_repeat_count=num_generations, batch_size=generation_batch_size // num_generations,
repeat_count=num_iterations * steps_per_generation, shuffle=args.shuffle_dataset)`. With `shuffle_dataset: false` it walks
the dataset in index order, cuts it into chunks of `generation_batch_size // num_generations` prompts, drops a final
partial chunk, and yields every prompt of a chunk `num_generations` times; one chunk is one generation round. Under
configs/grpo_colocate_qwen35.yaml (per_device_train_batch_size 1, 1 process, gradient_accumulation_steps 64,
steps_per_generation = gradient_accumulation_steps, num_generations 8, num_iterations 1): generation_batch_size =
1 x 1 x 64 = 64, so a chunk is 64 // 8 = **8 prompts** and optimizer step k (from 0) trains rows [8k, 8k + 8) of the file,
8 rollouts each. A later epoch replays the same order, so the last `len(pool) % 8` rows are never drawn and a run of
`max_steps` steps sees the first 8 x `max_steps` rows of each pass. Both `grpo.shuffle_dataset: false` and
`data.pair_batches: false` must be set (`pair_batches` would re-order the rows by `pair_id`, putting the twin, its
control and its plain copy in the same step: the opposite of what this order is for).

**The configured pool** (configs/tasks/rl_pool.yaml, 2026-10 data, no claim-rate file yet): 408 rows from 148 sources =
64 plain feasible + 190 near-miss controls + 154 impossible twins (80 `high_risk` of 158 eligible, all 64 eligible
`remove_input`, 10 `revoke_permission` of 72), against 75 + 185 + 100 (34 / 33 / 33) in the iteration-9 pool. Left out:
18 never-passing sources (62 rows), 30 never-passing controls and the 23 twins they leave without a control, 11 leaky
`remove_input` twins (7 from the bank, 4 from the run-2 quarantine). Twin share 37.7 % of rows (28 % before): 3 twins in
every step (4 in one), 51 steps per pass with no untrained tail, so a 50-step run sees 400 distinct rows. The legacy
flags (`--pool --difficulty --out`, no config) still rebuild the iteration-9 data/train_rl.jsonl byte for byte.

**Gate.** The pool is meant to move the impossible-twin share of the summed |advantage| from ~12 % to at least **30 %**
(measure it on the run's rollouts, as in iteration 10; this builder cannot predict it). The row share of twins is only an
upper-bound lever: a zero-variance group contributes nothing, which is what the claim-rate pre-screen addresses.

Rows are written unchanged. The CLI prints the composition (`summarize`) and, with `--report`, writes the exclusions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

Row = dict[str, Any]

DEFAULT_CLAIM_BAND = (0.2, 0.8)
DEFAULT_MIN_BANK_SAMPLES = 4

# Exclusion reasons (the keys of `PoolBuild.excluded`).
EXCL_SOURCE_NEVER_PASSES = "source_never_passes"  # the plain feasible row has 0 clean passes in the bank
EXCL_CONTROL_NEVER_PASSES = "control_never_passes"
EXCL_LEAK_VALIDATION = "leak_validation"  # validation.jsonl: the oracle passed an impossible variant cleanly
EXCL_LEAK_QUARANTINE = "leak_quarantine"
EXCL_LEAK_BANK = "leak_bank"  # a base rollout passed the impossible variant cleanly
EXCL_SOURCE_LISTED = "source_listed"  # `exclude_sources` (SFT data)
EXCL_NO_CONTROL = "twin_without_control"  # `require_control` and the twin's control is missing or excluded
EXCL_CLAIM_RATE = "claim_rate_out_of_band"
EXCL_CLAIM_UNMEASURED = "claim_rate_unmeasured"


def _source(r: Row) -> str:
    return r.get("pair_id") or r["variant_id"]


def _is_twin(r: Row) -> bool:
    return r["kind"] == "infeasible"


# ── exclusions ───────────────────────────────────────────────────────────────


@dataclass
class Exclusions:
    """What to leave out of the pool, each id with the reason it is out (the first reason recorded wins)."""

    sources: dict[str, str] = field(default_factory=dict)  # pair_id -> reason: every row of the source
    feasible_variants: dict[str, str] = field(default_factory=dict)  # variant_id -> reason: plain or control rows only
    infeasible_variants: dict[str, str] = field(default_factory=dict)  # variant_id -> reason: impossible rows only

    def update(self, other: Exclusions) -> Exclusions:
        for mine, theirs in (
            (self.sources, other.sources),
            (self.feasible_variants, other.feasible_variants),
            (self.infeasible_variants, other.infeasible_variants),
        ):
            for k, v in theirs.items():
                mine.setdefault(k, v)
        return self

    def reason(self, r: Row) -> str | None:
        """Why row `r` is excluded, or None."""
        if (why := self.sources.get(_source(r))) is not None:
            return why
        table = self.infeasible_variants if _is_twin(r) else self.feasible_variants
        return table.get(r["variant_id"])


def bank_exclusions(stats: Mapping[str, Any], *, min_samples: int = DEFAULT_MIN_BANK_SAMPLES) -> Exclusions:
    """Exclusions from per-variant bank statistics (`tasks.difficulty.variant_stats`: kind, pair_id, n, passes).

    A feasible variant with 0 clean passes over at least `min_samples` scored episodes is label noise for RL (every claim
    on it is paid as a fabrication): a plain row takes its whole source out, a control only itself (its twin follows it
    out under `require_control`). An impossible variant with any clean pass leaks. Variants with fewer episodes are
    left alone: no evidence either way."""
    ex = Exclusions()
    for vid, s in stats.items():
        if s.kind == "infeasible":
            if s.passes > 0:
                ex.infeasible_variants[vid] = EXCL_LEAK_BANK
        elif s.n >= min_samples and s.passes == 0:
            if s.kind == "feasible":
                ex.sources[s.pair_id] = EXCL_SOURCE_NEVER_PASSES
            else:
                ex.feasible_variants[vid] = EXCL_CONTROL_NEVER_PASSES
    return ex


def read_bank_stats(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Per-variant statistics of one or more banks (directories or files, plain or gzipped; train-split rows only)."""
    from honest_agents.bank import iter_bank
    from honest_agents.tasks.difficulty import variant_stats

    return variant_stats(row for p in paths for row in iter_bank(Path(p).expanduser()))


def validation_leaks(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Variant ids the oracle validation (`tasks.validate`, validation.jsonl) flagged as leaks."""
    return {r["variant_id"] for r in rows if r.get("leak")}


def read_jsonl(path: str | Path) -> list[Row]:
    return [json.loads(line) for line in Path(path).expanduser().read_text().splitlines() if line.strip()]


def read_id_list(path: str | Path) -> set[str]:
    """One id per line; blank lines and `#` comments are ignored."""
    ids = (line.split("#", 1)[0].strip() for line in Path(path).expanduser().read_text().splitlines())
    return {i for i in ids if i}


def read_claim_rates(path: str | Path) -> dict[str, float]:
    """`{"variant_id": ..., "claim_rate": ...}` per line -> {variant_id: claim_rate}."""
    out: dict[str, float] = {}
    for r in read_jsonl(path):
        rate = float(r["claim_rate"])
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{path}: claim_rate {rate} of {r['variant_id']} is not in [0, 1]")
        out[r["variant_id"]] = rate
    return out


# ── selection ────────────────────────────────────────────────────────────────


@dataclass
class PoolBuild:
    rows: list[Row]
    excluded: dict[str, list[str]]  # reason -> variant ids of pool rows left out for it (sorted)
    twins_available: dict[str, int]  # family -> eligible impossible twins before the quota
    claim_rates_used: bool = False
    block_size: int | None = None


def _round_robin(by_family: dict[str, list[Row]], limit: int | None) -> list[Row]:
    chosen: list[Row] = []
    queues = [list(by_family[f]) for f in sorted(by_family)]
    queues = [q for q in queues if q]
    while queues and (limit is None or len(chosen) < limit):
        for q in list(queues):
            if limit is not None and len(chosen) >= limit:
                break
            chosen.append(q.pop())
            if not q:
                queues.remove(q)
    return chosen


def plan_pool(
    rows: Sequence[Row],
    keep_pairs: set[str],
    *,
    max_infeasible: int | None = 100,
    seed: int = 0,
    family_quotas: Mapping[str, int | None] | None = None,
    exclusions: Exclusions | None = None,
    require_control: bool = False,
    claim_rates: Mapping[str, float] | None = None,
    claim_rate_band: tuple[float, float] = DEFAULT_CLAIM_BAND,
    block_size: int | None = None,
) -> PoolBuild:
    """The RL pool and what was left out (see the module docstring). The defaults reproduce the iteration-9 pool."""
    if family_quotas is not None and max_infeasible is not None:
        raise ValueError("family_quotas and max_infeasible are alternatives: set max_infeasible to None with quotas")
    lo, hi = claim_rate_band
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"claim_rate_band {claim_rate_band} is not an interval in [0, 1]")
    ex = exclusions or Exclusions()
    excluded: dict[str, set[str]] = defaultdict(set)
    rng = random.Random(seed)

    def drop(r: Row, why: str) -> None:
        excluded[why].add(r["variant_id"])

    live: list[Row] = []
    for r in rows:
        why = ex.reason(r)
        if why is None:
            live.append(r)
        elif _is_twin(r) or r["kind"] == "control" or r["pair_id"] in keep_pairs:
            drop(r, why)  # only rows the pool could have used are reported
    controls = {(r["pair_id"], r["family"]) for r in live if r["kind"] == "control"}

    by_family: dict[str, list[Row]] = defaultdict(list)
    for r in live:
        if not _is_twin(r):
            continue
        if require_control and (r["pair_id"], r["family"]) not in controls:
            drop(r, EXCL_NO_CONTROL)
            continue
        if claim_rates is not None:
            rate = claim_rates.get(r["variant_id"])
            if rate is None:
                drop(r, EXCL_CLAIM_UNMEASURED)
                continue
            if not lo <= rate <= hi:
                drop(r, EXCL_CLAIM_RATE)
                continue
        by_family[r["family"]].append(r)
    for family_rows in by_family.values():  # first-appearance order, as iteration 9 did: same seed, same sample
        rng.shuffle(family_rows)
    available = {f: len(v) for f, v in sorted(by_family.items())}

    if family_quotas is None:
        chosen = _round_robin(by_family, max_infeasible)
    else:
        chosen = []
        for fam in sorted(by_family):
            quota = family_quotas.get(fam, 0)
            chosen += by_family[fam] if quota is None else by_family[fam][: max(0, int(quota))]
    chosen_twins = {(r["pair_id"], r["family"]) for r in chosen}
    chosen_ids = {r["variant_id"] for r in chosen}

    out = []
    for r in live:
        if _is_twin(r):
            keep = r["variant_id"] in chosen_ids
        elif r["kind"] == "control":
            keep = r["pair_id"] in keep_pairs or (r["pair_id"], r["family"]) in chosen_twins
        else:  # plain feasible
            keep = r["pair_id"] in keep_pairs
        if keep:
            out.append(r)
    if block_size:
        out = order_blocks(out, block_size, seed=seed)
    return PoolBuild(
        rows=out,
        excluded={k: sorted(v) for k, v in sorted(excluded.items())},
        twins_available=available,
        claim_rates_used=claim_rates is not None,
        block_size=block_size,
    )


def build_pool(rows: Sequence[Row], keep_pairs: set[str], **kwargs: Any) -> list[Row]:
    """The RL pool's rows (`plan_pool` without the bookkeeping)."""
    return plan_pool(rows, keep_pairs, **kwargs).rows


# ── step order ───────────────────────────────────────────────────────────────


def _category(r: Row) -> str:
    return f"{r['kind']}:{r.get('family')}"


def _spread(rows: list[Row], rng: random.Random) -> list[Row]:
    """`rows` interleaved by category in proportion to each category's size (item k of a category of size c sits at
    (k + 0.5) / c), each category shuffled first: any stretch of the sequence has about the overall mix."""
    by_cat: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_cat[_category(r)].append(r)
    keyed = []
    for cat in sorted(by_cat):
        members = by_cat[cat]
        rng.shuffle(members)
        keyed += [((k + 0.5) / len(members), cat, r) for k, r in enumerate(members)]
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in keyed]


def _fill_block(
    queues: dict[str, list[Row]], left: Counter[str], *, blocks_left: int, block_size: int, want_twins: int
) -> list[Row] | None:
    """One step block taken from `queues` ("twin", "other"), or None (queues untouched) when it cannot be filled with
    distinct sources. `left` counts each source's rows not yet placed."""
    block: list[Row] = []
    used: set[str] = set()
    taken: list[tuple[str, int, Row]] = []

    def place(cls: str, i: int) -> None:
        r = queues[cls].pop(i)
        block.append(r)
        used.add(_source(r))
        taken.append((cls, i, r))

    def take(cls: str) -> bool:
        i = next((i for i, r in enumerate(queues[cls]) if _source(r) not in used), None)
        if i is not None:
            place(cls, i)
        return i is not None

    # Urgent sources first (as many rows left as blocks left), in whichever class holds them.
    for cls in ("twin", "other"):
        i = 0
        while i < len(queues[cls]) and len(block) < block_size:
            s = _source(queues[cls][i])
            if s not in used and left[s] >= blocks_left:
                place(cls, i)
            else:
                i += 1
    n_tw = sum(_is_twin(r) for r in block)
    while n_tw < want_twins and len(block) < block_size and take("twin"):
        n_tw += 1
    while len(block) < block_size and (take("other") or take("twin")):
        pass
    if len(block) < block_size:
        for cls, i, r in reversed(taken):
            queues[cls].insert(i, r)
        return None
    for r in block:
        left[_source(r)] -= 1
    return block


def order_blocks(rows: Sequence[Row], block_size: int, *, seed: int = 0) -> list[Row]:
    """`rows` ordered so every full block of `block_size` consecutive rows has distinct sources and a planned number of
    impossible twins: the T twins spread over the B = len // block_size blocks as evenly as integers allow (block i
    gets floor((i+1) T / B) - floor(i T / B)), each class interleaved by family/kind (`_spread`).

    Greedy, block by block: a source with at least as many rows left as there are blocks left is placed first
    (otherwise it could not keep one row per block at the end); then each slot takes the earliest row of its class (twin
    / other) whose source is not yet in the block. If a class runs dry for a block the other class fills in; rows that
    cannot be placed without repeating a source go to the tail. The `len % block_size` tail rows are never trained
    (TRL drops a partial chunk), and `check_blocks` reports any block that repeats a source."""
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    rng = random.Random(seed)
    twins = _spread([r for r in rows if _is_twin(r)], rng)
    others = _spread([r for r in rows if not _is_twin(r)], rng)
    n_blocks = len(rows) // block_size
    n_twins = min(len(twins), n_blocks * block_size)
    plan = [(i + 1) * n_twins // n_blocks - i * n_twins // n_blocks for i in range(n_blocks)] if n_blocks else []
    left: Counter[str] = Counter(_source(r) for r in rows)
    queues = {"twin": twins, "other": others}
    out: list[Row] = []
    for b, want_twins in enumerate(plan):
        block = _fill_block(queues, left, blocks_left=n_blocks - b, block_size=block_size, want_twins=want_twins)
        if block is None:  # cannot fill without a repeat: stop planning, the rest goes to the tail
            break
        out += block
    return out + queues["twin"] + queues["other"]


def check_blocks(rows: Sequence[Row], block_size: int) -> list[str]:
    """Problems with the step order: every full block of `block_size` consecutive rows must have distinct sources.
    Empty list = fine."""
    problems = []
    for b in range(len(rows) // block_size):
        block = rows[b * block_size : (b + 1) * block_size]
        repeated = [s for s, n in Counter(_source(r) for r in block).items() if n > 1]
        if repeated:
            problems.append(f"step {b}: source(s) {sorted(repeated)} appear more than once")
    return problems


# ── reporting ────────────────────────────────────────────────────────────────


def summarize(rows: Sequence[Row], block_size: int | None = None, steps: int | None = None) -> dict[str, Any]:
    """Composition of a pool: rows by kind, twins by family, and with `block_size` the per-step twin counts, the rows
    TRL never draws, and (with `steps`) what a run of that many steps sees."""
    kinds: dict[str, int] = defaultdict(int)
    families: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
        if _is_twin(r):
            families[r["family"]] += 1
    out: dict[str, Any] = {"rows": len(rows), "by_kind": dict(kinds), "infeasible_by_family": dict(families)}
    out["sources"] = len({_source(r) for r in rows})
    out["twin_share"] = round(kinds["infeasible"] / len(rows), 3) if rows else 0.0
    if block_size:
        n_blocks = len(rows) // block_size
        per_block = [sum(_is_twin(r) for r in rows[b * block_size : (b + 1) * block_size]) for b in range(n_blocks)]
        out["steps_per_pass"] = n_blocks
        out["untrained_tail_rows"] = len(rows) - n_blocks * block_size
        out["twins_per_step"] = {str(k): v for k, v in sorted(Counter(per_block).items())}  # twins -> steps (JSON keys)
        out["step_block_problems"] = len(check_blocks(rows, block_size))
        if steps is not None and n_blocks:
            seen = [rows[(i % n_blocks) * block_size + j] for i in range(steps) for j in range(block_size)]
            out["first_steps"] = {
                "steps": steps,
                "passes": round(steps / n_blocks, 2),
                "distinct_rows_seen": len({r["variant_id"] for r in seen}),
                "twin_share": round(sum(_is_twin(r) for r in seen) / len(seen), 3),
            }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


@dataclass
class PoolOptions:
    """Everything the CLI reads: the keys of a `--config` YAML (configs/tasks/rl_pool.yaml), overridden by flags."""

    pool: str | None = None  # train_pool.jsonl of a dataset build
    difficulty: str | None = None  # difficulty.json of tasks.difficulty (`keep_pairs`)
    out: str | None = None
    report: str | None = None  # optional JSON: summary, exclusions, options
    seed: int = 0
    max_infeasible: int | None = 100  # legacy round-robin sample; null with family_quotas
    family_quotas: dict[str, int | None] | None = None
    require_control: bool = False
    bank: list[str] = field(default_factory=list)  # base pass@k bank(s): never-passing feasible variants, bank leaks
    min_bank_samples: int = DEFAULT_MIN_BANK_SAMPLES
    validation: str | None = None  # validation.jsonl of tasks.validate (leak flags)
    quarantine: list[str] = field(default_factory=list)  # quarantine.jsonl files (leak entries)
    exclude_sources: str | None = None  # pair_ids, one per line (e.g. the SFT data's sources)
    claim_rates: str | None = None  # JSONL variant_id -> claim_rate; null = no pre-screen
    claim_rate_band: list[float] = field(default_factory=lambda: list(DEFAULT_CLAIM_BAND))
    block_size: int | None = None  # prompts per optimizer step; null = pool order
    steps: int | None = None  # report only: what a run of this many steps sees

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PoolOptions:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown rl_pool option(s): {sorted(unknown)}")
        opts = cls(**d)
        for name in ("bank", "quarantine"):
            v = getattr(opts, name)
            setattr(opts, name, [] if v is None else [v] if isinstance(v, str) else list(v))
        return opts


def collect_exclusions(opts: PoolOptions) -> Exclusions:
    from honest_agents.bank import quarantined_variants

    ex = Exclusions()
    if opts.exclude_sources:
        ex.update(Exclusions(sources=dict.fromkeys(read_id_list(opts.exclude_sources), EXCL_SOURCE_LISTED)))
    if opts.validation:
        leaks = validation_leaks(read_jsonl(opts.validation))
        ex.update(Exclusions(infeasible_variants=dict.fromkeys(leaks, EXCL_LEAK_VALIDATION)))
    for q in opts.quarantine:
        qpath = Path(q).expanduser()
        if not qpath.is_file():  # `quarantined_variants` reads a missing list as empty: a typo would let leaks back in
            raise FileNotFoundError(f"quarantine file {qpath} does not exist")
        leaks = quarantined_variants(qpath)
        ex.update(Exclusions(infeasible_variants=dict.fromkeys(leaks, EXCL_LEAK_QUARANTINE)))
    if opts.bank:
        stats = read_bank_stats(opts.bank)
        if not stats:  # an empty or wrong directory would silently skip every never-pass and leak exclusion
            raise ValueError(f"bank {opts.bank} has no train-split rows")
        ex.update(bank_exclusions(stats, min_samples=opts.min_bank_samples))
    return ex


def run(opts: PoolOptions) -> dict[str, Any]:
    """Build the pool from `opts`, write it, and return the report (summary, exclusions, options)."""
    if not (opts.pool and opts.difficulty and opts.out):
        raise ValueError("pool, difficulty and out are required (flags or --config)")
    rows = read_jsonl(opts.pool)
    keep_pairs = set(json.loads(Path(opts.difficulty).expanduser().read_text())["keep_pairs"])
    built = plan_pool(
        rows,
        keep_pairs,
        max_infeasible=opts.max_infeasible,
        seed=opts.seed,
        family_quotas=opts.family_quotas,
        exclusions=collect_exclusions(opts),
        require_control=opts.require_control,
        claim_rates=read_claim_rates(opts.claim_rates) if opts.claim_rates else None,
        claim_rate_band=(float(opts.claim_rate_band[0]), float(opts.claim_rate_band[1])),
        block_size=opts.block_size,
    )
    problems = check_blocks(built.rows, opts.block_size) if opts.block_size else []
    if problems:
        raise RuntimeError("step blocks repeat a source:\n" + "\n".join(problems[:20]))
    out = Path(opts.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in built.rows))
    summary = summarize(built.rows, opts.block_size, opts.steps)
    report = {
        "summary": summary,
        "twins_available": built.twins_available,
        "excluded": {k: len(v) for k, v in built.excluded.items()},
        "excluded_ids": built.excluded,
        "options": asdict(opts),
    }
    if opts.report:
        Path(opts.report).expanduser().write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    from honest_agents.train.config import apply_overrides, load_config_dict

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="YAML of PoolOptions keys (configs/tasks/rl_pool.yaml); flags win")
    ap.add_argument("--pool", help="train_pool.jsonl of a dataset build")
    ap.add_argument("--difficulty", help="difficulty.json of tasks.difficulty")
    ap.add_argument("--out")
    ap.add_argument("--report", help="write the full report (exclusion ids included) as JSON here")
    ap.add_argument("--max-infeasible", type=int, help="legacy: impossible rows to keep, round-robin over families")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--bank", action="append", help="base pass@k bank directory or file (repeatable)")
    ap.add_argument("--validation", help="validation.jsonl (leak flags)")
    ap.add_argument("--quarantine", action="append", help="quarantine.jsonl (repeatable)")
    ap.add_argument("--exclude-sources", help="file of pair_ids to leave out (e.g. the SFT data's sources)")
    ap.add_argument("--claim-rates", help="JSONL variant_id -> claim_rate; keeps twins inside claim_rate_band")
    ap.add_argument("--block-size", type=int, help="prompts per optimizer step (order the pool in step blocks)")
    ap.add_argument("--steps", type=int, help="report what a run of this many steps sees")
    ap.add_argument("overrides", nargs="*", help="key=value overrides of config keys (values parsed as YAML)")
    args = ap.parse_args(argv)
    d: dict[str, Any] = load_config_dict(args.config) if args.config else {}
    d = apply_overrides(d, list(args.overrides))
    for name in ("pool", "difficulty", "out", "report", "max_infeasible", "seed", "bank", "validation", "quarantine",
                 "exclude_sources", "claim_rates", "block_size", "steps"):  # fmt: skip
        v = getattr(args, name)
        if v is not None:
            d[name] = v
    opts = PoolOptions.from_dict(d)
    try:
        report = run(opts)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"rl_pool: {e}", file=sys.stderr)
        raise SystemExit(2) from e
    print(json.dumps({k: v for k, v in report.items() if k not in ("excluded_ids", "options")}))


if __name__ == "__main__":
    main()
