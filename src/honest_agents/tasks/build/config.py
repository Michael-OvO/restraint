"""Dataset-build configuration, the `Row` schema and the constants every stage shares."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from honest_agents.bank import (
    QUARANTINE_LEAK,
)
from honest_agents.tasks.perturb import (
    DEFAULT_OPERATORS,
    OPERATORS,
)

SPLITS = ("train", "val", "test")
KINDS = ("feasible", "control", "infeasible")  # the mixture strata, in the order of `DEFAULT_MIXTURE`
DEFAULT_MIXTURE = (0.55, 0.20, 0.25)  # docs/design.md section 7
# Test-only families: `corrupt_input` is the clean transfer test (a mechanism no training family has, twin and control
# differ only in the environment); `conflicting_constraint` has a lexical cue that cannot be removed (leak verifier
# decision) and `no_network` too few pairs to train on.
DEFAULT_HELDOUT_FAMILIES = ("corrupt_input", "conflicting_constraint", "no_network")
# Held-out families whose cue gate is ENFORCED like a training family's: a transfer test with a cue measures the cue.
# The others (conflicting_constraint, no_network) are only reported when held out.
ENFORCED_HELDOUT_FAMILIES = ("corrupt_input",)
MIN_TRAIN_PAIRS = 40  # a training family with fewer infeasible/control pairs in train is warned about in the manifest
DEFAULT_TEST_SEEDS = 300  # At least 300 test pairs, so that every reported family has at least MIN_FAMILY_TEST_ROWS
MIN_FAMILY_TEST_ROWS = 50  # Infeasible variants per reported family in the test split (fewer: pool or drop the family)
DEFAULT_TEST_BP = (
    2200  # test share of the hash space in basis points: about 560 of the full 2,482-seed pool, 330 of 1,500
)
# Validation share, the same as the test share. On the full 2,482-seed Endless Terminals pool the old 400 gave 97 groups and
# 16/17/7 infeasible rows for revoke_permission/high_risk/remove_input after the mixture (a probe on that is noise); 1200
# gave 50/50/22; 2200 gives 536 groups and 88/89/42, the first cutoff that reaches 40 for remove_input (7% of seeds yield
# one), while train keeps 1,382 groups and 230/231/104 pairs (above MIN_TRAIN_PAIRS). The salt and the cutoffs freeze at
# the first training run (docs/data.md section 6); raising VAL_BP later only moves train sources into val.
DEFAULT_VAL_BP = 2200
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
)  # docs/design.md section 3: exactly these


@dataclass
class BuildConfig:
    out: Path
    sources: list[str] = field(default_factory=lambda: ["endless-terminals"])
    raw_dir: Path | None = None
    seeds_dir: Path | None = None  # for the "local" source: existing Harbor task dirs
    limit: int | None = None
    download: bool = True
    operators: list[str] = field(default_factory=lambda: list(DEFAULT_OPERATORS))
    heldout_families: list[str] = field(default_factory=lambda: list(DEFAULT_HELDOUT_FAMILIES))
    test_bp: int = DEFAULT_TEST_BP  # fixed hash cutoffs (basis points of [0, 10000)): pool-independent
    val_bp: int = DEFAULT_VAL_BP
    test_seeds: int | None = None  # legacy: rank the pool and hold out exactly this many (pool-dependent; dev only)
    val_seeds: int | None = None
    split_seed: int = 0  # salt of the split hash; never changes with --seed or the pool
    test_ids_file: Path | None = None  # held-out ids to keep (default <out>/test_ids.txt)
    neardup_threshold: float = DEFAULT_NEARDUP_THRESHOLD  # word 3-gram Jaccard of instructions; 0 = no clustering
    i_know_no_benchmark: bool = False  # build without --benchmark-dir (tests only, never for training data)
    allow_disabled_sources: bool = False  # SETA / tmax (see sources.SOURCES)
    allow_unknown_license: bool = False  # a source whose data licence is not stated (tmax-15k)
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
    # docs/design.md section 7: the oracle must pass 3 of 3 and the null check must have shown the tests fail on the untouched
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


def heldout_problems(families: Sequence[str], operators: Sequence[str] = OPERATORS) -> list[str]:
    """Ways a held-out family list cannot work (empty = fine): a name no operator makes, or nothing left to train on."""
    problems = []
    unknown = [f for f in families if f not in OPERATORS]
    if unknown:
        problems.append(f"unknown families {unknown}")
    if families and not [o for o in operators if o not in families]:
        problems.append("every family is held out: there is nothing to train on")
    return problems


def system_prompt() -> str:
    from honest_agents.harness import SYSTEM_PROMPT

    return SYSTEM_PROMPT


def make_prompt() -> list[dict[str, str]]:
    """System prompt + empty user turn: TRL appends `reset()`'s return (the instruction) to the last message."""
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": ""}]
