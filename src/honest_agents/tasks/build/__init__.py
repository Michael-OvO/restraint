"""Build the honest-agents task dataset: sources -> dedup -> split -> variants -> (validate) -> mixture -> JSONL.

    python -m honest_agents.tasks.build --out data --sources endless-terminals --limit 600 \
        --test-seeds 300 --val-seeds 60 --mixture 0.55,0.20,0.25

Layout written under --out:
    seeds/<source>/<task>/          normalised feasible seed tasks
    tasks/<variant_id>/             plain feasible copies, infeasible variants and their near-miss controls
    train_pool.jsonl                every train row before the difficulty gate and the mixture (what Phase 0 can sample)
    val_pool.jsonl                  every val row before the mixture (what `tasks.probe_set` draws the probe list from)
    {train,val,test}.jsonl          rows for TRL / the pilot / evaluation (docs/design.md section 3 columns)
    split_seeds.json                seed group -> split, kept across rebuilds so a test seed never reaches training
                                    (`--split-seeds-file` / `--split-seeds` reuses one when building into a fresh --out)
    test_ids.txt                    every held-out seed group plus a sha256 line; commit it. A build refuses to emit a
                                    train, val or train_pool row for any pair_id or source in it (`--test-ids-file`)
    relabels.json                   variants validation relabelled as controls (re-applied on every rebuild)
    dedup.jsonl, validation.jsonl, manifest.json

The build is atomic: every dataset file above (train_pool, train, val, test, manifest, split_seeds, test_ids, relabels,
dedup) is written to a temporary sibling directory `.<out>.build-<pid>` and renamed into `--out` only after every gate
has passed, manifest last. A refused or crashed build leaves no train, val, test or train_pool behind. `seeds/`,
`tasks/` and `raw/` are caches that a retry reuses, so they stay.

docs/design.md section 7 rules implemented here:

* **Pool-independent splits**: a source group goes to `test` / `val` / `train` by `sha256(salt|group) mod 10000`
  against fixed cutoffs (`--test-bp 2200 --val-bp 2200`, about 560 and 540 of the full 2,482-task pool), so changing `--limit`,
  the sample seed or the pool size never moves a source. The legacy `--test-seeds N --val-seeds M` ranks the pool
  instead (only for tiny dev pools; it warns). Either way `split_seeds.json` and `test_ids.txt` persist the assignment
  and the build refuses to put a test id into any other split. A pair never crosses splits.
* **Near-duplicate clusters** (`--neardup-threshold`): sources whose instructions (word 3-gram Jaccard) or tests (5-gram)
  nearly coincide are split as one unit, so a paraphrase of a test task never trains.
* **Decontamination is mandatory**: `--benchmark-dir` must be given and non-empty (`scripts/fetch_benchmarks.sh`),
  unless `--i-know-no-benchmark`. Seeds with a Terminal-Bench canary, or whose Dockerfile COPY/ADDs tests or expected
  files, are rejected; `environment/{task.json,test_initial_state.py,container.def}` are deleted (see `sources.py`).
* **Three held-out families** appear only in `test`: `corrupt_input`, `conflicting_constraint` and `no_network`
  (default; `conflicting_constraint` is not built by default since DEF-27, its twins being passable, so the test split
  holds it only when `--operators` names it). Their controls are held out with them. `corrupt_input` is the transfer test: the input is present but its
  bytes are noise, a mechanism no training family has, and its twin and control differ only in the environment (which
  file is corrupted). The independent leak verification (docs/data.md) found a lexical cue in conflicting_constraint
  that no template change removes (the control's path is introduced as stale) and one training pair of no_network, so
  those two are zero-shot test families with a construct-validity caveat. The families whose twin differs only in the
  environment (`high_risk`, `revoke_permission`, `remove_input`) train. The cue gate refuses the build for a training
  family AND for `corrupt_input` (`config.ENFORCED_HELDOUT_FAMILIES`: a transfer test with a cue measures the cue) and
  only reports for the other held-out families, so those can never block a build.
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
  `--allow-legacy-verifier` (docs/design.md section 6: `train/` refuses them anyway).
"""

from honest_agents.tasks.build.cli import main, parse_args
from honest_agents.tasks.build.config import (
    COLUMNS,
    DEFAULT_HELDOUT_FAMILIES,
    DEFAULT_MIXTURE,
    DEFAULT_TEST_BP,
    DEFAULT_VAL_BP,
    ENFORCED_HELDOUT_FAMILIES,
    BuildConfig,
    Row,
    heldout_problems,
    make_prompt,
    normalise_mixture,
)
from honest_agents.tasks.build.mixture import balance_mixture
from honest_agents.tasks.build.pairs import enforce_pairing
from honest_agents.tasks.build.pipeline import build, row_dict, write_jsonl
from honest_agents.tasks.build.reports import HELDOUT_CAVEAT, cue_metrics, cue_watch
from honest_agents.tasks.build.seeds import check_licenses, collect_seeds, dedup_seeds, filter_hermetic
from honest_agents.tasks.build.splits import (
    assign_seed_splits,
    check_no_test_leak,
    hash_key,
    ids_digest,
    phrasing_split,
    read_test_ids,
    split_bucket,
    split_for_bucket,
    write_test_ids,
)

__all__ = [
    "COLUMNS",
    "DEFAULT_HELDOUT_FAMILIES",
    "DEFAULT_MIXTURE",
    "DEFAULT_TEST_BP",
    "DEFAULT_VAL_BP",
    "ENFORCED_HELDOUT_FAMILIES",
    "HELDOUT_CAVEAT",
    "BuildConfig",
    "Row",
    "assign_seed_splits",
    "balance_mixture",
    "build",
    "check_licenses",
    "check_no_test_leak",
    "collect_seeds",
    "cue_metrics",
    "cue_watch",
    "dedup_seeds",
    "enforce_pairing",
    "filter_hermetic",
    "hash_key",
    "heldout_problems",
    "ids_digest",
    "main",
    "make_prompt",
    "normalise_mixture",
    "parse_args",
    "phrasing_split",
    "read_test_ids",
    "row_dict",
    "split_bucket",
    "split_for_bucket",
    "write_jsonl",
    "write_test_ids",
]
