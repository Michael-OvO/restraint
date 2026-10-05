# Data: sources, licences, families and yield

What the task dataset is made of, which families train, how many tasks stand behind each, and under what licence the
source data may be used. Code: `src/honest_agents/tasks/`; the build is `scripts/build_dataset.sh`. Everything here
was measured without Docker or a GPU, on real Endless Terminals (ET) seeds from the local cache.

## 1. Training and held-out families

| family | role by default | why |
|---|---|---|
| `high_risk` | train | twin and control differ only in the environment (a README or the data itself); no text cue |
| `revoke_permission` | train | same; the lock sits in the image, the instruction is byte-identical |
| `remove_input` | train | same; filtered so a recreated input cannot pass (section 4) |
| `remove_tool` | train, but yields 0 variants on real seeds | the mandate and no-substitute gates reject every ET task |
| `conflicting_constraint` | **held out** (test only) | a lexical cue no template removes (below) |
| `no_network` | **held out** (test only) | about one pair per 150 seeds: too few to train or report |

**Decision (taken while the owner was away; revisit freely).** An independent verifier re-measured the leak fixes on
170, 261 and 335 real seeds (`docs/research/leak-register.md`, section "Independent verification"). Split stability,
held-out families absent from train, near-duplicates, benchmark overlap (max Jaccard 0.0094), emitted environment files,
label strings, the prompt column and twin structure were clean. `conflicting_constraint` was not:

1. the decoy path is introduced as stale or unrelated, so a rule on the forbidden path's own sentence gets 86 percent
   accuracy and a classifier on the lines that mention the path gets AUC 1.0;
2. the fraction of the path's name tokens found in the task's original instruction gives AUC 0.99 (train) and 1.00
   (test): the required path comes from the instruction, the decoy is borrowed from other tasks;
3. the enforced cue gate failed on a 335-seed pool (name-token AUC 0.634 on 52 pairs) while passing at 261 and 170
   seeds, so a gate on a family with this few pairs depends on the pool.

So the families whose twin differs only in the environment train, and `conflicting_constraint` and `no_network` are
**zero-shot test families**. This is a deliberate design change, not a weakening of a check:

- the cue gate (`--enforce-cue-band`, on by default) is a hard refusal for every family in the training mix and only
  reports for a held-out family (`manifest.json` `cue_metrics.heldout_violations`, with the caveat), so a test-only
  family can never block a build;
- the leakage regression tests stay strict for every training family, and assert that the held-out families and their
  controls are absent from train, val and train_pool and present in test;
- any number for a held-out family is quoted with the construct-validity caveat (a policy can learn "forbidden path
  described as unrelated means proceed") and kept out of headline results;
- `no_network` stays available (`HELDOUT=conflicting_constraint scripts/build_dataset.sh`) but is never in the default mix.

`high_risk` and `revoke_permission` are therefore **not** held out any more. Both previous reasons for holding them out
(a clean environment-mechanism transfer test, a `high_risk` zero-shot test) are given up for training coverage; the
transfer claim is now carried by `conflicting_constraint` (a different mechanism: contradictory instructions) and
`no_network`.

## 2. Yield

Seeds that yield at least one infeasible variant (each has a matched control), per 150 real ET seeds. "Before" is the
register's pre-fix recount (`docs/research/audit2-memo.md`); "now" is the current code on a 335-seed pool (74 test, 10 val,
251 train by the hash split; `python -m honest_agents.tasks.build ... --no-enforce-cue-band`, manifest key `yield`).

| family | before the leak fixes | after the leak fixes (register 6.2) | now (335 seeds) | seeds / independent source tasks now |
|---|---|---|---|---|
| high_risk | 148 | 105 | 102.5 | 229 |
| revoke_permission | 101 | 96 | 96.7 | 216 |
| conflicting_constraint | 72 | 28 | 29.1 | 65 |
| remove_input | 31 | 12 | 7.6 | 17 |
| no_network | 1 | 1 | 1.8 | 4 |
| remove_tool | 0 | 0 | 0.0 | 0 |

The statement "training keeps about 4% of seeds" that the first design note carried is wrong and is gone: with the
default families about 85 percent of seeds (283 of 335, train pool and test) yield at least one trainable infeasible
variant, almost all of them high_risk or revoke_permission, with remove_input a small minority.

`manifest.json` now carries, per split and family, `yield.<split>.families.<family>`: `pool_infeasible`,
`pool_controls`, `pool_source_tasks` (independent source tasks, near-duplicate clusters counted once),
`pool_yield_per_150_seeds`, `emitted_pairs` and `emitted_source_tasks` (after the mixture). The build prints the same
table. `training.pairs_per_family` lists the pairs in `train.jsonl` per training family and `training.thin_families`
those under 40 (a warning is logged): on the 335-seed pool remove_input has 14, so it needs a full 1,500-seed build
before it can be reported alone.

## 3. Licences

| source | data licence | where it was read | code licence | status |
|---|---|---|---|---|
| `endless-terminals` (`obiwan96/endless-terminals`) | MIT | dataset card README front-matter (`license: mit`, which sits after the body, so the Hub API returns `license: None`) | Apache-2.0 (`github.com/kanishkg/endless-terminals`, the paper's generator) | known; both permissive |
| `seta-env` (`camel-ai/SETA-Env`) | Apache-2.0 | dataset card (not re-read) | | known, **disabled** (canary GUIDs, leak audit L4) |
| `tmax-15k` (`TMaxxx/TMax-15K-Harbor`) | **unspecified** (upstream `allenai/TMax-15K` is ODC-BY-1.0) | none on the re-upload | | **unknown: refused** unless `--allow-unknown-license`; also disabled |

Notes: the ET tasks are LLM-generated, so the generating model's terms apply as well; ask the authors to fix the card
(judgement, not done). `local` task dirs carry whatever `[metadata.provenance] license` says and are otherwise recorded as
"unknown" without being refused (they are the owner's own files, and the test suite builds from them).

The build records these facts in `manifest.json` (`licenses.<source>`: `license`, `known`, `basis`, `code_license`,
`note`) and refuses a registered source whose licence is unknown before any download. `SourceInfo` in
`tasks/sources.py` is the single record: change a licence there, with its basis.

## 4. `remove_input`: recreatable inputs

About a quarter of the impossible `remove_input` variants could be passed by recreating the input (3 clear and 2 unsure of
13 hand-read, audit 2 D1): a C file whose output the instruction prints, a private key a test only compares with itself,
a log the test re-parses. Three layers now stop that:

1. **Static candidate filter** (`perturb._input_candidates`). Besides the earlier rules (content floor of 80 characters and
   4 lines, not a pointer file, under half of its lines and words in the instruction, the oracle does more than move or
   copy it) the tests must depend on what the file held: `analysis.content_bound_tokens` needs a distinctive value of the
   file (a number of 3+ characters, or a 6+ character word that is not a language or schema keyword) that the pytest
   files spell out and that neither the instruction nor any other file of the image shows. A test that recomputes its
   expectation from the file at run time has no such token: any plausible recreation passes, so the file is no candidate.
2. **Repair probe** (`validate.py`, needs Docker): recreate the removed path empty and rerun the oracle; a pass relabels
   the variant as a near-miss control (`relabel_to_control`). It does not try a plausible non-empty recreation, which only
   a Docker run can show; the static layer above is the cover until then.
3. **Hand audit**: `scripts/build_audit_set.sh` sheets now show the test file names, what each test asserts (first 200
   characters of each) and the oracle command, plus a checklist line per variant, so a reviewer can judge "truly
   impossible" (`--no-show-tests` hides them). Audit every `remove_input` test row by hand (audit 2, section 5).

Measured on the 335 cached real seeds, candidates before the change versus after: **35 candidates in 28 seeds before, 19
in 17 after** (16 dropped, every one with no content-bound test value: the C file, the key, log and csv files the tests
re-parse, JSON schemas, config files). Of the five audit-named variants the earlier filters already removed 2; the new one
removes 2 more (`legacy_hello.c`, `perf_eng_key`) and keeps one unsure case (`mock_opt_solver`, whose objective value
123.45 appears only in the tests). The price is yield: 12.5 to 7.6 seeds per 150. Nothing here ran in Docker, so the
remaining leak rate is unmeasured; the hand audit of test rows is what measures it.

## 5. revoke_permission labels

The lock sits on the file's directory when the file is the only deliverable there. `expected_targets` now lists both the
file and the locked directory (and the control's `decoy_targets` the same two), so a correct directory diagnosis is not
scored `wrong_target` (`perturb._lock_targets`).

## 6. Atomic builds and instruction formatting

A build writes every dataset file into a temporary sibling directory `.<out>.build-<pid>` and renames them into `--out`
only after every gate passed, manifest last; a failed gate or crash leaves no `train`, `val`, `test`, `train_pool` or
`manifest` behind (a previous complete dataset stays as it was). `seeds/`, `tasks/` and `raw/` are caches a retry reuses.

Every instruction, the plain copy included, is written in the one format `insert_paragraph` produces (paragraphs
separated by one blank line, no trailing space after a paragraph, one final newline), so formatting does not separate a
perturbed row from a plain one (`taskfs.normalise_instruction`).

## 7. Open

- `remove_tool` control keeps the `/etc/alternatives` link while the impossible variant deletes it (register, smaller
  items). Deleting it in the control would dangle the moved symlinks the oracle needs; the right fix needs a Docker run.
  No ET task yields a `remove_tool` variant today, so it is latent.
- conflicting_constraint decoy names are longer and have more digits than deliverables (low, held out, not changed).
- Whether a plausible non-empty recreation passes a `remove_input` variant needs the Docker repair probe.
