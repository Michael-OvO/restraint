# Data

What the task dataset is made of, which families train, how many tasks stand behind each, under what licence the source data
may be used, which leaks were found and how each was closed, and how to rebuild it. Code: `src/honest_agents/tasks/`; the build is
`scripts/build_dataset.sh`. Everything here was measured without Docker or a GPU, on the full real Endless Terminals (ET) pool
(2,492 tasks, 2,482 of them convertible; section 8 has the numbers and the build cost). Nothing has been oracle-validated yet
(that needs Docker).

## 1. Sources and licences

The first dataset uses **Endless Terminals only**, converted to the hermetic pytest verifier ([design.md](design.md) section 7).

| source | data licence | where it was read | status |
|---|---|---|---|
| `endless-terminals` (`obiwan96/endless-terminals`) | MIT (generator code: Apache-2.0, `kanishkg/endless-terminals`) | dataset card front-matter (`license: mit` sits after the body, so the Hub API returns `None`); the tasks are LLM-generated, so the generating model's terms may apply too | in use, the whole pool (2,492 tasks, 0.41 GB on the Hub of which 0.30 GB is model trajectories the build drops; 105 MB kept) |
| `seta-env` (`camel-ai/SETA-Env`) | Apache-2.0 | dataset card, not re-read | **disabled**: its tasks carry Terminal-Bench canary GUIDs |
| `tmax-15k` (`TMaxxx/TMax-15K-Harbor`) | unspecified on the re-upload (upstream `allenai/TMax-15K` is ODC-BY-1.0, contains Gemini outputs) | none | **refused** unless `--allow-unknown-license`; also disabled; no oracle solutions, so never a validated source |
| AbstentionBench | CC BY-NC 4.0 | | evaluation only, not vendored |
| Terminal-Bench 2.0, TB Lite, Agentic Abstention, AgentAbstain | benchmarks | | deduplicated out of training (section 4) |

`local` task dirs carry whatever `[metadata.provenance] license` says and are otherwise recorded as "unknown" without being
refused (they are the owner's own files, and the test suite builds from them). The build records the facts in `manifest.json`
(`licenses.<source>`: `license`, `known`, `basis`, `code_license`, `note`) and refuses a registered source with an unknown licence
before any download. `tasks.sources.SourceInfo` is the single record: change a licence there, with its basis. Open: ask the ET
authors to fix the dataset card (judgement, not done). Releasing the dataset needs the same table checked again.

## 2. Families and variants

Each operator makes an infeasible variant plus a near-miss control that shares the `pair_id`, differs only in the object that
makes the obstacle harmless, and records `decoy_targets`. Each infeasible variant records `expected_targets`.

| family | what is impossible | role by default | why |
|---|---|---|---|
| `high_risk` | an irreversible deletion, symmetric twins: identical wording, only the directory's contents differ (a README or the data itself) | train | no text cue |
| `revoke_permission` | the lock sits in the image, the instruction is byte-identical; the lock is on the file's directory when the file is the only deliverable there, so `expected_targets` lists both the file and the directory (`perturb/revoke_permission._lock_targets`) | train | no text cue |
| `remove_input` | a build-time input is deleted; filtered so a recreated input cannot pass (section 5) | train | no text cue |
| `remove_tool` | a tool is removed (the instruction must mandate it and no interpreter may substitute) | **not built by default** (`perturb.NOT_BUILT_BY_DEFAULT`; `--operators remove_tool` turns it on) | structural: 1 variant in 2,482 ET seeds (section 8) |
| `conflicting_constraint` | the instruction forbids the path the task needs | **held out** (test only) | a lexical cue no template removes (section 4) |
| `no_network` | the network is cut | **held out** (test only) | about 2 pairs per 150 seeds: too few to train or report |

Yield, seeds that give at least one infeasible variant (each with a matched control), per 150 real ET seeds. "Before" is the
register's pre-fix recount, "first pass" the code after the first leak fixes on 335 seeds, "full pool" the current code on all
2,482 convertible seeds (`python -m honest_agents.tasks.build ...`, manifest key `yield`). On the full pool every near-duplicate
cluster is a single seed (largest instruction 3-gram Jaccard between two seeds 0.076, threshold 0.1), so "independent source
tasks" equals the seed count behind each family:

| family | before the leak fixes | first pass (335 seeds) | full pool per 150 | seeds / independent source tasks, full pool |
|---|---|---|---|---|
| high_risk | 148 | 102.5 | 105.2 | 1,740 |
| revoke_permission | 101 | 96.7 | 94.8 | 1,568 |
| conflicting_constraint | 72 | 29.1 | 33.5 | 555 |
| remove_input | 31 | 7.6 | 11.4 | 188 |
| no_network | 1 | 1.8 | 2.5 | 41 |
| remove_tool | 0 | 0.0 | 0.1 | 1 |

2,207 of 2,482 seeds (89%) yield at least one trainable infeasible variant (1,141 seeds yield two trainable families, 74 all
three). The manifest carries, per split and family, `yield.<split>.families.<family>` (`pool_infeasible`, `pool_controls`,
`pool_source_tasks`, `pool_yield_per_150_seeds`, `emitted_pairs`, `emitted_source_tasks`) and `training.pairs_per_family` /
`training.thin_families` (fewer than 40 pairs logs a warning; on the full pool no training family is thin: remove_input has 104
pairs in train). Controls are 1:1 at generation. Phrasing pools (8 train, 4 test) exist for `conflicting_constraint` and
`high_risk`; the environment operators use name pools; `perturb.POOLS` partitions all of them between train and test.

**Why `remove_tool` is out of the default list.** Every ET image is ubuntu plus python3 plus pytest, so a command in an
oracle script (10,173 mentions over 2,482 seeds) is almost always an essential binary (9,264), absent from the image (418),
installed by the oracle itself (2), or a non-essential tool that is present (489). Of the 489, 435 are not mandated by the
instruction; of the 54 that are (mostly curl, rsync, make, sqlite3, ssh-keygen, jq), 38 can be replaced by python, 15 are used by
the verifier, and one seed (`task_000000_b314bfef`, gpg) is left. That is a property of the source, not a threshold: a source
whose images ship real tools could turn the operator on. The operator and its tests stay.

**Held-out decision** (taken while the owner was away; revisit freely). The first plan held out `high_risk` and
`revoke_permission` as clean mechanism-transfer tests. An independent verifier then re-measured the leak fixes on 170, 261 and 335
real seeds (grouped cross-validation by source, shuffled-label null, bootstrap over pairs). Split stability, held-out families
absent from train, near-duplicates, benchmark overlap (max Jaccard 0.0094 over 912 texts), emitted environment files, label strings,
the prompt column and twin structure were clean. `conflicting_constraint` was not: the decoy path is introduced as stale or
unrelated, so a rule on the forbidden path's own sentence reaches 86% accuracy and a classifier on the lines mentioning the path
AUC 1.0; the fraction of the path's name tokens found in the task's original instruction gives AUC 0.99 (train) and 1.00 (test); and
the enforced cue gate failed on the 335-seed pool (name-token AUC 0.634 on 52 pairs) while passing at 261 and 170, so a gate on a
family with this few pairs depends on the pool. So families whose twin differs only in the environment train, and
`conflicting_constraint` and `no_network` are **zero-shot test families**. This is a design change, not a weakening of a check:

- the cue gate (`--enforce-cue-band`, on by default) refuses a build for any family in the training mix and only reports for a
  held-out one (`manifest.json` `cue_metrics.heldout_violations`), so a test-only family can never block a build;
- the leakage tests stay strict for every training family and assert that the held-out families and their controls are absent from
  train, val and train_pool and present in test;
- any number for a held-out family carries the construct-validity caveat (a policy can learn "forbidden path described as unrelated
  means proceed") and stays out of headline results;
- `HELDOUT=conflicting_constraint scripts/build_dataset.sh` also trains `no_network` (about 4 pairs per 335 seeds; an owner decision
  to revisit).

With these defaults training sees three of AgentAbstain's eight scenarios.

## 3. conflicting_constraint after the fix

The first design (the control forbids an existing input, the variant forbids a missing output) let `test -e` predict the label (control
426/432 present, infeasible 44/432). Now a pair shares one image and one instruction up to one path in the constraint sentence. The
task names a deliverable D; a second path D' (same directory and extension, a name recombined from the words of other tasks'
deliverables, never a copy) is introduced as a stale leftover in both variants with as many mentions as D. The impossible variant
forbids D, the control forbids D'. When a stale stub of D is planted, an identical stub of D' goes beside it. The enforced
classifiers use folds equal to donor groups (`analysis/cues.donor_group`), so a control's borrowed words never come from the fold
being tested.

| cue (conflicting_constraint, held out by source) | before | after |
|---|---|---|
| exists at boot, AUC | 0.007 (old code) | 0.50 |
| name-token naive Bayes, grouped CV | 0.82 | 0.49 (57 pairs) |
| structure logistic regression (depth, extension, mentions, position, size, name length), grouped CV | 0.74 | 0.46 |
| mentions in instruction, AUC | 0.66 | 0.50 |
| whole-instruction bag-of-words, source-grouped CV | 0.51 | 0.51 |
| share of the path's name words found in the seed's own instruction (`name_token_frac`), full pool, 129 test pairs | 0.99 | 0.984 (reported while held out; enforced for any family that trains) |

**Remaining, not removable:** the label is whether the forbidden path is the one the task requires, and a classifier given only the
lines that mention the forbidden path reaches AUC 1.0 because the control's path is introduced as stale. That is the conflict itself
in lexical form. Opener and tail pools (8 and 6 train, 3 and 2 test phrasings) keep any single template from being a marker, but the
generator is the same for test, so this is a construct-validity limit to state next to any number. Decoy names are also longer and
have more digits than deliverables (low severity; held out, not changed).
Since `name_token_frac` is enforced, a build that trains `conflicting_constraint` (`HELDOUT=no_network`) is refused at AUC 0.98
unless `NO_ENFORCE_CUE_BAND=1`: the family can only be reported zero-shot until its decoy stops borrowing words.

## 4. Leak register: status of every finding

The leakage audit (static plus real ET data, 150 seeds, no Docker) found no blocker: no hidden tests, oracle, `task.toml` or label
column reaches the policy, and no test or benchmark content is in train. It found one real shortcut and several cheap guards. All
are fixed locally; Docker-only checks are in section 6.

| ID | Finding | Status | Proof |
|---|---|---|---|
| L1 | conflicting_constraint path-existence shortcut (accuracy 0.935) | **fixed**; one lexical cue remains by design (section 3) | `tasks/perturb/conflicting_constraint.py` (`_cc_plan`, `_borrowed_sibling`, `_cc_edit`), `tasks/analysis/cues.py` (`cue_classifiers`, `ENFORCED_FEATURES`); `tests/test_data_leakage.py::test_every_training_family_has_a_neutral_existence_cue`, `::test_forbidden_path_existence_rule_is_at_chance_on_the_emitted_tasks`; `tests/tasks/test_tasks_perturb.py`, `test_tasks_analysis.py` |
| L2 | test seeds move when the pool changes (78 of 300 old test seeds fell into train at LIMIT=2492 in simulation) | **fixed**: fixed-hash split, `test_ids.txt` with its hash, `train/` refuses a test id | `tasks/build/splits.py` (`split_bucket`, `write_test_ids`, `check_no_test_leak`); `tests/test_data_leakage.py::test_rebuild_at_a_different_pool_size_keeps_the_test_ids`, `::test_a_frozen_test_id_stays_in_test_and_a_tampered_file_is_refused`, `::test_train_accepts_the_built_rows_and_refuses_a_test_id` |
| L3 | variant id, operator and `__control` in Docker names and error text | **fixed in the training path** (Docker unverified): opaque `ep<hex>` ids, fixed `[sandbox unavailable]` text, also in OpenEnv | `harness.py` (`harbor_names`, `SANDBOX_UNAVAILABLE`), `trainer.py` (`_tool_error_text`), `openenv_env/server/environment.py`; `tests/test_data_flow.py` |
| L4 | decontamination off by default | **fixed**: benchmark dir mandatory; seeds with a canary or a COPY/ADD of tests, expected files or `task.json` are rejected; `task.json`, `test_initial_state.py` and `container.def` are deleted from every emitted variant | `tasks/build/seeds.py::check_benchmark_config`, `tasks/taskfs.py::scrub_environment`; `tests/test_data_leakage.py::test_the_build_fails_without_a_benchmark_dir`, `::test_no_emitted_environment_file_holds_the_expected_final_state` |
| L5 | label strings in Dockerfile comments | **fixed**: stripped, and emitted files are scanned (dropped variants counted in `manifest.json`) | `perturb/emit.py::scrub_labels`, `analysis/labels.py::label_leaks`, `build/reports.py::emitted_label_leaks`; `tests/test_data_leakage.py::test_no_label_word_in_any_emitted_environment_file_or_comment` |
| L6 | split guards accept rows with no split | **fixed**: `split == "train"` required in `train.check_rows`, the pilot and `difficulty.py` | `tests/test_data_flow.py::test_check_rows_train_requires_split_train`, `::test_pilot_refuses_non_train_and_missing_splits`, `::test_difficulty_counts_only_train_rows` |
| L7 | eval-sourced quarantine entries change `test.jsonl` on rebuild | **fixed**: eval, test and openenv sources ignored by default | `bank.DEFAULT_IGNORED_SOURCES`; `tests/test_data_flow.py::test_eval_quarantine_entries_are_ignored_by_default` |
| L8 | held-out families are described in the system prompt (a claim limit, not a leak) | **documented**: call them "held-out variants", run the neutral-prompt arm | [evaluation.md](evaluation.md) section 1, [proposal.md](proposal.md) section 7.1 |
| L9 | the cue gate was blind to the environment text of the environment-only families (a twin and its control share one instruction, so every path feature is pair-identical): README wording and length, lock owner and mode | **fixed**: grouped-CV classifiers over the planted text are a hard gate for training families (section 4.1); the first full-pool build found a real shortcut, README length at AUC 0.91, and the junk texts were rewritten to match the risky ones | `tasks/analysis/envtext.py`, `tasks/perturb/pools.py` (`_JUNK_WHAT`, `_make_sites`), `tests/tasks/test_tasks_envtext.py`, `tests/test_data_leakage.py::test_the_planted_environment_text_of_every_training_family_is_neutral_except_its_carrier`, `tests/tasks/test_tasks_perturb.py::test_every_risky_text_has_a_junk_text_of_the_same_shape_and_they_are_dealt_as_pairs` |
| L10 | the name-token-share feature (AUC 1.0 for a borrowed-name decoy) was not enforced | **fixed**: `name_token_frac` is in `ENFORCED_FEATURES`, computed against the seed's own instruction | `tasks/analysis/cues.py`; `tests/tasks/test_tasks_envtext.py::test_a_decoy_that_borrows_names_from_other_tasks_is_refused_for_a_training_family` |
| L11 | the empty-build guard sat after the seed collection, so a build whose seeds were all removed by decontamination or the hermetic filter wrote a manifest with `n_seeds: 0`; a cue metric that crashed counted as clean | **fixed**: the guard is after both filters and names the counts; an enforced build fails on a crashed metric; both leave no dataset file | `tasks/build/pipeline.py`; `tests/tasks/test_tasks_build.py::test_build_refuses_legacy_seeds_by_default`, `::test_build_where_decontamination_removes_every_seed_fails_atomically`, `::test_a_cue_metric_that_crashed_fails_an_enforced_build_instead_of_passing_it` |
| pairing | feasible rows came from unpaired sources | **fixed**: `balance_mixture(paired=True)` | manifest `pairing` |
| quality | paragraph insertion broke sentences or lists in 51% of cc variants; remove_input targets under 40 characters; revoke locked all of `/home/user`; high_risk scope clauses | **fixed**, static only: 20 variants per family read by hand, none bad | `tests/tasks/test_tasks_perturb.py`, `tests/test_data_leakage.py::test_operators_insert_a_whole_paragraph_after_a_finished_one` |

Smaller items, not fixed (the twin-structure numbers below are from the 335-seed pool; the full-pool values are in section 8): the no-network sandbox still answers ICMP and DNS (fidelity only); the `remove_tool` control keeps the
`/etc/alternatives` link while the impossible variant deletes it (latent: no ET task yields a `remove_tool` variant; deleting it
in the control would dangle symlinks the oracle needs, and the right fix needs a Docker run); Qwen3.5-4B post-dates ET and
Terminal-Bench, so report deltas between arms only and add a Qwen3-4B control; the tmax-4b reference model may overlap the test set
(unverified). Twin-structure cue measurements on the default mix (335 seeds): every enforced feature and both grouped-CV classifiers
sit at AUC 0.50 for the three training families in train and test, because the twins have a byte-identical instruction. Impossible
against the plain feasible copy scores 0.70 to 0.76 (the perturbation marker is shared by both classes, so it carries no label), and a
Dockerfile-diff classifier scores 0.94 to 1.00, which is the environment cue itself (what the agent must find by looking), not a
text cue.

### 4.1 Environment text (L9)

The twins of `high_risk`, `revoke_permission` and `remove_input` differ only in the environment, which is the evidence the agent has
to find by looking. So the words of the README, the owner and mode of a lock and the moved-aside file score AUC 1.0 in any
classifier, by design (the manifest lists them as `env:carrier_readme`, `env:carrier_lock`, `env:carrier_files` and does not
enforce them: `analysis.envtext.CARRIER_CHANNELS`). What must not separate the two sides is everything around the evidence. The
build reads three channels off each emitted variant (the README-like files, the other planted files with their names, the
`chown` and `chmod` words), classifies them with grouped cross-validation (folds are the donor groups, so a source, its twin and its
control sit in one fold), and refuses the build when, for a training family, any of these leaves [0.4, 0.6]:

| gated entry | what it is |
|---|---|
| `env:readme_words`, `env:readme_chars`, `env:readme_commas` | the README's length and punctuation, one feature each |
| `env:style` | logistic regression over the README's words, characters, commas, digits, apostrophe and capitalised words |
| `env:readme`, `env:files`, `env:lock` | naive Bayes over the words of a channel that is not the family's carrier |

First full-pool run: `high_risk` failed with README length at AUC 0.91 (chars 0.86, style classifier 0.96; val and test alike). The
risky texts ("the only copy of last night's production database backups") were longer and had more commas and apostrophes than the
junk ones ("a transcoding cache"), so a policy could learn "long README, abstain" without reading it. The junk texts are now
rewritten to match the risky text at the same index in word count, character count, commas, apostrophes and digits, and
`_make_sites` deals them as pairs, so a twin's README and its control's have one shape (train and test pools; a unit test checks it).
After the fix every gated entry sits at 0.50 in train, val and test on the full pool, while the carriers stay at 1.0. A gate on the
env text cannot see what the parser does not model: `remove_input`'s moved file is not read from the Dockerfile (its carrier AUC is
0.50 on the real pool), so that family's environment check is only that nothing else differs.

## 5. remove_input: recreatable inputs

About a quarter of the impossible `remove_input` variants could be passed by recreating the input (3 clear and 2 unsure of 13
hand-read): a C file whose output the instruction prints, a private key a test only compares with itself, a log the test re-parses.
Three layers stop that:

1. **Static candidate filter** (`perturb/remove_input._input_candidates`). A content floor of 80 characters and 4 lines, not a pointer
   file, under half of its lines and words in the instruction, the oracle does more than move or copy it, and the tests must depend on
   what the file held: `analysis/filters.content_bound_tokens` needs a distinctive value of the file (a number of 3+ characters, or a
   6+ character word that is not a keyword) that the pytest files spell out and that neither the instruction nor any other file of the
   image shows. A test that recomputes its expectation from the file at run time has no such token, so the file is no candidate.
   The static plausible-recreation probe sits on top (`analysis/filters.guessable_tokens`): a candidate is dropped when every one of
   those tokens can be rebuilt from the visible text under another case, plural or tense, or with the commas of a number
   (`4521` for `4,521`).
2. **Repair probes** (`validate.py`, need Docker): recreate the removed path empty and rerun the oracle; if that fails, recreate it with
   plausible content (`analysis.plausible_recreation`: the code blocks and indented lines of the instruction plus a sibling file of
   the same extension, never the removed file itself) and rerun it. A pass in either probe relabels the variant as a control
   (`relabel_to_control`; the control's oracle then makes the same recreation, a `plausible:<path>` entry in `relabels.json`).
   Validation results made before the plausible probe are not current and are re-run (`settings.repair_plausible`).
3. **Hand audit**: `scripts/build_audit_set.sh` sheets show the test file names, what each test asserts (first 200 characters) and the
   oracle command, plus a checklist line per variant (`--no-show-tests` hides them). Audit every `remove_input` test row by hand.

On the 335 cached seeds this cut candidates from 35 (28 seeds) to 19 (17 seeds), at the price of yield (12.5 to 7.6 per 150 seeds).
One unsure case remains (`mock_opt_solver`). On the full pool (2,482 seeds) the static plausible-recreation probe changes little:
candidates 213 to 212, impossible variants 189 to 188, so it flips 1 variant (0.5%) to a non-candidate; 17 further candidates have
some but not all bound tokens guessable and stay. The real flip rate of the Docker probe, like the remaining leak rate, is
unmeasured until a Docker run.

## 6. Rebuilding and checking

```bash
scripts/fetch_benchmarks.sh benchmarks      # pinned commits of the benchmark texts to deduplicate against; mandatory
scripts/build_dataset.sh data               # whole pool: git clone -> hermetic conversion -> variants -> validation (if Docker) -> splits
```

The whole pool builds in about 75 s on a laptop (the download is one shallow `git clone`, 5 s; section 8). `build_dataset.sh` is
configured by environment variables (the header of the script lists them all): `LIMIT` (empty = the whole pool, the default; a number
samples), `SOURCES`, `BENCH_DIR` (required; `I_KNOW_NO_BENCHMARK=1` is for tests only), `TEST_BP` / `VAL_BP` (2200 / 2200 of 10,000:
564 test and 536 val sources of 2,482), `HELDOUT` (`conflicting_constraint,no_network`), `MIXTURE` (`0.55,0.20,0.25`), `NEARDUP_THRESHOLD`,
`QUARANTINE_FILE`, `DIFFICULTY_JSON`, `VALIDATE` (`auto` = if `docker info` works), `NO_ENFORCE_CUE_BAND`, `ALLOW_UNKNOWN_LICENSE`.
Output: `{train,val,test,train_pool}.jsonl`, `test_ids.txt`, `manifest.json` (yield, training pairs per family, cue metrics, licences,
pairing), `tasks/`, `dedup.jsonl`, `seeds/<source>/_rejected.jsonl` (written when the hermetic conversion refused a seed).

- **Atomic**: every dataset file is written into a temporary sibling `.<out>.build-<pid>` and renamed into `--out` after every gate
  passed, manifest last. A failed gate or crash leaves no dataset file behind and a previous complete dataset stays. `seeds/`,
  `tasks/` and `raw/` are caches a retry reuses.
- **With Docker** the build runs five steps, all resumable: build, validate, rebuild (relabels harmless `remove_input` variants as
  controls, quarantines leaks), validate again, final rebuild. **Without Docker** it builds unvalidated and says so; re-run on the
  Linux node before training.
- **Frozen after the first training run**: the salt (`--split-seed`, default 0) and the cutoffs (`TEST_BP=2200`, `VAL_BP=2200`).
  Nothing has trained yet, so they are still free; the exact freeze command and the recommended values are in section 8.
- **Format**: every instruction, the plain copy included, goes through `taskfs.normalise_instruction` (paragraphs separated by one
  blank line, one final newline), so formatting does not separate a perturbed row from a plain one.
- **Audit set**: `VALIDATE=1 scripts/build_audit_set.sh data/audit` (20 to 30 sources, the training families plus controls), read the
  sheets, type `ok|wrong|unsure` on each `Verdict:` line, then `scripts/build_audit_set.sh --tally data/audit`.

**To run inside an episode container on the node** (once per operator and control; the task id or an operator name must appear in
none of the output): `cat /etc/hosts /etc/resolv.conf; hostname; cat /proc/self/cgroup`; `getent hosts $(hostname -i)`; `env; cat
/proc/1/environ | tr '\0' '\n'`; `ls -la / /tmp; ls -d /tests /solution /logs /app`; `grep -v overlay /proc/self/mountinfo`; `find /
-xdev ( -name task.json -o -name 'test_*' -o -name solve.sh -o -name task.toml ) -not -path '/usr/*' -not -path '/proc/*'`. Under
deny-all: `curl -sS -m5 https://pypi.org; ping -c1 8.8.8.8; getent hosts pypi.org`. Compare `stat` and `ls -lu` on the high_risk
twins right after boot (atime), and check that `test -e <forbidden path>` agrees with the static estimate.

## 7. Open

- The Docker-only checks: oracle validation of every variant (the audit-set gate is at least 95% validating as labelled), the real
  leak rate, and how many `remove_input` variants the empty and the plausible repair probes flip (the static probe flips 1 of 189).
- Whether `remove_tool` can yield anything on a source other than ET (structural zero on ET, section 2).
- Whether to train `no_network` (owner decision: 41 pairs in the whole pool, 21 in train, 15 in test; the two-family list
  `HELDOUT=conflicting_constraint` trains it with about 21 pairs).
- Whether to keep `conflicting_constraint` as a reported zero-shot family (owner decision, section 3: `name_token_frac` AUC 0.984,
  129 test pairs); it stays held out and out of headline numbers.
- The probe size: `probe.max_variants` is 100 in `configs/base.yaml`, a hash-chosen subset of `val.jsonl`, so only about 100 of
  the 974 validation rows are probed per evaluation. The larger validation split pays off only if that is raised (a change request
  to the config owner) or if checkpoint selection reads the whole file.
- The test split has 42 impossible `remove_input` pairs and 15 `no_network` pairs (< 50, the reporting floor
  `MIN_FAMILY_TEST_ROWS`): report them pooled or not at all.

## 8. Full-pool build (2026-10-05)

Source: `obiwan96/endless-terminals` at git revision `26ecf78458e7f756e4d06780d5fbf3dd78e91815` (the manifest records it as
`source_revisions`), 2,492 task directories; 10 are refused by the hermetic conversion (the image does not install python3), leaving
2,482 seeds. Benchmarks: `scripts/fetch_benchmarks.sh` at the pinned commits, 912 texts (terminal-bench-2 89, TBLite 100,
agentic-abstention 197, agentabstain 526; the clones of the first two were missing from the earlier offline check, which saw
197 texts). Decontamination against all of them: 0 seeds dropped at the 0.5 Jaccard threshold, maximum word 8-gram Jaccard 0.0038,
49 of 2,482 seeds share any 8-gram with any benchmark text. The max overlap is recorded in `manifest.json` `benchmark_overlap`.

**Before and after** on the same pool (`before` = the code at the start of this pass with `VAL_BP=400`, cue gate not enforced
because it did not exist for the env text; `after` = this code with the defaults):

| | before | after |
|---|---|---|
| seeds / train / val / test source groups | 2,482 / 1,821 / 97 / 564 | 2,482 / 1,382 / 536 / 564 |
| training families (default) | remove_tool, remove_input, revoke_permission, high_risk | remove_input, revoke_permission, high_risk |
| train pairs per family (remove_tool / remove_input / revoke / high_risk) | 1 / 140 / 302 / 302 | n/a / 104 / 230 / 231 |
| val impossible rows (remove_input / revoke / high_risk) | 7 / 16 / 17 | 42 / 88 / 89 |
| test impossible rows (remove_input / revoke / high_risk / cc / no_network) | 42 / 357 / 394 / 129 / 15 | unchanged |
| rows train / val / test | 3,311 / 177 / 2,438 | 2,512 / 974 / 2,438 |
| independent source clusters | 2,482 (no cluster at 0.1) | same |
| cue gate, training families | passes (path features only) | passes, now incl. `name_token_frac` and the env text (first run failed: README length 0.91) |
| empty build after a filter | writes `n_seeds: 0` | fails, no dataset file |
| build wall time / peak RSS | 89 s / 497 MB | 71 s (70 s with the pool already cloned) / 497 MB |

Where the time went (profile shares of a 70 s build): collecting and normalising the seeds about a quarter (`copytree` and the
hermetic conversion), variants about 55% (each variant copies its task directory and Harbor loads it), near-duplicate clustering
14 s before and 4 s after (the inverted
index now counts shared shingles from the postings and only intersects pairs that can reach the threshold; the pairs are identical
to the old search, a unit test compares against brute force), the rest I/O. Also in this pass: `TaskFacts.shared` analyses a seed
once for all operators and controls, and `_recombined_name` splits the name bank once per task (conflicting_constraint, 9.5 s of
a profile). The generated task directories are byte-identical to the build before those two changes.

**Cue gate on the full pool (AUC, train / val / test; `n` is pairs):** every gated path feature, `cv:name_tokens`, `cv:structure`,
`env:style`, `env:readme_*` and `env:files` is 0.50 for `high_risk`, `revoke_permission` and `remove_input` in train, val and
test. The carriers are `env:carrier_readme` 1.00, `env:carrier_lock` 1.00, `env:carrier_files` 0.50
(not parsed). Held-out: `conflicting_constraint` `name_token_frac` 0.984, `cv:name_tokens` 0.567, `cv:structure` 0.514,
everything else 0.50; `no_network` has too few pairs to judge. The 335-seed failure (name-token AUC 0.634 on 52 pairs) was a pool
artefact of a family that no longer trains.

**Split cutoffs: final values and recommendation.** The split is a fixed hash of the source id, so the test set never moves with the
pool. `TEST_BP` stays 2200 (564 test sources; every reported family has at least 42 impossible test rows except no_network).
`VAL_BP` goes from 400 to 2200 because that is the smallest cutoff at which `remove_input` reaches 40 impossible validation rows
(measured by simulating the real mixture on the full pool, matching the real builds for 400 and 1200):

| `VAL_BP` | val groups | val impossible rows (revoke / high_risk / remove_input) | train groups | train pairs (revoke / high_risk / remove_input) |
|---|---|---|---|---|
| 400 | 97 | 16 / 17 / 7 | 1,821 | 303 / 303 / 139 |
| 1200 | 297 | 50 / 50 / 22 | 1,621 | 269 / 270 / 124 |
| 1500 | 377 | 62 / 62 / 30 | 1,541 | 257 / 257 / 116 |
| 2000 | 492 | 83 / 83 / 35 | 1,426 | 236 / 236 / 111 |
| **2200** | **536** | **88 / 89 / 42** | **1,382** | **230 / 231 / 104** |

The cost is about a quarter of the train rows (the train mixture is capped by its plain feasible rows, so the surplus high_risk and
revoke_permission pairs were unused anyway). The owner may prefer 1200 (about 300 groups, 50 impossible rows for two families, 22
for remove_input). The recommendation is the whole pool (`LIMIT` empty: it costs 75 s and `LIMIT=1500` would give about 22
remove_input validation pairs at these cutoffs), `TEST_BP=2200`, `VAL_BP=2200`, salt 0. **Not frozen: `configs/test_ids.txt`
does not exist.** To freeze after the owner agrees (and before the first training run):

```bash
scripts/fetch_benchmarks.sh benchmarks
scripts/build_dataset.sh data                       # on the Linux node with Docker this validates; locally it builds unvalidated
cp data/test_ids.txt configs/test_ids.txt            # 564 ids and a sha256 line; later builds read it and never shrink it
git add configs/test_ids.txt && git commit -m "Freeze the held-out test ids"
```

After that the salt and `TEST_BP` / `VAL_BP` must not change (raising `TEST_BP` moves already-trained sources into test unless that
file is supplied; the build refuses a test id in any other split).

**Not verified without Docker or a GPU:** that the oracle passes every feasible variant 3 of 3 and fails every impossible one, the
repair probes of `remove_input` (empty and plausible), the leak rate of the impossible variants, the `README`/`ls -l` reading of the
twins inside a real container, and the training value of any of it.
