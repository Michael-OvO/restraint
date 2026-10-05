# Data

What the task dataset is made of, which families train, how many tasks stand behind each, under what licence the source data
may be used, which leaks were found and how each was closed, and how to rebuild it. Code: `src/honest_agents/tasks/`; the build is
`scripts/build_dataset.sh`. Everything here was measured without Docker or a GPU, on real Endless Terminals (ET) seeds from the
local cache. Nothing has been oracle-validated yet (that needs Docker).

## 1. Sources and licences

The first dataset uses **Endless Terminals only**, converted to the hermetic pytest verifier ([design.md](design.md) section 7).

| source | data licence | where it was read | status |
|---|---|---|---|
| `endless-terminals` (`obiwan96/endless-terminals`) | MIT (generator code: Apache-2.0, `kanishkg/endless-terminals`) | dataset card front-matter (`license: mit` sits after the body, so the Hub API returns `None`); the tasks are LLM-generated, so the generating model's terms may apply too | in use |
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
| `remove_tool` | a tool is removed (the instruction must mandate it and no interpreter may substitute) | train, but yields 0 variants on real seeds | the gates reject every ET task |
| `conflicting_constraint` | the instruction forbids the path the task needs | **held out** (test only) | a lexical cue no template removes (section 4) |
| `no_network` | the network is cut | **held out** (test only) | about 2 pairs per 150 seeds: too few to train or report |

Yield, seeds that give at least one infeasible variant (each with a matched control), per 150 real ET seeds. "Before" is the
register's pre-fix recount, "now" the current code on a 335-seed pool (74 test, 10 val, 251 train by the hash split;
`python -m honest_agents.tasks.build ... --no-enforce-cue-band`, manifest key `yield`):

| family | before the leak fixes | after the first fix pass | now (335 seeds) | seeds / independent source tasks now |
|---|---|---|---|---|
| high_risk | 148 | 105 | 102.5 | 229 |
| revoke_permission | 101 | 96 | 96.7 | 216 |
| conflicting_constraint | 72 | 28 | 29.1 | 65 |
| remove_input | 31 | 12 | 7.6 | 17 |
| no_network | 1 | 1 | 1.8 | 4 |
| remove_tool | 0 | 0 | 0.0 | 0 |

About 85% of seeds (283 of 335) yield at least one trainable infeasible variant, almost all high_risk or revoke_permission. The
manifest carries, per split and family, `yield.<split>.families.<family>` (`pool_infeasible`, `pool_controls`, `pool_source_tasks`,
`pool_yield_per_150_seeds`, `emitted_pairs`, `emitted_source_tasks`) and `training.pairs_per_family` / `training.thin_families`
(fewer than 40 pairs logs a warning: on the 335-seed pool remove_input has 14, so it needs a full 1,500-seed build before it can be
reported alone). Controls are 1:1 at generation; about 80% of infeasible rows keep theirs after the mixture. Phrasing pools
(8 train, 4 test) exist for `conflicting_constraint` and `high_risk`; the environment operators use name pools; `perturb.POOLS`
partitions all of them between train and test.

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

**Remaining, not removable:** the label is whether the forbidden path is the one the task requires, and a classifier given only the
lines that mention the forbidden path reaches AUC 1.0 because the control's path is introduced as stale. That is the conflict itself
in lexical form. Opener and tail pools (8 and 6 train, 3 and 2 test phrasings) keep any single template from being a marker, but the
generator is the same for test, so this is a construct-validity limit to state next to any number. Decoy names are also longer and
have more digits than deliverables (low severity; held out, not changed).

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
| pairing | feasible rows came from unpaired sources | **fixed**: `balance_mixture(paired=True)` | manifest `pairing` |
| quality | paragraph insertion broke sentences or lists in 51% of cc variants; remove_input targets under 40 characters; revoke locked all of `/home/user`; high_risk scope clauses | **fixed**, static only: 20 variants per family read by hand, none bad | `tests/tasks/test_tasks_perturb.py`, `tests/test_data_leakage.py::test_operators_insert_a_whole_paragraph_after_a_finished_one` |

Smaller items, not fixed: the no-network sandbox still answers ICMP and DNS (fidelity only); the `remove_tool` control keeps the
`/etc/alternatives` link while the impossible variant deletes it (latent: no ET task yields a `remove_tool` variant; deleting it
in the control would dangle symlinks the oracle needs, and the right fix needs a Docker run); Qwen3.5-4B post-dates ET and
Terminal-Bench, so report deltas between arms only and add a Qwen3-4B control; the tmax-4b reference model may overlap the test set
(unverified). Twin-structure cue measurements on the default mix (335 seeds): every enforced feature and both grouped-CV classifiers
sit at AUC 0.50 for the three training families in train and test, because the twins have a byte-identical instruction. Impossible
against the plain feasible copy scores 0.70 to 0.76 (the perturbation marker is shared by both classes, so it carries no label), and a
Dockerfile-diff classifier scores 0.94 to 1.00, which is the environment cue itself (what the agent must find by looking), not a
text cue.

## 5. remove_input: recreatable inputs

About a quarter of the impossible `remove_input` variants could be passed by recreating the input (3 clear and 2 unsure of 13
hand-read): a C file whose output the instruction prints, a private key a test only compares with itself, a log the test re-parses.
Three layers stop that:

1. **Static candidate filter** (`perturb/remove_input._input_candidates`). A content floor of 80 characters and 4 lines, not a pointer
   file, under half of its lines and words in the instruction, the oracle does more than move or copy it, and the tests must depend on
   what the file held: `analysis/filters.content_bound_tokens` needs a distinctive value of the file (a number of 3+ characters, or a
   6+ character word that is not a keyword) that the pytest files spell out and that neither the instruction nor any other file of the
   image shows. A test that recomputes its expectation from the file at run time has no such token, so the file is no candidate.
2. **Repair probe** (`validate.py`, needs Docker): recreate the removed path empty and rerun the oracle; a pass relabels the variant as
   a control (`relabel_to_control`). It does not try a plausible non-empty recreation; only a Docker run can show that.
3. **Hand audit**: `scripts/build_audit_set.sh` sheets show the test file names, what each test asserts (first 200 characters) and the
   oracle command, plus a checklist line per variant (`--no-show-tests` hides them). Audit every `remove_input` test row by hand.

On the 335 cached seeds this cut candidates from 35 (28 seeds) to 19 (17 seeds), at the price of yield (12.5 to 7.6 per 150 seeds).
One unsure case remains (`mock_opt_solver`). The remaining leak rate is unmeasured until a Docker run.

## 6. Rebuilding and checking

```bash
scripts/fetch_benchmarks.sh benchmarks      # pinned commits of the benchmark texts to deduplicate against; mandatory
LIMIT=1500 scripts/build_dataset.sh data    # seeds -> hermetic conversion -> variants -> validation (if Docker) -> splits
```

`build_dataset.sh` is configured by environment variables (the header of the script lists them all): `LIMIT` (1500 seeds per source),
`SOURCES`, `BENCH_DIR` (required; `I_KNOW_NO_BENCHMARK=1` is for tests only), `TEST_BP` / `VAL_BP` (2200 / 400 of 10,000: about 330 test
and 60 val sources of 1,500), `HELDOUT` (`conflicting_constraint,no_network`), `MIXTURE` (`0.55,0.20,0.25`), `NEARDUP_THRESHOLD`,
`QUARANTINE_FILE`, `DIFFICULTY_JSON`, `VALIDATE` (`auto` = if `docker info` works), `NO_ENFORCE_CUE_BAND`, `ALLOW_UNKNOWN_LICENSE`.
Output: `{train,val,test,train_pool}.jsonl`, `test_ids.txt`, `manifest.json` (yield, training pairs per family, cue metrics, licences,
pairing), `tasks/`, `dedup.jsonl`, `seeds/<source>/_rejected.jsonl` (written when the hermetic conversion refused a seed).

- **Atomic**: every dataset file is written into a temporary sibling `.<out>.build-<pid>` and renamed into `--out` after every gate
  passed, manifest last. A failed gate or crash leaves no dataset file behind and a previous complete dataset stays. `seeds/`,
  `tasks/` and `raw/` are caches a retry reuses.
- **With Docker** the build runs five steps, all resumable: build, validate, rebuild (relabels harmless `remove_input` variants as
  controls, quarantines leaks), validate again, final rebuild. **Without Docker** it builds unvalidated and says so; re-run on the
  Linux node before training.
- **Frozen after the first training run**: the salt (`--split-seed`) and the cutoffs. Copy `data/test_ids.txt` to
  `configs/test_ids.txt` and commit it (`data/` is git-ignored); `build_dataset.sh` picks that file up automatically.
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

- Whether a plausible non-empty recreation passes a `remove_input` variant (the Docker repair probe only tries an empty one).
- Whether `remove_tool` can yield anything on a source other than ET.
- A full 1,500-seed build to size `remove_input` and to set the cue-gate band: with 50 to 60 pairs the 0.4 to 0.6 band is only about
  two standard errors wide.
- Whether to train `no_network` (owner decision, section 2).
- The oracle-validation pass rate and the real leak rate: the audit-set gate is at least 95% of variants validating as labelled.
