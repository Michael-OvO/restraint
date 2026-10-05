# Leak Register: Honest Agents training data

**Bottom line:** No blocker was found. No hidden tests, oracle, task.toml or label column reaches the policy, and no test or benchmark content is in train. One real shortcut (L1) lives inside training data and should be fixed before the paid run. A second item (L2) bites only on a rebuild on the node, and two more (L3, L4) are cheap guards. All checks were static or on real Endless Terminals (ET) data (150 seeds, 846 to 5,136 variants). Docker was not available.

## 1. Verdict per channel

- **Agent-visible: clean, with one fix.** All 443 rows share one system prompt and an empty user turn. The 450 real ET Dockerfiles contain 0 COPY/ADD lines (reproduced by the sceptic). The env passes no mounts, and tests are uploaded only at grading into a random /tmp dir. Variant names appear only in boot-error text and possibly in container names (L3).
- **Split: clean today, fragile.** pair_id overlap across train/val/test is 0. Held-out families and their controls are absent from train and val, and no training code reads test. The risk is a rebuild without split_seeds.json (L2).
- **Benchmark: clean today, unguarded.** Over 715 benchmark texts (TB2, TBLite, Agentic Abstention rewrites, AgentAbstain) against 441 ET instructions and 856 variants, 0 are flagged at thresholds 0.5 down to 0.1. The dedup guard is off by default and `benchmarks/` does not exist (L4).
- **Structural cue: leaking in one family.** For conflicting_constraint, "forbidden path exists" predicts the label at about 94 to 98% (L1). For all other families, instruction text separates impossible from control at AUC 0.50 (byte-identical in 888, 606, 186 and 6 pairs).
- **Answer: clean.** There are no tests or solutions in the image. Test constants are derived from input data, so the agent has to do the task to get them.
- **Label/metadata flow: clean.** Dataset columns never reach the prompt, because TRL appends only the `reset()` observation. Reward-invariance property tests passed. Checkpoint selection uses val only.
- **Needs Docker:** container and DNS names, environment variables, /proc mounts, atime and no-network behaviour (section 4).

## 2. Leaks, ranked (severity after the sceptics)

**L1. conflicting_constraint: path-existence shortcut. MAJOR, confirmed on real data.**
The control forbids an input that already exists, while the infeasible variant forbids a required output that does not exist yet. Rerun by the sceptic:
- Forbidden path present at boot: control 426/432 (98.6%), infeasible 44/432 (10.2%).
- The cc "absent means impossible" rule gets 0.935 accuracy and AUC.
- Impact: this family is about half of the training infeasible rows (36 of 68, or 26 of 52). A policy can learn `test -e` instead of reading the conflict, and it will over-abstain on harmless constraints.

Fix:
- (a) Add controls that forbid a non-existent, non-required sibling path.
- (b) Add infeasible variants that forbid a required output that pre-exists as a stub.
- (c) Write the existence AUC per family into manifest.json and refuse the build outside 0.4 to 0.6.

Effort: hours to 1 day.

**L2. Test seeds move on a rebuild without split_seeds.json. Rated major by the auditor and minor by the sceptic; I treat it as major once you rebuild on the node.**
`assign_seed_splits` ranks seeds within whatever pool is present, and `build_dataset.sh` has no split-file variable. A simulation (the sceptic did not rerun it) puts 78 of 300 old test seeds into train at LIMIT=2492. Nothing is wrong today because `data/` does not exist yet.
- Fix: make the split `hash(seed, group) % 10000` against fixed cutoffs. Commit test_ids.txt with its hash. Have train.py refuse any train or val pair_id found in it. Add SPLIT_SEEDS_FILE to both scripts.
- Effort: hours.

**L3. Variant id, operator and `__control` in Docker names and error text. MINOR (downgraded from major by the sceptics).**
- The session id is `<variant_id>__<uuid>`, used as the compose project name. A boot failure shows the model the compose error with that name. The episode is masked, so no reward flows.
- Reverse-DNS discovery inside the sandbox is unverified.
- Same family: env.py:213-214 returns the exception text for mid-episode errors, unmasked.
- Fix: use an opaque `ep<uuid>` id in `harbor_names`, and return a fixed "[sandbox unavailable]" string in `_wait_boot`, `bash()` and the trainer. Update tests/test_env.py:165. Effort: hours.

**L4. Decontamination off by default. MINOR now, major if SETA or tmax sources are enabled.**
- The dedup guard warns instead of failing.
- A SETA task carries the Terminal-Bench canary GUID. One of 8 cached tmax tasks ships a `_perturbation.json` containing the expected output into its image.
- ET is clean today, with 0 hits.
- Fix: make `--benchmark-dir` mandatory. Add `fetch_benchmarks.sh` with pinned SHAs (tb2 2fd12b8, tblite 5c37b41). Reject any seed whose files contain a canary or a COPY/ADD of tests, expected files or task.json. Delete task.json, test_initial_state.py and container.def from `environment/`. Keep seta and tmax off until then. Effort: hours.

**L5. Latent label strings. MINOR.**
Variant Dockerfiles carry comments such as `# honest-agents: high_risk abstain twin` (888 abstain, 888 act). They are not in the image today, so this is safe. Strip them and assert that no emitted file matches /abstain|control|twin|decoy|impossible/.

**L6. Split guards accept rows with no split column. MINOR.**
This applies to the pilot, the difficulty gate and train.check_rows. It needs an operator mistake. Require split == 'train' in all three places.

**L7. Eval-sourced quarantine entries change test.jsonl on rebuild. MINOR.**
Reproduced: 232 to 230 rows. Ignore `source='eval'` entries by default.

**L8. Held-out families are described in the system prompt. MAJOR as a claim, not a leak.**
The prompt names missing permissions and irreversible actions, which are the two held-out mechanisms. Say "held-out variants, prompt describes the behaviour", and run the neutral-prompt arm.

**Smaller items:**
- The mixture gives a 55% infeasible prior per family, and feasible rows come from unpaired sources (AUC 0.67, but accuracy equals majority). Draw infeasible and control rows as pairs.
- Qwen3.5-4B post-dates ET and TB2. Report deltas between arms only, and add Qwen3-4B as a control.
- The no-network sandbox still answers ICMP and DNS (fidelity only).
- The remove_tool control keeps the /etc/alternatives link (latent, 0 variants yet).
- The tmax-4b reference model may overlap our test set (unverified).

## 3. Task quality

Sceptics partially confirmed these. Rates are from the auditors' real-data runs.

| Operator | Bad-variant rate | Rule change |
|---|---|---|
| conflicting_constraint | Paragraph breaks the sentence or list in 51.4% (sceptic rerun). Control oracle mutates the forbidden path in 4-5 of 100. | Insert only after a sentence-final paragraph, not after a heading, `:` or list. Reject paths that are mv/rm/redirect operands in the oracle. |
| remove_input | 11/34 files under 40 characters, 14/31 with 3 or fewer lines. | Floor of 80 characters and 4 lines, token overlap with the instruction under 0.5, no link-only targets. Run the repair probe. |
| revoke_permission | 6/101 sources have unreadable inputs. 36/101 lock all of /home/user. | Skip owner-only sources. Lock only the target. |
| high_risk | Scope clause ("no other files") in 18.9% of sources. Only 8 train and 4 test sites, separable by filename and README. | Skip scope-clause sources. Grow the site pool to 30 or more with randomised names. |

Other quality notes:
- 21/150 sources are copy-from-instruction. Report pass rates with and without them.
- Real yield per seed (of 150): high_risk 148, revoke_permission 101, conflicting_constraint 72, remove_input 31, no_network 1, remove_tool 0. Training therefore has only two infeasible families (the sceptic called this a 150-seed artefact; recount on a full build).

## 4. Needs Docker (run on the node)

Run these inside an episode container, once per operator and control. The task id or an operator name must appear in none of them.
```
cat /etc/hosts /etc/resolv.conf; hostname; cat /proc/self/cgroup
getent hosts $(hostname -i)
python3 -c "import socket;print(socket.gethostbyaddr(socket.gethostbyname(socket.gethostname())))"
env; cat /proc/1/environ | tr '\0' '\n'
ls -la / /tmp; ls -d /tests /solution /logs /app 2>&1
grep -v overlay /proc/self/mountinfo | head
find / -xdev \( -name task.json -o -name 'test_*' -o -name solve.sh -o -name task.toml \) -not -path '/usr/*' -not -path '/proc/*'
```
- Under deny-all: `curl -sS -m5 https://pypi.org; ping -c1 8.8.8.8; getent hosts pypi.org`.
- Compare `stat` and `ls -lu` on the high_risk twins right after boot (atime).
- Check that `test -e <forbidden path>` agrees with the static L1 estimate.
- Run the Docker repair probe on remove_input.

## 5. Regression tests to add, and timing

Add these to the repo:
1. Prompt column constant; `reset()` returns only instruction.md.
2. `harbor_names` carries no variant id; the boot-failure text is generic.
3. Existence AUC of the forbidden path is in [0.4, 0.6] per family.
4. No COPY/ADD source, canary or label word in any emitted `environment/` file or comment.
5. Held-out families and controls are absent from train, val and train_pool, and test holds both.
6. Rebuild with a fresh `--out` leaves the test ids unchanged. train.py refuses test_ids.
7. The build fails without `--benchmark-dir`.
8. Pilot, difficulty and train.check_rows reject non-train or missing splits.
9. An eval-sourced quarantine entry leaves test.jsonl unchanged.
10. No insertion breaks a sentence or list.

**Delay the first paid step.** Fix L1, L2, L3 and L4 first, about one day in total. L1 is the only item that biases the result inside train, and L2 invalidates it if the node rebuilds. Everything else can follow. Where the auditors and sceptics disagreed (L2, L3, L4 and the claim limits), the sceptics' lower severity reflects how likely each is to bite today, and I kept the fixes because they are cheap.
## 6. Status after the fix pass (2026-10-05)

Measured on the cached real Endless Terminals (ET) seeds: 300 distinct source tasks, 66 test, 224 train, 10 val under the hash split (`manifest.json` of an end-to-end build). Nothing here ran in Docker.

| Item | Status | Proof (file, test) |
|---|---|---|
| L1 path-existence shortcut | **fixed**; one semantic cue remains by design (see 6.1) | `tasks/perturb.py` (`_cc_plan`, `_borrowed_sibling`, `_cc_edit`), `tasks/analysis.py` (`cue_classifiers`, `ENFORCED_FEATURES`); `tests/test_data_leakage.py::test_every_family_has_a_neutral_existence_cue`, `::test_forbidden_path_existence_rule_is_at_chance_on_the_emitted_tasks`; `tests/tasks/test_tasks_perturb.py` (cc tests), `tests/tasks/test_tasks_analysis.py` (cv tests) |
| L2 test seeds move on rebuild | **fixed** | `tasks/build.py` (`split_bucket`, `write_test_ids`, `check_no_test_leak`); `tests/test_data_leakage.py::test_rebuild_at_a_different_pool_size_keeps_the_test_ids`, `::test_a_frozen_test_id_stays_in_test_and_a_tampered_file_is_refused`, `::test_train_accepts_the_built_rows_and_refuses_a_test_id` |
| L3 names and error text | **fixed in the training path**, Docker unverified; OpenEnv `Session.start` now returns the fixed string to the client | `harness.py` (`harbor_names`, `SANDBOX_UNAVAILABLE`), `trainer.py` (`_tool_error_text`), `openenv_env/server/environment.py`; `tests/test_data_flow.py` |
| L4 decontamination | **fixed** (benchmark dir mandatory, canary and COPY screen, `scrub_environment` now also runs on every emitted variant, so a `local` seed's `task.json` cannot ship) | `tasks/build.py::check_benchmark_config`, `tasks/taskfs.py`, `tasks/perturb.py::_emit`; `tests/test_data_leakage.py::test_the_build_fails_without_a_benchmark_dir`, `::test_no_emitted_environment_file_holds_the_expected_final_state` |
| L5 label strings | **fixed** | `perturb.scrub_labels`, `analysis.label_leaks`, `build.emitted_label_leaks` (every variant, feasible included; dropped variants counted in `manifest.json`); `tests/test_data_leakage.py::test_no_label_word_in_any_emitted_environment_file_or_comment` |
| L6 split guards | **fixed** (`train.check_rows`, pilot, `difficulty.py`) | `tests/test_data_flow.py::test_check_rows_train_requires_split_train`, `::test_pilot_refuses_non_train_and_missing_splits`, `::test_difficulty_counts_only_train_rows` |
| L7 eval quarantine | **fixed**: eval, test and openenv sources are ignored by default | `bank.DEFAULT_IGNORED_SOURCES`; `tests/tasks/test_tasks_build.py::test_eval_sourced_quarantine_entries_leave_test_unchanged_unless_asked`, `::test_held_out_and_openenv_quarantine_entries_leave_test_unchanged` |
| L8 claim limit | **documented** (not a code fix) | `docs/EVAL.md` section 1, `docs/proposal.md` 7.1 |
| Pairing | **fixed** (`balance_mixture(paired=True)`) | build manifest `pairing`: cc 44/44, remove_input 21/21 in train |
| Task quality (section 3) | **fixed**, static only | `tests/tasks/test_tasks_perturb.py`; 20 variants per family read by hand, no bad one found |

Regression tests of section 5: (1,2) `tests/test_data_flow.py`; (3,4,5,6,7,10) `tests/test_data_leakage.py`; (8) `tests/test_data_flow.py`; (9) `tests/tasks/test_tasks_build.py`.

### 6.1 conflicting_constraint after the fix

The old design (control forbids an existing input, variant forbids a missing output) is gone. Now a pair shares one image and one instruction up to one path in the constraint sentence. The task names a deliverable D. A second path D' (same directory and extension, a name recombined from the words of other tasks' deliverables, never a copy of one) is introduced as a stale leftover that nothing uses, in both variants, with as many mentions as D has. The impossible variant forbids D, the control forbids D'. When a stale stub of D is planted, an identical stub of D' goes in beside it, so `ls`/`stat`/size agree.

| cue (conflicting_constraint, held out by source) | before | after |
|---|---|---|
| exists at boot, AUC | 0.007 (70 pairs, 150 sources, old code) | 0.50 |
| name-token naive Bayes, grouped CV | 0.82 (35 pairs, 300 sources, previous design) | 0.49 (57 pairs) |
| structure logistic regression (depth, extension, mentions, position, size, name length), grouped CV | 0.74 | 0.46 |
| mentions in instruction, AUC | 0.66 | 0.50 |
| position, depth, extension, named, size | 0.50 | 0.50 |
| whole-instruction bag-of-words, source-grouped CV | 0.51 | 0.51 |

The cross-validated classifiers are enforced by the build (`--enforce-cue-band` is the default), with folds equal to donor groups (`analysis.donor_group`), so a control's borrowed words never come from the fold being tested. Sample sizes are small: 57 pairs from 57 source clusters (about 0.05 standard error per AUC), so a band of 0.4 to 0.6 is only about two standard errors wide until a full build supplies several hundred pairs.

**Remaining, not removable:** the label is whether the forbidden path is the one the task requires. A classifier given only the lines that mention the forbidden path reaches AUC 1.0 (it was 0.77 before), because the control's path is introduced as stale. That is the conflict itself in lexical form, and the opener and tail pools (8 and 6 train, 3 and 2 test phrasings) are there so no single template is a marker. A policy can learn "forbidden path described as unrelated means proceed". Held-out test phrasings differ, but the generator is the same, so this is a construct-validity limit to state next to any cc number.

### 6.2 Yield per family (300 sources, all as train pools)

| family | pairs now | per 150 sources now | per 150 sources before |
|---|---|---|---|
| high_risk | 211 | 105 | 147 |
| revoke_permission | 192 | 96 | 96 |
| conflicting_constraint | 57 | 28 | 70 |
| remove_input | 25 | 12 | 34 |
| no_network | 3 | 1 | 1 |
| remove_tool | 0 | 0 | 0 |

Under the default held-out families, training has two infeasible families (conflicting_constraint, remove_input) plus rare no_network; high_risk and revoke_permission are test only. (This was the integrator's default; superseded by the decision below. Current yields: `docs/DATA.md`.)

## Independent verification (2026-10-05) and the resulting decision

An independent verifier re-measured the fixed pipeline on 170, 261 and 335 real cached Endless Terminals seeds with its own scripts (grouped cross-validation by source, shuffled-label null, bootstrap intervals over pairs). It did **not** fully confirm the integrator's report.

**Clean:** split stability (pools of 170 and 335 sources give 0 changes on shared sources), held-out families and their controls absent from train and validation, near-duplicates (word 3-gram Jaccard under 0.05 for all test seeds), benchmark overlap (max Jaccard 0.0094 over 912 texts), emitted environment files (no label word, canary or expected-state field), the prompt column, twin structure, and the 50/50 pairing.

**Weak: conflicting_constraint.**
- The decoy path is introduced as stale or unrelated, so a rule on the forbidden path's own sentence gets 86% accuracy, and a classifier that sees only the lines mentioning the path gets AUC 1.0.
- The fraction of the path's name tokens found in the task's original instruction gives AUC 0.99 on train and 1.00 on test: the required path comes from the instruction, the decoy is borrowed from other tasks.
- The enforced cue gate failed on a 335-seed pool (name-token AUC 0.634, 52 pairs) while passing at 261 and 170 seeds, so "every enforced feature in band" depends on the pool. The 0.4 to 0.6 band is only about two standard errors wide at this size.
- A failed build left partial output files on disk.
- Low severity: decoy names are longer and have more digits; perturbed rows end with a newline and have trailing spaces stripped while plain rows do not; `no_network` has one training pair.

**Decision (taken while the owner was away; revisit freely).** Families whose twin and control differ only in the environment have no text cue, so they train. Families with a lexical cue or too few pairs are held out and used as zero-shot transfer tests, with the construct-validity caveat and without headline claims.
- Default training families: `high_risk`, `revoke_permission`, `remove_input`.
- Default held-out families: `conflicting_constraint`, `no_network`.
- The build is atomic and a held-out family can never block it.
- The cue gate is a hard refusal only for training families.

### Implementation of the verifier decision (data component, 2026-10-05)

Done in `tasks/build.py`, `scripts/build_dataset.sh`, `scripts/build_audit_set.sh`, `tests/tasks/test_tasks_build.py`, `tests/test_data_leakage.py`; the narrative is in `docs/DATA.md`.

| verifier finding | action | proof |
|---|---|---|
| cc has a lexical cue; no_network has one pair | default held-out families `conflicting_constraint,no_network`; `high_risk`, `revoke_permission` train | `build.DEFAULT_HELDOUT_FAMILIES`; `test_default_heldout_families_are_the_two_with_a_cue_or_too_few_pairs` |
| cue gate depends on the pool | hard refusal only for training families; held-out cue reported (`cue_metrics.heldout_violations`, caveat in the manifest) | `test_the_cue_gate_refuses_a_training_family_but_only_reports_a_held_out_one`, `test_a_held_out_family_with_a_cue_never_blocks_the_build_but_a_training_one_does`, `test_held_out_families_have_their_cue_reported_not_enforced` |
| a failed build leaves train.jsonl and test.jsonl | atomic build through a temporary sibling directory, manifest published last | `test_a_build_that_fails_a_gate_leaves_no_dataset_files`, `test_a_crash_while_writing_leaves_no_dataset_files_and_a_retry_succeeds`, `test_a_failed_rebuild_keeps_the_previous_complete_dataset` |
| perturbed rows end in a newline and lose trailing spaces, plain rows do not | every instruction, the plain copy included, goes through `taskfs.normalise_instruction` | `test_every_variant_formats_its_instruction_like_the_plain_copy` |
| too few pairs per family | `manifest.training.pairs_per_family`, warning under 40 | `test_the_manifest_reports_training_pairs_and_warns_about_a_thin_family` |
| decoy names longer and with more digits | not changed: cc is held out and test-only; caveat stands | `docs/DATA.md` section 7 |

The strict leakage tests stay strict for every training family (`test_every_training_family_has_a_neutral_existence_cue`:
the same thresholds, features and classifiers as before, for high_risk, revoke_permission and remove_input); the held-out
families are asserted absent from train, val and train_pool and present in test. This is a design change, not a weakening.

Cue measurements on the new default mix, 335 distinct real cached ET seeds (74 test, 10 val, 251 train by the hash split;
`build` with `--no-enforce-cue-band`, so the numbers are measured, not gated). Pairs in the split, then AUC of every
enforced feature and of both grouped-CV classifiers (name tokens, structure):

| split | family | role | pairs | every feature and classifier AUC |
|---|---|---|---|---|
| train | high_risk | train | 45 | 0.50 |
| train | revoke_permission | train | 44 | 0.50 |
| train | remove_input | train | 14 | 0.50 (thin: under 40 pairs) |
| test | high_risk | train | 49 | 0.50 |
| test | revoke_permission | train | 43 | 0.50 |
| test | remove_input | train | 3 | 0.50 (not judged, too few) |
| test | conflicting_constraint | held out | 11 | 0.50 (cv structure 0.44); reported only |

The twins of the training families have a byte-identical instruction, so every text feature is exactly 0.5 by
construction. The verifier's own classifiers (`verify/b_clf.py`, hashed word and char n-grams, source-grouped CV, shuffled
null) on the same build: instruction impossible against control 0.50 for every family and split (cc test 0.51 word, 0.43
char, 11 pairs); impossible against the plain feasible copy 0.70 (train, all families) and 0.76 (test), the same for
controls against feasible (0.70, 0.75), so the perturbation marker is shared by both classes and carries no label; the
Dockerfile-diff classifier gets 0.94 to 1.00, which is the environment cue itself (what the agent must discover by
looking), not a text cue. Shuffled-label null means 0.44 to 0.53. Docker was not available: nothing here ran in a sandbox.
