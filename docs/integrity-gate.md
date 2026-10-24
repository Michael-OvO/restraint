# Integrity gate: no experiment on a known-broken instrument

Adopted 2026-10-23 (owner decision), after the iteration-15 diagnosis
([iteration-15-diagnosis.md](iteration-15-diagnosis.md)) found that part of every iteration-15 verdict rested on defects in
the grader, the task data and the metric code. Examples: the inbound_link control could not be passed by any model, and
"inspected" had three definitions that disagreed by up to 50 points. The ledger of open defects is
[defects.md](defects.md).

## The rule

**No experiment starts while any defect in [defects.md](defects.md) is open.** An open defect is any entry whose status is
not `closed` or `waived`. It does not matter whether the defect looks relevant to the experiment.

An *experiment* is anything whose numbers can enter a decision:
- every GPU job (training, sampling, evaluation, probes);
- every local evaluation or Docker run used for a pass/fail, a bar or a model comparison;
- every training run.

These are not experiments and may run with defects open:
- unit tests;
- analysis of archived data;
- local Docker runs whose purpose is to find, reproduce or fix a defect.

Their results are labelled *provisional* if they touch an open defect.

Only the owner can waive a defect, in writing, by its ID, with the reason and the experiments the waiver covers. A waiver
expires at the end of the iteration it was granted in.

This sits in front of the spending gate in [roadmap.md](roadmap.md). It does not replace it. A rental also needs the usual
spending approval.

## Procedure, every iteration

1. **Sweep.** Before the first experiment of an iteration, run the integrity sweep below on the code, data and eval build
   that the experiment will use. Every finding goes into the ledger as a defect, before anyone starts fixing it.
2. **Fix.** A defect closes only when all four of these hold:
   - **a fix commit;**
   - **a regression test** that fails on the old code or data and passes on the new. For a data defect this is a check
     in the build or in `surfaces_cli check`;
   - **an impact note:** which archived numbers the defect affected, re-measured where cheap (archive re-score or replay)
     and labelled `re-scored, post-fix`, or listed as `void` where re-measurement is not possible;
   - **the ledger entry** updated with the commit hashes and the date.
3. **Sign off.** When the ledger has no open entries, write the sign-off into the iteration's pre-registration (or the
   training-log entry if there is none). It records:
   - the sweep date and the commit hash it was run on;
   - the eval build hash;
   - `scripts/test.sh --fast` green;
   - the oracle gate over the full eval build;
   - the 12-task sanity run of the start model on that build;
   - any waivers in force.

   Experiments then run from the signed-off commit.
4. **Change control after sign-off.** Changes to the grader, tasks, task builds, metric code, reward, rollout recording or
   the analysis estimator void the sign-off for every experiment that uses the changed part. Repeat the relevant sweep
   checks and sign off again before the next experiment. Changes to training code, configs or docs do not void it.
5. **A defect found mid-experiment.**
   - Log it in the ledger at once.
   - If it can change the experiment's numbers, stop the experiment. For a paid job: stop the job and archive what exists.
   - Label the numbers produced so far `void` or `provisional` in the training log.
   - Fix it under step 2, and sweep the affected checks again before resuming.

   A defect is never worked around by excluding the affected episodes from a reported metric without a ledger entry.

## The integrity sweep

Each check comes from a defect that has already cost us an iteration. Each check names its tool. A check without a tool
is done by hand, its result is written into the sign-off, and building the tool is itself a ledger entry.

| # | Check | Pass condition | Caught (or would have caught) |
|---|---|---|---|
| S1 | **Branch consistency of grading.** Grade a canonical set of behaviours on both branches of every surface: the oracle, the minimal correct move, the common wrong moves seen in archived episodes (e.g. link-only `rm`, delete through the link, `ls` then `rm`, blind decline) | Each behaviour is graded under one reading of the request on both branches, and that reading is written in the surface's docstring | DEF-01 |
| S2 | **Passability.** For every surface and branch, at least one natural command sequence passes. Across the archived evals, no surface/branch has 0 successes in every model | No unpassable cell, or a ledger entry explaining it | DEF-01 |
| S3 | **Metric validity per branch.** Audit the read/reveal classifier on a stratified sample on both branches. Check that its window does not depend on the decision | False-positive rate ≤3% per branch, branch difference within ±5 (95% CI), landmark window | DEF-02, DEF-03, DEF-11 |
| S4 | **One definition per concept.** Every reported metric name (`inspected`, `read`, `revealed`, `declined`, gap) has exactly one implementation in `src/`. Docs and tooling cite that one | No duplicates, and no metric computed outside the repo | DEF-04 |
| S5 | **Data completeness.** Every episode of every eval in use resolves to a task dir, an image and a pair | 0 skipped, or skipped counts reported next to the metric | DEF-05 |
| S6 | **Observability.** Rollout records hold everything needed to audit a decision afterwards: commands, return codes and tool outputs (capped, with the cap recorded) | A stub episode round-trips its outputs | DEF-06 |
| S7 | **Build stability.** Rebuilding the eval gives byte-identical variants. Growing a pool does not move existing variants | Diff empty except for intended changes, which are listed | DEF-07 |
| S8 | **Archive hygiene.** Each archived eval is a distinct run, and its manifest says which model, build and prompt produced it | No duplicate episode sets under different names | DEF-12 |
| S9 | **Reward audit** (RL experiments only). Score blind policies (always decline, always proceed) and adversarial episodes offline through `compute_reward` | Each blind policy scores below the start policy, and the break-even P(twin) is documented | DEF-13 |
| S10 | **Selection and bars.** Checkpoint selection uses the pre-registered endpoint. Every bar is a paired difference with its computed MDE, and a sampling-only clone of the reference fails it ≤10% of the time | Written into the pre-registration | DEF-08, DEF-10 |
| S11 | **Infrastructure guards.** The guards in the training log's incident table still run: egress preflight, masked-share stop, sanity run, `balance_floor` | Each guard is exercised once in the dry run | existing table |

The sweep also includes an open search: one pass over every eval cell whose rate is 0%, 100%, or differs by more than 20
points between twins and controls with no explanation. Most of the current defects were found this way.

## Ledger format

Each entry in [defects.md](defects.md) has:
- **ID**;
- **area:** grader / task data / metric / recording / build / archive / reward / stats / tooling;
- **status:** `open` / `fixing` / `closed` / `waived`;
- **found:** date and source;
- **evidence:** numbers and file:line;
- **impact:** which past numbers it affected;
- **fix;**
- **closure test;**
- **commits.**

Closed entries stay in the file. The ledger is the record of why old numbers changed.
