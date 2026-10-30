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
| S12 | **Spec audit.** Every source whose archived episodes look like a hidden or contradictory requirement (nobody passes it reliably, and the failures are near misses: `tasks.spec_audit`'s rule) has a verdict from reading its instruction against its tests; defects are in this ledger and excluded downstream | No unreviewed candidate; each defect names its ledger entry; the scan is not older than the archive's eval runs | DEF-66 |

The sweep also includes an open search: one pass over every eval cell whose rate is 0%, 100%, or differs by more than 20
points between twins and controls with no explanation. Most of the current defects were found this way.

## Running the gates efficiently

A gate that takes hours gets skipped. So every check runs at the cheapest level that can answer it. Docker work runs
only for inputs that changed since the last pass. A routine sweep must not rebuild images, scan Docker, or write episode
files.

**Three levels.**

| Level | What it may do | Budget | When it runs |
|---|---|---|---|
| L0, static | Pure Python in memory: code, task dirs, pool JSON, the reward function, compact archive indexes. No Docker, no network. The only disk write is pytest's temp dir | ≤60 s for the whole level | inside `scripts/test.sh --fast`, so on every test run |
| L1, cached Docker | Boot a sandbox only for an input whose cache key is not yet in the result cache | ≤15 min after a typical change; a full run only when a key every pair shares changes | on demand (`... --changed`) and at sign-off |
| L2, full Docker | Re-run everything, ignoring the cache | overnight, unattended, under caffeinate, concurrency ≤4 | only at sign-off after a change to shared code (grader, verifier, harness, base image), and once per iteration as an audit |

**Which level each sweep check uses.**

| Check | Level | How it avoids Docker or repeats |
|---|---|---|
| S1 branch consistency | L0, plus L1 for the behaviours that touch the real filesystem | Each surface's graded reading is declared as data: `oracle_lines`, `test_lines`, `irreversible_paths`. An L0 test asserts that one canonical behaviour set gets the same reading on both branches, from those declarations alone. Docker only confirms that the declared behaviour is what `rm` really does: one pair per surface, cached |
| S2 passability | L0 | Computed from a compact archive index (below) that counts successes per surface, branch and model. No gunzip of episode files |
| S3 metric validity | L1 | Replay outputs are cached by key, so a metric change re-scores cached outputs in memory without booting anything |
| S4 one definition | L0 | AST/grep test |
| S5 completeness | L0 | Path existence against the archive index |
| S6 observability | L0 | Stub episode round-trip in memory |
| S7 build stability | L0 | Build the variants in memory and compare their hashes with a committed manifest (`data/pools/build_manifest.json`). Never write the build to disk to diff it |
| S8 archive hygiene | L0 | Content hashes stored in each `MANIFEST.sha256`; no re-hashing of archives that are unchanged since the last check (their mtime and size match) |
| S9 reward audit | L0 | `compute_reward` is pure; blind policies are scored on the archive index |
| S10 bars and power | L0 | The estimator runs on the archive index; bootstrap seeds fixed, ≤2,000 resamples at L0 and more only for the pre-registration |
| S11 infra guards | L1 | Exercised once in the dry run against the fake sandbox. Only the egress preflight needs a real container |
| S12 spec audit | L0 | Reads the committed `data/spec_audit/candidates.json` and `reviews.jsonl`; the archive is only globbed to count eval runs. `spec_audit scan` (about a minute) refreshes the candidates when new evals are archived |

**The result cache** (L1).
- **Location:** `~/.cache/honest-agents/gate/` on the Mac and `/root/.cache/honest-agents/gate/` on a node, outside the
  repo and the archive.
- **Contents:** one JSONL row per check, holding the verdict and the small outputs the check needs (capped). Never
  images, containers or full episodes.
- **Key:** sha256 of the task dir contents, plus the sha256 of the source files the check imports (grader, verifier,
  harness, `risk_surfaces.py`), plus the command list, plus the base image digest.
- **Invalidation:** a change to one task re-runs that task only. A change to the grader re-runs everything, which is
  correct, and is the case that goes to L2.
- **Integrity:** the cache is append-only and has a schema version. A corrupt or old-version row is ignored, never trusted.

**The archive index.** One compact table built once per archived eval, about 1 MB instead of a 30-100 MB unpacked
`episodes.jsonl`. It has one row per episode: ids, surface, branch, case, reward, command list hash, read labels, and the
first-destructive-command index. It also stores the source file's sha256, so a changed or new archive rebuilds only its
own index. L0 checks and archive analyses read the index, never the gzipped episodes.

**Docker hygiene.**
- Boot from prebuilt images with the build cache kept. Never `docker system prune` or a cache prune during a sweep.
- Never enumerate all images or containers in a loop.
- Tear down each sandbox as soon as its check ends, and stop jobs with SIGINT so teardown runs (training-log incident
  table).
- One concurrency setting (≤4 on the Mac).

**The tools.** `python -m honest_agents.gate sweep [--level 0|1|2] [--changed]` runs the levels (`gate.sweep`; L0 is
also `tests/test_gate.py` in `scripts/test.sh --fast`), `python -m honest_agents.gate index` stores the archive index
(`gate.index`), and `gate.cache` is the result cache that `tasks.g1_surfaces` and `analysis.replay_episodes` use. At
L1 the sweep also runs S3 (a stratified pair-replayed sample of the reference and comparison surface evals behind the
replay cache, audited by `eval.read_audit`; `--s3-sample`) and S11 (`gate.guards`: each guard once against fakes, the
real egress probe in one container, and the sanity-run rule on the job files passed with `--chains`) (DEF-23).

**Measured, not assumed.** The sweep CLI records each check's wall time, sandbox boots, cache hits and bytes written.
These numbers go into the sign-off. A check that exceeds its level's budget twice in a row is entered as a tooling defect
in the ledger: it is fixed (made cheaper, cached or moved to a lower level), not tolerated.

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
