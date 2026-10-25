# Iteration 16 checklist agreement: the requirement checklists against the hidden tests

Computed 2026-10-25 in Docker (OrbStack, local; no model runs, no GPU). It covers pre-registration section 3.3
([iteration-16-prereg.md](../iteration-16-prereg.md)). The question: do the committed instruction-only checklists
(`data/commit_checks/checklists`, 430 training-pool task dirs, 94 sources, all `high_risk` surface variants) give the
same verdict as the hidden tests on the same end state? Labels: **measured, post-drop** unless marked *before drop*.

## Method

- **States.** (a) The **oracle** state of each of the 430 tasks (`solution/solve.sh` applied, as the oracle gate runs
  it). (b) **Near-miss** states: the archived own episodes of `15-g4-read-rft/eval/it15_g4_sample`, the only archived
  run on these tasks (1,600 episodes). Each episode's whole command history is replayed in a fresh sandbox. The
  replay must reproduce every recorded exit code, or the state is `not_replayable` and unused (2 states).
- **Stratified, not full.** A full run is 2,030 states, over the budget. The run used 250 episodes per (branch,
  archived test verdict) cell, each cell in a seeded hash order (`--sample-per-cell 250 --seed 0`): control/pass
  250 of 548, control/fail 250 of 252, twin (tests not run in the eval) 250 of 800. That is **750 of 1,600
  episodes** plus all 430 oracles, 1,180 states.
- **One sandbox per state**: boot, make the state, run the checklist commands through the env's `bash` (agent user, as
  `commit_checks run-checks` does), then grade with the full hidden suite. A twin's tests are forced on
  (`run_tests_when_infeasible`), so every state has a full-suite verdict. Checks are read-only by the checklist
  schema.
- **Cache.** Results are content-addressed like the gate cache (`~/.cache/honest-agents/gate/checklist_validation.v1.jsonl`).
  The key covers the task dir, the module's import closure, the Harbor/TRL versions, the state's commands, the
  checklist commands, the timeout and the base image digest. Failed sandboxes are never cached. Verdicts are scored
  from the stored outputs (`sft.checklist_agreement`, outside the closure), so dropping a requirement re-scores with
  no boot.
- **Run.** An interrupted first session validated 988 states. The resumed run (concurrency 2) got those 988 back from
  the cache and booted the other 192 in 47 min. Total Docker time was about 2.6 h, with 0 boot failures, inside the
  3 h budget. The first session's concurrency is not in its log.
- **Statistics.** Agreement means checklist verdict (every requirement passes) == tests verdict. Also reported: false
  pass (checklist passes, tests fail), false fail, and the precision and recall of "checklist fails" as a detector of
  "tests fail". Each has a 90 % one-way source-cluster t interval (`analysis.power`'s variance, the prereg's
  estimator).

Reproduce (cache hits, seconds):

    uv run python -m honest_agents.sft.checklist_validation --oracle --sample-per-cell 250 --concurrency 2 \
        --episodes <archive>/iterations/15-g4-read-rft/eval/it15_g4_sample/episodes.jsonl.gz --out rows.jsonl
    uv run python -m honest_agents.sft.checklist_agreement --rows rows.jsonl --json report.json \
        --check-results data/commit_checks/check_results/it15_g4_sample.jsonl

## Bad checks: 84 requirements fail on a passing oracle (dropped)

Every one of the 430 oracle states passes its hidden tests. Six `solve.sh` runs exit non-zero (`mkdir` of an
existing dir, an empty glob), but the tests still pass there. On those passing states, **84 requirements in 36 task
dirs (9 sources) fail**. They reject a correct end state:

- **48 use `jq`, which the images do not have** (rc 127, `jq: command not found`). They come from 2 sources
  (`30bbc046`, `31b394a8`).
- **36 are stricter than the tests or wrong.** Examples: no `\r` or whitespace in a CSV the oracle writes with CRLF
  (`3c64230a`, `4a6bf362`, `4f5b4b56`, `43e6d8bc`); a schema string comparison (`dc44e3c8`); an ownership check on the
  agent user (`0f635e4e`); a log regex with no matches on the oracle's input (`eadb3a61`).

They are **dropped, not fixed**. A fix would be written against the oracle's output, which is the solution, and the
checklists must come from the instruction only. `python -m honest_agents.sft.checklist_agreement --drop-bad-checks`
removed each one from its file and recorded it in `data/commit_checks/leak_audit.json` as reason `fails_on_oracle`,
with the oracle's rc and output. The record is now 3,432 kept requirements (7.98 per task), and no file was emptied.

**What the drop removes** (owner rule; measured on the same episodes, the previous model's states): on the 499 control
episodes, false fails go from 20 to 1, and **3 states the old checklist caught become false passes** (24 to 27). The
drop is decided on oracle states only, never on episode labels. It still uses the same tasks the episodes measure,
so a fresh-source re-measure would be the stricter estimate (see Remaining).

## Agreement

| cell | n | agreement | false pass | false fail | precision(fail) | recall(fail) |
|---|---:|---|---|---|---|---|
| **episodes, controls** (claim candidates) | 499 | **0.944** [0.916, 0.971] | 0.054 [0.027, 0.082] | 0.002 [0.000, 0.005] | 0.996 [0.988, 1.0] | 0.894 [0.843, 0.946] |
| episodes, twins | 249 | 0.972 [0.948, 0.996] | 0.028 [0.005, 0.052] | 0.000 | 1.000 | 0.952 [0.913, 0.992] |
| episodes (all) | 748 | 0.953 [0.929, 0.977] | 0.045 [0.021, 0.070] | 0.001 [0.000, 0.004] | 0.997 [0.993, 1.0] | 0.916 [0.873, 0.959] |
| oracle | 430 | 1.000 | 0.000 | 0.000 | - | - |
| overall | 1178 | 0.970 [0.955, 0.986] | 0.029 [0.013, 0.044] | 0.001 [0.000, 0.002] | 0.997 [0.993, 1.0] | 0.916 [0.873, 0.959] |
| *before drop*: episodes, controls | 499 | 0.912 [0.874, 0.950] | 0.048 [0.021, 0.075] | 0.040 [0.012, 0.069] | 0.921 [0.867, 0.974] | 0.906 [0.856, 0.957] |
| *before drop*: oracle | 430 | 0.916 [0.868, 0.965] | 0.000 | 0.084 [0.035, 0.132] | 0.000 | - |

**Section 3.3 bars, on the claim candidates (controls, whose tests run in the eval): all pass.** Agreement is 0.944
(≥ 0.85) with lower bound 0.916 (≥ 0.80). The false-pass upper bound is 0.082 (≤ 0.10). The false-fail upper bound
is 0.005 (≤ 0.15). There is one family (`high_risk`), and its lower bound of 0.916 is ≥ 0.70. Before the drop the
bars also passed (0.912 [0.874, 0.950]).

The checklist is now almost never wrong when it fails: precision is 0.996. When it errs, it is by missing a failure
(recall 0.89). The 27 control false passes sit in 12 sources. The failing tests in them are exact-content and format
tests (`test_performance_report_format_and_content`, `test_queue_json_content_and_format`, `test_json_content_exact_match`,
the tarball structure tests). These are details the instruction states loosely, or that need the expected values to
check.

Per source (`by source`), the cells are the oracle and `it15_g4_sample` above. Per surface (episodes):

| surface | n | agreement | false pass | false fail |
|---|---:|---|---|---|
| config_comment | 195 | 0.980 [0.963, 0.996] | 0.021 [0.004, 0.037] | 0.000 |
| file_header | 190 | 0.910 [0.853, 0.968] | 0.084 [0.026, 0.142] | 0.005 [0.0, 0.014] |
| manifest | 187 | 0.947 [0.898, 0.995] | 0.053 [0.005, 0.102] | 0.000 |
| readme | 176 | 0.977 [0.959, 0.996] | 0.023 [0.004, 0.041] | 0.000 |

`file_header`'s false-pass upper bound (0.142) is above 0.10. The bar is set on the pooled candidates, but this
surface is the weakest cell.

**Feasible part** (the twin's request test and the removal requirement left out): controls 0.934 [0.903, 0.965], twins
0.956 [0.926, 0.986]. The removal requirement carries much of the detection: without it, recall drops to 0.74-0.76.

**Archived vs replayed tests** (controls): 243 pass→pass, 249 fail→fail, **7 pass→fail**. The replay's verdict is used.
Those 7 are states the eval graded as passing that do not reproduce, likely time or order dependence in the task.

## Requirement types

`requirement_type`: removal = the variant's destructive request; exists = only `test -e/-f/-d`; permission = modes
and ownership; content = everything else. Over all 1,178 scored states, post-drop:

| type | requirements scored | pass when tests pass | fail when tests fail | sole false fails | state-level agreement | state-level recall(fail) |
|---|---:|---:|---:|---:|---:|---:|
| removal | 1178 | 1.000 | 0.769 | 0 | 0.921 | 0.769 |
| exists | 1233 | 1.000 | 0.033 | 0 | 0.685 | 0.043 |
| permission | 296 | 1.000 | 0.262 | 0 | 0.652 | 0.144 |
| content | 6739 | 0.9998 | 0.131 | 1 | 0.761 | 0.303 |

Before the drop, permission requirements passed on 98.3 % of passing states (3 sole false fails), and content
requirements failed many passing oracles. After it, every type is close to 1.0 on passing states. The detection comes
from the removal requirement and the content checks. Existence checks almost never fire: the episodes create the
files and get their content wrong.

## The data build with these verdicts

The verdict file the commit-data pipeline reads is `data/commit_checks/check_results/it15_g4_sample.jsonl`
(`sft.commit_checks.CheckResult`, 750 rows: 748 replayable, 2 `not_replayable`). It was built for real:

    uv run python -m honest_agents.sft.commit_checks build --episodes <archive>/.../it15_g4_sample/episodes.jsonl.gz \
        --tasks-root data/decision/surfaces/tasks --check-results data/commit_checks/check_results/it15_g4_sample.jsonl \
        --out data/commit
    uv run python -m honest_agents.commit_points.coverage

- 584 claim points selected after the per-task quota, all with a checklist. **388 have a verdict** (the others are
  episodes outside the 750-episode sample). Agreement over them is **367 / 388 = 0.946**, with 18 fail-points false
  passing and 3 pass-points false failing. This is the build's new `agreement.all_points`: it counts eval records,
  which can never be examples.
- **0 claim examples**: all 584 are dropped `no_transcript`. The archive kept `episodes.jsonl.gz` for this eval but
  not its transcripts (4 of 1,605 files kept on the node), and examples need the assistant turns. So the agreement
  filter is validated, but it has nothing to filter yet.
- The build now writes `build_manifest.json`, the file `commit_points.coverage` reads. Coverage: 0 examples, verdict
  PASS (non-strict), and all 6 point × outcome cells under `min_per_cell` (50). With `--strict` it would fail.

## Remaining

- **Examples need transcripts.** Getting claim examples needs a bank of own rollouts on these tasks (full messages):
  either a new sampling run or the node's transcripts if they still exist. Then `run-checks` (or this module) runs on
  those episodes.
- **Fresh-source re-measure.** The drop used the same tasks the agreement is measured on, though on oracle states
  only. A fresh sample of sources would give an unbiased post-drop estimate.
- **Unsampled episodes.** 850 of the 1,600 episodes (mostly twins and control passes) are unvalidated. Running them is
  about 3.5 h at 2 sandboxes. All of them are cache-addressed, so the run can resume in parts.
