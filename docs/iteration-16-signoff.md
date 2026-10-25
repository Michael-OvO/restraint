# Iteration 16: integrity sign-off (draft, NOT signed off)

Draft written under [integrity-gate.md](integrity-gate.md) step 3. **The gate is not passed.** The ledger
([defects.md](defects.md)) still has 8 entries that are not `closed` or `waived` (2 `open`, 6 `fixing`). The oracle
gate and S3 have no valid result on the swept commit: their cached results belong to `77f7f33`, and the cached L1
re-run on `1b490a5` could not boot a sandbox. The sanity run needs a GPU. No experiment may start from this commit.
This file records what was checked and what is true as of the sweep below.

## Recorded

| Field | Value |
|---|---|
| Sweep date | 2026-10-25 (the final check ran 14:50-15:20 EDT) |
| Commit swept | `1b490a5` (main, clean tree) |
| Eval build | `data/pools/build_manifest.json` sha256 `09628ed96f81f5d4efae7460945d23328bbd186d20083c8ec29116594f72b2c5` (last changed `49e8965`, unchanged since the last sign-off draft); 20 sources x 6 surfaces, test split, seed 0 |
| `scripts/test.sh --fast` | green with Docker unreachable: 4603 passed, 36 skipped, 109 s. Run with `DOCKER_HOST` pointed at a missing socket, because the OrbStack engine stopped answering at about 14:50 (every `docker info` hung; the first plain `--fast` run hung at collection on them and was stopped with SIGINT). Docker-detecting tests skip, so 2 more skips than in a run with Docker up |
| L0 sweep (`gate sweep --level 0`) | S1, S2, S4-S10 PASS; 33.3 s (budget 60 s); 0 boots; 0 bytes written |
| L1 sweep (`gate sweep --level 1`, cached) | **FAIL (infrastructure): no sandbox booted.** G1 0/120 (0 cached, 120 run, 480 boots, every one refused: `network_mode='no-network' is not supported`), S3 ERROR (94 of 96 replays per eval did not boot), S11 FAIL (egress preflight, below). 163 s, 857 boots attempted, 0 cache hits. No result row was written to the cache (`g1_surfaces.v1.jsonl` and `replay_episodes.v1.jsonl` unmodified); the 14.2 MB written are the 68 rebuilt archive index tables |
| Why the cache missed | the G1 and replay cache keys hash their import closures, and two files in them changed after the full gate on `77f7f33`: `tasks/taskfs.py` (DEF-45, `8a79378`) and `tasks/validate.py` (DEF-43, `931ce6c`, `328abd8`). So the 120/120 G1 result and the S3 replays of `77f7f33` do not cover `1b490a5`, and the next L1 run re-boots all 120 pairs and the S3 sample (gate step 4). S7 shows the variant bytes are unchanged (240/240, 1048/1048) |
| Oracle gate over the full eval build | **not valid on `1b490a5`.** Last pass: 120/120 on `77f7f33` with DEF-47's fix (`510d8e6`), 2026-10-25, scratchpad `fix/fullgate/` |
| S3 (metric validity) | not valid on `1b490a5` (sandboxes did not boot). Last pass on `77f7f33` + `510d8e6`: it15_g2_surface4 96/96 and it15_sft_v2_surface4 94/96 audited, k8 and prefix pass, FP 0 on both branches |
| S11 (infra guards) | FAIL on `1b490a5`: the real egress probe failed 3 times (Docker engine not answering). Masked-share stop, sanity-run rule (no `--chains` passed) and `balance_floor` pass against fakes. Last full pass on `77f7f33` |
| 12-task sanity run of the start model | pending (needs GPU) |
| Waivers in force | none |

Logs: scratchpad `fix/signoff/` (`sweep_l0.json`, `sweep_l1.json`, `sweep_l1.log`, `test_fast_nodocker.log`,
`provenance.txt`).

## L0 sweep detail (`1b490a5`)

| Check | Status | Detail |
|---|---|---|
| S1 | PASS | 120/120 pairs read one way on both branches |
| S2 | PASS | 12 surface/branch cells over 68 evals; no cell with 0 successes everywhere |
| S4 | PASS | the inspection metric is defined only in `eval/read_diagnosis.py` |
| S5 | PASS | 0 task dirs missing; 3520 high_risk episodes of 8/8 evals in use, 0 unresolved |
| S6 | PASS | a stub episode round-trips commands, exit codes and capped outputs |
| S7 | PASS | eval 240/240 and decision 1048/1048 variants byte-identical; 0 bytes written (DEF-22 closed) |
| S8 | PASS | 68 episode files, 1 declared duplicate group, 0 undeclared |
| S9 | PASS (as a check) | it15_g2_surface4: the archived G3 reward FAILS the audit (adversarial and blind below start, break-even 0.39); the symmetric reward passes (break-even 0.50) |
| S10 | PASS | 10/10 estimator inputs present; L0 resamples <= 2000 |

## Ledger status (57 entries)

49 `closed`: DEF-01 to DEF-09, DEF-11 to DEF-19, DEF-21 to DEF-42, DEF-44, DEF-45, DEF-47, DEF-49, DEF-52 to DEF-56.

Not closed (each blocks sign-off):

| ID | Status | What remains |
|---|---|---|
| DEF-10 | fixing | the pre-registration (`docs/iteration-16-prereg.md`) is a DRAFT, not approved by the owner; its computed-at-build bars and DEF-20's guards are open |
| DEF-20 | fixing | twin task_success: recorded from now on, void on the archive, its S3 validity check on real twins (Docker) pending; `revealed` needs pair replays of all three surface runs |
| DEF-43 | fixing | owner decision on releasing the 24 select variants (the tracked 8-entry `data/select_quarantine.txt` is the proposal) |
| DEF-46 | fixing | the guard fails closed (`2d0b492`); the six iteration-15 models' sanity runs need a GPU |
| DEF-48 | open | `run_eval --diff` / `--noninferiority` ignore `--exclude-quarantined`; no fix yet |
| DEF-50 | fixing | AbstentionBench export ingest (new code) and a node export of the items |
| DEF-51 | open | the base model's Hub revision is pinned in no config or serve script |
| DEF-57 | fixing | the never-trained repo scope still depends on gitignored local files |

## What blocks sign-off, exactly

1. The 8 ledger entries above (or an owner waiver of each, in writing, by ID).
2. A working Docker engine, then `gate sweep --level 1` on the commit that closes them: G1 re-boots all 120 pairs and
   S3 re-replays its sample (the cache keys changed with `taskfs.py` and `validate.py`), and S11's real egress probe.
3. The 12-task sanity run of the start model on the eval build (GPU).
4. Owner approval of the pre-registration and of the GPU spend.

## Worktrees and branches

`git branch --no-merged main`: only `design/rts` (worktree `/Users/michael/Documents/Github/restraint-rts`, RTS
scoring design, not a defect fix). The worktrees `restraint-wt/{commit-data,flaky,gate-s3s11,power-prereg,probe-cov,reward}`
(all at `3d7bf08`), the two `.claude/worktrees/agent-*` and the scratchpad `fix/review-DEF-46/old` (detached
`3d7bf08`) hold no commits that main lacks.
