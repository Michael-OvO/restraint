# Iteration 16: integrity sign-off (draft, NOT signed off)

Draft written under [integrity-gate.md](integrity-gate.md) step 3. **Every automated check passes on `8d45d25`**: tests,
L0, and the full L1 run with the oracle gate, S3 and S11. Sign-off still waits on 5 ledger entries, all `fixing`. Each
needs an owner decision or a GPU, not more local work. The sanity run of the start model needs a GPU. No experiment
may start before the owner signs off.

## Recorded

| Field | Value |
|---|---|
| Sweep date | 2026-10-25/26 (L0 on the Mac, 2026-10-25 18:3x EDT; L1 on a remote node, 2026-10-25 23:18 to 2026-10-26 00:00 UTC) |
| Commit swept | `8d45d25` (main, clean tree): the merge of DEF-48, DEF-51 and DEF-57 |
| Eval build | `data/pools/build_manifest.json` sha256 `09628ed96f81f5d4efae7460945d23328bbd186d20083c8ec29116594f72b2c5`; 20 sources x 6 surfaces, test split, seed 0 |
| `scripts/test.sh --fast` | green: 4656 passed, 36 skipped, 102 s (Mac) |
| L0 sweep (`gate sweep --level 0`) | S1, S2, S4-S10 PASS; 22.4 s (budget 60 s); 0 boots; 0 bytes written (Mac) |
| Oracle gate (G1) over the full eval build | **PASS 120/120 pairs**: 600 boots, 0 cached, 2,309 s |
| S3 (metric validity) | **PASS**: it15_g2_surface4 96/96 audited, it15_sft_v2_surface4 94/96 audited; k8 and prefix windows pass; FP twin/control 0/0/0/0; replay bar met. 376 boots, 1,415 s |
| S11 (infra guards) | **PASS**: 4 guards exercised; the real egress probe passed (1 boot) |
| Where L1 ran | Vast VM 57730801 (Ubuntu 22.04 VM template, Docker 29.0.3, x86_64, 19 vCPU, 49 GB), a copy of `8d45d25` plus the two S3 evals from the archive, `--concurrency 12`. G1 and S3+S11 ran as two parallel processes. Local Docker on the Mac is limited to a few sandboxes by owner rule; a full refill overheated it on 2026-10-25. Destroyed after the run |
| L1 budget | both runs were over the 900 s L1 budget because the cache was empty, which is a full refill. A cached re-run on the same platform takes seconds |
| 12-task sanity run of the start model | pending (needs GPU) |
| Waivers in force | none |

Results and cache rows: `~/Documents/honest-agents-archive/staging/gate-node-20261026/` (`gate_g1.json`,
`gate_s3.json`, `summary.txt`, `g1_surfaces.v1.jsonl`, `replay_episodes.v1.jsonl`). The cache keys include the base
image digests, so these x86_64 rows only hit on an x86_64 node and never on the Mac's arm64 engine.

## Ledger status (57 entries)

52 `closed`. Not closed (each blocks sign-off):

| ID | Status | What remains | Needs |
|---|---|---|---|
| DEF-10 | fixing | the pre-registration (`docs/iteration-16-prereg.md`) is a DRAFT; its computed-at-build bars and DEF-20's guards are open | owner approval |
| DEF-20 | fixing | twin task_success: recorded from now on, void on the archive, and its validity check on real twins is pending; `revealed` needs pair replays of all three surface runs | remote Docker batch / GPU |
| DEF-43 | fixing | release of the 24 select variants that failed only under the old umask (the tracked 8-entry `data/select_quarantine.txt` is the proposal) | owner decision |
| DEF-46 | fixing | the guard fails closed (`2d0b492`); the six iteration-15 models' sanity runs | GPU |
| DEF-50 | fixing | AbstentionBench export ingest and a node export of the items | GPU node |

## What blocks sign-off, exactly

1. The 5 ledger entries above, or an owner waiver of each, in writing, by ID.
2. The 12-task sanity run of the start model on the eval build (GPU).
3. Owner approval of the pre-registration and of the GPU spend.

If none of the gated files change (grader, verifier, harness, task builds, metric code), the L1 results above stay
valid for the commit that closes the ledger: integrity-gate.md step 4.

## Worktrees and branches

`git branch --no-merged main`: only `design/rts` (worktree `/Users/michael/Documents/Github/restraint-rts`, RTS
scoring design from another session, not a defect fix).
