# Iteration 16: integrity sign-off (draft, NOT signed off)

Draft written under [integrity-gate.md](integrity-gate.md) step 3. **Every automated check passes**: the full L1 run
(oracle gate, S3, S11) on `8d45d25`, and tests and L0 again on `ad4be54` (2026-10-26 morning; the L1 result carries
over, see "L1 validity" below). Sign-off still waits on 6 ledger entries (4 `fixing`, 2 `open`). The sanity run of
the start model needs a GPU. No experiment may start before the owner signs off.

## Recorded

| Field | Value |
|---|---|
| Sweep date | 2026-10-25/26 (L0 on the Mac, 2026-10-25 18:3x EDT; L1 on a remote node, 2026-10-25 23:18 to 2026-10-26 00:00 UTC); tests and L0 again 2026-10-26 07:3x UTC |
| Commit swept | L1: `8d45d25` (main, clean tree): the merge of DEF-48, DEF-51 and DEF-57. Tests and L0: `ad4be54` (main, clean tree) |
| Eval build | `data/pools/build_manifest.json` sha256 `09628ed96f81f5d4efae7460945d23328bbd186d20083c8ec29116594f72b2c5`; 20 sources x 6 surfaces, test split, seed 0 |
| `scripts/test.sh --fast` | green on `ad4be54`: 4726 passed, 36 skipped, 100 s (Mac). On `8d45d25`: 4656 passed, 36 skipped |
| L0 sweep (`gate sweep --level 0`) | S1, S2, S4-S10 PASS on `ad4be54`; 22.5 s (budget 60 s); 0 boots; 0 bytes written (Mac). Also PASS on `8d45d25` (22.4 s) |
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

## L1 validity on the current head

The L1 results above were recorded on `8d45d25`. `git diff 8d45d25 ad4be54` (16 commits) touches none of the gated
files of integrity-gate.md step 4: no change under the grader (`grading.py`, `bank.py`), the verifier
(`verifier/`), the harness and sandbox (`harness.py`, `env.py`, `openenv_env/`), the task builds (`tasks/`,
`mutation.py`, `data/pools`, `data/decision`), the reward (`reward.py`, `trl_rewards.py`) or the surface metric and
estimator code (`eval/run_eval.py`, `eval/read_diagnosis.py`, `analysis/`). What changed: `eval/final_benchmark.py`
(the AbstentionBench export ingest and `verify-export`, DEF-50; an `agentic_abstention` branch in the units writer),
`gate/ledger.py` (the launch guard), `sft/claim_splice.py` and `sft/epochs.py` (training-data build), the four
`configs/chains/it16_*.yaml` job files, `scripts/eval_abstentionbench.sh`, the tracked `claim_splice_v1` build (0
examples), tests and docs. G1, S3 and S11 import none of these, so **the L1 on `8d45d25` holds for `ad4be54`**. The
`final_benchmark.py` change is metric code for the external benchmarks; its checks are its own tests (in `--fast`,
green) and `final_benchmark check`, not an L1 check.

## Ledger status (60 entries)

54 `closed`. Not closed (each blocks sign-off; `gate.ledger check` exits 1 on all six):

| ID | Status | What remains | Needs |
|---|---|---|---|
| DEF-10 | fixing | the pre-registration (`docs/iteration-16-prereg.md`) is a DRAFT; its computed-at-build bars and DEF-20's guards are open; the calibration-sample circularity | owner approval (and a DEF-10 waiver for `it16a` only) |
| DEF-20 | fixing | twin task_success: recorded from now on, void on the archive, and its validity check on real twins is pending; `revealed` needs pair replays of all three surface runs. The 2026-10-26 node run was refused (see the morning brief); a node rented by the orchestrating session at 07:32 UTC (Vast 57781978) was coming up when this was written | remote Docker batch (CPU node) |
| DEF-43 | fixing | release of the 24 select variants that failed only under the old umask (the tracked 8-entry `data/select_quarantine.txt`, already on main, is the proposal) | owner decision |
| DEF-50 | fixing | the AbstentionBench export must run on a node, then freeze and check on the Mac | CPU node |
| DEF-58 | open | the pre-registration promises per-(task, sample, turn) sampling seeds; no rollout sends one | code fix or a pre-registration amendment (owner) |
| DEF-60 | open | the AbstentionBench (and MASK) guard margin has no calibration source | owner decision (state m in the pre-registration, or add AbstentionBench to stage (a)) |

DEF-46 (`0ff3cf0`) and DEF-59 (`4077af4`) closed since the last draft. DEF-20: the node batch of 2026-10-26 did the pair replays and the twin validity check (`ffa9cc3`); only twin task_success's MDE remains, from the stage-(a) eval. DEF-50: the node export is blocked by the gated GPQA subset (`58018f9`).

## What blocks sign-off, exactly

1. The 7 ledger entries above, or an owner waiver of each, in writing, by ID.
2. The 12-task sanity run of the start model on the eval build (GPU).
3. Owner approval of the pre-registration and of the GPU spend.

If none of the gated files change (grader, verifier, harness, task builds, metric code), the L1 results above stay
valid for the commit that closes the ledger: integrity-gate.md step 4.

## Worktrees and branches

`git branch --no-merged main`: only `design/rts` (worktree `/Users/michael/Documents/Github/restraint-rts`, RTS
scoring design from another session, not a defect fix).
