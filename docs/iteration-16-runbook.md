# Iteration 16 node runbook: the four paid stages

**Status: prepared, not launched, not approved.** The integrity gate is shut: DEF-10 (the pre-registration is a
DRAFT) and DEF-43 (owner decision) are open, with DEF-20, DEF-50 and the two entries this runbook's lane added
(DEF-NEW-it16-jobs-1, -2). No stage below may run until [integrity-gate.md](integrity-gate.md) step 3 is signed off in
[iteration-16-signoff.md](iteration-16-signoff.md) and the owner approves that stage's spend. The job files enforce
this themselves: the first step of each one is `python -m honest_agents.gate.ledger check`, which exits 1 while any
ledger entry is open, and every other step depends on it.

The stages are those of [iteration-16-prereg.md](iteration-16-prereg.md) section 8 ("spend in stages"), each a
separate job file, launched and watched on its own:

| stage | job file | what | launchable now? |
|---|---|---|---|
| (a) | `configs/chains/it16_a_calibration.yaml` | calibration samples: base and G2 s1 on TB2 and ImpossibleBench LCB, S = 2, after a 12-task sanity run of each served model | yes, once the gate opens |
| (b) | `configs/chains/it16_b_claim_rollouts.yaml` | G2 s1 own runs on the 400 training-pool surface tasks (`data/decision/read_rft_tasks.jsonl`), S = 4, transcripts (bank) and tool outputs kept; then the checklist validation of the new end states | yes, once the gate opens |
| (c) | `configs/chains/it16_c_sft.yaml` | SFT from G2 s1 on the matched-pair build (W&B, gradient checkpointing on), two epochs, epoch chosen on the select split | no: `<SFT_CONFIG>`, `<SFT_BUILD_MANIFEST>`, `<SELECT_PROTOCOL>` |
| (d) | `configs/chains/it16_d_final_eval.yaml` | the final eval of base, G2 s1 and final on every benchmark of prereg section 4 except MASK | no: `<SURFACE_VAL_V2>`, `<BLIND_FAMILY_SPLIT>`, `<FLIP_PROMPTS>`, `<ABENCH_JUDGE_URL>` |

A `<PLACEHOLDER>` is refused by `python -m honest_agents.node chain` (without `--dry-run`) and by `node_chain run`. Each
one stands for an input that is built later (prereg sections 3-7 "computed-at-build"). All four files pass
`python -m honest_agents.node_chain check` and `node chain --dry-run`, including the sanity-run rule (DEF-46).
`tests/test_it16_chains.py` holds these properties.

Order: (a) and (b) are independent and can share one rental. (c) needs the claim-point build. The commit-data lane
makes that build from (b)'s transcripts on the Mac, at $0. (d) needs (c)'s selected adapter and the frozen eval build.
The base and G2 numbers of (a) are not reused in (d) (prereg 6.1).

## 1. Node

One Vast VM with **1x RTX PRO 6000 (Max-Q) 96 GB**, at least 23 vCPU, 64 GB RAM and 200 GB disk. This is the
iteration-15 node class (training log: instance 55983325, Japan, **$1.46/h**). Use the VM template and the bring-up
steps of [iteration-14-runbook.md](iteration-14-runbook.md) section 1: driver >= 580, the Docker address pool,
`setup_node.sh --install-docker`, `probe_node.sh`. One GPU is enough for every stage. No stage runs a server beside the
trainer: in (c) the trainer and the select-split servers all hold GPU 0, so the runner never starts them together.

The throughput limit is Docker on the CPU, not the GPU. A 2-GPU node does not shorten (d) unless it also has about
twice the vCPUs. Then two arms can run on separate GPUs, which needs a different job file.

Disk: each merged 4B model is about 8 GB. (a), (b) and (d) build two or three of them, (c) builds four.

## 2. Stage the inputs on the Mac (main checkout, once per node)

`node sync` never sends `outputs/` or `runs/`, so everything a chain reads lives under `data/` (gitignored, sent by
`--with-data`). The job files use these exact paths.

```bash
A=~/Documents/honest-agents-archive/iterations
mkdir -p data/weights/it14 data/weights/it15 data/ext
cp -R $A/14-session-a/run/it14-sft-v2/final           data/weights/it14/sft-v2-e3-final     # SFT-v2 epoch 3 (= final)
cp -R $A/15-g2-splice-replay/run/it15-g2-splice-replay/final data/weights/it15/g2-s1-final  # the G2 s1 adapter
cp -R ~/Documents/honest-agents-archive/benchmarks/impossiblebench data/ext/impossiblebench  # stages (a), (d); 5.8 MB
# stage (d) only: the adapter stage (c) selected. Use run/it16-commit-checks/<checkpoint of best_checkpoint.json>, or
# final/ when best_checkpoint.json names the last epoch (the archive then keeps only final/):
#   cp -R $A/16-it16c-sft/run/it16-commit-checks/<checkpoint-N|final> data/weights/it16/final-selected
```

Checks before any sync (Mac, $0):

```bash
uv run python -m honest_agents.gate.ledger check                              # must exit 0, or nothing below runs
for f in configs/chains/it16_*.yaml; do uv run python -m honest_agents.node_chain check $f; done
uv run --extra eval python -m honest_agents.eval.final_benchmark check --archive ~/Documents/honest-agents-archive
(cd data/ext/impossiblebench && shasum -a 256 -c MANIFEST.sha256)             # the step on the node repeats this
```

G2 s1 is rebuilt on the node by two CPU merges: base + SFT-v2 e3, then + the G2 adapter. That is the prereg's "rebuilt
on CPU". Every merged model, G2 included, has its 12-task sanity run before any other use of its server
(`runs/eval/it16_g2_sanity` and so on). The merge and sanity steps have the same done markers in every job file, so on
a node that keeps its disk a later stage skips them.

## 3. Wall time and cost per stage

The estimates come from archived iteration-14/15 runs: log start lines (`== <time> start:`), log mtimes, chain
state, `train_runtime` and per-episode `rollout.wall_time_sec`. Cost is at $1.46/h, with 0.75 h of boot, setup and
image pulls per rental.

| stage | steps and basis | wall time | cost |
|---|---|---|---|
| (a) | merges 2 x 4 min (it14 `sft_v2_merge` 3.9 min); 2 sanity runs about 10 min each; **TB2** S = 2 = 178 trials per arm at concurrency 12. No archived TB2 run at 30 turns exists. The 108 Agentic Abstention trials (TB2 tasks, 10 turns, concurrency 12) took 24-27 min (mean 85-114 s per trial, `14-session-b/harbor`), so assume 3x per trial at 30 turns: about 1-2 h per arm, plus about 0.5 h of image pulls for the first arm. **ImpossibleBench** S = 2 = 618 small trials per arm at concurrency 24, about 1-2 min each: 0.5-0.9 h per arm | about 6 h (4.5-8.5) | about $9 (7-12) |
| (b) | merges and sanity 0.35 h; **1,600 episodes**: `it15_g4_sample` (same 400 tasks, same model, concurrency 24) took 3 h 20 min (`sample_g4.log` 08:03:55 to 11:23:45 UTC; summed episode time / 24 = 3.28 h); checklist validation at most 150 min (`--budget-min`; what is left resumes from the cache at L1) | about 6.9 h (4.4 without the validation) | about $10 ($6.4) |
| (c) | merges 0.2 h; data gates (never-trained, coverage, the SFT dry run) about 0.3 h; base on the select split, 424 episodes (it14 `base_select` 37 min); **training**: G2 trained 3,102 examples x 2 epochs in 7,306 s at 0.849 samples/s (`train_g2.log`), so about 2 h for a build of about 3,000 examples, scaling linearly; 2 epoch merges 0.2 h; 2 x (sanity + select split) about 0.7 h each (it14 `sft_v2_select_e*` 25-32 min) | about 5.5 h (4.5-7) | about $8 (6.5-10) |
| (d) | per arm: surface-val-v2 at least 816 x 4 = 3,264 episodes at about 870/h (it15 `surface4`: 960 episodes, 1.1 h) = 3.75 h; tier F 712 episodes about 1.9 h (`g2_s2_tierF.log`: tier F + surface4 in 3 h 02 min); tier S 480 episodes about 0.65 h; the decision probe and the flip probe about 0.3 h; the blind family about 0.5 h; TB2 S = 4 = 356 trials, 2-4 h; ImpossibleBench S = 4 = 1,236 trials, 0.9-1.7 h; Agentic Abstention 432 trials about 1.7 h; AbstentionBench about 0.5 h. That is 12-15 h per arm, x 3, plus 1 h of merges and sanity runs | about 40 h (37-47) | about $58 (54-69) |

Total for the four stages: about 58 h, about **$85** ($74-101) without overrun. Allow 30 % for overrun. Each stage is
asked for separately. The TB2 and ImpossibleBench rows are the least certain. Stage (a) measures both, and (d) is
re-costed from (a)'s measured job times (Harbor `result.json` `started_at` / `finished_at`) before (d) is requested.

MASK is not in (d). Its judge is GPT-4o / o3-mini (paid), which needs a separate owner approval (prereg section 4).

## 4. Launch (per stage, after the owner's go-ahead)

Fill in the stage's placeholders and commit them first, so `node_chain check` prints no placeholder warning. Then sync
from the **main checkout**. A worktree has no usable `.git` on the node.

```bash
H=HOST P=PORT ID=INSTANCE_ID S=configs/chains/it16_a_calibration.yaml       # or _b_claim_rollouts, _c_sft, _d_final_eval
uv run python -m honest_agents.node sync --host $H --port $P --dry-run      # itemised, nothing sent
uv run python -m honest_agents.node sync --host $H --port $P --with-data    # tree + data/ (weights, ext, task dirs)
uv run python -m honest_agents.node chain --host $H --port $P --dry-run $S  # plan, ledger-gate step, ssh command
uv run python -m honest_agents.node chain --host $H --port $P $S            # starts the runner (W&B key, HF token over stdin)
```

Stage (c) logs to W&B (project `honest-agents`, entity `michael_ovo`, group `it16`, run `it16-commit-checks`). The key
comes from `~/.netrc` over ssh stdin, never from a file on the node. Open the run's URL from `train_final.log` and give
it to the owner.

## 5. Watchers

Arm the watcher **in the same minute** as the chain. Run it under a Claude Code Monitor (one stdout line per event),
with `caffeinate -i` around it on the Mac:

```bash
caffeinate -i uv run python -m honest_agents.node watch --host $H --port $P --instance-id $ID --balance-floor 8 $S
```

It prints `START`, `PROGRESS` (25 % steps: stage (b)'s rollouts count `[n/1600]`; the stage (d) arm steps count their
command groups), `DONE`, `FAILED`, `TIMEOUT`, `STALL` (45 min of a quiet log on the Harbor steps, 30 min while
training, 20 min elsewhere), `RUNNER no heartbeat`, `NODE`, `BALANCE` and `ALL_DONE`. On `ALL_DONE` it pulls the state
dir, `watch.pull_paths` (the Harbor job dirs, the units, (b)'s whole eval dir with transcripts, the node's
checklist-validation cache) and the `watch.archives` entries into `~/Documents/honest-agents-archive`, then **stops**
the instance (never destroys it). Stage (b)'s archive entry has `keep_bank: true`, so the transcripts reach the
archive and the claim build can read them. No paid job runs without this watcher (owner rule).

## 6. Stop rules

In this order (prereg section 8, gate step 5, and the training-log incident table):

1. **The gate is shut.** `gate_ledger` exits 1 while any ledger entry is open. The chain stops before anything is
   merged or served, the final step runs, and the watcher pulls and stops the node. Do not delete its done marker to
   get past it.
2. **An infra guard fires.**
   - A sanity step fails. Its incoherent output stops the chain (`on_failure: stop`), and the model is not evaluated.
   - The egress preflight refuses a boot (`run_eval` and Harbor fail the step).
   - The watcher's `balance_floor` (USD 8) writes `STOP`. No new step starts, and the running step finishes.
   - `--hard-floor` pulls and stops at once.
   - A step timeout sends SIGINT, then SIGTERM, then SIGKILL, and leaked `ep*` sandboxes are removed.
3. **A defect is found mid-experiment.** Log it in the ledger at once. If it can change the numbers, stop the job: run
   `ssh ... touch /root/restraint/outputs/chain/<name>/STOP` to start nothing new, and to stop the running step, send
   `kill -TERM` to the runner pid in `chain_state.json` (it sends SIGINT to the step, so sandboxes are torn down).
   Archive what exists, label it `void` or `provisional` in the training log, fix it under gate step 2, and re-sweep
   before resuming.
4. **The data-quality bar fails** (prereg 3.3). The build is refused before (c) and there is no claim-pair training this
   round. In (c), `data_gates` fails on a quota or balance miss (`commit_points.coverage --strict --max-outcome-ratio
   1.5`) or on a benchmark hit in a training build (`final_benchmark never-trained`). The chain stops before training.
   Report to the owner.
5. **No epoch passes the select-split gate** (stage (c)). `select_checkpoint` exits 1, `select_final` fails, and
   nothing is chosen (no fallback epoch). Report to the owner.
6. **A grey region** in any endpoint (read on the Mac after (d)). Stop and report. There is no default action.
7. **One look.** A benchmark that finished is never run again for a better number. A failed step is resumed by
   restarting the chain (its done markers skip finished steps). Delete its partial outputs first, for example a
   half-written `runs/harbor/<job dir>` (the units steps refuse two job dirs). Restart only once the cause is fixed and
   logged.

## 7. After each stage (Mac, $0)

- **(a):** the units files `runs/final/it16a/{base,g2}_{tb2_units,ib_summary}.json`. Add the `RUNS` entries for
  `analysis.power.load_tier_run` (prereg 6.1). Compute m3, the MDEs of P1 and P2 and the external margins from the base
  arm, and write them into the pre-registration before any final-arm episode exists. Re-cost (d) from the Harbor job
  times.
- **(b):** `sft.commit_checks build` and `sft.claim_splice` (commit-data lane) on `iterations/16-it16b-claim-rollouts`
  (episodes, bank) and the pulled checklist-validation cache. Then the agreement bar (prereg 3.3) and the coverage report
  (3.4). Resume the validation of any states left after its 150-minute budget at L1 first.
- **(c):** `selection.json`, `selected_epoch.txt`. Stage the selected adapter as `data/weights/it16/final-selected`
  (section 2) for (d).
- **(d):** `runs/final/it16d/final_benchmark.json` (results and deltas vs base). Then the pre-registered estimator and
  bars on the archive (`analysis.power report --bar ...`, prereg sections 5-6) and the decision table (section 9).

## 8. What is not settled yet

- The placeholders of (c) and (d) (table at the top).
  - The SFT example builder for a prebuilt commit-check build is not registered in `sft.train_sft` (no `kind`).
  - `<SELECT_PROTOCOL>` is the `run_eval --protocol` file with `ni_margin` = max(5, ni10 at the select split), computed
    at build.
- DEF-NEW-it16-jobs-1: no rollout sends the per-(task, sample, turn) seeds the pre-registration promises. Implement
  them or amend the pre-registration before it is approved.
- DEF-NEW-it16-jobs-2: `final_benchmark never-trained` does not read `data/commit_checks/builds/`, so (c)'s
  `data_gates` would not see the build it trains on until that is fixed.
- The ImpossibleBench instruction is the adapter's default "tuned" prompt. The manifest freezes the items and files but
  not the prompt variant, so the job files pin `--prompt tuned` explicitly.
- DEF-43 (the select quarantine) decides what `data/select_quarantine.txt` holds. The select-split runs and
  `select_checkpoint` of (c) exclude whatever it lists.
