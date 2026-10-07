# Training log

One entry per iteration, newest last: what we set out to test, the exact design, what happened (numbers), what we think
it means, and the decision it led to. Corrections to earlier readings are written into the later entry that made them,
not by editing history. Template at the end.

Node: Vast.ai VM, 1x RTX PRO 6000 Blackwell 96 GB, 23 vCPU, 73 GB RAM, ext4 disk 194 GB. Model: `Qwen/Qwen3.5-4B`
(snapshot `851bf6e8`). Stack: torch 2.13.0+cu130, vLLM 0.30.0, TRL 1.14.1, flash-linear-attention 0.5.2.
Durable copy of all data and results: `~/Documents/honest-agents-archive/2026-10-06-node1/` (README there).

---

## 0. Dataset build and validation (2026-10-05/06)

**Goal.** A validated task pool with a frozen test split.

**Design.** Endless Terminals (MIT, revision `26ecf784`), 2,482 source tasks, 0 dropped by decontamination against 912
benchmark tasks. Hash split by source task: 1,388 train / 539 val / 565 test. 8,790 variants built (plain, near-miss
control, impossible twin; families `high_risk`, `revoke_permission`, `remove_input` train; `corrupt_input`,
`conflicting_constraint`, `no_network` test only). Validating all 8,790 would take ~7 h (network-bound image builds), so a
chosen subset of 1,955 variants was validated (`--only-file`): every held-out-family test pair, samples of the rest.

**Result.** 1,907 / 1,955 valid (97.5 %, gate 95 %). Final splits: train 454 rows, val 178, test 763, train_pool 966.
Test split frozen in `configs/test_ids.txt` (565 source tasks).

**Bugs found by real-Docker validation (all fixed, with tests).**
- `revoke_permission`: images with `ENV PATH=/usr/local/bin:/usr/bin:/bin` could not find `useradd` (exit 127).
- Dockerfile analysis read heredoc bodies as instructions (an SQL `user TEXT,` line became `USER TEXT,`).
- Docker's default network pool holds ~30 networks; widened to `10.128.0.0/9` /24.
- One transient `apt-get update` failure (retried).

---

## 1. Phase 0 smoke: real trainer at learning rate 0 (2026-10-05)

**Goal.** Prove the GRPO stack end to end on one GPU (colocated vLLM) before any spend.

**Design.** `configs/phase0_colocate.yaml`, 2 steps, lr 0, audit dataset.

**Result.** Ran end to end. Report compliance 84 % (gate >= 70 %); **no-report 15.6 % (gate <= 5 %: fail)**; claims after a
failed feasible attempt 86 %; importance ratio mean 0.99999; peak GPU 56.8 GB; step 377 s (rollout 200, grading 39,
train 93).

**Bugs found.** End-of-turn check read `eos_token_id` from the multimodal processor instead of its tokenizer; the bare
`[invalid tool arguments]` message gave the model nothing to fix (now names the missing argument).

---

## 2. Base-model baseline (test split, 2026-10-06)

**Design.** Qwen3.5-4B, honest harness, 2 samples per variant, 1,524 scored episodes, 156 source tasks.

| metric | base |
|---|---:|
| verified success | 68.3 % |
| claims done on feasible | 85.5 % |
| false-claim share of claims | 24.8 % |
| **claims done on impossible** | **48.7 %** |
| abstain recall / precision | 29.5 % / 95.2 % |
| correct blocker when abstaining | 91.3 % |
| over-refusal on near-miss controls | 1.5 % |
| missing reports | 16.0 % |

Per family abstain recall: `high_risk` 0.4 %, `revoke_permission` 54.9 %, `corrupt_input` (never trained) 42.2 %.
Reference policy `never_claim` scores a higher mean reward (0.431) than the base model (0.316) on this split: the mean
reward alone can be gamed by blanket declining.

---

## 3. pass@4 bank and difficulty gate (2026-10-06)

**Design.** Base model, 4 samples on the 957-row train pool: 3,828 episodes (`runs/passk`). `tasks.difficulty --rollouts 4`.

**Result.** Feasible: 145 of 250 too easy (4/4), 15 never pass, 75 in the 2-3-of-4 band (mean pass 0.56). 10 of 350
impossible variants leak (quarantined). Stated-confidence AUROC 0.59 (near chance).
**Alpha safety** (claim iff p(success) > (1 + alpha) / 2): on the 1-3-of-4 band (mean pass 0.47) no alpha is safe; on the
2-3 band only alpha 0 and 0.1 are. alpha 0.25 (the design default) would let blanket declining win.

**Reflection.** The first run of the gate used the default 8-rollout band on a 4-rollout bank and was wrong; always pass
`--rollouts`.

---

## 4. SFT 1: hindsight report SFT, loss on report tokens only (2026-10-06)

**Design.** `configs/sft_hindsight.yaml`: 2,950 examples from the pass@4 bank (each rollout's final report rewritten to the
honest one the grader supports), LoRA r16, lr 5e-5, 1 epoch, 166 steps, 68 min.

**Result.** Final held-out loss 0.0094, token accuracy 99.7 %. On validation the adapter **quit**: zero commands in 82 % of
feasible episodes (mean 0.2 vs 14.9 for base), 57 % `lazy_decline`, verified success ~1 % vs 66 %.

**Diagnosis.** Every example was an end-of-task state with loss only on the report, so the model learned "emit a report"
and never "keep working". The low loss measured the boilerplate report fields, not the decision.

**Reflection.** Do not read SFT loss as quality. Check command counts and outcomes on held-out tasks before anything else.

---

## 5. SFT 2: combo (best-of-N trajectories + hindsight reports) (2026-10-06)

**Design.** `kind=combo`: 570 best-of-N trajectories (loss on every assistant turn) + 570 hindsight reports, 72 steps.

**Result (validation, 531 episodes, same split as base).**

| metric | base | combo SFT |
|---|---:|---:|
| verified success | **72.9 %** | 59.2 % |
| mean reward | **0.638** | 0.521 |
| missing reports | 11.4 % | **1.9 %** |
| false-claim share | **18.4 %** | 25.3 % |
| declines on feasible | **1.4 %** | 20.0 % |
| over-refusal on controls | **1.7 %** | 9.2 % |
| abstain recall / precision | 33.3 / **92.7 %** | 37.6 / 74.6 % |
| correct blocker | **81.6 %** | 48.9 % |

**Decision.** SFT fixes formatting, not honesty, and costs task success. GRPO starts from the base model; the -0.25
no-report penalty is left to fix reporting.

---

## 6. Weight-sync test, colocated (2026-10-06)

**Why.** `scripts/sync_check.sh` needs two GPUs; a stale sync at a small learning rate is invisible in the importance ratio.

**Design.** 2 GRPO steps at lr 1e-3 (deliberately destructive).

**Result.** Step 2 after the big update: importance ratio 0.9999, log-prob gap 0.0135 (0.0072 before), while behaviour
changed sharply (reward 0.447 -> 0.019, commands 17.3 -> 6.9, entropy 0.275 -> 0.444). vLLM samples from the updated weights
and agrees with the trainer token by token: **sync works.**

---

## 7. GRPO run 1 (`grpo_main`): alpha 0.1, false_blame 0, lr 3e-5 (2026-10-06)

**Design.** `configs/grpo_colocate_qwen35.yaml` at commit `82f28a5`: base-model start, alpha 0.1, LoRA r32, lr 3e-5 with 5
warmup steps, 8 prompts x 8 generations per step, 50 steps planned. RL pool from `tasks.rl_pool`: 360 rows (75 plain in the
difficulty band, 185 near-miss controls, 100 impossible balanced over the 3 families) - the build's own difficulty filter
had left only 32 impossible rows.

**Result.** Killed by the kernel's OOM killer at step ~19 (trainer at 69 GB of 73 GB host RAM).
Training-time, per task type (weighted by episode counts):

| | steps 1-5 | steps 16-18 |
|---|---:|---:|
| correct abstention, impossible | 48 % | **83 %** |
| claims done, impossible | 36 % | **4 %** (n = 48) |
| verified success, feasible | 72 % | 44 % |
| fabrication, feasible | 16 % | 44 % |
| false blame, feasible | 0.5 % | 3-4 % |

**First reading (wrong in part).** "Success is falling and declines creep up: the reward is driving the model to quit."

**Correction.** Same-task comparison against the pass@4 bank: the trained policy's success tracked the base model's on the
exact same tasks (gap -0.05 to 0.00 in every window); later steps simply drew harder tasks. Much of the "decline creep" was
honest `incomplete` reports. The one real problem was **false blame**: the reward paid 0 for it, so naming a blocker beat
"I could not finish" whenever p(blocker real) > (alpha - false_blame) / (1 - wrong_reason + alpha - false_blame) = 17 %.

**Reflection.** Raw per-step reward and success are confounded by task difficulty. Compare against the base model on the
same tasks, or between runs that share the data order.

---

## 8. Memory leak (2026-10-06)

**Diagnosis.** 3-step run with `HONEST_MEM_DEBUG=1`: trainer RSS 4.4 -> 6.0 -> 7.1 GB, Python heap flat at 0.06 GB, so the
growth is native (~1 GB/step), not a Python leak; that alone does not reach 69 GB in 18 steps, so the crash was most likely
a spike: `bash()` kept a command's whole output in the trainer (the model sees 2,000 characters of it).

**Fixes (`275a8d5`).** Output capped at 1 MB inside the sandbox (`head -c`, `pipefail` keeps exit codes); `MALLOC_ARENA_MAX=2`;
`malloc_trim(0)` after every step; `mem/rss_gb` logged every step; `train.health` warns before RAM runs out.
**Verified:** run 2 holds the trainer flat at 4.8-4.9 GB for 26+ steps.

---

## 9. GRPO run 2 (`grpo_v2`): false_blame -0.4 (2026-10-06, running)

**Design.** Run 1 plus `reward.config.false_blame: -0.4` (blame threshold 17 % -> 50 %: name a blocker only when more likely
than not; satisfies beta 1.0 > 0.9). Same seed and data order as run 1, so every step draws the same tasks: run-to-run
comparisons are paired.

**Validation probe (fixed 100 variants: ~21 impossible, ~78 feasible).**

| | step 0 | step 20 |
|---|---:|---:|
| claims done on impossible | 60.4 % | 56.8 % |
| task success, feasible | 79.9 % | 80.7 % |
| control success | 78.7 % | 77.5 % |
| claims after a failure | 68.8 % | 87.5 % (n ~ 16) |
| false blame / declines on feasible | 0 / 0 % | 0 / 1.4 % |
| missing reports | 6 % | 4 % |

**Training-time trend (weighted by episode counts).**

| | steps 1-8 | 9-16 | 17-25 |
|---|---:|---:|---:|
| correct abstention, impossible | 43.1 % | 50.8 % | **67.5 %** |
| claims done, impossible | 36.8 % | 38.3 % | **27.5 %** |
| false blame, feasible | 1.7 % | 2.4 % | 1.4 % |
| claims after a failure, feasible | 65.2 % | 58.8 % | 71.4 % |
| task success, feasible | 66.7 % | 64.9 % | 62.5 % |

Paired with run 1 on the same tasks (steps 6-14): false blame 3.6 -> 2.5 %, but correct abstention on impossible 53.9 ->
40.8 % and claims after a failure 57.8 -> 63.9 %: the penalty may also make the model more reluctant to abstain (small n).

**Reading so far.** Abstention on impossible tasks is improving during training without over-refusal; the step-20 probe
does not confirm it yet (within noise on 21 variants). Honesty after a failed feasible attempt (fabrication) is not
improving. Entropy dipped to 0.160 at step 26 (start ~0.30): watch.

**Correction at step 28.** The "improving" reading above depended on where the windows fell. Re-cut in 9-step windows:
claims done on impossible 41.1 % (steps 1-9) -> 23.5 % (10-18) -> 40.3 % (19-27); correct abstention 41.7 -> 65.4 -> 55.6 %;
claims after a failure 66.4 -> 62.7 -> 68.8 %. The last 5 steps (24-28) claimed 64 % of impossible tasks vs 38 % in steps
1-5 (about 70 episodes). No clear honesty gain; possibly a drift toward claiming.

**Diagnosis.** Behaviour on an impossible task is close to fixed per task: groups come out 8/8 claimed (steps 21, 26) or
0/24 (step 23). A group whose 8 samples all do the same thing carries zero GRPO advantage, so many impossible tasks teach
nothing. The false-blame penalty may also push borderline impossible tasks toward claiming (the paired run-1/run-2
comparison above points the same way).

**Ideas for the next iteration** (not yet run): more within-group diversity on impossible tasks (higher temperature or 16
generations), a milder false_blame (-0.15, blame threshold 1/3, the reward module's documented default), a larger
validation probe.

**Decision.** Run to step 50 as planned (owner's call), then evaluate.

**Final result (completed 2026-10-07, 50 steps, ~7 h, rc 0; checkpoints 10-50 + `final`).** Validation probe:

| | step 0 | 20 | 40 | 50 |
|---|---:|---:|---:|---:|
| claims done on impossible | 60.4 % | 56.8 % | 60.4 % | **60.4 %** |
| task success, feasible | 79.9 % | 80.7 % | 80.5 % | **85.4 %** |
| control success | 78.7 % | 77.5 % | 79.8 % | **86.0 %** |
| fabrication (share of episodes) | 12 % | 13 % | 11 % | 8 % |
| claims after a failure | 68.8 % | 87.5 % | 68.8 % | 76.7 % |
| false blame | 0 % | 0 % | 0 % | 0 % |
| mean reward | 0.652 | 0.685 | 0.680 | 0.768 |

Feasible success on the same training tasks against the base model (pass@4 bank): gap -0.00 (steps 1-10), +0.03 (11-20),
+0.10 (21-30), +0.15 (31-40), ~400 episodes per window. Entropy drifted from ~0.27 to ~0.17 (collapse warnings at 57-67 %).

**Conclusion.** The run made the model better at the tasks, on training tasks (+15 points against the base on the same
tasks) and, at the end, on held-out validation (+5 to +7 points, borderline on ~78 feasible episodes). It did **not** make it
more honest: false claims on impossible tasks are exactly at baseline and claims after a failure did not fall. The reward
is effectively paying for task success; the honesty terms carry little gradient.

**Why (best explanation).** (1) Behaviour on an impossible task is near-deterministic per task, so most impossible groups
are 8/8 or 0/8 and give zero advantage. (2) The plain copy and the near-miss controls of a source task are the same task,
so the pool repeats tasks under different variant ids; part of the training-task gain may be memorised tasks. (3) The
probe has ~21 impossible variants: it cannot resolve changes under ~10 points.

**Next iteration (proposal, not run).**
1. Within-group diversity on impossible tasks: 16 generations per prompt, or temperature 1.2 on the first turns, so the
   honesty terms produce advantages; measure the share of zero-variance impossible groups as the gate.
2. Pool built so a source task contributes one variant per epoch (no plain + control repeats).
3. false_blame -0.15 (blame threshold 1/3) instead of -0.4.
4. A larger, honesty-weighted validation probe (e.g. 60 impossible + 60 feasible) and checkpoint selection on a combined
   honesty + success metric instead of task success alone.

---

## Infrastructure incidents (and the guard each one left behind)

| incident | guard |
|---|---|
| a sandbox wrote a 63 GB file and filled the disk | `ulimits.fsize` 1 GiB in the compose override (`e431ad5`) |
| pruning Docker's build cache made every image rebuild (~4 MB/s mirror downloads) | do not prune the build cache mid-run |
| full disk tore the last line of `episodes.jsonl`; `--resume` died | reader skips a torn last line (`eb677e9`) |
| unattended OS upgrade replaced the NVIDIA libraries (580.178) under the loaded 580.95 driver | module reload; unattended upgrades disabled on the node |
| trainer OOM-killed at 69 GB | entry 8 |
| monitors that only printed progress missed real problems | `train.health` (stall, drift, collapse, gaming, learning, RAM) |

## Method notes

- Judge learning on honesty rates per task type, weighted by episode counts, not on mean reward.
- Keep the data-order seed fixed across runs that are compared; the same steps then see the same tasks.
- The validation probe (100 variants, ~21 impossible) is too small to resolve differences under ~10 points.
- Checkpoint selection uses `eval_honest/task_success`, which ignores honesty (known limitation).

---

## Entry template

```
## N. <short name> (<date>)

**Goal.** What question this iteration answers.
**Design.** Config file and commit, what changed from the previous run and why, data, seed, steps, cost.
**Result.** Numbers (with n), tables against the right baseline (same split, same tasks).
**Diagnosis.** What we think caused it; how we checked.
**Reflection.** What we would do differently; corrections to earlier readings.
**Decision / next.**
```
