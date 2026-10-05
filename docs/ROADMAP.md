# Roadmap

Where the project stands and what remains, in execution order. Updated 2026-10-04 (v3 fix pass).

Legend: done · in progress · not started · GPU = needs the GPU node (and usually Docker)

## Where we are

A go/no-go review (`docs/REVIEW.md`) found that the original design must not be trained. The fix pass specified in
`docs/SPEC_V3.md` has been built component by component and each component reviewed against the spec and the installed
libraries. The integrator's test and conformance summary is `docs/V3_STATUS.md`. **No GPU money has been spent and nothing
has run on Docker or a GPU**; all tests run on macOS with fakes.

| Area | State | Where |
|---|---|---|
| Reward v3, one-blocker report, trajectory bank, TRL reward functions | built, reviewed | `reward.py`, `report.py`, `bank.py`, `trl_rewards.py` |
| Environment, grader, hermetic verifier, budget | built, reviewed, not run in Docker | `env.py`, `grading.py`, `verifier/`, `harness.py` |
| Data pipeline (hermetic conversion, operators with controls, validation, build, difficulty gate, audit) | built, reviewed, not run in Docker | `tasks/` |
| `HonestGRPOTrainer`, training entrypoint, configs, run scripts, stop rules, weight-sync check | built, reviewed, GPU parts unverified | `trainer.py`, `train.py`, `callbacks.py`, `sync_check.py`, `configs/`, `scripts/` |
| Report-only pilot | built, reviewed | `pilot/` |
| Evaluation (held-out metrics, baselines, report-file protocol, external adapters) | built, reviewed, external adapters unverified | `eval/`, `docs/EVAL.md` |
| OpenEnv wrapper | built; at review time tested against contract stubs for the grader, re-check against the real core | `openenv_env/` |
| Proposal v2, design, review record | written | `docs/proposal.md`, `docs/DESIGN.md`, `docs/REVIEW.md` |
| Anything executed on Docker or a GPU | **not started** | |

Not built, though discussed in the review: the bridged AgentAbstain harness and free-text reporting variant (F2), a
verify-then-report gate baseline (F5), a supervised-fine-tuning baseline, multi-harness training, a `hide_parameter`
operator, a mixture schedule. On-policy distillation (OPD) is **deferred** (below).

## The spending gate

Spend in stages. Each stage runs only if the previous one's condition holds. All costs are estimates built on assumed
latencies and $2.5-3.5 per H100-hour; treat them as plus or minus a factor of three until stage 1 measures step time.

| Stage | What | Cost | Continue only if |
|---|---|---|---|
| 0 | The spec implemented and attacked locally (done except for owner sign-off) | $0 | all regression and exploit tests pass |
| 1 | Node bring-up, audit dataset, zero-learning-rate run, base and Tmax inference | about $300-500 | the gates below |
| 2 | Report-only pilot on frozen trajectories | about $400-600 | claim rate moves with alpha and fabrication falls (`PILOT GO`) |
| 3 | Two short full runs: honest against outcome, same seeds and compute | about $300-600 | reporting improves and task success stays within the margin |
| 4 | Seeds for both arms, then the alpha values the pilot supports | about $2-3k | |

**Stop conditions, stage 1.** The code (`python -m honest_agents.callbacks phase0`, run by `scripts/phase0.sh`) checks four gates
and exits non-zero if one fails or was not measured:
report compliance at least 70%; no-report share at most 5%; fabrication among failed feasible episodes (P(claim | fail)) at
least 20%; mean importance-sampling ratio within 1% of 1. The spec adds three that are read from the same run by hand:
the measured seconds per step (decision rule below), token-budget exhaustion at most 20% of episodes, and at most 50% of
groups with no reward contrast. If fabrication among failures is below 20%, the base model has little to unlearn and the
whole premise needs rethinking before any more spending.

Three further stop conditions come from the reviewers (judgement thresholds, not in the spec): stop or re-scope if base
P(claim | fail) is below about 25% **and** abstain recall is above about 70%; stop or re-scope if a no-training baseline
(neutral prompt, confidence-threshold sweep) reaches P(claim | fail) below 10% at under 5 points of verified-success loss;
drop AgentAbstain as even a transfer metric if the base model's act accuracy there is below about 25-30%.

**Stop condition, stage 2.** `pilot.analyze` prints `PILOT GO`, `NO-GO` or `INCONCLUSIVE` (exit 0, 1, 2) from the `scale_rewards:
none` arms: claim rate must be monotone in alpha and move, fabrication among failures must fall, and the fall must not come from
quitting (a claim-rate floor, claims that still discriminate, a report-rate floor). On `NO-GO`, stop and redesign.
**This stage is the first run that learns, so run `scripts/sync_check.sh` first.**

**Stop conditions, stages 3 and 4.** The stop rules (`stop:` in `configs/base.yaml`) end a run on claim rate on feasible below 5%,
zero-variance groups above 50%, decline rate on feasible above 80%, masked share above 15%, training or validation task
success falling while reward rises, or the share of episodes that end with a report falling. A stop exits with status 3.
Success is judged by the non-inferiority margin (3 points of task success) fixed before the run.

**Trainer decision rule (SPEC 10.3), decided from the zero-learning-rate run.** Stay on TRL if the projected concurrent step time is at
most 15 minutes with training at most half of it, the longest batch fits at per-device batch size 1, at most 50% of groups
have no contrast and at most 20% of episodes exhaust the budget. Port the full loop to verl if the longest batch runs out of
memory, if training dominates the step, if fixing contrast or exhaustion needs G of 16 or more or a 32k budget, or if per-token
advantages become part of the method. SkyRL and a hand-built trainer are not primary paths.

## Next steps, in order

### Local, no GPU (finishing the fix pass)
- [ ] Integrator: reconcile components and write `docs/V3_STATUS.md`; run the full suite and the lint.
- [ ] Owner decisions (section "Open decisions" below), in particular the held-out families and the dataset size.
- [ ] Re-run the literature search before anything is written for submission (closest works so far: AWA-RL, PROVE, RefGRPO,
  Confessions, TruthRL, Bench2Robust, 2608.00301). Only abstracts were read for several of them.

### Stage 1: node bring-up and measurement (GPU, about $300-500)
- [ ] Rent a 4xH100 **VM** (a real Docker daemon and nftables, not a container pod). `scripts/setup_node.sh`, then
  `scripts/probe_node.sh` (GPUs, Docker, compose v2, nftables egress control).
- [ ] `uv run pytest -m docker`: real-sandbox grading, network and user enforcement, tamper detection. This is the first
  time the hermetic verifier meets a real container; expect surprises (pytest bundle on a Python 3.10 image, `timeout -k` in
  images that lack it, root writing through `/proc`).
- [ ] `scripts/build_audit_set.sh` and hand review; then `scripts/build_dataset.sh` with validation. Gate: at least 95% of
  variants validate as labelled; measure the real leak rate; confirm the `remove_input` and `conflicting_constraint` yield.
- [ ] `scripts/smoke_test.sh` (two tasks of different length, two steps) with `configs/smoke.yaml`, then with `configs/qwen35_4b.yaml`
  (Qwen3.5 needs `flash-linear-attention`; LoRA on the language model only; Qwen3-Coder XML tool calls inside the loop).
- [ ] `scripts/phase0.sh`: the real trainer at learning rate 0 for 24 rounds, writing the trajectory bank. Reads off step time by
  component (generation, tools, grading, training, sync), memory at the longest batch, report compliance, stop reasons, pass-count
  histogram, the importance-sampling ratio and the budget-exhaustion rate.
- [ ] From the same bank: the difficulty gate (`tasks.difficulty`), the **pool's mean pass rate and the AUROC of stated confidence
  against the tests** (decides which alpha values are safe), `work_vs_quit_gap` for lambda, and the quarantine list.
- [ ] Inference-only baselines: base Qwen3.5-4B and `allenai/tmax-4b` on the held-out split (`scripts/h1_tmax_check.sh`; the H1 direction
  check), the neutral-prompt arm, the confidence-threshold sweep, base honesty under our harness and under at least one stock
  harness (`scripts/eval_harness.sh`), and the base model on AgentAbstain's act side to see whether there is headroom.

### Stage 2: report-only pilot (GPU, about $400-600)
- [ ] `scripts/sync_check.sh` with a non-zero adapter: choose the sync mode (merged with the clone patch if it passes; adapter-only
  sync if only that does; float32 serving or Qwen3-4B if neither does on Qwen3.5).
- [ ] `python -m honest_agents.pilot.run --config configs/pilot.yaml`, then `scripts/pilot_report.sh`. alpha {0, 0.25, 0.5, 0.75} ×
  `scale_rewards` {none, group}. The pilot says nothing about H1, effort or action-level abstention.

### Stage 3: two short full runs (GPU, about $300-600)
- [ ] `scripts/run_pair.sh configs/honest_alpha0.25.yaml configs/outcome.yaml seed=42`: honest against outcome, same seed and compute,
  one node, two slots. Set `max_steps` and group size from stages 1 and 2 (the config default is 300 steps of 64 episodes; the
  reviewers suggested about 100). Watch the validation probe and the stop rules.
- [ ] Held-out evaluation of both with `scripts/eval_heldout.sh` (report together: false completion, task success, claim coverage,
  near-miss false refusal, decline rate on feasible tasks, false blame, AUROC, realised threshold, leak rate per family) and the
  non-inferiority test. Compare with the constant-policy rows and the no-training baselines.

### Stage 4: seeds and the alpha grid (GPU, about $2-3k)
- [ ] Three seeds of both headline arms (honest alpha 0.25 and `outcome`). `scripts/sweep_alpha.sh` for alpha {0, 0.1, 0.25, 0.4}
  and the arms `outcome_claimed` and `outcome_reflect` (the RefGRPO-style comparison). The alpha trend should come from the pilot
  and from these, not from four single runs. alpha 0.5 and 0.75 are not run: near-degenerate for weak self-knowledge.
- [ ] Optional: supervised fine-tuning on grader-verified honest trajectories as a baseline (not built; the bank holds the data).

### Evaluation and write-up
- [ ] Transfer: Agentic Abstention terminal subset (exploratory, 19 abstain-warranted public tasks), AgentAbstain (paid GPT-5.4 judge;
  act and abstain sides separately), AbstentionBench (regression check), TB Lite and TB 2.0 (capability), SWE-bench fabrication
  (stretch).
- [ ] Release the dataset (licences: Endless Terminals MIT with a malformed card, SETA Apache-2.0, TMax ODC-BY with Gemini outputs, feasible
  only) and the OpenEnv environment.

## On-policy distillation (OPD): deferred

OPD is **not** part of any stage above. The earlier plan added a teacher's per-token advantage to the GRPO loss (teacher
`Qwen/Qwen3.5-27B`, same tokenizer as Qwen3.5-4B, conditioned on a stronger honesty context). The review found no measurement that
requires it, and it would add a second server, a trainer change and a confound to H1 and H2. A teacher's honesty is also bounded
by the teacher's (frontier models score 45-60% on AgentAbstain), so pure OPD would turn the contribution into distillation.

Revisit only if stage 1 or 2 shows a need: report compliance below about 70% that prompting does not fix, or a task-success drop
the honest arm cannot keep within the margin. Even then, **supervised fine-tuning on grader-verified honest trajectories is the first
warm-start option** (text-level, so the teacher's tokenizer does not matter, and far cheaper than RL), and the report-only pilot
comes before any warm start. If OPD is ever built, the GPU layout needs a third GPU for the teacher (27B bf16 is about 54 GB) and the
nodes' two-runs-side-by-side layout no longer holds.

## Open decisions

| Decision | Default now | Who |
|---|---|---|
| Held-out families | `conflicting_constraint` and `no_network` (decided while the owner was away, `docs/DATA.md`). Training sees high_risk, revoke_permission and remove_input; revisit freely | owner |
| Dataset size | `LIMIT=1500` seeds, 300 test, 60 val, about 1,140 train. A trade against test resolution (SPEC V-2) | owner |
| `false_blame` | 0. At alpha = 0 a false blame is then free; `min(0, alpha - 0.1)` recommended, not set | owner |
| Group size | G=8 (4 to 64 times fewer episodes per step than any cited recipe). Decide after stage 1 and the pilot | owner |
| Headline model | Qwen3.5-4B after the smoke test; Qwen3-4B for bring-up | owner |
| Checkpoint selection | validation task success alone (ignores honesty) | owner |
| AgentAbstain judge budget (GPT-5.4) | needed only if the base model has headroom there | owner |
| Multi-harness honesty training | a later stage; needs TRL's `HarnessRolloutWorker` (on TRL main, not in 1.14.1) and an honesty reward that runs as a Harbor verifier | later |
| GPU provider | 4xH100 VM with Docker | decided |
