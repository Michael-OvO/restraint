# Roadmap

Where the project stands, the spending gate, what runs where, and what remains, in execution order.

## Where we are

A go/no-go review found that the original design must not be trained ([reviews.md](reviews.md)). The fixes were built component by
component, reviewed against the installed libraries, attacked again by a second audit and a leakage audit, and cleaned up. **No GPU
money has been spent and nothing has run on Docker or a GPU**; every test runs on macOS with fakes. Stage 0 of the gate below is met:
the regression and exploit tests exist and pass, and every config and script resolves (`tests/test_pipeline_integration.py`).

| Area | State | Where |
|---|---|---|
| Reward, one-blocker report, trajectory bank, TRL reward functions | built, reviewed | `reward.py`, `report.py`, `bank.py`, `trl_rewards.py` |
| Environment, grader, hermetic verifier, budget | built, reviewed, not run in Docker | `env.py`, `grading.py`, `verifier/`, `harness.py` |
| Data pipeline (hermetic conversion, operators with controls, validation, build, difficulty gate, audit) | built, reviewed, not run in Docker | `tasks/` |
| `HonestGRPOTrainer`, entrypoint, configs, run scripts, stop rules, weight-sync check | built, GPU parts unverified | `trainer.py`, `train/`, `callbacks.py`, `sync_check.py`, `configs/`, `scripts/` |
| Supervised stages (hindsight report SFT, best-of-N trajectory SFT) | built, CPU dry run only; example labels (hindsight blockers and abstentions only on infeasible rows) must be read on a real bank, and best-of-N trains template-rendered calls rather than banked `completion_ids` | `sft/`, `configs/sft_*.yaml`, `scripts/run_sft.sh` |
| Report-only pilot | built | `pilot/` |
| Evaluation (held-out metrics, baselines, report-file protocol, external adapters) | built, external adapters unverified | `eval/` |
| OpenEnv wrapper | built; its suite runs in a separate venv | `openenv_env/` |
| Anything executed on Docker or a GPU | **not started** | |

Not built, though discussed: the bridged AgentAbstain harness and free-text reporting variant, a verify-then-report gate baseline,
multi-harness training, a `hide_parameter` operator (a stub that needs an LLM client), a mixture schedule, and the glue that starts
GRPO from an SFT adapter (see "SFT, then GRPO"). On-policy distillation (OPD) is deferred (below).

## The spending gate

Spend in stages. Each stage runs only if the previous one's condition holds. Costs are estimates built on assumed latencies and $2.5
to 3.5 per H100-hour; treat them as plus or minus a factor of three until Stage 1 measures step time.

| Stage | What | Cost | Continue only if |
|---|---|---|---|
| 0 | The fixes implemented and attacked locally (done) | $0 | all regression and exploit tests pass |
| 1 | Docker checks and audit set on the rented H100 box, sync check, base evaluation, Phase 0, hindsight then best-of-N SFT, one short warm-started GRPO run ([node-runbook.md](node-runbook.md)) | about $200 to 400 (factor of three either way) | the Stage 1 gates and the evaluation gate below |
| 2 | Report-only pilot on frozen trajectories | about $400 to 600 | claim rate moves with alpha and fabrication falls (`PILOT GO`) |
| 2b | Part of Stage 1 now: supervised stage on the Phase-0 and pass@8 banks (below) | included above | evaluation gate and `--spread` |
| 3 | Two short full runs: honest against outcome, same seeds and compute | about $300 to 600 | reporting improves and task success stays within the margin |
| 4 | Seeds for both arms, then the alpha values the pilot supports | about $2 to 3k | |

Cost model behind these (the review's, never timed): $10 to 14 per 4-GPU node-hour, 64 episodes per step, 300 steps, about 14 tool calls
per episode. With one rank and parallel tools, two runs sharing a node, a run is about 11 minutes per step, 55 hours and $277 to 388,
nine runs $2.5 to 3.5k. Add 30 to 50% for bring-up, crashes and evaluation. With one rank and serial tools it would have been about 21
minutes per step; the old layout could not complete a step at all. Not in the model: 1% of calls hitting a command timeout adds about 27
minutes per step on one rank, and 1 to 3% of episodes hitting the test timeout adds 6 to 11 minutes.

**Stop conditions, Stage 1.** Four gates (report compliance at least 70%, no-report at most 5%, fabrication among failed feasible
episodes at least 20%, importance ratio within 1% of 1) plus the by-hand reads and the stop conditions are in
[node-runbook.md](node-runbook.md) section 4. If fabrication among failures is below 20%, the base model has little to unlearn and the
premise needs rethinking before more spending.

**Stop condition, Stage 2.** `pilot.analyze` prints `PILOT GO`, `NO-GO` or `INCONCLUSIVE` (exit 0, 1, 2) from the `scale_rewards: none`
arms: claim rate must fall with alpha and move, fabrication among failures must fall, and the fall must not come from quitting (a
claim-rate floor, claims that still discriminate, a report-rate floor). On NO-GO, stop and redesign. A GO is necessary, not sufficient.
**This stage is the first run that learns, so run `scripts/sync_check.sh` first.**

**Stop conditions, Stages 3 and 4.** The stop rules (`stop:` in `configs/base.yaml`, listed in [design.md](design.md) section 8) end a run;
a stop exits with status 3. Success is judged by the non-inferiority margin (3 points of verified success) fixed before the run
([evaluation.md](evaluation.md) section 2).

**Trainer decision**, from the zero-learning-rate run: stay on TRL or port the loop to verl ([design.md](design.md) section 8).

## What runs where

| Where | What | Needs |
|---|---|---|
| macOS (the dev machine) | all unit tests (`scripts/test.sh`), dataset build without validation, every `--dry-run`, SFT and pilot dry runs, bank re-scoring, metric recomputation, the pilot stop-rule simulations, ruff | nothing |
| macOS, optional | a Docker smoke of `pytest -m docker` under Colima or OrbStack (arm64: a hint only; not installed, steps in [node-runbook.md](node-runbook.md) section 1) | a Docker daemon the owner installs |
| **One rented H100 VM (x86) with Docker, authoritative** | `pytest -m docker`, audit set, oracle validation, container-level leak checks ([data.md](data.md) section 6); then, on its one GPU with the colocated config: base evaluation, Phase 0 (`configs/phase0_colocate.yaml`), pass@8 rollouts, both SFT stages, evaluation of the adapters, one short GRPO run | Linux, Docker daemon, compose v2, nftables, one GPU |
| A 2-GPU VM, about an hour | weight-sync check and the full smoke test (they need a vLLM server on a second GPU; NCCL cannot pair two ranks on one device) | 2 GPUs |
| 4xH100 VM with Docker | Phase 0 at the server layout, the pilot arms (one vLLM per arm), Stages 3 and 4 (two run slots) | 4 GPUs |

One rental first ([node-runbook.md](node-runbook.md) section 1): Docker-only risks come first and need no GPU, so check them before the GPU hours start.

## SFT, then GRPO

The plan the code supports (`sft/`, [design.md](design.md) section 9): a cheap supervised stage on the model's own Phase-0 rollouts, then one
short GRPO run from its result. SFT on grader-verified honest trajectories is far cheaper than RL in the reference guide (1.5 to 6.5 hours
against 32 to 46) and a baseline reviewers will expect.

1. Phase 0 writes the bank (Stage 1); the pass@8 run over the whole pool writes the bigger bank best-of-N needs. The sft configs default `data.bank` to `outputs/phase0/bank`, which Phase 0 does not write: pass `data.bank=outputs/phase0/<run>/bank`.
2. `DRY_RUN=1 scripts/run_sft.sh configs/sft_hindsight.yaml` on the CPU, then read `<output_dir>/examples.jsonl`: the rewritten reports and
   labels are verified by eye before any GPU time (the gate: no paid training until labels are verified).
3. Train: `scripts/run_sft.sh configs/sft_hindsight.yaml`, or `kind=combo` with `configs/sft_bestofn.yaml`.
4. Sample a new bank with the SFT adapter and run `python -m honest_agents.sft.train_sft --spread <bank>`; compare with the starting bank. A
   collapsed spread means GRPO has nothing to learn from; stop.
5. The evaluation gate of the node runbook (base against each adapter on the test split), then one short GRPO run from the SFT result.

**Gap**: `train/` has no way to start from an adapter (`model.name` loads a base model, and `train_sft` saves a LoRA adapter to
`<output_dir>/final`). Step 5 needs either a merge step that writes full weights and `model.name` pointing at them, or an `init_adapter`
option. Neither is written; do it before step 5, and re-run the weight-sync check on the result. The SFT arm is also a baseline of its own
(SFT only, no RL) in the evaluation tables.

## Next steps, in order

### Local, no GPU
- [ ] Owner decisions (below), in particular the held-out families and the dataset size.
- [ ] The SFT warm-start glue (above) and its test; `sync_check.py` for the colocated engine (it needs a server and so a second GPU today); the sft configs' default `data.bank`.
- [ ] Re-run the literature search before anything is written for submission (closest works so far: AWA-RL, PROVE, RefGRPO, Confessions,
  TruthRL, Bench2Robust, 2608.00301). Only abstracts were read for several of them.

### Stage 1: one H100 box, in this order (about $200 to 400)
Commands, gates and stop conditions: [node-runbook.md](node-runbook.md) (sections 2 and 4). Each step runs only if the previous one passed.
- [ ] Docker: `pytest -m docker`, `scripts/fetch_benchmarks.sh`, audit set with hand review, full dataset with oracle validation (at least 95% validate).
- [ ] Weight-sync check and smoke on a 2-GPU VM (a few dollars), or `sync_check.py` taught the colocated path (local list above).
- [ ] Base evaluation on the test split (inference only): reported numbers are abstain recall and near-miss false-refusal rate, plus report rate, task success and the stock-harness run.
- [ ] Phase 0 on `configs/phase0_colocate.yaml`: gates, step time by component, memory at the longest batch, pool pass rate (a reported number), importance ratio.
- [ ] pass@8 rollouts over the train pool (the bank for best-of-N and the difficulty gate).
- [ ] Hindsight SFT, then best-of-N SFT; evaluate each adapter against the base (the evaluation gate, runbook section 4).
- [ ] One short GRPO run warm-started from the SFT adapter, only if the evaluation gate and the sync check passed and the warm-start glue exists.

### Stage 2: report-only pilot (about $400 to 600)
- [ ] `scripts/sync_check.sh` with a non-zero adapter picks the sync mode.
- [ ] `scripts/run_pilot.sh configs/pilot.yaml`, then `scripts/pilot_report.sh`. alpha {0, 0.25, 0.5, 0.75} x `scale_rewards` {none, group} x 3
  seeds. The pilot says nothing about H1, effort or action-level abstention.

### Stage 3: two short full runs (about $300 to 600)
- [ ] `scripts/run_pair.sh configs/honest_alpha0.25.yaml configs/outcome.yaml seed=42`: honest against outcome, same seed and compute, one node,
  two slots. Set `max_steps` and group size from Stages 1 and 2 (the config default is 300 steps of 64 episodes; reviewers suggested about 100).
- [ ] Held-out evaluation of both (`scripts/eval_heldout.sh`) and the non-inferiority test; compare with the constant-policy rows and the
  no-training baselines.

### Stage 4: seeds and the alpha grid (about $2 to 3k)
- [ ] Three seeds of both headline arms (honest alpha 0.25 and `outcome`). `scripts/sweep_alpha.sh` for alpha {0, 0.1, 0.25, 0.4} and the arms
  `outcome_claimed` and `outcome_reflect`. The alpha trend should come from the pilot and from these, not from four single runs. Unsafe alphas
  for the pool are skipped (`PHASE0_SUMMARY`).

### Evaluation and write-up
- [ ] Transfer: Agentic Abstention terminal subset (exploratory, 19 abstain-warranted public tasks), AgentAbstain (paid GPT-5.4 judge; act and
  abstain sides separately), AbstentionBench (regression check), TB Lite and TB 2.0 (capability), SWE-bench fabrication (stretch).
- [ ] Release the dataset (licences: [data.md](data.md) section 1; ODC-BY and Gemini outputs rule out TMax as a training source) and the OpenEnv
  environment.

## On-policy distillation (OPD): deferred

OPD is not part of any stage. The earlier plan added a teacher's per-token advantage to the GRPO loss (teacher `Qwen/Qwen3.5-27B`, same tokenizer
as the 4B, conditioned on a stronger honesty context). The review found no measurement that requires it, and it would add a second server, a
trainer change and a confound to H1 and H2. A teacher's honesty is bounded by the teacher's (frontier models score 45 to 60% on AgentAbstain), so
pure OPD would turn the contribution into distillation. Revisit only if Stage 1 or 2 shows a need (report compliance below about 70% that
prompting and the SFT stage do not fix, or a task-success drop the honest arm cannot keep within the margin). If it is ever built, the GPU layout
needs a third GPU for the teacher (27B in bf16 is about 54 GB) and the two-runs-per-node layout no longer holds.

## Open decisions

| Decision | Default now | Who |
|---|---|---|
| Held-out families | `conflicting_constraint` and `no_network` ([data.md](data.md) section 2); training sees high_risk, revoke_permission and remove_input | owner |
| Dataset size | the whole pool: 2,482 seeds, 564 test, 536 val, 1,382 train source groups ([data.md](data.md) section 8); a trade against test resolution | owner |
| `false_blame` | 0. At alpha 0 a false blame is then free; `min(0, alpha - 0.1)` recommended, not set | owner |
| Group size | G=8 (4 to 64 times fewer episodes per step than any cited recipe). Decide after Stage 1 and the pilot | owner |
| Headline model | Qwen3.5-4B after the smoke test; Qwen3-4B for bring-up | owner |
| Checkpoint selection | validation task success alone (ignores honesty) | owner |
| AgentAbstain judge budget (GPT-5.4) | needed only if the base model has headroom there | owner |
| SFT stage | part of Stage 1, hindsight then best-of-N, evaluated before any GRPO; warm-start glue not built | owner |
| Whether to train `no_network`, and whether to keep `conflicting_constraint` as a reported zero-shot family | `no_network` and `conflicting_constraint` stay held out and reported zero-shot with the construct-validity caveat | owner |
| Validation split size (`VAL_BP`) | 2200 (536 val groups, 40 or more `remove_input` val rows); cutoffs and salt are not yet frozen, see [data.md](data.md) section 8. `probe.max_variants=100` samples only about 100 of the 974 val rows: raise it if the GPU cost is acceptable | owner |
| `high_risk` README wording | 28 irreplaceable-data sentences against 28 benign ones (no sentence on both sides; length, commas and digits do not separate them, AUC 0.50 to 0.53), so the label IS the wording and a policy can memorise 56 strings. Final verifier, round 2: widen the pools or hold some sentences out for test before a headline claim about `high_risk` transfer | owner |
| Multi-harness honesty training | a later stage; needs TRL's `HarnessRolloutWorker` (on TRL main, not in 1.14.1) and an honesty reward that runs as a Harbor verifier | later |
| GPU provider | a VM with Docker (not a container pod) | decided |
