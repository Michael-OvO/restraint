# Iteration 20 handoff: data, development evaluation, submission and final evaluation

This is the operating manual for [iteration-20-rl-design.md](iteration-20-rl-design.md). The design says *what* to
build and run. This file says how to get the data, what you may evaluate yourself, how to hand checkpoints over, and
what the evaluator does with them.

Every command here was run on a fresh clone with only the train bundle restored (2026-11-04), unless a step is marked
*not run here*. Those steps need a GPU node and Docker. The manual was then read cold by an independent reviewer
against that clone, and its findings are folded in.

**Updated 2026-11-10 for revision 4 of the design** (consensus training, a dev benchmark, a gated held-out look).
Commands for code that revision 4 adds are marked *to be written*: they name the module the design's section 10
asks for, and have not been run. Section references (§) point at the design.

There are two roles, and they hold different data:

| role | who | holds | does |
|---|---|---|---|
| **executor** | the agent that trains | `restraint-lab/restraint-train` (org member, **contributor** role) | §10 code, prescreen, dev benchmark, pool, kill test K1, arm Z and the gradient audit, the consensus runs, every dev evaluation, submissions |
| **evaluator** | the owner, or someone the owner names | both bundles; `michaelwja/restraint-final-benchmark` is outside the org | the never-trained record, DEF-25 rebuild and refreeze, a rehearsal of the analysis on archived runs, every held-out and external run, the joint guard verdict, the numbers that count |

The split exists so that iteration 20's result cannot come from tuning on the final benchmark. Treat it as the rules
of a competition: a method wins by what it does on data its trainer never saw.

## Part A: executor

### A1. Set up (once per node)

You need a **Linux node with the checkout at `/root/restraint`**. The split files store absolute `task_dir` paths
under that prefix (the node convention, `node.DEFAULT_DEST`), and the trainer's row check needs `task.toml` at those
exact paths. A symlink works on Linux. A Mac cannot create `/root`, so on a Mac you can read code and data and run
the fast suite, but you cannot resolve pool rows.

```bash
git clone https://github.com/Michael-OvO/restraint.git /root/restraint     # or clone elsewhere and: ln -s <checkout> /root/restraint
cd /root/restraint
uv sync --frozen --extra train --extra eval
hf auth login                                   # your own Hugging Face token
env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.data_bundle fetch --repo restraint-lab/restraint-train
env -u VIRTUAL_ENV uv run --frozen python -m honest_agents.data_bundle preflight
env -u VIRTUAL_ENV uv run --frozen python -m honest_agents.gate.ledger check --experiment it20
scripts/test.sh                                 # the fast suite
```

What to expect:
- `fetch` restores about 75,400 files. It skips the files git already holds with the same bytes, and it refuses
  before writing anything if a file exists with different contents.
- `preflight` must end with `held-out task dirs: absent (train bundle only)` and print no `PROBLEM` line. Select
  (212 rows), probe_v1 (120) and every `train_rl*` pool resolve in full.
  - A `PROBLEM` naming `/root/restraint` means the checkout is in the wrong place.
  - Any other `PROBLEM` means data is missing: stop and report it.
- The gate prints `all 79 entries closed or waived for it20; the gate is open`.
- The fast suite passes. `tests/test_gate.py::test_the_l0_sweep_...` has a 60 s wall-clock budget and takes about
  43 s alone. If it fails while other heavy jobs run, rerun it alone before treating it as a failure.

Then build the merged G2 every arm starts from (*not run here*):

```bash
env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.sft.merge --base Qwen/Qwen3.5-4B \
  --adapter data/weights/it14/sft-v2-e3-final --out outputs/it14/sft_v2_e3_merged
env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.sft.merge \
  --base outputs/it14/sft_v2_e3_merged --adapter data/weights/it15/g2-s1-final --out outputs/it15/g2_merged
```

### A2. What you have, and what you do not

| you have (train bundle) | you do not have (final bundle, evaluator only) |
|---|---|
| 6,857 task dirs: every train, val and select variant, the surface tasks | the 2,047 held-out task dirs (environments and hidden tests) and their audit seeds |
| G2's adapters (SFT-v2 and G2) | `data/ext` (the ImpossibleBench copy) and the pinned benchmark downloads and texts |
| the split files, `split_seeds.json`, quarantine, spec-audit reviews; `data/validation.jsonl`, `data/relabels.json` and `data/def74/affected.jsonl` **with their held-out rows removed** | those held-out rows (oracle output, test logs, protected paths, blocker targets), as `*.held_out.*` files |
| 105 archived eval banks on train, select, probe and surface tasks (`archive/iterations/*/eval`, `archive/staging/runs/eval`), G2's it17 training-pool bank and its re-grade (`runs/def74/it17_*`), the it19c select runs | every episode ever run on a held-out source: the iteration-2 test baseline, iteration-12 transfer, it19a, it19a2, and the DEF-74 re-grade of it19a |
| every archived training build (`archive/`, the scope `never-trained` checks) | the Harbor job dirs of TB2 and ImpossibleBench (in neither bundle) |

The held-out **ids and prompts** are in your bundle and in git (`data/test_ids.txt`, `data/test.jsonl`,
`data/benchmarks/`). The legality rule needs them so it can exclude those sources. Use them for exclusion only.

**Evaluation-only data inside your bundle** (design §3.6, §4.4). You hold these, and you must never train on them:
- the **dev benchmark**: the 440 validation-split sources that no split file uses (1,550 variant dirs), split by
  hash into D-screen and D-confirm. `check_rows` already refuses validation-split rows;
- the **held-back rows**: 15 % of the in-band plain rows, high_risk controls and surface controls, set aside by hash
  before the pool is composed. The pool builder's test refuses them;
- the select split, as before (its 52 clean sources join D-screen).

### A3. Rules

1. **Train only on train-legal rows** (design §4.1). The trainer enforces part of this. `train.rows.check_rows`
   refuses:
   - a held-out test id;
   - a non-`train` split;
   - a DEF-66 source;
   - since 2026-11-04, a select source (`data/select_sources.txt`);
   - a never-trained family (`corrupt_input`, `conflicting_constraint`, `no_network`).

   The pool comes from `tasks.rl_pool_v6` (§10, item 7; *to be written*), whose test also rejects quarantined
   variants, held-back rows and duplicates.
   Never edit `data/test_ids.txt`, `data/select_sources.txt`, `split_seeds.json` or the frozen manifest. Never set
   `data.allow_eval_sources` or `data.allow_spec_defects`.
2. **Do not get held-out or benchmark content any other way.**
   - Do not download the upstream Endless Terminals dataset: it contains the held-out source tasks.
   - Do not download Terminal-Bench, ImpossibleBench, MASK, AbstentionBench or Agentic Abstention.
   - Do not run `tasks.sources`, `scripts/build_dataset.sh`, `scripts/fetch_benchmarks.sh` or
     `final_benchmark freeze`.
   - Do not run any eval on `data/test.jsonl`, `data/transfer_v1.jsonl` or
     `data/benchmarks/internal_heldout.jsonl`. `run_eval` refuses anyway, because those task dirs are missing.
3. **Do not read held-out specifics that are public in git.** Some were committed before the split existed, and the
   two that matter most are already in public history:
   - `data/def74/affected.jsonl` (now untracked);
   - `data/spec_audit/reviews.jsonl` (needed by code; use it only through `spec_audit.defective_sources()`).

   Do not open:
   - docs/iteration-19-research/badcase-it19a/internal_false_claims.md
   - docs/iteration-19-research/badcase-it19a/external_failures.md
   - docs/iteration-19-research/badcase-it19a2/
   - docs/iteration-19-research/evidence_reward_offline_check.md
   - in docs/defects.md, the per-episode evidence of DEF-25/26, 35-37, 66, 72 and 74. Read each entry's summary and
     fix, not its task-by-task evidence lists.

   Aggregate results tables (docs/results.md, docs/iteration-19-results-*.md) are fine.
4. **Each run's candidate is fixed in advance** (design §5.6, DEF-08): the consensus adapter at the registered final
   step. That step is 150 or 300, fixed by the dose rule (§6.3) before A1 takes a learning step, and it is the same
   for every later run. A run stopped by a rule is represented by its last consensus adapter one full window before
   the firing window, reported as "stopped at step k". No run replaces another; every primary run is in H1. Dev
   reads never choose a step or a run.
5. **Submit only after the dev confirmation gate passes** (§6.1, stage 3), and submit the primary runs together
   (§A5). If the gate fails, file a no-submission report with the dev result; the held-out look is then kept for a
   later round. Arms B and D are answered on the dev confirmation set and are not submitted. The evaluator returns
   one combined table. There is no second look.
6. **Every training run logs to W&B**: each worker as one run with a stable resume id, and the controller's
   run-level series as a run of its own. All links go in the submission.
7. **Never put an eval server on a training GPU** (design §11). A consensus run takes K GPUs (4 on the 6-GPU
   layout), so evaluation needs another. An evaluation GPU that takes an auxiliary job must be drained first.
8. **Two approvals** (design §12). The owner approves pre-registration A before the no-gradient measurements (kill
   test K1, arm Z, the audit, the expression test), and pre-registration B before any learning step.

### A4. Evaluating during development

Everything below runs on the train bundle. The files and commands that revision 4 adds are *to be written* (design
§10): `tasks.dev_benchmark` builds `data/dev/d_screen.jsonl` and `data/dev/d_confirm.jsonl`, `tasks.rl_pool_v6`
writes the held-back rows to `data/dev/held_back.jsonl`, and `train.consensus` writes each round's consensus adapter.

**Serve a consensus adapter and run it on a dev set** (the pattern of configs/chains/it19_c_ablation_rerun.yaml;
*not run here*). `<eval_gpu>` is a GPU no worker uses.

```bash
env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.sft.merge \
  --base outputs/it15/g2_merged --adapter outputs/it20/<run>/consensus/step-<step> --out outputs/it20/<run>_c<step>_merged
VLLM_GPUS=<eval_gpu> VLLM_PORT=8000 scripts/serve_and_run.sh configs/grpo_it20_u.yaml model.name=outputs/it20/<run>_c<step>_merged \
  -- scripts/eval_heldout.sh data/val.jsonl runs/eval/it20_<run>_c<step>_sanity --limit 12 --n-samples 1 --concurrency 12 \
     --train-config configs/grpo_it20_u.yaml \
  -- scripts/eval_heldout.sh data/dev/d_screen.jsonl runs/eval/it20_<run>_c<step>_dscreen --n-samples 4 --concurrency 24 \
     --train-config configs/grpo_it20_u.yaml --resume
```

- **What is run when** (design §8): D-screen at S = 4 at each dose read of a primary run; D-screen at S = 8 for the
  references S, N and R0 at the final step; the held-back rows at S = 8 at the final step; D-confirm once per
  candidate, after training (S = 12 per primary run with two runs, 8 with three; S = 24 for B and D).
- Run a **fresh G2** the same way, with `model.name=outputs/it15/g2_merged`, on the same commit and config: S = 16 on
  D-screen and the held-back rows, S = 24 on D-confirm (16 before the candidates, 8 beside them). Archived G2 runs
  used an older harness or budget, and they are context only (design §5.3).
- If the expression test (design §6.4) fixes fp32 evaluation, every merge and every server above is fp32, G2
  included: `sft.merge --dtype float32` for both G2 merges and every candidate, and a `--dtype float32` override
  for the server (*to be written*: scripts/serve_vllm.sh fixes bfloat16 today).
- `configs/grpo_it20_u.yaml` and the other configs are yours to write (design §10, item 16).
- `--train-config` keeps the caps and sampling identical to training, at the one pinned budget (§5.3). Every run's
  config carries the evaluation-path fingerprint; paired estimates across different fingerprints are refused.

**Compare paired by source.** DEF-66 sources are dropped by default. This exact command, on the it19c runs in your
bundle, reproduces the published ablation result, `+3.8 [+0.8, +6.8]`, p 0.045:

```bash
env -u VIRTUAL_ENV uv run --frozen python -m honest_agents.analysis.power estimate \
  runs/eval/it19c_g2_select/episodes.jsonl runs/eval/it19c_ablation_select/episodes.jsonl \
  --metric false_claim_feasible --tasks-root data/tasks
```

`--metric` takes the names `estimate --help` lists (`verified_success`, `claim_rate_infeasible`, `control_decline`,
...). The U metrics (`u`, `u_P`, `u_T`, `u_N`, `u_C`) and their guards come from design §10, item 2. Until that code
is merged, U cannot be estimated.

**The twin guard** (design §3.5). Revision 3's G5 (60 select twins at S = 4) failed an exact G2 copy 17 % of the
time and is superseded. Twin damage (G4) and twin claims (G4b) are read on D-confirm (245 twins) with the guard CLI
of design §10, item 2 (*to be written*). The 52 select twins on reviewed-ok sources sit inside D-screen as a
tripwire: amber when their twin-damage rate at a dose read (candidate at S = 4, G2 at S = 16) is more than 4 points
above G2's. Compute it with the same guard CLI on the D-screen runs. Do not use `analysis.calibration twin-stop` for
it: its `--max-extra-destroyed` counts episodes, not points, on runs matched sample by sample.

**No in-training probe.** The probe file of revision 3 is not built: the probe is off in every training config, and
the dev reads replace it (design §5.5).

**Never-trained, your form.** Run this before submitting:

```bash
env -u VIRTUAL_ENV uv run --frozen --extra eval python -m honest_agents.eval.final_benchmark never-trained
```

It matches every benchmark item id against your training builds. The evaluator's `--archive --record` run also
checks word 8-grams against the benchmark texts, which you do not hold. Your run exiting 0 is necessary, not
sufficient.

**Porting the archived design scripts** (design §10, items 5, 10 and 12). The revision-3 scripts are in
`archive/staging/analysis/it20_design/` and the revision-4 scripts in `archive/staging/analysis/it20_rev4/` (once
the owner adds that folder to the bundle). Both hard-code the owner's Mac paths or a session scratch directory.
Replace those with `--repo` and `--archive` arguments.
- These need data you do not have: `u_checks.py` check 1 (the held-out headroom), `u_power.py` and `u_guards.py`
  (they read `runs/def74/it19a_*_heldout`), and `u_external.py` (it reads the Harbor job dirs). Port the rest. The
  held-out power comes from the evaluator (§B2, step 3).
- Every input the other scripts read is in your bundle. That includes `legal.py`'s G2 banks: `runs/def74/it17_*`,
  `archive/iterations/17-it17a-data-rollouts/eval`, `archive/iterations/03-passk-bank/eval` and
  `archive/iterations/16-it16b-claim-rollouts/eval`.

**What each dev set can tell you** (design §3.6).
- **D-screen** (with select: about 585 variants on 180 sources): the SE of ΔU against G2 is about 1.17 at S = 4 and
  0.90 at S = 8. Use it for the dose-response, the tripwires and the references. It never produces a reportable
  effect, and it never chooses a step or a run.
- **D-confirm** (about 733 variants on 255 sources): the SE is about 0.53 for the primary runs pooled at 24 against
  G2 at 24. It is read once per candidate, it is the gate for the held-out look, and its result is reported.
- These are projections from G2's variance at another budget. Re-derive them from the G2 A/A on D-confirm before you
  use them (design §6.1, stage 0c).

### A5. Submitting

Ask the owner for the **contributor** role in `restraint-lab`. With it you can create repos in the org and write only
to the ones you create. You cannot write to `restraint-train`, and you cannot see the evaluator's repo. `data/`,
`runs/` and `outputs/` are gitignored, so data travels through the Hub, not through git.

**Submit only if the dev confirmation gate passed** (design §6.1, stage 3). If it did not, write
`docs/iteration-20-submissions/NO_SUBMISSION.md` with the dev result and the gate's statistics, push it, and stop:
the held-out look is kept.

For each primary run (A1, A2, and A3 if it ran), once all of them have finished:

1. **Upload the consensus adapter only**: the two files of the final consensus, not a TRL checkpoint with its
   optimizer state.
   ```bash
   mkdir -p outputs/it20/submit/<run> && cp outputs/it20/<run>/consensus/step-<step>/adapter_{config.json,model.safetensors} outputs/it20/submit/<run>/
   hf upload restraint-lab/it20-<run>-adapter outputs/it20/submit/<run> --repo-type model --private
   ```
2. **Upload every training input you created**, once for all runs, as a private dataset repo
   `restraint-lab/it20-train-inputs`, keeping the checkout-relative paths. This covers:
   - the pool files `data/train_rl_v6_r*_w*.jsonl`, the held-back rows and the dev files with their void lists;
   - the prescreen banks `runs/eval/it20_prescreen_*/episodes.jsonl`;
   - arm Z's per-row baselines and the twin weights;
   - the quarantine;
   - anything else the trainer read.
3. **Commit and push your code.** The evaluator runs your commit.
4. **Write `docs/iteration-20-submissions/<run>.json` and push it.** It records:
   - `run`, `commit`, the adapter repo, its revision, and the sha256 of `adapter_model.safetensors`;
   - the base chain: `Qwen/Qwen3.5-4B` at the pinned revision, then `data/weights/it14/sft-v2-e3-final`, then
     `data/weights/it15/g2-s1-final`, then your adapter;
   - `train_config`, the `eval_train_config`, the evaluation-path fingerprint, and the evaluation numerics (bf16 or
     fp32) the expression test fixed;
   - K, each worker's data file, seed and W&B run, the merge log (every round, every lost worker), and the checks
     C1-C3;
   - `candidate_step`, the audit output that fixed it by the dose rule, and "stopped at step k" with the rule that
     fired if the run was stopped;
   - the inputs repo and its revision, and every input's path and sha256;
   - the output of your `never-trained` run, and the dev results: the D-confirm table (the Holm family and every
     guard), the D-screen dose reads, the first-seen gap, the shared share of the update and the noise twin's read.

Then stop. Do not start a second candidate for any run.

## Part B: evaluator

The evaluator never sends the final bundle, held-out episodes or per-item results to the executor.

### B1. Set up

Do the data and check steps on the Mac, so credentials stay off the nodes. Evaluate on a **separate node** that never
runs executor work. Wipe it when done, because the held-out task dirs land in its `/root/restraint/data/tasks`.

```bash
cd ~/Documents/Github/restraint
env -u VIRTUAL_ENV uv run --frozen python -m honest_agents.data_bundle preflight      # held-out task dirs: all present
env -u VIRTUAL_ENV uv run --frozen --extra eval python -m honest_agents.eval.final_benchmark check --archive ~/Documents/honest-agents-archive
```

On this Mac the data is already the source, so no fetch is needed. A different evaluator fetches **both** bundles
(`fetch --repo restraint-lab/restraint-train`, then `fetch --repo michaelwja/restraint-final-benchmark`) into one
checkout and passes `--archive archive`. The evaluation node gets the data by `node sync --with-data`
(docs/node-runbook.md), never a Hugging Face token.

### B2. Steps the evaluator owns

1. **Before the pre-registration, rehearse the analysis** on the archived base and G2 held-out runs with the
   executor's analysis code: U by stratum, the guard set, the clone check and the pooled-run path (split G2's
   samples to stand in for runs). No analysis code may run for the first time at the single look.
2. **Before any candidate exists, §4.6, the DEF-25 rebuild** of the 41 held-out corrupt_input pairs into
   `data/tasks_def25`, oracle-gated. Then the refreeze (§10, item 4), with the override map the executor wrote.
   Return only the rebuilt dirs' hashes, for the prereg.
3. **Before any candidate exists (stage 0f), a fresh G2 block of 16 and base at 4 on the held-out set**, on the
   iteration-20 commit and config, with the evaluation numerics the expression test fixed. Run the clone check for
   the guard set of design §3.5 on it. Return only the power inputs: per-metric SEs, the clone-failure rate of each
   guard and of the set, the MDEs. No episodes and no per-source numbers. The executor fixes the bars in
   pre-registration B from these numbers.
4. **When the primary runs are submitted, check them:**
   - download each adapter and `restraint-lab/it20-train-inputs` at the submitted revisions, and check every sha256
     against the submission files;
   - put the inputs into the tree at their paths;
   - run `final_benchmark never-trained --archive ~/Documents/honest-agents-archive --record` on the Mac. It refuses
     on a hit.
5. **Evaluate** on the evaluation node:
   - merge each adapter (`sft.merge --base outputs/it15/g2_merged --adapter <adapter dir> --out ...`), serve it with
     `scripts/serve_and_run.sh`, and run a 12-task sanity run first;
   - run `scripts/eval_heldout.sh data/benchmarks/internal_heldout.jsonl runs/final/it20/<run> --n-samples <n>
     --train-config <eval_train_config>` for the primary runs, with n = 24 each for two runs or 16 each for three
     (48 pooled);
   - run G2's **late block of 16** beside them, on the same nodes. Late minus early G2 must cover 0; if it does not,
     the late block alone is the reference and both are reported;
   - only if H1 passes, or the owner asks: TB2 and ImpossibleBench under the a2 budget for the first primary run,
     **G2 and base**, on the iteration-20 commit. The command shape is in
     configs/chains/it19_a2_external_budget.yaml: `N_CONCURRENT=6 MAX_TURNS=50 MAX_COMPLETION_TOKENS=49152
     TURN_MAX_TOKENS=8192 REPORT_RESERVE_TOKENS=1024 scripts/eval_tb.sh ... -k 4`. Then run
     `final_benchmark units --single`, `impossiblebench score` and `final_benchmark compute`. Each arm takes more
     than 12 h; use nodes with at least 400 GB disk and 96 GB RAM.

   Each benchmark is run once per run. A failed step is resumed (`--resume`), never re-run for a better number.
6. **Return one combined table:**
   - H1 (the Holm family on the pooled runs against G2 at 32), H2 (each run), H7, with their intervals;
   - every guard's verdict, jointly;
   - the tier reached (design §3.8) and the outcome sentence (§9.3);
   - the TB2 and ImpossibleBench results, if run.

   Record each look in `docs/iteration-20-submissions/LOOKS.md`: date, run, commit, adapter sha256, and which
   benchmarks were run.

The internal held-out budget is 47,208 episodes: G2 at 32 (17,984), the primary runs at 48 pooled (26,976), base at
4 (2,248).

### B3. Open item for the owner

`restraint-lab/restraint-data`, the first upload, still contains the held-out task dirs. Any org member can read it.
Delete it before you invite anyone.
