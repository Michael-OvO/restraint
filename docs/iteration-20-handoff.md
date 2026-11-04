# Iteration 20 handoff: data, development evaluation, submission and final evaluation

This is the operating manual for [iteration-20-rl-design.md](iteration-20-rl-design.md). The design says *what* to
build and run. This file says how to get the data, what you may evaluate yourself, how to hand checkpoints over, and
what the evaluator does with them.

Every command here was run on a fresh clone with only the train bundle restored (2026-11-04), unless a step is marked
*not run here*. Those steps need a GPU node and Docker. The manual was then read cold by an independent reviewer
against that clone, and its findings are folded in.

There are two roles, and they hold different data:

| role | who | holds | does |
|---|---|---|---|
| **executor** | the agent that trains | `restraint-lab/restraint-train` (org member, **contributor** role) | §10 code, prescreen, pool, pilot, arms, select-split, probe and G5 evaluation, submissions |
| **evaluator** | the owner, or someone the owner names | both bundles; `michaelwja/restraint-final-benchmark` is outside the org | the never-trained record, DEF-25 rebuild and refreeze, every held-out and external run, the joint guard verdict, the numbers that count |

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

### A3. Rules

1. **Train only on train-legal rows** (design §4.1). The trainer enforces part of this. `train.rows.check_rows`
   refuses:
   - a held-out test id;
   - a non-`train` split;
   - a DEF-66 source;
   - since 2026-11-04, a select source (`data/select_sources.txt`);
   - a never-trained family (`corrupt_input`, `conflicting_constraint`, `no_network`).

   The pool comes from `tasks.rl_pool_v5` (§10, item 6), whose test also rejects quarantined variants and duplicates.
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
4. **Each arm's candidate is fixed in advance:** step 150, or the arm's last checkpoint (every 5 steps) before a hard
   stop (design §5.5, DEF-08), reported as "stopped at step k". This applies to A1, A2, A3, B and C alike. A2 and A3
   never replace A1 in H1. Select-split results are for development and the stop rules. They do not choose a
   candidate.
5. **Submit all arms at once** (§A5). The evaluator returns one combined table after the last arm is in, so no arm's
   held-out result can shape another arm. There is no second look.
6. **Every training run logs to W&B**, and the link goes in the submission.
7. **Never put an eval server on a training GPU** (design §11). Five colocated runs take five GPUs, so evaluation
   needs a sixth.

### A4. Evaluating during development

Everything below runs on the train bundle.

**Serve a checkpoint and run it on the select split** (the pattern of configs/chains/it19_c_ablation_rerun.yaml;
*not run here*). `<eval_gpu>` is a GPU no training run uses.

```bash
env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.sft.merge \
  --base outputs/it15/g2_merged --adapter outputs/it20/<run>/checkpoint-<step> --out outputs/it20/<arm>_c<step>_merged
VLLM_GPUS=<eval_gpu> VLLM_PORT=8000 scripts/serve_and_run.sh configs/grpo_it20_u.yaml model.name=outputs/it20/<arm>_c<step>_merged \
  -- scripts/eval_heldout.sh data/val.jsonl runs/eval/it20_<arm>_c<step>_sanity --limit 12 --n-samples 1 --concurrency 12 \
     --train-config configs/grpo_it20_u.yaml \
  -- scripts/eval_heldout.sh data/select.jsonl runs/eval/it20_<arm>_c<step>_select --n-samples 4 --concurrency 24 \
     --train-config configs/grpo_it20_u.yaml --resume \
  -- scripts/eval_heldout.sh data/it19/select_high_risk_twins.jsonl runs/eval/it20_<arm>_c<step>_twins --n-samples 4 \
     --concurrency 24 --train-config configs/grpo_it20_u.yaml --resume
```

- Run a **fresh G2** the same way, with `model.name=outputs/it15/g2_merged`, on the same commit and config. Archived
  G2 runs used an older harness or budget, and they are context only (design §5.3).
- `configs/grpo_it20_u.yaml` (and the B and C configs) are yours to write (design §10, item 9).
- `--train-config` keeps the caps and sampling identical to training, at the one pinned budget (§5.3).

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

**G5, the twin guard** (design §3.5). G5 runs on select, so you run it, and the evaluator re-runs it for the verdict
(§B2, step 4):

```bash
env -u VIRTUAL_ENV uv run --frozen python -m honest_agents.analysis.calibration twin-stop \
  --reference runs/eval/it20_g2_twins --candidate runs/eval/it20_<arm>_c<step>_twins --tasks-root data/tasks \
  --spec-defects include --expect-matched 240 --max-extra-destroyed <k>
```

- `--spec-defects include` is the CLI default; it is written out here because the library default is `exclude`.
- The command covers all 60 twins × 4 = 240 episodes.
- k comes from the clone check (provisional 6).
- Keep the twin run at S = 4. The command refuses if fewer than 240 episodes match.

**The in-training probe.** You build `data/probe_v1_def66.jsonl` yourself: `data/probe_v1.jsonl` without the rows
whose `pair_id` ends in one of the 5 DEF-66 hashes (009a1afa, 089a919e, 64c0ee72, 658f04e0, f81b900c). The `source`
column is the dataset name and identical on every row, so filter on `pair_id`. That leaves 103 rows (69 feasible, 34
infeasible). Record the file's sha256 in the prereg. The probe runs inside training (`probe.every_steps` 15, S = 4).

**Never-trained, your form.** Run this before submitting:

```bash
env -u VIRTUAL_ENV uv run --frozen --extra eval python -m honest_agents.eval.final_benchmark never-trained
```

It matches every benchmark item id against your training builds. The evaluator's `--archive --record` run also
checks word 8-grams against the benchmark texts, which you do not hold. Your run exiting 0 is necessary, not
sufficient.

**Porting the archived design scripts** (§10, item 10). The scripts in `archive/staging/analysis/it20_design/`
hard-code the owner's Mac paths. Replace those with `--repo` and `--archive` arguments.
- These need data you do not have: `u_checks.py` check 1 (the held-out headroom), `u_power.py` and `u_guards.py`
  (they read `runs/def74/it19a_*_heldout`), and `u_external.py` (it reads the Harbor job dirs). Port the rest. The
  held-out power comes from the evaluator (§B2, step 2).
- Every input the other scripts read is in your bundle. That includes `legal.py`'s G2 banks: `runs/def74/it17_*`,
  `archive/iterations/17-it17a-data-rollouts/eval`, `archive/iterations/03-passk-bank/eval` and
  `archive/iterations/16-it16b-claim-rollouts/eval`.

**What the select split can and cannot tell you.** It has 212 variants at S = 4, and 52 sources once DEF-66 is
excluded. A difference of about 3 points is at its noise floor. Use it to catch regressions and to apply the stop
rules. Do not use it to forecast the held-out result.

### A5. Submitting

Ask the owner for the **contributor** role in `restraint-lab`. With it you can create repos in the org and write only
to the ones you create. You cannot write to `restraint-train`, and you cannot see the evaluator's repo. `data/`,
`runs/` and `outputs/` are gitignored, so data travels through the Hub, not through git.

For each arm, once all arms have finished:

1. **Upload the adapter only**: the two files, not the TRL checkpoint with its optimizer state.
   ```bash
   mkdir -p outputs/it20/submit/<arm> && cp outputs/it20/<run>/checkpoint-<step>/adapter_{config.json,model.safetensors} outputs/it20/submit/<arm>/
   hf upload restraint-lab/it20-<arm>-adapter outputs/it20/submit/<arm> --repo-type model --private
   ```
2. **Upload every training input you created**, once for all arms, as a private dataset repo
   `restraint-lab/it20-train-inputs`, keeping the checkout-relative paths. This covers:
   - the pool files `data/train_rl_v5_s*.jsonl` and `data/probe_v1_def66.jsonl`;
   - the prescreen banks `runs/eval/it20_prescreen_*/episodes.jsonl`;
   - the twin weights;
   - the quarantine;
   - anything else the trainer read.
3. **Commit and push your code.** The evaluator runs your commit.
4. **Write `docs/iteration-20-submissions/<arm>.json` and push it.** It records:
   - `arm`, `commit`, the adapter repo, its revision, and the sha256 of `adapter_model.safetensors`;
   - the base chain: `Qwen/Qwen3.5-4B` at the pinned revision, then `data/weights/it14/sft-v2-e3-final`, then
     `data/weights/it15/g2-s1-final`, then your adapter;
   - `train_config`, and the `eval_train_config` (the `--train-config` for evaluation, normally the same);
   - `candidate_step`, and why (step 150, or "stopped at step k" with the stop rule that fired);
   - the inputs repo and its revision, and every input's path and sha256;
   - the output of your `never-trained` run, the W&B run URL, the select and twin run dirs, and your `power estimate`
     and `twin-stop` outputs.

Then stop. Do not start a second candidate for any arm.

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

1. **Before any candidate exists, §4.5, the DEF-25 rebuild** of the 41 held-out corrupt_input pairs into
   `data/tasks_def25`, oracle-gated. Then the refreeze (§10, item 4), with the override map the executor wrote.
   Return only the rebuilt dirs' hashes, for the prereg.
2. **Before any candidate exists, §12, item 6, a fresh G2 and base on the held-out set at S = 16**, on the
   iteration-20 commit and config. Run the clone check for G1-G4 and G6 on it. Return only the power inputs and the
   held-out half of the clone check: per-metric SEs, the clone-failure rate, the MDE. No episodes and no per-source
   numbers. The executor fixes k and the bars in the prereg from these numbers and from G5's clone check on select.
3. **When all arms are submitted, check them:**
   - download each adapter and `restraint-lab/it20-train-inputs` at the submitted revisions, and check every sha256
     against the submission files;
   - put the inputs into the tree at their paths;
   - run `final_benchmark never-trained --archive ~/Documents/honest-agents-archive --record` on the Mac. It refuses
     on a hit.
4. **Evaluate every arm** on the evaluation node:
   - merge each adapter (`sft.merge --base outputs/it15/g2_merged --adapter <adapter dir> --out ...`);
   - serve it with `scripts/serve_and_run.sh`;
   - run `scripts/eval_heldout.sh data/benchmarks/internal_heldout.jsonl runs/final/it20/<arm> --n-samples 16
     --train-config <eval_train_config>` for A1, A2, A3, B and C;
   - re-run G5 for each candidate against the fresh G2, as in §A4;
   - for A1, **G2 and base**, run TB2 and ImpossibleBench under the a2 budget, on the iteration-20 commit. The
     command shape is in configs/chains/it19_a2_external_budget.yaml: `N_CONCURRENT=6 MAX_TURNS=50
     MAX_COMPLETION_TOKENS=49152 TURN_MAX_TOKENS=8192 REPORT_RESERVE_TOKENS=1024 scripts/eval_tb.sh ... -k 4`. Then
     run `final_benchmark units --single`, `impossiblebench score` and `final_benchmark compute`. Each arm takes more
     than 12 h; use nodes with at least 400 GB disk and 96 GB RAM.

   Each benchmark is run once per arm. A failed step is resumed (`--resume`), never re-run for a better number.
5. **Return one combined table, after the last arm:**
   - H1-H4 with their intervals, per arm against the fresh G2 and base;
   - every guard's verdict, G1-G6 jointly;
   - the TB2 and ImpossibleBench deltas.

   Record each look in `docs/iteration-20-submissions/LOOKS.md`: date, arm, commit, adapter sha256, and which
   benchmarks were run.

### B3. Open item for the owner

`restraint-lab/restraint-data`, the first upload, still contains the held-out task dirs. Any org member can read it.
Delete it before you invite anyone.
