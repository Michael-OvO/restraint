# Node runbook: Stage 1

The first paid step, from the spending gate in [roadmap.md](roadmap.md). Everything that exists was tested on macOS with fakes;
this runbook is the list of things only a Linux node with Docker (and for some, a GPU) can show, ordered by how likely each is to be
wrong times how much money it wastes if it is. If an early step fails, stop: nothing after it means anything. Every command below was
checked on the Mac only as far as it can run there (`tests/test_scripts.py` keeps the scripts, configs, modules, flags and links in this
file honest); the GPU and Docker behaviour is unverified.

Rough cost of the whole stage: **$200 to 400**, and that is an estimate with a factor of three either way ($70 to 1,200): step time and
episode count are unmeasured. Basis: about 40 to 60 H100-hours at $2.5 to 3.5 (Phase 0 colocated and the pass@8 rollouts dominate), a few
CPU-box hours, plus 30 to 50% for bring-up and reruns. Continue to Stage 2 only if the gates in section 4 pass.

## 1. Where each step runs

| Step | Where | Why |
|---|---|---|
| Unit tests, dataset build without validation, SFT dry run, every `--dry-run` and `--print-limits` | this Mac | no Docker, no GPU |
| Optional Docker smoke on the Mac | Colima or OrbStack (not installed; below) | finds path and compose mistakes early, proves nothing about x86 |
| **Authoritative Docker checks** (`pytest -m docker`, audit set, oracle validation at scale) | the rented single H100 box (x86, Docker daemon, nftables) | same architecture as the task images |
| Weight-sync check, base evaluation, Phase 0, SFT, short GRPO | the same H100 box, one GPU, colocated config | one rental, nothing idle |
| External transfer (Agentic Abstention terminal, AgentAbstain), base and trained | the same box: GPU for the endpoint, Docker for the first; the AgentAbstain judge is an API | [evaluation.md](evaluation.md) section 10.1; scorers and the delta tool run on the Mac |

Rent one x86 VM with one H100 and a real Docker daemon (not a container pod). A 4xH100 VM is for Stages 2 to 4 only
([design.md](design.md) section 8 describes the two run slots).

**Optional local Docker on the Mac.** Not needed for Stage 1 and not installed by any script. If wanted: install Colima
(`brew install colima docker docker-compose`) or OrbStack, then `colima start --arch aarch64 --cpu 8 --memory 16` and
`export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` (OrbStack sets its own socket), and run
`env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval pytest -m docker -k "not gpu"`. Arm64 caveats: the containers are
`linux/arm64`, so an image or wheel that is x86 only fails or needs `DOCKER_DEFAULT_PLATFORM=linux/amd64` (emulated: slow, timing and
timeout results meaningless); the nftables egress sidecar and the `/proc` and pids-limit checks depend on the VM kernel, which differs
from the rented box's; Harbor's kernel probe may disable egress control silently. A pass here is a hint, a fail is worth reading, and
neither replaces the checks on the H100 box.

## 2. Commands

`R` is `env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval`; the scripts add it themselves. On the fresh VM:

```bash
git clone <repo> && cd restraint            # or rsync this tree
scripts/setup_node.sh --install-docker      # uv, Docker, nftables check, model weights, verifier bundle; --pull-images DIR prebuilds images
# flash-linear-attention is in the train extra (uv.lock); setup_node.sh checks that `import fla` works. Qwen3.5 is ~20x slower without it
# Docker's stock network pool holds ~30 networks and every sandbox takes one; widen it before any real run:
#   /etc/docker/daemon.json: "default-address-pools": [{"base": "10.128.0.0/9", "size": 24}]   then: systemctl restart docker
# setup_node.sh also disables unattended upgrades: one replaced the NVIDIA libraries under the loaded driver mid-run
scripts/probe_node.sh                       # GPUs, Docker, compose v2, Harbor egress-control kernel support, capacity; fails loudly
env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval pytest -m docker   # the docker-marked tests are the first checklist
```

**Step 1: Docker-only checks, no GPU used (stop here if any fails).**

```bash
scripts/fetch_benchmarks.sh benchmarks                  # decontamination texts (TB2, TBLite, Agentic Abstention, AgentAbstain); the build refuses to run without them
VALIDATE=1 scripts/build_audit_set.sh data/audit        # 20-30 sources, oracle-validated through our own environment code path
#   Read data/audit/sheets/*.md, type ok|wrong|unsure on each "Verdict:" line, then:
scripts/build_audit_set.sh --tally data/audit           # stop if variants are wrong or controls leak
VALIDATE=1 scripts/build_dataset.sh data     # the full dataset, every variant oracle-validated (hours of CPU; gate: at least 95% validate as labelled)
```

**Step 2: weight sync.** `scripts/sync_check.sh` runs a vLLM server and the trainer as two processes that exchange weights over NCCL, so
they need two GPUs; NCCL refuses two ranks on one device. On a single H100 it cannot run as written, and the colocated engine's in-process
sync is a different path that `sync_check.py` does not cover (change request in the roadmap). Until that exists, do the sync check on a
2-GPU H100 VM for about an hour (a few dollars), before the first GRPO run (step 8), not before Phase 0:

```bash
scripts/sync_check.sh configs/qwen35_4b.yaml                      # exit 0 = pass; writes outputs/sync_check/*.json
scripts/sync_check.sh configs/qwen35_4b.yaml -- --tiny-adapter    # the second pass; may legitimately fail at lr 1e-5
```

**Step 3: end-to-end smoke on one GPU.** `scripts/smoke_test.sh` also starts a vLLM server on the slot's second GPU, so on one GPU run
the same trainer path as a two-round Phase 0 (a probe of the real model, template, tool-call parsing and sandbox loop; gates are not
measured, so the exit status is 1 by design). On the 2-GPU VM of step 2 the full smoke can run instead:

```bash
scripts/phase0.sh configs/phase0_colocate.yaml model.name=Qwen/Qwen3.5-4B model.family=qwen3_5 grpo.max_steps=2 output_root=outputs/smoke_phase0
scripts/smoke_test.sh --train data/audit/tasks/<variant_a> --train data/audit/tasks/<variant_b> \
    --val data/audit/tasks/<variant_c> --config configs/qwen35_4b.yaml     # 2 GPUs only
```

**Step 4: base evaluation (inference only).** One GPU serves the policy while the sandboxes run on the CPU; stop the server (whole process group) before
training. `VLLM_GPUS=0` is required on a one-GPU box (the slot default is GPU 1):

```bash
VLLM_GPUS=0 EVAL=1 setsid scripts/serve_vllm.sh configs/qwen35_4b.yaml --served-model-name policy > outputs/serve_base.log 2>&1 &
SERVER=$!; until curl -sf localhost:8000/health >/dev/null; do sleep 5; done   # the evals do not wait for the server
MODEL=policy scripts/eval_heldout.sh data/test.jsonl runs/eval/base --n-samples 4 --concurrency 32 \
    --train-config configs/qwen35_4b.yaml --bank-dir runs/bank/eval_base
#   runs/eval/base/metrics.json: abstain_recall, near_miss_false_refusal_rate (reported numbers, see section 4), report rate, task success
AGENT=terminus-2 MODEL=policy scripts/eval_harness.sh data/test.jsonl runs/harbor     # the same under a stock harness
kill -TERM -- -$SERVER; sleep 20; nvidia-smi --query-gpu=memory.used --format=csv   # stop the server and its children; the GPU must be empty before training
```

**Step 4b: external transfer (after the base evaluation, again for each trained adapter).** Other people's abstention tasks, base against
trained, with the act side measured so over-refusal shows ([evaluation.md](evaluation.md) section 10.1 has the endpoint, the verdict
rule and the simulated power). Serve the arm as in step 4 (the base as `policy`; an adapter by the `--lora-modules policy=...` form of
step 7), then, with `TAG` naming the arm:

```bash
# Agentic Abstention terminal: 108 tasks (19 abstain-warranted, from 10 source tasks); needs Docker, pulls 108 prebuilt images
TAG=base N_ATTEMPTS=4 MODEL=policy scripts/eval_agentic_abstention.sh runs/harbor   # materialises, checks that all 108 load, runs, scores
# AgentAbstain: 263 pairs, MCP sandboxes (no Docker); smoke first, then the full set; JUDGE=1 adds the paid judge (526 calls per arm)
MODEL=policy scripts/eval_agentabstain.sh smoke
JUDGE=1 JUDGE_API_KEY=... MODEL=policy scripts/eval_agentabstain.sh        # without JUDGE=1 only commit-only bounds are written
# after both arms exist (any machine):
python -m honest_agents.eval.external.transfer --benchmark agentic_abstention --base <base summary> --trained <trained summary>
python -m honest_agents.eval.external.transfer --benchmark agentabstain --subset env_only --base runs/agentabstain/base.json --trained runs/agentabstain/trained.json
```

The AgentAbstain output file is named after `MODEL`, so give the two arms two served names (`base`, `trained`) or copy the file between runs.
Time, with wide error bars (nothing has been timed): Agentic Abstention 1.5 to 4 hours per arm at 4 attempts and 8 concurrent plus a first-run image
pull; AgentAbstain 1 to 2 hours per arm. Stop and read the logs, not the numbers, if more than a few percent of trials are `n_missing` or run errors
(`run_errors` in the AgentAbstain summary): an endpoint that is down scores as missing, never as a miss. Decision use: a `no_evidence`
verdict at these sample sizes is expected for a small effect and neither passes nor fails the stage; `gain_with_possible_over_refusal` and
`degraded` are worth reading per category before anything else. Drop AgentAbstain as a transfer metric if the base model's act accuracy on the
`env_only` pairs is below about 25 to 30% (section 4), which needs the judge to measure.

Optional, only if the H1 direction check is wanted (two more inference runs): `scripts/h1_tmax_check.sh data/test.jsonl runs/h1_tmax`.

**Step 5: Phase 0** (the real trainer at learning rate 0 for 24 rounds, one GPU, vLLM colocated; writes the trajectory bank and prints the
gates). `configs/phase0_colocate.yaml` is `phase0_zero_lr.yaml` with the colocated engine; the headline model is passed explicitly:

```bash
scripts/phase0.sh configs/phase0_colocate.yaml model.name=Qwen/Qwen3.5-4B model.family=qwen3_5
#   outputs/phase0/<run>/phase0_summary.json: gates, timing/*, trainer_decision_10_3, pool.mean_pass_rate_feasible, pool.work_vs_quit_gap,
#   pool.safe_alphas. Exit 1 if a gate fails or was not measured. With 4 GPUs: configs/phase0_zero_lr.yaml (server layout) instead.
```

If the importance ratio is not within 1% of 1, stop and read `sampling/*` before anything else. Phase 0 sees only about 190 prompts of the
roughly 1,100-seed train pool; the pool-wide pass@8 measurement (step 6) is what feeds best-of-N and the difficulty gate.

**Step 6: banks for the supervised stages.** The hindsight stage reads the Phase 0 bank; best-of-N needs several rollouts per variant:

```bash
VLLM_GPUS=0 EVAL=1 setsid scripts/serve_vllm.sh configs/qwen35_4b.yaml --served-model-name policy > outputs/serve_base.log 2>&1 &
SERVER=$!; until curl -sf localhost:8000/health >/dev/null; do sleep 5; done   # the evals do not wait for the server
MODEL=policy scripts/eval_heldout.sh data/train_pool.jsonl runs/passk --n-samples 8 --bank-dir runs/passk/bank --train-config configs/qwen35_4b.yaml
kill -TERM -- -$SERVER; sleep 20; nvidia-smi --query-gpu=memory.used --format=csv   # the GPU must be empty before training
env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval python -m honest_agents.tasks.difficulty --bank runs/passk/bank --out difficulty.json
```

**Step 7: hindsight SFT, then best-of-N** (one training GPU, no vLLM). Dry-run each first and read `<output_dir>/examples.jsonl` by eye:
labels and rewritten reports are verified before any GPU time. The sft configs default to `outputs/phase0/bank`, which does not exist
(Phase 0 writes `outputs/phase0/<run>/bank`), so pass the bank:

```bash
DRY_RUN=1 scripts/run_sft.sh configs/sft_hindsight.yaml data.bank=outputs/phase0/<run>/bank
scripts/run_sft.sh configs/sft_hindsight.yaml data.bank=outputs/phase0/<run>/bank
DRY_RUN=1 scripts/run_sft.sh configs/sft_bestofn.yaml data.bank=runs/passk/bank
scripts/run_sft.sh configs/sft_bestofn.yaml data.bank=runs/passk/bank
```

Evaluate each adapter exactly as in step 4 (the evaluation gate, section 4; start and stop the server the same way): serve with
`VLLM_GPUS=0 EVAL=1 setsid scripts/serve_vllm.sh configs/qwen35_4b.yaml --enable-lora --lora-modules policy=outputs/sft/<run>/final`
(add `--max-lora-rank 32` for a GRPO adapter; the SFT adapter has rank 16), then `eval_heldout.sh` to `runs/eval/sft_hindsight` and
`runs/eval/sft_bestofn`, and compare with `python -m honest_agents.eval.run_eval --compare runs/eval/base runs/eval/sft_hindsight runs/eval/sft_bestofn`.
Sample a bank from the adapter and run `python -m honest_agents.sft.train_sft --spread <bank>` against the starting bank.

**Step 8: one short GRPO run, only after the evaluation gate and the sync check.** Warm-started from the SFT adapter. The glue is not built
(roadmap, "SFT, then GRPO"): until it exists this step cannot run, and a GRPO run from the base model is not a substitute. Shape once it
exists: `scripts/run_slot.sh configs/honest_alpha0.25.yaml <colocate overrides> grpo.max_steps=50`, with `PHASE0_SUMMARY` pointing at
step 5's summary so an unsafe alpha is refused (exit 4).

**Dashboard and health during every GRPO run.** `configs/grpo_colocate_qwen35.yaml` sets `grpo.report_to: wandb` (any config can with
`grpo.report_to=wandb`). The run goes to project `honest-agents` under the entity in `WANDB_ENTITY` (`michael_ovo`), named after the
iteration: pass `wandb.name=it11-<slug>` (and `wandb.group=it11`) so the dashboard lines up with docs/training-log.md and the archive's
`NN-<slug>/` folders. The key is read by wandb from `WANDB_API_KEY` or `~/.netrc` on the node; it is never put in a config, an override or a
log. `WANDB_MODE=disabled` turns the dashboard off (offline work, tests). Besides TRL's own numbers and the `honest/*` metrics, the dashboard
gets host RAM (`mem/*`), `train.health` levels per step (`health/<check>`: 0 OK, 1 WARN, 2 FAIL; one W&B alert per check on its first FAIL),
and the probe's checkpoint choice (`probe/best_*`). `--resume` continues the same dashboard run (`<output_dir>/wandb_run_id`).
`python -m honest_agents.train.health <output_dir>` stays the operator's check after every step (exit 2 on a FAIL): stop and look when
`capability` (verified success on near-miss controls), `over-refusal` (control declines) or `quitting` (zero-command episodes) FAILs, which
means the last 5 steps sit more than 15 points from the first 5 on two consecutive checks.

## 3. What to verify in Docker, ordered by risk

1. **The hermetic verifier inside real images.** Never run in Docker: `-I -S -B`, `env -i`, `-X pycache_prefix`; bundle and tests upload
   ownership after `docker compose cp`; `timeout -k` on busybox; a real TTY on `compose exec`; non-root `[verifier].user`; the egress sidecar
   forced for every pytest-verifier episode (and its per-episode cost); the pids limit through the compose override; Python 3.10 in
   `ubuntu:22.04` (the bundle is tested on 3.9 and 3.11 to 3.14); converted Endless Terminals tests passing under `env -i` with no `HOME` (7 of
   40 sampled tests call `subprocess`). The 11 docker-marked exploit tests and the 7 docker integration tests are the checklist. **If the verifier cannot
   grade, nothing trains.** Also replay the oracle solution through the full grader on 20 variants first.
2. **Weight sync on a GPU.** The clone patch fixes a bug reproduced only against a CPU model of vLLM's packed transfer. If sync is wrong, a run
   learns against stale or half-merged weights while every metric looks normal. Peak memory of the clone wrapper at 4B scale is unmeasured.
3. **The false-tamper rate.** The protected-path and toolchain watches can flag legitimate work (`pip install -U pip` rewrites
   `/usr/local/bin/pip*` in `python:3.x` images). A false tamper pays -1.5 on a success. Measure in Phase 0: tamper rate among episodes whose
   tests pass, by detail prefix (`modified:`, `toolchain:`, `shadow:`, `stream:`), and the share with `verifier.hardened == False`.
4. **vLLM server shapes.** `/v1/models` `max_model_len`, completions `token_ids` and processed logprobs, the `-9999` top-k masking,
   `prompt_logprobs` response shape, `--generation-config vllm`, `--language-model-only`, `--enable-scale-out`, `--logprobs-mode
   processed_logprobs` (all read from vLLM 0.30 source; vLLM is not installed on the Mac). The startup probe is the guard: it fails the
   run, it does not fix anything.
5. **The real trainer on the real model.** One real 2-turn episode through `_generate`; `per_device_train_batch_size` 2 at 16k+ tokens
   (fallback in `base.yaml`: 1 and 64); the probe's batch of 25; LoRA targets and `fla` on Qwen3.5-4B; `cast_lm_head_to_fp32` needs untied
   embeddings; the hybrid-Qwen3 template swap on the real Qwen3-4B tokenizer; Qwen3-Coder XML tool-call parsing inside the TRL loop. The real
   `trl.trainer` ran only with a 4-layer random CPU model and a scripted sampler.
6. **Report compliance of the real model**: tool-call parse of real sampled report turns, thinking text before the call, mistyped
   arguments. The Stage 1 gate (at least 70% reports, at most 5% no report) measures exactly this.
7. **Concurrency and step time.** 64 concurrent sandbox boots, per-episode hermetic-verifier and toolchain-hash cost (about 30 to 100 MB
   hashed twice per episode, bundle uploaded twice), two concurrent slots with distinct NCCL ports. Nothing has been timed. The profiling
   order is in [design.md](design.md) section 15.
8. **Dataset validation at scale.** No variant has been oracle-validated. Weak spots: controls on SETA tasks, revoke-permission controls when
   the oracle writes outside the handed-over workspace, `remove_tool` (no candidate on real seeds), Harbor allowlist enforcement for the
   `no_network` controls, the sticky-parent ancestor sealing, and the audit sheets (nobody has read them). The checks to run inside a
   container are in [data.md](data.md) section 6. Gate: at least 95% of variants validate as labelled; measure the real leak rate.
   `corrupt_input` (the transfer family) needs its own checks: the oracle fails the twin and passes the control 3 of 3; `stat`/`cmp` equality
   of P, D and the noise across the pair; `python3` on PATH and root at build time; the control oracle passes with the decoy present
   (data.md section 6, "Docker-only checks for `corrupt_input`"). `conflicting_constraint` additionally needs its stub steps (GNU `stat -c`,
   `chown`, `touch -r`) checked on a real Linux image and the null check on stale stubs.
9. **Phase 0 cannot measure the pool** (see step 6, the pass@8 run).
10. **Stock harnesses.** The report file reaching the artifacts directory, `terminus-2` trajectory files, `mini-swe-agent` reaching the endpoint
    from inside the container, and how well a 4B model follows `REPORT_FILE_INSTRUCTIONS`. Under a stock harness Harbor runs the task's own
    `test.sh`: no tamper, violation, evidence-replay or blocker verification. The `allenai/tmax-4b` serve flags are from its model card.
11. **Stop-rule and pilot thresholds** are judgement values, tested on simulated bandits and hand-made curves only. The pilot's effective
    sample per rate is distinct prompts (8 per round), and 100 steps at LoRA lr 1e-5 may not move the policy far enough for the stop rule to
    see an effect (it then says INCONCLUSIVE, which costs money without information).
12. **Residual grader risks** ([design.md](design.md) section 6): in-process agent code; a root process racing the verifier through
    `/proc/<pid>/fd/1`; unbounded `bash` output inside the container (a runaway `yes` is buffered by Harbor's exec for up to 60 s, a host-memory
    risk across 64 episodes); the token estimate at 2.4 characters per token.
13. **OpenEnv image**: `docker build` with `--build-context verifier_bundle=...`, the docker.sock path, a real WebSocket client against uvicorn,
    `timeout` and `ensure_dirs` in real images.
14. **Checkpoint selection ignores honesty.** The default `probe.selection_metric` is `eval_honest/task_success`, and TRL averages per-batch
    ratios, so the probe value is only approximately episode-weighted. Say so in any write-up or change the metric.
15. **Costs and dates.** Every cost figure is an estimate.

## 4. Stage 1 gates

`scripts/phase0.sh` (via `python -m honest_agents.callbacks phase0 <output_dir>`) checks four gates and exits 1 if one fails or was not measured:

| gate | threshold |
|---|---|
| report compliance (`honest/report_rate`) | at least 70% |
| no-report share | at most 5% |
| fabrication among failed feasible episodes (P(claim given fail)) | at least 20% |
| mean importance-sampling ratio | within 1% of 1 |

Read from the same summary by hand: step time (the trainer decision rule in [design.md](design.md) section 8), token-budget exhaustion at
most 20% of episodes and at most 50% of groups with no reward contrast (both computed under `trainer_decision_10_3`), peak memory at the longest
batch (`gpu/peak_memory_gb`), the pool's mean pass rate and which alphas are safe (`pool.safe_alphas`, `pool.alpha_is_safe`),
`pool.work_vs_quit_gap` for lambda, tamper rate among passing episodes, and the confidence AUROC of the base model. If fabrication among
failures is below 20%, the base model has little to unlearn and the premise needs rethinking before more spending.

Three stop conditions (judgement thresholds): stop or re-scope if base P(claim given fail) is below about 25% **and** abstain recall
above about 70%; stop or re-scope if a no-training baseline (neutral prompt, confidence-threshold sweep) reaches P(claim given fail) below 10%
at under 5 points of verified-success loss; drop AgentAbstain even as a transfer metric if the base model's act accuracy there is below about
25 to 30%.

Also measure here: the false-blame and decoy-blame rates (`honest/false_blame_rate`, `honest/control/decoy_blame_rate`) and the base-model
honesty under our harness against a stock one (if they differ widely before any training, cross-harness claims need multi-harness training).

**Reported numbers (not gates).** The owner reads these three before any training and nothing stops on them alone: the pool pass rate
(`pool.mean_pass_rate_feasible` in `phase0_summary.json`), abstain recall (`abstain_recall`, and `abstain_strict_recall`) and the
false-refusal rate on near-miss controls (`near_miss_false_refusal_rate`), both in `runs/eval/base/metrics.json` per family. A high base
abstain recall with a low false-refusal rate means there is little to gain; the stop conditions above say what to do.

**Evaluation gate before GRPO** (judgement values, fixed before the SFT runs; compare each SFT adapter with the base on `data/test.jsonl`
under the same sampler): verified task success within 3 points of the base ([evaluation.md](evaluation.md) section 2), abstain recall
up, near-miss false-refusal rate up by at most 5 points, report rate at least 70% and no-report share at most 5%, and the `--spread`
reward spread not collapsed. If any fails, do not start GRPO: fix the SFT mixture (or stop). The decision is made on the test split only
once; use `data/val.jsonl` while tuning.

**Stop conditions, whole stage.** Stop and re-scope on: the verifier or the audit set failing in Docker (nothing trains); fewer than 95%
of variants validating as labelled; a failed weight-sync check; Phase 0 importance ratio outside 1% or a failed gate; the three stop
conditions above; an SFT adapter that fails the evaluation gate twice; or spend passing three times the estimate in the cost line at the
top without a measured step time to justify more.

Then Stage 2 is `scripts/run_pilot.sh configs/pilot.yaml` (use `SLOT=0 SHARD=0/2` and `SLOT=1 SHARD=1/2` for the two run slots), whose last
line is `PILOT GO`, `PILOT NO-GO` or `PILOT INCONCLUSIVE`; run `scripts/sync_check.sh` first, because the pilot is the first run that learns.
