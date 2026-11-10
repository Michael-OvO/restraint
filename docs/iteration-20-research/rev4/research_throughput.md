# research:throughput


## question

What does the iteration-20 plan cost in wall-clock on a 6-8 GPU cluster, where are the bottlenecks, and what training scale is actually achievable in 7 and 14 days?


## method

All numbers come from train-side logs only; no held-out path was opened; no Docker, GPU or repo write.

1. Training step timing: per-step rows of train_log.jsonl (step_time, rollout/seconds, rollout/generation_seconds, rollout/tool_seconds, rollout/slowest_tool_turn_seconds, timing/grading_seconds, timing/train_seconds, completions/*, gpu/peak_memory_gb, num_tokens) and run_config.json for six GRPO runs: iterations 07, 09, 12 (B0), 15 (G3), the it19c method arm (outputs/it19/it19-crgrpo) and the it19c ablation re-run "r2" (outputs/it19/it19-crgrpo-ablation-r2, 30 steps; 1x RTX PRO 6000 WS 96 GB, 14 vCPU, colocated vLLM at 0.3, 16,384 tokens, 36 turns, beta 0.04, images prebuilt). All past runs used 8 prompts x 8 generations = 64 episodes per step; 128 has never been run. Node specs from docs/iteration-19-signoff.md (lines 44-46 and 95) and docs/training-log.md.
2. Mechanism: read the rollout loop (src/honest_agents/trainer.py, lines 1017-1019, 1049-1060, 1212-1218), the sandbox boot and grading path (src/honest_agents/env.py, lines 195-206, 244-251, 502-503; src/honest_agents/harness.py, line 445), and TRL's order of generation, log-prob forward passes and rewards (.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py, lines 2488, 2698-2791, 2809).
3. Per-round structure: parsed every completions_*.parquet of five runs (118 steps, 7,552 episodes) into model turns and tool responses; the parsed turn counts equal the logged rollout/regenerated_turns exactly. Regressed generation seconds on the sum over rounds of the longest reply among active episodes ("straggler sum"). Projected other batch sizes by re-batching real 8-episode prompt groups (400-500 resamples).
4. Boot timing: one harbor DEBUG line per sandbox boot in the archived trainer logs of iterations 11, 12 and 15 (bursts of 64 and 120).
5. Evaluation: rollout.wall_time_sec, n_model_calls and token counts of 13 allowed eval runs (select, train-pool, twins, surface; 120-1,860 episodes each), concurrency from config.json argv, cross-checked against vLLM's own 10-second metric lines in the archived serve logs. OLS of wall time on calls and tokens with a bootstrap by source.
6. Timeouts: rc 124/137 in extra.history of G2's train-pool, select and surface runs, and in probe transcripts; cost in a synchronous batch simulated by sampling variants.
7. Memory: linear fit of gpu/peak_memory_gb on completions/max_length per run; KV capacity from the vLLM startup line (1,954,649 tokens in 69.62 GiB) and peak simultaneous context from the parsed rounds.
8. Images: Dockerfiles of train-split, non-select task dirs only.
Scripts and outputs are in the scratch directory listed under artifacts; run each with  env -u VIRTUAL_ENV uv run --frozen python <script>  from the repo root.


## headline

As written, one arm costs about 39 h: 27.8 h for 150 steps at about 11.1 min per 128-episode step (range 10.6-11.7), plus 11 h for eleven in-trainer probes. Five parallel arms therefore take about 1.6 days, and the whole programme (38.5k-episode prescreen, pilot, arms, select, held-out S=16 for seven models, external runs) about 3.5-4 days of machine time on 6-8 GPUs if nothing fails. The GPU is not the bottleneck: in the best measured run the GPU computes for 26% of a step, 37% is Docker (boot wait, command execution, grading) and 36% is generation that waits for the longest reply in each of about 32 synchronous rounds, with the average episode using 12% of its decoding slot. In evaluation 62-82% of an episode's wall time is fixed sandbox overhead and only 4-7 of 24-48 live episodes are in the model at once. Three low-risk changes (probe out of the trainer, grading started before the reference forward pass, next step's sandboxes pre-booted) would bring an arm from 39 h to about 20 h; with them 7 days on 6 GPUs fits a 5-arm screen plus 5 arms of 150-300 steps (19k-38k episodes per arm), and 14 days on 8 GPUs fits two 6-arm screens plus 6 arms of 450 steps (58k episodes per arm) with 5-7 days of reserve. Two cluster requirements in the plan need correcting: colocated 16k training needs 96 GB cards (predicted peak 77-85 GB at a full-length sequence; the quoted 74 GB was measured at 12.6-13.7k tokens), and at 128 episodes the colocated KV cache at 0.3 sits at 86% of capacity on average and above it in the worst steps.


## results


**1. Measured step time and its split at 64 episodes per step, on the hardware closest to plan 20 (r2: 96 GB card, colocated, 16,384 tokens, 36 turns, beta 0.04, prebuilt images).**

- *numbers*: n=30 steps. Step 386 s [95% bootstrap 362-411; min 283, max 557] = generation 137 s [123-153] + tool time 89 s [83-97] (of which the first-turn boot wait 61 s [60.6-62.1]) + grading 54 s [53.7-54.7] + backward 65 s [60-72] + other forward passes and overhead 37 s [34-39]. Shares: generation 36%, Docker (tools + grading) 37%, GPU compute (backward + other) 26%. 598 episodes per hour. Tokens per step 259k (prompt + completion); completion mean 1,930 tokens per episode.
- *source*: outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl; scratch scripts ci.py and model_fit.py
- *confidence*: verified

**2. Other runs and hosts are slower, and the differences come from sandbox work, not from the GPU.**

- *numbers*: Median step at 64 episodes: it09 423 s (n=50), B0 515 s (n=15), G3 634 s (n=15), it19c method arm 659 s (n=7; two arms and an evaluation on one 128-vCPU VM), it11 609 s and it13 566 s (2x RTX 4090, server mode). Tool time median 84 s (r2) to 320 s (G3); grading 0.66-1.05 s per episode on single-run hosts and 2.14 s on the shared it19c VM.
- *source*: scratch script all_runs.py and model_fit.py over the archived train_log.jsonl files of iterations 07, 09, 11, 12, 13, 15 and it19c
- *confidence*: verified

**3. The training rollout is turn-synchronous: every round waits for all tool calls, then generates for all active episodes in one batch. Generation time is set by the longest reply in each round, not by the number of tokens.**

- *numbers*: generation seconds = 6.0 + 2.44 s per 1,000 characters of the straggler sum on r2 (R2 0.94, RMSE 10.6 s, n=30); slope 2.12-2.44 and R2 0.82-0.98 in all five runs (118 steps). Per round the longest reply is about 1,600-2,100 characters against a mean of 300-500. A mean episode needs 17.0 s of its own generation (p99 52.8 s, n=1,920) inside a 137 s generation phase: 12.3% slot use; the synchronous loop takes 2.9-3.1 times the longest single episode's own generation. Rounds with 16 or fewer of 64 episodes still active carry 22-25% of generation time.
- *source*: src/honest_agents/trainer.py (lines 1017-1019, 1049-1060, 1212-1218); scratch scripts rounds.py, model_fit.py, sim128.py, tail_rounds.py
- *confidence*: verified

**4. The 61 s 'slowest tool turn' is the wait for sandbox boots, not a command timeout. Boots run 16 at a time; cold images multiply the wait.**

- *numbers*: Warm boot rate at 16 concurrent boots: 64 sandboxes in a median 54.7 s on the it11 host (n=11 bursts), about 61 s on r2 (n=30 steps, range 58-67), 120 sandboxes in 124-131 s on the B0 host (n=3) and 152 s on the G3 host (n=1): 0.8-1.2 boots per second on three hosts. With images being built during the step: 64 boots took 80-171 s (B0, median 146 s, n=15) and 109-298 s (G3, median 238 s, n=15). Only 2 command timeouts in r2's 1,920 training episodes.
- *source*: scratch script boot_bursts.py on ~/Documents/honest-agents-archive/iterations/{11-sft-grpo,12-harness-b0,15-g3-grpo}/logs; src/honest_agents/harness.py (line 445); src/honest_agents/env.py (lines 195-206, 244-251)
- *confidence*: verified

**5. A single episode that keeps hitting the 60 s command timeout stalls the whole synchronous batch. Timeouts are rare but concentrated in a few tasks.**

- *numbers*: r2 probe at step 15: one episode (task 5df28bd5, feasible) timed out 10 times; probe tool time 749 s against 237 s at step 30 (1 timeout), probe runtime 1,146 s against 601 s. G2 on the train pool (it17_rollouts_feasible): 52 timeouts in 26,967 commands (0.19%); 14 of 1,860 episodes have one (0.75% [95% by source 0.05-1.94%]); 48 of the 52 come from one source (b795bc54), up to 12 in one episode; 5 of 205 sources have any. Simulated cost at 16 prompts x 8: mean 1.38 timed-out rounds per step = about +90 s, 23% of steps affected, p95 13 rounds = about +14 min. On select and surface tasks the rate is 0.005-0.009% of commands.
- *source*: scratch scripts probe_timeouts.py, probe_rows.py and timeouts.py; runs/def74/it17_rollouts_feasible/episodes.jsonl; outputs/it19/it19-crgrpo-ablation-r2/probe/
- *confidence*: verified

**6. Backward pass, extra forward passes and grading all scale linearly with the batch, so they double at 128 episodes; generation grows by only about a quarter.**

- *numbers*: r2: backward = -24.8 + 0.338 s per 1,000 tokens (R2 0.97), other = -12.8 + 0.188 s per 1,000 tokens (R2 0.97); grading 0.85 s per episode at 64 and 0.83 s per episode in the 120-episode probes. Re-batching real prompt groups: generation 138 s at 64 episodes, 173 s at 128 with the straggler-only model [5-95% over batch composition 131-288] and 209 s with a straggler-plus-volume model.
- *source*: scratch scripts other_fit.py, sim128.py, project.py
- *confidence*: derived

**7. Projected plan-20 step (16 prompts x 8 = 128 episodes, synchronous loop, colocated, prebuilt images, r2-class host).**

- *numbers*: Step 636-699 s, mid 668 s = 11.1 min: generation 173-209 + boot wait 123 + command rounds about 41 + grading 108 + backward 131 + other 73. 690 episodes per hour. 150 steps = 27.8 h (26.5-29.1). Timeout-prone tasks in the pool add 0 to about 90 s per step on average (up to 3.75 h per arm). A host that builds images during steps or shares its Docker daemon would be 1.3-1.8 times slower. Pilot of 10 steps = 1.85 h plus the step-0 probe.
- *source*: scratch script project.py and its output project_out.txt
- *confidence*: derived

**8. The in-trainer probe is the largest avoidable cost of an arm.**

- *numbers*: 11 warm 120-episode probes on four runs: median 1,052 s (min 601, max 1,551) = 8.8 s per episode, against about 5.2 s per training episode. Plan 20's probe is 103 rows x 4 = 412 episodes, 11 times per arm: about 60 min each (34-89), 11.0 h per arm = 28% of a 38.9 h arm. In G3's real run probes took 5,807 s of 15,481 s (37.5%); in r2 3,314 s of 14,894 s (22%). One batch of 412 is not an option colocated: about 1.4M tokens of simultaneous context against a 482k-token KV cache.
- *source*: scratch scripts probe_rows.py and project.py; ~/Documents/honest-agents-archive/iterations/15-g3-grpo/logs/train_g3.log (line 4663: 15 steps in 4:18:01)
- *confidence*: derived

**9. Evaluation throughput per Docker VM, and why: the sandbox lifecycle, not the model.**

- *numbers*: At concurrency 24: 700 episodes per hour (it16b host, n=1,600), 945-947 (two RTX 5090 hosts, n=424 and 1,860), 932 (128-core 5090 host, n=424), 1,341 (r2 host, n=848; the server log shows 848 episodes in 39.2 min); at concurrency 48: 1,784 (one run, n=424). Mean wall per episode 64-124 s [bootstrap by source within +/-7%]. Fixed overhead per episode 36-108 s = 62-82% of wall; per-token cost implies 82-129 tokens per second per stream. vLLM during these runs: 4.1-7.4 requests running on average (p90 9-13), 0 waiting, KV cache 1.1-4.6% used, 283-725 generated tokens per second. Server start 55 s; adapter merge under 1 min.
- *source*: scratch scripts eval_runs.py, eval_decomp.py, vllm_metrics.py; src/honest_agents/eval/rollout.py (lines 497, 559, 602-618); ~/Documents/honest-agents-archive/staging/outputs/logs/serve_ablation_select.log and serve_rollouts_feasible.log
- *confidence*: verified

**10. Hours per plan item.**

- *numbers*: Prescreen 38,500 episodes: 28.7-55.0 VM-hours at concurrency 24 (6 VMs 4.8-9.2 h, typical 6.8 h; 8 VMs 3.6-6.9 h, typical 5.1 h); 21.6 VM-hours at concurrency 48 (3.6 h / 2.7 h); plus image builds of about 1.8 h per VM on 6 VMs or 1.4 h on 8. One arm: 38.9 h as written; five arms in parallel 38.9 h. Select S=4 at three checkpoints for five arms plus a fresh G2: 13,568 episodes = 10.1-19.4 VM-hours plus about 4 min of serving overhead per run. G5: 480 episodes = 0.4-0.7 VM-hours (1,440 for all arms = 1.1-2.1). Probes if moved out of the trainer: 22,660 episodes = 12.7-32.4 VM-hours. Held-out S=16 for seven models: 62,944 episodes = 46.9-89.9 VM-hours at concurrency 24 (6 VMs 7.8-15.0 h, typical 11.1 h; 8 VMs 5.9-11.2 h, typical 8.3 h), 35.3 VM-hours at concurrency 48 (5.9 h / 4.4 h). External runs: more than 12 h per model per the plan, three models in parallel. Whole plan as written: about 89-94 h (3.8-4 days) on 6 GPUs and about 80 h (3.3 days) on 8, with no failure, no re-pilot and no waiting on approvals.
- *source*: scratch script project.py (project_out.txt); external figure from docs/iteration-20-rl-design.md (lines 490-493)
- *confidence*: derived

**11. GPU memory is set by the single longest sequence, not by the number of episodes; the plan's '74 GB at 16k' was measured at shorter sequences.**

- *numbers*: Peak GB = 45.7 + 2.37 per 1,000 tokens of the longest completion (r2, R2 0.84, n=30); same form in it07, it09, B0 and G3 (R2 0.58-0.82). The 74.4 GB peaks occurred at 12,630-13,689 tokens. Predicted at a full 16,384-token completion: 77.5-84.5 GB across five colocated runs. Longest completion in 135 logged steps: 14,589 tokens; 11 of 135 steps (8%) exceeded 12,000, so about 15% of 128-episode steps would. Server mode (training process only): 37-43 GB at 10-11k tokens, predicted 50-53 GB at 16,384. Host RAM is not binding: trainer RSS 4.9-5.5 GB, 93 GB free at 64 sandboxes.
- *source*: scratch script mem_fit.py; docs/iteration-20-rl-design.md (line 569)
- *confidence*: derived

**12. At 128 episodes the colocated vLLM KV cache at utilisation 0.3 is at its limit.**

- *numbers*: KV size 28,076 tokens per GiB (1,954,649 tokens in 69.62 GiB). Capacity at 0.3: about 482k tokens on a 96 GB card, about 350k on an 80 GB card. Peak simultaneous context: 216k tokens at 64 episodes (n=30 steps, max 273k); 416k at 128 (resampled: p90 453k, max 521k); 620k at 192; 818k at 256.
- *source*: scratch script kv_demand.py; ~/Documents/honest-agents-archive/staging/outputs/logs/serve_ablation_select.log (lines 53-55)
- *confidence*: derived

**13. The token budget and the turn cap are not throughput levers.**

- *numbers*: At the 12,288 cap only 16 of 1,860 G2 train-pool episodes (0.86%) and 1 of 848 select episodes (0.12%) stopped on the token budget, so 16,384 changes under 1% of episodes. Cutting turns from 36 to 28 removes 9-10% of generation time and cuts 11-14% of episodes; to 24 turns removes 20% and cuts 21-23%. Capping a turn at about 1,024 tokens saves 10 s of 173 s.
- *source*: scratch scripts budget_share.py, tail_rounds.py, sim128.py
- *confidence*: verified

**14. vCPU count does not predict sandbox speed across the hosts used so far; the plan's '>= 64 vCPU' is not supported by the data.**

- *numbers*: Fixed overhead per evaluation episode at concurrency 24: 36-44 s on the 14-vCPU host, 62-71 s on the 64-core host, 76 s and 89 s on two 128-vCPU hosts, 108 s on the 128-vCPU host while another arm trained. The 14-vCPU host also has the fastest training step (363 s median).
- *source*: scratch script eval_decomp.py; node specs in docs/iteration-19-signoff.md (lines 44-46, 95), docs/iteration-17-signoff.md (line 15), docs/iteration-18-signoff.md (line 12), docs/iteration-16-signoff.md (line 4)
- *confidence*: verified

**15. Two arms and an evaluation on one VM ran far slower than one arm alone; how much is the shared Docker daemon and how much the host itself is not separated.**

- *numbers*: it19c method arm on the shared 128-vCPU, 2-GPU VM: median step 659 s, tool time 210 s, grading 137 s (n=7). Same configuration without the report term alone on the 14-vCPU VM: 363 s, 84 s, 54 s (n=30). Two other 128-vCPU hosts were also slow when running a single job (76-89 s fixed overhead per episode).
- *source*: scratch script model_fit.py; docs/iteration-19-signoff.md (lines 65-70, 82-96)
- *confidence*: not verified

**16. Episodes per hour by batch geometry: larger batches are only modestly more efficient, and prompts x generations at a fixed product costs the same.**

- *numbers*: 8x8: 383 s per step, 602 episodes per hour (measured 598). 16x8: 668 s, 690 per hour. 32x4: 674 s, 683 per hour. 24x8: 941 s, 734 per hour. 32x8: 1,210 s, 761 per hour. An 8x8 step costs 6.4 min against 11.1 min for 16x8, so the same wall time buys 1.74 times as many optimizer steps at 13% fewer episodes. 192 and 256 episodes exceed the KV cache at 0.3.
- *source*: scratch script project.py
- *confidence*: derived

**17. The prescreen needs thousands of image builds, each with its own apt and pip install.**

- *numbers*: 4,280 train-split, non-select variants in the four trained families (the plan counts 4,098 after further exclusions): 4,280 distinct images, 4,244 with an apt or pip layer, 958 distinct first heavy instructions (1,498 by cache key); every source's variants share one heavy layer. Measured build rate: 510 images in 70 min on one host. All prescreen images on one host: about 11 h; sharded over 6 VMs about 1.8 h each, over 8 about 1.4 h. A training VM needs about 1,175 images (pool 800, probe 103, select 212, twins 60): about 2.7 h. Image sizes on disk were not measured.
- *source*: scratch script images.py; docs/iteration-19-signoff.md (lines 76-78)
- *confidence*: derived

**18. Resume works, but what is pulled off a node today cannot continue a run exactly.**

- *numbers*: --resume restarts from the last checkpoint, checks the data order, rebuilds stop-rule memory from the log history and skips the step-0 probe. With save_steps 5 a crash loses at most 5 steps, about 55 min at 11.1 min per step. Local copies of checkpoints are 267 MB (adapter, tokenizer, trainer state) with no optimizer or scheduler state; the watcher pulls only when the chain has ended. it19c lost a host right after step 15 with the checkpoints on it and the arm was re-run from step 0.
- *source*: src/honest_agents/train/run.py (lines 306-311, 338); src/honest_agents/callbacks.py (lines 534-540); src/honest_agents/train/rows.py (lines 286-304); src/honest_agents/archive.py (line 11); src/honest_agents/node_watch.py (line 10); docs/iteration-19-signoff.md (lines 91-98)
- *confidence*: verified

## design_recommendations


**1. Take the validation probe out of the trainer. Serve each probed checkpoint on an evaluation VM (merge, 12-task sanity, then run_eval on the 103-row probe at S=4) and let a small sidecar write the stop-rule-3 verdict to a file the trainer checks each step. On a 6-GPU layout with one evaluation VM, probe every 30 steps instead of 15.**

- *rationale*: Eleven warm in-trainer probes cost a median 8.8 s per episode because the probe is one more synchronous batch, and a single timing-out episode adds about 10 min. The existing evaluation path does the same 412 episodes in 14-26 min without stopping training, and stop rule 3 already needs two consecutive probes, so a lag of two or three steps does not change it.
- *expected_measurable_effect*: Arm wall-clock 38.9 h to 27.8 h (-28%); training episodes per hour of arm time 494 to 690. Measure as the arm's total train_log time span divided by 150 steps.
- *confidence*: high

**2. Start grading as soon as the rollout loop ends: call env._start_grading() for every environment in HonestGRPOTrainer._generate right after _tell_environments. score_batch then only collects the results.**

- *rationale*: TRL runs the old and reference log-prob forward passes (about 73 s at 128 episodes) between generation and rewards, and grading (about 108 s) only starts afterwards. _start_grading is idempotent and the exit reason is already set at that point, so the two can overlap.
- *expected_measurable_effect*: About 60-70 s less per 128-episode step (668 s to about 600 s, -10%). Measure timing/grading_seconds plus the unexplained remainder of step_time before and after on a 5-step run.
- *confidence*: medium

**3. Pre-boot the next step's sandboxes during the backward pass (the data order is fixed, so the next 16 rows are known), and in the timing pilot test env.max_concurrent_boots at 16, 32 and 64.**

- *rationale*: After grading, every sandbox is gone and Docker is idle for about 200 s of GPU work, while the next step then waits about 123 s for 128 boots in waves of 16. Boot waves take 12-17 s on three different hosts, which suggests fixed waits rather than CPU, so a larger wave may also help at no code cost.
- *expected_measurable_effect*: Pre-boot: about 120 s less per step (668 s to 545 s; with early grading about 480 s, 150 steps in about 20 h, 960 episodes per hour). Larger boot waves: up to 60-80 s less per step if waves do not stretch. Measure rollout/slowest_tool_turn_seconds.
- *confidence*: medium

**4. Keep timeout-prone tasks out of the pool and the probe: drop any source with a command timeout (rc 124 or 137) in its S=8 prescreen episodes, and add a per-episode wall-clock cap so one hung episode cannot hold a round for more than a few minutes.**

- *rationale*: One source produced 48 of G2's 52 timeouts on the train pool, with up to 12 in one episode; in a synchronous batch each costs every other episode 60-80 s. The rule depends only on wall time, applies to all arms alike and touches about 2% of sources.
- *expected_measurable_effect*: Removes a mean of about 90 s per step (up to 3.75 h per arm) and the 10-20 min outlier steps. Measure the gap between mean and median rollout/tool_seconds and the count of steps over 15 min.
- *confidence*: medium

**5. Before fixing timeouts or the schedule, qualify every VM type with a 30-minute test: 64 and 128 no-op sandbox boots, an evaluation of 200 episodes at concurrency 24, 48 and 96, and two training steps at 128 episodes that log step split, peak GPU memory and vLLM preemption warnings. Replace the '>= 64 vCPU' requirement with a measured one: at least 1 warm boot per second and at most 60 s of fixed overhead per evaluation episode.**

- *rationale*: Sandbox speed varied 2.5-fold across the hosts used so far and did not follow vCPU count; 128 episodes per step has never been run; the KV cache is predicted to be at 86% of capacity.
- *expected_measurable_effect*: Turns the 26.5-29.1 h per-arm estimate (and its 1.3-1.8 times slow-host tail) into a measured number before any arm starts, and sizes chain timeouts from it.
- *confidence*: high

**6. Run every evaluation and the prescreen at concurrency 48 once the qualification test confirms it, and give each run its own VM: one Docker daemon per training run, evaluations on separate VMs. If the cluster offers CPU-only VMs, point their sandboxes at one vLLM endpoint with --base-url.**

- *rationale*: At concurrency 24-48 only 4-7 requests are in the model at a time and the KV cache is under 5% used; the one run at 48 reached 1,784 episodes per hour against 932-947 on comparable hosts at 24. The one case of two arms plus an evaluation on a shared VM ran 1.8 times slower per step.
- *expected_measurable_effect*: Prescreen on 6 VMs 6.8 h to 3.6 h; held-out S=16 for seven models 11.1 h to 5.9 h on 6 VMs. Measure episodes per hour from the run directory.
- *confidence*: medium

**7. Correct the GPU requirement: colocated training at 16,384 tokens only on 96 GB cards. On 80 GB cards either cap max_completion_length at 12,288 or run vLLM as a separate server. At 128 episodes per step keep vLLM at 0.3 only if the timing pilot shows no sustained preemption; otherwise use 12,288 with utilisation 0.4.**

- *rationale*: Peak memory follows the longest sequence at about 2.4 GB per 1,000 tokens; a full-length sequence is predicted at 77.5-84.5 GB with vLLM holding 28.5 GB. The 12,288 cap touches under 1% of episodes and frees about 9 GB, enough to raise the KV cache from 482k to about 749k tokens.
- *expected_measurable_effect*: No out-of-memory stop in 750 arm-steps; gpu/peak_memory_gb stays under 85 GB on 96 GB cards; vLLM preemption count near zero.
- *confidence*: medium

**8. Make long runs restartable in practice: run each arm as five 30-step segments with --resume; after every save push the full checkpoint directory (adapter, optimizer, scheduler, RNG and trainer state) plus the run's jsonl files to shared storage; keep one spare VM with the merged G2 and all images ready; build images once and serve them from a cluster registry instead of rebuilding per node.**

- *rationale*: A 39 h arm is ten times longer than any past run. The one host loss so far cost a full re-run because checkpoints lived only on the node, and the current pull tooling keeps no optimizer state and pulls only at the end of a chain. A registry also guarantees every arm and node runs identical images, which per-node apt installs do not.
- *expected_measurable_effect*: A host loss costs at most 5 steps (about 55 min) plus about 15 min of restart, instead of the whole arm (up to 39 h) plus 2.7 h of image builds. Test by killing and resuming one arm during the pilot and checking that the step count and the stop-rule windows continue.
- *confidence*: high

**9. Replace the single 150-step shot with a staged programme. Schedule A (7 days, 6 GPUs: 5 training, 1 evaluation): prescreen and pool by hour 30; screen 5 arms x 60 steps (11.1 h); confirm with the winner x 3 seeds plus B and C at 150 steps (27.8 h) or 300 steps (55.6 h); final select, G5 and held-out S=16 (about 8 h); external (12-20 h). Total about 4.1 days at 150 steps or 5.3 days at 300. Schedule B (14 days, 8 GPUs: 6 training, 2 evaluation): two screening waves of 6 arms x 75 steps (10-14 h each), then 6 arms x 450 steps (60 h with the step-time fixes, 83 h without), then evaluations. Total 6.2-7.5 days. Schedule C (14 days, 6 GPUs, only the probe moved out): two 5-arm screens of 60 steps and 5 arms x 300 steps, about 7.5 days.**

- *rationale*: A screening arm of 60 steps costs 11.1 h and 7,680 episodes, so ten to twelve configurations can be tried before committing seeds, and the confirmatory stage can be two to three times longer than planned inside the same calendar. The reserve in each schedule covers one full re-run of the longest arm.
- *expected_measurable_effect*: Episodes per confirmatory arm: 19,200 (150 steps), 38,400 (300), 57,600 (450), against 19,200 in the plan, plus 5-12 screened configurations. Reserve: 1.7-2.9 days in A, 6.5-7.8 days in B, about 6.5 days in C.
- *confidence*: medium

**10. If the screen needs more optimizer steps rather than more precise steps, run screening arms at 8 prompts x 8 generations.**

- *rationale*: A 64-episode step costs 6.4 min against 11.1 min at 128, because only generation is sub-linear; choosing 16x8 or 32x4 at 128 episodes makes no difference to cost (668 s against 674 s).
- *expected_measurable_effect*: 1.74 times as many optimizer steps per hour for 13% fewer episodes per hour (602 against 690).
- *confidence*: medium

**11. Spend the spare evaluation capacity on the primary contrast: run A1 and the fresh G2 at S=32 on the held-out set instead of S=16.**

- *rationale*: One model at S=16 is 8,992 episodes, about 7-10 VM-hours, small next to a 28 h arm. The plan's own power table shows the no-interaction standard error scaling as one over the square root of S.
- *expected_measurable_effect*: 17,984 extra episodes, about 2.2-3.2 h on 6 VMs. MDE90 without interaction 2.53 to about 1.79 U per 100; with the 0.744 interaction term 3.56 to about 3.10. Derived from the table in docs/iteration-20-rl-design.md (lines 193-196), not re-estimated.
- *confidence*: low

## caveats

- No run has ever used 128 episodes per step. The 11.1 min step is a projection from 64-episode steps; it assumes per-stream decoding speed is unchanged at 128 concurrent streams (speed was the same at about 6 and at up to 64 streams, 82-129 tokens per second) and that KV preemption at utilisation 0.3 costs little. Both need the two-step timing pilot.
- The reference host (r2) is the fastest sandbox host in the record. Hosts that were 1.4-2.5 times slower per sandbox operation exist in the logs, and the cluster's VMs are unknown. All calendar figures assume r2-class hosts, prebuilt images and one Docker daemon per run.
- The slow it19c method arm cannot be split into shared-daemon contention and a slow host; the recommendation of one VM per run rests on that single comparison plus the general pattern.
- Early grading, pre-booting and larger boot waves were read from the code and the timing split, not run. The savings are upper bounds until measured.
- Evaluation at concurrency 48 rests on one run of 424 episodes; concurrency 96 and CPU-only sandbox VMs are untested.
- Image sizes on disk were not measured, so the 300 GB disk requirement is unchecked for a node holding all prescreen images; sharding the prescreen keeps each node to about one sixth to one eighth of them.
- External benchmark durations (more than 12 h per model) are taken from the plan and the training log; the underlying run directories were not opened under the hygiene rule.
- Calendar totals cover machine time only. Writing the code in section 10 of the plan, spec-audit reviews, transcript reads and owner approvals are not included, and a 450-step run on 16 prompts needs 7,200 prompt visits (an 800-row pool repeated 9 times, or a larger pool), which depends on what the prescreen finds.
- The MDE figures for S=32 are scaled from the plan's own table and were not recomputed from episodes.
- Timeout rates are from G2 at the 12,288 budget; a trained policy may run different commands.

## artifacts

- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/project.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/project_out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/all_runs.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/ci.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/rounds.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/model_fit.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/sim128.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/tail_rounds.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/other_fit.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/boot_bursts.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/timeouts.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/probe_timeouts.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/probe_rows.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/eval_runs.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/eval_decomp.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/vllm_metrics.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/mem_fit.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/kv_demand.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/budget_share.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/walltime/images.py