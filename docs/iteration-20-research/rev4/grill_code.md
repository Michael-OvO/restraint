# grill:code


## lens

code reality: does revision 3 match what the code at HEAD (5a91a9d) and the installed libraries (TRL 1.14.1, transformers 5.18.0, peft 0.21.2) actually do


## verdict

Revision 3 is mostly accurate about the code, but it has no blocker-level error and six major gaps. I checked every code-facing claim against HEAD and the installed TRL. The config chain, the KL reference, the sampler order, the DAPO normaliser, the route by which a per-row weight reaches the reward, and nearly every line reference are correct.

Three claims are wrong:
- "peak 74 GB at 16k": the 74.4 GB peak was at 12,650 tokens; no episode reached 16,384.
- "best_checkpoint.json will still name 150": after an early stop the file is never written.
- Rule 1 "reacts within 5 steps; it19c collapsed in 7": replayed on the it19c method arm, it fires at step 7, the collapse step itself.

One resolution is incomplete: the read-split pilot rule (the answer to Method B1) cannot be computed from the columns §4.3 defines.

The larger problems are omissions:
1. beta 0.04 does not anchor the policy in this trainer, and the pilot's KL gate is an extrapolation that the plan's own reference run puts on the threshold.
2. The per-episode table written every step lacks the columns the pilot rules, the stop rules and the revisit diagnostic need, and no code item computes them.
3. The 128-episode geometry with truncation unmasked has never run.
4. "Fresh G2 on the same commit" is unenforced and false under the handoff's schedule.
5. About a third of §10 can only be executed by the evaluator, after the executor has written it blind.
6. §10 is about 2,000-2,300 source lines plus tests (my estimate), with three high-risk items and several missing ones.

My probability that revision 3 as written yields a significant positive H1 with every guard passing is about 10 % (range 5-15 %). Reasons:
- The detectable effect (2.5-3.6 U) sits at the top of the plan's own credible range.
- The only comparable run showed no train-side movement in U even on its own training prompts: pass 2 minus pass 1 on the same 120 prompts was -0.1 points, SE 2.6 (my replay of the it19c ablation; its reward was the CR-GRPO table, not U).
- The KL term gives no real protection against the capability drift that run showed.
- I put roughly a 15-20 % chance on an execution-level failure (out-of-memory, harness mismatch, a blind-code bug at the single look) making the result unusable.

The changes below do not make the effect larger by themselves. They make a real effect measurable earlier and at smaller standard error, and they remove the ways the run can be voided.

Scratch scripts and outputs are under ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/code-reality (trainlog2.py, adv.py, revisit.py, sim_stop.py, bank2.py, counts.py). Nothing in the repo or the archive was changed.


## findings


**1. M1 [major]**

- *title*: beta 0.04 does not hold the policy near G2, and the pilot's KL gate sits on its own threshold
- *evidence*: What the plan gets right: the reference is merged G2 with the adapter disabled (.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py, lines 2777-2789), and beta times a per-token KL estimate is added on the same token mean (lines 3159-3166 and 3210-3211; use_bias_correction_kl defaults to true, grpo_config.py line 967).

What the only comparable run shows (it19c ablation, beta 0.04, lr 3e-5; outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl): KL rose roughly linearly to 0.021 at step 30 with no plateau. Slope 0.00085 +- 0.00008 per step over steps 6-10, and 0.00058 +- 0.00014 over steps 11-30. beta x KL at step 30 is 0.0008.

Analytic size, not measured: the KL gradient per token is about beta x |log-ratio| = 0.04 x 0.2 = 0.008 at KL 0.02. The mean |advantage| under plain U on the same 1,920 rollouts is 0.35 (my computation from the archived completions parquet). That is about 2 %.

§6's gate applied to that run (fit steps 6-10, project to step 150) gives 0.123. Scaled to lr 2e-5 it is 0.055 if KL scales with lr squared and 0.082 if linearly. The threshold is 0.08. The same run's later slope is 32 % below the 6-10 fit.
- *why_it_matters*: §14 lists KL as the control for capability loss and KL runaway. As implemented it is a logged number: the only real controls are the learning rate and the hard stop at 0.1. Whether the gate trips is close to a coin flip on the plan's own reference data. A trip sends every arm back to a fresh start at 1.5e-5, because a changed learning rate is ignored on resume (transformers restores the optimizer and scheduler state: .venv/lib/python3.12/site-packages/transformers/trainer.py, lines 3795 and 3822-3845).
- *proposed_change*: 1. Measure the gradient-norm ratio |grad KL| / |grad policy| on the first pilot steps by generalising _measure_grad_ratio (src/honest_agents/trainer.py, lines 832-883), which today only runs in crgrpo mode.
2. Make the pilot 20 steps and fit the KL slope on steps 6-20. Pre-register the decision on that fit.
3. If an anchor is wanted, choose beta by a short ladder (0.04, 0.3, 1.0; 20 steps each) and take the smallest beta whose slope over steps 11-20 is at most 0.0003 per step.
4. State in §14 that at beta 0.04 the KL term is a monitor.
- *measurable_effect*: KL slope per step and the gradient-norm ratio, both logged. Target: projected KL(150) at most 0.05 from a 15-point fit instead of a 5-point fit. On it19c data the 5-point fit over-projects the realised later slope by about 46 % (0.00085 against 0.00058).

**2. M2 [major]**

- *title*: The per-episode training table lacks the columns the pilot rules, stop rules and revisit diagnostic need, and no code item computes them
- *evidence*: A per-episode table is already written every step: completions_<step>.parquet with step, prompt, completion, reward, case, feasible, group, honest_total, loop_exit, masked_reason, n_actions, passed, status, stop_reason, tests_passed, advantage (read from the archived it19c ablation; columns come from EXTRA_COLUMNS in src/honest_agents/trl_rewards.py, lines 473-488).

Missing columns: is_control, family, damage (violation, tamper), readme_read, unweighted U, u_weight, loss-token count.

§6's read-split rule needs G2's decline rate among its unread episodes per prompt. §4.3 stores only the marginals g2_decline and g2_read, which cannot give it.

§10 item 10 has only `push` and `weights`. Nothing computes the §6 pooled bounds, the KL fit or the revisit slope that §4.4 and §7 rely on.

Demonstration on the it19c ablation (steps k and k+15 saw the same 120 prompts, 8 samples each; script revisit.py in my scratch directory): pass 2 minus pass 1 U = -0.1 points, SE 2.6, 90 % CI [-4.5, +4.2]. Verified success +1.2 [-1.2, +3.5]. Scaled to 800 prompts, the SE is about 1.0.
- *why_it_matters*: The read-split rule is the plan's own answer to Method B1 (G3's blanket declining) and cannot be computed as specified. The revisit comparison is the only train-side learning curve with a usable standard error, it costs no extra rollouts, and it is readable at step 100, before any held-out evaluation is paid for.
- *proposed_change*: 1. Add the missing columns to EXTRA_COLUMNS.
2. Have rl_pool_v5 write g2_decline_unread and g2_n_unread per row.
3. Add `analysis.u_dynamics pilot`, `rules` and `revisit` subcommands that read the parquet files and the pool file and cluster by source.
4. Pre-register epoch 2 minus epoch 1 and epoch 3 minus epoch 1 train U (w = 1) as a secondary endpoint and as an early readout.
- *measurable_effect*: A train-side delta U with SE about 1.0 point (MDE90 about 2.5), from files already written every step. For comparison the held-out SE is 0.86-1.21 (§3.4). The it19c reference value is -0.1 +- 2.6.

**3. M3 [major]**

- *title*: Stop rule 1 fires at the collapse, not before; a status-logit monitor that would lead it exists in the trainer but is off
- *evidence*: Replay of rule 1 (5-step pooled P(claim | pass), one-sided 95 % upper bound below 0.963) on the it19c method arm's parquet (outputs/it19/it19-crgrpo/completions; script adv.py): it fires at step 7 (p 0.832, n 137), not at step 6 (0.977). On the ablation's 30 steps it never fires (minimum 0.969).

docs/iteration-19-research/badcase-it19c/method_stop.md (section 1 table): mean |z| at the status token fell from 15.0 at step 1 to 9.15 at step 5 while P(claim | pass) was still 1.00. At step 7, 20 of 31 passing episodes declined. The step-5 checkpoint already scored verified success -8.3 [-12.1, -4.4] on select (docs/iteration-19-results-crgrpo.md, Method arm).

The exact z is computed by StatusLogits and status_position (src/honest_agents/trainer.py, lines 91-132 and 154-215) but only when the CR-GRPO report term is on (lines 796-800). method_stop.md notes that z was not saved per row.
- *why_it_matters*: §5.5 makes a stopped A1's candidate the last checkpoint before the rule fired. In the one observed collapse that checkpoint was already damaged. The binary claim rate cannot show erosion of the claim margin, because G2 claims at |z| near 15. z is also a continuous per-step measure of the report decision, which the plan otherwise sees only through rare binary flips.
- *proposed_change*: 1. Run the status hook in monitor-only mode: detached, no mask change, no loss term.
2. Log mean z on passing claims, failing claims and twin claims per step, and z per episode in the parquet.
3. Add a stop on mean z over passing claims below 60 % of its steps 1-2 value for 2 consecutive steps.
4. Define the candidate as the last checkpoint before the first step of the firing window.
- *measurable_effect*: On the it19c method arm the z rule trips at step 6 (9.15 / 15.0 = 61 % at step 5, 8.20 at step 6), one step before rule 1. It also gives a per-step curve of the claim margin on fails against passes, from about 75 passing and 15 failing claims per step. Not verified on the ablation, where z was not logged.

**4. M4 [major]**

- *title*: The 128-episode geometry with truncation unmasked has never run, and the memory claim is wrong
- *evidence*: No config sets gradient_accumulation_steps above 64 (grep of configs). TRL sizes vLLM for batch x steps_per_generation = 128 concurrent sequences (grpo_trainer.py line 1087).

Memory, from the it19c ablation log (script trainlog2.py): peak allocated = 45.7 GB + 2.37 MB per token of the step's longest completion (r = 0.92 over 30 steps). The 74.4 GB peak at step 21 was at 12,650 tokens on a 96 GB card with vLLM at 0.3. No episode reached 16,384. Extrapolated to 16,384 the peak is about 84.5 GB on 96 GB. On an 80 GB card vLLM's share is 4.8 GB smaller, giving about 79.7 GB: no headroom. §11 allows 80 GB cards and says "peak 74 GB at 16k".

With mask_truncated_completions false (grpo_trainer.py lines 2539-2547), cut-off episodes enter the backward pass for the first time.

Time: a 64-episode step took 385 s on the re-run node (generation 137, tools 89, grading 54, train 65). The incident table in docs/training-log.md gives 14 min on the other node. So 150 steps of 128 episodes is roughly 32-70 h per arm. Eleven probes of 412 episodes add about 6-14 h (120-episode probes took 10-26 min).
- *why_it_matters*: An out-of-memory error at an unpredictable step, or an arm that takes three days, would be discovered during a five-arm parallel launch. The plan has no wall-clock estimate, and the job-file timeouts it asks to be "sized" depend on one.
- *proposed_change*: 1. Add a zero-lr dress rehearsal to §12 before the pilot: 3 steps at lr 0 with the it20 config, the trajectory bank on, and one forced full-length episode. configs/phase0_zero_lr.yaml and `callbacks phase0` already exist for this.
2. Gates: peak at most 90 % of the card, step time recorded, KV preemptions counted, step-0 probe with zero masked episodes.
3. Require 96 GB cards, or set vLLM utilisation from the measured trainer peak.
4. Put the wall-clock estimate in §11.
- *measurable_effect*: Peak GB at a true 16,384-token episode, seconds per step at 128 episodes, preemption count and masked share, all before any learning step. It also yields 384 pure-G2 episodes under the training harness. Cost: about 40-90 minutes on one GPU.

**5. M5 [major]**

- *title*: "Fresh G2 on the same commit" is unenforced, and the handoff's schedule makes it false
- *evidence*: docs/iteration-20-handoff.md (section B2, step 2) runs G2 and base on the held-out set before any candidate exists. The candidates are evaluated after training, on the executor's later commit.

An eval run's config.json has no commit or code fingerprint. Its keys in runs/eval/it19c_g2_select/config.json are argv, base_url, budget, environment_type, forced_attempt, harness, mode, n_infeasible_skipped, n_samples, n_tasks, protocol, removed_tools, reward, rollout, split, system_prompt_source, train_config. There is no git or commit reference in src/honest_agents/eval/run_eval.py.

check_comparable compares only the split, rollout keys and reward config (src/honest_agents/eval/select_checkpoint.py, lines 295-325).

Harness drift between arms has already changed this project's numbers (DEF-65, DEF-71, DEF-77, cited in docs/results.md and the training log).
- *why_it_matters*: H1 is A1 against G2. If any evaluation-affecting file changes between the early G2 run and the candidate runs, the contrast partly measures the harness, and nobody can show otherwise.
- *proposed_change*: 1. Write a harness fingerprint into every run's config.json: sha256 over harness.py, env.py, grading.py, reward.py, eval/rollout.py, the system prompt and the tool schemas. Make `power estimate` refuse two runs whose fingerprints differ.
2. Have the evaluator run G2 a second time in the same batch as the arms. The late G2 is H1's reference. The early G2 stays as the power and clone-check input.
- *measurable_effect*: Early G2 against late G2 is a real A/A test on the full pipeline. Its delta U should cover 0, and its spread measures the run-to-run term that §3.4 calls weakly determined (SE 0.86 against 1.21). Cost: 8,992 episodes, about 7 GPU-hours at the 1,305 episodes per hour of the it19c select run (848 episodes in 39 minutes by file times).

**6. M6 [major]**

- *title*: About a third of §10 can only be executed by the evaluator, and §10 is larger than the plan admits
- *evidence*: The executor has no held-out task dirs and no Harbor job dirs, and may not run `final_benchmark freeze` (handoff sections A2 and A3). So items 3 (external units), 4 (refreeze, N override map), 8 (`--record`), the joint clone check on held-out and the H2 pooled estimate first execute on the evaluator's machine.

Missing code paths:
- `power estimate` takes exactly two episode files (src/honest_agents/analysis/power.py, lines 1240-1243), so H2's pooled 48-against-16 estimate with a four-run void union has none.
- clone_pass_joint (line 912) is a library function with no CLI for the G1-G6 set.

My size estimate for §10, from the modules it extends (rl_pool_v3.py 347 lines, callbacks.py 923, archived u_checks.py 342), in source / test lines:
- item 1: about 80 / 200
- item 2: 250 / 300
- item 3: 150 / 150
- item 4: 120 / 150
- item 5: 350 / 400
- item 6: 500 / 400
- item 7: 150 / 100
- item 8: 80 / 100
- item 9: about 250 lines of YAML
- item 10: 350 / 200
Total about 2,000-2,300 source lines and about as many test lines. Highest risk: 5 (stop rules), 6 (pool builder), 4 (refreeze).

Not in §10 at all: the harness fingerprint, H2 pooling, a clone-check CLI, the pilot and revisit analysis, the KL stop rule (no KL rule exists in src/honest_agents/callbacks.py), a candidate pointer on an early stop, and validation of u_weight.
- *why_it_matters*: The round has one look. A bug in blind-written analysis code is found either after the look, when a fix looks like tuning, or not at all.
- *proposed_change*: 1. Evaluator-side rehearsal before training starts: run the complete it20 analysis (U by stratum, guards, clone check, the H2 path with G2's samples split three ways) on the archived base and G2 held-out runs. Freeze that commit in the prereg.
2. Add the missing items to §10 and split it into what A1 needs and what can wait (arm C, external units, N rebuild).
3. Add acceptance tests on archived train-side data: rule 1 fires at step 7 on the it19c method arm and never on the ablation; `power estimate` reproduces +3.8 [+0.8, +6.8].
- *measurable_effect*: The rehearsal must reproduce the published check-1 numbers (G2 U 33.3 on the S = 2 set in docs/iteration-20-research/u_checks.txt). Any mismatch is a bug found while fixing it is free.

**7. m1 [minor]**

- *title*: The per-row twin weight has two silent-failure modes
- *evidence*: Probe rows have no u_weight column, so _label returns None (src/honest_agents/trl_rewards.py, lines 409-413 and 573-582). `None * float` raises, and _never_raise then masks the whole batch (lines 832-879). The probe would report nothing, with no error.

If the implementer defaults None to 1.0 to fix that, a training twin row with a missing weight silently trains at w = 1, the non-neutral reward of Method B1.

§12 item 5 checks the weights' values offline, not that the trainer applies them.
- *why_it_matters*: The weight is the plan's core fix for blanket declining, and nothing verifies it at run time.
- *proposed_change*: 1. In check_rows, for modes u and u_success: every infeasible training row carries 0.2 < u_weight <= 1 and every feasible row carries 1.
2. Default None to 1.0 only for validation rows.
3. Add a startup assertion that the step-0 probe has zero masked episodes.
4. Add tests for both cases.
- *measurable_effect*: Two unit tests, plus one startup check. `honest/reward/u` (unweighted, already logged for every mode by batch_metrics line 288-290) appears beside TRL's weighted `reward` on the dashboard.

**8. m2 [minor]**

- *title*: After an early stop best_checkpoint.json is not written, and `final/` holds the stopped state
- *evidence*: With a fixed step, ValidationProbe writes the pointer only when that step's checkpoint is saved (src/honest_agents/callbacks.py, lines 650-665). train() always saves the current adapter to `final` (src/honest_agents/train/run.py, line 346). §5.5 says the file "will still name 150".
- *why_it_matters*: A submission script that trusts `final/` or the pointer would upload the damaged adapter.
- *proposed_change*: On a hard stop, write candidate.json naming the last checkpoint before the firing window and the rule that fired. Correct the sentence in §5.5.
- *measurable_effect*: One file per stopped arm. The handoff's copy command reads the step from it.

**9. m3 [minor]**

- *title*: Pilot and resume mechanics the plan does not state
- *evidence*: 1. The first optimizer step has lr 0 (warmup lambda current_step / 5: .venv/lib/python3.12/site-packages/transformers/optimization.py, lines 74-76; the it19c log shows lr 0 and KL 0 at step 1). So steps 1-2 roll out pure G2, and a 10-step pilot has only 5 full-lr steps.
2. A learning rate changed in the config is ignored on resume, so a re-pilot must be a fresh run directory.
3. A pilot segment with max_steps 10 must also override probe.fixed_checkpoint_step, because check_selection refuses a fixed step past max_steps (src/honest_agents/train/grpo.py, lines 181-182). It also triggers an extra probe at step 10: a last-step probe is added when max_steps is not a multiple of every_steps (src/honest_agents/train/plan.py, line 277).
- *why_it_matters*: The pilot's behavioural abort rules watch a policy that has barely moved, and a re-pilot costs more than the plan implies.
- *proposed_change*: State the segment overrides in §6. Use steps 1-2 (256 episodes) as a fresh G2 baseline under the training harness. Require a new run_name for any re-pilot.
- *measurable_effect*: 256 baseline episodes at no cost. No mixed-life train_log.jsonl.

**10. m4 [minor]**

- *title*: The override list in §5.4 omits values the run needs
- *evidence*: probe.samples_per_variant is 1 in the chain and must be set to 4 (configs/grpo_it12.yaml, line 26). run_name, output_root and the W&B name are inherited as outputs/it15/it15-g3-grpo (configs/grpo_it15_g3.yaml, lines 7-13), so five arms would collide. stop.decline_window is null, so rule 2's 10-step window is not set. data.pair_batches false is inherited from configs/grpo_colocate_qwen35.yaml (line 45) and is what keeps the block order; the plan never names it.
- *why_it_matters*: Each is a one-line config value, but a missed one changes the probe size or the data order silently.
- *proposed_change*: Add these to the override row and add a config test that asserts them for all three it20 configs.
- *measurable_effect*: One test file. The dry-run summary prints episodes_per_probe 412.

**11. m5 [minor]**

- *title*: The in-training probe costs about 24 % of the training episodes for a collapse detector
- *evidence*: data/probe_v1.jsonl without the 5 DEF-66 sources is 103 rows from 35 sources (35 plain, 34 controls, 34 twins; script counts.py). Eleven probes of 412 episodes are 4,532 episodes against 19,200 training episodes. With 35 sources, my estimate of the paired SE of probe U against step 0 is at least 4 points, so rule 3 (upper bound below -5) needs a fall of about 12 points.
- *why_it_matters*: The probe cannot resolve the effect the plan hopes for. The revisit comparison (M2) and the z monitor (M3) do more for less.
- *proposed_change*: Probe every 30 steps (6 probes), or keep the cadence and say plainly that the probe is a collapse detector.
- *measurable_effect*: About 2,060 fewer episodes per arm, roughly 3-6 hours each.

**12. m6 [minor]**

- *title*: Arm C's paraphrased system prompts are not covered by the prompt-budget check
- *evidence*: check_prompt_budget tokenises harness.SYSTEM_PROMPT for every task, not the row's own prompt (src/honest_agents/train/startup.py, lines 66-74).
- *why_it_matters*: A paraphrase longer than the slack in the 3,072-token reserve could make vLLM reject a request mid-run.
- *proposed_change*: Check each row's own system message, or require every paraphrase to be no longer than the default in tokens.
- *measurable_effect*: A startup failure instead of a mid-run one.

**13. m7 [minor]**

- *title*: §10 item 7 does not say what the rebuild uses as input
- *evidence*: There is no seed cache in data (no seeds or raw directory). The handoff forbids the upstream dataset, tasks.sources and build_dataset.sh (section A3, rule 2). The builder has a `local` source over existing task dirs (src/honest_agents/tasks/build/config.py, line 65). Whether that reproduces the twins byte for byte is not verified.
- *why_it_matters*: The executor may be unable to do item 7 as written.
- *proposed_change*: Name the `local` source over the feasible dirs as the input and test it on 20 variants first. Or drop item 7 and rely on the oracle gate plus the prescreen leak rule that §4.2 already requires for every pool variant.
- *measurable_effect*: Count of variants whose rebuild differs. A non-zero count on known-good variants means the method is unusable.

**14. m8 [minor]**

- *title*: Candidate code-level changes that are not worth building, and two small signal items
- *evidence*: Not worth building:
- Leave-one-out baseline: TRL's advantage is reward minus the group mean (grpo_trainer.py line 2841), which equals 7/8 of the leave-one-out advantage at G = 8. It is only a scale change.
- num_iterations above 1 is refused by check_batch_geometry (src/honest_agents/train/grpo.py, line 122).
- vllm_enable_sleep_mode exists (grpo_config.py line 607) but sleeps the engine at level 2 after every generate call, that is between turns (vllm_generation.py lines 750-752). It is untested with this trainer.

Small signal items:
- In my simulation of the planned pool (script sim_stop.py; G2 pass-rate distributions fitted to the it17 bank) about 33 % of groups have zero variance under U at G = 8. The 75 always-pass rows are zero-variance 90 % of the time, about 8 % of all rollouts.
- Failing episodes are 21 % of episodes but 33 % of generated tokens in the it17 bank (script bank2.py), so the DAPO token sum weights them 1.9 times per episode.
- *why_it_matters*: Saves effort on changes with no gain and points at two cheap ones.
- *proposed_change*: Skip the three. Make the `push` analysis token-weighted for action tokens. Consider replacing the 75 always-pass rows with in-band rows, since the guards and stop rules already cover retention.
- *measurable_effect*: Zero-variance group share falls from about 0.33 to about 0.25 in the simulation (the ~0.25 assumes the retention pairs are swapped as well). The push report gains a token-weighted column.

## strongest_improvements


**1. Make training-side learning measurable from files the run already writes**

- *description*: Add the missing per-episode columns (is_control, family, damage, readme_read, unweighted U, u_weight, loss tokens, status logit z) to the completions table. Add `analysis.u_dynamics revisit`, `pilot` and `rules`. Pre-register epoch 2 minus epoch 1 and epoch 3 minus epoch 1 train U as a secondary endpoint and an early readout. Run the existing status-logit hook in monitor-only mode.
- *expected_gain*: A train-side delta U with SE about 1.0 point (800 prompts x 8 samples per epoch), readable at step 100. A continuous per-step curve of the claim margin. A one-step earlier warning than rule 1 on the it19c collapse.
- *how_measured*: Paired by prompt and clustered by source from completions_<step>.parquet. On the it19c ablation the same analysis gives -0.1 +- 2.6 points with 120 prompts. The z rule is replayed on the it19c method arm (trips at step 6; rule 1 at step 7).
- *cost_and_risk*: About 200-300 source lines plus tests, and no extra rollouts. The hook touches the loss path, so it must be detached and tested against the existing report-term tests. The endpoint is in-sample, so it bounds the held-out effect from above and does not replace it.

**2. Add a fixed-state claim probe using code that already exists**

- *description*: For every arm's candidate, and for checkpoints 60, 105 and 150, score pi(claim) at the status token on G2's own transcripts from sources outside the pool (the fresh G2 select run and prescreen variants not selected). Use `python -m honest_agents.train.confidence_exact --status-score <adapter> --bank <run dir>` (src/honest_agents/train/confidence_exact.py, lines 554-589), then `analysis.calibration paired`.
- *expected_gain*: A direct, low-noise measure of whether RL moved the claim decision the right way on fixed states: AUROC of pi(claim) for failure, and the share of failing claims below 0.5, each with a source-cluster interval. It is far better powered than held-out delta U: K2 resolved +0.097 [+0.023, +0.183] with this readout (docs/results.md, row 19 (b) K2).
- *how_measured*: Paired G2 (adapter off) against candidate (adapter on) on the same transcripts. No sandbox and no sampling noise from actions.
- *cost_and_risk*: One config file and a few GPU-minutes per checkpoint on the evaluation GPU. It is a mechanistic secondary result, not a behavioural one, and it must use transcripts from sources outside the training pool.

**3. Two rehearsals before any learning step**

- *description*: (a) Executor: 3 zero-lr steps at the it20 geometry with the bank on and one forced full-length episode. (b) Evaluator: run the full it20 analysis code on the archived base and G2 held-out runs and freeze that commit in the prereg.
- *expected_gain*: Removes the execution failures most likely to void the round: out-of-memory at a 16,384-token episode (extrapolated peak 84.5 GB on a 96 GB card, about 79.7 GB on 80 GB), unknown step time at 128 episodes, a silently masked probe, and blind-written analysis code first run at the single look.
- *how_measured*: (a) Peak GB, seconds per step, KV preemptions, masked share. (b) Reproduction of the published G2 U of 33.3 (S = 2 set).
- *cost_and_risk*: (a) About 40-90 minutes on one GPU. (b) No GPU: it reads archived runs on the evaluator's machine. Risk is low; it adds about a day before the pilot.

**4. Run G2 twice and fingerprint the harness**

- *description*: Store a sha256 of the evaluation-affecting code and prompts in every run's config.json and refuse paired estimates across different fingerprints. The evaluator runs G2 once early (power and clone inputs) and once interleaved with the arms (H1's reference).
- *expected_gain*: Makes "fresh G2 on the same commit" checkable. It also turns the plan's two power scenarios (MDE90 2.5 or 3.6) into one measured number, because early G2 against late G2 is an A/A test on the real pipeline.
- *how_measured*: A/A delta U and its interval at S = 16. The joint guard failure rate of the A/A pair, against the simulated clone check.
- *cost_and_risk*: About 60-80 source lines and one extra held-out run: 8,992 episodes, about 7 GPU-hours. If the A/A interval excludes 0, the round must stop and find the harness difference, which is the point.

**5. Replace the KL projection gate with measured quantities**

- *description*: Extend the pilot to 20 steps. Fit the KL slope on steps 6-20. Log the gradient-norm ratio of the KL term to the policy term on the first steps. Decide beta and lr by a pre-registered rule on those numbers. Say plainly that beta 0.04 is a monitor.
- *expected_gain*: A gate whose outcome does not hinge on a 5-point extrapolation that the plan's own reference data puts at 0.055-0.123 against a 0.08 threshold. If the measured ratio confirms about 2 %, the plan can choose a real anchor or none, knowingly. I cannot promise a larger effect from this, only fewer wasted restarts.
- *how_measured*: KL slope per step and its fit error from 15 points, and the logged gradient-norm ratio, compared with the it19c values (0.00085 and 0.00058 per step).
- *cost_and_risk*: About 40 source lines to generalise the existing ratio measurement, plus 10 more pilot steps that are the run's own steps 11-20. A beta ladder, if run, costs 3 x 20 steps on idle GPUs.

## things_revision3_got_right

- The KL reference is what the plan says: a new LoRA on merged G2, with reference log-probabilities computed with adapters disabled (TRL grpo_trainer.py lines 2777-2789; no separate ref model for a PEFT policy, lines 968-975).
- The inherited config chain and nearly every override are right. I traced grpo_it15_g3 -> grpo_it13 -> grpo_it12_96gb -> grpo_it12 -> grpo_colocate_qwen35 -> base and confirmed: beta 0.0, lr 3e-5, accumulation 64, mask_truncated_completions true (base.yaml line 132), probe.steps [5, 10, 15] overriding every_steps, min_infeasible 40, shuffle_dataset false, pair_batches false, 16,384 / 4,096 tokens, 36 tool turns. A YAML null does override the inherited probe.steps (_deep_merge in train/config.py lines 288-295).
- The data-order claim holds. RepeatSampler iterates range(n) when shuffle is off (TRL utils.py lines 890-909), 800 rows / 16 prompts gives 50 steps per epoch, so steps k, k+50 and k+100 see the same prompts. This is what makes the revisit comparison possible; keep it.
- Batch geometry is exact. DAPO divides by the token count of the whole generation round (grpo_trainer.py lines 3229-3234), so batch 1 x accumulation 128 gives one token-mean gradient, and check_batch_geometry forces one round per optimizer step so a checkpoint never falls inside a round.
- Never masking a token is right and matches the code: outside crgrpo mode the status mask and report term are off (train/run.py lines 215-219). The it19c collapse came from training the status token by the report term alone.
- The per-row weight route works: TRL forwards every dataset column to the reward function (grpo_trainer.py lines 1669-1670) and HonestHarborEnv.reset accepts extra columns (env.py line 195). Writing the weight per row rather than by family is correct, because surface pairs share the high_risk label.
- The episode-level leak mask (mask_leaking_groups) is kept, and scale_rewards none is enforced in code (train/grpo.py lines 236-237).
- save_steps 5 with a fixed candidate step, and stop-rule state rebuilt from log_history on resume (callbacks.py lines 500-512 and 534-540), make a 150-step run resumable. Optimizer and scheduler state are restored from the checkpoint.
- One pinned budget through --train-config is enforceable: run_eval reads the caps, sampling, report turns and command timeout from the training config (eval/run_eval.py lines 300-354).
- The stop thresholds I could test are sensible. In a 400-run simulation of an unchanged G2 on the planned pool, the inherited rules false-stop 2.8 % of runs, and the zero-variance share averages 0.33 against the 0.7 limit. Rule 1 with its 3-point margin never fires on the it19c ablation and does fire on the collapsed method arm.
- Weight sync is not a concern at these learning rates: in the it19c ablation the sampler-trainer log-probability gap stayed flat at 0.006-0.008 from step 1 to step 30 while KL rose to 0.021, so the merged bf16 weights carried the update at 3e-5.
- The estimator and significance rule match the code: paired_estimate is at power.py line 687, uses a two-way source x pair variance with df = min(sources, pairs) - 1, and its `significant` flag is the sign-flip test the plan correctly tells readers not to use.
- The counts I could check are right: probe_v1 minus 5 DEF-66 sources is 103 rows (69 feasible, 34 infeasible); 60 select sources; split 1,388 / 539 / 565; 76 defective sources; 60 high_risk select twins; G2's pool pass rates 0.847 plain and 0.744 control with P(claim | pass) 0.991-0.995.
- The executor / evaluator split with all arms submitted before one combined table, S = 16 for every arm, and the fresh G2 baseline are the right structure; M5 only asks that the baseline also be re-run alongside the arms.