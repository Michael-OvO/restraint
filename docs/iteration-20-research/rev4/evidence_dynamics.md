# evidence:dynamics


## summary

Scope: train-side data only (it17 G2 pool banks, it19c select runs, ablation and G3 training completions/logs/adapters). Scripts and outputs are in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dynamics (load.py, describe.py, sim.py, logs.py, misc.py, lora_geom.py, adam_null.py and their *_out.txt files).

1. The only comparable RL run (it19c ablation, 30 steps, 8x8, lr 3e-5, beta 0.04) produced an update that is statistically a momentum random walk. Its 5-step LoRA increments have cosines 0.708 / 0.404 / 0.217 / 0.119 / 0.062 at lags 1-5; AdamW on pure noise gives 0.688 / 0.387 / 0.219 / 0.122 / 0.066. An independent run on the same prompts per step (the stopped method arm, 7 steps) has cosine 0.005 with it; G3 has -0.005. Implied per-coordinate signal-to-noise per step is at most about 0.02-0.05, so signal is at most 1-6 % of the squared displacement at step 30.

2. That noise-dominated update was not harmless. Recomputed on the select split (S=4, 184 variants, paired, source bootstrap, 90 %): U changed by -6.5 [-10.9, -2.2] against G2 (pass rate -3.4 [-7.0, 0.0], false claims per feasible episode +3.8 [+0.9, +6.7]). On the training prompts themselves, pass rate at the second visit (15 updates later) changed by +1.5 [-2.3, +5.3]; 24 prompts went up, 23 down, 27 unchanged. No train-side measurement shows a positive RL effect on solving.

3. Plan 20's step carries about 11 effective pass-versus-fail contrasts (N_eff = sum of n_pass*n_fail/G over feasible groups; simulated 11.2, sd 2.6, for 128 episodes), against 5.3 measured in the ablation's 64-episode steps. That is 87 per 1,000 episodes, 40 % of the theoretical maximum of 219. Simulated zero-variance groups under U: 30 % (real G=8 groups under U: 25-28 %); 69 % of episodes carry a non-zero advantage.

4. Advantage mass under plan 20's estimator (U, c=d=1, twin weight 0.44, no scaling): 69 % on pass/fail, 16 % on claim decisions within pass or fail, 15 % on twins (real v3/v4 groups: 67-68 / 11-12 / 20-22 %). The solve push is +0.137 per episode; the claim-given-fail push -0.028; the twin claim push -0.030; the claim-given-pass push +0.003.

5. DAPO's token sum is the correct trajectory-level policy gradient, so it adds variance, not bias. Failing feasible episodes are 16 % of episodes but 42 % of the gradient weight; failing episodes in the top length quartile are 6 % of episodes and 25 % of the weight. Of 64 episodes per step, 46 have a non-zero advantage, the effective sample size is 22, and the top 5 episodes carry 38 % of the weight. Fails are 38-57 % longer than passes overall but only 7 % longer within a mixed group: the length gap is between prompts.

6. G=16 against G=8 buys little per episode (93 against 87 contrasts per 1,000 episodes); G=4 is worse (75). Doubling prompts to 32 doubles N_eff per step (22.4). Dropping the 75 always-pass plain rows gives +13 % (12.6); band [0.25, 0.75] gives +34 % (15.0) but raises the share of plain rows with p below 0.5 from 10 % to 34 %.

7. KL: the ablation's KL grew 0.000725 per step (SE 0.000095) over steps 6-30; a power-law exponent of 1.05-1.11 (diffusion, not drift); a saturating fit (K_inf 0.031) is not distinguishable. Projected KL at step 150 under lr 2e-5: 0.02 (saturating), 0.05 (linear, lr squared), 0.07 (linear, lr). beta 0.04 is weak: its per-token coefficient is about 2 % of the mean absolute advantage at KL 0.02 and 3.5 % at KL 0.05, and G3 (beta 0) and the ablation (beta 0.04) have the same update norm at step 15 (0.708 against 0.705).

8. Under an Adam noise model, signal grows as lr x steps x sqrt(batch) and noise energy as lr squared x steps. Plan 20 against the ablation: signal x5.3, noise energy x2.2-2.7. So plan 20 is better conditioned, but it breaks even on pass rate only if the ablation's true signal gain was at least 2.4-3.5 points in 30 steps, which the data cannot confirm. Plausible change in training-distribution pass rate after 150 steps: about -8 to +6 points, centre near -2 to +1, sign not identified.

9. Top three bottlenecks: (a) gradient noise: about 11 contrasts per step, each spread uniformly over about 1,900 model tokens and 16 actions; (b) drift outruns signal at this lr and beta: the KL budget is spent on noise, and the project saw regressions at action KL 0.009-0.02 per token (it18, ablation) while plan 20 warns at 0.05 and stops at 0.1; (c) little information per prompt: 3 visits x 8 samples give about 3.2 contrasts per in-band prompt over the whole run, and 30 % of groups plus all twin rows (25 % of episodes) carry nothing for solving.


## facts


**1. What u_checks CHECK 2 computes. Advantages are std-normalised within each prompt group: a = (u - group mean) / (group sd with ddof 1 + 1e-4); a zero-variance group contributes zeros but still counts in n. Variance shares are a sequential split of sum a^2: between pass and fail cells, then damage within pass/fail, then claim within (pass/fail, damage). Pushes are sums of a_i times a centred indicator divided by all samples n: push_solve uses (y_i - group mean y); push_claim|pass and |fail centre the claim indicator within the pass or fail cell; 'via capability' is the between-cell part; push_damage uses the damage indicator.**

- *value*: Bank and select groups have G = 4 (S = 4 samples per variant); training groups have G = 8. Units are sd-normalised, not U points.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_design/u_checks.py (lines 201-239 group_contrib, 258-275 derive, 309-335 inputs)
- *confidence*: verified

**2. u_checks CHECK 2 headline values at c = 1 (std-normalised).**

- *value*: it17a bank (400 groups, n 1,600): pass/fail share 80.4 % [72.9, 87.4], claim 19.6 %, push_solve +0.120 [+0.102, +0.137], push_claim|pass +0.005, |fail -0.028, via capability +0.043, overall +0.020 [+0.003, +0.037], zero-variance groups 63.0 %. it19c select (184 groups): shares 44.7 / 38.2 (damage) / 17.1, push_solve +0.054, push_claim overall -0.043, zero-variance 70.1 %; infeasible rows push_claim -0.117. Ablation training groups (240, G=8): 57.0 / 27.8 / 15.2, push_solve +0.170, push_claim|fail -0.120, overall -0.025, push_damage -0.082, zero-variance 27.5 %; infeasible push_claim -0.264. G3 groups (99): 56.0 / 30.3 / 13.7, push_solve +0.173, zero-variance 25.3 %. c = 2 and 3 deepen the net blanket claim push (bank -0.023, -0.026, -0.028; ablation -0.109, -0.117, -0.121).
- *source*: docs/iteration-20-research/u_checks.txt (lines 65-77, 104-116, 143-155, 184-196, 223-235, 342-354, 381-393, 815-828)
- *confidence*: verified

**3. u_checks does not use plan 20's estimator. It divides by the group sd and uses G = 4 for the banks; plan 20 uses scale_rewards none, G = 8 and twin weights. The plan itself lists this as finding M2 and requires a re-run.**

- *value*: Std-normalisation removes the effect of the twin weight and inflates low-variance groups; all numbers below were recomputed with advantage = U - group mean.
- *source*: docs/iteration-20-rl-design.md (line 114 and line 636); ~/Documents/honest-agents-archive/staging/analysis/it20_design/u_checks.py (line 216)
- *confidence*: verified

**4. All episodes used in the new analyses are train-split sources; no held-out file was opened.**

- *value*: bank_feas 1,600 episodes kept (split bucket train), bank_twin 422 (train), select_g2 and select_abl 736 each (train + select), ablation completions 1,919, G3 completions 792 after DEF-66 exclusion.
- *source*: command: env -u VIRTUAL_ENV uv run --frozen python ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dynamics/load.py <scratch>; output in load_log.json in the same directory
- *confidence*: verified

**5. Per-prompt pass distribution of G2 on the train pool (it17 bank, S = 4, 12,288-token budget).**

- *value*: Plain (177 variants): 0/4 5.6 %, 1/4 3.4 %, 2/4 7.3 %, 3/4 13.6 %, 4/4 70.1 %; mean p 0.847; mixed 24.3 %. Controls (223): 9.4 / 8.5 / 9.9 / 19.3 / 52.9 %; mean p 0.744; mixed 37.7 %. Beta-binomial fits: plain Beta(0.93, 0.17), E[p(1-p)] 0.069; controls Beta(0.88, 0.31), E[p(1-p)] 0.105.
- *source*: command: describe.py <scratch> 0.44 (section B) and sim.py <scratch> (priors); outputs describe_w044.txt and sim_out.txt in the scratch directory; data runs/def74/it17_rollouts_feasible/episodes.jsonl
- *confidence*: verified

**6. Claim rates of G2 on feasible train-pool rows.**

- *value*: P(claim | pass) 0.993 (n = 1,264); P(claim | fail) 0.5625 (n = 336): plain 0.639, controls 0.526. Select split: 1.000 (n = 363) and 0.688 (n = 109). Ablation training rollouts: 0.987 and 0.525. G3 rollouts: 0.991 and 0.370.
- *source*: describe_w044.txt section A (scratch directory), from the same bank, runs/eval/it19c_g2_select/episodes.jsonl and the archived completions
- *confidence*: verified

**7. Twin claim and damage rates of G2 on train-side twins.**

- *value*: it17 twin bank (211 variants, S = 2): claim 17.8 %, damage 10.2 %; 73.5 % of variants claim 0 of 2. By family: high_risk claim 0.19 / damage 0.20; remove_input 0.30 / 0.00; revoke_permission 0.02 / 0.00. Select twins (66 variants, S = 4): claim 11.4 %, 68.2 % at 0 of 4. Surface-pair twins in training pools (G = 8): claim 31.5 % (ablation) and 32.9 % (G3); 26.7 % and 28.9 % of groups at 0 of 8.
- *source*: describe_w044.txt sections A and B; data ~/Documents/honest-agents-archive/iterations/17-it17a-data-rollouts/eval/it17_rollouts_infeasible/episodes.jsonl.gz and the archived completions
- *confidence*: verified

**8. Real G = 8 training groups scored with plan 20's estimator (U with c = d = 1, twin weight 0.44, advantage = U - group mean, no scaling).**

- *value*: Zero-variance groups: 27.5 % of 240 (ablation; plain 23.3 %, control 25.6 %, twin 32.2 %) and 25.3 % of 99 (G3). Episodes with non-zero advantage: 72.3 % and 74.7 %. Share of sum |A|: pass/fail on feasible rows 68.0 % / 66.6 %; claim or damage within pass or fail 11.9 % / 11.1 %; twin rows 20.2 % / 22.3 %. Mean |A| per episode 0.330 / 0.325. With twin weight 1.0 the twin share is 36.5 % / 39.5 % of |A| and 41.8 % / 45.9 % of A^2.
- *source*: commands: describe.py <scratch> 0.44 and describe.py <scratch> 1.0; outputs describe_w044.txt and describe_w1.txt (section C); data ~/Documents/honest-agents-archive/staging/outputs/it19/it19-crgrpo-ablation-r2/completions and ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/it15-g3-grpo/completions
- *confidence*: verified

**9. U is a coarser reward than the ones those runs trained on, so more groups carry no gradient.**

- *value*: Logged zero-variance group share under the runs' own rewards: 18.3 % (ablation, CR-GRPO table) and 5.0 % (G3, honest reward with efficiency bonus); under U on the same groups: 27.5 % and 25.3 %.
- *source*: logs_out.txt (honest/zero_variance_group_share) from outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl and ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/it15-g3-grpo/train_log.jsonl; describe_w044.txt section C
- *confidence*: verified

**10. Effective number of pass-versus-fail contrasts per step in the real runs, N_eff = sum over feasible groups of n_pass * n_fail / G.**

- *value*: Ablation: mean 5.28 per step (sd 1.97, min 0.88, max 8.62) from 5.0 feasible groups and 64 episodes; G3: 4.47 (sd 2.14).
- *source*: describe_w044.txt section C
- *confidence*: verified

**11. DAPO in TRL sums the per-token loss over all model tokens of the batch and divides by one global token count; with an episode-level advantage that is the trajectory-level REINFORCE sum, so an episode's gradient weight is |A_i| times its model-token count.**

- *value*: loss = (per_token_loss * mask).sum() / num_items_in_batch for loss types cispo, dapo, vespo; mask = completion_mask * tool_mask. Logged completions/mean_length (1,168-2,932) matches model-token counts I tokenised from the completions within about 3 %.
- *source*: .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 2549-2550 and 3229-3234); sanity check command in the session (ratio of tokenised total to logged length 1.81, model share 0.575)
- *confidence*: verified

**12. Token counts of passing against failing feasible episodes in training rollouts.**

- *value*: Ablation: pass mean 1,881 model tokens (median 1,608), fail 2,589 (median 2,155), ratio 1.38. G3: 1,814 against 2,849, ratio 1.57. Inside mixed groups the gap almost vanishes: 2,270 (pass) against 2,432 (fail), ratio 1.07, fail longer in 50 % of 108 groups (G3: 2,282 against 2,427, 48.9 % of 45 groups). Pass rate falls with length: 0.83 at 1-2k tokens, 0.57 at 3-5k, 0.47 above 5k (ablation).
- *source*: describe_w044.txt section C and misc_out.txt section 5
- *confidence*: verified

**13. Share of the DAPO gradient weight (|A| x model tokens) by episode type, real G = 8 groups under U.**

- *value*: Ablation: feasible fails are 15.7 % of episodes, 38.0 % of sum |A| and 41.7 % of DAPO weight; feasible passes 46.8 % / 39.8 % / 41.0 %; twin claim or damage 12.0 % / 11.2 % / 9.4 %; clean twin non-claims 25.5 % / 11.0 % / 8.0 %. Feasible fails in the top length quartile (at least 2,456 tokens): 6.3 % of episodes, 14.3 % of sum |A|, 24.8 % of DAPO weight; top decile (at least 3,465 tokens): 3.0 % / 7.1 % / 15.8 %. G3: 17.4 % of episodes are feasible fails with 42.3 % of the weight; top-quartile fails 8.0 % of episodes and 28.4 % of the weight. Token weighting raises the solve push per feasible episode from 0.198 to 0.237 (+20 %).
- *source*: describe_w044.txt section C
- *confidence*: verified

**14. Effective sample size of a 64-episode step under U with DAPO weights.**

- *value*: Ablation: 46.2 episodes with non-zero advantage; effective sample size 28.2 by |A| and 21.9 by |A| x tokens; the top 5 episodes carry 38.3 % of sum |A| x tokens and 57.4 % of sum A^2 x tokens. G3 (53 scored episodes per step after DEF-66): 39.5, 24.6, 18.3, 43.3 %, 62.4 %.
- *source*: misc_out.txt section 4
- *confidence*: verified

**15. Simulated plan-20 step (6 plain with 25 % always-pass, 2 lone controls, 2 surface pairs, 1 high_risk pair, 1 remove_input or revoke pair; G = 8; bands applied at an S = 8 prescreen; twin weight 0.44; 6,000 simulated steps).**

- *value*: Zero-variance groups 30.2 % (plain 40.2 %, controls 25.8 %, twins 21.8 %); non-zero-advantage episodes 69.2 %; 11.2 groups per step with gradient; mean |A| 0.398; |A| shares 68.8 % pass/fail, 15.7 % claim, 15.4 % twin (A^2 shares 79.7 / 11.1 / 9.2); N_eff 11.17 per step (sd 2.61), 87 per 1,000 episodes; pushes per episode: solve +0.137, claim|pass +0.003, claim|fail -0.028, twin claim -0.030; feasible pass rate 0.764; false claims per feasible episode 0.135. With twin weight 1.0 the twin share is 29.4 % of |A| and 34.4 % of A^2 and the twin claim push -0.069.
- *source*: command: sim.py <scratch>; output sim_out.txt and sim_table.csv in the scratch directory
- *confidence*: derived

**16. Group size against number of prompts, same simulation.**

- *value*: 16 x 16 (256 episodes): N_eff 23.9 per step, 93 per 1,000 episodes, zero-variance 20.3 %. 32 x 8 (256): 22.4, 88 per 1,000, 30.1 %. 32 x 16 (512): 48.0, 94 per 1,000. 32 x 4 (128): 9.6, 75 per 1,000, zero-variance 44.8 %. Per-episode efficiency follows (G-1)/G: 0.75, 0.875, 0.9375 for G = 4, 8, 16.
- *source*: sim_out.txt
- *confidence*: derived

**17. Pool band choices, 16 x 8, same simulation.**

- *value*: Plan (band [0.4, 0.875], 25 % always-pass): N_eff 11.17. Same band without always-pass rows: 12.58 (+13 %), zero-variance 24.7 %. Band [0.25, 0.875]: 13.20. Band [0.125, 0.875]: 13.10. Band [0.25, 0.75]: 14.98 (+34 %), zero-variance 16.5 %, but feasible pass rate 0.624, false claims per feasible episode 0.221, claim|fail push -0.049. No band: 8.23, zero-variance 43.7 %.
- *source*: sim_out.txt
- *confidence*: derived

**18. What the plain band selects after an S = 8 prescreen (beta-binomial model).**

- *value*: Band [0.4, 0.875]: 25.9 % of the plain pool, mean true p 0.755, E[p(1-p)] 0.153, zero-variance at G = 8 26.3 % (14.3 % at G = 16), 10.3 % of rows with p below 0.5. Band [0.25, 0.75]: 21.0 % of the pool, mean p 0.588, E[p(1-p)] 0.197, 34.1 % of rows below 0.5. 8/8 rows: mean p 0.981, zero-variance 89 % in the model.
- *source*: sim_out.txt (last block)
- *confidence*: derived

**19. The model overstates how fixed the always-pass rows are. Prompts that passed 4 of 4 in the bank passed less often when sampled 8 times in training (16k budget, slightly drifted policy).**

- *value*: 34 groups: pass rate 0.882; 55.9 % of groups at 8 of 8. Bank 3-of-4 prompts: 0.766 (23 groups, 13.0 % at 8 of 8). Bank 2-of-4: 0.519 (13 groups). So the simulated 30 % zero-variance share is an upper estimate; the real G = 8 groups gave 25-28 %.
- *source*: commands run in the session joining bank counts to the ablation and G3 groups (output in the transcript); data as above
- *confidence*: verified

**20. Logged series, ablation r2 (30 steps, 64 episodes per step, lr 3e-5 after 5 warmup steps, beta 0.04, DAPO, scale_rewards none, max_grad_norm 1.0).**

- *value*: KL 0 at steps 1-2, 0.0038 at step 10, 0.0103 at step 15, 0.0082-0.0220 over steps 16-30 (mean 0.0149). Clip fraction 0 at every step (low, high and region). Grad norm mean 0.142, range 0.067-0.229, so the clip at 1.0 never binds. Reward mean 0.34 (sd across steps 0.23). Feasible pass rate 0.749 (first half 0.741, second half 0.757). Feasible claim rate 0.871; infeasible claim rate 0.315. Truncated share mean 0.26 %, max 4.7 %. Model tokens per episode mean 1,930. Step time mean 386 s (rollout 229 s, training 65 s); 3.21 h for 30 steps; peak GPU memory 74.4 GB. Importance-sampling ratio mean 1.0000, per-token max reaching the cap of 3.0. Combined loss mean 0.027 (sd 0.047); the KL term is not logged separately and beta x KL is at most 0.0009.
- *source*: outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl and run_config.json in the same directory; command logs.py <scratch>, output logs_out.txt
- *confidence*: verified

**21. Logged series, G3 (15 steps, beta 0, KL not logged) and the stopped method arm (7 steps).**

- *value*: G3: grad norm mean 0.137 (0.089-0.176); clip fraction 0; feasible pass rate 0.718 in steps 1-7 and 0.606 in steps 8-15; feasible claim rate 0.854 then 0.784; truncated share 0.42 % mean; step time 639 s. Method arm: grad norm 0.216; feasible claim rate 0.95 to 0.275 by step 7; step time 686 s. The first optimizer step has lr 0 in all three runs.
- *source*: ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/it15-g3-grpo/train_log.jsonl; outputs/it19/it19-crgrpo/train_log.jsonl; logs_out.txt
- *confidence*: verified

**22. KL growth fit on the ablation.**

- *value*: Linear, steps 6-30: slope 0.000725 per step (SE 0.000095), R^2 0.72; through the origin against full-lr steps: 0.000764 per step. Power law: exponent 1.05 (all steps) and 1.11 (steps 8-30), i.e. linear in steps, not quadratic. Saturating fit K_inf (1 - exp(-n / tau)): K_inf 0.031, tau 29.8 steps, SSE 2.45e-4 against 2.72e-4 for the line; K_inf between 0.018 and unbounded is not rejected, so 30 steps cannot tell the two apart. Mean KL: 0.0048 in steps 6-15, 0.0136 in steps 16-22, 0.0161 in steps 23-30.
- *source*: logs_out.txt and misc_out.txt section 1
- *confidence*: verified

**23. Projected KL for plan 20 (lr 2e-5, 150 steps), by scaling assumption.**

- *value*: Saturating fit (diffusion proportional to lr squared, restoring rate proportional to lr): 0.0135 at step 50, 0.018 at 100, 0.020 at 150. Linear with slope proportional to lr squared: 0.050 at 150. Linear with slope proportional to lr (the plan's implicit scaling): 0.071-0.075. From the LoRA displacement (KL about 0.024 x squared norm): 0.060-0.072. If the update were signal-dominated (KL proportional to (lr x steps) squared): 0.20. Batch size does not enter under noise-dominated Adam.
- *source*: logs_out.txt (projection block) and misc_out.txt section 1
- *confidence*: derived

**24. The pilot KL gate (linear extrapolation from steps 6-10, threshold 0.08) sits on the edge of its own noise.**

- *value*: The ablation's steps 6-10 give slope 0.000853 per step, extrapolating to 0.123 at step 150 for lr 3e-5; scaled to lr 2e-5 that is 0.055 (lr squared) or 0.082 (lr). The same run's later slope was lower (mean KL rose only 0.0136 to 0.0161 between steps 16-22 and 23-30), so an early-slope extrapolation is biased high.
- *source*: KL series in logs_out.txt (steps 6-10: 0.00055, 0.00104, 0.00169, 0.00303, 0.00382); gate text in docs/iteration-20-rl-design.md (lines 457-458)
- *confidence*: derived

**25. LoRA displacement of the ablation across checkpoints (dW = (alpha/r) B A summed over 248 modules; B starts at 0).**

- *value*: Frobenius norm at steps 5, 10, 15, 20, 25, 30: 0.196, 0.502, 0.705, 0.854, 0.975, 1.074. Squared norm grows 0.043-0.049 per full-lr step in every 5-step block (0.0426, 0.0490, 0.0466, 0.0442, 0.0403). Growth exponent of the norm against cumulative lr: 0.565 (0.5 is a random walk, 1.0 a steady drift; AdamW on pure noise gives 0.601). Sum of squared increment norms over squared final norm: 0.332 (pure-noise AdamW: 0.335).
- *source*: command: lora_geom.py <scratch>/lora_geom.json on ~/Documents/honest-agents-archive/staging/outputs/it19/it19-crgrpo-ablation-r2/checkpoint-{5,10,15,20,25,30}/adapter_model.safetensors; null from adam_null.py, output adam_null_out.txt
- *confidence*: verified

**26. The time structure of the ablation's update matches AdamW (beta1 0.9) fed pure noise.**

- *value*: Mean cosine between 5-step increments at lags 1-5: measured 0.708, 0.404, 0.217, 0.119, 0.062; pure-noise simulation 0.688, 0.387, 0.219, 0.122, 0.066. A per-coordinate signal-to-noise of 0.10 per step would give 0.723, 0.454, 0.300, 0.204, 0.140. The repo sets no optimizer options, so the transformers defaults apply (adam_beta1 0.9, adam_beta2 0.999, weight decay 0).
- *source*: lora_geom output and adam_null_out.txt; defaults in .venv/lib/python3.12/site-packages/transformers/training_args.py (lines 243-246, 820-832); grep of src/honest_agents/train and configs found no override
- *confidence*: verified

**27. Two independent runs from the same weights, the same LoRA init and the same prompts at every step moved in orthogonal directions.**

- *value*: Cosine between the method arm's update (7 steps) and the ablation's at step 5 / step 10: 0.005 / 0.006. Both runs read the same data order (sha256 e56d85f9...), the same 8 prompts at each of steps 1-7, and the LoRA A matrices differ by 0.05 in squared norm out of 2,645. G3 at step 15 against the ablation at step 15: -0.005 (different reward; 41 of 120 prompts shared). In the noise model a cosine of 0.005 corresponds to a per-coordinate signal-to-noise of about 0.02 per step (0.0044 at 0.02, 0.0146 at 0.05), i.e. signal is about 1 % of the squared displacement at step 30 (5.8 % at 0.05). The method arm's update also contains its report-term gradient (norm ratio 0.84-1.64 to the main gradient), which can dilute the cosine by up to about half.
- *source*: lora_geom output; adam_null_out.txt; outputs/it19/it19-crgrpo/data_order.json and the ablation's data_order.json; per-step prompt-set comparison run in the session; docs/iteration-19-results-crgrpo.md (lines 30-31) for the gradient ratio
- *confidence*: derived

**28. beta 0.04 had no measurable effect on the update norm over 15 steps and is small against the policy-gradient term.**

- *value*: Update norm at step 15: G3 (beta 0) 0.708; ablation (beta 0.04) 0.705. Per token, both terms multiply grad log pi: the policy-gradient coefficient is the advantage (mean |A| 0.33-0.40 under U, non-zero on about 70 % of episodes); the KL coefficient is beta x (1 - exp(ref - logp)), about beta x sqrt(2 KL): 0.008 at KL 0.02 (2.2 % of mean |A|), 0.013 at KL 0.05 (3.5 %), 0.018 at KL 0.1 (5 %). beta 0.2 would give 11 % and 18 % at KL 0.02 and 0.05; beta 1.0 gives 56 % and 88 %. A weak restoring rate fitted from the slowing of squared-norm growth is 0.002 per step (relaxation about 500 steps at lr 3e-5), not distinguishable from zero.
- *source*: lora_geom output; logs_out.txt (force comparison and OU fit); KL form in .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 3160-3166 and 3210-3211)
- *confidence*: derived

**29. Change on the training prompts themselves in the ablation (each of 120 prompts sampled 8 times at step k and again at step k + 15; paired by prompt, source bootstrap, 90 %).**

- *value*: Pass rate, plain + control (75 prompts): 0.741 to 0.757, +1.5 points [-2.3, +5.3]; plain -0.3 [-5.8, +5.1]; controls +2.8 [-2.1, +7.8]. U with twin weight 1: -1.3 [-6.2, +3.6]; with weight 0.44: 0.0 [-4.1, +4.3]. Twin claim rate +3.6 [-0.6, +7.8]. False claims per feasible episode +0.3 [-3.2, +3.7]. Model tokens -1 [-54, +51]. Per prompt: 24 up, 23 down, 27 unchanged; mixed to all-pass 6, all-pass to mixed 5.
- *source*: describe_w044.txt section D and misc_out.txt section 3
- *confidence*: verified

**30. Change on unseen train-side prompts: ablation at step 30 minus G2 on the select split (S = 4, 184 variants after DEF-66, paired by variant, source bootstrap, 90 %). The U difference is new; the pass and false-claim differences reproduce the published table.**

- *value*: U, all variants: 30.8 to 24.3, -6.5 [-10.9, -2.2]. U plain -7.2 [-14.9, 0.0]; controls -8.0 [-16.4, 0.0]; twins -4.5 [-13.1, +4.0]. Pass rate plain + control: 76.9 to 73.5, -3.4 [-7.0, 0.0]. False claims per feasible episode 15.9 to 19.7, +3.8 [+0.9, +6.7]. Twin claim rate +1.9 [-2.3, +6.1]; twin damage +2.7 [-1.7, +7.0].
- *source*: misc_out.txt section 2, from runs/eval/it19c_g2_select/episodes.jsonl and runs/eval/it19c_ablation_select/episodes.jsonl; published values in docs/iteration-19-results-crgrpo.md (lines 40-50)
- *confidence*: verified

**31. The in-training probe (data/probe_v1.jsonl, 80 feasible episodes, S = 1) shows no trend.**

- *value*: Ablation task success at steps 0 / 15 / 30: 0.713 / 0.688 / 0.713. G3 at steps 0 / 5 / 10 / 15: 0.738 / 0.738 / 0.750 / 0.700.
- *source*: logs_out.txt (probe series), from the two train_log.jsonl files
- *confidence*: verified

**32. The project has seen regressions at action KL levels far below plan 20's limits.**

- *value*: Iteration 18 aborted at a held-out action KL of 0.02 per token and held drift at 0.009; twins destroyed were still 31 of 120 against G2's 13. Iteration 17 at 0.18 per token: 53 of 120. The ablation at 0.015-0.02 per token (on-policy estimator): select U -6.5. Plan 20 warns at 0.05 and stops at 0.1. The estimators differ (it18: G2's own next tokens on held-out transcripts; TRL: sampled tokens), so the levels are indicative only.
- *source*: docs/iteration-18-design.md (lines 13-15 and 44-48); docs/results.md (rows for iterations 17 and 18); docs/iteration-20-rl-design.md (lines 470-471)
- *confidence*: verified

**33. Scaling of plan 20 against the ablation under noise-dominated Adam (per-step displacement proportional to lr and independent of batch size; signal per step proportional to lr x sqrt(effective contrasts)).**

- *value*: Signal displacement: (2e-5 / 3e-5) x (147 / 27 full-lr steps) x sqrt(11.17 / 5.28) = 5.3x. Noise energy (squared displacement): 0.444 x 5.44 = 2.4x by the lr-squared rule; 2.2x to 2.7x from the fitted displacement (projected squared norm 2.54-3.07 against 1.152). Signal share of the final squared displacement: about 5-11 % if the ablation's per-coordinate signal-to-noise was 0.02, up to about 40 % if it was 0.05 (ablation itself: 1-6 %). At a fixed total number of episodes and a fixed noise budget, the split between batch size and steps does not change the signal; total episodes and per-episode signal do.
- *source*: logs_out.txt (projection block), sim_out.txt, adam_null_out.txt (150-step block)
- *confidence*: derived

**34. Estimated change in training-distribution pass rate after 150 steps, with assumptions: gain proportional to signal displacement, harm proportional to noise energy, the ablation's select change (-3.4 points) = gain g minus harm d.**

- *value*: Plan-20 change on unseen pool prompts is about 5.3 g - k (g + 3.4) with k between 2.2 and 2.7: -7.5 to -9.2 points if g = 0; break-even needs g of 2.4 to 3.5 points; +6 if g = 5. If the ablation's true select change was 0 (the upper end of its interval), the plan's change is at least 0. Adding prompt-specific gains on rows already visited twice (revisit +1.5 [-2.3, +5.3] per 15 updates in the ablation), the plausible range on the training distribution is about -8 to +6 points with the centre near -2 to +1. No measurement fixes the sign.
- *source*: model stated here; inputs from misc_out.txt sections 2-3 and logs_out.txt
- *confidence*: derived

**35. Plan 20's own training log can measure the training-distribution change, because TRL repeats the file order every epoch.**

- *value*: Epoch 1 against epoch 3 on the same feasible prompts at G = 8: with 600 prompts and E[p(1-p)] 0.14, SE 0.76 points, MDE90 2.2 points (2.6 with a design effect of 1.3). In the ablation the same contrast had 75 prompts and a 90 % half-width of 3.8 points.
- *source*: misc_out.txt section 6; epoch order in docs/iteration-20-rl-design.md (lines 306-308)
- *confidence*: derived

**36. Step time at plan 20's 128 episodes per step has not been measured.**

- *value*: Measured at 64 episodes: 386 s per step (ablation rerun node), 639 s (G3), 686 s (method arm), i.e. 6.0-10.7 s per episode. A linear scale-up would put 150 steps at roughly 32-57 h per arm before probes.
- *source*: logs_out.txt (step_time rows)
- *confidence*: not verified

**37. Coherent low-dimensional pushes do move within 7-15 steps at 64 episodes per step, unlike solving.**

- *value*: G3: control declines in rollouts 16 to 39 % in 15 steps with read rate unchanged (46 / 46 %). Method arm: pi(claim) at the status token 0.73 to 0.08 over steps 1-7.
- *source*: docs/iteration-15-g3-rl-read.md (lines 70-72 and 84-87); docs/iteration-19-results-crgrpo.md (lines 28-31)
- *confidence*: verified

## implications_for_plan

- Restate section 9 with the train-side prior: the only comparable run moved select U by -6.5 [-10.9, -2.2] and trained-prompt pass rate by +1.5 [-2.3, +5.3]. The expected effect of revision 3 as written is centred at or below zero, not between 0 and +5. Expected measurable effect: none on the run itself; it changes what H1 is powered against and makes the noise controls below necessary.
- Cut the noise energy before spending GPUs: lr 1e-5 instead of 2e-5 at the same 150 x 128. Under the Adam noise model this gives noise energy x0.25 and signal x0.5, so the signal-to-harm ratio doubles. Expected measurable effect: squared LoRA displacement at step 150 about 0.65-0.8 instead of 2.5-3.1, and KL at step 150 about 0.012 (lr-squared scaling) to 0.037 (lr scaling) instead of 0.02-0.075.
- Use the seeds as one candidate: average the merged weight deltas of A1-A3 (or all five GPUs if B and C are dropped) and pre-register the average as the candidate. Noise energy falls by the number of seeds at the same signal, which is what tripling the batch would buy, at no extra compute. Expected measurable effect: squared norm of the averaged delta about 1/3 of a single seed's plus the shared part; KL of the average about a third. Untested for RL deltas in this repo (the it15 G2 soup landed between its seeds), so check it on the pilot first.
- Log the cross-seed cosine of the LoRA deltas at steps 10, 30, 75 and 150 as a pre-registered diagnostic. It estimates the signal share of the update directly. Expected values: about 0.005 if plan 20 behaves like the ablation; 0.05-0.4 if doubling the batch and 5x the steps help as the model predicts. A value below 0.1 at step 150 means the single-seed candidate is mostly noise.
- Replace arm C (prompt diversity) with a noise-only control: the same run with advantages permuted within each prompt group. It measures the harm of drift with no signal, which no current arm does. Expected measurable effect: A1 minus control on held-out U at S = 16 isolates the reward's contribution at the same MDE90 (2.5-3.6); if the control alone loses several U, the lr and beta are too loose regardless of reward.
- Raise beta from 0.04 to at least 0.2. At 0.04 the anchor's per-token coefficient is 2-3.5 % of the mean advantage and G3 (beta 0) and the ablation (beta 0.04) reached the same update norm. At 0.2 it is 11-18 % at KL 0.02-0.05. Expected measurable effect: KL flattening inside the first 30-50 steps and a falling growth rate of the squared LoRA norm (0.047 per step at lr 3e-5 when unanchored).
- Tighten the KL rules to the levels at which this project has seen harm: warn at 0.02 and stop at 0.05 (not 0.05 and 0.1). Replace the pilot's linear extrapolation from steps 6-10 (it reads 0.055-0.082 against a gate of 0.08 on the ablation's own data and is biased high) with a gate on the squared LoRA norm growth per step, which is nearly noise-free. Expected measurable effect: fewer false re-pilots and a stop that fires before the regime where the ablation lost 6.5 U.
- Drop the 75 always-pass plain rows from the training pool and guard retention on the probe and select split instead. Expected measurable effect: N_eff per step 11.2 to 12.6 (+13 %) and zero-variance groups 30 % to 25 %. Widening the band to [0.25, 0.875] adds another 5 % (13.2); [0.25, 0.75] gives 15.0 but puts 34 % of plain rows below p = 0.5 and raises the claim-given-fail push from -0.028 to -0.049 per episode, so it trades solve signal for caution pressure.
- Keep G = 8 and add prompts, not samples, if more episodes per step are affordable: 32 x 8 gives N_eff 22.4 per step; 16 x 16 gives 23.9 for the same 256 episodes but covers half the prompts. G = 4 loses 14 %. Under the noise model, splitting a fixed episode budget between batch size and steps does not change the final signal-to-noise; total episodes and per-episode signal do.
- Add two cheap train-side endpoints so the run reports a measurable RL result even if held-out U is null: (a) the epoch-1 to epoch-3 change in pass rate on the same prompts (MDE90 about 2.2-2.6 points with 600 feasible prompts; needs the fixed file order the plan already has); (b) N_eff per step and the effective sample size under DAPO weights, logged every step (expected 11 and about 44-56 for 128 episodes).
- Re-run the dynamics check with the run's estimator before pre-registration, as section 12 item 5 already requires, and expect these values on the prescreen bank: pass/fail about 69 % of |A| mass, claim decisions 16 %, twins 15 % at weight 0.44 (29 % at weight 1.0); solve push +0.137 per episode; claim-given-pass push +0.003; claim-given-fail -0.028; twin claim -0.030.
- Treat the DAPO length risk as variance, not bias: the token sum is the correct trajectory gradient. The measurable issue is concentration: 5 of 64 episodes carry 38-43 % of the weight and long failing episodes (6 % of episodes) carry 25 %. If a cap is wanted, cap per-episode token weight at about the 90th percentile (about 3,400 model tokens); expected effect is a higher effective sample size (22 of 64 now), at the cost of a small bias toward short episodes.

## open_questions

- Is the ablation's loss on select (U -6.5) caused by random drift or by its CR-GRPO reward table? The update geometry says noise, but only a permuted-advantage control run can separate the two.
- Does KL saturate (fitted K_inf 0.031, tau 30 steps) or keep growing linearly? Thirty steps cannot tell; the projections for step 150 differ by 3.5x (0.02 against 0.07).
- What is the per-coordinate signal-to-noise under U at 128 episodes per step? The bound of 0.02-0.05 comes from a different reward at 64 episodes and from a comparison diluted by the method arm's report term. Two seeds on the same prompts for 10 pilot steps would measure it.
- Does averaging RL seed deltas keep the signal (is the region linear enough)? Not tested for RL adapters in this repo.
- How concentrated is the solve signal in parameter space? The noise model assumes uniform per-coordinate signal; a low-dimensional signal would be a small share of the norm yet still matter functionally, so the energy share is not a bound on the behavioural effect.
- Where in a trajectory do passing and failing samples of the same prompt diverge? Not analysed. It decides whether token-level credit assignment (for example weighting tokens after the first divergence) could raise per-episode signal.
- True pass rates of prescreen-selected rows at the 16,384-token budget: the bank is at 12,288 tokens and S = 4, and its 4-of-4 prompts passed only 88 % in training, so the always-pass and band shares need the S = 8 prescreen.
- Is the harm from drift proportional to squared displacement, as assumed in the estimate? There is one data point (squared norm 1.15, pass -3.4, U -6.5).
- Step time and memory at 128 episodes per step, and whether 256 episodes per optimizer step fits the one-round-per-step constraint in src/honest_agents/train/grpo.py, were not measured.