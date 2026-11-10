# research:algorithm-feasibility


## question

Which stronger RL algorithm components can be added to this codebase cheaply and safely, and which are traps? (baselines, more signal per step, an exact report-decision term, loss normalisation, trust region, behaviour retention, turn-level credit, instrumentation)


## method

Read in full: src/honest_agents/trainer.py, trl_rewards.py, crgrpo.py, train/grpo.py, train/run.py, train/config.py, the relevant parts of callbacks.py, bank.py and train/confidence_exact.py, and the installed TRL 1.14.1 trainer (.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py: _prepare_inputs, _generate_and_score_completions, _compute_loss; grpo_config.py loss_type docs; utils.py shuffle/split). Docs read: iteration-20-rl-design.md, iteration-19-results-crgrpo.md, badcase-it19c/method_stop.md, iteration-20-research/u_checks.txt.

New computations (all train-side; every source checked against data/split_seeds.json, none is "test"; DEF-66 sources, infra errors and leak groups dropped; repo scoring via eval.records.read_records + r.score()). Scripts and their saved outputs are in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/ ; run each from the repo root with `env -u VIRTUAL_ENV uv run --frozen python <script>`.
- a_baselines.py: analytic second moment of the advantage for group-mean, leave-one-out, cache+LOO blend and oracle baselines at G=8, plus a 400k-group Monte Carlo of a Bernoulli bandit, p = 0.4 to 0.875.
- c_exact_term.py: runs/it19/k2a_status_g2.jsonl (G2's pi(claim) at the status token, 747 rows / 94 sources after DEF-66). Exact term coefficient dU*pi(1-pi) with dU = +1 pass, -1 fail, -0.44 twin, against iteration 19's linear delta (+0.9 / -0.386 / -3V, V = 0.7) with and without its one-sided freeze; bootstrap by source.
- be_logs.py: outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl: wall-clock shares, clip ratio, grad norm, KL growth law and projections.
- e_traj.py + e_null.py: LoRA checkpoints 5, 10, 15, 20, 25, 30 of the it19c ablation (archive staging/outputs/it19/it19-crgrpo-ablation-r2): cosines between 5-step weight increments in (A,B) space and in Delta-W = B@A space, against a simulated Adam null (beta1 0.9, beta2 0.999, the run's lr schedule, pure noise or noise plus a persistent signal of power rho).
- bdgh.py: G2 rollouts: pool bank runs/def74/it17_rollouts_feasible (400 variants x 4) and the select split with three G2 runs pooled into G=8 groups (it19c S=4 + it17 S=2 + it16 S=2; 184 groups). Zero-variance share, band-selection check, Beta-binomial fit, tokens by outcome and loss-type weighting, turn-conditional LOO baseline, revisit power, cosine between arm A and arm B advantages. Training reward = U with twin weight 0.44.
- g_report_tokens.py: 1,600 G2 pool transcripts: share of model-written characters after the status value of the report call.
- h_revisit.py, h_curve.py: per-episode completions parquet of the ablation run (30 steps x 64): same-prompt revisit gain (120 prompts, two visits 15 steps apart, 8 rollouts each) and a G2-baseline-adjusted curve.
Nothing ran on a GPU; no file in the repo or archive was changed (git status clean).


## headline

No cheap estimator trick adds much signal: leave-one-out is the group mean times a constant 8/7, a cached-value control variate cuts advantage variance by at most 6.7 % (12.5 % for a perfect baseline), a turn-conditional baseline by 0 %, and the exact report term is safe (net push 40-60 times smaller than iteration 19's, zero at saturation) but inert, because G2's status token is saturated (15 of 747 rows have pi(claim) between 0.05 and 0.95). The mechanism that matters is noise: the only full-episode GRPO run from G2 was an Adam random walk. Its KL grew as step^1.01 (SE 0.10), its weight-increment cosines sat at or below the pure-noise null (0.066 against 0.130 at lag 4; a persistent signal of 1 % of the noise power would give 0.224), and its own training prompts did not improve on revisit (dU -1.3 [-6.2, +3.6] per 100). On the same fit the current plan (lr 2e-5, 150 steps) reaches KL about 0.047, 2.2-3 times the drift at which that run lost 3.4 points of verified success; TRL's PPO clip never fires (clip ratio 0.0 on all 37 logged steps), and beta = 0.04 caps the KL's pull on a decision token at 0.04 advantage units against a mean |advantage| of 0.17. What to add: a zero-lr gradient signal-to-noise probe and per-step meter, per-episode records with a same-prompt revisit gate (MDE90 2.0-3.5 U per 100 per seed, against about 7.8 for the probe), lr 1e-5 with a KL budget near 0.012 held by an adaptive beta, 256 episodes per step if the hours allow, and optionally zero policy-gradient weight on the 12 % of model text written after the status value.


## results


**1. Hooks TRL 1.14.1 already offers, so none of the components below needs a change to TRL or a new fingerprint. Advantage is reward minus the group nanmean that includes the sample itself. _compute_loss takes advantages of shape (B,) or (B,T), so per-token advantages are native. self.beta is read on every loss call, so it can be changed between optimizer steps. For a LoRA policy the KL reference is the same model with adapters disabled (merged G2). Row-aligned tensors added to the batch dict survive TRL's shuffle and split. HonestGRPOTrainer already overrides _generate_and_score_completions and _compute_loss, and StatusLogits already gives the status logit in fp32 from the training forward.**

- *numbers*: trl 1.14.1; LoRA r 32, 248 modules, 64,929,792 trainable parameters
- *source*: .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 2821, 2841, 3116-3120, 3159-3166, 3210, 972-975, 2778-2779); .venv/lib/python3.12/site-packages/trl/trainer/utils.py (lines 1036-1081); src/honest_agents/trainer.py (lines 154-215, 734-738, 796-830)
- *confidence*: verified

**2. The PPO clip, the gradient clip and the vLLM importance correction are all inert in this trainer geometry. One generation round is one optimizer step (enforced), so the new/old probability ratio is 1 when the loss is computed. Tuning epsilon, epsilon_high or delta changes nothing. The only per-step limiter is the learning rate.**

- *numbers*: clip_ratio/region_mean = 0.0 on all 30 ablation steps and all 7 method-arm steps; grad_norm 0.067-0.229 (mean 0.142) against max_grad_norm 1.0; importance ratio mean 0.9998-1.0002, at-cap share at most 2.1e-5
- *source*: be_logs.py on outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl; src/honest_agents/train/grpo.py (lines 116-123)
- *confidence*: verified

**3. The it19c ablation (plain full-episode GRPO from G2, 64 episodes per step, lr 3e-5, beta 0.04) moved like an Adam random walk, with no detectable persistent direction. Three independent signs: (1) KL grew linearly in steps, not quadratically; (2) cosines between successive 5-step LoRA weight increments are at or below what pure gradient noise gives through Adam's momentum; (3) the run's own training prompts did not improve when revisited 15 steps later.**

- *numbers*: (1) log-log slope of KL on steps 8-30: 1.01 (SE 0.10); random walk predicts 1, coherent drift 2; 0.00072 KL per step. (2) (A,B) space, lags 1-4: observed 0.654 / 0.341 / 0.163 / 0.066; pure-noise null 0.694 / 0.400 / 0.230 / 0.130; with a persistent signal of 0.5 % of noise power 0.714 / 0.439 / 0.278 / 0.179; with 1 %: 0.732 / 0.473 / 0.320 / 0.224. Delta-W space: 0.703 / 0.402 / 0.220 / 0.113. Norm of Delta-W grows as step^0.69. (3) n = 120 prompts x 8 rollouts x 2 visits: dU -1.3 [-6.2, +3.6] per 100, d pass +1.0 [-1.4, +3.3], d claim +2.7 [+0.3, +5.1], d damage +0.9 [-0.7, +2.4] (90 %, bootstrap by source)
- *source*: be_logs.py, e_traj.py, e_null.py, h_revisit.py (outputs beside each script); checkpoints under ~/Documents/honest-agents-archive/staging/outputs/it19/it19-crgrpo-ablation-r2
- *confidence*: derived

**4. KL projection for the current plan under the same noise-walk fit (KL = c * lr^2 * steps). The plan would end at 2.2-3 times the drift of the ablation, the run that lost 3.4 points of verified success. Under this model the drift per step does not depend on batch size (Adam normalises the step), while the signal moved per step scales with sqrt(batch); so the gain-to-drift ratio scales as sqrt(batch)/lr, and the drift paid for a given gain depends only on total episodes times per-episode signal-to-noise.**

- *numbers*: lr 2e-5 x 150 steps: 0.047; lr 1.5e-5: 0.026; lr 1e-5: 0.012; lr 3e-5 x 150: 0.105. Ablation: 0.0213 at step 30, mean of steps 26-30 0.0155, verified success -3.4 [-6.9, +0.1], false claims +3.8 [+0.8, +6.8]. Model indices relative to the plan (128 per step, 2e-5): signal moved 1.00, gain/drift 1.00; 128 at 1e-5: 0.50, 2.0; 256 at 1e-5: 0.71, 2.83; 256 at 2e-5: 1.41, 1.41; the ablation: 0.18, 0.47
- *source*: be_logs.py; docs/iteration-19-results-crgrpo.md (lines 41-43)
- *confidence*: derived

**5. beta = 0.04 is a weak guard for a single decision token. From TRL's per-token KL (exp(ref - cur) - (ref - cur) - 1), the expected coefficient on the status logit is beta * (pi_theta - pi_G2), so at most 0.04 even if the claim decision collapses completely. The policy-gradient coefficient on the same token can reach 0.25. The token-mean KL metric also barely sees it.**

- *numbers*: KL pull at most 0.04 against mean |advantage| 0.168 [0.129, 0.211] at G=8 (184 groups). P(claim | pass) falling from about 1 to 0.9 is 0.83 nats at one token of about 1,300-1,500 model tokens (pass episodes: mean 1,299-1,490), about 6e-4 of token-mean KL, 3-4 % of the 0.0155-0.021 level the ablation reached
- *source*: .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 3161-3163, 3210); bdgh.py
- *confidence*: derived

**6. (a) Baselines. Leave-one-out is exactly the group mean scaled by G/(G-1): same direction, no variance change, only a 1.143 times larger step (it differs slightly only for groups with masked rollouts). A blend of the cached S=8 G2 pass rate with the LOO mean is unbiased but buys little, and becomes worse than LOO once the policy has moved 0.2 on a prompt with p = 0.875.**

- *numbers*: Monte Carlo: LOO variance / (group-mean variance x (8/7)^2) = 1.0000 at p = 0.4, 0.5, 0.625, 0.75, 0.875; group-mean estimate = 0.875 x true gradient. Advantage second moment relative to LOO: optimal blend weight 0.533 on the cache, ratio 0.933 at every p with no drift; 0.936-0.939 at drift 0.05; 0.943-0.956 at 0.1; 0.973-1.024 at 0.2; a perfect baseline 0.875. Upper bound on the gain (one-parameter bandit, score fully correlated with reward): blend/LOO variance 0.584 / 0.340 / 0.659 / 0.884 / 0.965 at p = 0.4 / 0.5 / 0.625 / 0.75 / 0.875; with 10 nuisance score dimensions the p = 0.5 ratio is already about 0.92, with 100 about 0.93
- *source*: a_baselines.py
- *confidence*: derived

**7. (b) Signal per step. Under training U, more than half of unselected prompts give no contrast at G=8. Band selection at the prescreen roughly doubles the reward variance per rollout; always-pass retention rows carry about one seventh of an in-band row's variance. G=16 at a fixed rollout count gains 6.7 % of within-prompt variance and halves the prompts per step. Rollouts dominate wall-clock, so skipping the backward pass of zero-variance rows saves almost nothing and removes their KL term. Larger generation batches need no extra GPU memory: one vLLM call per turn serves all live episodes from a fixed KV pool (vllm_gpu_memory_utilization 0.3) and the loss runs one row per micro-batch.**

- *numbers*: Zero-variance share at G=8: all 0.576 [0.505, 0.642] (184 groups); plain 0.673, control 0.545, twin 0.530. Beta-binomial fit of the pool (400 variants x 4, Beta(0.86, 0.23)): band 4-7 of 8 keeps 31.0 % of variants, E[p(1-p)] 0.159 against 0.087 for all (1.83 times), zero-variance share 0.244 at G=8 and 0.128 at G=16; 8-of-8 rows: E[p(1-p)] 0.022, zero-variance 0.859. Independent check on select (band on 4 rollouts, measured on 4 fresh ones, n = 24 of 118): within-group variance 0.104 against 0.048, fresh pass rate 0.812. LOO factor 1.143 at G=8, 1.067 at G=16. Wall-clock per 64-episode step (mean 386 s): rollout 59 %, grading 14 %, loss forward+backward 17 %, scoring 9 %; zero-variance rows 18.3 % of groups, so skipping them saves at most 3.1 %
- *source*: bdgh.py; be_logs.py; .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 1859-1864)
- *confidence*: derived

**8. (c) Exact expected-gradient term for the report decision. The maths is sound: for a finished episode the status token's gradient is dU * pi * (1 - pi) on the claim logit, with dU = +1 on a pass, -1 on a fail, -w on a twin (damage cancels); it can be written as a per-token advantage (dU * pi(sampled) with sign + for a claim, - for a non-claim) without masking anything. It does not have iteration 19's failure: the push is bounded by 0.25, vanishes at saturation, and its net batch sum is 40-60 times smaller. But it is nearly inert, because G2's status decision is close to deterministic: the gradient lives on 7.5 % of rows, and a sampled policy gradient on the same token is just as small. More than half of failing episodes are confident false claims that no policy-gradient term at the status token can move.**

- *numbers*: 747 rows, 94 sources. Rows with pi between 0.05 and 0.95: 15 (2.0 %); of the 13 feasible ones 23 % passed. |logit| > 6: pass/claim 98.1 % (n = 474), fail/claim 88.1 % (n = 84), twin/non-claim 96.7 % (n = 120). Failing feasible episodes (n = 140): 56.4 % have pi > 0.99, 40.7 % have pi < 0.5. Mean exact coefficient per episode: fail/claim -0.0107 [-0.0191, -0.0033], fail/non-claim -0.0247 [-0.0430, -0.0111], pass/claim +0.0012 [+0.0003, +0.0023], twin/non-claim -0.0006 [-0.0013, -0.0001]. Net per 128-episode step in the iteration-20 mix: exact term -0.29; iteration 19's linear delta -8.0 unfrozen and -12.0 with its one-sided freeze (in iteration 19's own 64-episode mix: -0.125 against -7.7). Top 5 % of rows carry 97.8 % of the exact push; 56 of 747 rows exceed 1e-3. Upper bound on the effect: 4 of 84 false claims are uncertain (4.8 %), about 0.65 false claims per 100 feasible episodes
- *source*: c_exact_term.py on runs/it19/k2a_status_g2.jsonl; docs/iteration-19-research/badcase-it19c/method_stop.md (lines 62-90)
- *confidence*: derived

**9. (d) Loss normalisation. Failing episodes are 1.8-1.9 times longer than passing ones. dapo and dr_grpo sum over tokens, which is the correct estimator for a terminal reward; they differ only by a constant against a per-batch normaliser. grpo and bnpo divide each episode's gradient by its own length (TRL's own docs: with per_device_train_batch_size 1, bnpo equals grpo), which gives each token of a failing episode about half the weight of a token of a passing one, and the repo's micro-batch regrouping is exact only for dapo, cispo and vespo. The feared dominance of long failures under dapo is small.**

- *numbers*: Model tokens, pool: pass 1,490 (n = 1,264), fail 2,778 (n = 336), fail+claim 2,232, fail+non-claim 3,479; select: pass 1,299 (n = 727), fail 2,304 (n = 217). Share of |advantage| mass on failing episodes: token-sum 58.7 % against per-episode 53.4 % (pool); 47.8 % against 41.3 % (select, twins 20.0 % against 25.6 %). Token-weighted advantage imbalance: -0.051 [-0.086, -0.018] (pool, S=4), -0.001 [-0.040, +0.033] (select, G=8); mean advantage per trained token under dapo -0.015 and -0.0002
- *source*: bdgh.py; .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 3214-3235); grpo_config.py loss_type docs; src/honest_agents/trainer.py (lines 325-327)
- *confidence*: derived

**10. (g) Turn-level credit. A turn-conditional leave-one-out baseline (mean reward of the other rollouts still running at that turn; unbiased) gives no variance reduction at G=8, although long episodes do fail more: the relation is mostly between prompts, which the group baseline already absorbs. Credit restricted to late turns is biased and would take the gradient off the early evidence-reading turns. One exact device exists: once the status value is sampled, U cannot depend on the rest of the report except through a parse failure, so those tokens have zero true advantage, yet today they receive the episode's advantage as pure noise.**

- *numbers*: Sum over turns of (u - b_turn)^2 / (u - b_LOO)^2: 1.011 [1.000, 1.023] at shrinkage 1, 1.004 [0.996, 1.013] at 2, 1.000 [0.994, 1.006] at 4; on turns after the 15th (21.2 % of turns) 1.041 / 1.020 / 1.005. Pass rate by model calls: 1-8: 0.930 (n = 230); 9-15: 0.842 (366); 16-22: 0.735 (181); 23+: 0.431 (167). Text after the status value: 12.2 % of all model-written characters, 17.4 % as a per-episode mean (median 603 of 3,604 characters); the whole report turn 21.4 % pooled; status is the first field in 1,414 of 1,582 reports. G2 parse errors: 0 of 2,544 feasible episodes; no usable report 0.30 % on passes (n = 1,991), 3.07 % on fails (n = 553)
- *source*: bdgh.py; g_report_tokens.py; h_curve.py; src/honest_agents/report.py (lines 246-247)
- *confidence*: derived

**11. (h) Measurability. The completions parquet already holds step, prompt id, status, pass, case, n_actions and advantage for every training episode, enough to compute a same-prompt revisit gain offline (done here for the ablation). It lacks damage, family, control flag, twin weight, model-token count and the status-token probability. The revisit gain is far better powered than the in-run probe, and a G2 per-prompt baseline tightens the learning curve.**

- *numbers*: Revisit gain SE: measured on the ablation 3.0 per 100 on 120 prompts, which scales to 1.18 at 800 prompts, MDE90 3.45 for one seed and 2.0 for three; from G2's within-prompt variance of U (0.145 [0.105, 0.188] select, 0.269 [0.233, 0.304] pool) SE 0.67-0.92, MDE90 1.97-2.68, three seeds 1.14-1.55. Probe (103 rows x S=4), paired between two steps: SE about 2.65, MDE90 about 7.8. G2-baseline adjustment of the curve: variance -25 % (only 28 prompts had a baseline)
- *source*: h_revisit.py; h_curve.py; bdgh.py; src/honest_agents/trl_rewards.py (lines 473-516)
- *confidence*: derived

**12. Side result for arm allocation: at the start, arm B (u_success) and arm A (u) give nearly the same gradient.**

- *numbers*: Cosine between the two advantage vectors on G2's G=8 groups: 0.893 [0.857, 0.923]; plain 0.899, control 0.911, twin 0.988; 68.5 % of groups have identical advantages under both rewards
- *source*: bdgh.py
- *confidence*: derived

## design_recommendations


**1. ADOPT, MAIN ARM, FIRST: a gradient signal-to-noise probe before any training, and the same meter in every step. (1) Run 16 steps of 128 episodes at lr 0 on pool v5 (configs/phase0_zero_lr.yaml is the pattern), save each step's flattened LoRA gradient (fp16, 130 MB each), and compute the mean pairwise cosine offline. (2) In ProbeSafeGRPOTrainer.training_step (src/honest_agents/train/run.py, lines 163-170) keep two gradient buckets by micro-batch parity in the length-sorted order and log their cosine, plus the cosine with the previous step's gradient. About 60 lines for the meter, 40 for the dump. Tests: the two buckets sum to the applied gradient; cosine 1 on duplicated halves and about 0 on independent noise; the loss and the update are bit-identical with the meter on.**

- *rationale*: The only comparable run was indistinguishable from a random walk (KL exponent 1.01, increment cosines at or below the noise null, no revisit gain). Every estimator trick examined buys under 13 %. Whether U on pool v5 carries a learnable shared signal at 128 episodes per step is the single unknown that decides the round, and it can be measured in 3-4 GPU-hours without moving the model.
- *expected_measurable_effect*: Mean pairwise gradient cosine across steps = rho/(1+rho), with rho the shared-signal to noise power at 128 episodes. Provisional pre-registered gate from the Adam model: at least 0.027 means signal exceeds noise displacement 2:1 by step 150 (launch as planned); 0.007-0.027 means launch only with 256 per step; below 0.007 means do not launch, change pool or reward. The ablation's bound was rho below 0.005 at 64 episodes.
- *confidence*: medium

**2. ADOPT, MAIN ARM: per-episode records and a same-prompt revisit gate. Add to EXTRA_COLUMNS / episode_columns (src/honest_agents/trl_rewards.py, lines 473-516): u, claim, damage, u_weight, family, is_control, pair_id, model_tokens (sum of tool_mask), status_logprob (sampling log-prob at the status position, via trainer.status_position), epoch, and the row's g2_pass. About 40 lines. Add `analysis.u_dynamics curves` (about 200 lines): per-step U with a source bootstrap, the G2-baseline-adjusted curve, revisit gain epoch 2 minus 1 and 3 minus 1, and an offline replay of the stop rules. Tests: column types for the parquet writer, masked rows, a synthetic run with a known gain.**

- *rationale*: TRL repeats the same prompt order each epoch, so every prompt is rolled out 8 times at steps k, k+50, k+100. That paired design is 2-4 times more sensitive than the probe and costs nothing. The existing parquet lacks damage, weights and token counts, so U cannot be recomputed exactly offline today.
- *expected_measurable_effect*: Revisit gain in U with MDE90 2.0-3.5 per 100 for one seed and 1.1-2.0 for three seeds pooled, against about 7.8 for the 103-row probe at S=4. Suggested rule: if the pooled epoch-2 minus epoch-1 gain has a 90 % upper bound below +2, stop; training is not improving even its own prompts. It is a training-set gain, so an upper bound on generalisation.
- *confidence*: high

**3. ADOPT, MAIN ARM: lr 1e-5 (not 2e-5), and re-anchor the KL rules to evidence: warn at 0.012, and hold the budget with an adaptive beta instead of stopping. A callback on_step_end sets trainer.beta = clip(beta * exp(0.5 * (kl_ema / K - 1)), 0.04, 2.0) with K = 0.012; beta is read live by TRL, must stay above 0, and is appended per step to a JSONL for resume, as crgrpo_state.jsonl is. About 60 lines in src/honest_agents/callbacks.py plus wiring in train/run.py. Tests: controller arithmetic, resume, and that the next loss call uses the new beta. Leave epsilon, epsilon_high, delta and max_grad_norm alone: they never bind.**

- *rationale*: Under the fitted noise walk the plan ends at KL 0.047, 2.2-3 times the level at which the ablation lost 3.4 points, and gain-to-drift scales as sqrt(batch)/lr. A fixed beta of 0.04 did not bend the ablation's linear KL growth. Within 150 steps a lower lr reaches the same budget as a controller with as much signal, so the controller is a cap that keeps a run alive, not a source of gain.
- *expected_measurable_effect*: Token-mean KL at step 150 at or below 0.012 (plan: about 0.047). Gain-to-drift index 2.0 against the plan's 1.0, with half the signal moved; guard G1 (task success) lower bound expected to stay above -3. Check: the KL exponent on steps 10-150 from train_log.jsonl; near 1 means still noise-driven, near 2 means coherent learning.
- *confidence*: medium

**4. ADOPT IF THE HOURS ALLOW (decide from the probe): 32 prompts x 8 generations = 256 episodes per step. Config only: gradient_accumulation_steps 256, per_device_train_batch_size 1. No extra GPU memory. Needs 256 sandboxes (about 128 vCPU, address pools sized for 256) and about twice the step time (about 26 minutes per step extrapolated from 386 s per 64 episodes; 150 steps about 64 h). Keep G = 8.**

- *rationale*: Total episodes times per-episode signal-to-noise is the only quantity that lowers the drift paid per unit of learning; batch size, lr and step count otherwise move along the same curve. More prompts per step also halves between-prompt noise, which G = 16 does not.
- *expected_measurable_effect*: Split-half cosine about doubles; signal moved per step x1.41 at unchanged KL per step; gain-to-drift index 2.83 with lr 1e-5 against 1.0 for the plan. Zero-variance group share is unchanged and is not the target.
- *confidence*: medium

**5. SEPARATE ARM (or main arm with the owner's sign-off on the no-mask rule): per-token advantages that give zero policy-gradient weight to report tokens after the status value. The tokens stay in the loss mask, so they keep the KL term and count in the dapo normaliser. Build the (B,T) advantage tensor in HonestGRPOTrainer._generate_and_score_completions (src/honest_agents/trainer.py, lines 734-738), reusing status_position (lines 91-132) without touching tool_mask. About 50 lines. Tests against the real TRL _compute_loss: identical loss when no status token is found; gradient at those tokens equals the KL gradient only; mask unchanged.**

- *rationale*: Exact, not a heuristic: claims_complete is status == complete, so once the status is sampled the reward cannot depend on later report tokens except through a parse failure (0 of 2,544 G2 episodes). Today those tokens get the episode advantage as pure noise, and under plain U a malformed report on a failed episode scores 0 instead of -1, a channel this closes.
- *expected_measurable_effect*: Policy-gradient noise removed from 12.2 % of model-written text (17.4 % per episode). Share of reports with status first stays within 5 points of G2's 89 %; no-report rate on failing episodes stays at or below G2's 3.1 %. Split-half cosine up by at most about 6 % relative (unverified: assumes equal gradient power per token).
- *confidence*: medium

**6. BUILD, KEEP OFF UNLESS THE PILOT MOVES (insurance): a decision-token KL anchor. Add lambda * KL(Bernoulli(pi_G2) || Bernoulli(pi_theta)) at each row's status token, with the policy logit from StatusLogits (trainer.py, lines 154-215) and G2's probability from inputs['ref_per_token_logps'] at the same position; no extra forward pass, the status token stays in the main loss, lambda 0.5-1. About 90 lines. Tests as tests/test_crgrpo_report_term.py: zero loss and zero gradient at G2, gradient lambda * (pi_theta - pi_G2), hook reaches every found row.**

- *rationale*: beta 0.04 pulls a decision token back with at most 0.04 advantage units and the token-mean KL barely registers a 10-point fall of P(claim | pass). The anchor has zero gradient at G2, so it adds no supervised confound to the RL-versus-SFT claim. It is insurance: the plain-GRPO ablation did not drift on claims (+0.4 [-2.3, +3.1]).
- *expected_measurable_effect*: P(claim | pass) on training rollouts within 1 point of the G2 baseline; stop rule 1 does not fire. No expected gain in U.
- *confidence*: medium

**7. REJECT for the main arm, and not worth a GPU arm: the exact expected-gradient report term. If the owner wants it anyway, it is a per-token advantage at the status position (dU * pi(sampled), sign by claim), about 40 lines on top of the per-token path, replacing the sampled term rather than adding to it. Do log the status-token probability per episode (included in the records above).**

- *rationale*: Safe but inert. G2's status token is saturated: 2.0 % of rows are uncertain, the net push is -0.29 per 128-episode step against -12.0 for iteration 19's term, and 56.4 % of failing episodes are confident false claims no policy gradient at this token can move. Under plain U, false claims can fall only through solving more or through actions that change what the status token sees.
- *expected_measurable_effect*: At most 0.65 fewer false claims per 100 feasible episodes if every uncertain false claim flipped; expected effect near zero. No collapse risk.
- *confidence*: high

**8. REJECT: leave-one-out baseline and the cached-value control variate.**

- *rationale*: LOO is the group mean times 8/7, the same as raising lr by 14 %. The blend with the S=8 prescreen rate cuts the advantage second moment by 6.7 % at best, less as the policy moves, and needs rewards stashed out of the reward function into an advantage rewrite (about 40 lines) for that.
- *expected_measurable_effect*: LOO: none. Blend: split-half cosine up by at most about 7 % relative; worse than LOO once a prompt's pass rate has moved 0.2 from G2's at p = 0.875.
- *confidence*: high

**9. REJECT: G = 16 at a fixed rollout budget; skipping zero-variance groups before the backward pass; num_iterations above 1; tuning the PPO clip.**

- *rationale*: G = 16 gains 6.7 % of within-prompt variance and halves the prompts. Skipping zero-variance rows saves at most 3.1 % of wall-clock and removes the KL term from exactly the rows that anchor easy-task behaviour. A second pass over the same batch repeats the same noise and breaks resume order (refused by check_batch_geometry). The clip never fires.
- *expected_measurable_effect*: None positive; skipping would lower the KL coverage of always-pass rows to zero.
- *confidence*: high

**10. KEEP dapo with scale_rewards none; REJECT grpo, bnpo and dr_grpo.**

- *rationale*: Token-sum is the correct estimator for a terminal reward. grpo and bnpo are the same loss at micro-batch 1 and halve the per-token weight of failing episodes (1.8-1.9 times longer). dr_grpo equals dapo up to a constant. The common-mode imbalance under dapo is small (-0.051 to -0.001).
- *expected_measurable_effect*: No change. Monitor mean advantage per trained token (expected between -0.015 and 0) and actions per episode.
- *confidence*: high

**11. REJECT: turn-level credit heuristics (advantage only after the last environment-changing command; turn-conditional baselines) and SFT replay of G2's decision data inside the RL run. Defer a KL anchor on a fixed bank of G2 decision states (about 120 lines and extra forward passes) until on-policy anchoring is shown insufficient.**

- *rationale*: Late-turn-only credit is biased and removes the gradient from the early evidence-reading turns, the behaviour twins are meant to teach. The unbiased turn-conditional baseline measured 0 % gain. SFT replay has a non-zero gradient at G2, so any gain could be more SFT rather than RL, which breaks the attribution the core claim needs, and it needs a second data path in the trainer (150-200 lines).
- *expected_measurable_effect*: None positive.
- *confidence*: medium

**12. Consider re-purposing arm B's GPU (owner's call, it is the pre-registered claim-term contrast): give it to the 256-episode main run or to the post-status zero-advantage arm.**

- *rationale*: Arm B's advantages are 89 % aligned with arm A's on G2's rollouts and identical on 68.5 % of groups, so a single-seed A minus B contrast is unlikely to resolve anything at an MDE of 2.5-3.6.
- *expected_measurable_effect*: Either twice the episodes behind the primary endpoint, or one clean test of an algorithmic component.
- *confidence*: low

## caveats

- The random-walk evidence comes from one run (it19c ablation: CR-GRPO payoff table, not U; 64 episodes per step; lr 3e-5; 30 steps). The noise null assumes independent, isotropic gradient noise across steps and Adam beta1 0.9; signal specific to one prompt counts as noise in it. Observed cosines are slightly below the null, which the model does not explain.
- The KL projections (KL proportional to lr^2 x steps), the gain-to-drift index (sqrt(batch)/lr), the cosine gate thresholds and the idea that capability loss scales with KL are model derivations fitted to that one run. None was verified by a training run. The -3.4 point loss itself has a 90 % interval of [-6.9, +0.1].
- Status-token probabilities are from K2a's scoring of bank transcripts with a canonical report prefix (747 rows, 94 sources), not from the training forward of a live rollout. The binary claim / non-claim approximation ignores other tokens' mass.
- The G=8 groups are three G2 select runs pooled (it19c S=4, it17 S=2, it16 S=2). They may differ in commit and were run at the archived 12,288-token budget, not the planned 16,384. Pool-bank figures are S=4.
- All training-reward computations use a fixed twin weight of 0.44; the real per-kind weights come from the prescreen. The select-split prompts are not band-selected, so zero-variance shares there overstate pool v5's.
- The baseline variance figures are for the advantage second moment (the limit of many score dimensions). The one-parameter bandit shows larger gains (down to 0.34 of LOO variance at p = 0.5); a real trajectory sits near the second-moment figure but this was not measured on gradients.
- The share of text after the status value is in characters, not tokens. The 'exact zero advantage' claim holds up to report parse failures (0 of 2,544 in G2) and treats evidence_commands as evaluation-only, as the code comment states.
- Revisit gains are gains on training prompts and include memorisation. The baseline-adjusted curve used only 28 prompts. The probe MDE of about 7.8 uses training-U variance and is a lower bound for evaluation U.
- Line counts and wall-clock figures for new code are estimates; no code was written or tested, and nothing ran on a GPU. The 256-episode step time is a linear extrapolation from 64-episode steps.
- The arm A / arm B cosine is computed on G2's rollouts at the start of training; it says nothing about how the arms diverge later.

## artifacts

- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/load.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/a_baselines.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/a_baselines.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/c_exact_term.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/c_exact_term.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/be_logs.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/be_logs.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/e_traj.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/e_traj.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/e_null.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/e_null.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/bdgh.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/bdgh.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/g_report_tokens.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/g_report_tokens.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/h_revisit.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/h_revisit.out.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/h_curve.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rlalgo/h_curve.out.txt