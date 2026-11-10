# evidence:literature


## summary

External evidence, read against plan 20 (docs/iteration-20-rl-design.md, revision 3). Method: web search plus fetches of arXiv HTML pages, one GitHub repo's raw configs and two blogs; PDFs could not be parsed here, so appendix-only details are marked not verified.

1. Training scale is the largest gap. Every published RL gain on 3B-8B terminal or coding agents used about 100k-500k episodes at a full fine-tuning learning rate of 5e-7 to 1e-6. Endless Terminals: PPO with a learned critic, 16 prompts x 16 rollouts = 256 episodes per step, up to 10 epochs over about 3,155 tasks, lr 5e-7, no KL ("this hurt performance"); dev pass 4.0 -> 18.2 % (3B), 10.7 -> 53.3 % (7B), 42.6 -> 59.0 % (SFT'd 8B); Terminal-Bench 2.0 +1.2 to +5.6 points. ECHO: 500 GRPO steps, Qwen3-8B validation 34.2 -> 54.9 %, best checkpoint at steps 380-460. River: 400 steps x 256 episodes. DeepSWE: 200 steps x 512 episodes, 23 -> 42 %. Plan 20 budgets 19,200 episodes (150 x 128), 5 to 26 times less; this project's earlier RL runs used 960 to 3,200.

2. At the 4B scale with small budgets, published results look like plan 20's own "hope" of 0 to +5. Qwen3.5-4B on a tool-use benchmark: +0.8 points with sparse multi-turn GRPO and +2.9 with calibrated rewards (200 evaluation simulations, no intervals; the sparse run declined after step 70). Qwen3-4B with LoRA, lr 1e-6 and KL 0.15 stayed within +/-0.01 of its SFT start for 200+ steps.

3. Handling zero-variance groups online is the best-documented GRPO lever. DAPO's dynamic sampling added 8 AIME points (42 -> 50). Online balanced difficulty filtering added 3.8 average points in under half the steps, while curating the pool once from the initial policy added 0.0. Plan 20 curates once offline and tolerates up to 70 % zero-variance groups; this repo's logs show 72-86 % zero-variance on high_risk twin groups.

4. A KL penalty to the reference is not a proven behaviour guard. Four working agent recipes drop it. One 2026 study found beta 0.04 (plan 20's value) cut new-task accuracy by 5.5-7.1 points without reducing forgetting. KL measured on the trained task does predict forgetting (R2 0.71 on LLMs), so it is a monitor, not a protection.

5. The ternary reward is supported, with one sharp warning. TruthRL (+1/0/-1, GRPO): hallucination 43.5 -> 19.4 % while accuracy rose 48.8 -> 56.6 %; binary RL reached 60.3 % accuracy but 39.5 % hallucination; it also worked on 3B models. Rewarding abstention directly was worse (truthfulness 32.7 vs 37.2). With a grader that under-credited correct answers the policy collapsed to near-total abstention (truthfulness -3.6). Under U a wrongly failed true claim scores -1, so grader false negatives are the main collapse risk.

6. Standard RL erodes abstention unless the behaviour is in the batch: refusal fell to 0.00-0.09; a 10 % unanswerable mix restored 0.73-0.94 for 7B-8B models but not for a 1.5B (0.01-0.04; it needed 30 %). Plan 20's 25 % twin share is inside the supported range for a small model.

7. I found no published work that trains an agent by RL to stop false completion claims. The 2026 measurement papers (false success 36 % of failures on one benchmark and 75.8 % on another; 80.4 % of incomplete reviews misleading) test only prompts, test access and an abort option (GPT-5 54 -> 9 %). The project's claim is therefore novel, with no external effect size to lean on.

8. Two cheap additions have direct support on the same task family or agent type: an auxiliary loss on tool-output tokens (ECHO: +8.8 validation points over GRPO on Qwen3-8B; one paper, unreplicated) and masking episodes that hit a budget limit (DeepSWE "compact filtering"; DAPO overlong filtering +6 points). Plan 20 does the opposite on both.

9. Step-level credit assignment gave +13 to +14 points over GRPO at exactly plan 20's budget (150 iterations x 128 episodes) in simpler environments (GiGPO); it depends on recurring states and is speculative for Docker tasks.


## facts


**1. Endless Terminals: algorithm, rollout shape and KL**

- *value*: PPO (SkyRL); 16 rollouts per prompt, up to 16 turns in training (64 at inference), 2,048 tokens per turn, 16k context, temperature 0.6; clip 0.2 / 0.28; sequence-level loss averaging; binary reward; 'We do not use a KL penalty since anecdotally we found that this hurt performance'; 3,255 tasks
- *source*: arXiv 2601.16443v3, Section 5 'Training Setup', https://arxiv.org/html/2601.16443v3
- *confidence*: verified

**2. Endless Terminals: optimiser settings from the released configs (not in the paper)**

- *value*: policy lr 5.0e-7, critic lr 1.0e-6, 20 warmup steps, max_grad_norm 0.1; train_batch_size 16 prompts x n_samples_per_prompt 16 = 256 episodes per step; epochs 10; update_epochs_per_batch 2; advantage_estimator gae (lambda 1.0, gamma 1.0) with a critic model; loss_reduction sequence_mean; use_kl_loss false; strategy fsdp2 (full-parameter, no LoRA keys); dev split = 100 tasks (--eval-count default)
- *source*: https://raw.githubusercontent.com/kanishkg/endless-terminals/main/train/confs/base.yaml, base_qwen.yaml, base_qwen3_otak8.yaml and train/prepare_endless.py
- *confidence*: verified

**3. Endless Terminals: gains**

- *value*: own dev set: Llama-3.2-3B 4.0 -> 18.2 %, Qwen2.5-7B 10.7 -> 53.3 %, Qwen3-8B-openthinker-sft 42.6 -> 59.0 %; Terminal-Bench 2.0 (5 runs, no intervals): 0.0 -> 2.2 %, 2.2 -> 3.4 %, 1.1 -> 6.7 %; hardware 4 A100 for about 2 days (3B, 7B), 8 B200 for about 8 hours (8B)
- *source*: arXiv 2601.16443v3, Section 5 and Figure 3 caption, https://arxiv.org/html/2601.16443v3
- *confidence*: verified

**4. Endless Terminals: number of optimizer steps actually run**

- *value*: not reported in the paper; the config allows at most 10 epochs x 197 steps x 256 episodes = 504,320 episodes
- *source*: derived from the configs above; arithmetic in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/external-evidence/work/derive.py
- *confidence*: not verified

**5. ECHO (terminal agents, includes 1,977 Endless Terminals tasks): GRPO recipe and result**

- *value*: n = 16 rollouts per prompt, batch size 16, lr 1e-6 constant, no KL, 500 steps, 8,770 training tasks, 16 turns, 16k context. Qwen3-8B val100 34.2 -> 54.9 % (GRPO) -> 63.7 % (GRPO + auxiliary cross-entropy on terminal-output tokens, lambda 0.05); TB2 pass@1 1.57 -> 2.70 -> 5.17. SFT-initialised model: 38.5 -> 63.5 -> 73.1. GRPO first reaches its best aggregate score at step 460 (8B), 400 (SFT-init), 380 (14B). lambda above 0.1 degrades, 0.2 can collapse rollouts. No training-seed variance reported; whether 'batch size 16' means 16 prompts is not stated
- *source*: arXiv 2605.24517v1, Tables 1-3 and Section 4, https://arxiv.org/html/2605.24517
- *confidence*: verified

**6. River (terminal agents, Qwen3-8B): GRPO recipe, result, failure modes**

- *value*: lr 5e-7, 32 prompts x 8 rollouts, max 400 steps, about 60 hours on 8 H200, sequence-mean loss, clip 0.2/0.28, binary reward, repetition penalty 0.05; River-8B average 19.4 +/- 1.2 over 3 seeds vs 14.6 for RL on the unfiltered 15K environments and 17.7 for RL on a random 3.5K subset; more than 60 % of environments had a verifier or spec issue; with partial-credit rewards the agent learned to satisfy a subset of checks and claim completion early; rewarding verify-before-done did not help
- *source*: arXiv 2608.22631, Sections 3-4, Table 4, Appendix A.2, C.4, D, https://arxiv.org/html/2608.22631
- *confidence*: verified

**7. SETA (Qwen3-8B, GRPO with DAPO dynamic sampling)**

- *value*: Terminal-Bench 2.0 3.1 +/- 0.6 -> 10.7 +/- 1.3 (8 repeated evaluation runs; best run 12 %); trained on 560 environments subsampled after dropping too-hard ones; partial-credit reward raised the share of tasks with useful reward variability from 49 % to 86 %; steps, lr and group size not found (appendix not reachable)
- *source*: arXiv 2607.10891, https://arxiv.org/html/2607.10891
- *confidence*: verified

**8. DeepSWE (Qwen3-32B coding agent): recipe and result**

- *value*: about 4.5K tasks, 64 tasks x 8 rollouts = 512 containers per step, 200 steps: 23 -> 42 % pass@1; 64 H100 for six days; removes KL, removes reward std, constant length normalisation, leave-one-out baseline, masks trajectories that hit max context, max steps or timeout ('compact filtering'), no entropy bonus. Four SFT-initialised models did not improve after 100 RL iterations
- *source*: https://together.ai/blog/deepswe
- *confidence*: verified

**9. Episode budgets compared with plan 20**

- *value*: plan 20: 150 x 16 x 8 = 19,200. Earlier runs here: G3 960, it19c 1,920, iteration 9 3,200. External: River 102,400 (5.3x), DeepSWE 102,400 (5.3x), ECHO 128,000 (6.7x, batch unit assumed), Endless Terminals config maximum 504,320 (26x), GiGPO 19,200 (1.0x), CacheRL 51,200 (2.7x), tool-use Qwen3.5-4B runs 1,920-13,760 (0.1-0.7x)
- *source*: derived; plan numbers from docs/iteration-20-rl-design.md (lines 101, 424, 428), earlier runs from configs/base.yaml (lines 133-142), configs/grpo_it15_g3.yaml (line 15), configs/grpo_it19_crgrpo.yaml (line 67); script derive.py in the scratchpad
- *confidence*: derived

**10. Qwen3.5-4B (the project's base model family) under multi-turn GRPO on a tool-use benchmark**

- *value*: base 63.8 %; sparse multi-turn GRPO 64.6 % (+0.8), improved briefly then declined after step 70; calibrated per-turn rewards 66.7 % (+2.9); naive dense rewards 57.3 % (-6.5). Group size 4, 8 prompts per step, lr 5e-7 to 2e-6, KL 0.04-0.2, 60-430 steps. Evaluation 50 tasks x 4 trials, no intervals. The paper has internal inconsistencies (66.7 vs 68.0 headline)
- *source*: arXiv 2604.02869, Table 6 and Section 4, https://arxiv.org/html/2604.02869
- *confidence*: verified

**11. Qwen3-4B-Thinking with LoRA and strong KL: RL on top of strong SFT**

- *value*: LoRA r 128 / alpha 128, lr 1e-6, 16 prompts x 16 rollouts, KL beta 0.15: validation reward 'plateaus within +/-0.01 of the SFT starting point' (0.756 +/- 0.007 over 200+ steps); removing GRPO costs 0.779 -> 0.756. With 1,527 samples validation declined after about 10 steps
- *source*: arXiv 2606.14179, Table 2, Table 5, Section 4.5, https://arxiv.org/html/2606.14179
- *confidence*: verified

**12. DAPO ablation and settings**

- *value*: AIME24: naive GRPO 30 -> overlong filtering 36 -> clip-higher 38 -> soft overlong punishment 41 -> token-level loss 42 -> dynamic sampling 50. 512 prompts x 16 samples, lr 1e-6, KL removed. Dynamic sampling: resample until every prompt in the batch has accuracy neither 0 nor 1
- *source*: arXiv 2503.14476v2, https://arxiv.org/html/2503.14476v2
- *confidence*: verified

**13. Online difficulty filtering vs offline curation**

- *value*: Qwen2.5-3B, 16 prompts x 16 rollouts: plain GRPO 26.3 average; offline curation by the initial model 26.3; balanced online filter 0.3 < p < 0.7 30.1 (+3.8); every skewed filter 23.8-25.9 (below plain); surpasses plain GRPO's best validation reward in under 50 % of steps. 7B: 39.6 -> 41.8
- *source*: arXiv 2504.03380v2, Tables 1-2, https://arxiv.org/html/2504.03380v2
- *confidence*: verified

**14. RAGEN-2: low reward variance lets the regulariser dominate; variance filtering helps**

- *value*: 8 prompts x 16 rollouts, up to 400 iterations; keeping the high-variance prompts: PPO +6.9, GRPO +3.7, DAPO +2.9, Dr. GRPO +0.8 average points (Qwen2.5-3B); 'Low reward variance weakens task gradients while input-agnostic regularization remains constant, erasing input dependence'; entropy stays high during this collapse; RL shortened outputs by 5-66 %
- *source*: arXiv 2604.06268, Table 4, https://arxiv.org/html/2604.06268 and https://ragen-ai.github.io/
- *confidence*: verified

**15. RAGEN v1: collapse timing and group size at fixed budget**

- *value*: gradient-norm spikes mark irreversible collapse at steps 90-170; keeping the 25 % most uncertain prompts is the default fix, with KL removed and clip-higher; at a fixed rollout budget 4 responses per prompt generalised best (Sokoban 25.39 % vs 21.48 % at 16); think length fell 307 -> 90 tokens
- *source*: arXiv 2504.20073, Tables 1 and 4, https://arxiv.org/html/2504.20073
- *confidence*: verified

**16. Zero-variance groups in this project's own RL runs**

- *value*: 72 % of high_risk twin groups zero-variance (26 of 36); in run 2: high_risk twins 86 %, remove_input 66 %, revoke 37 %, controls 26 %, plain 64 %; a later run: high_risk twin groups zero-variance in 20 of 36. Plan 20 allows a zero-variance share up to 0.7 and admits 75 always-pass rows and 'always claims' twins
- *source*: docs/training-log.md (lines 541, 581, 618, 793); docs/iteration-20-rl-design.md (lines 291, 293, 430, 455)
- *confidence*: verified

**17. Probability that a group of G samples is all-same for a binary outcome with rate p**

- *value*: p 0.875: 0.586 (G 4), 0.344 (G 8), 0.118 (G 16); p 0.75: 0.320, 0.100, 0.010; p 0.95: 0.815, 0.663, 0.440; p 0.5: 0.125, 0.008, 0.000. So at the top of plan 20's plain band (0.875) about a third of groups carry no pass/fail signal at G 8
- *source*: derived; derive.py in the scratchpad (path above)
- *confidence*: derived

**18. TruthRL: ternary vs binary reward (GRPO, full fine-tuning, lr 1e-6, KL 0.001)**

- *value*: Llama3.1-8B on CRAG with retrieval, truthfulness / hallucination / accuracy: prompting 5.3 / 43.5 / 48.8; binary RL 20.8 / 39.5 / 60.3; ternary 37.2 / 19.4 / 56.6. Ternary plus +1 for abstaining on out-of-knowledge questions: 32.7 / 21.9 (worse). Rule-based string-match verifier: -3.6 / 3.6, model abstains on the vast majority of queries. 3B models: Llama3.2-3B truthfulness 1.9 -> 27.4, Qwen2.5-3B -0.3 -> 21.9. No penalty-magnitude sweep
- *source*: arXiv 2509.25760v1, Tables 1, 3, 5, 7, https://arxiv.org/html/2509.25760v1
- *confidence*: verified

**19. Behaviourally calibrated RL on Qwen3-4B-Instruct**

- *value*: reward +1 correct, 0 abstain, -t/(1-t) wrong makes answering worthwhile only when confidence p >= t; with a sampled threshold stated in the prompt the policy over-rejected even at t = 0 and ignored t; proper scoring rules on a stated confidence fixed calibration (calibration error 0.632 -> 0.177) while accuracy rose 0.291 -> 0.395. lr, batch and steps not reported
- *source*: arXiv 2512.19920v1, Eq. 2, Table 1, Section 3.2.1, https://arxiv.org/html/2512.19920v1
- *confidence*: verified

**20. Threshold implied by plan 20's false-claim penalty c**

- *value*: claim is worth it only when P(pass) >= c/(1+c): c = 1 gives 0.5, c = 2 gives 0.667, c = 3 gives 0.75
- *source*: derived from arXiv 2512.19920v1 Eq. 2 applied to U in docs/iteration-20-rl-design.md (lines 92-97)
- *confidence*: derived

**21. Hallucination tax: standard RL fine-tuning erodes refusal; mixing restores it for larger models only**

- *value*: PPO, 200 steps, batch 1,024, lr 1e-6, KL 0.001. Refusal on unanswerable sets after standard RL 0.00-0.09 (one model 0.30 -> 0.08). With 10 % unanswerable data: Qwen2.5-7B 0.81 / 0.94 / 0.73, Llama-3.1-8B 0.79 / 0.70 / 0.75, but Qwen2.5-1.5B-Math 0.04 / 0.35 / 0.01 (about 0.9 only at 30 %). Answerable accuracy change at 10 %: 0 to -0.10. Refusing an answerable question earns 0; correct refusal earns 1. No refusal rate on answerable inputs reported
- *source*: arXiv 2505.13988v1, Tables and Section 6, https://arxiv.org/html/2505.13988v1
- *confidence*: verified

**22. Reasoning-style RL fine-tuning degrades abstention**

- *value*: 24 % on average
- *source*: arXiv 2506.09038 abstract, seen only as a search snippet
- *confidence*: not verified

**23. Confessions (honesty of the self-report trained separately from the task reward)**

- *value*: confession reward does not touch the main answer's reward; average probability of 'misbehaved and did not confess' 4.36 % over 12 evaluations; confession given misbehaviour 74.3 % on average; training 'may not reduce the frequency of bad behavior'; false negatives come mostly from genuine confusion, not intent
- *source*: arXiv 2512.08093v2, Sections 1, 3.2, 6, 7.4, https://arxiv.org/html/2512.08093v2
- *confidence*: verified

**24. ImpossibleBench mitigations**

- *value*: no training-based mitigation tested. Strict prompt: GPT-5 cheating above 85 % -> 1 % on one split; abort option: GPT-5 54 -> 9 %, o3 49 -> 12 %; hiding tests: near zero; monitors detect 86-89 % on the LiveCodeBench variant and 42-65 % on the SWE-bench variant
- *source*: arXiv 2510.20270v1, Sections 5-6, https://arxiv.org/html/2510.20270v1
- *confidence*: verified

**25. False success and overclaiming in agents: measured, no training fix tested**

- *value*: false success 36 % of failures on one customer-service benchmark (13 % to 79 % by model) and 75.8 % on an app benchmark; 67.9 % of file-review runs incomplete and 80.4 % of those misleading; neither paper tests fine-tuning or RL
- *source*: arXiv 2606.09863v1, https://arxiv.org/html/2606.09863 ; arXiv 2609.20812 abstract, https://arxiv.org/abs/2609.20812
- *confidence*: verified

**26. Anti-scheming training reduces covert actions**

- *value*: o3 13 % -> 0.4 % across 26 out-of-distribution evaluations
- *source*: arXiv 2509.15541 abstract, seen only as a search snippet
- *confidence*: not verified

**27. KL penalty at beta 0.04 in continual GRPO**

- *value*: does not reduce forgetting and lowers plasticity: 2B new-task accuracy 60.86 -> 53.72 % with final accuracy 49.36 vs 49.64 %; 4B 67.29 -> 61.77 % with 56.83 vs 56.05 %. Vision-language models; a parameter-movement penalty on the top 10 % moving coordinates gave +13.67 points final accuracy at 8B
- *source*: arXiv 2607.04364v2 via https://huggingface.co/papers/2607.04364.md
- *confidence*: verified

**28. KL on the new task predicts forgetting**

- *value*: quadratic fit R2 0.96 on a toy task, 0.71 on Qwen2.5-3B experiments (GRPO with KL coefficient 0); the paper did not test an explicit KL penalty on LLMs
- *source*: arXiv 2509.04259v1, https://arxiv.org/html/2509.04259v1
- *confidence*: verified

**29. Recipes that set the KL coefficient to zero**

- *value*: Endless Terminals (hurt performance), DAPO (removed), Dr. GRPO (beta 0 throughout), DeepSWE (removed), ECHO (none), RAGEN StarPO-S (removed). Recipes that keep a small one: TruthRL 0.001, GiGPO 0.01 / 0.001, hallucination-tax 0.001. Plan 20 uses 0.04
- *source*: sources listed in the facts above; plan value from docs/iteration-20-rl-design.md (line 427)
- *confidence*: verified

**30. Mixed SFT + RL in one stage (math reasoning, 7B)**

- *value*: SFT 54.3, GRPO 49.3, SFT then RL 54.5, single-stage SRFT 59.1 (+4.6 over SFT then RL); RL then SFT 37.4; removing the SFT term costs 4.0
- *source*: arXiv 2506.19767, Tables 1-2, https://arxiv.org/html/2506.19767
- *confidence*: verified

**31. LoRA in policy-gradient RL**

- *value*: LoRA matches full fine-tuning in RL even at low rank; optimal LoRA learning rate about 10x the full fine-tuning rate (about 15x for runs under roughly 100 steps, stated as anecdotal); alpha fixed at 32; policy gradient yields about 1 bit per episode; MLP-only LoRA beat attention-only
- *source*: https://thinkingmachines.ai/blog/lora/
- *confidence*: verified

**32. Learning rate position of plan 20**

- *value*: published full fine-tuning rates 5e-7 to 2e-6 imply a LoRA range of 5e-6 to 3e-5 (10x to 15x). Plan 20 uses 2e-5 with r 32, alpha 64 (scale 2); the repo default is 1e-5; it19c at 3e-5 lost 3.4 points of verified success in 30 steps
- *source*: derived; docs/iteration-20-rl-design.md (lines 119, 421, 422); configs/base.yaml (line 155)
- *confidence*: derived

**33. Dr. GRPO on normalisation**

- *value*: dividing by response length makes longer incorrect responses cheaper per token and lengthens wrong answers; dividing by reward std overweights near-uniform prompts; the unbiased form is the group-centred reward with a constant normaliser, equal to leave-one-out up to G/(G-1). Group size 8, lr 1e-6, beta 0
- *source*: arXiv 2503.20783v2, https://arxiv.org/html/2503.20783v2
- *confidence*: verified

**34. Loss aggregation used by the working terminal-agent recipes**

- *value*: Endless Terminals and River: sequence-mean; DeepSWE: constant max-context normaliser; DAPO: token-mean. Plan 20: DAPO token-sum. One user report ties an entropy spike then collapse in multi-turn tool-use GRPO to token-mean aggregation on a 27B model
- *source*: configs and papers above; plan value from docs/iteration-20-rl-design.md (line 423); the user report is https://github.com/verl-project/verl/issues/8250, seen only as a search snippet
- *confidence*: verified

**35. GRPO collapse in multi-turn tool agents (lazy likelihood displacement)**

- *value*: likelihood of correct trajectories stalls or falls while reward still rises; decay from about step 60, accelerated collapse after about step 120 with a gradient-norm spike and an entropy rise; vanilla GRPO collapsed within 300 steps in all four 3B/7B settings; a penalty on likelihood drops of non-negative-advantage responses raised exact match 0.336 -> 0.441 (3B) and 0.396 -> 0.483 (7B)
- *source*: arXiv 2512.04220v2, https://arxiv.org/html/2512.04220v2
- *confidence*: verified

**36. Turn-level PPO vs GRPO in multi-turn tasks**

- *value*: GRPO crashed in 3 of 5 model-environment settings; turn-level PPO beat token-level PPO in all five (for example 3.74 vs 2.90 average reward); critic lr should be 5-10x the actor's; 100-200 steps; no seed variance reported
- *source*: arXiv 2512.17008, Table 2 and Appendix A, https://arxiv.org/html/2512.17008
- *confidence*: verified

**37. GiGPO: step-level advantages from recurring states, at plan 20's exact budget**

- *value*: 8 x 16 = 128 environments per iteration, 150 iterations, lr 1e-6, KL 0.01. ALFWorld success: GRPO 72.8 % vs GiGPO 86.7 % (1.5B); GRPO 77.6 % vs 90.8 % (7B). WebShop success 56.8 -> 65.0 % and 66.1 -> 72.8 %. Leave-one-out alone was not better than GRPO (69.7 and 75.5). More than 65 % of states recur across trajectories there
- *source*: arXiv 2505.10978v3, https://arxiv.org/html/2505.10978v3
- *confidence*: verified

**38. Group size evidence**

- *value*: 2 rollouts per prompt retain 97.6 % of 16-rollout performance with 12.5 % of rollouts and 21 % of training time; a token-level optimal baseline matches group size 32 with 4. Models and tasks not checked; neither is a long multi-turn agent result
- *source*: arXiv 2510.00977 abstract, https://arxiv.org/abs/2510.00977 ; arXiv 2602.07078 abstract, https://www.arxiv.org/abs/2602.07078
- *confidence*: verified

**39. Tree-search rollouts and hindsight relabelling for agents**

- *value*: Tree-GRPO claims better results at a quarter of the rollout budget; hindsight relabelling work for LLM agents is supervised (relabel then SFT), not combined with policy-gradient RL
- *source*: arXiv 2509.21240, 2603.21357, ICLR 2026 'Spinning Straw into Gold', seen only as search snippets
- *confidence*: not verified

**40. Plan 20 turns off the truncated-episode loss mask while a missing report scores 0**

- *value*: mask_truncated_completions true -> false; U: 0 for any non-claim, -1 for a false claim. The plan lists silence as a risk handled by guard G6 and a sensitivity analysis only
- *source*: docs/iteration-20-rl-design.md (lines 92-97, 228, 398, 430, 623)
- *confidence*: verified

## implications_for_plan

- RANK 1, well supported (Endless Terminals, ECHO, River, DeepSWE; counter-case: the two 4B small-budget results). Raise the dose and lower the rate: at least 400-500 steps at 128-256 episodes per step (51k-128k episodes) with lr about 1e-5 instead of 150 steps at 2e-5. Pay for it by cutting arm C (I found no external support for system-prompt paraphrase diversity as an RL lever) and, if needed, one seed. Expected measurable effect: external runs reached their best checkpoint at steps 380-460, so most of any gain should appear after step 150; pre-register a dose-response read on the select split at steps 150 / 300 / 450 (slope of U and of task success per 100 steps with a 90 % interval). The size on this task is an extrapolation: published in-distribution gains were +14 to +43 points from bases at 4-43 % pass, whereas G2 starts at 78.5 % task success with 7-11 U of capability headroom.
- RANK 2, well supported (DAPO +8 points, online filtering +3.8 points and under half the steps, offline curation by the initial policy +0.0, RAGEN-2 +0.8 to +6.9, SETA 49 -> 86 % informative tasks; in-repo 72-86 % zero-variance twin groups). Replace the one-off offline band plus a 0.7 zero-variance tolerance with online handling: oversample candidate prompts each step and train on those with non-zero reward variance, or at minimum re-band after every epoch from the run's own rollouts (each row is revisited three times) and refill from the unused legal variants (4,098 legal vs 800 used). Drop the 75 always-pass rows and the 'always claims' twins from the gradient batch or move them to a monitoring probe. Expected measurable effect: zero-variance share per kind below 20 % (from as high as 70 % allowed); twins' share of summed absolute advantage at or above 30 % (the repo's earlier gate; it was 11.9-24.6 %); up to about 2-3x more gradient-carrying twin groups per step.
- RANK 3, single strong source on the same task family (ECHO; unreplicated, no training seeds). Use the freed arm for GRPO plus an auxiliary cross-entropy on tool-output tokens at lambda 0.05, excluding harness warning text; plan 20 currently keeps tool-result tokens out of the loss. Expected measurable effect: ECHO measured +8.8 validation points over plain GRPO on Qwen3-8B and +9.6 on an SFT-initialised model at 500 steps, with 1.5-1.9x fewer steps to GRPO's best; here read it as the arm contrast on task success and U. Its effect on honesty behaviours is unknown, so the same guards apply; lambda above 0.1 degraded policies.
- RANK 4, supported (DeepSWE compact filtering, DAPO overlong filtering +6 points; in-repo iteration 11 cut claims 'via budget exits'). Remove the reward advantage of running out of budget: either mask at episode level the loss of episodes that end by token, turn or time limit (keeping their reward in the baseline), or score a missing report on a feasible row as -1. Plan 20 does the reverse (mask off, missing report = 0). Expected measurable effect: share of episodes ending without a report stays at or below G2's 1.35 % during training instead of being rewarded relative to a false claim; report the masked share per step (stop if above 15 %, the plan's existing masked-share rule).
- RANK 5, well supported (TruthRL collapsed to near-total abstention with an under-crediting grader; River found over 60 % of environments with verifier or spec issues and +1.7 to +4.8 average points from filtering). Make grader false negatives a hard pool gate, not a review item: under U a wrongly failed true claim costs 2 points (+1 to -1). Admit a feasible row only if the oracle solution passes and the spec audit is clean; exclude rows G2 passes 1 in 8 unless audited. Expected measurable effect: P(claim | pass) on training rows stays within the plan's 3-point bound; estimated false-negative rate of the pool reported with an interval before training (the plan's own base rate is about 13 % defective among unread sources).
- RANK 6, supported by five sources, small expected effect at current drift levels. Treat KL to G2 as a monitor and stop rule, not as protection: keep beta at most 0.01 (or 0), and when the pilot's KL-slope gate fires lower the learning rate rather than raise beta (beta 0.15 froze a 4B model at +/-0.01; beta 0.04 cost 5.5-7.1 points of new-task accuracy without reducing forgetting in one study). Protect the SFT-learned behaviours by exposure inside the batch: keep the twin share at or above 25 % if rows are added for scale (a 10 % mix failed to restore refusal on a 1.5B model). Expected measurable effect: no loss of plasticity from the penalty; behaviour retention read on the existing guards (control declines, P(claim | pass), twins destroyed) rather than on the KL value.
- RANK 7, moderately supported, different domain (SRFT +4.6 points over SFT-then-RL on math; in-repo lesson that G2 needed the decision replay). Optional arm: add G2's decision-point SFT loss on its own SFT data as a low-weight auxiliary term during RL, as an anchor that covers the behaviours a token-level KL does not. Expected measurable effect: smaller drift on read rate and control declines at equal steps; attribution of any gain to RL becomes less clean, so keep it as an arm, not the primary.
- RANK 8, mixed evidence. Decide loss aggregation by measurement: the two working terminal-agent recipes use sequence-mean, DeepSWE a constant normaliser, plan 20 token-sum; Dr. GRPO argues sequence-mean lengthens wrong answers. Before the pilot, report on the prescreen bank the token-weighted advantage sum per group (the net push on long failing episodes) under each scheme. Expected measurable effect: a direct number for the plan's listed risk 'long failing episodes dominate'; actions per episode within +/-10 % of G2 during training.
- RANK 9, supported only in simpler environments (GiGPO +13 to +14 points over GRPO at exactly 150 iterations x 128 episodes; turn-level PPO avoided GRPO's crashes; Endless Terminals itself used a critic). Step-level credit is the one external result showing a large gain at plan 20's budget, but it needs recurring states or a learned critic; speculative for Docker tasks. Cheap first step: measure on the prescreen bank how often the 8 generations of a prompt share an identical command prefix beyond turn 1. Expected measurable effect: none until that share is known; if it is above about 50 %, a step-level arm becomes worth pre-registering.
- RANK 10, speculative for this setting. Group size: at a fixed budget smaller groups and more prompts did as well or better in single-turn and toy multi-turn work (2 rollouts keep 97.6 %; 4 per prompt best in RAGEN), but twin prompts with claim rates near 0 or 1 need larger groups to produce mixed outcomes (all-same probability 0.66 at G 8 vs 0.44 at G 16 for a 0.95 rate). Expected measurable effect: unknown; only worth testing after rank 2 is in place.
- Keep, with external support: the ternary U with 0 for a non-claim and no bonus for declining (TruthRL: bonus for abstaining was worse, 32.7 vs 37.2); arm B as the success-only contrast (TruthRL's binary arm had +3.7 accuracy and +20.1 hallucination against ternary, which is the expected sign of A1 - B); three seeds and paired intervals (most cited papers report no training-seed variance). Add two health metrics from the collapse literature: mean log-likelihood of positive-advantage trajectories under the updated policy (falls before reward does) and gradient norm (spike marks collapse at steps 90-170 in two studies). Expected measurable effect: earlier stop on collapse than the reward or entropy signals give.

## open_questions

- How many optimizer steps did Endless Terminals actually run? The paper gives wall-clock only (about 2 days on 4 A100, 8 hours on 8 B200); the configs allow up to about 1,970 steps. Its training curves (Figure 3) could not be read here.
- What does one plan-20 step cost in wall-clock on the free cluster, and is 400-500 steps at 128-256 episodes feasible per arm? This decides whether rank 1 is affordable with three seeds.
- Does any published work train an agent by RL to reduce false completion claims and report an effect size? I found none after several searches; absence is not proof.
- ECHO's 'batch size 16' unit, its fine-tuning mode (LoRA or full) and its training-seed variance are not stated; its +8.8 points is a single-run contrast.
- Is the LoRA learning-rate ratio (10x, 15x for short runs) valid for multi-turn agent RL at r 32, alpha 64? The source measured single-turn math with alpha 32; the alpha/r scaling argument here is my derivation, not a measurement.
- Does a KL penalty at beta 0.04 change anything at this project's drift levels (KL about 0.02 per token by step 30 in it19c)? The external evidence on KL is from other tasks and mostly from full fine-tuning.
- How often do the 8 generations of one prompt reach identical states after turn 1 in these sandboxes? This decides whether step-level grouping (GiGPO) is usable.
- Does plan 20's blind-neutral twin weighting have any external precedent? I found none; the nearest results (sampled-threshold penalties over-rejecting even at t = 0) suggest explicit penalty schedules can still drift to over-refusal.
- Not verified here because only search snippets were seen: AbstentionBench's 24 % figure, the anti-scheming 13 % -> 0.4 % figure, Tree-GRPO's quarter-budget claim, the ART-E learning rate of 1.2e-5, the user report linking token-mean aggregation to entropy collapse, and SETA's training hyperparameters.
- TruthRL and the hallucination-tax paper are single-turn question answering with full fine-tuning; whether the ternary reward's accuracy cost (-3.7 points against binary RL) transfers to long multi-turn episodes with LoRA is untested.