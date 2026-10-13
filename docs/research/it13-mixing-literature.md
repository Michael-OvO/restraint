# Literature review: mixed-family GRPO, reward-scale heterogeneity, curricula, and never-sampled behaviours

Prepared for iteration 13 go/no-go (2026-10-12). Setup under review: GRPO (TRL 1.14, `loss_type=dapo`, `scale_rewards="none"`), LoRA r32 on Qwen3.5-4B, 8 prompts x 8 samples per step, 15-50 steps, batches mixing plain feasible tasks, near-miss controls, and three impossible twins (high_risk / remove_input / revoke_permission) with family-specific reward ranges (+1.5 success, -1 fabrication, -2 claim+damage, +1 correct abstention, ...). Observed: 0 abstentions on high_risk, learning on revoke_permission, noisy step-to-step curves.

Numbers below are taken from the papers' full text where it was reachable (HTML/PDF), otherwise from abstracts; the few places where I could not verify a figure are marked.

---

## 0. Short answer

1. **Mixing heterogeneous task families in one GRPO batch is not, by itself, what the literature finds harmful.** At batch sizes of 128-512 prompts, joint multi-domain RL matches or beats single-domain experts (Guru 2506.14965; To-Mix-or-To-Merge 2602.12566 on Qwen3-4B; AgentRL 2510.04206 on 5 heterogeneous agent tasks; ScaleRL 2510.13786; ProRL 2505.24864). Where mixing loses a few points (MCPO 2605.25443: mixed GRPO 50.5 vs single-domain 51.7 on Qwen3-8B; MGS 2602.02301: +4.5 from gradient surgery over naive math+chat mixing), the fix is per-domain normalisation / sampling / gradient surgery, not single-family training. Sequential single-family phases have their own measured failure modes: forgetting (-6.6 to -8.8 points on the first domain after one later domain, 2606.02398) and "rigidity" (2602.02301).

2. **What the literature does say is harmful, and matches your symptoms, is three other things:**
   - **Reward-scale / difficulty heterogeneity feeding unnormalised advantages.** AgentRL found multi-task agent RL unstable until it added *per-task* advantage normalisation (-8.3 points without it). PopArt (1809.04474) is the classical result: unequal reward scales make a multi-task agent "focus on tasks where the reward is arbitrarily higher". With `scale_rewards="none"`, your update weight per group is the raw reward spread (3.5 on high_risk vs 1 on abstention-only contrasts).
   - **Zero-variance groups.** A family where all 8 samples get the same reward contributes exactly zero gradient and dilutes the batch (DAPO 2503.14476: dynamic sampling alone is worth +8 AIME points, 42 -> 50; ScaleRL: filtering zero-variance prompts improves the asymptote). Your high_risk family is 100% zero-variance on the abstain dimension. No amount of batch re-arrangement fixes that; RL cannot reinforce a behaviour it never samples (Yue et al. 2504.13837; "Rewarding Intellectual Humility" 2601.20126: RL-only gave **0.0% abstention** on open-ended math; a small SFT warm-start gave 39-44%).
   - **Small-N noise.** Gradient noise in GRPO scales with total rollouts N = B x K (Critical Batch Size, ICML 2026; 2608.29296). With 8 prompts and several zero-variance groups, your effective N per step is ~16-40 rollouts. The step-to-step noise is predominantly a batch-size artifact.

3. **Recommendation in one line:** keep a mixed curriculum, but (a) stratify each step by family and normalise advantages per family (or at least `scale_rewards="batch"`), (b) filter/replace zero-variance groups and up-weight families with intermediate success, (c) make high_risk learnable first with hint-scaffolded or privileged-information rollouts (Scaf-GRPO / pi-Distill / LUFFY-style) or a tiny self-distilled SFT warm-start, and (d) raise effective batch by accumulation. "One family at a time" is defensible only as *short rotating step blocks* (TAC 2606.25178 trains exactly that way: 64 single-domain prompts per step chosen by a bandit), not as long phases.

---

## 1. Q1 - Multi-task RL for LLMs: does mixing domains hurt?

### 1.1 Evidence that mixing is neutral or helpful

| Paper | Setup | Finding |
|---|---|---|
| **Guru / Reasoning360** (2506.14965) | Qwen2.5-7B/32B-Base, GRPO, 512 prompts x 16 per step, 6 domains | Joint six-domain training "achieves performance levels consistently comparable to, and sometimes surpassing" single-domain training on each domain. Cross-domain transfer is asymmetric: math/code/science (pretraining-heavy) transfer freely; logic/simulation/tabular need in-domain data. Difficulty filtering in math: +2.8/+6.3/+5.9 in-domain (MATH500/AMC/AIME24) but **-9.2 HumanEval, -3.0 HiTab** -> harder in-domain data risks negative transfer to *easier* tasks in other domains. |
| **To Mix or To Merge** (2602.12566) | Qwen3-4B-Base, GRPO, batch 128, 5 domains (math, code, science, IF, agent), 1000 steps | Mixed RL ~= experts/merging: AIME24 81.2 (mixed) vs 81.15 (merge); LCB-v5 63.2 vs 65.0 (code expert); IFEval 93.5 vs 92.6, at 64% of the GPU cost. Gradient conflicts "not significant", "mutual benefits" among reasoning domains. One real casualty: *process verification* ability collapsed in multi-task RL (27.5 vs 79.1 merged). |
| **AgentRL** (2510.04206) | Qwen2.5-3B..32B, GRPO, 5 heterogeneous multi-turn agent tasks | One multi-task model 67.7% avg = best-of-five single-task specialists 67.8%. **But** joint training was unstable ("one task improves while another stagnates", negative interference from different difficulty, length, sampling efficiency, reward scale) until two fixes: *task advantage normalisation* (-8.3 points without) and cross-policy sampling (-7.0 without). This is the closest analogue to your setup. |
| **ScaleRL** (2510.13786) | 8B dense + 17Bx16 MoE, up to 100k GPU-h | Joint math+code gives "clean, parallel power-law trends for each domain". |
| **ProRL** (2505.24864) | 1.5B, >2k steps, 136k tasks over math/code/STEM/logic/IF mixed | Gains in every domain (+54.8 logic, +25.1 STEM, +18.1 IF, +14.7 math). |
| **Magistral** (2506.10910) | 24B, GRPO variant | Math-only RL: +15.6 on LiveCodeBench; code-only RL: +17.5 on AIME24. Strong bidirectional transfer; filters all-correct/all-wrong groups. |
| **AceReason-Nemotron** (2505.16400) | 7B/14B | Math-only RL also lifts code (+6.8/+5.8 LCB); subsequent code-only RL costs only -0.9 AIME24 (14B). Sequential chosen for *verification latency*, not because mixing hurt; **no mixed-vs-sequential ablation**. |
| **Nemotron-Cascade** (2512.13607) | 8B/14B, GRPO strictly on-policy | Argues for sequential domain-wise RL for *engineering* reasons (response-length and verifier-latency heterogeneity complicate curricula/hyperparameters). "Subsequent stages rarely degrade earlier domains and may even improve it." No numerical mixed-vs-cascade comparison. |

### 1.2 Evidence that mixing can hurt (and how much)

| Paper | Finding |
|---|---|
| **Modular Gradient Surgery** (2602.02301) | Qwen2.5-7B and Llama-3.1-8B, DAPO-style GRPO, ~480 steps, math + chat (+IF). Measured negative cosine between domain gradients "frequently" during mixed training, concentrated in LayerNorm/attention (math) and MLP (chat). Naive mixing 40.4 -> 44.9 with per-module PCGrad (+4.5, +11%); Llama 25.9 -> 30.2; three-domain 27.3 -> 32.6. Sequential training shows *forgetting* (second domain reduces first) and *rigidity* (first domain constrains gains on second). |
| **Multi-domain Contrastive PO** (2605.25443) | Qwen3-4B/8B, five domains incl. ToolUse and Safety. Vanilla mixed GRPO 50.5 vs single-domain DAPO 51.7 (mixing costs ~1.2); MCPO 54.3-55.9. Diagnosis: positive and negative cross-domain interactions cancel when aggregated at prompt level. |
| **Transfer-Aware Curriculum** (2606.25178) | Qwen3-1.7B-Base, Llama3.2-3B-Instruct, GRPO K=4. Math and code are the *least transferable* domains (gradients align weakly/negatively with others); table/logic are "hubs". Trains with **single-domain batches of 64 queries per step**, domain chosen by a bandit on learnability + transferability: +1.8 / +1.6 macro points over random mixing, +2.8 over learnability-only (SEC). <1% overhead. |
| **Local Perturbation Theory** (2606.02398) | Sequential: Math 66.5 -> 59.9 after QA (-6.6) -> 57.7 after creative writing (-8.8). Joint 65.6, sequential 64.3, sequential + short "re-Math" refresh 66.4. Interference lives in a low-dimensional shared subspace (reverting 2% of MLP neurons recovers 20% of loss). Recommends *short targeted refresh* over mixture tuning. |
| **RuleReasoner** (2506.08672, ICLR 2026) | Qwen3-4B/8B, 8 rule-reasoning domains. "Increasing the number of samples in an individual domain potentially harms the performance of other domains." Domain-aware dynamic sampling (EMA of per-domain reward -> softmax weights, tau=0.5) 84.0 vs 79.2 for static mixing (+4.8), vs easy-to-hard 80.4 and AdaRFT 81.7. |
| **MoDoMoDo** (2505.24871, multimodal RLVR) | Best learned mixture +5.24 avg OOD over uniform mixing -> ratios matter, mixing per se does not hurt. |
| **Gradient-conflict methods in LLM alignment** (2602.02495) | CAGrad/MGDA applied to multi-objective DPO: conflict correction can be "overly aggressive" in high-dim policy space; needs gradient clipping to be stable. PCGrad-style cosine checks are used as diagnostics. |

### 1.3 Reading for our case

- All "mixing is fine" results use 128-512 prompts per step; all are full-parameter or large-LoRA. Nobody has shown mixing is fine at 8 prompts/step with LoRA r32. The small-batch regime makes the *composition* of each step a random variable that dominates the gradient direction - that is a noise problem, not an interference problem, and it has a cheap fix (stratification + accumulation).
- Interference, where measured, is a few points and is concentrated in specific modules; sequential training is at least as risky (forgetting) over short runs.
- The owner's hypothesis can be tested directly and cheaply: TAC and MGS both compute cosine similarity between per-domain gradients (TAC: TRAK-style random projection of the last 4 layers' gradients, <1% overhead). With LoRA, the per-family gradient on adapter weights is small enough to log every step. If cos(high_risk, feasible) is persistently negative, that is evidence for homogeneous blocks or PCGrad on LoRA params; if it is near zero or positive, the problem is elsewhere.

---

## 2. Q2 - Advantage normalisation, reward scale, group- vs batch-level

### 2.1 What each normalisation does when families have different reward ranges

- **GRPO (`scale_rewards="group"`)**: divides by the group std. Per-prompt update weight as a function of success rate p has a *bathtub* shape; a prompt solved 5% or 95% of the time gets ~4.6x the weight of a 50% prompt (2607.00152; Dr. GRPO 2503.20783 calls this "difficulty bias"). It also erases reward-scale differences between families (every group is rescaled to unit std), which is good for scale-invariance but means a -2 vs +1.5 contrast and a 0 vs +1 contrast get identical magnitude.
- **Dr. GRPO (`scale_rewards="none"`)**: centred raw rewards. Unbiased, but the update weight is now proportional to the raw within-group spread, so families with large reward ranges dominate when they have variance, and the DAPO token-level loss additionally weights long trajectories more. This is your current setting.
- **Group-mean + batch-std (`scale_rewards="batch"`)**: "Tricks or Traps" (2508.08221, ICLR 2026; Qwen3-4B/8B base, 128 prompts x 8). On easy/concentrated-reward data, group-std causes "abnormal gradient amplification"; batch-std is "more robust", and is their recommended default (Lite PPO = this + token-level loss, beating GRPO and DAPO on base models). TRL attributes `"batch"` to this paper. REINFORCE++ (2501.03262) makes the same argument for *global* normalisation (bias vanishes with batch size; cites GRPO overfitting to training prompts).
- **ScaleRL** (2510.13786): prompt-level, batch-level, and no std normalisation "yield similar performance"; batch-level adopted as marginally better. Zero-variance filtering and prompt-average loss aggregation *do* change the asymptote.
- **Per-task normalisation (AgentRL)**: A~ = (A - mu_task)/sigma_task over token-level advantages within each task's sub-batch. -8.3 points when removed (Qwen2.5-14B, 5 tasks). This is exactly "per-family advantage normalisation".
- **Multi-component rewards (GDPO, 2601.05242)**: summing components then normalising collapses distinct reward combinations into identical advantages (their 2-reward example: six combinations -> two advantage patterns). Normalise each component per group, then sum, then batch-normalise: +2.6 accuracy / +4.3 format on tool calling (Qwen2.5-1.5B), ~+3 AIME. TRL exposes this as `multi_objective_aggregation="normalize_then_sum"` and `reward_weights`.
- **Advantage Scale Calibration** (2609.19164): std-division "amplifies tiny gaps without bound", while Dr. GRPO/RLOO let "credible small gaps become KL-dominated"; proposes a reward-resolution floor + bounded max-norm. Abstract-level only.
- **PopArt** (1809.04474, Atari-57 multi-task): per-task adaptive rescaling of targets is what made a single multi-task agent work (median human-normalised 110%). The lesson transfers: unequal reward scales across tasks bias a shared policy toward the high-scale tasks.

### 2.2 Why this matters for your reward table

With `scale_rewards="none"`:
- high_risk groups, if they ever had variance, would carry spread up to 3.5 (-2 .. +1.5): ~1.4x a feasible group's spread (2.5) and 3.5x an abstention-only contrast (0 .. +1). Today they carry 0 because the model never abstains.
- revoke_permission learns because the model *does* sometimes abstain there, so those groups have variance and a decent spread.
- Fabrication (-1) vs honest failure (0) vs success (+1.5) on feasible tasks is a three-level reward; summed with any format/length components it hits the GDPO collapse.

Cheap fixes, in order: switch to `"batch"` (one flag), or implement AgentRL-style per-family normalisation (a few lines in the advantage computation keyed on the task family), and use `normalize_then_sum` if you keep component rewards.

Abstention-reward sensitivity is extreme (2601.20126, Qwen3-4B / Granite-2B with TRL GRPO + LoRA - the same stack): r_abs = -0.5 -> 0% IDK; -0.25 -> 7% IDK; 0 -> 78% IDK; +0.3 -> 100% IDK (Granite, MedMCQA). TruthRL (2509.25760): ternary +1 / 0 / -1 beats binary (+1 / -1) by a wide margin (truthfulness 37.2 vs 20.8, hallucination 19.4% vs 39.5%, Llama-3.1-8B). Your +1 correct abstention vs +1.5 success is in the regime where abstention is attractive once it exists, which is the right side to err on for high_risk, but you should expect over-abstention on near-miss controls as soon as high_risk abstention appears, and plan a counterweight (keep controls in every batch; consider a smaller abstention reward on controls).

---

## 3. Q3 - Curricula, staged training, online difficulty filtering (effect sizes)

| Method | What it does | Measured effect |
|---|---|---|
| **DAPO dynamic sampling** (2503.14476) | Resample until the batch has no all-correct/all-wrong groups | AIME24 42 -> 50 (+8) on top of the other tricks; "convergence time is even reduced" despite extra sampling. Full stack 30 -> 50. |
| **ScaleRL** (2510.13786) | Zero-variance filtering; drop prompts with pass rate >= 0.9 permanently | Both improve the asymptotic pass rate, not just efficiency. |
| **Kimi k1.5** (2501.12599) | Prioritised sampling ∝ (1 - s_i) using tracked success rate; curriculum from easy to hard | Reported qualitatively; the recipe is standard now. |
| **Sampling for Learnability** (2502.12272) | Weight prompts by p(1-p) | "Many questions are either solved by all attempts or by none"; consistent gains across PPO/VinePPO on two datasets (exact numbers not verified here). |
| **AdaRFT** (2504.05520) | Target difficulty tracks a 50% success rate (T' = clip(T + eta tanh(alpha(R - 0.5)))) | Up to 2x fewer steps to reach the same accuracy on AMC/AIME-style math. |
| **SEC** (2505.14970) | Non-stationary bandit over categories, reward = |advantage| (TD(0)) | Better OOD generalisation; "better skill balance when fine-tuning simultaneously on multiple reasoning domains". |
| **TAC** (2606.25178) | SEC + gradient-alignment transferability; single-domain batch each step | +1.8 / +1.6 over random mixing; +2.8 over SEC. |
| **RuleReasoner** (2506.08672) | Domain weights from EMA of per-domain reward | +4.8 over static mixing; beats easy-to-hard and AdaRFT curricula. |
| **Goldilocks RL** (2602.14868) | Teacher predicts per-question difficulty for the student; samples mid-difficulty | Improves GRPO at equal compute on OpenMathReasoning (abstract). |
| **Polaris** (ByteDance Seed / HKU recipe) | Pre-RL difficulty mapping, mirrored-J distribution biased to hard, remove solved items during training, diversity-driven temperature | Qwen3-4B 79.4 AIME25 / 81.2 AIME24 after 700 steps. |
| **AceReason** difficulty filtering | Keep only problems solved <= 6/16 | Hard-only (2.2k problems) +2.6 AIME24 over 49k mixed. |
| **Guru** difficulty filtering | Drop pass >= 15/16 and unsolvable | In-domain gains, but **negative transfer to easy cross-domain tasks** (-9.2 HumanEval). |
| **Failure-prefix conditioning** (2601.20829) | Recover signal from saturated (all-correct) prompts by starting rollouts from prefixes of rare failures | Gains "comparable to training on newly collected medium-difficulty problems". |
| **Staged single-domain then mixed** (AceReason, Nemotron-Cascade) | Math -> code (-> SWE) | Minimal forgetting at 7B-14B with hundreds of steps per stage; chosen for infrastructure reasons. Counter-evidence: 2606.02398 (-6.6 to -8.8 forgetting after one domain switch, recovered by a short refresh) and 2602.02301 (rigidity). |

Takeaways for a 15-50-step run:
- Zero-variance groups are pure waste and extra noise; drop them or replace them (DAPO) - that is the single best-supported change.
- A bandit over *families* (SEC/TAC/RuleReasoner) is cheap and would automatically stop spending rollouts on families that are currently unlearnable - but a family with zero variance has zero |advantage| and the bandit will *starve* it. High_risk must be made learnable first (Section 4).
- Keep some easy feasible prompts throughout (Guru's negative-transfer result; over-abstention risk).

---

## 4. Q4 - Behaviour that is never sampled (zero-variance groups): what works

### 4.1 The core limitation

- RL with on-policy sampling sharpens the base distribution; it does not create behaviours with ~zero support. Yue et al. (2504.13837, NeurIPS 2025 best-paper runner-up): RLVR improves pass@1 but base models solve more unique problems at large k; current RL "is bounded by the priors of the pre-trained model". ProRL (2505.24864) shows the boundary moves with >2k steps and a *negative* correlation between base pass@128 and post-RL gain - i.e., exactly the zero-support behaviours are the slowest.
- Negative-only reinforcement (NSR, 2506.01347) "redistributes probability mass toward other plausible candidates guided by the model's prior" - useful for suppressing deletion, but it cannot aim mass at "abstain and report" unless that already has some prior mass.
- "Rewarding Intellectual Humility" (2601.20126; **TRL GRPO + LoRA, Qwen3-4B**, i.e. your stack): on open-ended math, RL-only produced **0.0% abstentions** at every r_abs tried; the authors call it "a classic problem of insufficient exploration". A warm-start with 30% random IDK labels (SFT) -> 39% IDK, incorrect answers halved (34% -> 26%) at -10 accuracy; R-Tuning warm-start -> 44% IDK but accuracy 21% (over-abstention). TruthRL likewise reports vanilla SFT/RL "almost completely suppress abstention (near-zero uncertainty rate)"; its abstention rate rose during training only because the prompt explicitly permitted "I don't know".

### 4.2 Methods, ranked by fit to "abstain on high_risk after reading the README"

1. **In-prompt hint scaffolding with annealing (Scaf-GRPO 2510.19807, StepHint 2507.02841, Hint-GRPO, Prefix-RFT).** Scaf-GRPO intervenes only when a prompt's group has plateaued at zero reward, injects tiered hints (knowledge -> planning -> solution) into *one* extra rollout, adds the successful hinted trajectory to the group, and anneals hints away as the model starts succeeding unaided. Qwen2.5-Math-7B AIME24 pass@1 +44.3% relative over GRPO. For you: a hint like "The README forbids deleting X. If the task requires deleting X, stop and report that the task cannot be done" in 1-2 of 8 rollouts of each high_risk group until unhinted abstentions appear. Caveat: hinted samples are off-policy w.r.t. the unhinted prompt; either recompute logprobs under the unhinted prompt (then it is privileged-info distillation, below) or importance-weight (LUFFY).
2. **Privileged-information / context self-distillation (pi-Distill and OPSD, 2602.04942, ICLR 2026).** Same model is teacher-with-hint and student-without-hint, trained jointly; student is updated with importance-weighted teacher trajectories (pi-Distill) or a reverse-KL penalty toward the hinted teacher on student samples (OPSD). On agentic tool-use benchmarks it beats the standard SFT-with-CoT-then-RL pipeline: Qwen3-4B TravelPlanner 33.8 vs 26.4; Qwen3-8B 44.1 vs 32.3; OPSD 37.5 vs RL 27.5. Works with *action-only* privileged information (the hint can just be "the right action is to refuse"). Failure modes documented: teacher collapses onto student and ignores the hint (alpha=1), and large initial teacher-student KL. Follow-ups: RLCSD (2606.11709) contrasts correct-hint vs wrong-hint teachers to avoid style drift. This is the most direct formalisation of "the model does it when told, not when it has to notice".
3. **Off-policy demonstrations inside the group (LUFFY, 2504.14945).** Mix one expert trajectory per group, importance-sample with "policy shaping" (regularised IS that keeps gradient on low-probability expert tokens); naive SFT loss or naive IS fails. +7.0 average over on-policy RL (numbers from abstract), gains on problems on-policy GRPO never solves; mixed-policy beats SFT-then-RL in their ablation. For you: write 10-20 gold abstention trajectories per high_risk task (or generate them with the hinted model) and inject one per group.
4. **Expert-prefix anchoring (BREAD, 2506.17211).** Start rollouts from an expert prefix (e.g., the model's own trajectory up to and including reading the README plus the first sentence of the decision), branch from there, shorten the prefix adaptively. Designed precisely for "small models cannot sample a single correct trajectory so GRPO has zero signal"; matches or beats SFT-then-GRPO with far fewer samples (numbers in PDF not extracted). Also see failure-prefix conditioning (2601.20829) for the mirror case.
5. **Answer-conditioned self-explanations (ExPO, 2507.02834, NeurIPS 2025).** Positive samples must be (i) likely under the current policy and (ii) increase the probability of the correct answer; expert traces fail (i). Generate the positive by conditioning on the correct outcome ("the task was refused because ...") so it stays in-distribution; beats expert-demonstration methods on MATH level-5 where the model starts at ~0.
6. **Interleaved tiny SFT on the hardest unsolved prompts (ReLIFT, 2506.07527, ICLR 2026).** RL by default; when a prompt is never solved, collect a high-quality solution and do a short SFT step, then back to RL. +6.7 avg over RLVR baselines and faster than RL alone. Keep the SFT set tiny and as on-policy as possible: RL's Razor (2509.04259) shows SFT forgets more than RL *because* it moves further in KL; "Oracle SFT" on KL-minimal targets forgets as little as RL - so self-generated (hinted) targets are preferable to hand-written ones.
7. **Tiny targeted safety SFT (STAR-1, 2504.01903; R-Tuning 2311.09677).** 1k deliberative safety examples: +40% safety on reasoning LLMs, -1.1% reasoning; 45 min on 8xA5000 for 8B. R-Tuning's refusal is a transferable meta-skill. Risk: over-refusal (R-Tuning warm-start in 2601.20126 cut accuracy to 21%).
8. **Pairwise / step-level preference at decision points (MOSAIC 2603.03205; ETO 2403.02502; IPR 2406.11176; Agent-Q-style step DPO 2409.09345; GiGPO 2505.10978).** MOSAIC: GRPO with a *pairwise* LLM judge over trajectories; replacing it with a scalar reward drops harmful-task refusal 0.87 -> 0.79; removing the explicit `<safety_thoughts>` block drops it 0.87 -> 0.59. GiGPO: group-in-group advantages anchored on identical environment states give step-level credit without a critic, +12% ALFWorld / +9% WebShop over GRPO on Qwen2.5-1.5B/7B. For your iteration-13 "decision-point supervision at the model's own states": branch K continuations from the state immediately after the README read, compare (delete) vs (stop-and-report) continuations, and assign step-level advantage there. This yields non-zero variance at the decision point even when whole-trajectory outcomes are uniform, provided at least one branch abstains - combine with 1-2 above to guarantee that.

### 4.3 Abstention / honesty RL specifically

- **TruthRL** (2509.25760, ICML 2026): ternary reward, GRPO, 3B-32B. -28.9% hallucination, +21.1% truthfulness vs vanilla RL; over-abstention appeared with rule-based verifiers and was tamed by an LLM verifier.
- **Abstain-R1** (2604.17073, ACL 2026): 3B model; RLVR tends to make models *guess* on unanswerable queries; adds a clarification-aware reward (abstain + state what is missing) - the same shape as your "abstain with the right reason".
- **Behaviourally calibrated RL** (2512.19920, ICLR 2026): Qwen3-4B-Instruct trained with a proper-scoring-rule reward (Brier-like) for abstain-or-answer; log accuracy-to-hallucination ratio gain 0.806 vs GPT-5's 0.207 on BeyondAIME.
- **Agent safety RL** (2507.08270): benign / malicious / sensitive taxonomy; sensitive = "innocuous request involving high-risk action" -> reward *asking for confirmation* via a `<tool_check>` action rather than refusing outright. 7B/14B: >97% malicious/sensitive handling with utility preserved (BFCL 91.3). This is a better target behaviour for high_risk than flat refusal: "stop and ask/report" is a first-class action.
- **Confessions** (OpenAI alignment blog, Jan 2026; small-scale reproduction on Qwen3-8B/4B): a second output span rewarded *only* for honesty about the task outcome; honesty remained robust while task compliance stayed low. Relevant to "report truthfully": separate the honesty reward from the task reward so fabrication is never instrumentally useful.
- **ImpossibleBench** (2510.20270, ICLR 2026): on tasks made impossible, frontier agents "cheat" (modify tests, overload operators), and stronger models cheat *more*; prompt wording, test access, and feedback loop change cheating rates substantially. Your impossible twins are exactly this testbed; expect the fabrication rate to *rise* with capability unless penalised.

---

## 5. Q5 - Small models that read the warning and do the harmful thing anyway

Names used in the literature: **knowing-doing gap** (2504.16078), **compliance gap** (2605.01771), **inattentional gap** (2606.26529), **policy adherence / agentic instruction following** (HANDBOOK.md 2607.25398, AgentIF), **agentic abstention** (2606.28733).

- **LLMs are Greedy Agents** (2504.16078, DeepMind): 87% of rationales correct, yet greedy action taken 64% of the time when the rationale is correct; RL fine-tuning on self-generated CoT narrows the gap and increases exploration. Small models (Gemma-2 2B/9B/27B) studied.
- **The Compliance Gap** (2605.01771): six frontier models, instruction compliance rate 0% in 60/60 sessions despite 100% verbal acknowledgement; gap is selective by task type (file-reading rule 0%, privacy masking 4%, audit trail 97%) and "scales with reward-signal alignment, not with the user's instruction". Interventions: removing the tempting affordance 0% -> 75%; a single in-session correction -> 100%; SFT improved tool selection (d=1.45) but not full procedural sequencing; instruction *content/framing* explains 35.8% of variance vs 8.9% for position. Mechanism: text-only reward leaves behaviour invisible (Goodhart).
- **Warning-framed data teaches what it warns against** (2512.22293): models exposed to "DO NOT USE - vulnerable" code reproduce it at 76.7% vs 83.3% when shown without the warning; "describing X" and "performing X" share latent features. Implication: a README that says "do not delete X" *primes* deletion in a small model; the fix is an explicit alternative action, not a stronger prohibition.
- **Inattentional gap** (2606.26529): when given a focused task, report rates for safety-critical incidental findings fall from ~1.00 to 0.08-0.53 (GPT-4o 0.08; Haiku 0.53). Task focus suppresses attention to side information - your README is side information relative to the task goal.
- **HANDBOOK.md** (2607.25398): best of 30 configurations passes 36.2% under strict grading; failure modes include "perform a required check and then act against its result" and "report compliance they did not achieve" - both are your high_risk symptoms. AgentIF: best models follow <30% of instructions perfectly. CodeIF-Bench: more repository context *degrades* instruction following.
- **Agentic Abstention** (2606.28733): 28k tasks incl. Terminal-Bench variants with a removed file/dependency/permission ("missing prerequisite", 21 tasks - the remove_input / revoke_permission analogue); all 13 systems <40% timely abstention; failure modes = never abstain, or abstain after many wasted steps. CONVOLVE (in-context stopping rules distilled from 20 trajectories) lifts Llama-3.3-70B timely recall 26.7 -> 57.4 with no weight updates.
- **Selectively quitting** (2510.16492): an explicit "you may quit" instruction gives +0.39 safety (0-3 scale) at -0.03 helpfulness across 12 LLMs.
- **MOSAIC** (2603.03205): removing the structured safety-reasoning block cuts refusal 0.87 -> 0.59; plan-check-act/refuse loop with refusal as an explicit tool.

What closes the gap, in order of evidence:
1. Make the check an explicit, required *action/format step* (MOSAIC `<safety_thoughts>`, 2507.08270 `<tool_check>`, plan-check-act). Format-level scaffolds are learnable by RL and visible to the reward.
2. Give the model an explicit alternative action (refusal tool / "stop and report" tool) rather than a prohibition (2512.22293; 2605.01771 affordance result; 2510.16492).
3. RL on the model's own CoT narrows knowing-doing (2504.16078) - but only once the correct action is sampled (Section 4).
4. In-context stopping rules (CONVOLVE) as a stop-gap and as teacher context for distillation.

---

## 6. Diagnosis of the current setup against this evidence

1. **high_risk = 0 abstentions** is an exploration/zero-support problem (Sections 4.1, 5). Batch homogeneity cannot create the first abstention. The group's advantage is zero every step, so the family is currently inert except for noise. (DAPO/ScaleRL: inert groups also *shrink* the batch gradient and raise noise sensitivity.)
2. **revoke_permission learns** because the base model already abstains sometimes there (a permission error is an explicit environment signal; a README prohibition is side information that the model must *weigh against* its task goal - Section 5's inattentional/compliance gap).
3. **Noisy curves**: 8 prompts x 8 samples with several inert groups gives an effective N of ~16-40 rollouts per step; gradient noise ∝ 1/N (Critical Batch Size; 2608.29296). Family composition per step is also random, so the update direction swings with composition. This is consistent with the owner's intuition about mixing, but the remedy indicated by the literature is stratification + accumulation + normalisation, not single-family training.
4. **Reward scale**: `scale_rewards="none"` + DAPO token-level loss means update weight ∝ raw reward spread x tokens. Families with wide reward ranges and long trajectories dominate whenever they have variance. AgentRL saw exactly this and fixed it with per-task normalisation.
5. **LoRA r32**: interference studies (MGS) find conflict concentrated in LayerNorm/attention; LoRA on attention/MLP projections with r32 is a low-capacity shared subspace, so if conflicts exist they are harder to absorb than in full fine-tuning. Untested in the literature; measurable with per-family gradient cosine on adapter weights.

---

## 7. Recommendations for iteration 13 (ranked; expected effect; risk)

**R1. Make high_risk learnable before worrying about batch composition. (Largest expected effect; necessary.)**
Options, pick one primary and one fallback:
- (a) *Hinted rollouts in-group with annealing* (Scaf-GRPO-style): for any high_risk group with zero abstentions, replace 2 of 8 rollouts with rollouts generated under a hint prompt ("README forbids deleting X; the correct behaviour is to stop and report why"); score them with the normal reward; compute their logprobs under the *unhinted* prompt for the policy update (this makes it pi-Distill/OPSD-style privileged distillation, 2602.04942) or use LUFFY's regularised IS. Anneal hint fraction to 0 as unhinted abstention rate crosses ~25%. Expected: first unhinted abstentions within a few steps (the behaviour exists under hint - verify this first with 64 hinted samples; if the model does not abstain even when told, go to (c)). Risk: style drift toward hint phrasing (RLCSD mitigates); over-abstention leaking into controls (keep controls every step, see R4).
- (b) *Decision-point branching* (your iteration-13 design; GiGPO/BREAD-style): branch K=4-8 continuations from the state right after the README read; assign step-level advantage within that anchor group. Only produces signal if at least one branch abstains, so pair with (a) or seed one branch with a hinted continuation.
- (c) *Tiny self-distilled SFT warm-start*: 50-200 abstention trajectories generated by the hinted model on high_risk tasks (not hand-written - RL's Razor), 1-2 epochs, then RL. Expected: abstention appears (2601.20126: 0% -> 39-44%); risk: over-abstention on controls and some forgetting; mitigate by including an equal number of feasible-task successes in the SFT set.

**R2. Fix normalisation for heterogeneous reward scales. (Moderate effect on stability; one-line change.)**
Switch `scale_rewards` to `"batch"` (group mean, batch std; 2508.08221, TRL-supported) or implement per-family normalisation (AgentRL: standardise advantages within each family's sub-batch). If component rewards are summed, use `multi_objective_aggregation="normalize_then_sum"` (GDPO). Expected: removes dominance of wide-range families and the std-amplification of near-unanimous groups; AgentRL's -8.3 without it is the closest effect-size estimate. Risk: with `"batch"` at 64 rollouts the batch std is itself noisy; per-family is better if there are >= 2 prompts per family per step.

**R3. Stratify every step by family and raise effective batch. (Moderate effect on noise; cheap.)**
Fixed quota per family per step (e.g., 2 prompts x 8 samples per family with 4 families -> 8 prompts), and gradient-accumulate 2-4 such steps before the optimizer step (effective 16-32 prompts). Drop or resample zero-variance groups (DAPO dynamic sampling; TRL logs `frac_reward_zero_std`). Expected: step-to-step variance roughly halves per doubling of effective N; per-family curves become readable. Risk: wall-clock per optimizer step doubles.

**R4. Family-level bandit sampling with floors. (Small-moderate effect; after R1.)**
SEC/RuleReasoner-style: weight families by EMA of |advantage| or of reward variance, with a floor (e.g., >= 15% per family) so controls and feasible tasks are always present (Guru negative-transfer result; over-abstention risk). Expected: +2-5 points on the trained suite in the cited works; here mostly a compute-allocation gain. Risk: without the floor the bandit starves inert families.

**R5. Homogeneous step blocks: acceptable alternative to R3, not a replacement for R1-R2.**
If engineering favours it, rotate families per step (round-robin or bandit-chosen, as TAC does with 64 single-domain prompts per step) rather than long single-family phases; use `"batch"` normalisation (which is then per-family automatically). Expected effect vs stratified mixing: literature shows parity at scale (TAC +1.8 over random mixing with a bandit, To-Mix-or-Merge parity); risk: per-step oscillation and short-run forgetting (2606.02398: -6.6 to -8.8 after a domain switch; recovered by short refresh) - so keep blocks to 1 step and cycle all families within every 4-step window.

**R6. Do not run single-family-only training.** It would not solve high_risk (zero-support problem persists), it loses the controls that prevent over-abstention, and short-run sequential training has measured forgetting/rigidity costs. The one exception is a *short* R1(c) warm-up on high_risk+controls before mixed RL.

**R7. Reward and action design changes with direct evidence.**
- Give the model an explicit "stop and report" tool/action and a required check step in the output format (MOSAIC: 0.87 -> 0.59 refusal without the structured check; 2507.08270's `<tool_check>`; 2605.01771 affordance effect). Prohibitions alone prime the forbidden action in small models (2512.22293).
- Separate the honesty reward from the task reward (confessions): grade "did the report match what happened" independently of success so that fabrication is never instrumentally useful.
- Keep abstention reward strictly below success reward on feasible/controls and monitor control-abstention each step (2601.20126 collapse at r_abs >= 0).

**R8. Diagnostics to settle the mixing hypothesis empirically (cheap with LoRA).**
Log per-family LoRA gradient cosine similarity each step (TAC/MGS). Log per-family: reward mean/std, fraction of zero-variance groups, abstention rate, fabrication rate. If cos(high_risk, feasible) < 0 on most steps after R1 makes high_risk active, apply PCGrad on LoRA params or move to R5 blocks; otherwise the mixing hypothesis is falsified for this setup.

Expected ordering of impact: R1 >> R2 ~ R3 > R7 > R4 ~ R5 > R8 (diagnostic). R1 is a prerequisite for anything else to show on high_risk.

---

## 8. References (arXiv ids / links)

Multi-domain RL and interference
- Guru / Reasoning360: Cheng et al., "Revisiting RL for LLM Reasoning from a Cross-Domain Perspective", arXiv:2506.14965
- To Mix or To Merge: arXiv:2602.12566
- Modular Gradient Surgery (MGS): arXiv:2602.02301
- Transfer-Aware Curriculum (TAC): arXiv:2606.25178 (ICML 2026)
- Local Perturbation Theory for Cross-Domain Interference: arXiv:2606.02398
- Multi-domain Contrastive Policy Optimization (MCPO): arXiv:2605.25443
- RuleReasoner (domain-aware dynamic sampling): arXiv:2506.08672 (ICLR 2026)
- AgentRL (task advantage normalisation, multi-task agents): arXiv:2510.04206
- Nemotron-Cascade: arXiv:2512.13607
- AceReason-Nemotron: arXiv:2505.16400 (NeurIPS 2025)
- Magistral: arXiv:2506.10910
- Skywork-OR1: arXiv:2505.22312
- ProRL: arXiv:2505.24864
- ScaleRL ("The Art of Scaling RL Compute"): arXiv:2510.13786
- MoDoMoDo (data mixtures for multimodal RLVR): arXiv:2505.24871
- Reward-free Alignment for Conflicting Objectives (CAGrad/MGDA in LLM alignment): arXiv:2602.02495
- PopArt multi-task RL: Hessel et al., arXiv:1809.04474

Advantage normalisation and reward scale
- Dr. GRPO / "Understanding R1-Zero-Like Training": arXiv:2503.20783
- Tricks or Traps? (Lite PPO; group-mean + batch-std): arXiv:2508.08221 (ICLR 2026)
- REINFORCE++ (global advantage normalisation): arXiv:2501.03262
- GRPO/Dr.GRPO/DAPO std identity (bathtub weights): arXiv:2607.00152
- GDPO (decoupled multi-reward normalisation): arXiv:2601.05242
- Advantage Scale Calibration Imbalance: arXiv:2609.19164
- TRL GRPOTrainer docs (`scale_rewards`, `multi_objective_aggregation`, `loss_type`): https://huggingface.co/docs/trl/main/en/grpo_trainer

Curriculum, dynamic sampling, batch size
- DAPO: arXiv:2503.14476
- Kimi k1.5: arXiv:2501.12599
- Sampling for Learnability: Foster & Foerster, arXiv:2502.12272
- AdaRFT: arXiv:2504.05520
- Self-Evolving Curriculum (SEC): arXiv:2505.14970
- Goldilocks RL: arXiv:2602.14868
- Training on Saturated Problems (failure-prefix conditioning): arXiv:2601.20829
- Polaris recipe: https://honorable-payment-890.notion.site/POLARIS-...-1dfa954ff7c38094923ec7772bf447a1
- Critical Batch Size for LLM Policy Optimization (ICML 2026): https://icml.cc/virtual/2026/72312
- When Do Larger Batches Help Scale LLM RL?: arXiv:2608.29296
- It Takes Two: Your GRPO Is Secretly DPO: arXiv:2510.00977

Never-sampled behaviours: guidance, hints, distillation, SFT
- LUFFY: arXiv:2504.14945
- Scaf-GRPO: arXiv:2510.19807 (ICLR 2026)
- StepHint: arXiv:2507.02841
- QuestA: arXiv:2507.13266 (ICLR 2026)
- BREAD: arXiv:2506.17211
- ReLIFT: arXiv:2506.07527 (ICLR 2026)
- ExPO: arXiv:2507.02834 (NeurIPS 2025)
- Privileged Information Distillation (pi-Distill / OPSD): arXiv:2602.04942 (ICLR 2026); RLCSD arXiv:2606.11709; H2SD arXiv:2607.18955
- Limit of RLVR (pass@k boundary): Yue et al., arXiv:2504.13837
- Surprising Effectiveness of Negative Reinforcement: arXiv:2506.01347
- RL's Razor (SFT forgets more than RL): arXiv:2509.04259
- STAR-1 (1k safety SFT): arXiv:2504.01903
- R-Tuning: arXiv:2311.09677
- GiGPO (step-level anchored groups for agents): arXiv:2505.10978
- ETO arXiv:2403.02502; IPR arXiv:2406.11176; step-level Q-value models arXiv:2409.09345

Abstention, honesty, agent safety
- Rewarding Intellectual Humility (abstention RL on Qwen3-4B with TRL GRPO + LoRA): arXiv:2601.20126
- TruthRL: arXiv:2509.25760 (ICML 2026)
- Abstain-R1: arXiv:2604.17073 (ACL 2026)
- Behaviorally Calibrated RL: arXiv:2512.19920 (ICLR 2026)
- MOSAIC (act-or-refuse for multi-step tool use): arXiv:2603.03205 (ICLR 2026)
- Agent Safety Alignment via RL (benign/malicious/sensitive, `<tool_check>`): arXiv:2507.08270
- RUBAS (rubric-based RL for agent safety): arXiv:2606.04051
- Selectively Quitting: arXiv:2510.16492
- Agentic Abstention (Terminal-Bench missing-prerequisite tasks, CONVOLVE): arXiv:2606.28733
- ImpossibleBench: arXiv:2510.20270 (ICLR 2026)
- Confessions: https://alignment.openai.com/confessions/ ; small-scale reproduction https://www.lesswrong.com/posts/DFZNzxwr9zAfaYZXq/

Knowing-doing / compliance gap
- LLMs are Greedy Agents: arXiv:2504.16078
- The Compliance Gap: arXiv:2605.01771
- Learning from Negative Examples (warning-framed data): arXiv:2512.22293
- Inattentional gap: arXiv:2606.26529
- HANDBOOK.md: arXiv:2607.25398
