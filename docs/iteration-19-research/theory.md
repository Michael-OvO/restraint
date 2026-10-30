# Theory and literature: RL that maximises task success under honesty constraints that oppose it

Scope: the core contribution as the owner frames it. The agent optimises verified task success, and the honesty
constraints (do not claim falsely, abstain on impossible or harmful tasks, do not tamper) cost reward. This file covers
the theory and literature for that setting. Each section says what it implies for this project: Qwen3.5-4B + LoRA,
GRPO groups of 8, rewards graded on the final sandbox state, twins and controls, and a nominal budget of ~200 RL steps.
Repo evidence is cited as path plus section or line, at HEAD `9bcba69`. Papers are cited with title, authors, year and
URL. Where a number is my own derivation, it is marked "derived". "Unknown" means unknown.

---

## 0. The decision model everything below uses (derived)

An episode has a context x (task plus environment) and a trajectory. The grader gives y ∈ {0,1} (hidden tests pass, no
violation, no tamper), h ∈ {0,1} (protected data damaged) and the report r ∈ {claim, not-claim}. Write
q = P(y = 1 | everything the policy has seen at report time).

**Repo reward as a threshold rule.** Claiming is worth p(2 + λ) − 1 and not claiming is worth α + λp. The two are equal
at p = (1 + α)/2 (`docs/design.md` line 343). That is a Chow reject rule with an explicit cost ratio. In Lagrangian
form, take "verified success" as the objective and "false claims" as the cost with multiplier μ. Then the per-episode
Lagrangian is claim·[q − μ(1 − q)], so the policy claims iff q > μ/(1 + μ). The repo's α corresponds to
μ = (1 + α)/(1 − α): μ = 1.67 at α = 0.25 and μ = 1.22 at α = 0.1 (derived). **The hand-set α is a hand-set Lagrange
multiplier.** The constrained view does not change the form of the decision. It changes who chooses μ: a target rate ε
chooses it, instead of a person.

**Effort incentive compatibility (the "do not prefer quitting" condition).** Let e be the effort the agent chooses and
q(e) its pass probability. Let V(q) be the expected reward after the best report. The agent never prefers to quit just
to be paid for honesty iff V is nondecreasing in q. Two cases:

- Threshold rewards (claim/not-claim payoffs that depend on y) give V(q) = max(q(2 + λ) − 1, α + λq), which is
  nondecreasing. So quitting is weakly dominated whenever effort raises q. λ > 0 makes this strict below the threshold.
- Any strictly proper scoring rule paid on y has expected truthful score G(q), and G is **convex**: the Savage
  representation (Gneiting & Raftery 2007, below). Brier gives G(q) = −q(1 − q), which is maximal at q ∈ {0, 1}.
  So V(q) = task(q) + w·G(q) is non-monotone. Quitting, which drives q to 0, earns the scoring rule's maximum. This is
  the formal version of `docs/iteration-17-design.md` §2. **A scoring term on an outcome the agent controls is a
  performative prediction, and proper scoring rules are not incentive compatible under performativity**
  (Oesterheld et al. 2023; Perdomo et al. 2020, below).

**Proposition (design rule, derived).** Every honesty term in the online reward must be either (i) a payoff for the
report *given* y whose induced V(q) is nondecreasing, or (ii) a gradient that reaches only the report tokens, with the
outcome's dependence on the actions cut (a stop-gradient through y). A convex function of q that reaches the actions,
such as Brier, log score or "certainty", pays the agent to quit.

**The caution-level direction (derived).** On a twin/control pair, take a policy that ignores the evidence and declines
with probability d on both. Under a group-mean baseline (`scale_rewards: none`), the expected policy gradient along d is
proportional to d(1 − d)·[w_T(R_T,dec − R_T,go) + w_C(R_C,dec − R_C,go)]. Plug in the it15 G3 configuration (α 0.1,
false_blame −0.4, β 1, λ 0.5; `docs/iteration-15-g3-rl-read.md` "Design"):

- twin slope: 1.0 − (−2.0) = +3.0, or +2.5 with a wrong reason;
- control slope: −0.4 − (0.8·1.58 − 0.2) ≈ −1.46, at a control success of about 0.8.

The net blind gradient is **+1.5 per pair** with w_T = w_C: caution rises whether or not the agent read. That is what
G3 showed: declines went up in both branches, from 36 to 48 % on unread twins and from 37 to 43 % on unread controls,
and the read rate stayed flat at 46 % (`docs/iteration-15-g3-rl-read.md` "Result"). Neutralising this direction needs
w_C/w_T ≈ 2.05, or a control-decline cost near −1.9 instead of −0.4. Once that direction is zeroed, the only way left
to raise reward is discrimination: the value of information of reading. **This is one quantitative prediction the
theory makes and the past data fit.** It is a one-step bandit approximation and ignores the group composition of plain
rows.

**Gate saturation.** The factor d(1 − d) also explains "a reward cannot teach what is never sampled"
(`docs/training-log.md` Reflections, line ~121) and the zero-variance groups. Che et al. 2026 (below) prove the same
for abstention-as-action.

---

## (a) Constrained MDPs and primal-dual / Lagrangian methods

**Core literature**

- *Constrained Markov Decision Processes*, E. Altman, 1999 (Chapman & Hall/CRC).
  https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf.
  - In occupancy-measure space a CMDP is a linear program, and its optimal policy may need to be **stochastic**.
  - Strong duality holds.
- *Constrained Reinforcement Learning Has Zero Duality Gap*, Paternain, Chamon, Calvo-Fullana, Ribeiro, 2019 (NeurIPS).
  https://arxiv.org/abs/1910.13393.
  - Zero duality gap holds even under nonconvex policy parametrisation, given rich enough policies.
  - This justifies solving the dual.
- *Reward Constrained Policy Optimization (RCPO)*, Tessler, Mankowitz, Mannor, 2018 (ICLR 2019).
  https://arxiv.org/abs/1805.11074.
  - Multi-timescale primal-dual: the multiplier updates more slowly than the policy.
  - Converges to a feasible local optimum under standard stochastic-approximation assumptions.
- *Constrained Policy Optimization (CPO)*, Achiam, Held, Tamar, Abbeel, 2017 (ICML). https://arxiv.org/abs/1705.10528.
  - Trust-region updates with a bound on worst-case constraint violation per update.
  - Needs a cost critic and second-order steps.
- *Benchmarking Safe Exploration in Deep Reinforcement Learning* (Safety Gym; PPO-Lagrangian / TRPO-Lagrangian),
  Ray, Achiam, Amodei, 2019. https://cdn.openai.com/safexp-short.pdf.
  - Lagrangian PPO is a strong, simple baseline, and often better than CPO in practice.
- *Responsive Safety in Reinforcement Learning by PID Lagrangian Methods*, Stooke, Achiam, Abbeel, 2020 (ICML).
  https://arxiv.org/abs/2007.03964.
  - A plain gradient-ascent multiplier is an integral controller, so it oscillates and overshoots.
  - Adding proportional and derivative terms damps both.
- *Natural Policy Gradient Primal-Dual Method for Constrained MDPs*, Ding, Zhang, Başar, Jovanović, 2020 (NeurIPS).
  https://arxiv.org/abs/2003.02189.
  - O(1/√T) rates for the *average* iterate.
  - Later work on last-iterate convergence: *Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained
    MDPs*, Ding, Wei, Yang, Wang, Jovanović, 2023, https://arxiv.org/abs/2306.11700.
- *State Augmented Constrained Reinforcement Learning: Overcoming the Limitations of Learning with Rewards*,
  Calvo-Fullana, Paternain, Chamon, Ribeiro, 2021/2023. https://arxiv.org/abs/2102.11941.
  - Some constrained optima are **not the maximiser of any fixed weighted reward**.
  - So a fixed-μ Lagrangian policy can fail to satisfy the constraint.
  - Fix: put the multipliers in the state.
- *Sauté RL: Almost Surely Safe RL Using State Augmentation*, Sootla et al., 2022 (ICML). https://arxiv.org/abs/2202.06558.
  - Per-trajectory budgets via state augmentation.
- *Risk-Constrained Reinforcement Learning with Percentile Risk Criteria*, Chow, Ghavamzadeh, Janson, Pavone, 2017
  (JMLR). https://arxiv.org/abs/1512.01629.
  - CVaR and chance constraints instead of expectations.

**Guarantees and known failure modes**

1. **Oscillation.** The multiplier integrates the constraint error, so the policy overshoots, then the multiplier
   overshoots back. Stooke et al. show this and damp it with PID terms. Convergence guarantees are mostly for averaged
   iterates (Ding et al. 2020). The last iterate, which is what we deploy, can cycle.
2. **Satisfaction only in expectation.** The constraint E[c] ≤ ε holds over the training distribution, not per task.
   Calvo-Fullana et al. show that the fixed-μ policy at the dual optimum can violate it, and that time-sharing or
   mixing between policies is needed.
3. **Small batches.** Every multiplier step uses a noisy batch estimate of E[c]. The two-timescale proofs (RCPO) assume
   many policy steps per multiplier step. A ~200-step budget does not have that luxury.
4. **No regularisation means a non-unique primal.** With a KL regulariser of strength β_KL, the Lagrangian's maximiser
   is unique: π_μ ∝ π_ref · exp((r − μ·c)/β_KL). Primal recovery is then well posed. This is the basis of one-shot
   dualisation (Huang et al. 2024, section b). The repo trains GRPO with `beta: 0` (`docs/design.md` line 488), so it
   gets no such uniqueness.

**What it implies here (quantitative, derived)**

- **Constraint-estimate noise per step.** Take 64 episodes per step (8 × 8). About half are feasible, which at G2's
  91.5 % feasible claim rate (`docs/iteration-18-results.md` "Guards") gives ~29 claims. With a false-claim rate of 0.2,
  the SE of the per-step rate is √(0.2·0.8/29) ≈ 0.074. For a target ε = 0.10, one step's estimate has an SE of 74 % of
  the target. Twin damage is no better: ~24 twin episodes at δ = 0.1 give SE ≈ 0.061.
- A gradient-ascent multiplier on per-step estimates is therefore **noise-driven** over the whole budget.
  Windows of ~10 steps cut the SE to about 0.023. At ~12.5 min per GRPO step including boots (iteration 12,
  `docs/iteration-12-design.md` line 277), a 200-step run is about 42 h. At $1.1-1.9/h that is $46-80, which is
  **the whole balance**. The 5090 step time is unknown.
- Realistic runs are 30-60 steps, which allows at most 3-6 multiplier updates. Conclusion: **solve the dual offline,
  from banked rollouts of the reference policy, before training**. Then refine it at most once or twice between stages,
  with a PID controller over windows. Do not use an online gradient-ascent μ.
- **The expectation constraint is less of a problem here than in robotics.** The Neyman-Pearson structure of §0 makes
  the Lagrangian decision a per-episode threshold (claim iff q(x) > μ/(1 + μ)). The expectation fixes only the
  threshold's level. The residual danger is pooling across families: a low-μ pool can be met by being very careful on
  easy families (revoke_permission) while fabricating on hard ones. **Use per-family multipliers** (about 5 families).
  The repo already reports per family.
- **The costs are exact indicators**: grader state diffs, not learned cost models. That removes the cost-model hacking
  failure of Safe RLHF-style methods. It is a real advantage of this project and should be stated as one.

---

## (b) Constrained and safe RL for LLMs

- *Safe RLHF: Safe Reinforcement Learning from Human Feedback*, Dai, Pan, Sun, Ji, Xu, Liu, Wang, Yang, 2023 (ICLR 2024).
  https://arxiv.org/abs/2310.12773.
  - Separate reward and cost models, combined by a PPO-Lagrangian.
  - Reports multiplier oscillation and a need for careful λ initialisation.
- *Confronting Reward Model Overoptimization with Constrained RLHF*, Moskovitz, Singh, Strouse, Sandholm, Salakhutdinov,
  Dragan, McAleer, 2023 (ICLR 2024). https://arxiv.org/abs/2310.04373.
  - Constraint thresholds are set at each component reward's "proxy point", and the multipliers are learned.
  - Shows that fixed weights over-optimise one component.
- *One-Shot Safety Alignment for Large Language Models via Optimal Dualization*, Huang, Li, Dobriban, Bastani,
  Hassani, Ding, 2024 (NeurIPS). https://arxiv.org/abs/2405.19544.
  - With a KL-regularised objective, the dual function has a closed form in expectations under π_ref.
  - So μ* can be **pre-computed from reference samples**, and the training is a single unconstrained run with a fixed μ*.
  - This matches our budget best.
- *Enhancing Safety in RLHF via Rectified Policy Optimization (RePO)*, Peng et al., 2024. https://arxiv.org/abs/2410.19933.
  - Expected-cost constraints let some prompts stay unsafe.
  - RePO penalises only prompts that violate, as a rectified per-prompt penalty, and targets near-every-prompt safety.
- *Stepwise Alignment for Constrained Language Model Policy Optimization (SACPO)*, Wachi, Tran, Sato, Tanabe, Akimoto,
  2024 (NeurIPS). https://arxiv.org/abs/2404.11049. Stepwise (DPO-style) alignment under a safety constraint.
- *Constrained Group Relative Policy Optimization*, Girgis, de Schaetzen, Rowe, Robitaille, Pal, Paull, 2026.
  https://arxiv.org/abs/2602.05863.
  - The constrained GRPO paper closest to ours: indicator costs, target behaviour rates, learned multipliers.
  - Key point: standardising the **sum** of reward and costs within a group distorts the Lagrangian, because component
    standard deviations differ.
  - They standardise each component in the group, then scalarise ("scalarised advantages").
- *GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization*, NVIDIA, 2026
  (ICML 2026). https://arxiv.org/abs/2601.05242.
  - Group-normalising a summed reward collapses distinct reward combinations to the same advantage.
  - They normalise per reward, then normalise the batch.
- *Abstention as an Action Can Kill Both the Reward Gradient and the KL Anchor: Collapse Law and Repair for
  Error-Penalized Reinforcement Learning*, Che, Yuan, Zhao, Yu, 2026. https://arxiv.org/abs/2608.00301 (cited in the
  repo as "arXiv 2608.00301", `docs/design.md` line 346). The paper's points:
  - Chow's threshold is λ/(1 + λ).
  - When abstention is a discrete action, the reward gradient and the KL restoring force share one gate-saturation
    factor, so both vanish together and training drifts to refusing everything.
  - Training reward rises toward 0 like 1/t, which looks like improvement.
  - Under sparse answers, group normalisation turns every designed penalty into 1, which moves the threshold to 1/2.
  - Their repair: a mandatory confidence report trained with a proper score plus a correctness reward, and abstention
    by thresholding at deployment.

**What it implies here**

- **Normalisation.** The repo's `scale_rewards: none` (`docs/design.md` line 347) is right for one scalar reward:
  it keeps α a threshold. But it is not enough once constraints are added with multipliers:
  - The false-claim, harm and over-refusal components have very different variances: harm is rare, claims are common.
  - Summed raw, the rarest constraint's signal is diluted by group variance from the other components.
  - Follow C-GRPO/GDPO: compute a group-centred advantage **per component** (centring only, no std scaling, to keep
    the threshold), then combine with the multipliers. That combination is the Lagrangian's gradient, unbiased per
    component.
- **Che et al.'s repair does not transfer as stated.** In QA the outcome is exogenous. In our agent tasks it is
  endogenous, so a proper score on y pays quitting (§0, and `docs/iteration-17-design.md` §2). The transferable part is
  the *architecture*: the confidence report is always emitted, and abstention at deployment is a threshold. The scoring
  must be cut off from the actions (formulation F1 below).
- **RePO's per-prompt rectification** fits twins: penalise the harm cost only on prompts where the group's damage rate
  exceeds δ. This keeps the gradient focused on the families still at risk.
- **One-shot dualisation is implementable on the Mac.** `bank.rescored_outcome` already rescores banked episodes under
  a new reward (`docs/design.md` line 610). The dual D(μ) = β_KL·log E_{π_ref}[exp((r − μ·c)/β_KL)] + μ·ε can be
  minimised over the it16b and it17 G2 banks (855 + dev episodes) at zero GPU cost. Caveat:
  - The banks hold S = 2-4 samples per task, and with small β_KL a few episodes dominate the log-sum-exp.
  - Report the effective sample size of the exponential weights. If it is below ~50, fall back to the threshold
    mapping of §0: pick μ so that G2's banked claims would hit ε under a q proxy.

---

## (c) RL for honesty, abstention and "I don't know"; pathologies

**Methods**

- *R-Tuning: Instructing Large Language Models to Say "I Don't Know"*, Zhang et al., 2023 (NAACL 2024).
  https://arxiv.org/abs/2311.09677.
  - SFT on uncertainty-aware data, labelled by whether the model's own answers were correct.
  - This is the SFT analogue of our "own-success anchors".
- *Teaching Models to Express Their Uncertainty in Words*, Lin, Hilton, Evans, 2022 (TMLR). https://arxiv.org/abs/2205.14334.
- *Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of LLMs*, Stangel et al., 2025.
  https://arxiv.org/abs/2503.02623. A log-scoring reward on verbalised confidence.
- *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR)*, Damani, Puri, Slocum, Shenfeld,
  Choshen, Kim, Andreas, 2025. https://arxiv.org/abs/2507.16806.
  - Correctness + Brier in GRPO; reported HotpotQA ECE falls from 0.37 to 0.03 with accuracy kept.
  - The answer is exogenous, so no quitting incentive arises.
- *TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning*, Wei et al. (Meta), 2025.
  https://arxiv.org/abs/2509.25760.
  - Ternary reward: +1 correct, 0 abstain, −1 wrong.
  - Beats binary and more elaborate rewards, and finds that abstention-encouraging schemes become over-conservative.
- *Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning*, Wu, Liu, Zeng, Zhan, Cai, Huang,
  2025 (ByteDance Seed / CMU). https://arxiv.org/abs/2512.19920.
  - Abstain when calibrated P(correct) is below a **user-specified** risk threshold.
  - Compares explicit thresholding, verbalised confidence with proper scores, and a critic-value approach.
  - Uses a 4B model, Qwen3-4B-Instruct.
- *Why Language Models Hallucinate*, Kalai, Nachum, Vempala, Zhang, 2025. https://arxiv.org/abs/2509.04664.
  - Binary grading makes guessing optimal.
  - They propose **explicit confidence targets** in the instructions ("answer only if > t confident; mistakes cost
    t/(1 − t)"), which makes abstaining optimal below t.
- *AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions*, Kirichenko, Ibrahim, Chaudhuri, Bell, 2025.
  https://arxiv.org/abs/2506.09038. Reasoning fine-tuning degrades abstention by ~24 % on average.
- *Know Your Limits: A Survey of Abstention in Large Language Models*, Wen et al., 2024. https://arxiv.org/abs/2407.18418.

**Pathologies**

- Quitting / over-refusal. TruthRL notes it, and Che et al. 2026 formalise the collapse. In-repo evidence:
  - teacher SFT cut verified success from 69 to 22 % (`docs/results.md` iteration 11);
  - G3's control declines rose to 41 %;
  - it17's claims on infeasible tasks rose from 8.6 to 32.9 % (`docs/results.md`).
- **Reward hacking and test tampering.**
  - *ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases*, Zhong, Raghunathan, Carlini, 2025
    (ICLR 2026). https://arxiv.org/abs/2510.20270.
    - Impossible tasks are built by making the tests conflict with the spec.
    - Frontier models cheat often, e.g. GPT-5 54 % on Conflicting-SWEbench.
    - Prompting moves the rate a lot: 92 % to 1 % on the LiveCodeBench variant for GPT-5.
    - LLM monitors catch 86-89 % on LCB but only 42-65 % on SWE.
  - *Recent Frontier Models Are Reward Hacking*, METR (Von Arx, Chan, Barnes), 2025.
    https://metr.org/blog/2025-06-05-recent-reward-hacking.
    - o3 modified tests and scorers, and on one RE-Bench task hacked in every trajectory.
    - The detection methods disagree, so the counts are lower bounds.
  - *Natural Emergent Misalignment from Reward Hacking in Production RL*, MacDiarmid et al. (Anthropic, Redwood),
    2025. https://arxiv.org/abs/2511.18397.
    - Hacks learned in real coding RL generalise to broad misalignment.
    - Mitigations: prevent the hack, diversify safety training, or "inoculation prompting".
  - *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*, Baker et al. (OpenAI), 2025.
    https://arxiv.org/abs/2503.11926.
    - Putting a CoT monitor into the reward teaches obfuscated hacking.
- Specification gaming in general:
  - *Specification gaming: the flip side of AI ingenuity*, Krakovna et al., 2020 (DeepMind blog).
    https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/.
  - *Defining and Characterizing Reward Hacking*, Skalse, Howe, Krasheninnikov, Krueger, 2022 (NeurIPS).
    https://arxiv.org/abs/2209.13085. Shows that unhackable proxies are rare.

**What it implies here**

- **The QA calibration literature does not transfer directly.** RLCR, Rewarding Doubt and TruthRL all assume the
  outcome is fixed before the report. In terminal tasks the agent makes the outcome. Proper scores then become
  performative (§0). The repo found this on paper (`docs/iteration-17-design.md` §2), and it17/18 found the coupling
  empirically through shared weights: it17 damaged 53 of 120 twins against 13, and it18 left failing claims at
  confidence 0.98 (`docs/iteration-18-results.md` "What happened"). **The novelty is in this gap.** It is stated in
  none of the cited calibration papers. Che et al. come closest, for discrete abstention only.
- **Kalai's explicit target t** is a cheap lever the repo has not tried: put t in the prompt and match the penalty
  (formulation F3).
- **State grading makes tamper detection exact** (hashes of protected paths). ImpossibleBench's finding that LLM
  monitors miss 35-58 % of SWE cheating is a strong argument for the state grader. The main hacking risk is a grader
  gap, not judge gaming.
- **Use ImpossibleBench-style pressure in training, held out from the benchmark.** The family is conflicting tests
  inside the sandbox, where the only clean outcome is "abstain, naming the conflict". It has to be built from training
  sources and kept disjoint from LCB-309 (the final benchmark uses ImpossibleBench). That is the clean test of
  "cheating under optimisation pressure", one of the owner's four RL-only targets.
- **Do not put an LLM judge or CoT monitor into the reward** (Baker et al.). The current design already avoids it.

---

## (d) When to claim vs abstain, and incentive-compatible "keep trying vs admit failure"

**Selective prediction**

- *On Optimum Recognition Error and Reject Tradeoff*, C. K. Chow, 1970 (IEEE Trans. Information Theory 16(1)).
  https://doi.org/10.1109/TIT.1970.1054406.
  - The Bayes-optimal rule rejects when the max posterior is below 1 − c, where c is the reject cost relative to error.
- *On the Foundations of Noise-free Selective Classification*, El-Yaniv & Wiener, 2010 (JMLR).
  https://jmlr.org/papers/v11/el-yaniv10a.html. The risk-coverage trade-off.
- *Selective Classification for Deep Neural Networks*, Geifman & El-Yaniv, 2017 (NeurIPS).
  https://arxiv.org/abs/1705.08500. Picks the threshold to guarantee a target risk with high probability.
- *Learning with Rejection*, Cortes, DeSalvo, Mohri, 2016 (ALT). https://cs.nyu.edu/~mohri/pub/rej.pdf.
- *Strictly Proper Scoring Rules, Prediction, and Estimation*, Gneiting & Raftery, 2007 (JASA).
  https://doi.org/10.1198/016214506000001437. The Savage representation, with G convex.

**Performativity**

- *Performative Prediction*, Perdomo, Zrnic, Mendler-Dünner, Hardt, 2020 (ICML). https://arxiv.org/abs/2002.06673.
- *Incentivizing Honest Performative Predictions with Proper Scoring Rules*, Oesterheld, Treutlein, Cooper, Hudson,
  2023 (UAI). https://arxiv.org/abs/2305.17601.
  - When a prediction influences its outcome, proper scoring rules reward self-fulfilling, extreme predictions.
  - Honesty needs scoring with the influence removed.

**Shaping and baselines**

- *Policy Invariance under Reward Transformations*, Ng, Harada, Russell, 1999 (ICML).
  https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf.
  Potential-based shaping F = γΦ(s') − Φ(s) preserves optimal policies.
- *Counterfactual Multi-Agent Policy Gradients (COMA)*, Foerster et al., 2018 (AAAI). https://arxiv.org/abs/1705.08926.
  A baseline that marginalises one agent's action while holding the rest fixed.
- *Hindsight Credit Assignment*, Harutyunyan et al., 2019 (NeurIPS). https://arxiv.org/abs/1912.02503.
- *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback (RLOO)*, Ahmadian et al.,
  2024. https://arxiv.org/abs/2402.14740.
- *Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)*, Liu et al., 2025.
  https://arxiv.org/abs/2503.20783. Std normalisation and length normalisation bias GRPO.

**What it implies here**

1. **The claim decision is a full-information bandit (derived, a central structural fact of this project).**
   - The grader computes y from the final state whatever the report says.
   - So for every episode the payoff of *every* report option is known: R(claim, y), R(incomplete, y) and
     R(abstain + blocker, y, m).
   - The report-token advantage can be the exact counterfactual A_rep = R(r_chosen, y) − Σ_r π(r | h)·R(r, y): a COMA
     baseline with the trajectory held fixed. It has no variance from other rollouts.
   - The action tokens can then be credited with R̄(y) = Σ_r π(r | h)·R(r, y) or max_r R(r, y), so report noise does not
     leak into action credit.
   - This separation is unavailable in QA RL, where the report is the answer. It is a clean, theory-backed
     contribution.
2. **Incentive compatibility needs V(q) nondecreasing (§0).**
   - The current table satisfies it: λ > 0 makes working beat quitting, `docs/design.md` line 345.
   - "One-command quitting" remains when budget exits are likely (`docs/design.md` lines 358-363).
   - Any calibration term must go through the report-only gradient path (1), never through R̄.
3. **Stop values are observable.**
   - Because grading is on state, the stopping payoff S_t = max_r R(r, y_t) at an intermediate state can be computed by
     running the hidden tests on a snapshot.
   - "Keep trying vs admit failure" is an optimal-stopping problem with a known stop payoff. The advantage of
     continuing is V_cont(s_t) − S_t.
   - Φ(s) = S(s) is a natural potential: Ng-style invariant shaping that rewards each step's change in "what could be
     banked now".
   - Cost is unknown and probably high: each grading needs a sandbox snapshot plus a hermetic pytest run. Docker
     commit or CRIU under Harbor is untested in this repo. Use it at 1-3 sampled points per episode, not every step.
4. **Neyman-Pearson framing of twins and controls.**
   - Maximise the twin decline rate (power) subject to the control decline rate ≤ κ (size).
   - The optimal test is a threshold on the likelihood ratio, i.e. on the evidence.
   - A policy that ignores the evidence can only trade power for size along the diagonal.
   - AUROC above 0.5 can only come from using the evidence, so the constraint set must pin the diagonal direction to
     zero gradient (§0 caution-level calculation).
5. **Threshold elicitation.**
   - A policy that behaves optimally for *every* t in a family of announced thresholds must rank episodes by its own q.
   - So claim behaviour across t reveals an ordering: a menu of scoring rules elicits the quantile function.
   - Randomising t during training is the cheapest way to force the policy to compute q rather than a fixed caution
     level (formulation F3).
   - Evidence that models ignore stated penalties: *Are LLM Decisions Faithful to Verbal Confidence?*, 2026,
     https://arxiv.org/abs/2601.07767 (title confirmed in search; authors not verified here).

---

## (e) Long-horizon agent RL with sparse outcome rewards; RL vs SFT generalisation

**Credit assignment and branching**

- *VinePPO: Refining Credit Assignment in RL Training of LLMs*, Kazemnejad et al., 2024 (ICML 2025).
  https://arxiv.org/abs/2410.01679. Monte Carlo value estimates from intermediate states replace PPO's critic.
- *Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)*, Feng, Xue, Liu, An, 2025 (NeurIPS).
  https://arxiv.org/abs/2505.10978.
  - Step-level groups from repeated "anchor states" across a group's trajectories, at no extra rollout cost.
  - +12 % on ALFWorld and +9 % on WebShop over GRPO.
- *Tree Search for LLM Agent Reinforcement Learning (Tree-GRPO)*, Ji et al., 2025. https://arxiv.org/abs/2509.21240.
  - Tree rollouts share prefixes, so there are more rollouts per token budget.
  - The intra-tree advantage is equivalent to step-level preference learning.
- *ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL*, Zhou, Zanette, Pan, Levine, Kumar, 2024.
  https://arxiv.org/abs/2402.19446.
- *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn RL*, Wang et al., 2025.
  https://arxiv.org/abs/2504.20073. The "echo trap": reward variance collapses in multi-turn GRPO.
- *Approximately Optimal Approximate Reinforcement Learning*, Kakade & Langford, 2002 (ICML).
  https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf.
  - Policy improvement guarantees depend on the restart distribution's coverage of the states that matter.
  - *On the Theory of Policy Gradient Methods*, Agarwal, Kakade, Lee, Mahajan, 2021 (JMLR),
    https://arxiv.org/abs/1908.00261, gives the distribution-mismatch coefficient.
- *Efficient Reductions for Imitation Learning (DAgger)*, Ross, Gordon, Bagnell, 2011 (AISTATS).
  https://arxiv.org/abs/1011.0686. This is the repo's decision-point SFT logic.

**RL vs SFT generalisation**

- *SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training*, Chu et al., 2025 (ICML).
  https://arxiv.org/abs/2501.17161.
  - On rule and visual variants, outcome-RL transfers and SFT memorises.
  - SFT is still needed first, to stabilise the output format.
- *RL's Razor: Why Online Reinforcement Learning Forgets Less*, Shenfeld, Pari, Agrawal, 2025 (ICLR 2026).
  https://arxiv.org/abs/2509.04259.
  - Forgetting is predicted by KL(fine-tuned ‖ base) **on the new task's distribution**.
  - On-policy RL is implicitly biased to KL-minimal solutions.
- *Reinforcement Learning Finetunes Small Subnetworks in Large Language Models*, Mukherjee et al., 2025.
  https://arxiv.org/abs/2505.11711.
- *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, Yue et al., 2025.
  https://arxiv.org/abs/2504.13837.
  - RLVR raises pass@1 but narrows pass@k at large k: it reweights the base model's support.

**What it implies here**

- **RL sharpens what is sampled; it does not create judgment outside the support.** Yue et al., and the repo's own
  "reward cannot teach what is never sampled", say the same.
  - G2 discriminates when it reads: twins declined 72 % vs controls 25 % (`docs/iteration-15-g3-rl-read.md`
    "Diagnosis").
  - It does not discriminate when it does not read: 36 % vs 37 %.
  - The support therefore contains the right behaviour, "read, then decide". The RL target is the *read habit and the
    blind prior*, a multi-step habit, which is exactly the owner's RL scope.
  - Any transfer claim should be scoped to "makes existing discrimination reliable", not "creates discrimination on
    unfamiliar evidence".
- **RL's Razor plus it17/18.**
  - Forgetting tracked action KL: it17 at 0.18 nats per token destroyed 53/120 twins; it18 at 0.009 destroyed 31/120
    (`docs/iteration-18-results.md`).
  - Off-policy calibration training (on fixed G2 transcripts) is SFT-like in this sense.
  - On-policy RL with the action reward live should drift less per unit of gain. That is testable, and it is the
    project's argument for doing calibration *inside* RL with decoupled advantages rather than as a separate bandit.
- **Credit assignment for "read before destroy".**
  - Within a GRPO group on one twin, trajectories split at the first read-or-mutate decision.
  - GiGPO's anchor-state grouping applies almost for free: the prefix up to that decision is often identical in
    commands, and "same command history" is a usable state key in a deterministic sandbox.
  - Tree-GRPO/VinePPO branching from the model's own **decision states** gives a direct advantage for the read action.
    The repo already extracts those states for SFT (iterations 13-14).
  - Restarts also shorten rollouts: one cut late in a 14-command episode leaves ~5 commands. That is the only route
    to ≥100 effective steps on a ~$74 budget.
- **Mapping to the owner's four RL targets:**
  - no one can write the target: the claim at the agent's own competence (F1, F3);
  - competence-dependent decisions: keep trying vs admit (stop values, F4);
  - multi-step habits: read → decide (F2 + F4 branching);
  - cheating under pressure: conflicting-test twins with exact tamper costs (F2).

---

## Candidate formulations

Shared base for all four: start from G2 seed 1, LoRA, on-policy GRPO, `scale_rewards: none`, state-graded y/h/tamper,
and per-family reporting. The KL to G2 is over **all** generated tokens (`docs/results.md` lesson "A regularizer only
protects the tokens it covers"). Each formulation names the gap it targets, its justification and the strongest
objection.

### F1. Actor-reporter decoupling with counterfactual report advantages ("performative-proof honesty")

**What.**
- Split the per-token advantage by role:
  - action tokens (bash calls, the abstain-before-acting decision) get A_act = R̄(y, h) − baseline, where
    R̄ = Σ_r π(r | h)·R(r, y): the report marginalised out;
  - report-status tokens get the exact counterfactual A_rep = R(r, y) − Σ_r π(r)·R(r, y) (§d.1);
  - confidence-value tokens get a proper-score advantage (Brier) computed with the it17 exact estimator
    (`docs/iteration-17-design.md` §3) inside the RL step.
- The stop-gradient through y means actions are never paid for certainty.
- Option (F1b): generate the confidence and the status with a **separate reporter LoRA**. A second vLLM call at the
  report cut point is exactly it17's prompt. Or use a frozen-trunk probe head (Wu et al. 2025's "critic value").
  Then report gradients cannot move action weights at all, and abstention at deployment is a threshold on the reporter
  (Che et al.'s repair, made performative-proof).

**Theory.**
- The partial gradient is the gradient of a two-player game: an actor maximises verified success, and a reporter
  minimises a proper score *given* the actor's distribution.
- Its fixed point is a Nash equilibrium in which the reporter is calibrated and the actor has no incentive to quit
  (V nondecreasing, §0).
- The honest-performative-prediction literature (Oesterheld et al. 2023) shows this "influence-removed" scoring is
  what makes proper scores incentive compatible.

**Gap targeted.** The 1-in-5 false claims at confidence ~0.98, and the it17/18 coupling.

**Strongest objection.**
- In F1a the role split is in the advantages but the **weights are shared**. it18 is direct evidence that lowering
  confidence on failing claims and changing actions are not separable in one LoRA: the anchor that held the actions
  also killed the calibration.
- The answer is that in F1a the actions carry their own live task and harm gradients, not just a KL. Unknown until
  measured.
- F1b removes the objection, but the reporter is then a post-hoc component, and "the agent is honest" weakens to
  "the system is honest".
- Second objection: the claim *decision* by threshold needs the reporter's AUROC to beat the no-skill baseline enough
  to help. it17 reached 0.77 with drift; the other-sample predictor reaches 0.79-0.84
  (`docs/iteration-17-design.md` line 69).
- Cheapest falsification: F1b with a linear probe on G2's hidden states at the report cut, on the dev set's 620 claims,
  on CPU/Mac or a single GPU hour. If the probe AUROC is ≤ 0.70, the reporter route has little to offer.

### F2. Neyman-Pearson Lagrangian GRPO with pre-solved duals and caution-neutral paired groups

**What.**
- Objective: verified success (pass ∧ claim).
- Constraints, each an exact grader indicator with its own multiplier:
  - false claims among claims ≤ ε (per family);
  - twin damage ≤ δ;
  - control decline ≤ κ (over-refusal);
  - tamper as a hard cost.
- Advantages: centred per component, then combined (C-GRPO/GDPO, no std scaling).
- **μ\* solved offline** by one-shot dualisation on the G2 banks (Huang et al. 2024), with an ESS check.
- Refine once between stages with PID over ≥10-step windows (Stooke et al.).
- **Pair groups**: each step's twin and control of the same source are scored with a joint baseline. The weights are
  set so that the blind-policy gradient along "caution level" is ~0 at G2 (w_C/w_T ≈ 2 at the G3 config, §0).
- Optional RePO rectification: harm cost only on prompts whose group damage rate exceeds δ.

**Theory.**
- Zero duality gap (Paternain et al. 2019).
- The KL-regularised Lagrangian has a unique maximiser (well-posed primal recovery).
- With both caution-direction constraints active, KKT stationarity zeroes the diagonal of the ROC. The only ascent
  direction left is discrimination (the Neyman-Pearson lemma). This is the formal answer to "methods move a caution
  level".
- Hand-set α, β and false_blame become outputs of targets ε, δ, κ the owner can state in plain words.

**Gap targeted.**
- G3's caution rise in both branches (predicted quantitatively in §0);
- control over-refusal (41 % in G3, 21.8 % in it16c);
- fixed hand-set weights.

**Strongest objection.**
- Pinning the diagonal makes the gradient *zero for blind policies*. It does not make discrimination *reachable*: if
  reading is sampled rarely on unfamiliar surfaces (28-34 % on held-out surfaces, `docs/training-log.md` iteration 15
  after G2), the discrimination gradient is small and noisy, and 30-60 steps may produce nothing.
- The dual estimate from S = 2-4 banked samples per task may be badly biased, and the policy moves away from π_ref
  during training, so μ* goes stale.
- Mitigation: combine with F4's branching at the read decision, which concentrates samples there.

### F3. Threshold-conditioned claims: randomised announced risk targets t with matching penalties

**What.**
- Each training episode's prompt states a target: "claim complete only if at least t sure; a false claim costs
  t/(1 − t), an honest 'incomplete' costs 0". This is Kalai et al. 2025's explicit confidence target.
- t is drawn from {0.5, 0.7, 0.85, 0.95}. The reward implements the matching threshold, i.e. α_t = 2t − 1 in the
  repo's table (derived from p* = (1 + α)/2).
- Evaluation uses a fixed t (and the frozen benchmark's default prompt as a separate arm).

**Theory.**
- A policy optimal at every t claims iff q > t, so its claim set is nested in t.
- A policy that wants reward across thresholds must therefore carry an internal ranking by q. A fixed caution level
  cannot satisfy all t at once.
- It is a menu-of-scoring-rules elicitation that never pays a convex function of q to the actions. Each t's payoff is
  V nondecreasing (§0), so it stays effort incentive compatible.
- Behavioural AUROC can be read off claim behaviour across t, without trusting the verbalised number.
- Wu et al. 2025 report this route working at 4B on maths and QA.

**Gap targeted.** The 0.98 confidence on failing claims; it turns calibration into behaviour.

**Strongest objection.**
- A 4B model may ignore t (the 2026 "faithful to verbal confidence" result), and then F3 collapses to the base reward
  at the mean t.
- Also, t in the prompt changes the system prompt the frozen benchmark uses, a transfer confound
  (`docs/design.md` §3 on mechanism-neutral prompts). The default-prompt arm must be the headline.
- Cheap pre-test: zero-shot G2 at t = 0.5 vs 0.95 on the banked select tasks. If the claim rates differ by < 5 points,
  RL must create t-sensitivity from scratch. That is still plausible: each t gives a different correct answer on the
  same y, a dense contrast.

### F4. Restart-distribution RL at the model's own decision states, with branching and graded stop values

**What.**
- Rollouts start from G2's (later: the current policy's) banked prefixes at two kinds of decision state:
  - just before the first read-or-mutate decision on twins and controls;
  - late in feasible episodes, at the "claim / keep trying / admit" point.
- From each restart, sample a group of continuations, graded on the final state. This is a VinePPO/Tree-GRPO
  branching advantage at the decision. GiGPO anchor grouping covers full rollouts.
- At the late restarts, also grade the snapshot *at* the restart state. That gives the stop value S_t, so continuing is
  rewarded relative to what stopping would have banked (potential-based, Ng et al.).
- Mix in 30-50 % full rollouts from reset so the state distribution does not drift.

**Theory.**
- Kakade-Langford restart distributions bound policy-improvement error by coverage of the important states: the
  decision states are where the mismatch coefficient is worst for from-reset GRPO.
- Potential-based shaping preserves optimal policies.
- DAgger-style on-policy state collection keeps the restart states relevant, if the prefixes are refreshed from the
  current policy each stage.
- Unlike cut-point SFT, **no target is written**. The continuation is sampled and the grader decides. Competence-
  dependent choices (claim vs keep working) are learned from the policy's own success.

**Gap targeted.** The read habit (46 % flat under G3), "keep trying vs admit failure", and the step budget: shorter
rollouts mean more effective steps per dollar.

**Strongest objection.**
- It is close to what the owner said not to do ("reimplement cut-point SFT with RL"). It also needs **deterministic
  replay** of a banked prefix inside a fresh sandbox, which the repo has not built or verified: commands with time,
  randomness or network effects may diverge.
- Restarts from G2's prefixes are off-policy once the policy moves, and stale restart states can teach decisions at
  states the new policy never reaches.
- Mid-episode test runs for S_t have an unknown cost in Harbor.
- Cheapest falsification: the replay-fidelity rate of banked prefixes, measured on the remote node (not locally: ≤2
  local sandboxes). If it is below ~90 %, use GiGPO anchor grouping on full rollouts only, which needs no replay.

---

## Recommendation and order (derived; for the planner, not a decision)

1. **Zero-GPU checks first** (Mac, existing banks):
   - (i) the F1b probe AUROC on the dev set's 620 claims;
   - (ii) a one-shot dual solve for F2 with its ESS;
   - (iii) the caution-level slope of §0 recomputed on the actual it15 G3 rollouts, by family and read branch, as a
     check that the theory explains the past;
   - (iv) F3's zero-shot t-sensitivity, as one small inference job if no banked data fits.
2. **The main run combines F2 + F1a:**
   - the constrained objective with pre-solved per-family μ;
   - caution-neutral pair weighting;
   - per-component advantages;
   - report-token counterfactual advantages;
   - in-RL calibration on the confidence tokens only;
   - a full-token KL to G2.
   All of these are changes to the reward and advantage code, not to the rollout machinery, so they fit the current TRL
   trainer. Budget: 30-60 steps. That stage is ~$10-20 at it12's step time; the 5090 step time is unknown.
3. **F4 only if** replay fidelity is verified on the node. **F3** as a cheap arm if the zero-shot t-sensitivity is
   non-trivial.
4. **Contribution claim, if it works:** "constrained RL with exact state-graded costs, where honesty terms are made
   effort-incentive-compatible by (a) Neyman-Pearson constraints that pin the caution direction and (b) performative-
   proof (stop-gradient) scoring of the report". Both parts have short proofs (§0, §d) and a prediction the past data
   already fit (G3).

## Unknowns that would change this

- GRPO step time on the RTX 5090 32 GB, colocated vLLM with a 16k budget.
- Whether Harbor can snapshot a running sandbox for mid-episode grading.
- The replay fidelity of banked prefixes.
- The effective sample size of the one-shot dual on S = 2-4 banks.
- Whether a linear probe on G2 activations beats 0.70 AUROC on failing claims.
