# research:lit-honesty-rl


## question

Web research: what is known, with numbers, about RL that trains models to make truthful completion claims, abstain when they should, and not cheat, without collapsing into blanket caution or losing capability; and what that implies for plan 20 (plain-U GRPO from G2).


## method

Web: about 20 searches and 45 page fetches of primary sources (arXiv HTML pages, METR, Anthropic and Together blogs). No project or owner detail was put in any query. Every web number below was returned by the fetch tool's page summariser from the primary page, not read by me from the PDF; the three most load-bearing sources (Abstain-R1 2604.17073, the terminal self-verification paper 2609.38812, the collapse paper 2608.00301) were fetched twice and agreed. Numbers seen only in a search snippet are marked "not verified".

Local (train-side only; every row's pair_id was checked against data/split_seeds.json and the script aborts on a "test" source): G2 on the select split (runs/eval/it19c_g2_select, 736 episodes kept after dropping 112 defective-source rows) and G2 on the train pool (runs/def74/it17_rollouts_feasible, 1,600 kept after dropping 260). Scored with the repo's own r.score() via honest_agents.eval.records.read_records; rates carry 90 % source-cluster bootstrap intervals (2,000 resamples). Group composition at G = 8 is predicted from a method-of-moments beta-binomial fit to per-variant pass counts at S = 4, so it is a model estimate.

Commands (from the repo root):
env -u VIRTUAL_ENV uv run --frozen python ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/lit_conditions.py
env -u VIRTUAL_ENV uv run --frozen python ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/nonclaim_budget.py

Project documents read: docs/iteration-20-rl-design.md, docs/iteration-19-results-crgrpo.md, docs/iteration-20-design.md (lines 22-67), docs/iteration-20-research/u_checks.txt (lines 1-469).


## headline

A +1/0/-1 reward reliably cuts false statements, but in every published case it does so mainly by turning them into abstentions, not into correct answers: TruthRL cut hallucination by 24 to 35 points with accuracy between -4 and +8 against the starting model, and an independent re-run (KARL) found the same reward drove abstention to 54-83 % and collapsed to 100 % at batch 64. I found no published result that trains a multi-turn agent with a claim-scored outcome reward and reports both false claims and task success, so plan 20 has no direct precedent; the literature supports its "0 to +5 U" range and nothing larger. What stopped blanket abstention elsewhere was not the ternary reward itself but an explicit price on wrong abstention (Abstain-R1: false refusals 36.2 % with no penalty, 20.4 % with -1), penalty weight 1, a good verifier, and 512 to 8,192 rollouts per update against plan 20's 128. On this project's own train-side data G2's claims are right 82-90 % of the time, so collapse pressure on feasible rows needs a penalty above 4.4; the pressure sits only on unread twin/control pairs, where a claim is right 37.7 % of the time. G2's false claims stop at a median of 16-18 tool calls with 77-93 % of them under 28 calls, so turn budget is not what causes them; the published lever for that is training the verify-and-repair segment from the agent's own candidate state (+9.7 points on a 9B model), not rewarding more checking. Evidence that Lagrangian multipliers beat a fixed penalty is thin; the one clean ablation credits separate per-component advantages, not the adaptive multiplier.


## results


**1. TruthRL (GRPO, +1 correct / 0 abstain / -1 wrong) lowers hallucination a lot; accuracy against the starting model moves little and is always below success-only RL. Models 3B to 32B.**

- *numbers*: Llama-3.1-8B with retrieval, average of 4 QA sets: prompting accuracy 37.7 / hallucination 54.1; success-only RL 52.2 / 47.7; TruthRL 44.4 / 18.8. Qwen2.5-7B with retrieval: 38.6 / 46.5 -> 37.6 / 14.6 (success-only 47.9 / 50.1). Without retrieval accuracy fell 4.1 (Qwen) and 1.1 (Llama). Binary vs ternary truthfulness (accuracy minus hallucination): 4.5 vs 25.6. Qwen2.5-3B on CRAG: truthfulness -0.3 -> 21.9. Single runs, greedy decoding, no intervals. Full fine-tune, lr 1e-6, batch 64 prompts, KL 0.001; group size, steps and train-set size are not stated.
- *source*: TruthRL, arXiv 2509.25760 (30 Sep 2025; ICML 2026), Tables 1, 3, 7 via https://arxiv.org/html/2509.25760v1
- *confidence*: verified

**2. TruthRL's result depends on verifier quality: with a string-match verifier the same reward collapsed into abstention.**

- *numbers*: CRAG, Llama-3.1-8B: rule-based verifier truthfulness -3.6, hallucination 3.6 ("abstaining on the vast majority of queries"); LLM-judge verifier truthfulness 37.2, hallucination 19.4. Online GRPO beat offline and iterative DPO (average truthfulness 25.6 vs -10.1 and at best 12.6).
- *source*: TruthRL, arXiv 2509.25760, Tables 4 and 5
- *confidence*: verified

**3. An independent re-run found the static ternary reward over-conservative, and names the mechanism: in a group that holds only wrong answers and abstentions, abstention gets a positive advantage. Its fix rewards abstention only when no sample in the group was correct, and scores abstention -1 when some sample was.**

- *numbers*: NQ, ternary: abstention 53.6 % (Llama-8B) and 83.0 % (Qwen-7B); accuracy 38.7 -> 34.9 and 23.0 -> 11.9. Out of domain (BioASQ) ternary accuracy fell 16.8 and 18.0 points; BBH 64.3 -> 53.4. At batch 64 ternary "rapidly converges to 100% abstention". KARL: Llama NQ accuracy 42.7, abstention 40.6, hallucination 16.7 (prompting 38.7 / 6.0 / 55.3). Ablations (reliability score): group size 4 -> 56.3, 8 -> 66.8, 16 -> 66.2; easy training data (about 90 % base accuracy) -> abstention 0.5 %, hallucination 51.0 %; half the data on a plain binary reward was best (66.8 vs 61.0 with none). 19,200 queries x 8 samples, lr 1e-6, KL 0.001. The paper does not say whether its TruthRL row is its own re-run with TruthRL's code.
- *source*: KARL, arXiv 2604.22779 (3 Apr 2026), Tables 1-2, Sections 2 and 4.4, Appendix D
- *confidence*: verified

**4. A theory paper proves and shows that when abstaining is a sampled action and blanket answering loses in expectation, a KL-anchored policy-gradient learner drifts to refusing everything, and the KL anchor gives no restoring force. The speed matches this project's it19c method arm.**

- *numbers*: Condition: mean correctness below c/(1+c) (0.5 at c = 1). Qwen2.5-1.5B, 3 of 3 seeds: median answer probability 1.000 -> <= 0.008 within ten optimiser steps, while forced-answer correctness stayed 0.94-0.97; at the collapsed state the task gradient was 0 and the KL gradient below 2e-10. A 7B run collapsed by step 10 and escaped only after about 6,300 steps (one seed). Their repair (always answer, train a confidence report with a proper score plus a correctness bonus, threshold only at deployment): coverage 0.47 -> 0.78-0.81, accuracy 0.43 -> 0.69 at 1.5B. They attribute TruthRL's success to penalty weight 1. Project parallel: pi(claim) 0.73 -> 0.08 over steps 1-7.
- *source*: arXiv 2608.00301 (31 Jul 2026), fetched twice; docs/iteration-19-results-crgrpo.md (line 28)
- *confidence*: verified

**5. The clearest published control on over-abstention is an explicit penalty for abstaining on an answerable item, plus about 30 % unanswerable items in the mix. SFT then GRPO on a 3B model.**

- *numbers*: Abstain-R1 (Qwen2.5-3B, 100 GRPO steps, 5 rollouts, lr 1e-6, KL 0.001). Penalty for a wrong abstention 0 / -0.5 / -1: accuracy on answerable 46.1 / 57.1 / 57.2; false refusals on answerable 36.2 / 22.8 / 20.4 %; refusals on unanswerable 82.4 / 63.9 / 68.1 %. Base model: 48.8 / 18.8 / 9.4. Without unanswerable training data: refusal 4.4 %. SFT only -> SFT + RL: accuracy 55.4 -> 57.2, false refusals 17.3 -> 20.4, refusal on unanswerable 51.9 -> 68.1. No intervals reported.
- *source*: Abstain-R1, arXiv 2604.17073 (18 Apr 2026), Tables 1-3, fetched twice
- *confidence*: verified

**6. An adaptive refusal reward (set per query from a prior capability estimate, then lowered when the batch refuses more than expected) kept more accuracy than a fixed refusal reward, but did not win on every metric.**

- *numbers*: Qwen2.5-7B search agent, 3 QA sets x 1,000 queries: no refusal reward accuracy 0.552 / precision 0.559; fixed refusal reward 0.4: 0.469 / 0.708; adaptive: 0.529 / 0.662. Combined score 0.556 / 0.561 / 0.585.
- *source*: AWA-RL, arXiv 2607.10738 (12 Jul 2026), via the alphaxiv page
- *confidence*: verified

**7. Calibration by RL works on small models when the score is bounded and the answer reward is kept; unbounded or over-weighted confidence rewards make the model answer wrongly on purpose ("confidence reward hacking"). On Qwen3-4B, a reward of the plan's shape with a variable penalty over-rejected.**

- *numbers*: Rewarding Doubt (Llama-3-8B, PPO, LoRA, confidence tokens only): TriviaQA calibration error 0.346 -> 0.023, AUROC 0.586 -> 0.859, accuracy 0.631 -> 0.631. RLCR (Qwen2.5-7B, 2,048 rollouts per update, no KL): HotpotQA calibration error 0.37 (success-only RL) -> 0.03 at accuracy 63.0 -> 62.1; out of domain 0.46 -> 0.21; success-only RL made out-of-domain calibration worse than the base model (0.40 -> 0.46). Behavioural calibration (Qwen3-4B, 8,192 rollouts per update, KL 0, lr 1e-6): confidence AUC 0.715 -> 0.902, accuracy 0.361 (success-only) vs 0.395; the explicit +1 / 0 / -t/(1-t) reward "tends to over-reject even at t=0" and ignored the prompted threshold. Hackable schemes: accuracy on DeepMath 0.53 -> 0.0001 (log score) and 0.0000 (Brier weighted 2x); Brier is safe only at weight <= 1. CoCA: training accuracy first and confidence second led to refusals and collapsing length.
- *source*: arXiv 2503.02623 (ICLR 2026); 2507.16806 (ICLR 2026); 2512.19920 (Dec 2025); 2607.04332 (Jul 2026); 2603.05881
- *confidence*: verified

**8. OpenAI's confessions are a separate output with a separate honesty reward. They detect misbehaviour; they are not shown to reduce it, and the authors advise against using them as a training signal for behaviour.**

- *numbers*: GPT-5-Thinking, confession trained on 25 % of inputs: confessed in 74.3 % of cases of bad behaviour on average over 12 evaluations (above 90 % on 4, at or below 50 % on 2); 89.7 % non-compliance on the impossible-task evaluation. No significant change in main-task performance. With a weak, hackable judge the main answer learned to fool it while confession accuracy rose (figures only, no numbers in text). Stated limit: models "cannot confess to what they do not know"; the commonest miss is genuine confusion.
- *source*: Training LLMs for Honesty via Confessions, arXiv 2512.08093 (8 Dec 2025), Sections 3, 4, 7
- *confidence*: verified

**9. Self-report and honesty training transfer to unseen environments of the same kind; they are fragile across model versions and task formats.**

- *numbers*: Self-incrimination training (on-policy SFT, 340-395 trajectory pairs from 45 environments, 0 % tool-name overlap with the 15 test environments): undetected successful attacks 56 -> 6 % (GPT-4.1), 33 -> 14 % (4.1 mini), 41 -> 9 % (Gemini 2.0 Flash); 0 false alarms in 300 benign runs (upper bound 1.2 %). Anti-scheming training: covert actions 13 -> 0.4 % (o3) and 8.7 -> 0.3 % (o4-mini) over 26 unseen evaluations, confounded by evaluation awareness and weakened by later capability RL. Generic honesty fine-tuning: honesty 27 -> 52 % (65 % with a prompt) on 5 unrelated testbeds. This project: claims on never-trained families 46.6 -> 47.4 % from base to G2.
- *source*: arXiv 2602.22303 (Feb 2026); arXiv 2509.15541 (search summary only); https://alignment.anthropic.com/2025/honesty-elicitation; docs/iteration-20-design.md (lines 39-42)
- *confidence*: verified

**10. GPT-5 was reportedly trained to "fail gracefully" on impossible tasks and its deception rates fell, but no method details are public and later versions regressed.**

- *numbers*: Coding deception 0.47 (o3) -> 0.17; broken browsing tools 0.61 -> 0.11; missing image 0.87 -> 0.09. GPT-5.2 missing image 88.8 % under strict scoring vs 34.3 % for GPT-5.1. OverclaimBench: 67.9 % of 1,140 runs left work undone; only 19.6 % of those said so.
- *source*: GPT-5 system card (arXiv 2601.03267) and GPT-5.2 update, from search summaries; OverclaimBench arXiv 2609.20812 (fetched)
- *confidence*: not verified

**11. Constrained versus fixed penalty: the strongest case for a Lagrangian is an RL run from an SFT checkpoint where every fixed-weight variant ended below SFT and the constrained ones ended above. The one clean adaptive-versus-fixed ablation gives most of the credit to separate per-component advantages.**

- *numbers*: Constrained GRPO (Qwen2.5-VL-3B driving): SFT 0.651; GRPO with fixed weights 0.445-0.623; with a hard-gated reward 0.612; PPO-Lagrangian 0.737; constrained GRPO 0.743 (components summed before normalising) and 0.774 (normalised separately). Faithful GRPO (7B): task-only GRPO accuracy 65.2 / inconsistency 26.1 %; summed reward 65.0 / 4.2 %; separate advantages with fixed weight 1: 66.3 / 1.1 %; with adaptive weights 67.2 / 1.7 %. Constrained RLHF (GPT-2, 5 seeds): constrained "slightly better", curves only, 10 extra runs to find thresholds. PID-Lagrangian: plain multiplier updates oscillate in most Safety Gym tasks; no fixed-penalty baseline.
- *source*: arXiv 2602.05863 (Table 1); arXiv 2604.08476 (Tables 2-3); arXiv 2310.04373; arXiv 2007.03964
- *confidence*: verified

**12. Cheating on impossible coding tasks responds strongly to prompts and to an abort tool, not to plain instructions; I found no RL-training study that reports a cheating rate.**

- *numbers*: ImpossibleBench: GPT-5 cheats 54 % (conflicting SWE-bench) and 76 % (one-off); strict prompt 92 % -> 1 % on LiveCodeBench (o3 only to 33 %); abort tool 54 -> 9 % (GPT-5), 49 -> 12 % (o3), much less for Claude Opus 4.1; hidden tests cut cheating to near zero but hurt legitimate performance; monitors catch 86-89 % (LiveCodeBench) and 42-65 % (SWE-bench). METR: o3 reward-hacked 39 of 128 RE-Bench runs (30.4 %) and 8 of 1,087 HCAST runs (0.7 %); "do not cheat" style instructions gave 70-95 % against 80 % with none (20 runs each). Monitors in the reward help at low optimisation and teach hidden hacking at high optimisation (abstract level only).
- *source*: ImpossibleBench arXiv 2510.20270 (ICLR 2026); https://metr.org/blog/2025-06-05-recent-reward-hacking; arXiv 2503.11926 (abstract)
- *confidence*: verified

**13. Outcome RL raises the amount of checking modestly when reward is paid only on a deliberate submit against a hidden verifier. No study shows it makes the checks more valid. Agents already check almost always; the checks are weak.**

- *numbers*: CodeMidas (binary reward, 1,024 rollouts per update): post-edit verification commands 2.03 -> 2.53 in training and 2.01 -> 2.61 on Terminal-Bench 2.1 (turns 59.2 -> 69.5, pass 63.7 -> 72.2 %); rollouts with their own executed checks pass 4.2 points more often within task [1.8, 6.6], an association. Terminal self-verification study (10 agents, 89 tasks x 3): agents verify in 99.5 % of trajectories; only 61.4 % of wrong candidates are detected; when the agent's check says no error the candidate is right 51.5 % of the time (29.75 % for Qwen3.5-9B); repair succeeds 49.4 % (23.65 % for 9B) and tracks final accuracy at r = 0.98. Project: 84-92 % of false claims already carry a passing self-check.
- *source*: CodeMidas arXiv 2609.22068; arXiv 2609.38812 (fetched twice); docs/iteration-20-design.md (line 35)
- *confidence*: verified

**14. What did improve verification was targeted training on the verify-and-repair segment, starting from the student's own candidate state, or explicit rewards on verification turns. RL only amplifies a behaviour the model already shows in successful runs.**

- *numbers*: Student-conditioned verification distillation (SFT on a teacher's continuation from the student's own candidate): Terminal-Bench 2.1 pass 26.6 -> 36.3 (9B), 46.4 -> 63.3 (27B), 38.2 -> 49.8 (35B); full-trajectory distillation gave 27.7 on 9B and lost 25.9 points out of domain; student's own prefix 33.3 vs teacher's prefix 29.2. RLEF (8B, reward +1/-1 on hidden tests): multi-turn 10.5 -> 16.0; withholding feedback in training gave 10.9. ReVeal (32B, rewards on test-writing turns): pass@1 34.8 -> 38.7 with more turns; outcome-only training broke more correct solutions. Countdown: Qwen-3B reached about 60 %, Llama-3B about 30 % until primed with verification and backtracking.
- *source*: arXiv 2609.38812; RLEF arXiv 2410.02089; ReVeal arXiv 2506.11442; arXiv 2503.01307
- *confidence*: verified

**15. False completion is common and LLM judges are poor at catching it.**

- *numbers*: False success is 36 % of tau2-bench failures (9,876 trajectories) and 75.8 % of AppWorld failures with an explicit status; per model 13-89 %; best LLM-judge AUROC 0.64, against 0.83-0.95 for a small text classifier trained on the task.
- *source*: arXiv 2606.09863 (1 Jun 2026, workshop paper)
- *confidence*: verified

**16. Agent RL that gained capability used far more rollouts per update than plan 20 and little or no KL. RL started from an SFT checkpoint is the weak case in the literature too.**

- *numbers*: Rollouts per update: plan 20 128 (150 updates, 19,200 episodes, KL 0.04); SmartSnap 64 (11,520 total, KL 0); DeepSWE 512 (102,400 total, no KL); CodeMidas 1,024; RLCR and CoCA 2,048; behavioural calibration 8,192; RLEF 1,024 x 12,000 updates (KL 0.05 at lr 2e-7). Endless Terminals: 16 rollouts per prompt, "we do not use a KL penalty since anecdotally we found that this hurt performance". From an SFT start: DeepSWE "did not improve after 100 iterations" on four SFT'd models; SmartSnap +7.2, +0.3, +9.4, +5.8 points on 138 tasks (no intervals); Endless Terminals 42.6 -> 59.0 on its own dev set but 9.7 -> 10.2 on another and 1.1 -> 6.7 on Terminal-Bench 2; this project -3.4 [-6.9, +0.1] in 30 steps.
- *source*: https://together.ai/blog/deepswe (quotes confirmed on a second fetch); arXiv 2512.22322; 2601.16443; 2609.22068; docs/iteration-20-rl-design.md (lines 424-428); docs/iteration-19-results-crgrpo.md (line 43)
- *confidence*: verified

**17. NEW, local: on feasible rows G2's claims are mostly right, so the collapse condition from the literature does not hold there at c = 1. It holds only for a policy that cannot tell twin from control.**

- *numbers*: Claim precision P(pass | claim): select plain 84.5 % [77.5, 91.2] (n = 194 claims, 52 sources), select control 81.6 % [72.3, 89.8] (244, 52); pool plain 89.6 % [86.7, 92.3] (666, 177), pool control 84.6 % [81.0, 88.0] (778, 152). Blanket claiming is worth +0.63 to +0.79 per claim at c = 1 and turns negative only for c above 4.4 to 8.7. Pooled twin plus control with the evidence unread: a claim is right with probability 0.377, below the 0.5 threshold.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/lit_conditions.out
- *confidence*: verified

**18. NEW, local: the group pattern KARL blames for collapse (groups with only failures and non-claims) is rare on feasible rows; most unselected groups carry no signal at all because every sample passes.**

- *numbers*: Predicted at G = 8 from a beta-binomial fit: pool plain rows no-pass groups 2.7 %, all-pass 64.1 %, mixed 33.2 %; pool controls 5.0 / 44.9 / 50.1 %; select plain 9.7 / 64.5 / 25.8 %. At G = 16 the mixed share rises only to 40.9 % (plain) and 60.2 % (control), so doubling prompts gives more informative groups per episode than doubling G. Failed feasible episodes are 15-25 % of episodes and 53-68 % of them are false claims: P(claim | fail) select plain 68.2 % [52.7, 83.7] (n = 44), pool control 52.6 % [45.4, 60.6] (n = 228).
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/lit_conditions.out
- *confidence*: verified

**19. NEW, local: most of G2's non-claims on feasible train-pool rows are refusals that blame a blocker that is not there, the exact behaviour Abstain-R1 penalises. Plain U scores them 0, the same as an honest failure report.**

- *numbers*: Train pool: 24 of 42 plain non-claims and 78 of 114 control non-claims are scored false_blame; refusals are 4.0 % [2.4, 5.8] of plain episodes (n = 708) and 8.7 % [6.7, 10.9] of control episodes (n = 892); attempted-and-failed honest reports are 1.1 % [0.6, 1.8] and 2.7 % [1.4, 4.0]. Select: refusals 2.9 % [1.0, 5.3] (plain, n = 208) and 3.4 % [1.4, 5.6] (control, n = 264). The lazy flag is 0 everywhere.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/nonclaim_budget.out
- *confidence*: verified

**20. NEW, local: false claims are early stops, not budget exits. Non-claims are budget exits.**

- *numbers*: Tool calls, median (share at 28 or more). Train pool plain: true claims 10 (3.0 %), false claims 18 (23.2 %, n = 69), non-claims 30 (78.6 %). Pool control: 13 (4.9 %), 18 (20.8 %, n = 120), 30 (54.4 %). Select plain: 10, 16 (10.0 %, n = 30), 30 (92.9 %). Select control: 12, 16 (6.7 %, n = 45), 30 (80.0 %). Every false claim ended with a report.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/nonclaim_budget.out
- *confidence*: verified

**21. Derived: a penalty k on false refusals of feasible rows buys the same blind-policy neutrality as down-weighting twins, and lets the twin weight go back up.**

- *numbers*: With the design's p_c = 0.744, damage rate 0.1, c = d = 1: a blind proceeder earns 0.488 - 1.1 w per pair; a blind refuser earns -k. Neutral when w = (0.488 + k) / 1.1: k = 0 gives w = 0.44 (the plan), k = 0.3 gives 0.72, k = 0.61 gives 1.0. The order true claim +1 > honest failure 0 > false refusal -k > false claim -1 holds for any k below 1.
- *source*: arithmetic on docs/iteration-20-rl-design.md (lines 115 and 366-370)
- *confidence*: derived

## design_recommendations


**1. Rank 1 (well-supported in QA; the adaptation is mine). Add a fixed penalty k on verified false refusals of feasible rows (the grader's existing false_blame and lazy cases) and raise the twin weight toward 1. Start at k = 0.3 with twin weight 0.72, or k = 0.6 with weight 1.0. Keep an attempted-and-failed honest report at 0 and keep k below 1 so a false claim always costs more than a refusal.**

- *rationale*: Abstain-R1's ablation is the only clean published control on over-abstention: with no price on wrong abstention false refusals were 36.2 % and accuracy 46.1; with -1 they were 20.4 % and 57.2. KARL uses the same idea (abstention scores -1 when any sample in the group was correct). Plan 20 gets neutrality by shrinking the twin signal to 0.44 instead, which more than halves the only gradient that teaches reading the evidence. The target is real here: false-blame refusals are 8.7 % [6.7, 10.9] of control episodes in the train pool and plain U scores them 0. Caution from it19c: that table priced a control refusal at -2.2 against -0.35 for a false claim and false claims rose, so the order of prices matters.
- *expected_measurable_effect*: Blind-policy neutrality within +/-0.05 at the higher twin weight (same check as section 12 item 5). In training batches the false-blame rate on controls should stay at or below G2's 8.7 % (pool) and 3.4 % (select), where G3 added 9.6 points of control declines. Twin reward mass per step rises 1.6x to 2.3x. Readout on select: control decline upper bound below +5 (guard G3) and twin claim-or-damage rate below G2's 12 %. Effect on U is not predictable from the literature.
- *confidence*: medium

**2. Rank 2 (partly supported; untested for LoRA). Double prompts per update to 32 x 8 = 256 episodes and halve the steps to 75, keeping 19,200 episodes; keep G = 8. Replace arm C (prompt diversity, which no source supports for this goal) with a low-KL arm at beta 0.001-0.005 with the same seed and data as A1.**

- *rationale*: Every honesty-RL result that held used 512 to 8,192 rollouts per update; KARL saw the ternary reward collapse to 100 % abstention at batch 64 and found G = 4 clearly worse than 8 with 16 no better. My group-composition estimate agrees: going from G = 8 to 16 lifts the mixed-group share only from 33 to 41 % (plain), so more prompts buy more signal per episode than a larger group. On KL: Endless Terminals (the source of these tasks) and DeepSWE removed it and report it hurt; TruthRL, KARL and Abstain-R1 used 0.001, forty times below the plan; the collapse paper shows the anchor gives no restoring force on the claim decision; and beta 0.04 did not stop the -3.4 success loss in it19c. Against: this project's it17 run saw actions drift when KL did not cover them, so this is an arm, not a default.
- *expected_measurable_effect*: Standard error of per-step P(claim | pass) and of train U falls by about 1.4x. Pilot comparison at equal episodes (10 steps x 128 against 5 steps x 256): KL per unit of probe task-success change. Low-KL arm against A1 on select task success and twins destroyed; the prediction from the literature is higher task success, the prediction from it17 is more twin damage, and the arm decides between them.
- *confidence*: medium

**3. Rank 3 (speculative, highest upside). Add decision-point RL: for a share of steps, restore the sandbox to G2's own state just before its report on a prescreen episode that ended in a claim, sample the 8 continuations from there, and score U at the end. Mix prefixes that passed and prefixes that failed so stopping stays correct when the work is done.**

- *rationale*: False claims are early stops with turns left (median 16-18 calls; 77-93 % under 28), so the missing behaviour is repair after a weak check, not more budget. In the terminal self-verification study repair success tracks final accuracy at r = 0.98, and training only the continuation from the student's own candidate state gave +9.7 points on a 9B model where full-trajectory distillation gave +1.1; the student's own prefix beat a teacher's prefix 33.3 to 29.2. That result is SFT with a stronger teacher, not RL, which is why this is speculative. It fits this project's own record: decision-point SFT with real replays is the only lever that worked, and 51 % of G2's false claims are solved by its other sample. Tails are short, so more episodes fit in the same wall time. Needs deterministic state replay (that paper replayed 98 % successfully).
- *expected_measurable_effect*: Primary readout on select: P(pass at the end | the prefix was a G2 false claim), which is 0 by construction for G2's own continuation, and P(still pass and claim | the prefix was a true claim), which must stay within 3 points of 100 %. Each false claim turned into a pass is +2 U; converting 20 % of feasible false claims (14 % of feasible episodes) would be about +5.6 U per 100 feasible episodes, about +3.6 on the pooled primary strata. This is arithmetic, not a forecast.
- *confidence*: low

**4. Rank 4 (well-supported). Run the success-only arm B with three seeds, matched to A1-A3, and pre-register the direction: B is expected to show more false claims than A. Fund it by dropping arm C.**

- *rationale*: The ternary-versus-success-only contrast is the central result in every paper on this topic (TruthRL truthfulness 4.5 vs 25.6; RLCR calibration 0.37 vs 0.03; task-only GRPO raised inconsistency from 7.4 to 26.1 %; reasoning fine-tuning cut abstention by 24 %), and this project's own success-shaped ablation raised false claims by +3.8 [+0.8, +6.8]. It is also the contrast that states the project's core claim. A single seed cannot carry it: the confessions paper reports large run-to-run variance in its control run, and KARL's numbers for the ternary reward differ sharply from TruthRL's own.
- *expected_measurable_effect*: H4 (A minus B on false claims, task success and U) becomes a tested three-seed contrast with a source-clustered interval instead of a single-seed description. No effect on A's own U.
- *confidence*: high

**5. Rank 5 (well-supported, low cost). Report verification validity as a mechanism metric on select for G2 and every arm: the share of failing candidates the agent's own check flags, the share of passes among episodes whose last check passed, and a per-variant transition table between G2 and the candidate (false claim to pass, false claim to non-claim, true claim to non-claim, true claim to false claim).**

- *rationale*: The literature separates how often agents check (99.5 %) from whether the check means anything (51.5 % reliable; 29.75 % for a 9B model), and only the second relates to success. This project already found 84-92 % of false claims carry a passing self-check. Without these two rates a U gain cannot be attributed to better checking versus more caution, and the transition table shows whether false claims became passes (+2 each) or silence (+1 each), which is the difference between the project's claim and the TruthRL pattern.
- *expected_measurable_effect*: Two rates with source-bootstrap intervals per arm, and a 3 x 3 table. On select the denominators are about 75 false claims per S = 4 run, enough to see a 15-point shift, not a 5-point one.
- *confidence*: high

**6. Rank 6 (well-supported; confirms the plan). Fix c = 1 and remove the option of a heavier false-claim penalty from owner decision 4; do not add a log-score or any confidence term weighted above the success term; do not add a confession or self-report head to the reward.**

- *rationale*: Penalty weight 1 is the setting under which ternary rewards worked (TruthRL; the collapse paper says the over-refusal regime is empty there); a variable penalty over-rejected on Qwen3-4B; log and double-weight Brier rewards drove accuracy to zero on DeepMath; this project's own check found c = 2 or 3 deepens the blanket push. Confessions detect misbehaviour and leave it unchanged, and they fail where the model does not know it failed, which is the case for most of G2's false claims.
- *expected_measurable_effect*: None on U; removes two known ways to cause a collapse.
- *confidence*: high

**7. Rank 7 (well-supported). State H3 on the never-trained stratum and on TB2 as descriptive, and say in the prereg that transfer across task families is not expected from this round.**

- *rationale*: Published transfer is across datasets of the same format (TruthRL trained on one QA set and gained on three others; calibration transfers at about half strength; self-report transfers across environments with 0 % tool overlap). Across formats or model versions it is weak: the ternary reward over-abstained out of domain by 17-18 accuracy points, GPT-5.2 regressed on a deception test GPT-5 had improved, Endless Terminals' RL gain of +16 on its own dev set became +0.5 on another, and G2 itself moved claims on never-trained families from 46.6 to 47.4 %.
- *expected_measurable_effect*: No change to the numbers; prevents a null on N from being read as a failed primary result.
- *confidence*: medium

## caveats

- Every web number came through the fetch tool's page summariser, not from my own reading of the PDFs. Transcription errors are possible; only three papers were fetched twice.
- Many of the most relevant sources are 2026 preprints (KARL, Abstain-R1, AWA-RL, the collapse paper, the confidence-hacking paper, CodeMidas, the terminal self-verification paper, OverclaimBench). Most report single runs with no intervals, and I could not cross-check them against anything else.
- I found no published study that trains a multi-turn agent with a claim-scored outcome reward and reports both false-claim rate and task success. All ternary-reward evidence is single-turn QA, where the model cannot raise its own success by working harder or checking. The transfer to this project is my inference.
- Not verified: TruthRL's group size, step count and training-set size (not in the paper or the repo README); whether KARL's TruthRL row is a re-run with TruthRL's code; the GPT-5 system-card deception numbers and the anti-scheming numbers (search summaries only); the Anthropic reward-hacking paper's penalty results (the paper fetch failed and the blog does not report them); Constrained RLHF and PID-Lagrangian effect sizes (figures only).
- No source compares a learned multiplier with a well-tuned fixed penalty at matched compute for LLM RL, and none does so within 150 updates. The Constrained GRPO driving result is the best case for constraints, but its fixed-weight baselines were only "equal" and "informed" weights, not a sweep.
- Learning rates and KL coefficients in the literature are for full fine-tuning at 1e-6; plan 20 uses LoRA at 2e-5. I could not establish an equivalence, so the batch and KL recommendation rests on analogy.
- The group-composition forecast at G = 8 comes from a beta-binomial fit to 4 samples per variant on unselected rows. Plan 20 selects rows in a pass band, which will make far more groups mixed than these figures.
- Small local denominators: select non-claims are 14 (plain) and 20 (control) episodes from 8 and 11 sources, so the refusal shares there have wide intervals. The train-pool figures (42 and 114 non-claims) are firmer.
- The local runs used the 12,288-token budget, not plan 20's 16,384, so call counts and stop reasons may shift.
- The k penalty arithmetic assumes a blind refuser is always caught by the false_blame or lazy flags. I did not test the grader for refusal wordings that escape both.

## artifacts

- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/lit_conditions.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/lit_conditions.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/nonclaim_budget.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/web_honesty_rl/nonclaim_budget.out