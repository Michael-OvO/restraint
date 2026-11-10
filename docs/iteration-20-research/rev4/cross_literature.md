# cross:literature — external claims and prior art (joint stage, iteration 20 revision 3 → 4)

Scope: cross-examination of the three literature reports
(`~/Documents/honest-agents-archive/staging/analysis/it20_rev4/inputs/evidence_literature.md`,
`~/Documents/honest-agents-archive/staging/analysis/it20_rev4/inputs/research_lit-agentic-rl.md`,
`~/Documents/honest-agents-archive/staging/analysis/it20_rev4/inputs/research_lit-honesty-rl.md`)
and of the candidate mechanism in
`~/Documents/honest-agents-archive/staging/analysis/it20_rev4/inputs/orchestrator_notes.md` (section 6).

## 0. Method

- The earlier reports read every page through a page summariser. I did not. I downloaded each primary page with
  `curl` (generic client string, no project detail in any request), stripped the HTML to text, and searched the text
  for the exact figures. Every "confirmed" below means I saw the number in the page text, usually inside its table row.
  Scripts: `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/scripts/fetch.py`,
  `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/scripts/htmltext.py`,
  `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/scripts/slice.py`.
  Pages and text dumps: `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/dl/<name>/`.
- 32 primary pages searched in full text (papers, two blogs, one personal blog post), 3 released config files read,
  35 abstract pages read, and 31 more arXiv identifiers checked for existence and title only. 10 web searches for
  prior art (generic wording).
- Limits: values that exist only in figures were not read (TMax group-size ablation, DeepSWE compact-filtering
  ablation, VinePPO accuracy curves, Endless Terminals training curves). PDFs were not parsed. Arithmetic marked
  "derived" is mine.
- Local reads (read-only): `configs/base.yaml` (lines 18-25, 155) and
  `docs/iteration-20-rl-design.md` (lines 421-430), for the LoRA scale and
  learning rate only. No held-out path was opened.

## 1. Conclusions in plain language

1. No citation is fabricated. Every paper named in the three reports exists with the stated title, and 21 of the 22
   load-bearing claims I re-read carry the stated figures. One does not: the "beta 0.04 in continual GRPO" numbers
   were read from version 2 of a paper that is now at version 4, and the 4B result reversed (KL 0.04 now costs 0.7
   points of new-task accuracy and gains 3.7 points of final accuracy). That citation cannot carry "KL 0.04 is harmful".
2. Of the other 21, 7 stand as stated and 14 are correct in their figures but need a correction or a caveat about
   what they show. The ones that change the plan:
   TMax's 33.6 → 54.3 is from the base model with no SFT, with the checkpoint picked on the same benchmark; the
   same recipe on a strong start (27B, 74 % at the start) gained +3.2 ± 2.3 and +0.3 ± 2.5. The "+3.8" for online
   filtering needs a 0.3-0.7 pass-rate band; removing zero-variance groups alone gave +1.0. Abstain-R1's penalty cut
   false refusals but also cut correct refusals by 14 points. Endless Terminals never published its episode count.
3. Decision-point RL is not new. Six works branch rollouts from intermediate states of the agent's own trajectories
   (VinePPO, Tree-GRPO, DEEP-GRPO, Branching Policy Optimization, PivoARL, and SCVD as an SFT version). The credible
   ones report +0.5 to +5 points over whole-episode GRPO at equal rollout budget on search-agent and maths tasks. None
   is on Docker terminal tasks, none starts from an SFT checkpoint with LoRA at 4B, none measures an honesty outcome.
4. Every one of those works keeps whole-episode rollouts from the true start state in the objective, and all branch
   from fresh on-policy trajectories. Two report what happens otherwise: Tree-GRPO collapses when only the
   branch-level advantage is used with few branches; DEEP-GRPO loses 1.8 points when the branch objective is weighted
   10x. The candidate as written (prefixes replayed from stored G2 episodes) has no precedent for the stale-prefix
   part. External evidence supports decision-point groups as one of two streams, not as the only objective.
5. The one paper that matches the candidate most closely (Branching Policy Optimization, sandbox snapshots, SWE-bench)
   fails a throughput sanity check by about two orders of magnitude. Its "+3.6 to +6.1 points" must not be cited.
6. Evidence that behaviour learned from mid-trajectory starts shows up in full episodes exists but is indirect: SCVD
   trained only the continuation after the student's own first complete candidate and gained +9.7 to +16.9 points on
   full Terminal-Bench 2.1 episodes, with right-to-wrong flips under 2 %. That is SFT from a much stronger teacher.
   No study trains by RL on branch states only and evaluates full episodes.
7. What I would bet: a two-stream arm (whole-episode groups plus decision-point groups, equal weight) beats a
   whole-episode-only arm on dev feasible-row U at equal sandbox time with probability 0.55, by +1 to +3 U if it
   does. A decision-point-only arm beats whole-episode with probability 0.25. Scale alone (50k-100k episodes) raises
   G2's in-distribution task success by 3 points or more with probability 0.35.

## 2. Part 1: spot-check of the load-bearing claims

Verdicts: C = confirmed as stated; CC = confirmed with a correction; NC = could not be confirmed; W = wrong.

| # | Claim (as used by the reports and notes) | Verdict | Exact figures seen in the source | Source URL |
|---|---|---|---|---|
| 1 | TMax took Qwen3.5-4B from 33.6 to 54.3 on Terminal-Bench Lite; 500 steps x 256 episodes; lr 1e-6; no KL; zero-variance filtering; fp32 LM head | CC | Table 3: 4B TB-Lite 33.6±1.8 → 54.3±1.9, TB 2.1 15.5±1.5 → 24.6±1.8. Table 13: training steps 500, 8 prompts x group size 32, lr 1e-6 constant, KL 0.0, filter zero-std true, active sampling true, LM head fp32 true, loss DPPO, max async steps 4, temperature 1.0. Corrections: (a) the loss is DPPO, not GRPO; (b) "When training Qwen 3.5 and 3.6 models, we do not perform any initial SFT"; (c) "we report scores on the checkpoint scoring the highest on Terminal-Bench Lite" (5 evaluations, so the TB-Lite gain carries up to about 2 points of selection optimism, derived; TB 2.1 is the unselected number); (d) all-zero group rate about 40 %, so more than 128k episodes were generated; (e) LoRA or full fine-tuning is not stated; (f) strong start: 27B 74.2±1.5 → 77.4±1.8 and 43.6±1.9 → 43.9±1.6 | https://arxiv.org/html/2606.23321 (v2, 2 Nov 2026) |
| 2 | Endless Terminals: PPO, 16 rollouts per prompt, 16 turns, no KL ("hurt performance"), lr 5e-7, 16 x 16 per step; dev 4.0 → 18.2, 10.7 → 53.3, 42.6 → 59.0; TB2 0.0 → 2.2, 2.2 → 3.4, 1.1 → 6.7 | CC | Paper text confirms every gain and "We do not use a KL penalty since anecdotally we found that this hurt performance", sequence-level loss, clip 0.2/0.28, temperature 0.6, 4 A100 about 2 days / 8 B200 about 8 hours. Configs confirm lr 5.0e-7, critic lr 1.0e-6, train_batch_size 16, n_samples_per_prompt 16, epochs 10, update_epochs_per_batch 2, gae, sequence_mean, use_kl_loss false, fsdp2. Corrections: steps and episodes actually run are not published anywhere (504,320 is a config maximum); on the OpenThinker dev set the SFT-start model moved only 9.7 → 10.2; whether the released configs are the paper runs is not stated | https://arxiv.org/html/2601.16443v3 ; https://raw.githubusercontent.com/kanishkg/endless-terminals/main/train/confs/base_qwen3_otak8.yaml |
| 3 | ECHO: GRPO 34.2 → 54.9, plus auxiliary loss 63.7 on Qwen3-8B; SFT-initialised 38.5 → 63.5 → 73.1; 16 x 16, lr 1e-6, no KL, 500 steps, lambda 0.05 | CC | Table 1 confirms all val100 figures and TB2 pass@1 1.57 → 2.70 → 5.17; recipe confirmed (temperature 0.8, gradient clip 0.2, sequence-level loss). Corrections: from the SFT start the auxiliary loss gave +9.6 on in-distribution val100 but TB2 pass@1 7.64 → 7.87 (+0.23, paper's SE about 1.5), pass@3 14.38 → 13.82, pass@5 unchanged; the appendix lists lambda 0.02 for the SFT start although the main text says 0.05 for all, and "run length 500-1000 steps"; single training run per cell | https://arxiv.org/html/2605.24517 |
| 4 | River: lr 5e-7, 32 x 8, 400 steps, about 60 h on 8 H200, sequence-mean, no KL, repetition penalty 0.05; SFT 12.5 → 19.4; random subset 17.7; over 60 % of environments have a quality issue; rewarding verify-before-done did not help; partial rewards led to premature completion claims | CC | Table 4 and text confirm every item ("rewarding verification brings no additional gains ... verification is correlated with task performance but does not directly cause task success"; "prematurely claims task completion"). Breadth table confirmed: 2 domains 49.2 vs 8 domains 53.9. Correction: "mean ± std over 3 seeds" is also attached to released baseline checkpoints, so these are evaluation seeds; training-seed variance is not reported | https://arxiv.org/html/2608.22631 |
| 5 | DeepSWE: 512 containers per step, 200 steps, 23 → 42 %; no KL; compact filtering; "four SFT-initialised models did not improve after 100 RL iterations" | CC | Verbatim: "We have attempted RL on top of four SFT'ed models, Claude-Sonnet 3.7/4 with thinking/non-thinking trajectories on top of Qwen3-32B. For all attempts, the model performance did not improve after 100 iterations." 512 = BS 64 x 8; 64 H100, six days; "With just 200 steps of RL training ... 23→42%". Context to carry with the quote: the SFT was distillation from Claude on a 32B model, and the item sits in a list the authors preface with "the same might not indicate a negative assertion"; compact-filtering evidence is a figure only | https://www.together.ai/blog/deepswe |
| 6 | SETA: Qwen3-8B, GRPO with DAPO dynamic sampling, TB2 3.1 → 10.7; 560 environments; partial reward raises informative tasks 49 → 86 %; hyperparameters not retrievable | CC | 3.1±0.6 → 10.7±1.3 (8 repeated runs), 560 environments, 49 % → 86 %, "dynamic sampling following DAPO, filtering out samples with uniform rewards" all confirmed. Correction: the hyperparameters are in its Table 9: lr 1.70e-5 constant, KL 0.0, PPO clip 0.4, batch size 8 with 16 trajectories per task (128 per update if the batch unit is tasks), 40 epochs, temperature 1.0 | https://arxiv.org/html/2607.10891 |
| 7 | DAPO: dynamic sampling adds 8 AIME points | C | Table 1: 30 → 36 (overlong filtering) → 38 → 41 → 42 → 50 (dynamic sampling). 512 prompts x 16, lr 1e-6. One training run, AIME24 avg@32 | https://arxiv.org/html/2503.14476v2 |
| 8 | Online difficulty filtering +3.8; offline curation +0.0 | CC | Table 1 (Qwen2.5-3B, 16 x 16, from an SFT checkpoint): plain GRPO 26.3; offline curation by the initial model 26.3; balanced 0.3<p<0.7 30.1. Corrections: removing zero-variance prompts only (0<p<1) gives 27.3, that is +1.0; offline curation with an external 7B model as difficulty proxy gives 28.4, that is +2.1; every skewed filter is below plain (23.8-25.9) | https://arxiv.org/html/2504.03380v2 |
| 9 | TruthRL: binary 20.8 / 39.5 / 60.3 vs ternary 37.2 / 19.4 / 56.6 (truthfulness / hallucination / accuracy); bonus for abstaining worse (32.7); rule-based verifier collapse (-3.6 / 3.6) | C | All confirmed in Tables 1, 3, 5, 7; lr 1e-6, batch 64, KL 0.001, full fine-tuning. 3B models 27.4 and 21.9 | https://arxiv.org/html/2509.25760v1 |
| 10 | KARL re-run: ternary reward gives abstention 54-83 %, collapses at batch 64 | CC | Table 1: ternary on NQ, abstention 53.6 % (Llama-8B) and 83.0 % (Qwen-7B), accuracy 34.9 and 11.9. Text: "most severe at a batch size of 64, where the model rapidly converges to 100% abstention". Corrections: batch 64 is 64 queries x G 8 = 512 rollouts per update; the row is KARL's own re-implementation on closed-book NQ with 64-token answers, where base accuracy is 38.7 % and 23.0 % (blanket answering loses), not a re-run of TruthRL's setup; easy data (about 90 % accuracy) gave 0.5 % abstention and 51.0 % hallucination | https://arxiv.org/html/2604.22779 |
| 11 | Abstain-R1: false refusals 36.2 % → 20.4 % with a penalty | CC | Table 3, penalty 0 / -0.5 / -1: accuracy on answerable 46.1 / 57.1 / 57.2; false refusals 36.2 / 22.8 / 20.4; refusals on unanswerable 82.4 / 63.9 / 68.1. Corrections: the penalty also removes 14-18 points of correct refusals; wrong answers on answerable items rise 17.7 → 20.1 → 22.4 (derived as 100 minus the two reported rates); with the penalty false refusals (20.4) are still above the SFT start (17.3); the reward prices a refusal (-1) below a wrong answer (0), the opposite order from the proposed k < 1; 100 steps, batch 256, single runs | https://arxiv.org/html/2604.17073 |
| 12 | "Verify-and-repair segment from the agent's own candidate state, +9.7 points on a 9B model" | CC | Qwen3.5-9B TB 2.1 pass@1 26.59±2.31 → 36.33±2.12 (+9.74); full-trajectory distillation 27.72 (+1.12) and -25.87 on SWE-bench Verified vs -2.20; 27B +16.85; 35B-A3B 38.20 → 49.81; student prefix 33.33±1.40 vs teacher prefix 29.21±2.42; replay of candidate states valid in 2,498 of 2,549 (98.00 %). Correction to keep attached: this is SFT on a teacher (GLM-5.2, 78.65 % pass) continuation, only teacher-successful continuations kept; not RL. The prefix-source gap is about 1.5 SE | https://arxiv.org/html/2609.38812 |
| 13 | Continual GRPO: beta 0.04 cut new-task accuracy 5.5-7.1 points without reducing forgetting | W for the current version | Version 2 (13 Jul 2026) has the cited numbers: 2B MFT 60.86 → 53.72, MFN 49.36 vs 49.64; 4B MFT 67.29 → 61.77, MFN 56.83 vs 56.05. Version 4 (5 Nov 2026) has different numbers: 4B GRPO MFT 59.34 / MFN 48.46 vs GRPO with beta 0.04 MFT 58.67 / MFN 52.15 (-0.67 plasticity, +3.69 final accuracy); 2B 55.90 / 44.29 vs 48.82 / 42.32 (-7.08, -1.97). The "+13.67 at 8B" sentence is gone. Vision-language QA, single runs; the no-KL baseline itself moved by about 8 points between versions | https://arxiv.org/html/2607.04364 (v4) ; https://arxiv.org/html/2607.04364v2 |
| 14 | LoRA learning-rate ratio 10x (15x for short runs) | CC | "the optimal LR for LoRA is consistently 10x the one used for FullFT in the same application, for both supervised learning and reinforcement learning"; "around 15x ... for short runs" (anecdotal, under about 100 steps). Correction: the article fixes alpha = 32 (W + (alpha/r) BA), and shows alpha → 2 alpha is equivalent to doubling the B-matrix learning rate. This repo runs r 32, alpha 64 (`configs/base.yaml`, lines 22-23, no rsLoRA), so the rule maps a full-fine-tuning 1e-6 to about 5e-6 here, not 1e-5 (derived; exact for B, not for A). RL evidence in the article is single-turn maths | https://thinkingmachines.ai/blog/lora/ |
| 15 | OpenThoughts-Agent: 64 x 8, 48 steps, lr 5e-6, no KL, collapse after the peak, about +5 | C | Appendix D: RLOO, 64 prompts, 8 samples "(so 512 trajectories per gradient update)", global steps 48, lr 5e-6, KL control no, token-mean; RL raised tool calls +31 % and think tokens +116 %; "the Qwen3-8B model ... is unable to benefit from agentic RL"; export at step 45 | https://arxiv.org/html/2606.24855 |
| 16 | LOOP: LoRA r 16 alpha 32, lr 5e-5, 240 rollouts per iteration, 72 tasks; reward normalisation -9; KL -1 to -3 | CC | LoRA r 16, alpha 32 on attention and MLP; lr 5e-5; 40 tasks x 6; 24 scenarios, 72 tasks; 71.3±1.3 vs 61.9±4.0 with normalisation; GRPO 58.0±1.8 / 39.5±1.9 vs no KL 59.0±1.4 / 42.7±1.3. Correction: the KL difference is 0.4 and 1.4 SE (derived), and ± is over 5 evaluation runs of one checkpoint; reward is the fraction of unit tests passed, not binary | https://arxiv.org/html/2502.01600v3 |
| 17 | Abstention-collapse theory paper: answer probability 1.000 → ≤ 0.008 within ten steps, 3 of 3 seeds; repair raises coverage 0.47 → 0.78-0.81 | C | Confirmed, with forced-answer correctness 0.94-0.97 throughout; the paper states TruthRL is consistent because it uses penalty weight 1 and its data plausibly violate the collapse condition | https://arxiv.org/html/2608.00301 |
| 18 | Hallucination tax: standard RL drops refusal to 0.00-0.09; a 10 % unanswerable mix restores 0.73-0.94 at 7-8B, not at 1.5B | C | Table 2 confirmed, including Qwen2.5-1.5B-Math 0.04 / 0.35 / 0.01 with the mix | https://arxiv.org/html/2505.13988v1 |
| 19 | GiGPO: +13 to +14 points over GRPO at 150 iterations x 128 | C (figures) | ALFWorld 72.8 → 86.7 (1.5B), 77.6 → 90.8 (7B); WebShop success 56.8 → 65.0, 66.1 → 72.8; group size 8. The 150-iteration count and the "65 % of states recur" figure were not re-read | https://arxiv.org/html/2505.10978v3 |
| 20 | Qwen3.5-4B tool-use: +0.8 sparse, +2.9 calibrated | CC | Base 63.8, sparse 64.6 "then significantly declined after step 70", dense 57.3. Correction: the calibrated run is listed as "59.1-66.7 %" across checkpoints; 66.7 is the top of that range | https://arxiv.org/html/2604.02869 |
| 21 | CacheRL: Qwen3-4B, LoRA r 128, lr 1e-6, KL 0.15, flat within ±0.01 | CC | Confirmed (0.756±0.007 over 200+ steps, 16 x 16). Correction: the authors blame cache fidelity ("50% of rollouts receiving non-exact cached tool results"), and the LoRA rate is 10x below the LoRA rule; it is not clean evidence that RL on a strong SFT start fails | https://arxiv.org/html/2606.14179 |
| 22 | Tree-GRPO: better at a quarter of the budget (was "not verified") | C | Qwen2.5-3b multi-hop average EM: budget 4 per prompt, chain 31.8 vs tree 36.8; budget 16, chain 33.9 vs tree 36.5. So tree at budget 4 beats chain at budget 16 | https://arxiv.org/html/2509.21240 |

Also confirmed without correction: CodeMidas (verification commands 2.03 → 2.53, batch 32 x 32 rollouts, lr 5e-6,
TB 2.1 63.7 → 72.2; https://arxiv.org/html/2609.22068), RAGEN-2 variance filtering gains +6.9 / +3.7 / +2.9 / +0.8
(https://arxiv.org/html/2604.06268), and "tends to over-reject even at t=0" on Qwen3-4B
(https://arxiv.org/html/2512.19920v1).

Existence only (title and date match, body not re-read; do not make these load-bearing): 2511.16108, 2512.22322,
2605.02964, 2609.20812, 2606.09863, 2602.05863, 2604.08476, 2512.23165, 2602.03025, 2505.16421, 2502.18449,
2512.04220, 2512.17008, 2509.04259, 2506.19767, 2503.20783, 2504.20073, 2512.08093, 2510.20270, 2410.02089,
2503.02623, 2507.16806, 2602.22303, 2510.00977, 2602.07078, 2603.21357, 2506.09038, 2509.15541, 2510.18874,
2607.04332, 2603.05881, 2607.10738. The workshop abstract "When the Loop Closes" is listed at
https://neurips.cc/virtual/2026/171163; its figures were not re-read.

No citation was unfindable.

## 3. Corrections that change the plan

1. **Scale claim.** "Every published terminal-agent RL gain used 100k-500k episodes (Endless Terminals, ECHO, River,
   DeepSWE)" (notes, section 2). Endless Terminals' count is unpublished. Budgets that are actually stated: 24,576
   (OpenThoughts-Agent), 102,400 (River, DeepSWE), 128,000 kept (TMax, and ECHO if its batch unit is prompts); LOOP's
   127,000 is the earlier report's derivation and was not re-read. Episodes per update: 128 (SETA, if its batch unit
   is tasks) to 1,024 (CodeMidas), not "240-512".
2. **What TMax shows for this project.** The 4B result is RL from the base model. For a start as strong as G2 (78 %
   on its own distribution) the nearest published datapoint is TMax-27B: +3.2 ± 2.3 (TB-Lite, selected) and
   +0.3 ± 2.5 (TB 2.1), derived SE of the difference from the table's standard errors. TMax also reports that Endless
   Terminals tasks are easy for these models (Table 1: 92 % pass@1 for the reference model) and that by the end of
   training each step on them "generates seven times as many rollouts as the minimum needed to fill the batch".
3. **KL.** Drop the continual-GRPO citation as evidence of harm (row 13). What remains: eight recipes whose beta 0 I
   saw in the source (Endless Terminals, DAPO, DeepSWE, TMax, River, OpenThoughts-Agent, SETA, ECHO), one
   within-noise ablation (LOOP), one anecdote (Endless Terminals), and honesty-RL recipes that keep 0.001 (TruthRL
   and KARL seen; Abstain-R1 and the hallucination-tax paper per the earlier reports).
4. **Filtering.** "+3.8" belongs to a narrow pass-rate band; the pure zero-variance filter is +1.0 in that study.
5. **Refusal penalty.** Abstain-R1 supports "a penalty stops RL-induced over-refusal" and equally supports "it costs
   correct refusals" (-14 points). Its price order is the one iteration 19c showed to be dangerous here.
6. **LoRA rate.** At this repo's alpha / r = 2, "1e-5" is the equivalent of about 2e-5 in the cited article's
   parametrisation, that is 20x a full-fine-tuning 1e-6 and 40x the 5e-7 that the two SFT-start terminal recipes
   used (River, Endless Terminals). Counter-datapoint: LOOP also ran alpha / r = 2, at 5e-5, on a 32B model.
7. **Auxiliary loss on tool output.** From an SFT start its gain was in-distribution only (row 3).

## 4. Part 2: prior art for decision-point RL

### 4.1 Works that start rollouts from intermediate states of the agent's own trajectories

| Work | Setting | What was reset, and how | How the branch state was chosen | Reward and loss | Measured gain over whole-episode RL | Failure modes reported |
|---|---|---|---|---|---|---|
| VinePPO, https://arxiv.org/html/2410.01679 (code released) | Maths reasoning, 1.1B and 7B, full fine-tuning lr 1e-6 | Text only: "reset directly to any intermediate state simply by re-feeding the partial context" | Every reasoning step of each training trajectory | Binary correctness; K auxiliary rollouts per state give a Monte Carlo value; the auxiliary rollouts are not trained on | Reaches PPO's peak in fewer steps and "up to 3.0x" less wall-clock; accuracy numbers are in figures (not read) | Each iteration costs up to 2x (7B) and 5x (1.1B) more |
| Tree-GRPO, https://arxiv.org/html/2509.21240 (ICLR 2026, code released) | Search agents (retrieval tool calls), Qwen2.5 1.5B-7B and Llama 3B | The context prefix; there is no sandbox state | Random node expansion; each node is a full thought-action-observation step | Outcome reward only; advantage inside a tree (siblings) plus across trees | Multi-hop QA, Qwen2.5-3b, average EM, equal budget: 4 per prompt 31.8 → 36.8 (+5.0); 8 per prompt 31.4 → 36.4; 16 per prompt 33.9 → 36.5 (+2.6). Single-hop: +3.2, +1.9, +0.8, +0.5 as the budget grows from 2 to 16 | With 2 branches per tree the within-tree advantage alone "collapse"s on all four datasets; the across-tree baseline is needed as a fallback. Tool calls per episode rose 2.4 → 3.0 |
| DEEP-GRPO, https://arxiv.org/html/2602.14169 (code released) | Maths (0.5B, 7B) and the Tree-GRPO search-agent setup (3B) | Text or context prefix | "Pivots": deep, recoverable states inside unsuccessful trajectories, scored by a learned utility; 8 dense continuations at one pivot beat 4 at each of two pivots | Binary; two streams: standard GRPO on root rollouts plus a local objective on the continuations, prefix gradients masked, weight lambda | GSM8K 0.5B: 67.7 vs GRPO 64.1 / 65.9 / 66.0 / 66.2 at group size 8 / 16 / 32 / 64. Search agent: HotpotQA 45.1 vs Tree-GRPO 42.4 vs GRPO 39.0; 2Wiki 43.9 vs 43.7 vs 36.3 (whether the extra continuations are budget-matched was not established) | lambda 0.1 → 65.4, 1.0 → 67.7, 10 → 65.9: a dominant branch objective "hinders global policy learning". Pooling root and branch rollouts in one batch duplicates prefix gradients K times and biases the baseline |
| Branching Policy Optimization, https://arxiv.org/html/2607.14171 | WebShop, ALFWorld, SWE-bench Verified; Qwen2.5-7B from SFT | Sandbox snapshot and restore (overlayfs for Docker, pickled simulator) | Top-M highest-entropy steps of a fresh backbone trajectory | Outcome; leave-one-out baseline over siblings, propagated to the prefix with discount 0.95; KL 0.05 | Claims +3.6 to +6.1 points over GRPO at matched returns and 38 % fewer steps | **Not credible as a number**: see 4.2. The variance identity it proves (branch baseline removes the prefix-explained part of return variance) is standard and holds regardless |
| GiGPO, https://arxiv.org/html/2505.10978v3 | ALFWorld, WebShop; 1.5B-7B | No reset: it groups identical states that recur across the group's trajectories | Any recurring state | Outcome; step-level relative advantage inside each state group | ALFWorld 72.8 → 86.7 and 77.6 → 90.8; WebShop +8.2 and +6.7 | Needs recurring states; the authors call re-rolling from each state "prohibitively expensive" |
| PivoARL, https://arxiv.org/html/2607.03702v1 (code released) | Sokoban, Minesweeper, WebShop, ALFWorld, search QA; Qwen3-4B | Local retry from the "pivotal" erroneous turn, reusing the correct prefix | The model's own structured reflection names the turn | Outcome plus a reflection reward; rewards the correct prefix and isolates the erroneous suffix | States about +10 % Pass@1 over GiGPO on agent tasks and 42-44 % fewer interaction turns than full retries (not re-read in the tables) | Built for a multi-attempt protocol |
| SCVD, https://arxiv.org/html/2609.38812 (SFT, not RL) | Terminal-Bench 2.1; Qwen3.5 9B / 27B / 35B | The student runs live to its first complete candidate; a teacher continues from the same state. In the diagnostic, states are rebuilt by replaying commands in a fresh container (98.0 % valid) | First complete candidate, correct or not | Cross-entropy on the teacher continuation only; only teacher-successful continuations kept | Full-episode pass@1 +9.74 / +16.85 / +11.61 over base, +4.49 to +8.61 over full-trajectory distillation | Full-trajectory distillation lost 25.87 points out of domain; SCVD -2.20 on the 9B |
| Agent-R, https://arxiv.org/abs/2501.11425 (SFT) | Three interactive environments | Tree search; splice from the first error step to a sibling correct path | The actor model identifies its first error step | Supervised | +5.59 % (abstract) | not read |
| DAgger for agents, https://arxiv.org/abs/2605.12913 (SFT) | SWE agents, 4B and 8B | Turn-level mixing of student and teacher actions | Student-visited states | Teacher labels | +3.9 (4B) and +3.6 (8B) on SWE-bench Verified over the strongest baseline (abstract) | not read |

Classical RL with resets (abstracts read): Salimans and Chen 2018, https://arxiv.org/abs/1812.03381 (episodes start
from states of one demonstration); Jump-Start RL, https://arxiv.org/abs/2204.02372 (a guide policy rolls in, the
learner continues, the roll-in is shortened as the learner improves); RFCL, https://arxiv.org/abs/2405.03379 (reverse
curriculum from demonstration states, then a forward curriculum "to perform well on the full initial state
distribution"); Tavakoli et al. 2018, https://arxiv.org/abs/1811.11298 (restart from the agent's own past states;
"altering the distribution of initial states could change the optimal policies when searching within a restricted
class of policies"). Sandbox infrastructure: DeltaBox, https://arxiv.org/abs/2605.22781 (checkpoint 14 ms, rollback
5 ms) and Crab, https://arxiv.org/abs/2604.28138 (names "RL rollout branching" as a use case).

### 4.2 Why the closest match cannot be used

Branching Policy Optimization reports, for SWE-bench Verified: batch 128 prompts, 8 rollouts per prompt, average
rollout 182 s, 5,000 gradient steps, GRPO wall-clock 74.1 h on 8 A100 with sandboxes on "a separate 32-core pool".
Derived: 74.1 h / 5,000 = 53 s per step; one step needs 128 x 8 x 182 s = 186,368 rollout-seconds; that is about
3,500 sandboxes and model sessions running at once (about 870 if "gradient accumulation 4" means 32 prompts per
step). Neither fits 32 cores and 8 shared GPUs. Its compute matching is also inconsistent (13 rollouts against 8,
"sub-sampled to match"), no code link is given, and a 7B model at 29.8 % on SWE-bench Verified with 25 tool calls
would be an outlier. Treat the paper as a statement of the idea and the variance identity only.

### 4.3 Does behaviour learned from mid-trajectory starts transfer to full episodes?

- **For**: SCVD puts loss only on the post-candidate continuation and gains +9.7 to +16.9 on full episodes; its
  decomposition shows wrong-to-right transitions up 5.62 / 9.74 / 8.61 points with right-to-wrong under 2 %, and
  initial-candidate accuracy also rose 5.5 points on average although no loss touched that segment. Tree-GRPO and
  DEEP-GRPO evaluate full episodes and gain. SCVD also states that a correct candidate "can provide supervision for
  how to verify and terminate", which is the same pairing as the candidate's 1:1 true-claim and false-claim prefixes.
- **Conditions under which it was shown**: branch states came from fresh rollouts of the current policy (Tree-GRPO,
  DEEP-GRPO, VinePPO) or of the student being trained (SCVD); root rollouts stayed in the objective (Tree-GRPO,
  DEEP-GRPO); classical methods return to the true start distribution before they finish (RFCL, Jump-Start RL).
- **Against**: Tree-GRPO collapse with the branch-only advantage at 2 branches; DEEP-GRPO at lambda 10; one unvetted
  blog experiment where SFT on verified one-action repairs "recovered only 25.4% of the gain from
  interaction-bracketed online RL" (https://ajing.github.io/posts/2026-08-14-counterfactual-trajectory-repair-sft-vs-rl/).
  The spill-over SCVD reports (earlier segments change without loss on them) cuts both ways here: iterations 17-19
  saw twin actions move when only report tokens were trained.
- **Silent**: no study trains by RL on branch states alone and tests full episodes; none uses branch prefixes from a
  frozen earlier policy; none is on Docker terminal tasks with RL.
- **Base rate for the kill test**: in the SCVD diagnostic, base Qwen3.5-9B detects 69.75 % of its wrong candidates
  and repairs 23.65 % of the detected ones, about 16 % conversion per attempt on Terminal-Bench 2.1 (derived). The
  proposed bar (40 % of false-claim-prefix groups mixed at 8 continuations) needs a mean conversion of about 6 % per
  continuation if rows were alike (derived: 1 - 0.938^8 = 0.40); heterogeneity across rows raises the requirement.
- **Over-verification and over-extension**: River found a verify-before-done reward changed behaviour without gain.
  OpenThoughts-Agent's RL raised tool calls +31 % and think tokens +116 % and then "the policy over-extends" (reward
  collapse, export moved to step 45). Tree-GRPO lengthened episodes. Under U a budget exit scores 0 and a false claim
  -1, so "keep working until the budget ends" is paid on failing rows; decision-point groups make that path shorter.

## 5. Part 3: what external evidence supports, contradicts, or does not address

### (a) Decision-point RL as the main arm

- Supports: a branch-state baseline removes the prefix-explained part of return variance (identity; measured as
  better credit assignment in VinePPO); +5.0 EM at equal budget in a search agent (Tree-GRPO), +1.5 over a group-64
  GRPO (DEEP-GRPO); failed-trajectory pivots with dense continuations work better than spread branches (DEEP-GRPO);
  segment-only training reaches full episodes (SCVD).
- Contradicts: making it the only objective (Tree-GRPO collapse, DEEP-GRPO lambda 10); expecting a large effect (gains
  shrink as budget grows: single-hop +3.2 → +0.5; the one large claim is not credible).
- Silent: Docker terminal agents, LoRA at 4B from an SFT start, a claim-scored reward, stale prefixes from a frozen
  policy, any honesty outcome, any interval over training seeds.
- Bet: two-stream arm beats whole-episode-only on dev feasible-row U at equal sandbox time, P = 0.55, size +1 to +3 U
  if so. Decision-point-only arm beats whole-episode-only, P = 0.25. The decision-point stream moves a twin guard
  beyond its bound at some checkpoint, P = 0.35.

### (b) 50k-100k episodes with LoRA at about 1e-5 from an SFT start

- Supports the scale: River (+6.9 at 102k from SFT), ECHO (38.5 → 63.5 in-distribution from SFT), TMax Qwen3-8B (SFT
  9.1 → 17.8 TB-Lite), OpenThoughts-Agent (+5 at 24.6k), LOOP (LoRA, about 127k).
- Contradicts a large expectation: every SFT start above was weak on its training distribution (at most 42.6 %); the
  strong-start datapoint is +3.2 ± 2.3 / +0.3 ± 2.5 (TMax-27B); DeepSWE saw nothing in 51,200 episodes on four SFT'd
  32B models; Endless Terminals' SFT-start gain was +16.4 on its own dev set and +0.5 on another.
- Silent: LoRA RL on a multi-turn terminal agent near 4B (LOOP is 32B; CacheRL is confounded); the right LoRA rate at
  alpha / r = 2.
- Bet: P(feasible pass rate up by at least 3 points at 50k-100k episodes) = 0.35; P(above 0) = 0.70.

### (c) Online zero-variance filtering

- Supports: DAPO +8 (one run); TMax and SETA build it in; RAGEN-2 +0.8 to +6.9; balanced band +3.8.
- Contradicts or limits: the pure zero-variance filter was +1.0; the cost is rollouts (TMax: about 40 % of groups
  dropped on its own data, up to 7x oversampling on Endless Terminals tasks).
- Silent: ternary rewards. All evidence is binary. KARL attributes the abstention trap to groups that contain only
  failures and abstentions; a variance filter keeps exactly those groups whenever claim and non-claim are mixed.
- Bet: P(filter with refill raises informative groups per step by at least 1.3x) = 0.85; P(it adds at least 1 dev U
  at equal total rollouts) = 0.45.

### (d) Dropping the KL penalty

- Supports: eight working recipes at beta 0; Endless Terminals' anecdote; LOOP +1.0 / +3.2 without KL (0.4 / 1.4 SE).
- Contradicts: Tree-GRPO reports "an appropriately weighted KL term can help" (figure only); the continual-GRPO
  paper's current 4B numbers favour beta 0.04 on final accuracy (+3.7); the honesty-RL recipes keep 0.001.
- Silent: whether a KL to the SFT reference protects an SFT-taught behaviour in agent RL. The abstention-collapse
  paper shows it gives no restoring force once the answer/abstain choice saturates; the hallucination-tax paper shows
  refusal is kept by items in the batch (10 % at 7-8B, 30 % at 1.5B), not by the 0.001 KL it ran with.
- Bet: P(beta 0 against 0.04 changes dev U by more than 1 point in either direction) = 0.30.

### (e) Penalty on unwarranted refusals of feasible tasks

- Supports: Abstain-R1 (false refusals 36.2 → 20.4, accuracy 46.1 → 57.2); KARL's variant (abstention scores -1 only
  when some sample in the group was correct).
- Contradicts: correct refusals fell 82.4 → 68.1; wrong answers rose 17.7 → 22.4 (derived); the fixed-threshold
  penalty over-rejected on Qwen3-4B; this project's own 19c price table.
- Silent: multi-turn agents; a penalty smaller than the false-claim penalty; graders that must tell an unwarranted
  refusal from an honest failure report.
- Bet: P(k = 0.3 lowers control false-blame refusals by at least 2 points on dev) = 0.50; P(it raises twin
  claim-or-damage by at least 2 points) = 0.35.

## 6. Decisions

| Topic | Decision | Confidence |
|---|---|---|
| Use of the three literature reports | Adopt, with the corrections in section 3. Remove the beta 0.04 continual-GRPO citation and the Branching Policy Optimization effect size from any argument | high |
| (a) Decision-point RL | Adopt as one of two streams inside the main arm, not as the sole objective; measure first. Before training: share of false-claim-prefix groups with mixed U at 8 continuations under G2 (pass at 40 % or more, n at least 100 prefixes so the 95 % interval is about ±10 points, source-clustered) and replay fidelity (pass at 90 % or more; SCVD reached 98 %). Design constraints from prior art: keep whole-episode groups from the true start in every step; separate baselines per stream; loss on continuation tokens only; stream weight 1; branch from failed and passed episodes; refresh prefixes from the current policy at fixed intervals, or state the staleness as a named risk with a full-episode dev read every 25 steps | medium |
| (b) Scale and rate | Adopt 50k-100k episodes for the main arm as a pre-registered dose with a dose-response read; forecast +1 to +4 points of feasible pass, not more. Learning rate: measure first between 5e-6 and 1e-5 (the rule gives about 5e-6 at alpha / r = 2), chosen by the gradient-signal audit and the KL slope, subject to the repo's bf16 merge floor | medium |
| (c) Online filtering | Adopt filtering on U variance with stratified refill, plus re-banding of the pool from the run's own rollouts. Log the oversampling factor and the share of kept groups with no true claim; stop rule if that share exceeds its lr-0 value by a pre-registered margin | medium |
| (d) KL | Adopt beta at most 0.001 as the default, with KL to G2 as a monitor and stop rule. Protection of honesty behaviour rests on twins and pair groups in every batch | medium |
| (e) Refusal penalty | Measure first, as a single-factor arm (k = 0.3, twin weight raised to keep the blind policy neutral). Pass: control decline rate down by at least 2 points with the upper bound of the change in twin claim-or-damage rate at or below +2. Not in the main arm | low |
| Auxiliary loss on tool output | Reject for the main arm; optional late arm. From an SFT start its out-of-distribution gain was +0.23 | medium |
| Budget-limit episodes | External evidence favours an episode-level loss mask with the reward kept in the baseline (DAPO +6, one run; DeepSWE, figure only). Under U the alternative pays over-extension. Whether this is compatible with the owner's token rule is for the other judges | low |
| Loss aggregation | No external basis to prefer one: TMax and OpenThoughts-Agent use token-level; Endless Terminals, River and ECHO use sequence-mean; DAPO's token-level step was +1. Measure | low |
| fp32 LM head | Measure first. TMax states it is "especially crucial when training Qwen 3.5"; the repo has it off (`configs/base.yaml`, line 18) | medium |

## 7. Blockers

1. A plan that cites Branching Policy Optimization's "+3.6 to +6.1 points at matched compute" as the expected effect
   of decision-point RL rests on numbers that fail a throughput check by about 100x.
2. A plan that cites "beta 0.04 cut new-task accuracy 5.5-7.1 points without reducing forgetting" cites a superseded
   version; the current one shows the opposite sign on final accuracy at 4B.
3. A plan that forecasts a large U gain for G2 from scale by pointing at TMax-4B (33.6 → 54.3) compares a base-model
   start with an SFT start at 78 %. The matching external datapoint is +3.2 ± 2.3 / +0.3 ± 2.5.
4. A decision-point arm with no whole-episode stream, or with prefixes frozen at G2 for the whole run and no
   full-episode check during training, contradicts every prior method and matches two reported failure patterns.
5. A refusal penalty justified by Abstain-R1 without its cost (14 points of correct refusals) and without noting its
   reversed price order would repeat the iteration-19c mistake with a citation attached.
6. Any statement that Endless Terminals used 100k-500k episodes; the number does not exist in public.

## 8. Not verified

- Figure-only results: TMax group size 8 vs 32, DeepSWE compact filtering, VinePPO accuracy gains, Tree-GRPO's KL
  ablation, Endless Terminals' step count.
- Whether TMax, SETA and ECHO used LoRA or full fine-tuning (none says LoRA; Endless Terminals and River are full).
- Whether SETA's "batch size 8" counts tasks, and how many of its 40 epochs ran.
- PivoARL's and Agent-R's tables; AWA-RL's body (abstract only: precision up to +10.3 points, "marginal sacrifice in
  raw accuracy"; it is the nearest multi-turn precedent for an abstention-aware reward, in a search agent).
- DEEP-GRPO's budget matching against GRPO on the search-agent task.
- The 31 identifiers listed under "existence only".

## 9. Artifacts

- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/scripts/fetch.py`
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/scripts/htmltext.py`
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/scripts/slice.py`
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/cross_literature/dl/` (one directory per fetched page, with `page.html` and, where dumped, `text.txt`)
