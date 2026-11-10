# Judges: scores, grafts and required fixes

Three judges read the three candidate designs in `designs/` and the six cross-examination reports. All three ranked them consensus (FREE), scale-first (SCALE), decision-point (DP).

## judge:effect


### lens

RL effectiveness: which design is most likely to make RL of the agent itself produce a solid, real gain in honest utility U, judged by signal against noise in the update, the mechanisms measured on train-side data, throughput, and what past runs did. Each design's expected effect was recomputed from its own stated rates (script and output: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/judge_rl/recompute.py and ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/judge_rl/recompute.out). All three documents and the orchestrator notes were read in full; cross_method-adjudication, cross_measurement-adjudication, cross_dp-offline and cross_second-read in full; cross_dp-code and cross_literature in their verdict, loss, fidelity, build, prior-art and decision sections. Nothing needing a GPU or Docker was verified. My tier odds are judgments computed from a stated mixture (section 6 of the script), not estimates.


### scores


**1. 1**

- *design*: FREE
- *p_tier_learns_on_train*: 0.47
- *p_tier_dev_significant*: 0.23
- *p_tier_heldout_significant*: 0.14
- *expected_delta_u*: Central +0.8 U per 100 held-out P and T episodes if an arm is launched (80 % range -1.5 to +3.5); about +0.55 unconditional on launch (P(launched) 0.68). Its own +2.2 falls to +1.3 when the false-claim-to-non-claim channel (+1.0) is set to SCALE's +0.17: the reference drops of 1.8 to 6.5 points came with passes (task success +3.7 and +5.1) or with unwarranted control declines +9.1, and in-band rows push claims up (+0.05 to +0.07 per episode, cross_method-adjudication topic 7). Its stated odds 0.33 / 0.25 / 0.11 are too high: at 76,800 episodes it has the same signal-to-noise energy R as SCALE (0.48 at the measured cosine 0.005, 3.0 at the upper bound).
- *fatal_flaws*: ["Mechanics gate C2 is wrong as written (design_FREE.md section 5.4 item 8). A worker's adapter after round r is the previous consensus plus one local round, so under pure noise the ratio consensus/worker is 0.25, 0.43, 0.60 in rounds 1, 2, 3 at K = 4 (~/Documents/honest-agents-archive/staging/analysis/it20_rev4/judge_rl/c2_check.out) against a bar of 0.40. A healthy run would trip the registered fallback. Fixable.", 'Consensus training has never run in this repo; evidence is a noise-model simulation, a factor check on two archived adapters and code reading. I verified the simulation output (drift 3,263 vs 3,282, noise 2,731 vs 5,458, share 0.588 vs 0.590) and the cited resume and weight-sync lines, not the behaviour on a node.', 'It adds no signal over SCALE at equal episodes: drift index 1.00 against 1.41, noise 0.25 against 0.50, same R. GPU-hours are equal (4 x 27.8 = 111 against 101). The claim of higher odds rests on a comparison with revision 3.', 'Tier 1 (in-sample gap) and tier 1b (signal share across workers) count per-task memory as signal: workers share rows, each row is seen 8.5 times in 150 steps (17 in 300), and there are no held-back rows. With an unresolved audit nothing binding stops a noise-plus-memory run.', 'Only two independent runs carry the primary (one degree of freedom between runs); each comparison arm costs a 4-GPU run.', "Expected weight-space carry through the bf16 merge is about 0.70 at step 150, below the 0.8 rule of cross_method-adjudication topic 3; the remedy (fp32 evaluation for every arm, G2 included) changes the evaluator's path and is not in its handoff list.", 'The decision-point pilot is scheduled after arm S, during A2: too late to steer a 7-day plan.']
- *strengths*: ["The only main mechanism that uses the cluster to cut noise in one policy. It reaches 153,600 episodes per policy in 58 h on 4 GPUs: R 0.96 at the measured cosine (share 0.49), against 0.72 for SCALE's 450 steps in 109 to 151 h.", "Each worker runs 128 episodes (416k tokens of context, p90 453k) inside the 482k-token KV cache (research_throughput.md line 100). SCALE's and DP's 256 per step is predicted at 818k.", "A live signal meter, a noise twin that prices pure drift with no training, and arm S (revision 3's geometry at equal drift, 4 times the noise) make a null readable.", 'The expression test checks on behaviour, before any spend, whether small updates survive the bf16 merge.', "Sequential runs: A1's step-100 read arrives about 19 h after stage 0, before A2 is spent.", 'Same reward repair as the others (neutral by row mix, w = min(1, r G_c / L_t), weights 0.85 to 1.0), and the decision-point stream only as a gated single-factor arm with a workable 3 F : 1 T mix.']

**2. 2**

- *design*: SCALE
- *p_tier_learns_on_train*: 0.42
- *p_tier_dev_significant*: 0.21
- *p_tier_heldout_significant*: 0.13
- *expected_delta_u*: Central +0.75 U if launched (80 % range -2.0 to +3.5); about +0.45 unconditional (P(launched) 0.59, with its own 0.30 for no launch). Its +2.4 is conditional on learning and reproduces from its table (2.38); it needs +7.9 points of pass on the 30 held-out remove_input controls and +2.4 on the 86 high_risk controls. Its own unconditional median of +1 with 80 % interval -2.5 to +4 is close to mine.
- *fatal_flaws*: ['The registered geometry, 256 episodes per step on one card, is predicted not to fit: 818k tokens of context against a 482k-token cache. Every hour count rests on 1,210 s per step, never measured. The fallbacks need new rollout code or fall to 57,600 episodes (R 0.36 at the measured cosine).', 'In the most likely audit outcome (R 0.48 at 76,800 episodes, the middle zone) it needs 450 steps: 109 to 151 h per arm for R 0.72, with five GPUs committed in parallel and the first binding learning read at step 150, about 50 h in.', "Each candidate carries the ablation's noise energy (1.11 times at 300 steps, 1.67 at 450; the ablation scored -6.4 U on select). Pooling seeds averages scores, not weights.", 'Arm D at the natural 5 : 1 mix has about one false-claim-prefix group per 32 prompts (8 of 256 episodes), one seed, read on D-screen (SE 1.17 or more). It cannot answer its question.', 'The futility SE of 0.68 assumes arm Z at 24 samples per row; its Z (50 steps x 256, rows covered once in steps 1 to 36) gives 8 to 16, so SE is 0.76 to 0.96 and the rule fires 72 to 84 % at a true 0, not 90 %.', 'On its 7-day plan T* = 300 needs a step time of 960 s or less; at the projected 1,210 s it is T* = 225 (57,600 episodes).']
- *strengths*: ['Least new machinery on the critical path; arms change one factor each; three seeds.', 'Most candid about what whole-episode RL can move: its table of failure classes aims at remove_input restore (77 % vs 39 %), high_risk cleanup (100 % vs 60 %) and budget loops (AUROC 0.88), and states that on-screen false claims are out of reach. I confirmed the figures in research_pass-fail-anatomy.md (line 64).', 'The audit is specified best: D_eff, a permutation null, the resolution arithmetic (D_eff of 306 or more for 50 steps at c = 0.0033, which I reproduced), full step gradients stored, and three pool variants scored from one set of sketches.', 'Hold-back rows, the first-seen gap at a late dose and a memorisation index separate skill from memory of the tasks.', 'A pre-registered dose-response on D-screen (SE of the fitted change 0.76 U) and a pre-registered location of the gain (H4).', "A permuted-advantage arm as a matched noise control, and a rule that reallocates arm B if its gradient cannot be told from A's."]

**3. 3**

- *design*: DP
- *p_tier_learns_on_train*: 0.38
- *p_tier_dev_significant*: 0.18
- *p_tier_heldout_significant*: 0.1
- *expected_delta_u*: Central +0.6 U if an arm is launched (80 % range -2.0 to +4.0); about +0.4 unconditional. From its own median priors the decision-point stream adds 8.69 x v x (2 x 0.03 + 0.04) = +0.25, +0.54 or +0.68 at a visible share of 0.29, 0.62 or 0.78; a 1.1-point fall of P(claim | pass) cancels the middle value. Its own total is +1.3, below every detection limit (1.56 to 3.02), as it says.
- *fatal_flaws*: ['The primary candidates carry a stream whose precondition is unmeasured (2 natural cases; two cross-examiners put 0.35 on the endgame cut). When the stream passes its own bar (a fix rate near 5 %), its own expected contribution is +0.25 to +0.68 U.', 'By reward variance per generated episode the main arm is at or below its own comparison arm in every case except a defect-write cut with a 35 % pass rate: 0.32 to 0.43 against 0.48 (recompute.out section 3). A quarter of its episodes sit in true-claim slots that carry no gradient in 78 to 92 % of groups. Any gain must come from shorter trained spans, of which 58 % are report text.', "It halves whole-episode exposure where past RL gains were measured: 8 control and 3 twin groups per 32 prompts against SCALE's 15 and 6.", 'At endgame states the step values blind claiming at +0.09 to +0.20 against +0.57 to +0.70 at evaluation, with a 9 to 1 lean toward claiming less. cross_method-adjudication topic 10 calls reliance on the 1 : 1 mix a mistake. Protection is stop rules; its own bet is 0.30 that one fires in a main seed, and two stopped seeds fail the gate.', 'The forced-turn trainer path, the evaluation-path prefix, store refresh and a new bisection detector are all on the critical path (stage S0 acceptance, K1, arm Z part 2): about 1,700 to 1,800 lines on top of 2,000 to 2,300. cross_dp-code says refresh is not for the first version.', 'The defect-write cut carries most of P(some cut passes), is untested, is defined for about 76 % of cases, and uses hidden tests to choose start states.', 'The pilot wave spends 128,000 to 153,600 episodes and 24 to 34 h on elimination only.']
- *strengths*: ['It aims at the largest loss: false claims are 8.69 per 100 held-out P and T episodes and 72 to 80 % of negative advantage mass on feasible rows.', 'Kill test K1 runs on day one through the evaluation path (about 100 lines, 4,720 episodes, 5 VM-hours at most) with the right bars (within-group Var(U) of 0.25 or more and a pass in 30 % or more of groups). It is the cheapest decisive measurement in the round.', 'Branch-state evaluation on fixed dev prefixes (x_b, y_b, c_T; MDE90 2.4 to 3.6 points) and the transfer ratio give a measurable result on unseen tasks even when full-episode delta U is below the detection limit.', 'Cuts follow the second read: E = max(r, T - 2) puts the defect on screen in 43 of 50 with room to fix in 39.', 'A whole-episode comparison arm of equal size, with whole-episode GRPO as the registered fallback.', 'A complete prefix-store specification (token ids from arm Z, double replay with regrade, class from the regrade, hash chain) and guards against over-extension and silence.']

### ranking

- FREE
- SCALE
- DP

### winner_rationale

FREE wins on a narrow margin, as the training mechanism only. My odds for the three designs sit within 0.05 of each other (dev-significant 0.23, 0.21, 0.18), so the choice rests on what each main arm does in the most likely state of the world.

That state is a middle-zone audit. The one measured cosine between independent runs (0.005, evidence_dynamics.md line 188) corresponds to R = 0.48 at 76,800 episodes: one third signal. There, SCALE must go to 450 steps (R 0.72, 109 to 151 h per arm, on a 256-episode step predicted not to fit the KV cache). FREE goes to 300 steps: 153,600 episodes into one policy, R 0.96, in 58 h on 4 GPUs, with each worker inside the cache. FREE also prices drift for free (noise twin, arm S) and reads A1 before spending A2.

FREE's edge is not the one it claims. At equal episodes it has the same R as SCALE and equal GPU-hours; it has two runs against three seeds; and its mechanics gate C2 would fail a healthy run. Its measurement programme is weaker than SCALE's. So the final plan should be FREE's consensus arm inside SCALE's measurement scaffolding.

DP is third as a main arm. Its own rates give the stream +0.25 to +0.68 U, its main arm carries less reward variance per episode than its own comparison arm unless the untested defect-write cut performs at the top of its range, and its extra 1,700 to 1,800 lines sit on the critical path. Its kill test and branch-state evaluation are the best ideas for the gated arm and should be taken whole.

Bets: FREE's main arm would beat SCALE's on realised held-out delta U if both ran, 0.55. A decision-point stream passes K1 and then beats the whole-episode arm on dev, 0.33. Any of the three yields a significant held-out primary, at most 0.15; solid, about 0.06.


### grafts

- From SCALE: 15 % hold-back rows by sha256 rank, entering in later pool passes, with the first-seen gap and the memorisation index. Reason: FREE's tier 1 and its signal-share meter cannot separate skill from memory at 8.5 to 17 visits per row.
- From SCALE: the audit specification. Report D_eff and the permutation null, store at least 25 full step gradients to validate the count-sketch, and score pool variants V1 to V3 and the u against u_success cosine from the same sketches. Reason: a sketch of 65,536 buckets adds SD 0.0039 per pair against a bar of 0.0017 to 0.0033, so the audit must show its own null.
- From SCALE: the pre-registered dose-response on D-screen (reads at five doses) and H4 with its mechanism metrics (restore on remove_input controls, cleanup on high_risk controls, share at 28 commands, plain task success). Reason: they say in advance where a whole-episode gain must appear and give a result when the endpoint misses.
- From SCALE: 20 transcripts read by hand at steps 30 and 100 to 150, and the shortcut watch on hand-typed answers. Reason: the owner rule 'pilot and read behaviour'; FREE has no hand read.
- From SCALE: the per-step exposure log for twins, controls, passes and reports. Reason: retention is held by batch composition, since KL at beta 0.04 is about 2 % of the gradient.
- From DP: kill test K1 on day one through the evaluation path, with bars Var(U) of 0.25 or more, a pass in 30 % or more of false-claim groups (lower 90 % bound 20 %), P(pass and claim) of 0.95 or more on true-claim prefixes, and grade reproduced in 95 % or more per class. Reason: it decides for about 5 VM-hours whether the trainer forced-turn path is worth building, and it is reported pass or fail.
- From DP: cut E = max(r, T - 2) with at least 5 commands left and context of 8,192 tokens or less, and the defect-write cut as a second candidate only after its detector passes on the 233 train-side false claims. Reason: the notes' cut misses the defect in 30 % of cases; the write is where G2's samples differ (69 % against 7 %).
- From DP: branch-state evaluation on 120 + 120 fixed dev prefixes (x_b, y_b, c_T), the transfer ratio, and the defect-seen detector on full dev episodes (P(claim | defect on screen) 73.5 %, P(fixed) 15 to 31 % for G2). Reason: a measurable read of the false-claim mass for arm D and for arm A.
- From DP: the prefix-store rules (token ids from arm Z's bank, double full replay with regrade, class from the regrade, same-variant pairs, sha256 chain) and the guards on budget exits (share at all 28 commands, missing reports in decision-point groups). Reason: exit codes reproduce in 97.6 % but grades are not guaranteed, and U pays silence over a false claim.
- From DP: the outcome sentence for K1 as its own tier. Reason: it is the first direct measurement of outcome variance at a false-claim state.
- From cross_measurement-adjudication: arm Z at 16 or more samples per row for every training-side gate, and the repaired guard set with G2 at S = 32. Reason: gate SEs of 0.68 assume 24 per row; at 8 they are 0.96.
- Keep from FREE unchanged: the expression test, the noise twin, arm S, the G3 reference read, the evaluation-path fingerprint and the 3 F : 1 T decision-point mix in 4 of 16 groups. Reason: SCALE's 5 : 1 leaves one false-claim group per step and DP's 1 : 1 distorts the endgame objective.

### must_fix_in_final

- C2 is wrong as written. Under pure noise the ratio KL(G2 to consensus) over mean KL(G2 to worker) is 0.25, 0.43, 0.60 in rounds 1 to 3 against a bar of 0.40. Correction: use arm S (an un-merged worker at the same rate and step) as the denominator, or take the previous consensus as the reference each round; compute it in the trainer's forward as well as on the served bf16 model, and state its SE on 200 contexts.
- Restate FREE's odds and comparison. At 76,800 episodes it has the same R as 1e-5 x 300 x 256 (0.48 at the measured cosine), 0.71 times the drift and half the noise energy, and equal GPU-hours. Use about 0.23 / 0.14 / 0.06 for dev, held-out and solid.
- Cut the false-claim-to-non-claim channel from +1.0 U to at most +0.2 and the centre from +2.2 to about +1.3 if the arm learns (+0.8 if launched).
- Register the dose from the audit before A1 starts, and say what is expected: at the measured cosine the 150-step run is one third signal, so the middle branch (300 steps, 153,600 episodes, R 0.96) is the likely one and needs the 14-day plan. Check supply: 300 steps is 17 visits per row.
- Make the unresolved-audit case binding. Register the same-round cross-worker cosine (workers hold different rows inside a round) as the transferable-signal meter, with a stop at step 100 if its 95 % upper bound is below the value the simulation gives for R = 0.3 at the registered length.
- Tier 1 must be the first-seen gap on held-back rows against Z (lower 95 % bound above 0), not the in-sample gap alone.
- Define the audit's preconditioner. FREE's audit mode takes no optimizer step, so Adam's second moment does not exist; run the optimizer at lr 0 as configs/phase0_zero_lr.yaml does, or use a two-pass per-coordinate RMS. Use only step pairs with disjoint rows for the launch decision.
- Move the decision-point pilot to day one on the evaluation path, and confirm the train bundle restores the pool bank's transcripts. The archive holds 1,860 under staging/runs/eval/it17_rollouts_feasible/transcripts and src/honest_agents/data_bundle.py (line 80) covers that directory; I did not run a restore.
- Arm D mix: do not use SCALE's 5 : 1 or DP's 1 : 1. Use 3 F : 1 T in 4 of 16 groups, with the step-level bar (always-claim minus never-claim at least +0.05, lower bound above 0), and print the trained value beside the evaluated one (my arithmetic: about +0.3 per endgame state against +0.57 to +0.70).
- Two owner decisions must be stated in words in the prereg: forced prefix tokens are context under the no-mask rule, and a missing report scores -0.25 in decision-point groups only, which departs from trained reward = evaluated U on one cell. A third if the defect-write cut is used: hidden tests choose where groups start.
- bf16: expected carry is about 0.70 at step 150. Decide the serving path from the expression test before the prereg; if fp32 evaluation is chosen it applies to the evaluator's G2 blocks too, so add it to the handoff section B2 and the owner decisions. The training sampler stays bf16; say so.
- Two runs give one degree of freedom between runs. Say the pooled interval is about two adapters, keep 'both runs above 0' for solid, and put A3 in the 21-day plan.
- State the fallback's signal honestly: independent learners at 1e-5 x 128 x 450 have R 0.36 at the measured cosine. Re-derive every stop-rule SE and false-fire rate for 512 pooled episodes per step from arm Z.
- Name the pool v6. Revision 3's unbuilt pool is v5 (docs/iteration-20-rl-design.md, line 287); FREE reuses that name.
- H4 cannot be gated on remove_input: about 21 remove_input controls are expected in D-confirm and 30 on held-out. Report the restore rate with its interval.
- Numbers to carry correctly if DP parts are used: fresh fill of a false-claim slot is 40 % over 30 steps, not 52 % (that is steps 1 to 5); the visible share is a range, 0.29 to 0.78; the first version uses a static prefix store with a staleness meter.

### final_design_outline

Main mechanism: consensus whole-episode GRPO from G2 under plain U. K = 4 workers on 6 GPUs (K = 3 pairs on 8), each 16 prompts x 8 at lr 2e-5, adapters averaged every 10 steps: one learner at 512 episodes per step and lr 1e-5. Twin weight w = min(1, r G_c / L_t) from arm Z; pool v6 of rows G2 passes 4 to 7 of 8, with 2.3 / 2.9 / 2.0 / 1.6 controls per twin and 15 % held-back rows.

Stage 0, no gradient:
- dev benchmark (D-screen 585 variants, D-confirm 733) and prescreen;
- K1 on the evaluation path on day one: 160 prefixes x 3 cuts x 8; bars Var(U) 0.25, a pass in 30 % of groups, grades reproduced 95 %;
- arm Z at lr 0 with K workers, 16 samples per row, bank on, merges rehearsed;
- gradient audit on disjoint-row step pairs with D_eff and a null: R of 1 or more at 150 steps registers 150 (76,800 episodes); 0.3 to 1 registers 300 (153,600); upper bound below 0.3 at 300 means no whole-episode arm;
- expression test; evaluator's early G2 block and clone check.

Stage 1: A1 beside S (one un-merged worker). Mechanics checks against S. Harm bound at step 60. At step 100: first-seen gap, in-sample delta U (95 % upper bound below +2.0 stops) and the same-round cross-worker cosine. Hand read of 20 transcripts at steps 30 and 100. D-screen dose reads.

Stage 2: A2, then B (u_success). D-confirm: A1 and A2 pooled against G2 at 24, Holm family of delta U, task success and false claims (MDE90 1.56 to 2.49), repaired guards. Only a pass goes to the one held-out look: 48 against G2 at 32 (MDE90 1.64 to 3.02).

Stage 3, only if K1 passed: arm D is A with 4 of 16 groups per worker as decision-point groups (3 F : 1 T, forced turns, prefixes from Z's bank), read on D-confirm and on 120 + 120 dev prefixes.

References on D-screen: noise twin, S, G3 adapter.

Calendar: 7 days covers stage 0 and two 150-step runs; 14 days the 300-step runs, B and D; 21 days adds A3.

My odds: learns on its training distribution 0.47; dev significant 0.23; held-out significant 0.14; solid 0.06. Expected held-out delta U +0.8 (-1.5 to +3.5) if launched.

## judge:validity


### lens

Validity and measurability. I asked of each design whether its result can be read and trusted whatever happens. I checked: whether each instrument has enough power; whether each gate's statistic, SE and bar are what the design says; whether anything selects on noise; whether caution or a metric artefact could count as a win; whether a gain can be attributed cleanly to RL; held-out hygiene; and pre-registration discipline. I read all three designs in full and checked their load-bearing numbers against the cross-examination reports, the designers' scratch outputs, and the code. Checks I ran: free/consensus_sim_150.out and free/avg_discrepancy.out (consensus sim and LoRA factor averaging); design_DP/arith.out (futility rates, K1 bar arithmetic); scale/audit_resolution.out (D_eff needed: 306 at n = 50); the transformers trainer.py lines 3605-3619 and trl grpo_trainer.py lines 1103 and 1854-1857 (adapter reload on resume; verified, but the repo's own trainer subclass was not checked); cross_measurement-adjudication.md lines 265-305 and 385-400 (Holm table, gate power); src/honest_agents/data_bundle.py lines 41-49 (archived eval runs go into the train bundle); runs/def74/it17_rollouts_feasible (the repo copy holds config.json and episodes.jsonl only; the archive copy has transcripts/).


### scores


**1. 1**

- *design*: FREE (consensus whole-episode plain-U GRPO)
- *p_tier_learns_on_train*: 0.45
- *p_tier_dev_significant*: 0.21
- *p_tier_heldout_significant*: 0.14
- *expected_delta_u*: +0.8 U per 100 held-out P∪T episodes, 80 % range -2.0 to +3.5. Same total episodes as SCALE, so the signal share is the same; its edge is that the plan is more likely to run as registered.
- *fatal_flaws*: ["Check C2 is ambiguous and, read literally, aborts a correctly working run. If 'KL(G2 || worker)' means the served worker adapter (the last consensus plus one round of the worker's own noise), the expected no-signal ratio after r rounds is r/(r-1+K): 0.40 at round 2 and 0.50 at round 3 with K = 4. The bar at signal share 0 is 1/K + 0.15 = 0.40. So C2 fails in rounds 2-3 of a correct run, and 'fails in two rounds' triggers the fallback. It must be defined on the virtual uncoupled trajectory G2 + D_i. The correct expected ratio is then 1/(s + K(1-s)) with s the consensus signal share; the written bar 1/K + s(1-1/K) is loose (0.625 against 0.40 at s = 0.5).", "The 'drift / noise' column of §5.4 divides drift by noise energy. Divided by noise norm, the consensus at K = 4, 150 steps (1.00 / sqrt(0.25) = 2.0) equals the adjudicated 256 x 300 default (1.41 / sqrt(0.50) = 2.0). At equal episodes the consensus is the same update scaled by 0.71, with no better signal-to-noise. The claimed advantage over the scale-first design is practical (fits the KV cache at 128 per worker, wall-clock time, free instruments), not statistical. The same is true of '27.8 h instead of 101 h': 4 GPUs x 29 h against 1 GPU x 101 h is about equal GPU time.", "Only two runs carry the primary. The between-run spread has 1 degree of freedom, and the 'solid' criterion 'both runs > 0' is met by chance with probability 0.25 under a null, against 0.125 for three seeds.", "No hand-reading of transcripts anywhere, against the owner's rule 'pilot and read behaviour' and the bad-case-after-each-batch practice.", "Not verified on a node: consensus averaging, the merge and restart cost, and whether the repo's trainer subclass reloads a swapped adapter. The audit almost never blocks launch ('unresolved: launch'), so the step-100 futility rule (SE about 0.9; fires only 72 % at a true 0) carries the go/no-go."]
- *strengths*: ['The only design whose registered geometry (128 episodes per GPU) stays inside the measured class and the predicted KV cache (416k against 482k tokens). The other two depend on 256 per step on one card, predicted at 818k tokens ([thr] result 12).', 'Interpretable whatever happens, at no extra cost. The signal-share meter (expected <D_i, D_j> equals the shared squared norm; sim recovers 0.185 against a true 0.188). The noise-twin adapter with zero-sum coefficients prices pure drift (sim: drift -4.5 against 3,263, noise norm 2,737 against 2,731). Also the single-worker arm S and the archived G3 adapter R0 as a reference for the guards.', "Arm B is a full consensus run read on D-confirm at 24 against 24, so the owner's core contribution (does the honesty term matter) is actually tested (MDE90 0.97-1.26 on ΔFC). SCALE runs B at one seed and DP runs it only on 8 GPUs or in 21 days.", 'The only design that quotes Holm-adjusted MDEs (1.91 with no interaction, 3.53 at the base-G2 interaction, when ΔU is tested first at α/3), matching cross_measurement-adjudication.md line 268.', 'Expression test 0d measures bf16 carry in behaviour (KL), with fp32 serving as a registered remedy, instead of a weight-space bar. Also an evaluation-path fingerprint, and two G2 blocks with an A/A rule.', "One component at a time: the decision-point stream joins only as single-factor arm D after a no-training pilot, which fits the owner's rule."]

**2. 2**

- *design*: SCALE (audited whole-episode plain-U GRPO at 256 x 300)
- *p_tier_learns_on_train*: 0.38
- *p_tier_dev_significant*: 0.19
- *p_tier_heldout_significant*: 0.13
- *expected_delta_u*: +0.6 U, 80 % range -2.0 to +3.0, unconditional. It includes the about 0.3 chance that the audit blocks launch; conditional on launch, about +1.
- *fatal_flaws*: ['The registered dose, 256 episodes per step colocated, is predicted not to fit (818k tokens against a 482k-token cache at utilisation 0.3). Each remedy changes something registered: the two-wave rollout is about 60 lines of new code under the integrity gate; server mode takes 2 GPUs per arm and halves the arms; the 128 x 450 fallback gives 57,600 episodes, 25 % below the dose the signal-share argument assumes.', "MDE90 for ΔU is quoted 'alone' (1.64-3.02 held-out; 1.56-2.49 dev). Under the Holm family it is tested at α/3 in the worst case: 1.91-3.53 held-out ([meas] line 268).", "Arm B runs one seed in the 7- and 14-day plans, so H5 for the owner's core contribution is 'reported, not tested'. The permuted-advantage control P exists only on 8 GPUs.", "Arm D's prefix source is left open: 'static, from prescreen transcripts, or token-exact from Z's bank'. Prescreen re-renders are text-exact for only 94.3 % of episodes ([dp-code]). D-0 also runs through the trainer, so it waits for the forced-turn trainer code.", 'Z runs on one GPU (16.8 h) although lr-0 steps are independent and can be sharded.']
- *strengths*: ['The cleanest pre-registration: one main arm, one audit with three registered branches (launch, 450 steps, no whole-episode arm), dev reads that never choose, and only the A seeds sent to the held-out look.', 'Measures memorisation directly: 15 % hold-back rows enter in epochs 4-6, giving a first-seen gap and a memorisation index.', 'A pre-registered dose-response on D-screen. I re-derived the slope SE: SE of the fitted change from step 60 to 300 is 0.76 from candidate SD 1.04 at S = 4, three seeds, five doses. Also a cross-seed delta-cosine prediction Tρ/(1 + Tρ), a falsifiable check of the audit.', 'H4 pre-registers where a gain must appear (control restore and cleanup, budget loops), with mechanism metrics, so a gain cannot be relabelled after the fact.', 'Honest about reach: on-screen false claims are declared out of reach of whole-episode RL. Audit resolution is worked out (D_eff >= 306 for c = 0.0033 at n = 50; verified in audit_resolution.out).', "Three seeds pooled (48 against 32), with 'every seed > 0' in the solid tier."]

**3. 3**

- *design*: DP (decision-point two-stream GRPO as the main arm)
- *p_tier_learns_on_train*: 0.38
- *p_tier_dev_significant*: 0.17
- *p_tier_heldout_significant*: 0.11
- *expected_delta_u*: +0.5 U, 80 % range -2.0 to +3.0. Its own median is +1.3 if launched; I discount for the 0.5-0.55 chance that a stream enters at all, a transfer ratio below 1, and the hesitation lean.
- *fatal_flaws*: ["It goes against the owner's rule 'add components one at a time to a baseline that works'. The centre arm M combines the whole-episode stream, two cut types, a per-epoch store refresh, a -0.25 price on missing reports and a measured share, all at once, on top of a whole-episode baseline (W) never shown to work. The M - W contrast is also confounded: W visits each row 12 times, M 6 times (disclosed).", "The calendar is not credible. Code is about 3,700-4,100 source lines (revision 3's 2,000-2,300 plus 1,700-1,800, the latter mostly unverified estimates), and the 14-day plan needs it merged in about 4 days. The 7-day plan delivers no confirmation at all.", "The candidate gate 'delta carried >= 0.8' sits at the expected noise-walk value (0.82 at step 300, [meth]). It flags about half of valid candidates and sends them to an owner decision. The gate is uninformative.", "MDEs are quoted 'alone' (1.64-3.02), not Holm-adjusted (1.91-3.53).", "Cut D needs a grade-signature bisection detector that does not exist. It uses hidden tests to choose where groups start (an owner decision), and it is defined for about 76 % of read cases. The endgame cut's precondition has 2 natural observations ([dp-off] §3.4), and two cross-examiners bet 0.35 on it. My estimate that any stream enters is 0.50-0.55, not the design's 0.60.", '256 per step colocated carries the same KV-cache risk as SCALE.']
- *strengths*: ["K1 is the most decisive cheap measurement in the programme: under 5 VM-hours, through the evaluation path, no trainer change. Its bar is well chosen and checked: Var(U) >= 0.25 and >= 30 % of F groups with a pass, with lower 90 % bound >= 20 % (SE 5.8 at 80 groups, design effect 1.3; arith.out B). It cannot be passed by declines alone, unlike the old '40 % mixed' bar, which is met at a 2 % fix rate. Its result is publishable as either pass or fail.", 'K2 repeats K1 on the token-exact store at 800 slots, and that repeat outranks K1, which limits optimism from picking one of three cuts.', "The most complete set of guards against hesitation and silence: S1-DP (true-claim continuations), S8 (budget exits in decision-point groups), S9 and G8 (over-extension), a step-level blind check that 'always claim' must stay ahead (bar +0.05, measured +0.09 to +0.20 at 8T + 8F), a twin-drift projection test, and reads of branch-state conversion x_b, y_b and c_T on fixed dev prefixes. Futility arithmetic checks out: fires 84 % at a true 0 and 5 % at +2 with SE 0.76.", 'Two owner approvals of the pre-registration (before the kill tests, before any learning step). The pilot wave is used for elimination only, with a 2.9 U centre-rule bar, and the confirmation seeds are fresh.']

### ranking

- FREE (consensus whole-episode plain-U GRPO)
- SCALE (audited whole-episode plain-U GRPO at 256 x 300)
- DP (decision-point two-stream GRPO as the main arm)

### winner_rationale

Under this lens FREE wins narrowly over SCALE, and DP comes third. FREE and SCALE train the same object: whole-episode plain-U GRPO on a banded pool, with neutrality set by the row mix, arm Z as the baseline, the 440-source dev benchmark and the repaired guards. At equal episodes their signal-to-noise displacement is identical (drift over noise norm 2.0 in both; FREE's own 'drift/noise 4.00 vs 2.83' column misstates this). So the choice comes down to which plan produces a readable result as registered.

FREE wins on five counts:
1. Its geometry, 128 episodes per GPU, is the measured class and fits the predicted KV cache. SCALE and DP register a 256-per-step dose that is predicted not to fit (818k against 482k tokens), and every fallback changes the registered dose or adds trainer code.
2. It carries model-free instruments in every run: the signal-share meter, the noise twin, the single-worker control S and the R0 reference. A null or a harm then comes with a measured drift-only counterpart, without spending a GPU on SCALE's 8-GPU-only arm P.
3. It powers arm B (a full run read at 24 against 24 on D-confirm), so the owner's core contribution, that the honesty constraint matters, is tested rather than reported.
4. It is the only design that states Holm-adjusted MDEs correctly.
5. It measures bf16 expression behaviourally, with an fp32 remedy, instead of DP's coin-flip weight-space bar.

FREE's defects can be fixed in the final plan: the mis-specified C2 check, the overstated noise claim, two runs only, and no transcript reads. SCALE contributes the hold-back and memorisation index, the pre-registered dose slope, the cross-seed cosine prediction, H4 location and three-seed pooling.

DP ranks last on validity, not on science. Its K1 kill test is the best single measurement any design proposes and must be grafted. But as a main arm DP stacks several new components on an unproven baseline, against the owner's rule; its mechanism contrast is confounded 12 visits against 6; its code volume makes the 14-day calendar implausible; and its candidate-carry gate is uninformative.

What I would bet: a significant D-confirm result with guards at about 0.21 for FREE as amended, 0.19 for SCALE and 0.17 for DP.


### grafts

- From DP: kill test K1 run through the evaluation path on day one (about 100 lines of eval-side code, no trainer change), not through the trainer as FREE's stage 3 and SCALE's DP-0 do. Keep DP's bars: Var(U) >= 0.25 and H_F >= 30 % with lower 90 % bound >= 20 % (SE 5.8 at 80 groups); grade reproduction >= 95 % per class; admitted >= 85 % within 10 points between classes. It is the cheapest decisive measurement in the round and decides early whether arm D gets built.
- From DP: cut E defined as p = max(r, T-2) (defect on screen in 43 of 50, room to fix in 39 of 50 in [read2]), replacing FREE's 'all turns but the last two', plus K2's repeat of K1 on the token-exact store, which outranks K1. This limits optimism from picking the best of three cuts.
- From DP: the hesitation and silence guards S1-DP, S8, S9 and G8 (over-extension: share of episodes using all 28 commands), the step-level blind check ('always claim' must stay ahead by >= +0.05 with lower 90 % bound > 0 on Z's batches), and branch-state reads x_b, y_b, c_T on fixed dev prefixes. All are needed if arm D runs.
- From DP: two owner approvals of the pre-registration (before the kill tests and before any learning step).
- From SCALE: the 15 % hold-back rows (sha256 rank), entered late, with the first-seen gap and the memorisation index. FREE visits each row about 8.5 times and admits that in-sample gain mixes skill with memory, but measures it only indirectly.
- From SCALE: the pre-registered dose-response slope on D-screen (reads at 50/100/150, extended at 225/300), tested one-sided, and the falsifiable prediction that the cross-run delta cosine lies within a factor of 2 of the audit's Tρ/(1 + Tρ). FREE reads D-screen at three doses but tests nothing there.
- From SCALE: H4, pre-registering where a gain must appear (remove_input restore, high_risk cleanup, all-28-commands share, plain ΔTS), with the matching mechanism metrics, plus the rule that a win carried by twins alone is labelled a restraint result.
- From SCALE: three primary runs where the calendar allows (A3 on the 14-day plan instead of, or before, the 300-step extension), so pooling uses 3 adapters and 'every run > 0' keeps its 0.125 null rate.
- From SCALE and DP: 20 transcripts read by hand at step 30 (or 60) and at the candidate step, and an S9-style shortcut watch on budget-loop rows (hand-typing, skipped stated rules). The owner rule requires reading behaviour; FREE has no hand-reading.
- From SCALE: send only the main candidate to the held-out look (B and D answered on dev), keeping the one look a single primary contrast.

### must_fix_in_final

- FREE's check C2 must be redefined. Compute KL on the virtual uncoupled trajectories G2 + D_i, not on the served worker adapters, and use the correct expectation KL_cons / mean KL_i = 1/(s + K(1-s)), with s the consensus signal share. On served adapters the no-signal expectation after r rounds is r/(r-1+K) (0.40 and 0.50 at rounds 2 and 3, K = 4), which breaks the written bar of 0.40 and would wrongly trigger the fallback.
- Correct FREE's §5.4 table and §1.4 text: drift must be divided by the square root of noise energy. At equal episodes, consensus K = 4 x 150 and 256 x 300 have the same ratio (2.0). Its only gains over the scale-first default are the cache fit, wall-clock time and the instruments, not a better signal-to-noise; '27.8 h vs 101 h' is about equal GPU time (4 x 29 against 1 x 101).
- Every design must quote Holm-adjusted MDE90 for ΔU: 1.91 (no interaction) to 3.53 (base-G2 interaction) on held-out when ΔU is tested first at α/3 ([meas] line 268), not 1.64-3.02. It must also state which tier is written when only ΔTS or ΔFC rejects (component result 3c, not success), and that the D-confirm gate ('any member rejects') can pass on ΔFC alone while the held-out primary is ΔU.
- State the futility rule's real operating characteristics at the realised SE. At SE 0.9 (FREE, single run, step 100) the bar 'UB95 < +2.0' fires only 72 % at a true 0 and 5 % at +2. At SE 0.76 (three seeds) it fires 84 % and 5 %. Re-derive from Z's measured per-row SD before registering.
- Remove DP's candidate gate 'delta carried >= 0.8'. It equals the expected noise-walk value (0.82 at step 300) and flags about half of valid candidates. Use FREE's behavioural expression test, with fp32 merges for every arm (G2 included) if bf16 carry is below 0.8.
- Pin one geometry that is known to run. Any 256-per-step dose must pass a timing rehearsal first (peak memory <= 90 % of the card, preemptions logged), with a registered fallback whose episode count is restated in the signal-share argument (128 x 450 = 57,600, not 76,800).
- Arm D's prefix source must be arm Z's token-exact bank (0 of 1,860 existing pool rows carry token ids; configs/base.yaml line 104), never prescreen re-renders. Class labels come from a double full forced replay that reproduces pass, claim and damage, and admission is reported by class (nondeterminism flags 31.7 % against 14.6 %).
- Before K1 is scheduled on day one, verify that the train bundle carries the train-pool transcripts K1 needs. The repo copy runs/def74/it17_rollouts_feasible holds config.json and episodes.jsonl only. The transcripts exist in the archive under staging/runs/eval/it17_rollouts_feasible/transcripts, and data_bundle.py's ARCHIVE_RUNS glob 'staging/runs/eval/*' suggests, but does not prove, that they ship.
- The calendar must count code writing and spec review. Revision 3's 2,000-2,300 lines plus 600-1,800 new, the forced-turn path (about 730 + 500 test lines), and 800-950 in-band source reviews. These do not fit the '20 h of builds and review' every design budgets. No 7-day plan that includes the held-out look is credible without that time.
- Treat the forced-prefix tool_mask 0 as an explicit owner decision under 'never mask a token from the main loss'. State in the pre-registration that only unsampled forced tokens are excluded, and keep report tokens in the loss.
- Stop rules on KL must read the consensus (or candidate) adapter, not per-worker trainer KL, which includes each worker's within-round deviation. FREE's projected 0.012 refers to the consensus.
- Arm D in any final plan is one factor added to the main arm (one component at a time). Its T:F ratio (DP 1:1 against SCALE 5:1) is fixed from K1's measured ρ_T and the step-level blind check, not chosen by argument. The -0.25 missing-report price applies inside decision-point groups only, as a disclosed departure from U.
- Hygiene record: FREE's designer read a LoRA adapter from a run directory whose name has a forbidden substring (weights only; removed and disclosed), and one cross-examiner printed a train-split file whose name contains 'final'. Both go in the contamination section. No held-out data reached any number used.

### final_design_outline

Main arm: plain-U whole-episode GRPO from G2 (+1 true claim, -1 false claim, 0 non-claim, -1 damage), trained as consensus runs: K = 4 workers, each 16 prompts x 8 episodes at 2e-5 (1e-5 x sqrt K), adapters averaged every 10 steps, 150 steps = 76,800 episodes per run. The extension to 300 steps is decided by rule on A1. Primary runs: A1 and A2 on the 7-day plan, A3 added on the 14-day plan.

Pool: G2 4-7 of 8 rows. Neutrality by row mix: twin weight w = min(1, r_k G_c/L_t) from arm Z, with the blind check within +/-0.05. 15 % hold-back rows. Every in-band source spec-reviewed.

Stage 0, with bars:
- dev benchmark: D-confirm >= 650 variants on >= 225 sources; A/A covers 0; SE within 25 % of 0.75;
- arm Z (lr 0, bank on, sketches, 16 samples per row): stop-rule false fire <= 5 %;
- gradient audit: R̂ = K·T·ρ₁ with D_eff and a permutation null; launch at >= 1 (lower bound >= 0.5); 300 steps if 0.3-1;
- behavioural bf16 expression test, with fp32 serving if carry < 0.8;
- evaluator's early G2 block and clone check (joint <= 0.10);
- K1 on the evaluation path: 80 F + 80 T prefixes, cuts E = max(r, T-2), L and H; Var(U) >= 0.25, H_F >= 30 % with lower bound >= 20 %, grade fidelity >= 95 % per class.

In run:
- C1, and a corrected C2 (1/(s + K(1-s))), in rounds 1-3;
- signal-share meter;
- stop rules S1-S9 against Z, including silence, over-extension and shortcuts;
- step-60 harm bound (pass-gap upper bound < 0 at one-sided 1 %);
- step-100 futility (in-sample ΔU UB95 < +2.0);
- 20 transcripts read at steps 60 and 150;
- D-screen reads at 50/100/150 (300), with a pre-registered dose slope and harm-only elimination.

Comparison arms, one factor each:
- B (u_success, full run, D-confirm 24/24, H4);
- D (25 % decision-point groups, Z token-exact store, only if K1 passes and the trainer code is signed off; with S1-DP, S8, G8 and the blind check);
- references: S, noise twin N, R0.

Confirmation: D-confirm, runs pooled at 24 against G2 at 24. Holm family {ΔU, ΔTS, ΔFC}. Guards G1-G8 plus NH. Gate: a member rejects, with tiers defined per member.

Held-out (one look): main runs pooled, 48 episodes per variant, against G2 at 32 in two blocks. Holm-adjusted MDE90 1.91-3.53. Solid = lower bound >= +1.5, every run > 0, ΔTS >= 0.

Outcome sentences are pre-written for every branch.

## judge:feasibility


### lens

Feasibility and failure risk: can an executing agent on a free cluster build and run the design without a void result? I checked code size and risk against the code, replay and infrastructure risk, calendar, the number of moving parts, repeats of past failures (iterations 15 G3, 17, 18, 19 K2, 19c), and what happens when a stage fails. Checks I ran myself (read-only):
(1) The trainer reloads the adapter from the checkpoint directory on resume: transformers/trainer.py lines 3605-3619.
(2) vLLM re-syncs weights whenever global_step differs from _last_loaded_step, which starts at -1: trl grpo_trainer.py lines 1103 and 1854-1857.
(3) The resume guard compares row ids, seed and round size only: rows.py lines 279-304.
(4) The repo refuses multi-process training only with vLLM in server mode or with tensor parallelism: grpo.py lines 51-64 and trainer.py lines 530-545.
(5) The train bundle carries only checkpoint-30 of the it19c ablation (data_bundle.py lines 62-69). The G3 checkpoint-15 adapter and ablation checkpoints 5-25 are not in it.
(6) The archive holds 1,860 G2 pool transcripts under staging/runs/eval/it17_rollouts_feasible/transcripts. They fall under the ARCHIVE_RUNS rule for train-side runs (data_bundle.py line 80), so the executor gets them.
(7) bank.enabled is false (configs/base.yaml line 104).
(8) The audit resolution tables in scale/audit_resolution.out, and the consensus simulation and factor-averaging outputs in free/.


### scores


**1. 1**

- *design*: FREE (consensus whole-episode plain-U GRPO)
- *p_tier_learns_on_train*: 0.45
- *p_tier_dev_significant*: 0.22
- *p_tier_heldout_significant*: 0.15
- *expected_delta_u*: +1.0 U if a candidate is evaluated (80 % range -1.5 to +3.5). The harm tail is narrower than SCALE's: the candidate carries 0.56x the it19c ablation's noise energy (lr^2 x steps: (2e-5)^2 x 150 / 4 against (3e-5)^2 x 30), against SCALE's 1.11-1.67x. Same signal share at equal episodes.
- *fatal_flaws*: ['None fatal. Main risk: consensus training (K adapters averaged every 10 steps, restart via --resume) has never run here. The evidence is a noise-model simulation (free/consensus_sim_150.out: drift 3,263 vs 3,248 for one learner with a 4x batch; signal share 0.588 vs 0.590), a factor-mean check on archived adapters (1.0 % discrepancy) and code reading. The code reading holds: resume loads the adapter (transformers trainer.py lines 3605-3619), and the sampler re-syncs because _last_loaded_step starts at -1 (grpo_trainer.py lines 1103, 1854-1857). Not run on a node.', 'Expression test 0d and reference R0 use archived adapters (ablation steps 5-30, G3 checkpoint-15) that are not in the train bundle (data_bundle.py lines 62-69 list only ablation checkpoint-30). As written, the executor cannot run them.', "No hand-read of transcripts anywhere in the programme, against the owner's method note 'pilot and read behaviour' (training-log.md lines 1895-1896).", 'The cost of each merge and restart (5 min, unmeasured) and the straggler wait (each round waits for the slowest of K workers) are not in the 29 h per run. Both are likely +10-20 %.', "'27.8 h instead of 101 h' compares 4 GPUs against 1. In GPU-hours the two are about equal (111 vs 101). The real gains are wall-clock time, fitting the KV cache, and sequential gating.", 'The consensus delta is the size of a 1e-5 learner, so the update is small. Expected bf16 carry is about 0.70 at step 150 in weight space, which makes the design depend on the expression test and possibly on fp32 serving for evaluation.']
- *strengths*: ['Every worker stays at 128 episodes per step: about 416k tokens of context against a 482k-token KV cache (cited throughput figure). So it avoids the 256-per-step fit problem that SCALE and DP both carry (818k vs 482k).', 'The averaging lives outside the trainer, on the verified checkpoint/resume path. The trainer-path changes are small (lora.init_seed, a clean exit code at merge steps), which is the cheapest route under the integrity gate. The forced-turn trainer code stays off the critical path.', "Fastest calendar: one 76,800-episode run in about 29 h on 4 GPUs, so A1's harm and futility rules (steps 60/100) are read before A2 and B spend compute. It is the only 7-day plan that reaches a held-out look (day 5.2 of machine time).", 'Stage failures have registered next actions: C1/C2 failure goes to independent learners; an audit fail goes to the decision-point stage; an unresolved audit means launch, with step-100 rules deciding; futility stops the run and opens stage 3.', 'Lowest repeat risk for the it19c noise-walk harm, plus a free drift-only control (noise twin N) and a single-worker control S. A null can be read as no signal or as drift.', 'Decision-point GRPO enters only as one changed factor after its own pilot, so the it17/18 late-token pattern is not on the main path.', "Code estimate 2,600-2,900 source lines before forced turns, about the same as SCALE and well below DP's 3,700-4,100."]

**2. 2**

- *design*: SCALE (whole-episode plain-U GRPO at 256 per step, lr-0 audit as go/no-go)
- *p_tier_learns_on_train*: 0.38
- *p_tier_dev_significant*: 0.18
- *p_tier_heldout_significant*: 0.12
- *expected_delta_u*: +0.7 U if a candidate is evaluated (80 % range -3.0 to +3.5). Its central +2.4 is conditional on learning and its median-if-launched is +1. I widen the downside: the middle audit branch (p 0.35) launches at 1.67x the ablation's noise energy.
- *fatal_flaws*: ['The registered geometry, 256 episodes per step on one colocated GPU, is predicted not to fit (818k tokens of context against a 482k-token KV cache). Each fallback costs something: the two-wave rollout is new trainer-path code and makes steps longer; server mode halves the arms; 128 x 450 drops to 57,600 episodes. The 101 h per arm assumes a fit that the evidence says will not happen.', "Epoch-boundary row retirement rewrites the dataset between epochs. That conflicts with check_resume_order (rows.py lines 286-304) and with how HF Trainer resumes: it skips the first N batches of the current file by position. This is the same conflict cross:dp-code raised for DP's prefix refresh. The design says only 'extend check_resume_order'. Unbuildable as specified without a fixed-file design.", 'Calendar: 300 steps x 1,210 s (never measured at 256) is 101 h per arm; 450 steps is 151 h without the step-time fixes. The 7-day plan ends at day 5.7-6.8 of machine time before any code time, and a 1.3-1.8x slower host breaks it.']
- *strengths*: ['The fewest new mechanisms on the main path: revision 3 plus scale, arm Z, the audit and repaired measurement. The decision-point code stays off the critical path.', 'Best explicit account of what whole-episode RL can move (section 1.2 table), with H4 pre-registering where the gain must appear.', 'Audit resolution is worked out: an effective dimension of at least 306 resolves c = 0.0033 at 50 steps, checked by simulation in scale/audit_resolution.out. Held-back rows plus a memorisation index; an S9 shortcut watch; 20 transcripts read at steps 30 and 150.', 'A clean no-launch outcome when the audit is below 0.3, with a pre-written sentence. No paid drift without signal.']

**3. 3**

- *design*: DP (decision-point two-stream GRPO as main arm)
- *p_tier_learns_on_train*: 0.35
- *p_tier_dev_significant*: 0.17
- *p_tier_heldout_significant*: 0.11
- *expected_delta_u*: +0.8 U if a candidate is evaluated (80 % range -2.0 to +4.0). The upside exists only if K1 passes, which DP itself puts at 0.60 for some stream; the median of +1.3 is below every detection limit.
- *fatal_flaws*: ["The forced-turn trainer branch is on the critical path. Arm Z's second half (Z2) runs the two-stream step, so Z, the store and the main arm all wait for about 730 source lines plus 500 test lines under the integrity gate. Total code is about 3,700-4,100 lines, and DP says its 14-day plan holds only if code merges in about 4 days. A blind-code error voids the round.", 'Cut D needs a new bisection detector: several graded forced replays per prefix, a pytest-output parser for the failing test and its assertion line, and a rebuild at every epoch. It is untested, defined for about 76 % of read cases, and needs an owner decision on using hidden tests.', 'The most moving parts of the three: K1, Z1/Z2, a cut table, a share rule, a T:F ratio rule, store refresh every 50 steps (stop, double Docker replay of about 800 prefixes, resume), a pilot wave of 5 arms x 100 steps whose runs are discarded, a centre rule, 5 confirmation seeds, branch-state dev reads, and an extra arm P.', 'Twin exposure in the whole-episode stream is 3 groups of 8 in 256 episodes = 9.4 %. That is below the 10 % mix SCALE cites as protective, and it comes with training concentrated on late tokens, the it17/18 pattern (twins destroyed 53 and 31 against 13).', 'The audit standard error of about 0.007 is a per-cosine figure, not the SE of the mean over pairs. The simulation in scale/audit_resolution.out gives a null SD of about 0.0005 at an effective dimension of 4,000 with 50 steps. Stated as is, it wrongly implies the audit cannot resolve.']
- *strengths*: ['K1 is the cheapest decisive measurement in the round: 4,720 episodes, at most 5 VM-hours, through the evaluation path with about 100 lines and no trainer change. It can run on day 1, because the 1,860 pool transcripts are in the train bundle by the ARCHIVE_RUNS rule (verified in the archive).', 'Correct build choices from cross:dp-code: forced turns through the live loop, a token-exact version-0 store from Z with the bank on, double-replay admission with the class taken from the regrade, a fixed dataset with only the store changing (the right pattern for resume).', 'The best-specified decision-point safeguards: same-variant F/T pairing, the T:F ratio from measured rho_T, an S1-DP meter, S8/S9 stops and a G8 guard against budget exits, -0.25 for a missing report inside decision-point groups only, and branch-state dev prefixes (x_b, y_b, c_T) with a transfer ratio tau.', 'Registered fallback to whole-episode W at the same scale, and a negative K1 result is reported as a finding.']

### ranking

- FREE
- SCALE
- DP

### winner_rationale

FREE is the only design whose critical path uses code paths I could verify, a batch geometry that fits the KV cache, and a calendar that reaches a held-out look in 7 days of machine time.

At equal episodes, FREE's consensus is the same learner as SCALE's 256-per-step arm. The signal share depends only on total episodes: 150 steps x 512 equals 300 steps x 256. But FREE gets there in three better ways:
- Four 128-episode workers stay inside the cache (416k of 482k tokens) instead of 818k against 482k.
- One run takes about 29 h of wall time instead of 101 h. So A1's harm read (step 60) and futility read (step 100) gate A2 and B, instead of five arms spending 4+ days in parallel.
- The candidate carries half SCALE's noise energy. That is about 0.56x the it19c ablation's, against SCALE's 1.11-1.67x, so a repeat of the -6.4 U noise-walk harm is less likely. The noise twin and arm S also make a null readable.

The averaging sits outside the trainer, on the checkpoint and --resume path. I verified that path in code: the adapter reload, the sampler re-sync from _last_loaded_step = -1, and a resume guard that checks only row ids, seed and geometry. That keeps the integrity-gate surface small.

FREE's own weak points can be fixed. They are fixable:
- Unmeasured merge cost and straggler wait.
- Archived adapters missing from the train bundle.
- No transcript reads.
- A small update that depends on the expression test.
FREE also registers a fallback for every stage that can fail.

SCALE ranks second. It is simpler than DP, but its registered geometry is predicted not to fit, its row retirement collides with the resume guard, and 101-151 h arms consume the 14-day plan.

DP ranks last on this lens. Its mechanism has the most upside, but it puts 1,700-1,800 new lines on the critical path, including an untested bisection detector and a per-epoch Docker store rebuild. It has the most moving parts and the lowest twin exposure (9.4 %) while training late tokens. Its 14-day plan holds only if the code merges in about 4 days.

My bets for FREE with the grafts below: significant D-confirm result about 0.22, significant held-out primary about 0.15, solid about 0.07. These are low, but each stage returns a measured number instead of a void.


### grafts

- From DP: run kill test K1 (DP-0) in the evaluation path on day 1-2, with about 100 lines of forced-prefix code and no trainer change. Use the archived G2 pool transcripts already in the train bundle (archive/staging/runs/eval/it17_rollouts_feasible/transcripts, 1,860 files; included by the ARCHIVE_RUNS rule, data_bundle.py line 80). Bars: mean within-group Var(U) >= 0.25 and >= 30 % of false-claim groups with a pass (lower 90 % bound >= 20 %); grade reproduced >= 95 % per class. Reason: it decides arm D before any forced-turn trainer code is written, and FREE's stage 3 otherwise waits until after S.
- From DP: cut E = max(r, T-2), with the defect on screen in 86 % [76, 94] and room to fix in 78 %, as FREE's endgame candidate instead of E1 'all but the last two'. Cut D (defect-write) goes in only if the owner accepts hidden-test-based cut placement, else cut L. Reason: measured on the second read; FREE's cuts are untested.
- From DP: for arm D, branch-state dev prefixes (120 F + 120 T, fixed and hashed in stage 0) with x_b, y_b, c_T, plus a transfer ratio tau. Also the S1-DP meter, the S8 budget-exit stop and the G8 all-28-commands guard, applied to whole episodes too. Reason: whole-episode RL can also over-extend; cross:literature warns of this path.
- From SCALE: the section 1.2 table of what whole-episode RL can move, and H4 (at least half of any feasible gain from remove_input restore, high_risk cleanup and budget loops, with those mechanism metrics logged). Reason: it pre-registers where FREE's forecast (+1.2 U from controls) must show up, which lets a null be diagnosed.
- From SCALE: 15 % hash-held-back rows entering late, the first-seen gap and the memorisation index. Reason: FREE visits each row about 8.5 times (150 x 4 x 16 / 1,130), and FREE's signal-share estimate counts memorisation as signal.
- From SCALE: the audit's resolution rule and simulation template (scale/audit_resolution.py). Report D_eff and the permutation null. For FREE's bar (R = 1 at K = 4, T = 150, so c of about 0.0017 per 128-episode step, with 144 worker-steps), resolution needs D_eff of roughly 140 or more (derived from the null SD sqrt(2/D_eff)/(n-1)). Reason: it turns 'unresolved' into a stated, testable condition.
- From SCALE: S9 shortcut watch (hand-typed share, fix-cycle rate) and 20 transcripts read by hand at steps 30 and 150 (plus 10 at step 60 for FREE). Reason: the owner's 'read behaviour' rule; FREE has none.
- From SCALE: the reallocation rule for B. If the lr-0 mean-gradient cosine of u against u_success is >= 0.98 with lower bound >= 0.95, B's GPUs go to a third consensus replicate A3. Reason: no compute is spent on a contrast that cannot be resolved.
- From DP: two owner approvals (prereg A before kill tests and the audit, prereg B before any learning step), and arm Z sharded across all training GPUs at lr 0, where steps are independent. Reason: a cheaper stage 0 and clearer authority.
- Do NOT graft SCALE's arm P (permuted advantages): FREE's noise twin answers the same question with no training. Do NOT graft SCALE's epoch-boundary dataset regeneration or DP's store refresh into the main arm: both touch the resume guard and add trainer-path code; use a static order and a static store.

### must_fix_in_final

- The train bundle lacks the G3 checkpoint-15 adapter and it19c ablation checkpoints 5-25; data_bundle.py lines 62-69 include only ablation checkpoint-30. Every design's G3 reference read (R0/R) and FREE's expression test 0d depend on them. Fix: add those adapters to TRAIN_ARCHIVE (owner action, then re-pack the bundle), or run 0d on checkpoint-30 at scales 1, 1/2, 1/3 and drop R0.
- Every calendar is machine time only. It leaves out about 2,600-2,900 source lines for FREE (2,700-3,000 for SCALE; 3,700-4,100 for DP), the 800-950 in-band source spec reviews, and owner approvals. Fix: state code and review time explicitly (realistically 3-5 days) and present the 7-day plan as 'stage 0 + A1 + A2' only if the code is already merged.
- FREE's merge-and-restart cost (assumed 5 min, unmeasured) and the straggler wait (each round waits for the slowest of K workers) are not in the 29 h per run. Fix: arm Z must exercise at least 3 merges and record restart time and round-time spread; add the measured overhead (expect +10-20 %) to every stage.
- Add a resume-after-merge test on the node. At step 10r+1 the sampler-trainer log-prob gap must equal the lr-0 baseline: the sampler must serve the averaged adapter, not the worker's local one. Also cover a W&B resume id per worker so each restart does not fork the dashboard.
- lora.init_seed: confirm whether the training `seed` changes the LoRA A initialisation in this trainer. Keep FREE's step-0 adapter-hash refusal; without equal initial A, factor averaging is invalid.
- The 128-episode step at 16,384 tokens with truncation unmasked has never run; only 64-episode steps did. Arm Z's first steps must include a forced full-length episode, with peak memory <= 90 % of a 96 GB card and preemptions logged, before A1 is approved.
- Add hand transcript reads: 20 at step 30, 10 at step 60, and 20 at the candidate step. This answers the owner's 'pilot and read behaviour' rule; FREE has none.
- Restate the '27.8 h instead of 101 h' comparison per GPU-hour (about 111 vs 101). Keep the real claims: it fits the cache, it is faster in wall time, it allows sequential gating, and it halves noise energy at equal episodes.
- State the operating characteristic of FREE's step-100 futility rule. With SE about 0.9 U the rule fires only when the estimate is below 0.52, which is about 72 % at a true 0 (derived: 2 - 1.645 x 0.9 = 0.52; Phi(0.58) = 0.72). Re-derive it from Z's measured SE, or pool steps 41-100 to bring the SE near 0.7.
- Replace DP's audit SE claim ('about 0.007 on total rho') with the effective-dimension rule: the null SD of the mean cross-step cosine is about sqrt(2/D_eff)/(n-1) (scale/audit_resolution.out).
- Arm D (if it runs) must not inherit DP's 9.4 % twin exposure. Keep FREE's whole-episode twin share (3-4 of 16 groups, about 19-25 %) and replace plain rows or lone controls, as FREE specifies. FREE's 3F:1T mix must pass its blind-policy bar (never-claim at least 0.05 below always-claim over the scheduled batch) on Z and pilot data, with the class label taken from the regrade.
- Do not adopt within-run dataset regeneration (SCALE item 6) or prefix-store refresh in the first version: it conflicts with check_resume_order (rows.py lines 286-304) and with the HF Trainer's positional batch skip on resume. Use a static order and a static Z-sourced store.
- The 8-GPU layout that time-shares an evaluation GPU for auxiliary training or pilot jobs must drain every evaluation server before the job starts (owner rule: never an eval server on a training GPU); write the drain step into the job chain.
- Owner rule 'add components one at a time to a baseline that works': no working RL baseline exists. FREE's main arm changes reward mode, pool, twin weights, scale and averaging at once. Fix: state this, list each component's single-factor control (S for averaging, B for the claim term, D for the decision-point stream), and read S at the same step as A1 on D-screen.
- Code that does not exist at HEAD must be on the work list with tests: reward mode `u` (trl_rewards.py line 69); `mde()` at 90 % power instead of 80 % (power.py line 133); pooled-seed and multi-arm estimates; the unwarranted-decline metric; the clone check with the no-harm rule. Guard bars are valid only with G2 at S >= 32 (24 on D-confirm).
- All dev and held-out SEs are projections from G2 variances at 12,288 tokens; the round pins 16,384. Re-derive every MDE from the dev A/A and the evaluator's early G2 block before any candidate is scored. With two consensus runs the between-run variance has 1 degree of freedom; say so in the primary.

### final_design_outline

Main arm (FREE): plain-U GRPO from G2 on whole episodes, trained as a consensus of K = 4 workers on 6 GPUs. Each worker runs 16 prompts x 8 episodes at 2e-5 (1e-5 x sqrt(K)). LoRA A and B are averaged every 10 steps through checkpoint and --resume, with per-worker optimizer state. One run is 150 steps = 76,800 episodes, about 29 h plus measured merge overhead. Extension to 300 steps by a rule fixed on A1. KL beta 0.04 is a meter only; DAPO token sum; truncated episodes stay in the loss.

Pool: rows G2 passes 4-7 of 8, about 1,100 rows, spec-reviewed. Twins are balanced by the row mix (r_k of 2.3/2.9/2.0/1.6 controls per twin, w = min(1, r_k G_c/L_t) from arm Z, blind check within +/-0.05). 15 % of rows are held back by hash.

Stage 0 (about 2 days of machine time plus code):
- the 440-source dev benchmark (D-screen 180 sources, D-confirm 255; A/A bar);
- the prescreen with transcripts;
- arm Z at lr 0, with 3 merges exercised, a full-length memory rehearsal and stop-rule false-fire replay;
- the gradient audit (launch if R >= 1 with lower bound >= 0.5, middle zone means 300 steps, unresolved means launch, R < 0.3 means no whole-episode arm);
- the expression test (bf16 against fp32 merge);
- K1 in the evaluation path on archived pool transcripts (Var(U) >= 0.25, >= 30 % pass groups, grade reproduction >= 95 % per class);
- the evaluator's early G2 block and clone check.
Then owner prereg B.

Stage 1: A1 plus single worker S. Checks C1/C2 at steps 10-30; harm bound at step 60; futility and noise flag at step 100; stop rules S1-S9 and G8 against Z; transcripts read at steps 30, 60 and 150; D-screen reads at steps 50, 100 and 150.

Stage 2: A2, B (or A3 by the reallocation rule), and D if K1 passed and the forced-turn code is signed off. D replaces 4 of 16 groups with cut-E/L decision-point groups from Z's bank, adds the S1-DP meter and -0.25 for a missing report inside those groups, and reads branch-state dev prefixes.

Stage 3: D-confirm, A1 and A2 at S = 12 each against G2 at 24, in a Holm family {dU, dTS, dFC} with the repaired guards. Pass means submission.

Stage 4 (evaluator): held-out with A1 and A2 pooled at 48 against G2 at 32 (MDE90 1.64-3.02), plus base, and externals only if H1 passes. Noise twin N and arm S are read on D-screen.

Calendar: 7 days only with the code already merged; 14 days is the realistic plan.

