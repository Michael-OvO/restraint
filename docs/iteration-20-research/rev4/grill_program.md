# grill:program


## lens

Programme design and decision quality: is revision 3 the right experiment, staged the right way, for an executor on a free multi-GPU cluster with a one-shot held-out submission?


## verdict

Revision 3 is a valid experiment but a weak programme: one configuration, one seed for the primary, one fixed checkpoint, run open-loop into a one-shot look. I found no validity blocker from this lens. My probability that it yields a significant positive H1 with all guards passing is about 7 % (range 5-12 %); this is a judgment, not a computation.

Why so low:
- **The only comparable run scored negative on the plan's own endpoint.** I computed U on the train-side select runs: the it19c ablation against G2 is ΔU −6.5 [−11.1, −2.0], p 0.026 (52 sources, S = 4).
- **That run showed no gain on its own training prompts.** Second visit minus first visit on the same 120 prompts: ΔU −1.3 [−6.3, +3.7].
- **H1 needs a true effect of at least 2.5-3.6** at lr 2e-5, β 0.04, step 150, seed 1, none of which was tuned.
- **The executor cannot see an effect of that size before the look.** Select at S = 4 has SE 2.71 on ΔU; the 103-row probe has a projected SE of about 3.65.
- **Every non-ideal event ends the round.** A pilot abort other than KL, a stop at step 40, or a flat probe has no decision path, and the handoff forbids a second candidate.
- **The pilot's KL gate sits on its own threshold.** By the plan's arithmetic the registered lr projects to about 0.08 at step 150, which is the abort level.

The free compute is spent on three seeds of an untested configuration and two attribution arms that cannot resolve their contrasts. Meanwhile 440 clean validation-split sources (1,376 P ∪ T variants, 3.7 times the held-out source count) sit unused.

A staged programme fixes this without touching held-out hygiene: a pre-registered screen on a dev set, then fresh seeds of the winner on a disjoint dev set, then a pooled-seed primary on held-out. I put it at roughly 12-15 % for the held-out primary and 20-25 % for a significant dev result. It also gives a first measured signal at step 50 instead of at the evaluator's table.

Not verified: step time at 128 episodes per step, eval throughput on the cluster, who pays for the evaluator's node, and the dev-set variance (projected from one RL run on 52 select sources).


## findings


**1. P1 [major]**

- *title*: The executor has no instrument that can see a 3-point change in U, although a powered dev set is available in the train bundle
- *evidence*: Computed with ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/programme/dev_power.py on runs/eval/it19c_g2_select and runs/eval/it19c_ablation_select (score_records, DEF-66 and quarantine out, void union over both arms): SE of paired ΔU is 2.71 at 52 sources, S = 4, so the 90 % half-width is 4.5 and MDE90 is about 8.2. A split-half A/A of G2 gives the same noise (implied SE 2.64). The in-training probe (103 rows, S = 4) has a projected SE of about 3.65 for U against step 0, so stop rule 3 in docs/iteration-20-rl-design.md (line 470) fires only when the point estimate is below about −11 twice. The handoff says select's noise floor is 'about 3 points' (docs/iteration-20-handoff.md, lines 193-195); for U it is worse. Available and unused (dev_inventory3.py in the same scratch directory): the val split has 539 sources; minus the 98 val.jsonl sources (which include the probe's 40) and the DEF-66 sources, 440 clean sources remain with 1,376 P ∪ T variants. The frozen held-out set has 118 sources. Legality rule 1 keeps val out of the pool, and src/honest_agents/train/rows.py (lines 145-147) refuses any non-train row in training. Only 9 of the 1,376 variants have an oracle-validation row.
- *why_it_matters*: Every development decision (pilot read, stop rules, select at steps 60/105/150) is made on instruments whose noise is larger than the whole hoped effect of 0 to +5. The first number that can tell +3 from −3 is the one-shot held-out table. The plan cannot be steered, and a null cannot be explained.
- *proposed_change*: Add a section 4.6 'Dev benchmark (executor)'. Build two source-disjoint sets from the 440 clean val sources, stratified by family, with a seed-free hash rank:
- D-screen: the 52 select sources plus 140 val sources, about 622 P ∪ T variants.
- D-confirm: the other 300 val sources, about 939 variants.
Oracle-gate every dev variant, run the spec-audit scan on G2's dev bank, record sha256s, and add a dev never-trained check. Run G2 once as the shared reference (D-screen S = 16, D-confirm S = 24). Replace the select evaluations of section 8 item 2 with D-screen, and rewrite the handoff's 'what the select split can tell you' paragraph. Keep select's 60 twins for G5.
- *measurable_effect*: Executor-side SE of ΔU falls from 2.71 to about 1.2 on D-screen (candidate S = 4 against G2 S = 16) and to about 0.52 on D-confirm (three seeds at S = 8 pooled, G2 S = 24); MDE90 falls from 8.2 to about 3.5 and 1.5. These are projections from the select variance components through analysis.power.design_variance; re-derive them from a G2 split-half on the dev sets in stage 0. Cost: about 2,488 episodes per screened checkpoint and 7,512 per confirmed seed.

**2. P2 [major]**

- *title*: Five arms fixed in advance test one untuned configuration; the free compute is not used to find a configuration that works
- *evidence*: docs/iteration-20-rl-design.md (lines 432-444) fixes A1-A3, B and C at lr 2e-5, β 0.04, 150 steps. The lr is justified only as 'below it19c's 3e-5, for KL' (line 422). Section 9 (lines 496-502) calls the effect a hope of 0 to +5. Train-side evidence I computed: the it19c ablation against G2 on select is ΔU −6.5 [−11.1, −2.0], p 0.026 (U_P −7.2, U_T −6.2; dev_power.py). Its second pass against its first pass on the same 120 training prompts is ΔU −1.3 [−6.3, +3.7], task success +1.5 [−2.3, +5.4] (revisit.py, from the archived completions of it19-crgrpo-ablation-r2). No RL run in the project has had a second seed or a second lr.
- *why_it_matters*: With a prior centred below zero and one draw, a positive result is luck. A null is a statement about one hyperparameter point, not about 'plain outcome RL on this pool' as section 9 claims.
- *proposed_change*: Replace section 5.5 and section 6 with a pre-registered screen. Stage 1 runs eight arms for 50 steps (one epoch) on the seed-1 file, so every arm sees the same prompts at the same step:
- S0: lr 0 null arm (calibration)
- S1: centre (u, lr 2e-5, β 0.04)
- S2: lr 1e-5
- S3: lr 4e-5
- S4: lr 4e-5 with β 0.16
- S5: u_success (revision 3's arm B)
- S6: prompt-diverse (revision 3's arm C)
- S7: pool without twins (plain rows and lone controls)
At step 50, rank the arms that pass the safety gates by D-screen ΔU. The top three continue to step 150 with D-screen evaluations at 100 and 150. The winner is the (config, step) cell with the highest D-screen ΔU; a cell within one SE of the centre arm at the same step loses to the centre. If no cell has ΔU above 0, run one pre-registered fallback wave, then stop with a screen-null write-up. The grid, seeds, files, metric, tie-breaks and maximum of two waves are fixed in the prereg.
- *measurable_effect*: Configurations with a measured ΔU against G2 on unseen sources rise from 1 to 7, each at SE about 1.2. The round returns a dose-response table over lr, β, reward and pool whatever the outcome. Training episodes for the screen: 6,400 (null) + 44,800 (seven arms to step 50) + 38,400 (three arms to step 150), against 96,000 for revision 3's five arms. My judgment is that the chance of reaching a significant dev result roughly doubles; that is not computed.

**3. P3 [major]**

- *title*: The candidate rule (step 150, or the last checkpoint before a stop) ignores that earlier checkpoints were the better ones in this project's RL runs
- *evidence*: docs/iteration-20-rl-design.md (lines 442-444) and docs/iteration-20-handoff.md (lines 108-111). History: iteration 12 B0 selected checkpoint 10 of 15 because step 15 failed the control-decline rule (docs/training-log.md, line 890). Iteration 13's erosion rule fired and checkpoint 10 was best on the probe (lines 83-84); it was later better than its SFT start on every tier-F row, selected-on and one seed (lines 1117-1124). Only G3 preferred its last step (docs/iteration-15-g3-rl-read.md, line 77). In the ablation, task success was −3.4 by step 30 (docs/iteration-19-results-crgrpo.md, line 43). The code already allows a pre-registered non-capability selection metric (src/honest_agents/train/grpo.py, line 164, check_selection), and src/honest_agents/eval/select_checkpoint.py exists.
- *why_it_matters*: A fixed last step is a bet that U rises monotonically for 150 steps, five times longer than any earlier run, when two of three earlier runs were best before their end. The rule was forced by the lack of a powered dev set, not by evidence.
- *proposed_change*: Rewrite the candidate rule: the candidate step is chosen on D-screen among steps 50, 100 and 150 by the rule in P2, recorded in a committed selection file before stage 2 starts. Stage 2 then trains fresh seeds to that step only, so the selected run itself is never the candidate. Record a forecast for the next stage that subtracts the selection optimism (about 2 U for the best of up to 21 cells at candidate-side noise of about 1.05, under a global null).
- *measurable_effect*: The candidate step becomes a measured choice with SE about 1.2 per cell. If U peaks before step 150, the candidate gains the difference between the peak and step 150, which D-screen measures directly. The optimism is paid on dev, not on held-out, because stage 2 uses new seeds and disjoint sources.

**4. P4 [major]**

- *title*: H1 rests on seed A1 alone, which wastes two seeds' held-out episodes and makes the primary hostage to one run
- *evidence*: H1 is A1 against G2; 'A2 and A3 never replace A1 in H1' (docs/iteration-20-rl-design.md, lines 130-131 and 443). From the plan's own power table (lines 193-196): SE 0.86 at S = 16 without interaction means 0.37 pt² per arm. Pooling three seeds (48 episodes per variant) against G2 at 16 gives SE 0.70, MDE90 2.07. With G2 at S = 32 it gives SE 0.56, MDE90 1.63. With the base-G2-sized interaction fully shared across seeds the figures are 1.10 and 1.02 (MDE90 3.2 and 3.0), against 1.21 (3.56) for A1 alone. On select, the RL-against-G2 unit interaction was small: 0.0028 against a sampling term of 0.137 at S = 4 (dev_power.py). Of the 7 held-out runs (62,944 episodes), only A1 and G2 enter H1.
- *why_it_matters*: The owner's question is whether the method works, not whether one checkpoint does. If A1 trips a stop rule or is the unlucky seed, the primary is tested on it even when A2 and A3 finish well. The plan already defines the pooled estimator as H2.
- *proposed_change*: Make the pooled-seed contrast the primary: the mean over the three stage-2 seeds per variant against G2, 90 % lower bound above 0, with each seed's point estimate above 0 as the replication condition. Run G2 at S = 32 on held-out and each seed at S = 16. A stopped seed counts at its last checkpoint. Report single-seed contrasts as secondary.
- *measurable_effect*: Held-out MDE90 for the primary falls from 2.5-3.6 to 1.6-3.0. Held-out episodes fall from 62,944 to 47,208 (G2 17,984, three seeds 26,976, base at S = 4 2,248), a 25 % cut, when B, C and base at S = 16 are dropped (P7, minor m5).

**5. P5 [major]**

- *title*: No decision path: a pilot abort, a stop at step 40 or a flat probe ends the round, and the held-out look is spent regardless
- *evidence*: Section 6 gives a response only for the KL slope (docs/iteration-20-rl-design.md, lines 457-458); the other five abort rules have none. Section 7 has no rule for a flat probe; rule 3 fires only on a large drop. A stopped arm's candidate is its last checkpoint and is still evaluated at S = 16 (docs/iteration-20-handoff.md, lines 108-111 and 265-270). The handoff forbids anything further: 'Then stop. Do not start a second candidate for any arm' (line 228). The it19 prereg had the same gap: 'there is no default action for grey' (docs/training-log.md, line 1852).
- *why_it_matters*: The one-shot look is the scarcest resource in the project; the design already lists its contamination. Revision 3 spends it even when every train-side indicator says the candidate is worse than G2, for example on a model stopped at step 35. A null that a dev set could have established then costs the held-out set's remaining cleanliness.
- *proposed_change*: Add a section 'Decision table' and a submission gate:
- Stage-1 arm stops or fails a safety gate: that arm is eliminated and the programme continues with the others.
- No stage-1 cell has D-screen ΔU above 0: one fallback wave, then a screen-null write-up; no submission.
- Stage 2: submit only if the pooled D-confirm ΔU has its 90 % lower bound above 0, each seed is above 0, and the guards pass on D-confirm at revision 3's bars, with G5 on the select twins.
- Otherwise file a no-submission report with the dev numbers. The held-out look stays unspent for the next round.
Rewrite handoff rule 4, rule 5 and section A5 to match.
- *measurable_effect*: Under a true null the look is spent about 5 % of the time (the one-sided α of the dev gate) instead of 100 %. Every fork has a registered next action, so the number of terminal states with no action falls from at least three (pilot abort, early stop, flat probe) to zero.

**6. P6 [major]**

- *title*: The pilot's KL gate extrapolates a line from steps 6-10, and by the plan's own numbers the registered lr sits on the abort threshold
- *evidence*: Gate: projected KL at step 150, fitted on steps 6-10, above 0.08 means lower the lr or raise β and re-pilot (docs/iteration-20-rl-design.md, lines 457-458). Section 1 says the ablation's KL was 'linear at about 0.0008 per step' (line 119). I fitted outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl: steps 6-10 give 0.00085 per step, which extrapolates to 0.123 at step 150; steps 17-30 give 0.00019 per step, which extrapolates to 0.039. Mean KL over steps 17-30 is 0.0153, so the curve flattens after step 17. Scaled by 2/3 for lr 2e-5, the steps 6-10 fit gives 0.082, just above the gate; if KL scales with lr squared it gives about 0.055. The response does not say which knob, by how much, or how many re-pilots.
- *why_it_matters*: The gate reads the steepest part of a flattening curve, so it is likely to fire on a run that would have stayed under the 0.05 warning. Each firing pushes the lr down, which lowers the effect the round hopes to detect, and the choice between lr and β is left to the executor's discretion.
- *proposed_change*: Drop the 10-step extrapolation. Measure instead: in stage 1 every arm runs 50 steps, and the KL gate is the EMA at step 50 at or below 0.04 together with a slope over steps 31-50 that projects below 0.10 at step 150. The lr and β question is answered by arms S2, S3 and S4 rather than by a re-pilot loop. Correct section 1's 'linear' statement to the fitted values.
- *measurable_effect*: The KL decision uses 20 post-warmup steps instead of 5, on a horizon 3 times closer to the target. On the ablation's log the two fits differ by a factor of 3.2 at step 150 (0.123 against 0.039), which is the size of the error removed. Re-pilot loops fall from unbounded to zero.

**7. P7 [major]**

- *title*: The attribution arms B and C cannot resolve their contrasts and take two GPUs and 29 % of the held-out episodes
- *evidence*: H4 is A1 − B and C − A1 on held-out U, 'single-seed contrasts' (docs/iteration-20-rl-design.md, lines 141-142). A single-seed contrast at S = 16 has SE 0.86-1.21 (lines 193-196), so it detects only a difference of 2.5-3.6, while the whole effect is hoped at 0 to +5 (line 501). B and C cost 38,400 training episodes and 17,984 of the 62,944 held-out episodes.
- *why_it_matters*: The owner's core claim is that the honesty term matters. If the claim term contributes half of a 3-point effect, revision 3's design reports 'inconclusive' by construction. A positive H1 with B equally good would then be read as 'RL adds capability', with no power to say otherwise.
- *proposed_change*: Move attribution to the dev sets. In stage 1, u_success and prompt-diverse are screen arms (S5, S6). In stage 2, train three seeds of the winner and three seeds of its reward counterpart (u against u_success) to the same step, and pre-register the pooled A − B contrast on D-confirm for U and for false claims per feasible episode. Do not send B or C to held-out.
- *measurable_effect*: The attribution contrast's SE falls from 0.86-1.21 (single seed, held-out) to about 0.5 (three seeds against three seeds at S = 8 on about 939 variants; projection, sampling term only), so MDE90 falls from 2.5-3.6 to about 1.5. Held-out episodes saved: 17,984.

**8. P8 [major]**

- *title*: The plan does not say what counts as a solid improvement, what is written under each outcome, or put the real forks to the owner
- *evidence*: Section 9 (docs/iteration-20-rl-design.md, lines 496-502) gives one sentence on a null. There is no outcome table and no smallest effect of interest. Section 13 (lines 607-615) asks the owner about seven settings, five of which are technical defaults with a stated recommendation (DEF-63, DEF-25, c and d, the refreeze, S). It does not ask whether a dev gate may withhold the look, what the primary is, or what the evaluator may spend.
- *why_it_matters*: The owner asked for solid, measurable improvements over the last results. A single pass/fail on held-out, with a low chance of passing, gives no partial credit and no agreed reading of the likely outcomes.
- *proposed_change*: Add a 'Success ladder' with a number against it19c at each rung:
- Rung 1, RL learns on its own prompts: train-prompt ΔU against the lr-0 arm over steps 1-50, paired by prompt, lower bound above 0. it19c's nearest figure: −1.3 [−6.3, +3.7].
- Rung 2, RL beats G2 on unseen train-side sources: pooled D-confirm ΔU lower bound above 0. it19c on select: −6.5 [−11.1, −2.0].
- Rung 3, held-out pooled primary with guards.
Add an outcome table stating the write-up sentence for each combination. Reduce section 13 to four owner decisions: (1) the dev gate may withhold the look (default yes); (2) pooled seeds as the primary (default yes); (3) the evaluator's node-hours and whether externals run only after the internal primary passes (default yes); (4) the screen grid. List the other settings as defaults that stand unless the owner objects.
- *measurable_effect*: Registered measurable endpoints rise from one (held-out H1) to three, each with a stated SE (about 1.2, 0.5 and 0.6-1.0) and a baseline number from it19c. Owner decisions fall from seven to four.

**9. P9 [major]**

- *title*: The evaluator's half has no time or cost budget, and most of it does not serve the primary
- *evidence*: Neither document states a training time, an evaluation time or an evaluator budget; a search of both for hours and budgets finds only 'more than 12 h' per external arm (docs/iteration-20-rl-design.md, lines 493 and 613). Evaluator load: 7 models × 562 variants × 16 = 62,944 internal episodes, 1,440 G5 episodes and three external arms. Measured throughput: 848 select episodes in 38 minutes at concurrency 24 when warm, about 1,341 episodes per hour, and 286 per hour on a cold cache (wall times in runs/eval/it19c_ablation_select and runs/eval/it19c_g2_select). At the warm rate the internal part is about 47 node-hours. The handoff puts evaluation on a separate synced node (docs/iteration-20-handoff.md, lines 236-237). The last such node cost $4.53 per hour and the roadmap gives 'about $100 or more' (docs/training-log.md, line 1705; docs/roadmap.md, lines 8-9). Whether the evaluator's node is free is not verified.
- *why_it_matters*: If the evaluator runs out of time or money part-way, the registered evaluation gets cut after candidates exist, which is a validity risk. Five of the seven internal runs (A2, A3, B, C, base) do not enter H1.
- *proposed_change*: Add an evaluator budget table to section 8 and to handoff section B2: episodes, node-hours at a measured rate, and the order of work. Order: fresh G2 at S = 32 before candidates; the three seeds at S = 16; base at S = 4; externals for the pooled candidate, G2 and base only if the internal primary passes or the owner asks. Measure the node's episodes per hour on the 12-task sanity run and size timeouts from it.
- *measurable_effect*: Internal held-out episodes fall from 62,944 to 47,208 (about 47 to 35 node-hours at 1,341 per hour). The external runs (more than 36 hours) become conditional. All registered held-out episodes then serve the primary or its replication.

**10. m1 [minor]**

- *title*: The checklist runs the pilot before the owner's approval, while the header says approval comes before any training
- *evidence*: docs/iteration-20-rl-design.md (line 90): 'Pre-register and get the owner's approval before any training'. Section 12 lists the pilot as item 7 and approval as item 8 (lines 604-605). The pilot is 10 training steps of every arm.
- *why_it_matters*: An executor cannot follow both. It also leaves open whether steps 11-150 continue the pilot or restart.
- *proposed_change*: State that stage 0 and the stage-1 screen are covered by the prereg approval, that approval comes before any optimiser step, and that every later stage starts from its registered gate without a new approval.
- *measurable_effect*: One contradiction removed; owner approvals needed during the run fall to one.

**11. m2 [minor]**

- *title*: The select evaluations at steps 60, 105 and 150 feed no rule
- *evidence*: Section 8 item 2 runs 16 select evaluations 'for development. No selection' (docs/iteration-20-rl-design.md, lines 487-488). No stop rule in section 7 reads them. That is 13,568 episodes at SE 2.71 on ΔU.
- *why_it_matters*: Data with no decision attached invites post-hoc reading.
- *proposed_change*: Replace them with the D-screen evaluations that drive the stage-1 ranking (P1, P2). Keep select only for G5.
- *measurable_effect*: 13,568 episodes move from no registered use to the selection rule, at SE 1.2 instead of 2.71.

**12. m3 [minor]**

- *title*: No eval-path freeze between the fresh G2 runs and the candidate runs
- *evidence*: The fresh G2 held-out run happens before any candidate exists (docs/iteration-20-handoff.md, lines 255-258), while the executor keeps committing and each arm's submission records its own commit (lines 217-219). A harness change once moved G2's select verified success from 72.7 % to 69.1 % (docs/results.md, lines 57-58), which is larger than H1's MDE.
- *why_it_matters*: A staged programme has more code changes between the reference and the candidates. A paired contrast across two eval paths is not a paired contrast.
- *proposed_change*: Tag an eval-path freeze at the end of stage 0 and record a hash of the eval path (harness, grader, run_eval, the budget config). The evaluator checks each submission's hash against the G2 run's and re-runs G2 if they differ. The same rule applies to the G2 dev references.
- *measurable_effect*: Cross-commit contrasts fall to zero by construction; the check is one hash comparison per submission.

**13. m4 [minor]**

- *title*: Per-row G2 baselines come from the same prescreen that selected the rows
- *evidence*: Rows are chosen by their S = 8 prescreen pass rate and the same numbers become the stop-rule baselines (docs/iteration-20-rl-design.md, lines 280-285). A Beta-binomial fit to G2's it17 pool bank (band_bias.py in the scratch directory) gives in-band rows a true pass rate 0.6 points above the observed one, about +1.2 U with no training, and puts always-pass rows at a true rate of 0.975. A split-half at S = 2 shows the effect directly: rows selected at 1 of 2 passes score 0.652 on their other two samples.
- *why_it_matters*: The bias is small at S = 8 but has the sign of a learning effect in any comparison of training U with the prescreen.
- *proposed_change*: Use the lr-0 arm (S0) as the reference for every training-side comparison: same prompts, same step, same harness, no selection. It also measures the false-fire rate of every stop rule on a run that cannot drift.
- *measurable_effect*: Removes about +1.2 U of selection bias from training-side U comparisons; gives a measured false-stop rate from 50 null steps (6,400 episodes).

**14. m5 [minor]**

- *title*: Base at S = 16 on held-out is context only
- *evidence*: Section 8 item 1 lists base 'as context' (docs/iteration-20-rl-design.md, lines 480-483). It enters no hypothesis and costs 8,992 held-out episodes.
- *why_it_matters*: Evaluator time is the constrained side of the programme.
- *proposed_change*: Run base at S = 4 on held-out.
- *measurable_effect*: 6,744 held-out episodes saved, about 5 node-hours at 1,341 per hour.

**15. m6 [minor]**

- *title*: The submission format has no place for selection provenance
- *evidence*: The submission JSON records the arm, commit, adapter, configs, candidate step and inputs (docs/iteration-20-handoff.md, lines 218-226). It has no field for a screen grid, dev results or the selection rule's output.
- *why_it_matters*: A staged design is honest only if the evaluator can check that the candidates follow from the registered rule.
- *proposed_change*: Add a selection block: the grid, the per-cell D-screen table, the rule and its output, the D-confirm results, the dev-set sha256s, and W&B links for every screened arm including the losers. Add one programme-level file listing every run started.
- *measurable_effect*: The evaluator can re-derive the candidate from the committed table in one command; unreported runs fall to zero by check.

**16. m7 [minor]**

- *title*: No wall-clock plan, and step time at 128 episodes is unmeasured
- *evidence*: Neither document gives a training duration. The ablation's steps averaged 386 s at 64 episodes (outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl); it19c's method arm took about 14 minutes per step (docs/training-log.md, line 1876). A host outage has already cost a run (line 1880). At 128 episodes per step, 150 steps is roughly 25-35 hours per arm; that is my estimate, not a measurement.
- *why_it_matters*: Timeouts sized to a guessed speed killed two it19c steps. A staged plan needs stage durations to schedule GPUs.
- *proposed_change*: Add a schedule table with measured step time from the first five steps of the null arm, stage durations, GPU counts (8 training and 2-3 evaluation, or two waves on 4-5), and a checkpoint pull at every stage gate.
- *measurable_effect*: Timeouts and GPU allocation rest on a measured step time rather than on G3's or it19c's.

## strongest_improvements


**1. Build a powered dev benchmark from the unused val split**

- *description*: Two source-disjoint sets from the 440 clean val sources: D-screen (52 select sources plus 140 val sources, about 622 variants) and D-confirm (300 val sources, about 939 variants). Oracle-gate them, hash them, add a dev never-trained check, and run G2 once as a shared high-S reference. The trainer already refuses val rows.
- *expected_gain*: Executor-side SE of ΔU from 2.71 to about 1.2 (screen) and about 0.5 (confirm, three seeds pooled). The first reliable signal moves from the evaluator's final table to step 50 of stage 1.
- *how_measured*: A split-half A/A of the G2 dev reference must reproduce the projected SE within 25 % before stage 1 starts; analysis.power estimate with the U metric on every dev contrast.
- *cost_and_risk*: About 32,500 G2 reference episodes and a Docker oracle gate on 1,376 variants (CPU only). Only 9 of those variants have an oracle-validation row today, and roughly 13 % of unaudited sources may be spec-defective, which adds noise but no bias to a paired contrast. The variance figures are projections from one RL run on 52 sources.

**2. Replace the five fixed arms with screen, then replicate**

- *description*: Stage 1: eight arms for 50 steps (lr-0 null, centre, lr 1e-5, lr 4e-5, lr 4e-5 with β 0.16, u_success, prompt-diverse, no-twin pool) on the same prompt order. Rank on D-screen; the top three continue to step 150 with evaluations at 100 and 150. Stage 2: three fresh seeds of the winning (config, step) and three of its reward counterpart, evaluated on D-confirm. Everything is fixed in the prereg, with at most two waves.
- *expected_gain*: Seven configurations measured instead of one; the checkpoint is chosen rather than assumed; a properly powered attribution contrast (SE about 0.5 instead of 0.86-1.21). My judgment is that the chance of a significant held-out primary roughly doubles, from about 7 % to 12-15 %.
- *how_measured*: D-screen ΔU per cell with its SE; pooled D-confirm ΔU with a 90 % interval; train-prompt ΔU against the null arm per epoch (SE about 1.2, scaled from the ablation's revisit contrast).
- *cost_and_risk*: About 369,000 executor episodes against 172,000, and two to three more days of wall-clock, all on the free cluster. Needs 8 training GPUs or two waves. The screen winner will read about +2 U by chance under a global null, which is why stage 2 uses new seeds and disjoint sources. More code: a dev builder, a selection CLI, a null config.

**3. Gate the one-shot look on dev, with a decision table**

- *description*: Submit only if pooled D-confirm ΔU has its 90 % lower bound above 0, each seed is above 0, and the guards pass on D-confirm. Otherwise file a no-submission report and keep the look. Every stage-1 and stage-2 event (stop, abort, flat curve, screen-null) has a registered next action. Rewrite handoff rules 4 and 5 and section A5 accordingly.
- *expected_gain*: Under a true null the look is spent about 5 % of the time instead of always. Terminal states with no registered action fall from at least three to zero.
- *how_measured*: The gate's inputs are the D-confirm interval and guard verdicts, committed before submission; the evaluator checks them against the selection block.
- *cost_and_risk*: The owner may want a held-out number even after a dev null; make that an explicit owner decision with 'withhold' as the default. A conservative gate can block a real but small effect: at a true +2 the gate passes about 85 % of the time on my projected SE.

**4. Make pooled seeds the held-out primary and rebalance the samples**

- *description*: Primary: the mean of the three stage-2 seeds against G2, with each seed above 0 as the replication condition. Evaluator runs G2 at S = 32, each seed at S = 16, base at S = 4; B and C stay on dev. Externals run only if the internal primary passes.
- *expected_gain*: Held-out MDE90 from 2.5-3.6 to 1.6-3.0, with 25 % fewer held-out episodes (47,208 against 62,944).
- *how_measured*: Re-derive the SE from the fresh G2 run and from the D-confirm interaction estimate before any candidate is graded, as section 12 item 6 already requires.
- *cost_and_risk*: The upper end of the range assumes a method-by-source interaction as large as base against G2; the select data suggests it is much smaller for an RL arm, from one run. The estimand becomes the average over three seeds, which is what the owner's claim is about.

**5. Replace the 10-step pilot with measured gates at step 50 and a null arm**

- *description*: Drop the linear KL extrapolation from steps 6-10. Gate each stage-1 arm at step 50 on measured KL (EMA at or below 0.04, slope over steps 31-50 projecting below 0.10 at step 150), on the plan's pooled claim and decline bounds, and on train-prompt ΔU against the lr-0 arm. The null arm also supplies unbiased per-prompt baselines and the stop rules' false-fire rate.
- *expected_gain*: Removes an extrapolation that differs by a factor of 3.2 between early and late fits on the ablation's log (0.123 against 0.039 at step 150). Removes the unbounded re-pilot loop and about +1.2 U of baseline selection bias.
- *how_measured*: KL EMA and slope from each arm's train log; the null arm's 50 steps give the empirical rate at which each stop rule fires on a run that cannot drift.
- *cost_and_risk*: 6,400 episodes for the null arm. The step-50 thresholds are new numbers; set them from the null arm and the ablation log in the prereg, not afterwards.

## things_revision3_got_right

- The executor and evaluator split, with two data bundles, one combined table and no held-out result returned to the executor before submission. A staged design must keep this unchanged.
- A fresh G2 on the same commit and the one pinned budget, run before any candidate exists, with power and the joint clone-failure rate (at most 0.10) re-derived before grading.
- U as the single endpoint, with significance defined only as the 90 % lower bound above 0 on the paired, source-clustered estimator, and the code's `significant` flag explicitly not used.
- S = 16 per arm. On select the RL-against-G2 interaction was small next to sampling noise, so the lower MDE figure (about 2.5) is the more likely one.
- The pooled-seed estimator and its void union are already defined as H2; it only needs promoting to the primary.
- Seeds share one fixed composition with per-seed files, and B and C reuse seed 1's order, so arms are comparable step by step. The screen should keep this.
- TRL repeating the file order each epoch, which makes epoch-on-epoch U on the same prompts measurable (SE about 1.2 at 800 prompts). Promote it from a diagnostic to a gate.
- Stop rules written as pooled one-sided bounds against per-prompt G2 baselines. On the ablation's log the base rates sit well under the thresholds (truncated share mean 0.003 against 0.10; zero-variance share mean 0.18 against 0.7).
- The offline checks before any training: analysis.u_dynamics on real groups, the blind-neutrality check within ±0.05, and the legality and never-trained checks enforced in the trainer.
- Arm B (u_success) as the clean isolation of the claim term. Keep it, as a screen arm and as the stage-2 counterpart.
- Section 9's plain statement that the effect is a hope, and the disclosure of the design's contamination.