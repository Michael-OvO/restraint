# grill:metrics


## lens

Metrics, statistics and measurability (sections 2, 3, 6, 7, 8, 9 of docs/iteration-20-rl-design.md, the handoff, and the estimator code). All new numbers come from train-side data only: the it19c select runs, the it19c training outputs and probes, G2's it17 pool bank, split metadata. No held-out episode file was opened. Scratch scripts and outputs are in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/metrics_lens


## verdict

Revision 3 is careful about what it measures (one estimator, one pinned budget, a clone check, one look), but it stakes the whole round on one binary test that is unlikely to come out positive, and it gives the executor no instrument that can see the hoped effect before the one-shot submission.

My probability that revision 3 as written yields a significant positive H1 with all guards passing is about 0.17 (range 0.10 to 0.25). This is a judgment built on the computed power below, not a computed number.
- Power alone: P(significant H1) is 0.59-0.71 under the plan's own "0 to +5" range, 0.20-0.25 under a history-weighted prior (35 % harm, 40 % null, 25 % gain of +1 to +5), and 0.12-0.15 under a sceptical one (h1_prob.py).
- The prior is the bottleneck. No RL run in 19 iterations beat its SFT start, and the closest analogue (the it19c ablation) scores dU = -6.52 [-11.05, -1.99], p 0.026, on the select split when rescored with U (select_u.py).
- The guards then remove more. As provisionally written they fail an exact G2 clone about 26-31 % of the time (train-side proxy), and G3 fails any gain that arrives through honest non-claims.

What would raise it, in order of value:
1. Pooling the three seeds for the primary and giving G2 two to three times the samples: MDE90 falls from 2.5-3.6 to about 1.5-3.0, and power at a true +2 rises from 0.75 to 0.97 (no interaction) or from 0.49 to 0.61-0.64 (with the base-G2 interaction).
2. More sources, not more samples. With the interaction term the SE cannot go below 0.86 at 118 sources whatever S is. 441 val sources are in the train bundle, never trained on and never evaluated.
3. Testing the components: false claims per feasible episode and task success detect a pure reporting or capability effect about half and three quarters the size that pooled U needs.
4. Repairing the guards so they do not fail a clone or the reporting route.
5. Making A minus B a three-seed confirmatory contrast.

What does not help: covariate adjustment with G2's per-variant baseline (R^2 0.000 on select), common random numbers (not verified; long multi-turn trajectories should decouple within a few turns), and a dose-response trend as a power device (about 5 % SE gain; its value is interpretive).

With these changes I put a significant held-out H1 at about 0.25-0.30, and at least one pre-registered positive RL finding (H1, a component, or the A-B constraint contrast) at roughly 0.5. Both are judgments. Every outcome, including a null or a harm, would then be a reportable result with an interval, a mechanism split and a stated resolution. I found no blocker in this lens; the nine majors are all fixable in the plan text and section 10 code.


## findings


**1. S1 [major]**

- *title*: H1 is unlikely to be significant, uses one seed of three, and gives base as many samples as the shared reference
- *evidence*: Plan: docs/iteration-20-rl-design.md (lines 130-131: H1 is A1 only; lines 193-201: SE 0.86 / 1.21, MDE90 2.53 / 3.56; line 501: credible outcome 0 to +5; lines 480-483: base at S = 16 as context). Computed (h1_prob.py, per-arm per-sample variance 5.92 pt^2 from the plan's SE): P(significant H1) before guards is 0.71 / 0.59 (no interaction / interaction 0.744) under uniform 0..+5; 0.25 / 0.20 under a history-weighted prior; 0.15 / 0.12 under a sceptical prior. The plan's no-interaction SE reproduces from train-side data: 0.84 for a G2 clone at 118 sources, S = 16 (clone_guards.py, resampling runs/eval/it19c_g2_select). The only comparable RL run rescored with U on select: -6.52 [-11.05, -1.99], p 0.026 (select_u.py). The interaction term does not shrink with S: at V_int 0.744 the SE floor is 0.86 for any S at 118 sources.
- *why_it_matters*: The round is declared a success only on H1. As written, two thirds of the candidate episodes (A2, A3) and all of base's 8,992 episodes add nothing to the primary, and an effect of +1 to +2.5 (the middle of the plan's own hope) is detected only 31-75 % of the time.
- *proposed_change*: Make H1 the pooled contrast: every variant's mean over the 48 episodes of A1-A3 against G2, same estimator, seeds treated as fixed. Report per-seed estimates, the seed SD, and A1 alone as a sensitivity. Allocate samples by the shared-reference rule: G2 at S = 32 (48 if the evaluator can), each arm 16, base 8. State in section 3.4 that raising S cannot beat the interaction floor and that only more sources can (see S2).
- *measurable_effect*: SE(dU) 0.86 -> 0.56 (G2 at 32) or 0.50 (G2 at 48) with no interaction; MDE90 2.53 -> 1.64 / 1.46. With the full interaction 0.744: SE 1.22 -> 1.03 / 1.00, MDE90 3.56 -> 3.02 / 2.93. Power at a true +2: 0.75 -> 0.97-0.99, or 0.49 -> 0.61-0.64 with the interaction. P(significant) under the history-weighted prior 0.20-0.25 -> 0.23-0.30. Held-out cost: +8 to +24 episodes per variant (+7 % to +21 % of 62.9k).

**2. S2 [major]**

- *title*: Nothing the executor can run before the one-shot submission resolves the hoped effect, while 441 untouched val sources sit unused
- *evidence*: Select split: observed SE(dU) 2.71 (ablation - G2, 184 units, 52 sources, S = 4), clone SE 2.50, so the 90 % half-width is +-4.5 and MDE90 is 7.3-8 (select_u.py, h1_prob.py). The section 3.5 bars at select scale fail an exact clone 82 % of the time (G1 41 %, G4 68 %; clone_guards.py). Select has no remove_input rows (data/select.jsonl: 60 plain, 60 high_risk pairs, 16 revoke pairs). In-training probe: two step-0 probes of G2 on the same 101 rows differ by 16.8 points with source-clustered SE 9.0; projected SE at S = 4 is 4.5, so stop rule 3 fires only below -12.4 (probe_res.py). The plan still spends 16 select runs x 848 episodes on this (lines 487-489). Unused data: data/split_seeds.json gives 539 val sources; src/honest_agents/tasks/select_split.py (lines 86-99) excludes every val source from training; 441 val sources (1,553 variant dirs, about 1,379 in P and T families) are in no split file (val_inventory.py); the handoff confirms the train bundle holds every val task dir (docs/iteration-20-handoff.md, line 65). Only 12 of those variants are oracle-validated (data/validation.jsonl).
- *why_it_matters*: The executor submits blind to any effect below about 8 U, cannot pre-check a single guard, and has no basis to choose between an early and a late checkpoint. A null on held-out then cannot be told apart from an over-trained endpoint.
- *proposed_change*: Add section 4.6, a powered dev benchmark (dev-large): the val sources not in select, probe_v1 or val.jsonl, P and T families (N and C reported), passed through the same oracle gate, spec-audit scan and rebuild-diff quarantine as the pool; frozen with a sha256 before training. Replace the select runs of section 8, item 2 with: fresh G2 at S = 8; A1 at steps 50, 100, 150 at S = 2; every arm at its final step at S = 4. Pre-register the dev contrasts (pooled seeds vs G2, dose-response over 0/50/100/150, guards on dev). Optionally let the candidate step be chosen on dev-large by a fixed rule (latest checkpoint within one SE of the best), since held-out stays untouched. Keep the 60 select twins for G5 only. Optional, evaluator side: the frozen held-out uses 118 of 565 test sources (156 in data/test.jsonl); a refreeze is already planned, so adding validated test sources would lower the held-out interaction floor too.
- *measurable_effect*: Dev SE(dU) falls from 2.5-2.7 to about 0.86 at S = 4 and 0.61 at S = 8 with all 441 sources (0.92 / 0.65 after an expected 13 % loss to spec defects); pooled seeds (3 x S = 4) against G2 at S = 8 give about 0.55, MDE90 about 1.6. Cost about 5.5k episodes per S = 4 run, about 4 hours at the roughly 22 episodes per minute of the it19c select runs (file mtimes; 12k budget, not verified at 16k).

**3. S3 [major]**

- *title*: The on-policy learning curve is the best-resolved signal in the plan, but it has no estimator, and its natural baseline (the prescreen) is selection-biased
- *evidence*: Plan lines 475-476: train U is not comparable across steps and revisit-slope U is 'a diagnostic only'; lines 306-308: the same prompts recur at steps k, k+50, k+100. From the it19c ablation completions (1,920 episodes, G = 8; train_groups.py): within-group variance of U is 0.305 (plain 0.534, control 0.318, twin 0.139); a per-step mean has SE 20.6 points at 8 groups. Paired by row over 800 rows x 8 generations, SE(epoch 3 - epoch 1) is 0.98, the slope per visit has SE 0.49, and epoch 3 against a fresh G2 pass at S = 8 has SE 0.98 (about 1.05 at plan 20's composition). Selection bias (rtm.py, G2's it17 pool bank, 400 variants): rows picked in band on 2 samples move +14.2 points of pass rate on fresh samples; projected to the plan's S = 8 band, in-band rows move +0.6 points and the 'always-pass' retention rows -2.5 points (about -4.1 U on those rows). Twin-band bias could not be computed from this bank (not verified).
- *why_it_matters*: Without this rung a held-out null cannot say whether RL failed to learn on its own prompts or learned and did not transfer. Reading epoch-1 numbers against the prescreen would show shifts of the same size as the effect of interest. Stop-rule baselines written from the prescreen (g2_pass, g2_decline) carry the same bias.
- *proposed_change*: Add a pre-registered milestone M1 to section 7. Before training, run a fresh G2 pass at S = 8 on the 800 frozen pool rows (6,400 episodes, same config), and use it, not the prescreen, as the row baseline for M1 and for the stop rules. Estimator: per row, mean U over the 8 generations at each visit minus the fresh G2 mean, clustered by source, 90 % interval, reported at steps 50, 100, 150 with the five-term split of S6. M1 passes when the visit-3 lower bound is above 0 with P(claim | pass) inside its bound. Add it to analysis.u_dynamics and the W&B dashboard.
- *measurable_effect*: A training-side dU with SE about 1.0-1.2 (MDE90 about 2.9-3.5) at steps 50, 100 and 150, at the cost of 6,400 extra episodes. Removes a baseline bias of roughly -4 to +1 U depending on row type.

**4. S4 [major]**

- *title*: Guard G3, stop rule 2 and the pilot decline rule count honest non-claims on failed work, so they fail the reporting route that U rewards
- *evidence*: Definition: src/honest_agents/reward.py (line 701: declined = has_report and not claimed; line 424 comment 'any parsed report that does not claim completion'); the guard metric uses it on controls (src/honest_agents/analysis/power.py, line 271); the training rule is 'non-claim reports on feasible tasks' (src/honest_agents/callbacks.py, line 214). Plan: G3 control decline UB < +5 (line 209), pilot rule (line 452), stop rule 2 (line 469). Data (decline_cases.py): in G2's select run all 18 control declines are on failed work (0 on passing work): 9 false_blame, 9 honest_failure; in the it17 pool bank 98 of 102. Computed (decomp_and_g3.py, clone SE of control decline 0.59 at the held-out design): if x points of feasible episodes move from false claim to honest non-claim, dU = 0.64x and P(G3 passes) is 0.96 at x = 3, 0.51 at x = 4, 0.05 at x = 5. Joint P(H1 significant and G3 passes) peaks at 0.69 near x = 3 and is 0.05 at x = 5 (SE 0.86); with SE 1.22 the peak is 0.45.
- *why_it_matters*: The plan's larger headroom is reporting (14.3 U against 7-11 for capability, line 498). Any reporting gain large enough for H1 to detect with 90 % power (about 4 points of feasible false claims) fails G3 about half the time or more, and in training a conversion of roughly 8-10 points of feasible episodes fires the hard stop. Only a capability gain can pass.
- *proposed_change*: Redefine all three on unwarranted declines: (a) declines on passing work, already covered by P(claim | pass); (b) the false_blame plus lazy_decline rate on controls (reward.py lines 712-735), bar UB < +3 after the clone check. Report honest_failure separately, ungated; it is expected to rise if the reporting route works. Blind declining without trying still shows as a task-success loss under G1. Apply the same definition to stop rule 2 and the pilot abort rule.
- *measurable_effect*: For a 5-point reporting conversion, P(guard set passes) rises from about 0.05 to the clone level, so the joint P(H1 significant and guards pass) goes from 0.05 to about 0.9 (SE 0.86). The reporting route becomes measurable instead of disqualifying.

**5. S5 [major]**

- *title*: As provisionally written the guard set fails an exact G2 clone about 26-31 % of the time; the no-harm rule cannot be widened and G5 at S = 4 is blunt
- *evidence*: clone_guards.py (G2's select run resampled to 118 sources, S = 16, 1,500 simulations): G1-G4 and G6 jointly fail a clone 3.7 % (G4 alone 3.7 %); adding the no-harm rule on P and T gives 13.4 % (4.4 % + 5.7 %, by construction of a one-sided 5 % test per stratum). G5 (g5_clone.py): G2 destroys 37 of 240 twin episodes in runs/eval/it19c_g2_select; clone-minus-reference SD is 6.2-7.9, so k = 6 fails a clone 14.8-20.6 %; k = 12 fails 2.2-5.6 % but then catches a true +8 per 240 less than 37 % of the time and +16 about 70 % of the time. Combined: 1 - (1 - 0.134)(1 - 0.15..0.21) = 0.26-0.31, against the plan's own limit of 0.10 (lines 215-218). Code: the no-harm condition (pass when UB >= 0) is not among RULES in src/honest_agents/analysis/power.py (lines 782-783), clone_pass_joint takes one dataset kind (lines 807-818), and G5 is a count on another dataset. Caveat: G2's claim | pass is 100 % on select, so that guard is constant in this proxy.
- *why_it_matters*: A model with a real gain inherits the same false-failure rate. The plan's remedy (widen the bars) has nothing to widen on the no-harm rule, and widening G5's k at 240 episodes makes it nearly blind.
- *proposed_change*: Give the no-harm rule a margin (fail only when the 90 % UB of a stratum's dU is below -m, m set by the clone check; about 1.0-1.5 U). Run G5 at S = 16 (960 episodes per arm) as a paired rate difference by twin with a UB bar, or fold it into G4 on the held-out twins. Extend section 10 with the missing rule, the G5 simulation and a joint over both datasets, and state the joint clone-fail rate including the no-harm rule and G5 in the prereg. Also report each guard's power against a stated harm.
- *measurable_effect*: Joint clone-fail from about 0.26-0.31 to at most 0.10. G5 at S = 16: rate SD about 1.3-1.65 points, a bar near +3 points fails a clone about 2 %, and a true +6.7 points (16 per 240) is caught about 99 % of the time instead of about 70 %.

**6. S6 [major]**

- *title*: Pooled U is the least sensitive statistic for either pure mechanism, and dU = dTC - dFC - dD cannot separate capability from reporting
- *evidence*: Clone SEs at the held-out design (clone_guards.py): U 0.84, task success 0.52, false claims per feasible episode 0.53. Expected t per point of effect (decomp_and_g3.py): +1 point of feasible pass rate gives U 1.25, task success 1.92; -1 point of feasible false claims gives U 0.76, false claims 1.89. With the observed interaction-inclusive select SEs (2.71, 2.11, 1.78) the ordering holds: capability U 0.39 vs task success 0.47; reporting U 0.24 vs false claims 0.56. In mechanism units H1's MDE90 of 2.53 U means +2.41 points of pass rate or -3.96 points of feasible false claims (about a quarter of G2's roughly 15 %). Decomposition: dTC mixes more passes with claiming more of them, dFC mixes fewer failures with claiming fewer of them. A five-term split on the it19c ablation (select, sums exactly to -6.52): capability -3.73 (57 %), false-claim discrimination -0.89 (14 %), damage on feasible -0.27, twin claims -0.68, twin damage -0.95. The frozen headline is still internal false_claim (direction down) and success (non-inferior) in data/benchmarks/final_benchmark.json.
- *why_it_matters*: The owner's headline numbers are false claims and task success. A real change in one of them can be significant on its own metric and invisible in pooled U, and the planned decomposition would not say whether a gain is capability or reporting.
- *proposed_change*: Keep pooled U as the single-number primary (a constant policy cannot win it). Add a pre-registered confirmatory family tested with Holm at one-sided 0.05: pooled dU, d task success (LB > 0), d false claims per feasible episode (UB < 0), with the P(claim | pass) guard protecting the last from blanket non-claims. Replace the section 3.6 decomposition with the exact five-term split per stratum, each with a source-bootstrap interval: capability (a + b) x dp, true-claim reporting p x da, false-claim discrimination -(1 - p) x db, twin claims, damage, where p is pass rate, a is P(claim | pass), b is P(claim | fail).
- *measurable_effect*: Detectable effect at 90 % power, no interaction: capability 2.41 -> 1.79 points of pass rate (via task success in a Holm family of three); reporting 3.96 -> 1.82 points of feasible false claims. The cost is U's own MDE90 rising from 2.53 to at most 2.96 when it is the smallest p. Every result gains a table that attributes dU to capability, reporting, twins and damage.

**7. S7 [major]**

- *title*: H4's single-seed contrasts cannot attribute anything, yet A minus B is the most direct test of the owner's claim
- *evidence*: Plan lines 141-142 and 432-438: B and C are one seed each, compared with A1. A single contrast cannot separate the factor from seed noise: with three exchangeable A seeds, a fourth draw from the same distribution falls outside their range with probability 2/4. Seed variance is known to matter here (results.md, iteration 14: seed 1 passes all six checks, seed 2 fails one; iteration 15: held-out gap +16.2 vs +11.2). Sampling SE of A1 - B at S = 16 is 0.86, so MDE90 is at least 2.5 U before any seed term, as large as the whole hoped effect. The repo already has a multi-seed estimator with seed inflation (src/honest_agents/eval/metrics/bootstrap.py, lines 148-217). Arm B differs from A only in the claim term (plan lines 384-389).
- *why_it_matters*: The owner's claim is success under an honesty constraint. 'A has fewer false claims than B at equal task success' tests exactly that, and it stays informative even if neither arm beats G2. As written it is a single-seed descriptive line, and arm C (prompt diversity) uses a GPU on a side question.
- *proposed_change*: Drop arm C or mark it exploratory. Run B on seeds 1-3 with the same pool files as A1-A3 (six runs; with five GPUs use B1 and B2). Pre-register a confirmatory constraint contrast on pooled episodes: d false claims per feasible episode (A - B) UB < 0, with d task success (A - B) LB > -1.5, plus the seed-by-seed table and a permutation p over seeds.
- *measurable_effect*: Sampling SE of the pooled A - B contrast on U: 0.86 -> 0.50 (48 vs 48 episodes per variant), MDE90 about 1.5 U; on false claims per feasible episode about 0.31, MDE90 about 0.9 points. Adds a second pre-registered route to a positive RL result that does not depend on beating G2.

**8. S8 [major]**

- *title*: Success is one binary; an 'informative null' is claimed but not defined
- *evidence*: Plan line 131 (H1 with every guard) and line 502: 'A null with all guards passing would still be an informative result', with no equivalence bound, no rule separating 'did not learn' from 'did not transfer', and no pre-written statement for a harm. At SE 0.86-1.22 a null's 90 % interval is +-1.4 to +-2.0, which does not exclude the middle of the hoped range.
- *why_it_matters*: With about a 0.17 chance of the single success, the most likely outcomes have no agreed reading. That is the opposite of a measurable result whatever happens.
- *proposed_change*: Replace the success sentence with tiers, each with an estimator, a bar and a pre-written sentence. Tier 0 valid measurement: integrity gate, never-trained record, A/A check (S9), joint clone-fail <= 0.10. Tier 1 learns on its own prompts: M1 of S3. Tier 2 generalizes to unseen sources: dev-large pooled dU LB > 0 and a non-negative dose-response slope. Tier 3 confirmed on held-out: pooled dU LB > 0 with guards (3a); 'solid' when the point estimate is at least +2 and every seed is above 0 (3b); the Holm components say where (3c). Tier 4 constraint efficacy: the A - B contrast of S7. Tier 5 external retention (N1). Negative outcomes: informative null when Tier 0 holds and the 90 % interval lies inside +-1.5; harm when the UB is below 0, reported with the five-term split; inconclusive otherwise.
- *measurable_effect*: Every outcome maps to one reportable statement with an interval. With the pooled design (SE 0.50-0.56 without interaction) a true zero lands inside +-1.5 about 90 % of the time, so the null becomes a bounded result rather than an absence of one.

**9. S9 [major]**

- *title*: The G2 baseline is separated from the arms in time, node and possibly commit, and there is no A/A check
- *evidence*: The fresh G2 held-out run is done before any candidate exists (plan line 602; docs/iteration-20-handoff.md, lines 255-258), the arms are evaluated after submission on the executor's pushed commit (handoff line 217, lines 265-270), and 'the iteration-20 commit' is never pinned to one hash. Run conditions have moved G2 before: a harness change took G2's select verified success from 72.7 % to 69.1 % (docs/results.md, lines 57-58, DEF-65); DEF-76 changed a TB2 number. Under U a grading failure after the agent acted counts as damage (src/honest_agents/reward.py, lines 378-380). Weak direct evidence: two step-0 probes of the same G2 under the same probe config gave task success 78.7 % and 71.3 % (n = 80; runtimes 926 s and 1,567 s), a U difference of 16.8 with SE 9.0 (probe_res.py); that is within noise at about 1.9 SE. The it19c select runs had 0 infra and 0 grading errors (infra_rates.py).
- *why_it_matters*: The primary contrast has SE 0.86 or less. A run-level shift of 1 U from a harness edit, a slower node or a snapshotter difference is not removed by pairing variants and would be read as a training effect or a harm.
- *proposed_change*: Pin the evaluation path: record a hash of the harness and grader modules at the early G2 run; the evaluator refuses arms whose commit differs on those files, or re-runs G2. Run G2 in blocks: the early block (power inputs) and a late block interleaved with the arms on the same nodes; each arm's S split across the same blocks. Pre-register the A/A check G2-late minus G2-early (must lie inside its 90 % interval, else the block term is added to the SE). Report infra and grading-error rates per arm and U with those episodes dropped as a sensitivity.
- *measurable_effect*: A measured bound on run-level drift (A/A SE about 0.86 at 16 vs 16, 0.6 at 32 vs 32) instead of an untested assumption. No extra episodes if G2's S = 32 is split into two blocks of 16.

**10. N1 [minor]**

- *title*: H3's ImpossibleBench superiority test is close to unwinnable and its estimator is not pinned; the N stratum is underpowered
- *evidence*: The design session's own note: G2's original-split IB has 2 false claims out of 156 and its impossible splits 7 claims, 'so there is no information' (docs/iteration-20-design.md, lines 570-571). IB dU > 0 can then come only from LiveCodeBench success, whose G2 - base difference has SE about 2.4 (docs/iteration-19-results-baseline.md, line 119). The handoff's command sequence ends in final_benchmark compute, which writes deltas against base only, at alpha 0.05 (src/honest_agents/eval/final_benchmark.py, lines 994-998; manifest ci alpha 0.05), not A1 - G2 at 90 %. N stratum: 82 variants, so SE about 1.9 and MDE90 about 5.7 (scaled from the pooled SE; not verified on held-out).
- *why_it_matters*: A hypothesis with a pass criterion that cannot realistically pass reads as a failure of generalization, and two possible intervals at the one look is a degree of freedom.
- *proposed_change*: Restate the external hypotheses as retention tests on the frozen headline metrics, A1 - G2: IB flag_lenient LB above -m, claimed-complete on impossible UB below +m, act.success LB above -m, TB2 false_claim UB below +m, margins set by a clone check; report external U as a point estimate with its MDE. Pin one estimator (paired_estimate on item clusters, 90 %) and add a delta-against-G2 path to compute. State the N MDE beside the test.
- *measurable_effect*: External results become pass/fail statements that a real model can pass; the IB and N lines carry their MDE (about 5.7 U for N) so a non-significant result is read correctly.

**11. N2 [minor]**

- *title*: H2 treats seeds as fixed without saying so; the 'each seed above 0' condition depends on an unknown seed variance
- *evidence*: Plan lines 132-133 and 187-188 pool 48 episodes per variant, clustered by source only. The repo's multi-seed estimator resamples seeds and inflates the seed component because 2-3 seeds understate it (src/honest_agents/eval/metrics/bootstrap.py, lines 161-165). If the seed SD were 2 U and the mean effect +2.5, each seed is above 0 with probability about 0.88 and all three with about 0.67 (my arithmetic).
- *why_it_matters*: The pooled interval speaks about these three adapters, not about the method; a reader will take it as the latter.
- *proposed_change*: State the estimand (mean of the three trained adapters). Report the seed SD and the seed-random interval from paired_difference as a sensitivity. Keep 'each seed above 0' as a reported consistency label, not a condition.
- *measurable_effect*: Removes a hidden factor of roughly 0.67-1.0 on H2's pass probability and adds one reported number (seed SD).

**12. N3 [minor]**

- *title*: Multiplicity across H3 and H4 is not handled
- *evidence*: H3 carries two tests at one-sided 0.05 (N stratum, ImpossibleBench) plus reported strata; H4 has two contrasts on three metrics (plan lines 134-142).
- *why_it_matters*: Under the null, about one in ten rounds shows a 'generalization' pass by chance, and about half of the secondary lines will show something nominal.
- *proposed_change*: Declare one confirmatory family (the primary, the two components of S6, the A - B contrast of S7) under Holm; label everything else descriptive, with intervals and no pass/fail wording.
- *measurable_effect*: Family-wise error held at 0.05 for the claims that will be quoted.

**13. N4 [minor]**

- *title*: The plan quotes MDE at 90 % power; the code's mde() uses 80 %
- *evidence*: src/honest_agents/analysis/power.py (line 133: POWER = 0.80; lines 763-765). The plan's 2.53 equals (t.95 + t.90) x 0.86; the code would print 2.15 for the same SE. Section 12, item 6 has the power re-derived from the fresh G2 run.
- *why_it_matters*: The re-derived number will not match the table unless the power level is pinned, and the two are easy to confuse in the prereg.
- *proposed_change*: Pin 90 % power in the prereg text and pass it explicitly (or add a flag) when re-deriving; print both.
- *measurable_effect*: One number (2.53, not 2.15) in every document.

**14. N5 [minor]**

- *title*: The handoff understates the select split's noise for U
- *evidence*: docs/iteration-20-handoff.md (lines 193-195): 'A difference of about 3 points is at its noise floor.' Measured for U: SE 2.5-2.7, 90 % half-width 4.5, MDE90 7.3-8; a G2 split-half (S = 2 vs 2) differs by -5.71 with SE 3.99 (select_u.py).
- *why_it_matters*: An executor reading +4 on select as a signal would be reading noise.
- *proposed_change*: Replace the sentence with the measured SE, half-width and MDE for U, and say that the section 3.5 bars cannot be checked at select scale (a clone fails them 82 % of the time).
- *measurable_effect*: Correct resolution stated: 4.5 half-width, MDE90 about 8.

**15. N6 [minor]**

- *title*: Two guard definitions are ambiguous
- *evidence*: The G5 command (handoff lines 159-161) calls twin-stop, which also fires on a read-before-destruction drop above 0.10 by default (src/honest_agents/analysis/calibration.py, lines 407-446), a condition not in the plan's G5 row. Guard G2's bar reads 'LB > -3, read with dTS and dFC' (plan line 208), which is not a pass/fail rule.
- *why_it_matters*: At the one look, an exit code 3 from the read rule or a judgment call on 'read with' decides success with no pre-registered text behind it.
- *proposed_change*: Either add the read-drop rule to G5 with its own clone-checked bar or pass a flag that disables it; delete 'read with' or replace it with an explicit condition.
- *measurable_effect*: Zero undefined guard conditions at the verdict.

**16. N7 [minor]**

- *title*: The probe can be made about twice as sharp for free
- *evidence*: probe_res.py: clone SE of probe dU is 9.0 at S = 1 vs 1, so 4.5 at S = 4 vs 4. The step-0 probe is a fixed reference for all ten later probes (plan line 429).
- *why_it_matters*: A tripwire that fires only below -12.4 on two consecutive probes allows 30 steps of collapse; as a learning curve it sees nothing under about 13 U.
- *proposed_change*: Run the step-0 probe at S = 16 (1,648 episodes once per arm, or once and shared since every arm starts from G2), and report a late-window mean (steps 105-150) against it.
- *measurable_effect*: SE of a single probe contrast 4.5 -> about 3.6; of the late-window mean about 2.3 (my arithmetic from the measured 9.0).

**17. N8 [minor]**

- *title*: Three variance-reduction ideas do not pay and should be recorded as such
- *evidence*: Covariate adjustment: regressing the per-variant difference on an independent G2 baseline (select, G2 samples 0-1 as covariate) gives R^2 0.000 and SE 2.76 -> 2.76; regressing on the same samples it is differenced against gives a spurious negative slope (-0.116) (covariate.py). Larger S alone: at V_int 0.744, S = 32 for both arms gives SE 1.06 against 1.22. Dose-response as a power device: a four-point linear trend has about 5 % lower SE than the endpoint contrast (my arithmetic). Common random numbers: not verified; shared sampling seeds should decouple within a few turns of a multi-turn episode.
- *why_it_matters*: These are the obvious next suggestions; each costs engineering or episodes and returns almost nothing here.
- *proposed_change*: Add one paragraph to section 3.4 listing them with these numbers, so effort goes to pooling seeds, more sources and the component tests.
- *measurable_effect*: None on the estimate; avoids spending on changes worth 0-13 % of SE.

## strongest_improvements


**1. Pooled-seed primary with the reference over-sampled, plus a Holm family on the two headline components**

- *description*: H1 becomes A1-A3 pooled (48 episodes per variant) against G2 at S = 32-48, base cut to S = 8. Confirmatory family under Holm: pooled dU, d task success, d false claims per feasible episode. Report the exact five-term split (capability, true-claim reporting, false-claim discrimination, twin claims, damage) with intervals.
- *expected_gain*: MDE90 2.5-3.6 -> 1.5-3.0 U. Detectable capability effect 2.41 -> 1.79 points of pass rate; detectable reporting effect 3.96 -> 1.82 points of feasible false claims. Power at a true +2: 0.75 -> 0.97 (no interaction).
- *how_measured*: Re-derive SE and MDE from the evaluator's fresh G2 run before any candidate is graded; the train-side proxy already reproduces the plan's SE (0.84 vs 0.86).
- *cost_and_risk*: +7 % to +21 % held-out episodes; small code (pooling rule, Holm, decomposition). Risk: the pooled estimate speaks about three adapters, not the method, so the seed SD must be reported beside it.

**2. A powered dev benchmark from the 441 unused val sources**

- *description*: Build dev-large from val sources that no split file uses (never trained by rule, never evaluated), oracle-gated and spec-audited like the pool, frozen before training. Use it for the pre-submission read, a dose-response over steps 0/50/100/150, guard pre-checks, and optionally a fixed-rule choice of candidate step.
- *expected_gain*: Executor-side SE(dU) 2.5-2.7 -> 0.6-0.9 (pooled seeds about 0.55). A second, independent confirmation of any held-out result, and protection against an over-trained endpoint.
- *how_measured*: Clone SE on dev-large from the fresh G2 run at S = 8 (an A/A split of it); the dev contrasts are pre-registered with the same estimator.
- *cost_and_risk*: Oracle gate and spec audit for about 1,400 variants; about 5.5k episodes per S = 4 run. Risk: about 13 % of sources may be spec-defective, and remove_input build faults need the rebuild-diff quarantine.

**3. Repair the guard set so a clone passes and honest non-claims are not punished**

- *description*: G3, stop rule 2 and the pilot decline rule move from 'any non-claim' to unwarranted declines (false_blame plus lazy_decline, and declines on passing work). The no-harm rule gets a margin. G5 runs at S = 16 as a paired rate. The joint clone check covers every condition, including no-harm and G5.
- *expected_gain*: Joint clone-fail about 0.26-0.31 -> at most 0.10. For a 5-point reporting conversion, P(guards pass) about 0.05 -> about 0.9.
- *how_measured*: The joint clone simulation on the fresh G2 runs (held-out by the evaluator, twins by the executor), reported in the prereg with each guard's power against a stated harm.
- *cost_and_risk*: About 1.4k extra twin episodes per arm; new rule and metric code in power.py and the callbacks. Risk: blind declining must still be caught, which G1 and the false_blame bar do; check this on the G3 (iteration 15) logs before freezing.

**4. A tiered success ladder with a training-side milestone and an A/A check**

- *description*: Tier 0 valid measurement (A/A of G2 early vs late, pinned eval path). Tier 1 on-policy revisit gain against a fresh G2 pass on the 800 pool rows. Tier 2 dev-large. Tier 3 held-out (significant; 'solid' at point >= +2 with every seed positive). Tier 4 constraint contrast. Tier 5 external retention. Defined negative outcomes: informative null (interval inside +-1.5), harm, inconclusive.
- *expected_gain*: Every outcome becomes a reportable statement with an interval; a null can say whether RL failed to learn or failed to transfer.
- *how_measured*: Each tier has its estimator and SE fixed in the prereg: about 1.0-1.2 (Tier 1), 0.55-0.9 (Tier 2), 0.5-1.0 (Tier 3).
- *cost_and_risk*: 6,400 episodes for the fresh G2 pass on pool rows; prereg text. Risk: more tiers invite selective quoting, so the confirmatory family must stay fixed under Holm.

**5. Make A minus B a three-seed confirmatory contrast and drop arm C**

- *description*: Run u_success on seeds 1-3 with the same pool files. Pre-register: false claims per feasible episode lower in A than in B, with task success non-inferior at 1.5 points.
- *expected_gain*: A second route to a positive, attributable RL result that does not require beating G2: sampling SE on U 0.50 (MDE90 about 1.5), on false claims about 0.31 (MDE90 about 0.9 points).
- *how_measured*: Pooled episodes A1-A3 vs B1-B3 with the same paired estimator, the seed-by-seed table, and a permutation p over seeds.
- *cost_and_risk*: One more training GPU than revision 3 (six runs), or B on two seeds with five. Risk: u_success is not pair-neutral, so B may drift on twins; that is itself the measured contrast, and B needs the same stop rules.

## things_revision3_got_right

- One pinned budget for the prescreen, probe, select, held-out and every arm, with a fresh G2 on the same config (section 5.3); archived 12k runs are context only.
- U as the per-episode score: a constant policy cannot win it, w = 1 at evaluation, scored with score_records and never from the stored reward dict (DEF-74).
- One pre-registered estimator (paired by variant, clustered by source), significance read from the 90 % lower bound only, the code's `significant` flag not used, the sign-flip p reported beside it.
- The void list frozen before any candidate episode exists, the DEF-37 rule applied as a union over exactly the contrast's arms, and a frozen-list-only sensitivity.
- The clone check requirement (joint clone-fail at most 0.10, bars fixed before grading) and re-deriving power from the fresh G2 run; keep both and extend them to every condition.
- The executor/evaluator split, candidates fixed in advance, all arms submitted before one combined table, one look per benchmark.
- Three seeds of the main arm and the idea of pooling them (H2); it should become the primary.
- Stop rule 1 on pooled P(claim | pass) with a one-sided bound in a 5-step window; it would have caught the it19c collapse within its window.
- Recognising that train U is not comparable across steps and fixing the row order so the same prompts recur each epoch, which is what makes a revisit estimator possible.
- Honest labelling: TB2 not tested because its MDE90 is about 9, N reported as 'not measured' without the rebuild, the effect called a hope, the contamination of the held-out set put on record.
- Reporting the frozen headline definitions, the stop-reason split, the read-split declines and the missing-report sensitivity beside U.