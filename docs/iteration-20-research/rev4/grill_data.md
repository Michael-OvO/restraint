# grill:data


## lens

Training data, pool and curriculum: is pool v5 the right one to produce a large, general, measurable effect?


## verdict

The pool is legal, clean at the text level, and about the right size, but it is unlikely to deliver a significant H1 as written. My probability of a significant positive H1 is about 15 % (range 10-20 %).

Why so low:
- The pool trains only rows G2 already passes about 76-86 % of the time. Those rows hold about 27 % of G2's feasible false claims on the select split; 63 % sit on rows G2 never passes, which the band excludes.
- If RL closes the same share of the pass gap as it9 did from base (31 %), pooled held-out U moves about +1.2 to +2.5, against an MDE of 2.5-3.6.
- The one RL-from-G2 run with a KL term showed no gain on its own training rows.
- About 64 % of groups per step carry no pass/fail contrast, and the twin weights are about half what the document states.

The plan also leaves three train-side resources unused that would make a real effect measurable:
- about 435-440 val sources never used for anything;
- an existing surface evaluation set on never-trained sources;
- the archived G3 adapter, which under plain U scores +14.3 [+5.5, +23.1] over G2 on 35 never-trained sources (post hoc, one seed, S = 2).

With those changes I would put the chance of at least one pre-registered, significant positive RL result at roughly 40 %.

All scratch scripts are in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/pool-lens/ and use train-side data only.


## findings


**1. D1 [major]**

- *title*: Re-scored under plain U, the archived G3 run beats G2 on never-trained sources; the plan calls the it19c ablation the only comparable run
- *evidence*: Plan text: docs/iteration-20-rl-design.md (line 119, 'The only comparable run lost capability'; line 115, G3 named as the failure to avoid).

I scored archived runs with plain U (w = 1), paired by source, DEF-66 out, 90 % t intervals (scratch u_history.py, g3_tierS.py, surf_eval.py):
- G3 c15 minus G2, tier S (data/probe_v1.jsonl, 35 val sources, 240 episodes per arm, S = 2): +14.3 [+5.5, +23.1]; G2 level +36.3. Twins +17.2 [+3.2, +31.1], controls +11.7 [-6.3, +29.7], plain +7.1 [-4.4, +18.7].
- G3 c15 minus G2, surface set (data/decision/surfaces_val, 17 val sources, 960 episodes per arm): +10.8 [+5.1, +16.5]; G2 level -32.2.
- it13 GRPO c10 minus its SFT start, tier F (89 val sources): +3.2 [-1.3, +7.8], controls +19.1 [+6.3, +32.0]; new-surface set +9.9 [+1.0, +18.8].
- it19c ablation c30 minus G2, select (52 sources, S = 4): -5.9 [-10.6, -1.2].

Raw tier S counts, G2 to G3: twin claims 14 to 5 of 80, twin damage 8 to 1, plain passes 64 to 66 with claims 77 to 72, control passes 59 to 63. Both arms used the same config (12,288 tokens, 36 rounds).

docs/iteration-15-g3-rl-read.md (lines 88-92) already says G3 'is the best model so far on the familiar set'.

The adapter is archived: ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/checkpoint-15/adapter_model.safetensors (259.8 MB).

Pools v3 (G3) and v4 (it19c) have the same composition: 45 twins, 45 controls, 30 plain on 64 and 65 sources (data/train_rl_v3.report.json, data/train_rl_v4.report.json). So the pool is not what differed; the reward, the KL term and the second pass are.

Caveats: post hoc, several cuts examined, one seed, old harness and pre-DEF-74 grading. G3 failed its registered endpoints (read rate 46 to 46 %, surface control declines 31 to 41 %).
- *why_it_matters*: The expected effect ('0 to +5, a hope') and the decision to neutralise the twin term rest on an incomplete reading of history. Under the plan's own metric, an existing RL adapter may already clear the H1 bar, and testing it costs no training.
- *proposed_change*: 1. Add a table to section 1 with every past RL run against its starting model under U, on train-side sets.
2. Add arm R0, the archived G3 c15 adapter, fixed in advance. The executor evaluates it against a fresh G2 on the val dev set (D2); the evaluator runs it on the held-out set at the pinned budget.
3. Soften line 119 to say the it19c ablation is one of two comparable runs.
4. If R0 confirms, an arm with G3's reward on pool v5 is the natural attribution arm (a method-lens decision).
- *measurable_effect*: R0 minus G2 on val dev at S = 8 (about 11k episodes per arm) has SE of about 0.7-0.8 U. A true effect of +3.5, a quarter of the post hoc estimate, would be detected with power above 0.99. On the held-out set at S = 16, power for +3 is 0.80-0.97 using the plan's SEs of 1.21 and 0.86.

**2. D2 [major]**

- *title*: No well-powered train-side measurement: about 435-440 never-used val sources and an existing surface evaluation set sit idle
- *evidence*: docs/iteration-20-handoff.md (lines 193-195): the select split has 52 sources after DEF-66 and 'a difference of about 3 points is at its noise floor'. Line 65 confirms every val task dir is in the train bundle.

Inventory from ~/Documents/honest-agents-archive/staging/analysis/it20_design/legal_rows.json:
- The val split has 539 sources and 1,883 variants in data/tasks.
- 1,645 variants on 529 sources pass every legality rule except the split.
- Removing the sources of data/val.jsonl and data/probe_v1.jsonl leaves 1,372 variants on 440 sources: 438 plain, 300 high_risk pairs, 127-129 revoke pairs, 39 remove_input pairs.
- 435 val sources are referenced by no jsonl under data/, data/decision, data/it17, data/it19, data/judgment or data/pools.

The frozen held-out set has 118 sources (data/benchmarks/internal_heldout.jsonl: 562 rows, all under data/tasks).

Surface sets on never-trained sources already exist:
- data/decision/surfaces_val/tasks: 240 dirs, 20 val sources, 6 surfaces;
- data/decision/surfaces/tasks on select sources: 32 trained-surface pairs, 24 inbound_link and 24 empty_backup pairs.
Total about 200 pairs on 48 sources. I did not verify whether the select-source surface pairs are oracle-gated.

G2's U there is -32.2 per 100 (surf_eval.py), against +33.3 on the internal held-out set.

No iteration-20 evaluation set contains a surface task: 0 of 562 held-out rows, 0 of 212 select, 0 of 60 G5 twins, 0 of 120 probe_v1.

Power scaling from the plan's own table (design lines 192-196): SE squared = 0.018 (+0.744 with interaction) + 11.54/S at 118 sources and 431 P and T variants.
- *why_it_matters*: The plan's primary can detect only 2.5-3.6 U and its development set only about 3 points. The effect history supports is at or below that. A larger dev set of the same format, never trained and never looked at, would turn a likely 'null within MDE' into a measured number. It also lets the method be iterated without spending the single held-out look.
- *proposed_change*: 1. Define val-dev: the 440 val sources outside val.jsonl and probe_v1, P and T families only. Extend the prescreen to it (S = 8, about 11k episodes, +28 %) to get a fresh G2 baseline at the pinned config.
2. Pre-register dev ΔU for every arm at its candidate step, one look per arm by the executor. Submit to the evaluator only arms the executor has already frozen.
3. Add stratum S: the 200 surface pairs on 48 never-trained sources, reported separately for trained and held-out surfaces.
4. Keep the held-out set as the headline.
- *measurable_effect*: Dev-set MDE90 on pooled P and T: about 1.4-1.9 U at S = 16 (22k episodes per arm) and 2.0-2.4 at S = 8, against 2.5-3.6 on the held-out set. These are extrapolated from the plan's SE table, not measured.

Stratum S: SE about 2.0 at 45 sources and S = 8, scaled from the G3 minus G2 interval, so MDE90 is about 6 U. G2's level there is -32 and both past positive RL runs moved it by about +10.

**3. D3 [major]**

- *title*: The band rows the pool trains on hold about a quarter of the false claims; the reachable effect sits at or below the MDE
- *evidence*: Select split, G2, 8 pooled samples per variant from three runs, DEF-66 out, 118 feasible variants (scratch headroom_tiers.py):
- never-pass (0/8): 13 % of variants, 63 % of G2's feasible false claims (97 of 154), claim-given-fail 0.81;
- band (4-7/8): 31 % of variants, 27 % of false claims, pass 0.78;
- always-pass (8/8): 53 % of variants, no false claims.
If every band-row episode passed and was claimed, U would rise 11.4 per 100 feasible episodes (plain 8.2, controls 14.0).

Independent corroboration: docs/iteration-19-research/badcase-it19c/ablation_select.md (section 2, paired variant buckets): the 17 'both fail all 4' variants carry 51 of G2's 75 false claims.

Held-out P and T is 64 % feasible (118 plain plus 158 controls of 431), so the select-based ceiling is about 7.4 U pooled. The published held-out S = 2 numbers give about 12 (docs/iteration-20-research/u_checks.txt, lines 4-20: pass@1 against pass@k).

History: it9 from base gained +11.6 points of pass (SE 2.1) on sources never trained before, steps 21-50, which closed 31 % of the gap (it9_sources.py). The gain sat on rows base passed 1-2 of 4 times (+0.26 to +0.33) and was +0.05 on 3-of-4 rows.

The same closure on G2's band rows (true pass 0.76-0.86) gives about +1.2 to +2.5 U pooled. That is my arithmetic, not a measurement.

it19c ablation from G2: pass gap against G2's bank on the same training rows was -4.6 points (SE 3.3) in pass 1 and -3.5 (SE 3.6) in pass 2 (abl_gap.py).
- *why_it_matters*: Section 9 quotes 14.3 U of report headroom and 7-11 of capability. A solve-gradient method on band rows cannot reach most of the report headroom, because it sits on rows G2 never passes. H1 needs a third to a half of the whole reachable ceiling.
- *proposed_change*: 1. Restate section 9 with the tier table and the it9-proportional forecast.
2. Add a pre-registered secondary endpoint H1b: ΔU on the headroom stratum. Define it only from runs independent of the comparison: variants where the evaluator's base S = 16 run passes fewer than 16 of 16, or the archived it19a G2 run failed at least once. Do not define it from the fresh G2 run, which would add regression to the mean.
3. Report ΔU by tier (never, band, always) in every results table.
- *measurable_effect*: For an effect that lives on band rows, the z-score of the stratum endpoint is about 1.6 times the pooled one. That is my variance arithmetic: per-variant within-variance 0.055 for band rows, 0.0075 for always-pass rows, 0.0375 for twins, at S = 16. Power for a +1.35 pooled effect rises from 0.30-0.47 to about 0.8. No extra episodes are needed, because base at S = 16 is already planned.

**4. D4 [major]**

- *title*: Twin weights are about half what the document states, two unit kinds sit on the drop rule, and the fallback cannot be filled in band
- *evidence*: docs/iteration-20-rl-design.md (lines 115 and 370) work the weight with δ = 0.1 and get w of about 0.44. Lines 377-378 drop a kind at w ≤ 0.2 and give its slots to plain rows.

Measured P(damage | claim) on twins in train-side G2 banks (scratch delta.py):
- surface twins 0.97-1.00 (719 episodes; it16b);
- data/tasks high_risk twins 0.95 (218 episodes; it17a);
- remove_input and revoke 0.00.
The 0.1 in the document is the unconditional damage rate.

With measured δ and control pass rates (pairs.py):
- surface: p_c 0.726, w 0.23;
- data/tasks high_risk: p_c 0.80-0.84, w 0.30-0.35;
- remove_input: p_c 0.59-0.63, w 0.18-0.26;
- revoke: w 0.66.

P(kind dropped) by a normal approximation: surface about 0.14, remove_input about 0.44.

Paired controls have no pass floor in section 4.4 (lines 289-297). Among pairs with an in-band twin, controls passing at most 1 of 4 are 12 % (surface), 30 % (remove_input) and 3 % (high_risk).

In-band plain supply is about 291 [184, 421] on the select-based estimate and about 360 on the bank-based one. The plan already uses 225 of them, and P(fewer than 225) is 0.17 (band.py, band_bank.py).
- *why_it_matters*: At w = 0.23 a destroyed and falsely claimed twin costs 0.45 in training and 2 at evaluation, where 36 % of P and T variants are twins. The twin rows take 25 % of episodes and carry a few percent of the advantage. Whether 134 of the 200 twin rows exist at all is decided by a noisy threshold after the prescreen. Rarely passing controls also push 'decline on controls', which is G3's signature.
- *proposed_change*: 1. Correct the worked numbers in sections 1 and 5.1 to the measured δ.
2. Neutrality caps the twin penalty at 2·p_c − 1, so choose pairs whose control passes reliably: control at least 6 of 8 on one half of the prescreen samples, with p_c estimated on the other half.
3. Write the pool composition for each drop outcome into the prereg before the prescreen.
4. Give dropped twin slots to in-band controls, not plain rows.
- *measurable_effect*: Twin penalty per proceed, w·(1 + δ), rises from about 0.45 to about 0.70 at p_c 0.85. P(surface kind dropped) falls from about 0.14 to about 0. The share of paired controls passing at most 1 in 4 falls from 12-30 % to 0. All three are read from `analysis.u_dynamics weights` on the prescreen. The blind-neutrality check within ±0.05 still applies, with p_c taken from the held-back half.

**5. D5 [major]**

- *title*: A quarter of the pool is surface pairs on at most 90 sources, in evidence formats no iteration-20 endpoint tests
- *evidence*: docs/iteration-20-rl-design.md (lines 289-297): 100 surface pairs (200 rows) against 50 data/tasks high_risk pairs.

No evaluation set contains a surface task (D2). The held-out T stratum is 86 + 86 high_risk, 42 + 42 revoke and 30 + 30 remove_input variants, all in data/tasks format.

Legal supply: 902 data/tasks high_risk pairs on 902 sources, against 360 surface pairs on 90 sources. Each source's surface pairs share one instruction and one deletion target.

Ranking by variant hash, 100 surface pairs land on about 68 distinct sources (my calculation from about 262 in-band pairs on 90 sources).

By held-out headroom (U best-of-k minus U, u_checks.txt lines 4-20): high_risk controls hold 29 % of the P and T deficit, plain 28 %, remove_input controls 14 %, revoke controls 8 %, all twins 21 %.

On non-readme trained surfaces G2 claims on 47-58 % of twins and destroys 48-59 % (delta.py). On held-out high_risk twins it claims 8.7 % and damages 7.6 %.
- *why_it_matters*: The pool spends 12.5 % of episodes on surface twins whose behaviour no endpoint sees. A third of its paired-control solve signal comes from 7 % of the legal sources. The stratum with the most held-out headroom, high_risk controls in data/tasks format, gets 50 pairs out of 902 available.
- *proposed_change*: Keep surface pairs, because they are where both past positive RL runs gained most (D1), but make them count:
1. measure them with stratum S (D2);
2. at most one surface pair per source, so at most 90, about 50 in the pool;
3. raise data/tasks high_risk pairs to about 100, chosen by the D4 rule.
- *measurable_effect*: Distinct sources behind the pair rows rise from about 170 to about 200 or more. Rows in the held-out T format rise from 81 % to about 90 % of the pool. Surface training becomes a measured quantity (stratum S, MDE about 6 U) instead of an unmeasured cost.

**6. D6 [major]**

- *title*: Only about 5.8 of 16 groups per step carry a pass/fail contrast; allocation by source and by pass count would raise that by 25-45 %
- *evidence*: Fit of G2's true pass-rate distribution on the select split (scratch band.py, band2.py, src_corr.py):
- Plain rows: 23 % in band (4-7 of 8), 62 % at 8/8, 15 % at 0-3 of 8.
- 59 % of in-band plain rows are 7/8 rows with true pass about 0.90. A group of 8 on such a row is mixed with probability 0.46, against 0.71-0.90 for 4-6 of 8 rows.
- 41 % of in-band visits are all-pass, and 17 % of in-band rows are all-pass on all three visits.
- Band membership repeats on an independent S = 8 screen only 57-59 % of the time.
- 8/8 rows are mixed with probability 0.15 (plain).

Per step under section 4.4: 4.5 × 0.59 + 1.5 × 0.15 + 2 × 0.62 + 4 × 0.41 = 5.8 mixed feasible groups.

The 75 always-pass rows and 16 revoke pairs are 13.4 % of episodes. The 75 rows supply about 3 % of the pool's pass variance. Revoke twins are claimed 2 % of the time in the bank, and revoke or remove_input twin groups were zero-variance 67-73 % of the time in real GRPO groups (zerovar.py, it19c ablation and G3 completions).

Difficulty is a property of the source: plain and high_risk-control pass counts correlate 0.92 across 52 select sources. About 40 % of sources [30, 52] have any in-band feasible row, so about 520 of the 1,290 legal sources [384, 667].

Section 4.4 ranks by variant hash (lines 314-315), so a source's plain row and its control are both picked. I estimate about 283 distinct in-band base tasks among the 325 in-band plain and lone-control rows.
- *why_it_matters*: The reward's gradient is mostly 'solve', and it exists only in groups with both a pass and a fail. Two thirds of each step's compute produces none. Retention rows duplicate what the KL term and the 38-41 % all-pass visits already provide.
- *proposed_change*: 1. One feasible unit per source across the whole pool.
2. Fill feasible slots in ascending observed pass count: 4-6 of 8 first, then 7 of 8.
3. Drop the 75 always-pass rows and the revoke pairs. Keep revoke as in-band lone controls.
4. Step composition: 5 in-band plain, 5 in-band lone controls, 3 pairs (2 data/tasks high_risk and 1 surface, with high-pass controls per D4). That is 250 plain, 250 lone controls and 150 pairs: still 800 rows, 3 epochs, 150 steps.
- *measurable_effect*: Mixed feasible groups per step rise from 5.8 to 7.2-8.4 (+25 to +45 %). Summed within-group pass variance per step rises from 1.06 to 1.33-1.70. Distinct in-band base tasks rise from about 283 to about 500. All three are computed by `analysis.u_dynamics` on the prescreen before the pool is frozen. Twin episodes per step fall from 32 to 24.

**7. D7 [major]**

- *title*: Row baselines come from the samples used for selection, so there is no unbiased learning curve and no positive-learning gate before the held-out look
- *evidence*: docs/iteration-20-rl-design.md (lines 280-285): g2_pass and the other per-row baselines are written from the S = 8 prescreen that also selects the band. Lines 475-476 say train U is not comparable across steps, so the only training diagnostic is the revisit slope.

Under the select-based fit, in-band rows observed at 0.80 have a true pass of 0.86, so a spurious +6 points at step 1. The bank-based fit gives less, and 13 pool-v4 rows show +0.6 to +3.8 (rtm.py, band.py). The bias is the same size as the +4 to +9 points being sought on band rows.

In it9 the paired gap against the base bank on the same variants was the signal that worked: -0.4, +2.6, +10.1 and +13.3 points on never-seen sources over steps 1-10, 11-20, 21-30 and 31-40 (it9_sources.py; docs/training-log.md, lines 476-477).

The pilot in section 6 has abort rules for harm only.
- *why_it_matters*: Epoch 1 is 50 steps of first-visit rows on new sources, which is a free generalisation measurement, but only against unbiased baselines. Without a positive gate, five arms run 150 steps and spend the one held-out look even if nothing was learned on the training rows, as in the it19c ablation.
- *proposed_change*: 1. After the pool is frozen, run G2 once more on the 800 pool rows at S = 8 (6.4k episodes). Use that run for g2_pass, g2_u, the stop-rule baselines and p_c.
2. Log the per-step paired ΔU and Δpass against it, by row kind.
3. Pre-register an epoch-1 gate: pooled over A1-A3, paired Δpass on first-visit feasible rows in steps 26-50 has a one-sided 95 % lower bound above 0. If it fails, stop, record a train-side null, and do not submit.
- *measurable_effect*: Steps 26-50 hold about 300 feasible groups per seed. With it9's per-group spread the SE is about 1.5 points of pass for one seed and under 1 point pooled over three. A +4 point gain on band rows is visible by step 50. The cost is 6.4k episodes (+17 % of the prescreen).

**8. m1 [minor]**

- *title*: 800 rows × 3 epochs against more unique rows seen once: keep 3 epochs, but the revisit slope cannot separate learning from memorisation
- *evidence*: Supply: in-band feasible rows are about 290-360 plain plus 500-580 controls, on about 520 sources (D6). 2,400 unique rows would be mostly 8/8 rows, giving about 4.6 mixed groups per step against 5.8 for the plan.

History:
- it9 at steps 41-50: revisited rows +9.5 points (SE 4.6) against never-seen sources +11.6 (SE 5.1) (it9_sources.py).
- it19c ablation: pass 2 minus pass 1 on the same 75 feasible rows +1.5 points (SE 2.3) (revisit.py).

No past run trained on more than 132 sources or more than 2 passes. Sizes: it9 360 rows on 132 sources; pool v2 370 on 108; pools v3 and v4 120 on 64 and 65.

src/honest_agents/train/rows.py (lines 157-161) refuses duplicate variant ids, so the file-level choices are 800 × 3, 1,200 × 2 or 2,400 × 1.
- *why_it_matters*: Repeats neither helped nor hurt at up to 2 passes, so the plan's choice is defensible. The third pass is new territory, and the revisit-slope diagnostic confounds memorisation with learning.
- *proposed_change*: Keep 800 × 3. Report epoch 2 minus epoch 1 and epoch 3 minus epoch 2 on the same rows beside the dev-set ΔU (D2) at steps 50, 100 and 150. A train gain without a dev gain is memorisation.
- *measurable_effect*: The gap between the revisit gain on pool rows and the dev gain at the same step, in points of pass. It is zero under pure generalisation.

**9. m2 [minor]**

- *title*: Several selection and fallback rules do nothing or contradict each other at S = 8
- *evidence*: docs/iteration-20-rl-design.md:
- Lines 316-317: widening the twin band by 0.05 per end up to [0.15, 0.85] adds no pass count, because rates at S = 8 are multiples of 0.125.
- Lines 291-293: the twin band admits 8/8 ('always claims') but not 7/8.
- Lines 319-320: 'share of plain rows with G2 p < 0.5' is 0 by construction, because the lowest in-band value is 4/8.
- Lines 263-264 speak of admitted rows passing 1 in 8, which the plain band excludes.
- No fallback is defined for a plain-row shortfall, which has probability 0.17 (D4).
- *why_it_matters*: The executor will have to improvise the pool after the prescreen, outside the prereg.
- *proposed_change*: State every band as pass or claim counts out of 8 (plain and lone controls 4-7; twins 2-8). Define fallbacks in counts. Define the plain shortfall fallback as in-band controls.
- *measurable_effect*: Zero undefined cases when the builder's test enumerates all count combinations; the shortfall record in the prereg stays empty or names a pre-written branch.

**10. m3 [minor]**

- *title*: The family mix of the 100 lone controls is unspecified and would default away from the held-out headroom
- *evidence*: In-band control supply by family (band_bank.py): high_risk about 479, revoke about 72, remove_input about 31, which is 82 / 12 / 5 % under hash order. Held-out T-control headroom from u_checks.txt (lines 8-11): high_risk 58 %, remove_input 27 %, revoke 16 %.
- *why_it_matters*: remove_input controls have the lowest held-out U (0.40) and would get about 5 lone-control rows.
- *proposed_change*: Fix the lone-control quotas by family, for example 55 % high_risk, 25 % remove_input (all in-band ones, about 31), 20 % revoke.
- *measurable_effect*: remove_input control rows in the pool rise from about 5 to about 31. Per-family ΔU on T controls is reported.

**11. m4 [minor]**

- *title*: remove_input pairs are fragile on three counts
- *evidence*: - Supply: 18 of 46 measured legal twins were claimed at least once at S = 2, so about 25 of the 64 legal pairs are in band, against a quota of 34 (legal_rows.json).
- Weight: w is 0.18-0.26, so P(dropped) is about 0.44 (D4).
- Controls: 30 % pass at most 1 in 4, and 12 of 64 never pass (pairs.py).
- *why_it_matters*: The family may vanish from the pool after the prescreen, or enter with controls that reward declining.
- *proposed_change*: Decide now: remove_input enters as in-band lone controls (m3). Its twins enter only as pairs whose control passes at least 6 of 8 on the held-back half.
- *measurable_effect*: P(remove_input absent from the pool) falls from about 0.44 to 0. Its paired controls passing at most 1 in 4 fall from 30 % to 0.

**12. m5 [minor]**

- *title*: A never-trained-family dev stratum exists on val, and it would rehearse the DEF-25 rebuild
- *evidence*: Val-split variants in data/tasks: 66 corrupt_input pairs, 27 conflicting_constraint pairs, 5 no_network pairs (legal_rows.json). The held-out set has 41, 12 and 11. docs/iteration-20-rl-design.md (lines 325-334): N depends on a rebuild that is sensitive to the storage driver, and is 'not measured' if the rebuild is not done.
- *why_it_matters*: H3 on N rests on 41 pairs and a fragile rebuild that only the evaluator runs.
- *proposed_change*: The executor rebuilds the 66 val corrupt_input pairs into a separate directory with the DEF-25 fix, oracle-gates them, and reports dev ΔU on N and C as a rehearsal and a second estimate.
- *measurable_effect*: N-stratum pairs measured rise from 41 to 107 across dev and held-out. The rebuild's twin-corruption rate on val (target 100 %) is known before the evaluator touches held-out.

**13. m6 [minor]**

- *title*: The prescreen may run under a different host load from training, which moves rows across the band edge
- *evidence*: docs/training-log.md (line 1876): on a cold node the slowest tool turn reached 1,720 s. Band membership repeats only 57-59 % of the time even without a load change (band2.py). I have no measurement of load effects on pass rates.
- *why_it_matters*: 59 % of in-band plain rows are at 7 of 8. A small timeout-driven shift changes which rows are selected and biases the baselines.
- *proposed_change*: Run the prescreen and the D7 re-run at the training sandbox concurrency. Record command-timeout counts. Add a pilot check that the pooled pass rate in steps 1-5 is within 3 points of the unbiased baseline.
- *measurable_effect*: Absolute difference between the steps 1-5 pass rate and the re-run baseline on the same rows, in points. The pilot aborts above 3.

## strongest_improvements


**1. Build the measurement from idle train-side data: a 440-source val dev set and the surface stratum**

- *description*: Extend the prescreen to the 440 val sources that are in neither val.jsonl nor probe_v1 (1,372 P and T variants in the held-out's own format) to get a fresh G2 baseline. Pre-register dev ΔU per arm, one look by the executor. Add stratum S, the 200 surface pairs on 48 never-trained sources that already exist under data/decision. Keep the held-out set as the headline.
- *expected_gain*: MDE90 on pooled P and T falls from 2.5-3.6 U to about 1.4-1.9 at S = 16, or 2.0-2.4 at S = 8. Stratum S has an MDE of about 6 U where G2 sits at -32 and both past positive RL runs moved about +10.
- *how_measured*: `analysis.power estimate --metric u` paired by source on val-dev and stratum S, against the fresh G2 from the extended prescreen. SEs are re-derived from that G2 run before any arm is graded.
- *cost_and_risk*: About 11k more prescreen episodes, plus 11-22k per arm on val-dev. The executor holds the set, so it supports the claim but does not replace the custodian-held result. Select-source surface pairs need an oracle gate.

**2. Add the archived G3 adapter as a zero-training RL arm and put a U re-score of history in section 1**

- *description*: Under plain U on never-trained sources, G3 c15 minus G2 is +14.3 [+5.5, +23.1] on tier S and +10.8 [+5.1, +16.5] on the surface set. The it19c ablation is -5.9 [-10.6, -1.2]. Evaluate the archived checkpoint-15 adapter as arm R0, fixed in advance, on val-dev and then on the held-out set at the pinned budget.
- *expected_gain*: A second, independent chance of a significant positive RL result. If the true effect is a quarter of the post hoc estimate (+3.5), power is above 0.99 on val-dev and 0.80-0.97 on the held-out set.
- *how_measured*: R0 minus fresh G2, ΔU with the 90 % lower bound, same estimator and guards as H1. Control-decline and G5 guards are reported, because G3 raised surface control declines from 31 to 41 %.
- *cost_and_risk*: No training. One more evaluation arm (about 9k held-out episodes). The estimate is post hoc, one seed, old harness; expect shrinkage. G3 may fail the control-decline guard on surface-type rows.

**3. Re-allocate the same 800 rows by source and by pass count**

- *description*: One feasible unit per source across the pool. Fill by ascending pass count. Drop the 75 always-pass rows and the revoke pairs. Step composition of 5 in-band plain, 5 in-band lone controls with fixed family quotas, and 3 pairs.
- *expected_gain*: Pass/fail-mixed feasible groups per step rise from 5.8 to 7.2-8.4. Distinct in-band base tasks rise from about 283 to about 500. Compute stays at 150 steps × 128 episodes.
- *how_measured*: `analysis.u_dynamics push` on the prescreen grouped as G = 8: mixed-group count, summed pass variance and solve push per step, before and after.
- *cost_and_risk*: Builder changes only. In-band source supply is 384-667 at 90 %, so the low end forces some 8/8 controls. Twin episodes per step fall from 32 to 24.

**4. Fix the pair design: measured δ, pairs with reliably passing controls, data/tasks high_risk pairs in the majority**

- *description*: Use the measured P(damage | proceed) of 0.95-1.0 for high_risk kinds. Select pairs whose control passes at least 6 of 8 on one half of the prescreen samples, and estimate p_c on the other half. Take about 100 data/tasks high_risk pairs and at most 50 surface pairs, one per source. Write the composition for every drop outcome into the prereg.
- *expected_gain*: Twin penalty per proceed rises from about 0.45 to about 0.70. The chance that a kind is dropped after the prescreen falls from 0.14 (surface) and 0.44 (remove_input) to about 0. Paired controls that rarely pass fall from 12-30 % to 0.
- *how_measured*: `analysis.u_dynamics weights` output per kind; the blind-neutrality check within ±0.05 on held-back samples; in training, unread control declines against G2 on the same prompts.
- *cost_and_risk*: Halving the samples for selection lowers band precision for pairs. High-pass controls give less solve signal, which the lone controls replace.

**5. Unbiased row baselines, a paired learning curve and an epoch-1 gate**

- *description*: Re-run G2 on the frozen pool rows at S = 8 and use that for all baselines. Log per-step paired ΔU and Δpass against it. Require a positive first-visit Δpass over steps 26-50, pooled over seeds, before continuing and before any submission.
- *expected_gain*: Removes a selection bias of up to +6 points of pass, the size of the effect sought. Detects a +4 point gain on band rows by step 50. Protects the single held-out look from a run that learned nothing on its own rows.
- *how_measured*: One-sided 95 % lower bound of the paired Δpass on first-visit feasible rows, steps 26-50, A1-A3 pooled; shown on the W&B dashboard by row kind.
- *cost_and_risk*: 6.4k episodes. A false stop is possible if learning is slow to appear: it9's gain appeared at steps 21-30 at lr 3e-5 without KL, and lr 2e-5 with KL may be slower. Consider placing the gate at step 75.

## things_revision3_got_right

- The source-level legality rule and its exclusions (select, DEF-66, never-trained families, held-out surfaces). The 4,098 variants on 1,290 sources reproduce from the archived inventory. A proxy text check found no near-duplicates across splits: the largest 8-gram containment of any select or val instruction in a legal train plain instruction is 0.093 (scratch neardup.py). I did not check the held-out split, by rule.
- Prescreening at the one pinned budget instead of reusing 12,288-token banks, and at S = 8 on every legal row. Band membership repeats only 57-59 % of the time even at S = 8, so fewer samples would be worse.
- 800 unique rows × 3 epochs. In-band rows sit on about 520 sources, so 2,400 unique rows would be mostly always-pass rows (about 4.6 mixed groups per step against 5.8). Two past runs show no memorisation effect from a second pass.
- Excluding never-pass rows from the pool. They hold 63 % of G2's feasible false claims on select but give no solve signal, and history shows a report-only gradient becomes a blanket shift (it19c method arm).
- Selecting pairs on the twin's claim band and computing p_c on those pairs' controls, which keeps p_c free of selection bias. Any control floor must keep this property by using held-back samples.
- Twin and control in the same block, at most one unit per source per block, and seeds that differ only in block order over one fixed pool, so H2 pools like with like.
- Dropping a kind that cannot be made neutral rather than clipping its weight, and writing weights per row because surface and data/tasks pairs share the high_risk label.
- Prescreen leaks leave the pool and go to the quarantine, and the episode-level leak mask stays on in training.
- Never rebuilding data/ in place; rebuilding both members of the corrupt_input pairs into a separate directory; reporting N as 'not measured' if the rebuild is not done.
- A fresh G2 on the same commit and budget for every comparison, candidates fixed in advance, and all arms submitted before one combined table.