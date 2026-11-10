# evidence:measurement


## summary

1) The executor's current instruments cannot see the effect plan 20 hopes for. For a paired ΔU against G2 (MDE90 = (t.95,df + t.90,df) x SE, df = sources - 1): select split at S=4 has SE 2.8 and MDE90 8.4 U (5.9 at S=8, 4.2 at S=16); the 103-row probe at S=4 has SE 3.3-3.4 and MDE90 about 10; using all four planned select evals (steps 0/60/105/150) as a trend still gives MDE90 8.0. Plan 20's hope is 0 to +5.
2) A much stronger instrument is already in the train bundle and unused: 440 val sources (of 539) that no archived run has ever evaluated and that are not known-defective, with 1,550 built variant dirs: 1,376 P+T variants (440 plain, 300 high_risk, 129 revoke_permission, 39 remove_input pairs), 60 corrupt_input pairs (N) and 27 cue pairs (C). They are not oracle-validated (7 of 1,550) and not spec-audited; expect about 17 % attrition (select kept 176 of 212).
3) Predicted dev-U power (train-side variance, mean within-variant variance of U 0.25): all 440 sources, G2 banked once at S=16, each candidate at S=8 gives SE 0.64 and MDE90 1.9 U with no interaction, 2.5 U with the plan's base-G2-sized interaction; a cheap S=4 screen gives 2.4 / 2.9. The smallest design reaching MDE90 2.5 without interaction is about 250 sources at S_ref 16 / S 8 (6.3k candidate episodes, about 4.8 h at the measured 21.8 episodes/min); with plan-size interaction nothing under about 440 sources reaches it.
4) The plan's power table is internally consistent and matches train-side variance: per-kind G2 variances applied to the held-out P+T shape (431 variants) predict SE 0.85 at S=16 and 0.98 at S=12 (plan: 0.86, 0.99). H1 power at a true +2.5 is 89 % without interaction and 66 % with it; at +1 it is 31 % / 20 %.
5) The guard stack as written fails its own clone check. On a held-out-shaped pseudo-set from the train-pool bank at S=16, G1-G4 and G6 jointly fail an exact G2 clone 0.9 % of the time, but the no-harm rule (90 % upper bound below 0 in P or T) fails a clone 5 % per stratum, 10.5 % jointly, and G5 at k=6 fails a clone 15 % (SD of clone minus reference destroyed twins is 6.2-6.4 per 240 episodes). Meanwhile G2 (P(claim|pass)), G3 and G6 bars sit 7-15 clone-SDs wide and could be tightened for free.
6) Stop-rule noise: rule 1's 'about 345 passes, SE about 0.5' holds (0.35-0.72 pts depending on what is held fixed; false-fire about 0; fires at a true 3.6-4.2 point drop). Rule 3 (probe U) is nearly blind: it fires only below -10.6 U, so a loss the size of it19c's would trip it 10 % of the time per probe.
7) Under U the only comparable RL run lost ground and an older one gained: it19c ablation minus G2 on select is -6.4 U [-11.1, -1.6] (p 0.038); G3 c15 minus G2 is +12.4 [+4.1, +20.7] on probe_v1 (S=2) and +10.8 [+5.0, +16.6] on the val surface set, the latter while control declines rose 31 to 41 %. So U alone can score a caution-shifting run as a win; guards and the per-stratum split carry the claim.
8) The model-by-variant interaction for RL-from-start pairs is unresolved on existing sets (point estimates -0.001, +0.041, +0.076, -0.010 per variant; every 90 % interval includes 0, three include the plan's 0.032-0.039). A dev-U of about 1,100 variants would resolve it before the held-out look.
9) Dose-response adds little power by itself (linear trend over 4 checkpoints at S each: SE 5 % below the end-point contrast, at twice the episodes; the same episodes spent on the end point give 29 % lower SE) but it is a strong shape check (4 increasing means under no effect: 4.2 %) and mean(50,100,150) minus step 0 is 18 % tighter than the end point when the gain plateaus early.
10) H2 pooling buys only an 18 % SE reduction (0.86 to 0.70, MDE90 2.06) because G2 stays at S=16; with G2 at S=48 it would be 0.50 / 1.46. Its interval ignores seed-to-seed variance.


## facts


**1. Val split size and what is built for it in data/tasks (all in the train bundle)**

- *value*: 539 val sources; 1,883 variant dirs, every one with task.toml and directory name matching metadata: 539 plain; pairs (twin+control): high_risk 369, revoke_permission 160, remove_input 45, corrupt_input 66, conflicting_constraint 27, no_network 5. Train split has 4,974 dirs; 4,974 + 1,883 = 6,857 = the train bundle's task-dir count.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/measure/inv1.py and inv3.py (run); tomllib cross-check command (0 missing, 0 mismatches); docs/iteration-20-handoff.md (line 65)
- *confidence*: verified

**2. Which val sources are already used**

- *value*: val.jsonl (tier F): 98 sources / 178 rows; probe_v1 (tier S): 40 sources, a subset of val.jsonl; surfaces_val and the own-prefix decision probe: 20 val sources, all inside probe_v1; the DEF-63 surface decision probe uses 72 train-split sources and 0 val sources; DEF-66 defective: 10 val sources (43 train, 23 test, 76 total); oracle validation rows exist for 104 val sources (194 variants, 188 ok).
- *source*: inv3.py output; src/honest_agents/tasks/select_split.py (lines 100-109); configs/chains/it16_d_final_eval.yaml (lines 224-226) for tier F = val.jsonl, tier S = probe_v1.jsonl
- *confidence*: verified

**3. Unused val sources available for a dev-U benchmark**

- *value*: 440 val sources are in neither val.jsonl nor surfaces_val nor the defective list; 1,550 variant dirs: 440 plain; high_risk 300 pairs, revoke_permission 129, remove_input 39 (P+T = 1,376 variants; 353 sources have at least one T pair); corrupt_input 60 pairs (N = 120 variants); conflicting_constraint 22 + no_network 5 pairs (C = 54 variants). Only 7 of the 1,550 variants have an oracle-validation row. A scan of task ids in about 95 archived train-side eval runs found 0 of these sources ever evaluated; a scan of data/ found them only in split_seeds.json and 4-5 validation ids.
- *source*: inv3.py and inv4.py output (scratch dir above); data-file scan command run in this session
- *confidence*: verified

**4. Existing executor instruments after DEF-66 exclusion**

- *value*: select: 212 rows / 60 sources -> 184 rows / 52 sources (176 scored after the 8 quarantined variants); probe_v1: 120 -> 103 rows / 35 sources (69 feasible, 34 infeasible); val.jsonl: 178 -> 157 rows / 89 sources; select high_risk twins: 60 -> 52 (G5 uses all 60).
- *source*: command run with spec_audit.defective_sources() on data/select.jsonl, data/probe_v1.jsonl, data/val.jsonl, data/it19/select_high_risk_twins.jsonl
- *confidence*: verified

**5. Never-trained-family variants also exist on train-split sources**

- *value*: Train split minus select and defective sources: 139 corrupt_input pairs, 59 conflicting_constraint pairs, 21 no_network pairs (by directory name).
- *source*: inv3.py output, last line
- *confidence*: verified

**6. Handoff reference estimate reproduced**

- *value*: false_claim_feasible: reference 15.9, candidate 19.7, diff +3.8 [+0.8, +6.8], 52 sources, sign-flip p 0.045.
- *source*: env -u VIRTUAL_ENV uv run --frozen python -m honest_agents.analysis.power estimate runs/eval/it19c_g2_select/episodes.jsonl runs/eval/it19c_ablation_select/episodes.jsonl --metric false_claim_feasible --tasks-root data/tasks
- *confidence*: verified

**7. The scratch U scorer matches the published design-session numbers**

- *value*: G2 it19c select without the quarantine void: 184 variants, 52 sources, U 0.3084 (u_checks: +0.308); G2 it17 pool bank: 400 variants, U 0.6663 (u_checks: +0.666).
- *source*: ulib.py (scratch dir) run; docs/iteration-20-research/u_checks.txt (lines 44 and 56)
- *confidence*: verified

**8. G2 per-episode U: level and variance components on train-side sets (P+T; components after removing family-by-kind means)**

- *value*: select it19c S=4 (176 variants, 52 sources): U 36.2; within-variant 0.247; between-variant-within-source 0.085; between-source 0.062. Train-pool bank feasible S=4 (400 variants, 177 sources): U 66.6; 0.269; 0.049; 0.116. Val tier F S=4 (155 variants, 89 sources): U 50.5; 0.216; 0.093; 0.136. Without centring, between-variant-within-source is 0.38-0.41 (twin vs control vs plain means).
- *source*: a1_levels.py output (scratch dir); runs/eval/it19c_g2_select, runs/def74/it17_rollouts_feasible, archive iterations/15-g2-splice-replay/eval/it15_g2_tierF
- *confidence*: verified

**9. Pooled within-variant variance of U per episode by kind (G2, train-side)**

- *value*: plain 0.204; control high_risk 0.294, remove_input 0.378, revoke_permission 0.176; twin high_risk 0.426, remove_input 0.090, revoke_permission 0.008. Weighted to the held-out P+T shape the mean is 0.249. No train-side G2 episode exists on corrupt_input, conflicting_constraint or no_network, so N and C variances were assumed (0.05-0.38).
- *source*: a6_predict.py output (scratch dir)
- *confidence*: verified

**10. ΔU of the only comparable RL runs against G2 on the select split**

- *value*: it19c ablation (30 steps) minus G2, P+T, 176 variants, S=4: -6.39 [-11.15, -1.64], SE 2.84, sign-flip p 0.038 (plain -7.7, control -7.4, twin -4.5; true claims -2.3, false claims +2.8, damage +1.3). it19c method c5 minus G2: -5.5 [-11.7, +0.6], SE 3.66. For the ablation V_obs 8.05 pt^2 = V_samp 8.05 pt^2, so the interaction estimate is 0.
- *source*: a2_pairs.py, a3_bykind.py, a7_history.py output (scratch dir)
- *confidence*: verified

**11. Fresh-G2 reproducibility across commits on select (empirical A/A)**

- *value*: U 36.9 (it16 run, S=2), 39.2 (it17, S=2), 36.2 (it19c, S=4); pairwise differences +2.3, -3.0, -0.7 with SE 3.0-3.2. Destroyed high_risk twins differ more: 13/120 (10.8 %) in it16 and it17 against 37/240 (15.4 %) in it19c.
- *source*: a2_pairs.py and a5_g5_stop_probe.py output; archive iterations/16-it16c-g2-select and 17-it17b-confidence eval runs
- *confidence*: verified

**12. Predicted SE(ΔU) and MDE90 of the existing instruments (clone variances; interaction = the plan's 0.744 pt^2 spread over about 476 variants = 0.0354 per variant)**

- *value*: select: S=4 SE 2.81 / MDE90 8.35 (with interaction 3.15 / 9.35); S=8 1.98 / 5.90 (2.44 / 7.25); S=16 1.40 / 4.17 (2.00 / 5.93). probe (101 scored): S=4 3.41 / 10.2 (3.89 / 11.7); S=8 2.41 / 7.2. val.jsonl: S=4 2.69 / 7.95. Observed check: the ablation pair's SE at S=4 was 2.84.
- *source*: a6_predict.py output; formula SE^2 = sum_k n_k (s2_ref,k/S_ref + s2_cand,k/S_cand)/n^2 + sigma_I^2/n, MDE90 = (t.95,df + t.90,df) x SE
- *confidence*: derived

**13. Predicted SE(ΔU) and MDE90 for a P+T dev-U from unused val sources, symmetric S, after 17 % attrition (nominal in brackets)**

- *value*: 200 sources (515 variants): S=4 MDE90 4.60, S=8 3.25, S=16 2.30 [nominal 4.19 / 2.96 / 2.09]; with plan-size interaction 5.20 / 4.06 / 3.35. 300 sources (777): 3.75 / 2.65 / 1.87 [3.41 / 2.41 / 1.71]; with interaction 4.24 / 3.31 / 2.73. 400 sources (1,034): 3.23 / 2.29 / 1.62 [2.95 / 2.08 / 1.47]; with interaction 3.66 / 2.86 / 2.36. Smallest symmetric S for MDE90 <= 2.5: about 7,000-7,500 episodes per arm without interaction (300 sources x S=9, 400 x S=7); with interaction 400 sources x S=13 or 440 x S=11.
- *source*: a6_predict.py output (scratch dir)
- *confidence*: derived

**14. Asymmetric designs (G2 reference banked once, shared by every candidate), after 17 % attrition**

- *value*: 440 sources: S_ref 16 / S 8 gives SE 0.64, MDE90 1.88 (2.49 with interaction); S_ref 16 / S 4: 2.42 (2.92); S_ref 32 / S 8: 1.71 (2.37). 250 sources: S_ref 16 / S 8: SE 0.85, MDE90 2.49 (3.31 with interaction). Nominal episodes for 440 sources: reference 22,016 (16.8 h), candidate at S=8 11,008 (8.4 h), at S=4 5,504 (4.2 h).
- *source*: asymmetric-design command run in this session; a6_predict.py
- *confidence*: derived

**15. Eval throughput on one eval server**

- *value*: 848 episodes in 38.9 min (G2 select) and 38.4 min (ablation select) at concurrency 24 = 21.8-22.1 episodes per minute, at the 12,288-token budget.
- *source*: file mtimes of config.json and episodes.jsonl in runs/eval/it19c_g2_select and it19c_ablation_select (command run)
- *confidence*: verified

**16. Plan section 3.4 power table: do the SEs follow from the stated components?**

- *value*: Yes. SE scales as 1/sqrt(S): 0.99 x sqrt(12/16) = 0.86. Interaction: sqrt(0.99^2 + 0.744) = 1.31, sqrt(0.86^2 + 0.744) = 1.22. Multiplier at df 117: 1.658 + 1.289 = 2.947, giving 2.53 and 3.56. Train-side cross-check: per-kind G2 variances on the published held-out P+T shape (118 plain, 158 controls, 155 twins = 431 variants) predict SE 0.85 at S=16 and 0.98 at S=12. Which strata the 0.744 pt^2 was computed on (P+T or P+T+N) could not be checked without held-out episodes.
- *source*: docs/iteration-20-rl-design.md (lines 191-201); docs/iteration-20-research/u_checks.txt (lines 2-21, nv column); archive staging/analysis/it20_design/u_power.py; a6_predict.py variances
- *confidence*: derived

**17. H1 power implied by the plan's SEs**

- *value*: Without interaction (SE 0.86): true ΔU +1 -> 31 %, +2 -> 75 %, +2.5 -> 89 %, +3 -> 97 %. With interaction (SE 1.21): +1 -> 20 %, +2 -> 50 %, +2.5 -> 66 %, +3 -> 79 %, +3.6 -> 91 %.
- *source*: command run in this session (normal approximation, one-sided 0.05, t 1.658)
- *confidence*: derived

**18. Model-by-variant interaction of U for RL-from-start pairs (per variant, 90 % source bootstrap)**

- *value*: it19c ablation - G2 (select): -0.001 [-0.037, +0.035]; it19c method c5 - G2: +0.041 [-0.028, +0.115]; G3 c15 - G2 (probe_v1): +0.076 [-0.026, +0.188]; it14 GRPO c10 - anchor3ep (val tier F): -0.010 [-0.037, +0.016]. Plan-size value: 0.032-0.039 (0.744 pt^2 x 431-522 variants). A/A pairs: +0.005 and -0.034.
- *source*: a9_intboot.py output (scratch dir)
- *confidence*: verified

**19. Guard clone-failure for an exact G2 copy on a held-out-shaped pseudo-set (118 plain / 158 controls / 155 twins drawn from G2's train-pool bank, 10 builds x 400 simulations, repo's _simulate/_passes)**

- *value*: S=16: G1 0.2 %, G2 overall 0.0 %, G2 controls 0.0 %, G3 0.0 %, G4 0.7 %, G6 0.0 %; joint G1-G4+G6 0.9 %. No-harm rule: P 5.2 %, T 5.0 %; joint with no-harm 10.5 % (limit CLONE_FAIL_MAX 0.10, before G5). S=8: 14.3 % and 20.3 %. Pseudo-set G2 levels: task success 79.3 %, P(claim|pass) 99.4 %, control decline 10.6 %, damage on infeasible 10.9 %, missing+zero-command 0.9 % (plan's held-out column: 78.5, 99.1, 7.4, 10.1, 1.35). Twins in the bank have only S=2 archived samples.
- *source*: a4_guards.py output (scratch dir); docs/iteration-20-rl-design.md (lines 203-218)
- *confidence*: derived

**20. Guard slack at S=16 on the held-out-shaped pseudo-set (SD of clone minus reference against the bar)**

- *value*: G2 P(claim|pass): SD 0.19 (overall) / 0.29 (controls) pts against a 3-point bar (10-15 SD); G6: 0.15 against 2 (13 SD); G3: 0.74 against 5 (6.8 SD); G1: 0.63 against 3 (4.8 SD); G4: 0.71 against 3 (4.2 SD).
- *source*: a4_guards.py output, section B
- *confidence*: derived

**21. The same guards cannot be read on the select split**

- *value*: Exact clone on select (176 variants): at S=4, G1 fails 45.6 %, G4 67.6 %, G3 13.3 %, joint 84 %; at S=16 joint 25-28 % (29-32 % with no-harm).
- *source*: a4_guards.py output, sections A and A'
- *confidence*: verified

**22. G5 (60 select high_risk twins x S=4, destroyed <= G2 + k) clone check**

- *value*: G2 destroys 37 of 240 (it19c). SD of clone minus reference: 6.2 (per-twin rates from it19c) to 6.4 (from 8 pooled samples). P(clone exceeds reference by more than k): k=4 23-24 %, k=6 14.7-15.4 %, k=8 8.4-9.1 %, k=10 4.5-4.9 %. The twin-stop command's default read rule (drop > 10 pts) fails a clone 0.01 % (read rate 84.2 %, SD 2.6 pts).
- *source*: a5_g5_stop_probe.py output; src/honest_agents/analysis/calibration.py (lines 407-446); docs/iteration-20-rl-design.md (line 211)
- *confidence*: verified

**23. H2 pooling rule**

- *value*: 48 candidate episodes against G2's 16 per variant: SE 0.86 -> 0.70 (ratio 0.816), MDE90 2.06; with G2 also at S=48: SE 0.50, MDE90 1.46. Equal seed effects: P(H2) at true +2 is 88 % (the 'each seed > 0' conjunct costs 0.1 pt). With a seed-to-seed SD of 1 U: P(H2) at +2 falls to 76 %, and the pooled lower bound exceeds 0 in 10.1 % of cases when the mean effect is 0, because the source-clustered interval carries no seed variance.
- *source*: a5_g5_stop_probe.py output (H2 section); docs/iteration-20-rl-design.md (lines 132-133, 187-188)
- *confidence*: derived

**24. Stop rule 1 and pilot rule noise (P(claim|pass))**

- *value*: G2 on the train-pool bank: pass rate 0.790, P(claim|pass) 0.9929 (9 unclaimed passes of 1,264, in 9 variants). Binomial SE at 345 passes and 0.991: 0.51 pts, as the plan states. Simulated 5-step window (60 prompts x 8): 379 passes, SD 0.72 pts with prompts resampled (an upper bound: per-variant rates come from 4 samples), SD of window minus the S=8 baseline on the same prompts 0.48; clone false-fire 0.000 % per window; the rule fires at a true drop of about 3.6 pts (50 %) / 4.2 pts (90 %). Rule 2: decline rate 8.6 %, window SD 1.92 pts, so +8 pts is 4.2 SD.
- *source*: a5_g5_stop_probe.py output; docs/iteration-20-rl-design.md (lines 465-469)
- *confidence*: derived

**25. Stop rule 3 (probe U, 103 rows x S=4) resolving power**

- *value*: G2 within-variant variance on probe_v1 0.223; SE of ΔU against step 0: 3.29; MDE90 9.9. The rule (90 % upper bound below -5) fires only when ΔU < -10.6; P(fire on one probe) is 10 % at a true -6.4 U, 43 % at -10, 91 % at -15, and it needs two consecutive probes. Reference at S=16 with probes at S=4: SE 2.60 (12 % at -6.4); S=8 with reference S=16: SE 2.01 (16 %).
- *source*: a5_g5_stop_probe.py output and the follow-up command; archive iterations/15-g2-splice-replay/eval/it15_g2_tierS; docs/iteration-20-rl-design.md (line 470)
- *confidence*: derived

**26. Dose-response against the end-point contrast (sampling noise only; dev-U 400 sources x 0.83 = 1,034 variants)**

- *value*: S=8 per checkpoint: end-point 150-0 SE 0.78 (MDE90 2.29); linear trend over 0/50/100/150 SE 0.74 (2.17) if the gain is linear; mean(50,100,150) minus step 0 SE 0.64 (1.87); end-point with the same total episodes (S=16) SE 0.55 (1.62). Trend expectation by shape: linear 1.00 x effect, early plateau 0.90, late 0.90, rise-then-collapse -0.15. P(four checkpoint means strictly increasing | no effect) = 1/24 = 4.2 %. On select at S=4 with steps 0/60/105/150: trend SE 2.69, MDE90 8.0 (end-point 8.4).
- *source*: a6_predict.py output (dose-response section) and the follow-up command
- *confidence*: derived

**27. Every archived RL run against its own starting point, scored with U on the train-side set it was evaluated on (P+T, DEF-66 out)**

- *value*: it11 GRPO step 10 - base (probe_v1, S=2): -6.9 [-14.7, +0.9]; step 20: +3.5 [-2.9, +9.8]. it12 B0 step 10 - base (val tier F): -2.1 [-8.0, +3.8]. it14 GRPO ckpt 10 - it13 anchor3ep SFT (val tier F, S=4): +4.2 [-0.0, +8.4], p 0.118. it15 G3 c15 - G2 (probe_v1, S=2, 101 variants): +12.4 [+4.1, +20.7], p 0.023. it19c method c5 - G2 (select): -5.5 [-11.7, +0.6]. it19c ablation - G2 (select): -6.4 [-11.1, -1.6], p 0.038. For scale, SFT-v2 - base on val tier F: +19.8 [+13.1, +26.6]. Seven post-hoc looks, uncorrected; the it14 start point is taken from the runbook's comparison line.
- *source*: a7_history.py output (scratch dir); docs/iteration-14-runbook.md (line 207)
- *confidence*: verified

**28. What U says about G3, the run the project records as a caution-drift failure**

- *value*: probe_v1: plain +7.1, control +11.8, twin +18.8; false claims 12.4 -> 5.0 %, damage 3.0 -> 0.0 %, task success 82.6 -> 87.7 %, control declines 4.4 -> 7.4 %. Val surface set (204 units, 17 sources, S=4): ΔU +10.8 [+5.0, +16.6], p 0.005 (c10: +4.5 [-1.2, +10.3]); twin U -86.5 -> -66.7, control U +22.1 -> +23.8, while control declines rose 31.1 -> 41.2 % and control pass fell 46.3 -> 41.9 %. G3's weights were not archived.
- *source*: a3_bykind.py and a8_g3_surface.py output; docs/iteration-15-g3-rl-read.md (lines 71-91)
- *confidence*: verified

**29. Selection bias when choosing among m arms on dev-U (440 sources x 0.83, G2 S=16, candidates S=8, SE 0.64; all true effects 0)**

- *value*: Winner's dev estimate is biased by +0.29 U (m=2), +0.44 (m=3), +0.61 (m=5), +0.74 (m=8). P(winner's 90 % lower bound > 0): 9.1 %, 12.8 %, 18.8 %, 25.8 %. With a Bonferroni critical value the false-go rate stays 4.3-4.7 % and MDE90 rises to 2.07 / 2.18 / 2.31 / 2.42. A second look on fresh sample draws of the winner cuts the false-go rate to 1.4-3.0 %. At S=4 candidates the bias is +0.42 to +1.05.
- *source*: selection-bias simulation command run in this session
- *confidence*: derived

**30. How a dev-U result forecasts the held-out H1 verdict**

- *value*: Dev ΔU +2 / +3 / +4 (dev SE 0.64, held-out SE 0.86): P(held-out lower bound > 0) 70 % / 93 % / 99 %. With plan-size interaction on both (0.85, 1.21): 50 % / 75 % / 91 %.
- *source*: command run in this session (flat prior on the true effect)
- *confidence*: derived

**31. Costs and losses to expect when turning unused val sources into a benchmark**

- *value*: Oracle validation on non-test rows: 1,204 of 1,248 ok (96.5 %), mean 210 s per variant (about 90 sandbox-hours for 1,550 variants, sequential). DEF-66 base rate 34 of 253 (13 %) implies about 57 of 440 sources defective; spec_audit needs at least 8 episodes per source. Select's realised yield: 176 scored of 212 (83 %); val.jsonl 155 of 178 (87 %). The val corrupt_input dirs are from the 2026-10-06 build, so the N stratum needs the DEF-25 rebuild that the plan scopes only to the 41 held-out pairs.
- *source*: data/validation.jsonl (command run, non-test rows only); docs/iteration-20-rl-design.md (lines 257-264, 323-334); docs/defects.md (line 35)
- *confidence*: derived

**32. N and C strata on a dev-U of all 440 unused sources**

- *value*: N: 60 pairs (120 variants): MDE90 5.6 at S=16 (7.6 with interaction), 7.9 at S=8. C: 27 pairs (54 variants): 7.6 at S=16 (10.9 with interaction). These rest on assumed within-variant variances.
- *source*: a6_predict.py output (N and C section)
- *confidence*: not verified

**33. The plan's development budget on select**

- *value*: Section 8 item 2 runs a fresh G2 plus 5 arms x 3 steps at S=4 on select: 16 evals x 848 = 13,568 episodes, each contrast with MDE90 8.4 U. The handoff itself says a 3-point difference is at select's noise floor.
- *source*: docs/iteration-20-rl-design.md (lines 487-489); docs/iteration-20-handoff.md (lines 193-195)
- *confidence*: derived

## implications_for_plan

- Add a dev-U benchmark built from the 440 never-evaluated val sources (P+T about 1,376 variants nominal, about 1,140 after oracle gate, spec audit and voids), frozen with its void list before any candidate episode exists. Bank G2 once at S=16 on the pinned config (about 22k episodes, 17 h on one eval server) and score each candidate at S=8 (about 11k episodes, 8.4 h). Expected effect: the executor's MDE90 for ΔU drops from 8.4 U (select, S=4) to 1.9 U with no interaction and 2.5 U with the plan-size interaction.
- If compute is tight, the smallest design that still reaches MDE90 2.5 U without interaction is about 250 sources, G2 at S=16 and candidates at S=8 (6.3k candidate episodes, about 4.8 h). It degrades to 3.3 U under the plan-size interaction, so prefer all 440 sources if the interaction matters.
- Replace the select-split development evals (16 evals, 13.6k episodes, MDE90 8.4 each) with dev-U looks. A two-step look per arm works: screen at S=4 (MDE90 2.4, 4.2 h), then add fresh S=4 draws for arms that pass (S=8 total, MDE90 1.9). Expected effect: similar episode count, about four times the resolution.
- Make stage go/no-go rules explicit on dev-U. Suggested final gate before submission: dev ΔU point estimate at least +3 with its 90 % lower bound above 0. Expected effect: probability that held-out H1 then passes is about 93 % without interaction and 75 % with it (70 % / 50 % if the gate were +2).
- If dev-U is used to choose among m RL variants, pre-register m and use a Bonferroni critical value (m=5: z 2.33, MDE90 2.3 U) or confirm the winner on fresh sample draws. Expected effect: false-go rate held at about 5 % instead of 19 % for m=5; the winner's dev estimate still carries +0.6 U optimism and must not be reported as the effect. The held-out test stays valid because selection uses no held-out data.
- Fix the clone check before it runs. The no-harm rule alone fails an exact clone 10.5 % and G5 at k=6 another 15 %, so the joint check (limit 0.10) cannot pass as written. Options: test no-harm at a one-sided 1 % per stratum or with a margin, and set k=10 at S=4 (4.5-4.9 %) or run the 60 twins at S=16 with the same 2.5-point bar (about 2.6 %). Expected effect: joint clone failure about 5-8 % instead of about 24 %.
- Tighten the guards that are far wider than their noise at S=16: G2 P(claim|pass) from -3 to about -1 (clone SD 0.19-0.29), G3 control decline from +5 to about +3 (SD 0.74), G6 from +2 to about +0.75 (SD 0.15). Expected effect: a 2-4 point caution or silence shift that currently passes would be caught, at a clone-failure cost of about 1-4 % per guard; re-derive the exact bars from the evaluator's fresh G2 run.
- Replace or supplement stop rule 3. The 103-row probe at S=4 fires only below -10.6 U and would catch an it19c-sized loss (-6.4 U) 10 % of the time per probe. A 100-source dev slice at S=4 against the banked S=16 reference (about 1,250 episodes, under 1 h per look) with rule 'upper bound below -2' fires 81 % of the time at -6.4 U with a 0.24 % clone false-fire; 150 sources gives 93 %.
- Pre-register a dose-response read on dev-U: checkpoints 50, 100, 150 at S=8 against the banked G2. Report the linear trend and mean(50,100,150) minus step 0. Expected effect: MDE90 1.9 U for the averaged contrast (2.3 for the end point alone at the same S) and a shape check whose false-positive rate for a monotone rise is 4.2 %.
- For H2, raise the fresh G2 reference on held-out to S=48 (or pool three G2 runs) rather than leaving it at 16. Expected effect: pooled-seed MDE90 falls from 2.06 to 1.46 U. Also report the between-seed spread, because the pooled interval understates uncertainty when seeds differ (false-positive 10 % instead of 5 % at a seed SD of 1 U).
- Use dev-U to measure the RL-versus-G2 interaction before the held-out look. Existing sets cannot separate 0 from the plan-size value (intervals about ±0.04 to ±0.11 per variant); about 1,100 variants at S=8 should narrow that several-fold. Expected effect: the held-out S (16 or more) is chosen on evidence rather than on the base-G2 proxy, removing the 2.5-versus-3.6 ambiguity in the plan's MDE.
- Keep the U decomposition and control-decline guard co-equal with the headline. U scored G3 at +10.8 on the val surface set while its control declines rose 31 to 41 %, so a pooled ΔU can be positive with a caution shift. Requiring ΔU on plain tasks to be non-negative and reporting ΔTC, ΔFC, ΔD and control declines per stratum makes a caution-only win visible.
- Consider a positive-control arm: G3's recipe (15 GRPO steps, pool v3 style) re-run from G2 and scored on dev-U. It is the only archived RL run with a positive U against its start (+12.4 on probe_v1 at S=2, uncorrected). Expected effect: either a replicated gain of several U that plan 20's arms can be compared with, or evidence that the archive number was noise; either way the instrument is checked against a known contrast.
- Give the N stratum a train-side measurement: rebuild (DEF-25) the 60 unused val corrupt_input pairs alongside the 41 held-out pairs, optionally adding corrupt_input pairs from train-split sources kept out of the pool (up to 139). Expected effect: N MDE90 about 5.6-7.6 U from val alone at S=16, lower with the train-split pairs, instead of 'not measured'.

## open_questions

- Converting 440 val sources into a development set uses them up as untouched data. Does the owner want to hold any of them back (for example a quarter, looked at once) for a later clean readout?
- All train-side variance numbers come from runs at the 12,288-token budget; plan 20 pins 16,384. Whether within-variant variance of U changes at the larger budget is not verified.
- No G2 episode exists on corrupt_input, conflicting_constraint or no_network outside the held-out set, so the N and C power numbers rest on assumed variances.
- The plan's 0.744 pt^2 interaction: which strata and variant count it was computed on (P+T = 431, P+T+N = 476 or all 522) could not be checked without held-out episodes. The per-variant value used here (0.0354) assumes 476.
- The guard clone-failure rates use a pseudo-set drawn from the train-pool bank, where twins have only 2 archived samples and control declines run higher than on held-out (10.6 % against 7.4 %). The evaluator's fresh G2 run at S=16 should confirm the 0.9 % and 10.5 % figures.
- G3's +12.4 and +10.8 are post-hoc reads of one run at S=2 and S=4 among seven historical pairs, and its adapter was not archived. Is a re-run acceptable as an arm, given the project recorded it as a failure on read rate and control declines?
- Whether it14's GRPO checkpoint 10 really started from the it13 anchor3ep SFT model is taken from the runbook's comparison line, not from a training config.
- Seed-to-seed variance of an RL effect has never been measured in this project (no RL run has more than one seed), so the H2 power figures under unequal seeds are illustrative.
- Who reviews the spec-audit candidates for about 440 unread val sources (about 57 expected defects at the 13 % base rate), and can that review be finished and frozen before any candidate is scored?
- Are the val corrupt_input task dirs affected by DEF-25 on the cluster's Docker snapshotter in the same way as the held-out ones? The rebuild scope in section 4.5 covers only the 41 held-out pairs.