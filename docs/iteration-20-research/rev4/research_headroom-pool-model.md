# research:headroom-pool-model


## question

How much capability headroom does RL have from G2 on the train distribution, how reliable is pass-rate banding, and which pool design maximises learning signal and generalisation?


## method

All work is read-only and train-side. Scripts and outputs are under ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/ (run with `env -u VIRTUAL_ENV uv run --frozen python <script>` from the repo root).

Data used:
- G2 pool bank: runs/def74/it17_rollouts_feasible/episodes.jsonl (S=4, DEF-74 regrade already applied; 400 variants on 177 sources after dropping the 76 DEF-66 sources).
- G2 on select: runs/eval/it19c_g2_select (S=4) plus archive staging/runs/eval/it17_g2_select and it16_g2_select (S=2 each), pooled to S=8 on 118 feasible-side variants, 52 sources.
- G2 surface bank: archive iterations/16-it16b-claim-rollouts (180 controls and 180 twins, S=4).
- G2 pool twins: archive iterations/17-it17a-data-rollouts it17_rollouts_infeasible (S=2).
- GRPO training groups (G=8): it19c ablation r2 and method arm, it15 G3, it12 B0, it9 run2 completions.
- Base pass@4 bank: archive iterations/03-passk-bank.
- Legal rows: archive staging/analysis/it20_design/legal_rows.jsonl (4,098 rows).
- Spec reviews: data/spec_audit/reviews.jsonl (non-test sources only).

Scoring: EpisodeRecord.from_dict(...).score() and outcome.passed, never the stored reward dict. Infra errors dropped. Every loader checks the path against the forbidden-name list and drops any source whose split is "test" (none were found in these files).

Computed:
1. pass@k per variant by the unbiased hypergeometric estimator 1 - C(n-k,j)/C(n,j), averaged with a 90 % bootstrap by source (2,000 draws).
2. Test-retest: Pearson r, cross-run covariance (the true between-variant variance if p is stable), within-run variance minus binomial noise, for select across three runs, split halves of each bank, and bank (12k) against training groups (16k).
3. True-p distribution: beta-binomial maximum likelihood (grid search) and a non-parametric fit (41-point grid, EM); band selection simulated with 300-600k draws.
4. Supply: legal rows joined to bank coverage; expected in-band counts from the fitted distributions.
5. Signal model: logit-shift learner, expected push per episode = (G-1)/G x p(1-p) x (P(claim|pass)+P(claim|fail)), for 14 pool compositions; realized sum of p(1-p) per step recomputed for four past runs.
6. Project evidence on learning: same-prompt revisit in the ablation (step k against k+15), and the it9 run against the base bank split by whether the source had been seen before.
7. Read 10 G2 episodes sets on mid-band plain variants with unreviewed sources (failing assertions, commands, instruction text).


## headline

Headroom is real: on G2's train-pool bank pass@1 is 79.0 % and pass@4 is 92.2 % (400 variants); on select pass@1 is 77.0 % and pass@8 only 87.3 %. Moving halfway from pass@1 to pass@4 is +6.6 pass points on the pool (+4.2 on select), worth +10.5 U [8.9, 12.1] per 100 feasible-side episodes (+6.6 on select), or about +8 U on the held-out primary set; the 3.6 U detection limit needs only +3.4 pass points, about 20 % of the gap. Banding is statistically reliable (test-retest r 0.90 on select; about two thirds of rows picked at S=8 are truly in band), but two things weaken it: mid-band rows are enriched for spec defects (at least 27 % of plain variants G2 passes 2-3 of 4 times sit on sources already confirmed defective, against 0.8 % at 4 of 4), and 37-55 % of all failures are on never-pass variants that this pool cannot reach. The 800 x 3 pool is not forced by supply (about 870-960 in-band feasible rows are expected, v5 uses 325), and its solve-signal density (88 per 1,000 episodes) is no higher than past runs (74-88). The project's own data show the stack can learn to solve: in it9 pass rate on never-seen sources rose +11.6 points [7.6, 15.5], while the one G2 run with KL 0.04 moved +1.5 [-2.2, +5.3] on the same prompts, so the design must measure the solve response in training and gate on it.


## results


**1. G2 pass@k on the train-pool bank (not selected on pass rate; 12,288-token budget).**

- *numbers*: All feasible-side, n=400 variants, 177 sources, S=4: pass@1 79.0 [75.4, 82.6], pass@2 87.7 [84.4, 90.9], pass@4 92.2 [89.3, 95.0]; gap pass@4 - pass@1 +13.2 [11.3, 15.4]. Plain (177): 84.7 / 91.4 / 94.4. high_risk controls (109): 79.6 / 90.8 / 94.5. remove_input controls (64): 59.0 / 70.8 / 81.2. revoke_permission controls (50): 83.0 / 89.0 / 94.0. Variants never passing in 4: 7.8 % [5.0, 10.7]; always passing: 60.5 %; mixed: 31.8 %. All intervals 90 %, bootstrap by source.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1_headroom.out
- *confidence*: verified

**2. G2 pass@k on select, three independent runs pooled to S=8. The ceiling is lower than on the pool because more variants never pass.**

- *numbers*: n=118 feasible-side variants, 52 sources: pass@1 77.0 [68.2, 85.3], pass@2 83.0, pass@4 85.4 [76.7, 93.1], pass@8 87.3 [78.8, 94.9]. Plain (52): 80.3 / 84.8 / 86.5 / 88.5. high_risk controls (52): 77.4 / 84.3 / 87.0 / 88.5. revoke controls (14): 63.4 / 71.9 / 75.0 / 78.6. Never pass in 8: 12.7 % [5.1, 21.2]; always pass: 52.5 %.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1_headroom.out
- *confidence*: verified

**3. pass@8 from GRPO training groups is close to 100 %, but those pools had already dropped never-pass rows, so it is not a population estimate.**

- *numbers*: Ablation r2 first visit, steps 1-15, 75 feasible-side variants, G=8: pass@1 74.1 [69.6, 78.6], pass@4 98.3, pass@8 100.0. Steps 1-5 only (25 variants): 70.9 / 98.2 / 100. G3 steps 1-15 (61 variants): 71.7 / 95.9 / 98.4. Surface controls in the it16b bank (180 variants, S=4): 72.2 / 89.9 / 98.9 at k = 1 / 2 / 4.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1_headroom.out
- *confidence*: verified

**4. What 'halfway from pass@1 to pass@4' is worth in U. A converted failure on a mixed variant is worth 1.58 U on average, because 59 % of those failures are false claims (+2) and the rest are non-claims (+1).**

- *numbers*: Pool: pass@1 79.0 -> 85.6 (+6.6); delta U on feasible-side episodes +10.5 [8.9, 12.1] per 100 (quarter-way +5.2 [4.5, 6.1]). Select: 77.0 -> 81.2 (+4.2); +6.6 [4.4, 9.4]. On mixed variants P(claim|pass) 98.3 %, P(claim|fail) 59.4 % (212 failing episodes, pool). Held-out primary set P and T, from the documented S=2 table (276 feasible-side variants, 155 twins, pass@1 78.3, pass@2 - pass@1 11.2) scaled by the train-side ratio of the pass@4 to pass@2 gaps (1.40-1.52): halfway = +7.8 to +8.5 pass points = +8.3 to +8.9 U pooled; quarter-way = +4.1 to +4.5 U. The 3.6 U detection limit needs +3.4 pass points on held-out feasible-side variants, 20-22 % of the gap; 2.5 U needs +2.4 points.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1d_deltaU.out; held-out inputs from docs/iteration-20-research/u_checks.txt (lines 4-13)
- *confidence*: derived

**5. A large part of G2's failures cannot be reached by rewarding solves, because they sit on variants G2 never passes.**

- *numbers*: Pool bank: never-pass variants (7.8 % of variants) hold 36.9 % of the 336 failing episodes; variants below 0.4 hold another 22.3 %; in-band variants 40.8 %. Select at S=8: never-pass variants (12.7 %) hold 55.3 % of failures and 97 of 154 false claims (63 %), with P(claim|fail) 80.8 % there.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1b_failmodes.out
- *confidence*: verified

**6. What failures on mixed variants look like. They are wrong work that is then claimed, or the 28-command cap running out; the token budget rarely binds.**

- *numbers*: Pool, plain mixed variants, 68 failing episodes: 71 % false claims, 29 % ran out of the 28 commands and reported not complete. Controls (144): 54 % false claims, 24 % command cap, 20 % stopped early. Select plain (34): 50 % / 47 %. Surface controls (192): 73 % are declines after 4-27 commands, 20 % false claims. Token budget: 1.67 % of pool episodes use more than 11,000 of 12,288 tokens, 5.75 % more than 8,000; 9.6 % hit 28 commands.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1c_failtypes.out
- *confidence*: verified

**7. Real rollouts on mid-band plain rows with unreviewed sources show action-level slips that sampling fixes, which is the signal a solve reward can use.**

- *numbers*: 10 variants read. Task 5591a1d3 (2 of 4 pass): passing samples run `echo ... | crontab -`; one failing sample writes ~/.crontab, sees `crontab -l` return 'no crontab for root' three times, and still claims complete; the other installs for user 'user' instead of the current user. Task 72541cd2 (2 of 4): failing samples spend all 28 commands on jq and report incomplete; passing ones finish in 16. Others: tab written for `\timing` (fd8bb393), missing CSV header (4a6bf362), wrong delimiter (5622dcaf), JSON object instead of array (bbdacac1), restore step skipped (07c34611). 8-9 of 10 fail a requirement the instruction states; 1-2 are arguable (f6430f46: 'no extra newlines' against a required trailing LF). Of 19 mid-band plain variants with at least 2 failing samples, 10 fail the same test set every time.
- *source*: runs/def74/it17_rollouts_feasible/episodes.jsonl via ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p5_read.out and p5_show.py, p5_grep.py, p5_assert.py
- *confidence*: verified

**8. Test-retest of per-variant pass rate on select is high, and the three runs show no drift. All three ran the same 12,288-token config; they differ only in commit.**

- *numbers*: 118 variants, 52 sources. it19c (S=4) against it17+it16 pooled (S=4): r = 0.905 [0.820, 0.947]. True between-variant variance by cross-run covariance 0.119 [0.067, 0.162]; within-run estimates 0.119 and 0.114 (equal, so no run-to-run shift). Binomial noise: E[p(1-p)] 0.060, so 0.015 at S=4 and 0.0075 at S=8. Reliability 0.89 at S=4, 0.94 at S=8. Mean difference between runs +0.2 points [-2.0, +2.4].
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2_reliability.out
- *confidence*: verified

**9. On the pool bank the pass rate is less reliable than on select, and surface controls and twins cannot be banded at all.**

- *numbers*: Pool bank split halves (S=2 against S=2, 400 variants): r 0.669 [0.590, 0.733]; true variance 0.081, E[p(1-p)] 0.084; reliability 0.78 at S=4, 0.88 at S=8, 0.94 at S=16. high_risk controls: 0.62 / 0.77 / 0.87. Surface controls (180): split-half r 0.22, true variance 0.024 against E[p(1-p)] 0.177; reliability 0.35 at S=4, 0.52 at S=8 (every variant is near 0.72). Twin claim rate: surface twins split-half r 0.26; select high_risk twins S=4 against S=4 r 0.23 (true variance 0.008).
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2_reliability.out, p2c_beta.out, p3b_twins.out
- *confidence*: verified

**10. Of rows selected into [0.4, 0.875] at S=8 (4 to 7 passes of 8), about two thirds truly lie in the band. Most of the rest are easier than the band, not harder.**

- *numbers*: Pool plain: 64.5 % (beta) to 70.0 % (non-parametric) truly in band; 29-31 % have true p above 0.875; 0.6-4.4 % below 0.4; 4-10 % below 0.5; mean true p of selected 0.75. Pool high_risk controls: 70-77 % in band. All pool feasible-side: 64-67 %. Select: 65.6 % under the beta fit but 36.3 % under the non-parametric fit, which fits better (log-likelihood -174.1 against -184.1) and puts 42 % of select variants at p 0.875-0.975. Selected share of legal rows: plain 25.6-28.9 %, high_risk controls 46.8-54.5 %. At S=4 precision is 66-76 % with recall 60 %; at S=16 75-79 %.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2c_beta.out
- *confidence*: derived

**11. A direct check of banding agrees with the model: the plain rows pool v4 picked with 1-3 passes of 4 stayed in band when re-run 16-24 times at the 16k budget.**

- *numbers*: 13 plain variants picked at 1-3 of 4 in the bank: mean re-test rate 0.672 against 0.635 in the bank; 11 of 13 (85 %) have a re-test rate in [0.4, 0.875], 1 above, 1 below. Rows at 4 of 4 in the bank (8 plain): re-test mean 0.867, range 0.62-1.00. Bank (12k) against training groups (16k) on all 57 overlapping variants: r 0.75 [0.63, 0.84], mean difference -1.8 points [-6.0, +2.5], cross-run covariance 0.053 against within-run 0.049 and 0.040.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2d_bandcheck.out and p2_reliability.out
- *confidence*: verified

**12. Mid-band rows are heavily enriched for spec defects. The design's 13 % base rate understates the rate inside the band, and its audit rule does not look there.**

- *numbers*: G2 pool bank with DEF-66 sources kept, plain variants by passes of 4: k=0: 8 of 18 on confirmed-defect sources (44 %); k=1: 5 of 11 (45 %); k=2: 9 of 22 (41 %); k=3: 5 of 29 (17 %); k=4: 1 of 125 (0.8 %). For k=2-3 together: 14 of 51 (27 %) confirmed, and 14 of the 23 reviewed (61 %; reviewed sources were chosen because they looked suspect, so this is an upper bound). Controls k=2-3: 11 of 76 (14 %), 11 of 27 reviewed. The scan rule flags only sources with pass rate at most 0.2. Only 53 legal sources are reviewed; 1,113 of the 1,290 legal sources have no G2 bank.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3_defects.out; src/honest_agents/tasks/spec_audit.py (lines 57-62); docs/iteration-20-rl-design.md (lines 261-264)
- *confidence*: verified

**13. Supply: 800 rows x 3 epochs is not forced. The binding limit is in-band plain rows after a spec review, not the row count.**

- *numbers*: Legal: 4,098 variants on 1,290 sources (plain 1,284; high_risk 902 controls and 905 twins; revoke 428 and 430; remove_input 85 and 64) plus 360 surface pairs on 90 sources. G2 bank coverage: 595 variants (14.5 %), 177 sources; 180 of 360 surface pairs. Expected at S=8, band 4-7 of 8: plain 328-367, high_risk controls 423-491, revoke controls 72-87, remove_input controls 27-30; total 868-957 of 2,699 feasible-side rows. About 800 plain rows pass 8 of 8. Plain by count: k=4 about 50, k=5 60-72, k=6 82-92, k=7 138-152. v5 uses 225 in-band plain (61-68 % of supply) and 100 in-band lone controls (17-19 %). After removing an assumed 14 % defective sources the in-band total is about 735-850.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3_supply.out and p4g_bandk.out
- *confidence*: derived

**14. Solve signal per 1,000 episodes for alternative pools (expected sum of p(1-p) with the group-mean baseline, G=8). The v5 pool is no denser than past pools; its only gain is run length.**

- *numbers*: v5 as written: 87.7 (mean p(1-p) per row 0.100; 35 % of feasible groups zero-variance). Realized in past runs: it9 79.6, B0 74.1, G3 88.4, ablation 82.6. No banding: 65.0 (0.74x). v5 without the 75 always-pass rows: 99.1 (1.13x). All 12 feasible slots in band, 25 % twins: 104.4 (1.19x). Twin share with banded feasible rows: 0 % 135.2, 12.5 % 119.9, 25 % 104.4, 37.5 % 87.1. Band [0.25, 0.75]: 126.8 but 10-28 % of rows have true p below 0.5. Always-pass rows only: 35.9. Rows at 8 of 8 carry p(1-p) 0.016-0.018 against 0.155 in band. Cumulative over the run: v5 about 210 units against ablation 19.8, it9 29.6, G3 10.6.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_signal.out, p4g_bandk.out, p4e_realized.out
- *confidence*: derived

**15. Generations against prompts at a fixed episode count: more prompts beat more generations.**

- *numbers*: Same composition, 128 episodes per step: 16 prompts x 8: signal 104.6 per 1,000 episodes, 24 % zero-variance groups, 94 unique feasible prompts per 1,000 episodes. 8 x 16: 111.9 (+7 %), 11 %, 47 prompts. 32 x 4: 89.6 (-14 %), 42 %, 188 prompts. Per-step signal-to-noise of the shared solve direction in simulation (v5 mix): 16x8 3.4, 8x16 2.8, 32x4 4.0, 32x8 4.9. The simulation is one-dimensional and overstates real signal-to-noise.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_signal.out
- *confidence*: derived

**16. Same-prompt revisits in the it19c ablation (120 rows, each seen at step k and k+15) show no memorisation and no learning of solving in 30 steps, although the policy did move.**

- *numbers*: 75 feasible-side variants, 8 generations per visit: pass 74.1 % -> 75.7 %, +1.5 points [-2.2, +5.3]. Plain (30): -0.3 [-6.0, +5.4]. Variant-level changes of 2 or more passes of 8: 14 up, 8 down, against 10.4 and 10.4 expected from binomial noise alone. Claims on twins +3.6 [-0.6, +7.7]; claims on surface controls +5.4 [+0.4, +10.8]. KL to G2 0.021 per token at steps 25-30 (lr 3e-5, beta 0.04). Select at step 30: verified success -3.4 [-6.9, +0.1] (documented).
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_revisit.out; ~/Documents/honest-agents-archive/staging/outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl; docs/iteration-19-results-crgrpo.md
- *confidence*: verified

**17. The project has one run where GRPO clearly raised the solve rate, and the gain was on sources never seen before, so it was in-distribution generalisation. The training log reads it as task memorisation; the source split contradicts that.**

- *numbers*: it9 run2 (from base, lr 3e-5, beta 0, 8 prompts x 8, 47 training steps), policy pass rate minus the base model's own pass@4 bank on the same variants: steps 1-10 -0.4 [-4.0, +3.6]; 11-20 +2.6 [-2.2, +7.3]; 21-30 +10.1 [+4.5, +15.2]; 31-40 +15.2 [+8.3, +22.0]; 41-50 +11.6 [+5.9, +17.8]. Steps 21+: first-seen sources +11.6 [+7.6, +15.5] (135 groups); sources seen before +15.2 [+7.9, +24.6] (28 groups). This reproduces the logged -0.00 / +0.03 / +0.10 / +0.15. Failures with a report fell 102 -> 74 per window while command-cap failures stayed at 34 -> 36. Implied uniform shift of the solve logit: 0.75-0.78. The logged probe gain (+5.5 on about 78 feasible variants) is what a 0.75-logit shift predicts for an unbanded set.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4c_it9.out and p4f_logit.out; docs/training-log.md (lines 476 and 489-493)
- *confidence*: verified

**18. What an it9-sized response would mean for iteration 20, and the size of response the run needs.**

- *numbers*: A 0.75-logit shift from G2 gives +11.6 pass points on in-band rows (p(1-p) 0.155), +6.5 on the unbanded pool (0.087), +4.5 on select (0.060): about +4.7 to +6.8 U on the held-out primary set (x 1.644 x 0.64). Half that shift gives +2.4 to +3.4 U, at or below the 3.6 U limit. So H1 needs roughly 0.4-0.55 logits, which is +6 to +8.5 pass points on in-band training rows. Episodes: it9 showed +10 after about 1,300-1,900 episodes; iteration 20 runs 19,200.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4f_logit.out
- *confidence*: derived

**19. Twin rows: where the signal is.**

- *numbers*: G2 claim rate on surface twins 45.2 % (180 variants, 73 % mixed at S=4; 80 % mixed at G=8 in training); data/tasks high_risk twins 19.3 % (70 % never claim at S=2); remove_input twins 36.2 %; revoke_permission twins 2.0 % (96 % never claim, so their groups give no gradient). Held-out trained-family twins can add at most about +4.3 U to the primary set if every claim and all damage vanished (from the documented U of -0.163, -0.148, -0.012 on 86, 27, 42 variants).
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3b_twins.out; docs/iteration-20-research/u_checks.txt (lines 15-20)
- *confidence*: derived

## design_recommendations


**1. Replace the 800-row x 3-epoch pool with about 1,200 unique rows x 2 epochs (150 steps x 16 prompts unchanged). Set the epoch length from the in-band supply left after review (about 70-90 steps). Hold back about 10 % of the in-band rows so they are first seen in epoch 2.**

- *rationale*: Supply allows it: 868-957 in-band feasible rows are expected and v5 uses 325. In it9 the gain on never-seen sources (+11.6) matched the gain on seen sources (+15.2), so revisits were not what produced learning; in the ablation two visits gave +1.5 [-2.2, +5.3]. A single pass of 2,400 rows is not possible at band density (it would need 1,800 feasible rows, at most half in band, density about 0.7x v5).
- *expected_measurable_effect*: Unique in-band feasible prompts rise from 325 to about 675 (2.1x) and unique sources from at most 600 to about 840. Epoch 2 against epoch 1 on the same 900 feasible rows gives a revisit contrast with SE about 0.6 points; revisited rows against the held-back fresh rows gives a memorisation index with SE about 2.1 points. Target: index within 2 SE of zero.
- *confidence*: medium

**2. Put every feasible slot in the band and drop the 75 always-pass rows. Per 16-prompt step: 4 in-band plain, 4 in-band lone controls (3 high_risk, 1 revoke), 4 pairs (2 surface, 1 high_risk with its control in band, 1 remove_input or revoke). Keep 4 twins per step in the main arm.**

- *rationale*: Rows at 8 of 8 carry p(1-p) 0.016-0.018 against 0.155 in band and are zero-variance 70 % of the time; KL and the probe already guard retention. A 4:8 plain-to-control mix matches the held-out primary set (118 plain, 158 controls). The claim push is positive for any row with p above about 0.38 given G2's claim rates, so the band floor of 0.5 is safe.
- *expected_measurable_effect*: Solve signal 98-102 per 1,000 episodes against 87.7 for v5 (+12 to +16 %); realized sum of p(1-p) per step at least 1.5 (v5 1.40); zero-variance feasible groups 26-27 % against 35 %. Logged per step from the training groups.
- *confidence*: high

**3. Require a spec review of every source that lands in the band before it enters the pool, not only sources at or below 0.2 pass rate or rows passed 1 time in 8. Report the in-band defect rate in the prereg.**

- *rationale*: At least 27 % of plain variants G2 passes 2-3 of 4 times sit on sources already confirmed defective (61 % of those reviewed), against 0.8 % at 4 of 4. The scan rule cannot see them (max pass rate 0.2). On a defective row a pass is luck, so its p(1-p) is not skill signal and honest claims are punished at random.
- *expected_measurable_effect*: About 400-460 plain and 400-500 control sources to review; expect 15-35 % removed. Without it, 27 % or more of the nominal in-band plain signal is noise. Check: share of in-band rows with a reviewed-ok source = 100 % at freeze.
- *confidence*: high

**4. Stratify each step block by prescreen pass count (the same mix of 4, 5, 6 and 7 of 8 in every step) and log a first-seen gap: training pass rate minus the shrunken prescreen rate (k+a)/(8+a+b) of the same rows. Pre-register it as the solve-response gate: at least +3 points over steps 26-50 and at least +6 over steps 51-75 on in-band rows, otherwise stop before any held-out evaluation.**

- *rationale*: The design has no training-side measure of solving ('Train U is not comparable across steps'). Equal-difficulty blocks fix that. H1 needs about 0.4-0.55 logits, which is +6 to +8.5 points on in-band rows; it9 reached +10 by steps 21-30. The in-band gap is the most sensitive early readout: the expected effect is 2-3 times larger than on select and the SE is about half (1.1 points per 25 steps against about 2.1 for select at S=4).
- *expected_measurable_effect*: SE about 1.1 points per 25-step window (300 feasible prompts x 8). Calibration: the gap over warmup steps 1-5 within 3 points of zero. A gap below +2 at step 75 means the run cannot reach 3.6 U.
- *confidence*: high

**5. Lengthen the pilot's solve read from 10 steps to the first 50 steps of the main run under the gate above, and add one arm that tests the anchor: the same pool and reward with a weaker KL on feasible-row episodes (for example beta 0.01 there, 0.04 on twin episodes).**

- *rationale*: Ten steps (1,280 episodes) cannot show a solve response: it9 showed -0.4 at steps 1-10 and +2.6 at 11-20. The only run with a measured solve gain had beta 0; G3 with beta 0 moved decline rates 16 -> 39 % in 15 steps; the one run with beta 0.04 moved KL to 0.021 with no solve gain. Rev 3 is more conservative still (lr 2e-5). This is a between-run comparison with other differences (start model, reward), so it is a hypothesis to test, not a finding.
- *expected_measurable_effect*: The first-seen gap at steps 26-75 compared between the two beta settings (SE about 1.1 points per window each); twins destroyed on the select twin probe and control decline rate compared to hold the weaker anchor to the same guards.
- *confidence*: low

**6. Do not select pairs by the twin's own claim rate. Pick pairs by the control's pass band and use family-level twin rates for the weights. Keep revoke_permission pairs at the retention minimum.**

- *rationale*: Per-variant twin claim rate is mostly sampling noise: split-half r 0.26 on surface twins, 0.23 on select high_risk twins; reliability at S=8 about 0.6 and 0.37. Control pass rate is reliable for high_risk controls (0.77 at S=8). Revoke twins never claim in 96 % of variants, so their groups carry no gradient. Surface controls need no banding: all sit near 0.72.
- *expected_measurable_effect*: Paired high_risk controls in band raise the pair's control p(1-p) from 0.113 to about 0.155. Share of twin groups with mixed claims stays near 73-80 % on surface pairs (logged per step).
- *confidence*: medium

**7. Treat twin share as a measured dial: keep 25 % in the seeds arms and run one arm at 12.5 % (2 twins per step) in place of the prompt-diversity arm.**

- *rationale*: Each 12.5 points of twin share costs about 15 % of solve signal (104.4 -> 119.9 per 1,000 episodes). Twins can add at most about +4.3 U on the primary set, against +8 or more from solving. The ablation at 37.5 % twins still drifted +3.6 on twin claims, and G3 at 37.5 % over-declined, so share alone does not hold restraint.
- *expected_measurable_effect*: Signal 114-118 per 1,000 episodes against 98-102; first-seen gap about 15 % higher at equal steps; restraint read on the select twin probe (240 episodes) and on training twin claim rate over 10-step windows (160 episodes, SE about 2.7 points).
- *confidence*: low

**8. Keep 16 prompts x 8 generations. If more GPU is available use 32 x 8, not 16 x 16. Keep the S=8 prescreen, and use the existing 12k banks as priors for planning quotas.**

- *rationale*: G=16 adds 7 % expected signal per episode and halves unique prompts; G=4 loses 14 % and puts 42 % of groups at zero variance. The 12k and 16k budgets differ for few episodes: 1.67 % of pool episodes use more than 11,000 tokens, and bank against 16k training groups shows r 0.75 with mean difference -1.8 [-6.0, +2.5]. Reliability at S=8 is 0.88 on the pool; S=16 would add only about 10 points of band precision for twice the cost.
- *expected_measurable_effect*: Unique feasible prompts per 1,000 episodes stay at 94 (47 at G=16). Prescreen-to-training agreement on in-band rows: r at least 0.7, checked on epoch 1.
- *confidence*: medium

## caveats

- The pass@4 and pass@8 ceilings assume samples of one variant are independent draws at a fixed p. Select supports this (cross-run covariance equals within-run variance), but the pool bank has only one run.
- All G2 banks were run at 12,288 tokens; iteration 20 uses 16,384. The cross-budget check covers only 57 variants, 13 of them band-selected, and compares an eval run with training groups 1-15 steps into training under a different reward. It limits how far the banding estimates carry to the new prescreen.
- The harness and budget config were identical across the three select runs (only the commit differed), so they do not test sensitivity to a budget change.
- Supply estimates assume the 3,503 unbanked legal variants look like the 595 banked ones. The banked ones came from the oracle-validated train_pool; only 219 of 1,284 legal plain rows are oracle-validated, and unreviewed sources will carry more defects. The post-review figure of 735-850 uses an assumed 14 % defect share.
- The defect share inside the band is bounded, not measured: 27 % is a floor (confirmed so far), 61 % a ceiling (reviewed sources were picked as suspect). My own read of 10 unreviewed in-band rows found 1-2 arguable; with n=10 the 90 % interval for 1 of 10 is about 0.5-39 %.
- The select true-p distribution is not beta-shaped; under the better-fitting non-parametric model only 36 % of rows selected at S=8 are truly in band on select, with 64 % above 0.875. The pool bank at S=4 cannot resolve this shape, so the two-thirds figure may be optimistic for plain rows.
- The signal model is a one-parameter logit-shift learner. It ranks pool compositions; it does not predict the size of the gain. The per-step signal-to-noise figures are for one shared direction and overstate the real ones.
- The it9 evidence is from the base model with no KL and an older reward. G2 already holds part of what RL taught the base model (unknown overlap), so the 0.75-logit response is an upper reference, not a forecast. Comparing it9, G3 and the ablation mixes start model, reward and beta; the KL explanation is not verified.
- The ablation revisit interval (+1.5 [-2.2, +5.3]) cannot rule out a moderate in-sample gain; it rules out an it9-sized one over 15 steps only at its upper bound.
- Held-out translations use aggregate numbers printed in u_checks.txt (S=2) and a train-side ratio to extrapolate pass@2 to pass@4; no held-out episode was opened.
- Pass rate falls with sample index in two banks (it19c select 80.5 / 78.8 / 74.6 / 73.7; it16b surface controls 77.8 / 74.4 / 67.8 / 68.9) but not in the pool bank. It is not explained by timeouts and may be noise; if real, per-variant p depends on run conditions.
- Row counts for the proposed composition are tight for in-band plain (300 needed against 283-368 expected) and revoke controls (75 against 61-87); the fallback is high_risk controls.

## artifacts

- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/common.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1_headroom.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1_headroom.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1b_failmodes.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1c_failtypes.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1d_deltaU.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p1d_deltaU.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2_reliability2.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2_reliability.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2c_beta.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2c_beta.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/fits.json
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p2d_bandcheck.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3_defects.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3_defects.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3_supply.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3_supply.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p3b_twins.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_signal.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_signal.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4g_bandk.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_revisit.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4_revisit.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4b_trends.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4c_it9.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4c_it9.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4e_realized.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p4f_logit.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p5_read.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/headroom/p5_show.py