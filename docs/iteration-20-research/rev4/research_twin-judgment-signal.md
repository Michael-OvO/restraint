# research:twin-judgment-signal


## question

Under the weighted reward U, does the policy gradient on twin and control groups reinforce reading the evidence (judgment) or only a caution level, and is G2's existing restraint safe under it?


## method

All work is read-only on train-side data; no held-out path was opened (held-out numbers are quoted only from docs/iteration-20-research/u_checks.txt). Scripts and outputs are under ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/ and were run from the repo root with `env -u VIRTUAL_ENV uv run --frozen python <script>`.

Data (every source checked against data/split_seeds.json: all train or val, none test; the 76 DEF-66 sources from spec_audit.defective_sources() dropped):
- G2 on select: runs/eval/it19c_g2_select (S=4) plus archive staging/runs/eval/it17_g2_select and it16_g2_select (S=2 each). Pooled they give real groups of 8 samples per variant (52 sources).
- G2 on the train pool: archive staging/runs/eval/it17_rollouts_infeasible (twins, S=2) and runs/def74/it17_rollouts_feasible (controls and plain, S=4).
- G2 on the 400 train-pool surface tasks: archive staging/runs/eval/it16_claim_rollouts (180 pairs after exclusions, S=4, 85 sources).
- G2 and G3 c15 on the val surface set: archive iterations/15-g2-splice-replay/eval/it15_g2_surface4 and iterations/15-g3-grpo/eval/it15_g3_c15_surface4.
- The it19c ablation: runs/eval/it19c_ablation_select, runs/eval/it19c_ablation_c15_twins; K2a twins from archive staging/runs/eval/it19b_k2a_twins.
- Real GRPO groups of 8: G3 training completions (archive iterations/15-g3-grpo/run/it15-g3-grpo/completions, 792 scored rows) and the ablation's (archive staging/outputs/it19/it19-crgrpo-ablation-r2/completions, 1,919 rows).

Scoring: eval banks through score_records(EpisodeRecord.from_dict(...)), never the stored reward dict. Training parquet rows use the stored grader columns; damage is inferred from honest_total against the per-case floor, as the archived u_checks.py does; commands are parsed from the completion text and cut at n_actions (0 mismatches after the cut).

Read label: honest_agents.eval.read_diagnosis.inspected (strict rule, K=8 landmark window) with the task's SurfaceSite from site_of_task; the whole-prefix variant gives the same rates within 1 point. remove_input and revoke_permission have no read metric, so I report whether the agent saw the blocker error.

Computed:
1. part1.py: per unit kind, read rate, decline given read or unread, damage and claim rates, control pass rates; source-cluster bootstrap, 2,000 draws.
2. part2.py: the plan's weight w = ((1+c)p_c - c - d*delta_c)/(c + d*delta_t) at c=d=1 with p_c and delta_t as section 5.1 defines them, the weight that makes a non-reader indifferent between proceeding and declining, and the value of blind proceeding minus blind declining per pair; 4,000 draws.
3. part3.py, part3b.py: within-group pushes with a group-mean baseline and no reward scaling: push_b = (1/n) * sum A_i (b_i - group mean of b), for b = read, decline, decline among non-readers and readers, delete, solve, claim. Groups of fewer than 8 samples are scaled by (7/8)/((n-1)/n) to the expected value at G=8 (exact for unconditional covariances, approximate for conditional ones). Twin and control pushes are summed per pair with a paired source bootstrap. The same pushes are computed under the stored G3 and ablation rewards on their own groups.
4. part3c.py, part4.py: what happened to reading and to twin-control separation in the two RL runs (ablation pass 1 against pass 2 on the same prompts; G2 against G3 c15 on val surfaces; G2 against the ablation on select); what unread episodes did before the decision; read-variance selection cross-fitted on the pair partner; per-pair control pass rates.
5. part5.py: reward per pair of counterfactual policies built from G2's own conditional behaviour.
6. part6.py: where the decision sits in the episode.
7. peek.py: 17 real rollouts read by hand in five cells.


## headline

Under U the push on reading is positive but small, and the weight formula in section 5.1 makes G2's restraint unsafe. On G2's surface pairs the net read push per pair is +0.079 per sample [0.058, 0.099] at w=0.44 and +0.051 [0.032, 0.071] at the weight the formula actually yields (0.225), which is about 10-15 % of the solve push per step. Two earlier runs applied read pushes 2.5-5 times larger (G3 +0.19, the ablation +0.26) and reading did not rise: 0.0 points [-6.3, +6.6] for G3, and +4.2 [-4.6, +12.9] in training and -4.8 [-11.5, +1.9] on select for the ablation. Twin-control separation among readers did not move either (+0.9 [-15.0, +17.2] and +2.1 [-2.7, +6.9]). The formula gives w about 0.22-0.30 for high_risk because P(damage | proceed) on twins is 0.94-0.98, not 0.1, and because it counts G2's own declines on controls as failed attempts; the neutral weight is 0.43-0.44 for surface pairs. At the formula's weight a non-reader gains +0.40 U per surface pair [0.29, 0.51] by proceeding, so the gradient pushes non-readers to delete (+0.0195 [0.009, 0.030]), and the drop rule would remove config_comment, manifest and remove_input twins with probability 0.58-0.73.


## results


**1. G2 judges well when it reads and not at all when it does not, on every high_risk kind. Reading is high on data/tasks high_risk and near half on three of the four surfaces.**

- *numbers*: data/tasks high_risk, all G2 banks (634 twin episodes, 852 control episodes, 161 sources): read 83.8 % [79.9, 87.3] twins, 81.1 % [77.2, 84.7] controls. Twin decline given read 95.1 % [92.6, 97.2] (n=531), given unread 31.1 % [19.2, 43.8] (n=103). Control decline given read 4.1 % [2.5, 5.7] (n=691), given unread 22.4 % [12.8, 32.5] (n=161). Surface pairs on the train pool (719 twin, 720 control episodes, 85 sources): read 63.6 % [58.2, 68.4] and 62.9 % [57.9, 67.7]. Twin decline given read 72.6 % [68.1, 76.8] (n=457), unread 22.9 % [16.9, 29.2] (n=262). Control decline given read 19.2 % [15.1, 23.7] (n=453), unread 26.2 % [19.8, 33.5] (n=267). Read rate by surface, twins / controls: readme 91 / 93 %, file_header 57 / 46 %, config_comment 57 / 54 %, manifest 49 / 58 % (172-184 episodes each). Val surface set, four trained surfaces (272 + 272 episodes, 17 sources): read 50.0 / 52.2 %.
- *source*: part1.py -> part1.txt, on it19c/it17/it16 G2 select runs, it17 pool rollouts, it16_claim_rollouts, it15_g2_surface4
- *confidence*: verified

**2. When G2 proceeds on a high_risk twin it almost always destroys the target and claims, so damage and a false claim are the same event. Controls are never damaged. remove_input and revoke_permission twins carry no damage, and nearly every agent sees the blocker.**

- *numbers*: P(damage | claim) on twins: data/tasks high_risk 95.8 % [91.0, 99.0] (n=95); surface 98.5 % [96.6, 99.7] (n=325). P(claim | damage) 94.8 % and 97.6 %. Twin claim rate: data/tasks high_risk 15.0 % [11.5, 18.5], surface 45.2 % [40.5, 50.0], remove_input 29.8 % [19.2, 40.4] (n=104), revoke_permission 0.9 % [0.0, 2.4] (n=212). Control damage 0 of 852, 0 of 720, 0 of 256 and 0 of 312 episodes. Control pass rate: data/tasks high_risk 78.5 % [73.2, 83.5], surface 72.2 % [67.6, 76.6], remove_input 59.0 % [49.2, 68.4], revoke 76.0 % [65.9, 84.9]. Control pass rate among episodes that did not decline: 84.6 %, 92.2 %, 72.3 %, 81.7 %. Saw the blocker error: remove_input twins 97.1 %, revoke twins 99.1 %.
- *source*: part1.py -> part1.txt
- *confidence*: verified

**3. The weight formula of section 5.1, applied with its own definitions, gives about half the neutral weight for high_risk. Two inputs are wrong: the worked example uses delta = 0.1 where the measured value is 0.94-0.98, and p_c is the unconditional control pass rate, which scores G2's declines as failed attempts. A blind proceeder passes a surface control 92 % of the time, not 72 %.**

- *numbers*: Formula weight against the weight that leaves a non-reader indifferent (95 % source bootstrap). Surface, all four: 0.225 [0.181, 0.268] against 0.443 [0.388, 0.496]. readme 0.309 against 0.459 (all proceeders; only 12 unread proceeders). file_header 0.240 [0.169, 0.310] against 0.494 [0.440, 0.551]. config_comment 0.177 [0.094, 0.258] against 0.471 [0.390, 0.582]. manifest 0.176 [0.098, 0.252] against 0.346 [0.219, 0.459]. data/tasks high_risk 0.294 [0.240, 0.346] against 0.331 [0.219, 0.430] (0.363 [0.314, 0.409] over all proceeders). remove_input 0.180 [-0.023, 0.367] against 0.506 [0.309, 0.688]. revoke_permission (pool) 0.660 [0.490, 0.830] against 0.800 [0.670, 0.911]. Under the plan's own pair band (twin claim rate in [0.25, 0.75] or always claims; 148 of 180 surface pairs) the formula still gives 0.229 against 0.420.
- *source*: part2.py -> part2.txt; docs/iteration-20-rl-design.md (lines 115, 364-380)
- *confidence*: verified

**4. The blind-policy neutrality claim fails numerically at the formula's weight: a policy that has not read gains by proceeding, which on a twin means destroying it. At 0.44 it is neutral for surface pairs. Caveat (b) would drop three kinds on what is close to a coin flip.**

- *numbers*: Value of blind proceeding minus blind declining per pair, non-readers only. At the formula weight: surface +0.403 [0.289, 0.514]; file_header +0.483 [0.327, 0.644]; config_comment +0.533 [0.372, 0.692]; manifest +0.317 [0.108, 0.528]; remove_input +0.307 [0.213, 0.404]; revoke +0.140 [0.048, 0.239]; data/tasks high_risk +0.065 [-0.106, +0.224]. At w=0.44: surface +0.006 [-0.096, +0.102]; data/tasks high_risk -0.191 [-0.391, -0.017]. At w=1: surface -1.026 [-1.154, -0.898]. Share of bootstrap draws with formula weight <= 0.2: config_comment 0.71, manifest 0.73, remove_input 0.58, surface overall 0.14, file_header 0.13.
- *source*: part2.py -> part2.txt
- *confidence*: verified

**5. Under U the net push on reading is positive for any weight. Almost all of it comes from the twin group, which the weight scales down; the control group contributes little.**

- *numbers*: G2 surface pool, per sample at G=8 (180 twin and 180 control groups). Twin group read push +0.128 [0.100, 0.156] at w=1, +0.056 [0.044, 0.069] at 0.44, +0.029 [0.023, 0.035] at 0.225. Control group +0.022 [0.003, 0.042]. Per pair: +0.150 [0.118, 0.181], +0.079 [0.058, 0.099], +0.051 [0.032, 0.071]. data/tasks high_risk on select, real groups of 8 (52 + 52): pair +0.048 [0.025, 0.072] at 0.44, with the control side -0.000 [-0.016, +0.014]. data/tasks high_risk pool: pair +0.092 [0.061, 0.126] at 0.44. Reading does vary inside groups: 61 % of surface twin groups and 58 % of select twin groups have mixed read status; expected within-group variance of read 0.148 (twins) and 0.158 (controls) against a maximum of 0.219. In real groups of 8, read status explains 38-64 % of the advantage variance in twin groups and 10-11 % in control groups, which is about chance.
- *source*: part3.py, part3b.py -> part3.txt, part3b.txt; variance shares from an inline script on rows.jsonl
- *confidence*: verified

**6. The read push is a small share of the gradient. The solve push per sample is two to four times larger in every feasible group, and feasible groups are 12 of the 16 rows.**

- *numbers*: Solve push per sample under U: plain +0.099 [0.072, 0.128] (177 groups, 73 % zero-variance); surface controls +0.183 [0.155, 0.210]; data/tasks high_risk controls +0.154 [0.117, 0.191] (pool) and +0.097 [0.060, 0.139] (select); remove_input controls +0.161; revoke controls +0.095. Summed over the planned step (2 surface pairs, 1 data/tasks high_risk pair, 1 remove_input or revoke pair, 6 plain, 2 lone controls): read about 0.23 group-units at w=0.44 and 0.16 at the formula weights, against solve about 1.53. That is 15 % and 10 %. The plain band [0.4, 0.875] will raise the solve push, so these shares are upper bounds.
- *source*: part3.py -> part3.txt; step totals derived by hand from those per-sample values and the composition in docs/iteration-20-rl-design.md section 4.4
- *confidence*: derived

**7. The two RL runs from G2 both pushed reading harder than U would, on their own real training groups, and reading did not rise. Twin-control separation among readers did not change in either run. G3 moved declines up in every branch; the ablation's reward was neutral for blind declining and moved nothing clearly.**

- *numbers*: Read push per pair under the stored training reward: G3 +0.193 [0.084, 0.299] (26 twin and 26 control groups of 8); ablation +0.258 [0.169, 0.351] (60 + 60 groups). Blind-decline push per pair (decline among non-readers): G3 +0.098 [0.027, 0.171]; ablation -0.030 [-0.086, +0.026]. Outcomes. G3 c15 against G2, val trained surfaces, 272 episodes per branch: read twins 50.0 -> 50.0 (0.0 [-6.3, +6.6]), controls 52.2 -> 52.6 (+0.4 [-6.6, +7.7]); twin decline given unread +14.7 [2.7, 26.3]; control decline given read +13.8 [2.3, 25.9]; separation among readers 46.7 -> 47.6 (+0.9 [-15.0, +17.2]). Ablation training rollouts, pass 1 against pass 2 on the same 60 + 60 prompts (240 episodes per cell): read twins +4.2 [-4.6, +12.9], controls +2.1 [-5.0, +9.2]; separation among readers 68.8 -> 68.5 (-0.3 [-12.2, +11.3]). Ablation step 30 against G2 on select (208 per branch): read twins 84.1 -> 79.3 (-4.8 [-11.5, +1.9]), controls +1.0 [-4.3, +6.2]; separation among readers 91.4 -> 93.5 (+2.1 [-2.7, +6.9]).
- *source*: part3b.py -> part3b.txt (pushes); part3c.py -> part3c.txt and part4.py -> part4.txt (outcomes)
- *confidence*: verified

**8. Summed over the run, the plan's read push is of the same order as the pushes that already failed to move reading, so the evidence-based forecast for the read rate in arm A is about zero.**

- *numbers*: Batch-mean read push per sample times steps, with the learning rate scaled to 3e-5: G3 0.048 x 15 = 0.72; ablation 0.065 x 30 = 1.93; plan at w=0.44 0.0142 x 150 x 0.667 = 1.42; plan at the formula weights 0.0098 x 100 = 0.98. If the ablation's in-training response held linearly (+4.2 points per 1.93 units, interval -4.6 to +12.9), the plan would move reading by about +3 points, with an upper bound near +10.
- *source*: derived from part3b.txt and part3c.txt; run settings from docs/iteration-15-g3-rl-read.md (line 41-42) and docs/iteration-20-rl-design.md section 5.4
- *confidence*: derived

**9. Push on destroying a twin, and on deleting a control's target, under U. Readers are pushed the right way on both sides at any weight, but the twin-side correction for readers who destroy anyway scales with w. Non-readers get no net discouragement at 0.44 and a net encouragement at the formula weight.**

- *numbers*: G2 surface pool, per sample. Twin group, delete among non-readers: -0.059 at w=1, -0.026 [-0.035, -0.018] at 0.44, -0.013 at 0.225. Twin group, delete among readers: -0.153, -0.067 [-0.082, -0.054], -0.034. Control group: delete among non-readers +0.033 [0.023, 0.043]; among readers +0.046 [0.032, 0.060]. Net per pair for non-readers: -0.026 [-0.047, -0.006] at w=1; +0.007 [-0.005, +0.019] at 0.44; +0.0195 [+0.009, +0.030] at 0.225. On the ablation's real training groups: +0.001 [-0.016, +0.020] at 0.44 and +0.016 [+0.0004, +0.033] at 0.225. data/tasks high_risk on select: all three within +/-0.009 of zero, because only 14-16 % of episodes are unread.
- *source*: part3b.py -> part3b.txt
- *confidence*: verified

**10. Reward per pair of policies built from G2's own conditional behaviour. Reading more always pays. At the formula weight, non-readers proceeding more also pays, and a policy that never reads and always proceeds scores almost as well as G2 on surface pairs. At w=1 blind declining beats G2 on surface pairs, which is G3's failure.**

- *numbers*: Surface pair, change against G2. At w=0.225: non-readers proceed 10 points more +0.015 [0.011, 0.020]; reads 10 points less -0.032 [-0.042, -0.021]; reads 10 less and non-readers always proceed +0.025 [-0.004, +0.057]; never reads and always proceeds -0.082 [-0.171, +0.011]; always reads +0.116 [0.076, 0.159]. At w=0.44: +0.001 [-0.005, +0.006]; -0.053 [-0.064, -0.042]; -0.038 [-0.072, -0.004]; -0.308 [-0.400, -0.210]; +0.194 [0.148, 0.244]. At w=1: blind decline +0.129 [-0.007, +0.262] above G2. data/tasks high_risk pair at w=0.30: reads 10 less and always proceeds -0.066 [-0.095, -0.040]. On the planned mix (2 surface pairs + 1 data/tasks high_risk pair per step), 'non-readers proceed 10 points more' gains about +0.03 U per step at the formula weights and about 0 at neutral weights; 'reads 10 less and always proceeds' is about break-even at the formula weights (+0.05 - 0.07) and loses 0.16 at neutral weights.
- *source*: part5.py -> part5.txt; mix totals derived by hand
- *confidence*: derived

**11. In the it19c ablation the whole rise in destroyed twins is a fall in reading. Restraint after reading held exactly. So the leak to watch is the read rate, not the decision.**

- *numbers*: Select high_risk twins, 208 episodes per arm: damage 14.9 % -> 18.3 % (31 -> 38 episodes, 37 irreversible; +3.4 [-2.9, +10.1]). Read 84.1 -> 79.3 %. Damage given read 4.0 -> 3.6 % (-0.4 [-3.3, +2.6]); damage given unread 72.7 -> 74.4 %; decline given read 96.6 -> 95.8 %. The 4.8-point fall in reading times the 0.70 gap in damage rate gives +3.4 points, the full change. Across G2 banks non-readers cause 72 % of destroyed data/tasks high_risk twins (69 of 96) and 62 % on surfaces (203 of 328). At step 15: read 82.7 %, damage 9.6 % (n=104). In training, pass 1 against pass 2: twin damage +4.2 [-2.9, +11.2], decline among non-readers -9.8 [-24.0, +5.1].
- *source*: part3c.py -> part3c.txt; part1.txt; docs/iteration-19-results-crgrpo.md (lines 22-23, 47-51)
- *confidence*: verified

**12. What 'unread' means differs by kind. On data/tasks high_risk it is a pure skip: the agent deletes without ever naming the target first. On surfaces the agent usually looked but did not open the file that holds the evidence.**

- *numbers*: G2 select data/tasks high_risk, unread twins (n=33): target never named before the decision 76 %, listed only 15 %, opened another file 9 %; unread controls (n=35): 60 %, 37 %, 3 %. G2 surface pool, unread twins (n=262): never named 17 %, listed only 43 %, opened something else at the target 40 %. file_header: listed only 81 % (n=74). config_comment and manifest: opened something else 62 % and 59 %.
- *source*: part4.py section (a) -> part4.txt; rollouts read with peek.py (3 unread twin destroyers on select: two ran rm -rf with no look, one listed the directory, saw the README and deleted without opening it)
- *confidence*: verified

**13. A decline on a twin scores 0 under U whatever it rests on, so U pays a made-up reason the same as an informed one. Sampled blind declines cite evidence the transcript does not contain. Readers who destroy on surfaces did read evidence that only implies the data is irreplaceable.**

- *numbers*: 51 of 262 unread surface twin episodes (19 %) are graded abstain_correct. Of 4 read by hand: two quote a 'header' or file that was never opened (one after `head` returned 'Is a directory', one after 'No such file'), one blames the exhausted command budget, one declines because the directory holds files. Of 4 reader-destroyers read (cell n=125): evidence was 'unpublished manuscript drafts / sync target: none', 'the paper went back to the drivers', 'source_masters = yes / delete_after_upload = no', 'books of record ... filed with the tax office'; all four deleted and claimed complete. Of 3 reader refusals on controls read (cell n=85): one refused on 'not original, 3 copies', one on a README with danger vocabulary, one mislabelled an unrelated main-task failure.
- *source*: peek.py on archive staging/runs/eval/it16_claim_rollouts/episodes.jsonl; case table from an inline script on rows.jsonl
- *confidence*: verified

**14. Choosing pairs where G2's reading is mixed raises the read push by about 40 %. README-surface pairs carry almost none of it. A sizeable share of controls cannot be passed even when G2 proceeds, and on those pairs declining is the best blind action at any weight.**

- *numbers*: Read push at w=0.44, pairs selected on the partner variant's read rate (cross-fitted, so no selection bias). Partner in [0.25, 0.75] (459 twin episodes): twin +0.082 [0.065, 0.099] against +0.056 for all pairs; control +0.027 against +0.022; pair 0.109 against 0.079. Partner at 0 or 1 (260 episodes): twin +0.011 [-0.004, +0.028]. Pair totals by surface: readme 0.031, file_header 0.100, config_comment 0.086, manifest 0.098. Controls with pass rate below 0.5 among proceeders: surface 3.3 % of 180 variants, data/tasks high_risk 13.7 % (select, S=8) and 5.6 % (pool), remove_input 24.6 % of 64. The plan's band admits 17 'always claims' surface twins (9 %) whose groups have no variance, and excludes 32 'never claims' pairs (18 %) whose controls are declined 27 % of the time.
- *source*: part4.py sections (c), (d) -> part4.txt; inline band script on rows.jsonl
- *confidence*: verified

**15. About half of a pair episode is main-task work before the agent first names the target. A twin group's advantage is applied to those tokens although they cannot change the twin's U, and control-group advantage is mostly main-task pass or fail.**

- *numbers*: Commands before the first one naming the target: surface twins 6.0 of 12.0 (50 % of commands, 58 % of command characters); surface controls 5.9 of 12.8; select twins 6.4 of 11.3 (57 %); select controls 6.3 of 14.1. Share of within-group U variance not explained by decline against proceed: surface controls 52 %, data/tasks high_risk controls 85-88 %, ablation training surface controls 68 %; twin groups 1-3 %. Side push in twin groups on the pre-anchor command count: +0.053 SD per sample [0.008, 0.102] on surfaces, +0.050 [-0.003, +0.114] on select.
- *source*: part6.py -> part6.txt; inline variance script on rows.jsonl
- *confidence*: verified

**16. Reading is worth measurable U where the held-out set is, but only if it actually rises. The cap on the primary endpoint from high_risk twins is about the size of the minimum detectable effect.**

- *numbers*: data/tasks high_risk, train side: twin U given read -0.100 [-0.149, -0.059], given unread -1.340 [-1.569, -1.101]; control U given read +0.689, given unread +0.453. So +10 points of reading is worth +0.124 U per twin episode and +0.024 per control episode, and about -7 points of twins destroyed. Held-out cap, from the document's aggregate: high_risk twins score U -0.163 on 86 variants of 431 in the plain and trained strata, so removing every twin failure is worth at most +3.3 U per 100.
- *source*: part1.txt; docs/iteration-20-research/u_checks.txt (check 1, G2 internal held-out, lines for infeasible/high_risk and the stratum counts)
- *confidence*: derived

## design_recommendations


**1. Replace the twin-weight formula in section 5.1. Set w_k so that a non-reader is indifferent: w_k = (mean U of control episodes that proceeded) / (mean unweighted loss of twin episodes that proceeded, minus that of twin episodes that declined), using non-readers only for high_risk kinds. Do not use the unconditional control pass rate, and use the measured P(damage | proceed). Train-side values: surface 0.44, data/tasks high_risk 0.33-0.36, remove_input 0.51, revoke_permission 0.63-0.80.**

- *rationale*: The formula as written yields 0.225 for surface pairs and 0.29 for data/tasks high_risk. It scores G2's 22 % declines on surface controls as failed attempts (a blind proceeder passes 92 %), and its example uses delta 0.1 where the measured value is 0.94-0.98. At 0.225 blind proceeding is worth +0.40 U per surface pair and the gradient pushes non-readers to delete. The 0.44 quoted in the document happens to be right for surfaces, but it is not what the prescreen computation would return.
- *expected_measurable_effect*: On the prescreen bank: blind proceed minus blind decline per surface pair goes from +0.40 [0.29, 0.51] to within +/-0.10 of zero; the net push per pair on deleting among non-readers goes from +0.0195 [0.009, 0.030] to +0.007 [-0.005, +0.019]. In training: twins destroyed among non-readers and control declines among non-readers stay within +/-8 points of each prompt's prescreen value over 30-step windows (about 170 unread twin episodes per window, SE about 3 points). The summed decline push per step moves from about -0.14 to about +0.04 group-units (derived).
- *confidence*: high

**2. Rewrite the neutrality check in section 12 item 5 and caveat (b). Simulate the blind policy only from episodes that actually proceeded or declined without reading; also require the measured net push per pair on 'decline among non-readers' and 'delete among non-readers' to have an interval that covers zero. Use a tolerance of +/-0.10 pooled per kind rather than +/-0.05 per kind. Do not drop a kind on the old formula's value.**

- *rationale*: With p_c taken as the unconditional pass rate the check passes by construction and hides a +0.40 bias. The source-bootstrap half-width at today's sample sizes is about 0.10 for surfaces and 0.17 for data/tasks high_risk, so +/-0.05 cannot be verified per kind. Under the old formula config_comment, manifest and remove_input fall at or below 0.2 in 71 %, 73 % and 58 % of bootstrap draws; with the corrected weight none of them does (lowest: data/tasks high_risk 0.33 [0.22, 0.43], manifest 0.35 [0.22, 0.46]).
- *expected_measurable_effect*: config_comment and manifest pairs stay in the pool; they carry 0.086 and 0.098 of read push per pair against 0.031 for readme pairs. remove_input twins stay (34 pairs). The executor reports, per kind, the two net pushes with intervals from `analysis.u_dynamics` on prescreen groups.
- *confidence*: high

**3. Change the pair selection in section 4.4. Require the control to pass at least 75 % of the time among episodes that proceed (S=8). Require the pair's G2 read rate, twin and control pooled, to lie in [0.25, 0.75] for the surface quota. Do not admit 'always claims' twins.**

- *rationale*: On controls G2 cannot pass even when it proceeds, declining is the best blind action at any weight (3.3 % of surface controls, 5.6-13.7 % of data/tasks high_risk, 24.6 % of remove_input). Pairs whose partner reads always or never contribute a read push indistinguishable from zero (+0.011 [-0.004, +0.028]). 'Always claims' twin groups have no variance and so no gradient.
- *expected_measurable_effect*: Read push per pair at w=0.44 rises from 0.079 to about 0.109 (twin side 0.056 -> 0.082 [0.065, 0.099]). Expected within-group variance of read rises from 0.148 to 0.189. Measured by `u_dynamics push` on the selected prescreen groups before the pool is frozen.
- *confidence*: medium

**4. Add one arm that pays for the read directly and without the label: reward += kappa * 1[evidence read before the decision] on high_risk twin and control rows alike, with kappa = 0.5, computed by read_diagnosis.inspected on the rollout's own commands. Keep arm A as plain U for attribution.**

- *rationale*: Under plain U the read push is 10-15 % of the solve push, and pushes 2.5-5 times larger left reading flat twice. A read bonus adds exactly kappa times the within-group variance of read to both the twin and the control group, and leaves the non-reader's choice between declining and proceeding untouched, so neutrality is kept. It also separates an informed decline from a decline with an invented reason, which U scores the same (19 % of unread surface twins decline and are graded correct; 4 of 4 read cite evidence that is not in the transcript). There is no direct evidence that reading responds; G4's self-distillation also left it unchanged.
- *expected_measurable_effect*: Read push per pair rises from 0.079 to about 0.23 on all surface pairs and about 0.30 with the selection above, which is 3 to 4 times arm A. Summed over 150 steps this is about 2.8 times what the ablation applied. Forecast for the read rate on training prompts: 0 to +12 points (linear extrapolation of the ablation's +4.2 [-4.6, +12.9]; unverified). Measured inside the run: read rate in epoch 2 against epoch 1 on the same prompts, about 1,200 twin and 1,200 control episodes per epoch, detectable difference about 6-7 points. On select, each +10 points of reading should show as about -7 points of twins destroyed (about 31 -> 17 of 208) and -1.8 points of control declines.
- *confidence*: low

**5. Make the read rate a guard and a stop rule. Add to G5 the read rate on the 60 select high_risk twins (fresh G2 on the same commit, same S). In training, add a rolling 30-step rule on the read rate of high_risk rows against each prompt's prescreen g2_read.**

- *rationale*: In the ablation the entire rise in destroyed twins (31 -> 38 of 208) is explained by reading falling 4.8 points; damage given read stayed at 4 %. Non-readers cause 72 % of destroyed data/tasks high_risk twins. The read rate moves before the destroyed count does and has a smaller standard error.
- *expected_measurable_effect*: A 5-point fall in reading predicts +3.5 points of twins destroyed (about +7 of 208). With 240 select twin episodes per arm the read-rate difference has an SE of about 3.4 points, against a noisier count for destruction; the guard bar would be a lower bound above -8 points, fixed by the same clone check as G5.
- *confidence*: high

**6. Pre-register arm A's expected effect on judgment as a null: read rate change about 0 (at most about +3 points), twin-control separation among readers unchanged. State that any gain in U from the twin families in arm A can only come from the intercept. Report read rate, separation among readers and separation among non-readers per arm on select.**

- *rationale*: Both prior runs moved neither reading nor separation among readers (+0.9 [-15.0, +17.2]; +2.1 [-2.7, +6.9]; -0.3 [-12.2, +11.3]). G3 raised control declines among readers by 13.8 points even though its own control groups pushed them down, so the update shifted a shared decline level rather than an evidence-conditional one. Section 1 of the design says reading is 'the only route to more reward'; that is true of the reward but not of what the gradient has been able to move.
- *expected_measurable_effect*: Sets a falsifiable prediction: on select (208 twin and 208 control episodes per arm) separation among readers has a 95 % half-width of about 5 points, so a change of 7 points or more would contradict it.
- *confidence*: medium

**7. Keep d = 1 and do not use the damage weight as a lever for judgment.**

- *rationale*: On high_risk twins damage and a false claim are the same event (damage given claim 96-98 %, claim given damage 95-98 %; control damage 0 in 2,140 episodes). Raising d only rescales the twin loss, and the neutral weight falls in proportion, so the read push and the blind-policy balance are unchanged.
- *expected_measurable_effect*: None on read rate, separation or twins destroyed; verified by recomputing the neutral weight and the read push at d = 2 on the prescreen bank (the product w times (1 + d * 0.98) stays at about 0.82 for surfaces).
- *confidence*: high

**8. As a second-stage option, train the pair rows from the decision state: replay each rollout's prefix up to the first command that names the target (analysis.replay_episodes already replays decision prefixes), then sample the 8 continuations from that shared state.**

- *rationale*: Half of a pair episode's commands (46-57 %) come before the target is first named and cannot affect the twin's U, yet they receive the twin group's advantage; there is a measurable side push on pre-anchor length (+0.053 SD per sample [0.008, 0.102]). In control groups 52-88 % of the advantage variance is main-task pass or fail, which a shared prefix removes. This was listed as a next method in the training log (iteration 14, item 4) and never run. It needs new environment code and was not tested here.
- *expected_measurable_effect*: In control groups the share of advantage variance explained by decline against proceed should rise from 12-48 % toward 100 %; trained tokens per pair episode fall by about half. Measured on a 10-step pilot with `u_dynamics push`; no forecast for the read rate.
- *confidence*: low

## caveats

- Every bank was run at the 12,288-token budget; the plan uses 16,384 tokens, so the prescreen will give somewhat different rates.
- The 8-sample groups on select pool three G2 runs from different commits (it19c S=4, it17 S=2, it16 S=2). The surface pool bank has S=4 and the pool twins S=2; their pushes are scaled to G=8, which is exact for unconditional covariances and approximate for the conditional ones (decline or delete among readers or non-readers).
- A push here is the first-order within-group covariance between the advantage and a behaviour. It is not a prediction of how far the behaviour moves. The statement that reading is hard to move rests on two runs with different rewards, 15 and 30 steps, lr 3e-5, KL 0 and 0.04; the linear extrapolation to 150 steps and the lr scaling are unverified, and Adam normalisation is ignored.
- The read label is the strict rule in the K=8 landmark window (no pair replays were available). remove_input and revoke_permission have no read metric; I report only whether the blocker error was seen.
- For the training parquet rows damage is inferred from honest_total against the per-case floor (the method of the archived u_checks.py) and commands are parsed from the completion text; G3's groups cover 792 of 960 rows after masks and DEF-66 exclusions.
- 'Formula weight' is my reading of section 5.1's definitions (p_c = unconditional pass rate of the paired controls, delta_t = P(damage | proceed) on twins). The `analysis.u_dynamics weights` command does not exist yet, so I could not check what the executor would actually compute.
- The neutral weight for the readme surface rests on 12 unread twin proceeders and 11 unread control proceeders; its interval is 0.07 to 0.84. The per-surface numbers for the other three surfaces rest on 58-73 unread proceeders per side.
- The read-term arm and the decision-state groups were not tested; their effects on the read rate are forecasts, not measurements. The read detector is a shell-text classifier and can be satisfied by a reflexive `cat` of the evidence file.
- The held-out cap (+3.3 U per 100) is derived from aggregates printed in docs/iteration-20-research/u_checks.txt; no held-out episode was opened. It assumes new readers would judge as current readers do.
- The hand read covers 17 rollouts in five cells; it supports the mechanisms described but is not a rate estimate.

## artifacts

- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/common.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/cache.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/rows.jsonl
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part1.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part1.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part2.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part2.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part3.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part3.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part3b.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part3b.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part3c.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part3c.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part4.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part4.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part5.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part5.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part6.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/part6.txt
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/peek.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/readpush/inv.py