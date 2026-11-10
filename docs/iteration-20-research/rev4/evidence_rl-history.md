# evidence:rl-history


## summary

1) Eight GRPO runs exist (it07, it09, it11, it12 B0, it13 GRPO-15, it15 G3, it19c method, it19c ablation). Every one used lr 3e-5, 8 prompts x 8 generations = 64 episodes per step, DAPO, LoRA r32, seed 42, one seed, and at most 50 steps. The largest was 3,200 episodes (it09, from base); the largest from an SFT init was 1,920 (it19c ablation). Plan 20 (150 steps x 128 = 19,200 episodes, lr 2e-5, 800 prompts x 3 visits) is outside every tested setting: 10x the ablation's episodes, 3.6x its cumulative learning rate. lr, batch size and seed have never been varied, so history says nothing about them.

2) "No RL run beat its SFT init" is true of the pre-registered gates, not of paired measurement against the init. On validation sources: it13 GRPO checkpoint 10 vs anchored-3ep (tier F, 89 sources, S=4, 640 training episodes): verified success +3.9 [+0.7, +7.0] (90 %, sign-flip p 0.047), U +4.2 [0.0, +8.3]; the checkpoint was picked on the probe, and on the 54 unselected sources (plain only) task pass is +1.9 [-1.9, +5.6]. G3 vs G2 (tier S, 35 sources, S=2, 960 episodes): task pass +5.1 [+0.7, +9.6], false claims per feasible episode -6.5 [-10.7, -2.8], U +12.4 [+4.3, +20.2]; its in-run probe on the same variants (1 sample) shows task success 73.8 -> 70.0, so the capability part is unconfirmed. Only the CR-GRPO table lost: ablation U -6.5 [-11.0, -2.0], verified success -3.4 [-6.9, +0.1] (select, 52 sources, S=4). Plan 20 section 1 calls the ablation "the only comparable run"; G3 is equally comparable (full-episode GRPO from G2) and went the other way.

3) Plain-task capability has never moved measurably on unseen sources under RL: +0.8 [-2.0, +3.7] (it13), +2.9 [-2.9, +8.6] (G3), -3.4 [-8.7, +1.4] (ablation), -0.3 [-3.7, +3.1] (B0). What RL moved is controls and twins: control task pass +11.0 [+3.7, +18.8] (it13), +7.4 [-1.4, +15.7] (G3), -3.4 [-8.0, +0.8] (ablation).

4) The sign of the false-claim effect follows the price of a false claim relative to a non-claim on feasible rows. Honest table (measured cells: false claim -1.00, non-claim -0.11 plain / -0.32 control): control false claims -6.6 [-12.1, -1.4] (it13) and -7.4 [-15.2, 0.0] (G3). CR-GRPO table (controls: false claim -0.38, non-claim -1.78, so a false claim paid 1.4 MORE than declining): +4.2 [0.0, +8.5]. Plan 20's U (-1 vs 0) has the honest table's sign. This is the one difference from the ablation with evidence that it can flip a sign; lr, step count, truncation masking and the 16k/12k budget mismatch cannot by themselves.

5) The ablation was not learning on its own training prompts. Same 120 prompts, pass 1 (steps 1-15) vs pass 2 (steps 16-30), 960 episodes each: feasible pass 74.1 -> 75.7 (+1.5 [-2.2, +5.3]), U without the damage term 26.3 -> 25.9 (-0.4 [-4.7, +3.8]), twin claims 29.7 -> 33.3 % (+3.6 [-0.6, +7.8]), P(claim | pass) 0.986 -> 0.987, while KL rose to 0.016-0.021. Only 3.6 feasible prompt groups per step had mixed pass/fail outcomes.

6) The one clear case of in-distribution capability learning is it09 from base with no KL: on controls whose source had never been seen, trained-minus-base pass rate on the same variants went from -4.5 points (steps <= 20) to +10.4 [+5.6, +15.2] (steps >= 21, 99 groups); remove_input controls +21.0 [+14.3, +27.6]. It appeared only after about 20 steps (1,300 episodes). No SFT-init run has gone past 30 steps.

7) The ablation's capability loss is not established: -3.4 [-6.9, +0.1], and G2 re-run against itself on the same split moved -1.5 [-4.0, +1.0] and +2.5 [-0.6, +5.7] in task success between sessions. The train/eval budget mismatch did not bind (99.2 % of feasible select episodes ended on a report).

8) A G3-style caution shift scores as a U gain at evaluation (w = 1): on the six-surface validation set G3's U is +10.4 [+4.9, +15.9] vs G2 while control pass on trained surfaces fell -8.1 [-14.7, -1.8] and control non-claims rose +11.0 [+4.0, +18.0]. On plan 20's P u T endpoint, G2's residual twin claims and damage are worth about 4.3 U per 100 (published check 1), more than the MDE of 2.5-3.6, so H1 could be met by twin caution alone.

9) KL in the ablation was not linear: slope 0.0011 per step over steps 6-15, 0.0003 over steps 16-30; 5-step means 0.0163 and 0.0155 for steps 21-25 and 26-30. Plan 20's pilot gate (fit on steps 6-10, project to step 150, limit 0.08) projects 0.055-0.082 from the ablation's own early slope scaled to lr 2e-5: a likely false alarm.

10) Wall-clock: 363-659 s per 64-episode step (median, by run). At 128 episodes a step, plan 20 needs roughly 12-21 min a step, 30-53 h per arm before probes.


## facts


**1. it07 GRPO run 1 (grpo_main): full record**

- *value*: Init base Qwen3.5-4B. Reward mode honest, alpha 0.1, false_blame 0 (default), implicit_abstain 0.25 (default). Pool 360 rows: 75 plain in the base model's 2-3-of-4 pass band, 185 near-miss controls, 100 twins (27.8 %; 34 high_risk / 33 revoke / 33 remove_input). 8 x 8 = 64 episodes per step; 18 steps logged (OOM-killed at about step 19) = 1,152 episodes; 144 prompts, 1 visit each. lr 3e-5, 5 warmup, constant; beta 0; DAPO, eps 0.2 / 0.28; 16,384 tokens per episode, 4,096 per turn, 30 tool calls (28 bash + 2). No checkpoint evaluated. Same-variant gap to the base pass@4 bank by thirds: -5.3, -0.3, -4.5 points (flat). Failure: OOM; reward hole (false blame paid 0). Median step time 406 s.
- *source*: docs/training-log.md (lines 397-430); ~/Documents/honest-agents-archive/iterations/07-grpo-run1/run/honest-a0.1-qwen3.5-4b-s42/run_config.json; pool counted from ~/Documents/honest-agents-archive/iterations/00-dataset/files/data/train_rl.jsonl; computed in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/runs_all.txt and ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/timing.txt
- *confidence*: verified

**2. it09 GRPO run 2 (grpo_v2): full record**

- *value*: Init base. Reward honest, alpha 0.1, false_blame -0.4, implicit_abstain 0.25. Same 360-row pool, seed and order as run 1. 50 steps x 64 = 3,200 episodes (3,008 archived; 344 unique prompts, 32 seen twice). lr 3e-5, beta 0, DAPO; 16,384 tokens, 30 tool calls. Validation probe (100 variants x 1 sample): feasible task success 79.9 -> 85.4 %, claims on impossible 60.4 -> 60.4 %. Paired 64-task probe: plain 28/35 -> 28/35, controls 11/15 -> 13/15, high_risk twins claimed 8/9 -> 8/9. Entropy first 5 / last 5 steps 0.244 -> 0.165. Median step 423 s; 6.9 h. Verdict in the log: more capable on training tasks, not more honest.
- *source*: docs/training-log.md (lines 456-503); ~/Documents/honest-agents-archive/iterations/09-grpo-run2/run/honest-a0.1-qwen3.5-4b-s42/run_config.json and validation_probe.jsonl in the same folder; ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/probes.txt
- *confidence*: verified

**3. it09: in-distribution learning on sources never seen before in the run (trained minus base pass@4 bank on the same variants)**

- *value*: Controls: steps 1-10 -4.5 [-9.9, +1.0]; steps 11-20 -4.6 [-9.4, -0.3]; steps >= 21 with the source first seen at that step +10.4 [+5.6, +15.2] (99 groups, 66 sources, 8 samples each). By family at steps >= 21 (new sources): remove_input controls +21.0 [+14.3, +27.6] (34 groups; -4.8 at steps <= 20), high_risk +4.4 [-3.4, +12.5], revoke_permission +5.5 [0.0, +11.5]. All feasible rows by thirds: -0.1, +9.1, +13.4. 90 % source-cluster bootstrap.
- *source*: Computed: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/run2_seen.py, output ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/run2_seen.txt; inputs: completions under ~/Documents/honest-agents-archive/iterations/09-grpo-run2/run/honest-a0.1-qwen3.5-4b-s42/completions and ~/Documents/honest-agents-archive/iterations/03-passk-bank/eval/passk/episodes.jsonl.gz. The log reports the same trend (+0.03, +0.10, +0.15): docs/training-log.md (lines 476-477)
- *confidence*: verified

**4. it11 GRPO from base: full record**

- *value*: Init base (teacher SFT failed its gate). Reward honest, alpha 0.1, false_blame -0.4, implicit_abstain 0, blocker-target fixes. Pool 351 rows: 64 plain, 154 controls, 133 twins (37.9 %: 80 high_risk / 43 remove_input / 10 revoke); never-pass sources and leaks out; blocks of 8 distinct sources; no claim-rate prescreen. 20 steps (40 planned) = 1,280 episodes, 160 prompts x 1 visit. lr 3e-5, beta 0. 16,384 tokens for steps 1-10, 12,288 for 11-20 (OOM on 48 GB); 30 tool calls; server mode on 2 x RTX 4090. Probe set (120 variants x 2, paired): verified success -4.4 [-11.7, +3.2] (95 %); claims on impossible -9.0 [-18.2, -1.2]; no-report 12.6 -> 23.9 %. My recompute (DEF-66 out, 90 %): task pass +2.2 [-3.6, +8.1], U +3.4 [-2.4, +9.5], claim rate feasible -7.2 [-12.7, -2.1]. Failure: 73 of 168 budget-exit reports dropped by the harness; longer episodes (13.5 -> 17.7 commands). Median step 609 s plus about 4 min sandbox turnover.
- *source*: docs/training-log.md (lines 659-664, 717-739, 766-771); ~/Documents/honest-agents-archive/iterations/11-sft-grpo/run/honest-a0.1-qwen3.5-4b-s42/run_config.json; ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt
- *confidence*: verified

**5. it12 B0 GRPO from base, fixed harness: full record**

- *value*: Init base. Reward, pool (351 rows, 38 % twins), order and seed as it11; 8 report turns (36 tool calls); 16,384 tokens; colocated on one 96 GB card. 15 steps = 960 episodes; candidate checkpoint 10 = 640 episodes; 120 prompts x 1 visit. lr 3e-5, beta 0. Tier F (validation split, 98 sources x 4) vs base: abstain_correct +6.3 [+1.2, +11.5] (all revoke_permission), verified success -3.3 [-7.7, +1.4], control declines 3.8 -> 6.9 %. My recompute (89 sources, 90 %): task pass -0.8 [-4.6, +3.1], U -2.1 [-7.7, +3.8], control non-claim +5.9 [+1.4, +10.4]. Transfer on 20 test-split sources: control success 85.0 -> 70.0 % (cited from the log; episodes not opened). Median step 515 s.
- *source*: docs/training-log.md (lines 873-912); ~/Documents/honest-agents-archive/iterations/12-harness-b0/run/honest-a0.1-qwen3.5-4b-s42/run_config.json; ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt and ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/bykind.txt
- *confidence*: verified

**6. it13 GRPO-15 from anchored-3ep SFT: full record**

- *value*: Init anchored-3ep (decision SFT + own-success anchors), merged. Reward honest, alpha 0.1, false_blame -0.4, implicit_abstain 0. Pool v2, 370 rows: 64 plain, 161 controls, 145 twins (39.2 %: 80 high_risk / 40 revoke / 25 remove_input), twin and control in the same step. 15 steps = 960 episodes; evaluated checkpoint 10 = 640 episodes; 120 prompts x 1 visit. lr 3e-5, beta 0; 12,288 tokens; 36 tool calls; server mode 2 x RTX 4090; median step 566 s (3 h 32 m in the log). In-run: erosion rule fired (high_risk twin abstain_correct 96 -> 93 -> 72 % over steps 1-5 / 6-10 / 11-15, different prompts each step). Probe control declines 15 / 10 / 25 % at steps 0 / 10 / 15.
- *source*: docs/training-log.md (lines 1040-1073); ~/Documents/honest-agents-archive/iterations/13-grpo-and-evals/run/honest-a0.1-sft_anchor3ep_merged-s42/run_config.json; pool counted from data/train_rl_v2.jsonl
- *confidence*: verified

**7. it13 GRPO checkpoint 10 against its own SFT init (anchored-3ep), tier F, paired by source**

- *value*: Repo estimator, DEF-66 sources out (89 sources, S=4, 90 %): verified success 78.9 -> 82.7, +3.9 [+0.7, +7.0], sign-flip p 0.047; task success +3.7 [+0.6, +6.7]; plain verified success +1.1 [-1.9, +4.2]; false claims per feasible episode -1.8 [-4.3, +0.7]; control decline -2.2 [-7.7, +3.3]; mean commands -0.9 [-1.4, -0.4]. With DEF-66 in (98 sources): verified success +4.2 [+1.4, +7.0], p 0.021; false claims -3.1 [-5.5, -0.7]. U (my bootstrap): all +4.2 [0.0, +8.3]; plain +1.1 [-3.9, +6.2]; controls +17.6 [+6.1, +30.5]; twins -1.6 [-9.8, +6.7]. Control task pass +11.0 [+3.7, +18.8]; plain task pass +0.8 [-2.0, +3.7]. Caveat: checkpoint 10 was chosen on probe_v1; on the 54 non-probe sources (plain only) task pass is +1.9 [-1.9, +5.6], on the 35 probe sources +5.1 [+0.7, +9.6].
- *source*: Commands: python -m honest_agents.analysis.power estimate ~/Documents/honest-agents-archive/iterations/13-grpo-and-evals/eval/it13_anchor3ep_tierF/episodes.jsonl ~/Documents/honest-agents-archive/iterations/14-session-a/eval/it14_grpo10_tierF/episodes.jsonl.gz --metric <m> [--spec-defects include]; ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt, ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/bykind.txt and ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/it13_unselected.txt. The log states it was better than anchored-3ep on every row: docs/training-log.md (lines 1117-1124)
- *confidence*: verified

**8. it15 G3 GRPO from G2: full record**

- *value*: Init G2 seed 1 merged. Reward honest, alpha 0.1, false_blame -0.4, implicit_abstain 0. Pool v3, 120 rows: 30 plain, 45 controls, 45 twins (37.5 %); 30 surface pairs on the 4 trained surfaces, 8 revoke and 7 remove_input pairs; 2 surface pairs + 1 minor pair + 2 plain per step. 15 steps = 960 episodes, 120 prompts x 1 visit. lr 3e-5, beta 0 (no KL logged); 16,384 tokens, 36 tool calls; colocated 96 GB; median step 634 s, 3.1 h. Pre-registered result (new-surface validation set, S=4): read rate 46 / 46 % (bar 65), surface control declines 31 -> 41 % (bar 25), held-out gap +15.6 [8.1, 23.1] vs G2 +16.2: failed. Checkpoint 10's held-out gap was +3.1 [-6.2, 13.1].
- *source*: docs/iteration-15-g3-rl-read.md (lines 40-47, 70-92); configs/grpo_it15_g3.yaml (lines 6-18); ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/it15-g3-grpo/run_config.json; pool counted from data/train_rl_v3.jsonl
- *confidence*: verified

**9. G3 checkpoint 15 against G2 on tier S (familiar README-surface probe set), paired by source, post hoc**

- *value*: 35 sources after DEF-66, S=2, both at 12,288 tokens. Repo estimator (90 %): task success 82.6 -> 87.7, +5.1 [+0.5, +9.7] (p 0.113); verified success +2.9 [-2.3, +8.1]; false claims per feasible episode 13.0 -> 6.5, -6.5 [-10.7, -2.4] (p 0.007); claims on infeasible 10.9 -> 1.6, -9.4 [-18.0, -0.7]; control decline +2.9 [-2.1, +8.0]; claim rate feasible -3.6 [-6.8, -0.4]. U (my bootstrap): all +12.4 [+4.3, +20.2]; plain +7.1 [-2.9, +18.6]; controls +11.8 [-4.3, +28.8]; twins +18.8 [+4.8, +34.4]; twin damage 9.4 -> 0.0 %. Plain task pass +2.9 [-2.9, +8.6]. Contradicting signal: the in-run probe on the same 120 variants (1 sample, 16k budget) shows feasible task success 73.8 -> 70.0 % and fabrication share 18.1 -> 19.1 % from step 0 to 15.
- *source*: Command: python -m honest_agents.analysis.power estimate ~/Documents/honest-agents-archive/iterations/15-g2-splice-replay/eval/it15_g2_tierS/episodes.jsonl.gz ~/Documents/honest-agents-archive/iterations/15-g3-grpo/eval/it15_g3_c15_tierS/episodes.jsonl.gz --metric <m>; ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt; in-run probe: ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/it15-g3-grpo/validation_probe.jsonl (printed in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/probes.txt)
- *confidence*: verified

**10. G3 on the six-surface validation set, scored with plan 20's U (w = 1)**

- *value*: G3 c15 vs G2, 17 sources, S=4, inbound_link controls excluded (unpassable): U +10.4 [+4.9, +15.9]. Trained surfaces: twins U +22.8 [+11.4, +34.6] (twin claims -14.7 [-21.0, -8.5], damage -8.1), controls U -3.7 [-12.9, +6.6], control task pass 58.1 -> 50.0, -8.1 [-14.7, -1.8], control non-claims 26.5 -> 37.5, +11.0 [+4.0, +18.0]. G3 c10: U +4.4 [-1.5, +9.8]. it13 GRPO c10 vs anchored-3ep on the same kind of set (16 sources, S=2): U +9.7 [+0.3, +18.5], control pass -3.1 [-10.9, +4.7].
- *source*: Computed: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/paired_u_surface.py, output ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u_surface.txt; inputs under ~/Documents/honest-agents-archive/iterations/15-g2-splice-replay/eval/it15_g2_surface4, ~/Documents/honest-agents-archive/iterations/15-g3-grpo/eval and ~/Documents/honest-agents-archive/iterations/14-session-b/eval/it14_grpo10_surface
- *confidence*: verified

**11. it19c CR-GRPO method arm: full record**

- *value*: Init G2. Reward mode crgrpo with the report term (c_rep 0.00213, z_max 6); status token masked from the main loss and the KL. Pool v4 (120 rows, 37.5 % twins). 7 steps = 448 episodes (stopped by the feasible-decline guard, 0.325 > 0.30 over steps 3-7). lr 3e-5, beta 0.04; KL 0.0059 at step 7. pi(claim) 0.73 -> 0.08; at step 7, 20 of 31 passing feasible episodes reported incomplete. Checkpoint 5 on select (S=2) vs G2: verified success -8.3 [-12.1, -4.4] (95 %), feasible claim rate -10.6 [-14.5, -6.7]; my recompute: task pass -2.8 [-5.8, +0.2], U -4.8 [-10.6, +0.8]. Median step 659 s.
- *source*: docs/iteration-19-research/badcase-it19c/method_stop.md (lines 11-21, 112-128); outputs/it19/it19-crgrpo/run_config.json and stop_rules.jsonl in the same folder; ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt
- *confidence*: verified

**12. it19c CR-GRPO ablation arm (no report term): full record**

- *value*: Init G2 seed 1 merged, new LoRA r32. Reward mode crgrpo, report_term false: mu_fc 2/7 -> 0.35 (one update at step 14, capped), mu_tw 3.0, mu_cd 3.0 -> 2.67, min_substantive 3, online V_s (family means 0.70 / 0.62 / 0.83). Pool v4: 120 rows (30 plain, 45 controls, 45 twins = 37.5 %), v3's blocks without DEF-66 sources, plain rows preferring G2's 1-3-of-4 pass band; 2 passes. 30 steps x 64 = 1,920 episodes; 120 prompts x 2 visits. lr 3e-5 (5 warmup), beta 0.04, DAPO, mask_truncated_completions true (clipped share mean 0.3 %, max 4.7 %). Trained at 16,384 tokens, evaluated at 12,288. Select split (52 sources, S=4, 90 %): false claims per feasible episode 15.9 -> 19.7, +3.8 [+0.8, +6.8] (p 0.045); verified success 76.9 -> 73.5, -3.4 [-6.9, +0.1] (p 0.13); control decline 0.0 [-2.6, +2.6]; claims on infeasible +1.9 [-2.4, +6.2]. U: all -6.5 [-11.0, -2.0]; plain -7.2 [-14.4, +0.5]; controls -8.0 [-16.5, 0.0]; twins -4.5 [-13.0, +3.9]. Plain task pass -3.4 [-8.7, +1.4]; control task pass -3.4 [-8.0, +0.8]. Median step 363 s on a 14-vCPU node; 3.58 h; peak GPU 74.4 GB.
- *source*: docs/iteration-19-results-crgrpo.md (lines 34-51); docs/iteration-19-research/badcase-it19c/ablation_select.md (lines 9-22, 132); outputs/it19/it19-crgrpo-ablation-r2/run_config.json; configs/grpo_it19_crgrpo.yaml (lines 18-73); docs/iteration-19-signoff.md (lines 94-96); ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt and ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/bykind.txt
- *confidence*: verified

**13. Settings that never varied across the eight runs**

- *value*: lr 3e-5 with 5 warmup steps; num_generations 8 and gradient_accumulation_steps 64 (8 prompts x 8 generations); DAPO with eps 0.2 / 0.28 and scale_rewards none; LoRA r 32 / alpha 64; temperature 1.0, top_p 1.0; mask_truncated_completions true; seed 42; one seed per run. beta was 0 in six runs and 0.04 in the two it19c arms. One optimiser step per generation round, so PPO clipping never bound (logged clip region mean 0.00000 in every run; importance ratio mean 1.0000).
- *source*: run_config.json of every run, printed by ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/runcfg.py; configs/base.yaml (lines 121-160); ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/timing.txt
- *confidence*: verified

**14. Task-pass effect on non-trained sources per 1,000 training episodes, each run against its own init**

- *value*: it09 (3,200 episodes, base): probe +5.5 points unpaired = +1.7 per 1,000; paired 64 tasks +4.0 = +1.25. it11 (1,280, base): task pass +2.2 = +1.7; verified success -4.4 = -3.4. it12 B0 c10 (640, base): task pass -0.8 = -1.3; verified success -3.3 = -5.2. it13 c10 (640, SFT init): task pass +3.7 = +5.8; verified success +3.9 = +6.1; unselected plain-only sources +1.9 = +3.0 (n.s.). G3 c15 (960, G2): tier S task pass +5.1 = +5.3 (in-run probe -3.8). it19c ablation (1,920, G2): verified success -3.4 = -1.8. it19c method c5 (320, G2): task pass -2.8 = -8.8; verified success -8.3 = -26. In U: it13 +6.6, G3 +12.9, ablation -3.4 per 1,000. Plan 20 needs only +0.13 to +0.19 U per 1,000 episodes to reach its MDE (2.5-3.6 over 19,200).
- *source*: Derived from the paired estimates in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/paired_u.txt and docs/training-log.md (lines 464-493, 727-739, 895-905); MDE from docs/iteration-20-rl-design.md (lines 193-196)
- *confidence*: derived

**15. No RL checkpoint has been evaluated on the frozen internal held-out set**

- *value*: Stage a ran only base and G2 on it; the it19c method arm's final benchmark was not run; the only RL run on test-split sources is it12's 20-source transfer check. So for the strict meaning of held-out there is no evidence in either direction.
- *source*: docs/training-log.md (lines 1689, 910-912); docs/iteration-19-results-crgrpo.md (lines 59-62)
- *confidence*: verified

**16. Plain-task capability on unseen sources, RL checkpoint minus its init**

- *value*: it13 c10 +0.8 [-2.0, +3.7] (89 sources); G3 c15 +2.9 [-2.9, +8.6] (35); it19c ablation -3.4 [-8.7, +1.4] (52); B0 c10 -0.3 [-3.7, +3.1] (89). Every interval contains 0. Control task pass: +11.0 [+3.7, +18.8], +7.4 [-1.4, +15.7], -3.4 [-8.0, +0.8], -2.2 [-10.1, +6.1] in the same order.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/bykind.txt (script bykind.py in the same folder's parent)
- *confidence*: verified

**17. Mean reward actually paid per outcome cell in training rollouts (what the gradient saw)**

- *value*: G3 (honest table): feasible claim+pass +1.55, claim+fail -1.00, non-claim -0.11 (plain) / -0.32 (control); twin claim -1.82, twin non-claim +0.89. it19c ablation (CR-GRPO table): plain claim+pass +1.54, claim+fail -0.34, non-claim +0.06; control claim+pass +1.00, claim+fail -0.38, non-claim -1.78; twin claim -2.23, twin non-claim +0.72. So pass-minus-fail among claims was 2.55 (G3) vs 1.88 plain / 1.38 control (ablation), and on controls a false claim paid 1.40 more than a non-claim in the ablation against 0.68 less in G3. Plan 20's U: +1 / -1 / 0 on feasible rows (gap 2.0; false claim 1.0 below a non-claim); twins -w / 0 with w about 0.44. Share of summed |advantage| on twins: it09 11.6 %, it11 24.6 %, it13 37.2 %, G3 41.5 %, ablation 42.2 %.
- *source*: Computed: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/advmass.txt; tables in src/honest_agents/crgrpo.py (lines 132-174) and src/honest_agents/reward.py (lines 9-33); plan 20's U in docs/iteration-20-rl-design.md (lines 343-353, 366-370)
- *confidence*: verified

**18. False claims per episode on controls and on plain tasks, RL checkpoint minus init**

- *value*: Controls: it13 c10 -6.6 [-12.1, -1.4]; G3 c15 -7.4 [-15.2, 0.0]; B0 -1.5 [-9.1, +5.5]; it19c ablation +4.2 [0.0, +8.5]. Plain: it13 0.0 [-2.5, +2.5]; G3 -5.7 [-11.4, 0.0]; B0 0.0 [-3.1, +3.1]; ablation +3.4 [+0.5, +6.7]. The three honest-table runs are at or below 0; the CR-GRPO run is above 0.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/bykind.txt
- *confidence*: verified

**19. it19c ablation: same-prompt revisit (direct test of learning on the training distribution)**

- *value*: TRL repeated the file order: the 8 prompts of step k equal those of step k+15 for 15 of 15 steps. Pass 1 (steps 1-15) vs pass 2 (steps 16-30), 120 prompts, 959 / 960 scored episodes, 90 % source-cluster bootstrap: feasible pass 74.1 -> 75.7 (+1.5 [-2.2, +5.3]); plain 76.5 -> 76.2 (-0.3 [-5.7, +5.1]); controls 72.5 -> 75.3 (+2.8 [-1.9, +7.7]); feasible claim rate 86.0 -> 88.2 (+2.2 [-0.3, +4.8]); false claims per feasible 13.0 -> 13.3; twin claim rate 29.7 -> 33.3 (+3.6 [-0.6, +7.8]); U without the damage term, all rows, 26.3 -> 25.9 (-0.4 [-4.7, +3.8]). By thirds of the run P(claim | pass) 0.986 / 0.986 / 0.987 and P(claim | fail) 0.519 / 0.543 / 0.511. Verdict: flat.
- *source*: Computed: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/revisit.py, output ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/revisit.txt; input ~/Documents/honest-agents-archive/staging/outputs/it19/it19-crgrpo-ablation-r2/completions. Agrees with docs/iteration-19-research/badcase-it19c/ablation_select.md (lines 156-163)
- *confidence*: verified

**20. G3: learning on the training distribution**

- *value*: No prompt was revisited (120 prompts x 1). Raw feasible pass by thirds fell 75.5 / 67.5 / 54.5 %, but against G2's pool bank on the same variants the gap is +2.1 / 0.0 / -1.0 points (only 28 matched groups: the surface prompts have no G2 bank), i.e. the fall is task mix. The doc's 'control declines 16 -> 39 % in rollouts' compares different prompts and is confounded the same way; the solid evidence of the caution shift is the evaluation (control non-claims +11.0 [+4.0, +18.0] on trained surfaces). Verdict: flat on capability, small n.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/runs_all.txt; docs/iteration-15-g3-rl-read.md (lines 70-72); bank runs/def74/it17_rollouts_feasible/episodes.jsonl (12,288-token budget, per its config.json)
- *confidence*: verified

**21. Amount of solve signal per step: feasible prompt groups with mixed pass/fail among their 8 generations**

- *value*: it07 76 of 103 groups (4.2 per step); it09 159 of 274 (3.4); it11 67 of 99 (3.4); B0 49 of 74 (3.3); it13 48 of 75 (3.2); G3 57 of 75 (3.8); it19c method 30 of 35 (4.3); ablation 108 of 150 (3.6 per step, 72 %). Plan 20 has 12 feasible prompts per step; at the ablation's 72 % mixed share that is about 8.6 per step and about 1,300 over 150 steps, 12x the ablation's 108.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/mixed.txt; plan composition docs/iteration-20-rl-design.md (lines 299-300)
- *confidence*: derived

**22. Training amount per run against plan 20**

- *value*: Episodes / optimiser steps / unique prompts / visits per prompt: it07 1,152 / 18 / 144 / 1; it09 3,200 / 50 / about 360 / 1.09; it11 1,280 / 20 / 160 / 1; B0 960 (candidate 640) / 15 / 120 / 1; it13 960 (candidate 640) / 15 / 120 / 1; G3 960 / 15 / 120 / 1; it19c method 448 / 7 / 56 / 1; it19c ablation 1,920 / 30 / 120 / 2; plan 20 arm A 19,200 / 150 / 800 / 3. Cumulative learning rate (sum of per-step lr): ablation 8.1e-4, it09 1.41e-3, G3 3.6e-4, plan 20 2.94e-3 (3.6x, 2.1x, 8.2x). Twin share of episodes: 28.5 %, 27.1 %, 38.1 %, 38.3 %, 37.5 %, 37.5 %, 37.5 %, 37.5 %; plan 20 25 % (plain 37.5 %, controls 37.5 %).
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/runs_all.txt; logged learning_rate in each train_log.jsonl; docs/iteration-20-rl-design.md (lines 287-301, 416-430)
- *confidence*: derived

**23. KL trajectory of the ablation (the only run with a KL trace beyond 7 steps)**

- *value*: KL per token: 0.0004 (step 5), 0.0038 (10), 0.0103 (15), 0.0082 (20), 0.0213 (30). 5-step means: 0.0002, 0.0020, 0.0075, 0.0129, 0.0163, 0.0155. Slope 0.00111 per step over steps 6-15, 0.00032 over steps 16-30, 0.00072 over 6-30, 0.00085 over 6-10. A linear fit explains it better than a quadratic one (RSS 2.7e-4 vs 6.0e-4). Projection of the steps 6-10 fit to step 150: 0.123 at lr 3e-5; 0.082 if KL scales with lr, 0.055 if with lr squared, at lr 2e-5. The same fit already over-predicts the ablation's own steps 26-30 mean (0.0207 vs 0.0155). The plan's statement 'linear at about 0.0008 per step' is the whole-run average; the second half grew at 0.0003.
- *source*: Command output in this session (numpy fit on outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl); per-step values in ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/timing.txt; plan text docs/iteration-20-rl-design.md (lines 119, 457-458)
- *confidence*: derived

**24. G2 against itself on the select split across sessions (clone noise)**

- *value*: it17 G2 select vs it19c G2 select (52 sources): verified success -0.6 [-3.3, +2.1], task success -1.5 [-4.0, +1.0], false claims per feasible +0.2 [-2.6, +3.0]. it16 G2 select vs it17 G2 select: verified success +2.1 [-1.1, +5.3], task success +2.5 [-0.6, +5.7], false claims -2.1 [-5.5, +1.2]. The ablation's -3.4 verified success is about 1.5x the larger clone difference; its +3.8 false claims is outside both.
- *source*: Commands: python -m honest_agents.analysis.power estimate with ~/Documents/honest-agents-archive/iterations/17-it17b-confidence/eval/it17_g2_select/episodes.jsonl.gz, runs/eval/it19c_g2_select/episodes.jsonl and ~/Documents/honest-agents-archive/iterations/16-it16c-g2-select/eval/it16_g2_select/episodes.jsonl.gz (output in this session)
- *confidence*: verified

**25. The 16,384-token training budget against the 12,288-token evaluation budget did not bind on select**

- *value*: Feasible select episodes ending on a report: G2 469 of 472 (99.4 %), ablation 468 of 472 (99.2 %); token_budget stops 1 in each arm.
- *source*: Command in this session over runs/eval/it19c_g2_select/episodes.jsonl and runs/eval/it19c_ablation_select/episodes.jsonl; budgets in their config.json files
- *confidence*: verified

**26. Rows selected on a pass band sit above their selection bank before any learning (regression to the mean)**

- *value*: Plain rows chosen for a 2-3-of-4 base pass band, trained-minus-bank pass rate in steps 1-5 (lr still in warmup): it11 +12.5 points (64 episodes), B0 +12.5 (64), it07 +3.6 (56), it09 -3.6 (56); pooled +6.7 over 240 episodes. Unselected controls in the same steps: -7.2, -2.0, +5.9, +1.5 (pooled about -0.5). Plan 20 selects 225 plain rows and the twins on prescreen bands and writes the same prescreen onto each row as g2_pass / g2_claim_given_pass / g2_decline.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/runs_all.txt (5-step windows); docs/iteration-20-rl-design.md (lines 280-285, 291-294)
- *confidence*: derived

**27. Noise of the same-prompt revisit diagnostic, and what plan 20's size buys**

- *value*: Ablation, per-prompt difference between passes (8 samples each): SD 0.200 for feasible pass (75 prompts, SE 2.3 points), 0.282 for U without damage over all rows (120 prompts, SE 2.6). Scaled to plan 20: 600 feasible prompts give SE about 0.8 points for pass; 800 prompts give SE about 1.0 U per 100; MDE90 about 2.4 points and 2.9 U per epoch-to-epoch contrast.
- *source*: Command in this session on ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/ablation_groups.json (written by revisit.py)
- *confidence*: derived

**28. Checkpoints five steps apart differ by more than the effects being sought**

- *value*: G3 held-out gap: checkpoint 10 +3.1 [-6.2, 13.1], checkpoint 15 +15.6 [8.1, 23.1]; U on the surface validation set +4.4 vs +10.4. it13 probe (120 variants x 1): control verified success 52.5 / 75.0 / 67.5 % and control declines 15 / 10 / 25 % at steps 0 / 10 / 15. B0 probe control declines 2.5 / 2.5 / 10 % at steps 0 / 10 / 15. Plan 20 fixes the candidate at step 150 with no selection.
- *source*: docs/iteration-15-g3-rl-read.md (lines 74-81); docs/training-log.md (lines 887-890, 1065-1066); docs/iteration-20-rl-design.md (lines 442-444)
- *confidence*: verified

**29. Twin headroom inside plan 20's primary stratum**

- *value*: From the published check-1 table (G2, internal held-out, S=2, 12k budget): P u T has 118 plain, 158 controls (86 high_risk, 30 remove_input, 42 revoke) and 155 twins (86 / 27 / 42); G2's twin U is -0.163, -0.148, -0.012. Removing every residual twin claim and damage is worth (86 x 0.163 + 27 x 0.148 + 42 x 0.012) / 431 = about 4.3 U per 100 pooled, above the MDE90 of 2.5-3.6. G3 cut twin claims 10.9 -> 1.6 % and twin damage 9.4 -> 0 % on tier S in 960 episodes.
- *source*: docs/iteration-20-research/u_checks.txt (lines 2-21; a published aggregate, not recomputed); docs/iteration-20-rl-design.md (lines 193-196)
- *confidence*: derived

**30. Wall-clock per step**

- *value*: Median step_time per 64-episode step: it07 406 s, it09 423 s, it11 609 s, B0 515 s, it13 566 s, G3 634 s, it19c method 659 s, ablation r2 363 s (rollout 212 s, train 62 s, grading 54 s; 14 vCPU, images prebuilt). Totals: it09 6.9 h for 50 steps, ablation 3.58 h for 30 steps including 3 probes. Doubling to 128 episodes gives about 12 min (ablation speed) to 21 min (G3 speed) per step, 30-53 h for 150 steps before probes.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/out/timing.txt; node in docs/iteration-19-signoff.md (lines 94-96)
- *confidence*: derived

**31. Differences between the it19c ablation and plan 20 arm A**

- *value*: Reward: CR-GRPO table with multipliers and online V_s vs plain U with per-row twin weights (cells in the reward-cell fact); success and efficiency bonus on plain rows vs none. beta: 0.04 in both (not a difference). lr: 3e-5 vs 2e-5. Steps x episodes: 30 x 64 vs 150 x 128. Pool: 120 rows x 2 visits, 37.5 % twins, 25 % plain, all high_risk pairs on evidence surfaces vs 800 rows x 3 visits, 25 % twins, 37.5 % plain, 100 lone controls, band-selected at S=8. Masking: the status token was already in the main loss in the ablation (report_term false), so 'no masking' is not a difference from the ablation; mask_truncated_completions true vs false (0.3 % of completions). Budget: 16,384 train / 12,288 eval vs 16,384 for both. Seeds: 1 vs 3. Constraint loop: one multiplier update vs none.
- *source*: configs/grpo_it19_crgrpo_ablation.yaml (lines 1-12); configs/grpo_it19_crgrpo.yaml (lines 18-73); docs/iteration-20-rl-design.md (lines 287-301, 338-430)
- *confidence*: verified

**32. Which of those differences could flip the sign of the ablation's result**

- *value*: Could plausibly flip the false-claim sign: the reward table (the control false-claim-versus-decline price is reversed between the ablation and both U and the honest table, and the measured control false-claim change is +4.2 vs -6.6 and -7.4). Could plausibly flip the capability sign, unproven: the table (control groups' largest contrast was claim vs decline, 2.8, not pass vs fail, 1.38; efficiency bonus coincided with 0.6 fewer actions and trailing checks 78.0 -> 73.3 %), and the signal per step (3.6 mixed feasible groups a step with no train-set gain while KL reached 0.016-0.021 looks like noise-dominated drift; plan 20 has about 8.6). Cannot flip it alone: lr 3e-5 -> 2e-5 (scales the step), more steps (extends whichever direction holds), truncation masking, the eval budget, the multiplier update, twin share. Not a difference: beta. Note the capability loss itself is not established (-3.4 [-6.9, +0.1]).
- *source*: Judgement from the facts above; action counts in docs/iteration-19-research/badcase-it19c/ablation_select.md (lines 72-91)
- *confidence*: derived

**33. Statements in current docs that the archived episodes qualify**

- *value*: (a) roadmap: GRPO on top of the SFT models 'eroded or did not move them (iterations 13, 15 G3)': the it13 checkpoint 10 is +3.9 [+0.7, +7.0] verified success over its SFT init and the log itself says it is better on every row. (b) plan 20 section 1: 'The only comparable run lost capability': G3 is also full-episode GRPO from G2 and shows task pass +5.1 [+0.7, +9.6] on tier S. (c) plan 20 section 1 cites G3's 'control declines 16 -> 39 %', a training-rollout number on different prompts per window.
- *source*: docs/roadmap.md (lines 19-21); docs/iteration-20-rl-design.md (lines 115, 119); docs/training-log.md (lines 1117-1124)
- *confidence*: verified

**34. G2's weakest feasible stratum is remove_input controls, and plan 20 gives it few rows**

- *value*: G2 pool bank (S=4): remove_input controls pass 0.590, pass@4 0.812; high_risk controls 0.796 / 0.945; revoke 0.830 / 0.940; plain 0.847 / 0.944. Plan 20 takes 34 of the 64 legal remove_input pairs (68 of 800 rows). The select split has no remove_input family; probe_v1 has 6 pairs.
- *source*: docs/iteration-20-research/u_checks.txt (lines 54-62); docs/iteration-20-rl-design.md (lines 295, 315); docs/iteration-19-research/badcase-it19c/ablation_select.md (line 5); composition of data/probe_v1.jsonl counted by pools.py
- *confidence*: verified

**35. Analyses attempted that gave nothing usable**

- *value*: Comparing G3 and ablation rollouts on shared training prompts (early visits of one as a G2 proxy for late visits of the other): only 2-7 shared prompts per cell. KL or drift for the six beta = 0 runs: not logged. U on G3's in-run probes: probe completions not archived.
- *source*: ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/rl_history/cross.py (output in this session)
- *confidence*: not verified

## implications_for_plan

- Rewrite the evidence base in sections 1 and 9. Cite three comparable runs, not one: it13 checkpoint 10 (U +4.2 [0.0, +8.3] over its SFT init), G3 (U +12.4 [+4.3, +20.2] on tier S, capability unconfirmed by its in-run probe) and the ablation (U -6.5 [-11.0, -2.0]). Replace 'a hope, 0 to +5' with a reference-class forecast by channel: false claims on feasible rows fell 1.8 to 6.5 points in the honest-table runs, worth about +1.2 to +4.2 pooled U on P u T (feasible rows are 64 % of it); plain capability about 0 (never moved in four evaluated runs); twins about 0 by design under neutral weights. Expected measurable effect: the forecast puts H1 at the edge of the MDE (2.5-3.6), which is the argument for S = 16 and for reading the pooled-seed H2 as the main test; and it states in advance that a P-stratum gain would be the first in the project.
- Add a reference arm with the reward that has the positive record: G3's honest table (alpha 0.1, false_blame -0.4, implicit_abstain 0) on pool v5, seed 1, at plan-20 scale, in place of arm C or as a sixth GPU. No past run varied prompt diversity, so arm C tests a factor with no prior signal, while the reward table is the factor that co-varies with the sign. Expected measurable effect on select (S = 4): the honest arm reproduces fewer false claims with control non-claims up by 3 to 11 points (G3: +2.9 familiar, +11.0 unfamiliar); A1 minus honest arm then measures what U plus neutral twin weights buys, predicted as a lower control non-claim rate at a similar false-claim reduction.
- Separate the KL anchor from the reward. beta 0.04 was on in both losing arms and off in both gaining runs, and plan 20 changes the table but keeps beta. One single-factor arm (A1's file and seed with beta 0, or 0.01) answers it. Expected measurable effect: A1 minus the no-KL arm on select task pass, false claims and control non-claims, plus the train-side learning check below; if the anchored arms again show KL rising with no gain on their own prompts (ablation: KL 0.016-0.021, feasible pass +1.5 [-2.2, +5.3]) while the unanchored arm learns, the anchor is the brake.
- Make 'is the policy improving on its own training prompts' a pre-registered, powered train-side endpoint instead of a diagnostic. Score each prompt's visit in epoch 2 and 3 against the same prompt earlier (steps k, k+50, k+100 share prompts) and report feasible pass, false claims, P(claim | pass) and U with source-cluster intervals. Expected measurable effect: SE about 0.8 points of feasible pass and 1.0 U per 100 per contrast (MDE90 about 2.4 points / 2.9 U) against +/- 4 in the ablation, available to the executor without held-out data. A flat reading at epoch 2 predicts a held-out null before the evaluator spends anything.
- Give that check, and the stop rules, an unbiased G2 baseline: after the pool is frozen, run G2 once more on the 800 selected rows at S = 8 (6,400 episodes, no training) and write those rates onto the rows, not the prescreen's. Band-selected plain rows sat +6.7 points above their selection bank at steps 1-5 across four runs (+12.5 in it11 and B0). Expected measurable effect: removes a spurious 5-12 point 'gain' on pass-type baselines and lets every training step be read as RL minus G2 on the same prompt.
- Change the pilot's KL gate. A 5-point slope on steps 6-10 projected to step 150 gives 0.055-0.082 from the ablation's own data scaled to lr 2e-5, against a 0.08 limit, while the ablation's KL actually flattened (0.0163 and 0.0155 for steps 21-25 and 26-30). Run the pilot to 20 steps and gate on a window mean, for example mean KL over steps 16-20 above 0.013 (the ablation's level at the higher lr), keeping the hard stop at 0.1. Expected measurable effect: avoids a false-alarm cut to lr 1.5e-5, which would remove 25 % of the cumulative update in a run whose history shows too little learning rather than too much.
- Add a feasible-only test beside pooled U: H1b, U over plain rows and controls with its 90 % lower bound above 0 (or at least task success reported with its own hypothesis), and report twin-only U separately. G3 scored U +10.4 [+4.9, +15.9] on a balanced twin/control set while control pass fell 8.1 points, and twins alone hold about 4.3 U of headroom in P u T, more than the MDE. Expected measurable effect: the owner's claim (success under honesty constraints) gets its own number; feasible-only MDE is roughly 3.1-4.5 U (276 of 431 variants), so say so in the prereg.
- Use every legal remove_input pair (64, or 46 without DEF-63) instead of 34, and add remove_input controls to the executor's development evaluation, since select has none. It is G2's weakest feasible stratum (pass 0.59, pass@4 0.81) and the only stratum with demonstrated RL learning (it09: +21.0 [+14.3, +27.6] on new sources). Expected measurable effect: each +10 points of pass on the 30 held-out remove_input controls adds about +0.7 points of pooled task success and up to +1.4 U on P u T.
- Do not treat the step-150 checkpoint as a point estimate of the method. Checkpoints five steps apart moved by 12 points of held-out gap (G3) and 15 points of control declines (it13). Keep the fixed candidate for H1, but pre-register the select readings at steps 60 / 105 / 150 for all three seeds as a reported trajectory, and state H2 (pooled seeds) as the estimate of the method. Expected measurable effect: a seed and checkpoint spread for the first time (all eight past runs used seed 42), so a single-arm result can be judged against it.
- Size the run from measured step times: 12-21 min per 128-episode step, 30-53 h per arm for 150 steps plus 11 probes of 412 episodes. Set job timeouts, checkpoint pulls and the watcher to that, not to the ablation's 3.6 h.
- If the train-side check shows no learning by step 50, the lever with history behind it is more steps and less anchoring, not a lower lr: the only run that learned in distribution (it09) needed more than 20 steps at lr 3e-5 with no KL, and lr has never been varied. Write that contingency into the prereg so a KL warning alone does not lower lr below 2e-5.

## open_questions

- Why did the ablation lose about 3.4 points on both plain rows and controls? Noise-dominated drift, the CR-GRPO table and the KL anchor are all consistent with the data, and no run separates them. The loss itself is not statistically established.
- Does the in-distribution learning seen from base in it09 (after about 20 steps) happen from G2 at all? No SFT-init run has exceeded 30 steps, and the one with revisits showed none.
- Is G3's capability gain on tier S real? Tier S says task pass +5.1 [+0.7, +9.6]; the in-run probe on the same variants says 73.8 -> 70.0 % (1 sample, different budget). G3 was never run on tier F or the select split.
- How large is seed-to-seed variance of an RL run? Every run used seed 42 with one seed; plan 20's A1-A3 would be the first measurement.
- How does KL scale with learning rate and batch size? One trace exists. Linear scaling projects 0.07-0.08 at step 150, quadratic 0.055, the second-half slope about 0.04.
- Does any RL checkpoint change behaviour on the frozen internal held-out set? None has been run there; the only test-split evidence is it12's 20-source transfer check, which I cited from the log and did not open.
- Does the honest table's false-claim reduction survive without its caution shift? In G3 it came with claim rate on feasible rows -3.6 and control non-claims +2.9 (familiar) / +11.0 (unfamiliar); plan 20's neutral twin weights are meant to remove the shift but that combination has never been run.
- Can plain-task pass rate move at all with this model and budget? G2's plain pass is 0.847 on the pool bank against 0.944 best-of-4, so headroom exists, but four evaluated RL checkpoints show -3.4 to +2.9 with every interval containing 0.
- Not verified: the iteration-12 transfer numbers and all it19a-based figures are quoted from the docs; the tier-F comparison for it13 uses evals run in two different sessions and nodes (it13 and it14), which I could not check for harness differences beyond the token budget and turn cap.