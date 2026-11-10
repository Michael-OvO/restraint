# research:pass-fail-anatomy


## question

What separates G2's passing samples from its failing samples of the SAME task, and therefore what would an outcome-driven policy gradient (plan 20's plain-U GRPO) actually reinforce: a transferable behaviour such as "verify, then fix", or task-specific luck?


## method

Data (train side only; every source checked against data/split_seeds.json, 0 rows from the test split; nothing held-out opened). Scored with the repo's own score_records / EpisodeRecord, never the stored reward dict.
- Train pool, G2: runs/def74/it17_rollouts_feasible/episodes.jsonl (1,860 feasible and control episodes, 465 variants x S=4, 205 sources).
- Select split, G2: runs/eval/it19c_g2_select/episodes.jsonl (544 feasible/control episodes, 136 variants x S=4), pooled where stated with the archive's it17_g2_select and it16_g2_select (S=2 each, same budget) to give 8 samples per variant.
- RL training groups of 8: the it19c full-episode GRPO ablation, ~/Documents/honest-agents-archive/staging/outputs/it19/it19-crgrpo-ablation-r2/completions (30 steps, 150 feasible/control groups) and the method arm's steps 1-6. "rl_early" = steps 1-5 of both (50 groups, policy close to G2); "rl_all" = all steps (185 groups).
- Its probe at steps 0/15/30: outputs/it19/it19-crgrpo-ablation-r2/probe (120 val-split tasks, 1 sample each).
Definitions. Mixed variant: at least one passing and one failing sample of the same variant. False claim: failed and reported complete. Non-claim fail: failed and reported incomplete, abstained, or gave no report. Budget used up: all 28 commands used, or a "budget exhausted" message, or the episode ended without a report.
Step 1, features (features.py): about 60 per episode from the command history (command, exit code, output) and the report: commands, tokens, writes (repo's mutation.is_mutating), commands before the first write, a check after the last edit, read-back of a written path after the last edit, edits after the first check, write-read-rewrite "fix cycles" on the same path, non-zero exit codes, error text after the last edit, timeouts, byte-exact viewers (od, cat -A, xxd), heredoc use, hand-typed versus tool-derived writes, repeats of a failed command, re-reading input files named in the instruction, budget used up, restore of a moved input (mv/cp X.suffix X), rm -r, report status, confidence, list lengths, evidence commands.
Step 2, statistics: within each mixed variant, mean over passing samples minus mean over failing samples; averaged over variants; 95% intervals from 2,000 bootstrap resamples of sources (stats1.py, stats3.py, fixcycle.py). Within-variant conditional logistic model (stats2.py, stats4.py, written in numpy: likelihood of the observed passing subset among all subsets of the same size; 26 process features, ridge 1; 5-fold cross-validation by source). Reported: out-of-fold within-variant AUROC (chance that a passing sample is scored above a failing sample of the same variant by a model that never saw that source; 0.5 = no information), with a source bootstrap, and McFadden pseudo-R2 against the equal-odds null. Repeated separately for false claims and non-claim fails, and with a 58-feature command-verb representation as a robustness check (verbs.py).
Step 3, reading: 28 mixed pairs (one pass, one fail of the same variant), seeded sample, one per source: 16 train pool, 6 select, 6 RL steps 1-5 (render_pairs.py; rendered files in pairs/). Each pair: instruction, every command with exit code and output of both samples, both reports, and the failing test messages. Long commands and outputs were clipped (300 / 200 characters); pairs 04, 10 and 14 were read as command lists plus test failures only. Labels in pair_labels.json.
Extra: failure composition by claim type, budget and failing-test category (failreasons.py, levers.py); where plain U's advantage mass falls (advmass.py); whether G2's own evidence commands tell true from false claims (evidence.py); the ablation's change on the SAME training variants between their two visits 15 steps apart (revisit.py) and on the never-trained probe (probe_drift.py).
All scripts and outputs are under ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/. Run from the repo root with: env -u VIRTUAL_ENV uv run --frozen python <script>. Order: build_table.py, then the stats scripts.


## headline

It is not "verify, then fix". G2's failing samples are of two kinds. About one third of fails in mixed groups (95 of 295 on the train pool) are give-ups: the 28-command budget ran out in a debugging loop, or a control was declined; these are easy to tell from passes of the same task (within-variant AUROC 0.88 [0.78, 0.95], cross-validated by source). The other two thirds are false claims, which carry about 80% of plain U's negative advantage mass, and they cannot be told from passes by any process feature (AUROC 0.55 [0.49, 0.61]): a check after the last edit was run in 94.3% of passes and 93.9% of false claims, and write-read-rewrite fix cycles occur in 29.8% and 29.3%. In 28 pairs read in full, "verified against the spec and fixed" was the difference in 3 (11%); in 10 of the 18 false claims the defect was on the failing sample's own screen and it claimed anyway. So pass/fail RL on whole episodes would reinforce three things: trained-family conventions on controls (move the input back: 77% of passes against 39% of fails; do the requested cleanup: 100% against 60%), shorter episodes that do not loop, and per-task content that does not carry to other sources. The one comparable run shows the size: after an update on each task, pass rate on the same 75 training variants moved +1.5 points [-3.0, +6.1] and 0.0 [-11.4, +11.3] on the unseen probe.


## results


**1. How much pass/fail contrast exists inside a group. At S=4 only about a third of G2's feasible/control variants have both outcomes; half always pass. Groups of 8 on the it19c training pool (which was chosen for mid pass rates) are mostly mixed. About half of all pass/fail variance on the train pool is sample-to-sample inside a variant; the rest is fixed by the source.**

- *numbers*: Train pool (465 variants, 205 sources): mixed 35.9% [31.1, 40.6], all-pass 52.5%, all-fail 11.6%. Select (136 variants, 60 sources): mixed 25.0% [16.8, 34.1]. RL groups of 8: mixed 86% [75, 96] at steps 1-5 (50 groups), 74.6% [65.1, 83.2] over 30 steps (185 groups). Variance of pass, train pool: between sources 38%, between variants of one source 12%, within variant 49% [42, 57]; select: 67% / 0% / 33% [21, 47]. Share of within-group U variance that is pass versus fail: 0.90 (pool), 0.87 (select), 0.89-0.92 (RL groups).
- *source*: stats1.py (output stats1_full.out, structure blocks); levers.py item 5 (levers.out); failtypes.py
- *confidence*: verified

**2. Generic process features explain little of pass versus fail inside a variant. The features most often named as good practice barely differ. A model fitted on other sources ranks a passing sample above a failing one of the same variant 70% of the time; about 87% of the within-variant deviance is unexplained.**

- *numbers*: Train pool, 167 mixed variants, 116 sources, 373 passing and 295 failing episodes, pass minus fail [95% by source]: check after last edit 93.9% vs 85.1%, +8.8 [3.7, 13.8]; read-back of a written path after last edit 91.8% vs 80.0%, +11.8 [4.9, 18.8]; non-zero exit codes 1.18 vs 1.94, -0.76 [-1.39, -0.25]; commands 16.8 vs 17.6, -0.8 [-1.7, +0.2]; completion tokens 2,202 vs 2,468, -265 [-451, -96]; commands before first write 3.34 vs 3.35; re-read of input files after first write 72.6% vs 71.7%, +0.9 [-4.4, +6.5]; byte-exact viewer used 57.2% vs 58.4%, -1.2 [-8.3, +6.4]; error text after the last edit 30.0% vs 28.0% (n.s.); timeouts 0.012 vs 0.038 (n.s.); reported confidence 0.998 vs 0.991. Conditional logit, 26 process features, out-of-fold: AUROC 0.70 [0.64, 0.74], pseudo-R2 0.13 (pool); 0.72 [0.67, 0.76], pseudo-R2 0.13 (RL groups, 138 mixed); 0.53 [0.46, 0.61] on select (53 mixed variants, too few to fit 26 features). Adding the report fields (claimed, list lengths): 0.71 (pool).
- *source*: stats1_full.out; stats2.out; stats4.out
- *confidence*: verified

**3. The 0.70 comes almost entirely from give-up fails. Passing samples and falsely-claimed samples of the same task look the same in process terms; passing samples and non-claim fails look very different. This holds in the train pool, the select split and the RL groups, and with a richer 58-feature command-verb representation.**

- *numbers*: Pass vs FALSE CLAIM (pool, 132 variants, 287 pass / 200 fail episodes): AUROC 0.55 [0.49, 0.61], pseudo-R2 -0.06; check after last edit 94.3% vs 93.9%, +0.4 [-4.1, +5.2]; read-back 92.4% vs 91.3%; budget used up 9.0% vs 12.1%, -3.2 [-10.2, +3.5]; commands 16.4 vs 16.5; byte-exact viewer 56.6% vs 67.2%, -10.6 [-17.5, -3.7] (used MORE in false claims). RL groups (89 variants): 0.575 [0.51, 0.63]. Plain tasks only: 0.61 [0.51, 0.71] (pool, 53 variants), 0.51 [0.46, 0.56] (RL, 48). Verb features: 0.52-0.55 (pool), 0.53-0.57 (RL). Pass vs NON-CLAIM FAIL (pool, 65 variants, 130 / 95): AUROC 0.88 [0.78, 0.95], pseudo-R2 0.56; budget features alone 0.82 [0.75, 0.90]; budget used up 15.6% vs 56.9%, -41.3 [-54.7, -27.7]; non-zero exit codes 1.6 vs 4.1; check after last edit 95.1% vs 63.1%; rm -r run 36.9% vs 5.1%. RL groups (99 variants): 0.85 [0.80, 0.89].
- *source*: stats3.out; stats4.out; verbs.out
- *confidence*: verified

**4. "Verify, then fix", measured directly as a path that is written, read back and written again, is not associated with passing. Episodes with more fix cycles fail more. An outcome gradient would push this behaviour slightly down, not up.**

- *numbers*: At least one fix cycle: 31.1% of passes vs 33.2% of fails, -2.1 [-9.7, +5.1] (pool, 167 mixed variants); pass vs false claim 29.8% vs 29.3%, +0.5 [-7.7, +8.3]; RL groups (108 mixed, ablation) 36.8% vs 38.8%, -2.0 [-10.3, +6.2]. Number of fix cycles 0.60 vs 0.82, -0.23 [-0.42, -0.05]. Across all 1,860 pool episodes: pass rate 59.1% with a fix cycle, 76.7% without.
- *source*: fixcycle.py (fixcycle.out)
- *confidence*: verified

**5. Reading 28 mixed pairs: the concrete difference between the pass and the fail of the same task, by category.**

- *numbers*: (a) both had the same defect, only the pass checked it against the spec and fixed it: 3/28 = 11% [4, 27] (sha256 file format; pip into the venv; comma-separated vmstat rows). (b) different reading of an ambiguous, inconsistent or unstated detail: 5/28 = 18% [8, 36] (trailing-newline wording twice; a rule for unknown groups; a temp file the hidden test forbids but the instruction allows; a spec that says 9 lines and shows 8, where the fail counted, "fixed" it and lost). (c) ran out of commands or tokens: 7/28 = 25% [13, 43]; in 3 of the 7 the pass typed the answer by hand or skipped a stated process rule ("do not hard-code", "parallel") while the fail tried to follow it and looped; in 3 a wrong flag or tool syntax was retried for 10-25 commands. (d) tool or format slip in the write: 5/28 = 18% [8, 36] (literal backslash-n, tab from printf, uniq -c padding, jq object instead of array, an unwanted trailing newline). (e) other: 8/28 = 29%: a clearly stated item missed in the first write 3 (sort order twice, a header line); control handling 5 (remove_input control: input moved back or not, 3; high_risk control: cleanup done or declined, 2). Cross-cutting: in 8/28 (10/28 counting two wrong orders) the failing sample's own read-back showed the defect and it still claimed complete; that is 10 of the 18 false claims read, 56% [34, 75]. 8/28 passing samples fixed a content defect after a read-back. By bank the mix is similar (pool 16, select 6, RL 6).
- *source*: pairs/pair_00.txt to pair_27.txt and pair_labels.json in the scratch folder; sample drawn by render_pairs.py (seed 2020)
- *confidence*: verified

**6. On controls the strongest and most consistent pass/fail separator is a trained-family convention, not a general skill.**

- *numbers*: remove_input controls: input moved or copied back to its stated path in 77.1% of passes vs 38.6% of fails, +38.6 [19.5, 56.2] (pool, 35 mixed variants); RL groups 91.3% vs 20.4%, +71.0 [44.6, 92.9] (16 groups). high_risk controls: rm -r run in 100% of passes vs 60.2% of fails, +39.8 [28.1, 51.8] (pool, 57 variants); RL groups 100% vs 30.3%, +69.7 [56.6, 81.5] (58 groups). Cost on the train pool: "input not back at its path" 36.4 U per 100 remove_input control episodes [25.6, 47.7] (n=308, pass rate 52.3%, U 22.1); "requested cleanup not done" 11.2 U per 100 high_risk control episodes [7.7, 15.0] (n=520).
- *source*: stats3.out (control blocks); levers.out item 1
- *confidence*: verified

**7. Where G2 loses U on feasible and control tasks, and how much of it has a generic signature. Most of the loss is false claims made within budget, the kind with no process signal. Fails are near-misses concentrated on one trap per task.**

- *numbers*: Train pool: U 53.7 per 100; lost 36.6 to false claims, 9.2 to non-claim fails, 0.6 to unclaimed passes. Of 511 fails: false claim with budget left 57.3% [49.3, 64.7] (15.8 per 100 episodes); non-claim with budget used up 22.1% [15.5, 28.9] (6.1 per 100); non-claim with budget left 11.4% (3.1); false claim with budget used up 9.2% (2.5). Select (167 fails of 544): 70.7% [57.7, 81.0] / 20.4% [11.2, 31.7] / 4.8% / 4.2%; U 46.3. Plain tasks, pool (n=820, U 62.3): false claims within budget cost 27.8 U per 100, budget-used-up non-claim fails 5.2 U [3.2, 7.7]. False claims by failing-test type (pool, n=340): content or value mismatch 57.4%, missing artifact 12.9%, newline/whitespace 10.6%, remove_input input not restored 9.4%, permissions 5.6%. Failing episodes pass 64% of their tests on average; 48% pass at least three quarters; 5.3% pass none. In 81% of mixed variants with two or more fails (71 of 88 pool; 13 of 16 select) all fails share a failing test; in 51% the failing-test sets are identical.
- *source*: failreasons.out; levers.out items 1, 3
- *confidence*: verified

**8. Under plain U with a group-mean baseline, most of the negative advantage lands on the samples that cannot be told from passes.**

- *numbers*: Share of absolute advantage mass on feasible/control groups: train pool (S=4): passes 45.0%, false claims 43.5% [41.3, 45.4], budget-used-up non-claim fails 6.0%, other non-claim fails 4.6%; so 80% of the negative mass is on false claims. RL groups of 8: false claims 34.9% [30.8, 38.3], non-claim fails 13.5%; 72% of the negative mass. Group types in RL groups: pass + false claim only 21%, pass + non-claim only 26.5%, all three 27%, all-pass 25%.
- *source*: advmass.py (advmass.out)
- *confidence*: verified

**9. The budget is the one generic lever. Using the whole command budget goes with failure, and failing samples that run out are stuck repeating a failed command. The policy is told how much budget is left only when 5 or fewer commands remain.**

- *numbers*: Pool: 10.5% of episodes (196) use all 28 commands; they pass 24.0% (false claim 24.0%, non-claim 52.0%); pass rate by commands used: 0-7 87.6%, 8-12 84.3%, 13-17 75.9%, 18-22 63.0%, 23-27 62.8%. Within mixed variants: budget used up 9.6% of passes vs 24.3% of fails, -14.7 [-22.5, -7.4] (pool); RL groups 7.5% vs 28.8%, -21.3 [-29.0, -14.3]. Among the 113 budget-used-up non-claim fails: 54% retried a failed command prefix 3 or more times (passes 9%); 33% had 3 or more failing commands in a row (passes 1%). On tasks whose instruction forbids hard-coding (RL, 24 mixed groups, 8 sources): budget used up 7.1% of passes vs 53.5% of fails, -46.4 [-65.3, -27.0]; hand-typed share of writes 41.4% vs 34.1%, +7.3 [2.7, 13.0].
- *source*: stats3.out; levers.out item 2; table.jsonl tabulation by commands; src/honest_agents/harness.py (lines 109-110 and 217)
- *confidence*: verified

**10. A required verification step would not help as things stand: G2 already verifies almost every time, and its checks do not discriminate. The evidence commands in its report are never shown back to it and never rewarded.**

- *numbers*: Check after the last edit in 94% of both passes and false claims (above). Evidence commands on "complete" claims (pool, 5,868 commands): 81.1% display-only (cat, ls, wc, od), 11.9% able to fail on content. Claims with a failing evidence command: true claims 10.6% (142/1,338), false claims 15.0% (51/340). Among claims that carry an assertion-style command, it failed in 13.6% of true claims (40/295) and 14.3% of false claims (10/70). Select: 17.6% (12/68) vs 3.4% (1/29).
- *source*: evidence.py (evidence.out); levers.out item 4; src/honest_agents/grading.py (lines 187-191 and 1160-1162); src/honest_agents/reward.py (line 362)
- *confidence*: verified

**11. What the only comparable run actually changed. Every training variant of the it19c full-episode GRPO ablation (lr 3e-5, 8 prompts x 8 generations, 30 steps) was visited twice, 15 steps apart. Pass rate on the SAME variants did not move measurably, nor did any pass-associated behaviour on the unseen probe.**

- *numbers*: Same 75 feasible/control variants (65 sources, 8 samples per visit): pass 74.1% to 75.7%, +1.5 points [-3.0, +6.1]; U +1.8 per 100 [-6.3, +9.5]; budget used up 10.9% to 10.0%, -0.9 [-3.5, +1.8]; commands 16.3 to 16.1. Variants mixed at first visit (55): 64.7% to 68.2%; all-pass at first visit (20): 100% to 96.3% (regression to the mean in both directions). Probe, 80 feasible/control tasks, step 30 minus step 0: pass 71.3% to 71.3%, 0.0 [-11.4, +11.3]; check after last edit 91.2% to 91.2%; budget used up 10.0% to 10.0%; heredoc use 46.3% to 55.0%, +8.7 [-0.0, +18.3]; rm -r on controls 55.0% to 45.0%, -10.0 [-19.5, -2.4] (more control declines, the wrong direction).
- *source*: revisit.py (revisit.out); probe_drift.py (probe_drift.out); run settings in outputs/it19/it19-crgrpo-ablation-r2/run_config.json
- *confidence*: verified

**12. A denser outcome signal exists but is modest. Using the fraction of hidden tests passed instead of pass/fail adds contrast mainly inside all-fail groups.**

- *numbers*: Train pool, S=4: groups with any within-group spread in pass/fail 35.9%, in U 42.4%, in test pass fraction 43.4%. Of the 54 all-fail groups, 65% have spread in test pass fraction and 48% have spread in U (claim against non-claim). Select: 25.0% / 33.1% / 33.1%; 39% of 28 all-fail groups.
- *source*: levers.out item 6
- *confidence*: verified

**13. Implied size of a capability gain on unseen sources under plan 20 as written. The behaviours with a consistent within-group signal are family conventions (trained families only) and avoiding budget exhaustion. For plain tasks on unseen sources the generic ceiling is small, and the comparable run captured none of it. Best-of-k headroom is mostly sample luck on a per-task trap and is not reachable by learning from other tasks.**

- *numbers*: Ceiling on plain tasks if every budget-used-up non-claim fail became a claimed pass: +5.2 U per 100 plain episodes [3.2, 7.7] (pool); +6.7 [2.5, 11.7] (select). The ablation moved budget exhaustion by -0.9 points [-3.5, +1.8] on its own training tasks and 0.0 on the probe. False claims within budget (27.8 U per 100 plain episodes on the pool) have out-of-source AUROC 0.55 [0.49, 0.61]. Plan 20 needs a held-out gain of 2.5-3.6 U to be detected (design section 3.4).
- *source*: levers.out item 1; revisit.out; stats4.out; docs/iteration-20-rl-design.md (lines 193-200 and 496-502)
- *confidence*: derived

## design_recommendations


**1. Replace arm C (system-prompt paraphrases) with a decision-point arm: half of each step's prompts are shared-prefix groups. Take an episode from the prescreen, replay its commands in a fresh sandbox up to and including the first read-back after its last write (analysis.replay_episodes already replays prefixes with an exit-code fidelity check), and sample the 8 generations as continuations from that state. Use prefixes from false-claim episodes and from passing episodes 1:1, so the right continuation is "fix or downgrade the claim" in one and "claim" in the other. Reward stays plain U at the end; the prefix is context, every generated token is trained and under the KL. Refresh prefixes from the current policy every 50 steps. Before any training, run a no-gradient pilot: 60 false-claim prefixes and 60 passing prefixes x 8 continuations with G2.**

- *rationale*: False claims within budget are 57-71% of all fails and carry 72-80% of plain U's negative advantage mass, but whole-episode groups give no transferable contrast for them (within-variant AUROC 0.55 [0.49, 0.61]; fix cycles 29.8% vs 29.3%). In 10 of 18 false claims read, the defect was on the failing sample's own screen at the read-back. A shared prefix removes the per-task content variance and puts the whole group contrast on the one decision that is general: act on what the read-back shows. G2 does sometimes act on it (8 of 28 passing samples fixed a defect after reading it back), so continuations should be mixed. This is the RL counterpart of the decision-point SFT that produced G2.
- *expected_measurable_effect*: Pilot (about 960 episodes, no training): share of false-claim-prefix groups with both outcomes, and P(claim | defect on screen) against P(claim | clean state). Go if at least 40% of groups are mixed. In training: that claim gap on select-split prefixes should widen by 15 points or more by step 60. If RL converts 15% of the false claims whose defect is visible (56% of 15.8 per 100 episodes), that is 1.3 fewer false claims per 100 feasible/control episodes: +1.3 U if they become honest non-claims, +2.7 U if they become passes. Read on held-out as false claims per feasible episode and U on P and T.
- *confidence*: medium

**2. Pre-register a same-row gain gate. Plan 20 already repeats the same prompts at steps k, k+50 and k+100. At steps 100 and 150 compute, on feasible and control rows, pass rate and U of epoch 2 (and 3) minus epoch 1 for the same rows, with a 90% interval by source. If the upper bound of same-row U gain is below 2.5 per 100 at step 100, stop the arm and do not spend the S=16 held-out evaluation on it. Report the gain split into rows whose first-epoch fails were false claims and rows whose fails were non-claims.**

- *rationale*: A gain on unseen sources cannot exceed the gain on the rows that were trained on. In the it19c ablation the same 75 variants moved +1.5 pass points [-3.0, +6.1] and +1.8 U [-6.3, +9.5] after an update on each; the probe moved 0.0. Plan 20 has a lower learning rate and three visits, so this is the cheapest early reading of whether anything is being learned at all.
- *expected_measurable_effect*: With about 600 feasible/control rows x 8 samples per epoch and the per-row spread seen in the ablation (0.20 for pass, 0.35 for U), the standard error is about 0.8 pass points and 1.4 U per 100, against 2.3 and 4.0 in the ablation. A same-row gain of 3 pass points or more is detectable at step 100; a null there predicts a held-out null at no evaluation cost.
- *confidence*: high

**3. Treat the budget as a design variable before training. (1) Measurement only: run G2 on the select split (S=4) with the command cap at 40 instead of 28 and count how many budget-used-up fails convert. (2) Show the remaining budget on every tool output, not only when 5 or fewer commands remain, and add a one-line notice when a command with the same first 30 characters has failed twice. Apply the same harness to every arm, G2 and base. (3) In the plain band of pool v5, prefer rows whose prescreen fails are non-claim or budget fails over rows whose fails are false claims.**

- *rationale*: Budget exhaustion is the only generic behaviour with a strong within-group signal (AUROC 0.82-0.88 against non-claim fails). 10.5% of pool episodes use all 28 commands and pass 24%; 54% of the budget-used-up non-claim fails repeat a failed command three or more times (passes 9%). The policy cannot see the budget until the last 5 commands (harness.py lines 109-110, 217). If a cap of 40 rescues most of these episodes, "be faster" is fitting an arbitrary cap and will not carry to TB2's 50 turns; if it does not, the loops are a real capability failure that RL can be pointed at.
- *expected_measurable_effect*: Budget-used-up non-claim fails are 6.1 per 100 feasible/control episodes on the pool (6.2 on select; 5.2 U per 100 plain episodes [3.2, 7.7]). Halving them by conversion to passes is +3 U. Track on the probe: budget-used-up rate (G2 10%), share of episodes with 3 or more retries of a failed command, pass rate on plain rows.
- *confidence*: medium

**4. Do not add a required verification step, a verification bonus, or an evidence gate in this round. If verification is to matter, first make it visible to the policy: run the report's evidence commands at submit time and return their exit codes once before the report is final, and test that as a separate arm only after an offline check shows the commands can separate true from false claims.**

- *rationale*: G2 already runs a check after its last edit in 94.3% of passing and 93.9% of falsely-claimed episodes and uses byte-exact viewers more often in false claims (67.2% vs 56.6%). Its evidence commands are 81% display-only and fail as often on true claims (10.6%) as on false ones (15.0%); assertion-style ones fail in 13.6% vs 14.3%. Evidence is replayed only at evaluation and never shown to the agent (grading.py lines 187-191, 1160-1162). RL selects among behaviours that already vary with the outcome; this one does not, so rewarding or requiring it would add ritual, not correctness.
- *expected_measurable_effect*: Avoids a null arm. The offline go/no-go for any later evidence arm: on a fresh G2 bank, assertion-style evidence must fail at least 3 times as often on false claims as on true claims; today the ratio is 1.05 (pool).
- *confidence*: high

**5. Report the primary result per stratum and per family, and pre-register where the gain is expected: T controls (remove_input and high_risk) through family conventions, P through fewer budget fails. Add pre-registered diagnostics on the probe: restore rate on remove_input controls, cleanup rate on high_risk controls, budget-used-up rate, fix-cycle rate, and hand-typed share of writes on rows whose instruction forbids hard-coding. Raise remove_input from 34 pairs if the held-out T stratum weights it more than 8.5% of rows, and review the remove_input controls whose instruction says not to move or rename the input.**

- *rationale*: The largest consistent pass/fail separators are family conventions: input moved back in 77% of passes vs 39% of fails (36.4 U per 100 remove_input control episodes at stake), cleanup done in 100% vs 60% (11.2 U per 100 high_risk control episodes). A pooled gain driven by these is real but is not general capability, and it should not be read as such. The outcome gradient also favours shorter episodes and, where a task forbids hard-coding, hand-typed answers (passes use up the budget 7% vs 54% of fails there; hand-typed share +7.3 points [2.7, 13.0]); 3 of 7 budget pairs read were won by skipping a stated rule. Fix cycles go with failing (59% vs 77% pass), so the gradient pushes them down.
- *expected_measurable_effect*: If restore rate on remove_input controls rises from about 55% to 80%, that is roughly +15 U per 100 remove_input control episodes; cleanup on high_risk controls from 89% to 95% is roughly +4 U per 100. Shortcut flags: hand-typed share on no-hard-code rows up by more than 5 points, or fix-cycle rate (G2 24% of episodes) down by more than 5 points, at any probe.
- *confidence*: medium

**6. Screen the mid-band pool rows for coin-flip traps before freezing pool v5. From the S=8 prescreen, flag every band row where at least 75% of failing samples fail the same single test, read the instruction against that test, and drop rows where the requirement is ambiguous, contradictory or absent from the instruction. Keep rows where the trap is stated clearly.**

- *rationale*: The band [0.4, 0.875] selects for tasks decided by one trap: in 81% of mixed variants all fails share a failing test. In the pairs read, 5 of 28 (18% [8, 36]) were decided by an ambiguous or unstated detail (trailing-newline wording, a hidden no-extra-files test, a spec that says 9 lines and shows 8). These rows give gradient that is noise at best, and in one case punished a sample for checking and fixing.
- *expected_measurable_effect*: Roughly one in six band rows dropped or replaced (to be counted); the share of mixed groups whose contrast is a defensible requirement rises accordingly. Check afterwards by re-reading 25 pairs from pilot steps 1-10: category (b) should fall below 10%.
- *confidence*: medium

**7. If a denser reward is wanted, test it as its own arm and keep it off the claim: add a small term on the fraction of hidden tests passed that does not depend on the report (for example 0.2 x pass fraction), leaving the claim term of U unchanged.**

- *rationale*: Failing episodes are near-misses (64% of tests passed on average; 48% pass three quarters or more), and 65% of the pool's all-fail groups have spread in test pass fraction but none in pass/fail. Tying partial credit to the claim would pay false claims that pass more than half the tests, so it must stay claim-independent. Overall coverage rises only from 42.4% to 43.4% of S=4 groups, so the expected effect is small.
- *expected_measurable_effect*: Zero-variance group share on training batches down by about the share of all-fail groups with test spread (about 7.5% of S=4 pool groups); same-row pass gain on rows that were all-fail in the prescreen, read against arm A.
- *confidence*: low

## caveats

- Features are built from command text with regular expressions (the repo's mutation classifier counts any file redirect as a write). They measure whether a check was run, not whether it was the right check; content-level correctness is not captured, which is the point of the finding but also its limit.
- The select and RL banks have no assistant prose locally; only commands, outputs and reports were used there. Transcripts exist for the train pool but were not needed for the features.
- The RL groups come from the it19c runs, whose reward was CR-GRPO without the report term, not plain U, and whose policy drifted over 30 steps (KL about 0.021 at step 30). Steps 1-5 are close to G2; results on all 30 steps agree with them. The revisit and probe results describe that run, at lr 3e-5 and 8 prompts per step, not plan 20.
- The pair reading is 28 pairs. Category shares have wide intervals (for example 11% [4, 27] for verify-and-fix). Long commands and outputs were clipped to 300 and 200 characters; pairs 04, 10 and 14 were read as command lists plus test failures only. Labels are one reader's judgement.
- "Budget used up" on the banks is inferred from 28 commands used or an episode that ended without a report; the banks' stored outputs do not include the harness's budget lines, the RL completions do.
- Conditional-logit intervals come from one 5-fold split with a source bootstrap of out-of-fold scores; three repeated splits gave means within 0.03 of the reported values. On the select split (53 mixed variants) the 26-feature model is too large to say anything.
- The U-point effects in the recommendations are ceilings or scenario arithmetic from train-side rates, not predictions; the capture fractions are assumptions. The continuation pass rate from a read-back state (the decision-point pilot) has not been measured: no GPU was used.
- Held-out composition was not opened, so the weight of each family in the held-out T stratum, and therefore the pooled effect of family conventions, is not known here.
- The train pool bank and the select banks were run at 12,288 tokens; plan 20 pins 16,384. Token truncation was rare (14 of 1,860 pool episodes), the 28-command cap is the same.

## artifacts

- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/load.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/features.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/build_table.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/table.jsonl
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats1.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats1_full.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats2.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats2.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats3.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats3.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats4.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/stats4.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/verbs.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/verbs.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/fixcycle.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/fixcycle.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/failreasons.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/failreasons.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/levers.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/levers.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/advmass.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/advmass.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/evidence.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/evidence.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/revisit.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/revisit.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/probe_drift.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/probe_drift.out
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/render_pairs.py
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/pairs
- ~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/pair_labels.json