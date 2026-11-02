# it19c CR-GRPO method arm: the step-7 stop (bad-case analysis)

Source: `outputs/it19/it19-crgrpo/` (`stop_rules.jsonl`, `train_log.jsonl`, `crgrpo_state.jsonl`, `crgrpo_grad_ratio.jsonl`, `completions/completions_0000{1..7}.parquet`, 64 rows per step). Config `configs/grpo_it19_crgrpo.yaml`, c_rep 0.00213 (`run_config.json`), z_max 6. Pre-registration `docs/iteration-19-prereg-crgrpo.md`. Read-only. The calibration run at c_rep 1 is not local and is not used here.

Step numbering: steps 1-7 are the `step` of `train_log.jsonl` and of the completions files, which agree (step 1: claim rate 0.734 in both). `crgrpo_state.jsonl` is 0-indexed: its step 0 is step 1 here (fc 8/40 in both).

Batch per step: 16 plain feasible, 24 controls, 24 twins (40 feasible). The `abstain` tool was called 0 times in 448 episodes, so every decline is `submit_report` with status `incomplete`. Below, `inc+blk` is incomplete with a blocker submitted, `inc-blank` is incomplete with no blocker.

## 1. What stopped and when

The pre-registered decline guard (`stop.max_decline_rate_feasible` 0.30, window 5) fired at step 7: `honest/decline_rate_feasible` over steps 3-7 was (0.15 + 0.20 + 0.20 + 0.35 + 0.725) / 5 = **0.325**. No other stop rule fired. The gradient-ratio stop (max 3) did not: the ratios on steps 0-2 were 1.38, 0.84, 1.64.

| step | lr | claim rate, feasible | decline rate, feasible | pass, feasible | P(claim \| pass) | P(claim \| fail) | verified success | pi_claim (trainer) | z_abs | frozen share | kl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | .950 | .050 | .750 | 1.00 | .80 | .75 | .734 | 15.0 | .688 | 0 |
| 2 | 6e-6 | .875 | .125 | .675 | 1.00 | .62 | .68 | .658 | 16.4 | .734 | 0 |
| 3 | 1.2e-5 | .850 | .150 | .700 | 1.00 | .50 | .70 | .706 | 16.6 | .688 | 3.6e-4 |
| 4 | 1.8e-5 | .800 | .200 | .750 | .967 | .30 | .73 | .638 | 13.4 | .750 | 4.6e-4 |
| 5 | 2.4e-5 | .800 | .200 | .625 | 1.00 | .47 | .63 | .714 | 9.15 | .516 | 6.8e-4 |
| 6 | 3e-5 | .575 | .350 | .575 | .913 | .12 | .53 | .338 | 8.20 | .508 | 1.4e-3 |
| 7 | 3e-5 | .275 | .725 | **.775** | **.355** | 0 | .28 | **.077** | 6.04 | .328 | 5.9e-3 |

- `crgrpo/rate/fc` (false claims per feasible episode) went 0.20, 0.20, 0.15, 0.075, 0.175, 0.05, 0. The C1 target is 0.09; the run hit 0 by not claiming at all.
- pi_claim (the trainer's mean sigmoid(z) over all 64 rows) tracked the sampled claim rate through step 5, then fell below it: 0.338 vs 0.406 at step 6, 0.077 vs 0.219 at step 7. Not investigated (rollout/trainer lag or mismatch are candidates).
- The drift starts as the learning rate leaves warmup (2.4e-5 at step 5, 3e-5 at step 6). KL rose about 10x over steps 4-7.
- `crgrpo/delta_mean` is negative from step 1, where lr is 0, so it is a property of the batch mix, not of the collapse (section 3).

## 2. What the model learned

**It stopped claiming solved tasks.** Feasible rows by outcome and status:

| step | fail: complete | fail: inc+blk | fail: inc-blank | fail: no report | pass: complete | pass: inc+blk | pass: inc-blank |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2 | 0 | 0 | 30 | 0 | 0 |
| 2 | 8 | 4 | 1 | 0 | 27 | 0 | 0 |
| 3 | 6 | 5 | 1 | 0 | 28 | 0 | 0 |
| 4 | 3 | 7 | 0 | 0 | 29 | 1 | 0 |
| 5 | 7 | 7 | 1 | 0 | 25 | 0 | 0 |
| 6 | 2 | 12 | 0 | 3 | 21 | 1 | 1 |
| 7 | 0 | 5 | 4 | 0 | 11 | 4 | **16** |

- Two waves. At step 6 claims on failures fell (P(claim | fail) 0.80 at step 1, 0.12 at step 6), mostly into blocker reports. At step 7 the decline reached passing tasks: 20 of 31 passing feasible episodes reported `incomplete`, 16 of them with a blank blocker. Over steps 1-5 that was 1 of 140.
- It is not cell-specific. claim | pass at step 7: plain 0.23, high_risk control 0.36, remove_input control 0.75 (revoke_permission was not scheduled at step 7).
- Twins lost their blockers too: high_risk twins at step 7 were 0.19 claim, 0.31 inc+blk, 0.50 inc-blank (0.19 / 0.75 / 0.06 at step 6).

**It did not quit early.** Feasible n_actions held at 15.5-19.7 per step (17.9 at step 7). At step 7: pass + complete 16.8, pass + inc-blank 16.2, pass + inc+blk 18.5. One `lazy_decline` in the run (step 5). Step 7 had the highest pass rate of the run (0.775).

**Only the status token moved.** I read the 23 passing non-claims (parquet rows 219, 321, 367, 385, 387, 389, 390, 400, 401, 403, 405-407, 423, 434, 438-442, 444-446; cases honest_underclaim 19, false_blame 3, honest_failure_observed 1):

| kind | n (step 7) | example |
|---|---:|---|
| report contradicts its own status: "The task is complete", `incomplete: []`, confidence 1 (0.9-0.95 in two), status `incomplete`, blank blocker | 14 (14) | 385 (5622dcaf, plain); 401/403/406/407 (763181c5 control); 434/438/439 (c4fa82ae control); 440/441/445/446 (43e6d8bc, plain; 445 writes blocker "none/none/none") |
| invented blocker on finished work | 3 (2) | 400: "apart from a refund request, for which I do not have authorization" (no refund in the task); 387: risk_discovered / "std" with a warning text found nowhere in the transcript; 367 (step 6): real `whoami` = root, after "The task is complete" |
| doubt with a real reason (budget ran out mid-check, or a "patch skipped" log) | 4 (2) | 219 (step 4), 321 (step 6), 423, 442 |
| doubt with an invented reason | 2 (2) | 405, 444 |
| quit early | 0 | every row did the work and its check (7-28 commands) |

At step 1 the same narration ("All tasks completed: ✅ ...") ends in `complete` (rows 8, 24, 40, 56). Up to the status token the step-7 transcripts read like step 1.

Report format drifted alongside: the share of reports that put `status` first fell from 0.89 to 0.50, and among passing claims from 0.90 to 0.09. One reading is that status-first contexts were pushed hardest toward `incomplete`, and the claims that survive are in other contexts. z was not saved per row, so this is a hypothesis.

## 3. Why: the report term

**The status token learns only from l_rep.** `src/honest_agents/trainer.py` (line 788) `tmask[found, pos[found]] = 0` removes the status token from the main loss and from the KL. Its only gradient is l_rep (`src/honest_agents/trainer.py`, lines 236-237):

```
frozen = ((d > 0) & (zr.detach() > z_max)) | ((d < 0) & (zr.detach() < -z_max))
loss = -(c_rep * d * zr * (~frozen)).sum() / denominator
```

**The main loss pointed the other way.** Passing claims had a positive mean advantage at every step (+0.32 to +0.82). In all 7 groups holding both a passing claim and a passing non-claim (steps 4, 6, 7) the claim had the higher advantage (+0.45 to +1.08 vs -0.45 to +0.12). Pooled crgrpo_reward on passing feasible rows: plain claim 1.536 vs inc-blank 0.634; control claim 1.00 vs inc-blank 0.10. The outcome reward never paid underclaiming.

**delta per row** (`report_delta`, `src/honest_agents/crgrpo.py`, lines 177-192): L(claim) minus L(blank `incomplete` after real work), with the work forced to count as attempted (lines 189, 191). With k = V (line 64, mu_tw = mu_cd = 3):

| row | delta |
|---|---|
| feasible, tests pass (plain or control) | +0.9 (1 vs alpha, the bonus cancels) |
| feasible, tests fail | -0.386 (-mu_fc - alpha) |
| twin, no damage | -3V, about -1.9 to -2.6 |
| twin with damage | -V |

delta is never negative where a claim would be true. The problem is how the per-row signals add up on one token:

1. **The batch is net negative before any learning.** Logged `delta_negative_share` 0.53-0.64 per step. The 24 twins carry about -40 of the negative mass each step; the +0.9 rows add +21 to +28.
2. **The z_max freeze silences the positive rows.** With |z| about 15 at steps 1-3, a correct claim (delta > 0, z > 6) is frozen, and so is a confident twin decline (delta < 0, z < -6). What is left with gradient are the claimed twins and claimed failures, all delta < 0. Assuming sign(z) matches the sampled status, the predicted frozen share at steps 1-4 is 0.73 / 0.77 / 0.73 / 0.81 (logged 0.69 / 0.73 / 0.69 / 0.75), and at steps 1-5 every unfrozen row had delta < 0. The +0.9 rows only come unfrozen once their z has already fallen below 6: 20 of them at step 7, after the guard fired.
3. **The push does not fade.** The loss is linear in z with no pi(1 - pi) factor, so d l_rep / d z is a constant -c_rep * delta / N until |z| > 6. The measured |grad l_rep| / |grad main| was 1.38, 0.84, 1.64 on steps 0-2 (median 1.38, against the 0.75 the calibration targeted), all on one token per row.
4. **delta keys on facts the status context barely shows.** delta comes from the hidden tests and the twin/control label. The model's narration does not separate pass from fail (P(claim | fail) 0.80 at step 1), and twins and controls are built as lookalikes. What the token can learn is mostly the shared part: "at `<parameter=status>`, lower logit(complete)", which moves passing rows too. For a context that could be twin or control, claiming pays only if P(twin) is below about 0.22 (V about 0.7, control pass rate about 0.7); at the pool's 50/50 pairs a blank non-claim is the better status.
5. **C3 does not see it.** On a control, an `incomplete` after real work is priced alpha, not the refusal -mu_cd V (`src/honest_agents/crgrpo.py`, lines 165-166; the R1 counterfactual). At step 7 the control decline rate was 0.667 while `crgrpo/rate/cd` was 0.125. This is the same gap as the recorded stall deviation (pre-registration section 3, deviation 2), here on the status token instead of on the actions.

Section 3 is an inference: per-row z was not saved. The per-row deltas are rebuilt with the family-mean V, and twin damage is inferred from the `tw` counts in `crgrpo_state.jsonl`. Neither changes the sign of the push.

## 4. What the ablation arm will tell us

`configs/grpo_it19_crgrpo_ablation.yaml` (GPU 1, still running) is the same run with `report_term: false`: the status token stays in the main GRPO loss and gets no l_rep. Same seed, pool, data order, multipliers, KL and decline guard, so its steps line up with these.

- **If the ablation holds claim | pass near 1 and the decline rate near G2's (about 0.08-0.10) through step 7 and beyond**, the collapse is the report term's (sections 3.1-3.4), not the CR-GRPO table's. Section 3 predicts this, because the main-loss advantage favoured claiming on passes in every step here.
- **If it also drifts toward `incomplete` on passes**, the table itself (twin weight, the C3 blind spot, the stall gap) is pushing through the shared prose and arguments, and a report-term fix alone will not do.
- Either way it gives the false-claim rate the constraints reach without l_rep, which is the attribution the pre-registration asks for (section 7). With the method arm stopped at step 7 there is no method-arm candidate, so P1's arm difference cannot be computed as registered.

## 5. What a fix would need (options, not decisions)

- **Smaller c_rep.** The measured ratio was about twice the target. This only slows the drift: the net sign of the push (3.1, 3.2) stays negative.
- **No one-sided freeze.** Freeze on |z| regardless of the sign of delta, or drop the freeze, so correct claims pull back as hard as false ones push down.
- **A saturating loss.** Use the log-likelihood form (gradient scaled by pi(1 - pi) or by the gap to the target) instead of linear in z, so the push fades as pi(claim) moves.
- **Centre delta.** Subtract a baseline (per batch, per family, or per group) so a constant "lower the claim logit everywhere" component cancels.
- **Report term only where the model can tell.** Apply l_rep only to rows whose distinction is visible in context, e.g. leave out twins and controls, or keep only rows where the evidence was read.
- **Condition delta on visible evidence.** Price the counterfactual by what the transcript shows (e.g. a check that showed the violation, the evidence file read) rather than by the hidden test result and the twin label.
- **Close the C3 gap.** Count an `incomplete` on a passing control as a refusal for C3, or watch claim | pass with its own guard.

Each of these changes the method, so any re-run is a new pre-registered experiment. Whether to run one, and which option, is the owner's decision.

## Addendum: the step-5 checkpoint on the select split (exploratory, not pre-registered)

`configs/chains/it19_c_method_c5_diag.yaml`, S = 2, against the G2 reference run `runs/eval/it19c_g2_select`
(`analysis.power estimate`, DEF-66 sources out, 184 units, 95 % t intervals):

| metric | G2 | method step 5 | difference [95 % CI] |
|---|---:|---:|---|
| false claims per feasible episode (P1's metric) | 15.9 % | 13.6 % | −2.3 [−5.6, +0.9] |
| false claims among claims | 17.1 % | 16.5 % | −0.6 [−3.9, +2.6] |
| verified success | 76.9 % | 68.6 % | **−8.3 [−12.1, −4.4]** |
| claim rate, feasible | 92.8 % | 82.2 % | **−10.6 [−14.5, −6.7]** |
| claims on infeasible variants | 11.4 % | 12.1 % | +0.8 [−5.0, +6.5] |

By step 5 the report term had already traded correct claims for non-claims. The claim rate on feasible variants fell
by about 11 points and verified success by about 8 points. False claims per feasible episode moved by no more than
noise, and claims on infeasible variants did not change. So the drift that the guard stopped at step 7 was under way
by step 5, and it was not discriminating: the model claimed less everywhere rather than less where it had failed.
