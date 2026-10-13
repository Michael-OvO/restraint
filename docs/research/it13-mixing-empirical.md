# Does mixing task families in a GRPO step hurt? Empirical analysis of our archived runs

Scratch: `/tmp/it13_research/empirical/` (`build_master.py`, `analyse.py`, `analyse2.py`, `results.txt`, `master.parquet`, `bank.parquet`, `step_composition.csv`). Read-only on the archive and the repo.

## 0. Data and setup

| run | steps | rows | groups | pool | notes |
|---|---:|---:|---:|---|---|
| it11 (`11-sft-grpo`) | 20 | 1280 | 160 | train_rl 351 rows | GRPO from base, old harness (late reports dropped) |
| it12 B0 (`12-harness-b0`) | 15 | 960 | 120 | same pool, **same groups per step as it11** (verified) | fixed harness (8 report-only calls) |
| it09 run 2 (`09-grpo-run2`) | 51 | 3264 | 408 | 720-row pool (older, many degenerate twins) | different reward constants (gamma = 0.25) |

- 8 prompts (groups) x 8 generations per step. `scale_rewards: "none"`, so **advantage = reward - group mean** exactly (recomputed, max |diff| 2e-7). No std scaling anywhere.
- Task types ("famkind" = family/kind): high_risk twin + control, remove_input twin + control, revoke_permission twin + control, plain feasible. A step contains 5-7 of the 7 famkinds (it11 mean 5.75; it09 3-7, mean 5.3).
- Reward ladder actually observed: success_claimed +1.54, abstain_correct +1.0, abstain_wrong_reason +0.5, honest_failure +0.1, implicit_abstain 0 or -1 (budget exit, -1 if damage), no-report -0.25..+0.33, false_blame -0.4, fabrication -1.0, infeasible_claimed -1.89 on average (-1 claim, -1 damage penalty on nearly all high_risk twins).
- Base rates: iteration-03 pass@4 bank (957 variants x 4 samples; covers 349/351 pool variants; 159/160 it11 groups). Its reward config differs (alpha 0.25), so only case labels / behaviour rates are compared, not reward values.
- Behaviour features parsed from completions: `n_actions`, claim/decline/explicit-abstain/no-report indicators (from `case`), `budget_exhausted` (= stop_reason != report), README read (regex on `cat/head/... README`), number of inspection commands, completion chars.

The "gradient-direction proxy" used below is `push_f(x) = sum_i adv_i * x_i` over rows of famkind f. Because advantages sum to zero within a group, this equals `sum_i adv_i (x_i - x_group_mean)`, which is the REINFORCE gradient on a logit-linear parameter for behaviour x under a group baseline. It is a proxy for marginal (context-free) behaviour, not for what a context-conditioned policy learns; that caveat matters for the "claim" feature (see A3).

## 1. Ranked findings (numbers first, inference marked)

1. **~75 % of the per-step reward variance never reaches the gradient.** Pooled per-step reward sum-of-squares splits (it11 / it12 / it09): between family 23.5 / 25.1 / 18.6 %; twin-vs-control within family 48.7 / 45.0 / 37.6 %; between groups within famkind 4.5 / 4.9 / 14.7 %; **within group (= advantage) 23.3 / 24.9 / 29.1 %**. The jagged logged `reward` curve is mostly task-type composition, and group centring removes it. (Data.)
2. **Composition explains at most about half of the step-to-step noise in mean reward, and none of the noise in advantage mass or gradient norm.** Leave-one-step-out composition prediction of step mean reward: R2 0.43 (it11), 0.16 (it12), 0.01 (it09). Cleanest estimate: it11 and it12 saw identical task sets per step for steps 1-15; their step mean rewards correlate r = 0.69 (r2 = 0.48), but logged grad_norm correlates only r = 0.33 and loss r = 0.32. Per-step sum|adv| is not predicted by composition (LOO R2 <= 0 in all runs); grad_norm has CV 0.18-0.23 and correlates 0.64-0.77 with sum|adv| but |r| <= 0.30 with the number of twin or high_risk-twin groups in the step. (Data.) Inference: the advantage signal is not made non-smooth by the mix; what is non-smooth is a diagnostic the optimiser does not see.
3. **No family dominates the gradient.** it11 share of sum|adv| (rows share in brackets): high_risk twin 13.1 % (22.5), high_risk control 24.4 % (23.1), remove_input twin 9.4 % (12.5), remove_input control 21.9 % (13.1), revoke twin 2.1 % (3.1), revoke control 8.8 % (7.5), feasible 20.3 % (18.1). Twins together 24.6 % of |adv| for 38 % of rows (it12: 25.7 %; it09: 11.7 %). Per step, no famkind exceeds 53 % of the step's |adv| in it11 (mean shares 2-23 %); in it09 one famkind exceeded 50 % in 19 of 51 steps and one step had zero mass. (Data.)
4. **Reward scale does not inflate the twins' weight; the twins are under-weighted because half their groups are degenerate.** Zero-variance group share: high_risk twin 55.6 % (it11) / 46.4 % (it12) / 86.1 % (it09); remove_input twin 45 / 33 / 64 %; revoke twin 20 / 0 / 37 %; controls and feasible 0 % in it11/it12. |adv| per *live* group is similar across famkinds (it11: 3.7-5.8; high_risk twin 4.6). The -2 floor does not create outsized groups because live high_risk twin groups contrast -1.89 (claim+damage) with -0.27 (budget exit) or +0.5 (wrong-reason decline), a gap of about 1.6-2.4, comparable to success (+1.55) vs no-report (-0.1) or vs fabrication (-1). (Data.)
5. **The high_risk twin signal is real but points at the wrong behaviour, and that is independent of mixing.** In 836 high_risk twin training rows across the three runs there are **0 `abstain_correct`** (749 claimed, 69 implicit/budget, 18 wrong-reason); 0 README reads. Inside live high_risk twin groups (it11) the cases are 91 infeasible_claimed, 30 implicit_abstain, 7 abstain_wrong_reason; the top-advantage row is a budget exit in 11 of 16 live groups (it12: 10 of 15), a wrong-reason decline in 3 (4), and in 2 (1) an undamaged -1 claim out-ranking damaged -2 claims. The "actual/binary" variance ratio is 13.4 for high_risk twins vs 3-6 for every other famkind: the variance is among *negative* outcomes, not good-vs-bad flips. By contrast, in remove_input and revoke live twin groups the top row is `abstain_correct` in 100 % (11/11 and 4/4 in it11; 10/10 and 3/3 in it12). (Data.)
6. **The length conflict the previous review found is confirmed, but it is a within-family twin-vs-control conflict concentrated in high_risk, not a cross-family one.** Within-group corr(adv, n_actions): high_risk twin **+0.35** (it11) / +0.33 (it12) / +0.24 (it09); every other famkind negative (-0.01 to -0.46). Longer half of each live group: high_risk twin mean adv **+0.275**, high_risk control -0.067, remove_input twin -0.199, feasible -0.167 (it11+it12). Raw push on n_actions (it11): high_risk twin +178 vs high_risk control -201; remove_input twin -250, feasible -252. Within the high_risk family twin+control nearly cancel: net/gross 0.06 (n_actions), 0.07 (is_claim), 0.03 (budget) in it11. In it12 the fixed harness halved budget exits on controls, their counter-push shrank (-63), and the twin lesson "work longer, claim less" won inside high_risk (net/gross 0.50 / 0.72). (Data.) Inference: single-family high_risk training would keep this cancellation (it is twin vs control, both high_risk) and removing controls would turn the lesson into "stall until the budget runs out".
7. **Cross-family direction alignment is moderate and is what the design asks for.** Net/gross across famkinds (it11 / it12 / it09): n_actions 0.65 / 0.31 / 0.83; is_claim 0.27 / 0.00 / 0.56; is_decline 0.59 / 0.34 / 0.49; budget_exhausted 0.57 / 0.12 / 0.82; passed 1.0; explicit_abstain 1.0 (only twins have it, all push it up); no_report 1.0 (all push it down). Per step (fewer famkinds present) the mean net/gross is higher: n_actions 0.58-0.65, is_claim 0.48-0.84. (Data.) Inference: the claim/decline cancellation is the intended discrimination signal (twins down, controls up); it is a conflict only where the policy cannot observe what distinguishes twin from control. For remove_input and revoke the blocker is encountered during work and the correct abstention is ranked top in every live group; for high_risk the distinguishing evidence (README) is never read, so there the cancellation is real at the policy's own state.
8. **Learning per family tracks signal type, not mixing.** Training-rollout trends (per 10 steps, t in brackets): high_risk twin claim -0.03 (t -0.5, it11), -0.08 (t -0.8, it12), 0.00 (it09); abstain_correct flat at 0 everywhere. remove_input twin abstain_correct +0.08 (t 0.9, it11), +0.18 (t 1.3, it12); paired with the bank on the same variants, the last it12 bin is 0.80 vs bank 0.55. revoke twin: only 5 groups in it11 and 3 in it12 (24 rows), so the it12 tier-F gain (+9 of 56) cannot be traced in training data; in it09 (46 revoke twin groups) they learned to decline earlier (n_actions -1.37/10 steps, t -3.2; budget exits -0.05, t -2.0). Controls regressed via budget exits in it11: revoke control claim -0.30/10 steps (t -4.4), budget +0.24 (t 3.4); high_risk control claim -0.11 (t -1.9), budget +0.12 (t 2.5); in it12 revoke control still -0.24 (t -2.3) but high_risk control +0.06 (ns). it09 remove_input controls gained capability (pass +0.08/10 steps, t 3.4). (Data.)
9. **Per-family advantage normalisation would move weights by at most ~5 points and would not change directions.** Pooled within-group reward std by famkind is 0.37-0.87 (ratio 2.3). Rescaling advantages by per-famkind std (per step) changes it11 shares: high_risk twin 13.1 -> 15.6 %, remove_input control 21.9 -> 16.9 %, others within 2 points; per-run std: high_risk twin 18.2 %. TRL's default group-std scaling: high_risk twin 12.4 %, high_risk control 28.3 %. Net/gross alignment barely moves (it11 is_claim 0.27 -> 0.22 / 0.08; n_actions 0.65 -> 0.65 / 0.55). (Data.)
10. **Homogeneous blocks would make the per-step gradient magnitude less smooth, not more.** Actual mixed steps: sum|adv| CV 0.30 (it11), 0.31 (it12), 0.54 (it09). Bootstrapped 8-group single-famkind blocks from the same groups: block mass 16 (high_risk twin) to 46 (remove_input control) in it11, a 2.8x swing between consecutive blocks of different families; a high_risk twin block has a 40 % chance (it11) / 30 % (it12) of carrying less than half the mixed-step mean, and in it09 a 29 % chance of zero mass (95 % below half). Direction would also flip systematically at every block boundary because each famkind's push sign is fixed (n_actions: 6 famkinds negative, 1 positive). (Data + simulation on recorded rows.)

## 2. A1: variance decomposition and step-to-step noise

Per-step reward SS shares, pooled over steps:

| run | between family | twin vs control within family | between group within famkind | within group (-> advantage) |
|---|---:|---:|---:|---:|
| it11 | 0.235 | 0.487 | 0.045 | 0.233 |
| it12 | 0.251 | 0.450 | 0.049 | 0.249 |
| it09 | 0.186 | 0.376 | 0.147 | 0.291 |

Per step (it11) the within-group share ranges 0.09-0.54. The dominant component everywhere is twin-vs-control inside a family (the -1.9 vs +1.0 contrast), which is by construction and never enters the loss.

Step mean reward noise:

| run | std(step mean) | std(composition-predicted, LOO famkind means) | std(residual) | R2 | R2 family-only |
|---|---:|---:|---:|---:|---:|
| it11 | 0.174 | 0.129 | 0.132 | 0.43 | 0.26 |
| it12 | 0.170 | 0.108 | 0.156 | 0.16 | 0.04 |
| it09 | 0.324 | 0.247 | 0.323 | 0.01 (in-sample OLS 0.27, adj 0.15) | 0.17 |

corr(step mean reward, number of high_risk twin groups) = -0.60 (it11), -0.35 (it12), 0.00 (it09). The paired it11/it12 comparison (same tasks per step) gives r = 0.69 for step mean reward, 0.91 for step claim rate, 0.90 for mean n_actions, 0.68 for sum|adv|, 0.53 for zero-variance share, but only 0.33 for grad_norm and 0.32 for loss.

Train-log correlations (steps with grad_norm): grad_norm vs sum|adv| 0.70 / 0.68 / 0.64; vs number of twin groups -0.23 / -0.30 / 0.07; vs number of high_risk twin groups 0.03 / 0.06 / -0.07; vs frac_reward_zero_std -0.31 / -0.34 / 0.05. grad_norm CV 0.18 / 0.19 / 0.23 (max_grad_norm 1.0, mean 0.07-0.08, never clipped).

Per-step sum|adv| vs composition (LOO per-famkind mass per group): R2 -0.12 (it11), -0.16 (it12), negative in it09; corr -0.29 / -0.20 / -0.54. In it09 the per-step zero-variance share does track the number of twin groups (r = 0.92) because 64-86 % of that pool's twin groups were degenerate; in it11/it12 it does not (r = -0.15 / -0.06).

## 3. A2: advantage mass by famkind

it11 (it12 similar, it09 in `results.txt`):

| famkind | rows | row share | share sum abs adv | share sum adv^2 | zero-var groups | abs adv per live group | var ratio actual/binary | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_risk twin | 288 | 0.225 | 0.131 | 0.118 | 0.556 | 4.56 | 13.4 | -1.72 |
| high_risk control | 296 | 0.231 | 0.244 | 0.257 | 0.000 | 3.66 | 4.7 | 1.02 |
| remove_input twin | 160 | 0.125 | 0.094 | 0.089 | 0.450 | 4.73 | 3.2 | 0.69 |
| remove_input control | 168 | 0.131 | 0.219 | 0.229 | 0.000 | 5.80 | 4.3 | 0.56 |
| revoke twin | 40 | 0.031 | 0.021 | 0.010 | 0.200 | 2.91 | 1.0 | 0.78 |
| revoke control | 96 | 0.075 | 0.088 | 0.084 | 0.000 | 4.09 | 5.2 | 1.12 |
| feasible | 232 | 0.181 | 0.203 | 0.214 | 0.000 | 3.90 | 4.7 | 0.95 |

By family: high_risk 37.5 % of |adv| (45.6 % of rows), remove_input 31.3 % (25.6), feasible 20.3 % (18.1), revoke 10.9 % (10.6). The "var ratio actual/binary" is sum adv^2 divided by the same quantity computed on a good/bad (reward > 0) indicator; 13.4 for high_risk twins means their within-group variance comes from contrasts among negative outcomes (-1.89 vs -0.27 vs +0.5), i.e. the gradient there is "claim-and-destroy is worse than stalling or blaming capability", not "claim is worse than correct abstention".

Per-step |adv| share, it11: means 0.13 / 0.23 / 0.10 / 0.23 / 0.02 / 0.09 / 0.20 in the order above, std 0.04-0.15, max 0.53 (feasible, one step). it12 max 0.52 (high_risk twin, one step). it09: max 0.99, 19 steps with a >50 % famkind.

## 4. A3: direction conflicts

Within-group correlation corr(adv, x - group mean), it11 (it12 / it09 agree in sign for every cell):

| famkind | n_actions | is_claim | is_decline | explicit_abstain | budget_exhausted | n_inspect | comp_chars | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high_risk twin | **+0.35** | -0.88 | +0.88 | +0.42 | **+0.67** | +0.17 | +0.27 | - |
| high_risk control | -0.24 | +0.45 | -0.07 | - | -0.42 | -0.08 | -0.26 | +0.91 |
| remove_input twin | -0.46 | -0.88 | +0.94 | +0.94 | -0.20 | -0.22 | -0.37 | - |
| remove_input control | -0.01 | +0.45 | -0.21 | - | -0.36 | +0.05 | -0.11 | +0.88 |
| revoke twin | -0.21 | - | - | +0.94 | -0.94 | -0.22 | -0.21 | - |
| revoke control | -0.43 | +0.68 | -0.17 | - | -0.63 | -0.06 | -0.37 | +0.87 |
| feasible | -0.34 | +0.43 | -0.12 | - | -0.37 | -0.15 | -0.30 | +0.87 |

Within-family cancellation of raw pushes (twin + control):

| run | family | feature | twin push | control push | net | net/gross |
|---|---|---|---:|---:|---:|---:|
| it11 | high_risk | n_actions | +178 | -201 | -23 | 0.06 |
| it11 | high_risk | is_claim | -30.6 | +26.7 | -3.9 | 0.07 |
| it11 | high_risk | budget_exhausted | +23.1 | -24.6 | -1.6 | 0.03 |
| it11 | remove_input | n_actions | -250 | -4.5 | -255 | 1.00 |
| it11 | remove_input | is_claim | -20.4 | +26.5 | +6.1 | 0.13 |
| it11 | revoke | n_actions | -6.2 | -125 | -131 | 1.00 |
| it12 | high_risk | n_actions | +191 | -63 | +128 | 0.50 |
| it12 | high_risk | is_claim | -32.1 | +5.3 | -26.8 | 0.72 |
| it12 | remove_input | n_actions | -149 | +20 | -129 | 0.76 |
| it09 | high_risk | n_actions | +89 | -139 | -50 | 0.22 |
| it09 | high_risk | is_claim | -12.8 | +29.5 | +16.7 | 0.39 |

Reading: length is rewarded only on high_risk twins (stalling beats destroying); every other famkind, including the other two twin types, rewards shorter episodes (correct abstentions and successes are shorter than budget exits and fabrications). The claim/decline opposition between twins and controls is the designed contrast. Whether it is "conflict" depends on observability: remove_input/revoke twins hit the blocker during work (and their correct abstentions top every live group), high_risk twins never read the README (0 of 836 rows), so the twin and its control present the same state to the policy and the pushes cancel.

Caveat on the proxy: the net push on n_actions was negative (shorter) in it11, yet it11's episodes got longer in every famkind (+1.8 to +3.2 actions per 10 steps; probe 13.5 -> 17.7 commands). A linear-feature push does not predict realised drift; the realised drift matched the one positive push (high_risk twins), possibly because that lesson is consistent (corr +0.35, one mechanism) while the "shorter" push is entangled with pass/fail. This is inference; the data cannot separate it from the harness leak (dropped late reports) that it12 fixed, after which the length slopes were mixed and non-significant.

Sign flips of the per-step net push between consecutive mixed steps: n_actions 21 % (it11) / 43 % (it12) / 40 % (it09); is_claim 42 / 64 / 56 %. The is_claim flips are mostly noise around a near-zero net (cancellation by design). Under single-famkind blocks the sign per block is fixed by the famkind (table above), so flips would occur deterministically at family boundaries.

## 5. A4: per-family learning with the mix controlled

Training rollouts, binned, with the pass@4 bank rate on the same variants (paired). Selected rows (full table in `results.txt`):

| run | famkind | bin | claim | bank claim | abstain_correct | bank abstain_correct | budget exit | pass | bank pass | n_actions | bank n_actions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| it11 | high_risk twin | s01-05 | 0.93 | 0.86 | 0.00 | 0.00 | 0.04 | 0 | 0 | 16.1 | 16.6 |
| it11 | high_risk twin | s16-20 | 0.92 | 0.97 | 0.00 | 0.00 | 0.08 | 0 | 0 | 17.7 | 11.6 |
| it12 | high_risk twin | s11-15 | 0.78 | 0.75 | 0.00 | 0.00 | 0.11 | 0 | 0 | 17.6 | 18.5 |
| it11 | remove_input twin | s01-05 | 0.18 | 0.25 | 0.75 | 0.70 | 0.05 | 0 | 0 | 14.4 | 14.5 |
| it11 | remove_input twin | s16-20 | 0.00 | 0.00 | 0.98 | 0.90 | 0.02 | 0 | 0 | 17.1 | 10.9 |
| it12 | remove_input twin | s11-15 | 0.10 | 0.40 | 0.80 | 0.55 | 0.05 | 0 | 0 | 11.8 | 14.6 |
| it11 | revoke twin (n=8-16/bin) | s16-20 | 0.00 | 0.00 | 0.69 | 0.50 | 0.25 | 0 | 0 | 27.2 | 26.4 |
| it12 | revoke twin (n=8/bin) | s11-15 | 0.00 | 0.00 | 0.88 | 1.00 | 0.00 | 0 | 0 | 25.4 | 27.5 |
| it11 | revoke control | s01-05 | 1.00 | 0.92 | - | - | 0.00 | 0.88 | 0.92 | 15.4 | 15.7 |
| it11 | revoke control | s16-20 | 0.62 | 0.92 | - | - | 0.29 | 0.71 | 0.83 | 21.8 | 16.4 |
| it11 | high_risk control | s16-20 | 0.76 | 0.88 | - | - | 0.22 | 0.79 | 0.88 | 18.5 | 15.1 |
| it12 | high_risk control | s11-15 | 0.96 | 0.92 | - | - | 0.04 | 0.83 | 0.81 | 16.5 | 16.9 |
| it11 | feasible | s16-20 | 0.77 | 0.93 | - | - | 0.23 | 0.70 | 0.68 | 20.0 | 14.7 |
| it12 | feasible | s11-15 | 0.82 | 0.68 | - | - | 0.07 | 0.80 | 0.57 | 20.3 | 19.8 |

Linear trends (per 10 steps, t-stat) are in finding 8. it09 (50 steps, older pool): revoke twins declined earlier over training (n_actions -1.37/10 steps, t -3.2), remove_input controls gained (pass +0.08/10, t 3.4; reward +0.21, t 3.6), high_risk twins flat at 0 abstain_correct, zero-variance 43-90 % per bin.

In-run probes (held-out, 1 sample, 20 twins per family): it11 high_risk claims 0.85 -> 0.80, remove_input decline 1.0 -> 0.6, revoke decline 0.79 -> 0.43 (steps 0 -> 20); it12 high_risk claims 1.0 / 1.0 / 1.0, remove_input decline 0.8 / 0.8 / 0.67, revoke decline 1.0 / 1.0 / 1.0, control declines 0 / 0 / 0.10 for high_risk and 0 / 0 / 0.17 for remove_input (steps 0 / 10 / 15).

Signal availability at base (bank, variants it11 trained on): expected share of live groups with 8 samples from the bank's p(reward > 0): high_risk twin 0.26, remove_input twin 0.28, revoke twin 0.00, high_risk control 0.45, remove_input control 0.84, revoke control 0.52, feasible 0.90. Observed in it11: 0.44, 0.55, 0.80, 1.0, 1.0, 1.0, 1.0 (training at T = 1 with the finer reward ladder produces more live groups than the binary bank estimate). High_risk twins have the least base signal of any famkind, and (finding 5) the signal they have is of the wrong kind.

## 6. A5: counterfactual normalisation and homogeneous blocks

Share of sum|adv| by famkind, it11:

| famkind | rows | actual | TRL group-std | famkind-std per step | famkind-std per run | batch-std per step |
|---|---:|---:|---:|---:|---:|---:|
| high_risk twin | 0.225 | 0.131 | 0.124 | 0.156 | 0.182 | 0.129 |
| high_risk control | 0.231 | 0.244 | 0.283 | 0.253 | 0.232 | 0.235 |
| remove_input twin | 0.125 | 0.094 | 0.082 | 0.084 | 0.111 | 0.097 |
| remove_input control | 0.131 | 0.219 | 0.167 | 0.169 | 0.166 | 0.224 |
| revoke twin | 0.031 | 0.021 | 0.033 | 0.034 | 0.037 | 0.020 |
| revoke control | 0.075 | 0.088 | 0.094 | 0.096 | 0.084 | 0.092 |
| feasible | 0.181 | 0.203 | 0.217 | 0.208 | 0.188 | 0.202 |

Pooled within-group std by famkind (it11): 0.48, 0.69, 0.55, 0.87, 0.37, 0.70, 0.72. Alignment under the schemes (it11 net/gross): is_claim 0.27 / 0.28 / 0.22 / 0.08; n_actions 0.65 / 0.76 / 0.65 / 0.55 (actual / group-std / famkind-step / famkind-run). it12 and it09 show the same pattern (shifts <= 5 points, alignment unchanged or lower).

Homogeneous 8-group blocks (bootstrap from each run's recorded groups):

| run | famkind | block mass mean | block CV | P(block < half the mixed-step mean) | P(block zero) |
|---|---|---:|---:|---:|---:|
| it11 (mixed steps: mean 27.8, CV 0.30) | high_risk twin | 16.3 | 0.45 | 0.40 | 0.01 |
| | high_risk control | 29.1 | 0.29 | 0.03 | 0 |
| | remove_input twin | 20.8 | 0.37 | 0.18 | 0 |
| | remove_input control | 46.4 | 0.10 | 0 | 0 |
| | feasible | 31.3 | 0.25 | 0.01 | 0 |
| it09 (mixed steps: mean 24.8, CV 0.54) | high_risk twin | 4.3 | 0.99 | 0.95 | 0.29 |
| | remove_input twin | 7.9 | 0.70 | 0.79 | 0.09 |
| | remove_input control | 34.8 | 0.30 | 0.02 | 0 |

## 7. Conclusions

What the data supports:

- **Against "mixing makes the advantage signal non-smooth or harmful."** The step-to-step jaggedness is in the raw reward (75 % of its variance is between task types), which group-centred GRPO never uses. Advantage mass per step is not predicted by composition, grad_norm is stable (CV ~0.2) and uncorrelated with twin counts, and the one paired experiment we have (it11 vs it12, identical task sets per step) shows the composition-shared component of step reward is about half while the shared component of grad_norm/loss is ~0.1. Homogeneous blocks would raise, not lower, the step-to-step variation in gradient magnitude (2.8x swing between family blocks in it11; 40 % of high_risk-twin blocks under half the mixed mean; 29 % empty blocks in the it09 pool) and would flip gradient direction at every block boundary.
- **The real conflict is within-family, at the same observable state, in high_risk.** Twins reward stalling (longer, +0.275 for the longer half) and controls punish it; with no README reads the two are indistinguishable to the policy and cancel (net/gross 0.03-0.07 in it11). This is not fixed by training one family at a time (the pair is inside the family) and would be made worse by dropping controls (the twin lesson alone is "run out of budget"). It is the same diagnosis as the iteration-12 reflection: the signal that exists cannot be acted on because the evidence is never in context.
- **Where the signal is of the right kind, learning happened under the mix.** remove_input and revoke twins rank correct abstention top in 100 % of live groups; revoke moved in it12 (tier F) and in it09 (earlier declines, t -3.2), remove_input improved relative to the bank in training rollouts (+0.25 in the last it12 bin). high_risk never moved in any run, and its zero-variance share stayed 46-56 % (it11/it12) and 86 % (it09) regardless of what else was in the step.
- **Per-family normalisation is not a lever.** Within-group std differs at most 2.3x across famkinds, re-weighting shifts shares by <= 5 points, and direction alignment does not improve (it gets slightly worse per run).

What the data does not say (inference, or untestable here):

- No archived run trained one family alone or in homogeneous blocks, so the counterfactuals are simulations on recorded rows; they capture magnitude and direction of the recorded advantages, not how the policy would have evolved. The push proxy also failed to predict the it11 length drift, so treat direction conclusions as about the signal, not the outcome.
- Curriculum ordering: nothing in the three runs varied the order. The only ordering-adjacent evidence is pool rebalancing (it09 -> it11), which cut the high_risk twin zero-variance share from 86 to 56 % and raised the twins' |adv| share from 12 to 25 %, without producing a single correct high_risk abstention.
- If anything, the data argues for more mixing-with-weighting rather than less: oversample the twin types that carry a correct-abstention signal (remove_input, revoke: 5 and 3 revoke twin groups in 20/15 steps is too few to learn from or to measure), keep controls paired with their twins, and for high_risk change what is observable at the decision point (the iteration-13 plan) rather than how steps are composed.

## 8. Limitations

- Three runs, one seed each; the it11/it12 pairing is the only controlled comparison and covers 15 steps.
- README detection is a regex on the completion text (`cat|head|... README`); it found 0 reads on high_risk twins, consistent with the bad-case review's manual count.
- The pass@k bank used alpha = 0.25 / gamma = 0.25, so only case labels and behaviour rates were paired with it, not rewards; the bank is at temperature/sampling settings of iteration 03, which generate fewer live groups than training at T = 1.
- The gradient-direction proxy is marginal and linear; GRPO with per-token loss and importance weights also weights by sequence length and clipping, which this analysis ignores.
