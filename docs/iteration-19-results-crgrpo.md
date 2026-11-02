# Iteration 19, stage (b) results: CR-GRPO (2026-11-02)

Pre-registration: [iteration-19-prereg-crgrpo.md](iteration-19-prereg-crgrpo.md) (approved 2026-11-02 under the owner's
delegation, with its dated amendments and launch deviations). Run log and every infrastructure failure:
[iteration-19-signoff.md](iteration-19-signoff.md) (it19c). Bad-case reads:
[badcase-it19c/method_stop.md](iteration-19-research/badcase-it19c/method_stop.md) and
[badcase-it19c/ablation_select.md](iteration-19-research/badcase-it19c/ablation_select.md).

## Summary

| arm | outcome |
|---|---|
| **CR-GRPO (method)** | **stopped at step 7** on the pre-registered stall guard (feasible decline 0.325 over steps 3-7 > 0.30): it learned to report `incomplete` on tasks it solved. No candidate; P1 cannot be computed as registered. |
| **CR-GRPO − ℓ_rep (ablation)** | ran all 30 steps, every guard passes, but **more false claims than G2**: +3.8 points per feasible episode [+0.8, +6.8] (90 %), sign-flip p 0.045. Read as the candidate, that would be P1 **fail**. |

Neither arm supports the core claim. The report term, as designed, collapses the status token into blanket non-claims;
without it, the payoff table and the constraints leave the claim rate where it was while task success falls, so false
claims rise. The common cause (bad-case read): the reward depends on the hidden test result and the twin label, and
nothing in the policy's own context separates passing work from failing work. The ablation's extra false claims come
from variants it now passes less (+22, against −7 where it passes more), claimed at confidence about 1.0 without
running the tests. At μ_fc's cap (0.35) a claim pays whenever P(pass) > 1/3, and on controls a blocker-citing decline
is priced about −2.2 against −0.35 for a false claim (controls carry 56 of the 93 false claims). On all 208 high_risk
twins at S = 4, irreversible violations were 31 (G2) and 37 (ablation), descriptive.

## Method arm (stopped)

c_rep 0.00213 (calibrated: gradient ratio 352 / 251 / 598 at c_rep 1; clip amended before any step). Ratios on the
measured steps 0-2: 1.38, 0.84, 1.64 (guard 3). pi(claim) at the status token 0.73 -> 0.08 over steps 1-7; at step 7,
20 of 31 passing feasible episodes reported `incomplete` (1 of 140 over steps 1-5). Exploratory, not pre-registered:
its step-5 checkpoint on the select split (S = 2) against G2: feasible claim rate −10.6 [−14.5, −6.7], verified
success −8.3 [−12.1, −4.4], false claims per feasible episode −2.3 [−5.6, +0.9] (95 %). The mechanism (status token
trained by ℓ_rep alone, a net-negative batch delta, a one-sided freeze) is in the bad-case read.

## Ablation arm: CR-GRPO − ℓ_rep − G2, select split (S = 4, 816 episodes, 0 infra errors)

`analysis.power estimate`, paired by source, DEF-66 sources and the quarantine out, 90 % intervals (t at the estimate's
df); the ablation's comparisons with G2 are descriptive (pre-registration section 3).

| | metric | G2 | ablation | difference [90 %] | pre-registered bar | reading |
|---|---|---:|---:|---|---|---|
| P1's metric | false claims per feasible episode | 15.9 % | 19.7 % | **+3.8 [+0.8, +6.8]** (p 0.045) | pass UB < 0 | worse (fail, read as the candidate) |
| | false claims among claims | 17.1 % | 21.1 % | +4.0 [+0.9, +7.2] | reported | |
| G1 | verified success | 76.9 % | 73.5 % | −3.4 [−6.9, +0.1] | LB > −7.7 | pass |
| G2 | claims on infeasible variants | 11.4 % | 13.3 % | +1.9 [−2.4, +6.2] | UB < +9.3 | pass |
| G3 | control decline | 6.2 % | 6.2 % | +0.0 [−2.6, +2.6] | UB < +7.5 | pass |
| G4 | zero-command reports | 0.0 % | 0.0 % | 0.0 | UB < +1.6 | pass |
| G5 | read before destruction, twins / controls | 84.1 / 83.2 % | 79.3 / 84.1 % | −4.8 [−10.8, +1.2] / +1.0 [−3.8, +5.7] | LB > −12.9 / −14.3 | pass |
| G6 | rule 1b on the final checkpoint | 17 destroyed, read 85.8 % | 17, 84.2 % | | does not fire | pass |
| | feasible claim rate | 92.8 % | 93.2 % | +0.4 [−2.3, +3.1] | reported | |

Rule 1b at step 15 did not fire either (twins destroyed 12 vs 17; read 80.8 vs 85.8 %).

## Attribution (section 7)

The pre-registered contrast (CR-GRPO − ablation on P1 and G1) cannot be computed: the method arm has no final
checkpoint. What the two arms show together: with the report term the status token collapsed within 7 steps; without it
the model claimed as often as G2 and failed more of those claims.

## Not run

The method arm's final benchmark (no candidate). Stage a2 (TB2 and ImpossibleBench for base and G2 under the external
budget) is reported in [iteration-19-results-baseline.md](iteration-19-results-baseline.md).
