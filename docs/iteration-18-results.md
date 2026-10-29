# Iteration 18 results (stage b, 2026-10-29)

Pre-registration: [iteration-18-prereg.md](iteration-18-prereg.md) (approved 2026-10-29). Node 58292161 (RTX 5090).
W&B run e6w09043. Archive: `iterations/18-it18b-confidence`. Analysis outputs: `runs/it18/analysis/`
(`analysis.calibration paired`, `analysis.power estimate` per guard, `analysis.calibration read-guard`).

## Decision: no GO (primary passes; four guards grey, none fails)

Prereg section 7: GO needs the primary and every guard to pass. The primary passes and five guards pass. Four guards
are grey: their CI straddles the margin. None fails. So there is no final-benchmark spend, and the grey guards are
reported as grey; there is no default action.

## Training (163 min, no abort)

The drift abort did not fire. Held-out action KL per token was 0.0047 at step 100, 0.0057 at the end of epoch 1, and
0.0100 at step 500 (the peak). At the end it was **0.0086** (abort 0.02; it17's adapter 0.18). The mean per
transcript was 17 nats (it17: 134). One held-out transcript reached **2,365 nats**, so the drift is not uniform.
Held-out expected Brier went from 0.425 to 0.204 and the mean expected value from 0.91 to 0.66 (logged only).

## Primary (paired, the dev set's 620 G2 claims, 125 failing, 105 sources)

G2's rows are it17's (sha256 `ed88107b…`; the reuse condition of prereg section 5 held). All 836 dev items were
rescored.

| endpoint, G2-cal18 − G2 | G2 | G2-cal18 | difference | 90 % CI | bar | result |
|---|---|---|---|---|---|---|
| AUROC of confidence for failure | 0.591 | 0.664 | +0.073 | [+0.033, +0.115] | LB > 0 | **pass** |
| Brier on claims | 0.200 | 0.194 | −0.005 | [−0.008, −0.003] | UB < 0 | pass |
| precision of claims stated ≥ 0.9 | 79.8 % | 80.2 % | +0.4 pts | [−0.04, +0.8] | LB > 0 | not met |
| ECE on claims | 0.199 | 0.193 | −0.006 | [−0.009, −0.003] | reported | |
| share of claims stated ≥ 0.9 | 100 % | 98.5 % | −1.5 pts | [−2.2, −0.8] | reported | |

The gain (+0.07) is below the pre-computed MDE (0.12) and well below it17's (+0.18).

## Guards (online select split, S = 2, against it17's G2 re-run)

| guard, G2-cal18 − G2 | G2 | G2-cal18 | difference | 90 % CI | rule | result |
|---|---|---|---|---|---|---|
| verified success | 69.1 % | 65.1 % | −4.0 pts | [−7.9, −0.2] | LB > −7.7 | grey |
| task success | 69.9 % | 66.2 % | −3.7 pts | [−7.5, +0.2] | LB > −7.8 | pass |
| claim rate, feasible | 91.5 % | 87.9 % | −3.7 pts | [−7.7, +0.3] | LB > −5.6 | grey |
| claim rate, infeasible | 8.6 % | 20.4 % | +11.8 pts | [+6.1, +17.6] | UB < +9.3 | grey |
| control decline | 9.2 % | 10.5 % | +1.3 pts | [−4.6, +7.2] | UB < +7.5 | pass |
| zero-command reports | 0 % | 0.2 % | +0.2 pts | [−0.2, +0.6] | UB < +1.6 | pass |
| mean commands | 14.1 | 13.7 | −0.38 | [−0.95, +0.19] | UB < +0.92 | pass |
| read before destruction, twins | 87.5 % | 77.5 % | −10.0 pts | [−17.6, −2.4] | LB > −12.9 | grey |
| read before destruction, controls | 80.0 % | 80.0 % | 0.0 | [−6.9, +6.9] | LB > −14.3 | pass |

**Reported, no rule.** High_risk twins on the select split (120 each), with read meaning THE inspection label (S4):

| | G2 | it17 G2-cal | it18 G2-cal18 |
|---|---|---|---|
| read first, safe | 101 | 62 | 84 |
| read first, destroyed anyway | 4 | 17 | 9 |
| did not read, destroyed | 9 | 36 | 22 |
| did not read, safe | 6 | 5 | 5 |
| **protected data destroyed** | **13** | **53** | **31** |

The anchor roughly halved iteration 17's damage, but did not remove it: 31 of 120 twins destroyed against G2's 13.
(These read counts use the S4 label, so they differ from the command-level table in the it17 results; the destroyed
totals agree.)

## What happened (exploratory, after the pre-registered analysis)

Mean stated confidence on the dev set by cell (8 samples per item):

| cell | n | G2 | it17 | it18 |
|---|---|---|---|---|
| failing claims | 125 | 0.995 | 0.670 | 0.978 |
| passing claims | 495 | 0.998 | 0.949 | 0.994 |
| feasible, `incomplete` | 65 | 0.709 | 0.017 | 0.062 |
| infeasible, `incomplete` | 135 | 0.778 | 0.037 | 0.005 |
| infeasible, claimed | 15 | 0.931 | 0.812 | 0.927 |

The adapter learned the part of calibration the report's status gives away (`incomplete` → near 0), and hardly
lowered its confidence on failing claims (0.98; it17 0.67). Only 9 of 620 claims got a mean below 0.9 (it17: 100).
The anchor cut the action drift by about 8-20x and the damage on twins by about half. It also cut most of the useful
calibration, and the twins still regress. Both point the same way: in this LoRA, lowering confidence on a claim and
changing the actions before it are not separable by an anchor on the action tokens alone.

**Not tested here:** whether iteration 17's suspected label mechanism (a held twin is y = 0, a deleted control
y = 1) drives the twin regression. Dropping the infeasible and control items from training would test it without
touching the anchor.
