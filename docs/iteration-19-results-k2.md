# Iteration 19, stage (b), K2 results (2026-11-01)

Design: [iteration-19-design-crgrpo.md](iteration-19-design-crgrpo.md), section 3 (K2 pass and kill rules fixed there
before the run). Job file `configs/chains/it19_b_k2.yaml`, node 58586655, right after it19a. W&B run 4ztiq12w.
Archive: `iterations/19-it19b-k2`. Cost of the session: about $2.

## Verdict: grey on both parts; no pre-registered kill fired, no pass

## K2a: the report term alone

Training: 264 optimiser steps (1 epoch) on 2,108 items (374 DEF-66 items dropped), 64 min, no abort. Held-out drift
(200 items): 0.0041 per token at step 100, 0.0031 at the end (abort 0.02).

Paired readout on the dev set (`--status-score`, exact π(claim) from the logits; `analysis.calibration paired`, DEF-66
excluded: 558 claims, 84 failing, 93 sources):

| endpoint | G2 | K2a | difference [90 % CI] | rule | result |
|---|---:|---:|---|---|---|
| AUROC of π(claim) for failure | 0.599 | 0.696 | +0.097 [+0.023, +0.183] | ≥ G2 + 0.08, LB > 0 | pass |
| failing claims crossing π < 0.5 | 2.4 % | 9.5 % | | ≥ 25 % | **fail** |
| passing claims crossing π < 0.5 | 0.0 % | 1.3 % | | ≤ 8 % | pass |
| Brier on claims | 0.146 | 0.138 | −0.008 [−0.016, −0.000] | reported | |
| share of claims at π ≥ 0.9 | 98.9 % | 95.0 % | −3.9 [−5.8, −2.2] | reported | |

The kill rule (gain < 0.05, or movement only with drift above 0.02) did not fire; the pass rule's crossing condition
failed. The decision ranks failures better but moves about one failing claim in ten across the claim line, so most
reports would not change.

## K2b: behaviour on the 120 select high_risk twins (60 x 2), against it17's G2 re-run

| | G2 | K2a | kill rule | result |
|---|---:|---:|---|---|
| protected data destroyed | 13 / 120 | 21 / 120 | > G2 + 8 (> 21) | at the limit, not fired |
| read before destruction (S4) | 87.5 % | 78.3 % (−9.2) | a drop of more than 10 points | inside, by 0.8 points |
| claims on twins | 13 | 21 | | |

Neither kill fired, by the narrowest margins. The direction is it17 / it18's: training the report decision on G2's
transcripts made the agent more careless on twins (8 more destructions), smaller than it17 (53 / 120) and it18
(31 / 120). At 120 twins this difference is not established either way.

## What it says

- The report term moves the claim decision in the right direction (AUROC +0.10) but not far enough, at this step
  count and weight, to change most reports.
- Even with the action anchor and a drift of 0.003 per token, the twin behaviour slides toward the it17 / it18
  failure. Three iterations now point the same way: in this LoRA, training anything about the report on G2's own
  transcripts couples with the actions.
- The design's fallback (section 3, K2) applies in spirit: a **separate reporter adapter** (G2 acts; the adapter
  only writes the status) keeps the actions G2's by construction, at the price of the claim becoming "the system
  reports honestly".

## Options for the owner

1. **The full CR-GRPO run** (on-policy, the constraints on the actions themselves): about $20-30 with evals; the
   balance is $15.49, so it needs a top-up. Risk: K2 suggests the report part couples; the run would rest on the
   constraint part.
2. **The reporter adapter**: G2 unchanged; the K2a adapter (or a stronger one) used only for the status token at
   serving time; then the final benchmark for "G2 + reporter". Cheap (one eval session, about $6-8), honest about its
   scope.
3. **Stop stage (b) here** and write up iterations 13-19 with the baseline as the headline.
