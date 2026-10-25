# Iteration 16 power: variance components, MDE for 20 vs 68 sources, clone-pass (DEF-10)

Computed 2026-10-24 with `python -m honest_agents.analysis.power report --samples 4 8` (S0.11 of
[iteration-16-direction.md](../iteration-16-direction.md)) from the archived evals, on `data/decision/surfaces_val/tasks`
(20 val sources, 240 task dirs) and the canonical inspection metric (`eval.read_diagnosis.inspected`, K = 8; see
[it15-read-metric-recompute.md](it15-read-metric-recompute.md)). Labels: **re-scored, post-fix**. Every number here is
from the archive. Nothing was run in Docker. The tool runs in about 4 s.

Runs: G2 s1 `15-g2-splice-replay/eval/it15_g2_{surface4,tierS,tierF}` (the reference), G2 s2
`15-g2-seed2/eval/it15_g2_s2_{surface4,tierF}`, SFT-v2 `15-g0-judgment/eval/it15_sft_v2_surface4` (DEF-12: the copy
under 15-g2-splice-replay is the same episodes) and `14-session-a/eval/it14_sft_v2_{tierS,tierF}`. Tier S has no second
G2 seed, so it has no seed scenario.

## What the tool does

- **Metrics.** Surface: `read` (both branches; `read_heldout`), `twin_abstain_correct`, `control_decline` and the paired
  gap (per source × surface cell, twin abstain_correct minus control decline; `gap_trained`, `gap_heldout`). Tiers:
  `false_claim_share` and `verified_success` (`episode_ratios`; verified success is over all feasible episodes, plain
  and controls). Inbound_link control *success* is not used anywhere (pre-DEF-01 grading); decline and abstain_correct
  do not depend on it.
- **Variance components.** Method of moments on per-unit values: source, evidence pool pair, cell (twin and control of
  one cell), unit and sampling. On G2 s2 − G2 s1 they are the seed interactions; on SFT-v2 − G2 s1 the model
  interactions (one seed each, so seed variance is included).
- **MDE.** Paired design, power 0.8, 90 % two-sided interval, t with K − 1 df. Three noise scenarios: `sampling` (a
  clone of the same model), `seed` (a second training seed, within lineage) and `model` (a different model). At K = 68 the
  pair term uses a uniform draw over each surface's test-split pairs (23 / 20 / 20 / 20 / 13 / 13; the 98-source build
  draws 22 / 19 / 20 / 19 / 13 / 13 of them, DEF-07). The new sources are assumed to vary like the old 20, and the new
  pairs like the old pairs. The pair term is a floor that more sources do not remove.
- **`ni10`** is the smallest non-inferiority margin that a clone of the reference fails at most 10 % of the time (sweep
  S10). It is (t.95 + t.90)·SE. A margin set *at* the MDE fails a clone 20 % of the time; the clone-pass table confirms
  this (80-81 %).
- **The estimator** to pre-register is `paired_estimate`. It takes per-unit paired differences, gives a two-way
  (source × pair) cluster-robust 90 % CI, and runs a sign-flip by source plus a sign-flip by pair. A pass needs both
  p < 0.10.

A check on the variance model: the design SE of a seed difference at K = 20 matches the estimator's SE on the archived
G2 s2 − G2 s1 pair. For example: read 2.0 / 2.0, gap 4.6 / 4.5, tier F verified success 1.9 / 1.8. On the held-out
gap the design is the conservative one (9.7 / 7.3).

## MDE table: 20 vs 68 sources (points, S = 4 samples per task)

| metric | K | MDE sampling | MDE seed | MDE model | ni10 sampling | ni10 seed |
|---|---|---|---|---|---|---|
| read | 20 | 4.3 | 5.2 | 6.3 | 5.1 | 6.1 |
| read | 68 | 2.3 | 2.9 | 3.3 | 2.7 | 3.4 |
| read, held-out | 20 | 6.5 | 10.2 | 11.4 | 7.6 | 12.1 |
| read, held-out | 68 | 3.4 | 6.5 | 6.7 | 4.0 | 7.7 |
| twin abstain_correct | 20 | 7.5 | 8.1 | 10.4 | 8.8 | 9.6 |
| twin abstain_correct | 68 | 3.9 | 4.3 | 6.1 | 4.6 | 5.0 |
| control decline | 20 | 7.4 | 8.2 | 7.8 | 8.7 | 9.6 |
| control decline | 68 | 3.9 | 4.7 | 4.4 | 4.6 | 5.6 |
| gap (all surfaces) | 20 | 10.5 | 11.8 | 14.5 | 12.4 | 13.9 |
| gap (all surfaces) | 68 | 5.5 | 6.3 | 8.9 | 6.5 | 7.4 |
| gap, trained | 20 | 12.4 | 14.8 | 19.4 | 14.7 | 17.5 |
| gap, trained | 68 | 6.5 | 8.8 | 12.2 | 7.7 | 10.3 |
| gap, held-out | 20 | 19.3 | 25.2 | 20.4 | 22.8 | 29.7 |
| gap, held-out | 68 | 10.2 | 13.3 | 10.7 | 12.0 | 15.6 |
| false_claim_share, tier S (K = 40, S = 2) | 40 | 8.6 | - | 11.8 | 10.1 | - |
| false_claim_share, tier F | 98 | 4.5 | 4.7 | 5.1 | 5.3 | 5.5 |
| verified success, tier S (K = 40, S = 2) | 40 | 8.1 | - | 11.1 | 9.5 | - |
| verified success, tier F | 98 | 4.4 | 4.7 | 4.9 | 5.1 | 5.5 |

At S = 8 (same tool, `--samples 4 8`), the K = 68 MDEs fall by about another 30 % under the sampling scenario but only
10-20 % under seed or model noise (gap 3.9 / 4.9 / 8.0, held-out gap 7.2 / 11.1 / 8.0). Interaction terms do not shrink
with samples, so spend should go into sources and pairs, not samples (as iteration-16-direction says).

Compared with the iteration-15 diagnosis, the held-out gap MDE of 18-20 at 20 sources is confirmed: 19.3 sampling-only,
25.2 within lineage. The pooled read MDE is now 4.3 at α = 0.10 two-sided, on the K = 8 canonical metric. The
diagnosis had about 6, at 5 % and on the old rule.

## Variance components (pp², G2 s1)

| metric | source | pair | cell | unit | sampling (S = 4) | seed × source | seed × pair | seed × unit | model × source | model × pair | model × unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| read | 111 | 16 | 319 | -6 | 337 | -2 | 26 | 179 | 19 | -39 | -154 |
| read, held-out | 111 | 140 | 62 | 17 | 250 | -27 | 77 | 385 | -172 | 52 | -152 |
| twin abstain_correct | 195 | -62 | - | 257 | 500 | -1 | -149 | 172 | 91 | 180 | -174 |
| control decline | 38 | 68 | - | 80 | 484 | -80 | 94 | 20 | -33 | 53 | -54 |
| gap | 155 | 103 | - | -89 | 984 | -30 | 19 | 484 | 130 | 475 | -430 |
| gap, held-out | 173 | 53 | - | -238 | 1115 | -269 | -452 | 1549 | -377 | -186 | 251 |
| false_claim_share, tier F | 254 | - | - | 543 | 283 | 29 | - | -147 | -63 | - | 172 |
| verified success, tier F | 392 | - | - | 827 | 271 | 46 | - | -82 | -11 | - | 149 |

These are moment estimates from 20 sources and one pair of seeds, so they are noisy. Negative values are reported as
computed and floored at 0 in the MDE, which makes the MDE conservative. The robust reading:
- Sampling dominates every surface metric at S = 4.
- The read rate has a large *cell* term: twin and control of one cell are read alike, as reading happens before the
  model can see which branch it is on.
- The held-out gap carries a large seed × unit term: the two G2 seeds differ on individual held-out tasks.

## Clone-pass (G2 s1 resampled against itself; G2 s2 against G2 s1), 2,000 simulations

| bar | sampling clone passes | seed clone passes | S10 |
|---|---|---|---|
| soup S1/R2 held-out part, `gap_heldout ≥ 16.2` (G2 s1's own point estimate) | 53 % | 18 % | **fails** |
| S1 trained part, `gap_trained ≥ 20.6` | 59 % | 51 % | **fails** |
| R2 trained part, `gap_trained ≥ 16.6` | 92 % | 88 % | passes |
| gap non-inferiority at ni10 (−12.4) | 91 % | 46 % | passes |
| gap non-inferiority at the MDE (−10.5) | 80 % | 29 % | fails |
| held-out gap non-inferiority at ni10 (−22.8) | 91 % | 51 % | passes |
| control decline non-inferiority at ni10 (+8.7) | 91 % | 9 % | passes |
| read non-inferiority at ni10 (−5.1) | 90 % | 13 % | passes |
| tier F verified success non-inferiority at ni10 (−5.1) | 90 % | 96 % | passes |

The full table, with every metric and guard, is the `## Clone-pass` section of the tool's output.

The seed clone takes G2 s2's archived per-task rates as its truth. That truth includes one realisation of s2's own
sampling noise and its real differences from s1, such as control decline +6.2 [+1.9, +10.6]. So the seed column says
what a within-lineage pass is worth *against this particular s1*: little, for read and control decline at K = 20.

Rules for the pre-registration (S10):
- every margin is at least `ni10` under the sampling scenario at the planned design, not the MDE;
- a bar at a reference point estimate is never used;
- the seed clone's pass rate is reported beside each bar.

## Re-scored iteration-15 comparisons (the estimator, candidate − G2 s1, 90 % CI, sign-flip p by source / pair)

| candidate | held-out gap | gap | read | control decline | twin abstain_correct |
|---|---|---|---|---|---|
| G2 s2 | −5.0 [−17.6, +7.6] | −2.3 [−10.0, +5.4] | −2.7 [−6.1, +0.7] | +6.2 [+1.9, +10.6] (p .017 / .058) | +4.0 [−0.7, +8.6] |
| soup | −3.1 [−16.0, +9.7] | −2.3 [−10.9, +6.3] | −1.8 [−4.7, +1.2] | +0.4 [−5.1, +5.9] | −1.9 [−7.8, +4.0] |
| G3 c10 | −13.1 [−24.2, −2.1] (p .028 / .106) | −1.7 [−9.6, +6.2] | −2.5 [−5.8, +0.8] | +6.7 [+1.2, +12.1] (p .033 / .053) | +5.0 [+0.6, +9.4] (p .079 / .097) |
| G3 c15 | −0.6 [−8.4, +7.2] | +0.6 [−6.4, +7.7] | +0.5 [−3.0, +4.0] | +8.8 [+3.0, +14.5] (p .010 / .007) | +9.4 [+4.6, +14.2] (p .006 / .003) |
| G4 | −8.1 [−20.0, +3.7] | −0.8 [−7.3, +5.6] | −0.1 [−3.1, +2.9] | −3.8 [−7.8, +0.3] | −4.6 [−9.1, −0.0] |
| SFT-v2 | −18.8 [−28.2, −9.3] | −17.9 [−27.1, −8.8] | −0.4 [−4.2, +3.4] | +2.7 [−1.8, +7.2] | −15.2 [−21.8, −8.6] |

- The soup verdict stands as "no detectable change": −3.1 on the held-out gap with an MDE of 19. The soup's S1/R2 *fail*
  is void as a verdict, because the bar is one that a G2 copy fails about half the time.
- G3 c10's held-out-gap drop passes the source sign-flip but not the pair sign-flip, so it is not a pass under the
  pre-registered rule.
- G3 c15's raised decline on both branches (+8.8 control decline, +9.4 twin abstain) is resolved by both tests. It is
  the iteration-15 "blind declining" finding under the canonical numbers.
- No follow-up moved read rate by more than 2.5 points, and the read MDE is about 4-5. This agrees with the
  diagnosis.

## Caveats

- One pair of G2 seeds, and SFT-v2 has a single surface run (DEF-12). Seed and model interaction components therefore
  have very few degrees of freedom.
- Clone simulations keep the archived build's sources fixed: no source resampling, so source main effects cancel. They
  resample each task's samples with a variance correction of sqrt(S/(S − 1)). A plain resample would understate
  sampling noise by 25 % at S = 4.
- The K = 68 rows extrapolate from the 20 archived sources. When surface-val-v2 exists, re-run the tool on its first
  G2 eval and replace this table with numbers from that build.
- 2,000 simulations give a Monte Carlo SE of about 0.7 points on a 90 % pass rate. A bar is flagged only when its clone
  fail rate exceeds 10 % by more than 2 MC SE.
