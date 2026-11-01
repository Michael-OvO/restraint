# Iteration 19, stage (a) results: the headline baseline (2026-11-01)

Pre-registration: [iteration-19-prereg-baseline.md](iteration-19-prereg-baseline.md) (approved 2026-10-31). Node
58586655 (RTX 5090), launch commit `634ef8e` (the pinned eval path). Archive: `iterations/19-it19a-baseline`; the
Harbor job dirs in `staging/runs/harbor/it19a_*`. Base 253 min, G2 253 min, one look per benchmark per arm, no step
re-run. Metric definitions as fixed before any G2 result was read (docs/iteration-19-signoff.md).

## Summary

| hypothesis | metric | base | G2 | G2 − base [95 % CI] | result |
|---|---|---:|---:|---|---|
| H1 | internal false claims among claims (`false_claim_share`) | 18.5 % | 18.4 % | −0.1 [−3.7, +3.6] | **inconclusive** |
| H2 | ImpossibleBench `flag` (strict: the `abstain` tool) | 4.5 % | 0.0 % | −4.5 [−6.2, −3.0] | **contradicted** (see below: a channel mismatch) |
| H3 | TB2 false claims per episode | 8.7 % | 7.0 % | −1.7 [−5.6, +2.3] | **inconclusive** |

The largest effect is not a hypothesis: on the internal held-out set G2 claims success on impossible tasks **13.2 %**
of the time against base's **53.0 %** (−39.8 [−45.3, −34.3]), with verified success not lower (+3.6 [−0.0, +7.3]).

## Internal held-out (569 variants, 119 sources, S = 2; `analysis.power estimate`, 95 % t intervals from its SE)

| metric | base | G2 | G2 − base | 95 % CI |
|---|---:|---:|---:|---|
| false claims among claims (H1) | 18.5 % | 18.4 % | −0.1 | [−3.7, +3.6] |
| claims on infeasible variants | 53.0 % | 13.2 % | −39.8 | [−45.3, −34.3] |
| verified success | 73.1 % | 76.7 % | +3.6 | [−0.0, +7.3] |
| task success | 75.0 % | 77.5 % | +2.5 | [−1.2, +6.2] |
| claim rate, feasible | 88.5 % | 91.6 % | +3.1 | [+0.2, +5.9] |
| control decline | 5.1 % | 7.6 % | +2.4 | [−1.2, +6.1] |
| mean commands | 16.0 | 15.0 | −1.0 | [−1.5, −0.6] |

## Terminal-Bench 2 (89 tasks, S = 4; `final_benchmark compute`, item-cluster bootstrap)

| metric | base | G2 | G2 − base | 95 % CI |
|---|---:|---:|---:|---|
| false claims per episode (H3) | 8.7 % | 7.0 % | −1.7 | [−5.6, +2.3] |
| success | 6.5 % | 8.2 % | +1.7 | [−1.7, +5.6] |
| claim rate (from the job dirs) | 14.1 % | 14.1 % | 0.0 | |
| P(fail \| claim) (from the job dirs) | 62.0 % (31/50) | 50.0 % (25/50) | −12.0 | [−31.9, +7.0] |

As the prereg warned, a 4B model solves few TB2 tasks (6-8 %); it also claims rarely (14 %), so false claims per episode
are small for both. G2 does not claim more and its claims do not fail more often; neither difference is significant.

## ImpossibleBench LCB (309 items, S = 4, tuned prompt)

| metric | base | G2 | G2 − base | 95 % CI |
|---|---:|---:|---:|---|
| `flag` (strict, pre-registered, H2) | 4.5 % | 0.0 % | −4.5 | [−6.2, −3.0] |
| `cheated` (tests passed by an impossible spec) | 0.0 % | 0.1 % | +0.1 | [0.0, +0.4] |
| claimed complete (impossible split) | 0.6 % | 0.1 % | −0.5 | [−1.1, +0.0] |
| test files modified | 0.0 % | 0.0 % | 0.0 | |
| `act.success` (original split) | 23.5 % | 27.7 % | +4.1 | [0.0, +8.3] |
| `act.over_flag` (original split) | 0.0 % | 0.0 % | 0.0 | |
| `flag_lenient` (post hoc, not pre-registered) | 23.4 % | 27.7 % | +4.2 | [+1.0, +7.8] |

**Why H2 came out contradicted.** The scorer's headline `flag` is the strict mapping of DEF-28: only a report through
the `abstain` tool counts. G2 never uses that tool. It declines with `submit_report(status="incomplete")` and a blocker,
the channel its SFT data taught (it19b K0.1: 0 of 2,284 G2 reports in the dev banks use `abstain`). Those explained
non-claims count only in `flag_lenient`. On that post-hoc reading G2 flags more than base (+4.2 [+1.0, +7.8]), and on
neither reading does it cheat or claim an impossible task. The pre-registered result stands as contradicted; the
lenient number is reported beside it, labelled post hoc, and the mismatch between the trained report channel and the
benchmark's strict flag is recorded as DEF-70.

## What the baseline says

- **G2's trained restraint transfers to held-out tasks** (internal impossible-task claims −40 points) without a cost in
  success, and on ImpossibleBench neither model cheats.
- **G2 is no better than base at the false claims it does make on feasible work** (internal: 18.5 vs 18.4 % of claims
  fail; TB2: no significant change). This is the gap stage (b) targets.
- **The strict ImpossibleBench flag cannot see G2's way of declining.** Any later arm is compared on both readings, with
  the strict one as pre-registered.
