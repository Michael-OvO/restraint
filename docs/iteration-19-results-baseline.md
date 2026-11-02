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

## Addendum 2026-11-02: the internal held-out after the DEF-74 re-grade

Stage a2's `def74_regrade` replayed the remove_input control episodes the build's own backup had failed (`runs/def74`):
base 2 (1 changed), G2 4 (4 changed), each from `fabrication` to `success_claimed`. Re-scored with
`analysis.power estimate` on the 562-variant set (DEF-72's spec defect and the other DEF-66 sources out by default; the
569-variant table above includes them):

| metric | as run: base / G2, G2 − base [95 % CI] | after the re-grade |
|---|---|---|
| false claims among claims (H1) | 18.1 / 17.8 %, −0.4 [−3.4, +2.7] | 18.0 / 17.2 %, **−0.8 [−3.9, +2.3]** |
| claims on infeasible variants | 53.1 / 12.9 %, −40.2 [−44.9, −35.6] | unchanged |
| verified success | 73.4 / 77.2 %, +3.8 [+0.7, +6.9] | 73.5 / 77.8 %, **+4.3 [+1.1, +7.4]** |
| task success | 75.3 / 77.9 %, +2.6 [−0.5, +5.8] | 75.4 / 78.5 %, +3.1 [−0.1, +6.3] |
| claim rate, feasible | 88.4 / 91.6 %, +3.2 [+0.9, +5.6] | unchanged |

H1 stays inconclusive; no conclusion above changes. The re-graded runs are the reference for it19c's final benchmark.

## Addendum 2026-11-02: Terminal-Bench 2 under the external budget (stage a2, DEF-71)

Amendment 2026-11-01b: 50 turns, 49,152 tokens per episode, 8,192 per turn, a 1,024-token report reserve, Harbor
concurrency 6 for both arms (the base arm on instance 58779312; G2 on 58815330, its last 66 trials resumed on a 98 GB
VM, 58845540, after Harbor's own process was twice OOM-killed; docs/iteration-19-signoff.md). One look per arm;
`final_benchmark compute`, item-cluster bootstrap, 89 tasks, 354 units each:

| metric | base | G2 | G2 − base [95 % CI] |
|---|---:|---:|---|
| false claims per episode (H3) | 16.1 % | 14.1 % | −2.0 [−6.3, +2.8] |
| success | 11.0 % | 11.3 % | +0.3 [−2.6, +3.4] |

After DEF-76: four of G2's trials were graded on a full disk and re-run (the table is the re-graded one; as first
graded, G2 was 13.6 %, −2.5 [−7.1, +2.3]). **H3 stays inconclusive** under the external budget, as it was under the training budget (−1.7 [−5.6, +2.3]). Both arms
reach a report far more often than at it19a, and claim more (base 14.1 % -> 26.0 %; docs/iteration-19-research/
badcase-it19a2/base_tb2.md). ImpossibleBench (H2) is still running for both arms.
