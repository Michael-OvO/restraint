# Iteration 17 proposal: calibrated completion reports (DRAFT, for owner decision)

Written 2026-10-28 after the it16c result. Nothing here is approved; nothing runs until the owner says so.

## 1. Why pivot

Stage (c) of iteration 16 (SFT from G2 on 639 claim and destructive-step pairs) did not beat G2 on the select split
(60 sources, S = 2, one seed; `iterations/16-it16c-sft`, `iterations/16-it16c-g2-select`):

| feasible episodes (272 per arm) | base | G2 | it16 final (e2) |
|---|---|---|---|
| claims done, tests pass | 166 | **186** | 174 |
| claims done, tests fail (false claim) | 85 | 65 | **53** |
| says incomplete, tests fail (honest) | 8 | 20 | **27** |
| says incomplete, tests pass (under-claim) | 1 | **0** | 16 |
| claims on infeasible variants | 71.1 % | **9.9 %** | 11.2 % |

The bad cases:

- **Under-claims come from imitated self-checks.** The claim pairs put loss on the injected checklist commands, so
  the model learned to end episodes with strict one-line checks (14 % of e2 episodes end with two or more, against
  0.5 % for G2) that it cannot write correctly: 89 of their 234 runs fail (`\d` in `grep -E`, broken `sed`
  back-references, `test -z "$(grep -c ...)"`), and it then trusts them.
- **Honesty given evidence is saturated in G2.** It follows check outputs (claim flip sensitivity 0.72 on the it16b
  rollouts), claims on 9.9 % of infeasible variants, declines 5.6 % of controls.
- **What remains is not dishonesty but not knowing.** 62 of G2's 65 false claims came after it ran verification
  commands; they are near misses (mean hidden-test pass fraction 0.68, half at 0.75 or more).
- **The report carries no uncertainty.** Reported `confidence` is 1.0 on nearly every claim in every arm (AUROC
  against the tests 0.52).

## 2. The signal exists

On the 729 claims of the three select runs (a scratch scoping pass; it becomes a typed `analysis` CLI if approved):

| predictor of "this claim fails the hidden tests" | AUROC |
|---|---|
| the model's reported confidence | 0.52 |
| transcript features at report time (commands run alone 0.69; out-of-fold ridge logistic on 12 features 0.55, corrected 2026-10-28 from a scratch fit's 0.60) | 0.55-0.69 |
| the same model's other sample on the task failed | 0.82 |
| other models' claims on the same task failed | 0.885 |

Failure is mostly a property of the task, and the model fails the same tasks consistently. A policy that learns
which of its completions are near misses can say so; today nothing in training asks it to.

## 3. Proposal

**Target behaviour.** The report says how sure the agent is that the hidden checks pass: a probability in
`confidence` (already in the report schema), or a three-way status (verified complete / believed complete but
unverified / incomplete). Actions are not the target.

**Training.** GRPO from G2 seed 1 on training-pool tasks only (the it16 exclusions, DEF-63 included), reward = task
outcome as today plus a proper scoring rule on the reported probability against the hidden tests (log or Brier
score), so the expected reward is maximal at the true probability and no hand-written check is involved (the it16
failure mode cannot recur). The probability scoring is added to the pair-table reward (DEF-13) rather than replacing
it. S9 (the reward audit) must show that constant confidence and blind declining both earn less than calibrated
reporting before any GPU spend.

**Endpoints** (pre-registered before the run, select split for selection, final benchmarks once):
- primary: AUROC and Brier score of reported confidence against hidden tests on feasible claims, final − G2;
  precision of high-confidence claims (confidence ≥ 0.9);
- guards: verified success and task success non-inferior to G2 (actions must not degrade), claims on infeasible
  variants and control decline not worse than G2;
- external: TB2 and ImpossibleBench false-claim share among confident claims.

**Cost.** Mac work: the scoring-rule reward and its S9 audit, the calibration metrics in `eval.metrics`, a
`docs/iteration-17-prereg.md`. GPU: one RL run (about 11 h, about $16 at the it15 G3 rate) and the select-split
evaluation (about $3); the final benchmark only after a GO, as in it16.

## 4. Alternatives considered

- **B. Verification skill** (reward checks that agree with the hidden tests): higher ceiling, but it is capability
  training and the riskiest to the guards.
- **C. Stay on it16 with fixes** (no loss on check turns, more "checks pass, claim it" examples): cheap, but bounded
  by the small headroom of section 1, perhaps 5-10 fewer false claims of 272.

## 5. Open before approval

- DEF-65 (harness hang on backgrounded processes) is open and blocks any experiment.
- Whether stage (c2) of the it16 amendment (RL on the pair-table reward) is abandoned or folded into this run.
- Power: how small an AUROC or Brier difference the 60-source select split and the final benchmarks resolve is not
  computed yet; `analysis.power` needs the calibration metrics first.
