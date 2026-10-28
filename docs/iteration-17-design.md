# Iteration 17 design: calibrated confidence in the completion report (DRAFT)

Follows [iteration-17-proposal.md](iteration-17-proposal.md) (owner, 2026-10-28: "Sure let's do it. Make sure we get
every detail right"). This file settles the method before the pre-registration; it changes the proposal's training
method (section 3 below), so it needs the owner's read before any GPU spend.

## 1. What is trained, and what must not change

**Trained:** the number G2 writes in `confidence`, its probability that the task is fully and correctly done (the
report schema's definition, `harness.SYSTEM_PROMPT`). **Not trained:** the actions, whether and when to report, the
report's status, blocker and text. The guards measure exactly these (section 5).

## 2. Why not a scoring-rule term in the online reward

Any proper scoring rule S(c, y) pays a calibrated agent its expected score at its own pass probability q, and that
expectation is best where the outcome is certain. For Brier, `-w(c - y)^2` costs a calibrated agent `w q (1 - q)`,
which is 0 at q = 0: quitting makes the outcome certain. The repository already records this (`RewardConfig.brier_weight`
must stay below `success_bonus`; docs/design.md, docs/proposal.md INC-4), and the guard only keeps trying strictly
better than quitting: the margin for effort on a low-q task shrinks from `lambda q` to `q (lambda - w (1 - q))`.
Two alternatives checked on paper and rejected:

- a relative Brier against the group's leave-one-out pass rate: a quitter's certain 0 beats the group's mixed baseline,
  so it is paid **more** for quitting;
- subtracting the score's own expectation at c: no longer proper (for Brier the optimum moves to c = (2q + 1) / 4).

The credit for the confidence term reaches the actions because one policy writes both, so a calibration reward in the
online loop always prices effort.

## 3. Method: a confidence-only bandit on the model's own transcripts

- **Prompt:** an archived G2 episode up to and including its own report turn, with the report's tool-call arguments
  cut right after `"confidence": ` (G2 writes `confidence` last: 933 of 1,599 it16b reports put it at position 7 of 8).
  A report without a confidence (624 of 1,599) gets the key appended in the same place. Nothing else of the report
  is changed.
- **Sample:** K continuations of the number only (stop at `,` / `}`); a continuation that does not parse as a number
  in [0, 1] scores the worst Brier (1.0).
- **Reward:** `-(c - y)^2`, y = the archived episode's `passed` (hidden tests pass, no violation, no tamper; 0 on an
  infeasible variant). Within a group all samples share y, so the GRPO advantage is the Brier difference; over the
  training distribution the optimum is c = P(y = 1 | transcript, report). Brier, not log score: bounded, no clipping.
- **Loss:** only the sampled number tokens (the prompt, the transcript and the rest of the report are context). This is
  the narrowest target available; iteration 4's report-only SFT (loss on the whole report at end states) taught the
  model to report at once (zero commands in 82 % of feasible episodes, docs/training-log.md), which a number-only
  loss does not train. The guards still check it (section 5).
- **Start:** G2 seed 1, LoRA, KL to G2. On-policy in the sense that matters: the transcripts and reports are G2's own.
- **Estimator (2026-10-28):** the expectation over the value is computed exactly, not sampled
  (`train.confidence_exact`): Qwen3.5 writes the value one digit per token, so one forward pass over `prompt + "0."`
  gives the probability of every value to a tenth; the same objective as GRPO's, with no sampling variance and about
  8x cheaper (GRPO with 8 samples over 5k-token prompts would take about 10 h per epoch). Tested against brute force
  and against sampling (`tests/test_train_confidence_exact.py`).

## 4. Data

Training-pool sources only, every exclusion of it16 (`select_split.excluded_from_commit_training`: select, val, test
and the final decision probe's sources, DEF-63).

| source | episodes | informative (feasible claims) |
|---|---|---|
| it16b bank, G2 S = 4, 49 non-probe sources | 855 | about 430 (312 pass, 114 fail on mined claim points) |

About 430 informative claims on 49 sources is thin for a 4B LoRA. **Planned:** a fresh G2 rollout with bank on more
training-pool tasks (the it16b job shape, about $5), sized at build so that the claims hold at least ~150 failures.
Twins and declines stay in (their y is 0; low confidence on an `incomplete` report is part of calibration) but are
reported apart, since a model can calibrate them from its own status alone.

## 5. Evaluation

**Power decides the design** (`analysis.calibration power`, 2026-10-28, on the it16 select runs). The unit is a
feasible claim; failure is mostly a property of the task (the same model's other sample predicts it at AUROC
0.79-0.84, other models' claims at 0.86-0.92), so more samples of the same tasks buy little: the source-cluster SE of
a claims AUROC shrinks as S^-0.05 to S^-0.34, not S^-0.5. On the select split (60 sources) the minimum detectable
AUROC difference (80 % power, one-sided 0.05) is 0.21 at S = 2, 0.16 at S = 4 and 0.13 at S = 8. Only more sources,
or a paired design, buy power.

**Primary evaluation: paired, on the same episodes.** The actions are not trained, so the trained model's confidence
can be scored on the very claims G2 made: for every held-out G2 episode, its own report cut at `confidence` (exactly
the training prompt) is completed by the calibrated model (`num_samples` draws, their mean as the stated value, and
the first draw alone as a check), and the two confidences of the same claim are compared against the same hidden-test
outcome. The episodes:
- G2's banked select-split run (`iterations/16-it16c-g2-select`, 60 sources, S = 2, its transcripts kept);
- training-pool sources held out from the confidence training by source (`data.heldout_fraction`, decided at build
  so the dev set reaches about 120 sources with the select split).
Endpoints on these claims, calibrated − G2, with a paired source-cluster bootstrap (90 % CI):
- **primary:** AUROC of confidence for failure among feasible claims (G2 0.518 on the select split);
- **secondary:** Brier and ECE on the same claims; precision of claims stated at ≥ 0.9; the share stated at ≥ 0.9.
A reference row scores the out-of-fold transcript-feature model on the same claims (AUROC 0.55 on the it16 select
runs; the strongest single feature, the number of commands, 0.69), so a gain is read against what simple features
already give.

**Online check: the actions are unchanged.** The calibrated model is also served and run on the select split (S = 2,
as it16c): verified success, task success, claim rate on feasible and infeasible variants, control decline,
zero-command report rate and mean commands, each non-inferior to G2 with margins from `analysis.power` at the select
split's K and S. The online confidences are scored too (unpaired; reported, the power above applies), as a check that
the confidence the model writes in a live episode behaves as the paired score says.

**Final benchmark:** only after the select-split gate passes, as in it16 (prereg 10.3 / 10.4).

## 6. What is built before any GPU spend

1. A typed CLI for the scoping analysis and the reference predictor (`analysis`), replacing the scratch scripts.
2. The bandit's data builder (prompt cut at `"confidence": `, exclusions, labels) with tests on real bank rows,
   including the chat template's rendering of a cut tool call.
3. The trainer (GRPO over number-only completions, loss mask, KL), with a CPU smoke run on a tiny model.
4. Calibration metrics: AUROC and Brier exist (`eval.metrics.calibration`); add claim-only variants and ECE, and the
   power of their differences at the select split (`analysis.power`).
5. The pre-registration (`docs/iteration-17-prereg.md`), L1 on the harness commit of DEF-65, the ledger clear.

## 7. Owner decisions

- 2026-10-28, "Yes, revised method (Recommended)": the method of section 3 replaces the proposal's online RL, and the
  data rollout of section 4 is planned (its GPU spend is asked for separately, as is the training run's).
