# Iteration 18 design: calibrated confidence with the actions anchored (DRAFT)

Follows [iteration-17-results.md](iteration-17-results.md) (NO-GO: the primary passed, the infeasible claim-rate guard
failed). The owner, on the cause: "I thought we had kl divergence no?". Iteration 17's KL covered the value's token
tree only; the LoRA weights are shared by every token, so the actions drifted. This iteration changes that and nothing
else.

## 1. What changes from iteration 17

| | iteration 17 | iteration 18 |
|---|---|---|
| items, held-out split, start model, LoRA, learning rate, epochs | frozen (it17 prereg section 9) | **the same files and sha256s** |
| loss | expected Brier + 0.04 x KL to G2 on the value's tree | the same, **+ 1.0 x KL from G2 over G2's own action tokens** |
| drift monitor | none | held-out action KL per token at every evaluation |
| abort | none | **stop when held-out action KL per token > 0.02** |
| guards | seven online guards | the same seven, **+ read-before-destruction on twins and on controls** |

One change to the method (the anchor), so a different outcome can be attributed to it. The label mechanism that
iteration 17's results name as likely (a held twin is y = 0, a deleted control y = 1) is left in place on purpose:
the anchor must hold the actions **under** that pressure, which is the claim to test. If it cannot, changing the
labels is the next experiment, not part of this one.

## 2. The anchor

`train.confidence_exact`, `exact.anchor_beta` (commit `b8084a7`):

- **Which tokens.** Every assistant-turn token in the item's transcript before the report's `confidence` value
  (`assistant_positions`, found from the chat template's markers): the commands, the reads, the reasoning, the
  report's status and the text before the value. The tokens whose behaviour must not change.
- **The estimate.** On up to `anchor_max_tokens` = 512 positions sampled per item (a stable sample per item), the k3
  estimator of KL(policy || G2) at G2's own next token, `exp(r) - 1 - r` with r = log p_G2 - log p_policy, averaged
  and multiplied by the transcript's action-token count. So the term is a per-transcript KL (it17's adapter: 134 nats
  per transcript) on the same scale as the expected Brier (at most 1) times `anchor_beta`.
- **The cost.** The same two forwards per item as iteration 17 (G2 with the adapter off, the policy): the logits are
  kept at the three value nodes and the sampled action positions in one pass (`logits_to_keep` as an index tensor,
  identical to a full pass, tested). The training time is about iteration 17's (146 min).

`anchor_beta` = 1.0 was chosen before any GPU run: at 1.0 an adapter that drifts as far as iteration 17's (134 nats
per transcript) pays about 130 times the largest Brier it could save, while the value itself needs a few nats per
item. It is fixed in the pre-registration; it is not tuned on the dev set.

## 3. Drift monitor and abort

At every evaluation (`eval_steps` = 100, and at each epoch end) the trainer logs, on the 200 held-out items, the
action KL from G2 per token and per transcript (`eval/action_kl_per_token`, `eval/action_kl_per_transcript`), whatever
`anchor_beta`. If the per-token value exceeds `abort_action_kl_per_token` = 0.02 the run stops (no `final` adapter;
the chain stops and the watcher stops the node). 0.02 is a ninth of iteration 17's measured drift (0.18 per token,
`confidence_exact --drift` on its adapter, 40 held-out items). The abort is a stopping rule, not a selection: no
checkpoint is chosen by it.

## 4. Checked before any GPU spend

- **The "actions cannot change" claim, tested.** CPU mechanism test (Qwen3-0.6B, 60 items, learning rate 2e-4, 24
  optimiser steps, anchor off vs on): held-out action KL per token at the end **0.959 off, 0.026 on** (37x lower);
  held-out expected Brier 0.105 at step 0, 0.000 off and 0.0003 on (the anchor does not stop the value from being
  learnt). The test ran at four times it18's learning rate on a model a seventh of the size, so 0.026 being above
  the 0.02 abort is not a forecast for the GPU run; the run's own monitor is.
- **The read guards, backtested.** `analysis.calibration read-guard` on iteration 17's select runs (G2 re-run vs
  G2-cal): read-before-destruction on twins 87.5 → 65.8 % (−21.7 points, 90 % CI [−30.6, −12.7], margin 12.9) and on
  controls 80.0 → 60.8 % (−19.2 [−29.1, −9.2], margin 14.3). The it17 adapter passes neither, so the guard would
  have caught it on its own.
- **The same split.** `train.confidence --dry-run` on `configs/confidence/it18.yaml`: 2,422 training items on 159
  sources, 757 held out on 46, the cells of it17 prereg section 9 exactly.
- **The reference arm is reusable.** Nothing on the evaluation path (harness, env, grading, reward, verifier,
  `eval/`, the eval scripts, the serving config, the select protocol and split) changed between the it17b run
  (after `3cdb17c`) and this design (`git diff 3cdb17c` on those paths is empty). G2's it17 rescore rows and select
  re-run are therefore the reference arm again, as the it17 prereg would have produced them today (section 5 of the
  prereg says how this is checked at launch).

## 5. Cost

One stage, it18b, on the stopped it17 node (58122470, RTX 5090, $1.11/h; G2 merged and the Docker images already on
it): training about 2.5 h, the candidate's rescore about 15 min, its select run about 30 min, merges and sanity
evals about 20 min. About 3.5-4 h, about $4-5. If the abort fires, the chain stops after training.

## 6. Owner decisions

- (pending) this design, the pre-registration ([iteration-18-prereg.md](iteration-18-prereg.md)), the waivers the
  ledger needs for it18b, and the GPU spend.
