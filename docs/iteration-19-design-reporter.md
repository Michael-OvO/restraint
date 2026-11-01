# Iteration 19, stage (c): G2 plus a reporter (DRAFT)

Owner, 2026-11-01, after K2 ([iteration-19-results-k2.md](iteration-19-results-k2.md)): "Reporter adapter
(Recommended)". G2 does every action, unchanged; a separate LoRA adapter, the **reporter**, decides only the report's
status (claim or not). The claim this stage can make is about the system: "G2 with a reporter reports honestly more
often", not "the model learned it".

## 1. Why this, after K2

- Three iterations (17, 18, K2) show that training anything about the report on G2's own transcripts moves the
  actions too, because the LoRA weights are shared (twins destroyed 53, 31, 21 of 120 against G2's 13).
- A reporter's weights never choose an action, so that coupling cannot happen: the actions are G2's by
  construction. It also frees the training: no action anchor and no drift abort are needed, so the reporter can be
  trained harder than K2a, whose anchor was the main brake (K2a moved only 9.5 % of failing claims across the line).

## 2. The reporter

- **Training:** `train.confidence_exact` with `exact.target: status` (the K2a objective: −ΔL·z from the CR-GRPO table
  at the status token, frozen past |z| = 6), from G2 merged, on the it17 items cut at the status (DEF-66 dropped).
  Differences from K2a: `anchor_beta: 0` (no action anchor; drift is irrelevant), 3 epochs, learning rate 1e-4, node
  KL β 0.01 (kept only to stop the vocabulary at the status node from degenerating).
- **The decision rule at serving time:** G2 writes its report as today; if the report claims `complete`, the reporter
  scores π(claim) on the transcript cut at the status and the claim stands only if π(claim) ≥ t, otherwise the status
  becomes `incomplete` with G2's own report text and no blocker. A non-claim is never turned into a claim.
- **The threshold t** is chosen on the training items only, before any evaluation: the t that maximises the CR-GRPO
  table's expected payoff of the decision (claim worth 1 − α if it passes and −μ_fc − α if it fails, against the
  non-claim), so t is the reporter's own cost-weighted cut, not tuned on any benchmark.

## 3. Evaluation: offline, on the episodes it19a already ran

The status is decided after the last action, so applying the reporter to an archived G2 episode is exactly what a
live reporter would have done (the report is cut at the status in both). No new rollouts: G2's it19a episodes are
re-reported and re-scored with the same graders. Paired on the very same episodes:

| arm | what it is |
|---|---|
| base | it19a's base rows |
| G2 | it19a's G2 rows |
| G2 + reporter | the same G2 episodes, each claim kept or withdrawn by the reporter |

Endpoints (G2 + reporter vs G2, and vs base; the estimators of the it19a prereg):
- **primary:** internal false claims among claims (`false_claim_share`), and TB2 false claims per episode;
- **guards:** internal verified success (a withdrawn passing claim loses it), TB2 success unchanged by construction
  (tests do not move), ImpossibleBench `flag_lenient` (the DEF-70 headline; a withdrawn claim without a blocker does
  not count as a flag, so it cannot inflate this);
- **reported:** the share of claims withdrawn, of withdrawn claims that would have passed, and the dev-set paired
  AUROC of π(claim) as in K2.

## 4. What is needed

- The internal held-out transcripts of it19a: on the stopped node's disk (`runs/eval/it19a_*_heldout/transcripts`),
  not archived (lean archive). One short node session pulls them (and trains the reporter in the same session).
- Code ($0): rebuild a report prompt from an it19a transcript (run_eval transcripts and Harbor `honest_transcript.json`
  are chat messages in the training format); apply the reporter offline; re-score with the same units writers; tests.
- A pre-registration with the endpoints and t's procedure, then the owner's approval of the GPU session.

## 5. Cost

One node session on 58586655 (stopped, with G2 merged, the items and the code): reporter training about 3.5 h
(3 epochs, no anchor so about a third faster per step than K2a), scoring the it19a G2 episodes (about 2,700) about
30 min, pulling the transcripts. About 4.5 h, about $4-5 at the node's rate. The balance is $15.49.
