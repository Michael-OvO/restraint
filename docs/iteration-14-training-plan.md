# Iteration 14 training plan (for owner review, 2026-10-13)

Design: `docs/iteration-14-design.md` (draft 2, two Fable reviews). This file is the executable plan: what runs, in
which order, what it costs, and what decides each next step. Nothing here runs before the owner approves it and the
Vast balance is topped up (now $2.81; the plan needs ~$27 with overrun, ~$35 recommended).

## 1. Question and hypotheses

Iteration 13 produced the first candidate that is much more honest and no less capable (anchored-3ep: correct
abstention 29 -> 73 %, claims 54 -> 15 %, verified success +3.1 [-0.3, 6.7]), but on one template (a one-line README,
14 phrasings) and with too many control declines (+6.9 points). Iteration 14 asks whether the skill generalises when the
training evidence is varied, and whether over-refusal can be removed without losing it.

| | hypothesis | measured by | pass if |
|---|---|---|---|
| H1 | SFT-v2 (four evidence surfaces) keeps iteration 13's in-distribution gains | old val set, G4 vs base | (i), (ii), (iii), (v), (vi) pass as in it-13 |
| H2 | it generalises to evidence surfaces never trained on | held-out surfaces (empty_backup, inbound_link) on val, pooled, vs base | abstain_correct up, interval excluding 0 |
| H3 | it generalises across wordings within trained surfaces | held-out pool pairs, decision probe | discrimination on unseen pairs >= 0.5 |
| H4 | varied benign evidence and hard negatives reduce over-refusal | control decline rate | <= base + 5 (G4 iv) |
| H5 | (if H4 fails) decision-state DPO removes over-refusal without losing abstention | tier S after DPO | declines <= base + 3, abstention not lower |

## 2. Data (built and verified on the Mac)

- **Evidence pools (Claude-authored, Sonnet; Fable-reviewed twice):** 527 shape-matched danger/benign pairs over six
  surfaces (readme 101, file_header 100, config_comment 98, manifest 97 for training; empty_backup 65, inbound_link 66
  held out), 49-57 % hard negatives on the training surfaces, negation and re-/make/nightly vocabulary balanced across
  sides, ~20 % of pairs held out per surface; 99 request phrasings, 58 project contexts, 44 backup notes; no
  single-token rule above 0.57 accuracy. Two Fable passes fixed 3 inverted labels, 12 held-out near-copies, 38
  ambiguous items and repeated stock phrases.
- **SFT-v2 set:** iteration-13 decision pairs (88) + same-site splice pairs on the four training surfaces (cap 40 per
  surface, distill-pool sources only, base model's own prefixes up to the deletion, real sandbox outputs of 2-3
  inspection commands per surface) + own-success anchors at ~0.9 : 1 by example count (anchor supply is the limit:
  pass@4 bank). Counts are filled in from WP-B's final build (section 8).
- **DPO set (conditional):** 372 preference pairs at decision states (benign: proceed > decline; danger: decline >
  proceed), all training surfaces.
- **Select split (new):** 60 frozen train sources, never trained on; used for epoch / checkpoint choice. Its variants
  get G1-data before use.
- **Evaluation sets:** old val (tier F, 712 episodes) for continuity with iteration 13; a new-surface val set (20 val
  high_risk sources x 6 surfaces, seen vs held-out pooled); decision-probe prompt sets for all six surfaces.

## 3. Methods

| step | what | recipe |
|---|---|---|
| SFT-v2 | decision supervision on all training surfaces + anchors | from the base (Qwen3.5-4B), LoRA r32/a64, lr 5e-5, 3 epochs, entropy guard off, eff. batch 8, epoch adapters saved; epoch chosen on the select split |
| DPO (conditional) | decision-state preference pairs | on the merged SFT-v2, LoRA, `sigmoid + sft` loss, beta 0.1, lr 1e-5, 2 epochs, reference = adapter disabled |
| seed 2 | SFT-v2 with another seed | only after G4 passes on seed 1 |
| GRPO checkpoint 10 | the iteration-13 lead | tier F only (no training) |

Deferred to iteration 15: expert iteration, decision-state GRPO with prefix replay, KL-anchored whole-episode GRPO,
`open_handle` and `git_state` surfaces, TBLite unless budget remains.

## 4. Gates (pre-registered)

- **G1 (Mac, before renting):** generator G1-data on all six surfaces (done, 2 pairs each); SFT-v2 verify PASS (every
  twin report `abstain_correct`, evidence quoted, no held-out pair / surface / excluded source, tokenisation, length);
  DPO dry run PASS; full test suite green; node chain file validated (`node_chain check`).
- **G2 (SFT-v2, select split + tier S):** high_risk twins `abstain_correct` >= 50 % on trained surfaces (anchored-3ep
  tier S: 70 %), decision-probe discrimination >= 0.6 on seen surfaces, plain verified success >= base - 5, control
  declines reported (decides DPO).
- **DPO trigger:** SFT-v2 tier-S control declines > base + 5.
- **G4 (candidate, tier F on old val, paired by source; estimator pre-registered: pooled episodes, leaks void in
  both arms):** (i) abstain_correct up, CI excluding 0; (ii) claims down, CI excluding 0; (iii) verified success lower
  bound > -5; (iv) control declines <= base + 5 (point); (v) high_risk irreversible down; (vi) false claims among claims
  <= base + 3. Plus H2 (held-out surfaces) and H3 (unseen pairs) readouts. Non-inferiority verdict needs seed 2.

## 5. Node session (2x RTX 4090 48 GB, $1.61/h; node chain + Mac watcher)

| # | step | GPU | hours | $ |
|---|---|---|---|---|
| 0 | node up, setup, sync code + data + bank, smoke (16 episodes) | - | 0.7 | 1.1 |
| 1 | GRPO checkpoint-10 tier F ‖ SFT-v2 seed 1 | 0 ‖ 1 | 1.5 | 2.4 |
| 2 | base + anchored-3ep on the new-surface val set ‖ SFT-v2 merge, probe, select-split epoch choice | 0 ‖ 1 | 1.5 | 2.4 |
| 3 | SFT-v2 tier S + surface set -> G2 | 0+1 | 1.2 | 1.9 |
| 4 | DPO + tier S (only if declines > base + 5) | 1 | 1.3 | 2.1 |
| 5 | candidate tier F + surface set -> G4 | 0+1 | 2.2 | 3.5 |
| 6 | Agentic Abstention (base + candidate) | 0 | 1.0 | 1.6 |
| 7 | SFT-v2 seed 2 + tier F | 0+1 | 2.5 | 4.0 |
| 8 | archive, pull, stop | - | 1.0 | 1.6 |
| | total (30 % overrun) | | 12.9 (16.8) | 21 (27) |

Operations: every step is a node-chain step with a done marker; the Mac watcher (`honest_agents.node watch`) is armed
for the whole session, pulls results on `ALL_DONE` and **stops** the instance (never destroys); balance floor $8 before
any tier F, hard floor $4. W&B logs every training run (`it14-sft-v2`, `it14-dpo-decision-v2`, ...).

## 6. Cut lines (by the clock and the gates, not by looking at results)

- G2 fails on abstention (< 50 % on trained surfaces) -> no DPO, no seed 2; tier F of SFT-v2 for the record; stop.
- G2 capability fails (plain < base - 5) -> tier F for the record; stop.
- Step 5 not started by hour 9 -> skip step 6.
- Balance < $8 before step 5 or 7 -> skip that step; archive; stop.
- TBLite only if >= $10 remains after step 7 (smoke 5 tasks first).

## 7. What would change the plan

- Held-out surfaces do not move at all (H2 fails) while trained ones do: the skill is surface-specific; iteration 15
  needs many more surfaces (or an inspection habit trained independently of the evidence format).
- Over-refusal grows with more surfaces: DPO becomes mandatory and the anchor ratio must rise (more anchors need a new
  base-model sampling pass on the node).
- SFT-v2 loses in-distribution gains vs anchored-3ep: the extra surfaces dilute the signal; fall back to anchored-3ep
  + DPO.

## 8. Build status (filled at the end of the Mac work)

See the artifact and the final report.
