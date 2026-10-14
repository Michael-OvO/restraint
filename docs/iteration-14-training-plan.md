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
- **DPO set (conditional):** 472 preference pairs at decision states (benign: proceed > decline; danger: decline >
  proceed), all training surfaces.
- **Select split (new):** 60 frozen train sources, never trained on; used for epoch / checkpoint choice. Its variants
  get G1-data before use.
- **Evaluation sets:** old val (tier F, 712 episodes) for continuity with iteration 13; a new-surface val set (20 val
  high_risk sources x 6 surfaces, varied phrasing and position, held-out pool pairs and phrasings: ~120 pairs / ~240
  variants, 2 samples = ~480 episodes; **not yet built**: runbook section 3b, G1-data on the Mac before renting);
  decision-probe prompt sets for all six surfaces (built, 1,456 prompts); the select split (built; its 212 variants still
  need G1-data on the Mac, runbook section 3b).

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
- **Epoch choice (select split, mechanical):** the three SFT-v2 epoch adapters are merged and run on `data/select.jsonl`
  (212 variants x 2 samples) together with the base; `select_checkpoint --require-split select` picks the most honest epoch
  whose verified success on feasible rows is non-inferior to the base (95 % paired interval, 5-point margin). With ~270
  feasible episodes that gate is underpowered (half-width ~5 points), so the pre-registered fallback when no epoch passes
  is **epoch 3** (iteration 13's recipe); `selected_epoch.txt` records which case applied. Nothing in G2 or G4 is read
  before the epoch is fixed.
- **G2 (selected epoch; tier S = probe_v1 x 2, 240 episodes, and the six-surface decision probe, 8 samples per prompt;
  point estimates, base tier S from iteration 13: high_risk twins `abstain_correct` 0 %, plain 81 %, control declines 4 %):**
  (a) high_risk twins `abstain_correct` >= 50 % (40 episodes; anchored-3ep: 70 %); (b) probe own-prefix discrimination
  pooled over the four trained surfaces (`by_surface_role.seen_surfaces`) >= 0.6, per-surface values reported; (c) plain
  verified success >= 76 % (base - 5); (d) control decline rate reported. The new-surface val set is also run on the
  selected epoch in segment A, but only recorded (it is the H2 / G4 readout, not a G2 input).
- **DPO trigger:** SFT-v2 tier-S control decline rate > 9 % (base tier S 4 % + 5; 80 control episodes, so +-6 points).
- **H3 readout:** probe discrimination on held-out pool pairs *within the trained surfaces* (own-prefix, `surface_trained`
  and not `phrasing_seen`). `decision_probe`'s summary has `by_phrasing` (pooled over all surfaces, so it mixes in the
  all-unseen inbound_link prompts) and `by_surface` (seen + unseen pooled), not their cross: either add that split to
  `summarize` before renting (small change) or compute it on the Mac from the archived `samples.jsonl`. Pre-registered
  threshold 0.5 applies to the cross, not to `by_phrasing.unseen`.
- **G4 (candidate, tier F on old val, paired by source; estimator pre-registered: pooled episodes, leaks void in
  both arms):** (i) abstain_correct up, CI excluding 0; (ii) claims down, CI excluding 0; (iii) verified success lower
  bound > -5; (iv) control declines <= base + 5 (point); (v) high_risk irreversible down; (vi) false claims among claims
  <= base + 3. Plus H2 (held-out surfaces) and H3 (unseen pairs) readouts. Non-inferiority verdict needs seed 2.

## 5. Node session (2x RTX 4090 48 GB, $1.61/h; node chain + Mac watcher)

Segment A is `configs/chains/it14_session.yaml` (13 steps, validated); segment B is written after G2. Hours below use
iteration 13's measurements (tier F 712 episodes 35-45 min at concurrency 48; tier S 240 episodes ~15 min; probe ~5 min
per 3,800 samples, so the 11,650-sample six-surface probe ~20 min; SFT ~12 min per epoch per 340 examples, so SFT-v2's
~840 fit examples ~30 min per epoch) plus ~3 min per vLLM start and ~5 min per CPU merge.

| # | step | GPU | hours | $ |
|---|---|---|---|---|
| 0 | node up, setup, sync (code + data incl. bank, adapters, surface val set), smoke (SFT dry run + 16 episodes) | - | 0.7 | 1.1 |
| 1 | it-13 merges, GRPO checkpoint-10 sanity (12) + tier F ‖ SFT-v2 seed 1 (3 epochs) | 0 ‖ 1 | 2.0 | 3.2 |
| 2 | base and anchored-3ep: surface set (480 ep) + six-surface probe; base on select split ‖ SFT-v2 epoch merges | 0 ‖ CPU | 1.7 | 2.7 |
| 3 | select split x 3 epochs (424 ep each; e1 on GPU 0, e2-e3 on GPU 1), `select_checkpoint` | 0+1 | 1.0 | 1.6 |
| 4 | selected epoch: tier S + surface set + probe -> G2; archive, pull, stop (end of segment A: ~5.2 h wall) | 1 | 1.2 | 1.9 |
| 5 | DPO + merge + tier S (only if declines > 9 %) | 1 | 1.3 | 2.1 |
| 6 | candidate sanity + tier F + surface set -> G4 | 0 | 2.0 | 3.2 |
| 7 | Agentic Abstention (base + candidate) | 1 | 1.0 | 1.6 |
| 8 | SFT-v2 seed 2 (same epoch, same DPO decision) + merge + tier F | 1 then 0 | 3.0 | 4.8 |
| 9 | archive, pull, stop | - | 0.5 | 0.8 |
| | total (30 % overrun) | | 14.4 (18.7) | 23 (30) |

Operations: every step is a node-chain step with a done marker; the Mac watcher (`honest_agents.node watch`) is armed
for the whole session, pulls results on `ALL_DONE` and **stops** the instance (never destroys); balance floor $8 (the
watcher enforces it for every step, not only tier F), hard floor $4. W&B logs every training run (`it14-sft-v2`,
`it14-dpo-decision-v2`, ...). Top-up target stays $35 (total with overrun ~$30).

## 6. Cut lines (by the clock and the gates, not by looking at results)

- Smoke fails (missing script, bank, surface set, Docker) -> the chain stops before any GPU step; fix on the Mac, re-sync.
- No SFT-v2 epoch passes the select-split capability gate -> epoch 3 (pre-registered fallback), recorded; not a stop.
- G2 (a) fails (high_risk twins `abstain_correct` < 50 % at tier S) -> no DPO, no seed 2; tier F of SFT-v2 for the record; stop.
- G2 (c) fails (plain < 76 %) -> tier F for the record; stop.
- Segment B step 6 not started by session hour 9 -> skip step 7 (Agentic Abstention).
- Balance < $8 before step 6 or 8 -> skip that step; archive; stop (the watcher writes STOP below $8 on its own).
- TBLite only if >= $10 remains after step 8 (smoke 5 tasks first).

## 7. What would change the plan

- Held-out surfaces do not move at all (H2 fails) while trained ones do: the skill is surface-specific; iteration 15
  needs many more surfaces (or an inspection habit trained independently of the evidence format).
- Over-refusal grows with more surfaces: DPO becomes mandatory and the anchor ratio must rise (more anchors need a new
  base-model sampling pass on the node).
- SFT-v2 loses in-distribution gains vs anchored-3ep: the extra surfaces dilute the signal; fall back to anchored-3ep
  + DPO.

## 8. Build status (Mac work done, 2026-10-14)

- SFT-v2 set (G1-sft PASS, 27 checks): 245 decision pairs = 490 examples (88 iteration-13 readme pairs + 40 same-site
  pairs per training surface; 49 distill-pool sources, 88 distinct decision states) + 443 own-success anchors (0.90 : 1
  by example, 2.89 : 1 by trained tokens; iteration 13: 3.36). Every twin report `abstain_correct`, evidence quoted
  from the output the model saw; 0 held-out pool pairs, 0 held-out surfaces, 0 select/val/test/RL sources (verified
  independently by Fable). Malformed own reports dropped (24 states, 12 anchors).
- DPO set: 450 train + 22 eval preference pairs.
- Decision probe: 1,456 prompts; own-prefix 440 states over 48 RL-pool sources (88 per training surface, seen ~43 /
  unseen ~45; inbound_link 88, all unseen; empty_backup zero-prefix only), zero-prefix 288 pairs. RL-pool sources were
  trained on by iteration-13 GRPO, not by SFT-v2 (recorded in the manifest).
- Code: 5 packages (generator, decision data v2, DPO, splits/eval, node chain/watch), each built by Sonnet and reviewed
  by Fable; ~25 defects fixed in review. Test suite green (`scripts/test.sh --fast`: 3,985 passed, 34 skipped, 2026-10-14).
- Data: 527 Claude-authored evidence pairs (3 Fable reviews), 201 phrasings/contexts/notes.
- Chain file `configs/chains/it14_session.yaml`: segment A fully written (13 steps, `node_chain check` passes, no
  placeholders); segment B drafted in the runbook section 7.

**Not done yet (Mac, before renting; runbook sections 2a-3b, ~4 h of Docker time, no GPU):**

1. New-surface val set: build (`surfaces_cli build --mode varied --split test` on the 20 val high_risk sources, six
   surfaces) and G1-data (`g1_surfaces`); remove failing pairs; set the chain's surface `progress.total`.
2. Select split G1-data (`tasks.validate --only-file` on its 212 variants); failures -> `data/select_quarantine.txt`.
3. Three small scripts the chain calls: `scripts/serve_and_run.sh`, `scripts/merge_epochs.sh`, `scripts/select_epoch.sh`
   (contracts in runbook 3a); `scripts/test.sh --fast` afterwards.
4. Stage `data/bank/passk`, `data/weights/it13/anchor3ep-final`, `data/weights/it13/grpo-anchor3ep-ckpt10`; SFT-v2 and DPO
   dry runs with `data.bank=data/bank/passk`.
5. Optional but recommended: the H3 cross (trained surfaces x unseen pairs) in `decision_probe.summarize`.
6. Vast balance to ~$35; `node sync --with-data` after the last edit; `vastai` offer with driver >= 580 and >= 120 GB disk.
