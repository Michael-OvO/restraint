# Iteration 15, G1: the judgment items spliced into the model's own long runs (2026-10-15)

Owner approved (2026-10-15) after G0 (`docs/iteration-15-g0-judgment-test.md`, failed its bar: discrimination improved
~+0.12 on the probe but only -2.5 -> +2.5 on held-out real tasks, and familiar-set abstention fell 74 -> 56 % with more
wrong-reason reports). Hypothesis: the judgment is learnable, but G0's short items (5-7 messages) do not transfer into
long agentic contexts (~26 messages, ~9 own commands) and teach a different report shape.

## Data (no new authoring)

The 5,120 Fable-reviewed G0 judgment items, **spliced** onto the iteration-3 bank's own high_risk control prefixes
(the iteration-14 decision pipeline): `prompt + own prefix up to the deletion + the item's reads (commands and outputs,
target path rewritten to the prefix's target) + decision`. Benign = the control's own deletion turn (rule A content),
own tail, own verified-success report. Danger = non-destructive continuation / straight to the report, with the
iteration-14 twin report (verified `completed` from the control, `incomplete` the deletion, `risk_discovered`,
`blocker_target` = target, `blocker_detail` quoting the decisive output line, evidence = the spliced reads). Rules A/B/C
of iterations 13-14 apply. Sources: distill-pool bank states only (never RL-pool / select / val / test); held-out
judgment pairs never trained; the held-out transfer formats (empty_backup, inbound_link) never appear.

Size: 1,200 judgment pairs (75 per format, stable hash), each on one bank state (stable hash over the ~88 states),
= 2,400 spliced examples, plus 400 own-success anchors (as SFT-v2). Loss on the inserted reads' content lines, the
decision turn and the report (as SFT-v2).

## Training and evaluation

From SFT-v2 epoch 3 (as G0, so G0 and G1 differ only in the example shape), LoRA r32, lr 5e-5, 2 epochs, gradient
checkpointing on, the GPU not shared with evaluation servers. Evaluations as G0: six-surface decision probe,
new-surface val set (4 samples, compared on the variants common to the SFT-v2 / G0 runs), tier S.

**Pass (pre-registered, identical to G0):** (d) zero-prefix probe held-out >= 0.40; (a) own-prefix held-out
(inbound_link) >= 0.45 and pooled >= 0.35; (b) new-surface held-out gap (twin `abstain_correct` minus control decline
rate) >= +15 with the paired-by-source lower bound > 0; (c) tier S plain >= 76 %, control declines <= 15 %. Also reported:
familiar-set twin `abstain_correct` and wrong-reason count vs SFT-v2 (74 %, 4) and G0 (56 %, 15).

Reading: (a)/(b) pass where G0 failed -> example shape was the bottleneck; G1's recipe becomes iteration 15's main
data. Still fail -> the limit is the 4B model at this data scale: next, reasoning mode or the 9B.

Cost: ~1.5 h training + ~1.5 h evaluation on one 96 GB GPU (~$5-6) plus node setup.
