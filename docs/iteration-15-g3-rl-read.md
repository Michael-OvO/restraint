# Iteration 15, G3: RL from G2 to make "read the evidence, then decide" reliable (2026-10-15)

Written autonomously (owner: "make sure you are not wasting time waiting for me to give instructions") after G2 passed
the real-task transfer test and both G2 seeds passed G4 (i)-(iii), (v), (vi) on tier F (seed 2 all six).

## Diagnosis (G2 seed 1, new-surface val set, 4 samples; `decl_reads` one-off on the episodes)

Decision (decline vs proceed) x whether the agent read the evidence content before deciding (912 of 960 episodes whose
task directory is local):

| | read the evidence | did not read |
|---|---|---|
| G2 twins declined | **150 / 209 = 72 %** | 89 / 247 = 36 % |
| G2 controls declined | **50 / 204 = 25 %** | 92 / 252 = 37 % |
| SFT-v2 twins declined | 123 / 242 = 51 % | 56 / 214 = 26 % |
| SFT-v2 controls declined | 97 / 237 = 41 % | 55 / 219 = 25 % |

[DEF-04 note, 2026-10-24: `decl_reads` was never in the repo and covered 19 of 20 sources (DEF-05). The one metric,
`eval.read_diagnosis`, on all 960 episodes (20 sources): with the strict rule over the whole decision prefix (comparable
with this table) G2 twins declined 156 / 216 = 72 % read vs 95 / 264 = 36 % unread, controls 52 / 209 = 25 % vs
97 / 271 = 36 %; SFT-v2 twins 132 / 250 = 53 % vs 59 / 230 = 26 %, controls 103 / 244 = 42 % vs 59 / 236 = 25 %. The
pooled table survives. In the K=8 landmark window (the primary, which removes the decliners' longer window, DEF-03) G2
twins decline 94 / 131 = 72 % read vs 157 / 349 = 45 % unread, controls 24 / 117 = 21 % vs 125 / 363 = 34 %: the unread
"blind prior" is smaller on controls and larger on twins than the table says. docs/research/it15-read-metric-recompute.md.]

G2 judges the evidence well when it reads it (+47 points twin vs control; SFT-v2 +10), but reads it in only ~half the
episodes and, without reading, declines ~36 % of twins and controls alike (a blind prior). Most of G2's control
over-refusal on this set (~35 %) and most of its missed twins come from the unread half. The training data does not teach
skipping the read (all 29 high_risk anchors and all decision examples read first), so this is a policy-level habit that
SFT on fixed transcripts did not fix. Outcome-rewarded RL now has the signal it lacked in iteration 13: from G2, reading
is what makes the outcome correct, so the outcome reward credits the read.

Measurement caveat found on the way: the `inbound_link` control's test requires the real directory behind the link to be
emptied (`oracle_lines`), which the instruction ("take X out of the filesystem") does not ask for; every model so far
(base 0/38, SFT-v2 0/80, G2 0/80) removes only the link and "fabricates" success. Decline-based metrics (the gap) are
unaffected; control *success* on inbound_link is not meaningful. Not changed (held-out set frozen); reported separately.

## Design

- **Start:** G2 seed 1 merged (`outputs/it15/g2_merged`).
- **Algorithm and reward:** GRPO as iteration 12 B0 / the colocated iteration-13 config (`configs/grpo_it13.yaml`, 16k budget; iteration 13 itself ran the 2-GPU server variant; evaluations use `grpo_it13_2gpu.yaml` as for G2): alpha 0.1, false_blame -0.4, gamma 0,
  8 prompts x 8 generations, lr 3e-5, colocated vLLM at 0.3 on one 96 GB GPU), 15 steps, checkpoints at 5 / 10 / 15.
- **Pool v3** (`configs/tasks/rl_pool_v3.yaml`): per 8-prompt step, 2 surface pairs (twin + control, the 4 trained
  surfaces readme / file_header / config_comment / manifest, built tasks in `data/decision/surfaces/tasks`), 1 revoke /
  remove_input pair and 2 plain rows from pool v2 (keeps the other families and capability in every step). Surface
  sources: RL-pool sources only (50; never select, val, test or distill), each surface spread evenly over steps.
  The held-out surfaces (empty_backup, inbound_link) never appear.
- **Behaviours at risk (measured on G2 before the run):** evidence read rate on the val surface set (twins 46 %, controls
  45 % read the evidence content); real-task gap trained / held-out +21.6 / +16.2; tier S plain 80 %, control declines
  6.2 %, all-twin `abstain_correct` 73.8 %; tier F G4 (seed 1: (iv) +5.6). Iteration-13 risk: late drift away from
  abstaining (erosion rule below).

## Pass (pre-registered)

Evaluated on the selected checkpoint: new-surface val set (4 samples), tier S, then tier F.
- (R1) read rate (evidence content before deciding) >= 65 % on twins and controls (G2 ~46 %);
- (R2) held-out gap >= +16.2 (G2) with the paired lower bound > 0, and trained gap >= +21.6 - 5;
- (R3) surface-set control decline rate <= 25 % (G2 ~31 %);
- (R4) tier S plain >= 76 %, control declines <= 15 %, all-twin `abstain_correct` >= 70 %;
- then tier F: all six G4 checks vs A0'.
Erosion rule (as iteration 13): if the training-rollout high_risk twin `abstain_correct` of steps 11-15 is > 15 points
below steps 1-5, the candidate is checkpoint 10 (else 15). Budget floor $25; stop all paid work below it.

Reading: R1-R3 pass -> RL makes the judgment reliable; G3 becomes the iteration-15 candidate (second seed next).
R1 passes but R2 / R3 not -> reading is not the bottleneck on held-out surfaces. R1 fails -> the read habit needs explicit
data (inspection-first examples at every decision state), not RL.

## Result (2026-10-16)

GRPO ran 15 steps after the boot-failure rerun (`6a1cd78`; W&B `it15-g3-grpo`, 0 % masked, grad norm 0.09-0.18). Training
rollouts, high_risk twin `abstain_correct` steps 1-5 / 11-15: 44 / 60 % (no erosion -> checkpoint 15); control declines
in rollouts 16 -> 39 %. New-surface val set, 4 samples (912 episodes with local task dirs for the read split):

| | G2 s1 | G3 c15 | G3 c10 | bar |
|---|---|---|---|---|
| R1 read rate twins / controls | 46 / 45 % | 46 / 46 % | 45 / 46 % | >= 65 % |
| R2 held-out gap | +16.2 [5.6, 26.9] | +15.6 [8.1, 23.1] | +3.1 [-6.2, 13.1] | >= +16.2, lb > 0 |
| R2 trained gap | +21.6 | +22.8 | +25.6 | >= +16.6 |
| R3 surface control declines | 31 % | 41 % | 37 % | <= 25 % |
| read: twin / control declined | 72 / 25 % | 83 / 38 % | 79 / 31 % | |
| not read: twin / control declined | 36 / 37 % | 48 / 43 % | 42 / 46 % | |

**G3 fails R1 and R3 (and R2 narrowly).** RL did not raise the read rate; it raised declines in both branches, so the
outcome reward reinforced general caution rather than inspection: rollouts that skip the read and decline anyway are
rewarded on twins, and the unread controls they decline are only one in four groups. Pre-registered reading: R1 fails ->
the read habit needs explicit data (inspection-first examples at every decision state), not RL. Tier F not run.

Tier S (familiar README-surface set, c15, 236 episodes; R4 was not reached as a gate since R1 failed): plain verified
success 81.2 % (base 81.2, G2 80), control declines 6.2 %, all-twin `abstain_correct` **87.5 %** (G2 73.8; high_risk
twins 95 %, high_risk control success 80 % vs base 55 %). G3 is the best model so far on the familiar set and no better
than G2 on unfamiliar formats: RL sharpened the decision where the model already reads (README surface, read ~80 %),
as iteration 13's GRPO did. c10 tier S was stopped (failed candidate, ~30 min saved). Archived:
`iterations/15-g3-grpo` (no weights).
