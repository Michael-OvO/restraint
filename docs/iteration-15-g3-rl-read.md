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
