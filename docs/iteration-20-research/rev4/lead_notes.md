# Orchestrator notes for the joint stage (iteration 20, revision 3 → 4)

These are the lead's working conclusions after reading the ten grill/evidence reports. They are inputs to test, not
conclusions to defend. Disagree where the evidence says so.

## 1. Where a solid U gain can come from (arithmetic on published held-out aggregates, u_checks check 1)

Primary strata P ∪ T on the frozen held-out set: 118 plain, 158 trained-family controls (86 high_risk, 42 revoke, 30
remove_input), 155 trained-family twins (86 / 42 / 27). Feasible share 64 %, twins 36 %.

- G2 on feasible rows: pass 0.783, false claims 13.6 per 100 episodes, P(claim | pass) about 0.99, P(claim | fail)
  about 0.63.
- Turning one failing feasible episode into a pass is worth about +1.62 U (it was −1 with probability 0.63, becomes
  +0.99). So **pooled ΔU ≈ 1.04 × Δ(points of feasible pass rate)**.
- Gap to best-of-2 on feasible rows is 11.2 points of pass (0.783 → 0.895). A third to a half of it is +3.5 to +6 U.
- Twins: G2's residual twin false claims plus damage are worth 4.3 pooled U in total. Half is about +2.
- Report-only levers are capped at +0.2 to +2.4 (iteration-20-design.md §1, fact 4).
- Controls: G2 does not claim on 9.7 % of control episodes (decline 7.4 %). Restoring half of unwarranted control
  declines is worth about +1.

So a solid result (pooled ΔU ≥ +4) needs RL to raise feasible task success by about 3 to 5 points while holding
restraint. Revision 3 treats task success only as a guard.

## 2. The binding problem is gradient signal-to-noise, not the reward

From evidence:dynamics (train-side data):
- The it19c ablation's 30-step update is statistically a momentum random walk (lag cosines of LoRA increments match
  AdamW on pure noise; an independent run on the same prompts has cosine 0.005 with it). Signal is at most 1-6 % of
  the squared displacement.
- A 128-episode step carries about 11 effective pass/fail contrasts, each spread over about 1,900 model tokens.
- KL β 0.04 is about 2 % of the policy-gradient force: a meter, not an anchor.

Under an Adam model, drift ∝ lr × steps × ρ and noise displacement ∝ lr × sqrt(steps × c) (ρ = per-step normalised
signal ∝ sqrt(batch), c ≈ 19 for β1 = 0.9). For a fixed learned displacement, the noise energy that comes with it is
∝ 1 / (total episodes × per-episode SNR²). The learning rate only sets how far the policy moves. Consequences:

- **Total on-policy episodes × per-episode signal quality is the quantity to maximise.** Every published terminal-agent
  RL gain used 100k-500k episodes (Endless Terminals, ECHO, River, DeepSWE) at a low rate; revision 3 budgets 19.2k.
- The learning rate should fall as the episode count rises (LoRA rate about 10x a full-fine-tuning rate of 5e-7 to
  1e-6, so about 5e-6 to 1e-5).
- Per-episode signal can rise by: online handling of zero-variance groups (DAPO dynamic sampling +8 points; online
  difficulty filter +3.8 points while offline curation gave +0.0), allocation by pass count and one unit per source
  (+25-45 % mixed groups), dropping always-pass rows (+13 %), an auxiliary loss on tool-output tokens (ECHO, +8.8
  points on Qwen3-8B, one unreplicated paper).

## 3. Lead's own proposals to test (not yet reviewed by anyone)

1. **A learning-rate-0 run (arm Z) with three jobs.** G2 through the real trainer at lr 0 on the seed-1 order:
   (a) unbiased per-prompt G2 baselines under the same sampler and load (no selection bias, unlike the prescreen);
   (b) the measured false-fire rate of every stop rule; (c) a **gradient-signal audit**: measure the policy gradient's
   signal and noise directly, before any learning step.
2. **Split-half gradient agreement.** With `per_device_train_batch_size` 1 and accumulation over the step, snapshot
   the accumulated gradient at the half-way micro-step: g_A (first half of the prompt groups) and g_B (second half).
   E[g_A · g_B] = ‖μ‖² (the squared true gradient), and (‖g_A‖² + ‖g_B‖²)/2 − g_A · g_B estimates the noise of a
   half batch. Averaged over 25-30 steps this gives the per-step SNR directly, and so the episodes per step needed
   for a signal-dominated update. It costs one extra copy of the LoRA gradient per step, and it can stay on during
   training as a live meter that the update still carries signal (a positive-learning test at the gradient level,
   far better powered than outcome metrics).
3. **Estimator selection by measurement.** At lr 0 every reward variant sees the same rollouts. Per-episode gradients
   are already separate backward passes; a count-sketch of each (k ≈ 65k buckets) plus the episode's outcome lets
   any reward (u, u_success, w = 1, neutral weights), baseline (group mean, leave-one-out, cached value), batch
   geometry or pool mix be compared offline by resampling: mean-gradient norm, noise, and the cosine between the
   gradients of two rewards (how different arms A and B really are). This replaces argument with a number.
4. **Scale as a pre-registered dose.** Decide episodes per step and total steps from the audit, with a floor from
   the literature (≥ 50k-100k episodes for the main arm), and read a dose-response on a powered dev set.
5. **G3's archived adapter as a metric control.** Under U it scores about +12 on train-side sets while control pass
   falls and control non-claims rise: a known caution shift. Used as a reference arm it shows whether the guard set
   catches a caution shift that U alone rewards. It is not a result to claim.
6. **Primary that caution cannot win.** Pooled U at w = 1 can be won by twin caution alone (4.3 U of headroom,
   above the MDE). Options: a Holm family (pooled ΔU, Δ task success > 0, Δ false claims per feasible episode < 0);
   or U on feasible rows as primary with twins as a non-inferiority guard. To be settled by the judges.

## 4. Points where the reports disagree (the joint stage must settle these)

- Twin weights: keep blind-neutral weights (corrected for the measured damage rate 0.93-0.97, so about 0.21-0.39,
  not 0.44), or w = 1 (arm W), or neutral for high_risk only.
- KL: drop to a monitor (β ≤ 0.01 or 0), or find a binding β by a ladder. Literature and grill:method say monitor.
- Fresh G2 pass on the pool as baseline (data D7, metrics S3) versus the lr-0 arm Z (history F1, programme P2).
- Below-band rows (G2 pass ≤ 3/8) in the pool: grill:method says add 100 (they hold 52-73 % of feasible false
  claims); the dynamics say they push claims down on rows where the policy cannot tell pass from fail.
- Screen design: an 8-arm, 50-step screen ranked by dev ΔU (programme P2) versus choosing the recipe by measured
  gradient SNR and running fewer, longer arms. At 50 steps the true differences are likely below the dev noise, so
  a ΔU ranking would select on noise (selection optimism about 2 U for the best of 21 cells).
- Truncated episodes: revision 3 unmasks them (rule: never mask a token); DeepSWE/DAPO mask episodes that hit a
  budget limit. An episode-level mask is not a token mask.
- Loss aggregation: DAPO token-sum (revision 3) versus sequence-mean (Endless Terminals, River).
- Whether the held-out look should be gated on a dev result (programme P5) or always spent.

## 5. Fixed constraints (owner rules)

- The agent itself is trained by RL. Results must be measurable, with intervals, and pre-registered.
- Executor holds only the train bundle; evaluator holds held-out; no held-out look by the executor; no new analysis
  of held-out episodes in this design session.
- Integrity gate (docs/integrity-gate.md): no experiment with an open defect. Pre-registration before training.
- Never mask a token from the main loss; add components one at a time to a baseline that works; pilot and read
  behaviour; smoothed stop rules; never an eval server on a training GPU; every run logs to W&B.
- The cluster is free; size unknown (assume 6-8 GPUs ≥ 80 GB, parameterise by GPU count). No deadline, but calendar
  time is not unlimited: say what fits in 7, 14 and 21 days.
- The new plan keeps the format of docs/iteration-20-rl-design.md (same section order where possible) and must keep
  docs/iteration-20-handoff.md consistent.

## 6. After the deep research: what changed, and the lead's candidate design

Four independent reports (failure-taxonomy, pass-fail-anatomy, lit-honesty-rl, twin-judgment-signal) reached the same
mechanism from different data. Summary of what they found on train-side data:

- **Whole-episode pass/fail RL has little transferable signal.** Inside a variant, false claims cannot be told from
  passes by any process feature (AUROC 0.55 [0.49, 0.61]); a check after the last edit ran in 94.3 % of passes and
  93.9 % of false claims. What whole-episode RL would reinforce: family conventions on controls (move the input back
  77 % vs 39 %), shorter episodes that do not loop, and per-task content.
- **But the false claims are fixable from what is on screen.** In 77 % [64, 88] of 64 false claims read in full, the
  agent's own check printed the violating state before the claim; 69 % [54, 84] break a requirement the instruction
  states outright; 86 % [76, 94] had at least 3 of 28 bash calls left (median 13). 69 % [55, 82] judged convertible
  to a pass, 14 % only to an honest non-claim, 17 % unreachable. (One reader; needs an independent second read.)
- **The report token itself is saturated** (15 of 747 rows have pi(claim) between 0.05 and 0.95), so any term that
  acts on the status token alone is inert; the decision has to be re-opened earlier, at a state where the policy can
  still look and fix.
- **Reading is not moved by whole-episode rewards** (two runs with a read push 2.5-5x plan 20's: read rate 0.0
  [-6.3, +6.6] and -4.8 [-11.5, +1.9]).
- **Scale exists in the literature, and it is feasible here**: positive results used 24.6k-128k episodes at 240-512
  per update, KL 0-0.001; TMax took Qwen3.5-4B (this project's base) from 33.6 to 54.3 on Terminal-Bench Lite with
  500 steps x 256 episodes. One arm of 450 steps x 128 fits in about 60 h with three step-time fixes.

### Candidate: decision-point GRPO (DP-GRPO) under plain U

The RL analogue of the one lever that worked in SFT (supervision at the model's own decision states, iterations
13-15), with an outcome reward.

- **Shared-prefix groups.** Take one of the policy's own stored episodes on a train-legal variant, replay its commands
  in a fresh sandbox up to a decision point (`analysis.replay_episodes` already replays prefixes with an exit-code
  fidelity check), and sample the 8 generations as continuations from that state. The prefix is context; every
  generated token trains; the reward is plain U at the end of the continuation.
- **Endgame points**: after the first read-back that follows the last write (or just before the last write, chosen by
  a pilot). Prefixes from false-claim episodes and from true-claim episodes 1:1. On a true-claim prefix the right
  continuation is "claim"; on a false-claim prefix it is "compare with the instruction, fix, then claim" (+1) or
  "report incomplete" (0), against "claim" (-1). A blind "never claim" policy scores 0 on both where "always claim"
  scores +1 and -1: neutral by construction at 1:1, the same logic as twin/control pairs. The fact that separates the
  two cases is on screen, so judgment is the only way to gain.
- **Pre-decision points for pair rows**: prefix up to the first command that names the target; continuations read or
  do not read, then decline or proceed.
- **Why it should raise signal**: the group baseline becomes a state value, so the advantage is about the decision,
  not about earlier luck; the trained tokens per episode fall from about 1,900 to a few hundred; each converted false
  claim is +2 U.
- **Unknowns to measure before any training (kill tests)**: the share of false-claim-prefix groups with mixed
  outcomes under G2 (go if at least 40 %; a deeper branch point if not), replay fidelity (at least 90 % replayable),
  and whether behaviour learned at branch states shows up in full episodes (the evaluation is always full episodes).
- **Known risk**: training near the report has moved twin actions three times (iterations 17, 18, 19 K2: twins
  destroyed 53 / 31 / 21 against 13). Those runs trained report tokens on frozen transcripts with no action
  consequence; here continuations act and are graded, twins and pre-decision pair groups stay in the batch, and the
  twin guard is run at higher power. This must be attacked, not assumed.

The joint stage should test this candidate hard, compare it with a scale-first whole-episode design, and say which
should be the main arm and which the comparison.
