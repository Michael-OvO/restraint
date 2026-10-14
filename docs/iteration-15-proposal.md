# Iteration 15 proposal: teach the judgment, not the format (draft, 2026-10-14)

Status: proposal agreed in direction by the owner (2026-10-14); to be turned into a design and reviewed before any
spend, as in iterations 13-14.

## Why (iteration 14's lesson)

SFT-v2 passed all six G4 checks on the familiar template (abstain +42, capability +5.1 [+0.9, +9.2], control declines
+3.1), but on unfamiliar evidence formats it does not discriminate: on the new-surface val set it declines impossible
twins and their safe controls at about the same rate (held-out surfaces: 39.5 % vs 39.5 %; trained surfaces 39 % vs
34 %), and the decision probe's discrimination is 0.20. DPO lowered over-refusal on new formats (controls 9-12 %
declined) by lowering caution overall (twins 9-12 %), not by reading better. The models learned "be cautious in this
kind of situation", conditioned on surface cues, not "read the evidence and judge it".

## What changes

1. **Train the judgment as explicit reasoning in the model's own words** (deliberative style): before an irreversible
   action, one reasoning step that names what will be destroyed, quotes the evidence, and states the conclusion. Either
   with Qwen3.5's thinking mode on, or as a short visible line. Supervised on own-register text, so it is the model's
   wording, not a template.
2. **Decouple the two skills.** (a) Judgment: a large sandbox-free set, Claude-authored, of "here is a file / log /
   config / listing; is it safe to delete X, and why?" items across many formats and domains (target >= 5,000 items,
   counterfactual pairs, half of them hard). (b) Habit: inspect before `rm`, one fixed behaviour, trained in the agentic
   splice as now. Today both are entangled in ~500 long trajectories, so the judgment sees only a few hundred examples.
3. **Make surface cues useless by construction.** At the training-example level (not only the pools): the same format
   and wording with the meaning flipped by one fact; danger in calm wording and safe content in alarming wording as the
   majority of judgment items.
4. **Decision-state RL** (deferred from iteration 14): RL from mid-task decision states across many formats, so the
   reward lands on "read, then decide"; RL is expected to generalise better than SFT. GRPO checkpoint 10's new-surface
   result (iteration 14) is the first data point for whole-episode RL.
5. **Primary metric = discrimination on held-out formats**: twin `abstain_correct` minus control decline rate,
   paired by source, on held-out surfaces (and the decision probe's held-out discrimination). Declines alone can be
   raised by caution, which is what iteration 14 showed.

## Cheap first test (gate before the full iteration)

Judgment-only SFT on ~5,000 text items (no sandboxes; a few GPU dollars), then the six-surface decision probe and the
new-surface val set: does held-out discrimination move from ~0 toward >= 0.3 (probe) / >= +15 points (val)? If not,
test whether the 4B can hold the judgment at all (the 9B base as a reference; it failed the same way untrained in
iteration 13).

## Carry-overs from iteration 14

- Candidate to beat: SFT-v2 epoch 3 (all six G4 checks), plus GRPO checkpoint 10 (strongest abstention and capability,
  control declines +6.2) and DPO (control declines +0.3).
- Infrastructure: node chain + watcher (worked; three ad-hoc sub-chains were needed after a stop on a pruned G1 failure,
  now fixed), >= 400 GB disk if Terminal-Bench evals run on the node, `dpo.max_pair_tokens` for 48 GB GPUs, W&B
  project set on every training step.
