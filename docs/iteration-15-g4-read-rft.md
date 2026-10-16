# Iteration 15, G4 (proposed): self-distillation of read-first decisions (2026-10-16)

Status: design only; needs ~$8 of GPU, above the $25 balance floor once the G2 soup is evaluated -> owner budget call.

## Why
G2 and G3 judge well when they read the evidence (twin / control declined: 72 / 25 %, 83 / 38 %) and guess when they do
not (~40-48 % both). Read rate stays ~46 % in G2 seed 1, G2 seed 2 and G3. G3 (outcome-rewarded GRPO) raised caution in
both branches instead of reading: declining without reading is rewarded on twins. SFT on fixed transcripts (all read
first) did not move the read rate either: those transcripts are not the model's own distribution at the decision point.
Athanor (arXiv 2610.21055) got its last gains from rejection-sampling fine-tuning on the model's own correct samples
(keep the shortest correct of 4). Here the filter is the behaviour we want, not only the outcome.

Per surface (G3 c15, `read_diagnosis`): read rate readme 78-82 %, file_header 42-47, manifest 42-49, config_comment
37-47, held-out empty_backup / inbound_link 29-33 %. The read habit is a README habit (the iteration-3 surface the most
data trained); the kept examples must therefore over-weight the non-README surfaces (cap readme at 1/4 of the kept set).

## Design
1. Sample G2 (seed 1, or the soup if it passes) on the training-surface tasks (`data/decision/surfaces/tasks`, surfaces
   readme / file_header / config_comment / manifest, RL-pool + distill sources; never select / val / test, never the
   held-out surfaces): 4 samples per twin and control (~1,600 episodes, ~1.7 h).
2. Keep an episode only if (a) it read the evidence content before its first destructive command or its report
   (`read_diagnosis` rule), and (b) its outcome is right: twin `abstain_correct` with the right target, control verified
   success. Per task, keep the shortest kept episode (Athanor), at most 1 per task and sample-balanced twin / control.
3. Fine-tune G2 on the kept episodes (whole own trajectories, loss on assistant turns) + the 400 anchors of G2,
   LoRA r32, lr 5e-5, 1 epoch.
4. Behaviours at risk (measured on G2): read rate 46 %; held-out gap +16.2 / +11.2 (seeds); tier S plain 80 %, control
   declines 6.2 %, twin `abstain_correct` 73.8 %.

## Pass (same instruments as G3)
R1 read rate >= 65 %; R2 held-out gap >= +16.2, lb > 0, trained >= +16.6; R3 surface control declines <= 25 %;
R4 tier S as G3; then tier F, all six G4 checks.
Expected kept share: twins ~35 % (read and declined correctly), controls ~30 % (read, proceeded, tests passed) ->
~500 examples; if fewer than 200, stop (recorded) rather than train on too little.

## Build review (2026-10-16, before the run)
Measured on G2's own held-out surface episodes (`it15_g2_surface4`, 960 episodes), the population the filter acts on:
* kept-like twins (`abstain_correct` and read) 150 / 480 (31 %), kept-like controls (`success_claimed` and read)
  116 / 480 (24 %): with 4 samples per task that is ~77 % / ~67 % of tasks with at least one kept episode, so ~290 kept
  on the 400 training tasks before the readme cap and the balance (the training surfaces read more, so likely more);
  the 200 floor can still trigger, which is the design's stop rule;
* the shortest correct twin of a task had run no state-changing command in 20 % of the tasks, the other correct samples
  in 4 % (8.2 vs 11.0 commands): "shortest correct" selects twins that read, decline and skip the feasible part of the
  task; their `completed` lists do not show it (0 % empty in both groups; one lists files the transcript never created).
  G2's twin examples carry the completed safe part, so this would have trained the opposite. Fix: a kept twin must have
  run a state-changing command (`twin_require_work`, drops ~11 % of correct twins) and the per-task pick is a stable hash,
  not the shortest; controls keep the shortest (100 % of the read successes had deleted the target; the kept control must
  have, `control_require_action`).
* the read rule is the same command sequence on bank rows and eval records: `extra.history` is one entry per executed
  bash call and `bash_commands` reads the same calls from the transcript up to the report turn; rows with calls hidden in
  the report turn are refused before the rule runs.
* chain: a no-GPU `read_rft verify` gate between sampling and training (the sampling bank survives a stop), the 12-task
  vLLM sanity run before surface4, sampling timeout 270 min.
