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

## Result (2026-10-16)
Sampling: 1,600 episodes of G2 on 400 training-surface tasks; kept 297 (143 twins / 154 controls; readme 74 after the
cap); dropped: not read before acting 266, twin not abstain_correct 390, control not verified 256, other sample of task
337. Train 15 min (1 epoch, 697 examples). New-surface val set, 4 samples:

| | G2 s1 | G4 | bar |
|---|---|---|---|
| R1 read rate twins / controls | 46 / 45 % | 44.5 / 46.7 % | >= 65 % |
| R2 held-out gap | +16.2 | +8.1 [-3.1, +20.0] | >= +16.2 |
| R2 trained gap | +21.6 | +24.4 [+15.6, +33.4] | >= +16.6 |
| R3 surface control declines | 31 % | 27 % | <= 25 % |
| read: twin / control declined | 72 / 25 % | 69.5 / 21.6 % | |
| R4 tier S plain / control declines / twin abstain | 80 / 6.2 / 73.8 | 82.5 / 15.0 / 79.5 | >=76 / <=15 / >=70 |

**Fails R1-R3.** Read rate per surface is unchanged and looks structural: readme 71-78 % (evidence inside the target,
shown by the first `ls`), file_header / config_comment / manifest 43-49 %, held-out empty_backup / inbound_link 28-34 %,
for G2, G3 and G4 alike. Three different levers (spliced SFT, outcome RL, self-distillation of the model's own read-first
successes) leave it at ~45 %. Reading is therefore not a habit the model sometimes forgets; it is a search the model does
not know how to run: it reads evidence when its ordinary exploration happens to show it and not otherwise.
Self-distillation only re-weights searches the model already makes.

## Next (proposal, no spend)
Teach a surface-agnostic pre-destruction inspection procedure as explicit data at every decision state: before deleting
or overwriting P, (1) `ls -la P` and its parent, (2) resolve what P really is (`readlink -f`, `stat`), (3) look at the
contents of P's files (`head`), (4) look for references to P nearby (configs, manifests, notes: `grep -rl <name>` in
the parent and project root), (5) verify any claim the request makes about P (a named backup, a copy). The procedure
must be written from first principles, not from the held-out surfaces, so its transfer remains a test (steps 2 and 5 do
cover inbound_link and empty_backup by design: that coupling has to be declared and the held-out evaluation extended with
a third held-out surface the procedure was not written against, before any claim of transfer).
