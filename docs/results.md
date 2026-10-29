# Results, iteration by iteration

Model: Qwen3.5-4B with LoRA (iterations 0-12 also Qwen3-4B for bring-up). Tasks: Endless Terminals sandboxes with
feasible tasks, near-miss controls and impossible "twins" (`high_risk`, `revoke_permission`, `remove_input`, plus
held-out families and evidence surfaces). Unless a row says otherwise, differences are paired by source against the
base model with 95 % intervals; iteration 17's are 90 %. "Six checks" means the final gate of iterations 13-15.
Every number links to the document it comes from (Sources, below); the full narrative is
[training-log.md](training-log.md).

| Iteration | What was tried | Key result | Verdict |
|---|---|---|---|
| 0 | Built and validated the dataset (2,482 source tasks, 8,790 variants, hash split by source) | 1,907 / 1,955 validated variants valid (97.5 %, gate 95 %); test split frozen (565 sources) | infrastructure |
| 1 | Phase-0 smoke test: the real GRPO trainer at learning rate 0 | Ran end to end; report compliance 84 % (gate ≥ 70 %), no-report 15.6 % (gate ≤ 5 %: fail) | infrastructure |
| 2 | Base-model baseline on the test split (1,524 episodes) | Verified success 68.3 %; claims on impossible tasks 48.7 %; `high_risk` abstain recall 0.4 %; a blanket-decline policy out-scored the base on mean reward (0.431 vs 0.316) | infrastructure |
| 3 | pass@4 bank (3,828 episodes) and difficulty gate | 75 of 250 feasible tasks in the 2-3-of-4 band; stated-confidence AUROC 0.59; alpha 0.25 would let blanket declining win | infrastructure |
| 4 | SFT on hindsight reports, loss on report tokens only | Held-out loss 0.0094, yet zero commands in 82 % of feasible episodes; verified success ~1 % vs 66 % | failed |
| 5 | Combo SFT (best-of-N trajectories + hindsight reports) | Missing reports 11.4 → 1.9 %, but verified success 72.9 → 59.2 % and false-claim share 18.4 → 25.3 % | failed |
| 6 | Weight-sync test, 2 GRPO steps at a deliberately destructive lr 1e-3 | Importance ratio 0.9999 while reward fell 0.447 → 0.019: weight sync works | infrastructure |
| 7 | GRPO run 1 (alpha 0.1, false_blame 0, lr 3e-5) | OOM-killed at step ~19 (69 of 73 GB host RAM); an apparent success drop matched the base on the same tasks; found a false-blame reward hole | failed (crashed) |
| 8 | Memory-leak diagnosis | RSS 4.4 → 6.0 → 7.1 GB with a flat Python heap; after the fixes the trainer held 4.8-5.0 GB for 50 steps | infrastructure |
| 9 | GRPO run 2 (false_blame −0.4, 50 steps) | Probe task success 79.9 → 85.4 %; claims on impossible tasks 60.4 → 60.4 %; `high_risk` twins claimed 8/9 → 8/9 | mixed (more capable, not more honest) |
| 10 | Verified trajectory review of runs 1-2, SFT and evals | 46 findings (15 confirmed, 24 corrected, 7 not reproduced); `high_risk` twins claimed 306/324, 0/324 read the README | analysis |
| 11 | Teacher (Sonnet) SFT, then GRPO from base with a new reward and pool | Teacher SFT verified success 69.4 → 21.9 % (−47.5 [−56.6, −39.0]); GRPO claims on impossible −9.0 [−18.2, −1.2], but via budget exits | failed |
| 12 | Fixed harness; hint distillation; GRPO from base (B0) | Hint: README read 2 % / 9 %, 0 correct abstentions; B0 abstain_correct +6.3 [+1.2, +11.5] (all `revoke_permission`), verified success −3.3 [−7.7, +1.4] | mixed |
| 13 | Decision-point SFT at the model's own states + own-success anchors; GRPO from it; a 9B control | Tier F abstain +44.8 [34.4, 55.6], claims −38.8, verified success +3.1 [−0.3, 6.7], control declines +6.9 [0.3, 14.2] (fails +5); 5 of 6 checks; GRPO eroded it; the 9B discriminates no better than the 4B | mixed |
| 14 | SFT-v2 (four evidence surfaces, 490 decision + 443 anchor examples), DPO, 2 seeds, transfer evals | Seed 1 passes all six checks (abstain +42.0, verified success +5.1 [+0.9, +9.2], control declines +3.1); seed 2 fails control declines (+6.6); unfamiliar formats: probe discrimination 0.20, Agentic Abstention recall 1/19 → 0/19 | mixed (familiar evidence yes, unfamiliar no) |
| 15 G0 | Judgment-only SFT on 5,120 Sonnet-authored decision items | Held-out probe 0.34 (bar 0.40); real-task held-out gap +2.5 [−6.9, +11.9] (bar +15) | failed |
| 15 G1 | Judgment items spliced into the model's own long prefixes, reads masked | Probe 0.76, real-task gap −0.6; the inspection habit was lost (read rate ≤ 5 % on every surface) | failed |
| 15 G2 | G1 data + the SFT-v2 decision replay (real inspection turns) + anchors; 2 seeds | Real-task gap trained +21.6 [+12.5, +30.3], held-out +16.2 [+5.6, +26.9] (pass); tier F seed 1 five of six checks, seed 2 all six | mixed (**became the reference model**) |
| 15 G3 | GRPO from G2 on four trained surfaces (15 steps) | Read rate 46 / 46 % (bar ≥ 65 %); surface control declines 41 % (bar ≤ 25 %) | failed |
| 15 soup | Weight average of the two G2 seeds | Held-out gap +13.1 [+3.1, +23.1] (bar +16.2); lands between the seeds | failed |
| 15 G4 | Self-distillation of 297 own read-first correct episodes | Read rate 44.5 / 46.7 %; held-out gap +8.1 [−3.1, +20.0] | failed |
| 15 diagnosis | Re-analysis of G3, soup and G4 with independent verifiers | Read-rate change vs G2 within ±1.6 (MDE ~6); model identity explains 0.0 % of read variance; one surface's control unpassable (0/76): a grader defect | analysis |
| 16 prep | Integrity round (defect ledger, tiered sweeps), checklist pipeline, frozen final benchmark | Ledger 60 entries (53 closed); oracle gate 120/120; checklist agreement on controls 0.944 [0.916, 0.971] | infrastructure |
| 16 (b) | G2 claim rollouts for matched-pair data | 45 of 94 sources dropped as overlap with the final decision probe (DEF-63); a 639-example frozen build; claim flip sensitivity 0.72 | infrastructure |
| 16 (c) | SFT from G2 on matched claim / destructive-step pairs | Select split (60 sources, S = 2), base / G2 / final: verified success 64.8 / 72.7 / 68.0 %; claims on infeasible 71.1 / 9.9 / 11.2 %; control decline 3.5 / 5.6 / 21.8 %; false claims G2 65 vs 53, under-claims 0 vs 16 | failed (not better than G2) |
| 17 | Confidence-only training on G2's own transcripts: exact expected Brier on the `confidence` value, KL to G2 on that value | **Primary passes:** confidence AUROC for failure 0.591 → 0.774 (+0.182 [+0.135, +0.228]); **a guard fails:** claims on infeasible 8.6 → 32.9 % (+24.3 [+17.3, +31.4]); protected data destroyed on 53 / 120 twins vs 13 / 120 | failed (NO-GO) |

Notes:
- **Iteration 14, Agentic Abstention.** The recall is the re-scored value (DEF-28); an earlier "37 → 74 %" counted
  `incomplete` reports as abstentions and is void.
- **Iteration 15, inspection rates.** Only the canonical `eval.read_diagnosis` values are cited; a one-off tool's
  per-surface rates are void (DEF-04).
- **Iteration 15, held-out-gap bars.** The diagnosis found them at the noise floor (MDE 18-20 points), so the soup, G3
  and G4 verdicts rest on the read rate, the endpoint those evals resolve.
- **Iteration 16.** Stage (a) (calibration samples) was dropped by an owner amendment. Stage (c)'s epoch 2 passed its
  select-split gate against the base (verified success non-inferior), but it was not better than G2, so the RL stage
  (c2), the validation gate and the final benchmark (d) did not run; the owner chose the iteration-17 pivot instead.
- **Iteration 17.** Its guards compare against a G2 re-run under the current harness (verified success 69.1 %), not
  the iteration-16 G2 run (72.7 %); the harness changed in between (DEF-65).

## Lessons (as the log states them)

- **Loss is not behaviour.** "Low SFT loss said nothing about behaviour; check command counts and outcomes on held-out
  tasks first." (training-log.md, "Current state, Reflections"; iteration 4)
- **Compare on the same tasks.** "Per-step training numbers are confounded by which tasks each step draws: compare
  against the base model on the same tasks." (training-log.md, iteration 7)
- **A reward cannot teach what is never sampled.** "Check that the behaviour occurs in rollouts (group variance) before
  tuning the reward." (training-log.md, "Current state, Reflections")
- **Cloning a stronger teacher copies its surface.** "Behaviour cloning from a much stronger teacher copies its surface
  (short trajectories, early decisions) without its skill." (training-log.md, iteration 11)
- **A decision problem, not an exploration problem.** "A 4B model that reads an explicit 'only copy, do not delete'
  warning and deletes anyway has a decision problem, not an exploration problem." (training-log.md, iteration 12)
- **Methods moved caution, not judgment.** "On unfamiliar evidence formats no method (SFT, RL, DPO) produces
  discrimination: they move a caution level." (training-log.md, iteration 14)
- **Check what a change removes.** "G1 was avoidable: it masked the reads and dropped the decision replay at once,
  erasing the inspection habit." (training-log.md, "Reflection (G0 → G2)")
- **A regularizer only protects the tokens it covers.** Iteration 17's KL anchored the confidence value only; the
  shared weights let the actions drift (blind deletions 4 → 30 of 120 twins; action KL from G2 0.18 nats per token).
  ([iteration-17-results.md](iteration-17-results.md))

## Sources

Paths under `docs/`; line numbers at commit `ef1ebd3`.

- 0-14: [training-log.md](training-log.md), the iteration's own section (iteration 0 at line 177, through iteration 14
  at line 1065), with the "Latest" summaries at lines 14-71.
- 15: [training-log.md](training-log.md) lines 1183-1324 (G0, G1, G2);
  [iteration-15-g3-rl-read.md](iteration-15-g3-rl-read.md) "Result"; [iteration-15-g2-soup.md](iteration-15-g2-soup.md)
  "Result"; [iteration-15-g4-read-rft.md](iteration-15-g4-read-rft.md) "Result";
  [iteration-15-diagnosis.md](iteration-15-diagnosis.md) "Executive summary".
- 16: [training-log.md](training-log.md) lines 1330-1373; [iteration-16-signoff.md](iteration-16-signoff.md);
  [iteration-16-prereg.md](iteration-16-prereg.md) sections 10-11; stage (c)'s numbers from the archived select runs
  (`iterations/16-it16c-sft/eval/it16_{base,final}_select*`, `iterations/16-it16c-g2-select`, `metrics.json`) and
  [iteration-17-proposal.md](iteration-17-proposal.md) "Why pivot".
- 17: [iteration-17-design.md](iteration-17-design.md), [iteration-17-prereg.md](iteration-17-prereg.md),
  [iteration-17-results.md](iteration-17-results.md).
