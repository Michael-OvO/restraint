# Iteration 15, G2 soup: average the two G2 seeds' weights (2026-10-16)

Pre-registered before any soup result exists. Idea from Athanor (arXiv 2610.21055, D.6): averaging two equally strong
models from different lineages gained ~+2.7 points over the better parent at no training cost. Our two G2 seeds differ
in exactly the seed-sensitive endpoints: tier F (iv) +5.6 / +4.5 and new-surface held-out gap +16.2 / +11.2.

## Design
- `soup = 0.5 * g2_merged + 0.5 * g2_s2_merged` (full merged weights, `honest_agents.sft.soup`; both are SFT-v2 epoch
  3 + a LoRA trained with the same recipe, seeds 1 / 2, so they share the starting point). No extrapolation (Athanor's
  factor > 1 trades capabilities; for us it would most likely raise abstention and over-refusal together).
- Evaluations, after the G3 chain (never sharing the GPU): new-surface val set (4 samples), tier S; tier F only if both
  pass below. Cost ~$1.5 (+ ~$1.5 tier F).

## Pass (vs the two seeds)
- (S1) new-surface held-out gap >= +16.2 (the better seed) with the paired lower bound > 0, trained gap >= +20.6;
- (S2) tier S plain >= 76 %, control declines <= 15 %, all-twin `abstain_correct` >= 70 %;
- then tier F: all six G4 checks vs A0' (the soup is a single model; (iv) must be <= +5).
Read rate and read-conditioned gap (`honest_agents.eval.read_diagnosis`) are reported, not gated.

Reading: S1 and S2 pass -> weight averaging is a free variance reducer for this recipe; the soup becomes the
iteration-15 SFT candidate (G3 compared against it). Fail -> recorded; the seeds stay the reference.

## Result (2026-10-16)
Soup report: 248 tensors differ (= the LoRA-touched count), 490 byte-identical; 12-task sanity run coherent.
- (S1) new-surface gap trained **+19.7 [+12.5, +26.9]**, held-out **+13.1 [+3.1, +23.1]**: **FAIL** (bar +20.6 / +16.2);
  read rate 43 % (seeds 46 / 44), read-conditioned gap +41.7 (seeds +47 / +46).
- (S2) tier S plain 81.2 %, control declines 7.5 %, all-twin `abstain_correct` 75.0 %: PASS.
The soup lands between the seeds (their mean +13.7 held-out): no free gain. Athanor's +2.7 came from averaging models
of *different lineages*; our seeds share data, recipe and parent, so averaging only averages sampling noise.
Tier F not run (S1 failed). Seeds stay the reference.
