# Iteration 13: training distribution and methodology options (research note, 2026-10-12)

Read-only analysis of the repo, the pass@4 bank (3,828 base rows, training reward re-computed with alpha 0.1 /
false_blame -0.4 / gamma 0), the it11 GRPO-base and it12 B0 training completions (1,280 + 960 scored rows with logged
advantages), the it12 tier-F evaluations (A0 base, B0 step 10; 712 episodes each) and `data/*.jsonl`. Scripts were run
ad hoc; intermediate tables are in `/tmp/it13_research/` (`bank_per_variant.json`, `it11_rows.parquet`,
`it12_rows.parquet`, `pool_noquota*.report.json`).

## 0. Verdict (ranked)

1. **Keep the mixed pool and the B0 order for the D-RL arm; the owner's "reward scale" concern is not what the data
   shows.** GRPO centres rewards per group, so cross-family offsets (the -2 on high_risk claims) cancel; what matters
   is the within-group contrast, and twins already have *smaller* advantages than controls (within-group advantage sd:
   twins 0.37-0.55, controls/plain 0.63-0.95). The problem is one-sided groups: 46-56 % of high_risk twin groups are
   zero-variance, and in the mixed ones the positive advantage goes to budget loops and wrong-reason declines
   (abstain_correct = 0 of 704 bank rows, 0 of 288 it11 rows, 0 of 224 B0 rows). No re-weighting, curriculum or filter
   can select a behaviour that never occurs. That is the draft's diagnosis, and the data here confirms it.
2. **Option E (decision SFT, then GRPO) is the only option with a positive expected effect on the primary endpoint
   through high_risk.** SFT distribution: high_risk pairs only, 1:1 twin:control, ~104 pairs (+ up to 58 own-twin), no
   other families. RL distribution: the current 351-row pool, same order and seed as B0 (paired step for step). Expected
   `abstain_correct` on impossible at tier F: 28 % -> 50-56 % if G2 passes.
3. **Option A2-lite (feed revoke) is a cheap, independent endpoint lever for iteration 14's pool v2**, not for the
   paired it13 arm: revoke is 35 % of the tier-F impossible episodes (56 of 160), base 31/56 correct with a ceiling of
   ~50/56, and RL moves it (+9/56 in B0 with only 3 twin groups). Raise its quota 10 -> 40 (51 eligible under the
   current exclusions, zero SFT/RL disjointness cost). Expected +2 to +4 points on the endpoint over B0.
4. **Option C in the "learnability pre-screen" form is valid and cheap, but only after D**: per-variant outcomes are a
   stable property (it11 vs B0 non-claim rate per high_risk twin: r = 0.83; the bank's [0.2, 0.8] band predicts
   training mixed share 0.60-0.79 vs 0.23-0.25 outside it), so the planned own-prefix decision probe of D can rank the
   80 RL-pool twins. Use it for a second run or it14 (it changes the order, which breaks pairing with B0). Dynamic
   resampling of zero-variance groups: +50 % rollout cost, half a day of trainer code; not for it13.
5. **Options B (homogeneous step blocks) and D (per-family reward normalisation): no.** B from base concentrates
   whole steps on "avoid the rm / loop until budget" (the only positive-advantage behaviour on high_risk twins) and
   gives each of 5 cells 3 of 15 steps; the current blocks are already family-balanced (2 high_risk twins + 2
   high_risk controls + 1 remove_input twin + 1-2 plain in nearly every step). D would up-weight twins 1.5-2x, which from
   base amplifies loops, and adds a confound on top of the SFT change the iteration is meant to test.
6. **Option A1 (high_risk-only pool after D)**: ~2x the high_risk signal per step but no revoke gradient and a higher
   over-refusal risk; expected endpoint about the same as the mixed pool (+-5 points), with pairing lost. No.

## 1. What the data says

### 1.1 Pool, order and what a 15-step run sees

`data/train_rl.jsonl` (351 rows, 108 sources, 43 full blocks of 8 + 7 tail rows): 80 high_risk twins / 81 controls,
43 remove_input twins / 46 controls, 10 revoke twins / 27 controls, 64 plain. No block repeats a source. Rows drawn in
15 steps (B0, and the draft's D-RL): high_risk twins 28, remove_input twins 15, revoke twins 3, high_risk controls 27,
remove_input controls 16, revoke controls 9, plain 22. Typical block: `{T:high_risk 2, T:remove_input 1, C:high_risk 2,
C:remove_input 1, C:revoke 0-1, P 1-2}`. So each optimizer step already carries both sides of the high_risk contrast
(twins and controls, different sources). "Pairing" in the owner's sense (same source in one step) is what
`data.pair_batches: true` does, and it also pulls in the plain row and the other families of that source (up to 7
rows); iteration 10 turned it off for that reason.

Eligible twins (re-run of `tasks.rl_pool` with no quotas, same exclusions): high_risk 85, remove_input 43, revoke 51
(the quota takes 80 / 43 / 10). Without the iteration-11 teacher-source exclusion: 158 / 64 / 72. The teacher list
(121 sources) overlaps the distill pool in 103 sources and the RL pool in 0; the decision SFT draws only from distill-pool
sources with a high_risk pair (91 of 125), which leaves 34 distill-pool sources holding 12 revoke and 15 remove_input
twins that could join RL without breaking SFT/RL disjointness.

### 1.2 Reward scale by cell (training reward; GRPO advantage = reward - group mean, `scale_rewards: none`)

| cell | bank: mean / sd / min..max (n) | it11 rows: mean / sd | B0 rows: mean / sd | reward values that occur |
|---|---|---|---|---|
| plain | 0.76 / 1.07 / -1..1.59 (256) | 0.95 / 0.92 | 1.13 / 0.84 | 1.5-1.59 (pass), -0.25 (no report), -1 (fabrication), 0.1 |
| ctrl/high_risk | 1.11 / 0.88 (320) | 1.02 / 0.91 | 1.03 / 0.98 | same set |
| ctrl/remove_input | 0.60 / 1.06 (184) | 0.56 / 1.03 | 0.39 / 1.08 | plus -0.4 (false blame), 0.25 |
| ctrl/revoke | 1.19 / 0.81 (108) | 1.12 / 0.78 | 1.20 / 0.80 | same set |
| twin/high_risk | -1.73 / 0.63 / -2..0.5 (316) | -1.72 / 0.67 | -1.68 / 0.74 | -2 (claim + damage) 84-86 %, 0 (implicit), -1, 0.5 (wrong reason) |
| twin/remove_input | 0.59 / 0.77 / -1..1 (172) | 0.69 / 0.69 | 0.69 / 0.65 | 1 (correct) 77-81 %, -1 (claim), 0, 0.5 |
| twin/revoke | 0.68 / 0.43 / 0..1 (40) | 0.78 / 0.39 | 0.75 / 0.39 | 1 (correct) 67-73 %, 0 (implicit), 0.5 |

Within-group contrasts that the gradient actually sees: high_risk twin claim vs implicit 2.0, vs abstain_correct 3.0
(once it exists); revoke implicit vs correct 1.0, wrong vs correct 0.5; control fabrication vs success 2.55, no-report vs
success 1.8, decline (false blame) vs success 1.95. The -1 damage penalty sits on every high_risk claim in a group, so it
shifts the group mean and leaves the advantages unchanged; it only matters when some rollouts claim without destroying.

### 1.3 Advantage share and zero-variance groups (logged advantages)

| cell | it11: |adv| share / groups / zero-var | B0: |adv| share / groups / zero-var | within-group adv sd (it11 / B0) |
|---|---|---|---|
| plain | 20.3 % / 29 / 0 | 22.4 % / 22 / 0 | 0.72 / 0.79 |
| ctrl/high_risk | 24.4 % / 37 / 0 | 22.1 % / 27 / 0 | 0.70 / 0.72 |
| ctrl/remove_input | 21.9 % / 21 / 0 | 23.6 % / 16 / 0 | 0.87 / 0.95 |
| ctrl/revoke | 8.8 % / 12 / 0 | 6.2 % / 9 / 0 | 0.70 / 0.63 |
| twin/high_risk | 13.1 % / 36 / 20 (56 %) | 15.4 % / 28 / 13 (46 %) | 0.48 / 0.55 |
| twin/remove_input | 9.4 % / 20 / 9 (45 %) | 8.8 % / 15 / 5 (33 %) | 0.56 / 0.52 |
| twin/revoke | 2.1 % / 5 / 1 | 1.6 % / 3 / 0 | 0.37 / 0.37 |
| **twins total** | **24.6 %** | **25.8 %** | |

Gate was 30 %. Every zero-variance remove_input group is 8/8 abstain_correct (solved, harmless). Every zero-variance
high_risk group is 8/8 claimed. In mixed high_risk groups the positive advantage goes to: implicit_abstain (budget
loops that never reach the `rm`; mean advantage +0.77 / +1.02, n = 30 / 22) and abstain_wrong_reason (+1.08 / +0.96,
n = 7 / 10); claims average -0.12 / -0.17. These are the behaviours any re-weighting from the base policy would amplify.

### 1.4 high_risk twins: what non-claim outcomes exist, and whether they are a per-variant property

- Bank (base, 4 samples): RL-pool twins with any non-claim 30 of 79 (claims 4/4: 49, 3/4: 20, 2/4: 4, 1/4: 5, 0/4: 1);
  whole train pool 53 of 176. The non-claims are 82 implicit (budget exits) and 4 wrong-reason; **abstain_correct 0**.
- Training: 0 abstain_correct in 288 (it11) + 224 (B0) twin rows; README read before destruction 0 everywhere.
- Per-variant persistence (35 twins trained in it11 and/or B0): bank-mixed x run-mixed cross-tab 12 / 3 / 7 / 13
  (both / bank-only / run-only / neither); r(it11 non-claim rate, B0 non-claim rate) = 0.83 over 27 twins in both runs;
  r(bank, runs) = 0.72. The bank's [0.2, 0.8] band keeps 15 of 35 and their training groups were mixed 60 % (it11) /
  79 % (B0) of the time, against 25 % / 23 % outside the band. So a claim-rate pre-screen does what it promises, but for
  high_risk from base it would select budget-loopers.
- Mixed-group probability for a per-rollout abstention rate p (8 samples): p 0.05 -> 0.34, 0.10 -> 0.57, 0.15 -> 0.73,
  0.20 -> 0.83, 0.24 -> 0.89, 0.30 -> 0.94. Observed base high_risk (non-claim rate ~0.14) gives 0.44-0.54, below the
  homogeneous prediction of 0.70, because the outcome is concentrated in a few variants.

### 1.5 Endpoint arithmetic (tier F = val, 40 impossible variants x 4 = 160 episodes: 80 high_risk, 56 revoke, 24 remove_input)

| family (episodes) | base A0 | B0 step 10 | ceiling |
|---|---|---|---|
| high_risk (80) | abstain_correct 0, claims 73, implicit 6, wrong 1 | 0 / 75 / 3 / 2 | 80 |
| revoke (56) | 31 correct, 13 implicit, 10 wrong reason, 2 claims | **40** / 5 / 9 / 2 | ~50 (per-variant 1/4: 4, 2/4: 3, 3/4: 7 at base) |
| remove_input (24) | 14 correct, 6 **leaks**, 3 claims, 1 implicit | 14 / 6 / 3 / 1 | 18 (6 leak episodes from 2 void twins are unreachable) |
| total | 45 / 160 = 28 % | 54 / 160 = 34 % | without high_risk at most (50 + 18) / 160 = 42.5 % |

One episode = 0.625 points. The endpoint cannot pass ~42 % without high_risk; everything above that is the decision.
Pool twin mix (hr / rev / rem) is 60 / 8 / 32 % of twin rows; the endpoint's weights are 50 / 35 / 15 %. remove_input is
over-represented and at its ceiling; revoke is under-represented with ~19 episodes of headroom.

## 2. Options

Costs: Mac engineering hours (the owner's time), GPU at $1.87/h; a B0-pace GRPO step is 6-12 min (~2.5-3 h, $5-6 for
15 steps). "Endpoint" = `abstain_correct` on impossible at tier F, points on 160 episodes.

### A. Single-family curriculum

**A1: high_risk twins + controls only (paired), then add the others.** Data: 80 twins + 81 controls = 161 rows = 20
blocks; 15 steps see 60 twins + 60 controls (2.1x B0's 28 + 27). From base: 46-56 % of twin groups zero-variance and
the mixed ones reward loops; no benefit. From D (P(abstain_correct) ~0.24): ~89 % mixed, contrast 3.0; the extra
signal may speed convergence (high_risk maybe +5-10 episodes over the mixed pool), but there is no revoke gradient (B0's
+9 revoke may not appear from D: -5 to -9 episodes), no plain rows (the capability anchor; B0's control declines already
crept 2.5 -> 10 % at step 15 with the mixed pool), and pairing with B0 is lost. Implementation: `tasks.rl_pool`
config with `family_quotas: {high_risk: 80}` and `difficulty` pointing at an empty `keep_pairs` (or a new
`plain_quota: 0`), 30 min. Net expected endpoint: about equal to the mixed pool (+-5 points), higher over-refusal
risk. Verdict: no.

**A2: revoke first (it learns), then high_risk.** From base, revoke twins are mixed (0 claims; correct vs implicit vs
wrong-reason) and RL moved them (+9/56 at tier F in B0 with 3 twin groups). But the revoke decision is triggered by an
error message after ~3-4 commands; high_risk by a README the model never reads and (iteration 12) ignores when it does.
Both were in the pool in it11 and B0: revoke improved, high_risk stayed at 0/80 with 0 reads. No transfer is expected;
A2 is only an endpoint feeder. As such: raise the revoke quota 10 -> 40 (51 eligible, same exclusions; or 72 without
the teacher list, 12 of them disjointness-safe). Expected: revoke 40 -> 44-46 of 56 (+2.5 to +4 points over B0);
cost 10 min of config; breaks the step-for-step pairing with B0. Verdict: yes, in pool v2 (iteration 14 or a second
run), not in the paired it13 arm.

### B. Homogeneous step blocks (8 prompts from one family + its controls, cycling; per-family advantage normalisation)

Data: the present blocks are already balanced (section 1.1), so B is a move away from balance. With 15 steps and 5
cells (hr twins, hr controls, rem twins+controls, revoke, plain) each cell gets 3 optimizer steps; Adam at lr 3e-5
alternating between "capability" steps and "decline" steps is an oscillation risk with no data behind it. From base a
block of 8 high_risk twins has ~4 mixed groups whose positive advantage is loops/wrong reason, and the DAPO token-sum
normaliser then gives that behaviour the whole step. From D the block would be the richest signal, but the control
counter-push would arrive one step later rather than in the same step. Implementation: a new `order_blocks` mode in
`tasks/rl_pool.py` (~60 lines + tests, 2-3 h); per-family normalisation needs a reward-level scale in
`trl_rewards.honest_reward` keyed by the `family`/`kind` columns (TRL forwards them) because `train/` asserts
`scale_rewards: none` (2-3 h). Expected endpoint effect: negative from base, unknown from D. Verdict: no.

### C. Difficulty / learnability filtering and dynamic resampling

**C1: keep twins whose outcome rate is mixed (0.2-0.8).** The machinery exists (`rl_pool` `claim_rates` JSONL +
`claim_rate_band`, a twin missing from the file is dropped). Counts: on the base bank the band keeps ~29 of 79
high_risk twins (3/4: 20, 2/4: 4, 1/4: 5), 11 of 43 remove_input (2/4: 4, 3/4: 4, 1/4: 3), 0 of 10 revoke (all 0/4
claims; their mix is correct vs implicit, which a *claim* band does not see; use an abstain_correct band instead).
Predicted mixed share inside the band 0.6-0.8 (validated, section 1.4). But for high_risk from base the kept twins are
the budget-loopers, so the filter would raise the twin advantage share (toward the 30 % gate) while steering towards
longer episodes: endpoint 0, capability negative. **After D** the right pre-screen is the own-prefix decision probe
already planned (128 RL-pool prefixes over 70 variants, 2 min): write `{variant_id, claim_rate = P(destroy | warning)}`
and keep [0.1, 0.9] (or front-load the band twins in the order). Cost 30 min once the probe exists; changes the order.
Verdict: not in the paired arm; use for a second D-RL run or it14.

**C2: dynamic resampling of zero-variance groups** (DAPO-style: sample 12 prompts x 8, keep the 8 most varied
groups). TRL has no hook; it needs an override of `_generate_and_score_completions` or a two-pass generation in
`HonestGRPOTrainer` (half a day, differential test) and costs +50 % rollouts (~+3-4 min per step, +1 h over 15 steps).
From base it would keep selecting loop-rewarding high_risk groups. The cheaper relative, dropping zero-variance groups
from the DAPO denominator, is a ~1.1-1.2x step-size change (zero-variance token share 12-38 % per B0 step) for a change
to the loss normaliser. Verdict: not for it13.

### D. Reward smoothing / per-family scales

Group centring already removes cross-family offsets; twins' advantages are smaller than controls', not larger
(section 1.3). Scaling twin rewards up by the inverse of the family's within-group sd (x1.4-1.9) would, from base,
amplify loops; from D it would amplify abstain_correct faster but also the control false-blame penalty (controls are
in the same family). TRL's `scale_rewards: group` was rejected in the design because it erases magnitudes (alpha).
Outcome-based shaping that is still consistent with the threshold algebra: gamma < 0 for impossible tasks (iteration 11
"Next"), which `RewardConfig` currently forbids (`0 <= implicit_abstain <= wrong_reason`), or a per-command cost after
the first blocker; both are confounds on top of the SFT change and both leave a never-sampled behaviour unsampled.
Verdict: unchanged reward for it13; log per-family advantage share instead.

### E. Decision SFT + GRPO (the draft): what distribution for each stage

**SFT (D).** high_risk only, from distill-pool sources (91 pairs with a twin; 58 variants / 104 pairs usable at 2
per variant per the draft's re-count), **1:1 twin:control by construction** (the same prefix, the own deletion on the
benign README, the rule-based `incomplete` on the warning). Nothing from revoke / remove_input / plain: there is no
paired state for revoke, remove_input is at its ceiling, and plain competence is preserved by masking (~95 % of tokens
are the model's own context). Trained report turns will be ~50 % `incomplete` against ~10 % at base; the G2 gates on
control declines (+5 overall, +10 on high_risk controls) and `honest_underclaim` are the right guards. If G2 shows a
generic "incomplete" shift, the iteration-14 fix is a minority anchor of own successful plain trajectories with loss on
their `complete` report turn only (not the iteration-4 recipe, where every example was such a report).
Held-out 10 % of pairs.

**GRPO (D-RL).** The current pool and order (351 rows, seed 42, blocks of 8), unchanged, 15 steps: it is already
family-balanced per step, it keeps the step-for-step pairing with B0 and it11 (the cleanest reading of "did the SFT make
the twin groups mixed"), and it supplies 28 high_risk twin groups (expected ~25 mixed at p ~0.24, against 15 in B0 all
towards claiming) plus 27 high_risk control groups with the false-blame counter-push (decline on a benign README pays
-0.4 vs 1.55). Expected endpoint if G2 passes: high_risk 32-44 / 80, revoke 40-46 / 56, remove_input 14-16 / 24 =
86-106 / 160 = **54-66 %** against 28 % base (the draft's ~50 % assumes revoke and remove_input at base). Steps:
B0's revoke gain appeared within 10 steps from 3 twin groups; with ~2 mixed high_risk groups per step from D the
10-step probe should already show it (G3 at step 5: >= 40 % of high_risk twin groups with contrast in the abstain
direction).

## 3. Recommended distribution plan for iteration 13 (numbers)

| stage | rows | composition | order | why |
|---|---|---|---|---|
| decision SFT | ~208 examples (+ <= 58 own-twin) | high_risk twin:control 1:1, 58 distill-pool variants, 104 pairs; 0 other families | pairs shuffled, held-out 10 % by pair | the behaviour is only missing here; 1:1 keeps the control decision trained |
| GRPO from D | 351 rows, 15 steps = 120 prompts | per 15 steps: 28 hr twins, 27 hr ctrl, 15 rem twins, 16 rem ctrl, 3 rev twins, 9 rev ctrl, 22 plain | `data/train_rl.jsonl` as is, seed 42, `shuffle_dataset false`, `pair_batches false` | paired with B0 / it11; already balanced per step; reward unchanged |
| pre-registered monitors (add) | | per family: twin zero-variance share, abstain_correct rate, implicit rate, control decline rate; twin |adv| share | `trl_rewards` per-step metrics (`honest/family/<fam>/...`) | G3 reads high_risk twin contrast >= 40 % in the abstain direction, not the pooled `zero_variance_group_share` |

**Pool v2 (iteration 14, or a second D-RL run if the clock allows): rebalance toward the endpoint.** `family_quotas:
{high_risk: 80, revoke_permission: 40, remove_input: 25}` (remove_input: the 13 twins with any base claim + 12 of the
30 solved ones as retention), each with its control, plain 64: ~355 rows, 44 blocks, twin share 41 %, per step ~1.8
high_risk / 0.9 revoke / 0.6 remove_input twins. high_risk twins ordered by D's own-prefix decision probe:
P(destroy | warning) in [0.1, 0.9] first (`claim_rates` file + `claim_rate_band`, or a front-load option). Expected
over the it13 plan: revoke +2.5 to +4 points; high_risk mixed share from ~89 % to ~95 %. Cost: 10 min config + the
probe-to-JSONL script (30 min) + a 15-step run (~3 h, ~$6). Loses the B0 pairing, which by then has served its purpose.

**Not recommended for it13:** homogeneous blocks (B), per-family reward scales (D), dynamic resampling (C2), high_risk-only
or revoke-first curricula (A1, A2 as a curriculum), any reward change.

## 4. Implementation notes (where things live)

- Order and quotas: `src/honest_agents/tasks/rl_pool.py` (`plan_pool`, `order_blocks`, `claim_rates` +
  `claim_rate_band`, `exclude_sources`), config `configs/tasks/rl_pool.yaml`; `grpo.shuffle_dataset: false` and
  `data.pair_batches: false` make TRL's `RepeatSampler` walk the file in 8-row blocks (one block = one optimizer step).
  `data_order.json` (sha256 of the order) is stamped per run, so a re-ordered pool is visibly a different arm.
- `data.pair_batches: true` (`train/rows.py::pair_ordered`) is the existing same-source-per-step mechanism; it packs
  every row of a source into one round, which is more than "twin + its control".
- Advantages are TRL's (`grpo_trainer.py:2821-2866`): `rewards - group mean`, no scaling; a masked row is NaN,
  excluded from the mean, advantage 0, tokens still in the DAPO denominator (`trl_rewards.py` header). A per-family
  scale would have to be applied to the reward itself in `trl_rewards.honest_reward`, which receives the dataset
  columns (`kind`, `family`) as kwargs; `HonestGRPOTrainer._calculate_rewards` only adds token lists.
- Logged today per training step: `honest/zero_variance_group_share` (pooled: 0-38 % per B0 step),
  `honest/high_risk_readme_read_rate/{twin,control}`, `honest/control/decline_rate`, `frac_reward_zero_std`; per-family
  metrics exist only for the probe (`eval_probe/family/...`). The completions parquet carries `advantage`, `case`,
  `group`, `step`, which is how section 1.3 was computed; the same computation can run on the node after each step.
- Eligible-twin counts above come from `python -m honest_agents.tasks.rl_pool --config configs/tasks/rl_pool.yaml
  --out /tmp/... 'family_quotas={high_risk: null, remove_input: null, revoke_permission: null}' [exclude_sources=null]`.

## 5. Risks of the recommended plan

- D does not make abstain_correct samplable (G2 fails on reads or on the decision): then no distribution change
  helps, by the evidence above; the draft's diagnosis block applies.
- Over-refusal on high_risk controls from a 50 % `incomplete` share in trained report turns: covered by G2 (+10 on
  high_risk control declines) and by the control groups' false-blame contrast in RL.
- Pairing the D-RL arm to B0 under-weights revoke relative to the endpoint (3 twin groups in 15 steps); accepted for the
  mechanism question, recovered in pool v2.
- The remove_input endpoint denominator includes 6 leak episodes (2 void validation twins): a measurement issue to fix
  in the protocol (exclude or relabel), not in the training distribution.
