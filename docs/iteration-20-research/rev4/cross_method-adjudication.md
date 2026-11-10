# Cross-examination: the RL-method disagreements (iteration 20, revision 3 to 4)

Role: cross-examiner. One decision per topic: adopt (with the value), reject, or measure first (with the measurement,
sample size and pass bar). Everything here is read-only on the repo and the archive, train-side data only.

## 0. What I did

**Read.** In full: orchestrator_notes.md and 14 reports (grill_method, grill_history, grill_code, grill_data to its
last finding (verdict and all 13 findings), evidence_dynamics, evidence_literature, evidence_rl-history, research_twin-judgment-signal,
research_algorithm-feasibility, research_lit-honesty-rl, research_lit-agentic-rl, research_throughput,
research_headroom-pool-model, research_pass-fail-anatomy, research_failure-taxonomy). By section (verdict plus the
findings that touch these ten topics): grill_metrics (verdict, S4, N8), grill_program (verdict, P2, P6),
evidence_measurement (summary, facts 10, 11, 18, 28). I did not read the remaining findings of those three.

**Checked in code** (HEAD 5a91a9d, TRL 1.14.1, PEFT 0.21.2):
- docs/iteration-20-rl-design.md (lines 92-126, 287-476)
- src/honest_agents/reward.py (lines 172-179, 697-760)
- src/honest_agents/trl_rewards.py (line 69: `PER_EPISODE_MODES` has no `u`; mode `u` is not written yet)
- src/honest_agents/trainer.py (lines 34-44, 294-327, 438-454, 722-732)
- .venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py (lines 2539-2549, 2770-2792, 2916-2929, 3157-3166, 3207-3235)
- .venv/lib/python3.12/site-packages/trl/generation/vllm_generation.py (lines 440-470)
- .venv/lib/python3.12/site-packages/peft/tuners/lora/layer.py (lines 1005-1008, 1034-1036, and the Linear forward at about lines 1082-1113)
- src/honest_agents/sft/merge.py (lines 24-56)
- configs/grpo_colocate_qwen35.yaml (lines 18-21), configs/base.yaml (lines 18, 123-170)
- docs/iteration-18-design.md (lines 9-16, 30-50), docs/results.md (lines 37-40), docs/training-log.md (lines 1756-1759)

**New computations** (scripts and outputs in
~/Documents/honest-agents-archive/staging/analysis/it20_rev4/xmethod/ ;
run from the repo root with `env -u VIRTUAL_ENV uv run --frozen python <script>`):

| script | what it does | data |
|---|---|---|
| twin_weights.py (output twin_weights.out) | the four twin-weight proposals on G2's banks, proceed-conditional, source bootstrap 2,000 draws, 90 % | readpush/rows.jsonl built by research:twin-judgment-signal (train/val sources only, DEF-66 out) |
| below_band.py (below_band.out) | what plain U reinforces on rows with G2 pass at most 3/8, against in-band rows | same table: pool bank it17 (S = 4), select (three G2 runs, S = 8) |
| contrasts.py, retire.py (.out) | pass/fail contrasts per 1,000 generated episodes under filtering, refill, allocation, group size | beta-binomial fits from evidence:dynamics; no data file read |
| bf16_survival.py, bf16_by_module.py | share of a LoRA delta that survives the bf16 merge that vLLM samples from and `sft.merge` evaluates | archived it19c-ablation adapters (checkpoints 5-30), cached Qwen3.5-4B base weights |
| sync_gap.py | sampler-trainer gap against KL, per step | outputs/it19/it19-crgrpo-ablation-r2/train_log.jsonl |

No held-out path was opened. Held-out numbers below are the aggregates already printed in the documents.

Labels used: **verified** = I ran it or read the code line; **cited** = a report's number I did not re-run;
**derived** = arithmetic or a model on top of verified inputs; **not verified** = said so.

## 0.1 Decisions at a glance

| # | topic | decision | confidence |
|---|---|---|---|
| 1 | twin weights | **Adopt** neutrality by schedule, weight as the residual: w_k = min(1, r_k · G_c,k / L_t,k), with r_k controls of the kind per twin in the frozen pool (lone controls counted). Build the pool at r_k about 2.3 (surface), 2.9 (data/tasks high_risk), 2.0 (remove_input), 1.6 (revoke); w_k then lands at 0.85-1.0. **Reject** the revision-3 formula, the refusal penalty k as the neutrality device, and w = 1 at one control per twin for remove_input / revoke. | high on the rejections, medium on the values |
| 2 | KL to G2 | **Adopt** β = 0.04 fixed, described as a meter. **Reject** an adaptive β and a decision-token anchor in the main arm. Stop levels: amber 0.015, 0.02 needs a passed learning gate to continue, hard stop 0.04. | medium-high |
| 3 | lr, batch, steps | **Adopt, conditional on three measurements:** lr 1e-5, 32 × 8 = 256 episodes per step, 300 steps (76,800 episodes), pre-registered extension to 450 (115,200). Fallback 1e-5 × 128 × 450. Never below 1e-5 with bf16 serving. | medium-low (the measurements decide) |
| 4 | zero-variance groups | **Adopt** offline allocation by pass count, no always-pass rows, one unit per source (+21 to +26 % contrasts per 1,000 batch episodes). **Adopt if the data-order check can take it** retirement of all-same rows at epoch boundaries (+8 to +19 %). **Reject** within-step filter-and-refill as a signal lever (0 % per generated episode) and skipping the backward pass. The three are compatible. | high |
| 5 | group size | **Keep G = 8.** Reject 16 at a fixed rollout budget (+7 % contrasts, half the prompts). | high |
| 6 | loss and truncation | **Keep** DAPO token-sum, truncated episodes unmasked, a missing report scores 0. **Measure** three aggregations in the zero-lr audit before any switch. Add a +2-point stop rule on missing or zero-command reports. | medium |
| 7 | below-band rows | **Exclude** from the whole-episode gradient pool. **Use** about 100 spec-audited ones as a named probe, and as prefix sources for the decision-point arm. | medium-high |
| 8 | sampler/trainer gap | **Reject** the probability-gap token mask. An fp32 head is not a config switch here (tied embeddings). Keep the truncated importance ratio; add two monitors. | medium-high |
| 9 | ECHO loss, SFT replay | **Reject** both in the main arm. ECHO: at most one exploratory arm with an advantage-zeroed twin, after the main arm passes its learning gate. SFT replay: reject. | medium |
| 10 | decision-point GRPO | **Parallel arm, gated.** Not the main mechanism until three no-gradient tests pass; a mistake if run endgame-only, at a 1:1 prefix mix as the only neutrality device, or on unaudited never-pass prefixes. | medium |

---

## 1. Twin weights

### Positions
- **Revision 3** (design lines 364-380): w_k = (2·p_c − 1 − δ_c)/(1 + δ_t) with p_c the unconditional pass rate of the paired controls; worked value 0.44 at p_c 0.744 and δ 0.1; kinds with w ≤ 0.2 lose their twins.
- **research:twin-judgment-signal**: w that leaves a non-reader indifferent: surface 0.44, data/tasks high_risk 0.33-0.36, remove_input 0.51, revoke 0.63-0.80.
- **research:lit-honesty-rl, rank 1**: w toward 1 with a penalty k on verified false refusals of feasible rows (k 0.3 with w 0.72, or k 0.6 with w 1.0).
- **grill:method M3**: neutral weights for high_risk only; w = 1 for remove_input and revoke; an arm W with w = 1 everywhere.
- **grill:history F6**: the 100 lone controls break pair neutrality toward proceeding by +0.10 to +0.12 per control-looking episode.

### Facts checked (twin_weights.py; verified)
Definitions: proceed = claimed complete or caused damage. G_c = mean U of control episodes that proceeded. L_t = mean
unweighted loss (claim + damage) of twin episodes that proceeded. Not proceeding scores 0 on both sides.

| kind (G2 banks) | sources | twin eps / proceed | control eps / proceed | G_c | L_t | w as written | w, one control per twin | controls per twin at which w = 1 is neutral |
|---|---|---|---|---|---|---|---|---|
| surface high_risk (pool it16) | 85 | 719 / 333 | 720 / 561 | 0.850 [0.796, 0.900] | 1.961 [1.935, 1.982] | 0.224 [0.187, 0.261] | 0.434 [0.406, 0.459] | 2.31 |
| data/tasks high_risk (pool + select) | 161 | 634 / 100 | 852 / 783 | 0.701 [0.621, 0.777] | 1.910 [1.856, 1.955] | 0.291 [0.244, 0.336] | 0.367 [0.324, 0.409] | 2.72 |
| same, non-readers only | 161 | 103 / 73 | 161 / 121 | 0.603 [0.441, 0.745] | 1.890 [1.821, 1.946] | 0.112 [0.021, 0.204] | 0.319 [0.234, 0.394] | 3.13 |
| remove_input | 64 | 104 / 31 | 256 / 200 | 0.490 [0.337, 0.627] | 1.000 | 0.180 [0.008, 0.336] | 0.490 [0.337, 0.627] | 2.04 |
| revoke_permission | 64 | 212 / 2 | 312 / 285 | 0.642 [0.484, 0.793] | 1.000 | 0.519 [0.361, 0.678] | 0.642 [0.484, 0.793] | 1.56 |

These reproduce research:twin-judgment-signal within its intervals (0.44; 0.33-0.36; 0.51; 0.63-0.80). P(damage | proceed)
on high_risk twins is 0.96-0.985, not 0.1.

Arithmetic of each proposal, as blind proceed minus blind decline (positive = the reward pays a policy that has not
read the evidence for proceeding, which on a twin means destroying it):

| scheme | surface pair | data/tasks high_risk pair | remove_input pair | revoke pair |
|---|---|---|---|---|
| revision 3 as written | +0.41 | +0.15 | +0.31 | +0.12 |
| non-reader indifference, one control per twin | 0 | 0 | 0 | 0 |
| w = 1, one control per twin (grill:method M3 for the last two; arm W) | −1.11 | −1.21 | −0.51 | −0.36 |
| k = 0.3 with w = 0.72 (lit-honesty's starting point) | −0.28 | −0.49 | about −0.01 | about +0.07 |

- **Lone controls (F6).** With revision 3's composition and hash-order lone controls (about 82 of 100 are data/tasks
  high_risk; cited from grill:data m3), data/tasks high_risk has 2.64 controls per twin. Per control-looking episode
  of that kind, blind proceeding is then worth +0.36 at the written weight, +0.32 at the pair-neutral weight and
  **−0.02 at w = 1**. Pooled over all kinds on revision 3's step (4 twins, 4 paired controls, 2 lone controls):
  +0.26 per control-looking episode at the written weights and +0.14 at pair-neutral weights (derived from the table).
  So F6 is right in sign and understated in size; the plan's own tolerance is ±0.05.
- **Penalty k.** For w = 1 to be neutral at one control per twin, k must be 1.16 [1.10, 1.23] (surface) and
  1.94 [1.58, 2.37] (data/tasks high_risk). Both are above 1, which breaks the order "a false claim costs more than a
  refusal" that lit-honesty itself requires. Only remove_input (0.70 [0.49, 0.97]) and revoke (0.74 [0.41, 1.17]) are
  reachable. With measured inputs, k = 0.3 gives a neutral w of 0.58 (surface) and 0.47 (data/tasks), not 0.72;
  k = 0.6 gives 0.73 and 0.56, not 1.0. The report's arithmetic used p_c 0.744 and δ 0.1, the same two inputs
  revision 3 got wrong.
- **The penalty can be dodged.** `false_blame` needs status `abstain` or an external blocker that is false, a decoy or
  unchecked (reward.py lines 712-714). A non-claim with no blocker is `honest_failure`. `lazy` needs fewer than
  `min_attempt_actions` = 1 commands (line 178, line 729), so it catches only zero-command reports. Share of G2's
  control refusals that carry either flag: surface 0.96 (n = 159), remove_input 0.73 (n = 56), data/tasks high_risk
  0.62 (n = 69), revoke 0.48 (n = 27). A blind refuser that reports "incomplete" without naming a blocker pays nothing.
- **Closeness to the evaluated objective** (price of a destroyed-and-claimed high_risk twin; 1.9-2.0 at evaluation):
  revision 3 as written 0.44-0.56; pair-neutral 0.70-0.85; k = 0.3 variant 0.89-1.14; schedule-neutral 1.6-1.96.

### Decision: adopt "neutral by schedule, weight as the residual"
Neutrality depends on the product w_k × (twins per control of the kind), not on w_k alone. So fix the balance mostly
with the row mix and keep the per-episode reward at the evaluated U:

1. **Formula.** w_k = min(1, r_k · G_c,k / L_t,k). r_k = controls of kind k in the frozen pool (paired and lone) per
   twin of kind k. G_c,k and L_t,k are proceed-conditional, from an unbiased G2 pass on the frozen pool (not the
   samples that selected it). For high_risk kinds use non-readers when each side has at least 30 unread proceeders,
   else all proceeders, and report both.
2. **Composition.** Build each kind at r_k ≈ L_t / G_c: 2.3 (surface), 2.9 (data/tasks high_risk, between the two
   estimates), 2.0 (remove_input), 1.6 (revoke). Per 16 prompts this is about 3 twins, 8 same-kind controls and
   5 plain rows (6 / 16 / 10 at 32 prompts). The feasible mix 8 : 5 is close to the held-out one (158 controls,
   118 plain).
3. **Expected weights.** 0.85-1.0. If the controls are also band-selected for contrast (topic 4) their G_c falls to
   about 0.6 and w = 1 needs r ≈ 3.2; at r = 2.67 the weight is about 0.83.
4. **Check before freezing** (replaces §12 item 5): per kind and pooled, blind proceed minus blind decline per
   control-looking episode within ±0.05 with a 90 % interval that covers 0.
5. **Never** drop a kind's twins while keeping its controls. If a kind cannot reach w ≥ 0.5, cut its twins and its
   controls together.
6. Supply limit to respect: 85 legal remove_input controls (cited), so at most about 40 remove_input twins.

Twin share of the step's reward mass (twins per 16 prompts × w): 3 × 0.95 ≈ 2.9, against 1.0 for revision 3 as
written and 1.8 for pair-neutral weights at 4 twins. The reading signal is not halved to buy neutrality.

**Rejected.** (a) The written formula: it pays blind proceeding on every kind. (b) k as the neutrality device: needs
k > 1 on the kinds that hold 150 of 200 twin pairs, is dodged by wording, and changes the reward on feasible rows.
It may be tried as an arm; it is not a fix. (c) w = 1 at one control per twin for remove_input and revoke: blind
declining wins by 0.51 and 0.36 per pair, and remove_input controls are G2's weakest feasible stratum. Revoke twins
have no gradient either way (2 of 212 proceed).

**Confidence.** High that the written formula is wrong and that the lone controls must be counted. Medium on the
values: every bank ran at 12,288 tokens, revoke rests on 2 proceeders, and a "kind" is my unit for what a blind
policy cannot tell apart; the pooled check in step 4 covers the case where it generalises across kinds.

---

## 2. KL to G2

### Positions
β 0.04 as written with warn 0.05 / stop 0.1; a monitor only at β 0 to 0.01 (grill:method M6, grill:code M1, both
literature reports); raise β to at least 0.2 (evidence:dynamics); an adaptive β from 0.04 to 2.0 holding KL near
0.012 (research:algorithm-feasibility, rec 3); a decision-token anchor built and kept off (same report, rec 6).

### Facts checked
- **The term** (verified, grpo_trainer.py lines 3159-3163 and 3210-3211): β × (exp(d) − d − 1) per token with
  d = ref − current, added to the per-token policy loss. Reference log-probs are computed only when β ≠ 0
  (lines 2770-2792 sit under that branch; line 3159). So a meter needs β > 0.
- **Size** (derived): at KL 0.02 the per-token coefficient is about β·sqrt(2·KL) = 0.008, against a mean |advantage|
  of 0.33-0.40 under U on real groups of 8 (cited, evidence:dynamics fact 8): about 2 %.
- **It did nothing measurable** (verified): LoRA update norm at step 15 is 0.708 for G3 (β 0) and 0.705 for the
  ablation (β 0.04), same lr and batch. Step 30 of the ablation: 1.074.
- **KL trace of the ablation** (verified, sync_gap.py): mean 0.0002 (steps 1-5), 0.0048 (6-15), 0.0149 (16-30);
  0.0213 at step 30.
- **KL levels at which behaviour regressed** (verified in the documents):

| run | what was trained | drift per token | estimator | behaviour |
|---|---|---|---|---|
| it17 | confidence value only, frozen transcripts, KL 0.04 on that value | 0.18 | G2's own action tokens, held-out transcripts | twins destroyed 53 / 120 vs 13 |
| it18 | same + 1.0 × KL over G2's action tokens | 0.009 | same | 31 / 120 vs 13 |
| K2 | status token only, with the action anchor | 0.0031 | same | 21 / 120 vs 13 |
| it19c ablation | whole episodes, β 0.04 | 0.0149 mean of steps 16-30 | TRL, on-policy sampled tokens | select U −6.5 [−11.0, −2.0], pass −3.4 [−6.9, +0.1] (cited) |

- **K2 is inside G2's own range** (verified on the 104 DEF-66-free select twins): G2 destroyed 10 (9.6 %), 12 (11.5 %)
  and 31 of 208 (14.9 %) in three sessions; K2a 19 of 104 (18.3 %); the whole-episode ablation 38 of 208 (18.3 %) at
  step 30 and 10 of 104 at step 15. Binomial SE is about 3-4 points before clustering. So the clear cases are it17
  and it18, and whole-episode GRPO reached K2's level too.
- **What a binding anchor would do** (derived). With a restoring rate κ per step the displacement is an
  Ornstein-Uhlenbeck process: mean μ/κ, standard deviation σ/sqrt(2κ). Signal-to-noise is that of a free run of
  2/κ steps. A binding anchor is early stopping; steps beyond 2/κ buy nothing. The report that proposes the
  controller says the same ("a cap that keeps a run alive, not a source of gain").
- **Literature** (cited): every positive agent recipe that states β uses 0 to 0.001; β 0.04 cost 5.5-7.1 points of
  new-task accuracy in one study; LOOP lost 1-3 points with a KL term. Those are full fine-tuning runs with larger
  KL magnitudes; here the term is 2 % of the gradient, so it can neither protect nor cost much.

### Decision
- **Adopt β = 0.04, fixed, described in §1, §5.4 and §14 as a meter.** Any value in (0, 0.04] is immaterial to the
  update; 0.04 keeps the only logged KL trace comparable and changes one factor fewer.
- **Reject the adaptive β.** Hold the KL budget with lr and steps (topic 3). A controller that binds wastes the
  episodes after it binds.
- **Reject the decision-token anchor in the main arm.** The plain ablation did not drift on claims
  (P(claim | pass) 0.986 → 0.987, cited), no token is masked under U, and stop rule 1 watches it. Build it only if
  rule 1 fires in a pilot.
- **Do not raise β to 0.2 or more.** It would start to bind (11-18 % of the gradient at KL 0.02-0.05) with the
  effect just derived.
- **Stop levels** (KL EMA per token, trainer estimate):
  - amber at 0.015, the level of the only measured harm;
  - at 0.02 the arm continues only if the train-side learning gate has already passed (same-prompt revisit or
    progress set, whichever revision 4 registers);
  - hard stop at 0.04 (revision 3: 0.1).
  - Replace the steps 6-10 extrapolation gate: on the ablation it projects 0.055-0.082 against a bar of 0.08
    (cited from three reports, consistent with the trace above).

**Confidence.** Medium-high. The one thing not measured is whether harm scales with KL at all: there is one data
point at 0.015.

---

## 3. Learning rate, episodes per step, steps

### Positions
Revision 3: 2e-5 × 150 steps × 128 = 19,200 episodes. Alternatives: 1e-5 at the same size (evidence:dynamics);
1e-5 with 256 per step if the hours allow (research:algorithm-feasibility); at least 51,200 episodes at 256 per step
(lit-agentic-rl); 400-500 steps at lr about 1e-5 (evidence:literature); history says a lower lr is not the lever
(evidence:rl-history).

### Facts checked
- **The noise model's bookkeeping** (verified against dynamics/adam_null_out.txt): with s the per-step
  per-coordinate signal-to-noise, the signal share of the squared displacement after T steps is T·s²/(1 + T·s²)
  (s 0.05, T 150: formula 0.27, simulation 0.262; s 0.02: 0.057 and 0.054). s² scales with episodes per step. So the
  share depends only on **total episodes × per-episode signal power**. The learning rate sets how far the policy
  moves, not how clean the move is.
- **Bound from the ablation** (cited): s ≤ 0.02-0.05 at 64 episodes, i.e. per-episode signal power ≤ 0.6e-5 to
  3.9e-5 (CR-GRPO table, pool v4).
- **Indices** (derived). S = lr × steps × sqrt(batch); N = lr² × steps; both relative to revision 3. R = total
  episodes × per-episode power = signal energy over noise energy at the end, at the ablation's bounds. KL at the end
  from the ablation's 0.000725 per step at 3e-5 scaled by lr²; the saturating fit gives about 0.008-0.010 for every
  1e-5 row.

| option | episodes | signal moved S | noise energy N | gain-to-drift S/N | R at the ablation's bounds | projected KL at the end | hours per arm (no fixes / three fixes) |
|---|---|---|---|---|---|---|---|
| 2e-5 × 150 × 128 (revision 3) | 19,200 | 1.00 | 1.00 | 1.00 | 0.12-0.75 | 0.047 | 28 / 20 |
| 1e-5 × 150 × 128 | 19,200 | 0.50 | 0.25 | 2.00 | 0.12-0.75 | 0.012 | 28 / 20 |
| 1e-5 × 450 × 128 | 57,600 | 1.50 | 0.75 | 2.00 | 0.36-2.2 | 0.036 | 84 / 60 |
| **1e-5 × 300 × 256** | 76,800 | 1.41 | 0.50 | 2.83 | 0.48-3.0 | 0.024 | 101 / 73 |
| 1e-5 × 450 × 256 | 115,200 | 2.12 | 0.75 | 2.83 | 0.72-4.5 | 0.036 | 151 / 109 |
| 1.5e-5 × 300 × 256 | 76,800 | 2.12 | 1.13 | 1.89 | 0.48-3.0 | 0.054 | 101 / 73 |
| 2e-5 × 300 × 256 | 76,800 | 2.83 | 2.00 | 1.41 | 0.48-3.0 | 0.096 | 101 / 73 |

  Hours: 668 s per 128-episode step and 1,210 s per 256-episode step, × 0.72 with the three step-time fixes (cited,
  research:throughput results 7 and 16; never measured at either size).
  Revision 3 cannot be signal-dominated even at the top of the bound (R ≤ 0.75). 300 × 256 is signal-dominated only
  if the truth sits in the upper two thirds of the bound's range.
- **Why batch 256 and not just more steps.** At fixed episodes and fixed S, halving the batch needs lr / sqrt(2)
  for the same noise energy. That is 7e-6, which runs into the next fact.
- **bf16 rounding is real at these rates (new; verified in weight space).** vLLM samples from, and `sft.merge`
  evaluates, weights merged in bf16 (vllm_generation.py lines 447-469; layer.py line 1008; merge.py lines 24-56). A
  delta element smaller than half a bf16 step of its weight vanishes. On the archived ablation adapters and the base
  weights (every 8th of 248 modules; every 4th gives 0.478 and 0.772 at steps 10 and 30):

| adapter | rms ΔW / rms W | share of the delta carried (projection) | cosine | weight elements changed |
|---|---|---|---|---|
| ablation step 5 (lr 3e-5) | 0.03 % | 0.20 | 0.39 | 7 % |
| step 10 | 0.07 % | 0.46 | 0.60 | 18 % |
| step 15 | 0.10 % | 0.59 | 0.68 | 25 % |
| step 30 | 0.16 % | 0.76 | 0.78 | 35 % |
| step 30 scaled × 1/3 (lr 1e-5, same step) | 0.05 % | 0.35 | 0.52 | 13 % |
| scaled × 0.83 (lr 1e-5, about step 150 by the measured norm growth) | 0.13 % | 0.69 | 0.74 | 30 % |
| scaled × 1.5 (lr 1e-5, about step 450) | 0.23 % | 0.88 | 0.86 | 46 % |

  The trainer's own forward has the same kind of rounding per activation: the LoRA branch is added to a bf16 base
  output and cast back to bf16 (layer.py, Linear forward; LoRA parameters are fp32 by PEFT's default). That is why
  the sampler-trainer gap stays flat while KL rises (verified: mean |Δ log p| 0.0061, 0.0067, 0.0065 over steps 1-5,
  6-15, 16-30; slope 0.008 ± 0.012 per unit of KL; it follows entropy instead). The flat gap shows that trainer and
  sampler agree with each other. It does not show that the parameter update is fully expressed. grill:code's
  "weight sync is not a concern" is therefore overstated, and the repo's own note that a 1e-5 update "can round
  away" (grpo_colocate_qwen35.yaml lines 18-19) is quantified here.
  **Not verified:** the behavioural size of this (no forward pass was run), and the same numbers on merged G2 (not
  on disk; I used the base model's weights, whose magnitudes are the same to first order).
- **Cluster limit** (cited, research:throughput result 12): 256 concurrent episodes need about 818k tokens of
  context against a colocated KV cache of 482k at utilisation 0.3 on a 96 GB card.

### Decision: adopt, conditional on three measurements
**Value.** lr 1e-5, constant after 5 warmup steps; 32 prompts × 8 = 256 episodes per step; 300 steps = 76,800
episodes per arm. Pre-registered extension to 450 steps (115,200) if the learning gate has passed by step 300 and
KL EMA is below 0.03. Fallback when 256 per step cannot run: 1e-5 × 128 × 450 (57,600). Do not go below 1e-5 with
bf16 serving.

**Measurement 1: the zero-lr gradient audit** (also arm Z). 30 steps × 256 episodes at lr 0 on the frozen pool
(7,680 episodes, about 10 GPU-hours), storing one sketch of the score sum per episode.
- Statistic: ρ = signal power over noise power of a 256-episode step, from the mean cosine c between gradients of
  different steps (c = ρ/(1 + ρ); 435 step pairs) or between half-steps (c = ρ/(ρ + 2)).
- Pass: 300·ρ ≥ 1 with a 90 % lower bound ≥ 0.5, i.e. c ≥ 0.0033. Launch as registered.
- Between 0.3 and 1: register 450 steps from the start, or the estimator or pool variant with the higher measured ρ.
- Below 0.3: do not launch a whole-episode arm on this pool and reward.
- At the end of training, report the cross-seed cosine of the LoRA deltas; the audit predicts R/(1 + R).
- Whether 30 steps resolve c = 0.003 depends on the effective dimension of the gradient noise, which is not known;
  the audit must report the permutation null of its own statistic.

**Measurement 2: delta carried through the merge.** Run bf16_survival.py on every saved checkpoint (8 seconds on a
CPU). Expected at lr 1e-5 if the update is a noise walk: about 0.35 at step 30, 0.46 at step 50, 0.70 at step 150,
0.82 at step 300, 0.89 at step 450. Rule: the candidate checkpoint must carry at least 0.8 (a noise walk at 1e-5 only just reaches it at step 300,
so report it rather than assume it); if the step-50 value is below 0.4, restart at 1.5e-5 (one pre-registered
change). Consequence for the pilot: at 1e-5 a 10-step pilot shows almost nothing
(KL about 0.0004); behaviour rules need 30 or more full-rate steps.

**Measurement 3: the timing rehearsal** (grill:code M4, research:throughput rec 5): two 256-episode steps at lr 0
with one forced full-length episode: step time, peak memory, KV preemptions. If 256 does not fit colocated, use
vLLM in server mode or the 128-episode fallback.

**Confidence.** Medium-low. The direction (several times the episodes, lr no higher than revision 3's) has support
from the noise model, the literature's scale and the cluster's hours. The exact lr rests on a KL scaling fitted to
one run, and the bf16 fact pulls the other way. My probability that the audit passes for whole-episode U on a
reallocated pool is about 40 %.

---

## 4. Zero-variance groups

### Positions
Online filtering with refill (evidence:literature rank 2, lit-agentic-rl rec 2); no filtering, because skipping
removes the KL term from always-pass rows and saves 3 % of wall-clock (research:algorithm-feasibility rec 9); pool
allocation by pass count and one unit per source (grill:data D6).

### Facts checked (contrasts.py, retire.py; derived from the fitted pass-rate distributions)
A group of G samples of a row with pass rate p gives (G − 1)·p(1 − p) pass/fail contrasts, so the unit is contrasts
per 1,000 **generated** episodes. The model's zero-variance share for band rows at G = 8 is 26 % (plain) and 23 %
(controls); real groups of 8 gave 25-28 % (cited).

| option | plain rows | control rows | whole step |
|---|---|---|---|
| revision 3 (band in hash order; 25 % of plain rows always-pass) | 104 | 143 | 93 at 6 plain : 6 control : 4 twin |
| band only, no always-pass rows | 134 (+29 %) | 143 | 104 (+12 %) |
| allocation by pass count, 4-6 of 8 first then 7 of 8 | 140 at a quota of 250 (+34 %); supply-limited beyond about 280 | 174 at a quota of 250 (+21 %); 148 at 400 (+3 %) | 112 (+21 %) at 6 : 6 : 4; 117 (+26 %) at 5 plain : 8 control : 3 twin |
| within-step filter and refill (DAPO dynamic sampling) | 134 per generated episode (**+0 %**); 182 per trained episode at 1.36 × the rollouts | 143 (**+0 %**); 185 at 1.29 × | +0 % per generated episode |
| retire a row after two all-same visits, slots to the rows still active (no reserve) | +11 % at 6 visits, +19 % at 12 | +8 % and +15 % | about +8 to +17 % on the feasible part |
| rows at 4-6 of 8 only, plus retirement | 173-177 | 176-180 | supply: about 160 plain rows after review |

Other numbers: a row at 7 of 8 is mixed in 55-60 % of its groups, a row at 8 of 8 in 11-18 %, rows at 4-6 in
79-95 %. Fresh-row replacement instead of retirement gives +9 to +11 % but needs 39-45 % more unique rows, which the
in-band plain supply (about 330 before review) does not have.

### Are they compatible? Yes
They act at three different places.
- **Pool build:** allocation by pass count, no always-pass rows, one unit per source.
- **Between epochs:** retirement on the run's own rollouts.
- **Within a step:** nothing. Every generated group goes through the loss. A zero-variance group adds no
  policy-gradient term, so "filtering" it changes nothing except removing its KL coverage; "refilling" is only a
  larger batch, and in the noise model a larger batch at the same generated episodes gives the same end-of-run
  signal share. The literature's gains (+8 and +3.8 points) are per optimizer step at more rollouts, and the
  "offline curation +0.0" result is about a policy that moves a long way; here pass rates are expected to move by
  single points.

### Decision
- **Adopt** the pool-build changes: +21 to +26 % contrasts per 1,000 batch episodes.
- **Adopt, if the data-order check can accept a regenerated order** (`--resume` checks it today, cited from
  research:throughput result 18): a deterministic epoch-boundary rule that retires rows whose last two visits were
  all-pass or all-fail. Worth +8 to +19 % more in a run with 6-12 visits per row. It concentrates visits on fewer
  rows, so it needs the held-back-row memorisation check. If the code cost is high, skip it; it is the smallest of
  the three.
- **Reject** within-step filter-and-refill as a signal lever and skipping the backward pass of zero-variance groups.

**Confidence.** High on the ranking; the absolute numbers come from a beta-binomial fit to S = 4 banks at 12,288
tokens and should be recomputed on the prescreen.

---

## 5. Group size 8 or 16 at a fixed rollout budget

### Positions
G = 16 (lit-agentic-rl rec 2; TMax found larger groups reduced collapse); keep 8 and add prompts
(evidence:dynamics, research:headroom-pool-model, research:algorithm-feasibility, lit-honesty-rl).

### Facts checked (contrasts.py)
Band rows, contrasts per 1,000 episodes / zero-variance groups / prompts per 128 episodes:

| G | plain | controls | zero-variance | prompts |
|---|---|---|---|---|
| 4 | 115 | 123 | 0.41-0.45 | 32 |
| 8 | 134 | 143 | 0.23-0.26 | 16 |
| 16 | 144 (+7 %) | 154 (+7 %) | 0.11-0.14 | 8 |

Per-episode efficiency is exactly (G − 1)/G. G = 16 halves the distinct prompts per step; between-prompt variation
of the gradient is then averaged over half as many prompts. The one simulation of that trade-off puts the per-step
signal-to-noise of a shared direction at 3.4 for 16 × 8 against 2.8 for 8 × 16 (cited, research:headroom-pool-model
result 15; one-dimensional, overstated in level). KARL: G = 8 66.8, G = 16 66.2 (cited).

### Decision
**Keep G = 8. Reject 16.** With the per-episode sketches of the audit the between-prompt and within-prompt variance
components can be estimated, which would also say whether 32 prompts × 4 is better; do not adopt 4 without that
(twin groups at claim rates of 15-45 % are mixed 48 % of the time at G = 4 against 73 % at G = 8).

**Confidence.** High.

---

## 6. Loss aggregation, truncated episodes, missing reports

### Positions
DAPO token-sum and unmasked truncation (revision 3); sequence-mean (Endless Terminals, River); an episode-level mask
on budget-limit episodes (DeepSWE, DAPO); score a missing report on a feasible row as −1 (evidence:literature
rank 4).

### Facts checked
- **TRL at micro-batch 1** (verified, lines 3214-3235): `dapo` is the token sum over the whole generation round
  divided by one global count; `grpo` and `bnpo` are each episode's token mean; `dr_grpo` equals `dapo` up to a
  constant. The KL term is inside the per-token loss, so it follows the same aggregation. The repo's length-sorted
  regrouping is exact only for the token-sum losses and is skipped for the others (trainer.py lines 34-44, 729).
- **What "truncated" is** (verified, trainer.py line 296; grpo_trainer.py lines 2539-2547): the last sampled turn
  does not end on EOS. `mask_truncated_completions` zeroes the whole episode's loss and attention mask. It does not
  touch episodes that end at the command cap with a report. Share: 0.3 % of ablation episodes, at most 4.7 % in a
  step (cited).
- **Weight concentration under the token sum** (cited, evidence:dynamics facts 13-14): 5 of 64 episodes carry 38 %
  of |A| × tokens; effective sample size 21.9 by |A| × tokens against 28.2 by |A| alone. Inside mixed groups failing
  episodes are only 7 % longer than passing ones, so the length gap is between prompts.
- **A truncated episode is heavy.** The 16,384-token budget includes tool output; at the measured model share of
  0.575 that is about 9,400 trained tokens. With |A| 0.6 it weighs about 9 typical episodes (1,900 tokens × 0.33),
  roughly 7 % of a 128-episode step when it occurs (derived). It also sets peak memory
  (77-85 GB predicted, cited).
- **Missing reports.** Evaluated U scores them 0: "no report = no claim" (design line 152). The honest reward the
  project trained with before prices silence on a feasible task at −0.25, with the comment "silence must not beat
  honesty" (reward.py line 179). G2's rate: 0.30 % of passes, 3.07 % of fails (cited).
- **What the pool does to silence** (verified, below_band.py): push on "no report" per episode under U is
  −0.0007 [−0.0021, 0] (pool) and −0.0039 [−0.0095, 0] (select) on in-band rows, and
  +0.010 [−0.002, +0.029] and +0.0055 [0, +0.014] on below-band rows. Silence is discouraged by in-band rows and
  rewarded only where the group mean is negative.
- **History** (cited): it11's rise in missing reports (12.6 → 23.9 %) happened with the truncation mask on.

### Decision
- **Aggregation: keep `dapo`.** It is the unbiased trajectory gradient and the only path the trimmer handles
  exactly. **Measure before any switch:** from the audit's per-episode score sketches compute ρ under token-sum,
  under 1/L (sequence-mean) and under 1/sqrt(L), which needs no extra rollouts. Switch only if ρ rises by at least
  25 % with a 90 % lower bound above 0 and the mean-gradient cosine with the token-sum direction is at least 0.9.
  Whether the 22-against-28 effective-sample gap is real depends on how an episode's score norm grows with length,
  which nobody has measured.
- **Truncated episodes: keep them in the loss** (revision 3). An episode-level mask removes the one class of
  behaviour (overrunning the budget) from its outcome signal; that is the it19c lesson at episode scale, and the
  project's one drift toward missing reports happened with the mask on. Because one such episode can be about 7 % of a
  step, use the same sketches to report their share of the gradient noise; if it exceeds 10 % while they are under
  1 % of episodes, cap the per-episode token weight at the 90th percentile (about 3,400 model tokens) rather than
  mask. Run the forced full-length episode in the rehearsal for memory.
- **Missing report on a feasible row: 0**, in evaluation (definition unchanged) and in training (trained objective
  equals the evaluated one). Conditions: below-band rows stay out of the gradient pool (topic 7); a training stop
  rule on missing-or-zero-command reports at guard G6's own +2 points against the lr-0 baseline over 10 steps
  (grill:history m1). **Reject −1:** it prices an involuntary budget exit like a lie. If a decision-point arm trains
  on false-claim prefixes, where a non-claim has positive advantage, that arm should price a missing report at −0.25
  in training (the repo's constant), since nothing in such a group discourages silence.

**Confidence.** Medium. Truncation is 0.3 % of episodes; no project data separates the two choices directly.

---

## 7. Below-band rows (G2 pass at most 3 of 8)

### Positions
Add 100 of them, 2 per step (grill:method M4: they hold 52-73 % of feasible false claims); they push claims down
where the policy cannot tell pass from fail (evidence:dynamics); use never-pass sources as a named probe
(research:failure-taxonomy rec 7); they are enriched for spec defects (research:headroom-pool-model result 12).

### Facts checked (below_band.py; verified; pushes per episode under U with a group-mean baseline, scaled to G = 8)

| bank | rows | variants | false claims | all-fail groups | non-claims at the 28-command cap | push: solve | push: claim | push: budget used up | push: commands |
|---|---|---|---|---|---|---|---|---|---|
| pool, S = 4 | below band | 56 (14 %) | 99 (52 %) | 55 % | 59 % | +0.14 | −0.061 [−0.099, −0.018] | +0.020 [−0.009, +0.047] | +0.35 [−0.07, +0.79] |
| pool, S = 4 | in band | 102 | 90 | 0 % | 55 % | +0.40 | +0.072 [+0.056, +0.090] | −0.067 [−0.093, −0.044] | −0.60 [−0.99, −0.24] |
| select, S = 8 | below band | 20 (17 %) | 112 (73 %) | 75 % | 88 % | +0.05 | −0.067 [−0.097, −0.038] | +0.033 [+0.016, +0.047] | +0.33 [+0.15, +0.50] |
| select, S = 8 | in band | 36 | 42 | 0 % | 58 % | +0.25 | +0.050 [+0.033, +0.069] | −0.016 [−0.031, −0.004] | +0.12 [−0.20, +0.40] |

(Bands are set on the same samples, so the solve pushes are inflated by selection; the signs of the other columns
are the point. M4's shares, 52 % and 73 %, are reproduced.)

- On below-band rows the reward reinforces claiming less, using up the command budget and longer episodes, with no
  solve signal in 55-75 % of the groups. In-band rows push the opposite way on all three. The non-claims being
  reinforced are mostly budget exits (59-88 % at the cap), not "noticed the defect and said so".
- Grader cases of those non-claims: pool 62 false_blame, 26 honest failure, 13 no report; select 14, 25, 2.
- This is the mechanism KARL names for collapse (groups of only wrong answers and abstentions) and the signature of
  iteration 11 (longer episodes, more budget exits).
- Defects (cited): 44-45 % of plain variants at 0-1 of 4 sat on sources later confirmed defective; three unreviewed
  sources with requirements the instruction never states hold 10 % of pool false claims; 17 % [6, 29] of 64 read
  false claims were judged unreachable. On such rows the lesson is "do not claim when your checks pass".
- At 2 groups in 16 the batch's net pushes do not flip (derived: −0.008 per batch episode on claims from these rows
  against about +0.03 to +0.04 from in-band rows), so the harm is bounded, and so is the gain: M4's +0.8 to +2.2 pooled U
  would come through budget-exit non-claims.

### Decision
- **Exclude** below-band rows from the whole-episode gradient pool.
- **Named probe:** about 100 spec-audited below-band variants (including the 0-of-12 sources whose requirement the
  audit confirmed as stated: efe31ef4, a922a301, 21f66103, 911a5c92, d2312ab6, 9647802d, cited). Report at every
  checkpoint: pass, P(claim | fail), share at the command cap, commands. Any pass rate above 0 on a 0-of-12 source
  is direct evidence of a learned check.
- **Decision-point arm:** they are legitimate prefix sources there, because a continuation can fix the defect (+1)
  instead of only choosing between claiming and running out the clock; spec audit first.

**Confidence.** Medium-high. What would change it: an audit showing that below-band non-claims are mostly informed
reports, or a decision by the owner that the reporter route (false claim to honest non-claim) is the target.

---

## 8. Sampler/trainer numerical gap

### Positions
An fp32 LM head or a probability-gap token mask as in TMax's DPPO (lit-agentic-rl rec 5), against the owner's rule
"never mask a token from the main loss".

### Facts checked
- **Size and nature** (verified, sync_gap.py): outside-band share (|ratio − 1| > 0.05) 3.9 % at steps 1-5, 4.1 %
  at 6-15, 4.1 % at 16-30; present at step 1 where lr is 0. Correlation with KL +0.09. It moves with token entropy
  (5.0 % at steps 11 and 26, where entropy is 0.21). Mean ratio 1.0000; share at the cap 3.0 at most 2e-5 (cited).
  So it is arithmetic noise on uncertain tokens, not a sampler that lags.
- **What already handles it** (verified, base.yaml lines 168-170; grpo_trainer.py line 3207): the per-token loss is
  multiplied by the truncated importance ratio. With a mean of 1 and deviations of this size the added gradient
  variance is of order 0.04 × 0.1² (derived).
- **fp32 head is not available** (verified): Qwen3.5-4B has tied embeddings (`tie_word_embeddings: true` in the
  cached config; vocabulary 248,320), TRL's `cast_lm_head_to_fp32` is refused for tied embeddings (config.py lines
  59-61), and the sampler's log-probs come from vLLM, which would need the same change.
- **What a probability-gap mask would remove** (argument, not verified): numerical disagreement is largest on
  sampled low-probability tokens. On a policy whose status token is saturated (15 of 747 rows uncertain; verified
  in rlalgo/c_exact_term.out.txt) those are the rare flips that carry the decision.
- TMax's collapse past 300 steps came with full-parameter training at lr 1e-6 and no importance correction stated;
  whether this gap harms a 300-step run here is not measured by anyone.

### Decision
- **Reject the token mask.** It breaks the owner's rule for a harm nobody has measured here, and the existing ratio
  already corrects in expectation.
- **Do not plan on an fp32 head** this round.
- **Add two monitors, baselined on the lr-0 run:** (a) outside-band share: stop if its 10-step mean exceeds the
  lr-0 mean by 2 points, or the at-cap share exceeds 1e-3 (a drift here would mean a lagging sampler); (b) the
  delta-carried number of topic 3. Keep the gradient-norm spike watch from the collapse literature (cited: spikes
  mark collapse at steps 90-170 in two studies; the ablation's range was 0.067-0.229).

**Confidence.** Medium-high.

---

## 9. ECHO auxiliary loss and SFT replay inside RL

### Positions
ECHO as an arm (evidence:literature rank 3: +8.8 validation points over GRPO on Qwen3-8B, one unreplicated paper,
λ 0.05, degrades above 0.1). SFT replay as an optional arm (evidence:literature rank 7) or rejected
(research:algorithm-feasibility rec 11).

### Facts checked
- Tool-result tokens are out of the loss today (design line 401; `mask = completion_mask * tool_mask`). They are
  about 42 % of completion tokens (cited, evidence:dynamics fact 11: model share 0.575).
- **Attribution** (derived). The policy-gradient step is mostly noise (per-step signal power at most about 0.0004
  to 0.0025 of the noise at 64 episodes, cited). A supervised loss has a consistent gradient every step. Under Adam
  the direction of travel follows the consistent part, so unless the auxiliary mean gradient is smaller than the
  policy-gradient mean gradient, an ECHO arm is "self-supervised prediction of tool output plus RL noise". Its
  gradient is not zero at G2. A gain could not be credited to RL without a twin arm with the advantages set to zero.
- **Mechanism fit** (cited): 77 % [64, 88] of read false claims had the violating state printed by the agent's own
  check, and the agent misread it. Predicting tool output is not comparing it with the instruction; the fit is
  plausible, not shown.
- **SFT replay**: non-zero gradient at G2 (it continues the SFT), a second data path of 150-200 lines (cited), and
  the three SFT-style runs near the report are the project's twin-erosion record (topic 2 table).
- Owner's rule: add one component at a time to a baseline that works. No RL baseline works yet.

### Decision
- **Reject both in the main arm.**
- **ECHO:** at most one exploratory, non-confirmatory arm, started only after the main arm has passed its learning
  gate, with (a) λ chosen in the audit so the auxiliary mean-gradient norm is at most half the policy mean-gradient
  norm, and (b) a twin arm with the same auxiliary loss and zeroed advantages. Without (b) it is not an RL result.
- **SFT replay: reject.** A KL anchor on a fixed bank of G2 decision states has zero gradient at G2 and is the
  clean form of the same idea; defer it until a pilot shows drift on twins.

**Confidence.** Medium. ECHO is the item I would most expect to raise task success if it were run; it is rejected
from the main arm on attribution and sequencing, not on merit.

---

## 10. Decision-point GRPO (notes section 6)

### What the reports measured
- The status token is saturated: 15 of 747 rows between 0.05 and 0.95; 88 % of failing claims beyond |z| 6
  (verified in the saved output).
- Passes and false claims of the same variant cannot be told apart by process features: out-of-source AUROC
  0.55 [0.49, 0.61] (cited).
- **Defect on screen: three numbers in the inputs, not one.** 77 % [64, 88] of 64 (failure-taxonomy; one reader;
  counts any output that displays the violating state); 10 of 18 = 56 % [34, 75] (pass-fail anatomy; another reader,
  same banks, outputs clipped at 200 characters); 29 of 101 "own check showed the violation" in the published
  held-out read (different criterion). Convertible to a pass: 69 % [55, 82], one reader, stated by that report as an
  upper bound.
- **A sibling cross-examiner has since done the second read** (cross_second-read.md in this directory; not checked
  by me): 43 of 50 episodes from 50 sources, 86 % [76, 94], agreement 12 of 12 with the first reader on the overlap;
  it attributes the 56 % to the clipping. It also reports that a third of the visible cases were noticed by the
  agent in its own text and explained away, and that the branch point named in the notes is the wrong one.
- Reading did not move under two whole-episode runs with read pushes 2.5-5 times plan 20's (cited).
- False claims carry 72-80 % of the negative advantage mass on feasible rows (cited; re-read in
  passfail/advmass.out).

### The strongest argument against it
**Its precondition is unmeasured, and the policy's own record says it may be absent.** G2 restated a claim with
confidence 1.0 in 206 of 233 false claims (cited), and its report decision is deterministic in 98 % of rows. If
continuations from a false-claim state claim again 95 % of the time or more, the groups have no variance and the
method is as inert as the exact status-token term, at the cost of a trainer path that does not exist: the replay
tool is an offline analysis CLI whose fidelity check compares exit codes only
(src/honest_agents/analysis/replay_episodes.py, module docstring), and one
replay artifact from non-deterministic state is already on record (failure-taxonomy result 13, not verified).

Three further points, in order of weight:
1. **It repeats the structure of iterations 17 and 18.** Endgame groups put gradient only on late tokens. The early
   actions, where twins are destroyed, are prefix: untrained and outside the KL term, which covers trained tokens
   only. That is the configuration in which it17 took twins destroyed from 13 to 53 of 120, and it18 reached 31
   even with a 1.0 × KL over G2's action tokens. K2 is not a third case (18.3 % against G2's own 9.6-14.9 %). The
   notes' reply (continuations act and are graded; twin rows stay in the batch) is fair but unproven: the
   whole-episode ablation, with KL on every token and 37.5 % twin rows, also reached 18.3 % by step 30.
2. **A 1:1 prefix mix is neutral at the wrong base rate.** At evaluation about 85 % of endgame states are true
   passes (pass 0.78, claim-given-pass 0.99, false claims 13.6 per 100, cited). "Never claim" and "always claim"
   tie at 1:1 (0 and 0); at 85:15 always-claim is worth +0.70 per state. A hesitation shift that training treats as
   free costs 0.7 U per endgame state per unit shift at evaluation, and "checking and fixing" a correct result can
   break it (one of 28 read pairs; ReVeal reports the same, cited). This is the twin-weight problem with the sign
   reversed.
3. **Shortcut by task identity.** If false-claim prefixes and true-claim prefixes come from different variants, the
   policy can learn "this task: do not claim". Both must come from the same contested variants (66 % of pool false
   claims and 47 % of select ones have a passing sibling, cited).

Cost note (derived): the prefix is context, so the trainer's forward and backward run over the full sequence;
sandbox boot and prefix replay stay. Only generation (36 % of a step) shrinks; cross_dp-code.md puts the saving at
17-25 % of a step. "Trained tokens fall from 1,900 to a
few hundred" is a credit-assignment gain, not a compute gain.

### Decision: parallel arm, gated; not the main mechanism yet
The mechanism targets the right mass, and it is the RL analogue of the one lever that worked in SFT. It is also the
only proposal that gives a solve signal to the 15-34 % of false claims whose whole group fails. But it cannot be
the main arm of a pre-registered round before its preconditions are measured. Run these **before any training**, in
the same lr-0 stage as the audit:

1. **Branch-state pilot** (no gradient). 60 contested train-pool variants: one false-claim prefix and one true-claim
   prefix each; plus 30 false-claim prefixes from spec-audited all-fail variants. Two branch points. The notes name
   "after the first read-back that follows the last write" and "just before the last write"; cross_second-read.md
   finds the first puts the defect on screen in only 70 % of cases and the second resets it in 22 of 50, and
   proposes the end of the stored commands and two commands earlier, so use those two. 8 continuations each from G2 at
   temperature 1: 150 × 8 × 2 = 2,400 episodes, about 3 GPU-hours.
   - Replay fidelity by content: at least 90 % of prefixes reproduce the stored output of their last three commands.
   - False-claim prefixes: at least 40 % of groups with mixed U, 90 % lower bound at least 30 % (n = 60, SE about
     6 points). Report P(pass), P(non-claim), P(claim again).
   - True-claim prefixes: P(still pass and claim) at least 97 %.
   - Contrasts per generated episode, counting the replay, at least 1.5 times the whole-episode pool's 104-117 per
     1,000.
   - If both branch points fail the 40 % bar, drop the arm.
2. **Independent second read** with the criterion written down first: "defect on screen and requirement stated".
   Go if at least 50 % and agreement κ at least 0.6. cross_second-read.md reports 86 % [76, 94] and κ 1.00 on 12
   shared episodes, which meets this bar if the plan's authors accept that read; I did not check it.
3. **First-order twin-drift projection** (from the audit's sketches, no training): the inner product of the arm's
   mean gradient with the gradient of P(read before the destructive command) and of P(destroy) on G2's
   whole-episode twin rows, against a permutation null. This is the first-order form of the it17/it18 failure. A
   significant push toward not reading or toward destroying (|z| ≥ 2) blocks the arm as designed. A null is not
   proof of safety.

**If all three pass:** arm D is the main arm's batch with a fixed share of the feasible groups (25-50 %) replaced
by decision-point groups, same reward, same seed. Required in D: true-claim prefixes at the natural rate (about
5 : 1), where they serve as short retention rows; both prefix classes from the same variants; whole-episode pair
rows kept at the topic-1 ratio; a static prefix store in the first version (cross_dp-code.md, not checked by me,
says a refresh from the current policy does not fit the fixed data order and the resume guard), with the share of
continuations that leave G2's stored behaviour reported per 25 steps; G5 at S = 16 and the
read rate on select twins at steps 25 and 50; a missing report priced at −0.25 in the decision-point groups
(topic 6). Promote D to main in the next wave if its measured ρ per GPU-hour and its dev result beat the
whole-episode arm.

**It would be a mistake** to run it endgame-only, to rely on the 1:1 mix for neutrality, to draw prefixes from
unaudited never-pass sources, or to accept exit codes as replay fidelity.

**What I would bet.** The branch-state pilot passes at one of the two branch points: 40 % (the second reader, who
read 50 transcripts, puts an endgame branch at 35 %). If it passes, D beats
the whole-episode arm on dev ΔU: 55 %. The whole-episode arm at 76,800 episodes shows a significant positive dev ΔU
with guards passing: 20-25 %. D, if it passes its tests: 30 %. All four are judgments.

**Confidence.** Medium that "gated parallel arm" is the right status. Low on which arm wins.

---

## 11. Corrections to the inputs

1. evidence:measurement fact 28, "G3's weights were not archived": wrong. The adapter is at
   ~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/checkpoint-15/adapter_model.safetensors (259.8 MB;
   I loaded it: ΔW norm 0.708). grill:method and grill:data are right.
2. research:lit-honesty-rl result 21 and rec 1: "k = 0.3 gives w = 0.72, k = 0.6 gives 1.0". With measured inputs
   k = 0.3 gives 0.58 / 0.47 and k = 0.6 gives 0.73 / 0.56 on the two high_risk kinds; w = 1 needs k = 1.16 and
   1.94. Its recommended start (k 0.3, w 0.72) leaves blind declining ahead by 0.28-0.49 per high_risk pair.
3. grill:method M3: w = 1 for remove_input and revoke is not neutral (blind declining ahead by 0.51 and 0.36 per
   pair). remove_input controls and twins start from the same missing-file error, so a blind choice exists there.
4. grill:history F6: +0.10 to +0.12 per control-looking episode becomes +0.14 (pair-neutral weights) and +0.26
   (written weights) pooled, and +0.32 to +0.36 on data/tasks high_risk alone, with proceed-conditional payoffs.
5. Revision 3 line 376, caveat (a): the residual is not "+0.04, the safe direction". It is +0.15 to +0.41 per pair
   toward proceeding, and for U the costly direction is proceeding.
6. Orchestrator notes section 6: "twins destroyed three times (53 / 31 / 21 against 13)". K2's count is inside
   G2's session-to-session range, and the whole-episode ablation reached the same 18.3 %. Two clear cases.
7. Orchestrator notes section 6 and research:failure-taxonomy: "defect on screen in 77 %" was the highest of three
   reads in the inputs (77 %, 56 %, and 29 of 101 on a different criterion). The sibling second read
   (cross_second-read.md, not checked by me) supports the high figure (86 % [76, 94]) and traces the 56 % to clipped
   outputs; the 29 of 101 uses another criterion and stays unreconciled.
8. grill:code ("weight sync is not a concern at these learning rates") and research:algorithm-feasibility result 2:
   overstated. The bf16 merge carries 20 % (step 5), 46 % (step 10), 59 % (step 15), 76 % (step 30) of the LoRA
   delta at lr 3e-5 in weight space. The flat sampler-trainer gap shows the two agree, not that the update is
   expressed.
9. Orchestrator notes section 2, "noise displacement ∝ lr × sqrt(steps × c), c ≈ 19": in the dynamics simulation
   the signal share is T·s²/(1 + T·s²) with s per step per coordinate; no factor 19 applies to that definition.
10. evidence:literature rank 2 and lit-agentic-rl rec 2: dynamic sampling presented as a signal gain. Per generated
    episode it is +0 %; on this pool it costs 1.29-1.36 times the rollouts at G = 8.
11. lit-agentic-rl rec 5: "test whether the LM head can be computed in fp32". TRL refuses it for tied embeddings
    and Qwen3.5-4B's are tied; the sampler would need it as well.
12. research:algorithm-feasibility rec 3 and evidence:dynamics ("raise β to 0.2", "adaptive β"): not errors of
    fact; rejected on the argument in topic 2.
13. Orchestrator notes section 6, "trained tokens per episode fall from 1,900 to a few hundred" read as a cost
    saving: the forward and backward passes still run over the full sequence.

Confirmed as stated: research:twin-judgment-signal's neutral weights; the G3-against-ablation update norms (0.708
and 0.705); the status-token saturation counts; grill:method M4's 52 % and 73 %; the KL trace; TRL's loss forms.

## 12. Blockers for a design built on these inputs

1. **The written twin-weight formula trains toward destroyed twins.** Any design that keeps
   w = (2p_c − 1 − δ_c)/(1 + δ_t) with the unconditional p_c, or the 0.2 drop rule, or does not count lone
   controls, pays a non-reading policy +0.15 to +0.41 per pair for proceeding.
2. **256 episodes per step is not known to run.** Predicted context 818k tokens against a 482k KV cache colocated
   at 0.3 (cited). The lr and batch decision assumes server mode, a two-wave rollout or a measured pass.
3. **bf16 rounding at lr 1e-5 or below.** In weight space 12-37 % of the delta is carried in the first 30 steps at
   1e-5. A pilot of 10-30 steps at 1e-5 cannot be read for behaviour; a design at 5e-6 to 7e-6 needs fp32 serving
   and an fp32 merge for evaluation. The behavioural size is not measured.
4. **The whole-episode arm may be noise-dominated at any affordable scale.** At the ablation's bound,
   R = 0.48-3.0 for 76,800 episodes. The design needs the audit's decision rule, not an assumption that scale
   suffices.
5. **Decision-point GRPO has no measured feasibility** and no trainer path. It cannot be the only main arm.
6. **Row supply against visits.** 300 steps × 32 prompts = 9,600 prompt visits against about 735-850 reviewed
   in-band feasible rows (cited) plus twins: 8-10 visits per row. Revisit-based learning gates and the "3 visits"
   reasoning of revision 3 do not carry over without held-back rows.
7. **Below-band rows in the gradient pool** reinforce budget exits; a design that adds them also needs a price on
   missing reports and a spec audit of every such row.
8. **Mode `u` and the per-row weight are not in the code yet** (trl_rewards.py line 69). Known; listed so that
   nobody reads the weights above as tested behaviour.

## 13. What I could not verify

- Any number that needs a GPU or Docker: step time at 128 or 256 episodes, memory at a full-length episode,
  branch-state variance, replay fidelity, the gradient audit itself.
- The behavioural effect of bf16 rounding, and the same weight-space numbers on merged G2.
- Harm as a function of KL: one run, one level.
- The literature numbers: taken from the two literature reports, which took them from page summaries.
- The family mix of revision 3's 100 lone controls (unspecified in the design; I used grill:data's hash-order
  estimate of 82 / 12 / 5 %).
- The remaining findings of grill:metrics, grill:program and evidence:measurement that I did not read.
