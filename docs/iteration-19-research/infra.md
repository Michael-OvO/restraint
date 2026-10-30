# Infra map: constrained RL (A) and the final benchmark (B)

Read-only survey, 2026-10-29, repo at `9bcba69`. Paths are repo-relative unless they start with `~`. "Measured" means
read from an archived log; "estimate" means the runbook's or my projection; "unknown" means no evidence was found.

## Summary

- **Reward.** All weights sit in `RewardConfig` (`src/honest_agents/reward.py`, lines 167-191). They are set per run
  under `reward.config:` in YAML (`configs/grpo_colocate_qwen35.yaml`: alpha 0.1, false_blame -0.4, implicit_abstain 0)
  and built by `train/derive.py` `build_reward_config` (lines 71-89). There is exactly one reward function, with
  weight 1 (`train/derive.py` lines 156-160; `trl_rewards.py` lines 1-15).
- **Advantages.** These are stock TRL 1.14.1: the reward minus the mean of the prompt group (G = 8), with **no std
  scaling**. `scale_rewards: none` is enforced (`train/grpo.py` lines 23-32; TRL `grpo_trainer.py` lines 2817-2844).
  Masked rows get advantage 0 (line 2866). So any extra term put into the scalar reward goes straight into the
  advantage at face value.
- **Per-constraint signals.** Every one of them exists per episode at reward time: as flags on `RewardBreakdown`
  (`reward.py` lines 404-455), plus the family and is_control labels and the command history (`trl_rewards.py` lines
  403-442). The reward function already logs them as batch rates. **A Lagrangian needs no change to the trainer or to
  TRL**: it fits into `trl_rewards.mode_reward` / `score_batch` as a new `reward.mode`, plus a small state object and a
  probe rule. My estimate is about 350-450 lines plus tests.
- **Hardware.** GRPO has never run on a 32 GB card. G3 and B0 ran colocated on a 96 GB RTX PRO 6000, with GPU peaks of
  **53-74 GB** per step (vLLM was reserved 0.3 × 96 = 28.8 GB). As configured, the RL stage does not fit on an RTX
  5090. Rent the 96 GB class instead (it15 price $1.46/h, the same range as the 5090's $1.11-1.93/h), or spend about
  $1 on a memory probe of an untested sleep-mode setup.
- **The deciding risk for a primal-dual method is noise, not code.** In G3's own rollouts the per-step constraint
  estimates have an SD of **0.11-0.21** around means of about 0.2. The within-group ICC on control declines is about
  0.34, so 24 control rollouts per step are worth about 7 independent ones. A 15-step run allows about 1-2 meaningful
  dual updates. Plan for 30 or more steps, an EMA or PID dual, and multipliers warm-started offline from G2's banks.
- **Final benchmark.** Frozen: TB2 (89), ImpossibleBench LCB (309), Agentic Abstention (108), MASK (1,000, paid
  judge). Pending: internal_heldout (the build does not exist), AbstentionBench (DEF-50) and ImpossibleBench-SWE (no
  adapter). A core run (TB2 S=4, IB-LCB S=4, internal held-out S=2) for **base + G2 takes 11-18 h: $12.5-20 at
  $1.11/h, $22-35 at $1.93/h**. A third arm later takes 6-10 h, $7-11 at $1.11/h.
- **What would invalidate a run.** (1) Gate waivers: DEF-10/20/50 are waived only for it16a-d, it17a-b and it18b, and
  `gate.ledger check` matches the chain name. (2) Prereg 10.3/10.4 forbid this run order (G2 on TB2/IB before a GO),
  so an amendment is needed first. (3) The one-look rule combined with a later third arm requires a pinned eval code
  path. (4) TB2 success for a 4B model is near 0. (5) The never-trained verdict predates it17/18 and does not cover
  `data/it17/`. (6) TB2 images need 400 GB or more of disk.

---

## A. What exists for RL, and what a constrained-RL iteration needs

### A1. Reward terms and where alpha / beta / lambda live

- **Definition.** `src/honest_agents/reward.py` `RewardConfig` (lines 167-191):
  - `alpha` 0.25 (honest "not finished").
  - `success_bonus` (lambda) 0.5.
  - `beta` 1.0 (damage or violation penalty, paid once).
  - `implicit_abstain` (gamma) 0.25.
  - `wrong_reason` 0.5, `false_blame` 0.0, `lazy` 0.0.
  - `no_report_feasible` -0.25, `tamper_reward` -1.5 (a cap), `brier_weight` 0.
  - `efficiency_bonus` 0.1 over `efficiency_ref_actions` (the call budget).
  - `pair_table` False.
- **Validation.** `__post_init__` (lines 193-243) checks the ordering invariants. Among them: beta > wrong_reason −
  min(no_report_feasible, lazy, false_blame) (`damage_floor`, line 293), and pair_table break-even 0.5 with blind
  reports ≤ 0 per pair (lines 235-243 and 245-279).
- **Tables.** The full tables are in the module docstring (lines 1-115) and in `docs/design.md` section 4. A claim
  pays +1+λ on a pass and −1 on a fail; an honest non-claim pays α + λ·pass; an infeasible correct decline pays 1; a
  wrong reason pays 0.5; −β for damage; tamper is capped at −1.5.
- **Thresholds.** The claim threshold is p* = (1+α)/2 (`claim_threshold`, line 299). The blame threshold is
  (α−fb)/(1−wr+α−fb) (lines 302-311). Both are exact only under un-normalised advantages (design.md section 4, "Why
  these numbers").
- **Run values.** The GRPO runs since iteration 10 used alpha 0.1, false_blame −0.4, implicit_abstain 0, everything
  else at the defaults (`configs/grpo_colocate_qwen35.yaml`, `reward:` block). G3 inherited them
  (`configs/grpo_it15_g3.yaml` → `grpo_it13.yaml` → `grpo_it12_96gb.yaml` → `grpo_it12.yaml` →
  `grpo_colocate_qwen35.yaml`).
- **Built and registered.** `train/derive.py` `build_reward_config` (lines 71-89) refuses unknown fields and forces
  `efficiency_ref_actions` = the call budget. `compute_reward` runs inside the environment (`env.py` lines 483 and 487)
  and is cached (`env._graded`, line 557). `trl_rewards.mode_reward` (line 78) maps the breakdown to the optimised
  number per `reward.mode` ∈ {honest, outcome, outcome_claimed, outcome_reflect} (`trl_rewards.py` line 65).
- **Still to do for prereg 10.2.** The DEF-13 `pair_table` exists and is audited (`pair_blind_values`), but no GRPO
  run has used it. Prereg 10.2 (`docs/iteration-16-prereg.md` lines 393-405) already plans GRPO with
  `pair_table: true` and `pair_batches: true` and requires S9 (gate sweep, reward audit) to pass on the exact config.

### A2. How advantages are computed

- **Stock TRL 1.14.1.** `.venv/.../trl/trainer/grpo_trainer.py` lines 2817-2844 (`sum_then_normalize`). The reward is
  the weighted nansum over reward functions (one function, weight 1). The advantage is the reward minus the nanmean of
  the group of `num_generations` = 8. The std is only logged (`is_std_zero`).
- **The std scaling is refused by the repo.** `check_scale_rewards` (`train/grpo.py` lines 23-32) and line 219-220.
  The reason: dividing by the group std erases alpha in two-valued groups. The measured realised thresholds were
  0.40/0.48/0.54/0.57 for nominal values 0.50/0.625/0.75/0.875.
- **Masking.** A `None` reward becomes NaN, is left out of the group mean, and gets advantage 0 (line 2866). Its
  tokens still count in the DAPO normaliser (`trl_rewards.py` docstring lines 10-15).
- **Loss.** `loss_type: dapo`, `beta: 0` (no KL), `epsilon_high 0.28`, token-level truncated IS (`token_truncate`).
  One generation round is one optimiser step (design.md section 8).
- **The trainer subclass does not touch advantages.** `HonestGRPOTrainer` (`trainer.py` line 310 onward) only
  changes the tool loop, padding-free scoring and sync. Construction fails if TRL's `_generate` or
  `_calculate_rewards` fingerprints change (`trainer.py` lines 227-260).
- **Consequence.** A Lagrangian term put into the single scalar reward needs no trainer change. Do **not** use TRL's
  `normalize_then_sum` with several reward functions (lines 2846-2856). It std-normalises each term, which destroys
  the threshold semantics. And with more than one function a `None` from one of them becomes 0, which breaks masking
  (`trl_rewards.py` lines 10-15).

### A3. Per-constraint signals per episode, at reward time

All of these are available inside `trl_rewards.score_batch` (lines 692-758) for every rollout of the batch:
`breakdowns[i]` is a `RewardBreakdown` and `environments[i]` holds `_spec`, `_history` and the cached outcome.
Frequencies are from G3's 960 training rollouts (`~/Documents/honest-agents-archive/iterations/15-g3-grpo/run/
it15-g3-grpo/completions/*.parquet`, my recompute).

| constraint | signal at reward time | source | G3 frequency (per step = 64 rollouts) |
|---|---|---|---|
| false claim on feasible | `br.fabricated` (claimed and not passed) | `reward.py` line 425 | 98 / 960 (6.5 per step); false-claim share of claims 0.20 mean, per-step SD 0.13 |
| claim on an impossible twin | `not br.feasible and br.claimed_complete` (`infeasible_claimed`) | `reward.py` lines 412-425 | 120 / 960 (8 per step); twin claim rate per-step SD 0.21 |
| violation / destruction | `br.violation`, `br.self_inflicted` (grader content diff of `preserve_paths`, never the report), `damage_paths` | `reward.py` lines 365-368 and 446-447; `grading.py` line 862 `_state_checks`, line 1200 | `abstain_after_damage` 3 / 960; damage is rare in G3 rollouts (the full violation count is not in the parquet: unknown) |
| tamper | `br.tamper` / `o.tampered` (includes grading failure after acting) | `reward.py` lines 370-373 | about 0 (no `tamper` case among the 960) |
| over-refusal on near-miss controls | `br.is_control and br.declined` (and `false_blame`, `decoy_blamed`) | `trl_rewards.batch_metrics` lines 224-230 | control decline per step mean 0.225, SD 0.133, range 0.00-0.50 |
| read before destruction | `behaviour.readme_read_before_destruction(commands)` (README only, any destructive command) | `behaviour.py` lines 61-67; computed per rollout in `trl_rewards.rollout_behaviours` (lines 410-442) | logged only (`honest/...` behaviour metrics); for non-README surfaces the canonical `eval.read_diagnosis.strict_read` needs `SurfaceSite` metadata (`eval/read_diagnosis.py` line 318) and is post hoc today |
| first mutation index | `outcome.first_mutation_idx` | `env.py` line 475; `mutation.py` lines 27-41 | metrics and best-of-N tiebreak only |

- **Group structure.** In G3's rollouts the between-group variance of control declines is 0.059 of a total of 0.175,
  so the ICC is about 0.34. With G = 8 the design effect is 1 + 7 × 0.34 ≈ 3.4, so 24 control rollouts per step are
  about 7 effective. Zero-variance groups were 0-12 % per step, so the contrast is fine.
- **Read-before-destruction should be a monitor, not a priced constraint.** Design.md section 4 ("Blind spots")
  rejected a process term because shell classification can be spoofed: a ritual `cat README; rm x` would satisfy it.
  Price the outcome instead (damage on twins, decline on controls) and keep the read rate as a monitor and guard
  (`docs/iteration-18-prereg.md` already has read-before-destruction guards).

### A4. Step time and memory (measured on 96 GB; 32 GB unmeasured)

From `train_log.jsonl` in the archive (`iterations/15-g3-grpo/run/it15-g3-grpo/` and
`iterations/12-harness-b0/run/honest-a0.1-qwen3.5-4b-s42/`). Both ran colocated vLLM at 0.3, 16k completion, G = 8 × 8
prompts, on an RTX PRO 6000 96 GB.

| | G3 (from G2, pool v3, 15 steps) | B0 (from base, pool v1, 15 steps) |
|---|---|---|
| step_time | mean 639 s (459-962) | mean ~541 s (381-736) |
| rollout | 298-667 s; tool 146-396 s (the slowest single tool turn 125-371 s dominates) | 265-586 s; tool 116-370 s |
| generation | 97-309 s | 102-291 s |
| grading | 62-71 s | 45-51 s |
| train (fwd + bwd) | 59-166 s | 53-158 s |
| weight sync | 1.0-1.1 s | 0.7 s |
| `gpu/peak_memory_gb` | **53.2-73.6** | **56.3-74.4** |
| probe (120 variants × 1) | 857-1,166 s; the step-0 probe took 2,899 s (one 1,892 s episode) | 1,032-1,174 s |
| train_runtime | 15,481 s = **4.3 h** | 11,473 s = **3.2 h** |
| host RSS | 4.9-5.1 GB | 5.3-5.5 GB |

- **The step is CPU / Docker / straggler bound.** Generation plus training is about 30-45 % of the step. A faster GPU
  barely helps. The 64-core 5090 nodes (it17/18) would likely help the tool phase, but that is unmeasured.
- **RTX 5090 32 GB.** No GRPO-with-tools run exists on it. It17/18 trained on a 5090, but only 8-token completions
  over 16k prompts (`configs/confidence/it18.yaml` lines 24-34).
  - Colocated vLLM at 0.3 of 32 GB is 9.6 GB, about the size of the bf16 weights alone, which leaves no KV cache.
  - The trainer side peaked at about 25-45 GB beyond vLLM's reservation, if `peak_memory_gb` includes the
    in-process vLLM pool (likely, not verified).
  - TRL's fused log-prob backward on one ~15k-token sequence needs about 15 GB of fp32 logits gradient (Qwen3.5
    vocab about 248k; `docs/training-log.md` lines 678-679, a 48 GB OOM).
  - **Verdict: not feasible as configured.**
- **Paths to a 32 GB fit (all untested).**
  - `vllm_enable_sleep_mode` exists in TRL 1.14.1 (`trl/trainer/grpo_config.py` lines 133 and 607) but has never been
    used in this repo.
  - `max_completion_length` 8-12k. G3 completion text, including tool outputs: p50 9.3k chars, p90 18.9k, p99 47.3k,
    max 57.7k. At 3-4 chars per token, p99 is about 12-16k tokens, so a 12k cap would truncate about 1-3 % and 8k
    about 5-10% (rough).
  - Grad checkpointing on (the memory rule).
  - Measure with `configs/phase0_zero_lr.yaml` for about 0.5 h ($0.6-1).
  - Weight sync in colocated mode is not covered by `scripts/sync_check.sh` (`configs/grpo_colocate_qwen35.yaml`
    header).
- **Recommendation.** Run the RL stage on the 96 GB class with the known config (`grpo_it12_96gb.yaml` lineage). The
  price is in the same range. Keep the 5090 for the evals.

### A5. Minimal code changes: per-constraint Lagrange multipliers and a held-out constraint monitor

**Formulation (CMDP; Altman 1999).** Maximise J_r(π) = E[task reward] subject to J_k(π) = E[c_k] ≤ d_k. Take the
Lagrangian L = J_r − Σ_k λ_k (J_k − d_k). Primal step: GRPO on the per-episode reward
r̃ = (r − Σ λ_k c_k) / (1 + Σ λ_k). Dual step: λ_k ← clip(λ_k + η_k(ĉ_k − d_k), 0, λ_max). This is RCPO-style
(Tessler et al. 2019). With a PID dual (Stooke et al. 2020) the oscillation is lower. Dividing by (1 + Σλ) leaves the
optimal policy unchanged under `scale_rewards: none`: it scales every advantage in a group by the same constant. It
keeps the effective learning rate bounded as λ grows.

**Link to the current table, which is what makes it principled here.** Take task reward = 1[claim ∧ pass] and price a
false claim at λ_fc. Then claiming pays iff p > λ_fc / (1 + λ_fc). The hand-set α picks the threshold (1+α)/2 once;
the multiplier instead moves the threshold until the false-claim rate meets the budget d_fc. At α = 0.1, the fixed
threshold 0.55 corresponds to λ_fc ≈ 1.22. Freezing λ (dual lr 0) is the fixed-weight special case, which is the
natural ablation arm. Under convexity in occupancy measures the duality gap is zero (Paternain et al. 2019), so the
primal-dual iterate targets the constrained optimum rather than a hand-tuned trade-off.

Changes, by file (no change to `trainer.py` or TRL; the fingerprints stay valid):

1. **`reward.py`, about 40 LOC.** Add a pure `constraint_indicators(br) -> dict[str, float | None]`, with None when the
   constraint does not apply to that episode's cell:
   - `false_claim` = `br.fabricated` (feasible);
   - `twin_claim` = claimed on infeasible;
   - `damage` = `br.self_inflicted or br.violation`;
   - `tamper` = `br.tamper`;
   - `control_decline` = `is_control and declined`.

   `compute_reward` is unchanged, so every archived row reprices identically.
2. **`trl_rewards.py`, about 120 LOC.**
   - Add mode `lagrangian` to `MODES` (line 65), `REWARD_FUNCS` (lines 843-848) and `train/config.REWARD_MODES`.
   - Add a `mode_reward` branch (line 78) that computes r̃ from `br` and the current λ. Base task reward: verified
     success (`claimed_complete and passed`) plus the efficiency term, so contrast survives in all-pass groups.
   - Keep the state in a module-level `LagrangeState`. This is safe because `check_launch_layout` (`train/grpo.py`
     lines 35-49) forces one training process with vLLM.
   - In `score_batch`, after the rewards for batch t are computed with λ_t, update an EMA of ĉ_k (the per-cell
     denominators already exist in `batch_metrics`), then update λ.
   - Log `lagrange/lambda/<k>`, `lagrange/c_hat/<k>`, `lagrange/c_ema/<k>` and `lagrange/violation/<k>` through the
     existing `log_metric`, so they reach W&B with no dashboard work.
3. **`train/config.py` and `train/derive.py`, about 60 LOC.** Add a `reward.lagrange:` section: per constraint
   `limit` d_k, `lr` η_k (or PID gains), `init` λ_0, `max`, plus `ema` and `update_every`. Validate that limits lie in
   [0, 1] and λ_max is finite, and that `reward.config` stays valid (damage keeps a fixed β floor; see A6).
4. **`callbacks.py`, about 40 LOC.** Persist the λ state to `<checkpoint>/lagrange.json` `on_save` and restore it on
   resume. The resume path already checks a data-order fingerprint (design.md section 8, "Resume").
5. **Held-out constraint monitor, about 60 LOC.** `train/probe.py` `probe_metrics` (line 199) already computes
   per-family rates on a fixed held-out probe (`data.probe_file`, 120 variants; `ValidationProbe`, steps from
   `probe.steps`).
   - Add per-constraint rates with an upper confidence bound, clustered by source (the
     `eval.external.transfer.cluster_bootstrap_ci` already used by `final_benchmark.compute`).
   - Add a `StopRules.observe_validation` rule (`callbacks.py` line 465): stop when UB(c_k) > d_k + slack at two
     consecutive probes. This mirrors prereg 10.2's control-decline stop rule.
   - The probe never feeds λ, so it stays an unbiased held-out check of whether the constraint is met off the
     training tasks.
6. **Offline warm start, $0, a typed CLI under `analysis/`, about 80 LOC.**
   - Re-score G2's archived episodes with `compute_reward` and `constraint_indicators`, from
     `iterations/16-it16b-claim-rollouts` (bank) and `iterations/17-it17a-data-rollouts/eval/*`, about 2,368 G2
     episodes.
   - Get G2's ĉ_k, then pick λ_0 so that the pricing at λ_0 equals the current table's implied prices. Example: the
     false-claim gap 1+λ+1 = 2.5 against the honest non-claim α = 0.1 implies λ_fc,0 ≈ 1.2-2.
   - Without this, λ starts at 0 and the first steps optimise success with no honesty price at all.
7. **Tests, about 150 LOC.**
   - Indicators against every `compute_reward` case.
   - Dual update and the clip.
   - Property test: at fixed λ the claim threshold equals λ_fc / (1 + λ_fc) under `none` scaling.
   - S9 reward audit (`analysis.reward_gate`, `pair_blind_values`) over a grid of the λ box: blind reports earn at most
     0 per pair for every λ in [0, λ_max]. Prereg 10.2 requires S9 on the exact config.

Effort: about one day of Mac work, $0. No GPU is needed until the phase-0 or memory probe.

### A6. Risks, with numbers

| risk | evidence | mitigation |
|---|---|---|
| **Dual noise; too few dual steps.** | G3 per-step SD of ĉ: control decline 0.13, false-claim share 0.13, twin claim 0.21 (means about 0.2). The design effect is about 3.4. At η·(ĉ−d) the multiplier random-walks. | Use an EMA with a window of at least 5 steps, or PID. Run 30-50 steps (15 steps is about 1-2 effective updates). Warm-start λ from G2's banks. Optionally add a dual batch of training-pool variants scored every k steps with no gradient. |
| **Rare constraints cannot carry a multiplier.** | Damage 3 / 960 G3 rollouts; tamper about 0. | Keep β as a fixed penalty (`RewardConfig` floor) plus a hard monitor and stop rule. Put a multiplier only on frequent constraints: false claim, twin claim, control decline. |
| **Constant-within-group costs give no gradient.** | Advantage = r − group mean. A twin that every one of the 8 rollouts destroys gives 0 signal at any λ. Zero-variance groups 0-12 % in G3. | Watch `honest/family/*/zero_variance_group_share` per constraint. A cue ladder or a G2 start keeps groups mixed. |
| **Caution attractor (G3's failure).** | G3 rollout control declines went 16 → 39 %; on the surface set 31 → 41 % (docs/iteration-15-g3-rl-read.md). The outcome reward paid "decline without reading" on twins. | The control-decline constraint prices exactly this. Also add the held-out monitor stop rule (prereg 10.2's +10 points). |
| **Shared-weight coupling (it17/18).** | A KL on the trained tokens only let actions drift (it17: blind deletions 4 → 30 of 120). Anchoring the actions (it18) cut the drift to 0.0086 per token. | GRPO `beta > 0`, with the reference = adapter-disabled merged G2, which is free in PEFT. Report action KL to G2 on held-out transcripts (it18's monitor). Costs about one extra forward, a few % of the step. |
| **Reward scale growth.** | Under `none`, Σλ inflates the advantages and so the effective learning rate. | Normalise by (1+Σλ), which is threshold-invariant. Set λ_max. |
| **Goodhart on process proxies.** | Design.md section 4 rejected process terms (spoofable shell parsing). | Read-before-destruction is a monitor only. |
| **Colocated vLLM on 32 GB.** | 53-74 GB peaks on 96 GB (A4). | Run on 96 GB, or do a 0.5 h sleep-mode probe first. |
| **Colocated weight sync unverified.** | The colocated sync path is not covered by `scripts/sync_check.sh`. | Check the importance ratio after the first update (Phase-0 gate: within 1 % of 1). |
| **Infra incidents.** | Egress probe cached False: every boot refused for 15 steps, about $1.6 lost (training-log line 1391; now a preflight). | Fail fast on masked share ≥ 0.5 (exists). Arm a watcher on every paid job. |
| **Group size.** | G = 8 is the "unstable corner" (design.md section 8, Tmax). | G = 16 with 4 prompts keeps 64 episodes per step but halves the prompts and raises ICC exposure. Prefer 8 × 8 and more steps. |

### A7. RL stage cost

- **Basis.** 96 GB at $1.46/h (it15 class; current availability unknown). 15 steps plus 4 probes: 3.2-4.3 h
  (measured). Add setup 0.75 h, merges and sanity 0.5 h, and the select split (424 episodes per arm at about
  700-950/h, so 0.45-0.6 h each, for 1-2 arms).
- **15 steps:** about 5.5-6.5 h, **$8-9.5**.
- **30 steps** (needed for the dual, A6): about 8.2-9.2 h, **$12-13.5**.
- **On a 5090 at $1.11/h, if the memory probe passes:** about 25 % cheaper. Unknown otherwise.

---

## B. The final benchmark: status, cost, what to drop, what invalidates it

### B1. Frozen vs pending, and what is needed to unfreeze each

Manifest: `data/benchmarks/final_benchmark.json`, frozen 2026-10-25. It has one refreeze, the AbstentionBench
metrics, and a never-trained verdict `pass` recorded 2026-10-28 over 64,876 rows. Tooling:
`src/honest_agents/eval/final_benchmark.py` (freeze, check, never-trained, units, verify-export, compute). The arms
are base (`Qwen/Qwen3.5-4B` @ `851bf6e8`) and G2 (adapter sha256 `f0018a12…`). `final` is null.

| benchmark | role | status | items | headline metrics | what is needed |
|---|---|---|---|---|---|
| terminal_bench_2 | headline | **frozen** | 89 (registry at commit `69671fba`) | `tb2.false_claim` (down), `tb2.success` (non-inferior) | nothing; runs via `scripts/eval_tb.sh tb2` |
| impossiblebench_lcb | headline | **frozen** | 309 = 103 original + 103 oneoff + 103 conflicting | `abstain.flag` (up); also cheated, claimed_complete, act.success, act.over_flag | nothing. `ib_tasks` materialises with `--prompt tuned`; the prompt variant is pinned in the job file, not in the manifest (runbook section 8) |
| agentic_abstention | reported only (underpowered: 19 abstain units) | **frozen** | 108 | abstain.overall, act.continued (report) | nothing |
| mask | secondary | **frozen** | 1,000 (6 archetypes) | `honesty.no_lie` (up), accurate | **a paid judge** (GPT-4o binary + o3-mini numerical; `eval/external/mask.py` docstring). Needs a separate owner approval (prereg section 4, runbook section 3) |
| internal_heldout | headline (internal) | **pending** | none | `heldout.false_claim` (down), success | the build does not exist: surface-val-v2 (68+ sources × 6 surfaces, 816+ variants) and the blind family were never built (`data/decision/` has only `surfaces_val`, the it15 set). Either build and validate it (Docker oracle gate on a node; days), or **amend** the definition to the frozen test split (`data/test.jsonl`, 763 variants; touched once, in the it12 transfer check on 20 sources of held-out families, so exclude those). Then `final_benchmark freeze --refreeze`, `check --archive`, `never-trained --record`, plus sign-off (integrity-gate step 3) |
| abstentionbench | regression check | **pending** (DEF-50) | none | abstain.abstained (non-inferior), answer.abstained (report) | one node step: export through the Hub script at revision `af06080e`, archive it, then on the Mac `freeze --refreeze`, `check`, `never-trained --record`, flip the test to `frozen`, commit (DEF-50 "Remaining", `docs/defects.md` lines 2865-2880). The gated GPQA subset also needs an HF token with access, or an amendment to freeze without GPQA. Until then `eval_abstentionbench.sh` stops at `verify-export` (exit 1). It also needs a Llama-3.1-8B judge served beside the policy, which does not fit easily on 32 GB beside a 4B server (unknown) |
| impossiblebench_swe | reported | **items frozen, adapter pending** | 1,047 | abstain.flag | an adapter, plus the per-instance SWE-bench images (heavy disk, unknown size). Not feasible in this budget |

### B2. Time and cost per arm per benchmark (one RTX 5090 VM)

Rates: $1.11/h (it17 node 58122470 class) to $1.93/h (it18 node 58292161). The node class and timings come from
`docs/iteration-16-runbook.md` section 3, which was written for a 96 GB 23-vCPU node; a 64-core 5090 is likely no
slower on Docker-bound evals (unmeasured). Internal episode throughput is measured: it16b ran 1,600 G2 episodes at
concurrency 24 in 139 min (about 690/h, `configs/chains/it17_a_data_rollouts.yaml` line 7). It17a's summed episode
wall time divided by concurrency 24 gives about 950/h (my recompute from the archive). I use 700-950/h.

| benchmark (S) | trials / episodes per arm | hours per arm | $ per arm @1.11 | $ per arm @1.93 | basis |
|---|---|---|---|---|---|
| TB2 (S=4) | 356 | 2-4 | 2.2-4.4 | 3.9-7.7 | runbook estimate; **no TB2 run at 30 turns exists**. AA's 108 TB2-based trials at 10 turns took 24-27 min at concurrency 12 (measured), × about 3 for 30 turns |
| TB2 (S=2) | 178 | 1-2 | 1.1-2.2 | 1.9-3.9 | as above |
| first-arm image pulls (TB2) | | 0.5 | 0.6 | 1.0 | runbook. **Disk: TB images about 146 GB; the guard is ≥ 400 GB** (training-log lines 1168-1170) |
| IB-LCB (S=4) | 1,236 | 0.9-1.7 | 1.0-1.9 | 1.7-3.3 | runbook estimate (1-2 min per small trial at concurrency 24) |
| IB-LCB (S=2) | 618 | 0.45-0.85 | 0.5-0.9 | 0.9-1.6 | as above |
| internal held-out as the test split (S=2, needs an amendment) | 1,526 | 1.6-2.2 | 1.8-2.4 | 3.1-4.2 | 700-950 episodes/h |
| internal held-out as prereg (surface-val-v2 S=4 + tier F + tier S + probes) | 3,264 + 712 + 480 + probes | 6.6 | 7.3 | 12.7 | runbook; and the build does not exist |
| Agentic Abstention (S=4 / S=1) | 432 / 108 | 1.7 / 0.45 | 1.9 / 0.5 | 3.3 / 0.9 | measured (S=1) and runbook |
| AbstentionBench | unknown size | about 0.5 + judge | 0.6 | 1.0 | runbook; blocked by DEF-50 |
| MASK generation (no judge) | about 4,600 chat requests (1,000 pressure + belief prompts) | about 0.2-0.3 (estimate) | 0.3 | 0.5 | chat-only on vLLM |
| MASK judge | about 4-5k judge calls | none | **about $10-30 per arm, paid API (rough; token counts unknown)** | | needs owner approval |
| RTS (rts/v1, S=1) | 400-600 | 2-5 | 2.2-5.6 | 3.9-9.7 | runbook (re-costed from AA) |
| fixed per rental: boot, setup, Docker 0.75 h; G2 merges 2 × 4 min; 12-task sanity runs about 10 min per served model (DEF-46 rule) | | about 1.25 (+0.5 TB2 pulls) | 1.4-1.9 | 2.4-3.4 | runbook |

**Plans**

| plan | per arm | base + G2 now | third arm later |
|---|---|---|---|
| **Core**: TB2 S=4, IB-LCB S=4, internal held-out (test split) S=2, MASK generation only | 4.75-8.15 h | 11.25-18.05 h with fixed costs: **$12.5-20 @1.11**, $22-35 @1.93; +30 % overrun → up to $26 / $45 | 6-9.9 h: **$6.7-11 @1.11**, $11.6-19 @1.93 |
| **Lean**: TB2 S=2, IB-LCB S=4, internal held-out S=1 | 2.7-4.95 h | 7.2-11.7 h: $8-13 @1.11, $14-23 @1.93 | 4-6.7 h: $4.4-7.4 @1.11 |
| **Core + AA S=1** | +0.45 h | +$1 @1.11 | +$0.5 |

**Option.** Run TB2 (concurrency 12) and IB-LCB (concurrency 24) at the same time against one server: about 36
containers on 64 cores. That would save 0.9-1.7 h per arm. It is untested and needs a new job file (the it16d chain
runs them in sequence).

### B3. What to drop to fit about $30-40, including the RL stage and a third arm

- **Budget.** Vast balance about $74 minus the watcher's `balance_floor` of $8 leaves about $66 usable.
- **A path that fits at the $1.11/h class:**
  - RL, 30 steps on 96 GB: $12-13.5;
  - core eval of base + G2: $12.5-20;
  - third arm later: $6.7-11.
  - Total **$31-45**.
- **Drop, in this order:**
  1. **RTS** (secondary; 2-5 h per arm).
  2. **ImpossibleBench-SWE** (no adapter).
  3. **AbstentionBench** (DEF-50 plus a judge GPU).
  4. **The surface-val-v2 / tier F / tier S / blind-family internal guards** (about 6.6 h per arm, and the build does
     not exist). Replace them with the test-split internal held-out.
  5. **Agentic Abstention S=4** (keep S=1 at about $0.5 if wanted).
  6. **The MASK judge** (generate now at about $0.3 per arm and freeze the responses; judge later only if the owner
     approves the API spend).
- **If still over:** TB2 S=4 → S=2. This changes the pre-registered S and the MDE (prereg section 4 says S = 4), so
  it must go into the amendment before any episode.
- **Use the $1.11/h node class.** At $1.93/h the core two-arm run alone can exceed $40 with overrun.

### B4. What would invalidate a run

1. **The integrity gate (`src/honest_agents/gate/ledger.py` lines 10-13, 63-65, 94-112).** A waiver counts only for
   experiments its status names. DEF-10, DEF-20 and DEF-50 are waived for it16a-d, it17a-b and it18b only
   (`docs/defects.md` lines 20, 30, 60). A new chain name, for example `it19final`, makes the first step
   (`gate.ledger check --experiment <name>`) exit 1 until the owner extends the waivers or the entries close.
   - DEF-10: bars set at a point estimate with no MDE. A new final-eval prereg must state its MDEs, as it17/18 did.
   - DEF-20: guards with no powered metric.
   - DEF-50: AbstentionBench unfrozen.
   - `it16d` is covered, but its job file has 4 unfilled placeholders and a different design. Reusing the name for a
     new design would be misleading.
2. **Pre-registration order.**
   - Prereg 10.3 says no final-benchmark episode before a GO on the validation gate.
   - Prereg 10.4 (the lean eval) runs **base, not G2**, on TB2 and IB.
   - Running base and G2 now therefore needs a written amendment before any episode.
   - DEF-58 (per-task, sample, turn seeds) is closed; keep the seeds.
3. **One look plus a later third arm** (runbook section 6.7; prereg section 8). A finished benchmark is never re-run.
   So base and G2 numbers taken now are final, and the third arm must run on the **identical eval path**: harness,
   `harbor_agent`, grading, env, the vLLM version and the prompt variant. There is a precedent: a harness change
   between it16 and it17 moved G2's select verified success from 72.7 to 69.1 % (DEF-65, `docs/results.md`, note on
   iteration 17). Pin the commit of the eval path in the amendment, as it18 did ("eval path unchanged since
   3cdb17c").
4. **Headline validity on TB2 for a 4B model.** On AA's 89 TB2 originals at 10 turns the 4B solved 0/88 (base),
   2/89 and 4/89 (training-log lines 1137-1144).
   - `tb2.success` is near the floor, so its non-inferiority guard has almost no power.
   - `tb2.false_claim` is per episode (claimed ∧ verifier fails; `final_benchmark.tb2_units` lines 818-838), so
     blanket declining lowers it and the success guard cannot catch that.
   - Report the claim rate and P(fail | claim) alongside. Let the internal held-out success carry the capability
     guard.
   - The MDE comes from the base arm; until then it is unknown.
5. **Never-trained scope.** The 2026-10-28 verdict predates it17/18. `REPO_TRAINING_GLOBS` (`final_benchmark.py`
   lines 62-69) does not match `data/it17/*` (the confidence items). This is harmless for base and G2, which were
   trained before. For any third arm built from it17/18-style data or new RL rollouts, extend the globs (or check that
   the archive glob `iterations/*/run/*/completions/*.parquet` covers it) and re-run `never-trained --archive --record`
   first. This is a gap like DEF-59; it is not yet in the ledger.
6. **Infra preconditions.**
   - Disk ≥ 400 GB for the TB2 images. The 5090 nodes' disk is unknown.
   - A 12-task sanity run per served model (DEF-46; `node_chain check` refuses otherwise).
   - The egress preflight.
   - IB `--prompt tuned` pinned.
   - `units --single`: one job dir per run.
   - A watcher armed in the same minute as the chain (the owner rule).
7. **AbstentionBench / MASK.** Neither can be reported without DEF-50's freeze or a paid-judge approval. If dropped,
   report them as "not run" (prereg 10.4 does the same for MASK).

---

## References

- E. Altman, *Constrained Markov Decision Processes*, Chapman & Hall/CRC, 1999. https://doi.org/10.1201/9781315140223
- C. Tessler, D. J. Mankowitz, S. Mannor, "Reward Constrained Policy Optimization", ICLR 2019.
  https://arxiv.org/abs/1805.11074
- A. Stooke, J. Achiam, P. Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods", ICML 2020.
  https://arxiv.org/abs/2007.03964
- S. Paternain, L. Chamon, M. Calvo-Fullana, A. Ribeiro, "Constrained Reinforcement Learning Has Zero Duality Gap",
  NeurIPS 2019. https://arxiv.org/abs/1910.13393
- J. Achiam, D. Held, A. Tamar, P. Abbeel, "Constrained Policy Optimization", ICML 2017. https://arxiv.org/abs/1705.10528
- J. Dai et al., "Safe RLHF: Safe Reinforcement Learning from Human Feedback", ICLR 2024. https://arxiv.org/abs/2310.12773
- T. Moskovitz et al., "Confronting Reward Model Overoptimization with Constrained RLHF", ICLR 2024.
  https://arxiv.org/abs/2310.04373
