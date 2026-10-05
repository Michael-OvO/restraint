# Decision memo: is the reward good enough, and what to fix before GPU spend

**Bottom line.** Keep the current reward, with one deletion and some tightened checks. Don't redesign it before the first run. The reward arithmetic is correct and well tested. The larger risks are in the evaluation plan, the grader-failure path, the data labels and trainer efficiency, and none of those is a reward problem.

Terms used below:
- **alpha** is the reward for a correct non-claim on a feasible task (default 0.25). It is the one real tuning knob.
- **lambda** is the work bonus for a passing episode (default 0.5).
- **w** is the credit for an honest decline on an infeasible task (default 0.5).
- **m** is 1 if the named blocker target matches exactly, else 0.
- **p** is the model's own probability that its work passes.
- **NI** means non-inferiority testing: showing the trained model loses no more than a margin on success.
- **Sceptic** means the second reviewer who re-checked each auditor finding.

## 1. Is the reward good enough?

**Current table (feasible tasks):**
- Claim done and pass: 1 + lambda.
- Claim done and fail: -1.
- Don't claim: alpha + lambda·pass.
- Silent, no report: below alpha.

**Infeasible tasks:**
- Honest decline: w + (1−w)·m.
- Silent: 0.25.

**Violations and tampering:**
- Tamper: -1.5.
- Violation: -beta.
- The efficiency bonus is at most 0.1.

Claiming is worth p(2+lambda) − 1 and not claiming is worth alpha + lambda·p. These cross at p = (1+alpha)/2 = 0.625 by default.

**Are the parameters arbitrary?** Mostly no. This is the best-supported finding of the audit.
- An exact optimal-policy solver and a GRPO simulation (group-relative RL, G=8) both put the claim threshold at (1+alpha)/2. That holds for every setting of lambda, beta, the tamper reward and the efficiency bonus.
- Moving every constant except alpha by ±50% did not change the learned policy. Learned thresholds overshot by only 0.02–0.07.
- Only alpha and the blame threshold are real choices. The blame threshold is set by wrong_reason and false_blame together, which act as one parameter. The other constants only need to hold an ordering, such as a tampered outcome never beating its clean twin. That held for all beta ≥ 0 over a 4×6 grid.
- Anchors regenerate the shipped values closely: lambda 0.42 vs 0.5, tamper -1.52 vs -1.5.
- Caveat: this evidence comes from a stylised simulation, not a real 4B agent. It supports "no sensitivity at the decision level", nothing more.

**Is it a step function?** Partly. The sceptic confirmed the key point.
- Expected reward is linear in p, so a threshold policy is the correct Bayes decision. That step is principled.
- The real artefacts are small. The lazy cliff pays -0.1 at 0 commands and 0.25 at 1, and one trivial command defeats it. Exact-target match is binary, 0.5 vs 1.0. Pass is all-or-nothing.
- Neither side proved that partial credit for pass fraction or near-miss targets helps. The auditor called it prone to being gamed (Goodhart), and the sceptics said to wait for Stage 1 data.

**Smooth confidence term.** Confidence is currently unrewarded.
- Overconfidence costs exactly 0. One run therefore buys one operating point, and each alpha needs its own run.
- The simulation showed an all-task Brier term (B2) cuts regret from 0.11 (shipped feasible-only version) to 0.032, and a fully continuous variant reaches 0.017. A dedicated run per alpha reaches 0.005–0.008. I did not re-run these regret numbers; the sceptic called them unverified.
- Auditor and sceptic disagreed on severity. The auditor said major. The sceptic said minor, because the pool gate (safe_alphas, callbacks.py:572) already screens out hard pools. The sceptic noted the headline "31% collapse" averages over a pool that gate excludes. Neither design was tested on a real model, and the Brier signal is tiny (overconfidence margin 0.002–0.007).
- The shipped brier_weight is feasible-only, so it is the wrong shape. Leave it off or delete it.

**Recommended reward: keep the current table, with these edits.**
1. Set lazy = 0. This removes the only true cliff, and it is inert at 1 command anyway.
2. Tighten validation: wrong_reason ≤ 0.9, and min_attempt_actions ≤ 2 or dropped. false_blame stays as is, since the owner rejected changing it (V3_STATUS:83).
3. Hard-skip any alpha whose (1+alpha)/2 exceeds the Phase-0 pass rate minus 0.1. Today safe_alphas only reports.
4. Free constants: **alpha** (grid 0, 0.125, 0.25, 0.375) and the **blame threshold** (via wrong_reason). Everything else is ordering-only and fixed.
5. Document the constants in the paper as ordering constraints, not tuned values.
6. Report the realised threshold. On infeasible tasks a decline pays w, which raises the threshold by up to 0.125 at q=1 (RD-5).

**Optional arm, after Stage 1.** Add one B2 ablation:
R += −0.4·(p − pass)², applied on every graded episode, with p = 0.5 if there is no report. This needs lambda ≥ 0.5. Cost is about 2–3 days with no Docker or GPU. It is an ablation, not a pre-run change.

## 2. Re-examined risk register (ranked)

1. **Hermetic verifier, weight sync, false tamper.** Already known and still top. Only a Docker run can retire them. Replay the oracle solution through the full grader on 20 variants first.
2. **Grader failures are invisible to stop rules (CA-1).** Confirmed by the sceptic. See section 3.
3. **Evaluation cannot support the headline claims (section 4).** New since V3_STATUS.
4. **Data labels.** About a quarter of remove_input variants leak, by 13 hand-read variants (3 clear, 2 unsure). The sceptic cut the training impact, because the leak mask neutralises a leaky group. The risk remains in test rows. Hand-audit every test remove_input row.
5. **Training covers two mechanisms, not four (D2).** Already known, but "about 4% of seeds" is wrong. Recomputed: 148 high_risk, 101 revoke, 72 conflicting_constraint, 31 remove_input, 1 no_network, 0 remove_tool infeasible variants per 150 seeds. Scope the paper claim to two mechanisms.
6. **Trainer padding waste (T1).** Padded tokens are 1.7–3.5× actual, and the 2-sequence logits at 16k tokens need about 15 GiB in bf16 plus 30 GiB for an fp32 copy. The 12–17 h run-time figure is an unverified estimate, so don't quote it.
7. **Premise unmeasured.** No gate checks pool pass rate, abstain recall or false refusal. The sceptic downgraded this to minor, since a low pass rate makes the intended behaviour easier. It does weaken the "calibrated claiming" story.
8. **Smaller items.** These are an eos check, a vLLM server bound to 0.0.0.0 in dev mode, and decontamination silently off when benchmarks/ is missing (0/150 hits when checked by hand).

## 3. What the double check found

**Reward correctness.** An independent reference implementation agreed with compute_reward on 200,000 random outcomes, with 0 disagreements. The claim threshold held in 180 of 180 parameter combinations.

**Mutation testing.** I count 88 deliberate bugs injected. 78 were caught. All 36 in reward.py were caught. 10 survived: 3 are real gaps (a probe death counts as "verified", `~/` expansion, self_check ignoring tamper) and 7 are benign. Killer tests exist. The sceptic could not re-run them outside the repo, so copy them into tests/.

**Confirmed bugs.**
- **CA-1 (major, confirmed).** A grading error becomes a tamper outcome scored -1.5, and the reward_error_rate stop rule never sees it. In the failing test all rewards were -1.5, tamper_rate 1.0 and reward_error_rate 0.0. A partial outage (about 20% flaky exec) would add noise and fire nothing. Fix:
  - count outcomes with grading_error as the error rate;
  - add a tamper-rate rule on a 5-step window;
  - retry transport errors once.

  Effort: hours.
- **CA-2 (minor).** A teardown exception after a good grade scores a correct episode -1.5. Wrap the stop call in suppress plus a timeout.
- **T2 (minor).** No stop rule on sampler drift or clipped share. Add two rules and a probe-logprob check after each sync.
- **Stage 1 gate.** A tiny-adapter sync check is worth adding, since the current one cannot see bf16 merge rounding.

**Cue leakage numbers.**
- In conflicting_constraint, the path name alone predicts infeasible vs control with 76% accuracy (chance 50%). Infeasible variants forbid deliverables and controls forbid inputs.
- The sceptic accepted the number but did not re-run it. The sceptic also noted the cue is partly the correct concept. The risk is over-generalising "output-looking path means abstain".
- The full instruction text alone gave 0.50, but the sceptic called that a weak-classifier artefact.
- Cheap fix: add a decoy output path as a control, and report over-abstention on held-out controls.
- revoke_permission labels the file, but the lock is on the parent directory, so a correct directory diagnosis scores as the wrong target. Impact is small, about wrong_reason instead of 1.0. Add the directory to expected_targets.

## 4. What the planned experiments can and cannot show

**Cannot show yet (all major, partly assumption-based).**
- **NI is on task_success, not verified success (SP1, confirmed).** A policy that works but never claims passes NI. Hand case: diff 0.0 and non-inferior, with verified success 0.0 vs 0.5. Make verified_success_rate primary and keep task_success as secondary.
- **Seed variance is outside every interval (SP2).** With a 3-point margin and 3 seeds, NI power at true difference 0 is 0.73, 0.46 or 0.30 for seed SD of 1, 2 or 3 points. The SDs are assumptions, not measurements. Add a seed-aware interval and set the margin after measuring seed SD.
- **The confidence-based realised threshold is not identified (SP3).** True thresholds 0.5, 0.7 and 0.9 all read 0.6 when confidence is coarse. Use the outcome bracket [pass rate among unclaimed, 1 − fabrication] instead.
- **The alpha trend can't be resolved pairwise at one run per alpha (SP4).** Claim a monotone trend, not "smooth control".
- **The pilot GO/NO-GO rule is noise-sensitive (SP5).** With no true effect it says GO 0.3% of the time. With a real 0.2 drop it says GO 62% without seed noise and 10% at 0.3 logit seed noise. INCONCLUSIVE never fires. The 0.02 monotone tolerance drives most NO-GOs. Replace it with cluster-bootstrap confidence intervals, and raise sweep.seeds to 2–3 at no GPU cost.

**Can show.** Task-sampling precision is good: abstain recall to about 3.4 points and P(claim|fail) to about 4 points at one seed. The most important limits are per-family. no_network cannot be reported. Held-out transfer (H3) is weak with only two trained mechanisms.

Smaller points:
- A pilot GO is necessary, not sufficient.
- No multiplicity control. Pre-register two primary endpoints.
- m=1 certifies the decision, not the diagnosis.

## 5. Ordered fix list and Stage 1 plan

| # | Fix | Effort |
|---|---|---|
| 1 | CA-1: count grading errors, tamper-rate stop rule, one retry | hours |
| 2 | NI primary = verified success; seed-aware interval; two pre-registered endpoints | hours to days |
| 3 | Hard-skip unsafe alphas; set lazy=0; tighten validators | hours |
| 4 | T1: sort and trim micro-batches, with a loss-equality test; default per_device 1 until OOM headroom is measured | hours |
| 5 | T2 and the eos assertion: drift and clipped-share rules; bind vLLM to 127.0.0.1 | hours |
| 6 | Data: decoy-output controls, revoke directory target, remove_input candidate filter, audit every test remove_input row | hours |
| 7 | CA-2 teardown guard; copy killer tests into tests/ | hours |
| 8 | Pilot: bootstrap intervals, seeds 2–3, count prompts not episodes | days |
| 9 | Optional B2 ablation after Stage 1 | 2–3 days |

**Stage 1 changes.**
- Split the rental, as the sceptic confirmed (R5). Run docker tests, the oracle replay, validation, the null-agent check and the audit on a CPU Docker VM. Use a single GPU for the base eval and sync check. Rent the 4×H100 node only after those pass.
- Add to Phase 0 these reported numbers: pool pass rate and histogram, abstain-correct rate on impossible variants, false refusal on near-miss controls, and the clipped share split by feasible and infeasible.
- Hard gates stay no_report and P(claim|fail). Do not hard-gate on pass rate.
- Fix NI and the pilot rule before any paid training run.

## Status (overnight pass)

Reward/grader (CA-1, CA-2), trainer (T1, T2, tiny-adapter sync, safe-alpha skip), stats (SP1-SP5) and data fixes are applied; see `docs/V3_STATUS.md` section 1a. The ten mutation survivors have killers in `tests/test_mutation_survivors.py`; V06 is an equivalent mutant.
