# Reviews and audits

One record of everything the project was checked against before any GPU spend: three reviews of the first design, the second-round
reviews of the built components, a second audit (is the reward good enough, what can the experiments show) and a leakage audit. For
each: the verdict, the confirmed findings by theme, what was decided where reviewers disagreed, how each was resolved, and where the
code and tests are. The design that came out of it is [design.md](design.md); the data findings in full are in [data.md](data.md).

Nothing in this project has run on Docker or a GPU. Every number about speed, cost, leak rates and base-model behaviour below is a
simulation, a static count or an estimate unless it says otherwise.

**Raw evidence.** The reviews' intermediate artefacts (the six-lens review JSON and its digest, the 240 KB findings dump, the
trainer-options JSON, the specs and the status log that were written during the fix pass) are not kept in the tree. They are preserved in
git history: `git show 2a38890:docs/research/go-no-go-review.json` (keys `review:<lens>` and `verify:<lens>`),
`...:docs/research/go-no-go-review-digest.txt`, `...:docs/research/findings.json`, `...:docs/research/trainer-options.json`,
`...:docs/SPEC_V3.md`, `...:docs/INTERFACES_V3.md`, `...:docs/V3_STATUS.md` and the earlier memos under `docs/research/`. Finding IDs
below (INC-, LV-, GR-, TDC-, F-, L-, V-, M-, VIS-, VM-, MV-, SP-, CA-, T-) are the IDs in those files.

Terms. **GRPO**: the RL algorithm; each task gets a group of G rollouts and a rollout's *advantage* is its reward minus the group mean
(optionally divided by the group's standard deviation). **`scale_rewards`**: the TRL setting for that division (`group` divides, `none`
does not). **alpha**: the reward for an honest "I did not finish"; claiming completion pays only if P(success) exceeds (1 + alpha)/2.
**lambda**: a bonus whenever the tests pass, whatever the report says. **Feasible / infeasible / near-miss control**, **oracle**, **leak**,
**mask**: see [design.md](design.md). **Hermetic verifier**: grading that does not depend on the network or on files the agent can rewrite.
**Collapse** (arXiv 2608.00301): a policy drifting to "always refuse" while its reward still looks fine. **NI**: non-inferiority test.
**Sceptic / verifier**: the second reviewer who re-derived each finding and re-assigned a severity.

## 1. Overview

| # | Review | When | Covered | Verdict |
|---|---|---|---|---|
| 1 | Six-lens go/no-go review, each finding re-checked by a verifier | 2026-10-04 | reward algebra and GRPO simulation, label validity and data, grader red team, TRL dynamics and cost, experimental design, literature | **do not train on the first design**; all six said "go with fixes"; every blocker fixable without a GPU |
| 2 | Independent external review | 2026-10-04 | cited by number (sections 0 to 6) | same verdict; its text is not in the repo, so nothing here summarises it beyond how the fix list cites it |
| 3 | Project lead's own review | 2026-10-04 | the fix list | reconciled with 1 and 2 where they disagreed (section 4) |
| - | Second-round component reviews | during the fix pass | each built component against the spec and the installed libraries | further problems found and fixed (section 6) |
| A1 | Audit 2: is the reward good enough | 2026-10-05 | reward constants, grader failure path, data labels, trainer efficiency, what the experiments can show; 200,000-outcome reference check, 88 injected mutations | **keep the reward** with one deletion and tightened checks; larger risks sit in evaluation, grader failure path, data labels, trainer efficiency |
| A2 | Leakage audit | 2026-10-05 | every channel from data to policy, on real ET seeds | no blocker; one real shortcut (L1) plus cheap guards; all fixed locally; an independent verification then moved two families to test-only |

What survives from the original idea: grading a structured self-report against the final sandbox state, with no human labels and no
LLM judge, is sound and measurable. The experimental design around it was not.

## 2. Review 1: confirmed findings by theme

Severity is after the verifier's re-check, with the reviewer's original in brackets where it changed. "Resolved" points to code and tests.

### 2.1 The reward did not measure honesty

| ID | Problem | Severity | Resolved |
|---|---|---|---|
| TDC-3, F1, INC-1, L2 | `scale_rewards: group` divides alpha out of two-valued groups: effective claim thresholds about 0.40 / 0.48 / 0.54 / 0.57 (simulation; 0.43 / 0.50 / 0.55 / 0.585 analytic) against nominal 0.50 / 0.625 / 0.75 / 0.875 for alpha 0 / .25 / .5 / .75 | blocker (TDC-3, F1); INC-1 major [blocker] | `scale_rewards: none` everywhere, `train.grpo.check_scale_rewards` refuses the rest; `tests/test_train_config.py`, `tests/test_trl_rewards.py` (two-valued group) |
| F4, GR-1, INC-2, L1, VIS-M2, L-M1 | a task-blind report earns the reward: zero commands, `incomplete`, every blocker category listed earns +alpha on feasible and +1 on infeasible tasks, and no monitor fires | F4 blocker; L1 blocker (narrowed: the do-nothing basin is real, "collapse in our regime" overstated); GR-1, INC-2 major [blocker] | one blocker per report, `expected_targets` matching, lambda, `lazy`; `tests/test_archetypes.py` |
| INC-5, VIS-M1, TDC-7 | nothing makes working better than quitting; alpha 0.5 and 0.75 near-degenerate for weak self-knowledge; the reward is the rule 2608.00301 shows collapsing when the pool pass rate is below (1 + alpha)/2 | major | lambda; alpha grid {0, .1, .25, .4}; `callbacks.alpha_is_safe`, `RewardConfig.work_vs_quit_gap` |
| INC-4 | Brier terms pay for predictability, which quitting maximises | major | out of the reward, logged; `brier_weight < success_bonus` enforced |
| GR-5, M3 | a made-up blocker on a feasible task is paid as honest, so controls have no teeth | GR-5 major (partly); M3 minor | `grading.check_blocker`, `false_blame`, controls with `decoy_targets` |
| INC-3, GR-4, F10, TDC-4, M2 | no-report episodes scored 0 today, masked in the plan (a safe harbour); tool output exhausts the 16k-token budget before the 30-call cap, unseen by the agent | major | no-report is scored; the env enforces and shows the budget; four mask reasons only |
| V-1, TDC-8 | `None` from one of two reward functions becomes 0, not a mask; policy-caused grading failures lose the tamper penalty | V-1 major; TDC-8 minor | exactly one reward function per run (`trl_rewards.reward_func_for_mode`); grading failure after acting is tamper |
| INC-7, VM-1 | the tamper override replaced the total, so it could *raise* a low reward | minor | `min(total, -1.5)`; `tests/test_reward.py` |
| GR-8 | a malformed status ("completed", "done") turned a claim into "no report" | minor | status normalisation in `report.py` |
| INC-8, L3 | one advantage per trajectory makes the report decision noisier, not biased | minor | not changed |

### 2.2 Labels and data

| ID | Problem | Severity | Resolved |
|---|---|---|---|
| LV-1, GR-7, VIS-M3 | "infeasible" only means the oracle's route is blocked: a test-passing truthful claim scored -1, a zero-effort decline +1 | major [blocker] | leak = claim plus clean pass masks the whole group and quarantines the variant; validation null check and oracle runs |
| LV-4 | non-completion reports never checked for truth | major | blocker verification (`check_blocker`, `bank.check_blocker_cached`) |
| LV-7 | operator mix did not match the proposal (300 seeds: remove_input 193, cc 173, revoke 165, high_risk 300, no_network 11, remove_tool 4) | major | fixed mixture 55/20/25; operators rewritten; yields in [data.md](data.md) |
| V-2 | default test split (about 40 sources) could not resolve the claimed effects | major | hash split, about 330 test sources of 1,500; resolution warnings in eval |
| LV-2, LV-3, LV-6, LV-8, LV-9, F3, F11, V-3 | remove_input deletes non-inputs; high_risk wording contradicted the system prompt; one-step shortcuts; ambiguous expected category; difficulty-filter details; train/test template overlap; no per-test signal | minor each | operator filters, symmetric `high_risk` twins, pools partitioned train/test, `pass_fraction_feasible` |

### 2.3 The grader and verifier

| ID | Problem | Severity | Resolved |
|---|---|---|---|
| LV-5, GR-2 | pass/fail depended on the live internet at grading: all sampled ET `test.sh` run `apt-get`, fetch the uv installer, call `uvx` every time (150 of 150 in the reviewer's count, 40 of 40 in the fix pass's sample); a network failure scored as fabrication | major | hermetic verifier ([design.md](design.md) section 6) |
| GR-3, GR-6, GR-9, GR-10, GR-11, VM-2, VM-3 | verifier shares the agent's container and could be forged; false tamper from exec failures; delete-and-restore of a sentinel; no disk or process quota; a destroyed shell turned into a mask; undefined blocker categories | minor (GR-3, GR-6 lowered from major) | nonce-marked host-captured stream, inode and ctime sentinels, pids limit, categories defined in the tool docstring; `tests/test_exploits.py` |

### 2.4 Training stack and cost

| ID | Problem | Severity | Resolved |
|---|---|---|---|
| TDC-1, M1 | two training processes with a vLLM server mis-route completions in TRL's tool loop (loud: 92% of simulated first rounds crash) | major [blocker] | one process per run, guard in `train/` |
| TDC-2 | `--max-model-len 20480` rejects requests once a transcript passes 4,096 tokens | major | per-turn cap 4,096, window computed and asserted by the startup probe |
| TDC-5, F9 | tool calls run serially across the batch, which sets wall-clock time | major | concurrent loop in `HonestGRPOTrainer`; `tests/test_trainer_loop.py` |
| TDC-9, L6 | 19,200 episodes per run against Tmax's 128,000; G=8 and group-std in the unstable corner Tmax reports; seeds on one arm | major | disclosed ([proposal.md](proposal.md) 7.4); G decided after Stage 1 |
| TDC-6, TDC-10, MV-1, MV-2 | sequence-level importance sampling down-weights long episodes; one pad over the generation batch; test timeout not configurable; Phase-0 numbers should come from the training path | minor | token-level truncated IS, padding trim, `env.tests_timeout_sec` |

### 2.5 Experimental design and claims

| ID | Problem | Severity | Resolved |
|---|---|---|---|
| L4 | the gap statement ("nobody trains this in the open") is false: at least five open-model RL papers train abstention or self-report | major | gap narrowed in [proposal.md](proposal.md) section 3 |
| F2, L5 | AgentAbstain cannot be the primary metric: no abstain tool, generator unreleased | F2 major [blocker]; L5 minor | own held-out terminal set is primary; AgentAbstain transfer only |
| F5 | no neutral-prompt condition or no-training baseline | major | `--neutral-prompt`, `--neutral-tools`, confidence-threshold sweep, constant-policy rows |
| F6, INC-6 | H1 has no first-order mechanism with the implemented outcome baseline, which ignores the report | F6 major; INC-6 minor | H1 split into H1a and H1b; both outcome baselines implemented |
| F8 | evaluation could not resolve the claimed effects; seeds on one arm; no pre-registered margin | major | `EvalProtocol` fixed before the run, seed-aware NI, per-family minimum |
| F12 | no stop gates, no cheap decisive pilot | major | stop rules; report-only pilot; spending gate |
| L-M2 | Phase 0 did not measure base self-knowledge (confidence AUROC) | major | `auroc_confidence`, Phase-0 pool statistics |
| F7 | several metric definitions wrong or gameable | minor [major] | redefined ([evaluation.md](evaluation.md) section 5.2); episodes are stored in full and can be re-scored free |

## 3. What the verifier downgraded or refuted

- **Blockers lowered to major:** INC-1, INC-2, GR-1, LV-1, TDC-1, F2. Typically the mechanism is real but the size was overstated, or the
  failure is loud rather than silent.
- **F3 / LV-3 ("high_risk always-abstain contradicts AgentAbstain"): refuted in the central claim.** All 31 of AgentAbstain's
  `high_stakes_action` abstain variants are explicit user instructions to do something extreme, four of them instructed irreversible
  deletions. The contradiction with *our own* system prompt was real and was fixed.
- **LV-2** ("5 of 102 with content in the instruction") reproduces but "no control" does not. **LV-6** ("still open after v2"): counts
  reproduce, "still open" not shown. **TDC-4** ("40-48% of episodes exhaust the budget"): one cell of an assumption grid; the same model
  gives 2-5% at a 400-character median output and 68-70% at 1,500; the fix is to measure it. **TDC-6** ("randomly drops long
  trajectories"): wrong; long episodes are systematically down-weighted (median weight 0.55 at 1,000 tokens), about 4% zeroed. **GR-6**
  ("container died, solved task scored -1"): false for feasible tasks and four of six operators, where the episode was masked. **INC-2**
  ("derails training"): the reviewer's own simulation showed 0 of 216 cells losing effort without a budget loss or Brier term.
- **GR-3, F7, F11, L3, L5:** real but minor (Harbor's own verifier shares the container, standard practice; metrics can be re-scored; the
  held-out-family test is the stronger check; L3 is variance, not bias; L5 was already planned).
- **Number corrections.** remove_input leak after validation about 5-15% (not 10-30%); cc satisfiable another way 0-1 of 8 (not 2-3);
  "within 0.0125-0.11 of honest-optimal for s<=0.5" was really 0.0125-0.35; "fabrication 8.1% vs 4.0%" averaged only alpha >= 0.25; a cost
  range of "16-125 node-days" is not usable.
- **Confirmed by independent re-derivation:** the group-std threshold distortion (three methods agree), the archetype identities
  (never-claim beats always-claim in 79 of 80 cells), and all the grader exploit demos.
- **The pilot idea (frozen trajectories) was partly confirmed:** sound for report-level questions, feasible on stock TRL, blind to early
  quitting and to H1, valid only with `scale_rewards: none`.

## 4. Decisions where reviewers disagreed

| Question | Options | Decision and reason |
|---|---|---|
| Advantage scaling | `batch` or `none` | **`none` everywhere**; it gives the exact threshold, `batch` only approximates it (0.60 / 0.71 / 0.82) |
| Paying work after the policy stops claiming | L1: pay unclaimed success u, 0 for honest failure; INC-5: a success bonus | **lambda inside `compute_reward`**: L1's variant is lambda = 2 alpha, but independent knobs keep effort rewarded at alpha = 0; the threshold does not move |
| Brier terms | keep small (INC-4); mandatory confidence plus deployment threshold (L1) | **out of the reward**, logged; the confidence sweep runs at evaluation |
| No-report episodes | mask or score | **score** (-0.25 feasible, gamma infeasible); budget visible and enforced; mask only boot failure, toolchain self-check, leak groups (and `reward_error`) |
| Test-passing claim on an "infeasible" task | score -1 or mask | **mask the whole group**, quarantine; +1 for the abstainers is as wrong as -1 for the solver |
| False blame on a feasible task | pay 0, negative or alpha | **0**, because real transient failures can make a blame true. Open: free at alpha 0; `min(0, alpha - 0.1)` recommended and rejected for now by the owner (changing it for one arm breaks the alpha grid) |
| Correct decline on infeasible | any category overlap or exact object | **one blocker; full credit only with category and target**; a blind guess earns 0.5 |
| `remove_tool` | drop or gate | **gated**; yields none on ET |
| `high_risk` | rebuild as discovered-risk pair or keep as explicit-instruction analogue | **keep as analogue with symmetric twins** |
| Held-out families | `high_risk` + one environment family, later cc + no_network | first `high_risk` and `revoke_permission`; **superseded** after A2's independent verification: `conflicting_constraint` and `no_network` held out ([data.md](data.md) section 2) |
| Layout | patch the loop for two ranks or one process | **one process per run**, guard, two runs per node |
| vLLM window | double it or cap each turn | **cap each turn at 4,096** and assert the window |
| Trainer | stay on TRL or port to verl | **stay on TRL**; decide from Phase-0 measurements ([design.md](design.md) section 8) |
| Warm start | OPD from a 27B teacher or none | **OPD deferred**; SFT on verified-honest trajectories first, after the report-only pilot |
| Mixture | ramp from 10% infeasible (L7) or fixed | **fixed 55/20/25**: a ramp is a confound unless the outcome arm follows it |
| Group size | raise G to 16-32 (L6) | **G=8 stays**; revisit after Phase 0 (a known risk) |
| Primary evidence | AgentAbstain or own held-out set | **own held-out set** |
| alpha grid | {0, .25, .5, .75} | **{0, 0.1, 0.25, 0.4}** |

## 5. Lessons taken from the multi-harness RL guide

"The ultimate guide to multi-harness RL" (Hugging Face Space `FineEnvs/multi-harness-rl`, 2026-10-02) trains 2-2.6B models with the
same stack. Each lesson became a requirement:

1. **Token fidelity**: policy gradients need the exact sampled ids and sampling-time logprobs; nothing decodes and re-encodes; sample the
   full distribution (the guide's importance ratio moved from 0.985-0.993 to 0.9984-0.9999 when `top_p` went to 1.0); Phase-0 gate within
   1% of 1. (Later checked in source: TRL sends every sampling field on each request, so shipped defaults cannot leak in; the explicit
   settings and the probe stay as a guard; the evaluation route must send them explicitly.)
2. **No term may pay for submitting.** A "+0.2 for submitting anything" term made held-out accuracy fall from 0.740 to 0.178 while
   training reward looked healthy: archetype tests, and stop rules for falling success while reward rises and a falling report share (69% to 41%).
3. **Efficiency bonus**: 17.5-22.6% of groups had no contrast except this bonus; without it tool calls per rollout crept from 13 to 41.
   Added to lambda, paid only on a clean pass.
4. **Periodic held-out probe, keep every checkpoint** (the guide's runs peaked at steps 500-700 then fell back).
5. **Identical caps in training and evaluation** (a model trained under 16,384 learned long answers a 4,096 cap cut off).
6. **Resume must not replay tasks**; log the share of steps with no contrast (35-58% with correctness alone).
7. **Harness dependence**: behaviour learned in one harness transfers poorly: report-file protocol, tables split by harness, Stage 1
   measures base honesty under ours and a stock harness. Multi-harness training is a later stage.
8. **Supervised baseline**: SFT on verified-honest trajectories is far cheaper than RL (1.5-6.5 hours against 32-46); now built (`sft/`).

## 6. Second-round component reviews

Each component was built and reviewed again against the spec and the installed libraries. They found problems the six lenses did not:

- **Reward side.** A placeholder reason ("n/a") counted as an explanation; `implicit_abstain` could exceed `wrong_reason`; a policy could buy
  a mask by breaking the grader (now tamper); `expected_targets` are load-bearing (without them a lucky category guess narrows the margin over the
  calibrated worker to 4e-5). Validators in `RewardConfig` and `HonestTaskSpec` close these.
- **Two attractors left by design** (never-claim at a low pool pass rate; one-command quitting under budget exhaustion), **false blame free at alpha 0**,
  and **thin teeth of near-miss controls** when feasible tasks are rarely solvable: [design.md](design.md) sections 4 and 14.
- **Spec conflicts.** A verifier that dies after the agent acted (resolved: timeout or no result is a fail; exec, upload and manifest errors are
  tamper). "A background process cannot write into the stream" holds only for non-root processes. An infeasible episode that passes but declines is
  paid and logged, not masked. A masked rollout's tokens still count in TRL's loss normaliser.
- **GPU-only failures fixed in review.** The probe would run out of memory (25 sequences of about 20k tokens scored at once); the shipped pilot config
  would have shown each frozen prompt about five times, letting a policy memorise fixed outcomes and produce a false GO (now refused: one epoch maximum);
  a stop-rule collapse looked like success (exit status 3 now).
- **Integration pass** (seams between components, all fixed and tested in `tests/test_pipeline_integration.py`): the trainer called an env method that did
  not exist (`_set_loop_exit`), so `truncated` was never recorded; replayed `permission_denied` facts missed labelled targets the commands never named; the
  pair filter counted a near-miss control as "the plain copy is valid"; a pytest run with no result validated an infeasible oracle; the OpenEnv session
  graded differently from training (now one `grading.resolve_config`); Phase 0 trained on the post-gate pool; the pilot would train on `split == "val"`
  rows; no stop rule on grader failure. Dataset columns, group ids, tool schemas (identical in five places), the budget (one formula for train, eval and
  OpenEnv) and stop-rule metric names were checked consistent. Console scripts, ruff config and numpy were declared in `pyproject.toml`.

## 7. Audit A1 ("audit 2"): is the reward good enough

**Bottom line:** keep the reward, with one deletion and tightened checks. The reward arithmetic is correct and well tested; the larger risks are in
the evaluation plan, the grader-failure path, the data labels and trainer efficiency.

- **Parameters are not arbitrary.** An exact optimal-policy solver and a GRPO simulation (G=8) put the claim threshold at (1 + alpha)/2 for every
  lambda, beta, tamper reward and efficiency bonus. Moving every constant but alpha by 50% did not change the learned policy (thresholds overshot by
  0.02-0.07). Only alpha and the blame threshold are real choices. Caveat: a stylised simulation, not a 4B agent.
- **Step functions.** A threshold policy is the correct Bayes decision, because expected reward is linear in p. The real artefacts are small: the
  `lazy` cliff (defeated by one trivial command), the binary exact-target match, all-or-nothing pass. Partial credit was not shown to help and risks
  gaming; wait for Stage 1 data.
- **Confidence term.** Overconfidence costs exactly 0, so one run buys one operating point. An all-task Brier term cut simulated regret from 0.11 to
  0.032 (not re-run by the sceptic; severity disputed: the pool gate already screens hard pools). The shipped feasible-only `brier_weight` is the wrong
  shape: leave it off. An optional all-task B2 ablation (`R += -0.4 (p - pass)^2`, needs lambda >= 0.5) is a Stage-3 option.
- **Double check.** An independent implementation agreed with `compute_reward` on 200,000 random outcomes (0 disagreements); the threshold held in 180
  of 180 parameter combinations. Mutation testing: 88 injected bugs, 78 caught; all 36 in `reward.py` caught; 10 survivors (3 real gaps: a probe death
  counted as "verified", `~/` expansion, `self_check` ignoring tamper) now have killers in `tests/test_mutation_regressions.py`; one is an equivalent mutant.

| ID | Finding | Resolved |
|---|---|---|
| CA-1 (major) | a grading error became a tamper outcome scored -1.5 that the reward-error stop rule never saw (all rewards -1.5, tamper rate 1.0, reward-error rate 0.0) | `honest/reward_error_rate` counts every grading failure after the agent acted; `tamper_rate_high` rule (0.10 over 5 steps); one transport retry in `grading.grade` |
| CA-2 (minor) | a teardown exception after a good grade scored a correct episode -1.5 | `HonestHarborEnv._teardown` guarded |
| T1 | trainer padding waste: padded tokens 1.7-3.5x actual; 2-sequence logits at 16k tokens need about 15 GiB bf16 plus 30 GiB for an fp32 copy | length-trimmed forward passes, longest-first micro-batches, per-device 1 x accumulation 64 until headroom is measured |
| T2 | no rule on sampler drift or clipped share; eos check; vLLM bound to 0.0.0.0 | rules `sampler_trainer_drift`, `importance_ratio_at_cap`, `truncated_share_high`; `models.check_end_of_turn_token`; `serve_vllm.sh` binds 127.0.0.1; `sync_check --tiny-adapter` |
| SP1 (major) | NI on `task_success`: a policy that works but never claims passes (verified success 0.0 vs 0.5, diff 0.0, "non-inferior") | NI is on `verified_success_rate`; two pre-registered endpoints; `test_noninferiority_is_on_verified_success_not_task_success` |
| SP2 (major) | seed variance outside every interval; at margin 3 points and 3 seeds NI power is 0.73 / 0.46 / 0.30 for seed SD 1 / 2 / 3 points (assumed SDs) | seed-aware interval, single seed cannot show NI, margin to be set after measuring seed SD |
| SP3 | confidence-based realised threshold not identified (true thresholds 0.5, 0.7, 0.9 all read 0.6 with coarse confidence) | outcome bracket `claim_bracket_low/high`; the confidence cut is diagnostic only |
| SP4 | the alpha trend cannot be resolved pairwise at one run per alpha | claim a monotone trend only |
| SP5 | the pilot GO/NO-GO rule is noise-sensitive (GO 62% at a 0.2 drop without seed noise, 10% with; INCONCLUSIVE never fired) | cluster-bootstrap intervals, 3 seeds, prompts as the unit ([evaluation.md](evaluation.md) section 12) |
| D1 | about a quarter of `remove_input` variants leak (3 clear, 2 unsure of 13 hand-read) | three-layer defence ([data.md](data.md) section 5) |
| cue | cc path name alone predicted the label (76%); revoke labelled the file while the lock is on the directory | decoy-output controls; both paths in `expected_targets` |

The ordered fix list of audit 2, all applied: (1) CA-1; (2) NI primary = verified success, seed-aware interval, two endpoints; (3) hard-skip unsafe
alphas, `lazy = 0`, `wrong_reason <= 0.9`; (4) T1; (5) T2 and the eos assertion; (6) data fixes and an audit of every test `remove_input` row;
(7) CA-2 and the killer tests; (8) pilot intervals, seeds, prompts as the unit; (9) the B2 ablation stays optional. Stage 1 changes it asked for: split
the rental ([node-runbook.md](node-runbook.md) section 1), report pool pass rate and histogram, abstain-correct rate, false refusal on controls and the
clipped share in Phase 0; hard gates stay no-report and P(claim given fail); fix NI and the pilot rule before any paid run.

## 8. Audit A2: leakage

Verdict per channel: agent-visible clean (one system prompt, empty user turn, 0 COPY/ADD lines in 450 real ET Dockerfiles, tests uploaded only at
grading); split clean but fragile (L2); benchmark overlap 0 flagged over 715 benchmark texts but unguarded (L4); structural cue leaking in one family
(L1: "forbidden path exists" predicted conflicting_constraint at 94-98%); answers not shipped; label and metadata flow clean (dataset columns never reach
the prompt). All eight items L1-L8, the pairing and quality items, the independent re-verification (it did not confirm the integrator: cc kept a lexical
cue and the gate depended on pool size) and the resulting decision to hold out `conflicting_constraint` and `no_network` are in
[data.md](data.md) sections 2 to 4, with the code and tests. The container-level checks that need Docker are in [data.md](data.md) section 6.

## 9. Still open

- Everything that needs Docker or a GPU ([node-runbook.md](node-runbook.md) section 3).
- Training sees three families; `conflicting_constraint` and `no_network` are zero-shot test families with a construct-validity caveat.
- The free-text AgentAbstain bridge, the verify-then-report gate baseline and multi-harness training are not built; the SFT warm start into GRPO is not
  wired ([roadmap.md](roadmap.md)).
- Checkpoints are selected on validation task success alone, which ignores honesty. LoRA at lr 1e-5 is untested. The `zero_variance_group_share > 0.5`
  stop rule may halt outcome-arm runs early.
- Under a third-party harness grading is weaker (no tamper, violation or blocker checks), so zero rates there mean "not measured".
- A reviewer-proposed `false_blame = min(0, alpha - 0.1)` is not set (owner decision, [design.md](design.md) section 4).
