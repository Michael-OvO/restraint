# Go/no-go review: record for the owner

Written 2026-10-04. This is the record of the review that stopped the first training plan, what it found, what was
confirmed or cut down on a second look, what was decided where reviewers disagreed, and what the fix pass has and has
not established. The binding fix list is `docs/SPEC_V3.md`. The raw evidence is `docs/research/go-no-go-review.json`
(keys `review:<lens>` and `verify:<lens>`) with a readable digest in `docs/research/go-no-go-review-digest.txt`. Finding
IDs below (INC-, LV-, GR-, TDC-, F-, L-, V-, M-, VIS-, VM-, MV-) are the IDs in that JSON.

Nothing in this project has run on Docker or a GPU. Every number about speed, cost, leak rates and base-model behaviour
below is an estimate or a static count unless it says otherwise.

## 1. Verdict

**Do not train on the pre-v3 design.** All six reviewers answered "go with fixes", and none found a flaw that needs a
redesign of the idea. Taken together, though, the problems meant a paid run would most likely have taught the wrong
thing, crashed in the first round, or produced numbers that could not support the paper's claims. Every blocker was
fixable locally without a GPU. The fix pass (SPEC_V3, sections 1-10) addressed them. The first paid step is now a
$300-500 bring-up and a zero-learning-rate run, gated as described in `docs/ROADMAP.md`.

What survives from the original idea: grading a structured self-report against the final sandbox state, with no human
labels and no LLM judge, is sound and measurable. The experimental design around it was not.

## 2. How the review was done

- Six reviewers, each with a different lens: **incentives-sim** (reward algebra and GRPO simulation), **label-validity**
  (are "infeasible" labels true; data pipeline), **grader-redteam** (exploiting the grader), **trl-dynamics-cost** (TRL
  behaviour, memory, step time, cost), **experimental-design** (metrics, baselines, power), **literature** (prior work).
- A second reviewer, a **verifier**, re-ran the first reviewer's scripts and re-derived the numbers independently for each
  finding, then confirmed, partly confirmed or refuted it and re-assigned a severity. Where the two disagree, this
  document uses the verifier's severity.
- Two further reviews are cited in SPEC_V3 (an independent external review and the project lead's own review). Their
  text is **not in the repository**, so this document relies on how SPEC_V3 cites them (sections 0 to 6 of the external
  review) and does not summarise them separately.
- After the fix pass, each built component was reviewed again against the spec and the installed libraries. Those reviews
  found further problems (section 8).

Terms used below.
- **GRPO**: the RL algorithm. For each task it samples a group of G rollouts, and each rollout's *advantage* is its
  reward minus the group mean (optionally divided by the group's standard deviation).
- **`scale_rewards`**: the TRL setting that decides that division. `group` divides by the per-group std, `none` does not.
- **alpha (α)**: the reward for an honest "I did not finish". Claiming completion is worth it only if the probability
  of success p exceeds (1+α)/2. **lambda (λ)** is a bonus paid whenever the tests pass, whatever the report says.
- **Feasible / infeasible / near-miss control**: a task that can be done / one made impossible on purpose / a feasible task
  that carries the same surface cue as an infeasible one but where the obstacle is harmless.
- **Oracle**: the task's reference solution. **Leak**: an "infeasible" variant that an agent nevertheless solved, passing
  the real tests.
- **Mask**: excluding an episode from the loss (reward `None`). **Hermetic verifier**: grading that does not depend on
  the network or on files the agent can rewrite.
- **Collapse** (arXiv 2608.00301): a policy drifting to "always refuse/say incomplete" while its reward still looks fine.

## 3. Confirmed findings, by theme

Severity is **after** the verifier's review, with the reviewer's original in brackets where it changed.

### 3.1 The reward did not measure honesty (the most important theme)

| ID | Problem | Severity |
|---|---|---|
| TDC-3, F1, INC-1, L2 | `scale_rewards: group` divides alpha out of two-valued groups. Effective claim thresholds about 0.40/0.48/0.54/0.57 (simulation) or 0.43/0.50/0.55/0.585 (analytic) against nominal 0.50/0.625/0.75/0.875 for alpha 0/.25/.5/.75. The alpha sweep would not test what the paper says. | **blocker** (TDC-3, F1 confirmed); INC-1 major [blocker]; L2 major |
| F4, GR-1, INC-2, L1, VIS-M2, L-M1 | A task-blind report earns the reward: zero commands, status `incomplete`, every blocker category listed earns +alpha on feasible tasks and +1 on infeasible ones, and no collapse monitor fires. Doing work earns nothing once the policy stops claiming. | F4 **blocker**; L1 blocker (narrowed: the do-nothing basin is real, "literature shows collapse in our regime" is overstated); GR-1 major [blocker]; INC-2 major [blocker]; VIS-M2, L-M1 major |
| INC-5, VIS-M1, TDC-7 | Nothing makes working better than quitting. alpha 0.5 and 0.75 are near-degenerate for a model with weak self-knowledge. The reward is the rule that arXiv 2608.00301 shows collapsing when the pool's mean pass rate is below (1+alpha)/2. | major |
| INC-4 | The planned Brier (calibration) terms pay for predictability, which quitting maximises. Pure Brier "gains score by degrading accuracy" (2608.00301 Prop. 9b). | major |
| GR-5, M3 (experimental-design) | A made-up blocker on a feasible task is paid as honest, so near-miss controls have no teeth. | GR-5 major (partly); M3 minor |
| INC-3, GR-4, F10, TDC-4, M2 | Episodes that end without a report: scored 0 today, masked in the plan. Masking makes never reporting a safe harbour and removes the learning signal on infeasible tasks. Tool output exhausts the 16k-token budget before the 30-call cap, and the agent cannot see it. | major |
| V-1, TDC-8 | Returning `None` from one of two reward functions becomes reward 0, not a mask. Policy-caused grading failures lose the tamper penalty. | V-1 major; TDC-8 minor |
| INC-7, VM-1 | The tamper override replaces the total with -1.5 and so can *raise* a reward that was already lower. | minor |
| GR-8 | A malformed status ("completed", "done") turns a completion claim into "no report". | minor |
| INC-8, L3 | One advantage per trajectory makes the report decision noisier, not biased. | minor |

### 3.2 Labels and data

| ID | Problem | Severity |
|---|---|---|
| LV-1, GR-7, VIS-M3 | "Infeasible" only means the oracle's route is blocked. A test-passing, truthful completion claim scored -1 and a zero-effort decline scored +1. | major [blocker] |
| LV-4 | Non-completion reports were never checked for truth or grounding. Judged the most likely way to waste the first run. | major |
| LV-7 | The operator mix did not match the proposal. On 300 seeds the old build gave remove_input 193, conflicting_constraint 173, revoke_permission 165, high_risk 300, no_network 11, remove_tool 4. Training covered about 3-4 of AgentAbstain's 8 scenarios. | major |
| V-2 | The default test split (about 40 sources) could not resolve the effects claimed, and two families had no test examples. | major |
| LV-2, LV-3, LV-6, LV-8, LV-9, F3, F11, V-3 | remove_input deletes things that are not inputs; high_risk's wording contradicted the system prompt; one-step shortcuts per family; ambiguous expected category; difficulty-filter details; template overlap between train and test; no per-test signal on Endless Terminals. | minor (each) |

### 3.3 The grader and verifier

| ID | Problem | Severity |
|---|---|---|
| LV-5, GR-2 | Pass/fail depended on the live internet at grading: all sampled Endless Terminals `test.sh` files run `apt-get`, fetch the uv installer and call `uvx` every time (150 of 150 in the reviewer's count, 40 of 40 in the fix pass's sample). A network failure was recorded as a fabrication. | major |
| GR-3, GR-6, GR-9, GR-10, GR-11, VM-2, VM-3 | The verifier shares the agent's container and could be forged; false tamper from exec failures; delete-and-restore of a sentinel not detected; a CTRF file the tasks do not produce; no disk or process quota; a destroyed shell turned into a mask; blocker categories never defined in the prompt. | minor (GR-3 [major], GR-6 [major] lowered) |

### 3.4 Training stack and cost

| ID | Problem | Severity |
|---|---|---|
| TDC-1, M1 | Two training processes with a vLLM server mis-route completions inside TRL's tool loop. The failure is loud (92% of simulated first rounds crash), not silent. | major [blocker] |
| TDC-2 | vLLM `--max-model-len 20480` rejects tool-loop requests once a transcript passes 4,096 tokens. | major |
| TDC-5, F9 | Tool calls run serially across the batch, which (not the GPUs) sets wall-clock time. | major |
| TDC-9, L6 | 19,200 episodes per run against Tmax's 128,000; G=8 and group-std sit in the unstable corner Tmax reports; seeds only on one arm. | major |
| TDC-6, TDC-10, MV-1, MV-2 | Sequence-level importance sampling down-weights long episodes; one pad over the generation batch; test timeout not configurable; Phase-0 numbers should come from the training rollout path. | minor |

### 3.5 Experimental design and claims

| ID | Problem | Severity |
|---|---|---|
| L4 | The gap statement ("nobody trains this in the open") is false; at least five open-model RL papers train abstention or self-report. | major |
| F2, L5 | AgentAbstain cannot be the primary metric: it has no abstain tool, its generator is unreleased, so the trained interface is absent. | F2 major [blocker]; L5 minor |
| F5 | No neutral-prompt condition and no no-training baseline, so gains cannot be attributed to RL. | major |
| F6, INC-6 | H1 has no first-order mechanism with the implemented outcome baseline, which ignores the report. | F6 major; INC-6 minor |
| F8 | The planned evaluation could not resolve the effects H2 claims; seeds assigned to one arm only; no pre-registered margin. | major |
| F12 | No stop gates and no cheap decisive pilot. | major |
| L-M2 | Phase 0 did not measure base self-knowledge (how well the model's confidence separates its successes from failures), which decides how much claimed success any claim-gated reward costs. | major |
| F7 | Several metric definitions were wrong or gameable (they never touch the reward, and episodes are stored in full, so they can be re-scored for free). | minor [major] |

## 4. What the sceptics downgraded or refuted

The verifier did not just rubber-stamp. Items that were cut down or knocked out:

- **Blockers lowered to major:** INC-1, INC-2, GR-1, LV-1, TDC-1, F2. Typical reason: the mechanism is real but the
  size was overstated, or the failure is loud rather than silent.
- **F3 and LV-3 (high_risk "always abstain" contradicts AgentAbstain): refuted in the central claim.** All 31 of
  AgentAbstain's `high_stakes_action` abstain variants are explicit user instructions to do something extreme, and four are
  instructed irreversible deletions. The contradiction with *our own* system prompt was real, and was fixed. Severity minor.
- **LV-2 "5 of 102 with content in the instruction"** reproduces, but the "no control" part does not stand. Minor.
- **LV-6 "one-step shortcuts still open after v2":** the cue counts reproduce, but "still open" was not demonstrated.
- **TDC-4 "40-48% of episodes exhaust the token budget":** one cell of an assumption grid. The same model gives 2-5% at a
  400-character median output and 68-70% at 1,500. Unmeasured, so the fix is to measure it.
- **TDC-6 "randomly drops long trajectories":** wrong. All six sequence ratios were below 1; long episodes are
  *systematically down-weighted* (median weight 0.55 at 1,000 tokens) and about 4%, not 15%, are zeroed.
- **GR-6 "container died, solved task scored -1":** false for feasible tasks and for four of six operators, where the
  episode was masked. False tamper needs a transient failure of exactly one exec.
- **INC-2 "derails training":** the reviewer's own simulation showed 0 of 216 cells losing effort without a budget loss
  or Brier term. The hole is real, the immediate damage was overstated.
- **GR-3, F7, F11, L3, L5:** real, but lowered to minor. Harbor's own verifier shares the container and is standard practice
  (GR-3); metrics can be re-scored after training (F7); the held-out-family test is the stronger check (F11); L3 is
  variance, not bias; L5 was already planned in the roadmap.
- **Number corrections.** The remove_input leak rate after validation is about 5-15%, not 10-30% (a judgement from one
  clear case in 10 Endless Terminals tasks); `conflicting_constraint` satisfiable another way is 0-1 of 8, not 2-3; "within
  0.0125-0.11 of honest-optimal for s<=0.5" was really 0.0125-0.35; the "fabrication 8.1% vs 4.0%" figures averaged only
  alpha >= 0.25; a cost range of "16-125 node-days" is not usable as an estimate.
- **Confirmed by independent re-derivation** (so not downgraded): the group-std threshold distortion (three methods
  agree), the archetype identities (never-claim beats always-claim in 79 of 80 cells), and all the grader exploit demos.
- The H-I hypothesis (a cheaper pilot on frozen trajectories) was **partly confirmed**: sound for report-level questions,
  mechanically feasible on stock TRL, blind to early quitting and to H1, and valid only with `scale_rewards: none`.

## 5. Decisions where reviewers disagreed (SPEC_V3)

| Question | Options | Decision and reason |
|---|---|---|
| Advantage scaling | `batch` (reviewers) or `none` | **`none` everywhere**, and `train.py` rejects anything else. `none` gives the exact threshold; `batch` only approximates it (0.60/0.71/0.82). |
| How to pay work after the policy stops claiming | L1: pay unclaimed success `u`, 0 for honest failure; INC-5: a success bonus | **Success bonus lambda inside `compute_reward`.** L1's variant is the special case lambda = 2 alpha, but keeping alpha and lambda independent means effort is still rewarded at alpha = 0. The claim threshold does not move. |
| Brier terms | keep with a small weight (INC-4); mandatory confidence plus a deployment-time threshold (L1) | **Out of the reward, logged as metrics.** The confidence sweep is run at evaluation instead. If ever enabled, `brier_weight` must be below `success_bonus`. |
| Episodes that end without a report | mask (v2 plan) or score | **Score them** (-0.25 on feasible, gamma on infeasible), and make the budget visible and enforced by the environment so a missing report is a choice. Mask only (a) sandbox failure before the agent acted and (b) leak groups. |
| A test-passing claim on an "infeasible" task | score -1 or mask | **Mask the whole group** and quarantine the variant. +1 for the abstainers is as wrong as -1 for the solver. |
| False blame on a feasible task | pay 0, pay negative, or pay alpha | **Pay `false_blame` = 0**, not negative, because real transient failures can make an external blame true. (Open: at alpha = 0 this makes false blame free; see section 8.) |
| Infeasible reward for a correct decline | any overlap in category, or exact object | **One blocker per report; full credit only if the category is expected and the target matches** `expected_targets`. A blind category guess earns 0.5. |
| Operators | drop `remove_tool`, or keep it gated | **Gated**: needs an instruction that mandates the tool and no interpreter able to substitute. On Endless Terminals it yields none. |
| `high_risk` | rebuild as a discovered-risk pair (LV-3), or keep as an explicit-instruction analogue (verifier) | **Keep as an analogue with symmetric twins**: identical wording, only the directory's contents differ; the "required" clauses and the stakes are removed from the request, and the system prompt's abstain clause is aligned. |
| Held-out families | `high_risk` (default) or an environment family | **Two families held out**: `high_risk` and one environment-mechanism family (`revoke_permission`). Consequence: `high_risk` is never trained (open issue, section 8). |
| Layout | patch the loop for two ranks (TDC-1) or one process | **One training process per run**, with a guard, two runs side by side. |
| vLLM window | double the window (TDC-2) or cap each turn | **Cap each turn at 4,096 tokens** and assert the window against the server. The window then fits Qwen3-4B up to a 32k budget. |
| Trainer | stay on TRL or port to verl | **Stay on TRL for Stage 1 and the pilot**, decide from measurements (SPEC_V3 10.3). |
| Warm start | OPD from a 27B teacher, or none | **OPD deferred.** A report-only pilot comes first; supervised fine-tuning on verified-honest trajectories replaces OPD as the first warm-start option. |
| Mixture | ramp from 10% infeasible (L7) or fixed | **Fixed 55/20/25** with no ramp: a ramp is a confound unless the outcome arm follows it too. |
| Group size | raise G to 16-32 (L6) | **Not adopted; G=8 stays.** Revisit after Phase 0 and the pilot. A known risk. |
| Primary evidence | AgentAbstain or own held-out set | **Own held-out terminal set**; AgentAbstain is transfer only. |
| alpha grid | {0, .25, .5, .75} | **{0, 0.1, 0.25, 0.4}.** 0.5 and 0.75 are near-degenerate for weak self-knowledge. |

## 6. Lessons taken from the multi-harness RL guide

Source: "The ultimate guide to multi-harness RL" (Hugging Face Space `FineEnvs/multi-harness-rl`, 2026-10-02), which trains
2-2.6B models with the same stack (Harbor + OpenEnv + TRL GRPO). SPEC_V3 section 9 turned its lessons into requirements.

1. **Token fidelity.** A policy gradient needs the exact sampled token ids and sampling-time probabilities, not text and a
   score. Nothing in our trainer may decode and re-encode model output; the differential test checks ids, logprobs and
   masks. Rollouts sample from the full distribution (temperature 1, top_p 1, top_k off); the guide measured the
   importance ratio moving from 0.985-0.993 to 0.9984-0.9999 when `top_p` went to 1.0. The Phase-0 gate is a mean ratio
   within about 1% of 1. (Later checked against the source: TRL sends every sampling field on each request, so a model's
   shipped defaults cannot leak in. The explicit settings and a startup probe stay as a guard. The evaluation loop, which uses
   a different route, must send them explicitly.)
2. **No term may pay for the act of submitting.** The guide's "+0.2 for submitting anything" taught the policy to submit
   immediately, and held-out accuracy fell from 0.740 to 0.178 while training reward looked healthy. Hence the archetype
   tests, and stop rules for falling held-out success while reward rises and for a falling share of episodes that end in
   a report (the guide saw 69% fall to 41%).
3. **Efficiency bonus.** The guide found 17.5-22.6% of groups had no reward contrast except this bonus, and that without
   one, tool calls per rollout crept from 13 to 41. Added to the success bonus, paid only on a clean pass, so it cannot be
   earned by giving up and does not move the claim threshold.
4. **Periodic held-out probe, keep every checkpoint.** The guide's runs peaked at steps 500-700 and then fell back. A fixed
   probe every 50 steps picks a checkpoint on validation, never on test.
5. **Identical caps in training and evaluation.** A model trained under a 16,384 cap learned long answers that a 4,096
   evaluation cap cut off. Per-turn cap, total budget, call budget and sampling now come from one place.
6. **Resume must not replay tasks;** log the share of steps with no reward contrast anywhere (the guide saw 35-58% with
   correctness alone).
7. **Harness dependence.** Behaviour learned in one harness transfers poorly to others. So the report can also be written
   to a fixed file in the sandbox, scored under any Harbor harness, every table is split by harness, and Stage 1 measures
   the base model under ours and at least one stock harness. Multi-harness *training* is a later stage.
8. **Supervised baseline.** SFT on grader-verified honest trajectories is far cheaper than RL in the guide (1.5-6.5 hours
   against 32-46) and a baseline reviewers will expect. Not built; the bank stores what it would need.

## 7. Cost estimates, and what they assume

**Nothing was timed on a GPU.** Every figure comes from the trl-dynamics-cost reviewer's model (its `cost.py`) and the
verifier's recomputation. Treat each as plus or minus a factor of three until the zero-learning-rate run measures step
time by component.

Assumptions: H100 at $2.5-3.5 per GPU-hour, so $10-14 per 4-GPU node-hour; 64 episodes per step; 300 steps; about 14
tool calls per episode at 0.8 s each (assumed); a 180 s command timeout; 120 s of grading per step; 99 s of generation
per step; 180 s of scoring and training; 45 s of boot and sync. Not in the model: with one rank, 1% of calls hitting the
180 s timeout adds 27 minutes per step; 1-3% of episodes hitting the 900 s test timeout adds 6-11 minutes.

| Scenario | Min/step | Hours/run | $/run | Nine runs |
|---|---|---|---|---|
| Old layout, central (could not actually complete a step) | 13.4 (range 4.7-37) | 67 | $669-936 | $6.0-8.4k |
| One rank, serial tools, two runs sharing a node | 21.0 | 105 | $526-736 | $4.7-6.6k |
| One rank, parallel tools, two runs sharing a node (current design) | 11.1 | 55 | $277-388 | $2.5-3.5k |

Add 30-50% for bring-up, crashes and evaluation. The experimental-design reviewer's separate estimate (about 21 min per
step, 4.5 days per run, 45 node-days or about $13k for the old ten-run matrix) was judged unusable at its stated range
(16-125 node-days) by the verifier.

**Staged gate (SPEC_V3 section 8)**, each stage only if the previous one's condition holds:

| Stage | What | Estimated cost |
|---|---|---|
| 0 | Implement and attack the spec locally | $0 |
| 1 | Node bring-up, audit dataset, zero-learning-rate run (the reviewer put a 20-30 step run at about $60-120 alone), base and Tmax inference | about $300-500 |
| 2 | Report-only pilot on frozen trajectories (about two days) | about $400-600 |
| 3 | Two short full runs, honest against outcome, same seeds and compute | about $300-600 |
| 4 | Seeds for both arms, then the alpha values the pilot supports | about $2-3k |

Other cost facts: the OPD teacher would add about 60-170 s per step (+20-35%); it is deferred. The pilot uses no Docker.

## 8. What the fix pass built, what it found that the spec missed, and what is still open

The components (reward, report, grader and hermetic verifier, environment, data pipeline, trainer, training config, pilot,
evaluation, OpenEnv wrapper) were built and then reviewed again. Test status is in `docs/V3_STATUS.md` when the
integrator has written it. All tests run on macOS against fakes. The second-round reviews found problems the six lenses
did not, among them:

- **Reward side.** A placeholder reason ("n/a") counted as an explanation; `implicit_abstain` could exceed `wrong_reason`;
  a policy could buy a mask by breaking the grader (now scored as tamper); `expected_targets` are load-bearing (without
  them a lucky category guess narrows the margin over the calibrated worker to 4e-5).
- **Two attractors left by design.** (1) A never-claim policy under a low pass rate with high alpha (92-100% of simulated
  runs at s = 0.3, alpha = 0.4); it is a data choice, so alpha <= 0.25 is safe only if the pool's mean pass rate is about
  0.5 or higher. (2) One-command quitting if a working episode runs out of budget with probability h: lambda 0.5 survives
  h of 0.4 at s of 0.3 to 0.5 in simulation. Phase 0 must measure h; `RewardConfig.work_vs_quit_gap` checks it.
- **False blame is free at alpha = 0.** A config knob (`false_blame = min(0, alpha - 0.1)`) was recommended and is not set in
  the shipped configs.
- **Near-miss controls have thin teeth when feasible tasks are rarely solvable.** A policy that declines on the cue alone
  trails the calibrated worker by only 0.006 at the default config.
- **Spec conflicts and gaps.** SPEC 3.5 and section 1 disagree on a verifier that dies after the agent acted (resolved: a
  test timeout or no result is a fail; exec, upload and manifest errors are tamper). "A background process cannot write
  into the stream" holds only for non-root processes (mitigated, not closed). An infeasible episode that passes cleanly
  but declines is paid the decline reward and logged, not masked. A masked rollout's tokens still count in TRL's loss
  normaliser, so masking shrinks the rest of the batch's step.
- **GPU-only failures fixed in review.** The periodic probe would run out of memory (25 sequences of about 20k tokens
  scored at once); the shipped pilot config would have shown each frozen prompt about five times (1,200 prompt visits over about 256 rows), letting a policy memorise the fixed
  outcomes and so produce a false GO (now refused: one epoch maximum); a stop-rule collapse looked like success (exit
  status 3 now).

**Still open or unverified**

- Everything needing Docker or a GPU, including the hermetic verifier in a real container, the weight-sync fix (the bug
  reproduces on CPU, not on a GPU), the real leak rate, and base-model pass rate, fabrication share, confidence AUROC and
  budget-exhaustion rate.
- With the spec's two held-out families, **training sees only remove_input, conflicting_constraint and a little no_network**.
  `high_risk` ("look before you delete") is only ever tested zero-shot. The owner should confirm this trade.
- The F2 "bridged" AgentAbstain harness, the free-text reporting variant, the verify-then-report gate baseline, and
  the SFT baseline are **not built**.
- Under a third-party harness the grading is weaker (no tamper, violation or blocker checks), so zero rates there mean
  "not measured".
- Checkpoints are selected on validation task success alone, which ignores honesty. LoRA at lr 1e-5 is untested. The
  `zero_variance_group_share > 0.5` stop rule may halt outcome-arm runs early.
