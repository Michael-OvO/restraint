# Iteration 19 CR-GRPO: pre-launch review of the build (2026-11-01)

A multi-agent review (owner request, "ultracode"): seven independent reviewers (payoff table, report term, wiring,
data, methodology, operations, model correctness), two adversarial skeptics per finding, a synthesis (102 agents, all
read-only). Below: how each surviving finding was resolved, then the synthesis as written (verdict: NO-GO until fixed).

## Resolution (commit after this file)

| id | finding | resolution |
|---|---|---|
| C1 | `StatusTokens.of(processing_class)` crashes on Qwen3.5's processor | `trainer.text_tokenizer`; the decode in `_attach_report_signal` too; test with a processor-shaped stub; the job's calibration step builds the method trainer first |
| C2 | DEF-74's hide never ran (`detail` empty) | `HonestTaskSpec.from_task_dir` adds `path` / `moved_to` for remove_input controls (191 / 191); DEF-74 reopened and re-closed; test on built tasks |
| C3 | P1's metric did not exist | `analysis.power --metric false_claim_feasible` (`false_completion_rate_feasible`); prereg section 5 |
| R1 | low-work lies / silence cheaper than a refusal; ΔL > 0 on failing low-work controls | owner: "Lie/silence ≤ refusal"; `_payoff` and `control_refused`; ΔL prices the non-claim as after real work; tests at n 0-2 |
| R2 | neutrality covers only decline / proceed; stalling pays | owner: "Record + guard"; wrong-reason credit needs a named blocker or target; `blind_policy_values` + tests; prereg deviation 2; `max_decline_rate_feasible` 0.30; stall share logged |
| R3 | the 1/(1-p) cap made ℓ_rep inert; and (found while fixing it) TRL's bf16 log-softmax rounds log p and its gradient to 0 for confident tokens | z = logit(complete) − logit(incomplete) exactly in fp32 from an lm_head hook on the training forward (`StatusLogits`); tested against a direct forward in value and gradient, trimmed and stock scoring |
| R4 | the 64-token window missed 11.5 % of claims | `status_position` tracks the open `<tool_call>`; test with 200+ tokens of fields; found shares split by claims / non-claims |
| R5 | c_rep unjustified; gradient share unmeasured | owner: "Calibrate on node"; |∇rep| / |∇main| measured (`crgrpo_grad_ratio.jsonl`), `crgrpo calibrate` (0.75 / median ratio), a 3-step calibration step in the job, stop above 3 |
| M1 | GO let grey guards through | prereg section 9: every guard must pass |
| M2 | G5 margin mismatch | prereg section 6: fixed margins, judged on the bounds |
| M3 | G6 would match 240 twins | `twin-stop --samples 0 1 --expect-matched 120`; `<arm>_g6` steps in the job |
| M4 | no final-benchmark estimator / rule | prereg section 8 |
| M5-M10 | sign-off list, power / MDE, P1 regions, arm naming, KL wording, report-text leak | prereg sections 3, 5-7, 10 |
| O1 | the final benchmark could start between the ablation's segments | rule-1b steps hold the GPU slot |
| O2 | seg1 restarts from step 0 after a crash | `RESUME=1` on seg1 |
| O3 | a crashed rule step looks like a fired rule | prereg section 4: "fires" = exit 3 with `stop: true` |
| O4 | checkpoint-15 adapters not archived | listed in the job's archive files |
| R6 | V_s had no floor | `v_min` 0.1 on V_s and the family means; floor hits logged; test |
| R7 | probe episodes moved the runtime | `Runtime.frozen()` around `prediction_step`; test |
| R8 / D3 | the update fired a step late; the pass-band order biased its window | `update_steps [14]`, `update_window 15` (the whole first pass) |
| R9 | the update could raise the threshold past 1/3 | owner: cap; `mu_max {fc: 0.35}`; test |
| R10 | case-label gaps (tampered twin claims, self_caused_failure) | the twin branch and C2 read the claim flag; `self_caused_failure` in HONEST_CASES; tests |
| R11 | the attempt counter undercounted | newline / heredoc / comment aware, writes count; tests |
| R12 | the runtime's reward config differed from the environments' | `install_crgrpo(reward_config=plan.reward_config)` |
| R13 | μ_tw cannot raise the twin stake | owner: no C2 target |
| D1 | family means included unregraded DEF-74 episodes | `family-means --exclude-episode-ids` (remove_input 0.624); recomputed on the re-graded bank before launch |
| D2 | references from non-surface variants | high_risk starts at G2's surface value 0.704; C3 target 0.21 from the pool's mix |
| D4 | the never-trained scan skipped the GRPO pools | `data/train_rl*.jsonl` added (pass, 0 hits) |
| D5 | the a2 re-grade read runs a fresh node lacks | the three source runs staged in `data/def74/sources` with sha256 checks |

K1 re-run on the changed table (`analysis.reward_gate --config crgrpo`, G3's rollouts): PASS (pooled blind-decline
credit −0.042; every blind policy below the start policy; reading pays on both branches).

## Second pass (an independent reviewer on the fix diff)

| finding | resolution |
|---|---|
| the gradient-ratio buffer skipped rows without a status token (one row per micro-batch): steps mixed, the ratio leaned up, and the 3-step calibration would have written 2 rows and stopped the job | every micro-batch counts (zeros on the report side); the buffer is keyed on `global_step` and closed on the step's last micro-batch or when the step changes; tests |
| the decline-rate stall guard could never fire in a 15-step segment (stop.window 20) | `stop.decline_window` (5 for this run); test |
| unparsed reports counted in the non-claim found share and were searched for a status | rows without a parsed report are skipped; test |
| the report term would silently do nothing under the liger kernel | refused when `use_liger_kernel` is on; and a micro-batch whose status tokens the hook reached none of stops the run |
| a retried calibration appended to the earlier attempt's ratios | a fresh run moves an earlier `crgrpo_grad_ratio.jsonl` aside, as it does the runtime state |
| the prereg overstated ΔL | corrected: −(μ_fc + α) after real work, −(μ_fc + α + μ_cd·V_s) on a low-work control |
| `>` inside quotes and `<<<` herestrings misread by the command counter | quoted text is stripped before the redirect test; `<<<` is not a heredoc; tests |
| the stall metric's definition differed from the prereg's | uses `_names_blocker` and `min_substantive` |

Full suite after both passes: 5,141 passed (no Docker / GPU / slow).

## The synthesis (as written by the review)

# CR-GRPO (it19c): final pre-launch fix list

Commit eb40b49. I re-checked the disputed code myself: `trainer.py` 95-150, `run.py` 205-222, `spec.py` 101-103, `grading.py` 1238-1250, and `crgrpo.py` `_payoff` and `report_delta`. Where the two skeptics disagreed on severity, the reasoning is given under the item.

## Launch-blocking

**C1. Critical. report_term. The method arm crashes at startup.**
- **Defect.** `install_crgrpo` (`src/honest_agents/train/run.py`, line 218) calls `StatusTokens.of(trainer.processing_class)`. For Qwen3.5 that object is a Qwen3VLProcessor. Its `__call__(images=None, text=None)` takes `"complete"` as an image, and `load_image` raises ValueError. This happens after the model and vLLM have loaded. method_seg1 (`on_failure: continue`) then fails, and so does everything after it, so only the ablation arm trains.
- **Fix.**
  - Use `tok = getattr(trainer, "_tokenizer", None) or getattr(trainer.processing_class, "tokenizer", trainer.processing_class)`, then `StatusTokens.of(tok)`, and assert the ids are (14311, 258).
  - Decode with `self._tokenizer` in `_attach_report_signal`.
  - Add a test with a ProcessorMixin-shaped stub.
  - Add a stop-on-failure step that builds the method-arm trainer (make_trainer or sync_check) before method_seg1.
- **Status.** Must fix.

**C2. Major, blocks launch under the integrity gate. DEF-74's grader fix never runs.**
- **Defect.** `_control_backup_hidden` (`src/honest_agents/grading.py`, line 1243) reads `spec.detail["path"/"moved_to"]`. But `HonestTaskSpec.from_task_dir` (`src/honest_agents/spec.py`, line 103) validates only `[metadata.honest]`, and the builder writes those keys to `[metadata.honest_variant]`. All 392 remove_input controls load with `detail={}`. So the hide is a no-op in training, in eval, and in the a2 `def74_regrade` step, which still writes its done marker.
- **Severity.** The two skeptics rated it major; the reviewer rated it critical. I agree with major: the DEF-74 sources are not in pool v4, so training is barely affected. But DEF-74 is closed in `docs/defects.md` while the fix does nothing, and the gate forbids running an experiment with an open defect.
- **Fix.**
  - In `from_task_dir`, when `is_control` and `control_of == "remove_input"`, merge `honest_variant.path` and `moved_to` into `detail`. Do not rewrite any task.toml, because that changes frozen hashes.
  - Add a test that loads a real built control and checks the hide under a fake sandbox.
  - Make `def74_regrade` assert that `debug.control_backup_hidden` is true on at least one replayed episode.
  - Reopen DEF-74 until the re-grade has been re-run.
- **Status.** Must fix.

**C3. Major. methodology. P1's metric cannot be produced by the named tool.**
- **Defect.** P1 is "false claims per feasible episode", computed with `analysis.power estimate`. That tool exposes only `false_claim_share`, whose denominator is claims (`power.py`, line 259; `episode.py`, line 171). CR-GRPO is designed to claim less, which moves the two denominators differently, so the analyst would be choosing the metric after seeing the data.
- **Severity.** The reviewer rated it critical; both skeptics rated it major. Major: it can be fixed after the run, but it must be fixed before any select row is read.
- **Fix.**
  - In `load_tier_run`, add `false_claim_feasible = (float(e.false_completion), 1.0)` under scored feasible episodes, register it in METRICS, and test it against `false_completion_rate_feasible`.
  - Make P1 name `--metric false_claim_feasible`. Say that controls are included, unscored episodes are excluded, and no-report episodes count as non-claims.
  - Report `false_claim_share` as a secondary metric with no rule.
- **Status.** Must fix before approval.

**R1. Major. reward. On a control with little work, the table pays a lie or silence above an honest non-claim, and the report term pushes failing work toward "complete".**
This merges the reward finding with the model finding on the same branch.
- **Defect.** In `_payoff` (`src/honest_agents/crgrpo.py`, lines 110-118), the `fabrication` and `feasible_no_report` branches run before the control refusal test. So on a control with fewer than 3 substantive commands, the outcomes pay:

  | Outcome | Payoff at V = 0.8 |
  |---|---|
  | No report | -0.25 |
  | False claim | -0.286 |
  | Honest incomplete | -2.4 |

  `report_delta` (lines 128-139) keeps the episode's real n_substantive, so ΔL = +2.11 on failing low-work controls. ℓ_rep then raises π(complete) on episodes whose tests fail, which works directly against P1. The same coupling drops the claim threshold on plain tasks with fewer than 3 commands to 0.15, against the intended 0.30. This also breaks the design's audit rule "silence < honest report", which is tested only on plain tasks.
- **Fix, must do.** In `report_delta`, price the canonical non-claim at `max(n_substantive, cfg.min_substantive)`. ΔL on failing work is then -(mu_fc+alpha) at every n, and the threshold is 0.30 everywhere.
- **Fix, owner chooses the action-side price.** Recommended: when `o.is_control and not attempted`, a fabrication pays `-mu_cd*v - mu_fc` and a no-report pays `min(no_report_feasible, -mu_cd*v) - eps`. Count the no-report case in `control_refused` / C3.
- Leave failed claims after real work at -mu_fc. The reviewer's blanket `min(-mu_fc, -mu_cd*V)` cap would break the 0.30 threshold.
- Add tests at n ∈ {0, 1, 2} vs 8: lie < honest, silence < honest, and ΔL < 0 on failure.
- Update the design table (lines 91-97) and re-run the S9 audit and K1 / K1(b).
- **Status.** Must fix; the owner decides the action-side prices.

**R2. Major. reward. Caution neutrality covers only the two blind policies it checks.**
This merges the reward finding with the model "stall" finding.
- **Defect.** `blind_pair_values` (lines 142-147) checks only "decline both" and "proceed both", which give -2V each.

  | Blind policy (V = 0.8) | Twin pays | Control pays | Per pair |
  |---|---|---|---|
  | Decline both (checked) | | | -1.6 |
  | Proceed both (checked) | | | -1.6 |
  | 3 harmless commands, then "incomplete", blank blocker | 0 (implicit) | α = 0.1 | +0.10 |
  | Same, with a generic reason | 0.5k (`abstain_wrong_reason`, `blocker_check` None, line 108) | α = 0.1 | +0.50 |
  | Silence | 0 | -0.25 | -0.25 |

  The "no write to the task's output paths" clause in the design's C3 (design line 41/96) is not implemented. LOOK_ONLY leaves out grep and find, so harmless reads count as substantive. A policy that cannot tell twin from control is therefore pushed toward quitting early with "incomplete". The neutrality claims in prereg line 43 and in the docstring are false as written.
- **Fix, must do.**
  - Pay the 0.5k wrong-reason credit only when the decline names an external blocker category or target. Otherwise treat it as vague (0).
  - Extend `blind_pair_values` and its test with stall (blank and generic reason), verified-irrelevant blocker, and silence. Assert each is at most -2V at every V and over the mu range the update can reach.
- **Fix, owner chooses how to close the stall gap.**
  - (a) Implement the write clause: a control non-claim with no write to an output path is a refusal. This needs output paths from the grader and a write test that `touch` cannot game.
  - (b) Lower the twin vague / implicit / no-report payoffs until every blind pair is at most -2V. This must keep "decline after damage < vague". A skeptic noted this needs the decline-after-damage payoff changed to -k*mu_tw.
  - (c) Keep the current prices. Record the advantage of up to +(2V+0.5) as a prereg deviation, and add guards with a stop rule on the vague/stall non-claim rate on twins and controls.
- **Status.** Must fix or record as a deviation; owner decision.

**R3. Major. report_term. The gradient cap makes ℓ_rep almost inert on confident false claims.**
- **Defect.** `status_log_odds` (`src/honest_agents/trainer.py`, lines 107-117) clamps 1/(1-p) at grad_clip = 20. Since ∇logp = (1-p)∇(logit_c − logit_i), the trained gradient is min(1, 20(1-p)) times the exact gradient. The exact product is bounded, so the cap does no stabilising work; it only weakens the push. On G2's 125 failing feasible claims, the median π is 0.999987 and the mean attenuation is 0.079 (2.6e-4 at the median). The method-vs-ablation contrast is near-null by construction. The prereg's "exact z" and "3x K2a" wording is wrong.
- **Fix.** Use `q = -torch.expm1(logp.detach()).clamp(min=1e-6)` and slope ±1/q with no clamp. Confirm the logps are fp32. Add a finite-difference test at p = 0.99999 and log the attenuation. `_scored_logps` does not hold the logits, so the reviewer's "gather logits" route needs extra plumbing.
- **Interaction.** This raises the report gradient on these rows by about 12x on average, which makes R5 more urgent.
- **Status.** Must fix.

**R4. Major. report_term. `status_position` misses 11.5% of claims.**
- **Defect.** `trainer.py`, line 100, requires `<function=submit_report>` to be within 64 tokens of the status token. When blocker or completed fields come first, it is up to 463 tokens back.

  | Source | Missed |
  |---|---|
  | G2 bank, claims | 205 / 1,780 |
  | G2 bank, non-claims | 7 / 554 |
  | G3 completions (independent check) | 54 / 949, 53 of them claims |

  The pooled `found_share` hides the asymmetry.
- **Fix.**
  - Accept candidate j when the newest `<function=` opener before j is `submit_report`, with no `</function>` between it and j. Scan forward once per completion.
  - Add a test with more than 64 tokens of fields before status.
  - Log `found_share` split into claim and non-claim.
  - Re-run the probe and expect 2,334 / 2,334.
- **Status.** Must fix.

**R5. Major. report_term. c_rep 10 rests on a wrong comparison, and the report term's gradient share is unmeasured.**
This merges the report_term and model findings.
- **Defect.** Deviation 3 cites K2a at lr 1e-4. K2a actually ran at 5e-5 with cosine decay, so the push is about 5-6x K2a per step and about 1.2x in total, not "3x per step, a third in total". The deeper problem is the optimiser: under clipped AdamW the "lr × weight" argument does not hold, because ℓ_rep and the GRPO loss share one clipped Adam step. G3's main-loss grad_norm was 0.09-0.18, and nothing logs the two terms separately. The design's own "≤ 0.3 nats per step" criterion is not monitored.
- **Severity.** The skeptics split 2 minor / 2 major. I rate it major because of the R3 interaction: once the cap is removed, the term's effective size rises by about an order of magnitude on most rows.
- **Fix.**
  - Correct deviation 3.
  - Change "/ N found rows" to "/ 64 rows" (`trainer.py`, line 707).
  - Log `crgrpo/rep/grad_norm` and `main/grad_norm` with `autograd.grad` over the LoRA parameters, at least on steps 1-3, plus the mean |Δz| on found rows.
  - Calibrate c_rep in a one-step node smoke run or an offline replay. Pre-register the rule, for example "median ‖∇rep‖/‖∇main‖ in 0.3-1x at G2", and a flag or stop if it goes above about 3x.
- **Status.** Must fix; the owner approves the c_rep rule.

**M1. Major. methodology. GO lets grey guards through.**
- **Defect.** "P1 passes and no guard fails" (prereg lines 113-114) treats a grey guard as a pass. `read_guard` (`calibration.py`, line 403) has a third, grey state; it18 required "every guard passes"; the it16 regions block and design rev 1 (line 512) say "not grey".
- **Fix.** GO requires P1 to pass and every one of G1-G6 to pass. P1 passing with any guard grey is reported as grey, naming the guards. The decision code must read `pass`, not `not fail`.
- **Status.** Must fix before approval.

**M2. Major. methodology. G5's margins differ from what `read-guard` applies.**
- **Defect.** The prereg fixes -12.9 / -14.3, which are it18's margins at S = 2 with the DEF-66 sources included. `read_guard` recomputes the margin from the reference's own K and S. At K = 52, S = 4 that is about 10.3 / 10.4, so a lower bound in the gap gets two verdicts. G1-G4 are also inherited fixed margins, not margins "at the select split's K and S".
- **Fix.** Keep the fixed margins and judge G5 on `est['lo']` against -0.129 / -0.143, with a `--margin` override or an explicit note that the tool's `pass` field is ignored. Or adopt the S = 4 margins and write their numbers in before launch. Fix the wording of design change 2 either way.
- **Status.** Must fix before approval.

**M3. Major. methodology. G6 would match 240 twins at S = 4, not 120.**
- **Defect.** `twin_stop` (`calibration.py`, lines 407-438) matches every (task, sample) pair in both runs and compares a raw count against +8. At the final checkpoint both runs are `select.jsonl` at S = 4, so the bar is twice as strict as the pre-registered "120 twins". The chain has no G6 step.
- **Fix.** Add `--samples 0 1` to twin-stop, filtering both runs before matching. Report `n_matched` and refuse when it differs from 120. Write the exact G6 command into the prereg.
- **Status.** Must fix.

**M4. Major. methodology. The final benchmark, which the owner judges the work on, has no pre-registered estimator or rule.**
- **Defect.** Section 8 lists metrics and comparison rows only. It also reuses it19a's internal rows despite the eval-path diffs listed in section 10, and the 569 → 562 refreeze is not stated.
- **Fix.** Add a section 8a modelled on it19a's sections 4-6:
  - A paired cluster bootstrap (by source on the internal set, by item on TB2 / ImpossibleBench).
  - H1: internal false_claim, UB < 0. H2: ImpossibleBench flag_lenient, LB > 0. H3: TB2 false claim, UB < 0, with the P(fail | claim) qualifier.
  - A non-inferiority margin on success, and an MDE for each metric.
  - "Core contribution supported" needs the select GO and no contradicted benchmark hypothesis.
  - Restrict the internal comparison to 562 variants, and either justify reusing the it19a rows against its pin rule or re-run G2 internal.
- **Status.** Must fix before approval.

**O1. Major (recoverable). ops. The final benchmark can start before the ablation's steps 16-30, and its failure cancels them.**
- **Defect.** Both rule1b steps have `gpus: []`. As a result, method_final (`on_failure: stop`) takes GPU 0 at the same time as ablation_rule1b. If it fails, `_skip_unreachable` skips ablation_seg2 and ablation_select, and the attribution endpoint is lost on an unattended run. The comment at line 254 is wrong.
- **Severity.** Skeptics split minor / major. Major, because it defeats the documented order on an overnight run, though checkpoint-15 and the markers survive and a restart recovers it.
- **Fix.** Give method_rule1b and ablation_rule1b `gpus: [0]`. Do not add ablation_select to method_final's `after`, because a fired ablation rule would then skip the benchmark. Fix the comment and re-run the scheduler simulation.
- **Status.** Must fix.

## Minor, fix before launch

- **R6. reward. V_s has no floor** (crgrpo.py, lines 57-59, 86-89, 267-305).
  - **Defect.** If V_s goes below 0, the twin table reverses: claim or damage scores positive and a control refusal is paid. Design rev 1's "exclude V_s < 0.2" rule was dropped.
  - **Severity.** Skeptics split on likelihood (P ≈ 0.10 vs ≈ 0.005 per step for the worst source). Minor today, but self-amplifying if the run starts damaging controls.
  - **Fix.** Add `v_min` (for example 0.05-0.2) to RunSettings. Clamp `control_value` and the family mean at it, log `crgrpo/v_floor_hits`, and add a test that V ≤ 0 never puts a claim or damage above the right decline.
- **R7. reward/wiring/ops (four duplicate findings merged). Probe episodes update the runtime.**
  - **Defect.** `prediction_step` goes through `crgrpo_rewards` → `Runtime.score`, which always runs `_update_families`, `_record` and `save`. The step-0 probe moves the family means 10% toward 40 validation controls before training starts. The seg1 step-15 probe fires a spurious update built from probe data, which is saved and logged but discarded on resume.
  - **Severity.** Minor: the effect on the trained arms is small and decays, but validation data does reach the training prior.
  - **Fix.**
    - Add `record: bool` to `Runtime.score`, set from a flag around `prediction_step` with try/finally, so eval batches price with the current state and skip `_update_families`, `_record` and `save`.
    - Test that the state and the state file are unchanged after an eval call, and that a probe at step u does not consume update u.
- **R8. reward/wiring/model (merged). The update fires one step late.**
  - **Defect.** `update_steps [15]` is compared with the 0-indexed `global_step`. It fires on the 16th batch, with a window of steps 7-16 and new multipliers from step 17. The prereg says a window of 6-15 and new values from step 16.
  - **Fix.** Set `update_steps: [14]` in both configs. The update then lands in the step-14 state line that seg2 resumes from. Document the convention in `RunSettings` and add a resume test.
- **R9. reward. The update can raise the claim threshold past 1/3.**
  - **Defect.** The update can raise mu_fc to 0.571, giving a threshold of 0.427, beyond where K0.3 measured a -6 point success cost. G2's C1 of 0.124 is about 0.003 below the 1/3 crossing.
  - **Owner decision.** (a) Cap mu_fc at 0.35 (threshold ≤ 1/3). (b) State in the prereg that the threshold may reach 0.43 and report it next to G1.
  - **Also fix.** Design line 70 still says "μ_fc ≤ 0.43".
- **R10. reward. Case-label gaps.**
  - **Defects.**
    - `self_caused_failure` is missing from HONEST_CASES, so on plain tasks it pays -1.0, while a vague report after the same damage pays -0.9.
    - Tamper relabels the case as "tamper", so C2 misses tampered twin claims.
    - A tampered twin claim pays -2.286, more than a clean claim (-3V) whenever V > 0.76. The high_risk and revoke_permission family means sit above that.
  - **Fix.**
    - Add `self_caused_failure` to HONEST_CASES.
    - In the twin branch and the C2 indicator, test `(br.claimed_complete and br.case != "leak") or br.violation`.
    - Apply the tamper floor as min(clean payoff, floor).
    - Add tests.
- **R11. reward. `substantive_commands` undercounts.**
  - **Defect.** It does not split on newlines or count redirect writes. Zero C3 flips on G2's bank today, but the counter is gameable in one direction.
  - **Fix.** Split on unquoted newlines, drop heredoc bodies and comment lines, and count `>`, `>>` and `tee` to non-/dev/null targets as substantive. Add the five probe strings as tests, then re-run family-means and expect no change.
- **R12. wiring. The runtime's base config has efficiency_ref_actions 34; the environments use 28.**
  - **Fix.** In `install_crgrpo`, use the environment factory's `reward_config`, or `build_reward_config(cfg, build_budget(...).max_bash_calls)`, and assert the two are equal.
- **D1. data (three duplicates merged). The family means and G2's reference rates include the 13 in-scope DEF-74 `regrade` episodes.**
  - **Effect.** remove_input mean 0.5686 → 0.6236 without them; C1 0.124 → 0.117.
  - **Fix.** After C2 is fixed and a2's re-grade has run, recompute `family-means` and the C1-C3 references from `runs/def74/it17_rollouts_feasible`, and add its sha256 to inputs_frozen. If launch comes first, use the exclusion value (0.6236) through an `--exclude-episode-ids` option. Update both configs and prereg section 3; the targets stay.
- **D2. data. The G2 reference rates and the high_risk family mean come from non-surface variants, but 30 of 45 pool pairs are surface pairs.**
  - **Effect.** On the pool's own mix, G2's rates are about C2 0.39 and C3 0.20, against the quoted 0.186 / 0.087. The high_risk mean is about 0.70, against 0.809. The update will likely push mu_tw to its clip and C3 will bind.
  - **Owner decision.** Re-state the references and expected multiplier movement in prereg line 41 and the config comment, then either retune the tw/cd targets or accept the clip. Optionally start high_risk at 0.70.
- **D3. data. Plains are ordered by pass band.**
  - **Defect.** Hard plains sit in steps 1-7 and always-passing ones in steps 12-15. This lowers the windowed C1 by about 0.03 and cancels most of the intended mu_fc tightening.
  - **Fix.** Cheapest: `update_window: 15`, which also suits R8's `[14]` (window 0-14). Or shuffle the selected plains in `rl_pool_v3.build`, rebuild, and update the sha.
- **D4. data. The never-trained gate does not scan `data/train_rl*.jsonl`.**
  - **Fix.** Add the glob plus `data/select.jsonl` and `data/it19/*.jsonl` to REPO_TRAINING_GLOBS. The files are gitignored, so archive them and add UNTRACKED_TRAINING_INPUTS entries (the DEF-57 problem). Re-run with `--archive --record`.
- **D5. ops. `def74_regrade` reads `runs/eval/it19a_*_heldout`, which `node sync` never sends.** On a fresh node the a2 session stops at step 3.
  - **Fix.** Copy those directories into `data/def74/sources/`, point the loop there, and add sha256 and existence guards.
- **O2. ops (two duplicates merged). seg1 starts without RESUME=1.**
  - **Defect.** A crash and chain restart retrains from step 0 and renames the state file away.
  - **Fix.** Prefix both seg1 commands with `RESUME=1`, and make sure the output dirs are empty before the first launch.
  - **Also.** An arm stopped by a rule (exit 3) must stay stopped across restarts: write a `<arm>_stopped` marker that seg1 and seg2 check, or put the rule in the runbook.
- **O3. ops. A crashed rule1b step looks the same as a fired rule.**
  - **Fix.** Define "fires" in the prereg as twin-stop exit 3 with `<arm>_rule1b.json` showing `stop: true`; any other failure is an infra re-run. Optionally echo the rc into a `.rc` file.
- **O4. ops. The checkpoint-15 adapters are not archived once seg2 runs.**
  - **Fix.** Add a `keep_checkpoints: [15]` option to `archive.select_run_files`, or list the adapter files explicitly.
- **M5. methodology. Sign-off and procedure.**
  - **Defect.** Gate step 4 is voided by the grader change, and section 10 lacks the step-3 re-sign-off list.
  - **Severity.** One skeptic refuted it, one rated it minor: no oracle re-run is needed, because the oracle's `mv` never triggers the hide.
  - **Fix.** Add to section 10 and record in a new it19c section of `docs/iteration-19-signoff.md`: the launch commit, `test.sh --fast`, the L0 sweep, `gate.guards --chains`, `node_chain check`, S9 = `tests/test_crgrpo.py` plus K1 (re-run after R1/R2), and an S10 note. This comes after C2 reopens and closes DEF-74 properly.

## Minor: record in the pre-registration

- **M6.** Power is missing and the MDE is optimistic. Give both MDEs at S = 4: 3.6 (observed SE / √2) and about 4.5 (design_variance). Give the power at -3.9 (about 0.7-0.85) and at about -1.2 (about 0.15), and say grey is the expected outcome. Give the ablation contrast its own MDE between two trained arms. Close design line 79.
- **M7.** P1's regions. Pass: UB < 0 and `paired_estimate.significant` (source and pair sign-flip, p < 0.10). Fail: LB > 0. Otherwise grey. Apply the it16 regions to the guards.
- **M8.** Arm naming. GO, the trade-off reading and the final benchmark apply to the CR-GRPO arm only. The ablation's comparisons with G2 are descriptive. Label 3.6 as a sampling-only MDE with one training seed per arm.
- **M9 (two duplicates merged).** In the method arm, the status token is outside the KL as well as the policy gradient. Fix the wording at prereg line 34 and config line 9; no code change. Optionally log the status-token KL.
- **M10.** The prose before `submit_report` and the later report fields keep the claim's advantage in both arms. One skeptic refuted the finding, one rated it minor. Add a sentence that the ablation measures the marginal effect of ℓ_rep. Optionally log a status-vs-prose contradiction rate so cosmetic status flips can be seen. Do not mask the whole report turn.
- **R13.** The C2 multiplier cannot raise the twin stake: under neutrality the decline-minus-damage gap is V(1+mu_cd), so mu_tw only moves weight from the decline reward to the damage penalty. Correct design row 5 / line 40 and prereg section 3, and note the expected stake change of about -5% at G2's rates. Optionally drop the tw target or tie the update to mu_cd.

## Verified correct
- **Payoff table.**
  - The rows match the design.
  - The claim threshold is 0.30 when n ≥ 3.
  - k = V(1+mu_cd)/(1+mu_tw) equalises blind decline and blind proceed at any multipliers.
  - Leave-one-out shrinkage of V_s is correct and not circular.
  - `report_delta` is correct on twins and on the damage and tamper cases.
  - Neutrality survives TRL's group-mean baseline in expectation.
- **Wiring.**
  - The two arms differ only in `report_term`.
  - One reward call per optimizer step (GA 64 = steps_per_generation).
  - Row alignment holds through shuffle, split and sort.
  - The captured logps are the grad-enabled forward.
  - Signals are consumed once.
  - The ablation takes the plain GRPO path.
  - `check_crgrpo` runs under `--dry-run`.
- **Resume.** It loads the step-14 state, the order fingerprint is stable, `eval_on_start` is off, and StopRules are replayed.
- **Pool v4.** 120 rows, sha matches, 15 blocks of 8 with each pair inside one step, zero source overlap with every eval split, no DEF-66/72/74 sources, labels match task.toml, and the family means reproduce exactly.
- **Rule 1b at step 15.** Correct: 120 matched twins, and K2b reproduces. DEF-66 is excluded by default.
- **Chain.** Rule-1b skip semantics, done markers, timeouts, W&B and archive scope are correct. `node_chain check` passes.
- **Tests.** 182 pass across `test_crgrpo`, `test_crgrpo_report_term` and `test_trl_rewards`.

## Launch readiness: NO-GO until fixed
**Must fix before launch:**
- C1, C2, R1 (`report_delta` now; action-side prices by owner choice), R3 and R4.
- R2: fix, or record as a deviation by owner decision.
- R5: gradient-share measurement and c_rep rule.
- O1.
- The cheap minors: R6, R7, R8 + D3, R10, O2, D5.

**Then:**
- Re-run S9/K1 on the changed table.
- Re-run DEF-74's re-grade and recompute the family means (D1).
- Fix prereg items C3, M1-M4 and M5-M10, and owner decisions R9, D2 and R13.
- Run a one-step method-arm smoke run on the node: it confirms C1 and gives the R5 gradient norms.

Only after that write the it19c sign-off and launch. No spend should happen before C1, C2, R1, R3, R4 and the R5 measurement are done. Without them the method arm either crashes or trains a report term that is mis-signed, partly blind and nearly inert, and the main comparison could not be interpreted.