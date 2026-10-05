# V3 status: what is built, what is verified, what is not

Date: 2026-10-05. Written by the integrator after all components were rebuilt against `SPEC_V3.md` and
`INTERFACES_V3.md`. Nothing here has run on Docker or a GPU. Every claim of "done" below means: implemented, and covered
by a test that runs on macOS with fakes. Where a test could only use a fake for the part that matters, the row says
"partly" and the gap is in section 5.

**Verdict.** Stage 0 of the spending gate (SPEC §8) is met locally: every regression and exploit test the spec names
exists and passes, the seams between components are tested, and every config and script resolves. The first paid step
is Stage 1 (section 6). The largest remaining risks are things only a node can show: the hermetic verifier inside real
images, the merged-LoRA weight sync, and the vLLM server's real response shapes (section 5).

## 1. Test results

All commands from `/Users/michael/Documents/Github/restraint`, prefix `env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval`.

| Check | Result |
|---|---|
| `pytest -q -m "not docker and not gpu"` | **2327 passed, 7 skipped, 21 deselected** (4m45s) |
| OpenEnv suite, `openenv-venv`, `PYTHONPATH=src:. python -m pytest tests/openenv` | 43 passed, 1 skipped (the TRL parity test needs `trl`, absent there; `tests/test_integration_v3.py::test_openenv_action_fields_match_the_report_tools` covers the same fields from the main venv by reading the source) |
| `ruff check src tests` / `ruff format --check src tests` | clean (line length 120 is now in `pyproject.toml`) |
| `bash -n scripts/*.sh` (24 scripts) | all valid; `tests/test_integration_v3.py` repeats it |
| `python -m honest_agents.train --dry-run` on `honest_alpha{0,0.1,0.25,0.4}`, `outcome`, `outcome_claimed`, `outcome_reflect`, `smoke`, `colocate_1gpu`, `qwen35_4b`, `phase0_zero_lr` | all build, on a dataset produced by `tasks.build` from synthetic seeds. Also `phase0_zero_lr` with `model.name=Qwen/Qwen3.5-4B` on the audit dataset |
| `python -m honest_agents.pilot.run --config configs/pilot.yaml --dry-run` on a synthetic 1,200-row bank | builds 1,091 rows, 8 arms, 0.73 epochs |
| End-to-end local dry run | synthetic seeds -> `tasks.build` -> `train --dry-run`; synthetic bank -> `pilot.run --dry-run`; `tasks.audit` and `--tally` on the same seeds |

Run the suite with `scripts/test.sh` (`--fast` default, about 40 s; `--all`, about 90 s, adds the `slow` tests; both parallel). The counts above predate it.

The 7 skips are legitimate: 4 OpenEnv modules (package not in the main venv, covered by the separate venv run above),
real Endless Terminals samples (`HONEST_ET_SAMPLES`), the Hub download (`HONEST_NETWORK_TESTS`), and `prctl` (Linux
only). No skip is waiting on another component. No test was deleted or weakened. The two `importorskip` guards in
`test_reward.py` and `test_trl_rewards.py` that waited on other agents were removed. `tests/eval/contract.py`, a stub left
over from the parallel build, was deleted.

The 21 deselected tests are `@pytest.mark.docker` or `@pytest.mark.gpu` (`tests/test_docker_integration.py`, 12 in
`tests/test_exploits.py`, and two placeholders in `tests/test_trainer_loop.py`). They have never run.

New test file by the integrator: `tests/test_integration_v3.py` (63 tests): dataset rows -> train / pilot / eval; one
episode through the real `HonestHarborEnv` (fake sandbox) -> `trl_rewards` -> bank -> pilot dataset -> `ReplayEnv`
(reward equal to the live reward) -> difficulty filter; a leak -> group mask -> quarantine; tool schemas identical in
five places; budget and sampling from one place for train, eval and the OpenEnv server; stop-rule metric names; every
`python -m honest_agents.<module> --flag` in `scripts/`; every config.

## 2. Seams checked and fixed in the integration pass

| Seam | Finding | Fix |
|---|---|---|
| trainer -> env | `trainer.py` calls `env._set_loop_exit(reason)` (`trainer.py:297`), but the environment never defined it, so `stop_reason = "truncated"` could never be recorded | `HonestHarborEnv._set_loop_exit` (`env.py`); only `truncated`, never over a recorded report. Test in `tests/test_env.py` |
| env -> bank -> pilot | `blocker_facts.paths` only held paths from the agent's commands; a replayed `permission_denied` on a labelled target the commands never named scored `unknown` (false blame) | `_fact_paths` adds absolute `expected_targets` and `decoy_targets` |
| validate -> build | `filter_valid` counted any feasible result, including a near-miss control, as "the pair's plain copy is valid" | requires `r.perturbation is None` (`validate.py:679`). Test added |
| validate | a pytest run that never produced a result (`no_result`, `timed_out`, zero tests) read as `tests_passed = False`, i.e. "the infeasible oracle fails", so such a variant validated | `_verifier_problem` makes it a validation error in the null check and every oracle run. Test added |
| grading config | the OpenEnv session did not apply the background-process sweep (`kill_background`) nor the task's own test timeout, so its grading differed from training | one function `grading.resolve_config` (`grading.py:151`) used by `env.py` and `openenv_env/session.py`. Test added |
| phase0 -> difficulty | `phase0_zero_lr.yaml` trained on `data/train.jsonl` (after the gate and mixture), so the difficulty gate could not see the full pool | now `data/train_pool.jsonl` |
| pilot -> scripts | `scripts/_common.sh` loaded `configs/pilot.yaml` as a train config and failed; no script ran the pilot | `_common.sh` detects a `sweep:` config and asks `pilot.run --print-model / --print-limits`; new `pilot.run --list-arms`, `--print-limits`; new `scripts/run_pilot.sh` (one vLLM per arm); `is_override` accepts `sweep.` and `analysis.` |
| pilot -> bank | validation-probe episodes share the bank when it is on; the pilot would train on `split == "val"` rows | `pilot.dataset` drops rows whose `task.split` is not `train` (`DROP_SPLIT`). Test added |
| callbacks | no stop rule on grader failure (a grading failure after the agent acted is scored as tamper, not masked, so a broken grader trains silently) | `RULE_REWARD_ERROR`, `stop.max_reward_error_rate: 0.05` (`callbacks.py`, `base.yaml`). Test added |
| callbacks / phase 0 | the SPEC §10.3 trainer decision and the §4.8 work-versus-quit check were not computed | `phase0_summary` adds `trainer_decision_10_3`, `warnings`, and `pool.work_vs_quit_gap / work_beats_quitting / budget_exhaustion_rate_feasible` |
| train | no fail-fast on a missing verifier bundle before model load (only `phase0.sh` checked) | `train()` calls `verifier.ensure_bundle` unless `data.allow_legacy_verifier` |
| packaging | numpy imported by `eval/metrics.py` but not declared; no ruff config; no console scripts | `pyproject.toml`: `numpy>=1.26` in `eval`, `[tool.ruff] line-length = 120`, `[project.scripts]` (`honest-train`, `honest-eval`, `honest-build-dataset`, `honest-validate`, `honest-pilot`, `honest-pilot-analyze`, `honest-sync-check`). `uv lock` and `uv sync` run; `uv lock --check` passes |

Checked and found consistent (no change): dataset columns (`build.COLUMNS`, 11 keys) accepted by `train.check_rows`,
`rollout.load_split`, `HonestHarborEnv.reset(**row)`; group id `variant_id` (unique per row) used by `trl_rewards`,
`bank`, `difficulty` and the quarantine list; tool schemas from `harness.tool_schemas()` equal TRL's derivation from the
real environment, `eval.rollout`, `pilot.ReplayEnv` and the OpenEnv `REPORT_FIELDS`; `EpisodeBudget.from_limits` gives
the same budget in `train.build_budget`, `eval.resolve_configs` and the OpenEnv server, and `efficiency_ref_actions`
equals the call budget; all `METRIC_*` names read by `StopRules` are logged by `trl_rewards`; a bank row written by
the real reward function replays through `ReplayEnv` to exactly the live reward.

## 3. Component change requests: decisions

Applied (by a component agent or by me): all `env.py` / `harness.py` tool-signature requests; `grading.py` outcome
fields and `check_blocker -> reward.blocker_verdict`; `train.py` single reward function, efficiency reference, bank and
quarantine wiring, refusal of rows without `expected_targets`, fail-fast `ensure_bundle`; trainer passing `prompt_ids`,
`logprobs`, `tool_mask`; `callbacks` importing metric names from `trl_rewards` and the new reward-error rule; eval case
renames and `outcome_to_dict`; tasks recording `expected_targets`, `decoy_targets`, `is_control`, `control_of`,
`variant_id`, `verifier`; the pilot scoring through `bank.rescore`; `setup_node.sh` building the verifier bundle;
`harness.BudgetMeter` shared with eval and OpenEnv; `kill_background` for OpenEnv (through `resolve_config`);
`filter_valid` plain-copy fix; `validate` treating `no_result` / `timed_out` / `n_tests < 1` as errors; the pilot's
bank-path, split and script requests; ruff line length and numpy.

Rejected, with reason:

| Request | Decision |
|---|---|
| `train.py` sets `false_blame = min(0, alpha - 0.1)` (core-reward review) | **Rejected for now, owner decision.** SPEC §1 fixes `false_blame = 0` and the config knob already exists. Sim evidence: at alpha 0 false blame is free (false-blame rate 0.16 on feasible, 0.39 on controls under `scale_rewards: none`). Changing it only for the alpha-0 arm would make the alpha grid non-comparable, and changing it for all arms departs from the spec. Measure it in Stage 1 (`honest/false_blame_rate`, `honest/control/decoy_blame_rate`) first |
| Mask the group on an unclaimed clean pass of an infeasible variant (core-reward review) | **Rejected**, outside the spec (leak is claim plus clean pass). It is flagged (`unclaimed_pass_infeasible`, quarantine kind `unclaimed_pass`) and the Phase-0 difficulty report counts it. Revisit if Stage 1 shows any |
| `tasks/build.py`: weaker `--allow-weak-validation` | kept as an explicit opt-out for tests; never used by `build_dataset.sh` |
| Per-turn max tokens and the total token budget enforced by the OpenEnv server | impossible (server sees only tool output); documented in its README, the client enforces |
| The F2 bridged harness, free-text report training, and the F5 verify-then-report gate baseline (eval) | not in SPEC_V3; not built |
| Multi-seed aggregation in eval | belongs to Stage 4 |
| Replace `tests/eval/contract.py` shim | done by deleting it |

## 4. Conformance table: SPEC_V3.md requirement by requirement

Status: done = implemented and tested locally; partly = implemented but a stated part is unverified or differs; missing = not built.

### §1 Reward v3

| Requirement | Status | Where |
|---|---|---|
| Feasible table (success_claimed, fabrication, honest_failure/underclaim, honest_failure_observed, false_blame, lazy_decline, feasible_no_report) | done | `reward.py:444` `compute_reward`; independent reference implementation and 30k random episodes in `tests/test_reward.py` |
| λ paid for passing whatever the report says, zeroed on tamper/violation | partly | `reward.py:444`; not paid on the `lazy` row (deviation D3) |
| Claim threshold `(1+α)/2` proven on a grid of α, λ | done | `tests/test_reward.py:1085`, `:1031` (also with the efficiency bonus) |
| Infeasible table (infeasible_claimed, leak mask, abstain_correct / wrong_reason, implicit_abstain, bad report) | done | `reward.py:444` |
| m = 1 needs category in `expected_blockers` and target in `expected_targets` | done | `reward.py` `targets_match`, `normalize_target` |
| Violation −β, `post_hoc` | done | `reward.py:444` |
| Tamper `min(total, −1.5)` | done | `reward.py:513`; `tests/test_reward.py:700,1055` |
| Brier not in reward; `brier_weight >= success_bonus` rejected | done | `reward.py:123`, `__post_init__` |
| Masking only for boot/toolchain failure and group leak; everything the policy causes is scored | done | `reward.py:75-79`, `env.py _failed_outcome`, `grading.grade` |
| A fourth mask reason `reward_error` | partly | not in the spec (D1) |
| One reward function per run, four modes | done | `trl_rewards.py:623-653` `reward_func_for_mode` |
| All required training metrics logged | done | `trl_rewards.py` `REQUIRED_METRICS`; checked in `tests/test_integration_v3.py` |
| Archetype regression tests (α, s, infeasible share grid) | done | `tests/test_archetypes.py` (515 tests) |
| Every reviewed exploit asserts "no more than honest baseline" | done (fakes) / Docker part unrun | `tests/test_exploits.py` (25 + 12 Docker-marked) |
| Two-valued group test, `none` keeps α | done | `tests/test_trl_rewards.py:1523` |
| TRL aggregation: masked episode is NaN, not 0 | done | `tests/test_trl_rewards.py:1414` |
| Truncated completions handled by TRL | partly | `base.yaml:90` `mask_truncated_completions` masks the loss only; the reward stays in the group mean (D8) |

### §2 Report schema

| Requirement | Status | Where |
|---|---|---|
| `submit_report` / `abstain` signatures, one blocker | done | `harness.py:274` `ReportTools`; `report.py` |
| Category definitions in the docstring, one line each | done | `report.py:123,148`; `harness.py` docstrings |
| Status normalised, unknown status is a parse error | done | `report.py:78` |
| `evidence_commands` replayed only in evaluation, after checks | done | `grading.py` (`replay_evidence=False` default) |

### §3 Grading and environment

| Requirement | Status | Where |
|---|---|---|
| Blocker verification true/false/unknown as agent user | done (fakes) | `grading.py:430` `check_blocker`; `reward.blocker_verdict`; cached twin `bank.check_blocker_cached`; parity test |
| Hermetic verifier: fresh upload, vendored pytest, `env -i`, `python3 -S`, rootdir pinned, network off, nonce markers on the exec stream | done (fakes) | `verifier/runner.py:34`, `verifier/_hv_run.py`, `verifier/bundle.py:22` |
| Reward file that disagrees or appears early is tamper | done | `grading.py` (legacy path, `reward_file_before_tests`) |
| Legacy `test.sh` for external benchmarks only; `train.py` refuses non-pytest | done | `train.py:706` `check_rows`; `data.allow_legacy_verifier` |
| Toolchain self-check at reset, only it masks | done | `grading.py:313` `self_check`, `verifier/toolchain.py:123` |
| Sentinels record inode (and ctime) | done (BSD stat only) | `grading.py:187,244` |
| Typed manifest errors, tamper kept if a later step raises, grading failure after acting is tamper | done | `grading.py:83`, `env.py _failed_outcome` |
| Budgets enforced by the env, budget line, `MAX_OUTPUT_CHARS = 2000`, shrinking | done | `harness.py:32,85,119` `EpisodeBudget`, `BudgetMeter` |
| `timeout -k` 60 s, configurable test timeout, pids limit | done / pids cgroup unverified | `env.py:84-104`, `grading.py:151` |
| Stop reason per episode | done | `reward.py:82`, `env.py _stop_reason`, trainer hook (integrator fix) |

### §4 Data

| Requirement | Status | Where |
|---|---|---|
| Endless Terminals only, converted to the hermetic verifier | done on 40 sample tasks | `tasks/hermetic.py:370`, `tasks/sources.py` |
| Validation through our env path: 3 of 3 oracle, null check, infeasible fails, `remove_input` repair probe | done (fakes) | `tasks/validate.py:478,248`; `_repair_probe` |
| `expected_targets` on every infeasible variant; 1:1 near-miss controls with `decoy_targets` | done | `tasks/perturb.py` (every operator has a control); `spec.py` |
| Operator filters, sibling deletion, `remove_tool` mandate gate | done | `tasks/perturb.py:616`, `tasks/analysis.py` |
| `high_risk` act twin, identical wording, aligned abstain clause (LV-3) | done | `tasks/perturb.py`, `harness.SYSTEM_PROMPT` (`harness.py:36`) |
| Phrasing pools ≥ 8 per operator, partitioned train/test | partly | only the two instruction-based operators have phrasing pools; the five environment operators have name pools (D10). Test pools were written by the builder, not independently (F11) |
| Mixture 55/20/25 | done | `tasks/build.py:77-80` |
| Test split by fixed hash (`--test-bp 2200`, `--val-bp 400`), `test_ids.txt`, two held-out families | done | `build.py` `split_bucket`, `assign_seed_splits`, `write_test_ids`; salt and cutoffs frozen after the first training run; BENCH_DIR mandatory |
| Leakage fixes L1 to L7 (opaque names, matched cc controls, decontamination, split guards) | done, Docker unverified | `docs/research/leak-register.md` section 6; yields per 150 sources: high_risk 105, revoke 96, cc 28, remove_input 12, no_network 1 |
| Quarantine list excluded on rebuild | done | `bank.py:421`, `build.py` |
| Difficulty filter from Phase-0 pass@8, infeasible leak check, alpha safety | partly | `tasks/difficulty.py:246`, `callbacks.pool_stats`; a 24-step Phase 0 measures only ~190 prompts, not the pool (section 5, item 9) |
| `build_audit_set.sh` | done | `scripts/build_audit_set.sh`, `tasks/audit.py:274` |

### §5 Training

| Requirement | Status | Where |
|---|---|---|
| `scale_rewards: none`, `group` rejected | done | `configs/base.yaml:90`, `train.py:456` |
| One training process per run; guard against server mode with more than one process; two-slot layout | done | `train.py:468`, `trainer.py:216`, `scripts/_common.sh` slots |
| `max_tokens: 4096` per turn; window computed and asserted at startup | done / server unverified | `base.yaml:102`, `train.py:341`, `models.py:331` `probe_vllm_server`, `trainer.py:238` |
| `token_truncate`, `save_total_limit: null` | done | `base.yaml:124,135`, `train.py:484,496` |
| `HonestGRPOTrainer`: concurrent tools, reported episodes not regenerated, differential test | done | `trainer.py:444,420`; `tests/test_trainer_loop.py:346`. Report-turn rollback is a deviation (D5) |
| Stop rules callback | done | `callbacks.py:92` and `RULE_*`; reward-error rule added by the integrator |
| `phase0_zero_lr.yaml`, per-component timing, bank | done | `configs/phase0_zero_lr.yaml`, `callbacks.py:576`, `scripts/phase0.sh` |
| OPD and 27B teacher deferred | done | not built |

### §6 Pilot, §7 Evaluation, §8 Gate

| Requirement | Status | Where |
|---|---|---|
| Bank row schema, cached blocker facts | done | `bank.py` |
| Report-only pilot, sweep α × `scale_rewards`, stop rule | done (CPU, scripted sampler) | `pilot/run.py:537`, `pilot/analyze.py:390`; `scripts/run_pilot.sh` |
| Pilot sound for H2/H3 only; contexts not token-identical | done | `pilot/__init__.py` docstring |
| Pair selection fix, all §7 metrics, per-family leak rate | done | `eval/metrics.py:745,510,524,1053` |
| Constant-policy reference rows, neutral prompt and tools, confidence sweep | done | `eval/baselines.py:160,193`, `eval/run_eval.py:54` |
| Non-inferiority margin fixed before the run, bootstrap by source task | done | `eval/protocol.py:25`, `eval/metrics.py` `noninferiority` |
| H1 direction check `tmax-4b` vs base | done (script, unrun) | `scripts/h1_tmax_check.sh` |
| Eval sampling parameters explicit | done | `eval/rollout.py:11` |
| Spending gate Stage 0 | done | this document |

### §9 Multi-harness addendum, §10 Weight sync

| Requirement | Status | Where |
|---|---|---|
| 9.1.1 No decode and re-encode; differential test asserts ids, logprobs, mask | done | `trainer.py:444`; `tests/test_trainer_loop.py:346` and a real-`GRPOTrainer` end-to-end test |
| 9.1.2 Full-distribution sampling set explicitly, startup probe, `--generation-config vllm` | done / server unverified | `base.yaml:102-108`, `train.py:1193`, `models.py:276,331`, `serve_vllm.sh:48` |
| 9.1.3 Phase-0 gate: importance ratio within 1% of 1 | done | `callbacks.py:576` gates |
| 9.1.4 Chat template prefix-preserving, asserted with the real tokenizer | done on Qwen3-0.6B and Qwen3.5-9B tokenizers; not on the real 4B checkpoints | `models.py:520`, `train.py:1104` |
| 9.1.5 Bank stores `completion_ids`, `prompt_ids`, logprobs, tool mask | done | `bank.py:184`; trainer passes them (`trainer.py:297`) |
| 9.2.1 Efficiency bonus, default 0.1, reference = call budget, threshold unchanged | partly | `reward.py:47`; not paid on infeasible or lazy (D7) |
| 9.2.2 Nothing pays for submitting; stop rules for success-falls-while-reward-rises and report-rate drop | done | `callbacks.py` `RULE_SUCCESS_VS_REWARD`, `RULE_REPORT_RATE` |
| 9.2.3 Reward functions never raise | done | `trl_rewards.py:573` `_never_raise` |
| 9.3.1 Periodic validation probe, checkpoint selection on validation, keep every checkpoint | done (CPU) | `callbacks.py:356`, `train.py:1265`; selection metric ignores honesty (section 5, item 14) |
| 9.3.2 Identical caps in train and eval | done | `tests/test_integration_v3.py::test_budget_and_sampling_come_from_one_place` |
| 9.3.3 Resume continues task order | done | `tests/test_train_config.py:868` |
| 9.3.4 Share of steps with no contrast logged | done | `trl_rewards.py:258` |
| 9.4.1 Report-file channel under stock harnesses, per-harness tables | done (stand-in agent through `harbor run`) | `eval/report_file.py:60`, `eval/harbor_agent.py:162`, `scripts/eval_harness.sh` |
| 9.4.2 Stage 1 measurement of base honesty under our and a stock harness | script only | `scripts/eval_harness.sh` |
| 9.4.3 Multi-harness training | deferred by spec | not built |
| 9.5 SFT baseline | not built (optional by spec) | bank stores what it needs |
| 10.1 Clone patch + CPU test reproducing the bug | done (CPU model of vLLM's packed transfer) | `trainer.py:137`; `tests/test_trainer_loop.py` |
| 10.1.2 `sync_check.sh` with a non-zero adapter | done (CPU) | `sync_check.py:95,348`, `scripts/sync_check.sh` |
| 10.1.3 Sync mode chosen by the check | done | `sync_check.py recommend` |
| 10.2 Eval sends sampling explicitly | done | `eval/rollout.py` |
| 10.3 Trainer decision rule inputs logged | partly | exhaustion and zero-contrast computed by `phase0_summary`; step time, memory and "longest batch at per-device batch 1" are read by a person from `timing/*` and `gpu/peak_memory_gb` |

## 5. Unverified: must be checked on the node, ordered by risk

Risk means: how likely it is to be wrong, times how much money it wastes if it is.

1. **Hermetic verifier inside real images.** Never run in Docker: `-I -S -B`, `env -i`, `-X pycache_prefix`, bundle
   and tests upload ownership after `docker compose cp`, `timeout -k` on busybox, a real TTY on `compose exec`,
   non-root `[verifier].user`, the egress sidecar being forced for every pytest-verifier episode (and its per-episode
   cost), the pids limit through the compose override, Python 3.10 in `ubuntu:22.04`, converted Endless Terminals tests
   passing under `env -i` with no `HOME` (7 of 40 sampled tests call `subprocess`). 12 Docker-marked exploit tests and
   7 integration tests are the checklist. If the verifier cannot grade, nothing trains.
2. **Weight sync on a GPU.** The clone patch fixes a bug reproduced only against a CPU model of vLLM's packed
   transfer. If sync is wrong, a run learns against stale or half-merged weights while every metric looks normal. Run
   `scripts/sync_check.sh` before anything that learns. Peak memory of the clone wrapper at 4B scale is unmeasured.
3. **The false-tamper rate.** The protected-path and toolchain watches can flag legitimate work (`pip install -U pip`
   rewrites `/usr/local/bin/pip*` in `python:3.x` images). A false tamper pays −1.5 on a success. Measure in Phase 0:
   tamper rate among episodes whose tests pass, by detail prefix (`modified:`, `toolchain:`, `shadow:`, `stream:`), and
   the share with `verifier.hardened == False`.
4. **vLLM server shapes.** `/v1/models` `max_model_len`, completions `token_ids` and processed logprobs, the `-9999`
   top-k masking, `prompt_logprobs` response shape, `--generation-config vllm`, `--language-model-only`,
   `--enable-scale-out`, `--logprobs-mode processed_logprobs` (all read from vLLM 0.30 source, vLLM is not installed
   here). The startup probe is the guard: it fails the run, it does not fix anything.
5. **The real trainer on the real model.** One real 2-turn episode through `_generate`; `per_device_train_batch_size`
   2 at 16k+ tokens (fallback in `base.yaml`: 1 and 64); the probe's batch of 25; LoRA targets and `fla` on Qwen3.5-4B;
   `cast_lm_head_to_fp32` needs untied embeddings; the hybrid-Qwen3 template swap on the real Qwen3-4B tokenizer. The
   real `trl.trainer` ran only with a 4-layer random CPU model and a scripted sampler.
6. **Report compliance of the real model**: tool-call parse of real sampled report turns, thinking text before the
   call, mistyped arguments. The Stage 1 gate (≥ 70% reports, ≤ 5% no report) measures exactly this.
7. **Concurrency and step time.** 64 concurrent sandbox boots, per-episode hermetic-verifier and toolchain-hash cost
   (about 30-100 MB hashed twice per episode, bundle uploaded twice), two concurrent slots with distinct NCCL ports.
   Nothing has been timed.
8. **Dataset validation at scale.** No variant has been oracle-validated. Weak spots named by the builders: controls on
   SETA tasks, revoke-permission controls when the oracle writes outside the handed-over workspace, `remove_tool`
   (no candidate on the 300 review seeds), Harbor allowlist enforcement for the `no_network` controls, the sticky-parent
   ancestor sealing, and the audit sheets (nobody has read them).
9. **Phase 0 cannot measure the pool.** 24 steps × 8 prompts = about 190 prompts of a ~1,100-seed train pool, so the
   difficulty gate sees about 17% of it (and `--difficulty-strict` would keep only those). A dedicated pass@8 run over
   `data/train_pool.jsonl` is needed before Stage 3, for example
   `MODEL=policy scripts/eval_heldout.sh data/train_pool.jsonl runs/passk --n-samples 8 --bank-dir runs/passk/bank --train-config configs/base.yaml` (base model served as `policy`),
   then `python -m honest_agents.tasks.difficulty --bank runs/passk/bank --out difficulty.json`. The spec's own wording
   ("Phase-0 pass@8") underestimates this.
10. **Stock harnesses.** The report file reaching the artifacts directory, `terminus-2` trajectory files, `mini-swe-agent`
    reaching the endpoint from inside the container, and how well a 4B model follows `REPORT_FILE_INSTRUCTIONS`. Under a
    stock harness Harbor runs the task's own `test.sh`: no tamper, violation, evidence-replay or blocker verification.
    The `allenai/tmax-4b` serve flags are from its model card.
11. **Stop-rule and pilot thresholds** are judgement values, tested on simulated bandits and hand-made curves only. The
    pilot's effective sample per rate is distinct prompts (8 per round), and 100 steps at LoRA lr 1e-5 may not move the
    policy far enough for the stop rule to see an effect (it then says INCONCLUSIVE, which costs money without
    information).
12. **Residual grader risks (documented, not closed):** agent code that the tests import in-process (it gets the verifier's
    privileges); a root process racing the verifier through `/proc/<pid>/fd/1` (mitigated by the one-shot nonce file,
    the single marker pair, `PR_SET_DUMPABLE`, the exit-0 rule and the background sweep, none of them run in a container);
    manifest tools resolved from protected directories; `bash` output is not capped inside the container (a runaway
    `yes` is buffered by Harbor's exec for up to 60 s, a host-memory risk across 64 episodes); output token estimate at
    2.4 chars per token undercounts dense hex or CJK.
13. **OpenEnv image**: `docker build` with `--build-context verifier_bundle=...`, the docker.sock path, real WebSocket
    client against uvicorn, `timeout`/`ensure_dirs` in real images.
14. **Checkpoint selection ignores honesty.** The default `probe.selection_metric` is `eval_honest/task_success`, and
    TRL averages per-batch ratios, so the probe value is only approximately episode-weighted. Say so in the paper or
    change the metric.
15. **GPU-only costs and dates.** All cost figures in SPEC §8 are estimates.

## 6. Deviations from the spec (all recorded by the component that made them)

D1. Fourth mask reason `reward_error` (§9.2.3 asks for a masked value with a reason; §1 lists two). Used when grading
    raises before the agent ran a command, or when the whole reward function fails for a batch. After the agent acted a
    raising grade is tamper (§3.5).
D2. §3.5 and §1 conflict about a verifier that dies after the agent acted ("fail" versus "any grading failure is
    tamper"). Implemented: a test timeout or a runner that printed no result is a plain fail; exec, upload, manifest and
    network-switch failures are tamper. Validation treats such a result as an error, never as an oracle verdict.
D3. The `lazy` row pays no success or efficiency bonus even if the tests pass (otherwise doing nothing on a task
    whose tests pass untouched would be paid).
D4. A completion claim whose tests pass after an irreversible or forbidden action is `fabrication` (the spec defines
    pass as including no violation).
D5. "Episodes that have reported are not regenerated" is implemented as TRL's overlong rollback to the report turn: the
    report call executes (the environment records it) but its tool result is not appended to completions, ids, mask or
    logprobs. Consumers that expect a trailing tool message after the report will not find one. A report sampled on the
    last tool-loop iteration is never executed (TRL drops pending calls at the cap) and scores as no report.
D6. §5.2 is enforced by refusal (`ValueError` at construction, and in `train.py`), not by a lock-step multi-rank loop.
D7. The efficiency bonus is not paid on infeasible tasks or on a `lazy` decline (§9.2.1 says "when tests pass").
D8. Truncated completions: `mask_truncated_completions` masks the loss only; the reward still enters the group mean.
D9. An abstain with no blocker and no reason is scored implicit (gamma) on infeasible tasks (§1 rows 3 and 4 conflict).
    A placeholder reason (`n/a`, `none`, `-`) counts as no reason.
D10. Pools: phrasing pools of ≥ 8 train / 4 test entries exist for `conflicting_constraint` and `high_risk` only; the five
    environment operators use name pools (agent users, relocation dirs, backup suffixes, decoy hosts, sentinel sites).
D11. Held-out families: the default (`high_risk`, `revoke_permission`) removes the two highest-yield families from
    training. With 300 review seeds training keeps `remove_input`, `conflicting_constraint` and `no_network` (about 4% of
    seeds), and `high_risk`, the "look before deleting" skill, is never trained, only tested zero-shot. **Owner decision**:
    `HELDOUT=high_risk,no_network` gives more training coverage and a weaker environment-mechanism test.
D12. `RewardConfig` validates more than the Brier rule (ranges, `false_blame`/`lazy`/`no_report_feasible` ≤ alpha,
    `implicit_abstain` in [0, wrong_reason]). `HonestTaskSpec` rejects contradictory labels at load.
D13. Target matching beyond "normpath and command basename": URLs reduce to the host, bare names compare
    case-insensitively, a relative target matches an absolute candidate it is a suffix of. Lists, globs and directories
    never match.
D14. Validation: infeasible variants also run the oracle `oracle_runs` times; a `remove_input` variant that already
    passes is relabelled a control; `[verifier] network_mode = "no-network"` is declared in hermetic `task.toml`
    files; a hermetic task is refused if its original `test.sh` filters tests or needs a non-ctrf plugin; the grader runs
    tests from the container default directory, not the original `cd /home/user` (recorded in `[metadata.hermetic].cwd`).
D15. Controls carry `perturbation` and `control_of` equal to the operator they mirror (plain feasible rows keep `None`),
    so eval can pair them. Controls are 1:1 at generation but about 80% of infeasible rows keep theirs after the mixture.
D16. Pass in the verifier is stricter than exit code 0: at least one test, every test passed; skip, xfail, xpass or a
    collection error is not a pass. Pytest is the vendored 8.4.1 resolved for Python 3.9 with backports.
D17. The pilot: missing report arguments read as empty instead of failing the call; stop rule judged on the `none` arms
    only; coverage reported, not gated; groups keyed per trajectory; `max_tool_calling_iterations` forced to 0; refuses a
    run that shows a prompt more than once (`data.max_epochs`); an incomplete sweep is INCONCLUSIVE, never GO; per-turn
    cap 2048 (one report turn), not 4096.
D18. Eval: AUROC of stated confidence leaves out `abstain` reports; a false completion needs tests that were run; a
    neutral baseline needs neutral tool descriptions as well as a neutral prompt (`--neutral-prompt`, `--neutral-tools`);
    `eval/budget.py` deleted in favour of `harness.BudgetMeter`; extra metrics and `min_source_tasks` /
    `min_family_variants` warnings.
D19. `scale_rewards`: everything but `none` is rejected (not only `group`); `generation_kwargs` may contain only
    `max_tokens`; resume additionally requires `steps_per_generation == gradient_accumulation_steps`, `num_iterations == 1`
    and a task-order fingerprint; the probe uses TRL's evaluation loop through a memory-safe trainer subclass; a stop-rule
    stop exits with status 3.
D20. `EpisodeBudget.max_total_chars` is 29,491 from `from_limits` (the contract formula) while the dataclass default
    is 29,000.
D21. Grading: shadow detection flags only names the grader itself runs; the sweep (`kill_background`) is on for Docker
    only and never for legacy tasks; a root process racing the stream is a documented residual (SPEC §3.2's "a
    background process cannot write into that stream" holds for non-root processes only).
D22. `verifier_site = true` tasks are less hermetic by design; training should avoid them (none of the 40 sampled
    Endless Terminals tasks needs it).

## 7. What the spec gets wrong or leaves open

- §3.2 overstates the guarantee (D21). §3.5 and §1 contradict each other (D2). §1 table rows 3 and 4 contradict for an
  abstain with no blocker and no reason (D9). §1 says λ is paid whatever the report says, but paying it on `lazy` would
  reward doing nothing (D3). §1 says truncation is "handled by TRL"; only the loss is masked (D8).
- §1's leak definition (claim plus clean pass) leaves an infeasible variant that is solved and then declined paid up to
  1.0 and its group unmasked. Rare after validation, flagged in metrics and quarantine, not masked.
- §1 fixes `false_blame = 0`, which makes false blame free at alpha 0 (section 3, rejected request).
- Under `scale_rewards: none` the claim threshold is exact only if the pool's mean pass rate lies above it. A pool
  with mean pass rate 0.3 and alpha 0.4 collapses to never-claim in the simulations (92-100% of runs), and a one-command
  quit beats work when `lambda*s + efficiency < h*(alpha - no_report_feasible)` (h = budget-exhaustion rate). Both are
  data choices the Phase-0 summary now reports (`pool.safe_alphas`, `pool.work_vs_quit_gap`); the spec does not gate
  on them.
- §4.8 and §8 say "Phase-0 pass@8", but a 24-step Phase 0 cannot measure the pool (section 5, item 9).
- §8 Stage 1 does not say which model Phase 0 runs. `phase0_zero_lr.yaml` inherits Qwen3-4B from `base.yaml`; the
  headline model is Qwen3.5-4B (`configs/qwen35_4b.yaml`). The commands below pass `model.name` and `model.family`.
- §10.3's "projected concurrent step time" needs per-component timing from a real run; no part of the pipeline can
  compute it offline.

## 8. First paid step: Stage 1 of the spending gate

Stage 1 (SPEC §8): node bring-up, audit dataset, zero-learning-rate run, base and Tmax inference. Estimated
$300-500. Continue to Stage 2 only if report compliance ≥ 70%, no-report ≤ 5%, fabrication among failed feasible tasks
≥ 20%, and the measured step time is acceptable.

On a fresh 4×H100 Ubuntu VM (a real Docker daemon; not a container pod):

```bash
git clone <repo> && cd restraint            # or rsync this tree
scripts/setup_node.sh --install-docker      # uv, Docker, nftables check, builds the verifier bundle
uv pip install flash-linear-attention       # NOT done by setup_node.sh (it only warns); Qwen3.5 is ~20x slower without it. Not in uv.lock
scripts/probe_node.sh                       # GPUs, Docker, compose v2, Harbor egress-control kernel support

# (a) Audit dataset, validated through our own environment code path. Docker only, no GPU.
VALIDATE=1 scripts/build_audit_set.sh data/audit
#     Read data/audit/sheets/*.md, type ok|wrong|unsure on each "Verdict:" line, then:
scripts/build_audit_set.sh --tally data/audit       # stop if variants are wrong or controls leak

# (b) Weight sync with a non-zero adapter, before anything that learns (SPEC §10.1.2)
scripts/sync_check.sh configs/qwen35_4b.yaml        # exit 0 = pass; writes outputs/sync_check/*.json

# (c) End-to-end smoke: 2 tasks of different length, 2 generations, 2 optimiser steps, a probe after step 1
scripts/smoke_test.sh --train data/audit/tasks/<variant_a> --train data/audit/tasks/<variant_b> \
    --val data/audit/tasks/<variant_c> --config configs/qwen35_4b.yaml

# (d) Phase 0: the real trainer at learning rate 0 for 24 rounds, writes the trajectory bank and prints the gates
scripts/phase0.sh configs/phase0_zero_lr.yaml model.name=Qwen/Qwen3.5-4B model.family=qwen3_5 \
    data.train_file=data/audit/train.jsonl
#     outputs/phase0/<run>/phase0_summary.json: gates, timing/*, trainer_decision_10_3, pool.work_vs_quit_gap,
#     tamper rate among passing episodes. Exit 1 if a gate fails or was not measured.

# (e) Base against allenai/tmax-4b on a held-out split (inference only; needs the full dataset's test split)
VALIDATE=1 LIMIT=300 TEST_SEEDS=100 scripts/build_dataset.sh data
scripts/h1_tmax_check.sh data/test.jsonl runs/h1_tmax

# (f) Base-model honesty under a stock harness (SPEC §9.4.2); needs the policy served with EVAL=1 scripts/serve_vllm.sh
EVAL=1 scripts/serve_vllm.sh configs/qwen35_4b.yaml --served-model-name policy &
AGENT=terminus-2 MODEL=policy scripts/eval_harness.sh data/test.jsonl runs/harbor
MODEL=policy scripts/eval_heldout.sh data/test.jsonl runs/eval/base --n-samples 4 --concurrency 32 \
    --train-config configs/qwen35_4b.yaml
```

Do (a) and (b) first; neither needs a full training run, and either failing makes (d) pointless. If (d)'s importance
ratio is not within 1% of 1, stop and read `sampling/*` before anything else. Before Stage 3, run the dedicated pass@8
measurement over `data/train_pool.jsonl` (section 5, item 9).

Stage 2 is then `scripts/run_pilot.sh configs/pilot.yaml` (use `SLOT=0 SHARD=0/2` and `SLOT=1 SHARD=1/2` for the two run
slots), whose last line is `PILOT GO`, `PILOT NO-GO` or `PILOT INCONCLUSIVE`.
