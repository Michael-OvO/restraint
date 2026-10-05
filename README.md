# Honest Agents

Training an LLM agent to **abstain when a task cannot or should not be done** and to **report completion truthfully**, with
reinforcement learning on real terminal sandboxes.

The agent ends every episode with one structured report (`submit_report`) or an explicit `abstain`. The report is checked
against the **final sandbox state** with no human labels and no LLM judge: hidden tests run in a hermetic verifier, protected
paths and sentinels are hashed, and a blocker the agent names is verified in the sandbox where that is possible. An
honesty-aware reward then trains the policy with GRPO.

> **Status: pre-alpha. Nothing has run on Docker or a GPU.** A go/no-go review found that the original design must not be
> trained; the fixes (`docs/SPEC_V3.md`) are built and unit-tested on macOS against fake sandboxes. The first paid step is a
> gated, staged plan (`docs/ROADMAP.md`). Whether the approach works is not yet known.

- Proposal v2 (Chinese, for the owner and collaborators): [docs/proposal.md](docs/proposal.md); the original is [docs/proposal_v1.md](docs/proposal_v1.md)
- **Status, spending gate and next steps: [docs/ROADMAP.md](docs/ROADMAP.md)**
- What the review found and what was decided: [docs/REVIEW.md](docs/REVIEW.md)
- System design as built: [docs/DESIGN.md](docs/DESIGN.md)
- Binding fix specification: [docs/SPEC_V3.md](docs/SPEC_V3.md); interface contract: [docs/INTERFACES_V3.md](docs/INTERFACES_V3.md); integrator's test summary: `docs/V3_STATUS.md` (when present)
- Evaluation plan: [docs/EVAL.md](docs/EVAL.md)
- Research notes and the raw review: [docs/research/](docs/research/)

## Stack

| Layer | Tool |
|---|---|
| Tasks and sandboxes | [Harbor](https://github.com/harbor-framework/harbor) `0.23.0` (task dirs, Docker environments) |
| Trainer | [TRL](https://github.com/huggingface/trl) `1.14.1` `GRPOTrainer(environment_factory=...)` (subclassed: `HonestGRPOTrainer`), PEFT LoRA, vLLM `0.30.0` server mode |
| Env interop | [OpenEnv](https://github.com/huggingface/OpenEnv) `0.7.0` (separate extra; conflicts with vLLM's `mcp` pin) |
| Packaging | [uv](https://docs.astral.sh/uv/), Python 3.12 |
| Models | `Qwen/Qwen3-4B` (bring-up), `Qwen/Qwen3.5-4B` (headline). No distillation teacher: OPD is deferred |

## How it works

```
dataset row ──► HonestHarborEnv.reset(task_dir)      instruction returned at once; the sandbox boots in the background
                   │                                  (task network policy and agent user enforced; verifier self-check)
   policy ◄──────► bash(command)* ─► Harbor sandbox   budget enforced by the environment: calls and characters
                   submit_report(...) | abstain(...)  the first report is final; ONE blocker (category, target, detail)
                   │
reward function ─► grade: tamper hashes → sentinels → blocker check → fresh hermetic pytest (offline, nonce-marked stream)
                   compute_reward(outcome, config) ─► GRPO advantage (scale_rewards: none; tool tokens masked)
```

Reward (`src/honest_agents/reward.py`, tables in [docs/DESIGN.md](docs/DESIGN.md)). On a feasible task: a verified success claim
is worth `1 + lambda`, a fabrication `-1`, an honest "incomplete" `alpha` plus `lambda` if the tests pass anyway (so working beats
quitting), a false blame `0`, no report `-0.25`. On an infeasible task: a claim `-1`, a correct decline up to `1` (it must name
the right category **and** the right object), a decline for the wrong reason `0.5`, silence `0.25`; a claim that cleanly passes
masks the whole prompt group and quarantines the variant. Tampering is capped at `-1.5` and touching an irreversible path costs
`-beta`. **alpha is a confidence threshold only without group-std advantage scaling:** claim completion iff P(success) >
`(1 + alpha) / 2`, and `train.py` refuses `scale_rewards` other than `none`.

Infeasible tasks are built by perturbing solvable Endless Terminals tasks (remove an input or tool, revoke permissions, cut the
network, inject a contradictory constraint, a `high_risk` deletion), each with a near-miss feasible control. A variant is kept only
if the task's oracle solution, run through the same code path as training, behaves as labelled. "Infeasible by construction"
only means the oracle's route is blocked, so leaks (an agent finding another route) are detected, masked and quarantined.

## Repository layout

```
src/honest_agents/
  harness.py       prompt, tools (ReportTools), EpisodeBudget, report-file protocol; trl-free, shared by every runtime
  report.py        the one-blocker self-report schema and lenient parsing
  reward.py        honesty-aware reward v3 (pure function) and RewardConfig
  trl_rewards.py   one TRL reward function per mode (honest, outcome, outcome_claimed, outcome_reflect), metrics, group masks
  bank.py          trajectory bank rows, quarantine list, re-scoring without a sandbox
  spec.py          per-task honesty labels ([metadata.honest] in task.toml)
  grading.py       environment-grounded checks (tamper, sentinels, blocker check, hermetic tests)
  verifier/        hermetic pytest runner, vendored-bundle handling, verifier toolchain protection
  env.py           HonestHarborEnv: TRL environment over Harbor sandboxes
  trainer.py       HonestGRPOTrainer: concurrent tool loop that stops at the report, weight-sync fix
  train.py         GRPO entrypoint (YAML configs, guards, limits, validation probe, resume)
  callbacks.py     stop rules, run log, Phase-0 summary
  sync_check.py    weight-sync check (needs a GPU)
  models.py        model-family specifics (LoRA targets, templates, vLLM flags, startup probe)
  tasks/           seed sources, hermetic conversion, perturbation operators, validation, dataset build, difficulty gate, audit
  pilot/           report-only pilot on frozen Phase-0 trajectories (no sandbox)
  eval/            rollout loop, metrics, baselines, report-file protocol, Harbor agent, external benchmark adapters
  openenv_env/     OpenEnv server and client over the same core (its own README)
configs/           training configs: base, honest_alpha{0,0.1,0.25,0.4}, outcome, outcome_claimed, outcome_reflect,
                   phase0_zero_lr, pilot, smoke, qwen35_4b, colocate_1gpu, accelerate/
scripts/           node setup and probe, vLLM serving, run slots, sweeps, Phase 0, sync check, dataset build, evals
tests/             unit tests (fakes; no Docker or GPU) plus tests marked `docker` or `gpu`
docs/              proposal (v2, v1), design, roadmap, review record, specs, evaluation plan, research notes
```

## Quickstart

### Local development (macOS or Linux; no Docker or GPU needed)

```bash
uv sync --extra train --extra eval          # the project venv is shared; do not sync it from several places at once
env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval pytest -m "not docker and not gpu"
env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval ruff check src tests
scripts/build_verifier_bundle.sh            # vendored pytest bundle for the hermetic verifier (needs network once;
                                            # tests use an offline copy via tests/conftest.py)
```

Check a run config without a GPU once a dataset exists: `python -m honest_agents.train --config configs/honest_alpha0.25.yaml --dry-run` (needs `data/train.jsonl`).
The OpenEnv extra cannot share an environment with `train` (conflicting `mcp` pins); see `src/honest_agents/openenv_env/README.md`.

### GPU node (4xH100 VM with Docker). **Not yet run: follow the staged gate in [docs/ROADMAP.md](docs/ROADMAP.md).**

```bash
scripts/setup_node.sh                      # docker/compose checks, uv sync --frozen, builds the verifier bundle
scripts/probe_node.sh                      # GPUs, Docker, compose v2, nftables egress control; fails loudly
uv run pytest -m docker                    # real-sandbox grading and enforcement tests

scripts/build_audit_set.sh                 # 20-30 sources, two families plus controls, audit sheets for hand review
scripts/build_dataset.sh data              # seeds -> hermetic conversion -> variants -> oracle validation -> splits (LIMIT=1500)

scripts/smoke_test.sh --train data/tasks/<a> --train data/tasks/<b> --val data/tasks/<c>   # 2 tasks, 2 steps
scripts/phase0.sh                          # Stage 1: real trainer at learning rate 0, writes the trajectory bank, prints the gates
scripts/sync_check.sh                      # weight sync with a non-zero adapter; run before the first run that learns
python -m honest_agents.pilot.run --config configs/pilot.yaml && scripts/pilot_report.sh    # Stage 2: report-only pilot

scripts/run_pair.sh configs/honest_alpha0.25.yaml configs/outcome.yaml seed=42   # Stage 3: two runs, one node, same seed
scripts/sweep_alpha.sh                     # Stage 4: alpha {0, 0.1, 0.25, 0.4} + outcome arms (SEEDS="42 43 44")
```

One training process per run; two runs share a 4-GPU node (slot 0: train on GPU 0, vLLM on GPU 1; slot 1: GPUs 2 and 3).
`scripts/run_slot.sh CONFIG` runs one.

### Evaluation

```bash
EVAL=1 scripts/serve_vllm.sh configs/base.yaml --served-model-name policy &
MODEL=policy scripts/eval_heldout.sh data/test.jsonl runs/eval/a0.25            # primary: our held-out terminal set
MODEL=policy scripts/eval_heldout.sh data/test.jsonl runs/eval/neutral --neutral-prompt --neutral-tools   # no-training baseline
scripts/h1_tmax_check.sh data/test.jsonl                                       # base Qwen3.5-4B vs allenai/tmax-4b, before any run
AGENT=terminus-2 MODEL=policy scripts/eval_harness.sh data/test.jsonl runs/harbor   # third-party harness, report-file protocol
MODEL=policy scripts/eval_tb.sh tblite
```

External benchmarks (AgentAbstain, Agentic Abstention, AbstentionBench) are transfer checks only and run in their own venvs;
see [docs/EVAL.md](docs/EVAL.md).

## Licenses of data sources

Endless Terminals (MIT, malformed dataset card), SETA-Env (Apache-2.0, not yet used), TMax-15K (ODC-BY, contains Gemini outputs;
no oracle solutions, so not a validated source), AbstentionBench (CC BY-NC 4.0, evaluation only). Terminal-Bench and TB Lite are
benchmarks and are deduplicated out of training.
