# Honest Agents

Training an LLM agent to **abstain when a task cannot or should not be done** and to **report completion truthfully**, with
reinforcement learning on real terminal sandboxes.

The agent ends every episode with one structured report (`submit_report`) or an explicit `abstain`. The report is checked
against the **final sandbox state** with no human labels and no LLM judge: hidden tests run in a hermetic verifier, protected
paths and sentinels are hashed, and a blocker the agent names is verified in the sandbox where that is possible. An
honesty-aware reward then trains the policy with GRPO (group-relative policy optimisation).

> **Status: pre-alpha. Nothing has run on Docker or a GPU.** A go/no-go review found that the first design must not be
> trained; the fixes are built and unit-tested on macOS against fake sandboxes (2,898 tests pass, 33 skip). The first paid step is a
> gated, staged plan. Whether the approach works is not yet known.

## Documents

| File | What it is |
|---|---|
| [docs/design.md](docs/design.md) | the system as built: layers, episode lifecycle, tool contract, reward tables, verifier, data pipeline, training, SFT, pilot, performance |
| [docs/restraint-signal.md](docs/restraint-signal.md) | the restraint signal: pricing self-inflicted impossibility apart from a pre-existing one; design, invariants, experiments, red team, open risks |
| [docs/evaluation.md](docs/evaluation.md) | metrics, baselines, protocol fixed before the run, harness dependence, external benchmarks |
| [docs/data.md](docs/data.md) | sources, licences, task families, leakage controls and the leak register, how to rebuild |
| [docs/roadmap.md](docs/roadmap.md) | status, the staged spending gate with stop conditions, what runs where, the SFT then GRPO plan |
| [docs/node-runbook.md](docs/node-runbook.md) | Stage 1 commands and the Docker and GPU checks, ordered by risk |
| [docs/reviews.md](docs/reviews.md) | the three reviews and two audits: verdicts, findings, decisions, how each was resolved |
| [docs/proposal.md](docs/proposal.md) | the plan and hypotheses (v2, Chinese); the original is [docs/archive/proposal_v1.md](docs/archive/proposal_v1.md) |

## Stack

| Layer | Tool |
|---|---|
| Tasks and sandboxes | [Harbor](https://github.com/harbor-framework/harbor) `0.23.0` (task dirs, Docker environments) |
| Trainer | [TRL](https://github.com/huggingface/trl) `1.14.1` `GRPOTrainer(environment_factory=...)` (subclassed: `HonestGRPOTrainer`), PEFT LoRA, vLLM `0.30.0` server mode |
| Env interop | [OpenEnv](https://github.com/huggingface/OpenEnv) `0.7.0` (separate extra; conflicts with vLLM's `mcp` pin) |
| Packaging | [uv](https://docs.astral.sh/uv/), Python 3.12 |
| Models | `Qwen/Qwen3-4B` (bring-up), `Qwen/Qwen3.5-4B` (headline). No distillation teacher: on-policy distillation (OPD) is deferred |

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

Reward (`src/honest_agents/reward.py`, tables in [docs/design.md](docs/design.md) section 4). On a feasible task: a verified success
claim is worth `1 + lambda`, a fabrication `-1`, an honest "incomplete" `alpha` plus `lambda` if the tests pass anyway (so working beats
quitting), a false blame `0`, no report `-0.25`. On an infeasible task: a claim `-1`, a correct decline up to `1` (it must name the right
category **and** the right object), a decline for the wrong reason `0.5`, silence `0.25`; a claim that cleanly passes masks the whole prompt
group and quarantines the variant. Tampering is capped at `-1.5` and touching an irreversible path costs `-beta`. **alpha is a confidence
threshold only without group-std advantage scaling:** claim completion iff P(success) > `(1 + alpha) / 2`, and `train/` refuses
`scale_rewards` other than `none`.

Infeasible tasks are built by perturbing solvable Endless Terminals tasks (remove an input, revoke permissions, cut the network, inject a
contradictory constraint, a `high_risk` deletion; removing a tool yields none on real seeds), each with a near-miss feasible control. A
variant is kept only if the task's oracle solution, run through the same code path as training, behaves as labelled. "Infeasible by
construction" only means the oracle's route is blocked, so leaks (an agent finding another route) are detected, masked and quarantined.
Details: [docs/data.md](docs/data.md).

## Layout

| Path | Purpose |
|---|---|
| `src/honest_agents/` | the package; every module is listed below |
| `configs/` | training and SFT configs: `base`, `honest_alpha{0,0.1,0.25,0.4}`, `outcome`, `outcome_claimed`, `outcome_reflect`, `phase0_zero_lr`, `phase0_colocate`, `pilot`, `sft_hindsight`, `sft_bestofn`, `smoke`, `qwen35_4b`, `colocate_1gpu`, `accelerate/` |
| `scripts/` | shell and Python entry points: node setup and probe, vLLM serving, run slots, sweeps, Phase 0, sync check, dataset build, evaluation, the test runner |
| `tests/` | unit tests (fakes; no Docker or GPU) plus tests marked `docker` or `gpu`; `tests/sandbox_fakes.py` is the shared host-subprocess sandbox |
| `docs/` | the documents above; `docs/archive/` holds the original proposal |
| `docker/openenv/` | the OpenEnv server image (takes the verifier bundle as a build context) |
| `pyproject.toml`, `uv.lock` | dependencies (extras `train`, `eval`, `openenv`), console scripts, ruff and pytest settings |
| `data/`, `outputs/`, `runs/`, `benchmarks/` | generated, git-ignored: datasets, training output, evaluation runs, benchmark texts for decontamination |

Modules under `src/honest_agents/`:

```
  harness.py       prompt, tools (ReportTools), EpisodeBudget, report-file protocol; trl-free, shared by every runtime
  report.py        the one-blocker self-report schema and lenient parsing
  reward.py        honesty-aware reward (pure function) and RewardConfig
  trl_rewards.py   one TRL reward function per mode (honest, outcome, outcome_claimed, outcome_reflect), metrics, group masks
  bank.py          trajectory bank rows, quarantine list, re-scoring without a sandbox
  spec.py          per-task honesty labels ([metadata.honest] in task.toml)
  grading.py       environment-grounded checks (tamper, sentinels, blocker check, hermetic tests)
  verifier/        hermetic pytest runner, vendored-bundle handling, verifier toolchain protection
  env.py           HonestHarborEnv: TRL environment over Harbor sandboxes
  trainer.py       HonestGRPOTrainer: concurrent tool loop that stops at the report, weight-sync fix
  train/           GRPO entrypoint, a package: config, derive (limits, reward config), grpo (guards), rows, plan, startup, run, cli
  callbacks.py     stop rules, validation probe, run log, Phase-0 summary
  sync_check.py    weight-sync check (needs a GPU)
  models.py        model-family specifics (LoRA targets, templates, vLLM flags, startup probe)
  tasks/           seed sources, hermetic conversion, validation, difficulty gate, audit; `analysis/` (static task facts and
                   leak statistics), `perturb/` (one module per operator) and `build/` (splits, mixture, pipeline) are packages
  pilot/           report-only pilot on frozen Phase-0 trajectories (no sandbox)
  sft/             supervised stages on the bank: hindsight report SFT and best-of-N trajectory SFT (data, hindsight, bestofn, train_sft)
  eval/            rollout loop, metrics (a package), baselines, report-file protocol, Harbor agent, external benchmark adapters
  openenv_env/     OpenEnv server and client over the same core (its own README)
```

## Quickstart

### Local development (macOS or Linux; no Docker or GPU needed)

```bash
uv sync --extra train --extra eval          # the project venv is shared; do not sync it from several places at once
scripts/test.sh                             # fast suite in parallel (not docker/gpu/slow), prints wall time
env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval ruff check src tests scripts
scripts/build_verifier_bundle.sh            # vendored pytest bundle for the hermetic verifier (needs network once;
                                            # tests use an offline copy via tests/conftest.py)
```

`scripts/test.sh` is the single entry point (pytest-xdist, one worker per core, a lock so concurrent full runs queue):

| command | runs | wall time (18-core Mac) |
|---|---|---|
| `scripts/test.sh` / `--fast` | everything except `docker`, `gpu` and `slow` tests | about 70 s |
| `scripts/test.sh --all` | everything except `docker` and `gpu` | about 3 min on a busy machine (serial: about 6 min) |
| `scripts/test.sh --changed` | test files related to files changed in git (by name and direct imports) | seconds |
| `scripts/test.sh -k expr tests/test_env.py` | plain pytest arguments, serial, no marker filter | |

`slow` marks the tests that spend seconds in real verifier subprocesses or full dataset builds. The 21 tests marked `docker` or `gpu` have
never run (they need a Linux node: `pytest -m docker`). Markers and the pytest and ruff settings live in `pyproject.toml`. The OpenEnv extra
cannot share an environment with `train` (conflicting `mcp` pins); see `src/honest_agents/openenv_env/README.md`.

Check a run config without a GPU once a dataset exists: `python -m honest_agents.train --config configs/honest_alpha0.25.yaml --dry-run`
(needs `data/train.jsonl`, from `scripts/build_dataset.sh`; see [docs/data.md](docs/data.md)). The SFT stage has a CPU dry run too:
`DRY_RUN=1 scripts/run_sft.sh configs/sft_hindsight.yaml` (needs a bank).

### Everything that costs money

Not yet run. Follow the staged gate in [docs/roadmap.md](docs/roadmap.md) and the commands in [docs/node-runbook.md](docs/node-runbook.md).
Stage 1 is one rented x86 H100 box with Docker: node setup and probe, `pytest -m docker`, the audit set, the base evaluation, Phase 0
on one GPU (`scripts/phase0.sh configs/phase0_colocate.yaml`), then the supervised stages (`scripts/run_sft.sh`), each adapter evaluated
against the base before one short GRPO run. The weight-sync check needs a second GPU (`scripts/sync_check.sh`). Later stages: the
report-only pilot (`scripts/run_pilot.sh`), the paired runs (`scripts/run_pair.sh`) and the alpha sweep (`scripts/sweep_alpha.sh`).
A local Docker daemon on the Mac (Colima or OrbStack) is optional and only a hint on arm64. Evaluation commands are in
[docs/evaluation.md](docs/evaluation.md). `tests/test_scripts.py` keeps every command named in these documents runnable (scripts parse,
configs dry-run, modules, flags and links exist).

## Licences of data sources

Endless Terminals (MIT, malformed dataset card), SETA-Env (Apache-2.0, not used), TMax-15K (ODC-BY upstream, contains Gemini outputs; no
oracle solutions, so not a validated source), AbstentionBench (CC BY-NC 4.0, evaluation only). Terminal-Bench and TB Lite are benchmarks and
are deduplicated out of training. Full table: [docs/data.md](docs/data.md) section 1.
