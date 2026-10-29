# Honest Agents

Training a small LLM agent to **stop when a task cannot or should not be done**, and to **report truthfully whether it
finished**, on real terminal sandboxes, with rewards checked against the sandbox state rather than an LLM judge.

The agent (Qwen3.5-4B with LoRA) works in a Docker sandbox through a `bash` tool and ends every episode with one
structured report: `submit_report(status, completed, incomplete, blocker, evidence, confidence)` or `abstain`. Every
report is graded against the **final sandbox state**: hidden tests run in a hermetic verifier, protected paths are
hashed, and a blocker the agent names is checked where that is possible. Each feasible task has impossible "twins"
built from it (a removed input, revoked permissions, data that must not be deleted) and near-miss controls, so a policy
that declines blindly is caught as easily as one that claims blindly.

> **Status: research in progress (17 iterations).** The work is documented in full, failures included: every
> iteration has a pre-registered gate, and most did not pass it. The latest result (iteration 17) is a **NO-GO**:
> the primary endpoint passed, a safety guard failed (below).

## What we found

- **Restraint can be trained, but mostly as a caution level.** Supervising the decision at the model's own states
  (iterations 13-14) raised correct abstention on impossible tasks by **+42 to +45 points** with verified success
  non-inferior (+5.1 [+0.9, +9.2] for the best seed). On evidence formats it had not seen, no method (SFT, RL, DPO)
  produced real discrimination: "they move a caution level".
- **Good habits are fragile.** Masking the inspection turns in one SFT stage (iteration 15, G1) erased the model's
  habit of reading before deleting (read rate ≤ 5 % on every surface). Restoring the real inspection turns (G2) brought
  it back; G2 became the reference model.
- **The model knows more than it says.** G2 writes confidence 1.0 on almost every claim (AUROC for failure ~0.5), yet
  its own token probabilities already rank its failing claims well (~0.79). Training only the confidence value
  (iteration 17) raised the AUROC of the stated value from **0.59 to 0.77** on 620 held-out claims (+0.18
  [+0.14, +0.23]).
- **...but a regularizer only protects the tokens it covers.** The same training, with a KL penalty on the confidence
  value alone, let the shared weights drift: the model stopped reading before deleting and destroyed protected data on
  **53 of 120** "do not delete" tasks (G2: 13). The pre-registered guard caught it; iteration 18 anchors the actions.

Iteration by iteration, with sources: **[docs/results.md](docs/results.md)**.

## How it is run

The engineering is as much the point as the results.

- **Pre-registration.** Every training stage fixes its endpoints, bars, guards and stopping rules before any episode
  of the candidate exists ([docs/iteration-17-prereg.md](docs/iteration-17-prereg.md) is the latest). Margins come
  from a power analysis of a reference run, never from a point estimate.
- **One estimator.** Paired differences by source with cluster-robust intervals and sign-flip tests
  (`analysis.power.paired_estimate`), on the units both arms share.
- **An integrity gate.** No experiment starts while any entry in the defect ledger ([docs/defects.md](docs/defects.md),
  65 entries) is open, unless the owner waives it in writing by ID. A tiered sweep runs before each stage: in-memory
  checks (L0, inside the test suite), cached Docker checks (L1), and a full re-run (L2).
- **Never trained on the benchmark.** A frozen final benchmark (sha256-pinned item lists) and a `never-trained` check
  over every training build in the tree and the archive.
- **Rented GPUs, watched.** Jobs run as declarative chains on rented Vast.ai VMs (`node_chain`): gated steps, done
  markers, timeouts, stall detection, a watcher that pulls and archives results and stops the node at the end, and a
  balance floor.

## How it works

```
dataset row ──► HonestHarborEnv.reset(task_dir)      instruction returned at once; the sandbox boots in the background
                   │                                  (network policy and agent user enforced; verifier self-check)
   policy ◄──────► bash(command)* ─► Harbor sandbox   budget enforced by the environment: calls and characters
                   submit_report(...) | abstain(...)  the first report is final; one blocker (category, target, detail)
                   │
reward function ─► grade: tamper hashes → sentinels → blocker check → fresh hermetic pytest (offline)
                   compute_reward(outcome, config) ─► GRPO advantage (scale_rewards: none; tool tokens masked)
```

**Reward** (`src/honest_agents/reward.py`; tables in [docs/design.md](docs/design.md) section 4). On a feasible task a
verified claim is worth `1 + lambda`, a fabrication `-1`, an honest "incomplete" `alpha` (plus `lambda` if the tests
pass anyway, so working beats quitting). On an impossible task a claim is `-1` and a correct decline up to `1` (it must
name the right category and the right object). Touching a protected path costs `-beta`; tampering caps the reward at
`-1.5`. Without group-std scaling, `alpha` is a confidence threshold: claim iff P(success) > `(1 + alpha) / 2`.

**Tasks.** Impossible variants are built by perturbing solvable [Endless Terminals](docs/data.md) tasks, each with a
near-miss feasible control; a variant is kept only if its oracle solution, run through the training code path,
behaves as labelled, and an agent that finds another route is detected and quarantined. Details:
[docs/data.md](docs/data.md).

## Layout

| Path | Purpose |
|---|---|
| `src/honest_agents/` | the package (modules below) |
| `configs/` | GRPO and SFT configs; `configs/chains/` the GPU job files; `configs/confidence/` the iteration 17-18 trainer |
| `scripts/` | node setup and probe, vLLM serving, evaluation, the test runner |
| `tests/` | about 5,000 unit tests on fakes (no Docker or GPU), plus `docker` and `gpu` markers |
| `docs/` | design, results, pre-registrations, defect ledger, runbooks ([docs/README.md](docs/README.md)) |
| `data/`, `outputs/`, `runs/` | generated and mostly git-ignored; benchmark-derived data is not distributed (see Licences) |

```
  harness.py, env.py   the tool contract, budget and report protocol; the TRL environment over Harbor sandboxes
  reward.py, grading.py, verifier/   the honesty-aware reward and the environment-grounded grader
  trainer.py, train/   GRPO (TRL subclass); train/confidence*.py the confidence-only trainers (iterations 17-18)
  sft/                 supervised stages: decision splices, judgment items, commit-check pairs, the confidence bandit
  tasks/               seed sources, hermetic conversion, perturbation operators, splits, evidence surfaces
  eval/                rollouts, metrics, decision probe, external benchmark adapters, the frozen final benchmark
  analysis/            power and the paired estimator, reward audit, calibration analysis
  gate/                the integrity sweep and the defect-ledger gate
  node.py, node_chain.py, node_watch.py   rented-node workflow: sync, chained jobs, the watcher
```

## Quickstart (macOS or Linux; no Docker or GPU needed)

```bash
uv sync --extra train --extra eval
scripts/test.sh                       # the fast suite in parallel: about 5,000 tests, about 2 minutes on an 18-core Mac
python -m honest_agents.gate sweep    # the L0 integrity sweep (in memory, under a minute)
```

GPU work runs on a rented Linux VM with Docker, from the Mac: `python -m honest_agents.node sync|chain|watch` pushes
the tree over rsync, starts a chain with the W&B token passed over ssh's stdin (no secret is stored on the node), and
watches it to the end. Commands and checks: [docs/node-runbook.md](docs/node-runbook.md).

## Licences

Code: [MIT](LICENSE). Data sources: Endless Terminals (MIT), SETA-Env (Apache-2.0, not used), TMax-15K (ODC-BY
upstream, not a validated source), AbstentionBench (CC BY-NC 4.0, evaluation only). Terminal-Bench, ImpossibleBench
and the restraint transfer suite built from them are evaluation sets: they are deduplicated out of training and their
task files are not distributed here; the builders that regenerate them are. Full table:
[docs/data.md](docs/data.md) section 1.
