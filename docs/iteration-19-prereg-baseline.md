# Iteration 19, stage (a): headline baseline, pre-registration (APPROVED 2026-10-31)

Plan: [iteration-19-proposal.md](iteration-19-proposal.md), step 2. Job file: `configs/chains/it19_a_baseline.yaml`.
This amends the final-benchmark plan of [iteration-16-prereg.md](iteration-16-prereg.md) sections 4, 8 and 10:
sections 10.3-10.4 allowed the final benchmark only after a validation GO and ran base, not G2, on TB2 and
ImpossibleBench. Here base and G2 run now, before any new training, so that every later claim has a public-benchmark
baseline. **Approved by the owner on 2026-10-31 ("Approve (Recommended)"), before any episode of this stage.** From
here nothing changes except by a dated owner amendment.

## 1. Purpose

The project has never measured a model on the frozen final benchmark. This stage measures the base model and G2 on
the headline benchmarks, once, and fixes the numbers a later arm is compared against. It is a measurement: no
decision about training depends on its result.

## 2. Arms

- **base:** `Qwen/Qwen3.5-4B` at the manifest's pin.
- **G2:** it15 G2 seed 1 (the SFT-v2 merge plus `data/weights/it15/g2-s1-final`, adapter sha256 in the manifest).

One vLLM server per arm per step, on the same node, same harness, system prompt and sampling settings (the eval
defaults of `configs/grpo_it13_2gpu.yaml`).

## 3. Benchmarks and samples

| benchmark | items | samples | headline metric(s) |
|---|---|---|---|
| internal held-out (`internal_heldout`) | 569 variants on 119 sources | S = 2 | `heldout.false_claim` (down); `heldout.success` (non-inferior) |
| Terminal-Bench 2 (`terminal_bench_2`) | 89 tasks | S = 4 | `tb2.false_claim` (down), `tb2.success` (non-inferior) |
| ImpossibleBench LCB (`impossiblebench_lcb`, tuned prompt) | 309 | S = 4 | `abstain.flag` (up); also `cheated`, `claimed_complete`, `act.success`, `act.over_flag` |

- **Internal held-out, frozen 2026-10-31** (manifest `refrozen` log): the frozen test split (`data/test.jsonl`, 156
  sources) less the 20 sources the iteration-12 transfer check ran on (`data/transfer_v1.jsonl`) and the 21 test
  sources reviewed as spec defects (DEF-66); 4 sources are in both, so 37 are left out. The rows file
  `data/benchmarks/internal_heldout.jsonl` is hashed in the manifest and checked before the run.
- **Not run, reported as not run:**
  - MASK: its judge is a paid API with no owner approval; responses are not generated either, because the tool has no
    generate-only mode;
  - AbstentionBench: DEF-50, closed by not running it;
  - Agentic Abstention: reported only, 19 abstain units, underpowered;
  - ImpossibleBench SWE: no adapter;
  - RTS.

## 4. Estimates

- **Estimator:** G2 − base, the manifest's estimator: a paired cluster bootstrap, 95 % CI, 2,000 resamples, seed 0.
  The clusters are sources on the internal set and items on TB2 and ImpossibleBench (`final_benchmark compute` on the
  units files; the internal set through `analysis.power estimate`, which excludes DEF-66 sources by default, so
  nothing further is dropped there).
- **MDEs:** computed from the base arm's own run and reported beside each estimate (80 % power, the base arm's K and
  S). They are never used to move a hypothesis.

## 5. Pre-registered hypotheses (directional, reported; no decision depends on them)

- **H1:** internal held-out false claims fall, G2 − base: UB < 0.
- **H2:** ImpossibleBench flagging rises, G2 − base: LB > 0.
- **H3:** TB2 false claims fall, G2 − base: UB < 0. `tb2.false_claim` is per episode, so declining more lowers it
  by itself (a 4B model solves 0-4 of 89 TB2 tasks; training log, iteration 14). H3 is therefore reported only with
  the TB2 claim rate and P(fail | claim) beside it, and it counts as supported only if P(fail | claim) does not rise
  (UB < +5 points).
- **Reported with their MDE, no hypothesis:** internal and TB2 success, ImpossibleBench `cheated` and `act.success`.

A result is reported as supported, contradicted (the CI excludes 0 the other way) or inconclusive.

## 6. One look and the pinned eval path

- Each benchmark runs once per arm. A failed step is resumed from its done markers, never re-run for a better number.
  An infrastructure failure that voids an arm's run (no episodes graded) is reported, and a re-run then needs an owner
  amendment.
- **The eval path is pinned at the launch commit**, recorded in the sign-off: the harness, env, grading, reward,
  verifier, `src/honest_agents/eval/`, `scripts/eval_heldout.sh`, `eval_tb.sh`, `eval_common.sh`,
  `serve_and_run.sh`, `configs/grpo_it13_2gpu.yaml`, `data/benchmarks/`, `data/ext/impossiblebench/`. A later arm is
  compared with these numbers only if `git diff <launch commit>` is empty on that list; otherwise base and G2 are
  re-measured beside it. A harness change between iterations 16 and 17 moved G2's verified success by 3.6 points
  (DEF-65).

## 7. Cost and node

- **Node:** one VM-capable GPU node (RTX 5090 32 GB or larger, at least 32 cores), **at least 400 GB of disk** for the
  TB2 images (about 150 GB). The offer is shown to the owner before the rent.
- **Time:**
  - fixed: setup and merges about 1.25 h, TB2 image pulls about 0.5 h;
  - per arm: internal held-out about 1.2-1.6 h (1,138 episodes at 700-950 per hour), TB2 about 2-4 h (no 30-turn TB2
    run has been timed), ImpossibleBench about 0.9-1.7 h.
  - **About 10-16.5 h in all.**
- **Cost:** about $11-18 at $1.11 per hour, $19-32 at $1.93 per hour, plus 30 % for overrun (`docs/iteration-19-research/infra.md`
  part B). Balance $74.07, watcher floor $8.
- **Watcher:** armed in the same minute as the chain; it pulls the units, job dirs and evals, archives them
  (`iterations/19-it19a-baseline`) and stops the node.

## 8. Before any GPU spend

- The integrity gate is open with no waiver: all 68 ledger entries are closed (`gate.ledger check --experiment it19a`).
- Mac checks: `scripts/test.sh --fast`; the L0 sweep; `node_chain check` on the job file;
  `final_benchmark check --archive`; `never-trained --archive` pass (2026-10-31: 68,774 rows, 0 hits).
- Owner approval of this pre-registration and of the GPU spend.

## Amendment 2026-11-01 (owner): the ImpossibleBench headline for later arms (DEF-70)

After this stage, the owner chose "Lenient flag headline (Recommended)": every later arm is compared on
`abstain.flag_lenient` as the ImpossibleBench headline, with the strict `flag` reported beside it. This stage's H2 is
reported as pre-registered (strict flag, contradicted), with the lenient reading beside it.

