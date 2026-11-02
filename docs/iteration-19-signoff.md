# Iteration 19 sign-off

## it19a, the headline baseline (owner, 2026-10-31)

- **Pre-registration approved:** "Approve (Recommended)" (docs/iteration-19-prereg-baseline.md as of this commit).
- **Ledger:** all 69 entries closed; `gate.ledger check --experiment it19a` exits 0 with no waiver (DEF-10 and DEF-20
  closed as superseded, DEF-50 by not running AbstentionBench: owner, 2026-10-31).
- **Node:** 54988871 (Taiwan) never accepted SSH, not after a reboot; stopped and destroyed. The owner then chose
  "58352110 Japan (Recommended)" (RTX 5090, 30 cores, 197 GB RAM, 436 GB disk, $0.91/h), instance 58586655, and
  "Destroy all three (Recommended)" for the stopped nodes 58579674, 58122470, 58292161 (all archived).
- **Launch:** 2026-10-31 from commit `634ef8e`: the eval path of prereg section 6 is pinned at this commit.
- **GPU spend (first offer):** "Rent 54988871 (Recommended)": RTX 5090 32 GB, 32 cores, 123 GB RAM, Taiwan, $0.78/h; about 10-16.5 h,
  about $8-13 (about $17 with overrun); a watcher stops the node when the chain ends. Balance before: $29.29 (after
  DEF-69).
- **Pushing:** the repo is public; the owner chose "Push at milestones (Recommended)".
- **Mac checks:** `scripts/test.sh --fast` 5,040 passed (the L0 timing test passes alone in 40 s); L0 sweep PASS
  (37.9 s); `node_chain check` on the job file; `final_benchmark check --archive` ok; `never-trained --archive` pass
  (68,774 rows, 0 hits); `data/ext/impossiblebench` staged and its MANIFEST verified.

- **Metric definitions, fixed 2026-11-01 before any G2 result was read** (the prereg names the manifest metrics; this
  records how each is computed): internal `heldout.false_claim` = `false_claim_share` (false completions / claims whose
  tests ran; docs/evaluation.md 5.1, primary endpoint 2), reported with `verified_success` and `claim_rate_feasible`
  (`analysis.power estimate`, DEF-66 excluded, which removes nothing more since the set is already defect-free);
  internal `heldout.success` = `task_success`; TB2 and ImpossibleBench as their units writers define them
  (`final_benchmark compute`).

## it19b K2 (owner, 2026-11-01)

- **GPU spend:** "Run K2 after it19a (Recommended)": `configs/chains/it19_b_k2.yaml` on node 58586655 right after the
  baseline, about 2-2.5 h, about $2-3. The it19a watcher was re-armed without the node stop (so the node is not
  stopped and restarted between the two chains, the DEF-69 risk); the it19b watcher stops the node at its end.
- **Ledger:** `gate.ledger check --experiment it19b` exits 0 (all entries closed).
- **Mac checks:** K0 / K1 / K1(b) pass (docs/iteration-19-design-crgrpo.md); the K2a trainer, the scorer and the
  payoff table pass their tests; `scripts/test.sh --fast` 5,057 passed (the L0 timing test passes alone);
  `node_chain check` on the job file.

## it19c CR-GRPO (2026-11-02, under the owner's delegation)

- **Owner approval:** before an overnight run the owner wrote "you have the right to execute all commands, have the
  server use anything at your will, and do all the training. Just make sure we're moving and pushing as fast as
  possible ... If you think the resource is too scarce, it's also okay. We can rent a bigger machine." The
  pre-registration (docs/iteration-19-prereg-crgrpo.md) is approved under that delegation, with two launch deviations
  recorded at its end (a2 still running; two GPUs in parallel).
- **Node and spend:** the only VM-capable 96 GB offers at launch were 57881531 (1 GPU, 12 vCPU: rented, found too
  CPU-poor for Docker-bound rollouts, the iteration-16 runbook asks >= 23 vCPU, destroyed unused) and 58236770:
  2x RTX PRO 6000 WS 96 GB, 128 vCPU, 504 GB RAM, Czechia, **$4.53/h**, instance 58781495; about 17 h, about $77.
  Balance before: $113.79 (stage a2 runs at the same time on instance 58779312, $0.67/h).
- **Stage a2 / DEF-74:** a2's `def74_regrade` ran first (2026-11-02, 3 min): 27 + 2 + 4 episodes replayed in the three
  in-scope runs (the other 58 rows of `data/def74/affected.jsonl` are older archived runs), 14 / 1 / 4 changed to
  `success_claimed`. `crgrpo family-means` on `runs/def74/it17_rollouts_feasible`: remove_input **0.6225** (215
  proceeds; 0.6236 on 202 before), revoke_permission 0.8338 unchanged, high_risk kept at the surface value 0.7043.
  Fixed in `configs/grpo_it19_crgrpo.yaml` (the ablation inherits it).
- **Ledger:** `gate.ledger check --experiment it19c`: all 75 entries closed or waived.
- **Mac checks:** `scripts/test.sh --fast` 5,140 passed, 1 failed: the L0 timing test (77 s against 60 s under the
  parallel suite; passes alone in 41 s, as at it19a); `tests/test_crgrpo.py` 48 passed; `node_chain check` and
  `gate.guards --chains` on the two-GPU job file: pass; `final_benchmark never-trained --archive`: pass.
- **K1** (`analysis.reward_gate --config crgrpo` on G3's rollouts): PASS, pooled blind-decline credit −0.042, every
  blind policy below the start policy, reading pays on both branches (unchanged). Run with G2's surface-4 eval as well,
  that set reads FAIL on one line only: blind *proceeding* is level with G2's own policy (−0.036 [−0.154, +0.095]);
  every blind decline policy is below it (+0.199 to +0.464). It is not part of K1 and not a blind-decline credit; noted.
- **Evaluation-path diff** since `634ef8e`: exactly the files listed in the pre-registration section 10
  (`git diff --stat 634ef8e`: eval_tb.sh, impossiblebench.py, final_benchmark.py, harbor_agent.py, metrics/episode.py,
  metrics/scored.py, reporter.py (unused), rollout.py, grading.py, spec.py).
- **Job file:** `configs/chains/it19_c_crgrpo.yaml`, two-GPU layout (method on GPU 0; G2 reference then ablation on
  GPU 1, vLLM port 8001, MASTER_PORT per arm), reviewed by a 3-lens adversarial workflow before launch: 3 confirmed and
  fixed (a method-side `on_failure: stop` in calibration or the final benchmark would have skipped the parallel
  ablation arm's remaining steps; `method_final` waited on G6, so a G6 crash would have dropped the final benchmark;
  later steps now continue on failure and skip only their own dependents), 1 refuted (an infra-errored twin does not
  end an arm), minor kept: a failed step's leaked sandboxes are removed only when neither GPU is busy, and a step of
  one arm cannot be retried without a side job file while the other arm runs.
