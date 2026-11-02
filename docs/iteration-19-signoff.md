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

## it19c CR-GRPO (DRAFT: not signed; nothing approved)

To be filled at launch, in this order (pre-registration section 10):

- **Owner approval** of docs/iteration-19-prereg-crgrpo.md and of the spend (about $31-44 on a 96 GB-class node; the
  node launch asked separately, with the offer).
- **Stage a2 done**, its DEF-74 re-grade applied (`runs/def74/*`), and `python -m honest_agents.crgrpo family-means` on
  the re-graded it17a bank fixed in both configs (and the high_risk surface value unchanged at 0.704 or recomputed).
- **Ledger:** `gate.ledger check --experiment it19c` exits 0.
- **Mac checks at the launch commit:** `scripts/test.sh --fast`; the L0 sweep; `gate.guards --chains`;
  `node_chain check configs/chains/it19_c_crgrpo.yaml`; S9 = `tests/test_crgrpo.py` and K1
  (`analysis.reward_gate --config crgrpo` on G3's rollouts: PASS on 2026-11-01 after the review fixes, pooled
  blind-decline credit −0.042); `final_benchmark never-trained --archive` (pass, the GRPO pools in scope).
- **Evaluation-path diff** since `634ef8e` limited to the list in the pre-registration section 10.
