# Iteration 17 sign-off

## it17a (owner, 2026-10-28)

- **GPU spend:** "Approve it17a (Recommended)": one GPU VM (48 GB, 24+ cores), the L1 sweep on the DEF-65 harness
  commit and G2's training-pool rollouts with bank (`configs/chains/it17_a_data_rollouts.yaml`), about 4.5-5 h,
  about $5, a watcher that stops the node when the chain ends. The offer is shown to the owner before the launch.
- **Waivers:** "Waive for it17a, it17b (Recommended)": DEF-10, DEF-20, DEF-50, by ID, for it17a and it17b only
  (`gate.ledger check --experiment it17a` / `it17b` exit 0; `it17c` stays shut).
- **Mac checks before launch:** `scripts/test.sh --fast` 5,017 passed (on 5852c44); L0 sweep PASS (36.8 s);
  `confidence_bandit verify-rollout` 0 problems; the job file passes the chain sanity rules.
- **L1:** not run on the Mac (Docker batch work belongs on a node; the DEF-65 harness change invalidates every L1
  key): the job's `gate_l1` step runs it on the node before any episode and stops the chain if it fails.
