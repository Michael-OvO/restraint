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

- **Node:** "Rent 52960034 (Recommended)": RTX 5090 32 GB, 64 cores, 125 GB RAM, Ontario, $1.11/h.
- **L1 on the DEF-65 harness commit (node 58122470):** PASS, 2026-10-28. G1 oracle gate 120/120 pairs (run fresh
  under the new harness on the first attempt, then cached); S3 metric validity on `it15_g2_surface4` (96/96 audited)
  and `it15_sft_v2_surface4` (94/96), k8 and prefix audits pass, replay bar met; S11 guards and the real egress probe
  pass. Two failed attempts before it, neither a finding: S3's two archived evals were not on the node (fixed in the
  job file, `455bc38`), then one S3 sandbox did not boot (boot failures are never cached; the retry booted it).
