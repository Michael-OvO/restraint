# Iteration 18 sign-off

## it18b (owner, 2026-10-29)

- **Pre-registration approved:** "Approve (Recommended)" (docs/iteration-18-prereg.md as of this commit).
- **Waivers:** "Waive for it18b (Recommended)": DEF-10, DEF-20, DEF-50, by ID, for it18b only
  (`gate.ledger check --experiment it18b` exits 0; `it18c` stays shut).
- **GPU spend:** "Restart 58122470 (Recommended)": the it17 node (RTX 5090 32 GB, 64 cores, $1.17/h), about 3.5-4 h,
  about $4-5; a watcher stops it when the chain ends or the drift abort fires.
- **Reference reuse condition (prereg section 5):** `git diff 3cdb17c HEAD` on the evaluation path is empty at launch.
- **Mac checks:** the job file passes `node_chain check`; the trainer, chain and node tests pass (93); the it18 config
  reproduces it17's split exactly (`train.confidence --dry-run`); the config's sha256 is the one the chain checks.
  `scripts/test.sh --fast` 5,031 passed on 127f632 (the one failure, the L0 sweep's timing test, passes alone in 39 s).
