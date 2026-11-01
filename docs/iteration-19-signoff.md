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
