# Iteration 16: integrity sign-off (draft, NOT signed off)

Draft written under [integrity-gate.md](integrity-gate.md) step 3. **The gate is not passed:** the ledger
([defects.md](defects.md)) still has 10 entries that are not `closed` or `waived` (8 `open`, 2 `fixing`), and the
oracle gate and the sanity run have not been run on this build. No experiment may start from this commit. This file
records what was checked and what is true as of the sweep below; the remaining fields are marked as such.

## Recorded

| Field | Value |
|---|---|
| Sweep date | 2026-10-25 (the fix round ran 2026-10-24/25) |
| Commit swept | `1d1cdc0` (main) |
| Eval build | `data/pools/build_manifest.json` sha256 `09628ed96f81f5d4efae7460945d23328bbd186d20083c8ec29116594f72b2c5` (last changed `49e8965`); 20 sources x 6 surfaces, test split, seed 0 |
| `scripts/test.sh --fast` | green: 4401 passed, 34 skipped, 105 s |
| L0 sweep (`gate sweep --level 0`) | S1, S2, S4-S10 PASS; 22.9 s (budget 60 s); 0 boots; 18,793,272 bytes written (S7's temporary rebuild, DEF-22) |
| L1 sweep | not run: the result cache holds no oracle-gate rows (`~/.cache/honest-agents/gate/` has only the replay cache), so L1 would boot all 120 pairs, well over the 30 min allowed for this check |
| S3 (metric validity), S11 (infra guards) | not run by the sweep (DEF-23 open; S11 has no tool) |
| Oracle gate over the full eval build | **not run** on `1d1cdc0`. Earlier gates were stratified samples (see DEF-16, DEF-25) |
| 12-task sanity run of the start model | pending (needs GPU) |
| Waivers in force | none |

## L0 sweep detail

| Check | Status | Detail |
|---|---|---|
| S1 | PASS | 120/120 pairs read one way on both branches |
| S2 | PASS | 12 surface/branch cells over 68 evals; no cell with 0 successes everywhere |
| S4 | PASS | the inspection metric is defined only in `eval/read_diagnosis.py` |
| S5 | PASS | 0 task dirs missing; 3520 high_risk episodes of 8/8 evals in use, 0 unresolved |
| S6 | PASS | a stub episode round-trips commands, exit codes and capped outputs |
| S7 | PASS | eval 240/240 and decision 1048/1048 variants byte-identical; writes 18.8 MB to disk (DEF-22) |
| S8 | PASS | 68 episode files, 1 declared duplicate group, 0 undeclared |
| S9 | PASS (as a check) | it15_g2_surface4: the archived G3 reward FAILS the audit (adversarial and blind below start, break-even >= 0.5); the symmetric reward passes. DEF-13 stays `fixing` |
| S10 | PASS | 10/10 estimator inputs present; L0 resamples <= 2000 |

## Still open (blocks sign-off)

| ID | Status | What remains |
|---|---|---|
| DEF-10 | fixing | the tool is done; the pre-registration its closure test names does not exist yet |
| DEF-13 | fixing | reward design (reopened 2026-10-25); blocks RL only, but it is still not closed or waived |
| DEF-16 | open on main | probe-role surface builds never oracle-gated. A fix and a closure ledger entry sit on the unmerged branch `fix/probe-gate` (worktree `restraint-wt/probe-gate`, 9 commits, head `d2ec433`); not merged, so not closed |
| DEF-19 | open | local sandbox exec umask 0000 (the `fix/probe-gate` umask 022 change is related, also unmerged) |
| DEF-20 | open | three iteration-16 guards have no powered metric |
| DEF-21 | open | timing-dependent forge-reward test in `--fast` |
| DEF-22 | open | L0 S7 writes an 18.8 MB rebuild to disk (confirmed by this sweep) |
| DEF-23 | open | the sweep does not run S3 or S11 |
| DEF-24 | open | the replay cache key misses imported sources and library versions |
| DEF-34 | open | S3 audits a listing reveal as a false positive |

Also outstanding: the full oracle gate over the eval build and the sanity run. The sign-off must be redone on the commit
that closes these entries.

## Unmerged branches

- `fix/probe-gate` (worktree `/Users/michael/Documents/Github/restraint-wt/probe-gate`): DEF-16 fix, unmerged.
- `design/rts` (worktree `/Users/michael/Documents/Github/restraint-rts`): RTS scoring design and code, unmerged. Not a
  defect fix.
