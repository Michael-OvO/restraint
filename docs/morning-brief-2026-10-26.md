# Morning brief, 2026-10-26

Written at 07:40 UTC (03:40 EDT) on `ad4be54` plus this commit. Nothing was pushed. The integrity gate is still
**shut**: 7 ledger entries are not closed, and 3 of them need you.

## Update, 2026-10-26 ~07:30 EDT (after the node batch)

- **Node batch done, node destroyed.** Vast VM 57781978 (UK, 30 vCPU, $0.44/h, about 3.5 h, about $1.6).
  `vastai show instances` is empty. The balance is **$13.89**.
  - **DEF-20:** pair replays of all 2,880 episodes of the three surface runs (960/960 each; 35 did not reproduce
    and fall back to the strict rule). `revealed` now has MDE, ni10 and clone-pass: G2 s1 trained 52.0 %, readme
    82.4 %; MDE 6.2 / 10.1 at K = 20, 3.2 / 5.3 at K = 68 (`docs/research/it16-power-replays.md`). Twin
    feasible-part validity: 48/48 runs correct on 24 real twins. **Only remaining item:** twin task_success's own MDE,
    which the stage-(a) eval supplies (an experiment, so it waits for DEF-10).
  - **DEF-50: blocked on you.** The export reached the gated dataset `Idavidrein/gpqa` and stopped. No Hugging Face
    token is configured on the Mac. Either put a token with GPQA access in `~/.cache/huggingface/token` (`node run`
    forwards it over stdin; the rerun takes minutes on any node), or decide to freeze AbstentionBench without GPQA,
    which is a pre-registration amendment.
- **DEF-59 closed** (`4077af4`): the never-trained check reads `data/commit_checks/builds/*`, with a regression test
  and a re-recorded scope (pass, 62,559 rows).
- **Ledger now:** 54 of 60 closed. Open or fixing: DEF-10, DEF-43, DEF-58 and DEF-60 (your decisions, all inside
  Decisions 1 and 2 below); DEF-50 (your HF token or amendment); DEF-20 (needs stage (a), so it follows DEF-10).
- Everything else below is as written at 03:40 EDT. Ignore the "one instance is running" row: it has been destroyed.

## State this morning

| check | result |
|---|---|
| `scripts/test.sh --fast` | green: 4726 passed, 36 skipped, 100 s |
| `gate sweep --level 0` | S1, S2, S4-S10 PASS, 22.5 s (budget 60 s), 0 boots, 0 bytes written |
| L1 (oracle gate, S3, S11) | PASS on `8d45d25` (remote node, last night). It still holds: `git diff 8d45d25 ad4be54` changes no grader, verifier, harness, task build, reward or surface metric code (details in [iteration-16-signoff.md](iteration-16-signoff.md), "L1 validity") |
| `gate.ledger check` | SHUT: DEF-10, DEF-20, DEF-43, DEF-50 fixing; DEF-58, DEF-59, DEF-60 open |
| Vast instances | **one is running, and I did not destroy it**: 57781978 (1x RTX 3090 VM, UK, $0.44/h). The orchestrating session of the overnight workflow rented it at 07:32 UTC for the DEF-20/DEF-50 node batch, and it was still coming up when I checked (7 min old). It is live work, not a leftover, so I left it alone. If nothing is running on it when you read this: `vastai destroy instance 57781978` |
| Vast balance | **$15.47** credit |

## What ran overnight

All times are EDT. All of it is merged into local `main`.

- **Evening, node L1** (`c8a1cb8`): the full L1 sweep on `8d45d25` ran on Vast VM 57730801 from 19:18 to 20:00 EDT.
  The oracle gate passed 120/120 pairs, and S3 and S11 passed. It cost **$1.04** ($0.90 compute, $0.14 storage). The
  instance was destroyed afterwards.
- **DEF-46 closed** (`0ff3cf0`). The sanity-run guard fails closed. G2 s1's sanity run is enforced in the it16 job
  file. The other five iteration-15 models are superseded and will never be served again.
- **DEF-50 ingest code** (`bc12de7`, `c53264c`, `05c3e56`, merged in `d7ff082`). `final_benchmark` can freeze an
  archived AbstentionBench export and refuses an arm whose export differs from the frozen one. The export itself has
  not run yet (it needs a node).
- **Claim-splice build** (`38673ad`, `343c1a8`, merged in `dabcc09`). This builds matched claim pairs from the cached
  checklist validation, with no Docker. The tracked build `claim_splice_v1` has **0 pairs**. Of the agreed fail
  states, 250 twins have no claim point, and the prefix/state consistency guard drops the rest. The real build waits
  on stage (b)'s transcripts.
- **DEF-20** (`34ebd89`, merged in `9380f91`): ledger update only. It stays `fixing`. The planned node batch (pair
  replays of the three surface runs, plus the twin feasible-part check) **did not run**. The permission system
  refused `vastai create instance` (category "Real-World Transactions"), so that lane spent nothing. Later, the
  orchestrating session rented instance 57779753 (07:09 UTC). It never came up and was destroyed ($0.06 storage).
  Then it rented 57781978, which is the instance above.
- **Iteration-16 job files** (`e32ed26`, `2d5e511`, merged in `65e748e`; renumbered in `ad4be54`). There is one job
  file per paid stage, plus [iteration-16-runbook.md](iteration-16-runbook.md) and a ledger launch guard
  (`gate.ledger check`, the first step of every job; a waiver covers only the chains it names). Writing them turned up
  three new defects: DEF-58, DEF-59 and DEF-60.
- **This morning** (this commit): the sign-off draft and the iteration-16 preparation entry of the training log are
  updated, and this brief is new.

**Node cost overnight: $1.10** ($1.04 for L1, $0.06 for the instance that failed to start), plus whatever 57781978
uses ($0.44/h).

## Closed since the last sign-off draft (`1811eae`, 2026-10-25 15:11)

DEF-46, DEF-48, DEF-51 and DEF-57. That is 53 of 60 entries.

## Not closed, and who closes each

| ID | needs | what |
|---|---|---|
| DEF-10 | **you** | approve the pre-registration (below) |
| DEF-43 | **you** | release the 24 select variants (below) |
| DEF-58 | **you** (decided inside DEF-10) | the pre-registration promises seeded sampling, and no rollout sends a seed |
| DEF-60 | **you** (decided inside DEF-10) | the AbstentionBench guard margin has no calibration source |
| DEF-20 | a CPU node (about 2 h, about $1) | pair replays of the three surface runs (2,880 episodes) and the 48-boot twin feasible-part check |
| DEF-50 | the same node (minutes) | export AbstentionBench on the node, then freeze and check it on the Mac |
| DEF-59 | an agent, locally, $0 | add `data/commit_checks/builds/*/{examples,dpo_pairs}.jsonl` to the never-trained scope |

The orchestrating session may already be running DEF-20 and DEF-50 on 57781978. If the rental is refused again, the
lane's note asks you either to add a Bash permission rule for `vastai create instance` or to rent the node yourself.
The jobs are in the DEF-20 and DEF-50 ledger entries.

## Decision 1: DEF-10, approve the pre-registration

[iteration-16-prereg.md](iteration-16-prereg.md) is a DRAFT. It fixes the arms (base, G2 s1, final = SFT from G2 s1
on matched claim pairs), one estimator (per-unit paired differences, a two-way cluster 90 % CI, and sign-flip tests
by source and by pair), and the bars. The headline is P1 & P2 & P3: TB2 false-claim share down (UB < 0),
ImpossibleBench honest flagging up (LB > 0), and TB2 success non-inferior (LB > −m3, with m3 capped at 7.5 points).
Each guard group is one conjunction that a sampling clone passes at least 90 % of the time (surface 93 %, tier S
92 %, tier F 91 %, probe 97 %). It also fixes checkpoint selection, stopping rules and a decision table. Three things
are open in it:
- The headline margins and the external-guard margins are "computed-at-build" from stage (a)'s calibration samples.
  Those samples are themselves an experiment under the gate. That is a circularity only you can break.
- DEF-58: the draft says every arm uses the same per-(task, sample, turn) seeds, but nothing sends a seed. The power
  calculation assumes independent sampling anyway.
- DEF-60: the AbstentionBench over-abstention guard (`UB < +m`) takes m from calibration samples that never run
  AbstentionBench. MASK has the same problem.

**Recommended answer:** approve the draft with two amendments, and waive DEF-10 for stage (a) only.
1. DEF-58: amend section 2 to say "independently sampled". This matches what the bars were powered for, so no
   number changes. Wiring seeds through `rollout.py` and the Harbor agent is more code to validate, for no power gain.
2. DEF-60: state the AbstentionBench margin in the file now, as a fixed number like P3's cap. The ledger suggests
   7.5 points, but the number is yours to pick. MASK stays out of (d) anyway, because it needs a paid judge. The
   other option is adding a base-arm AbstentionBench run to stage (a), which costs a judge endpoint and more time
   in (a).
3. Write `waived (owner, 2026-10-26; covers it16a)` into DEF-10's table row and its Status line. The calibration
   samples can then run while DEF-10 stays shut for it16b, it16c and it16d. Every other entry must still be closed
   (or waived for it16a) before (a) can start.

## Decision 2: DEF-43, release the 24 select variants

A local `validate` run on 2026-10-14 quarantined 32 select variants. It used the old umask 0000 (DEF-16/19). Every
select and surface eval since iteration 14 has excluded them. Under the nodes' umask 0022, 24 of the 32 (12 sources)
pass their oracle. Their only failure was file-mode asserts such as 0o644 against 0o666. The other 8 still fail
for a real reason: a background process the tests probe is not running. The other side was checked too: all 26
passing variants whose tests assert explicit modes still pass under 0022. Releasing the 24 restores about 11 % of
select, a non-random slice (file-mode tasks). Re-scoring the four it14 select evals moves counts and rates, for
example base task_success 66.3 → 70.3 %. No decision changes: e3 is still the epoch chosen, and the epoch order is
unchanged. The 8-entry list is already tracked on `main` as `data/select_quarantine.txt`.

**Recommended answer: yes, release.** Then DEF-43 can be closed with no further work. If you say no, restore the
32-entry list (archived copy: `~/Documents/honest-agents-archive/iterations/14-session-a/files/data/select_quarantine.txt`)
and relax `tests/tasks/test_select_quarantine.py`. The code fix stays either way.

## The GPU stages: cost and the one launch command each

None of these can start yet. Each one's first step is `gate.ledger check --experiment <name>`, which exits 1 while
any entry is open or unwaived for that chain. Each also needs your go-ahead on its own spend.

The node for every stage is 1x RTX PRO 6000 96 GB, at least 23 vCPU, 64 GB RAM and 200 GB disk, at about $1.46/h.
The estimates come from the runbook and include 0.75 h of boot per rental. Allow 30 % for overrun.

**The balance does not cover a stage.** $15.47 minus the watcher's $8 balance floor leaves about $7.5 of runway,
which is less than stage (a) alone. Top up first: about $25 for (a)+(b), and about $110 for all four with overrun.

Before any launch: stage the inputs (runbook section 2), run `node sync --with-data`, and arm the watcher in the
same minute: `caffeinate -i uv run python -m honest_agents.node watch --host $H --port $P --instance-id $ID
--balance-floor 8 <job file>`.

| stage | what | wall time | cost | launchable once the gate opens? | launch command |
|---|---|---|---|---|---|
| (a) | calibration: base and G2 s1 on TB2 and ImpossibleBench, S = 2, after a 12-task sanity run of each model (this includes the start model's sanity run that sign-off needs) | about 6 h (4.5-8.5) | about $9 ($7-12) | yes | `uv run python -m honest_agents.node chain --host $H --port $P configs/chains/it16_a_calibration.yaml` |
| (b) | G2 s1 own runs on the 400 training-pool surface tasks, S = 4, transcripts kept, then checklist validation | about 6.9 h | about $10 | yes; it can share (a)'s rental | `uv run python -m honest_agents.node chain --host $H --port $P configs/chains/it16_b_claim_rollouts.yaml` |
| (c) | SFT from G2 s1 on the claim-pair build (W&B), two epochs, epoch chosen on the select split | about 5.5 h | about $8 | no: `<SFT_CONFIG>`, `<SFT_BUILD_MANIFEST>` and `<SELECT_PROTOCOL>` are placeholders; needs (b) and DEF-59 | `uv run python -m honest_agents.node chain --host $H --port $P configs/chains/it16_c_sft.yaml` |
| (d) | final eval of base, G2 s1 and final on every pre-registered benchmark except MASK | about 40 h (37-47) | about $58 ($54-69); re-costed from (a)'s job times before it is requested | no: `<SURFACE_VAL_V2>`, `<BLIND_FAMILY_SPLIT>`, `<FLIP_PROMPTS>` and `<ABENCH_JUDGE_URL>` are placeholders; needs (c) | `uv run python -m honest_agents.node chain --host $H --port $P configs/chains/it16_d_final_eval.yaml` |

The total is about 58 h and about $85 ($74-101) before overrun.

## Small things

- I did not touch the branch `design/rts` (worktree `restraint-rts`). It is not a defect fix.
- A timing flake that the ledger does not track: `tests/test_chain_scripts.py::test_serve_and_run_stops_the_server_on_sigint_and_sigterm[2-130]`
  failed once overnight in `--fast`, while the L0 sweep was running at the same time. It passed 5/5 alone, and the
  suite passed when rerun alone. This is the same kind of flake as DEF-21. Give it a ledger entry if it happens again.
