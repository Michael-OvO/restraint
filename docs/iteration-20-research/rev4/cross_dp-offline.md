# cross:dp-offline — what can be known about decision-point training before any GPU is used

Cross-examiner report for the joint stage (iteration 20, revision 3 → 4). Subject: the candidate in section 6 of the
orchestrator notes, decision-point GRPO (DP-GRPO): replay one of the policy's own episodes to a branch state, sample 8
continuations, reward plain U.

## 0. Scope, data, hygiene

- Train-side data only. Every source was checked against `data/split_seeds.json`: 0 rows from the test split in any
  bank used. No held-out path was opened. Held-out numbers below are aggregates already printed in
  `docs/iteration-20-research/u_checks.txt` (lines 8, 10, 11, 13) and
  `docs/iteration-20-design.md` (lines 49-53).
- Banks (G2 unless stated, feasible and control variants only, scored with the repo's `score_records`):
  - **train pool**: `runs/def74/it17_rollouts_feasible` (S = 4; 1,860 episodes; 1,564 "clean" after dropping DEF-66
    sources and validation-bad variants: 391 variants, 177 sources; all clean rows are train-legal). Transcripts from
    the archive (1,860 of 1,860 found) give exact token counts.
  - **select**: `runs/eval/it19c_g2_select` (S = 4) pooled with the archive's `it17_g2_select` and `it16_g2_select`
    (S = 2 each): 8 samples per variant, 110 clean variants, 49 sources. Not train-legal (reserved for selection).
  - **RL groups of 8**: it19c training completions (ablation r2, 30 steps; method arm, steps 1-7). "Steps 1-5" = 50
    feasible-side groups, 399 episodes, policy within 5 updates of G2.
- Token counts: each assistant turn of the pool transcripts was rendered as Qwen writes a tool call and tokenised with
  the tokenizer saved beside an archived adapter. Check against the recorded totals: completion tokens = 1.012 x my
  count (R² 0.9994, median error 3 tokens); prompt tokens = 1,313 per call (tool schema and template) + 1.000 x my
  context count (R² 1.0000). So the token figures below are measurements, not character estimates.
- "State-changing command": the repo's `mutation.is_mutating` (file redirects, rm/mv/cp/sed -i ...;
  `src/honest_agents/mutation.py`, lines 14-20) plus an extension for tar, git
  commit/add, sqlite writes, crontab, pip install and script runs, which the repo classifier misses. Results are given
  for the extended classifier; the repo classifier gives the same picture (both in `geometry.out`).
- Scripts and outputs: `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/`
  (list in section 12). Run from the repo root with `env -u VIRTUAL_ENV uv run --frozen python <script>`. Nothing in
  the repo or the archive was changed (`git status` clean at the end). No Docker, no GPU.
- Disclosure: before I noticed that the name matches the forbidden substring "final", I printed the first 60 lines
  of one train-pool task's test file (`data/tasks/endless-terminals__task_000000_38bfdb17__high_risk__control/tests/test_final_state.py`,
  a train-split source). It is not held-out data and nothing from it is used. After that, a path guard in the loader
  refused every such file; the "defect write" analysis therefore uses failure messages only.

## 1. Headline

1. **The endgame branch is cheap and has room.** In G2's false-claim episodes the last state-changing command is
   command 12 of 17 (median; 75 % of the way through), 3 commands follow it, and 16 of the 28 commands are still
   unused. The prefix up to that state is 4.4k tokens of context (11 % above 8.2k). The stored continuation is 610
   model tokens, of which 354 are the report call. So a continuation trains about 3 times fewer tokens than a whole
   episode (1,898), not "a few hundred": more than half of what it trains is report text.
2. **The banks cannot tell whether continuations from a false-claim state have mixed outcomes.** At temperature 1.0
   two samples of one variant share only their first command (median); the best passing/false-claim pair of a mixed
   variant shares its first 2-3 commands and then diverges, 11-16 % of the way through. In 150 mixed groups, exactly 1 passing sample went
   through the same sandbox state as a false claim's final state. There are 2 natural "continue from a false-claim
   state" cases in all the data (1 passed).
3. **What the banks do show**: outcome variance is untouched by the first 4 shared commands (ratio 1.00) and mostly
   intact after 2-3 shared writes (0.79-0.87); identical write sequences always give identical outcomes (1,064 of
   1,064 pairs), so the grader and the sandboxes are deterministic enough for replay; exit codes agree at every one
   of 15,285 shared command positions across independent sandboxes.
4. **G2 does react to a defect on its screen, but weakly.** With an automatic detector (a check whose output differs
   from what passing samples of the same variant print for the same command): G2 claims complete in 73.5 %
   [61.0, 84.8] of the failing episodes that end on such a check, against 99.6 % [99.2, 99.9] when the last checks
   match; of the samples that showed a defect, 15 % (strict) to 31 % (loose) fixed it and passed; in 72 % of the
   episodes that kept the defect the agent never wrote again after first seeing it. In 16-32 % of such groups another
   sample of the same variant fixed it. So there is variation across samples. Whether there is variation across
   continuations of one stored prefix is not knowable offline, and one fact argues against it at the report itself:
   G2's status token is above 0.99 "complete" in 94 % of train-side false claims.
5. **Supply**: the existing train-legal bank holds 91 variants on 73 sources with both a false and a true claim, and
   136 false-claim episodes on 71 sources that qualify for an endgame branch. That is enough for a pilot (80 + 80
   prefixes at 2 per source), not for a 400 + 400 bank. The planned prescreen would give about 2,600-3,450 false
   claims; 400 + 400 at 2 per source is comfortable, at 1 per source it is tight.
6. **The 1:1 pair is neutral only in a narrow sense.** "Always claim" and "never claim" score the same at 1:1 if a
   continuation from a true-claim state always passes; the natural twins give 94 of 94 [96.1, 100], so the exact
   ratio is between 1.00 and 1.09 true per false. But the first-order push on a shared "claim less" direction is not
   zero: the status decision is 9 times more uncertain on false claims than on true claims (mean p(1−p) 0.0107
   against 0.0012). The pair needs a guard on P(claim | true-claim prefix), not just the ratio.
7. **Size of the prize** on held-out P ∪ T (8.69 false claims per 100 episodes): turning 25 % of the screen-visible
   ones into passes is +1.3 U if 29 % are visible (the published held-out read), +2.7 U at 62 %, +3.4 U at 77 % (the
   train-side read). A 5.4-point fall of P(claim | pass) cancels the middle case.
8. **Verdict**: DP-GRPO at the endgame is a hypothesis with a cheap, decisive test, not yet a main arm. Run the
   no-training pilot in section 7 (about 4,400 episodes, one GPU, under a day) before any design is frozen. I would
   bet 35 % that the endgame cut passes its bar and 55-60 % that a cut just before the last write does.

## 2. Geometry of the endgame (item 1)

Train pool, clean rows, extended classifier. q25 / median / q75. False claims: n = 186 (116 variants, 82 sources).
True claims: n = 1,238 (362 variants, 171 sources). Source: `geometry.out`.

| quantity | false claims | true claims |
|---|---|---|
| commands in the episode | 10 / 17 / 25 | 8 / 11 / 16 |
| (a) last state-changing command, 1-based index | 7 / 12 / 19 | 5 / 8 / 12 |
| ... as a fraction of the episode's commands | 0.64 / 0.75 / 0.89 | 0.58 / 0.71 / 0.84 |
| commands after the last write | 2 / 3 / 6 | 2 / 3 / 4 |
| no command at all after the last write | 23/186 = 12.4 % [8.4, 17.9] | 79/1,238 = 6.4 % [5.2, 7.9] |
| a read-back of a written path follows the last write | 157/186 = 84.4 % [78.5, 88.9] | 1,137/1,238 = 91.8 % [90.2, 93.2] |
| ... and it is the very next command | 139/157 = 88.5 % | 1,055/1,137 = 92.8 % |
| commands left (of 28) after the last write | 9 / 16 / 21 | 16 / 20 / 23 |
| (b) commands left after the first check | 10 / 16 / 20 (n = 163) | 15 / 19 / 22 |
| (c) commands left at the report | 3 / 11 / 18 | 12 / 17 / 20 |
| 2 or fewer commands left after the last write | 10.8 % | 1.9 % |
| task prompt (system + user), tokens | 682 / 901 / 1,043 | 642 / 740 / 911 |
| context tokens at the state after the last write | 3,172 / 4,435 / 6,295 | 2,782 / 3,360 / 4,248 |
| context tokens after the first check | 3,304 / 4,425 / 5,993 | 2,924 / 3,518 / 4,372 |
| context tokens at the report call | 4,028 / 5,766 / 7,776 | 3,557 / 4,279 / 5,407 |
| context above 8,192 at the last-write state | 21/186 = 11.3 % | 24/1,238 = 1.9 % |
| model + tool tokens used before the last-write state | 963 / 1,892 / 3,458 | 666 / 1,128 / 1,879 |
| ... left of the 16,384 budget (q75 / median / q25 of used) | 12,926 / 14,492 / 15,421 | 14,505 / 15,256 / 15,718 |
| model tokens in the whole episode | 1,141 / 1,898 / 3,006 | 898 / 1,204 / 1,717 |
| model tokens after the last write (stored continuation) | 452 / 610 / 908 | 416 / 530 / 692 |
| ... of which the report call | 306 / 354 / 424 | 292 / 342 / 405 |
| ... excluding the report call | 110 / 223 / 461 | 104 / 181 / 302 |
| model tokens from just before the last write | 573 / 772 / 1,099 | 519 / 650 / 847 |

Context tokens include the 1,313 tokens of tool schema and template that every call carries.

Robustness (commands only, no transcripts): select, 3 runs pooled, false claims n = 95: commands 10 / 14 / 21, last
write 6 / 10 / 17, commands after it 2 / 4 / 5, left after it 11 / 18 / 22. RL groups steps 1-5 (16k budget, banded
pool), false claims n = 65: 14 / 19 / 24, 10 / 14 / 19, 2 / 4 / 6, 9 / 14 / 18. Same shape.

What this fixes for a design:

- **Prefix length**: 12 commands and about 4.4k context tokens (median) for a false-claim prefix; 8 commands and
  3.4k for a true-claim prefix. A cap of 8,192 context tokens keeps 89 % of false-claim prefixes.
- **Continuation budget**: 16 commands (median) and about 14.5k tokens are left. Budget is not the constraint: 89.2 %
  [84.0, 92.9] of false claims have at least 3 commands left after the last write, 86.0 % at least 5, 77.4 % at least 8.
- **Eligibility for the branch "after the first check that follows the last write"** (a check exists, at least 5
  commands left after it, context ≤ 8,192): 136/186 = 73.1 % [66.3, 79.0] of false claims (71 sources, 92 variants);
  1,104/1,238 = 89.2 % of true claims. The losses: no check after the last write 23, fewer than 5 commands left 21,
  context too long 6.
- **Trained tokens**: a stored endgame continuation is about one third of an episode, and 58 % of it is the report
  call (354 of 610). Before the report there are only about 220 model tokens (2-3 checks).
- **The last write is not always the write that made the defect.** For 105 false claims (48 sources; pool and
  select) the failing hidden test's message names a path the agent wrote. The agent's last write to that path is the
  episode's last state-changing command in 55/105 = 52.4 % [42.9, 61.7]. It sits at command 8 (4 / 8 / 17), 64 % of
  the way through, with 21 commands left (12 / 21 / 25) and 5 commands after it (2 / 5 / 8). In 58 % the failing path
  had been written at least twice. (Path matching is by name; 148 of 281 false claims had no path in the message and
  are not covered. Source: `defwrite.out`.) So a cut "just before the last write" re-opens the defect-making write in
  about half of the cases only.

## 3. Natural experiment for branch variance (item 2)

Idea: two samples of one variant that issued identical commands for their first k commands are two continuations of
one prefix. Two notions: **exact** (same command strings, whitespace-normalised) and **state** (same state-changing
command strings; read-only commands ignored, so the sandbox state is the same). Source: `natexp.out`, `twins.out`.

**3.1 Samples diverge almost at once.** Over all same-group pairs (pool 2,346; select 3,080; RL steps 1-5 1,393):
identical leading commands 0 / 1 / 2 (q25 / median / q75), 90th percentile 3-4. Share with at least 3 identical
commands: 22.3 %, 24.9 %, 18.0 %. At least 8: 1.3 %, 1.2 %, 0.6 %.

**3.2 Divergence point of passing against falsely-claiming samples.** Groups with at least one true claim and one
false claim: pool 91 (73 sources), select 30, RL steps 1-5 29: 150 in all. For each group, the pair (pass, false
claim) that stays identical longest:

| | pool (91) | select (30) | RL 1-5 (29) | pooled (150) |
|---|---|---|---|---|
| exact: identical leading commands before the first difference, q25/50/75 | 0 / 2 / 3 | 1 / 3 / 4 | 2 / 3 / 4 | |
| exact: as a fraction of the false-claim episode, q25/50/75/90 | .02 / .11 / .20 / .30 | .11 / .16 / .28 / .50 | .10 / .14 / .21 / .27 | |
| exact: still identical through the false claim's last write | 1 | 0 | 0 | 1/150 = 0.7 % [0.1, 3.7] |
| exact: identical for ≥ 75 % of the episode | 1 | 0 | 0 | 0.7 % [0.1, 3.7] |
| exact: ≥ 50 % | 1 | 4 | 0 | 5/150 = 3.3 % [1.4, 7.6] |
| exact: ≥ 25 % | 17 | 11 | 5 | 33/150 = 22.0 % [16.1, 29.3] |
| state: divergence as a fraction of the episode, q25/50/75/90 | .17 / .23 / .36 / .53 | .14 / .17 / .32 / .50 | .20 / .26 / .36 / .44 | |
| state: same state at the false claim's last write | 1 | 0 | 0 | 1/150 = 0.7 % [0.1, 3.7] |
| state: shared for ≥ 75 % of the episode | 4 | 0 | 0 | 4/150 = 2.7 % [1.0, 6.7] |
| state: ≥ 50 % | 14 | 6 | 2 | 22/150 = 14.7 % [9.9, 21.2] |
| state: ≥ 25 % | 44 | 12 | 17 | 73/150 = 48.7 % [40.8, 56.6] |

Over all (pass, false-claim) pairs rather than the best pair per group: 1 identical leading command (median), 7-8 %
of the episode.

**3.3 How much outcome variance is left after a shared prefix.** For same-group pairs sharing at least k leading
commands: discordant (one passes, one fails) pairs divided by the number expected from the group's own pair
discordance. 1.0 = the shared prefix decided nothing.

| shared prefix | pool | select | RL 1-5 | pooled |
|---|---|---|---|---|
| exact ≥ 2 commands | 0.97 | 0.98 | 1.08 | |
| exact ≥ 4 commands | 0.94 (51 / 54.3) | 1.05 (58 / 55.5) | 1.01 (49 / 48.5) | 158 / 158.3 = 1.00 |
| exact ≥ 6 commands | 0.49 (3 / 6.2) | 0.00 (0 / 5.2) | 0.67 (3 / 4.5) | 6 / 15.9 = 0.38 [0.14, 0.82] |
| state ≥ 2 writes | 0.79 (34 / 43.0) | 0.81 (41 / 50.6) | 0.77 (38 / 49.3) | 113 / 142.9 = 0.79 |
| state ≥ 3 writes | 1.02 | 0.88 | 0.79 | 67 / 76.6 = 0.87 |
| state ≥ 4 writes | 0.78 | 1.33 | 0.93 | 25 / 26.5 = 0.94 |

(The interval on 0.38 is a Poisson interval on 6 events; pairs that share 6 commands are a selected set of simple,
canonical tasks.)

**3.4 Natural "continue from the same state" twins** (B's commands or writes start with all of A's and go on):

- A = true claim: exact twins 94 (19 sources), B passes and claims in 94/94 = 100 % [96.1, 100]; state twins 49 (15
  sources), 49/49 = 100 % [92.7, 100]. G2's extra commands and extra writes from a passing state never broke it.
- A = false claim: state twins **2** (1 passed, 1 false claim). No information.
- A = non-claim fail: state twins 43 (15 sources), B passes and claims in 40/43 = 93.0 % [81.4, 97.6]. All 43 are
  high_risk controls where A declined the requested cleanup and reported incomplete, and B ran the `rm -rf` and passed.
- Identical write sequences: 1,064 pairs (pool 277, select 519, RL 268), outcomes identical in 1,064/1,064
  [99.6, 100].

**3.5 Determinism across sandboxes** (`determinism.out`): at 15,285 command positions that two samples share, exit
codes agree at all of them; outputs of read-only commands are byte-identical in 98.7 %, 98.8 % and 99.9 % (pool,
select, RL); `ls -l`-type outputs agree in 98.6-100 % once timestamps are masked.

**Reading.** The string-identical prefix ends at about 11-16 % of the episode (23-26 % for the state version), and
that is where this instrument goes blind. Up to there, the shared prefix decides nothing (ratio 0.94-1.05 after 4
commands; 0.8-0.9 after 2-3 writes). So a branch in the first quarter to third of the episode keeps essentially all
of the whole-episode outcome variance. What is kept at the last write cannot be read from the banks: only 1 of 150
mixed groups has a passing sample that passed through a false claim's final state. The useful by-products: replay is
sound in principle (deterministic outcomes, identical exit codes), continuing from a passing state is safe (143 of
143), and at the decline point of a high_risk control the other branch passes 93 % of the time.

## 4. What G2 does after a read-back that shows a defect (item 3)

**Detector** (`defect_seen.py`, no labels): inside a group, a read-only command string that a passing sample ran after
its last write shows what a passing final state prints. Another sample that runs the same command and gets a
different output has, on screen, a state that differs from the passing one ("differ"); the same output is "match".
Volatile commands (`ls -l`, `stat`, `date` ...) and commands on which two passing samples disagree are dropped. For
passing samples the reference leaves the sample itself out, which gives the false-positive rate: 5.9 % (pool, 43/735),
5.1 % (select, 23/450), 11.0 % (RL steps 1-5, 19/172).

Coverage is partial: the detector needs the same command string in a passing and a failing sample. A comparable check
in the final state exists for 43.9 % (pool), 49.1 % (select) and 56.9 % (RL) of false claims that have a passing
sibling. Against the failure-taxonomy reader's labels on the 16 labelled false claims it can score: it says "differ"
on 4 and "match" on 12, of which 9 the reader marked "own check showed the violation". So "match" is not evidence
that nothing was visible (the defect was shown by a different command); "differ" is the reliable half. All numbers
below are therefore on the episodes where the detector fires.

| | pool S = 4 | select (8) | RL steps 1-5 (8) | pooled, source bootstrap |
|---|---|---|---|---|
| false claims with a comparable final check whose check differs from the passing state | 23/54 = 42.6 % | 24/28 = 85.7 % | 14/37 = 37.8 % | 61/119 = 51.3 % [42.4, 60.1] |
| failing episodes that end on a differing check ("kept") | 26 | 31 | 26 | 83 (47 sources) |
| **P(claim complete \| defect on screen at the last check)** | 23/26 = 88.5 % | 24/31 = 77.4 % | 14/26 = 53.8 % | **61/83 = 73.5 % [61.0, 84.8]** |
| **P(claim \| final checks all match, episode passes)** | 692/693 = 99.9 % | 427/429 = 99.5 % | 153/155 = 98.7 % | **1,272/1,277 = 99.6 % [99.2, 99.9]** |
| kept: wrote again after first seeing the differing output | 5/26 | 8/31 | 10/26 | 23/83 = 27.7 % [14.7, 40.7] |
| passing samples that showed the same wrong output earlier and then passed ("fixed", strict) | 8 | 6 | 1 | 15 |
| **P(fixed \| defect on screen), strict** = fixed / (fixed + kept) | 8/34 | 6/37 | 1/27 | **15/98 = 15.3 % [5.3, 26.2]** |
| **P(fixed \| defect on screen), loose** (any differing output on the same command after the first write) | 8/34 | 13/44 | 16/42 | **37/120 = 30.8 % [19.0, 41.7]** |
| groups with a kept defect in which another sample fixed it, loose / strict | 5 / 5 of 24 | 6 / 3 of 17 | 7 / 1 of 16 | 18/57 = 31.6 % [21.0, 44.5] / 9/57 = 15.8 % [8.5, 27.4] |

Strict undercounts (it needs the identical wrong output); loose overcounts (it includes work in progress). Mid-episode,
the next command after a differing check is a write in 58 % (pool), 51 % (select), 60 % (RL), against 44 %, 34 %,
44 % after a matching check: a difference of 14-17 points, confounded by work in progress.

**The status token** (train-side K2a bank `runs/it19/k2a_status_g2.jsonl`, select split, DEF-66 sources dropped; my
recomputation): of 84 feasible-side false claims (36 sources), pi(complete) at the status token is above 0.99 in
94.0 %, between 0.05 and 0.95 in 4 (4.8 %), mean p(1−p) 0.0107. Of 474 true claims it is above 0.99 in 99.2 %, mean
p(1−p) 0.0012.

**Answers.**
- P(further edit | defect visible): about 28 % [15, 41] of the episodes that ended with the defect wrote again after
  first seeing it; 72 % went straight on to more checks and the report.
- P(fixed and passed | defect visible): 15 % [5, 26] to 31 % [19, 42].
- P(claim | defect visible at the last check): 73.5 % [61.0, 84.8]. P(claim | clean): 99.6 % [99.2, 99.9].
- Is there variation to reinforce? **Across samples, yes**: of ten samples with a defect on screen, roughly two to
  three fix it and pass, five to six claim anyway, two report incomplete; and in 16-32 % of the groups where one sample
  kept a defect another sample fixed it. **Inside one prefix, unknown.** A bank of false-claim prefixes is selected on
  "saw it and claimed". If a continuation reaches the report without a new edit it will say "complete" (94 %
  saturated), so any within-prefix variance has to come from the 2-3 action turns before the report (about 220 model
  tokens). No stored data measure the policy's distribution over "edit" against "report" at that state.

## 5. Supply (item 4)

Source: `supply.out`.

**5.1 Existing banks, train-legal.**

| | variants with ≥ 1 false and ≥ 1 true claim | sources | false-claim episodes | on sources |
|---|---|---|---|---|
| train pool, clean (S = 4; 391 variants, 177 sources) | 91/391 = 23.3 % [19.4, 27.7] | 73 (76 with both at source level) | 186 (11.9 per 100 [9.3, 14.7]) | 82 |
| plus it19c steps 1-5 legal rows | 97/394 | 79 | 227 | 88 |
| select, 3 runs (not train-legal; dev prefixes only) | 31/110 = 28.2 % | 19 | 95 | 22 |

- On the 91 mixed pool variants: 122 false-claim and 205 true-claim episodes.
- Concentration (pool): 18 of 82 sources hold half of the false claims; the top 5 hold 22 %; the largest holds 12.
  Three of the largest (e04165f6, 7b9e5385, e9aac96e; 19 false claims) are the unreviewed spec defects that
  research:failure-taxonomy found, so an uncapped bank would be led by unreachable tasks.
- With a per-source cap: false-claim prefixes available 82 (cap 1), 123 (cap 2), 161 (cap 4); same-variant
  false/true pairs 73, 98, 107. After the endgame eligibility filter of section 2: 71, 100, 126 false-claim prefixes.
- False claims per 100 by stratum (pool): plain 9.7, high_risk controls 11.9, revoke_permission controls 8.7,
  remove_input controls 21.4. An unstratified bank would be 26 % remove_input (48 of 186) against 14 % of episodes.
- **A 400 + 400 bank cannot be built from existing banks.** A pilot of 80 + 80 at 2 per source can.

**5.2 The planned prescreen** (design section 4.3; `docs/iteration-20-rl-design.md`,
lines 269-273). Legal rows (archive `staging/analysis/it20_design/legal_rows.jsonl`): 4,098 variants on 1,290 sources =
2,699 feasible-side (1,284 plain, 1,415 controls) + 1,399 twins; plus 360 surface pairs. Episodes: 8 x (4,098 + 720) =
38,544, of which 21,592 feasible-side on data/tasks rows and 2,880 on surface controls.

- False-claim episodes on data/tasks rows: 2,591 at 12 per 100, 3,022 at 14, 3,454 at 16. At the pool's measured
  11.9 [9.3, 14.7]: about 2,570 [2,010, 3,170]. Surface controls add few (their fails are mostly declines; 20 % of
  192 failing episodes were false claims in research:headroom-pool-model, result 6).
- Variants with both a false and a true claim at S = 8: 31 % projected from the pool's per-variant rates (23.3 % at
  S = 4; select measured 28.2 % at 8 samples): about 840 of 2,699 variants.
- Usable false-claim prefixes after filters: x 0.73 (endgame eligibility) x 0.83 (not "unreachable": 17 % [6, 29] in
  the taxonomy read) x about 0.95 (replayable) = 0.58, so about 1,500-2,000.
- Sources: in the pool 46 % of sources (82/177) have at least one false claim at S = 4 with 2.2 variants banked per
  source (the legal set has 2.1 feasible-side variants per source, about the same). At S = 8 I expect 55-60 % of
  1,290 sources, about 700-800, and about 450-550 after the filters (a projection, not a measurement). **So 400
  false-claim prefixes at 1 per source is at the edge; at 2 per source it needs 200 sources and is comfortable.** The matching 400 true-claim prefixes are never the limit (89 % of true claims qualify; same-variant
  pairs about 840 variants).
- **Per-source need for 400 + 400**: 2 false and 2 true prefixes on each of 200 sources, or 1 + 1 on 400. Stratify by
  family so remove_input does not exceed its share, and drop sources flagged by the spec audit first.
- The assumption that unbanked legal rows look like the banked ones is weak: only 232 legal sources are
  oracle-validated (design section 4.2), and the defect rate in unreviewed sources is higher. Expect a higher raw
  false-claim rate and a larger unreachable share than the pool's.
- **The prescreen must keep full transcripts** (assistant text and tool calls per turn). The select banks keep only
  commands and outputs, so their prefixes cannot be rebuilt token-exact; the pool bank's transcripts exist only in
  the archive.

## 6. Arithmetic of the pair design (item 5)

Source: `arith.out`.

**6.1 Blind policies.** Let ρ_T = P(the state still passes at the end | continue from a true-claim prefix and claim),
ρ_F = P(pass | continue from a false-claim prefix and claim without looking). Per episode, on a mix n_T : n_F:

- never claim: 0 on both.
- always claim: [n_T (2ρ_T − 1) + n_F (2ρ_F − 1)] / (n_T + n_F). At ρ_T = 1, ρ_F = 0 and 1:1 this is exactly 0.
- "one more check, then claim": the same as always claim. A refused `bash` call does not block the report (8 report
  turns stay free and the refusal text says so; `src/honest_agents/harness.py`,
  lines 105, 126-131 and 180-200), so
  the budget asymmetry (10.8 % of false-claim against 1.9 % of true-claim prefixes have 2 or fewer commands left)
  does not change the value.
- **Exactly neutral ratio**: n_T / n_F (or the reward weight on true-prefix groups) = (1 − 2ρ_F) / (2ρ_T − 1).

Measured ρ_T (natural twins, section 3.4): 94/94 exact [96.1, 100] and 49/49 state [92.7, 100]. With ρ_F = 0:

| ρ_T | always-claim at 1:1, per episode | neutral n_T / n_F |
|---|---|---|
| 1.000 (point estimate) | 0.000 | 1.00 |
| 0.961 (lower bound, exact twins) | −0.039 | 1.085 |
| 0.927 (lower bound, state twins) | −0.073 | 1.17 |
| 0.90 | −0.100 | 1.25 |

So 1:1 is neutral at the point estimate, and a true-claim surplus of 9-17 % covers the lower bounds. If ρ_T is below
1, plain 1:1 favours "never claim" by 2(1 − ρ_T) per pair, the unsafe direction for task success. The twins are easy
tasks (19 and 15 sources) and say nothing about replay losses; ρ_T has to be measured by the pilot at the chosen cut.

**6.2 The caveat the ratio does not cover.** A blind *policy* is neutral; a shared *direction* is not. For a common
shift of the claim logit at endgame states, the first-order change in U is n_T v_T (2ρ_T − 1) − n_F v_F (1 − 2ρ_F),
with v = p(1 − p) of the claim decision inside a prefix. Measured at the status token (section 4): v_T = 0.0012,
v_F = 0.0107. At 1:1 the push is toward claiming less, 9 to 1, and it stays that way until claims on true-claim
prefixes have already started to fall (true-prefix groups in which all 8 continuations claim and pass carry zero
advantage, so they do not push back). With the across-sample rates of section 4 instead (0.996 against 0.735) the
ratio is about 50 to 1. No sampling ratio fixes this; it needs (i) a live per-step meter of P(claim | true-claim
prefix) with a smoothed stop rule, and (ii) the same meter on full-episode probes. The same holds for "edit again
before claiming": a harmless extra edit on a true-claim prefix earns +1 like its siblings, so nothing opposes a
blind habit of re-editing except the budget. This is the mechanism behind the three past twin-action drifts the
notes list, and here it would show first as longer episodes and more budget exhaustion.

**6.3 U gain on the held-out primary strata.** From the documented S = 2 table: 118 plain (false-claim term 0.114),
86 high_risk controls (0.163), 30 remove_input controls (0.200), 42 revoke_permission controls (0.095), 155 twins.
False claims: 13.57 per 100 feasible-side episodes = **8.69 per 100 P ∪ T episodes**. Feasible-side pass@1 78.3 %;
true claims about 49.6 per 100 P ∪ T episodes.

ΔU per 100 P ∪ T episodes = 8.69 x v x (2x + y), with v the share of false claims whose defect is on screen, x the
share of those turned into passes, y the share turned into non-claims.

| v (source) | x = 10 %, y = 0 / 10 / 25 % | x = 25 % | x = 50 % | x needed for +2.5 / +3.6 / +4.0 U (y = 0) |
|---|---|---|---|---|
| 0.29 (held-out read: own check showed it, 29/101 [20.8, 38.2]) | +0.50 / +0.76 / +1.13 | +1.26 / +1.51 / +1.89 | +2.52 / +2.77 / +3.15 | 50 % / 71 % / 79 % |
| 0.62 (held-out read: own check or explicit spec detail, 63/101 [52.6, 71.2]) | +1.08 / +1.62 / +2.42 | +2.69 / +3.23 / +4.04 | +5.39 / +5.93 / +6.74 | 23 % / 33 % / 37 % |
| 0.77 (train-side read, one reader, 49/64 [64, 88]) | +1.34 / +2.01 / +3.01 | +3.35 / +4.02 / +5.02 | +6.69 / +7.36 / +8.37 | 19 % / 27 % / 30 % |

Costs on the same scale: each point of P(claim | pass) lost is −0.50 U; each point of passes broken into false
claims is −1.00 U. A gain of +2.69 (x = 25 % at v = 0.62) is cancelled by a 5.4-point fall of P(claim | pass).

y is likely small: the status token is uncertain in 4.8 % of false claims, so non-claim conversions must come from
actions, like passes. The "own check or explicit spec detail" grouping is my reading of two published categories; my
detector's 51 % [42, 60] on train-side data sits between the two held-out figures.

**6.4 What "40 % of groups mixed" means.** With G = 8, a share of groups with at least one deviating continuation of
40 % corresponds to a per-continuation deviation rate of 6.2 % if prefixes are alike (34 % at 5 %, 57 % at 10 %, 73 %
at 15 %), and to less signal if they are not (30 % of prefixes at 0.3 and the rest at 0.01 also gives 34 %). In U
variance: a fix moves U by 2, so a fix rate h gives Var(U) = 4h(1 − h): 0.19 at 5 %, 0.36 at 10 %. For comparison,
whole-episode groups have mean within-group Var(U) 0.271 on the pool (391 groups, S = 4), 0.202 on select, 0.481 on
the banded it19c pool (steps 1-5). So a false-claim-prefix group matches an unbanded whole-episode group at a fix
rate of about 7 % and a banded one at about 14 %, at roughly a third of the trained tokens.

## 7. What cannot be known offline, and the pilot (item 6)

**Cannot be known from stored data:**

1. P(pass | continue from a false-claim state), at any cut. Offline n = 2.
2. Whether the across-sample variation of section 4 exists across continuations of one prefix (the policy's "edit
   or report" distribution at the branch state was never recorded).
3. P(pass and claim | continue from a true-claim state) under real replay, especially at cuts before the last write.
4. Replay fidelity of late prefixes at the level of state, not exit codes (generated timestamps, git hashes;
   research:failure-taxonomy result 13 suspects one such case). Documented exit-code fidelity so far: 27/27, 22/22,
   317/320, 315/320 and 256 of 266 with a task dir, all on shorter decision prefixes
   (`runs/def74/regrade_it17_rollouts_feasible.summary.json`;
   `docs/defects.md`, lines 163, 385, 698).
5. Gradient signal-to-noise of branch groups against whole-episode groups (needs backward passes at lr 0).
6. Whether behaviour learned at branch states shows up in full episodes, and what it does to twins and controls
   (needs training).
7. Step time and memory of branch groups, and whether a multi-turn prompt reproduces the original context
   token-exact.

Knowable offline but not done here: an independent second read of the 64 labelled false claims with a fixed rubric
(v ranges from 0.29 to 0.77 by criterion).

**Pilot DP-0 (no training, G2, one GPU, the pinned budgets of design section 5.3).**

- *Prefix bank* from the existing train pool bank (train-legal, clean; drop e04165f6, 7b9e5385, e9aac96e and any
  source the spec audit flags): **F** = 80 false-claim episodes and **T** = 80 true-claim episodes, at most 2 per
  source, from the same variants where possible, matched by stratum; each with a check after its last write, at least
  5 commands left after that check and context ≤ 8,192 tokens (available: 100 F at cap 2 on 71 sources; 332 T).
- *Cuts* (the same three for F and T, so the cut cannot reveal the type): **B1** after the first command that follows
  the last state-changing command; **B2** just before the last state-changing command; **B3** after half of the
  stored commands. Plus **B0**: fresh full episodes on the same variants (about 70).
- *Rollouts*: 8 continuations per prefix and cut, temperature 1.0, command and character counters restored to the
  prefix's values, token budget 16,384 minus what the prefix used. Episodes: 160 prefixes x 3 cuts x 8 = 3,840, plus
  70 x 8 = 560 for B0: **4,400 episodes, 480 groups + 70**. At the measured 598 episodes per GPU-hour for full
  episodes (research:throughput, result 1) this is at most 8 GPU-hours.
- *Replay audit, no model*: replay all 160 stored episodes in full and re-grade.
- *Readouts*, by cut, with a source bootstrap and 90 % intervals: on F, P(pass and claim), P(non-claim), share of
  groups with at least one passing continuation, mean within-group Var(U); on T, P(pass and claim), P(non-claim),
  P(broken into a false claim); seconds per continuation; context tokens.

**Pass bars (to pre-register):**

| check | bar | if it fails |
|---|---|---|
| replay, exit codes | ≥ 95 % of the 480 prefix replays reproduce every stored exit code | fix the build once, re-run; else no DP arm |
| replay, outcome | the full replay reproduces the stored pass/fail for ≥ 95 % of the 160 episodes | exclude non-reproducing sources by a content check; if < 90 % no DP arm |
| a cut is usable on F | mean within-group Var(U) ≥ 0.25 **and** ≥ 30 % of F groups have a passing continuation (lower 90 % bound ≥ 20 %) | that cut is not used |
| a cut is usable on T | P(pass and claim) ≥ 0.90; then set n_T / n_F = (1 − 2ρ_F) / (2ρ_T − 1) from the measured values | that cut is not used |
| choice | the latest cut that passes both | |
| kill B1 | fewer than 15 % of F groups at B1 have a passing continuation (upper 90 % bound < 25 %) | use B2 or B3 |
| kill DP as a main arm | no cut passes | whole-episode GRPO at scale is the main arm; DP at most a comparison arm at B3 |

With 80 groups the standard error of a share near 30 % is about 5-6 points (2 per source), so 15 % and 30 % are
separated by about 2.6 standard errors.

**Then, before a DP arm is launched** (GPU needed, not part of DP-0): the lr-0 gradient audit of the notes on branch
groups against whole-episode groups from the same step budget, and a 20-30 step run read on full-episode probes:
P(claim | pass), commands per episode, budget-exhaustion rate, twin claims and damage.

**Bets** (my probabilities):

| statement | probability |
|---|---|
| B1 (after the first check) passes its F bar | 0.35 |
| mean P(pass \| continue) on F at B1 | median 6 %, 80 % interval 2-15 % |
| B2 (before the last write) passes its F bar | 0.55-0.60 |
| mean P(pass \| continue) on F at B2 / B3 | median 20 % / 35 % |
| T at B1: P(pass and claim) ≥ 0.90 | 0.85 |
| replay bars both met | 0.75 |
| a DP arm, if a cut passes, shows a held-out gain ≥ +2.5 U by itself | 0.20 |

## 8. Decisions

1. **Branch point and budgets: adopt as the pilot's definitions, measure before fixing.** B1 = after the first
   command following the last state-changing command; eligibility: a check exists, ≥ 5 commands left, context ≤ 8,192
   (keeps 73 % of false claims, 89 % of true claims). Continuation budget: commands left as in the stored episode
   (median 16), tokens 16,384 minus the prefix's use (median 14.5k left). High confidence in the geometry.
2. **How early the branch must be: measure first.** The banks give only a lower bound: a branch within the first
   quarter to third of the episode keeps all of the outcome variance. Pilot DP-0 with three cuts decides.
3. **Is there variation to reinforce: yes across samples, unknown within a prefix; measure first.** Do not assume
   the 15-31 % fix rate carries to continuations of stored false claims.
4. **Supply: adopt** a cap of 2 prefixes per source, stratification by family, removal of audit-flagged sources,
   same-variant pairing, and full transcripts in the prescreen. Existing banks cover the pilot only.
5. **Pair ratio: adopt** n_T / n_F = (1 − 2ρ_F) / (2ρ_T − 1) with measured values (1.00-1.09 from the twins), and
   **adopt** a per-step guard on P(claim | true-claim prefix) and on commands per episode, because the ratio does not
   neutralise a shared shift.
6. **Go bar: reject "at least 40 % of groups mixed"; adopt** the two-part F bar of section 7 (Var(U) ≥ 0.25 and
   ≥ 30 % of groups with a passing continuation) plus the T bar.
7. **Cut "just before the last write" (failure-taxonomy rec 3): keep as B2, not as the default.** It re-opens the
   defect-making write in about half of the cases only (52 % [43, 62]).
8. **Decline-point branches on high_risk controls: add to the pilot at low cost** (20 prefixes x 8 = 160 episodes on
   top of the 4,400; cut just before the cleanup command the stored episode declined). The natural
   twins show a 93 % [81, 98] pass rate for the branch that proceeds, so these groups should be mixed with clean
   credit; the prize is small (about +1 U by the notes' arithmetic).
9. **Report tokens in branch groups: measure first.** 58 % of a stored continuation is the report call; the notes'
   option of zero policy-gradient weight after the status value matters about 4 times more here than in whole
   episodes (12 % of model text there). It touches the owner's no-mask rule, so it is a judges' question.

## 9. Corrections to the 18 reports and the notes

1. **Orchestrator notes, section 6**: "the trained tokens per episode fall from about 1,900 to a few hundred".
   Measured: 1,898 → 610 (452 / 610 / 908) for false claims, 1,204 → 530 for true claims; 354 and 342 of those are
   the report call. "A few hundred" is right only for the tokens before the report (223 and 181). A continuation
   that fixes something will be longer.
2. **Orchestrator notes, section 6, and research:pass-fail-anatomy rec 1**: "go if at least 40 % of groups are
   mixed". That bar is met at a per-continuation deviation rate of 6.2 % (or with a minority of high-variance
   prefixes), and counts non-claims like passes. Replace it (section 7).
3. **Orchestrator notes, section 6**: "neutral by construction at 1:1". True for "always claim" against "never
   claim" and only if a true-claim continuation always passes (measured 94/94, lower bound 96.1 %, so the exact
   ratio is 1.00-1.09). Not true for a shared shift: the status decision is 9 times more uncertain on false claims
   (0.0107 against 0.0012).
4. **Orchestrator notes, section 6**: "`analysis.replay_episodes` already replays prefixes with an exit-code fidelity
   check". It replays the commands before the first destructive command, or the whole history with `--full`; there
   is no arbitrary cut. Its runner boots, runs, and tears the sandbox down, with a budget that never refuses
   (`src/honest_agents/analysis/replay_episodes.py`, lines 12-19, 207-233,
   416-442). Nothing in `env.py`, `harness.py` or `trainer.py` resumes a rollout from a replayed state. The command
   runner and the fidelity check are reusable; the resume path is new code.
5. **research:failure-taxonomy, result 6**: "86 % [76, 94] had at least 3 of 28 bash calls left (median 13)" is the
   read sample of 64. On all 233 false claims of the same population: 185/233 = 79.4 % [69.9, 88.0] by source
   bootstrap, median 12, 33 (14.2 %) with none left. At the branch point the figure is better: 89.2 % have at least 3
   left after the last write (median 16).
6. **research:failure-taxonomy rec 3** ("replay up to the last state change") and **research:pass-fail-anatomy rec 1**
   ("up to and including the first read-back after the last write"): the first exists for every false claim but
   re-opens the defect-making write in only 52 % [43, 62] of identified cases; the second does not exist for 12.4 %
   (no command after the last write) to 15.6 % (no read-back of a written path) of false claims.
7. **Orchestrator notes, section 6, "77 % [64, 88] ... printed the violating state"**: one reader, a broad criterion.
   The published held-out read has 29 of 101 (29 %) "own check showed the violation" and 34 more "missed an explicit
   spec detail"; my automatic detector gives 51 % [42, 60] where it can score, and agrees with the reader on only 3
   of 12 "own check showed" cases it scores (it looks at one command, the reader at all of them). Any U forecast
   should carry v as a range, 0.29 to 0.77.
8. **research:pass-fail-anatomy, RL-group features**: its completion parser pairs a tool response with the wrong
   call when an assistant turn holds several calls. 28 of 1,480 RL episodes (1.9 %) and 133 of 24,283 commands
   (0.55 %) lose their exit code and output. Too small to change its conclusions; fixed in my loader.
9. **Confirmed, not corrected**: research:algorithm-feasibility's status-token saturation (I get 4 of 84 false
   claims uncertain); research:failure-taxonomy's concentration (18 sources hold half of the pool's false claims);
   the pool's within-group Var(U) of 0.269 (I get 0.271).

## 10. Blockers for a design built on these inputs

1. **No measurement of within-prefix outcome variance exists.** A plan that makes endgame DP-GRPO the main arm
   before pilot DP-0 rests on an unmeasured quantity for which the only direct offline evidence is n = 2.
2. **No code resumes a rollout from a replayed prefix** (correction 4): sandbox hand-over, restored command and
   character counters, a multi-turn prompt with earlier tool calls, and a cut index are all new. The integrity gate
   applies to them.
3. **Prefix transcripts.** The prescreen as specified writes `episodes.jsonl`; a prefix bank needs the full
   per-turn messages. The select banks do not have them.
4. **The pair ratio does not protect against a shared "claim less" or "edit again" shift** (section 6.2). A design
   that cites 1:1 neutrality as its safeguard is incomplete without the true-prefix claim meter and a full-episode
   budget guard.
5. **Existing banks cannot supply the training bank** (at most 100 eligible false-claim prefixes at 2 per source), so
   a DP arm depends on the prescreen and on the spec audit of its sources.
6. **Replay fidelity is shown for exit codes on short decision prefixes only.** State-level fidelity of 12-command
   prefixes with generated timestamps or commit hashes is unverified.

## 11. Caveats

- "State-changing" is a regular-expression classifier. With the repo's own classifier the last write moves by less
  than one command in the median and no conclusion changes.
- The detector of section 4 covers 44-57 % of false claims with a passing sibling and has a false-positive rate of
  5-11 %; its "differ" events are 83 episodes on 47 sources. Intervals are source bootstraps. Strict and loose fix
  rates bracket the truth; neither is a measurement of it.
- The "defect write" covers 105 of 281 false claims (those whose failure message names a path), matched by name.
- Select-split and RL-group token counts were not computed (no transcripts for select); token figures are from the
  pool bank, which ran at 12,288 tokens. Token truncation was rare there, and the 28-command cap is the same.
- RL groups come from the it19c runs (CR-GRPO payoff, not plain U); steps 1-5 are close to G2 but not G2.
- The natural twins are a selected set (samples that happened to issue the same commands), mostly simple tasks.
- Held-out translations use only the aggregates printed in the documents and assume train-side shares transfer.
- The prescreen supply figures assume unbanked legal rows resemble the banked, oracle-validated ones.
- The bets are judgment, not estimates.

## 12. Artifacts

Scripts and their saved outputs (complete paths):

- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/dp_load.py` (loader, hygiene guard, aligned completion parser)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/episodes.pkl` (episode cache)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/load_log.json` (row counts per bank)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/common.py` (extended state-change classifier, Wilson interval, source bootstrap)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/tok_pool.py` (per-turn token counts for the pool transcripts)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/tok_pool.pkl` (token cache)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/tok_calib.json` (calibration against recorded totals)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/geometry.py` (item 1)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/geometry.out` (item 1 output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/defwrite.py` (the write that leaves the failing state)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/defwrite.out` (its output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/natexp.py` (item 2)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/natexp.out` (item 2 output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/twins.py` (continue-from-the-same-state twins)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/twins.out` (its output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/determinism.py` (cross-sandbox determinism of shared prefixes)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/determinism.out` (its output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/defect_seen.py` (item 3 detector)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/defect_seen.out` (item 3 output per data set)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/pooled.py` (item 3 pooled with source bootstrap)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/pooled.out` (its output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/supply.py` (item 4)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/supply.out` (item 4 output)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/arith.py` (item 5)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-offline/arith.out` (item 5 output)

Reused from the earlier workflows:

- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/passfail/features.py` (read-back and write-target helpers)
- `~/Documents/honest-agents-archive/staging/analysis/it20_rev4/falseclaims/labels_joined.json` (the 64 reader labels)
