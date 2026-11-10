# Cross-examination: decision-point rollouts in this trainer (shared-prefix groups)

Scope: can "replay a stored episode's commands up to a decision point, then let the policy continue, prefix as context,
every generated token trained" be built correctly in the current trainer, and how. Read-only work; no Docker, no GPU.
Scratch scripts and their outputs are in
`~/Documents/honest-agents-archive/staging/analysis/it20_rev4/dp-code/`.

## Verdict

**Buildable with stated limits.** Confidence 0.85 that the mechanism below passes its unit and node acceptance tests
in about 650-750 source lines plus about 500 test lines. The limits:

1. Build it as **forced turns through the live tool loop** (design F below), not as "prefix in the dataset prompt"
   (design P). P breaks four existing contracts; F breaks none.
2. The prefix context can never be byte-identical to the archived episode: across nodes and days only 33-49 % of
   shared prefixes of 3-6 commands print byte-identical outputs (mostly `ls -l` dates). The continuation must see the
   **re-executed** outputs, and the fidelity check must be exit codes plus normalised outputs plus a re-grade, not bytes.
3. No stored G2 episode on a train-legal variant has token ids today (0 of 1,860 rows in the pool's bank). Exact
   model-turn tokens need a new rollout pass through the trainer with the bank on (the lr-0 arm gives it for free).
   Rebuilding from stored messages is exact for 94.3 % of episodes at text level, not 100 %.
4. Exit-code fidelity is not grade fidelity. Silent divergence is documented (git commit hashes, DEF-01). Every prefix
   must be admitted by a full forced replay that reproduces the archived grade.
5. Refreshing prefixes from the current policy during a run is not supported by the fixed data order and the resume
   guard without more code. First version: a static prefix store.
6. It does not make an episode cheaper on the GPU. The context is still run forward and backward. Expect about
   17-25 % shorter steps (less generation), not a multiple.

## Method

- Read in full: `analysis/replay_episodes.py`, `env.py`, `harness.py`, `trainer.py`, `trl_rewards.py`,
  `train/rows.py`, `train/grpo.py`, `train/run.py`, `eval/rollout.py`, `eval/records.py`, the relevant parts of
  `models.py`, `bank.py`, `pilot/`, `mutation.py`, `behaviour.py`, `eval/read_diagnosis.py`, `grading.py`, and the
  installed TRL 1.14.1 `grpo_trainer.py` (`_tokenize_prompts`, `_generate_single_turn`, `_get_tool_suffix_ids`,
  `_generate`, `_generate_and_score_completions`, `_compute_loss`, `_full_logits_logps`) and
  `generation/vllm_generation.py`.
- Seven measurements on train-side data only (sources with split `test` dropped in every script; none were present):
  M1 round-trip of sampled turns through parse and render (local Qwen3.5-4B tokenizer), M2/M2b/M2c determinism of
  identical command prefixes within a run and across runs, M3 where the candidate branch points fall and what budget
  is left, M5 reliability of the regex "last write", M6 model tokens per turn, M7 outcomes of naturally shared
  prefixes. Plus the archived replay and regrade summaries.

## 1. How an episode gets its conversation and sandbox today, and where a prefix goes

### Today

- Dataset row: `task_dir` plus labels, and a constant `prompt` = `[system: SYSTEM_PROMPT, user: ""]`
  (`src/honest_agents/train/rows.py`, lines 82-94). `check_rows` refuses any
  other prompt (lines 153-157) and requires `variant_id` to be unique (lines 163-167).
- TRL calls `environment.reset(**row)` for each of the G copies of a row and appends the returned string to the last
  message's content (`.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py`, lines 2437-2450).
  `HonestHarborEnv.reset` returns the instruction at once and boots the sandbox in the background
  (`src/honest_agents/env.py`, lines 195-206); the first tool call waits
  for the boot (line 248).
- The prompt is tokenised once with the chat template and the tool schemas (TRL lines 1794-1804). From then on the
  context is built by **concatenating token ids only**: prompt ids + sampled turn ids + tool-suffix ids
  (`src/honest_agents/trainer.py`, lines 1110-1123, 1248-1252). The sampler
  gets those id lists (TRL lines 1860-1865; trainer.py lines 1212-1218) and the scoring forward runs on the same
  lists (TRL lines 2553 and 3085). So "the continuation is scored on the token sequence the sampler saw" holds by
  construction for whatever ids are used as context. The real requirement is different: the branched context should
  equal what a full episode would have held at that point.
- What a full episode's context holds, piece by piece:
  - system + tools + instruction, rendered by the template with `enable_thinking: False`
    (`src/honest_agents/models.py`, line 135);
  - each assistant turn: the sampled ids, ending on `<|im_end|>`;
  - each tool block: `_get_tool_suffix_ids` (TRL lines 1959-2009), which renders
    `\n<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
    (seen in the archived completions text);
  - `content` is `BudgetMeter.render(rc, out)`: `[exit code: N]` plus the output cut head-and-tail at a cap that
    shrinks from 2,000 to 500 characters as the character budget runs down, plus a `[budget: ...]` line when low
    (`src/honest_agents/harness.py`, lines 208-240, 250-255);
  - before that, the output is capped inside the sandbox at 1 MB (`capture_script`, harness.py lines 586-619), cut on
    the host with a note (env.py lines 271-272), and a timeout adds a line (env.py lines 273-274);
  - a refused call returns the refusal text and is not in the history (harness.py lines 180-200; env.py lines
    252-254); an invalid call returns `{'error': ...}` from the trainer (trainer.py lines 952-965).

### Is the stored record enough to rebuild the prefix byte for byte? No.

| source | what it holds | enough? |
|---|---|---|
| `episodes.jsonl` (`extra.history`) | executed commands, exit codes, outputs cut at 4,000 chars (harness.py lines 261-283; eval/records.py lines 61-91) | No. No model text. Refused and invalid calls are absent: 173 of 1,860 pool episodes (9.3 %) have one in the transcript, 62 (3.3 %) have a multi-call turn (bank of `17-it17a-data-rollouts/eval/it17_rollouts_feasible`). |
| eval-run bank row / transcript | every message as shown, tool calls as parsed JSON (eval/rollout.py lines 364-378, 308) | Model turns only as parsed messages, so only a canonical re-render. **0 of 1,860 rows carry `completion_ids`.** Tool results are as shown under the source run's budget (the pool ran at 12,288 tokens, so its character budget and warning lines differ from plan 20's). |
| trainer bank row (`bank.enabled: true`) | `prompt_ids`, `completion_ids`, `tool_mask`, `logprobs` (`src/honest_agents/bank.py`, lines 206-212) | Yes for model turns. But `bank.enabled` is false in `configs/base.yaml` (line 104) and the it19c run directories hold no bank. |

M1, parse then render on real sampled turns (script `m1_roundtrip.py`, 30 parquet files of the it19c ablation r2,
1,920 training episodes, text level):

- non-final turns: 30,348 of 30,679 reproduce exactly (98.92 %); 312 of the 331 misses are whitespace (for example a
  blank line before `</parameter>`), 5 are malformed calls;
- episodes whose every non-final turn reproduces: **1,811 of 1,920 = 94.3 % [binomial 95 %: 93.3, 95.3]**;
- this is an upper bound for token identity: the same text can be a different token sequence than the one sampled, and
  the parquet has no ids to check.

Parsing is lossy by design (`qwen3_5_template` strips whitespace around parameter values and JSON-parses them,
`.venv/lib/python3.12/site-packages/trl/chat_template_utils.py`, lines 353-378). Note that the evaluation loop itself
re-sends parsed messages each turn (eval/rollout.py lines 327-341), so evaluation contexts are already canonical
re-renders while training contexts are raw ids. A canonical prefix is therefore no further from evaluation than
training already is.

### Outputs differ on replay, almost always

M2b/M2c (scripts `m2b_cross_run.py`, `m2c_prefix_level.py`): the same select variant in two G2 runs on different nodes
and days, compared wherever two episodes issued identical leading commands (so the sandbox state was the same).

| comparison | n | exit codes equal | output byte-identical | identical after timestamp normalisation |
|---|---|---|---|---|
| per command, it17 vs it19c select | 2,594 commands | 100.0 % | 80.1 % | 99.5 % |
| per command, it16 vs it19c select | 2,529 | 99.9 % | 80.4 % | 99.2 % |
| per command, it16 vs it17 select | 1,261 | 99.5 % | 78.8 % | 98.7 % |
| whole shared prefix >= 3 commands (mean 4.1), it17 vs it19c | 400 pairs, 113 variants | 100 % | 48.5 % [37.8, 57.7] | 97.0 % [93.8, 99.7] |
| whole shared prefix >= 5 commands (mean 5.9), it17 vs it19c | 105 pairs, 49 variants | 100 % | 33.3 % [15.9, 47.7] | 89.5 % [78.4, 98.1] |
| same run, same node (M2, train pool) | 3,942 commands | 100 % | 98.8 % | 99.0 % |

Intervals are bootstraps over variants. At step 0 only 68 % of outputs are byte-identical across runs and 100 % after
normalising dates: the difference is the date column of `ls -l`. Residual differences are `ping` times, `uptime`,
`df`, `date`. A median false-claim prefix has 13 commands (M3), so a byte-identical replayed prefix will be rare.

Consequence. If the archived tool text were used as context, the context would disagree with the live sandbox in most
prefixes. If the re-executed text is used, the stored model turns were conditioned on slightly different text (dates).
The second is the right choice: the context is true to the state the continuation acts on, and the grader reads that
state. This is also what makes design P unattractive (below).

### Where to inject: two designs

**Design P, prefix as dataset prompt** (what the report-only pilot of iterations 17-18 did without a sandbox,
`src/honest_agents/pilot/dataset.py`, lines 1-29). Problems in the sandbox
setting:

1. Contracts: `load_rows` requires the prompt to end with a user message (rows.py lines 86-87), `check_rows` requires
   the constant prompt (lines 153-157), and TRL appends `reset()`'s return to the last message (TRL lines 2444-2450).
2. Nested tool-call messages cannot go through Arrow: every call gains a null entry for every argument name used
   elsewhere and the Qwen3.5 template renders them (pilot/dataset.py, lines 24-28).
3. Budgets: TRL counts `max_completion_length` on completion tokens only (trainer.py lines 1135-1138, 1225-1233) and
   the turn cap is one counter for the whole batch (line 1049). A prefix in the prompt gets a fresh 16,384 tokens and
   36 turns unless three per-row offsets are added to the ported loop. The vLLM window is sized as 3,072 + 16,384 +
   4,096 (`configs/base.yaml`, line 54; `train/derive.py`, lines 22-60; `train/startup.py`, lines 60-92), so a long
   prefix in the prompt can also overflow the server.
4. First turn: in server mode TRL sends `all_prompts[::num_generations]` with `n = G`
   (`.venv/lib/python3.12/site-packages/trl/generation/vllm_generation.py`, lines 602-632). All G prompts must be
   identical, so the context has to be the archived outputs, not each member's re-executed ones.
5. The replay would have to run inside the boot coroutine; `bash()` blocks on that same loop (env.py lines 244-279),
   so the env needs a second async exec path.

**Design F, forced turns through the live loop (recommended).** The row keeps the constant prompt. The first k
assistant turns are not sampled: their stored token ids are handed to the loop in place of a generation. Everything
else is the existing loop: the turn is parsed, its tool calls run in the member's own sandbox through `env.bash`, the
tool block is rendered by `_get_tool_suffix_ids`, the budgets count, and at turn k+1 real generation starts. The
forced tokens get `tool_mask` 0. This is what
`tests/test_trainer_loop.py::test_real_grpo_trainer_end_to_end_keeps_sampled_ids_and_stops_at_the_report` (lines
1527-1633) already does with a scripted `_generate_single_turn`.

- Empty prefix is literally the normal code path.
- The context is built by the same functions as a full episode; model turns are the stored ids, tool text is live.
- No change to `env.py`, `harness.py`, `grading.py`, `reward.py` (keeps the integrity-gate result caches valid; the
  replay cache key covers those files, replay_episodes.py lines 79-83).
- Groups: TRL's baseline is the mean over the G consecutive copies of one dataset row (TRL lines 2821-2822; the
  sampler repeats each index G times in a row, `trl/trainer/utils.py`, lines 904-908). One row per prefix gives a
  shared-prefix group with no change to the advantage code.

## 2. Budget accounting

| budget | where it is tracked | in design F |
|---|---|---|
| 16,384 tokens | TRL's loop: `len(pct) - len(prompt_ids[i]) > max_completion_length` and the cut of the last turn (trainer.py lines 1135-1138, 1225-1233) | forced turns and their tool blocks are in `completion_ids`, so they count. No new code. |
| 36 tool turns | `iteration_num < self.max_tool_calling_iterations`, one counter for the batch (trainer.py lines 1049, 1270) | a forced turn uses an iteration like a sampled one. No new code. |
| 28 bash calls, 29,491 output characters | `BudgetMeter.calls / chars / refusals / hit` (harness.py lines 146-151), charged in `render` (229-240) and `refusal` (180-200), called from `env.bash` (env.py lines 252-279) | the prefix runs through `env.bash`, so the meter is charged with the live outputs. No new code. |
| 4,096 tokens per turn | vLLM `max_tokens` | forced turns were sampled under the same cap. |

So a branched episode cannot exceed a normal one. Pre-charging without replay is possible for the meter (plain
attributes) but not for tokens and turns, which have no per-row state in TRL; that is one more reason against P and
against a snapshot path.

M3 (script `m3_branch_points.py`, G2 on the train pool, 1,860 feasible and control episodes, plan-20 budget of 28
calls): what is left at the branch point "after the first read-back that follows the last write".

| prefix class | n with that point | prefix commands median / p90 / max | calls left median | fewer than 3 left | none left | prefix tokens after the prompt, estimate median / p90 / max |
|---|---|---|---|---|---|---|
| false claim (340) | 319 (93.8 %) | 13 / 24 / 28 | 15 | 6.9 % | 2.5 % | 2,430 / 6,160 / 11,930 |
| true claim (1,338) | 1,253 (94.9 % of those with a write) | 9 / 18 / 28 | 19 | 1.7 % | 0.7 % | 1,640 / 4,230 / 11,070 |
| non-claim (182) | 103 | 21 / 28 / 28 | 7 | 38.8 % | 19.4 % | 5,490 / 11,090 / 12,360 |

Token estimates are the episode's server-side token count scaled by the prefix's share of characters. "About 12k
tokens of prefix" is the maximum, not the typical case. Prefixes with fewer than 3 calls left should be excluded.

## 3. Which tokens enter the loss; masks, DAPO, KL, memory

- Loss mask is `completion_mask * tool_mask` (TRL lines 2549, 3088). In F the forced tokens and the tool blocks have
  `tool_mask` 0; every token the policy samples in this episode has 1. Two lines change in the ported loop: the
  initial mask (trainer.py line 1038) and the per-turn extension (line 1243). Sampling log-probs for forced tokens
  are set to 0.0, as for tool tokens (line 1245); the importance ratio is masked the same way (TRL lines 2713-2717).
- Owner rule "never mask a token from the main loss": no sampled token of the current policy is masked. Forced tokens
  were not sampled in this episode; they are context, exactly as a prompt. With the optional re-split below they are
  literally prompt tokens. The pre-registration should say this in words.
- `mask_truncated_completions` looks only at the last token (TRL lines 2540-2547). No turn-1 assumption. Plan 20 turns
  it off anyway.
- DAPO normaliser: `num_items_in_batch = loss_mask.sum()` (TRL line 2550), used at lines 3229-3235. No turn-1
  assumption, but its size changes. M6 (script `m6_turn_tokens.py`, 639 r2 episodes ending in a report): an episode
  has 1,802 model tokens on average; the report turn alone is 366 (25.4 %); the last 4 turns are 41.7 %. So a
  continuation as long as the original tail trains about 750 tokens, not "a few hundred", and in a batch that is half
  shared-prefix groups those groups carry about 29 % of the summed loss (0.417 / 1.417). The learning rate, the batch
  mix and any stop rule on `completions/mean_length` have to be set with that in mind.
- KL term: added per token and summed under the same mask (TRL lines 3159-3166, 3210-3211). It acts on the
  continuation tokens only. No assumption about turn 1.
- CR-GRPO pieces (`status_position`, `StatusLogits`, trainer.py lines 91-215) index inside the completion and skip
  `tool_mask` 0 tokens; unaffected, and off under plain U.
- A masked rollout still counts its sampled tokens in the normaliser (trl_rewards.py docstring, lines 11-15); the same
  will hold for a rollout masked for a diverged prefix.
- Memory. The backbone runs on prompt + completion and keeps the graph for all of it, so the backward pass costs the
  same as a normal episode of the same total length; F cannot exceed today's worst case because the total length is
  bounded as today. The log-prob head runs over `logits_to_keep = completion length` (TRL lines 2556, 1552-1569), so
  with the prefix inside the completion nothing is saved. Optional re-split after the rollout (move the forced part
  from `completion_ids` to `prompt_ids` inside our `_generate` override, trainer.py lines 579-610) would cut the head
  to the continuation. The measured slope is 2.37 GB per 1,000 tokens of the longest completion
  (research_throughput, finding 11); how much of that is the head is not measured. Treat the re-split as a second
  step, tested for equal loss and gradients.

## 4. Replay fidelity and cost

Exit-code fidelity from existing results:

| replay | n attempted | exit codes identical | source |
|---|---|---|---|
| decision prefix (before the first destructive command on the target), G2 s1 | 959 | 947 (98.75 %) | archive `staging/def20-def50-node-20261026/replays/replay_g2s1.summary.json` |
| same, G2 s2 | 959 | 949 (98.96 %) | `replay_g2s2.summary.json` |
| same, SFT-v2 | 960 | 949 (98.85 %) | `replay_sftv2.summary.json` |
| full history, it17 train pool, remove_input controls (11-28 commands, median 20) | 27 | 27 | `runs/def74/regrade_it17_rollouts_feasible.jsonl` |
| full history, inbound_link controls (DEF-01) | 263 (+1 boot failure, 10 without a task dir) | 256 (97.3 %) | `docs/defects.md`, lines 163-164 |
| full history pooled | 290 | 283 = **97.6 % [Wilson 95 %: 95.1, 98.8]** | the two rows above |

The iteration-16 bar was at most 10 % not replayable per branch (`docs/iteration-16-direction.md`, line 73); all runs
meet it. These replays were of val-split and train-split sources.

Exit codes are not enough:

- DEF-01: in three sources "the replay fails it despite matching exit codes, so the replay diverges from the archived
  final state" (defects.md lines 165-167). In the G2 s1 file those sources hold 12 of 76 attempted episodes.
- The DEF-74 regrade: I checked research_failure-taxonomy finding 13 in the data. Episodes s0, s1, s2 of
  `task_000000_5830f696__remove_input__control` run `git commit` (steps 14, 11, 16), read the new hash, and write it
  into an audit log by hand (steps 18, 15, 20). A replay commits at a new time, gets a new hash, and the hard-coded
  old hash no longer matches. All exit codes are 0 and the rows are `replayable` with `mismatch_at` null. That is 3
  of 27 episodes in that file. The mechanism is confirmed; I could not confirm without Docker that the tests then
  fail for that reason.
- Risk flags in G2's false-claim prefixes (M3, crude regexes, an upper bound): any 31.7 %, `git commit` 3.4 %, a date
  or time written into state 26.6 % (includes plain `touch`), background process 1.9 %, network 1.6 %; in true-claim
  prefixes 14.6 %, 0.4 %, 9.3 %, 1.0 %, 4.6 %. The classes differ, so an exclusion rule changes the class mix.

What breaks determinism, from the data: file dates in `ls -l` (image build or boot time, and the agent's own writes),
`date`, `uptime`, `df`, `free`, `ping` and `dd` timings, process ids, git hashes. Not seen in the samples but
possible: `mktemp` names, random ports, network.

Cost:

- Archived replays: 12,583 s for 960 episodes and 1,806 sandbox boots (7.0 s of wall per boot; three such jobs ran
  side by side; concurrency not recorded). An earlier sample: 42 boots in 418 s at concurrency 2 (defects.md lines
  384-386). The 27 full-history regrades took 143.1 s including grading.
- Inside the trainer the prefix costs tool rounds only, the same rounds a full episode would have run. In r2's log a
  tool round takes a median 0.73 s outside the boot wait (30 steps, 31.5 rounds per step), so a 13-turn prefix is
  about 10 s of a step whose boot wait is 61 s. A prefix with a 60 s timeout re-pays it (1.3 % of false-claim
  prefixes); exclude those.
- Step time for an all-branched step, from r2's split (363 s at 64 episodes: generation 126, tools 84, grading 54,
  training 62, other 37): only generation shrinks, to roughly a quarter to a half if the continuation is 4-7 of 15
  turns. About 270-300 s, so 17-25 % shorter. Training time follows total tokens (prefix included), so it does not
  fall. This is arithmetic on one host's split, not a measurement.

Snapshot path: none exists. `grep` finds `docker commit` only as a planned item in
`docs/iteration-20-design.md` (lines 435 and 554); Harbor's Docker
environment has no commit or snapshot call. It could be built through a prebuilt-image task (harness.py lines 564-570)
but it loses running processes, needs the boot-time baseline carried across (grading compares against it, env.py
lines 387-395), forces archived text as context, and needs one image per prefix on every node. Replay is cheap enough
that it is not worth it now.

Also: `replay_episodes.HarborRunner` runs with a budget that never refuses (replay_episodes.py lines 215-219) and
tears the sandbox down. Its fidelity logic and cache are reusable; its outputs are not the model-facing text and it
cannot hand a live sandbox to the trainer.

## 5. Decision-point detection

| point | existing code | status |
|---|---|---|
| before the report turn | `bank.split_at_report` (used by `pilot/dataset.py`) | exact, no classifier |
| k turns before the report | none | trivial, exact |
| first state-changing command | `mutation.first_mutation_index` (`src/honest_agents/mutation.py`, lines 31-33) | first only; a regex |
| last state-changing command | none in the repo (scratch: `passfail/features.py`) | missing |
| first read-back after the last write | none in the repo (scratch: `passfail/features.py`, `READBACK` + `write_targets`) | missing |
| first destructive command on the target | `replay_episodes.decision_index` (lines 160-169) | exists; this is the END of the decision prefix |
| first command naming the target | `read_diagnosis.landmark_anchor` (`src/honest_agents/eval/read_diagnosis.py`, lines 287-293) | high_risk only: `site_of_task` returns None for other families (`tasks/perturb/risk_surfaces.py`, lines 224-237) |

M5 (script `m5_last_write_misses.py`, train pool): the regex "last write" is not reliable. In 294 of 1,820 episodes
(16.2 %) a command after it may itself write (a Python or shell script, `git commit`, `tar -x`, ...; `git commit` is
not in `_MUTATING`). The repo's three classifiers (`mutation.is_mutating`, `crgrpo._substantive`,
`behaviour.is_destructive`) agree on the last-write index in 651 of 1,820 episodes (35.8 %).

Missing, in order of need: (1) a tested `branch_points` module; (2) a state-based last-write detector (a file
manifest after each command during the admission replay; one more exec per command); (3) a family-agnostic anchor for
revoke_permission and remove_input built on `spec.expected_targets` and `decoy_targets`. The mechanism itself should
take only an integer (number of forced turns), so the first pilot can use the exact points (report turn; 2 and 4 turns
before it) and the detector-based points can follow.

## 6. Grading

Identical in meaning, with two notes.

- The grade is a function of the final sandbox state, the boot-time baseline, the report, the action count and the
  stop reason (`src/honest_agents/grading.py`, lines 1078-1205; env.py lines
  456-504). The baseline is taken before any command (env.py lines 387-395). Hidden tests run on the final state.
  Damage is baseline against final state, so it covers what the prefix did and what the continuation did.
- Nothing in the grader asks who issued a command. `env._history` is filled only by `bash()` (env.py line 278) and in
  F every prefix command goes through `bash()`.
- Note 1: `agent_acted = bool(self._history)` (env.py line 457). A grading failure after a forced prefix is scored as
  tamper against a continuation that may have done nothing. Rare; count it.
- Note 2: a prefix that already damaged a preserved path gives every continuation -1 (a zero-variance group). Exclude
  such prefixes at admission, or keep them knowingly.
- Reward plumbing: `trl_rewards.group_ids` uses `variant_id` (trl_rewards.py lines 72, 99-112). Two prefixes of one
  variant need their own id. `variant_id` cannot be overloaded because `check_rows` compares it with task.toml
  (rows.py lines 186-187); add a `group_id` column and prefer it.

## 7. Build spec (design F)

### Row schema

A branched row is a normal train row plus:

| column | type | meaning |
|---|---|---|
| `group_id` | str | `<variant_id>#<prefix_id>`; the prompt-group id |
| `prefix_id`, `prefix_sha256` | str | key and hash into the prefix store |
| `prefix_class` | str | `false_claim`, `true_claim`, `pair_predecision` (from the admission regrade, not the archive) |
| `branch_point` | str | `report`, `report-2`, `report-4`, `readback`, ... |
| `forced_n_turns`, `prefix_tokens` | int | for guards and logs |

The prefix store (one JSON file, hash recorded in the run config) maps `prefix_id` to: `turn_ids` (one id list per
forced assistant turn, each ending on the end-of-turn token), `source` (`ids` or `messages`), `commands` and `rcs`
(the executed bash calls expected during the prefix, in order), and the source episode. Keeping the id lists out of
the dataset avoids Arrow trouble and the kilobyte-long reward-argument warnings that `_Tokens` exists to prevent
(trainer.py lines 457-466).

### Files and functions

| file | change | lines |
|---|---|---|
| `src/honest_agents/train/prefix.py` (new) | `PrefixStore`; `turn_spans(tool_mask)`; `forced_turns_from_ids(bank_row, k)`; `forced_turns_from_messages(bank_row, k, tokenizer, tools, template_kwargs)` (tool-call arguments from JSON strings to dicts; asserts the generation prompt is a token prefix of the rendered turn); `build` CLI with admission filters (train split, not select, not a never-trained family, not masked or tampered, no timeout in the prefix, at least 3 calls left, token cap) | 220 |
| `src/honest_agents/trainer.py` | `_generate_and_score_completions`: load each row's forced turns before `super()`. `_generate_single_turn`: on the first call return stored ids for forced groups and call TRL for the rest (whole groups only, asserted). `_tool_call_loop`: split active rows into forced and free before the generation call; `tool_mask` 0 and log-prob 0.0 for forced turns; count `forced_turns`. After the loop: compare `env._history` with the expected commands and exit codes, set a per-row prefix status. Optional `resplit_prefix` | 110 |
| `src/honest_agents/trl_rewards.py` | `GROUP_KEYS = ("group_id", "variant_id", "task_dir")`; mask rollouts whose prefix diverged or did not complete with `apply_mask(br, "prefix_diverged")` (accepts any reason, reward.py lines 849-855); metrics `dp/branched_share`, `dp/prefix_diverged_share`, and per `prefix_class`: pass, claim, false-claim, continued-with-a-command, mixed-group share | 60 |
| `src/honest_agents/train/rows.py`, `config.py`, `plan.py` | uniqueness and order fingerprint on `group_id`; `data.branch_file`, `data.prefix_store`; validate branched rows | 60 |
| `src/honest_agents/train/startup.py` | `check_forced_rows`: with the real tokenizer every forced turn ends on the end-of-turn token, parses to the stored calls, and `prefix_tokens` + turn cap fits the budget | 30 |
| `src/honest_agents/analysis/prefix_check.py` (new; leaves `replay_episodes.py` and its cache key alone) | forced-tail admission: full replay with the archived report under the training budget, re-grade, compare pass, claim and damage; normalised-output comparison; run twice | 120 |
| `src/honest_agents/analysis/branch_points.py` (new) | last write, read-back after it, first naming of a target; later the manifest-based detector | 120 |
| `src/honest_agents/bank.py` | record the number of forced tokens in `extra`, so a bank reader can tell forced turns from tool blocks | 10 |

About 730 source lines. Not touched: `env.py`, `harness.py`, `grading.py`, `reward.py`, the TRL methods under
`UPSTREAM_FINGERPRINTS` (trainer.py lines 282-292).

### Tests (about 500 lines)

- `tests/test_trainer_loop.py`:
  1. every existing differential test against TRL's loop still passes with the new code present (empty prefix equals
     today, exactly);
  2. record a scripted episode, then force its first k turns and script the rest: `completion_ids`, exits, tool calls
     and their order are identical for every k; `tool_mask` is 0 exactly on the forced turns and tool blocks;
  3. all turns forced including the report: nothing is generated, the report is executed, exit `report`;
  4. budgets: forced turns count against `max_completion_length`, the turn cap and a real `BudgetMeter`;
  5. mixed batch: only free groups reach the generator on the first call; a group with mixed forced state raises;
  6. loss on the tiny real Qwen3.5 config: zero gradient through forced positions as targets, normaliser equals the
     sampled tokens, and the re-split layout gives the same loss and gradients;
  7. a different exit code at forced step j masks that rollout only and the group mean uses the rest.
- `tests/test_prefix_rows.py`: store from ids and from messages, hashes, Arrow round trip, row guards.
- `tests/test_trl_rewards.py`: group key, mask, new metrics.
- Docker-marked: 20 pool episodes forced in full through the trainer loop on real sandboxes equal the
  `prefix_check` grade.

### Staged acceptance

| stage | where | test | pass bar |
|---|---|---|---|
| 0 | laptop, no Docker | unit tests above | all green, including the unchanged TRL differential tests |
| 1 | laptop or node, tokenizer only | build the store; for id sources, prompt + forced turns + tool blocks equals the bank's ids up to the cut | 100 % for id sources; for message sources report the exact-render share (measured 94.3 % at text level) |
| 2 | node with Docker, no GPU, under 30 min | forced-tail on 300 pool episodes (150 false claims, 150 true claims, stratified by source), each replayed twice | exit codes identical >= 95 % (evidence 97.6 %); of those, pass, claim and damage reproduced >= 95 % in each class; the two replays agree >= 98 %; admitted share >= 85 % overall and within 10 points between classes |
| 3 | node, one GPU, learning rate 0, about 2 h | (a) rows with an empty prefix against plain rows; (b) rows with all turns forced through the trainer; (c) branched rows | (a) same metrics within sampling noise; (b) grade reproduced >= 95 % of admitted prefixes; (c) no episode over 28 calls, 36 turns or 16,384 tokens, `dp/prefix_diverged_share` <= 5 %, peak memory not above a normal step |
| 4 | same, 960-1,920 episodes | the kill test asked for in the notes: 60 false-claim and 60 true-claim prefixes x 8 continuations of G2 at 2-3 branch points | science, not code: mixed-outcome share on false-claim groups >= 40 %, and true-claim prefixes still pass and claim |

### Risks

1. Admission bias. Excluded prefixes are not a random subset (nondeterminism flags are 2-3 times more common in
   false-claim prefixes). Report the admitted share by class and by source.
2. Prefix class from the archive can be wrong after replay; take it from the admission regrade.
3. Message-sourced prefixes are canonical re-renders (5.7 % of episodes differ somewhere in text). Prefer id sources
   from a trainer pass with the bank on.
4. The ported loop gains a branch TRL does not have. The differential tests keep the unforced path honest; the forced
   path rests on tests 2-5.
5. Prefix refresh during a run conflicts with `check_resume_order` (rows.py lines 286-304) unless the store, not the
   dataset, is what changes. Not in the first version.
6. Regex branch points are wrong in about 1 episode in 6. Start with exact points.
7. Integrity gate: this is a trainer change; it needs a defect-ledger entry, a sweep and a sign-off before any run.
8. Cluster differences. Exit-code fidelity was measured on the project's own nodes; a different Docker or kernel can
   move it (DEF-16 was such a case for the umask).

## What I would bet

- F passes stages 0, 1 and 3 within about 750 source lines: 0.85.
- Exit-code fidelity >= 95 % on the cluster: 0.85.
- Grade reproduction >= 95 % with no filter: 0.55; with the admission filter and its second replay: 0.80.
- Admitted share >= 85 %: 0.70.
- Kill test (outside the code question; my guess from what the code and data show): mixed-outcome share >= 40 % on
  false-claim groups at the report turn 0.2, after the read-back 0.4, before the last write 0.6. Existing data cannot
  settle it: only 10 pool pairs share 10 or more identical commands, and among pairs sharing 6-9 commands with a
  false claim (n = 20) the other sample ended differently in 50 % [30, 70] (M7, selection-biased).

## Corrections to the 18 reports and the notes

1. Notes section 6, "`analysis.replay_episodes` already replays prefixes with an exit-code fidelity check": true, but
   it cannot serve a training rollout. It uses a never-refusing budget, tears the sandbox down, and checks exit codes
   only. Reusable: the fidelity logic, the cache, `decision_index`.
2. Notes and research_twin-judgment-signal item 8, "prefix up to the first command that names the target
   (`replay_episodes.decision_index`)": `decision_index` returns the first **destructive** command on the target.
   The first command naming the target is `read_diagnosis.landmark_anchor`, and only high_risk tasks have a site.
3. Notes section 6, "the trained tokens per episode fall from about 1,900 to a few hundred": the report turn alone is
   366 tokens on average and the last 4 turns are 41.7 % of an episode's model tokens, so a tail-length continuation
   trains about 750 tokens; only a branch right before the report gives a few hundred.
4. research_lit-honesty-rl item 3, "Tails are short, so more episodes fit in the same wall time": only generation
   shrinks. Boot, grading and the backward pass over the whole context stay. About 17-25 % per step.
5. Notes kill test, "replay fidelity (at least 90 % replayable)": the exit-code figure is already 97.6 % [95.1, 98.8]
   on full histories and 98.8-99.0 % on decision prefixes. The open quantity is grade reproduction, which no archived
   run measures cleanly, and which has known failures. The test should be on grades.
6. research_failure-taxonomy finding 13 (marked not verified): the mechanism is confirmed in the episode data (commit,
   then the hash written by hand, all exit codes 0, status `replayable`).
7. research_pass-fail-anatomy item 1, "the prefix is context, every generated token is trained and under the KL":
   correct, and it needs no change to the loss; but "refresh prefixes from the current policy every 50 steps" is not
   supported by the current data-order and resume guards.
8. Notes section 6, "prefixes from false-claim episodes and from true-claim episodes 1:1": the class label must come
   from a regrade of the replayed episode, and the two classes will lose different shares at admission.
