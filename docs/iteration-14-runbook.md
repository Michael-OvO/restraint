# Iteration 14 node runbook (chained session, no idle node)

Why: iteration 13 lost ~$9 when a GRPO run finished with no watcher armed and the node idled 5.5 h. Every node job now
lives in one **chain file** (`honest_agents.node_chain`), and one Mac-side **watcher** (`honest_agents.node_watch`) is the
only long-lived monitor: it prints one line per event, and on `ALL_DONE` pulls the results and **stops** (never destroys)
the Vast instance. Design: `docs/iteration-14-design.md` section 5. Do not rent anything until the Mac-side gates pass and
the owner has approved the plan; ask before every launch and stop.

## 1. Bring up the node

1. Search offers (`vastai search offers`, or the `vastai:search` skill): 2x RTX 4090 (~$1.61/h) is the plan. Check
   `driver_version >= 580` (torch 2.13 / vLLM 0.30 need CUDA 13) and the host's GPU initialises.
2. Launch with the VM template, hash **`efddf9e90256ed2dbbae8734d8300e69`** ("Ubuntu Desktop (VM)"). It works on Blackwell and
   Ada (the older `ubuntu_cli_22.04` KVM image fails on Blackwell: sshd refuses the key). Register the key
   (`~/.ssh/vast_ed25519`, Vast key id 1496289) and use `--ssh --direct`. Note the instance id and the ssh host and port.
3. First login, one time (`ssh -i ~/.ssh/vast_ed25519 -p PORT root@HOST`):
   * Docker address pool (every sandbox takes a network; the stock pool holds ~30). Put
     `{"default-address-pools": [{"base": "10.128.0.0/9", "size": 24}]}` in `/etc/docker/daemon.json`, then
     `systemctl restart docker`. Without the compose plugin: `apt install docker-compose-v2`.
   * `scripts/setup_node.sh --install-docker` (after the first sync below): uv, Docker, nftables check, weights, verifier
     bundle; it also disables unattended upgrades. Then `scripts/probe_node.sh` must pass (GPUs, Docker, compose v2, pools).
4. No GitHub key and no secret is ever stored on the node.

## 2. Sync code and data (from the Mac main checkout, not a worktree)

```bash
H=HOST P=PORT
python -m honest_agents.node sync --host $H --port $P --dry-run        # itemised, nothing sent
python -m honest_agents.node sync --host $H --port $P --with-data      # tree incl. .git, plus data/
# bring up as in section 1 step 3 (setup_node.sh, probe_node.sh), then sync again after any local change
```

The chain file travels with `sync` (it lives under `configs/chains/`), so sync **after** the last edit of it.

### 2a. Stage the bank and the iteration-13 adapters under `data/` (once, on the Mac)

`sync` never sends `outputs/` or `runs/`, and the SFT builders read the iteration-3 pass@4 bank (15 MB) while the GRPO
checkpoint-10 tier F needs two archived adapters (248 MB each). Everything the chain reads therefore lives under `data/`
(gitignored, sent by `--with-data`); the chain's paths are fixed to these:

```bash
A=~/Documents/honest-agents-archive/iterations
mkdir -p data/bank data/weights/it13
cp -R $A/03-passk-bank/eval/passk/bank data/bank/passk
cp -R $A/13-sft-decision-anchor/run/it13-sft-decision-anchor-3ep/final data/weights/it13/anchor3ep-final
cp -R $A/13-grpo-and-evals/run/honest-a0.1-sft_anchor3ep_merged-s42/checkpoint-10 data/weights/it13/grpo-anchor3ep-ckpt10
DRY_RUN=1 scripts/run_sft.sh configs/sft_decision_v2_qwen35.yaml data.bank=data/bank/passk       # the override the chain uses
```

GRPO checkpoint 10 was trained on the merged anchored-3ep model, so the chain merges twice: base + `anchor3ep-final` ->
`outputs/it13/sft_anchor3ep_merged`, then that + `grpo-anchor3ep-ckpt10` -> `outputs/it14/grpo10_merged`. Disk: every
merged 4B model is ~8 GB (two here, three SFT-v2 epochs, later DPO): rent >= 120 GB of disk.

### 3a. Scripts the chain needs (TODO before renting; not yet in the repo)

The chain file is complete, but three small `scripts/*.sh` it calls do not exist yet. The smoke step fails within minutes
if they are missing (`command not found`), so nothing beyond the smoke is wasted, but write and test them first:

| script | contract |
|---|---|
| `scripts/serve_and_run.sh CONFIG [dotted.key=value ...] -- CMD [ARGS] [-- CMD2 ...]` | `source scripts/_common.sh`; `slot_env`; `trap stop_vllm EXIT INT TERM`; `EVAL=1 start_vllm outputs/logs/serve_<step>.log CONFIG <overrides> --served-model-name policy`; `wait_vllm`; then each command group after a `--` in order with `MODEL=policy BASE_URL=http://127.0.0.1:$VLLM_PORT/v1` exported, stopping at the first non-zero status (exit with it). The server is served on `VLLM_GPUS` / `VLLM_PORT` from the step's `env`. This is the "one repo script that starts, waits, evaluates and stops" the old chain comments asked for; every evaluation and probe step uses it. |
| `scripts/merge_epochs.sh RUN_DIR OUT_PREFIX BASE` | for every `RUN_DIR/checkpoint-*` in ascending step order (k = 1, 2, 3): `uv run ... python -m honest_agents.sft.merge --base BASE --adapter RUN_DIR/checkpoint-N --out OUT_PREFIX_e<k>_merged`; refuse if the count is not 3 (`save_total_limit: 3`, `save_strategy: epoch`). |
| `scripts/select_epoch.sh BASE_EVAL OUT_DIR FALLBACK_EPOCH EVAL_DIR...` | `uv run ... python -m honest_agents.eval.select_checkpoint --base BASE_EVAL EVAL_DIR... --require-split select --out OUT_DIR/selection.json`; exit 1 (no epoch passes the capability gate) is **not** an error: the epoch is then FALLBACK_EPOCH (pre-registered: 3). The chosen epoch k is the trailing number of `selected` (`..._e2` -> 2). Write `OUT_DIR/selected_epoch.txt` (one line: `k` and whether it was selected or the fallback) and `ln -sfn sft_v2_e<k>_merged OUT_DIR/sft_v2_selected_merged`. |

Preferably `select_checkpoint` grows `--fallback-step` / `--link` options and `sft.merge` an `--epochs RUN_DIR` mode, so the
two glue scripts become one-liners; the contracts above are what the chain relies on either way. Test the three on the
Mac without a GPU where possible (argument parsing, the `--` split, the fallback path with a fake `selection.json`) and
run `scripts/test.sh --fast`.

### 3b. Build and validate the evaluation sets the chain reads (Mac, Docker)

**New-surface val set** (plan H2; not yet built: `data/decision/surface_pairs.jsonl` holds only `sft` and `probe` rows).
The 20 `high_risk` twin sources of `data/val.jsonl`, all six surfaces, varied request phrasing and position, held-out
pool pairs and phrasings (`--split test`): ~120 pairs = ~240 variants.

```bash
uv run python - <<'PY'   # TODO: a `tasks.surfaces_cli sources --split val` subcommand would replace this one-off
import json; rows=[json.loads(l) for l in open("data/val.jsonl")]
ids=sorted({r["pair_id"] for r in rows if r.get("family")=="high_risk" and not r.get("is_control")})
open("data/decision/val_high_risk_sources.txt","w").write("\n".join(ids)+"\n"); print(len(ids),"sources")
PY
uv run python -m honest_agents.tasks.surfaces_cli build --bank data/tasks --out data/decision/surfaces_val \
    --mode varied --split test --ids-file data/decision/val_high_risk_sources.txt            # all six surfaces; run ONCE (the jsonl appends)
uv run python -m honest_agents.tasks.g1_surfaces --variants data/decision/surfaces_val/surface_variants.jsonl \
    --out data/decision/surfaces_val/g1.jsonl --concurrency 6                               # G1-data: ~1-2 h on OrbStack
```

Remove both task dirs of every pair that fails G1 (keep `g1.jsonl`), then count the dirs: the chain's surface steps have
`progress.total` = 2 x that count (480 for 240 dirs; adjust the four `/480\]` patterns). The chain evaluates the
**directory** `data/decision/surfaces_val/tasks` (run_eval reads `[metadata.honest]` incl. `detail.surface` from each
`task.toml`); it must not read `surface_variants.jsonl`, whose `task_dir` column is cwd-relative and would be resolved
against the file's own directory. The node smoke runs 16 episodes of this directory first.

**Select split G1-data** (design section 4: its 212 variants were never validated; `data/validation.jsonl` has none of them):

```bash
uv run python - <<'PY'   # TODO: same remark; `select_split` could write the ids file
import json; ids=[json.loads(l)["variant_id"] for l in open("data/select.jsonl")]
open("data/select_variant_ids.txt","w").write("\n".join(ids)+"\n")
PY
uv run python -m honest_agents.tasks.validate --tasks-dir data/tasks --out data/select_validation.jsonl \
    --only-file data/select_variant_ids.txt --concurrency 8                                 # ~1-2 h
```

Variants that fail (`ok: false`, `leak: true`, `relabel_to_control: true`) go into `data/select_quarantine.txt`, one JSON
line per variant in `bank.QuarantineEntry` form, e.g. `{"variant_id": "<id>", "kind": "leak", "source": "validation",
"detail": "G1 select"}`; every select and surface evaluation in the chain passes `--exclude-quarantined
data/select_quarantine.txt` (a missing or empty file excludes nothing). Do not rewrite `select.jsonl`: the split is frozen.

## 3. Write the chain file

`configs/chains/it14_session.yaml` is segment A, fully written (13 steps, no placeholders; `node_chain check` passes). What
still has to be true before it runs is listed at the top of the file and in sections 2a-3b. The Mac and the runner refuse to
start while any `<UPPER_CASE>` or `<CMD: ...>` token is left, so segment B's file (section 7) should be drafted the same way. Check:

```bash
python -m honest_agents.node_chain check configs/chains/it14_session.yaml     # plan, dependency check, placeholder warning
```

Fields (details in the `honest_agents.node_chain` docstring):

| field | meaning |
|---|---|
| `steps[].command` | run as `bash -lc` in the repo root, own process group |
| `steps[].gpus` | steps with overlapping GPU sets never run together; disjoint ones run in parallel when their dependencies allow |
| `steps[].done` | marker file; exit 0 **and** marker present = done. A marker that already exists means "done, skip" (resume), so delete stale markers of an earlier attempt |
| `steps[].after` | dependencies; omitted = previous step (`sequential: true`, default); `[]` = none |
| `steps[].timeout_min` | SIGINT (sandbox teardown runs), then SIGTERM, then SIGKILL to the process group |
| `steps[].on_failure` | `stop`: no new step starts, running ones finish; `continue`: independent steps still run (dependents are skipped) |
| `steps[].env` | environment for the command; this is where a step is pinned to its GPUs and port (below) |
| `steps[].progress` | `{pattern, total}`: last regex match in the log tail becomes `PROGRESS` events (25 % steps) |
| `steps[].stall_min` | log quiet this long while running: `STALL` (default 20) |
| `final.command` | always runs last, even after a failure or STOP: archive to `/root/staging` (honest_agents.archive `--dest`) |
| `watch.pull_paths`, `watch.archives` | what the Mac pulls after `ALL_DONE` (the state dir is always pulled) |
| `instance_id`, `balance_floor` | defaults for the watcher (flags override) |

Every step command must be a script from the repo (`scripts/*.sh`, `uv run python -m ...`), not an ad-hoc line. `gpus:`
only schedules; the scripts read `SLOT` (default 0 = training GPU 0, **vLLM GPU 1**, port 8000), so an evaluation that is to
serve on GPU 0 needs `env: {VLLM_GPUS: "0", VLLM_PORT: "8000", BASE_URL: "http://localhost:8000/v1"}` and a second server on
GPU 1 its own port (8001). One vLLM server per step, started and stopped inside the step (`start_vllm` / `trap stop_vllm`):
when a step ends, the runner kills whatever still carries the step's tag (`HA_CHAIN_STEP` in the environment, found through
/proc, so a `setsid` server is reached), and the state file notes `N leftover process(es) killed`. Stop evaluations with the
chain (SIGINT), never `pkill`.

## 4. Start the chain, then the watcher

```bash
python -m honest_agents.node chain --host $H --port $P --dry-run configs/chains/it14_session.yaml   # plan + ssh command
python -m honest_agents.node chain --host $H --port $P configs/chains/it14_session.yaml              # starts the runner
python -m honest_agents.node watch --host $H --port $P --instance-id ID --balance-floor 8 configs/chains/it14_session.yaml
```

`chain` sends the W&B key and HF token over ssh stdin exactly like `node run` (names printed, values never) and starts the
runner detached (`/root/chain-NAME.log`). **Arm the watcher in the same minute as the chain**: run it under a Claude Code
Monitor (each stdout line is one notification; it prints only on events). One runner per state dir (a lock refuses a second).

Event lines: `START`, `PROGRESS name 50% (n/N)`, `DONE name 12m`, `FAILED|TIMEOUT name exit N`, `SKIPPED`, `STALL`,
`CHAIN failed|stopped|aborted: reason` (verdict while the final archive runs) or `CHAIN refused: ...` (see section 6),
`RUNNER no heartbeat` (runner process dead: restart `node chain`; done markers resume it), `NODE ssh failing 3 polls` / `ssh back`
(blips are silent below the limit), `NODE no chain state` (runner never started), `BALANCE`, `ALL_DONE status`, `PULL ok|FAILED`,
`INSTANCE ID stopped (not destroyed)` (confirmed with `vastai show instance ID`: `intended_status` stopped) or
`STOP FAILED: run vastai stop instance ID`.

Watcher flags: `--poll 60`, `--ssh-fail-limit 3`, `--balance-every 5`, `--hard-floor USD` (pull and stop at once),
`--no-stop`, `--dry-run` (prints the pull and stop commands). Below `balance_floor` it writes the chain's `STOP` file (no new
step; the current ones finish; the final archive runs; then the normal pull and stop). To stop by hand:
`ssh ... touch /root/restraint/outputs/chain/it14/STOP`.

## 5. After the stop

The instance is stopped, not destroyed (disk kept, small storage charge). Check `~/Documents/honest-agents-archive`
(`staging/` and `iterations/14-...`), write the training-log entry, then, with the owner's go-ahead, either
`vastai start instance ID` for the next segment (a new chain file; `state_dir` differs) or destroy it. Re-sync the archive
before any destroy (`reference-data-archive`).

## 6. Failure cases

* Step failed: the chain stops starting steps (policy `stop`); the final archive still runs, the watcher pulls and stops.
  Fix, delete the failed step's stale outputs and marker, `vastai start instance ID`, re-sync, restart `node chain` (it removes
  the stale `ALL_DONE` first; finished steps are skipped by their markers), then arm the watcher again.
* `CHAIN refused`: the restarted runner found processes of the earlier run still alive (pids in the state file's
  `stop_reason`; typically the runner died but its step did not). Nothing was started and no `ALL_DONE` will come: wait for
  them or `kill -TERM PID`, then restart `node chain`. A runner that dies mid-chain leaves its steps running (own sessions).
* Leaked sandboxes: after a timeout, abort or failure the runner removes `ep*` containers created since that step started
  (deferred while another step runs, so parallel sandboxes are untouched; with `on_failure: continue` they can therefore
  linger for the length of the parallel step).
* Watcher Mac asleep: use `caffeinate -i` around it; the node idles at ~$1.6/h until it is back.
* `vastai show user --raw` unreadable 3 times: `BALANCE unreadable`; the floor is not enforced, check the balance by hand.
  The watcher reads the `credit` field only (`balance` is a different field, 0 on this account).
* Pull slow or hung: each rsync has `--timeout=300` (stalled transfer) and a 1 h wall limit, two attempts; the stop follows
  the pull, so a pathological pull delays the stop (worst case hours). `PULL FAILED` lists the commands to rerun by hand
  (`node pull`, `honest_agents.archive`); the node's `/root/staging` copy survives a stop.
* A stopped Vast instance is not guaranteed to restart (its GPUs may be rented meanwhile). Segment A therefore archives
  the SFT-v2 adapters (final + selected epoch) to the Mac; if the instance cannot be started for segment B, rent a new one,
  stage `data/weights/it14/` the way section 2a stages the iteration-13 adapters, and re-merge on the node (~10 min).

## 7. Reading G2 and the segment-B drafts (after segment A has been pulled)

**G2 (owner, from the archive; plan section 4):** `eval/it14_sft_v2_tierS/metrics.json` (high_risk twins
`abstain_correct`, plain verified success, control decline rate) and `it14_sft_v2/summary.json` of the probe
(`sets.own_prefix.by_surface_role.seen_surfaces.discrimination`, `by_surface`). The epoch that was evaluated is in
`selected_epoch.txt` (selected or fallback). Comparisons on the Mac, no GPU:

```bash
S=~/Documents/honest-agents-archive/iterations; A13=$S/13-sft-decision-anchor/eval; A14=$S/14-it14-session/eval   # names as archived
uv run python -m honest_agents.eval.compare_runs $A14/it14_grpo10_tierF $A13/it13_a0p_base_tierF        # GRPO ckpt 10 vs A0' (G4 table)
uv run python -m honest_agents.eval.compare_runs $A14/it14_grpo10_tierF $S/13-grpo-and-evals/eval/it13_anchor3ep_tierF
uv run python -m honest_agents.eval.compare_runs $A14/it14_anchored_surface $A14/it14_base_surface       # anchored-3ep on the surface set: seen vs held-out pooled
uv run python -m honest_agents.eval.compare_runs $A14/it14_sft_v2_surface $A14/it14_base_surface         # SFT-v2 H2 readout (recorded; G4 if no DPO)
```

**Segment B chain (write `configs/chains/it14_segment_b.yaml`, `state_dir: outputs/chain/it14b`, same conventions).** The
node keeps `outputs/it14/sft_v2_selected_merged` and `outputs/it14/sft_v2_e*_merged` across a stop. Step drafts:

```bash
# DPO (only if tier-S control declines > base + 5 = 9 %): GPU 1, on the merged selected epoch; then merge and tier S
CUDA_VISIBLE_DEVICES=1 env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.sft.dpo \
    --config configs/sft_dpo_decision_v2_qwen35.yaml model.name=outputs/it14/sft_v2_selected_merged data.bank=data/bank/passk
env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.sft.merge --base outputs/it14/sft_v2_selected_merged \
    --adapter outputs/sft/it14-dpo-decision-v2/final --out outputs/it14/dpo_merged
scripts/serve_and_run.sh configs/grpo_it13_2gpu.yaml model.name=outputs/it14/dpo_merged \
    -- scripts/eval_heldout.sh data/probe_v1.jsonl runs/eval/it14_dpo_tierS --n-samples 2 --concurrency 24 --train-config configs/grpo_it13_2gpu.yaml
# Candidate (CAND = dpo_merged or sft_v2_selected_merged) tier F + surface set, GPU 0: G4 and H2 (sanity run first, as in segment A)
scripts/serve_and_run.sh configs/grpo_it13_2gpu.yaml model.name=outputs/it14/CAND \
    -- scripts/eval_heldout.sh data/val.jsonl runs/eval/it14_cand_sanity --limit 12 --n-samples 1 --concurrency 12 --train-config configs/grpo_it13_2gpu.yaml \
    -- scripts/eval_heldout.sh data/val.jsonl runs/eval/it14_cand_tierF --n-samples 4 --concurrency 48 --train-config configs/grpo_it13_2gpu.yaml \
    -- scripts/eval_heldout.sh data/decision/surfaces_val/tasks runs/eval/it14_cand_surface --n-samples 2 --concurrency 48 --train-config configs/grpo_it13_2gpu.yaml --exclude-quarantined data/select_quarantine.txt
# Agentic Abstention, base and candidate (one serve_and_run each; TAG names the arm)
TAG=base N_ATTEMPTS=4 scripts/serve_and_run.sh configs/grpo_it13_2gpu.yaml -- scripts/eval_agentic_abstention.sh runs/harbor
TAG=cand N_ATTEMPTS=4 scripts/serve_and_run.sh configs/grpo_it13_2gpu.yaml model.name=outputs/it14/CAND -- scripts/eval_agentic_abstention.sh runs/harbor
# SFT-v2 seed 2 (GPU 1; a different run_name, or it overwrites seed 1), the SAME epoch as seed 1 (no re-selection), tier F
CUDA_VISIBLE_DEVICES=1 scripts/run_sft.sh configs/sft_decision_v2_qwen35.yaml data.bank=data/bank/passk seed=2 \
    run_name=it14-sft-v2-s2 train.extra.run_name=it14-sft-v2-s2
scripts/merge_epochs.sh outputs/sft/it14-sft-v2-s2 outputs/it14/sft_v2_s2 Qwen/Qwen3.5-4B      # then serve outputs/it14/sft_v2_s2_e<k>_merged, tier F as above
# Mac, afterwards: G4 and the two-seed non-inferiority verdict
uv run python -m honest_agents.eval.compare_runs <it14_cand_tierF> <it13_a0p_base_tierF>
uv run python -m honest_agents.eval.run_eval --noninferiority <it14_cand_tierF>,<it14_s2_tierF> <it13_a0p_base_tierF>
```

The seed-2 arm is the same recipe at a different seed; if DPO was applied to seed 1, seed 2 gets the same DPO stage (the
non-inferiority verdict is about the candidate recipe, not about SFT-v2 alone).
