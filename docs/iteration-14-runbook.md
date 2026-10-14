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

## 3. Write the chain file

Copy `configs/chains/it14_session.yaml` and fill every `<PLACEHOLDER>`; the Mac and the runner both refuse to start while
any `<UPPER_CASE>` or `<CMD: ...>` token is left. Check it locally:

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
