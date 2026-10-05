# honest-agents on OpenEnv

The honest-agents environment as an [OpenEnv](https://github.com/huggingface/OpenEnv) 0.7.0 server (`openenv` on
PyPI, **not** `openenv-core`). It is for interop, release and evaluation, not for training. GRPO training uses
`honest_agents.env.HonestHarborEnv` in-process (see `docs/DESIGN.md`, `docs/SPEC_V3.md`). Both run the same trl-free
core (`harness`, `spec`, `report`, `grading`, `verifier`, `reward`), so an episode is graded and scored by the same
code on either path.

## Episode

| step | action (`HonestAction`) | result |
|---|---|---|
| reset | `reset(task_id=...)` or `reset(split=..., index=...)` | the sandbox boots with the task's network policy and `[agent].user` enforced, protected paths are baselined and the verifier toolchain is checked. Returns `instruction` and `system_prompt` (identical to training) |
| tool | `kind="bash", command=...` | `output` = `[exit code: N]\n<stdout+stderr>`, `exit_code`, `steps_remaining`, `chars_remaining`, `turns_remaining` |
| end | `kind="submit_report", status, completed, incomplete, blocker_category, blocker_target, blocker_detail, evidence_commands, confidence` | `done=True` and `reward`, after grading |
| end | `kind="abstain", reason, blocker_category, blocker_target, confidence` | `done=True` and `reward`, after grading |
| end | `kind="end"` (optional `stop_reason`: `silent_stop` or `truncated`) | the model stopped without a report: graded as it stands |

* **One blocker per report**: a category, the specific object (`blocker_target`: a command name, a path or a host) and
  a detail. The argument names and order are those of `honest_agents.harness.ReportTools`
  (`tests/openenv/test_parity.py`). The pre-v3 `blockers` list is rejected by the action schema. To prompt a model with
  the training tool schemas, call `honest_agents.harness.tool_schemas()` on the client side (it needs `transformers`,
  which the server image does not carry).
* **Budgets** are `harness.EpisodeBudget.from_limits(HONEST_MAX_STEPS, HONEST_MAX_COMPLETION_TOKENS)`, the same
  construction as in training and evaluation. `bash` refuses to run, and says so, once the call budget
  (`HONEST_MAX_STEPS` − 2) or the tool-output character budget is used. Outputs are cut to 2000 characters, less as the
  budget runs down, and carry a remaining-budget line near the end. Each command runs under `timeout -k 5 <t>` inside
  the sandbox. The server sees only tool output, not the model's own tokens, so the client must enforce the per-turn
  and total token caps itself.
* The first report is final. After `HONEST_MAX_STEPS` actions of any kind the episode ends without a report. An episode
  that ends without a report is **scored** (`feasible_no_report` or `implicit_abstain`), never masked. If no report
  tool was called, a report file at `/logs/agent/honest_report.json` in the sandbox (same schema) is used instead.
* The terminal observation's `metadata` holds:
  * `reward_breakdown`: the v3 breakdown from `honest_agents.reward` (case, base, bonus, efficiency, penalty, and
    flags such as `passed`, `false_blame`, `blocker_check`, `category_match`, `target_match`, `leak`);
  * `outcome`: `tests_passed`, `pass_fraction`, tamper and violation details, the parsed report, the blocker verdict
    and the task's labels (`feasible`, `expected_blockers`, `expected_targets`, ...);
  * `stop_reason` (`report`, `call_budget`, `char_budget`, `silent_stop`, `truncated`), `masked_reason`,
    `grading_error`, and `debug` (test log, command history, cached blocker facts).
* `reward` is `None` only when the episode is masked: the sandbox or the verifier toolchain failed before the agent
  ran a command (`boot_failure`, `toolchain_self_check`), or a completion claim passed the tests of a variant labelled
  infeasible (`group_leak`: the label is wrong; the variant is appended to the quarantine list when
  `HONEST_QUARANTINE_FILE` or `HONEST_BANK_DIR` is set). A grading failure after the agent ran a command is scored as
  tamper. A boot failure during `reset` is returned as an error, and the client may retry the reset.
* Tasks with `[metadata.honest] verifier = "pytest"` are graded by the hermetic verifier with the network off. That
  needs the vendored pytest bundle on the server (`HONEST_VERIFIER_BUNDLE`, see `honest_agents.verifier`).
* `state` never exposes the honesty labels. They appear only in the terminal `metadata`.
* The group-level leak mask of training (every rollout of the same prompt is masked) has no counterpart here: a server
  session is one episode. A client that runs several episodes of one variant must apply it itself
  (`honest_agents.trl_rewards.mask_leaking_groups`).

## Running

The server and the training stack cannot share a venv: the `openenv` and `train` extras conflict on `mcp` (vllm).

```bash
uv sync --extra openenv            # in a separate checkout/venv from training
HONEST_TASKS_DIR=/srv/honest/tasks uvicorn honest_agents.openenv_env.server.app:app --port 8000
```

```python
from honest_agents.openenv_env import HonestEnvClient

with HonestEnvClient(base_url="http://localhost:8000").sync() as env:
    r = env.reset(task_id="train/hello")  # waits for the sandbox
    r = env.bash("echo hi > hello.txt")
    r = env.submit_report("complete", completed=["hello.txt"], confidence=0.9)
    print(r.reward, r.observation.metadata["reward_breakdown"]["case"])
    # or, when blocked:
    # env.abstain("jq is not installed", blocker_category="missing_capability", blocker_target="jq", confidence=0.9)
```

Use the WebSocket client. Episodes are stateful, and the stateless HTTP `/reset` and `/step` routes that OpenEnv adds
in simulation mode create a new environment for every request. `/reset` would boot a sandbox only to destroy it.
`HonestEnvClient` sets the message timeout to 1200 s, because `reset` can include an image build and the final step runs
the task's tests.

Task discovery follows OpenEnv's TaskProvider routes (`/honest_agents/splits`, `.../tasks`). The routes return task IDs
only. Split `all` contains every directory under `HONEST_TASKS_DIR` that has a `task.toml`. Each top-level
subdirectory (for example `train/`, `test/`) is also its own split.

| env var | default | |
|---|---|---|
| `HONEST_TASKS_DIR` | – | task root. Paths outside it are refused unless `HONEST_ALLOW_ANY_TASK_DIR=1` |
| `HONEST_MAX_CONCURRENT_ENVS` | 64 | WebSocket sessions; each one is one sandbox plus one loop thread |
| `HONEST_MAX_CONCURRENT_BOOTS` | 16 | process-wide cap on simultaneous `docker compose up` |
| `HONEST_SESSION_TIMEOUT_S` | 1800 | idle sessions are reaped, which stops leaked sandboxes. `0` disables reaping |
| `HONEST_MAX_STEPS` | 30 | tool-call turns per episode (TRL's `max_tool_calling_iterations`); bash calls = this − 2 |
| `HONEST_MAX_COMPLETION_TOKENS` | 16384 | total token budget the character budget is derived from |
| `HONEST_COMMAND_TIMEOUT_S` | 60 | per bash call (`timeout -k 5` inside the sandbox) |
| `HONEST_VERIFIER_BUNDLE` | user cache dir | vendored pytest bundle for the hermetic verifier |
| `HONEST_QUARANTINE_FILE` | – | where leaky "infeasible" variants are recorded |
| `HONEST_ENVIRONMENT_TYPE` | docker | Harbor backend (`docker`, `daytona`, `e2b`, ...) |
| `HONEST_ENFORCE_NETWORK` | 1 | apply the task's per-phase Harbor network policy |
| `HONEST_REWARD_CONFIG` / `HONEST_GRADING_CONFIG` | `{}` | JSON kwargs for `RewardConfig` / `GradingConfig` (e.g. `{"alpha": 0.5}`, `{"replay_evidence": true}`). `efficiency_ref_actions` defaults to the bash-call budget |
| `HONEST_TRIALS_DIR` | system tmp | Harbor trial dirs (see below) |

## Running in a container (`docker/openenv/`)

```bash
# the hermetic verifier's vendored pytest bundle is a named build context (the build fails without a valid one)
docker build -f docker/openenv/Dockerfile --build-context verifier_bundle=$HONEST_VERIFIER_BUNDLE \
  -t honest-agents-openenv .                                             # from the repo root
docker run --rm -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /srv/honest/tasks:/srv/honest/tasks:ro -v /srv/honest/trials:/srv/honest/trials \
  -e HONEST_TASKS_DIR=/srv/honest/tasks -e HONEST_TRIALS_DIR=/srv/honest/trials \
  honest-agents-openenv
```

The image is `python:3.12-slim` with the docker CLI and the compose plugin, not `openenv-base`. `openenv-base` ships
Python 3.11, and Harbor 0.23 needs Python 3.12 or later. Every sandbox is a sibling container started on the host
daemon through the mounted socket. Caveats (from `docs/research/`; inferred from Harbor's design and not yet tested
end to end):

1. **Bind-mount sources resolve on the host.** Harbor's docker backend shells out to `docker compose`, and any bind mount
   it or a task's own compose file declares is interpreted by the host daemon, using host paths. Mount the task root
   and `HONEST_TRIALS_DIR` at the **same path** inside the container as on the host, as in the command above.
   Build contexts are streamed by the CLI, so building images from task `environment/` dirs works either way.
   `HonestSession` passes no trial log mounts, which keeps the surface small.
2. **The socket is root on the host.** Anyone who can reach the server can run arbitrary commands in sandboxes, and the
   server can start any container on the host. Do not expose port 8000 beyond a trusted network.
3. **Network policies** (`no-network`, allowlist) use Harbor's egress sidecar (`NET_ADMIN`, nftables) on the **host**
   daemon. Hermetic (`verifier = "pytest"`) tasks switch to `no-network` for grading, so they need it even when the
   agent phase is public. Run `scripts/probe_node.sh` on the node first. If the host kernel lacks nftables `fib` support, starting an
   infeasible `no_network` variant raises an error instead of silently running with the network enabled.
4. Container-based GPU pods (RunPod, k8s) often have no Docker daemon or only a mounted socket with mismatched paths.
   There, use a VM or bare-metal node, or a remote Harbor backend (`HONEST_ENVIRONMENT_TYPE=daytona`; whether it
   enforces network policies is unverified).
5. `stop(delete=True)` runs `docker compose down --rmi local`. Compose-built images are tagged per compose project
   (= per session), so this removes only that session's tag; the BuildKit cache survives and later boots of the task
   rebuild from cache. Prebuilt `docker_image` tasks skip the build entirely.
6. Harbor guards image builds with a class-level `asyncio.Lock` keyed by `environment_name`, and an `asyncio.Lock`
   shared by two sessions' event loops deadlocks or raises "bound to a different event loop". Docker sessions therefore
   use a per-episode `environment_name`, and the first boot of a task without `docker_image` in this process holds a
   per-task thread lock so its image is built once; later boots run concurrently (up to `HONEST_MAX_CONCURRENT_BOOTS`).
   See `honest_agents.harness.start_sandbox`.

Sessions and teardown: `close()` (client disconnect or the idle reaper) waits for an in-flight reset or step to finish
before stopping the sandbox. OpenEnv's reaper also reaps WebSocket-owned sessions whose last *completed* message is
older than `HONEST_SESSION_TIMEOUT_S`, so keep that value above the longest boot plus grading time. A reaped session
answers further messages with an error, and the client must reconnect. Split and index listings are cached per
process, so restart the server after adding tasks (`task_id` resets always see the filesystem).

## Layout

* `session.py`: `HonestSession`, the trl-free and openenv-free episode driver (boot, budgeted exec, report, grade).
  `honest_agents.env` needs trl, so only the boot sequence and the `bash` exec wrapper are restated here; the budget
  (`harness.BudgetMeter`), report parsing, grading, reward and quarantine are the shared core.
* `models.py`: `HonestAction`, `HonestObservation`, `HonestState`.
* `server/environment.py`: `HonestEnvironment(Environment)`. It has `SUPPORTS_CONCURRENT_SESSIONS = True` and runs a
  private event-loop thread per instance, so slow sandbox calls never block the server loop.
* `server/app.py`: `build_app()` / `app`, which wraps `create_app` with a `ConcurrencyConfig`.
* `client.py`: `HonestEnvClient(EnvClient)` with `bash` / `submit_report` / `abstain` / `end` helpers.
