# Versions

Checked 2026-10-05 against PyPI and GitHub. `uv.lock` is the source of truth; this page records why each pin is where it
is and what to re-check on the GPU node. Policy: the trainer overrides private TRL internals (`HonestGRPOTrainer`) and
depends on exact Harbor, vLLM and OpenEnv behaviour, so API-sensitive pins move only when the new version is inside the
range the other pins declare, the code this repo calls was diffed with no behavioural change, and the full suite passes.
Tooling and pure-Python utilities follow the latest release the resolver allows.

| Package | Pinned (lock) | Latest | Decision | Reason |
|---|---|---|---|---|
| harbor | 0.24.0 (was 0.23.0) | 0.24.0 | bumped | Diffed every Harbor module the repo imports (see below): no behavioural change on the paths we use. Suite green. Unverified without Docker. |
| trl | 1.14.1 | 1.14.1 | current | Declares `vllm<=0.30.0,>=0.20.0`. |
| vllm | 0.30.0 | 0.31.0 | kept | TRL 1.14.1 caps vllm at 0.30.0, and 0.31.0 also needs `transformers<5.18.0` (the lock has 5.18.0, the latest). Re-check when TRL raises its cap. |
| peft | 0.21.2 | 0.21.2 | current | |
| transformers | 5.18.0 | 5.18.0 | current | Transitive; vllm 0.30.0 accepts it. |
| accelerate | 1.15.0 | 1.15.0 | current | Used only through `accelerate launch` (`scripts/train.sh`) and by TRL. |
| datasets | 5.0.1 | 5.0.1 | current | |
| deepspeed | removed | 0.19.7 | removed | Referenced nowhere (no config, script or import); LoRA on one GPU per run needs no ZeRO. Dropped from the `train` extra (also drops hjson, msgpack). |
| openenv | 0.7.0 | 0.7.0 | current | Separate extra; conflicts with `train` over `mcp` (see `pyproject.toml`). |
| numpy | 2.5.3 | 2.5.3 | current | Floor stays `>=1.26` (only lower bound needed). |
| openai | 2.54.0 | 3.24.0 | kept (cap) | `litellm` (a Harbor dependency, 1.103.3) requires `openai<3`; 2.54.0 is the newest allowed. The repo uses only `AsyncOpenAI` chat completions. |
| pydantic | 2.13.5 | 2.13.5 | current | |
| pytest / pytest-asyncio / pytest-xdist | 9.1.1 / 1.4.0 / 3.8.0 | same | current | |
| ruff | 0.16.10 | 0.16.10 | current | |
| uv_build (build backend) | `>=0.12.23,<0.13.0` (was `>=0.11.32,<0.12.0`) | 0.12.23 | bumped | Build isolation fetches it; `uv sync` builds the package fine. |
| uv (the tool) | 0.11.32 on the dev Mac | 0.12.23 | not installed | Installing software is out of scope for the unattended run; the lock and `uv lock --check` work with 0.11.32. Run `uv self update` on the node and the Mac. |
| torch | 2.13.0 (via vllm 0.30.0) | 2.14.1 | kept | vllm pins `torch==2.13.0` exactly. |
| flash-linear-attention | not in the lock | 0.5.2 | kept out | See the node re-check list. |
| `python:3.12-slim-bookworm` (OpenEnv image) | tag | `3.12-slim-trixie` exists | kept | The Dockerfile installs Docker's apt repo for bookworm; moving to trixie needs a Docker build to verify. |
| `alpine:3.23.4@sha256:5b10...` (`scripts/probe_node.sh`) | digest | Harbor 0.24.0 uses the identical digest | kept | Must equal Harbor's `_EGRESS_CONTROL_KERNEL_PROBE_IMAGE` (unchanged 0.23.0 to 0.24.0). |
| agentic-abstention `ebbf1f5...` | commit | HEAD is the same commit (2026-09-28) | current | |
| AgentAbstain GitHub `cfc3faf...` | commit | HEAD is the same commit (2026-09-28) | current | |
| agentabstain HF dataset `8422284...` | commit | HEAD is the same commit (2026-07-17) | current | |
| AbstentionBench `e291841...` | commit | HEAD is the same commit (2025-08-27) | current | |
| pytest in the verifier bundle (`8.4.1`) | pin | 9.1.1 | kept | Must match the task images' own `test.sh` pin, not the dev pytest. |

## Harbor 0.23.0 to 0.24.0: what was diffed

Wheels of both versions were unpacked and the modules this repo imports or relies on were compared.

- Byte-identical: `models.agent.context`, `utils.env`, `trial.network_policy` (`resolve_trial_network_plan`),
  `models.trial.paths`, `models.trial.result`, `agents.options`, `trial.artifact_handler`, the egress sidecar directory,
  `docker-compose-egress-control.yaml`, and the kernel probe image digest.
- Additive only, with defaults that keep old behaviour: `BaseEnvironment.__init__` gained `stream=False` and
  `enable_environment_dir_upload=True`; `EnvironmentConfig.stream=False`; `EnvironmentCapabilities.stream=False`;
  `EnvironmentFactory` gained new environment types and `connect_ssh`; `BaseAgent.DEFAULT_SETUP_TIMEOUT_SEC` (360.0, the
  value `Trial._AGENT_SETUP_TIMEOUT_SEC` had); `TrialEnvironmentConfig` field descriptions.
- Behaviour changes that do not touch our paths: tar extraction filter now skips FIFOs and device nodes instead of
  aborting (`transfer_tar_filter`); `Task.load` resolves the verifier image through
  `resolve_verifier_environment_definition` (returns `None` for tasks with no separate verifier, which is every task we
  build, and we pass `VerifierEnvironmentMode.SHARED`); `DockerEnvironment._write_resources_compose_file` is now async and
  only changes output when GPUs are requested (our task configs request none); the image-content hash for prebuilt
  detection is now streamed and prefixed for files >= 4 GiB only.
- `DockerEnvironment._image_build_locks` (the class-level dict that `harness.py` works around) still exists with the same
  type and use.

## Re-check on the GPU node (first boot, before paid training)

1. Run `scripts/probe_node.sh` and a one-task `scripts/smoke_test.sh`: Harbor 0.24.0 was only exercised with fakes on the
   Mac (no Docker). Specifically a real `DockerEnvironment` start, exec, upload, download, stop with egress control on,
   and the verifier-in-place grading path.
2. `flash-linear-attention` (latest 0.5.2, depends on `fla-core==0.5.2` and `transformers>=4.45`): not in `uv.lock` and not
   installed by `setup_node.sh`, so Qwen3.5 runs about 20x slower without it. Install it after `uv sync`
   (`uv pip install flash-linear-attention==0.5.2`) and confirm `import fla` works, that Triton compiles for the H100, and
   that the model logs the fast path rather than the torch fallback. If it pulls a different torch, stop: vLLM 0.30.0
   needs `torch==2.13.0` exactly.
3. CUDA and torch: vLLM 0.30.0's wheel is built against torch 2.13.0; check `nvidia-smi` driver support for the CUDA
   build that wheel names, and that `torch.cuda.is_available()` is true before the first run.
4. TRL source fingerprints (the trainer checks them at construction) and the startup probe for vLLM logprob handling
   (`docs/node-runbook.md`) must pass unchanged; any TRL or vLLM bump must redo that diff.
5. Revisit the vllm cap when TRL raises it above 0.30.0 (vllm 0.31.0 also needs `transformers<5.18.0`, so it would
   force a transformers downgrade as well).
