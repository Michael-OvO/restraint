# Sourced by the other scripts. Resolves the repo root and the model family for a run config, the run slot (which GPUs
# and ports a run owns) and the vLLM server lifecycle.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
UV_RUN=(env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval)

# is_pilot_config <config.yaml>: the report-only pilot's config has a top-level `sweep:` key and is not a
# `honest_agents.train` run config (`python -m honest_agents.pilot.run` reads it).
is_pilot_config() { grep -qE '^sweep:' "$1"; }

# model_from_config <config.yaml> [overrides...] -> prints "<model name> <family>" (resolves `base:` inheritance, offline).
model_from_config() {
  if is_pilot_config "$1"; then
    "${UV_RUN[@]}" python -m honest_agents.pilot.run --config "$@" --print-model
    return
  fi
  "${UV_RUN[@]}" python - "$@" <<'PY'
import sys
from honest_agents.train import load_run_config, resolve_family
cfg = load_run_config(sys.argv[1], sys.argv[2:])
print(cfg.model.name, resolve_family(cfg))
PY
}

# limits_json <config.yaml> [overrides...] -> JSON with max_model_len, the three limits and the server url/group port.
# The vLLM window comes from the same numbers the trainer and the environment use (honest_agents.train.build_limits).
limits_json() {
  local cfg="$1"; shift
  if is_pilot_config "$cfg"; then
    "${UV_RUN[@]}" python -m honest_agents.pilot.run --config "$cfg" --print-limits "$@"
    return
  fi
  "${UV_RUN[@]}" python -m honest_agents.train --config "$cfg" --print-limits "$@"
}

# uses_vllm_server <config.yaml> [overrides...]: exit 0 if the run generates through a separate vLLM server.
uses_vllm_server() {
  if is_pilot_config "$1"; then return 0; fi  # the pilot always samples its report turn from a server
  "${UV_RUN[@]}" python - "$@" <<'PY'
import sys
from honest_agents.train import load_run_config
g = load_run_config(sys.argv[1], sys.argv[2:]).grpo
sys.exit(0 if g.get("use_vllm") and g.get("vllm_mode", "colocate") == "server" else 1)
PY
}

# is_override <arg>: true for a config override (`grpo.max_steps=3`, `seed=7`), false for vLLM flags and their values
# (`--lora-modules policy=outputs/x` must reach `vllm serve` untouched, so only the config's own top-level keys count).
is_override() { [[ "$1" =~ ^(model|lora|data|env|reward|launch|vllm|probe|stop|bank|trainer|grpo|sweep|analysis)\.[A-Za-z0-9_.]+= || "$1" =~ ^(seed|output_root|run_name)= ]]; }

# alpha_flags: fills ALPHA_FLAGS from PHASE0_SUMMARY (path of a phase0_summary.json) and ALLOW_UNSAFE_ALPHA=1. With a
# summary, `honest_agents.train` refuses an honesty-reward alpha whose claim threshold (1 + alpha) / 2 exceeds the Phase-0
# pool pass rate minus 0.1 (exit status 4); without one nothing is checked.
alpha_flags() {
  ALPHA_FLAGS=()
  [[ -n "${PHASE0_SUMMARY:-}" ]] && ALPHA_FLAGS+=(--phase0-summary "$PHASE0_SUMMARY")
  [[ "${ALLOW_UNSAFE_ALPHA:-0}" == 1 ]] && ALPHA_FLAGS+=(--allow-unsafe-alpha)
  return 0
}
# check_alpha <config.yaml> [overrides...]: exit 0 if the run's alpha is allowed (or nothing to check), 4 if refused
# (the reason is on stderr). Cheap to call before starting a vLLM server.
check_alpha() {
  local cfg="$1"; shift
  [[ -n "${PHASE0_SUMMARY:-}" ]] || return 0
  is_pilot_config "$cfg" && return 0
  alpha_flags
  "${UV_RUN[@]}" python -m honest_agents.train --config "$cfg" --check-alpha "${ALPHA_FLAGS[@]}" "$@"
}

# Run slots. A 4-GPU node runs two concurrent runs, each with ONE training GPU and ONE vLLM GPU (TRL slices gathered
# completions per rank, so a run may not span two training processes; docs/design.md section 8). Slot s owns GPUs 2s (training) and
# 2s+1 (vLLM), vLLM port 8000+s and NCCL weight-transfer port 51216+s. Any of the variables may be set to override.
slot_env() {
  local s="${SLOT:-0}"
  [[ "$s" =~ ^[0-9]+$ ]] || { echo "SLOT must be a non-negative integer, got '$s'" >&2; return 1; }
  : "${TRAIN_GPUS:=$((2 * s))}"
  : "${VLLM_GPUS:=$((2 * s + 1))}"
  : "${VLLM_PORT:=$((8000 + s))}"
  : "${GROUP_PORT:=$((51216 + s))}"
  if [[ "$TRAIN_GPUS" == *,* ]]; then echo "TRAIN_GPUS=$TRAIN_GPUS: one training GPU per run (docs/design.md section 8)" >&2; return 1; fi
  export SLOT="$s" TRAIN_GPUS VLLM_GPUS VLLM_PORT GROUP_PORT
}
# slot_overrides: the config overrides that point a run at its slot's server and rendezvous port (one per line).
slot_overrides() {
  echo "grpo.vllm_server_base_url=http://127.0.0.1:${VLLM_PORT}"
  echo "grpo.vllm_group_port=${GROUP_PORT}"
  echo "launch.num_processes=1"
}

# load_slot_overrides: fills the array SLOT_OV from slot_overrides (no `mapfile`, which bash 3.2 on a laptop lacks).
load_slot_overrides() {
  SLOT_OV=()
  local line
  while IFS= read -r line; do SLOT_OV+=("$line"); done < <(slot_overrides)
}

# vLLM server lifecycle shared by sweep/smoke/phase0 scripts. The server runs in its own session (setsid) so that
# stopping it kills the whole tree (uv -> vllm api server -> workers); killing only the top pid can leave workers
# holding the GPU and port, and the next run's /health wait would then talk to a dying server.
VLLM_PID="${VLLM_PID:-}"  # inherited by train.sh when run_slot.sh started the server: wait_vllm can see it die
vllm_url() { echo "http://127.0.0.1:${VLLM_PORT:-8000}"; }
start_vllm() {  # start_vllm <log> <serve_vllm.sh args...>
  local log="$1"; shift
  local t=0
  while curl -sf "$(vllm_url)/health" >/dev/null 2>&1; do
    (( t++ > 60 )) && { echo "a vLLM server is already answering on $(vllm_url); stop it first" >&2; return 1; }
    sleep 2
  done
  mkdir -p "$(dirname "$log")"
  setsid "$REPO_ROOT/scripts/serve_vllm.sh" "$@" >"$log" 2>&1 &
  VLLM_PID=$!
  export VLLM_PID
}
wait_vllm() {  # wait_vllm [timeout_sec]: fail fast if the server process died instead of waiting out the timeout
  local deadline=$(( SECONDS + ${1:-${VLLM_WAIT_TIMEOUT:-3600}} ))
  echo "waiting for vLLM at $(vllm_url) ..."
  until curl -sf "$(vllm_url)/health" >/dev/null 2>&1; do
    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM exited before it became healthy" >&2; return 1; fi
    (( SECONDS < deadline )) || { echo "vLLM at $(vllm_url) not healthy after ${1:-${VLLM_WAIT_TIMEOUT:-3600}}s" >&2; return 1; }
    sleep 5
  done
}
stop_vllm() {
  [[ -n "$VLLM_PID" ]] || return 0
  kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
  local t=0
  while kill -0 -- "-$VLLM_PID" 2>/dev/null; do
    (( t++ > 60 )) && { kill -KILL -- "-$VLLM_PID" 2>/dev/null || true; break; }
    sleep 1
  done
  wait "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""
}
