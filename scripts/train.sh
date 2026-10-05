#!/usr/bin/env bash
# Launch one GRPO run: ONE training process on ONE GPU, against the vLLM server of the same run slot
# (scripts/serve_vllm.sh). Two runs can share a 4-GPU node: SLOT=0 (train GPU 0, vLLM GPU 1, port 8000) and SLOT=1
# (train GPU 2, vLLM GPU 3, port 8001). Use scripts/run_slot.sh to start the server and the run together.
#   SLOT=1 scripts/train.sh configs/honest_alpha0.25.yaml [key.path=value ...]
# Env: SLOT (0), TRAIN_GPUS / VLLM_PORT / GROUP_PORT (from SLOT), ACCEL_CONFIG (configs/accelerate/single_gpu.yaml),
#      WAIT_VLLM=1 (default), VLLM_WAIT_TIMEOUT (3600 s), RESUME=1 to continue from the run's last checkpoint-*.
source "$(dirname "$0")/_common.sh"
CONFIG="${1:?usage: train.sh CONFIG [overrides...]}"; shift
slot_env
load_slot_overrides
ACCEL="${ACCEL_CONFIG:-configs/accelerate/single_gpu.yaml}"

# Validate the config (and catch typos, unsafe settings, a bad dataset) before paying for model loading.
"${UV_RUN[@]}" python -m honest_agents.train --config "$CONFIG" --dry-run "${SLOT_OV[@]}" "$@" >/dev/null

if [[ "${WAIT_VLLM:-1}" == 1 ]] && uses_vllm_server "$CONFIG" "${SLOT_OV[@]}" "$@"; then
  wait_vllm
fi

export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
FLAGS=()
[[ "${RESUME:-0}" == 1 ]] && FLAGS+=(--resume)
exec "${UV_RUN[@]}" accelerate launch --config_file "$ACCEL" --num_processes 1 \
  -m honest_agents.train --config "$CONFIG" ${FLAGS[@]+"${FLAGS[@]}"} "${SLOT_OV[@]}" "$@"
