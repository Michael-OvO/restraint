#!/usr/bin/env bash
# One complete run in one slot: start this slot's vLLM server, train against it, stop the server.
#   SLOT=0 scripts/run_slot.sh configs/honest_alpha0.25.yaml [key.path=value ...]
# Logs: outputs/logs/<run>_slot<N>_{vllm,train}.log. The exit status is the training run's.
# Env: SLOT, TAG (log name; default: config name), RESUME=1, everything train.sh / serve_vllm.sh read.
source "$(dirname "$0")/_common.sh"
CONFIG="${1:?usage: run_slot.sh CONFIG [overrides...]}"; shift
slot_env
TAG="${TAG:-$(basename "$CONFIG" .yaml)}_slot${SLOT}"
mkdir -p outputs/logs
trap stop_vllm EXIT INT TERM
load_slot_overrides
if uses_vllm_server "$CONFIG" "${SLOT_OV[@]}" "$@"; then
  start_vllm "outputs/logs/${TAG}_vllm.log" "$CONFIG" "$@"
else
  echo "$CONFIG generates in-process (colocate): no vLLM server"
fi
set +e
"$REPO_ROOT/scripts/train.sh" "$CONFIG" "$@" 2>&1 | tee "outputs/logs/${TAG}_train.log"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
