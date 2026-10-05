#!/usr/bin/env bash
# Report-only pilot (docs/design.md section 10, docs/roadmap.md Stage 2) in one run slot: for every arm of the sweep, start this slot's vLLM
# server, train that arm against it, stop the server. One arm per server because each arm opens its own weight-sync
# group (honest_agents.pilot.run warns about several arms in one process). Run scripts/sync_check.sh first: this is the
# first run that learns (docs/design.md section 8).
#   SLOT=0 SHARD=0/2 scripts/run_pilot.sh [configs/pilot.yaml] [key.path=value ...]
#   SLOT=1 SHARD=1/2 scripts/run_pilot.sh ...        # two shards use the two run slots of a 4-GPU node
# Env: SLOT (0), SHARD (K/N, default all arms), RESUME=1, FORCE=1, VLLM_* (scripts/serve_vllm.sh).
# Prints the verdict (PILOT GO / NO-GO / INCONCLUSIVE; exit 0 / 1 / 2) when the whole sweep ran in this invocation.
source "$(dirname "$0")/_common.sh"
CONFIG="configs/pilot.yaml"
if [[ $# -gt 0 && "$1" == *.yaml ]]; then CONFIG="$1"; shift; fi
slot_env
load_slot_overrides
PILOT=("${UV_RUN[@]}" python -m honest_agents.pilot.run --config "$CONFIG")
SHARD_ARGS=(); [[ -n "${SHARD:-}" ]] && SHARD_ARGS=(--shard "$SHARD")
FLAGS=(); [[ "${RESUME:-0}" == 1 ]] && FLAGS+=(--resume); [[ "${FORCE:-0}" == 1 ]] && FLAGS+=(--force)

# Fail before any GPU time is spent: dataset, grpo section, context window, epochs.
"${PILOT[@]}" --dry-run "${SLOT_OV[@]}" "$@" >/dev/null
ARMS=()
while IFS= read -r a; do ARMS+=("$a"); done < <("${PILOT[@]}" --list-arms ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"} "$@" 2>/dev/null)
(( ${#ARMS[@]} > 0 )) || { echo "no arms selected" >&2; exit 1; }
mkdir -p outputs/logs
trap stop_vllm EXIT INT TERM
rc=0
for arm in "${ARMS[@]}"; do
  TAG="pilot_${arm}_slot${SLOT}"
  start_vllm "outputs/logs/${TAG}_vllm.log" "$CONFIG" "${SLOT_OV[@]}" "$@" || { rc=1; break; }
  wait_vllm || { rc=1; stop_vllm; break; }
  set +e
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "${PILOT[@]}" --arm "$arm" ${FLAGS[@]+"${FLAGS[@]}"} "${SLOT_OV[@]}" "$@" \
    2>&1 | tee "outputs/logs/${TAG}_train.log"
  (( ${PIPESTATUS[0]} == 0 )) || rc=1
  set -e
  stop_vllm
done
if [[ -z "${SHARD:-}" && $rc == 0 ]]; then
  "${PILOT[@]}" --analyze-only "$@" || rc=$?
fi
exit "$rc"
