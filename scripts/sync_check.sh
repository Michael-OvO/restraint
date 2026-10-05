#!/usr/bin/env bash
# Weight-sync check (SPEC_V3 10.1.2): with a NON-ZERO adapter, do the log-probabilities vLLM assigns to a fixed set of
# token sequences (prompt_logprobs, 2k-20k tokens) match the trainer's own after TRL's merged-LoRA sync? A zero-lr run
# cannot tell (the delta is zero). Run it once before the report-only pilot, which is the first run that learns.
#   scripts/sync_check.sh [CONFIG=configs/base.yaml] [key.path=value ...] [-- sync_check args]
# Starts this slot's vLLM server, builds the real trainer on the slot's training GPU, syncs a random adapter with the
# patched path (and, with --compare-stock, with TRL's stock path) and compares. Exit 0 = pass. Writes
# outputs/sync_check/<config>_slot<N>.json. Pass: mean importance ratio within 1% of 1, mean |gap| no larger than the base
# model's, no NaN. If it fails: adapter-only sync, float32 serving, or Qwen3-4B (the JSON's "recommendation").
# --bank DIR   use the token ids of banked Phase-0 trajectories instead of synthetic transcripts.
# --tiny-adapter   second pass (run it too): an adapter of one learning step (B std 1e-5) instead of std 0.01, which is
#                  20-60x larger than early training deltas. Passes only if at least half of the delta survives the bf16
#                  merge into the served dtype (--served-dtype, default bfloat16); a failure means early updates never reach
#                  the sampler (serve float32, sync the adapter only, or raise the learning rate).
source "$(dirname "$0")/_common.sh"
CONFIG="configs/base.yaml"
if [[ $# -gt 0 && "$1" == *.yaml ]]; then CONFIG="$1"; shift; fi
OVERRIDES=(); EXTRA=("--compare-stock")
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; EXTRA+=("$@"); break; fi
  OVERRIDES+=("$1"); shift
done
slot_env
load_slot_overrides
mkdir -p outputs/sync_check
TAG="$(basename "$CONFIG" .yaml)_slot${SLOT}"
trap stop_vllm EXIT INT TERM
start_vllm "outputs/sync_check/${TAG}_vllm.log" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"}
wait_vllm
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
"${UV_RUN[@]}" python -m honest_agents.sync_check --config "$CONFIG" --output "outputs/sync_check/${TAG}.json" \
  ${EXTRA[@]+"${EXTRA[@]}"} "${SLOT_OV[@]}" ${OVERRIDES[@]+"${OVERRIDES[@]}"}
