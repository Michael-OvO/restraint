#!/usr/bin/env bash
# vLLM generation server for GRPO server mode: one GPU per run slot (SLOT=0: GPU 1, port 8000; SLOT=1: GPU 3, port 8001).
#   scripts/serve_vllm.sh configs/base.yaml [dotted.key=value overrides...] [extra vllm serve args...]
# Pass the SAME overrides the training run gets (e.g. grpo.max_completion_length=32768): `--max-model-len` is computed
# from the run's limits (prompt reserve + total budget + per-turn cap) and asserted against this server at train start.
# Env: SLOT (0), VLLM_GPUS / VLLM_PORT (from SLOT), VLLM_TP (default: #GPUs), VLLM_GPU_MEM (0.9),
#      VLLM_MAX_MODEL_LEN (only to RAISE the window; lower than the computed value is refused),
#      EVAL=1 to also enable the OpenAI tool/reasoning parsers (evaluation; not needed for training).
# Flags mirror trl 1.14.1 scripts/vllm_serve.py (the deprecated `trl vllm-serve`), checked against vllm 0.30.0.
# `--generation-config vllm` makes vLLM ignore the checkpoint's generation_config.json (Qwen3-4B: temperature 0.6,
# top_k 20, top_p 0.95), so a request that leaves a field out is still sampled from the full distribution.
source "$(dirname "$0")/_common.sh"
CONFIG="${1:?usage: serve_vllm.sh CONFIG [dotted.key=value ...] [vllm args...]}"; shift
OVERRIDES=(); EXTRA=()
for a in "$@"; do if is_override "$a"; then OVERRIDES+=("$a"); else EXTRA+=("$a"); fi; done
slot_env

read -r MODEL FAMILY < <(model_from_config "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
[[ -n "${MODEL:-}" && -n "${FAMILY:-}" ]] || { echo "could not resolve model/family from $CONFIG" >&2; exit 1; }
LIMITS="$(limits_json "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})"
NEED_LEN="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["max_model_len"])' "$LIMITS")"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-$NEED_LEN}"
(( MAX_LEN >= NEED_LEN )) || { echo "VLLM_MAX_MODEL_LEN=$MAX_LEN is below the $NEED_LEN this run needs" >&2; exit 1; }

GPUS="$VLLM_GPUS"
TP="${VLLM_TP:-$(awk -F, '{print NF}' <<<"$GPUS")}"
FAMILY_FLAGS=()
[[ "$FAMILY" == "qwen3_5" ]] && FAMILY_FLAGS+=(--language-model-only)
if [[ "${EVAL:-0}" == 1 ]]; then
  if [[ "$FAMILY" == "qwen3_5" ]]; then TOOL_PARSER=qwen3_coder; else TOOL_PARSER=hermes; fi
  FAMILY_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3
                 --default-chat-template-kwargs '{"enable_thinking": false}')
fi

echo "serving $MODEL (family $FAMILY) on GPUs $GPUS, TP=$TP, port $VLLM_PORT, max_model_len $MAX_LEN (run needs $NEED_LEN)"
export CUDA_VISIBLE_DEVICES="$GPUS" VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
exec "${UV_RUN[@]}" vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$VLLM_PORT" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "${VLLM_GPU_MEM:-0.9}" \
  --max-model-len "$MAX_LEN" \
  --dtype bfloat16 \
  --enable-prefix-caching \
  --weight-transfer-config '{"backend":"nccl"}' \
  --logprobs-mode processed_logprobs \
  --max-logprobs -1 \
  --enable-scale-out \
  --generation-config vllm \
  ${FAMILY_FLAGS[@]+"${FAMILY_FLAGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
