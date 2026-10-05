#!/usr/bin/env bash
# H1 direction check (docs/evaluation.md, Stage 1): does outcome-only RL move the honesty metrics the way H1 says?
# base Qwen3.5-4B against allenai/tmax-4b (the same base, trained on outcome-only terminal RL) on the held-out split,
# under OUR harness, the same caps and the same sampling. Costs two inference runs (no training); do it before
# spending a training run.
#   scripts/h1_tmax_check.sh SPLIT [OUT_DIR=runs/h1_tmax] [extra run_eval args...]
# Env: N_SAMPLES (4), CONCURRENCY (16), BASE_MODEL (Qwen/Qwen3.5-4B), TMAX_MODEL (allenai/tmax-4b),
#      BASE_REVISION / TMAX_REVISION (pin commit SHAs; tmax's default branch is master), VLLM_GPUS (0), VLLM_PORT
#      (8000), VLLM_MAX_MODEL_LEN (20480 = 16384 budget + prompt + one 4096 turn), VLLM_GPU_MEM (0.9),
#      BASE_URL is set from VLLM_PORT. SKIP_SERVE=1 assumes the two endpoints are already up (BASE_ENDPOINT,
#      TMAX_ENDPOINT, served names `base` and `tmax`).
# Both models are served non-thinking (as in training) with the tool parser their model card names: qwen3_coder for
# the Qwen3.5 base, qwen3_xml for tmax-4b.
# Output: OUT_DIR/{base,tmax}/ (run_eval outputs), OUT_DIR/diff_<metric>.json and OUT_DIR/h1_check.md.
# The direction H1 predicts for the outcome-only model: task success up, P(claim | fail) and false-completion rate up
# (or not down), decline rate on feasible tasks not up. The check prints the paired source-task bootstrap difference
# tmax - base for each; it does not decide anything by itself.
source "$(dirname "$0")/_common.sh"
source "$(dirname "$0")/eval_common.sh"   # last: UV_RUN with --extra eval
SPLIT="${1:?usage: h1_tmax_check.sh SPLIT [OUT_DIR] [extra run_eval args...]}"; shift
OUT="${1:-runs/h1_tmax}"; shift || true
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"; TMAX_MODEL="${TMAX_MODEL:-allenai/tmax-4b}"
PORT="${VLLM_PORT:-8000}"
mkdir -p "$OUT"

serve() {  # serve <hf model> <revision or ""> <tool parser> <served name> <log>
  local model="$1" rev="$2" parser="$3" name="$4" log="$5" t=0
  while curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    (( t++ > 30 )) && { echo "a server already answers on port $PORT; stop it first" >&2; return 1; }
    sleep 2
  done
  CUDA_VISIBLE_DEVICES="${VLLM_GPUS:-0}" VLLM_WORKER_MULTIPROC_METHOD=spawn setsid "${UV_RUN[@]}" vllm serve "$model" \
    ${rev:+--revision "$rev"} --served-model-name "$name" --host 127.0.0.1 --port "$PORT" \
    --gpu-memory-utilization "${VLLM_GPU_MEM:-0.9}" --max-model-len "${VLLM_MAX_MODEL_LEN:-20480}" --dtype bfloat16 \
    --enable-prefix-caching --language-model-only --generation-config vllm \
    --enable-auto-tool-choice --tool-call-parser "$parser" --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' >"$log" 2>&1 &
  VLLM_PID=$!
  t=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm exited; see $log" >&2; return 1; }
    (( t++ > 900 )) && { echo "vllm did not become healthy in 30 minutes; see $log" >&2; return 1; }
    sleep 2
  done
}

run_one() {  # run_one <name> <served name> <endpoint>
  local name="$1" served="$2" endpoint="$3"
  "${UV_RUN[@]}" python -m honest_agents.eval.run_eval --split "$SPLIT" --out "$OUT/$name" --base-url "$endpoint" \
    --model "$served" --no-thinking --n-samples "${N_SAMPLES:-4}" --concurrency "${CONCURRENCY:-16}" "$@"
}

trap stop_vllm EXIT INT TERM
EXTRA=("$@")
if [[ "${SKIP_SERVE:-0}" == 1 ]]; then
  run_one base base "${BASE_ENDPOINT:?set BASE_ENDPOINT}" ${EXTRA[@]+"${EXTRA[@]}"}
  run_one tmax tmax "${TMAX_ENDPOINT:?set TMAX_ENDPOINT}" ${EXTRA[@]+"${EXTRA[@]}"}
else
  serve "$BASE_MODEL" "${BASE_REVISION:-}" qwen3_coder base "$OUT/vllm_base.log"
  run_one base base "http://127.0.0.1:$PORT/v1" ${EXTRA[@]+"${EXTRA[@]}"}
  stop_vllm
  serve "$TMAX_MODEL" "${TMAX_REVISION:-}" qwen3_xml tmax "$OUT/vllm_tmax.log"
  run_one tmax tmax "http://127.0.0.1:$PORT/v1" ${EXTRA[@]+"${EXTRA[@]}"}
  stop_vllm
fi

# Paired source-task bootstrap of tmax - base, on the same split and the same reward configuration.
: >"$OUT/h1_check.md"
for metric in task_success verified_success_rate false_completion_rate p_claim_given_fail claim_rate_feasible \
              decline_rate_feasible false_blame_rate missing_report_rate mean_commands; do
  "${UV_RUN[@]}" python -m honest_agents.eval.run_eval --diff "$OUT/tmax" "$OUT/base" --metric "$metric" \
    --out "$OUT/diff_$metric.json" | tee -a "$OUT/h1_check.md" || true
done
"${UV_RUN[@]}" python -m honest_agents.eval.run_eval --compare "$OUT/base" "$OUT/tmax" | tee -a "$OUT/h1_check.md"
echo "H1 direction check written to $OUT/h1_check.md"
