#!/usr/bin/env bash
# AbstentionBench (single-turn QA abstention; checks RL did not break / over-trigger abstention outside agents).
#   MODEL=policy JUDGE_BASE_URL=http://localhost:8001/v1 scripts/eval_abstentionbench.sh [OUT=runs/abstentionbench/MODEL]
# Env: BASE_URL (policy), JUDGE_BASE_URL (required; serve their default judge, e.g.
#      `uv run --frozen --extra train vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8001` on a free GPU),
#      JUDGE_MODEL (meta-llama/Llama-3.1-8B-Instruct), LIMIT (all), CONCURRENCY (64).
# The HF dataset is a loading script that needs datasets<=3.6.0, so it is exported to JSONL once from a separate py3.11
# venv under $HONEST_EXT_DIR/abstentionbench-export, and refused unless its sha256 is the one that
# data/benchmarks/final_benchmark.json froze: the first run exports, then stops until that export is archived and frozen
# (ledger DEF-50). Data license: CC BY-NC 4.0 (research use only).
source "$(dirname "$0")/eval_common.sh"
require_model
: "${JUDGE_BASE_URL:?set JUDGE_BASE_URL to an OpenAI-compatible endpoint serving the judge model}"
OUT="${1:-runs/abstentionbench/${MODEL//\//_}}"
AB="$EXT_DIR/AbstentionBench"
EXPORT_DIR="$EXT_DIR/abstentionbench-export"
DATA="$EXT_DIR/abstentionbench.jsonl"
clone_pinned "$ABSTENTIONBENCH_REPO" "$ABSTENTIONBENCH_COMMIT" "$AB"   # only for the judge prompt
if [[ ! -s "$DATA" ]]; then
  mkdir -p "$EXPORT_DIR"
  PY="$(ext_venv "$EXPORT_DIR" 3.11)"
  ext_pip "$PY" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch
  ext_pip "$PY" "datasets==3.6.0" gdown pandas pydantic jsonlines requests wget numpy
  "$PY" "$REPO_ROOT/src/honest_agents/eval/external/abstentionbench_export_standalone.py" --out "$DATA" \
    --revision "$ABSTENTIONBENCH_HF_REVISION"
fi
# the items are frozen by the export's sha256 (ledger DEF-50): no arm runs on any other export, nor before the freeze
"${UV_RUN[@]}" python -m honest_agents.eval.final_benchmark verify-export --bench abstentionbench "$DATA"
"${UV_RUN[@]}" python -m honest_agents.eval.external.abstentionbench run \
  --data "$DATA" --out "$OUT" --base-url "$BASE_URL" --model "$MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" --judge-model "${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}" \
  --judge-prompt-file "$AB/recipe/evaluation_judge_prompts.py" \
  --concurrency "${CONCURRENCY:-64}" --no-thinking ${LIMIT:+--limit "$LIMIT"}
