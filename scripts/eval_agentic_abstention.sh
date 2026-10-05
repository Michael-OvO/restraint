#!/usr/bin/env bash
# Agentic Abstention terminal subset (immediate instruction-level rewrites of Terminal-Bench 2.0) via Harbor + Docker.
#   MODEL=policy scripts/eval_agentic_abstention.sh [JOBS_DIR=runs/harbor]
# Env: BASE_URL, N_CONCURRENT (8), MAX_TURNS (10 = their max_interaction_rounds, for comparable timely recall).
# Steps: pinned checkout → TB 2.0 tasks exported by `harbor download` → their materializer (stdlib + toml, run in an
# ephemeral `uv run --no-project --with toml` env) → harbor run with HonestPolicyAgent → our timely-recall scorer.
# The 21 delayed tasks are released only as prose specs (no task dirs); see docs/EVAL.md for the rebuild plan.
source "$(dirname "$0")/eval_common.sh"
JOBS_DIR="${1:-runs/harbor}"
require_model
AAB="$EXT_DIR/agentic-abstention"
TB2="$EXT_DIR/tb2-export"
DATASET="$EXT_DIR/aa-immediate"
clone_pinned "$AGENTIC_ABSTENTION_REPO" "$AGENTIC_ABSTENTION_COMMIT" "$AAB"
if [[ ! -d "$TB2" ]]; then
  "${UV_RUN[@]}" harbor download terminal-bench@2.0 --export -o "$TB2"
fi
if [[ ! -f "$DATASET/manifest.jsonl" ]]; then
  env -u VIRTUAL_ENV uv run --no-project --with toml python \
    "$AAB/terminal/scripts/materialize_harbor_instruction_level_abstention_dataset.py" \
    --input "$AAB/terminal/data/immediate/immediate_rewrites_267.jsonl" \
    --cache-root "$TB2" --output "$DATASET" --allow-partial --force
fi
JOB="agentic_abstention_immediate__${MODEL//\//_}__$(date +%Y%m%d_%H%M%S)"
"${UV_RUN[@]}" harbor run -p "$DATASET" -e docker -n "${N_CONCURRENT:-8}" -o "$JOBS_DIR" --job-name "$JOB" -y \
  -a honest_agents.eval.harbor_agent:HonestPolicyAgent -m "$MODEL" \
  --ak "api_base=$BASE_URL" --ak "max_turns=${MAX_TURNS:-10}" --ak enable_thinking=false
"${UV_RUN[@]}" python -m honest_agents.eval.external.agentic_abstention \
  --job-dir "$JOBS_DIR/$JOB" --manifest "$DATASET/manifest.jsonl"
"${UV_RUN[@]}" python -m honest_agents.eval.harbor_results "$JOBS_DIR/$JOB"
