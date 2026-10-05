#!/usr/bin/env bash
# AgentAbstain (263 act/abstain pairs, MCP tool environments, NOT terminal) against our vLLM endpoint.
#   MODEL=policy scripts/eval_agentabstain.sh [smoke|full]
#   JUDGE=1 JUDGE_API_KEY=... MODEL=policy scripts/eval_agentabstain.sh      # also the paid judge (see COST)
# Env: BASE_URL (policy), WORKERS (8), TASK_CONFIG (src/configs/tasks_preview.yaml = 263 pairs; `smoke` uses their
#      one-pair tasks_smoke.yaml), JUDGE (1 = run their GPT judge; default off), JUDGE_API_KEY (OpenAI key, needed with
#      JUDGE=1; an exported key alone never starts the judge), JUDGE_BASE_URL (optional OpenAI-compatible gateway).
# Runs in its own py3.11 venv under $HONEST_EXT_DIR/agentabstain (their requirements: openai-agents, fastmcp, ...).
# Scoring is ours (honest_agents.eval.external.agentabstain): the deterministic commit check always, their judge verdicts
# when JUDGE=1 has written eval.json. Without the judge the summary holds commit-only BOUNDS on operational pairs and
# nothing for informational ones; it is not the paired accuracy.
# COST (judge): gpt-5.4-2026-03-05 (eval/configs/default.yaml, temperature 0, max_tokens 1024): one paid call per
# rollout = 526 calls for the full set per model; the price per call is not measured. Run `smoke` first.
source "$(dirname "$0")/eval_common.sh"
MODE="${1:-full}"
require_model
AA="$EXT_DIR/agentabstain"
clone_pinned "$AGENTABSTAIN_REPO" "$AGENTABSTAIN_COMMIT" "$AA"
PY="$(ext_venv "$AA" 3.11)"
if [[ ! -f "$AA/.venv/.installed" ]]; then
  ext_pip "$PY" -r "$AA/requirements.txt" && touch "$AA/.venv/.installed"
fi
if [[ ! -d "$AA/data/tasks" ]]; then
  (cd "$AA" && "$PY" -c "from huggingface_hub import snapshot_download; \
    snapshot_download('antiquality/agentabstain', repo_type='dataset', local_dir='data')")
fi
if [[ "$MODE" == smoke ]]; then TASKS=src/configs/tasks_smoke.yaml; else TASKS="${TASK_CONFIG:-src/configs/tasks_preview.yaml}"; fi

LAUNCHER="$REPO_ROOT/src/honest_agents/eval/external/agentabstain_standalone.py"
(cd "$AA" && "$PY" "$LAUNCHER" infer --base-url "$BASE_URL" --model "$MODEL" --task-config "$TASKS" \
  --workers "${WORKERS:-8}")

RESULTS="$AA/results"
OUT="runs/agentabstain/${MODEL//\//_}.json"
mkdir -p runs/agentabstain
if [[ "${JUDGE:-0}" == 1 ]]; then
  : "${JUDGE_API_KEY:?JUDGE=1 needs JUDGE_API_KEY (the judge is a paid API)}"
  # The judge talks to OpenAI (or JUDGE_BASE_URL), never to the policy endpoint.
  (cd "$AA" && env -u OPENAI_BASE_URL OPENAI_API_KEY="$JUDGE_API_KEY" ${JUDGE_BASE_URL:+OPENAI_BASE_URL="$JUDGE_BASE_URL"} \
    "$PY" -m eval.runner --provider openaisdk --model "$MODEL")
  # their own leaderboard numbers, as a cross-check of ours
  (cd "$AA" && "$PY" "$LAUNCHER" summarize --model "$MODEL" --out "$REPO_ROOT/runs/agentabstain/${MODEL//\//_}.theirs.json")
fi
"${UV_RUN[@]}" python -m honest_agents.eval.external.agentabstain --tasks "$AA/data/tasks.jsonl" \
  --results "$RESULTS" --provider openaisdk --model "$MODEL" --out "$OUT"
echo "summary with units for the transfer endpoint: $OUT"
