#!/usr/bin/env bash
# Terminal-Bench capability + fabrication eval through Harbor (Docker), with our policy harness.
#   MODEL=policy scripts/eval_tb.sh {tblite|tb2|NAME@VERSION|PATH_TO_TASKS} [JOBS_DIR=runs/harbor] [harbor args...]
# NAME@VERSION is any Harbor registry dataset, e.g. swebench-verified@1.0 for SWE-bench fabrication.
# Env: BASE_URL, N_CONCURRENT (8), MAX_TURNS (30), AGENT (honest | terminus-2; terminus-2 = stock harness baseline);
# the honest agent's external budget (DEF-71): MAX_COMPLETION_TOKENS, TURN_MAX_TOKENS, REPORT_RESERVE_TOKENS (unset: the
# training budget, as it19a ran).
#      REPORT_FILE=1 appends the report-file instruction to every task and collects the file (docs/evaluation.md section 7), so a stock
#      harness also yields honesty metrics; without it a stock harness has no report channel and only task success is
#      meaningful. Results are reported per harness. For our own tasks under a stock harness use eval_harness.sh.
# Output: JOBS_DIR/<job>/ (Harbor) and JOBS_DIR/<job>/honest_eval/{episodes.jsonl,metrics.json,summary.md}.
source "$(dirname "$0")/eval_common.sh"
TARGET="${1:?usage: eval_tb.sh tblite|tb2|NAME@VERSION|PATH [JOBS_DIR] [harbor args...]}"; shift
JOBS_DIR="${1:-runs/harbor}"; shift || true
require_model
case "$TARGET" in
  tblite) SRC=(-d openthoughts-tblite@2.0); TAG=tblite ;;
  # tb2 resolves through the committed registry (data/benchmarks/registry_tb2.json: every task at one commit), not the
  # hosted one, which can move: the final benchmark (data/benchmarks/final_benchmark.json) is frozen at that commit.
  tb2)    SRC=(-d terminal-bench@2.0 --registry-path "$REPO_ROOT/data/benchmarks/registry_tb2.json"); TAG=tb2 ;;
  *@*)    SRC=(-d "$TARGET"); TAG="${TARGET%%@*}" ;;
  *)      SRC=(-p "$TARGET"); TAG="$(basename "$TARGET")" ;;
esac
TAG="${TAG//\//_}"
JOB="${TAG}__${MODEL//\//_}__${AGENT:-honest}__$(date +%Y%m%d_%H%M%S)"

if [[ "${AGENT:-honest}" == terminus-2 ]]; then
  AGENT_ARGS=(-a terminus-2 -m "hosted_vllm/$MODEL" --ak "api_base=$BASE_URL" --ak "max_turns=${MAX_TURNS:-30}")
else
  AGENT_ARGS=(-a honest_agents.eval.harbor_agent:HonestPolicyAgent -m "$MODEL"
              --ak "api_base=$BASE_URL" --ak "max_turns=${MAX_TURNS:-30}" --ak enable_thinking=false)
  # DEF-71: the external budget (the same for every arm). Unset keeps the training budget of it19a's runs.
  if [[ -n "${MAX_COMPLETION_TOKENS:-}" ]]; then AGENT_ARGS+=(--ak "max_completion_tokens=$MAX_COMPLETION_TOKENS"); fi
  if [[ -n "${TURN_MAX_TOKENS:-}" ]]; then AGENT_ARGS+=(--ak "max_tokens_per_turn=$TURN_MAX_TOKENS"); fi
  if [[ -n "${REPORT_RESERVE_TOKENS:-}" ]]; then AGENT_ARGS+=(--ak "report_reserve_tokens=$REPORT_RESERVE_TOKENS"); fi
fi
REPORT_ARGS=()
if [[ "${REPORT_FILE:-0}" == 1 ]]; then
  mkdir -p "$JOBS_DIR"
  "${UV_RUN[@]}" python -m honest_agents.eval.report_file instructions --out "$JOBS_DIR/_report_file_instructions.md"
  REPORT_ARGS=(--extra-instruction-path "$JOBS_DIR/_report_file_instructions.md"
               --artifact "$("${UV_RUN[@]}" python -m honest_agents.eval.report_file path)")
fi
"${UV_RUN[@]}" harbor run "${SRC[@]}" -e honest_agents.eval.harbor_env:UmaskDockerEnvironment -n "${N_CONCURRENT:-8}" -o "$JOBS_DIR" --job-name "$JOB" -y \
  "${AGENT_ARGS[@]}" ${REPORT_ARGS[@]+"${REPORT_ARGS[@]}"} "$@"
"${UV_RUN[@]}" python -m honest_agents.eval.harbor_results "$JOBS_DIR/$JOB"
