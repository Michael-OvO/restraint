#!/usr/bin/env bash
# Terminal-Bench capability + fabrication eval through Harbor (Docker), with our policy harness.
#   MODEL=policy scripts/eval_tb.sh {tblite|tb2|NAME@VERSION|PATH_TO_TASKS} [JOBS_DIR=runs/harbor] [harbor args...]
# NAME@VERSION is any Harbor registry dataset, e.g. swebench-verified@1.0 for SWE-bench fabrication.
# Env: BASE_URL, N_CONCURRENT (8), MAX_TURNS (30), AGENT (honest | terminus-2; terminus-2 = stock harness baseline).
#      REPORT_FILE=1 appends the report-file instruction to every task and collects the file (SPEC_V3 9.4), so a stock
#      harness also yields honesty metrics; without it a stock harness has no report channel and only task success is
#      meaningful. Results are reported per harness. For our own tasks under a stock harness use eval_harness.sh.
# Output: JOBS_DIR/<job>/ (Harbor) and JOBS_DIR/<job>/honest_eval/{episodes.jsonl,metrics.json,summary.md}.
source "$(dirname "$0")/eval_common.sh"
TARGET="${1:?usage: eval_tb.sh tblite|tb2|NAME@VERSION|PATH [JOBS_DIR] [harbor args...]}"; shift
JOBS_DIR="${1:-runs/harbor}"; shift || true
require_model
case "$TARGET" in
  tblite) SRC=(-d openthoughts-tblite@2.0); TAG=tblite ;;
  tb2)    SRC=(-d terminal-bench@2.0); TAG=tb2 ;;
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
fi
REPORT_ARGS=()
if [[ "${REPORT_FILE:-0}" == 1 ]]; then
  mkdir -p "$JOBS_DIR"
  "${UV_RUN[@]}" python -m honest_agents.eval.report_file instructions --out "$JOBS_DIR/_report_file_instructions.md"
  REPORT_ARGS=(--extra-instruction-path "$JOBS_DIR/_report_file_instructions.md"
               --artifact "$("${UV_RUN[@]}" python -m honest_agents.eval.report_file path)")
fi
"${UV_RUN[@]}" harbor run "${SRC[@]}" -e docker -n "${N_CONCURRENT:-8}" -o "$JOBS_DIR" --job-name "$JOB" -y \
  "${AGENT_ARGS[@]}" ${REPORT_ARGS[@]+"${REPORT_ARGS[@]}"} "$@"
"${UV_RUN[@]}" python -m honest_agents.eval.harbor_results "$JOBS_DIR/$JOB"
