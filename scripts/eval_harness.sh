#!/usr/bin/env bash
# Honesty metrics under a THIRD-PARTY Harbor harness, through the report-file protocol (docs/evaluation.md section 7).
#   AGENT=terminus-2 MODEL=policy scripts/eval_harness.sh SPLIT [JOBS_DIR=runs/harbor] [harbor args...]
# SPLIT is our JSONL split (task_dir column) or a directory of Harbor task dirs. The agent has no submit_report /
# abstain tool: the task instruction gets harness.REPORT_FILE_INSTRUCTIONS appended (`--extra-instruction-path`) and the
# file it writes is downloaded as a trial artifact (`--artifact`), then scored by the same compute_reward as our own
# harness. Results are reported per harness, and the agent name is recorded on every episode.
# Env: BASE_URL (policy endpoint, http://localhost:8000/v1), AGENT (terminus-2 | mini-swe-agent | opencode | ...),
#      N_CONCURRENT (8), N_ATTEMPTS (1; `-k`), MAX_TURNS (30, terminus-2 only),
#      AA_MANIFEST (a materialised Agentic Abstention manifest.jsonl, with SPLIT = its dataset directory: also score the
#      job with the timely-recall scorer, turns counted from the harness's ATIF trajectory; eval_agentic_abstention.sh),
#      AGENT_ARGS (replaces the default `--ak` arguments; installed agents such as mini-swe-agent run INSIDE the
#      sandbox and read the endpoint from the environment, e.g. AGENT_ARGS="--ae OPENAI_BASE_URL=http://HOST:8000/v1").
# What this does not give: Harbor runs the task's own tests/test.sh, so there is no tamper check, no path-violation
# check, no evidence replay and no blocker verification (external blockers on feasible tasks count as false blame), and
# the command count comes from the harness's trajectory.json when it writes one (docs/evaluation.md).
source "$(dirname "$0")/eval_common.sh"
SPLIT="${1:?usage: eval_harness.sh SPLIT [JOBS_DIR] [harbor args...]}"; shift
JOBS_DIR="${1:-runs/harbor}"; shift || true
require_model
AGENT="${AGENT:-terminus-2}"
NAME="$(basename "${SPLIT%.jsonl}")"
TASKS="$JOBS_DIR/_tasks/$NAME"
INSTRUCTIONS="$JOBS_DIR/_report_file_instructions.md"
mkdir -p "$JOBS_DIR"
"${UV_RUN[@]}" python -m honest_agents.eval.report_file link-split "$SPLIT" "$TASKS"
"${UV_RUN[@]}" python -m honest_agents.eval.report_file instructions --out "$INSTRUCTIONS"
REPORT_PATH="$("${UV_RUN[@]}" python -m honest_agents.eval.report_file path)"
if [[ -n "${AGENT_ARGS:-}" ]]; then
  read -r -a AGENT_ARGV <<<"$AGENT_ARGS"
else
  AGENT_ARGV=(--ak "api_base=$BASE_URL")
  [[ "$AGENT" == terminus-2 ]] && AGENT_ARGV+=(--ak "max_turns=${MAX_TURNS:-30}")
fi
JOB="${NAME}__${MODEL//\//_}__${AGENT}__$(date +%Y%m%d_%H%M%S)"
"${UV_RUN[@]}" harbor run -p "$TASKS" -e honest_agents.eval.harbor_env:UmaskDockerEnvironment -n "${N_CONCURRENT:-8}" -k "${N_ATTEMPTS:-1}" -o "$JOBS_DIR" \
  --job-name "$JOB" -y -a "$AGENT" -m "hosted_vllm/$MODEL" "${AGENT_ARGV[@]}" \
  --extra-instruction-path "$INSTRUCTIONS" --artifact "$REPORT_PATH" "$@"
"${UV_RUN[@]}" python -m honest_agents.eval.harbor_results "$JOBS_DIR/$JOB"
if [[ -n "${AA_MANIFEST:-}" ]]; then
  "${UV_RUN[@]}" python -m honest_agents.eval.external.agentic_abstention --job-dir "$JOBS_DIR/$JOB" --manifest "$AA_MANIFEST"
fi
echo "per-harness metrics: $JOBS_DIR/$JOB/honest_eval/summary.md"
