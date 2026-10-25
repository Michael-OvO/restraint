#!/usr/bin/env bash
# Agentic Abstention terminal subset (immediate instruction-level rewrites of Terminal-Bench 2.0) via Harbor + Docker.
#   MODEL=policy scripts/eval_agentic_abstention.sh [JOBS_DIR=runs/harbor]
# Env: BASE_URL, N_CONCURRENT (8), N_ATTEMPTS (1; `-k`), MAX_TURNS (10 = their max_interaction_rounds, for comparable
#      timely recall), TAG (suffix of the job name, e.g. base / trained: the transfer endpoint needs one run per arm).
# Steps: pinned checkouts (their repo; Terminal-Bench 2.0 at the commit scripts/fetch_benchmarks.sh decontaminates
# against) -> their materializer (stdlib + toml, in an ephemeral `uv run --no-project --with toml` env) -> every task
# must load with harbor's Task and the manifest must be 108 rows (19 abstain-warranted, 89 originals) -> harbor run with
# HonestPolicyAgent -> our timely-recall scorer, whose summary (`units`) feeds `external.transfer` (docs/evaluation.md).
# Needs Docker (every task pulls its prebuilt alexgshaw/<task> image) and the policy endpoint; the 21 delayed tasks are
# released only as prose specs (no task dirs), see docs/evaluation.md. Their abstain overlay is NOT needed: it only
# patches terminus-2; run a stock terminus-2 with `AA_MANIFEST=$DATASET/manifest.jsonl scripts/eval_harness.sh $DATASET`.
source "$(dirname "$0")/eval_common.sh"
JOBS_DIR="${1:-runs/harbor}"
require_model
TB2_REPO=https://github.com/laude-institute/terminal-bench-2
TB2_COMMIT=2fd12b8
AAB="$EXT_DIR/agentic-abstention"
TB2="$EXT_DIR/tb2-export"
DATASET="$EXT_DIR/aa-immediate"
clone_pinned "$AGENTIC_ABSTENTION_REPO" "$AGENTIC_ABSTENTION_COMMIT" "$AAB"
GIT_LFS_SKIP_SMUDGE=1 clone_pinned "$TB2_REPO" "$TB2_COMMIT" "$TB2"
if [[ ! -f "$DATASET/manifest.jsonl" ]]; then
  env -u VIRTUAL_ENV uv run --no-project --with toml python \
    "$AAB/terminal/scripts/materialize_harbor_instruction_level_abstention_dataset.py" \
    --input "$AAB/terminal/data/immediate/immediate_rewrites_267.jsonl" \
    --cache-root "$TB2" --output "$DATASET" --allow-partial --force
fi
"${UV_RUN[@]}" python -m honest_agents.eval.external.agentic_abstention --check-dataset "$DATASET"
JOB="agentic_abstention_immediate__${MODEL//\//_}${TAG:+__$TAG}__$(date +%Y%m%d_%H%M%S)"
"${UV_RUN[@]}" harbor run -p "$DATASET" -e honest_agents.eval.harbor_env:UmaskDockerEnvironment -n "${N_CONCURRENT:-8}" -k "${N_ATTEMPTS:-1}" -o "$JOBS_DIR" \
  --job-name "$JOB" -y -a honest_agents.eval.harbor_agent:HonestPolicyAgent -m "$MODEL" \
  --ak "api_base=$BASE_URL" --ak "max_turns=${MAX_TURNS:-10}" --ak enable_thinking=false
"${UV_RUN[@]}" python -m honest_agents.eval.external.agentic_abstention \
  --job-dir "$JOBS_DIR/$JOB" --manifest "$DATASET/manifest.jsonl"
"${UV_RUN[@]}" python -m honest_agents.eval.harbor_results "$JOBS_DIR/$JOB"
echo "summary with units for the transfer endpoint: $JOBS_DIR/$JOB/agentic_abstention_summary.json"
