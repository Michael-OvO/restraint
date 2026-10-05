#!/usr/bin/env bash
# Phase 0 (SPEC_V3 5.7, section 8 stage 1): the real trainer and rollout path at learning rate 0 for 24 rounds, in one
# run slot. Writes the trajectory bank (outputs/phase0/<run>/bank) the report-only pilot trains on, then prints the
# Stage-1 summary and gates (report compliance, no-report share, fabrication among failed feasible tasks, mean
# importance-sampling ratio within 1% of 1) and exits 1 if one fails or was not measured.
#   scripts/phase0.sh [CONFIG=configs/phase0_zero_lr.yaml] [key.path=value ...]
# A zero learning rate cannot check weight sync (the adapter delta is zero): scripts/sync_check.sh does that.
# Not run here: base-model honesty under a stock harness (SPEC_V3 9.4.2), see scripts/eval_heldout.sh.
source "$(dirname "$0")/_common.sh"
CONFIG="configs/phase0_zero_lr.yaml"
if [[ $# -gt 0 && "$1" == *.yaml ]]; then CONFIG="$1"; shift; fi
slot_env
"$REPO_ROOT/scripts/probe_node.sh"
"${UV_RUN[@]}" python - <<'PY'
from honest_agents.verifier import bundle
bundle.ensure_bundle()   # raises with the build command if the vendored pytest bundle is missing
print("verifier bundle ok:", bundle.default_bundle_dir())
PY
# The summary directory comes from the config, so it follows overrides (seed, output_root, run_name).
OUT="$("${UV_RUN[@]}" python - "$CONFIG" "$@" <<'PY'
import sys
from honest_agents.train import load_run_config, output_dir
print(output_dir(load_run_config(sys.argv[1], sys.argv[2:])))
PY
)"
rc=0
"$REPO_ROOT/scripts/run_slot.sh" "$CONFIG" "$@" || rc=$?
echo "=== Phase 0 summary: $OUT"
mkdir -p "$OUT"
set +e
"${UV_RUN[@]}" python -m honest_agents.callbacks phase0 "$OUT" | tee "$OUT/phase0_summary.json"
(( ${PIPESTATUS[0]} == 0 )) || rc=1
exit "$rc"
