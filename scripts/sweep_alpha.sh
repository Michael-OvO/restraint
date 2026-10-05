#!/usr/bin/env bash
# Arms of the main comparison, two at a time (one per run slot), each with its own vLLM server that is restarted per
# run so no run inherits another's pushed weights or a wedged server.
#   scripts/sweep_alpha.sh [BASE_CONFIG=configs/base.yaml] [extra overrides...]
# Env: ALPHAS ("0 0.1 0.25 0.4"), OUTCOME_ARMS ("outcome outcome_claimed outcome_reflect"; empty for none),
#      SEEDS ("42"), SLOTS ("0 1"). A failed run is recorded and the sweep continues.
#      PHASE0_SUMMARY=<phase0_summary.json>: skip (not fail) every honest arm whose alpha is unsafe for the Phase-0 pool,
#      i.e. (1 + alpha) / 2 above its pass rate minus 0.1, with the reason printed; ALLOW_UNSAFE_ALPHA=1 runs them anyway.
source "$(dirname "$0")/_common.sh"
BASE="${1:-configs/base.yaml}"; shift || true
ALPHAS="${ALPHAS:-0 0.1 0.25 0.4}"; SEEDS="${SEEDS:-42}"; SLOTS=(${SLOTS:-0 1})
OUTCOME_ARMS="${OUTCOME_ARMS-outcome outcome_claimed outcome_reflect}"
mkdir -p outputs/logs

# Job = "tag|override override ...". Seeds are the outer loop so a stopped sweep still holds complete seeds.
JOBS=()
for seed in $SEEDS; do
  for a in $ALPHAS; do JOBS+=("honest_a${a}_s${seed}|reward.mode=honest reward.config.alpha=$a seed=$seed"); done
  for m in $OUTCOME_ARMS; do JOBS+=("${m}_s${seed}|reward.mode=$m seed=$seed"); done
done

FAIL_FILE="$(mktemp)"; SKIP_FILE="$(mktemp)"
worker() {  # worker <slot> <index-in-SLOTS>
  local slot="$1" k="$2" i tag ov rc
  shift 2
  for (( i = k; i < ${#JOBS[@]}; i += ${#SLOTS[@]} )); do
    tag="${JOBS[$i]%%|*}"; ov="${JOBS[$i]#*|}"
    echo "=== [slot $slot] $tag"
    rc=0
    # shellcheck disable=SC2086
    check_alpha "$BASE" $ov "$@" || rc=$?
    if (( rc == 4 )); then  # honest_agents.train.EXIT_UNSAFE_ALPHA: the reason is on stderr; not a failure
      echo "=== [slot $slot] $tag SKIPPED: unsafe alpha for the Phase-0 pool (ALLOW_UNSAFE_ALPHA=1 to run it)"
      echo "$tag" >>"$SKIP_FILE"
      continue
    elif (( rc != 0 )); then
      echo "=== [slot $slot] $tag FAILED the alpha check rc=$rc"; echo "$tag" >>"$FAIL_FILE"; continue
    fi
    # shellcheck disable=SC2086
    SLOT="$slot" TAG="$tag" "$REPO_ROOT/scripts/run_slot.sh" "$BASE" $ov "$@" >/dev/null 2>&1 || rc=$?
    if (( rc == 0 )); then
      echo "=== [slot $slot] $tag done"
    elif (( rc == 3 )); then  # honest_agents.train.EXIT_STOPPED_BY_RULE: a collapse caught by a stop rule, not a crash
      echo "=== [slot $slot] $tag STOPPED BY A STOP RULE (outputs/*/$tag*/stop_rules.jsonl)"; echo "$tag(stop-rule)" >>"$FAIL_FILE"
    else
      echo "=== [slot $slot] $tag FAILED rc=$rc (outputs/logs/${tag}_slot${slot}_*.log)"; echo "$tag" >>"$FAIL_FILE"
    fi
  done
}
PIDS=()
for k in "${!SLOTS[@]}"; do worker "${SLOTS[$k]}" "$k" "$@" & PIDS+=($!); done
for p in "${PIDS[@]}"; do wait "$p" || true; done
if [[ -s "$SKIP_FILE" ]]; then echo "SKIPPED (unsafe alpha): $(tr '\n' ' ' <"$SKIP_FILE")"; fi
rm -f "$SKIP_FILE"
if [[ -s "$FAIL_FILE" ]]; then echo "FAILED runs: $(tr '\n' ' ' <"$FAIL_FILE")"; rm -f "$FAIL_FILE"; exit 1; fi
rm -f "$FAIL_FILE"
echo "sweep complete"
