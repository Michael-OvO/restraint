#!/usr/bin/env bash
# Build the honest-agents task dataset (sources -> dedup -> split -> variants -> validate -> mixture).
#
#   scripts/build_dataset.sh [OUT_DIR=data] [extra build args...]
#
# Env:
#   LIMIT=1500           seed tasks sampled per source (empty = all). Splits are a fixed hash of the source id, so changing
#                        LIMIT (or the pool) never moves a source between train, val and test.
#   SOURCES="endless-terminals"   first dataset is Endless Terminals only, converted to the hermetic pytest verifier
#   BENCH_DIR=benchmarks  eval benchmarks to deduplicate against: REQUIRED, the build fails without it. Run
#                         scripts/fetch_benchmarks.sh first (pinned commits). I_KNOW_NO_BENCHMARK=1 skips it (tests only).
#   TEST_BP=2200 VAL_BP=400   fixed split cutoffs in 1/10000 of the hash space (about 330 test and 60 val sources of 1,500)
#   TEST_SEEDS= VAL_SEEDS=    LEGACY: rank the current pool and hold out exactly this many (pool-dependent; dev only)
#   SPLIT_SEEDS_FILE=     split_seeds.json of an earlier build (seed group -> split), reused when building into a fresh OUT
#   TEST_IDS_FILE=        committed test_ids.txt (ids + sha256); default OUT/test_ids.txt. The build refuses any train,
#                         val or train_pool row for an id in it. Commit the file after the first build.
#   NEARDUP_THRESHOLD=0.1  near-duplicate sources (instruction 3-gram Jaccard) are split as one unit; 0 = off
#   QUARANTINE_INCLUDE_EVAL=  1 = honour quarantine entries found while evaluating (they change test.jsonl; default off)
#   NO_ENFORCE_CUE_BAND=  1 = only warn when a family's existence AUC leaves [0.4, 0.6] (default: the build fails)
#   HELDOUT=high_risk,revoke_permission   families seen only in the test split: high_risk + one environment mechanism
#   MIXTURE=0.55,0.20,0.25        plain feasible, near-miss control, infeasible shares of train and val
#   QUARANTINE_FILE=      quarantine list (leaky variants); listed variants are excluded and validation leaks appended
#   DIFFICULTY_JSON=      Phase-0 gate from `python -m honest_agents.tasks.difficulty`; filters the train split only
#   VALIDATE=auto         1 = oracle-validate every variant (needs Docker), 0 = skip, auto = if `docker info` works
#   CONCURRENCY=8         concurrent sandboxes during validation
#   TASK_DIR_ROOT=        path of <OUT_DIR>/tasks on the training node, if different
#   ALLOW_LEGACY=1        keep seeds that are not graded by the hermetic verifier (never for training)
#
# With validation the build runs five steps, all resumable: build, validate, rebuild (relabels remove_input variants the
# repair probe proved harmless into near-miss controls, quarantines leaks), validate again (the relabelled controls),
# final rebuild. Without Docker it builds without validation and says so: re-run on the Linux node before training.
source "$(dirname "$0")/_common.sh"

OUT="${1:-data}"; shift || true  # data/train.jsonl = configs/base.yaml data.train_file
LIMIT="${LIMIT-1500}"
SOURCES="${SOURCES:-endless-terminals}"
BENCH_DIR="${BENCH_DIR:-benchmarks}"
TEST_BP="${TEST_BP:-2200}"
VAL_BP="${VAL_BP:-400}"
HELDOUT="${HELDOUT:-high_risk,revoke_permission}"
MIXTURE="${MIXTURE:-0.55,0.20,0.25}"
VALIDATE="${VALIDATE:-auto}"
CONCURRENCY="${CONCURRENCY:-8}"

ARGS=(--out "$OUT" --sources $SOURCES --heldout-families "$HELDOUT" --mixture "$MIXTURE"
      --test-bp "$TEST_BP" --val-bp "$VAL_BP" --seed 0)
[[ -n "${TEST_SEEDS:-}" ]] && ARGS+=(--test-seeds "$TEST_SEEDS")
[[ -n "${VAL_SEEDS:-}" ]] && ARGS+=(--val-seeds "$VAL_SEEDS")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
if [[ -n "${I_KNOW_NO_BENCHMARK:-}" ]]; then
  echo "WARNING: I_KNOW_NO_BENCHMARK set: seeds are NOT deduplicated against the eval benchmarks." >&2
  ARGS+=(--i-know-no-benchmark)
elif [[ -d "$BENCH_DIR" ]]; then
  ARGS+=(--benchmark-dir "$BENCH_DIR")
else
  echo "ERROR: $BENCH_DIR not found. Decontamination is mandatory: run scripts/fetch_benchmarks.sh $BENCH_DIR first." >&2
  exit 2
fi
[[ -n "${SPLIT_SEEDS_FILE:-}" ]] && ARGS+=(--split-seeds "$SPLIT_SEEDS_FILE")
# The committed list wins: configs/test_ids.txt (copy OUT/test_ids.txt there after the FIRST build and commit it; data/ is
# git-ignored). The salt (--split-seed) and the cutoffs (TEST_BP, VAL_BP) are frozen after the first training run: raising
# TEST_BP later moves already-trained sources into test unless that file is supplied.
[[ -z "${TEST_IDS_FILE:-}" && -f configs/test_ids.txt ]] && TEST_IDS_FILE=configs/test_ids.txt
[[ -n "${TEST_IDS_FILE:-}" ]] && ARGS+=(--test-ids-file "$TEST_IDS_FILE")
[[ -n "${NEARDUP_THRESHOLD:-}" ]] && ARGS+=(--neardup-threshold "$NEARDUP_THRESHOLD")
[[ -n "${QUARANTINE_INCLUDE_EVAL:-}" ]] && ARGS+=(--quarantine-include-eval)
[[ -n "${TASK_DIR_ROOT:-}" ]] && ARGS+=(--task-dir-root "$TASK_DIR_ROOT")
[[ -n "${QUARANTINE_FILE:-}" ]] && ARGS+=(--quarantine-file "$QUARANTINE_FILE")
[[ -n "${DIFFICULTY_JSON:-}" ]] && ARGS+=(--difficulty-json "$DIFFICULTY_JSON")
[[ -n "${NO_ENFORCE_CUE_BAND:-}" ]] && ARGS+=(--no-enforce-cue-band)
[[ -n "${ALLOW_LEGACY:-}" ]] && ARGS+=(--allow-legacy-verifier)

if [[ "$VALIDATE" == auto ]]; then
  if docker info >/dev/null 2>&1; then VALIDATE=1; else VALIDATE=0; fi
fi

BUILD=("${UV_RUN[@]}" python -m honest_agents.tasks.build)
VALIDATE_CMD=("${UV_RUN[@]}" python -m honest_agents.tasks.validate --tasks-dir "$OUT/tasks" --out "$OUT/validation.jsonl"
              --concurrency "$CONCURRENCY")
if [[ "$VALIDATE" == 1 ]]; then
  "${BUILD[@]}" "${ARGS[@]}" "$@"                                                        # 1. generate variants
  "${VALIDATE_CMD[@]}"                                                                   # 2. oracle, null check, repair probe
  "${BUILD[@]}" "${ARGS[@]}" --no-download --validation-jsonl "$OUT/validation.jsonl" "$@"   # 3. relabel, quarantine leaks
  "${VALIDATE_CMD[@]}"                                                                   # 4. validate the relabelled controls
  "${BUILD[@]}" "${ARGS[@]}" --no-download --validation-jsonl "$OUT/validation.jsonl" "$@"   # 5. final rows
else
  echo "WARNING: building WITHOUT oracle validation (no Docker). Re-run on the Linux node before training." >&2
  "${BUILD[@]}" "${ARGS[@]}" "$@"
fi
echo "dataset: $OUT/{train,val,test}.jsonl  (manifest: $OUT/manifest.json)"
