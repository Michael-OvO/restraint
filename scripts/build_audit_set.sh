#!/usr/bin/env bash
# Build the small audited dataset for hand review (SPEC_V3 section 4.9): 20-30 source tasks, a few operator families
# plus their near-miss controls, one markdown audit sheet per variant (instruction diff, environment diff, perturbed
# object, labels, the tests and the oracle command, validation result, a checklist, a verdict line). The default families
# are the TRAINING families; the test-only families (conflicting_constraint, no_network) are zero-shot and are audited
# with FAMILIES="conflicting_constraint" when wanted.
#
#   scripts/build_audit_set.sh [OUT_DIR=data/audit] [extra audit args...]
#   scripts/build_audit_set.sh --tally data/audit          # count the verdicts after reviewing the sheets
#
# Env:
#   N_SOURCES=24          source tasks (20-30)
#   FAMILIES="remove_input revoke_permission high_risk"   the operator families to audit (the training families)
#   SHOW_TESTS=1          tests and oracle command on every sheet (default); SHOW_TESTS=0 hides them
#   SOURCES="endless-terminals"
#   LIMIT=300             seed tasks sampled before selection
#   BENCH_DIR=benchmarks  eval benchmarks to deduplicate against: REQUIRED (scripts/fetch_benchmarks.sh); I_KNOW_NO_BENCHMARK=1
#                         skips it (tests only)
#   SPLIT_SEEDS_FILE=     split_seeds.json of the dataset build, passed to tasks.audit as --split-seeds
#   TEST_IDS_FILE=        committed test_ids.txt, passed as --test-ids-file: the audit must not pick test sources, because
#                         the humans who read the sheets then tune operators (needs tasks.audit to accept both flags)
#   VALIDATE=auto         1 = oracle-validate the audit variants first (needs Docker), 0 = skip, auto = if `docker info` works
#   CONCURRENCY=4         concurrent sandboxes during validation
#   TASK_DIR_ROOT=        path of <OUT_DIR>/tasks on the training node, if different
#   ALLOW_LEGACY=1        keep seeds that are not graded by the hermetic verifier
# The result, <OUT_DIR>/train.jsonl, is a dataset in the normal row format and can be the bring-up dataset.
source "$(dirname "$0")/_common.sh"

if [[ "${1:-}" == --tally ]]; then
  exec "${UV_RUN[@]}" python -m honest_agents.tasks.audit --tally "${2:?usage: build_audit_set.sh --tally AUDIT_DIR}"
fi

OUT="${1:-data/audit}"; shift || true
N_SOURCES="${N_SOURCES:-24}"
FAMILIES="${FAMILIES:-remove_input revoke_permission high_risk}"
SHOW_TESTS="${SHOW_TESTS:-1}"
SOURCES="${SOURCES:-endless-terminals}"
LIMIT="${LIMIT-300}"
BENCH_DIR="${BENCH_DIR:-benchmarks}"
VALIDATE="${VALIDATE:-auto}"
CONCURRENCY="${CONCURRENCY:-4}"

ARGS=(--out "$OUT" --sources $SOURCES --families $FAMILIES --n-sources "$N_SOURCES" --seed 0)
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")
[[ "$SHOW_TESTS" == 0 ]] && ARGS+=(--no-show-tests)
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
[[ -n "${TEST_IDS_FILE:-}" ]] && ARGS+=(--test-ids-file "$TEST_IDS_FILE")
[[ -n "${TASK_DIR_ROOT:-}" ]] && ARGS+=(--task-dir-root "$TASK_DIR_ROOT")
[[ -n "${ALLOW_LEGACY:-}" ]] && ARGS+=(--allow-legacy-verifier)

if [[ "$VALIDATE" == auto ]]; then
  if docker info >/dev/null 2>&1; then VALIDATE=1; else VALIDATE=0; fi
fi

"${UV_RUN[@]}" python -m honest_agents.tasks.audit "${ARGS[@]}" "$@"
if [[ "$VALIDATE" == 1 ]]; then
  "${UV_RUN[@]}" python -m honest_agents.tasks.validate --tasks-dir "$OUT/tasks" --out "$OUT/validation.jsonl" \
    --concurrency "$CONCURRENCY"
  # Rebuild the sheets with the validation results shown (variants are regenerated identically, so hashes match).
  "${UV_RUN[@]}" python -m honest_agents.tasks.audit "${ARGS[@]}" --no-download --validation-jsonl "$OUT/validation.jsonl" "$@"
else
  echo "WARNING: sheets written WITHOUT validation results (no Docker). Validate on the Linux node and rerun." >&2
fi
echo "audit set: $OUT/INDEX.md (sheets in $OUT/sheets/, dataset $OUT/train.jsonl). After review: scripts/build_audit_set.sh --tally $OUT"
