#!/usr/bin/env bash
# The one way to run the tests (macOS or Linux, no Docker or GPU needed: sandboxes are host subprocesses).
#   scripts/test.sh                 # --fast: not docker/gpu/slow, in parallel (about 70 s on an 18-core Mac)
#   scripts/test.sh --all           # everything except docker/gpu, in parallel (about 3 min)
#   scripts/test.sh --changed       # only the test files related to files changed in git (see scripts/changed_tests.py)
#   scripts/test.sh [-k expr] tests/test_env.py::test_x    # anything else goes to pytest as is: serial, no marker filter
# Mode flags take extra pytest args after them (`scripts/test.sh --fast -k exploits`). Env: JOBS (xdist workers, default
# auto), TEST_LOCK_TIMEOUT (seconds before a lock is considered stale, default 1800).
# Parallel full runs share one lock, so two of them queue instead of piling their subprocess-heavy tests onto the same
# cores (the lock is a mkdir, which is atomic on macOS where there is no flock).
source "$(dirname "$0")/_common.sh"

LOCK="${TMPDIR:-/tmp}/honest-agents-tests.lock"
STALE="${TEST_LOCK_TIMEOUT:-1800}"
JOBS="${JOBS:-auto}"
mode=fast
case "${1:-}" in
  --fast|--all|--changed) mode="${1#--}"; shift ;;
  --help|-h) sed -n 2,11p "$0"; exit 0 ;;
  "") ;;
  *) mode=passthrough ;;
esac

take_lock() {
  local waited=0
  until mkdir "$LOCK" 2>/dev/null; do
    local holder; holder="$(cat "$LOCK/pid" 2>/dev/null || true)"
    local age=$(( $(date +%s) - $(stat -f %m "$LOCK" 2>/dev/null || stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
    if (( age > STALE )) || { [[ -n "$holder" ]] && ! kill -0 "$holder" 2>/dev/null; }; then
      echo "test.sh: removing a stale lock (pid ${holder:-?}, ${age}s old)" >&2
      rm -rf "$LOCK"; continue
    fi
    (( waited % 15 == 0 )) && echo "test.sh: another full run (pid ${holder:-?}) holds the lock; queued ${waited}s" >&2
    sleep 1; waited=$(( waited + 1 ))
  done
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
  trap 'exit 130' INT TERM
}

start=$SECONDS
args=()
case "$mode" in
  fast) take_lock; args=(-n "$JOBS" --dist loadfile -m "not docker and not gpu and not slow" "$@") ;;
  all) take_lock; args=(-n "$JOBS" -m "not docker and not gpu" "$@") ;;
  changed)
    files=()
    while IFS= read -r f; do [[ -n "$f" ]] && files+=("$f"); done < <("${UV_RUN[@]}" python scripts/changed_tests.py)
    if (( ${#files[@]} == 0 )); then echo "test.sh: no test is related to the changed files"; exit 0; fi
    if [[ "${files[0]}" == tests ]]; then  # shared test support or dependencies changed: everything may be affected
      echo "test.sh: shared support changed, running the fast suite"
      take_lock; args=(-n "$JOBS" --dist loadfile -m "not docker and not gpu and not slow" "$@")
    else
      echo "test.sh: ${#files[@]} related file(s): ${files[*]}"
      args=(-m "not docker and not gpu" "${files[@]}" "$@")  # slow tests included: the change is what is being checked
      (( ${#files[@]} > 2 )) && args=(-n "$JOBS" "${args[@]}")
    fi ;;
  passthrough) args=("$@") ;;
esac

rc=0
"${UV_RUN[@]}" pytest -q -p no:cacheprovider "${args[@]}" || rc=$?
echo "test.sh: $mode run took $(( SECONDS - start ))s (exit $rc)"
exit "$rc"
