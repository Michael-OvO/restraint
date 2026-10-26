#!/usr/bin/env bash
# One vLLM server for one model, a list of commands against it, and the server stopped on every exit path.
#   scripts/serve_and_run.sh CONFIG [dotted.key=value ...] -- CMD [ARGS...] [-- CMD2 [ARGS...] ...]
# e.g. scripts/serve_and_run.sh configs/grpo_it13_2gpu.yaml model.name=outputs/it14/grpo10_merged \
#        -- scripts/eval_heldout.sh data/val.jsonl runs/eval/x --n-samples 4 -- scripts/eval_heldout.sh data/probe_v1.jsonl runs/eval/y
# The server is `EVAL=1 start_vllm outputs/logs/serve_<step>.log CONFIG <overrides> --served-model-name policy` on VLLM_GPUS /
# VLLM_PORT (SLOT, default 0: GPU 1, port 8000; a chain step sets them in its `env`). Each command group after a `--`
# runs in order with MODEL=policy and BASE_URL=http://127.0.0.1:$VLLM_PORT/v1 exported; the first non-zero status is
# the script's. The server (its own setsid session) is stopped on a normal end, a failing command, `set -e` and on
# SIGINT / SIGTERM (the node runner sends SIGINT, then SIGTERM, to the step's process group): the signal is forwarded to
# the running command (its own process group), which gets WAIT_CHILD_SEC (30) to tear down its sandboxes, then the
# server is stopped.
# Env: SERVE_TAG (log name; default the chain step name from HA_CHAIN_STEP, else port<VLLM_PORT>), VLLM_WAIT_TIMEOUT (3600),
#      DRY_RUN=1 prints the plan (server command, ports, command groups) and exits 0 without starting anything.
source "$(dirname "$0")/_common.sh"
CONFIG="${1:?usage: serve_and_run.sh CONFIG [dotted.key=value ...] -- CMD [ARGS...] [-- CMD2 ...]}"; shift
SERVE_ARGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do SERVE_ARGS+=("$1"); shift; done
[[ $# -gt 0 ]] || { echo "serve_and_run.sh: no command after '--'" >&2; exit 2; }
# command groups: GROUP_WORDS holds all the words, GROUP_LEN the length of each group
GROUP_WORDS=(); GROUP_LEN=()
while [[ $# -gt 0 ]]; do
  shift  # the '--'
  n=0
  while [[ $# -gt 0 && "$1" != "--" ]]; do GROUP_WORDS+=("$1"); shift; n=$((n + 1)); done
  (( n > 0 )) || { echo "serve_and_run.sh: empty command after '--'" >&2; exit 2; }
  GROUP_LEN+=("$n")
done
slot_env
STEP="${HA_CHAIN_STEP:-}"; TAG="${SERVE_TAG:-${STEP%%.*}}"; TAG="${TAG:-port${VLLM_PORT}}"
LOG="outputs/logs/serve_${TAG}.log"
export MODEL=policy BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "serve: EVAL=1 GPUs=$VLLM_GPUS port=$VLLM_PORT log=$LOG $CONFIG ${SERVE_ARGS[*]-} --served-model-name policy"
  echo "env: MODEL=$MODEL BASE_URL=$BASE_URL"
  pos=0; i=0
  for n in "${GROUP_LEN[@]}"; do
    i=$((i + 1)); echo "command $i: ${GROUP_WORDS[*]:pos:n}"; pos=$((pos + n))
  done
  exit 0
fi

CHILD_PID=""; BG_BEFORE=""
cleanup() {  # idempotent: runs on EXIT, which every path (normal, failure, INT, TERM via `exit`) reaches
  trap - EXIT
  trap ':' INT TERM  # a second signal while we tear down must not abandon the server
  # A signal after the command's `&` but before `CHILD_PID=$!` runs this with CHILD_PID still empty (DEF-61); `$!`
  # already holds the command's pid then, and differs from the one recorded just before the fork.
  if [[ -z "$CHILD_PID" && -n "$BG_BEFORE" && "$!" != "$BG_BEFORE" ]]; then CHILD_PID=$!; fi
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
    kill -INT -- "-$CHILD_PID" 2>/dev/null || true
    local t=0
    while kill -0 "$CHILD_PID" 2>/dev/null && (( t++ < ${WAIT_CHILD_SEC:-30} )); do sleep 1; done
    kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  stop_vllm
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

EVAL=1 start_vllm "$LOG" "$CONFIG" ${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"} --served-model-name policy
wait_vllm
echo "vLLM ready: $MODEL at $BASE_URL"

pos=0; i=0
for n in "${GROUP_LEN[@]}"; do
  i=$((i + 1)); cmd=("${GROUP_WORDS[@]:pos:n}"); pos=$((pos + n))
  echo "serve_and_run: command $i: ${cmd[*]}"
  # Job control puts the command in its own process group, so a forwarded signal reaches it, and keeps SIGINT
  # deliverable to it (a plain `&` child of a non-interactive shell ignores SIGINT). Only after the server started:
  # `setsid` forks instead of exec'ing when its caller is a group leader.
  BG_BEFORE=$!  # set: start_vllm ran in the background
  set -m
  "${cmd[@]}" </dev/null &
  CHILD_PID=$!
  set +m
  rc=0; wait "$CHILD_PID" || rc=$?
  CHILD_PID=""; BG_BEFORE=""
  if (( rc != 0 )); then echo "serve_and_run: command $i exited with status $rc" >&2; exit "$rc"; fi
done
