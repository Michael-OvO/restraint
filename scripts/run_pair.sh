#!/usr/bin/env bash
# Two concurrent runs on one 4-GPU node: CONFIG_A in slot 0 (GPUs 0+1, port 8000) and CONFIG_B in slot 1 (GPUs 2+3,
# port 8001). Same overrides (seed, ...) go to both, so honest vs outcome is a same-seed, same-compute comparison.
#   scripts/run_pair.sh configs/honest_alpha0.25.yaml configs/outcome.yaml [key.path=value ...]
source "$(dirname "$0")/_common.sh"
A="${1:?usage: run_pair.sh CONFIG_A CONFIG_B [overrides...]}"; B="${2:?usage: run_pair.sh CONFIG_A CONFIG_B [overrides...]}"; shift 2
SLOT=0 "$REPO_ROOT/scripts/run_slot.sh" "$A" "$@" & PA=$!
SLOT=1 "$REPO_ROOT/scripts/run_slot.sh" "$B" "$@" & PB=$!
rc=0
wait "$PA" || { echo "slot 0 ($A) FAILED" >&2; rc=1; }
wait "$PB" || { echo "slot 1 ($B) FAILED" >&2; rc=1; }
exit "$rc"
