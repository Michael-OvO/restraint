#!/usr/bin/env bash
# Choose the SFT epoch on the select split, mechanically (docs/iteration-14-training-plan.md section 4).
#   scripts/select_epoch.sh BASE_EVAL OUT_DIR FALLBACK_EPOCH EVAL_DIR...
# e.g. scripts/select_epoch.sh runs/eval/it14_base_select outputs/it14 3 runs/eval/it14_sft_v2_select_e1 ...e2 ...e3
# Runs `eval.select_checkpoint --base BASE_EVAL EVAL_DIR... --require-split select --out OUT_DIR/selection.json`. Exit 0
# means an epoch passed the capability gate (non-inferior verified success, 95% paired interval, 5-point margin); exit 1
# means none did (selection.json says `selected: null`), which is NOT an error: the select-split gate is underpowered (~270 feasible episodes), so the
# pre-registered fallback FALLBACK_EPOCH (3) applies. Any other status (runs on different splits or settings) fails the
# script. Writes OUT_DIR/selected_epoch.txt (`k selected|fallback`) and the symlink OUT_DIR/sft_v2_selected_merged ->
# sft_v2_e<k>_merged (PREFIX=sft_v2 to change). The decision logic is honest_agents.sft.epochs `decide`.
source "$(dirname "$0")/_common.sh"
BASE="${1:?usage: select_epoch.sh BASE_EVAL OUT_DIR FALLBACK_EPOCH EVAL_DIR...}"
OUT_DIR="${2:?usage: select_epoch.sh BASE_EVAL OUT_DIR FALLBACK_EPOCH EVAL_DIR...}"
FALLBACK="${3:?usage: select_epoch.sh BASE_EVAL OUT_DIR FALLBACK_EPOCH EVAL_DIR...}"
shift 3
(( $# > 0 )) || { echo "select_epoch.sh: no epoch evaluation directory given" >&2; exit 2; }
PY=(env -u VIRTUAL_ENV uv run --frozen --extra train --extra eval python)
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/selection.json"  # a stale file must not pass for this run
rc=0
"${PY[@]}" -m honest_agents.eval.select_checkpoint --base "$BASE" "$@" --require-split select \
  --out "$OUT_DIR/selection.json" || rc=$?
"${PY[@]}" -m honest_agents.sft.epochs decide "$OUT_DIR/selection.json" "$OUT_DIR" \
  --fallback "$FALLBACK" --status "$rc" --prefix "${PREFIX:-sft_v2}"
cat "$OUT_DIR/selected_epoch.txt"
