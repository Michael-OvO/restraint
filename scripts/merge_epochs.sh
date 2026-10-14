#!/usr/bin/env bash
# Merge every epoch adapter of an SFT run into its own full model (CPU; ~5 min each).
#   scripts/merge_epochs.sh RUN_DIR OUT_PREFIX BASE [EXPECT]
# e.g. scripts/merge_epochs.sh outputs/sft/it14-sft-v2 outputs/it14/sft_v2 Qwen/Qwen3.5-4B
# Writes OUT_PREFIX_e<k>_merged for k = 1..EXPECT (default 3: `save_strategy: epoch`, `save_total_limit: 3`). The epoch of
# RUN_DIR/checkpoint-N comes from that checkpoint's trainer_state.json (honest_agents.sft.epochs), not from its position in
# the listing; the script refuses unless the epochs are exactly 1..EXPECT. DRY_RUN=1 prints the merges and runs none.
source "$(dirname "$0")/_common.sh"
RUN_DIR="${1:?usage: merge_epochs.sh RUN_DIR OUT_PREFIX BASE [EXPECT]}"
OUT_PREFIX="${2:?usage: merge_epochs.sh RUN_DIR OUT_PREFIX BASE [EXPECT]}"
BASE="${3:?usage: merge_epochs.sh RUN_DIR OUT_PREFIX BASE [EXPECT]}"
EXPECT="${4:-3}"
PY=(env -u VIRTUAL_ENV uv run --frozen --extra train python)
EPOCHS="$("${PY[@]}" -m honest_agents.sft.epochs list "$RUN_DIR" --expect "$EXPECT")"
while IFS=$'\t' read -r k ckpt; do
  out="${OUT_PREFIX}_e${k}_merged"
  echo "epoch $k: $ckpt -> $out"
  [[ "${DRY_RUN:-0}" == 1 ]] && continue
  "${PY[@]}" -m honest_agents.sft.merge --base "$BASE" --adapter "$ckpt" --out "$out"
done <<<"$EPOCHS"
