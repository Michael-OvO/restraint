#!/usr/bin/env bash
# Supervised stage before GRPO: validate the config and build the examples on the CPU (dry run, no model), then train.
#   scripts/run_sft.sh [configs/sft_hindsight.yaml | configs/sft_bestofn.yaml] [key.path=value ...]
#   scripts/run_sft.sh configs/sft_bestofn.yaml kind=combo   # report SFT + best-of-N trajectories in one adapter
#   DRY_RUN=1 scripts/run_sft.sh ...       # only the dry run: examples, labels, tokenised sizes, trainer config
# Env: RESUME=1 to continue from the run's last checkpoint-*, CUDA_VISIBLE_DEVICES (one training GPU; no vLLM needed).
# The dry run writes <output_dir>/examples.jsonl: read a few rewritten reports before spending GPU time (gate: no paid
# training until labels are verified). `--spread BANK` (python -m honest_agents.sft.train_sft --spread BANK) compares
# the within-group reward spread of banks sampled before and after the stage.
source "$(dirname "$0")/_common.sh"
CONFIG="configs/sft_hindsight.yaml"
if [[ $# -gt 0 && "$1" == *.yaml ]]; then CONFIG="$1"; shift; fi
SFT=("${UV_RUN[@]}" python -m honest_agents.sft.train_sft --config "$CONFIG")

"${SFT[@]}" --dry-run "$@"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
FLAGS=(); [[ "${RESUME:-0}" == 1 ]] && FLAGS+=(--resume)
exec "${SFT[@]}" ${FLAGS[@]+"${FLAGS[@]}"} "$@"
