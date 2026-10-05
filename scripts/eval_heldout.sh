#!/usr/bin/env bash
# Held-out honesty eval on our own task variants (needs Docker for the sandboxes and a served policy).
#   MODEL=policy scripts/eval_heldout.sh SPLIT OUT_DIR [extra run_eval args...]
# e.g. MODEL=policy scripts/eval_heldout.sh data/test.jsonl runs/eval/a0.25 --n-samples 4 --concurrency 32
# Env: BASE_URL (http://localhost:8000/v1). Serve the policy with `EVAL=1 scripts/serve_vllm.sh CONFIG
# --served-model-name policy` (adds the tool-call parser and non-thinking chat-template default).
# Baselines without training: add --neutral-prompt (or --system-prompt-file FILE) for the neutral-prompt arm of the
# {base, trained} x {honesty prompt, neutral prompt} table. The confidence-threshold sweep and the constant-policy
# reference rows (always claim, never claim, zero-work decline) are part of every metrics.json.
# Other flags: --train-config configs/X.yaml (identical caps and sampling to training), --bank-dir DIR (also write a
# trajectory bank for the report-only pilot), --quarantine-file / --exclude-quarantined, --protocol FILE.
# Re-score / compare afterwards without rollouts:
#   uv run ... python -m honest_agents.eval.run_eval --out OUT_DIR --metrics-only --alpha 0.5
#   uv run ... python -m honest_agents.eval.run_eval --compare runs/eval/a0 runs/eval/a0.25 runs/eval/outcome
#   uv run ... python -m honest_agents.eval.run_eval --noninferiority a0.25_s1,a0.25_s2 outcome_s1,outcome_s2   # run dirs; verified success; seeds comma separated
source "$(dirname "$0")/eval_common.sh"
SPLIT="${1:?usage: eval_heldout.sh SPLIT OUT_DIR [args...]}"; OUT="${2:?usage: eval_heldout.sh SPLIT OUT_DIR}"
shift 2
require_model
exec "${UV_RUN[@]}" python -m honest_agents.eval.run_eval \
  --split "$SPLIT" --out "$OUT" --base-url "$BASE_URL" --model "$MODEL" --no-thinking "$@"
