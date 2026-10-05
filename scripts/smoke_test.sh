#!/usr/bin/env bash
# End-to-end smoke test: 2 tasks of different length, 2 generations, 2 optimizer steps, one training GPU + one vLLM GPU,
# and a validation probe after step 1 so the evaluation path runs. The two training tasks must differ in episode length:
# that is what exposed the two-rank routing bug (TDC-1), and the run is single-process for that reason.
# Gate for every new model family (run with configs/qwen35_4b.yaml before Qwen3.5 experiments).
#   scripts/smoke_test.sh --train TASK_DIR [--train TASK_DIR2] [--val TASK_DIR] [--config CONFIG] [overrides...]
#   scripts/smoke_test.sh TASK_DIR            # one training task, no validation probe
# Every TASK_DIR is one Harbor task with [metadata.honest] verifier = "pytest" (instruction.md, task.toml, environment/,
# tests/). The validation task must not be one of the training tasks. Env: SLOT (0), VLLM_GPUS / TRAIN_GPUS.
source "$(dirname "$0")/_common.sh"
TRAIN_DIRS=(); VAL_DIR=""; CONFIG_FOR_MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) TRAIN_DIRS+=("$(cd "$2" && pwd)"); shift 2 ;;
    --val) VAL_DIR="$(cd "$2" && pwd)"; shift 2 ;;
    --config) CONFIG_FOR_MODEL="$2"; shift 2 ;;
    *.yaml) CONFIG_FOR_MODEL="$1"; shift ;;
    *) if is_override "$1" || [[ "$1" == -* ]]; then break; fi
       TRAIN_DIRS+=("$(cd "$1" && pwd)"); shift ;;
  esac
done
(( ${#TRAIN_DIRS[@]} >= 1 )) || { sed -n 2,12p "$0" >&2; exit 2; }
slot_env
MODEL_OVERRIDES=()
if [[ -n "$CONFIG_FOR_MODEL" && "$CONFIG_FOR_MODEL" != configs/smoke.yaml ]]; then
  # Use the model of CONFIG with smoke-sized everything else.
  read -r MODEL FAMILY < <(model_from_config "$CONFIG_FOR_MODEL")
  MODEL_OVERRIDES=("model.name=$MODEL" "model.family=$FAMILY")
fi

OUT=outputs/smoke
mkdir -p data "$OUT"
"${UV_RUN[@]}" python - "$VAL_DIR" "${TRAIN_DIRS[@]}" <<'PY'
import json, sys
from pathlib import Path
from honest_agents.spec import HonestTaskSpec
val, train = sys.argv[1], sys.argv[2:]
def rows(dirs, split):
    out = []
    for d in dirs:
        spec = HonestTaskSpec.from_task_dir(d)
        name = Path(d).name
        out.append({"task_dir": d, "variant_id": f"smoke-{split}-{name}", "pair_id": f"smoke-{split}-{name}",
                    "feasible": spec.feasible, "is_control": spec.is_control, "kind": spec.kind, "split": split,
                    "source": name})
    return out
Path("data/smoke.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(train, "train")))
Path("data/smoke_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows([val] if val else [], "val")))
PY
if [[ -z "$VAL_DIR" ]]; then PROBE_OV=("data.eval_file=null"); else PROBE_OV=(); fi
"$REPO_ROOT/scripts/probe_node.sh" || echo "WARNING: node probe failed; continuing"

# serve the model of CONFIG (serve_vllm.sh reads model, family and the window from it)
trap stop_vllm EXIT INT TERM
SMOKE_OV=(${MODEL_OVERRIDES[@]+"${MODEL_OVERRIDES[@]}"} ${PROBE_OV[@]+"${PROBE_OV[@]}"} "$@")
VLLM_GPU_MEM=0.85 start_vllm "$OUT/vllm.log" configs/smoke.yaml "${SMOKE_OV[@]}"

set +e
"$REPO_ROOT/scripts/train.sh" configs/smoke.yaml "${SMOKE_OV[@]}" 2>&1 | tee "$OUT/train.log"
rc=${PIPESTATUS[0]}
set -e

echo "=== smoke checks"
chk() { if grep -q "$1" "$2" 2>/dev/null; then echo "  [ok] $3"; else echo "  [??] $4"; fi; }
chk "rollout chat template:" "$OUT/train.log" "rollout chat template checked" "template log line not found"
chk "chat template is prefix-preserving" "$OUT/train.log" "tool-flow template is prefix-preserving" "prefix check not logged"
chk "vLLM probe passed" "$OUT/train.log" "vLLM startup probe passed (window, token ids, processed logprobs)" "startup probe not logged"
chk "honest/masked_share" "$OUT/train.log" "honesty metrics logged" "honesty metrics not seen"
[[ -n "$VAL_DIR" ]] && chk "eval_honest/task_success" "$OUT/train.log" "validation probe ran" "validation probe metrics not seen"
grep -qiE "Traceback" "$OUT/vllm.log" && echo "  [??] errors in vllm.log (check weight sync, --language-model-only)" || echo "  [ok] vllm.log has no traceback"
exit "$rc"
