#!/usr/bin/env bash
# Analyse a report-only pilot and print the go / no-go line (docs/design.md section 10, docs/roadmap.md Stage 2). No GPU, no Docker.
#   scripts/pilot_report.sh [OUTPUT_ROOT=outputs/pilot] [CONFIG=configs/pilot.yaml] [extra analyze args...]
# Writes <OUTPUT_ROOT>/analysis.json and <OUTPUT_ROOT>/curves.csv. Exit status: 0 GO, 1 NO-GO, 2 INCONCLUSIVE.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ROOT="${1:-outputs/pilot}"; shift || true
CONFIG="${1:-configs/pilot.yaml}"; shift || true
[[ -d "$ROOT" ]] || { echo "no pilot output directory: $ROOT" >&2; exit 2; }
exec env -u VIRTUAL_ENV uv run --frozen --extra train python -m honest_agents.pilot.analyze "$ROOT" --config "$CONFIG" "$@"
