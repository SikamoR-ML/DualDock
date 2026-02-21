#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/experiments/dual_b1_vs_b2_3080.toml}"
RUN_NAME="${2:-}"

PY_BIN="${PYTHON:-python3}"
CMD=("$PY_BIN" -m dualdock.run_experiment --config "$CONFIG_PATH")
if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run-name "$RUN_NAME")
fi

"${CMD[@]}"
