#!/usr/bin/env bash
set -euo pipefail

# Load conda into this non-interactive shell
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found at $HOME/miniforge3/etc/profile.d/conda.sh"
  echo "Check your Miniforge install path."
  exit 1
fi

cfg="${1:-}"
stage="${2:-analyze}"
run_name="${3:-run_$(date +%F_%H-%M-%S)}"
out="results/${run_name}"

conda activate dualdock-mac

python -m src.pipeline.run --config "$cfg" --out "$out" --stage "$stage"
echo "[OK] $out"
