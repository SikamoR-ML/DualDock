#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[check] repo: $(pwd)"

BAD=0

echo "[check] searching for absolute machine paths in tracked files..."
if git grep -nE '(/home/[^/]+|/Users/|\\wsl\.localhost|C:/Users/)' -- . ':(exclude)scripts/check_portability.sh' ; then
  BAD=1
else
  echo "[ok] no machine-specific absolute paths found"
fi

echo "[check] ensure main runner exists..."
test -f dualdock/run_experiment.py && echo "[ok] dualdock/run_experiment.py" || BAD=1

echo "[check] ensure wrapper entrypoints exist..."
test -f integrations/reinvent4/external_process/boltz_single_wrapper.py && echo "[ok] integrations/reinvent4/external_process/boltz_single_wrapper.py" || BAD=1
test -f integrations/reinvent4/external_process/boltz_dual_wrapper.py && echo "[ok] integrations/reinvent4/external_process/boltz_dual_wrapper.py" || BAD=1

echo "[check] ensure example configs exist..."
test -f configs/experiments/single_b1_3080.toml && echo "[ok] single_b1_3080.toml" || BAD=1
test -f configs/experiments/dual_b1_vs_b2_3080.toml && echo "[ok] dual_b1_vs_b2_3080.toml" || BAD=1
test -f configs/experiments/single_b1_a6000.toml && echo "[ok] single_b1_a6000.toml" || BAD=1
test -f configs/experiments/dual_b1_vs_b2_a6000.toml && echo "[ok] dual_b1_vs_b2_a6000.toml" || BAD=1
test -f configs/hardware/RTX_3080.toml && echo "[ok] RTX_3080.toml" || BAD=1
test -f configs/hardware/A6000.toml && echo "[ok] A6000.toml" || BAD=1

if [[ "$BAD" -ne 0 ]]; then
  echo "[fail] portability checks failed" >&2
  exit 1
fi

echo "[ok] portability checks passed"
