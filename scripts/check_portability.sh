#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[check] repo: $(pwd)"

# Fail if we find typical non-portable paths
BAD=0

echo "[check] searching for absolute/local machine paths..."
if grep -RInE '(/home/|~/|/Users/|miniforge3|conda/envs)' . ; then
  BAD=1
else
  echo "[ok] no obvious absolute paths found"
fi

echo "[check] ensure ExternalProcess entrypoint exists..."
test -x scripts/dualdock_external.sh && echo "[ok] scripts/dualdock_external.sh" || BAD=1

echo "[check] ensure priors exist..."
test -f priors/reinvent.prior && echo "[ok] priors/reinvent.prior" || BAD=1

echo "[check] ensure configs exist..."
test -f configs/reinvent4/staged_learning_dualdock.toml && echo "[ok] staged_learning_dualdock.toml" || BAD=1
test -f configs/reinvent4/stage2_scoring_dualdock.toml && echo "[ok] stage2_scoring_dualdock.toml" || BAD=1
test -f configs/targets.toml && echo "[ok] configs/targets.toml" || BAD=1

if [[ "$BAD" -ne 0 ]]; then
  echo "[fail] portability checks failed" >&2
  exit 1
fi

echo "[ok] portability checks passed"
