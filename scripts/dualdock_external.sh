#!/usr/bin/env bash
set -euo pipefail

# repo root = one level above scripts/
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# IMPORTANT: do not print anything to stdout except JSON from Python
exec python "$ROOT/scripts/dualdock_predict.py" --config "$ROOT/configs/targets.toml"

