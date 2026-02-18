#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG="runs/dualdock_$(date +%F_%H%M%S)/logs"
mkdir -p "$LOG"

reinvent -l "$LOG/rl.log" configs/reinvent4/staged_learning_dualdock.toml
