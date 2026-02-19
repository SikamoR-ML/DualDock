#!/usr/bin/env bash
set -euo pipefail

# ---- Paths ----
DUALDOCK_DIR="/home/sikamor/projects/DualDock"
WRAPPER="/home/sikamor/projects/REINVENT4/scripts/boltz2_predict.py"

TARGET_DIR="${DUALDOCK_DIR}/assets/targets/adrenoreceptors"
POS_SMI="${DUALDOCK_DIR}/bench/clean/pos.smi"
NEG_SMI="${DUALDOCK_DIR}/bench/clean/neg.smi"
OUT_DIR="${DUALDOCK_DIR}/bench/clean"
LOG_DIR="${DUALDOCK_DIR}/logs/clean_bench"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/run_${TS}.log"

# ---- Compute params (override via env before запуском, если надо) ----
export BOLTZ_TARGET_DIR="${TARGET_DIR}"
export BOLTZ_TIMEOUT="${BOLTZ_TIMEOUT:-900}"
export BOLTZ_STEPS="${BOLTZ_STEPS:-10}"
export BOLTZ_AFF_STEPS="${BOLTZ_AFF_STEPS:-10}"
export BOLTZ_SCORE_MODE="${BOLTZ_SCORE_MODE:-pred}"

# Wrapper logging
export BOLTZ_DEBUG="${BOLTZ_DEBUG:-1}"
export BOLTZ_BOLTZ_STDERR="${BOLTZ_BOLTZ_STDERR:-1}"
export BOLTZ_KEEP_TMP="${BOLTZ_KEEP_TMP:-0}"
export BOLTZ_LOG_DIR="${BOLTZ_LOG_DIR:-${LOG_DIR}}"

TARGETS=("b1_full" "b1_nomsa" "b1_notemplate")

echo "[bench] start ${TS}" | tee -a "${MAIN_LOG}"
echo "[bench] wrapper=${WRAPPER}" | tee -a "${MAIN_LOG}"
echo "[bench] target_dir=${TARGET_DIR}" | tee -a "${MAIN_LOG}"
echo "[bench] score_mode=${BOLTZ_SCORE_MODE} timeout=${BOLTZ_TIMEOUT} steps=${BOLTZ_STEPS} aff_steps=${BOLTZ_AFF_STEPS}" | tee -a "${MAIN_LOG}"

# ---- Guards ----
if [[ ! -f "${WRAPPER}" ]]; then
  echo "[bench] ERROR: wrapper not found: ${WRAPPER}" | tee -a "${MAIN_LOG}"
  exit 2
fi
if [[ ! -f "${POS_SMI}" ]]; then
  echo "[bench] ERROR: pos.smi not found: ${POS_SMI}" | tee -a "${MAIN_LOG}"
  exit 2
fi
if [[ ! -f "${NEG_SMI}" ]]; then
  echo "[bench] ERROR: neg.smi not found: ${NEG_SMI}" | tee -a "${MAIN_LOG}"
  exit 2
fi

for t in "${TARGETS[@]}"; do
  if [[ ! -f "${TARGET_DIR}/${t}.yaml" && ! -f "${TARGET_DIR}/${t}.yml" ]]; then
    echo "[bench] ERROR: target template missing for ${t} in ${TARGET_DIR} (need ${t}.yaml or ${t}.yml)" | tee -a "${MAIN_LOG}"
    exit 2
  fi
done

run_set () {
  local label="$1"      # pos or neg
  local in_file="$2"    # pos.smi or neg.smi
  local tgt="$3"        # b1_full etc
  local out_jsonl="${OUT_DIR}/${label}_scores_${tgt}.jsonl"

  echo "[bench] ---- target=${tgt} set=${label} ----" | tee -a "${MAIN_LOG}"
  echo "[bench] input=${in_file}" | tee -a "${MAIN_LOG}"
  echo "[bench] output=${out_jsonl}" | tee -a "${MAIN_LOG}"

  rm -f "${out_jsonl}"

  local n_total
  n_total="$(wc -l < "${in_file}" | tr -d ' ')"
  local i=0

  # Read first column as SMILES (tab-separated), ignore rest
  while IFS=$'\t' read -r smi rest; do
    [[ -z "${smi}" ]] && continue
    i=$((i+1))

    # Progress to main log every 5 molecules
    if (( i % 5 == 0 )); then
      echo "[bench] ${tgt} ${label} progress ${i}/${n_total}" | tee -a "${MAIN_LOG}"
    fi

    # One molecule per call (stable, no big timeouts)
    printf "%s\n" "${smi}" | python3 "${WRAPPER}" >> "${out_jsonl}"

  done < "${in_file}"

  echo "[bench] DONE target=${tgt} set=${label} wrote $(wc -l < "${out_jsonl}" | tr -d ' ') lines" | tee -a "${MAIN_LOG}"
}

for tgt in "${TARGETS[@]}"; do
  export BOLTZ_TARGET="${tgt}"

  run_set "pos" "${POS_SMI}" "${tgt}"
  run_set "neg" "${NEG_SMI}" "${tgt}"
done

echo "[bench] finished $(date +%Y%m%d_%H%M%S)" | tee -a "${MAIN_LOG}"
echo "[bench] main log: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
