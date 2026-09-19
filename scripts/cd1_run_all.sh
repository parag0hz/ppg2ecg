#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/cd1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
echo "[$(date -Is)] CD1 START repo=$(git rev-parse --short HEAD)"
"$PY" scripts/cd1_train.py > "$LOG/train.log" 2>&1; rc=$?
if [ $rc -eq 3 ]; then  # declared fallback: non-finite loss -> restart once at lr 1e-4
  echo "[$(date -Is)] non-finite at lr 1e-3, fallback to 1e-4"; rm -rf outputs/cd1_vitaldb_armD_seed42
  "$PY" scripts/cd1_train.py --lr 1e-4 > "$LOG/train_fallback.log" 2>&1; rc=$?
fi
echo "[$(date -Is)] train rc=$rc"
[ -f outputs/cd1_vitaldb_armD_seed42/TRAINING_DONE ] || { echo "ABORT: training incomplete"; exit 1; }
"$PY" scripts/cd1_evaluate.py > "$LOG/eval.log" 2>&1; echo "[$(date -Is)] eval rc=$?"
"$PY" scripts/rf1_verdict.py --arm D > "$LOG/verdict.log" 2>&1; echo "[$(date -Is)] CD1 ALL DONE rc=$?"
