#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/rf1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
echo "[$(date -Is)] RF1 START repo=$(git rev-parse --short HEAD)"
"$PY" scripts/rf1_make_pairs.py > "$LOG/pairs.log" 2>&1; echo "[$(date -Is)] pairs rc=$?"
"$PY" scripts/rf1_train.py > "$LOG/train.log" 2>&1; echo "[$(date -Is)] train rc=$?"
[ -f outputs/rf1_vitaldb_armR_seed42/TRAINING_DONE ] || { echo "ABORT: training incomplete"; exit 1; }
"$PY" scripts/rf1_evaluate.py > "$LOG/eval.log" 2>&1; echo "[$(date -Is)] eval rc=$?"
"$PY" scripts/rf1_verdict.py > "$LOG/verdict.log" 2>&1; echo "[$(date -Is)] RF1 ALL DONE rc=$?"
