#!/usr/bin/env bash
# Evaluate every U2 corpus whose arm C and arm I are both trained and which has no metrics yet.
# Chained: one corpus after another inside this single script, no wait guards.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step
cd "$ROOT"
for c in u2_bidmc u2_wesad u2_dalia u2_wildppg u2_mimicbp u2_ucibp; do
  [ -f "artifacts/u2_paired/metrics_${c}.csv" ] && { echo "[skip] $c already evaluated"; continue; }
  [ -f "outputs/${c}_armC_seed42/TRAINING_DONE" ] && [ -f "outputs/${c}_armI_seed42/TRAINING_DONE" ] || {
    echo "[wait] $c pair not complete yet"; continue; }
  echo "[eval] $c  $(date -Is)"
  .venv/bin/python scripts/u2_evaluate.py --corpus "$c" || echo "[FAILED] $c"
done
echo "[eval] pending pass done $(date -Is)"
