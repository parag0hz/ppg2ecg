#!/usr/bin/env bash
# U3-B: both arms at exactly 14,000 optimizer steps (checkpoint_last.pt).
# Preregistration docs/U3_GATE_AND_SELECTION_SENSITIVITY_PREREGISTRATION.md §4. Trains nothing.
set -uo pipefail
cd /home/kwy00/ppg2ecg-one-step
for c in u2_bidmc u2_wesad u2_dalia u2_wildppg u2_mimicbp u2_ucibp; do
  [ -f "artifacts/u2_paired/metrics_${c}_last.csv" ] && { echo "[skip] $c"; continue; }
  echo "[u3b] $c  $(date -Is)"
  .venv/bin/python scripts/u2_evaluate.py --corpus "$c" --checkpoint last || echo "[FAILED] $c"
done
echo "[u3b] done $(date -Is)"
