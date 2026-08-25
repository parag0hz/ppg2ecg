#!/usr/bin/env bash
# Evaluate a finished experiment with the frozen evaluation code. Usage: bash scripts/run_eval_chain.sh {otcfm|imf} EXP_NAME MANIFEST [PROCESSED_DIR]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="$1"; EXP="$2"; MANIFEST="$3"; PROCESSED="${4:-data/processed/v0_8s}"
OUT="$ROOT/outputs/$EXP"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
test -f "$OUT/TRAINING_DONE" || { echo "$EXP not finished"; exit 1; }
if [ "$OBJ" = "otcfm" ]; then
  "$ROOT/.venv/bin/python" scripts/eval_a0_nfe_curve.py --out-dir "$OUT" --manifest "$MANIFEST" --processed "$PROCESSED" --batch-size 64 --noise-seed 0 --diversity-seeds 4 --diversity-windows 256 --bench-repeats 10 2>&1 | grep -vE "Warning|warn|conv1d" | tee "$OUT/eval.log"
else
  "$ROOT/.venv/bin/python" scripts/eval_a2.py --out-dir "$OUT" --manifest "$MANIFEST" --processed "$PROCESSED" --steps 1,2,4 --batch-size 64 --noise-seed 0 --diversity-seeds 4 --diversity-windows 256 --bench-repeats 10 2>&1 | grep -vE "Warning|warn|conv1d" | tee "$OUT/eval.log"
fi
touch "$OUT/EVAL_DONE"
