#!/usr/bin/env bash
# Generic launcher: preflight gate -> background training with the FROZEN A0-b (otcfm) or A2 (imf) recipe.
# Usage: bash scripts/run_exp.sh {otcfm|imf} EXP_NAME MANIFEST [PROCESSED_DIR] [extra training args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBJ="$1"; EXP="$2"; MANIFEST="$3"; PROCESSED="${4:-data/processed/v0_8s}"; shift 4 || shift $#
OUT="$ROOT/outputs/$EXP"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
COMMON_SEL="--select fixed_cfm --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --val-mae-every 0"
if [ "$OBJ" = "otcfm" ]; then
  "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "$OUT" --exp-name "$EXP" --seed 42 --window-s 8 --patience 20 --manifest "$MANIFEST" --processed "$PROCESSED" $COMMON_SEL --gen-diag-every 5
  nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a0 --out-dir "$OUT" --exp-name "$EXP" --seed 42 --patience 20 --epochs 300 $COMMON_SEL --gen-diag-every 5 --gen-diag-windows 128 --processed "$PROCESSED" --manifest "$MANIFEST" "$@" > "$OUT/train.log" 2>&1 &
elif [ "$OBJ" = "imf" ]; then
  "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "$OUT" --exp-name "$EXP" --seed 42 --window-s 8 --patience 20 --manifest "$MANIFEST" --processed "$PROCESSED" --objective imeanflow $COMMON_SEL --gen-diag-every 1 --n-step 1
  nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a2 --out-dir "$OUT" --exp-name "$EXP" --seed 42 --patience 20 --min-delta 1e-4 --epochs 300 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 1 --gen-diag-windows 128 --cond-mode h_only --h-scale 1 --micro-batch 32 --val-batch 32 --processed "$PROCESSED" --manifest "$MANIFEST" "$@" > "$OUT/train.log" 2>&1 &
else
  echo "unknown objective $OBJ"; exit 1
fi
echo $! > "$OUT/train.pid"
echo "[run_exp] $OBJ $EXP started pid=$(cat "$OUT/train.pid") log=$OUT/train.log"
