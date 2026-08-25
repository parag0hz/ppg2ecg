#!/usr/bin/env bash
# Arm A0 launcher: preflight (hard gate) -> background training with nohup. Usage: bash scripts/run_a0.sh [--resume]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP=a0_penguin_otcfm_ppgdalia_8s_seed42
OUT="$ROOT/outputs/$EXP"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "$OUT" --exp-name "$EXP" --seed 42 --window-s 8
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a0 --out-dir "$OUT" --exp-name "$EXP" --seed 42 --processed data/processed/v0_8s --manifest data/manifests/split_p0_holdout_seed42.json "$@" > "$OUT/train.log" 2>&1 &
echo $! > "$OUT/train.pid"
echo "[run_a0] training started pid=$(cat "$OUT/train.pid") log=$OUT/train.log"
