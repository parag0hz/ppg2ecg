#!/usr/bin/env bash
# Arm A2: Improved MeanFlow on the identical S5 backbone (docs/A2_IMEANFLOW_PREREGISTRATION.md). Preflight gate -> background training.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP=a2_imeanflow_s5_ppgdalia_8s_seed42
OUT="$ROOT/outputs/$EXP"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "$OUT" --exp-name "$EXP" --seed 42 --window-s 8 --patience 20 --objective imeanflow --select fixed_cfm --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --val-mae-every 0 --gen-diag-every 1 --n-step 1
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a2 --out-dir "$OUT" --exp-name "$EXP" --seed 42 --patience 20 --min-delta 1e-4 --epochs 300 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 1 --gen-diag-windows 128 --cond-mode h_only --h-scale 1 --processed data/processed/v0_8s --manifest data/manifests/split_p0_holdout_seed42.json "$@" > "$OUT/train.log" 2>&1 &
echo $! > "$OUT/train.pid"
echo "[run_a2] training started pid=$(cat "$OUT/train.pid") log=$OUT/train.log"
