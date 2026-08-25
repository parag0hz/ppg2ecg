#!/usr/bin/env bash
# Arm A0-b: identical to A0 except checkpoint selection = deterministic fixed-bank val CFM loss, patience 20, min_delta 1e-4
# (docs/A0B_BASELINE_STABILIZATION_PREREGISTRATION.md). Usage: bash scripts/run_a0b.sh [--resume]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP=a0b_penguin_otcfm_ppgdalia_8s_seed42
OUT="$ROOT/outputs/$EXP"
SEL="--select fixed_cfm --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --val-mae-every 0 --gen-diag-every 5"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
"$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "$OUT" --exp-name "$EXP" --seed 42 --window-s 8 --patience 20 $SEL
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a0 --out-dir "$OUT" --exp-name "$EXP" --seed 42 --patience 20 --epochs 300 $SEL --gen-diag-windows 128 --processed data/processed/v0_8s --manifest data/manifests/split_p0_holdout_seed42.json "$@" > "$OUT/train.log" 2>&1 &
echo $! > "$OUT/train.pid"
echo "[run_a0b] training started pid=$(cat "$OUT/train.pid") log=$OUT/train.log"
