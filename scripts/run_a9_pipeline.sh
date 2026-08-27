#!/usr/bin/env bash
# A9 orchestration (WildPPG, ECG target = global TRAIN-ONLY z instead of per-window normalisation).
# Frozen A4/A6 recipes; the ONLY change is --processed (pre-normalisation ECG) + --target-norm.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
M=data/manifests/split_a4_wildppg_seed42.json; P=data/processed/wildppg_8s_prenorm; N=artifacts/a9_ecg_representation_control/normalization.json
PRE="--dataset WildPPG --raw-checksums data/raw/WildPPG/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220 --patience 20 --min-delta 1e-4"
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { until [ "$(free_mib)" -ge 22000 ]; do sleep 120; done; }
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 120; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
preflight() { "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "outputs/$1" --exp-name "$1" --seed 42 --window-s 8 --manifest $M --processed $P $PRE "${@:2}" > "outputs/$1/preflight.log" 2>&1 && grep -q "PREFLIGHT OK" "outputs/$1/preflight.log"; }
# --- MSE proxy (cheapest, runs first so the control is available early)
EXP=a9_mse_fullbackbone_wildppg_globalz_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A9 $EXP start $(date +%H:%M)"
preflight $EXP --objective mse_regression_full --n-val-banks 0 || { echo "A9 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a5 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --target-norm $N --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --model full_backbone --x-const 0.1 --t-const 0.5 --cond-scale 0.05 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A9 $EXP FAILED"; exit 1; }; echo "A9 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/eval_a9.py --arm mse --out-dir outputs/$EXP > outputs/$EXP/eval.log 2>&1 || { echo "A9 $EXP EVAL FAILED"; tail -3 outputs/$EXP/eval.log; exit 1; }; grep -E "^regressor" outputs/$EXP/eval.log
# --- OT-CFM
EXP=a9_otcfm_wildppg_globalz_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A9 $EXP start $(date +%H:%M)"
preflight $EXP --objective otcfm --select fixed_cfm --n-val-banks 4 --bank-seed 1000 --gen-diag-every 0 || { echo "A9 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a0 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --target-norm $N --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --select fixed_cfm --n-val-banks 4 --bank-seed 1000 --val-mae-every 0 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A9 $EXP FAILED"; exit 1; }; echo "A9 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/eval_a9.py --arm otcfm --out-dir outputs/$EXP > outputs/$EXP/eval.log 2>&1 || { echo "A9 $EXP EVAL FAILED"; tail -3 outputs/$EXP/eval.log; exit 1; }; grep -E "^heun25|^euler1" outputs/$EXP/eval.log
# --- iMeanFlow
EXP=a9_imeanflow_wildppg_globalz_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A9 $EXP start $(date +%H:%M)"
preflight $EXP --objective imeanflow --n-step 1 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 0 || { echo "A9 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a2 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --target-norm $N --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --cond-mode h_only --h-scale 1 --micro-batch 32 --batch-size 64 --val-batch 32 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A9 $EXP FAILED"; exit 1; }; echo "A9 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/eval_a9.py --arm imf --out-dir outputs/$EXP > outputs/$EXP/eval.log 2>&1 || { echo "A9 $EXP EVAL FAILED"; tail -3 outputs/$EXP/eval.log; exit 1; }; grep -E "^meanflow1" outputs/$EXP/eval.log
touch outputs/A9_DONE; echo "A9 PIPELINE FINISHED $(date +%H:%M)"
