#!/usr/bin/env bash
# A7 orchestration (MIMIC-BP, PPG->ABP): waits for A6 to finish and >= 22 GiB free, then OT-CFM -> iMF -> MSE(full backbone) training
# (A4 schedule unit: 220-step rounds, <= 4096-window val/test subsets) and prediction dumps. Exact A0-b / A2 / A6 recipes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
M=data/manifests/split_a7_mimicbp_official.json; P=data/processed/mimicbp_8s; DS=mimicbp
PRE="--dataset MIMIC-BP --raw-checksums data/raw/MIMIC-BP/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220 --patience 20 --min-delta 1e-4"
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { until [ -f outputs/A6_DONE ] && [ "$(free_mib)" -ge 22000 ]; do sleep 120; done; }
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 120; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
preflight() { "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "outputs/$1" --exp-name "$1" --seed 42 --window-s 8 --manifest $M --processed $P $PRE "${@:2}" > "outputs/$1/preflight.log" 2>&1 && grep -q "PREFLIGHT OK" "outputs/$1/preflight.log"; }
# --- OT-CFM (A0-b recipe)
EXP=a7_otcfm_${DS}_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A7 $EXP start $(date +%H:%M)"
preflight $EXP --objective otcfm --select fixed_cfm --n-val-banks 4 --bank-seed 1000 --gen-diag-every 0 || { echo "A7 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a0 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --select fixed_cfm --n-val-banks 4 --bank-seed 1000 --val-mae-every 0 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A7 $EXP FAILED"; exit 1; }; echo "A7 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/predict_a7.py --arm otcfm --out-dir outputs/$EXP > outputs/$EXP/predict.log 2>&1 || { echo "A7 $EXP PREDICT FAILED"; exit 1; }; grep -E "^heun25|^euler1" outputs/$EXP/predict.log
# --- iMeanFlow (A2 recipe)
EXP=a7_imeanflow_${DS}_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A7 $EXP start $(date +%H:%M)"
preflight $EXP --objective imeanflow --n-step 1 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 0 || { echo "A7 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a2 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --cond-mode h_only --h-scale 1 --micro-batch 32 --batch-size 64 --val-batch 32 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A7 $EXP FAILED"; exit 1; }; echo "A7 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/predict_a7.py --arm imf --out-dir outputs/$EXP > outputs/$EXP/predict.log 2>&1 || { echo "A7 $EXP PREDICT FAILED"; exit 1; }; grep -E "^meanflow1" outputs/$EXP/predict.log
# --- MSE full-backbone proxy (A6 recipe)
EXP=a7_mse_fullbackbone_${DS}_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A7 $EXP start $(date +%H:%M)"
preflight $EXP --objective mse_regression_full --n-val-banks 0 || { echo "A7 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a5 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --model full_backbone ${A6_XCONST_ARGS:-} --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A7 $EXP FAILED"; exit 1; }; echo "A7 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/predict_a7.py --arm mse --out-dir outputs/$EXP > outputs/$EXP/predict.log 2>&1 || { echo "A7 $EXP PREDICT FAILED"; exit 1; }; grep -E "^regressor" outputs/$EXP/predict.log
touch outputs/A7_DONE; echo "A7 PIPELINE FINISHED $(date +%H:%M)"
