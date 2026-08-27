#!/usr/bin/env bash
# A8 orchestration (MIMIC-BP, PPG->ABP, GLOBAL TRAIN-ONLY z-normalised target): OT-CFM -> iMF -> MSE, frozen A7 recipes,
# only --target-norm added. Waits for >= 22 GiB free GPU.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
M=data/manifests/split_a7_mimicbp_official.json; P=data/processed/mimicbp_8s; N=artifacts/a8_abp_scale_control/normalization.json
PRE="--dataset MIMIC-BP --raw-checksums data/raw/MIMIC-BP/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220 --patience 20 --min-delta 1e-4"
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { until [ "$(free_mib)" -ge 22000 ]; do sleep 120; done; }
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 120; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
preflight() { "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "outputs/$1" --exp-name "$1" --seed 42 --window-s 8 --manifest $M --processed $P $PRE "${@:2}" > "outputs/$1/preflight.log" 2>&1 && grep -q "PREFLIGHT OK" "outputs/$1/preflight.log"; }
# --- OT-CFM
EXP=a8_otcfm_mimicbp_globalz_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A8 $EXP start $(date +%H:%M)"
preflight $EXP --objective otcfm --select fixed_cfm --n-val-banks 4 --bank-seed 1000 --gen-diag-every 0 || { echo "A8 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a0 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --target-norm $N --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --select fixed_cfm --n-val-banks 4 --bank-seed 1000 --val-mae-every 0 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A8 $EXP FAILED: $(grep -E 'Error|Traceback' outputs/$EXP/train.log | tail -2 | tr '\n' ' ')"; exit 1; }; echo "A8 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/predict_a7.py --arm otcfm --out-dir outputs/$EXP > outputs/$EXP/predict.log 2>&1 || { echo "A8 $EXP PREDICT FAILED"; exit 1; }; grep -E "^heun25|^euler1" outputs/$EXP/predict.log
# --- iMeanFlow (the run A8 is about)
EXP=a8_imeanflow_mimicbp_globalz_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A8 $EXP start $(date +%H:%M)"
preflight $EXP --objective imeanflow --n-step 1 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 0 || { echo "A8 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a2 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --target-norm $N --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --cond-mode h_only --h-scale 1 --micro-batch 32 --batch-size 64 --val-batch 32 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A8 $EXP FAILED: $(grep -E 'Error|Traceback' outputs/$EXP/train.log | tail -2 | tr '\n' ' ')"; exit 1; }; echo "A8 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/predict_a7.py --arm imf --out-dir outputs/$EXP > outputs/$EXP/predict.log 2>&1 || { echo "A8 $EXP PREDICT FAILED"; exit 1; }; grep -E "^meanflow1" outputs/$EXP/predict.log
"$ROOT/.venv/bin/python" scripts/imf_diagnostics_a8.py --run outputs/$EXP --label A8-globalz-best >> outputs/$EXP/predict.log 2>&1; tail -1 outputs/$EXP/predict.log
# --- MSE proxy
EXP=a8_mse_fullbackbone_mimicbp_globalz_seed42; mkdir -p outputs/$EXP; wait_gpu; echo "A8 $EXP start $(date +%H:%M)"
preflight $EXP --objective mse_regression_full --n-val-banks 0 || { echo "A8 $EXP PREFLIGHT FAILED"; tail -3 outputs/$EXP/preflight.log; exit 1; }
nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a5 --exp-name $EXP --out-dir outputs/$EXP --manifest $M --processed $P --target-norm $N --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --model full_backbone --x-const 0.1 --t-const 0.5 --cond-scale 0.05 --gen-diag-every 0 --val-every-steps 220 --val-subsample 4096 > outputs/$EXP/train.log 2>&1 &
wait_done $EXP || { echo "A8 $EXP FAILED"; exit 1; }; echo "A8 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-140)"
"$ROOT/.venv/bin/python" scripts/predict_a7.py --arm mse --out-dir outputs/$EXP > outputs/$EXP/predict.log 2>&1 || { echo "A8 $EXP PREDICT FAILED"; exit 1; }; grep -E "^regressor" outputs/$EXP/predict.log
touch outputs/A8_DONE; echo "A8 PIPELINE FINISHED $(date +%H:%M)"
