#!/usr/bin/env bash
# A5 orchestration: wait until the GPU has >= 22 GiB free (an unrelated process may hold it), then A5a -> A5b -> A5c (train + eval).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while [ "$(free_mib)" -lt 22000 ]; do sleep 120; done; }
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 120; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
run_one() {  # name manifest processed extra_train_args... ; EVAL_EXTRA env for eval
  local EXP="$1" M="$2" P="$3"; shift 3
  wait_gpu; echo "A5 $EXP start $(date +%H:%M) (GPU free $(free_mib) MiB)"
  mkdir -p "outputs/$EXP"
  "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "outputs/$EXP" --exp-name "$EXP" --manifest "$M" --processed "$P" --objective mse_regression --patience 20 --min-delta 1e-4 --n-val-banks 0 ${PREFLIGHT_EXTRA:-} > "outputs/$EXP/preflight.log" 2>&1 || { echo "A5 $EXP PREFLIGHT FAILED"; tail -3 "outputs/$EXP/preflight.log"; exit 1; }
  grep -q "PREFLIGHT OK" "outputs/$EXP/preflight.log" || { echo "A5 $EXP PREFLIGHT NOT OK"; exit 1; }
  nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_a5 --exp-name "$EXP" --out-dir "outputs/$EXP" --manifest "$M" --processed "$P" --seed 42 --patience 20 --min-delta 1e-4 --epochs 300 "$@" > "outputs/$EXP/train.log" 2>&1 &
  echo $! > "outputs/$EXP/train.pid"
  wait_done "$EXP" || { echo "A5 $EXP FAILED: $(grep -E 'Error|Traceback' outputs/$EXP/train.log | tail -2 | tr '\n' ' ')"; exit 1; }
  echo "A5 $EXP done $(date +%H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-160)"
  "$ROOT/.venv/bin/python" scripts/eval_a5.py --out-dir "outputs/$EXP" --manifest "$M" --processed "$P" ${EVAL_EXTRA:-} > "outputs/$EXP/eval.log" 2>&1 || { echo "A5 $EXP EVAL FAILED"; exit 1; }
  grep -E '^regressor' "outputs/$EXP/eval.log" | cut -c1-200
}
PREFLIGHT_EXTRA="" EVAL_EXTRA="" run_one a5a_mse_regressor_dalia_testS2_seed42 data/manifests/split_p0_holdout_seed42.json data/processed/v0_8s
PREFLIGHT_EXTRA="" EVAL_EXTRA="" run_one a5b_mse_regressor_dalia_testS1_seed42 data/manifests/split_a3_testS1_valS11.json data/processed/v0_8s
PREFLIGHT_EXTRA="--dataset WildPPG --raw-checksums data/raw/WildPPG/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220" EVAL_EXTRA="--subsample 4096" run_one a5c_mse_regressor_wildppg_seed42 data/manifests/split_a4_wildppg_seed42.json data/processed/wildppg_8s --val-every-steps 220 --val-subsample 4096
touch outputs/A5_DONE; echo "A5 PIPELINE FINISHED $(date +%H:%M)"
