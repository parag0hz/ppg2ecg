#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/lw1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
COMMON=(--seed 42 --max-steps 14000 --epochs 500 --val-every-steps 220 --val-subsample 1024 --batch-size 64 --lr 1e-3
        --weight-decay 0.01 --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000 --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32
        --processed "$ROOT/data/processed/v1_vitaldb" --manifest data/manifests/split_v1_vitaldb_seed42.json --aux-lambda 0.5)
train() { $PY -m ppg2ecg.training.train_a2 --exp-name "lw1_$1" --out-dir "$ROOT/outputs/lw1_$1" "${COMMON[@]}" "${@:2}" > "$LOG/train_$1.log" 2>&1; }
echo "[$(date -Is)] LW1 START repo=$(git rev-parse --short HEAD)"
train W5 --aux-loss mae_pcc --aux-alpha 0.5 & a=$!; train SP --aux-loss stft & b=$!
wait $a; echo "[$(date -Is)] W5 rc=$?"; wait $b; echo "[$(date -Is)] SP rc=$?"
train W0 --aux-loss mae_pcc --aux-alpha 0.0 & a=$!; train W10 --aux-loss mae_pcc --aux-alpha 1.0 & b=$!
wait $a; echo "[$(date -Is)] W0 rc=$?"; wait $b; echo "[$(date -Is)] W10 rc=$?"
for x in W0 W5 W10 SP; do [ -f "outputs/lw1_$x/TRAINING_DONE" ] || { echo "ABORT: $x incomplete"; exit 1; }; done
P=(); for x in I W0 W5 W10 SP; do $PY scripts/lw1_evaluate.py --arm $x > "$LOG/eval_$x.log" 2>&1 & P+=($!); done
for p in "${P[@]}"; do wait $p; done; echo "[$(date -Is)] EVAL DONE"
$PY scripts/lw1_verdict.py > "$LOG/verdict.log" 2>&1; echo "[$(date -Is)] LW1 ALL DONE rc=$?"
