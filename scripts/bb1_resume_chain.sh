#!/usr/bin/env bash
# BB1 execution-order fix: arm A OOM'd at start while B ran in parallel (no weight update happened for A).
# Wait for B's PID, then train A alone with the unchanged preregistered argv, then evaluate and decide.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/bb1_logs"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
PB="$1"
COMMON=(--seed 42 --max-steps 14000 --epochs 500 --val-every-steps 220 --val-subsample 1024
        --batch-size 64 --lr 1e-3 --weight-decay 0.01 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000
        --processed "$ROOT/data/processed/v1_vitaldb" --manifest data/manifests/split_v1_vitaldb_seed42.json
        --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32)
echo "[$(date -Is)] waiting for B pid $PB"
while kill -0 "$PB" 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] B finished: $([ -f outputs/bb1_vitaldb_B_seed42/TRAINING_DONE ] && echo done || echo NOT-done)"
"$PY" -m ppg2ecg.training.train_a2 --exp-name bb1_vitaldb_A_seed42 --out-dir "$ROOT/outputs/bb1_vitaldb_A_seed42" \
  "${COMMON[@]}" --h-dim 256 --backbone s5 > "$LOG/train_A.log" 2>&1
echo "[$(date -Is)] A rc=$?"
[ -f outputs/bb1_vitaldb_A_seed42/TRAINING_DONE ] && [ -f outputs/bb1_vitaldb_B_seed42/TRAINING_DONE ] || { echo "ABORT: training incomplete"; exit 1; }
"$PY" scripts/bb1_evaluate.py --arm I > "$LOG/eval_I.log" 2>&1 & p1=$!
"$PY" scripts/bb1_evaluate.py --arm A > "$LOG/eval_A.log" 2>&1 & p2=$!
"$PY" scripts/bb1_evaluate.py --arm B > "$LOG/eval_B.log" 2>&1 & p3=$!
wait $p1; wait $p2; wait $p3
echo "[$(date -Is)] EVAL DONE"
"$PY" scripts/bb1_verdict.py > "$LOG/verdict.log" 2>&1
echo "[$(date -Is)] ALL DONE rc=$?"
