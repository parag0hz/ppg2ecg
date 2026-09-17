#!/usr/bin/env bash
# BB1: train A and B in parallel, then evaluate I/A/B in parallel, then the verdict. Waits on PIDs only.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/bb1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
COMMON=(--seed 42 --max-steps 14000 --epochs 500 --val-every-steps 220 --val-subsample 1024
        --batch-size 64 --lr 1e-3 --weight-decay 0.01 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000
        --processed "$ROOT/data/processed/v1_vitaldb" --manifest data/manifests/split_v1_vitaldb_seed42.json
        --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32)
echo "[$(date -Is)] START repo=$(git rev-parse --short HEAD)"
"$PY" -m ppg2ecg.training.train_a2 --exp-name bb1_vitaldb_A_seed42 --out-dir "$ROOT/outputs/bb1_vitaldb_A_seed42" \
  "${COMMON[@]}" --h-dim 256 --backbone s5 > "$LOG/train_A.log" 2>&1 & pa=$!
"$PY" -m ppg2ecg.training.train_a2 --exp-name bb1_vitaldb_B_seed42 --out-dir "$ROOT/outputs/bb1_vitaldb_B_seed42" \
  "${COMMON[@]}" --h-dim 128 --backbone attn --attn-heads 4 > "$LOG/train_B.log" 2>&1 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
echo "[$(date -Is)] TRAIN DONE A rc=$ra B rc=$rb"
[ -f outputs/bb1_vitaldb_A_seed42/TRAINING_DONE ] && [ -f outputs/bb1_vitaldb_B_seed42/TRAINING_DONE ] || { echo "ABORT: training incomplete"; exit 1; }
for a in I A B; do "$PY" scripts/bb1_evaluate.py --arm $a > "$LOG/eval_$a.log" 2>&1 & eval "p$a=\$!"; done
wait $pI; wait $pA; wait $pB
echo "[$(date -Is)] EVAL DONE"
"$PY" scripts/bb1_verdict.py > "$LOG/verdict.log" 2>&1
echo "[$(date -Is)] ALL DONE rc=$?"
