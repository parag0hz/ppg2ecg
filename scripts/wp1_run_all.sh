#!/usr/bin/env bash
# WP1: 4 folds x {C, I, D}. Per fold: C || I, then D (from that fold's C); fold k's evaluation runs in the
# background while fold k+1 trains. PIDs only.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/wp1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
COMMON=(--seed 42 --max-steps 14000 --epochs 500 --val-every-steps 220 --val-subsample 1024 --batch-size 64 --lr 1e-3
        --weight-decay 0.01 --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000 --processed "$ROOT/data/processed/u2_wildppg")
echo "[$(date -Is)] WP1 START repo=$(git rev-parse --short HEAD)"
EVALS=()
for k in 0 1 2 3; do
  MAN="data/manifests/split_wp1_fold$k.json"
  $PY -m ppg2ecg.training.train_a0 --exp-name wp1_f${k}_armC --out-dir "$ROOT/outputs/wp1_f${k}_armC" --manifest "$MAN" \
     "${COMMON[@]}" --segment-len 4 --n-step 25 --select fixed_cfm --val-mae-every 0 > "$LOG/train_f${k}_C.log" 2>&1 & pc=$!
  $PY -m ppg2ecg.training.train_a2 --exp-name wp1_f${k}_armI --out-dir "$ROOT/outputs/wp1_f${k}_armI" --manifest "$MAN" \
     "${COMMON[@]}" --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32 > "$LOG/train_f${k}_I.log" 2>&1 & pi=$!
  wait $pc; echo "[$(date -Is)] fold $k C rc=$?"
  $PY scripts/cd1_train.py --out-dir "$ROOT/outputs/wp1_f${k}_armD" --teacher "$ROOT/outputs/wp1_f${k}_armC/checkpoint_last.pt" \
     --manifest "$MAN" --processed data/processed/u2_wildppg > "$LOG/train_f${k}_D.log" 2>&1 & pd=$!
  wait $pi; echo "[$(date -Is)] fold $k I rc=$?"; wait $pd; echo "[$(date -Is)] fold $k D rc=$?"
  for a in C I D; do [ -f "outputs/wp1_f${k}_arm${a}/TRAINING_DONE" ] || { echo "ABORT: fold $k arm $a incomplete"; exit 1; }; done
  ( for a in C I D; do $PY scripts/wp1_evaluate.py --fold $k --arm $a > "$LOG/eval_f${k}_${a}.log" 2>&1 || exit 1; done ) & EVALS+=($!)
done
for p in "${EVALS[@]}"; do wait $p; echo "[$(date -Is)] eval pid $p rc=$?"; done
$PY scripts/wp1_verdict.py > "$LOG/verdict.log" 2>&1; echo "[$(date -Is)] WP1 ALL DONE rc=$?"
