#!/usr/bin/env bash
# SR1: seeds 1 and 2 for arms C, I, S (seed 42 exists), then evaluate all 9 runs, then the frozen verdicts.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/sr1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
COMMON=(--epochs 500 --max-steps 14000 --val-every-steps 220 --val-subsample 1024 --batch-size 64 --lr 1e-3
        --weight-decay 0.01 --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000
        --processed "$ROOT/data/processed/v1_vitaldb" --manifest data/manifests/split_v1_vitaldb_seed42.json)
echo "[$(date -Is)] SR1 START repo=$(git rev-parse --short HEAD)"
for seed in 1 2; do
  $PY -m ppg2ecg.training.train_a0 --exp-name sr1_C_seed$seed --out-dir "$ROOT/outputs/sr1_C_seed$seed" --seed $seed \
     "${COMMON[@]}" --segment-len 4 --n-step 25 --select fixed_cfm --val-mae-every 0 > "$LOG/train_C_seed$seed.log" 2>&1 & pc=$!
  $PY -m ppg2ecg.training.train_vimf --arch S --init scratch --seed $seed --micro-batch 64 \
     --out-dir "$ROOT/outputs/sr1_S_seed$seed" > "$LOG/train_S_seed$seed.log" 2>&1 & ps=$!
  wait $pc; echo "[$(date -Is)] C seed$seed rc=$?"; wait $ps; echo "[$(date -Is)] S seed$seed rc=$?"
  $PY -m ppg2ecg.training.train_a2 --exp-name sr1_I_seed$seed --out-dir "$ROOT/outputs/sr1_I_seed$seed" --seed $seed \
     "${COMMON[@]}" --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32 > "$LOG/train_I_seed$seed.log" 2>&1
  echo "[$(date -Is)] I seed$seed rc=$?"
done
for seed in 1 2; do for a in C I S; do
  [ -f "outputs/sr1_${a}_seed${seed}/TRAINING_DONE" ] || { echo "ABORT: ${a} seed${seed} incomplete"; exit 1; }
done; done
for a in C I S; do
  ( for seed in 42 1 2; do "$PY" scripts/sr1_evaluate.py --arm $a --seed $seed > "$LOG/eval_${a}_seed${seed}.log" 2>&1 || exit 1; done ) & eval "p$a=\$!"
done
wait $pC; echo "[$(date -Is)] eval C rc=$?"; wait $pI; echo "[$(date -Is)] eval I rc=$?"; wait $pS; echo "[$(date -Is)] eval S rc=$?"
"$PY" scripts/sr1_verdict.py > "$LOG/verdict.log" 2>&1
echo "[$(date -Is)] SR1 ALL DONE rc=$?"
