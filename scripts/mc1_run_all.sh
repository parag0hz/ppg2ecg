#!/usr/bin/env bash
# MC1 overnight: wait for ED2 (PID $1), then per corpus train C || I, then D, then evaluate (in background while the
# next corpus trains). PIDs only.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/mc1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date -Is)] waiting for ED2 pid $1"; while kill -0 "$1" 2>/dev/null; do sleep 60; done
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
echo "[$(date -Is)] MC1 START repo=$(git rev-parse --short HEAD)"
EVALS=()
for c in dalia bidmc capnobase; do
  MAN="data/manifests/split_mc1_$c.json"
  PROC="$ROOT/$($PY -c "import json;print(json.load(open('$MAN'))['extra']['processed'])")"
  COMMON=(--seed 42 --max-steps 14000 --epochs 5000 --val-every-steps 220 --val-subsample 1024 --batch-size 64 --lr 1e-3
          --weight-decay 0.01 --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
          --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000 --processed "$PROC" --manifest "$MAN")
  $PY -m ppg2ecg.training.train_a0 --exp-name mc1_${c}_armC --out-dir "$ROOT/outputs/mc1_${c}_armC" "${COMMON[@]}" \
     --segment-len 4 --n-step 25 --select fixed_cfm --val-mae-every 0 > "$LOG/train_${c}_C.log" 2>&1 & pc=$!
  $PY -m ppg2ecg.training.train_a2 --exp-name mc1_${c}_armI --out-dir "$ROOT/outputs/mc1_${c}_armI" "${COMMON[@]}" \
     --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32 > "$LOG/train_${c}_I.log" 2>&1 & pi=$!
  wait $pc; echo "[$(date -Is)] $c C rc=$?"
  $PY scripts/cd1_train.py --out-dir "$ROOT/outputs/mc1_${c}_armD" --teacher "$ROOT/outputs/mc1_${c}_armC/checkpoint_last.pt" \
     --manifest "$MAN" --processed "${PROC#$ROOT/}" > "$LOG/train_${c}_D.log" 2>&1 & pd=$!
  wait $pi; echo "[$(date -Is)] $c I rc=$?"; wait $pd; echo "[$(date -Is)] $c D rc=$?"
  ok=1; for a in C I D; do [ -f "outputs/mc1_${c}_arm${a}/TRAINING_DONE" ] || ok=0; done
  if [ $ok -eq 1 ]; then $PY scripts/mc1_evaluate.py --corpus $c > "$LOG/eval_$c.log" 2>&1 & EVALS+=($!)
  else echo "[$(date -Is)] $c SKIPPED (training incomplete)"; fi
done
for p in "${EVALS[@]}"; do wait $p; echo "[$(date -Is)] eval pid $p rc=$?"; done
echo "[$(date -Is)] MC1 ALL DONE"
