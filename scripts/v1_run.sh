#!/usr/bin/env bash
# V1: one arm (C|I) on v1_vitaldb with the unchanged U2 recipe. Usage: scripts/v1_run.sh C
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; arm="$1"; slug=v1_vitaldb
LOG="$ROOT/outputs/v1_logs"; mkdir -p "$LOG"; cd "$ROOT"; export PYTHONDONTWRITEBYTECODE=1
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
COMMON=(--seed 42 --max-steps 14000 --epochs 500 --val-every-steps 220 --val-subsample 1024
        --batch-size 64 --lr 1e-3 --weight-decay 0.01
        --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000)
out="$ROOT/outputs/${slug}_arm${arm}_seed42"; man="data/manifests/split_${slug}_seed42.json"; proc="$ROOT/data/processed/$slug"
echo "[$(date -Is)] START $arm repo=$(git rev-parse --short HEAD)"
if [ "$arm" = C ]; then
  "$PY" -m ppg2ecg.training.train_a0 --exp-name "${slug}_armC_seed42" --out-dir "$out" --processed "$proc" --manifest "$man" \
    "${COMMON[@]}" --segment-len 4 --n-step 25 --select fixed_cfm --val-mae-every 0 > "$LOG/train_armC.log" 2>&1
else
  "$PY" -m ppg2ecg.training.train_a2 --exp-name "${slug}_armI_seed42" --out-dir "$out" --processed "$proc" --manifest "$man" \
    "${COMMON[@]}" --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32 > "$LOG/train_armI.log" 2>&1
fi
echo "[$(date -Is)] END $arm rc=$? done=$([ -f "$out/TRAINING_DONE" ] && echo yes || echo no)"
