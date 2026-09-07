#!/usr/bin/env bash
# M3 ARM E — SEC-iMF (docs/M3_SEC_IMEANFLOW_PREREGISTRATION.md §6-§8).
# Exactly the frozen A4 / M2-U recipe, with the single permitted difference: --m3-arm E adds the clean-endpoint
# structural auxiliary at lambda_SEC = 0.10. 66 validation rounds = 14,409 optimizer steps, early stopping
# disabled, checkpoint selection by fixed_imf_mse only. TRAIN12 only; kjd/ssx never loaded. Arm U is NOT retrained.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/outputs/m3_bench"; mkdir -p "$BENCH"
OUT="$ROOT/outputs/m3_arm_E_seed42"
cd "$ROOT" || exit 1
if [ -f "$OUT/TRAINING_DONE" ]; then echo "$(date -Is) SKIP arm E (TRAINING_DONE exists)" | tee -a "$BENCH/PROGRESS.log"; exit 0; fi
echo "$(date -Is) START arm E -> $OUT" | tee -a "$BENCH/PROGRESS.log"
"$PY" -m ppg2ecg.training.train_a2 --exp-name m3_arm_E_seed42 --out-dir "$OUT" --m3-arm E \
  --processed data/processed/wildppg_8s \
  --manifest data/manifests/split_a4_wildppg_seed42.json \
  --seed 42 --epochs 66 --no-early-stop --patience 20 --min-delta 0.0001 \
  --batch-size 64 --micro-batch 32 --lr 0.001 --weight-decay 0.01 \
  --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128 \
  --p-mean -0.4 --p-std 1.0 --data-proportion 0.5 --norm-p 1.0 --norm-eps 0.01 \
  --jvp-mode forward --cond-mode h_only --h-scale 1.0 --c1-arm B \
  --val-batch 32 --n-val-banks 4 --bank-seed 1000 \
  --gen-diag-every 1 --gen-diag-windows 128 --val-every-steps 220 --val-subsample 4096 \
  >> "$BENCH/arm_E.log" 2>&1
echo "$(date -Is) $([ $? -eq 0 ] && echo DONE || echo FAILED) arm E" | tee -a "$BENCH/PROGRESS.log"
