#!/usr/bin/env bash
# M2 paired training (docs/M2_STRUCTURE_WEIGHTED_IMEANFLOW_PREREGISTRATION.md §3, §7, §33 steps 10-12).
#
# Runs arms STRICTLY IN SEQUENCE on one GPU, each for exactly 66 epochs with early stopping DISABLED, so the arms
# are step-for-step paired at A4's realised budget of 302,478 optimizer steps and neither exceeds it.
#   U : uniform control      (--m2-arm U -> structure_weight=None, i.e. the untouched pre-M2 code path)
#   S : GSW-iMF, the proposal (--m2-arm S)
#   X : shifted control      — ONLY if the primary gates pass; not launched here
# Everything except the loss multiplier is identical: seed, init, data order, batch composition, RNG streams,
# optimizer, schedule, step count, checkpoint rule, precision, hardware.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/outputs/m2_bench"
PROGRESS="$BENCH/PROGRESS.log"
ARMS=("${@:-U S}")
mkdir -p "$BENCH"

log() { echo "$(date -Is) $*" | tee -a "$PROGRESS"; }

common=(--processed data/processed/wildppg_8s
        --manifest data/manifests/split_a4_wildppg_seed42.json
        --seed 42 --epochs 66 --no-early-stop --patience 20 --min-delta 0.0001
        --batch-size 64 --micro-batch 32 --lr 0.001 --weight-decay 0.01
        --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --p-mean -0.4 --p-std 1.0 --data-proportion 0.5 --norm-p 1.0 --norm-eps 0.01
        --jvp-mode forward --cond-mode h_only --h-scale 1.0 --c1-arm B
        --val-batch 32 --n-val-banks 4 --bank-seed 1000
        --gen-diag-every 1 --gen-diag-windows 128 --val-every-steps 220 --val-subsample 4096)

cd "$ROOT" || exit 1
log "==== M2 paired training start (arms: ${ARMS[*]}) ===="
for arm in ${ARMS[*]}; do
  out="$ROOT/outputs/m2_arm_${arm}_seed42"
  if [ -f "$out/TRAINING_DONE" ]; then log "SKIP  arm $arm (TRAINING_DONE exists)"; continue; fi
  log "START arm $arm -> $out"
  "$PY" -m ppg2ecg.training.train_a2 --exp-name "m2_arm_${arm}_seed42" --out-dir "$out" \
        --m2-arm "$arm" "${common[@]}" >> "$BENCH/arm_${arm}.log" 2>&1
  rc=$?
  log "$( [ $rc -eq 0 ] && echo DONE || echo FAILED ) arm $arm (rc=$rc)"
done
log "==== M2 paired training finished ===="
