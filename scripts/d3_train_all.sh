#!/usr/bin/env bash
# D3 training (docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md §6).
#
# Trains the three NEW iMeanFlow arms STRICTLY IN SEQUENCE on one GPU, each under the frozen recipe:
# seed 42, 66 validation rounds = 14,409 optimizer steps, early stopping disabled, checkpoint selection by
# fixed_imf_mse ONLY -- never by HR, RR, SBP, DBP or any D3 table metric.
#   uci_bp      PPG -> ABP,         8 s
#   wesad_resp  PPG -> respiration, 4 s
#   bidmc_resp  PPG -> respiration, 4 s
# NOT retrained (reused frozen arms): MIMIC-BP (a7_imeanflow), WildPPG (A4), PPG-DaLiA (A3/A0-b).
#
# Idempotent: an arm whose TRAINING_DONE exists is skipped. A failed arm is logged and the run continues.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
BENCH="$ROOT/outputs/d3_bench"; mkdir -p "$BENCH"
PROGRESS="$BENCH/PROGRESS.log"
log(){ echo "$(date -Is) $*" | tee -a "$PROGRESS"; }

# corpus:segment_len — small corpora first so a fault surfaces in minutes, not hours
ARMS=("bidmc_resp:4" "wesad_resp:4" "uci_bp:8")

common=(--seed 42 --epochs 66 --no-early-stop --patience 20 --min-delta 0.0001
        --batch-size 64 --micro-batch 32 --lr 0.001 --weight-decay 0.01
        --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --p-mean -0.4 --p-std 1.0 --data-proportion 0.5 --norm-p 1.0 --norm-eps 0.01
        --jvp-mode forward --cond-mode h_only --h-scale 1.0 --c1-arm B
        --val-batch 32 --n-val-banks 4 --bank-seed 1000
        --gen-diag-every 1 --gen-diag-windows 128 --val-every-steps 220 --val-subsample 4096)

cd "$ROOT" || exit 1
log "==== D3 training start (arms: ${ARMS[*]}) ===="
for spec in "${ARMS[@]}"; do
  c="${spec%%:*}"; seg="${spec##*:}"
  out="$ROOT/outputs/d3_${c}_seed42"
  if [ -f "$out/TRAINING_DONE" ]; then log "SKIP  $c (TRAINING_DONE exists)"; continue; fi
  log "START $c (segment_len ${seg}s) -> $out"
  "$PY" -m ppg2ecg.training.train_a2 --exp-name "d3_${c}_seed42" --out-dir "$out" \
        --processed "data/processed/${c}_${seg}s" \
        --manifest "data/manifests/split_d3_${c}_seed42.json" \
        "${common[@]}" >> "$BENCH/${c}.log" 2>&1
  rc=$?
  log "$( [ $rc -eq 0 ] && echo DONE || echo FAILED ) $c (rc=$rc)"
done
log "==== D3 training finished ===="
