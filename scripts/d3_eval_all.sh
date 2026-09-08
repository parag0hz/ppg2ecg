#!/usr/bin/env bash
# D3 evaluation, sequential on one GPU. Idempotent: a corpus whose metrics CSV exists is skipped.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"; A="$ROOT/artifacts/d3_penguin_six_dataset"; B="$ROOT/outputs/d3_bench"
mkdir -p "$B"; cd "$ROOT" || exit 1
log(){ echo "$(date -Is) $*" | tee -a "$B/EVAL.log"; }
log "==== D3 evaluation start ===="
for c in wesad_resp mimicbp uci_bp; do
  [ -s "$A/metrics_${c}.csv" ] && { log "SKIP $c"; continue; }
  log "START $c"
  "$PY" scripts/d3_evaluate.py --corpus "$c" >> "$B/eval_${c}.log" 2>&1
  log "$( [ $? -eq 0 ] && echo DONE || echo FAILED ) $c"
done
log "==== D3 evaluation finished ===="
