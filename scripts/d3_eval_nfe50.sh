#!/usr/bin/env bash
# D3 budget match: iMF at 50 NFE, PENGUIN's own inference budget (deviation D3-2, disclosed in the report).
# ALL of NFE 1/2/4/50 end up in the table; nothing is selected after the fact.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; B="$ROOT/outputs/d3_bench"
mkdir -p "$B"; cd "$ROOT" || exit 1
log(){ echo "$(date -Is) $*" | tee -a "$B/EVAL50.log"; }
# wait for any evaluation still holding the GPU
while pgrep -f "d3_evaluate.py --corpus" >/dev/null; do sleep 30; done
log "==== D3 NFE-50 budget match start ===="
for c in bidmc_resp wesad_resp mimicbp uci_bp; do
  log "START $c @50"
  "$PY" scripts/d3_evaluate.py --corpus "$c" --nfes 50 >> "$B/eval50_${c}.log" 2>&1
  log "$( [ $? -eq 0 ] && echo DONE || echo FAILED ) $c"
done
log "==== D3 NFE-50 budget match finished ===="
