#!/usr/bin/env bash
# D3: finish every remaining evaluation, sequentially, with NO pgrep wait loop.
# (A `pgrep -f "d3_evaluate.py --corpus"` guard self-matches the shell that runs it and hangs forever.)
# Idempotent per (corpus, nfe-set): the evaluator merges rows keyed by (arm, nfe, metric).
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; B="$ROOT/outputs/d3_bench"
mkdir -p "$B"; cd "$ROOT" || exit 1
log(){ echo "$(date -Is) $*" | tee -a "$B/EVAL_ALL.log"; }
log "==== D3 remaining evaluations ===="
# 1) mimicbp at its frozen NFE set (iMF 1/2/4 + OT-CFM 50)
log "START mimicbp (frozen NFE set)"
"$PY" scripts/d3_evaluate.py --corpus mimicbp >> "$B/eval_mimicbp.log" 2>&1
log "$( [ $? -eq 0 ] && echo DONE || echo FAILED ) mimicbp"
# 2) the 50-NFE budget match for every corpus (deviation D3-2)
for c in bidmc_resp wesad_resp mimicbp uci_bp; do
  log "START $c @NFE50"
  "$PY" scripts/d3_evaluate.py --corpus "$c" --nfes 50 >> "$B/eval50_${c}.log" 2>&1
  log "$( [ $? -eq 0 ] && echo DONE || echo FAILED ) $c@50"
done
log "==== D3 evaluations finished ===="
