#!/usr/bin/env bash
# Finish D3: rerun mimicbp (iMF 1/2/4 + OT-CFM 50, now that the bare-backbone/Heun path exists) at both NFE sets.
# No pgrep guard: this script is chained AFTER the running one by launching it in the same tmux session queue.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; B="$ROOT/outputs/d3_bench"
cd "$ROOT" || exit 1
log(){ echo "$(date -Is) $*" | tee -a "$B/EVAL_ALL.log"; }
log "START mimicbp (iMF 1/2/4 + OT-CFM 50)"
"$PY" scripts/d3_evaluate.py --corpus mimicbp >> "$B/eval_mimicbp.log" 2>&1
log "$( [ $? -eq 0 ] && echo DONE || echo FAILED ) mimicbp"
log "START mimicbp @NFE50 (iMF budget match)"
"$PY" scripts/d3_evaluate.py --corpus mimicbp --nfes 50 >> "$B/eval50_mimicbp.log" 2>&1
log "$( [ $? -eq 0 ] && echo DONE || echo FAILED ) mimicbp@50"
log "==== D3 ALL EVALUATIONS COMPLETE ===="
