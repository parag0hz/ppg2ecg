#!/usr/bin/env bash
# U1 tail: run the two remaining Tier-B datasets (WildPPG, UCI-BP) CONCURRENTLY.
#
# Scheduling change only. Each dataset is still upstream preprocess.py + train.py, unmodified,
# with exactly the prereg §3.2 overrides; the two processes share nothing (separate procdata,
# separate log dirs, separate hydra run dirs, own seed inside each process), so no preregistered
# quantity is affected. Recorded in the report as an execution note.
#
# Everything is chained inside THIS one script: it waits for MIMIC-BP's marker in the progress
# log (a file grep, never a process-name match), stops the sequential runner, then forks.
set -uo pipefail

ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/configs/upstream_u1"
PROC="$ROOT/data/processed/upstream_u1"
LOGROOT="$ROOT/outputs/u1_upstream"
PROG="$LOGROOT/progress.log"
say() { echo "[$(date -Is)] $*" | tee -a "$PROG"; }

export PYTHONDONTWRITEBYTECODE=1
if ! "$PY" -c "import torch, numpy as np; np.linalg.eigh(np.eye(4)*-1j)" >/dev/null 2>&1; then
  export MKL_THREADING_LAYER=SEQUENTIAL
fi

# ---- 1. wait for MIMIC-BP to finish, then stop the sequential runner --------
say "PARALLEL-TAIL armed: waiting for MIMIC-BP to finish"
while ! grep -q "TRAIN DONE MIMIC-BP" "$PROG"; do
  if ! tmux has-session -t u1 2>/dev/null; then say "sequential runner already gone"; break; fi
  sleep 15
done
if tmux has-session -t u1 2>/dev/null; then
  say "stopping sequential runner (tmux kill-session u1) so WildPPG/UCI-BP can run in parallel"
  tmux kill-session -t u1
  sleep 5
fi
# the sequential runner may have started WildPPG preprocessing before it was stopped
rm -rf "$PROC/WildPPG"
say "PARALLEL-TAIL START  free VRAM check:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$PROG"

# ---- 2. one chain per dataset, the two chains run concurrently -------------
chain() {  # $1 dataset, $2 subject_num, $3 epoch cap, then extra hydra overrides
  local ds="$1" sn="$2" cap="$3"; shift 3
  local pre_extra=() tr_extra=()
  if [ "$ds" = "WildPPG" ]; then
    local view="$ROOT/data/raw_u1_wildppg_view/WildPPG"
    local n; n=$(ls "$view"/*.mat 2>/dev/null | wc -l)
    if [ "$n" -ne 14 ]; then say "ABORT $ds: view has $n subjects, expected 14"; return 1; fi
    if ls "$view" | grep -qE 'kjd|ssx'; then say "ABORT $ds: FIREWALL VIOLATION kjd/ssx in view"; return 1; fi
    ls "$view"/*.mat | xargs -n1 basename > "$LOGROOT/wildppg_view_subjects.txt"
    say "FIREWALL OK $ds: 14 subjects, kjd/ssx absent"
    pre_extra=("preprocess.rawdata_path=$ROOT/data/raw_u1_wildppg_view" "preprocess.WildPPG.subject_num=14")
    tr_extra=("preprocess.WildPPG.subject_num=14")
  fi

  if [ "$(ls "$PROC/$ds" 2>/dev/null | wc -l)" -ne "$sn" ]; then
    say "PREPROCESS START $ds"
    ( cd "$ROOT/external/PENGUIN/src" && \
      "$PY" preprocess.py --config-path "$CFG" --config-name preprocess.yaml \
        preprocess.dataset="$ds" \
        hydra.run.dir="$ROOT/outputs/hydra/u1_preprocess_${ds}" hydra.output_subdir=.hydra \
        "${pre_extra[@]}" ) >"$LOGROOT/preprocess_${ds}.log" 2>&1
    local rc=$?; [ $rc -ne 0 ] && { say "PREPROCESS FAILED $ds rc=$rc"; return $rc; }
    say "PREPROCESS DONE $ds  files=$(ls "$PROC/$ds" | wc -l)"
  else
    say "PREPROCESS SKIP $ds (already $sn files)"
  fi

  say "SPLIT $ds"
  "$PY" "$ROOT/scripts/u1_split_manifest.py" --procdata "$PROC" --dataset "$ds" \
     --subject-num "$sn" --fold-num 8 --seed 42 \
     --out "$LOGROOT/split_${ds}.json" >"$LOGROOT/split_${ds}.log" 2>&1
  local rc=$?; [ $rc -ne 0 ] && { say "SPLIT FAILED $ds rc=$rc"; return $rc; }
  say "SPLIT DONE $ds  $(python3 -c "
import json;d=json.load(open('$LOGROOT/split_${ds}.json'))
print('windows', d['n_windows'], 'total', d['n_windows_total'], 'dupgroups', len(d['duplicate_groups']))")"

  say "TRAIN START $ds  (epoch cap $cap)"
  ( cd "$ROOT/external/PENGUIN/src" && \
    "$PY" train.py --config-path "$CFG" --config-name config.yaml \
      train.dataset="$ds" train.model=PENGUIN train.logging.description=u1 \
      train.training.epoch_num="$cap" \
      hydra.run.dir="$ROOT/outputs/hydra/u1_train_${ds}" hydra.output_subdir=.hydra \
      "${tr_extra[@]}" ) >"$LOGROOT/train_${ds}.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && { say "TRAIN FAILED $ds rc=$rc (see train_${ds}.log)"; return $rc; }
  say "TRAIN DONE $ds"
  grep -E "^(MAE|HeartRateError|RespRateError|SBPError|DBPError|Inference Time) " \
    "$LOGROOT/train_${ds}.log" | sed "s/^/[$ds] /" | tee -a "$PROG"
}

chain WildPPG 14 12 & PID_W=$!
chain UCI-BP   8 12 & PID_U=$!
wait $PID_W; RC_W=$?
wait $PID_U; RC_U=$?
say "PARALLEL-TAIL rc: WildPPG=$RC_W UCI-BP=$RC_U"

# ---- 3. U1-P1: UCI-BP subject-duplication check (file hashes only) ---------
if [ -f "$LOGROOT/split_UCI-BP.json" ]; then
  say "U1-P1 UCI-BP duplication check:"
  python3 -c "
import json
d=json.load(open('$LOGROOT/split_UCI-BP.json'))
g=d['duplicate_groups']
print('  duplicate groups:', len(g))
for k,v in g.items(): print('   ', sorted(v))
print('  val :', d['val']); print('  test:', d['test']); print('  train:', sorted(d['train']))
leak=[n for k,v in g.items() for n in v if n in d['test'] and any(o in d['train'] for o in v)]
print('  TEST SUBJECT DUPLICATED INTO TRAIN:', bool(leak), leak)
" | tee -a "$PROG"
fi

say "U1 ALL DONE"
