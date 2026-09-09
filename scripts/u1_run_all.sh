#!/usr/bin/env bash
# U1 — upstream PENGUIN, as shipped, on all six shipped datasets.
# Preregistration: docs/U1_UPSTREAM_PENGUIN_ASSHIPPED_PREREGISTRATION.md
#
# Upstream code is executed UNMODIFIED. Only the config overrides declared in prereg §3.2 are applied.
# Every step is chained inside this one script: no "wait for the other job" guards anywhere.
set -uo pipefail

ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/configs/upstream_u1"
PROC="$ROOT/data/processed/upstream_u1"
LOGROOT="$ROOT/outputs/u1_upstream"
PROG="$LOGROOT/progress.log"

mkdir -p "$LOGROOT" "$PROC"
say() { echo "[$(date -Is)] $*" | tee -a "$PROG"; }

# ---- preflight -------------------------------------------------------------
if [ -n "$(git -C "$ROOT/external/PENGUIN" status --porcelain)" ]; then
  say "ABORT: external/PENGUIN is dirty"; exit 1
fi
UPSTREAM_SHA=$(git -C "$ROOT/external/PENGUIN" rev-parse HEAD)
if [ "$UPSTREAM_SHA" != "6cd70cdefb91f10efeb8dce34019b5067cb25344" ]; then
  say "ABORT: external/PENGUIN pin moved: $UPSTREAM_SHA"; exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
if ! "$PY" -c "import torch, numpy as np; np.linalg.eigh(np.eye(4)*-1j)" >/dev/null 2>&1; then
  say "preflight: exporting MKL_THREADING_LAYER=SEQUENTIAL"
  export MKL_THREADING_LAYER=SEQUENTIAL
fi
say "U1 START  repo=$(git -C "$ROOT" rev-parse --short HEAD)  upstream=$UPSTREAM_SHA"

# ---- helpers ---------------------------------------------------------------
run_preprocess() {  # $1 dataset, $2.. extra overrides
  local ds="$1"; shift
  say "PREPROCESS START $ds"
  ( cd "$ROOT/external/PENGUIN/src" && \
    "$PY" preprocess.py --config-path "$CFG" --config-name preprocess.yaml \
      preprocess.dataset="$ds" \
      hydra.run.dir="$ROOT/outputs/hydra/u1_preprocess_${ds}" hydra.output_subdir=.hydra "$@" \
  ) >"$LOGROOT/preprocess_${ds}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then say "PREPROCESS FAILED $ds rc=$rc"; return $rc; fi
  say "PREPROCESS DONE $ds  files=$(ls "$PROC/$ds" | wc -l)"
}

run_split() {  # $1 dataset, $2 subject_num
  say "SPLIT $1"
  "$PY" "$ROOT/scripts/u1_split_manifest.py" --procdata "$PROC" --dataset "$1" \
     --subject-num "$2" --fold-num 8 --seed 42 \
     --out "$LOGROOT/split_${1}.json" >"$LOGROOT/split_${1}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { say "SPLIT FAILED $1 rc=$rc"; return $rc; }
  say "SPLIT DONE $1"
}

run_train() {  # $1 dataset, $2.. extra overrides
  local ds="$1"; shift
  say "TRAIN START $ds  overrides: $*"
  ( cd "$ROOT/external/PENGUIN/src" && \
    "$PY" train.py --config-path "$CFG" --config-name config.yaml \
      train.dataset="$ds" train.model=PENGUIN train.logging.description=u1 \
      hydra.run.dir="$ROOT/outputs/hydra/u1_train_${ds}" hydra.output_subdir=.hydra "$@" \
  ) >"$LOGROOT/train_${ds}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then say "TRAIN FAILED $ds rc=$rc (see train_${ds}.log)"; return $rc; fi
  say "TRAIN DONE $ds"
  grep -E "^(MAE|HeartRateError|RespRateError|SBPError|DBPError|Inference Time) " \
    "$LOGROOT/train_${ds}.log" | tee -a "$PROG"
}

do_dataset() {  # $1 dataset, $2 subject_num, then extra overrides shared by both stages
  local ds="$1"; local sn="$2"; shift 2
  if [ ! -d "$PROC/$ds" ] || [ "$(ls "$PROC/$ds" 2>/dev/null | wc -l)" -ne "$sn" ]; then
    run_preprocess "$ds" "$@" || return 1
  else
    say "PREPROCESS SKIP $ds (already $sn files)"
  fi
  run_split "$ds" "$sn" || return 1
  return 0
}

# ---- Tier A: shipped schedule, uncapped -----------------------------------
WILD_VIEW="preprocess.rawdata_path=$ROOT/data/raw_u1_wildppg_view"
WILD_N="preprocess.WildPPG.subject_num=14"

# WildPPG firewall: the view the loader globs must hold exactly 14 subjects and never kjd/ssx.
wildppg_firewall() {
  local view="$ROOT/data/raw_u1_wildppg_view/WildPPG"
  local n; n=$(ls "$view"/*.mat 2>/dev/null | wc -l)
  if [ "$n" -ne 14 ]; then say "ABORT: WildPPG view has $n subjects, expected 14"; return 1; fi
  if ls "$view" | grep -qE 'kjd|ssx'; then say "ABORT: FIREWALL VIOLATION — kjd/ssx present in the WildPPG view"; return 1; fi
  ls "$view"/*.mat | xargs -n1 basename > "$LOGROOT/wildppg_view_subjects.txt"
  say "FIREWALL OK: WildPPG view = 14 subjects, kjd/ssx absent"
}

do_dataset BIDMC 53      && run_train BIDMC
do_dataset WESAD 15      && run_train WESAD
do_dataset PPG-DaLiA 15  && run_train PPG-DaLiA

# ---- U1-P1: UCI-BP subject-duplication check (file hashes only) ------------
# runs after UCI-BP preprocessing below

# ---- Tier B: epoch-capped (prereg §3.4) -----------------------------------
do_dataset MIMIC-BP 1524 && run_train MIMIC-BP train.training.epoch_num=20
wildppg_firewall && do_dataset WildPPG 14 "$WILD_VIEW" "$WILD_N" && run_train WildPPG "$WILD_N" train.training.epoch_num=12
do_dataset UCI-BP 8      && run_train UCI-BP train.training.epoch_num=12

say "U1 ALL DONE"
