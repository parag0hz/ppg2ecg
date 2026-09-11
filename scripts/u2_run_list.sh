#!/usr/bin/env bash
# U2 step (c): paired iMeanFlow (arm I) vs OT-CFM (arm C), compute-matched.
# Preregistration: docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_PREREGISTRATION.md (435c50f)
#
# Both arms of a dataset run back to back so a partial stage still yields complete pairs.
# Sequential by construction: every step is chained inside THIS script, no wait guards anywhere.
set -uo pipefail

ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
LOGROOT="$ROOT/outputs/u2_paired"
PROG="$LOGROOT/progress_${U2_STREAM:-x}.log"
STEPS=14000

mkdir -p "$LOGROOT"
say() { echo "[$(date -Is)] $*" | tee -a "$PROG"; }

cd "$ROOT"
if [ -n "$(git -C external/PENGUIN status --porcelain)" ]; then say "ABORT: external/PENGUIN dirty"; exit 1; fi
[ "$(git -C external/PENGUIN rev-parse HEAD)" = "6cd70cdefb91f10efeb8dce34019b5067cb25344" ] || { say "ABORT: PENGUIN pin moved"; exit 1; }
if [ -d external/iMeanFlow ] && [ -n "$(git -C external/iMeanFlow status --porcelain)" ]; then say "ABORT: external/iMeanFlow dirty"; exit 1; fi
export PYTHONDONTWRITEBYTECODE=1
say "U2 STREAM ${U2_STREAM:-x} START repo=$(git rev-parse --short HEAD) steps=$STEPS"

# identical on both arms (prereg §6): same data, seed, optimiser, budget, validation cadence
COMMON=(--seed 42 --max-steps "$STEPS" --epochs 500 --val-every-steps 220 --val-subsample 1024
        --batch-size 64 --lr 1e-3 --weight-decay 0.01
        --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000)

run_arm() {   # $1 slug, $2 arm (C|I)
  local slug="$1" arm="$2"
  local out="$ROOT/outputs/${slug}_arm${arm}_seed42"
  local proc="$ROOT/data/processed/${slug}"
  local man="data/manifests/split_${slug}_seed42.json"
  if [ -f "$out/TRAINING_DONE" ]; then say "SKIP ${slug} arm ${arm} (TRAINING_DONE present)"; return 0; fi
  say "TRAIN START ${slug} arm ${arm}"
  if [ "$arm" = "C" ]; then
    "$PY" -m ppg2ecg.training.train_a0 --exp-name "${slug}_armC_seed42" --out-dir "$out" \
      --processed "$proc" --manifest "$man" "${COMMON[@]}" \
      --segment-len 4 --n-step 25 --select fixed_cfm --val-mae-every 0 \
      >"$LOGROOT/train_${slug}_armC.log" 2>&1
  else
    "$PY" -m ppg2ecg.training.train_a2 --exp-name "${slug}_armI_seed42" --out-dir "$out" \
      --processed "$proc" --manifest "$man" "${COMMON[@]}" \
      --m2-arm U --no-early-stop --micro-batch 32 --val-batch 32 \
      >"$LOGROOT/train_${slug}_armI.log" 2>&1
  fi
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out/TRAINING_DONE" ]; then
    say "TRAIN FAILED ${slug} arm ${arm} rc=$rc (see train_${slug}_arm${arm}.log)"; return 1
  fi
  say "TRAIN DONE ${slug} arm ${arm}  $("$PY" -c "
import json;d=json.load(open('$out/training_summary.json'))
print('opt_steps', d.get('opt_steps'), 'rounds', d.get('epochs_run'), 'best', d.get('best_epoch'), 'time_s', round(d.get('total_train_time_s',0)))")"
}

pair() {  # $1 slug -- both arms back to back
  run_arm "$1" C
  run_arm "$1" I
}

# dataset list comes from the command line (disjoint per stream)

say "U2 ALL DONE"
for slug in "$@"; do pair "$slug"; done
say "U2 STREAM DONE ($*)"
