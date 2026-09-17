#!/usr/bin/env bash
# VM1: wait for the BB1 chain (PID $1), then train S, P, B, evaluate all three, decide. PIDs only, no pgrep.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/vm1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$(git -C external/iMeanFlow status --porcelain)" ] || { echo "ABORT: iMeanFlow dirty"; exit 1; }
echo "[$(date -Is)] waiting for BB1 chain pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] BB1 chain finished; repo=$(git rev-parse --short HEAD)"
T="$PY -m ppg2ecg.training.train_vimf"
pick_mb() {  # $1 init
  for mb in 64 32 16; do
    rm -rf "$ROOT/outputs/vm1_dryrun"
    if $T --arch B --init "$1" --out-dir "$ROOT/outputs/vm1_dryrun" --max-steps 3 --log-every 3 --micro-batch $mb > "$LOG/dryrun_$1_$mb.log" 2>&1; then
      echo $mb; return; fi
  done
  echo 0
}
$T --arch S --init scratch --out-dir "$ROOT/outputs/vm1_S_seed42" --micro-batch 64 > "$LOG/train_S.log" 2>&1
echo "[$(date -Is)] S rc=$?"
MB=$(pick_mb official); echo "[$(date -Is)] B/2 micro-batch $MB"
[ "$MB" != 0 ] || { echo "ABORT: B/2 does not fit"; exit 1; }
rm -rf "$ROOT/outputs/vm1_dryrun"
$T --arch B --init official --out-dir "$ROOT/outputs/vm1_P_seed42" --micro-batch "$MB" > "$LOG/train_P.log" 2>&1
echo "[$(date -Is)] P rc=$?"
$T --arch B --init scratch --out-dir "$ROOT/outputs/vm1_B_seed42" --micro-batch "$MB" > "$LOG/train_B.log" 2>&1
echo "[$(date -Is)] B rc=$?"
for a in S P B; do [ -f "outputs/vm1_${a}_seed42/TRAINING_DONE" ] || { echo "ABORT: $a incomplete"; exit 1; }; done
"$PY" scripts/vm1_evaluate.py --arm S > "$LOG/eval_S.log" 2>&1 & e1=$!
"$PY" scripts/vm1_evaluate.py --arm P > "$LOG/eval_P.log" 2>&1 & e2=$!
"$PY" scripts/vm1_evaluate.py --arm B > "$LOG/eval_B.log" 2>&1 & e3=$!
wait $e1; wait $e2; wait $e3
echo "[$(date -Is)] EVAL DONE"
"$PY" scripts/vm1_verdict.py > "$LOG/verdict.log" 2>&1
echo "[$(date -Is)] ALL DONE rc=$?"
