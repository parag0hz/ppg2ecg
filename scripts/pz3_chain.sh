#!/usr/bin/env bash
# PZ3 runs after the SR1 chain (PID $1) has produced outputs/sr1_eval/arm_*_seed42.npz.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; cd "$ROOT"; export PYTHONDONTWRITEBYTECODE=1
echo "[$(date -Is)] waiting for SR1 pid $1"
while kill -0 "$1" 2>/dev/null; do sleep 60; done
for a in C I S; do [ -f "outputs/sr1_eval/arm_${a}_seed42.npz" ] || { echo "ABORT: missing arm_${a}_seed42.npz"; exit 1; }; done
echo "[$(date -Is)] SR1 done, starting PZ3"
"$ROOT/.venv/bin/python" scripts/pz3_compare.py > outputs/pz3.log 2>&1
echo "[$(date -Is)] PZ3 rc=$?"
