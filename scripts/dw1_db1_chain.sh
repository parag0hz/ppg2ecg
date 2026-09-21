#!/usr/bin/env bash
# Wait for LW1 (PID $1), then DW1 (no training), then DB1. PIDs only.
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; cd "$ROOT"; export PYTHONDONTWRITEBYTECODE=1
echo "[$(date -Is)] waiting for LW1 pid $1"; while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] DW1 START repo=$(git rev-parse --short HEAD)"
$PY scripts/dw1_depth_width.py > outputs/dw1.log 2>&1; echo "[$(date -Is)] DW1 rc=$?"
$PY scripts/db1_discriminative_hr.py > outputs/db1.log 2>&1; echo "[$(date -Is)] DB1 rc=$?"
