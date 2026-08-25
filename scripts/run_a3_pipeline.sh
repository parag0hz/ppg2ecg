#!/usr/bin/env bash
# A3 orchestration (sequential GPU use): wait OT-CFM(S1) training -> evaluate -> launch iMF(S1) -> wait -> evaluate -> compare/report.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
M=data/manifests/split_a3_testS1_valS11.json
OT=a3_otcfm_ppgdalia_testS1_seed42; IMF=a3_imeanflow_ppgdalia_testS1_seed42
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 120; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
wait_done "$OT" || { echo "A3 OT-CFM FAILED"; exit 1; }
echo "A3 OT-CFM done $(date +%H:%M): $(cat outputs/$OT/training_summary.json | tr -d '\n' | cut -c1-200)"
bash scripts/run_eval_chain.sh otcfm "$OT" "$M" > "outputs/$OT/eval_chain.log" 2>&1 || { echo "A3 OT-CFM EVAL FAILED"; exit 1; }
grep -E '^(heun|euler) ' "outputs/$OT/eval.log" | cut -c1-200
bash scripts/run_exp.sh imf "$IMF" "$M" data/processed/v0_8s > "outputs/${IMF}_launch.log" 2>&1 || { echo "A3 iMF LAUNCH FAILED: $(tail -3 outputs/${IMF}_launch.log)"; exit 1; }
echo "A3 iMF launched $(date +%H:%M)"
wait_done "$IMF" || { echo "A3 iMF FAILED"; exit 1; }
echo "A3 iMF done $(date +%H:%M): $(cat outputs/$IMF/training_summary.json | tr -d '\n' | cut -c1-200)"
bash scripts/run_eval_chain.sh imf "$IMF" "$M" > "outputs/$IMF/eval_chain.log" 2>&1 || { echo "A3 iMF EVAL FAILED"; exit 1; }
grep -E '^meanflow ' "outputs/$IMF/eval.log" | cut -c1-200
PYTHONPATH=src .venv/bin/python scripts/compare_a2.py --otcfm "outputs/$OT" --imf "outputs/$IMF" --a0 "outputs/$OT" --manifest "$M" --report docs/A3_SUBJECT_REPLICATION_REPORT.md --title "A3 Subject Replication Report (PPG-DaLiA, test S1)" --prereg docs/A3_A4_REPLICATION_PREREGISTRATION.md --dataset-label "PPG-DaLiA (test S1)" > "outputs/$IMF/compare.log" 2>&1
grep -E '"verdict"|"replication_verdict"|"pointwise_error_inversion"|wrote' "outputs/$IMF/compare.log"
touch outputs/A3_DONE; echo "A3 PIPELINE FINISHED $(date +%H:%M)"
