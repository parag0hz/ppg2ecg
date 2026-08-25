#!/usr/bin/env bash
# A4 orchestration (single GPU): wait for A3 to finish -> OT-CFM(WildPPG) train -> eval -> iMF(WildPPG) train -> eval -> compare/report.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
M=data/manifests/split_a4_wildppg_seed42.json; P=data/processed/wildppg_8s
OT=a4_otcfm_wildppg_seed42; IMF=a4_imeanflow_wildppg_seed42
export PREFLIGHT_EXTRA="--dataset WildPPG --raw-checksums data/raw/WildPPG/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220"
TRAIN_EXTRA="--val-every-steps 220 --val-subsample 4096"
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 120; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
until [ -f outputs/A3_DONE ]; do sleep 300; done
echo "A4 start $(date +%H:%M) (A3 finished)"
bash scripts/run_exp.sh otcfm "$OT" "$M" "$P" $TRAIN_EXTRA > "outputs/${OT}_launch.log" 2>&1 || { echo "A4 OT-CFM LAUNCH/PREFLIGHT FAILED: $(grep -E 'PREFLIGHT|Error' outputs/${OT}_launch.log | tail -2)"; exit 1; }
echo "A4 OT-CFM launched"
wait_done "$OT" || { echo "A4 OT-CFM FAILED"; exit 1; }
echo "A4 OT-CFM done $(date +%H:%M): $(tr -d '\n' < outputs/$OT/training_summary.json | cut -c1-200)"
PYTHONPATH=src .venv/bin/python scripts/eval_a0_nfe_curve.py --out-dir "outputs/$OT" --manifest "$M" --processed "$P" --subsample 4096 --batch-size 64 --noise-seed 0 --diversity-seeds 4 --diversity-windows 256 --bench-repeats 10 > "outputs/$OT/eval.log" 2>&1 || { echo "A4 OT-CFM EVAL FAILED"; exit 1; }
grep -E '^(heun|euler) ' "outputs/$OT/eval.log" | cut -c1-200
bash scripts/run_exp.sh imf "$IMF" "$M" "$P" $TRAIN_EXTRA > "outputs/${IMF}_launch.log" 2>&1 || { echo "A4 iMF LAUNCH/PREFLIGHT FAILED: $(grep -E 'PREFLIGHT|Error' outputs/${IMF}_launch.log | tail -2)"; exit 1; }
echo "A4 iMF launched $(date +%H:%M)"
wait_done "$IMF" || { echo "A4 iMF FAILED"; exit 1; }
echo "A4 iMF done $(date +%H:%M): $(tr -d '\n' < outputs/$IMF/training_summary.json | cut -c1-200)"
PYTHONPATH=src .venv/bin/python scripts/eval_a2.py --out-dir "outputs/$IMF" --manifest "$M" --processed "$P" --subsample 4096 --steps 1,2,4 --batch-size 64 --noise-seed 0 --diversity-seeds 4 --diversity-windows 256 --bench-repeats 10 > "outputs/$IMF/eval.log" 2>&1 || { echo "A4 iMF EVAL FAILED"; exit 1; }
grep -E '^meanflow ' "outputs/$IMF/eval.log" | cut -c1-200
PYTHONPATH=src .venv/bin/python scripts/compare_a2.py --otcfm "outputs/$OT" --imf "outputs/$IMF" --a0 "outputs/$OT" --manifest "$M" --processed "$P" --report docs/A4_WILDPPG_REPLICATION_REPORT.md --title "A4 WildPPG Replication Report" --prereg docs/A3_A4_REPLICATION_PREREGISTRATION.md --dataset-label "WildPPG (test kjd, ssx)" > "outputs/$IMF/compare.log" 2>&1
grep -E '"verdict"|"replication_verdict"|"pointwise_error_inversion"|wrote' "outputs/$IMF/compare.log"
touch outputs/A4_DONE; echo "A4 PIPELINE FINISHED $(date +%H:%M)"
