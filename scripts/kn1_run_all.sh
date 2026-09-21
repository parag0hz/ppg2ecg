#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/kwy00/ppg2ecg-one-step; PY="$ROOT/.venv/bin/python"; LOG="$ROOT/outputs/kn1_logs"; mkdir -p "$LOG"; cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -z "$(git -C external/PENGUIN status --porcelain)" ] || { echo "ABORT: PENGUIN dirty"; exit 1; }
COMMON=(--seed 42 --max-steps 14000 --epochs 500 --val-every-steps 220 --val-subsample 1024 --batch-size 64 --lr 1e-3
        --weight-decay 0.01 --h-dim 128 --blocks 4 --ssm-ratio 2.0 --mlp-ratio 2.0 --sample-rate 128
        --gen-diag-every 0 --patience 999999 --n-val-banks 4 --bank-seed 1000
        --processed "$ROOT/data/processed/v1_vitaldb" --manifest data/manifests/split_v1_vitaldb_seed42.json)
IMF=(--m2-arm U --no-early-stop --micro-batch 32 --val-batch 32)
echo "[$(date -Is)] KN1 START repo=$(git rev-parse --short HEAD)"
$PY -m ppg2ecg.training.train_a2 --exp-name kn1_I_kan_outer --out-dir "$ROOT/outputs/kn1_I_kan_outer" "${COMMON[@]}" "${IMF[@]}" --backbone kan-outer > "$LOG/train_IKo.log" 2>&1 & p1=$!
$PY -m ppg2ecg.training.train_a0 --exp-name kn1_C_kan_outer --out-dir "$ROOT/outputs/kn1_C_kan_outer" "${COMMON[@]}" --segment-len 4 --n-step 25 --select fixed_cfm --val-mae-every 0 --backbone kan-outer > "$LOG/train_CKo.log" 2>&1 & p2=$!
wait $p2; echo "[$(date -Is)] C-Ko rc=$?"
$PY -m ppg2ecg.training.train_a2 --exp-name kn1_I_kan_all --out-dir "$ROOT/outputs/kn1_I_kan_all" "${COMMON[@]}" "${IMF[@]}" --backbone kan-all > "$LOG/train_IKa.log" 2>&1 & p3=$!
wait $p1; echo "[$(date -Is)] I-Ko rc=$?"; wait $p3; echo "[$(date -Is)] I-Ka rc=$?"
for d in kn1_I_kan_outer kn1_I_kan_all kn1_C_kan_outer; do [ -f "outputs/$d/TRAINING_DONE" ] || { echo "ABORT: $d incomplete"; exit 1; }; done
for a in IKo IKa CKo; do $PY scripts/kn1_evaluate.py --arm $a > "$LOG/eval_$a.log" 2>&1 & eval "e$a=\$!"; done
wait $eIKo; wait $eIKa; wait $eCKo; echo "[$(date -Is)] EVAL DONE"
$PY scripts/kn1_verdict.py > "$LOG/verdict.log" 2>&1; echo "[$(date -Is)] KN1 ALL DONE rc=$?"
