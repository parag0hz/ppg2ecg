#!/usr/bin/env bash
# B1-v2 orchestration: 6 fixed-budget runs (vanilla + curriculum per dataset), each followed by evaluation of the FINAL (primary)
# and BEST-validation (secondary) checkpoints with the frozen eval pipeline. GPU-gated (>= 22 GiB free).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { until [ "$(free_mib)" -ge 22000 ]; do sleep 120; done; }
wait_done() { until [ -f "outputs/$1/TRAINING_DONE" ] || [ -f "outputs/$1/TRAINING_FAILED" ]; do sleep 180; done; [ -f "outputs/$1/TRAINING_DONE" ]; }
run_one() {  # EXP ARM MANIFEST PROCESSED TSCHED EXTRA_TRAIN... (EVAL_EXTRA env for eval)
  local EXP="$1" ARM="$2" M="$3" P="$4" TS="$5"; shift 5
  mkdir -p "outputs/$EXP"; wait_gpu; echo "B1 $EXP start $(date +%m-%d\ %H:%M)"
  "$ROOT/.venv/bin/python" scripts/preflight_a0.py --out-dir "outputs/$EXP" --exp-name "$EXP" --seed 42 --window-s 8 --patience 20 --min-delta 1e-4 --manifest "$M" --processed "$P" --objective imeanflow --n-step 1 --n-val-banks 4 --bank-seed 1000 --gen-diag-every 1 ${PREFLIGHT_EXTRA:-} > "outputs/$EXP/preflight.log" 2>&1
  grep -q "PREFLIGHT OK" "outputs/$EXP/preflight.log" || { echo "B1 $EXP PREFLIGHT FAILED"; tail -3 "outputs/$EXP/preflight.log"; exit 1; }
  nohup "$ROOT/.venv/bin/python" -m ppg2ecg.training.train_b1_fixed_compute --exp-name "$EXP" --out-dir "outputs/$EXP" --arm "$ARM" --t-schedule "$TS" --manifest "$M" --processed "$P" --seed 42 --epochs 300 --patience 20 --min-delta 1e-4 --n-val-banks 4 --bank-seed 1000 --cond-mode h_only --h-scale 1 --micro-batch 32 --batch-size 64 --val-batch 32 --gen-diag-every 5 "$@" --resume > "outputs/$EXP/train.log" 2>&1 &
  echo $! > "outputs/$EXP/train.pid"
  wait_done "$EXP" || { echo "B1 $EXP FAILED: $(grep -E 'Error|Traceback' outputs/$EXP/train.log | tail -2 | tr '\n' ' ')"; exit 1; }
  echo "B1 $EXP done $(date +%m-%d\ %H:%M): $(tr -d '\n' < outputs/$EXP/training_summary.json | cut -c1-200)"
  for CK in final best; do
    CKF="checkpoint_final.pt"; [ "$CK" = "best" ] && CKF="checkpoint_best.pt"
    mkdir -p "outputs/$EXP/eval_$CK"
    "$ROOT/.venv/bin/python" scripts/eval_a2.py --out-dir "outputs/$EXP/eval_$CK" --checkpoint "outputs/$EXP/$CKF" --manifest "$M" --processed "$P" --steps 1,2,4 ${EVAL_EXTRA:-} > "outputs/$EXP/eval_$CK.log" 2>&1 || { echo "B1 $EXP EVAL($CK) FAILED"; tail -3 "outputs/$EXP/eval_$CK.log"; exit 1; }
    echo "  eval_$CK: $(grep -m1 "meanflow steps=1 " outputs/$EXP/eval_$CK.log | cut -c1-170)"
  done
  "$ROOT/.venv/bin/python" scripts/imf_diagnostics_a8.py --run "outputs/$EXP" --label "B1-$EXP-best" --checkpoint checkpoint_best.pt --manifest "$M" --processed "$P" --out artifacts/b1_gap_curriculum/imf_diagnostics.csv >> "outputs/$EXP/eval_final.log" 2>&1 || true
  "$ROOT/.venv/bin/python" scripts/imf_diagnostics_a8.py --run "outputs/$EXP" --label "B1-$EXP-final" --checkpoint checkpoint_final.pt --manifest "$M" --processed "$P" --out artifacts/b1_gap_curriculum/imf_diagnostics.csv >> "outputs/$EXP/eval_final.log" 2>&1 || true
}
MS2=data/manifests/split_p0_holdout_seed42.json; MS1=data/manifests/split_a3_testS1_valS11.json; MW=data/manifests/split_a4_wildppg_seed42.json
PD=data/processed/v0_8s; PW=data/processed/wildppg_8s
PREFLIGHT_EXTRA="" EVAL_EXTRA="" run_one b1v2_vanilla_fixed_dalia_s2_seed42    vanilla    $MS2 $PD 66000
PREFLIGHT_EXTRA="" EVAL_EXTRA="" run_one b1v2_curriculum_fixed_dalia_s2_seed42 curriculum $MS2 $PD 66000
PREFLIGHT_EXTRA="" EVAL_EXTRA="" run_one b1v2_vanilla_fixed_dalia_s1_seed42    vanilla    $MS1 $PD 65400
PREFLIGHT_EXTRA="" EVAL_EXTRA="" run_one b1v2_curriculum_fixed_dalia_s1_seed42 curriculum $MS1 $PD 65400
PREFLIGHT_EXTRA="--dataset WildPPG --raw-checksums data/raw/WildPPG/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220" EVAL_EXTRA="--subsample 4096" run_one b1v2_vanilla_fixed_wildppg_seed42    vanilla    $MW $PW 65482 --val-every-steps 220 --val-subsample 4096
PREFLIGHT_EXTRA="--dataset WildPPG --raw-checksums data/raw/WildPPG/CHECKSUMS.sha256 --val-subsample 4096 --val-every-steps 220" EVAL_EXTRA="--subsample 4096" run_one b1v2_curriculum_fixed_wildppg_seed42 curriculum $MW $PW 65482 --val-every-steps 220 --val-subsample 4096
touch outputs/B1_DONE; echo "B1 PIPELINE FINISHED $(date +%m-%d\ %H:%M)"
