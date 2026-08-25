#!/usr/bin/env bash
# After A0-b training: NFE-curve evaluation (identical to A0), A0 vs A0-b comparison + mechanical gate, report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
test -f "$OUT/TRAINING_DONE" || { echo "A0-b not finished"; exit 1; }
"$ROOT/.venv/bin/python" scripts/eval_a0_nfe_curve.py --out-dir "$OUT" --batch-size 64 --noise-seed 0 --diversity-seeds 4 --diversity-windows 256 --bench-repeats 10 2>&1 | grep -vE "Warning|warn|conv1d" | tee "$OUT/eval.log"
"$ROOT/.venv/bin/python" scripts/compare_a0b.py 2>&1 | tee "$OUT/compare.log"
touch "$OUT/EVAL_DONE"
