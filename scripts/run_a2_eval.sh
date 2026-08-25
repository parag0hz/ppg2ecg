#!/usr/bin/env bash
# After A2 training: 1-NFE (+2/4-step diagnostic) evaluation with the same paired noise, then the controlled comparison,
# recovery scores, verdict, figures and docs/A2_IMEANFLOW_REPORT.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/a2_imeanflow_s5_ppgdalia_8s_seed42"
export PYTHONPATH="$ROOT/src" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
test -f "$OUT/TRAINING_DONE" || { echo "A2 not finished"; exit 1; }
"$ROOT/.venv/bin/python" scripts/eval_a2.py --out-dir "$OUT" --steps 1,2,4 --batch-size 64 --noise-seed 0 --diversity-seeds 4 --diversity-windows 256 --bench-repeats 10 2>&1 | grep -vE "Warning|warn|conv1d" | tee "$OUT/eval.log"
"$ROOT/.venv/bin/python" scripts/compare_a2.py 2>&1 | tee "$OUT/compare.log"
touch "$OUT/EVAL_DONE"
