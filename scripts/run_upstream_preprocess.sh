#!/usr/bin/env bash
# Official PENGUIN preprocessing for PPG-DaLiA, upstream code UNMODIFIED, patched config from configs/upstream.
# Expects data/raw/PPG-DaLiA/PPG_FieldStudy/S{1..15}/S{n}.pkl  (scripts/download_dalia.sh)
set -euo pipefail
# hydra.run.dir is redirected into outputs/ so Hydra never writes into external/PENGUIN (keeps upstream clean)
export PYTHONDONTWRITEBYTECODE=1   # keep external/PENGUIN free of __pycache__
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Preflight: on this box numpy(MKL).eigh after `import torch` can segfault in a clean environment (see docs/ENVIRONMENT.md);
# PENGUIN's S5 init calls np.linalg.eigh. If the probe fails, fall back to sequential MKL threading (no effect on GPU numerics).
if ! "$ROOT/.venv/bin/python" -c "import torch, numpy as np; np.linalg.eigh(np.eye(4)*-1j)" >/dev/null 2>&1; then
  echo "[preflight] numpy eigh after torch import failed -> exporting MKL_THREADING_LAYER=SEQUENTIAL" >&2
  export MKL_THREADING_LAYER=SEQUENTIAL
fi
cd "$ROOT/external/PENGUIN/src"
"$ROOT/.venv/bin/python" preprocess.py --config-path "$ROOT/configs/upstream" --config-name preprocess.yaml preprocess.dataset=PPG-DaLiA hydra.run.dir="$ROOT/outputs/hydra/${HYDRA_JOB:-preprocess}_$(date +%Y%m%d_%H%M%S)" hydra.output_subdir=.hydra "$@"
