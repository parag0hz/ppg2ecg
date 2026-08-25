#!/usr/bin/env bash
# Official PENGUIN training on PPG-DaLiA (arm A0 of docs/PREREGISTRATION_V0.md). NOT run in the setup session.
# Upstream code UNMODIFIED; patched config from configs/upstream; logs+ckpt -> outputs/upstream/.
# Usage: bash scripts/run_upstream_train.sh [extra hydra overrides, e.g. seed=43 train.logging.description=seed43]
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
"$ROOT/.venv/bin/python" train.py --config-path "$ROOT/configs/upstream" --config-name config.yaml train.dataset=PPG-DaLiA train.model=PENGUIN hydra.run.dir="$ROOT/outputs/hydra/${HYDRA_JOB:-train}_$(date +%Y%m%d_%H%M%S)" hydra.output_subdir=.hydra "$@"
