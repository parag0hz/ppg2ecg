"""Access the *unmodified* upstream PENGUIN checkout (external/PENGUIN) without copying code.

Policy (docs/PENGUIN_AUDIT.md): upstream is imported in place via sys.path; no file under
external/PENGUIN is ever edited. `assert_upstream_pinned()` fails loudly if the checkout drifts.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPO_ROOT / "external" / "PENGUIN"
UPSTREAM_SRC = UPSTREAM_ROOT / "src"
UPSTREAM_COMMIT = "6cd70cdefb91f10efeb8dce34019b5067cb25344"  # main @ 2025-09-18, cloned 2026-08-25


def upstream_git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(UPSTREAM_ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()],
    }


def assert_upstream_pinned() -> dict:
    st = upstream_git_state()
    if st["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream PENGUIN commit drifted: {st['commit']} != {UPSTREAM_COMMIT}")
    if st["dirty_files"]:
        raise RuntimeError(f"upstream PENGUIN checkout is dirty: {st['dirty_files']}")
    return st


def add_upstream_to_path() -> Path:
    if not UPSTREAM_SRC.exists():
        raise FileNotFoundError(f"upstream src not found: {UPSTREAM_SRC} (run: git clone https://github.com/Neurogica/PENGUIN external/PENGUIN)")
    sys.dont_write_bytecode = True  # never leave __pycache__ inside the read-only upstream checkout
    p = str(UPSTREAM_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
    return UPSTREAM_SRC


def import_upstream_penguin():
    """Return the upstream `PENGUIN` nn.Module class (src/models/PENGUIN.py)."""
    add_upstream_to_path()
    from models.PENGUIN import PENGUIN  # noqa: PLC0415  (upstream module)

    return PENGUIN


def import_upstream_preprocess():
    """Return upstream `preprocess(trial_data, cfg, bandpass, freq_range, zscore, normalize)` (src/preprocess.py)."""
    add_upstream_to_path()
    from preprocess import preprocess  # noqa: PLC0415  (upstream module)

    return preprocess


def import_upstream_compute_metrics():
    """Return upstream `compute_metrics(pred, target, task_metric, cfg)` (src/utils/help_func.py) for HeartRateError parity."""
    add_upstream_to_path()
    from utils.help_func import compute_metrics  # noqa: PLC0415  (upstream module)

    return compute_metrics
