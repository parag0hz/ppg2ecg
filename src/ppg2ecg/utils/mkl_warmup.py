"""Import this module BEFORE `import torch` (it runs a numpy LAPACK call at import time).

Why: on this machine torch's bundled libgomp (GOMP_5.0) gets mapped before Anaconda MKL initialises its GNU-OpenMP threading
layer, after which a threaded complex `numpy.linalg.eigh` — exactly PENGUIN's S5 init (256x256 Hermitian) — segfaults.
Initialising MKL first (any LAPACK call) removes the crash and keeps threaded-MKL numerics unchanged. A venv `sitecustomize.py`
cannot be used because Anaconda ships its own `lib/python3.13/sitecustomize.py` that shadows it. See docs/ENVIRONMENT.md.
Usage (entry points, tests/conftest.py):   import ppg2ecg.utils.mkl_warmup  # noqa: F401   (before `import torch`)
"""
import sys as _sys

WARMED_BEFORE_TORCH = "torch" not in _sys.modules
if WARMED_BEFORE_TORCH:
    import numpy as _np

    _np.linalg.eigh(_np.eye(2))
