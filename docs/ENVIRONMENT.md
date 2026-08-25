# Environment Audit

Audited 2026-08-25 14:26 KST on the workstation that will run all experiments. Re-verify any time with
`.venv/bin/python scripts/check_env.py`.

## Hardware

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA GeForce **RTX 5090** (Blackwell, compute capability **12.0 / sm_120**) |
| VRAM | **32,607 MiB** (33.7 GB); 173 MiB used by Xorg/gnome-shell at idle |
| GPU power / state at audit | 32 W / 575 W cap, P8, 55 °C |
| CPU | Intel Core **i5-14600K**, 20 logical CPUs (1 socket, 2 threads/core) |
| RAM | **62 GiB** total, ~43 GiB available at audit (2 GiB swap) |
| Disk | `/` (nvme0n1p2) 3.6 TB, **1.2 TB free** (66 % used) — `/home/kwy00` and `/tmp` are on the same filesystem |

## Software

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-136-generic |
| NVIDIA driver | **580.173.02** (reports CUDA 13.0) |
| CUDA toolkit | `nvcc` **not on PATH**, no `/usr/local/cuda*`; CUDA runtime comes from pip (`nvidia-cuda-runtime 13.0.96`, `cuda-toolkit 13.0.2`) — enough for PyTorch, not for building custom kernels |
| Python (project) | **3.13.9** (`/home/kwy00/anaconda3/bin/python3`, conda `base`) → project venv `.venv` created with `--system-site-packages` |
| PyTorch | **2.11.0+cu130**, cuDNN 91900, `torch.cuda.is_available() = True`, `torch.compile` available |
| BF16 | `torch.cuda.is_bf16_supported() = True`; bf16 matmul verified finite. TF32 matmul is **off** by default (`allow_tf32=False`) — leave off for reproducibility unless pre-registered |
| Alternative env | conda `ecg` (Python 3.10.20, torch 2.11.0+cu128) — **not used** (PENGUIN requires Python ≥ 3.12); `/home/kwy00/sci/` is read-only for this project |
| git | 2.34.1 |
| git-lfs | **not installed** (not needed: raw data is git-ignored; upstream checkout has no LFS objects) |
| Compiler | gcc / g++ 11.4.0; `cmake`, `ninja` **absent**; `uv` **absent** (upstream README uses `uv sync`; we use pip in a venv instead) |

## Project virtual environment (`.venv`)

Created: `python3 -m venv --system-site-packages .venv` (inherits torch 2.11.0+cu130, numpy 2.3.5, scipy 1.16.3,
hydra-core 1.3.5, omegaconf 2.3.1, matplotlib 3.10.6, pandas 2.3.3, scikit-learn 1.7.2, h5py 3.15.1 from conda base),
then `pip install neurokit2==0.2.12 biosppy peakutils thop wandb einops pytest ruff` and `pip install -e .`.

| Package | Installed | PENGUIN `uv.lock` pin | Note |
|---|---|---|---|
| torch | 2.11.0+cu130 | 2.8.0 | newer; sm_120 requires ≥ 2.7/cu128. Upstream S5 (`torch.jit.script`, `torch.vmap`, complex64) runs — verified by `scripts/smoke_test_penguin.py` |
| numpy | 2.3.5 | 2.3.3 | |
| scipy | 1.16.3 | 1.16.2 | `signal.resample`, `butter`, `filtfilt`, `stats.zscore` — same major/minor |
| neurokit2 | 0.2.12 | 0.2.12 | exact pin (used in upstream HR error) |
| biosppy | 2.2.4 (`__version__` reports 2.1.2) | 2.2.3 | needed `peakutils` installed manually (missing transitive dep) |
| hydra-core / omegaconf | 1.3.5 / 2.3.1 | 1.3.2 / 2.3.0 | |
| thop | 0.1.1.post2209072238 | same | only for upstream `summarize()` |
| wandb | 0.28.2 | 0.21.4 | imported at top of upstream `train.py` even when disabled |
| einops, pytest, ruff | 0.8.2 / 8.4.2 / – | – | ours only |
| jax[cpu] | 0.11.1 | – | optional; only `tests/test_imeanflow.py::test_parity_with_official_jax_objective` (skipped if absent) |

Known differences vs upstream's locked environment: Python 3.13 (upstream ≥ 3.12), torch 2.11 vs 2.8. Numerics of
FFT resampling / filtfilt are scipy-version dependent at the 1e-12 level only.

## Known environment hazards
- **Stale `LD_LIBRARY_PATH`** in the login profile: `/home/kwy00/anaconda3/lib/python3.10/site-packages/torch/lib:/usr/local/cuda-11.8/lib64`
  — neither directory exists (left over from an older env). Harmless, but remove it if a future CUDA library mix-up appears.
- **numpy (MKL 2025) eigensolver after `import torch`**: pip torch 2.11 bundles its own `libgomp.so.1` (GOMP_5.0) while Anaconda's
  libgomp is GOMP_6.0 and both get mapped. One independent auditor observed `np.linalg.eigh` segfaulting (exit 139) after
  `import torch` in a clean shell — PENGUIN's S5 init calls exactly that (`S5_init.py:63`). In the project shell it does **not**
  reproduce (6 variants tried on 2026-08-25: default env, `env -i`, `LD_LIBRARY_PATH` unset, scipy/neurokit2 import orders, with/without
  `MKL_THREADING_LAYER=SEQUENTIAL` — all exit 0; the full upstream import chain builds the 4.57 M-param model fine). Because the
  failure mode is environment-sensitive, `scripts/run_upstream_*.sh` run a probe first and export `MKL_THREADING_LAYER=SEQUENTIAL`
  only if the probe fails (affects CPU numpy threading only; GPU numerics untouched). Alternative mitigations verified by the auditor:
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `LD_PRELOAD=$CONDA_PREFIX/lib/libgomp.so.1`, or one `np.linalg.eigh` call before `import torch`.
- **UPDATE 2026-08-25 19:00 — the segfault became deterministic**: `import torch` followed by a threaded complex 256×256
  `numpy.linalg.eigh` (exactly PENGUIN's S5 init) crashed on every attempt (real matrices and small complex ones were fine), even
  though identical model builds had worked minutes earlier in the same venv. Root cause: torch's bundled `libgomp.so.1` (GOMP_5.0)
  is mapped before MKL initialises its GNU-OpenMP threading layer. **Fix in place**: every entry point (training loops, preflight, evaluation,
  `tests/conftest.py`) does `import ppg2ecg.utils.mkl_warmup` **before** `import torch`; the module runs `numpy.linalg.eigh(np.eye(2))`
  at import time, which initialises MKL's threading layer first. Verified to remove the crash while keeping threaded-MKL numerics.
  (A venv `sitecustomize.py` does not work: Anaconda's own `lib/python3.13/sitecustomize.py` shadows it.) `MKL_THREADING_LAYER=SEQUENTIAL` and
  `MKL_NUM_THREADS=1` also work (different MKL execution path); `LD_PRELOAD` of Anaconda's libgomp does **not**.
- `ruff` inside `.venv` resolves to the conda-base package (`--system-site-packages`); use `/home/kwy00/anaconda3/bin/ruff`.

## Determinism notes
- Upstream `fix_seed` sets `torch.use_deterministic_algorithms = True` as an **attribute assignment** (no effect); our
  `ppg2ecg.utils.seed.seed_everything` calls the function (`warn_only=True`) and sets `CUBLAS_WORKSPACE_CONFIG`.
- The associative scan in S5 uses `torch.vmap` + custom reductions; bit-level run-to-run determinism on GPU is
  **not guaranteed** and must be measured (planned: two identical seeds → compare checkpoints).

## Capacity estimate for this project (from `scripts/smoke_test_penguin.py`, see docs/EXPERIMENT_LOG.md for numbers)
- PENGUIN shipped config: ~few-million parameters, batch 64 × 512 samples → training step peak memory well below 4 GB;
  the 25-step Heun validation pass (50 NFE) dominates wall-clock per epoch, not memory.
- The full NFE sweep + 3 seeds fits on this single GPU; no multi-GPU or mixed precision is needed (fp32 throughout, prereg §4).
