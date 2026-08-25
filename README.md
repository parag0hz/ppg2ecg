# ppg2ecg-one-step

> **Can PPG-conditioned ECG reconstruction be reduced to one-step generation without sacrificing clinically
> meaningful ECG morphology and conditional fidelity?**

Independent research repository (started 2026-08-25). Baseline: **PENGUIN** (OT-CFM + Flow-SSM/S5, Neurogica 2025),
vendored *unmodified* under `external/PENGUIN/` and imported in place.

## Plan (see `docs/`)
1. Reproduce PENGUIN on PPG-DaLiA (arm A0) — `docs/PENGUIN_AUDIT.md`, `configs/upstream/`.
2. Inference-only **NFE curve** on the same checkpoint: Euler {25,10,5,2,1} NFE, Heun {50,20,10,4,2} NFE (arm A1).
3. Decide from pre-registered margins whether 1-NFE fails on morphology or conditioning — `docs/PREREGISTRATION_V0.md`.
4. Only then: swap the objective (OT-CFM → iMeanFlow) on the identical backbone (arm A2).

Principles: strong baseline first; one factor at a time; metrics and thresholds documented before results;
strict subject/normalisation leakage checks; no code copied from other projects without explicit approval.

## Layout
```
configs/            baseline record, NFE sweep, smoke; configs/upstream = patched copy of upstream Hydra configs
data/{raw,processed,manifests}   raw/processed are git-ignored; manifests (subject splits, inventories) are committed
docs/               RESEARCH_QUESTION, PREREGISTRATION_V0, ENVIRONMENT, DATA_PROTOCOL, PENGUIN_AUDIT, EXPERIMENT_LOG
external/PENGUIN/   upstream checkout @ 6cd70cd (read-only)
src/ppg2ecg/        data · flow · evaluation · models · training · utils
scripts/            check_env, download_dalia, verify_dalia, make_split_manifest, run_leakage_checks,
                    smoke_test_penguin, upstream_split_probe, run_upstream_{preprocess,train}.sh
tests/              unit tests incl. bit-exact parity with upstream (preprocess, Heun sampler, CFM targets)
outputs/ artifacts/ run outputs (git-ignored)
```

## Quick start
```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[dev]" peakutils
.venv/bin/python scripts/check_env.py
bash scripts/download_dalia.sh && .venv/bin/python scripts/verify_dalia.py
.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_test_penguin.py --full
.venv/bin/python scripts/make_split_manifest.py && .venv/bin/python scripts/run_leakage_checks.py
# baseline reproduction (arm A0) — not run in session 0:
bash scripts/run_upstream_preprocess.sh && bash scripts/run_upstream_train.sh
```
