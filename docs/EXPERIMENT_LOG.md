# Experiment Log

Append-only. Every entry: date, what was run (command + commit), where outputs are, what was concluded.

## 2026-08-25 — Session 0: workspace, audit, baseline-readiness (no training)
- Created `/home/kwy00/ppg2ecg-one-step/` as an independent git repo (nested under the unrelated `/home/kwy00` repo; ignored there).
- Environment audit → `docs/ENVIRONMENT.md` (RTX 5090 32 GB, torch 2.11.0+cu130, Python 3.13, bf16 OK).
- Cloned upstream `https://github.com/Neurogica/PENGUIN` @ `6cd70cd` (main, clean) → `external/PENGUIN/` (never edited).
- Source audit → `docs/PENGUIN_AUDIT.md`. Headline: shipped config **cannot run on PPG-DaLiA as-is** (`DaLiA` vs
  `PPG-DaLiA` key), "25 steps" = **50 NFE** (Heun), no EMA/no scheduler despite config, dead `cross_attn`, glob-order split.
- Patched config copy → `configs/upstream/` (3 patches, documented); Hydra compose verified.
- PPG-DaLiA: not present locally; UCI direct link (CC BY 4.0) verified; download started to `data/raw/ppg+dalia.zip`.
- Code written: `ppg2ecg.data` (loader, PENGUIN-faithful preprocessing, deterministic splits, leakage checks),
  `ppg2ecg.flow` (CFM re-statement, Euler/Heun samplers with NFE counting), `ppg2ecg.evaluation` (metrics v0, efficiency),
  `ppg2ecg.models` (upstream import shim). Tests in `tests/`; smoke test `scripts/smoke_test_penguin.py`.
- Manifests: `data/manifests/split_p0_holdout_seed42.json`, `split_p1_kfold5_seed42.json`; leakage checks PASS (subject level).
- Results of tests / smoke test: see the "Session 0 results" entry below (appended after they ran).

### Session 0 results (2026-08-25, no training)
| Check | Result | Evidence |
|---|---|---|
| Hydra compose of shipped upstream config for PPG-DaLiA | **fails** (`ConfigAttributeError: Key 'PPG-DaLiA' is not in struct`) | dry-run with `hydra.compose`; fixed in `configs/upstream/preprocess.yaml` |
| Upstream preprocessing end-to-end (`scripts/run_upstream_preprocess.sh`) | OK, 39 s, 15 files, upstream tree clean | `data/processed/upstream/PPG-DaLiA/subject{0..14}.pkl` |
| Our pipeline vs upstream processed arrays (S1, S6, S15) | **bit-exact** (max abs diff 0.0, x and y) | `scripts/check_processed_parity.py` → `data/manifests/processed_parity_upstream.json` |
| Unit tests (`tests/`, 16 tests incl. preprocess / Heun / CFM parity with upstream) | **16 passed** | `PYTHONPATH=src .venv/bin/python -m pytest` |
| Leakage checks P0 + P1 (subject-level) | PASS ×6 | `scripts/run_leakage_checks.py` |
| Window-level disjointness on real processed data (P0) | PASS (train 28055 / val 2262 / test 2051 unique windows, 0 overlaps) | `check_processed_parity.py` |
| Upstream split on this filesystem | val = **S4**, test = **S10**, train = others | `scripts/upstream_split_probe.py` → `data/manifests/split_upstream_probe_thisfs.json` |
| Upstream model on torch 2.11 / RTX 5090 (`scripts/smoke_test_penguin.py --full`) | imports, trains, samples | `outputs/smoke/penguin_smoke_2026-08-25T144430.json` |
| Parameter count (shipped DaLiA config) | 4,568,707 total; 264,192 in never-called `cross_attn` (+2 in unused `revin`) → 4,304,513 effective | smoke JSON |
| Training step, batch 64 × 512, fp32 | loss 1.99 (finite), **peak 9,175 MiB**, ~140–300 ms/step (first steps) | smoke JSON |
| Gradient flow at init | step 1 updates only `final_layer.linear` (2 tensors), step 2: 16/161 tensors — adaLN-Zero cascade, not a bug | smoke JSON |
| Sampler parity | our `heun_sample` == upstream `.sample()` **bit-exact**, NFE 50 for 25 steps | smoke JSON, `tests/test_upstream_parity.py` |
| Upstream `HeartRateError` with `segment_len=4` | **2× time compression confirmed** (true 60 → 119.7 bpm, 90 → 179.6, 120 → no beats → error 0.0); correct with `segment_len=8` | in-session probe; pinned by `tests/test_upstream_parity.py` |
| Inference cost (batch 64, fp32, median of 5) | Heun 25 steps/50 NFE: **1,865 ms** (34 samp/s); Euler 1 NFE: **38 ms** (1,703 samp/s); peak 958 MiB | smoke JSON (random weights; latency only) |

Cost implications for arm A0: train ≈ 28k windows/64 ≈ 440 steps/epoch (~1 min) + val (1 subject, ~2.3k windows,
50 NFE) ≈ 36 batches × 1.87 s ≈ 70 s ⇒ **≈ 2–2.5 min/epoch**, ≤ 300 epochs ⇒ worst case ~12 h/seed, typical with
patience-10 early stopping ~2–5 h/seed. GPU memory < 10 GB ⇒ two seeds could run concurrently on the 32 GB card.

Deviations from plan: none. Things done beyond the minimum: dataset downloaded (public, CC BY 4.0, 2.9 GB) and upstream
preprocessing executed (39 s) so that leakage/parity checks run on real data — no model training was started.

## 2026-08-25 — Session 1: GitHub connection + A0 (PENGUIN, PPG-DaLiA, 8 s, seed 42)
- Decisions frozen before training (see PREREGISTRATION_V0 §6/§9): **8 s windows**, deterministic P0 split (test S2, val S11), seed 42 only.
- Built `data/processed/v0_8s/` (ours, 16,181 windows, per-file sha256 in `MANIFEST.json`) and `data/processed/upstream_8s/`
  (unmodified upstream `preprocess.py` with `preprocess.segment_len=8`) for parity checking.
- `external/PENGUIN` registered as a git **submodule** pinned at `6cd70cd` (existing clone reused; nothing deleted).
