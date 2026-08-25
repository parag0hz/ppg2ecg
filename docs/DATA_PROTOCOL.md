# Data Protocol — PPG-DaLiA (v0)

## 1. Source and licence
- **PPG-DaLiA** (Reiss, Indlekofer, Schmidt, Van Laerhoven, 2019), UCI ML Repository #495.
- Licence: **CC BY 4.0** (verified on the UCI page 2026-08-25). No registration / DUA / manual consent required.
- Direct download (verified `HTTP 200`): `https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip`
  → `bash scripts/download_dalia.sh` (downloads, tests the zip, extracts to `data/raw/PPG-DaLiA/`, writes
  `data/raw/CHECKSUMS.sha256`). Raw data is git-ignored; only checksums and manifests are committed.
- Expected layout (what upstream `load_PPG_DaLiA` reads): `data/raw/PPG-DaLiA/PPG_FieldStudy/S{1..15}/S{n}.pkl`.
- Each pickle (`encoding="latin1"`): `signal.chest.ECG` (700 Hz), `signal.wrist.BVP` (64 Hz), plus ACC/EDA/TEMP/Resp,
  `activity` labels (8 activities + transient), `label` = reference HR per 8 s window with 2 s shift, `rpeaks` = reference
  R-peak sample indices at 700 Hz (from the dataset authors). 15 subjects, ~2 h 20 min each (~36 h total).

## 2. Preprocessing v0 = PENGUIN-faithful (`ppg2ecg.data.preprocess`, parity-tested against upstream)
| Step | PPG (input) | ECG (target) |
|---|---|---|
| Windowing | 4 s, non-overlapping, trailing < 4 s dropped (`sliding_window_view[::win]`) | same, on the 700 Hz stream |
| Resampling | FFT `scipy.signal.resample` → 512 samples (up from 256) | → 512 samples (down from 2800) |
| Filter | Butterworth order 4, **0.5–4 Hz band-pass**, `filtfilt` (per window) | Butterworth order 4, **0.5 Hz high-pass**, `filtfilt` |
| Normalisation | per-window z-score → per-window min-max to **[-1, 1]** | same |

Consequences that the evaluation must respect:
- Every statistic is **window-local**; there are no train-set statistics and no cross-subject statistics.
- The target ECG is scaled by *its own* min/max ⇒ absolute ECG amplitude (mV) is unrecoverable; all amplitude-based
  metrics (MAE/RMSE) live in normalised space. Rhythm/morphology metrics (R-peaks, HR, RR, QRS width, template
  correlation) are scale-free and are therefore the primary outcomes.
- Zero-phase filtering on isolated 4 s windows has edge transients (0.5 Hz high-pass ↔ 2 s time constant); we keep it
  for faithfulness and note it as a known artefact (evaluation windows discard nothing; upstream doesn't either).
- PPG and ECG windows are index-aligned assuming a common t = 0 (the dataset is synchronised). Upstream never checks that
  the two streams yield the same number of windows; `windows_for_subject(..., align="truncate")` keeps
  `min(n_ppg, n_ecg)` and `scripts/verify_dalia.py` reports any mismatch per subject.

## 3. Splits (all subject-level; `ppg2ecg.data.splits`, manifests in `data/manifests/`)
| Protocol | Train / Val / Test | Purpose |
|---|---|---|
| **Upstream** | 13 / 1 / 1 subjects, `random.sample(glob(...))` after `random.seed(42)` — depends on directory-listing order | reproduction arm A0 only; actual subjects recorded post-hoc with `scripts/upstream_split_probe.py` |
| **P0-holdout** | 13 / 1 / 1, subjects chosen from the *sorted* list with `random.Random(42)` | our deterministic analogue of upstream |
| **P1-kfold5** | 5 folds × (10 / 2 / 3); every subject is a test subject exactly once | main claims (NFE curve, one-step arms), reported as mean ± SD over folds/subjects |

## 4. Leakage checks (must pass before any number is reported; `scripts/run_leakage_checks.py`, `tests/test_splits_leakage.py`)
1. `train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`, and every subject assigned exactly once.
2. No processed window hash (float32, 6 decimals) appears in two splits.
3. Preprocessing is window-local: output for window *i* is unchanged when every other window is replaced by noise.
4. Inference receives only PPG: `sample(ppg)` signature has no target argument; behavioural check that the output is
   invariant to anything but `ppg` and the noise seed. Target normalisation is applied only to build the reference
   for metrics, never to the input path.
5. Consecutive-window concatenation (8 s HR windows) never crosses a subject boundary (windows are grouped per subject).

## 5. Data status (2026-08-25)
- No local copy existed anywhere under `/home/kwy00` (searched read-only).
- Downloaded from the UCI URL (2,865,111,320 bytes; `unzip -t` clean; sha256 in `data/raw/CHECKSUMS.sha256`), extracted to
  `data/raw/PPG-DaLiA/PPG_FieldStudy/S{1..15}/S{n}.pkl` (+ `readme.pdf`, ~3.0 GB on disk). 15/15 subjects present.
- `scripts/verify_dalia.py` (→ `data/manifests/dalia_raw_inventory.json`): ECG and BVP durations agree to the second for
  every subject (ECG 700 Hz, BVP 64 Hz), so PPG/ECG window counts match for all 15 subjects (no truncation needed).
  Per-subject 4 s windows: 1312 (S6) … 2662 (S10); **total 32,368 windows** (≈ 36.0 h). Dataset `rpeaks` and 8 s HR labels exist.
- Upstream `preprocess.yaml` says `sample_num: 16181`, `duration: 35.96` for DaLiA: 16181 = Σ_subjects ⌊n_4s/2⌋ **exactly**,
  i.e. the bookkeeping was done with **8 s segments** (same for WildPPG: 390216 × 8 s = 867.1 h). The shipped global
  `segment_len: 4` therefore disagrees with the per-dataset bookkeeping — flagged in `docs/PENGUIN_AUDIT.md`
  (paper text decides which window length the reported numbers used). v0 keeps the shipped 4 s.
- Official upstream preprocessing ran end-to-end (`scripts/run_upstream_preprocess.sh`, 39 s) →
  `data/processed/upstream/PPG-DaLiA/subject{0..14}.pkl` (subject{i} = S{i+1}); upstream tree stays clean.
- `scripts/check_processed_parity.py`: our `ppg2ecg.data` pipeline vs upstream-written arrays — see docs/EXPERIMENT_LOG.md.
