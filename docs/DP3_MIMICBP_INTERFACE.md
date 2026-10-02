# DP3 — MIMIC-BP interface (frozen before any MIMIC-BP waveform value is read)

**Evidence label:** ECG-target-blind cross-task external cohort.
"MIMIC-BP PPG had previously been used in unrelated PPG-to-ABP experiments, whereas ECG reconstruction targets were
withheld from the PPG-to-ECG architecture-development process."

Records:
- `artifacts/dp3_mimicbp/dataset_schema.json`, `dataset_adapter.json`, `external_subject_manifest.json` and
  `external_exclusion_rules.json`, written by `scripts/dp3_mimicbp.py interface`. That stage reads file names, `.npy`
  headers (mmap shapes / dtypes) and loader text only.
- The per-window exclusion log `external_exclusion_log.csv` is written only during the frozen evaluation.

## Schema (metadata)

| item | value |
|---|---|
| path | `data/raw/MIMIC-BP` (MIMIC-III Waveform Matched Subset curation, Dataverse v2.2) |
| subjects | 1,524 (`p<ID>`; numeric part used as the integer patient id) |
| files | ppg, ecg, abp, resp: 1,524 each, shape (30, 3750) float64; labels (30, 2) |
| sampling rate | 125 Hz (`read_data.py`: `fs = 125`) |
| segments | 30 × 30 s per subject; PPG and ECG of a segment share sample indices |
| synchronisation | as curated by the dataset authors (ECG / PPG fundamental frequency within 0.3 Hz, pulse-arrival-time consistency) |
| official split lists | train 1,100 / val 195 / test 229 (not used: DP3 evaluates all subjects as one cohort) |

## Frozen interface

- **Channels:** the single `ppg` and the single `ecg` file per subject. The ECG lead identity is not given locally
  (UNKNOWN). The DP0 models were trained on VitalDB ECG_II, so a lead mismatch is part of the domain shift. ABP and
  RESP are not read.
- **Windows:**
  - 4 s = 500 samples at 125 Hz, non-overlapping, starting at t = 0 of each 30 s segment;
  - 7 windows per segment, the last 2 s dropped;
  - window id = subject rank · 210 + segment · 7 + k.
- **Preprocessing** (the V1 / DP0 implementation, `ppg2ecg.data.preprocess.preprocess_windows`, unchanged since
  `a15b354`):
  - FFT resampling 500 → 512 samples (128 Hz);
  - PPG band-pass 0.5–4 Hz; ECG high-pass 0.5 Hz (4th-order Butterworth, zero-phase);
  - per-window z-score, then min-max to [−1, 1]. Nothing is fitted.
- **Alignment:** none. There is no shift, no pulse-transit compensation and no offset search.
- **Exclusions:**
  - R1: unreadable / wrong shape (subject);
  - R2: raw non-finite or constant window;
  - R3: non-finite after preprocessing;
  - R4: neurokit HR of the preprocessed ECG window outside [30, 200] bpm or not finite (the frozen V1 / DP0 rule);
  - R5: subject with no remaining window.
- **Reference R:** neurokit on the preprocessed ECG window.
- **Events:** the frozen DP0 detector, threshold 0.35, refractory 32, σ 20 ms raster.
- **Noise:** `scaleflow.window_noise(int pid, window id)`, the same for G and S1.
- **K16 subset:** salted rank `dp3-k16-v1` over the eligible window ids, first 2,000.
- **PPG shuffle:** `default_rng(20261002).permutation` over the eligible windows (PPG only).
