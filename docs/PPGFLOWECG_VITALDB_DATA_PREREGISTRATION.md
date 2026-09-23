# PPGFlowECG × VitalDB — evaluation-data construction (preregistration)

Frozen and pushed before any window is built or any number is computed. Head at freeze: `489885c`.

## Why a new window set
The released PPGFlowECG model takes **10 s at 128 Hz (1,280 samples)** of PPG and generates 10 s of ECG (official
`main.py`: `sample_shift(..., [1280, 1], ...)`; VAE latent 4 × 40). The project's VitalDB V1 windows are 4 s, so they cannot
be fed to the model. A 10-s window set is built on **exactly the V1 test patients**, anchored on the V1 test windows, with a
rule fixed here.

## Window rule
- **Cases / patients:** the V1 test split only (`data/manifests/split_v1_vitaldb_seed42.json`, `test`: 1,224 cases,
  1,156 patients), in manifest order. No training or validation case is read.
- **Anchors:** every V1 test window (case, `window_index` k; 19,543 in total). The 10-s window is the raw span
  **[k·2000, k·2000 + 5000)** samples of `PLETH` and `ECG_II` at 500 Hz (it starts where the V1 window starts and extends
  6 s beyond it).
- **Exclusion (only these):** the span runs past the end of either channel; any non-finite sample in either channel; either
  channel constant over the span. Excluded anchors are counted and listed; nothing else is dropped, and no quality-based
  selection is applied (the official pipeline's flat / SQI screening is a *selection* step that would use the reference ECG;
  it is not used).
- Anchors whose 10-s spans overlap within a case are all kept; the number of overlapping pairs is reported.
- The rule is not revised after any result is seen.

## Preprocessing — the official PPGFlowECG transforms, executed verbatim
Functions taken verbatim from `external/PPGFlowECG` @ `56b2cd2` (`scripts/external/ppgflowecg/official.py`, source
sha256 recorded), run with the paper's pinned `mne 1.8.0` / `neurokit2 0.1.7` / `biosppy 2.2.3`, per window (the released
pipeline processes pre-segmented 10-s windows):
1. PPG: `ppg_clean_elgendi_mne(sr=500, 0.5 Hz, 8 Hz)` — Butterworth band-pass, order 3 (IIR, mne). ECG:
   `ecg_clean_nk_mne(sr=500, 0.5 Hz, None)` — Butterworth high-pass order 5 + 50 Hz notch. (VitalDB was recorded on 60 Hz
   mains; the official 50 Hz notch is kept as released.)
2. `data_resampler` (polyphase `resample_poly`) 500 → 128 Hz → 1,280 samples.
3. `data_normalizer`: per-window z-score.
4. step 2 `_batch_savgol`: Savitzky–Golay smoothing, PPG window 7 / order 2, ECG window 11 / order 2.

Not applied, stated now: the ECG polarity-inversion check of step 1 (it would change only the reference ECG, never the model
input; VitalDB `ECG_II` is a standard monitor lead) and all quality-selection steps.

**Model input** = processed PPG. **Reference ECG** = processed ECG — the same space as the model's training targets and as
the `gt` array the official evaluator receives. Its Hamilton HR is the reference functional Y\*.

## Output
`outputs/ppgflowecg_external/data/vitaldb10_test.npz` (gitignored): `ppg`, `ecg` [N, 1280] float32, `patient`, `caseid`,
`anchor_window_index`, `v1_order` (position of the anchor among the 19,543 V1 test windows). Build log and counts:
`artifacts/ppgflowecg_external/data_build.json`. Script `scripts/external/ppgflowecg/build_vitaldb10.py`.

Absolute numbers from this set are **not** comparable with the project's 4-s VitalDB results, nor with PPGFlowECG's
published tables.
