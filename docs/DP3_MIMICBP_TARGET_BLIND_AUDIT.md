# DP3 — Stage A: MIMIC-BP ECG-target-use audit (and Stage B integrity)

> **TARGET-BLIND AUDIT: PASS.** No MIMIC-BP ECG waveform or ECG-derived target was ever used in this project.
> - No such data were used for training, model / checkpoint selection, architecture or loss design, threshold selection,
>   reconstruction performance, ECG morphology or ECG-derived event analysis, or PPG → ECG hypothesis formation.
> - **Evidence label:** "ECG-target-blind cross-task external cohort".
> - **Required disclosure:** "MIMIC-BP PPG had previously been used in unrelated PPG-to-ABP experiments, whereas ECG
>   reconstruction targets were withheld from the PPG-to-ECG architecture-development process."
> - MIMIC-BP is **not** project-naive, fully fresh, completely unseen or independent of all prior research activity.
>
> Records:
> - `artifacts/dp3_mimicbp/target_blind_audit.json`, produced by `scripts/dp3_mimicbp.py target_audit`;
> - `artifacts/dp3_mimicbp/s1_integrity_audit.json`, produced by `scripts/dp3_mimicbp.py integrity`.

## 1. Method (metadata only; no MIMIC-BP waveform value was read)

- **Text scan:** every text file under `docs/`, `scripts/`, `src/`, `artifacts/`, `data/manifests/`, `tests/` and
  `external/PENGUIN/{src,config}`.
  - 116 files mention MIMIC-BP (pattern `mimic[-_ ]?bp`).
  - Every line that also contains an ECG-target term (ecg, PPG-to-ECG, reconstruct, morphology, R-peak) was listed
    and adjudicated.
  - The DP2 / DP3 audit files themselves are excluded from the scan.
- **Code paths:**
  - which MIMIC-BP signal kinds the project loader (`src/ppg2ecg/data/mimicbp.py`) and the upstream PENGUIN loader
    (`external/PENGUIN/src/utils/load_data.py`, read only) open;
  - the U2 build target (`scripts/u2_build.py`);
  - any Python line that loads `MIMIC-BP/ecg` or `*_ecg.npy`.
- **Processed files:** the array names (zip listing, no values) of `data/processed/mimicbp_8s` and
  `data/processed/u2_mimicbp`.
- **Outputs and history:** the names of every output path containing "mimic", any file under them with "ecg" in its
  name, and every git commit message mentioning MIMIC.

## 2. Findings

| check | result |
|---|---|
| project loader kinds | ppg, abp, labels |
| upstream PENGUIN loader kinds | ppg, abp |
| U2 MIMIC-BP target | ABP |
| Python lines loading MIMIC-BP ECG arrays | none |
| processed MIMIC-BP array names | `x`, `y` (PPG / ABP), `pid`, `segment_idx`, `window_start_s`, `label_sbp`, `label_dbp` / `x`, `y`, `subject`, `window_index` |
| output files with "ecg" under MIMIC outputs | none |
| same-line MIMIC-BP + ECG-term co-mentions | 18 lines, all adjudicated as non-use |

The co-mentions fall into four kinds:
- the dataset's channel listing (A7 audit, technical spec, loader docstring, DP2 inventory);
- the explicit exclusion from ECG work: D1 "**EXCLUDED** — target is ABP, not ECG", and predict_a7 "no ECG metrics";
- ABP-only preprocessing branches;
- figure captions "MIMIC-BP (ABP)".

**Prior PPG / ABP use (disclosed):**
- `84223f0` / `fed5b8c` A7: PPG → ABP on the official split (train 1,100 / val 195 / test 229);
- `9ff77b5` A8: ABP target scale;
- `2842c11` U1 and `e09ee6d` U2: the ABP task;
- `eddbe60` / `b1eb66a` EXP-D: ABP width / depth.
- A7's ABP results were also contrasted with ECG findings on other datasets in program summaries ("이 결과는 ECG에
  특이적이다. PPG→ABP(MIMIC-BP)에서는 …"). That contrast used MIMIC-BP **ABP** outcomes only.

## 3. Stage B — frozen S1 integrity (PASS)

- **Accounting:** total 943,372; shared 246,720 (stem + blocks 1–6, dilations 1, 2, 4, 8, 16, 32); adapters 8,320;
  private blocks 7–8 per task; widths 71 / 30. The separate specialists have 1,191,910, so the saving is 20.85 %.
- **Checkpoints and freeze:** the detector / P / G / S1 seed-42 checkpoints match the DP0 hashes, and the DP0
  lock-freeze manifest is intact.
- **Detector rule:** threshold 0.35, refractory 32, σ 20 ms.
- **Preprocessing:** `src/ppg2ecg/data/preprocess.py` and `src/ppg2ecg/evaluation/rpeaks.py` are unchanged since
  `a15b354`, which predates the V1 corpus build.
