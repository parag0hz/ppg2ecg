# A7 — ABP dataset audit (written 2026-08-27, BEFORE any A7 training)

Purpose: choose, by the pre-stated rules (A7 spec §14), the PPG→ABP dataset for the cross-target generalisation test, and document its
properties, PENGUIN's preprocessing, and leakage risks. Sources of truth: the pinned PENGUIN code (`external/PENGUIN` @ `6cd70cd`:
`config/preprocess.yaml`, `src/utils/load_data.py`, `src/preprocess.py`, `src/utils/help_func.py`), the PENGUIN paper (arXiv:2602.03858
§4.1–4.3), the MIMIC-BP paper (Sanches et al., *Scientific Data* 11:1233, 2024, PMC11568151) and the Harvard Dataverse record.

## 1. Candidates supported by PENGUIN
| | UCI-BP (Kachuee et al. 2015) | MIMIC-BP (Sanches et al. 2024) |
|---|---|---|
| Official source | UCI ML Repository #340 (`Part_1..4.mat`, MATLAB v7.3) | Harvard Dataverse doi:10.7910/DVN/DBM1NF, **v2.2 (2024-11-21)** |
| License | CC BY 4.0 | **ODbL 1.0** (Dataverse record); the paper itself is CC BY-NC-ND 4.0 |
| Access | open download | open download via the Dataverse API (verified: unauthenticated GET, md5-checked); the record flags "file access request" but no file is restricted |
| Raw size | 3.1 GB | **878 MB** (9 files; ppg 276 MB, abp 245 MB, ecg 156 MB, resp 243 MB, labels 0.7 MB) |
| Records / subjects | 12,000 records from MIMIC-II, **no subject identifiers** (literature: "suppressing the patient identification information … data leakage becomes unavoidable"); PENGUIN's `subject_num: 8` are 8 *pseudo-subjects* (Part_k halves), not people | **1,524 de-identified subjects** with unique IDs `p<ID>` (MIMIC-III Waveform Matched Subset), 30 disjoint 30 s segments per subject (380 h) |
| Signals | PPG, ABP (mmHg), ECG @125 Hz | PPG (raw ADC units 0–4), **ABP mmHg**, ECG, RESP @125 Hz, raw ("stored exactly as read by wfdb") |
| Official split | none | **train 1,100 / val 195 / test 229 subjects** shipped (`*_subjects.txt`), subject-disjoint |
| Synchronisation | as in MIMIC-II | curated: ECG/PPG fundamental-frequency match within 0.3 Hz and pulse-arrival-time consistency per segment; ABP bounds (SBP ≥ 60, DBP ≤ 120, 30–200 mmHg, PP ≤ 100 and ≥ SBP/4), saturation (> 0.04 s constant) removed |
| Labels | none | `p<ID>_labels.npy` [30, 2] = median per-beat SBP/DBP per segment (verified on p000188 seg 0: label 90.5/69.6 vs median peak/trough 90.5/69.2) |

## 2. Selection (rule §14, decided before any result)
Priority 1 (paired PPG + continuous ABP + **reliable subject identity** for subject-disjoint splits): MIMIC-BP ✓, UCI-BP ✗ (no IDs;
a window-level or pseudo-subject split would be leakage-prone and is forbidden by the rule). Priority 2 (PENGUIN preprocessing
reproducible): both ✓. Priority 3 (physical mmHg at evaluation): both ✓ (ABP stored raw). **→ MIMIC-BP is selected; UCI-BP is not used.**

## 3. Local raw data (`data/raw/MIMIC-BP/`, `CHECKSUMS.sha256`)
All 9 Dataverse files downloaded 2026-08-27 and md5-verified against the Dataverse metadata (all OK). Extracted: 1,524 × 5 npy files;
waveforms `[30, 3750]` float64, labels `[30, 2]`; **no NaN/Inf, no constant segment**; ABP range 28.4–188.7 mmHg; SBP median 111.4
(IQR 99.9–124.7, 63.8–185.8), DBP median 57.1 (49.7–65.6, 31.1–115.6), PP median 55.0 mmHg (ICU population, arterial line). The
official split lists cover all 1,524 IDs exactly once (no overlap). The neonate p028331 (paper caveat) is in the official *train* list
and is kept (official split, unchanged).

## 4. PENGUIN preprocessing (pinned code) and what we reproduce
- Windowing (`load_data.py` L130–146): per subject, `sliding_window_view(x, (1, fs·L))[:, ::fs·L]` on the `[30, 3750]` array → non-
  overlapping L-second windows **within** each 30 s segment, PPG and ABP indexed identically (alignment preserved); L = 8 → 3 windows per
  segment, the last 6 s dropped → 90 windows/subject → **137,160 windows** — exactly `sample_num` in the shipped config (which therefore
  was computed with 8 s windows although `segment_len: 4` is shipped — the same 4 s vs 8 s discrepancy documented for ECG in
  `docs/PENGUIN_AUDIT.md`; the paper evaluates "over an 8-second window"). We use **8 s** as in A0–A6.
- `preprocess()` per window: FFT resample 125 → 128 Hz (1000 → 1024 samples); PPG band-pass 0.5–4 Hz + z-score + min-max to [−1, 1];
  **ABP: `label_bandpass/zscore/normalize = False`** — paper: "for ABP, no further pre-processing was applied, as its amplitude carries
  critical physiological meaning" → targets stay in **mmHg**. No inverse transform is needed and no target statistic is used at inference.
- Reproduced by `src/ppg2ecg/data/mimicbp.py` + `scripts/build_processed_mimicbp.py` → `data/processed/mimicbp_8s/<pid>.npz`
  (x float32 [n, 1024] normalised PPG, y float32 [n, 1024] ABP mmHg, segment_idx, window_start_s, label_sbp/dbp); MANIFEST.json with
  per-file sha256: **1,524 subjects, 137,160 windows, 0 dropped**.
- PENGUIN's own split (`load_dataset_path`): `random.sample` over processed files, `fold_num 8` → val 190 / test 190 / train 1,144 subjects
  with an unseeded-order glob (non-reproducible, as documented for DaLiA). Paper: "6:1:1 … no subject overlap". We use the **official
  MIMIC-BP split** (rule §17: official reproducible subject split first): `data/manifests/split_a7_mimicbp_official.json`
  (train 1,100 / val 195 / test 229; source-file sha256 recorded).
- PENGUIN metrics (`help_func.compute_metrics`): `SBPError` = mean |max(pred) − max(target)| and `DBPError` = mean |min(pred) − min(target)|
  per window, in the target's units (mmHg since ABP is raw). Paper Table 1, MIMIC-BP: PENGUIN SBP 17.43 / DBP 11.34 mmHg (RDDM 20.26 /
  10.49; best baseline DBP); UCI-BP 12.61 / 7.14. (For reference, the MIMIC-BP paper's "average guessing" baseline on its own protocol is
  SBP 13.96 / DBP 9.22 — PENGUIN's window-max/min metric is a different, harder quantity.)
- Training recipe (paper §4.3): AdamW(0.9, 0.999) lr 1e-3, batch 64, ≤ 300 epochs, patience 10, L = 4, n = 128, m = 256; 25 Heun steps.
  **No task-specific optimiser setting for ABP** → our frozen recipe (AdamW 1e-3 / wd 0.01 / batch 64 / seed 42) applies unchanged.

## 5. Leakage analysis
- Subject-level: official split is subject-disjoint (verified: pairwise intersections empty). Record-level: one record set per subject.
- Window-level: windows are non-overlapping within segments and segments are disjoint → exact-hash overlap check run in preflight.
- Normalisation: PPG statistics are per window from the *input* only; ABP is untouched → **no target-derived statistic is used at
  training or inference; no inverse transform**. SBP/DBP are evaluated directly in mmHg on the raw target.
- Alignment: PPG/ABP windows share indices within the same segment; the curation enforced beat-level synchronicity (PAT consistency).
- Resampling parity: both channels FFT-resampled 1000 → 1024 samples per window (same as ECG protocol).
- Known caveats: ICU/arterial-line population, drugs/interventions unknown; SOFA scores available but unused; one neonate in train.

## 6. Discrepancies (paper / code / local)
| Item | Paper | Code | Local |
|---|---|---|---|
| Window length | 8 s (metrics), Fig. 2 "4-second segment" | `segment_len: 4`, `sample_num` consistent with 8 s | 8 s |
| Split | 6:1:1 subject-disjoint | random glob-order 8-fold slices | official MIMIC-BP lists |
| Patience | 10 | 10 | 20 (A0-b protocol, min_delta 1e-4; frozen since A0-b) |
| License | — | — | ODbL 1.0 (data), CC BY-NC-ND (paper) — non-commercial research use |
