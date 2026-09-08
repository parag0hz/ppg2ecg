# D4 — KANFlow-Protocol Datasets: Our Numbers Beside KANFlow's Published Table

**Descriptive compilation, not a new experiment.** Every number in the "ours" columns was produced by the
**D1 benchmark** (`docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md`, run 2026-09-04/05) and is read
here from `outputs/d1_<corpus>_seed42/eval/summary_by_nfe.csv`. **No model was trained, no metric was
recomputed, and no preregistration gate applies** — D4 only re-tabulates frozen D1 output next to
KANFlow's Table II and Table III.

## 1. The comparison is NOT like-for-like. Three reasons, all measured.

**1.1 Amplitude normalisation differs, so MAE and RMSE are on different scales.**
KANFlow §V: signals are *"resampled to 128 Hz and Z-score normalized"*. Our pipeline is PENGUIN's:
z-score **then** min-max to [−1, 1]. Measured on the same corpora:

| corpus | our target std | z-scored std | scale factor |
|---|---|---|---|
| BIDMC | 0.3627 | 1.0000 | **2.76×** |
| CAPNO | 0.2145 | 1.0000 | **4.66×** |
| VitalDB | 0.3932 | 1.0000 | **2.54×** |

Any amplitude-proportional error is therefore **2.5–4.7× smaller on our scale by construction**.
**Our MAE 0.465 against KANFlow's 0.53 on BIDMC is not evidence of an advantage**, and neither is any
other MAE or RMSE cell. They are reported for completeness and must not be ranked.

**1.2 Our FD is a different quantity.** KANFlow's FD is the Gaussian/FID-style Fréchet distance fitted
either in the raw flattened 512-dim waveform space or, for datasets with < 3000 test segments, in a
PCA ≤ 32 space averaged over 5 trials — and KANFlow itself states FD *"should be interpreted within each
dataset"*. Our `fid_default_features` uses a **surrogate feature map** (recorded in D2 as explicitly not
comparable to any published FID). The two FD columns are not the same measurement; ours is reported so
nothing is hidden, and the discrete Fréchet column is included because it is well-defined, but neither
may be compared to 30.71 / 25.57 / 3.04.

**1.3 Window length and screening differ.** KANFlow uses **4 s** segments with **NeuroKit2 SQI screening**
(ECG quality plus PPG peak-detection and template matching) before metrics; D1 uses **8 s** windows and
**no screening whatsoever**. Splits also differ: KANFlow splits BIDMC/CAPNO/MIMIC-AFIB *within subject*
and VitalDB *by subject*; D1 is subject-level 70/15/15 at seed 42 for all three.

Only **Micro-F1, Macro-F1, RR-MAE (ms) and MAE_HR (bpm)** are on comparable scales, and even those carry
the window-length and screening differences above.

## 2. BIDMC

| metric | NFE 1 | NFE 2 | NFE 4 | NFE 10 | NFE 25 | NFE 50 | KANFlow |
|---|---|---|---|---|---|---|---|
| MAE ‡ | 0.465 | 0.460 | 0.453 | 0.454 | 0.455 | 0.456 | 0.53 |
| RMSE ‡ | 0.560 | 0.557 | 0.549 | 0.551 | 0.553 | 0.553 | 0.82 |
| FD (FID-style) ‡ | 157.7 | 141.3 | 161.9 | 190.7 | 203.4 | 207.8 | 30.71 |
| FD (discrete) | 0.925 | 0.815 | 0.791 | 0.788 | 0.792 | 0.794 | — |
| **Micro-F1** ↑ | 0.494 | 0.493 | 0.497 | 0.498 | 0.500 | 0.499 | **0.70** |
| **Macro-F1** ↑ | 0.494 | 0.493 | 0.498 | 0.498 | 0.500 | 0.499 | **0.72** |
| **RR-MAE (ms)** ↓ | **16.75** | 17.51 | 17.75 | 18.05 | 18.13 | 18.16 | 19.80 |
| **MAE_HR (bpm)** ↓ | 1.97 | **1.59** | 1.60 | 1.63 | 1.61 | 1.65 | 10.29 |

## 3. CAPNO (CapnoBase)

| metric | NFE 1 | NFE 2 | NFE 4 | NFE 10 | NFE 25 | NFE 50 | KANFlow |
|---|---|---|---|---|---|---|---|
| MAE ‡ | 0.386 | 0.373 | 0.353 | 0.340 | 0.335 | 0.334 | 0.56 |
| RMSE ‡ | 0.472 | 0.456 | 0.436 | 0.426 | 0.422 | 0.421 | 0.81 |
| FD (FID-style) ‡ | 271.3 | 243.8 | 251.0 | 277.8 | 290.6 | 295.1 | 25.57 |
| FD (discrete) | 0.902 | 0.831 | 0.790 | 0.772 | 0.768 | 0.764 | — |
| **Micro-F1** ↑ | 0.505 | 0.491 | 0.496 | 0.503 | 0.507 | **0.510** | 0.40 |
| **Macro-F1** ↑ | 0.499 | 0.485 | 0.490 | 0.498 | 0.503 | 0.505 | 0.54 |
| **RR-MAE (ms)** ↓ | 16.69 | 17.77 | 17.41 | 16.11 | **14.87** | 14.88 | 20.86 |
| **MAE_HR (bpm)** ↓ | **17.51** | 18.07 | 19.14 | 20.00 | 20.28 | 20.38 | **5.75** |

## 4. VitalDB

| metric | NFE 1 | NFE 2 | NFE 4 | NFE 10 | NFE 25 | NFE 50 | KANFlow |
|---|---|---|---|---|---|---|---|
| MAE ‡ | 0.400 | 0.404 | 0.395 | 0.395 | 0.396 | 0.397 | 0.56 |
| RMSE ‡ | 0.478 | 0.483 | 0.473 | 0.472 | 0.473 | 0.474 | 0.89 |
| FD (FID-style) ‡ | 169.1 | 131.4 | 126.9 | 161.6 | 182.8 | 189.9 | 3.04 |
| FD (discrete) | 0.726 | 0.680 | 0.660 | 0.650 | 0.649 | 0.650 | — |
| **Micro-F1** ↑ | 0.752 | 0.759 | 0.768 | **0.772** | 0.772 | 0.771 | 0.76 |
| **Macro-F1** ↑ | 0.754 | 0.758 | **0.765** | 0.765 | 0.762 | 0.761 | 0.78 |
| **RR-MAE (ms)** ↓ | 17.85 | 15.93 | 14.98 | 14.32 | 14.16 | **14.09** | 15.24 |
| **MAE_HR (bpm)** ↓ | 8.02 | 7.47 | **7.28** | 7.40 | 7.55 | 7.61 | **0.67** |

‡ = scale- or definition-incomparable to KANFlow (§1.1, §1.2). Report only, never rank.

## 5. MIMIC-AFib — deliberately empty

KANFlow reports MIMIC-AFib (MAE 0.47, RMSE 0.71, FD 36.61, Micro-F1 0.79, Macro-F1 0.79, RR-MAE 22.19,
MAE_HR 1.14). **We cannot fill this row and will not guess it.**

1. **We do not have the waveforms.** Our index scan of the MIMIC-III matched set found 12,940 records
   carrying both PLETH and II across 598,163 hours; two-channel 16-bit raw for that set is ≈ **1.08 TB**.
2. **The cohort has five irreconcilable definitions** (`docs/PREPROCESSING_CONVENTIONS_SURVEY.md` §3.5):
   RDDM and everything downstream say 35 subjects (19 AF); the only released annotation file (figshare
   batch1) has 45 (23 AF / 22 non-AF); Bashar 2019's text says N = 60 while its Tables 2+3 list 50. The
   figshare and IEEE lists overlap on 43 subjects with zero label conflicts, but 7 are IEEE-only and 2
   figshare-only.
3. **The citation chain is broken at the root.** RDDM, PPGFlowECG and KANFlow all cite Bashar et al. 2019
   *"Noise detection in electrocardiogram signals…"* for MIMIC-AFib, but that paper is **ECG-only** and
   defines no PPG pipeline.

Any number we produced would depend on a subject list nobody can reconstruct, and would therefore be
comparable to no published value. The row stays empty.

## 6. What can honestly be said

On the four comparable metrics:

- **Beat detection (Micro/Macro-F1).** VitalDB is essentially matched (0.772 vs 0.76 / 0.765 vs 0.78).
  CAPNO is better for us on Micro-F1 (0.510 vs 0.40) and slightly worse on Macro-F1 (0.505 vs 0.54).
  **BIDMC is clearly worse (0.500 vs 0.70).**
- **RR-MAE (ms).** We are better on all three: BIDMC 16.75 vs 19.80, CAPNO 14.87 vs 20.86,
  VitalDB 14.09 vs 15.24. This is matched-beat R-R interval error and is conditional on successful
  matching, so a lower value with a lower F1 partly reflects scoring fewer, easier beats.
- **MAE_HR (bpm).** Split: BIDMC much better (1.59 vs 10.29), CAPNO much worse (17.51 vs 5.75),
  VitalDB much worse (7.28 vs 0.67). **D2 established that HR error is satisfied by beat rate alone** —
  a PPG-peak template with F1 0.061 scores 1.58 bpm on CapnoBase — so no HR cell should be read as
  reconstruction quality in either direction.
- **NFE.** Nothing improves monotonically with budget. Micro-F1 moves by ≤ 0.02 from NFE 1 to 50 on every
  corpus, and MAE_HR gets *worse* with more steps on BIDMC and CAPNO. This reproduces the D1 finding that
  inference budget buys motion, not accuracy.

## 7. Claim boundary

D4 makes no claim of superiority or inferiority over KANFlow. The normalisation, FD definition, window
length, SQI screening and split all differ, and `docs/PREPROCESSING_CONVENTIONS_SURVEY.md` §6 documents
that no cross-paper number in this field is comparable without restating the pipeline. D4 licenses no
future work and reopens no verdict: M2 remains D and M3 remains D.
