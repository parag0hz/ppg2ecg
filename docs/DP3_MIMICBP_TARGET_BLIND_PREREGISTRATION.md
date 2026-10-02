# DP3 — MIMIC-BP ECG-target-blind cross-task external validation — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-02).**
- Committed and pushed with the MIMIC-BP adapter **before any MIMIC-BP waveform value is read and before any model
  performance on MIMIC-BP exists**.
- Seeds 43 / 44, the internal multi-seed characterization and the compute baselines were completed and committed first
  (`3a2d8bd`). None of them involves MIMIC-BP, and none was used for any decision below: every rule here comes from the DP3 spec
  and DP0.
- It is never edited afterwards; deviations are dated amendments written before ECG-outcome access.
- The final freeze (`artifacts/dp3_mimicbp/final_freeze_manifest.json`, a separate later commit) hashes this document,
  the code, the adapter and every checkpoint.

## 1. Evidence status (exact language)

- **Label:** "**ECG-target-blind cross-task external cohort**".
- **Required disclosure:** "MIMIC-BP PPG had previously been used in unrelated PPG-to-ABP experiments, whereas ECG
  reconstruction targets were withheld from the PPG-to-ECG architecture-development process."
- **Never call MIMIC-BP:** project-naive, fully fresh, completely unseen, completely untouched, or independent of all
  prior research activity.
- **Target-blind audit (Stage A, `docs/DP3_MIMICBP_TARGET_BLIND_AUDIT.md`, commit `7c2274c`): PASS.** No MIMIC-BP ECG
  waveform or ECG-derived target was used before. Prior PPG → ABP use is disclosed: A7, A8, U1, U2 and EXP-D.

## 2. Frozen model and seeds (Stage B integrity PASS)

- **Model:** DualReadout-ECG **S1 MIDDLE**, exactly as in DP0.
  - Stem + blocks 1–6 shared; adapter + blocks 7–8 + decoder private per task; widths 71 / 30.
  - 943,372 waveform parameters (246,720 shared, 8,320 adapter) against 1,191,910 for the two specialists, a saving of
    20.85 %.
  - Specialists: P = WW-L1 (593,577), G = SCALEFLOW-COUPLED (598,333).
- **Detector:** the frozen DP0 detector (`outputs/dp0_dualreadout/detector.pt`): threshold 0.35, refractory 32,
  Gaussian raster σ = 20 ms. It is never retrained or tuned.
- **Seeds:**
  - **42 (primary):** the frozen DP0 checkpoints, hash-verified.
  - **43 / 44 (predefined robustness replications):** P, G and S1 trained from scratch on **DP-TRAIN only** (the
    2,100 DP0 patients) with the DP0 protocols, only the seed changed. Seed-dependent parts: initialization, the batch
    streams and the FM noise / t.
  - Records: `training_seed_manifest.json` (config sha256 identical across seeds; NaN = 0; 20,000 / 20,000 updates).
- **No search and no rescue:**
  - nothing is changed in architecture, widths, dilations, adapters, decoders, Haar, losses, optimizer, LR, batch,
    updates, detector, raster, solver, NFE or normalization;
  - no MIMIC-BP data enters any training, fine-tuning or domain adaptation;
  - no best seed is chosen, and no seed is dropped or added.

## 3. Cohort and adapter (`dataset_schema.json`, `dataset_adapter.json`, `external_subject_manifest.json`)

- **Cohort:** all 1,524 MIMIC-BP subjects (`data/raw/MIMIC-BP`, MIMIC-III Waveform Matched Subset curation) form one
  external evaluation cohort. There is **no calibration split**.
  - Each subject has `ppg` and `ecg`, shape (30, 3750), at 125 Hz (verified from metadata).
  - The subject-ID list has a sha256 in the manifest.
- **Channels:** exactly one PPG and one ECG channel exist per subject, and both are used. The ECG lead identity is not
  stated in local metadata (UNKNOWN; the DP0 models were trained on VitalDB ECG_II). No lead or channel choice uses
  model performance.
- **Windows:**
  - 4 s, non-overlapping, starting at each 30 s segment start, stride 4 s;
  - 7 windows per segment; the last 2 s of each segment are dropped (no padding, no cross-segment window);
  - order: subject id, then segment, then window;
  - window id = subject rank · 210 + segment · 7 + k (at most 320,040 windows);
  - V1's candidate subsampling is not applied, so all windows are used.
- **Resampling and normalization:** `ppg2ecg.data.preprocess.preprocess_windows`, the V1 / DP0 implementation
  (unchanged since `a15b354`).
  - FFT resampling 500 → 512 samples per window.
  - PPG: 0.5–4 Hz band-pass. ECG: 0.5 Hz high-pass.
  - Then per-window z-score and min-max to [−1, 1]. Nothing is fitted on any data.
- **Temporal alignment:** none. PPG and ECG windows take identical sample indices of the same segment. There is no
  shift, no pulse-transit compensation and no offset search; the physiological delay stays part of the task.
- **Exclusions** (`external_exclusion_rules.json`, logged per subject and window in `external_exclusion_log.csv`):
  - R1: unreadable file or shape ≠ (30, 3750) → subject;
  - R2: raw window with a non-finite sample or zero SD (PPG or ECG) → window;
  - R3: non-finite value after preprocessing → window;
  - R4 (the frozen V1 / DP0 reference-ECG validity rule): neurokit HR of the preprocessed ECG window not finite or
    outside [30, 200] bpm → window;
  - R5: subject with no remaining window → subject.
  - No exclusion uses a model output, morphology, FP, F1, FD or subject difficulty.
- **Reference R:** neurokit (`rpeaks.detect_rpeaks`) on the preprocessed 128 Hz ECG window, the DP0 convention.
- **Events:** frozen detector → threshold 0.35, refractory 32 → raster, identical for every model and run.
- **Noise:** `scaleflow.window_noise(int(pid[1:]), window id)`, identical for G and S1 (and for the shuffle).
- **K16 subset:** 2,000 eligible windows by salted rank `dp3-k16-v1` over window ids, fixed before outcomes.

## 4. Metrics (DP0 definitions, unchanged)

- **Point** (P and S1 point):
  - beat-aligned morphology correlation on the (reference R, detector event) pairs;
  - patient-pooled FP / window, recall, precision and F1 (equal patient weight; neurokit on the output, ±50 ms);
  - pooled F1, RR-MAE, HR-MAE and MAE;
  - FD (descriptive).
- **Generative** (G and S1, one sample per window, Euler NFE 8):
  - FD (KANFlow raw 512-d Gaussian Fréchet);
  - descriptive corr, FP, recall, F1, MAE, the AF0 spectral discrepancy and a diversity ratio.
- **Detector** (descriptive, after opening): patient-pooled precision, recall, F1 and FP / window; RR-MAE; HR-MAE.
- **K16** (secondary, G42 and S1-42):
  - within-condition waveform diversity and beat-aligned diversity;
  - the generated / real diversity ratio;
  - R-time seed SD;
  - single-sample vs K16 consensus HR MAE;
  - sample-mean corr.
  - No uncertainty-calibration claim is made from K16.
- **Patient-level heterogeneity** (seed 42, descriptive): per-patient distributions of Δcorr, ΔFP and ΔHR-MAE. There
  are no subgroups.

## 5. Gates (seed 42 primary; identical definitions for seeds 43 / 44)

| gate | quantity | pass rule |
|---|---|---|
| T1 | corr(S1 point) − corr(P) | 95 % CI lower > −0.02 |
| T2 | FP(S1) − FP(P); recall(S1) − recall(P) | FP CI upper < +0.05 **and** recall CI lower > −0.01 |
| T3 | FD(S1 gen) − FD(G) | 95 % CI upper < +1.0 (a negative estimate is not a superiority claim) |
| T4 | S1 generative with PPG shuffled across windows (`default_rng(20261002)`) − conditioned; raster, noise, model fixed; shared features recomputed | FD CI entirely > 0 **and** corr CI entirely < 0 |
| T5 | 1 − P_S1 / (P_P + P_G) | ≥ 15 % (structural: 20.85 %) |

**Bootstrap:**
- 2,000 patient-clustered replicates, seed 20261002;
- the patient is the unit, and all same-seed comparisons are paired;
- FD is recomputed inside every replicate (exact sufficient-statistics bootstrap, verified against `kanflow_fd`).

## 6. Verdicts (fixed now)

- **Primary:**
  - **TARGET-BLIND EXTERNAL CONFIRMED** only if seed 42 passes T1–T5;
  - otherwise **TARGET-BLIND EXTERNAL FAILED**.
- **Seeds (T1–T4):**
  - **ROBUST-3/3:** 42, 43 and 44 all pass;
  - **ROBUST-2/3:** exactly two pass;
  - **SEED-SENSITIVE:** one or zero pass.
  - Seed 42 stays primary regardless.
- **Paper readiness:**

  | label | condition |
  |---|---|
  | STRONG GO | seed 42 confirmed and ROBUST-3/3, with DP0 still internally confirmed |
  | GO WITH SEED LIMITATION | seed 42 confirmed and ROBUST-2/3 |
  | INTERNAL + CROSS-TASK EVIDENCE ONLY | seed 42 failed; the DP0 internal confirmation stands |

  A confirmed seed 42 with SEED-SENSITIVE robustness is reported as a confirmed primary with seed sensitivity, with no
  GO label.
- **Compute claim:** only relative to SEPARATE-CACHED (`compute_cached.json`). If S1 does not beat it, only the
  parameter saving is claimed.

## 7. Order and no rescue

**Order:**
1. adapter + this preregistration (commit);
2. final freeze (commit);
3. `eval_external` (refuses without the committed, unchanged freeze);
4. report.

**After MIMIC-BP ECG values are opened, none of the following is done:**
- retraining or fine-tuning, or adding seed 45;
- changing the detector, threshold, refractory, alignment, resampling, normalization, lead, exclusions, architecture,
  margins, NFE or solver;
- dropping subjects or choosing a seed.

The result is reported as it is.

## 8. Tests

`tests/test_dp3_mimicbp.py` has 17 test cases covering the 34 spec items:
- **Frozen S1:** accounting, sharing graph and ownership counts.
- **Seeds and data:** seed-42 hashes; seeds 43 / 44 differ only by seed (config hashes); detector frozen with threshold
  0.35 / refractory 32; DP-TRAIN split hash; no MIMIC-BP in training and no fine-tuning code in evaluation.
- **Training protocol:** exact L1 and FM losses; round-robin masks and counts; Euler 8; the Haar round trip.
- **Adapter** (synthetic subjects): deterministic windows; R2 exclusions; window ids; frozen resampler and
  normalization; metadata-only channel rules; no alignment search.
- **Seals:** the external loader is blocked without a freeze or a preregistration.
- **Evaluation:** noise pairing; shuffle keeps raster and noise; patient-clustered bootstrap; the cached-separate
  generation is bit-identical to naive.
- **Verdicts:** gate margins, the seed classification with seed 42 primary, no architecture search, and the
  target-blind audit PASS.

**Synthetic dry run:** `eval_external` and `summarize` ran end to end on 18 synthetic MIMIC-BP-shaped subjects
(3,779 windows) with the real frozen checkpoints in a scratch directory. The numbers mean nothing, and the real MIMIC-BP
directory was not touched.
