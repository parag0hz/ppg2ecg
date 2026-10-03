# DP5 — MESA Sleep project-naive external validation of DualReadout — PREREGISTRATION DRAFT

**Status: DRAFT (2026-10-03), written before MESA access.** The NSRR data request was submitted by the user on
2026-10-03; no MESA file has been downloaded.
- It is frozen (committed and pushed) only after the header-only audit (`docs/DP5_MESA_DATA_AUDIT.md` §6) and before any
  MESA signal sample is read.
- The decisions marked **D1–D4** are the user's.
- Every other rule is copied from DP0 / DP3 unchanged.
- Results enter the manuscript as a new external-validation section, whatever they are.

## 1. Question and evidence label

- **Question:** do the frozen DualReadout-ECG S1 MIDDLE models keep the capability of their same-seed specialists on a
  cohort that no study in this project has used?
- **Label, used only if every condition in §8 holds:** "**project-naive external cohort**".
- **Disclosure:** the waveform models are frozen. Under D1 the timing front-end is re-fitted on a disjoint MESA
  calibration split.

## 2. Frozen models (no training of any waveform model)

| seed | P (WW-L1) | G (SCALEFLOW-COUPLED) | S1 MIDDLE | source |
|---|---|---|---|---|
| 42 (primary) | `outputs/dp0_dualreadout/point.pt` | `gen.pt` | `S1.pt` | DP0, hash-verified against `artifacts/dp0_dualreadout/lock_freeze_manifest.json` |
| 43 / 44 | `outputs/dp3_mimicbp/seed{43,44}/point.pt` | `gen.pt` | `S1.pt` | DP3, hash-verified against `artifacts/dp3_mimicbp/final_freeze_manifest.json` |

- **Unchanged:** architecture, widths, adapters, Haar, losses, Euler NFE 8 and the solver.
- **No MESA waveform enters training:** no MESA signal enters any waveform-model training, fine-tuning or adaptation.
- **Seeds:** no seed is added, dropped or selected.

## 3. Cohort, split and windows

**Eligibility** (from the Stage-A header audit, before any sample is read):
- an EDF with exactly one channel labelled `EKG` and one labelled `Pleth`, both at 256 Hz;
- a duration of at least 600 s (the V0 rule);
- a readable header.
- The two files with documented corruption (1738, 6476) are excluded by documentation, not by values.

**Split:**
- Eligible mesaids are sorted numerically and permuted with `numpy.random.default_rng(20261002)`.
- **MESA-CAL:** the first round(0.30 · n), used only for the timing detector (D1).
- **MESA-TEST:** the rest, opened once, after the final freeze.

**Windows** (the V1 rules):
- non-overlapping 4 s windows (1,024 samples at 256 Hz) over the whole recording;
- candidate windows taken at `linspace` positions — 128 per subject in MESA-CAL, 32 per subject in MESA-TEST;
- survivors of the exclusions subsampled by `linspace` to at most 64 (CAL) or 16 (TEST) per subject;
- PPG and ECG windows take identical sample indices: no shift, no pulse-transit compensation, no offset search.
- Window id = subject rank × 10,000 + candidate index (noise key).

**Preprocessing:**
- `ppg2ecg.data.preprocess.preprocess_windows`, unchanged: FFT resampling 1,024 → 512.
- PPG: `PPG_KW` (0.5–4 Hz band-pass). ECG: `ECG_KW` (0.5 Hz high-pass).
- Then per-window z-score and min-max to [−1, 1]. Nothing is fitted.

**ECG polarity (D3):** no polarity flip, as in DP3. The fraction of subjects with a negative-dominant reference R
deflection is reported descriptively.

**Exclusions** (target side only, as in DP3; logged):
- R1: unreadable EDF or missing channel → subject;
- R2: raw non-finite or zero-SD window (PPG or ECG) → window;
- R3: non-finite after preprocessing → window;
- R4: neurokit reference ECG with fewer than 2 R peaks or HR outside [30, 200] bpm → window;
- R5: no remaining window → subject.
- **Not used:** the oximeter status channel, sleep stages, events and any annotation. No exclusion uses a model output.

## 4. Timing condition (D1)

**Primary — site-calibrated detector:**
- The RD1 / C0 protocol is trained from scratch on MESA-CAL: RhythmTCN, BCE to a Gaussian σ = 20 ms target at the
  neurokit reference R, 14,000 steps × 64, seed 42.
- It is frozen before MESA-TEST is built.
- Events: threshold 0.35, refractory 32, raster σ = 20 ms.

**Secondary — frozen VitalDB detector:** `outputs/dp0_dualreadout/detector.pt`, unchanged.
- All gates are computed and reported.
- No verdict depends on them.

**Both:** one raster per window, identical for P, G and S1 and for the shuffle.

## 5. Metrics and gates (DP3 definitions; seed 42 primary)

**Metrics:** as DP3 §4 — beat-aligned correlation, patient-pooled FP / recall / F1, RR-MAE and HR-MAE, KANFlow FD;
K16 on 2,000 windows by salted rank `dp5-k16-v1`; detector metrics.

**Gates:**

| gate | quantity | pass rule |
|---|---|---|
| X1 | corr(S1 point) − corr(P) | 95 % CI lower > −0.02 |
| X2 | FP(S1) − FP(P); recall(S1) − recall(P) | FP CI upper < +0.05 **and** recall CI lower > −0.01 |
| X3 | FD(S1 gen) − FD(G) | CI upper < +1.0 |
| X4 (D2) | S1 gen with PPG shuffled across MESA-TEST windows (`default_rng(20261002)`) − conditioned; raster, noise and model fixed | FD CI entirely > 0 **and** corr CI entirely < 0 (unchanged from DP0 / DP3) |
| X5 | parameter saving | ≥ 15 % (structural 20.85 %) |

**Bootstrap:** 2,000 patient-clustered replicates, seed 20261002, paired within seed; exact FD bootstrap.

**Noise:** `scaleflow.window_noise(int(mesaid), window id)`.

**D2 (decided):** X4 is kept exactly as frozen in DP0 / DP3. No new conditional-dependence measure is added.

## 6. Verdicts (fixed at freeze)

- **Primary:**
  - **PROJECT-NAIVE EXTERNAL CONFIRMED** only if seed 42 passes X1–X5 with the primary detector;
  - otherwise **PROJECT-NAIVE EXTERNAL FAILED**.
- **Seeds (X1–X4):** ROBUST-3/3, ROBUST-2/3 or SEED-SENSITIVE. Seed 42 stays primary.
- **Secondary detector:** reported per gate, with no verdict.

## 7. Order and no rescue

**Order:**
1. Stage-A header audit (headers only).
2. This document frozen + adapter + tests (commit, push).
3. Build MESA-CAL; train the detector; detector metrics on MESA-CAL only.
4. Final freeze manifest hashing this document, code, the detector and all 9 waveform checkpoints (commit, push).
5. Build MESA-TEST and evaluate once (refused without the committed, unchanged freeze).
6. Report.

**After MESA-TEST opens, none of the following is done:**
- retraining, re-calibration or a second detector fit;
- changing a threshold, window rule, exclusion, polarity, normalization or margin;
- dropping subjects or choosing a seed.

## 8. Conditions for the "project-naive" label

The label is used only if all three hold:
1. the repository search in the data audit still shows no prior MESA use at freeze time;
2. no MESA signal sample was read before the freeze, except MESA-CAL after the preregistration commit;
3. the MESA-TEST evaluation ran exactly once.

## 9. Decisions for the user (before freeze)

| id | decision | status |
|---|---|---|
| D1 | detector policy | **decided by the user, 2026-10-03:** primary = site-calibrated on MESA-CAL (30 % of subjects); secondary = frozen VitalDB detector |
| D2 | condition-use gate | **decided by the user, 2026-10-03:** X4 exactly as in DP0 / DP3; no new measure |
| D3 | ECG polarity | default, not yet confirmed: no flip (as DP3); descriptive polarity report |
| D4 | windows per subject | default, not yet confirmed: V1 rules, CAL 128 → ≤ 64, TEST 32 → ≤ 16 |
| D5 | EDF reader | **decided by the user, 2026-10-03:** a minimal numpy reader in the repository, cross-checked against `pyedflib` installed only in a scratch environment (the project `.venv` is not changed) |

## 10. Planned code (after access)

- **`scripts/dp5_mesa.py`, stages:**
  - `header_audit`;
  - `build cal` / `build test`;
  - `train_detector`;
  - `freeze`;
  - `eval_test`;
  - `summarize`.
- **EDF reading (D5):** a minimal EDF reader, numpy only, tested on synthetic EDF files and cross-checked against
  `pyedflib` in a scratch environment.
- **`tests/test_dp5_mesa.py`:** the split and seals, header-only reading, the window rules, the frozen detector and
  checkpoints, the gate rules, the noise pairing and the shuffle.
