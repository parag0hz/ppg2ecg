# C0 audit (2026-10-01) — before training and before any ARCH-VAL outcome

C0 is a new prospective architecture line. It is not BF1 and not a D0 continuation. BF0 (`504395d` / `d380e3b`) and D0
(`d85f3ef` / `a54ec7e`) stay closed.

## 1. Existing conventions reused (no new tuning)

| component | convention (source) |
|---|---|
| timing detector | RD1 `RhythmTCN` (Global-TCN, 328,897 parameters); Gaussian target σ = 20 ms at reference R; BCE; 14,000 steps, batch 64 windows, AdamW lr 1e-3, weight decay 0.01, seed 42, epoch-permutation batches; events by `extract_events(threshold 0.35, refractory 32)` (`scripts/rd1_direct_rpeak.py`) |
| BF0 deterministic beat | `BeatFlowNet` (482,049 parameters) called with x_t = 0, t = 0; L1 on R-aligned 166-sample segments; 20,000 steps × 256 beats, AdamW 1e-3 / 0.01, clip 1.0, seed 42 (`bf0_run._train_beat_model`) |
| BF0 renderer | Hann overlap-add with renormalization, nearest-covered-value fill, empty window = template baseline (`beatfirst/render.assemble`, `bf0_run.conditions`) |
| reference R | neurokit on the ECG (`rpeaks.detect_rpeaks`). The RD1 cache `outputs/rd1_detector/train_rpeaks.npz` covers every V1-train window; C0 slices it per role and re-verifies 200 salted windows of the role |
| event metrics | `paper_metrics.rpeak_prf_at` / `beat_level_metrics` (±50 ms, one-to-one); F1 over evaluable windows (≥ 1 reference beat) |
| waveform metrics | FD `paper_metrics.kanflow_fd` (raw regime, n ≥ 3,000); matched-pair beat-aligned correlation (`render.matched_pairs` / `pair_correlations`); S4 / S5 (`m1_structural.qrs_core_morphology`); spectral ratio deviation (`m1_structural.spectral_metrics`, band mean); MAE / PCC (`metrics.signal_metrics`) |
| bootstrap | patient-clustered, equal patient weight (`bf0_run.patient_resamples`), 2,000 replicates, C0 seed 20261001; FD recomputed inside each replicate |

## 2. New split (metadata only)

- **Source.** The split is built from `data/manifests/split_v1_vitaldb_seed42.json` (`extra.patient_of_case`). No data file
  was opened to build it.
- **Rule.** The 4,337 V1-TRAIN patients are sorted and permuted with `default_rng(20261001)`. The first 434 go to
  ARCH-HOLDOUT, the next 433 to ARCH-VAL, and the remaining 3,470 to ARCH-TRAIN.

| role | patients | cases | windows |
|---|---|---|---|
| ARCH-TRAIN | 3,470 | `split_manifest.json` | 231,220 |
| ARCH-VAL | 433 | | 28,649 |
| ARCH-HOLDOUT | 434 | | 28,531 (manifest count; not loaded) |

- **Overlap checks:** train–val, train–holdout, val–holdout, old-val–ARCH and old-test–ARCH are all 0.
- **Old splits:** the old validation (289 patients) and test (1,156 patients) are never used.
- **Audit loads.** Only ARCH-TRAIN and ARCH-VAL were loaded. Neither has a window without reference beats.
  - ARCH-TRAIN median reference RR: **100 samples (0.781 s)**. This is the edge / singleton RR for every arm.
  - Reference peaks: ARCH-TRAIN 1,048,496; ARCH-VAL 129,825.

## 3. History of ARCH-HOLDOUT patients (disclosed)

These patients belong to the V1 TRAIN pool, so:

- they were in the training data of earlier models (RD1, the BF0 beat models, the iMF / CD / PENGUIN generators);
- they were part of TRAIN-only diagnostics, such as RD1 cache verification and D0's crossfade statistics on 3,000 salted
  V1-train windows.

No outcome on them informed C0. Every C0 component (detector, BF0-DET-RETRAIN, C0, C0-LOCAL-ONLY) is trained from
scratch on ARCH-TRAIN. ARCH-HOLDOUT is "untouched by C0", not "never seen by the project".

## 4. Decisions on points the specification leaves open (frozen in the preregistration)

1. **Training events.** C0, C0-LOCAL-ONLY and BF0-DET-RETRAIN are trained at **reference R peaks**, the BF0 convention.
   Every arm is **evaluated** at the retrained detector's events.
2. **Placed events.** The retrained detector's events are used as is (threshold 0.35, refractory 32). The BF0 timing head
   is not used, because the specification names a single detector.
3. **Integer convention.** The residual grid is τ = −38 … +57 samples (96 samples). The maximum support is
   L = 38.4 / R = 57.6 samples, exactly 300 / 450 ms. The envelope is evaluated at integer τ with real-valued
   L_i = min(38.4, 0.45 RR_prev) and R_i = min(57.6, 0.45 RR_next). A missing neighbour uses the ARCH-TRAIN median RR
   (100 samples).
4. **Global spline.** A centred cardinal cubic B-spline with knot spacing 32 samples (250 ms) and 17 coefficients at
   t = 0, 32, …, 512. Each coefficient comes from encoder features averaged over ±16 samples around its control point.
5. **C0 training.** 20,000 steps, batch 64 windows (RD1's window-level convention), AdamW 1e-3 / 0.01, clip 1.0,
   seed 42, L1 over the full window, last checkpoint.
6. **Network.** One 8-block dilated encoder (64 channels). The local decoder is four FiLM blocks conditioned on
   [window encoder mean, RR_prev, RR_next]. Parameters: C0 592,770 (1.23 × BF0-DET's 482,049); C0-LOCAL-ONLY 588,545.
7. **BF0-DET-RETRAIN rendering.** BF0's renderer and conditioning, at the same placed events. The template and singleton
   RR come from ARCH-TRAIN.

## 5. Known limitation, recorded before results

C0-LOCAL-ONLY sets g = 0, as specified. On this data the ECG baseline lies near −0.5 (BF0's template baseline −0.54),
so outside the local supports LOCAL-ONLY is forced to 0. A C0 − LOCAL-ONLY difference therefore cannot separate "a
time-varying global context" from "any non-zero baseline". The specification forbids further ablations before the
holdout, so no extra arm is added.

## 6. Synthetic dry run

All stages ran end to end on neurokit-simulated windows in a scratch directory, including a forced holdout path; the
synthetic numbers mean nothing.

- **Holdout seal:** loading the holdout before a freeze manifest raised `PermissionError`.
- **Bugs fixed before the commit:** freeze-path handling for outputs outside the repository, and the figure's label
  placement.
