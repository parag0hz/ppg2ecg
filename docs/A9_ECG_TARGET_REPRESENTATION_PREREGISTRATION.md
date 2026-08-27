# A9 Pre-registration — ECG Target-Representation Mirror Control (WildPPG)

Written 2026-08-28 **before any A9 training**; frozen by the commit that introduces it. One intervention only: how the ECG target is
represented. No new architecture, objective, loss, split, dataset, seed or hyper-parameter. WildPPG only; DaLiA is **not** run.

## 1. Question and the confound it tests
The ECG line (A2–A6) used the frozen PENGUIN protocol, in which the **target ECG is normalised per window** (z-score + min-max to
[−1, 1]); A8's ABP work used a **global train-only affine** target transform. A reviewer can therefore object:
> "The observed ECG one-step attenuation is caused primarily by per-window target normalisation."
A9 tests exactly this objection, and nothing else.
**RQ1** does the deterministic MSE proxy still attenuate ECG morphology under global train-only normalisation? **RQ2** does OT-CFM 1-NFE
remain the closest model to that proxy? **RQ3** does iMeanFlow-1 still recover morphology/amplitude relative to OT-CFM-1? **RQ4** does the
pointwise-error inversion persist *within* the new representation? **RQ5** does the WildPPG timing-vs-morphology dissociation persist?

## 2. Frozen dataset, split and windows (A4/A5/A6 protocol, unchanged)
WildPPG, 16 participants, green PPG, four sites pooled with tiled sternum ECG, 8 s windows @128 Hz, same gap/constant-window handling,
same deterministic split (`data/manifests/split_a4_wildppg_seed42.json`, sha256 `bc168144…`; train 12, val an0/k2s, test kjd/ssx), same
uniform-stride ≤ 4,096-window val/test subsets (test = the same 3,907 windows), same 220-step rounds, same paired noise seed 0, same
PPG-shuffle derangement seed 1, same pre-registered example windows. No new split, no new subset, no new tolerance.

## 3. ECG source stage (§4 of the task spec — no inverse transform of the old targets)
The A9 targets are built from the raw WildPPG `.mat` files through the **identical** pipeline up to but excluding per-window
normalisation: FFT resample to 128 Hz → 4th-order Butterworth 0.5 Hz high-pass (zero-phase) → window extraction. Implemented as
`scripts/build_processed_wildppg.py --ecg-normalization none` → `data/processed/wildppg_8s_prenorm/` (389,355 windows, 861 dropped —
identical to the frozen dataset). Verified: **PPG tensors, window indices and site labels are bit-identical** to `wildppg_8s` for all 16
subjects; the only difference is that the ECG keeps its native amplitude (`zscore=False, normalize=False`).

## 4. The single intervention — global train-only affine ECG normalisation
`y_global = (y_ecg − mu_train)/sigma_train`, **one scalar pair** over all pre-window-normalisation ECG samples of the 12 TRAIN subjects:
**mu_train = 1.575417, sigma_train = 10501.669122** (native WildPPG ECG units), n = 300,309,504 samples
(`artifacts/a9_ecg_representation_control/normalization.json`, with split/processed hashes, train subject list, source-stage description
and the code SHA; computed under a file-open guard that raises if any val/test subject file is touched → `leakage_check.ok = true`).
Forbidden and not used: per-window, per-subject, per-recording, per-activity, per-site statistics; val/test statistics; clipping; robust,
min-max or quantile scaling. PPG preprocessing is untouched.

## 5. Leakage / parity gate (must pass before training)
(1) subject-disjoint split; (2) the A9 test windows are asserted equal to the A4 ones; (3) PPG bit-exact vs A4; (4) the ECG source is the
pre-normalisation stage of the same pipeline (manifest records `zscore/normalize = false`); (5) affine round-trip test; (6) no val/test
statistics (guard + unit test); (7) no target information in the PPG path; (8) window-hash disjointness (preflight); (9) no NaN/Inf;
(10) PPG↔ECG alignment unchanged (same window indices); (11) filtering/resampling unchanged (same `preprocess_windows` call).

## 6. Representation geometry audit (measured before training; `representation_geometry.json`)
Fixed train subset (first 6 train subjects, 4,096 windows, prior seed 0):
| Quantity | window-norm | global-z | native (pre-norm) |
|---|---:|---:|---:|
| target mean / std | −0.371 / 0.363 | 0.001 / 1.223 | 13.85 / 12838 |
| per-window mean std | 0.265 | 0.031 | 327.7 |
| per-window std (mean ± sd) | 0.244 ± 0.042 | 0.709 ± 0.995 | 7450 ± … |
| ‖y‖ / ‖e‖ vs standard-normal prior | **0.493** | **0.710** | 7460 |
| ‖y − e‖ | 35.96 | 42.90 | 238,465 |
| prior share of interpolant energy at t = 0.25 / 0.5 / 0.75 | 0.292 / 0.788 / 0.971 | 0.069 / 0.401 / 0.858 | ≈ 0 |
| HF-energy ratio (> 15 Hz) | 0.1854 | 0.1854 | 0.1854 |
| derivative RMS / median max-slope | 15.2 / 105 | 38.4 / 158 | — |
Both representations are O(1) against the prior (0.49 vs 0.71), so — unlike A8, where raw ABP was 81.6× — **A9 compares local versus
global normalisation, not a scale mismatch** (§27 of the task spec). What differs is amplitude heterogeneity: window-norm makes every
window ≈ equal amplitude, global-z preserves the native spread (per-window std sd 0.04 → 1.00; train subjects' ECG std spans 2,533–24,720,
a 9.8× range). Spectral shape is unchanged by an affine map (identical HF ratio), as expected.

## 7. Measurement-invariance check (data only, before training)
The frozen R-peak detector gives **identical results in both representations** on the 3,907 test windows: identical beat counts in
100.0 % of windows (mean |Δ| = 0.000) and 1.000 R-peak position agreement at the 50 ms tolerance. Timing, morphology and HF metrics are
therefore directly comparable across representations. Caveat recorded in advance: 10 % of test windows have global-z GT std ≤ 0.009, so
the mean amplitude ratio can be inflated by near-flat windows; the **median** amplitude ratio (an existing field of the frozen metric CSV)
is reported alongside and used when the mean and median disagree by more than 0.15.

## 8. Models — exactly three retrained runs, frozen recipes, target representation only
| Run | Objective | Recipe |
|---|---|---|
| `a9_mse_fullbackbone_wildppg_globalz_seed42` | MSE proxy | A6 full backbone (4,568,707 params / 4,304,513 effective), x_const 0.1, t_const 0.5, cond 0.05·E(t), deterministic val-MSE |
| `a9_otcfm_wildppg_globalz_seed42` | OT-CFM | A0-b/A4: fixed-bank validation CFM loss (4 banks, seed 1000) |
| `a9_imeanflow_wildppg_globalz_seed42` | Improved MeanFlow | A2/A4: h-only conditioning, boundary v_θ, p = 1, c = 0.01, logit-normal(−0.4, 1), 50 % r = t, forward-mode JVP, micro-batch 32 × 2 |
All: AdamW lr 1e-3, wd 0.01, effective batch 64, seed 42, 220-step rounds, max 300, patience 20, min_delta 1e-4, test never used for
selection. The A4 (OT-CFM, iMF) and A6c (MSE) window-norm runs are **reused as the reference**, not retrained.

## 9. Evaluation
Predictions are produced in the trained (global-z) space and evaluated there with the **unchanged frozen metric code**
(`ppg2ecg.evaluation.metrics`, `rpeaks`): HR error, template correlation, amplitude ratio (mean and median), conditioning gain via the
same PPG-shuffle derangement, beats/reference, HF-energy ratio, R-peak precision/recall/F1 (50 ms), RR MAE, QRS-width error, upstream
HR error, latency, NFE, plus RMSE/MAE. The inverse transform `y = sigma_train·y_global + mu_train` is available and used only where a
native-unit statement is needed; all reported metrics are affine-robust ratios/correlations/timings except RMSE/MAE.
**Cross-representation rule (frozen):** absolute RMSE/MAE are **never** compared between representations; only the *ranking* of
{MSE, OT-1, OT-50, iMF-1} by RMSE/MAE within each representation is compared. Window-norm predictions are never "un-normalised" using
test-window statistics.
Arms: MSE 1 forward; OT-CFM 50/4/1 NFE (2/10/20 also dumped); iMF 1 NFE primary, 2/4 diagnostic.

## 10. Frozen operational definitions
**Structural attenuation** (within global-z) for model M ∈ {MSE, OT-1}: `morph(M) < morph(OT50) − 0.10` **and** at least one of
(a) `|amp(M) − 1| > |amp(OT50) − 1| + 0.10`, (b) HF-energy ratio materially lower than OT-50's (`HF(M) < 0.5·HF(OT50)`).
**iMF recovery**: `Recovery = (M1 − O1)/(O50 − O1)` for morphology and for amplitude fidelity `Q_amp = −|amp − 1|`; computed when the
denominator ≥ 0.05, otherwise reported as ill-conditioned with the raw differences. Recovery is present if morphology recovery ≥ 0.5 **and**
amplitude fidelity improves over OT-1.
**Conditional-mean similarity**: waveform RMSE/MAE/PCC between predictions plus |Δ morph|, |Δ amp|, |Δ HF| and a beat-timing distance;
"closest" = smallest waveform RMSE and ≥ 2 of 3 statistic votes.
**Pointwise-error inversion** (within a representation): the model with the best RMSE also ranks last on template correlation or on
amplitude fidelity.
**QRS region**: ±100 ms around GT R-peaks (A5 definition); QRS vs non-QRS RMSE, QRS energy retention, peak amplitude ratio, max local
slope, QRS width, HF energy.

## 11. Hypotheses (frozen)
**H9.1** under global-z the MSE proxy still shows materially lower morphology/amplitude fidelity than OT-50. **H9.2** OT-1 remains the
closest of {OT-1, OT-50, iMF-1} to the MSE proxy. **H9.3** iMF-1 improves morphology and amplitude fidelity over OT-1. **H9.4** the
timing-vs-morphology dissociation persists (mean-like solutions keep aligned beat timing better than sharp morphology; iMF restores
morphology with greater beat-placement variability). **H9.5** (secondary) the pointwise-error inversion persists within global-z; H9.5 may
fail without invalidating H9.1–H9.3.

## 12. Verdict (frozen)
- **REPRESENTATION-ROBUST**: (1) MSE attenuation persists, (2) OT-1 attenuation persists, (3) OT-1 remains closest to the MSE proxy,
  (4) iMF-1 materially improves morphology over OT-1 (recovery ≥ 0.5 or, if ill-conditioned, morph(M1) > morph(O1) + 0.10).
  Reading: *ECG one-step conditional-mean-like attenuation cannot be explained solely by per-window target normalisation.*
- **REPRESENTATION-SENSITIVE**: under global-z the MSE proxy reaches OT-50-level morphology (within 0.10), or OT-1's collapse
  substantially disappears, or the iMF advantage largely vanishes.
- **MIXED**: mean-like similarity persists but the size of the attenuation or of the iMF recovery changes substantially.
- **UNINTERPRETABLE**: the global-z representation produces training pathology or an extreme transport mismatch that makes the controlled
  comparison impossible. No rescue by re-normalisation.

## 13. Result-dependent behaviour is forbidden
Whatever the outcome: no change of normalisation, no re-training, no subject/subset change, no LR/loss change, no DaLiA run, no new
architecture. Wording: "timing variability" / "beat-placement variability" — never "phase diversity"; the standing A5 §13 forbidden list
applies. **After A9: STOP** and report to the user before anything else.
