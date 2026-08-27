# A8 Pre-registration — ABP Target-Scale Sensitivity Control

Written 2026-08-27 **before any A8 training**; frozen by the commit that introduces it. One intervention only: the representation of the
ABP target. No new method, no new architecture, no new loss, no tuning.

## 1. Question
> Does the pathological iMeanFlow failure on MIMIC-BP (A7: template correlation 0.140, HF-energy ratio 0.550 vs GT 0.043, upstroke-slope
> ratio 6.13, DBP MAE 21.8 mmHg, RMSE 32.3) arise from the raw-mmHg target scale and the resulting transport/objective geometry, or does it
> persist after a single fixed train-only affine normalisation?

Two questions are kept strictly separate: **Q1** can iMeanFlow train stably on ABP after normalisation? **Q2** does ABP benefit from a
generative one-step model over a deterministic conditional predictor? A YES on Q1 does not imply YES on Q2.

## 2. Correction carried into A8
The A7 report's alternative explanation §9.4 claimed the adaptive weight is "effectively 1.0" on raw-mmHg targets, citing
`lossW 1.0000`. That was a misreading: `train_loss_weighted = mean(δ²·w)` ≈ 1 by construction whenever δ² ≫ c = 0.01, and it is also
≈ 1.0000 in the ECG runs (A2 0.99996, A4 0.99995). The claim is withdrawn in the A7 report and replaced by the transport-geometry
statement that A8 tests. What genuinely differs between the runs is the magnitude of δ² and of the target relative to the prior
(measured in §5 below).

## 3. Frozen dataset (identical to A7, bit-exact)
MIMIC-BP v2.2, official subject split (train 1,100 / val 195 / test 229; manifest sha256 `c52de946…217a7e`), processed
`data/processed/mimicbp_8s` (137,160 windows, MANIFEST sha256 recorded in `normalization.json`), PPG preprocessing and ABP resampling
unchanged, evaluation on the **same 3,435-window uniform test subset** and the same paired noise seed 0 / derangement seed 1 as A7.
No new windows, no new split.

## 4. The single intervention — global train-only affine target normalisation
`y_norm = (y_mmHg − mu_train) / sigma_train` with **one scalar pair** computed from every ABP sample value of the TRAIN subjects only.
Computed and frozen before training (`scripts/compute_abp_norm.py`, guarded so that opening any val/test file raises):
**mu_train = 77.571767 mmHg, sigma_train = 22.275611 mmHg**, n = 101,376,000 samples from 1,100 subjects
(`artifacts/a8_abp_scale_control/normalization.json`, with split/processed hashes and the code SHA). No per-subject, per-recording or
per-window statistics; no val/test statistics; no separate SBP/DBP scaling; no clipping, quantile or min-max transform. The PPG inputs are
untouched. Every prediction is inverse-transformed `y_mmHg = sigma_train · y_norm + mu_train` before any metric.

## 5. Pre-training transport-scale audit (frozen definitions; run before training)
`scripts/transport_geometry_a8.py` on a deterministic train subset (first 64 train subjects, 4,096 windows, prior seed 0):
target mean/std, per-window L2 norm, prior norm, norm ratio, ‖y − e‖, ‖e − y‖ (the OT-CFM conditional velocity), and the interpolant
`z_t = (1−t)y + t·e` norm/std plus the prior's share of the interpolant energy at t = 0, 0.25, 0.5, 0.75, 1.
Result (`transport_geometry.json`): target/prior norm ratio **81.58 → 0.95**, ‖y − e‖ **2608 → 44.8**, prior share of interpolant energy at
t = 0.5 **0.0001 → 0.488**, target std 22.7 → 1.02. (Descriptive audit of the data, not a model result.)

## 6. Models — exactly three retrained runs, frozen recipes
| Run | Objective | Recipe |
|---|---|---|
| `a8_otcfm_mimicbp_globalz_seed42` | OT-CFM (unchanged) | A0-b/A7: fixed-bank validation CFM (4 banks, seed 1000) |
| `a8_imeanflow_mimicbp_globalz_seed42` | Improved MeanFlow (unchanged) | A2/A7: h-only conditioning, boundary v_θ, p = 1, c = 0.01, logit-normal(−0.4, 1), 50 % r = t, forward-mode JVP, micro-batch 32 × 2 |
| `a8_mse_fullbackbone_mimicbp_globalz_seed42` | MSE proxy (unchanged) | A6/A7 full backbone, x_const 0.1, t_const 0.5, cond 0.05·E(t) |
All: same S5 backbone and parameter counts, AdamW lr 1e-3, wd 0.01, effective batch 64, seed 42, 220-step rounds, max 300, patience 20,
min_delta 1e-4, same deterministic selection criteria, test never used for selection. **Only the target representation changes.**

## 7. Leakage tests (`tests/test_target_norm.py`, run before training)
(1) mu/sigma from train subjects only — enforced at computation time by a file-open guard and asserted against the shipped JSON;
(2) val/test target files are never opened during the statistic computation; (3) inverse transform round-trips; (4) clinical metrics are
computed in mmHg (scale-check: absolute errors scale exactly with sigma, correlation is scale-free); (5) PPG preprocessing untouched
(the same processed files are read); (6) the A8 test windows are asserted equal to the A7 ones in the analysis script.

## 8. Diagnostics (frozen definitions)
**Adaptive weight** (`w = 1/(δ² + c)^p`, unchanged): per-optimiser-step mean, std, min, max, p01, p10, p25, median, p75, p90, p99, the
weighted and unweighted loss, `w_saturation_frac = fraction with |w − 1| < 1e-6`, and `w_near_lower_frac = fraction with w < 1e-4`; logged
every round for A8 and recomputed at the A7-raw best checkpoint on the same fixed validation subset (`scripts/imf_diagnostics_a8.py`,
512 validation windows, (t, r, e) drawn once with seed 1000, identical for every checkpoint). A4 (WildPPG ECG) is included as a
native-scale reference.
**Objective/JVP**: ‖u‖, ‖v_θ‖, ‖V‖, ‖du/dt‖, ‖(t−r)·du/dt‖, ‖V − (e − y)‖, total gradient norm (float64) and max |grad|, at the same
fixed subset; for A8 additionally at the first and best checkpoints where available. Raw checkpoints from A7 are reused (no retraining
of A7).

## 9. Training safety
NaN/Inf does not license any change: no LR change, no clipping, no loss rescaling, no change of the normalisation. Only the micro-batch
may change on OOM, keeping the effective batch at 64. **Failure is a result.**

## 10. Evaluation
Predictions are produced in the trained space and inverse-transformed to mmHg, then evaluated with the **unchanged A7 code**
(`ppg2ecg.evaluation.abp_metrics`): SBP/DBP MAE (window max/min and beat medians), pulse-template correlation, PP error and ratio,
amplitude ratio, upstroke-slope ratio, HF-energy ratio (> 5 Hz), systolic-peak precision/recall/F1 and timing MAE, pulse-interval MAE,
pulse count ratio, RMSE/MAE, peak-region (±150 ms) vs non-peak RMSE, peak-region energy ratio, peak amplitude error, latency, NFE, and the
A7 shuffle penalties. NFE arms: OT-CFM 50/4/1 (20/10/2 also dumped as before), iMF 1 primary + 2/4 diagnostic, MSE 1 forward.

## 11. Hypotheses (frozen)
**H8.1** global normalisation substantially reduces the target/prior scale mismatch (audited in §5 before training).
**H8.2** if A7's failure was scale-driven, normalised iMF leaves the pathological HF/slope regime, shows non-degenerate adaptive-weight
behaviour, and substantially improves physiological metrics.
**H8.3** if the effect is iMF-specific, the MSE proxy and OT-CFM stay qualitatively similar to their raw A7 counterparts after inverse
transformation.
**H8.4** if the normalised MSE proxy stays near A7 MSE performance, the deterministic-predictability observation for PPG→ABP stands.

## 12. Verdict rules (frozen, evaluated in this order)
**SCALE SENSITIVITY CONFIRMED** — all four:
1. normalised iMF-1 improves ≥ 10 % relative over raw iMF-1 on ≥ 3 of {SBP MAE ↓, DBP MAE ↓, template correlation ↑, peak F1 ↑}
   (higher-is-better metrics use the relative gain);
2. HF excess `HF_pred / HF_GT` (raw: 0.550/0.043 = 12.8×) drops by ≥ 50 %;
3. slope-ratio error |ratio − 1| (raw: 5.13) drops by ≥ 50 %;
4. the adaptive-weight / objective diagnostics move to a clearly different, non-degenerate regime (reported with the frozen statistics
   above; judged on the median/p10/p90 of w, δ², ‖u‖ and the residual norm relative to the A7-raw and A4-ECG references).
**PARTIAL SCALE SENSITIVITY** — iMF clearly stabilises but not all four hold, or OT-CFM/MSE also change materially so the iMF-specific
effect cannot be isolated.
**NOT SUPPORTED** — normalised iMF keeps a raw-like pathological profile (excessive HF, bad slope, poor correlation, poor peak F1, high
DBP error).
**CONFOUNDED** — global normalisation materially changes OT-CFM *and* MSE as well, i.e. the intervention changed the task; causal
interpretation withheld.
**"Materially changed" (frozen)**: for the MSE proxy or OT-CFM-1, ≥ 2 of: |Δ template correlation| > 0.10; relative change of SBP or DBP
MAE > 20 %; |Δ peak F1| > 0.10.

## 13. Additional frozen analyses
Conditional-mean similarity (waveform RMSE/MAE/PCC and morphology/amplitude-PP/spectral distances between R, O1, O50, M1 in the
normalised runs; is O1 still closest to R?); pointwise-metric inversion (global, peak-region and non-peak RMSE vs correlation and peak F1
rankings); peak-region analysis with the A7 definition (±150 ms); the A7 raw-vs-normalised comparison table; the deterministic
10/50/90-percentile example windows **reused from A7** (no new selection), plotted in mmHg with raw and normalised rows; the
transport-geometry figure; and the adaptive-weight figure (histogram/ECDF, per-round median/p10/p90, saturation fraction).

## 14. Wording and stop rule
Allowed: "the frozen ECG-tuned iMeanFlow recipe did not transfer to raw-scale ABP"; after results, statements strictly following §12.
Forbidden: "MeanFlow does not work for ABP", "MeanFlow recovery is ECG-specific", "ABP proves generative transport is unnecessary", plus
the standing A5 §13 list. If normalised iMF succeeds while the MSE proxy remains best, the reading is "iMeanFlow's previous failure was
scale-sensitive, but deterministic regression remains sufficient or superior for this ABP protocol"; if it fails, only "the failure cannot
be explained solely by raw target scale". **After A8: STOP** — no respiration experiments, no new iMeanFlow variant, report to the user first.
