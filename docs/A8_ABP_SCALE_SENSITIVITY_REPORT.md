# A8 — ABP Target-Scale Sensitivity Control

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats.
> A missed beat therefore incurs no explicit penalty in either metric — it is excluded from the denominator
> rather than scored — so neither metric is monotonic in event coverage: both may rise or fall when the
> matched set changes. Values and specifications here are unchanged; only the labels and their scope are
> made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Pre-registration `docs/A8_ABP_SCALE_SENSITIVITY_PREREGISTRATION.md` (commit `d6ca9dd`, written and pushed before any A8 training).
Artefacts `artifacts/a8_abp_scale_control/` (`normalization.json`, `transport_geometry.json`, `imeanflow_diagnostics.csv`,
`controlled_results.csv`, `prediction_similarity.csv`, `peak_region_analysis.csv`, `summary.json`, `figures/`). Runs
`outputs/a8_{otcfm,imeanflow,mse_fullbackbone}_mimicbp_globalz_seed42`.

## Research question
Did the pathological iMeanFlow result on MIMIC-BP (A7) come from the **raw mmHg target scale and the resulting transport/objective
geometry**, or does it persist after a single fixed train-only affine normalisation? **Answer: it was scale-driven.** With one global
train-only z-transform of the target — nothing else changed — iMeanFlow-1 leaves the pathological regime entirely (pulse-template
correlation 0.140 → **0.876**, HF-energy ratio 0.550 → **0.050** vs GT 0.043, upstroke-slope ratio 6.13 → **1.49**, systolic-peak F1
0.336 → **0.874**, RMSE 32.3 → **18.2 mmHg**), while OT-CFM and the MSE proxy are unchanged within the pre-registered tolerance.
**Frozen verdict: SCALE SENSITIVITY CONFIRMED.**
Separately (and unchanged): the deterministic MSE proxy remains the best model on this protocol, so *iMeanFlow's previous failure was
scale-sensitive, but deterministic regression remains sufficient or superior for this ABP protocol*.

## Frozen intervention
Only the target representation changed: `y_norm = (y_mmHg − mu_train)/sigma_train`, **one scalar pair** from all ABP samples of the
1,100 TRAIN subjects (**mu 77.571767 mmHg, sigma 22.275611 mmHg**, n = 101,376,000). Dataset, split, windows, PPG preprocessing,
architectures, objectives, optimiser, schedule, selection criteria, evaluation code, test subset (3,435 windows), paired noise seed 0 and
derangement seed 1 are identical to A7. Every prediction is inverse-transformed to mmHg before any metric.

## Train-only normalisation and leakage verification
`scripts/compute_abp_norm.py` computes the statistics with a file-open guard that raises if any val/test file is touched
(`leakage_check.ok = true`, 0 val/test files opened). Unit tests (`tests/test_target_norm.py`): forward/inverse round-trip; the transform
is a single global affine (a subject offset survives it — no per-window centring); streaming statistics match numpy on held-out fixtures
and only the given subjects contribute; the shipped JSON covers exactly the 1,100 train subjects with none of the 195 + 229 val/test
subjects; clinical metrics scale exactly with sigma (so they must be, and are, computed after the inverse transform). The analysis
asserts that the A8 test windows are bit-identical to the A7 ones and that the derangement permutation is the same.

## Raw-vs-normalised transport geometry (measured before training, `transport_geometry.json`)
| Quantity (4,096 train windows, prior seed 0) | raw mmHg | global-z |
|---|---:|---:|
| target mean | 79.68 | 0.09 |
| target std | 22.71 | 1.02 |
| ‖y‖ per window | 2608.4 | 30.34 |
| ‖e‖ (prior) | 31.99 | 31.99 |
| **‖y‖ / ‖e‖** | **81.58** | **0.95** |
| ‖y − e‖ (= conditional velocity ‖e − y‖) | 2608.6 | 44.78 |
| prior share of interpolant energy at t = 0.25 / 0.5 / 0.75 | 0.000 / 0.0001 / 0.0013 | 0.096 / 0.488 / 0.896 |
In raw space the interpolant `z_t = (1−t)y + t·e` is the target for essentially all t — the noise contributes 0.01 % of the energy even
at t = 0.5 — so the "transport" the model must learn is dominated by a large constant offset, and the conditional velocity is ~80× the
prior scale. Normalisation restores the intended geometry (H8.1 ✓, quantified before any model was trained).

## Training
| Run | Rounds (best) | Selection metric | Time | Peak VRAM |
|---|---|---|---|---|
| OT-CFM raw (A7) → global-z (A8) | 117 (97) → **207 (187)** | 6.49 → 0.1224 (different units) | 2.5 h → 4.5 h | 19.2 GiB |
| iMeanFlow raw → global-z | 70 (50) → **88 (68)** | 112.5 → 0.1213 | 3.5 h → 4.3 h | 17.7 GiB |
| MSE proxy raw → global-z | 66 (46) → **68 (48)** | 219.0 → 0.4368 mmHg²-equivalent | 1.2 h → 1.2 h | 19.2 GiB |
No NaN/Inf, no protocol deviation, no tuning; micro-batch unchanged (32 × 2 for iMF).

## iMeanFlow objective diagnostics (fixed 512-window validation subset, same (t, r, e) draw, seed 1000)
| | δ² (sum over 1024 dims) | w median | w p10 / p90 | w saturation (\|w−1\|<1e-6) | ‖u‖ | residual ‖V−(e−y)‖ | grad norm | target std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A7 raw-mmHg (best)** | 120,334 | 1.90e-05 | 4.3e-06 / 1.6e-04 | 0.000 | 2511 | 277.5 | 5.11 | 21.79 |
| **A8 global-z (best)** | **130.0** | **1.26e-02** | 5.1e-03 / 2.6e-02 | 0.000 | 42.5 | 10.11 | 7.41 | 0.978 |
| A4 ECG reference (native scale) | 124.5 | 9.77e-03 | 5.5e-03 / 1.9e-02 | 0.000 | 35.6 | 10.63 | 2.38 | 0.298 |
The normalised ABP run sits in the **same objective regime as the ECG runs** (δ² 130 vs 124.5; w median 1.3e-2 vs 9.8e-3), three orders
of magnitude away from the raw-mmHg run. Per-round weight percentiles for the A8 run are in `training_log.csv`
(`w_mean/median/p01…p99/min/max/std/saturation`), plotted in `figures/a8_adaptive_weight.png`. **Correction carried from A7:** the
weighted loss ≈ 1.0 in *all* runs including ECG (it is ≈ 1 whenever δ² ≫ c = 0.01), so it never distinguished the datasets; the
distinguishing quantities are δ², w and ‖u‖ above.

## Controlled physiological results (3,435 test windows, mmHg, all inverse-transformed)
| Model | Eval | Scale | SBP MAE ↓ | DBP MAE ↓ | Morph ↑ | PP ratio | Slope ratio | HF (GT .043) | Peak F1 ↑ | RMSE ↓ | shuffle SBP/DBP | shuffle morph |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE proxy | 1 fwd | raw | 14.31 | 8.72 | 0.929 | 0.96 | 0.91 | 0.022 | 0.945 | 13.10 | +1.93 | +0.171 |
| MSE proxy | 1 fwd | **global-z** | **14.05** | **8.69** | 0.929 | 0.97 | 0.91 | 0.019 | 0.948 | **13.03** | +1.77 | +0.169 |
| OT-CFM | 1 NFE | raw | 15.09 | 9.51 | 0.904 | 0.92 | 0.93 | 0.024 | 0.913 | 14.64 | +0.27 | +0.177 |
| OT-CFM | 1 NFE | **global-z** | **13.95** | 8.98 | **0.934** | 1.03 | 1.00 | 0.022 | **0.947** | 13.43 | +1.62 | +0.174 |
| OT-CFM | 4 NFE | raw | 14.89 | 9.39 | 0.909 | 0.94 | 0.97 | 0.022 | 0.919 | 14.69 | +0.28 | +0.173 |
| OT-CFM | 4 NFE | global-z | 16.17 | 15.61 | 0.825 | 1.11 | 3.62 | 0.164 | 0.706 | 15.69 | +1.38 | +0.488 |
| OT-CFM | 50 NFE | raw | 15.94 | 9.80 | 0.884 | 1.05 | 1.20 | 0.037 | 0.883 | 16.12 | +0.37 | +0.155 |
| OT-CFM | 50 NFE | global-z | 18.08 | 12.30 | 0.922 | 1.03 | 1.19 | 0.036 | 0.901 | 17.79 | +1.37 | +0.172 |
| iMeanFlow | 1 NFE | raw | 16.28 | 21.82 | 0.140 | 1.30 | 6.13 | 0.550 | 0.336 | 32.27 | +0.05 | +0.004 |
| iMeanFlow | 1 NFE | **global-z** | 18.75 | **11.98** | **0.876** | 1.07 | **1.49** | **0.050** | **0.874** | **18.20** | +0.52 | +0.173 |
(2/10/20 NFE and iMF 2/4 NFE are in `controlled_results.csv`.)

## Raw → normalised iMeanFlow change (frozen §12 criteria)
| Criterion | Raw | Global-z | Change | Threshold | Pass |
|---|---:|---:|---:|---|---|
| SBP MAE | 16.28 | 18.75 | **−15.1 %** (worse) | ≥ +10 % | ✗ |
| DBP MAE | 21.82 | 11.98 | **+45.1 %** | ≥ +10 % | ✓ |
| Template correlation | 0.140 | 0.876 | **+525 %** | ≥ +10 % | ✓ |
| Peak F1 | 0.336 | 0.874 | **+160 %** | ≥ +10 % | ✓ |
| ≥ 3 of 4 improved | | | 3 / 4 | ≥ 3 | **✓** |
| HF excess (HF_pred/HF_GT) | 12.88× | **1.17×** | −90.9 % | ≥ 50 % drop | **✓** |
| Slope-ratio error \|ratio−1\| | 5.13 | **0.489** | −90.5 % | ≥ 50 % drop | **✓** |
| Objective diagnostics regime | δ² 1.2e5, w 1.9e-5 | δ² 130, w 1.3e-2 | ECG-like | non-degenerate shift | **✓** |
All four conditions hold → **SCALE SENSITIVITY CONFIRMED**. The single failing sub-metric (SBP MAE) is reported openly: the normalised
iMF reconstructs systolic peaks with the right sharpness but overshoots their height on some beats (PP ratio 1.07), which costs window-max
accuracy even though every structural metric improves.

## Baseline stability (§12 "materially changed" = ≥ 2 of: |Δmorph| > 0.10, rel. SBP or DBP change > 20 %, |ΔF1| > 0.10)
| Model | Δ morph | rel. SBP | rel. DBP | Δ peak F1 | flags | materially changed |
|---|---:|---:|---:|---:|---:|---|
| MSE proxy | 0.0003 | 1.8 % | 0.3 % | 0.002 | 0 | **No** |
| OT-CFM 1 | 0.029 | 7.6 % | 5.5 % | 0.033 | 0 | **No** |
| OT-CFM 50 | 0.038 | 13.4 % | 25.5 % | 0.018 | 1 | No |
The intervention did not change the task: the deterministic proxy is essentially identical in both scales, and OT-CFM-1 changes only
mildly (slightly better). Not CONFOUNDED. One genuine anomaly is recorded rather than smoothed over: in the normalised run **OT-CFM at
4 NFE is worse than at 1 and at 50 NFE** (slope 3.62, HF 0.164, F1 0.706) — the same intermediate-NFE instability the ECG experiments
showed at 2 NFE (A0-b/A3/A4 `heun1`), now appearing on ABP after normalisation.

## Peak-region analysis (GT systolic peaks ±150 ms, A7 definition unchanged)
| Model | RMSE peak (raw → z) | RMSE non-peak (raw → z) | peak-region energy ratio (raw → z) |
|---|---|---|---|
| MSE proxy | 15.20 → 15.19 | 10.60 → 10.54 | 1.13 → 1.12 |
| OT-CFM 1 | 16.71 → 15.55 | 12.21 → 11.00 | 1.09 → 1.31 |
| OT-CFM 50 | 19.01 → 20.44 | 12.99 → 14.73 | 1.67 → 1.44 |
| iMeanFlow 1 | 40.90 → **21.43** | 24.50 → **14.74** | 2.10 → **1.58** |
No model under-fills the systolic region on ABP in either scale (energy ratio ≥ 1.1); normalisation halves iMF's over-sharpening.

## Conditional-mean similarity
| Distance from the MSE proxy | raw | global-z |
|---|---|---|
| R–O1 waveform RMSE / PCC | **7.19 / 0.940** | **5.60 / 0.976** |
| R–O50 | 9.73 / 0.866 | 12.81 / 0.909 |
| R–M1 | 29.57 / −0.005 | 13.11 / 0.867 |
| closest to R (RMSE, statistic votes) | O1 (3/3) | O1 (2/3) |
OT-CFM-1 remains the closest model to the deterministic proxy in both scales, and after normalisation it is *very* close (PCC 0.976).
iMeanFlow-1 moves from an outlier (PCC ≈ 0) into the same neighbourhood (PCC 0.867) — further evidence that the raw-scale run was not
modelling the target at all.

## Pointwise metric behaviour
Ranking by global RMSE, peak-region RMSE and non-peak RMSE is `R < O1 < O50 < M1` in **both** scales, and the physiological rankings
agree (peak F1 identical order; template correlation identical except that in global-z O1 edges past R, 0.934 vs 0.929).
**No pointwise-error inversion on ABP in either scale** — the ECG-specific metric failure does not appear here, before or after the
intervention.

## Hypothesis tests
- **H8.1** ✓ — norm ratio 81.58 → 0.95, ‖y − e‖ 2608 → 44.8, prior share of interpolant energy at t = 0.5 0.0001 → 0.488 (measured before training).
- **H8.2** ✓ — normalised iMF leaves the pathological HF/slope regime (12.9× → 1.17× GT HF; slope error 5.13 → 0.49), its objective
  statistics move into the ECG regime, and 3 of 4 pre-registered physiological metrics improve by ≥ 10 %.
- **H8.3** ✓ — the MSE proxy and OT-CFM stay qualitatively identical after inverse transformation (0 material-change flags for R and O1).
- **H8.4** ✓ — the normalised MSE proxy matches its raw counterpart (SBP 14.05 vs 14.31, RMSE 13.03 vs 13.10) and remains the best or
  joint-best model, so the deterministic-predictability observation for PPG→ABP stands.

## Verdict
**SCALE SENSITIVITY CONFIRMED.**

## Alternative explanations (kept open)
1. **Optimisation, not geometry**: normalisation also changes the loss curvature/effective learning rate, so part of the gain may be
   ordinary optimisation improvement rather than transport geometry per se. Evidence for a real geometric component: the OT-CFM run needed
   ~2× more rounds to converge in the normalised space while ending at nearly the same physiology, whereas iMF changed qualitatively — the
   MeanFlow objective involves the interval term `(t−r)·du/dt`, which is directly tied to the interpolant path that raw scaling degenerates.
2. **Selection-metric change**: each objective's validation metric is computed in the trained space, so the *stopping point* differs
   between scales; the pre-registered criteria were kept identical in form, and the MSE proxy's near-identical result shows this did not
   drive the effect.
3. **Residual SBP overshoot**: normalised iMF is worse on SBP MAE; a genuine remaining weakness, not a scale artefact.
4. **Single seed / single dataset**: the entire A8 comparison rests on one seed per model and one ABP dataset.

## Limitations
One seed per model, one dataset, one test subset (3,435 windows); ICU/arterial-line population; the affine transform is the only
intervention tested (no per-subject or robust scaling was tried, by protocol); the A7 raw runs were not retrained; comparisons across
scales inherit different selection-metric units; the intermediate-NFE anomaly (OT-CFM 4 NFE, normalised) is reported but not explained.

## Updated scientific interpretation
- **What we can now say.** The A7 iMeanFlow failure on ABP was caused by the raw target scale / transport geometry, not by anything
  intrinsic to the target: one global train-only affine transform restores ECG-like objective statistics and near-ECG-quality one-step ABP
  reconstruction. The A7 headline (no one-step structural attenuation on ABP, no pointwise inversion, deterministic proxy best) is
  **unchanged and now better supported**: it holds identically in both target representations.
- **What we still cannot say.** That ABP benefits from a generative one-step model: with normalisation the MSE proxy (SBP 14.05, RMSE
  13.03) and OT-CFM-1 (13.95, 13.43) are statistically the same and both beat 50 NFE and iMF-1 — so a deterministic regressor remains
  sufficient or superior for this ABP protocol. Nor can we say anything general about MeanFlow on ABP beyond "the frozen ECG-tuned recipe
  did not transfer to raw-scale targets, and does transfer after a single global affine normalisation".

## Recommended next experiment
Pre-register a **target-representation control on ECG** (the mirror of A8): train the frozen OT-CFM/iMF/MSE trio on ECG targets that are
*de-normalised* to a raw physical scale. If ECG's one-step attenuation and pointwise inversion survive de-normalisation, the ECG effect is
genuinely about the conditional distribution rather than the representation — the single strongest remaining threat to the A2–A6 line.
(Respiration as a smooth-target control remains available but is deliberately deferred.)
