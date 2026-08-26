# A7 Pre-registration — Cross-Target Generalisation to PPG→ABP (MIMIC-BP)

Written 2026-08-27 **before any A7 training**; frozen by the commit that introduces it (dataset audit: `docs/A7_ABP_DATASET_AUDIT.md`).
Question: is the one-step structural attenuation → conditional-mean-like behaviour → MeanFlow recovery pattern specific to ECG, or
does it occur for another sharp physiological waveform (arterial blood pressure)? No new method; everything frozen from A0-b/A2/A6.

## 1. Dataset and split (frozen)
MIMIC-BP v2.2 (Harvard Dataverse doi:10.7910/DVN/DBM1NF; ODbL 1.0; raw md5-verified; `data/raw/MIMIC-BP/CHECKSUMS.sha256`), selected by
the pre-stated rule (UCI-BP has no subject identifiers). **Official subject split** train 1,100 / val 195 / test 229
(`data/manifests/split_a7_mimicbp_official.json`, manifest sha256 `c52de946…217a7e`, source-file sha256 recorded inside). Windows:
PENGUIN-faithful 8 s non-overlapping windows within each 30 s segment (3 per segment, 90 per subject) → 99,000 / 17,550 / 20,610 windows;
`data/processed/mimicbp_8s/` (MANIFEST.json, per-file sha256; 137,160 windows, 0 dropped). Validation/test use the same uniform-stride
≤ 4,096-window subsets as A4/A5/A6; training unit = A4's 220-step validation round; max 300 rounds; patience 20; min_delta 1e-4.

## 2. Preprocessing (upstream, unchanged)
PPG: FFT resample 125→128 Hz, band-pass 0.5–4 Hz, per-window z-score, min-max to [−1, 1]. **ABP: resample only — raw mmHg** (upstream
`label_bandpass/zscore/normalize = False`; paper: "no further pre-processing … amplitude carries critical physiological meaning").
No inverse transform; no target statistic at inference; SBP/DBP evaluated in mmHg on the raw target.

## 3. Leakage gate (preflight, must pass before training)
Subject overlap 0; record = subject; exact window-hash overlap 0 across splits; window-local PPG normalisation (no target term);
PPG/ABP window alignment (same indices within a segment); resampling parity (both 1000→1024); no NaN/Inf; ABP mmHg sanity
(min ≥ 20, max ≤ 260 on every split, recorded in provenance). FAIL → no training.

## 4. Frozen models (exact recipes, seed 42, AdamW 1e-3 / wd 0.01 / effective batch 64)
- **A: OT-CFM** — unmodified PENGUIN model, A0-b recipe (fixed-bank validation CFM loss, 4 banks, seed 1000); `a7_otcfm_mimicbp_seed42`.
- **B: Improved MeanFlow** — A2 frozen implementation (h-only cond, forward-mode JVP, micro-batch 32×2, fixed iMF validation objective);
  `a7_imeanflow_mimicbp_seed42`.
- **C: MSE proxy** — A6 full-backbone deterministic regressor (`S5FullBackboneRegressor`, constant state / t = 0.5 exactly as frozen in
  the A6 pre-registration incl. its gradient-flow-selected constant), validation MSE; `a7_mse_fullbackbone_mimicbp_seed42`.
PENGUIN's paper specifies no ABP-specific optimiser setting (audit §4), so the recipe is unchanged. The ECG-specific per-epoch
generation diagnostics are disabled (`--gen-diag-every 0`); selection criteria are objective-based and unchanged. Test data are never
used for selection. Nothing is tuned after results.

## 5. Evaluation arms
OT-CFM 50 / 20 / 10 / 4 / 2 / 1 NFE (Heun 25/10/5/2/1, Euler 1); iMF 1 NFE primary (2 / 4 diagnostic); MSE 1 forward evaluation.
Paired noise seed 0 (identical x0 / e across arms); PPG-shuffle control with derangement seed 1; latency batch 64.

## 6. ABP metrics (`src/ppg2ecg/evaluation/abp_metrics.py`, unit-tested on synthetic pulses)
Primary: (1) SBP MAE and (2) DBP MAE [mmHg] — PENGUIN definition (window max / min) **and** beat-level (median systolic peak / diastolic
trough, the MIMIC-BP label definition); (3) pulse-template correlation (matched systolic peaks, −0.25…+0.55 s); (4) pulse-pressure error
[mmHg]; (5) pulse-pressure ratio pred/GT and waveform amplitude ratio (std); (6) systolic-peak timing MAE [ms] (matched, tol 100 ms);
(7) pulse count ratio; (8) RMSE / MAE [mmHg]; (9) sharpness: median max-upstroke-slope ratio (dP/dt) and HF-energy ratio (> 5 Hz);
(10) latency / NFE. Also systolic-peak precision/recall/F1 (100 ms) and pulse-interval MAE. Peak detection: `scipy.signal.find_peaks`,
min distance 0.3 s, prominence 0.25·(p95 − p5) of the **GT** window applied to both signals (a flat prediction yields no peaks).

## 7. Sharp-event region (frozen)
Peak region = **±150 ms around GT systolic peaks**: peak-region RMSE, non-peak RMSE, peak-region energy ratio (pred/GT variance inside
the region), maximum upstroke slope, peak amplitude error (matched peaks).

## 8. Conditioning sensitivity (frozen)
`shuffle_sbpdbp_penalty = mean(SBP_MAE_shuffled − SBP_MAE_correct, DBP_MAE_shuffled − DBP_MAE_correct)` (window definition) and
`shuffle_morph_penalty = Morph_correct − Morph_shuffled`, same derangement for every arm.

## 9. Hypotheses (frozen)
H7.1 OT-CFM-1 loses morphology / pulse amplitude / upstroke sharpness relative to OT-CFM-50. H7.2 the MSE proxy is low-RMSE with
attenuated sharp structure, and O1 is closer to it (prediction space) than O50 / iMF-1 are. H7.3 iMF-1 recovers morphology / pulse
amplitude / sharpness relative to O1 and approaches O50. H7.4 the attenuated MSE / O1 outputs look better on RMSE/MAE (inversion).

## 10. Structural recovery score
Recovery = (iMF1 − OT1)/(OT50 − OT1) on: template correlation, pulse-pressure fidelity (−|PP ratio − 1|), sharpness (−|slope ratio − 1|);
`ill-conditioned` if |OT50 − OT1| < 0.05 (correlation / ratios). SBP/DBP MAE reported raw, not as recovery.

## 11. Cross-model similarity, inversion, Pareto
As A5: waveform RMSE / MAE / PCC and statistic distances (|Δ amp ratio|, |Δ template corr|, |Δ HF ratio|) between R (MSE), O1, O50, M1;
"closest" = smallest waveform RMSE and ≥ 2/3 statistic votes. Inversion: rank {R, O1, M1, O50} by global RMSE, peak-region RMSE,
non-peak RMSE, template correlation, PP fidelity — YES if R or O1 is best on global RMSE while ranking last on template correlation or PP
fidelity. Pareto: NFE / latency vs SBP MAE, template correlation, |PP ratio − 1|.

## 12. Verdict (frozen)
attenuation(O1) = corr(O1) < corr(O50) − 0.10 and |PP ratio(O1) − 1| > |PP ratio(O50) − 1| + 0.15 and slope ratio(O1) < slope ratio(O50) − 0.15;
attenuation(R) = the same conditions for R vs O50 and RMSE(R) ≤ RMSE(O50); closest(O1→R) as in §11; recovery(iMF1) = ≥ 2 of 3 recovery scores
≥ 0.5 (ill-conditioned entries count as not recovered); inversion as in §11.
- **STRONG CROSS-TARGET SUPPORT**: attenuation(O1), attenuation(R), closest(O1→R), recovery(iMF1) and inversion all hold.
- **PARTIAL CROSS-TARGET SUPPORT**: attenuation(O1) holds and ≥ 2 of the other four hold.
- **NOT GENERALIZED**: attenuation(O1) fails, or ≤ 1 of the other four holds.

## 13. Qualitative windows (deterministic)
10th / 50th / 90th percentile windows of OT-50's mean(SBP MAE, DBP MAE) (window definition) on the test subset; rows PPG / GT ABP /
MSE / OT-50 / OT-4 / OT-1 / iMF-1, y-axis in mmHg, same time axis.

## 14. Wording and stop rule
Allowed / forbidden wording as A5 §13. After A7: STOP — no respiration experiments, no dataset/hyper-parameter changes regardless of
the verdict.
