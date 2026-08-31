# A7 — Cross-Target Generalisation to PPG→ABP (MIMIC-BP)

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats.
> A missed beat therefore incurs no explicit penalty in either metric — it is excluded from the denominator
> rather than scored — so neither metric is monotonic in event coverage: both may rise or fall when the
> matched set changes. Values and specifications here are unchanged; only the labels and their scope are
> made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Pre-registration `docs/A7_ABP_PREREGISTRATION.md` and dataset audit `docs/A7_ABP_DATASET_AUDIT.md` (commit `84223f0`, both before any
A7 training). Artefacts `artifacts/a7_abp_generalization/` (`abp_metrics.csv`, `cross_model_similarity.csv`, `peak_region_analysis.csv`,
`pareto.csv`, `summary.json`, `figures/`). Runs `outputs/a7_{otcfm,imeanflow,mse_fullbackbone}_mimicbp_seed42`.

## 1. Answer
**Does the one-step structural attenuation → conditional-mean-like behaviour → MeanFlow recovery pattern occur for another sharp
physiological waveform?** **No — on PPG→ABP none of the three parts reproduces.** OT-CFM at 1 NFE loses nothing relative to 50 NFE
(pulse-template correlation 0.904 vs 0.884; SBP MAE 15.09 vs 15.94 mmHg; RMSE 14.6 vs 16.1); the MSE conditional-mean proxy is not an
attenuated shortcut but **the best model on every ABP metric** (SBP 14.31 / DBP 8.72 mmHg, correlation 0.929, systolic-peak F1 0.945,
RMSE 13.10); and Improved MeanFlow at 1 NFE **degrades** the target (correlation 0.140, HF-energy ratio 0.550 vs GT 0.043, RMSE 32.3).
**Frozen verdict: NOT GENERALIZED.** The ECG finding stands as an ECG(-like) finding, not a universal property of one-step conditional
flows.

## 2. Setup (frozen; audit + prereg)
MIMIC-BP v2.2 (ODbL; 1,524 subjects; official subject-disjoint split 1,100 / 195 / 229; ABP raw **mmHg**, PPG band-pass + per-window
z-score/min-max, both resampled 125→128 Hz; 8 s windows = 99,000 / 17,550 / 20,610; test evaluated on the pre-registered uniform
3,435-window subset). Preflight gate passed for every run (subject/window disjointness, window-local PPG normalisation, no target
statistic at inference, mmHg sanity 28–189, no NaN/Inf, hashes). Three models, identical recipe (seed 42, AdamW 1e-3 / wd 0.01 / batch
64, 220-step rounds, patience 20, min_delta 1e-4, test never used for selection):
| Model | Selection | Rounds (best) | Time | Peak VRAM |
|---|---|---|---|---|
| OT-CFM (A0-b recipe) | fixed-bank CFM loss (4 banks, seed 1000) → 6.49 | 117 (97) | 2.5 h | 19.2 GiB |
| iMeanFlow (A2 recipe, h-only) | fixed iMF objective → 112.49 | 70 (50) | 3.5 h | 17.7 GiB |
| MSE proxy (A6 full backbone, x 0.1 / t 0.5 / cond 0.05·E(t)) | validation MSE → 218.96 mmHg² | 66 (46) | 1.2 h | 19.2 GiB |
Paired noise seed 0 (identical x0 / e for OT-CFM and iMF), PPG derangement seed 1, latency at batch 64.

## 3. Controlled results (3,435 test windows, mmHg)
| Model | Eval | SBP MAE ↓ | DBP MAE ↓ | beat SBP/DBP | Morph ↑ | PP ratio | Amp | Upstroke slope ratio | HF (>5 Hz) | Peak F1 ↑ | Peak timing MAE | RMSE ↓ | Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **MSE proxy** | 1 fwd | **14.31** | **8.72** | 14.05 / 8.76 | **0.929** | 0.96 | 0.96 | 0.91 | 0.022 | **0.945** | **31.0 ms** | **13.10** | 84 ms |
| OT-CFM | 1 NFE | 15.09 | 9.51 | 15.32 / 9.44 | 0.904 | 0.92 | 0.93 | 0.93 | 0.024 | 0.913 | 35.1 ms | 14.64 | 85 ms |
| OT-CFM | 2 NFE | 14.98 | 9.44 | 15.20 / 9.49 | 0.908 | 0.92 | 0.94 | 0.94 | 0.023 | 0.918 | 34.7 ms | 14.65 | 170 ms |
| OT-CFM | 4 NFE | 14.89 | 9.39 | 15.15 / 9.44 | 0.909 | 0.94 | 0.95 | 0.97 | 0.022 | 0.919 | 34.6 ms | 14.69 | 329 ms |
| OT-CFM | 10 NFE | 14.80 | 9.35 | 15.12 / 9.36 | 0.904 | 0.96 | 0.97 | 1.01 | 0.024 | 0.915 | 35.0 ms | 14.83 | 836 ms |
| OT-CFM | 20 NFE | 14.89 | 9.47 | 15.08 / 9.34 | 0.896 | 0.99 | 1.00 | 1.09 | 0.030 | 0.906 | 35.5 ms | 15.15 | 1671 ms |
| OT-CFM | 50 NFE | 15.94 | 9.80 | 14.97 / 9.45 | 0.884 | 1.05 | 1.06 | 1.20 | 0.037 | 0.883 | 36.6 ms | 16.12 | 4167 ms |
| iMeanFlow | 1 NFE | 16.28 | 21.82 | 15.22 / 18.45 | 0.140 | 1.30 | 1.22 | 6.13 | **0.550** | 0.336 | 48.1 ms | 32.27 | 84 ms |
| iMeanFlow | 2 NFE | 14.82 | 15.23 | 15.34 / 13.27 | 0.204 | 1.15 | 1.12 | 3.10 | 0.444 | 0.331 | 47.9 ms | 28.83 | 170 ms |
| iMeanFlow | 4 NFE | 14.84 | 12.11 | 14.72 / 10.44 | 0.241 | 1.08 | 1.05 | 2.30 | 0.406 | 0.334 | 47.5 ms | 26.63 | 335 ms |
GT HF-energy ratio 0.043; GT window SBP/DBP 116.2 / 57.5 mmHg. No arm produced a peak-free window. Upstream reference: PENGUIN reports
SBP 17.43 / DBP 11.34 mmHg on MIMIC-BP with a different (glob-order) split — our 50-NFE reproduction is in the same range.

## 4. Hypotheses (frozen §9) — all four fail
- **H7.1 (OT-1 attenuates)** ✗. Relative to 50 NFE, 1 NFE has *higher* template correlation (+0.020), lower SBP/DBP MAE (−0.85 / −0.29),
  lower RMSE (−1.5) and a slope ratio nearer 1 (0.93 vs 1.20). The NFE trend is monotone but in the opposite direction to ECG: more steps
  add amplitude and high-frequency energy (PP ratio 0.92 → 1.05, HF 0.024 → 0.037, slope 0.93 → 1.20), i.e. **50 NFE slightly overshoots**
  sharpness while 1 NFE is marginally conservative.
- **H7.2 (proxy attenuated, O1 closest)** ✗ / ✓-but-trivial. The MSE proxy is *not* attenuated (it beats every generative arm on SBP, DBP,
  correlation, F1 and RMSE; slope ratio 0.91, PP 0.96). The "closest" term is satisfied (waveform RMSE R–O1 7.19 < R–O50 9.73 < R–M1 29.57;
  3/3 statistic votes; PCC 0.94), but it carries no attenuation meaning here: {R, O1 … O50} form one tight cluster (O1–O50 distance 6.08,
  even smaller than R–O1) and iMF-1 is the outlier.
- **H7.3 (MeanFlow recovery)** ✗. iMF-1 is worse than OT-1 on every structural metric. Recovery scores: morphology and PP fidelity are
  **ill-conditioned** (|OT50 − OT1| = 0.020 and 0.027 < 0.05); the sharpness score evaluates to 37.8, a **degenerate value** — its
  denominator is negative because OT-50 is *less* faithful than OT-1 — and must not be read as recovery. 1 of 3 ≥ 0.5 → not recovered.
- **H7.4 (pointwise inversion)** ✗. The ranking by RMSE (R < O1 < O50 < M1) is identical to the ranking by template correlation and by
  SBP MAE, and the peak-region and non-peak RMSE orders agree. Pointwise error and physiology agree on ABP.

## 5. Sharp-event region (±150 ms around GT systolic peaks) and conditioning
| Model | RMSE peak | RMSE non-peak | peak/non-peak | peak-region energy ratio | peak amplitude MAE | shuffle SBP/DBP penalty | shuffle morph penalty |
|---|---|---|---|---|---|---|---|
| MSE proxy | 15.20 | 10.60 | 1.43 | 1.13 | 14.17 | **+1.93 mmHg** | +0.171 |
| OT-CFM 1 | 16.71 | 12.21 | 1.37 | 1.09 | 15.65 | +0.27 | +0.177 |
| OT-CFM 50 | 19.01 | 12.99 | 1.46 | 1.67 | 15.93 | +0.37 | +0.155 |
| iMeanFlow 1 | 40.90 | 24.50 | 1.67 | 2.10 | 20.30 | +0.05 | +0.004 |
Every model keeps ≥ 100 % of the GT variance inside the peak regions (OT-50 167 %, iMF-1 210 % — over-sharpening, not attenuation), the
exact opposite of the ECG QRS result (12–35 % for the one-step arms). Conditioning: the MSE proxy has by far the largest SBP/DBP shuffle
penalty; iMF-1's penalties are ≈ 0, i.e. its output is nearly PPG-independent noise.

## 6. Quality–compute Pareto
Pareto-optimal (NFE × {SBP MAE, correlation, |PP − 1|}): the **MSE proxy (1 forward)**, OT-CFM 4/10/20 NFE. OT-CFM 50 NFE is dominated
(by OT-20, OT-10 and the proxy); every iMeanFlow arm is dominated. On this target the generative machinery buys nothing: one deterministic
forward pass is both the cheapest and the most accurate option (`figures/a7_pareto.png`).

## 7. Qualitative (`figures/a7_examples.png`, deterministic 10/50/90 % windows of OT-50's mean SBP/DBP error, y in mmHg)
MSE proxy, OT-50 and OT-1 all draw full-height, correctly timed pulses with the dicrotic notch; the proxy tracks the diastolic decay
slightly better and OT-50 adds visible high-frequency ripple. iMeanFlow-1 produces dense spiky noise around the correct pressure range.

## 8. Interpretation (within the allowed wording)
- The attenuation we documented for ECG is **not** a generic property of one-step OT-CFM. Its appearance depends on how much of the
  target is predictable from PPG: on MIMIC-BP the ABP pulse is nearly determined by the PPG pulse (a deterministic MSE predictor reaches
  correlation 0.93 and peak F1 0.945), so the conditional-mean-like solution *is* the sharp waveform and one step suffices. On ECG the
  conditional mean is far from any single plausible ECG (best morphology 0.65–0.68 even at 50 NFE), so collapsing to it destroys the QRS.
- Consistently with this, on ABP the low-NFE arms are the *conservative* ones and extra steps add sharpness that overshoots the target
  (HF 0.024 → 0.037 vs GT 0.043; slope ratio 0.93 → 1.20), which costs SBP accuracy at 50 NFE.
- Improved MeanFlow, which restored structure on ECG at one evaluation, injects high-frequency energy on a smooth target (HF 12.9× GT at
  1 NFE, decreasing to 9.5× at 4 NFE) and loses PPG conditioning almost entirely. We report this as an empirical failure on this target
  under our frozen recipe; we do not claim it is a property of the objective in general (see limitations).
- Practical reading: for PPG→ABP with this backbone, a deterministic regressor is the right tool; the ECG result should be framed as
  target-dependent, tied to how multi-valued the PPG→target relation is.
- Not claimed: that OT-CFM mathematically converges to the conditional mean; that MeanFlow solves multimodality; that ECG is inherently
  multimodal; that WildPPG proves phase diversity.

## 9. Alternative explanations for the ABP null result
1. **Predictability of the target** (favoured): the PPG→ABP map is close to deterministic at window level, so mean-seeking costs nothing.
2. **Preprocessing asymmetry**: ABP is raw mmHg (no per-window normalisation), so a large low-frequency/offset component dominates the
   loss and the metrics; ECG windows are z-scored and min-max scaled, which amplifies the relative weight of the QRS. Our metric set
   (correlation, PP ratio, slope ratio, peak F1, peak-region RMSE) is scale-free or peak-anchored and shows the same conclusion, but the
   two tasks are not perfectly comparable.
3. **Spectral content**: the ABP GT has 4.3 % of its energy above 5 Hz versus 20–32 % above 15 Hz for ECG; there is simply less sharp
   structure to lose.
4. **Target scale / transport geometry**: the frozen A2 recipe was tuned for z-scored targets in [−1, 1]; MIMIC-BP targets are raw mmHg
   (mean ≈ 77.6, std ≈ 16.7) against a standard-normal prior, so the interpolant `z_t = (1−t)x + t·e` is dominated by the target for
   almost all t and the conditional velocity `e − x ≈ −x` is two orders of magnitude larger than in the ECG runs. This is a plausible
   cause of the iMF failure and a limitation of the controlled design, not evidence about MeanFlow in general. **A8 tests it directly.**

**Correction (2026-08-27, made during the A8 preflight, after this report was first committed).** An earlier version of this section
stated that "the adaptive weight is effectively 1.0 throughout training (observed `lossW 1.0000`), so its variance-normalising role is
lost". That was a misreading of the log: `train_loss_weighted` is the *weighted loss* `mean(δ² · w)`, which is ≈ 1 by construction
whenever `δ² ≫ c = 0.01` — and it is also ≈ 1.0000 in the ECG runs (A2: 0.99996, A4: 0.99995). The observation therefore does **not**
distinguish ABP from ECG and is withdrawn. What does differ is the magnitude of `δ²` itself (per-element MSE 0.17 on ECG vs 136.9 on
raw-mmHg ABP, i.e. `δ²` ≈ 176 vs ≈ 140,000 summed over 1024 dims) and hence the target/prior scale ratio; the corrected statement is the
transport-geometry one above. No number in this report changes.

## 10. Limitations
- One seed per model; one dataset for ABP; 3,435-window test subset; ICU/arterial-line population with the known MIMIC-BP caveats.
- Raw-mmHg targets change the loss scale (train MSE 200–1300 mmHg²) but the recipe was kept frozen by protocol; no tuning was done.
- SBP/DBP are computed as window max/min (PENGUIN definition) and as beat medians (MIMIC-BP definition); both agree.
- iMF's failure is confounded with the objective's scale sensitivity (§9.4); a rescaled variant was **not** tried (no new methods).
- Every result uses the official split; no dataset, split, metric or hyper-parameter was changed after seeing results.
