# A5 — Conditional-Mean Control: does one-step OT-CFM behave like an MSE regressor?

Pre-registration: `docs/A5_CONDITIONAL_MEAN_CONTROL_PREREGISTRATION.md` (commit `cc28ad9`, Amendment 1 in `d961000`, both before the
corresponding results). Analysis artefacts: `artifacts/a5_conditional_mean_control/` (`summary.json`, `cross_model_similarity.csv`,
`qrs_region_analysis.csv`, `pareto.csv`, `figures/`). Runs: `outputs/a5a_mse_regressor_dalia_testS2_seed42`,
`outputs/a5b_mse_regressor_dalia_testS1_seed42`, `outputs/a5c_mse_regressor_wildppg_seed42` (config, provenance, logs, metrics; checkpoints
and predictions kept locally, not in git). Generated 2026-08-26/27.

## 1. Summary
A deterministic **MSE regressor** built from the *same* PENGUIN Flow-SSM/S5 backbone (generative inputs removed, §3) was trained on the
three frozen splits of A2/A3/A4 and compared, on identical test windows, with the frozen OT-CFM (1 and 50 NFE) and Improved-MeanFlow
(1 NFE) predictions. On every dataset the regressor (R) reproduces the OT-CFM 1-NFE signature — the **lowest RMSE/MAE of all four models
together with the strongest amplitude and morphology attenuation** — and OT-CFM 1-NFE (O1) is by far the model closest to R
(prediction-to-prediction RMSE 0.08–0.13 vs 0.26–0.35 for OT-50/iMF-1; O1 wins all three statistic-distance votes on all three datasets).
On beat-synchronised WildPPG the regressor keeps beat timing (R-peak F1 0.436 vs 0.440 for OT-50) and PPG dependence (shuffle gain 5.70 vs
7.16) while losing amplitude (0.25) and sharpness (template correlation 0.33) — exactly the pattern OT-CFM 1-NFE shows there. iMeanFlow-1
restores amplitude/morphology relative to R but places beats less precisely (F1 0.385, RR-interval MAE 25.7 ms vs 16.7 ms).
**Pre-registered verdict: STRONG SUPPORT** (all §12 conditions met on all three datasets). Allowed reading: OT-CFM 1-NFE *empirically
approaches the behaviour of an MSE-trained conditional-mean proxy*; on temporally aligned data that proxy retains rhythm and averages away
sharp morphology. Not claimed: convergence to the exact conditional mean, multimodality, or "phase diversity".

## 2. Hypothesis and questions (frozen, §1–2 of the pre-registration)
RQ1 does a direct MSE regressor show the attenuated low-RMSE behaviour of OT-CFM 1-NFE? — **Yes, on all three datasets.**
RQ2 is OT-CFM 1-NFE closer to the regressor than iMF-1 or OT-50? — **Yes, on all three datasets, on every distance.**
RQ3 does temporal alignment decide what the conditional-mean-like solution keeps? — **Yes:** unsynchronised DaLiA → beat-free, near-flat
output (beats/reference 0.07 / 0.30); synchronised WildPPG → aligned, attenuated beats (F1 0.44, beats 0.74).
H1 ✓ (3/3), H2 ✓ (DaLiA 2/2), H3 ✓ (WildPPG), H4 ✓ (WildPPG). Rule terms per dataset are in `summary.json → datasets.*.terms`.

## 3. Model, training, data (as pre-registered + Amendment 1)
- `S5ConditionalMeanRegressor` (`src/ppg2ecg/models/regressor.py`): unmodified upstream PENGUIN model with `pre_conv_target` (x_t stem)
  and `timestep_embedder` deleted; conditioning vector = 0; the target stream receives a **learned constant state token** (128 params).
  Total **3,990,787** parameters (= 4,568,707 − 528,640 − 49,408 + 128); 264,194 never called (`cross_attn`, `revin`, as upstream) and
  819,200 adaLN `Linear.weight`s structurally inactive with cond = 0 → **effective 2,907,393** (generative arms: 4,304,513 effective).
- **Amendment 1 (dead-start).** The originally pre-registered zero target-stream input cannot train: block outputs are identically 0 and
  upstream zero-initialises `final_layer.linear.weight`, so only `final_layer.linear.bias` ever receives gradient (unit-tested). The first
  A5a run therefore converged to a constant (−0.348, per-window std 2.5e-7, RMSE 0.290, val-MSE plateau 0.0983 from epoch 9) and A5b was
  stopped at epoch 1; both are archived in `outputs/aborted/*_zero_state_deadstart/` and are **not** results. The token restores gradient
  flow to every active parameter within ≤ 5 steps (`tests/test_regressor.py`). Hypotheses, thresholds, wording and every other protocol
  item were unchanged; the amendment was committed (`d961000`) before the amended runs started.
- Loss: plain MSE ("MSE conditional-mean proxy"). AdamW 1e-3 / wd 0.01 / batch 64 / fp32 / seed 42; selection = deterministic validation
  MSE (full S11 set for A5a/b; A4's 4,096-window uniform subset and 220-step rounds for A5c), min_delta 1e-4, patience 20, max 300.
- Runs: A5a 40 epochs (best 20, val MSE 0.0881, 45 min, 18.3 GiB); A5b 31 (best 11, 0.0878, 35 min); A5c 54 rounds (best 34, 0.0854,
  64 min, 20.6 GiB). Inputs/targets are bit-identical to those of the paired generative arms (asserted in `analyze_a5.py`).
- Data: `split_p0_holdout_seed42` (test S2), `split_a3_testS1_valS11` (test S1), `split_a4_wildppg_seed42` (test kjd, ssx; 3,907-window
  subset), processed `v0_8s` / `wildppg_8s`; preflight OK (subject/window disjointness, window-local normalisation, hashes) in each run.

## 4. Main metrics (test sets; R = regressor, O1 = OT-CFM Euler-1, O50 = OT-CFM Heun-25, M1 = iMeanFlow-1)
| Dataset | Model | HR err ↓ | morph ↑ | amp | gain ↑ | beats/ref | RMSE ↓ | MAE ↓ | HF ratio | R-peak F1 | RR MAE ms | latency (b64) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DaLiA S2 | **R** | 35.71 | 0.160 | **0.063** | 2.37 | 0.07 | **0.289** | **0.200** | 0.001 | 0.012 | — | 80 ms |
| | O1 | 41.96 | 0.217 | 0.145 | 0.24 | — | 0.304 | 0.221 | 0.150 | 0.079 | — | 82 ms |
| | O50 | 8.08 | 0.650 | 0.949 | 5.69 | — | 0.435 | 0.354 | 0.269 | 0.140 | — | 4171 ms |
| | M1 | 9.58 | 0.595 | 0.896 | 4.47 | — | 0.443 | 0.366 | 0.252 | 0.139 | — | 82 ms |
| DaLiA S1 | **R** | 32.34 | 0.148 | **0.052** | 1.32 | 0.30 | **0.318** | **0.251** | 0.001 | 0.054 | — | 80 ms |
| | O1 | 35.23 | 0.168 | 0.207 | 0.28 | — | 0.347 | 0.282 | 0.102 | 0.069 | — | 82 ms |
| | O50 | 8.16 | 0.683 | 0.870 | 8.77 | — | 0.448 | 0.353 | 0.254 | 0.125 | — | 4156 ms |
| | M1 | 11.96 | 0.581 | 0.711 | 4.78 | — | 0.449 | 0.372 | 0.303 | 0.124 | — | 82 ms |
| WildPPG | **R** | 19.22 | 0.331 | **0.254** | 5.70 | 0.74 | **0.343** | **0.288** | 0.005 | 0.436 (P .48 / R .42) | 16.7 | 80 ms |
| | O1 | 15.59 | 0.379 | 0.321 | 6.64 | — | 0.355 | 0.301 | 0.065 | 0.481 (.52 / .46) | 15.1 | 81 ms |
| | O50 | 9.43 | 0.670 | 0.977 | 7.16 | — | 0.440 | 0.369 | 0.263 | 0.440 (.44 / .44) | 21.2 | 4159 ms |
| | M1 | 11.85 | 0.551 | 1.039 | 4.29 | — | 0.485 | 0.414 | 0.220 | 0.385 (.40 / .38) | 25.7 | 81 ms |
GT HF-energy ratio (> 15 Hz): 0.323 / 0.198 / 0.280. DaLiA beat-level columns (F1, RR) are uninterpretable (devices not
beat-synchronised) and shown only for completeness; on DaLiA 65 % (S2) / 33 % (S1) of the regressor's windows contain no detectable beat.
The regressor's HF ratio (0.001–0.005) shows that MSE removes essentially all high-frequency (QRS) energy; OT-1 keeps a little (0.07–0.15).

## 5. Prediction-to-prediction similarity (§8; identical windows; smaller = closer, PCC larger = closer)
| Dataset | pair | waveform RMSE | MAE | PCC | Δamp | Δmorph | ΔHF |
|---|---|---|---|---|---|---|---|
| DaLiA S2 | **R–O1** | 0.085 | 0.078 | 0.103 | 0.08 | 0.06 | 0.149 |
| DaLiA S2 | R–O50 | 0.310 | 0.256 | 0.009 | 0.89 | 0.49 | 0.268 |
| DaLiA S2 | R–M1 | 0.314 | 0.268 | 0.027 | 0.83 | 0.44 | 0.251 |
| DaLiA S2 | O1–O50 | 0.278 | 0.220 | 0.183 | 0.80 | 0.43 | 0.119 |
| DaLiA S2 | O1–M1 | 0.300 | 0.253 | 0.169 | 0.75 | 0.38 | 0.103 |
| DaLiA S2 | O50–M1 | 0.186 | 0.134 | 0.747 | 0.05 | 0.06 | 0.017 |
| DaLiA S1 | **R–O1** | 0.134 | 0.122 | 0.061 | 0.16 | 0.02 | 0.101 |
| DaLiA S1 | R–O50 | 0.316 | 0.256 | 0.023 | 0.82 | 0.53 | 0.253 |
| DaLiA S1 | R–M1 | 0.290 | 0.240 | 0.019 | 0.66 | 0.43 | 0.302 |
| DaLiA S1 | O1–O50 | 0.271 | 0.203 | 0.059 | 0.66 | 0.52 | 0.152 |
| DaLiA S1 | O1–M1 | 0.292 | 0.241 | -0.026 | 0.50 | 0.41 | 0.202 |
| DaLiA S1 | O50–M1 | 0.234 | 0.183 | 0.653 | 0.16 | 0.10 | 0.050 |
| WildPPG | **R–O1** | 0.079 | 0.065 | 0.519 | 0.07 | 0.05 | 0.059 |
| WildPPG | R–O50 | 0.260 | 0.197 | 0.180 | 0.72 | 0.34 | 0.257 |
| WildPPG | R–M1 | 0.349 | 0.297 | 0.151 | 0.79 | 0.22 | 0.215 |
| WildPPG | O1–O50 | 0.236 | 0.168 | 0.240 | 0.66 | 0.29 | 0.198 |
| WildPPG | O1–M1 | 0.332 | 0.274 | 0.142 | 0.72 | 0.17 | 0.156 |
| WildPPG | O50–M1 | 0.226 | 0.168 | 0.710 | 0.06 | 0.12 | 0.043 |
On every dataset O1 is the closest model to R on waveform RMSE/MAE, on prediction PCC and on all three statistic distances (3/3 votes);
O50 and M1 are 3–4× farther from R in waveform RMSE and are close to *each other* (PCC 0.75 on S2). On DaLiA the R–O1 PCC is low in
absolute terms (0.10 / 0.06) because both signals are near-flat, so the closeness there is a closeness of amplitude/energy statistics;
on WildPPG the R–O1 PCC is 0.52 (vs 0.18 for O50, 0.15 for M1) — genuine waveform agreement between the deterministic proxy and the
one-step OT-CFM sample.

## 6. Residual and QRS-region analysis (§9; QRS = ±100 ms around GT R-peaks, same neurokit detector)
| Dataset | Model | RMSE all | RMSE QRS | RMSE non-QRS | QRS/non-QRS | amp ratio in QRS | amp ratio non-QRS | residual std | residual LF (<5 Hz) | residual HF (>15 Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| DaLiA S2 | R | 0.293 | 0.414 | 0.231 | 1.79 | 0.15 | 0.35 | 0.237 | 0.56 | 0.19 |
| DaLiA S2 | O1 | 0.310 | 0.425 | 0.253 | 1.68 | 0.27 | 0.65 | 0.239 | 0.57 | 0.18 |
| DaLiA S2 | O50 | 0.449 | 0.534 | 0.412 | 1.30 | 0.83 | 1.93 | 0.327 | 0.64 | 0.15 |
| DaLiA S2 | M1 | 0.457 | 0.540 | 0.422 | 1.28 | 0.81 | 1.90 | 0.317 | 0.67 | 0.14 |
| DaLiA S1 | R | 0.322 | 0.460 | 0.260 | 1.77 | 0.18 | 0.40 | 0.264 | 0.62 | 0.12 |
| DaLiA S1 | O1 | 0.355 | 0.484 | 0.300 | 1.62 | 0.33 | 0.68 | 0.272 | 0.67 | 0.11 |
| DaLiA S1 | O50 | 0.460 | 0.568 | 0.418 | 1.36 | 0.72 | 1.44 | 0.350 | 0.64 | 0.12 |
| DaLiA S1 | M1 | 0.462 | 0.556 | 0.426 | 1.31 | 0.61 | 1.24 | 0.323 | 0.70 | 0.11 |
| WildPPG | R | 0.361 | 0.428 | 0.332 | 1.29 | 0.27 | 0.30 | 0.221 | 0.72 | 0.12 |
| WildPPG | O1 | 0.375 | 0.440 | 0.348 | 1.27 | 0.35 | 0.38 | 0.223 | 0.73 | 0.11 |
| WildPPG | O50 | 0.461 | 0.547 | 0.425 | 1.29 | 0.88 | 0.94 | 0.299 | 0.69 | 0.13 |
| WildPPG | M1 | 0.511 | 0.585 | 0.480 | 1.22 | 1.09 | 1.42 | 0.313 | 0.75 | 0.10 |

QRS-window share of samples: DaLiA S2 27.6 % (10748 GT R-peaks), DaLiA S1 25.1 % (10988 GT R-peaks), WildPPG 27.3 % (40523 GT R-peaks).
Reading (WildPPG is the interpretable case): R and O1 have the *lowest* error everywhere, inside and outside the QRS windows, yet their
predicted energy inside the QRS windows is only 27–35 % of the GT's (O50 88 %, M1 109 %). Their advantage in RMSE comes from *not
committing* to sharp deflections: the QRS/non-QRS error ratio is 1.29 for R vs 1.29 (O50) / 1.22 (M1) on WildPPG and 1.7–1.8 vs 1.3 on
DaLiA, i.e. the low-RMSE models still lose most of their error at the QRS, but they lose less there than a model that draws a full-height
R-wave at a slightly wrong position. Residual spectra are LF-dominated for all models (56–75 % below 5 Hz).

## 7. Pointwise-error inversion (§10)
| Dataset | RMSE ranking (best → worst) | amplitude-fidelity ranking | template-corr ranking | R RMSE rank | inversion(R) | inversion(O1) |
|---|---|---|---|---|---|---|
| DaLiA S2 | R < O1 < O50 < M1 | O50, M1, O1, R | O50, M1, O1, R | 1 | **YES** | YES |
| DaLiA S1 | R < O1 < O50 < M1 | O50, M1, O1, R | O50, M1, O1, R | 1 | **YES** | YES |
| WildPPG | R < O1 < O50 < M1 | O50, M1, O1, R | O50, M1, O1, R | 1 | **YES** | YES |
The deterministic regressor achieves the most favourable pointwise error of all four models on all three datasets while ranking last on
amplitude fidelity and template correlation — the pre-registered definition of a "deceptively favourable" pointwise error. (Note also that
on S2 a constant predictor at the GT mean would reach RMSE 0.250, below every model: pointwise error on these normalised windows is
dominated by offset/low-frequency terms.)

## 8. Quality–compute Pareto (§11; `pareto.csv`, `figures/*_pareto.png`)
| Dataset | arm | NFE | latency (b64) | HR err | morph | \|amp−1\| | RMSE | F1 | Pareto-optimal (NFE × {HR, morph, amp}) |
|---|---|---|---|---|---|---|---|---|---|
| DaLiA S2 | OT-CFM heun25 | 50 | 4171 ms | 8.08 | 0.650 | 0.05 | 0.435 | 0.140 | yes |
| DaLiA S2 | OT-CFM heun10 | 20 | 1649 ms | 8.29 | 0.646 | 0.05 | 0.433 | 0.140 | yes |
| DaLiA S2 | OT-CFM heun5 | 10 | 826 ms | 9.08 | 0.610 | 0.03 | 0.426 | 0.141 | yes |
| DaLiA S2 | OT-CFM heun2 | 4 | 328 ms | 15.76 | 0.419 | 0.20 | 0.421 | 0.129 | no ← iMeanFlow:meanflow1;iMeanFlow:meanflow2;iMeanFlow:meanflow4 |
| DaLiA S2 | OT-CFM heun1 | 2 | 162 ms | 30.55 | 0.110 | 1.32 | 0.621 | 0.100 | no ← iMeanFlow:meanflow1;iMeanFlow:meanflow2 |
| DaLiA S2 | OT-CFM euler1 | 1 | 82 ms | 41.96 | 0.217 | 0.85 | 0.304 | 0.079 | no ← iMeanFlow:meanflow1 |
| DaLiA S2 | iMeanFlow meanflow1 | 1 | 82 ms | 9.58 | 0.595 | 0.10 | 0.443 | 0.139 | yes |
| DaLiA S2 | iMeanFlow meanflow2 | 2 | 163 ms | 8.00 | 0.660 | 0.08 | 0.445 | 0.136 | yes |
| DaLiA S2 | iMeanFlow meanflow4 | 4 | 327 ms | 7.02 | 0.719 | 0.07 | 0.439 | 0.135 | yes |
| DaLiA S2 | MSE regressor regressor | 1 (1 fwd) | 80 ms | 35.71 | 0.160 | 0.94 | 0.289 | 0.012 | no ← iMeanFlow:meanflow1 |
| DaLiA S1 | OT-CFM heun25 | 50 | 4156 ms | 8.16 | 0.683 | 0.13 | 0.448 | 0.125 | yes |
| DaLiA S1 | OT-CFM heun10 | 20 | 1652 ms | 7.95 | 0.676 | 0.12 | 0.449 | 0.123 | yes |
| DaLiA S1 | OT-CFM heun5 | 10 | 826 ms | 8.37 | 0.632 | 0.07 | 0.450 | 0.121 | yes |
| DaLiA S1 | OT-CFM heun2 | 4 | 327 ms | 16.40 | 0.407 | 0.34 | 0.499 | 0.107 | no ← iMeanFlow:meanflow1;iMeanFlow:meanflow2;iMeanFlow:meanflow4 |
| DaLiA S1 | OT-CFM heun1 | 2 | 163 ms | 41.89 | 0.070 | 1.56 | 0.753 | 0.046 | no ← OT-CFM:euler1;iMeanFlow:meanflow1;iMeanFlow:meanflow2;MSE regressor:regressor |
| DaLiA S1 | OT-CFM euler1 | 1 | 82 ms | 35.23 | 0.168 | 0.79 | 0.347 | 0.069 | no ← iMeanFlow:meanflow1 |
| DaLiA S1 | iMeanFlow meanflow1 | 1 | 82 ms | 11.96 | 0.581 | 0.29 | 0.449 | 0.124 | yes |
| DaLiA S1 | iMeanFlow meanflow2 | 2 | 164 ms | 11.56 | 0.606 | 0.26 | 0.446 | 0.124 | yes |
| DaLiA S1 | iMeanFlow meanflow4 | 4 | 327 ms | 11.26 | 0.635 | 0.25 | 0.443 | 0.120 | yes |
| DaLiA S1 | MSE regressor regressor | 1 (1 fwd) | 80 ms | 32.34 | 0.148 | 0.95 | 0.318 | 0.054 | no ← iMeanFlow:meanflow1 |
| WildPPG | OT-CFM heun25 | 50 | 4159 ms | 9.43 | 0.670 | 0.02 | 0.440 | 0.440 | yes |
| WildPPG | OT-CFM heun10 | 20 | 1630 ms | 9.58 | 0.648 | 0.00 | 0.440 | 0.443 | yes |
| WildPPG | OT-CFM heun5 | 10 | 811 ms | 9.95 | 0.586 | 0.07 | 0.440 | 0.445 | yes |
| WildPPG | OT-CFM heun2 | 4 | 320 ms | 15.37 | 0.377 | 0.48 | 0.475 | 0.402 | no ← iMeanFlow:meanflow1;iMeanFlow:meanflow2;iMeanFlow:meanflow4 |
| WildPPG | OT-CFM heun1 | 2 | 160 ms | 40.61 | 0.106 | 2.09 | 0.760 | 0.073 | no ← OT-CFM:euler1;iMeanFlow:meanflow1;iMeanFlow:meanflow2;MSE regressor:regressor |
| WildPPG | OT-CFM euler1 | 1 | 81 ms | 15.59 | 0.379 | 0.68 | 0.355 | 0.481 | no ← iMeanFlow:meanflow1 |
| WildPPG | iMeanFlow meanflow1 | 1 | 81 ms | 11.85 | 0.551 | 0.04 | 0.485 | 0.385 | yes |
| WildPPG | iMeanFlow meanflow2 | 2 | 163 ms | 11.62 | 0.601 | 0.06 | 0.467 | 0.394 | yes |
| WildPPG | iMeanFlow meanflow4 | 4 | 320 ms | 12.18 | 0.637 | 0.05 | 0.461 | 0.397 | yes |
| WildPPG | MSE regressor regressor | 1 (1 fwd) | 80 ms | 19.22 | 0.331 | 0.75 | 0.343 | 0.436 | no ← OT-CFM:euler1;iMeanFlow:meanflow1 |
At one network evaluation the regressor is dominated by iMeanFlow-1 on every dataset (HR, morphology and amplitude all better at the
same cost) and — on DaLiA — also by OT-CFM 1-NFE; it never enters the Pareto front. The regressor's forward pass is not a generative NFE
(no noise, no time), so its point is drawn for cost reference only.

## 9. Qualitative examples (§14; `figures/a5{a,b,c}_examples_{quantile,fixed}.png`)
Same pre-registered windows as A2/A3/A4 (HR-error quantiles 10/50/90 % of the 50-NFE arm, and fixed positions); rows PPG / GT /
regressor / OT-50 / OT-1 / iMF-1 with R-peak markers and shaded ±100 ms QRS windows. DaLiA: the regressor is a near-flat line with small
ripples at the PPG rhythm; OT-1 is the same line plus low-level noise; OT-50 and iMF-1 draw full-height beats. WildPPG: the regressor and
OT-1 draw small (≈ 0.25 amplitude) upward deflections *at the GT R-peak positions* and nothing else; OT-50 draws full beats at the same
positions; iMF-1 draws full beats with more baseline activity and occasional spurious peaks.

## 10. Verdict (frozen rules, §12)
| Term | DaLiA S2 | DaLiA S1 | WildPPG |
|---|---|---|---|
| attenuation(R): amp(R) < amp(O50) − 0.25, morph(R) < morph(O50) − 0.10, RMSE(R) ≤ RMSE(O50) | ✓ | ✓ | ✓ |
| closest(O1→R): RMSE(R,O1) minimal and ≥ 2/3 statistic votes | ✓ (3/3) | ✓ (3/3) | ✓ (3/3) |
| WildPPG preserved(R): F1(R) ≥ 0.8·F1(O50), gain(R) ≥ 0.5·gain(O50) | — | — | ✓ (0.436 ≥ 0.352; 5.70 ≥ 3.58) |
| H1 / H2 / H3 / H4 | ✓ / ✓ / — / — | ✓ / ✓ / — / — | ✓ / — / ✓ / ✓ |
**Overall: STRONG SUPPORT.** H4 reading on WildPPG: iMF-1 morph 0.551 > 0.331 + 0.10, |amp − 1| 0.04 < 0.75, and its beat timing is less
precise than the regressor's (F1 0.385 < 0.436; RR MAE 25.7 > 16.7 ms).

## 11. Interpretation (within the allowed wording)
- OT-CFM at 1 NFE **exhibits conditional-mean-like attenuation** and **empirically approaches the behaviour of an MSE-trained
  conditional-mean proxy** on the same backbone: same low-RMSE/low-amplitude/low-morphology signature, closest waveforms, same
  QRS-energy loss, same dependence on temporal alignment. The one-step OT-CFM sample is *not* identical to the proxy — it retains a
  little more HF energy (0.07–0.15 vs 0.001–0.005) and a noise-seed-dependent residual (A4 seed std 0.035) — but it is far closer to the
  proxy than to its own 50-NFE solution or to iMF-1.
- What the conditional-mean-like solution keeps is decided by alignment: with unsynchronised wrist/chest pairs (DaLiA) the MSE-optimal
  prediction is essentially beat-free; with synchronised devices (WildPPG) it keeps the beat *timing* and the PPG dependence and only
  averages away amplitude and QRS sharpness. This is the mechanism behind the A2/A3 vs A4 difference (REPLICATED vs PARTIAL).
- Finite-interval MeanFlow escapes the attenuated regime at one evaluation (amplitude ≈ 1, template correlation 0.55–0.60) at the price of
  less precise beat placement on WildPPG (F1 0.385 vs 0.436/0.481, RR MAE 25.7 ms). We describe this as greater timing variability, not
  as evidence of "phase diversity" or of a multimodal posterior.
- Not claimed: that OT-CFM mathematically converges to the conditional mean; that MeanFlow "solves multimodality"; that PPG→ECG is
  inherently multimodal; that WildPPG proves phase diversity.

## 12. Limitations and threats
- One seed per model; one (S2, S1) or two (kjd, ssx) test participants; 3,907-window WildPPG subset; A3 shares the validation subject with A2.
- The regressor is a proxy: same backbone but 2.9 M effective parameters (adaLN weights inactive with cond = 0, no time path) and a
  learned constant token in place of the noisy state; it reached its best validation MSE in 11–34 epochs/rounds with the OT-CFM optimiser
  settings. A better-optimised MSE model could sharpen somewhat; it cannot, by construction, recover a full-amplitude single sample.
- The initial zero-state control was untrainable (Amendment 1). The fix was minimal and pre-committed, but it is a post-hoc amendment.
- Beat-level metrics on DaLiA are not interpretable; HR error on WildPPG for the regressor (19.2 bpm) is worse than for OT-1 (15.6)
  despite similar F1 — the neurokit HR estimate is fragile when 26 % of beats are missed; RR-interval MAE and F1 are the timing metrics.
- Prediction PCC on DaLiA is near zero for every pair and is reported but not interpreted.
- Pointwise metrics on per-window-normalised 8 s windows are offset-dominated (constant GT-mean predictor RMSE 0.250 on S2).

## 13. Reproducibility
`bash scripts/run_a5_pipeline.sh` (preflight → `ppg2ecg.training.train_a5` → `scripts/eval_a5.py`, a → b → c; GPU-gated) then
`PYTHONPATH=src .venv/bin/python scripts/analyze_a5.py`. Provenance per run (`provenance.json`): git `d961000`, upstream PENGUIN `6cd70cd`,
processed-file hashes, leakage checks, parameter counts, GPU (RTX 5090, torch 2.11.0+cu130). Tests: `tests/test_regressor.py` (param
identity, dead-start mechanism, gradient flow, determinism). Aborted zero-state runs: `outputs/aborted/*_zero_state_deadstart/`.
