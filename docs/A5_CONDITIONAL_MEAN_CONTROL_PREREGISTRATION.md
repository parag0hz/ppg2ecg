# A5 Pre-registration — Conditional-Mean Control

Written 2026-08-26 **before any A5 training**; frozen by the commit that introduces it. No new architecture, loss, split,
preprocessing, metric, or hyper-parameter is introduced; nothing existing (PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`, A2/A3/A4
checkpoints, splits, processed data, metrics) is changed or retrained. Seed 42 only.

## 1. Research questions
- **RQ1** Does a direct MSE PPG→ECG regressor exhibit the same attenuated, low-RMSE waveform behaviour as one-step OT-CFM?
- **RQ2** Is OT-CFM 1-NFE quantitatively closer to the deterministic regression solution than iMeanFlow 1-NFE or OT-CFM 50-NFE?
- **RQ3** Does temporal alignment determine what survives in the conditional-mean-like solution (WildPPG: timing/rhythm preserved,
  sharp morphology attenuated)?

## 2. Frozen hypotheses
- **H1** OT-CFM-1 and the MSE regressor resemble each other in amplitude attenuation / low-RMSE behaviour more than either resembles
  OT-CFM-50 or iMF-1.
- **H2** On PPG-DaLiA both the MSE regressor and OT-CFM-1 average away beat structure severely.
- **H3** On WildPPG the MSE regressor attenuates sharp morphology while relatively preserving PPG-aligned rhythm/timing and conditioning.
- **H4** iMeanFlow restores morphology/amplitude relative to the regressor but is more variable in exact beat timing on WildPPG.
These are hypotheses, not results.

## 3. Model — `S5ConditionalMeanRegressor` (`src/ppg2ecg/models/regressor.py`)
The unmodified upstream PENGUIN Flow-SSM/S5 model is instantiated and its two generative-input modules are **deleted**:
`pre_conv_target` (stem of the noisy state x_t) and `timestep_embedder` (time conditioning). The remaining modules run unchanged
with x_t-embedding = 0 and conditioning vector = 0 (adaLN driven by its biases). PPG → `pre_conv_ppg` → 4 Flow-SSM blocks (PPG
stream + zero-input target stream) → summed block outputs → `final_layer` → ECG. **Parameters: 3,990,659** = 4,568,707 − 578,048
(528,640 target stem + 49,408 time embedder); 264,192 of these are upstream's never-called `cross_attn` weights (also present in the
generative models). No module is added or widened. Unit tests: `tests/test_regressor.py` (module deletion, parameter identity with the
backbone minus removed modules, determinism, conditioning sensitivity, finite gradients).

## 4. Loss and wording
`L = E‖ECG − f(PPG)‖²` (plain MSE on the [−1, 1]-normalised 8 s windows). The squared-error-optimal deterministic predictor is the
conditional expectation E[ECG | PPG]; the trained network is therefore reported as an **MSE conditional-mean proxy** — never as the
exact conditional mean. No L1 or other losses.

## 5. Training protocol (as A0-b/A2 wherever applicable)
AdamW lr 1e-3, weight-decay 0.01, effective batch 64 (no accumulation needed), fp32, seed 42, max 300 epochs (A5a/b) or 300 rounds of
220 steps (A5c, the A4 rule), patience 20, min_delta 1e-4. Checkpoint selection: **deterministic validation MSE** — full validation set
for A5a/b (S11, 1,131 windows); for A5c the same ≤ 4,096-window uniform validation subset as A4 (`--val-subsample 4096`), recorded in
provenance. Diagnostic per epoch/round: HR error, template correlation, amplitude ratio, beats/reference on the first 128 validation
windows (never used for selection). No change of LR/patience/loss after results.

## 6. Experiments (existing manifests reused verbatim)
| Exp | Split | Processed data | Paired generative predictions |
|---|---|---|---|
| A5a `a5a_mse_regressor_dalia_testS2_seed42` | `split_p0_holdout_seed42.json` (test S2, val S11) | `data/processed/v0_8s` | A0-b (OT-CFM), A2 (iMF) |
| A5b `a5b_mse_regressor_dalia_testS1_seed42` | `split_a3_testS1_valS11.json` (test S1, val S11) | `data/processed/v0_8s` | A3 OT-CFM, A3 iMF |
| A5c `a5c_mse_regressor_wildppg_seed42` | `split_a4_wildppg_seed42.json` (test kjd, ssx; val an0, k2s) | `data/processed/wildppg_8s`, 4,096-window test subset (same stride) | A4 OT-CFM, A4 iMF |
Inputs/targets are bit-identical to those used by the generative arms; no extra filtering, smoothing or alignment.

## 7. Evaluation (existing code, `scripts/eval_a5.py`)
HR error, template correlation, amplitude ratio, conditioning gain (same PPG-shuffle derangement seed 1; no noise variables),
beats/reference, RMSE, MAE, HF-energy ratio, latency (one forward evaluation, batch 64), parameter count; WildPPG additionally
R-peak precision/recall/F1 and RR-interval MAE as primary timing metrics; on DaLiA the same are secondary/uninterpretable. Upstream HR
error (corrected) also reported.

## 8. Prediction-to-prediction similarity (new analysis, `scripts/analyze_a5.py`)
On identical test windows: R = regressor, O1 = OT-CFM Euler-1, O50 = OT-CFM Heun-25, M1 = iMF-1 (saved paired predictions of the frozen
checkpoints; regenerated only if missing). (A) waveform RMSE and MAE between R and each of O1/O50/M1; (B) prediction PCC (interpreted
with caution on DaLiA); (C) |amp(R) − amp(·)|; (D) |morph(R) − morph(·)|; (E) |HF(R) − HF(·)|. "Closest" = smallest distance.

## 9. Residual and QRS-region analysis (frozen definitions)
Residual r = prediction − GT per window: std; energy fraction below 5 Hz (LF) and above 15 Hz (HF) of the residual spectrum;
**QRS region = ±100 ms around every GT R-peak** detected by the same neurokit pipeline used everywhere (`ppg2ecg.evaluation.rpeaks.detect_rpeaks`);
QRS-region RMSE vs non-QRS RMSE per model. Primary on WildPPG (aligned), secondary on DaLiA.

## 10. Pointwise-error inversion test
Per dataset: rank {regressor, OT-1, OT-50, iMF-1} by GT-relative RMSE and by the physiological metrics; answer YES/NO to "does the
deterministic regressor achieve a deceptively favourable pointwise error while losing sharp waveform structure?" — YES if its RMSE is
the best or second best of the four while its amplitude ratio and template correlation are both below OT-50's by > 0.1 / > 0.05.

## 11. Quality–compute Pareto (no new training)
x = network evaluations (OT-CFM 50/20/10/4/2/1, iMF 1/2/4, regressor 1 forward evaluation) and latency; y = HR error, template
correlation, amplitude fidelity (|amp − 1|). Pareto-dominance recorded numerically.

## 12. Verdict rules (frozen)
Per dataset, define: **attenuation(R)** = amp(R) < amp(O50) − 0.25 and morph(R) < morph(O50) − 0.10 and RMSE(R) ≤ RMSE(O50);
**closest(O1→R)** = RMSE(R,O1) < min(RMSE(R,O50), RMSE(R,M1)) and O1 is the closest to R in ≥ 2 of the 3 statistic distances (C, D, E);
**WildPPG timing/conditioning preserved(R)** = F1(R) ≥ 0.8·F1(O50) and gain(R) ≥ 0.5·gain(O50).
- **STRONG SUPPORT**: attenuation(R) and closest(O1→R) hold on all three datasets, and the WildPPG preservation condition holds.
- **PARTIAL SUPPORT**: attenuation(R) and closest(O1→R) hold on ≥ 2 datasets, or all hold except the WildPPG preservation condition.
- **NOT SUPPORTED**: closest(O1→R) fails on ≥ 2 datasets, or attenuation(R) fails on ≥ 2 datasets.
Whatever the outcome, no protocol changes and no new method.

## 13. Allowed wording
"OT-CFM 1-NFE exhibits conditional-mean-like attenuation"; "consistent with a deterministic conditional-mean proxy"; if STRONG SUPPORT:
"OT-CFM 1-NFE empirically approaches the behaviour of an MSE-trained conditional-mean proxy". Not allowed: claims that OT-CFM
mathematically converges to the conditional mean, that MeanFlow solves multimodality, that PPG→ECG is inherently multimodal, or that
WildPPG proves phase diversity.

## 14. Qualitative figures
Same pre-registered windows as A2 (A0 quantiles/fixed on S2), A3 (S1) and A4 (WildPPG test subset); rows PPG / GT / regressor /
OT-50 / OT-1 / iMF-1; R-peak markers on all rows; same y-scale.

## 15. GPU scheduling note
A5 trainings run sequentially (a → b → c) and only start when the GPU has ≥ 22 GiB free (an unrelated vLLM process occupied ~30 GiB at
the time of writing); waiting does not alter the protocol.
