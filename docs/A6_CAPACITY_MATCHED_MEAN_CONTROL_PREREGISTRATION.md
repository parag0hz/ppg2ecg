# A6 Pre-registration — Capacity-Matched Conditional-Mean Control

Written 2026-08-27 **before any A6 training**; frozen by the commit that introduces it. Tests only one objection to A5; no new method.

## 1. Objection tested
A5's regressor had 3,990,787 total / 2,907,393 effective parameters versus 4,568,707 / 4,304,513 for the generative PENGUIN model
(same exclusion of the never-called `cross_attn`/`revin`). A reviewer may argue: "the MSE regressor is smoother because it has
substantially lower effective capacity, not because MSE favours a conditional expectation." A6 tests exactly this.

## 2. Model — `S5FullBackboneRegressor` (`src/ppg2ecg/models/regressor.py`)
The **unmodified upstream PENGUIN model** (all modules: `pre_conv_target`, `timestep_embedder`, 4 Flow-SSM blocks with adaLN, final
layer, PPG stem/conditioning; same width/depth) is instantiated and run through the upstream `forward_step` with
- state input `x_const = ones_like(target)` — deterministic, sample-independent, **non-learnable**, non-zero (A5's zero input was a
  dead start; A5 Amendment 1);
- fixed auxiliary time constant `t_const = 0.5` (activates the existing time-conditioning pathway for capacity parity; no flow
  interpretation, no r);
- no noise, no sample-dependent latent, no target information in the forward pass (asserted by test); inference fully deterministic.
The output is interpreted directly as `ECG_hat`; loss = `MSE(ECG_hat, ECG)`. **Total parameters 4,568,707 = OT-CFM/iMF; effective
4,304,513** (identical exclusion definition). All adaLN weights are active (cond = E(0.5) ≠ 0).

## 2b. Choice of the state constant (decided by the hard test, before A6 training; rule fixed before its result)
The spec's example `x_const = ones` is the default. A constant is *admissible* only if the full-configuration model trained with the exact
A6a recipe (batch 64, AdamW 1e-3) shows PPG-dependent structure on the validation diagnostic within 12 epochs (beats/reference ≥ 0.3 or
amplitude ratio ≥ 0.02 at any epoch) — the behaviour the A5 state-token regressor showed at epoch 1. Candidates are screened in the
order 1.0 → 0.1 → 0.01 and the **first admissible constant is frozen**; `t_const = 0.5` is not screened. Rationale: with a large
constant the target stream is dominated by the constant stem output, the LayerNorm Jacobian shrinks and the PPG-pathway gradient
vanishes (measured 1e-10…1e-12 for `ones` after 300 steps vs 1e-2 for the OT-CFM objective on the same backbone), producing a constant
output — a second, softer form of the A5 dead start. Screening traces are archived under `outputs/gradcheck_a6_x*/` (not results) and
summarised in `artifacts/a6_capacity_control/state_constant_screening.json`.

## 2c. Amendment (before A6 training): the conditioning vector must be scaled — `cond = 0.05 · E(t_const)`
**Screening result (12 epochs of the exact A6a recipe each; `outputs/gradcheck_a6_*/`, `artifacts/a6_capacity_control/state_constant_screening.json`).**
| state / conditioning | train MSE at ep 12 | max amp | max beats | admissible |
|---|---|---|---|---|
| x = 1.0, cond = E(0.5) (spec default) | 0.1200 (flat from ep 3) | 0.00 | 0.00 | ✗ |
| x = 0.1, cond = E(0.5) | 0.1200 (identical trajectory) | 0.00 | 0.11 | ✗ |
| x = 0.1, cond = 0 (time path off) | 0.1125 | 0.11 | 0.21 | ✓ (but 819,200 adaLN weights inactive → not capacity-matched) |
| x = 0.1, cond = 0.05·E(0.5) | 0.1116 | 0.11 | 0.24 | ✓ |
| x = 1.0, cond = 0.05·E(0.5) | 0.1200 (flat) | 0.00 | 0.00 | ✗ |
**Diagnosis.** The state-constant magnitude is irrelevant (x = 1.0 and 0.1 give numerically identical trajectories; the stem output is
bias-dominated and LayerNorm-normalised). What blocks training is the *unscaled* fixed conditioning vector: every adaLN `Linear.weight`
row receives a coherent gradient ∝ SiLU(E(0.5)) over 128 inputs, so under AdamW the modulation outputs (shift/scale/gate) move ~25×
faster than their biases alone, and the deterministic MSE objective — whose only informative signal is the weak, unsynchronised PPG→ECG
relation — collapses to the constant solution. Scaling the constant vector by 0.05 keeps the time path *active* (cond ≠ 0, all
4,304,513 effective parameters receive gradient; verified by `tests/test_regressor.py`) while restoring the bias-like update rate;
the value 0.05 was chosen a priori (one value tried) and is frozen. `cond = 0` would also train but reproduces A5's inactive adaLN
weights and is therefore not used. With the scaled cond the state constant *does* matter: x = 1.0 (stem output norm 1.70 per time step) still collapses, x = 0.1 (norm
0.37, close to the bias-only 0.32) trains — both a large constant stem output and the unscaled cond independently block training.
Applying the §2b order (1.0 → 0.1 → …) with the scaled cond, the first admissible constant is 0.1. **Frozen A6 model: `x_const = 0.1`,
`t_const = 0.5`, `cond_scale = 0.05`; nothing else changed.** Both deviations from the spec's example (`ones`, unscaled `E(0.5)`) are
forced by the hard test, decided before any A6 result, and archived with their traces. The generative arms are untouched (they use the upstream
path with the unscaled cond).

## 3. Gradient-flow hard test (before training; `scripts/gradflow_a6.py`, `tests/test_regressor.py`)
Recorded at optimiser steps 0/1/5/20/50 on real windows: module-level gradient norms of the state stem, timestep embedder, PPG stem, S5
blocks (PPG and target streams), adaLN, cross-stream MLPs, final layer; final-layer input non-zero; determinism; no target in the forward
signature. Criterion: at step 0 only the final layer has gradient (upstream zero-initialises `final_layer.linear.weight` — identical in
the generative model), by step 5 the state stem / timestep embedder / adaLN / final layer and by step 20 **every** pathway including the
S5 blocks and PPG stem have non-zero gradient, and the output depends on the PPG after the recorded steps. The A5 zero-state dead start
(nothing beyond the final bias ever trains) must not reproduce. Results saved to `artifacts/a6_capacity_control/gradient_flow.json`.

## 4. Parameter-parity table (auto-generated in the report)
| Model | Total | Effective |
|---|---:|---:|
| PENGUIN OT-CFM | 4,568,707 | 4,304,513 |
| iMeanFlow (same backbone, h-only cond) | 4,568,707 | 4,304,513 |
| A5 regressor (state token) | 3,990,787 | 2,907,393 |
| A6 full-backbone MSE | 4,568,707 | 4,304,513 |

## 5. Experiments (existing manifests; no new split)
A6a `a6a_fullbackbone_mse_dalia_testS2_seed42` (test S2, val S11, `v0_8s`); A6b `a6b_fullbackbone_mse_dalia_testS1_seed42` (test S1,
val S11); A6c `a6c_fullbackbone_mse_wildppg_seed42` (test kjd, ssx; val an0, k2s; 4,096-window val/test subsets, 220-step rounds).

## 6. Training (identical to A5)
AdamW lr 1e-3, wd 0.01, batch 64, fp32, seed 42, max 300 epochs/rounds, deterministic validation-MSE selection, patience 20, min_delta
1e-4. No change after results.

## 7. Evaluation (A5 implementation unchanged: `scripts/eval_a5.py`, `scripts/analyze_a6.py`)
Compared on identical test windows: Rsmall (A5), **Rfull (A6)**, O1, O50, M1. Metrics: HR error, template correlation, amplitude
ratio, conditioning gain (same derangement), RMSE, MAE, HF ratio, beats/reference, latency; WildPPG R-peak F1 and RR MAE.
Prediction distances RMSE/MAE/PCC(Rfull, ·) for · ∈ {O1, O50, M1, Rsmall} and the amp/morph/HF statistic distances; QRS-region
(±100 ms) RMSE as in A5.

## 8. Hypotheses (frozen)
H6.1 Rfull shows the same low-RMSE / attenuated-morphology / low-HF behaviour as Rsmall. H6.2 O1 is closer to Rfull than O50 and M1
are. H6.3 On WildPPG Rfull keeps aligned beat timing (F1) while sharp morphology stays weak.

## 9. Verdict rules (frozen; A5 §12 terms applied to Rfull)
attenuation(Rfull) = amp(Rfull) < amp(O50) − 0.25 and morph(Rfull) < morph(O50) − 0.10 and RMSE(Rfull) ≤ RMSE(O50);
closest(O1→Rfull) = waveform RMSE(Rfull,O1) < min(RMSE(Rfull,O50), RMSE(Rfull,M1)) and O1 wins ≥ 2 of the 3 statistic distances;
preserved(Rfull, WildPPG) = F1(Rfull) ≥ 0.8·F1(O50) and gain(Rfull) ≥ 0.5·gain(O50).
- **CAPACITY OBJECTION RESOLVED**: attenuation(Rfull) and closest(O1→Rfull) hold on all three datasets (and preserved on WildPPG).
- **CAPACITY SENSITIVE**: attenuation(Rfull) fails on ≥ 2 datasets because morphology/amplitude recover (morph(Rfull) ≥ morph(O50) − 0.10
  or amp(Rfull) ≥ amp(O50) − 0.25) — i.e. A5's smoothing is plausibly a capacity artefact.
- **MIXED**: anything else (dataset-dependent).
Quantitative Rfull–Rsmall comparison (Δmorph, Δamp, ΔRMSE, waveform RMSE(Rfull,Rsmall)) is reported regardless of verdict.

## 10. Wording
Same allowed/forbidden wording as A5 §13. Whatever the outcome, no protocol change and no new method.
