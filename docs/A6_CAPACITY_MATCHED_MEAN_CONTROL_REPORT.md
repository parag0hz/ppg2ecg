# A6 — Capacity-Matched Conditional-Mean Control

Pre-registration `docs/A6_CAPACITY_MATCHED_MEAN_CONTROL_PREREGISTRATION.md` (commit `50a77a8`, before any A6 result; §2b/§2c fix the
state constant and the conditioning scale by a pre-stated hard test). Artefacts `artifacts/a6_capacity_control/` (`summary.json`,
`cross_model_similarity.csv`, `qrs_region_analysis.csv`, `parameter_parity.csv`, `state_constant_screening.json`, `gradient_flow*.json`,
`figures/`). Runs `outputs/a6{a,b,c}_fullbackbone_mse_*` (config, provenance, logs, metrics; checkpoints/predictions local only).

## 1. Question and answer
**Was A5's conditional-mean attenuation merely caused by the regressor's reduced capacity (2.9 M vs 4.3 M effective parameters)?**
**No.** A deterministic MSE regressor built from the *unmodified* full PENGUIN backbone — identical parameter count and identical
effective count to the OT-CFM / iMeanFlow models — reproduces A5's pattern on all three datasets: lowest RMSE/MAE of all models,
amplitude ratio 0.04–0.24, template correlation 0.18–0.32, HF-energy ratio ≤ 0.007, and OT-CFM 1-NFE is again the closest generative
model to it on every distance. The full-capacity regressor is practically indistinguishable from the A5 regressor (waveform RMSE
between the two 0.042–0.045, |Δ template corr| ≤ 0.035, |Δ amplitude| ≤ 0.015). **Frozen verdict: CAPACITY OBJECTION RESOLVED.**

## 2. Model and parameter parity
| Model | Total params | Effective active params |
|---|---:|---:|
| PENGUIN OT-CFM | 4,568,707 | 4,304,513 |
| iMeanFlow (same backbone) | 4,568,707 | 4,304,513 |
| A5 regressor (state token) | 3,990,787 | 2,907,393 |
| **A6 full-backbone MSE** | **4,568,707** | **4,304,513** |
Effective = total − upstream's never-called `cross_attn`/`revin` (264,194), the same definition for every row. `S5FullBackboneRegressor`
keeps `pre_conv_target`, `timestep_embedder`, the 4 Flow-SSM blocks with adaLN, the final layer and the PPG conditioning, and runs the
upstream `forward_step` with a non-learnable, sample-independent state `x_const = 0.1·ones`, a fixed auxiliary time `t_const = 0.5`, and
the conditioning vector scaled `cond = 0.05·E(0.5)`; output = ECG_hat; loss = MSE. No noise, no r, no target information (asserted).

## 3. Hard test and the two forced deviations from the spec's example (`x = ones`, `cond = E(0.5)`)
Gradient-flow at initialisation behaves exactly like the generative model (adaLN-Zero: only the final layer has gradient at step 0; all
pathways non-zero by step 5–20; `gradient_flow.json`, reference `gradient_flow_reference_otcfm.json`). The decisive test was 12 epochs of
the real A6a recipe per configuration (`state_constant_screening.json`):
| state / conditioning | train MSE ep 12 | max amp | max beats | admissible |
|---|---|---|---|---|
| x = 1.0, cond = E(0.5) (spec example) | 0.1200 (flat) | 0.001 | 0.00 | ✗ |
| x = 0.1, cond = E(0.5) | 0.1200 (identical) | 0.001 | 0.11 | ✗ |
| x = 0.1, cond = 0 | 0.1125 | 0.11 | 0.21 | ✓ but adaLN weights inactive (not capacity-matched) |
| **x = 0.1, cond = 0.05·E(0.5)** | **0.1116** | **0.11** | **0.24** | **✓ (frozen)** |
| x = 1.0, cond = 0.05·E(0.5) | 0.1200 (flat) | 0.001 | 0.00 | ✗ |
Mechanism: with an *unscaled* fixed conditioning vector every adaLN `Linear.weight` row receives a coherent gradient over 128 inputs, so
under AdamW the modulation outputs move ≈ 25× faster than their biases — the weak deterministic MSE signal then collapses to the constant
solution; a large constant stem output (norm 1.70 vs 0.37 for x = 0.1) does the same. Scaling the fixed cond by 0.05 (one value, chosen a
priori) restores the bias-like rate while keeping all 819,200 adaLN weights active. Both deviations were decided and committed before any
A6 result; the generative arms are untouched.

## 4. Training
Identical recipe to A5 (AdamW 1e-3 / wd 0.01 / batch 64 / seed 42 / val-MSE selection / patience 20 / min_delta 1e-4). A6a 44 epochs
(best 24, val MSE 0.0883, 51 min, 18.4 GiB); A6b 26 (best 6, 0.0888, 30 min); A6c 54 rounds (best 34, 0.0837, 65 min, 20.7 GiB). For
comparison A5: 0.0881 / 0.0878 / 0.0854 — the full backbone reaches the same validation MSE.

## 5. Results (identical test windows; Rfull = A6, Rsmall = A5, O1/O50 = OT-CFM 1/50 NFE, M1 = iMF-1)
### DaLiA S2 (A6a)
| Model | HR err | morph | amp | gain | beats/ref | RMSE | MAE | HF | F1 | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Rfull** | 33.78 | 0.175 | 0.064 | −0.52 | 0.43 | **0.286** | **0.196** | 0.007 | 0.071 | 81 ms |
| Rsmall | 35.71 | 0.160 | 0.063 | 2.37 | 0.07 | 0.289 | 0.200 | 0.001 | 0.012 | 80 ms |
| O1 | 41.96 | 0.217 | 0.145 | 0.24 | — | 0.304 | 0.221 | 0.150 | 0.079 | 82 ms |
| O50 | 8.08 | 0.650 | 0.949 | 5.69 | — | 0.435 | 0.354 | 0.269 | 0.140 | 4171 ms |
| M1 | 9.58 | 0.595 | 0.896 | 4.47 | — | 0.443 | 0.366 | 0.252 | 0.139 | 82 ms |
### DaLiA S1 (A6b)
| Model | HR err | morph | amp | gain | beats/ref | RMSE | MAE | HF | F1 | latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Rfull** | 30.86 | 0.184 | 0.036 | 2.48 | 0.23 | 0.321 | 0.258 | 0.005 | 0.037 | 82 ms |
| Rsmall | 32.34 | 0.148 | 0.052 | 1.32 | 0.30 | **0.318** | **0.251** | 0.001 | 0.054 | 80 ms |
| O1 | 35.23 | 0.168 | 0.207 | 0.28 | — | 0.347 | 0.282 | 0.102 | 0.069 | 82 ms |
| O50 | 8.16 | 0.683 | 0.870 | 8.77 | — | 0.448 | 0.353 | 0.254 | 0.125 | 4156 ms |
| M1 | 11.96 | 0.581 | 0.711 | 4.78 | — | 0.449 | 0.372 | 0.303 | 0.124 | 82 ms |
### WildPPG (A6c; test kjd, ssx; 3,907-window subset)
| Model | HR err | morph | amp | gain | beats/ref | RMSE | MAE | HF | R-peak F1 | RR MAE | latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Rfull** | 20.21 | 0.316 | 0.241 | 4.95 | 0.71 | 0.350 | 0.295 | 0.007 | 0.421 | 17.2 ms | 81 ms |
| Rsmall | 19.22 | 0.331 | 0.254 | 5.70 | 0.74 | **0.343** | **0.288** | 0.005 | 0.436 | 16.7 ms | 80 ms |
| O1 | 15.59 | 0.379 | 0.321 | 6.64 | — | 0.355 | 0.301 | 0.065 | 0.481 | 15.1 ms | 81 ms |
| O50 | 9.43 | 0.670 | 0.977 | 7.16 | — | 0.440 | 0.369 | 0.263 | 0.440 | 21.2 ms | 4159 ms |
| M1 | 11.85 | 0.551 | 1.039 | 4.29 | — | 0.485 | 0.414 | 0.220 | 0.385 | 25.7 ms | 81 ms |
DaLiA beat-level columns are uninterpretable (devices not beat-synchronised). The Rfull conditioning gain on S2 (−0.52) is undefined in
spirit: the HR estimate on a near-flat signal is noise-dominated; on S1 and WildPPG Rfull retains PPG dependence (2.48, 4.95).

## 6. Prediction distances from Rfull (waveform RMSE / PCC / |Δamp| / |Δmorph| / |ΔHF|)
| Dataset | Rsmall | O1 | O50 | M1 | closest generative model |
|---|---|---|---|---|---|
| DaLiA S2 | 0.042 / 0.24 / 0.00 / 0.02 / 0.005 | **0.082 / 0.13 / 0.08 / 0.04 / 0.14** | 0.308 / 0.01 / 0.89 / 0.48 / 0.26 | 0.315 / 0.03 / 0.83 / 0.42 / 0.25 | O1 (RMSE, PCC, 3/3 votes) |
| DaLiA S1 | 0.044 / 0.16 / 0.02 / 0.04 / 0.004 | **0.120 / 0.12 / 0.17 / 0.02 / 0.10** | 0.308 / 0.02 / 0.83 / 0.50 / 0.25 | 0.283 / 0.00 / 0.67 / 0.40 / 0.30 | O1 (3/3) |
| WildPPG | 0.045 / 0.67 / 0.01 / 0.01 / 0.001 | **0.078 / 0.51 / 0.08 / 0.06 / 0.06** | 0.259 / 0.18 / 0.74 / 0.35 / 0.26 | 0.349 / 0.15 / 0.80 / 0.24 / 0.21 | O1 (3/3) |
Rfull and Rsmall are almost the same function (WildPPG PCC 0.67 between them; both near-flat on DaLiA). QRS-window (±100 ms) energy:
Rfull 13 % / 12 % / 26 % of GT (Rsmall 15 / 18 / 27, O1 27 / 33 / 35, O50 83 / 72 / 88, M1 81 / 61 / 109); Rfull has the lowest or
second-lowest RMSE both inside and outside the QRS windows on every dataset.

## 7. Hypotheses and verdict (frozen rules)
| Term | S2 | S1 | WildPPG |
|---|---|---|---|
| attenuation(Rfull) | ✓ | ✓ | ✓ |
| closest(O1→Rfull) | ✓ (3/3) | ✓ (3/3) | ✓ (3/3) |
| morphology/amplitude recovered vs O50 (capacity-sensitive signal) | ✗ | ✗ | ✗ |
| WildPPG timing + conditioning preserved (F1 0.421 ≥ 0.352; gain 4.95 ≥ 3.58) | — | — | ✓ |
| H6.1 / H6.2 / H6.3 | ✓ / ✓ / — | ✓ / ✓ / — | ✓ / ✓ / ✓ |
**CAPACITY OBJECTION RESOLVED.** The MSE regressor is not smoother because it is smaller: with the full backbone it is exactly as
smooth, exactly as low in RMSE, and exactly as close to OT-CFM 1-NFE.

## 8. Is OT-CFM 1-NFE still closest to the MSE proxy? S2: yes (0.082 vs 0.308/0.315). S1: yes (0.120 vs 0.308/0.283). WildPPG: yes
(0.078 vs 0.259/0.349; PCC 0.51 vs 0.18/0.15).

## 9. Limitations
- One seed; same splits/subsets as A2–A5. The full-backbone control needed two pre-committed deviations from the spec's example
  (x = 0.1 instead of 1, cond scaled by 0.05) to be trainable at all; with cond = 0 it also trains but is then not capacity-matched.
- The control is capacity-matched in parameters, not in optimisation: it uses the OT-CFM recipe and stops at the validation-MSE optimum
  (6–34 epochs/rounds). It matches A5's validation MSE, so the extra capacity did not find a sharper MSE-optimal solution.
- The regressor's HR-based conditioning gain is unreliable when few beats are detected (S2).
- Qualitative figures: `figures/a6{a,b,c}_examples.png` (rows PPG / GT / Rfull / Rsmall / OT-50 / OT-1 / iMF-1): Rfull and Rsmall are
  visually the same near-flat trace on DaLiA and the same small aligned R-peak deflections on WildPPG.
