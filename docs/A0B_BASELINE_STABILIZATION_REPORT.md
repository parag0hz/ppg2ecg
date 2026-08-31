# A0-b Baseline Stabilisation Report

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats, so
> neither can fall when a beat is missed. Values and specifications here are unchanged; only the labels and
> their scope are made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Generated from `outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42/` (and `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/`). Pre-registration: `docs/A0B_BASELINE_STABILIZATION_PREREGISTRATION.md`.

## Frozen protocol
Identical to A0 except checkpoint selection (deterministic fixed-bank validation CFM loss, 4 banks, min_delta 1e-4, patience 20). Seed 42, test S2, val S11, 8 s @ 128 Hz, upstream PENGUIN model class, OT-CFM, AdamW 1e-3 / wd 0.01 / batch 64.

## Training
- A0-b: 85 epochs, best epoch **65** (val_cfm_fixed 0.16445), early stopped: True, 1.77 h, peak 18.0 GiB
- A0 : 21 epochs, best epoch 11, checkpoint scored on the same fixed banks: 0.19045
- Figure `outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42/figures/a0_vs_a0b.png` (NFE curves + validation loss curves).

## Main comparison (test S2, paired noise seed 0)
| metric | A0 50 NFE | A0-b 50 NFE | A0 1 NFE | A0-b 1 NFE |
|---|---:|---:|---:|---:|
| HR error (bpm) | 10.986 | 8.081 | 39.216 | 41.960 |
| morph corr | 0.662 | 0.650 | 0.136 | 0.217 |
| amplitude ratio | 0.832 | 0.949 | 0.197 | 0.145 |
| conditioning gain (bpm) | 3.844 | 5.693 | 0.397 | 0.237 |
| RMSE | 0.472 | 0.435 | 0.295 | 0.304 |
| MAE | 0.400 | 0.354 | 0.208 | 0.221 |
| PCC (diag.) | 0.002 | 0.001 | 0.007 | 0.008 |
| R-peak F1 (diag.) | 0.141 | 0.140 | 0.087 | 0.079 |
| latency ms/batch64 | 4174.604 | 4171.376 | 82.166 | 82.196 |

Full A0-b NFE curve:
| Solver | Steps | Actual NFE | HR err (bpm) | morph corr | amp ratio | cond gain (bpm) | RMSE | MAE | PCC* | R-F1* | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Heun | 25 | 50 | 8.08 | 0.650 | 0.949 | 5.69 | 0.435 | 0.354 | 0.001 | 0.140 | 4171 |
| Heun | 10 | 20 | 8.29 | 0.646 | 0.952 | 5.51 | 0.433 | 0.352 | 0.001 | 0.140 | 1649 |
| Heun | 5 | 10 | 9.08 | 0.610 | 0.971 | 4.82 | 0.426 | 0.342 | 0.001 | 0.141 | 826 |
| Heun | 2 | 4 | 15.76 | 0.419 | 1.203 | 2.07 | 0.421 | 0.326 | 0.001 | 0.129 | 328 |
| Heun | 1 | 2 | 30.55 | 0.110 | 2.317 | -0.15 | 0.621 | 0.475 | 0.000 | 0.100 | 162 |
| Euler | 1 | 1 | 41.96 | 0.217 | 0.145 | 0.24 | 0.304 | 0.221 | 0.008 | 0.079 | 82 |

\* secondary diagnostics: absolute beat-level temporal correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion.

## Pre-registered questions
1. **Was A0 under-trained?** YES — A0-b best epoch 65 (A0: 11); val_cfm_fixed A0-b best 0.16445 vs A0 checkpoint 0.19045 (rule: best epoch > 21 and improvement > 1e-4).
2. **50-NFE quality change:** ΔHR -2.90 bpm (CI A0 (10.409, 11.562), A0-b (7.567, 8.551)), Δmorph -0.012 (CI A0 (0.649, 0.675), A0-b (0.635, 0.667)), Δamp +0.117, Δgain +1.85 bpm → CHANGED beyond the margins (1.0 bpm / 0.05).
3. **Does the 1-NFE collapse persist?** YES — criteria {"hr_rise": true, "morph_drop": true, "amp_collapse": true, "gain_loss": true}; 1 NFE: HR 41.96 bpm, morph 0.217, amp 0.145, gain 0.24 vs 50 NFE: HR 8.08, morph 0.650, amp 0.949, gain 5.69.
4. **Checkpoint artefact or objective/sampler limitation?** objective/sampler limitation (gap persists after stabilisation).

## iMeanFlow gate (mechanical, prereg §6)
**GO** — the 50→1 NFE structural gap persists after stabilisation.

## Baseline stabilisation conclusion
- **A0 was under-trained.** With a deterministic selection criterion the same recipe trains to epoch 65 (85 run, patience 20) and
  reaches a fixed-bank validation CFM loss of 0.1645 vs 0.1904 for the A0 checkpoint; the training loss keeps decreasing slowly to
  0.153. The stochastic per-epoch validation MAE of A0 (±0.1 swings) had stopped training at epoch 21 on noise.
- **50-NFE quality improved on rate/conditioning, not on morphology.** HR error 10.99 → **8.08 bpm** (95 % CIs 10.4–11.6 vs
  7.6–8.6, non-overlapping), conditioning gain 3.8 → **5.7 bpm**, amplitude ratio 0.83 → 0.95, RMSE 0.472 → 0.435; beat-template
  correlation unchanged (0.662 → 0.650, CIs overlap). The better-trained model is a better *rate* model; QRS-level shape is where
  the OT-CFM + S5 baseline saturates.
- **The 1-NFE collapse is unchanged** (HR 42.0 bpm, template corr 0.22, amplitude 0.15, conditioning gain 0.24, seed std 0.03):
  all four pre-registered criteria fail, so the 50 → 1 NFE gap is a property of the objective/sampler, not of the A0 checkpoint.
  Notably the stabilised model degrades *earlier* along the NFE axis (4 NFE: HR 15.8 vs 11.2 for A0; 2 NFE: amplitude 2.3 =
  noise-dominated output) — a better-fitted velocity field is more curved along the noise→ECG path, so coarse Euler/Heun
  integration hurts more. This strengthens the case that the multi-step transport itself is the bottleneck.
- **Gate: GO** for the objective-swap line (A2 = iMeanFlow on the identical backbone). Reference for A2 = this A0-b checkpoint at
  50 NFE (HR 8.08, morph 0.650, amp 0.95, gain 5.69) and its 1-NFE Euler collapse as the floor.
- Housekeeping note: A0-b's training RNG stream diverges from A0 after epoch 1 because A0's per-epoch stochastic validation
  consumed the CPU RNG; epoch-1 losses are identical (0.3393), later epochs differ in the third decimal for that reason only.
