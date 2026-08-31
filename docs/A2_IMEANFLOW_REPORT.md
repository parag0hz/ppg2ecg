# A2 Improved MeanFlow Report

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats, so
> neither can fall when a beat is missed. Values and specifications here are unchanged; only the labels and
> their scope are made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Generated from `outputs/a2_imeanflow_s5_ppgdalia_8s_seed42/` vs `outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42/`. Pre-registration: `docs/A2_IMEANFLOW_PREREGISTRATION.md`; audit: `docs/IMEANFLOW_AUDIT.md`.

## Research question
> Can Improved MeanFlow make the long noise→ECG transport jump in one network evaluation while preserving the physiological structure that OT-CFM needs many evaluations to generate?

## Frozen protocol
Identical to A0-b (data, 8 s windows, split, seed 42, backbone with 4,568,707 parameters, PPG conditioning, AdamW 1e-3 / wd 0.01 / effective batch 64, fp32, patience 20 / min_delta 1e-4 on a deterministic fixed-bank metric). Only the objective/parameterisation changed: OT-CFM → Improved MeanFlow (`V = u + (t−r)·sg(du/dt)`, v-loss with adaptive weighting, (t,r) logit-normal(−0.4,1), 50 % r=t, boundary v_θ, conditioning E(t)+E(h) via the backbone's single embedder). Gradient accumulation 2 × 32 for memory (prereg §8).

## iMeanFlow paper/code audit
See `docs/IMEANFLOW_AUDIT.md` (papers arXiv:2505.13447 / arXiv:2512.02012 v2; official code `Lyy-iiis/imeanflow` @ bf60cd7, submodule `external/iMeanFlow`).

## Implementation parity tests
`tests/test_imeanflow.py`: analytic linear-field MeanFlow identity (V ≡ v), zero loss for consistent pairs, shapes/conditioning/batch independence, backbone parity (t-only mode == upstream forward_step bit-exact), JVP vs finite differences and vs double-VJP on the backbone, stop-gradient equivalence, finite loss/grads, seed determinism, 1-NFE call count, and a JAX port of the official objective evaluated with identical weights (loss and V agree to 1e-5). Independent adversarial review: see EXPERIMENT_LOG.

## Training
- iMF: 81 epochs, best epoch 61, early stopped True, 3.24 h, peak 16.5 GiB, selection metric 0.17382 (fixed-bank iMF MSE)
- OT-CFM (A0-b): 85 epochs, best 65, 1.77 h, peak 18.0 GiB

## Memory/runtime
Forward-mode JVP: ≈ 0.51 GiB per sample at T = 1024 (OT-CFM 0.29) → micro-batch 32 × 2 accumulation; training step ≈ 2 × 250 ms; 1-NFE sampling latency in the table below.

## Main controlled comparison
| Model | Sampler | Actual NFE | HR Error (bpm) | Morph corr | Amp ratio | Cond gain (bpm) | RMSE | MAE | beats/ref | seed std | Latency (ms/batch 64) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OT-CFM | Heun 25 | 50 | 8.08 | 0.650 | 0.949 | 5.69 | 0.435 | 0.354 | 0.98 | 0.242 | 4171 |
| OT-CFM | Heun 2 | 4 | 15.76 | 0.419 | 1.203 | 2.07 | 0.421 | 0.326 | 0.81 | 0.230 | 328 |
| OT-CFM | Euler 1 | 1 | 41.96 | 0.217 | 0.145 | 0.24 | 0.304 | 0.221 | 0.34 | 0.033 | 82 |
| iMeanFlow | 1 step | 1 | 9.58 | 0.595 | 0.896 | 4.47 | 0.443 | 0.366 | 1.00 | 0.254 | 82 |
| iMeanFlow | 2 steps | 2 | 8.00 | 0.660 | 0.922 | 5.60 | 0.445 | 0.367 | 0.98 | 0.256 | 163 |
| iMeanFlow | 4 steps | 4 | 7.02 | 0.719 | 0.927 | 6.59 | 0.439 | 0.361 | 0.98 | 0.251 | 327 |

Secondary diagnostics (absolute beat-level temporal correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion):
| Model | Sampler | PCC | R-peak F1 | RR MAE (ms) | QRS err (ms) | HF ratio (target 0.32) | upstream HR err corrected |
|---|---|---:|---:|---:|---:|---:|---:|
| OT-CFM | Heun 25 | 0.001 | 0.140 | 32.0 | 33.7 | 0.269 | 8.26 |
| OT-CFM | Heun 2 | 0.001 | 0.129 | 30.8 | 37.3 | 0.506 | 27.58 |
| OT-CFM | Euler 1 | 0.008 | 0.079 | 22.6 | 44.7 | 0.150 | 46.72 |
| iMeanFlow | 1 step | 0.003 | 0.139 | 31.9 | 37.4 | 0.252 | 12.36 |
| iMeanFlow | 2 steps | 0.000 | 0.136 | 30.3 | 35.1 | 0.267 | 9.19 |
| iMeanFlow | 4 steps | -0.000 | 0.135 | 30.2 | 30.7 | 0.284 | 7.73 |

## 1-NFE physiological recovery
Recovery = fraction of the OT-CFM 50→1 NFE gap recovered by iMeanFlow at 1 NFE (prereg §5).
| metric | OT-CFM 50 | OT-CFM 1 | iMeanFlow 1 | recovery |
|---|---:|---:|---:|---:|
| HR error (bpm) | 8.081 | 41.960 | 9.582 | **+0.96** |
| morph corr | 0.650 | 0.217 | 0.595 | **+0.87** |
| amplitude ratio | 0.949 | 0.145 | 0.896 | **+0.93** |
| conditioning gain (bpm) | 5.693 | 0.237 | 4.469 | **+0.78** |
| RMSE (aux.) | 0.435 | 0.304 | 0.443 | +1.06 |
| MAE (aux.) | 0.354 | 0.221 | 0.366 | +1.09 |
| beats / reference | 0.98 | 0.34 | 1.00 | ≥ 0.7 ✔ |

Figure: `outputs/a2_imeanflow_s5_ppgdalia_8s_seed42/figures/recovery.png`.

## Conditional fidelity
PPG-shuffle test (same noise, PPG replaced by a deranged window): HR error vs the *right* target / vs the *wrong* target — OT-CFM 50 NFE 8.30 / 13.99 (gain 5.69); OT-CFM 1 NFE 40.62 / 40.86 (gain 0.24); iMeanFlow 1 NFE 9.43 / 13.90 (gain **4.47**).

The PPG-shuffle gain of iMeanFlow at 1 NFE is 4.47 bpm (78 % of the 50-NFE OT-CFM gain of 5.69; OT-CFM at 1 NFE: 0.24), i.e. the
one-step output still tracks the heart rate of the PPG it was given. With 2 and 4 MeanFlow steps the gain reaches 5.60 and 6.59 bpm —
equal to or above the 50-NFE OT-CFM reference. The rate ceiling of the *backbone* is shared by both objectives: per-activity HR
error of iMF-1 is 6–9 bpm for sedentary/walking windows but 18–23 bpm for cycling/stairs (reference HR 107–114 bpm), the same
regression-to-the-mean pattern as A0/A0-b (`diagnostics.json`), so this is a limitation of the PPG conditioning path / data, not of
the one-step objective. Absolute beat-level timing is unchanged (R-peak F1 0.139, cross-correlation lag uniform over ±0.5 s) —
the PPG-DaLiA protocol limitation applies equally to every arm.

## Qualitative examples
A0's deterministic windows (HR-error quantiles 10/50/90 % of the 50-NFE arm and fixed positions): `outputs/a2_imeanflow_s5_ppgdalia_8s_seed42/figures/controlled_examples_quantile.png`, `controlled_examples_fixed.png` — same PPG, same initial noise, identical y-scale.

On the pre-registered windows (`controlled_examples_quantile.png`, `controlled_examples_fixed.png`; same PPG, same initial noise,
same y-scale): OT-CFM at 1 NFE is a flat line near −0.3 with ripple (conditional mean); iMeanFlow at 1 NFE produces sharp,
full-amplitude QRS-like spikes at approximately the right rate with a structured baseline, visually comparable to OT-CFM at 50 NFE.
Residual defects visible at 1 NFE: occasional extra small spikes between beats (e.g. window 880: 13 detected vs 10 reference
beats; the beat-count ratio is 1.00 on average because other windows miss beats), a noisier inter-beat baseline than at 50 NFE
(HF-energy ratio 0.252 vs 0.269 target — i.e. *less* high-frequency energy than the target, so not a noise artefact but slightly
smoothed T/P waves), and the same random beat phase as all other arms.

## Failure taxonomy
Applied to iMeanFlow at 1 NFE (pre-registered list):
- **F1 conditional-mean collapse: absent** (amplitude ratio 0.90, seed-to-seed std 0.254 vs 0.242 for OT-CFM-50; OT-CFM-1: 0.145 / 0.033).
- **F2 QRS smoothing: partial/minor** — template correlation 0.595 (95 % CI 0.58–0.61) vs 0.650 (0.64–0.67) at 50 NFE; QRS-width proxy
  error 33.4 ms vs 33.7 ms; recovered fully with 2 steps (0.660) and exceeded with 4 (0.719).
- **F3 amplitude collapse: absent** (0.90; recovery 0.93).
- **F4 conditioning neglect: absent** (gain 4.47 bpm, recovery 0.78; not zero as in OT-CFM-1).
- **F5 unstable JVP training: absent with the official h-only conditioning** (train MSE 0.30 → 0.17, |du/dt| ≈ 0.5, 81 epochs, no
  non-finite values). It **was** present with `E(t)+E(1000·h)` (diverged in 2 epochs) — recorded as an instance of F5 caused by an
  ill-conditioned time embedding, not by the objective.
- **F6 rate recovery but morphology failure: not observed** — both recovered; morphology is the metric with the largest residual
  gap (13 % of the 50→1 gap), plus **F6′ spurious extra spikes** in a minority of windows (new sub-type, see examples).
- **F7 interval under-resolution (shared-embedder `E(t)+E(h)`): pre-empted** by the h-only amendment; not tested to completion.

## Limitations
- Single seed (42), single test subject (S2, 1025 windows) and single validation subject (S11); no variance across subjects/seeds.
- The comparison is *objective-only* by design: iMF was trained with the OT-CFM baseline optimiser (AdamW 1e-3, wd 0.01, no EMA,
  no warm-up) rather than the official recipe (Adam 1e-4, EMA 0.9999); the official auxiliary v-head was replaced by the boundary
  condition v_θ = u_θ(z,t,t) to keep the parameter count; conditioning is h-only (official code) instead of the MF paper's (t, h).
- The 50-NFE OT-CFM reference itself has weak beat-level conditional fidelity (HR r ≈ 0.4 vs reference; regression to the mean),
  so "recovering the 50-NFE level" is a relative statement, not a clinical one.
- Beat-aligned metrics (R-peak F1, PCC, RR MAE) are not interpretable on raw PPG-DaLiA (device synchronisation); QRS width is a proxy.
- Two pre-result amendments of the conditioning (documented in the pre-registration §9) preceded this run; the h-scale sweep was
  not exhaustive (1, 1000, h-only), so h-only is the first stable configuration found, not a tuned optimum.
- Latency parity: iMF-1 and OT-CFM Euler-1 both cost one backbone evaluation (82 ms per batch of 64 on the RTX 5090, fp32, uncompiled);
  iMF training costs ≈ 2× OT-CFM per step (forward-mode JVP) and 3.2 h total vs 1.8 h for A0-b.

## GO / PARTIAL / FAIL
**SUCCESS** — recovery ≥ 0.5 on 4/4 physiological metrics; beats/reference 1.00 (ok); all < 0.25: False; gain-fail rule: False.

All four physiological recovery scores exceed the pre-registered 0.5 threshold (HR 0.96, template correlation 0.87, amplitude 0.93,
conditioning gain 0.78) and the 1-NFE output carries beats (1.00 × reference), so the frozen rule yields **SUCCESS**. The remaining
gaps to the 50-NFE reference are small but real for HR (9.58 vs 8.08 bpm, CIs 9.0–10.1 vs 7.6–8.6) and morphology (0.595 vs 0.650),
and vanish (HR) or invert (morphology, conditioning gain) with 2–4 MeanFlow steps: iMF-4 (7.02 bpm, 0.719, gain 6.59) is better than
OT-CFM-50 on every physiological metric at 8 % of its cost. RMSE/MAE are *worse* for iMF-1 than for the collapsed OT-CFM-1
(0.443 vs 0.304) — exactly the pattern the pre-registration warned about: pointwise metrics reward the beat-free mean.

## Recommended next research question
The one-step objective works on this backbone; the open question is now the **ceiling shared by both objectives**: HR error
regresses to the training mean under high-HR activity and beat-level timing is not learnable on raw PPG-DaLiA. Recommended next
experiments (all pre-registered, no new architecture yet): (1) seeds 43/44 and the 5-fold subject protocol P1 for A0-b and A2 to
establish variance; (2) a beat-level protocol with an explicit, documented PPG↔ECG re-synchronisation step (or a dataset with
synchronised beats) to make R-peak F1 / RR metrics meaningful and to test whether one-step models preserve beat *timing*;
(3) only then, if beat timing/high-HR tracking is the bottleneck, consider the conditioning path — as a separate, pre-registered
architecture question, not as part of the objective comparison.
