# A4 WildPPG Replication Report

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats, so
> neither can fall when a beat is missed. Values and specifications here are unchanged; only the labels and
> their scope are made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Generated from `outputs/a4_imeanflow_wildppg_seed42/` vs `outputs/a4_otcfm_wildppg_seed42/` — dataset WildPPG (test kjd, ssx), test subject(s) ['kjd', 'ssx'], val ['an0', 'k2s']. Pre-registration: `docs/A3_A4_REPLICATION_PREREGISTRATION.md`; audit: `docs/IMEANFLOW_AUDIT.md`.

## Research question
> Can Improved MeanFlow make the long noise→ECG transport jump in one network evaluation while preserving the physiological structure that OT-CFM needs many evaluations to generate?

## Frozen protocol
Identical to A0-b (data, 8 s windows, split, seed 42, backbone with 4,568,707 parameters, PPG conditioning, AdamW 1e-3 / wd 0.01 / effective batch 64, fp32, patience 20 / min_delta 1e-4 on a deterministic fixed-bank metric). Only the objective/parameterisation changed: OT-CFM → Improved MeanFlow (`V = u + (t−r)·sg(du/dt)`, v-loss with adaptive weighting, (t,r) logit-normal(−0.4,1), 50 % r=t, boundary v_θ, conditioning E(t)+E(h) via the backbone's single embedder). Gradient accumulation 2 × 32 for memory (prereg §8).

## iMeanFlow paper/code audit
See `docs/IMEANFLOW_AUDIT.md` (papers arXiv:2505.13447 / arXiv:2512.02012 v2; official code `Lyy-iiis/imeanflow` @ bf60cd7, submodule `external/iMeanFlow`).

## Implementation parity tests
`tests/test_imeanflow.py`: analytic linear-field MeanFlow identity (V ≡ v), zero loss for consistent pairs, shapes/conditioning/batch independence, backbone parity (t-only mode == upstream forward_step bit-exact), JVP vs finite differences and vs double-VJP on the backbone, stop-gradient equivalence, finite loss/grads, seed determinism, 1-NFE call count, and a JAX port of the official objective evaluated with identical weights (loss and V agree to 1e-5). Independent adversarial review: see EXPERIMENT_LOG.

## Training
- iMF: 66 epochs, best epoch 46, early stopped True, 3.61 h, peak 18.8 GiB, selection metric 0.11946 (fixed-bank iMF MSE)
- OT-CFM (A0-b): 210 epochs, best 190, 5.15 h, peak 20.2 GiB

## Memory/runtime
Forward-mode JVP: ≈ 0.51 GiB per sample at T = 1024 (OT-CFM 0.29) → micro-batch 32 × 2 accumulation; training step ≈ 2 × 250 ms; 1-NFE sampling latency in the table below.

## Main controlled comparison
| Model | Sampler | Actual NFE | HR Error (bpm) | Morph corr | Amp ratio | Cond gain (bpm) | RMSE | MAE | beats/ref | seed std | Latency (ms/batch 64) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OT-CFM | Heun 25 | 50 | 9.43 | 0.670 | 0.977 | 7.16 | 0.440 | 0.369 | 0.98 | 0.194 | 4159 |
| OT-CFM | Heun 2 | 4 | 15.37 | 0.377 | 1.479 | 3.39 | 0.475 | 0.379 | 0.83 | 0.249 | 320 |
| OT-CFM | Euler 1 | 1 | 15.59 | 0.379 | 0.321 | 6.64 | 0.355 | 0.301 | 0.78 | 0.035 | 81 |
| iMeanFlow | 1 step | 1 | 11.85 | 0.551 | 1.039 | 4.29 | 0.485 | 0.414 | 0.93 | 0.299 | 81 |
| iMeanFlow | 2 steps | 2 | 11.62 | 0.601 | 0.936 | 5.17 | 0.467 | 0.400 | 0.92 | 0.285 | 163 |
| iMeanFlow | 4 steps | 4 | 12.18 | 0.637 | 0.948 | 5.39 | 0.461 | 0.393 | 0.90 | 0.264 | 320 |

Secondary diagnostics (absolute beat-level temporal correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion):
| Model | Sampler | PCC | R-peak F1 | RR MAE (ms) | QRS err (ms) | HF ratio (target 0.32) | upstream HR err corrected |
|---|---|---:|---:|---:|---:|---:|---:|
| OT-CFM | Heun 25 | 0.066 | 0.440 | 21.2 | 29.9 | 0.263 | 9.92 |
| OT-CFM | Heun 2 | 0.041 | 0.402 | 20.6 | 36.3 | 0.637 | 26.81 |
| OT-CFM | Euler 1 | 0.141 | 0.481 | 15.1 | 75.1 | 0.065 | 19.10 |
| iMeanFlow | 1 step | 0.049 | 0.385 | 25.7 | 41.7 | 0.220 | 13.51 |
| iMeanFlow | 2 steps | 0.057 | 0.394 | 24.3 | 35.9 | 0.223 | 12.90 |
| iMeanFlow | 4 steps | 0.058 | 0.397 | 23.7 | 33.7 | 0.228 | 12.20 |

## 1-NFE physiological recovery
Recovery = fraction of the OT-CFM 50→1 NFE gap recovered by iMeanFlow at 1 NFE (prereg §5).
| metric | OT-CFM 50 | OT-CFM 1 | iMeanFlow 1 | recovery |
|---|---:|---:|---:|---:|
| HR error (bpm) | 9.433 | 15.588 | 11.855 | **+0.61** |
| morph corr | 0.670 | 0.379 | 0.551 | **+0.59** |
| amplitude ratio | 0.977 | 0.321 | 1.039 | **+0.90** |
| conditioning gain (bpm) | 7.161 | 6.637 | 4.288 | **-4.47** |
| RMSE (aux.) | 0.440 | 0.355 | 0.485 | +1.52 |
| MAE (aux.) | 0.369 | 0.301 | 0.414 | +1.65 |
| beats / reference | 0.98 | 0.78 | 0.93 | ≥ 0.7 ✔ |

Figure: `outputs/a4_imeanflow_wildppg_seed42/figures/recovery.png`.

## Conditional fidelity
PPG-shuffle test (same noise, PPG replaced by a deranged window): HR error vs the *right* target / vs the *wrong* target — OT-CFM 50 NFE 9.08 / 16.24 (gain 7.16); OT-CFM 1 NFE 15.68 / 22.32 (gain 6.64); iMeanFlow 1 NFE 11.93 / 16.22 (gain **4.29**).

WildPPG changes the character of the OT-CFM 1-NFE failure. The four WildPPG devices are time-synchronised, so the one-step
conditional mean E[x₁ | PPG] is a *beat-aligned* average: OT-CFM-1 keeps the PPG dependence (shuffle gain 6.64 bpm vs 7.16 at 50 NFE),
the best beat timing of all arms (R-peak F1 0.48, PCC 0.14) and a moderate HR error (15.6 bpm), while losing amplitude (0.32) and QRS
sharpness (QRS-width error 75 ms, template corr 0.38). iMeanFlow-1 restores amplitude (1.04) and sharpness (41.7 ms; corr 0.55) and
improves HR (11.85, CI 11.5–12.3 vs 15.0–16.2), but its output depends *less* on the PPG than either OT-CFM arm (gain 4.29; right-target
HR error 11.9 vs wrong-target 16.2) and its beat timing is less precise (F1 0.385, PCC 0.05). With 2–4 MeanFlow steps the gain rises to
5.2–5.4 bpm and morphology to 0.60–0.64, HR stays ≈ 11.6–12.2 — still short of the 50-NFE reference on HR and gain. Per site
(sternum/head/wrist/ankle) the pattern is uniform: OT-1 HR 16.1/13.0/14.0/19.3 → iMF-1 12.0/10.7/12.0/12.8 (OT-50 9.7/8.3/9.1/10.6).

## Qualitative examples
A0's deterministic windows (HR-error quantiles 10/50/90 % of the 50-NFE arm and fixed positions): `outputs/a4_imeanflow_wildppg_seed42/figures/controlled_examples_quantile.png`, `controlled_examples_fixed.png` — same PPG, same initial noise, identical y-scale.

Pre-registered windows (test kjd/ssx, OT-50 HR-error quantiles): on a clean window (ssx 2439) OT-CFM-1 shows small, correctly timed
QRS bumps riding on a flat baseline — the attenuated aligned mean — whereas iMeanFlow-1 produces full-amplitude spikes with more
baseline wander and a few extra spikes; on the noisy-ECG participant kjd (windows 297, 415) OT-CFM-1 is flat, OT-CFM-50 keeps sharp
beats, and iMeanFlow-1 produces plausible beats on 297 but a wandering, partly spurious trace on 415. Same PPG, same noise, same scale.

## Failure taxonomy
- F1 conditional-mean collapse (iMF-1): absent (amp 1.04, seed std 0.30).
- F2 QRS smoothing: partial (corr 0.551 vs 0.670; QRS-width error 41.7 vs 29.9 ms) — recovered to 0.637 / 33.7 ms with 4 steps.
- F3 amplitude collapse: absent (recovery 0.91) — the clearest replicated effect.
- **F4 conditioning neglect: PRESENT relative to this dataset's baseline** — iMF-1 gain 4.29 < OT-CFM-1 6.64 < OT-CFM-50 7.16
  (recovery −4.5 by the pre-registered formula, because the OT-CFM-1 reference already retains almost all of the gain).
- F5 unstable training: absent (66 rounds, best 46, smooth loss).
- **F6′/F8 beat-timing imprecision**: iMF-1 R-peak F1 0.385 vs 0.481 (OT-1) / 0.440 (OT-50); beats/reference 0.93 — the one-step
  generative sample places sharp beats less precisely than the aligned mean does (new sub-type on synchronised data).
- The OT-CFM 1-NFE failure on WildPPG is itself a different type than on DaLiA: F3+F2 (amplitude/sharpness collapse of an aligned mean)
  rather than F1 (beat-free mean).

## Limitations
- Single seed; two test participants (one flagged "noisy ECG" by the dataset authors); 4,096-window uniform test subset; validation on a
  3,785-window subset; rounds of 220 steps instead of epochs (pre-registered); 861 constant-gap windows dropped (0.22 %); four PPG sites
  pooled as in PENGUIN (per-site numbers reported). Same frozen recipe as A2/A3 (baseline optimiser, boundary v_θ, h-only conditioning).
- The conditioning-gain recovery score is ill-conditioned when OT-CFM-1 retains most of the gain (denominator 0.52 bpm); the
  pre-registered rule nevertheless flags it, and we report it as such.

## GO / PARTIAL / FAIL
**PARTIAL** (A2 recovery rule) — recovery ≥ 0.5 on 3/4 physiological metrics; beats/reference 0.93 (ok); all < 0.25: False; gain-fail rule: False.
**Replication rule (A3/A4 §4): PARTIAL** — iMF-1 better than OT-CFM-1 on 3/4 of ['hr_abs_err_bpm', 'morph_corr', 'amp_ratio', 'cond_gain_bpm']; severe negative recovery: True; collapse signature: False.
**Pointwise-error inversion** (OT-CFM-1 has the best RMSE/MAE while physiology collapses): YES (RMSE OT-50 0.440, OT-1 0.355, iMF-1 0.485).

iMeanFlow-1 improves 3 of 4 physiological metrics over OT-CFM-1 (HR, morphology, amplitude) with recovery 0.61 / 0.59 / 0.91 and beats
0.93, but conditioning gain is worse than OT-CFM-1 (severe negative recovery) → **PARTIAL** under both the replication rule and the A2 rule.
Ordering test: A (OT-1 ≪ OT-50) holds for HR (+6.2 bpm), morphology (−0.29) and amplitude (0.32) but *not* for conditioning gain or beat
timing; B (iMF-1 ≫ OT-1) holds for HR/morphology/amplitude, not for gain/timing; C (iMF-1 → OT-50) leaves +2.4 bpm HR and −0.12 corr.
Pointwise-error inversion replicates (OT-1 RMSE 0.355 is the best while its physiology is the worst).

## Recommended next research question
The effect that replicates across subject *and* dataset is the recovery of **amplitude and QRS sharpness** in one step. Whether one-step
*generation* is also needed for rhythm/conditioning depends on whether the PPG–ECG pairing is beat-synchronised: on WildPPG the one-step
conditional mean already carries rhythm and conditioning. Next: (1) quantify this explicitly — compare iMF-1 against the one-step
conditional-mean regressor as a *second baseline* on both datasets; (2) seeds/folds for variance; (3) a DaLiA beat-level protocol with
documented re-synchronisation to test whether the DaLiA result changes character once the pairing is aligned.
