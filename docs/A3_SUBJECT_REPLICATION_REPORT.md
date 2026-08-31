# A3 Subject Replication Report (PPG-DaLiA, test S1)

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats.
> A missed beat therefore incurs no explicit penalty in either metric — it is excluded from the denominator
> rather than scored — so neither metric is monotonic in event coverage: both may rise or fall when the
> matched set changes. Values and specifications here are unchanged; only the labels and their scope are
> made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


Generated from `outputs/a3_imeanflow_ppgdalia_testS1_seed42/` vs `outputs/a3_otcfm_ppgdalia_testS1_seed42/` — dataset PPG-DaLiA (test S1), test subject(s) ['S1'], val ['S11']. Pre-registration: `docs/A3_A4_REPLICATION_PREREGISTRATION.md`; audit: `docs/IMEANFLOW_AUDIT.md`.

## Research question
> Can Improved MeanFlow make the long noise→ECG transport jump in one network evaluation while preserving the physiological structure that OT-CFM needs many evaluations to generate?

## Frozen protocol
Identical to A0-b (data, 8 s windows, split, seed 42, backbone with 4,568,707 parameters, PPG conditioning, AdamW 1e-3 / wd 0.01 / effective batch 64, fp32, patience 20 / min_delta 1e-4 on a deterministic fixed-bank metric). Only the objective/parameterisation changed: OT-CFM → Improved MeanFlow (`V = u + (t−r)·sg(du/dt)`, v-loss with adaptive weighting, (t,r) logit-normal(−0.4,1), 50 % r=t, boundary v_θ, conditioning E(t)+E(h) via the backbone's single embedder). Gradient accumulation 2 × 32 for memory (prereg §8).

## iMeanFlow paper/code audit
See `docs/IMEANFLOW_AUDIT.md` (papers arXiv:2505.13447 / arXiv:2512.02012 v2; official code `Lyy-iiis/imeanflow` @ bf60cd7, submodule `external/iMeanFlow`).

## Implementation parity tests
`tests/test_imeanflow.py`: analytic linear-field MeanFlow identity (V ≡ v), zero loss for consistent pairs, shapes/conditioning/batch independence, backbone parity (t-only mode == upstream forward_step bit-exact), JVP vs finite differences and vs double-VJP on the backbone, stop-gradient equivalence, finite loss/grads, seed determinism, 1-NFE call count, and a JAX port of the official objective evaluated with identical weights (loss and V agree to 1e-5). Independent adversarial review: see EXPERIMENT_LOG.

## Training
- iMF: 36 epochs, best epoch 16, early stopped True, 1.43 h, peak 16.5 GiB, selection metric 0.17127 (fixed-bank iMF MSE)
- OT-CFM (A0-b): 114 epochs, best 94, 2.35 h, peak 18.0 GiB

## Memory/runtime
Forward-mode JVP: ≈ 0.51 GiB per sample at T = 1024 (OT-CFM 0.29) → micro-batch 32 × 2 accumulation; training step ≈ 2 × 250 ms; 1-NFE sampling latency in the table below.

## Main controlled comparison
| Model | Sampler | Actual NFE | HR Error (bpm) | Morph corr | Amp ratio | Cond gain (bpm) | RMSE | MAE | beats/ref | seed std | Latency (ms/batch 64) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OT-CFM | Heun 25 | 50 | 8.16 | 0.683 | 0.870 | 8.77 | 0.448 | 0.353 | 1.03 | 0.204 | 4156 |
| OT-CFM | Heun 2 | 4 | 16.40 | 0.407 | 1.336 | 3.05 | 0.499 | 0.392 | 0.78 | 0.276 | 327 |
| OT-CFM | Euler 1 | 1 | 35.23 | 0.168 | 0.207 | 0.28 | 0.347 | 0.282 | 0.38 | 0.024 | 82 |
| iMeanFlow | 1 step | 1 | 11.96 | 0.581 | 0.711 | 4.78 | 0.449 | 0.372 | 1.03 | 0.213 | 82 |
| iMeanFlow | 2 steps | 2 | 11.56 | 0.606 | 0.739 | 6.18 | 0.446 | 0.363 | 1.02 | 0.231 | 164 |
| iMeanFlow | 4 steps | 4 | 11.26 | 0.635 | 0.753 | 6.10 | 0.443 | 0.359 | 0.97 | 0.231 | 327 |

Secondary diagnostics (absolute beat-level temporal correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion):
| Model | Sampler | PCC | R-peak F1 | RR MAE (ms) | QRS err (ms) | HF ratio (target 0.32) | upstream HR err corrected |
|---|---|---:|---:|---:|---:|---:|---:|
| OT-CFM | Heun 25 | -0.002 | 0.125 | 33.2 | 29.9 | 0.254 | 8.49 |
| OT-CFM | Heun 2 | -0.001 | 0.107 | 33.7 | 31.6 | 0.607 | 36.17 |
| OT-CFM | Euler 1 | -0.001 | 0.069 | 31.4 | 46.3 | 0.102 | 46.59 |
| iMeanFlow | 1 step | 0.001 | 0.124 | 31.4 | 30.8 | 0.303 | 15.83 |
| iMeanFlow | 2 steps | 0.001 | 0.124 | 35.5 | 30.7 | 0.291 | 13.61 |
| iMeanFlow | 4 steps | 0.002 | 0.120 | 33.1 | 29.1 | 0.296 | 12.19 |

## 1-NFE physiological recovery
Recovery = fraction of the OT-CFM 50→1 NFE gap recovered by iMeanFlow at 1 NFE (prereg §5).
| metric | OT-CFM 50 | OT-CFM 1 | iMeanFlow 1 | recovery |
|---|---:|---:|---:|---:|
| HR error (bpm) | 8.157 | 35.228 | 11.960 | **+0.86** |
| morph corr | 0.683 | 0.168 | 0.581 | **+0.80** |
| amplitude ratio | 0.870 | 0.207 | 0.711 | **+0.76** |
| conditioning gain (bpm) | 8.766 | 0.277 | 4.780 | **+0.53** |
| RMSE (aux.) | 0.448 | 0.347 | 0.449 | +1.01 |
| MAE (aux.) | 0.353 | 0.282 | 0.372 | +1.28 |
| beats / reference | 1.03 | 0.38 | 1.03 | ≥ 0.7 ✔ |

Figure: `outputs/a3_imeanflow_ppgdalia_testS1_seed42/figures/recovery.png`.

## Conditional fidelity
PPG-shuffle test (same noise, PPG replaced by a deranged window): HR error vs the *right* target / vs the *wrong* target — OT-CFM 50 NFE 7.74 / 16.51 (gain 8.77); OT-CFM 1 NFE 33.50 / 33.77 (gain 0.28); iMeanFlow 1 NFE 12.45 / 17.23 (gain **4.78**).

On S1 the OT-CFM 50-NFE reference has a larger PPG-shuffle gain than on S2 (8.77 vs 5.69 bpm): S1's recording has more HR
variation for the conditioning to track. iMeanFlow at 1 NFE keeps 4.78 bpm of it (53 % of the gap; OT-CFM-1: 0.28), and 6.1–6.2 bpm
with 2–4 steps. So the one-step output still depends on the given PPG, but less tightly than on S2 (78 %); conditioning is the
metric with the smallest recovery on both subjects.

## Qualitative examples
A0's deterministic windows (HR-error quantiles 10/50/90 % of the 50-NFE arm and fixed positions): `outputs/a3_imeanflow_ppgdalia_testS1_seed42/figures/controlled_examples_quantile.png`, `controlled_examples_fixed.png` — same PPG, same initial noise, identical y-scale.

Pre-registered windows (S1, 50-NFE HR-error quantiles): OT-CFM-1 is again a flat line; iMeanFlow-1 produces beat-bearing traces at
roughly the right rate (beats/reference 1.03) but with visibly smaller and less uniform R-peak amplitudes than on S2 (amplitude
ratio 0.71 vs 0.90) and occasional extra small spikes; the inter-beat baseline is close to the target's HF content (HF ratio 0.30 vs
target ≈ 0.29). OT-CFM-4 is the noise-dominated regime seen on S2.

## Failure taxonomy
- F1 conditional-mean collapse: absent (amp 0.71, seed std 0.213 ≈ OT-CFM-50's 0.204).
- F2 QRS smoothing: minor (template corr 0.581, CI 0.567–0.595 vs 0.683 at 50 NFE; recovered to 0.635 with 4 steps).
- F3 amplitude collapse: absent, but amplitude is the weakest morphology term on S1 (0.71; 2–4 steps 0.74–0.75).
- F4 conditioning neglect: absent (gain 4.78; recovery 0.53 — lowest of the four).
- F5 unstable training: absent (36 epochs, best 16 — the deterministic criterion plateaued earlier than on S2's split; train MSE 0.18).
- F6′ spurious spikes: present in a minority of windows, as on S2.

## Limitations
- Same as A2 (single seed, single test subject per split, baseline optimiser, boundary v_θ, beat-level metrics not interpretable on
  raw PPG-DaLiA); S11 shared as validation subject with A2 (subject-robustness, not a fully independent confirmation).
- The iMF run stopped after 36 epochs (best 16) versus 81 (best 61) on the S2 split — the fixed-bank criterion is identical, so this
  reflects the different training set (S2 in, S1 out), not a protocol change; the 50-NFE OT-CFM reference also trained longer here (114 epochs).

## GO / PARTIAL / FAIL
**SUCCESS** (A2 recovery rule) — recovery ≥ 0.5 on 4/4 physiological metrics; beats/reference 1.03 (ok); all < 0.25: False; gain-fail rule: False.
**Replication rule (A3/A4 §4): REPLICATED** — iMF-1 better than OT-CFM-1 on 4/4 of ['hr_abs_err_bpm', 'morph_corr', 'amp_ratio', 'cond_gain_bpm']; severe negative recovery: False; collapse signature: False.
**Pointwise-error inversion** (OT-CFM-1 has the best RMSE/MAE while physiology collapses): YES (RMSE OT-50 0.448, OT-1 0.347, iMF-1 0.449).

iMeanFlow-1 beats OT-CFM-1 on all four physiological metrics with no negative recovery (HR 0.86, morphology 0.80, amplitude 0.76,
conditioning 0.53; beats 1.03) → **REPLICATED** under the frozen rule (and SUCCESS under the A2 rule). The ordering A (OT-1 ≪ OT-50),
B (iMF-1 ≫ OT-1) and C (iMF-1 approaches OT-50: residual +3.8 bpm HR, −0.10 template corr) holds on the new subject. The pointwise-error
inversion replicates: OT-CFM-1 has the best RMSE (0.347) while its physiology is destroyed.

## Recommended next research question
Unchanged from A2: variance (seeds, 5-fold) and a re-synchronised beat-level protocol; the smaller amplitude/conditioning recovery on S1
suggests adding the amplitude ratio and the shuffle gain as *primary* claims in any follow-up rather than HR alone.
