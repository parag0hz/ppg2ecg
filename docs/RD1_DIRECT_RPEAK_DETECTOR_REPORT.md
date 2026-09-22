# RD1 — A direct PPG → R-peak detector matches consensus decoding on F1 and beats it on RR-MAE; generation is not necessary for event estimation

Prereg `90fbc9b`, frozen before any number. R1's Global-TCN unchanged (328,897 params), Gaussian targets σ = 20 ms at the
reference R peaks (neurokit on the target ECG — the same reference the standard pipeline uses), BCE, V1 VitalDB train
(288,400 windows), AdamW 1e-3 / wd 0.01 / batch 64 / 14,000 steps / seed 42 (**51 s** of training). Peaks extracted
with R1's frozen rule (threshold 0.35, refractory 32 samples). Scored with the repository's per-window functions
(`rpeak_prf_at`, `beat_level_metrics`) fed the detector's peak sequence; comparators are the SR1 seed-42 arrays and ED1's
consensus decoding at its frozen parameters (recomputed from the ED1 sample cache; the recomputation reproduces ED1's
numbers: iMF F1 0.7650, RR 10.77, HR 6.94). V1 test: 1,156 patients, 19,543 windows; 2,000-replicate patient-clustered
CIs. Raw: `artifacts/rd1_direct_rpeak/result.json`; per-window `outputs/rd1_detector/test_metrics.npz`.

## Results (V1 VitalDB test)
| estimator | NFE | R-peak F1 @ 50 ms | precision / recall | RR-MAE (ms) | HR error (bpm) † | params | CPU ms / window |
|---|---|---|---|---|---|---|---|
| **direct R-peak detector (RD1)** | — | **0.7725** [0.762, 0.783] | 0.803 / 0.759 | **7.67** [7.51, 7.84] | 4.43 (88.6 % of windows) | 0.33 M | **1.9** |
| iMF consensus-decoded | 16 | 0.7650 [0.756, 0.774] | | 10.77 [10.57, 10.99] | 6.94 (97.0 %) | 4.30 M | 728 |
| CD consensus-decoded | 16 | 0.7664 [0.757, 0.775] | | 12.25 [12.04, 12.49] | 7.18 (99.4 %) | 4.30 M | 711 |
| iMF single sample | 1 | 0.6741 | | 20.82 | 10.14 (100 %) | 4.30 M | 45 |
| PENGUIN 50 NFE | 50 | 0.6447 | | 23.80 | 9.69 (100 %) | 4.30 M | 2,286 |
Micro-F1 0.815, Macro-F1 0.771 for the detector. † HR error is defined only where both sides have ≥ 2 beats (the pipeline's
Ω_HR rule); the detector emits **no peak at all in 9.2 % of windows** (1,791; 991 patients), so its marginal HR is on an
easier subset — read the paired differences below, not the marginals.

## Paired differences, detector − other [95 % CI]; per-patient win rate
| vs | F1 | RR-MAE (ms) | HR (bpm, common windows) |
|---|---|---|---|
| iMF consensus-decoded | **+0.0075** [+0.0039, +0.0109] · 58 % | **−2.85** [−3.01, −2.69] · 90 % | **−0.35** [−0.47, −0.23] · 70 % |
| CD consensus-decoded | +0.0061 [+0.0030, +0.0093] · 58 % | −4.16 [−4.33, −3.98] · 95 % | −0.31 [−0.45, −0.17] · 72 % |
| iMF single sample | +0.098 [+0.095, +0.102] · 94 % | −12.60 · 99 % | −3.61 · 92 % |
| PENGUIN 50 NFE | +0.128 [+0.123, +0.133] · 94 % | −15.73 · 99 % | −3.41 · 91 % |

On the 88.4 % of windows where every arm's HR is defined: detector 4.38, iMF decoded 4.73, CD decoded 4.70, PENGUIN-50 7.81,
and the DB1 direct HR regressor 4.16 — the regressor remains the best rate estimator.

## Reading it
1. **F1: the detector matches consensus decoding** — +0.006 / +0.0075, significant but below the program's 0.02 F1
   margin (NO MEANINGFUL CHANGE by the BB1 rule; 38 % of patients better by > 0.02, 23 % worse). **RR-MAE: the detector
   is clearly better** (7.7 vs 10.8 / 12.2 ms, 90–95 % of patients). Both generative arms and the detector are far above
   any single generated sample (0.67) and PENGUIN-50 (0.64).
2. **Where the difference comes from.** The detector is silent on 9.2 % of windows (hard PPG; there every arm is near
   zero — F1 0.10–0.14 for the generative decodings, 0.13 for PENGUIN-50) and better on the remaining 91 % (0.849 vs
   0.830). Consensus decoding always returns a beat train; the detector effectively abstains. For RR intervals the
   detector's sharper timing (σ = 20 ms target) is what wins.
3. **Cost.** 0.33 M parameters (13× fewer), 51 s of training, 1.9 ms per window on 4 CPU threads — 380× faster than the
   16-sample consensus (728 ms) and 24× faster than one generated sample (45 ms).
4. **Consequence for the paper, as preregistered.** With DB1 (rate) and RD1 (events) both won by direct discriminative
   models, "generation is necessary for HR / R-peak estimation" is not a claim this program can make. The claim is
   scoped to: *when a structured conditional generative output (an ECG waveform with its beat train and sample spread) is
   required, how to allocate a fixed inference budget (mostly width, enough depth for diversity) and how to extract the
   functional (median for rate, event-level consensus for beats)*. The consensus recipe is what makes the generative
   route reach the discriminative level (F1 0.67 → 0.77) at 16 NFE; it does not surpass it.
5. Single seed (42), VitalDB only; N1 had already shown the same ordering on WildPPG for the R1 detector. No result-
   dependent choice: σ, threshold, refractory, steps and comparators are those of the preregistration.
