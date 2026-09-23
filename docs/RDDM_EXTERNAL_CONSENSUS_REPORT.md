# RDDM-EXT — External consensus validation on the released RDDM checkpoint: **PARTIAL**

Prereg `d294729` (`docs/RDDM_EXTERNAL_CONSENSUS_PREREGISTRATION.md`), frozen and pushed before any number. Released RDDM
checkpoint (sha256 recorded in R0), official loader / sampler / data pipeline, **T = 10 fixed (20 NFE per sample)**,
K ∈ {1, 2, 3, 4, 8, 16} nested draws 0 … 15. **Training changed: NO. Sampler changed: NO.** Adding samples here *increases*
the budget (20 K NFE per window); **this is not an external test of fixed-budget width versus depth.**
Commands: `scripts/rddm_build_vitaldb.py`, `scripts/rddm_external_consensus.py gen | analyze`, `scripts/rddm_ext_figures.py`.
Raw: `outputs/rddm_ext_raw/` (per-sample HR, F1, RR, waveforms; gitignored). Results: `artifacts/rddm_ext/{result.json,
k_curve.csv, posthoc.json, rddm_ext.png}`. Patient-clustered 95 % CIs (2,000 replicates).

## 1. VitalDB external performance (primary; RDDM never trained on VitalDB — zero-shot transfer)
19,543 V1 test windows / 1,156 patients, rebuilt from raw with RDDM's preprocessing (rebuilt ECG vs V1 target: median
correlation 0.979, so the windows align). T\* = the project's V1 reference HR (the truth used by every VitalDB experiment).

| quantity | value |
|---|---|
| single-sample HR error (K = 1) | **8.753 [8.367, 9.137] bpm**; HR defined for 100 % of samples |
| PPG peak counting, same windows, same T\* | **7.413 [6.978, 7.854]** — RDDM K = 1 is **worse by +1.23 [+0.98, +1.48]** |
| K = 16 median consensus | 8.198 [7.835, 8.580] — still worse than PPG peaks |
| per-sample R-peak F1 @ 50 ms vs RDDM's own target | **0.083** (RDDM's own corpora: 0.61–0.85) |
| per-sample RR-MAE / waveform MAE / RMSE | 25.5 ms / 0.149 / 0.239 |
| *post-hoc:* lag of generated vs target ECG (cross-correlation, ±500 ms, draw 0) | **median +109 ms, IQR [+47, +180]**; only 7.3 % of windows within ±50 ms |

RDDM transfers the **rate** roughly but not the **beat timing**: its VitalDB beats are systematically about 0.1 s late
(post-hoc; a sensor-site / pulse-transit-time difference from its training corpora is a plausible cause, not verified).
Absolute zero-shot performance is below a model-free PPG baseline — **the preregistered instability flag is raised**.
None of these numbers is comparable with the PENGUIN / iMF tables (different preprocessing, reference waveform, domain).

## 2. K curve (VitalDB, median pooling, nested draws)
| K | total NFE / window | HR error C_K | mean individual error I_K | consensus gain G_K | C_K − C_1 (common windows) | patients better than K = 1 |
|---|---|---|---|---|---|---|
| 1 | 20 | 8.753 [8.367, 9.137] | 8.753 | 0 | — | — |
| 2 | 40 | 8.508 [8.136, 8.888] | 8.754 | 0.246 [0.230, 0.263] | −0.245 [−0.296, −0.197] | 62 % |
| 3 | 60 | 8.409 [8.031, 8.796] | 8.751 | 0.341 [0.306, 0.374] | −0.343 [−0.413, −0.277] | 64 % |
| 4 | 80 | 8.311 [7.937, 8.693] | 8.734 | 0.423 [0.395, 0.449] | −0.442 [−0.508, −0.377] | 68 % |
| 8 | 160 | 8.237 [7.866, 8.622] | 8.737 | 0.500 [0.469, 0.530] | −0.516 [−0.587, −0.444] | 70 % |
| 16 | 320 | **8.198 [7.835, 8.580]** | 8.737 | **0.539 [0.507, 0.571]** | **−0.554 [−0.629, −0.479]** | 70 % |

Coverage 100 % at every K (all contrasts on 100 % common windows). **A-C1 holds** (CI of C_16 − C_1 below 0, G_16 lower
bound > 0). **A-C2 holds** (K = 3, 4, 8 all significant). Robustness: against RDDM's own target HR, C_16 − C_1 =
−0.567 [−0.640, −0.495]. RDDM's own HR extractor (Hamilton, secondary): 9.10 → 8.32 bpm at K = 16.
**Magnitude:** the gain is **6.2 % of the single-sample error**. Our models on the same windows gained 34 % (iMF,
S = 1 at K = 16, EXP-B) — RDDM benefits from consensus, but much less.

## 3. Median vs mean
| K | median − mean (partition estimate), VitalDB |
|---|---|
| 2 | 0 by construction (checked: exactly 0) |
| 3 | **+0.018 [−0.001, +0.036]** |
| 8 | −0.009 [−0.031, +0.013] |
| 16 | median 8.198 vs mean 8.201 |

**The robust-median advantage does not reproduce on RDDM:** median and mean are indistinguishable at every K
(**A-C3(ii) fails**). On our models the median beat the mean by 0.43–1.13 bpm from K = 3 (B3-BOOT §2.3).

## 4. K = 2 / K = 3 behaviour
- Median ≡ mean at K = 2 (A-C3(i) holds, trivially).
- The mean operator's doubling gains are non-increasing from K = 1 (0.248 → 0.149 → 0.088 → 0.051; **A-C3(iii) holds**), and
  so are the **median's** (0.248 → 0.170 → 0.075 → 0.045) — **no K = 2 anomaly on VitalDB.**
- This is what the operator explanation of the anomaly predicts (post-hoc reading): the anomaly appears only where the
  median beats the mean. Across RDDM's own corpora the one place the median does beat the mean (**CAPNO**, K = 3:
  −0.063 [−0.112, −0.026]) is also the one place the anomaly reappears (median doubling gains 0.117 → **0.150** → 0.062); where
  median ≈ mean (VitalDB, WESAD, DALIA) there is none.

## 5. Functional-error dependence
**A-M1 (primary mechanism, VitalDB patient level) holds.** 1,153 patients with ≥ 8 eligible windows:

| Spearman with patient consensus gain G_p | point | 95 % patient bootstrap | sign |
|---|---|---|---|
| **mean cross-sample functional-error correlation ρ̄_p** | **−0.414** | **[−0.460, −0.366]** | negative in 100 % |
| functional SD | +0.387 | [+0.333, +0.435] | positive in 100 % |
| functional MAD | +0.332 | [+0.278, +0.384] | positive in 100 % |
| waveform pairwise RMS of the 16 generated waveforms | **−0.126** | [−0.183, −0.069] | **negative in 100 %** |

`|r_err| − |r_wave|` = +0.289 [+0.222, +0.357]. On an independently developed diffusion model, the gain again tracks the
non-redundant part of the functional errors, and waveform diversity again does not explain it — here it is weakly
**opposite** to the "more diverse waveforms → more gain" story.
The corpus-level quantity matches the small gain: RDDM's VitalDB samples are highly redundant, **ρ̄ = 0.890**
(K_eff ≈ 1.1 of 16 under the mean-estimator intuition), higher than every model-depth condition of ours (0.56–0.86).
*Post-hoc scale check:* patients with larger errors have more shared error (Spearman(ρ̄_p, I_p) = +0.37), which would
bias the absolute association towards positive; against the relative gain G_p / I_p the association is −0.56
[−0.60, −0.51].

**A-M3 (across corpora, descriptive, no inference) goes the wrong way in absolute units:** Spearman(ρ̄, G) over the 5
corpora = **+0.9** (VitalDB has both the highest ρ̄ and the largest absolute G). The corpora differ in error scale by
6× (C_1 1.38 → 8.75 bpm), so absolute G mixes redundancy with difficulty. *Post-hoc:* against the relative gain G / I the
ordering is −0.9 (VitalDB 6 %, DALIA 14 %, CAPNO 16 %, WESAD 18 %, BIDMC 20 %). The preregistered descriptive result
is reported as it came out (+0.9); the scale-normalised reading is post-hoc.

## 6. Original-corpus supporting results (≈ 80 % RDDM training subjects — not generalisation evidence)
| corpus | subjects / windows | C_1 | C_16 | C_16 − C_1 | win rate | G_16 | ρ̄ | median − mean at K = 3 |
|---|---|---|---|---|---|---|---|---|
| WESAD | 15 / 21,711 | 2.800 | 2.287 | −0.513 [−0.587, −0.442] | 100 % | 0.510 | 0.706 | +0.006 [−0.017, +0.025] |
| CAPNO | 42 / 5,040 | 2.304 | 1.887 | −0.418 [−0.591, −0.268] | 95 % | 0.357 | 0.699 | **−0.063 [−0.112, −0.026]** |
| DALIA | 15 / 32,368 | 3.251 | 2.772 | −0.479 [−0.549, −0.411] | 100 % | 0.467 | 0.763 | +0.010 [−0.009, +0.033] |
| BIDMC | 51 / 6,120 | 1.384 | 1.116 | −0.268 [−0.364, −0.188] | 92 % | 0.285 | 0.653 | −0.016 [−0.047, +0.008] |

Consensus improves every corpus (relative reduction 15–19 % at K = 16, vs 6 % on VitalDB). Hamilton-extractor HR also
improves everywhere (e.g. DALIA 3.47 → 2.92).

## 7. Statistical caveats
- One checkpoint, one depth (T = 10), one functional family (HR); K raises the budget — no depth / width statement.
- VitalDB is zero-shot for RDDM and RDDM underperforms a model-free baseline there; the consensus result is about the
  relative behaviour of this checkpoint's samples, not about a useful external ECG generator.
- The mechanism evidence is associational (patient-level Spearman across 1,153 units) and absolute G is scale-dependent;
  the relative-gain readings are post-hoc.
- The original corpora contain RDDM training subjects and small subject counts (15 for WESAD / DALIA); treated as supporting.
- Condition encoders were recomputed per draw (deterministic; identical outputs); NFE per window = 20 K denoiser calls.

## 8. External replication verdict: **PARTIAL**
| preregistered item | result |
|---|---|
| A-C1 primary consensus (C_16 < C_1 and G_16 > 0) | **holds** |
| A-C2 K = 3, 4, 8 each better than K = 1 | **holds** |
| A-C3(i) median ≡ mean at K = 2 | holds (trivially) |
| A-C3(ii) median better than mean at K = 3 | **fails** (+0.018 [−0.001, +0.036]) |
| A-C3(iii) mean doubling gains non-increasing | holds |
| A-M1 patient-level Spearman(ρ̄_p, G_p) < 0 | **holds** (−0.414 [−0.460, −0.366]) |
| instability flag (a) K = 1 not better than PPG peaks | **raised** (+1.23 [+0.98, +1.48]) |
| instability flag (b) coverage < 0.90 | not raised (1.000) |

Per the frozen partition: A-C1 holds, but A-C3(ii) fails and the instability flag is raised → **PARTIAL**.
**What generalised:** multiple-sample median consensus reduces the functional error of an independently developed,
released diffusion checkpoint at every K, on an external corpus and on all four of its own corpora; and the size of that
gain is again governed by the redundancy of the samples' functional errors, not by waveform diversity.
**What did not generalise:** the robust-median advantage over the mean and the associated K = 2 anomaly (both absent on
VitalDB; RDDM's sample errors are highly redundant and not heavy-tailed), the magnitude of the gain (6 % vs ~34 %), and
useful absolute zero-shot performance (worse than PPG peaks; beats ~109 ms late).
