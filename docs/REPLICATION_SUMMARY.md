# Replication Summary — does the frozen one-step iMeanFlow effect replicate?

Pre-registration: `docs/A3_A4_REPLICATION_PREREGISTRATION.md`. All runs: seed 42, identical backbone/objectives/recipes; only the split (A3) or the dataset (A4) changed.

## Cross-experiment comparison (test set, paired noise seed 0)
| Experiment | Dataset | Test subjects | OT50 HR | OT4 HR | OT1 HR | iMF1 HR | OT50 Morph | OT1 Morph | iMF1 Morph | OT50 Amp | OT1 Amp | iMF1 Amp | OT50 Gain | OT1 Gain | iMF1 Gain |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 | PPG-DaLiA | S2 | 8.08 | 15.76 | 41.96 | **9.58** | 0.650 | 0.217 | **0.595** | 0.95 | 0.15 | **0.90** | 5.69 | 0.24 | **4.47** |
| A3 | PPG-DaLiA | S1 | 8.16 | 16.40 | 35.23 | **11.96** | 0.683 | 0.168 | **0.581** | 0.87 | 0.21 | **0.71** | 8.77 | 0.28 | **4.78** |
| A4 | WildPPG | kjd, ssx | 9.43 | 15.37 | 15.59 | **11.85** | 0.670 | 0.379 | **0.551** | 0.98 | 0.32 | **1.04** | 7.16 | 6.64 | **4.29** |

## Recovery of the 50→1 NFE gap by iMeanFlow at 1 NFE
| Experiment | HR | Morphology | Amplitude | Conditioning | beats/ref (iMF1) | A2 rule | Replication rule | Pointwise-error inversion |
|---|---:|---:|---:|---:|---:|---|---|---|
| A2 (PPG-DaLiA S2) | +0.96 | +0.87 | +0.93 | +0.78 | 1.00 | SUCCESS | **REPLICATED** | YES |
| A3 (PPG-DaLiA S1) | +0.86 | +0.80 | +0.76 | +0.53 | 1.03 | SUCCESS | **REPLICATED** | YES |
| A4 (WildPPG kjd, ssx) | +0.61 | +0.59 | +0.90 | -4.47 | 0.93 | PARTIAL | **PARTIAL** | YES |

## Ordering test (the scientific claim)
| Experiment | A: OT1 ≪ OT50 (HR) | B: iMF1 ≫ OT1 (HR) | C: iMF1 → OT50 (HR gap left, bpm) | A (morph) | B (morph) | C (morph gap left) |
|---|---|---|---:|---|---|---:|
| A2 | ✔ | ✔ | +1.50 | ✔ | ✔ | -0.055 |
| A3 | ✔ | ✔ | +3.80 | ✔ | ✔ | -0.102 |
| A4 | ✔ | ✔ | +2.42 | ✔ | ✔ | -0.119 |

## Training
| Experiment | OT-CFM epochs/rounds (best) | iMF epochs/rounds (best) | OT-CFM h | iMF h |
|---|---|---|---:|---:|
| A2 | 85 (65) | 81 (61) | 1.8 | 3.2 |
| A3 | 114 (94) | 36 (16) | 2.4 | 1.4 |
| A4 | 210 (190) | 66 (46) | 5.1 | 3.6 |

## Overall verdict: **SUBJECT-ROBUST, DATASET-UNCERTAIN**

(Rule, prereg Part II §8: STRONG = A3 and A4 replicated; SUBJECT-ROBUST/DATASET-UNCERTAIN = A3 replicated, A4 not; DATASET-ROBUST/SUBJECT-UNCERTAIN = A4 replicated, A3 not; MIXED = partial somewhere; NOT ROBUST = only A2.)

## Scientific interpretation
- **What replicates everywhere (subject and dataset):** the pre-registered ordering for HR and morphology — OT-CFM at 1 NFE loses
  amplitude and QRS sharpness (A), Improved MeanFlow at 1 NFE restores them (B) and approaches the 50-NFE OT-CFM reference (C: residual
  +1.5 / +3.8 / +2.4 bpm HR, −0.06 / −0.10 / −0.12 template correlation on S2 / S1 / WildPPG). Amplitude recovery is 0.76–0.93 in all
  three experiments; the pointwise-error inversion (the physiologically worst arm has the best RMSE/MAE) also replicates three times.
- **What does not replicate uniformly:** the *nature* of the OT-CFM one-step failure. On PPG-DaLiA (wrist/chest devices not
  beat-synchronised) the one-step conditional mean is a beat-free flat line, so iMeanFlow recovers rhythm, amplitude, morphology and
  conditioning together (A2, A3 → REPLICATED). On WildPPG (four synchronised devices) the conditional mean is an *aligned but attenuated
  and smoothed* ECG: it already carries rhythm (HR 15.6 bpm), the PPG dependence (shuffle gain 6.6 of 7.2 bpm) and the best beat timing
  of all arms (R-peak F1 0.48); iMeanFlow-1 then improves HR, amplitude and sharpness but shows a weaker PPG dependence (gain 4.3) and
  less precise beat placement (F1 0.39) → PARTIAL.
- **Integrated verdict: SUBJECT-ROBUST, DATASET-UNCERTAIN.** The one-step recovery of physiological *waveform structure* (amplitude,
  QRS sharpness, template correlation) is robust; the claim that one-step generation recovers *conditional rhythm fidelity* beyond what a
  one-step conditional mean provides holds only where the conditional mean is uninformative about beats (unsynchronised pairing).
- Secondary: multi-step MeanFlow (2–4 NFE) improves morphology on every dataset and, on DaLiA, beats the 50-NFE OT-CFM reference on HR;
  on WildPPG it does not close the HR/conditioning gap.

## Limitations
- One training seed per arm; one (A2/A3) or two (A4) test participants per experiment; A3 shares the validation subject with A2.
- Beat-level metrics (R-peak F1, PCC, RR MAE) are uninformative on raw PPG-DaLiA (device synchronisation) but informative on WildPPG;
  the two datasets therefore test different aspects of the effect.
- WildPPG evaluation uses a 4,096-window uniform subset of the test recordings and a 220-step validation round (pre-registered);
  4 PPG sites pooled as in PENGUIN; constant-gap windows dropped (0.22 %).
- The conditioning-gain recovery score is ill-conditioned when the OT-CFM 1-NFE baseline already retains the gain (A4).
- All runs use the OT-CFM baseline optimiser for the objective-only comparison; the official iMF recipe (EMA, lr 1e-4, aux head) was not used.

## A5 — Conditional-mean control (added 2026-08-27; `docs/A5_CONDITIONAL_MEAN_CONTROL_REPORT.md`)
An MSE regressor on the same S5 backbone (generative inputs removed; 3,990,787 params, 2,907,393 effective) was trained on the A2, A3 and A4
splits and compared with the frozen OT-CFM-1/OT-CFM-50/iMF-1 predictions on identical test windows.
| Dataset | R: HR / morph / amp / gain / RMSE | closest model to R (wave RMSE) | attenuation(R) | timing+conditioning kept (WildPPG) |
|---|---|---|---|---|
| DaLiA S2 | 35.7 / 0.160 / 0.06 / 2.37 / **0.289** | OT-1 (0.085; OT-50 0.310, iMF-1 0.314) | ✓ | — |
| DaLiA S1 | 32.3 / 0.148 / 0.05 / 1.32 / **0.318** | OT-1 (0.134; 0.316, 0.290) | ✓ | — |
| WildPPG | 19.2 / 0.331 / 0.25 / 5.70 / **0.343** | OT-1 (0.079, PCC 0.52; 0.260, 0.349) | ✓ | ✓ F1 0.436 (OT-50 0.440), gain 5.70 (7.16) |
Pre-registered verdict **STRONG SUPPORT**: OT-CFM 1-NFE empirically approaches the behaviour of an MSE-trained conditional-mean proxy —
lowest RMSE of all models with the strongest amplitude/morphology attenuation (pointwise-error inversion for the regressor on 3/3
datasets) — and temporal alignment decides what the proxy keeps (beat-free on DaLiA; aligned attenuated beats on WildPPG). This is the
mechanism behind the SUBJECT-ROBUST, DATASET-UNCERTAIN pattern above: on WildPPG the one-step conditional-mean-like solution already carries
rhythm and PPG dependence, so iMF-1's gains are confined to amplitude/morphology and come with less precise beat placement (F1 0.385,
RR MAE 25.7 ms vs 16.7 ms for the regressor). Caveat: the originally pre-registered zero-state regressor was untrainable (adaLN-Zero
dead-start) and was replaced, before the amended runs, by a learned constant state token (Amendment 1).

## A6 — Capacity-matched control (added 2026-08-27; `docs/A6_CAPACITY_MATCHED_MEAN_CONTROL_REPORT.md`)
The A5 regressor had fewer effective parameters (2.9 M vs 4.3 M). A6 re-ran the MSE control on the **unmodified full PENGUIN backbone**
(4,568,707 params = OT-CFM/iMF; deterministic constant state x = 0.1, fixed t = 0.5, cond = 0.05·E(t), both fixed by a pre-registered
hard test) on the same three splits: S2 morph 0.175 / amp 0.06 / RMSE 0.286; S1 0.184 / 0.04 / 0.321; WildPPG 0.316 / 0.24 / 0.350 with
F1 0.421 and gain 4.95 — within 0.035 (morph), 0.015 (amp) and 0.007 (RMSE) of the A5 regressor, waveform RMSE between the two
0.04–0.05; OT-CFM 1-NFE remains the closest generative model (3/3 datasets, 3/3 votes). **Verdict: CAPACITY OBJECTION RESOLVED** — the
conditional-mean-like attenuation is a property of the MSE objective on this backbone, not of reduced capacity.
