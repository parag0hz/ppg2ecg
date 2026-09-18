# PZ1 — How much is personalisation worth? (preregistration; no training)

Frozen and pushed before any PZ1 number. Question: given the beat positions, how much better is **that patient's own
beat template** than a population template? This bounds what any personalised generator can gain on morphology,
because N2 showed the shape information is in the template, not in the PPG.

| | |
|---|---|
| Data | V1 VitalDB **test** patients (1,156 cases, 16 windows each), 4 s @ 128 Hz |
| Split inside a case | the **first 4 windows by window index** are the calibration block; the remaining 12 are scored. Disjoint, so a template never sees a scored window |
| Beat geometry | frozen `stamping.template_geometry()`: 83 samples, R at index 32; template = `build_template_a` (median beat, median peak-to-peak scaling) |
| Positions | GT R peaks of the target ECG (neurokit), identical for every arm — this is a morphology-only test |

## Arms
| arm | template |
|---|---|
| **GLOBAL** | median beat over V1 **train** patients (200 cases × first 2 windows, fixed) |
| **SUBJ** | median beat of that patient's calibration block (≥ 5 beats, else GLOBAL, counted) |
| **SUBJ-RR** | SUBJ stretched in time by (median RR of the scored window) / (median RR of the calibration block) |
| **ORACLE** | median beat of the scored window itself — the upper bound |

## Metrics (per scored window, patient-clustered bootstrap 2,000, seed 20260911)
`morph_corr` = mean per-beat Pearson correlation on the 83-sample beat window (primary); `qrs_core_rmse` on
[r−10, r+15]; whole-window PCC; MAE.

## Decision
**PERSONALISATION WORTH IT** iff SUBJ − GLOBAL on `morph_corr` is ≥ **+0.05** with the 95 % CI excluding 0
(the N2 bar). Otherwise **NOT WORTH IT AT THIS SCALE**. Also reported, not gated: the fraction of the
(ORACLE − GLOBAL) gap that SUBJ closes, SUBJ-RR, and the per-patient distribution of the gain.
