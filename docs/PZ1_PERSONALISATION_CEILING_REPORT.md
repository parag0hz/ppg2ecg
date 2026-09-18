# PZ1 — Personalisation ceiling on VitalDB: **PERSONALISATION WORTH IT** (just over the bar)

Prereg `eb516f8`, frozen before any number. No training, no model: beat positions are the GT R peaks for every arm,
so this measures **morphology only**. V1 test patients; each case's first 4 windows (by window index) are the
calibration block and the remaining 12 are scored — disjoint, so no template ever sees a scored window.
14,627 scored windows, 1,155 patients, 0 fallbacks to the population template. Raw: `artifacts/pz1_personalisation/`.

| arm | template | morph_corr ↑ | QRS-core RMSE ↓ | window PCC ↑ | MAE ↓ |
|---|---|---|---|---|---|
| GLOBAL | 200 train patients | 0.6975 [0.689, 0.706] | 0.3378 | 0.3763 | 0.3839 |
| **SUBJ** | the patient's own calibration block | **0.7508 [0.742, 0.759]** | **0.3031** | **0.4261** | **0.3527** |
| SUBJ-RR | SUBJ stretched by the RR ratio | 0.6660 | 0.3211 | 0.3864 | 0.3610 |
| ORACLE | the scored window's own beats | 0.8688 | 0.1573 | 0.5101 | 0.2504 |

**Verdict: PERSONALISATION WORTH IT.** SUBJ − GLOBAL = **+0.0533 [+0.0485, +0.0581]** on `morph_corr`, at the frozen
+0.05 bar with the CI clear of 0. It closes **31 %** of the GLOBAL → ORACLE gap and helps **84 % of patients**.

## Reading it
1. **The gain is real but small, and it sits just above the bar** (0.0533 vs 0.050). A different bar would have
   flipped the verdict; the CI, not the point estimate, is what makes it defensible.
2. **Two thirds of the achievable morphology gap is NOT covered by a static personal template** (31 % closed). The rest
   is within-patient, window-to-window variation — which is what a generator conditioned on PPG could in principle
   supply, and is the same quantity N2 found the PPG does not carry at the single-window level.
3. **RR stretching hurts** (−0.031 vs GLOBAL): rescaling the whole 83-sample beat by the RR ratio distorts the QRS,
   whose width does not scale with heart rate. If personalisation is pursued, the template must be conditioned, not
   warped this way.
4. Scope: anaesthetised VitalDB patients, regular rhythm. A wearable cohort with more rhythm variation may behave
   differently in either direction.
