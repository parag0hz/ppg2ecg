# PZ3 — Per-patient comparison against PENGUIN, on the personalised axes (preregistration)

Frozen before any PZ3 number. No training: the V1 / SR1 checkpoints at 14,000 steps, seed 42. Question: **for how many
individual patients does our method beat PENGUIN, and does the answer depend on how hard the patient is?**

- **Data.** V1 VitalDB test: 1,156 patients, 19,543 windows. Noise seeds 0–3, per-window metric averaged over seeds.
- **Arms.** PENGUIN 50 NFE (reference), PENGUIN 1 NFE, iMF 1 NFE (arm I), small vanilla DiT 1 NFE (arm S at its frozen
  VM1 CFG), iMF 1 NFE with HR consensus K = 16, and — diagnostic, GT timing — the 32 s calibrated personal template of
  PZ2 (`k8`) and the population template (`GLOBAL`).
- **Metrics.**
  1. `HR error` and `R-peak F1` — the V1 definitions, averaged per patient.
  2. `morph_corr@GT` — the PZ1 measure applied to a generated waveform: beats cut at the **target's** R peaks from the
     generated signal, per-beat Pearson against the target beats. Morphology with timing held fixed.
  3. `SBV` — the PZ2 difficulty number of that patient (calibration block only).
- **Reported per arm vs PENGUIN-50.**
  * **Win rate** = fraction of patients on which the arm is better, with a 95 % bootstrap CI over patients, per metric.
  * Patient-clustered mean difference with CI (the V1 statistic).
  * Win rate and mean difference **split by SBV tertile** (low / mid / high), using PZ2's tertile edges.
- **Decision (per arm, per metric).** **WINS FOR MOST PATIENTS** if the win-rate CI lies above 0.5; **LOSES FOR MOST**
  if it lies below 0.5; otherwise **SPLIT**. No metric is dropped or added after the results; `morph_corr@GT` is the
  primary personalised axis because PZ1 / PZ2 are defined on it.
