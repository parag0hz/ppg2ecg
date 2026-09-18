# PZ2 — Calibration budget and a per-patient difficulty axis (preregistration; no training)

Frozen before any PZ2 number. Same data, templates and morphology-only protocol as PZ1 (`eb516f8`): V1 VitalDB test
patients, GT R peaks for every arm, 83-sample beats, `build_template_a`.

- **Calibration budget.** The template is built from the first **k ∈ {1, 2, 4, 8} windows** (4, 8, 16, 32 s) of each
  case. To keep the comparison exact, **every k is scored on the same windows: those after the 8th** (8 windows per
  case). GLOBAL (population) and ORACLE (the scored window's own beats) are scored on the same set.
- **Saturation rule (stated now).** The budget is called **saturated at k** if `morph_corr(2k) − morph_corr(k)` < 0.01.
- **Difficulty axis — SBV (subject beat variability).** For each patient, `SBV = median over calibration beats of
  (1 − Pearson(beat, that patient's calibration template))`, computed on the calibration block only (k = 8), so it
  never sees a scored window. It is the morphology analogue of the SDS difficulty metric of PPG2BP-net
  (Sci Rep 2023), which measures intra-subject variation from the calibration value.
- **Reported:** morph_corr, QRS-core RMSE and MAE per k; personalisation gain (SUBJ − GLOBAL) and the fraction of the
  ORACLE gap closed, **split by SBV tertile**; the per-patient gain distribution. Descriptive — no pass/fail claim
  beyond the saturation rule.
