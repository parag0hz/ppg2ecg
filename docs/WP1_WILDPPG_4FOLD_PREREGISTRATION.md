# WP1 — Second corpus: WildPPG, subject-level 4-fold, arms PENGUIN / iMF / consistency distillation (preregistration)

Frozen and pushed before any WP1 weight update. Question: **do the VitalDB claims transfer to a wearable corpus?**

## Data and folds
- The 14 WildPPG subjects of the U2 corpus (`data/processed/u2_wildppg`, upstream preprocessing, 4 s @ 128 Hz;
  `kjd` / `ssx` are not in it and are asserted absent).
- `numpy.random.default_rng(20260919).permutation` of the 14 → test blocks of 4, 4, 3, 3; within each fold the first
  two of the remaining permutation order are validation, the rest train (8 or 9). **Every subject is tested exactly
  once.** Manifests `data/manifests/split_wp1_fold{0..3}.json` (`scripts/wp1_build_folds.py`).
- Evaluation windows: exact `linspace` of **3,000 windows per test subject** (of ~44–55 k), fixed now for compute.

## Arms, per fold (seed 42, 14,000 steps each; argv = the U2 / V1 recipe)
- **C** PENGUIN (OT-CFM) — `train_a0`. Evaluated at NFE 50 and 1.
- **I** iMF on the PENGUIN backbone — `train_a2`. NFE 1 + HR consensus K = 16.
- **D** consistency distillation from **that fold's** C — `scripts/cd1_train.py`, the CD1 recipe. NFE 1 + K = 16.

## Claims (pooled over folds; cluster = subject, n = 14; margins HR 1.0 bpm, R-peak F1 0.02)
- **H1** iMF-1 non-inferior to PENGUIN-50 (HR upper CI < +1.0 and F1 lower CI > −0.02).
- **H2** iMF-1 K = 16 consensus beats one PENGUIN-50 sample (upper CI < 0).
- **H2b** CD-1 K = 16 consensus beats one PENGUIN-50 sample (upper CI < 0).
- **H4** CD-1 vs iMF-1 — the BB1 rule (IMPROVES / WORSE / MIXED / NO MEANINGFUL CHANGE).
- Reported, not gated: per-fold values, per-subject HR win rates against PENGUIN-50, FD per fold.
- **Stated limitation, before the result:** with 14 subjects the subject-level CIs will be wide; a non-significant
  result here is weak evidence either way and will be reported as such.
