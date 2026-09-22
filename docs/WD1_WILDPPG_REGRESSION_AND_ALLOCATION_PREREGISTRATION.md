# WD1 — WildPPG: direct HR regression, and the width-vs-depth ordering on a wearable corpus (preregistration)

Frozen and pushed before any WD1 weight update. WP1's 4 subject folds (14 subjects, each tested once); per-subject
3,000 evaluation windows (the WP1 set); no tuning on any test fold.

## Part A — direct HR regression (DB1's model, unchanged)
- DB1's 1-D CNN (0.91 M params), L1 on HR/100, AdamW 1e-3, wd 0.01, batch 64, **14,000 steps**, seed 42, trained per
  fold on that fold's train subjects. Input is the same 4 s / 512-sample window, so **no architectural change**.
- Compared, paired on the same windows: PPG peak counting; PENGUIN 50 NFE (one sample); iMF K = 16 consensus; CD K = 16
  consensus; and the WildPPG best allocation from Part B.
- Reported: overall HR MAE (subject-macro, 14-subject clustered CI), per fold, per subject, paired differences.
- **Outcome classes stated now:** (A) regressor clearly better (CI of `reg − consensus` < 0); (B) consensus better
  (CI > 0); (C) CI spans 0 → comparable. Whichever holds is reported as such.

## Part B — width vs depth at B = 32 (three points per generator)
- iMF, CD, PENGUIN(Euler), WP1 fold checkpoints; conditions (K 32, S 1), (K 16, S 2), (K 1, S 32); median pooling.
- **Hypothesis.** width-heavy beats pure depth (`(32,1) − (1,32)` CI upper < 0) for each generator, as on VitalDB.
  Ordering is the primary question; absolute HR is secondary.
