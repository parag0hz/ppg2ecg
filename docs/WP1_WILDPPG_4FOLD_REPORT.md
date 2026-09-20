# WP1 — WildPPG, subject-level 4-fold: **H1, H2 and H2b all hold; consensus wins on 14/14 subjects**

Prereg `6eb9068`, frozen before any WP1 weight update. 14 WildPPG subjects, each tested exactly once across four
folds; 3,000 evaluation windows per test subject; arms trained per fold with the unchanged U2/V1 recipe (14,000
steps, seed 42). Cluster = subject (n = 14), 2,000-replicate bootstrap. Raw: `artifacts/wp1_wildppg/verdict.json`.

| arm | NFE | HR error (bpm) ↓ | R-peak F1 ↑ | RR-MAE (ms) ↓ | MAE ↓ |
|---|---|---|---|---|---|
| PENGUIN (teacher) | 50 | 13.66 [10.76, 17.70] | 0.353 | 26.8 | 0.345 |
| PENGUIN | 1 | 15.43 [13.23, 17.87] | 0.401 | **17.6** | **0.271** |
| iMF | 1 | 12.61 [10.15, 16.05] | 0.371 | 27.0 | 0.353 |
| consistency distillation | 1 | 12.32 [9.60, 16.29] | 0.382 | 27.1 | 0.328 |
| **iMF, K = 16 consensus** | 16 | **9.27 [6.54, 13.07]** | — | — | — |
| **CD, K = 16 consensus** | 16 | 9.80 [6.87, 14.07] | — | — | — |

## Verdicts (frozen rules)
- **H1 — iMF-1 non-inferior to PENGUIN-50: HOLDS.** HR −1.046 [−1.748, −0.377] (upper bound well under +1.0);
  F1 +0.018 [+0.011, +0.026].
- **H2 — iMF consensus beats one PENGUIN-50 sample: HOLDS.** −4.386 bpm [−4.891, −3.894].
- **H2b — CD consensus beats PENGUIN-50: HOLDS.** −3.858 bpm [−4.502, −3.230].
- **H4 — CD vs iMF at 1 NFE: NO MEANINGFUL CHANGE.** HR −0.298 [−1.154, +0.570]; F1 +0.011 [+0.003, +0.018].
- **Per-subject HR win rate against PENGUIN-50: 14/14 for iMF consensus, CD consensus and CD-1**; 10/14 for iMF-1.

## What transfers, and what does not
1. **The consensus result transfers, and it is larger here than on VitalDB**: −4.4 bpm against PENGUIN-50 (VitalDB:
   −3.0), with every one of the 14 subjects improved. Wearable data has more beat-placement variance for pooling to
   remove, which is exactly the mechanism the X4-0 source-sensitivity measurement predicted.
2. **CD's VitalDB advantage over iMF does not transfer** (H4 = no meaningful change here, IMPROVES on VitalDB).
   Whatever CD gained from a stable anaesthesia rhythm does not survive the wearable setting.
3. **Absolute quality is far worse on WildPPG** (PENGUIN-50 HR 13.66 vs 9.69; F1 0.353 vs 0.645). The claims are about
   *relative* behaviour; the corpus remains hard for every method, as U1 and D1 already showed.
4. **PENGUIN at 1 NFE again has the best MAE and RR-MAE and the worst FD** (fold FDs 6.9 / 3.9 / 3.8 / 32.5 for the
   teacher vs 11.4 / 6.3 / 12.6 / 17.7 for iMF-1), i.e. the collapsed-mean pattern reappears unchanged.
5. **Fold 3 is an outlier for FD across all arms** (32.5 / 17.7 / 34.5): its test subjects are the noisy-ECG ones.
   Reported, not excluded.
6. 14 subjects is a small cluster count; the intervals above are correspondingly wide, as the preregistration stated.
