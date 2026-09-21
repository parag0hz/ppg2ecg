# MC1 — Three more ECG corpora: **HR consensus holds on 5 of 5 corpora; consensus decoding does not transfer to the small ones**

Prereg `e193d62`, frozen before any MC1 weight update. PPG-DaLiA (15 subjects, 4 test), BIDMC (51, 15 test),
CapnoBase (42, 13 test); subject-level holdout, unchanged recipe (14,000 steps, seed 42), ED1 decoding parameters
transferred without re-tuning. Cluster = subject. Raw: `artifacts/mc1_multicorpus/*.json`.

## Verdicts (frozen rules)
| corpus | H2: iMF K=16 HR consensus < PENGUIN-50 | H2b: CD consensus < PENGUIN-50 | H-ED: decoded F1 ≥ single + 0.05 (iMF / CD) |
|---|---|---|---|
| PPG-DaLiA | **holds** −0.72 [−1.29, −0.20] | **holds** −2.10 [−2.33, −1.81] | fails −0.022 / fails +0.008 |
| BIDMC | **holds** −2.41 [−3.52, −1.12] | **holds** −2.37 [−3.55, −1.17] | fails +0.020 / fails +0.011 |
| CapnoBase | **holds** −2.05 [−4.20, −0.29] | fails −0.26 [−1.19, +0.71] | fails +0.012 / fails +0.037 |

## All five corpora, one table (HR error in bpm ↓ · R-peak F1 ↑)

| corpus (test subjects) | PENGUIN 50 NFE | iMF 1 NFE | **iMF K=16 HR consensus** | consensus − PENGUIN-50 | iMF decoded F1 (single → decoded) |
|---|---|---|---|---|---|
| VitalDB (1,156) | 9.69 · 0.645 | 10.14 · 0.674 | **6.67** | **−3.01** [−3.13, −2.90] | 0.674 → **0.765** |
| WildPPG (14, 4-fold) | 13.66 · 0.353 | 12.61 · 0.371 | **9.27** | **−4.39** [−4.89, −3.89] | 0.371 → **0.439** |
| BIDMC (15) | 8.93 · 0.550 | 7.94 · 0.584 | **6.53** | **−2.41** [−3.52, −1.12] | 0.584 → 0.604 |
| CapnoBase (13) | 7.54 · 0.601 | 6.75 · 0.652 | **5.51** | **−2.05** [−4.20, −0.29] | 0.652 → 0.664 |
| PPG-DaLiA (4) | 13.42 · 0.147 | 16.74 · 0.147 | **12.70** | **−0.72** [−1.29, −0.20] | 0.147 → 0.125 |

## Full metric set on the new corpora (single seed; patient-macro means)
| corpus | arm | HR | F1 | RR-MAE | MAE | RMSE | FD | Micro-F1 | morph@GT |
|---|---|---|---|---|---|---|---|---|---|
| DaLiA | PENGUIN-50 | 13.42 | 0.147 | 31.3 | 0.384 | 0.477 | 27.5 | 0.151 | 0.000 |
| | PENGUIN-1 | 37.63 | 0.055 | 34.3 | 0.329 | 0.383 | 47.8 | 0.062 | 0.002 |
| | iMF-1 | 16.74 | 0.147 | 31.5 | 0.405 | 0.492 | 32.0 | 0.150 | 0.001 |
| | CD-1 | 14.77 | 0.147 | 31.0 | 0.381 | 0.458 | 33.6 | 0.151 | −0.001 |
| | iMF decoded | 24.06 | 0.125 | 30.8 | 0.343 | 0.408 | 37.7 | 0.127 | 0.002 |
| BIDMC | PENGUIN-50 | 8.93 | 0.550 | 17.9 | 0.357 | 0.449 | 19.0 | 0.571 | 0.143 |
| | PENGUIN-1 | 7.10 | 0.568 | 15.8 | 0.357 | 0.446 | 19.8 | 0.586 | 0.146 |
| | iMF-1 | 7.94 | 0.584 | 14.7 | 0.366 | 0.461 | 18.9 | 0.598 | 0.124 |
| | CD-1 | 7.03 | 0.579 | 18.5 | 0.371 | 0.465 | 15.3 | 0.597 | 0.159 |
| | iMF decoded | 6.70 | 0.604 | 14.7 | 0.347 | 0.440 | 31.6 | 0.616 | 0.139 |
| CapnoBase | PENGUIN-50 | 7.54 | 0.601 | 11.9 | 0.333 | 0.405 | 7.7 | 0.659 | 0.202 |
| | PENGUIN-1 | 5.86 | 0.608 | 12.1 | 0.327 | 0.400 | 9.4 | 0.665 | 0.195 |
| | iMF-1 | 6.75 | 0.652 | 12.1 | 0.334 | 0.406 | 5.9 | 0.706 | 0.211 |
| | CD-1 | 7.83 | 0.618 | 13.3 | 0.341 | 0.413 | 7.1 | 0.671 | 0.203 |
| | iMF decoded | 7.06 | 0.664 | 11.1 | 0.302 | 0.377 | 11.9 | 0.711 | 0.221 |

## Reading it
1. **The HR-consensus claim now holds on five corpora out of five for iMF** (four of five for CD), spanning surgery,
   ICU, anaesthesia and two wearable datasets. This is the robust result of the programme.
2. **Consensus decoding is corpus-dependent.** It clears the +0.05 F1 bar on VitalDB (+0.091) and WildPPG (+0.068) and
   misses it on all three MC1 corpora (+0.020, +0.012, −0.022). On **PPG-DaLiA it is harmful** (HR 16.7 → 24.1): the
   samples do not agree on where beats are, exactly the failure P1 found on the same dataset, whose wrist-PPG / chest-ECG
   streams are only coarsely synchronised. Decoding needs sample agreement on positions; where that is absent it
   should not be used, and the vote statistics give a way to detect that.
3. **On PPG-DaLiA nothing works at the beat level**: every arm's F1 is ≈ 0.15 and `morph_corr@GT` ≈ 0, as D2 and U1
   already reported. Only the rate-level consensus gains anything there (−0.72 bpm).
4. **PENGUIN at 1 NFE is again strong on BIDMC and CapnoBase** (HR 7.10 and 5.86, better than its own 50-NFE setting)
   and collapses on DaLiA (37.6). The "collapsed mean keeps the rhythm when the rhythm is regular" pattern repeats.
5. Small corpora, single seed, 13–15 test subjects (4 for DaLiA): the intervals are wide and these rows are
   supporting evidence, not headline numbers.
