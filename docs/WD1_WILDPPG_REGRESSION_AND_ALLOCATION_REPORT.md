# WD1 — WildPPG: the regressor wins again (narrowly); width beats depth for the one-step generators, not clearly for PENGUIN

Prereg `90fbc9b`, frozen before any number. WP1's four subject folds (14 subjects, each tested once; 3,000 windows per
subject), the WP1 fold checkpoints; DB1's regressor retrained per fold with no architectural change (input is the same
512-sample window). Subject-clustered bootstrap (n = 14; intervals are wide, as the preregistration warned).
Raw: `artifacts/wd1_wildppg/part_A.json`, `part_B.json`.

## Part A — direct HR regression (outcome class **A: regressor better**, but by half the VitalDB margin)
| estimator | HR error (bpm) [95% CI] | fold 0 / 1 / 2 / 3 |
|---|---|---|
| **direct HR regression** (0.91 M params, 80 s training per fold) | **8.76** [6.16, 12.17] | 6.46 / 10.42 / 13.86 / 4.28 |
| iMF K = 16 consensus | 9.27 [6.54, 13.07] | 6.65 / 10.10 / 16.05 / 4.66 |
| CD K = 16 consensus | 9.80 [6.87, 14.07] | 6.77 / 9.03 / 18.69 / 5.73 |
| PENGUIN 50 NFE (one sample) | 13.66 [10.76, 17.70] | 9.97 / 14.15 / 21.21 / 10.14 |
| PPG peak counting | 15.20 [13.63, 16.94] | 14.10 / 18.19 / 13.24 / 14.59 |

Paired: regressor − iMF consensus **−0.52 [−1.08, −0.04]** (class A; win rate 10/14 subjects); regressor − CD consensus
−1.04 [−2.37, +0.16] (class C); regressor − PENGUIN-50 −4.90; regressor − PPG peaks −6.54.

- **The boundary holds on a wearable corpus:** if only the rate is wanted, the direct estimator is the right tool.
  The margin over consensus is half of VitalDB's (0.52 vs 0.98) and the CD contrast is not significant.
- **PPG peak counting collapses on wearable data** (15.2, vs 9.06 on VitalDB): the trivial estimator does not transfer;
  both learned routes do.
- No support for outcome B: on the hard fold (2) everything degrades together (13.9 / 16.1 / 21.2), the generative
  route does not pull ahead.

## Part B — width vs depth at B = 32
| generator | (K 32, S 1) width | (K 16, S 2) | (K 1, S 32) depth | width − depth | width beats depth? |
|---|---|---|---|---|---|
| iMF | **9.12** | 9.45 | 12.44 | −3.34 [−4.78, −1.75] | **yes** (11/14 subjects) |
| consistency distillation | **9.68** | 9.86 | 14.21 | −4.65 [−6.77, −2.32] | **yes** (11/14) |
| PENGUIN (Euler) | 11.57 | **9.99** | 13.39 | −1.83 [−4.91, +0.62] | **no** (8/14); but (16, 2) − depth = −3.41 [−4.36, −2.49] |

- **For the one-step generators the VitalDB ordering transfers**: pure depth is 3.3–4.7 bpm worse than width, and the
  (32, 1) / (16, 2) pair is indistinguishable (CIs of their difference span 0).
- **For PENGUIN the picture is the DW2 mechanism in action**: at S = 1 its samples are near-identical (pairwise RMS
  0.07), so pure width (11.57) buys little; **two steps restore diversity and (16, 2) is its best point (9.99),
  significantly better than pure depth**. The width extreme itself is not significantly better than depth here.
  "Enough depth for the samples to differ, then width" describes all three generators; "S = 1 always" does not.
- PENGUIN (16, 2) at 32 NFE (9.99) is 3.7 bpm better than PENGUIN's shipped 50-NFE setting (13.66) with no retraining.
- Single seed per fold; 14 subjects.
