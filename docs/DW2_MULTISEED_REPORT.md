# DW2 part A — Width beats depth in 3/3 training seeds for all three models; the width-heavy optimum is seed-stable

Prereg `90fbc9b` (part A), frozen before any number. V1 VitalDB test (1,156 patients, 19,543 windows), B = 32 NFE per
window, HR error patient-macro, median pooling, noise seeds 0 … K−1 — all unchanged from DW1. Checkpoints: iMF and
PENGUIN seeds 42 / 1 / 2 from V1 + SR1; consistency distillation seed 42 from CD1 and **seeds 1 / 2 distilled for DW2 from
the seed-1 / seed-2 PENGUIN with the unchanged CD1 recipe** (`outputs/dw2_D_seed{1,2}`). Conditions were fixed from the
DW1 seed-42 grid and **not re-chosen per seed**. Raw: `artifacts/dw2_multiseed/grid.csv`, `result.json`;
figure `artifacts/dw2_multiseed/dw2a_seeds.png`. Script `scripts/dw2_multiseed.py` (2,000-replicate patient-clustered
bootstrap, seed 20260911).

## HR error (bpm) per seed, and 3-seed mean ± SD
| model | allocation (K, S) | seed 42 | seed 1 | seed 2 | mean ± SD |
|---|---|---|---|---|---|
| **iMF** | width (32, 1) | 6.45 | 6.18 | 6.22 | 6.29 ± 0.15 |
| | DW1 best (16, 2) | **6.18** | **6.12** | **6.18** | **6.16 ± 0.04** |
| | depth (1, 32) | 8.10 | 7.97 | 8.70 | 8.26 ± 0.39 |
| **consistency distillation** | width (32, 1) = DW1 best | **6.14** | **6.20** | **6.44** | **6.26 ± 0.16** |
| | balanced (16, 2) | 6.25 | 6.47 | 6.98 | 6.57 ± 0.37 |
| | depth (1, 32) | 8.06 | 11.73 | 10.62 | 10.14 ± 1.88 |
| **PENGUIN (Euler)** | width (32, 1) | 7.37 | 7.29 | 7.53 | 7.40 ± 0.13 |
| | DW1 best (8, 4) | **6.30** | **6.46** | **6.55** | **6.44 ± 0.12** |
| | depth (1, 32) | 9.27 | 11.03 | 10.96 | 10.42 ± 1.00 |

## Paired difference against the depth extreme (1, 32), per seed [95 % patient-clustered CI]; per-patient win rate
| model | arm − depth | seed 42 | seed 1 | seed 2 | CI upper < 0 in |
|---|---|---|---|---|---|
| iMF | width − depth | −2.20 [−2.37, −2.05] · 74 % | −2.00 [−2.15, −1.86] · 80 % | −2.79 [−2.96, −2.62] · 81 % | **3 / 3** |
| | best − depth | −2.42 [−2.57, −2.29] · 79 % | −2.06 [−2.21, −1.92] · 81 % | −2.83 [−3.00, −2.67] · 82 % | 3 / 3 |
| CD | width − depth | −2.20 [−2.37, −2.03] · 78 % | −6.14 [−6.40, −5.88] · 92 % | −4.41 [−4.66, −4.18] · 89 % | **3 / 3** |
| | balanced − depth | −2.03 [−2.19, −1.88] · 76 % | −5.75 [−6.01, −5.49] · 91 % | −3.80 [−4.00, −3.59] · 87 % | 3 / 3 |
| PENGUIN | width − depth | −1.92 [−2.17, −1.68] · 72 % | −3.74 [−4.03, −3.47] · 82 % | −3.45 [−3.76, −3.16] · 79 % | **3 / 3** |
| | best − depth | −2.98 [−3.18, −2.80] · 85 % | −4.57 [−4.77, −4.37] · 93 % | −4.43 [−4.66, −4.21] · 90 % | 3 / 3 |

## Verdict against the preregistered criteria
1. **Success criterion met for every model: `width − depth` CI upper < 0 in 3 / 3 seeds** (iMF, CD, PENGUIN), and the
   DW1 best / balanced point also beats depth in 3 / 3. The DW1 finding is not a seed-42 artefact.
2. **The optimal region is seed-stable.** The best of the three allocations is the same in every seed: iMF (16, 2) in
   3 / 3, CD (32, 1) in 3 / 3, PENGUIN (8, 4) in 3 / 3. The CD "balanced" (16, 2) point is worse than pure width in
   3 / 3 (by 0.1–0.5 bpm), and PENGUIN pure width is worse than (8, 4) in 3 / 3 (by 0.8–1.1 bpm) — both orderings
   replicate exactly as DW1 had them.
3. **"Mostly width + a small amount of depth" holds, with the model-specific amount of depth DW2 part B predicted:**
   S = 1 for CD, S = 2 for iMF (0.04–0.28 bpm better than S = 1), S = 4 for PENGUIN. The width-heavy allocations are
   also the *stable* ones: their 3-seed SD is 0.04–0.16 bpm, whereas pure depth varies by 0.39 (iMF), 1.00 (PENGUIN)
   and 1.88 bpm (CD) across training seeds. Where the budget goes into one long trajectory, the result inherits the
   seed-to-seed variance of the single-sample model; where it goes into samples, the median averages it out.
4. **PENGUIN's width value is reported, not read as a consensus gain, as preregistered:** its (32, 1) mean 7.40 sits
   1.3 bpm below the 32-draw single-sample mean (8.68, DW2-B erratum), consistent with the near-identical S = 1 samples
   (pairwise RMS 0.07) giving little to pool.
5. Consistency check: PENGUIN's (1, 32) Euler values 9.27 / 11.03 / 10.96 track the same seeds' shipped 50-NFE Heun
   results (SR1: 9.69 / 11.04 / 11.19), so the depth extreme is a faithful stand-in for the reference sampler.

Three training seeds; VitalDB only (WildPPG single-seed allocation result in WD1 part B). No result-dependent choice was
made after the numbers: the conditions, pooling rule and criterion are those of the preregistration.
