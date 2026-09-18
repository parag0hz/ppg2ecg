# PZ3 — Per-patient comparison against PENGUIN: we win on rate, lose on shape

Prereg `af3954a`, frozen before any number. No training. V1 test: 1,156 patients; HR and F1 reuse the SR1 seed-42
per-window arrays; `morph_corr@GT` was generated here (4 noise draws per arm). Template arms are scored on the 9,760
windows that follow each case's 8-window calibration block. Raw: `artifacts/pz3_per_patient/`.

| metric | arm | value | PENGUIN-50 | win rate over patients [95% CI] | verdict |
|---|---|---|---|---|---|
| HR error ↓ | PENGUIN 1 NFE | 8.47 | 9.69 | 0.684 [0.657, 0.713] | WINS FOR MOST |
| | **iMF 1 NFE** | 10.14 | 9.69 | 0.407 [0.380, 0.436] | **LOSES FOR MOST** |
| | small DiT 1 NFE | 8.71 | 9.69 | 0.691 [0.666, 0.719] | WINS FOR MOST |
| | **iMF 1 NFE, K = 16 consensus** | **6.67** | 9.69 | **0.942 [0.928, 0.955]** | **WINS FOR MOST** |
| R-peak F1 ↑ | PENGUIN 1 NFE | 0.756 | 0.645 | 0.962 [0.951, 0.973] | WINS FOR MOST |
| | iMF 1 NFE | 0.674 | 0.645 | 0.849 [0.828, 0.869] | WINS FOR MOST |
| | small DiT 1 NFE | 0.629 | 0.645 | 0.413 [0.385, 0.442] | LOSES FOR MOST |
| morph_corr@GT ↑ | PENGUIN 1 NFE | 0.359 | 0.179 | 0.949 [0.936, 0.961] | WINS FOR MOST |
| | iMF 1 NFE | 0.197 | 0.179 | 0.687 [0.661, 0.713] | WINS FOR MOST |
| | small DiT 1 NFE | 0.150 | 0.179 | 0.324 [0.298, 0.351] | LOSES FOR MOST |
| | population template (GT timing) | 0.676 | 0.179 | 0.996 | diagnostic |
| | 32 s personal template (GT timing) | **0.743** | 0.179 | 0.996 | diagnostic |

## How to read `morph_corr@GT` — the caveat that matters
Beats are cut at the **target's** R peaks for every arm. For the two template arms the beats are also *stamped* at
those positions, so they are scored on shape alone. A generator places its own beats, so any misplacement shows up
here as low correlation: for generators this axis is **placement and shape together**, and the template rows are an
upper reference, not a fair opponent. That gap (0.74 vs 0.15–0.36) is the same statement as N1's "the null method
dominates", measured per patient: **99.6 % of patients are better served by stamping a template at the right
positions than by any generator's waveform.**

## By patient difficulty (SBV tertiles from PZ2)
| arm | SBV low | SBV mid | SBV high |
|---|---|---|---|
| iMF 1 NFE (win rate vs PENGUIN-50) | 0.633 | 0.719 | 0.714 |
| small DiT 1 NFE | 0.328 | 0.314 | 0.331 |
| 32 s personal template (value) | 0.788 | 0.770 | 0.670 |

Neither generator's advantage depends much on difficulty; the personal template degrades on hard patients (0.788 →
0.670), which is where the remaining headroom is.

## Summary
- **What we win:** beat rate. The consensus arm beats PENGUIN-50 on **94 % of patients** at 16 NFE against 50.
- **What we lose:** individual beat placement and shape at one sample. The small DiT trades F1 and morphology for HR;
  iMF-1 loses HR to PENGUIN-50 on most patients even though it wins on the mean in some seeds.
- **What neither method touches:** the template gap. Morphology is still better served by stamping than by generating.
