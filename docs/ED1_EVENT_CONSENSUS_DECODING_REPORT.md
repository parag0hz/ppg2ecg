# ED1 — Event-level consensus decoding: **PARTIAL** for both generators (F1 passes, morphology bar missed)

Prereg `93ab678`, implementation `ae6a2ce`, parameters frozen from validation patients `bc88553` — all before the
test set was touched; the test set was evaluated once. No training. VitalDB V1 test: 1,156 patients, 19,543 windows;
K = 16 one-step samples per window. The consensus waveform went through the same standard pipeline as every other
arm. Raw: `artifacts/ed1_consensus_decoding/`.

| metric | iMF single | **iMF consensus-decoded** | CD single | **CD consensus-decoded** | PENGUIN-50 |
|---|---|---|---|---|---|
| R-peak F1 ↑ | 0.674 | **0.765** | 0.674 | **0.766** | 0.645 |
| RR-MAE (ms) ↓ | 20.8 | **10.8** | 23.8 | **12.3** | 23.8 |
| HR error (bpm) ↓ | 10.14 | 6.94 | 8.23 | 7.18 | 9.69 |
| morph_corr@GT ↑ | 0.220 | 0.337 | 0.228 | 0.337 | 0.179 (PZ3) |
| FD ↓ | **4.13** | 24.80 | **4.99** | 25.73 | 4.30 |

## Verdict (frozen rule: F1 gain ≥ +0.05 AND morph gain ≥ +0.15, CIs excluding 0)
| generator | F1 gain | morph_corr@GT gain | verdict |
|---|---|---|---|
| iMF | **+0.091 [+0.088, +0.093]** ✔ | +0.118 [+0.113, +0.122] ✘ (bar +0.15) | **PARTIAL** |
| consistency distillation | **+0.092 [+0.089, +0.095]** ✔ | +0.109 [+0.103, +0.115] ✘ | **PARTIAL** |

## What it shows
1. **Beat placement improves substantially and consistently.** F1 rises by 0.09 for both generators, to 0.765 — the
   highest of any arm in the programme (PENGUIN-50 0.645, PENGUIN-1 0.756) — and **RR-interval error is halved**
   (20.8 → 10.8 ms). Against PENGUIN-50: F1 +0.120 [+0.117, +0.124], HR −2.22 bpm.
2. **Morphology improves by half (0.22 → 0.34) but misses the preregistered +0.15 bar**, and remains far below the
   template reference (0.68–0.74 with GT timing). Better placement alone does not close the shape gap.
3. **The decoded waveform leaves the ECG distribution: FD 4.1 → 24.8.** Aligned averaging plus a median baseline
   produces a cleaner-than-real signal (the sample-level texture is averaged away). This is a real cost and any use
   of the decoded waveform as a *waveform* (rather than as an event sequence) must report it.
4. **For HR alone the simple median-of-HR consensus is still better** (by 0.80 bpm for iMF, 1.08 for CD). Decoding is
   for events and intervals, not for the rate.
5. **The per-beat reliability signal is weak here.** Spearman(vote-position SD, |timing error|) = +0.07 / +0.06; keeping
   only events with ≥ 75 % of the votes raises precision 0.84 → 0.89 (iMF) at recall 0.75 → 0.71. Useful as a filter,
   not as a calibrated uncertainty.
6. Parameter sensitivity is low: the whole validation grid spans 0.015 F1.
