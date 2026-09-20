# ED2 — Consensus decoding on WildPPG and on HRV: the event gain transfers; HRV improves relatively but stays unusable absolutely

Prereg `6d68350`, frozen before any number. No training. Decoding algorithm and parameters are ED1's, **not re-tuned**.
Raw: `artifacts/ed2_decoding_transfer/part_A.json`, `part_B.json`.

## Part A — WildPPG (14 subjects, WP1 folds): **PARTIAL for both generators, as on VitalDB**

| metric | iMF single | iMF decoded | CD single | CD decoded |
|---|---|---|---|---|
| R-peak F1 ↑ | 0.371 | **0.439** | 0.382 | **0.447** |
| RR-MAE (ms) ↓ | 27.0 | **18.6** | 27.1 | **19.4** |
| HR error ↓ | 12.61 | 11.23 | 12.32 | 12.92 |
| morph_corr@GT ↑ | 0.069 | 0.110 | 0.080 | 0.114 |
| FD per fold | — | 40.9 / 20.9 / 39.5 / 21.7 | — | 31.4 / 20.9 / 22.9 / 39.2 |

- **F1 gain: iMF +0.068 [+0.058, +0.077], CD +0.065 [+0.058, +0.073]** — passes the +0.05 bar on a second corpus with
  parameters chosen on a different one. Against PENGUIN-50: F1 +0.086 (iMF), +0.094 (CD).
- RR-interval error falls by 7 ms (26–31 %) for both.
- The morphology bar (+0.15) is missed by a wide margin here (+0.04, +0.03): on WildPPG, beat shape at the true
  positions is near zero for every arm, so there is little to recover.
- HR: the decoded waveform helps iMF (−1.34 bpm) but not CD (+0.60, CI spans 0). For the rate, the HR-median
  consensus of WP1 (9.27) remains clearly better than decoding (11.23).
- FD is degraded, as on VitalDB.

## Part B — Short-term HRV on VitalDB (1,224 contiguous 2-minute blocks, 1,156 patients, 0 cases skipped)

| arm | SDNN error (ms) ↓ | RMSSD error ↓ | mean-RR error ↓ | SDNN Spearman ↑ |
|---|---|---|---|---|
| PENGUIN-50 | 81.2 [77.5, 84.8] | 106.6 | 59.5 | 0.480 |
| iMF-1 single | 87.3 | 114.6 | 60.3 | 0.453 |
| CD-1 single | 60.5 | 84.7 | 41.3 | 0.434 |
| PRV — PPG peaks, no model | 62.7 [58.7, 66.8] | 81.1 | 49.1 | 0.339 |
| **iMF consensus-decoded** | **55.3 [51.5, 59.1]** | 74.8 | **38.6** | 0.443 |
| **CD consensus-decoded** | **54.0 [50.3, 57.6]** | **72.4** | 41.5 | 0.432 |

**All preregistered HRV claims hold:** decoded vs own single sample — iMF **−31.6 ms [−36.9, −26.2]**, CD −6.5
[−9.9, −3.1]; decoded iMF vs PENGUIN-50 **−25.5 [−30.6, −20.4]**; and decoded iMF also beats the no-model PRV
baseline, −6.6 [−9.5, −3.9].

**But the absolute level is not usable, and the paper must say so.** The reference median SDNN is 54 ms and the best
arm's SDNN error is 54 ms: the error is as large as the quantity. Rank agreement is weak for every arm (Spearman
0.34–0.48). Intervals come from within 4 s windows, which limits every method here, including the reference
computation. HRV therefore supports only a *relative* statement — decoding improves interval statistics over single
samples, PENGUIN-50 and PRV — and is not evidence of clinical utility.
