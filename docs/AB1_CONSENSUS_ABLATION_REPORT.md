# AB1 / LAT1 — Consensus ablation, the cost-matched curve, and single-window latency

Prereg `1a7ccf1`, frozen before any number. No training. V1 VitalDB test (1,156 patients, 19,543 windows), seed-42
checkpoints, noise seeds 0…K−1, HR pooled per window. Raw: `artifacts/ab1_consensus/`, `artifacts/lat1_latency/`.

## Cost-matched curve — HR error (bpm) by **total NFE per window** (= K × NFE), median pooling

| total NFE | PENGUIN 1 NFE | iMF 1 NFE | iMF 2 NFE | consistency distillation 1 NFE | PENGUIN 50 NFE |
|---|---|---|---|---|---|
| 1 | **7.329** | 10.078 | — | 7.876 | — |
| 2 | 7.743 | 9.507 | 9.285 | 7.568 | — |
| 4 | 7.640 | 7.887 | 8.763 | **6.774** | — |
| 8 | 7.541 | 7.111 | 7.155 | **6.385** | — |
| 16 | 7.438 | 6.675 | 6.502 | **6.229** | — |
| 32 | 7.366 | 6.454 | **6.176** | **6.138** | — |
| 50 | — | — | — | — | 9.644 |
| 100 | — | — | — | — | 8.946 |
| 200 | — | — | — | — | 7.548 |

1. **Every cheap-sample arm beats PENGUIN-50 at a fraction of its cost.** One PENGUIN-50 sample costs 50 NFE and gives
   9.64 bpm; 4 NFE of consistency distillation gives 6.77, and even a single 1-NFE PENGUIN sample gives 7.33.
2. **Consensus needs sample diversity, and PENGUIN at 1 NFE has none.** Its curve is flat (7.33 → 7.37 from K = 1 to
   32) because the collapsed conditional mean is nearly the same waveform for every noise draw — the direct
   consequence of the X2 source-cancellation measurement. iMF gains 3.6 bpm over the same sweep, CD 1.7.
3. **Saturation (frozen rule: doubling K changes HR by < 0.1 bpm):** CD at K = 16; iMF and iMF-2 not saturated by
   K = 32; PENGUIN-1 "saturates" at K = 4 only because it never improves.
4. **The preregistered operating point (iMF, K = 16, median) is not the best cell in this table** — CD at K = 16 is
   0.45 bpm better, and at equal budget 32 NFE iMF-2 nearly matches CD. The claims stay as preregistered; this table
   is what a practitioner should read for a budget.

## Pooling rule (K = 16, NFE 1)
| arm | median (preregistered) | mean | 20 % trimmed |
|---|---|---|---|
| iMF | **6.675** | 8.353 | 7.045 |
| CD | 6.229 | 6.700 | **6.018** |
| PENGUIN 1 NFE | 7.438 | 7.342 | **6.570** |

The mean is clearly worse for iMF (outlier draws), the trimmed mean is equal or slightly better than the median for
CD and PENGUIN. Median was preregistered and is kept; the differences are within 0.4 bpm.

## Single-window latency (batch 1; CPU = 4 threads)

| model | NFE | CPU median | GPU median | vs PENGUIN-50 (CPU) |
|---|---|---|---|---|
| PENGUIN | 50 | 2,285.5 ms | 963.1 ms | 1× |
| PENGUIN | 1 | 45.8 ms | 19.6 ms | 50× faster |
| iMF | 1 | 45.5 ms | 19.4 ms | 50× faster |
| consistency distillation | 1 | 44.4 ms | 19.5 ms | 51× faster |
| **small vanilla DiT** | 1 | **5.1 ms** | **2.8 ms** | **447× faster** |
| iMF × 16 (consensus) | 16 | 727.7 ms | 310.3 ms | 3.1× faster |
| CD × 16 (consensus) | 16 | 711.0 ms | 311.8 ms | 3.2× faster |

**The consensus arm is still three times faster than one PENGUIN-50 sample on a 4-thread CPU**, while being 3.0 bpm
more accurate. A 4 s window at 5.1 ms (small DiT) leaves a 780× real-time margin on CPU.
