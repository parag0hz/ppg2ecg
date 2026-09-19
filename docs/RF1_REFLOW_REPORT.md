# RF1 — Rectified flow (reflow, 2-RF) against iMF: **RF vs iMF = WORSE (on F1); RF is non-inferior to PENGUIN-50 at 1 NFE**

Prereg `49c2e23`, frozen before any RF1 weight update. Seed 42. Teacher pairs: 64,000 train windows through arm C at
50 NFE (30.6 min). Reflow: same PENGUIN backbone initialised from arm C, 14,000 steps on the teacher's own coupling
(31.6 min, final loss 0.0043). Evaluated with the SR1 protocol on the V1 test set, paired window-for-window with the
seed-42 arrays of arms C, I, S. Raw: `artifacts/rf1_reflow/`.

| arm | NFE | HR error ↓ | R-peak F1 ↑ | RR-MAE (ms) ↓ | FD ↓ | K = 16 consensus HR |
|---|---|---|---|---|---|---|
| PENGUIN (1-RF, arm C) | 50 | 9.686 | 0.645 | 23.8 | 4.30 | — |
| PENGUIN (1-RF) | 1 | 8.472 | 0.756 | 8.8 | 33.50 | 7.44 |
| **2-RF (reflow, arm R)** | **1** | **9.942** | **0.644** | 24.3 | **4.25** | 6.86 |
| 2-RF | 2 | 12.476 | 0.610 | 24.0 | 30.46 | — |
| 2-RF | 4 | 9.439 | 0.641 | 24.2 | 4.50 | — |
| iMF (arm I) | 1 | 10.139 | 0.674 | 20.8 | 4.13 | **6.67** |
| small vanilla DiT (arm S) | 1 | 8.712 | 0.629 | — | 7.84 | 6.55 |

## Verdicts (frozen rules, NFE 1)
- **RF vs iMF: WORSE.** HR −0.198 [−0.303, −0.094] (better, but under the 1.0 bpm margin); **F1 −0.030 [−0.032, −0.028]**,
  beyond the −0.02 margin with the CI below it.
- **RF vs PENGUIN-50: NON-INFERIOR.** HR +0.255 [+0.168, +0.343] < +1.0; F1 −0.0009 [−0.0019, +0.0001] > −0.02.
  Per-patient HR win rate against PENGUIN-50: 0.446.

## What it means
1. **Reflow does what it promises: one step now reproduces the teacher.** At 1 NFE the 2-RF model matches PENGUIN-50 on
   F1 (0.644 vs 0.645) and on waveform distribution (FD 4.25 vs 4.30), where the un-reflowed model collapses (FD 33.5).
   It is a faithful one-step copy of PENGUIN-50 — including PENGUIN-50's beat-placement level.
2. **So 1-NFE non-inferiority to PENGUIN is not unique to iMF.** Our H1 (REPLICATED in SR1) holds for reflow as well.
   The paper cannot present "one step without losing to PENGUIN" as iMF's contribution alone.
3. **Where iMF still differs:** R-peak F1 +0.030 over reflow at 1 NFE, better consensus HR (6.67 vs 6.86), and **no
   teacher**: iMF trains once from scratch, reflow needs a trained PENGUIN plus 30.6 min of teacher sampling. In
   end-to-end compute the two are close (arm C 52 min + pairs 31 min + reflow 32 min ≈ 115 min vs iMF 114 min).
4. **Consensus is method-agnostic.** Reflow also gains from pooling 16 cheap samples (9.94 → 6.86). The inference-time
   consensus is a property of cheap sampling, not of iMF — which makes it a more general contribution.
5. **Two NFE is broken for every Heun-sampled arm** (2-RF: HR 12.48, FD 30.5; PENGUIN: HR 30.0): a single Heun step
   overshoots. Not specific to reflow.
6. Timing note: the per-window times in `metrics.csv` for this run were measured while other evaluations shared the
   GPU; the single-process timings from V1 (≈0.57 ms at NFE 1 for both backbones) remain the ones to quote.
7. Single seed for arm R; SR1 showed the other arms' conclusions stable across seeds, but R has not been replicated.
