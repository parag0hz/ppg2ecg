# BB1 — Backbone sensitivity on VitalDB: **NOT A BOTTLENECK AT THIS SCALE** (single seed)

Prereg `a2900b6`; execution-order fix `09bd15d` (arm A OOM'd at start while sharing the GPU with B, no weight update
happened; it was retrained alone with the unchanged argv). Same data, split, budget (14,000 steps), seed 42 and test
protocol as V1 arm I. Test = 1,156 VitalDB patients / 19,543 windows; patient-clustered bootstrap (2,000), paired
against arm I on the same windows and noise seeds. Raw: `artifacts/bb1_backbone/`.

| arm | backbone | params | NFE 1 HR (bpm) | ΔHR vs I [95% CI] | NFE 1 F1 | ΔF1 vs I | FD | ms/window | HR consensus K=16 |
|---|---|---|---|---|---|---|---|---|---|
| **I** (V1 reference) | PENGUIN S5, h 128 | 4.30 M | 10.139 | — | 0.674 | — | 4.13 | 1.17 | 6.675 |
| **A** (4× wider) | PENGUIN S5, h 256 | 17.10 M | 10.463 | **+0.323 [+0.236, +0.415]** | 0.646 | −0.028 | 6.09 | 2.19 | 6.628 |
| **B** (other family) | S5 → 4-head self-attention | 3.78 M | 9.369 | **−0.770 [−0.878, −0.662]** | 0.662 | −0.012 | 8.16 | **0.55** | **6.341** |

Verdicts (frozen rule, NFE 1): **A = WORSE** (F1 −0.028 with CI below −0.02), **B = NO MEANINGFUL CHANGE**
(HR better by 0.77 < the 1.0 bpm margin; F1 −0.012 inside the margin). Stage: **NOT A BOTTLENECK AT THIS SCALE.**

## What it means
1. **Four times the parameters made it worse, not better** (HR, F1 and FD all worse at NFE 1 and 4). At this data and
   step budget the PENGUIN backbone is not capacity-limited; the ceiling is elsewhere.
2. **Swapping the sequence mixer moved HR by less than the non-inferiority margin** and cost waveform distribution
   (FD 8.16 vs 4.13) while halving inference time. Architecture family is not the lever either.
3. Reported, not gated: attention is **2.1× faster per window** than the S5 mixer at the same NFE, and its consensus
   HR (6.34) is the best of the three — consistent with the V1 finding that pooled cheap samples, not the backbone,
   are what moves HR.
4. Single seed; the margins above are wide relative to the CIs but seed variance is not measured.
