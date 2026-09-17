# VM1 — Official ("vanilla") improved MeanFlow on PPG → ECG: **MIXED vs the PENGUIN-backbone iMF**; ImageNet pretraining does not help

Prereg `09bd15d` (+ monitor fix `2e16b16`, before any VM1 run). Port of Lyy-iiis/imeanflow (JAX `bf60cd7`; the official
`torch` branch `0468798` for parameter names) to 1-D PPG → ECG: DiT trunk, auxiliary v head, in-model CFG, official
objective and optimiser (AdamW, wd 0, β₂ 0.95, lr 1e-4, warmup-const), no EMA, 14,000 steps, seed 42, VitalDB V1 split.
CFG setting chosen on the 289 validation patients only. Test = the same 1,156 patients / 19,543 windows as V1 and BB1,
paired against arm I. Raw: `artifacts/vm1_vanilla_imf/`.

| arm | init | params (inference) | chosen CFG | NFE 1 HR | ΔHR vs I | NFE 1 F1 | ΔF1 vs I | FD | ms/window | K=16 consensus HR | train time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **I** (V1 iMF) | — | 4.30 M | — | 10.139 | — | 0.674 | — | 4.13 | 1.17 | 6.675 | 2.2 h |
| **S** small DiT | scratch | 3.04 M | ω 1.5, [0.4, 0.65] | 8.712 | **−1.427** | 0.629 | **−0.045** | 7.84 | **0.074** | 6.548 | 21 min |
| **P** B/2 | **ImageNet** | 88.25 M | ω 1 (none) | 8.960 | **−1.177** | 0.636 | **−0.039** | 8.93 | 1.60 | **6.156** | 2.7 h |
| **B** B/2 | scratch | 88.25 M | ω 1 (none) | 8.630 | **−1.500** | 0.622 | **−0.052** | 4.37 | 1.59 | 6.162 | 2.7 h |

**Verdicts (frozen rule, NFE 1).** S, P, B vs I: all **MIXED** — each clears the HR improvement bar (≤ −1.0 bpm, CI
below 0) and each fails the F1 bar (≤ −0.02, CI below 0). **P vs B (pretraining at equal architecture): NO MEANINGFUL
CHANGE** (HR +0.323 [+0.209, +0.432] i.e. pretrained slightly worse, F1 +0.014 [+0.011, +0.017] slightly better; both
inside the margins).

## What it means
1. **The official design trades event F1 for HR.** Every vanilla arm places beats at a rate closer to the truth
   (HR −1.2 to −1.5 bpm) while matching fewer individual beats (F1 −0.04 to −0.05) — the same "rate, not position"
   pattern N1 found.
2. **ImageNet pretraining bought nothing here.** 88.1 M transferred parameters left both endpoints inside the margins,
   and the pretrained arm was the slower of the two to reach its own best HR. Cross-modal transfer from image
   generation does not carry into this task at this budget.
3. **In-model CFG was not selected.** Both B/2 arms chose ω = 1 (no guidance) on validation; only the small arm took
   ω = 1.5 with the official interval, and its gain over ω = 1 was 0.3 bpm HR at a 0.01 F1 cost.
4. **The small DiT is the fastest model in the programme**: 0.074 ms per window at NFE 1 — 16× faster than the
   PENGUIN-backbone iMF and **390× faster than PENGUIN at 50 NFE** (28.7 ms), at 8.71 vs 9.69 bpm HR error.
5. **Consensus still dominates**: pooling 16 cheap samples gives 6.16 bpm (P, B) against 9.69 for one PENGUIN-50 sample
   and 7.55 for four of them (200 NFE).
6. Caveats: single seed; the VM1 arms also change optimiser and schedule (the official recipe), so "architecture" and
   "recipe" are not separated here — only P vs B is a controlled single-factor comparison.
