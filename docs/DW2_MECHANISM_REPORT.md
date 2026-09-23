# DW2 part B — Why width works: sample diversity by depth, and the consensus gain it buys

Prereg `90fbc9b` (part B), frozen before any number. Seed-42 checkpoints, V1 VitalDB test. HR half from the stored DW1
per-sample matrices (K = 32/S, all 19,543 windows); waveform half on a fixed 2,000-window `linspace` subset regenerated
with K = min(8, 32/S). Raw: `artifacts/dw2_mechanism/`.

## Diversity by depth S (K samples of the same window)
| model | measure | S=1 | S=2 | S=4 | S=8 | S=16 |
|---|---|---|---|---|---|---|
| **PENGUIN (Euler)** | waveform pairwise RMS | **0.074** | 0.219 | 0.334 | 0.404 | 0.441 |
| | feature-space pairwise | **0.35** | 0.89 | 1.27 | 1.35 | 1.39 |
| | HR across-sample SD (bpm) | 5.92 | 5.96 | 6.15 | 5.58 | 4.09 |
| | consensus gain (bpm) | +1.32 | +1.72 | **+1.96** | +1.74 | +0.65 |
| **iMF** | waveform pairwise RMS | 0.478 | 0.506 | 0.491 | 0.488 | 0.484 |
| | feature-space pairwise | 1.29 | 1.53 | 1.55 | 1.45 | 1.40 |
| | HR across-sample SD | 10.37 | 9.27 | 7.37 | 5.40 | 3.15 |
| | consensus gain | **+3.67** | +3.24 | +2.51 | +1.81 | +0.51 |
| **consistency distillation** | waveform pairwise RMS | 0.423 | 0.463 | 0.475 | 0.471 | 0.465 |
| | feature-space pairwise | 1.28 | 1.34 | 1.37 | 1.39 | 1.40 |
| | HR across-sample SD | 7.17 | 8.14 | 6.83 | 5.33 | 3.23 |
| | consensus gain | +2.13 | **+2.70** | +2.19 | +1.58 | +0.55 |
Consensus gain = mean single-sample HR error (over the K draws) − K-sample median error, at B = 32 (K = 32/S).

## Reading it
1. **PENGUIN at one step produces almost identical waveforms** — pairwise RMS 0.074, 6.5× below iMF's 0.478, and 0.35 vs
   1.29 in feature space — and its diversity grows with depth (0.07 → 0.22 → 0.33 → 0.40 → 0.44). This is the "minimum
   depth to obtain diversity" the hypothesis asked for, and it is model-specific: **iMF and CD have full diversity at
   S = 1** (≈ 0.42–0.51 at every S); PENGUIN needs S ≈ 4–8 to reach the same level.
2. **The preregistered HR-SD rule gave the wrong verdict, and the report says why.** HR SD for PENGUIN at S = 1 is 5.9
   (ratio 0.96 to S = 4), so by that rule "no useful diversity" was not confirmed. The waveform measures show the
   samples *are* near-identical; the HR spread comes from the peak detector being unstable on the collapsed,
   QRS-less waveform (FD 33), not from generative diversity. HR SD is not a valid diversity measure for a degenerate
   generator; the waveform-level measures are the ones to use.
3. **Consensus gain tracks diversity, per model.** PENGUIN: gain rises with S while diversity rises (1.32 → 1.96 at S = 4)
   and then falls as K shrinks. iMF and CD: diversity is already maximal at S = 1–2 and gain falls monotonically with
   S because K falls. Across all three models the gain at S = 16 (K = 2) is 0.5–0.65 bpm.
4. **This explains the DW1 optima.** The best allocation is where diversity is (just) saturated: S = 1 for CD, S = 2 for
   iMF (two steps repair its under-dispersion, N6), S = 2–4 for PENGUIN. The slogan holds in the form *"spend enough
   steps for the samples to differ meaningfully, then spend the rest on samples"* — with "meaningfully" measured on
   the waveform, not on single-sample accuracy.
5. **Erratum to DW1 / AB1.** Those tables' K = 1 cells are the noise-seed-0 draw. Averaged over 32 draws, the single-sample
   HR error at S = 1 is PENGUIN **8.68** (seed 0: 7.33 — a favourable draw), iMF 10.13 (10.08), CD 8.27 (7.88). The DW1
   statement "PENGUIN at S = 1 is the best single-NFE HR" rests on that one draw; on the draw average PENGUIN (8.68)
   and CD (8.27) are comparable. The grid cells with K > 1 are unaffected.

## Correction note (2026-09-23, from EXP-B `6a2c8a8` and B3-BOOT, prereg `590d191`)
The numbers above are unchanged; this note corrects their interpretation.
- **The preregistered HR-SD criterion of part B failed** (PENGUIN S = 1 / S = 4 SD ratio 0.96), and that verdict stands.
- **The waveform-diversity interpretation given in "Reading it" (points 1, 3, 4) was exploratory, and it does not hold as a
  mechanism.** EXP-B / B3-BOOT show that across the 12 (model, depth) conditions at K = 16, waveform pairwise RMS is a
  weaker correlate of the consensus gain than the functional-error quantities (difference in |Spearman| +0.105,
  95 % patient-bootstrap interval [+0.070, +0.175]), and that **within a model its relation to the gain changes sign**
  (negative within iMF, positive within CD and PENGUIN, each in 100 % of 5,000 patient-bootstrap replicates). Functional-error
  dependence (mean cross-sample error correlation ρ̄, equivalently within-condition functional dispersion) keeps the same
  direction in every model and orders the gain over all four tested depths in every replicate.
- The statement "spend enough steps for the samples to differ meaningfully, measured on the waveform" is replaced by
  "spend enough steps for the samples' **functional errors** to be non-redundant". PENGUIN's near-identical one-step
  waveforms (RMS 0.074, reproduced) still carry a +1.24 bpm gain because their HR errors are correlated at 0.858, not at 1.
- Current statement and evidence: `docs/B3_FUNCTIONAL_ERROR_MECHANISM_REPORT.md`.
