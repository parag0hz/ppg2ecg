# DW1 — Fixed inference budget: depth (steps per sample) or width (number of samples)? (preregistration; no training)

Frozen and pushed before any DW1 number. V1 VitalDB test (1,156 patients, 19,543 windows), seed-42 checkpoints.

## Grid
Budget `B = K × S` network evaluations per window, `K` independent samples of `S` steps each.
`S ∈ {1, 2, 4, 8, 16, 32}`, `K = B / S`, for every `B ∈ {1, 2, 4, 8, 16, 32}`; the preregistered budgets for the claims are
**B = 16 and B = 32**. Noise seeds 0 … K−1; the first K samples at a given S are shared across budgets.

| model | trained for | S-step sampler |
|---|---|---|
| **iMF** (arm I) | one step | uniform MeanFlow schedule `[1/S]·S` |
| **Consistency distillation** (arm D) | one step | multistep consistency sampling, re-noising at `t = i/S` (CD1's rule, generalised) |
| **PENGUIN** (arm C) | many steps | forward Euler, S steps (NFE = S) |
Reference line: PENGUIN at its own setting (Heun 25 = 50 NFE), K = 1, 2, 4 from AB1.

## Estimate and statistics
HR = **median over the K samples** of each sample's HR (the preregistered pooling rule; K = 1 is the single sample).
Error against the reference HR, patient-macro, patient-clustered bootstrap (2,000). Cached AB1 matrices are reused for
(iMF, S = 1, 2), (CD, S = 1), (PENGUIN, S = 1). Also reported per (model, S): R-peak F1 and FD of the seed-0 sample, to
show what depth does to a single sample. Consensus *decoding* is not part of this grid.

## Hypotheses (paired within the model, at B = 16 and B = 32)
- **H-W (iMF, CD — trained for one step):** the width extreme (K = B, S = 1) has lower HR error than the depth extreme
  (K = 1, S = B): upper CI of `err(K=B,S=1) − err(K=1,S=B)` < 0.
- **H-D (PENGUIN — trained for many steps):** the reverse, upper CI of `err(K=1,S=B) − err(K=B,S=1)` < 0.
- Reported for every model and budget: the best cell, and whether an interior allocation beats **both** extremes
  (AB1 already suggests it may: iMF (8, 2) 6.50 < (16, 1) 6.68).
