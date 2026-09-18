# SR1 — Three seeds: **all three claims REPLICATED (3/3)**

Prereg `28c479a`, frozen before any SR1 weight update. Seeds 42 (existing), 1, 2 × arms C (PENGUIN), I (iMF on the
PENGUIN backbone), S (small vanilla DiT); everything else identical to V1 / VM1. Nine evaluations on the V1 test set
(1,156 patients, 19,543 windows), patient-clustered bootstrap, pairing within a seed. Raw: `artifacts/sr1_seeds/`.

| claim (frozen margins: HR 1.0 bpm, F1 0.02) | seed 42 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| **H1** iMF 1 NFE non-inferior to PENGUIN 50 NFE | pass | pass | pass | **REPLICATED (3/3)** |
| **H2** iMF 1 NFE, K = 16 HR consensus beats one PENGUIN-50 sample | pass | pass | pass | **REPLICATED (3/3)** |
| **H3** small DiT 1 NFE beats PENGUIN-50 on HR, non-inferior on F1 | pass | pass | pass | **REPLICATED (3/3)** |

## Seed spread (patient-macro means)

| arm · metric | seed 42 | seed 1 | seed 2 | mean | range |
|---|---|---|---|---|---|
| PENGUIN 50 NFE · HR | 9.686 | 11.037 | 11.187 | 10.637 | 1.50 |
| PENGUIN 1 NFE · HR | 8.472 | 8.229 | 9.109 | 8.603 | 0.88 |
| iMF 1 NFE · HR | 10.139 | 9.137 | 10.018 | 9.765 | 1.00 |
| small DiT 1 NFE · HR | 8.712 | 9.290 | 8.987 | 8.996 | 0.58 |
| iMF K = 16 consensus · HR | 6.675 | 6.283 | 6.362 | 6.440 | 0.39 |
| small DiT K = 16 consensus · HR | 6.548 | 6.765 | 6.248 | 6.520 | 0.52 |
| PENGUIN 50 NFE · R-peak F1 | 0.645 | — | — | — | — |
| PENGUIN 1 NFE · F1 | 0.756 | 0.760 | 0.745 | 0.753 | 0.015 |
| iMF 1 NFE · F1 | 0.674 | 0.666 | 0.658 | 0.666 | 0.016 |
| small DiT 1 NFE · F1 | 0.629 | 0.634 | 0.657 | 0.640 | 0.028 |

## What the spread changes
1. **The claims hold, and seed 42 was not a lucky draw for us** — it was the *best* seed for PENGUIN-50 (HR 9.686 vs
   11.04 and 11.19). Against the three-seed mean of 10.64 the margins are wider than V1 reported.
2. **Consensus is the most stable quantity in the programme**: range 0.39 bpm across seeds, against 1.50 bpm for
   PENGUIN-50 single samples. Pooling samples removes seed noise as well as sample noise.
3. **The CFG setting the small DiT selects is seed-dependent** (seed 42 chose ω = 1.5 with the official interval;
   seeds 1 and 2 chose ω = 1, i.e. no guidance), yet H3 passes in all three. In-model CFG is not carrying the result.
4. **PENGUIN at 1 NFE remains the strongest single-sample arm on HR and F1 on this corpus** (8.60 bpm, F1 0.753,
   three-seed means) — the collapsed conditional-mean waveform keeps the right beat rate under anaesthesia. Its
   waveform-distribution cost is in V1 (FD 33.5 vs 4.1) and in PZ3's morphology axis.
