# SR1 — Do the VitalDB claims survive three seeds? (preregistration)

Frozen and pushed before any SR1 weight update. Same corpus, split, budget (14,000 steps) and test protocol as V1 /
BB1 / VM1. Only the seed changes: **42 (done), 1, 2**.

## Arms (3 × 3 runs; seed 42 is the existing run, not retrained)
| arm | model | trainer | seed-42 run |
|---|---|---|---|
| **C** | PENGUIN (OT-CFM), h 128 | `train_a0` (V1 argv) | `outputs/v1_vitaldb_armC_seed42` |
| **I** | iMF on the PENGUIN backbone | `train_a2` (V1 argv) | `outputs/v1_vitaldb_armI_seed42` |
| **S** | vanilla iMF, small DiT | `train_vimf --arch S` (VM1 argv) | `outputs/vm1_S_seed42` |

## Evaluation (per run)
V1 test: 1,156 patients, 19,543 windows, noise seeds 0–3 (per-window metric averaged), plus HR consensus K = 16 at
NFE 1 (noise seeds 0–15). NFE: C ∈ {1, 2, 4, 50}, I and S ∈ {1, 2, 4}. Arm S re-selects its CFG setting **per seed** on
the 289 validation patients with the frozen VM1 rule. Pairing is within a seed (same windows, same noise seeds);
patient-clustered bootstrap, 2,000 replicates, seed 20260911.

## Claims tested per seed (each with the frozen margins: HR 1.0 bpm, R-peak F1 0.02)
- **H1 — iMF non-inferior to PENGUIN-50 at NFE 1**: upper CI of (I₁ − C₅₀) < +1.0 bpm for HR **and** lower CI of the F1
  difference > −0.02.
- **H2 — HR consensus beats one PENGUIN-50 sample**: upper CI of (I₁ K=16 − C₅₀ single) < 0.
- **H3 — the small DiT beats PENGUIN-50 on HR at NFE 1**: upper CI of (S₁ − C₅₀) < 0, with F1 lower CI > −0.02
  (superior on HR, non-inferior on F1).

## Replication verdict, per claim
**REPLICATED** if it holds in 3/3 seeds, **PARTIAL** in 2/3, **NOT REPLICATED** in ≤ 1/3. Also reported: per-metric mean
and range across seeds, and the same table for arm S's consensus. Nothing is tuned after results; no claim is added
or dropped after seeing a seed.
