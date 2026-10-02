# GX1 — cross-target generalization of partial DualReadout sharing (PPG → ABP) — REPORT

> **PRINCIPLE-LEVEL VERDICT: NOT SUPPORTED.**
> - The exact S1 MIDDLE topology (shared stem + blocks 1–6; private blocks 7–8 + decoders) was transferred to
>   PPG → ABP without any sharing-depth search.
> - It **preserved the point specialist (G1, G2) and the generative specialist (G3) in all three seeds**, and saved
>   20.85 % of waveform parameters (G5).
> - **The condition-use gate G4 failed for seeds 42 and 44.**
>   - With PPG shuffled, S1's generative D_ABP rose, but the 95 % interval reached below 0:
>     - seed 42: +5,093 [−92, +10,961] mmHg²;
>     - seed 44: +3,625 [−919, +9,286].
>   - The sample-level correlation collapse was large in every seed (−0.72 to −0.73).
> - **Seed 43 passed G1–G5.** Classification: **SEED-SENSITIVE (1 / 3)**. Seed 42, the primary, failed.
> - **G6 (fair compute) was supported.** Against separate-cached, S1 uses fewer FLOPs (1.49 vs 2.19 G) and is faster on
>   GPU (11.76 vs 13.33 ms) and on CPU (26.36 vs 32.03 ms).
> - **Evidence label:** CROSS-TASK GENERALIZATION. MIMIC-BP is a historically used ABP dataset; this is not fresh
>   external confirmation.

| commit | content |
|---|---|
| `4abd8cb` | data and evaluator audits, split, TRAIN-only margins, 9 trained models, GX-DEV (descriptive), compute, preregistration |
| `ef374e6` | final freeze (all checkpoints, code, manifests hashed) before GX-LOCK |
| this commit | single GX-LOCK evaluation, diagnostics, figure, report |

## 1. Data, split and target interface

- **Data:** MIMIC-BP was the only usable PPG + ABP cohort (`docs/GX1_ABP_DATA_AUDIT.md`).
- **Split:** the official pre-existing patient-disjoint split, frozen at A7:

  | role | subjects | windows | exclusions |
  |---|---|---|---|
  | GX-TRAIN | 1,100 | 230,812 | — |
  | GX-DEV | 195 | 40,592 | — |
  | **GX-LOCK** | **229** | **48,089 / 48,090** | 1 (R4) |

- **Target interface (only):**
  - ABP is FFT-resampled, then normalized by the frozen A8 global TRAIN-only z (77.571767 / 22.275611 mmHg);
  - one output channel;
  - mmHg metrics use the inverse transform.
- **Timing:** the RD1 / C0 detector protocol was retrained on GX-TRAIN. On GX-LOCK it reaches F1 0.813
  [0.775, 0.846], recall 0.806, precision 0.821 and FP 0.88 / window against the ECG R. Unlike DP3, the timing
  condition transfers here.

## 2. Evaluator and margins (frozen before training; `docs/GX1_ABP_EVALUATOR_AUDIT.md`)

- **D_ABP:** `kanflow_fd` on ABP in mmHg, the pre-GX1 FD applied to MIMIC-BP ABP in U2.
- **Margins (TRAIN only):**

  | gate | margin | derivation |
  |---|---|---|
  | G1 corr | −0.02 | DP0 |
  | G2 MAE_z | +0.0200 | 0.02 × GX-TRAIN target SD |
  | G3 D | +2,785.4 mmHg² | DP0 +1.0 × TRAIN total-variance ratio 254,029 / 91.2 |

## 3. Training (9 jobs, frozen DP0 trainers)

- **Models:** P_ABP (593,577, L1), G_ABP (598,333, FM, Euler 8) and S1_ABP (943,372, round-robin
  20,000 + 20,000), for seeds 42, 43 and 44.
- **Training times:** P 130.5–130.9 s, G 133.2–133.9 s, S1 303.3–306.0 s. NaN 0.

## 4. GX-LOCK results (229 patients, 48,089 windows; 95 % patient-bootstrap CI, 2,000 replicates)

**Point readout** (per-window Pearson r, normalized MAE, mmHg errors):

| seed | P corr | S1 corr | Δcorr (G1) | ΔMAE_z (G2) | P / S1 MAE (mmHg) | P / S1 SBP MAE | P / S1 DBP MAE | P / S1 MAP MAE |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.854 | 0.858 | +0.0046 [+0.0015, +0.0078] | +0.0080 [−0.0008, +0.0162] | 11.72 / 11.89 | 13.79 / 13.99 | 9.02 / 9.01 | 9.27 / 9.61 |
| 43 | 0.851 | 0.856 | +0.0055 [+0.0019, +0.0085] | +0.0053 [−0.0036, +0.0154] | 11.79 / 11.91 | 13.95 / 14.09 | 8.93 / 8.87 | 9.36 / 9.61 |
| 44 | 0.846 | 0.857 | +0.0110 [+0.0077, +0.0144] | −0.0325 [−0.0464, −0.0189] | 12.67 / 11.95 | 15.39 / 13.81 | 9.23 / 9.01 | 10.31 / 9.65 |

**Generative readout** (one sample per window, Euler 8):

| seed | G D_ABP | S1 D_ABP | ΔD (G3) | G / S1 corr (descr.) | S1 shuffle ΔD (G4) | S1 shuffle Δcorr (G4) | G shuffle ΔD (control) |
|---|---|---|---|---|---|---|---|
| 42 | 11,463 | 9,574 | −1,889 [−5,343, +1,487] | 0.836 / 0.837 | **+5,093 [−92, +10,961]** | −0.731 [−0.745, −0.716] | +15,207 [+10,208, +21,041] |
| 43 | 13,239 | 6,603 | −6,636 [−8,282, −4,954] | 0.843 / 0.843 | +22,613 [+15,530, +30,997] | −0.727 [−0.742, −0.711] | +6,583 [+2,615, +11,508] |
| 44 | 14,130 | 5,921 | −8,209 [−12,233, −4,207] | 0.850 / 0.844 | **+3,625 [−919, +9,286]** | −0.718 [−0.732, −0.702] | +8,713 [+5,800, +12,651] |

**Gates:**

| seed | G1 | G2 | G3 | G4 | G5 | seed pass |
|---|---|---|---|---|---|---|
| 42 | PASS | PASS | PASS | **FAIL** | PASS | NO |
| 43 | PASS | PASS | PASS | PASS | PASS | YES |
| 44 | PASS | PASS | PASS | **FAIL** | PASS | NO |

**Multi-seed: SEED-SENSITIVE (1 / 3).**

## 5. Reading

- **Capability preservation transferred.** In every seed the transferred S1 readouts were non-inferior to their ABP
  specialists:
  - point correlation was in fact slightly higher (+0.005 to +0.011);
  - normalized MAE was within +0.02;
  - D_ABP was lower than G's in all seeds (only seed 42's interval crosses 0).
- **The point readout's SBP / DBP / MAP errors were within about 0.35 mmHg of P for seeds 42 / 43**, and lower for
  seed 44 (descriptive).
- **The failure is in condition use at the population level.** For seeds 42 and 44, shuffling PPG left S1's
  population-level D_ABP increase uncertain, while the window-level correlation collapsed from about 0.84 to about
  0.11. So the S1 samples clearly follow their own PPG.
  - **Interpretation only, not tested:** S1's flow readout keeps a plausible marginal ABP distribution even under
    mismatched PPG, so an FD on the marginal distribution is a weak detector of conditioning.
  - The same FD-criterion failure pattern appeared for DP3 seed 42.
  - The frozen G4 rule is kept as it is, and the verdict is not rescued.
- **The ABP specialist G showed a significant FD increase under shuffle in all seeds.**

## 6. Parameter and compute efficiency

| system | waveform / pipeline params | FLOPs both (incl. detector) | GPU both, batch-1 | CPU 4-thread both | peak mem above weights |
|---|---|---|---|---|---|
| separate-naive | 1,191,910 / 1,849,704 | 2.52 G | 14.39 [14.27, 14.81] ms | 34.56 [33.98, 35.62] ms | 0.93 MiB |
| separate-cached | 1,191,910 / 1,520,807 | 2.19 G | 13.33 [13.21, 13.56] ms | 32.03 [31.46, 34.12] ms | 0.93 MiB |
| **S1_ABP** | 943,372 / 1,272,269 | 1.49 G | 11.76 [11.66, 11.95] ms | 26.36 [26.24, 26.84] ms | 1.05 MiB |

- **Parameter saving:** 20.85 % (G5).
- **G6 supported:** FLOPs lower, GPU lower and CPU lower than separate-cached. The cached generation is bit-identical
  to naive.

## 7. Optional sharing-depth diagnostics (seed 42, GX-LOCK; after the primary verdict; explanatory only)

| control | params (saving) | Δcorr | ΔMAE_z | ΔD (mmHg²) |
|---|---|---|---|---|
| FULL-SHARE-ABP (S0) | 861,196 (27.75 %) | +0.0072 [+0.0044, +0.0100] | +0.0028 [−0.0042, +0.0097] | +1,531 [−2,741, **+5,931**] |
| transferred S1 (primary) | 943,372 (20.85 %) | +0.0046 [+0.0015, +0.0078] | +0.0080 [−0.0008, +0.0162] | −1,889 [−5,343, +1,487] |
| STEM-ONLY-ABP (S2) | 1,189,900 (0.17 %) | +0.0006 [−0.0027, +0.0039] | +0.0070 [−0.0004, +0.0148] | +690 [−114, +1,499] |

- **Full sharing:** its generative interval exceeds the G3 margin, so full sharing again strained one readout. On ECG
  it was the point readout's events; here it is the generative distribution.
- **Stem-only:** preserves both readouts but saves almost nothing, as on ECG.
- These controls change nothing about S1 or the verdict. One seed and one population: descriptive only.

## 8. GX-DEV characterization (descriptive, computed before the freeze; no decision used it)

| seed | Δcorr | ΔMAE_z | ΔD (mmHg²) |
|---|---|---|---|
| 42 | +0.0041 | +0.0068 | −786 [−4,285, +2,800] |
| 43 | +0.0039 | +0.0068 | −7,586 |
| 44 | +0.0111 | −0.0407 | −6,398 |

## 9. Supported claims

- **On MIMIC-BP (cross-task, historically used for ABP):** the exact ECG-derived S1 topology, without sharing-depth
  search, kept point-specialist correlation and MAE and generative-specialist D_ABP within the frozen margins in all
  three training seeds.
- **Parameters:** 20.85 % fewer waveform parameters.
- **Compute:** lower combined two-output inference compute than a fair separate-cached pipeline (FLOPs, GPU and CPU).
- **Condition use:** the S1 generative samples follow their PPG at the window level (correlation drop of about 0.72 in
  every seed).

## 10. Unsupported claims

- the cross-target principle as defined (seed 42 fails G4; only 1 / 3 seeds pass) — **NOT SUPPORTED**;
- fresh external confirmation;
- population-level PPG-condition use of the S1 generative readout for seeds 42 / 44;
- partial sharing being universally optimal, all physiological tasks benefiting, a causal mechanism, or a
  general-purpose foundation architecture;
- better ABP estimation than the specialists as a claim (the point improvements are secondary observations);
- clinical BP-measurement validity.

## 11. No rescue

After GX-LOCK opened (21:57), the following were all left unchanged: architecture, sharing depth, seeds, evaluator,
margins, normalization, NFE, and exclusions. The diagnostics were trained only after the freeze and evaluated only
after the verdict.

## Files

- **Docs:** `docs/GX1_ABP_DATA_AUDIT.md`, `docs/GX1_ABP_EVALUATOR_AUDIT.md`,
  `docs/GX1_ABP_GENERALIZATION_PREREGISTRATION.md`, this report.
- **Code:** `scripts/gx1_abp.py`, `tests/test_gx1_abp.py`.
- **Artifacts:** `artifacts/gx1_abp/`
  - split, evaluator and margins;
  - training manifest and checkpoint hashes; the final freeze;
  - seed 42 / 43 / 44 metrics, bootstrap and gates;
  - condition shuffle; detector; lock cohort;
  - compute (naive / cached / s1, G6) and efficiency table;
  - multi-seed summary; diagnostics; GX-DEV characterization;
  - `table_main.csv`, `figure_generalization.png`.
- **Checkpoints (not committed):** `outputs/gx1_abp/`.
