# DP3 — MIMIC-BP ECG-target-blind cross-task external validation — REPORT

> **PRIMARY VERDICT: TARGET-BLIND EXTERNAL FAILED.**
> - Seed 42, the frozen DP0 model, passed T1, T2, T3 and T5 on MIMIC-BP but **failed T4**. With PPG shuffled, the FD
>   increase was +3.65 [−0.87, +8.34], and its interval includes 0. The correlation criterion of T4 was met.
> - Seeds 43 and 44 (predefined replications) passed T1–T4 → **ROBUST-2/3**. Seed 42 stays primary; it is not replaced.
> - **Paper readiness: INTERNAL + CROSS-TASK EVIDENCE ONLY.** The DP0 internal confirmation on AF-LOCK stands
>   unchanged.
> - **Domain shift in the conditioning:** the frozen timing detector essentially failed on MIMIC-BP (event recall 0.045,
>   FP 4.78 / window, F1 0.047, while HR-MAE was 2.5 bpm).
>   - The event metrics of both P and S1 therefore collapsed together.
>   - The T2 deltas sit on a near-zero recall.
>   - T1 rests on the beats where a detector event fell within ±50 ms of an ECG R-peak (patients with any such beat:
>     714 of 1,524).
> - **Evidence label:** ECG-target-blind cross-task external cohort. "MIMIC-BP PPG had previously been used in
>   unrelated PPG-to-ABP experiments, whereas ECG reconstruction targets were withheld from the PPG-to-ECG
>   architecture-development process."

| commit | content |
|---|---|
| `7c2274c` | Stage A target-blind audit (PASS) + Stage B integrity (PASS) |
| `3a2d8bd` | seeds 43 / 44 + internal multi-seed + fair compute |
| `658a931` | MIMIC-BP adapter + preregistration (frozen) |
| `c450c44` | final freeze (all checkpoints, code, rules hashed) |
| this commit | single external evaluation, tables, figures, report |

## 1. Target-blind audit and integrity

- **Stage A — PASS** (`docs/DP3_MIMICBP_TARGET_BLIND_AUDIT.md`).
  - No MIMIC-BP ECG waveform or ECG-derived target was used before: every loader and build path reads PPG / ABP
    only, and all 18 co-mentions were adjudicated as non-use.
  - Prior PPG → ABP use is disclosed: A7, A8, U1, U2, EXP-D.
- **Stage B — PASS:** S1 has 943,372 parameters (246,720 shared, 8,320 adapter). The seed-42 hashes match DP0, the
  detector rule is 0.35 / 32 / σ 20 ms, and the preprocessing code is unchanged since `a15b354`.

## 2. Data and interface

| population | role |
|---|---|
| DP-TRAIN (2,100 patients) | the only training population (seeds 43 / 44) |
| DP-DEV (300) / AF-LOCK (337) | internal characterization only |
| **MIMIC-BP** | 1,524 subjects, 125 Hz PPG and ECG, (30, 3750) per subject; all subjects eligible; 319,493 of 320,040 windows kept (547 excluded by R4, the frozen reference-HR rule; no R1 / R2 / R3 / R5 exclusion) |

**Interface** (`docs/DP3_MIMICBP_INTERFACE.md`):
- 4 s non-overlapping windows from each 30 s segment start (7 per segment);
- V1 / DP0 preprocessing: FFT 500 → 512, PPG 0.5–4 Hz, ECG 0.5 Hz high-pass, per-window z-score + min-max;
- the single ECG channel (lead UNKNOWN) and the single PPG channel;
- no alignment; the frozen DP0 detector; noise identical for G and S1.

## 3. Seeds 43 / 44 and internal robustness (descriptive)

**Training:** six jobs on DP-TRAIN.
- Config sha256 is identical across seeds per role; only the seed changed.
- P 130.8 / 130.7 s, G 133.2 / 132.9 s, S1 303.3 / 303.6 s (20,000 + 20,000 updates). NaN 0.

S1 − specialist, internal (95 % patient-bootstrap CI):

| population | seed | Δcorr | ΔFP | Δrecall | ΔFD |
|---|---|---|---|---|---|
| DP-DEV | 42 | −0.0036 [−0.0053, −0.0017] | +0.0366 [+0.0268, +0.0498] | +0.0015 | −0.66 [−0.90, −0.41] |
| DP-DEV | 43 | −0.0001 [−0.0020, +0.0029] | −0.0100 [−0.0169, −0.0032] | +0.0002 | +0.48 [−0.16, +1.03] |
| DP-DEV | 44 | −0.0026 [−0.0038, −0.0014] | −0.0186 [−0.0265, −0.0104] | −0.0005 | −0.34 [−0.58, −0.12] |
| AF-LOCK | 42 | −0.0048 [−0.0063, −0.0034] | +0.0420 [+0.0356, +0.0484] | +0.0012 | −0.61 [−0.81, −0.42] |
| AF-LOCK | 43 | −0.0008 [−0.0019, +0.0003] | −0.0200 [−0.0266, −0.0134] | −0.0003 | +0.55 [−0.12, +1.12] |
| AF-LOCK | 44 | −0.0023 [−0.0035, −0.0010] | −0.0256 [−0.0312, −0.0199] | −0.0006 | −0.51 [−0.74, −0.33] |

- **Mean ± SD over seeds:**

  | population | Δcorr | ΔFP | ΔFD |
  |---|---|---|---|
  | DP-DEV | −0.0021 ± 0.0018 | +0.0027 ± 0.0297 | −0.17 ± 0.59 |
  | AF-LOCK | −0.0026 ± 0.0020 | −0.0012 ± 0.0375 | −0.19 ± 0.64 |

- **Seed 42 sits near the event margin:** its ΔFP lies close to +0.05, whereas seeds 43 / 44 have negative ΔFP.
- **Seed 43's ΔFD upper bound exceeds +1.0** on both internal populations.
- None of this was used for any decision.

## 4. Fair compute (seed 42; batch-1, median [IQR])

| system | waveform / pipeline params | FLOPs point / gen / both (incl. detector) | GPU both (ms) | CPU 4-thread both (ms) | GPU point / gen (ms) |
|---|---|---|---|---|---|
| separate-naive | 1,191,910 / 1,849,704 | 0.94 / 1.58 / 2.52 G | 14.40 [14.29, 14.68] | 34.45 [34.25, 34.93] | 1.27 / 13.08 |
| **separate-cached** | 1,191,910 / 1,520,807 | 0.94 / 1.58 / 2.19 G | 13.10 [13.02, 13.35] | 32.47 [31.88, 35.41] | 1.26 / 12.35 |
| **S1** | 943,372 / 1,272,269 | 0.94 / 1.14 / 1.49 G | 11.61 [11.54, 11.73] | 26.61 [26.37, 27.56] | 1.32 / 11.15 |

- **Separate-cached:** one detector pass shared by P and G, and Haar(PPG) / Haar(raster) cached across the 8 NFE.
  - Its generation is **bit-identical** to naive.
  - P and G share no other computation, because G mixes x_t into every layer.
- **S1 versus separate-cached:**
  - GPU both outputs −11 %, CPU −18 %, FLOPs −32 %;
  - point-only latency is equal within noise (1.32 vs 1.26 ms);
  - peak batch-1 inference memory above the weights is about 1 MiB for every system.

## 5. External cohort and detector

**Frozen detector on MIMIC-BP** (descriptive; reference = neurokit R on the ECG):

| metric | value |
|---|---|
| precision | 0.049 [0.042, 0.056] |
| recall | 0.045 [0.039, 0.052] |
| F1 | 0.047 [0.040, 0.053] |
| FP / window | 4.78 [4.71, 4.84] |
| RR-MAE | 8.07 ms (matched beats only) |
| HR-MAE | 2.48 bpm |

- **Interpretation (not tested):** the detector produces about the right number of beats, so HR is accurate, but at
  times displaced by more than ±50 ms from the ECG R-peaks.
  - This is consistent with a PPG-to-ECG timing relation in MIMIC-BP that differs from VitalDB's, through pulse-arrival
    delay and / or acquisition alignment.
  - No shift was applied: the preregistration forbids it.
- **Consequence:** every event-based metric of every model is dominated by the conditioning detector, not by the
  waveform architecture. For comparison, the detector on VitalDB had F1 0.81.

## 6. Seed 42 primary result (MIMIC-BP, 1,524 patients, 319,493 windows)

| model | corr | FP / window | recall | F1 | RR-MAE | HR-MAE | MAE | FD |
|---|---|---|---|---|---|---|---|---|
| P42 | 0.6205 [0.6026, 0.6383] | 4.81 | 0.0465 | 0.048 | 8.60 | 2.81 | 0.396 | 33.86 (descr.) |
| S1-42 point | 0.6089 [0.5899, 0.6273] | 4.81 | 0.0462 | 0.048 | 8.81 | 2.68 | 0.402 | 23.91 (descr.) |
| G42 (1 sample) | 0.532 (descr.) | 4.87 | 0.048 | 0.049 | — | — | 0.423 | **17.83** |
| S1-42 gen | 0.528 (descr.) | 4.86 | 0.048 | 0.048 | — | — | 0.425 | **17.65** |

| gate | effect [95 % CI] | result |
|---|---|---|
| **T1** Δcorr | −0.0116 [−0.0169, −0.0064] | **PASS** (> −0.02) |
| **T2** ΔFP | −0.0028 [−0.0056, +0.0004] | **PASS** |
| **T2** Δrecall | −0.0002 [−0.0004, −0.0001] | **PASS** |
| **T3** ΔFD | −0.18 [−0.29, −0.06] | **PASS** (< +1.0) |
| **T4** FD shuffled − conditioned | +3.65 [−0.87, +8.34] | **FAIL** (not entirely > 0) |
| **T4** corr shuffled − conditioned | −0.225 [−0.237, −0.212] | (meets its criterion) |
| **T5** saving | 20.85 % | **PASS** |

- **T4:** S1's FD rose from 17.65 to 21.31 with shuffled PPG, but the patient-bootstrap interval reaches below 0.
- **The specialist G42 behaves the same way** (descriptive): +2.49 [−1.80, +6.91], corr −0.209.
- So seed 42's generative readouts, G and S1 alike, show weak population-level (FD) dependence on PPG in this domain.
  The sample-level dependence remains strong: corr falls by about 0.22.

## 7. Seeds 43 / 44 (identical protocol)

| seed | Δcorr | ΔFP | Δrecall | ΔFD | PPG shuffle ΔFD | PPG shuffle Δcorr | T1–T4 |
|---|---|---|---|---|---|---|---|
| 42 | −0.0116 [−0.0169, −0.0064] | −0.0028 [−0.0056, +0.0004] | −0.0002 | −0.18 [−0.29, −0.06] | +3.65 [−0.87, +8.34] | −0.225 | **FAIL (T4)** |
| 43 | +0.0051 [−0.0006, +0.0105] | −0.0152 [−0.0183, −0.0125] | +0.0003 | −0.69 [−1.12, −0.24] | +35.20 [+31.92, +38.45] | −0.236 | PASS |
| 44 | −0.0119 [−0.0167, −0.0068] | −0.0011 [−0.0034, +0.0010] | −0.0001 | +0.63 [+0.51, +0.75] | +11.15 [+7.05, +15.38] | −0.187 | PASS |

- **Absolute values:**
  - P corr 0.620 / 0.620 / 0.627; S1 point corr 0.609 / 0.625 / 0.615;
  - G FD 17.83 / 10.55 / 9.02; S1 FD 17.65 / 9.86 / 9.64.
- **Mean ± SD (min, max) across the 3 seeds:**
  - Δcorr −0.0061 ± 0.0097 (−0.0119, +0.0051);
  - ΔFP −0.0064 ± 0.0077;
  - ΔFD −0.08 ± 0.66 (−0.69, +0.63);
  - PPG shuffle ΔFD +16.7 ± 16.5 (3.7, 35.2).
- **Robustness: ROBUST-2/3.** Seeds 43 and 44 pass T1–T4; seed 42 fails T4.
- **The seed-42 generative pair has the highest FD of the three seeds** (G 17.8 vs 10.6 / 9.0). Its FD shuffle effect
  is the smallest.

## 8. K16 characterization (secondary; 2,000 windows, seed 42)

| | G42 | S1-42 |
|---|---|---|
| within-condition waveform diversity | 0.245 | 0.235 |
| ratio to √2 · RMS(ECG − μ_P) | 0.33 | 0.32 |
| beat-aligned diversity | 0.232 | 0.228 |
| generated / real beat diversity | 0.66 | 0.67 |
| R-time seed SD | 0 ms | 0 ms |
| HR MAE single sample → K16 consensus | 3.24 → 2.79 bpm | 3.17 → 2.76 bpm |
| sample-mean corr | 0.547 | 0.538 |

The samples are under-dispersed, and R timing does not vary across seeds. No uncertainty-calibration claim is made.

## 9. Patient-level heterogeneity (seed 42, descriptive)

| quantity | patients | median | 10–90 % range | fraction < 0 |
|---|---|---|---|---|
| Δcorr (S1 − P) | 714 with pairs | −0.002 | −0.070 to +0.034 | — |
| ΔFP | 1,524 | 0.000 | −0.029 to +0.024 | 41.5 % |
| ΔHR-MAE | 1,524 | −0.001 | −0.44 to +0.14 bpm | — |

There are no subgroups.

## 10. Reading

- **What DP3 tested:** whether sharing preserves specialist capability under a cross-source shift. On MIMIC-BP, the
  shared model and its specialists degraded together.
  - Morphology fell from about 0.80 to about 0.62, and events collapsed with the detector.
  - The paired S1 − specialist differences stayed small for every seed: |Δcorr| ≤ 0.012, |ΔFP| ≤ 0.016,
    |ΔFD| ≤ 0.69.
  - So the sharing itself did not add a material degradation in this domain.
- **Why the primary verdict is still FAILED:** the frozen condition-use gate T4 failed for seed 42 on its FD criterion.
- **Event-based conclusions here are weak:** the conditioning detector does not transfer to MIMIC-BP. T2 compares two
  models with recall ≈ 0.046, and T1 uses only the beats that the detector happened to place within ±50 ms.

## 11. Supported claims

- **Internal:** the DP0 internal confirmation on AF-LOCK (seed 42) stands. The internal multi-seed characterization
  shows small Δcorr for all seeds, with ΔFP and ΔFD varying by seed (§3).
- **Parameters:** S1 uses 943,372 waveform parameters vs 1,191,910 for two separate specialists (−20.85 %).
- **Compute:** relative to a fair separate-cached baseline, S1 needs less compute to produce both outputs.
  - GPU batch-1: 11.6 vs 13.1 ms.
  - CPU 4 threads: 26.6 vs 32.5 ms.
  - FLOPs incl. detector: 1.49 vs 2.19 G.
  - Point-only latency is the same.
- **On the ECG-target-blind cross-task MIMIC-BP cohort**, for all three seeds, S1's paired differences to the
  same-seed specialists stayed within the frozen T1 / T2 / T3 margins. However:
  - T2 is uninformative here (detector failure);
  - seed 42 failed T4, so this is not a confirmed external result.
- **Robustness:** across the two predefined replication seeds, both passed T1–T4 on MIMIC-BP (ROBUST-2/3).

## 12. Unsupported claims

- target-blind external confirmation (the primary failed);
- fully independent, project-naive, completely unseen or untouched external validation;
- event-level fidelity on MIMIC-BP (the detector does not transfer);
- better point reconstruction than WW-L1, or better generation than ScaleFlow;
- calibrated or patient-specific uncertainty;
- clinical validity, or generalization to all hospitals;
- a causal mechanism for the sharing-depth result;
- absolute state of the art, novelty or first-of-its-kind.

## 13. No rescue

After MIMIC-BP was opened, none of the following was done:
- no retraining or fine-tuning, and no seed added or dropped;
- no change to the detector, threshold, alignment, resampling, normalization, lead, exclusions, architecture, margins,
  NFE or solver;
- no subject removed.

The evaluation ran exactly once (19:39–20:49). The frozen files were verified unchanged at load time.

## Files

- **Docs:** `docs/DP3_MIMICBP_TARGET_BLIND_AUDIT.md`, `docs/DP3_MIMICBP_INTERFACE.md`,
  `docs/DP3_MIMICBP_TARGET_BLIND_PREREGISTRATION.md`, this report.
- **Code:** `scripts/dp3_mimicbp.py`, `tests/test_dp3_mimicbp.py`.
- **Artifacts:** `artifacts/dp3_mimicbp/`
  - audit, schema, adapter, manifests, rules, exclusion log;
  - training manifest, checkpoint hashes, freeze;
  - internal multi-seed and compute JSONs;
  - external seed 42 / 43 / 44 metrics, bootstrap and gates; the multi-seed summary; condition shuffle; detector;
    K16; cohort;
  - `table_external_main.csv`, `table_external_seeds.csv`, `table_compute.csv`;
  - `figure_external_main.png`, `figure_multiseed.png`, `figure_compute.png`.
- **Checkpoints (not committed):** `outputs/dp3_mimicbp/seed{43,44}/`. Seed 42 lives in `outputs/dp0_dualreadout/`.
