# GX1 — cross-target generalization of partial DualReadout sharing (PPG → ABP) — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-02).**
- Committed and pushed before GX-LOCK is read.
- The model hashes are frozen in a separate, later commit (`artifacts/gx1_abp/final_freeze_manifest.json`). GX-LOCK
  opens only after that commit.
- The GX-TRAIN models, the GX-DEV characterization (descriptive) and the compute measurement exist at this point.
  None of them is used to change anything below; every rule comes from the GX1 spec and DP0.

## 1. Question and evidence label

- **Question:** does the EXACT S1 MIDDLE sharing topology frozen on PPG → ECG keep point-specialist and
  generative-specialist capability on PPG → ABP, without any sharing-depth search?
- **Topology:** stem + blocks 1–6 shared; blocks 7–8 + decoder private per readout.
- **Evidence label:** CROSS-TASK GENERALIZATION. MIMIC-BP is a historically used ABP dataset, so this is not fresh
  external confirmation (`docs/GX1_ABP_DATA_AUDIT.md`).

## 2. Data and target interface

- **Split:** the pre-existing official MIMIC-BP split, frozen at A7 before GX1 and patient-disjoint.

  | role | subjects | windows |
  |---|---|---|
  | GX-TRAIN | 1,100 | 230,812 |
  | GX-DEV | 195 | 40,592 |
  | GX-LOCK | 229 | built after the freeze |

  - GX-LOCK's subjects were previously evaluated by other models (A7 / U2 / EXP-D for ABP; DP3 for ECG). This is
    disclosed.
- **Windows:** 4 s non-overlapping, 7 per 30 s segment. PPG gets the V1 / DP0 `PPG_KW` preprocessing.
- **ABP:** FFT resample 500 → 512 only, then the frozen A8 global TRAIN-only affine:
  `y_z = (y_mmHg − 77.571767) / 22.275611`. mmHg metrics use the exact inverse.
- **ECG:** used only for the reference-R raster (training), the detector targets and exclusion R4.
- **Exclusions:**
  - R1: unreadable;
  - R2: raw non-finite / constant PPG, ABP or ECG;
  - R3: non-finite after preprocessing;
  - R4: reference HR outside [30, 200] bpm;
  - R5: empty subject.
  - All are per window and logged. None is based on performance.
- **Timing condition:**
  - The RD1 / C0 detector protocol (DP0's) was trained on GX-TRAIN, seed 42. It is conditioning infrastructure, not
    one of the 9 jobs.
  - Events at threshold 0.35, refractory 32, σ 20 ms raster.
  - Training uses the reference-R raster and inference the detector raster, as in DP0.

## 3. Models and training (only the target interface changes)

- **Models** (all trained with the frozen DP0 trainers, given GX-TRAIN tensors):
  - P_ABP = WW-L1 (593,577; L1);
  - G_ABP = SCALEFLOW-COUPLED (598,333; linear-path FM, fixed Haar, Euler NFE 8);
  - S1_ABP = DualReadout S1 MIDDLE (943,372; widths 71 / 30).
- **Target interface:** one output channel, as for ECG; only the target values are normalized ABP.
- **Training:** 20,000 × 64, AdamW 1e-3 / 0.01, clip 1.0, last checkpoint. S1 uses round-robin with 20,000 POINT +
  20,000 FLOW masked updates (odd POINT → FLOW, even FLOW → POINT).
- **Seeds:** 42, 43 and 44 (9 jobs). There is no extra seed and no best-seed selection; seed 42 is primary.
- **Not changed:** sharing depth, blocks, widths, dilations, optimizer, updates, schedule, flow path, Haar, solver,
  NFE.

## 4. Evaluator (`docs/GX1_ABP_EVALUATOR_AUDIT.md`)

- **D_ABP:** `kanflow_fd` on ABP in mmHg over all GX-LOCK windows (one generation per window, `scaleflow.window_noise`,
  identical for G and S1). This is the pre-GX1 FD applied to MIMIC-BP ABP in U2.
- **Point metrics:** per-window Pearson r and normalized MAE; SBP / DBP / MAP MAE in mmHg (window max / min / mean).

## 5. Gates (per seed; seed 42 primary)

| gate | quantity | pass rule |
|---|---|---|
| G1 | corr(S1 point) − corr(P) | 95 % CI lower > −0.02 |
| G2 | MAE_z(S1 point) − MAE_z(P) | 95 % CI upper < **+0.0200** (0.02 × GX-TRAIN target SD) |
| G3 | D_ABP(S1 gen) − D_ABP(G) | 95 % CI upper < **+2,785.4 mmHg²** (DP0 +1.0 × TRAIN target total-variance ratio) |
| G4 | S1 gen with PPG shuffled across windows (`default_rng(20261002)`), raster / noise / model fixed, shared features recomputed − conditioned | D CI entirely > 0 **and** corr CI entirely < 0. The same test on G is a diagnostic control |
| G5 | 1 − P_S1 / (P_P + P_G) | ≥ 15 % (structural 20.85 %) |
| G6 (efficiency, not a seed gate) | S1 vs separate-cached, both outputs, batch-1 | FLOPs lower **and** (GPU or CPU latency lower) |

**Bootstrap:**
- 2,000 patient-clustered paired replicates, seed 20261002, patient = unit;
- D is recomputed inside every replicate (exact fastfd).

## 6. Verdicts (fixed now)

- **Seed:** a seed passes only if G1–G5 all pass.
- **Multi-seed:** GENERALIZES-3/3, GENERALIZES-2/3, or SEED-SENSITIVE (≤ 1 seed passes).
- **Principle:**
  - **CROSS-TARGET PRINCIPLE STRONGLY SUPPORTED:** all 3 seeds pass and G6 is supported.
  - **CROSS-TARGET PRINCIPLE SUPPORTED:** seed 42 passes and ≥ 2 / 3 seeds pass (saving ≥ 15 %).
  - **NOT SUPPORTED:** otherwise, i.e. seed 42 fails any primary gate or ≤ 1 seed passes.
  - The PPG → ECG evidence (DP0 internal confirmation) is taken as it stands.
- **Allowed claim if supported:** "The same intermediate sharing topology identified in PPG-to-ECG transferred without
  sharing-depth search to PPG-to-ABP while preserving deterministic and generative specialist capability."
- **Never claimed:** universal optimality, all tasks benefit, a causal mechanism, a foundation architecture.

## 7. Optional diagnostics (explanatory only)

- **Trained only after the final freeze commit:** FULL-SHARE-ABP (S0) and STEM-ONLY-ABP (S2), seed 42, the same
  protocol.
- **Evaluated only after the primary verdict exists**, on GX-LOCK, without the shuffle.
- They cannot replace S1 or change any verdict.

## 8. Order and no rescue

**Order:**
1. this preregistration (commit + push);
2. final freeze of all checkpoint / code / data-manifest hashes (commit + push);
3. `eval_lock` (refuses otherwise);
4. diagnostics;
5. report.

**After GX-LOCK opens, none of the following is done:** architecture or sharing-depth change, extra seeds, a new
evaluator, margin changes, retraining, a new loss, decoder, NFE or normalization, or performance-based exclusions.

## 9. Tests

`tests/test_gx1_abp.py` (9 test cases):
- the transferred topology and parameter counts;
- target-interface-only changes (frozen A8 pair, DP0 protocol, NFE 8);
- the official disjoint split;
- the GX-LOCK seal;
- the evaluator and TRAIN-only margins;
- the gate rules at the margins;
- wave correlation and BP errors;
- the shuffle keeps raster and noise, diagnostics only after the freeze / verdict, no best seed;
- adapter determinism and the R2 rule on a synthetic subject.
