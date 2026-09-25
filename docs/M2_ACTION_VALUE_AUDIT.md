# M2 — Cost-matched width-vs-depth action prediction: audit (no M2 result computed)

Audit 2026-09-25, `artifacts/m2_action_value/audit.json`. Only schemas, shapes, non-finite rates and provenance were
inspected; no action loss, no error against the reference and no predictive quantity was computed. M1 (`7526201`,
`cff9388`, `672e0d4`, `184755d`, `2d86f9e`) is not modified.

## Conditions and banks
M2 base depths S ∈ {1, 2, 4} (depth action at 2S ∈ {2, 4, 8}) × iMF, CD, PENGUIN (seed-42 checkpoints) = **9 conditions**.

| Model | Split | S used | Available K (rows) | Same seeds across S? | HR available? | Waveforms? | Usable |
|---|---|---|---:|---|---|---|---|
| iMF | val | 1, 2, 4, 8 | 16 each | exact (z0 is the only randomness) | yes | not needed | yes |
| iMF | test | 1, 2, 4, 8 | 32 / 32 / 16 / 16 | exact | yes | not needed | yes |
| CD | val | 1, 2, 4, 8 | 16 each | **z0 only** — re-noise tensors enter at t = j/S, so S and 2S trajectories are not paired | yes | not needed | yes |
| CD | test | 1, 2, 4, 8 | 32 / 16 / 16 / 16 | z0 only | yes | not needed | yes |
| PENGUIN | val | 1, 2, 4, 8 | 16 each | exact | yes | not needed | yes |
| PENGUIN | test | 1, 2, 4, 8 | 32 / 16 / 32 / 16 | exact | yes | not needed | yes |

Only seeds 0–15 are used. M2 does not need trajectory pairing between S and 2S: depth outcomes use seed IDs disjoint from
the pilot and from the width outcome.

## Checks
1. **Window alignment.** Validation: `vm1_evaluate.load("val")` order, 4,822 windows (M1 bank, `val_ref.npz`); test:
   `load("test")` order, 19,543 windows (reference and `pid` from `outputs/sr1_eval/arm_I_seed42.npz`, bit-identical to
   a fresh recomputation per the M1 audit). Every bank row has one column per window in that order.
2. **Patient IDs.** Validation 289, test 1,156 patients; **0 patients shared** between validation and test (and none with
   train). Multi-case patients are not contiguous in load order; all grouping is by `pid`.
3. **Reference HR.** 100 % finite on both splits (V1 builder kept only windows with reference HR in [30, 200] bpm).
4. **All 16 draw HR arrays** exist for every (model, depth) needed, on both splits (sha256 in `audit.json`).
5. **Depth S and 2S arrays**: S ∈ {1, 2, 4} and 2S ∈ {2, 4, 8} all present.
6. **Seed provenance.** Row k = noise seed k: `torch.Generator().manual_seed(k)` drawn over the whole split. Validation
   bank: `scripts/m1_val_bank.py` (M1, S = 1 regenerated bitwise identically). Test rows 0–15 come from AB1 / DW1 / EXP-B
   (batch 64 / 512 / 128, float32 / float64 storage; ≤ ~1e-5 bpm differences — M2 snaps HR to 1e-3 bpm as M1 did).
7. **Seed identity across depths**: exact for iMF and PENGUIN, z0-only for CD (above).
8. **Non-finite HR rate** (share of draw-level HR values that are NaN, i.e. < 2 detected R-peaks):

| condition | val S=1 | S=2 | S=4 | S=8 | test S=1 | S=2 | S=4 | S=8 |
|---|---|---|---|---|---|---|---|---|
| iMF | 0.80 % | 0.99 % | 2.37 % | 3.20 % | 0.80 % | 1.02 % | 2.39 % | 3.27 % |
| CD | 1.72 % | 0.27 % | 0.63 % | 0.65 % | 1.78 % | 0.26 % | 0.68 % | 0.64 % |
| PENGUIN | 5.09 % | 2.24 % | 1.41 % | 0.16 % | 5.29 % | 2.23 % | 1.46 % | 0.15 % |

   Windows with all 16 draws finite: 85–98 % per condition. M2 therefore conditions inclusion only on the reference and the
   4 pilot draws, and handles non-finite future draws explicitly (fallback), as the brief requires.
9. **Validation / test separation**: patient-disjoint; all M2 design choices (partitions, features, α, threshold,
   static actions, fallback constant, outlier rule) are fixed from the preregistration or from validation only.

## Other facts used by the preregistration
- HR functional: `v1_evaluate._hr` (neurokit R-peaks → 60·fs·(n−1)/span), NaN iff < 2 peaks.
- Existing physiological HR range convention: **[30, 200] bpm** (`scripts/v1_build_vitaldb.py:54`).
- No FLOPs/MACs infrastructure in the repository: cost is reported as vector-field NFE plus measured latency.
- The test draws were analysed before (DW1, EXP-B, B3, M1) at condition and window level; the M2 action outcomes
  (median of 8 width draws at S vs median of 4 depth draws at 2S, on disjoint seeds) have not been computed before.

**No hard stop:** validation banks and all action outcomes can be built without using test labels for any design decision.
