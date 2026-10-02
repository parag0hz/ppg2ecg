# GX1 — ABP evaluator audit (before any GX1 training)

**Result: a frozen, pre-GX1 ABP distributional evaluator exists.** `GX1 GENERATIVE EVALUATOR NOT FROZEN` does **not**
apply.

## What the repository already had before GX1

| metric | implementation | pre-GX1 use on ABP |
|---|---|---|
| FD (KANFlow Eq. 23): Gaussian Fréchet distance on raw flattened windows; ≥ 3,000 windows → raw space, covariance ddof 1 + 1e-4 I | `src/ppg2ecg/evaluation/paper_metrics.py::kanflow_fd`, unchanged since `a15b354` | U2 (`c753d13`): `FD_kanflow` on MIMIC-BP ABP in raw mmHg ("MIMIC-BP 38.6k vs 234k"; `scripts/u2_evaluate.py::pooled_extras`) |
| upstream raw-signal FD | PENGUIN `fid_features_to_statistics` | EXP-D ABP (`b1eb66a`), mmHg² |
| SBP / DBP / MAP errors | window max / min / mean (PENGUIN / EXP-D); beat-level in `abp_metrics.py` | A7, U2, EXP-D |
| waveform Pearson r, MAE, RMSE | per window | A7, EXP-D |
| spectral (HF > 5 Hz ratio), upstroke slope | `src/ppg2ecg/evaluation/abp_metrics.py` | A7 / A8 |

No MMD or learned ABP feature extractor exists in the repository, and none is introduced.

## Frozen GX1 choices (`artifacts/gx1_abp/evaluator_manifest.json`, `margins.json`)

- **D_ABP:** `kanflow_fd` on ABP **in mmHg** (predictions inverse-transformed with the frozen A8 pair), over all
  GX-LOCK windows, one generation per window. This is the U2 convention, without U2's 3,000-window subsample.
  - Patient-clustered bootstrap by `anchorflow.fastfd`, verified against `kanflow_fd` at run time.
- **G3 margin (TRAIN-only):**
  - DP0's ECG FD margin of +1.0 is converted to ABP units by the ratio of total target variances, computed on training
    data only (FD scales with the square of the signal unit):

    | quantity | value |
    |---|---|
    | tr(Cov) of GX-TRAIN ABP windows (mmHg) | 254,029.1 |
    | tr(Cov) of DP-TRAIN ECG windows | 91.199 |
    | **margin** | **+2,785.4 mmHg²** |

- **G2 margin (TRAIN-only):**
  - The ECG point gate used 0.02 on a unit-free correlation. The same 0.02 is applied as a fraction of the GX-TRAIN SD
    of the normalized ABP target (1.0001).
  - **Margin: +0.0200** in normalized MAE units (≈ 0.45 mmHg).
- **Point metrics:**
  - per-window Pearson r (G1) and normalized MAE (G2) against the true normalized ABP;
  - SBP / DBP / MAP MAE in mmHg (window max / min / mean), secondary.

Nothing above depends on any GX1 outcome. The margins were computed from GX-TRAIN (ABP) and DP-TRAIN (ECG) target
windows only, before any GX1 model existed.
