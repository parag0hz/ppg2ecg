# SF0 audit (2026-10-02) — before any SF0 training or SF-VAL outcome

## 1. Closed line untouched

No BF0 / D0 / C0 / C0-A / E0 / R1 file, checkpoint or verdict is modified. SF0 imports the C0 / C0-A / R1 modules.

## 2. Split (`split_manifest.json`)

- **Construction:** from C0's ARCH-TRAIN (3,470 patients), permuted with `default_rng(20261002)`.
  - SF-VAL: 433 patients, 29,223 windows.
  - SF-TRAIN: 3,037 patients, 201,997 windows.
- **Checks — all pass:**
  - the two sets are disjoint;
  - their union is ARCH-TRAIN;
  - no ARCH-VAL or HOLDOUT patient;
  - no old V1 validation or test patient;
  - exact counts.
- **Hashes:** `split_hashes.json`.
- **Evidence status:** SF-VAL patients were training data of the earlier detector and waveform models, so they are new
  to this line but not project-naive.

## 3. Reused conventions

| element | source |
|---|---|
| detector | RD1 RhythmTCN: Gaussian σ = 20 ms target, BCE, 14,000 × 64, AdamW 1e-3 / 0.01; events at threshold 0.35, refractory 32 (C0) |
| WW family | C0-A WWDet; C0 Encoder (64 ch, 8 dilated residual blocks) |
| metrics | C0-A `arm_metrics`, R1 patient-macro rows, `kanflow_fd`, matched-pair correlation, C0 bootstrap helpers (seed 20261002) |

## 4. Parameter matching (`parameter_match.json`)

Parameter count was the only criterion. Depths were fixed structurally.

| model | width | params | vs SCALEFLOW |
|---|---|---|---|
| SCALEFLOW-COUPLED | 50 | 598,333 | — (600k −0.28%) |
| SCALE-FM-INDEPENDENT | 50 | 593,485 | −0.81% |
| WW-FM | decoder 63 | 597,664 | −0.11% |
| WW-L1 | (WW-DET 72 × 5) | 593,577 | — |

## 5. Decisions the specification leaves open (frozen in the preregistration)

- **Training raster:** reference R (WW-DET convention) for all four models. The evaluation raster comes from the SF
  detector's events, identical for all four.
- **WW-FM design:**
  - x_t enters the decoder input together with the raster;
  - the time embedding is injected into the 5 decoder blocks only;
  - the PPG encoder is the condition encoder and has no time input.
- **Scale branches:**
  - 6 blocks, dilations 1–32;
  - coupling through the final branch features, projected to 16 channels, concatenated to the receiving branch's input;
  - nearest ×2 upsampling.
- **Noise map:** sha256 of "patient:window:20261002". Secondary samples add ":k{k}".
- **G1 labels and the TEST verdict:** preregistration §9 and §11.
- **Correlation population:** pairs finite in all six evaluated waveforms.

## 6. Synthetic dry run

Every stage from detector training to the figure ran on synthetic data (30 steps, FD stub). It found and fixed NaN-safe
handling in the figure and table.

The numbers mean nothing. They did show the compute profile:

- per-window FLOPs at NFE 8 are much lower for the scale models (shorter sequences per branch) than for WW-FM;
- batch-1 GPU latency is higher for the scale models (3 sequential branches × 6 blocks).
