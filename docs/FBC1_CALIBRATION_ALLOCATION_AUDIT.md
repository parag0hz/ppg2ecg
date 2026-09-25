# FBC1 — Functional Budget Calibration: audit (no FBC1 result computed)

Audit 2026-09-25, `artifacts/fbc1_calibration_allocation/audit.json` (stage `scripts/fbc1_run.py audit`).

**What was inspected:** schemas, shapes, draw-level and window-level non-finite HR rates, reference finiteness and sha256
hashes. **What was not computed:** no consensus error against a reference, no risk, no selection. M1 and M2 are not
modified.

## Generators, samplers, seeds
| model | checkpoint (training seed 42) | sha256 | S-step sampler (`dw1_depth_width.make_sampler`) |
|---|---|---|---|
| iMF | `outputs/v1_vitaldb_armI_seed42/checkpoint_last.pt` | `ddbc39e6…` | uniform MeanFlow schedule [1/S]·S |
| CD | `outputs/cd1_vitaldb_armD_seed42/checkpoint_last.pt` | `faea968b…` | consistency at t = 0, then re-noise at t = j/S (S − 1 extra N(0, I) draws from the same generator after z0) |
| PENGUIN | `outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt` | `05c9748b…` | forward Euler, S steps (NFE = S) |

- **Draw k** = `torch.Generator().manual_seed(k)`, with z0 = randn(n_windows, 1, 512) over the whole split and batch 512.
- The candidate with K samples at depth S uses seeds 0…K−1 (prefix reuse).
- HR = `v1_evaluate._hr` (neurokit R-peaks, 60·fs·(n−1)/span at 128 Hz; NaN iff < 2 peaks), snapped to 1e-3 bpm.

## Splits
- **Validation:** 289 patients, 4,822 windows. Order `vm1_evaluate.load("val")`, reference
  `outputs/m1_pilot_gain_prediction/val_ref.npz`.
- **Test:** 1,156 patients, 19,543 windows. Reference and pid from `outputs/sr1_eval/arm_I_seed42.npz`, in DW1 bank order.
- **0 patients are shared** between validation and test. Reference HR is 100 % finite on both splits.
- **Test is legacy:** DW1, DW2, EXP-B, M1 and M2 analysed it, and the grid and the historical static cells come from it.
- **Validation was used before** by M1/M2 (width vs depth at S ≤ 8, ≤ 16 draws) and by VM1 (CFG choice). No B = 16 / 32
  allocation risk has been computed on validation.

## Maximum available K per S
| split | S=1 | S=2 | S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|---|---|
| validation before FBC1 (M1 bank) | 16 | 16 | 16 | 16 | 0 | 0 |
| validation after FBC1 generation | **32** | 16 | 16 | 16 | **2** | **1** |
| test (DW1 bank `outputs/dw1_raw/hr_{arm}{S}.npy`) | 32 | 16 | 8 | 4 | 2 | 1 |

B = 32 needs K = 32/16/8/4/2/1 at S = 1/2/4/8/16/32; B = 16 needs K = 16/8/4/2/1 at S = 1/2/4/8/16. **Every cell is exactly
reconstructable on both splits.**

## Generation of missing validation draws (done; no test generation)
- Script `scripts/fbc1_val_bank.py`: frozen checkpoints and samplers, no training, no test window, no reference read.
- Generated per model: **S = 1 seeds 16–31, S = 16 seeds 0–1, S = 32 seed 0**, written to
  `outputs/fbc1_calibration_allocation/val_hr_{arm}_S{S}_seeds{a}_{b}.npy` (not committed).
- Determinism: seed 15 at S = 1 was regenerated for every model and is **bit-identical** to the M1 bank row (NaN pattern
  identical, max |diff| 0).
- Test: every required cell already exists in the DW1 grid, so **no test sample was generated**. DW1 took rows 0–15 at
  S = 1, 2 (iMF) and S = 1 (CD, PENGUIN) from AB1 (batch 64). The M2 audit showed differences of ≤ ~1e-5 bpm, which the
  1e-3 snap removes.

## Per model / split / budget / cell
| model | split | B | K | S | available? | source | needs generation? | draw non-finite | windows with no finite draw |
|---|---|---:|---:|---:|---|---|---|---:|---:|
| iMF | val | 32 | 32 | 1 | yes | M1 bank + FBC1 seeds 16–31 | yes (generated) | 0.78 % | 0.000 % |
| iMF | val | 32 | 16 | 2 | yes | M1 bank | no | 0.99 % | 0.000 % |
| iMF | val | 32 | 8 | 4 | yes | M1 bank | no | 2.30 % | 0.000 % |
| iMF | val | 32 | 4 | 8 | yes | M1 bank | no | 3.16 % | 0.187 % |
| iMF | val | 32 | 2 | 16 | yes | FBC1 generation | yes (generated) | 3.46 % | 1.307 % |
| iMF | val | 32 | 1 | 32 | yes | FBC1 generation | yes (generated) | 3.61 % | 3.608 % |
| iMF | val | 16 | 16 | 1 | yes | M1 bank | no | 0.80 % | 0.000 % |
| iMF | val | 16 | 8 | 2 | yes | M1 bank | no | 0.97 % | 0.000 % |
| iMF | val | 16 | 4 | 4 | yes | M1 bank | no | 2.39 % | 0.021 % |
| iMF | val | 16 | 2 | 8 | yes | M1 bank | no | 3.08 % | 1.058 % |
| iMF | val | 16 | 1 | 16 | yes | FBC1 generation | yes (generated) | 3.42 % | 3.422 % |
| CD | val | 32 | 32 | 1 | yes | M1 bank + FBC1 seeds 16–31 | yes (generated) | 1.72 % | 0.000 % |
| CD | val | 32 | 16 | 2 | yes | M1 bank | no | 0.27 % | 0.000 % |
| CD | val | 32 | 8 | 4 | yes | M1 bank | no | 0.62 % | 0.000 % |
| CD | val | 32 | 4 | 8 | yes | M1 bank | no | 0.64 % | 0.000 % |
| CD | val | 32 | 2 | 16 | yes | FBC1 generation | yes (generated) | 0.99 % | 0.228 % |
| CD | val | 32 | 1 | 32 | yes | FBC1 generation | yes (generated) | 1.37 % | 1.369 % |
| CD | val | 16 | 16 | 1 | yes | M1 bank | no | 1.72 % | 0.000 % |
| CD | val | 16 | 8 | 2 | yes | M1 bank | no | 0.27 % | 0.000 % |
| CD | val | 16 | 4 | 4 | yes | M1 bank | no | 0.68 % | 0.000 % |
| CD | val | 16 | 2 | 8 | yes | M1 bank | no | 0.58 % | 0.021 % |
| CD | val | 16 | 1 | 16 | yes | FBC1 generation | yes (generated) | 0.97 % | 0.975 % |
| PENGUIN | val | 32 | 32 | 1 | yes | M1 bank + FBC1 seeds 16–31 | yes (generated) | 5.11 % | 0.000 % |
| PENGUIN | val | 32 | 16 | 2 | yes | M1 bank | no | 2.24 % | 0.000 % |
| PENGUIN | val | 32 | 8 | 4 | yes | M1 bank | no | 1.36 % | 0.000 % |
| PENGUIN | val | 32 | 4 | 8 | yes | M1 bank | no | 0.19 % | 0.000 % |
| PENGUIN | val | 32 | 2 | 16 | yes | FBC1 generation | yes (generated) | 0.04 % | 0.000 % |
| PENGUIN | val | 32 | 1 | 32 | yes | FBC1 generation | yes (generated) | 0.06 % | 0.062 % |
| PENGUIN | val | 16 | 16 | 1 | yes | M1 bank | no | 5.09 % | 0.000 % |
| PENGUIN | val | 16 | 8 | 2 | yes | M1 bank | no | 2.12 % | 0.000 % |
| PENGUIN | val | 16 | 4 | 4 | yes | M1 bank | no | 1.47 % | 0.000 % |
| PENGUIN | val | 16 | 2 | 8 | yes | M1 bank | no | 0.20 % | 0.000 % |
| PENGUIN | val | 16 | 1 | 16 | yes | FBC1 generation | yes (generated) | 0.04 % | 0.041 % |
| iMF | test | 32 | 32 | 1 | yes | DW1 test bank | no | 0.82 % | 0.000 % |
| iMF | test | 32 | 16 | 2 | yes | DW1 test bank | no | 1.02 % | 0.000 % |
| iMF | test | 32 | 8 | 4 | yes | DW1 test bank | no | 2.42 % | 0.005 % |
| iMF | test | 32 | 4 | 8 | yes | DW1 test bank | no | 3.25 % | 0.174 % |
| iMF | test | 32 | 2 | 16 | yes | DW1 test bank | no | 3.72 % | 1.279 % |
| iMF | test | 32 | 1 | 32 | yes | DW1 test bank | no | 3.82 % | 3.822 % |
| iMF | test | 16 | 16 | 1 | yes | DW1 test bank | no | 0.80 % | 0.000 % |
| iMF | test | 16 | 8 | 2 | yes | DW1 test bank | no | 1.03 % | 0.000 % |
| iMF | test | 16 | 4 | 4 | yes | DW1 test bank | no | 2.35 % | 0.067 % |
| iMF | test | 16 | 2 | 8 | yes | DW1 test bank | no | 3.25 % | 1.049 % |
| iMF | test | 16 | 1 | 16 | yes | DW1 test bank | no | 3.64 % | 3.638 % |
| CD | test | 32 | 32 | 1 | yes | DW1 test bank | no | 1.80 % | 0.000 % |
| CD | test | 32 | 16 | 2 | yes | DW1 test bank | no | 0.26 % | 0.000 % |
| CD | test | 32 | 8 | 4 | yes | DW1 test bank | no | 0.67 % | 0.000 % |
| CD | test | 32 | 4 | 8 | yes | DW1 test bank | no | 0.60 % | 0.000 % |
| CD | test | 32 | 2 | 16 | yes | DW1 test bank | no | 1.04 % | 0.118 % |
| CD | test | 32 | 1 | 32 | yes | DW1 test bank | no | 1.63 % | 1.627 % |
| CD | test | 16 | 16 | 1 | yes | DW1 test bank | no | 1.78 % | 0.000 % |
| CD | test | 16 | 8 | 2 | yes | DW1 test bank | no | 0.27 % | 0.000 % |
| CD | test | 16 | 4 | 4 | yes | DW1 test bank | no | 0.69 % | 0.000 % |
| CD | test | 16 | 2 | 8 | yes | DW1 test bank | no | 0.60 % | 0.051 % |
| CD | test | 16 | 1 | 16 | yes | DW1 test bank | no | 1.13 % | 1.131 % |
| PENGUIN | test | 32 | 32 | 1 | yes | DW1 test bank | no | 5.27 % | 0.000 % |
| PENGUIN | test | 32 | 16 | 2 | yes | DW1 test bank | no | 2.23 % | 0.000 % |
| PENGUIN | test | 32 | 8 | 4 | yes | DW1 test bank | no | 1.46 % | 0.000 % |
| PENGUIN | test | 32 | 4 | 8 | yes | DW1 test bank | no | 0.15 % | 0.000 % |
| PENGUIN | test | 32 | 2 | 16 | yes | DW1 test bank | no | 0.05 % | 0.000 % |
| PENGUIN | test | 32 | 1 | 32 | yes | DW1 test bank | no | 0.10 % | 0.097 % |
| PENGUIN | test | 16 | 16 | 1 | yes | DW1 test bank | no | 5.29 % | 0.005 % |
| PENGUIN | test | 16 | 8 | 2 | yes | DW1 test bank | no | 2.25 % | 0.000 % |
| PENGUIN | test | 16 | 4 | 4 | yes | DW1 test bank | no | 1.48 % | 0.020 % |
| PENGUIN | test | 16 | 2 | 8 | yes | DW1 test bank | no | 0.15 % | 0.000 % |
| PENGUIN | test | 16 | 1 | 16 | yes | DW1 test bank | no | 0.07 % | 0.067 % |

## Missing HR
- **Draw level:** 0.04–5.3 % non-finite.
  - PENGUIN is worst at S = 1 (5.1 %).
  - iMF gets worse with depth (3.6 % at S = 32).
- **Window level:** with K ≥ 8 draws, windows with *no* finite draw are rare (≤ 0.07 %). They are not rare for the depth-heavy
  cells: iMF (1,32) 3.6 % (test 3.8 %), (2,16) 1.3 %; CD (1,32) 1.4 % (test 1.6 %); iMF (1,16) 3.4 %.
- **FBC1 rule:** such a window gets the frozen fallback **c_train = 73.143 bpm**, the median reference HR of the
  288,400 VitalDB **training** windows (`outputs/db1_hr_regressor/train_hr_labels.npz`, sha256 `c3ef48e3…`). No validation
  or test label is involved, and the deployed estimator always outputs a number.
- **DW1 historical convention:** `nanmedian` → NaN, and `cluster_ci` takes the per-patient `nanmean`, so such windows were
  *dropped*. That favours depth-heavy cells. FBC1 reports the DW1 convention only as a descriptive sensitivity. FBC1 legacy-test
  numbers for depth-heavy cells will therefore differ slightly from DW1's table.

## Other facts used by the preregistration
- **Historical static cells** (frozen DW1 `result.json`, prereg `cc902b3`, legacy test):

  | B | iMF | CD | PENGUIN |
  |---|---|---|---|
  | 32 | (16,2) | (32,1) | (8,4) |
  | 16 | (8,2) | (16,1) | (8,2) |

  The B = 32 cells were seed-stable in DW2.
- **Compute:** there is no FLOPs infrastructure in the repository. PyTorch 2.11's built-in `torch.utils.flop_counter` is
  available, and FBC1 records it if it runs (no counter is written). Timing uses the RTX 5090 with no concurrent GPU job. An
  unrelated CPU-bound process by the same user is running on the host, and it is not touched.

**No hard stop:** every candidate allocation can be reconstructed exactly, and the validation-only primary design needs no
test label for any design decision.
