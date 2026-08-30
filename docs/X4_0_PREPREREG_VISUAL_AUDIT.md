# X4-0 — Pre-preregistration visual audit (DEVELOPMENT EVIDENCE, NOT QUANTITATIVE)

Written 2026-08-31, **before** the X4-0 pre-registration. It records, in full, what has already been *seen* on the WildPPG
development-validation subjects, so that X4-0 is correctly labelled a **development / mechanism diagnostic** rather than an
independent confirmatory experiment. Nothing here may be converted into a quantitative claim.

## 1. What was viewed, and when

On 2026-08-31, at the user's request, three figures were generated and inspected on WildPPG **validation** subjects `an0` and `k2s`.
No test subject (`kjd`, `ssx`) was loaded at any point. No model was trained; every model was a frozen checkpoint.

| Item | Value |
|---|---|
| Subjects | `an0`, `k2s` (development validation) |
| Candidate pool | the deterministic stride subsample cached by X3-G0 (`outputs/x3_g0_coupling_geometry/pool_{an0,k2s}.npz`): 3,698 + 3,860 = 7,558 windows |
| Window-selection rule | `np.round(np.linspace(0, N-1, 6)).astype(int)[1:5]` over the concatenated pool — i.e. 4 evenly spaced positions, fixed **before** looking at any output |
| Gaussian source | one shared bank, `torch.Generator().manual_seed(0)`, reused for every NFE |
| Figures | `artifacts/nfe_visualization/fig1_one_step.png`, `fig2_nfe_sweep.png`, `fig3_trajectory.png` (+ `traces.npz`) |

**The four viewed windows, resolved to their original per-subject indices** (these are the windows X4-0 must exclude):

| Pooled index (as printed at the time) | Subject | Original `window_index` |
|---:|---|---:|
| 1511 | `an0` | 9066 |
| 3023 | `an0` | 18138 |
| 4534 | `k2s` | 5852 |
| 6046 | `k2s` | 16436 |

## 2. Frozen models used in the visualisation

| Label | Checkpoint | Round | Notes |
|---|---|---:|---|
| iMF | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` | 45 | `MeanFlowS5` on the PENGUIN S5 backbone; `cond_mode = h_only`, `h_scale = 1.0` |
| OT-CFM | `outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt` | 189 | upstream PENGUIN Flow-SSM/S5, OT-CFM objective |
| A6 MSE | `outputs/a6c_fullbackbone_mse_wildppg_seed42/checkpoint_best.pt` | 33 | deterministic conditional-centre proxy, one network evaluation |

All three were trained on the 12 WildPPG **training** subjects, so `an0`/`k2s` are held out for every one of them.

## 3. Qualitative observations recorded at the time

These are impressions from **four** windows. They motivate hypotheses; they are not evidence.

- **iMF-1 already produces ECG-like, high-amplitude, visibly sharp waveforms.** The one-step output is not a flattened envelope.
- Some windows look visually strong — beats in roughly the right places with plausible QRS shape.
- Other windows contain **extra, misplaced or mismatched sharp deflections**: events that look like beats but do not correspond to a
  ground-truth beat, and ground-truth beats with no clear counterpart.
- Increasing NFE **visually organises** the waveform further; the outputs look progressively tidier from 1 → 4 → 8 → 16.
- OT-CFM-1, by contrast, is a low-amplitude attenuated trace, and OT-CFM at 2–4 NFE is visibly unstable (high-frequency excursions)
  before settling by 8–16 NFE.
- In the trajectory figure, **much of the visible ECG structure appears late in the sampler** — intermediate states remain
  noise-dominated until the final one or two steps, for both iMF (8 steps) and OT-CFM (Heun-25).

## 4. Status and consequences for X4-0

- These four windows are **development evidence**. They may not be cited as quantitative support for any X4-0 conclusion.
- `an0`/`k2s` are therefore described throughout X4-0 as **development validation**, never as pristine confirmatory validation.
- X4-0's quantitative subsets are chosen by a **deterministic hash over (subject, original window index)** with fixed salts, and the
  four windows above are **excluded by construction** from every X4-0 metric subset.
- The observation "additional NFE visually organises the waveform" is exactly the kind of impression that motivates hypothesis **H1**;
  "extra / misplaced sharp deflections" motivates **H2**; the one-step boundary query motivates **H3**. X4-0 exists to test these
  quantitatively, on disjoint windows, under a frozen protocol.
- **No WildPPG test data (`kjd`, `ssx`) was accessed during the visualisation, and none will be accessed in X4-0.**
