# GX1 — PPG + ABP data audit (before any GX1 training)

**Selected:** MIMIC-BP, using its pre-existing official subject split.
**Evidence label:** CROSS-TASK GENERALIZATION (historically used ABP dataset; not fresh external confirmation).

## Candidates (metadata only)

| dataset | subjects | PPG | ABP | sampling | subject IDs | sync | prior project use | prior ABP use | source overlap |
|---|---|---|---|---|---|---|---|---|---|
| **MIMIC-BP** (`data/raw/MIMIC-BP`) | 1,524 | yes | yes (mmHg) | 125 Hz | yes (pNNNNNN) | curated per segment (shared indices; ECG / PPG f0 and pulse-arrival-time checks) | PPG → ABP: A7, A8, U1, U2, EXP-D; PPG → ECG: DP3 (all subjects) | **yes** (A7 / A8 / U1 / U2 / EXP-D) | MIMIC family (BIDMC, UCI-BP possible) |
| UCI-BP (`data/raw/UCI-BP`) | unknown (12,000 records) | yes | yes | 125 Hz | **no** | uncertain | D3, U1, U2 | yes | MIMIC-II |
| BIDMC (`data/raw/BIDMC`) | 53 records; ABP in 8, ART in 2 | yes | in ≤ 10 records | 125 Hz | yes | WFDB record | ECG / respiration (D1, D3, U1, U2, MC1, EXP-D) | no | MIMIC-II |
| VitalDB (local copy) | — | PLETH | **not present locally** (case files hold PLETH, ECG_II only) | 500 Hz | yes | one clock | the DP0 development source | — | — |

## Decision

- **No completely fresh PPG + ABP cohort exists on this machine.**
  - UCI-BP cannot be split by patient.
  - BIDMC has at most 10 ABP records.
  - The local VitalDB copy has no ABP track.
- **MIMIC-BP is the only usable cohort.** It was not used to design DualReadout: the architecture was developed and
  frozen on VitalDB, and DP3 only evaluated it afterwards. It was, however, used for PPG → ABP work (A7, A8, U1, U2,
  EXP-D) and for PPG → ECG evaluation (DP3). GX1 therefore proceeds as **cross-task generalization**, not as fresh
  external confirmation.
- **Split (GX1 spec §4):** the pre-existing frozen official MIMIC-BP split (`data/manifests/split_a7_mimicbp_official.json`,
  shipped with the dataset and frozen at A7) is patient-disjoint and of clear provenance, so it is used unchanged.

  | GX1 role | official role | subjects |
  |---|---|---|
  | GX-TRAIN | train | 1,100 |
  | GX-DEV | val | 195 |
  | GX-LOCK | test | 229 |

  - GX-LOCK is unopened for every GX1 model and is read only after the committed final freeze.
  - Its subjects were nevertheless evaluated for PPG → ABP by other models (A7 / U2 / EXP-D) and for PPG → ECG
    (DP3). This is disclosed.
- **Target normalization** (GX1 spec §2, frozen in existing code): the A8 global TRAIN-only affine,
  `y_z = (y_mmHg − 77.571767) / 22.275611` (`artifacts/a8_abp_scale_control/normalization.json`). It was computed from
  the same 1,100 official-train subjects. Every ABP metric in mmHg uses the inverse transform.

## Built populations (`artifacts/gx1_abp/split_manifest.json`)

| role | subjects | windows | exclusions |
|---|---|---|---|
| GX-TRAIN | 1,100 | 230,812 / 231,000 | R4 188 |
| GX-DEV | 195 | 40,592 / 40,950 | R4 358 |
| GX-LOCK | 229 | built only after the freeze | — |

- 4 s windows, 7 per 30 s segment; PPG `PPG_KW`; ABP FFT-resampled only.
- The ECG (`ECG_KW`) is used only for the reference-R raster, the detector targets and rule R4.
