# DP2 — Stage A: local PPG / ECG dataset audit

> **DP2 NO ELIGIBLE PRIMARY DATASET — HARD STOP after Stage A.**
> - Every PPG + ECG dataset on this machine is either already used for outcome-based work in this project
>   (**CONTAMINATED**) or structurally unusable for paired PPG → ECG confirmation (**INELIGIBLE**).
> - There is no FRESH-PRIMARY and no FRESH-SECONDARY candidate.
> - Per DP2 §12 the following were not done: no external split, no model training (seeds 43 / 44), no compute baseline,
>   no preregistration and no evaluation of any kind.
>
> Machine-readable records in `artifacts/dp2_external/`:
> - `dataset_inventory.json`, `dataset_provenance.json`, `dataset_eligibility.json`, `primary_dataset_selection.json`;
> - `table_dataset_inventory.csv`, `figure_dataset_audit.png`.
>
> Reproduce with `scripts/dp2_dataset_audit.py` (metadata only).

## 1. Method

- **Frozen model check first:** DualReadout-ECG S1 MIDDLE reproduces DP0 exactly. It has 943,372 waveform parameters
  (246,720 shared, 8,320 adapter); the separate specialists have 1,191,910, so the saving is 20.85 %. The DP0
  lock-freeze hashes are intact, and the seed-42 detector / P / G / S1 checkpoint hashes match.
- **Discovery:** path and name discovery only.

  | root | result |
  |---|---|
  | `/data`, `/datasets`, `/scratch`, `/workspace` | absent |
  | `/mnt` | empty |
  | `/opt` | no physiological data |
  | `/home` | only `/home/kwy00` is readable; searched with name and extension patterns (ppg, ecg, mimic, vitaldb, bidmc, capno, wesad, dalia, wildppg, abp, pleth, physionet, .hea, .edf, .vital) |

- **Excluded paths:**
  - by standing project rules: `/home/kwy00/sci` and `/home/kwy00/taeho` (another user's project);
  - environment, caches and package directories.
- **Metadata-only reads:**
  - WFDB / CSV header lines;
  - project-generated inventories (`dalia_raw_inventory.json`, WildPPG `INVENTORY.json`, the V0 VitalDB audit,
    split manifests);
  - `.npy` arrays opened with `mmap_mode="r"` for their shape;
  - MATLAB v7.3 files opened with h5py for dataset shapes.
  - Not opened at all: pickles, WFDB `.dat` signal files, VitalDB case arrays and the old VitalDB V1 TEST.
- **Prior use:** git history (268 commits), `docs/`, `artifacts/`, `scripts/` and split manifests were searched for
  every dataset alias. Contamination is recorded only with evidence (a commit and / or record).
- **Labels:** exactly one per dataset, with the frozen precedence INELIGIBLE (structural) > CONTAMINATED (prior
  outcome use) > PROVENANCE-UNCERTAIN > FRESH.

## 2. Inventory

| dataset | path | subjects | records | PPG / ECG | fs PPG / ECG (Hz) | synchronisation | subject IDs | prior project use | label |
|---|---|---|---|---|---|---|---|---|---|
| PPG-DaLiA | `data/raw/PPG-DaLiA` (25.1 GiB) | 15 | 15 | wrist BVP / chest ECG | 64 / 700 | UNCERTAIN (second-level only, A0) | yes | training, development, final reporting, hypothesis formation | **CONTAMINATED** |
| WESAD | `data/raw/WESAD` (16.4 GiB) | 15 | 15 | wrist BVP / chest ECG | 64 / 700 | UNCERTAIN (same device pair as DaLiA) | yes | training, final reporting | **CONTAMINATED** |
| WildPPG | `data/raw/WildPPG` (18.3 GiB) | 16 | 16 | 4 sites × 3 λ / sternum ECG | 128 / 128 | yes (per-site device clock) | yes | training, development, final reporting, hypothesis formation | **CONTAMINATED** |
| BIDMC | `data/raw/BIDMC` (0.34 GiB) | 53 | 53 × 480 s | PLETH / II (all 53), V, AVR | 125 / 125 | yes (WFDB multi-signal) | yes | training, final reporting | **CONTAMINATED** |
| CapnoBase | `data/raw/CapnoBase` (0.13 GiB) | 42 | 42 × 480 s | pleth / ecg (lead UNKNOWN) | 300 / 300 | yes (common clock) | yes | training, final reporting | **CONTAMINATED** |
| MIMIC-BP | `data/raw/MIMIC-BP` (6.0 GiB) | 1,524 | 45,720 × 30 s | ppg / ecg (lead UNKNOWN), abp, resp | 125 / 125 | yes (curated: f0 match ≤ 0.3 Hz, PAT consistency) | yes (pNNNNNN) | training, development, final reporting, hypothesis formation, model / loss selection (PPG → ABP) | **CONTAMINATED** |
| UCI-BP | `data/raw/UCI-BP` (6.3 GiB) | UNKNOWN | 12,000 | PPG / ECG / ABP | 125 / 125 | UNCERTAIN | **no** | training, final reporting | **INELIGIBLE** |
| VitalDB | `data/raw/VitalDB` (26.1 GiB) | 5,782 eligible | 6,156 cases | PLETH / ECG_II | 500 / 500 | yes (one clock) | yes | all DP0 training / development / lock; old V1 TEST NOT FRESH | **CONTAMINATED** |
| VitalDB residual (V0-excluded) | `data/raw/VitalDB/cases` | 113 (9 also in V1) | 113 cases | PLETH / ECG_II | 500 / 500 | yes | yes | referenced only (V0 eligibility audit) | **INELIGIBLE** |
| MIT-BIH Arrhythmia | `/home/kwy00/DESTINATION` (0.10 GiB) | UNKNOWN | 48 | **no PPG** / MLII, V1–V5 | — / 360 | — | no | never referenced | **INELIGIBLE** |
| MIMIC-AFib | `data/raw/MIMIC-AFib` | UNKNOWN | 0 waveforms | header index only (20,067 rows) | — | — | — | referenced only | **INELIGIBLE** |

**Not separate datasets:**
- `data/processed/*` (24 directories) and `data/raw_u1_wildppg_view` are derived copies of the raw datasets above.
- `data/pretrained/{imf, ppgflowecg, rddm}` are model weights.

## 3. Prior use evidence (verified commits / records)

- **PPG-DaLiA:**
  - `5e0aa35` A0 PENGUIN reproduction (train 13, val S11, test S2);
  - A3 one-step recovery on subject S1 (`ce13126`);
  - `150669b` D1;
  - `c753d13` U2;
  - `542e6be` P1 ("DaLiA samples disagree on beat positions");
  - `c3d672d` MC1 (15 subjects, 4 test).
- **WESAD:**
  - `8d53f65` D3 (respiration target);
  - `2842c11` U1;
  - `8bd682b` "U2: WESAD evaluated — non-inferior at k=1, but only just".
- **WildPPG:**
  - `bae0142` A4 (test kjd, ssx; val an0, k2s);
  - `3f93a4d` A9;
  - `ec7f1d4` WP1 4-fold;
  - `74858d0` ED2;
  - `d22d2ae` WD1;
  - `504395d` the BF0 design document cites "WildPPG에서 같은 구조(R1) 0.62" as design evidence.
- **BIDMC:**
  - `150669b` D1 (36 train / 8 test);
  - `8d53f65` D3;
  - `c3d672d` MC1 (51 subjects, 15 test);
  - `a4d66e5` EXP-D respiration.
- **CapnoBase:**
  - `150669b` D1;
  - `c3d672d` MC1 (42 subjects, 13 test).
- **MIMIC-BP:**
  - `fed5b8c` A7 PPG → ABP on the official split (train 1,100 / val 195 / test 229, i.e. all 1,524 subjects);
  - `9ff77b5` A8, an ABP target-scale decision made from A7 outcomes;
  - `e09ee6d` "U2: MIMIC-BP evaluated — and it exposes a defect in my own frozen gate";
  - `b1eb66a` EXP-D ABP.
  - The D1 preregistration excluded it from ECG work ("target is ABP, not ECG"). **Its ECG channel was never used, and
    no PPG → ECG outcome was ever computed on it.**
- **UCI-BP:** `8d53f65` D3, `2842c11` U1, and the U2 split manifest. It has no subject identifiers (A7 audit).
- **VitalDB:**
  - `080ec1c` V0;
  - `5d3d274` V1 (test 1,156 patients);
  - all DP0 populations (`724b883`);
  - old V1 TEST NOT FRESH (`17b43a6`).
- **MIT-BIH:** no reference in docs, scripts, src or artifacts.
- **MIMIC-AFib:**
  - `scripts/scan_mimic3wdb.py` (index scan, no waveform);
  - `docs/PREPROCESSING_CONVENTIONS_SURVEY.md`: "Do not use MIMIC-AFib for any comparative claim until a subject list
    is fixed".

## 4. Eligibility against the FRESH-PRIMARY criteria

- **Seven datasets fail criterion 7** (no prior outcome-based model / architecture / loss / threshold use in this
  project): PPG-DaLiA, WESAD, WildPPG, BIDMC, CapnoBase, MIMIC-BP and VitalDB. Each was used for training and final
  performance reporting in this project, and most also for hypothesis formation or selection.
- **Structural failures** (criteria 1–3):
  - UCI-BP has no subject IDs, so a patient-level split is impossible;
  - MIT-BIH has no PPG;
  - MIMIC-AFib has no waveforms;
  - the VitalDB residual is the same source, devices and institution as the DP0 development data (not a dataset shift)
    and failed the frozen V0 finite-fraction rule.
- **Closest to eligible: MIMIC-BP.**
  - It meets criteria 1–6 and 8–10: simultaneous PPG + ECG; 1,524 subject IDs; curated synchronisation; 125 Hz; no
    overlap with the VitalDB development population; PPG → ECG outcomes never computed.
  - Its ECG lead is UNKNOWN locally.
  - It fails criterion 7: all 1,524 subjects were used for outcome-based PPG → ABP model training, selection, reporting
    and target-representation decisions (A7, A8, U1, U2, EXP-D).
  - Under the frozen DP2 criteria it is therefore CONTAMINATED, not FRESH.
- **No FRESH-SECONDARY candidate exists either.** Every small dataset with simultaneous PPG + ECG (BIDMC, CapnoBase,
  PPG-DaLiA, WESAD, WildPPG) has prior outcome use.

## 5. Cross-dataset provenance

- **MIMIC (Beth Israel Deaconess ICU):** BIDMC (MIMIC-II matched), UCI-BP (MIMIC-II), MIMIC-BP (MIMIC-III matched) and
  the MIMIC-AFib index (MIMIC-III matched).
  - Patient overlap is possible: MIMIC-III waveforms span the MIMIC-II period, and no local mapping exists between
    sNNNNN and pNNNNNN identifiers.
  - They are never independent replications of each other.
- **Empatica E4 + RespiBAN wearables:** PPG-DaLiA and WESAD share the device pair and the research group. Subject
  overlap is UNKNOWN.
- **VitalDB (SNUH):** the DP0 development source. It shares no institution with the MIMIC or wearable datasets.
- **Independent single sources:** WildPPG (ETH Zurich), CapnoBase (UBC), and MIT-BIH (ECG only).

## 6. Consequence

- **DP2 NO ELIGIBLE PRIMARY DATASET.** No external split, no EXTERNAL-INTERFACE-CAL / LOCKED-TEST, no preprocessing
  interface, no new training, no compute baseline, no preregistration, no freeze and no model inference on any dataset.
- **DP0's status is unchanged:** CONFIRMED on AF-LOCK, which is an internal locked set and not project-naive.
- **The old VitalDB V1 TEST stays NOT FRESH** (DP1 audit) and was not opened.
