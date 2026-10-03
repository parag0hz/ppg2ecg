# DP5 — MESA Sleep project-naive external cohort — DATA AUDIT (documentation only, before access)

**Status (2026-10-03):** written before any MESA file was requested, downloaded or read.
- Everything below comes from the official NSRR documentation pages. They were read on 2026-10-03;
  sleepdata.org has a certificate-chain error, so the pages were fetched with `curl -k`.
- **No MESA data exists on this machine:** `data/raw/` has no MESA, NSRR or sleepdata entry.
- **Access request:** the user submitted the NSRR Data Access and Use Agreement on 2026-10-03; it was pending at
  commit time.
- **No prior MESA use:** a repository-wide search for `mesa`, `nsrr` and `sleepdata` in `*.py`, `*.md`, `*.json` and
  `*.yaml` (excluding `external/` and `obsidian_vault/`) found no use of MESA by any study.

## 1. Why MESA (decision 2026-10-03, by the user)

Among the candidates of the 2026-10-03 survey (subagent report; CardioWave-Portable, Aurora-BP, MESA, MC-MED, MOVER),
MESA was chosen as the single primary external cohort before any access.
- **Recording:** finger PPG and ECG are recorded by one recorder into one file. This is the closest sensor setup to
  VitalDB's finger PLETH, and synchronization risk is low.
- **Size:** 2,056 participants with raw polysomnography.
- **No shared source:** no institution or participant is shared with VitalDB, MIMIC or any other project dataset.

Only one external cohort is used, so no external dataset is chosen after seeing results.

## 2. Facts from the official documentation

| claim | quote | source |
|---|---|---|
| Raw PSG size | "Raw polysomnography data are available for 2,056 MESA Sleep participants." | [MESA PSG introduction](https://sleepdata.org/datasets/mesa/pages/polysomnography-introduction.md) |
| One file per recording | "Each recording has a signal file (.EDF) and two versions of the event scoring and epoch staging annotations (.XML)." | same |
| Recorder | "In-home polysomnography (PSG) was conducted using the Compumedics Somte System" | same |
| Finger PPG | "...ECG; leg movements, and finger pulse oximetry." | same |
| ECG channel | montage row "ECG \| EKG \| ECG \| - \| 256 \| - \| Ag/AgCl patch" (channel, EDF label, input 1, input 2, Hz, hardware filter, sensor) | [MESA montage](https://sleepdata.org/datasets/mesa/pages/equipment/montage-and-sampling-rate-information.md) |
| PPG channel | montage row "Plethysmography \| Pleth \| Pleth \| - \| 256 \| - \| Nonin 8000 sensor" | same |
| Montage may vary per study | "There may be a small proportion of studies and signals that do not match these standards exactly. Please review the settings at the individual sleep study level" | same |
| Subject identifier | "Datasets and raw data files are keyed on the identifier ( mesaid )" | [MESA dataset introduction](https://sleepdata.org/datasets/mesa/pages/dataset-introduction.md) |
| Population | "6,814 black, white, Hispanic, and Chinese-American men and women initially ages 45-84 at baseline in 2000-2002 ... Between 2010-2012, 2,237 participants also were enrolled in a Sleep Exam" | [NSRR MESA](https://sleepdata.org/datasets/mesa) |
| Known file issues | subjects 1738 and 6476 have "data loss or corruption"; the EDF of 0718 was re-exported in October 2020 | MESA PSG introduction, "Known issues" and changelog |
| Access | "access is only granted to individuals who have completed the web-based Data Access and Use Agreement (DAUA). Each DAUA submission is reviewed by the NSRR Review Committee" | [NSRR data security](https://sleepdata.org/about/data-security) |
| Review time | "After submission, please allow up to two (2) weeks for your request to be reviewed by the NSRR team." | [NSRR data request](https://sleepdata.org/data/requests/mesa/start) |
| Agreement term | "NSRR DAUAs expire 3 years from the date access is granted" | NSRR data security |
| Required citations | Zhang et al., JAMIA 2018 (doi:10.1093/jamia/ocy064); Chen et al., Sleep 2015 (doi:10.5665/sleep.4732); plus the acknowledgement text on the MESA page | NSRR MESA |
| Repository notice | banner: "This repository is under review for potential modification in compliance with Administration directives." | every sleepdata.org page |

## 3. Not established from documentation (to be checked by the header audit, Stage A)

- **ECG lead and electrode placement:** the montage gives only "ECG" (single input, Ag/AgCl patch). The lead is
  recorded as UNKNOWN, as for MIMIC-BP in DP3.
- **Per-file channel labels and sample rates:** the montage reflects the project start, and some files may differ.
- **Recording durations and the number of readable files:** to be confirmed.
- **Total download size:** not stated on the pages read. There is 1.4 TB free on this machine.
- **Pleth timing:** any internal processing delay of the Nonin 8000 pleth output relative to the ECG is not stated. The
  DP3 detector collapse (F1 0.047 on MIMIC-BP) is why the draft preregistration calibrates the timing front-end on a
  disjoint MESA split (decision D1).

## 4. Domain differences (disclosed, not corrected)

| | VitalDB (training) | MESA |
|---|---|---|
| setting | intra-operative, anaesthetized surgical patients | in-home, unattended overnight sleep |
| population | Korean surgical patients | US multi-ethnic community cohort; baseline ages 45–84 in 2000–2002, sleep exam 2010–2012 (so roughly 55–95 at the exam; derived, not stated) |
| PPG | SNUADC PLETH, 500 Hz | Nonin 8000 finger sensor via Compumedics Somte, 256 Hz |
| ECG | lead II, 500 Hz | single channel, lead not documented, 256 Hz |

## 5. Data handling

- **Storage:** `data/raw/MESA/`, which `.gitignore` already covers (checked with `git check-ignore`). Nothing from MESA is
  committed: no data, no per-subject values, no identifiers beyond counts and hashes.
- **Restrictions:** the DAUA terms (secure storage, no redistribution) apply. The 3-year expiry is recorded here.

## 6. Stage A — header-only audit (after access; before the preregistration is frozen)

**Read:**
- EDF fixed headers and signal headers only: label, transducer, physical dimension, prefiltering, samples per record,
  number of records and record duration.
- Signal samples are not read; the reader stops at the end of the header.

**Output:**
- `artifacts/dp5_mesa/header_audit.json`, with counts only:
  - files;
  - files with exactly one EKG and one Pleth channel at 256 Hz;
  - durations;
  - prefilter strings;
  - unreadable files.
- The eligibility list (mesaid hashes).

**Decision allowed from Stage A:** only the channel-name and eligibility rules in §3 of the preregistration, written as
a dated amendment before the freeze. Nothing about performance can be learned from headers.

## Appendix — draft text for the NSRR request form (for the user to adapt)

> **Intended use:** External validation of a pre-trained, frozen deep-learning model that reconstructs a single-lead ECG
> waveform and draws ECG samples from the finger photoplethysmogram, together with a beat-timing detector.
> - **Data used:** MESA polysomnography EDF files, Pleth and EKG channels only. No covariates, no linkage to other data
>   sources.
> - **Protocol:** analysis follows a preregistered protocol, frozen before data access. Results are reported in
>   aggregate only.
> - **Data handling:** data are stored on a secured, access-controlled workstation and never redistributed.
