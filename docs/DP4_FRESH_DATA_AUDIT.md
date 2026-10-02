# DP4 — project-naive data audit

> **DP4 NO PROJECT-NAIVE PRIMARY DATASET — HARD STOP.**
> - No dataset on this machine satisfies the DP4 hard requirements for a project-naive external PPG + ECG cohort.
> - Nothing new arrived since the DP2 audit.
> - MIMIC-BP is now also contaminated for PPG → ECG, by DP3.
> - No dataset was downloaded, no waveform value was read, no model was run, and no DP4 preregistration or freeze was
>   created.
>
> Record: `artifacts/dp4_external/dataset_audit.json`.

## Discovery (metadata only)

- **Since the DP2 inventory (2026-10-02 18:37):**
  - no new mount: only `/` is mounted on nvme0n1p2; `/media/kwy00`, `/mnt` and `/srv` are empty; `/data`,
    `/datasets`, `/scratch` and `/workspace` are absent;
  - no new data file larger than 1 MB in readable home directories;
  - no change under `data/raw`.
- **Excluded by standing project rules:** `/home/kwy00/sci` and `/home/kwy00/taeho`.

## Candidates

| dataset | subjects | PPG / ECG | prior project use | prior target use | label |
|---|---|---|---|---|---|
| PPG-DaLiA | 15 | yes / yes | training, outcomes, hypotheses | ECG | CONTAMINATED |
| WESAD | 15 | yes / yes | training, outcomes | respiration | CONTAMINATED |
| WildPPG | 16 | yes / yes | training, outcomes, hypotheses | ECG | CONTAMINATED |
| BIDMC | 53 | yes / yes | training, outcomes | ECG | CONTAMINATED |
| CapnoBase | 42 | yes / yes | training, outcomes | ECG | CONTAMINATED |
| MIMIC-BP | 1,524 | yes / yes | DP3 PPG → ECG outcomes on all subjects; earlier PPG → ABP | ECG, ABP | CONTAMINATED |
| VitalDB | 5,782 | yes / yes | all DP0 cohorts; old V1 TEST not fresh | ECG | CONTAMINATED |
| UCI-BP | unknown | yes / yes | training, outcomes | ABP | INELIGIBLE (no subject IDs) |
| VitalDB residual (V0-excluded) | 104 new + 9 overlapping | yes / yes | V0 eligibility audit only | none | INELIGIBLE for external confirmation |
| MIT-BIH | unknown | no / yes | none | none | INELIGIBLE (no PPG) |
| MIMIC-AFib index | — | no waveforms | referenced only | none | INELIGIBLE |

### The one subject-level project-naive group

The 104 VitalDB residual subjects have never been windowed or modelled. They still do not qualify:
- **Not an external domain shift:** they share the institution, devices, acquisition and preprocessing with every
  development cohort. DP4 asks about specialist preservation under external domain shift, and lists VitalDB among the
  datasets it must not fall back to.
- **Quality-excluded:** they were removed by the frozen V0 finite-fraction rule (median finite fraction 0.84).
- **Partial overlap:** 9 further residual subjects own V1 cases.

## Consequence

- **DP4 stops here.** A DP4 primary cohort would have to be newly acquired, by the user's choice of dataset; that is
  outside this audit.
- **Earlier conclusions are unchanged:** DP0 (internal, AF-LOCK) and DP3 (ECG-target-blind cross-task, primary FAILED,
  ROBUST-2/3).
