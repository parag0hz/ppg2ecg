# Per-Dataset PPG→ECG Preprocessing Conventions — Literature Survey

Status: **survey only. No experiment, no preregistration, no code change to any pipeline.**
Date: 2026-09-04. Method: 30 agents, 6 axes (PPG-DaLiA / BIDMC / CapnoBase / VitalDB / MIMIC /
cross-paper), each axis crosschecked against primary sources (paper PDFs, released code at pinned
tags, dataset landing pages). 24/24 crosschecks resolved; 4 returned corrections, recorded below.

## 0. Correction to an earlier claim in this project

An earlier turn presented PPGFlowECG's MCMED processor (`data_process_to_npz/step1.py`) as *the*
canonical preprocessing pipeline. That was an over-generalization from a single repository. The
survey refutes it: that file is one paper's choice for one dataset, and several of its parameters
(50 Hz notch, `zhao2018` ECG SQI, 10-s windows) are not shared by any other work on the same data.

## 1. Headline finding: the axis of variation is the paper lineage, not the dataset

The natural hypothesis — "each dataset has its own house style" — is **not what the sources show**.
What a dataset actually fixes is small; what papers freely choose is large, and the choices track
*who cited whom*, not which corpus is being processed.

**Fixed by the dataset (genuinely differs per dataset):**

| | PPG native | ECG native | Records | Ships quality labels? | Notes |
|---|---|---|---|---|---|
| PPG-DaLiA | wrist E4 BVP **64 Hz** | chest RespiBAN **700 Hz** | 15 subj, ~36 h | no | ECG is **already band-passed 0.5–100 Hz on-device**; ships manually-corrected R-peaks + HR label (8 s / 2 s) |
| BIDMC | 125 Hz | 125 Hz, **Lead II only** | 53 subj × 8 min | no (breath annots only) | Lead II is mechanically forced — the `.mat` release ships nothing else |
| CapnoBase | 300 Hz | 300 Hz | 42 rec × 8 min | **yes** — expert artifact intervals + peak labels | the only one of the five with shipped artifact labels |
| VitalDB | SNUADC/PLETH **500 Hz** | SNUADC/ECG_II **500 Hz** | 6,156 cases w/ both | no | recorded at SNUH ⇒ mains is **60 Hz**, not 50 |
| MIMIC-III matched | 125 Hz | 125 Hz | 12,940 rec w/ PLETH+II | no | 598,163 h; AF labels come from an external annotation file |

**Chosen by the paper (does NOT track the dataset):** resample target, filter band, filter family and
order, window length, overlap, normalization stack and its scope, split, SQI. Every one of these
varies more *between papers on the same dataset* than *between datasets within one paper*.

Concretely: on **one dataset (BIDMC)** the published PPG band is 1–8 Hz (CardioGAN), 0.5–8 Hz (RDDM),
0.3–8 Hz (CLEP-GAN), and unstated (PPGFlowECG). Meanwhile **one paper (RDDM)** applies the *same*
0.5–8 Hz to DaLiA, BIDMC, CapnoBase and MIMIC alike. The dataset is not the variable.

## 2. The one real lineage — and where it stops

**CardioGAN (AAAI'21) → RDDM (AAAI'24) → PPGFlowECG (2025), copied by Performer, f-GAN, PENGUIN.**
This is the closest thing to a standard, and it is *inheritance*, not independent convergence:
PENGUIN's paper says verbatim "we followed protocols in prior works [11, 22, 39]".

Shared core (safe to call conventional):
1. resample everything to **128 Hz** — including MIMIC/BIDMC's native 125 Hz, a 1.024× non-integer
   upsample that buys nothing physiologically and exists so that 4 s = 512 samples for a U-Net;
2. **4 s / 512-sample** windows (2021–24 generation);
3. **subject-level 80/20** split;
4. two-stage normalization: z-score **then** min-max to [−1, 1];
5. **no signal-quality screening at all**;
6. heart-rate MAE (bpm) as the headline metric.

Where the lineage breaks — a **generational split at ~2025**: PPGFlowECG, PG-LRF, P2E-VQ and
Physiology-Aware Masked all move to **10-s windows**, drop the min-max stage for **z-score only**,
and two of the three newest papers state **no filter cutoffs whatsoever**.

## 3. Disagreements that matter more than the "convention"

**3.1 ECG filtering is the least standardized choice in the field, and the variants are not minor.**

| Source | ECG band | Consequence |
|---|---|---|
| CardioGAN, f-GAN | FIR 3–45 Hz (order `int(0.3·fs)` = 38) | 3 Hz high-pass destroys ST/T morphology |
| RDDM **paper**, PPGFlowECG, PENGUIN, **ours** | 0.5 Hz high-pass only | diagnostic band retained |
| CLEP-GAN, "Beyond Single-Channel" | 0.4–45 Hz | — |
| Tang 2022 | Chebyshev-II 0.5–20 Hz | — |
| Uncertainty-Aware | 1–47 Hz | — |
| RDDM **released code** | **5–15 Hz, order 1, causal** | QRS-detection band; P and T waves are *removed*, then reconstructed and scored |

**3.2 Paper ≠ code, in the two most-cited repositories.**
- RDDM's paper states a 0.5 Hz ECG high-pass; `data.py` calls
  `nk.ecg_clean(..., method='pantompkins1985')`, which NeuroKit2 implements as a **1st-order
  Butterworth band-pass 5–15 Hz** applied with `sosfilt` (forward-only ⇒ phase distortion on top).
  These are not variants of one another.
- RDDM's paper states subject-specific z-score → min-max → segment. The code does
  `minmax_scale(X, (-1,1), axis=1)` on already-segmented arrays and **omits the z-score entirely**;
  the subject-specific stage lives in an unreleased script.
- CardioGAN's released `preprocessing.py` contains only two filter functions — no resampling, no
  segmentation, no normalization. The published pipeline is not reproducible from the repo alone.

**3.3 Reported filter orders are wrong in the secondary literature.**
`nk.ppg_clean` (NeuroKit2 default `elgendi`) is Butterworth 0.5–8 Hz at **order 3** for v0.2.4
(2023-04) through v0.2.11 (2025-05), and order 2 only from v0.2.12 (2025-07). RDDM (Aug 2023,
unpinned deps) therefore used **order 3**, applied zero-phase via `sosfiltfilt` ⇒ effective
magnitude order 6. Any claim of "order 2" for RDDM is anachronistic.

**3.4 VitalDB's notch frequency is wrong in the literature.**
PPGFlowECG's code hard-codes `freqs=50` and AnyPPG states "a 50 Hz notch filter". VitalDB was
recorded at Seoul National University Hospital, where **mains is 60 Hz**. SPOTR is the only paper
found that handles this correctly ("50/60 Hz depending on the acquisition region").

**3.5 "MIMIC-AFib" has five irreconcilable cohort sizes.**
RDDM and everything downstream: 35 subjects (19 AF). The only released annotation file (figshare
batch1): 45 subjects (23 AF / 22 non-AF). Bashar 2019 IEEE Access text: N = 60 (35 NSR + 25 AF);
its Tables 2+3 list 50. The figshare and IEEE Access lists overlap on 43 subjects with **zero label
conflicts**, but 7 are IEEE-only (NSR 7136, 8167, 8674; AF 57964, 63773, 70854, 85163) and 2 are
figshare-only. Worse, the citation chain is broken at the root: RDDM and PPGFlowECG both cite
Bashar et al. 2019 *"Noise detection in electrocardiogram signals…"* for MIMIC-AFib, but that paper
is **ECG-only** and defines no PPG pipeline. **Consequence: "MIMIC-AFib" results are not comparable
across papers, and the RDDM 35-subject cohort cannot be reconstructed from any public artifact.**

**3.6 "Fréchet Distance" means two different things.**
CardioGAN's paper defines the *discrete Fréchet distance* (min over order-preserving pairings of the
max Euclidean distance). RDDM's `metrics.py::calculate_FD` computes a **FID-style** Fréchet distance
between Gaussians fitted to feature activations. The numbers are not comparable; neither paper flags it.

**3.7 Powerline handling is absent field-wide.** Of 15 method papers, exactly **one**
(Physiology-Aware Masked) applies an explicit powerline filter. Everyone else either relies on a
45/47 Hz low-pass to subsume it, or does no ECG filtering at all. It cannot be inherited.

## 4. SQI: there is no convention, and the shipped labels go unused

| Work | Screening |
|---|---|
| CardioGAN, RDDM, Performer, f-GAN, PENGUIN, **ours** | **none** |
| CLEP-GAN, "Beyond Single-Channel" | manual hand-picking (34 / 30 "low-noise" records) |
| Zhu 2019 | the **only** user of CapnoBase's shipped expert artifact labels |
| PaPaGei, AnyPPG | flatline detection only (drop if >25% flat) |
| PulseDB | 4-stage rejection (saturation/flatline, HR plausibility, …) applied *before distribution* |
| PPGFlowECG | PPG template-matching SQI + `nk.ecg_quality(zhao2018)`, intersected |
| Pimentel 2016 (BIDMC, RR task) | explicit thresholded SQI — the *only* thresholded SQI in the BIDMC line |

**Correction found by crosscheck (CapnoBase):** the shipped peak labels are **not** screened against
the shipped artifact intervals. 177 PPG peaks and 7 ECG peaks fall strictly inside artifact
intervals, affecting 19/19 PPG-artifact cases and 4/9 ECG-artifact cases (worst: case 0031 with 50
PPG peaks; case 0035 has 15 in a single interval). Any consumer must intersect the two label sets
itself — assuming pre-screening leaks artifact-corrupted beats into both training and evaluation.

## 5. Where our pipeline sits

Ours (`src/ppg2ecg/data/preprocess.py`, PENGUIN-faithful): 128 Hz, per-window z-score → min-max
[−1,1], PPG Butterworth order 4 **0.5–4 Hz**, ECG Butterworth order 4 **0.5 Hz high-pass**, no notch,
no SQI, non-overlapping windows.

| Choice | Ours | Field | Assessment |
|---|---|---|---|
| 128 Hz | ✓ | dominant | conventional |
| z-score → min-max [−1,1] | ✓ | CardioGAN lineage | conventional |
| ECG 0.5 Hz HP only | ✓ | RDDM paper / PPGFlowECG / PENGUIN | conventional |
| no SQI | ✓ | CardioGAN, RDDM, Performer, f-GAN, PENGUIN | conventional **in this lineage**; PulseDB/PaPaGei/PPGFlowECG all screen |
| no notch | ✓ | 14/15 papers | conventional, and DaLiA's on-device 0.5–100 Hz + 700→128 Hz decimation makes it moot |
| per-window z-score scope | ✓ | PENGUIN code ✓; CardioGAN/RDDM papers use **subject-level** | **a real divergence** |
| **PPG 0.5–4 Hz** | ✓ | CardioGAN 1–8, RDDM 0.5–8, CLEP-GAN 0.3–8, PaPaGei 0.5–12, PPGFlowECG 0.5–8 | **we are the narrowest published band. 4 Hz is inherited from PENGUIN and is an outlier: every other PPG→ECG work keeps 8 Hz** |
| segment first, then filter per-window with `filtfilt` | ✓ (PENGUIN) | CardioGAN/RDDM/U-Net filter the **continuous** signal and segment last | **a real divergence** — order-4 `filtfilt` at 0.5 Hz over 512 samples produces per-window edge transients |

Two items are outliers rather than conventions: **the 4 Hz PPG upper cutoff** and **per-window
filtering after segmentation**. Both are inherited from PENGUIN, not independently justified here.

### The 8-second window
Our 8 s is not the reconstruction convention (4 s in 2021–24, 10 s in 2025+). It matches the
*HR-estimation* convention — Reiss et al. use 8 s / 2 s shift for DaLiA HR ground truth, and PaPaGei
uses 8 s / 75% on DaLiA. That is a defensible reason, but it means our window length is comparable
to the HR literature and **not** to the PPG→ECG reconstruction literature.

## 6. What this means for comparability

No cross-paper number in this field is comparable without restating the pipeline. Specifically:
- an HR-MAE comparison against PENGUIN is only meaningful at matched NFE and matched window length;
- an RMSE comparison against RDDM is meaningless unless one states whether RDDM's 5–15 Hz code path
  or its 0.5 Hz paper path produced the reference ECG;
- a Fréchet Distance comparison must state which of the two definitions is used;
- any "MIMIC-AFib" number is uninterpretable without the subject list.

## 7. Data acquired (this session)

| Dataset | Result | Size |
|---|---|---|
| VitalDB | **6,156 / 6,156 cases**, 0 fail — SNUADC/PLETH + SNUADC/ECG_II, raw 500 Hz float32, one npz per case | 27 GB |
| BIDMC | complete PhysioNet record set (53 subjects), 473 files | 206 MB |
| CapnoBase | **42 / 42 `.mat`** (bulk zip truncated at 53/133 MB; re-fetched per file via the Dataverse API) | 134 MB |
| MIMIC-III matched | index scan of 20,067 records: **12,940 carry both PLETH and II**, 598,163 h | index CSV only |

MIMIC waveforms are **not** downloaded: 2-channel 16-bit raw for the full PLETH+II set is ~1.08 TB.
A subject subset must be chosen first.

**Blocked:** (a) the MIMIC-AFib subject list — see §3.5, no public artifact reproduces RDDM's 35;
(b) MC-MED — PhysioNet credentialed access, requires CITI training + a signed DUA.

## 8. Recommendations (not implemented)

Ordered by expected information per unit of cost. None of these is started; each dataset addition
and each preprocessing change requires its own preregistration.

1. **PPG band ablation 0.5–4 vs 0.5–8 Hz on DaLiA, generator frozen where possible.** The single
   largest unjustified divergence from the field. Cheap, and directly tests whether the 4 Hz ceiling
   removes dicrotic-notch energy that beat placement depends on.
2. **Filtering order-of-operations control**: continuous-then-segment vs our segment-then-filter.
   Isolates the per-window `filtfilt` edge transient.
3. **Report at matched inference budget.** Our A4 result (HR MAE 9.43 bpm, better than PENGUIN's
   published 12.97, while R-peak F1@50 is 0.440) is currently NFE 4 against PENGUIN's 50 and 4.57 M
   params against 62.53 M. The HR metric is blind to the beat-level failure; publishing both, at
   matched NFE, is the honest presentation.
4. **SQI is a *reporting* decision before it is a *method* decision.** Our lineage does not screen,
   so adding SQI would make our numbers less comparable to PENGUIN/RDDM, not more. If added, report
   both screened and unscreened, with the discard fraction.
5. **CapnoBase is the highest-value new dataset** — it is the only corpus with expert artifact
   labels, which lets the SQI question be answered against ground truth rather than a heuristic.
   Note §4: peak labels must be intersected with artifact intervals manually.
6. **Do not use MIMIC-AFib for any comparative claim** until a subject list is fixed and published.
