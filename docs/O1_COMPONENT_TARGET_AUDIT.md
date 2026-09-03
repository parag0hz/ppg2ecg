# O1 — ECG component target audit

Written **before** the preregistration is frozen and before any probe exists (commit order steps 2–3). It fixes
what the targets are, proves the four-site ECG identity that the whole design depends on, and records target
validity and variability. No probe, no generator, no test subject (`kjd`/`ssx` never loaded).

Base commit `801bf607` (clean tree). Artifacts: `artifacts/o1_component_extractability/`.

---

## 1. Four-site ECG identity (preregistration §3) — **ESTABLISHED**

WildPPG pairs four PPG sites (`sternum`, `head`, `wrist`, `ankle`) with **one** sternum ECG. If `(subject,
window_index)` did not identify a single ECG waveform, four PPG rows would be counted as four independent ECG
observations and every interval in O1 would be too narrow.

`scripts/o1_four_site_audit.py` checked all 14 non-test subjects over the **full** processed arrays (342,471
rows, 85,830 distinct `window_index` groups):

| check | result |
|---|---|
| groups with ≥ 2 site rows | 85,830 / 85,830 (every group is multi-site) |
| ECG waveform sha256 identical inside the group | **85,830 / 85,830 (100 %)** |
| deep check (R-peak train **and** all nine scalar targets identical), 512 groups per subject | **7,168 / 7,168 (100 %)** |
| group sizes | 4 site rows for almost every group; a group has 3 when one site's window was dropped by the build's non-finite/constant rule (an0: 1 group of 3; k2s: 399 groups of 3) |
| distinct ECG waveforms per subject | e.g. an0 5,546 distinct for 22,183 rows (= rows / ≈ 4) |

**Consequence, frozen here**: the unit of independent ECG observation is `(subject, window_index)`, not the npz
row. The O1 cluster bootstrap resamples ECG windows and carries all of that window's site rows together
(preregistration §17), and target variability is computed once per unique ECG window (§7).

## 2. Targets and their definitions (preregistration §4–§5)

Every target is derived with existing frozen project code. **No new ECG delineator was written for O1** and the
R-peak detector is not switched: it is `ppg2ecg.evaluation.rpeaks.detect_rpeaks` (neurokit `ecg_clean` +
`ecg_peaks`), the same call used by R1, M1 and Q1. Per-beat quantities are aggregated per 8 s window with the
**median** (preregistration §5), which is why single-beat detector or morphology outliers cannot dominate a
window's label.

| id | target | definition (frozen primitives in bold) | unit |
|---|---|---|---|
| T1 | `beat_count` | `len(`**`detect_rpeaks`**`(y, 128))` | beats / 8 s |
| T2 | `median_RR_ms` | median of `diff(peaks)/128·1000`; needs ≥ 2 peaks | ms |
| T3 | `RR_IQR_ms` | p75 − p25 of the same RR series; needs ≥ 3 peaks | ms |
| T4 | `median_QRS_p2p` | per valid beat `ptp(y[r−10 : r+11])`, median | normalised ECG amplitude |
| T5 | `median_QRS_energy` | per valid beat `Σ y[r−10 : r+11]²`, median | amplitude² (21 samples) |
| T6 | `median_QRS_max_abs_derivative` | per valid beat `max\|`**`M1.d1`**`(y[r−11 : r+12])\|`, median | amplitude / sample |
| T7 | `median_QRS_curvature_energy` | per valid beat `mean(`**`M1.d2`**`(y[r−11 : r+12])²)`, median | amplitude² / sample⁴ |
| T8 | `median_QRS_width_ms` | per beat **`rpeaks.qrs_width_ms`** (q-window 0.08 s, s-window 0.12 s), median over finite values | ms |
| T9 | `ECG_HF_fraction` | **`metrics.hf_energy_ratio`** (≥ 15 Hz power fraction, whole window) | fraction |

QRS-core geometry is **M1 verbatim**: `CORE = round(80 ms · 128) = 10` samples, so the core is `[r−10, r+10]`
(21 samples); a beat is valid iff `r − 10 − 1 ≥ 0` and `r + 10 + 2 ≤ 1024` (i.e. `r ∈ [11, 1011]`), exactly the
`m1_structural.qrs_core_morphology` rule; `d1` is `np.diff` (no `fs` scaling) and `d2` is the fixed second
difference `x[n+1] − 2x[n] + x[n−1]`, both taken on the 23-sample span `[r−11, r+12)`.

**T8 was resolvable** — `rpeaks.qrs_width_ms` is the exact width implementation already used by
`metrics.rhythm_morphology_metrics` (`qrs_width_err_ms`) throughout the project — so T8 stays in the primary
set. No new width heuristic was designed.

**Two definitional choices are recorded here, before any result:** (i) T7 is the *energy* of the frozen `d2`
(mean of squares), while the generator-side crosswalk metric `qrs_curvature_err` is `mean|d2_pred − d2_gt|`;
the two share the same `d2` but are not the same functional. (ii) T2/T3 use the median/IQR of the RR series
derived from the frozen peak train (the project's existing `hr_bpm` uses the *mean* RR, which is why it is not
reused for a median-RR label).

**Event timing is not a new target.** The timing row of the map reuses the existing frozen R1 diagnostic
(F1@50/100/150/200, RR MAE) and is labelled *R1 existing diagnostic*; no new event probe is trained.

## 3. Cohort (preregistration §2)

The frozen R1 cohort is rebuilt exactly (`r1_cohort.cohort_positions`, salt `r1-global-rhythm-observability-v1`,
2,048 windows per subject × site for train/dev, 1,024 for validation) — no new cohort is selected.

| role | subjects | rows | unique ECG windows |
|---|---|---|---|
| `probe_train` | fex, l38, n31, ngh, p5d, p9p, qm9, trh, tz8, w4p (10) | 81,920 | — |
| `internal_dev` | u7y, e61 | 16,384 | — |
| `validation` | an0, k2s | 8,192 | 8,192 rows over the validation cohort |
| total | 14 subjects | **106,496** | **65,251 unique ECG windows** |
| test | kjd, ssx | **never loaded** | — |

## 4. Target validity (preregistration §6)

All nine targets are valid on **100.0 %** of the validation ECG windows, so all nine enter the **PRIMARY** map;
none is `SECONDARY / INSUFFICIENT COVERAGE`. Missingness rules (no imputation is ever applied): T2 needs ≥ 2
R peaks, T3 needs ≥ 3, T4–T8 need at least one beat whose QRS core fits inside the window.

## 5. Target variability (preregistration §7) — computed once per unique ECG window

| id | target | train median | train IQR | validation median | validation IQR | total var | between-subject | within-subject | within / total |
|---|---|---|---|---|---|---|---|---|---|
| T1 | beat_count | 10.0 | 3.0 | 9.0 | 1.0 | 4.584 | 2.010 | 2.457 | 0.54 |
| T2 | median_RR_ms | 792.97 | 226.56 | 808.59 | 123.05 | 53,047 | 20,622 | 30,362 | 0.57 |
| T3 | RR_IQR_ms | 33.20 | 46.88 | 23.44 | 23.44 | 23,253 | 4,555 | 17,777 | 0.76 |
| T4 | median_QRS_p2p | 1.6448 | 0.5053 | 1.7180 | 0.2065 | 0.1588 | 0.0431 | 0.1110 | 0.70 |
| T5 | median_QRS_energy | 6.8733 | 5.8140 | 8.2039 | 2.4291 | 9.761 | 3.050 | 6.546 | 0.67 |
| T6 | median_QRS_max_abs_derivative | 0.7271 | 0.2300 | 0.7554 | 0.0923 | 0.03782 | 0.01064 | 0.02594 | 0.69 |
| T7 | median_QRS_curvature_energy | 0.05954 | 0.03380 | 0.06201 | 0.01439 | 6.963e−4 | 1.780e−4 | 4.947e−4 | 0.71 |
| T8 | median_QRS_width_ms | 62.5 | 31.25 | 62.5 | 7.81 | 503.9 | 142.4 | 345.0 | 0.68 |
| T9 | ECG_HF_fraction | 0.2081 | 0.0924 | 0.1966 | 0.0483 | 0.01078 | 0.005656 | 0.004695 | **0.44** |

Reading (recorded before any probe exists): every target carries real within-subject variation, so none is a
purely static subject constant — but **T9 is the most subject-dominated** (56 % of its variance is between
subjects) and **T3 the least** (76 % within). This is exactly why the preregistration requires the SS-SHUFFLE
control before any target may be called window-specifically extractable: a good score on a subject-dominated
target can be produced by recognising the subject rather than the window.

Validation IQRs are systematically narrower than train IQRs (e.g. T4 0.207 vs 0.505) because the validation
cohort is two subjects: the normalised MAE of §14 is scaled by the **train** IQR for every arm, so this does not
advantage or disadvantage any arm, but it does mean normalised errors near 1.0 are not "chance-level" for the
validation population — the B0/B1/B2 baselines, not the value 1.0, are the reference.

## 6. Train-only scaling (preregistration §8)

`y_z = (y − median_train) / IQR_train`, both statistics computed on the **probe-train subjects' unique ECG
windows only** (`target_scaling.json`). Every IQR is far from zero, so no target is stopped for
`INSUFFICIENT TARGET VARIATION`. Validation statistics are never used for scaling — asserted by test.

## 7. Artifacts written by this stage

`four_site_target_audit.csv`, `four_site_target_audit_deep.csv`, `four_site_target_audit.json`,
`cohort_manifest.csv`, `target_definitions.json`, `target_validity.csv`, `target_variability.csv`,
`target_scaling.json`, `target_build_provenance.json`, plus the local target cache `_cache_targets.npz`
(gitignored).
