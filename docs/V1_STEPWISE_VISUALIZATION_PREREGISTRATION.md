# V1 — All-Subject Stepwise Waveform Visualization + ECG→PPG Event-Delay Audit — PROTOCOL

**Status:** frozen at this commit, pushed **before any V1 prediction, figure or delay statistic exists**.
**Type:** frozen-checkpoint forward inference, plotting, and a CPU-only physiological timing audit.

**NO TRAINING. NO NEW MODEL. NO ATTENTION IMPLEMENTATION. NO LOSS CHANGE. NO TEST ACCESS.
NO C2 TRAINING.**

---

## 1. Purpose

Two things, kept separate throughout:

1. **A systematic visualization** of what the frozen iMeanFlow produces per subject, PPG site and NFE.
2. **A physiological timing audit** of the ECG-R → PPG-pulse delay, asking whether PPG pulse timing could
   supply an **inference-safe** timing prior for ECG R-event location.

The audit is a feasibility question, **not** a model evaluation and **not** a method.

## 2. Provenance

Start HEAD `5154c170d8e86343530c43681ce168ec641cbcd8`; submodules PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`.
**C2 remains deferred with zero weight updates** (verified: no `outputs/*c2*`, no seed 40/41/43/44 run).

**Model — the frozen C1 baseline replay, nothing else:**

| | |
|---|---|
| path | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` |
| file sha256 | `557c70541f5cdd07819a3da04bb53477ac98827285507380` |
| `state_dict` sha256 | `47d7ccb94e5dbf7190d777f852b18f107f3ce2628d160b5e` |
| training round | 45 (best), arm `B` |
| selection metric | 0.11945885431656277 |

Nothing is overwritten.

## 3. Subjects — all non-test

**TRAIN (12):** `e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p` · **VAL (2):** `an0 k2s`.
**`kjd` and `ssx` are never loaded.**

Every figure, table and dashboard card carries **subject, split and PPG site**. Train-subject results are
**never** interpreted as generalization evidence and are never visually conflated with validation; every
aggregate either separates the splits or labels them explicitly. Qualitative generalization discussion uses
`an0`/`k2s` only; train subjects serve behaviour audit, site analysis and delay estimation.

## 4. Sites

`sternum`, `head`, `wrist`, `ankle`. Every subject carries all four (5,500–6,900 windows per subject×site).
Missing or unusable windows are skipped and the reason recorded in the manifest. **Site-resolved results are
always preserved before any pooling.**

## 5. Cohorts — metadata-only, nested, frozen before any prediction

One SHA256 rank, salt **`v1-all-subject-stepwise-visualization`**, key
`"{salt}|{subject}|{site}|{original_window_index}"`, smallest-hash-first **within each subject × site**.
Because the three cohorts are prefixes of the same ranking they are **nested**:

| cohort | per subject×site | total | use |
|---|---:|---:|---|
| **VIZ** | 8 | **448** | per-window stepwise figures and R-centred zooms |
| **METRICS** | 32 | **1,792** | NFE structural metrics (superset of VIZ) |
| **DELAY** | 128 | **≤ 7,168** | ECG→PPG delay audit, CPU only, no model (superset of METRICS) |

Selection may **not** use model error, F1, R-peaks, PPG peaks, morphology, amplitude or visual quality.
Written to `cohort_manifest.csv`.

## 6. NFE and source

Primary **NFE 1, 2, 4, 8**; secondary reference **NFE 50** (uniform schedules, `ER.UNIFORM[n]`).
**Evaluation Gaussian source seed 0, and for a given window the identical source tensor is used at every
NFE** — `e_NFE1 = e_NFE2 = e_NFE4 = e_NFE8 = e_NFE50` — so that nothing but the step count varies.
Realised step counts are asserted equal to the requested NFE.

## 7. Window figure

One high-resolution figure per VIZ window, **time axis exactly shared across panels**. Rows: PPG input ·
GT ECG · NFE 1 · NFE 2 · NFE 4 · NFE 8 · NFE 50.

**Predictions are NOT independently normalised for plotting**; the actual evaluation representation is
shown, on a common y-scale within the figure. Any additional visual scaling appears only as a separate,
clearly-labelled secondary figure.

## 8. Event markers

GT R-peaks from the frozen `rpeaks.detect_rpeaks` at its existing configuration, drawn as **vertical dashed
reference lines**. PPG systolic peaks detected independently (`s1_audit.dsp_ppg_peaks`, library defaults,
no tuning), drawn as a separate marker.

**R-peaks are visualisation and coordinate reference only. Nothing — prediction, PPG or ECG — is ever
shifted. No oracle alignment anywhere.**

## 9. ECG-R → PPG-peak delay audit

For each GT R-peak, the **first subsequent** PPG systolic peak with **80 ms ≤ delay ≤ 800 ms**, matched
**one-to-one** (a PPG peak is never reused for two R-peaks; a greedy forward scan in time order enforces
this). Beats whose search window would leave the 8 s window, and unmatched beats, are excluded and counted.

`delta_peak = t_PPG_peak − t_R`, in ms. Written to `r_to_ppg_peak_delays.csv` with columns
`subject, split, site, window_index, r_sample, ppg_peak_sample, delay_samples, delay_ms, preceding_RR_ms,
estimated_HR`.

Per subject × site: n matched, coverage, median, mean, SD, IQR, MAD, p5, p95. Distributions by subject, by
site, subject×site median and IQR heatmaps, delay vs HR, delay vs preceding RR.
**A fixed delay is not assumed — determining whether one is valid is the point.**

## 10. PPG foot — SECONDARY, defined here before any result

For each detected systolic peak, search **backward within a fixed 400 ms window** (51 samples at 128 Hz)
and take the **argmin of the PPG** in that region as the onset proxy, subject to: the foot must lie strictly
before the peak, and after the previous detected systolic peak. Beats with no valid region are skipped.
**This rule is not tuned after seeing results.** If it proves unstable — defined in advance as **> 20 % of
otherwise-matched beats failing the constraints** — the PPG-foot analysis is **omitted rather than
repaired**. Where computed, `delta_foot = t_PPG_foot − t_R` and its within-subject/site variance are
compared against `delta_peak`.

## 11. Event-timing condition feasibility — no model is trained

Per subject/site: `CV_delay`, `IQR_delay`, within-window delay variation, between-subject variation,
between-site variation.

**Fixed-delay estimator, evaluated on validation subjects using TRAIN-ONLY statistics.** A validation
subject's delay is **never** estimated from its own ECG.

- **A — global**: one train-only median delay over all 12 train subjects.
- **B — site-specific**: train-only median delay per site.
- **C — HR-conditioned**: preceding-RR terciles with **edges computed on train only**; the train-only
  median delay within each tercile. Defined here, before any validation performance is inspected.

Predicted R location = `t_PPG_peak − delay`. For every validation GT R-peak, the absolute distance to the
**nearest** predicted R location; reported as MAE, median AE, and coverage at **25 / 50 / 100 / 150 ms**.
The count of predicted R locations against GT beats is reported alongside, so over-production is visible.

## 12. Stepwise structural metrics

On the METRICS cohort, per NFE: raw RMSE · raw correlation · QRS fixed-coordinate RMSE · QRS-energy
deviation · p2p deviation · derivative RMSE · curvature error · F1 · F1 excess (S1 count-matched
random-phase floor) · beats-ratio deviation. Summarised over `1 → 2 → 4 → 8 → 50`, separately for **train**,
**val**, and **each site**.

**Prohibited:** oracle shift, oracle correlation, oracle QRS metrics, and prediction-aligned morphology as
primary evidence.

## 13. R-centred stepwise figure

Per VIZ window, GT-R-centred zooms over **−300 ms … +500 ms** (asymmetric on purpose: the PPG pulse follows
the R event). Rows: PPG on top, then GT ECG, NFE 1/2/4/8/50. `t_R = 0` marked; the PPG row marks the
detected PPG peak and its delay in ms. **No prediction alignment.**

## 14. Dashboard and figures

`artifacts/v1_stepwise_visualization/dashboard/index.html`: one card per subject with subject, split,
window count, the four sites and the median R→PPG delay per site; each links to `subject_<id>.html` with
four site sections carrying the stepwise figures, R-centred figures, delay distribution and NFE metric
table. **Relative local paths only, no external CDN.** Standalone PNGs are also written so nothing depends
on the HTML.

Group figures: `all_subject_nfe_metrics.png`, `all_subject_r_ppg_delay.png`, `site_delay_boxplot.png`,
`subject_site_delay_heatmap.png`, `delay_vs_hr.png`, `nfe_qrs_rmse_by_subject.png`,
`nfe_derivative_error_by_subject.png`, `nfe_f1_excess_by_subject.png`, `val_only_nfe_summary.png`.
**A cherry-picked "best examples" figure is forbidden.**

## 15. Interpretation of the delay

Permitted: *"PPG pulse events systematically follow ECG R events"*; *"the observed delay includes
electromechanical and vascular propagation components"*; *"a train-derived PPG-to-R timing prior may be
feasible."*

**Not permitted:** *"PPG peak uniquely determines R peak"*; *"the delay is pulse transit time."*
**R→PPG-peak delay is not pure PTT** and may not be called it, because it also contains the
electromechanical delay and the systolic rise time.

## 16. Feasibility verdict

Exactly one of: `FIXED PPG→R TIMING PRIOR PLAUSIBLE` · `SITE-AWARE TIMING PRIOR REQUIRED` ·
`ADAPTIVE DELAY MODEL REQUIRED` · `PPG PEAK TIMING TOO UNSTABLE FOR DIRECT CONDITIONING`.

Chosen on the validation timing-prior table: the best train-only predictor's coverage at 50 ms and 100 ms
and its median error, together with whether the site-specific variant beats the global one.

## 17. Deliverables

`docs/V1_ALL_SUBJECT_STEPWISE_VISUALIZATION_REPORT.md`; `artifacts/v1_stepwise_visualization/` with
`provenance.json`, `checkpoint_manifest.json`, `cohort_manifest.csv`, `predictions_manifest.csv`,
`metrics_by_window.csv`, `metrics_by_subject.csv`, `metrics_by_site.csv`, `r_to_ppg_peak_delays.csv`,
`delay_summary.csv`, `timing_prior_validation.csv`, `dashboard/`, `figures/`, `beat_zooms/`.

**V1 ends at its report. No method is implemented and no training is started, in any branch of the
outcome.** Checkpoints and prediction dumps never enter git.
