# E0 — WW-DET Spontaneous Event Audit — PREREGISTRATION (frozen)

**Status: FROZEN (2026-10-01).** Committed and pushed with the E0 implementation **before any event-level E0 outcome**
(taxonomy, counterfactual, localization, feature, probe) is computed on any population, ARCH-TRAIN included. It is never
edited afterwards; amendments are dated and separate. Hashes: `artifacts/e0_wwdet_event_audit/prereg_manifest.json`.

## 0. Evidence status — read first

- **E0 is a post-hoc mechanistic audit of frozen outputs.** It is analysis only: no ECG model, detector or guard is
  trained or changed, and no waveform is modified. The only fitted object is the diagnostic logistic probe of §9.
- **Neither evaluation population is fresh.**
  - ARCH-VAL was analysed by C0 and C0-A.
  - ARCH-HOLDOUT was opened by C0 (`f9e25b5`) and again by C0-A (`e3dc3f7`).
  - Nothing in E0 is validation, fresh or otherwise. ARCH-VAL is the primary E0 population, ARCH-HOLDOUT a secondary one.
- **Already known before this preregistration (disclosed).** C0-A published the aggregate event counts:
  - ARCH-VAL false R — WW-DET 14,994, placed events 11,379, C0 11,684.
  - ARCH-VAL missed R — WW-DET 29,125, placed 29,627, C0 29,714.
  - ARCH-HOLDOUT false R — 15,814 / 12,491 / 12,761.
  - ARCH-HOLDOUT missed R — 28,374 / 28,845 / 28,918.
  - The per-window F1 of WW-DET exceeds the placed events' F1.
  - WW-DET's output has ≥ 2 detected R in at least 645 (VAL) / 571 (HOLDOUT) windows that received no event.
  - By the identities of §3.3 these imply, on ARCH-VAL, type C ≥ 502 and type D ≥ 3,615.
  - Everything else is unknown at this commit: the type counts themselves, per-window and per-patient rates, the
    counterfactual metrics, localization, features and the probe.
  - The decision thresholds of §11 were set with this aggregate knowledge. Where possible they reuse the program's
    existing 0.02 materiality margin.
- **Exact reproduction (before this preregistration, `artifacts/e0_wwdet_event_audit/reproduction.json`).**
  - Frozen checkpoints matched their recorded sha256: detector, C0 and WW-DET.
  - The placed events and C0 renders equal C0's stored ones bit for bit.
  - WW-DET re-rendered with C0-A's own `render_ww`.
  - Placed-event, C0 and WW-DET F1 / precision / recall / FP / FN / RR-MAE / HR-MAE, with their CIs, reproduce the
    C0-A stored values on both populations with maximum absolute deviation **0.0**, and raw FP / FN totals are equal.

## 1. Frozen inputs (never modified)

| input | source |
|---|---|
| split | C0 ARCH-TRAIN / VAL / HOLDOUT (3,470 / 433 / 434 patients) |
| timing detector and placed events | C0 `detector.pt`; stored VAL / HOLDOUT events, re-verified by re-detection |
| CoherentBeat-C0 | C0 `c0.pt`, rendered by C0's `render_c0` |
| WW-DET | C0-A `ww.pt`, rendered by C0-A's `render_ww` (event raster σ = 20 ms) |
| reference R | the RD1 neurokit cache used by C0 and C0-A (`C0.reference_peaks`) |
| rendered R | `B._peaks` (neurokit) on each float64 render, exactly as C0 / C0-A |
| matching / metrics | `rpeaks.match_rpeaks`; `paper_metrics.rpeak_prf_at` and `beat_level_metrics` |
| PPG peaks (feature 16 only) | the existing project detector `s1_audit.dsp_ppg_peaks`; no new PPG detector |

## 2. Populations

- **ARCH-TRAIN** (3,470 patients, 231,220 windows) is used **only to fit the probe** and to build the QRS template.
  - The frozen detector is run on ARCH-TRAIN, and WW-DET and C0 are rendered at those events. This is inference only,
    under the same protocol as VAL / HOLDOUT.
  - The detector and WW-DET were trained on ARCH-TRAIN, so these are in-sample outputs and their type mix may differ
    from VAL / HOLDOUT (a limitation, reported).
- **ARCH-VAL** (433 patients, 28,649 windows) is the primary population. **ARCH-HOLDOUT** (434 patients, 28,531
  windows) is the previously opened secondary population.
- No old validation or test data and no external data are used.

## 3. Four-way taxonomy

### 3.1 Matching

The project convention throughout: one-to-one greedy assignment by |Δt| within ±50 ms (≤ 6.4 samples at 128 Hz),
`rpeaks.match_rpeaks`. Three independent matchings per window:

1. REFERENCE ↔ PLACED;
2. REFERENCE ↔ RENDERED;
3. PLACED ↔ RENDERED.

### 3.2 Types of every rendered detection

| | reference-matched (2) | not reference-matched |
|---|---|---|
| **placed-matched (3)** | **A** supported true event | **B** inherited false event |
| **not placed-matched** | **C** spontaneous recovery | **D** spontaneous hallucination |

- **Properties.** The types are mutually exclusive and exhaustive. A + C equals the TP and B + D the FP of the C0 / C0-A
  metric in every window; this is asserted at run time.
- **Audit trail.** Event-level identities (window, sample, type, assigned reference index, assigned placed index) are
  stored.
- **Triangle cases.** Type A detections whose placed partner is not the reference beat's placed partner are counted
  and reported.
- **Arms.** The taxonomy is computed for WW-DET and for CoherentBeat-C0, on ARCH-VAL and ARCH-HOLDOUT (and for WW-DET on
  ARCH-TRAIN for the probe).

### 3.3 Accounting identities (per window and pooled)

Notation: TP_P and FP_P are the placed events' true and false counts (matching 1).

- TP_rendered − TP_P = **C − L**, where L = TP_P − A (placed true events not carried as type A).
- FP_rendered − FP_P = **D − M**, where M = FP_P − B (placed false events not carried as type B).

### 3.4 Reported

- **Per type:** raw counts; per-window rate (raw / windows); patient-macro rate (equal-patient-weight mean of per-window
  counts, patient-bootstrap CI); percentage of rendered detections; A and C as a percentage of reference beats.
- **Primary mechanistic quantity:** ΔD = D_WW − D_C0 per window (paired, patient-bootstrap CI).
- **Secondary:** ΔC.
- **Also reported:** ΔA, ΔB and ΔFP (WW − C0).

## 4. Detection-level counterfactuals (diagnostics, never methods)

| set | definition |
|---|---|
| HARD GUARD | keep a WW detection iff it is within 50 ms of **any** placed event (not one-to-one) |
| ORACLE-1 | remove type D |
| ORACLE-2 | remove types B and D (B is inherited from the timing module; ORACLE-2 is an impossible-reference bound) |
| WW − C | remove type C (the contribution of C) |
| A + B only | remove types C and D (one-to-one version of the hard guard) |

- **Metrics for each set** (C0's functions, unchanged): per-window precision, recall, F1, FP, FN, RR-MAE and HR-MAE, as
  patient-cluster means with CIs. Also reported: paired differences against WW-DET, against the placed events and
  against C0, and pooled micro precision / recall / F1.
- **Micro chain:** placed events → carried (A + B) → + C → + D (= WW-DET).
- **No FD is computed**, because no waveform changes. The oracles are impossible-reference bounds, never achievable
  performance.

## 5. Localization and edge analysis (types C and D, both arms)

- **Distances:** to the nearest, previous and next placed R (ms); signed offset to the nearest.
- **Phase:** (t − r_prev) / (r_next − r_prev) when bounded by two placed events. Bins: 0–0.15, 0.15–0.30, 0.30–0.45,
  0.45–0.55, 0.55–0.70, 0.70–0.85, 0.85–1.00.
- **Distance to the nearest placed R:** < 100, 100–200, 200–350, 350–500, > 500 ms, split into after / before the
  nearest placed R. Neutral wording only; no P / QRS / T labels.
- **Region:** no placed event in the window / before the first / between / after the last placed event.
- **Distance to the nearest window edge:** < 125, 125–250, 250–500, ≥ 500 ms.

## 6. Local features (WW-DET types C and D; frozen definitions, no selection after outcomes)

At the detection t, t* is the waveform maximum within ±3 samples (±23 ms).

| # | feature | definition |
|---|---|---|
| 1 | amp_rel | x(t*) − median of x over ±250 ms |
| 2 | amp_abs | x(t*) (absolute level in the normalized ECG units) |
| 3 | prominence | `scipy.signal.peak_prominences` at t*, window ±250 ms |
| 4 | width_ms | `peak_widths` at half prominence |
| 5 / 6 | max_pos_slope / max_neg_slope | max / min first difference × fs within ±10 samples (±78 ms, the specified ±80 ms in whole samples) |
| 7 | curvature | \|x(t*−1) − 2x(t*) + x(t*+1)\| × fs² |
| 8 | log_symmetry | log(max rising slope on [t*−10, t*] / max falling-slope magnitude on [t*, t*+10]) |
| 9 | diff_rms | RMS first difference × fs over ±250 ms |
| 10 | hf_frac | 5–20 Hz power / power above 0 Hz, Hann-windowed ±250 ms segment |
| 11 | dist_nearest_ms | §5 |
| 12 | phase | §5 (undefined outside two placed events) |
| 13 / 14 | prev_rr_ms / next_rr_ms | the RR intervals the candidate would form with the previous / next placed event: t − r_prev, r_next − t |
| 15 | ppg_slope | mean PPG first difference × fs over ±5 samples |
| 16 | ppg_peak_lag_ms | signed (nearest PPG peak − t), the existing `dsp_ppg_peaks` |
| 17 / 18 | qrs_corr / qrs_l2 | Pearson correlation and normalized L2 of z-scored ±10-sample segments to the TRAIN QRS template |
| C0 | c0_prominence, c0_amp_rel, c0_max_pos_slope, c0_qrs_corr | the same definitions on the frozen C0 waveform at the same t (its own t*) |
| WW − C0 | d_prominence, d_amp_rel, d_max_pos_slope, d_qrs_corr | differences |

- **TRAIN QRS template:** the mean ARCH-TRAIN reference ECG over ±10 samples about every ARCH-TRAIN reference R peak
  whose support lies inside its window. It is built once in the ARCH-TRAIN stage and only loaded afterwards.
- **Missing values:** features undefined at edges or without placed events are left missing (no imputation in §8).

## 7. Same-location morphology

At every WW-DET type C and type D detection: ±250 ms segments, no warping, nan-padded at window edges, of WW-DET, C0,
reference ECG, PPG and the event raster. Reported as patient-macro mean curves (per-patient mean, then the mean over
patients).

## 8. Feature effects (C vs D)

- **Per feature:** patient-macro medians per type; event means; Cohen's d (C − D, pooled event SD) with a
  patient-cluster bootstrap CI computed from per-patient sufficient statistics.
- **Treatment:** effect-size analysis only. No p-values are reported, so no multiplicity correction is needed.

## 9. Diagnostic separability probe (the only fitted model)

- **Task:** label 1 = type C, 0 = type D, WW-DET detections only.
- **Model:** L2 logistic regression, `class_weight = "balanced"`, lbfgs.
- **Feature groups (exactly three):**
  - P1: event context, features 11–14;
  - P2: WW waveform, features 1–10, 17, 18;
  - P3: all features, P1 ∪ P2 ∪ {15, 16} ∪ the C0 contrasts and WW − C0 differences.
- **Fitting:** ARCH-TRAIN only.
  - Patient-grouped 5-fold CV (`GroupKFold`) chooses C ∈ {0.01, 0.1, 1, 10} by mean fold AUROC (ties → smaller C).
  - Imputation (TRAIN median plus one missingness indicator per feature with any missing value) and z-scoring are
    fitted on training data only, inside every fold.
  - Decision threshold: maximum balanced accuracy on the ARCH-TRAIN out-of-fold predictions at the chosen C.
  - The final probe is refitted on all ARCH-TRAIN data and then frozen.
- **Evaluation:** applied unchanged to ARCH-VAL and ARCH-HOLDOUT.
  - AUROC (primary) and AUPRC (positive = C, read against the reported prevalence), both with patient-bootstrap CIs.
  - Balanced accuracy at the TRAIN threshold, sensitivity for C, specificity against D.
  - A 10-bin quantile calibration table (descriptive).
- **No resampling** of VAL / HOLDOUT.

## 10. Statistics

- 2,000 patient-clustered bootstrap replicates, seed 20261001, equal patient weight (C0's `cluster_ci`).
- Paired WW − C0 and counterfactual comparisons use identical patient resamples.
- Detections are never resampled independently. Event-level statistics resample whole patients.

## 11. Frozen case rule

Each criterion is evaluated per population. CI = patient-bootstrap 95% CI; "WW" = WW-DET; ΔFP = per-window FP of WW −
C0.

| criterion | definition |
|---|---|
| K1 meaningful excess D | ΔD CI lower > 0 **and** ΔD ≥ 0.5 × ΔFP (hallucinations explain at least half of WW's false-detection excess over C0) |
| K2a nontrivial C | ΔC CI lower > 0 **and** pooled C / (C + D) ≥ 0.20 |
| K2b hard guard lowers F1 | F1(HARD GUARD) − F1(WW) CI upper < 0 |
| K2c hard guard loses recall materially | recall(HARD GUARD) − recall(WW) CI upper < −0.02 |
| K2 | K2a or K2b or K2c |
| K3 separable | P3 AUROC ≥ 0.80 on ARCH-VAL; ≥ 0.75 on ARCH-HOLDOUT |
| K4 oracle headroom | FP(ORACLE-1) − FP(WW) CI upper < 0 **and** its magnitude ≥ 0.5 × ΔFP **and** F1(ORACLE-1) − F1(WW) CI lower > 0 |
| B conditions | not K2a **and** D / (C + D) ≥ 0.5 **and** hard guard − WW: F1 CI lower > 0, FP CI upper < 0, recall CI lower > −0.02 |

**Cases:**

- **E0-A** (selective guard strongly motivated) = K1 ∧ K2 ∧ K3 ∧ K4.
- Otherwise **E0-B** (hard event lock motivated) if the B conditions hold.
- Otherwise **E0-C** (no local guard motivation).
- **Final case:** the ARCH-VAL case, kept only if ARCH-HOLDOUT gives the same case; otherwise E0-C.

**Statement rule.** If K2b or K2c holds, the report states that a future guard must be selective rather than hard.

**Consequences:**

- E0-A: only `docs/E1_SELECTIVE_EVENT_GUARD_ARCHITECTURE_DRAFT.md` (a design note that protects type C; nothing trained).
- E0-B: only a draft of the simplest event-constrained alternative.
- E0-C: this branch stops; the WW / C0 trade-off stays a measured trade-off.

## 12. Figures and atlas

- **`figure.png` panels:**
  - A — taxonomy;
  - B — type rates, WW vs C0;
  - C — micro precision / recall chain;
  - D — phase of C vs D;
  - E — same-location means;
  - F — probe ROC;
  - G — counterfactual precision / recall.
- **`atlas.png` (ARCH-VAL):** 12 WW type C, 12 type D, 12 type B, and 12 WW / C0 detection-status differences (a
  detection unmatched one-to-one by the other arm).
  - Sampling: salted ranks (`e0-atlas-v1-{C,D,B}`, `e0-atlas-v1-diff`), never hand-picked.
  - Each panel shows the reference ECG, WW, C0, PPG, reference R, placed R, WW R and C0 R.

## 13. Claim boundaries

- **Never claimed:**
  - fresh validation;
  - causal explanation of hallucinations;
  - that a learned guard will work or preserve FD;
  - clinical correctness of type C beyond the R-match definition;
  - overall superiority of CoherentBeat or of WW-DET;
  - justification of stochastic generation;
  - test, external or clinical generalization.
- **Language follows the frozen taxonomy.** The hard guard and oracles are diagnostics, not methods.

## 14. Implementation, tests, artifacts

- **Code:** `scripts/e0_event_audit.py` (stages `reproduce`, `manifest`, `train_infer`, `evaluate`, `figure`, `atlas`)
  and `src/ppg2ecg/eventaudit/{taxonomy,features,probe}.py`.
  - `evaluate` refuses to run without the committed preregistration manifest.
  - A synthetic dry run of every stage was done before this commit; it found and fixed a feature-merge column collision.
- **Tests:** `tests/test_e0_event_audit.py`, 18 tests:
  - reproduction;
  - one-to-one matching;
  - exclusive / exhaustive types and their definitions;
  - ambiguous matching;
  - hard guard; oracles;
  - phase range and edge handling;
  - TRAIN-only template;
  - TRAIN-only standardization;
  - grouped CV;
  - VAL / HOLDOUT kept out of probe fitting;
  - clustered bootstrap;
  - fixed atlas sampling;
  - case rules;
  - features;
  - no training code;
  - event-table regression.
- **Artifacts** (`artifacts/e0_wwdet_event_audit/`):
  - `audit.md`, `input_hashes.json`, `reproduction.json`, `prereg_manifest.json`
  - `event_taxonomy_{val,holdout}.json`, `event_level_{val,holdout}.parquet`
  - `counterfactual_metrics.json`, `localization.json`, `feature_summary.json`, `same_location.json`
  - `probe_config.json`, `probe_cv.json`, `probe_{val,holdout}.json`
  - `bootstrap.json`, `e0_case.json`, `train_population.json`, `qrs_template.json`, `atlas_index.json`
  - `atlas.png`, `figure.png`
- **Not committed:** caches in `outputs/e0_wwdet_event_audit/`.
- **Report:** `docs/E0_WWDET_SPONTANEOUS_EVENT_AUDIT_REPORT.md`. **HARD STOP** after E0.
