# Q1 — Conditional Support Degradation Audit — PREREGISTRATION

**Does PPG→ECG generation remain plausible when the PPG becomes uninformative?**

Frozen before any Q1 result is computed. Once committed and pushed, this document is never edited.

| | |
|---|---|
| Stage | Q1 (audit; no new method) |
| Base commit at design time | `55fe1e1b14b87b3e660f9dc0828a87e64ce032af` (R3 result) |
| HEAD at preregistration | `602bd378ba9d3a1d3cd2eddcb0203024c127c684` (documentation-only commit, see §0.1) |
| Upstream pins | PENGUIN `6cd70cdefb91f10efeb8dce34019b5067cb25344`, iMeanFlow `bf60cd7cb653f6628e59d48034b333c5eba445e2` |
| Environment | RTX 5090, torch 2.11.0+cu130, scipy 1.16.3, numpy 2.3.5, neurokit2 0.2.12, Python 3.13.9 |
| Test subjects | `kjd`, `ssx` — **never loaded** (`assert_no_test_subjects` first statement of every Q1 script) |
| C2 | remains deferred; Q1 starts no training |

---

## 0. What Q1 is and is not

Q1 manipulates **condition information**, not architecture. There is:

**NO** new generator · **NO** training or weight update of any kind · **NO** new attention/adapter · **NO** flow-objective change · **NO** test-subject access · **NO** C2 · **NO** hyperparameter search · **NO** NFE sweep.

The single question:

> When the conditioning PPG is progressively deprived of information, does the generator appropriately become uncertain, or does it continue to emit plausible-looking but **conditionally unsupported** ECGs?

### 0.1 Status declarations (frozen)

1. **Q1 was designed after V1/R1/R2/R3.** It is **exploratory / problem-discovery** evidence. It is **not** independent confirmation of any earlier verdict, and it does not confirm itself.
2. **Synthetic corruption demonstrates model response to controlled input degradation, not naturally occurring clinical artefact causality.** No claim about real-world PPG failure modes follows from §6.
3. **"Plausibility" refers only to the preregistered GT-independent marginal physiological proxies of §10.** It is not perceptual realism, not clinical realism, and not a calibrated probability.
4. The word **hallucination is not used** for any Q1 result. The primary terminology is **conditional-support / plausibility decoupling**. "Hallucination" may only be used if a *later* experiment establishes perceptual/clinical realism.
5. Two validation subjects (`an0`, `k2s`) only; these are **development** validation (four windows were visually previewed before X4-0 and are excluded from every frozen subset), so Q1 is a within-development audit.
6. **Deviation recorded at preregistration time**: the working tree at design time contained three untracked documentation files written earlier in the same session (project status summary, top-tier assessment, technical specification). To satisfy the "clean working tree" integrity requirement they were committed as documentation-only commit `602bd37` (parent `55fe1e1`) and pushed **before** this preregistration. No code, config, checkpoint or artifact was touched by that commit; every other integrity check (HEAD == origin/main, submodule pins, A4 md5 `31c042d291052fbb6dc15263ad316be2`, no C2 outputs) passed at `55fe1e1` and again at `602bd37`.

### 0.2 Prior context this builds on

- **R1**: whole-window PPG makes RR / global rhythm strongly observable (Global-TCN F1@150 = 0.8582, RR MAE 31.4 ms on 8,192 validation windows); exact R timing is limited (F1@50 = 0.6199).
- **R2/R3**: injecting rhythm information improves event correspondence but costs structure (R3 GTF-TRUE F1 excess +0.0406 [+0.0353, +0.0458] vs B, with S4 worsening −0.0069).
- **R3 GTF-ORACLE** (GT-R leakage; diagnostic only) reached F1 excess 0.8164, so the target-side interface capacity is not the only bottleneck.

Q1 therefore holds the architecture fixed and moves the condition.

---

## 1. Frozen components (no parameter may change)

| Role | Checkpoint | Identity asserted at runtime |
|---|---|---|
| **PRIMARY generator — arm B** | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` (C1 arm-B replay = A4 iMeanFlow weights, round 46) | file sha256 prefix `557c7054`, state_dict sha256 prefix `47d7ccb9`, `n_params_total = 4,568,707`, `imf_cfg.cond_mode = "h_only"`, `h_scale = 1.0` |
| **Support probe (not a generator arm)** | `outputs/r1_global_tcn_seed42/checkpoint_best.pt` (R1 Global-TCN, 328,897 params) | file sha256 prefix `bfe76ea6`, state_dict sha256 prefix `0986a7af` |
| **SECONDARY generator (optional)** | `outputs/r3_gtf_true_seed42/module_step2200.pt` on top of the same frozen generator + frozen Global-TCN | module file sha256 prefix `ebf55708`; fusion family `gtf`, gate `adaptive`, 12,849 trainable-at-R3-time parameters, all frozen here |
| **A4 generator (identity witness only)** | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` | md5 `31c042d291052fbb6dc15263ad316be2` unchanged |

All modules are loaded with `requires_grad_(False)` and `eval()`. Q1 constructs **no optimizer** and **no trainable parameter** (asserted by test). The R1 Global-TCN is used **only** as a quantitative proxy for how much R/rhythm information survives in the corrupted PPG — it is never an arm of the generator comparison. The secondary GTF-TRUE arm carries **stronger target-derived R supervision** (its scaffold comes from a probe trained on GT-R labels) and is labelled as such everywhere; the primary problem-discovery verdict rests on arm B alone.

---

## 2. Populations (all frozen, metadata-only selection)

| Cohort | Definition | Size | Use |
|---|---|---|---|
| **PRIMARY** | the exact frozen development cohort of C0/C1/R2/R3: `artifacts/x4_0_event_reliability/nfe_subset.json` (salt `x4-event-nfe-v2`, 1,024 array rows per subject, four previewed windows excluded), asserted element-for-element; all four PPG sites as already represented | **2,048** windows (an0 1,024 + k2s 1,024), 19,834 GT beats | corruption sanity, R1 support, arm-B fidelity, plausibility, controls, coupling |
| **UNCERTAINTY** | balanced metadata-only subset of PRIMARY: per (subject, site) take the 64 lowest SHA256 ranks of `"q1-uncertainty-v1\|{subject}\|{site}\|{window_index}"` where `window_index` is the npz per-site temporal index of that cohort row | **512** windows (2 subjects × 4 sites × 64) | 8-source uncertainty sweep |
| **MARGINAL REFERENCE** | GT ECG of the 12 WildPPG **train** subjects only; per (subject, site) the 256 lowest SHA256 ranks of `"q1-marginal-reference-v1\|{subject}\|{site}\|{window_index}"` | **12,288** windows | train-real reference intervals of §10 |
| **NATURAL QUALITY** | the frozen R1 **validation** cohort (salt `r1-global-rhythm-observability-v1`, 1,024 per subject × site) — the same 8,192-window population used as the R2/R3 site-wise secondary population | **8,192** windows | exploratory audit of §18 |
| **VISUAL ATLAS** | the `viz` rows of `artifacts/v1_stepwise_visualization/cohort_manifest.csv` restricted to the validation subjects (an0 32 + k2s 32) — no new example is selected | **64** windows | atlas of §21 |

Selection uses **only** subject, site and window index. It must not use PPG quality, R peaks, generator scores, corruption response, ECG morphology or visual inspection (asserted by test: the cohort builder receives no signal array).

**Naming note (frozen before results).** §18 of the task calls the natural-quality population "the V1 8,192 validation-window cohort". V1's own cohorts are 448 / 1,792 / 7,168 windows over 14 subjects; the only 8,192-window validation cohort in this repository is R1's (an0/k2s, 1,024 per subject × site). Q1 uses that R1 validation cohort and records the naming resolution here. Likewise, "the EXACT V1 frozen 64 visualization windows" resolves to the 64 V1 `viz` rows belonging to an0/k2s (V1 VIZ has 448 rows over 14 subjects; exactly 64 of them are validation-subject rows).

Source seed for all single-output fidelity evaluation: **0** (the frozen `[2048,1,1024]` bank, sha256 `868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f`, asserted). Primary NFE: **4** (uniform schedule `ER.UNIFORM[4]`). No NFE sweep — Q1 is not an efficiency experiment.

---

## 3. Where corruption is applied

Corruption is applied to the **exact frozen preprocessed model-input PPG** row (`data/processed/wildppg_8s/<id>.npz` key `x`, `[1024]` float32, 128 Hz, 8 s, already band-passed 0.5–4 Hz, z-scored and min-max mapped to [−1, 1]) — i.e. **after** the frozen preprocessing and normalisation pipeline.

- **No renormalisation after corruption** (asserted by test), because renormalising would undo the intended information degradation. Corrupted rows may leave [−1, 1].
- Corruption is computed in float64 and cast back to float32 for the model.
- The **GT ECG is never touched** (asserted by test).
- sha256 of the clean row block and of every corrupted row block is recorded per condition in `corruption_manifest.csv`.

---

## 4. Corruption families (exactly three, all deterministic)

Conditions (13 total, including the two inference-only controls of §7):

`CLEAN`, `LP_3.0Hz`, `LP_2.0Hz`, `LP_1.25Hz`, `SNR_20dB`, `SNR_10dB`, `SNR_5dB`, `SNR_0dB`, `DROP_0.5s`, `DROP_1.0s`, `DROP_2.0s`, `SHUFFLED`, `NULL`.

### A. BANDLIMIT — additional zero-phase low-pass

`scipy.signal.butter(4, fc / 64.0, btype="low")` then `scipy.signal.filtfilt(b, a, x)` (default `padlen`), applied to the already-preprocessed PPG. Cutoffs `fc ∈ {3.0, 2.0, 1.25}` Hz (Nyquist 64 Hz). Order **4**, chosen before evaluation to match the project's existing filter order; coefficients and the scipy version are recorded in `corruption_manifest.csv` / `provenance.json`.

*Purpose*: progressively remove local pulse morphology while often retaining coarse periodicity — the family that can separate "rhythm survives" from "waveform detail survives".

### B. IN-BAND NOISE — band-limited Gaussian at exact SNR

1. `seed = int(SHA256("q1-noise-v1|{subject}|{site}|{window_index}|{snr_tag}").hexdigest()[:16], 16)`; `n = np.random.default_rng(seed).standard_normal(1024)`. No dependence on global RNG state.
2. Band-limit: `butter(4, [0.5/64, 4.0/64], btype="band")` + `filtfilt` → `n_bp`.
3. Scale to the exact target SNR against the signal RMS: `x_corr = x + n_bp · (rms(x) / rms(n_bp)) · 10^(−SNR/20)`, so that `20·log10(rms(x) / rms(added noise)) = SNR` exactly (asserted by test to 1e-6 dB).

Levels: `SNR ∈ {20, 10, 5, 0}` dB.

*Purpose*: reduce condition information without introducing out-of-band artefacts (noise power outside 0.5–4 Hz is ≤ the filter's stop-band leakage; a test asserts ≥ 99 % of the added noise power lies in 0.4–4.5 Hz).

### C. TEMPORAL INFORMATION DROPOUT — contiguous linear-interpolation gap

Durations `{0.5, 1.0, 2.0}` s = exactly `{64, 128, 256}` samples. Start index
`start = 1 + int(SHA256("q1-drop-v1|{subject}|{site}|{window_index}|{dur_tag}").hexdigest()[:16], 16) mod (1024 − L − 1)`, so `1 ≤ start` and `start + L ≤ 1023`: both boundary samples `x[start−1]` and `x[start+L]` exist and the gap lies fully inside the window. The gap is replaced by
`x[start : start+L] = np.linspace(x[start−1], x[start+L], L + 2)[1:−1]`.

Placement is metadata-only: it must not depend on detected pulses, R peaks or any signal statistic (asserted by test).

*Purpose*: remove a localised fraction of condition information.

---

## 5. Corruption sanity (before any generator is run)

For every condition, per window, clean vs corrupted **PPG only** (no ECG involved):

| Metric | Definition |
|---|---|
| `ppg_corr` | Pearson correlation of clean and corrupted rows |
| `ppg_nrmse` | `‖x_c − x‖₂ / (‖x‖₂ + 1e-12)` |
| `rms_ratio` | `rms(x_c) / (rms(x) + 1e-12)` |
| `spec_l1` | L1 distance between the power spectra of `x` and `x_c`, each normalised to unit total power (rfft, 1024 samples) |
| `band_frac_{0.5-1,1-2,2-4,4-15,15-64}Hz` | fraction of total power in each band, for `x_c` (and for `x` in the CLEAN row) |
| `n_pulses` | pulse count from the frozen V1 PPG systolic-peak detector |
| `pulse_interval_mae_ms` | MAE between corrupted-PPG pulse intervals and clean-PPG pulse intervals, over one-to-one matched pulses (±150 ms); NaN if < 2 matched consecutive pairs |

**Monotonicity gate (STOP condition).** Within BANDLIMIT and NOISE, the macro median `ppg_corr` must be non-increasing and macro median `ppg_nrmse` non-decreasing with severity; within DROPOUT the same is required with severity ordered by gap duration. If any expected monotonicity is violated, the deviation is reported and Q1 **stops before generator evaluation**.

---

## 6. Does ECG-relevant support actually decrease? (PRIMARY SUPPORT AXIS)

The frozen R1 Global-TCN is run on every corrupted PPG with the exact frozen R1 inference recipe: `sigmoid(logits)`, `extract_events(threshold = 0.35, nms_refractory = 32 samples)`. Scored against GT ECG R peaks (frozen `detect_rpeaks`, neurokit) with the frozen one-to-one matcher:

`f1@50`, `f1@100`, `f1@150`, `f1@200`, `f1@250`, `missing`, `spurious`, `beats_ratio_dev`, `rr_mae_ms` (matched consecutive GT beats at 150 ms, as in R1), `rr_median_ae_ms`.

Aggregation: equal-subject macro. Per family we plot severity vs `f1@150` and severity vs `rr_mae_ms`.

**We do not assume corruption reduces support — we measure it.** A family in which strong low-pass preserves rhythm while destroying pulse morphology is itself an informative outcome (§17).

---

## 7. Generator conditional fidelity + dependence controls

Frozen arm B, NFE 4, source seed 0, **the same source tensor for every condition** (asserted by test), same GT ECG. Metrics are the frozen R2/R3 scoring pipeline, unchanged:

- **EVENT**: raw `f1`, `chance_f1` (count-matched random-phase floor, 20 draws, seed 20260901), `f1_excess`, `precision`, `recall`, `missing`, `spurious`, `beats_ratio_dev`.
- **STRUCTURE** (GT-fixed coordinates, **no waveform shift, no oracle alignment**): `raw_rmse`, `raw_corr`, `raw_qrs_rmse`, `qrs_deriv_rmse` (S4), `qrs_curvature_err` (S5), `qrs_e_dev` (S6), `p2p_dev` (S7), `hf_err` (S8, frozen HF metric).

Two **inference-only** controls on the 2,048 cohort (diagnostic, outside the naturalistic claim):

- **SHUFFLED** — the PPG row is replaced by another window of the same (subject, site): rank rows by `SHA256("q1-condition-shuffle-v1|{subject}|{site}|{window_index}")` and map rank *i* → rank *(i+1) mod n* (bijective, no fixed point; asserted by test).
- **NULL** — an all-zero normalised PPG of identical shape (exact zeros; asserted by test). Deliberately out-of-distribution and **not** a model of a physiological low-quality PPG.

---

## 8. GT-independent marginal plausibility proxies

Reference intervals are built **only** from TRAIN-subject GT ECG (§2 MARGINAL REFERENCE cohort). No validation target information may define them (asserted by test: the reference builder refuses any subject outside the 12 train subjects, and kjd/ssx are firewalled).

Per ECG window (real or generated), using the existing frozen implementations, with peaks detected **on that same waveform**:

| ID | Feature | Definition |
|---|---|---|
| P1 | detector success | `detect_rpeaks(x, 128)` returns ≥ 2 peaks (so HR is defined) → `detector_valid` |
| P2 | beat rate | `hr_bpm(peaks, 128)` |
| P3 | QRS width | median over beats of `qrs_width_ms(x, r, 128)` |
| P4 | QRS peak-to-peak | median over beats of `ptp(beat_window(x, r, 128))` (83-sample window, fully contained beats only) |
| P5 | max-derivative statistic | median over beats of `max\|diff(beat_window)\|·128` (the numerator of the existing `raw_slope_ratio`) |
| P6 | HF fraction | `hf_energy_ratio(x, 128, cutoff 15 Hz)` (whole window) |

From the train-real distribution of P2–P6 (windows where the feature is defined) take the **1st and 99th percentiles**. These bounds are frozen before any generated ECG is scored and are never tuned afterwards.

For generated ECG report `detector_valid` fraction, and for each of P2–P6 the fraction of windows whose feature is defined **and** inside `[p1, p99]`. `marginal_support_fraction` = the per-window mean of the five in-support indicators, then equal-subject macro.

**Wording (frozen)**: this is a *GT-independent marginal physiological support proxy*. It is **not** a validated clinical realism score and must never be labelled "realism".

---

## 9. Multi-source uncertainty

On the frozen 512-window UNCERTAINTY cohort, for every condition (including SHUFFLED and NULL), 8 samples from source seeds **0…7** at NFE 4:

| ID | Definition |
|---|---|
| U1 | mean over time of the pointwise standard deviation across the 8 sampled ECGs |
| U2 | mean pairwise waveform RMSE over the 28 sample pairs |
| U3 | standard deviation (ddof 0) of the 8 predicted beat counts |
| U4 | mean pairwise generated-event F1@50 ms over the 28 pairs (`peak_train_agreement`; **lower = more source-driven event variation**) |
| U5 | the same at 150 ms |
| U6 *(secondary)* | per-GT-beat timing SD across samples, using only GT beats matched within ±250 ms in ≥ 4 of the 8 samples; reported only if the matching is stable, not forced |

---

## 10. Statistics

Every corruption comparison is paired at the exact window level against `CLEAN`.

- Estimator: **subject-stratified paired bootstrap**, equal an0/k2s weight, **2,000** replicates, `np.random.default_rng(20260903)`.
- Verdict per comparison: `improves` if the 95 % interval lies entirely above 0, `worsens` if entirely below, else `unresolved` — always relative to the explicitly stored orientation.
- Orientations (stored in every row of `paired_bootstrap.csv`):
  - **SUPPORT**: positive = clean better than corrupted.
  - **FIDELITY**: positive = clean better than corrupted.
  - **PLAUSIBILITY**: positive = clean higher marginal support than corrupted.
  - **UNCERTAINTY**: positive = corrupted more uncertain / more diverse than clean.

### Support–fidelity coupling (§14 of the task)

At window level, per corruption family (pooling that family's levels, and also per level):

| Pair | X | Y |
|---|---|---|
| A | R1 `f1@150` | generator `f1_excess` |
| B | R1 `rr_mae_ms` | generator `beats_ratio_dev` |
| C | R1 `f1@150` | generator `raw_qrs_rmse` |

Report Spearman ρ with a subject-stratified bootstrap CI (2,000, seed 20260903). **Correlation is not causality** and will not be reported as such.

---

## 11. Preregistered verdict rules (frozen thresholds)

Severity levels used for the verdicts — the most severe **non-NULL** level of each family: BANDLIMIT `LP_1.25Hz`, NOISE `SNR_0dB`, DROPOUT `DROP_2.0s`.

**SUPPORT-DEGRADING** iff
- **S-A** R1 `f1@150` worsens vs CLEAN with the paired 95 % CI entirely in the worse direction, **and**
- **S-B** at least one of `rr_mae_ms`, `missing`, `spurious` also clearly worsens (CI entirely worse).

**CONDITIONAL-FIDELITY-DEGRADING** (only assessed for support-degrading families) iff
- **F-A** generator `f1_excess` worsens vs CLEAN with CI entirely worse, **and**
- **F-B** at least one of `raw_qrs_rmse`, `qrs_deriv_rmse`, `qrs_curvature_err` clearly worsens.

**MARGINAL-PLAUSIBILITY-PRESERVED** iff
- **P-A** `detector_valid` fraction drops by **< 0.05 absolute**, **and**
- **P-B** `marginal_support_fraction` drops by **< 0.05 absolute**.

**UNCERTAINTY-NONRESPONSIVE** iff both
- **U-A** U1 (pointwise sample SD) does **not** increase by **≥ 10 %** relative to CLEAN, **and**
- **U-B** at least one of U3 (beat-count SD) / event diversity (U4, where a clear increase in diversity means a clear *decrease* in pairwise event F1) also fails to show a clear increase (CI not entirely in the more-uncertain direction).

The 0.05 absolute and 10 % relative thresholds are frozen now and must not be changed after seeing results.

### Final Q1 verdict — exactly one

**A. CONDITIONAL-SUPPORT / PLAUSIBILITY DECOUPLING OBSERVED** — at least **two of the three** corruption families satisfy **all four** properties (SUPPORT-DEGRADING **and** CONDITIONAL-FIDELITY-DEGRADING **and** MARGINAL-PLAUSIBILITY-PRESERVED **and** UNCERTAINTY-NONRESPONSIVE). Meaning: the frozen generator keeps emitting outputs inside coarse real-ECG marginal support after the PPG loses substantial target-relevant information, without a commensurate rise in source uncertainty. **Not** a clinical-hallucination claim.

**B. CONDITION LOSS IS REFLECTED IN GENERATOR UNCERTAINTY** — at least two support/fidelity-degrading families show a clear uncertainty increase: U1 increases by ≥ 10 % **and** (U3 increases **or** pairwise event F1 falls clearly). Meaning: the stochastic output responds to degraded condition information. **No calibration claim.**

**C. OUTPUT PLAUSIBILITY COLLAPSES WITH CONDITION** — support/fidelity degrade **and** the marginal plausibility proxies also clearly degrade in ≥ 2 families (i.e. the P-A/P-B bounds are violated in the worse direction). Meaning: the generator visibly/marginally breaks rather than producing conditionally unsupported plausible outputs.

**D. NO CONSISTENT CONDITION-DEGRADATION PATTERN** — fewer than two families produce a coherent support-loss pattern.

Evaluation order: A is checked first; if A fails, B; then C; else D. The implementation of these rules is asserted by a test to match this section literally.

---

## 12. Secondary and exploratory analyses (do not enter the verdict)

- **§17 rhythm vs morphology**: in the BANDLIMIT family, if R1 `f1@150` / RR stay relatively stable while the generator's derivative / curvature / QRS metrics deteriorate, record the *exploratory hypothesis* that coarse rhythm information survives loss of local PPG morphology longer than fine ECG structural information. This is **not** an observability theorem; it motivates a later dedicated ECG-component observability map.
- **§18 natural quality audit** (exploratory, after the synthetic primary result is frozen): on the 8,192-window R1 validation cohort, two **PPG-only** measures reported separately (no ad-hoc composite SQI):
  - `Qnat1 periodicity_score` = maximum normalised autocorrelation of the window over lags corresponding to 30–200 bpm (at 128 Hz over 8 s: lags 38.4–256 samples → integer lags **39…256**).
  - `Qnat2 pulse_template_consistency` = median correlation between the frozen-V1-detected pulse snippets and the within-window median pulse template; **undefined** if fewer than 3 valid pulses.
  Within each (subject, site), defined scores are split into quartiles, then aggregated equal-weight over subject/site. Across quartiles report R1 `f1@150`, R1 `rr_mae_ms`, arm-B `f1_excess`, `raw_qrs_rmse`, `qrs_deriv_rmse`, `qrs_curvature_err`, marginal plausibility, and multi-source uncertainty where available.
- **§19 secondary GTF-TRUE**: if the frozen R3 GTF-TRUE module resolves cleanly, repeat the corruption curves for R1 support, `f1_excess`, `missing`/`spurious`, `qrs_deriv_rmse`, `qrs_curvature_err`, `marginal_support_fraction`. No 8-source sweep unless runtime is negligible. Labelled as carrying stronger target-derived event supervision.
- **§21 visual atlas**: the 64 frozen V1 validation viz windows, rows GT ECG / clean PPG / severe corrupted PPG / clean generated / severe generated / SHUFFLED generated / NULL generated, plus mean ± 1 SD envelope for uncertainty-cohort members. No prediction shifting. Annotations only if deterministic.

---

## 13. Runtime preflight (STOP gate)

Before the full sweep: exactly **100 windows × all corruption conditions × arm B × NFE 4**, measuring wall time and peak VRAM, then projecting the total for (i) the primary 2,048 single-source sweep, (ii) the 512 × 8 uncertainty sweep, (iii) SHUFFLE/NULL, (iv) the optional GTF secondary, (v) scoring. **If the projected total GPU time exceeds 4 hours, Q1 stops** and reports the projection. Cohorts and source counts are never silently reduced.

---

## 14. Required tests

Q1 is not run until the full suite passes. New tests assert: test-subject firewall; frozen checkpoint identity (file and state_dict hashes); **no optimizer and no trainable parameter anywhere in Q1** (static source check + runtime `requires_grad` check); exact frozen primary cohort (element-for-element vs `nfe_subset.json`); exact uncertainty-cohort hashing; corruption determinism (same inputs → identical bytes, twice, and independent of global RNG); noise hits the exact target SNR; noise power confined to the specified band within tolerance; **no post-corruption renormalisation** (static check + numerical check that a scaled input produces a correspondingly scaled corrupted output only where the corruption is linear, and that min/max are not forced back to ±1); dropout duration exact in samples; dropout placement metadata-only; low-pass coefficients frozen (exact values); GT ECG never corrupted; the same source tensor is used across clean/corrupted paired evaluation; exactly 8 source seeds 0…7; SHUFFLE is a bijection with no fixed point; NULL is exactly zeros; the plausibility reference uses train subjects only; validation GT never defines plausibility bounds; bootstrap uses equal subject weighting; the verdict implementation matches §11 literally.

---

## 15. Artifacts

`docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md` (this file), `docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_REPORT.md`, and `artifacts/q1_conditional_support/` containing `provenance.json`, `checkpoint_manifest.json`, `cohort_manifest.csv`, `uncertainty_cohort.csv`, `corruption_manifest.csv`, `corruption_sanity.csv`, `r1_support_metrics.csv`, `generator_fidelity_metrics.csv`, `marginal_plausibility_reference.json`, `marginal_plausibility_metrics.csv`, `uncertainty_metrics.csv`, `support_fidelity_correlations.csv`, `natural_quality_metrics.csv`, `natural_quality_quartiles.csv`, `paired_bootstrap.csv`, `decision.json`, `visual_atlas/`.

Never committed: raw data, predictions, checkpoints, large artifacts.

---

## 16. Claim boundaries

**Allowed if verdict A**: "Under controlled PPG information degradation, paired ECG fidelity declines while coarse marginal ECG plausibility remains largely preserved and source-driven output variability does not increase commensurately." / "This indicates conditional-support/plausibility decoupling in the evaluated generator."

**Not allowed** under any verdict: "the model hallucinates clinically dangerous ECGs"; "PPG quality causes clinical failure"; "the model is overconfident" (absent a calibrated probabilistic evaluation); "PPG does not contain ECG information"; any information-theoretic unobservability claim.

---

## 17. Next-step decision (recommendation only — nothing is implemented automatically)

- **A** → recommend *quality/observability-aware conditional generation* (observable component strongly condition-anchored; weakly supported component given uncertainty / abstention / wider conditional distribution). Method design requires a **new** preregistration.
- **B** → focus next on uncertainty calibration rather than new conditioning.
- **C** → the failure is ordinary distributional breakdown; the "plausible unsupported generation" story is not supported.
- **D** → do not build a method around PPG-quality uncertainty yet; consider a systematic ECG-component observability map.

---

## 18. Commit order

1 repository integrity → 2 preregistration → **3 commit + push preregistration** → 4 implementation → 5 tests → 6 commit + push implementation → 7 runtime preflight → 8 corruption sanity → 9 R1 support sweep → 10 primary arm-B single-source sweep → 11 8-source uncertainty sweep → 12 SHUFFLE/NULL controls → 13 freeze primary verdict → 14 natural-quality exploratory audit → 15 optional GTF secondary → 16 visual atlas → 17 report → 18 result commit + push → 19 STOP.
