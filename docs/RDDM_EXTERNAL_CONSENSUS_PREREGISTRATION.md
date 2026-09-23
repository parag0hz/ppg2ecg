# RDDM-EXT — External consensus validation on the released RDDM checkpoint (preregistration)

Frozen and pushed before any number of this experiment is computed. Head at freeze: `54f5b72` (R0). No training, no new
sampler, no respacing, no checkpoint modification, no upstream code edit.

## Question
> Does an independently developed pretrained diffusion model exhibit the same multiple-sample functional-consensus
> behaviour and functional-error dependence, even though its sampling depth cannot be changed?

**This is not an external validation of the fixed-budget width-versus-depth result.** RDDM's released checkpoint admits a
single depth (T = 10; R0 showed the official loader rejects any other `nT`). Adding samples here *increases* the budget
(20 NFE per sample). Nothing in this experiment may be described as depth-versus-width evidence.

## 1. Model and inference (identical for every K)
- Official checkpoint (`data/pretrained/rddm/`, sha256 `e90490a9…0eb9` / `9f8ef5d7…7223` / `3c7cbbc1…c3bd`), official loader
  `load_pretrained_DPM(PATH, nT=10, type="RDDM")`, official sampler `RDDM.forward(mode="sample")`, submodule
  `external/RDDM` @ `7d53488` unmodified.
- Per sample: T = 10 steps, **20 denoiser evaluations** (10 `region_model` + 10 `eps_model`, hook-verified). The two
  condition encoders have no stochastic component (no dropout / sampling in `model.py`), so their output is identical for
  every sample of a window: **2 encoder passes per window** in principle; the driver recomputes them per draw (identical
  values; reported as an implementation detail). Total per window at K samples: **20 K denoiser NFE + 2 encoder passes**.
- Samples: **K ∈ {1, 2, 3, 4, 8, 16}, nested draws** — draw k ∈ {0, …, 15}; K uses draws 0 … K−1. Draw k of a window is
  produced by seeding torch's global RNG with `1_000_003·k + b` before batch b (batch 512, fixed window order), as in
  `scripts/rddm_driver.py`. Sampler, checkpoint and preprocessing do not depend on K.

## 2. Data
### Primary — VitalDB V1 test (external to RDDM)
- Exactly the V1 test windows: 1,156 patients, 1,224 cases, **19,543 windows** (same `case`, same `window_index`, same
  order as `vm1_evaluate.load("test")`). RDDM was trained on WESAD, DALIA, CAPNO, BIDMC and MIMIC-AFib; VitalDB is external
  by construction, so this evaluation has no training-subject contamination. It is a **zero-shot domain transfer** for RDDM.
- RDDM-style input construction (`scripts/rddm_build_vitaldb.py`): for each V1 window, the finite raw context of up to
  ±10 s around it (raw PLETH and ECG_II at 500 Hz, context start aligned to the 128-Hz grid), polyphase resampling to
  128 Hz, neurokit `ecg_clean(method="neurokit")` / `ppg_clean(method="elgendi")` on the context (the filters of RDDM's
  paper, as in R0), then the window's 512 samples are cut out. The paper's subject z-score is omitted **because RDDM's
  per-window min-max (applied by its official `get_datasets`) is invariant to any positive affine map applied before it**,
  so it cannot change the model input or target. The files are then read by the **official** `get_datasets`, which applies
  its per-window min-max to [−1, 1], `ppg_clean`, and `ecg_clean(pantompkins1985)`.
- **Reference functional T\*** = the project's V1 reference HR of the same window (neurokit peaks of V1's target ECG →
  `hr_bpm`; defined and in [30, 200] bpm for all 19,543 windows by V1's construction). It is the truth used by every other
  VitalDB experiment. Robustness reference: HR of RDDM's own (Pan-Tompkins-cleaned) target.

### Secondary — RDDM's original corpora (supporting only)
WESAD (15 subjects, 21,711 windows), CAPNO (42, 5,040), DALIA (15, 32,368), BIDMC (51, 6,120), built as in R0. **All
subjects included; ≈ 80 % are RDDM training subjects** (split unpublished). T\* = neurokit HR of RDDM's target. Used only as
supporting evidence and as extra conditions for the exploratory across-corpus ordering; never as a generalisation metric.

## 3. Functional and estimator
- `Y_k(c)` = HR of generated sample k on window c by the project's standard evaluator (`rpeaks.detect_rpeaks(·, 128)` →
  `hr_bpm`, 4 s windows); undefined when fewer than two peaks (NaN).
- `T̂_K(c) = median(Y_0, …, Y_{K−1})` over the finite values (`nanmedian`); undefined if none is finite.
  **At K = 2 the median equals the arithmetic mean of the two values** — stated here, before any number.
- Secondary functional (no criterion): RDDM's own HR extractor (official `metrics.ecg_bpm_array`, Hamilton segmenter;
  generated ECG with `filter=True`, reference unfiltered) on the same samples, median-pooled; −1 treated as undefined.

## 4. Metrics (patient-clustered, 2,000-replicate bootstrap, seed 20260911 — `v1_evaluate.cluster_ci`)
Per K (nested): **C_K** = patient-macro |T̂_K − T\*| · **I_K** = patient-macro mean over the K draws of |Y_k − T\*| ·
**G_K = I_K − C_K** · **coverage** = share of windows with a defined T̂_K · **win rate** = share of patients whose mean
error at K is lower than at K = 1 (common windows). Secondary estimate: partition average over disjoint draw groups
(K = 1, 2, 4, 8, 16 from 16 draws; K = 3 from 5 groups of draws 0–14).
**Paired contrasts are computed on the common window set where both estimates are defined**; coverage is reported beside
every contrast.

## 5. Criteria (all fixed now)
- **A-C1 (primary consensus):** `C_16 − C_1` 95 % CI upper < 0 **and** `G_16` 95 % CI lower > 0 (VitalDB).
- **A-C2 (robustness):** `C_K − C_1` CI upper < 0 for each K ∈ {3, 4, 8} (VitalDB).
- **A-C3 (aggregation, secondary; question: does our models' K = 2 anomaly and K ≥ 3 median advantage reproduce?):**
  (i) median ≡ mean at K = 2 (by construction — checked numerically); (ii) at K = 3, `median − mean` (partition estimate)
  patient-bootstrap CI upper < 0; (iii) the mean operator's doubling gains (1→2→4→8→16) are non-increasing (point
  estimates). Reported for every corpus; judged on VitalDB.
- **A-M1 (primary mechanism, within VitalDB, patient level):** eligible windows = T\* finite and all 16 sample HRs finite;
  eligible patients = ≥ 8 eligible windows. Per patient: `ρ̄_p` = mean over the 120 sample pairs of the Pearson correlation
  of the signed errors `e_k = Y_k − T*` across the patient's eligible windows (pairs with zero variance skipped); `G_p` =
  mean over the same windows of `mean_k |e_k| − |median_k Y_k − T*|` (K = 16). Criterion: **Spearman(ρ̄_p, G_p) across
  eligible patients has a 95 % patient-bootstrap CI (2,000 replicates, seed 20260923) entirely below 0.**
  *(This replaces difficulty strata: every stratification considered would have to be defined through the test target or
  RDDM's outputs, so none is used.)*
- **A-M2 (comparative, secondary):** on the same patients, Spearman(G_p, ·) for the within-window functional SD and MAD and
  for the waveform pairwise RMS of the 16 generated waveforms (mean over the patient's eligible windows); per-replicate
  `|r_err| − |r_wave|` with its 95 % interval; sign consistency of each.
- **A-M3 (exploratory, no inference):** across the 5 corpora as conditions (VitalDB + 4 contaminated), corpus-level ρ̄
  (EXP-B3 definition), G, SD, MAD, waveform RMS — the ordering is reported; Spearman over 5 points is descriptive only.
- **Instability flag (absolute external performance):** raised if, on VitalDB, (a) `C_1` is not lower than the error of
  PPG peak counting on the same windows against the same T\* (`find_peaks`, distance 42, prominence 0.3, on the RDDM input
  PPG; paired CI of `C_1 − PPG` not entirely below 0), or (b) coverage at K = 1 is below 0.90.

## 6. Verdict (partition, fixed now)
- **STRONG EXTERNAL CONSENSUS REPLICATION** — A-C1 ∧ A-C2 ∧ A-C3(i, ii) ∧ A-M1, and no instability flag.
- **FAILED** — A-C1 fails. Sub-label, reported explicitly: *harmful* if `C_16 − C_1` CI lower > 0; otherwise *no gain*.
  Mechanism direction is reported alongside and does not rescue a failed consensus.
- **PARTIAL** — every other outcome (e.g. A-C1 holds but A-M1 does not; or all criteria hold but the instability flag is
  raised). The report names which criterion or flag caused it.
Failure categories are kept distinct (method does not transfer / sampler limitation / reproduction / provenance /
dataset-split incompatibility); this experiment can only speak to the first.

## 7. Secondary, RDDM-internal only (no criterion)
Per sample on VitalDB (K-invariant): R-peak F1 @ 50 ms and RR-MAE against the peaks of RDDM's own target, waveform MAE /
RMSE against that target, FD (project KANFlow FD on the ≤ 3,000-window linspace subset, mean over draws) and RDDM's raw-vector
FD (batches of 512, fixed order). Latency from R0 (batch 1: 112 ms GPU, 408 ms CPU per sample; K samples cost K×) and NFE
per K. None of these is compared with the PENGUIN / iMF tables: preprocessing, reference waveform and model domain differ.

## 8. Outputs
Script `scripts/rddm_external_consensus.py`; raw per-sample arrays `outputs/rddm_ext_raw/` (gitignored); results
`artifacts/rddm_ext/`; report `docs/RDDM_EXTERNAL_CONSENSUS_REPORT.md`.
