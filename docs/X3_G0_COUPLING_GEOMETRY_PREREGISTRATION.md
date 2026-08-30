# X3-G0 Pre-registration — Coupling cost geometry / feasibility gate

Written 2026-08-30, **before any X3-G0 metric is computed on real data**. Frozen by the commit that introduces it.
X3-G0 is a **zero-deep-training resource gate**: no model is trained or fine-tuned, no checkpoint or historical artefact is written.
Frozen-checkpoint inference (A6, iMF-1, OT-CFM) is permitted only *after* this document is pushed.
Starting state: `main` = `origin/main` = `7e71a10`, clean; submodules `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7`;
one GPU (RTX 5090, 32,109 MiB usable), torch 2.11.0+cu130.

## 1. Question

Before training any coupling-modified flow model: does finite minibatch assignment create measurable source dependence in
**QRS/HF conditional-residual structure**, and is that dependence controlled primarily by **minibatch capacity** (batch size B) or by
the **geometry of the coupling cost**?

X0 established that the one-step ECG failure is genuine structural attenuation, not recoverable timing displacement. X2 established
that the frozen independent-coupling OT-CFM source endpoint almost completely cancels the Gaussian source
(√R_source 3–4 %, ρ_J 0.035–0.042, cos(J_x v·d, −d) ≈ +0.999). Neither established that source dependence buys QRS morphology.
X3-G0 measures the intervention itself before spending training compute on it.

## 2. Prior art — what is not novel

Non-independent source–target coupling is prior art: minibatch OT-CFM (Tong et al., arXiv:2302.00482), Multisample Flow Matching
(Pooladian et al., arXiv:2304.14772), and condition-aware coupling C²OT (Cheng & Schwing, *The Curse of Conditions*,
arXiv:2503.10636, ICCV 2025, official code `github.com/hkchengrex/C2OT`). C²OT reports that naive OT **harms** conditional
generation, and its validated condition types (class labels, a 1-D scalar, unit-norm CLIP text embeddings) do **not** establish a
raw 1024-sample PPG waveform cosine distance as a principled condition metric — one reason no C²OT arm and no raw-PPG cosine cost is
used in G0. Spectral preprocessing of a matching cost is **not** declared novel here; it is used as a diagnostic probe of cost
geometry, and any novelty claim would require its own focused audit. Also noted for the record: the official C²OT cosine path
normalises the condition **in place** after a no-op `type_as`, so with non-unit-norm conditions it can silently mutate the caller's
tensor; any future faithful adapter must clone or normalise out-of-place and document this as a bug fix.

## 3. Pre-preregistration design audit (disclosed)

`docs/X3_G0_PREPREREG_DESIGN_AUDIT.md` records, in full, two read-only checks taken **before** this protocol was frozen: one on
WildPPG **validation** ground truth (which selected the WHITE and HF arms) and one on WildPPG **test** arrays already published by
X0/X2 (which demoted the residual-OT arm). Consequences, binding on this pre-registration:

- Primary inference uses **subject-grouped cross-fitting over the 12 WildPPG training subjects**, not validation.
- Any validation run is **secondary design-informed confirmation**, never independent confirmation.
- The residual cost is **secondary**; no gate verdict depends on it.
- The final status report must disclose the test access as **YES during the pre-prereg design audit, NO during G0**, enforced by the
  §12 firewall. An unqualified "test accessed: NO" is forbidden.

## 4. No hard impossibility claims

`log₂(B!)/B` may be reported **descriptively** as "maximum permutation-index coding budget per assigned pair". It is not mutual
information, not conditional residual entropy, not an ECG information requirement, and is **excluded from every GO/STOP rule**.
`d_PR` is a descriptive effective dimension and likewise carries no threshold. Forbidden statements include
"d_PR > 40 proves minibatch OT cannot work", "4.63 bits are insufficient for ECG", "ECG needs hundreds of bits".

## 5. Data, pool, folds, firewall

**Split** — the frozen A4 manifest `data/manifests/split_a4_wildppg_seed42.json`:
train `e61, fex, l38, n31, ngh, p5d, p9p, qm9, trh, tz8, u7y, w4p` (293,271 windows); val `an0, k2s`; test `kjd, ssx`.

**Window pool (frozen).** Per subject, a deterministic uniform stride subsample to at most **4,096** windows, using the repository's
existing rule `stride = ceil(n / 4096)`, `x[::stride]` (the same idiom as `scripts/eval_a0_nfe_curve.py:103-105` with
`--subsample 4096`). Rationale: R-peak detection and A6 inference over all 293k windows is not affordable; the pool is frozen before
any result and is independent of outcome. Expected pool ≈ 12 × 4,096 = 49,152 train windows and 2 × 4,096 = 8,192 validation windows.

**Cross-fitting (frozen).** 4 folds, 3 held-out subjects each, assigned by manifest order:

| Fold | Held-out (3) | Fit (9) |
|---|---|---|
| 0 | e61, fex, l38 | the other nine |
| 1 | n31, ngh, p5d | the other nine |
| 2 | p9p, qm9, trh | the other nine |
| 3 | tz8, u7y, w4p | the other nine |

Everything fitted inside a fold — the whitening PSD, the residual PCA basis, the ridge map, the R² baseline — uses **fit subjects
only**. All windows of a subject belong to that subject's fold, so no target window can appear in both fitting and evaluation.
Primary G0 estimates pool out-of-fold results across all 12 training subjects.

**Test firewall.** Any G0 loader raises immediately if its subject list intersects `{"kjd", "ssx"}`. The provenance record lists
every subject loaded and the report states the firewall held.

## 6. Conditional-centre proxy and residual

`m(c)` = the frozen A6 WildPPG capacity-matched deterministic model `outputs/a6c_fullbackbone_mse_wildppg_seed42/checkpoint_best.pt`
(round 33, sha256 `b32820a2…`). No retraining. Residual `r = y − m(c)`, described as **"empirical residual relative to the frozen
deterministic conditional-centre proxy"** — never "true stochastic residual", never `m(c) = E[Y|C]`.
**Stated limitation:** A6 was itself trained on these 12 training subjects, so train-subject residuals are *in-sample* and are not an
out-of-sample estimate of true conditional-mean error. G0 is a coupling-geometry/resource gate, not a population uncertainty estimate.

## 7. Cost geometries (frozen transforms)

All costs are squared-L2 between an isotropic Gaussian source and a transformed target, solved by **exact** Hungarian assignment
(`scipy.optimize.linear_sum_assignment`; no Sinkhorn in primary G0). Sources `x0 ~ N(0, I_1024)`.

**Cost identity used throughout.** Σᵢ‖xᵢ − y_π(i)‖² = Σ‖x‖² + Σ‖y‖² − 2Σ⟨xᵢ, y_π(i)⟩, and the first two terms are permutation
invariant, so every assignment here is `argmax_π Σ⟨x₀ᵢ, φ(y)_π(i)⟩`. Consequences: a single positive global scale on the target is an
assignment **no-op**, and a large target norm does not by itself dominate the assignment. Any claim that DC or low-frequency energy
drives the assignment must be demonstrated through the regret/dependence diagnostics, not asserted from norms.

- **RAW** (primary): `φ_RAW(y) = y`, the canonical WildPPG ECG training target, unmodified. `C_RAW(i,j) = ‖x0ᵢ − y_j‖²`.
- **WHITE** (primary, *train-spectrum-whitened target cost* / *spectrally preconditioned matching cost* — never "optimal whitening
  transport"): per fold, on **fit subjects only**, with `yc = y − mean_time(y)`, `PSD_fit(f) = mean_samples |rFFT(yc)|²`,
  frozen floor `psd_floor = 1e-3 · median_{f>0} PSD_fit(f)`, `w(f) = 1/√(max(PSD_fit(f), psd_floor))`, **`w(0) = 0`** (DC excluded),
  then globally normalised so `mean_{f>0} w(f) = 1` (a conditioning convenience; a global positive scale cannot change the
  assignment). `φ_WHITE(y) = irFFT(w · rFFT(y − mean_time(y)))`.
- **HF** (primary): `φ_HF(y) = irFFT(1[f > 15 Hz] · rFFT(y − mean_time(y)))`, a brick-wall projection on the rFFT grid at
  fs = 128 Hz, T = 1024. The `> 15 Hz` threshold matches X0's structural HF convention and the pre-prereg design audit; the
  **brick-wall waveform projection itself is a new diagnostic primitive**, frozen here before results. No switch to Butterworth or
  any other filter after seeing outcomes.
- **RESID** (secondary only, B ∈ {64, 256}): `φ_RESID(y, c) = y − m_A6(c)`. No scalar rescaling (deleted as an assignment no-op).
  Not called "oracle-only": at training time `y` exists and a frozen `m(c)` is evaluable, so such a cost is implementable — it is
  secondary because it needs an extra pretrained model and adds a two-stage confound. **No gate verdict depends on RESID.**
- **B = 1** is the independent / no-assignment reference (identity pairing).

## 8. Assignment budget and seeds

`B ∈ {1, 8, 32, 64, 128, 256, 512}`; `target_pair_budget = 32768`; `n_batches(B) = max(32, ceil(32768 / B))`, applied identically to
the fit pool and the held-out pool of each fold; `assignment_seed = 20260830`. Batches are drawn without replacement within a batch
from the relevant subject pool. Counts are frozen and are not changed after viewing outcomes.

## 9. Manipulation check — cross-objective regret (required)

Assignment-index disagreement alone cannot distinguish a real geometry change from near-tie churn (the design audit found ~44–50 %
index change between two 96 %-aligned costs). For each analysed batch and each ordered pair of cost geometries q, p ∈ {RAW, WHITE, HF}
(and RESID against RAW at its two B values):

    Regret(q ← p) = [Cost_q(π_p) − Cost_q(π_q)] / [E_rand Cost_q(π_rand) − Cost_q(π_q) + ε]

with `E_rand` estimated from a frozen deterministic set of **`n_random_regret = 16`** random permutations per analysed batch.
Regret ≈ 0 means π_p is essentially optimal under q despite differing indices; substantial positive regret means the geometry
genuinely differs. The full 3×3 regret matrix and the assignment-overlap fraction `mean(π_p == π_q)` (secondary) are reported per B.
No universal threshold is imposed.

## 10. Residual representations and effective dimension

- **FULL**: `r = y − m(c)`.
- **QRS** (frozen implementation, masked full window — preferred over per-beat patches so every example stays one window and the
  source input stays the full `x0`): GT R-peaks from the frozen X0 detector (`ppg2ecg.evaluation.rpeaks.detect_rpeaks`, neurokit) on
  the **ground-truth** ECG only; mask = union of **±100 ms** around each GT R-peak (X0's frozen QRS half-width `QRS_HALF_MS = 100`);
  `r_QRS = mask ⊙ r`. The larger −0.25/+0.40 s morphology beat window is **not** used for the QRS residual.
- **HF**: `r_HF = irFFT(1[f > 15 Hz] · rFFT(r − mean_time(r)))`, the same `f > 15 Hz` convention as the HF cost. This is a residual
  **projection** and must not be conflated with X0's `hf_energy_ratio` structural metric.

Descriptive effective dimension for GT waveform, FULL, QRS-masked and HF residuals: participation ratio
`d_PR = (Σλ)² / Σλ²`, plus `d90` and `d95`. Fold-wise statistics are fitted on fit subjects only when used for held-out diagnostics;
an all-train descriptive summary is reported separately. **No feasibility decision follows from this table.**

## 11. Dependence diagnostic

Per fold / B / cost geometry, assignment pairs carry `(x0, assigned target window, its condition, its residual)`; the target ECG and
its PPG condition always move together. For each residual domain (FULL, QRS, HF):

- **PCA output basis** fitted on **fit-subject** residuals only, retaining 95 % variance capped at **`max_components = 128`**;
  the retained dimension is recorded. The basis never sees held-out subjects.
- **Ridge diagnostic map** `z ≈ A x0` on fit pairs, minimising `(1/n)‖Z − XA‖² + λ‖A‖²` with **`λ = 1e-3`** (implemented directly as
  `A = (XᵀX/n + λI)⁻¹ (XᵀZ/n)`; if a library with an unnormalised SSE objective were used, `alpha = n·1e-3`). λ is fixed a priori and
  is never tuned from morphology or from any outcome. This is a **diagnostic regression, not a generative model**.
- **Held-out R²** on the held-out subjects' assignment pairs, with the R² baseline being the **fit-subject** mean only; out-of-fold
  predictions are pooled across all 12 training subjects for the primary estimate. Reported as
  **source-to-residual linear predictive dependence** — never mutual information, capacity, entropy, or a sufficient statistic.
- **Permutation null**, `n_perm = 100`, frozen: within each fold/B/cost, the assigned residual identities are shuffled relative to
  `x0` while preserving the source marginal, the residual marginal and fold membership; the map is refitted and re-evaluated.
  `XᵀX` is invariant under output permutation, so it is factorised once and only the cross term is recomputed.
  Reported: observed R², null mean, null 2.5–97.5 %, and the primary quantities **ΔR²_FULL, ΔR²_QRS, ΔR²_HF = observed − null mean**.
- **Relative predictive dependence across structural subspaces** (never "information allocation"):
  `QRS_relative = max(ΔR²_QRS, 0) / max(ΔR²_FULL, ε)`, `HF_relative` likewise.

No nonlinear dependence estimator is pre-registered; none will be added post hoc.

## 12. Linear endpoint proxy and structural leverage

`F_proxy(x0, c) = m_A6(c) + A_FULL x0`, with `A_FULL` reconstructed from the FULL-residual PCA diagnostic map of the corresponding
fold/B/cost. Called **cross-fitted linear source-residual endpoint proxy**. It is explicitly **not** a population flow optimum, not
`E[Y|x0,c]`, not a strict upper bound, not a trained flow, not a deployable method; a weak proxy is a resource-allocation signal, not
an impossibility proof. Optional explanatory variants `m(c) + A_QRS x0`, `m(c) + A_HF x0` may be reported.

**Proxy source bank**: `K_proxy = 32`, seeds 0…31, identical across every B and cost arm.
**Proxy evaluation windows (frozen)**: a deterministic uniform stride subsample of **64 held-out windows per fold** (256 total across
the 4 folds), chosen by `round(linspace(0, n_ho − 1, 64))` over the fold's held-out pool in index order. Metrics are computed **per
generated sample first** (never averaging the 32 waveforms before scoring), then averaged over sources and windows.

**Structural metrics** — reuse the validated X0 primitives without changing their semantics. Primary: morphology correlation,
amplitude ratio, QRS energy retention, max-slope retention, HF energy ratio. Secondary/descriptive: R-peak F1, RMSE, MAE, PCC.
**There is no F1 veto and no conditioning-gain veto in G0.**

**Reference models** (frozen-checkpoint inference generated after this pre-registration is pushed, on the same 256 proxy windows):
A6, iMF-1, OT-CFM-50 (Heun-25, 50 NFE). Because these subjects participated in training, the references are **descriptive in-sample**
values and are never presented as generalisation performance.

**Structural recovery fraction**, for a higher-is-better metric Q: `Recovery_iMF(Q) = [Q(proxy) − Q(A6)] / [Q(iMF1) − Q(A6)]`,
reported only when the denominator is stably positive and **N/A** otherwise; values are never clipped. Raw metric deltas are always
reported alongside, and the continuous fraction matters more than crossing any binary threshold.

## 13. Uncertainty

Windows are correlated. Held-out summaries use the fold's subject structure; where site labels are available and valid, clustering is
by `(subject, site)` via the deterministic `src/ppg2ecg/data/wildppg_sites.py` helper (fail-loud). Windows are never treated as
independent subjects. Fold-to-fold spread is reported for every headline quantity. G0 uses one A6 checkpoint and one frozen protocol;
no training-seed uncertainty is estimated and none is claimed.

## 14. Gate verdicts (resource gate, not a theorem)

Evaluated over the tested feasible range **B ≤ 512**, on the primary train-subject cross-fitted estimates.

- **RAW-COUPLING GO** — some B ≤ 512 has (ΔR²_QRS ≥ 0.05 **or** ΔR²_HF ≥ 0.05) for RAW, above its permutation-null interval, **and**
  the RAW proxy recovers ≥ 20 % of a stable A6→iMF-1 gap in **morphology** **and** in ≥ 1 of {QRS energy, slope, HF energy}.
- **COST-GEOMETRY LIMITED** — RAW stays weak (ΔR²_QRS < 0.02 **and** ΔR²_HF < 0.02 for all B ≤ 512) **but** WHITE or HF reaches
  ΔR²_QRS ≥ 0.05 or ΔR²_HF ≥ 0.05 with clearly non-zero cross-objective regret versus RAW.
- **SPECTRAL-COUPLING CANDIDATE** — COST-GEOMETRY LIMITED **and** the WHITE or HF proxy additionally recovers ≥ 20 % of a stable
  A6→iMF-1 gap in morphology **and** in ≥ 1 of {QRS energy, slope, HF energy}.
- **WEAK FINITE-BATCH LEVER UNDER TESTED RANGE** — for every B ≤ 512, RAW, WHITE and HF all have ΔR²_QRS < 0.02 **and**
  ΔR²_HF < 0.02, and the structural proxies recover < 10 % of stable A6→iMF-1 gaps.
- **INCONCLUSIVE** — regret is large but R² is ambiguous; nonlinearity may dominate; the structural proxy strongly disagrees with the
  dependence metrics; fold heterogeneity is severe; or only the secondary validation run looks strong.

A GO verdict does **not** start any training; it only licenses a separately pre-registered causal training experiment.

## 15. Secondary validation confirmation

After all primary train-crossfit analysis is complete, the identical frozen protocol is run on `an0, k2s`, with the whitener fitted on
the 12 training subjects only (never refitted on validation) and with no adjustment of B or cost choices. Labelled
**"secondary design-informed validation confirmation"**, never independent confirmation, because validation ground truth was used for
arm selection in the pre-prereg design audit.

## 16. Artefacts, implementation, scope

`artifacts/x3_g0_coupling_geometry/`: `protocol.json`, `provenance.json`, `preprereg_design_audit.json`, `fold_mapping.json`,
`residual_dimension.csv`, `assignment_dose_response.csv`, `assignment_overlap.csv`, `cross_objective_regret.csv`,
`source_residual_r2.csv`, `permutation_null.csv`, `structural_proxy_metrics.csv`, `recovery_fraction.csv`,
`secondary_validation.csv`, `gate_summary.json`, `figures/`. Large intermediate arrays go to `outputs/x3_g0_coupling_geometry/`
(git-ignored). No test arrays are produced. No historical artefact is modified.
Implementation: `src/ppg2ecg/evaluation/coupling_geometry.py`, `scripts/analyze_x3_g0_coupling_geometry.py`,
`tests/test_x3_g0_coupling_geometry.py`. Existing X0/X2 result implementations are not modified; any extracted primitive gets a
parity test.

Required unit tests (synthetic; run before real metrics): B=1 identity; known synthetic Hungarian optimum; ECG–condition pairing
preserved under permutation; source marginal unchanged; squared-L2 assignment equals maximum-inner-product assignment; positive global
target scaling is an assignment no-op; RAW cost parity; whitener fitted on fit subjects only; `w(0) = 0`; deterministic PSD floor;
HF transform uses exactly `f > 15 Hz`; regret 0 for identical optima; synthetic near-tie gives low overlap with low regret; synthetic
genuinely different geometry gives low overlap with substantial regret; PCA fit-subject only; a target window cannot cross folds;
normalised ridge λ convention; independent synthetic coupling gives ΔR² ≈ 0; known linear coupling gives positive held-out R²;
permutation null destroys the relationship; QRS mask is exactly ±100 ms around GT R-peaks; loading `kjd`/`ssx` raises;
residual global scaling is an assignment no-op; historical files are not overwritten.

**Scope.** X3-G0 stops after its report. No X3 training, no C²OT training, no reflow, no respiration, no ABP, no new seeds, and no
new architecture is started as part of X3-G0. B up to 512 is an assignment-geometry analysis only and does not imply a future
gradient batch of 512; a future experiment at B > 64 would need the coupling pool decoupled from the gradient minibatch, and G0
merely records that implication.

## 17. Language

Allowed if supported: "the raw waveform cost and spectrally preconditioned costs induce materially different finite-batch assignment
geometries"; "only spectrally preconditioned matching induces measurable source dependence in QRS/HF residual structure"; "the tested
finite minibatch assignments produce too little measurable QRS/HF residual leverage to justify expensive flow training".
Forbidden: "minibatch coupling is mathematically impossible"; "we invented frequency-aware OT"; "this proves optimal coupling should
whiten ECG"; "spectral coupling solves one-step ECG generation"; "C²OT has been disproven for PPG"; "4.63 bits are insufficient for
ECG"; "d_PR proves minibatch OT cannot work"; "the linear proxy is a strict upper bound"; calling any R² a mutual information.
