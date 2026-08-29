# X2 Pre-registration — Endpoint barycenter identity and source-noise cancellation in one-step conditional flow matching

Written 2026-08-30 **before any X2 metric is computed on real data**. Frozen by the commit that introduces it. X2 is an
ANALYSIS experiment: no training, no fine-tuning, no checkpoint is created, modified or regenerated. New inference from frozen
checkpoints is required and is performed only after this document is committed and pushed. Starting state: `main` = `origin/main` =
`5bf255b`, working tree clean, submodules `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7` (verified 2026-08-30).

## 1. Question

Does the **finite trained** conditional OT-CFM field actually realise the known source-endpoint barycentric degeneracy of
independently coupled flow matching? Concretely, for the frozen PENGUIN OT-CFM models: (i) is the one-step map from the source
endpoint nearly invariant to the source noise, (ii) how close is its source-averaged endpoint to deterministic and to same-model
multistep conditional-center proxies, (iii) how fast does source dependence reappear away from the endpoint, and (iv) does iMF-1
behave the same way?

## 2. Prior art — the identity is NOT novel (established before any X2 result)

Under an **independent source–target coupling** (x0 ~ N(0, I) drawn independently of x1) with the **linear conditional path**
x_t = (1−t)x0 + t·x1 and squared-error velocity regression, the population-optimal field is the conditional expectation
v*(x, t) = E[x1 − x0 | x_t = x]. At t = 0 the state is x_t = x0, so with x0 ⟂ (x1, c),

    v*(x0, 0, c) = E[x1 − x0 | x0, c] = E[x1 | c] − x0     ⇒     F*(x0, c) := x0 + v*(x0, 0, c) = E[x1 | c].

This is a **direct conditional corollary of known flow-matching / rectified-flow endpoint conditional-expectation results under
independent source–target coupling**, not a new theorem. Verified primary sources (fetched and quoted 2026-08-30):

| Source | Where | What it states |
|---|---|---|
| Frans, Hafner, Levine, Abbeel 2024, *One Step Diffusion via Shortcut Models*, arXiv:2410.12557 | §2 "Few-step ambiguity", Eq. (2), Fig. 2 caption | "At t = 0 the model receives pure noise as input and (x0, x1) are randomly paired during training, so the predicted velocity at t = 0 points towards the dataset mean. Thus, even at the optimum of the flow matching objective, one step generation will fail for any multi-modal data distribution." |
| Albergo, Boffi, Vanden-Eijnden 2023, *Stochastic Interpolants*, arXiv:2303.08797 | Remark 42, Eq. (4.12) | b(0, x) = α̇(0)x + β̇(0)E[x1 \| x0 = x] − …; for α = 1−t, β = t this is exactly b(0, x0) = E[x1 \| x0] − x0. |
| Albergo & Vanden-Eijnden 2022, arXiv:2209.15571 | App. B, Eqs. (B.18)–(B.19) | v0(x) = ȧ0·x + ḃ0∫x1ρ1(x1)dx1, i.e. E[x1] − x for the linear interpolant. |
| Liu, Gong & Liu 2022, *Flow Straight and Fast*, arXiv:2209.03003 | Eq. (2); distillation paragraph; Fig. 5 caption | v(x,t) = E[X1 − X0 \| X_t = x]; one-step update Z1 = Z0 + v(Z0, 0); "it generates the mean of π1 when simulated with a single Euler step". |
| Lee et al. 2024, *Improving the Training of Rectified Flows*, arXiv:2405.20320 | §2, §4.1 | "a model learns to simply output the dataset average when t = 1 [their noise end] … The meaningful part of the training thus happens in the middle of the interval"; the CFM optimum is the MMSE/conditional-expectation estimator. |

Closely related but **not** containing the t = 0 statement: Lipman et al. 2022 (arXiv:2210.02747), Tong et al. 2023
(arXiv:2302.00482), Lipman et al. 2024 *FM Guide and Code* (arXiv:2412.06264), MeanFlow (arXiv:2505.13447) and Improved MeanFlow
(arXiv:2512.02012), Flow Map Matching (arXiv:2406.07507), Consistency Models (arXiv:2303.01469) — these give the marginal-field =
conditional-expectation machinery and the curvature/straightness motivation. Salimans & Ho 2022 (arXiv:2202.00512) is the
diffusion-side origin of the "one step = blurry average" observation. None of the sources checked writes the **conditional-on-c**
form; that restatement is a one-line corollary and is presented as such, never as novelty. Not exhaustively searched: conditional /
image-to-image flow-matching literature — absence there is not asserted.

**What X2 can contribute empirically** (subject to the results): a controlled, pre-registered verification on a real *conditional
physiological* task of whether a finite trained model realises this behaviour, quantified against an independently trained
deterministic conditional-mean proxy and a same-model multistep barycenter proxy, plus the rate at which source dependence returns
off the endpoint, and a same-bank iMF-1 contrast.

## 3. Verified implementation facts (read-only audit, exact references)

**A. Independent source–target coupling.** Training runs through the unmodified upstream class:
`external/PENGUIN/src/models/PENGUIN.py:225-230` — `timestep = torch.rand(B, 1)`, `x_0 = torch.randn_like(x_1)`,
`x_t = (1 - timestep)·x_0 + timestep·x_1`, `self.dx_t = x_1 - x_0`; loss `F.mse_loss(self.pred_dx_t, self.dx_t)` (L261). Fresh
per-sample Gaussian source, **no minibatch-OT / Sinkhorn / assignment anywhere** (grep negative in `src/`). Restated bit-exactly in
`src/ppg2ecg/flow/cfm.py:18-31` (parity unit-tested in `tests/test_upstream_parity.py:44-58`). The repository name "OT-CFM" denotes
the Lipman et al. *OT conditional path* (linear / rectified-flow interpolant), **not** Tong et al. minibatch-OT coupling.

**B. Linear path with σ_min = 0.** x_t = (1−t)x0 + t·x1, u_t = x1 − x0; no σ_min term exists in the code, so x_{t=0} = x0 exactly.

**C. Euler 1-NFE is exactly the source-endpoint map.** `src/ppg2ecg/flow/samplers.py:57-67` with `_time_grid` (L35-36)
`linspace(0, 1, n_steps+1)`: for n_steps = 1 the grid is [0.0, 1.0] (float32 exact zero), the network is evaluated **once** at
t = 0 with x_t = x0, and the update is `x_t + 1.0·v`. Therefore the frozen `euler1.npz['pred']` **is** F_OT(x0, c) = x0 + v_θ(x0, c, 0).
Velocity call: `model.forward_step(x, ppg.unsqueeze(1), t)` (`scripts/eval_a0_nfe_curve.py:58`;
`external/PENGUIN/src/models/PENGUIN.py:197-209`). Time embedding at t = 0 is the sinusoidal embedder
(`PENGUIN.py:24-36`), finite and well defined ([cos(0)]×128 ++ [sin(0)]×128 → MLP); t enters raw in [0,1], no scaling; no RevIN, no
EMA, no target-norm (identity for all ECG runs).

**D. The source bank is deterministically reproducible.** `scripts/eval_a0_nfe_curve.py:108-109`:
`g = torch.Generator().manual_seed(args.noise_seed)`; `x0_all = torch.randn(n, 1, T, generator=g)` — a dedicated **CPU** generator,
float32, shape [n, 1, 1024], drawn once and sliced for every arm in window order (batch 64). `--noise-seed 0` in every pipeline.
Cross-check available: `derangement(n, 1)` (numpy `default_rng(1)`) must equal the stored `perm`.

**E. iMF-1 uses the numerically identical noise tensor but the opposite time convention.** `scripts/eval_a2.py:97-98` draws `e_all`
with the same generator/seed/shape, so e ≡ x0 per window index. But `src/ppg2ecg/flow/imeanflow.py:7-13, 155-167`: **t = 1 is noise,
t = 0 is data**, z_t = (1−t)x + t·e, and 1-NFE sampling is `x̂ = e − u_θ(e, ppg, t = 1, h = 1)` — a *minus* sign and a
trained **average**-velocity over the whole interval, conditioned on h = t − r = 1 via `cond = E(h)` (`imeanflow.py:50-51`,
`cond_mode = "h_only"` in all three checkpoints). iMF is therefore a genuine one-step map by construction, evaluated at its own
operating point; it is **not** the same mathematical object as v_θ(·,·,t=0) and no identity is imposed on it.

**F. Endpoint training mass (required wording).** Timesteps are drawn `torch.rand(B,1)` ~ U[0,1). *The endpoint t = 0 carries
essentially no explicit training mass under uniform timestep sampling and is not intentionally endpoint-supervised. Evaluation at
exactly t = 0 therefore probes a boundary value inferred primarily from near-zero training samples.* (Exact zero is representable by
the finite-precision RNG, so no absolute "never sampled" claim is made.) This is a stated limitation of the finite-model test, not of
the population identity.

**G. The MSE proxy is a different network, not `x_const + v`.** `src/ppg2ecg/models/regressor.py:93-106`: `S5FullBackboneRegressor`
returns `forward_step(0.1·ones, ppg, t = 0.5)` with `cond = 0.05·E(0.5)` **directly as the ECG**. Comparisons between F̄ and
M_A6 are cross-network similarity tests, never identity checks, and M_A6 is never called the true conditional mean.

## 4. Frozen datasets and conditions (audited counts, no new split)

| Condition | Split manifest | Test inputs (frozen, sha256 in provenance) | N | Clusters |
|---|---|---|---|---|
| WildPPG (original window-normalised protocol, A4/A6c) | `data/manifests/split_a4_wildppg_seed42.json` | `outputs/a4_otcfm_wildppg_seed42/predictions/test_inputs.npz` | **3907** | 8 = {kjd, ssx} × {sternum, head, wrist, ankle}, sizes 474–503 |
| PPG-DaLiA test S2 | `data/manifests/split_p0_holdout_seed42.json` | `outputs/a6a_fullbackbone_mse_dalia_testS2_seed42/predictions/test_inputs.npz` | **1025** | 1 (single subject) |
| PPG-DaLiA test S1 | `data/manifests/split_a3_testS1_valS11.json` | `outputs/a6b_fullbackbone_mse_dalia_testS1_seed42/predictions/test_inputs.npz` | **1151** | 1 (single subject) |

Window order and preprocessing are taken verbatim from these frozen `test_inputs.npz` files (the same arrays X0 used). The A9
global-z representation is **not** part of X2. WildPPG cluster labels are reconstructed by a deterministic repository helper
(`src/ppg2ecg/data/wildppg_sites.py`): concatenate `site` from `data/processed/wildppg_8s/{kjd,ssx}.npz` in manifest test order and
apply the same stride-12 subsample (46,884 → 3,907); the helper **asserts** N, the `window_start_s` equality with `test_inputs['starts']`,
the label set and the 8 non-empty clusters, and **raises** otherwise. Silent fallback to subject-only or two clusters is forbidden;
if the metadata cannot be reconstructed exactly the analysis fails loudly and stops.

## 5. Frozen model arms (no training; checkpoints read-only)

| Arm | WildPPG | DaLiA S2 | DaLiA S1 |
|---|---|---|---|
| OT-CFM (primary) | `outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt` (round 189) | `outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42/…` (64) | `outputs/a3_otcfm_ppgdalia_testS1_seed42/…` (93) |
| iMF (comparator) | `outputs/a4_imeanflow_wildppg_seed42/…` (45) | `outputs/a2_imeanflow_s5_ppgdalia_8s_seed42/…` (60) | `outputs/a3_imeanflow_ppgdalia_testS1_seed42/…` (15) |
| A6 full-backbone MSE (deterministic comparator) | `outputs/a6c_fullbackbone_mse_wildppg_seed42/…` (33) | `outputs/a6a_…` (23) | `outputs/a6b_…` (5) |

sha256 of every checkpoint is recorded in `artifacts/x0_error_decomposition/prediction_provenance.json` and re-recorded in the X2
provenance. `OT50` = the *same* OT-CFM checkpoint sampled with the canonical Heun 25-step sampler (**50 NFE**). For the MSE model,
source-noise sensitivity is reported as **"N/A — deterministic model has no source latent"**, never as a measured zero.

## 6. Frozen source-noise bank

K = **32** source seeds per condition, seeds **0 … 31**. Bank construction, per seed k, on CPU, float32:

    g = torch.Generator().manual_seed(k);  x0_k = torch.randn(N, 1, 1024, generator=g)

Seed 0 therefore reproduces the historical frozen evaluation bank **exactly** (identical generator type, seed, shape, dtype, order),
which gives a parity check: F_0 must reproduce `euler1.npz['pred']` up to GPU/batching float32 tolerance. The same bank is used for
iMF-1 (its `e`). Banks are regenerated on demand, never stored in full; provenance records the seeds, shapes, dtype, construction
code path, and a sha256 of each bank's first-window slice plus of the seed-0 bank. Inference is chunked over windows and looped over
seeds with the historical batch size 64; ordering is preserved exactly. `torch.no_grad()`, `.eval()`, float32, CUDA.

## 7. Primary map and confirmatory quantities

    F_k(c_i) = x0_{ik} + v_θ(x0_{ik}, c_i, t = 0),     t passed as a float32 zeros tensor [B, 1]
    F̄(c_i)  = (1/K) Σ_k F_k(c_i)

Reduction order (frozen): for window i, with unbiased variance over the K sources (ddof = 1) at each time sample τ, then the mean
over τ:

    V_end(i) = mean_τ Var_k[F_k(i, τ)]           V_src(i) = mean_τ Var_k[x0_k(i, τ)]
    R_source(i) = V_end(i) / V_src(i)            std-retention(i) = sqrt(R_source(i))

**Primary summary = median over test windows**; IQR and mean secondary; clustered CI per §10. `R_source` (variance) and
`sqrt(R_source)` (amplitude/std) are always reported as a pair and never conflated (e.g. a std retention of 3.5 % is a variance
retention of ≈ 0.12 %).

**Pairwise normalised endpoint displacement** (descriptive companion, not an independent proof) over all K(K−1)/2 = 496 pairs:

    D_pair(i) = mean_{a<b} RMSE_τ(F_a, F_b) / mean_{a<b} RMSE_τ(x0_a, x0_b)

**Source-retention slope** (first-order projection of endpoint variability onto the source direction):

    β(i) = Σ_k ⟨F_k − F̄, x0_k − x̄0⟩ / Σ_k ‖x0_k − x̄0‖²

β ≈ 0 ⇒ no first-order source retention; β ≈ 1 ⇒ identity-like retention (v ≡ 0). β estimates the normalised trace of the endpoint
Jacobian, whereas ρ_J (§8) estimates its RMS singular response; under local linearity ρ_J ≥ |β| and R_source ≈ ρ_J², so the three
quantities are related but not redundant, and their disagreement diagnoses nonlinearity over the noise scale.
**E_cancel is deliberately NOT used**: (v_a − v_b) + (x0_a − x0_b) ≡ F_a − F_b, so it is algebraically redundant with D_pair.

## 8. Local Jacobian source-cancellation diagnostic (frozen before results)

The population endpoint behaviour implies J_x F ≡ 0, equivalently J_x v ≡ −I. Measured with forward-mode JVP
(`torch.func.jvp`, validated 2026-08-30 against central finite differences on an untrained randomised backbone: relative error
1.5 × 10⁻² at ε = 10⁻², bit-exact on rerun):

    ρ_J(d) = ‖d + J_x v_θ(x0, c, 0)·d‖₂ / ‖d‖₂     (= ‖J_x F·d‖ / ‖d‖)

Frozen design: **64 windows per condition**, cluster-stratified (WildPPG: 8 per cluster at positions `round(linspace(0, m−1, 8))`
within each cluster's test indices sorted ascending; DaLiA: `round(linspace(0, N−1, 64))`); **4 directions per window**; direction
RNG **seed 20260301** (one CPU generator, drawn in window-major then direction-major order); each d normalised to unit RMS
(‖d‖₂ = √T); evaluated at the single source **x0 from seed 0**. Reported: median, mean, IQR, clustered bootstrap CI. Secondary
(not decisional): cos(J_x v·d, −d). The same protocol is applied to iMF-1 at its own operating point,
ρ_J^{iMF} = ‖d − J_e u_θ(e, c, t=1, h=1)·d‖ / ‖d‖ (note the minus sign of its one-step rule).

## 9. Conditional-center comparisons

**Comparator 1 — deterministic proxy M_A6** (cross-network approximation only; never "ground-truth E[x1|c]"). Per window:
RMSE and PCC of F̄ and of each F_k against M_A6, plus reference distances to GT.

    Q_MSE(i) = RMSE_τ(F̄_i, M_i) / median_k RMSE_τ(F_{k,i}, M_i);   Q_MSE = median over windows

**Pre-inference amendment 1 (stated here, before any result, and never revised afterwards).** The originally proposed rule
"H-X2-MEAN passes if Q_MSE < 0.80" is **logically self-defeating given H-X2-CANCEL** and is therefore demoted to a descriptive
statistic with a pre-stated prediction. Reason (algebra, using only *prior-observed context* — the published A4/A0-b/A3 seed-diversity
std 0.024–0.035 and the published A6 OT1↔MSE RMSE 0.078/0.082/0.120, both labelled as prior context, not blind thresholds): writing
F_k = m + ε_k with source-noise scale σ and systematic offset ‖m − M‖ = δ, we have median_k RMSE(F_k, M) ≈ √(δ² + σ²) and
RMSE(F̄, M) ≈ √(δ² + σ²/K), so Q_MSE ≈ √((δ² + σ²/32)/(δ² + σ²)). Source cancellation means σ ≪ δ, which forces Q_MSE → 1. Passing
H-X2-CANCEL therefore *guarantees* failing "Q_MSE < 0.80". Q_MSE is consequently reported with the advance prediction **Q_MSE ≈ 1
under successful cancellation; a Q_MSE materially below 0.80 would contradict H-X2-CANCEL**, and the confirmatory H-X2-MEAN test is
replaced by the comparative-closeness rules in §11, which test the intended scientific claim ("the source-averaged one-step endpoint
sits at a deterministic conditional center") without being an artefact of the noise magnitude.

**Comparator 2 — same-model multistep barycenter proxy B50.** On the *same frozen 64-window subset* as §8, with the *same* OT-CFM
checkpoint and the canonical Heun-25 (50 NFE) sampler, K50 = **8** source samples, seeds **0 … 7**:

    B50(c) = (1/8) Σ_{k=0}^{7} OT50(x0_k, c)

Because B50 is a Monte-Carlo mean of 8 samples it carries MC error; distances to it are therefore also reported **debiased**:

    d²_debiased(A, B50) = mean_τ (A − B50)² − V50/K50,   V50 = mean_τ Var_k[OT50_k] (ddof = 1)

Reported: RMSE and debiased distance of F̄, of M_A6 and of F_0 to B50; PCC(F̄, B50) (WildPPG primary, DaLiA descriptive). B50 is
called **"same-model multistep barycenter proxy"** everywhere — never "ground truth", "oracle" or "the conditional expectation".

## 10. Uncertainty

WildPPG: cluster bootstrap over the **8 (subject, site) clusters**, **2000** resamples, **seed 0**, resampling clusters with
replacement and pooling their windows; 95 % percentile CIs for every headline quantity and for paired model differences. DaLiA:
each condition is a **single held-out subject**, so a subject-level bootstrap is impossible; a window-level bootstrap (2000, seed 0)
is reported as a **descriptive within-subject** interval that explicitly does **not** estimate cross-subject generalisation. No
p-values. Bootstrap intervals do not capture between-training-seed uncertainty: every checkpoint is a single training seed (42).

## 11. Frozen hypotheses and decision rules

**H-X2-CANCEL** — OT-CFM t = 0 source dependence is strongly suppressed. Per condition, all three must hold:

    (C1) median_i R_source(i) < 0.05      (C2) median_i β(i) < 0.25      (C3) median ρ_J < 0.25

These are the thresholds as specified; they are mathematically well posed (each is a dimensionless ratio that is 0 for a constant
map and 1 for the identity map) and are **not** revised. Transparency note stated in advance: given the published prior context
(seed-diversity std 0.024–0.035 at 1 NFE, i.e. R_source ≈ 0.0006–0.0012 if that scale carries over), C1 is a *permissive* bar and is
expected to pass; C2 and C3 have **no prior measurement** in this project and are the informative criteria. Passing is therefore not
evidence of a surprising discovery; failing would be strong evidence against the finite model realising the identity.

**H-X2-MEAN** (revised per §9 amendment 1) — the source-averaged one-step endpoint sits at a deterministic conditional center:

    (M1) RMSE(F̄, M_A6) < RMSE(F̄, X) for every X ∈ {OT50 single sample (seed 0), iMF-1 (seed 0), GT}, in ≥ 2 of 3 conditions
    (M2) WildPPG only: PCC(F̄, M_A6) ≥ 0.60

DaLiA PCC is descriptive only and never pass/fail (near-flat outputs make waveform PCC unstable there). Q_MSE is descriptive with
the §9 advance prediction. The 0.078/0.082/0.120 A6 distances are **prior-observed context**, never blind thresholds.

**H-X2-IMF-DIFF** — descriptive only, no pass/fail: compare R_source, sqrt(R_source), D_pair, β and ρ_J of iMF-1 against OT-1 on the
identical bank. It is not pre-registered that iMF "must" retain source dependence.

**Overall verdict.** STRONG SUPPORT: H-X2-CANCEL passes in 3/3 conditions **and** H-X2-MEAN passes (M1 in ≥ 2/3 and M2 on WildPPG)
**and** no implementation/provenance failure. PARTIAL SUPPORT: cancellation clear in ≥ 2/3 but the conditional-center comparison is
mixed, or one diagnostic (R_source / β / ρ_J) materially disagrees with the others. NOT SUPPORTED: substantial source dependence
remains in ≥ 2/3 conditions under the frozen metrics. The B50 comparison, the t-profile and the iMF contrast are
supporting/exploratory and do **not** determine the verdict.

## 12. Exploratory t > 0 oracle path-state profile (never confirmatory)

Times **t ∈ {0.00, 0.01, 0.05, 0.10}**, on the frozen 64-window subset of §8 with the full K = 32 bank. The path state uses the
**ground-truth target**, x_t = (1−t)x0 + t·x1, so every table, figure and sentence reporting these numbers is labelled
**"ORACLE PATH-STATE DIAGNOSTIC"**; these are *not* realisable one-step generation states. One single coherent definition is frozen
here — the Euler-to-endpoint estimate from the path state and its source perturbation:

    G_t(x0_k) = x_t,k + (1 − t)·v_θ(x_t,k, c, t),     x_t,k = (1 − t)x0_k + t·x1,     s_k = (1 − t)x0_k
    R_source(t) = median_i [ mean_τ Var_k G_t / mean_τ Var_k s_k ]
    β(t)        = median_i [ Σ_k ⟨G_{t,k} − Ḡ_t, s_k − s̄⟩ / Σ_k ‖s_k − s̄‖² ]
    ρ_J(t)      = median [ ‖d + (1 − t)·J_{x_t} v_θ(x_t, c, t)·d‖₂ / ‖d‖₂ ]   (same 4 directions, seed 20260301, x0 = seed 0)

At t = 0 all three reduce **exactly** to the confirmatory definitions of §7–8. The purpose is solely to quantify how sharply the
t = 0 behaviour changes in the near-source region. No pass/fail threshold; no t value is added or removed after seeing results.

## 13. Qualitative windows (frozen before any X2 output)

WildPPG **[2439, 297, 415]** (reused verbatim from the X0 pre-registration), DaLiA S2 **[880, 482, 824]**, DaLiA S1 **[1067, 726, 221]**.
Figures use these indices only; no post-hoc example selection.

## 14. Artefacts, implementation, and outputs

`artifacts/x2_endpoint_identity/`: `summary.json`, `provenance.json`, `source_sensitivity.csv`, `jacobian_sensitivity.csv`,
`conditional_center_similarity.csv`, `same_model_barycenter.csv`, `t_profile_oracle.csv`, `clustered_bootstrap.csv`, `figures/`.
Large arrays (F̄ per condition; the K = 32 endpoints and B50 on the 64-window subset) go to `outputs/x2_endpoint_identity/` with
`.gitignore` rules added **before** inference; per-seed full-test tensors are never materialised or committed. **No existing
prediction, checkpoint, artifact or output file is overwritten** — X2 writes only to new paths.
Implementation: `src/ppg2ecg/evaluation/source_sensitivity.py` (pure metric functions), `src/ppg2ecg/data/wildppg_sites.py`
(fail-loud cluster labels), `scripts/analyze_x2_endpoint_identity.py` (driver), `tests/test_x2_endpoint_identity.py`.
Provenance records: repo SHA, pre-registration SHA, checkpoint paths + sha256, dataset/split, N, source seeds, direction seed,
bootstrap seed, chunk/batch sizes, device, torch/CUDA versions, dtype, sampling convention, exact NFE per arm, OT50 NFE semantics,
timestamps, script path + sha256.

Required unit tests (synthetic; run before real inference): (A) analytical field v = m(c) − x gives F ≡ m(c), R_source ≈ 0, β ≈ 0;
(B) v ≡ 0 gives F = x0, R_source ≈ 1, β ≈ 1; (C) ρ_J ≈ 0 for a constant map and ≈ 1 for the identity map, JVP validated against
finite differences; (D) deterministic bank reproducibility (identical across reruns; seed 0 bank matches the historical
construction); (E) no cross-window mixing (per-window statistics are unaffected by permuting other windows); (F) exact t = 0 call
semantics (t is a float32 zeros tensor, one network evaluation, `nfe == 1`); (G) controlled equivalence of x0 + v_θ(x0,·,0) with
`euler_sample(..., n_steps=1)` on a small tensor set (tolerance: bit-exact on identical batching; ≤ 1e-4 max abs vs the frozen
`euler1.npz` given GPU/batching float32 differences — bit-exact GPU parity is **not** claimed); (H) t-profile formula on an exact
synthetic straight field v = x1 − x0 gives G_t ≡ x1 and R_source(t) ≈ 0 for all t; (I) cluster-label reconstruction returns the 8
expected clusters and **raises** when metadata is missing or inconsistent.

## 15. Allowed and forbidden claims

Allowed: "known endpoint-barycenter degeneracy"; "conditional-mean-like attenuation"; "empirically realises / approximates the known
endpoint identity"; "consistent with source cancellation predicted by the population objective"; "finite trained model";
"same-model multistep barycenter proxy". Forbidden: "we discovered that flow matching converges to the conditional mean"; "we prove
OT-CFM mathematically converges to E[x1|c]"; any claim that multimodality is proven, that ECG is inherently multimodal, that
changing the coupling would fix the failure, that iMF is theoretically immune to conditional averaging, that the finite network
equals the population optimum, or that source-noise cancellation alone proves conditional-distribution collapse. Also carried over:
the X0/A5 forbidden list (phase modes, "transport is necessary", "timing uncertainty causes collapse", conditional-mean theorem).
Context that must be retained: A7/A8 showed on ABP that a deterministic conditional predictor can be an *excellent* solution —
conditional averaging is pathological only when it suppresses task-relevant target structure.

## 16. Scope

X2 stops after the report. No coupling experiment, no respiration experiment, no synthetic-ambiguity experiment, no new
architecture, no retraining, no A9-representation extension is started as part of X2.
