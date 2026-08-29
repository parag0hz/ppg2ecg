# X2 Report — Endpoint barycenter identity and source-noise cancellation in one-step conditional flow matching

Pre-registration: `docs/X2_ENDPOINT_IDENTITY_PREREGISTRATION.md` (commit `d2b0cff`, **pushed before any X2 real-data metric was
computed**). Analysis-only audit on frozen checkpoints: no training, no fine-tuning, no checkpoint or historical prediction created,
modified or overwritten. Artefacts: `artifacts/x2_endpoint_identity/`; large tensors `outputs/x2_endpoint_identity/` (git-ignored).
Implementation: `scripts/analyze_x2_endpoint_identity.py`, `src/ppg2ecg/evaluation/source_sensitivity.py`,
`src/ppg2ecg/data/wildppg_sites.py`, `tests/test_x2_endpoint_identity.py`.

## 1. Executive result

**Pre-registered verdict: PARTIAL SUPPORT.**

The frozen OT-CFM models exhibit **strong source-noise cancellation at the source endpoint** on all three ECG conditions:
different Gaussian sources are mapped to almost the same PPG-conditioned output (variance retention 0.10–0.17 %, i.e. an amplitude
retention of 3.2–4.1 %), the first-order source-retention slope is indistinguishable from zero (|β| ≤ 0.005), and the local Jacobian
of the endpoint map is near-zero (ρ_J 0.035–0.042) with the velocity Jacobian pointing almost exactly along −I
(cos(J_x v·d, −d) = +0.999). **H-X2-CANCEL passes 3/3.**

The source-averaged endpoint F̄ is by a wide margin the closest of the compared outputs to the deterministic A6 MSE proxy on all
three conditions (**M1 passes 3/3**), but the pre-registered WildPPG waveform-correlation criterion **M2 (PCC ≥ 0.60) FAILS**
(observed 0.545). By the frozen rule H-X2-MEAN therefore fails and the overall verdict is PARTIAL SUPPORT, not STRONG SUPPORT.
**This failure is reported as it stands; the threshold was not adjusted after seeing the result.**

Safe reading: *the frozen OT-CFM models exhibit strong source-noise cancellation at the source endpoint, quantitatively consistent
with the known endpoint-barycenter behaviour of independently coupled flow matching. In ECG reconstruction this provides a
mechanistic account of the conditional-mean-like attenuation observed at one NFE.* The one-step endpoint is a nearly deterministic
conditional predictor, but it is not the same function as the independently trained MSE regressor.

## 2. Prior art — what is NOT novel

Under an independent source–target coupling with the linear path and squared velocity regression, v*(x, t) = E[x1 − x0 | x_t = x];
at t = 0 the state is x0, so with x0 ⟂ (x1, c): **F*(x0, c) = x0 + v*(x0, 0, c) = E[x1 | c]**. This identity is a **direct
conditional corollary of known flow-matching / rectified-flow endpoint conditional-expectation results**, established here *before*
the experiment and never claimed as new (sources fetched and quoted 2026-08-30):

- **Frans et al. 2024**, *One Step Diffusion via Shortcut Models*, arXiv:2410.12557 §2: "At t = 0 the model receives pure noise as
  input and (x0, x1) are randomly paired during training, so the predicted velocity at t = 0 points towards the dataset mean. Thus,
  even at the optimum of the flow matching objective, one step generation will fail for any multi-modal data distribution."
- **Albergo, Boffi & Vanden-Eijnden 2023**, arXiv:2303.08797, Remark 42 / Eq. (4.12): b(0, x) = α̇(0)x + β̇(0)E[x1 | x0 = x] − …,
  which for α = 1−t, β = t is exactly E[x1 | x0] − x0.
- **Albergo & Vanden-Eijnden 2022**, arXiv:2209.15571, Eqs. (B.18)–(B.19): v0(x) = ȧ0·x + ḃ0∫x1ρ1(x1)dx1 = E[x1] − x.
- **Liu, Gong & Liu 2022**, *Flow Straight and Fast*, arXiv:2209.03003, Eq. (2), one-step update Z1 = Z0 + v(Z0, 0), Fig. 5 caption:
  "it generates the mean of π1 when simulated with a single Euler step".
- **Lee et al. 2024**, arXiv:2405.20320 §2/§4.1: the CFM optimum is the MMSE/conditional-expectation estimator, and a 1-rectified
  flow "learns to simply output the dataset average" at the noise end.

Closely related but without the t = 0 statement: Lipman et al. 2022 (2210.02747), Tong et al. 2023 (2302.00482), Lipman et al. 2024
FM Guide (2412.06264), MeanFlow (2505.13447), Improved MeanFlow (2512.02012), Flow Map Matching (2406.07507), Consistency Models
(2303.01469); Salimans & Ho 2022 (2202.00512) is the diffusion-side "one step = blurry average" origin. The **conditional-on-c**
restatement is a one-line corollary of the above; only its use as an empirical endpoint audit on a frozen conditional physiological
model is ours. Not exhaustively searched: conditional / image-to-image flow-matching literature, so absence there is not asserted.

## 3. The exact finite-model question

The identity is a **population** statement. A finite trained network need not realise it: it is trained on finitely many samples,
with finite capacity, and — critically — **the endpoint t = 0 carries essentially no explicit training mass under uniform timestep
sampling and is not intentionally endpoint-supervised** (`external/PENGUIN/src/models/PENGUIN.py:225`, `torch.rand(B,1)`).
Evaluation at exactly t = 0 therefore probes a boundary value inferred primarily from near-zero training samples. X2 asks whether the
learned field nevertheless behaves as the population theory predicts, and how close the resulting nearly deterministic map is to
independent conditional-center proxies.

Verified from source before the experiment: independent coupling `x_0 = torch.randn_like(x_1)` with **no** minibatch-OT / Sinkhorn /
assignment anywhere (`PENGUIN.py:228`; the repository's "OT-CFM" name denotes the Lipman *OT conditional path*, i.e. the linear /
rectified-flow interpolant); linear path with σ_min = 0 so x_{t=0} = x0 exactly; and Euler-1 is exactly the source-endpoint map —
`samplers.py:57-67` with `linspace(0,1,2)` evaluates the network **once** at float32 t = 0 with x_t = x0 and returns x0 + 1.0·v.

## 4. Frozen protocol and provenance

| Condition | N | Clusters | OT-CFM ckpt (round) | iMF ckpt | A6 MSE ckpt |
|---|---|---|---|---|---|
| WildPPG (A4/A6c protocol) | 3,907 | 8 = {kjd, ssx} × {sternum, head, wrist, ankle} | `a4_otcfm_wildppg_seed42` (189) | `a4_imeanflow_wildppg_seed42` (45) | `a6c_fullbackbone_mse_wildppg_seed42` (33) |
| PPG-DaLiA S2 | 1,025 | 1 (single held-out subject) | `a0b_penguin_otcfm_ppgdalia_8s_seed42` (64) | `a2_imeanflow_s5_ppgdalia_8s_seed42` (60) | `a6a_fullbackbone_mse_dalia_testS2_seed42` (23) |
| PPG-DaLiA S1 | 1,151 | 1 (single held-out subject) | `a3_otcfm_ppgdalia_testS1_seed42` (93) | `a3_imeanflow_ppgdalia_testS1_seed42` (15) | `a6b_fullbackbone_mse_dalia_testS1_seed42` (5) |

K = 32 source seeds 0–31 (`torch.Generator().manual_seed(k); torch.randn(N,1,1024)`, CPU float32 — the historical construction);
B50 = mean of K50 = 8 OT-CFM Heun-25 (**50 NFE**) samples on the frozen 64-window cluster-stratified subset; ρ_J on the same subset
with 4 unit-RMS directions (seed 20260301) at the seed-0 source; oracle t-profile t ∈ {0, .01, .05, .10}; bootstrap 2000, seed 0.
Checkpoint sha256, split manifests, subset indices, NFE counts, device (RTX 5090), torch 2.11.0+cu130 and script sha256 are in
`artifacts/x2_endpoint_identity/provenance.json`.

**Provenance validation — bit-exact.** Re-inference at source seed 0 reproduced the frozen prediction arrays **exactly**
(max |Δ| = 0.0 on all three conditions for OT-CFM-1 vs `euler1.npz`, for iMF-1 vs `meanflow1.npz`, and for the A6 regressor vs
`regressor.npz`). Observed NFE: 1 for the one-step arms, 50 for the Heun-25 reference. Bit-exact GPU parity was *not* assumed in the
pre-registration (a ≤ 1e-4 tolerance was allowed); it was achieved because the batching, device and library versions are unchanged.

## 5. Source cancellation at t = 0 (H-X2-CANCEL: PASS 3/3)

Medians over test windows; brackets are 95 % bootstrap intervals (WildPPG: cluster bootstrap over the 8 (subject, site) clusters;
DaLiA: window-level bootstrap, **descriptive within-subject only — it does not estimate cross-subject generalisation**).

| Condition | model | R_source (variance) | √R_source (amplitude) | β | D_pair | verdict terms |
|---|---|---|---|---|---|---|
| WildPPG | **OT-CFM-1** | **0.00168** | **0.0410** | **−0.0006** | 0.0367 | C1 ✓ C2 ✓ C3 ✓ |
| WildPPG | iMF-1 | 0.1500 | 0.3874 | +0.1094 | 0.3519 | (descriptive) |
| DaLiA S2 | **OT-CFM-1** | **0.00170** | **0.0412** | **+0.0043** | 0.0376 | C1 ✓ C2 ✓ C3 ✓ |
| DaLiA S2 | iMF-1 | 0.1025 | 0.3201 | +0.0911 | 0.3045 | (descriptive) |
| DaLiA S1 | **OT-CFM-1** | **0.00102** | **0.0319** | **−0.0044** | 0.0297 | C1 ✓ C2 ✓ C3 ✓ |
| DaLiA S1 | iMF-1 | 0.0773 | 0.2781 | +0.1134 | 0.2629 | (descriptive) |
| all | A6 MSE proxy | **N/A — deterministic model has no source latent** | N/A | N/A | N/A | — |

R_source and √R_source are reported as a pair and never conflated: an amplitude retention of ≈ 4 % is a **variance** retention of
≈ 0.17 %. All three frozen criteria (R_source < 0.05, median β < 0.25, median ρ_J < 0.25) hold in every condition. As stated in
advance, C1 was a permissive bar expected to pass given the published seed-diversity context (std 0.024–0.035); the informative and
previously unmeasured criteria are β and ρ_J, and both pass by an order of magnitude.

## 6. Local Jacobian diagnostic

| Condition | OT-CFM-1 ρ_J (median [q25, q75]) | cos(J_x v·d, −d) | iMF-1 ρ_J | cos |
|---|---|---|---|---|
| WildPPG | **0.0413** [0.0351, 0.0527] | **+0.999** | 0.4810 [0.4027, 0.5964] | +0.886 |
| DaLiA S2 | **0.0423** [0.0369, 0.0541] | **+0.999** | 0.5838 [0.4578, 0.7310] | +0.845 |
| DaLiA S1 | **0.0346** [0.0298, 0.0429] | **+0.999** | 0.4184 [0.3469, 0.5158] | +0.911 |

An identity-like (non-cancelling) map would give ρ_J = 1. The measured ρ_J ≈ 0.04 means the learned t = 0 field cancels ≈ 96 % of an
infinitesimal source perturbation in every probed direction, and the near-unit cosine says the velocity response is aligned with −d,
i.e. **J_x v ≈ −I** — exactly the local signature the population identity predicts. This is an independent local measurement, not a
restatement of the finite-source statistics: R_source uses O(1) perturbations while ρ_J is infinitesimal, and under local linearity
R_source ≈ ρ_J² (observed 0.0017 vs 0.0017 on WildPPG, 0.0017 vs 0.0018 on S2, 0.0010 vs 0.0012 on S1 — consistent, so the map is
close to linear over the noise scale).

`E_cancel` from the original proposal was deliberately not used: (v_a − v_b) + (x0_a − x0_b) ≡ F_a − F_b, so it is algebraically
redundant with D_pair.

## 7. Conditional-center comparison (H-X2-MEAN: M1 PASS 3/3, M2 FAIL)

Mean per-window RMSE of the source-averaged endpoint F̄ against each comparator:

| Condition | **F̄ vs A6 MSE** | F̄ vs OT-50 (seed 0) | F̄ vs iMF-1 (seed 0) | F̄ vs GT | M1 | PCC(F̄, A6) | Q_MSE |
|---|---|---|---|---|---|---|---|
| WildPPG | **0.0675** | 0.2515 | 0.3498 | 0.3527 | ✓ | **0.545** (M2 needs ≥ 0.60 → ✗) | 0.925 |
| DaLiA S2 | **0.0727** | 0.2976 | 0.3187 | 0.3010 | ✓ | 0.176 (descriptive) | 0.933 |
| DaLiA S1 | **0.1154** | 0.2808 | 0.3009 | 0.3458 | ✓ | 0.136 (descriptive) | 0.979 |

F̄ is 3.7–4.3× closer to the deterministic MSE proxy than to any other output, on every condition. **M2 fails on WildPPG: 0.545 vs
the frozen 0.60.** For context (prior-observed, not a blind threshold): the A6 report measured PCC 0.51 between the single seed-0
OT-1 sample and the MSE proxy; source averaging raised it to 0.545, which is a real but insufficient improvement. The honest reading
is that the one-step endpoint and the independently trained MSE regressor are *close but not the same function* — expected, since
they are different networks with different objectives, selection criteria and training lengths (round 189 vs 33), and A6 uses
cond = 0.05·E(0.5) rather than E(0).

**Q_MSE behaved exactly as predicted in pre-inference amendment 1**: 0.925 / 0.933 / 0.979, i.e. ≈ 1. Because cancellation makes the
source-noise component σ ≪ the systematic offset δ, averaging 32 sources can only remove a negligible part of the distance to the
proxy. This confirms the amendment's reasoning that the originally proposed "Q_MSE < 0.80" pass/fail would have been self-defeating,
and it is reported as a descriptive quantity, as pre-registered.

## 8. Same-model multistep barycenter proxy (supporting)

On the frozen 64-window subset, B50 = mean of 8 Heun-25 (50 NFE) samples from the **same** OT-CFM checkpoint. Distances are also
MC-debiased by subtracting V50/K50 (V50 = within-window variance across the 8 samples).

| Condition | RMSE F̄–B50 | RMSE A6–B50 | RMSE F0–B50 | RMSE GT–B50 | debiased sq-dist F̄ / A6 / F0 / GT | V50 |
|---|---|---|---|---|---|---|
| WildPPG | **0.0964** | 0.1103 | 0.0994 | 0.3730 | **+0.00066** / +0.00364 / +0.00138 / +0.14844 | 0.0730 |
| DaLiA S2 | **0.1097** | 0.1319 | 0.1078 | 0.3359 | **+0.00071** / +0.00755 / +0.00017 / +0.10576 | 0.0976 |
| DaLiA S1 | **0.1053** | 0.1409 | 0.1088 | 0.3464 | **+0.00079** / +0.01126 / +0.00157 / +0.11341 | 0.0870 |

The source-averaged one-step endpoint is **5.5× (WildPPG), 10.6× (S2) and 14.3× (S1) closer in debiased squared distance to the
same-model multistep barycenter proxy than the independently trained A6 MSE regressor is**, and PCC(F̄, B50) exceeds PCC(A6, B50) in
all three conditions (0.407 vs 0.283; 0.095 vs 0.040; 0.207 vs 0.064). Read within the same network, the one-step endpoint sits very
close to that network's own multistep sample average — which is the comparison the population identity actually speaks to, since the
A6 proxy is a different network. B50 is a Monte-Carlo estimate from 8 samples and is called **"same-model multistep barycenter
proxy"** throughout — never ground truth, oracle or "the conditional expectation".

## 9. iMF-1 comparison (descriptive, no pass/fail)

Evaluated on the numerically identical K = 32 source tensors, at its own one-step operating point and time convention
(x̂ = e − u_θ(e, c, t = 1, h = 1); **t = 1 is noise, t = 0 is data** in iMF, the opposite of OT-CFM). iMF-1 retains **28–39 %** of the
source amplitude (7.7–15.0 % of the variance) versus **3.2–4.1 %** (0.10–0.17 %) for OT-CFM-1, a slope β of +0.09…+0.11 versus
≈ 0, and a local Jacobian ρ_J of 0.42–0.58 versus 0.035–0.042 — an order of magnitude more source dependence on every measure. No
identity is imposed on iMF and none is tested for it; it is not claimed that iMF is theoretically immune to conditional averaging or
that source diversity by itself indicates better conditional modelling.

## 10. Exploratory ORACLE path-state t-profile

**ORACLE PATH-STATE DIAGNOSTIC — x_t = (1−t)x0 + t·x1 is built from the ground-truth target; these are NOT realisable one-step
generation states.** Exploratory only, no pass/fail, definition frozen before inference
(G_t = x_t + (1−t)·v_θ(x_t, c, t), source perturbation s = (1−t)x0; at t = 0 it reduces exactly to §5–6).

| t | WildPPG R_source / β / ρ_J | DaLiA S2 | DaLiA S1 |
|---|---|---|---|
| 0.00 | 0.00161 / −0.0004 / 0.0413 | 0.00169 / +0.0038 / 0.0423 | 0.00104 / −0.0042 / 0.0346 |
| 0.01 | 0.00199 / +0.0005 / 0.0442 | 0.00199 / +0.0052 / 0.0444 | 0.00132 / −0.0031 / 0.0360 |
| 0.05 | 0.00639 / +0.0045 / 0.0660 | 0.00468 / +0.0114 / 0.0599 | 0.00357 / +0.0013 / 0.0485 |
| 0.10 | 0.01430 / +0.0094 / 0.1006 | 0.01311 / +0.0191 / 0.0865 | 0.00979 / +0.0068 / 0.0735 |

Source sensitivity increases smoothly and monotonically — roughly 9× in variance and 2.4× in ρ_J between t = 0 and t = 0.10 — with no
discontinuity at the boundary. The t = 0 behaviour is thus not an isolated boundary artefact of an unsupervised endpoint but the
limit of a continuous trend, which is reassuring given that t = 0 carries essentially no explicit training mass.

## 11. Limitations and deviations

- **Single training seed (42) per checkpoint.** Bootstrap intervals do not capture between-training-seed uncertainty.
- **DaLiA has one held-out subject per condition**; its intervals are descriptive within-subject and say nothing about cross-subject
  generalisation. Only WildPPG has a genuine (8-cluster) subject×site bootstrap.
- **A6 is a cross-network proxy**, not E[x1|c]; B50 is a same-model Monte-Carlo estimate from 8 samples (debiased, but noisy).
- **The t > 0 profile uses the ground-truth target** and is labelled ORACLE everywhere; it is not a realisable generation diagnostic.
- **ρ_J is local** (infinitesimal directions at one source per window, 4 directions); the finite-source statistics complement it.
- **Deviations from the pre-registration: none after inference.** Two changes were stated in the pre-registration *before* any
  result: (i) pre-inference amendment 1 demoting "Q_MSE < 0.80" from pass/fail to descriptive with an advance prediction, and
  replacing H-X2-MEAN with comparative-closeness rules; (ii) dropping `E_cancel` as algebraically redundant. Two implementation
  defects were found and fixed **before** the reported run: a bootstrap that would have degenerated on DaLiA (one cluster) instead of
  the pre-registered window-level bootstrap, and clipped figure titles. The corrected run reproduces the first run's metrics exactly
  (the pipeline is deterministic and seed-0 parity is bit-exact); only the DaLiA intervals and figure rendering changed.
- **Nothing was overwritten and nothing was trained**: every frozen checkpoint/prediction retains its original mtime (2026-08-25 to
  2026-08-27); X2 wrote only to `artifacts/x2_endpoint_identity/` and `outputs/x2_endpoint_identity/`.

## 12. Allowed and forbidden claims

Supported by these data: the frozen OT-CFM models **empirically realise** the known endpoint-barycenter degeneracy at the source
endpoint — the one-step map is nearly independent of the Gaussian source (variance retention ≈ 0.1 %, J_x v ≈ −I); the resulting
nearly deterministic output is far closer to a deterministic conditional-center proxy than to any generative alternative, and closer
still to the same model's own multistep barycenter proxy; iMF-1 retains an order of magnitude more source dependence at its own
one-step operating point; source dependence returns smoothly, not abruptly, away from t = 0. Together with X0 this provides a
**mechanistic account of the conditional-mean-like attenuation** measured at 1 NFE.

Explicitly **not** claimed: that we discovered or proved that flow matching converges to the conditional mean (the identity is prior
art and the population statement is not established for the finite network); that the finite network equals the population optimum;
that F̄ equals E[x1|c] (the WildPPG PCC criterion failed at 0.545 — F̄ and the MSE regressor are close but distinct functions); that
multimodality is proven or that ECG is inherently multimodal; that changing the source–target coupling would necessarily fix the
clinical failure; that iMF is theoretically immune to conditional averaging, or that source diversity alone demonstrates better
conditional modelling; that source-noise cancellation by itself proves conditional-distribution collapse.

**Context that must not be dropped (A7/A8):** conditional-center prediction is *not* intrinsically pathological — on ABP a
deterministic conditional predictor was the best model, preserving task-relevant waveform structure (SBP MAE 14.3 mmHg, morphology
0.929, F1 0.945). Conditional averaging is a problem only when it suppresses task-relevant target structure, which is what X0 showed
for ECG QRS amplitude and sharpness.

## 13. Verdict

| Hypothesis | WildPPG | DaLiA S2 | DaLiA S1 | Result |
|---|---|---|---|---|
| **H-X2-CANCEL** (R_source < 0.05 ∧ β < 0.25 ∧ ρ_J < 0.25) | ✓ | ✓ | ✓ | **PASS 3/3** |
| **H-X2-MEAN** M1 (F̄ closest to A6 MSE) | ✓ | ✓ | ✓ | PASS 3/3 |
| **H-X2-MEAN** M2 (WildPPG PCC(F̄, A6) ≥ 0.60) | ✗ (0.545) | n/a | n/a | **FAIL** |
| H-X2-IMF-DIFF | descriptive | descriptive | descriptive | iMF-1 ≈ 10× more source-dependent |

**Overall: PARTIAL SUPPORT** — cancellation is clear and consistent in 3/3 conditions, while the conditional-center comparison is
mixed (the closeness ranking passes everywhere, the absolute waveform-correlation criterion fails on the primary dataset).

## 14. Implications for a possible next experiment (NOT started)

Because H-X2-CANCEL passes, the source-endpoint map of these frozen models is effectively a deterministic conditional predictor, so
the natural causal follow-up is whether the **source–target coupling** is what controls this behaviour: a separately pre-registered
comparison of independent coupling versus a properly implemented non-independent / OT-style coupling, holding the S5 backbone,
target representation, optimiser and training budget, velocity loss and 1-NFE inference fixed. The answer must not be assumed: the
data here show *that* the finite model cancels the source, not *that* coupling is the causal lever, and A7/A8 show a deterministic
conditional predictor can be an excellent solution. The M2 failure also leaves an open descriptive question — what systematically
separates the one-step endpoint from an independently trained MSE regressor, given that the endpoint is closer to the same model's
own multistep barycenter than to that regressor.

**No such experiment was started. X2 stops here, as pre-registered.**
