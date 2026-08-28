# B1 — Source audit: progressive temporal-gap weighting for MeanFlow training

Prepared 2026-08-28, before any B1 implementation was finalised or trained. Sources: arXiv:2511.19065 **v2 (2026-05-25)** (v1
2025-11-24) and the CVPR 2026 open-access version, both downloaded and read in full text; the two versions agree on the method.

## 1. Paper
"Understanding, Accelerating, and Improving MeanFlow Training" — Jin-Young Kim, Hyojun Go, Lea Bogensperger, Julius Erbach,
Nikolai Kalischek, Federico Tombari, Konrad Schindler, Dominik Narnhofer (Yonsei / ETH Zürich / UZH / MPI-ETH CLS / Google).
CVPR 2026. **Official implementation: none found** — no repository link in either PDF, on the arXiv page, or via search
(2026-08-28). The audit therefore rests on the paper text alone; equation and integration details are fully specified in the main
text (§5) and Appendix B.

## 2. Setting and definitions (paper notation)
Interpolant `z_t = (1−t)x + t·ε` with conditional velocity `v = ε − x` (t = 1 noise) — identical to our iMF convention.
**Temporal gap** `Δt = t − r`. MeanFlow objective decomposes (App. B Eq. 6) into `L_u(z_t, r, t)·I(t≠r) + L_v(z_t, t)·I(t=r)`;
in our iMF implementation the `t=r` rows are exactly the 50 % `fm_mask` boundary samples (`V = u(z,t,t)` = plain flow matching), so
the indicator maps one-to-one onto our frozen `sample_tr` boundary mask. Adaptive weighting (App. B Eq. 7):
`w_adp = sg(1/(‖e‖² + c)^p)`, p = 1 — the same form our frozen iMF uses (p = 1, c = 0.01).

## 3. The exact progressive weighting (main text §5)
> β(Δt, s) = 1 − s + λ·s·(1 − Δt)
- `s ∈ [0,1]` is **training progress**: `s = 1 − (i/T)^k` with `i` = current iteration, `T` = **the total number of training
  iterations**, and **k = 1 (linear) used in all of the paper's experiments** (Table 4 ablates k and finds the linear schedule best).
  At initialisation s = 1 → β = λ(1 − Δt) (small gaps prioritised); at convergence s = 0 → β = 1 (uniform).
- **λ = 1/E_Δt[1 − Δt]** — chosen "to maintain a uniform expectation at initialization" (mean β at s = 1 ≈ 1). The expectation is
  over the Δt distribution the weighting acts on; since β multiplies only the non-boundary term (below), we compute it over the
  frozen `sample_tr` **non-boundary** distribution.
- **Schedule horizon = the entire planned training run** (T = total iterations). No fraction-of-training completion rule appears.

## 4. Application order and boundary treatment (App. B, decisive)
Eq. 8: `L = E[ β(Δt,s)·L_u^adp(z_t,r,t)·I(t≠r) + α(t)·L_v^adp(z_t,t)·I(t=r) ]` (Eq. 9 = the DTD-sampling variant without α).
- **β multiplies only the adaptive-weighted non-boundary u-loss.** The boundary v-loss term carries no β.
- App. B, verbatim: adaptive normalisation "would be ineffective if our proposed weighting schedules … were applied directly to the
  raw losses. Therefore, **we strictly apply these weightings after the adaptive normalization**." → β is an external multiplier on
  the per-sample adaptively weighted loss; it must not enter the residual statistic that defines `w_adp`.

## 5. The source method has TWO components; B1-v2 isolates ONE
The paper's full strategy = (1) **accelerated v-learning** — a modified timestep distribution `p_acc(t)` (e.g. DTD) *or* a
time-dependent boundary weight `α(t)` (e.g. MinSNR) on the v-loss — plus (2) **progressive L_u weighting** (β above). All headline
numbers (FID 3.43 → 2.87, 2.5× fewer iterations) are for the **combination**; the paper's own ablation (Table 3-left) shows
`+ L_u weighting` alone improves the DiT-B/4 baseline 11.58 → 10.98 (1-NFE FID), a smaller but real isolated effect.
**B1-v2 implements component (2) only** — a *source-inspired isolated progressive-gap intervention*:
| Source component | In B1-v2? | Note |
|---|---|---|
| β(Δt,s) on the non-boundary adaptive-weighted loss | **yes** | exact equation, λ rule, linear s, T = total steps |
| α(t) boundary v-loss weighting (MinSNR) | **no** | boundary β ≡ 1 = vanilla behaviour (spec §3; matches Eq. 9's unweighted boundary) |
| p_acc(t) timestep-distribution change (DTD) | **no** | frozen `sample_tr` unchanged (spec forbids new samplers) |
| DiT/ImageNet/CFG stack | no | our frozen S5/ECG stack |
| baseline = original MeanFlow | deviation | ours = **Improved** MeanFlow (V-parameterisation, h-only conditioning, 50 % r=t); the adaptive weighting form is identical (p=1) |
The paper's full-method gains must NOT be read as B1-v2's expected effect; the isolated-component ablation (−0.60 FID on DiT-B/4)
is the closer analogue. The paper also does not use early stopping (fixed-epoch budgets), which B1-v2's fixed-compute design matches.

## 6. Consequences frozen into B1-v2
1. `beta(h, s) = 1 − s + λ·s·(1 − h)` with `h = t − r`, boundary rows β = 1. 2. `s = max(0, 1 − step/T_train)` (k = 1), step =
optimizer-update count, `T_schedule = T_train` = the frozen config's original 300-round budget (66,000 / 65,400 / 65,482 steps).
3. `λ = 1/E[1 − h] = 1.304639` from 1,000,000 deterministic non-boundary draws of the frozen sampler (seed 12345;
`artifacts/b1_gap_curriculum/curriculum_calibration.json`). 4. β applied strictly **after** the frozen adaptive weighting, outside
the `w_adp` statistic. 5. No α(t), no p_acc(t), no other source component.
