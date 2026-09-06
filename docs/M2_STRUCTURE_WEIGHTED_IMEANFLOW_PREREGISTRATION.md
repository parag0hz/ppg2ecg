# M2 — GSW-iMF: Gradient-Structure-Weighted Improved MeanFlow

**Status: PREREGISTRATION. Frozen on commit. Never edited post-hoc.**
Written 2026-09-07, BEFORE any M2 structure map, training run, or metric.

Method name, chosen here and never renamed: **GSW-iMF** (Gradient-Structure-Weighted improved
MeanFlow). The arm label is **S**.

## 0. Question

Does uniform iMeanFlow regression underweight temporally sparse, high-gradient ECG structure, and
does a **training-only, annotation-free** soft structural importance map improve local ECG structure
while preserving event fidelity and few-step inference?

M2 changes **only** the spatial weighting of the existing iMeanFlow regression loss. No new inference
module, no event predictor, no cross-attention, no STFT, no contrastive objective, no architecture
growth. At inference the proposed method is bit-identical in architecture and computational graph to
the baseline.

M2 does **not** attempt to solve event timing. It tests: *for the same conditional information and
the same iMeanFlow model, does increasing training emphasis on locally high-gradient target ECG
structure improve morphology/joint waveform structure without degrading event fidelity?*

## 1. Repository anchor — and the one deviation, recorded before any result

| check | required | actual | |
|---|---|---|---|
| start HEAD | `89c4fed527b6829143bd9fd09f03505eaff148fc` | `006b42e01f8dc6e7a8637356c9dbab91081b6dfa` | **DEVIATION** |
| HEAD == origin/main | yes | yes | PASS |
| clean tree | yes | yes | PASS |
| PENGUIN | `6cd70cde…` | `6cd70cdefb91f10efeb8dce34019b5067cb25344` | PASS |
| iMeanFlow | `bf60cd7c…` | `bf60cd7cb653f6628e59d48034b333c5eba445e2` | PASS |
| A4 checkpoint md5 | `31c042d291052fbb6dc15263ad316be2` | identical | PASS |
| E2 contract sha256 | `06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115` | identical | PASS |
| C2 | deferred, no outputs | no `outputs/c2*` | PASS |
| kjd / ssx | never loaded | absent from every manifest M2 uses | PASS |

**Deviation D-1 (approved by the user before this document was written).** The M2 specification named
`89c4fed` as the start HEAD. That was HEAD at E3 completion; the repository has since advanced by nine
commits (literature survey, D1, D2), all of which were themselves directed and preregistered. M2 is
re-anchored to `006b42e`. Justification recorded now, not later:
`git diff 89c4fed..006b42e -- src/ppg2ecg/flow/ src/ppg2ecg/training/ src/ppg2ecg/models/ configs/ external/`
is **empty**, and no commit in that range touches `outputs/` or `artifacts/e2_evaluation_contract/`.
Every object M2 depends on is bit-identical to what it was at `89c4fed`.

## 2. Audit of the existing loss — resolved from source, nothing guessed

`src/ppg2ecg/flow/imeanflow.py:104-127`:

```python
tt    = t.reshape(-1, 1, 1)
z_t   = (1 - tt) * x + tt * e                        # x = target ECG, e = noise; t=0 data, t=1 noise
v_tgt = e - x
u, dudt, V = compound_V(u_fn, z_t, t, r, v_tangent, jvp_mode)
delta2 = ((V - v_tgt) ** 2).flatten(1).sum(1)        # [B]  — SUM over (channel, time), NOT mean
w      = 1.0 / (delta2.detach() + norm_eps) ** norm_p # [B]  — PER-SAMPLE scalar, stop-gradient
loss   = (delta2 * w).mean()                          # mean over batch
```

Two facts the M2 specification's shorthand `mean(w_existing * err^2)` does not capture, and which
determine the implementation:

- **A-1.** The existing adaptive weight is **per-sample, shape `[B]`** — not per-timestep. It cannot
  be confused with the structure weight, which is per-timestep.
- **A-2.** The temporal reduction is a **sum**, not a mean. A mean-1-normalised structure weight
  therefore preserves the expected magnitude of `delta2`, which is precisely why §6.4 normalises.

Tensor layout: `x, e, ppg, z_t : [B, 1, T]`, `t, r : [B, 1]`, `T = 1024`.
Shapes, reduction order, `norm_p=1.0`, `norm_eps=0.01`, `jvp_mode="forward"`, and the stop-gradient on
`w` and on `v_tangent` are preserved exactly.

### 2.1 Resolution of an underspecification — frozen here

The specification says *"Preserve the EXISTING iMeanFlow weighting exactly. Do not replace it."* and
*"The ONLY algorithmic difference … must be multiplication by structure_weight."* Those two sentences
force a choice the specification does not state explicitly: whether the adaptive weight `w` is
computed from the **unweighted** or the **structure-weighted** `delta2`. Computing it from the
structure-weighted quantity would change `w` as well, which is a second difference. **M2 therefore
computes `w` from the unweighted `delta2`, exactly as the baseline does**, so that for identical
inputs `w` is bit-identical between arms U and S:

```python
delta2     = ((V - v_tgt) ** 2).flatten(1).sum(1)                        # unchanged
w          = 1.0 / (delta2.detach() + norm_eps) ** norm_p                # unchanged, bit-identical to U
delta2_sw  = ((V - v_tgt) ** 2 * structure_weight).flatten(1).sum(1)     # the ONLY new term
loss       = (delta2_sw * w).mean()
```

`structure_weight` has shape `[B, 1, T]` and broadcasts over the channel axis only. It carries no
gradient. Arm U is obtained by the identical code path with `structure_weight = 1`, which must
reproduce the baseline loss bit-exactly (unit-tested).

## 3. Frozen baseline configuration — resolved from `outputs/a4_imeanflow_wildppg_seed42/train_meta.json`

seed **42** · AdamW · lr **1e-3** · weight_decay **0.01** · no scheduler · batch **64** ·
micro_batch **32** · epochs 300 · patience 20 · min_delta 1e-4 · h_dim 128 · blocks 4 ·
ssm_ratio 2.0 · mlp_ratio 2.0 · sample_rate 128 · cond_mode `h_only` · h_scale 1.0 ·
p_mean −0.4 · p_std 1.0 · data_proportion 0.5 (applies to the **t/r sampler**, not the loader) ·
norm_p 1.0 · norm_eps 0.01 · jvp_mode forward · selection `fixed_imf_mse`, 4 banks, bank_seed 1000,
val_every_steps 220, val_subsample 4096 · processed `data/processed/wildppg_8s` ·
manifest `data/manifests/split_a4_wildppg_seed42.json` · params **4,568,707**.

A4 realised: **66 epochs, best epoch 45**, 12,982.8 s, peak 19,216 MiB.

**Compute budget (§11), frozen:** train windows = 293,271; ⌈293,271/64⌉ = **4,583 optimizer steps per
epoch**; A4's realised budget = 66 × 4,583 = **302,478 optimizer steps**. Both arms run **exactly 66
epochs**, early stopping **disabled**, so the arms are step-for-step paired and neither exceeds A4's
budget. Checkpoint selection inside that budget uses the unchanged `fixed_imf_mse` rule.

## 4. Determinism — measured, not assumed

A4 (trained 2026-08-26) and C1 arm B (trained 2026-09-01) used the same seed and configuration and
produced **byte-identical weights**: state sha256 `20ba7234f25e0fe30960ba0e441d4b9f…` for both, both
stopping at epoch 66 with best epoch 45 and selection metric `0.11945885431656277`.

Training in this repository is therefore **bit-exactly reproducible**, which upgrades two things the
specification could only hope for:

- **§12 reproduction tolerance is set to BIT-EXACT.** Fresh arm U must produce state sha256
  `20ba7234f25e0fe30960ba0e441d4b9f…` at its selected epoch. This is a stronger check than a numeric
  tolerance and is decided before U is run. If U does **not** match bit-exactly, the numeric fallback
  tolerances of §8 apply and the mismatch is reported as a deviation with its cause investigated; a
  mismatch alone is not automatically verdict F, but a mismatch **outside** the §8 numeric tolerances
  is.
- **§10 paired-run synchronisation is exact, not approximate.** Identical initialization, data order,
  batch composition, x0/noise stream and t/r stream follow from the shared seed. No limitation to
  record.

## 5. Data — frozen development split, unchanged

TRAIN12: `e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p` (293,271 windows)
VAL: `an0 k2s` — TEST: `kjd ssx` **FORBIDDEN, never loaded at any M2 stage**.
Preprocessing is historical A4's, unchanged: 128 Hz, 8 s = 1024 samples, PPG Butterworth 0.5–4 Hz,
ECG 0.5 Hz high-pass, per-window z-score then min-max to [−1, 1]. No new preprocessing.

## 6. Structural importance map — frozen formulas

Computed from the **normalised target ECG window `x` only**. No R peaks. No learnable component.

**6.1 First difference.** `d[0] = 0`; `d[n] = x[n] − x[n−1]` for n = 1…T−1. Not central. No `fs` rescale.

**6.2 Local structural energy.** `e[n] = |d[n]|`, convolved SAME-length with the fixed normalised
5-tap binomial kernel `k = [1, 4, 6, 4, 1] / 16`, **reflect** padding. Result `s[n]`. No other smoothing.

**6.3 Robust normalisation, per window independently.**
`q95 = ` empirical 95th percentile of `s`; `m[n] = clip(s[n] / (q95 + 1e-8), 0, 1)`.
No percentile learned from validation. No train-wide normaliser.

**6.4 Loss weight.** `lambda_struct = 2.0`, frozen and never changed after results.
`a[n] = 1 + 2.0 · m[n]`; `structure_weight[n] = a[n] / mean_n(a[n])`, so `mean_n(structure_weight) = 1`
per window to floating precision. Maximum unnormalised emphasis is 3×.

## 7. Arms

| arm | R annotation | spatial weighting | extra params | inference overhead |
|---|---|---|---:|---:|
| **U** | no | uniform (`structure_weight = 1`) | 0 | 0 |
| **S** | no | soft gradient-structure (§6) | 0 | 0 |
| **X** | no | S's map, circularly shifted **+256** samples | 0 | 0 |
| **Q** | yes, training only | hard `R ± 10`, `1 + 2·mask`, mean-1 normalised | 0 | 0 |

X is reached only if the primary gates pass (§13). Q is a secondary prior-art-style positioning
control, not required for the primary verdict, and is run only if compute permits after the primary
result is frozen. `T = 1024` is asserted before the shift; a different `T` is a STOP requiring a new
preregistration.

## 8. Baseline reproduction check (§12)

Primary: U's selected-epoch state sha256 equals A4's. Numeric fallback, compared on the frozen VAL
cohort under the identical protocol, decided now: **relative difference ≤ 2 %** for `rmse`,
`qrs_deriv_rmse`, `qrs_curvature_err`, `qrs_energy_dev`, `qrs_ptp_dev`, `hf_ratio_err`; **absolute
difference ≤ 0.02** for `corr`, `f1@50`, `f1@150`, `f1_excess@50`, `beats_ratio_dev`. Outside these →
**verdict F, BASELINE REPRODUCTION INVALID**, STOP.

## 9. Structure-map pretraining audit (§8) — TRAIN12 only, no validation

≥ 8,192 deterministic TRAIN12 windows (first 8,192 in manifest order over sorted subjects). Report
min/mean/max `structure_weight`, finite fraction, mean-normalisation error, q95 distribution, fraction
of samples with `m > 0.5` and `m > 0.8`, and the top-20 %-weight derivative-energy concentration.

- **S0-A** all weights finite.
- **S0-B** `|mean_n(structure_weight) − 1| ≤ 1e-6` per window.
- **S0-C** median over windows of the fraction of total `|d|` energy inside the top-20 % weight
  positions **> 0.50**.
- **S0-D** `ESS = (Σw)² / Σw²`; require `ESS/T ≥ 0.50` for **≥ 99.5 %** of audited windows.

Any failure → **verdict E, STRUCTURAL WEIGHT MAP INVALID**, STOP, do not train.

**Diagnostic only, no gate (§9):** after the map is frozen, GT R annotations may be used solely to
describe where weight falls — the fraction of excess weight `structure_weight − min(structure_weight)`
inside `R ± 10` versus outside. No R annotation enters arm S.

## 10. Evaluation

Frozen VAL cohort: all `an0` + `k2s` windows under the existing protocol. Primary **NFE 4**, source
seed **0**; also report NFE 1 and 2, and source seeds 0–3 for stability. No NFE selection, no source-seed
selection. Metric implementations are the existing frozen ones and their definitions are **not modified**:

- global: `rmse`, `corr`
- event (`ppg2ecg.evaluation` frozen code): raw `f1@50`, chance-floor `f1@50`, `f1_excess@50`,
  `f1@100`, `f1@150`, `f1@200`, precision, recall, missing, spurious, `beats_ratio_dev`, RR error.
  **Raw F1 is never reported alone.** The chance floor is the frozen `s1_audit` random-phase /
  circular-shift null.
- local structure, from `m1_structural.qrs_core_morphology` — note the repository's names, which M2's
  shorthand abbreviates: `qrs_deriv_rmse`, `qrs_curvature_err`, **`qrs_energy_dev`** (= "qrs_e_dev"),
  **`qrs_ptp_dev`** (= "p2p_dev").
- spectral: HF fraction error. No new STFT metric.
- source stability across seeds 0–3: waveform SD, beat-count SD. No mode-collapse claim from SD alone.

**Bootstrap (§15):** ECG-window clustered as established in E2 (all site rows sharing one target ECG
move together), subject-stratified, equal subject weight, **2,000 replicates, `default_rng(20260904)`**.
Effect convention: error metrics `effect = U − S`; higher-is-better metrics `effect = S − U`; positive
means S is better. Raw values and effects are both stored.

## 11. Primary gates (§16), NFE 4, source seed 0

- **M2-G1** `qrs_deriv_rmse`: relative reduction ≥ **5 %** AND 95 % paired CI for improvement > 0.
- **M2-G2** `qrs_curvature_err`: relative reduction ≥ **5 %** AND 95 % paired CI > 0.
- **M2-G3** at least one of `qrs_energy_dev`, `qrs_ptp_dev` improves with 95 % CI > 0.

## 12. Non-inferiority gates (§17)

- **M2-N1** `f1_excess@50`: S no worse than U by more than **0.020 absolute**.
- **M2-N2** `beats_ratio_dev`: no worse by more than **0.020 absolute**.
- **M2-N3** `rmse`: relative worsening ≤ **2 %**.
- **M2-N4** `corr`: absolute degradation ≤ **0.02**.
- **M2-N5** HF error: relative worsening ≤ **10 %** (guardrail, not a target).

**Subject consistency (§18):** report G1 and G2 effects separately on `an0` and `k2s`. Final support
requires the pooled gate to pass and neither subject to worsen by > 5 % relative on derivative **and**
curvature simultaneously. No population-generalisation claim from two subjects.

## 13. Stop rules

- G1 or G2 fails → **verdict D**, STOP. No lambda tuning, no smoothing change, no extra control arms.
- G1/G2 pass but any of N1–N4 fails → **verdict B**, STOP.
- Otherwise train **X** and test location specificity: **LS1** `qrs_deriv_rmse` S better than X with
  95 % CI > 0; **LS2** `qrs_curvature_err` S better than X with 95 % CI > 0; at least one with
  relative advantage ≥ **3 %**. Failure → **verdict C**.

## 14. Inference audit (§25)

Assert identical parameter count, identical forward graph, identical checkpoint tensor shapes,
identical NFE, **no structure-map code reachable at inference**, and **no ECG target access at
inference**. Latency measured on the same GPU and batch sizes; expected overhead ≈ 0. Any difference
in the inference path is a STOP.

## 15. Verdict tree (§30) — exactly one

**A** GSW-iMF SUPPORTED · **B** structure improves with unacceptable fidelity trade-off ·
**C** structure weight location not specific · **D** not supported · **E** structural weight map
invalid · **F** baseline reproduction invalid.

## 16. Claim boundary (§31)

Even under verdict A, M2 does **not** claim: first ECG structure weighting; first QRS-aware
generation; first target-derived weighting; SOTA; causal importance of derivative energy; that PPG
determines QRS morphology; that weighting solves event geometry; or that R timing is solved. RDDM and
other target-region-aware ECG generation remain relevant prior art, and avoiding R annotations is not
by itself a novelty claim. Safe framing: *"We adapt iMeanFlow to PPG-to-ECG generation and introduce a
zero-inference-cost soft structural weighting of its regression objective."*

M2 uses development subjects only, seed 42 only, no fresh test, no C2, no external dataset, and the
target ECG is used solely to construct the **training** loss weight.

## 17. Deviations

D-1 (§1) is recorded above. Any further deviation is recorded in a dated section of the M2 **report**,
never by editing this document.
