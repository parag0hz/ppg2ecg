# M3 — SEC-iMF: Structural Endpoint Consistency for Improved MeanFlow — REPORT

Preregistration `docs/M3_SEC_IMEANFLOW_PREREGISTRATION.md` (`d5db1c9`, frozen and pushed before any M3
training run). Erratum `88f85ce`. Implementation `c1aa8bd`. Numerical audit `b71f91c`.
`M3_START_SHA = 1cf0d361e739ca9f26d7cd9b988cbbd1e85e8e29` (the pushed M2 result).

# FINAL VERDICT — **D. SEC-IMEANFLOW NOT SUPPORTED**

Invoked under preregistration §11: *"G1 or G2 fails → verdict D, SEC-IMEANFLOW NOT SUPPORTED, STOP."*
Arm **V** was therefore not trained, and no NFE sweep, no extra source seeds, no multi-seed and no test
evaluation were run.

| gate | metric | U | E | rel. | required | CI | result |
|---|---|---|---|---|---|---|---|
| **G1** | `qrs_deriv_rmse` | 0.32100 | 0.31245 | **+2.66 %** | ≥ +5 % **and** CI > 0 | [+0.00832, +0.00877] **above 0** | **FAIL** |
| **G2** | `qrs_curvature_err` | 0.21437 | 0.19740 | +7.91 % | ≥ +5 % and CI > 0 | [+0.01679, +0.01715] above 0 | **PASS** |
| **G3** | per subject | — | — | — | neither worsens > 2 % on both | an0 +3.50 %/+8.94 %, k2s +1.84 %/+6.90 % | **PASS** |

**This is a different failure from M2, and the difference must not be blurred.** M2's arm S made both
structural metrics *worse*. M3's arm E **improves both, significantly** — every gate CI is entirely on
the favourable side — and passes G2 outright. G1 fails on its **magnitude** condition alone: +2.66 %
against a bar of 5 %.

Nothing in §10 or §11 conditions on the sign of the effect. A gate fails when either of its two
conjuncts fails, so a favourable-but-small G1 fails exactly as a hostile one would. The 5 % bar was
frozen before any result existed; lowering it now, or arguing that a significant improvement is
"substantively" a pass, is precisely what the preregistration exists to prevent.

**Counterfactual, recorded so the result is not over-read in the other direction:** had G1 passed, N3
(`qrs_ptp_dev`, −6.28 % against a 5 % margin) also fails, so the stop tree would have given **verdict
B**, not A. Arm E was never one threshold away from support.

## 1. Training

| | U (reused, not retrained) | E |
|---|---|---|
| optimizer steps | 14,409 | **14,409** |
| validation rounds | 66 | 66 |
| selected epoch | 45 | **60** |
| selection metric (`fixed_imf_mse`) | 0.11945885431656277 | 0.12485977127911865 |
| parameters | 4,568,707 | 4,568,707 (identical names/shapes) |
| GPU hours | 3.67 | **4.84** |
| peak VRAM | 19,216 MiB | **20,547 MiB** |
| state sha256 | `20ba7234…` | `e9a8c39a…` |

All twelve §15 post-train integrity checks pass: exactly 14,409 steps, 66 rounds, no early truncation
(`early_stopped=False`, zero early-stop events in the log), parameter count and tensor shapes identical
to U, selection by `fixed_imf_mse` only, the selected epoch is that metric's argmin, and no gate metric
is even present in the training log so it cannot have driven selection.

**Training cost rose 32 % (3.67 → 4.84 h) from the extra endpoint forward. Inference cost is unchanged**
— the sampler is untouched and no SEC symbol is reachable from it.

## 2. Primary results — NFE 4, source seed 0, VAL an0+k2s, 49,200 rows / 12,400 ECG clusters

| metric | U | E | effect | 95 % CI | rel. |
|---|---|---|---|---|---|
| RMSE | 0.41522 | 0.42046 | −0.00524 | [−0.00559, −0.00491] | −1.26 % |
| corr | 0.10802 | 0.12949 | +0.02147 | [+0.02069, +0.02221] | +19.87 % |
| raw F1@50 | 0.43832 | 0.45650 | +0.01818 | [+0.01670, +0.01959] | +4.15 % |
| floor F1@50 | 0.11703 | 0.11780 | +0.00077 | [+0.00038, +0.00119] | +0.66 % |
| F1_excess@50 | 0.32129 | 0.33870 | +0.01740 | [+0.01588, +0.01892] | +5.42 % |
| F1@100 | 0.61099 | 0.61697 | +0.00598 | [+0.00451, +0.00739] | +0.98 % |
| F1@150 | 0.71733 | 0.71836 | +0.00103 | [−0.00028, +0.00232] | +0.14 % |
| F1@200 | 0.78141 | 0.78530 | +0.00389 | [+0.00269, +0.00499] | +0.50 % |
| precision@50 | 0.44695 | 0.46525 | +0.01830 | [+0.01676, +0.01984] | +4.09 % |
| recall@50 | 0.43464 | 0.45243 | +0.01779 | [+0.01634, +0.01923] | +4.09 % |
| missing@50 | 5.60538 | 5.41537 | +0.19001 | [+0.17560, +0.20380] | +3.39 % |
| spurious@50 | 4.97760 | 4.84516 | +0.13244 | [+0.11682, +0.14799] | +2.66 % |
| beats_ratio_dev | 0.10896 | 0.09981 | +0.00915 | [+0.00821, +0.01010] | +8.40 % |
| `qrs_ptp_dev` | 0.48694 | 0.51754 | −0.03059 | [−0.03205, −0.02915] | −6.28 % |
| `qrs_energy_dev` | 0.64072 | 0.62719 | +0.01354 | [+0.01103, +0.01655] | +2.11 % |
| **`qrs_deriv_rmse`** | 0.32100 | 0.31245 | **+0.00855** | [+0.00832, +0.00877] | **+2.66 %** |
| **`qrs_curvature_err`** | 0.21437 | 0.19740 | +0.01696 | [+0.01679, +0.01715] | +7.91 % |
| HF error (`F4__ratio_dev`) | 0.38090 | 0.42088 | −0.03998 | [−0.04246, −0.03734] | −10.50 % |

Positive effect = E better. Raw F1 is never reported without its chance floor and excess.

Non-inferiority: **N1 PASS** (F1_excess improves) · **N2 PASS** (beats_ratio_dev improves) ·
**N3 FAIL** (`qrs_ptp_dev` −6.28 % vs a 5 % margin) · **N4 PASS** (+2.11 %) · **N5 PASS** (corr improves)
· **N6 PASS** (RMSE −1.26 % vs 5 %) · **N7 FAIL** (HF −10.50 % vs a 10 % margin, missing by 0.50 pp).
**N7 does not fail alone**, so §10's "if it alone fails the verdict must explicitly say *spectral
trade-off*" clause is **not** triggered.

## 3. The scientific content of the failure

**The auxiliary worked where it was applied, and the gate measures somewhere else.** The SEC loss is
computed over the **whole** waveform (prereg §4), and its effect is concentrated exactly where the
waveform is flat:

| region (mean \|D1 error\|) | U | E | rel. |
|---|---|---|---|
| background | 0.05450 | 0.03983 | **+26.91 %** |
| peri-QRS | 0.05696 | 0.04211 | **+26.08 %** |
| QRS core (`qrs_core__a3_dabs`) | 0.22322 | 0.21238 | **+4.85 %** |

G1's `qrs_deriv_rmse` lives only inside the GT-R-anchored ±80 ms QRS core. The training objective
flattened the background and peri-QRS regions by ~27 % and reached the QRS core by ~5 %. That is the
mechanism behind +2.66 %.

**It also shrank QRS amplitude.** `qrs_ptp_dev` worsens 6.28 % and `qrs_slope_dev` / `qrs_maxderiv_dev`
worsen 11.01 % — E's QRS complexes are smaller and less steep than U's, while their *shape error*
improves. A global derivative penalty rewards suppressing the largest derivatives in the signal, which
are the QRS upstrokes. This is a coherent account of why curvature improved 7.91 % while amplitude
degraded, and it is the honest reading of the trade-off.

## 4. Robustness and adversarial verification

- Both subjects improve on both gate metrics (an0 +3.50 %/+8.94 %, k2s +1.84 %/+6.90 %); **neither
  reaches 5 % on `qrs_deriv_rmse` alone**.
- The headline table was reproduced from the raw per-window CSV without the evaluator's aggregation
  path, and again independently by an auditor who re-implemented both the subject-macro and the
  clustered bootstrap from scratch (effect `0.008549213291832659`, CI `[0.00831662, 0.00876611]`).
- **No aggregation reaches 5 %.** Every alternative computed is *lower*: pooled row mean +2.58 %,
  pooled cluster mean +2.58 %, beat-weighted +2.56 %, mean of per-cluster relatives +2.54 %, median
  per-cluster +2.33 %. The alternative denominator (U−E)/E gives +2.74 %.
- **No uncertainty reading rescues G1.** At the CI's most favourable bound the improvement is +2.73 %.
  A 5 % improvement needs an effect of 0.01605 — 1.88× the observed point estimate.
- Five independent skeptics (the G1 number, arm identity, selection integrity, evaluator fidelity, and
  a steelman *for* arm E) plus an adjudicator attacked the verdict. **Zero overturning findings; all
  six returned D.**

## 5. Deviations and disclosures — 2026-09-08

**The float32 operator-equivalence criterion** is corrected by the append-only erratum `88f85ce`; the
frozen preregistration was not edited.

**The evaluator was edited after the freeze.** `scripts/m2_evaluate.py` gained the M3 arm paths and a
generalisation of the paired arm from a hard-coded `"S"`. Its **definitions are unmodified** — metrics,
detector, matcher, clustering, orientation rule, `BOOT_N = 2000` and `BOOT_SEED = 20260904` are
untouched — so it should be described as *definitions unmodified, arm registry extended*, not
"unchanged". The auditor found a latent foot-gun in the generalisation (a silent priority order would
pair U-vs-S if `--arms U,S,E` were ever passed); it did not occur here (`evaluation_meta.json` records
`arms: ["U","E"]` and every effect row carries `arm=E`) and the code now asserts that exactly one non-U
arm is supplied.

**The committed numerical and gradient diagnostics describe the network at INITIALISATION, not the
training run.** `scripts/m3_numerical_audit.py` builds a fresh model and loads no checkpoint — which is
what prereg §5 required, since the audit had to precede training. The consequence, measured here: at
that initialisation the SEC gradient reaches only **2 of 161 parameter tensors** (PENGUIN's adaLN-Zero
initialisation zeroes the residual modulation), whereas at the trained arm-E checkpoint it reaches
**140 of 161**. The reported `aux_over_base_ratio = 11.40` and `cosine = 0.581` are therefore
initialisation artefacts and must not be read as properties of the 14,409-step run.

**Arm E is worse on its own selection metric at 63 of 66 rounds** (better only at rounds 8, 14, 31;
mean gap +0.00786), and its selected `fixed_imf_mse` of 0.12486 is worse than U's 0.11946. That metric
is the uniform-weight iMF MSE, which arm E does not optimise — but it is the honest context for
"E improves two structural error metrics".

**`qrs_slope_dev` and `qrs_maxderiv_dev` are the same statistic** by construction in the frozen
`m1_structural` (both append `np.abs(dp).max()`), and are bit-identical here. They are one observation.

**Arm U's `training_summary.json` records `early_stopped: true`.** That field is the report-only
`no_improve >= patience` flag; the actual stop is gated on `and not args.no_early_stop`. U ran the full
66 rounds and 14,409 steps, exactly as E did.

**Preregistration §10 never defines "relative improvement".** The operative definition is inherited
through §9's "frozen M2 evaluation definitions, unmodified": `(U − E)/|U|` on the subject-macro. This is
immaterial here — every defensible denominator lands between +1.51 % and +2.74 % — but the report names
it rather than leaving it implicit.

**`endpoint_algebra_audit.json`'s `t1: 0.0` does not reproduce generically**; in float32 the t = 1
identity measures ~2.4e-07 across seeds. The gate threshold is 1e-6 and passes either way.

**Adversarial-coverage limitation.** All five skeptic lenses completed and each returned
`overturns_verdict: false`, but the adjudicator's context was truncated by the orchestration and it
read only three in full; it independently swept the remaining attack surfaces itself.

**Bootstrap scope.** Clusters are resampled *within* subject and subjects (n = 2) are never resampled,
so the narrow CIs reflect ~6,200 clusters per subject, not two subjects. This is what §9 froze and it
must not be read as population precision.

## 6. Claim boundary

M3 claims none of: that derivative or Sobolev losses are novel; that PPG determines ECG morphology;
that R timing is solved; any causal or observability claim; SOTA; population generalisation; or that
M2 was successful. **M2 remains VERDICT D — STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED, unchanged**, and
M2's post-hoc event improvements were not used as evidence for M3.

Development subjects only (`an0`, `k2s`; `kjd`/`ssx` never loaded), seed 42 only, no fresh test, no C2,
no external dataset, two validation subjects.

## 7. What M3 does establish

That **this** clean-endpoint structural consistency objective, at λ = 0.10 with a 0.5/0.5 D1/D2 mixture
computed globally, improves QRS-core curvature error by 7.91 % and derivative error by 2.66 % while
shrinking QRS amplitude by 6.28 % and worsening high-frequency content by 10.50 % — and that 2.66 % is
below the bar this project set for itself in advance.

## 8. Recommended next step

Per the verdict tree: **stop this direction inside M3.** No lambda tuning, no dropping curvature, no
derivative-only or curvature-only variant, no endpoint smoothing, no QRS masking, no arm V, no
multi-seed, no test. §3 of the mechanism analysis above — that the global auxiliary acts mostly outside
the QRS while the gate measures inside it — is a *new hypothesis*, and pursuing it requires a new
preregistration.
