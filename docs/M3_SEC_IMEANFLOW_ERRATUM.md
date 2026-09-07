# M3 — ERRATUM: float32 operator-equivalence description

**Append-only. The frozen preregistration `docs/M3_SEC_IMEANFLOW_PREREGISTRATION.md`
(`d5db1c919a5462e18bc26a0c7b828567f71ee978`) is NOT edited and remains immutable.**

Written 2026-09-07, before any M3 training run, checkpoint, or validation metric exists.

## 1. What is corrected

The preregistration's §4 prose states that the originally requested float32 absolute threshold of
**≤ 1e-6** against the frozen numpy operators "is not attainable", and its recorded
`deviation_from_specification` field cites a measured discrepancy of **1.2e-6**.

**Both statements are descriptive errors.** The committed machine-readable audit
`artifacts/m3_sec_imeanflow/derivative_operator_audit.json` records:

| quantity | measured | ≤ 1e-6 ? |
|---|---|---|
| `f32_d1_abs` — D1, float32 vs frozen | **2.384185791015625e-07** | **yes** |
| `f32_d2_abs` — D2, float32 vs frozen | **9.5367431640625e-07** | **yes** |

**These measured values are authoritative. D1 and D2 both satisfy the originally requested ≤ 1e-6
absolute threshold in the actual recorded audit.**

Unchanged and still exact:

| quantity | measured |
|---|---|
| `f64_d1`, `f64_d2` — float64 vs frozen | **0.000000e+00** |
| matched-dtype operator equivalence | **exactly 0.0 (bit-identical)** |
| output lengths | D1 → T−1, D2 → T−2, no padding |

## 2. Root cause — identified, not merely noted

The 1.2e-6 figure came from an **earlier throwaway probe that compared unlike inputs**. That probe
drew a float64 array `a`, handed torch a *rounded float32 copy* of it, and handed the frozen numpy
operator the *full float64* `a`. The resulting gap therefore contained the input-rounding error as
well as the operator arithmetic, and was not a measurement of operator equivalence at all.

The committed audit compares like with like — both implementations receive the identical float32
array — which is the correct test. Re-run over **20 independent samples**, the float32 D2 gap ranges
**7.15e-07 to 9.54e-07 and 0 of 20 exceed 1e-6**. The threshold is satisfied robustly, not marginally.

The prose in the preregistration reasoned from the discarded probe's number and generalised it into a
claim about attainability. That reasoning was wrong.

## 3. What this changes

**Nothing scientific.** This correction changes no:

- method · hyperparameter · loss · operator · gate · training rule · evaluation rule · scientific result.

`eps_struct = 1e-6`, `lambda_SEC = 0.10`, the 0.5/0.5 D1/D2 mixture, the D1/D2 definitions, the t/r
sampler, the t-dependent geometry, the optimizer, learning rate, seed, budget, checkpoint-selection
rule, metrics, gates, bootstrap and STOP tree are all untouched.

The operator-equivalence conclusion is also unchanged in substance — the operators were, and remain,
verified equivalent. Only the stated *reason* for the float32 criterion was wrong, and the criterion
itself is now known to be met on its original absolute terms rather than needing a relative restatement.

## 4. State when this erratum was written

No M3 training output, no M3 checkpoint, no M3 validation metric, no arm E or V, no `an0`/`k2s`
evaluation under M3, no `kjd`/`ssx` access, no C2, no lambda trial. Verified:
`outputs/m3_*` does not exist; `artifacts/m3_sec_imeanflow/` contains audits only.

M2 remains **VERDICT D — STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED**, unchanged.
