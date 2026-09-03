# O2b — Integer-grid canonical event warp: frozen specification and pre-declared audit

Written **before** the O2b preregistration is committed and **before** any O2b round-trip number exists
(commit-order steps 2–3). Base commit `cca96d6` (clean tree).

**Scope.** O2b is an **operator repair audit only**. It does not test the O2 factorization hypothesis, trains no
generator, and is not deployable: the GT ECG R schedule still builds the operator. The permitted name for the
object under test is **integer-grid oracle canonicalization operator**.

---

## 1. What O2 established, and what O2b changes

O2 (`docs/O2_ORACLE_EVENT_CANONICALIZATION_REPORT.md`, verdict *CANONICALIZATION OPERATOR REJECTED*) showed:

- warp monotonicity, inverse validity and slope-1 QRS cores: all correct;
- raw round-trip RMSE 0.01882 **PASS**, event F1@50 1.000 **PASS**, beat-count difference 0 **PASS**;
- but T4 0.26544, T6 0.38075, T7 0.49422, T8 0.25000 normalised AE against a 0.020 threshold — **FAIL**;
- mechanism: the canonical R position `q_k` is real-valued, so the QRS core is resampled at a **fractional
  offset**; bilinear interpolation is then a 2-tap low-pass. Evidence: the grid-aligned decile had *exactly zero*
  T6/T7 round-trip error, the centre-only variant was indistinguishable from the QRS-preserving one, and a pure
  0.5-sample bilinear shift reproduced 85–94 % of the damage.

**O2b changes exactly one thing**: the canonical event positions become integers. Everything else — boundaries,
the monotone piecewise-linear map and its inverse, `W = 10` protection, the bilinear `grid_sample` resampler,
no post-warp normalisation, no amplitude Jacobian — is unchanged. No new interpolation kernel of any kind (no
sinc, cubic, spline, Fourier or learned resampling), no peak snapping, no manual morphology correction.

## 2. Real-valued reference schedule (identical to O2, unchanged)

For `K ≥ 3`:

```
q_1_real = r_1 ,   q_K_real = r_K ,   q_k_real = r_1 + (k-1)/(K-1) · (r_K - r_1)   for k = 2 … K-1
```

## 3. Integer-grid schedule (the repair)

```
q_1_int = r_1 ,   q_K_int = r_K            (the detector returns integer sample indices, so the endpoints are already integers)
q_k_int = round_half_to_even(q_k_real)     for k = 2 … K-1
```

`round_half_to_even` is implemented explicitly and deterministically (`o2b_warp.round_half_to_even`), returns an
integer dtype, and its tie behaviour is asserted by test: `0.5→0, 1.5→2, 2.5→2, 3.5→4, −0.5→0, −1.5→−2`. No
stochastic rounding, no optimisation, no result-dependent adjustment.

## 4. Spacing safety (STRICT precheck, no repair)

With protected half-width `W = 10`, anchor ordering requires

```
q_{k+1,int} − q_{k,int} ≥ 2W + 1 = 21 samples          for every consecutive pair
q_1_int < q_2_int < … < q_K_int ,      0 ≤ q_k_int ≤ 1023
```

Every window reports `K`, the minimum original RR, the minimum real canonical spacing, the minimum integer
canonical spacing and whether it is ≥ 21.

**STRICT STOP RULE.** If *any* validation window violates the integer spacing condition, the verdict is
**INTEGER GRID SCHEDULE INVALID**, the report is written and O2b stops. The violation is **not** repaired by
shifting individual `q_k`, isotonic projection, clipping, reducing `W`, or dropping windows — O2b tests the
minimal one-line repair, not an optimised integer scheduler.

## 5. Integer anchors

Per beat, when inside `[0, 1023]`: `r_k − W → q_k_int − W`, `r_k → q_k_int`, `r_k + W → q_k_int + W`; plus the
boundary anchors `0 → 0` and `1023 → 1023`. Inside the protected core the target-minus-source offset is
`q_k_int − r_k`, an **integer**, so the local slope is exactly 1 **and** the local displacement is integer-valued.
The same strict-monotonicity keep-rule as O2 applies (both coordinates must increase by more than `EPS = 1e-3`);
a dropped centre anchor makes the window invalid, which under §4 is a STOP rather than a fallback.

## 6. Integer-offset audit (precheck)

For every beat: **(A)** `q_k_int − r_k` is exactly an integer; **(B)** inside `[r_k − W, r_k + W]` the inverse
sampling coordinates land on integer sample positions. The recorded quantity is

```
max_fractional_core_offset = max over all protected-core resampling coordinates of |coord − round(coord)|
```

**Required ≤ 1e-6 samples.** If it fails, O2b stops before the full Stage-0 metric computation
(**INTEGER GRID SCHEDULE INVALID**).

## 7. Resampler (unchanged)

The exact O2 resampler: `grid_sample(mode="bilinear", align_corners=True, padding_mode="border")` through
`[B, C, 1, L]`; identity rows returned bit-exactly; no renormalisation; no amplitude Jacobian. **The primary O2b
operator does not hard-copy protected-core samples** — a manual copy would be a second intervention, and the
point is to test whether integer `q_k` alone suffices under the same bilinear implementation.

## 8. Stage-0 gate — the exact O2 thresholds, not loosened

On the exact frozen O2 cohort (2,048 windows, an0/k2s, 19,834 GT beats), round-trip the GT ECG and require **all**:

| id | quantity | threshold |
|---|---|---|
| R0-1 | median raw waveform RMSE | ≤ 0.020 |
| R0-2 | median T6 (max abs derivative) normalised AE | ≤ 0.020 |
| R0-3 | median T7 (curvature energy) normalised AE | ≤ 0.020 |
| R0-4a | median T4 (QRS p2p) normalised AE | ≤ 0.020 |
| R0-4b | median T8 (QRS width) normalised AE | ≤ 0.020 |
| R0-5 | original-vs-round-trip detector F1@50 | ≥ 0.98 |
| R0-6 | median beat-count difference | = 0 |

Normalisation uses the **exact O1 train IQRs** (`artifacts/o1_component_extractability/target_scaling.json`);
targets use the exact O1 functionals; the detector, preprocessing and cohort are the exact O2 ones.

## 9. Pre-declared T8 support note (code fact, not a result)

The frozen width routine is `rpeaks.qrs_width_ms(sig, r, fs, q_win_s=0.08, s_win_s=0.12)`:
`a = max(0, r − round(0.08·128)) = r − 10`, `b = min(len, r + round(0.12·128) + 1) = r + 16`, `Q = argmin sig[a:r]`,
`S = argmin sig[r:b]`. Its **S search therefore extends to `r + 15`, five samples beyond the protected core
`[r − 10, r + 10]`**. This is stated here, before any O2b number exists, so that a possible T8 failure can be
read against it. **`W` is not changed**, and a T8 failure still fails the frozen gate (§8) regardless of this note.

## 10. Descriptive diagnostics (no thresholds, never change the verdict)

- **Repair ratios** `Error_O2b / Error_O2` for T4/T6/T7/T8 (§13 of the task).
- **Per-beat grid analysis**: original `r`, `q_real`, `q_int`, old fractional offset `dist(q_real, ℤ)`, new
  fractional offset (0 by construction), old shift `q_real − r`, new shift `q_int − r`; Spearman of the O2b
  round-trip T6/T7 error against `|q_int − q_real|` (the rounding perturbation). Association only, no causal
  language.
- **Schedule distortion**: per window `max |q_int − q_real|`, `median |q_int − q_real|`, canonical RR spread
  `max(diff(q_int)) − min(diff(q_int))`, `std(diff(q_int))`, and the relative deviation from the ideal spacing.
  Integer-grid timing irregularity is reported, not hidden.
- **Optional hard-copy diagnostic** (inference-only, never the primary operator, and run **only** if the primary
  integer operator still shows non-negligible T4/T6/T7 error after the verdict is frozen): copy protected-core
  samples sample-to-sample instead of resampling, to separate `grid_sample` numerics from coordinate geometry.

## 11. Verdict tree (frozen)

- **PRECHECK STOP — INTEGER GRID SCHEDULE INVALID** — §4 or §6 fails; no round-trip evaluation, no training.
- **A. INTEGER-GRID CANONICALIZATION OPERATOR ACCEPTED** — R0-1 … R0-6 all pass.
- **B. INTEGER GRID FIXES SHARPNESS BUT WIDTH REMAINS INVALID** — T4, T6, T7 pass and the raw/event gates pass,
  but T8 fails.
- **C. INTEGER-GRID REPAIR INSUFFICIENT** — any of T4/T6/T7 fails, or a raw/event gate fails, despite a valid
  integer schedule.

**Absolute rule: no generator is trained in O2b, even under verdict A.** A passing operator licenses a *new*
preregistration (O2c), nothing more.
