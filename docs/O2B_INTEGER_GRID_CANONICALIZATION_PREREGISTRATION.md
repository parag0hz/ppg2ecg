# O2b — Integer-Grid Canonicalization Repair — PREREGISTRATION

**Pre-training operator falsification only.**

Frozen before any O2b result. Once committed and pushed, never edited.

| | |
|---|---|
| Base commit | `cca96d6427c8b04843b6c258d2616420f0e7cc3f` (O2 rejection report), clean tree |
| Companion | `docs/O2B_INTEGER_GRID_WARP_AUDIT.md` — the frozen operator specification, committed together with this file |
| Upstream pins | PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`; A4 md5 `31c042d291052fbb6dc15263ad316be2` unchanged; C2 deferred with no outputs; `outputs/o2_canon_oracle_seed42/` does not exist |
| Test subjects | `kjd`, `ssx` — **never loaded** |

---

## 0. The single question

> If the fractional-offset bilinear resampling that O2 identified is removed, can the event-canonicalization
> operator itself preserve QRS morphology?

**NO generator training. NO GPU training. NO new model. NO attention. NO phase network. NO PPG→R predictor.
NO test-subject access. NO C2. NO novelty claim. NO SOTA claim.**

### 0.1 Status and claim boundary (frozen)

1. O2b is an **operator repair audit only**. It is not a generator experiment, not a factorization test, not a
   deployable method, and not evidence that PPG can supply exact R timing.
2. GT ECG R locations still build the operator — the oracle leakage is inherent and unchanged.
3. The permitted term is **integer-grid oracle canonicalization operator**, never "final canonicalized generator".
4. **The O2 factorization hypothesis remains NOT TESTED** until a later generator-training preregistration.
5. O2b was designed after O2; it is problem-discovery evidence, not independent confirmation.

## 1. Operator

Exactly as frozen in `docs/O2B_INTEGER_GRID_WARP_AUDIT.md`: the O2 real-valued schedule is computed unchanged,
interior canonical positions are then projected to integers with an explicit deterministic
`round_half_to_even`, endpoints stay at `r_1` and `r_K`, the ±`W = 10` anchors and boundary anchors are
unchanged, and the O2 bilinear `grid_sample` resampler is reused without modification (no new kernel, no
hard-copy in the primary operator, no post-warp normalisation, no amplitude Jacobian, identity rows bit-exact).

## 2. Cohort and metrics (all reused, none rebuilt)

The **exact** O2 frozen evaluation cohort: 2,048 windows of `an0`/`k2s` selected by
`ER.select_subset("x4-event-nfe-v2", …, 1024)`, asserted element-for-element against
`artifacts/x4_0_event_reliability/nfe_subset.json`, with **19,834** GT beats asserted. Exact same GT ECG,
preprocessing, frozen R detector (`rpeaks.detect_rpeaks`), O1 target primitives (`o1_targets.window_targets`)
and O1 **train** IQR normalisation as O2.

Round-trip `x_rt = W_int^{-1}(W_int(x))` of the GT ECG; metrics: raw RMSE, raw correlation, QRS-core RMSE, the
O1-aligned T4/T6/T7/T8 normalised absolute errors, original-vs-round-trip detector F1@50, beat-count difference.

## 3. Prechecks (before any round-trip metric)

- **Spacing** — every consecutive integer canonical pair ≥ `2W + 1 = 21` samples, strictly increasing, inside
  `[0, 1023]`. Any violation ⇒ **INTEGER GRID SCHEDULE INVALID**, report, STOP. No shifting, no isotonic
  projection, no clipping, no dynamic `W`, no dropped windows.
- **Integer offsets** — `q_k_int − r_k ∈ ℤ` for every beat, and `max_fractional_core_offset ≤ 1e-6` over all
  protected-core resampling coordinates. Failure ⇒ same STOP.

## 4. Stage-0 gate — the exact O2 thresholds

R0-1 raw RMSE ≤ 0.020 · R0-2 T6 ≤ 0.020 · R0-3 T7 ≤ 0.020 · R0-4a T4 ≤ 0.020 · R0-4b T8 ≤ 0.020 ·
R0-5 F1@50 ≥ 0.98 · R0-6 median beat-count difference = 0. **Not loosened, not reinterpreted, not selectively
reused** — the same numbers that rejected O2.

## 5. Verdicts — exactly one

- **PRECHECK STOP — INTEGER GRID SCHEDULE INVALID** (§3 fails)
- **A. INTEGER-GRID CANONICALIZATION OPERATOR ACCEPTED** (R0-1 … R0-6 all pass)
- **B. INTEGER GRID FIXES SHARPNESS BUT WIDTH REMAINS INVALID** (T4/T6/T7 and raw/event gates pass, T8 fails)
- **C. INTEGER-GRID REPAIR INSUFFICIENT** (any of T4/T6/T7 fails, or a raw/event gate fails, with a valid schedule)

Implemented in `o2b_warp.decide_o2b`, asserted by test to match this section. Under **A** the only permitted
statement is *"Integer-grid event canonicalization passes the frozen pre-training morphology-preservation
gate."* — never "factorization works" and never "event canonicalization improves ECG generation".

## 6. Absolute no-training rule

**Even under verdict A, O2b trains nothing.** No O2 training integration is implemented, no iMeanFlow is
trained, no B-vs-O2 comparison, no NFE evaluation, no multi-source evaluation. A passing operator licenses a
separate, newly preregistered experiment (**O2c — Oracle Integer-Grid Event-Canonicalized MeanFlow**). O2b stops
after its report. A test asserts that no generator-training entry point is reachable from the O2b scripts.

## 7. Descriptive diagnostics (never change the verdict)

Repair ratios `O2b / O2` for T4/T6/T7/T8; per-beat grid analysis with the Spearman association between the O2b
round-trip T6/T7 error and the rounding perturbation `|q_int − q_real|`; schedule-distortion audit (median/p90/max
`|q_int − q_real|`, canonical RR spread and std, relative deviation from ideal spacing); the pre-declared T8
support note of the audit §9; and the optional inference-only hard-copy diagnostic, run only if the primary
operator still shows non-negligible T4/T6/T7 error after the verdict is frozen.

## 8. Tests

Repository: firewall, PENGUIN/iMeanFlow pins, C2 untouched. Schedule: `q_real` matches O2 exactly, `q_int`
integer-valued, endpoints preserved, deterministic rounding with explicit tie behaviour, strictly increasing,
minimum spacing ≥ 21, no clipping or dynamic repair. Anchors: integer source R, integer target `q`, integer ±W
anchors, local slope exactly 1, integer target-minus-source core offset. Resampling: the same O2 resampler, no
new kernel, no post-warp normalisation, identity bit-exact. Grid: protected-core coordinates integer within
1e-6. Metrics: same O1 T4/T6/T7/T8 functions, same O1 train IQRs, same O2 cohort, same frozen detector, same
Stage-0 thresholds. Verdict: exact A/B/C/precheck implementation, and no reachable generator-training call.

## 9. Artifacts

`docs/O2B_INTEGER_GRID_WARP_AUDIT.md`, this file, `docs/O2B_INTEGER_GRID_CANONICALIZATION_REPORT.md`, and
`artifacts/o2b_integer_grid/`: `provenance.json`, `cohort_manifest.csv`, `integer_schedule_manifest.csv`,
`integer_spacing_audit.csv`, `integer_grid_core_audit.csv`, `schedule_distortion.csv`,
`warp_roundtrip_metrics.csv`, `stage0_result.json`, `o2_vs_o2b_comparison.csv`, `per_beat_grid_analysis.csv`,
`t8_support_audit.json`, `decision.json`, `figures/`, plus `hard_copy_diagnostic.csv` only if triggered. **No
model checkpoint, no training log and no generator prediction may exist.** `artifacts/*` and `outputs/*` remain
gitignored.

## 10. Figures

(1) O2 vs O2b T4/T6/T7/T8 round-trip nAE against the frozen 0.020 threshold; (2) distribution of
`q_real − q_int`; (3) old fractional-grid distance vs old T6/T7 error together with the O2b errors;
(4) the **exact** window O2 identified as its worst fractional-offset example — original ECG, O2 round-trip,
O2b round-trip. No cherry-picking.

## 11. Commit order

1 integrity → 2 warp audit/spec → 3 preregistration → **4 commit + push** → 5 integer-grid operator →
6 Stage-0 evaluator → 7 tests → **8 commit + push implementation** → 9 prechecks (**fail ⇒ report, result
commit, STOP**) → 10 Stage-0 round-trip → 11 freeze verdict → 12 secondary diagnostics → 13 figures →
14 report → 15 full test suite → **16 result commit + push** → 17 verify clean tree → 18 STOP.
