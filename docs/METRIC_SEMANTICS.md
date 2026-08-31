# Metric semantics — denominators, populations and what each number can and cannot fall for

Added 2026-08-31. **No previously reported value changes.** This document fixes the *labels* and the
*scope* of the beat-level metrics used throughout `docs/`, because two of them are conditional on
successful R-peak matching in a way their short names do not convey.

Read this before quoting any morphology or RR number from an A-series or X-series report.

---

## 1. The two conditional metrics

### 1.1 `morph` / `morphology_corr` → **matched-beat morphology correlation**

`src/ppg2ecg/evaluation/rpeaks.py:82-89`.

```python
def morphology_corr(ref_sig, pred_sig, ref_r, pred_r, matches, fs) -> float:
    cs = []
    for i, j in matches:                      # <- ONLY matched pairs
        wr, wp = beat_window(ref_sig, ref_r[i], fs), beat_window(pred_sig, pred_r[j], fs)
        ...
        cs.append(float(np.corrcoef(wr, wp)[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")
```

Four properties follow directly from that loop, and all four matter when reading a number:

1. **It is averaged only over beats that were successfully matched** within the 50 ms one-to-one
   tolerance (`matches` comes from `match_rpeaks(..., tol_ms=50.0)`). Unmatched ground-truth beats
   contribute nothing.
2. **Each beat is windowed on its own detected R-peak** — the reference on `ref_r[i]`, the prediction
   on `pred_r[j]`. Any residual timing error up to the full 50 ms tolerance is removed by
   construction. It is a *shape* statistic, not a placement statistic.
3. **It returns `NaN` for a window with zero matched beats**, and such windows are dropped from the
   mean rather than scored. The hardest windows — the ones where the model produced no usable beat at
   all — are therefore absent from the average.
4. **It cannot fall when a beat is missed.** Missing a beat removes a term from the average; it never
   adds a penalty. Under some conditions missing the hardest beats *raises* it.

**Therefore `morph = 0.78` means "the beats we found and paired within 50 ms have the right shape",
not "the reconstruction has the right shape".**

Measured coverage (WildPPG development subjects `an0`+`k2s`, the frozen X4-0 stage-B population of
2048 windows / 19,834 GT beats, Gaussian source seed 0, iMF NFE 8): the matched-pair denominator is
**7,910 of 19,834 GT beats = 39.9 %**, and **254 of 2048 windows (12.4 %) contribute nothing at all**.
Recomputing `morph` restricted to that denominator reproduces the recorded value to 2.0e-10, so this
is a measurement, not an inference from reading the code. Coverage differs by arm — OT-CFM-50's
matched set is larger — so **`morph` values are not directly comparable across arms with different
recall**.

### 1.2 `rr_mae_ms` → **matched-consecutive-beat RR MAE**

`src/ppg2ecg/evaluation/rpeaks.py:68-72`; the docstring already says so.

```python
"""MAE of RR intervals over consecutive reference beats that are BOTH matched."""
```

An RR interval enters the average only when **both** of its bounding reference beats were matched
within 50 ms. Windows contributing no such consecutive pair return `NaN` and are dropped.

Measured on the same 2048-window development population, seed 0: iMF NFE 8 contributes
**1,276 of 2048 windows (62 %)**, OT-CFM Heun-25 **1,383 of 2048 (68 %)**. A denominator-free
per-window mean-RR error computed on the same arms is substantially larger than the matched figure.
**Any claim resting on `rr_mae_ms` must be re-derived denominator-free before use.**

---

## 2. Metrics that are *not* conditional on matching

These already exist in the same scoring function and the same CSVs. Nothing new needs to be
implemented to report an all-GT-beat morphology figure.

| name | where | anchor | population |
|---|---|---|---|
| `same_coord_corr` | `scripts/analyze_x4_0_event_reliability.py:141` | GT R-peak, **no alignment** | every GT beat whose window fits in the array |
| `oracle_corr` | `analyze_x4_0_event_reliability.py:142` | GT R-peak, **per-beat oracle integer shift ≤ ±150 ms** | same |
| `oracle_absent` | `analyze_x4_0_event_reliability.py:133,140` | GT R-peak | same; fraction with `oracle_corr < 0.5` or `oracle_p2p_ratio < 0.2` |

On the 2048-window development population these cover **18,275 / 19,834 = 92.14 %** of GT beats, versus
39.9 % for `morph`.

Two cautions when using them:

- **`oracle_corr` grants a per-beat search over 39 candidate shifts and has no published null
  calibration in this repository.** Part of its gain over `same_coord_corr` is a selection effect of
  unknown size. Calibrating that floor is an open item.
- **`same_coord_corr` is 0.094–0.124 for every arm in the X4-0 sweep, including OT-CFM-50.** At exact
  GT coordinates no model in this project reproduces beat shape. It is a valid statistic but it does
  not discriminate between arms, so it cannot carry a headline on its own.

---

## 3. Reporting rule going forward

1. Use the full names in prose: **matched-beat morphology correlation** and
   **matched-consecutive-beat RR MAE**. The short column names `morph` / `rr_mae_ms` stay as they are
   in code and CSVs.
2. **A headline morphology claim must be accompanied by at least one detector-independent,
   all-GT-beat statistic** (`oracle_corr` and/or `same_coord_corr`) computed on the same population,
   together with the matched-beat coverage fraction.
3. State the matched-beat coverage whenever a `morph` or `rr_mae_ms` value is compared **across arms**;
   arms with different recall have differently-sized and differently-selected denominators.
4. `hr_bpm` (`rpeaks.py:62-65`) is `60 / (mean(diff(rpeaks)) / fs)` and is **exactly invariant to a
   constant time shift**. It carries no information about phase and must not be cited as evidence
   about temporal alignment.

---

## 4. What this does not change

No value in any committed report, table, CSV or artifact is revised by this document. The direction of
every qualitative conclusion drawn from these metrics is unaffected; what changes is that the
denominator is now stated, so that a reader does not take a matched-beat statistic for a
whole-reconstruction one. Where a conclusion was quantified as a gap-closure percentage on `morph`,
the corresponding all-GT-beat figure is smaller — see `docs/X4_0_EVENT_RELIABILITY_REPORT.md` and the
correction in commit `0eb4ff9`.
