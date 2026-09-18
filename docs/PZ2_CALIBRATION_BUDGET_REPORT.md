# PZ2 — Calibration budget and difficulty axis: **the curve has not saturated at 32 s**

Prereg `03ece22`, frozen before any number. No training. V1 VitalDB test patients; GT R peaks for every arm;
every budget is scored on the **same** windows (those after the 8th of each case), so the columns are comparable.
Raw: `artifacts/pz2_calibration_budget/`.

| template | calibration | morph_corr ↑ | gain vs GLOBAL [95% CI] | QRS-core RMSE ↓ | MAE ↓ | share of ORACLE gap |
|---|---|---|---|---|---|---|
| GLOBAL | — | 0.6596 | — | 0.3636 | 0.3935 | — |
| k = 1 | 4 s | 0.6535 | −0.006 [−0.013, +0.001] | 0.3739 | 0.3983 | −3 % |
| k = 2 | 8 s | 0.6804 | +0.021 [+0.015, +0.027] | 0.3530 | 0.3802 | 11 % |
| k = 4 | 16 s | 0.7076 | +0.048 [+0.042, +0.054] | 0.3313 | 0.3641 | 25 % |
| **k = 8** | **32 s** | **0.7283** | **+0.069 [+0.062, +0.075]** | **0.3154** | **0.3518** | **36 %** |
| ORACLE | — | 0.8506 | +0.191 | 0.1727 | 0.2557 | 100 % |

**Saturation (frozen rule: next doubling adds < 0.01).** Deltas are +0.027 (4→8 s), +0.027 (8→16 s), +0.021
(16→32 s) — **not saturated**. More calibration ECG would still help; 32 s is where this probe stops, not where the
curve flattens.

## Difficulty axis (SBV = median 1 − corr(beat, own template), computed on the calibration block only)
| SBV tertile | patients | SBV range | GLOBAL | 32 s | ORACLE | gain | share of gap closed |
|---|---|---|---|---|---|---|---|
| low | 384 | 0.007–0.028 | 0.706 | 0.778 | 0.880 | +0.072 | 41 % |
| mid | 399 | 0.029–0.048 | 0.692 | 0.754 | 0.867 | +0.062 | 35 % |
| high | 397 | 0.048–0.846 | 0.579 | 0.652 | 0.803 | +0.073 | 33 % |

## Reading it
1. **One window (4 s) of calibration is worse than the population template** (−0.006): a template from ~5 beats is
   noisier than the pooled one. Calibration only pays from ~8 s.
2. **32 s of the patient's own ECG buys +0.069 morph_corr, a third of the achievable gap.** The rest is
   within-patient, window-to-window variation — the part a PPG-conditioned generator would have to supply.
3. **The gain is roughly constant across difficulty (+0.06 to +0.07), but the headroom is not**: hard patients start
   lower (0.579) and end lower (0.652), and a static personal template closes a smaller share of their gap (33 % vs
   41 %). Personalisation does not rescue the patients who need it most.
4. SBV is the morphology analogue of PPG2BP-net's SDS (Sci Rep 2023): a per-patient difficulty number computed from
   calibration data only, used as a reporting axis rather than as a score.
