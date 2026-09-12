# N3 — Four PPG views of one heartbeat: REPORT

Preregistration `docs/N3_MULTIVIEW_BEAT_SHAPE_PREREGISTRATION.md` (`76a406f`, pushed before any N3 number
existed). **(GT-R anchor; oracle coordinate — diagnostic only.)** 61.9 s.

---

## 1. Verdict: **MARGINAL** — the extra views carry information that is real and negligible

Four simultaneous PPG sites against the same ECG, with the correct R position handed to every arm.

| arm | beat corr ↑ | beat RMSE ↓ | S4 ↓ | S5 ↓ |
|---|---|---|---|---|
| **T-FIXED** (one template, 0 parameters) | +0.9073 | **0.1703** | **0.0844** | 0.0623 |
| REG-1[best] (single view) | +0.9038 | 0.2041 | 0.0905 | 0.0623 |
| REG-1[final] | **+0.9085** | 0.1941 | 0.0886 | 0.0616 |
| **REG-4[best]** (four views) | +0.9079 | 0.1800 | 0.0852 | **0.0591** |
| REG-4-SHUFFLE[best] | +0.9007 | 0.2099 | 0.0907 | 0.0621 |
| IMF-4[best] | +0.6841 | 0.2722 | 0.2765 | 0.3755 |

| §6 condition | required | observed | |
|---|---|---|---|
| `REG-4 − T-FIXED` | CI > 0 and ≥ **+0.05** | **+0.0005** [+0.0003, +0.0008] | CI passes, **100× below the bar** |
| `REG-4 − SHUFFLE` | CI > 0 and ≥ **+0.025** | +0.0072 [+0.0069, +0.0075] | CI passes, **3.5× below the bar** |
| *secondary* `REG-4 − REG-1` | recorded only | **+0.0041** [+0.0038, +0.0044] | real, significant, negligible |

The preregistration fixed the words for exactly this outcome: **the views carry information that is real
but too small to matter.**

## 2. What changed from N2, and what did not

N2's single-view regressor was **worse** than the template (−0.0005, CI below zero). N3's four-view
regressor is **better** than the template (+0.0005, CI above zero), and beating a single view by +0.0041.
So the extra sites do contain non-redundant information about beat morphology — the sign flipped and the
effect is statistically unambiguous.

It is also **1/100 of the margin this program set for a morphology claim** (A0's +0.05), and smaller than
the difference between choosing REG-1's final checkpoint (+0.9085) and its own best (+0.9038). A method
whose entire contribution is smaller than its own checkpoint-selection noise is not a method.

**The N2 conclusion stands and is now stronger**: the ceiling is a property of PPG as a modality, not of
channel count. Adding three more optical views of the same heartbeat moves per-beat correlation by half a
thousandth against a zero-parameter template.

## 3. More input made generalisation worse

REG-4's internal-dev metric **rose monotonically** through training — 0.0504 at step 1,000 to 0.0731 at
step 6,000 — while its training loss fell three-fold (0.0775 → 0.0257). REG-1's dev metric stayed flat
(0.0478–0.0506). With four channels the model memorises the training beats; there is not enough
generalisable signal in the extra views to absorb the added capacity.

That is why REG-4[final] (+0.9060) is *worse* than REG-4[best] (+0.9079) and worse than REG-1[final]
(+0.9085). Reported because it is the opposite of the expected direction and because U3 established that
checkpoint choice can carry a whole verdict.

## 4. No site is hiding the signal

Single-view performance by site, from the same trained REG-1:

| site | beat corr | | site | beat corr |
|---|---|---|---|---|
| head | +0.9060 | | sternum | +0.9035 |
| wrist | +0.9037 | | ankle | +0.9024 |

Spread **0.0036** — smaller than the multi-view gain and far below the bar. The pooled REG-1 is not hiding
a strong site, and the sternum (closest to the heart, shortest pulse path) is not better than the head.

## 5. Flow matching remains worse at beat scale

IMF-4 reaches +0.6841 against REG-4's +0.9079 and the template's +0.9073, with S4 3.3× and S5 6.0× the
template's. N2 measured the same gap at one view (IMF −0.2306 behind REG); four views do not change it.
Conditional on timing, the beat distribution is close enough to a point mass that a stochastic objective
only pays variance for it.

## 6. Limits

1. Oracle coordinate throughout; nothing deployable. The result is conditional on perfect timing, which
   makes the negative stronger.
2. Correlation is compressed near 0.9. The template also wins beat RMSE (0.1703 vs 0.1800) and S4
   (0.0844 vs 0.0852) — REG-4 wins only S5 (0.0591 vs 0.0623) — so the compression is not masking a win
   on an uncompressed scale.
3. One architecture, one seed, 6,000 steps. REG-4's overfitting (§3) means a *smaller* or regularised
   four-view model might do better; it would still have to find 100× more signal than is there.
4. **Beat set differs from N2's.** N2 selected rows; N3 must select windows, because `select_subset` over
   row space almost never picks all four site-rows of one window (measured: 1 of 1,922). Realised beats
   are reported in `n3_results.json`. T-FIXED moves 0.9062 → 0.9073 across the two beat sets, which bounds
   the effect of that change at ~0.001 — the same order as every effect in §1.
5. WildPPG only; `an0` / `k2s` development validation; `kjd` / `ssx` never loaded.

## 7. Consequence

§1 of the preregistration named two futures. N3 selects the second, with the qualification that the
information is real:

- **The shape-side method direction does not revive.** Multi-view conditioning was the strongest remaining
  hypothesis for why N2 failed, and it moves the number by +0.0005 against a zero-parameter baseline.
- **PPG → beat morphology is closed** across every axis this program can test: single view (N2), four views
  (N3), MSE and flow-matching objectives (both), with and without oracle timing (N1 vs N2/N3).

What remains open is the timing side, which N1 showed is where PPG's information actually is, and which
has never been tested for *calibration* — whether existing generators' source-induced timing variability
matches the true uncertainty. That probe requires no training and runs next.
