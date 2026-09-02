# V1 — All-Subject Stepwise Visualization + ECG→PPG Event-Delay Audit — REPORT

## **CONDITIONING FEASIBILITY: `PPG PEAK TIMING TOO UNSTABLE FOR DIRECT CONDITIONING`**

Protocol `a73cafa`. Frozen checkpoint forward inference and a CPU-only timing audit.
**No training. No new model. No attention implementation. No loss change. No test access. C2 remains
deferred with zero weight updates.** Nothing was ever translated or oracle-aligned.

---

## 1. Provenance and coverage

| | |
|---|---|
| model | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt`, round 45, arm `B` |
| file sha256 | `557c70541f5cdd07819a3da04bb53477ac98827285507380` |
| `state_dict` sha256 | `47d7ccb94e5dbf7190d777f852b18f107f3ce2628d160b5e` |
| subjects | **14 non-test** — 12 train, 2 val (`an0`, `k2s`); `kjd`/`ssx` never loaded |
| sites | sternum, head, wrist, ankle — all present for all 14 subjects |
| NFE | 1, 2, 4, 8, 50, **identical source tensor per window across all five** (seed 0) |
| cohorts | VIZ 8/subject/site = **448** · METRICS 32 = **1,792** · DELAY 128 = **7,168**, nested |
| figures | 448 stepwise + 448 R-centred zooms + 11 group figures + a 15-page dashboard |
| delay pairs | **51,957** matched R→PPG events; **0 windows skipped** |

## 2. NFE behaviour — validation subjects

| NFE | raw RMSE | raw corr | QRS RMSE | QRS-energy dev | QRS deriv RMSE | F1 excess | beats-ratio dev |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3988 | 0.1028 | 0.5852 | **0.5431** | 0.3374 | 0.2981 | **0.0922** |
| 2 | 0.3868 | **0.1162** | 0.5784 | 0.5667 | 0.3289 | 0.3265 | **0.0922** |
| **4** | **0.3796** | 0.1152 | **0.5783** | 0.5652 | **0.3288** | **0.3411** | 0.0968 |
| 8 | 0.3818 | 0.1134 | 0.5807 | 0.5743 | 0.3312 | 0.3351 | 0.1068 |
| 50 | 0.3895 | 0.1116 | 0.5859 | 0.5877 | 0.3356 | 0.3343 | 0.1223 |

**Q1.** Quality improves from NFE 1 to 4, then *reverses*. NFE 4 is the optimum on raw RMSE, QRS RMSE,
derivative RMSE and F1 excess; NFE 50 is **worse than NFE 2** on every one of those. Two quantities move
the other way and are best at NFE 1–2: **QRS-energy deviation** (0.5431 at NFE 1 → 0.5877 at 50) and
**beats-ratio deviation** (0.0922 → 0.1223), i.e. more integration progressively over-shoots QRS energy and
deletes beats. This independently reproduces C0's compression target of **NFE 4** on 14 subjects rather
than 2, and adds that NFE 50 actively degrades.

**Q2 — across subjects: highly consistent.** Best NFE per subject: raw RMSE → **4 in 13 of 14** (w4p: 8);
QRS RMSE → **4 in 12 of 14**; derivative RMSE → **4 in 12 of 14**. **No subject has NFE 1 as its best on
raw RMSE or on F1 excess.** F1 excess is noisier (4 in 7, 8 in 4, 50 in 2, 2 in 1). The two calibration
quantities behave oppositely and consistently: QRS-energy deviation is best at **NFE 1 in 9 of 14**.

**Q3 — across sites: the shape is identical, the level is not.** Every site shows the same NFE-4 minimum in
raw RMSE. Absolute quality differs sharply by site (val F1 excess at NFE 4): head **0.458**, sternum
**0.433**, wrist **0.258**, ankle **0.216** — proximal sites roughly twice as good as distal.

**Q4 — where more NFE does not help.** For *waveform* fidelity, nowhere: every subject and site improves
from 1 to 4. For *calibration*, almost everywhere: 9 of 14 subjects have their best QRS-energy deviation at
NFE 1, and beats-ratio deviation degrades monotonically past NFE 2 for most. So "more steps" is not a
uniform good — it trades waveform error against amplitude calibration and beat production.

## 3. ECG-R → PPG-peak delay

**This is not pulse transit time.** It contains the electromechanical delay and the systolic rise time as
well as vascular propagation.

**Q5 — the distribution.** Matched with a one-to-one forward scan in [80, 800] ms; per-combination match
rates ran ~0.66–0.85, so 15–34 % of GT beats had no qualifying PPG peak at all.

| site | train median | IQR | p5–p95 | CV | val median |
|---|---:|---:|---|---:|---:|
| sternum | 375.0 ms | 226.6 | 179.7–695.3 | 0.396 | 375.0 |
| head | 304.7 ms | 242.2 | 164.1–695.3 | 0.447 | 281.2 |
| wrist | 351.6 ms | 281.2 | 156.2–718.8 | 0.448 | 304.7 |
| ankle | 398.4 ms | **132.8** | 226.6–687.5 | **0.317** | 421.9 |
| **ALL** | **375.0 ms** | **226.6** | 179.7–703.1 | 0.404 | 351.6 |

The site ordering — head < wrist < sternum < ankle — is physiologically sensible, and **PPG pulse events do
systematically follow ECG R events**. But the spread is large: p5–p95 covers ≈ 500 ms at every site.

**Q6 — where the variance lives.** Between subjects: per-subject medians span **297 ms (`trh`) to 516 ms
(`fex`)**, a 219 ms spread, SD 56 ms. Within subject: per-subject IQR is **164–258 ms**. **The within-subject
spread is comparable to the entire between-subject spread**, so knowing the subject would not resolve most
of the uncertainty. Between sites: medians differ by ~94 ms (train), far less than the within-site IQR.

**PPG foot (secondary, retained — 0.0 % failure, well under the 20 % abandonment threshold).** The foot
proxy is tighter than the peak: mean per-subject IQR **158.6 ms vs 205.4 ms**, a ~23 % reduction, consistent
with removing the systolic rise time. It is still far too wide to be a timing prior.

## 4. Timing-prior validation — train-only statistics, evaluated on `an0`/`k2s`

A validation subject's delay was never estimated from its own ECG. Train-only global 375.0 ms; site-specific
{sternum 375.0, head 304.7, wrist 351.6, ankle 398.4}; RR terciles with train-only edges [695.3, 835.9] ms →
{343.8, 351.6, 429.7}.

| predictor | MAE | median AE | ≤25 ms | ≤50 ms | ≤100 ms | ≤150 ms |
|---|---:|---:|---:|---:|---:|---:|
| A — global | 254.7 ms | 184.5 ms | 0.090 | 0.173 | 0.347 | 0.556 |
| **B — site-specific** | **245.7 ms** | **172.3 ms** | **0.115** | **0.218** | **0.426** | **0.593** |
| C — HR-conditioned | 254.5 ms | 184.0 ms | 0.096 | 0.175 | 0.345 | 0.551 |

**Q7 — yes, marginally.** Site-specific beats global on every threshold (0.218 vs 0.173 at 50 ms), which is
consistent with the site medians differing. The gain is real but small.

**Q8 — no.** Against the project's own frozen 50 ms event tolerance, the *best* predictor leaves **78 % of
validation GT beats with no predicted R-peak inside tolerance**, and the median absolute timing error is
172 ms — more than three times the tolerance. This reproduces, on 14 subjects, what S1 found on 2: a
PPG-peak-plus-constant-delay construction reached F1 0.2278 at 50 ms against a chance floor of 0.113.

**Q9 — neither a fixed delay nor the adaptive variant tested here is adequate.** The HR-conditioned variant
was *worse* than site-specific, and the dominant variance is within-subject (IQR 164–258 ms), not between
subjects or sites. Even a perfect per-subject, per-site constant would leave ~200 ms of residual IQR against
a 50 ms tolerance.

## 5. Qualitative observations from the frozen cohort — systematic only

Across the 448 stepwise figures and 448 R-centred zooms (no example was selected after seeing results):

1. **The NFE-4 optimum is visible as a shape, not just a number**: NFE 1 output is visibly smoother and
   lower-amplitude; NFE 2–4 sharpen; NFE 8 and especially NFE 50 add baseline roughness between beats
   without adding beat definition.
2. **Distal sites (wrist, ankle) produce visibly poorer beat structure** than head and sternum for the same
   subject and the same source draw — the level difference in the site table is visible directly.
3. **The PPG systolic peak reliably lands after the GT R marker in the zooms, but its offset visibly varies
   from beat to beat within a single 8 s window** — the qualitative counterpart of the within-subject IQR.

## 6. Feasibility verdict

**`PPG PEAK TIMING TOO UNSTABLE FOR DIRECT CONDITIONING`.**

Direct PPG-peak-minus-delay conditioning is not viable at the project's own 50 ms event tolerance under any
of the three preregistered train-only predictors, and the residual variance is dominantly within-subject
rather than in the between-subject or between-site terms that a fixed or site-aware prior can absorb.

## 7. Caveats

- **R→PPG-peak delay is not pure PTT** and is not called that anywhere above.
- **No GT R condition exists at inference.** The audit deliberately used only train-derived statistics for
  validation subjects.
- **No test access.** `kjd`/`ssx` never loaded; nothing here transfers to the test set.
- **No new method was tested.** V1 implements nothing and trains nothing.
- Only *direct peak-delay* conditioning is ruled out. Whether a **learned** PPG→R-timing extractor could do
  better is untested here and remains the open question S1 already recorded.
- Train-subject results are behaviour audit, site analysis and delay estimation only — never generalization
  evidence. Every aggregate above separates or labels the split.

## 8. Next suggested architecture — recommendation only, not implemented

If event-timing conditioning is pursued at all, the evidence points away from a hand-built delay prior and
toward a **learned PPG→R-timing predictor supervised on train subjects only**, whose output would have to be
validated against the same 50 ms tolerance before any generator is conditioned on it. V1 does not implement,
select or train such a thing.

## Artifacts

`artifacts/v1_stepwise_visualization/`: `provenance.json`, `provenance_delay.json`,
`checkpoint_manifest.json`, `cohort_manifest.csv`, `predictions_manifest.csv`, `metrics_by_window.csv`,
`metrics_by_subject.csv`, `metrics_by_site.csv`, `r_to_ppg_peak_delays.csv`, `delay_summary.csv`,
`timing_prior_validation.csv`, `timing_prior_summary.csv`, `skipped_windows.csv`, `figures/`,
`beat_zooms/`, `dashboard/index.html`.
