# N4 — Is the generator's timing variability a defect, or a posterior?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `a429ed8`. Pins `6cd70cd` / `bf60cd7c` unmodified. **No training, no weight update, no optimizer.**

---

## 1. Question

X4-0 measured that, with the PPG held fixed, **changing only the Gaussian source changes event identity**:
median predicted-to-predicted event F1 between two sources **0.30**, per-window beat count SD **1.2–1.8**,
and GT-anchored timing SD **≈ 50–57 ms**. X4-0 scored this as **PERSISTENT SOURCE SENSITIVITY** — a defect.

But three independent measurements in this repository say the *true* timing posterior is wide:

| | |
|---|---|
| V1 | PPG-peak → ECG-R delay **IQR 227 ms**, within-subject variance dominant; 50 ms coverage 0.22 |
| R1 | PPG-only R detection **F1@50 = 0.62**, residual jitter 50–150 ms |
| N1 / N2 / N3 | PPG determines *when* to that accuracy and carries essentially nothing about *what* |

If `p(beat time | PPG)` really is that broad, then a **correct** conditional sampler *must* produce
different beat times for different sources. X4-0's "defect" and "calibrated posterior sampling" predict
the same measurements and were never separated.

> **Q.** Is the frozen generator's source-induced timing spread **calibrated** against its actual timing
> error, or is it noise of the wrong size?

- **Calibrated** — X4-0's finding is reinterpreted, the existing models are already sampling a posterior
  nobody scores them as sampling, and a method whose *output object* is a calibrated timing posterior
  (rather than a waveform) has a measured foundation.
- **Not calibrated** — the variability is the wrong size for the uncertainty, X4-0's defect reading stands,
  and the timing-posterior direction has no more support than the shape direction N2/N3 closed.

## 2. No training, no new model

The only network used is the **frozen** iMeanFlow generator X4-0, R2 and R3 all used
(`outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt`, state sha256 `47d7ccb9…`), loaded in `eval()`
with `requires_grad=False`. N4 re-runs inference and computes statistics. Nothing is fitted, and no
threshold is chosen after seeing a calibration number.

## 3. Design

- **Cohort**: X4-0's frozen source subset — `artifacts/x4_0_event_reliability/source_subset.json`,
  256 windows each from `an0` and `k2s` (512 total), asserted element-wise.
- **Sources**: X4-0's `SOURCE_SEEDS = range(32)`, via the same `source_bank`, at **NFE 1 and NFE 4**
  (X4-0's own NFE axis; both published).
- Per GT R peak, collect the raw per-source offsets `o_k = t_k − t_GT` in ms for every source whose
  nearest predicted peak lies within X4-0's frozen `GT_ANCHOR_MS` window. A beat enters the analysis only
  if **≥ 16 of 32** sources detect it — X4-0's own frozen filter, reused unchanged.

**Censoring is declared, not hidden.** Beats detected by fewer than 16 sources are excluded from the
calibration statistics; their count and the overall detection rate are reported, and a calibration result
is only ever a statement about the beats the model does produce.

## 4. Calibration statistics — fixed here

For each eligible GT beat, with `K` detecting sources and offsets `{o_k}`:

| statistic | definition |
|---|---|
| **coverage(α)** | is `0` inside the central `1 − α` empirical interval of `{o_k}`? Computed at **α = 0.50, 0.20, 0.10** |
| **PIT** | rank of `0` among `{o_k}`, scaled to `(0, 1)`; uniform iff calibrated |
| spread | `SD(o_k)` |
| bias | `mean(o_k)` |

Aggregation is the **subject-macro** mean used throughout this program, with the paired
subject-clustered bootstrap (2,000 replicates, seed 20260911) for intervals.

## 5. Decision rule — fixed here

Let `Δ(α) = coverage(α) − (1 − α)` at the three levels.

| verdict | rule |
|---|---|
| **TIMING POSTERIOR CALIBRATED** | `|Δ(α)| ≤ 0.10` at **all three** levels |
| **OVERCONFIDENT** | `Δ(α) ≤ −0.10` at **≥ 2** levels (spread too narrow for the error) |
| **UNDERCONFIDENT** | `Δ(α) ≥ +0.10` at **≥ 2** levels (spread too wide) |
| **MISCALIBRATED (other)** | anything else |

`0.10` is the tolerance on a probability scale; it is deliberately loose, because the claim being tested is
"the right order of magnitude", not "perfectly calibrated".

**Secondary, recorded and unable to change the verdict:** the Spearman correlation between per-beat spread
and per-beat `|bias|`. A calibrated ensemble should show a positive relation — the model should be more
uncertain where it is more wrong. A spread that is the right size *on average* but carries no per-beat
information is a materially weaker result and must be reported in those words.

## 6. What a positive result does and does not license

- It does **not** mean the generator is good, that the timing is accurate, or that anything here is
  deployable. Calibration is a statement about the *relation* between spread and error, not about error.
- It does **not** by itself constitute a method. It establishes the premise that a timing-posterior output
  object is measuring something real, which the shape direction (N2/N3) failed to establish.
- It **does** reinterpret X4-0's PERSISTENT SOURCE SENSITIVITY, which stays a frozen verdict on its own
  terms; N4 adds the reading X4-0 could not test, and says so.

## 7. Firewalls

- `kjd` / `ssx` never loaded; `assert_no_test_subjects` at entry; `test_subjects_loaded: []`.
- Frozen generator state sha256 asserted before inference; A4 md5 `31c042d2…` re-checked; C2 deferred.
- `an0` / `k2s` are development validation with four pre-viewed windows (X4-0 §"Pre-prereg visual audit");
  this is not a test-set result and every table says so.
- No checkpoint, prediction or raw data enters git.

## 8. Cost

512 windows × 32 sources at NFE 1 and 4 on a frozen generator. Estimated minutes.

## 9. Reporting

`docs/N4_TIMING_CALIBRATION_REPORT.md`: coverage at all three levels with intervals, the PIT histogram,
the spread–error relation, the censoring rate, the §5 verdict, and an explicit statement of whether the
timing-posterior method direction is supported or closed.
