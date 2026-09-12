# N1 — The supervision-matched null method: does "detector + template" beat every generator?

**Preregistration. Frozen on commit and push; never edited afterwards.**

Start HEAD `2666501`. Pins `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7c` — unmodified.
**No training, no weight update, no optimizer is constructed in this stage.**

---

## 1. Question

Every generative arm in this program is scored on chance-corrected beat F1 (`f1_excess`) and QRS-core
structure (`qrs_deriv_rmse` = S4, `qrs_curvature_err` = S5) on one frozen 2,048-window cohort. On that
cohort, **at NFE 4** — the budget every R2/R3 gate was decided at — the trained arms sit at `f1_excess`
**+0.3176 (B)**, **+0.3369 (ADD)**, **+0.3341 (TF-TRUE)**, **+0.3349 (GTF-CONST)**, **+0.3582 (GTF-TRUE)**
(`artifacts/r3_rhythm_fusion/event_metrics.csv`, read before this preregistration was written and quoted
here so the bar is fixed in the open).

Two frozen facts from this repository bound what a training-free composition would score:

- **S1**: stamping the cropped QRS template (T-B) at *exact* GT R positions gives macro F1 **0.9993** at
  50 ms (`S1_G1_METRIC_VALIDITY_REPORT.md`).
- **R1**: the PPG-only Global-TCN rhythm scaffold detects R events at F1@50 ≈ **0.62**
  (`R1_..._REPORT.md`).

> **Q.** Does *R1 events + fixed QRS template stamp* — zero training, CPU minutes, no generator — beat
> every trained generative arm on the event axis of the very cohort those arms were judged on?

**Why this must be answered before any method paper.** If the null method wins on `f1_excess`, then no
result in this program, and no published result that reports this metric family, has demonstrated that a
*generative* model adds anything on the event axis. Every method proposal — including the beat-canonical
decomposition currently under consideration — must clear this bar first or be withdrawn. The assessment
`ASSESSMENT_TOPTIER_AND_IDEATION_2026-09-03.md` ranks this experiment 0 and predicts an excess of
**+0.48–0.58** from repository numbers; an exploratory read-only join over 360 shared windows reported
scaffold macro F1 0.620 vs B 0.455 vs GTF-TRUE 0.502. **That exploratory number is not this result** and
does not relieve the preregistered rule below.

## 2. Second question — is the structure gate informative for stamped signals?

R2 and R3 both returned "event gain with structure trade-off" because S4/S5 worsened. S4/S5 are evaluated
at peaks detected **in the signal being scored**, so a stamped signal is measured at its own stamp
centres. If T-B stamped at *exact GT* positions does not score clearly better than the generator B on S4,
then S4/S5 cannot discriminate a perfectly-placed canonical beat from a generator's output, and the
structure gate that decided R2 and R3 is **uninformative for stamping-type methods**. That is a statement
about the metric, not about any method, and it is gated separately in §6.

## 3. Arms — all training-free

| arm | events from | stamp | role |
|---|---|---|---|
| **N-R1** | R1 Global-TCN scaffold, `extract_events(threshold 0.35, refractory 32)` | T-B (QRS crop) | **the null method** |
| **N-R1-A** | same | T-A (full beat, overlap permitted) | template-extent control |
| **N-SHUFFLE** | R1 scaffold of the **partner** window (R2/R3 derangement, salt `r2-rhythm-shuffle-v1`) | T-B | window-specificity control |
| **N-GT (GT-R leakage; diagnostic only)** | ground-truth R peaks | T-B | ceiling / §2 gate |
| **N-CONST** | uniform grid at the window's own median RR, phase from the first R1 event | T-B | rhythm-without-placement control |

Comparators are the **frozen, already-published** rows of `artifacts/r3_rhythm_fusion/event_metrics.csv`
and `artifacts/r2_rhythm_transfer/` — **B**, **ADD**, **GTF-TRUE**, **TF-TRUE**, **GTF-CONST** at NFE 4.
Nothing is regenerated and no generator is loaded.

## 4. Frozen cohort and scoring — identical to R2/R3, imported not reimplemented

- Windows: `an0`, `k2s`, `ER.select_subset(salt="x4-event-nfe-v2", n_take=1024)` per subject, asserted
  element-wise against `artifacts/x4_0_event_reliability/nfe_subset.json`. **2,048 windows, 19,834 GT
  beats** — the run aborts if either differs.
- Scoring: `scripts/r2_evaluate.py: score()` and `macro_rows()`, imported unchanged. Metrics
  `f1_excess` (chance-corrected, 50 ms), `missing`, `spurious`, `beats_ratio_dev`, `raw_qrs_rmse`,
  `qrs_deriv_rmse` (S4), `qrs_curvature_err` (S5).
- Templates: the frozen S1 artifacts — `artifacts/s1_metric_validity/template_A.npy`
  sha256 `1a67569f8a02bc0027c0a60c4575d297dc2bc40eb0c3e285b9acf82daafd51eb`, T-B crop sha256
  `6f059015812308a9d71b4c72e08a6eeed68ee2ca29107bc801097d8a1748e595`. Rebuilt from disk and the hashes
  asserted; **not recomputed from data**.
- Uncertainty: paired subject-stratified bootstrap, `paired_stats.paired_subject_bootstrap`, 2,000
  replicates, the same settings R2/R3 used.

## 5. Primary decision rule — fixed here

Comparator is **GTF-TRUE at NFE 4**, the best event score any arm in this program has achieved.

| verdict | rule |
|---|---|
| **NULL DOMINATES** | `N-R1 − GTF-TRUE` on `f1_excess` has its paired 95 % CI entirely > 0 **and** point ≥ **+0.02** |
| **NULL COMPETITIVE** | CI entirely > 0 but point < +0.02 |
| **NULL DOES NOT DOMINATE** | CI includes 0, or entirely < 0 |

`+0.02` is the same minimal effect R2 and R3 were held to (`GATE_MIN_EFFECT`), reused so the null method
is judged by the bar the trained arms were judged by.

**Window-specificity qualifier.** The verdict carries "(not window-specific)" if
`N-R1 − N-SHUFFLE` on `f1_excess` does **not** have its CI entirely > 0 — i.e. the null's score does not
depend on which window's PPG produced the events.

## 6. Secondary gate — is S4/S5 informative? (fixed here)

| outcome | rule |
|---|---|
| **STRUCTURE GATE INFORMATIVE** | `N-GT − B` on S4 has its paired CI entirely on the improving side |
| **STRUCTURE GATE UNINFORMATIVE** | otherwise |

If UNINFORMATIVE, the report must state that R2's and R3's structure trade-off verdicts cannot be
transferred to stamping-type methods, and that any successor preregistration needs a coverage-explicit
per-beat structure metric. This gate is **diagnostic**: it changes no R2/R3 verdict, which stay frozen.

## 7. What this stage may and may not conclude

- It may conclude that a training-free composition out-scores the trained arms **on this cohort, on this
  metric family, at 50 ms**.
- It may **not** conclude that PPG→ECG is solved, that stamping is a deployable method, or that the
  generative arms are worthless — the null method inherits R1's timing and produces a canonical beat, so
  it cannot express per-beat morphology at all. Its morphology behaviour is reported, never sold.
- It is **not** a test-set result: `an0` / `k2s` are development validation and four of their windows were
  viewed before X4-0 was frozen. `kjd` / `ssx` are not loaded.

## 8. Firewalls

- No training. No weight update. No optimizer. Asserted by the absence of any `.backward()` /
  `torch.optim` construction in the N1 entry point, and by test.
- `kjd` / `ssx` never loaded — `ER.assert_no_test_subjects` at entry; `test_subjects_loaded: []`.
- N-GT is labelled **"(GT-R leakage; diagnostic only)"** in every table and figure.
- Frozen A4 checkpoint md5 `31c042d291052fbb6dc15263ad316be2` and the R2/R3 artifacts are read-only;
  their sha256 are recorded before use.
- C2 remains deferred. No prediction, checkpoint or raw data enters git.

## 9. Cost

CPU minutes plus one TCN forward pass over 2,048 windows (seconds). No generator inference.

## 10. Reporting

`docs/N1_NULL_METHOD_STAMPING_BASELINE_REPORT.md`: the arm × metric table against the frozen R2/R3 rows,
the §5 verdict with its qualifier, the §6 gate, the morphology behaviour of every stamped arm, and an
explicit statement of which previously-published readings in this program survive the result.
