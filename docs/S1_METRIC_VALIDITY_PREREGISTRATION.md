# S1 — Metric-Validity Audit (development-only) — PREREGISTRATION

**Status:** frozen at this commit. Pushed **before** any S1 real-data metric is computed.
**Type:** zero-training, frozen-checkpoint / no-checkpoint analysis. **NO TRAINING of any kind.**
**Population:** WildPPG **development validation only** — subjects `an0`, `k2s`.
**Firewall:** WildPPG test subjects `kjd`, `ssx` are **never loaded** in S1, in any form.
**Scope discipline:** S1 audits the *instruments*. It does not test any hypothesis about the models, and it
must not be used to rank arms or to select a method.

---

## 0. Why S1 exists, and what it is not

Four accumulated diagnostics (X0, X2, X3-G0, X4-0) rest on a small set of beat-level statistics. Two of
those statistics have since been shown to be conditional on successful ≤50 ms R-peak matching
(`docs/METRIC_SEMANTICS.md`, commit `92bf6dd`), and a third — the ±150 ms per-beat oracle translation —
grants a search over 39 candidate shifts with **no published null calibration anywhere in this repository**.

Before any further mechanism claim is made or any new model is trained, the instruments themselves must be
shown to be capable of measuring what they are quoted as measuring.

**S1 is therefore gated on a single prior question: can the frozen beat-level metric reward correct beat
placement at all?** If it cannot, every event-F1 number in this project is uninterpretable and no mechanism
interpretation built on them is worth running.

S1 is **not**: a model comparison, a method proposal, a test-set evaluation, a signal-quality study, or the
joint-fidelity experiment. Those are separate protocols with their own preregistrations.

---

## 1. Provisional central hypothesis (recorded, not tested here)

The programme's working hypothesis is provisionally reset to:

> **There is a joint-fidelity (Pareto) gap between event fidelity and waveform fidelity: no method in this
> project or in the published PPG→ECG literature achieves competent beat placement and competent waveform
> fidelity simultaneously.**

This replaces "can PPG→ECG be made one-step without losing physiology", which the record has answered
(NFE 1 → 50 moves iMF event F1 by +0.010; the one-step arms already lead on beats).

Two consequences bind S1's reporting:

- **Event F1 is retained as an instrument but is not a standalone objective or headline.** Its current
  leader in this project is a near-flat predictor. A number produced by S1 must never be presented as
  progress on its own.
- **A headline morphology figure must be accompanied by a detector-independent all-GT-beat statistic**
  (`oracle_corr` and/or `same_coord_corr`) on the same population, per `docs/METRIC_SEMANTICS.md` §3.

S1 tests none of this. It is recorded here so that S1's design choices are legible.

---

## 2. Population, frozen and reused

S1 reuses the **X4-0 stage-B population verbatim**; it does not re-select windows.

- Selection: `event_reliability.select_subset(salt="x4-event-nfe-v2", subject, n_total, n_take=1024,
  exclude=PREVIEWED_WINDOWS)`, SHA256 ranking over `"{salt}|{subject}|{window_index}"`.
- Result: **2,048 windows** (an0 1,024 + k2s 1,024), **19,834 GT beats**, recorded in
  `artifacts/x4_0_event_reliability/nfe_subset.json`.
- The four pre-viewed windows — (`an0`, 9066), (`an0`, 18138), (`k2s`, 5852), (`k2s`, 16436) — are excluded
  by construction (`docs/X4_0_PREPREREG_VISUAL_AUDIT.md`).
- `an0`/`k2s` are **development validation**, never pristine confirmatory validation.

Where S1 needs generated signals, the source is the **frozen** `outputs/a4_imeanflow_wildppg_seed42/` and
`outputs/a4_otcfm_wildppg_seed42/` checkpoints under forward inference with **Gaussian source seed 0**.
The recorded X4-0 headline table is 4-seed pooled; seed 0 runs ≈ 0.005–0.007 F1 above the pool, so
**S1 numbers must not be set against the recorded pooled values** — every S1 comparison recomputes its own
anchor on this population at seed 0.

`assert_no_test_subjects(...)` is called before first data access in every S1 script.

---

## 3. Pre-preregistration exposure disclosure (mandatory)

The following real-data quantities on `an0`/`k2s` were **already computed before this preregistration**, in
an exploratory local audit run at the user's direct instruction on 2026-08-31 (summary at
`/tmp/overnight_diagnostic_summary.md`, not committed; scripts under a session scratchpad, not committed).
They are **exploratory, not preregistered**, and are disclosed here so that S1's results are read correctly:

- PPG-systolic-peak → GT-R-peak delay distribution (pooled median 296.9 ms, IQR 164.1, within-window SD
  85.3 ms, across-window SD 82.1 ms; 31.8 % of GT beats with no PPG peak in (0,500] ms).
- A PPG-peak-plus-constant-delay event baseline at 50/75/150 ms in four delay regimes
  (macro F1 0.0351 / 0.2194 / 0.2278 / 0.2385, oracle constant-shift bound 0.2698).
- Count-matched phase-randomised and circular-shift chance floors (macro F1 ≈ 0.113–0.122 at 50 ms,
  ≈ 0.341–0.365 at 150 ms).
- Recomputed seed-0 anchors on this population: iMF-1 0.4144, iMF-8 0.4341, OT-CFM-50 0.4828 at 50 ms.
- The `morph` matched-pair denominator census (7,910 / 19,834 GT beats at iMF-8 seed 0).
- An amplitude-anchored three-way split of unmatched GT beats (ABSENT 65.9 % / WEAK 24.0 % /
  DETECTOR-MISS 10.1 %) whose ABSENT class is now known to be **confounded with displacement** — it scores
  amplitude within ±100 ms of the GT position, so a beat displaced by more than 100 ms is counted as
  absent. Repairing that confound is item **S1.4** below.
- A dev-side GT plausibility check (0 / 2,048 windows with an implausible beat count).

**No S1 threshold, tolerance, template, subset or gate below was chosen after seeing any of these.** The
S1.4 repair is a correction of a known design defect, not a re-tuning toward a result. Where an S1 item
re-derives a quantity that appears above, the S1 value is the record and the exploratory value is
disclosed alongside it.

---

## 4. G1 — HARD GATE: can the metric reward correct beat placement?

**This gate runs first and alone. Nothing else in S1 runs until it passes.**

### 4.1 Principle

Stamp a known-good QRS at the exact ground-truth R-peak positions and score the result with the frozen,
unmodified metric. A metric that cannot return ≈ 1.0 for a signal whose beats are in exactly the right
place cannot be used to argue that a model's beats are in the wrong place.

### 4.2 Template provenance — the binding constraint

**The template must not be derived from validation data.** Deriving it from `an0`/`k2s` would build the
instrument out of the data the instrument is being validated on. Two provenances are permitted and both are
specified in full below; the gate is decided on the first.

**Template A (PRIMARY) — train-only frozen canonical template.**

1. Subjects: the **10 training subjects excluding the two the WildPPG authors flag as noisy-ECG**, i.e.
   `e61, l38, n31, ngh, p9p, qm9, trh, tz8, u7y, w4p` (excluding `fex`, `p5d`; the flag and the decision not
   to drop them from training are recorded in `data/manifests/split_a4_wildppg_seed42.json → extra.note`).
   *This exclusion is a pre-committed rule about template quality, not a data selection: no S1 metric is
   computed on training subjects.*
2. Windows: `select_subset(salt="s1-template-v1", subject, n_total, n_take=256, exclude=())` per subject →
   2,560 training windows. Same SHA256 ranking primitive as §2.
3. Beats: `detect_rpeaks` (frozen, unmodified) on each GT ECG window; for every detected peak with a full
   window available, extract `beat_window(sig, r, fs, before_s=0.25, after_s=0.40)` — the identical window
   `morphology_corr` uses (83 samples at 128 Hz).
4. Template: the **element-wise median** across all extracted beats. Amplitude fixed to the **median
   peak-to-peak** of the same beat set. No smoothing, no fitting, no per-window adaptation.
5. Frozen: saved to `artifacts/s1_metric_validity/template_A.npy`; its SHA256, length, peak-to-peak and
   the beat count it was built from are recorded in the S1 report.

**Template C (SECONDARY) — fixed analytic template, no data dependence at all.**
Ricker (Mexican-hat) wavelet, fully specified: `w(t) = (1 − (t/σ)²)·exp(−t²/(2σ²))`, σ = 10 ms, sampled at
128 Hz on t ∈ [−80, +80] ms (21 samples), normalised to unit peak, then scaled to Template A's
peak-to-peak. Reproducible from this paragraph alone.

### 4.3 Stamping arms — all four reported, gate decided on T-A

Ground-truth R-peak **positions** for the val windows come from the frozen `detect_rpeaks` on the GT ECG.
(Using val GT *positions* is the point of the test; the constraint is on the template *shape*, which is
train-only or analytic.)

| arm | template | baseline | note |
|---|---|---|---|
| **T-A (GATE)** | A, full beat (83 samp) | zeros | superposition where beats overlap |
| T-B | A, QRS-only, t ∈ [−80, +120] ms | zeros | overlap-free at the observed minimum RR (320 ms) |
| T-C | C, analytic | zeros | data-independent control |
| T-D | A, full beat | val GT's < 1 Hz component | realism arm; low-frequency baseline only, no beat shape |

Beats overlap in T-A/T-D at high heart rate (83-sample template vs a 41-sample minimum RR); overlap is
handled by **superposition**, which is declared here and is what a real ECG does in the T–P region.
T-B exists so that the gate verdict cannot be an artifact of overlap.

### 4.4 Scoring — frozen and unmodified

`detect_rpeaks` → `match_rpeaks(..., tol_ms=50.0)` → `prf`, exactly as in
`src/ppg2ecg/evaluation/rpeaks.py`, applied identically to the GT and to the stamped signal. Macro mean of
per-window F1 with equal subject weight, over the 2,048-window population. 75 ms and 150 ms are reported as
diagnostics only; the gate is at 50 ms.

### 4.5 The gate

| | criterion |
|---|---|
| **PASS** | T-A macro F1 **≥ 0.95** at 50 ms |
| **FAIL** | T-A macro F1 **< 0.95** at 50 ms |

Also reported unconditionally, pass or fail: per-window F1 distribution (mean, SD, deciles, count at
F1 = 1.0, count at F1 < 0.5), `beats_ratio`, precision, recall, the same for T-B/T-C/T-D, and the 75/150 ms
values.

### 4.6 On FAIL — mandatory STOP

**If the gate fails, S1 stops immediately. Items S1.2 through S1.6 are NOT run, and no mechanism
interpretation is produced or revised.** The deliverable becomes an instrument report containing only:

1. the four arms' numbers, and
2. two pre-declared diagnostics of the gate itself: (a) a template-amplitude sweep at
   {0.25, 0.5, 1, 2, 4} × the frozen amplitude, and (b) a per-window characterisation of the failures
   (how many detected peaks, where they fell relative to the stamped positions).

Nothing else. A failed gate is a finding about this project's entire event-metric axis and must be reported
as such, not worked around. **No replacement metric may be designed inside S1** — that requires its own
preregistration, written after and citing the S1 instrument report.

---

## 5. Items that run only if G1 passes

### S1.2 — Zero-parameter DSP floor, scored as a waveform

The project has never compared any of its 28 training runs to a no-parameter method.
`nk.ppg_findpeaks(nk.ppg_clean(ppg))` at library defaults, no tuning, plus a single constant delay δ,
**template-stamped with Template A** so that it is scored through the identical detector path as a model
output rather than as a bare peak train.

Delay regimes, all reported, none selected: δ = 0; δ = the median GT-R-to-next-PPG-peak delay estimated on
**training subjects only**; and a leave-one-subject-out variant across the two val subjects. Tolerances
50/75/150 ms. Reported against the count-matched chance floor of §S1.4c.

*Reporting rule:* this is a floor, not a competitor. It is a **lower bound on PPG-only methods** — one
untuned detector, constant delay — and must be labelled so.

### S1.3 — Fidelity-axis census on the frozen arms, development only

For the frozen arms available on this population (iMF at NFE 1/4/8/50, OT-CFM Heun-25, and the frozen MSE
regressor if it can be evaluated without training), compute on the **same 2,048 dev windows, seed 0**:
`morph` **with its matched-beat coverage fraction**, `same_coord_corr`, `oracle_corr`, `oracle_absent`,
`oracle_qrs_energy_median`, `qrs_width_err_ms`, `beats_ratio`, `hf_ratio`, event F1/precision/recall.

**Purpose:** to make the joint-fidelity Pareto picture visible on a single population with stated
denominators. **S1 does not set any gate threshold from these numbers.** To keep the next protocol free of
outcome-driven thresholds, the *rule* for a future joint-fidelity gate is fixed **here, before the numbers
exist**, as a function of the frozen OT-CFM Heun-25 reference arm's dev values `R`:

- matched-beat morphology correlation ≥ 0.80 × R(morph), with matched-beat coverage ≥ 0.80 × R(coverage)
- `oracle_qrs_energy_median` ≥ 0.60 × R(oracle_qrs_energy_median)
- `qrs_width_err_ms` ≤ 1.50 × R(qrs_width_err_ms)
- `beats_ratio` ∈ [0.90, 1.10]  *(absolute, physiological, not anchored)*

The multipliers are committed now. No multiplier may be changed after S1.3 is computed.

### S1.4 — Displacement versus absence, with the oracle null calibrated

Settles a question two independent analyses left open, and repairs the confound disclosed in §3.

**(a) Displacement-aware three-way split.** For every unmatched GT beat, classify by searching the
generated signal over ±150 ms — **not ±100 ms** — for a detected peak and for a supra-threshold deflection:
`DISPLACED` (a detected peak at 50–150 ms), `WEAK` (a deflection present but below the detectability
threshold), `ABSENT` (neither). Thresholds are anchored to the GT distribution on the same windows
(the deflection threshold is the 5th percentile of GT matched-beat `amp_rel`), fixed before computation.

**(b) Oracle null calibration.** `oracle_corr` maximises over 39 candidate integer shifts in ±150 ms with
no published floor. Compute the same maximisation against a **randomly chosen other GT beat from the same
window** (20 draws, `default_rng(20260901)`), giving the chance level of the oracle's gain. If the gain over
`same_coord_corr` is at or near this floor, the "beats are present but displaced" reading built on
`oracle_corr` does not survive and must be withdrawn.

**(c) Count-matched event chance floor.** A rate-matched, random-phase peak train with the same beat count
per window (20 draws, `default_rng(20260901)`), and a circular-shift variant. Every event F1 in the S1
report is accompanied by its excess over this floor. **Raw event F1 may not be reported without it.**

### S1.5 — C2 perturbation re-analysis

`scripts/analyze_x4_0_event_reliability.py:335` computes the two condition-perturbation arms
(`source`: PPG fixed, Gaussian source re-drawn; `ppg_shuffle`: Gaussian source fixed and bit-identical, PPG
deranged). In the `ppg_shuffle` arm any source-determined peak lands at the **same integer index** and
contributes exactly 0 ms to `timing_disagreement_ms`, an outcome structurally impossible in the `source`
arm, so the two arms' timing statistics are not comparable as published.

Recompute both arms (i) excluding exact-zero Δt pairs, (ii) with an uncensored matcher, and (iii) against a
permutation chance floor. Report `wavePCC`, event F1 and timing for both arms with their denominators.

**Constraint:** the preregistered prohibition (X4-0 §10) on reading these two arms as a calibrated causal
ratio **remains in force**. S1.5 removes a comparability defect; it does not license the causal reading.
S1.5 also does **not** compute the band-limited or QRS-masked pred-vs-pred correlation that would separate
"the source determines beat placement" from "the source determines a low-frequency component that dominates
a whole-window correlation" — that is a new metric and requires its own preregistration.

### S1.6 — Development-side GT reliability (`an0`/`k2s` only)

Two-detector agreement F1 at 50 ms on the **reference** ECG, per subject, plus the RR-plausibility and
beat-count-plausibility census. Both detectors reported; **selecting the more favourable one is
prohibited.**

**The `kjd`/`ssx` GT reliability audit is explicitly NOT part of S1** and is deferred to a separate
protocol, for two reasons: S1 is development-only, and a test-side quality audit must have its QA rules
frozen before any new model's test performance is seen. That separate protocol must fix its exclusion rules
in advance and must report **full-test and GT-quality-stratified results together**; excluding a subject
after seeing performance is prohibited.

---

## 6. Statistics

Subject-stratified bootstrap, 2,000 resamples, seed **20260901**, equal subject weight, over windows within
subject. Intervals are reported for every headline S1 number. No significance test is claimed unless it is
actually run; "indistinguishable" may not be written without an interval or a test.

## 7. Analysis freeze

Every threshold, tolerance, template, subset, delay regime, null construction and gate above is fixed by
this commit. Any analysis added after data are seen must be labelled **POST-HOC** in the script, the output
JSON and the report, and must be **additive, never substitutive**. Every pre-registered grid is reported in
full; reporting the best cell of a grid is prohibited.

## 8. Compute and constraints

CPU only for G1, S1.2, S1.4b, S1.6. Frozen-checkpoint forward inference for S1.3, S1.4a, S1.4c, S1.5;
predictions cached once and shared across items. **No weight update anywhere. No new model. No training
run. No checkpoint written.** `external/PENGUIN` (`6cd70cd`) and `external/iMeanFlow` (`bf60cd7`) stay
byte-identical. Checkpoints, predictions and raw data never enter git.

## 9. Deliverables

- `docs/S1_METRIC_VALIDITY_REPORT.md`
- `artifacts/s1_metric_validity/` (gitignored): per-item CSV/JSON, `template_A.npy` + checksum, figures
- `scripts/analyze_s1_metric_validity.py`, plus any primitive added under `src/ppg2ecg/evaluation/`
- tests under `tests/` for every new primitive

## 10. Stop rules

1. **G1 FAIL → immediate stop.** §4.6 applies in full.
2. Any item whose required input is unavailable stops that item and is reported as unavailable; no proxy is
   substituted silently.
3. **No new training is started before the S1 report is written and reviewed** — including the
   joint-fidelity experiment, any beat-head or anchored-generator arm, and any external-baseline
   reproduction.
4. S1 produces no method recommendation. Whatever it finds, the next step is a separate preregistration.
