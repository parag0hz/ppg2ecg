# S1 Amendment 1 — Hard-Gate Deconfounding Before Execution

**Original preregistration:** `docs/S1_METRIC_VALIDITY_PREREGISTRATION.md`, commit **`b749339`**.
**Status of this amendment:** written, committed and pushed **before any S1 G1 real-data result was
computed**. At the time of writing, no stamped signal has been scored, no Template A has been built, and
no G1 number of any kind exists.

**The original preregistration document is not modified.** Its text stays byte-identical at `b749339`.
Every change is recorded here, as an amendment, so that the pre-committed artifact remains auditable.

## Why this amendment exists

Not because of a result — there is none. Because a **logical defect in the hard gate was found by reading
the preregistration itself**: a property the preregistration already states (full-beat template 83 samples
vs. an observed minimum RR of ~41 samples, §4.3) makes the primary gate arm unable to answer the question
the gate is for. That is a design error discoverable, and discovered, before execution.

**What is NOT changed:** the pass threshold (0.95), the primary tolerance (50 ms), the population (the
frozen X4-0 stage-B 2,048 windows on `an0`/`k2s`), the detector (`detect_rpeaks`), the matcher
(`match_rpeaks`, one-to-one, `tol_ms=50`), the aggregation (equal-subject macro mean of per-window F1),
the subjects, the subset salts, the test firewall, the stop rules, and the prohibition on designing a
replacement metric inside S1.

---

## 1A. The hard gate moves from T-A to T-B

### The defect

The preregistration's primary gate arm **T-A** stamps the full 83-sample canonical beat at every GT R-peak
on a zero baseline, with overlap handled by superposition (§4.3). The observed minimum RR on this
population is ≈ 41 samples (320 ms), so at high heart rate **T-A cannot avoid template superposition** —
the preregistration says so itself.

That confounds the gate. If T-A scores below 0.95, the result is consistent with two incompatible
explanations that T-A cannot separate:

1. the frozen detector + 50 ms matcher does not reward correctly-placed beats — *the finding the gate is
   for*; or
2. summing overlapping synthetic full beats produces a waveform whose T–P region misleads the detector —
   *an artifact of the stamping construction, saying nothing about the metric*.

A hard gate that cannot distinguish its own finding from its own construction artifact is not a hard gate.

### The repair

The gate question is narrowed to exactly what a metric-validity gate should ask:

> **Is an unmistakable QRS-like event, stamped at the exact GT R-peak position with no superposition,
> rewarded by the frozen detector and matcher?**

**The primary hard gate is now T-B.** T-A, T-C and T-D are retained, unchanged, as diagnostic controls.

### T-B, fully specified

| property | value |
|---|---|
| template | the QRS-only portion of the frozen Template A (§1B) |
| interval | **t ∈ [−80 ms, +120 ms]**, frozen |
| discrete support | **26 samples** at 128 Hz (see §1B for the rounding rule) |
| baseline | zeros |
| positions | exact GT R-peaks from the frozen `detect_rpeaks` on the GT ECG |
| superposition | **none** — asserted, not assumed (§1B, unit test) |
| detector | frozen `detect_rpeaks`, unmodified |
| matcher | frozen `match_rpeaks`, one-to-one, `tol_ms = 50.0` |
| scoring | frozen `prf`; equal-subject macro mean of per-window F1 |
| population | the frozen X4-0 stage-B 2,048 windows (an0 1,024 + k2s 1,024) |

**PASS:** T-B macro F1 **≥ 0.95** at 50 ms.
**FAIL:** T-B macro F1 **< 0.95** at 50 ms → immediate STOP; S1.2–S1.6 are not run; §4.6 of the
preregistration applies in full.

### The controls, and how each is read

| arm | role after this amendment |
|---|---|
| **T-B** | **PRIMARY HARD GATE** — overlap-free exact-placement sanity check |
| T-A | full-beat overlap / superposition sensitivity control |
| T-C | analytic, data-independent morphology control |
| T-D | low-frequency-baseline realism control |

**Pre-declared reading of a T-A/T-B split.** If **T-A < 0.95 while T-B ≥ 0.95, G1 is a PASS**, and the
licensed conclusion is: *summing overlapping synthetic full beats can perturb detector behaviour, but the
detector and matcher pass the clean QRS-placement sanity check.* T-A is then a statement about the
stamping construction, not about the metric.

The gate verdict is decided on T-B alone. T-A may not be used for the gate decision after this amendment.

---

## 1B. Template A construction and scaling, frozen exactly

The preregistration's phrasing "element-wise median … amplitude fixed to median peak-to-peak" admits more
than one implementation. The operation is fixed here, before Template A is built.

### Scaling

For the extracted train-only beats `x_b` (each of length 83, all aligned on their own detected R-peak):

```
T_raw[t] = median_b  x_b[t]                        # element-wise median over beats
A_target = median_b ( max(x_b) - min(x_b) )        # median peak-to-peak of the beat set
T_A      = T_raw * A_target / ( ptp(T_raw) + eps ) # eps = 1e-12
```

- **No DC re-centering** beyond whatever `T_raw` already is.
- **No smoothing. No fitting. No per-validation-window scaling. No subject-specific scaling.**
- **Exactly one global frozen Template A**, used by every arm.

### Provenance, unchanged from the preregistration

- Included train subjects: `e61, l38, n31, ngh, p9p, qm9, trh, tz8, u7y, w4p` (10).
- Excluded, as author-flagged noisy-ECG: **`fex`, `p5d`**
  (`data/manifests/split_a4_wildppg_seed42.json → extra.note`).
- Windows: `select_subset(salt="s1-template-v1", subject, n_total, n_take=256, exclude=())`, 256 per
  included subject.
- Beats: `beat_window(sig, r, fs, before_s=0.25, after_s=0.40)` — the identical window
  `morphology_corr` uses.
- Validation subjects contribute **no shape information whatsoever** to the template. Validation GT is
  used only for R-peak **positions**, at stamping time.

### Recorded to `artifacts/s1_metric_validity/template_A_metadata.json`

SHA256 of `template_A.npy`; length; `ptp(T_raw)`; `A_target`; final `ptp(T_A)`; number of training beats
used; the included subject list; the excluded noisy-ECG subjects.

### The discrete QRS crop — rounding rule, frozen

`beat_window(before_s=0.25, after_s=0.40)` at `fs = 128` returns `sig[r-32 : r+51]`, i.e. **length 83 with
the R-peak at index 32**.

The T-B interval is converted with `int(round(seconds * fs))`, both endpoints **inclusive**:

```
n_before = int(round(0.080 * 128)) = int(round(10.24)) = 10
n_after  = int(round(0.120 * 128)) = int(round(15.36)) = 15
support  = n_before + n_after + 1  = 26 samples  (= 203.125 ms)
T_B      = T_A[32 - 10 : 32 + 15 + 1] = T_A[22:48]      # length 26, R at index 10
```

**Overlap-freeness is a testable arithmetic claim, not an assumption.** Two stamps at `r1 < r2` occupy
`[r1-10, r1+15]` and `[r2-10, r2+15]`; they are disjoint iff `r1+15 < r2-10`, i.e. iff the RR interval in
samples satisfies

```
d >= 26 samples  (203.125 ms)
```

A unit test asserts, on the actual frozen population, that **every** consecutive GT RR interval in samples
is `>= 26`. **If that assertion is false for any window — because of an endpoint convention, a rounding
edge, or a genuinely shorter RR — execution STOPS before G1 and the discrepancy is reported.** The
interval `[−80, +120] ms` is **not** to be silently adjusted to make the assertion pass.

---

## 1C. Claim discipline on a G1 FAIL

The preregistration's §0 phrase "every event-F1 number in this project is uninterpretable" overstates what
a failed gate would license. It is withdrawn as a FAIL conclusion.

**On FAIL, the licensed statement is:**

> *The frozen detector-based event F1 has not passed a direct sanity check as a trusted instrument for
> exact beat placement under this synthetic stamping construction.*

or an equally conservative wording.

A FAIL:

- **does not** delete or revise any historical numeric value;
- **does not** declare all historical F1 values mathematically invalid;
- **does** find that the measurement basis for **placement-mechanism claims** is unverified, and blocks
  further mechanism interpretation until a separately preregistered instrument study addresses it.

The distinction matters because the gate probes one specific construction — a synthetic stamped signal —
and a construction-specific failure is not the same as a proof that a statistic computed on model outputs
is meaningless.

---

## 2. What this amendment does not touch

Threshold 0.95 · tolerance 50 ms (75/150 ms remain diagnostics) · population and subset salts · detector ·
matcher · one-to-one matching · equal-subject macro aggregation · subjects `an0`/`k2s` · the `kjd`/`ssx`
firewall · the pre-preregistration exposure disclosure (§3 of the preregistration) · the joint-fidelity
gate multipliers committed in §S1.3 · every stop rule · the prohibition on designing a replacement metric
inside S1 · the deferral of the `kjd`/`ssx` GT reliability audit to a separate protocol.

**NO TRAINING. No checkpoint is created or modified. No new architecture. S1.2–S1.6 remain not started.**
