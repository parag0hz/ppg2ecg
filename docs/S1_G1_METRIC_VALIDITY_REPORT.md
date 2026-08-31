# S1 G1 — Metric-Validity Hard Gate — REPORT

**Verdict: G1 = PASS.** T-B macro F1 = **0.9993** at 50 ms, threshold 0.95.

**G1 only.** S1.2–S1.6 were not started. No training, no checkpoint created or read, no test-subject access.

---

## 1. Repository provenance

| | |
|---|---|
| branch | `main` |
| start SHA (before this work) | `b749339652985df0ccb4522a906774a858111720` |
| SHA at G1 execution | `2e0c20b92dc8eb290f45bdaacc715361ce4fc7d5` (recorded in `provenance.json`) |
| working tree at start | clean |
| `external/PENGUIN` | `6cd70cdefb91f10efeb8dce34019b5067cb25344` (unchanged) |
| `external/iMeanFlow` | `bf60cd7cb653f6628e59d48034b333c5eba445e2` (unchanged) |
| GPU | not required and not used |
| historical `.pt` / `.npz` / `.csv` artifacts | not modified |

## 2. Original preregistration

`docs/S1_METRIC_VALIDITY_PREREGISTRATION.md`, commit **`b749339`**. **Not modified.** Its text is
byte-identical to the commit that froze it.

## 3. Amendment 1

`docs/S1_METRIC_VALIDITY_AMENDMENT_1.md`, commit **`dc75079`**.

## 4. Metric-semantics wording correction

Commit **`2e0c20b`** — a mathematical wording correction to `docs/METRIC_SEMANTICS.md` and the 15 pointer
notes it placed. No numeric content changed.

## 5. Ordering guarantee

**Both `dc75079` and `2e0c20b` were committed and pushed before any G1 real-data quantity was computed.**
At the time each was pushed, no Template A existed and no signal had been stamped or scored. The G1 run
began from a clean tree at `2e0c20b`, which `artifacts/s1_metric_validity/provenance.json` records as the
executing HEAD. The amendment was therefore written blind to every number in this report.

## 6. Template A — provenance, scaling, hash

Frozen train-only canonical beat, built exactly as Amendment 1 §1B specifies.

| | |
|---|---|
| subjects included (10) | `e61, l38, n31, ngh, p9p, qm9, trh, tz8, u7y, w4p` |
| subjects excluded (author-flagged noisy ECG) | **`fex`, `p5d`** |
| validation subjects contributing shape | **none** — `an0`/`k2s` supplied R-peak *positions* only |
| subset rule | `select_subset(salt="s1-template-v1", subject, n_total, n_take=256, exclude=())` |
| training beats used | **24,631** |
| length | 83 samples (R at index 32) |
| `ptp(T_raw)` | 1.607417 |
| `A_target` (median beat p2p) | 1.662163 |
| final `ptp(T_A)` | **1.662163** |
| SHA256 (array content) | `c21f3970e3152b4365dee11991cb750bfa635e46c8b4ab7a60fed7f6e9991e34` |
| SHA256 (`template_A.npy` file) | `1a67569f8a02bc0027c0a60c4575d297dc2bc40eb0c3e285b9acf82daafd51eb` |
| SHA256 (T-B crop) | `6f059015812308a9d71b4c72e08a6eeed68ee2ca29107bc801097d8a1748e595` |

Scaling applied verbatim: `T_A = T_raw · A_target / (ptp(T_raw) + 1e-12)`, no DC re-centering, no
smoothing, no fitting, no per-window or per-subject scaling. Unit-tested for formula parity and for being
a single global scalar multiple of `T_raw`.

## 7. Discrete QRS crop definition

`int(round(seconds × fs))`, both endpoints inclusive, at `fs = 128`:

```
n_before = int(round(0.080 × 128)) = int(round(10.24)) = 10
n_after  = int(round(0.120 × 128)) = int(round(15.36)) = 15
support  = 10 + 15 + 1 = 26 samples  (203.125 ms)
T_B      = T_A[32-10 : 32+15+1] = T_A[22:48]        # length 26, R at index 10
```

**Overlap-freeness, verified rather than assumed.** Two stamps `d` samples apart are disjoint iff
`d ≥ 26`. On the actual population: **minimum observed RR = 41 samples (320.31 ms)** over 17,786 RR
intervals; **0 of 2,048 windows violate the requirement** (`overlap_check.json`). The run additionally
asserts `stamp_supports_overlap == False` for all 2,048 T-B windows after construction: **T-B 0/2048**
overlapping, against **T-A 224/2048** (expected and declared). The frozen `[−80, +120] ms` interval was
not adjusted.

## 8–9. Results — all four arms at all three tolerances

Population for every row: the frozen X4-0 stage-B subset, **2,048 windows (an0 1,024 + k2s 1,024),
19,834 GT beats**. Macro = equal-subject mean of per-window F1.

| arm | tol (ms) | F1 macro | precision | recall | beats ratio | an0 F1 | k2s F1 | matched | missing | spurious | n_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T-A | 50 | 0.8570 | 0.7817 | 0.9621 | 1.2555 | 0.8592 | 0.8548 | 18,977 | 857 | 5,510 | 24,487 |
| T-A | 75 | 0.8570 | 0.7817 | 0.9621 | 1.2555 | 0.8592 | 0.8548 | 18,977 | 857 | 5,510 | 24,487 |
| T-A | 150 | 0.8570 | 0.7817 | 0.9621 | 1.2555 | 0.8592 | 0.8548 | 18,977 | 857 | 5,510 | 24,487 |
| **T-B (GATE)** | **50** | **0.9993** | **1.0000** | 0.9987 | 0.9987 | 0.9990 | 0.9996 | 19,806 | 28 | **0** | 19,806 |
| T-B | 75 | 0.9993 | 1.0000 | 0.9987 | 0.9987 | 0.9990 | 0.9996 | 19,806 | 28 | 0 | 19,806 |
| T-B | 150 | 0.9993 | 1.0000 | 0.9987 | 0.9987 | 0.9990 | 0.9996 | 19,806 | 28 | 0 | 19,806 |
| T-C | 50 | 0.9998 | 1.0000 | 0.9997 | 0.9997 | 0.9997 | 0.9999 | 19,827 | 7 | 0 | 19,827 |
| T-C | 75 | 0.9998 | 1.0000 | 0.9997 | 0.9997 | 0.9997 | 0.9999 | 19,827 | 7 | 0 | 19,827 |
| T-C | 150 | 0.9998 | 1.0000 | 0.9997 | 0.9997 | 0.9997 | 0.9999 | 19,827 | 7 | 0 | 19,827 |
| T-D | 50 | 0.8530 | 0.7771 | 0.9606 | 1.2647 | 0.8540 | 0.8519 | 18,942 | 892 | 5,692 | 24,634 |
| T-D | 75 | 0.8530 | 0.7771 | 0.9606 | 1.2647 | 0.8540 | 0.8519 | 18,942 | 892 | 5,692 | 24,634 |
| T-D | 150 | 0.8530 | 0.7771 | 0.9606 | 1.2647 | 0.8540 | 0.8519 | 18,942 | 892 | 5,692 | 24,634 |

Per-window distribution at 50 ms:

| arm | mean | SD | min | deciles (10–90 %) | F1 = 1 | F1 < 0.5 |
|---|---:|---:|---:|---|---:|---:|
| T-A | 0.8570 | 0.1250 | 0.083 | 0.783 0.800 0.818 0.818 0.857 0.889 0.947 0.952 1.000 | 327 / 2048 (16.0 %) | 49 (2.4 %) |
| **T-B** | 0.9993 | 0.0061 | 0.933 | 1.000 × 9 | **2020 / 2048 (98.6 %)** | **0** |
| T-C | 0.9998 | 0.0029 | 0.941 | 1.000 × 9 | 2041 / 2048 (99.7 %) | 0 |
| T-D | 0.8530 | 0.1284 | 0.083 | 0.750 0.783 0.818 0.818 0.857 0.900 0.947 0.952 1.000 | 327 / 2048 (16.0 %) | 49 (2.4 %) |

Every arm is **bit-identical across 50 / 75 / 150 ms**. For T-B and T-C this is forced: precision is
exactly 1.0000 with zero spurious peaks at 50 ms, so no widening can add a match. For T-A and T-D it means
none of their 5,510 / 5,692 spurious peaks lies within 150 ms of any of their 857 / 892 unmatched GT beats.

## 10. Per-subject

`an0` and `k2s` agree closely in every arm (T-B 0.9990 vs 0.9996; T-A 0.8592 vs 0.8548), so the gate
verdict is not carried by one subject. Macro and pooled coincide here because the two subsets are equal
in size (1,024 each).

## 11. HARD GATE verdict

| | |
|---|---|
| gate arm | **T-B** (Amendment 1 §1A) |
| tolerance | 50 ms |
| threshold | ≥ 0.95 |
| observed | **0.9993** |
| **verdict** | **PASS** |

T-A is a control and was **not** used for the gate decision.

## 12. T-A overlap-control interpretation

**T-A scores 0.8570 — below the 0.95 threshold.** Under the original preregistration, which named T-A the
primary gate, G1 would have been recorded as a **FAIL**. Amendment 1, written before any of these numbers
existed, is what prevents that: a sub-threshold T-A cannot distinguish a metric failure from an artifact of
its own stamping construction, which is precisely why the gate moved.

The T-A deficit is one of **excess** detections, not missed ones: recall 0.9621 but precision 0.7817, with
**24,487 predicted peaks against 19,834 GT beats (beats ratio 1.2555)** — 5,510 spurious against only 857
missing. The full 83-sample canonical beat carries a large post-QRS trough (visible in
`figures/g1_stamping_examples.png`) that the detector fires on as an additional beat.

Two reported quantities bound how much of this is superposition specifically: only **224 of 2,048 windows**
have overlapping T-A supports, while **1,721 of 2,048** score below F1 = 1. Overlap therefore cannot
account for most of the T-A deficit; the stamped beat's own morphology is implicated. That inference is
arithmetic on the numbers above, and no further analysis was run to separate the two mechanisms — doing so
is outside G1's preregistered scope.

## 13. What G1 establishes

**The frozen detector and 50 ms one-to-one matcher can reward an overlap-free, QRS-like event placed at the
exact GT R-peak position with the preregistered required fidelity** — 0.9993 macro F1, precision exactly
1.0000, 98.6 % of windows at F1 = 1.0, no window below 0.5. The data-independent analytic control (T-C,
0.9998) reaches the same place, so the result is a property of the instrument rather than of the particular
train-derived template.

## 14. What G1 does NOT establish

- **Not** that historical model F1 values are accurate.
- **Not** that detector artifact is absent from model outputs. G1 scored synthetic stamped signals whose
  QRS complexes are clean, isolated and identical; a generated waveform is none of those things, and T-A
  shows that a *realistic* full-beat morphology already costs 0.14 F1 on a signal whose beats are in
  exactly the right place.
- **Not** that the observed event failure is biological rather than instrumental.
- **Not** that the joint event-fidelity / waveform-fidelity Pareto gap is proven.
- **Not** that displacement versus absence is resolved.
- Nothing about `kjd`/`ssx`, about signal quality, or about any model.

The population is two development-validation subjects that have been visually inspected before (four
pre-viewed windows excluded by construction); no population-level inference follows.

## 15. Confirmations

- **No test access.** `kjd`/`ssx` were never loaded. `assert_no_test_subjects` is called before first data
  access; `provenance.json` records `test_subjects_loaded: []`. Neither string appears in the G1 script or
  in `src/ppg2ecg/evaluation/stamping.py`.
- **No training.** No weight was updated; no `.backward()`, optimizer, or `state_dict` call exists in the
  new code. No model of any kind was loaded — G1 needs none.
- **No checkpoint created, read or modified.** No GPU used.
- **S1.2, S1.3, S1.4, S1.5 and S1.6 were not started.** `provenance.json` records
  `items_run: ["G1"]`, `items_not_run: ["S1.2","S1.3","S1.4","S1.5","S1.6"]`.
- Full test suite green before execution (146 passed, 1 skipped), including 22 new static/synthetic G1
  tests covering the firewall, template provenance and scaling parity, the discrete crop, stamping
  behaviour, the 26-sample overlap boundary, detector/matcher parity, macro aggregation, artifact
  protection and rerun determinism.

## Artifacts

`artifacts/s1_metric_validity/` (gitignored): `template_A.npy`, `template_A_metadata.json`,
`overlap_check.json`, `g1_stamping_metrics.csv`, `g1_per_window.csv`, `g1_gate.json`, `provenance.json`,
`figures/g1_stamping_examples.png`, `figures/g1_f1_distributions.png`. No prediction dumps.
Code: `scripts/analyze_s1_metric_validity.py`, `src/ppg2ecg/evaluation/stamping.py`,
`tests/test_s1_g1_stamping.py`.
