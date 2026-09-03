# O2b — Integer-Grid Canonicalization Repair — REPORT

**Pre-training operator falsification only.**

| | |
|---|---|
| Preregistration | `docs/O2B_INTEGER_GRID_CANONICALIZATION_PREREGISTRATION.md` + `docs/O2B_INTEGER_GRID_WARP_AUDIT.md`, frozen and pushed as **`f2f3617`** before any O2b result |
| Implementation | **`6435baa`** (operator, Stage-0 evaluator, 14 tests; full suite 395 passed) |
| Status | **operator repair audit only** — the object under test is the *integer-grid **oracle** canonicalization operator*: not a generator experiment, not a factorization test, not deployable. O2b was designed after O2 and is problem-discovery evidence, **not independent confirmation** |
| **FINAL O2b VERDICT** | **INTEGER-GRID CANONICALIZATION OPERATOR ACCEPTED** |
| Generator | **none trained** — no generator run directory exists: `outputs/o2_canon_oracle_seed42/` is absent and the only `outputs/o2b*` entry is `o2b_stage0.log`, the captured stdout of the CPU Stage-0 run (`*.log` is gitignored). No checkpoint, no training log, no generator prediction |
| O2 factorization hypothesis | **STILL NOT TESTED** |

---

## 1. Repository

| | |
|---|---|
| start SHA | `cca96d6` (O2 rejection report) — HEAD == origin/main, clean tree |
| prereg SHA | **`f2f3617`** |
| implementation SHA | **`6435baa`** |
| result SHA | this commit |
| clean? | clean at both commit gates; the Stage-0 run was at `6435baa` with a clean tree (recorded in `provenance.json`) |
| test loaded? | **no** — `kjd`/`ssx` never loaded; neither O2b file mentions them |
| C2 status | **still deferred**, no C2 outputs; PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`, A4 md5 `31c042d2…` unchanged |

## 2. The O2 failure being repaired

O2 rejected the real-valued operator on the same 2,048-window cohort with T4 **0.26544**, T6 **0.38075**,
T7 **0.49422**, T8 **0.25000** normalised AE against a 0.020 threshold, while raw RMSE (0.01882), event F1@50
(1.000) and beat-count difference (0) passed. O2's diagnosis: the canonical R position `q_k` was real-valued, so
the QRS core was resampled at a **fractional offset** and bilinear interpolation acted as a 2-tap low-pass —
supported there by the grid-aligned decile having near-zero T6/T7 error (1.3e-06 and 3.7e-05), by the
centre-only variant being indistinguishable, and by a pure 0.5-sample shift reproducing 85–94 % of the damage.
*Erratum for the frozen companion documents*: the O2 report and this stage's audit §1 describe that decile as
"exactly zero"; the artifact values are the near-zero figures just quoted. Those documents are frozen and are not
edited retroactively.

**O2b changes exactly one thing**: interior canonical positions are projected to integers with
`round_half_to_even` (endpoints stay at `r_1`, `r_K`). The resampler, map and inverse, `W = 10` protection,
boundaries, normalisation policy and identity bit-exactness are the O2 ones, inherited unchanged.

## 3. Integer schedule audit (prechecks — all passed)

| quantity | value |
|---|---|
| windows | 2,048 (19,834 GT beats, frozen subset asserted) |
| beat count K | 7 … 16 |
| identity fallbacks (`K < 3`) | **0** |
| other invalid windows | **0** |
| **integer spacing violations** | **0** — minimum observed spacing **62 samples** against a required ≥ 21 |
| minimum real canonical spacing | 62.27 samples (minimum original RR 41 samples) |
| **max protected-core fractional coordinate** | **0.000000** (tolerance 1e-6) |
| all warps monotone with monotone inverse | yes |
| `q_int − r` integer for every beat | yes (19,834 / 19,834) |

The spacing constraint was never close to binding: the canonical schedule is uniform, so its spacing is the mean
RR of the window, and the fastest window in the cohort still leaves 62 samples between beats.

## 4. Schedule distortion introduced by rounding

| quantity | value |
|---|---|
| per-window max \|q_int − q_real\| | median 0.444, p90 0.500, **max 0.500** (as predicted by the tie convention) |
| per-window median \|q_int − q_real\| | 0.222 |
| canonical RR spread `max(diff q_int) − min(diff q_int)` | median 1, p90 1, **max 1 sample** |
| canonical RR standard deviation | median 0.433 samples |
| relative deviation from the ideal uniform spacing | median 0.67 %, max 1.43 % |
| per-beat shift \|q − r\| | O2 median 2.37 / max 160.50 → O2b median 2.00 / max 160.00, all integer |

Integer projection therefore costs at most half a sample per beat and leaves the canonical schedule irregular by
exactly one sample — that irregularity is reported, not hidden.

## 5. Stage-0 round trip — the exact O2 metrics and the exact O2 gate

| metric | O2 (fractional) | **O2b (integer)** | threshold | result |
|---|---|---|---|---|
| raw waveform RMSE | 0.018819 | **0.001696** | ≤ 0.020 | **PASS** |
| raw correlation | 0.997752 | **0.999978** | — | — |
| QRS-core RMSE | 0.035280 | **0.00000105** | — | — |
| **T4 p2p nAE** | 0.265442 | **0.000000** | ≤ 0.020 | **PASS** |
| **T6 max-derivative nAE** | 0.380747 | **0.000000** | ≤ 0.020 | **PASS** |
| **T7 curvature nAE** | 0.494225 | **0.0000109** | ≤ 0.020 | **PASS** |
| **T8 QRS-width nAE** | 0.250000 | **0.000000** | ≤ 0.020 | **PASS** |
| detector F1@50 (original vs round trip) | 1.000000 | **1.000000** (mean 0.99989) | ≥ 0.98 | **PASS** |
| median beat-count difference | 0 | **0** | = 0 | **PASS** |

Per-window distributions confirm the medians are not hiding a tail: T4 is **exactly zero in 76.6 %** of windows
(p90 1.3e-05, max 4.4e-02), T6 exactly zero in **79.7 %** (p90 2.0e-05, max 3.9e-02), T7 median 1.1e-05
(p90 6.7e-05, max 1.4e-02), T8 exactly zero in **99.95 %** (2,047 / 2,048; the single non-zero window has T8 = 0.25, i.e. one Q-or-S
trough displaced by exactly one sample — 7.8125 ms against the 31.25 ms train-IQR scale).

### Repair ratios (descriptive)

| metric | O2b / O2 |
|---|---|
| T4 p2p | **0.0000** |
| T6 max derivative | **0.0000** |
| T7 curvature | **0.0000221** |
| T8 width | **0.0000** |
| raw RMSE | 0.0901 |
| QRS-core RMSE | 0.0000297 |

The O2 diagnosis is confirmed in full: the entire morphology failure was the fractional offset. As
preregistered (§0.1 item 5), this is problem-discovery evidence about the operator and **not independent
confirmation** of anything.

### Did the integer projection introduce a new cost?

Spearman association between the per-window rounding perturbation `|q_int − q_real|` (median 0.222 samples) and
the O2b round-trip error: **−0.002 (T6)** and **+0.006 (T7)** — no association. Rounding the schedule does not
itself damage morphology. (Association only; no causal language.)

## 6. T8 support audit (diagnostic, `W` unchanged)

The frozen width routine `rpeaks.qrs_width_ms(sig, r, fs, q_win_s=0.08, s_win_s=0.12)` searches `Q` on
`[r − 10, r)` and `S` on `[r, r + 15]`, so it **can inspect 5 samples beyond the protected core `[r − 10, r + 10]`**
(pre-declared in the audit §9 before any O2b number existed). That mismatch is real and unchanged — yet T8 now
passes with a median of exactly 0 and 99.95 % of windows exactly 0. The earlier T8 failure was therefore also a
fractional-resampling artefact, not a consequence of the support mismatch. `W` was not modified, and no
post-hoc repair was applied.

## 7. Hard-copy diagnostic

**Not run**, as preregistered (§10 of the audit): it is only permitted when the primary integer operator still
shows non-negligible T4/T6/T7 error, and here T4 = T6 = 0.000000 and T7 = 1.1e-05. `hard_copy_diagnostic.csv`
therefore does not exist.

## 8. Gates

| id | check | result |
|---|---|---|
| precheck | integer spacing ≥ 21, strictly increasing, in range | **PASS** (min 62) |
| precheck | max protected-core fractional coordinate ≤ 1e-6 | **PASS** (0.000000) |
| R0-1 | raw RMSE ≤ 0.020 | **PASS** (0.001696) |
| R0-2 | T6 ≤ 0.020 | **PASS** (0.000000) |
| R0-3 | T7 ≤ 0.020 | **PASS** (0.0000109) |
| R0-4a | T4 ≤ 0.020 | **PASS** (0.000000) |
| R0-4b | T8 ≤ 0.020 | **PASS** (0.000000) |
| R0-5 | F1@50 ≥ 0.98 | **PASS** (1.000000) |
| R0-6 | median beat-count difference = 0 | **PASS** (0) |

## 9. FINAL O2b VERDICT

**INTEGER-GRID CANONICALIZATION OPERATOR ACCEPTED**

The only statement this licenses: *"Integer-grid event canonicalization passes the frozen pre-training
morphology-preservation gate."* It does **not** say that factorization works, and it does **not** say that event
canonicalization improves ECG generation — no generator was trained, and none will be in O2b.

## 10. Interpretation — three things kept separate

1. **Fractional-interpolation failure (O2).** Real-valued `q_k` forced the QRS through a 2-tap low-pass at every
   round trip; that, and nothing else, produced O2's T4/T6/T7/T8 rejection. Confirmed here by repair ratios of
   0.0000 once the offsets are integral.
2. **Integer-grid operator validity (O2b).** With integer `q_k` the protected core is sampled exactly on the
   grid (max fractional coordinate 0.000000), the operator reproduces the ECG to a median raw RMSE of 0.0017 and
   a QRS-core RMSE of 1.0e-06, and every preregistered morphology target is preserved to numerical precision —
   while the schedule really is equalised (per-beat shifts up to 160 samples, canonical RR irregular by exactly
   one sample).
3. **The factorization hypothesis is still NOT TESTED.** Whether expressing the beat schedule as a coordinate
   helps a generator — the actual O2 question — requires training, which is explicitly forbidden here and
   requires its own preregistration.

## 11. What this does NOT prove

- **No generator was trained**; no event-generation improvement of any kind was measured.
- **GT R leakage is inherent to the operator** — the schedule that builds the coordinate comes from the target
  ECG, at training and at inference. Nothing here is deployable, and nothing shows PPG can supply exact R timing.
- **Two development-validation subjects only** (`an0`, `k2s`), no test subjects, one frozen cohort.
- **No novelty, no SOTA, no information-theoretic conclusion.**
- Passing a *round-trip* gate is a necessary condition for a usable coordinate, not a sufficient one: a
  generator trained in this coordinate could still fail for reasons this audit cannot see.

## 12. Recommended next experiment (recommendation only — nothing implemented)

**O2c — Oracle Integer-Grid Event-Canonicalized MeanFlow**, with its own frozen preregistration: the O2 training
design (identical architecture, objective, optimizer and RNG to C1 arm B; the resolved compute-matched **10,046**
optimizer steps; no early stopping; no validation selection) run on top of the operator accepted here, evaluated
with the O2 primary contrast (B vs O2c at NFE 4 on the frozen 2,048-window cohort, O1-aligned T4/T6/T7/T8 errors,
ECG-window clustered paired bootstrap, the J1–J7 gate and the source-stability gates). O2c should keep this
Stage-0 gate as its first step — it cost seconds and it has now both rejected one operator and accepted its
repair.

---

### Artifacts

`artifacts/o2b_integer_grid/`: `provenance.json`, `cohort_manifest.csv`, `integer_schedule_manifest.csv`,
`integer_spacing_audit.csv`, `integer_grid_core_audit.csv`, `schedule_distortion.csv`,
`per_beat_grid_analysis.csv` (19,834 beats), `warp_roundtrip_metrics.csv` (2,048), `stage0_result.json`,
`o2_vs_o2b_comparison.csv`, `t8_support_audit.json`, `decision.json`, `figures/` (4). Absent by design:
`hard_copy_diagnostic.csv`, and any model checkpoint, training log or generator prediction. `artifacts/*` and
`outputs/*` remain gitignored; only code, tests and documents are committed.

Tests: `tests/test_o2b_integer_grid.py` (14) — firewall, submodule pins, C2 untouched, no reachable generator-training
call, real schedule identical to O2, deterministic `round_half_to_even` with explicit ties, integer endpoints
and monotonicity, the 21-sample spacing rule with no repair path, integer anchors with slope exactly 1,
protected-core coordinates integer within 1e-6, resampler inherited with no new kernel and no renormalisation,
identity bit-exactness, a synthetic-spike comparison against the fractional operator, exact reuse of the O2
cohort/metrics/IQRs/thresholds, the verdict tree, and the T8 support code fact. Full suite: **395 passed**.
