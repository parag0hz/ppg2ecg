# O2 — Oracle Event-Canonicalized MeanFlow — REPORT

**Can event geometry be separated from morphology generation?**

| | |
|---|---|
| Preregistration | `docs/O2_ORACLE_EVENT_CANONICALIZATION_PREREGISTRATION.md` + `docs/O2_CANONICAL_WARP_AUDIT.md`, frozen and pushed as **`1911fa6`** before any O2 result |
| Status | **oracle target-leakage diagnostic**, problem-discovery only, not independent confirmatory evidence |
| **FINAL O2 VERDICT** | **CANONICALIZATION OPERATOR REJECTED** (Stage-0 round-trip gate failed) |
| Consequence | **No generator was trained.** `outputs/o2_canon_oracle_seed42/` does not exist |

---

## Oracle disclosure

**The GT ECG R schedule would have been available at O2 inference: it builds the temporal coordinate at training
and at inference. O2 IS NOT DEPLOYABLE.** Nothing in this report is a deployable method, and nothing here shows
that PPG supplies exact R timing.

## 1. Repository

| | |
|---|---|
| start SHA | `8cb8c76` (O1 report) — HEAD == origin/main, clean tree, PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`, A4 md5 `31c042d2…` unchanged, no C2 outputs |
| prereg SHA | **`1911fa6`** (warp specification + preregistration + warp implementation) |
| implementation SHA | this commit (Stage-0 script, figures, 16 tests) — the *training* integration of §16 was never written, because Stage 0 stops before it |
| result SHA | this commit |
| clean? | tracked tree clean at the preregistration gate `1911fa6`. Stage 0 ran at `1911fa6` with **1 dirty file** — its own new script `scripts/o2_stage0_roundtrip.py`, which was uncommitted at run time and is committed with this report (`stage0_result.json` records `dirty_files: 1`). No frozen component, checkpoint or artifact was modified |
| test loaded? | **no** — `kjd`/`ssx` never loaded; neither O2 script mentions them |

## 2. Baseline compute resolution (resolved before Stage 0, unused because no training happened)

`artifacts/o2_oracle_canonicalization/baseline_step_resolution.json`:

| quantity | value |
|---|---|
| B best round | 46 (1-based; `best_epoch = 45`) |
| batches per epoch | `ceil(293,271 / 64) = 4,583` = `20 × 220 + 183` |
| **exact optimizer step at B's best round** | **10,046** |
| total for 66 rounds (cross-check) | 14,409 — **matches the independently recorded C2 preregistration figure exactly** |
| source | `train_meta.json`, `training_summary.json`, `training_log.csv` of `c1_imf_baseline_replay_seed42` and the frozen `batch_rounds` semantics (`train_a2.py:38-56`); all four sha256s recorded |

## 3. Warp validity (2,048-window frozen cohort, 19,834 GT beats)

| quantity | value |
|---|---|
| windows with `K < 3` (identity warp) | **0 (0.00 %)** — beat counts run 7 … 16, median 10 |
| other fallbacks (invalid anchors) | **0 (0.00 %)** — budget was 0.5 % |
| all warps monotone with monotone inverse, finite, slopes > 0 | **yes** |
| slope range, all segments | 0.361 … 3.862 |
| slope range **inside every QRS core** | 1.0000000 … 1.0000000 (max deviation 6e-15) |
| median / p90 / max absolute time shift `\|f(t) − t\|` | 6.6 / 18.6 / 160.5 samples |
| STOP gate on validity | not triggered |

The operator does exactly what it was specified to do: it equalises the inter-beat schedule, pins the QRS core to
slope 1, and never produces a non-monotone map.

## 4. Stage-0 round-trip — the gate that rejected it

Round-trip of the **GT ECG** (`x_rt = W^{-1}(W(x))`), medians over the 2,048 windows. Normalisation of T4/T6/T7/T8
uses the **exact O1 train IQRs** (0.50532 / 0.22995 / 0.03380 / 31.25).

| metric | QRS-preserving (primary) | centre-only (diagnostic) | threshold | result |
|---|---|---|---|---|
| raw waveform RMSE | **0.01882** | 0.01882 | ≤ 0.020 | **PASS** |
| raw correlation | 0.99775 | 0.99776 | — | — |
| QRS-core RMSE | 0.03528 | 0.03729 | — | — |
| **T4 p2p nAE** | **0.26544** | 0.26535 | ≤ 0.020 | **FAIL (13×)** |
| **T6 max-derivative nAE** | **0.38075** | 0.38092 | ≤ 0.020 | **FAIL (19×)** |
| **T7 curvature-energy nAE** | **0.49422** | 0.48570 | ≤ 0.020 | **FAIL (25×)** |
| **T8 QRS-width nAE** | **0.25000** | 0.25000 | ≤ 0.020 | **FAIL (12×)** |
| detector F1@50, original vs round-trip | 1.00000 (mean 0.99926) | 1.00000 (mean 0.99895) | ≥ 0.98 | **PASS** |
| median beat-count difference | 0 | 0 | = 0 | **PASS** |

Three of the six checks pass and four fail, by one to one-and-a-half orders of magnitude. Per preregistration
§12 the verdict is therefore **CANONICALIZATION OPERATOR REJECTED**; the iMeanFlow training of §14–§16 was not
started and the thresholds were not loosened.

## 5. Why it failed — the mechanism is identified, not guessed

The failure is **not** caused by the schedule equalisation, and **not** by stretching the QRS: it is caused by
**bilinear resampling at fractional sample offsets**.

1. **The centre-only diagnostic is indistinguishable from the primary operator** (T6 0.3809 vs 0.3808, T7 0.4857
   vs 0.4942). The ±10-sample slope-1 anchors — the entire QRS-preservation mechanism of the design — buy
   essentially nothing. §13 of the task asked exactly this question; the answer is *no, the slope-1 construction
   does not protect QRS shape*.
2. **The error tracks how far the canonical QRS lands from the sample grid.** With slope 1 the core is shifted by
   `q_k − r_k`, which is generally **not an integer**, so every core sample is interpolated between two
   neighbours. Spearman correlation between the per-window median distance-of-that-offset-to-an-integer and the
   round-trip error is **+0.404 (T6)** and **+0.432 (T7)**; the grid-aligned decile (n = 377) has **exactly
   0.0000** T6 and T7 round-trip error, while the worst-offset decile (n = 717) has 0.411 and 0.540.
3. **A pure half-sample shift reproduces almost the whole error.** Applying nothing but a 0.5-sample bilinear
   shift to 256 GT windows — no warp, no schedule change — gives nAE T4 0.248, T6 0.322, T7 0.463, i.e. 94 %,
   85 % and 94 % of the observed round-trip errors.

Bilinear interpolation at offset α is the 2-tap filter `(1−α, α)`, a low-pass whose attenuation is strongest for
the highest-frequency functionals. That is exactly the observed ordering of damage: curvature (second
derivative) > max derivative (first derivative) > peak-to-peak > whole-window RMSE, which is why raw RMSE and
event F1 sail through the gate while every QRS-shape functional fails it.

Figures: `figures/fig1_roundtrip_vs_grid_offset.png` (error vs grid offset, with the 0.020 threshold),
`figures/fig2_qrs_blunting_example.png` (worst-case window, whole window and one QRS complex).

## 6. What this means for the hypothesis

The preregistered hypothesis — *representing the beat schedule as a coordinate rather than generating it inside
the waveform stream avoids the R2/R3 event-vs-morphology trade-off* — **was not tested**. Stage 0 exists
precisely so that the hypothesis is not confounded by the operator: an operator that destroys 0.38 IQR units of
T6 and 0.49 of T7 *before any model runs* could never have supported a J2/J3 non-inferiority conclusion, and any
J1 event gain measured through it would have been read against a morphology floor set by the resampler rather
than by the generator.

The result is therefore a **negative result about the operator, not about the factorization idea**, and it is a
cheap one: 4.8 s of CPU, no GPU training, no generator touched.

## 7. What this does NOT prove

- **Nothing about the factorization hypothesis itself.** Coordinate-level event anchoring may still work with an
  operator that does not resample the QRS off-grid; O2 did not test it.
- **No deployability claim of any kind** — the design used GT R at inference by construction.
- **No novelty, no SOTA, no information-theoretic conclusion**, and no claim that a PPG-predicted schedule would work.
- **No generator was trained**, so there is no O2 arm, no bootstrap, no source-stability result, no site map and
  no oracle-interface comparison. The corresponding artifact files listed in the preregistration therefore do
  not exist; that is the expected consequence of the pre-training stop.
- The evidence is from **two development-validation subjects** on the frozen 2,048-window cohort.

## 8. Recommended next step (recommendation only — nothing implemented)

The diagnosis points at one specific, testable repair, which would need a **new preregistration**:

**Make the canonical schedule integer-valued.** If `q_k` is rounded to integers (and the ±W anchors with it), the
QRS core maps sample-to-sample with slope 1 *and zero fractional offset*, so the core is copied rather than
interpolated; the interpolation then only touches the inter-beat regions, where the O1 map says almost no
extractable morphology lives. The grid-aligned decile of this run — **exactly zero T6/T7 round-trip error** — is
the direct evidence that this repair would pass the same Stage-0 gate. A second, independent option is to replace
bilinear resampling with a shape-preserving one (band-limited/sinc or monotone cubic), but that adds an
interpolation-kernel choice the current protocol deliberately avoided.

Either way the next preregistration should keep this exact Stage-0 gate as its first, cheap falsification step —
it cost 4.8 s and it caught an operator that would otherwise have consumed a full compute-matched training run
and produced an uninterpretable morphology comparison.

---

### Artifacts

`artifacts/o2_oracle_canonicalization/`: `baseline_step_resolution.json`, `warp_manifest.csv` (2,048 rows),
`warp_slope_distribution.csv`, `warp_roundtrip_metrics.csv` (2,048), `center_only_roundtrip_metrics.csv` (2,048),
`stage0_result.json` (gate), `stage0_failure_diagnosis.json`, `figures/` (2). Not produced, by design:
`training_manifest.json`, `training_log.csv`, `initialization_hash.json`, `event_metrics.csv`,
`o1_aligned_component_metrics.csv`, `structure_metrics.csv`, `paired_bootstrap.csv`,
`multisource_event_stability.csv`, `site_metrics.csv`, `oracle_interface_comparison.csv`. As for R1–R3, Q1 and
O1, `artifacts/*` and `outputs/*` are gitignored; only code, tests and documents are committed.

Tests: `tests/test_o2_oracle_canonicalization.py` (16 tests — firewall, pins, C2 untouched, frozen cohort,
detector unchanged, canonical-schedule definition, monotone map and inverse, exact boundaries, `f(r_k)=q_k`,
`f^{-1}(q_k)=r_k`, slope 1 across the QRS core, identity bit-exactness and `K<3` identity, no amplitude Jacobian
and no renormalisation, same warp for PPG and ECG, GT R used only as coordinates, the Stage-0 gate and the four
verdicts, and consistency of the shipped Stage-0 artifacts with the rejection). Full suite: **381 passed**.
