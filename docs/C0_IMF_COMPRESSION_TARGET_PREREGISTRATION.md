# C0 — iMeanFlow Compression-Target Confirmation — PREREGISTRATION

**Status:** frozen at this commit, pushed **before any C0 real-data metric is computed**.
**Type:** zero-training, frozen-checkpoint forward inference only.
**Population:** WildPPG **development only** — `an0`, `k2s`. Test subjects `kjd`/`ssx` are never loaded.

**NO TRAINING. NO TEST. NO EXTERNAL BASELINE. NO NEW METHOD.**

---

## 1. The single question

> **Does the frozen iMeanFlow checkpoint exhibit a real oracle-free structural quality gap between
> NFE 2 and NFE 4/8 that is worth compressing?**

Exactly three outcomes are permitted:

- **A — COMPRESSION TARGET = NFE 4**
- **B — COMPRESSION TARGET = NFE 8**
- **C — COMPRESSION PREMISE NOT ESTABLISHED / INCONCLUSIVE**

C0 does **not** choose between residual flow, condition-informed source, distillation, shortcut models,
event heads or anchored transport. That is a later protocol.

## 2. Why the metric set is restricted (consequence of S1)

`docs/S1_METRIC_VALIDITY_REPORT.md` (commit `a29225a`) established that the ±150 ms per-beat oracle's
entire gain over the same-coordinate statistic is reproduced by pairing a beat with an **unrelated** beat
from the same window — excess over the null of +0.0001 to +0.0004, and −0.0007 for the MSE arm, against
gains of +0.28 to +0.59.

Therefore the following are **excluded from all prospective structural evidence in C0**, and may not be
used to choose the target:

- `oracle_corr`
- `oracle_absent`
- `oracle_qrs_energy_median`
- any metric inheriting the ±150 ms shift maximisation

Also constrained, from the same report:

- **matched-beat morphology** is conditional on successful ≤50 ms matching (denominator ~40 % of GT beats)
  and is **secondary only**;
- **raw event F1 alone** is known to favour a degenerate smooth solution (the MSE regressor held the
  highest F1 while retaining 1 % of GT oracle QRS energy), so it may not choose the target by itself.

C0's primary evidence is therefore **GT-fixed-coordinate structural metrics** that use no prediction
detector and no translation.

## 3. Frozen model and source

| item | value |
|---|---|
| checkpoint | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` (md5 `31c042d291052fbb6dc15263ad316be2`) |
| sampler | `event_reliability.sample_meanflow_schedule` with `ER.UNIFORM[n]` |
| NFE grid | **1, 2, 4, 8** — frozen; no value may be added after results. NFE 50 is deliberately excluded. |
| Gaussian source | `torch.randn(N, 1, 1024, generator=torch.Generator().manual_seed(0))` |

**Pairing requirement (binding).** The **same** source tensor row is reused for a given window across all
four NFE values. A fresh source is never drawn per NFE. This makes every NFE comparison paired at the
window level. The implementation asserts that the realised step count equals the requested NFE exactly.

## 4. Frozen population

The X4-0 stage-B selection, reused verbatim; no new window selection and no visual selection.

- `event_reliability.select_subset(salt="x4-event-nfe-v2", subject, n_total, n_take=1024)`
- subjects `an0` + `k2s` → **2,048 windows**, **19,834 GT beats**
- the four pre-viewed windows are excluded by construction
- `assert_no_test_subjects` is called before first data access

Values are **seed-0** and are **not comparable** to the recorded 4-seed-pooled X4-0 table.

## 5. Primary metrics — oracle-free, GT-fixed coordinates

All are computed by the existing frozen `alignment_diagnostics.beat_level_analysis`, taking only its
`raw_*` outputs. Every beat is anchored at its **GT R-peak coordinate**; every valid GT beat is included;
**no prediction R-peak is detected for inclusion** and **no shift search is performed**.

| # | metric | key | ideal | also reported |
|---|---|---|---|---|
| 1 | GT-centred beat correlation | `raw_corr` | higher | — |
| 2 | GT-centred QRS energy ratio | `raw_qrs_energy_ratio` | **1** | `abs(ratio − 1)` |
| 3 | GT-centred slope ratio | `raw_slope_ratio` | **1** | `abs(ratio − 1)` |
| 4 | GT-centred peak-to-peak ratio | `raw_p2p_ratio` | **1** | `abs(ratio − 1)` |
| 5 | GT-centred QRS RMSE | `raw_qrs_rmse` | lower | — |
| 6 | GT-centred beat-window RMSE | `raw_rmse` | lower | — |
| — | whole-window HF energy ratio | `metrics.hf_energy_ratio` | **no direction claimed** | GT reference reported |

**Per-window aggregation, frozen to the convention already in use** (`scripts/analyze_s1_remaining.py:189,192`):
`nanmean` for `raw_corr`, `raw_rmse`, `raw_qrs_rmse`; **`nanmedian`** for the three ratios. The deviation
statistic is `abs(window_aggregate − 1)`, computed from the aggregated window value.

**No arbitrary weighted structural score is formed.** HF is reported but carries no "larger is better"
reading and is **not** one of the six metrics the decision rule counts.

## 6. Secondary metrics — reported, never decisive alone

Event: `f1` @50 ms, the **count-matched random-phase chance floor** and **F1 excess over chance** using the
S1 construction verbatim (`s1_audit.chance_random_phase`, 20 draws, `default_rng(20260901)`), precision,
recall, `beats_ratio`, `missing`, `spurious`.

Conditional (explicitly labelled): matched-beat morphology correlation, matched-beat coverage,
zero-contribution-window fraction, matched-consecutive-beat RR MAE.

## 7. Paired statistics

Every arm sees the same windows, the same PPG, the same GT and the same Gaussian source, so all
comparisons are **paired**.

- subject-stratified **paired** bootstrap: window indices are resampled **within subject**, and the same
  resampled index set is applied to **both** arms of a comparison before the difference is taken
- 2,000 resamples, `default_rng(20260901)`, equal subject weight
- comparisons: **1→2**, **2→4**, **4→8**

**Orientation — positive always means the later NFE is better:**

| metric family | oriented difference |
|---|---|
| `raw_corr` | `later − earlier` |
| ratio deviations (`abs(r−1)` for QRS-E, slope, p2p) | `earlier − later` |
| `raw_qrs_rmse`, `raw_rmse` | `earlier − later` |
| event F1 excess (secondary) | `later − earlier` |

Report the point difference and the 95 % paired bootstrap CI. **No significance terminology.**

## 8. Frozen target-selection rule

No numeric effect threshold is invented after results. Only directional paired evidence is used.

- **clearly improves** ⟺ the 95 % paired bootstrap CI of the oriented difference lies entirely **> 0**
- **clearly worsens** ⟺ the CI lies entirely **< 0**
- otherwise **unresolved** on that metric

The six counted primary metrics are: `raw_corr`, `abs(QRS-E − 1)`, `abs(slope − 1)`, `abs(p2p − 1)`,
`raw_qrs_rmse`, `raw_rmse`. HF is excluded from counting.

### GATE A — is NFE 2 actually inferior to NFE 4?

A meaningful 2→4 gap exists **iff all three hold**:

1. NFE 4 **clearly improves** at least **two** of the six primary metrics over NFE 2; **and**
2. NFE 4 **clearly worsens none** of the six; **and**
3. event F1 excess does not show a clear collapse from NFE 2 to NFE 4 (i.e. the oriented F1-excess
   difference does **not** have a CI entirely < 0).

If Gate A fails → **C — COMPRESSION PREMISE NOT ESTABLISHED / INCONCLUSIVE**, and C0 stops after the report.

### GATE B — does NFE 8 justify itself over NFE 4?

Evaluated only if Gate A passes. NFE 8 becomes the target **iff all three hold**:

1. NFE 8 clearly improves at least **two** primary metrics over NFE 4; **and**
2. NFE 8 clearly worsens **none**; **and**
3. neither event F1 excess nor `beats_ratio` closeness to 1 clearly degrades.

Pass → **B, TARGET = NFE 8**. Otherwise → **A, TARGET = NFE 4**.

**Required wording discipline.** Failing Gate B is *not* a claim of equivalence. The permitted sentence is:
*"NFE 8 did not demonstrate sufficient incremental benefit over NFE 4 under the preregistered
target-selection rule."* Writing "NFE 4 and NFE 8 are identical" is prohibited — no equivalence test is
preregistered here.

## 9. Interpretation restrictions

Prohibited: "8-step is solved", "4-step is perfect", "SOTA", "external baselines are worse",
"Pareto-optimal in the field", "one-step failure is the central problem". Prohibited as evidence:
`oracle_corr`, `oracle_qrs_energy`, `oracle_absent`, matched morphology alone, event F1 alone.

Every claim is scoped: *"Under the frozen iMeanFlow checkpoint and this development protocol..."*
n = 2 development subjects, single source seed, previously visually inspected — no population-level
inference follows.

## 10. Analysis freeze

Every metric, aggregation, tolerance, NFE value, orientation, bootstrap setting and gate above is fixed by
this commit. Any analysis added after data are seen must be labelled **POST-HOC** in the script, the output
JSON and the report, and must be **additive, never substitutive**. Every preregistered grid is reported in
full; reporting only the favourable cell is prohibited.

## 11. Deliverables and stop rules

`docs/C0_IMF_COMPRESSION_TARGET_REPORT.md`; `artifacts/c0_imf_compression_target/` (gitignored) with
`primary_metrics.csv`, `secondary_metrics.csv`, `paired_bootstrap.csv`, `decision.json`,
`provenance.json`, `figures/`; code under `scripts/` and `src/ppg2ecg/evaluation/`; tests under `tests/`.

1. C0 ends at the verdict. **No compression method is implemented, selected or trained.**
2. `external/PENGUIN` (`6cd70cd`) and `external/iMeanFlow` (`bf60cd7`) stay byte-identical.
3. No checkpoint is created or modified. Checkpoints, predictions and raw data never enter git.
