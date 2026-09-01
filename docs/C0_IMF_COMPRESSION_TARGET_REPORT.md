# C0 — iMeanFlow Compression-Target Confirmation — REPORT

## **DECISION: COMPRESSION TARGET = NFE 4**

Gate A **PASS**, Gate B **FAIL**, under the rule frozen at `5df1a33` before any number here existed.

Protocol: `docs/C0_IMF_COMPRESSION_TARGET_PREREGISTRATION.md` (`5df1a33`).
Implementation-only commit, pushed before execution: `61967a1`.

**No training. No test access. No external baseline. No new method. No compression method selected.**

---

## 1. Provenance

| item | value |
|---|---|
| checkpoint | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt`, md5 `31c042d291052fbb6dc15263ad316be2` |
| population | X4-0 stage-B: `an0` 1,024 + `k2s` 1,024 = **2,048 windows**, **19,834 GT beats** |
| Gaussian source | seed 0, bank sha256 `868085798050102eb815e1700c8e9edb` — **one bank, reused for every NFE** |
| realised steps | NFE 1 → 1, 2 → 2, 4 → 4, 8 → 8 (asserted, not assumed) |
| oracle metrics in the decision | **none** |

Seed-0 values. **Not comparable** to the recorded 4-seed-pooled X4-0 table.

## 2. Table 1 — primary oracle-free structure (GT-fixed coordinates)

Every beat anchored at its GT R-peak; every valid GT beat included; no prediction detector gates inclusion;
no shift search anywhere. Ideal for the three ratios is **1**.

| NFE | raw corr | raw QRS-E | \|QRS-E−1\| | raw slope | \|slope−1\| | raw p2p | \|p2p−1\| | raw QRS RMSE | raw RMSE | HF (GT 0.1944) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0959 | 0.4569 | 0.6250 | 0.8879 | **0.2522** | 0.8215 | 0.2340 | 0.5691 | 0.4531 | 0.2230 |
| 2 | 0.1036 | 0.4790 | 0.6148 | 0.8326 | 0.2722 | 0.7871 | 0.2616 | 0.5580 | 0.4395 | 0.2241 |
| **4** | **0.1040** | 0.5004 | 0.6056 | 0.8089 | 0.2733 | 0.8165 | 0.2425 | **0.5462** | **0.4233** | 0.2305 |
| 8 | 0.1021 | **0.5164** | **0.5973** | 0.8304 | 0.2605 | **0.8313** | **0.2316** | 0.5485 | 0.4253 | 0.2384 |

HF carries no direction claim and is excluded from the counted set.

## 3. Table 2 — event / secondary (never decisive alone)

| NFE | F1 | chance | **excess** | precision | recall | beats ratio | missing | spurious | coverage | matched morph | matched RR MAE | zero-contrib |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4144 | 0.1213 | +0.2931 | 0.4188 | 0.4137 | **0.9810** | 0.5863 | 0.5673 | 0.4362 | 0.6679 | 25.37 | 0.0908 |
| 2 | 0.4273 | 0.1208 | +0.3066 | 0.4311 | 0.4270 | 0.9709 | 0.5730 | 0.5439 | 0.4502 | 0.7272 | 23.82 | 0.0996 |
| **4** | **0.4367** | 0.1192 | +0.3176 | 0.4435 | **0.4338** | 0.9492 | 0.5662 | 0.5154 | **0.4568** | 0.7743 | 23.08 | 0.1118 |
| 8 | 0.4341 | 0.1161 | **+0.3180** | **0.4459** | 0.4278 | 0.9161 | 0.5722 | **0.4882** | 0.4502 | **0.7795** | **22.65** | 0.1240 |

Matched morphology, coverage, zero-contribution fraction and matched RR MAE are **conditional** on
successful ≤50 ms matching and did not choose the target.

## 4. Table 3 — paired improvement

Subject-stratified **paired** bootstrap, 2,000 resamples, `default_rng(20260901)`, equal subject weight.
**Positive always means the later NFE is better.**

| comparison | metric | oriented Δ | 95 % CI | verdict |
|---|---|---:|---|---|
| 1→2 | raw_corr | +0.00768 | [+0.00584, +0.00947] | improves |
| 1→2 | \|QRS-E−1\| | +0.01026 | [+0.00201, +0.01854] | improves |
| 1→2 | \|slope−1\| | −0.02005 | [−0.02599, −0.01359] | **worsens** |
| 1→2 | \|p2p−1\| | −0.02764 | [−0.03290, −0.02251] | **worsens** |
| 1→2 | raw QRS RMSE | +0.01110 | [+0.00987, +0.01233] | improves |
| 1→2 | raw RMSE | +0.01360 | [+0.01239, +0.01479] | improves |
| 1→2 | *F1 excess* | +0.01349 | [+0.00916, +0.01801] | improves |
| 1→2 | *beats-ratio dev* | +0.00075 | [−0.00367, +0.00512] | unresolved |
| **2→4** | raw_corr | +0.00042 | [−0.00101, +0.00186] | unresolved |
| **2→4** | \|QRS-E−1\| | +0.00921 | [+0.00329, +0.01492] | **improves** |
| **2→4** | \|slope−1\| | −0.00106 | [−0.00559, +0.00351] | unresolved |
| **2→4** | \|p2p−1\| | +0.01914 | [+0.01507, +0.02317] | **improves** |
| **2→4** | raw QRS RMSE | +0.01175 | [+0.01046, +0.01304] | **improves** |
| **2→4** | raw RMSE | +0.01620 | [+0.01470, +0.01768] | **improves** |
| **2→4** | *F1 excess* | +0.01101 | [+0.00715, +0.01473] | improves |
| **2→4** | *beats-ratio dev* | −0.00088 | [−0.00481, +0.00301] | unresolved |
| **4→8** | raw_corr | −0.00197 | [−0.00274, −0.00122] | **worsens** |
| **4→8** | \|QRS-E−1\| | +0.00822 | [+0.00560, +0.01066] | improves |
| **4→8** | \|slope−1\| | +0.01277 | [+0.01008, +0.01547] | improves |
| **4→8** | \|p2p−1\| | +0.01090 | [+0.00873, +0.01309] | improves |
| **4→8** | raw QRS RMSE | −0.00226 | [−0.00265, −0.00191] | **worsens** |
| **4→8** | raw RMSE | −0.00197 | [−0.00231, −0.00163] | **worsens** |
| **4→8** | *F1 excess* | +0.00045 | [−0.00209, +0.00286] | unresolved |
| **4→8** | *beats-ratio dev* | −0.01894 | [−0.02252, −0.01558] | **worsens** |

## 5. Gates

### GATE A — is NFE 2 actually inferior to NFE 4? **PASS**

1. NFE 4 clearly improves **4** of 6 primary metrics over NFE 2 (`|QRS-E−1|`, `|p2p−1|`, `raw QRS RMSE`,
   `raw RMSE`) — requirement was ≥ 2. ✓
2. NFE 4 clearly worsens **0**. ✓ (`raw_corr` and `|slope−1|` are unresolved, not worsened.)
3. No event-F1 collapse — F1 excess *improves*, +0.01101 [+0.00715, +0.01473]. ✓

**A real 2 → 4 structural gap exists on this population.**

### GATE B — does NFE 8 justify itself over NFE 4? **FAIL**, on two independent grounds

1. NFE 8 clearly improves 3 primary metrics — the three ratio deviations. ✓ (≥ 2)
2. NFE 8 clearly **worsens 3**: `raw_corr`, `raw QRS RMSE`, `raw RMSE`. ✗ (requirement was zero)
3. `beats_ratio` deviation clearly degrades, −0.01894 [−0.02252, −0.01558]; beats ratio falls
   0.9492 → 0.9161, i.e. NFE 8 under-produces beats further. ✗

**NFE 8 did not demonstrate sufficient incremental benefit over NFE 4 under the preregistered
target-selection rule.** This is *not* a claim that NFE 4 and NFE 8 are equivalent — no equivalence test
is preregistered, and the metrics visibly move in both directions between them.

## 6. Scientific interpretation

Under the frozen iMeanFlow checkpoint and this development protocol, 2 → 4 is a genuine one-directional
structural gain: four of six GT-fixed metrics clearly improve, none worsens, and event F1 excess rises
alongside. 4 → 8 is not a gain but a **trade**: the three amplitude/energy/slope calibration ratios move
closer to 1 while pointwise fidelity (`raw_corr`, both RMSEs) and beat production all clearly degrade, so
the frozen rule cannot elevate NFE 8. The 1 → 2 step is itself mixed — it improves correlation, both RMSEs
and QRS energy while clearly worsening slope and p2p calibration — which is consistent with 4 → 8 and
suggests these ratio calibrations and pointwise fidelity are not co-monotone in NFE anywhere on this grid.
`raw_corr` never exceeds 0.1040 at any NFE, so none of these arms reproduces beat shape at exact GT
coordinates; the comparison is between weak arms, and C0 makes no claim that NFE 4 is good in absolute
terms. Beat production drifts monotonically away from 1 with NFE (0.9810 → 0.9161), matching the
S1 observation that additional integration progressively deletes beats. Matched morphology and matched RR
MAE both look monotonically better with NFE, but they are conditional on a matched set whose coverage also
changes (0.4362 → 0.4568 → 0.4502), which is exactly why the preregistration barred them from the
decision. Every statement here is confined to two development subjects at source seed 0.

## 7. Confirmations

- **No test access.** `kjd`/`ssx` never loaded; `assert_no_test_subjects` called before first data access;
  neither string appears in the analysis script; `provenance.json` records `test_subjects_loaded: []`.
- **No training.** No weight updated; no `.backward()`, optimiser, or `torch.save` in the C0 code path;
  `requires_grad_(False)` and `torch.no_grad()` on the inference path. No checkpoint created or modified.
- **No external baseline. No new method. No compression method implemented, selected or trained.**
- **Frozen checkpoint**, md5 recorded.
- **Same source bank across all NFEs** — drawn once before the NFE loop, sha256 recorded; a unit test pins
  that `torch.randn` is called exactly once in the script and before the loop.
- **Oracle metrics excluded** from the primary table and from the decision, verified by a source-text test.
- **Target chosen by the preregistered rule only**, on directional paired evidence with no numeric effect
  threshold invented after results.

## 8. Next research question (recorded, not started)

> *"Can the frozen NFE-4 physiological fidelity be preserved at 1–2 NFEs?"*

C0 does **not** choose between residual flow, condition-informed source, distillation, shortcut models,
event heads or anchored transport. That is the next preregistration, after review.

## Artifacts

`artifacts/c0_imf_compression_target/` (gitignored): `primary_metrics.csv`, `secondary_metrics.csv`,
`paired_bootstrap.csv`, `decision.json`, `provenance.json`, and figures
`c0_nfe_oracle_free_metrics.png`, `c0_nfe_event_vs_structure.png`, `c0_paired_improvement.png`.
Code: `scripts/analyze_c0_compression_target.py`, `src/ppg2ecg/evaluation/paired_stats.py`,
`tests/test_c0_compression_target.py`.
