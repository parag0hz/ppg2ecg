# Q1 — Conditional Support Degradation Audit — REPORT

**Does PPG→ECG generation remain plausible when the PPG becomes uninformative?**

| | |
|---|---|
| Preregistration | `docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md`, frozen and pushed as `2cde60a` **before** any Q1 computation |
| Implementation commit | `f1e0640` (code + 28 tests; full suite 341 passed) |
| Run commit | `f1e0640` with **3 dirty files** (the module-registration fix of §18 item 9) |
| Status | **exploratory / problem-discovery**. Not independent confirmation of anything. Two development-validation subjects. Synthetic corruption shows model response to controlled input degradation, **not** naturally occurring clinical artefact causality |
| Terminology | *conditional-support / plausibility decoupling*. The word **hallucination is not used** |
| Test subjects | `kjd`, `ssx` never loaded (`test_subjects_loaded: []`) |
| **FINAL Q1 VERDICT** | **NO CONSISTENT CONDITION-DEGRADATION PATTERN** (verdict D, by the frozen decision rule — see §11, which also states precisely *why* the residual bucket was reached) |

---

## 1. Repository integrity

| Check | Result |
|---|---|
| HEAD == origin/main at design time | `55fe1e1b14b87b3e660f9dc0828a87e64ce032af` ✓ |
| Working tree | tracked files clean; 3 untracked documentation files from the same session were committed as documentation-only `602bd37` (parent `55fe1e1`) and pushed **before** the preregistration, restoring a clean tree (disclosed in prereg §0.1) |
| PENGUIN submodule | `6cd70cdefb91f10efeb8dce34019b5067cb25344` ✓ |
| iMeanFlow submodule | `bf60cd7cb653f6628e59d48034b333c5eba445e2` ✓ |
| A4 checkpoint md5 | `31c042d291052fbb6dc15263ad316be2` unchanged ✓ |
| C2 | still deferred; no `outputs/*c2*`; Q1 started no training ✓ |
| Optimizer / trainable parameter in Q1 | none (static source audit + runtime `requires_grad` assertion) ✓ |

Commit order followed: integrity → preregistration (`2cde60a`, pushed) → implementation + tests (`f1e0640`, pushed) → preflight → sanity → support → fidelity → uncertainty → controls → verdict → natural-quality audit → secondary → atlas → report (this commit).

## 2. Frozen components

| Role | Path | Identity (asserted at runtime) |
|---|---|---|
| PRIMARY generator, arm B | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` | file `557c7054…`, state_dict `47d7ccb9…`, round 46 (0-based 45), 4,568,707 params, `cond_mode h_only`, `h_scale 1.0` |
| Support probe (not a generator arm) | `outputs/r1_global_tcn_seed42/checkpoint_best.pt` | file `bfe76ea6…`, state_dict `0986a7af…`, 328,897 params |
| SECONDARY generator | `outputs/r3_gtf_true_seed42/module_step2200.pt` on the same frozen generator | file `ebf55708…`; labelled *rhythm-augmented; its scaffold comes from a probe trained on GT-R labels* |

Every module `requires_grad_(False)` and `eval()`. Sampling: uniform `ER.UNIFORM[4]` (NFE 4), source bank `torch.randn(2048,1,1024, seed 0)` with sha256 `868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f` asserted and **shared by every condition**.

**Regression check against R2/R3**: the CLEAN arm-B row reproduces the frozen R2/R3 baseline exactly — `f1 0.4367`, `chance_f1 0.1192`, `f1_excess 0.31756`, `missing 0.5662`, `spurious 0.5154`, `beats_ratio_dev 0.1067` (R2 `event_metrics.csv`, arm B, NFE 4).

## 3. Populations

| Cohort | Size | Definition |
|---|---|---|
| PRIMARY | **2,048** windows (an0 1,024 + k2s 1,024), 19,834 GT beats | the frozen C0/C1/R2/R3 development cohort, asserted element-for-element against `artifacts/x4_0_event_reliability/nfe_subset.json` |
| UNCERTAINTY | **512** (2 subjects × 4 sites × 64) | lowest SHA256 ranks of `q1-uncertainty-v1\|subject\|site\|window_index`; metadata only |
| MARGINAL REFERENCE | **12,288** (12 train subjects × 4 sites × 256) | `q1-marginal-reference-v1`; GT ECG of TRAIN subjects only (builder refuses any other subject) |
| NATURAL QUALITY | **8,192** | the frozen R1 validation cohort (an0/k2s, 1,024 per subject × site) |
| VISUAL ATLAS | **64** | the V1 `viz` rows belonging to an0/k2s — no new example selected |

## 4. Controlled corruption and its sanity

Applied **after** the frozen preprocessing, to the model-input PPG, with **no renormalisation**. Achieved SNR is exact (max |error| 3.1e-07 dB over all windows and levels). Monotonicity gate: **PASS** in all three families.

| condition | PPG corr (med) | nRMSE | RMS ratio | spec L1 | pulses/win | pulse-interval MAE (ms) | sha256 (block) |
|---|---|---|---|---|---|---|---|
| CLEAN | 1.000 | 0.000 | 1.000 | 0.000 | 8.65 | 0.0 | `ecf127a1d3c3` |
| LP_3.0Hz | 0.995 | 0.090 | 0.985 | 0.062 | 8.57 | 22.6 | `33df3ffa6d29` |
| LP_2.0Hz | 0.970 | 0.226 | 0.947 | 0.188 | 8.09 | 52.1 | `45f71161dc69` |
| LP_1.25Hz | 0.895 | 0.433 | 0.801 | 0.541 | 7.07 | 67.7 | `fcebffd89be4` |
| SNR_20dB | 0.995 | 0.100 | 1.005 | 0.085 | 8.87 | 18.2 | `128ffbd428e7` |
| SNR_10dB | 0.950 | 0.316 | 1.048 | 0.275 | 9.37 | 45.3 | `cbde4d30feed` |
| SNR_5dB | 0.858 | 0.562 | 1.146 | 0.501 | 9.81 | 61.4 | `c71bea5c219d` |
| SNR_0dB | 0.682 | 1.000 | 1.413 | 0.798 | 10.53 | 72.9 | `84dc92c9c377` |
| DROP_0.5s | 0.988 | 0.145 | 0.991 | 0.132 | 8.53 | 2.0 | `ff502e7cea13` |
| DROP_1.0s | 0.930 | 0.353 | 0.984 | 0.348 | 8.04 | 1.7 | `a91f89d9751d` |
| DROP_2.0s | 0.822 | 0.563 | 0.956 | 0.608 | 7.04 | 2.6 | `e5016f1935ed` |
| SHUFFLED (control) | 0.001 | 1.406 | 1.001 | 1.143 | 8.65 | 89.8 | `0591d601f2fb` |
| NULL (control, OOD) | n/a (zero variance) | 1.000 | 0.000 | 1.000 | 0.00 | n/a | `2daeb1f36095` |

Dropout barely disturbs the *pulse interval* sequence (MAE ≈ 2 ms) while removing up to a quarter of the window — it removes local evidence without shifting the rhythm; low-pass and noise disturb both.

## 5. Does ECG-relevant support actually decrease? (measured, not assumed)

Frozen R1 Global-TCN on the corrupted PPG, frozen threshold 0.35 / refractory 32, scored against GT ECG R peaks. Equal-subject macro.

| condition | F1@50 | F1@150 | F1@200 | RR MAE (ms) | RR median AE (ms) | missing | spurious | beats dev |
|---|---|---|---|---|---|---|---|---|
| CLEAN | 0.6157 | **0.8598** | 0.8973 | **38.15** | 33.82 | 0.1062 | 0.2157 | 0.1183 |
| LP_3.0Hz | 0.5059 | 0.8106 | 0.8619 | 41.75 | 36.77 | 0.1553 | 0.2687 | 0.1249 |
| LP_2.0Hz | 0.3440 | 0.8010 | 0.8652 | 44.91 | 40.64 | 0.1717 | 0.2649 | 0.1162 |
| **LP_1.25Hz** | 0.2874 | **0.7403** | 0.8088 | **52.23** | 47.92 | 0.2555 | 0.2574 | 0.0995 |
| SNR_20dB | 0.5488 | 0.8288 | 0.8755 | 43.71 | 39.32 | 0.1325 | 0.2631 | 0.1418 |
| SNR_10dB | 0.3156 | 0.7356 | 0.8306 | 53.54 | 48.87 | 0.2163 | 0.3832 | 0.1839 |
| SNR_5dB | 0.2293 | 0.6754 | 0.7959 | 60.25 | 55.07 | 0.2671 | 0.4713 | 0.2218 |
| **SNR_0dB** | 0.2072 | **0.6246** | 0.7412 | **70.82** | 65.35 | 0.2878 | 0.6035 | 0.3237 |
| DROP_0.5s | 0.5619 | 0.8436 | 0.8817 | 44.61 | 39.65 | 0.1034 | 0.2699 | 0.1725 |
| DROP_1.0s | 0.5339 | 0.8262 | 0.8653 | 47.23 | 41.04 | 0.1080 | 0.3130 | 0.2093 |
| **DROP_2.0s** | 0.3878 | **0.7163** | 0.7624 | **66.11** | 57.01 | 0.1183 | 0.6499 | 0.5335 |
| SHUFFLED | 0.1355 | 0.3946 | 0.5066 | 100.21 | 98.51 | 0.5782 | 0.7047 | 0.2178 |
| NULL | 0.1547 | 0.4500 | 0.5316 | 92.05 | 89.78 | 0.4248 | 1.0011 | 0.5765 |

R1 F1@50, F1@150 and RR MAE degrade monotonically with severity in **all three families** (the `spurious` column of BANDLIMIT and the `missing` column of DROPOUT are not monotone), and at every severe level the decrease is confirmed by the paired bootstrap (§9). The SHUFFLED row is the empirical "no window-specific information" floor for this probe (F1@150 ≈ 0.39).

## 6. Generator conditional fidelity (frozen arm B, NFE 4, source seed 0)

| condition | F1 | chance | **F1 excess** | missing | spurious | beats dev | QRS RMSE | deriv RMSE | curvature | raw corr |
|---|---|---|---|---|---|---|---|---|---|---|
| CLEAN | 0.4367 | 0.1192 | **+0.3176** | 0.5662 | 0.5154 | 0.1067 | 0.5462 | 0.3220 | 0.2147 | +0.1040 |
| LP_3.0Hz | 0.3692 | 0.1182 | +0.2510 | 0.6350 | 0.5722 | 0.1227 | 0.5577 | 0.3212 | 0.2138 | +0.0632 |
| LP_2.0Hz | 0.2432 | 0.1157 | +0.1275 | 0.7620 | 0.6753 | 0.1465 | 0.5594 | 0.3091 | 0.2043 | +0.0181 |
| **LP_1.25Hz** | 0.2113 | 0.1134 | **+0.0979** | 0.7958 | 0.6774 | 0.1648 | 0.5545 | 0.3035 | 0.1993 | +0.0139 |
| SNR_20dB | 0.3843 | 0.1187 | +0.2656 | 0.6183 | 0.5667 | 0.1112 | 0.5485 | 0.3186 | 0.2118 | +0.0811 |
| SNR_10dB | 0.2277 | 0.1183 | +0.1094 | 0.7751 | 0.7180 | 0.1276 | 0.5502 | 0.3070 | 0.2027 | +0.0185 |
| SNR_5dB | 0.1812 | 0.1185 | +0.0628 | 0.8215 | 0.7630 | 0.1341 | 0.5470 | 0.3022 | 0.1991 | +0.0066 |
| **SNR_0dB** | 0.1666 | 0.1175 | **+0.0491** | 0.8346 | 0.7890 | 0.1543 | 0.5423 | 0.2982 | 0.1959 | +0.0058 |
| DROP_0.5s | 0.3886 | 0.1181 | +0.2705 | 0.6139 | 0.5614 | 0.1138 | 0.5475 | 0.3186 | 0.2121 | +0.0847 |
| DROP_1.0s | 0.3296 | 0.1184 | +0.2112 | 0.6728 | 0.6188 | 0.1304 | 0.5458 | 0.3120 | 0.2072 | +0.0644 |
| **DROP_2.0s** | 0.1949 | 0.1180 | **+0.0769** | 0.8068 | 0.7531 | 0.1650 | 0.5424 | 0.2993 | 0.1970 | +0.0220 |
| SHUFFLED | 0.1256 | 0.1173 | +0.0082 | 0.8768 | 0.8261 | 0.1604 | 0.5373 | 0.2941 | 0.1920 | −0.0009 |
| NULL | 0.1115 | 0.1037 | +0.0077 | 0.8991 | 0.6779 | 0.2626 | 0.5305 | 0.2835 | 0.1801 | +0.0017 |

Two facts dominate this table:

1. **Event correspondence collapses.** Of the clean conditional advantage over the shuffled floor (0.3176 − 0.0082 = 0.3094), only **29 % survives** LP_1.25Hz, **13 %** SNR_0dB and **22 %** DROP_2.0s.
2. **The three QRS-shape metrics that the verdict rule uses do not follow.** `raw_qrs_rmse` moves by at most +0.0132 / −0.0038 across the naturalistic conditions (−0.0157 at NULL), and `qrs_deriv_rmse` / `qrs_curvature_err` become *slightly lower* (nominally "better") as the condition degrades — including at the SHUFFLED and NULL extremes, which reach the nominally best values of the whole table. Those three metrics are the ones the preregistered F-B conjunct is built on.

The other preregistered structure metrics **do** track the condition, and must be reported alongside:

| condition | raw_rmse (S1) | raw_corr (S2) | raw_qrs_rmse (S3) | deriv RMSE (S4) | curvature (S5) | qrs_e_dev (S6) | p2p_dev (S7) | hf_err (S8) |
|---|---|---|---|---|---|---|---|---|
| CLEAN | 0.4233 | +0.1040 | 0.5462 | 0.3220 | 0.2147 | 0.6056 | 0.2425 | 0.0854 |
| LP_3.0Hz | 0.4338 | +0.0632 | 0.5577 | 0.3212 | 0.2138 | 0.6372 | 0.2551 | 0.0879 |
| LP_2.0Hz | 0.4455 | +0.0181 | 0.5594 | 0.3091 | 0.2043 | 0.7384 | 0.2931 | 0.0927 |
| LP_1.25Hz | 0.4447 | +0.0139 | 0.5545 | 0.3035 | 0.1993 | 0.8018 | 0.3283 | 0.0915 |
| SNR_20dB | 0.4282 | +0.0811 | 0.5485 | 0.3186 | 0.2118 | 0.6397 | 0.2519 | 0.0859 |
| SNR_10dB | 0.4395 | +0.0185 | 0.5502 | 0.3070 | 0.2027 | 0.7582 | 0.2655 | 0.0876 |
| SNR_5dB | 0.4415 | +0.0066 | 0.5470 | 0.3022 | 0.1991 | 0.8104 | 0.2782 | 0.0913 |
| SNR_0dB | 0.4409 | +0.0058 | 0.5423 | 0.2982 | 0.1959 | 0.8550 | 0.3029 | 0.0928 |
| DROP_0.5s | 0.4270 | +0.0847 | 0.5475 | 0.3186 | 0.2121 | 0.6422 | 0.2511 | 0.0856 |
| DROP_1.0s | 0.4316 | +0.0644 | 0.5458 | 0.3120 | 0.2072 | 0.6988 | 0.2733 | 0.0830 |
| DROP_2.0s | 0.4404 | +0.0220 | 0.5424 | 0.2993 | 0.1970 | 0.8535 | 0.3227 | 0.0869 |
| SHUFFLED | 0.4384 | −0.0009 | 0.5373 | 0.2941 | 0.1920 | 0.9074 | 0.2986 | 0.0860 |
| NULL | 0.4337 | +0.0017 | 0.5305 | 0.2835 | 0.1801 | 0.9580 | 0.6068 | 0.0885 |

So the structure family splits: **QRS *shape* error (S3–S5) is not conditionally anchored** (it is nominally best where the condition is least informative), while **QRS *energy* and *amplitude* deviation (S6, S7), whole-window RMSE (S1) and whole-window correlation (S2) do degrade with the condition** — `qrs_e_dev` 0.606 → 0.802 / 0.855 / 0.854 and `raw_corr` +0.104 → +0.014 / +0.006 / +0.022 at the three severe levels, with SHUFFLED/NULL the worst rows on S6/S7.

## 7. GT-independent marginal support (this is **not** realism)

Reference intervals from TRAIN-subject GT ECG only (12,288 windows; train-real detector-valid 0.9995): `hr_bpm [37.42, 118.70]`, `qrs_width_ms [46.88, 148.44]`, `qrs_p2p [0.344, 1.928]`, `max_deriv [12.00, 113.88]`, `hf_ratio [0.0040, 0.4433]`. For calibration, the GT ECG of the primary population itself scores detector-valid 1.0000 and marginal support 0.9987.

| condition | detector-valid | HR in-support | QRS-width | p2p | max-deriv | HF | **mean support** |
|---|---|---|---|---|---|---|---|
| CLEAN | 0.9980 | 0.9824 | 0.9941 | 0.9961 | 0.9990 | 1.0000 | **0.9943** |
| LP_3.0Hz | 0.9980 | 0.9751 | 0.9922 | 0.9946 | 0.9995 | 1.0000 | 0.9923 |
| LP_2.0Hz | 0.9976 | 0.9678 | 0.9893 | 0.9937 | 0.9971 | 0.9995 | 0.9895 |
| LP_1.25Hz | 0.9971 | 0.9648 | 0.9893 | 0.9902 | 0.9971 | 0.9995 | **0.9882** |
| SNR_20dB | 0.9980 | 0.9810 | 0.9917 | 0.9966 | 0.9990 | 0.9995 | 0.9936 |
| SNR_10dB | 0.9980 | 0.9741 | 0.9922 | 0.9971 | 0.9976 | 1.0000 | 0.9922 |
| SNR_5dB | 0.9985 | 0.9766 | 0.9927 | 0.9971 | 0.9995 | 1.0000 | 0.9932 |
| SNR_0dB | 0.9980 | 0.9741 | 0.9917 | 0.9966 | 0.9995 | 0.9995 | **0.9923** |
| DROP_0.5s | 0.9980 | 0.9761 | 0.9932 | 0.9946 | 0.9980 | 1.0000 | 0.9924 |
| DROP_1.0s | 0.9980 | 0.9771 | 0.9897 | 0.9932 | 0.9980 | 1.0000 | 0.9916 |
| DROP_2.0s | 0.9980 | 0.9707 | 0.9917 | 0.9917 | 0.9976 | 1.0000 | **0.9903** |
| SHUFFLED | 0.9980 | 0.9824 | 0.9941 | 0.9961 | 0.9990 | 0.9995 | 0.9942 |
| NULL | 0.9956 | 0.9209 | 0.9692 | 0.9307 | 0.9668 | 1.0000 | 0.9575 |

Marginal support is essentially flat: the largest drop from CLEAN across all naturalistic conditions is **0.0062** (LP_1.25Hz), against a preregistered tolerance of 0.05. Even with the PPG replaced by another window (SHUFFLED) the output stays at 0.9942, and with an all-zero PPG (NULL, deliberately out of distribution) at 0.9575.

## 8. Multi-source uncertainty (512 windows, source seeds 0–7)

| condition | pointwise SD (U1) | pairwise RMSE (U2) | beat-count SD (U3) | pairwise event F1@50 (U4) | @150 (U5) | GT-beat timing SD ms (U6) |
|---|---|---|---|---|---|---|
| CLEAN | 0.2981 | 0.4391 | 1.231 | 0.3859 | 0.6560 | 70.8 |
| LP_3.0Hz | 0.3037 | 0.4454 | 1.320 | 0.3722 | 0.6279 | 72.8 |
| LP_2.0Hz | 0.3114 | 0.4537 | 1.552 | 0.2709 | 0.5489 | 82.8 |
| **LP_1.25Hz** | 0.3122 (+4.7 %) | 0.4527 | 1.510 | 0.2198 | 0.5125 | 89.1 |
| SNR_20dB | 0.3015 | 0.4433 | 1.288 | 0.3269 | 0.6223 | 77.7 |
| SNR_10dB | 0.3048 | 0.4471 | 1.340 | 0.2569 | 0.5778 | 86.6 |
| SNR_5dB | 0.3057 | 0.4475 | 1.396 | 0.2315 | 0.5435 | 91.6 |
| **SNR_0dB** | 0.3067 (+2.9 %) | 0.4472 | 1.520 | 0.1949 | 0.4841 | 99.0 |
| DROP_0.5s | 0.3011 | 0.4425 | 1.283 | 0.3412 | 0.6257 | 76.0 |
| DROP_1.0s | 0.3064 | 0.4488 | 1.437 | 0.2697 | 0.5482 | 86.6 |
| **DROP_2.0s** | 0.3127 (+4.9 %) | 0.4551 | 1.687 | 0.1516 | 0.4022 | 109.7 |
| SHUFFLED | 0.2980 | 0.4388 | 1.209 | 0.4043 | 0.6653 | 72.0 |
| NULL | 0.3035 | 0.4308 | 1.832 | 0.0972 | 0.2880 | 121.6 |

The two levels of "uncertainty" behave differently:

- **Waveform-level variability is almost invariant**: the pointwise sample SD rises by only 2.9–4.9 % at the severe levels, and under SHUFFLED it does not rise at all (0.2980 vs 0.2981) even though nothing about the target is being conditioned on.
- **Event-level variability does rise clearly**: beat-count SD 1.23 → 1.51–1.69, pairwise event agreement 0.386 → 0.15–0.22 (F1@50) and 0.656 → 0.40–0.51 (F1@150); per-GT-beat timing SD 71 → 89–110 ms.

## 9. Paired clean → severe effects (2,000 subject-stratified replicates, seed 20260903)

Orientation is stored per row: SUPPORT / FIDELITY / PLAUSIBILITY positive = **clean better than corrupted**; UNCERTAINTY positive = **corrupted more uncertain/diverse**.

| family / severe | axis | metric | clean | corrupted | effect [95 % CI] | verdict |
|---|---|---|---|---|---|---|
| **BANDLIMIT / LP_1.25Hz** | SUPPORT | R1 F1@150 | 0.8598 | 0.7403 | **+0.1195** [+0.1104, +0.1285] | improves |
| | SUPPORT | R1 RR MAE | 38.15 | 52.23 | +16.86 [+14.94, +18.81] | improves |
| | FIDELITY | F1 excess | 0.3176 | 0.0979 | **+0.2197** [+0.2073, +0.2322] | improves |
| | FIDELITY | QRS RMSE | 0.5462 | 0.5545 | +0.0083 [+0.0060, +0.0106] | improves |
| | FIDELITY | deriv RMSE | 0.3220 | 0.3035 | −0.0185 [−0.0199, −0.0170] | *worsens* |
| | FIDELITY | curvature | 0.2147 | 0.1993 | −0.0154 [−0.0164, −0.0144] | *worsens* |
| | PLAUSIBILITY | detector-valid | 0.9980 | 0.9971 | +0.0010 [−0.0015, +0.0034] | unresolved |
| | PLAUSIBILITY | marginal support | 0.9943 | 0.9882 | +0.0062 [+0.0042, +0.0083] | improves |
| | UNCERTAINTY | pointwise SD | 0.2981 | 0.3122 | +0.0141 [+0.0122, +0.0160] | increases |
| | UNCERTAINTY | beat-count SD | 1.231 | 1.510 | +0.279 [+0.224, +0.330] | increases |
| | UNCERTAINTY | pairwise event F1@50 | 0.3859 | 0.2198 | +0.1661 [+0.1477, +0.1835] | more diverse |
| **NOISE / SNR_0dB** | SUPPORT | R1 F1@150 | 0.8598 | 0.6246 | **+0.2353** [+0.2246, +0.2450] | improves |
| | SUPPORT | R1 RR MAE | 38.15 | 70.82 | +32.25 [+30.38, +34.12] | improves |
| | FIDELITY | F1 excess | 0.3176 | 0.0491 | **+0.2684** [+0.2550, +0.2815] | improves |
| | FIDELITY | QRS RMSE | 0.5462 | 0.5423 | −0.0040 [−0.0063, −0.0018] | *worsens* |
| | FIDELITY | deriv RMSE | 0.3220 | 0.2982 | −0.0238 [−0.0253, −0.0223] | *worsens* |
| | FIDELITY | curvature | 0.2147 | 0.1959 | −0.0188 [−0.0199, −0.0177] | *worsens* |
| | PLAUSIBILITY | detector-valid | 0.9980 | 0.9980 | +0.0000 [−0.0024, +0.0024] | unresolved |
| | PLAUSIBILITY | marginal support | 0.9943 | 0.9923 | +0.0021 [+0.0003, +0.0038] | improves |
| | UNCERTAINTY | pointwise SD | 0.2981 | 0.3067 | +0.0086 [+0.0067, +0.0104] | increases |
| | UNCERTAINTY | beat-count SD | 1.231 | 1.520 | +0.289 [+0.234, +0.341] | increases |
| | UNCERTAINTY | pairwise event F1@50 | 0.3859 | 0.1949 | +0.1909 [+0.1710, +0.2097] | more diverse |
| **DROPOUT / DROP_2.0s** | SUPPORT | R1 F1@150 | 0.8598 | 0.7163 | **+0.1436** [+0.1362, +0.1514] | improves |
| | SUPPORT | R1 RR MAE | 38.15 | 66.11 | +27.53 [+25.84, +29.04] | improves |
| | FIDELITY | F1 excess | 0.3176 | 0.0769 | **+0.2406** [+0.2287, +0.2518] | improves |
| | FIDELITY | QRS RMSE | 0.5462 | 0.5424 | −0.0038 [−0.0060, −0.0019] | *worsens* |
| | FIDELITY | deriv RMSE | 0.3220 | 0.2993 | −0.0227 [−0.0241, −0.0213] | *worsens* |
| | FIDELITY | curvature | 0.2147 | 0.1970 | −0.0177 [−0.0188, −0.0167] | *worsens* |
| | PLAUSIBILITY | detector-valid | 0.9980 | 0.9980 | +0.0000 [−0.0024, +0.0025] | unresolved |
| | PLAUSIBILITY | marginal support | 0.9943 | 0.9903 | +0.0040 [+0.0021, +0.0060] | improves |
| | UNCERTAINTY | pointwise SD | 0.2981 | 0.3127 | +0.0146 [+0.0131, +0.0160] | increases |
| | UNCERTAINTY | beat-count SD | 1.231 | 1.687 | +0.456 [+0.401, +0.506] | increases |
| | UNCERTAINTY | pairwise event F1@50 | 0.3859 | 0.1516 | +0.2342 [+0.2156, +0.2524] | more diverse |

("*worsens*" on the three structure rows means the **clean** arm is worse than the corrupted arm on that metric — i.e. structure error does not increase with corruption.)

## 10. Inference-only controls (diagnostic; outside the naturalistic claim)

| condition | F1 excess | QRS RMSE | deriv RMSE | detector-valid | marginal support | pointwise SD | pairwise event F1@50 |
|---|---|---|---|---|---|---|---|
| CLEAN | +0.3176 | 0.5462 | 0.3220 | 0.9980 | 0.9943 | 0.2981 | 0.3859 |
| SHUFFLED | +0.0082 | 0.5373 | 0.2941 | 0.9980 | 0.9942 | 0.2980 | 0.4043 |
| NULL (OOD) | +0.0077 | 0.5305 | 0.2835 | 0.9956 | 0.9575 | 0.3035 | 0.0972 |

With a PPG belonging to a *different* window of the same subject and site, the generator emits ECG-like output with **unchanged** marginal support (0.9942 vs 0.9943), **unchanged** waveform-level sample spread (0.2980 vs 0.2981), *better* GT-fixed structure numbers — and essentially no conditional event correspondence (+0.0082). NULL is deliberately out of distribution and is not a model of a low-quality physiological PPG.

## 11. Preregistered verdict

| family | S-A | S-B | SUPPORT-DEGRADING | F-A | F-B | FIDELITY-DEGRADING | P-A | P-B | PLAUSIBILITY-PRESERVED | U-A | U-B | UNCERTAINTY-NONRESPONSIVE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BANDLIMIT | ✓ | ✓ | **yes** | ✓ | ✓ | **yes** | ✓ (+0.0010) | ✓ (+0.0062) | **yes** | ✓ (U1 +4.7 %) | ✗ | **no** |
| NOISE | ✓ | ✓ | **yes** | ✓ | ✗ | no | ✓ (+0.0000) | ✓ (+0.0021) | **yes** | ✓ (U1 +2.9 %) | ✗ | **no** |
| DROPOUT | ✓ | ✓ | **yes** | ✓ | ✗ | no | ✓ (+0.0000) | ✓ (+0.0040) | **yes** | ✓ (U1 +4.9 %) | ✗ | **no** |

Applying §11 of the preregistration in its frozen order:

- **A** needs ≥ 2 families with all four properties → 0 families (BANDLIMIT fails only on UNCERTAINTY-NONRESPONSIVE).
- **B** needs ≥ 2 families with a clear uncertainty increase, defined as U1 ≥ +10 % **and** (U3 up or event F1 down) → 0 families (U1 rose 2.9–4.9 %).
- **C** needs ≥ 2 families whose marginal plausibility clearly degrades → 0 families (largest drop 0.0062 ≪ 0.05).
- → **D. NO CONSISTENT CONDITION-DEGRADATION PATTERN**.

**What D does and does not say here.** D is the residual bucket of the frozen decision rule; its prose in the preregistration ("fewer than two families produce a coherent support-loss pattern") does **not** describe this run. All three families are SUPPORT-DEGRADING and all three are MARGINAL-PLAUSIBILITY-PRESERVED, and event fidelity collapses in all three with CIs far from zero. The verdict is D because two preregistered conjuncts failed in a way the preregistration did not anticipate:

1. **F-B** required at least one of `raw_qrs_rmse` / `qrs_deriv_rmse` / `qrs_curvature_err` to worsen alongside the event collapse. It **is** met in BANDLIMIT (`raw_qrs_rmse` 0.5462 → 0.5545, +0.0083 [+0.0060, +0.0106]) and fails in NOISE and DROPOUT, where all three of those QRS-shape metrics stay flat or move in the nominally better direction — as they also do under SHUFFLED and NULL, i.e. they are largely not conditionally anchored (§6). BANDLIMIT is therefore the only family that is both SUPPORT-DEGRADING and CONDITIONAL-FIDELITY-DEGRADING, and verdict A needs two.
2. **U-B** required *at least one* of beat-count SD / event diversity to fail to increase. **Both** increase clearly in every family — including BANDLIMIT, which is why the one family that satisfies S, F and P still fails A — while the metric the rule pairs them with (U1, waveform-level SD) rises < 5 %, so no family is "clearly responsive" under B's U1 ≥ 10 % gate either. The observed uncertainty response is *event-level only*.

The verdict stands as D. The substantive observation — support ↓ in all three families, paired event fidelity ↓↓ in all three, marginal plausibility ≈ preserved in all three, waveform-level variability ≈ flat while event-level variability rises — is reported as an exploratory pattern and **is not relabelled as verdict A**; A was missed by one family on the F-B conjunct and by the U-B conjunct in every family.

## 12. Support–fidelity coupling (association, not causation)

Spearman ρ with subject-stratified bootstrap CIs (2,000 replicates, seed 20260903), pooled per family over that family's levels **and** per level, as preregistered (§18 item 11 records that this table was recomputed after an internal review; point estimates are unchanged):

| family | R1 F1@150 → generator F1 excess | R1 RR MAE → beats-ratio dev | R1 F1@150 → QRS RMSE |
|---|---|---|---|
| BANDLIMIT | **+0.410** [+0.393, +0.432] | +0.299 [+0.273, +0.327] | +0.169 [+0.148, +0.201] |
| NOISE | **+0.472** [+0.454, +0.490] | +0.273 [+0.256, +0.296] | +0.150 [+0.129, +0.169] |
| DROPOUT | **+0.516** [+0.496, +0.536] | +0.280 [+0.257, +0.304] | +0.081 [+0.067, +0.114] |
| all naturalistic | +0.473 [+0.462, +0.484] | +0.279 [+0.270, +0.295] | +0.139 [+0.132, +0.158] |

(pooled rows recomputed at 2,000 replicates: BANDLIMIT +0.410 [+0.390, +0.432] / +0.299 [+0.277, +0.326] / +0.169 [+0.151, +0.202]; NOISE +0.472 [+0.453, +0.489] / +0.273 [+0.256, +0.298] / +0.150 [+0.129, +0.171]; DROPOUT +0.516 [+0.496, +0.535] / +0.280 [+0.258, +0.303] / +0.081 [+0.064, +0.113].)

Per level, for R1 F1@150 → generator F1 excess:

| level | ρ [95 % CI] | | level | ρ [95 % CI] |
|---|---|---|---|---|
| CLEAN | +0.587 [+0.557, +0.616] | | SNR_20dB | +0.574 [+0.544, +0.604] |
| LP_3.0Hz | +0.476 [+0.446, +0.515] | | SNR_10dB | +0.426 [+0.383, +0.459] |
| LP_2.0Hz | +0.385 [+0.344, +0.417] | | SNR_5dB | +0.355 [+0.316, +0.392] |
| LP_1.25Hz | +0.326 [+0.288, +0.366] | | SNR_0dB | +0.294 [+0.250, +0.334] |
| DROP_0.5s | +0.554 [+0.523, +0.584] | | SHUFFLED | +0.346 [+0.300, +0.385] |
| DROP_1.0s | +0.479 [+0.444, +0.511] | | NULL | **−0.003 [−0.046, +0.041]** |
| DROP_2.0s | +0.281 [+0.235, +0.318] | | | |

Window-level probe support and generator event fidelity move together (pooled ρ ≈ 0.41–0.52); the coupling to structure error is weak (ρ ≈ 0.08–0.17). The per-level rows add two checks: the coupling *weakens* as the condition degrades (+0.587 clean → +0.294 at SNR_0dB, +0.281 at DROP_2.0s), and it vanishes exactly where there is no conditioning signal at all (NULL, ρ = −0.003 with the CI covering zero). SHUFFLED retains ρ = +0.346 because probe and generator see the *same* partner PPG, so their outputs remain correlated with each other even though neither corresponds to the paired target.

## 13. Rhythm vs morphology (§17 of the task — exploratory hypothesis)

In the BANDLIMIT family, as the cutoff falls from ∞ to 1.25 Hz:

| quantity | CLEAN | LP_1.25Hz | relative change |
|---|---|---|---|
| R1 F1@150 (coarse rhythm) | 0.8598 | 0.7403 | −13.9 % |
| R1 F1@50 (exact timing) | 0.6157 | 0.2874 | **−53.3 %** |
| R1 RR MAE | 38.15 ms | 52.23 ms | +36.9 % |
| generator F1 excess (50 ms) | 0.3176 | 0.0979 | **−69.2 %** |
| generator deriv RMSE / curvature (S4/S5) | 0.3220 / 0.2147 | 0.3035 / 0.1993 | −5.7 % / −7.2 % (i.e. no degradation) |
| generator qrs_e_dev / p2p_dev (S6/S7) | 0.6056 / 0.2425 | 0.8018 / 0.3283 | +32.4 % / +35.4 % (degradation) |

Expressed against the SHUFFLED floor, LP_1.25Hz leaves **74 %** of the probe's measurable rhythm support but only **29 %** of the generator's conditional event advantage (SNR_0dB: 49 % vs 13 %; DROP_2.0s: 69 % vs 22 %).

Recorded exploratory hypothesis: **coarse rhythm information survives the loss of local PPG morphology considerably longer than the generator's ability to use it, and far longer than the generator's fine QRS-shape correspondence — which is essentially not conditionally anchored (`qrs_deriv_rmse` / `qrs_curvature_err` do not degrade under any condition, nor under SHUFFLED/NULL), although QRS energy/amplitude deviation and whole-window correlation do degrade with the condition (§6).** This is *not* an observability theorem; it motivates a dedicated ECG-component observability map.

## 14. Natural PPG quality audit (exploratory; does not enter the verdict)

8,192 R1-validation windows, two PPG-only scores reported separately, quartiles formed **within each (subject, site)** and then averaged with equal weight. `pulse_template_consistency` is undefined for 0.34 % of windows (< 3 detected pulses).

| score | quartile | mean score | R1 F1@150 | R1 RR MAE | B F1 excess | QRS RMSE | deriv RMSE | curvature | marginal support |
|---|---|---|---|---|---|---|---|---|---|
| periodicity | Q1 | 0.221 | 0.8043 | 47.9 | **+0.1940** | 0.5452 | 0.3077 | 0.2045 | 0.9927 |
| periodicity | Q2 | 0.375 | 0.8315 | 43.3 | +0.2357 | 0.5483 | 0.3123 | 0.2078 | 0.9927 |
| periodicity | Q3 | 0.523 | 0.8812 | 34.5 | +0.3566 | 0.5551 | 0.3246 | 0.2166 | 0.9930 |
| periodicity | Q4 | 0.699 | 0.9159 | 26.0 | **+0.4950** | 0.5445 | 0.3401 | 0.2284 | 0.9968 |
| template consistency | Q1 | 0.729 | 0.8044 | 48.6 | **+0.2049** | 0.5461 | 0.3063 | 0.2040 | 0.9943 |
| template consistency | Q2 | 0.849 | 0.8465 | 40.7 | +0.2607 | 0.5498 | 0.3141 | 0.2088 | 0.9914 |
| template consistency | Q3 | 0.914 | 0.8781 | 34.7 | +0.3379 | 0.5546 | 0.3260 | 0.2179 | 0.9938 |
| template consistency | Q4 | 0.966 | 0.9061 | 27.2 | **+0.4802** | 0.5423 | 0.3384 | 0.2266 | 0.9957 |

Naturally low-quality PPG reproduces the synthetic pattern: conditional event fidelity varies by a factor of ≈ 2.6 across the periodicity quartiles (0.194 → 0.495) and ≈ 2.3 across the template-consistency quartiles (0.205 → 0.480) while marginal support stays at 0.991–0.997 and the structure metrics barely move (and again move *with* fidelity, not against it). This is observational and confounded (quality co-varies with motion, site and physiology).

## 15. Secondary — frozen R3 GTF-TRUE (stronger target-derived event supervision; secondary, does not enter the verdict)

| condition | F1 excess (GTF-TRUE) | Δ vs arm B | missing | spurious | deriv RMSE | curvature | marginal support |
|---|---|---|---|---|---|---|---|
| CLEAN | +0.3582 | +0.0406 | 0.5233 | 0.4902 | 0.3289 | 0.2162 | 0.9946 |
| LP_1.25Hz | +0.1239 | +0.0260 | 0.7663 | 0.6725 | 0.3095 | 0.2008 | 0.9925 |
| SNR_0dB | +0.0549 | +0.0058 | 0.8254 | 0.7979 | 0.3010 | 0.1945 | 0.9938 |
| DROP_2.0s | +0.1143 | +0.0374 | 0.7662 | 0.7407 | 0.3044 | 0.1979 | 0.9920 |
| SHUFFLED | +0.0077 | −0.0005 | 0.8743 | 0.8446 | 0.2946 | 0.1881 | 0.9956 |
| NULL | +0.0081 | +0.0004 | 0.8943 | 0.7090 | 0.2847 | 0.1796 | 0.9727 |

(R1 support is a property of the corrupted PPG and is therefore identical to §5 for this arm.)

Paired clean → severe effects for this arm are the same shape as arm B's (F1 excess +0.234/+0.303/+0.244, all CIs far from zero; deriv and curvature "worsen" for clean, i.e. do not degrade; marginal support drops ≤ 0.0026). **Answer to the secondary question:** no robustness advantage worth acting on. **No arm × condition interaction test was run** — `secondary_gtf_paired.csv` contains only within-arm clean-vs-corrupted contrasts — so this is a descriptive comparison: the absolute advantage over arm B shrinks from +0.0406 (CLEAN) to +0.006 (SNR_0dB), while the share of each arm's own clean advantage (above its own SHUFFLED floor) that survives is 33 % / 14 % / 30 % for GTF-TRUE against 29 % / 13 % / 22 % for arm B at LP_1.25Hz / SNR_0dB / DROP_2.0s. Its extra event accuracy is essentially a clean-PPG property, and the arm is labelled throughout as carrying stronger target-derived event supervision (its scaffold comes from a probe trained on GT-R labels).

## 16. Visual atlas

64 figures (the frozen V1 validation viz windows) in `artifacts/q1_conditional_support/visual_atlas/` with `atlas_index.csv`. Each figure shows GT ECG, clean PPG, the three severe corrupted PPGs, the clean generated ECG, the three severe generated ECGs, the SHUFFLED and NULL generated ECGs, and the 8-source mean ± 1 SD envelope for CLEAN and SNR_0dB. No prediction is shifted; annotations (R1 F1@150, B F1 excess, marginal support, sample SD) are deterministic. The figures are consistent with §6–§8; no perceptual or clinical rating of them was made or is implied, and no conclusion in this report rests on them.

## 17. Runtime

| item | value |
|---|---|
| Preflight (100 windows × 13 conditions × arm B × NFE 4) | 6.42 s generation + 2.01 s scoring, peak 1,762.9 MiB → **projected 0.640 GPU-h** (budget 4.0), `stop: false` |
| Actual primary run | **893.6 s = 14.9 min = 0.248 GPU-h**, peak **1,816.0 MiB** |
| of which | corruption 11.3 s · R1 support 15.2 s · arm-B fidelity 219.8 s · plausibility 11.8 s · uncertainty 288.7 s · natural-quality 74.6 s · secondary GTF 236.6 s |
| Atlas | 64 figures, 28 MB (not committed) |
| Environment | RTX 5090, torch 2.11.0+cu130, numpy 2.3.5, scipy 1.16.3, neurokit2 0.2.12, Python 3.13.9; evaluator ran with `cudnn.deterministic = false` (framework default), as in R2/R3 |

## 18. Deviations from the preregistration

1. **HEAD advanced before the preregistration** by documentation-only commit `602bd37` to satisfy the clean-tree requirement (declared in prereg §0.1).
2. **Noise band-confinement test.** The preregistration said a test asserts ≥ 99 % of the added noise power lies in 0.4–4.5 Hz. That figure holds for the *filter design* (∫|H|⁴ over the band = **0.9948**), which the shipped test asserts; the *per-window periodogram* of a 1024-sample realisation gives 0.938 on average (min 0.812 over the 32 realisations the shipped test checks; 0.939 mean / 0.772 min over 512 primary-cohort windows at SNR_0dB) because of rectangular-window leakage, so the shipped test additionally requires the realised mean ≥ 0.90. Nothing in the results depends on this threshold.
3. **Pulse snippet for `pulse_template_consistency`** was not fixed by the preregistration; the implementation uses the existing frozen 83-sample `beat_window` centred on each detected systolic peak, a median template and the preregistered ≥ 3-pulse rule.
4. **`periodicity_score` uses the biased (1/N) autocorrelation normalisation**, so a perfectly periodic signal scores (N − lag)/N (0.875 at lag 128). Quartiles are formed within (subject, site), where this bias is common to all windows.
5. **Uncertainty banks** are the 512 cohort rows of full-length (2,048-row) seed-*s* draws, so the seed-0 uncertainty samples are byte-identical to the primary predictions for those windows (asserted in code).
6. **Three extra artifacts** beyond the preregistered §15 list: `runtime_preflight.json` (the §13 preflight record), `secondary_gtf_metrics.csv`, `secondary_gtf_paired.csv`.
7. **Atlas layout**: all three severe conditions are shown per window (12 rows) rather than a single "severe" row.
8. **`u6`** (secondary) uses the ±250 ms window with a ≥ 4-of-8 detection filter, as preregistered; it is reported but enters no gate.
9. **The run's working tree was dirty (3 files).** After the implementation commit `f1e0640`, the three Q1 scripts needed a one-line-block fix so that `scripts/r2_evaluate.py` is executed only once per process (a second execution replaces `sys.modules["r2_evaluate"]` and breaks `ProcessPoolExecutor` pickling). The fix is a `if "…" in sys.modules:` guard and nothing else (`git diff --stat`: 21 insertions, 10 deletions); it was made **before any Q1 result existed** (the preflight could not run without it) and is committed together with this report.
10. **NULL** has undefined PPG correlation (zero variance) and undefined pulse-interval MAE; the sanity table records n/a.
11. **Coupling table recomputed after an internal review.** The first pass computed the §12 Spearman CIs with 200 bootstrap replicates and only the per-family pools, whereas preregistration §10 specifies 2,000 replicates and both pooled *and per-level* correlations. `scripts/q1_recompute_coupling.py` (pure post-processing of the frozen per-window CSVs — no model is loaded and no window is regenerated) rebuilt the table at the preregistered 2,000 replicates with all 51 rows, and `scripts/q1_evaluate.py` now contains the corrected `coupling_rows()`. Point estimates are identical; only CI resolution changed (e.g. `ALL` F1@150 → F1 excess CI [+0.460, +0.484] → [+0.462, +0.484]). No verdict gate depends on this table.
12. *(merged into item 11)* — the preregistered per-level coupling rows are now present in `support_fidelity_correlations.csv` (column `level`).
13. **Atlas uncertainty envelope.** The preregistration restricted the mean ± 1 SD envelope to uncertainty-cohort members; none of the 64 atlas windows is in the 512-window uncertainty cohort, so the atlas draws its own 64-row seed-0…7 banks and shows the envelope for every atlas window. The atlas `sample_sd_*` annotations are therefore not comparable with the §8 U1 values.

## 19. What this does NOT prove

- **No test-set evidence.** `kjd`/`ssx` were never loaded; the population is two *development* validation subjects (four windows of which were visually previewed before X4-0 and are excluded from every frozen subset).
- **Synthetic corruption ≠ clinical artefact causality.** Nothing here shows that real-world PPG quality causes clinical failure.
- **The plausibility proxy is not clinical realism.** It is five marginal features against train-real 1st/99th percentiles. A page of ECGs can pass it and still be diagnostically wrong.
- **The plausibility gate had almost no room to fail.** `marginal_support_fraction` spans only 0.9575–0.9943 over all 13 conditions (0.9882 at the worst naturalistic level; 0.9575 only for the deliberately OOD NULL) — its entire observed range is narrower than the preregistered 0.05 tolerance, so MARGINAL-PLAUSIBILITY-PRESERVED was close to a priori certain and P-A/P-B should be read as "the proxy did not detect a change", not as evidence of preserved realism.
- **No calibration claim.** The generator is not shown to be "overconfident"; Q1 measures sample dispersion, not a calibrated predictive distribution.
- **No information-theoretic claim.** Nothing here says PPG does not contain ECG information, nor that anything is unobservable.
- **Association, not causation** in §12; **observational and confounded** in §14.
- **Single frozen generator, single seed, one NFE.** The secondary arm shares the same generator weights.

## 20. Recommended next experiment (recommendation only — nothing is implemented)

The preregistered mapping for verdict D is: *do not build a method around PPG-quality uncertainty yet; consider a systematic ECG-component observability map.* This run supports that mapping and sharpens it, because the reason Q1 landed in D is that **the QRS-shape channel used by the F-B conjunct (`raw_qrs_rmse` / `qrs_deriv_rmse` / `qrs_curvature_err`) carries little conditional support to lose** — SHUFFLED and NULL reach nominally better values on all three than CLEAN — so F-B was met only in BANDLIMIT, while the event channel (and, more weakly, QRS energy/amplitude and whole-window correlation) does carry support that degrades smoothly with the condition.

The natural next stage is therefore a **conditional-observability map for ECG components** on the frozen generator and the frozen probes: for each component (R-event timing at several tolerances, RR sequence, QRS core morphology, wave amplitudes, HF content), measure (i) how much of it is recoverable from PPG by a dedicated probe, (ii) how much of it the generator's output actually tracks window-by-window, and (iii) the gap between the two, with the SHUFFLED floor as the zero-support reference on every component. That map would decide whether the next method should be a *timing-anchored* generator (events strongly conditioned, morphology explicitly marginal) — the direction the R1–R3 series and Q1 jointly point to — and would give the component-level ceiling that a "quality/observability-aware conditional generation" method would have to beat. Design requires a **new preregistration**.

---

### Artifacts

`artifacts/q1_conditional_support/`: `provenance.json`, `checkpoint_manifest.json`, `runtime_preflight.json`, `cohort_manifest.csv`, `uncertainty_cohort.csv`, `corruption_manifest.csv`, `corruption_sanity.csv`, `r1_support_metrics.csv`, `generator_fidelity_metrics.csv`, `marginal_plausibility_reference.json`, `marginal_plausibility_metrics.csv`, `uncertainty_metrics.csv`, `support_fidelity_correlations.csv`, `natural_quality_metrics.csv`, `natural_quality_quartiles.csv`, `paired_bootstrap.csv`, `secondary_gtf_metrics.csv`, `secondary_gtf_paired.csv`, `decision.json`, `visual_atlas/` (64 figures + `atlas_index.csv`). Following the R1–R3 precedent, these artifacts stay local: `artifacts/*` is gitignored and no raw data, prediction, checkpoint or large artifact is committed.
