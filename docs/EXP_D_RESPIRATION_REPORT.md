# EXP-D Part A — Respiration → respiratory rate: **STRONG by the frozen rule, but not evidence for the principle** (the RR estimates carry no information about the reference; post-hoc, §9)

Preregistration `eddbe60` (`docs/EXP_D_FUNCTIONAL_GENERALIZATION_PREREGISTRATION.md`), audit
`docs/EXP_D_FUNCTIONAL_GENERALIZATION_AUDIT.md`, both pushed before any test sample was generated. **Training changed:
NO.** Commands: `scripts/expd/expd_run.py gen BIDMC | gen WESAD | analyze respiration`, post-hoc
`scripts/expd/expd_resp_posthoc.py`. Results: `artifacts/exp_d_functional_generalization/respiration/` (bootstrap.json,
fixed_budget.csv, mechanism.csv, latency.json, per_patient.csv, prereg_manifest.json, posthoc.json, figure.png). Raw:
`outputs/exp_d_functional_generalization/` (gitignored). 95 % CIs: subject-clustered bootstrap, 5,000 replicates, seed
20260924. Items marked *post-hoc* were not preregistered.

## 1. Data / checkpoint
U1 upstream PENGUIN as shipped (`external/PENGUIN` @ `6cd70cd`, unmodified), strict load clean.
**BIDMC (primary):** `PENGUIN_BIDMC_u1` (sha256 `fa8b3d96…`, saved epoch 6, seed 42), upstream split 41 / 6 / 6 subjects;
test **6 subjects, 720 windows of 4 s, 48 blocks of 60 s**. **WESAD (exploratory):** `PENGUIN_WESAD_u1` (`946dd2b1…`,
epoch 9); test **1 subject**, 1,605 windows, 107 blocks; block bootstrap, no verdict role. 0 non-finite samples anywhere.

## 2. RR extractor
Upstream `RespRateError`, unchanged: each 60-s block (15 consecutive 4-s windows of one subject) → generated block
low-passed (order-8 Butterworth, 1 Hz, `filtfilt`), reference block unfiltered → 60 × the dominant positive FFT frequency
(1 breath/min resolution). Consensus = median of the K sample RRs.

## 3. Fixed-budget grid (BIDMC, RR error in breaths/min, subject macro)
| S | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| **B = 32**, K = 32 / S | **2.458** [1.281, 3.958] | 2.771 [1.635, 4.146] | 2.885 [2.115, 3.854] | 3.083 [2.323, 3.833] | 3.896 [3.396, 4.302] | 4.646 [4.083, 5.208] |
| **B = 16**, K = 16 / S | **2.656** [1.521, 4.146] | 2.875 [1.896, 4.083] | 3.062 [2.521, 3.750] | 3.906 [3.490, 4.271] | 4.646 [4.083, 5.208] | — |
| K = 1 | 3.812 | 3.979 | 4.438 | 4.542 | 4.646 | 4.646 |

Error falls monotonically as budget moves from depth to width at both budgets; the test-best cell is the width extreme
at both budgets (exploratory label). WESAD shows the same shape (B32: 3.687 / 3.551 / 3.762 / 3.930 / 4.589 / 5.047).

## 4. B32 primary contrasts
| contrast | width cell | depth (1,32) | **difference** | subjects width better | blocks better / tied |
|---|---|---|---|---|---|
| **(32,1) − (1,32)** | 2.458 | 4.646 | **−2.188 [−3.375, −0.938]** | 5 / 6 | 69 % / 12 % |
| **(8,4) − (1,32)** | 2.885 | 4.646 | **−1.760 [−2.635, −0.999]** | 6 / 6 | 73 % / 10 % |

WESAD (exploratory): −1.360 [−2.084, −0.612] and −1.285 [−1.939, −0.607].

## 5. B16 secondary
(16,1) − (1,16) = **−1.990 [−3.167, −0.802]** (5 / 6); (4,4) − (1,16) = **−1.583 [−2.198, −1.094]** (6 / 6).
WESAD: −1.519 [−2.266, −0.738]; −1.107 [−1.855, −0.322]. Consensus checks (16,S) − (1,S): −1.16, −1.21, −1.52, −1.71 at
S = 1, 2, 4, 8, all CIs below 0.

## 6. Shipped PENGUIN reference (Heun, 25 steps = 50 NFE, K = 1; not a grid cell)
BIDMC **4.500 [3.812, 5.188]** (U1's upstream-printed value for the same checkpoint and split was 3.479 with a different
noise draw; single draws at S = 8 range 3.79–5.10 over 16 draws, so the two agree within draw-to-draw variation).
(32,1) − Heun-50 = −2.042 [−3.302, −0.781] at 32 vs 50 NFE; (8,4) − Heun-50 = −1.615 [−2.604, −0.750]; (1,32) ≈ Heun-50
(+0.146 [0.000, +0.396]). WESAD 4.972.

## 7. Waveform metrics (draw 0 per S; BIDMC)
| S | 1 | 2 | 4 | 8 | 16 | 32 | Heun-50 |
|---|---|---|---|---|---|---|---|
| per-window Pearson r with reference | −0.000 | 0.001 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |
| RMSE (normalised units) | 0.732 | 0.767 | 0.826 | 0.860 | 0.881 | 0.892 | 0.905 |
| FD (upstream, raw signal) | **149.1** | 107.2 | 64.4 | 51.4 | 45.5 | 42.8 | 40.5 |

Depth improves the waveform **distribution** (FD 149 → 43; WESAD 159 → 4.5), while RR error gets worse with depth — the
same waveform-vs-functional dissociation as in ECG. The preregistered collapse check flags **S = 1 as collapsed**
(FD 3.5 × that of S = 32); S = 4 is not (1.5 ×). RMSE *falls* with shallower S, consistent with shallow samples having
smaller amplitude around an uncorrelated reference. **No depth produces a waveform that follows the reference in time**: per-window correlation is ≈ 0 at every S
and for the shipped sampler (all CIs contain 0).
Latency (batch 1, 20 ms per NFE): every B32 cell ≈ 615–628 ms sequential; batched, (32,1) 20.6 ms vs (1,32) 615 ms;
Heun-50 963 ms. PENGUIN has no encoder / decoder, so equal NFE is equal compute here.

## 8. Mechanism (K = 16, S = 1, 2, 4, 8; BIDMC, 48 blocks)
| S | I | C | G [95 % CI] | G / I | SD | MAD | waveform RMS | ρ̄ [95 % CI] | K_eff |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.855 | 2.656 | 1.199 [0.690, 1.738] | 31 % | 3.59 | 2.47 | 0.300 | 0.300 [0.085, 0.418] | 2.9 |
| 2 | 4.238 | 2.771 | 1.467 [0.926, 2.008] | 35 % | 4.17 | 2.73 | 0.430 | 0.242 [0.093, 0.329] | 3.5 |
| 4 | 4.349 | 2.917 | 1.432 [0.922, 1.918] | 33 % | 4.40 | 2.78 | 0.636 | 0.230 [0.085, 0.309] | 3.6 |
| 8 | 4.305 | 2.833 | 1.471 [1.076, 1.832] | 34 % | 4.30 | 2.75 | 0.732 | 0.244 [0.092, 0.335] | 3.4 |

Spearman(ρ̄_S, G_S) = **−0.2** (bootstrap [−1.0, +0.8], 88 % of replicates negative) — negative, so "directionally
consistent" by the frozen definition, but essentially flat: ρ̄ and G barely change from S = 2 on. Waveform RMS +0.8,
SD +0.4, MAD +0.4. WESAD (exploratory): Spearman(ρ̄, G) −0.8 (100 % negative at block level), SD / MAD +1.0.
Functional errors here are **weakly shared** (ρ̄ 0.23–0.30, far below the ECG models' 0.56–0.94) — see §9 for why.

## 9. *Post-hoc*: the RR estimates carry no information about the reference RR
| BIDMC estimator | Spearman with reference RR over 48 blocks [subject bootstrap] | mean / SD across blocks | error − training-median constant |
|---|---|---|---|
| (1,32) single sample | +0.047 [−0.250, +0.264] | 16.5 / 4.82 | +2.54 [+1.52, +3.52] |
| shipped Heun-50 | +0.054 [−0.187, +0.264] | 16.5 / 4.62 | +2.40 [+1.38, +3.42] |
| (8,4) median | −0.112 [−0.309, +0.027] | 16.8 / 1.97 | +0.78 [+0.41, +1.19] |
| (32,1) median | −0.195 [−0.408, +0.061] | 16.6 / 0.86 | +0.35 [+0.08, +0.65] |

Reference RR: mean 18.6, SD 2.1 (range 14–22). The preregistered, report-only **training-median constant (17 breaths/min)
has error 2.104 [1.042, 3.417]** — lower than every cell, and significantly lower than each headline cell tested ((32,1), (8,4), (16,1), (4,4), (1,32), Heun-50). WESAD:
the same (no estimator correlates with the reference; the constant 3.430 is lower than every preregistered grid cell; only the K = 16 mechanism cells (16,4) / (16,8), 3.421,
are within 0.01 of it).
Reading: PENGUIN's generated respiration waveforms on these test subjects do not encode the subject's breathing rate
(neither the waveform nor its rate correlates with the reference). Each sample's RR is essentially a draw from the model's
typical-rate distribution (centre ≈ 16.5). Taking a median of K such draws shrinks the estimate towards that centre (SD
4.8 → 0.9), and because the centre lies within the reference range, the absolute error falls — **by variance reduction of
an uninformative estimator, not by better estimation**. This also explains the low ρ̄ (errors are mostly independent
sampling noise) and why the gain does not depend on depth. On this split the shipped model's upstream-printed error (3.48) is also above
the constant's (2.10 [1.04, 3.42]).

## 10. Failures
1. **The respiration functional is uninformative** (§9) — the fixed-budget question cannot be answered meaningfully on it.
2. **Preregistration defect:** the respiration verdict rule has no usability gate (the PPGFlowECG preregistration had one:
   "not usable if not below a constant baseline"). The constant was preregistered only as a reported sanity baseline.
   The frozen verdict is kept; its interpretation is corrected here rather than the rule after the fact.
3. S = 1 samples are collapsed by the preregistered FD criterion; the B32 / B16 successes rest on the (8,4) / (4,4) cells,
   which are not.
4. BIDMC has 6 test subjects (48 blocks); WESAD has one. CIs from a 6-cluster bootstrap are coarse.
5. No respiratory-event or timing metric exists upstream; none was added.

## 11. Verdict
| frozen rule item | result |
|---|---|
| B32 succeeds (a width contrast CI < 0, condition not collapsed) | yes — (8,4) −1.760 [−2.635, −0.999] ((32,1) also CI < 0 but collapsed) |
| B16 succeeds | yes — (4,4) −1.583 [−2.198, −1.094] |
| depth clearly better at B32 / no useful consensus gain | no / no |
| mechanism directionally consistent (Spearman(ρ̄, G) < 0) | yes, weakly (−0.2) |
| **category (frozen rule)** | **STRONG** |

**Interpretation.** By the letter of the frozen rule respiration is STRONG. It is **not** evidence that the allocation
principle generalises to respiratory rate: no estimator in the grid, including the shipped sampler, carries information
about the reference rate, and a training-set constant beats all of them. What the result does show is a general caution
for the whole programme: **median consensus lowers the error of an estimator even when that estimator is uninformative,
so a fixed-budget "width beats depth" result needs a usability gate (beats a trivial baseline, correlates with the
reference) before it can count as support.** For the ECG → HR results the gate is met (e.g. PPGFlowECG 6.16 vs constant
14.49; PENGUIN / iMF / CD HR track the reference), so they are unaffected.

## 12. What this means for Part B (for the decision at HARD STOP A)
The ABP verdict rule (frozen at `eddbe60`) has the same gap. Before any ABP number is computed, a dated amendment could
add — without changing any preregistered cell, contrast or category — a reported-and-gating usability check (each
functional's width and depth cells compared with the preregistered training-median constant, and the block-level
correlation of the estimates with the reference). Whether to amend is the user's decision; Part B has not been started.
