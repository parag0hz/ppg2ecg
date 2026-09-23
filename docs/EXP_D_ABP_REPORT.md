# EXP-D Part B — PPG → ABP → SBP / DBP / MAP: allocation **PARTIAL** (frozen rule) · evidentiary **NO SUPPORT** (amendment gate)

Preregistration `eddbe60` (`docs/EXP_D_FUNCTIONAL_GENERALIZATION_PREREGISTRATION.md`) and prospective amendment
**`65dcc97`** (`docs/EXP_D_ABP_PREREGISTRATION_AMENDMENT.md`, usability gate), both pushed before any ABP test sample
existed. **Training changed: NO.** Commands: `scripts/expd/expd_run.py gen MIMIC-BP | analyze abp` (unchanged frozen
analysis), `scripts/expd/expd_abp_gates.py` (amendment gate, committed with the amendment), post-hoc
`scripts/expd/expd_abp_posthoc.py`, figure `scripts/expd/expd_abp_figure.py`. Results:
`artifacts/exp_d_functional_generalization/abp/` (bootstrap.json, usability_gates.json, fixed_budget.csv, mechanism.csv,
latency.json, per_patient.csv, prereg_manifest.json, posthoc.json, figure.png). Raw: `outputs/exp_d_functional_generalization/`
(gitignored). 95 % CIs: subject-clustered bootstrap, 5,000 replicates, seed 20260924. *Post-hoc* items were not preregistered.

## 1. Data, checkpoint, scale
U1 upstream PENGUIN as shipped, `PENGUIN_MIMIC-BP_u1` (sha256 `02dd37c1…`, saved epoch 17, seed 42; strict load clean;
`external/PENGUIN` @ `6cd70cd` unmodified). Upstream split 1,144 / 190 / 190 subjects; test **190 subjects, 39,900 windows
of 4 s, 19,950 blocks of 8 s**. **ABP_PHYSICAL_SCALE = VALID**: target and output in mmHg end to end (no normalisation,
no inverse transform; audit). 0 non-finite samples in 3,351,600 generated windows (84 samples × 39,900). Pipeline check: the shipped Heun-50
draw 0 gives SBP 14.931 / DBP 9.226 mmHg vs the U1 upstream-printed 14.986 / 9.212.

## 2. Functionals (upstream, frozen)
SBP = block maximum, DBP = block minimum (upstream `SBPError` / `DBPError`), MAP = block time-average (*ours*); consensus =
median over the K sample values; MAE in mmHg, subject macro.

## 3. Fixed-budget grid (MAE, mmHg)
| functional | B32: (32,1) | (16,2) | (8,4) | (4,8) | (2,16) | (1,32) | B16: (16,1) | (8,2) | (4,4) | (2,8) | (1,16) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SBP** | **14.432** | 14.484 | 14.531 | 14.649 | 14.830 | 14.764 | **14.431** | 14.485 | 14.527 | 14.651 | 14.888 |
| **DBP** | 9.915 | 10.024 | 9.938 | 9.618 | 9.159 | **9.084** | 9.916 | 10.021 | 9.933 | 9.613 | **9.185** |
| **MAP** | 10.625 | 10.506 | 10.361 | 10.181 | **10.072** | 10.129 | 10.626 | 10.508 | 10.365 | 10.188 | **10.079** |

| reference lines | SBP | DBP | MAP |
|---|---|---|---|
| single sample K = 1 at S = 1 / 2 / 4 / 8 / 16 / 32 | 14.43 / 14.49 / 14.54 / 14.68 / 14.89 / 14.76 | 9.91 / 10.01 / 9.93 / 9.63 / 9.19 / 9.08 | 10.64 / 10.52 / 10.38 / 10.19 / 10.08 / 10.13 |
| shipped Heun-50, K = 1 | 14.931 [14.009, 15.933] | 9.226 [8.581, 9.904] | 10.268 [9.512, 11.079] |
| training-median constant (b_F) | 14.600 [13.673, 15.597] (114.7) | 9.420 [8.753, 10.105] (54.9) | 10.363 [9.586, 11.165] (75.8) |

**Each grid row is almost identical to the K = 1 row at the same S**: consensus changes nothing (§8). The B32 / B16 grids
therefore trace the single-sample depth curve: SBP is best shallow, DBP and MAP are best deep.

## 4. Preregistered contrasts and allocation verdicts (original rule of `eddbe60`)
| functional | B32 (32,1) − (1,32) | B32 (8,4) − (1,32) | B16 (16,1) − (1,16) | B16 (4,4) − (1,16) | allocation verdict |
|---|---|---|---|---|---|
| SBP | **−0.332 [−0.494, −0.174]** (68 % of subjects) | **−0.233 [−0.355, −0.114]** (65 %) | −0.457 [−0.671, −0.245] | −0.361 [−0.500, −0.227] | **success** (width-heavy better) |
| DBP | **+0.830 [+0.350, +1.284]** (40 %) | **+0.853 [+0.369, +1.316]** (39 %) | +0.731 [+0.454, +0.996] | +0.749 [+0.464, +1.019] | **depth-clear** |
| MAP | +0.496 [−0.018, +0.968] | +0.232 [−0.164, +0.593] | +0.547 [+0.139, +0.921] | +0.286 [+0.000, +0.551] | **no clear difference** (depth favoured at B16) |

**Original frozen ABP category: PARTIAL** (1 functional success, 1 depth-clear, 1 neither; not FAILED because only one
functional is depth-clear). Test-best cells (exploratory): SBP (32,1) / (16,1); DBP (1,32) / (1,16); MAP (2,16) / (1,16).
Against the shipped Heun-50: SBP (32,1) −0.499 [−0.693, −0.312], (8,4) −0.400; DBP (32,1) +0.688 [+0.050, +1.297], (8,4)
+0.711; MAP n.s.

## 5. Usability gate (amendment `65dcc97`; gate cell (8,4))
| functional | (8,4) MAE | constant MAE | **Gate A**: Δ_const [95 % CI] | **Gate B**: Spearman(consensus, reference) [95 % CI] | Pearson (secondary) | usability |
|---|---|---|---|---|---|---|
| SBP | 14.531 | 14.600 | −0.069 [−0.379, +0.232] → **FAIL** | 0.150 [0.072, 0.226] → **PASS** | 0.172 [0.064, 0.261] | **PARTIALLY INFORMATIVE** |
| DBP | 9.938 | 9.420 | +0.518 [−0.038, +1.060] → **FAIL** | 0.276 [0.186, 0.366] → **PASS** | 0.253 [0.167, 0.339] | **PARTIALLY INFORMATIVE** |
| MAP | 10.361 | 10.363 | −0.003 [−0.428, +0.408] → **FAIL** | 0.244 [0.154, 0.331] → **PASS** | 0.242 [0.159, 0.325] | **PARTIALLY INFORMATIVE** |

The generated SBP / DBP / MAP are weakly associated with the reference (ρ_S 0.15–0.28) but are **not better than a
training-median constant**.

## 6. Two verdicts per functional, and the cross-functional verdict
| functional | allocation verdict (frozen rule) | usability | **evidentiary verdict** |
|---|---|---|---|
| SBP | success | PARTIALLY INFORMATIVE | **NOT EVIDENCE FOR GENERALISATION** |
| DBP | depth-clear | PARTIALLY INFORMATIVE | **NOT EVIDENCE (not counterevidence)** |
| MAP | no clear difference | PARTIALLY INFORMATIVE | **UNINTERPRETABLE / NON-INFORMATIVE TASK** |

**Cross-functional evidentiary verdict: NO SUPPORT** — no functional is USABLE, so none can count for support or against it.
Respiration, retrospective label (interpretation only; the frozen STRONG is unchanged), recomputed with the same gate code:
Gate A +0.781 [+0.406, +1.188] FAIL, Gate B −0.112 [−0.298, +0.030] FAIL → **UNINFORMATIVE**; RR is not evidence either.

## 7. Waveform vs functional trade-off (draw 0)
| S | 1 | 2 | 4 | 8 | 16 | 32 | Heun-50 |
|---|---|---|---|---|---|---|---|
| per-window Pearson r with reference | 0.879 | 0.880 | 0.880 | 0.879 | 0.874 | 0.863 | 0.855 |
| RMSE (mmHg) | 14.58 | 14.45 | 14.32 | **14.21** | 14.26 | 14.59 | 14.91 |
| FD (upstream raw-signal, mmHg²) | 62,991 | 61,959 | 60,353 | 58,812 | 58,162 | **53,805** | 57,132 |

The ABP **shape** follows the reference (r ≈ 0.87 at every depth; unlike respiration and PPGFlowECG) — but the **level** does
not (§9). Depth improves the waveform distribution (FD −15 % from S = 1 to 32) while SBP gets worse with depth and DBP / MAP
better: the depth that is best for the waveform is not the depth that is best for every functional, and the best depth
differs between functionals of the same waveform. No width-condition collapse (r, RMSE, FD all within the frozen limits).
Latency: B32 cells ≈ 614–624 ms sequential at batch 1; batched, (32,1) 20 ms vs (1,32) 616 ms; Heun-50 959 ms. Equal NFE is
equal compute for PENGUIN.

## 8. Mechanism (K = 16, S = 1, 2, 4, 8; 19,950 blocks)
| functional | ρ̄_S (S = 1 → 8) | K_eff | G_S (mmHg) | G / I | Spearman(ρ̄, G) over S | subject-level Spearman(ρ̄_p, G_p), 190 subjects |
|---|---|---|---|---|---|---|
| SBP | 0.9964 → 0.9878 | 1.00–1.01 | 0.015 → 0.053 | 0.1–0.4 % | **−1.0** [−1.0, −0.8] | −0.29 … −0.26 (all CIs < 0) |
| DBP | 0.9857 → 0.9924 | 1.01 | 0.006 / 0.001 / 0.001 / 0.009 | ≈ 0.1 % | +0.4 [−0.4, +0.8] | **+0.22 … +0.30 (all CIs > 0)** |
| MAP | 0.9980 → 0.9966 | 1.00 | 0.021–0.022 | 0.2 % | +0.4 [−0.8, +1.0] | −0.32 … −0.30 (all CIs < 0) |

Waveform pairwise RMS of the 16 samples: 2.34 / 2.23 / 2.36 / 2.91 mmHg. Consensus checks (16,S) − (1,S): SBP and DBP ≈ 0
(all CIs contain 0), MAP −0.016 mmHg (CIs below 0, negligible).
**The samples' functional errors are almost entirely shared (ρ̄ ≈ 0.99, K_eff ≈ 1)**: K = 16 samples are worth one. This is
the redundancy end of the mechanism — high ρ̄, no consensus gain — and it is why width contributes nothing here. The
across-S question (lower redundancy → larger gain) goes the predicted way only for SBP; DBP and MAP do not, and DBP's
subject-level association has the opposite sign. With gains of hundredths of a mmHg these associations carry no practical
weight.

## 9. *Post-hoc*: what drives the allocation differences
| | S = 1 | 2 | 4 | 8 | 16 | 32 | Heun-50 |
|---|---|---|---|---|---|---|---|
| SBP bias (mean Ŷ − Y\*, mmHg) | −3.2 | −3.7 | −4.1 | −4.8 | −5.6 | −4.0 | −4.5 |
| DBP bias | +3.8 | +4.1 | +4.0 | +3.1 | +1.1 | −1.0 | −2.7 |
| MAP bias | +3.0 | +2.5 | +1.7 | +0.5 | −1.2 | −2.3 | −3.6 |
| generated pulse pressure (reference 60.5) | 53.5 | 52.7 | 52.5 | 52.6 | 53.8 | 57.5 | 58.7 |

The estimates barely vary between blocks (SD across blocks: SBP 3.0–6.3 vs reference 18.2 mmHg; DBP 3.5–4.1 vs 11.9;
MAP 2.9–3.4 vs 13.2): the model predicts close to the population level. Depth mainly **moves that level** (MAP bias
+3.0 → −2.3 mmHg) and widens the pulse pressure; the SBP / DBP / MAP errors change with S because the bias changes, and
the "width wins" for SBP is the shallow single sample's bias (K = 1 at S = 1: 14.432 = the (32,1) cell). Against the
training constant (paired): only the deep single sample's DBP beats it ((1,32) −0.335 [−0.637, −0.042]); no SBP or MAP
estimate does.

## 10. Failures
1. **No ABP functional is usable** (Gate A fails for all three): PENGUIN's generated ABP carries weak level information
   (ρ_S 0.15–0.28) and is not better than a training-median constant. The shipped sampler is at the constant's level too.
2. **Width does nothing** for ABP (ρ̄ ≈ 0.99): every allocation difference is a single-sample depth (bias) effect.
3. The preregistered mechanism direction holds for SBP only; DBP / MAP do not follow it, and DBP's subject-level
   association is reversed.
4. The frozen allocation rule would have counted SBP as a success of the width-heavy principle; the amendment's gate is
   what prevents that reading. MAP is a functional of ours (not upstream).
5. MIMIC-BP's checkpoint was trained with U1's epoch cap (20; best epoch 18) — the shipped-performance level reproduces the
   upstream print, but the model may be under-trained.

## 11. Claims
**Strengthened (descriptive):** (a) the need for a usability gate — on a second output, "width beats depth" (SBP) appears
without any consensus gain and without the functional beating a constant; (b) functional-error redundancy tracks the
consensus gain across outputs: large gains at ρ̄ 0.56–0.86 on our ECG models (e.g. iMF, K = 16 at S = 1, ≈ 34 % of the
single-sample error), 6 % at 0.89 (RDDM), 7–10 % at 0.92–0.94 (PPGFlowECG), 0.1–0.4 % at ≈ 0.99 (ABP) — the redundancy end is now observed directly; (c) depth
changes the waveform distribution and the functionals differently, and the best depth differs between functionals of one
waveform.
**Weakened:** "refine enough, then sample wide" as a principle beyond ECG → HR — neither respiration nor ABP provides
evidence for it; for ABP the width part of the principle is inert because the samples are redundant, and the depth part
is functional-specific (SBP shallow, DBP / MAP deep). The ECG → HR results are unaffected (they pass the usability gate).
**Still unsupported:** that the principle generalises across physiological functionals; that one S suits every functional;
anything about blood-pressure estimation quality from this checkpoint beyond "not better than a constant".

## 12. Verdicts
| | result |
|---|---|
| original frozen ABP category (`eddbe60`) | **PARTIAL** (SBP success · DBP depth-clear · MAP none) |
| usability (amendment `65dcc97`) | SBP / DBP / MAP all **PARTIALLY INFORMATIVE** (Gate A fail, Gate B pass) |
| evidentiary, per functional | SBP not evidence · DBP not evidence (not counterevidence) · MAP uninterpretable |
| **cross-functional evidentiary** | **NO SUPPORT** |
| respiration (retrospective label) | frozen STRONG unchanged; usability UNINFORMATIVE → not evidence |

HARD STOP B. The cross-functional synthesis (`docs/EXP_D_CROSS_FUNCTIONAL_SYNTHESIS.md`) and `docs/EXP_D_FINAL_REPORT.md`
were preregistered only for the case in which respiration and ABP are both valid tests; whether to write them with the
present outcome (neither output provides a usable functional) is left to the user.
