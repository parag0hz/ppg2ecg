# PPGFlowECG — external fixed-budget validation on the released checkpoint: **STRONG** (by the frozen rule; boundaries in §15–16)

Preregistrations frozen and pushed before any number: data rule `fc99f76` (`docs/PPGFLOWECG_VITALDB_DATA_PREREGISTRATION.md`),
experiment `3dee94f` (`docs/PPGFLOWECG_FIXED_BUDGET_PREREGISTRATION.md`). **Training changed: NO. Learned model changed:
NO. Sampler changed: NO** (the model's own explicit-Euler `sample_shift`, S ∈ {5, 10, 15, 20, 25} only). No re-spacing, no
noise injection, no upstream edit. Commands: `scripts/external/ppgflowecg/{pfe_audit, build_vitaldb10, smoke_test,
pfe_fixed_budget (gen | ham | proj | wave | analyze), pfe_posthoc, pfe_figures}.py`. Results:
`artifacts/ppgflowecg_external/` (figure `ppgflowecg_external.png`). Raw samples and per-sample values:
`outputs/ppgflowecg_external/` (gitignored). Patient-clustered 95 % CIs, 5,000 replicates, seed 20260924. Items marked
*post-hoc* were not preregistered.

## 1. Checkpoint provenance
Authors' officially released checkpoint; training provenance is strongly implied to be MCMED by the released training
configuration but is not explicitly documented for the checkpoint itself (`docs/PPGFLOWECG_PROVENANCE_AUDIT.md`: no
metadata in either file, no model card, MCMED hard-coded in the code paths; the prepared author query was not sent).
Consequently no number here is called a reproduction of the paper's tables, and VitalDB is treated as a zero-shot external
corpus for this checkpoint.

## 2. Repository / commit / hashes
| item | value |
|---|---|
| user-provided `ppgflowecg/` | not a git repository: paper PDF (`2509.19774v2.pdf`) + the two released checkpoints; no source |
| source used | `external/PPGFlowECG` submodule @ `56b2cd2cfa738388c60daccd788d511aa8698085`, `git status` clean before and after |
| `checkpoint-10.pt` | sha256 `50f1af67ad258ef5eea7ac0c09525f156f6b1997a3068575f08d4d8c6a05ecdb` (step 10,000) |
| `VAE-iter-40000.pth` | sha256 `c186633aa40fa2b2e32421588b8e68d7f95a30e8069f8161fbdbcedd97347d6d` |
| official functions reused | 13 functions of `calculate_metric.py` / `step1.py` / `step2.py`, extracted verbatim (AST), sha256 of each source text in `checkpoint_hashes.txt` |
| libraries | paper pins for preprocessing / Hamilton: mne 1.8.0, neurokit2 0.1.7, biosppy 2.2.3, ema-pytorch 0.7.7 (isolated dirs); torch 2.11.0+cu130 (released requirements: 2.7.0); project-standard evaluator: neurokit2 0.2.12 |
| load | official `Trainer(config) → Trainer.load(10)`; **strict** `load_state_dict`, **no missing / unexpected keys** (flow, EMA, VAE); sampler network = EMA model |
| parameters | flow 6,857,988; VAE encoder 96,491,984; ECG decoder 110,915,114 |
| audit items 1–16 | `audit.json` (shapes: input [B, 1280, 1], output [B, 1280, 1], latent [B, 4, 40]) |

Release-code findings, none requiring an upstream edit: (a) `Trainer.sample_shift` does **not forward** `sampling_steps`
(called with 5 it runs 10) — our driver passes S to the model's own `RectifiedFlow.sample_shift(num_steps=S)`, and at
S = 10 its output is **bitwise identical** to the released `Trainer.sample_shift` for the same seed; (b) `load_vae_ppg`
loads `encoder_ecg` weights into the PPG encoder — identical tensors in the checkpoint, so no effect; (c) the EMA network
stays in `train()` mode as released — all dropout p = 0 and no batch-dependent layers, so no effect;
(d) `calculate_metric.py` imports a non-existent `utils.data`, hence the verbatim function extraction; (e) the official
`ppg_bpm_array` is non-functional (§13).

## 3. Evaluation data construction (`data_build.json`)
V1 test split only; one 10-s window per V1 test window, raw span [k·2000, k·2000 + 5000) at 500 Hz; official PPGFlowECG
transforms verbatim (PPG Butterworth 0.5–8 Hz order 3; ECG high-pass 0.5 Hz order 5 + 50 Hz notch; `resample_poly` → 128 Hz;
per-window z-score; Savitzky–Golay 7/2 and 11/2). No quality selection, no polarity check.
**18,525 windows, 1,156 patients (all), 1,224 cases**; 1,018 of 19,543 anchors excluded (1,012 span past the record end,
6 non-finite); no overlapping spans. Sanity (no model): the first 4 s of the processed ECG vs V1's target ECG of the same
window, median correlation 0.924. Reference Hamilton HR Y\* defined for 99.94 % of windows (12 windows "no HR");
contrasts use 18,513 windows. Differences from the paper's pipeline that remain: VitalDB native rate 500 Hz for both
signals (the released step1 hard-codes MCMED rates ECG 125 Hz / PPG 25 Hz); the 50 Hz notch is applied to 60 Hz-mains data.

## 4. NFE accounting (`nfe_counts.json`, `latency.json`)
Forward hooks, per sampling call: vector-field calls = **S** exactly for S = 5, 10, 15, 20, 25 (one per Euler step); fixed
overhead **1 ECG-encoder + 1 PPG-encoder + 1 ECG-decoder** call per sample. Budget B = K × S counts vector-field NFE only.

| cell (K, S) | B | wall-clock, K sequential batch-1 calls (ms) | one call with the K samples batched (ms) |
|---|---|---|---|
| (1,10) / (2,5) | 10 | 30.0 / **39.6** | 30.1 / **20.4** |
| (1,15) / (3,5) | 15 | 40.1 / **60.0** | 40.3 / **21.3** |
| (1,20) / (2,10) / (4,5) | 20 | 50.4 / 61.4 / **79.2** | 51.0 / 30.9 / **26.3** |
| (1,25) / (5,5) | 25 | 63.2 / **98.9** | 60.4 / **24.7** |

RTX 5090, full official call. Batch 1: 9.7 ms fixed encoder/decoder overhead + 2.07 ms per Euler step. **Equal
vector-field NFE is not equal total compute**: each sample carries three VAE passes (96 M / 111 M-parameter networks vs a
6.9 M-parameter vector field). Run sequentially, width-heavy cells are 32–57 % slower than the same-NFE depth cell; batched
on the GPU they are faster. CPU batch 1: 231–297 ms per call (S = 5 → 25).

## 5. Smoke-test validity (`smoke_test.json`; 16 windows)
All checks pass at every S: finite outputs, shape [16, 1280], range ±4.6 (reference ±5.1), per-window SD 0.89–0.91
(reference 0.88); same seed → bitwise identical; different seeds → different samples (median pairwise waveform RMS
0.97–1.05, median correlation 0.28–0.41); **reference ECG replaced by zeros → bitwise identical samples** (the official
path uses the reference only for its latent shape — no reference leakage). PPG posterior SD / |mean| = 0.68. Generation
over the full set: **0 non-finite samples** in 1,482,000.

## 6. Single-sample depth curve (K = 1; step-count behaviour under our evaluation — not a reproduction)
| S | Hamilton HR error | − S = 10 | project HR error | − S = 10 | paper-literal MAE_hr (−1 kept) | "no HR" samples |
|---|---|---|---|---|---|---|
| 5 | **6.142** [5.747, 6.553] | −0.015 [−0.047, +0.015] | 7.850 [7.365, 8.352] | **+0.089** [+0.021, +0.159] | 6.890 | 1.33 % |
| 10 | 6.157 [5.770, 6.569] | — | 7.762 [7.295, 8.240] | — | 6.771 | 1.10 % |
| 15 | 6.165 [5.782, 6.573] | +0.008 [−0.017, +0.033] | 7.777 [7.316, 8.251] | +0.015 [−0.011, +0.041] | 6.728 | 0.98 % |
| 20 | 6.191 [5.809, 6.596] | **+0.034** [+0.008, +0.060] | 7.792 [7.334, 8.269] | +0.030 [−0.001, +0.061] | 6.709 | 0.91 % |
| 25 | 6.198 [5.815, 6.605] | **+0.041** [+0.012, +0.071] | 7.819 [7.361, 8.296] | **+0.056** [+0.023, +0.090] | 6.683 | 0.87 % |

Under the primary evaluator the ordering is exactly S = 5 < 10 < 15 < 20 < 25 — the direction of the paper's §4.5
observation — but the whole curve spans **0.06 bpm** and S = 5 is not distinguishable from S = 10. The observation does not
survive a change of evaluator: with project HR S = 5 is the worst, and under the paper-literal convention (a "no HR" output
scored as −1) the ordering reverses (S = 25 best), because shallower samples more often yield no Hamilton HR. What holds under
both HR evaluators is that **extra Euler steps beyond 5 do not make a defined single-sample HR more accurate**; what
depth does buy is fewer "no HR" outputs (1.33 % → 0.87 %), which only the paper-literal convention rewards.

## 7. Fixed-budget results (Hamilton HR, primary estimator = partition average over the 16 draws)
| budget | width-heavy | depth | **width − depth** | patients width better | nested draws | project HR (secondary) |
|---|---|---|---|---|---|---|
| **B10** | (2,5) 5.917 [5.533, 6.324] | (1,10) 6.157 [5.770, 6.569] | **−0.240 [−0.272, −0.209]** | 74 % | −0.291 [−0.397, −0.185] | −0.225 [−0.296, −0.152] |
| **B15** | (3,5) 5.720 [5.337, 6.121] | (1,15) 6.165 [5.782, 6.573] | **−0.445 [−0.485, −0.406]** | 82 % | −0.447 [−0.559, −0.340] | −0.770 [−0.865, −0.673] |
| **B20** | (4,5) 5.650 [5.267, 6.051] | (1,20) 6.191 [5.809, 6.596] | **−0.542 [−0.585, −0.499]** | 86 % | −0.483 [−0.588, −0.382] | −0.910 [−1.007, −0.812] |
| B25 (secondary) | (5,5) 5.611 [5.227, 6.012] | (1,25) 6.198 [5.815, 6.605] | **−0.588 [−0.634, −0.540]** | 86 % | −0.503 [−0.600, −0.401] | −1.096 [−1.204, −0.990] |

18,513 windows, 1,156 patients in every contrast; no ties. B20 intermediate (reported, not selected): (2,10) 5.953
[5.572, 6.358]; (2,10) − (1,20) = −0.238 [−0.267, −0.211]; (4,5) − (2,10) = −0.303 [−0.342, −0.265] — within B20 the
error falls monotonically as the budget moves from depth to width. B10 is a K = 2 median, i.e. a **mean of two** samples;
it is not robust-median evidence. Reduction relative to the depth cell: 3.9 % (B10), 7.2 % (B15), 8.8 % (B20), 9.5 % (B25).
Coverage (share of defined consensus values): K = 1 cells 98.7–99.1 %, width cells 99.8–100 %; undefined samples are
excluded, not penalised, which if anything favours the K = 1 cells.

## 8. Hamilton HR results (primary) — sanity and consensus
| quantity | value |
|---|---|
| (1,10), the paper's main setting | 6.157 [5.770, 6.569] bpm |
| PPG peak counting (`find_peaks`, distance 42, prominence 0.3), same Y\* | 7.657 [7.179, 8.170] → (1,10) is **better by −1.503 [−1.836, −1.193]**; instability flag **not raised** |
| official `ppg_bpm_array` (secondary PPG baseline) | **unusable**: "no HR" for 100 % of windows (§13) |
| constant predictor (median Y\* = 73.3 bpm) | 14.488 [14.035, 14.957] → the checkpoint is technically usable |
| K = 16 median at S = 5 (80 NFE) − K = 1 at S = 5 | **−0.658 [−0.709, −0.607]**, 91 % of patients (useful consensus gain → the FAILED flag is not raised) |
| K = 16 median at S = 5 / 10 / 15 / 20 / 25 | 5.484 / 5.540 / 5.560 / 5.573 / 5.576 |

## 9. Project-standard HR results (secondary; directional consistency only)
Every headline contrast has the same sign, all CIs exclude 0, and the magnitudes are larger (−0.23 / −0.77 / −0.91 / −1.10;
win rates 71 / 79 / 79 / 81 %). Absolute errors are higher than Hamilton's ((1,10) 7.762; (4,5) 6.881) and not comparable
with it. One difference from the primary: at K = 16 the project evaluator favours deeper samples ((16,5) 6.351 vs (16,10)
5.802), whereas Hamilton favours S = 5 (5.484 vs 5.540); project-HR coverage of single samples is lowest at S = 5
(94.4 % vs 97.0–98.2 %).

## 10. Waveform / event trade-off (per sample, mean of 16 draws, patient macro)
| S | MAE | RMSE | Pearson r | R-peak F1 @ 50 ms | RR-MAE (ms, matched beats) | pairwise waveform RMS | FD (official, draw 0) |
|---|---|---|---|---|---|---|---|
| 5 | 0.850 | 1.223 | 0.017 | 0.124 | 15.7 | 0.991 | 51.2 |
| 10 | 0.856 | 1.229 | 0.014 | 0.127 | 17.8 | 1.054 | 45.8 |
| 15 | 0.856 | 1.228 | 0.012 | 0.129 | 18.4 | 1.070 | 42.1 |
| 20 | 0.855 | 1.227 | 0.011 | 0.129 | 18.7 | 1.076 | 39.6 |
| 25 | 0.855 | 1.225 | 0.011 | 0.129 | 18.9 | 1.079 | 37.8 |

**No relative collapse at S = 5** (preregistered check: F1 0.124 vs best deep 0.129; RMSE 1.223 vs 1.225) — the
width-heavy cells lose nothing on these metrics. Depth does improve the waveform *distribution* (FD 51.2 → 37.8, −26 %),
as it did for PENGUIN / iMF, without improving HR.
**Absolute waveform fidelity is essentially zero at every depth**: per-sample correlation with the reference 0.01–0.02,
RMSE ≈ that of two uncorrelated signals of these variances (1.25), R-peak F1 0.12–0.13. *Post-hoc* (`posthoc.json`): the best
cross-correlation lag within ±600 ms (S = 10) has median +62.5 ms but IQR [−195, +242] ms, only 14 % of windows within
±50 ms; correlation at the best lag median 0.51. Two draws of the same window disagree on that lag by a median 102 ms.
The generated beats carry the **rate** but not a consistent **phase** relative to the reference (unlike RDDM, whose
VitalDB beats were consistently ~109 ms late). All HR results are therefore about an HR functional of samples that do not
reproduce beat timing on this corpus.

## 11. Functional-error dependence (fixed K = 16, same 16 draws at every S)
Ω_mech = 15,802 windows / 1,154 patients with Y\* and all 80 sample HRs defined (the easier 85 % of windows).

| S | individual I | consensus C | gain G [95 % CI] | G / I | functional SD | MAD | waveform pairwise RMS | ρ̄ [95 % CI] | K_eff |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 3.531 | 3.295 | 0.237 [0.208, 0.269] | 6.7 % | 0.945 | 0.328 | 1.013 | 0.939 [0.926, 0.950] | 1.06 |
| 10 | 3.598 | 3.299 | 0.299 [0.275, 0.325] | 8.3 % | 1.154 | 0.346 | 1.073 | 0.931 [0.917, 0.943] | 1.07 |
| 15 | 3.647 | 3.303 | 0.344 [0.318, 0.372] | 9.4 % | 1.284 | 0.370 | 1.090 | 0.925 [0.911, 0.937] | 1.08 |
| 20 | 3.675 | 3.306 | 0.368 [0.343, 0.395] | 10.0 % | 1.369 | 0.374 | 1.096 | 0.921 [0.905, 0.933] | 1.08 |
| 25 | 3.702 | 3.318 | 0.384 [0.358, 0.411] | 10.4 % | 1.441 | 0.394 | 1.099 | 0.916 [0.900, 0.929] | 1.09 |

- **Preregistered question — reproduces**: across the five depths Spearman(ρ̄_S, G_S) = **−1.0**, negative in 100 % of
  5,000 patient-bootstrap replicates (also −1.0 against G / I). Lower functional-error redundancy goes with larger gain.
- **But nothing is discriminated across S**: waveform pairwise RMS, SD and MAD are all +1.0 with G (MAD interval
  [+0.9, +1.0]). In this model every diversity measure increases monotonically with depth, so the across-S test cannot
  separate the functional-error account from the waveform-diversity account (unlike iMF, where waveform RMS flipped sign).
  With five points the result is descriptive.
- **Patient level (preregistered secondary, as RDDM-EXT)**, 1,110 patients with ≥ 8 windows: Spearman(ρ̄_p, G_p) = −0.507
  [−0.548, −0.465] at S = 5, −0.464 to −0.471 at S = 10–25, negative in 100 % of replicates at every S.
- *Post-hoc, patient level*: waveform pairwise RMS vs G_p ≈ 0 (+0.066 [+0.005, +0.126] at S = 5, +0.072 at S = 10,
  −0.011 to +0.011 at S = 15–25, CIs containing 0), while functional SD vs G_p = +0.73 to +0.79 and MAD +0.51 to +0.55.
  Across patients the gain tracks the functional quantities, not waveform diversity — the same pattern as B3-BOOT and
  RDDM-EXT, here found post-hoc.
- **Special case B largely applies**: the functional errors are highly redundant (ρ̄ 0.92–0.94, K_eff ≈ 1.06–1.09 of 16),
  so consensus buys only 7–10 % of the individual error on Ω_mech. Width still wins at fixed budget because depth buys
  nothing for HR: from S = 5 to 25 the individual error rises (3.53 → 3.70) while G rises almost as much (0.24 → 0.38),
  leaving the K = 16 consensus error flat (3.295 → 3.318).

## 12. Comparison with previous iMF / CD / PENGUIN findings
| finding in our models | PPGFlowECG (released, zero-shot VitalDB) |
|---|---|
| width beats depth at equal NFE (DW1 / DW2-A 3/3 seeds, WildPPG, EXP-A vs Heun) | **same direction** at B10–B25, all CIs < 0 |
| size: width-heavy − pure depth −1.4 to −4.6 bpm (DW1, DW2-A, EXP-A; e.g. iMF (16,2) 6.16 vs (1,32) 8.26, −25 %) | **much smaller**: −0.24 to −0.59 bpm (4–10 %) |
| why width wins: large consensus gains (iMF K = 16 gain ≈ 34 % of the single-sample error) outweigh what depth adds to one sample | depth adds **nothing** to a defined single-sample HR (6.14 → 6.20 over S = 5–25); the consensus gain is small (7–10 %) but uncontested |
| ρ̄ 0.56–0.86, K_eff 1.2–1.7; RDDM 0.89 | ρ̄ 0.92–0.94, K_eff ≈ 1.1 — the most redundant generator so far |
| what depth does: iMF improves the sample, PENGUIN makes samples poolable (G 1.24 → 2.43), CD neither | PENGUIN-like: G grows with S (0.24 → 0.38) but consensus error does not improve |
| gain tracks functional-error non-redundancy, not waveform diversity (B3-BOOT across conditions; RDDM-EXT patient level) | across S: tracks both (cannot discriminate); patient level: ρ̄_p −0.46 to −0.51 (prereg), waveform RMS ≈ 0 (post-hoc) |
| depth improves the waveform distribution (PENGUIN FD 33 → 4.5, iMF 4.2 → 3.2) | FD 51 → 38 with S, no HR benefit |
| median > mean from K = 3 (B3-BOOT); not on RDDM | not tested here |

## 13. Failures
1. **Official PPG HR baseline unusable**: `calculate_metric.ppg_bpm_array` calls `biosppy.signals.ppg.ppg_findpeaks`, which
   does not exist in the paper-pinned biosppy 2.2.3 (it is a neurokit2 name); the function's own `try/except` turns the
   AttributeError into "no HR" for **every** window. The preregistered primary PPG baseline (`find_peaks`) is unaffected.
2. **No beat alignment on VitalDB** at any depth (r ≈ 0.01, F1 ≈ 0.13; post-hoc: no consistent phase). The model's
   zero-shot waveform output is not a faithful ECG of these patients; only its rate information transfers.
3. **The paper's "T = 5 best" is evaluator-dependent here** (§6): primary Hamilton ordering matches the paper, but
   S = 5 ≈ S = 10, the project evaluator ranks S = 5 last, and the paper-literal −1 convention reverses the ordering.
4. Released-code defects documented in §2 (`sampling_steps` not forwarded; missing `utils.data`); worked around from our
   side without editing the upstream source.
5. Not every element of the previous-model story could be tested: the median-vs-mean question was not addressed, and the
   across-S mechanism test cannot discriminate waveform from functional diversity in this model.

## 14. Statistical caveats
- One released checkpoint, one external corpus used zero-shot, one functional (HR), one extractor as primary; four headline
  contrasts without multiplicity correction (all CI upper bounds ≤ −0.21).
- The budget equalises **vector-field NFE only**; width cells run 3K VAE passes, so they cost more total compute and are
  32–57 % slower when run sequentially (faster when batched). *Descriptive, post-hoc:* because the depth curve is flat,
  width also wins at matched sequential wall-clock ((2,5) 39.6 ms 5.917 vs (1,15) 40.1 ms 6.165; (3,5) 60.0 ms 5.720 vs
  (1,25) 63.2 ms 6.198).
- Draws share noise across S (common random numbers) and the partition estimator reuses the same 16 draws for every cell:
  the contrasts are paired by design. The nested-draw estimator agrees in sign and size.
- Y\* is the Hamilton HR of the processed reference ECG without quality selection; reference-side errors are common to
  all cells and cannot create a within-model contrast, but they inflate absolute errors.
- Undefined HRs (~1 % of single samples) are excluded rather than penalised; this favours K = 1 cells slightly.
- The mechanism analysis has five depths of one model, all of whose diversity measures move together; Ω_mech excludes the
  15 % hardest windows. The patient-level diversity analysis is post-hoc.
- Absolute numbers are not comparable with the project's 4-s VitalDB results or with the paper's MCMED tables.

## 15. Claim audit
| claim | status |
|---|---|
| "An independently developed latent rectified-flow generator exhibits the same fixed-budget preference for allocating compute toward multiple sufficiently refined samples." | **supported** for HR on VitalDB (B10, B15, B20 all significant, B25 as well), with the boundaries below |
| boundary: the preference here is small (4–10 %) and arises because depth beyond S = 5 buys no accuracy of defined HR estimates (only fewer "no HR" outputs), not because shallow samples are better samples | stated |
| boundary: equal vector-field NFE, not equal total compute or sequential wall-clock | stated |
| boundary: samples reproduce rate, not beat timing, on this corpus | stated |
| gain governed by functional-error non-redundancy | consistent (across S −1.0; patient level −0.46 to −0.51); **not discriminated** from waveform diversity across S; post-hoc patient-level evidence favours the functional account |
| paper §4.5 "T = 5 best" | direction seen under the primary evaluator only; not robust — not claimed |
| width always wins / T = 5 universally optimal / all flow models benefit | **not claimed** |
| checkpoint confirmed MCMED-trained | **not claimed** (provenance wording of §1) |
| improves PPGFlowECG's reported test performance | **not claimed** (different corpus, zero-shot, evaluator differences) |
| causality from an external model / generation beats direct HR prediction | **not claimed** |

## 16. External-replication verdict: **STRONG** (frozen rule of the preregistration §8)
| rule item | result |
|---|---|
| technical flags (non-finite > 1 %, Hamilton undefined > 50 % at (1,10), Y\* undefined > 50 %) | none (0 %, 1.1 %, 0.06 %) |
| not usable (E(1,10) ≥ constant baseline) | no (6.157 vs 14.488) |
| no useful consensus gain (E(16,5) − E(1,5) CI upper ≥ 0) | no (−0.658 [−0.709, −0.607]) |
| waveform / event collapse at S = 5 (relative to deeper S) | no |
| primary budgets succeeding (CI < 0) | **B10, B15, B20** (3 of 3; includes B15 and B20); none significant for depth |
| B25 (secondary, cannot raise the verdict) | also succeeds |
| instability flag (K = 1 worse than PPG peaks) | not raised |

**STRONG** means exactly this: on the authors' released PPGFlowECG checkpoint, used without retraining and with only the
paper-validated step counts, spending a fixed vector-field budget on several S = 5 samples and taking the median HR beats
spending it on one deeper Euler trajectory, at every preregistered budget, on 1,156 external patients. It does **not**
mean the consensus gain is large (ρ̄ ≈ 0.93, gains 4–10 %), that width is cheaper in total compute, or that the samples are
good ECGs of these patients (they carry rate, not beat timing). Stopping here: no respiration / ABP / new model.
