# D2 — Baseline Floor and Conditioning Controls: REPORT

Preregistration: `docs/D2_BASELINE_FLOOR_PREREGISTRATION.md` (commit `0ac1664`, pushed before any D2
number was computed). Implementation: `1dca6a1`. Run 2026-09-06, ~1.7 h, no training and no weight
update. Outputs: `outputs/d2_baselines/{summary_by_arm.csv,paired_vs_ours.csv,d2_meta.json}` and
`figures/` (FIG 10, FIG 11). Not in git by policy.

**Population validation.** For every corpus the loaded `(subject, window_index)` sequence was asserted
equal to D1's own `per_window_metrics.csv` sequence. The `ours` arm, regenerated from the D1
checkpoint with D1's `SOURCE_SEED`, reproduces D1's number exactly (CapnoBase F1@50: 0.499 = 0.499).

## 1. The headline table — R-peak F1 @ 50 ms, subject-macro

| arm | WildPPG | PPG-DaLiA | BIDMC | CapnoBase | VitalDB |
|---|---|---|---|---|---|
| **ours (NFE 1)** | **0.363** | **0.150** | **0.494** | **0.499** | **0.752** |
| B4 mismatched PPG | 0.141 | 0.149 | 0.130 | 0.270 | 0.114 |
| B0 wrong-window *(diag)* | 0.153 | 0.155 | 0.082 | 0.300 | 0.119 |
| B1 PPG + template | 0.194 | 0.123 | 0.388 | 0.061 | 0.205 |
| B2 PPG + mean beat | 0.198 | 0.150 | 0.377 | 0.059 | 0.203 |
| B3 train-mean waveform | 0.048 | 0.024 | 0.064 | 0.070 | 0.101 |
| B5 GT-timing *(GT-R leakage; diagnostic only)* | 1.000 | 0.999 | 1.000 | 1.000 | 0.997 |

Paired differences, ours − baseline, subject-clustered 95 % CI (`*` = excludes 0):

| vs | WildPPG | PPG-DaLiA | BIDMC | CapnoBase | VitalDB |
|---|---|---|---|---|---|
| B4 mismatched PPG | +0.223 [+0.13,+0.32]\* | +0.001 [+0.00,+0.00]\* | +0.364 [+0.10,+0.59]\* | +0.229 [+0.01,+0.45]\* | +0.639 [+0.60,+0.68]\* |
| B1 PPG + template | +0.170 [+0.15,+0.18]\* | +0.027 [+0.02,+0.03]\* | +0.105 [−0.30,+0.46] | +0.438 [+0.17,+0.71]\* | +0.547 [+0.49,+0.60]\* |
| B0 wrong-window | +0.210 [+0.11,+0.31]\* | −0.005 [−0.01,−0.00]\* | +0.411 [+0.20,+0.61]\* | +0.199 [−0.00,+0.41] | +0.633 [+0.59,+0.67]\* |

## 2. Outcome of the five preregistered predictions — 2 confirmed, 3 refuted

| | prediction | outcome |
|---|---|---|
| P1 | ours beats B1 and B2 on VitalDB, CI excludes 0 | **CONFIRMED** (+0.547, +0.549) |
| P2 | ours does **not** beat B1 on DaLiA or CapnoBase | **REFUTED on both** (+0.027\*, +0.438\*) |
| P3 | B4 lands inside the matched CI on ≥ 3 of 5 corpora | **REFUTED** — B4 is worse on all five |
| P4 | on VitalDB B4 is clearly worse | **CONFIRMED** (+0.639) |
| P5 | B0 exceeds our PCC on CapnoBase | **REFUTED** (ours 0.058 vs B0 0.032) |

**Every refutation went against my prior, and the prior was the pessimistic one.** This is the value
of having frozen the predictions: the 2026-09-05 spot check that motivated D2 measured CapnoBase PCC
as ours 0.053 vs null 0.070 and I read it as "the model does not condition on the PPG". That spot
check ran on the 96-window `waveforms_nfe1.npz` subset; on the full 360-window population the sign
reverses. The subset bias was the same defect later found and fixed in the D2 driver (§7).

## 3. What D2 establishes

**3.1 The model does condition on the PPG — on four of five corpora.** Feeding the same checkpoint,
the same NFE and the same noise draw, but a *different window's PPG from the same subject* (B4),
collapses R-peak F1: VitalDB 0.752 → 0.114, BIDMC 0.494 → 0.130, WildPPG 0.363 → 0.141, CapnoBase
0.499 → 0.270. The gap excludes zero everywhere. The hypothesis I offered on 2026-09-05 — that the
model had learned the ECG marginal rather than the PPG→ECG conditional — **is wrong except on DaLiA.**

**3.2 PPG-DaLiA is the genuine failure.** ours 0.150, B4 0.149, B0 0.155, B1 0.123. The conditioning
effect is +0.001, statistically non-zero only because the two test subjects give an almost degenerate
bootstrap. Against the wrong-window control the model is *worse* (−0.005\*). On DaLiA the model is
doing nothing that depends on the PPG it is given.

**3.3 On BIDMC the model is not distinguishable from PPG-peaks-plus-a-template.** +0.105 with CI
[−0.30, +0.46]. BIDMC's apparently respectable 0.494 is matched by a predictor that detects PPG
peaks, adds one corpus-wide constant, and stamps a fixed QRS shape. That is the single most
deflationary number in D2.

**3.4 RMSE is not merely uninformative here — it is anti-informative.** B3, which emits one fixed
train-mean waveform and uses nothing whatsoever from the input, has the **lowest RMSE on every
corpus** (0.309–0.405 vs ours 0.445–0.560). The perfect-timing oracle B5 has *worse* RMSE than our
model on three corpora (e.g. VitalDB 0.598 vs 0.478) while scoring F1 0.997 and PCC 0.657. Any
ranking by RMSE — and by MAE, MSE or PRD, which are monotone in the same error — rewards
under-modelling. **D1's RMSE/MAE/MSE/PRD columns should not be read as quality.**

**3.5 HR error is worse than useless as evidence.** B1 beats our model on HR error on three of five
corpora: BIDMC 1.602 vs 1.972, CapnoBase 1.584 vs 17.511, VitalDB 6.364 vs 8.019. A predictor with
F1 0.061 on CapnoBase gets HR error 1.58 bpm. This closes the question D1 §3.3 opened: the HR metric
is satisfied by the beat *rate*, which the PPG hands over directly, and it certifies nothing about
reconstruction.

**3.6 The whole reachable gap is timing.** B5 — a single fixed QRS template stamped at the true R
peaks, with no morphology modelling at all — reaches F1 ≈ 1.0 and PCC 0.61–0.77, against our
0.002–0.30. Knowing where the beats go is worth roughly 0.7 PCC; our model captures at most 0.30 of
it and usually 0.05.

## 4. Where the method actually stands

| corpus | verdict |
|---|---|
| VitalDB | clearly above every non-leaky baseline (+0.55 over B1). The one corpus where the method works. |
| WildPPG | above all baselines (+0.17 over B1) but at an absolute level of 0.363. |
| CapnoBase | above the baselines (+0.44), though B1 is unusually weak there (test placement ceiling 0.060). |
| BIDMC | **not distinguishable** from B1. |
| PPG-DaLiA | **no conditioning detectable.** |

## 5. Limitations

- **B1/B2 use a single corpus-wide PAT offset.** Measured offsets: WildPPG 61 samples (477 ms),
  CapnoBase 34 (266 ms), BIDMC 1 (8 ms), DaLiA 0, VitalDB 0. A 0 ms pulse-arrival time is not
  physiological; on DaLiA and VitalDB the fit found no informative offset, so B1/B2 are weaker there
  than a per-subject PAT would allow. A per-subject offset would need test ground truth, which is
  why it is not used — but it means B1 is a **floor, not a strong baseline**, especially on
  CapnoBase where its test placement ceiling is only 0.060 against a train match rate of 0.321.
- **Two-subject test splits** on WildPPG and DaLiA make their CIs descriptions of two people.
- **B0 and B5 consume test ground truth** and are diagnostics bounding what is reachable, never
  claims about our method. Labelled as such in every table and figure.
- D2 compares against *trivial predictors*, never against a published method.

## 6. What this changes about the research direction

D1 §8 concluded the binding constraint is beat placement. D2 sharpens it and removes two candidate
explanations:

1. **Conditioning is not the problem** (except on DaLiA). The model reads the PPG. It reads it
   *imprecisely*.
2. **Waveform-error metrics cannot guide this work.** RMSE prefers a constant; HR error prefers a
   template. Any future arm must be judged on timing-sensitive metrics — R-peak F1 across the 25 /
   50 / 100 ms sweep, and the E1–E3 placement decomposition — with RMSE reported only as a
   descriptive column.
3. **DaLiA and BIDMC are the diagnostic corpora**, for opposite reasons: DaLiA because nothing
   conditions, BIDMC because a template matches the model. Whatever explains those two explains the
   method's ceiling.

The two preprocessing outliers from `docs/PREPROCESSING_CONVENTIONS_SURVEY.md` §5 — the 4 Hz PPG
ceiling and per-window `filtfilt` after segmentation — remain the cheapest untested explanations, and
D2 now supplies the floor against which any change to them can be judged. Nothing in that direction
has been started; each needs its own preregistration.

## 7. Deviations from the preregistration

1. **Two defects found during the CapnoBase smoke test and fixed before the full run.** (a) The
   `ours` arm was being read from `waveforms_nfe1.npz`, which stores 16 rows per subject, so it would
   have been scored on 96 of 360 windows while every baseline saw all 360 — the paired comparison
   would not have been paired. `ours` and B4 are now generated over the full population in one code
   path with the same noise tensor. (b) The B1 QRS template sat on the ECG baseline (≈ −0.4), so
   additive placement stepped from 0 at every template edge; prereg §4 calls it "zero outside", and
   the template ends are now centred on zero.
2. **The PAT tie-break was underspecified** in prereg §4. It is implemented as "smallest maximiser"
   and measured to be moot: on CapnoBase train the match-rate curve has a single sharp maximum
   (plateau width 1 at offset 31), so it never binds on real data.
3. **`per_window_metrics.csv` and `per_subject_metrics.csv` were not written**, only the arm summary
   and the paired table. FIG 11 is therefore the paired-difference form rather than the per-window
   scatter named in prereg §6. No conclusion depends on the missing files; the driver has not been
   re-run to produce them because doing so would consume ~1.7 h for no change to any number.
