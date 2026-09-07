# M2 — GSW-iMF: Gradient-Structure-Weighted Improved MeanFlow — REPORT

Preregistration `docs/M2_STRUCTURE_WEIGHTED_IMEANFLOW_PREREGISTRATION.md` (commit `e8a8084`, pushed
before any structure map, training run or metric). Implementation `f84a923`. Anchor `006b42e`.

# FINAL VERDICT — **D. STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED**

Invoked under preregistration §13: *"G1 or G2 fails → verdict D, STOP. No lambda tuning, no smoothing
change, no extra control arms."* Arms **X** (shifted control) and **Q** (hard-QRS control) were
therefore **not trained**; `outputs/m2_arm_X_seed42` and `_Q_seed42` do not exist.

Both primary gates fail on **both** of their conditions — the relative change is an *increase* in
error rather than a ≥5 % reduction, and both confidence intervals lie entirely **below** zero, i.e.
arm S is significantly **worse**, not merely insufficiently better.

| gate | metric | U | S | rel. | required | CI | result |
|---|---|---|---|---|---|---|---|
| **G1** | `qrs_deriv_rmse` | 0.32100 | 0.33342 | **−3.87 %** | ≥ +5 % | [−0.01265, −0.01220] below 0 | **FAIL** |
| **G2** | `qrs_curvature_err` | 0.21437 | 0.24472 | **−14.16 %** | ≥ +5 % | [−0.03055, −0.03016] below 0 | **FAIL** |

## 1. Baseline reproduction (§12) — bit-exact

Fresh arm U reproduces the frozen A4 checkpoint **byte for byte**: state sha256
`20ba7234f25e0fe30960ba0e441d4b9f751c20003c85fee80ec3c08367aa095e`, selected epoch 45, selection
metric `0.11945885431656277`. This is far stronger than the numeric tolerance §8 allowed, and it means
the §10 pairing is exact rather than approximate: U and S share initialization
(`a79e527d…`, 4,568,707 parameters), data order, batch composition, RNG streams, optimizer, schedule
and step count (`opt_steps = 14409` for both), and differ **only** by the loss multiplier.

## 2. Structure-map audit (§9) — all four S0 gates PASS

| | literal wording (1 subject) | balanced (12 subjects) | gate |
|---|---|---|---|
| S0-A all finite | 1.000 | 1.000 | PASS |
| S0-B mean-normalisation error | 4.4e-16 | 4.4e-16 | PASS (≤ 1e-6) |
| S0-C top-20 % derivative energy (median) | 0.802 | 0.829 | PASS (> 0.50) |
| S0-D ESS/T ≥ 0.50 | 100 % | 100 % | PASS (≥ 99.5 %) |

Diagnostic, no gate (§9): **65.4 %** of the excess weight falls within GT R ± 10 samples although the
map never sees an R peak — it is built from waveform gradient alone. The map did what it was designed
to do; the hypothesis it serves is what failed.

## 3. Primary results — NFE 4, source seed 0, subject-macro over 12,400 ECG-window clusters

| metric | U | S | effect | 95 % CI | rel. |
|---|---|---|---|---|---|
| `rmse` | 0.41522 | 0.43620 | −0.02098 | [−0.02158, −0.02032] | −5.05 % |
| `corr` | 0.10802 | 0.12111 | +0.01308 | [+0.01239, +0.01371] | +12.11 % |
| `f1@50` (raw) | 0.43832 | 0.47157 | +0.03325 | [+0.03177, +0.03468] | +7.59 % |
| `floor_f1@50` | 0.11703 | 0.11805 | +0.00102 | [+0.00062, +0.00143] | +0.87 % |
| `f1_excess@50` | 0.32129 | 0.35352 | +0.03223 | [+0.03079, +0.03373] | +10.03 % |
| `f1@150` | 0.71733 | 0.74533 | +0.02800 | [+0.02668, +0.02928] | +3.90 % |
| `missing@50` | 5.60538 | 5.25206 | +0.35332 | [+0.33845, +0.36778] | +6.30 % |
| `spurious@50` | 4.97760 | 4.75705 | +0.22055 | [+0.20430, +0.23611] | +4.43 % |
| `beats_ratio_dev` | 0.10896 | 0.10401 | +0.00495 | [+0.00395, +0.00597] | +4.54 % |
| `qrs_ptp_dev` | 0.48694 | 0.39009 | +0.09685 | [+0.09514, +0.09844] | +19.89 % |
| `qrs_energy_dev` | 0.64072 | 0.57247 | +0.06826 | [+0.06307, +0.07393] | +10.65 % |
| **`qrs_deriv_rmse`** | 0.32100 | 0.33342 | **−0.01243** | [−0.01265, −0.01220] | **−3.87 %** |
| **`qrs_curvature_err`** | 0.21437 | 0.24472 | **−0.03035** | [−0.03055, −0.03016] | **−14.16 %** |
| `F4__ratio_dev` (HF) | 0.38090 | 0.39907 | −0.01817 | [−0.02239, −0.01408] | −4.77 % |

Positive effect = S better. Raw F1 is never reported without its chance floor and excess.

## 4. Robustness of the failure

- **Per subject (§18):** an0 −4.08 % / −14.94 %, k2s −3.67 % / −13.39 %. Both subjects fail
  independently on both gate metrics.
- **Per NFE (§24):** `qrs_deriv_rmse` −2.69 % / −3.73 % / −3.87 % and `qrs_curvature_err` −7.62 % /
  −11.98 % / −14.16 % at NFE 1 / 2 / 4. The failure **grows** with the sampling budget.
- **Source stability (§13):** over seeds 0–3 the per-arm SD is 1.2e-4 to 2.5e-4, two orders of
  magnitude smaller than the effects.
- **Not an averaging artifact:** on the element-wise paired rows, S beats U on `qrs_deriv_rmse` in
  only **24.2 %** of windows and on `qrs_curvature_err` in only **6.0 %**.
- **Independent recomputation:** the headline table was reproduced from the raw per-window CSV without
  reusing the evaluator's aggregation path, and again by an independent auditor using its own digest.

## 5. What arm S *did* improve — and why this is not the M2 conclusion

S's losses are confined to the **unnormalised pointwise-L2 family** (`rmse` −5.05 %,
`qrs_deriv_rmse` −3.87 %, `qrs_curvature_err` −14.16 %, `qrs_rmse_core` −1.53 %, and the region
`a1_abs`/`a2_sq`/`a3_dabs` columns). It wins essentially every **scale-normalised and event** metric:
`qrs_ptp_dev` +19.89 %, `qrs_energy_dev` +10.65 %, `corr` +12.11 %, `f1_excess@50` +10.03 %,
`f1@50` +7.59 %, `beats_ratio_dev` +4.54 %, `rr_mae_ms` +3.88 %.

**This does not rescue the hypothesis and is not M2's conclusion.** Preregistration §11 fixed G1 and
G2 on `qrs_deriv_rmse` and `qrs_curvature_err` before any result existed, and §13 makes a G1/G2
failure terminal regardless of what else moves. Reinterpreting an event-fidelity gain as success
afterwards is exactly what the preregistration exists to prevent. The observation describes a
**different hypothesis**, which would require its own preregistration; M2 may not claim it.

## 6. Claim boundary (§31)

M2 claims none of: first ECG structure weighting; first QRS-aware generation; first target-derived
weighting; SOTA; causal importance of derivative energy; that PPG determines QRS morphology; that
weighting solves event geometry; that R timing is solved. RDDM and other target-region-aware ECG
generation remain relevant prior art. M2 used development subjects only, seed 42 only, no fresh test,
no C2, no external dataset, and the target ECG only to construct the **training** loss weight.

## 7. Adversarial verification before this verdict was committed

Five independent skeptics (sign/orientation, arm identity, selection and budget, metric fidelity, and
a steelman *for* arm S) plus an adjudicator attacked the verdict. **Zero overturning findings; all six
returned verdict D.** They confirmed the checkpoints are not swapped, the map is computed from the
target ECG and not the PPG, the adaptive weight `w` came from the unweighted `delta2` as §2.1 required,
the arms are element-wise paired, and no forbidden subject was loaded. Findings that survived are
disclosed below.

## 8. Deviations and disclosures

**D-1 — anchor.** The specification named `89c4fed` as start HEAD; the repository had advanced nine
commits. M2 was re-anchored to `006b42e` with the user's approval, justified by
`git diff 89c4fed..006b42e` being empty over `src/ppg2ecg/flow/`, `training/`, `models/`, `configs/`
and `external/`, with no commit touching `outputs/` or the E2 contract. Recorded in the preregistration.

**D-2 — audit coverage.** The preregistration's literal wording ("first 8,192 in manifest order over
sorted subjects") draws every window from `e61`, which alone holds 22,228. Both the literal and a
per-subject-balanced audit were run; both pass all four S0 gates.

**D-3 — compute budget misstated in the preregistration.** §3 asserted 4,583 optimizer steps per epoch
and 302,478 total. That is **wrong**: this trainer's "epoch" is a `--val-every-steps 220` validation
round, not a pass over the data, so the realised budget is **14,409 optimizer steps** (66 rounds,
≈3.14 passes over 293,271 windows). Both arms ran exactly 14,409, and U reproduced A4 bit-exactly, so
the executed schedule *is* A4's realised schedule and the pairing is unaffected — the error is
confined to the preregistration's descriptive arithmetic. The already-committed
`paired_initialization_manifest.json` carries the wrong figures; it is left unedited (it is part of the
frozen pre-training record) and is corrected here.

**D-4 — metric-orientation defect in the evaluator, found by verification and fixed before commit.**
`is_lower_better()` classified 13 error-magnitude columns (`region_errors`' `_abs`, `_sq`, `_dabs`,
`_amp` in all three regions, plus `qrs_rmse_core`) as higher-is-better, inverting their `effect` and
`rel_improvement` signs. **No gate or non-inferiority metric was affected** — all ten are correctly
oriented — and the inversion ran *against* arm S on 11 of 13 rows, so correcting it makes S look worse,
not better. The rule is now explicit rather than suffix-guessed, raw band energies and beat counts are
marked `neutral` (no direction is claimed for them), and `bootstrap_effects.csv` was regenerated from
the unchanged per-window data. Corrected, `qrs_core__a3_dabs` — an independent QRS-core derivative
error — worsens by **7.98 %**, reinforcing G1 rather than opposing it.

**Selection sensitivity — measured on the full cohort, post-verdict, cannot change it.** U's
checkpoint was selected at epoch 45 and S's at epoch 60 by the same frozen `fixed_imf_mse` rule.
Comparing each arm's **final-epoch** checkpoint instead, over the identical 49,200-row / 12,400-cluster
cohort at NFE 4 seed 0:

| metric | selected (verdict basis) | final-epoch (diagnostic) |
|---|---|---|
| `qrs_deriv_rmse` | 0.32100 → 0.33342, **−3.87 %** | 0.32604 → 0.32953, **−1.07 %** |
| `qrs_curvature_err` | 0.21437 → 0.24472, **−14.16 %** | 0.22595 → 0.24164, **−6.94 %** |

**S remains worse on both metrics under either pairing, so the verdict is unchanged** — but roughly
half the headline magnitude is attributable to the checkpoint-selection rule, and under the
final-epoch pairing G1's deficit (−1.07 %) is small. This is disclosed so the magnitude is not
over-read; the verdict rests on the preregistered pairing, and no post-hoc pairing may be substituted
for it. (An independent auditor measured the same direction on a 4,000-window subset, −0.59 % and
−5.82 %; the full-cohort figures above supersede those.)

**Bootstrap scope.** `clustered_paired_bootstrap` resamples ECG-window clusters *within* subject and
never resamples subjects (n = 2). The ~4.5e-4 CI widths reflect ~6,200 clusters per subject, not two
subjects, and understate between-subject uncertainty by construction. This is exactly what §10 froze
and what §16's claim boundary already forbids generalising from; the tightness must not be read as
population precision.

**Duplicated statistic.** In the frozen `m1_structural.qrs_core_morphology`, `qrs_slope_dev` and
`qrs_maxderiv_dev` are the same quantity by construction and their rows are bit-identical. They are
one observation, not two.

**Cosmetic artifact.** `outputs/m2_arm_U_seed42/training_summary.json` records `"early_stopped": true`.
That field reports `no_improve >= patience`, not truncation: U ran all 66 rounds with
`--no-early-stop`, zero early-stop events in its log. No trajectory was cut short.

**Evaluator RNG reuse.** The chance-floor generator is seeded `1000 + i` where `i` is the index within
a 256-window chunk, so 256 seed streams repeat across chunks. It is deterministic and identical across
arms and therefore cannot bias U vs S, but `floor_f1@50` has less floor-sampling diversity than 49,200
independent draws would imply. No gate depends on it.

**S0 audit descriptive discrepancies.** The audit script reports the q95 distribution with zero-padded
`np.convolve` while the map uses reflect padding (median 0.39059 reported vs 0.39079 actual), and
`np.argpartition` tie-breaking shifts the top-20 % concentration median from 0.80225 to 0.80155.
Neither changes an S0 gate outcome.

## 9. Artifacts

`artifacts/m2_structure_weighted_imeanflow/`: `repository_integrity.json`, `imeanflow_loss_audit.json`,
`training_config_frozen.json`, `paired_initialization_manifest.json`,
`train_structure_map_summary_{literal,balanced}.json`, `bootstrap_effects.csv`, `gates.csv`,
`decision.json`, `provenance.json`, `evaluation_meta.json`. `validation_metrics.csv` (590,400 rows,
562 MB) is not committed; its sha256 is recorded in `provenance.json` per §28.

## 10. What M2 does not prove

Development subjects only (`an0`, `k2s`; `kjd`/`ssx` never loaded). Seed 42 only. No fresh test set.
No C2. No external dataset. Two validation subjects, so no population claim. The result is that *this*
soft gradient-structure weighting, at λ = 2.0 with this map and this recipe, does not improve
differential ECG morphology — not that spatial weighting in general cannot.

## 11. Recommended next step

Per the verdict tree: **stop this direction.** No lambda tuning, no smoothing change, no derivative-only
or curvature-only variant, no arm X, no arm Q. A different hypothesis requires a new preregistration.
