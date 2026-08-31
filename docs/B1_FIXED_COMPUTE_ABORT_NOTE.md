# B1-v2 Abort Note

> **Metric semantics note (added 2026-08-31).** In this document `morph` / `morphology_corr` is
> **matched-beat morphology correlation** and `rr_mae_ms` is **matched-consecutive-beat RR MAE**. Both are
> conditional on successful ≤50 ms one-to-one R-peak matching and are averaged only over matched beats, so
> neither can fall when a beat is missed. Values and specifications here are unchanged; only the labels and
> their scope are made explicit. See [METRIC_SEMANTICS.md](METRIC_SEMANTICS.md).


**Status: ABORTED / INCOMPLETE — NO CONFIRMATORY VERDICT** (terminated 2026-08-29 22:30, research-priority reallocation to X0).

## Original preregistered objective
`docs/B1_FIXED_COMPUTE_GAP_CURRICULUM_PREREGISTRATION.md` (commit `bdd6419`): at identical architecture, data, optimiser, random
streams, training compute and one-step inference cost, does the source-inspired progressive temporal-gap weighting
β(h,s) = 1 − s + λ·s·(1 − h) (λ = 1.304639, s = 1 − step/T, boundary β = 1, applied after the frozen adaptive weighting) improve
iMeanFlow's 1-NFE conditional ECG morphology over vanilla iMeanFlow? Primary comparison = FINAL fixed-budget checkpoints; success
threshold ΔMorph ≥ +0.03 on ≥ 2/3 conditions with frozen safety criteria.

## Original six-run plan
Fixed budgets = the frozen configs' original 300-round budgets: DaLiA S2 66,000 / DaLiA S1 65,400 / WildPPG 65,482 optimizer steps,
vanilla + curriculum each, early stopping diagnostic-only, ≈ 81 GPU-hours in total.

## Runs completed (2 of 6)
| Run | Steps | Wall | Best-val round | Historical early-stop round | Probe hash |
|---|---|---|---|---|---|
| `b1v2_vanilla_fixed_dalia_s2_seed42` | 66,000 | 11.98 h | 135 | 81 | `5c44decf…` |
| `b1v2_curriculum_fixed_dalia_s2_seed42` | 66,000 | 12.00 h (ratio 1.002) | 25 | 45 | `5c44decf…` (identical) |

## Run active at termination (1 partial)
`b1v2_vanilla_fixed_dalia_s1_seed42`: 173 / 300 rounds, 37,714 / 65,400 steps (s = 0.42), best-val round 16; SIGINT was ignored
(SIG_IGN inherited from the non-interactive nohup launch), SIGTERM terminated it after PID re-verification; `checkpoint_last.pt`
(round 173), `checkpoint_best.pt` and the 0/10/25/50 % fraction checkpoints are preserved; marker `TRAINING_ABORTED`. Not started:
S1 curriculum, WildPPG vanilla, WildPPG curriculum.

## Partial results already observed — NON-CONFIRMATORY PARTIAL RESULTS (DaLiA S2 only, 1-NFE, frozen A2 evaluation)
| Arm | Checkpoint | HR err | Morph | Amp | Cond gain | HF | RMSE | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla | FINAL (66k) | 8.15 | 0.576 | 0.768 | 6.42 | 0.269 | 0.402 | 81.7 ms |
| curriculum | FINAL (66k) | 10.29 | **0.542** | 0.732 | 4.34 | 0.302 | 0.405 | 81.4 ms |
| vanilla | best-val (r135) | 9.38 | 0.590 | 0.846 | 4.77 | 0.328 | 0.409 | 82.5 ms |
| curriculum | best-val (r25) | 10.64 | 0.563 | 0.899 | 2.69 | 0.304 | 0.421 | 82.6 ms |
FINAL paired deltas (curriculum − vanilla): ΔMorph −0.034, HR +26 %, Δamp-fidelity −0.036, Δgain −32 %; best-val: ΔMorph −0.027,
Δgain −44 %. These numbers are exploratory: one condition of three, one seed.

## Reason for termination
The experiment was terminated to reallocate computational resources toward a more fundamental error-decomposition question (X0).
Because the decision was made after observing results from the completed S2 pair, the preregistered B1-v2 confirmatory protocol was
not completed and **no confirmatory B1-v2 verdict is assigned**. The remaining ≈ 45 GPU-hours (S1 curriculum, WildPPG pair) were not
spent.

## What can be used from B1
- The implementation and its validation: the frozen loss re-statement + external β passes vanilla loss/gradient parity, end-of-schedule
  parity, adaptive-weight isolation, RNG-free β and paired-stream tests; the new fixed-compute driver reproduced the historical A2 run
  **to the printed precision at round 1** (mse 0.3007, |u| 0.757, val 0.23352) — a byte-level parity check of the driver.
- The source audit (`docs/B1_GAP_CURRICULUM_SOURCE_AUDIT.md`): exact equation, λ rule, schedule, application order, boundary handling.
- Compute observations: the curriculum has no measurable overhead (time ratio 1.002); a fixed 66k-step DaLiA budget costs 12.0 h.
- The vanilla fixed-budget endpoint itself is informative context: continuing past the historical early-stop (round 81) to 66k steps
  lowered HR error (9.58 → 8.15) and raised the conditioning gain (4.47 → 6.42) while slightly lowering morphology (0.595 → 0.576).
- The completed S2 pair as **exploratory** evidence that, on this condition, the isolated curriculum did not help at either checkpoint.

## What cannot be claimed
- Any preregistered 3-condition B1 conclusion (SUCCESS / NULL / HARMFUL / DATASET-SPECIFIC / TRADE-OFF).
- Any cross-dataset statement about curriculum benefit or harm (S1 and WildPPG were not completed).
- Any confirmatory or statistical claim; the observed S2 deltas are single-seed and outcome-aware.

## Preserved records
Pre-registration, source audit, calibration/budget/schedule artefacts and all six output directories' small files are kept unchanged;
checkpoints and predictions remain local. The B1 pre-registration is not edited.
