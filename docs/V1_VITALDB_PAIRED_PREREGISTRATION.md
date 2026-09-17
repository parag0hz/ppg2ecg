# V1 — iMF vs PENGUIN (OT-CFM) on VitalDB, patient-level split, single seed (preregistration)

Frozen and pushed before any VitalDB weight update or VitalDB metric in this stage. Single seed (42) by user
decision: a fast check, not a final result. Audit: `artifacts/v0_vitaldb_audit/` (`080ec1c`).

## Data
- Eligible cases = V0 rule (both tracks, ≥ 600 s, finite fraction ≥ 0.95): 6,043 cases, 5,782 patients.
- **Split by patient (`subjectid`)**: patients sorted numerically → `numpy.random.default_rng(20260917).permutation`;
  test = first round(0.20 n), val = next round(0.05 n), train = rest. All cases of a patient go with the patient.
- Windows: 4 s, 128 Hz, D1 VitalDB preprocessing (`PPG_KW`, `ECG_KW`, FFT resample 500 → 128 Hz). Per case,
  candidates at `linspace` over the case's non-overlapping 4 s windows: 128 (train) / 32 (val, test).
- Drop rule, applied identically to every split and every arm (target-side only, never model output):
  non-finite or constant raw window; non-finite after preprocessing; reference ECG (neurokit, 128 Hz) with < 2
  R peaks or HR outside 30–200 bpm. Survivors are subsampled by `linspace` to at most 64 (train) / 16 (val, test).
- Output `data/processed/v1_vitaldb/<case>.npz` (x, y, window_index, subjectid); manifest
  `data/manifests/split_v1_vitaldb_seed42.json`.

## Arms and training — U2 recipe, unchanged
- Arm C = OT-CFM (PENGUIN), arm I = iMF; same backbone, `COMMON` argv of `scripts/u2_run_list.sh`, 14,000
  optimizer steps, seed 42. Both evaluated at `checkpoint_last.pt` (U3-B).

## Evaluation (test patients)
- NFE: C ∈ {1, 2, 4, 50}, I ∈ {1, 2, 4}; 4 noise draws (seeds 0–3), per-window metric averaged over draws.
- Metrics: HR error, R-peak F1 @ 50 ms, RR-MAE, MAE, RMSE, KANFlow FD, micro/macro F1 (U2 definitions).
- **Bootstrap cluster = patient** (2,000 replicates, seed 20260911).
- **H1 (non-inferiority, U2 §8 margins, one-sided gate per U3-A):** iMF is non-inferior at the smallest
  k ∈ {1, 2, 4} where upper(I_k − C_50) < +1.0 bpm for HR AND lower(I_k − C_50) > −0.02 for F1.
  Gate: if C's own span |C_1 − C_50| < |δ| on a metric, a PASS on that metric is withheld (a FAIL stands).
- **H2 (superiority of HR consensus, hypothesis formed on P1 validation data):** iMF NFE 1, K = 16 (seeds 0–15),
  HR = median of per-sample HR. Superior iff upper(consensus − C_50 single-sample HR error) < 0.
  Also reported, not gated: iMF NFE 2 K = 16; C_50 K = 4.
- Reported, not gated: PPG-only HR (scipy `find_peaks` on the preprocessed PPG, distance 42 samples, prominence
  0.3) — the estimator that needs no ECG model at all; mean inference time per window per arm/NFE.

## Verdict strings
H1: `NON-INFERIOR at k=…` / `NOT NON-INFERIOR`. H2: `SUPERIOR` / `NOT SUPERIOR`. Nothing is tuned after results.
