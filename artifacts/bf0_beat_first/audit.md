# BF0 pre-commit fidelity audit (2026-09-30)

Audit of the uncommitted BF0 v2 draft against the BF0 execution specification, done **before** the preregistration
commit and before any real-data BF0 number existed.

## 1. State at the start of the audit

- `HEAD` = `0a0852b` (FBC1 compute accounting). Tracked diff: empty.
- BF0 files present and untracked: `docs/BF0_BEAT_FIRST_FEASIBILITY_PREREGISTRATION.md` (revision 2 draft),
  `docs/BEAT_FIRST_ARCHITECTURE_DESIGN_KO.md`, `scripts/bf0_run.py`, `src/ppg2ecg/beatfirst/` (`__init__`, `beats`,
  `model`, `render`, `timing`), `tests/test_bf0_beatfirst.py` (24 tests; full suite 826 passing).
- `outputs/bf0_beat_first/` and `artifacts/bf0_beat_first/` did not exist: **no BF0 checkpoint, render or metric had
  been produced.** No BF0 stage was run on real data during the audit. The only end-to-end execution was a dry run on
  neurokit-simulated ECG / PPG in a scratch directory (§4).

## 2. Items already satisfied (left unchanged)

| spec item | draft |
|---|---|
| §2 arms A / B / C on the same predicted positions, no timing jitter | A1 template, A2 deterministic (L1), A3 stochastic (16 realizations, s = 0 primary); positions `e + round(μ)`; σ never perturbs a waveform |
| §3 oracle arms, diagnostic only | O1–O3 at reference R with the same trained models; no training or selection uses them |
| §4 condition shuffles | S-PPG / S-RR with one deterministic donor rule (salted order, M/2 offset, never the same patient) |
| §5–7 no phase warp, no tangent projection, no end-to-end fine-tuning | fixed-time Hann overlap-add; separate stages |
| §9 FD sample-count fairness | one waveform per window per arm, A3 realization 0 |
| §10 gate definitions, margins, order | G1 (−0.02 F1, +2 ms RR), G3 (both FD differences with CI < 0), G4b (CI > 0; 0.25 ratio practical only), G4a (1 sample), G2 (A3-mean vs template, 0.02); order G1 → G3 → G4b → G4a → G2 |
| §13 bootstrap unit, count, seed | patient; 2,000 per-window / 1,000 FD resamples; seed 20260930 (kept; the spec's 5,000 default applies only if unspecified) |
| §15 training fairness | A2 and A3: same `BeatFlowNet`, beats, optimiser, steps, seed; differ only in the loss |
| §16 splits | BF0 v2 frozen: train for training, validation for calibration and every number, test never loaded. Kept as frozen (§16 forbids split changes); the validation-reuse limitation is now stated (below) |

## 3. Mismatches found and corrected before results

| # | spec | mismatch in the draft | correction |
|---|---|---|---|
| 1 | §14 matched populations, §10 G2 | G2's beat-aligned correlation used `rpeaks.morphology_corr` on each arm's own detections, so every arm had its own beat population | **matched pairs**: reference R ↔ placed position (±50 ms, one-to-one, both windows inside); the pairs depend only on (reference, positions), so A1 / A2 / A3 / A3-mean share them; pairs kept only if finite in every arm of the group (`render.matched_pairs`, `render.pair_correlations`, `bf0_run.window_pair_means`). The detection-based correlation is kept as a secondary metric |
| 2 | §20 verdict cases | verdict letters D / B / C / E / A′ / A | spec Case A–F (same conditions and order); pure functions `compute_gates`, `verdict_case` |
| 3 | §14 | per-window F1 counted windows without any reference beat as 0 in both arms | F1 averaged over evaluable windows (≥ 1 reference beat), the project's KANFlow Eq. 26–27 convention (`paper_metrics.macro_f1`), identical for every arm |
| 4 | §13 same resampled population | per-window CIs and FD CIs used different random draw sequences | one `patient_resamples` sequence: replicate r draws the same patients for every statistic; FD replicates take whole patients and must keep ≥ 3,000 windows (asserted) so `kanflow_fd` never changes regime |
| 5 | §8 seed roles, §18 | roles implied but not frozen as a list | prereg §2.5 (seed roles), `seed_manifest.json`, tests of the noise seeds and of the realization-0 FD waveform |
| 6 | §10 G4b, §23 panel D | D(A3)/D(real) not restricted to common windows; no iMF morphology-diversity comparator | ratio on windows where both are finite; seed-to-seed diversity for A3 (placed R) and iMF (reference R), secondary |
| 7 | §6, §11.4 | no R-location drift diagnostic | `render_drift`: detections vs placed positions (±50 ms) for A1–A3, A3 over 16 realizations, O1–O3 |
| 8 | §14 | missingness not reported | `missingness.json`: windows without positions / reference beats, per-arm non-finite windows, paired-window, pair and beat counts |
| 9 | §11.11 | interval width not stratified | width and coverage per detector-confidence tertile |
| 10 | §18, §19, §22, §23 | no hash manifest, no compute record, no required artifact files, no result figure | stages `manifest` (sha256 of prereg and code), `compute` (parameters, training time, NaN steps, forward passes, batch-1 latency CPU / GPU, software and hardware), `figure` (panels A–E, gate PASS / FAIL in titles); all artifact files of §22 |
| 11 | §18 | prereg did not freeze matched-population rules, FD sample-count rule, claim boundaries, no-BF1 hard stop, validation reuse | prereg §1 (validation reuse: ED1 chose its parameters on this split, FBC1 and M1 drew evidence from it), §2.5, §3.1, §3.2, §4.2–4.5 |
| 12 | §17 test coverage | missing tests for several required items | 23 tests added (below) |
| 13 | §9 FD fairness | A3 renders stored as float16 while A1 / A2 are float32 | A3 renders stored as float32, the same precision as every other arm |

Behaviour-preserving refactors: `shuffled_conditions` (S-PPG / S-RR condition swap, previously inline); JSON writer maps
nan / inf to null.

## 4. Synthetic dry run (no real data)

All nine stages were executed on 840 neurokit-simulated windows (30 + 24 synthetic patients) in a scratch directory,
with 30 training steps, 20 / 4 bootstrap replicates and a lowered detector threshold (synthetic PPG is outside RD1's
training distribution). It found one bug, fixed before the commit: artifact paths outside the repository crashed
`Path.relative_to` (new helper `rel`). Every required artifact file was produced. The dry-run numbers are meaningless
and were discarded.

## 5. Tests

- BF0-specific: 24 → **47** (`tests/test_bf0_beatfirst.py`). Added: shuffle rule vs a hand computation; S-PPG / S-RR
  replace only their own condition; frozen noise seeds per realization; FD uses realization 0 with equal window counts;
  the FD draw applies one window set to both arms; FD bootstrap refuses a regime switch; whole-patient bootstrap;
  equal patient weight; matched pairs (shared, edge / missed / extra beats dropped, oracle identity pairs); pair means
  over pairs finite in every arm; G4b diversity on the same beat set; seed diversity and render drift; gate rules at
  each margin (8 cases); verdict order; training never reads validation or oracle positions; JSON cleaning.
- Full suite: 826 → **849 passed**.
