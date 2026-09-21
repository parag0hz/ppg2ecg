# LW1 — Weighted auxiliary losses on the iMF endpoint: α·MAE + (1−α)·(1−PCC), and a spectral alternative (preregistration)

Frozen and pushed before any LW1 weight update. Seed 42; V1 VitalDB recipe, 14,000 steps; the only change is a
**training-only** auxiliary term on the one-step endpoint `x̂₀ = z_t − t·u(z_t, t, t)` (the M3 hook):
`L = L_iMF + λ · L_aux`, λ = 0.5 fixed.

## Arms (reference = arm I, no auxiliary loss)
| arm | L_aux | note |
|---|---|---|
| **W0** | (1 − PCC) | α = 0 |
| **W5** | 0.5·MAEₙ + 0.5·(1 − PCC) | α = 0.5 — the form proposed by the advisor |
| **W10** | MAEₙ | α = 1 |
| **SP** | multi-resolution STFT spectral convergence (n_fft 32 / 64 / 128) | phase-insensitive alternative |
Both W terms are dimensionless (MAEₙ = MAE divided by the target's mean absolute deviation; an uninformative
prediction scores ≈ 1 on each), so α mixes comparable quantities.

## Stated expectation (so the result can contradict it)
MAE and PCC are both maximised by the conditional mean. The W arms are therefore expected to **improve MAE / PCC and
to reduce sample diversity**, which should **shrink the consensus gain** and raise FD. SP is tolerant to small timing
shifts and penalises the loss of QRS energy, so it is expected not to shrink diversity. Either expectation may fail.

## Evaluation (V1 test, SR1 protocol)
NFE 1, noise seeds 0–3: HR, R-peak F1, RR-MAE, MAE, RMSE, window PCC, FD; K = 16 HR consensus; **sample diversity** =
mean over windows of the across-sample SD of HR (16 draws); **consensus gain** = single-sample HR error − K = 16 error.
Paired patient-clustered bootstrap against arm I.

## Decision per arm (the BB1 rule, NFE 1)
IMPROVES iff HR diff ≤ −1.0 bpm (CI upper < 0) or F1 diff ≥ +0.02 (CI lower > 0); WORSE for the mirror; MIXED; else
NO MEANINGFUL CHANGE. Reported beside it: whether K = 16 consensus HR improves or degrades against arm I's 6.675.
