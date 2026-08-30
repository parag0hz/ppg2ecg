# X3-G0 — Pre-preregistration design audit (DESIGN PILOT EVIDENCE, NOT CONFIRMATORY)

Written 2026-08-30, **before** the X3-G0 pre-registration, while auditing a *proposed* X3 training experiment and a first draft
of the G0 gate. Everything recorded here was observed **before** any G0 protocol was frozen. It is **design-pilot evidence used to
choose the G0 arms**, and it must never be presented as a G0 result or mixed into the primary G0 tables.

Full disclosure is the point of this document. Nothing observed is omitted, including one access that the later G0 protocol forbids.

## 1. What was inspected, and on which data

| # | Question being audited | Data touched | Model touched | Result used for |
|---|---|---|---|---|
| A | Is the proposed "oracle residual-OT" a genuinely different cost geometry from raw-waveform OT? | **WildPPG TEST subjects (kjd, ssx)** — the frozen 3,907-window arrays already published by X0/X2 | frozen A6c prediction array (`regressor.npz`), read-only | Demoting residual-OT from primary positive control to secondary |
| B | Do spectrally reweighted costs give genuinely different assignments? | WildPPG **validation** subjects (an0, k2s), ground-truth ECG only | none | Adding WHITE and HF as the primary G0 cost arms |

No model was trained, fine-tuned or re-run. No file was written or modified. Both checks were read-only.

### 1.1 The test-set access — stated plainly

Check A read `outputs/a4_otcfm_wildppg_seed42/predictions/test_inputs.npz` (`y`) and
`outputs/a6c_fullbackbone_mse_wildppg_seed42/predictions/regressor.npz` (`pred`) — i.e. **WildPPG test-subject arrays**. These are
frozen arrays whose aggregate statistics were already published in the X0 and X2 reports, and the check computed only
distributional summaries (a correlation, an energy ratio, an assignment-overlap fraction). It nevertheless **is** a test-set access,
and it **did** inform a design decision (demoting residual-OT).

Consequences, applied in the G0 pre-registration:

- The G0 protocol installs a hard firewall: no G0 code may load `kjd` or `ssx`, enforced by a raising assertion.
- The final X3-G0 status report must say **"test accessed during the pre-prereg design audit: YES (read-only, already-published
  frozen arrays); during G0 itself: NO"**. It must not claim an unqualified "NO".
- The residual-OT arm is **secondary** in G0 and no gate verdict depends on it, so the design decision this access informed cannot
  by itself drive the G0 conclusion.
- G0 re-derives the same quantity (alignment of `r = y − m_A6(c)` with `y`) on **train** subjects under the frozen protocol, so the
  conclusion does not rest on the test observation.

### 1.2 The validation access

Check B read ground-truth ECG for `an0` and `k2s` from `data/processed/wildppg_8s/`. This is the peek the G0 pre-registration
already accounts for by moving primary inference to subject-grouped cross-fitting over the 12 **training** subjects and labelling
any later validation run "secondary design-informed confirmation" rather than independent confirmation.

## 2. Exact code executed

Check A (test arrays; correlation, energy ratio, assignment overlap of raw-target vs residual cost):

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
ti = np.load('outputs/a4_otcfm_wildppg_seed42/predictions/test_inputs.npz', allow_pickle=True)
y  = ti['y'].astype(np.float64)                                    # x1
m  = np.load('outputs/a6c_fullbackbone_mse_wildppg_seed42/predictions/regressor.npz',
             allow_pickle=True)['pred'].astype(np.float64)         # m_A6(c)
r  = y - m
def cc(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a*b).sum(-1) / np.sqrt((a*a).sum(-1) * (b*b).sum(-1))
rng = np.random.default_rng(0); T = y.shape[1]
for B in (64, 256):
    ov = []
    for rep in range(20):
        idx = rng.choice(len(y), B, replace=False)
        x0  = rng.standard_normal((B, T))
        assign = lambda tgt: linear_sum_assignment(-(x0 @ tgt[idx].T))[1]
        ov.append((assign(y) == assign(r)).mean())
```

Check B (validation ground truth only; HF and whitened cost geometries):

```python
y = np.concatenate([np.load(f'data/processed/wildppg_8s/{s}.npz')['y'] for s in ('an0','k2s')]).astype(np.float64)
T, FS = y.shape[1], 128
f = np.fft.rfftfreq(T, 1/FS); Y = np.fft.rfft(y, axis=1)
y_hf = np.fft.irfft(Y * (f > 15.0), n=T, axis=1)                   # brick-wall HF projection
psd  = (np.abs(Y)**2).mean(0); w = 1.0/np.sqrt(psd + 1e-12); w /= w.mean()
y_wh = np.fft.irfft(Y * w, n=T, axis=1)                            # spectrally whitened
# then: assignment overlap of argmax_pi sum <x0, phi(y)> against the raw-target assignment, B in {64, 256}, 20 replicates
```

Note: check B's whitener differs from the frozen G0 `WHITE` transform (G0 removes the per-window mean, sets `w(0) = 0`, applies a
pre-registered PSD floor, and is fitted per cross-fitting fold on fit subjects only). Check B is therefore a rough pilot of the same
idea, not the frozen primitive.

## 3. Results observed

**Check A — WildPPG test subjects (n = 3,907 windows).**

| Quantity | Value |
|---|---|
| per-window corr(`y`, `r = y − m_A6(c)`) | mean **0.960**, median 0.982, 5th pct 0.856 |
| raw energy ratio ‖r‖² / ‖y‖² | mean 1.82 (inflated by a large constant offset in the A6 output) |
| variance-based Var(r)/Var(y), from the published X0 raw-metrics row (a = 0.241, ρ = 0.118) | **1.0014** |
| identical-assignment fraction, raw-target OT vs residual OT, B = 64 (20 replicates) | 0.562 ± 0.052 |
| identical-assignment fraction, B = 256 | 0.496 ± 0.037 |

**Check B — WildPPG validation subjects (n = 49,200 windows), ground truth only.**

| Quantity | Value |
|---|---|
| per-window corr(`y`, HF-projected `y`) | 0.436 |
| per-window corr(`y`, spectrally whitened `y`) | 0.690 |
| HF (>15 Hz) share of window variance | 0.192 |
| identical-assignment fraction vs raw-target OT, HF cost, B = 64 | 0.091 |
| identical-assignment fraction vs raw-target OT, whitened cost, B = 64 | 0.191 |
| identical-assignment fraction vs raw-target OT, HF cost, B = 256 | 0.045 |
| identical-assignment fraction vs raw-target OT, whitened cost, B = 256 | 0.143 |

## 4. Why this changed the G0 design

1. **The originally proposed "oracle residual-OT positive control" was demoted to secondary.** Subtracting the frozen A6 prediction
   leaves a target direction that is ~96 % aligned with the raw target and carries ~100 % of its variance, because A6 explains
   almost none of the ECG in a least-squares sense (amplitude ratio 0.24, waveform PCC 0.118 → Var(r)/Var(y) = 1.0014). A cost built
   on `r` therefore points almost where the raw cost points, so the intended "capacity vs cost geometry" contrast would have been a
   near-tie by construction, and the COST-MISALIGNED branch could essentially never fire.
2. **Spectrally reweighted costs were promoted to primary arms (WHITE, HF).** They point in materially different directions
   (corr 0.69 and 0.44 with the raw target) and produce almost entirely different assignments.
3. **The scalar residual rescaling `s` was deleted from the design.** For an exact one-to-one assignment,
   Σ‖x₀ᵢ − s·y_π(i)‖² = Σ‖x₀‖² + s²Σ‖y‖² − 2s·Σ⟨x₀ᵢ, y_π(i)⟩, and the first two terms are permutation-invariant, so the minimiser is
   `argmax Σ⟨x₀, y⟩` for every s > 0. A single positive global scale is an assignment no-op.
4. **A cross-objective regret manipulation check was added.** Check A shows ~44–50 % of assignment indices change between two costs
   that are 96 % aligned — i.e. index disagreement alone cannot distinguish a real geometry change from near-tie churn. G0 therefore
   measures, for each pair of cost geometries, how much worse one cost's optimal assignment is under the other cost, normalised by
   the random-assignment gap.
5. **Primary inference moved off the validation subjects.** Because check B used validation ground truth to select the WHITE and HF
   arms, validation is no longer an untouched confirmatory set; G0's primary estimate is subject-grouped cross-fitting over the 12
   training subjects, and any validation run is labelled secondary design-informed confirmation.

## 5. Status of these numbers

- They are **design-pilot evidence**, not G0 evidence.
- They must not appear in the primary G0 tables, and no G0 verdict may cite them.
- Two of them (the raw-vs-residual overlap, and the alignment of `r` with `y`) are re-derived inside G0 on training subjects under
  the frozen protocol; the G0 report compares the two and notes any disagreement.
- No further pre-registration-stage peeks were taken after this document was written.
