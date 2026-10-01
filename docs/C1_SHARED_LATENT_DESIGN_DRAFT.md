# C1 — Shared Window Latent for CoherentBeat — DESIGN DRAFT (not a preregistration, not an experiment)

**Status (2026-10-01): design note only.** C0's ARCH-HOLDOUT verdict was STRONG
(`docs/C0_COHERENTBEAT_FEASIBILITY_REPORT.md`), and the C0 preregistration allows only this document. Nothing here is
trained, tuned or evaluated. A C1 experiment would need its own audit, preregistration, split decision and hard stop.

## 1. What motivates C1

| finding | source |
|---|---|
| Independent per-beat noise over-disperses beats within a window: within-window beat diversity 2.33 × real | BF0 (`d380e3b`), G4b practical ratio |
| Independent absolute-level beats break at their junctions (neighbour ghosts with long support, level steps with short cells) | BF0 post-hoc, D0 (`a54ec7e`) |
| A deterministic global field plus compact local residuals keeps placed-event rhythm (ΔF1 −0.001), cuts false R detections (−0.205 / window) and FD (−8.6) against a retrained BF0 deterministic beat. Boundary-near false detections: 1 of 12,761 | C0 ARCH-HOLDOUT |

C0 tested only the **deterministic** substrate. The open question is whether stochastic morphology can be added to that
substrate without bringing back BF0's over-dispersion.

## 2. Candidate form

```
x(t) = g(c, z_window)(t) + Σ_i m_i(t − r_i) · q(c_i, RR_i, z_window)(t − r_i)
```

- `z_window`: one shared morphology latent per window, used by the global field and every local residual.
- `m_i`: C0's compact bump. The supports, spline basis and events stay frozen as in C0.
- No per-beat absolute level. Every beat returns to the shared global field.

## 3. First comparison a C1 preregistration should make

**Independent beat noise** (one latent per beat, as in BF0) **vs shared window noise** (one latent per window), with
everything else held equal:

- the same architecture;
- the same marginal noise dimension;
- the same NFE;
- the same training data (ARCH-TRAIN);
- the same compute, as closely as practical.

**Principal question:** can shared stochasticity reduce BF0-style within-window over-dispersion (diversity ratio to real
ECG) while preserving event fidelity (G1-type ΔF1 against placed events) and improving FD over C0's deterministic output?

## 4. Open design points, to settle in a preregistration (not here)

- **Training.** The stochastic family (conditional flow over the residual and spline coefficients, or another
  likelihood-based family), and its NFE.
- **Gates.** Which BF0 gates carry over (G1, G3, G4a, G4b) and which C0 gates (G2 false events, G3 FD vs deterministic
  C0).
- **Real diversity target.** Within-window diversity relative to real ECG, as in BF0. Seed-to-seed diversity reported
  separately.
- **Data.** ARCH-HOLDOUT has now been evaluated once by C0, so it is no longer untouched for a C1 claim. C1 needs a new
  primary evidence split, or an explicit statement of reuse.

## 5. C0 limitations C1 inherits

- **LOCAL-ONLY confound.** C0's LOCAL-ONLY ablation (g = 0) also removed the baseline offset. The role of a
  *time-varying* global field, as opposed to a constant offset, is untested.
- **Training-level confound.** C0 was trained at window level and BF0-DET at beat level, so architecture and training
  target level are confounded in C0's comparison.
- **Scope.** One dataset (VitalDB), one seed, one detector. Rendered rhythm inherits the detector's misses and false
  events.

This is a design note only. No C1 code, model or result exists.
