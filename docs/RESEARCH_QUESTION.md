# Research Question

> **Can PPG-conditioned ECG reconstruction be reduced to one-step generation without sacrificing
> clinically meaningful ECG morphology and conditional fidelity?**

Created: 2026-08-25. Status: **v0 — frozen before any training run.**

## 1. Framing

PENGUIN (Suzuki et al., Neurogica; arXiv:2602.03858, ICASSP 2026) reconstructs ECG from wrist PPG with a conditional flow-matching model
(OT-CFM objective, Flow-SSM / S5 backbone) sampled with a 25-step Heun solver
(= **50 velocity-network evaluations per sample**; see `docs/PENGUIN_AUDIT.md`).
The question is *not* "can we build a better architecture", but:

1. **How does reconstruction quality degrade as the number of function evaluations (NFE) shrinks**
   for the *same trained model* (25 → 10 → 5 → 2 → 1)?
2. **What exactly breaks at 1 NFE** — ECG *morphology* (QRS shape/width, P/T presence, beat-template
   correlation) or *conditional fidelity* (does the output still follow the PPG's rhythm, or does it
   regress to an unconditional "average ECG")?
3. **Only if (2) shows a real failure**: does a one-step objective (Improved MeanFlow, iMeanFlow) on the
   *identical* S5 backbone recover it?

## 2. Operational definitions

| Term | Operationalisation (details in `docs/PREREGISTRATION_V0.md`) |
|---|---|
| Clinically meaningful morphology | R-peak precision/recall/F1 (tolerance 50 ms), QRS-width error, beat-aligned template correlation, RR-interval MAE |
| Conditional fidelity | HR error vs. reference ECG **and** the *PPG-shuffle test*: with mismatched PPG, output HR must track the *given* PPG, not the target |
| "Without sacrificing" | Pre-registered non-inferiority margins in `PREREGISTRATION_V0.md` §5 (frozen before NFE-curve results are viewed) |
| One-step | NFE = 1 velocity/mean-flow network evaluation (Heun 1 step = 2 NFE is **not** one-step) |

## 3. Hypotheses (falsifiable)

- **H1 (degradation exists):** For OT-CFM + S5 sampled with Euler, at least one morphology metric
  crosses its non-inferiority margin at NFE ≤ 2.
- **H2 (failure mode):** At 1 NFE the dominant failure is *morphology blur* (averaging over the
  conditional distribution → widened QRS, reduced template correlation) rather than loss of rhythm
  (HR/R-peak F1 stay within margin). Alternative H2': rhythm is lost (conditioning failure).
- **H3 (objective, not architecture, is the bottleneck):** Replacing OT-CFM with iMeanFlow on the same
  backbone at 1 NFE recovers the failed metrics to within the 50-NFE Heun margin.
  H3 is only tested if H1 is confirmed.

## 4. What is explicitly out of scope (v0)

- New backbones (KAN, Mamba, attention variants), new losses, new conditioning mechanisms.
- Datasets other than PPG-DaLiA (WildPPG / BIDMC etc. only after the DaLiA pipeline is closed).
- Any distillation that needs a teacher other than the reproduced baseline.

## 5. Isolation principle

Architecture and objective effects are separated by design: **every comparison holds the backbone,
data protocol, split, preprocessing, and evaluation code fixed** and varies exactly one of
{sampler steps, objective}. Results that violate this are reported as exploratory only.
