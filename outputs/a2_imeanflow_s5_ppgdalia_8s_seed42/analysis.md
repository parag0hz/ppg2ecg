## Conditional fidelity
The PPG-shuffle gain of iMeanFlow at 1 NFE is 4.47 bpm (78 % of the 50-NFE OT-CFM gain of 5.69; OT-CFM at 1 NFE: 0.24), i.e. the
one-step output still tracks the heart rate of the PPG it was given. With 2 and 4 MeanFlow steps the gain reaches 5.60 and 6.59 bpm —
equal to or above the 50-NFE OT-CFM reference. The rate ceiling of the *backbone* is shared by both objectives: per-activity HR
error of iMF-1 is 6–9 bpm for sedentary/walking windows but 18–23 bpm for cycling/stairs (reference HR 107–114 bpm), the same
regression-to-the-mean pattern as A0/A0-b (`diagnostics.json`), so this is a limitation of the PPG conditioning path / data, not of
the one-step objective. Absolute beat-level timing is unchanged (R-peak F1 0.139, cross-correlation lag uniform over ±0.5 s) —
the PPG-DaLiA protocol limitation applies equally to every arm.

## Qualitative examples
On the pre-registered windows (`controlled_examples_quantile.png`, `controlled_examples_fixed.png`; same PPG, same initial noise,
same y-scale): OT-CFM at 1 NFE is a flat line near −0.3 with ripple (conditional mean); iMeanFlow at 1 NFE produces sharp,
full-amplitude QRS-like spikes at approximately the right rate with a structured baseline, visually comparable to OT-CFM at 50 NFE.
Residual defects visible at 1 NFE: occasional extra small spikes between beats (e.g. window 880: 13 detected vs 10 reference
beats; the beat-count ratio is 1.00 on average because other windows miss beats), a noisier inter-beat baseline than at 50 NFE
(HF-energy ratio 0.252 vs 0.269 target — i.e. *less* high-frequency energy than the target, so not a noise artefact but slightly
smoothed T/P waves), and the same random beat phase as all other arms.

## Failure taxonomy
Applied to iMeanFlow at 1 NFE (pre-registered list):
- **F1 conditional-mean collapse: absent** (amplitude ratio 0.90, seed-to-seed std 0.254 vs 0.242 for OT-CFM-50; OT-CFM-1: 0.145 / 0.033).
- **F2 QRS smoothing: partial/minor** — template correlation 0.595 (95 % CI 0.58–0.61) vs 0.650 (0.64–0.67) at 50 NFE; QRS-width proxy
  error 33.4 ms vs 33.7 ms; recovered fully with 2 steps (0.660) and exceeded with 4 (0.719).
- **F3 amplitude collapse: absent** (0.90; recovery 0.93).
- **F4 conditioning neglect: absent** (gain 4.47 bpm, recovery 0.78; not zero as in OT-CFM-1).
- **F5 unstable JVP training: absent with the official h-only conditioning** (train MSE 0.30 → 0.17, |du/dt| ≈ 0.5, 81 epochs, no
  non-finite values). It **was** present with `E(t)+E(1000·h)` (diverged in 2 epochs) — recorded as an instance of F5 caused by an
  ill-conditioned time embedding, not by the objective.
- **F6 rate recovery but morphology failure: not observed** — both recovered; morphology is the metric with the largest residual
  gap (13 % of the 50→1 gap), plus **F6′ spurious extra spikes** in a minority of windows (new sub-type, see examples).
- **F7 interval under-resolution (shared-embedder `E(t)+E(h)`): pre-empted** by the h-only amendment; not tested to completion.

## Limitations
- Single seed (42), single test subject (S2, 1025 windows) and single validation subject (S11); no variance across subjects/seeds.
- The comparison is *objective-only* by design: iMF was trained with the OT-CFM baseline optimiser (AdamW 1e-3, wd 0.01, no EMA,
  no warm-up) rather than the official recipe (Adam 1e-4, EMA 0.9999); the official auxiliary v-head was replaced by the boundary
  condition v_θ = u_θ(z,t,t) to keep the parameter count; conditioning is h-only (official code) instead of the MF paper's (t, h).
- The 50-NFE OT-CFM reference itself has weak beat-level conditional fidelity (HR r ≈ 0.4 vs reference; regression to the mean),
  so "recovering the 50-NFE level" is a relative statement, not a clinical one.
- Beat-aligned metrics (R-peak F1, PCC, RR MAE) are not interpretable on raw PPG-DaLiA (device synchronisation); QRS width is a proxy.
- Two pre-result amendments of the conditioning (documented in the pre-registration §9) preceded this run; the h-scale sweep was
  not exhaustive (1, 1000, h-only), so h-only is the first stable configuration found, not a tuned optimum.
- Latency parity: iMF-1 and OT-CFM Euler-1 both cost one backbone evaluation (82 ms per batch of 64 on the RTX 5090, fp32, uncompiled);
  iMF training costs ≈ 2× OT-CFM per step (forward-mode JVP) and 3.2 h total vs 1.8 h for A0-b.

## Verdict rationale
All four physiological recovery scores exceed the pre-registered 0.5 threshold (HR 0.96, template correlation 0.87, amplitude 0.93,
conditioning gain 0.78) and the 1-NFE output carries beats (1.00 × reference), so the frozen rule yields **SUCCESS**. The remaining
gaps to the 50-NFE reference are small but real for HR (9.58 vs 8.08 bpm, CIs 9.0–10.1 vs 7.6–8.6) and morphology (0.595 vs 0.650),
and vanish (HR) or invert (morphology, conditioning gain) with 2–4 MeanFlow steps: iMF-4 (7.02 bpm, 0.719, gain 6.59) is better than
OT-CFM-50 on every physiological metric at 8 % of its cost. RMSE/MAE are *worse* for iMF-1 than for the collapsed OT-CFM-1
(0.443 vs 0.304) — exactly the pattern the pre-registration warned about: pointwise metrics reward the beat-free mean.

## Recommended next research question
The one-step objective works on this backbone; the open question is now the **ceiling shared by both objectives**: HR error
regresses to the training mean under high-HR activity and beat-level timing is not learnable on raw PPG-DaLiA. Recommended next
experiments (all pre-registered, no new architecture yet): (1) seeds 43/44 and the 5-fold subject protocol P1 for A0-b and A2 to
establish variance; (2) a beat-level protocol with an explicit, documented PPG↔ECG re-synchronisation step (or a dataset with
synchronised beats) to make R-peak F1 / RR metrics meaningful and to test whether one-step models preserve beat *timing*;
(3) only then, if beat timing/high-HR tracking is the bottleneck, consider the conditioning path — as a separate, pre-registered
architecture question, not as part of the objective comparison.
