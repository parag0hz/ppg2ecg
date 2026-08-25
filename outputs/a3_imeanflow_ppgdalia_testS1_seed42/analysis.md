## Conditional fidelity
On S1 the OT-CFM 50-NFE reference has a larger PPG-shuffle gain than on S2 (8.77 vs 5.69 bpm): S1's recording has more HR
variation for the conditioning to track. iMeanFlow at 1 NFE keeps 4.78 bpm of it (53 % of the gap; OT-CFM-1: 0.28), and 6.1–6.2 bpm
with 2–4 steps. So the one-step output still depends on the given PPG, but less tightly than on S2 (78 %); conditioning is the
metric with the smallest recovery on both subjects.

## Qualitative examples
Pre-registered windows (S1, 50-NFE HR-error quantiles): OT-CFM-1 is again a flat line; iMeanFlow-1 produces beat-bearing traces at
roughly the right rate (beats/reference 1.03) but with visibly smaller and less uniform R-peak amplitudes than on S2 (amplitude
ratio 0.71 vs 0.90) and occasional extra small spikes; the inter-beat baseline is close to the target's HF content (HF ratio 0.30 vs
target ≈ 0.29). OT-CFM-4 is the noise-dominated regime seen on S2.

## Failure taxonomy
- F1 conditional-mean collapse: absent (amp 0.71, seed std 0.213 ≈ OT-CFM-50's 0.204).
- F2 QRS smoothing: minor (template corr 0.581, CI 0.567–0.595 vs 0.683 at 50 NFE; recovered to 0.635 with 4 steps).
- F3 amplitude collapse: absent, but amplitude is the weakest morphology term on S1 (0.71; 2–4 steps 0.74–0.75).
- F4 conditioning neglect: absent (gain 4.78; recovery 0.53 — lowest of the four).
- F5 unstable training: absent (36 epochs, best 16 — the deterministic criterion plateaued earlier than on S2's split; train MSE 0.18).
- F6′ spurious spikes: present in a minority of windows, as on S2.

## Limitations
- Same as A2 (single seed, single test subject per split, baseline optimiser, boundary v_θ, beat-level metrics not interpretable on
  raw PPG-DaLiA); S11 shared as validation subject with A2 (subject-robustness, not a fully independent confirmation).
- The iMF run stopped after 36 epochs (best 16) versus 81 (best 61) on the S2 split — the fixed-bank criterion is identical, so this
  reflects the different training set (S2 in, S1 out), not a protocol change; the 50-NFE OT-CFM reference also trained longer here (114 epochs).

## Verdict rationale
iMeanFlow-1 beats OT-CFM-1 on all four physiological metrics with no negative recovery (HR 0.86, morphology 0.80, amplitude 0.76,
conditioning 0.53; beats 1.03) → **REPLICATED** under the frozen rule (and SUCCESS under the A2 rule). The ordering A (OT-1 ≪ OT-50),
B (iMF-1 ≫ OT-1) and C (iMF-1 approaches OT-50: residual +3.8 bpm HR, −0.10 template corr) holds on the new subject. The pointwise-error
inversion replicates: OT-CFM-1 has the best RMSE (0.347) while its physiology is destroyed.

## Recommended next research question
Unchanged from A2: variance (seeds, 5-fold) and a re-synchronised beat-level protocol; the smaller amplitude/conditioning recovery on S1
suggests adding the amplitude ratio and the shuffle gain as *primary* claims in any follow-up rather than HR alone.
