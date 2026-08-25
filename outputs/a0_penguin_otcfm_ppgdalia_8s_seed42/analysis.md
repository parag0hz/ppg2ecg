## Morphology analysis
**What degrades first (50 → 4 NFE).** Rate-level metrics are essentially flat down to 4 NFE (HR error 10.99 → 11.59 → 12.25 →
11.17 bpm; the +1.27 bpm at 10 NFE nominally crosses the +1.0 bpm margin but is non-monotonic and its 95 % CI (11.6–12.9) barely
separates from the 50-NFE CI (10.4–11.6) — single noise seed, treat as borderline; RR MAE 34.9 → 34.6 ms; QRS-width proxy
33.7 → 33.9 ms), while the **beat-aligned template correlation is the first primary metric to cross its margin decisively**:
0.662 (50) → 0.664 (20) → 0.639 (10) → **0.475 (4 NFE, −0.19, CI 0.463–0.486)**.
Qualitatively (`figures/example_*.png`), the 4-NFE traces keep sharp QRS spikes at the right rate but the baseline between beats
becomes noisy (HF-energy ratio 0.32 → 0.42) — residual noise, not yet averaging.

**Collapse at ≤ 2 NFE.** Heun 1 step (2 NFE) and Euler 1 step (1 NFE) fail every rate/morphology criterion: HR error 36–39 bpm,
template correlation 0.23 / 0.14, predicted beats per window 4.9 / 4.3 vs 10.5 true (spurious detections, not real beats),
conditioning gain of the PPG-shuffle test ≈ 0 (−0.4 / +0.4 bpm vs 3.8 bpm at 50 NFE).

**1-NFE Euler = conditional-mean averaging, plus residual noise.** The 1-NFE output is `x0 + v(x0, t=0) ≈ E[x1 | PPG]`: amplitude
collapses (per-window std 0.046 vs 0.236 for the target and 0.196 at 50 NFE), the waveform is a flat line near −0.3 with small
ripples (`example_*` bottom rows), seed-to-seed diversity vanishes (std 0.062 vs 0.218), and the remaining energy is
high-frequency (HF ratio 0.64 vs 0.32) — i.e. the mean prediction still contains un-cancelled noise. This is why **RMSE improves
at 1 NFE (0.472 → 0.295) while every clinical metric fails**: global waveform error rewards the averaged, beat-free output.
RMSE/MAE/PCC must not be used as quality criteria for this task.

**Conditional fidelity of the baseline itself.** Even at 50 NFE the model tracks heart rate only coarsely: predicted vs
reference HR correlate at r = 0.40, with regression to the training mean (ref HR < 70 bpm: +11 bpm bias; 90–110: −6; > 110:
−28 bpm; stairs/cycling windows have 18–26 bpm error). The PPG-shuffle test confirms the conditioning carries rate information
(3.8 bpm gain) but not much more. **Beat-level alignment is absent at every NFE** (R-peak F1 0.14 at 50 ms, PCC ≈ 0, prediction-to-
target cross-correlation lag uniformly spread over ±0.5 s, no gain from a global shift). The data-level diagnostic
(`figures/dalia_sync.png`, `DATA_PROTOCOL.md` §6) shows the wrist-PPG/chest-ECG streams of PPG-DaLiA are only second-level
synchronised with ~20 ms/min relative drift, so beat-level phase is not a learnable target on the raw windows: the low F1 is a
property of the dataset, not evidence about the model, and F1/PCC/RR-MAE (beat-matched) are reported only for completeness.

## Limitations
- Single seed (42), single test subject (S2, 1025 windows) and single validation subject (S11); early stopping is driven by a
  noisy stochastic validation MAE (patience 10 fired at epoch 21 while the train loss was still falling) — the checkpoint is
  probably under-trained relative to a smoother criterion.
- Window length (8 s) and split are audit-based choices; the paper's held-out subject and window length are unknown, so the
  paper comparison is a feasibility gate, not a reproduction claim.
- Beat-aligned metrics are invalid on raw PPG-DaLiA (device synchronisation, §6 of DATA_PROTOCOL); QRS width is a QS-trough proxy
  on min-max-normalised signals; no absolute amplitude is available.
- Efficiency numbers are for the unmodified upstream S5 implementation (vmap + associative scan, fp32, no compile) on one GPU.
- The A0 loop is ours (mirrors upstream step-for-step, verified bit-exact for sampler/objective); the official upstream
  `train.py` with its glob-order split was not run.

## GO / NO-GO
Pre-registered questions (PREREGISTRATION_V0 §5): **H1 confirmed** — at NFE ≤ 2 (Euler 1, Heun 1) every primary metric that is
interpretable on this dataset fails by a wide margin (HR error +25–28 bpm, template correlation −0.43/−0.53, QRS proxy within
margin only because no real beats are detected); the first failure appears at 4 NFE (template correlation −0.19). The
hypothesis that "OT-CFM is already fine at 1–2 NFE" is **rejected**: the NO-GO condition for the one-step line does not hold.

Failure mode (H2 vs H2′): at 1 NFE the output is the conditional mean — beat morphology, rhythm and conditioning are lost
together (amplitude collapse, zero diversity, conditioning gain → 0). This is closer to **H2 (averaging/blur)** than to a pure
conditioning failure, but the distinction is moot because the mean waveform has no beats; the primary target for a one-step
objective is therefore "produce a beat-bearing, rate-correct waveform in one evaluation", measured by HR error, template
correlation on matched beats, amplitude (per-window std ratio) and the PPG-shuffle conditioning gain.

Verdict: **GO for the objective-swap line (A1′/A2)**, with two caveats that must be handled before A2 is meaningful:
1. the baseline's own conditional fidelity is weak (HR r = 0.40, regression to the mean) — a one-step model that merely
   matches 50-NFE OT-CFM inherits a weak ceiling; the comparison must be relative (recover the 50-NFE level at 1 NFE), not absolute;
2. beat-level metrics require a re-synchronised evaluation protocol on DaLiA (or a beat-synchronised dataset), otherwise
   "morphology preserved" can only be claimed at the template/rate level.

## Recommended next experiment
Not a new method yet. In order:
1. **Ceiling & variance of the baseline (cheap, no new code):** re-run A0 with a smoother model-selection criterion
   (val CFM loss or val MAE averaged over ≥ 3 noise draws) and seeds 43/44, to know whether HR r = 0.40 / 11 bpm is the
   OT-CFM+S5 ceiling on DaLiA or an early-stopping artefact. Pre-register as A0-b before running.
2. **Re-synchronised beat-level protocol (data work):** estimate the per-segment PPG↔ECG lag (cross-correlation of the PPG
   derivative with the R-peak train, smoothed over minutes) and re-cut windows; verify on the dataset R-peaks that the
   pulse-arrival delay becomes physiologically stable (≈ 250–400 ms, IQR < 60 ms in rest); re-evaluate A0 on the re-synchronised
   test windows. Only then are R-peak F1 / RR MAE meaningful.
3. **A1′ control on the same protocol**, then **A2 = iMeanFlow on the identical S5 backbone at 1 NFE**, judged against the 50-NFE
   reference with the margins of §5 on HR error, template correlation, amplitude ratio and conditioning gain.
