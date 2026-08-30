# X4-0 Pre-registration — iMeanFlow event reliability / source-condition / interval diagnostic

Written 2026-08-31, **before any X4-0 real-data metric is computed**. Frozen by the commit that introduces it.
**NO TRAINING.** Frozen-checkpoint inference and analysis only; no checkpoint is created or modified, no historical artefact is
overwritten. Starting state: `main` = `origin/main` = `8ad5f40`, clean; submodules `external/PENGUIN` @ `6cd70cd`,
`external/iMeanFlow` @ `bf60cd7`; one NVIDIA RTX 5090 (32,607 MiB), torch 2.11.0+cu130 (verified 2026-08-31).

## 1. Question and hypotheses

iMeanFlow already produces ECG-like and sufficiently sharp waveforms at 1 NFE. Why does **beat-level physiological reliability**
remain weaker than desired, and what is the highest-value next intervention? Three candidate mechanisms:

- **H1 — few-step / local refinement.** Additional evaluations improve event correspondence because later evaluations act on
  increasingly ECG-like states.
- **H2 — source-sensitive event organization.** Gaussian source variation may change event *count*, *presence* or *timing*, not only
  waveform morphology.
- **H3 — large-interval sensitivity.** The `h = 1` query used by 1-NFE inference is an extreme boundary query with zero exact
  training probability and vanishing near-boundary training mass.

**Critical distinction, frozen here:** existing source-diversity results (A4 iMF seed-to-seed pairwise waveform correlation
0.025–0.040) establish only that *waveform realization* is strongly source-sensitive. They do **not** establish that *event
organization* is source-sensitive. X4-0 measures event identity directly.

## 2. Claim discipline (binding on the report)

Forbidden: "iMF-1 lacks sharpness" (contradicted by X0); "57 % of QRS energy is in the wrong location" (that arithmetic mixed a
pooled-variance ratio with a median of per-beat ratios and is retracted); "h = 1 lies outside the support of the training
distribution" — use instead **"exact h = 1 has zero training probability and is an extreme boundary query; near-1 intervals have
vanishingly small training mass"**; "existing waveform diversity proves event identity is source-controlled"; "source perturbation ≈
PPG perturbation implies the model does not use PPG" (the perturbation magnitudes are not causally commensurate); "the Gaussian
source is wrong"; "PPG is ignored"; "h mismatch causes one-step failure"; "STFT will fix the problem"; "event conditioning is proven
necessary"; "OT-50 is an oracle". Preferred framing: *"iMF-1 already produces sharp ECG-like structure, while beat-level event
correspondence remains imperfect."*

## 3. Data policy and test firewall

Train (not used here): `e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p`. **Development validation: `an0`, `k2s`.**
**Test: `kjd`, `ssx` — NEVER LOADED.** A fail-loud assertion raises if any subject list intersects `{kjd, ssx}`; the provenance
records every subject loaded.

`an0`/`k2s` have already been visually inspected (`docs/X4_0_PREPREREG_VISUAL_AUDIT.md`), so they are **development validation**, not
pristine confirmatory validation. The four previously viewed windows — **(`an0`, 9066), (`an0`, 18138), (`k2s`, 5852),
(`k2s`, 16436)** — are excluded by construction from every X4-0 metric subset.

## 4. Frozen models

| Role | Checkpoint | Round |
|---|---|---:|
| Primary | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` | 45 |
| **Mandatory contextual reference** | `outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt` (Heun 25 steps = **50 NFE**) | 189 |

iMF config (from the checkpoint): `p_mean −0.4, p_std 1.0, data_proportion 0.5, norm_p 1.0, norm_eps 0.01, jvp_mode forward,
cond_mode h_only, h_scale 1.0`; backbone `h_dim 128, ssm_block_num 4, ssm_ratio 2.0, mlp_ratio 2.0`, seed 42. iMF convention:
`t = 1` noise, `t = 0` data; step `z_r = z_t − (t−r)·u_θ(z_t, r, t)`; conditioning on `h = t − r`; uniform N-step inference uses
`h = 1/N`. **OT-CFM-50 is a contextual yardstick only — never an oracle, and the mechanism verdict is not gated on iMF-vs-OT50 F1.**

## 5. Subset selection (deterministic, outcome-independent)

Candidate pool = **all original windows** of each validation subject (`an0` 22,183; `k2s` 27,017) in
`data/processed/wildppg_8s/{subject}.npz`. Ranking key: `SHA256("{salt}|{subject}|{window_index}")`, ascending; take the smallest N
per subject after removing the four previously viewed windows.

| Analysis | Salt | Per subject | Total |
|---|---|---:|---:|
| NFE frontier (§7) | `x4-event-nfe-v2` | 1024 | **2048** |
| Source diagnostic (§9) and condition perturbation (§10) | `x4-event-source-v2` | 256 | **512** |
| Interval stress (§11) | `x4-event-schedule-v2` | 512 | **1024** |

Frozen to `artifacts/x4_0_event_reliability/{nfe,source,schedule}_subset.json`. No outcome-dependent selection anywhere.

## 6. X4-0A — training-interval exposure audit (no waveform data)

`sample_tr()` with the frozen A4 configuration, `n = 2,000,000`, seed **20260830**; `h = t − r`. Report the exact-zero fraction, mean,
median, median conditional on `h > 0`, q75/q90/q95/q99/q99.9, the maximum observed `h`, and
`P(h ≥ x)` for `x ∈ {0.0625, 0.125, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95}`, with inference markers at `h = 1/NFE` for
NFE ∈ {1, 2, 4, 8, 16}. **Interpret only as training-interval exposure.** No performance inference from this distribution alone; it
is not an impossibility result.

## 7. X4-0B — iMF NFE event-quality frontier, and the mandatory OT-50 reference

Frozen iMF at **NFE ∈ {1, 2, 4, 8, 16, 25, 50}**, Gaussian source seeds **{0, 1, 2, 3}**, on the 2048-window subset.
**Matching requirement:** for a given window and seed, the *identical* initial Gaussian tensor is reused at every NFE; only the number
of evaluations changes. Waveforms are never averaged across seeds before scoring — individual realizations are scored first.
On the **exact same windows and the same four source seeds**, frozen OT-CFM is evaluated with Heun-25 (**actual NFE = 50**, asserted).

## 8. Metrics — three separate blocks, all reusing X0 semantics exactly

Verified reusable before writing this document: `ppg2ecg.evaluation.rpeaks.detect_rpeaks` / `match_rpeaks`,
`alignment_diagnostics.event_timing` (`tol_ms = 50`), `beat_level_analysis` (oracle ±150 ms), `segment_stats`, and the constants
`FS = 128`, `QRS_HALF_MS = 100`, `HF_CUT_HZ = 15`, `LOCAL_MAX_SHIFT_MS = 150`. **If exact reuse had been impossible the protocol
required STOP; it is possible, so no metric is reimplemented.**

- **Block A — event reliability:** precision, recall, F1, `n_pred_beats`, `n_gt_beats`, beats/reference, missing fraction, spurious
  fraction, RR MAE, matched-peak timing MAE, timing bias, timing SD.
- **Block B — detector-independent / oracle (diagnosis only, never deployable):** oracle-absent fraction, same-coordinate beat
  correlation, oracle ±150 ms beat correlation, oracle p2p / QRS-energy / slope retention.
- **Block C — waveform structure:** morphology correlation, RMSE, MAE, amplitude ratio, global QRS-energy ratio, max-slope ratio,
  HF-energy ratio.

Blocks are reported and interpreted separately. Morphology alone is never read as event reliability; F1 alone is never read as
waveform quality; QRS energy alone is never read as successful reconstruction.

**Uncertainty.** `an0` and `k2s` are always reported separately and pooled with equal subject weight. Paired uncertainty uses a
**subject-stratified window bootstrap**, 2000 resamples, seed 20260830 (resample windows within each subject, then average the two
subject estimates equally). These are **descriptive paired bootstrap intervals**, not population-confirmatory inference — two
development subjects cannot support that.

**NFE improvement analysis.** Paired differences versus NFE = 1 on identical (window, PPG, source), with attention to NFE 4/8/16 and
to whether morphology and event reliability saturate at different rates.

**FEW-STEP SATURATION flag (resource heuristic).** Raised if NFE 8 **or** 16 is simultaneously within 0.03 absolute of iMF-50 on
morphology **and** on oracle beat correlation, **and** (F1 within 0.03 of iMF-50 **or** oracle-absent ≤ iMF-50 + 0.05), **and**
spurious ≤ iMF-50 + 0.05, **and** RR MAE ≤ iMF-50 + 3 ms.

## 9. X4-0C — source-sensitive event organization (512 windows, seeds 0–31, NFE ∈ {1, 4, 8, 16})

The question is **not** whether waveforms change (they do). It is whether **event identity, count and timing** change.

- **A. Beat-count variability** across the 32 seeds per window: mean, SD, min, max of predicted beat count.
- **B. Predicted-to-predicted event consistency:** for every seed pair, match predicted peak train A against predicted peak train B
  with the frozen 50 ms one-to-one matcher (no GT). Report seed-pair precision / recall / **F1**; per window the median over pairs.
- **C. GT-anchored event presence:** for every GT beat and every source, is there a predicted event within ±150 ms?
  `detection_probability = matched seeds / 32`; report its distribution across GT beats.
- **D. GT-anchored timing variability:** timing SD across seeds, **restricted to GT beats detected in ≥ 16/32 sources**. This subset is
  biased toward consistently detectable beats and is reported as **"timing variability conditional on sufficient detection"**, always
  alongside the detection-probability distribution.
- **E. Across-seed task variability** per window: SD of F1, RR MAE, morphology, oracle beat correlation.

### 9.1 Flags — materiality and NFE-response are separated (frozen)

**(a) SOURCE-SENSITIVE EVENT ORGANIZATION — materiality.** Declared *material* if **at NFE = 1** at least **two** of:

1. median predicted-seed-pair event F1 **< 0.80**
2. median per-window beat-count SD **≥ 0.75** beats
3. median GT-anchored timing SD **≥ 15 ms** (conditional on ≥ 16/32 detections)
4. median per-window F1 SD across sources **≥ 0.05**

**No improvement at NFE 8/16 is required for this determination.**

**(b) NFE response — reported separately.** If a *material* indicator improves by **≥ 30 %** at NFE 8 or 16 versus NFE 1 →
**NFE-RESPONSIVE SOURCE SENSITIVITY**. If improvement is **< 30 %** while materiality persists → **PERSISTENT SOURCE SENSITIVITY**.
Persistent source sensitivity is **not a flag failure**; it is treated as the stronger and more important result.

Allowed wording: *"Low-NFE event organization is materially source-sensitive."* Forbidden: calling it "latent entanglement" as a
proven fact.

## 10. X4-0C2 — mandatory condition perturbation (not optional)

Reuses the project's existing deterministic no-fixed-point PPG permutation (`derangement` in `scripts/eval_a2.py`, numpy
`default_rng`); no alternative shuffle is invented. Same 512 windows, NFE ∈ {1, 8}, baseline source seeds {0, 1, 2, 3}. Three
conditions per baseline sample:

| | PPG | source |
|---|---|---|
| **A** baseline | correct `c_i` | `z_i` |
| **B** source perturbation | correct `c_i` | **different** source `z_j` |
| **C** condition perturbation | shuffled `c_perm(i)` | same `z_i` |

**Frozen source-pair mapping (fixed before any result):** baseline seed 0 → alternative seed 1; baseline seed 2 → alternative
seed 3. (Seeds 1 and 3 are therefore used only as alternatives in this comparison.)

Reported: `event_F1(A, B)` and `event_F1(A, C)` (predicted-vs-predicted, 50 ms matcher), beat-count difference, matched timing
difference, waveform PCC, morphology difference. For shuffled-PPG outputs the prior A2 two-target semantics are retained where
applicable (correct target for the shuffled PPG vs the original/wrong target).

**Interpretation constraint, frozen.** The derangement is a **strong condition-perturbation anchor** which may substitute a PPG with a
different heart rate and rhythm; it is **not** on a calibrated common causal scale with a Gaussian-source change. Source-induced event
disagreement is reported only as **descriptive context relative to that anchor**. No scalar "source/condition importance ratio" is
produced, and no claim of equal causal importance or of PPG being ignored is permitted.

## 11. X4-0D — fixed-NFE interval stress (1024 windows, seeds {0,1,2,3})

Same window, same source and same NFE at every schedule; only the allocation of the interval changes. Sampling runs `t = 1` (noise)
→ `t = 0` (data); the `h` list is in chronological sampler order; `sum(h) = 1`, final `t = 0` and the NFE count are asserted.

| NFE | Uniform | Large interval at NOISE end | Large interval at DATA end |
|---:|---|---|---|
| 4 | `U4 = [.25, .25, .25, .25]` | `LN4 = [.70, .10, .10, .10]` | `LD4 = [.10, .10, .10, .70]` |
| 8 | `U8 = [.125] × 8` | `LN8 = [.50] + [.50/7] × 7` | `LD8 = [.50/7] × 7 + [.50]` |

Metrics: F1, precision, recall, spurious, oracle-absent, oracle beat correlation, morphology, RR MAE; paired comparisons
`LN4 − U4`, `LD4 − U4`, `LN8 − U8`, `LD8 − U8`.

**LARGE-INTERVAL STRESS flag.** Material if LN or LD causes ΔF1 ≤ −0.05 **or** Δoracle beat correlation ≤ −0.05 **or**
Δspurious ≥ +0.10 **or** Δoracle-absent ≥ +0.10, **and** the effect has the same direction on `an0` and `k2s`, **and** the descriptive
paired bootstrap interval excludes zero for at least one event metric.

**Location.** LN and LD similarly harmful → interval magnitude implicated. LD worse → large near-data jump / insufficient final local
refinement. LN worse → noise-end long transformation. Neither differs → schedule sensitivity at `h ≤ 0.7` is not supported.

### 11.1 REQUIRED LIMITATION (stated before results; must also appear in the report)

These schedules **do not reproduce the NFE = 1, h = 1 query**. Maximum tested stress is `h = 0.70` (NFE 4) and `h = 0.50` (NFE 8),
whereas 1-NFE inference uses `h = 1` exactly. Therefore: a **positive** stress result is informative evidence that large-`h` queries
can degrade the frozen iMF; a **null** stress result is **weak** evidence against H3 and does **not** establish that `h = 1` is
harmless. This asymmetry is binding on the interpretation.

## 12. X4-0E — event-matching tolerance calibration (metric calibration, not model evaluation)

GT R-peak trains from the quantitative subset are perturbed with deterministic zero-mean Gaussian timing jitter of SD
**{5, 10, 20, 30, 40, 50} ms** (seed 20260830, peak count preserved), plus deterministic fixed shifts of **±{10, 20, 30, 40, 50} ms**,
then scored against the unperturbed GT with the same frozen 50 ms one-to-one matcher. Report F1, precision, recall.

**Interpretation, frozen.** This is a **timing-only degradation calibration**, never an upper bound and never "the maximum achievable
F1". Because peak count is preserved it isolates *timing* error and says nothing directly about missing/spurious events. The only
permitted inference has the form: *"if realistic timing jitter of 20–30 ms still yields synthetic F1 well above the observed model F1,
then the observed F1 deficit cannot be explained by timing error alone and missing/spurious events materially contribute."*

## 13. Latency

Frozen iMF at NFE ∈ {1, 4, 8, 16, 25, 50} and OT-CFM at NFE 50, batch 64, 20 warm-ups, 100 timed repetitions,
`torch.cuda.synchronize()` around each; data loading excluded. Report median, p10, p90, samples/s.

## 14. Mechanism classification (frozen)

- **CASE A — FEW-STEP SATURATION:** NFE 8/16 nearly reaches iMF-50 in *both* morphology and event reliability.
- **CASE B — LARGE-INTERVAL LIMITED:** fixed-NFE interval stress materially degrades event reliability.
- **CASE C — SOURCE-SENSITIVE EVENT ORGANIZATION:** material source sensitivity of event count / presence / timing, **including the
  persistent case** (materiality that NFE does not substantially relieve).
- **CASE D — LOCAL REFINEMENT EFFECT:** NFE improves event correspondence **while source sensitivity is weak or is largely relieved by
  higher NFE**, and interval stress is weak.
- **CASE E — MIXED:** more than one mechanism supported; recommend the cheapest isolated intervention, do not force a single story.
- **CASE F — NO CLEAR MECHANISM:** reconsider metric semantics, conditional ambiguity, representation or objective before designing a
  new model.

An especially important possible outcome is **A + C**: morphology saturating by NFE 8–16 while event F1 / spurious / source-event
variability remain poor, read as *"additional integration resolves much of the waveform-quality gap but does not fully resolve
event-identity reliability."* **No predicted values are hard-coded here and no outcome is asserted in advance.**

## 15. Next-experiment decision rules (nothing is started in X4-0)

**Event/rhythm-aware iMF** favoured if source changes event identity substantially, event metrics remain the major deficit, and
QRS energy / slope / HF are already adequate (future shape: PPG → event/rhythm encoder → soft event map → iMF conditioning; GT
R-peaks may supervise **training only**; inference must use PPG alone). **Horizon-aware iMF** favoured if fixed-NFE large-step stress
is clearly harmful. **Complex-STFT iMF** favoured only if sharpness is adequate, event/local morphology remains deficient, and neither
source-event sensitivity nor large-`h` sensitivity is the dominant lever — motivation would be phase-aware transient organization, not
adding HF energy; candidate grid `(16,2) (32,4) (64,8)`. **None is implemented or trained in X4-0.**

## 16. Outputs, implementation, tests

`artifacts/x4_0_event_reliability/`: `provenance.json`, `{nfe,source,schedule}_subset.json`, `h_distribution.{csv,json}`,
`nfe_metrics.csv`, `nfe_metrics_by_subject.csv`, `nfe_metrics_by_seed.csv`, `ot50_reference.csv`, `source_event_variability.csv`,
`source_gt_anchor_variability.csv`, `condition_perturbation.csv`, `source_vs_condition.csv`, `interval_stress.csv`,
`event_matching_calibration.csv`, `latency.csv`, `gate_summary.json`, `figures/`. No giant prediction dumps; small deterministic
traces only. Implementation: `src/ppg2ecg/evaluation/event_reliability.py`, `scripts/analyze_x4_0_event_reliability.py`,
`tests/test_x4_0_event_reliability.py`. Historical analysis scripts are not altered.

Required tests (synthetic, run before real metrics): test-subject firewall; `sample_meanflow` uniform parity; requested steps == actual
NFE; identical Gaussian tensor reused across NFE; OT-50 same window/source pairing; deterministic hash subsets; the four pre-viewed
windows excluded; X0 detector / 50 ms matcher / oracle ±150 ms / QRS ±100 ms / HF > 15 Hz parity; derangement parity with `eval_a2.py`
and no fixed points; source perturbation changes only the source; PPG perturbation changes only the PPG; identical predicted peak
trains → event F1 = 1; disjoint trains → 0; known synthetic shifts produce expected matches; GT-anchor detection-probability synthetic
test; the ≥16/32 conditional filter; `U4`/`U8` reproduce the standard samplers; LN/LD `h` sums = 1 and terminate at `t = 0`; same
source reused across schedules; subject-stratified bootstrap preserves subjects; jitter calibration deterministic; CUDA benchmark
synchronises; historical artefacts protected.

## 17. Scope

X4-0 stops after its report. **No** iMF/OT-CFM training, fine-tuning, coupling, C²OT, reflow, distillation, STFT model, event-aware
model or horizon-aware model is implemented or started. No new checkpoint is created. `kjd`/`ssx` are never loaded.
