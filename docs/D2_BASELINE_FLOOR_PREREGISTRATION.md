# D2 — Baseline Floor and Conditioning Controls

**Status: PREREGISTRATION. Frozen on commit. Not edited post-hoc.**
Written 2026-09-06, BEFORE any D2 metric is computed on real data.

## 1. Question

D1 reported R-peak F1 from 0.150 to 0.752 across five corpora with no reference point of any kind.
**Nothing in D1 says whether those numbers are good.** D2 supplies the missing reference: a set of
trivial predictors that use only information cheaply available from the PPG, scored on the *identical*
test population with the *identical* metric suite, so every D1 cell gains a floor beside it.

Second, D1 could not distinguish "the model reconstructs this ECG" from "the model emits a plausible
ECG at roughly the right rate". D2 settles that with a mismatched-conditioning control.

## 2. What is explicitly NOT being done

- **NO training. NO weight update. No optimiser touches any parameter.** D2 is inference and arithmetic.
- **NO architecture, loss, preprocessing, split, or hyperparameter change.** The D1 corpora, splits,
  checkpoints, metric suite, aggregation and bootstrap seed are reused byte-for-byte.
- **NO re-tuning of anything in response to a D2 number.**
- **NO** modification of `external/PENGUIN` (@6cd70cd) or `external/iMeanFlow` (@bf60cd7c), of the D1
  checkpoints, of the frozen B generator, of the A4 checkpoint (md5 `31c042d291052fbb6dc15263ad316be2`),
  or of the E2 contract.
- **NO** C2 work. C2 remains deferred.
- **NO SOTA claim.** D2 compares our method to *trivial predictors*, not to any published method.

## 3. Evaluation population — identical to D1, by construction

For each corpus: the same test subjects from `data/manifests/split_d1_<key>_seed42.json`, the same
`d1_common.capped_indices` selection with the same per-corpus cap (0 for dalia/bidmc/capnobase,
1024/subject for wildppg/vitaldb). D2 asserts, per corpus, that the loaded `(subject, window_index)`
sequence equals D1's `per_window_metrics.csv` sequence; a mismatch is a hard failure, not a warning.

Metrics: `ppg2ecg.evaluation.paper_metrics.paper_metric_table` + `evaluate_windows`, unchanged.
Aggregation: subject-macro mean primary, pooled window mean secondary, never merged.
Uncertainty: subject-clustered bootstrap, 2,000 replicates, **seed 20260904** (as D1).

## 4. The baselines

All fitted quantities come from **TRAIN subjects of that corpus only**. No test window, and no test
subject's ECG, ever enters a fitted parameter. Each baseline is deterministic.

| id | name | what it uses | leakage |
|---|---|---|---|
| **B0** | wrong-window null | that subject's own real ECG from a *different* window | uses real test ECG — **diagnostic only, labelled as such** |
| **B1** | PPG-rate + template QRS | PPG peak positions + a train-fitted QRS template and PAT offset | none |
| **B2** | train-mean beat at PPG peaks | as B1 but the template is the train-mean beat rather than a synthetic QRS | none |
| **B3** | train-mean waveform | nothing from this window at all | none |
| **B4** | mismatched-PPG control | **our D1 model**, conditioned on a *different* window's PPG from the same subject | none |
| **B5** | GT-timing template | the train template placed at the TRUE R peaks | **(GT-R leakage; diagnostic only)** |

Frozen definitions:

- **PPG peaks**: `neurokit2.ppg_findpeaks(method="elgendi")` at 128 Hz on the stored (already
  preprocessed) PPG. Detector and method fixed here; no alternative is tried.
- **PAT offset**: one integer sample offset per corpus, chosen on TRAIN windows as the value in
  [0, 128] (0–1000 ms) maximising the ±50 ms match rate between (PPG peak + offset) and the train GT
  R peaks. Fitted once, before any test window is loaded, and recorded in the manifest.
- **QRS template (B1)**: the train-mean beat restricted to ±60 ms around the R peak, zero outside;
  amplitude scaled so the template's peak equals the train-mean R-peak amplitude.
- **Train-mean beat (B2)**: the mean of all train beats over the own-centre support `[-10, +15]`
  samples used elsewhere in this project, extended to ±0.4 s, placed at each predicted position.
- **B3**: the per-corpus train-mean waveform, one fixed 1024-sample vector, emitted for every window.
- **B0**: for each test window of subject *s*, the GT of the *next* stored window of *s* (cyclic
  derangement over that subject's evaluated rows). It answers: what does a predictor score if it
  knows this person's ECG perfectly but not the timing? It is a **strong** control, not a weak one.
- **B4**: our D1 checkpoint, NFE 1, conditioned on the PPG of the cyclically-next window of the same
  subject, with the *same* noise draw as D1 used for that row. Everything except which PPG goes in
  is held fixed.
- **B5**: the B1 template placed at the reference R peaks of the true window. It is an upper bound
  for any template method and carries the mandatory label **(GT-R leakage; diagnostic only)**.

## 5. Directional predictions, fixed before any D2 number is computed

Recorded so that the outcome cannot be reinterpreted afterwards. These are predictions, not gates —
D2 has no pass/fail verdict; it produces a reference table.

1. **P1.** Our D1 model beats B1 and B2 on R-peak F1@50 ms on **VitalDB**, by a margin whose 95 %
   subject-clustered bootstrap CI excludes 0.
2. **P2.** On **PPG-DaLiA** and **CapnoBase**, our D1 model does **not** beat B1 on R-peak F1@50 ms
   (the CI of the difference includes 0, or the difference is negative).
3. **P3.** B4 (mismatched PPG) scores within the CI of our matched-PPG D1 result on at least three of
   the five corpora — i.e. on most corpora, which PPG is fed in does not matter.
4. **P4.** On **VitalDB**, B4 is clearly worse than matched (CI of the difference excludes 0).
5. **P5.** B0 exceeds our model's PCC on **CapnoBase** (already observed in the 2026-09-05 spot check
   at 0.070 vs 0.053; recorded here as a prediction for the full population, which is larger).

If P1 and P4 hold while P2 and P3 hold, the reading is: the method conditions on the PPG on VitalDB
and essentially not elsewhere. If P3 fails everywhere, the conditioning is real and the D1 failure is
about precision rather than about conditioning. Both outcomes are informative and both are reported.

## 6. Analysis plan

`outputs/d2_baselines/` gets, per corpus: `per_window_metrics.csv`, `per_subject_metrics.csv`,
`summary_by_arm.csv` (arm × metric, subject-macro mean with CI), and `d2_meta.json` (checkpoint
sha256, corpus identity hash, split manifest sha256, PAT offset, template hashes, seeds, population
assertion result).

The headline artefact is **one table**: rows = corpus × arm (ours at NFE 1, B0–B5), columns = the
D1 headline metrics. Plus a paired table of **differences** (ours − baseline) with subject-clustered
bootstrap CIs, since the paired comparison is the actual question.

Figures: **FIG 10** — per corpus, R-peak F1@50 ms of every arm as a bar with CI, with our model
marked; **FIG 11** — matched vs mismatched PPG (B4) scatter per window, per corpus.

## 7. Claim boundary

D2 establishes only where our D1 method stands relative to trivial, explicitly-defined predictors on
the same population. It does **not** establish anything about published methods; it does not rank
dataset difficulty (D1's 150× corpus-size confound is unchanged); and B0 and B5 use test-set ground
truth and are therefore diagnostics that bound what is achievable, never claims about our method's
performance. Every table and figure carries the arm labels including the leakage labels.

## 8. Deviations

Recorded in a dated section of the D2 **report**, never by editing this document.
