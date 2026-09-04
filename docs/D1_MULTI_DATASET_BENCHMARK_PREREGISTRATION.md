# D1 — Multi-Dataset Benchmark of the Frozen Methodology

**Status: PREREGISTRATION. Frozen on commit. Not edited post-hoc.**
Written 2026-09-04, BEFORE any D1 weight update and BEFORE any D1 real-data metric.

## 1. Question

What is the performance of **our existing methodology, unchanged**, when it is trained and evaluated
separately on each of five PPG→ECG corpora, reported over the full union of evaluation metrics used
by the three reference papers (KANFlow, PENGUIN, PPGFlowECG)?

This is a **descriptive benchmark**, not a method proposal. Nothing about the model, the objective,
the optimiser, the preprocessing, or the sampler is being changed or searched.

## 2. What is explicitly NOT being done

- **NO** architecture change, no attention/adapter/conditioning change, no new loss term.
- **NO** optimiser / LR / batch-size / seed change. Hyperparameters are **identical across corpora**;
  that identity is the point of the benchmark.
- **NO** preprocessing change. The frozen `PPG_KW` (0.5–4 Hz) and `ECG_KW` (0.5 Hz HP) are used
  unmodified, exactly as in every prior arm. The band ablation recommended by
  `docs/PREPROCESSING_CONVENTIONS_SURVEY.md` §8.1 is a **separate future experiment**, not this one.
- **NO** result-dependent tuning. No metric may be inspected before all runs finish.
- **NO** SOTA claim. No head-to-head "we beat X" claim (see §11).
- **NO** modification of `external/PENGUIN` (@6cd70cd) or `external/iMeanFlow` (@bf60cd7c).
- **NO** C2 training — C2 remains deferred.
- **NO** change to any frozen artefact: A4 checkpoint md5 `31c042d291052fbb6dc15263ad316be2`, the
  frozen B generator `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt`, the R1 probe, the
  O2b integer-grid operator, the E2 contract `artifacts/e2_evaluation_contract/contract_v1.json`
  (sha256 `06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115`).

## 3. Frozen methodology under test

Arm **B** of C1, unchanged:

| | value | source |
|---|---|---|
| objective | improved MeanFlow (iMeanFlow) | `src/ppg2ecg/flow/imeanflow.py` |
| backbone | upstream PENGUIN Flow-SSM (S5), unmodified | `external/PENGUIN` @6cd70cd |
| conditioning | `cond_mode="h_only"`, `h_scale=1.0` | C1 arm B |
| params | 4,568,707 | |
| h_dim / blocks / ssm_ratio / mlp_ratio | 128 / 4 / 2.0 / 2.0 | |
| optimiser | AdamW, lr 1e-3, wd 0.01, batch 64, micro-batch 32 | |
| epochs / patience / min_delta | 300 / 20 / 1e-4 | |
| selection | fixed-bank iMF validation MSE, 4 banks, `bank_seed=1000` | never a generation metric |
| p_mean / p_std / data_proportion | −0.4 / 1.0 / 0.5 | |
| norm_p / norm_eps / jvp_mode | 1.0 / 0.01 / forward | |
| seed | 42 | |
| driver | `python -m ppg2ecg.training.train_a2` via `scripts/d1_train.py` | |

Every corpus gets this **identical** argv. Nothing is tuned per corpus.

## 4. Corpora

| key | dataset | native PPG / ECG | subjects | windows (8 s @128 Hz) | status |
|---|---|---|---|---|---|
| `wildppg` | WildPPG | 128 / 128 Hz | 16 (**14 eligible**) | 342,471 eligible | processed |
| `dalia` | PPG-DaLiA | 64 / 700 Hz | 15 | ~16.4 k | convert from `upstream_8s` pickles |
| `bidmc` | BIDMC | 125 / 125 Hz | 53 | ~3.1 k | build |
| `capnobase` | CapnoBase | 300 / 300 Hz | 42 | ~2.5 k | build |
| `vitaldb` | VitalDB | 500 / 500 Hz | 6,156 cases → **subset** | target ~390 k | build |
| `mimicbp` | MIMIC-BP | — | 1,524 | 137,160 | **EXCLUDED** — target is ABP, not ECG |

**WildPPG subject rule (standing, absolute):** subjects `kjd` and `ssx` are **never loaded**, for any
purpose, in any D1 stage. The D1 WildPPG split is drawn from the remaining **14** subjects only. This
makes D1's WildPPG split *different* from the A4 split, so the D1 WildPPG run is a **new training
run**; the frozen C1 arm-B checkpoint is untouched and is reported separately as historical context,
not as a D1 row.

**VitalDB subset rule (frozen here, deterministic, no randomness):**
1. sort all case files by `caseid` ascending;
2. keep cases where both `ECG_II` and `PLETH` exist, finite fraction ≥ 0.95, duration ≥ 600 s;
3. take eligible cases in ascending `caseid` order until the cumulative `floor(N/(500·8))` window count
   first reaches **390,000** (chosen to match WildPPG's corpus size so the cross-corpus comparison is
   not confounded by an order-of-magnitude size difference);
4. the selected caseid list is recorded verbatim in the corpus MANIFEST.
Caveat recorded now: VitalDB's 6,388 cases map to 6,090 `subjectid`s, so caseid-level splitting is
marginally weaker than subject-level. We accept this and state it in the report.

**CapnoBase artifact policy:** two corpora are built from the same source.
`capnobase_8s` applies **no** artifact screening (matching every other corpus in this project and the
CardioGAN/RDDM/PENGUIN lineage, which screen nothing). `capnobase_8s_clean` drops any window
overlapping a shipped expert artifact interval. **The primary D1 row is the unscreened corpus**;
the screened corpus is a secondary disclosure row. Recorded fact to be reproduced by the builder: the
shipped peak labels are *not* pre-screened against the artifact intervals (≈177 PPG / 7 ECG peaks fall
inside them), so any consumer must intersect the two label sets itself.

## 5. Splits

Subject-level, seed 42, identical rule for every corpus:
sort subject ids → shuffle with `numpy.random.Generator(PCG64(42))` → partition 70/15/15
(train gets the ceiling), asserting **≥ 1 val and ≥ 1 test subject**. One manifest per corpus at
`data/manifests/split_d1_<key>_seed42.json`, written and committed **before** training.

No window from a val or test subject enters training. `assert_no_forbidden_subjects` raises if `kjd`
or `ssx` appears in any train or val list.

## 6. Evaluation

**Sampling budget.** Every corpus is evaluated at **NFE ∈ {1, 2, 4, 10, 25, 50}**. NFE 1 is our
headline (one-step is the method's claim); NFE 50 is PENGUIN's published operating point and exists so
that a budget-matched comparison is possible. Generation wall-clock and samples/s are recorded per NFE.

**Metrics.** The union of (a) every metric reported by KANFlow, PENGUIN and PPGFlowECG and (b) this
repo's existing suite. Concretely, at minimum:

- waveform: MAE, MSE, RMSE, PCC, cosine similarity, SNR (dB)
- **PRD, both published variants**, never collapsed:
  `prd_raw = 100·√(Σ(p−t)²/Σt²)` and `prd_meansub = 100·√(Σ(p−t)²/Σ(t−t̄)²)`
- **Fréchet distance, both definitions**, never conflated (the survey established that CardioGAN's
  *discrete* Fréchet and RDDM's *FID-style* Fréchet share a name and are not comparable). The FID-style
  number is reported with an explicit note that, absent RDDM's own feature network, it is **not**
  comparable to their published value.
- DTW (Sakoe–Chiba banded)
- rhythm: R-peak precision / recall / F1 at tolerances **25 / 50 / 100 ms**, HR MAE and HR RMSE (bpm),
  RR MAE (ms), QRS-width error (ms), beat-aligned morphology correlation
- region-disentangled (RDDM style): RMSE inside a ±50 ms QRS region vs its complement
- upstream parity: `penguin_hr_error` (unmodified upstream code path)
- efficiency: NFE, latency, samples/s, peak memory

Any metric named in a paper that cannot be implemented is listed explicitly as not implemented, with
the reason. No metric is silently dropped.

**Aggregation.** The **subject-macro mean** (mean over subjects of each subject's window mean) is
primary. The pooled window mean is reported as a separate secondary column. The two are never
conflated in one cell.

**Uncertainty.** Subject-clustered bootstrap, **2,000 replicates, seed 20260904** (this project's
current bootstrap seed). Reported as `mean [95% CI]`.

**Detector.** R peaks via the repo's existing `ppg2ecg.evaluation.rpeaks.detect_rpeaks` with the
`neurokit` detector, unchanged, on both reference and generated signals.

## 7. Analysis plan, fixed in advance

No hypothesis test is preregistered because D1 makes no comparative claim. The following are
**descriptive** and fixed before seeing results:

1. **Headline table**: rows = corpora, columns = every metric, at NFE 1 and NFE 50.
2. **NFE sweep table**: rows = (corpus × NFE).
3. **Corpus descriptors**: subjects/windows per split, native fs, disclosure flags.
4. **Published-number context table**: our number beside any published number for the same dataset,
   each row carrying a mandatory non-comparability note (window length, PPG band, NFE, parameter
   count, split). No row may be presented as like-for-like.
5. **Corpus-size confound panel**: metric vs. training-window count. Corpus size varies by two orders
   of magnitude across D1 (2.5 k → 390 k) and is a confound by construction. **No causal claim about
   dataset difficulty may be made from D1.**

Any corpus whose training split has fewer than 2,000 windows is flagged `small_corpus` in its run
metadata and that flag is printed in the report. Hyperparameters are **not** adjusted in response.

## 8. Figures (paper format)

`scripts/d1_figures.py` produces, deterministically, both `.png` (300 dpi) and `.pdf`:

- **FIG 1** qualitative reconstruction grid — rows = datasets, columns = test windows; GT ECG overlaid
  with generated ECG, conditioning PPG beneath, reference R peaks ticked. **Window selection is
  deterministic and stated in the caption**; no cherry-picking.
- **FIG 2** per-dataset headline metric bars with subject-clustered bootstrap CIs.
- **FIG 3** NFE-vs-quality tradeoff with a wall-clock panel; PENGUIN's NFE-50 operating point marked.
- **FIG 4** beat-level failure analysis — HR identity scatter + Bland–Altman + matched-beat timing
  error histogram. This figure exists specifically to show that an HR metric can look good while beat
  placement fails.
- **FIG 5** per-subject distributions within each dataset.
- **FIG 6** corpus-size confound panel.

If a corpus's evaluation output is missing, its panel is **skipped with a printed warning** — never
fabricated, never silently dropped.

## 9. Execution

`scripts/d1_run_all.sh` runs everything inside a detached **tmux** session `d1bench`, so the run
survives disconnection. Trainings run **strictly in sequence** (one 32 GB GPU). The script is
idempotent (skips corpora with a `TRAINING_DONE` / `EVAL_DONE` marker), continues past a failed
corpus rather than aborting, and maintains `outputs/d1_bench/PROGRESS.log` and
`outputs/d1_bench/STATUS.json`.

Expected cost: ~3.6 h per large corpus (the C1 arm-B reference run was 12,911 s / 66 epochs on
389 k windows), less for the small ones; ~12–18 h total including evaluation.

## 10. Artefacts

Committed: this preregistration, the loaders (`src/ppg2ecg/data/{bidmc,capnobase,vitaldb}.py`), the
builders, `src/ppg2ecg/evaluation/paper_metrics.py`, `scripts/d1_{common,train,evaluate,figures,report}.py`,
`scripts/d1_run_all.sh`, the tests, and the split manifests.
Never committed: processed corpora, checkpoints, generated waveforms, figures, raw data.

## 11. Claim boundary

D1 establishes **only** the following: the performance of one fixed methodology, trained separately
under one fixed recipe, on five corpora, measured over a stated metric set at stated inference budgets.

D1 does **not** establish, and no D1 output may be worded to suggest:
- that our method is better or worse than any published method — the preprocessing (PPG 0.5–4 Hz,
  8 s windows), the parameter count (4.57 M vs PENGUIN's 62.53 M), the split, and the inference budget
  all differ from every published protocol, and `docs/PREPROCESSING_CONVENTIONS_SURVEY.md` documents
  that no cross-paper number in this field is comparable without restating the pipeline;
- that one dataset is intrinsically harder than another — corpus size varies 150× and is confounded;
- anything about generalisation across corpora — every model is trained and tested within one corpus.

The word "confidence-calibrated" is not used anywhere in D1 outputs. Any oracle-conditioned arm, if
one is ever added, is labelled "(GT-R leakage; diagnostic only)".

## 12. Deviations

Any deviation from this document is recorded in a dated `## Deviations` section appended to the D1
**report**, never by editing this preregistration.
