# PPGFlowECG external fixed-budget validation — preregistration

Frozen and pushed **before any generated sample is scored against a reference** (no HR, error, F1 or waveform metric of
any PPGFlowECG output on VitalDB has been computed). Data rule: `docs/PPGFLOWECG_VITALDB_DATA_PREREGISTRATION.md`
(`fc99f76`). Head at freeze: `fc99f76`.

**Question.** With the authors' released checkpoint, no retraining and no change to the learned model, does reallocating
a fixed inference budget from per-sample ODE depth to several stochastic samples improve a downstream functional (HR)?
Same checkpoint · no retraining · no new sampler · no re-spacing · no loss / architecture change · no noise injection.

**Checkpoint wording (used everywhere):** authors' officially released checkpoint; training provenance is strongly
implied to be MCMED by the released training configuration but is not explicitly documented for the checkpoint itself.

## 1. What is fixed before this document (steps 1–3, all in `artifacts/ppgflowecg_external/`)
- Source: `external/PPGFlowECG` @ `56b2cd2` (unmodified; `git status` clean). Checkpoints `checkpoint-10.pt`
  (sha256 `50f1af67…ecdb`) and `VAE-iter-40000.pth` (`c186633a…d7d6d`), `checkpoint_hashes.txt`.
- Official load path (`Trainer(config) → Trainer.load(10)`): strict `load_state_dict`, **no missing or unexpected keys**
  (flow model, EMA, VAE encoder / decoders); checkpoint step 10,000; sampler network = EMA model, as released. `audit.json`.
- NFE (forward hooks, `nfe_counts.json`): per sampling call, vector-field calls = **S** exactly for S = 5, 10, 15, 20, 25;
  fixed overhead 1 × ECG-encoder, 1 × PPG-encoder, 1 × ECG-decoder call.
- Released-code finding: `Trainer.sample_shift(..., sampling_steps)` **does not forward** `sampling_steps`; called with 5
  it runs 10 steps. Our driver therefore passes S to the model's own `RectifiedFlow.sample_shift(num_steps=S)` (the same
  Euler loop, the argument the paper's §4.5 varies). At S = 10 the driver's output is **bitwise identical** to the
  released `Trainer.sample_shift` for the same seed.
- Stochasticity sources (official, nothing added): PPG-encoder posterior noise (the condition) and the Euler initial noise;
  the ECG-encoder call consumes noise but its output is used only for its shape (reference ECG replaced by zeros →
  bitwise-identical samples, `smoke_test.json`).
- Smoke test (16 windows, `smoke_test.json`): finite, [16, 1280], same seed → identical, different seeds → different.
- Data (`data_build.json`): **18,525 windows / 1,156 patients / 1,224 cases** (1,018 anchors excluded: 1,012 span past the
  record end, 6 non-finite; no overlapping spans).
- Latency (`latency.json`, RTX 5090, batch 1, full call): 20.0 / 30.3 / 40.6 / 51.0 / 61.3 ms at S = 5 / 10 / 15 / 20 / 25
  (≈ 9.7 ms fixed encoder/decoder overhead + 2.07 ms per Euler step).

## 2. Generation (one design serves every analysis)
- For every window, every **S ∈ {5, 10, 15, 20, 25}** and every **draw d ∈ {0, …, 15}**: one full official sampling
  call (ECG-encoder → PPG-encoder → S Euler steps → ECG-decoder). No S < 5 anywhere in this experiment.
- Batches: 512 windows in data order (batch b = windows 512b … 512b + 511). Seed `torch.manual_seed(1_000_003·d + b)`
  before each call, **the same for every S** — draw d at S and at S′ share their posterior and initial noise (common random
  numbers); draws are independent across d. Seeds and batch map are written to `seed_manifest.json`.
- Samples are stored (float16) in `outputs/ppgflowecg_external/gen/` (gitignored). 80 calls per window.

## 3. Functionals and reference
- **Primary — paper-faithful Hamilton HR**: the official `calculate_metric.ecg_bpm_array` (verbatim, biosppy 2.2.3 /
  neurokit2 0.1.7): generated sample `filter=True` (neurokit `pantompkins1985` clean), reference `filter=False` — exactly as
  the official `MAE_hr`. Y_k = HR(x_k); **Y\* = HR(processed reference ECG)**. An extractor exception, NaN or the official
  "no HR" value −1 makes that HR **undefined**.
- **Secondary — project-standard HR**: `ppg2ecg.evaluation.rpeaks.detect_rpeaks` + `hr_bpm` on generated and reference
  (its own Y\*). Used only for directional consistency with the iMF / CD / PENGUIN results; never mixed with the primary.
- HR windows Ω_HR = windows with Y\* defined (the secondary evaluator has its own Ω).

## 4. Cells, estimator and contrasts
Budget B = K × S vector-field NFE. Cells: **B10** (1,10), (2,5) · **B15** (1,15), (3,5) · **B20** (1,20), (2,10), (4,5) ·
**B25 (secondary budget)** (1,25), (5,5). Also computed: (1,5) and the K = 16 cells of §6.
- **Consensus** of a group of samples: median of its defined Y_k (undefined only if all are undefined). K = 2 median = mean;
  it is not read as robust-median evidence.
- **Primary estimator (partition average):** the 16 draws are split into disjoint groups of K in draw order
  (K = 1: 16 groups; 2: 8; 3: 5 (draws 0–14); 4: 4; 5: 3 (draws 0–14); 16: 1). Window error of cell (K, S) = mean over its
  groups with a defined consensus of |T̂_g − Y\*| — the expected error of one run of that allocation. *Robustness:* nested
  draws 0 … K − 1 only.
- **Aggregation:** window errors → patient mean → mean over patients (patient macro). Every contrast is computed on the
  windows where both cells are defined (coverage of every cell reported).
- **Headline contrasts (fixed; negative = width-heavy better):** B10 (2,5) − (1,10) · B15 (3,5) − (1,15) ·
  B20 (4,5) − (1,20) · B25 (5,5) − (1,25). (2,10) is reported, with (2,10) − (1,20), as a descriptive intermediate; it is
  never selected. No other cell is promoted after seeing results.
- **Inference:** paired patient-clustered bootstrap, **5,000 replicates, seed 20260924** (patients resampled with
  replacement; the statistic is the patient-macro difference). Reported: both absolute errors with CI, the difference with
  95 % percentile CI, and the per-patient win rate (share of patients whose mean difference is < 0; ties reported).
  No multiplicity correction; the verdict rule below is the only decision rule.

## 5. Depth curve (K = 1) and baselines
- E(1, S) for S = 5 … 25 with CI, and E(1, S) − E(1, 10) with paired CI; the ordering is reported as **"step-count
  behaviour under our evaluation"** (whether S = 5 is lowest and error grows with S, the paper's §4.5 observation) — not a
  reproduction.
- **PPG sanity baseline** (vs the same Y\*): primary `scipy.signal.find_peaks(ppg, distance=42, prominence=0.3)` on the
  processed PPG → `hr_bpm`; secondary the official `ppg_bpm_array`. If E(1, 10) is worse than the primary PPG baseline, the
  instability flag is raised and stated as a limitation; the within-model contrasts are still evaluated.
- **Constant baseline:** predicting the median Y\* over Ω_HR for every window.

## 6. Fixed-K mechanism (K = 16, all S, same 16 draws)
On Ω_mech = windows with Y\* and all 80 sample HRs (16 draws × 5 S) defined, per S:
individual error I_S = mean_k |Y_k − Y\*|; consensus error C_S = |median_k Y_k − Y\*|; **G_S = I_S − C_S**; functional SD
(ddof 1) and MAD (median |Y_k − median|); waveform pairwise RMS (mean over the 120 pairs of the RMS difference of the two
generated waveforms); all patient-macro. **ρ̄_S** = mean over the 120 draw pairs (k, l) of the Pearson correlation, across
windows of Ω_mech, of the signed errors e_k = Y_k − Y\*; K_eff = 16 / (1 + 15 ρ̄_S) (descriptive).
- Question: does low functional-error redundancy go with larger consensus gain across the five depths, as before?
  Reported: the ordering; Spearman(ρ̄_S, G_S) over the five S; also Spearman of G_S with waveform RMS, SD, MAD and of ρ̄_S
  with G_S / I_S; a **patient bootstrap** (5,000, seed 20260924: ρ̄_S and G_S recomputed from resampled patients) giving
  each Spearman's percentile interval and share of negative replicates. **"Reproduces"** = point estimate < 0 and ≥ 95 %
  of replicates < 0; otherwise "does not reproduce" (recorded as such). With five points this is descriptive only.
- Secondary, as in RDDM-EXT: at each S, patient-level Spearman(ρ̄_p, G_p) over patients with ≥ 8 Ω_mech windows.
- If samples are highly redundant (ρ̄ near 1, G near 0) that is reported as the result; nothing is added to create
  diversity.

## 7. Waveform / event metrics (secondary; per sample, averaged over the 16 draws, patient macro)
MAE, RMSE, Pearson r vs the processed reference; R-peak F1 at 50 ms and RR-MAE (project detector on both); the official
`calculate_fd` (raw-signal Fréchet distance, draw 0, all windows) per S. Width-heavy cells inherit the S = 5 per-sample
values. Wall-clock per cell: K sequential batch-1 calls, and one call with the K samples batched (GPU); NFE = K × S
vector-field calls plus K × (2 encoder + 1 decoder) calls.

## 8. Verdict (fixed; categories exactly as specified)
Primary budgets: B10, B15, B20. B25 is reported but cannot raise the verdict. A budget **succeeds** if its headline
contrast has a 95 % CI entirely below 0; it **fails significantly** if the CI is entirely above 0.
Flags, evaluated first:
- *Technical (→ UNINTERPRETABLE):* > 1 % non-finite generated samples, or Hamilton HR undefined for > 50 % of (1,10)
  samples, or Y\* undefined for > 50 % of windows.
- *Not usable (→ FAILED):* E(1,10) not below the constant baseline (patient-macro point estimates).
- *No useful consensus gain (→ FAILED):* E(16,5) − E(1,5) has a 95 % CI upper bound ≥ 0.
- *Waveform / event collapse:* at S = 5, mean per-sample R-peak F1 lower than the best of S ∈ {10, 15, 20, 25} by > 0.05
  absolute **and** > 20 % relative, or RMSE higher than the best by > 20 % relative, or > 1 % non-finite samples at S = 5.

Then:
- **STRONG** — ≥ 2 primary budgets succeed, at least one of them B15 or B20; no primary budget fails significantly; no
  collapse.
- **PARTIAL** — not STRONG, not FAILED: at least one primary budget succeeds, or ≥ 2 of the 3 primary point estimates are
  < 0; or STRONG-level HR results together with a collapse.
- **FAILED** — all three primary point estimates ≥ 0 (depth consistently better or equal), or no budget succeeds and ≤ 1
  point estimate is < 0, or one of the FAILED flags.
- **UNINTERPRETABLE** — technical flag.

## 9. Claim boundaries (fixed)
Allowed only if STRONG (PARTIAL: "in part"): "An independently developed latent rectified-flow generator exhibits the same
fixed-budget preference for allocating compute toward multiple sufficiently refined samples."
Not allowed under any outcome: width always wins; T = 5 is universally optimal; all flow models benefit; the checkpoint is
confirmed MCMED-trained; this improves PPGFlowECG's reported test performance; the external model proves causality;
generation beats direct HR prediction. Absolute zero-shot numbers are not called a reproduction of the paper's tables.

## 10. Outputs
`artifacts/ppgflowecg_external/`: audit.json, checkpoint_hashes.txt, nfe_counts.json, smoke_test.json, data_build.json,
latency.json, seed_manifest.json, fixed_budget.csv, fixed_budget_bootstrap.json, fixed_k_mechanism.csv,
fixed_k_bootstrap.json, per_patient_metrics.csv. Large arrays: `outputs/ppgflowecg_external/`. Report:
`docs/PPGFLOWECG_EXTERNAL_FIXED_BUDGET_REPORT.md`. After it: commit, push, stop — no respiration / ABP / new model.
