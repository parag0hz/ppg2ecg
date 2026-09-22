# Top-tier completion — preregistration (EXP-A … EXP-F)

Frozen and pushed before any number of EXP-A/B/C/D is computed. No new architecture, no new loss, no retraining of any
generator. Central question of the paper, fixed:

> Given a conditional generative model and a fixed inference budget B, how should compute be allocated between per-sample
> refinement depth S and the number of stochastic samples K, when the goal is to estimate a downstream functional T while
> retaining structured generative outputs?

Notation: condition c (PPG window), structured output x, functional T(x), per-sample NFE S, samples K, budget B = K × S,
estimator `T̂_{K,S}(c) = median_k T(x_k)`, `x_k ~ q_{θ,S}(x | c)`. Working statement under test: *refine enough, then sample
wide.* Nothing below assumes "width always wins", "S = 1 is optimal", "waveform diversity is the mechanism", or "generation
is necessary for HR / R-peak estimation".

## 0. Common rules (apply to every experiment)
- **Budget = network function evaluations.** Euler step = 1 NFE. Repository Heun (`ppg2ecg.flow.samplers.heun_sample`, bit-exact
  with upstream `PENGUIN.sample`, `tests/test_upstream_parity.py`) = **2 NFE per step** (predictor + corrector; the last step is
  also a full Heun step), so upstream's shipped `n_step = 25` = 50 NFE. A "step" of a different solver is never counted as one NFE.
- **Statistics.** Paired, subject-level (VitalDB: patient; WildPPG / MIMIC-BP / BIDMC: subject), subject-clustered bootstrap,
  2,000 replicates, seed 20260911 (`v1_evaluate.cluster_ci`); effect size with 95 % CI; practical margins HR 1.0 bpm,
  F1 0.02 (unchanged); statistical significance and margin are reported separately. Raw per-subject / per-window values saved
  under `outputs/tt_*_raw/` (gitignored) and per-condition summaries under `artifacts/tt_*/` (tracked).
- **Selection.** No allocation is chosen on a test set and then reported as a headline. Headline allocations are either
  preregistered here or selected on a validation split and evaluated on test once. Full test grids may be shown as analysis
  figures with the label *exploratory*.
- **Reference functional.** HR = `rpeaks.hr_bpm(detect_rpeaks(·))` (neurokit) on generated and reference ECG alike; R-peak F1 @ 50 ms,
  RR-MAE, FD (`paper_metrics.kanflow_fd`, linspace subset ≤ 3,000 windows) exactly as the standard pipeline.
- **Noise seeds.** Sample k of a window uses noise seed k − 1 (0 … K − 1), shared across conditions, as in DW1.
- **Reporting.** Every experiment gets raw JSON/CSV, the exact command, a report markdown, figures, and a RESULTS_MASTER
  update candidate. Failures are reported in the same place as successes.
- **Hard stop.** After EXP-A and EXP-B are reported, no further experiment starts until the results have been reviewed.
  EXP-C is prepared only up to the availability audit (C0) in parallel.

---

## EXP-A — PENGUIN solver fairness: does the depth arm lose because of Euler?

**Objection to remove.** DW1 / DW2-A / WD1-B sampled PENGUIN with forward Euler; upstream's default is Heun. A reviewer may say
the pure-depth baseline was weakened by an inferior solver. EXP-A gives the incumbent the strongest depth baseline available
at the same NFE.

**Data / models.** VitalDB V1 test (1,156 patients, 19,543 windows). PENGUIN (OT-CFM, arm C) checkpoints seed 42
(`outputs/v1_vitaldb_armC_seed42`), seed 1 (`outputs/sr1_C_seed1`), seed 2 (`outputs/sr1_C_seed2`), unchanged. No training.

**Samplers.** Euler (`euler_sample`, NFE = steps) and Heun (`heun_sample`, NFE = 2 × steps), both on the uniform grid
`linspace(0, 1, steps + 1)`, identical noise z₀ per noise seed. Heun cannot reach 1 NFE per sample; the shallowest Heun sample
costs 2 NFE.

**Grids (all cells evaluated; exact totals).**
- Primary budget **B = 32 NFE**:
  Heun `(K, steps)`: (1, 16), (2, 8), (4, 4), (8, 2), (16, 1) — per-sample NFE 32 / 16 / 8 / 4 / 2.
  Euler `(K, S)`: (1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1) — seed-42 cells reuse `outputs/dw1_raw/`; seed-1/2
  cells (32, 1), (8, 4), (1, 32) reuse `outputs/dw2_raw/`; the rest are generated.
- Secondary budget **B = 50 NFE** (the shipped budget): Heun (1, 25) [= shipped configuration], (5, 5), (25, 1); Euler (1, 50),
  (5, 10), (10, 5), (25, 2), (50, 1).
- Reference: the preregistered width-heavy Euler allocation from DW1 / DW2-A, **(8, E4)** = 32 NFE (not re-selected).

**Estimator.** HR median over the K samples (K = 1: the single sample; expected error reported as the mean over the 32 noise
draws where available, and the noise-seed-0 draw is labelled as such).

**Primary metric.** HR error, patient-macro, paired against the depth extreme of the same solver and budget. **Secondary.**
R-peak F1 and FD of the noise-seed-0 single sample per (solver, per-sample NFE) (what depth buys the waveform); per-patient win
rate; GPU batched latency per cell (ms / window, batch 512) and the cell's exact NFE.

**Hypotheses and success criteria (3 seeds each; all CIs paired, patient-clustered).**
- **A-H1 (within-Heun width vs depth):** `err(16, H1) − err(1, H16)` CI upper < 0 **and** `err(8, H2) − err(1, H16)` CI upper < 0,
  in 3/3 seeds. Success = both hold in every seed.
- **A-H2 (strongest depth baseline vs the preregistered width-heavy arm):** `err(8, E4) − err(1, H16)` CI upper < 0 **and**
  `err(8, E4) − err(1, E32)` CI upper < 0, in 3/3 seeds (reported against both depth baselines; no "best of" selection).
- **A-H3 (shipped budget):** `err(5, H5) − err(1, H25)` CI upper < 0 and `err(25, H1) − err(1, H25)` CI upper < 0, in 3/3 seeds.
- **Reported without a hypothesis:** solver recovery at pure depth, `err(1, H16) − err(1, E32)` and `err(1, H25) − err(1, E50)`;
  and the F1 / FD-versus-HR tradeoff across per-sample NFE for both solvers.

**Failure interpretation.** If A-H1 fails in any seed, the PENGUIN row of claim C2 ("width beats depth for all three models")
is restated as "for the Euler solver only" and the Heun cells are reported as the counter-evidence. If A-H2 fails, the
statement "the consensus recipe transfers to the incumbent model without retraining" is withdrawn for PENGUIN. If A-H3 fails,
the claim that a width-heavy allocation beats the shipped configuration at its own budget is withdrawn. A large Heun
recovery at pure depth with A-H1 still holding is reported as "the solver matters, and width still wins within solver".

Script `scripts/tt_expa_solver.py`; raw `outputs/tt_expa_raw/`; results `artifacts/tt_expa_solver/{grid.csv, result.json,
figures}`; report `docs/TT_EXPA_SOLVER_FAIRNESS_REPORT.md`.

---

## EXP-B — Why width reduces functional error: decomposition at fixed K, fixed S, and error correlation

DW2-B's preregistered HR-SD rule failed; its waveform-diversity reading is post-hoc and is **not** assumed here. EXP-B replaces
"waveforms must look diverse" with an estimation-error decomposition. VitalDB V1 test, seed-42 checkpoints (iMF arm I,
consistency distillation arm D, PENGUIN arm C with Euler), samplers exactly as DW1 (`dw1_depth_width.make_sampler`); B2 adds
seeds 1 and 2 as replication.

Per window c, sample k at depth S: `Y_{S,k}(c) = HR(x_{S,k})`, reference `T*(c) = HR(ECG_target)`; center
`m_S(c) = median_k Y_{S,k}(c)`. Quantities (each patient-macro with clustered CI, paired across S on the same windows):
1. **center error** `C(S) = |m_S(c) − T*(c)|` (this is the K-sample consensus error);
2. **dispersion** `D_MAD(S) = median_k |Y_{S,k} − m_S|` and `D_SD(S) = SD_k Y_{S,k}` (both reported);
3. **individual error** `I(S) = mean_k |Y_{S,k}(c) − T*(c)|`;
4. **median gain** `G(S) = I(S) − C(S)`.
Windows where the reference HR is undefined are excluded (Ω_HR rule); a sample with undefined HR counts as missing in the median
(`nanmedian`) and in `I`, and the fraction of such samples per condition is reported.

### B1 — K fixed, S varied (unconfounded depth effect)
K = 16 for every condition, noise seeds 0 … 15 shared across S; S ∈ {1, 2, 4, 8}; models iMF, CD, PENGUIN (Euler). Total NFE
therefore varies (16 … 128) — this is intended: B1 isolates depth from K.
- Reported per (model, S): `C, D_MAD, D_SD, I, G` on all 19,543 windows; per-sample R-peak F1 (mean over the 16 samples) from the
  same detections; on the fixed 2,000-window `linspace` subset (DW2-B's): waveform pairwise RMS (16 samples), feature-space
  pairwise distance, FD of each sample against the targets (mean over samples).
- **Criteria.** B1-1 (PENGUIN): `C(S=4) − C(S=1)` CI upper < 0. B1-2 (iMF): `C(S=2) − C(S=1)` CI upper < 0. B1-3 (CD):
  `C(S=2) − C(S=1)` CI includes 0 or lower bound > 0 (no improvement). These mirror the DW1 optima at fixed K; failure of any one
  means the DW1 optimum for that model was a K-effect, and the "enough depth" part of the working statement is dropped for it.
- **Preregistered reading of the decomposition (not a pass/fail):** between S = 1 and the model's chosen S (CD 1 → 2, iMF 1 → 2,
  PENGUIN 1 → 4), report ΔC, ΔI, ΔG with CIs. If ΔC ≈ ΔI (G unchanged), depth improves each sample's functional (the
  model/depth-dependent term); if ΔC arises mostly through ΔG, depth changes the pooling structure. If neither C nor I moves
  while FD moves, depth buys waveform fidelity only.

### B2 — S fixed, K varied
S fixed in advance from DW1 / DW2-A (**not re-selected**): CD S = 1, iMF S = 2, PENGUIN S = 4. K ∈ {1, 2, 4, 8, 16, 32}, noise
seeds 0 … 31. Seeds 42 (primary), 1 and 2 (replication).
- **Estimate of R(K, S).** Primary: partition average — the 32 draws are split into 32 / K disjoint groups of K consecutive
  noise seeds, the median error is computed per group and averaged (K = 1 is then the 32-draw mean). Secondary: nested
  (noise seeds 0 … K − 1), for continuity with DW1 / DW2-A. Marginal gain `R(K) − R(2K)` with CI.
- **Criterion B2-1:** for each model and seed, marginal gains are non-increasing in K (each doubling gains no more than the
  previous one, within the CI). **Reported:** the saturation K under AB1's frozen rule (doubling changes HR by < 0.1 bpm);
  the qualitative fit of `R(K, S) ≈ R_∞(S) + finite-sample term` is examined by plotting `R(K) − R(32)` against 1/K — **no
  parametric fit is reported as a result**.

### B3 — Functional error correlation across samples
Signed error `e_{S,k}(c) = Y_{S,k}(c) − T*(c)` from B1 (K = 16, S ∈ {1, 2, 4, 8}, three models). For every sample pair (k, k′),
Pearson (primary) and Spearman correlation across test windows with both values finite; report the mean and the 5–95 %
range over the 120 pairs, plus the one-way intraclass correlation ICC(1) (between-window variance share).
- **Hypothesis B3-1:** every condition with a positive median gain (CI lower > 0) has mean pairwise correlation ρ̄ < 0.9.
- **Hypothesis B3-2:** across the 12 (model, S) conditions, Spearman(ρ̄, G) < 0 (less identical errors → larger gain).
- **Interpretation.** Support for B3-1 and B3-2 means "multiple samples help because their functional errors are not identical
  and the median suppresses finite-sample / outlier error" — a statement that does not need waveform diversity. If PENGUIN at
  S = 1 shows ρ̄ near 1 together with its known gain (+1.32 bpm, DW2-B), the two are in tension and both are reported. If
  B3-2 fails, the functional-error explanation is not supported and is reported as such.

Script `scripts/tt_expb_decomposition.py`; raw `outputs/tt_expb_raw/`; results `artifacts/tt_expb_mechanism/`; report
`docs/TT_EXPB_DECOMPOSITION_REPORT.md`.

---

## EXP-C — Plug-and-play on models developed by other authors

Question: *does the allocation / consensus rule improve conditional generators developed by others, on their own
checkpoints, without retraining?* Rules: no architecture change, official code and official pretrained checkpoint only, no
retraining with our loss, no re-implementation presented as "the existing model", original and ours compared on the **same
checkpoint**, original setting reproduced in our environment before any comparison, failed reproductions recorded,
"reported" and "our reproduction" never in the same cell.

### C0 — Availability audit (may run in parallel with EXP-A/B; produces no metric)
Candidates: PENGUIN (in hand), RDDM (priority 1), MAGIC, UA-P2E, any other public PPG→ECG diffusion / flow model (priority 2).
For each: paper, venue / year, model family, original sampling steps and exact NFE, official code (URL, commit), pretrained
checkpoint (URL / hash), compatible dataset and its split, whether stochastic conditional samples are possible, whether the
sampler supports the step counts a matched-NFE grid needs, usable (yes / no / partial) with the reason. Primary sources only
(repository, paper); a model without an official implementation is recorded as unusable, not re-implemented.
Output `docs/TT_EXPC_EXTERNAL_MODEL_AUDIT.md`.

### C1 — RDDM before / after (only if C0 = usable)
1. Reproduce the original inference (official checkpoint, official dataset / split / preprocessing, original step count
   `S_orig`, K = 1) and record our reproduction next to the published number, with the ratio.
2. Budget `B = S_orig` NFE (exact network calls of the official sampler, counted by instrumenting it). Factorisations
   `(K, S) ∈ {(1, S_orig)} ∪ {(K, S) : K × S = B, S a step count the official sampler supports}`; if the sampler does not support
   arbitrary respacing, only supported schedules are used and listed. No forced 1-step sampling that the sampler does not define.
3. Functional: HR by the standard extractor on each generated ECG, `ĤR = median_k HR(x_k)`; also R-peak F1, RR-MAE, FD,
   latency.
4. **Headline allocation is selected on the official validation split** (or, if none exists, on a held-out slice of the
   training subjects, never on test); test evaluated once. Table: setting · K · S · total NFE · HR · R-F1 · RR-MAE · FD · latency,
   rows: original inference; same checkpoint + validation-selected allocation; pure width (if a supported schedule); pure depth.
5. **Criterion C1-1:** `err(selected) − err(original)` CI upper < 0 on the subject-clustered bootstrap. Failure is reported as
   "the rule does not transfer to RDDM" with the full grid.

### C2 — MAGIC before / after (only if C0 = usable)
Same procedure with the original step count (e.g. 50 → grid over the supported divisors), same selection and criterion (C2-1).

### C3 — UA-P2E (only if C0 = usable; handled separately)
UA-P2E already draws multiple posterior ECG samples and averages downstream classifier scores; it is **not** described as
"ours + consensus". Question: *holding total NFE fixed, does reallocating compute from denoising depth to more posterior
samples improve UA-P2E's own score aggregation?* Aggregation rule unchanged (theirs); only (K, S) changes at equal NFE; metrics
= the paper's (AUC / accuracy / calibration). **Criterion C3-1:** the paper's primary metric improves with CI excluding 0 on the
same checkpoint and split. Absence of official code or checkpoint → recorded, not re-implemented.

Reports `docs/TT_EXPC_<MODEL>_REPORT.md`, raw `artifacts/tt_expc_<model>/`.

---

## EXP-D — Same PENGUIN, other physiological outputs (respiration, ABP)

Models: upstream PENGUIN trained **as shipped** by U1 (`external/PENGUIN` @ `6cd70cd`, unmodified; checkpoints
`outputs/u1_upstream/PENGUIN_<dataset>_u1/ckpt/pretrain_ckpt.pth`), loaded into the unmodified upstream `PENGUIN` class with the
shipped configuration; splits = upstream's own (`outputs/u1_upstream/split_<dataset>.json`); preprocessing = upstream's
(`ppg2ecg.data.preprocess`, parity-tested). Samplers Heun / Euler as in EXP-A. Functionals per PENGUIN's own metric code ported
verbatim (`ppg2ecg.evaluation.penguin_metrics`, unit-tested): `resp_rate_error` (60 s = 15 consecutive 4 s windows, dominant
FFT frequency, prediction-only low-pass preserved), `sbp_error` / `dbp_error` (per 8 s window max / min, mmHg).

**Budgets and arms (both tasks).** Shipped: (1, H25) = 50 NFE. B = 50 grid: Heun (1, 25), (5, 5), (25, 1); Euler (1, 50), (5, 10),
(10, 5), (25, 2), (50, 1). Fixed rule transferred from ECG **as is**: (8, E4) = 32 NFE (cheaper than shipped; reported separately
from any selected arm). **Validation-selected arm:** the B = 50 cell with the lowest primary functional error on the upstream
validation split; test evaluated once. Functional consensus = median over K of the per-sample functional (respiratory rate per
60 s block computed on each sample's concatenated block; SBP / DBP / MAP per window).

### D1 — Respiration (BIDMC primary; WESAD secondary)
BIDMC upstream split: 41 / 6 / 6 subjects (6 test subjects ⇒ few 60 s blocks; CIs will be wide and the result is labelled
low-power). WESAD upstream split has a single test subject: reported as exploratory only, no criterion.
Primary: RR error (breaths / min), subject-clustered. Secondary: waveform MAE (upstream's), PCC; FD with the standard feature map
labelled exploratory; latency.
**Criterion D1-1:** `RRerr(validation-selected) − RRerr(shipped)` CI upper < 0. **D1-2 (fixed rule):** `RRerr(8, E4) − RRerr(shipped)`
CI upper < 0 at 32 vs 50 NFE. **Question recorded:** whether depth improves the waveform metric while width improves the rate.

### D2 — ABP (MIMIC-BP; the primary generalisation test)
MIMIC-BP upstream split: 1,144 / 190 / 190 subjects, 8 s windows. Primary: SBP MAE and DBP MAE (both, PENGUIN's definitions).
Secondary: MAP = window mean of the waveform, `|MAP_pred − MAP_target|` (not in the original protocol; labelled ours); waveform MAE
(raw mmHg, as upstream), PCC, FD (exploratory); latency.
**Criterion D2-1:** SBP **and** DBP MAE of the validation-selected arm are lower than shipped (CI upper < 0 for both). **D2-2 (fixed
rule):** the same for (8, E4). If only one of SBP / DBP improves, the result is "mixed" and reported as such. ECG's S is not
called optimal here; the transferred and the selected arms are reported side by side.

**Failure interpretation (D).** If neither the selected nor the fixed arm beats shipped on a task, the allocation principle is
scoped to ECG (and the paper says so); if the selected arm wins but the fixed one does not, the principle holds but the
depth requirement is task-specific; if both win, the principle generalises across PENGUIN's three targets.

Script `scripts/tt_expd_penguin_targets.py`; raw `outputs/tt_expd_raw/`; results `artifacts/tt_expd_targets/`; report
`docs/TT_EXPD_OTHER_TARGETS_REPORT.md`.

---

## EXP-E — Minimal formalisation (no experiment)
`docs/CONSENSUS_INFERENCE_THEORY_NOTE.md`: the estimator as a statistical-estimation problem; error decomposition
`m̂_{S,K} − y* = (m_S − y*) + (m̂_{S,K} − m_S)` (model / depth-dependent term + finite-K term); the asymptotic variance of a sample
median `1 / (4 K f_Y(m)²)` stated as intuition under independence and regularity, not asserted for our model; the mean-estimator
intuition `MSE ≈ bias_S² + S σ_S² / B` at fixed budget and why it admits an interior optimum, marked as intuition because the
method uses the median. **Written before EXP-B's numbers are read; no theory parameter is adjusted to the data afterwards.**

## EXP-F — Paper tables (assembled from the reports; no new numbers)
Table A (existing model before / after the inference wrapper, "training changed? = No"): PENGUIN (ECG; Resp / ABP if D runs),
RDDM and any reproducible external model from C. Table B (what extra inference compute is used for): depth scaling,
best-of-N / repeated sampling, verifier-based inference-time scaling, search / tree-based scaling, functional consensus
inference — columns: extra compute allocation · external verifier? · selects best sample? · aggregates samples? · structured
samples preserved? Ours: no verifier, no best-sample selection, multiple conditional samples, functional aggregation, samples
retained.

## Final deliverable
`docs/TOP_TIER_COMPLETION_REPORT.md` with the sections *Claims now supported / Claims weakened / Claims falsified / Post-hoc
observations / Remaining reviewer objections / Experiments that are no longer worth running*.
