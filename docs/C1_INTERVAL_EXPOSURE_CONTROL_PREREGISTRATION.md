# C1 — Target-Interval Exposure Control for 2-NFE Compression — PREREGISTRATION

**Status:** frozen at this commit, pushed **before any C1 weight update**.
**Population:** WildPPG **development only** — `an0`, `k2s`. Test subjects `kjd`/`ssx` are never loaded.

**C1 is a MECHANISM CONTROL, not a compression method.** It is not a new architecture, not a distillation
experiment, not an event-conditioning experiment, not a source-distribution experiment, not a residual-flow
experiment, not an external benchmark, and not a test-set experiment.

**Exactly one thing may differ between arms: the distribution of (t, r), equivalently h = t − r.**

---

## 1. Question, and the caveat that bounds it

> **Is part of the NFE 2 → NFE 4 gap caused by insufficient training exposure to the h = 0.5 interval
> queried by a uniform 2-NFE sampler?**

**This hypothesis is NOT already supported.** X4-0 measured the training exposure (exact h = 0 mass ≈ 0.50,
positive-h median ≈ 0.201, P(h ≥ 0.5) ≈ 0.042, P(h ≥ 0.7) ≈ 0.004, P(h = 1) = 0) but its preregistered
**inference-time interval-stress experiment did not detect a material large-h effect** for schedules up to
h = 0.70. C1 is therefore justified only as *"target-interval exposure remains a cheap, falsifiable
training-distribution hypothesis"* — never as *"large-h mismatch is known to cause the failure"*.

**If C1 fails, this hypothesis is killed and no further h-distribution tuning is done.** The next candidate
would be explicit NFE 4 → NFE 2 trajectory compression / distillation, under a separate preregistration,
not started inside C1.

## 2. Provenance, verified before writing this document

| item | value |
|---|---|
| start HEAD | `d618968b5358c2cd19cae2d9adce09aef49ad5de` (C0 result) |
| S1 | `a29225a` · C0 prereg `5df1a33` · C0 result `d618968` |
| frozen A4 checkpoint | `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt`, md5 `31c042d291052fbb6dc15263ad316be2` |
| submodules | `external/PENGUIN` `6cd70cd`, `external/iMeanFlow` `bf60cd7` |
| GPU | one RTX 5090, 32,607 MiB |

### 2.1 Baseline sampler audit (read-only, no model, no data)

`sample_tr` (`src/ppg2ecg/flow/imeanflow.py:69-80`) draws `t, r` i.i.d. logit-normal(`p_mean=-0.4`,
`p_std=1.0`), sorts them so `t ≥ r`, then sets `r = t` on the **first `int(B · data_proportion)` rows** —
**deterministic by row position, not a Bernoulli draw**. 2,000,000 Monte-Carlo draws at seed 20260901
reproduce the recorded exposure:

| quantity | measured | X4-0 record |
|---|---:|---:|
| P(h = 0) | **0.5000** | ~0.50 |
| positive-h median | **0.2011** | ~0.201 |
| P(h ≥ 0.5) | **0.0422** | ~0.042 |
| P(h ≥ 0.7) | **0.0042** | ~0.004 |
| max h | 0.9276 | 0.9297 |

No discrepancy; the sampler is not reinterpreted.

## 3. Three arms

Output directories, none of which may overwrite `outputs/a4_imeanflow_wildppg_seed42/`:
`outputs/c1_imf_baseline_replay_seed42/`, `outputs/c1_imf_h25_seed42/`, `outputs/c1_imf_h50_seed42/`.

### ARM B — `C1-B`, baseline replay
The current training sampler used **exactly**, no change to the h distribution. Establishes whether the
C0 NFE 2 → NFE 4 target gap survives a fresh, controlled training replay.

### ARM H25 — `C1-H25`, specificity control
The exact-h = 0 branch probability is **preserved exactly**. Among positive-h samples only, half are
forced to **h = 0.25**: `t ~ Uniform[0.25, 1]`, `r = t − 0.25`.

### ARM H50 — `C1-H50`, target intervention
Identical construction with **h = 0.50**: `t ~ Uniform[0.50, 1]`, `r = t − 0.50`.

Expected total mass in both intervention arms: **50 % h = 0 · 25 % original positive-h · 25 % forced**.

**OPERATIONALISATION (fixed here, before results).** "With probability 0.5" over positive-h rows is
realised **deterministically by row position**, matching `sample_tr`'s own `fm_mask` idiom: within a
micro-batch the positive-h rows are `[int(B·0.5), B)`, and the **second half of those by position** is
forced. This realises the stated 25 % / 25 % split *exactly* rather than in expectation. Forced `t` is
drawn from the dedicated (t, r) generator.

**Frozen and not tunable after results:** the 25 % total forced mass, the two forced h values, and the
absence of an h = 1 arm, an h grid search, a mixture-probability grid, α tuning, curriculum, altered
`p`/`c`, altered architecture, and altered loss. **No loss reweighting in any arm.**

## 4. The validation-bank trap, and how it is avoided

`train_a2.py:122` builds the deterministic selection banks with
`make_imf_banks(len(x_va), T, n_val_banks, bank_seed, **tr_kw)` where
`tr_kw = dict(p_mean, p_std, data_proportion)` (`:121`). **If the intervention were expressed by changing
`tr_kw`, the validation banks — and therefore the checkpoint-selection criterion itself — would differ
between arms, confounding the whole experiment.**

C1 therefore implements the intervention as a **post-processing step applied on top of `sample_tr` at
training time only**, leaving `tr_kw` untouched. Consequences, all asserted before training:

- `make_imf_banks(**tr_kw)` receives identical arguments in all three arms;
- `imf_bank_hash(banks)` is **identical** across B / H25 / H50;
- the selection criterion `fixed_imf_mse` over those banks is identical;
- arm B's consumption of the (t, r) stream is byte-identical to A4's.

## 5. Training protocol — the recovered A4 recipe, unchanged

Recovered from `scripts/run_exp.sh` and `outputs/a4_imeanflow_wildppg_seed42/train_meta.json`:

`train_a2.py`, seed **42**, epochs 300, patience **20**, `min_delta` 1e-4, batch 64, micro-batch 32,
val-batch 32, `lr` 1e-3, weight decay 0.01, `cond_mode` `h_only`, `h_scale` 1.0, `p_mean` −0.4,
`p_std` 1.0, `data_proportion` 0.5, `norm_p` 1.0, `norm_eps` 0.01, `jvp_mode` `forward`, blocks 4,
`h_dim` 128, `ssm_ratio` 2.0, `mlp_ratio` 2.0, `val_every_steps` 220, `val_subsample` 4096,
`gen_diag_every` 1, `gen_diag_windows` 128, manifest `data/manifests/split_a4_wildppg_seed42.json`,
processed `data/processed/wildppg_8s`.

**Checkpoint selection:** `criterion = fixed_imf_mse`, `n_val_banks` 4, `bank_seed` 1000, `min_delta` 1e-4,
patience 20 — recovered exactly, not invented. No hyperparameter tuning anywhere.

## 6. RNG control

The trainer already separates the streams: `seed_everything(42, deterministic=True)` (`:104`); `gen`
seeded 42 for the dataloader/window order (`:129-130`); **`tr_gen` seeded 43 for (t, r) only, CPU,
explicitly independent of the shuffle stream** (`:131-132`); and `e = torch.randn(..., device=device)`
(`:191`) drawing from the global CUDA stream in a fixed order and shape per step. The intervention touches
**only `tr_gen`**, so the other three streams are unaffected by construction.

**Asserted before training, not assumed** — for all three arms:

1. identical initial `state_dict` hash;
2. identical first-N dataloader window-index hash;
3. identical first-N Gaussian `e` hash;
4. identical validation-bank hash;
5. the (t, r, h) hash **differs** for H25/H50 and **matches** A4's construction for B.

**If clean separation cannot be demonstrated, C1 STOPS BEFORE TRAINING and reports why.** A shared global
seed is not accepted as evidence that the trajectories are paired.

## 7. Evaluation

Frozen C0 population: X4-0 stage-B, `an0` 1,024 + `k2s` 1,024 = **2,048 windows**, **19,834 GT beats**.
Evaluation Gaussian source bank **seed 0**, **the same source tensor at NFE 2 and NFE 4**. NFE grid is
**{2, 4} only** — no NFE 1, no NFE 8. `kjd`/`ssx` are never loaded, visualised or scored.

### Primary metrics — oracle-free, GT-fixed coordinates
Computed by the frozen `alignment_diagnostics.beat_level_analysis`, `raw_*` outputs only; every valid GT
beat included at its GT R-peak coordinate; no prediction detector gates inclusion; no shift search.
Per-window aggregation follows the C0 convention (`nanmean` for correlation and RMSEs, `nanmedian` for
ratios).

**Gap metrics, counted (these are exactly the four C0 showed clearly improving 2 → 4):**
**M1** `|raw_qrs_energy_ratio − 1|` · **M2** `|raw_p2p_ratio − 1|` · **M3** `raw_qrs_rmse` ·
**M4** `raw_rmse` — all lower-better.

**Always reported, never counted for gap closure:** **M5** `raw_corr` (higher-better),
**M6** `|raw_slope_ratio − 1|` (lower-better).

### Secondary — reported, never decisive
Event F1 @50 ms, count-matched random-phase chance floor and F1 excess (S1 construction verbatim, 20 draws,
`default_rng(20260901)`), beats-ratio deviation from 1, matched-beat coverage, matched-beat morphology.

**Excluded everywhere:** `oracle_corr`, `oracle_absent`, `oracle_qrs_energy_median`, and anything
inheriting the ±150 ms shift maximisation.

## 8. Statistics

Subject-stratified **paired** bootstrap, 2,000 resamples, `default_rng(20260901)`, equal subject weight,
using `paired_stats.paired_subject_bootstrap` from C0. Positive always means the later/intervention arm is
better. Difference-of-improvement (H50 improvement − H25 improvement) is computed for M1–M4 with the same
machinery; positive means H50 is more beneficial.

**Single training seed only. The bootstrap quantifies development-window uncertainty, NOT training-seed
variance. No causal certainty across training seeds may be claimed.**

## 9. Staged resource gate

### STAGE 1 — train and evaluate `C1-B` only

Orient every metric so positive = NFE 4 better than NFE 2. **PASS iff all three hold:**

1. at least **three** of M1–M4 have paired 95 % CI entirely **> 0**; **and**
2. **none** of M1–M4 has CI entirely **< 0**; **and**
3. event F1 excess does not clearly collapse from NFE 2 → NFE 4.

**On FAIL → FINAL VERDICT `BASELINE TARGET GAP NOT REPRODUCED`, immediate STOP.** H25 and H50 are not
trained. The scientific consequence is that the single-checkpoint C0 compression target is not stable
enough under training replay to support an interval-exposure mechanism study, and a multi-seed stability
protocol would be required next.

On PASS: write `docs/C1_BASELINE_REPLAY_GATE_REPORT.md`, commit, push, then proceed.

### STAGE 2 — train `C1-H25` and `C1-H50`
Under the frozen protocol, with no tuning after observing C1-B. Evaluate each at NFE 2 and NFE 4. **NFE 2
is the primary decision; NFE 4 is a preservation readout.**

## 10. Gap closure

The denominator is the **C1-B replay**, not the old C0 numbers. With Q oriented so higher = better:

`G_m = Q(B, NFE4) − Q(B, NFE2)` · `I_m(X) = Q(X, NFE2) − Q(B, NFE2)` · `C_m(X) = I_m(X) / G_m`

Raw values are reported even when `C < 0` or `C > 1`; **no clipping to [0, 1]**. `C_m` is computed only
where `G_m > 0`. Gap closure is **descriptive**; inference rests on the paired bootstrap of `I_m`.

## 11. H50 success gate

The h = 0.5 exposure hypothesis receives SUPPORT only if **all five** hold:

1. H50 at NFE 2 clearly improves at least **three** of M1–M4 versus C1-B at NFE 2 (paired CI entirely > 0);
2. H50 clearly worsens **none** of the six primary structural metrics (M1–M6);
3. at least **two** of M1–M4 have point gap closure **≥ 0.50**;
4. event F1 excess does **not** clearly worsen versus C1-B NFE 2;
5. beats-ratio deviation from 1 does **not** clearly worsen.

## 12. Specificity and final verdicts

- **A — `TARGET h=0.5 EXPOSURE SUPPORTED`**: H50 passes §11, **and** H50's improvement exceeds H25's on at
  least **two** of M1–M4 with paired difference-of-improvement CI entirely > 0, **and** H25 does not
  independently pass the §11 gate to approximately the same degree.
  Permitted reading: *"Targeted exposure to h = 0.5 specifically closes a material part of the NFE2→NFE4
  gap under this single-seed development protocol."* This is **not** a final method.
- **B — `GENERIC POSITIVE-h REWEIGHTING EFFECT`**: H50 passes §11 but H25 passes similarly, or H50 does not
  clearly outperform H25. Permitted reading: the benefit is not specific to h = 0.5; changing positive-h
  exposure acts as a more generic training-distribution regulariser. **It may not be called large-h
  mismatch.**
- **C — `INTERVAL-EXPOSURE HYPOTHESIS NOT SUPPORTED`**: H50 fails §11. Permitted reading: increasing direct
  training exposure to h = 0.5 is insufficient to explain or close the gap. **Kill further h-distribution
  tuning.**
- **D — `BASELINE TARGET GAP NOT REPRODUCED`**: Stage 1 fails.

## 13. NFE 4 preservation check

For H25 and H50, also evaluate NFE 4 against C1-B NFE 4 on all six primary metrics. **Report only; no
additional success scalar is created.** Flag **`NFE4 DEGRADATION`** if ≥ 2 primary metrics clearly worsen.
If H50 passes the NFE 2 gate but causes NFE 4 degradation, **that must be stated explicitly, not hidden.**

## 14. Sampler exposure validation, before training

2,000,000 draws per arm, seed 20260901, no PPG/ECG access. Report for B / H25 / H50: exact `P(h = 0)`,
`P(h = 0.25)`, `P(h = 0.50)`, positive-h median, `P(h ≥ 0.125)`, `P(h ≥ 0.25)`, `P(h ≥ 0.5)`,
`P(h ≥ 0.7)`, max h. **Training does not start if the measured exposure does not match this document.**

## 15. Implementation tests, before any training

Baseline-sampler parity with the historical implementation · exact-h = 0 branch probability preserved in
all arms · H25 forced interval exactly 0.25 · H50 exactly 0.50 · forced samples satisfy `0 ≤ r ≤ t ≤ 1` ·
`t − r` equals the target h within floating tolerance · no loss-weight change · identical initial-state
hash across arms · identical data-order hash · identical Gaussian-noise hash · only h/t/r differs ·
test-subject firewall · checkpoint-overwrite protection · evaluation source-bank parity · oracle metrics
absent from the decision path · C0 metric-primitive parity · paired bootstrap deterministic · full pytest.
**Any RNG-control assertion failure stops C1 before training.**

## 16. Interpretation restrictions

Prohibited: *"large h causes the one-step failure"*, *"training mismatch is proven"*, *"h = 0.5 is the only
problem"*, *"NFE4 quality is solved"*, *"we achieved compression"*, *"SOTA"*, and anything about
`kjd`/`ssx`. Even on a pass, C1 shows only that targeted interval exposure can improve NFE 2 **under this
training seed and this development population**; training-seed variance is not estimated, so no
generalisation beyond that is permitted. `oracle_corr`, `oracle_qrs_energy` and `oracle_absent` may not be
resurrected, and matched morphology may not serve as primary evidence.

## 17. Deliverables and stop rules

`docs/C1_BASELINE_REPLAY_GATE_REPORT.md` (Stage 1), `docs/C1_INTERVAL_EXPOSURE_CONTROL_REPORT.md` (final);
`artifacts/c1_interval_exposure/` (gitignored) with `sampler_exposure.csv`, `baseline_replay_metrics.csv`,
`intervention_metrics.csv`, `paired_differences.csv`, `gap_closure.csv`, `specificity.csv`,
`decision.json`, `provenance.json`, `figures/`.

1. C1 ends at its verdict. **No compression method is implemented, selected or trained inside C1.**
2. `outputs/a4_imeanflow_wildppg_seed42/` is never overwritten.
3. Submodules stay byte-identical. Checkpoints and prediction dumps never enter git.
