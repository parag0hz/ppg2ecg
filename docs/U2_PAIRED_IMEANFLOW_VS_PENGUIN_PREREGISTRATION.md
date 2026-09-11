# U2 — Is iMeanFlow better than PENGUIN? A paired, compute-matched test under the shipped protocol

**DRAFT preregistration — not yet frozen.** Freezes on commit and push; never edited afterwards.

Start HEAD `9ba7e2e`. Upstream pins `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7c`;
neither is modified by this stage.

---

## 1. Question

Does **Improved MeanFlow (iMeanFlow)** beat **PENGUIN's OT-CFM** on PENGUIN's own six tasks, when
architecture, data, preprocessing, split, optimiser, training compute and evaluation code are all held
identical and only the generative objective differs?

The claim being tested is **not** "iMeanFlow scores better at PENGUIN's 50-NFE budget". It is the claim a
few-step method actually makes:

> **iMeanFlow reaches OT-CFM's 50-NFE quality at NFE 1–4, i.e. at 12–50× less sampling compute.**

## 2. Why the evidence we already have does not answer it

| existing result | why it is not an answer |
|---|---|
| D3's six-dataset table | compares **our** arms to the **published** numbers, not to a PENGUIN we trained. Different window length, split, and (for three corpora) no OT-CFM partner at all — D3 says so itself |
| U1 | runs upstream's pipeline at 4 s on upstream's filesystem-order split; our arms run 8 s on our splits. Putting the two columns side by side repeats the budget-mismatch error D3 already had to correct |
| A3 (PPG-DaLiA), A4 (WildPPG) | genuine paired OT-CFM vs iMeanFlow runs — but only two datasets, 8 s, one seed |

What is missing is the paired comparison on the other four corpora, under one protocol, with a decision
rule fixed in advance.

## 3. What makes this comparison clean

Both arms are built by `ppg2ecg.models.build_penguin_backbone` — **the same 4,568,707-parameter PENGUIN
S5 backbone**, the same PPG conditioning path. `train_a0.py` and `train_a2.py` already accept identical
`--processed / --manifest / --seed / --epochs / --batch-size / --lr / --weight-decay / --h-dim / --blocks
/ --ssm-ratio / --mlp-ratio / --sample-rate / --val-every-steps / --val-subsample`. The only difference
between the arms is the objective and its sampler.

| | **arm C** | **arm I** |
|---|---|---|
| objective | OT-CFM (PENGUIN's, `flow/cfm.py`) | Improved MeanFlow (`flow/imeanflow.py`, frozen arm-U configuration) |
| trainer | `train_a0.py` | `train_a2.py` |
| sampler | Heun / Euler | `z_r = z_t − (t−r)·u(z_t, ppg, t, t−r)` |
| backbone | identical | identical |

## 4. Data — reuse U1's, unchanged

`data/processed/upstream_u1/<DATASET>/subject*.pkl`, produced by **upstream's own `preprocess.py`** during
U1: 4 s windows at 128 Hz, PPG band-pass [0.5, 4] + z-score + min–max, per-dataset label handling as
shipped (ECG high-pass 0.5; Resp low-pass 1; **ABP unfiltered and un-normalised, raw mmHg**). The sha256
of every file is recorded before training and re-checked after.

Running at the shipped 4 s rather than our 8 s is deliberate and is against our own interest: our arms were
developed and tuned at 8 s, and 4 s is the configuration U1 showed reproduces the published table. A win
here is therefore a win on their turf.

**ABP is a shared handicap, not a confound.** Upstream leaves ABP in raw mmHg (mean ≈ 88.8, sd ≈ 23.7), so
both objectives must integrate the same ~89 mmHg offset and ~24× scale gap. Identical for both arms;
recorded so it is not later mistaken for a property of either objective.

## 5. Split — declared here, identical for both arms

**Not** upstream's `random.sample`-over-`glob` split: it is filesystem-order dependent and leaves one test
subject on four of six corpora, which cannot support any comparison. Instead, per dataset:

```
subjects sorted by numeric index; perm = numpy.random.default_rng(20260911).permutation(n)
n_test = n_val = max(2, round(0.15 n));  test = perm[:n_test], val = perm[n_test:n_test+n_val], train = rest
```

| dataset | subjects | train / val / test |
|---|---|---|
| PPG-DaLiA | 15 | 11 / 2 / 2 |
| WildPPG | 14 (firewall) | 10 / 2 / 2 |
| BIDMC | 53 | 37 / 8 / 8 |
| WESAD | 15 | 11 / 2 / 2 |
| MIMIC-BP | 1524 | 1066 / 229 / 229 |
| **UCI-BP** | **4 unique** | **2 / 1 / 1** |

**UCI-BP is deduplicated first.** U1-P1 established from file hashes that `load_data.py:100-102` makes the
eight "subjects" four chunks used twice (`0≡2, 1≡3, 4≡6, 5≡7`), so upstream's own split put a test
subject's byte-identical twin in train. U2 keeps only subjects `{0, 1, 4, 5}` and splits those. This leaves
one test subject: **UCI-BP carries no interval and cannot contribute to the primary verdict** (§9). It is
run and reported anyway, marked low-power.

WildPPG firewall: `kjd` and `ssx` are absent from `data/processed/upstream_u1/WildPPG` by construction
(14-subject symlink view) and are re-asserted absent from all three lists before training.

## 6. Compute matching — exact optimizer steps, not epochs

Both arms get **exactly `S = 14,000` optimizer steps**, batch 64, AdamW lr 1e-3 / weight decay 0.01, fp32,
seed 42, no early stopping.

`batch_rounds` (byte-identical in both trainers) ends a round at `min(steps left in epoch, --val-every-steps)`,
so `--epochs × --val-every-steps` is **not** a step count — it is corpus-dependent. That is the arithmetic
error already recorded against the M2/M3 preregistration and realised in D3 as 4,752 / 8,250 / 14,331 steps
where 14,409 was intended. U2 does not repeat it.

**Required change:** add `--max-steps N` to `train_a0.py` and `train_a2.py`, identical implementation in
both — training halts after exactly N optimizer steps. Default `None` = current behaviour.

Before any U2 run starts, a regression test must show, on a deterministic smoke fixture, that with
`--max-steps` absent each trainer produces a **bit-identical state_dict and optimizer-step sequence** to the
same trainer at the current HEAD, and that the new branch is not entered when the flag is `None`. A full
arm-U retrain is deliberately **not** required: the flag is a no-op at its default, so the fixture proves
the same property in seconds. Arm-U's state sha256
`20ba7234f25e0fe30960ba0e441d4b9f751c20003c85fee80ec3c08367aa095e` is re-checked from disk (unchanged
file), not recomputed by retraining.

Step-matching means very different epoch counts (BIDMC ≈ 200 passes, MIMIC-BP ≈ 4). Both arms face the
same; realised epochs are reported per dataset.

**Checkpoint selection** is each arm's own frozen deterministic in-objective metric — arm C `fixed_cfm`,
arm I `fixed_imf_mse` — computed on the validation split, never on test. These differ between arms by
necessity (the objectives differ); the asymmetry is declared in §10, not hidden.

## 7. Evaluation — one code path, full NFE grid, both arms

| arm | published NFE points |
|---|---|
| arm C | 1 (Euler 1), 2 (Heun 1), 4 (Heun 2), **50 (Heun 25)** |
| arm I | 1, 2, 4 (iMF steps) |

Every point is published; none is selected after seeing results. Four evaluation noise draws per window
(seeds 0–3), mean and spread reported — this quantifies **sampling** stochasticity, which one training seed
cannot.

### Metrics

Primary metric per task is PENGUIN's own, computed by `evaluation/penguin_metrics.py`. Each task also
carries a **co-primary that requires conditioning**, because D2 established that the headline metrics do
not:

| task | PENGUIN metric | co-primary | why the co-primary is needed |
|---|---|---|---|
| ECG (PPG-DaLiA, WildPPG) | `HeartRateError` | **R-peak F1 @ ±50 ms** | D2: HR error is satisfied by beat **rate** alone; U1's figures show beats generated but not placed |
| Resp (BIDMC, WESAD) | `RespRateError` | **low-passed waveform Pearson r** (1 Hz low-pass applied to **both** signals, over the 60 s metric window) | upstream's metric is FFT-argmax and therefore **phase-blind**; U1's figures show near-antiphase windows that it cannot penalise |
| ABP (UCI-BP, MIMIC-BP) | `SBPError`, `DBPError` | — | SBP (window max) and DBP (window min) already differ in noise sensitivity; both are primary |

The co-primary correlation is **ours**, not upstream's, and is labelled as such: upstream low-passes the
prediction only (`help_func.py:191`), which is asymmetric; U2's version low-passes both.

**RMSE and MAE are reported but are never a primary endpoint.** D2 established that on every corpus the
arm that uses nothing at all (B3) attains the lowest RMSE.

### KANFlow's metric set, on both arms (declared secondary)

D4 compiled KANFlow's metrics for **our** arms on **KANFlow's** corpora. U2 computes the same set for
**both** arms on **PENGUIN's six**, so the two method families can finally be read off one table:

| metric | direction | carried-over caveat |
|---|---|---|
| Micro-F1, Macro-F1 (beat detection) | ↑ | — |
| RR-MAE (ms) | ↓ | conditional on successful ≤50 ms beat matching; a lower value with a lower F1 partly reflects scoring fewer, easier beats (`METRIC_SEMANTICS.md`) |
| MAE_HR (bpm) | ↓ | D2: satisfied by beat **rate** alone — never read as reconstruction quality |
| FD (FID-style) and FD (discrete Fréchet) | ↓ | our FD uses a surrogate feature map and is **not** comparable to any published FID (D2, D4 §1.2). Comparable **between our two arms**, which is all U2 uses it for |
| MAE, RMSE | ↓ | scale-dependent; D4 §1.1. Reported, never ranked |

These are secondary throughout: they do not enter the §8 primary endpoint or the §9 verdict. They are
computed by the same code for both arms, so the arm-vs-arm difference is meaningful even where the
absolute value is not comparable to a published number.

### Controls (no training cost, same test sets)

From `evaluation/baselines.py`: **B3** (train-mean / template — uses no PPG) and **B4** (wrong-window
conditioning). If an arm's "win" does not also beat B3 and B4, the win is uninformative and is reported
as such.

## 8. Primary endpoint and margins — fixed here

For each dataset, and for each `k ∈ {1, 2, 4}`:

> **H₀:** arm I at NFE k is worse than arm C at NFE 50 by more than the margin δ.
> **Non-inferiority** is declared when the upper bound of the 95 % CI of `(I_k − C_50)` lies below δ.

| metric | δ | provenance |
|---|---|---|
| HeartRateError | **+1.0 bpm** | A0's own preregistered non-inferiority margin |
| R-peak F1 | **−0.02** | A0's own preregistered non-inferiority margin |
| RespRateError | **+0.5 bpm** | ≈ 2× the NFE-induced spread D3 measured (BIDMC 4.05–4.31, WESAD 4.90–5.12) |
| Resp waveform r | **−0.05** | mirrors A0's morphology-correlation margin |
| SBPError / DBPError | **+1.0 mmHg** | ≈ one bootstrap CI half-width of D3's MIMIC-BP cells (e.g. 14.76 [14.00, 15.50]) |

**Secondary, also fixed now:** superiority at matched NFE — `I_k` vs `C_k` for k ∈ {1, 2, 4}, and
`I_4` vs `C_50` as a superiority test. Reported with the same intervals, never substituted for the primary.

### Informativeness gate — a non-inferiority pass is not automatically evidence

D4 measured that on its three corpora **no metric improves monotonically with NFE**: Micro-F1 moves by
≤ 0.02 from NFE 1 to 50 everywhere, and MAE_HR gets *worse* with more steps on BIDMC and CAPNO. Where the
sampling budget buys nothing, "iMF at NFE 1 is non-inferior to OT-CFM at NFE 50" is true of *any* method
and says nothing about iMeanFlow.

So for every dataset × metric U2 also reports **arm C's own span**, `Δ_C = |C_1 − C_50|`, and applies:

> if `Δ_C < δ` — arm C's own 50× budget increase moves the metric by less than the non-inferiority
> margin — the cell is marked **UNINFORMATIVE** and **cannot contribute to a dataset verdict**, whatever
> arm I scores on it.

If every primary and co-primary cell of a dataset is uninformative, that dataset is excluded from the
stage count in §9 exactly as UCI-BP is, and the exclusion is reported as a finding about the metric, not
hidden. This rule is fixed now, before any U2 number exists, precisely because it can only make the
verdict harder to pass.

A known case to watch, already measured: on both ECG corpora arm C is **worse at NFE 2 than at NFE 1**
(PPG-DaLiA HR 35.23 → 41.89; WildPPG 15.59 → 40.61, R-peak F1 0.481 → 0.073), because Heun's corrector
diverges at one giant step. Any apparent iMF landslide at NFE 2 is therefore a win over a broken
baseline configuration and must be reported as such, not as a method advantage.

## 9. Uncertainty and decision rule

Paired, subject-clustered bootstrap (`scripts/d1_common.py: subject_cluster_bootstrap`), 2,000 replicates,
`numpy.random.default_rng(20260911)`: clusters are resampled **once** per replicate and both arms are
scored on the same resample, so the interval is on the **difference**, not on two independent means. Both
arms see byte-identical test windows, which is what makes the pairing valid.

**Dataset verdict.** Non-inferior at the smallest k whose CI upper bound is below δ on **both** that task's
primary and co-primary, counting only cells that pass the §8 informativeness gate. A dataset is excluded
from the stage count if it has fewer than two test subjects (UCI-BP, point estimate only) **or** if every
one of its primary/co-primary cells is UNINFORMATIVE. Exclusions are reported, never silently dropped, and
the stage denominator is restated against the datasets that actually counted.

**Stage verdict**, fixed now over the five countable datasets:

| verdict | rule |
|---|---|
| **SUPPORTED** | non-inferiority at some k ≤ 4 on **≥ 4 of 5** datasets |
| **PARTIAL** | 2 or 3 of 5 |
| **NOT SUPPORTED** | ≤ 1 of 5 |

No metric is dropped, no dataset is dropped after the fact, and a failure on any dataset is reported with
the same prominence as a success.

## 10. Declared limitations

1. **One training seed (42).** Chosen for budget. Nothing in U2 can speak to robustness against training
   randomness, and no claim will. Sampling stochasticity *is* quantified (four noise draws); the arms are
   paired on identical data, split and seed, which removes between-run variance from the comparison but
   not from the generalisation claim.
2. **UCI-BP has one test subject after deduplication** — point estimate only, excluded from the verdict.
3. **Selection-criterion asymmetry**: arm C selects on `fixed_cfm`, arm I on `fixed_imf_mse`. A single
   shared criterion would have to be sampler-based, which would couple selection to the very axis under
   test. Neither criterion sees the test split.
4. **Step-matched, not epoch-matched**: intended, but it means small corpora see far more passes over the
   same data than large ones. Identical for both arms.
5. U2 compares two objectives on one backbone. It is not a comparison against the published PENGUIN
   numbers — U1 is, and the two must not be merged into one table.

## 11. Firewalls

- `external/PENGUIN` and `external/iMeanFlow` unmodified; pins asserted before and after every run.
- WildPPG `kjd` / `ssx` never loaded; asserted absent from train, val and test of every run.
- Frozen A4 checkpoint md5 `31c042d291052fbb6dc15263ad316be2` and arm-U state sha256
  `20ba7234…a095e` unchanged; re-checked at the end.
- C2 remains deferred.
- No checkpoint, prediction, `.pkl` or raw data enters git.
- `--max-steps` default `None` must leave both trainers bit-identical; proven by test before any U2 run.

## 12. Cost and order

12 runs (6 datasets × 2 arms), one seed. At 14,000 steps, batch 64, 4 s windows: arm C ≈ 0.2 s/step,
arm I ≈ 0.4 s/step (forward-mode JVP, micro-batch 32), plus a capped validation pass every 220 steps
(`--val-subsample` identical for both arms). Estimated ≈ 1 h (arm C) and ≈ 1.5 h (arm I) per dataset,
**≈ 15 h total**, run sequentially — U1 measured that concurrency on this GPU costs ~8 % rather than
saving time.

Order: BIDMC, WESAD, PPG-DaLiA, WildPPG, MIMIC-BP, UCI-BP. Both arms of a dataset run back to back so a
partial stage still yields complete pairs.

## 13. Reporting

`docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_REPORT.md`:

- the NFE × metric grid for **both** arms on all six datasets, paired intervals on the difference
- PENGUIN's task metrics **and** KANFlow's set (Micro-F1, Macro-F1, RR-MAE, MAE_HR, both FD variants),
  so D4's table and this one can be read together
- arm C's own span `Δ_C` beside every margin, and the UNINFORMATIVE marking wherever `Δ_C < δ`
- per-dataset and stage verdicts, with the denominator restated against the datasets that counted
- control comparisons (B3, B4), realised steps and epochs, split manifests
- every limitation in §10 restated against the actual numbers, including the NFE-2 Heun collapse

U2 does not update, merge with, or supersede D3, D4 or U1. Those tables stand; U2 is a fourth, separate
table with its own protocol.
