# U1 — Upstream PENGUIN, as shipped: six-dataset replication of the published table

**Preregistration. Frozen on commit and push; never edited post-hoc.**

Start HEAD `8d53f65f8176184f3fd2a48b1bbb82d08fd55921`.
Upstream pin `external/PENGUIN` @ `6cd70cdefb91f10efeb8dce34019b5067cb25344` — **not modified by this stage.**

---

## 1. Question

Does the released PENGUIN code (https://github.com/Neurogica/PENGUIN), executed **as shipped** in this
environment, reproduce the quantitative table of arXiv:2602.03858 on the six datasets it ships loaders for?

This is a replication of *their* code, not an evaluation of ours. No method of ours is involved and no
result of this stage may be used to select, tune or justify any of our arms.

## 2. What has and has not already been run here

| | status |
|---|---|
| upstream `preprocess.py` | **run**, 2026-08-25, PPG-DaLiA only (`outputs/hydra/preprocess_20260825_144903` @ 4 s, `…_151003` @ 8 s) |
| upstream `train.py`, any dataset | **never run** |
| upstream `train.py` on the five non-DaLiA datasets | **never run** |
| PENGUIN checkpoint or training log shipped with the release | **none** (`ckpt/` holds only `PaPaGei_S.pt`) — `PENGUIN_AUDIT.md` §38 |

Arm A0 (`docs/A0_PENGUIN_REPRODUCTION_REPORT.md`, 2026-08-25) is **not** an execution of upstream
`train.py`. It instantiated upstream's unmodified `PENGUIN` model class and its `train_flow` / `optimize`
methods inside **this project's own training loop**, on **this project's own deterministic split**, at
`segment_len = 8`. That report states it directly: *"`train.py` with its glob-order split was not run."*
Its headline numbers — HR error 10.99 bpm (ours, corrected), 11.74 (upstream metric, corrected),
25.14 (upstream metric as shipped, diagnostic only) against the paper's 15.64 — are therefore evidence
about the released *architecture and objective*, not about the released *pipeline*.

U1 runs the released pipeline: upstream's `preprocess.py` and `train.py`, upstream's split, upstream's
early stopping, upstream's metrics, at upstream's shipped `segment_len: 4`. Nothing of ours is in the loop.

A0's 8 s deviation was principled: at `segment_len: 8` the shipped `HeartRateError` resamples an 8 s window
to `128 * 4 = 512` samples and so compresses time 2× (`PENGUIN_AUDIT.md` §34), and `train.py:43`'s
`assert window_size % segment_len == 0` makes the 60 s respiration window impossible. **At the shipped
`segment_len: 4` neither problem arises**: `RR_seqlen = 128 * 4` equals the true window length and
`60 % 4 == 0`. The shipped configuration is internally consistent and is what U1 runs.

## 3. Design

For each of the six shipped datasets, in this order:

1. `external/PENGUIN/src/preprocess.py` → `data/processed/upstream_u1/<DATASET>/subject*.pkl`
2. `external/PENGUIN/src/train.py` → train, validate, early-stop, test — all inside upstream's own loop
3. record verbatim everything upstream prints, plus the realised split and window counts

Upstream code is executed unmodified. Hydra's run dir is redirected out of `external/PENGUIN` and
`PYTHONDONTWRITEBYTECODE=1` is set so the pinned tree stays clean; both are verified by
`git -C external/PENGUIN status --porcelain` being empty before and after every dataset.

### 3.1 Config: shipped defaults

Taken from `external/PENGUIN/config/` unchanged: `segment_len: 4`, `resample_rate: 128`, PPG band-pass
[0.5, 4] + z-score + min-max, per-dataset label filtering as shipped (ECG high-pass 0.5; Resp low-pass 1;
ABP unfiltered and un-normalised), model `PENGUIN` (`n_step: 25` → Heun 50 NFE), `lr 1e-3`,
AdamW `weight_decay 0.01`, `batch_size 64`, `epoch_num 300`, `earlystop_metric mae`,
`earlystop_patience 10`, `monitor_val true`, `fold_num 8`, `seed 42`.

### 3.2 Overrides — exhaustive list, fixed now

| # | override | reason | class |
|---|---|---|---|
| 1 | `preprocess.rawdata_path`, `preprocess.procdata_path`, `train.logging.log_path` → absolute paths | upstream defaults are relative to a cwd that would write inside the pinned tree | mechanical |
| 2 | config key `preprocess.DaLiA` → `preprocess.PPG-DaLiA` | upstream `train.py:32` does `getattr(cfg.preprocess, "PPG-DaLiA")` while `preprocess.yaml:29` defines `DaLiA`; **the release cannot run PPG-DaLiA without this rename** | required bug fix |
| 3 | WildPPG only: `rawdata_path` points at a 14-subject symlink view; `preprocess.WildPPG.subject_num=14` | this project's absolute rule — WildPPG subjects `kjd` and `ssx` are never loaded. Upstream globs all 16 `.mat` files and would load them | firewall (deviation) |
| 4 | Tier B only: `train.training.epoch_num` capped (§3.4) | wall-clock budget, fixed before any run | budget (deviation) |
| 5 | `train.logging.description=u1` | output directory naming | mechanical |

No other value is changed. Any further override discovered to be necessary is appended to §7 as a
deviation **before** the affected dataset is trained, never after.

### 3.3 Split

Upstream's own: `load_dataset_path` shuffles the `glob` of subject files with `random.sample` after
`fix_seed(42)`, then takes `val = files[:subject_num//8]`, `test = files[subject_num//8 : 2*subject_num//8]`,
train = the rest. The glob order is filesystem-dependent, so the realised split is **recorded** (subject
index → file → train/val/test) for every dataset, but it is not chosen by us and not fixed across datasets.
This is the split the released code performs; reproducing it is part of the question.

### 3.4 Tiers and epoch caps

Cost is dominated by validation, which runs the full 50-NFE Heun sampler on `subject_num//8` subjects every
epoch. Estimated from A0's measured rates (4.85 ms/window train, 66 ms/window val at 1024 samples; halved
at 512):

| dataset | ≈ windows @4 s | ≈ train | ≈ val | ≈ epoch | tier | `epoch_num` |
|---|---|---|---|---|---|---|
| BIDMC | 6.4 k | 4.9 k | 0.7 k | 0.6 min | **A** | 300 (shipped) |
| WESAD | 21.7 k | 18.8 k | 1.4 k | 1.6 min | **A** | 300 (shipped) |
| PPG-DaLiA | 32.4 k | 28.0 k | 2.2 k | 2.3 min | **A** | 300 (shipped) |
| MIMIC-BP | 274 k | 206 k | 34 k | 27 min | **B** | **20** |
| WildPPG | 780 k | 669 k | 56 k | 57 min | **B** | **12** |
| UCI-BP | 712 k | 534 k | 89 k | 70 min | **B** | **12** |

Tier A runs the shipped schedule to its own early stop, uncapped. Tier B caps `epoch_num` purely on
wall-clock (~9–14 h per dataset, ~36 h total); `earlystop_patience 10` is unchanged and may still fire
first. The caps are set **now**, from window counts and A0 timings only — no metric of any kind enters
them. A Tier-B cap is never raised or lowered after a run.

Note that the cap is not restrictive in optimisation terms: 12 epochs of UCI-BP is ≈ 100 k optimizer
steps against A0's entire 21-epoch PPG-DaLiA run of ≈ 4.6 k. If a Tier-B run is still improving at its
cap this is reported as a limitation, not repaired.

## 4. Endpoints

**Primary — exactly what upstream prints for the test split**, no post-processing:

| dataset | upstream metric | published value (Table 1) |
|---|---|---|
| PPG-DaLiA | `HeartRateError` | 15.64 bpm |
| WildPPG | `HeartRateError` | 12.97 bpm |
| BIDMC | `RespRateError` | 2.98 bpm |
| WESAD | `RespRateError` | 4.45 bpm |
| UCI-BP | `SBPError` / `DBPError` | 12.61 / 7.14 mmHg |
| MIMIC-BP | `SBPError` / `DBPError` | 17.43 / 11.34 mmHg |

Secondary, reported alongside and never substituted for the primary: upstream's `MAE`, its inference time,
the realised split, the realised window counts, epochs run, best epoch.

## 5. Decision rule (fixed now)

Per dataset-metric, on the ratio `r = ours / published`:

| verdict | rule |
|---|---|
| **REPRODUCED** | `0.80 ≤ r ≤ 1.25` |
| **PARTIAL** | `0.60 ≤ r < 0.80` or `1.25 < r ≤ 1.67` |
| **NOT REPRODUCED** | otherwise |

A ratio below 1 (we beat the paper) is *not* a success — it is as much a replication failure as a ratio
above 1, and is reported as such. The stage-level verdict is the count of each class; there is no
aggregate score and no metric is dropped.

**This rule cannot resolve the published numbers' provenance.** `PENGUIN_AUDIT.md` §38 records that no
checkpoint, log or split accompanies the release, and §22 that the code's 13/1/1 single-hold-out split
contradicts the paper's stated 6:1:1. A single-run match or mismatch is therefore evidence about
*the released code under its own defaults*, not proof about the paper's table. Stated here so it cannot
be claimed afterwards.

## 6. Pre-registered observations to check (predictions, not results)

Recorded before running so that confirmation cannot be presented as discovery.

- **U1-P1 — UCI-BP subject duplication.** `load_data.py:100-102` computes
  `part = sub_idx // 4 + 1` but `onset = sub_idx % 2 * 1500`. For `subject_num = 8` this yields
  parts `1,1,1,1,2,2,2,2` and onsets `0,1500,0,1500,0,1500,0,1500`, so subject 0 ≡ subject 2,
  1 ≡ 3, 4 ≡ 6, 5 ≡ 7 — **each of the 8 "subjects" is an exact duplicate of another**, and `Part_3.mat`
  / `Part_4.mat` are never read. Under upstream's random 6/1/1 split the test subject's duplicate lands
  in train with probability 6/7. Prediction: pairwise sha256 of the produced `subject*.pkl` files will
  show four identical pairs. This is verified from file hashes only — a data-level check, not a metric.
- **U1-P2 — `HeartRateError` zero-fill.** `help_func.py:184` returns `0.0` when no window has a valid
  predicted *and* target HR, biasing the mean downward. Not repairable without modifying upstream; its
  presence is noted with the result.
- **U1-P3 — window-length ambiguity is now decided in the code's favour.** Every dataset's shipped
  `sample_num` equals `duration * 3600 / 8`, i.e. an 8 s count, while `segment_len` is 4. U1 runs 4 s.
  The realised window counts are reported against `sample_num` so the factor is visible.

## 7. Deviations log (append-only)

| # | deviation | reason |
|---|---|---|
| D1 | WildPPG uses 14 of 16 subjects (`kjd`, `ssx` excluded), so `val = test = 14//8 = 1` subject instead of 2 | project-absolute firewall on the WildPPG test subjects |
| D2 | Tier-B `epoch_num` capped at 20 / 12 / 12 (MIMIC-BP / WildPPG / UCI-BP) | wall-clock budget, §3.4 |
| D3 | `preprocess.DaLiA` renamed to `preprocess.PPG-DaLiA` | the release does not run otherwise |

## 8. Firewall

- `external/PENGUIN` is read-only for this stage; a non-empty `git -C external/PENGUIN status --porcelain`
  aborts the run.
- WildPPG `kjd` / `ssx` are excluded at the filesystem level (symlink view) **and** asserted absent from
  the realised train/val/test lists before training starts. A violation aborts and deletes the outputs.
- No checkpoint, prediction, `.pkl`, `.npy` or figure produced by this stage enters git.
- This stage trains only upstream's model. It does not touch, retrain or re-select any of our arms, and
  the frozen A4/arm-U checkpoint md5 `31c042d291052fbb6dc15263ad316be2` is unchanged by construction.
- C2 training remains deferred and is not part of U1.

## 9. Reporting

`docs/U1_UPSTREAM_PENGUIN_ASSHIPPED_REPORT.md`: the six-row primary table with verdicts, the realised
splits and window counts, the raw upstream stdout excerpt per dataset, the U1-P1/P2/P3 outcomes, and the
deviations. Failures and aborted datasets are reported, not omitted.
