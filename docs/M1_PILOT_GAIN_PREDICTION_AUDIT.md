# M1 — Pilot→future-gain prediction: repository / data audit (M0)

Read-only audit. **No new test result was computed**: only file schemas, shapes, split metadata, finite-value counts and
checkpoint fingerprints. Head commit at audit time: `b1eb66a` (2026-09-23 22:48:21).

## 1. Question this audit must answer

M1 asks whether observable statistics of a *pilot* set of stochastic samples predict the consensus gain obtainable from
*disjoint future* samples of the same window. That requires, per (model, depth) condition:

- K ≥ 16 independent stochastic samples with the functional (HR) extracted per sample,
- exact window ↔ patient ↔ reference-HR alignment,
- **a VitalDB validation bank to fit on and a VitalDB test bank to evaluate once**, with no fitting on test.

## 2. Splits — found, not assumed

`data/manifests/split_v1_vitaldb_seed42.json`, protocol `V1-patient-holdout`, split seed **20260917**, rule documented in
`docs/V1_VITALDB_PAIRED_PREREGISTRATION.md`: eligible cases sorted numerically → `default_rng(20260917).permutation`;
test = first round(0.20·n) patients, **val = next round(0.05·n)**, train = the rest; all cases of a patient move together.

| split | cases | patients | windows | processed on disk |
|---|---:|---:|---:|---|
| train | 4,517 | 4,337 | 288,400 | 4,517 / 4,517 |
| **val** | **302** | **289** | **4,822** | **302 / 302** |
| **test** | **1,224** | **1,156** | **19,543** | **1,224 / 1,224** |

Windows are 4 s @ 128 Hz, D1 VitalDB preprocessing, drop rule applied target-side only. Loader
`scripts/vm1_evaluate.py::load(role)` concatenates `data/processed/v1_vitaldb/<case>.npz` (`x`, `y`, `window_index`,
`subjectid`) in manifest order; verified this audit: `load("val")` → X (4822, 512) float32, Y (4822, 512) float64,
pid (4822,) with 289 unique patients; `load("test")` → 19,543 windows / 1,156 patients. **Patient IDs are available for
both splits**, so patient-level grouping and patient-clustered bootstrap are possible on each.

## 3. Frozen checkpoints (seed 42, as used by DW1 / DW2 / EXP-B)

| model | arm | path (`checkpoint_last.pt`) | sha256 (first 16) |
|---|---|---|---|
| iMF | I | `outputs/v1_vitaldb_armI_seed42/` | `ddbc39e66f8a71c7` |
| consistency distillation (CD) | D | `outputs/cd1_vitaldb_armD_seed42/` | `faea968bf297e81f` |
| PENGUIN (Euler) | C | `outputs/v1_vitaldb_armC_seed42/` | `05c9748b6629f33d` |

Samplers are defined once in `scripts/dw1_depth_width.py::make_sampler`: iMF = uniform MeanFlow schedule `[1/S]·S`;
CD = multistep consistency with re-noising at `t = i/S`; PENGUIN = forward Euler, NFE = S.

## 4. HR functional (exact implementation used by DW/EXP-B)

`scripts/v1_evaluate.py`:

```python
def _hr(row):  return R.hr_bpm(R.detect_rpeaks(row, FS, "neurokit"), FS)      # FS = 128
def hr_batch(ex, Z):  return np.array(list(ex.map(_hr, list(Z), chunksize=256)))
```

Reference HR `y_i` is the same functional applied to the target ECG `Y`. The test reference is cached in
`outputs/sr1_eval/arm_{I,C,D}_seed42.npz` as `ref_hr` (+ `pid`); verified this audit: (19543,) float64, **100 % finite**,
and **bit-identical across the three arms**. No validation reference-HR cache exists; it is reproduced deterministically
by `hr_batch` on `load("val")[1]`.

## 5. Sample banks — what exists

Contents are **extracted HR only** (per-sample bpm matrices of shape `(K, n_windows)`), except a waveform subset
`outputs/tt_expb_raw/wave_{I,D,C}_seed42_S{1,2,4,8}.npy` of shape (16, 2000, 512) float16 on the fixed 2,000-window
`linspace` test subset. No generated waveforms are stored for the full test set and none for validation.

Sources: `outputs/dw1_raw/hr_{I,D,C}{S}.npy` (K = 32/S, float32) and `outputs/tt_expb_raw/hr_{I,D,C}_seed42_S{2,4,8}.npy`
(K = 16 or 32, float64). Where both exist the larger K is used. Multiseed banks (`hr_I_seed{1,2}_S2`, `hr_C_seed{1,2}_S4`)
exist for training seeds 1/2 but are **out of scope for M1** (M1 fixes the seed-42 checkpoints).

### Required table

| Model | Split | S | Available K | Same seeds across S? | HR available? | Waveforms available? | Usable |
|---|---|---:|---:|---|---|---|---|
| iMF | test | 1 | 32 | yes (shared `z0`) | yes | subset only (2,000 win) | **yes** |
| iMF | test | 2 | 32 | yes | yes | subset only | **yes** |
| iMF | test | 4 | 16 | yes | yes | subset only | **yes** |
| iMF | test | 8 | 16 | yes | yes | subset only | **yes** |
| iMF | test | 16 | 2 | yes | yes | no | no (K < 16) |
| iMF | test | 32 | 1 | yes | yes | no | no (K < 16) |
| CD | test | 1 | 32 | yes (shared `z0`; re-noise prefix) | yes | subset only | **yes** |
| CD | test | 2 | 16 | yes | yes | subset only | **yes** |
| CD | test | 4 | 16 | yes | yes | subset only | **yes** |
| CD | test | 8 | 16 | yes | yes | subset only | **yes** |
| CD | test | 16 | 2 | yes | yes | no | no (K < 16) |
| CD | test | 32 | 1 | yes | yes | no | no (K < 16) |
| PENGUIN | test | 1 | 32 | yes (shared `z0`) | yes | subset only | **yes** |
| PENGUIN | test | 2 | 16 | yes | yes | subset only | **yes** |
| PENGUIN | test | 4 | 32 | yes | yes | subset only | **yes** |
| PENGUIN | test | 8 | 16 | yes | yes | subset only | **yes** |
| PENGUIN | test | 16 | 2 | yes | yes | no | no (K < 16) |
| PENGUIN | test | 32 | 1 | yes | yes | no | no (K < 16) |
| iMF / CD / PENGUIN | **val** | 1, 2, 4, 8 | **0** | — | **no** | no | **not yet — must be generated (§7)** |

Finite-coverage of the first 16 draws on test (windows where all 16 sample HRs **and** the reference are finite; this is
the Ω_HR rule of EXP-B, reported here as data-quality metadata):

| model | S=1 | S=2 | S=4 | S=8 |
|---|---:|---:|---:|---:|
| iMF | 18,171 | 17,820 | 17,139 | 16,639 |
| CD | 17,839 | 18,910 | 18,402 | 18,457 |
| PENGUIN | 16,797 | 17,631 | 18,016 | 19,116 |

(out of 19,543; 85–98 %.)

## 6. Seed alignment across depths

`make_sampler`'s draw is `g = torch.Generator().manual_seed(seed)`, `z0 = torch.randn(len(X), 1, W, generator=g)` —
**`z0` depends on the seed and the array shape only, not on S**. Therefore sample k at depth S and at depth 2S start from
the identical initial noise for all three arms, and the depth-probe family `D_k = Y_{i,2S,k} − Y_{i,S,k}` is well defined
on paired seeds. Caveat to record in the preregistration: for **CD** the sampler additionally draws `S − 1` re-noising
tensors from the same generator stream after `z0`, so S and 2S share only the first `S − 1` of them; the pairing is exact
at `z0` and prefix-shared afterwards. For iMF and PENGUIN no extra draws exist and the pairing is exact.

## 7. Validation bank — does not exist; generation plan (to be executed only after preregistration)

No validation-split sample bank exists anywhere under `outputs/`. Fitting any predictor on test is therefore forbidden,
and the primary required condition is met **only after** a validation bank is generated. It can be generated cleanly with
already-frozen components — no retraining, no new preprocessing, no new functional:

| component | frozen source |
|---|---|
| windows | `vm1_evaluate.load("val")` → 4,822 windows / 289 patients (already preprocessed on disk) |
| checkpoints | the three seed-42 `checkpoint_last.pt` of §3, unchanged |
| samplers | `dw1_depth_width.make_sampler`, unchanged |
| depths | S ∈ {1, 2, 4, 8} (the depths where test has K ≥ 16) |
| draws | noise seeds **0 … 15** (the project convention "seeds 0 … K−1"; identical seeds across S) |
| functional | `v1_evaluate.hr_batch` / `_hr` (neurokit, 128 Hz), reference = same functional on `Y` |
| output | `outputs/m1_pilot_gain_prediction/val_hr_{arm}_S{S}.npy` (16, 4822) + `val_ref.npz` (`ref_hr`, `pid`) |

Cost: 4,822 windows × 16 draws × (1+2+4+8) NFE = **1,157,280 network evaluations per model**, 3.47 M total — 4.05× smaller
than the existing test banks, which were produced on this GPU in ~1 h for the largest single cell. Estimated wall-clock
≈ 30 min/model on the idle RTX 5090 (currently 191 MiB used), plus neurokit HR extraction (CPU pool) on 3 × 4 × 16 × 4,822
generated windows. For the optional depth-probe family, S = 16 at K = 16 would add 4,822 × 16 × 16 = 1.23 M NFE per model;
it is *not* required for the primary analysis (the pairs (1,2), (2,4), (4,8) already exist within S ∈ {1,2,4,8}).

**No fitting of any kind happens on test.** The validation bank is generated once, before the predictor is fit, and the
test banks above are read exactly once at evaluation time.

## 8. Status against the primary required condition

| requirement | status |
|---|---|
| VitalDB validation patients | ✅ 289 patients / 4,822 windows, processed and loadable |
| VitalDB test patients | ✅ 1,156 patients / 19,543 windows |
| K ≥ 16 at one or more valid depths | ✅ test: all three models at S ∈ {1, 2, 4, 8}; val: after §7 generation |
| exact window / patient / reference alignment | ✅ single loader order; `pid` and `ref_hr` aligned and verified |
| validation bank exists | ❌ **must be generated per §7** (documented before generation, as required) |

**Verdict: PROCEED.** No hard stop. The only missing element is the validation bank, which is reproducible from frozen
components under a documented plan; M1 may continue once that plan is preregistered and executed.

## 9. Items carried into the preregistration

1. Conditions: 3 models × S ∈ {1, 2, 4, 8} = 12, seed-42 checkpoints, both splits.
2. Ω_HR rule: keep windows where the reference HR and all 16 used draws are finite (EXP-B's rule, unchanged).
3. First 16 draws (seeds 0–15) are the M1 bank even where K = 32 exists, so every condition is treated identically.
4. Depth-probe family is **secondary** and, for CD, is prefix-paired rather than fully paired (§6).
5. Sanity check needs a validation-defined constant-HR baseline; the project convention is the training/validation-median
   constant (EXP-D Part B), and `docs/DB1_DISCRIMINATIVE_HR_BASELINE_REPORT.md` is the existing informativeness context
   for VitalDB HR (consensus 6.2–6.7 bpm; PPG peak counting 9.06; direct regressor 5.69).
6. No preregistered practical gain threshold τ for HR consensus gain exists anywhere in the project; per the M1 brief the
   sign-decision metric is therefore **omitted**, not invented.
