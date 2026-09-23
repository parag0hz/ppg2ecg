# RDDM R0 — Reproduction of the released checkpoint: **REPRODUCTION PARTIAL**; reduced-step sampling is **not supported** by the released model

Scope, stated first: this is **"reproduced inference on the publicly released checkpoint under the available evaluation
setting"**, not a reproduction of the paper's held-out numbers. RDDM's test split is unpublished, so every evaluation
window below comes from corpora in which ≈ 80 % of the subjects were RDDM's training subjects. No weight was changed;
no upstream line was edited.

## 1. Environment (recorded)
| item | value |
|---|---|
| Official repository | https://github.com/DebadityaQU/RDDM, pinned as submodule `external/RDDM` @ `7d5348843c3985c211a23ae5105a2d9497d5156a`, **unmodified** (`git status` clean) |
| Python / torch / CUDA / GPU | 3.13.9 / 2.11.0+cu130 / 13.0 / NVIDIA RTX 5090 (32 GB) |
| Extra packages | `similaritymeasures 1.4.0`, `torchmetrics 1.9.0`, `lightning-utilities 0.15.3` (imported by the official `metrics.py`) and `gdown 6.4.0` (download), installed with `--no-deps` into `outputs/rddm_env/pydeps` and used only through `PYTHONPATH`; `biosppy 2.1.2`, `neurokit2 0.2.12`, `scikit-learn 1.7.2` already in the project environment. No requirements file ships with RDDM. |
| Checkpoint source | Google Drive folder `1Z7JQ5VdTrekx4lbARJNIUiR5D-4Kz7wg`, linked from the official README (three files, dated 2024-03-27) |
| sha256 | `rddm_main_network.pth` `e90490a9039cdf147be29404be94d4b04bfcccc3c0d94ee009bbe42f2c040eb9` (366.8 MB) · `rddm_condition_encoder_1.pth` `9f8ef5d758962030958674526d450b41bc188e27b1edf2c3fe5c1a4bfe067223` · `rddm_condition_encoder_2.pth` `3c7cbbc1585395381faf20489bd7d9091aafc9d3cbddde67e92cc3d050f6c3bd` (107.8 MB each) — stored in `data/pretrained/rddm/` (gitignored) |
| Parameters | eps_model 45,828,129 · region_model 45,828,129 · condition encoders 26,926,016 × 2 · **total 145,508,290** |
| Commands | inputs `scripts/rddm_build_inputs.py`; official evaluator `PYTHONPATH=external/RDDM:outputs/rddm_env/pydeps .venv/bin/python scripts/rddm_official_eval.py`; per-window driver `… scripts/rddm_driver.py r0` / `diag` |
| Only non-code accommodation | the official `std_eval.eval_diffusion` reads data from the hard-coded relative path `../../ingenuity_NAS/.../datasets/`; the runner creates that relative layout as a **symlink** to our reconstructed inputs and sets the working directory so it resolves. |

## 2. Checkpoint provenance and split status
- **Training data (paper §4.1):** WESAD, DALIA, CAPNO, BIDMC and MIMIC-AFib combined; "data from 80 % of the subjects for
  training and the remaining 20 % … for cross-subject evaluation"; 500 epochs, batch 512, 4 × A100, β ∈ (1e−4, 0.2), T = 10.
- **Split: unpublished and unrecoverable.** Neither subject lists nor the `.npy` files exist in the repository, the paper,
  or the CardioGAN repository RDDM says it follows. **Consequence:** every window evaluated here belongs to a corpus in
  which RDDM trained on ≈ 80 % of the subjects; which windows are held out is unknown. MIMIC-AFib is not in hand (only its
  index CSV), so 4 of the paper's 5 corpora are evaluated.
- **Preprocessing (our reconstruction; the `.npy` build code is not released):** paper text followed literally —
  resample to 128 Hz (polyphase), ECG Butterworth high-pass 0.5 Hz and PPG Butterworth band-pass 0.5–8 Hz via neurokit
  (`ecg_clean(method="neurokit")`, order 5 + powerline; `ppg_clean(method="elgendi")`, order 2 — the paper cites neurokit),
  subject z-score, non-overlapping 4 s / 8 s windows from t = 0, non-finite / constant windows dropped. RDDM's own
  `get_datasets` then applies its per-window min-max to [−1, 1], `ppg_clean`, and `ecg_clean(pantompkins1985)` — so the
  **reference ECG in every metric is RDDM's Pan-Tompkins-cleaned target**, as in the paper's code.
  Windows: WESAD 15 subjects / 21,711 × 4 s; CAPNO 42 / 5,040; DALIA 15 / 32,368; BIDMC 51 / 6,120.
- **Synchronisation:** taken as shipped (common t = 0 per record). DaLiA / WESAD wrist-PPG vs chest-ECG alignment is only
  second-level (this project's earlier measurement), for RDDM exactly as for every other model.

## 3. Original sampler and NFE (verified)
Official `RDDM.forward(mode="sample")`, ancestral DDPM, **T = 10** steps, linear β from 1e−4 to 0.2 (ᾱ_T = 0.305).
Forward hooks on the two denoisers (no code change) count, per sampling pass, **10 `region_model` + 10 `eps_model` calls
= 20 NFE per sample**, plus **one pass of each condition encoder per window**. Stochastic: `x_T ~ N(0, I)` and fresh
noise at every step. GPU throughput 3.8–3.9 ms/window at batch 512; batch-1 latency **112 ms (GPU), 408 ms (CPU, 4
threads)** for one full 20-NFE sample.

## 4. Published vs reproduced
**Official evaluator, unmodified** (`std_eval.eval_diffusion`, shuffled batches of 512, seed 31). Published = paper
Tables 2–3; "repo log" = `eval-logs/standard_eval_RDDM.out` (the authors' own rerun on their split).

| corpus | metric | published | repo log | **our reproduction** | ours − published |
|---|---|---|---|---|---|
| WESAD | RMSE (4 s) | 0.21 | 0.2092 | **0.2344** | +0.024 (+12 %) |
| WESAD | FD (4 s) | 3.93 | 3.86 | **3.975** | +0.05 (+1 %) |
| WESAD | HR MAE, bpm (8 s, Hamilton) | 1.40 | 1.353 | **2.604** | **+1.20 (+86 %)** |
| CAPNO | RMSE | 0.19 | 0.1947 | **0.1848** | −0.005 (−3 %) |
| CAPNO | FD | 2.94 | 2.96 | **3.526** | +0.59 (+20 %) |
| DALIA | RMSE | 0.25 | 0.2507 | **0.2199** | −0.030 (−12 %) |
| DALIA | FD | 5.84 | 5.85 | **4.349** | −1.49 (−26 %) |
| DALIA | HR MAE (8 s) | 4.49 | 4.577 | **3.242** | −1.25 (−28 %) |
| BIDMC | RMSE | 0.24 | 0.2355 | **0.2105** | −0.030 (−12 %) |
| BIDMC | FD | 6.72 | 6.71 | **4.733** | −1.99 (−30 %) |
| MIMIC-AFib | RMSE / FD | 0.22 / 6.71 | 0.2229 / 6.75 | not run (data not in hand) | — |

**Our driver (independent draw, fixed batch order, per-window arrays saved)** agrees with the official evaluator to
within sampling noise (RMSE identical to 3 decimals; Hamilton HR 2.64 / 2.59 / 3.22 / 1.51 vs 2.60 / 2.55 / 3.24 / 1.50;
FD differs by batch composition). Our standard metrics on the same samples (4 s windows, neurokit detector,
subject-clustered 95 % CI):

| corpus | subjects / windows | HR error (bpm) | R-peak F1 @ 50 ms | RR-MAE (ms) | waveform MAE |
|---|---|---|---|---|---|
| WESAD | 15 / 21,711 | 2.80 [2.35, 3.35] | 0.611 [0.527, 0.671] | 23.6 [22.3, 24.9] | 0.117 |
| CAPNO | 42 / 5,040 | 2.30 [1.35, 3.57] | 0.824 [0.742, 0.896] | 15.7 [14.4, 17.1] | 0.091 |
| DALIA | 15 / 32,368 | 3.25 [2.67, 4.00] | 0.663 [0.548, 0.753] | 20.7 [18.4, 22.5] | 0.125 |
| BIDMC | 51 / 6,120 | 1.38 [1.14, 1.68] | 0.844 [0.783, 0.895] | 14.0 [12.9, 15.2] | 0.112 |

These are **not comparable** to this project's other tables: the reference here is a Pan-Tompkins-filtered ECG, the
windows include RDDM's training subjects, and the preprocessing differs.

## 5. Remaining mismatch
- **Direction is mixed.** DALIA and BIDMC come out *better* than published on RMSE, FD and (DALIA) HR; WESAD HR comes out
  1.9 × *worse*; CAPNO FD 20 % worse. Uniformly better numbers would be the expected signature of evaluating mostly on
  training subjects; the mixed pattern means the reconstruction is not exactly RDDM's pipeline, the subject mix matters,
  or both.
- **WESAD HR (+1.20 bpm) — cause not identified.** Candidates, none verified: our WESAD wrist-BVP / chest-ECG alignment,
  the unknown 3 held-out WESAD subjects being easier than the 15-subject average, unspecified preprocessing details
  (resampler, filter order before z-score). The per-subject arrays are saved (`outputs/rddm_raw/r0_*.npz`) if this needs
  chasing.
- FD depends on the shuffled batch composition in the official code (our fixed-order driver gives 4.21 vs 3.98 on WESAD
  from the same kind of samples), so FD differences of this size are partly definitional.
- The paper's held-out numbers **cannot** be reproduced without the split; no claim is made that they were.

## 6. Is reduced-step sampling technically justified? — **No**
Three independent reasons, the last one empirical:
1. **Training:** the weights were trained with the 10-step linear-β schedule (`train.py` builds `RDDM(n_T=nT)`; training
   draws `t ∈ {1, …, 9}` and conditions on `t / 10`).
2. **Paper:** RDDM is reported only at T = 10; the T = 25 / 50 rows are separately trained DDPM baselines.
3. **Code (diagnostic, `artifacts/rddm_r0/invalid_schedule_diagnostic.json`):** calling the official loader with
   `nT = 5` **fails**: `size mismatch for alpha_t: copying a param with shape torch.Size([11]) from checkpoint, the shape in
   current model is torch.Size([6])` (same for `alphabar_t`, `sqrtab`, `sqrtmab`, …). The 10-step schedule is stored
   **inside the released checkpoint** as registered buffers and loaded strictly. Any shallower RDDM sampler would require
   editing the loader or writing a respaced sampler — a new inference algorithm, which this project's rules exclude.

**Category, stated precisely: "sampler does not support shallow inference".** This is *not* evidence that the
allocation principle fails to transfer; the principle simply cannot be tested on this checkpoint at a fixed budget.

## 7. Can the width/depth experiment proceed on RDDM?
| question | feasible with the released checkpoint, no retraining, no new sampler? |
|---|---|
| Fixed-budget depth vs width (B = 20 NFE, several (K, S)) | **No** — only one valid depth (T = 10) |
| Same-checkpoint **width at the original depth** (K = 1, 2, 4, 8, 16 samples × 20 NFE; budget grows with K) | **Yes** |
| Functional-error dependence ρ̄ and consensus gain G at that single depth | **Yes** (one depth per corpus; no depth ordering possible) |
| RDDM's original downstream functional (Hamilton HR on 8 s windows) with median pooling | **Yes** |
| RDDM's CardioBench tasks (AFib, BP, stress, diabetes) | **No** — CardioBench code not released; not re-implemented |
| A held-out evaluation set | **Not among RDDM's corpora** (split unknown). The only held-out-by-construction data in hand is **VitalDB V1 test** (1,156 patients; RDDM never trained on VitalDB) — a zero-shot domain transfer, not a reproduction setting |

## Verdict: **REPRODUCTION PARTIAL**
The released checkpoint runs through the official, unmodified code; NFE and sampler are verified; 4 of 4 RMSE values are
within 12 % of published and FD / HR land in the published range with mixed direction; WESAD HR does not reproduce
(+86 %). The paper's held-out numbers are not reproducible because the split is unpublished, and all evaluated windows
include RDDM's training subjects.
