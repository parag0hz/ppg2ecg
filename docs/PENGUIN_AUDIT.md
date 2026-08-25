# PENGUIN Source Audit

Upstream: `https://github.com/Neurogica/PENGUIN` → `external/PENGUIN/`
**commit `6cd70cdefb91f10efeb8dce34019b5067cb25344`** (short `6cd70cd`), branch `main`, last commit 2025-09-18 "[alt] edit README.md",
**working tree clean (0 changed files)** at clone time (2026-08-25) and after every script in this repo
(`ppg2ecg.utils.upstream.assert_upstream_pinned()` enforces this). Licence BSD-3-Clause-Clear. 26 files; the PENGUIN path is
`config/*.yaml`, `src/train.py`, `src/preprocess.py`, `src/utils/{help_func,load_data}.py`, `src/models/PENGUIN.py`,
`src/models/layers/{S5,S5_init,S5_jax_compat}.py`. Other baselines (RDDM, RespDiff, CycleGAN, PaPaGei-S) are out of scope.

Method: every item below was read in the code (file:line), and where possible **executed** in this environment
(Hydra dry-run, upstream preprocessing on the real data, parity tests, GPU smoke test). Sections marked ⚠ are
findings that matter for reproduction. An independent multi-agent adversarial verification pass is summarised in §22.

## 1. ECG training entry point
- `src/train.py` — `@hydra.main(config_path="../config/", config_name="config.yaml")` → `main()` → `summarize()` (thop FLOPs)
  → `validation()` (train + val + test). Command: `python ./src/train.py train.dataset=PPG-DaLiA train.model=PENGUIN`
  run **from `src/`** (imports are `from utils.help_func …`, `from models.layers.S5 …`, i.e. `src/` must be the cwd / on `sys.path`).
- ⚠ README says `./src/training.py` — the file is `train.py` (doc-code mismatch).
- ⚠ `config/config.yaml` defaults `[_self_, preprocess, train, model]`; `config/preprocess.yaml` has the dataset block under the key
  **`DaLiA`**, while `train.py:32,118` and `preprocess.py:70` do `getattr(cfg.preprocess, cfg.train.dataset)` with `"PPG-DaLiA"`.
  **Verified by Hydra compose: `ConfigAttributeError: Key 'PPG-DaLiA' is not in struct`** for both `preprocess.py` and `train.py`.
  The shipped repo cannot run the PPG-DaLiA experiment as-is. Our fix is a patched *copy* (`configs/upstream/`, 3 documented patches).
- `wandb` is imported at module top even when `train.logging.wandb: false`; `thop` is required by `summarize()`.
- Log: `{log_path}/{model}_{dataset}_{description}/output.log` (overwritten each run), plots in `/plot`, checkpoint in `/ckpt/pretrain_ckpt.pth`.

## 2. Dataset loader
- `src/utils/load_data.py:80-95` `load_PPG_DaLiA(sub_idx, cfg, dataset_cfg)`: opens
  `{rawdata_path}/PPG-DaLiA/PPG_FieldStudy/S{i+1}/S{i+1}.pkl` (`pickle`, `encoding="latin1"`), takes
  `signal.chest.ECG` (700 Hz) and `signal.wrist.BVP` (64 Hz), windows both with
  `sliding_window_view(x, fs*segment_len)[::fs*segment_len]` (non-overlapping, trailing samples dropped). Nothing else from
  the pickle (activity labels, dataset `rpeaks`, HR labels) is used.
- `src/utils/help_func.py:222-267` `PPGDataset`: loads *all* per-subject pickles into two float32 tensors in RAM, `__getitem__`
  moves each item to GPU (`.to(device)`), `DataLoader(num_workers=0)`, `shuffle=True` for train only. `peak_roi` is only
  built for RDDM. No augmentation.
- ⚠ Upstream never checks that PPG and ECG produce the same number of windows. On the real data it happens to hold
  (durations agree to the second for all 15 subjects — `scripts/verify_dalia.py`).

## 3. PPG-DaLiA preprocessing (`src/preprocess.py:14-62`, driven by `config/preprocess.yaml`)
Per subject, per 4 s window, in this order:
1. `scipy.signal.resample(x, 128*4=512, axis=1)` — FFT resampling of each isolated window (PPG 256→512 up, ECG 2800→512 down).
2. Butterworth order 4 + `filtfilt` (zero-phase) on the 512-sample window: PPG **band-pass 0.5–4 Hz**; ECG `label_freq_range: [0.5, -1]`
   → **high-pass 0.5 Hz** (the `-1` selects the high-pass branch, L45-46).
3. `scipy.stats.zscore(axis=1)` — per window.
4. min-max to **[-1, 1]** per window (`(x-min)/(max-min+1e-8)*2-1`).
Output: `procdata_path/PPG-DaLiA/subject{i}.pkl` with `x_data`, `y_data` float64 `[n, 512]`.
- Executed here: 15 subjects, 39 s, **32,368 windows** total. Our re-statement `ppg2ecg.data.preprocess` is bit-exact against it.
- ⚠ Edge effects: 0.5 Hz high-pass on a 4 s window and FFT resampling assume periodic extension; both are per-window, so
  they are also applied at test time identically. Part of the protocol; not changed in v0.

## 4. Sampling rate — 128 Hz after resampling (`resample_rate: 128`); native 64 Hz BVP / 700 Hz ECG.

## 5. Window length — `segment_len: 4` s → 512 samples, stride 4 s (no overlap).
- ⚠ The per-dataset bookkeeping in `preprocess.yaml` (`DaLiA: sample_num 16181, duration 35.96`) corresponds **exactly** to
  8 s segments: Σ_subjects ⌊n_4s/2⌋ = 16181 and 16181×8 s = 35.96 h (WildPPG likewise: 390216×8 s = 867.1 h). So the config
  was at some point run with `segment_len: 8`; the shipped global value is 4. **The HR metric code is only correct when
  `segment_len == window_size (8)`** (§20) — a second, independent sign that the authors' reported runs used 8 s windows.
  Which one the paper's numbers used must come from the paper (§22-23). `sample_num`/`duration` are not read by the code.
  Both lengths are supported by the shipped code without edits (`preprocess.segment_len=8` CLI override); our pipeline is
  parameterised the same way.

## 6. Subject split (`src/utils/load_data.py:13-24`)
- `file_list = glob.glob(procdata/PPG-DaLiA/*)` → `random.sample(file_list, len(file_list))` → `val = first 15//8 = 1 file`,
  `test = next 1 file`, `train = remaining 13`. Split is at the **file = subject** level ⇒ no subject overlap between splits.
- ⚠ Single random hold-out, **not** cross-validation despite the name `fold_num: 8`; 1 validation subject, 1 test subject.
- ⚠ `glob.glob` order is directory-order dependent, so the seed-42 split is **not reproducible across machines**. On this
  filesystem it selects **val = S4, test = S10** (`scripts/upstream_split_probe.py`, recorded in `data/manifests/split_upstream_probe_thisfs.json`).
  The subjects are never printed by upstream.
- `python random` is seeded by `fix_seed(42)` (`help_func.py:80-90`) and not consumed before the split, so the split is
  deterministic *given* a fixed glob order.

## 7. Normalisation and leakage
- All statistics are **window-local** (§3); there is no train-set mean/std, no per-subject scaling, nothing fitted.
- Input path at inference: `model(ppg)` only (`train.py:63`); the target is used exclusively as the reference for MAE/HR
  (`train.py:71,76-79`). ⇒ no target statistics enter the input path. But: the reference ECG is itself per-window
  min-max scaled, so MAE lives in normalised units and amplitude (mV) is not modelled.
- `PPGDataset` concatenates subjects in file order; the 8 s HR windows are formed from *consecutive batches* of the test
  loader (`train.py:76-91`) — with a single test subject this never crosses a subject boundary.
- Our checks (`ppg2ecg.data.leakage`, `scripts/run_leakage_checks.py`, `scripts/check_processed_parity.py`): subject
  disjointness PASS, window-hash disjointness PASS on the real processed data, window-locality of preprocessing PASS.

## 8. Flow-SSM / S5 architecture (`src/models/PENGUIN.py`, `src/models/layers/S5*.py`)
Shipped DaLiA config (`config/model.yaml`): `h_dim 128, ssm_block_num 4, ssm_ratio 2.0, mlp_ratio 2.0, n_step 25, lr 1e-3`
(constructor defaults `h_dim 16, mlp_ratio 4.0` are overridden).
- Input stems (`L165-174`): two `Conv1d(1→128, k=32, same) → SiLU → Conv1d(128→128, k=32, same)` for PPG and for `x_t`
  (`kernel = sample_rate//4 = 32`; ≈1.06 M params, 23 % of the model).
- `Flow_SSM_Layer` ×4 (`L39-123`), each with two streams:
  - **PPG stream:** `adaLN(norm1) → S5(bidir) → gate` = `ppg_cond`; `ppg = res + ppg_cond`; then `mlp_ppg(adaLN(norm2(ppg)))`
    gated and added to **`res_ppg`** (the original input, `L109`) — i.e. the SSM branch is dropped from the stream's residual
    path and only survives via the MLP nonlinearity and via `ppg_cond` fed to the target stream. ⚠ deviates from the
    standard DiT wiring (`x = x + attn; x = x + mlp`); intentional or not, it is what the checkpoint will contain.
  - **Target stream:** `adaLN(norm1(x_t)) → S5(bidir) → gate` = `target_cond`; `target_cond = post_attn_target(pre_attn_target(target_cond) + pre_attn_ppg(ppg_cond))`;
    `x = res_target + target_cond`; then `mlp_target(adaLN(norm2(x)))` gated and added to **`res_target`** (`L120`, same
    residual quirk).
  - `nn.MultiheadAttention cross_attn` (`L45`) is constructed in every block but **never called** (dead: 264,192 params;
    names `pre_attn_*`/`post_attn_*` suggest attention was replaced by addition). `self.revin = Parameter(zeros(2))` is also unused.
- ⚠ **Blocks are not chained on the target side** (`forward_step`, `L203-206`): every block receives the *same* `x_t_emb`,
  while the PPG stream *is* chained; block outputs are **summed** (`all_dx_t += pred_dx_t`) and passed to `FinalLayer`
  (adaLN → Linear(128→1)). So the target path is 1-block deep ×4 in parallel; depth exists only on the PPG side.
- S5 (`S5.py:283-344`, `S5SSM`): `width=128`, `state_width=256` (`h_dim*ssm_ratio`), `block_count=1`, HiPPO-LegS **DPLR** init
  (`make_DPLR_HiPPO(256)`), `bcInit="dense"` (LeCun-normal B, C), **ZOH** discretisation (default), learnable `Lambda`
  (complex64), learnable `log_step` with `dt_min=1e-3, dt_max=0.1`, feed-through `D`, **bidirectional** = forward scan and
  reverse scan concatenated on the state axis (`apply_ssm`, `S5.py:59-63`; `C` is `(128, 512)`). Scan = `associative_scan`
  (`S5_jax_compat.py`) under `torch.vmap` over the batch; `binary_operator` is `torch.jit.script`-ed. Runs on torch 2.11 / sm_120.
- Parameter count (shipped config, measured): **4,568,707 total**, 4,304,513 effective (per-block ≈ 857 k: adaLN 198 k,
  2×S5 263 k, 5 MLPs 330 k, dead attention 66 k). Peak training memory at batch 64: **9.2 GB** (the complex scan intermediates
  dominate, not parameters); sampling 0.96 GB.
- adaLN-Zero init (`L185-195`): all adaLN output layers and the final Linear are zero ⇒ **the network output is identically 0
  at init**; first optimizer step only updates `final_layer.linear`, gradients then cascade (verified: 2 → 16 tensors with
  non-zero grad on steps 1→2). Not a bug, but it means early-training loss ≈ E‖x1−x0‖² ≈ 2 regardless of input.

## 9. PPG conditioning
- PPG enters only through the parallel S5 stream: `target_cond += pre_attn_ppg(ppg_cond)` (`L114-116`), per block, per
  time step (sequence-aligned addition, no attention, no concatenation with `x_t`, no classifier-free guidance / dropout).
- PPG is **not** part of the adaLN condition vector (`cond = timestep_emb`, `L200-201`).

## 10. Timestep conditioning
- `TimestepEmbedder` (`L13-36`): sinusoidal embedding of the raw `t ∈ [0,1]` (dim 256, `max_period 10000` — designed for
  integer diffusion steps; with t ≤ 1 only the lowest frequencies vary) → `Linear→SiLU→Linear` → 128. Feeds adaLN-Zero
  shift/scale/gate (12 chunks per block: SSM and MLP × shift/scale/gate × two streams) and the final layer's shift/scale.

## 11. "OT-CFM" loss (`L220-237, 260-266`)
- `t ~ U(0,1)` (`torch.rand`), `x0 ~ N(0, I)`, `x_t = (1−t)·x0 + t·x1`, target `v* = x1 − x0`, `loss = MSE(v_θ(x_t, ppg, t), v*)`.
- This is the Lipman et al. conditional-OT path with `σ_min = 0` and **independent (x0, x1) coupling** — there is **no
  minibatch optimal-transport coupling** (Tong et al.) anywhere in the code. Equivalent to rectified flow / linear interpolant.
- No loss weighting, no time-warping, no auxiliary losses. Our `ppg2ecg.flow.cfm.cfm_targets` reproduces `dx_t`/`pred_dx_t` bit-exactly.

## 12. Interpolation direction — `t = 0` is **noise**, `t = 1` is **ECG** (data); sampling integrates 0 → 1.

## 13. Heun solver (`heun_step`, `L211-218`; `sample`, `L239-252`)
- Uniform grid `linspace(0, 1, n_step+1)`; each step: `k1 = v(x, t_i)`, `x̃ = x + Δt·k1`, `x ← x + Δt/2·(k1 + v(x̃, t_{i+1}))`.
  **Two network evaluations per step, no Euler shortcut on the last step** ⇒ **NFE = 2·n_step = 50** for the shipped `n_step 25`.
- `.detach()` inside `heun_step`, but sampling runs *without* `torch.no_grad()` (`train.py:63`): activations are built then
  discarded — slower/more memory than necessary, no numerical effect. `x0` is drawn on CPU then moved (`torch.randn(B,1,T).to(device)`).
- Our `ppg2ecg.flow.samplers.heun_sample` is bit-exact with upstream `.sample()` (tiny and full config, CPU and CUDA).

## 14. Inference steps — `n_step: 25` Heun steps = **50 NFE**. Measured on RTX 5090, batch 64, fp32: 1,865 ms/batch
(34 samples/s). Euler 1 NFE: 38 ms/batch (1,703 samples/s). The user's plan "25/10/5/2/1 NFE" therefore maps to Euler
steps {25,10,5,2,1} and Heun steps {25,10,5,2,1} = NFE {50,20,10,4,2}; `configs/nfe_sweep.yaml` lists both.

## 15. Optimizer — `AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=0.01)` (`train.py:126`); AdamW default betas/eps.
No LR scheduler, **no gradient clipping, no EMA** (⚠ `train.yaml: ema_decay 0.999` is never read; grep confirms), no AMP (fp32).

## 16. LR — `1e-3` (`config/model.yaml PENGUIN.lr`). Constant.

## 17. Batch size — `64` (`train.yaml`), drop_last False; MAE is a mean over batches of per-batch means (last batch weighted equally).

## 18. Epochs — `epoch_num: 300` max with early stopping on **val MAE** (`earlystop_metric: "mae"`, `patience: 10`).
⚠ Val MAE is computed with the **full 50-NFE sampler every epoch** (`train.py:154`) — the dominant cost per epoch.
Train MAE (`train.py:71`) is *not* a sampling MAE: in train mode `forward` returns a **one-step Euler estimate** of `x1`
from the noised input (`L234`), so train/val MAE are not comparable.

## 19. Checkpoint selection — best val MAE epoch → `ckpt/pretrain_ckpt.pth` (`epoch, state_dict, cfg`), overwritten;
test uses this checkpoint via `load_checkpoint(..., weights_only=False)` with **`strict=False`** (`help_func.py:270-276`).
If `monitor_val` were false, the last epoch is saved instead.

## 20. ECG evaluation (`train.py:38-91`, `help_func.py:103-136, 163-188`)
- **MAE** on the [-1,1]-normalised 4 s windows (mean of per-batch means).
- **HeartRateError**: concatenate **2 consecutive 4 s test windows → 8 s** (`window_size 8`); then ⚠ **`signal.resample(x, 128 × segment_len = 512)`**
  (`help_func.py:177-180`) — with the shipped `segment_len 4` this squeezes the 1024-sample / 8 s window into 512 samples that are
  then treated as 128 Hz, i.e. a **2× time compression**: every HR estimate is doubled (verified: true 60 → 119.7, 90 → 179.6 bpm)
  and windows with true HR ≳ 100 bpm (effective ≳ 200 bpm after compression) return no beats → `-1` → masked; when no window
  survives the error is **0.0** (verified: a 120 bpm window scores 0.0 error against itself *and* against anything). Consequences:
  (i) the reported "bpm" error is in doubled units, (ii) high-HR activity (DaLiA subjects average 71–126 bpm; S5/S6/S11 > 100)
  is silently excluded, (iii) degenerate predictions are rewarded. With `segment_len 8` (`RR_seqlen 1024`) the metric is correct —
  see §5. `tests/test_upstream_parity.py::test_upstream_hr_metric_time_compression_with_4s_segments` pins this behaviour;
  `ppg2ecg.evaluation.metrics.penguin_hr_error(mode="as_shipped"|"corrected")` reproduces both variants.
  Remaining pipeline:
  **prediction only** is cleaned with `nk.ecg_clean(method="pantompkins1985")`, the target is not; R-peaks via biosppy
  `hamilton_segmenter → correct_rpeaks(tol=0.05) → extract_heartbeats(before 0.2, after 0.4)`; HR via
  `tools.get_heart_rate(smooth=True, size=3)` then **mean HR over the 8 s window**; windows where either side yields no HR
  are masked out; ⚠ if *no* window in the pair is valid the error is recorded as **0.0** (`help_func.py:186`) — a bias in
  favour of degenerate outputs. Only ECG metric reported; no R-peak F1 / morphology / RR metrics.
- **Inference time** = wall-clock per test batch of 64 incl. `cuda.synchronize` (without `no_grad`).
- `summarize()` prints thop's `Params` = 3.25 M — thop ignores the raw S5 tensors (1.05 M) and the dead attention; the true
  `sum(p.numel())` is 4.57 M. Do not compare the printed figure with the paper without knowing which counter was used.
- Qualitative plots for the first 20 8 s windows.
Our suite (`ppg2ecg.evaluation`) adds R-peak P/R/F1, RR MAE, QRS width, beat-template correlation, PCC/RMSE, NFE/latency/
memory, and reproduces upstream's HeartRateError via the unmodified upstream function for comparability.

## 21. Seed handling (`help_func.py:80-90`)
- `fix_seed(42)`: `random`, `numpy`, `torch`, `torch.cuda` seeded; `cudnn.deterministic = True`;
  ⚠ `torch.use_deterministic_algorithms = True` is an **attribute assignment**, not a call → no effect; `cudnn.benchmark`
  untouched. The split depends on unseeded glob order (§6). DataLoader shuffling uses the seeded global torch RNG.

## 22. Paper ↔ code and doc ↔ code mismatches
Documentation-level (verified in code):
| Item | Documentation / config says | Code does |
|---|---|---|
| Entry point | README: `src/training.py` | file is `src/train.py` |
| Dataset key | `preprocess.yaml: DaLiA` | code needs `PPG-DaLiA` → crashes |
| EMA | `train.yaml ema_decay: 0.999` | no EMA anywhere |
| Cross-validation | `fold_num: 8` | single 13/1/1 hold-out |
| Window bookkeeping | `sample_num 16181 / duration 35.96 h` (8 s units) | `segment_len: 4` |
| Sampling steps | `n_step: 25` | 50 network evaluations (Heun) |
| Attention | `cross_attn` / `pre_attn` / `post_attn` names, `figure/model_arch.png` | no attention is executed |
| Determinism | `use_deterministic_algorithms = True` | no-op assignment |
Paper-level. Paper: **Suzuki, Koyama, Hirano, Nagashima (Neurogica Inc.), "PENGUIN: General Vital Sign Reconstruction from
PPG with Flow Matching State Space Model", arXiv:2602.03858 (23 Jan 2026), accepted ICASSP 2026 (oral BISP-L5.5)**
(located and read in full by the paper-hunter agent; quotes are from the PDF text).

| Topic | Paper says | Code does | Status |
|---|---|---|---|
| Datasets for ECG | PPG-DaLiA and WildPPG (§4.1) | both loaders exist | match |
| Resampling / filters | 128 Hz; PPG Butterworth band-pass 0.5–4 Hz, z-score, [-1,1]; ECG high-pass 0.5 Hz, same scaling (§4.1) | identical (order 4, filtfilt, per window) | match (filter order & per-window scope are code-only) |
| Window length | not stated explicitly (4 s only implied by a figure caption) | `segment_len 4`; bookkeeping + HR metric imply 8 s (§5, §20) | **ambiguous → prereg §6 rule** |
| Split | "cross-subject … training, validation, and test splits at a **6:1:1 ratio**, no subject overlap" (§4.1) | `15 // 8` ⇒ **13 : 1 : 1** subjects, single run, glob-order dependent | **mismatch** (paper ratio ≈ 11:2:2 of 15) |
| Objective | OT-CFM [Lipman], p0 = N(0,1), x_t = (1−t)x0 + t x1, L_CFM = E‖u_θ − u‖² (§3.1, Eq. 3) | same; independent coupling, t ~ U(0,1) | match |
| Sampler | Heun, **25 sampling steps** (§3.1, §4.3) | 25 Heun steps = 50 NFE | match (NFE never stated) |
| Backbone | S5 dual-stream "Flow-SSM"; L = 4, n = 128, m = 256 (§3.2–3.3, §4.3) | `ssm_block_num 4, h_dim 128, state 256` | match |
| Conditioning | FiLM + scaling from sinusoidal timestep embedding; PPG "through an additive operation after a **linear projection**", explicitly not cross-attention (§3.3) | adaLN-Zero (FiLM+gate); PPG via **2-layer GELU MLPs** (`pre_attn_*`) + addition + another MLP (`post_attn_target`); unused `cross_attn` | partial mismatch (MLP vs linear) |
| Block wiring | not described | blocks summed, target stream not chained, residual quirk (§8) | paper-silent |
| Optimiser | AdamW (β 0.9, 0.999), lr 1e-3, batch 64, ≤ 300 epochs, early stopping patience 10 (§4.3) | same; plus weight_decay 0.01 (code only) | match |
| EMA / LR schedule / clipping | not mentioned | none | match (config `ema_decay` is dead) |
| Model selection | not stated | best val waveform-MAE of 50-NFE samples | paper-silent |
| HR Error | "MAE of heart rate in bpm … Hamilton QRS detector over an **8-second window**" (§4.2) | as §20, incl. pred-only cleaning, masking, 0.0 fallback, and the 2× compression when segment_len = 4 | match in words; implementation flaws paper-silent |
| Seeds / repeats / std | none reported; single numbers | seed 42, single run | paper-silent |
| Params / FLOPs / latency / NFE-quality | **none reported** | 4.57 M params (3.25 M by thop), ~3.2 GFLOPs per velocity eval, 1.8–1.9 s per batch of 64 at 50 NFE on RTX 5090 (our measurement) | paper-silent |
| **Results (Table 1, PPG-DaLiA HR Error, bpm)** | CycleGAN 23.61 · RDDM 16.43 · RespDiff 22.75 · PaPaGei-S 40.89 · **PENGUIN 15.64**; WildPPG PENGUIN 12.97 | — | target for arm A0 |
| Ablation (Table 2, DaLiA) | w/o FiLM 16.30 · w/o shift 15.72 · **w/o PPG cond. 24.40** | — | 24.40 = unconditional reference point for our PPG-shuffle test |

Most consequential for a faithful reproduction: (1) window length 4 s vs 8 s (changes the HR-metric semantics, §20);
(2) 13/1/1 single-subject test split (paper claims 6:1:1) ⇒ the 15.64 bpm number rests on **one test subject**, identity
unknown; (3) HR-error implementation biases (compression, 0.0 fallback, pred-only cleaning); (4) no seeds/variance reported.

## 23. Runtime hazards for reproduction on this machine
| Hazard | Status here | Mitigation in this repo |
|---|---|---|
| `DaLiA`/`PPG-DaLiA` key (§1) | hard crash | `configs/upstream/preprocess.yaml` patch |
| Relative paths + Hydra `outputs/` littering into the read-only checkout | would dirty upstream | absolute paths in `configs/upstream`, `hydra.run.dir` redirected to `outputs/hydra/` in `scripts/run_upstream_*.sh` |
| `utils/__init__.py` imports all baselines → preprocessing needs torch/thop/neurokit2/biosppy/matplotlib/h5py/pandas/wandb | satisfied by `.venv` | `pyproject.toml` deps; `peakutils` added for biosppy |
| numpy(MKL 2025).eigh after `import torch` (pip torch bundles an older `libgomp`) can **segfault** in a clean environment; S5 init calls `np.linalg.eigh` | reported by one auditor under a clean shell; **not reproducible in the project shell** (see docs/ENVIRONMENT.md) | preflight probe in `scripts/run_upstream_*.sh` exports `MKL_THREADING_LAYER=SEQUENTIAL` only if the probe fails |
| Same `description` ⇒ `output.log`/`pretrain_ckpt.pth` overwritten | — | always pass `train.logging.description=<tag>` |
| `torch.cuda.synchronize()` unconditional | fine on GPU | CPU-only smoke uses our scripts instead |
| Raw data ≈ 28 GB on disk (2.9 GB zip + ~25 GB pickles), processed 0.25 GB | 1.2 TB free | — |
| `@torch.jit.script` deprecated on torch 2.11 | works | `PYTORCH_JIT=0` fallback if ever removed |

## 24. Independent verification pass (multi-agent, adversarial)
Method: 5 lens-specific auditors (data / architecture / flow-training-eval / runtime / paper) produced findings; each high/medium
finding was handed to an adversarial verifier instructed to *refute* it by re-reading the code. 17 findings were verified:
**15 confirmed, 2 refuted**. The independent paper-vs-code comparison produced by the compare agent is reproduced verbatim in §25. Severity is the verifier's re-assessment for our reproduction.

### Confirmed
1. **[high]** The released PENGUIN code performs a single subject-level 13/1/1 hold-out, not 8-fold CV, and the identity of the val/test subject is filesystem-dependent and never logged. Precisely: `src/utils/load_data.py:15` builds `file_list = glob.glob(f"{procdata_path}/{dataset}/*")` with NO `sorted()` (the other four glob sites in the same file, lines 53, 132, 133, 151, do wrap in `sorted()`); line 16 shuffles it with the global `random.sample(file_list, len(file_list))`; line 17 sets `val_size = subject_num // fold_num` = 15 // 8 = 1 (`config/preprocess.yaml:31`, `config/train.yaml:17`); lines 20-22 take val = shuffled[0], test = shuffled[1], train = the other 13. `load_dataset_path` is called exact …
2. **[high]** The finding stands and is stronger than stated. (a) VERIFIED IN CODE: the global `segment_len: 4` (config/preprocess.yaml:9) is the only value the pipeline uses for windowing (load_data.py:86-93 `sliding_window_view(x, fs*segment_len)[::fs*segment_len]`), resampling (preprocess.py:37), model input length (help_func.py:55-58, :95) and the HR metric (help_func.py:177-178). The per-dataset `sample_num`/`duration` keys (preprocess.yaml:16,18,30,32,42,44,54,56,66,68,78,80) are read nowhere (grep for `.sample_num`, `.duration`, `["sample_num"]`, `["duration"]` in src/ returns nothing); the only dataset_cfg keys consumed at runtime are subject_num, ppg_fs, label_fs, label, label_bandpass/freq_range …
3. **[high]** HeartRateError is computed on an 8 s signal but resampled to a 4 s length and read at 128 Hz, so with the shipped segment_len=4 the metric reports doubled heart rates for HR <= ~100 BPM and silently scores most windows with true HR > 100 BPM as 0.0 error. Mechanism (all verified in code): config/train.yaml:27 sets window_size[HeartRateError]=8 and config/preprocess.yaml:9 sets segment_len=4; src/train.py:44-47 therefore buffers metric_seq_num=2 consecutive 512-sample test windows and src/train.py:85-86 passes the 1024-sample concatenation (8 s @128 Hz) to compute_metrics. src/utils/help_func.py:177-178 hard-codes RR_seqlen = 128*cfg.preprocess.segment_len = 512, :179-180 FFT-resample the 102 …
4. **[high]** HeartRateError zero-fills every 8-s test window in which R-peak/HR extraction fails on EITHER the prediction or the target, and the failure fraction is neither reported nor small on PPG-DaLiA. Mechanism (all verified in code at commit 6cd70cd): train.py:41-47 packs two consecutive 4-s/512-sample segments into one 1024-sample buffer (window_size=8, segment_len=4); train.py:76-91 calls compute_metrics once per such window; help_func.py:175 reshapes to a single row, :177-180 resamples the 1024-sample (8 s) window to RR_seqlen=128*4=512 and analyses it at 128 Hz (8 s read as 4 s, so apparent HR = 2x true HR); calc_ecg_hr (:127-135) therefore returns one value each for pred_hr/target_hr, so `mask …
5. **[medium]** The crash is real and reproducible as shipped, but the impact clause is overstated. Facts: config/preprocess.yaml:2 sets `dataset: PPG-DaLiA` and :6 lists `PPG-DaLiA`, yet the per-dataset block header at config/preprocess.yaml:29 is `DaLiA:`; config/train.yaml:7 also sets `dataset: PPG-DaLiA`. The code looks the block up by that name via getattr at src/preprocess.py:70, src/utils/load_data.py:14, src/train.py:32 and src/train.py:118. Hydra composes the config in struct mode (hydra/_internal/config_loader_impl.py:270 `OmegaConf.set_struct(cfg, True)`), and on the actual config dir (omegaconf 2.3.1/hydra 1.3.5; uv.lock pins 2.3.0/1.3.2, same missing-key semantics) `getattr(cfg.preprocess, 'PPG …
6. **[medium]** In PENGUIN.forward_step (src/models/PENGUIN.py:197-209) the noisy-target embedding is computed once, `x_t_emb = self.pre_conv_target(x_t)` (line 199), and the loop at lines 204-206 passes that same unmodified `x_t_emb` to every Flow_SSM_Layer while reassigning only `ppg_signal`; each block's `pred_dx_t` is summed into `all_dx_t` and the sum goes through `final_layer` (line 208). Inside Flow_SSM_Layer.forward, the block's target output is `dx_t = res_target + target_cond` (line 120) with `res_target = x_t` = the block input (line 112) and `target_cond = gate_mlp_target * mlp_target(modulate(norm2_target(x_t')))` (line 119), where `x_t' = x_t_emb + post_attn_target(pre_attn_target(gated S5(x_t …
7. **[medium]** The per-window preprocessing in src/preprocess.py makes the stored ECG target a window-dependent, boundary-distorted transform of the raw ECG, and the finding's mechanism is correct — but its magnitude is understated (~4x at the median, ~10x at p90), its localization is incomplete (both window ends, not just the first 0.5 s), and its causal attribution to the default padlen is wrong. Precisely: (1) src/utils/load_data.py:86-89 cuts non-overlapping 4 s ECG windows (2800 samples at 700 Hz); src/preprocess.py:37 FFT-resamples each window independently to 512 (axis=1), which assumes periodicity (a 0..1 ramp comes out 0.4085..1.0853). (2) src/preprocess.py:41-49 applies butter(4, 0.5/64, 'high')  …
8. **[low]** Train-state "MAE" and val/test "MAE" in src/train.py are computed by the same line (train.py:71, `mae = mean|pred_signal - target_signal|`) but on two different quantities, so they are not comparable statistics. (a) In train state, train.py:61 calls `model(ppg, target_signal=...)`, which PENGUIN.forward (src/models/PENGUIN.py:254-256) routes to `train_flow` (PENGUIN.py:220-237): it draws t ~ U(0,1) per sample (line 225), forms x_t = (1-t)x_0 + t x_1 (line 229), runs ONE network evaluation (line 233), and returns the Euler extrapolation pred_x_1 = x_t + (1-t)·v̂ (line 234, commented "Euler step for monitoring training"). Algebraically x_1 - pred_x_1 = (1-t)(v - v̂) with v = x_1 - x_0 (line 23 …
9. **[low]** Val/test sampling runs with autograd enabled: src/train.py:52-53 only calls model.eval() and src/train.py:63 calls model(ppg) with no torch.no_grad()/inference_mode (grep over src/ finds torch.no_grad only at src/models/layers/S5_jax_compat.py:287, inside the init-time _truncated_normal clip, not on any inference path). model.forward (src/models/PENGUIN.py:254-258) routes to sample() (:239-252), which loops n_step=25 (config/model.yaml:4) times over heun_step (:211-218); each heun_step calls forward_step twice (:214 and :216) and .detach()es each result, so a batch costs 50 forward_step evaluations, each of which builds an autograd graph (all module parameters require grad) that is released  …
10. **[low]** In Flow_SSM_Layer.forward (src/models/PENGUIN.py:86-123) each stream's final residual add uses the block INPUT, not the post-SSM intermediate. PPG stream: h1 = res_ppg + gate_ssm*S5(...) (lines 105-106), then out = res_ppg + gate_mlp*MLP(LN(h1)) (lines 108-109). Target stream: h1 = res_target + post_attn(pre_attn_target(gate_ssm*S5(x_t)) + pre_attn_ppg(ppg_cond)) (lines 113-117), then dx_t = res_target + gate_mlp*MLP(LN(h1)) (lines 119-120). So each block computes out = x + g*MLP(LN(x + branch)) rather than the DiT form (x + branch) + g*MLP(LN(x + branch)); the SSM (PPG) / fused SSM+PPG (target) term reaches the block output only via the MLP's input. Numerically confirmed: with non-zero adaL …
11. **[low]** PENGUIN's data/log paths are cwd-relative and Hydra does not chdir, so runs must be launched from the checkout root or given absolute overrides; Hydra additionally creates a timestamped outputs/ run dir in the cwd. Facts: src/preprocess.py:65 and src/train.py:202 call @hydra.main(version_base=None, ...); with hydra-core 1.3.x (uv.lock:310-311 pins 1.3.2; 1.3.5 installed) hydra/version.py:80-81 sets the base to 1.3, hydra/conf/__init__.py:52 defaults job.chdir=None, and hydra/core/utils.py:148-152 resolves _chdir=False, so os.getcwd() is untouched (empirically confirmed by outputs/hydra/preprocess_20260825_144903/.hydra/hydra.yaml: version_base '1.3', chdir: null). config/preprocess.yaml:3-4  …
12. **[low]** PENGUIN's S5 layers (src/models/PENGUIN.py:50,65: S5(h_dim, int(h_dim*ssm_ratio), bidir=True) with h_dim=128, ssm_ratio=2.0 from config/model.yaml:5,7) use the s5-pytorch defaults block_count=1, dt_min=0.001, dt_max=0.1, bcInit="dense" (src/models/layers/S5.py:302-304,326), so Lambda is the full 256-eigenvalue spectrum of a single HiPPO-N(256) matrix (S5.py:314-317; S5_init.py:44-68). Verified facts: Re(Lambda) = -0.5 exactly for all states (diag(S) is identically -0.5); |Im(Lambda)| min/25/50/75/max = 0.2/33/105/291/20860, matching the S4D-Inv closed form (N/pi)(N-1); eigenvalues come in exact +/- conjugate pairs and B_tilde rows are conjugate pairs up to a per-eigenvector phase, so the 256 …
13. **[low]** `summarize()` (src/utils/help_func.py:93-99) builds `sample = torch.randn((1, 1, 512))` (128 Hz x 4 s from config/preprocess.yaml:8-9) and calls `profile(model, inputs=sample)` with a bare tensor (help_func.py:97). thop 0.1.1.post2209072238 (uv.lock:1276-1277; identical version installed) executes `model(*inputs)` (thop/profile.py:212); star-unpacking a (1,1,512) tensor yields exactly one (1,512) tensor, which `PENGUIN.forward` (src/models/PENGUIN.py:254-258) routes to `sample()` because `target_signal is None`; `sample()` reads `B, T = ppg_signal.shape` (PENGUIN.py:240) and runs `n_step=25` (config/model.yaml:4) Heun steps (PENGUIN.py:246-249), each calling `forward_step` twice (PENGUIN.py: …
14. **[low]** summarize() (src/utils/help_func.py:93-100) is a print-only diagnostic invoked once at src/train.py:228 on a throwaway model (help_func.py:94); the model that is actually trained is built separately at src/train.py:125. It therefore affects no trained weight, checkpoint, or reported metric. But both numbers it prints are misleading:  (1) "Params: 3.25M" is an undercount. thop.profile (thop 0.1.1-2209072238, uv.lock:1276-1277) registers count_parameters as a forward hook only on modules whose exact type is in register_hooks (thop/profile.py:21-66, 188-203), and dfs_count sums total_params only from those hooked modules, with the root's own parameters starting at 0 (profile.py:214-233). With t …
15. **[low]** Import-time dependency hazard (packaging inconvenience, not a correctness issue). src/preprocess.py:11 `from utils.help_func import load_dataset` and src/train.py:14-24 both first execute src/utils/__init__.py, whose lines 1-7 import models.PENGUIN/RDDM/CycleGAN/PaPaGei_S/RespDiff (src/models has no __init__.py -> namespace package; PENGUIN.py:3-8, CycleGAN.py:3-8, PaPaGei_S.py:2-7, RespDiff.py:2-8 import torch; RDDM.py:1 also imports neurokit2) and lines 8-15 import utils.load_data (src/utils/load_data.py:5 `import h5py`, :7 `import pandas`). CORRECTION to the original title: the five heavy packages torch/thop/neurokit2/biosppy/matplotlib are required by src/utils/help_func.py's OWN module- …

### Refuted / downgraded
1. The code observation is accurate but the severity/impact is wrong for the target dataset. Verified in code: external/PENGUIN/src/utils/load_data.py:86-93 windows ECG (700 Hz, W=2800) and BVP (64 Hz, W=256) independently with sliding_window_view(...)[::W], giving floor(N/W) rows each; no equality assertion exists there, in src/preprocess.py:73-93 (lines 95-96 only print the last subject's shapes), or in src/utils/help_func.py PPGDataset (x and y concatenated separately at :232-233, __len__ uses x only at :256, __getitem__ indexes y with the same idx at :260; train.py:130-136 wraps it in a plain DataLoader). So a per-subject count mismatch WOULD cumulatively shift every later subject's (PPG, E …
2. The segfault is real on this machine, but the finding mis-states its mechanism, its scope, its headline impact, and one of its mitigations. CORRECTED: (1) Trigger and scope — VERIFIED: numpy.linalg.eigh/eigvalsh on a COMPLEX Hermitian matrix (the exact call at src/models/layers/S5_init.py:63 `np.linalg.eigh(S * -1j)`, reached from src/models/layers/S5.py:315 `make_DPLR_HiPPO(block_size)`) segfaults (exit 139) whenever MKL is running its GNU-OpenMP threading layer (libmkl_gnu_thread) and N is roughly >=64 (N=4/32/48 exit 0; N=64/128/256/384/512/1024 exit 139). The PENGUIN size is N=256 (config/model.yaml:5-7 h_dim=128, ssm_ratio=2.0 -> src/models/PENGUIN.py:50,65 `S5(h_dim, int(h_dim*ssm_rati …



## 25. Paper-vs-code comparison (compare agent, verbatim)

# PENGUIN on PPG-DaLiA — paper vs. code comparison (repo commit 6cd70cd, read-only)

**Paper found: yes.** arXiv:2602.03858v1 (ICASSP 2026, oral BISP-L5.5), Suzuki et al., Neurogica. Quotes below reference the pdftotext dump `penguin_raw.txt` (line numbers `P:n`). Every code claim was re-read in the checkout this session; tags: **[V]** verified in code, **[I]** inference, **[U]** unknown/unverifiable.

Two repository facts frame everything else:
- The DaLiA config block is named `DaLiA:` (`config/preprocess.yaml:29`) while `preprocess.yaml:2`, `train.yaml:7`, `load_data.py:80` and every `getattr(cfg.preprocess, ...)` (`preprocess.py:70`, `train.py:32,118`, `load_data.py:14`) use `PPG-DaLiA`. Git shows the initial commit 611a64a was self-consistent under `DaLiA` and commit 214cf0e renamed five sites but missed line 29. **[V]** The code as shipped cannot run on DaLiA without a one-line fix (or the override `+preprocess.PPG-DaLiA=${preprocess.DaLiA}`); the fix has zero numerical effect.
- `segment_len: 4` (`preprocess.yaml:9`) has been 4 since the initial commit, yet `sample_num: 16181` (`preprocess.yaml:32`) equals 35.96 h / 8 s, the HR metric is only self-consistent at 8 s (`help_func.py:177-178`), `PENGUIN.py:271-272` self-tests with 1024-sample (8 s) inputs and `PaPaGei_S.py:303` defaults `segment_len=8`. The paper's only window statement is the Fig. 2 caption "over a 4-second segment" (P:297). **[U]** Whether Table 1 was produced at 4 s or 8 s cannot be determined from the repo or the paper.

## Comparison table

| # | Topic | Paper says | Code does (file:line) | Status | Consequence for reproduction |
|---|---|---|---|---|---|
| **Data & preprocessing** |
| 1 | Datasets / task | PPG-DaLiA and WildPPG for ECG reconstruction (P:164) | `@register_dataset("PPG-DaLiA")` loader `load_data.py:80-95`; reads `signal.chest.ECG` (label, 700 Hz `preprocess.yaml:35`) and `signal.wrist.BVP` (input, 64 Hz `preprocess.yaml:33`) from `S{i}/S{i}.pkl` (`load_data.py:82-85`) [V] | match | Channel choice is unambiguous (DaLiA has one wrist BVP, one chest ECG). Other channels (ACC, HR labels) unused [V]. |
| 2 | Preprocessing lineage | "we followed protocols in prior works [11, 22, 39]" (P:167-168) | Own pipeline in `preprocess.py:14-62`; no import of RDDM/RespDiff preprocessing code [V] | paper-silent on specifics | The citation is not operational; use the code's pipeline as the protocol. |
| 3 | Resampling | "All signals were resampled to a uniform frequency of 128 Hz" (P:168) | `signal.resample(x, 128*segment_len, axis=1)` applied **per window** after windowing at native rate (`preprocess.py:37`; windows cut in `load_data.py:86-93`) [V]; FFT resample per 4-s window (periodicity assumption, edge ringing) [V, measured by data auditor] | match (rate) / paper-silent (per-window scope) | Replicate the window-then-resample order exactly; whole-record resampling gives a different target (~0.05 MAE units, auditor measurement). |
| 4 | PPG filtering + scaling | Butterworth band-pass 0.5-4 Hz, z-score, scale to [-1,1] (P:168-170) | `butter(4,[0.5/64,4/64],'band')` + `filtfilt` per window (`preprocess.py:41-49`, flags `preprocess.yaml:10-13`); `zscore(axis=1)` `preprocess.py:53`; per-row min-max to [-1,1] `preprocess.py:56-60` [V] | match; paper-silent on order 4, zero-phase, per-window statistics | Order/phase/scope are fixed only by code. Z-score is a no-op after min-max [V, auditor numeric]. |
| 5 | ECG filtering + scaling | High-pass 0.5 Hz Butterworth, same standardization/scaling (P:170-171) | `label_freq_range: [0.5, -1]` (`preprocess.yaml:37`) hits `elif freq_range[1] < 0` -> `butter(4, 0.5/64, 'high')` (`preprocess.py:45-46`); `label_zscore/normalize: True` (`preprocess.yaml:38-39`) [V] | match | Same as row 4. No notch, no explicit LP beyond FFT truncation at 64 Hz [V]. |
| 6 | Normalization scope | (not stated) | Per 512-sample window, both signals, no train-set statistics, identical path for train/test (`preprocess.py:53-60,73-90`) [V] | paper-silent | Targets are per-window min-max in [-1,1]; MAE numbers are in these units. No leakage [V]. |
| 7 | Window length / stride | Only Fig. 2 caption: "over a 4-second segment" (P:297); conclusion mentions "flexible windowing" (P:339); no count of segments given | `segment_len: 4` (`preprocess.yaml:9`), non-overlapping (`sliding_window_view(...)[::fs*4]`, `load_data.py:86-93`) -> 512 samples at 128 Hz; but bookkeeping `sample_num: 16181` = 8-s count (`preprocess.yaml:30-32`), HR metric consistent only at 8 s (`help_func.py:177-178`), `PENGUIN.py:271` uses 1024 samples, `PaPaGei_S.py:303` default 8 [V] | unverifiable (4 s vs 8 s) | Highest-impact ambiguity. Run both, or pick 8 s (the only setting where the shipped HR metric measures true bpm). Backbone is length-agnostic (`padding="same"` `PENGUIN.py:166-173`; T taken from input `PENGUIN.py:221,240`) [V]. |
| 8 | Split protocol | "cross-subject ... training, validation, and test splits at a 6:1:1 ratio with no subject overlap" (P:173-175) | `val_size = subject_num // fold_num = 15 // 8 = 1` (`load_data.py:17`, `train.yaml:17`, `preprocess.yaml:31`); val = 1 subject, test = 1 subject, train = 13 (`load_data.py:20-22`); called once (`train.py:129`), no fold loop (grep `fold`: only `load_data.py:17`, `train.yaml:17`) [V] | mismatch | Paper's 6:1:1 (≈11/2/2 for 15 subjects) is not what the code does; code is 13/1/1 on a single held-out subject. Test metric is a one-subject estimate. |
| 9 | Which subjects held out | (not stated) | `glob.glob(...)` unsorted (`load_data.py:15`; the four other loaders use `sorted()`, lines 53,132,133,151), then `random.sample` with Python RNG seeded 42 (`load_data.py:16`, `help_func.py:82`, `config.yaml:7`) -> permutation [10,1,0,11,...] of directory-listing order; never printed [V] | paper-silent / code nondeterministic across machines | The paper's test subject cannot be recovered. Fix the split explicitly and log it; report per-subject or LOSO. |
| 10 | Repeats / seeds / variance | None (no ±, no seed, no hardware anywhere in P) [V by grep] | `seed: 42` (`config.yaml:7`); `fix_seed` seeds python/numpy/torch/cuda, `cudnn.deterministic=True` (`help_func.py:80-89`); line 90 `torch.use_deterministic_algorithms = True` is an attribute assignment (no effect) [V]; single run | paper-silent | Single-seed, single-split numbers. Init RNG stream also depends on `summarize()` running first (`train.py:228-229`) [V]. |
| **Generative framework** |
| 11 | Framework | OT-CFM [13], p0 = N(0,1), conditioned on z in R^K (P:65,96-98) | `x_0 = randn_like(x_1)` (`PENGUIN.py:228`), PPG passed as conditioning input to every `forward_step` (`PENGUIN.py:233,214,216`) [V] | match | "OT" = Lipman conditional-OT path with sigma_min = 0, not minibatch OT: no `ot`/Sinkhorn/assignment code (grep) [V]. |
| 12 | Path / target velocity | x_t = (1-t)x0 + t x1, u = x1 - x0 (P:138-139); loss Eq. (3) | `x_t = (1-t)*x_0 + t*x_1` (`PENGUIN.py:229`), `dx_t = x_1 - x_0` (`:230`), `F.mse_loss(pred_dx_t, dx_t)` (`:261`) [V] | match | Plain velocity MSE, no time weighting. |
| 13 | t distribution | E over t (Eq. 3), distribution unspecified | `t = torch.rand(B,1)` uniform [0,1) (`PENGUIN.py:225`) [V] | paper-silent (standard reading) | Keep U(0,1). |
| 14 | Sampler | Heun's method (P:142); "25 sampling steps" (P:219) | `linspace(0,1,n_step+1)` (`PENGUIN.py:246`), `n_step: 25` (`model.yaml:4`), `heun_step` calls `forward_step` twice (`:214,216`) -> 50 NFE; full Heun on last step; single stochastic sample, no clipping (`:239-252`) [V] | match; paper-silent on NFE=50, uniform grid, single sample | Report NFE=50 explicitly when comparing cost. |
| 15 | Val/test grad mode | (n/a) | No `torch.no_grad` on inference path (grep: only `S5_jax_compat.py:287`); `heun_step` detaches (`PENGUIN.py:214,216`); `model.eval()` only (`train.py:52-53`) [V] | paper-silent | No numeric effect (no dropout/BN). Only affects the printed per-batch "Inference Time" (`train.py:57-65,102`). |
| **Architecture** |
| 16 | Backbone | S5 extended into Flow-SSM blocks; Eq. (4) with A,B,C,D bars (P:150-157) | `S5(h_dim, int(h_dim*ssm_ratio), bidir=True)` (`PENGUIN.py:50,65`); s5-pytorch port `S5.py:296-344` [V] | match; paper-silent on bidirectionality, ZOH, init | Details fixed only by code: bidir with shared Lambda/B, doubled C (`S5.py:46-53,184-186,201-202`); ZOH (`S5.py:142,227-228,103-116`); `block_count=1`, `dt in [1e-3,0.1]`, `bcInit='dense'` (`S5.py:302-304,326`); single HiPPO-N(256) block, Re(Lambda) = -0.5 (`S5_init.py:60,63`); no eigenvalue clipping (`S5.py:179-181`). |
| 17 | Sizes | L = 4, n = 128, m = 256 (P:218-219) | `ssm_block_num: 4, h_dim: 128, ssm_ratio: 2.0` -> state 256 (`model.yaml:5-7`; `PENGUIN.py:50,65,178`) [V] | match | — |
| 18 | FFN width | (not stated) | `mlp_ratio: 2.0` (`model.yaml:8`) overrides class default 4.0 (`PENGUIN.py:155`) -> Linear(128,256)-GELU-Linear(256,128) (`:56-60,76-80`) [V] | paper-silent | Use 2.0; class default would double FFN params. |
| 19 | Input embedding | "embedding x̂_t and z with one-dimensional convolutional layers" (P:180-181) | Two separate stems Conv1d(1,128,k=32,'same')-SiLU-Conv1d(128,128,k=32) with `k = sample_rate//4` (`PENGUIN.py:165-174`) [V] | match; paper-silent on depth/kernel | Kernel 32 = 0.25 s; even kernel with `'same'` pads asymmetrically [I from PyTorch semantics]. |
| 20 | Block composition | LayerNorm, FiLM, S5, scaling, FFN; γ,β,α from sinusoidal encoding of t (P:186-190) | `adaLN_modulation = SiLU->Linear(128, 12*128)` chunked to shift/scale/gate for {SSM,MLP}x{PPG,target} (`PENGUIN.py:44,88-101`); `modulate = x*(1+scale)+shift` on non-affine LN (`:83-84,48-49,63-64`); gates multiply branch outputs (`:105,108,113,119`); `TimestepEmbedder` sinusoidal 256-d, max_period 1e4, MLP (`:13-36`) [V] | match | t in [0,1] fed unscaled (all sinusoid args <= 1 rad) [V]; keep as-is. adaLN last Linear zero-init (`:185-189`). |
| 21 | Residual wiring within a block | Fig. 1(a): one skip per stream from block input to the plus after FFN-Scale; no plus between S5-Scale and second LayerNorm [V, figure viewed] | PPG: `out = res_ppg + gate*MLP(LN(res_ppg + gate*S5(...)))` (`PENGUIN.py:104-109`); target: `dx_t = res_target + gate*MLP(LN(res_target + fused))` (`:112-120`) [V] | match (final skip) / paper-silent (extra `res + branch` before LN2 at `:106,:117`) | Copy the exact wiring; a DiT-template rewrite ((x+branch)+MLP) is a different model. |
| 22 | Block stacking on the target stream | "passed through a stack of L Flow-SSM blocks" (P:182); Fig. 1(a) draws x̂_t flowing block 1 -> 2 -> ... -> N with per-block taps summed at a final plus [V, figure viewed] | `x_t_emb` computed once (`PENGUIN.py:199`) and passed unchanged to every block; only `ppg_signal` is chained (`:204-205`); block outputs summed (`:206`) then `final_layer` (`:208`) [V] | mismatch (figure/text vs code) | Code = 4 parallel target heads over the same embedding (each head sees PPG features of depth k); effective target depth 1. Reproduce the code, not the figure; do not "fix" it. |
| 23 | PPG conditioning | "additive operation after a linear projection", not cross-attention (P:190-198); figure shows single "Linear" boxes | `pre_attn_ppg`, `pre_attn_target`, `post_attn_target` are each 2-layer GELU MLPs (`PENGUIN.py:51-55,66-75`); fused as `post(pre_t(target_cond) + pre_p(ppg_cond))` (`:114-116`) per timestep; `ppg_cond` = gated S5 output of the PPG stream, pre-MLP (`:105`) [V] | mismatch (minor: MLP vs linear) | Additive per-timestep design matches; use the MLPs. |
| 24 | Cross-attention | Explicitly not used (P:191-193) | `self.cross_attn = nn.MultiheadAttention(128,1)` constructed (`PENGUIN.py:45`) but never called (grep: only line 45) — 264,192 dead params in state_dict and optimizer [V] | match (behavior) / code quirk | Exclude from parameter counts; harmless otherwise. Also `revin` (`:164`) and `self.mean/std` (`:160`) unused [V]. |
| 25 | Output head | Fig. 1(a) ends at a plus -> dx̂_t; text silent | `FinalLayer`: non-affine LN -> adaLN(shift,scale from t) -> Linear(128,1), zero-init (`PENGUIN.py:126-143,192-195`) [V] | paper-silent | Required for the 128->1 projection; LN removes the 4x skip scale from row 22 [V]. |
| **Training** |
| 26 | Optimizer | AdamW (β1=0.9, β2=0.999), lr 1e-3 (P:215-216) | `optim.AdamW(model.parameters(), lr=model_cfg.lr, weight_decay=...)` (`train.py:126`), `lr: 0.001` (`model.yaml:3`); betas left at torch default (0.9, 0.999) [V] | match | All params in one group incl. complex S5 params [V]. |
| 27 | Weight decay | (not stated) | `weight_decay: 0.01` (`train.yaml:15`) [V] | paper-silent | Equals torch's AdamW default, so a naive reader would land on 0.01 anyway. |
| 28 | Batch / epochs / early stop | batch 64, up to 300 epochs, patience 10 (P:216-218) | `batch_size: 64, epoch_num: 300, earlystop_patience: 10` (`train.yaml:13-19`); loop `train.py:142,163-179` [V] | match | — |
| 29 | Early-stopping metric / model selection | "early stopping" only; metric unnamed | `earlystop_metric: "mae"`, `monitor_val: true` (`train.yaml:18,20`); val MAE = mean |50-NFE Heun sample − target| on the 1 val subject (`train.py:63,71,154,163-174`); best ckpt reloaded for test (`:193`) [V] | paper-silent | Stochastic selection criterion on one subject; variance in the selected epoch is expected. |
| 30 | LR schedule / warmup / clipping / AMP | not mentioned | none (grep `scheduler|clip_grad|autocast`: no hits) [V] | match (absence) | — |
| 31 | EMA | not mentioned | `ema_decay: 0.999` in `train.yaml:16` but never read (grep `ema`: no hits in src/) [V] | match in effect (config key is dead) | Do not add EMA. |
| 32 | Train-mode "MAE" log | (n/a) | `train_flow` returns one-step Euler extrapolation (`PENGUIN.py:234`), so train MAE = (1−t)-weighted velocity error; val/test MAE = full sample (`train.py:61-63,71`) [V] | paper-silent | Never compare train vs val MAE curves. |
| 33 | Checkpointing | (n/a) | single `pretrain_ckpt.pth` overwritten on improvement (`train.py:172-174`); reload `weights_only=False`, `strict=False` (`help_func.py:271-273`) [V] | paper-silent | `strict=False` would hide missing keys in a port. |
| **Evaluation** |
| 34 | HR Error definition | MAE of HR (bpm), Hamilton method, 8-second window (P:204-206) | 8-s window = 2 consecutive 4-s test segments (`train.py:41-47,76-91`, `train.yaml:27`); then `RR_seqlen = 128*segment_len = 512` and `signal.resample(1024 -> 512)` read at 128 Hz (`help_func.py:177-180`) -> 8 s interpreted as 4 s; Hamilton via biosppy (`help_func.py:103-113`) [V] | mismatch as shipped (matches paper only if segment_len = 8) | With `segment_len=4` reported HR is ~2x true and windows with true HR > ~100 bpm fail the 40-200 gate. The paper's stated definition is implementable only at 8 s or with `RR_seqlen = input length`. |
| 35 | HR Error failure handling | (not stated) | `hr=[-1]` when <2 beats or gate empties (`help_func.py:116-125`); one window per call so mask is a single bool; `else 0.0` (`help_func.py:185-186`) contributes a perfect score; no count logged (`train.py:105-106,196-197`) [V] | paper-silent | Zero-fill biases HR Error downward and rewards peak-less predictions. Log failure counts; report an excluded-window variant. |
| 36 | HR Error signal cleaning | (not stated) | `nk.ecg_clean(method='pantompkins1985')` on **prediction only** (`help_func.py:131-132,181-182`) [V] | paper-silent | Asymmetric; keep for fidelity, note when comparing. |
| 37 | Waveform MAE | Not reported in paper | computed and printed per split (`train.py:71,110,195`) [V] | paper-silent | Extra metric available; units = per-window [-1,1]. |
| 38 | PPG-DaLiA result | HR Error: CycleGAN 23.61, RDDM 16.43, RespDiff 22.75, PaPaGei-S 40.89, PENGUIN 15.64 (P:242-247) | No PENGUIN checkpoint or log in repo (`ckpt/` has only `PaPaGei_S.pt`) [V] | unverifiable | Cannot confirm which split, segment length, or metric variant produced 15.64; given rows 7-9 and 34-35 an exact match is not a meaningful target. |
| 39 | Ablations | (ii) w/o FiLM 16.30, (iii) w/o Shift 15.72, (iv) w/o PPG 24.40 on DaLiA (P:315-328); "Shift conditioning" undefined | No ablation switches in `PENGUIN.py` [V] | code-silent | Ablations must be re-implemented; "Shift" meaning is [U] (plausibly β in FiLM). |
| 40 | Baselines | PaPaGei-S, CycleGAN, RDDM, RespDiff (P: Sec. 4.4) | Model configs present (`model.yaml:10-39`); same `train.py` loop; RDDM peak-ROI branch in `help_func.py:235-251` [V] | match (presence) / unverifiable (numbers) | Same caveats as row 38 for each baseline. |
| 41 | Params / FLOPs / latency / hardware | none reported (grep: no hits for FLOP/param/GPU/hardware/latency) [V] | `summarize()` prints thop Params 3.25M (misses S5 + MHA params; true 4.57M numel) and "GFLOPs" 60.77 = MACs over 50 NFE without S5 ops (`help_func.py:93-100`) [V, auditor-measured] | paper-silent / code misleading | Do not quote the script's numbers; count params directly (4,304,513 live numel) and state NFE. |
| 42 | Code availability / logging | Footnote: github.com/Neurogica/PENGUIN | `wandb.init(project="PPG_ICASSP")` (`train.py:224`), `wandb: false` default (`train.yaml:33`) [V] | match | README cites `./src/training.py` and `./configs/` but files are `src/train.py`, `config/` (`README.md:45-51`) [V]. |
| 43 | Environment | not stated | `uv.lock`: torch 2.8.0, biosppy 2.2.3, neurokit2 0.2.12, scipy 1.16.2, numpy 2.3.3, hydra 1.3.2, omegaconf 2.3.0, thop 0.1.1 (`uv.lock:36-37,310-311,545-546,563-564,752-753,1154-1155,1276-1277,1296-1297`); Python >=3.12 (`pyproject.toml:6`) [V] | paper-silent | biosppy's 40-200 bpm gate and neurokit cleaning versions affect the metric; pin them. |
| 44 | Paths / run location | (n/a) | cwd-relative `./data`, `./logs` (`preprocess.yaml:3-4`, `train.yaml:37`); Hydra `version_base=None` does not chdir (`train.py:202`, `preprocess.py:65`) [V] | code-only | Run from repo root or override with absolute paths. |

## Five most consequential mismatches for a faithful DaLiA reproduction

1. **Split: 6:1:1 (paper) vs 13/1/1 single hold-out with filesystem-dependent subject (code).** `load_data.py:15-22`, `train.yaml:17`. The paper's test subject is unrecoverable; the released code evaluates one subject per run, and the val subject (also one) drives early stopping. A faithful reproduction should fix and log the split (or run LOSO / k-fold) and report spread; a single 15.64 match is not achievable in principle.

2. **Segment length 4 s vs 8 s is undetermined, and it changes the metric.** `preprocess.yaml:9` = 4 (since the initial commit) vs. `sample_num: 16181` (8-s count), `help_func.py:177-178`, `PENGUIN.py:271`, `PaPaGei_S.py:303` all pointing to 8 s; paper Fig. 2 says 4 s. The choice fixes segment count (32,368 vs 16,181), model input length, and whether the HR metric is correct. Run both arms or use 8 s and state it.

3. **HR Error as shipped is not the paper's definition.** `help_func.py:177-180` resamples the 8-s window to 512 samples and reads it at 128 Hz (2x time compression) whenever `segment_len=4`; `help_func.py:185-186` zero-fills windows where either side fails detection (15% of DaLiA windows at 4 s per the auditor's measurement, up to 67% for high-HR subjects). Any HR-Error number from the shipped code at 4 s is in doubled-bpm units with silent dropouts. Set `RR_seqlen` to the true input length (or use 8-s segments), log failure counts, and report an excluded-window variant alongside the official one.

4. **Block stacking: figure/text say a stack of L blocks; code feeds the same `x_t_emb` to all 4 blocks and sums their outputs** (`PENGUIN.py:199,204-208`). Combined with the single-skip residual wiring (`PENGUIN.py:104-120`), the velocity network is effectively one block deep on the target stream with four parallel heads. A reimplementation from the paper/figure produces a different, deeper model; copy `forward_step` and `Flow_SSM_Layer.forward` verbatim.

5. **Conditioning projection and architectural details fixed only in code:** "linear projection" is three 2-layer GELU MLPs (`PENGUIN.py:51-55,66-75,114-116`); `mlp_ratio=2.0` (`model.yaml:8`); conv kernel 32 (`PENGUIN.py:166-173`); bidirectional S5 with single 256-state HiPPO block, ZOH, dt in [1e-3, 0.1], no eigenvalue clipping (`S5.py:50,65,302-304,315,326,179-181`); adaLN and final-layer zero-init (`PENGUIN.py:185-195`); unscaled t in a 256-d sinusoidal embedder (`PENGUIN.py:24-28`). None is in the paper; each changes the model if guessed differently. Weight decay 0.01 (`train.yaml:15`) and the stochastic val-MAE early-stopping criterion (`train.yaml:18-20`, `train.py:163-179`) are likewise code-only.

Secondary but blocking: the `DaLiA`/`PPG-DaLiA` config key (`preprocess.yaml:29`) must be patched before anything runs; the README's `./src/training.py` path (`README.md:51`) is wrong; the printed Params/GFLOPs (`help_func.py:93-100`) must not be quoted.
