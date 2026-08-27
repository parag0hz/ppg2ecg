# Experiment Log

Append-only. Every entry: date, what was run (command + commit), where outputs are, what was concluded.

## 2026-08-25 — Session 0: workspace, audit, baseline-readiness (no training)
- Created `/home/kwy00/ppg2ecg-one-step/` as an independent git repo (nested under the unrelated `/home/kwy00` repo; ignored there).
- Environment audit → `docs/ENVIRONMENT.md` (RTX 5090 32 GB, torch 2.11.0+cu130, Python 3.13, bf16 OK).
- Cloned upstream `https://github.com/Neurogica/PENGUIN` @ `6cd70cd` (main, clean) → `external/PENGUIN/` (never edited).
- Source audit → `docs/PENGUIN_AUDIT.md`. Headline: shipped config **cannot run on PPG-DaLiA as-is** (`DaLiA` vs
  `PPG-DaLiA` key), "25 steps" = **50 NFE** (Heun), no EMA/no scheduler despite config, dead `cross_attn`, glob-order split.
- Patched config copy → `configs/upstream/` (3 patches, documented); Hydra compose verified.
- PPG-DaLiA: not present locally; UCI direct link (CC BY 4.0) verified; download started to `data/raw/ppg+dalia.zip`.
- Code written: `ppg2ecg.data` (loader, PENGUIN-faithful preprocessing, deterministic splits, leakage checks),
  `ppg2ecg.flow` (CFM re-statement, Euler/Heun samplers with NFE counting), `ppg2ecg.evaluation` (metrics v0, efficiency),
  `ppg2ecg.models` (upstream import shim). Tests in `tests/`; smoke test `scripts/smoke_test_penguin.py`.
- Manifests: `data/manifests/split_p0_holdout_seed42.json`, `split_p1_kfold5_seed42.json`; leakage checks PASS (subject level).
- Results of tests / smoke test: see the "Session 0 results" entry below (appended after they ran).

### Session 0 results (2026-08-25, no training)
| Check | Result | Evidence |
|---|---|---|
| Hydra compose of shipped upstream config for PPG-DaLiA | **fails** (`ConfigAttributeError: Key 'PPG-DaLiA' is not in struct`) | dry-run with `hydra.compose`; fixed in `configs/upstream/preprocess.yaml` |
| Upstream preprocessing end-to-end (`scripts/run_upstream_preprocess.sh`) | OK, 39 s, 15 files, upstream tree clean | `data/processed/upstream/PPG-DaLiA/subject{0..14}.pkl` |
| Our pipeline vs upstream processed arrays (S1, S6, S15) | **bit-exact** (max abs diff 0.0, x and y) | `scripts/check_processed_parity.py` → `data/manifests/processed_parity_upstream.json` |
| Unit tests (`tests/`, 16 tests incl. preprocess / Heun / CFM parity with upstream) | **16 passed** | `PYTHONPATH=src .venv/bin/python -m pytest` |
| Leakage checks P0 + P1 (subject-level) | PASS ×6 | `scripts/run_leakage_checks.py` |
| Window-level disjointness on real processed data (P0) | PASS (train 28055 / val 2262 / test 2051 unique windows, 0 overlaps) | `check_processed_parity.py` |
| Upstream split on this filesystem | val = **S4**, test = **S10**, train = others | `scripts/upstream_split_probe.py` → `data/manifests/split_upstream_probe_thisfs.json` |
| Upstream model on torch 2.11 / RTX 5090 (`scripts/smoke_test_penguin.py --full`) | imports, trains, samples | `outputs/smoke/penguin_smoke_2026-08-25T144430.json` |
| Parameter count (shipped DaLiA config) | 4,568,707 total; 264,192 in never-called `cross_attn` (+2 in unused `revin`) → 4,304,513 effective | smoke JSON |
| Training step, batch 64 × 512, fp32 | loss 1.99 (finite), **peak 9,175 MiB**, ~140–300 ms/step (first steps) | smoke JSON |
| Gradient flow at init | step 1 updates only `final_layer.linear` (2 tensors), step 2: 16/161 tensors — adaLN-Zero cascade, not a bug | smoke JSON |
| Sampler parity | our `heun_sample` == upstream `.sample()` **bit-exact**, NFE 50 for 25 steps | smoke JSON, `tests/test_upstream_parity.py` |
| Upstream `HeartRateError` with `segment_len=4` | **2× time compression confirmed** (true 60 → 119.7 bpm, 90 → 179.6, 120 → no beats → error 0.0); correct with `segment_len=8` | in-session probe; pinned by `tests/test_upstream_parity.py` |
| Inference cost (batch 64, fp32, median of 5) | Heun 25 steps/50 NFE: **1,865 ms** (34 samp/s); Euler 1 NFE: **38 ms** (1,703 samp/s); peak 958 MiB | smoke JSON (random weights; latency only) |

Cost implications for arm A0: train ≈ 28k windows/64 ≈ 440 steps/epoch (~1 min) + val (1 subject, ~2.3k windows,
50 NFE) ≈ 36 batches × 1.87 s ≈ 70 s ⇒ **≈ 2–2.5 min/epoch**, ≤ 300 epochs ⇒ worst case ~12 h/seed, typical with
patience-10 early stopping ~2–5 h/seed. GPU memory < 10 GB ⇒ two seeds could run concurrently on the 32 GB card.

Deviations from plan: none. Things done beyond the minimum: dataset downloaded (public, CC BY 4.0, 2.9 GB) and upstream
preprocessing executed (39 s) so that leakage/parity checks run on real data — no model training was started.

## 2026-08-25 — Session 1: GitHub connection + A0 (PENGUIN, PPG-DaLiA, 8 s, seed 42)
- Decisions frozen before training (see PREREGISTRATION_V0 §6/§9): **8 s windows**, deterministic P0 split (test S2, val S11), seed 42 only.
- Built `data/processed/v0_8s/` (ours, 16,181 windows, per-file sha256 in `MANIFEST.json`) and `data/processed/upstream_8s/`
  (unmodified upstream `preprocess.py` with `preprocess.segment_len=8`) for parity checking.
- `external/PENGUIN` registered as a git **submodule** pinned at `6cd70cd` (existing clone reused; nothing deleted).
- GitHub: `origin = https://github.com/parag0hz/ppg2ecg.git`. The remote was NOT empty (GitHub "Initial commit" `9fd4ce5` with a 9-byte
  README). Local snapshot committed as `6330fa2` ("session 0: audit PENGUIN and prepare reproducible baseline", 67 files, 0.20 MB,
  gitlink `external/PENGUIN` @ 6cd70cd), then merged with the remote history (`--allow-unrelated-histories`, README conflict → local
  kept) as `6e5a4f1`, pushed to `main` (no force). Author identity set repo-locally (an e-mail later found to be wrong, see the authorship-fix entry below); push used the
  existing `gh` login as a one-off credential helper (`git -c credential.helper='!gh auth git-credential'`), no config/credential created.
- A0 preflight (`scripts/preflight_a0.py`) PASSED at commit `6e5a4f1` (0 dirty): subject/window/normalisation leakage checks OK,
  8 s @ 128 Hz, seed 42, 4,568,707 params, RTX 5090. Training launched via `scripts/run_a0.sh` → `outputs/a0_penguin_otcfm_ppgdalia_8s_seed42/`.
- A0 training finished 16:08: **21 epochs, early-stopped (patience 10), best epoch 11** (val MAE batch-mean 0.2989 on S11 with 50-NFE Heun
  samples), 2,997 s total (≈143 s/epoch), peak 18.4 GiB. Val MAE is noisy epoch-to-epoch (fresh sampling noise each epoch; e.g. 0.332 →
  0.435 → 0.369 → …), so upstream-style early stopping is partly driven by sampling noise — recorded as a limitation. Train CFM loss
  0.339 → 0.172 (still decreasing at stop). Evaluation (`scripts/eval_a0_nfe_curve.py`) launched on `checkpoint_best.pt`.
- A0 evaluation (`scripts/eval_a0_nfe_curve.py`, checkpoint_best = epoch 11, test S2, 1025 × 8 s windows, paired noise seed 0):
  | solver | steps | NFE | HR err (bpm) | R-F1 | RR MAE ms | RMSE | PCC | QRS err ms | morph corr | ms / batch 64 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | Heun | 25 | 50 | **10.99** | 0.141 | 34.9 | 0.472 | 0.002 | 33.7 | 0.662 | 4175 |
  | Heun | 10 | 20 | 11.59 | 0.140 | 35.2 | 0.471 | 0.002 | 33.7 | 0.664 | 1652 |
  | Heun | 5 | 10 | 12.25 | 0.138 | 36.0 | 0.466 | 0.002 | 33.1 | 0.639 | 825 |
  | Heun | 2 | 4 | 11.17 | 0.146 | 34.6 | 0.444 | 0.001 | 33.9 | **0.475** | 329 |
  | Heun | 1 | 2 | **36.32** | 0.087 | 37.3 | 0.479 | −0.001 | 33.9 | 0.230 | 165 |
  | Euler | 1 | 1 | **39.22** | 0.087 | 33.9 | 0.295 | 0.007 | 32.8 | 0.136 | 82 |
  Upstream `HeartRateError` at 50 NFE: corrected 11.74 bpm, as-shipped (diagnostic) 25.14 bpm. Paper: 15.64 → **PASS** (≤ 17.2).
- Diagnostics (`scripts/diagnose_a0_alignment.py`, `scripts/diagnose_dalia_sync.py`): beat counts match (10.5/10.5 per window) but
  prediction↔target lag is uniform over ±0.5 s (F1 0.14 even after per-window best shift → 0.27); predicted-vs-reference HR r = 0.40
  with regression to the mean; **raw PPG-DaLiA wrist/chest streams are not beat-synchronised** (pulse-arrival delay 800–900 ms in
  clean rest, ~20 ms/min drift) → beat-aligned metrics invalid on this dataset (DATA_PROTOCOL §6, PREREGISTRATION §9).
- Report: `docs/A0_PENGUIN_REPRODUCTION_REPORT.md`. Verdict: H1 confirmed (collapse at ≤ 2 NFE, morphology first at 4 NFE); GO for
  the objective-swap line with a weak-baseline caveat; next = A0-b (selection criterion + seeds) and a re-synchronised beat protocol.

## 2026-08-25 — Session 2: A0-b baseline stabilisation → iMeanFlow gate
- Pre-registered `docs/A0B_BASELINE_STABILIZATION_PREREGISTRATION.md` (only change vs A0: checkpoint selection = deterministic
  fixed-bank val CFM loss, 4 banks seed 1000, min_delta 1e-4, patience 20; 50-NFE generation diagnostic every 5 epochs on 128 val
  windows; stochastic val MAE no longer computed). Frozen in a local commit before launch.
- A0 checkpoint scored post hoc on the same fixed banks: **val_cfm_fixed = 0.19045** (bank hash `6b5c0139…`) — the reference for
  the "was A0 under-trained?" rule.
- A0-b launched (`scripts/run_a0b.sh`, seed 42, identical data/split/model/optimizer). iMeanFlow paper/code *audit* workflow
  started in parallel (read-only: locate + verify official sources, pinned clone under `external/iMeanFlow`); implementation waits
  for the mechanical gate.
- iMeanFlow audit (`docs/IMEANFLOW_AUDIT.md`): papers verified (MF arXiv:2505.13447 NeurIPS'25 oral; iMF arXiv:2512.02012 v2 CVPR'26
  highlight), official repo `Lyy-iiis/imeanflow` verified and cloned read-only to `external/iMeanFlow` @ `bf60cd7`. Core: `V = u + (t−r)·sg(du/dt)`,
  JVP tangent = model's own v, v-loss with adaptive weighting (p=1, c=0.01), (t,r) logit-normal(−0.4,1), 50 % r=t, 1-NFE `z0 = z1 − u(z1,0,1)`.
  Feasibility: forward-mode `torch.func.jvp` works through the unmodified S5 backbone and matches finite differences (CPU, O(ε²)).
- **Git authorship fix (2026-08-25 18:05):** the author e-mail used for the first commits was registered on GitHub to an unrelated
  third-party account, so GitHub attributed those commits to it. The sole author of this repository is **parag0hz**. History was
  rewritten (author/committer →
  `parag0hz <131474134+parag0hz@users.noreply.github.com>`, trees byte-identical, root commit `9fd4ce5` untouched) and force-pushed
  with lease; the pre-rewrite history (local-only backup branch) was deleted on 2026-08-27 together with the filter-branch leftovers and
  the reflog, so no object in this repository carries the wrong identity any more. **SHA mapping (old → new):**
  `6330fa2 → a15b354` (session 0), `6e5a4f1 → f2d814b` (merge; A0 provenance.json records `6e5a4f1`), `55e2f17 → 20cc6cd` (A0 results),
  `8998371 → 1ae155c` (A0-b freeze; A0-b provenance.json records `8998371`). Provenance files are left as written — resolve old SHAs
  via the backup branch or this table.
- **A0-b done (18:46)**: 85 epochs, best 65, 1.77 h, val_cfm_fixed 0.1645 (A0 ckpt 0.1904 → A0 under-trained: YES). Test S2, 50 NFE:
  HR err **8.08** bpm (A0 10.99), morph 0.650 (0.662), amp 0.95 (0.83), cond gain 5.69 (3.84), RMSE 0.435. 1 NFE Euler: HR 41.96, morph 0.217,
  amp 0.145, gain 0.24 → all four collapse criteria fail → **gate GO** (`outputs/a0b_…/comparison.json`, `docs/A0B_BASELINE_STABILIZATION_REPORT.md`).
- **Environment incident (19:00)**: the numpy-MKL `eigh`-after-torch segfault became deterministic for 256×256 complex matrices
  (S5 init) — every model build crashed, including the iMF memory probe. Not caused by the jax install (reproduced after uninstalling).
  Fixed by importing `ppg2ecg.utils.mkl_warmup` (a numpy LAPACK call) before torch in every entry point (docs/ENVIRONMENT.md). All A0/A0-b results were produced before the incident and are unaffected (same threaded-MKL numerics).
- iMeanFlow implemented (`src/ppg2ecg/flow/imeanflow.py`, 10 unit tests incl. analytic identity + JAX port of the official objective,
  all passing); adversarial review workflow launched; A2 training loop written (`ppg2ecg.training.train_a2`).
- **A2 launched (19:19)** at freeze commit `5276bb9` (preflight OK: same split/data/backbone, iMF bank hash `0f15f0d2…`, seed 42).
  Implementation review (multi-agent, adversarial) found no substantive deviation from the official objective; applied its
  recommendations before launch: body-level jax skip, validation JVP without autograd graph, bank-length assert, random row
  assignment of the r = t half in validation banks, 1-NFE diagnostics every epoch, non-finite checks on unweighted MSE/JVP,
  gradient-parity test vs the official JAX objective (float64, 1e-9). Memory: forward-mode JVP ≈ 0.51 GiB/sample ⇒ effective
  batch 64 via 2 × 32 accumulation (prereg §8).
- Implementation review (15 agents, 4 lenses + adversarial verification): 5 confirmed (2× jax-skip decorator — fixed; validation
  `enable_grad` — fixed; validation-bank r=t rows fixed to the first half — fixed by random row assignment; **shared-embedder
  `E(t)+E(h)` may under-resolve the interval `h` early in training** — kept as pre-registered, parameter-count constraint; logged as
  a candidate failure cause F7 for the taxonomy and as a limitation), 6 refuted (stale premises: driver/eval scripts exist,
  batch-64 infeasibility handled by accumulation, verdict rules made exhaustive), 21 low-severity notes.
- **A2 restarted (19:40)** at commit `1a2f9da` after a pre-result amendment: the shared-embedder conditioning `E(t)+E(h)` left the
  MeanFlow interval nearly invisible (cond variance 99.3 % explained by t+h; r decodable R² 0.18), so `h` is now scaled by 1000
  before the same sinusoidal embedder (t, h, r all decodable R² 1.00, **no added parameters**; DiT integer-timestep convention).
  The h_scale=1 run was stopped after epoch 1 (kept in `outputs/aborted/a2_hscale1_aborted_epoch1/`, no test result produced).
  Details in `docs/A2_IMEANFLOW_PREREGISTRATION.md` §9.
- **A2 restarted again (19:52, commit `62c2b15`)** — amendment 2: the h_scale=1000 run diverged in 2 epochs (train MSE 10.6 → 395,
  |du/dt| 9.7 → 20.7): the ×1000 scale amplifies the JVP term of the MeanFlow identity. Switched to the **official iMF conditioning:
  h-only (`cond = E(h)`, t inferred from z_t)** — resolves the interval fully with O(1) derivatives and no added parameters.
  Both aborted runs kept under `outputs/aborted/` (no test-set numbers produced). Prereg §9 records the full sequence.
- **A2 done (22:47)**: 81 epochs, best 61 (fixed-bank iMF MSE 0.1738), 3.24 h, peak 16.9 GiB, stable. Test S2, same paired noise as A0-b:
  | arm | NFE | HR err | morph | amp | gain | RMSE | seed std | ms/batch64 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | OT-CFM Heun 25 (A0-b) | 50 | 8.08 | 0.650 | 0.95 | 5.69 | 0.435 | 0.242 | 4171 |
  | OT-CFM Euler 1 (A0-b) | 1 | 41.96 | 0.217 | 0.15 | 0.24 | 0.304 | 0.033 | 82 |
  | **iMeanFlow 1 step** | **1** | **9.58** | **0.595** | **0.90** | **4.47** | 0.443 | 0.254 | 82 |
  | iMeanFlow 2 steps | 2 | 8.00 | 0.660 | 0.92 | 5.60 | 0.445 | 0.256 | 163 |
  | iMeanFlow 4 steps | 4 | 7.02 | 0.719 | 0.93 | 6.59 | 0.439 | 0.251 | 327 |
  Recovery of the 50→1 gap at 1 NFE: HR 0.96, morph 0.87, amplitude 0.93, conditioning 0.78; beats/ref 1.00 → **SUCCESS** (frozen rule).
  Residuals: morphology 0.595 vs 0.650 (CIs disjoint), occasional spurious spikes, high-HR under-estimation shared with the baseline.
  Report: `docs/A2_IMEANFLOW_REPORT.md`; figures `outputs/a2_…/figures/`.

## 2026-08-26 — Session 3: replication (A3 new DaLiA subject S1, A4 WildPPG)
- Pre-registered `docs/A3_A4_REPLICATION_PREREGISTRATION.md` (Part I frozen: test S1 / val S11 / train 13, recipes verbatim from A0-b
  and A2, replication verdict rule; Part II rules for WildPPG with values to be frozen after the audit). Manifest
  `split_a3_testS1_valS11.json` (sha256 `6d2999bd…`), leakage checks PASS. Generic launchers `scripts/run_exp.sh`, `run_eval_chain.sh`;
  `compare_a2.py` generalised (manifest-aware, replication verdict, pointwise-error inversion flag). WildPPG audit workflow started
  (read-only + official download only if no login/consent).
- WildPPG audit done (`docs/WILDPPG_AUDIT.md`): official ETH polybox share, anonymous download (19.6 GB, sha256), CC BY-NC-SA 4.0 (stale
  "review only" clause noted), 16 participants, 4 synchronised sites, green PPG 128 Hz, sternum lead-I ECG 128 Hz, no NaN (constant-filled
  gaps), 3 noisy-ECG participants. PENGUIN uses green PPG at all 4 sites (ECG tiled ×4); bookkeeping = 8 s windows.
- A4 Part II values frozen (prereg amendment): channel = PENGUIN's (4 green sites), 8 s @ 128 Hz, gap windows dropped (861, 0.22 %),
  split `split_a4_wildppg_seed42.json` (val an0,k2s / test kjd,ssx / train 12; sha256 `bc168144…`), val/test subsets ≤ 4096 windows,
  rounds of 220 steps. Processed `data/processed/wildppg_8s/` (389,355 windows). Leakage gate PASS (subject, window-hash, normalisation).
  Evaluation pipeline dry-run on WildPPG arrays with smoke checkpoints OK. A4 pipeline armed to start after A3 finishes (single GPU).
- **A3 done (05:29)**: OT-CFM(S1) 114 epochs (best 94, 2.35 h); iMF(S1) 36 epochs (best 16, 1.43 h). Test S1 (1,151 windows):
  | arm | NFE | HR | morph | amp | gain | RMSE |
  |---|---:|---:|---:|---:|---:|---:|
  | OT-CFM 50 | 50 | 8.16 | 0.683 | 0.87 | 8.77 | 0.448 |
  | OT-CFM 4 | 4 | 16.40 | 0.407 | 1.34 | 3.05 | 0.499 |
  | OT-CFM 1 | 1 | 35.23 | 0.168 | 0.21 | 0.28 | 0.347 |
  | **iMF 1** | **1** | **11.96** | **0.581** | **0.71** | **4.78** | 0.449 |
  Recovery HR 0.86 / morph 0.80 / amp 0.76 / gain 0.53; beats 1.03 → **REPLICATED** (A2 rule SUCCESS); pointwise-error inversion YES.
  Report `docs/A3_SUBJECT_REPLICATION_REPORT.md`. A4 (WildPPG) pipeline started automatically.
- **A4 OT-CFM (WildPPG) done 10:42**: 210 rounds × 220 steps (best 190, ~5 h, peak 20.7 GiB). Test subset (4,096 of 46.9 k windows, kjd+ssx):
  50 NFE HR 9.43 / morph 0.670 / amp 0.98 / gain 7.16 / **R-peak F1 0.44, PCC 0.07** (beat-level metrics carry information here — WildPPG's four
  devices are time-synchronised, unlike DaLiA's wrist/chest pair); 4 NFE HR 15.4 / morph 0.38 / amp 1.48; **1 NFE Euler: HR 15.6, morph 0.38,
  amp 0.32, QRS-width err 75 ms, gain 6.64, F1 0.48, seed std 0.035, RMSE 0.355** — i.e. on beat-synchronised data the 1-NFE conditional mean
  keeps rhythm/conditioning but loses amplitude and QRS sharpness (morphology collapse rather than total structural collapse as on DaLiA).
  Pointwise-error inversion again (RMSE 0.355 < 0.440). iMF (WildPPG) training launched automatically.
- **A4 iMF (WildPPG) done 14:43** (66 rounds, best 46, 3.6 h, peak 19.2 GiB). Test subset: iMF-1 HR 11.85 / morph 0.551 / amp 1.04 / gain 4.29 /
  beats 0.93 / F1 0.385 / RMSE 0.485. Recovery HR 0.61, morph 0.59, amp 0.91, gain **−4.5** (OT-1 already retains 6.64 of 7.16) →
  **PARTIAL** (both rules); inversion YES. Per site iMF-1 improves HR at all four sites. Reports: `docs/A4_WILDPPG_REPLICATION_REPORT.md`,
  `docs/REPLICATION_SUMMARY.md` → integrated verdict **SUBJECT-ROBUST, DATASET-UNCERTAIN** (A2 REPLICATED, A3 REPLICATED, A4 PARTIAL).
- **2026-08-26 A5 pre-registered (`cc28ad9`)** — Conditional-mean control: `S5ConditionalMeanRegressor` (PENGUIN backbone minus
  `pre_conv_target`/`timestep_embedder`, MSE only, AdamW 1e-3/wd 0.01/batch 64/seed 42, selection by deterministic validation MSE, patience
  20/min_delta 1e-4), three runs on the A2/A3/A4 manifests, H1–H4, QRS window ±100 ms, verdict thresholds, forbidden wording. GPU was
  occupied by an unrelated vLLM process (29.7 GB) at launch → pipeline gated on ≥ 22 GiB free; started 19:43 once it exited.
- **A5a zero-state run (aborted as a control failure)**: converged to a constant output (−0.348, per-window std 2.5e-7; val MSE plateau
  0.0983; RMSE 0.290 on S2, worse than the constant GT-mean predictor 0.250); A5b showed the same at epoch 1 and was stopped. Cause
  (gradient check): with x_t-embedding = 0 the block outputs are identically 0 and upstream zero-initialises `final_layer.linear.weight`,
  so only `final_layer.linear.bias` ever receives gradient. Archived `outputs/aborted/a5{a,b}_*_zero_state_deadstart/`.
- **Amendment 1 (`d961000`, before amended results)**: target stream fed a learned constant state token (128 params; total 3,990,787;
  819,200 adaLN weights inactive with cond = 0 → effective 2,907,393); mechanism + gradient flow unit-tested; NaN convention declared;
  everything else unchanged. Pipeline re-run 20:33 → 23:03: A5a 40 epochs (best 20, val MSE 0.0881, 45 min), A5b 31 (best 11, 0.0878,
  35 min), A5c 54 rounds (best 34, 0.0854, 64 min); peak 18.3 / 18.3 / 20.6 GiB.
- **A5 results**: regressor HR / morph / amp / gain / RMSE = S2 35.7 / 0.160 / 0.06 / 2.37 / 0.289; S1 32.3 / 0.148 / 0.05 / 1.32 / 0.318;
  WildPPG 19.2 / 0.331 / 0.25 / 5.70 / 0.343 with F1 0.436 (OT-50 0.440), RR MAE 16.7 ms. Closest model to the regressor on every dataset and
  every distance: OT-CFM 1-NFE (waveform RMSE 0.085 / 0.134 / 0.079 vs 0.26–0.35 for OT-50 and iMF-1; PCC 0.52 on WildPPG). Regressor has the
  best RMSE/MAE of all four models on 3/3 datasets while ranking last on amplitude/morphology (inversion YES ×3); QRS-window energy 15–27 %
  of GT for R (O1 27–35 %, O50 72–88 %, M1 61–109 %). Pareto: regressor dominated by iMF-1 at one evaluation everywhere. Frozen verdict
  **STRONG SUPPORT** (attenuation + closest(O1→R) on 3/3, WildPPG timing/conditioning preserved; H1–H4 all ✓). Report
  `docs/A5_CONDITIONAL_MEAN_CONTROL_REPORT.md`; artefacts `artifacts/a5_conditional_mean_control/`. Note: JAX-parity test in
  `tests/test_imeanflow.py` fails/hangs while another process holds the GPU (JAX init) — passes with the GPU free.
- **2026-08-27 A6 pre-registered (`2fc7841`)** — capacity-matched control `S5FullBackboneRegressor` (unmodified PENGUIN backbone,
  4,568,707 / 4,304,513 effective params, deterministic constant state + fixed t, MSE). Hard test before training: the spec's example
  (`x = ones`, `cond = E(0.5)`) does not train (constant solution, train MSE flat 0.1198 for 12 epochs; identical with x = 0.1); root cause
  isolated by 12-epoch runs: the *unscaled* fixed conditioning vector (coherent adaLN-weight updates ≈ 25× the bias rate) and a large
  constant stem output each block training; `cond = 0` or `cond = 0.05·E(0.5)` with x = 0.1 train (amp 0.11, beats 0.24); x = 1.0 with
  the scaled cond still fails. Frozen by the pre-stated order: **x_const 0.1, t 0.5, cond_scale 0.05** (all adaLN weights active).
  Screening traces `outputs/gradcheck_a6_*` (not results) summarised in `artifacts/a6_capacity_control/state_constant_screening.json`;
  gradient-flow artefacts (`gradient_flow*.json`). A6 pipeline started 01:46 (a → b → c).
- **A7 dataset audit (`docs/A7_ABP_DATASET_AUDIT.md`)**: MIMIC-BP v2.2 selected (UCI-BP has no subject identifiers); 878 MB downloaded
  from Harvard Dataverse (md5 OK), 1,524 subjects × 30 × 30 s @125 Hz, ABP raw mmHg, official train/val/test 1,100/195/229; PENGUIN
  ABP preprocessing = resample only (no normalisation), SBP/DBP = window max/min in mmHg; processed `data/processed/mimicbp_8s`
  (137,160 windows = PENGUIN's `sample_num`, 0 dropped). A7 pre-registered (`docs/A7_ABP_PREREGISTRATION.md`): three frozen models
  (OT-CFM A0-b recipe, iMF A2 recipe, A6 MSE proxy), A4 schedule unit, ABP metrics (`ppg2ecg.evaluation.abp_metrics`, unit-tested),
  ±150 ms systolic-peak region, shuffle penalties, H7.1–7.4, recovery score, verdict thresholds. Pipeline `scripts/run_a7_pipeline.sh`
  queued behind A6 (waits for `outputs/A6_DONE`).
- **A6 results (2026-08-27 01:46–04:16)**: a6a 44 epochs (best 24, val MSE 0.0883, 51 min), a6b 26 (best 6, 0.0888, 30 min), a6c 54
  rounds (best 34, 0.0837, 65 min). Full-backbone regressor: S2 HR 33.8 / morph 0.175 / amp 0.06 / RMSE 0.286; S1 30.9 / 0.184 / 0.04 /
  0.321; WildPPG 20.2 / 0.316 / 0.24 / 0.350, F1 0.421, gain 4.95 — same as A5 (waveform RMSE Rfull–Rsmall 0.042/0.044/0.045). Closest
  generative model to Rfull: OT-CFM-1 on 3/3 datasets (RMSE 0.082/0.120/0.078 vs 0.26–0.35 for OT-50/iMF-1). Frozen verdict
  **CAPACITY OBJECTION RESOLVED**. Report `docs/A6_CAPACITY_MATCHED_MEAN_CONTROL_REPORT.md`, artefacts `artifacts/a6_capacity_control/`.
  A7 pipeline started automatically on `outputs/A6_DONE`.
- **A7 results (2026-08-27 04:17–11:45, MIMIC-BP official split, 3,435-window test subset)**: OT-CFM 117 rounds (best 97, 2.5 h,
  val CFM 6.49); iMF 70 (best 50, 3.5 h, val 112.5); MSE proxy 66 (best 46, 1.2 h, val 219 mmHg²). SBP/DBP MAE [mmHg] — MSE 14.31/8.72,
  OT-1 15.09/9.51, OT-4 14.89/9.39, OT-50 15.94/9.80, iMF-1 16.28/21.82; pulse-template correlation 0.929 / 0.904 / 0.909 / 0.884 / 0.140;
  slope ratio 0.91 / 0.93 / 0.97 / 1.20 / 6.13; HF (>5 Hz) 0.022 / 0.024 / 0.022 / 0.037 / 0.550 (GT 0.043); systolic-peak F1 0.945 /
  0.913 / 0.919 / 0.883 / 0.336; RMSE 13.10 / 14.64 / 14.69 / 16.12 / 32.27. No attenuation at 1 NFE (it beats 50 NFE), MSE proxy best on
  every metric, no pointwise inversion, iMF-1 injects HF noise and loses conditioning (shuffle penalty 0.05 mmHg vs 1.93 for the proxy).
  Frozen verdict **NOT GENERALIZED**. Pareto-optimal: MSE proxy + OT-CFM 4/10/20; OT-50 dominated. Report
  `docs/A7_ABP_GENERALIZATION_REPORT.md`, artefacts `artifacts/a7_abp_generalization/`. **STOP after A7 as pre-registered** (no
  respiration experiments, no new methods).
