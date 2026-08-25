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
  kept) as `6e5a4f1`, pushed to `main` (no force). Author identity set repo-locally (an e-mail later found to belong to an unrelated account); push used the
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
- **Git authorship fix (2026-08-25 18:05):** the author e-mail used for the first commits (an e-mail registered to an unrelated account) is linked on GitHub to
  a different account (an unrelated third-party account), so GitHub attributed our commits to it. History was rewritten (author/committer →
  `parag0hz <131474134+parag0hz@users.noreply.github.com>`, trees byte-identical, root commit `9fd4ce5` untouched) and force-pushed
  with lease; the pre-rewrite history is kept locally as branch `backup/main-before-author-fix`. **SHA mapping (old → new):**
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
