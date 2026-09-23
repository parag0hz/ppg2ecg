# External model audit — which independently developed conditional generators can take our inference wrapper without retraining?

Audit date 2026-09-23. Supersedes the table format of `docs/TT_EXPC_EXTERNAL_MODEL_AUDIT.md` (`b294575`), whose facts were
re-checked here against primary sources. Primary sources only: the papers (arXiv PDFs read in full text where accessible),
the official repositories at the stated commits, the checkpoint hosts' file listings. A repository is treated as *official*
only when the paper or the authors' own README links it. No model was downloaded or run for this document.

**Principle for everything that follows:** same external checkpoint, different inference only. *Training changed? — NO.*

---

## 1. RDDM
| Field | Value |
|---|---|
| Paper title | Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation |
| Authors | Debaditya Shome, Pritam Sarkar, Ali Etemad |
| Venue / year | AAAI 2024 (Proc. AAAI 38(13), 15009–15019); arXiv 2308.13568 (v. 2023-12-28, comment "Accepted at AAAI 2024") |
| Official paper URL | https://ojs.aaai.org/index.php/AAAI/article/view/29422 · https://arxiv.org/abs/2308.13568 |
| Official repository URL | https://github.com/DebadityaQU/RDDM — **official**: paper footnote 1, "We make the code public to the research community" → this URL |
| Repository commit | `7d5348843c3985c211a23ae5105a2d9497d5156a` (2024-05-08; latest) |
| Licence | MIT |
| Official pretrained checkpoint | Google Drive folder linked from the official README ("Pretrained diffusion weights"): **exactly three files** — `rddm_main_network.pth`, `rddm_condition_encoder_1.pth`, `rddm_condition_encoder_2.pth` (folder listing read 2026-09-23; hashes recorded at download). No DDPM-baseline weights, no split files. |
| Checkpoint provenance | Loaded by the official `std_eval.py` with `nT = 10` (`load_pretrained_DPM(..., nT=10, type="RDDM")`). The repository's own evaluation log (`eval-logs/standard_eval_RDDM.out`) reproduces the paper's Table 2 closely (e.g. WESAD RMSE 0.209 / FD 3.86 vs paper 0.21 / 3.93; DALIA HR 4.58 vs 4.49 bpm), consistent with the released weights being the paper model. **Not explicitly stated** that the released weights are the exact paper checkpoint. |
| Dataset used by checkpoint | Paper §4.1: one combined training set from **WESAD, DALIA, CAPNO, BIDMC, MIMIC-AFib**, "data from 80 % of the subjects for training and the remaining 20 % … for cross-subject evaluation" (`data.py` default `datasets=["BIDMC","CAPNO","DALIA","MIMIC-AFib","WESAD"]`). |
| Published split available? | **No.** `data.py` reads pre-built `ecg_{train,test}_4sec.npy` files from an internal NAS path; neither the files nor the subject lists are released; the paper gives only the 80 / 20 rule. CardioGAN (whose setup RDDM says it follows) publishes no split either (`pritamqu/ppg2ecg-cardiogan` @ `d7239b0`). The official test split **cannot be reconstructed**. |
| Input condition | 4 s PPG @ 128 Hz (512 samples). Paper: resample to 128 Hz, PPG band-pass 0.5–8 Hz (Butterworth), subject z-score, min-max to [−1, 1], 4 s windows. Code additionally applies per-window `minmax_scale(·, (−1, 1))` and `nk.ppg_clean(…, 128)` in the Dataset. Two `ConditionNet`s encode the PPG. |
| Generated output | 4 s lead-II ECG @ 128 Hz (512 samples); target ECG in code = `nk.ecg_clean(…, method="pantompkins1985")` of the min-max-scaled window (paper: high-pass 0.5 Hz). |
| Generative family | Conditional DDPM with ROI-masked forward noise; two UNet denoisers (`region_model` ρφ, `eps_model` εθ, both `DiffusionUNetCrossAttention(512, 1)`); ancestral sampling. |
| Original sampler | DDPM ancestral sampler over the full schedule (`diffusion.py`, `RDDM.forward(mode="sample")`): for i = T … 1, `x ← region_model(x, cond2, i/T)`; `ε ← eps_model(x, cond1, i/T)`; `x ← (x − ε·(1−α_i)/√(1−ᾱ_i))/√α_i + √β_i·z`. Linear β ∈ (1e−4, 0.2). |
| Original sampling steps | **T = 10** (paper §4.1 "set the sampling steps to 10"; `std_eval.py` `nT=10`) |
| Actual NFE per sample | **20 network evaluations** (one `region_model` + one `eps_model` call per step) + 2 condition-encoder passes per window (deterministic, computable once per window and shareable across samples of that window). |
| Reduced-step sampling supported? | **No.** The released weights were trained on the 10-step linear-β schedule (`train.py` builds `RDDM(n_T=nT)`; training timesteps `randint(1, n_T)`). `load_pretrained_DPM` accepts another `nT`, but that builds a *different* β schedule the weights never saw. The paper reports RDDM **only at T = 10**; its T = 25 / 50 rows are separately trained DDPM baselines, not RDDM at other depths. |
| Arbitrary respacing supported? | **No.** No DDIM / respaced sampler exists in the repository. Implementing one would be a new inference algorithm — excluded by this project's rules. |
| Multiple stochastic samples possible? | **Yes** — `x_T ~ N(0, I)` and fresh `z` at every step. |
| Original downstream functional | HR MAE (bpm) on **8 s windows** (two 4 s generations concatenated), **Hamilton** R-peak segmenter (`biosppy`), windows where the generated ECG yields no HR dropped (`metrics.py` `MAE_hr`); RMSE; FD on raw 512-sample vectors per batch of 512 (shuffled, seed 31), averaged over batches. CardioBench (AFib VGG-13 on STFT, BP 1D-UNet, diabetes, stress) — **code not released** (README checklist "[ ] CardioBench"). |
| Can apply ours without retraining? | **Consensus wrapper: yes** (K independent samples at the original T = 10, median of the functional). **Fixed-budget depth/width allocation: no** — only one valid depth exists (see the two rows above). |
| Reproduction risk | **HIGH** for reproducing the paper's numbers (split unrecoverable; `.npy` build code unreleased, filter orders unspecified). **MEDIUM** for "released checkpoint under the available evaluation setting". |
| External-validation value | **MEDIUM** for the consensus / functional-error-dependence part (independent diffusion family, independent authors, official weights). **LOW** for the fixed-budget allocation claim (not testable without a new sampler). |

## 2. PPGFlowECG
| Field | Value |
|---|---|
| Paper title | PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection |
| Authors | Xiaocheng Fang, Jiarui Jin, Haoyu Wang, Che Liu, Jieyi Cai, Guangkun Nie, Jun Li, Hongyan Li, Shenda Hong |
| Venue / year | arXiv 2509.19774 (2025); repository description "[ICDM 2026]" (acceptance not independently verified) |
| Official paper URL | https://arxiv.org/abs/2509.19774 |
| Official repository URL | https://github.com/PKUDigitalHealth/PPGFlowECG — **official**: the paper states "the code is available at https://github.com/PKUDigitalHealth/PPGFlowECG" |
| Repository commit | `56b2cd2cfa738388c60daccd788d511aa8698085` (2026-01-21) |
| Licence | **None** (no licence file) |
| Official pretrained checkpoint | Hugging Face `XiaochengFang/PPGFlowECG` @ `fdebf3e3` (linked from the official README): `VAE-iter-40000.pth` 3.82 GB (sha256 `c186633a…d7d6d`), `checkpoint-10.pt` 97 MB (sha256 `50f1af67…ecdb`); Baidu mirror. |
| Checkpoint provenance | **UNRESOLVED.** No model card. Which of the paper's four corpora (MCMED, MIMIC-AFib, VitalDB, BIDMC) the weights were trained on is not stated anywhere read. The repository config's `results_folder: results/rectified_flow/mcmed` *suggests* MCMED — not confirmed. |
| Dataset used by checkpoint | Unknown (see above). MCMED is a credentialed PhysioNet corpus and is **not in hand**. |
| Published split available? | MCMED: "followed its official data split". VitalDB / BIDMC / MIMIC-AFib: "partitioned at the subject level into 80 % training and 20 % testing" — **subject lists not released**. |
| Input condition | 10 s PPG @ 128 Hz, z-scored (paper §4.2), encoded by the Stage-1 CardioAlign encoder |
| Generated output | 10 s ECG decoded by a VAE from the flow's latent |
| Generative family | Latent rectified flow (Transformer velocity field) |
| Original sampler | Explicit Euler on a uniform grid, `RectifiedFlow.sample(shape, num_steps)` |
| Original sampling steps | **T = 10** (config `sampling_steps: 10`; paper Table 1 "PPGFlowECG (T = 10)") |
| Actual NFE per sample | 10 flow-network calls + 1 VAE decode + encoder pass(es) — **per-call count to be instrumented**, not verified |
| Reduced-step sampling supported? | **Yes** — `num_steps` is an argument; Euler on any uniform grid; the paper ablates sampling steps (Fig. 5) |
| Arbitrary respacing supported? | Any integer number of uniform Euler steps (native); non-uniform schedules not implemented |
| Multiple stochastic samples possible? | Yes (Gaussian latent prior) |
| Original downstream functional | MAE_HR, FD, FID (ECGFounder features), MCMED multi-label CVD classification, MIMIC-AFib AF detection (classifier code: `evaluation/calculate_metric.py`, not audited) |
| Can apply ours without retraining? | **Technically yes, including the fixed-budget grid** (native step control). **Not as a headline** while provenance is unresolved. |
| Reproduction risk | **HIGH** (provenance unresolved; likely-training corpus not in hand; 3.8 GB VAE; 10 s windows differ from every other model here) |
| External-validation value | **HIGH if provenance is resolved** (latent rectified flow = a third family; step count natively adjustable). **LOW–MEDIUM as it stands.** |

## 3. PENGUIN (already used; listed for completeness and to state its status precisely)
| Field | Value |
|---|---|
| Paper title | PENGUIN: General Vital Sign Reconstruction from PPG with Flow Matching State Space Model |
| Authors | Suzuki et al. (Neurogica) — full author list in `docs/PENGUIN_AUDIT.md` |
| Venue / year | ICASSP 2026 (oral); arXiv 2602.03858 (23 Jan 2026) |
| Official paper URL | https://arxiv.org/abs/2602.03858 |
| Official repository URL | https://github.com/Neurogica/PENGUIN (paper footnote) — pinned as submodule `external/PENGUIN` |
| Repository commit | `6cd70cd` (never modified; enforced by `assert_upstream_pinned()`) |
| Licence | BSD-3-Clause-Clear |
| Official pretrained checkpoint | **None released** (`docs/PENGUIN_AUDIT.md` §38; U1 report §2) |
| Checkpoint provenance | **All PENGUIN checkpoints in this project were trained by us**: U1 = upstream `train.py` run as shipped on six corpora; V1 / SR1 = the unmodified upstream model class trained with this project's recipe on VitalDB. |
| Dataset used by checkpoint | ours (U1: BIDMC, WESAD, UCI-BP, MIMIC-BP, PPG-DaLiA, WildPPG; V1 / SR1: VitalDB) |
| Published split available? | No (upstream draws its split at run time; U1 recorded the draws) |
| Input / output | 8 s (upstream) or 4 s (V1) PPG → ECG / respiration / ABP |
| Generative family | Flow matching (OT-CFM), Flow-SSM (S5) backbone |
| Original sampler / steps / NFE | Heun, `n_step = 25` → **50 NFE** (bit-exact reimplementation, `tests/test_upstream_parity.py`) |
| Reduced-step / respacing | Yes — any Euler or Heun step count on the uniform grid (EXP-A: one-step Heun is degenerate) |
| Multiple stochastic samples | Yes |
| Original downstream functional | HR, RR (respiration), SBP / DBP error — PENGUIN's own metric code ported verbatim |
| Can apply ours without retraining? | Yes (EXP-A, DW1, DW2-A, WD1-B) |
| Reproduction risk | LOW (U1: 6 of 8 published cells reproduced with the released pipeline) |
| External-validation value | **Independent architecture and objective, but not an independent checkpoint.** It must not be described as "an external pretrained model". |

## 4. MAGIC
| Field | Value |
|---|---|
| Paper title | MAGIC: Multi-Grained Conditional Diffusion for High-Fidelity PPG-to-ECG Translation |
| Authors | Wen H, Chang S, Zhou L, Liu W, Pei J, Jiang S |
| Venue / year | IEEE Journal of Biomedical and Health Informatics, 2026; DOI 10.1109/JBHI.2026.3713977 (PMID 42461735) |
| Official paper URL | https://doi.org/10.1109/JBHI.2026.3713977 (subscription) |
| Official repository URL | **Not found.** GitHub repository search (six query forms), web search, Europe PMC record; the full text is paywalled, so a code link inside it could not be checked. |
| Official pretrained checkpoint | Not found |
| Generative family / sampler | Conditional diffusion Transformer (adaLN global condition + gated cross-attention local conditions); abstract: "lower FD with 50 sampling steps than RDDM with 500 steps" |
| Can apply ours without retraining? | **No — unavailable.** Not re-implemented (rule). |
| Reproduction risk / value | n/a — **UNAVAILABLE** |

## 5. UA-P2E
| Field | Value |
|---|---|
| Paper title | Uncertainty-Aware PPG-2-ECG for Enhanced Cardiovascular Diagnosis using Diffusion Models |
| Authors | Omer Belhasin, Idan Kligvasser, George Leifman, Regev Cohen, Erin Rainaldi, Li-Fang Cheng, Nishant Verma, Paul Varghese, Ehud Rivlin, Michael Elad (Verily; Technion) |
| Venue / year | arXiv 2405.11566 v3 (2025-04-20); ICLR 2025 workshop listing |
| Official paper URL | https://arxiv.org/abs/2405.11566 |
| Official repository URL | **None.** The paper contains no code statement (full text searched for "github", "code", "open-source", "reproduc"); no repository found under the authors' names. |
| Official pretrained checkpoint | None |
| Dataset | MIMIC-III matched waveform database (generator), CinC 2021 (classification) |
| Generative family / sampler | Conditional diffusion, DDIM T = 100 (also 25 / 50 reported), K = 100 posterior samples |
| Original downstream functional | Classifier scores averaged over the K posterior ECG samples |
| Note | UA-P2E **already uses multi-sample posterior aggregation**; that part is not ours. The only question it could host is whether reallocating a fixed NFE budget from DDIM depth to more posterior samples improves *its own* aggregation. |
| Can apply ours without retraining? | **No — unavailable**; not re-implemented. Reproduction priority: lowest. |

---

## 6. Decision (per the preregistered selection criteria)
Criteria: (1) independently developed, (2) official code, (3) official checkpoint, (4) stochastic conditional samples,
(5) sampling depth changeable, (6) original inference reproducible in our environment.

| model | (1) | (2) | (3) | (4) | (5) | (6) | status |
|---|---|---|---|---|---|---|---|
| RDDM | ✓ | ✓ | ✓ | ✓ | **✗** (fixed T = 10 schedule) | to be established in R0 (split unrecoverable) | **proceed to reproduction** — consensus wrapper testable; fixed-budget allocation **not** testable |
| PPGFlowECG | ✓ | ✓ | ✓ | ✓ | ✓ | provenance unresolved | **hold** until the checkpoint's training corpus is confirmed |
| PENGUIN | ✓ architecture | ✓ | **✗** (our training) | ✓ | ✓ | ✓ (U1) | in use; not an external checkpoint |
| MAGIC | ✓ | ✗ | ✗ | — | — | — | unavailable |
| UA-P2E | ✓ | ✗ | ✗ | — | — | — | unavailable |

**Consequence stated before any experiment:** no model in hand satisfies all six criteria. RDDM is the only independent
official checkpoint that can run, and on it only the *width* half of the claim (same checkpoint, K samples at the original
depth, at increasing budget) and the functional-error-dependence measurement at one depth are testable. The *fixed-budget
depth-versus-width* question can be asked of an external checkpoint only through PPGFlowECG, and only after its provenance
is resolved. If RDDM's reduced-step sampling is shown in R0 to be invalid, that is reported as **"sampler does not support
shallow inference"**, not as a failure of the method to transfer.
