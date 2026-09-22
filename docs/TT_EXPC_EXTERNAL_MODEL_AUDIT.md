# EXP-C / C0 — External PPG→ECG generator availability audit

Audit date 2026-09-22 (prereg `50948e2`, EXP-C C0). Sources are the official repositories and papers only; every claim below
was read on the linked page or in the downloaded file. No model was run and no metric was computed for this audit. Rule
applied: a model without an official implementation is recorded as unusable, not re-implemented.

## Summary table
| paper | venue / year | model family | original sampling (steps → exact NFE) | official code | pretrained ckpt | compatible dataset in hand | stochastic samples? | sampler supports matched-NFE grid? | usable? |
|---|---|---|---|---|---|---|---|---|---|
| **PENGUIN** (Flow-SSM, OT-CFM) | 2025 (ours: `external/PENGUIN` @ `6cd70cd`) | flow matching, Heun ODE | Heun 25 steps → **50 NFE** (2 calls / step; repository `nfe_of`) | yes (in hand) | none released upstream; **U1 checkpoints trained as shipped** (BIDMC, WESAD, UCI-BP, MIMIC-BP, DaLiA, WildPPG) + our V1/SR1 VitalDB checkpoints | VitalDB, WildPPG, BIDMC, MIMIC-BP, … | yes (Gaussian z₀) | yes (Euler / Heun, any step count) | **yes** (EXP-A, EXP-D) |
| **RDDM** — Shome, Sarkar, Etemad, *Region-Disentangled Diffusion Model for High-Fidelity PPG-to-ECG Translation* | AAAI 2024 (vol. 38, pp. 15009–15019); arXiv 2308.13568 | DDPM with region (ROI) model, two UNets (`DiffusionUNetCrossAttention` × 2) + two `ConditionNet`s | `n_T = 10` reverse steps; **each step calls `region_model` and `eps_model` once → 2 network calls / step → 20 NFE** per sample (+ 2 condition-encoder passes per window, once) | **yes**: github.com/DebadityaQU/RDDM @ `7d53488` (2024-05-08), MIT | **yes**: Google Drive folder linked in README (`rddm_main_network.pth`, `rddm_condition_encoder_1.pth`, `rddm_condition_encoder_2.pth`; also `ddpm_main_network_{nT}.pth` for the naive DDPM baseline) — **not yet downloaded; hash to be recorded on download** | trained on the concatenation of BIDMC, CAPNO, DALIA, MIMIC-AFib, WESAD (4 s @ 128 Hz, min-max to [−1, 1], `data.py`); their `.npy` train/test splits are **not** in the repo — split membership unknown ⇒ our BIDMC / CapnoBase / DaLiA recordings may overlap their **training** set | yes (`x_i = randn`, fresh `z` each step) | `n_T` is a constructor argument tied to the β schedule (`ddpm_schedule(β₁, β₂, T)`); the released weights are for n_T = 10. Shallower runs must use a **respaced** schedule (DDIM-style) or the naive-DDPM weights released at other `nT` — respacing is not implemented upstream | **partial / conditional**: same-checkpoint before/after is feasible with an explicitly documented respacing rule; the "original setting" can be reproduced only against the logged numbers (`eval-logs/standard_eval_RDDM.out`: WESAD RMSE 0.2092 / FD 3.86, CAPNO 0.1947 / 2.96, DALIA 0.2507 / 5.85, BIDMC 0.2355 / 6.71, MIMIC-AFib 0.2229 / 6.75; HR MAE WESAD 1.35, DALIA 4.58 bpm at 8 s) **only if their split files are obtained**; otherwise every number is "our data, their checkpoint" and is flagged as such |
| **MAGIC** — Wen, Chang, Zhou, Liu, Pei, Jiang, *MAGIC: Multi-Grained Conditional Diffusion for High-Fidelity PPG-to-ECG Translation* | IEEE JBHI 2026, DOI 10.1109/JBHI.2026.3713977 (subscription) | conditional diffusion Transformer (adaLN global condition + gated cross-attention local conditions) | abstract: "MAGIC also achieves lower FD with 50 sampling steps than RDDM with 500 steps" → 50 steps (network calls per step not verifiable without the paper) | **not found**: GitHub repository search (four queries), arXiv title search and Europe PMC record show no code link; full text paywalled | none found | four public datasets (abstract), incl. MIMIC-AFib | presumably (diffusion) | unknown | **no** (no official implementation; not re-implemented) |
| **UA-P2E** — Belhasin et al. (Verily / Technion), *Uncertainty-Aware PPG-2-ECG for Enhanced Cardiovascular Diagnosis using Diffusion Models* | arXiv 2405.11566v3 (2025-04-20); ICLR 2025 workshop listing | conditional diffusion, DDIM sampling, K = 100 posterior samples, downstream classifier-score aggregation | DDIM T = 100 (also T = 25, 50 reported); NFE = T per sample (single network) — from the paper text | **none**: no code statement in the paper (0 hits for "github", "open-source", "reproduc"); no repository found on GitHub | none | trained on MIMIC-III matched waveform DB; classification on CinC 2021; the paper itself notes RDDM and CardioGAN had no public code at the time | yes (by design) | n/a | **no** (no official implementation) — C3 cannot run |
| **PPGFlowECG** — Fang et al. (PKU Digital Health), *Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and CVD Detection* | ICDM 2026; arXiv 2509.19774 | two-stage: CardioAlign encoder + **latent rectified flow** (Transformer), explicit Euler in latent space | `sample.sampling_steps: 10` (config), Euler, 1 network call / step → **10 NFE** in latent space + VAE decode (1 fixed call) | **yes**: github.com/PKUDigitalHealth/PPGFlowECG @ `56b2cd2` (2026-01-21), no licence file | **yes**: HF `XiaochengFang/PPGFlowECG` (`VAE-iter-40000.pth`, `checkpoint-10.pt`; results folder in config = `results/rectified_flow/mcmed`) and a Baidu mirror | paper datasets MCMED, MIMIC-AFib, **VitalDB, BIDMC** (10 s windows @ 128 Hz, z-score); the released checkpoint appears to be the MCMED model (config path) — MCMED is not in hand; VitalDB / BIDMC are, but their split rule (`data_process_to_npz`) would have to be reproduced | yes (Gaussian latent) | yes (Euler, any step count; 1-step is a defined sampler for rectified flow) | **conditional**: same-checkpoint before/after is feasible on VitalDB / BIDMC **as a cross-dataset transfer** unless the checkpoint's training set is confirmed; original-setting reproduction requires MCMED |
| CardioPPG — Yukui-1999/CardioPPG (npj Digital Medicine 2025) | 2025 | (repository description: "AI modelling PPG to ECG … pilot study"); generator family not established in this audit | — | yes (repo) | not established | not established | not established | — | **not assessed** (not a diffusion / flow model as far as the README shows; out of EXP-C scope) |
| PG-LRF — arXiv 2605.12541 (2026-05) | preprint | physiology-guided latent rectified flow | — | not found in this audit | — | — | — | — | **no** (no implementation found) |
| Multichannel ViT PPG→ECG — arXiv 2505.21767 | preprint | deterministic ViT regressor | n/a | not found | — | — | no (deterministic) | n/a | **no** (not a stochastic generator) |

## Decisions for C1–C3 (per the preregistration's decision rules)
1. **C1 (RDDM)** is the only priority-1 external candidate that can run. Before any number: (a) download the three checkpoint
   files and record their SHA-256; (b) instrument `RDDM.forward(mode="sample")` to count network calls (expected 2 × n_T = 20 at
   n_T = 10, plus the two condition encoders once per window); (c) ask whether the original train/test `.npy` splits can be
   obtained (the repository ships none; `eval-logs/` gives the published-run numbers). If the splits cannot be obtained, C1 runs
   as **"their checkpoint, our recordings"** on BIDMC 4 s / CapnoBase 4 s / DaLiA (windows built by our pipeline, then min-max
   scaled as `data.py` does), with the caveat that some of our test recordings may have been in RDDM's training set — this
   caveat applies identically to the "before" and "after" arms, so the paired before/after comparison remains valid while any
   absolute number is not comparable to the paper. (d) Matched-NFE factorisations at B = 20: the released n_T = 10 schedule
   defines only the 10-step sampler; shallower per-sample depth requires respacing the linear β schedule (DDIM-style
   deterministic or stochastic respacing over a subset of the 10 timesteps). The respacing rule will be written into a C1
   amendment **before** any C1 number, validated on the training-side data (never on test), and its effect at K = 1 reported next
   to the original 10-step sampler so a degraded shallow baseline cannot be mistaken for a width effect. If no respacing
   reproduces the original K = 1 quality within a preregistered tolerance at the full 10 steps, C1 reports only the
   width-only comparison that keeps the original sampler: (K, 10 steps) for K ∈ {1, 2, 4, 8} at **increasing** budget (20 · K NFE),
   which answers "does consensus help RDDM" but not "at equal compute".
2. **C2 (MAGIC)**: not runnable — no official implementation. Recorded; no re-implementation.
3. **C3 (UA-P2E)**: not runnable — no official implementation; recorded.
4. **PPGFlowECG** replaces MAGIC as the second external candidate (official code + checkpoint + a native Euler sampler that
   supports every step count, so the matched-NFE grid is exact and no respacing is needed). Preconditions before any number:
   confirm the released checkpoint's training corpus from the HF model card / authors; obtain or reproduce their VitalDB or
   BIDMC preprocessing (`data_process_to_npz/step1.py`, `step2.py`, 10 s @ 128 Hz, z-score) so that the "original" arm is their
   sampler on their data format; count NFE = Euler steps (+ VAE decode, reported separately and identical across arms).
   Validation-selected allocation on a held-out slice of *their* training split if MCMED is unavailable — otherwise on a slice
   of VitalDB training patients disjoint from our V1 test — and one test evaluation.

## What was not verified
- RDDM checkpoint hashes and the exact `DiffusionUNetCrossAttention` constructor (model.py not read in this audit).
- MAGIC's per-step NFE and any code link inside the paywalled full text.
- PPGFlowECG's checkpoint provenance (which corpus) and licence.

Sources: RDDM repository (`README.md`, `data.py`, `diffusion.py`, `std_eval.py`, `eval-logs/standard_eval_RDDM.out` at commit
`7d53488`); AAAI 2024 paper page; MAGIC Europe PMC record for PMID 42461735; UA-P2E arXiv 2405.11566 PDF (v3); PPGFlowECG
repository at `56b2cd2` (`README.md`, `config/latent_rectified_flow.yaml`, `model/latent_rectified_flow/rectified_flow.py`),
HF model listing `XiaochengFang/PPGFlowECG`, arXiv 2509.19774 PDF; arXiv API records for 2605.12541 and 2505.21767.
