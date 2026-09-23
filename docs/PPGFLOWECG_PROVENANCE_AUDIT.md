# PPGFlowECG — checkpoint provenance audit (PART B; no inference was run)

Audit 2026-09-23. Every check below is reproduced by `scripts/pfe_provenance.py` → `artifacts/pfe_provenance/provenance.json`.
**No PPGFlowECG inference was run** (rule: none before provenance is confirmed). Checkpoints were opened only after a
static, non-executing parse of their pickles showed nothing but torch / collections types, and then with
`torch.load(weights_only=True)`.

## B1 — Official sources
| item | value |
|---|---|
| Paper | "PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection", Fang et al., arXiv 2509.19774 — **v2, 2026-01-21** (v1 2025-09-24); repository description "[ICDM 2026]" |
| Official code | https://github.com/PKUDigitalHealth/PPGFlowECG (linked in the paper) @ `56b2cd2cfa738388c60daccd788d511aa8698085` (2026-01-21). 16 commits 2025-09-23 → 2026-01-21, all by the first author; no licence file |
| Official checkpoint host | Hugging Face `XiaochengFang/PPGFlowECG` (linked from the official README) + a Baidu mirror (not audited) |
| HF history | 2 commits: `5852f3ad` 2025-09-25 "initial commit" (tree: `.gitattributes` only); `fdebf3e3` 2025-09-26 "Upload 2 files" (empty message). **No model card at any revision** (`cardData: None`). |
| `checkpoint-10.pt` | 97,361,007 bytes; sha256 `50f1af67ad258ef5eea7ac0c09525f156f6b1997a3068575f08d4d8c6a05ecdb` (local = HF LFS); added in `fdebf3e3` |
| `VAE-iter-40000.pth` | 3,820,526,453 bytes; sha256 `c186633aa40fa2b2e32421588b8e68d7f95a30e8069f8161fbdbcedd97347d6d` (local = HF LFS); added in `fdebf3e3` |

## B2 — Checkpoint introspection
| | `checkpoint-10.pt` (latent rectified flow) | `VAE-iter-40000.pth` (Stage-1 CardioAlign encoder / VAE) |
|---|---|---|
| pickle globals (static parse) | `collections.OrderedDict`, `torch.{Bool,Float,Long}Storage`, `torch._utils._rebuild_tensor_v2` — safe | `collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor_v2` — safe |
| top-level keys | `step`, `model`, `ema`, `opt` | `encoder_ecg`, `encoder_ppg`, `decoder_ecg`, `decoder_ppg`, `optimizer`, `scheduler`, `iteration`, `hparams` |
| iteration | `step = 10000` = the config's `max_steps` (`checkpoint-10` = 10th save at `save_cycle` 1000, i.e. the **final** step of a run) | `iteration = 40000` |
| embedded config / run name / output path / data path | **none** (0 path-like strings) | **none** (0 path-like strings) |
| corpus identifiers (MCMED, VitalDB, BIDMC, MIMIC, AFib, …) | **none** among 2,330 string constants | **none** |
| other | 6,857,988 parameters (latent 4 × 40, d_model 256, 4 blocks, 4 heads — the shipped `latent_rectified_flow.yaml`); full EMA and AdamW state (β = (0.9, 0.96) as the config; final lr 6.25e-6) | cosine scheduler state; `hparams` = `kld_weight`, `lambda_align`, `lambda_cross`, `lambda_infonce` only |

What the save code writes (repository) matches exactly and contains **no dataset field by design**: the flow solver saves
`{"step", "model", "ema", "opt"}` (`engine/solver.py`), the VAE trainer saves the four state dicts, optimizer, scheduler,
iteration and four loss weights (`model/cardioalign_encoder/train.py`).

## B3 — Repository / config trace (circumstantial evidence)
| evidence | location | points to |
|---|---|---|
| VAE save path hard-codes `mcmed`: `args.save_dir = os.path.join(args.save_dir, "mcmed")`, file name `VAE-iter-{iteration}.pth` | `model/cardioalign_encoder/train.py:214, 224` | MCMED; the released file name matches this code path |
| Flow `results_folder: results/rectified_flow/mcmed`, files named `checkpoint-{milestone}.pt` | `config/latent_rectified_flow.yaml:18`, `engine/solver.py:124` | MCMED; `checkpoint-10` = milestone 10 of `max_steps` 10000 |
| Flow training dataset hard-coded `dataset='MCMED'` | `main.py:124` | MCMED |
| VAE config `dataset: "MCMED"`; dataset class default `dataset='MCMED'` (per-subject `.lmdb`) | `config/cardioalign_encoder.yaml:19`, `utils/ppgecg_dataset.py:13` | MCMED |
| The only data processor released is `MCMEDProcessor` | `data_process_to_npz/step1.py:9` | no released loader for VitalDB, BIDMC or MIMIC-AFib (`vitaldb==1.5.6` appears only in `requirements.txt`) |
| README (every one of 12 revisions): "You can use our pre-trained model weights for fast PPG to ECG" — **no corpus stated**; the abstract in early revisions calls this "the first study to experiment on MCMED" | README history | the paper's focus is MCMED, not a statement about the released file |

## B4 — HF history
Two commits, no messages, no model card in either tree, no deleted files (the initial tree holds `.gitattributes` only). Nothing
in the HF history identifies the training corpus.

## Step flexibility (for the second axis)
- **Training** samples `t ~ Uniform[0, 1]` continuously (`RectifiedFlow.forward`: `t = torch.rand((b,))`, straight path
  `x_t = (1−t)·x0 + t·x1`, target drift `x1 − x0`; `train_num_points = 4` = four independent random t per batch, **not** a
  time grid). No schedule is stored in the checkpoint (contrast: RDDM stores its 10-step schedule as buffers).
- **Sampling** is explicit Euler on a uniform grid with `num_steps` as an argument (`RectifiedFlow.sample`, `sample_shift`);
  the solver and config default to 10 steps (`sampling_steps: 10`).
- **The authors vary the step count at inference:** paper §4.5 "Parameter Analysis": "To examine the effect of the ODE solver
  step count, we vary T from 5 to 25. As shown in Fig. 5, PPGFlowECG achieves its best performance at T = 5, and performance
  degrades as T increases. For the main experiments, we set T = 10 to match the default inference setting of baseline
  methods (e.g., RDDM)" (MCMED; Table 4 / Fig. 5).
- Not established: behaviour below T = 5 (the sampler defines T = 1…4, the authors did not evaluate them); per-sample NFE
  also includes one VAE decode per generated sample and one PPG-encoder pass per window, which a fixed-budget accounting
  would have to include.

## Verdict (two axes, per the rubric fixed in the request)
- **PROVENANCE UNRESOLVED.** Neither checkpoint embeds any corpus, path or run metadata, the model card does not exist, and no
  README revision or paper statement says which corpus the released files were trained on. The rubric's "STRONGLY IMPLIED"
  requires the checkpoint's own embedded metadata to name the corpus; here only the *surrounding* code and config do. The
  circumstantial evidence is nonetheless strong and one-directional: the released file names reproduce the MCMED-specific
  save paths of the released code, and the released code can load no corpus other than MCMED.
- **VARIABLE-STEP INFERENCE VALID** — continuous-time training, native step argument, and the authors' own inference-time
  sweep over T = 5…25; T < 5 is within the sampler's definition but outside what the authors evaluated.

**`PROVENANCE UNRESOLVED + VARIABLE-STEP VALID` → not promoted; no fixed-budget experiment is run.**
Even with provenance confirmed as MCMED, a fair original-setting reproduction would need MCMED (credentialed PhysioNet
access; **not in hand**); the only immediately available evaluation would again be a zero-shot transfer (e.g. VitalDB), which
is a different question from the one the fixed-budget claim asks. Author questions: `docs/PPGFLOWECG_AUTHOR_QUERY.md`
(prepared, not sent).
