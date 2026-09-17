# VM1 — Official ("vanilla") improved MeanFlow adapted to PPG → ECG, with and without ImageNet pretraining (preregistration)

Frozen and pushed before any VM1 weight update. VitalDB V1 data and split. Single seed (42) by user decision.
Runs after BB1 finishes (same GPU).

## Model — port of the official iMF (Lyy-iiis/imeanflow)
- Reference code: JAX `external/iMeanFlow` @ `bf60cd7` (unmodified) and the official PyTorch branch `torch` @ `0468798`
  (used only for parameter names). Port: `src/ppg2ecg/models/imf_dit.py`, `src/ppg2ecg/flow/imf_vanilla.py`.
- Kept: DiT with RoPE attention + QK RMSNorm, SwiGLU (8/3), zero-initialised residual gates and output layers,
  in-context tokens for h, CFG scale ω and CFG interval, shared trunk + u head + auxiliary v head, conditioning on
  h = t − r only; the official objective (logit-normal t, r with half r = t; ω on [1, 8]; guidance interval; guided
  v target from the v head; 10 % condition dropout; interval-masked v as JVP tangent; adaptive weight on u and v
  losses) and the official multi-step sampler.
- Adapted: 1-D patches of 8 samples (4 s window → 64 tokens); the class label → the PPG window, as one learned
  condition token per PPG patch plus that patch's embedding; the CFG "null class" → a learned null PPG embedding.

| arm | architecture | init | params (train / inference) |
|---|---|---|---|
| **S** | hidden 192, depth 6, 6 heads, aux-head depth 4 | scratch | 4.81 M / 3.04 M |
| **P** | official **B/2** trunk: hidden 768, depth 12, 12 heads, aux-head depth 8 | **official ImageNet checkpoint** | 144.9 M / 88.3 M |
| **B** | same as P | scratch | 144.9 M / 88.3 M |

P init: `huggingface.co/Lyy0725/iMF` `iMF-B-2.pth` (MIT licence), sha256 `5923f151471ad7d6dbc41bd06ffd8fb9b65e438b0d078a594ff6a3de6971b5b2`. Every name- and
shape-matching tensor is copied (88.14 M params: trunk, u head, embedders, tokens); the checkpoint has no v head,
so each v-head block is copied from the u-head block of the same index; the image patch embedder, output
projections, class table and class tokens are replaced by fresh task layers (0.12 M params).

## Training — official optimiser, project budget
AdamW (weight decay 0, betas 0.9 / 0.95), lr 1e-4, linear warmup over 220 steps then constant (official
warmup-const); batch 64; exactly 14,000 steps; seed 42; `checkpoint_last.pt` evaluated.
**Deviations from the official recipe, fixed now:** no EMA (a 0.9999 EMA after 14,000 steps still holds 25 % of the
initial weights) and no checkpoint selection, as for every other VitalDB arm. Micro-batch: 64 for S; for P and B the
largest of {64, 32, 16} that completes a 3-step dry run (gradient accumulation to 64; only the per-micro-batch split
of r = t rows and dropped conditions changes).

## Evaluation
1. **CFG setting, per arm, on the 289 validation patients only** (NFE 1, noise seeds 0–1): grid ω ∈ {1 (none), 1.5, 2,
   3, 5} × interval ∈ {(0, 1), (0.4, 0.65)}; choose the lowest patient-macro HR error; within 0.01 bpm of the best,
   the highest R-peak F1.
2. **Test (1,156 patients)** at that setting: the BB1 protocol — NFE 1, 2, 4 with noise seeds 0–3, HR consensus K = 16
   at NFE 1; plus NFE 1 without guidance (descriptive). Paired patient-clustered bootstrap against iMF arm I
   (`outputs/bb1_eval/arm_I.npz`, same windows and noise seeds).

## Decisions (at NFE 1; the BB1 rule)
IMPROVES: HR diff ≤ −1.0 bpm with CI upper < 0, or F1 diff ≥ +0.02 with CI lower > 0. WORSE: the mirror. Both → MIXED;
neither → NO MEANINGFUL CHANGE.
- **S vs I, P vs I, B vs I** — does the official iMF design beat the PENGUIN-backbone iMF?
- **P vs B** — does ImageNet pretraining help at equal architecture?
PENGUIN-50 (V1) is quoted beside them, unpaired. Nothing is tuned after results.
