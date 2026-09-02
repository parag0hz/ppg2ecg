# R3 — Disentangled Target-Side Rhythm Fusion + Adaptive Reliability Gating — PREREGISTRATION

**Status:** frozen at this commit, pushed together with `docs/R3_TARGET_STREAM_HOOK_AUDIT.md` **before any
R3 weight update** (the 100-step preflight is a weight update and runs only after this commit).
**Type:** method development / falsification. **NO full generator retraining. NO test. NO flow-objective
change. NO new R / event auxiliary loss. NO multi-seed. NO large Transformer. NO novelty claim. NO SOTA
claim. NO PENGUIN modification.**

---

## 0. Standing disclosures

- **R3 was designed after observing R1 and R2** (`docs/R1_PPG_GLOBAL_RHYTHM_OBSERVABILITY_REPORT.md`,
  `docs/R2_RHYTHM_SCAFFOLD_TRANSFER_REPORT.md`). It is **not independent confirmatory evidence**.
- **Supervision disclosure.** The R1 Global-TCN was trained with GT ECG R-peaks as labels. At R3 inference
  only PPG enters the Global-TCN; **GT R is unavailable to every deployable arm**. The distinction is one of
  training supervision and is never hidden; any comparison with other methods must give them equivalent
  event supervision.
- **ORACLE remains target leakage by design** and is diagnostic only; every ORACLE row is labelled
  *"(GT-R leakage; diagnostic only)"* (label applied by arm-set membership, never by string equality).
- **The gate is not assumed to be calibrated confidence.** It is the *Adaptive Rhythm Gate*; the phrase
  "confidence-calibrated" is **never** used in any R3 document. If the post-hoc diagnostic (§20) passes, the
  strongest permitted wording is "reliability-like behaviour observed".
- **Scaffold reliability shift** (R2 §0) persists: the Global-TCN was trained on 10 of the 12 R3 training
  subjects; the scaffold is in-sample during R3 training and out-of-sample on an0/k2s.
- **The hook is not a pure "WHEN" port.** `docs/R3_TARGET_STREAM_HOOK_AUDIT.md` §2.1 shows that `z_e` reaches
  the decoder through three routes, one of which is the raw per-block residual (`dx_t = z_e + …`, ×4 blocks)
  that bypasses the S5 / MLP paths and carries a large share of the first-order output response on the frozen weights
  (measured 0.21–0.70 depending on h and perturbation size, §2.1 of the audit). A fusion output added to `z_e` therefore also writes, to first
  order, directly into the decoder's input. The spec-literal formula `H_z' = H_z + (g ⊙) out` is kept
  (cancelling the residual would alter the frozen forward structure); the consequence is disclosed here and
  is measured by a preregistered, retrain-free **direct-route attribution diagnostic** (§18.2). Any WHEN /
  WHAT wording in the report is conditioned on that diagnostic.
- **R2 facts this design reacts to** (NFE 4, frozen 2,048 windows): B F1 excess 0.3176; ADD (R2 TRUE) 0.3369
  (+0.0194 vs B, CI > 0 but below +0.02); TRUE − SHUFFLE +0.0102; ADD worsened QRS-core derivative RMSE
  (−0.0013) and curvature error (−0.0010) vs B; degradation grew with injection magnitude (SHUFFLE < TRUE <
  ORACLE); ankle F1 excess worsened (−0.0063). R2's ORACLE ceiling through the additive path was +0.0425.

## 1. Questions

**Q1.** Can rhythm information be transferred through a **separate target-side temporal-fusion path**
without contaminating the PPG morphology representation?
**Q2.** If ungated fusion still harms unreliable regions, can an **inference-safe adaptive temporal gate**
suppress that harm without sacrificing event gain?

## 2. Provenance and frozen components

| item | value |
|---|---|
| start HEAD | `5e064ef659fb86b97fc4c28844ca9057a00c4dfd` == origin/main, clean |
| submodules | PENGUIN `6cd70cdefb91f10efeb8dce34019b5067cb25344`, iMeanFlow `bf60cd7cb653f6628e59d48034b333c5eba445e2` |
| C2 | deferred, zero weight updates; `outputs/r3_*` does not exist |
| test subjects `kjd`, `ssx` | **never loaded** (`assert_no_test_subjects` in every R3 entry point) |
| GENERATOR | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt`, round 45, file sha256 `557c7054…`, state_dict sha256 `47d7ccb9…` (= the A4 weights); 4,568,707 parameters in the state_dict (of which 264,194 belong to the never-executed `cross_attn` / `revin`; effective 4,304,513) |
| R1 GLOBAL-TCN | `outputs/r1_global_tcn_seed42/checkpoint_best.pt`, file sha256 `bfe76ea6…`, state_dict sha256 `0986a7af…`, 328,897 parameters |
| R2 TRUE ADDITIVE ADAPTER (arm ADD) | `outputs/r2_true_adapter_seed42/adapter_step2200.pt` (resolved from `artifacts/r2_rhythm_transfer/provenance.json` `adapters.true`), file sha256 `2d577897…`, adapter state_dict sha256 `f98057ca…`, 128 weights, L2 7.6975 — **never retrained** |
| R2 ORACLE ADDITIVE ADAPTER (diagnostic reference only, §17.3) | `outputs/r2_oracle_adapter_seed42/adapter_step2200.pt`, file sha256 `2802292b…`, state sha256 `c8827b1b…` |
| R2 SHUFFLE manifest / ORACLE cache (reused, §8) | `artifacts/r2_rhythm_transfer/shuffle_manifest.csv`; `artifacts/r2_rhythm_transfer/_cache_oracle_train.npz` sha256 `2e6c548c…` (not duplicated; path + sha recorded) |

Every hash is re-verified by the loaders and written to `frozen_checkpoint_manifest.json` at first use;
any mismatch or missing file → **STOP**.

## 3. Hook (from `docs/R3_TARGET_STREAM_HOOK_AUDIT.md`, `hook_audit.json`)

`H_z = z_e = backbone.pre_conv_target(z)` in `MeanFlowS5.u` (src, L49): `[B, C=128, T=1024]`, index-aligned
with the waveform sample axis (receptive field z[t−30 … t+32]), PPG not merged within one evaluation of `u`
(PPG meets the target stream only inside each block), before the four blocks and the final decoder; because
the target stream is not chained, a residual on `z_e` reaches all four blocks identically and, through the
per-block residual, the decoder input directly (§0). R3 inserts `H_z' = H_z + (g ⊙) fusion(H_z, s)` there in a
src/ subclass of `MeanFlowS5`. **R3 never touches `backbone.pre_conv_ppg`**, never modifies
`external/PENGUIN`, and never concatenates the scaffold into the PPG: the scaffold travels to `u()` as
channel 1 of the R2 `[B, 2, T]` carrier and is split before any backbone call; `pre_conv_ppg` receives
`ppg2[:, :1]` only (forward-hook test). Temporal length is exactly 1024 (spec §2).

## 4. Frozen sets and the trainable set

Generator (4,568,707) and Global-TCN (328,897): `requires_grad = False`; after the first optimizer step of
every process every frozen parameter is asserted to have `.grad is None` (STOP otherwise). The R2 adapter in
arm ADD is loaded frozen. **The only trainable parameters are the R3 fusion-module parameters** (§6–7; exact
name sets pre-declared there), asserted per arm and written to `parameter_manifest.json`.

## 5. Input rhythm scaffold

Exactly the R2 field: `s = sigmoid(GlobalTCN(PPG)).detach()`, `[B, 1, 1024]`, dense pre-NMS, no threshold,
no hard events, no shift, computed on the fly in 32-window batches; the Global-TCN is unchanged.

## 6. Core architecture — `RhythmCrossFusionAdapter` (frozen design; `src/ppg2ecg/flow/rhythm_fusion.py`)

WHEN = the scaffold; WHAT = the existing PPG morphology pathway; the scaffold enters only at the target-side
hook (with the §0 caveat about the direct decoder route).

| block | definition | params | parameter names |
|---|---|---|---|
| 6.1 rhythm tokens | `Conv1d(1 → 32, kernel 7, stride 4, padding 3, bias=False)` on `s` → `[B, 32, 256]`; token j is centred on sample 4j (impulse test) | 224 | `fusion.tok.weight` |
| 6.2 target queries | `Q = Conv1d(128 → 32, kernel 1, bias=False)` on `H_z` → `[B, 32, 1024]` | 4,096 | `fusion.q.weight` |
| 6.3 temporal identity | **fixed** sinusoidal encoding, d = 32, base 10,000, on a **shared sample-index coordinate**: queries at pos = t ∈ {0…1023}, tokens at pos = 4j ∈ {0, 4, …, 1020}; added to Q and to the tokens (K = V = tokens + PE); non-persistent buffers (excluded from checkpoints and hashes; recomputed == stored asserted). The top frequency pair (wavelength 2π samples) is aliased at the 4-sample token pitch — a property of the frozen choice, not tuned | 0 | — |
| 6.4 cross-attention | one layer, embed 32, **4 heads** of dim 8, `softmax(QKᵀ/√8)V` over the 256 tokens, no mask; explicit q/k/v/o `Linear(32, 32)` with bias (parameter-identical to `nn.MultiheadAttention(32, 4)`); chosen because `torch.func.jvp` (the frozen objective's forward-mode path) is not supported through the fused SDPA kernels of `nn.MultiheadAttention` under torch 2.11 (verified), whereas the explicit form matches finite differences; no self-attention, no FFN, no stack; a static test forbids `scaled_dot_product_attention` / `nn.MultiheadAttention` in the module | 4,224 | `fusion.attn.{q,k,v,o}.{weight,bias}` |
| 6.5 output projection | `Conv1d(32 → 128, kernel 1)` with bias, **weight and bias zero-initialised**; residual `H_z' = H_z + out` | 4,224 | `fusion.out.{weight,bias}` |
| **TF total** | | **12,768** (0.279 % of 4,568,707; 0.297 % of the effective 4,304,513; ≤ 50,000; < 1.1 %) | 12 tensors |

At step 0 the output projection is zero, so `H_z' == H_z` exactly for any scaffold. No capacity is added
after validation.

## 7. Adaptive Rhythm Gate (GTF arms)

Inference-safe features from `s` only (channel 1 of the carrier; never the PPG, never ECG, GT R, site or
validation information): `f1 = s`; `f2 = avg_pool1d(s, 33, stride 1, padding 16, count_include_pad=False)`
(fixed centred 33-sample mean; truncated at the edges); `f3 = |s[t] − s[t−1]|` with `f3[0] = 0` →
`F_gate [B, 3, 1024]`. Gate network `gate.0 = Conv1d(3 → 16, k=1)` → SiLU → `gate.2 = Conv1d(16 → 1, k=1)` →
sigmoid; `gate.2` weight zero, bias `logit(0.90) = ln 9 ≈ 2.1972`, so `g(t) ≈ 0.90` uniformly at init.
**Gate params = 64 + 17 = 81 (< 128)**; names `gate.{0,2}.{weight,bias}`. Gated fusion
`H_z' = H_z + g(t) ⊙ out(t)`; at step 0 `out = 0` so parity holds. The gate is trained only through the frozen
iMeanFlow objective. **GTF total = 12,849 (0.281 %; 16 tensors).**

**CONST-GATE control (GTF-CONST):** identical architecture, initialisation, optimizer and parameter count
(12,849); each gate feature is replaced by its per-window temporal mean broadcast over all 1024 positions
(`f_const_k(t) = mean_t f_k(t)`), so the gate can learn a window-level fusion strength but not local temporal
suppression. The TRUE scaffold still feeds K/V.

## 8. Arms — nine evaluated, six trained

| arm | model | scaffold (tokens, gate) | trained |
|---|---|---|---|
| **B** | frozen generator (`MeanFlowS5`) | — | no |
| **ADD** | R2 `RhythmMeanFlowS5` + frozen R2 TRUE adapter (§2) | s_pred(own) | no (R2) |
| **TF-TRUE** | ungated target-side fusion | s_pred(own) | yes |
| **TF-SHUFFLE** | same TF architecture / init / streams | s_pred(partner) | yes |
| **GTF-TRUE** | fusion + adaptive gate | s_pred(own) | yes |
| **GTF-SHUFFLE** | same GTF | s_pred(partner) for tokens and gate | yes |
| **GTF-CONST** | same GTF, gate features per-window means | s_pred(own) tokens; broadcast gate features | yes |
| **GTF-ORACLE (GT-R leakage; diagnostic only)** | same GTF | s_oracle (σ = 100 ms GT-R field, the R1 label) for tokens and gate; at validation from the window's own GT R | yes |
| **ADD-ORACLE (GT-R leakage; §17.3 reference only)** | R2 additive interface + frozen R2 ORACLE adapter | s_oracle | no (R2) |

SHUFFLE reuses the **frozen R2 derangement** (salt `r2-rhythm-shuffle-v1`, populations train / eval / viz;
bijective, fixed-point-free within subject × site; re-asserted against the rule and copied to
`artifacts/r3_rhythm_fusion/shuffle_manifest.csv`). The ORACLE training field reuses the R2 cache (§2).

## 9. Initialisation and randomness

Seed 42. All trained arms share the generator and Global-TCN checkpoints, the A4 loader order (three-tensor
`TensorDataset`, generator seed 42), the (t, r) stream (`sample_tr_c1(arm="B")`, generator seed 43), the
CUDA source stream (seeded 42, consumed only by the `e` draws), the optimizer configuration and the step
count. **Module initialisation:** `torch.manual_seed(42)` is called immediately before constructing the
fusion module on CPU; this also re-seeds the CUDA global RNG, which is harmless only because **no CUDA draw
occurs before step 1** (asserted by the probe hash). Fusion tensors are constructed in the fixed order
tokens → Q → q/k/v/o → out-proj; the gate (whose 3 → 16 layer is seed-42 random) is constructed strictly
**after** the out-projection, so the fusion subset is bit-identical between TF and GTF families, and all
arms of a family are bit-identical. `initialization_hashes.json` stores the sha256 of `named_parameters()`
(fusion subset and full) per arm and asserts TF-TRUE == TF-SHUFFLE, GTF-TRUE == GTF-SHUFFLE == GTF-CONST ==
GTF-ORACLE, and fusion-subset(TF) == fusion-subset(GTF). **The paired-randomness probe hash over the first
four micro-batches of (idx, t, r, e) is asserted equal to the R3 preflight's and to R2's
`04aad6ae5ec41798…` in every trained-arm process (STOP otherwise).**

## 10. Step-0 parity — STOP conditions

- **TF / GTF:** with the zero output projection, `torch.equal(model output, frozen baseline)` on the **full
  2,048-window primary population at NFE 4** for **zero, real (TRUE) and shuffled** scaffolds; the attention
  output is asserted finite. Any failure → STOP.
- **ADD:** same-process `torch.equal` between the R3 evaluator's ADD generation and R2's `RhythmMeanFlowS5`
  generation path with the same adapter (arm asserted `"true"`, generator sha asserted) on the full NFE-4
  tensor (STOP otherwise). **Regression vs R2:** ADD's per-window rows (all numeric `_score_chunk` keys) vs
  `artifacts/r2_rhythm_transfer/metrics_by_window.csv` (arm TRUE, nfe 4) and its macro rows vs R2's
  `event_metrics.csv` / `structure_metrics.csv`; every |Δ| written to `provenance.json`; **STOP if the macro
  F1 excess, S4 or S5 differ by more than 1e-6**; other differences flagged. The same regression is run for
  B (vs R2's B rows) and for ADD-ORACLE (vs R2's ORACLE rows; flagged, not stopped). cuDNN flags recorded.

## 11. Training objective and budget

Exact frozen `imeanflow_loss(norm_p=1.0, norm_eps=0.01, jvp_mode="forward")`; no R / event / RR /
morphology / QRS-weighting / derivative / curvature / confidence loss. **Exactly 2,200 AdamW steps per arm**
(lr 1e-3, weight decay 0.01 — the R2 configuration), batch 64 = 2 × 32, checkpoints at 0 / 550 / 1100 / 2200,
**primary = step 2200**, no early stopping, no selection, no validation window read. Driver
`src/ppg2ecg/training/train_r3_fusion.py`, `--arm ∈ {preflight, tf_true, tf_shuffle, gtf_true, gtf_shuffle,
gtf_const, gtf_oracle}`, one process per arm, refuse-to-overwrite, trainable-name-set assertion per family,
frozen-gradient assertion after step 1, logging as R2 plus `fusion_l2`, `out_proj_l2`, `gate_mean`,
`gate_std` (arms compared on the unweighted residual); checkpoints hold `named_parameters()` of the R3
module only plus provenance (arm, family, gate mode, step, generator / Global-TCN sha, probe hash, git).

## 12. Runtime preflight

100 discarded steps of **GTF-TRUE** (the most expensive deployable arm); s/step = mean over steps 2–100,
peak allocated / reserved VRAM; projected total = **6 arms × 2,200 × s/step**. **If > 6 GPU-hours: STOP.**
Preflight state discarded; every arm re-initialises from the identical seed in its own process. Pre-freeze
measurement (forward + backward, no optimizer step): ≈ 0.19 s per micro-batch, 4.1 GiB → expected
≈ 0.4 s/step, ≈ 1.5 GPU-h.

## 13. Evaluation population, NFE, metrics

**Primary:** the frozen C0/C1/R2 2,048-window development subset (`select_subset("x4-event-nfe-v2", ·, 1024)`
per an0/k2s, asserted against `nfe_subset.json`), seed-0 source bank (sha256 `86808579…` asserted), 19,834 GT
beats (asserted), frozen detector and matcher. Scoring: `_score_chunk`, `_chance_chunk`, `_peaks`, `pmap`,
`score`, `macro_rows`, `paired` copied verbatim from `scripts/r2_evaluate.py` (C0 text identity test kept);
everything arm-specific (arm set, three model classes, generation, site-wise, decision) is new in
`scripts/r3_evaluate.py`. **NFE 4 primary; NFE 1 and 2 secondary, all nine arms scored** — not expanded
after results.

Event metrics: raw F1, chance floor, **F1 excess (primary)**, precision, recall, missing fraction, spurious
fraction, matched coverage, beats ratio, beats-ratio deviation (missing = 1 − recall by construction, so
U3 / G-family reliability items name their carrier). Structure: **S1** raw RMSE, **S2** raw correlation,
**S3** fixed-coordinate QRS RMSE (C0), **S4** QRS-core derivative RMSE, **S5** QRS-core curvature error
(M1), secondary S6 QRS-energy deviation, S7 p2p deviation, S8 HF fraction (frozen definitions, R2 §14).
**Critical R3 structure metrics: S4 and S5.** No oracle alignment, no shifting, no matched-only morphology as
primary.

## 14. Bootstrap

Paired by validation window, subject-stratified, equal an0/k2s weight, 2,000 replicates,
`default_rng(20260902)` passed explicitly; `paired_subject_bootstrap(earlier = second-named arm, later =
first-named arm, orient)` so that **positive = the first-named arm better** for every metric (F1 / corr:
A − B; errors / deviations: B − A). Contrasts (NFE 4, event family + S1–S5): TF-TRUE vs B, TF-TRUE vs
TF-SHUFFLE, TF-TRUE vs ADD, TF-SHUFFLE vs B, GTF-TRUE vs B, GTF-TRUE vs GTF-SHUFFLE, GTF-TRUE vs GTF-CONST,
GTF-TRUE vs TF-TRUE, GTF-TRUE vs ADD, GTF-CONST vs B, GTF-SHUFFLE vs B, ADD vs B (`paired_bootstrap.csv`);
at NFE 1 / 2, event family only: TF-TRUE vs B, TF-TRUE vs TF-SHUFFLE, GTF-TRUE vs B, GTF-TRUE vs GTF-SHUFFLE.
Oracle contrasts in `oracle_diagnostic.csv` (§17.3). **NaN rule:** S1–S5 are NaN only for windows without a
valid GT beat (GT-determined, 0 on this population); if NaN patterns ever differ between arms, pairwise-
incomplete windows are dropped, `n_eff` / `nan_pairs` recorded and flagged in `provenance.json` — not
aborted. **Multiplicity:** no correction; 12 × 11 + 4 × 4 + oracle + site + phase + gate CIs are computed
(count written to `provenance.json`); U3, U5, G4, G5 are disjunctive and the multiplicity-weakest items;
the report names the carrier metric and the status of the alternatives. All calls share one resample-index
set (rng re-seeded per call). CI scope as R2 §15 (window sampling within two subjects, one realised run).
**SHUFFLE − B reading rule (from R2 §16):** U2 / G2 are read as window-specific transfer only if the
SHUFFLE − B F1-excess verdict is not "worsens"; the share (SHUFFLE − B) / (TRUE − B) is reported with the
R2 disclosure sentence.

## 15. Ungated success gate (U) — frozen

| item | requirement (NFE 4) |
|---|---|
| U1 | TF-TRUE vs B F1 excess CI entirely > 0 **and** point ≥ +0.020 |
| U2 | TF-TRUE vs TF-SHUFFLE F1 excess CI entirely > 0 |
| U3 | ≥ 1 of {missing, spurious, beats-ratio deviation} improves vs B with CI entirely > 0 (carrier named) |
| U4 | neither S4 nor S5 vs B has its oriented CI entirely < 0 |
| U5 | vs ADD: ≥ 1 of {S4, S5} improves with CI entirely > 0 and the other's CI is not entirely < 0 |
| U6 | TF-TRUE macro beats-ratio deviation < 0.20 |

## 16. Gated success gate (G) — frozen; always computed, decides only if U fails

| item | requirement (NFE 4) |
|---|---|
| G1 | GTF-TRUE vs B F1 excess CI entirely > 0 **and** point ≥ +0.020 |
| G2 | GTF-TRUE vs GTF-SHUFFLE F1 excess CI entirely > 0 |
| G3 | neither S4 nor S5 vs B has its oriented CI entirely < 0 |
| G4 | vs ADD: both S4 and S5 have positive point improvement, ≥ 1 with CI entirely > 0, the other's CI not entirely < 0 |
| G5 | GTF-TRUE vs GTF-CONST F1 excess non-inferior — lower 95 % bound of the oriented effect > −0.005 — **and** ≥ 1 of {S4, S5} improves vs GTF-CONST with CI entirely > 0 |
| G6 | GTF-TRUE macro beats-ratio deviation < 0.20 |

Qualifier fixed now: if GTF-TRUE vs GTF-CONST F1 excess is "worsens" while non-inferiority holds, any A / B
verdict carries *"CONST gate superior on F1 within margin"* and the report prints the realised CI half-width
next to the −0.005 margin. Gates U and G are not modified after results.

## 17. Decision — a total function of `decision.json`

### 17.1 Definitions

- `ev(X)` := X vs B F1 excess verdict "improves" **and** point ≥ +0.020 **and** X vs its SHUFFLE control
  F1 excess "improves" (i.e. exactly U1 ∧ U2 for TF-TRUE, G1 ∧ G2 for GTF-TRUE — "clearly improves" carries
  the +0.02 magnitude as in R2). `ev_ci_only(X)` (magnitude dropped) is recorded but never decides.
- `deg(X)` := (S4 or S5 vs B "worsens") **or** (the vs-ADD protection item for X fails: U5 for TF-TRUE, G4
  for GTF-TRUE) — i.e. "relative to B / R2 protection gates".

### 17.2 Verdict (first matching row)

| order | verdict | condition |
|---|---|---|
| 1 | **A. UNGATED TARGET-SIDE FUSION SUFFICIENT** | U1–U6 all pass. Tie-break stated in the report: GTF "clearly adds protection" iff GTF-TRUE vs TF-TRUE has (S4 improves ∧ S5 not worsens) ∨ (S5 improves ∧ S4 not worsens) and none of {F1 excess, missing, spurious, beats-ratio deviation} worsens; otherwise the simpler TF is preferred |
| 2 | **B. ADAPTIVE GATING REQUIRED AND SUPPORTED** | ¬A ∧ G1–G6 all pass. Necessity reading, fixed now: printed as *B (gating necessity: SEPARATED)* iff U4 failed ∨ (GTF-TRUE vs TF-TRUE S4 or S5 "improves" with the other not "worsens"); otherwise *B (gating supported; necessity not separated from ungated TF — TF failed U on {items})* |
| 3 | **C. EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS** | ¬A ∧ ¬B ∧ (ev(TF-TRUE) ∨ ev(GTF-TRUE)) ∧ deg(X) for every X with ev(X). Per-arm narrative recorded (TF couples / GTF couples / other arm without ev) |
| 4 | **D. TARGET-SIDE RHYTHM FUSION NOT SUPPORTED** | otherwise, with a **closed residual code list**: `D_NO_EVENT_GAIN` (¬ev(TF) ∧ ¬ev(GTF) — the only case that prints the spec's "neither … convincing" sentence), `D_SUBTHRESHOLD` (an arm has ev_ci_only but point < +0.020), `D_TF_U3_FAIL`, `D_TF_U6_CATASTROPHE`, `D_GTF_G5_NONINFERIORITY_FAIL`, `D_GTF_G5_STRUCTURE_VS_CONST_FAIL`, `D_GTF_G6_CATASTROPHE` (an ev-arm without deg whose remaining U / G items failed), printed verbatim as *D (formal residual: codes)* |

The decision function is implemented once (`rhythm_fusion.decide_verdict_r3`) and tested for totality and
for the truth of each printed narrative on an exhaustive grid of decision records. Verdict D never
escalates to a larger Transformer.

### 17.3 Oracle diagnostic (`oracle_diagnostic.csv`; GT-R leakage; diagnostic only)

Contrasts GTF-ORACLE vs B, GTF-ORACLE vs GTF-TRUE, GTF-ORACLE vs ADD-ORACLE (the R2 ORACLE adapter through
the additive interface, regenerated here and regression-checked against R2's ORACLE rows), ADD-ORACLE vs B,
on the event family and S1–S5. Ordered, total reading on the F1-excess verdict of GTF-ORACLE vs ADD-ORACLE
and the S4 / S5 verdicts of GTF-ORACLE vs B: (1) **LOWERED** if "worsens"; (2) **LIFTED** if "improves" with
point ≥ +0.010 (pre-specified "materially": ¼ of R2's +0.0425 ORACLE gain) and neither S4 nor S5 "worsens";
(3) **LIFTED_BUT_COUPLES** if "improves" with point ≥ +0.010 and S4 or S5 "worsens"; (4) **UNCHANGED** if
"unresolved" or "improves" with point < +0.010; (5) **OTHER** with the triple printed. Never in a method
table without the leakage label.

## 18. Secondary inference-time diagnostics (no retraining, not gates)

**18.1 Phase ablation.** TF-TRUE and GTF-TRUE with the scaffold rolled by **+256 samples = +2.0 s**
(`torch.roll`). Report F1 excess, missing, spurious, beats deviation, S4, S5, and the TRUE-vs-shifted paired
effect overall and by the R2 φ strata (in-phase / anti-phase / rest).

**18.2 Direct-route attribution (§0).** For trained TF-TRUE and GTF-TRUE at NFE 4, an evaluation-only mode
of the src/ subclass subtracts the fusion output from the summed block outputs (`all_dx −= n_blocks · (g ⊙)
out`), so the fusion acts only through the S5 / MLP routes. Report F1 excess, S4, S5 with and without the
direct route and the paired difference; reading fixed now: if ≥ 50 % of the F1-excess gain vs B survives
without the direct route, the report may describe the fusion as acting "through the target stream";
otherwise it must describe it as "predominantly a direct decoder-input write".

## 19. Site-wise (secondary, exploratory)

The frozen R1 8,192-window validation cohort (the spec's "V1 8192", as in R2 §11), seed-0 bank (sha256
recorded), NFE 4, arms **B, ADD, TF-TRUE, GTF-TRUE, GTF-CONST**; per site F1 excess, missing, spurious, beats
deviation, QRS RMSE, derivative RMSE, curvature; paired per-site CIs (uncorrected, exploratory) for
**GTF-TRUE − B ankle F1 excess** and **GTF-TRUE − TF-TRUE ankle F1 excess** explicitly, plus TF-TRUE − B and
ADD − B per site; budget guard: projected = 4 × Σ (NFE-4 generation + scoring time of the five arms on the
primary population) against 7,200 s, `skipped` marker otherwise; isolation from the primary provenance as
R2. Does not decide the verdict.

## 20. Gate interpretability diagnostic (post-hoc, analysis only)

GTF-TRUE on the primary population; GT labels used only for analysis, never for training or calibration.
Statistics, all with equal an0/k2s weight and subject-stratified window bootstrap (2,000, seed 20260902,
percentile 2.5 / 97.5):
(A) `ρ_A` = Spearman (`scipy.stats.spearmanr`, average ranks) within each subject between the window-mean
gate and the scaffold's own F1@50 (frozen R1 rule: `extract_events` 0.35 / refractory 32, 50 ms one-to-one;
windows with no scaffold event have F1 = 0 and are kept), averaged over subjects;
(B) the same vs scaffold F1@150;
(C) mean of g over the ±4-sample neighbourhood (edge-clipped) of every extracted scaffold peak matched to
GT R at 50 ms **minus** the same over unmatched peaks, peaks pooled within subject then equal-subject
averaged; `n_matched` / `n_unmatched` reported;
(D) gate mean / p10 / p90 by site on the R1 8,192 cohort;
(E) gate statistics for windows with good (scaffold F1@50 ≥ population median) vs poor extraction.
`gate_diagnostics.csv`. **Wording rule fixed now:** *RELIABILITY-LIKE BEHAVIOR OBSERVED* iff the ρ_A point
estimate ≥ 0.20 with its CI lower bound > 0 **and** (C) has CI lower bound > 0 **and** the gate is not
constant (std of g over all samples ≥ 0.01); otherwise *GATE NOT INTERPRETABLE AS CONFIDENCE* and the report
calls it only an adaptive modulation gate.

## 21. Early-NFE persistence diagnostic (secondary)

B, ADD, TF-TRUE, GTF-TRUE at NFE 1, 2, 4 with the R2 ±250 ms greedy one-to-one matcher; matched fraction,
mean |δ|, median |δ|, sign consistency (δ1 vs δ4), fraction NFE 4 strictly closer than NFE 1, ties;
intersection-set comparison as R2. No solver claim.

## 22. Visual atlas

The V1 validation VIZ cohort (64 windows, V1 32-row batch construction, seed-0 bank), rows: PPG, R1 scaffold
(value at each GT R), GT ECG, B, ADD, TF-TRUE, GTF-TRUE, GTF-CONST, GTF-ORACLE at NFE 4 (missing × and
spurious ▲ annotated; g(t) overlaid on the GTF rows), plus the GT-R-centred −300 … +500 ms zoom (V1 rule).
**GT R is drawn as a vertical reference only; no prediction is shifted.** The report's atlas section
answers the six spec questions (missing beats recovered? new spurious beats? QRS slope / shape smoother or
distorted? R2 derivative / curvature degradation visible? GTF suppresses injection in weak PPG segments?
ankle still pathological?) as fixed sub-headings with **counts over all 64 windows only**; no per-window
prose selection.

## 23. Tests — mandatory before the implementation commit (`tests/test_r3_rhythm_fusion.py`)

Firewall; generator and Global-TCN frozen (no grad); hook is `z_e` (the subclass adds to `z_e`; `pre_conv_ppg`
receives `ppg2[:, :1]` only — forward-hook test; `pre_conv_ppg` never referenced by the fusion module); only
the R3 module trainable (pre-declared name sets, 12 TF / 16 GTF tensors); TF params == 12,768 (≤ 50,000,
< 1.1 %); gate extra params == 81 (< 128); GTF-TRUE == GTF-CONST == GTF-SHUFFLE == GTF-ORACLE parameter
count (12,849); identical initialisation hashes within families and on the shared fusion subset across
families under the fixed construction order; final projection zero-initialised; gate final weight zero /
bias ln 9; step-0 exact parity (tiny backbone unconditionally; real checkpoint when present) for zero /
real / shuffled scaffolds; the carrier passes the frozen loss (forward-mode JVP through the explicit
attention) with finite gradients on the R3 module only; a static test that the module does not use
`scaled_dot_product_attention` / `nn.MultiheadAttention`; same source / (t, r) / loader streams (probe-hash
determinism); shuffle bijection / no fixed points (manifest); ORACLE reads GT R only in its arm (driver
statics); deployable arms never access ECG at inference (scaffold signature; gate features from channel 1
only — static and signature); same validation windows; no validation selection; fixed 2,200 steps;
positional encoding deterministic, non-persistent, query / token positions on the same sample axis
(token j ↔ sample 4j impulse test, monotone); gate features contain no GT information; edge conventions
(`f2` truncated mean, `f3[0] = 0`); CONST features are per-window constants; +2 s shift = exactly 256
samples; the direct-route cancellation mode is exact at zero output (identity); `decide_verdict_r3` total on
an exhaustive grid with every printed narrative true. Full suite.

## 24. Artifacts

`artifacts/r3_rhythm_fusion/`: `provenance.json`, `hook_audit.json`, `frozen_checkpoint_manifest.json`,
`parameter_manifest.json`, `initialization_hashes.json`, `shuffle_manifest.csv`, `runtime_preflight.json`,
`training_log_{tf_true,tf_shuffle,gtf_true,gtf_shuffle,gtf_const,gtf_oracle}.csv` (+ per-arm
`train_provenance_*.json`), `metrics_by_window.csv`, `event_metrics.csv`, `structure_metrics.csv`,
`paired_bootstrap.csv`, `oracle_diagnostic.csv`, `phase_ablation.csv` (+ summary),
`direct_route_attribution.csv`, `site_metrics.csv`, `gate_diagnostics.csv`, `nfe_event_persistence.csv`
(+ summary), `decision.json`, `visual_atlas/`. Module parameters only in
`outputs/r3_{tf_true,tf_shuffle,gtf_true,gtf_shuffle,gtf_const,gtf_oracle}_seed42/`; no duplicated
generator / Global-TCN / R2 checkpoints or caches; nothing under `outputs/`, `artifacts/`, `data/processed/`
enters git. Report `docs/R3_DISENTANGLED_RHYTHM_FUSION_REPORT.md` and the final response follow the spec's
§37 headings verbatim.

## 25. Claim boundaries

If gate U passes (verdict A): *"Separating PPG-derived rhythm conditioning from the morphology stem through
target-side temporal fusion improves event correspondence while avoiding the derivative / curvature
degradation observed under naive additive conditioning."* If verdict B with *gating necessity: SEPARATED*:
*"Adaptive temporal gating further protects structural fidelity under this development protocol."*; if B
without separation, only *"adaptive gating did not degrade structural fidelity"*. "Confidence-calibrated" is
never used; at most "reliability-like behaviour observed" (§20). Any "through the target stream" wording is
conditioned on §18.2. Never: solved PPG-to-ECG, exact R recovered from PPG, causal mechanism proven, SOTA,
novel, test-generalised. The Global-TCN's target-derived supervision is carried in every claim. Single
seed; development only; no multiplicity correction.

## 26. Commit order

1 integrity audit · 2 hook audit · 3 preregistration · 4 commit + push (this commit; SHA recorded in
`provenance.json`) · 5 implementation · 6 tests · 7 commit + push · 8 100-step GTF-TRUE preflight ·
9 discard · 10 re-initialise · 11 train six arms · 12 primary NFE 4 evaluation · 13 phase ablation ·
14 site evaluation · 15 gate diagnostic · 16 early-NFE diagnostic · 17 visual atlas · 18 report · 19 result
commit + push · 20 STOP.
