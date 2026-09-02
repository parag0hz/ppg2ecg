# R3 — Disentangled Target-Side Rhythm Fusion + Adaptive Rhythm Gate — REPORT

**Preregistration:** `docs/R3_DISENTANGLED_RHYTHM_FUSION_PREREGISTRATION.md` (frozen at `3d779fc`, untouched:
`git diff 3d779fc -- docs/R3_*` is empty). **Hook audit:** `docs/R3_TARGET_STREAM_HOOK_AUDIT.md` (same commit).
**Implementation:** `7f5dc7e`. **Type:** development-only probe of a small target-side fusion module on a
frozen generator. No full generator retraining, no test subjects, no flow-objective change, no new R/event
auxiliary loss, single seed, no large Transformer, no novelty claim, no SOTA claim. R3 was designed after
observing R1 and R2 and is not independent confirmatory evidence (prereg §0). The preregistration's title names
the gate "Adaptive Reliability Gating"; under the §20 outcome reported below the gate is read as an *adaptive
modulation gate*, and this report does not use "reliability" as its name.

## FINAL R3 VERDICT

**EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS** (verdict C of the frozen decision tree, prereg §17.2;
`decision.json`, produced by `rhythm_fusion.decide_verdict_r3`).

Per-arm narrative fixed by §17.1: `ev(TF-TRUE)` is false (U1 ∧ U2 fail); `ev(GTF-TRUE)` is true (G1 ∧ G2 pass)
and `deg(GTF-TRUE)` is true (S4 and S5 vs B worsen; the vs-ADD protection item G4 fails). Every arm with an
event gain couples it to a structure trade-off, so row 3 matches. Residual codes: none (row 4 not reached).
Gating-necessity reading: not applicable (row 2 not reached); for the record, GTF-TRUE vs TF-TRUE worsens both
S4 (−0.0041) and S5 (−0.0039), so the gate does **not** protect structure (`gtf_vs_tf_protects = false`).

### Ungated gate U (TF-TRUE; NFE 4, frozen 2,048 windows, paired subject-stratified bootstrap)

Requirement column: prereg §15 / §16 wording; "improves" / "worsens" denote oriented CIs entirely above / below zero.

| item | requirement | observed | pass |
|---|---|---|---|
| U1 | vs B F1 excess CI entirely > 0 **and** point ≥ +0.020 | +0.0165 [+0.0133, +0.0196] | **no** (CI > 0, magnitude short by 0.0035) |
| U2 | vs TF-SHUFFLE F1 excess CI entirely > 0 | +0.0006 [−0.0010, +0.0024] unresolved | **no** |
| U3 | ≥ 1 of {missing, spurious, beats-ratio deviation} improves vs B with CI entirely > 0 (carrier named) | missing +0.0200 [+0.0169, +0.0230]; beats-dev +0.0131 [+0.0100, +0.0160]; spurious unresolved | yes (carriers: missing, beats-ratio deviation) |
| U4 | neither S4 nor S5 vs B has its oriented CI entirely < 0 | S4 −0.0028 [−0.0032, −0.0025] worsens; S5 +0.0024 improves | **no** |
| U5 | vs ADD: ≥ 1 of {S4, S5} improves with CI entirely > 0 and the other's CI is not entirely < 0 | S5 +0.0034 improves, S4 −0.0015 [−0.0019, −0.0011] worsens | **no** |
| U6 | macro beats-ratio deviation < 0.20 | 0.0936 | yes |

### Gated gate G (GTF-TRUE)

| item | requirement | observed | pass |
|---|---|---|---|
| G1 | vs B F1 excess CI entirely > 0 **and** point ≥ +0.020 | +0.0406 [+0.0353, +0.0458] | yes |
| G2 | vs GTF-SHUFFLE F1 excess CI entirely > 0 | +0.0242 [+0.0192, +0.0292] | yes |
| G3 | neither S4 nor S5 vs B has its oriented CI entirely < 0 | S4 −0.0069 [−0.0076, −0.0063]; S5 −0.0015 [−0.0021, −0.0010]; both worsen | **no** |
| G4 | vs ADD: both S4 and S5 have positive point improvement, ≥ 1 with CI entirely > 0, the other's CI not entirely < 0 | S4 −0.0056 [−0.0063, −0.0050] worsens; S5 −0.0005 [−0.0011, +0.0000] unresolved | **no** |
| G5 | vs GTF-CONST F1 excess non-inferior (lower 95 % bound of the oriented effect > −0.005) **and** ≥ 1 of {S4, S5} improves vs GTF-CONST with CI entirely > 0 | non-inferior: yes (+0.0233 [+0.0184, +0.0282]; realised CI half-width 0.0049 vs margin 0.005); S4 −0.0040, S5 −0.0037 both worsen | **no** (structure part) |
| G6 | macro beats-ratio deviation < 0.20 | 0.0912 | yes |

Decision path: not A (U1, U2, U4, U5 fail); not B (G3, G4, G5 fail); C because GTF-TRUE has `ev` with `deg`
and TF-TRUE has no `ev`. The "CONST gate superior on F1 within margin" qualifier does not apply (GTF-TRUE vs
GTF-CONST F1 excess is "improves"). Oracle reading (§17.3): **LIFTED** (below).

Reading under verdict C (no §25 claim sentence is licensed): the target-side cross-attention interface with the
adaptive gate moves event correspondence more than the additive interface did (+0.0406 vs R2's +0.0194 on the
same frozen windows; GTF-TRUE vs ADD +0.0213 [+0.0161, +0.0263]), and the ankle regression of R2 is reversed
(site-wise, exploratory, §19), but the S4 trade-off of R2 is **not** removed — it is larger (−0.0056 vs ADD); the
S5 trade-off vs ADD is unresolved (−0.0005 [−0.0011, +0.0000]). On F1 excess and missing the ungated module is
indistinguishable from its SHUFFLE control (TF-TRUE ≈ TF-SHUFFLE; small CI-resolved differences remain on spurious
and S1–S5), so the R3 claim sentences for A and B are not licensed. Nothing here separates extractor from
interface causally.

## Repository

| item | value |
|---|---|
| preregistration commit | `3d779fc` (hook audit + prereg; frozen) |
| implementation commit | `7f5dc7e` (`src/ppg2ecg/flow/rhythm_fusion.py`, `src/ppg2ecg/training/train_r3_fusion.py`, `scripts/r3_prepare.py`, `scripts/r3_evaluate.py`, `scripts/r3_visual_atlas.py`, `tests/test_r3_rhythm_fusion.py`) |
| result commit | this commit: report, evaluator fixes (two; §Deviations) and their static tests |
| submodules | `external/PENGUIN` @ `6cd70cd`, `external/iMeanFlow` @ `bf60cd7`, untouched |
| A4 checkpoint md5 | `31c042d291052fbb6dc15263ad316be2`, unchanged |
| test subjects | `kjd` / `ssx` never loaded (`assert_no_test_subjects` in every script; `test_subjects_loaded: []`) |
| environment | RTX 5090, torch 2.11.0+cu130, numpy 2.3.5, neurokit2 0.2.12, Python 3.13.9; training (preflight + six arms) with `cudnn.deterministic = true` (`train_provenance_*.json`, `runtime_preflight.json`); evaluator and atlas at framework defaults (`cudnn_deterministic: false`, as R2's evaluator; not recorded for C0 / C1 / V1) |
| git state at evaluation | `7f5dc7e` + 2 dirty files (the variant-source fix and its test, both in this commit) |

## Hook audit (`docs/R3_TARGET_STREAM_HOOK_AUDIT.md`, `hook_audit.json`)

- Hook: `z_e = backbone.pre_conv_target(z)` in `MeanFlowS5.u` (`src/ppg2ecg/flow/imeanflow.py` L49), shape
  [B, 128, 1024], one-to-one with the waveform timeline, produced without stride or pooling. All five
  acceptance criteria hold (downstream of the noisy-waveform projection, not the PPG stem, waveform-aligned,
  before final decoding, residual path without touching the PPG encoder). The PPG stem (`pre_conv_ppg`,
  528,640 params) is untouched.
- Every block receives the same `z_e`; the target stream is not chained; block outputs are summed and decoded.
- **Not a pure "WHEN" port:** `z_e` reaches the decoder through three routes, one of which is the raw per-block
  residual (`dx_t = z_e + …`, ×4) that bypasses S5 / MLP; measured first-order direct-route share 0.21–0.70 on
  the frozen weights. This is why the preregistered direct-route attribution (§18.2) was run (below).
- Implemented as a `src/`-only subclass (`FusionMeanFlowS5`) inserting `z_e' = z_e + (g ⊙) out` between L49
  and L56; `external/` untouched.

## Frozen components (`frozen_checkpoint_manifest.json`; all hashes asserted before any step)

| component | path | file sha256 | state-dict sha256 | note |
|---|---|---|---|---|
| generator | `outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt` | `557c70541f5cdd07…` | `47d7ccb94e5dbf71…` | round 45, 4,568,707 state-dict params (4,304,513 effective), = A4 weights |
| Global-TCN | `outputs/r1_global_tcn_seed42/checkpoint_best.pt` | `bfe76ea6fd2842dc…` | `0986a7af1db29133…` | 328,897 params, R1 |
| ADD adapter (R2 TRUE) | `outputs/r2_true_adapter_seed42/adapter_step2200.pt` | `2d577897ad68917e…` | `f98057ca981bb840…` | L2 7.6975, 128 params |
| ADD-ORACLE adapter (R2) | `outputs/r2_oracle_adapter_seed42/adapter_step2200.pt` | `2802292b2dade392…` | `c8827b1b0a6d065f…` | GT-R leakage; diagnostic only |
| R2 oracle field cache | `artifacts/r2_rhythm_transfer/_cache_oracle_train.npz` | — | sha `2e6c548c10ba4cb1…` | re-hashed, verified |
| R2 shuffle manifest | copied to `artifacts/r3_rhythm_fusion/shuffle_manifest.csv` | — | — | derangement re-asserted (train 293,271 / eval 2,048 / viz 64) |

Streams: `seed_everything(42)`; loader generator 42, `(t, r)` generator 43, CUDA global RNG for `e`; the
probe hash over the first four micro-batches is `04aad6ae5ec41798…` in the preflight and in all six arms and
equals R2's (`probe_hash_matches_r2: true` everywhere). Step-0 parity (prereg §10): all eleven checks true
(`tf_zero`, `tf_true`, `tf_shuffled`, `tf_attn_finite`, `tf_step0_ckpt_equals_fresh`, the five `gtf_*`
counterparts, `add_same_process_torch_equal`); no STOP condition fired. Regression of the regenerated B, ADD
and ADD-ORACLE rows against R2: 0 window-level or macro flags > 1e-6.

## Parameters (`parameter_manifest.json`, `initialization_hashes.json`)

| module | params | trainable | composition |
|---|---|---|---|
| generator | 4,568,707 | no | — |
| Global-TCN | 328,897 | no | — |
| ADD (R2) | 128 | no (frozen R2 weights) | additive 1×1 at `pre_conv_ppg` output |
| TF | 12,768 (0.279 % of the generator) | yes | tokens Conv1d(1→32, k7, s4) 224 · Q Conv1d(128→32, k1) 4,096 · 4-head attention q/k/v/o 4,224 · out Conv1d(32→128, k1, zero-init) 4,224; fixed sinusoidal PE, no bias on tokens / Q |
| GTF | 12,849 = TF + 81 | yes | gate Conv1d(3→16, k1) → SiLU → Conv1d(16→1, k1), final weight 0, bias ln 9 (g₀ = 0.9); features s, 33-sample mean, |Δs| |

All four GTF arms share one initialisation hash; TF and GTF share the fusion-subset hash; construction order
fixed (fusion then gate). `assert_only_r3_trainable` / `assert_frozen_have_no_grad` hold at every step.

## Runtime

| stage | wall | s/step | peak alloc MiB | note |
|---|---|---|---|---|
| preflight (GTF-TRUE, 100 steps) | 39.7 s | 0.391 (step 1: 0.839) | 6,492 | projected 1.44 GPU-h for six arms; `stop: false`; weights discarded (§26 step 9) |
| tf_true | 887.9 s | 0.403 | 6,455 | |
| tf_shuffle | 875.5 s | 0.397 | 6,457 | |
| gtf_true | 891.9 s | 0.405 | 6,492 | |
| gtf_shuffle | 874.9 s | 0.397 | 6,494 | |
| gtf_const | 872.4 s | 0.396 | 6,491 | |
| gtf_oracle (GT-R leakage; diagnostic only) | 865.8 s | 0.393 | 7,638 | oracle field cache resident on GPU |
| six arms total | 5,268 s = 1.46 GPU-h (5,308 s incl. preflight) | | | |
| evaluation (`scripts/r3_evaluate.py`) | 860.6 s | | | 416 CIs in total (`n_ci_computed`); of which the site-wise stage: 339.6 s (projected 336.5 s, budget 7,200 s), 140 CIs |
| visual atlas | < 1 min | | | `atlas_summary.json` written within a minute of `provenance.json` |

Each arm: exactly 2,200 optimiser steps, AdamW 1e-3 / 0.01, batch 64 as 2 × 32, checkpoints at 0 / 550 / 1,100 /
2,200 (`outputs/r3_{arm}_seed42/module_step*.pt`, never in git). Training trajectories
(`training_log_{arm}.csv`; the objective column `loss_weighted` is normalised by the adaptive weighting and
stays ≈ 0.9999, so the raw `mse` column is shown):

| arm | mse mean steps 1–100 | mse mean steps 2,101–2,200 | ‖fusion‖ at 550 / 1,100 / 2,200 | ‖out‖ final | gate mean (std) final |
|---|---|---|---|---|---|
| tf_true | 0.1205 | 0.1224 | 9.69 / 10.33 / 11.29 | 4.73 | — |
| tf_shuffle | 0.1205 | 0.1229 | 9.62 / 10.27 / 11.21 | 4.67 | — |
| gtf_true | 0.1205 | 0.1204 | 9.97 / 10.92 / 12.20 | 5.00 | 0.773 (0.310) |
| gtf_shuffle | 0.1205 | 0.1230 | 9.69 / 10.35 / 11.29 | 4.75 | 0.917 (0.004) |
| gtf_const | 0.1205 | 0.1224 | 9.75 / 10.44 / 11.43 | 4.84 | 0.922 (0.000) |
| gtf_oracle (GT-R leakage; diagnostic only) | 0.1205 | 0.0946 | 11.24 / 12.23 / 13.68 | 5.94 | 0.888 (0.275) |

The batch / `(t, r)` / `e` sequence is identical across arms, so the columns are comparable step for step.
Only GTF-TRUE and GTF-ORACLE (GT-R leakage; diagnostic only) lower the raw mse below the ungated arms. The gate
becomes non-constant within a window only in those two arms (std 0.310 / 0.275); GTF-SHUFFLE's mean drifts to
0.917 with std 0.004, and GTF-CONST (0.922) cannot vary within a window by construction, its gate features being
per-window means.

## NFE 4 — Event correspondence (frozen 2,048 windows, 19,834 GT beats; `event_metrics.csv`)

| arm | F1 | chance floor | **F1 excess** | precision | recall | missing | spurious | beats ratio | beats-ratio dev | timing median ms |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.4367 | 0.1192 | 0.3176 | 0.4435 | 0.4338 | 0.5662 | 0.5154 | 0.9492 | 0.1067 | 23.4 |
| ADD (R2 TRUE, frozen) | 0.4562 | 0.1193 | 0.3369 | 0.4628 | 0.4534 | 0.5466 | 0.4995 | 0.9530 | 0.1047 | 23.4 |
| TF-TRUE | 0.4552 | 0.1211 | 0.3341 | 0.4597 | 0.4539 | 0.5461 | 0.5154 | 0.9693 | 0.0936 | 23.4 |
| TF-SHUFFLE | 0.4540 | 0.1206 | 0.3334 | 0.4581 | 0.4530 | 0.5470 | 0.5181 | 0.9711 | 0.0922 | 23.4 |
| GTF-TRUE | **0.4786** | 0.1204 | **0.3582** | 0.4844 | 0.4767 | **0.5233** | **0.4902** | 0.9669 | **0.0912** | 23.4 |
| GTF-SHUFFLE | 0.4548 | 0.1208 | 0.3340 | 0.4590 | 0.4538 | 0.5462 | 0.5171 | 0.9708 | 0.0926 | 23.4 |
| GTF-CONST | 0.4557 | 0.1207 | 0.3349 | 0.4605 | 0.4542 | 0.5458 | 0.5143 | 0.9685 | 0.0954 | 23.4 |
| GTF-ORACLE (GT-R leakage; diagnostic only) | 0.9384 | 0.1220 | 0.8164 | 0.9544 | 0.9295 | 0.0705 | 0.0380 | 0.9674 | 0.0490 | 15.6 |
| ADD-ORACLE (GT-R leakage; diagnostic only) | 0.4798 | 0.1197 | 0.3601 | 0.4870 | 0.4767 | 0.5233 | 0.4770 | 0.9536 | 0.1006 | 23.4 |
| PHASE-TF (TF-TRUE, scaffold +2.0 s) | 0.4563 | 0.1213 | 0.3350 | 0.4609 | 0.4549 | 0.5451 | 0.5145 | 0.9694 | 0.0941 | 23.4 |
| PHASE-GTF (GTF-TRUE, scaffold +2.0 s) | 0.4296 | 0.1185 | 0.3112 | 0.4383 | 0.4253 | 0.5747 | 0.5207 | 0.9460 | 0.1136 | 23.4 |
| NODIRECT-TF | 0.4517 | 0.1206 | 0.3311 | 0.4559 | 0.4506 | 0.5494 | 0.5176 | 0.9682 | 0.0980 | 23.4 |
| NODIRECT-GTF | 0.4761 | 0.1207 | 0.3554 | 0.4814 | 0.4747 | 0.5253 | 0.4929 | 0.9676 | 0.0934 | 23.4 |

Missing = 1 − recall by construction. The four arms whose gate is absent or ends up ≈ constant (TF-TRUE,
TF-SHUFFLE, GTF-SHUFFLE with gate std 0.004, GTF-CONST by construction) sit within 0.0015 of each other on F1
excess (0.3334–0.3349): the fusion path improves events by ≈ +0.016–0.017 over B whether its tokens see the
window's own scaffold (TF-TRUE, GTF-CONST — in GTF-CONST the own scaffold still feeds K/V and only the gate
features are per-window constants) or a deranged partner's (TF-SHUFFLE, GTF-SHUFFLE). Only GTF-TRUE (and the
leakage arms) separates from that cluster: separation requires both a locally varying gate and the window's own
scaffold (GTF-TRUE vs GTF-CONST +0.0233; GTF-TRUE vs GTF-SHUFFLE +0.0242); neither the gate with a partner
scaffold (GTF-SHUFFLE 0.3340) nor own-scaffold access without a varying gate (TF-TRUE 0.3341, GTF-CONST 0.3349)
separates. Which channel carries the difference is not identified by the frozen arms.

## Structural fidelity (NFE 4; `structure_metrics.csv`; lower is better except S2)

| arm | S1 raw RMSE | S2 raw corr | S3 QRS RMSE | **S4 QRS-core derivative RMSE** | **S5 QRS-core curvature error** | S6 QRS-energy dev | S7 p2p dev | S8 HF fraction |
|---|---|---|---|---|---|---|---|---|
| B | 0.4233 | 0.1040 | 0.5462 | 0.3220 | 0.2147 | 0.6056 | 0.2425 | 0.2305 |
| ADD | 0.4214 | 0.1129 | 0.5449 | 0.3233 | 0.2157 | 0.5925 | 0.2365 | 0.2322 |
| TF-TRUE | 0.4151 | 0.1096 | 0.5394 | 0.3248 | 0.2123 | 0.5851 | 0.2167 | 0.2497 |
| TF-SHUFFLE | 0.4166 | 0.1087 | 0.5399 | 0.3242 | 0.2121 | 0.5891 | 0.2199 | 0.2500 |
| GTF-TRUE | 0.4231 | 0.1126 | 0.5506 | **0.3289** | **0.2162** | 0.5531 | 0.2062 | 0.2390 |
| GTF-SHUFFLE | 0.4166 | 0.1090 | 0.5400 | 0.3243 | 0.2120 | 0.5894 | 0.2201 | 0.2505 |
| GTF-CONST | 0.4153 | 0.1096 | 0.5395 | 0.3249 | 0.2125 | 0.5842 | 0.2157 | 0.2491 |
| GTF-ORACLE (GT-R leakage; diagnostic only) | 0.3648 | 0.4372 | 0.4654 | 0.3131 | 0.2151 | 0.3958 | 0.1699 | 0.2544 |
| ADD-ORACLE (GT-R leakage; diagnostic only) | 0.4199 | 0.1198 | 0.5449 | 0.3253 | 0.2172 | 0.5732 | 0.2287 | 0.2331 |
| PHASE-GTF | 0.4209 | 0.0966 | 0.5375 | 0.3193 | 0.2124 | 0.6352 | 0.2498 | 0.2404 |
| NODIRECT-TF | 0.4217 | 0.1080 | 0.5443 | 0.3245 | 0.2126 | 0.5874 | 0.2171 | 0.2431 |
| NODIRECT-GTF | 0.4270 | 0.1128 | 0.5530 | 0.3290 | 0.2168 | 0.5537 | 0.2042 | 0.2338 |

The four arms whose gate is absent or ≈ constant improve S1, S3 and S5 vs B and worsen S4 by −0.002 to −0.003;
GTF-TRUE loses the S1 gain (unresolved vs B), is worse than B on S3 (−0.0044) and S5 (−0.0015), and worsens S4
the most among the nine arms. Rolling the scaffold by 2 s (PHASE-GTF — the preregistered §18.1 evaluation-time
variant, not one of the nine §8 arms; its S4 vs B is a point estimate without a paired CI) removes the whole
event gain (F1 excess 0.3112 < B 0.3176) and, at the point estimate, the entire S4 / S5 cost vs B (PHASE-GTF S4
0.3193 / S5 0.2124 vs B 0.3220 / 0.2147); the true-vs-shifted contrasts are S4 −0.0096, S5 −0.0038. The S4 cost
of GTF-TRUE is therefore tied to the true-timed scaffold; the roll contrast does not decompose that cost
additively (the shuffle contrast below does, under an explicit additivity assumption).

## Paired contrasts (NFE 4; subject-stratified paired bootstrap, 2,000 replicates, seed 20260902; positive = first-named better; `paired_bootstrap.csv`)

### TF-TRUE vs B

| metric | effect [95 % CI] | verdict |
|---|---|---|
| F1 excess | +0.0165 [+0.0133, +0.0196] | improves (below the +0.020 magnitude of U1) |
| missing | +0.0200 [+0.0169, +0.0230] | improves |
| spurious | −0.0000 [−0.0037, +0.0038] | unresolved |
| beats-ratio dev | +0.0131 [+0.0100, +0.0160] | improves |
| S1 raw RMSE | +0.0082 [+0.0073, +0.0090] | improves |
| S2 raw corr | +0.0056 [+0.0044, +0.0068] | improves |
| S3 QRS RMSE | +0.0068 [+0.0059, +0.0076] | improves |
| **S4** | −0.0028 [−0.0032, −0.0025] | **worsens** |
| **S5** | +0.0024 [+0.0020, +0.0027] | improves |

### TF-TRUE vs TF-SHUFFLE (specificity), and TF-SHUFFLE vs B

| metric | TF-TRUE − TF-SHUFFLE | verdict | TF-SHUFFLE − B | verdict |
|---|---|---|---|---|
| F1 excess | +0.0006 [−0.0010, +0.0024] | unresolved | +0.0158 [+0.0129, +0.0188] | improves |
| missing | +0.0009 [−0.0006, +0.0024] | unresolved | +0.0192 [+0.0163, +0.0219] | improves |
| spurious | +0.0027 [+0.0008, +0.0047] | improves | −0.0027 [−0.0065, +0.0011] | unresolved |
| beats-ratio dev | −0.0014 [−0.0030, +0.0004] | unresolved | +0.0145 [+0.0113, +0.0176] | improves |
| S4 | −0.0006 [−0.0007, −0.0005] | worsens | −0.0022 [−0.0025, −0.0019] | worsens |
| S5 | −0.0002 [−0.0003, −0.0001] | worsens | +0.0025 [+0.0022, +0.0029] | improves |

Shuffle share of the TF-TRUE gain (SHUFFLE − B over TRUE − B on F1 excess): 0.961. By the §14 reading rule,
TF-SHUFFLE − B F1 excess is "improves", not "worsens", so U2 is read as the window-specific-transfer test (which
it fails: +0.0006 [−0.0010, +0.0024] unresolved), with the R2
disclosure that the unaligned partner scaffold itself carries 0.961 of the TRUE gain through this interface
without any window-specific alignment. At NFE 4 the scaffold-specific part of the F1-excess gain is unresolved
(+0.0006 [−0.0010, +0.0024]); small scaffold-specific effects are resolved on spurious (+0.0027), S1–S3
(≤ +0.0014) and S4 / S5 (−0.0006 and −0.0002, both "worsens"), and at NFE 2 the F1-excess specificity is resolved (+0.0025 [+0.0010,
+0.0041]). In this seed the ungated module extracts little timing from the scaffold; its event gain is mostly a
scaffold-independent effect of the module.

### TF-TRUE vs ADD

| metric | effect | verdict |
|---|---|---|
| F1 excess | −0.0029 [−0.0067, +0.0007] | unresolved |
| missing | +0.0004 [−0.0032, +0.0038] | unresolved |
| spurious | −0.0159 [−0.0203, −0.0117] | worsens |
| beats-ratio dev | +0.0111 [+0.0080, +0.0143] | improves |
| S1 / S2 / S3 | +0.0063 / −0.0033 / +0.0055 | improves / worsens / improves |
| S4 | −0.0015 [−0.0019, −0.0011] | worsens (U5 fails) |
| S5 | +0.0034 [+0.0030, +0.0038] | improves |

### GTF-TRUE vs B

| metric | effect | verdict |
|---|---|---|
| **F1 excess** | **+0.0406 [+0.0353, +0.0458]** | improves (G1) |
| missing | +0.0429 [+0.0376, +0.0479] | improves |
| spurious | +0.0252 [+0.0193, +0.0308] | improves |
| beats-ratio dev | +0.0155 [+0.0122, +0.0189] | improves |
| S1 raw RMSE | +0.0002 [−0.0009, +0.0014] | unresolved |
| S2 raw corr | +0.0086 [+0.0063, +0.0108] | improves |
| S3 QRS RMSE | −0.0044 [−0.0057, −0.0032] | worsens |
| **S4** | **−0.0069 [−0.0076, −0.0063]** | **worsens** (G3) |
| **S5** | **−0.0015 [−0.0021, −0.0010]** | **worsens** (G3) |

### GTF-TRUE vs GTF-SHUFFLE (specificity), and GTF-SHUFFLE vs B

| metric | GTF-TRUE − GTF-SHUFFLE | verdict | GTF-SHUFFLE − B | verdict |
|---|---|---|---|---|
| F1 excess | +0.0242 [+0.0192, +0.0292] | improves (G2) | +0.0165 [+0.0135, +0.0195] | improves |
| missing | +0.0229 [+0.0180, +0.0277] | improves | +0.0199 [+0.0170, +0.0228] | improves |
| spurious | +0.0268 [+0.0209, +0.0326] | improves | −0.0017 [−0.0053, +0.0020] | unresolved |
| beats-ratio dev | +0.0014 [−0.0020, +0.0044] | unresolved | +0.0141 [+0.0108, +0.0172] | improves |
| S1 / S3 | −0.0066 / −0.0107 | worsens / worsens | +0.0068 / +0.0062 | improves / improves |
| S4 | −0.0046 [−0.0052, −0.0040] | worsens | −0.0023 [−0.0027, −0.0020] | worsens |
| S5 | −0.0042 [−0.0046, −0.0037] | worsens | +0.0026 [+0.0023, +0.0030] | improves |

Shuffle share of the GTF-TRUE gain: 0.405; GTF-SHUFFLE − B F1 excess is "improves", not "worsens", so G2 is
read as window-specific transfer, with the disclosure that the unaligned partner scaffold carries 0.405 of the
TRUE gain. The scaffold-specific part (+0.0242 of +0.0406) erases the S1 gain of the shuffle arm (GTF-TRUE vs B S1
unresolved), carries the whole S3 / S5 cost and about two thirds of the S4 cost; the scaffold-independent part already worsens S4 by −0.0023 [−0.0027, −0.0020] (reading
assumes additivity of the two paired contrasts).

### GTF-TRUE vs GTF-CONST (gate control)

| metric | effect | verdict |
|---|---|---|
| F1 excess | +0.0233 [+0.0184, +0.0282] | improves; non-inferiority holds (lower bound +0.0184 > −0.005) |
| missing | +0.0225 [+0.0175, +0.0273] | improves |
| spurious | +0.0241 [+0.0182, +0.0296] | improves |
| beats-ratio dev | +0.0042 [+0.0010, +0.0072] | improves |
| S1 / S2 / S3 | −0.0078 / +0.0030 / −0.0111 | worsens / improves / worsens |
| S4 | −0.0040 [−0.0046, −0.0034] | worsens (G5 structure part fails) |
| S5 | −0.0037 [−0.0041, −0.0032] | worsens |

### GTF-TRUE vs TF-TRUE (necessity / tie-break record) and GTF-TRUE vs ADD (G4)

| metric | GTF-TRUE − TF-TRUE | verdict | GTF-TRUE − ADD | verdict |
|---|---|---|---|---|
| F1 excess | +0.0241 [+0.0190, +0.0292] | improves | +0.0213 [+0.0161, +0.0263] | improves |
| missing | +0.0228 [+0.0177, +0.0277] | improves | +0.0233 [+0.0179, +0.0282] | improves |
| spurious | +0.0252 [+0.0192, +0.0308] | improves | +0.0093 [+0.0035, +0.0150] | improves |
| beats-ratio dev | +0.0023 [−0.0009, +0.0054] | unresolved | +0.0135 [+0.0100, +0.0168] | improves |
| S1 / S3 | −0.0080 / −0.0112 | worsens / worsens | −0.0017 / −0.0057 | worsens / worsens |
| S4 | −0.0041 [−0.0047, −0.0035] | worsens | −0.0056 [−0.0063, −0.0050] | worsens (G4) |
| S5 | −0.0039 [−0.0044, −0.0035] | worsens | −0.0005 [−0.0011, +0.0000] | unresolved |

ADD vs B is regenerated in the same process and reproduces R2 exactly (F1 excess +0.0194 [+0.0160, +0.0227];
S4 −0.0013; S5 −0.0010).

## Oracle diagnostic (`oracle_diagnostic.csv`; ORACLE = GT-R leakage, diagnostic only; never a method row)

| contrast (NFE 4) | F1 excess | missing | spurious | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|---|---|
| GTF-ORACLE vs B (GT-R leakage; diagnostic only) | +0.4989 [+0.4850, +0.5118] | +0.4956 | +0.4774 | +0.0586 | +0.3331 | +0.0809 | +0.0089 [+0.0069, +0.0110] improves | −0.0004 [−0.0017, +0.0010] unresolved |
| GTF-ORACLE vs GTF-TRUE (GT-R leakage; diagnostic only) | +0.4582 [+0.4445, +0.4717] | +0.4528 | +0.4522 | +0.0584 | +0.3246 | +0.0853 | +0.0158 improves | +0.0011 unresolved |
| GTF-ORACLE vs ADD-ORACLE (GT-R leakage; diagnostic only) | **+0.4564 [+0.4425, +0.4694]** | +0.4528 | +0.4390 | +0.0551 | +0.3173 | +0.0796 | +0.0122 improves | +0.0021 improves |
| ADD-ORACLE vs B (GT-R leakage; diagnostic only) | +0.0425 [+0.0379, +0.0474] | +0.0428 | +0.0384 | +0.0034 | +0.0158 | +0.0013 | −0.0033 worsens | −0.0025 worsens |

Reading by the ordered rule of §17.3: GTF-ORACLE vs ADD-ORACLE F1 excess "improves" with point ≥ +0.010 and
neither S4 nor S5 of GTF-ORACLE vs B "worsens" → **LIFTED**. Through the target-side interface an exact
scaffold reaches F1 excess 0.8164 (missing 0.071, spurious 0.038, timing median 15.6 ms) and improves
S1 / S2 / S3 substantially, where the R2 additive ORACLE adapter reached +0.0425 with an S4 / S5 cost. The
target-side interface can therefore carry far more event information than the additive one when a GT-derived
field is supplied. This does not identify whether the deployable arms are limited by the Global-TCN scaffold, by
its signal form (GTF-ORACLE was trained on the σ = 100 ms GT-R field, a different input distribution), or by the
module trained on it; it is a leakage measurement and says nothing about what any PPG-only arm can reach.

## Phase ablation (`phase_ablation.csv`, `phase_ablation_summary.csv`; scaffold rolled by +256 samples = +2.0 s; positive = TRUE better than shifted)

| arm | stratum (n) | F1 excess | missing | spurious | beats-ratio dev | S4 | S5 |
|---|---|---|---|---|---|---|---|
| GTF-TRUE | all (2,048) | **+0.0470 [+0.0406, +0.0536]** | +0.0514 | +0.0304 | +0.0224 | −0.0096 [−0.0105, −0.0088] worsens | −0.0038 [−0.0045, −0.0032] worsens |
| GTF-TRUE | in-phase (216) | +0.0121 [−0.0017, +0.0259] unresolved | | | | | |
| GTF-TRUE | anti-phase (559) | +0.0524 [+0.0400, +0.0649] improves | | | | | |
| GTF-TRUE | rest (1,273) | +0.0504 [+0.0426, +0.0590] improves | | | | | |
| TF-TRUE | all (2,048) | −0.0009 [−0.0021, +0.0002] unresolved | −0.0010 unresolved | −0.0009 unresolved | +0.0006 unresolved | −0.0002 [−0.0003, −0.0001] worsens | −0.0001 worsens |
| TF-TRUE | in-phase / anti-phase / rest | +0.0007 / −0.0001 / −0.0017 | all unresolved except rest "worsens" | | | | |

GTF-TRUE is phase-locked to the scaffold: shifting it by 2 s removes the whole gain and more (PHASE-GTF falls
below B), while the shifted arm has better S4 / S5 than the true-timed arm. The in-phase stratum, where a
2 s roll nearly re-aligns the beats, is unresolved as expected. On the all-window row TF-TRUE is indifferent to the
roll to within 0.001 on every event metric (all unresolved), with S4 / S5 differences ≤ 0.0003 in every stratum;
within strata the event-metric points stay ≤ 0.002 and are unresolved except the rest-stratum F1 excess (−0.0017,
shifted better): the ungated module extracts little timing from the scaffold.

## Direct-route attribution (`direct_route_attribution.csv`; prereg §18.2, evaluation-only cancellation of the raw residual route)

| arm | F1 excess full → no-direct | gain vs B full → no-direct | surviving share | S4 full → no-direct | S5 full → no-direct | reading |
|---|---|---|---|---|---|---|
| TF-TRUE | 0.3341 → 0.3311 | +0.0165 → +0.0135 | 0.82 | 0.3248 → 0.3245 | 0.2123 → 0.2126 | through the target stream |
| GTF-TRUE | 0.3582 → 0.3554 | +0.0406 → +0.0379 | 0.93 | 0.3289 → 0.3290 | 0.2162 → 0.2168 | through the target stream |

Both exceed the 50 % rule, so the fusion may be described as acting "through the target stream" (S5 / MLP
routes) rather than as a direct decoder-input write. The cancellation is first-order only (the raw residual
contribution of `(g ⊙) out` is subtracted from the summed block outputs); it does not remove second-order
interactions inside the frozen blocks.

## Site-wise (R1 8,192-window validation cohort, NFE 4; exploratory; 140 uncorrected CIs; `site_metrics.csv`)

| site | B | ADD | TF-TRUE | GTF-TRUE | GTF-CONST | GTF-TRUE − B | GTF-TRUE − TF-TRUE | TF-TRUE − B | ADD − B |
|---|---|---|---|---|---|---|---|---|---|
| sternum | 0.3784 | 0.4105 | 0.4007 | **0.4321** | 0.4012 | +0.0537 [+0.0486, +0.0593] | +0.0314 [+0.0264, +0.0363] | +0.0223 [+0.0189, +0.0257] | +0.0321 |
| head | 0.4239 | 0.4541 | 0.4466 | **0.4623** | 0.4467 | +0.0384 [+0.0333, +0.0433] | +0.0158 [+0.0111, +0.0207] | +0.0226 [+0.0196, +0.0259] | +0.0302 |
| wrist | 0.2625 | 0.2954 | 0.2774 | **0.2955** | 0.2770 | +0.0331 [+0.0277, +0.0383] | +0.0181 [+0.0131, +0.0229] | +0.0149 [+0.0119, +0.0179] | +0.0329 |
| ankle | 0.2165 | 0.2102 | 0.2259 | **0.2525** | 0.2258 | **+0.0360 [+0.0303, +0.0418]** | +0.0267 [+0.0211, +0.0321] | +0.0094 [+0.0064, +0.0122] | **−0.0063 [−0.0090, −0.0036]** |

(F1 excess; all listed CIs "improves" except ADD − B at the ankle, "worsens" as in R2.)

**Ankle.** R2's only site-wise regression — the additive adapter worsened ankle F1 excess by −0.0063 — is
reversed by both target-side arms: TF-TRUE +0.0094 and GTF-TRUE +0.0360, the latter with missing 0.633 vs
0.672 and spurious 0.584 vs 0.608. The GTF-TRUE gain is largest at the sternum and of similar size at the
head, wrist and ankle, so the target-side interface does not trade the weak sites for the strong ones. The
structure cost is uniform across sites: GTF-TRUE − B S4 worsens at every site (sternum −0.0082, head −0.0073,
wrist −0.0063, ankle −0.0067) and S5 worsens at every site (−0.0014 to −0.0025); TF-TRUE − B improves S5 and
worsens S4 at every site. Beats-ratio deviation improves for GTF-TRUE at every site (+0.0115 to +0.0187).

## Gate diagnostic (`gate_diagnostics.csv`; prereg §20; post-hoc, GT used for analysis only)

| statistic | value |
|---|---|
| (A) ρ_A Spearman within subject, window-mean gate vs scaffold F1@50 | **−0.491 [−0.525, −0.454]** |
| (B) same vs scaffold F1@150 | −0.638 [−0.666, −0.607] |
| (C) gate at matched scaffold peaks − at unmatched peaks (±4 samples) | **−0.3449 [−0.3577, −0.3309]** (matched mean 0.128, n 12,266; unmatched 0.471, n 9,699) |
| (D) gate mean by site, R1 8,192 cohort (p10–p90) | sternum 0.767 (0.72–0.85), head 0.755 (0.71–0.83), wrist 0.779 (0.71–0.86), ankle 0.776 (0.72–0.85) |
| (E) window-mean gate, good vs poor scaffold extraction (median F1@50 = 0.696) | 0.744 vs 0.796; difference −0.052 [−0.056, −0.048] |
| global gate (all samples, primary population) | mean 0.770, std 0.320, p10 0.128, p90 0.995; window-mean std 0.052 |
| GTF-CONST / GTF-SHUFFLE gates | 0.922 (std 0.0003) / 0.917 (std 0.004) — constant |
| VIZ cohort: gate in weak-scaffold segments (s < 0.35) vs strong | 0.970 (41,657 samples) vs 0.432 (23,879) |

**Wording by the frozen rule: GATE NOT INTERPRETABLE AS CONFIDENCE.** ρ_A is not ≥ 0.20 and (C) is not > 0
(both are strongly negative); only the non-constancy condition holds. The gate is therefore called an
*adaptive modulation gate* and nothing else in the body of this report (the preregistration's title is quoted as
such in the header). The observed pattern is the reverse of reliability
weighting: the gate opens (≈ 1) where the scaffold is weak and closes (≈ 0.1) around scaffold peaks that match
a GT beat, and it is slightly higher in windows whose scaffold was poorly extracted. Because the gate is a
pointwise function of the scaffold features multiplying the fusion output, it is itself a scaffold-shaped,
sample-aligned channel into `z_e`. The frozen arms cannot separate how much of the GTF-TRUE − TF-TRUE difference
(+0.0241) travels through that channel versus through the attention tokens; this is the open question named
in the recommendation.

## NFE event persistence (`nfe_event_persistence.csv`, `_summary.json`; ±250 ms greedy one-to-one; prereg §21)

| arm | matched fraction NFE 1 / 2 / 4 | mean abs δ ms 1 / 2 / 4 | median abs δ ms | sign consistency δ1 vs δ4 | NFE 4 strictly closer | ties | n (1 ∧ 4) |
|---|---|---|---|---|---|---|---|
| B | 0.821 / 0.823 / 0.816 | 75.1 / 73.1 / 71.5 | 54.7 / 46.9 / 46.9 | 0.911 | 0.186 | 0.680 | 15,290 |
| ADD | 0.829 / 0.830 / 0.825 | 72.8 / 70.6 / 69.2 | 46.9 / 46.9 / 46.9 | 0.908 | 0.187 | 0.679 | 15,464 |
| TF-TRUE | 0.843 / 0.845 / 0.840 | 74.2 / 72.3 / 70.5 | 54.7 / 46.9 / 46.9 | 0.902 | 0.190 | 0.673 | 15,857 |
| GTF-TRUE | 0.869 / 0.861 / 0.852 | 66.7 / 66.8 / 65.9 | 46.9 / 46.9 / 46.9 | 0.916 | 0.158 | 0.697 | 16,330 |

Intersection set (14,600 GT beats matched at NFE 1 and NFE 4 in all four arms): NFE 4 strictly closer
0.183 / 0.183 / 0.182 / 0.152; mean abs δ NFE 1 → 4: B 70.1 → 67.3, ADD 67.9 → 65.2, TF-TRUE 68.9 → 66.0,
GTF-TRUE 62.5 → 62.1. F1 excess at NFE 1 / 2 / 4: B 0.2931 / 0.3066 / 0.3176, GTF-TRUE 0.3505 / 0.3503 / 0.3582
(GTF-TRUE − B at NFE 1: +0.0575 [+0.0519, +0.0632]; vs GTF-SHUFFLE at NFE 1: +0.0464 [+0.0408, +0.0520];
TF-TRUE vs TF-SHUFFLE at NFE 1: +0.0010 unresolved). The GTF-TRUE advantage over B is largest at NFE 1 (+0.0575)
and shrinks to +0.0406 at NFE 4 as B improves with NFE, while GTF-TRUE's own F1 excess is flat (0.3505 / 0.3503 /
0.3582); its matched peaks move less between NFE 1 and 4 than any other arm's. No solver claim is made.

## Visual atlas (`visual_atlas/`, 64 VIZ windows, 619 GT beats, 128 PNGs = full + zoom per window; counts over all 64 windows only, no per-window selection; `atlas_summary.json`)

| arm | missing beats | spurious beats | windows with missing / with spurious | windows missing fewer than B | windows spurious more than B | windows F1 > B / < B | windows S4 lower than B | windows S5 lower than B |
|---|---|---|---|---|---|---|---|---|
| B | 359 | 327 | 57 / 59 | — | — | — | — | — |
| ADD | 348 | 327 | 55 / 58 | 12 | 9 | 14 / 8 | 19 | 20 |
| TF-TRUE | 353 | 338 | 56 / 58 | 10 | 13 | 12 / 12 | 19 | 35 |
| GTF-TRUE | 336 | 312 | 58 / 59 | 20 | 10 | 26 / 11 | 20 | 31 |
| GTF-CONST | 353 | 339 | 56 / 58 | 10 | 15 | 12 / 14 | 20 | 35 |
| GTF-ORACLE (GT-R leakage; diagnostic only) | 31 | 12 | 16 / 8 | 56 | 0 | 58 / 0 | 36 | 31 |

The six preregistered questions (§22), answered with the counts above:

**Missing beats recovered?** Partly. GTF-TRUE misses 336 of 619 GT beats vs B's 359 and has fewer missing
beats than B in 20 of 64 windows (ADD 12, TF-TRUE 10, GTF-CONST 10); 58 windows still contain at least one
missing beat. GTF-ORACLE (GT-R leakage; diagnostic only): 31 missing, fewer than B in 56 windows.

**New spurious beats?** GTF-TRUE has fewer spurious beats than B overall (312 vs 327) but more than B in 10
windows (ADD 9, TF-TRUE 13, GTF-CONST 15); every deployable arm has at least one spurious beat in 58–59 windows.

**QRS slope / shape smoother or distorted?** The atlas summary carries no separate slope count; the
window-level S4 (derivative) comparison is the proxy: each deployable arm has a lower QRS-core derivative
error than B in only 19–20 of 64 windows, i.e. the derivative is worse than B in the majority of windows for
every fusion arm, GTF-TRUE included.

**R2 derivative / curvature degradation visible?** Yes for the derivative (above). Curvature: TF-TRUE and
GTF-CONST have lower S5 than B in 35 windows and GTF-TRUE in 31 (ADD 20), pointing the same way as the primary S5
verdicts for ADD and TF-TRUE (ADD worsens, TF-TRUE improves vs B); the GTF-TRUE count (31 of 64, about half) does
not resolve on 64 windows.

**GTF suppresses injection in weak PPG segments?** No, on the atlas proxy (weak-scaffold segments, s < 0.35,
standing in for weak PPG; PPG quality itself is not measured): the GTF-TRUE gate averages 0.970 over weak-scaffold
samples (41,657) and 0.432 over strong segments (23,879) — the reverse of suppression; the gate opens where the
scaffold is weak and closes around scaffold peaks.

**Ankle still pathological?** Ankle windows (16): GTF-TRUE missing 85 / spurious 81 vs B 94 / 89; F1 higher
than B in 8 windows, lower in 2; S4 lower than B in 6 (ADD 94 / 89, 1 / 1, 4; TF-TRUE 92 / 87, 3 / 0, 5;
GTF-CONST 92 / 87, 4 / 2, 6; GTF-ORACLE (GT-R leakage; diagnostic only) 9 / 5, 16 / 0, 11). On the 64 VIZ
windows the wrist, not the ankle, carries the most missing / spurious beats for every deployable arm (GTF-TRUE
wrist 103 / 94 vs ankle 85 / 81; GT beats 153 vs 152); the ankle is the weakest site on F1 excess only in the
exploratory 8,192-window site-wise table (§19). The GTF-TRUE ankle improvement is visible in the counts.

## What this does NOT prove

- **Stronger-supervision caveat.** The Global-TCN was trained on GT ECG R-peaks; every TRUE arm carries
  target-derived training supervision the baseline never had. Comparisons with other methods must give them
  equivalent event supervision. ORACLE is target leakage by design and is a diagnostic only.
- **Scaffold reliability shift** (prereg §0): the Global-TCN is in-sample on 10 of the 12 training subjects
  and out-of-sample on an0 / k2s; the arms were trained with an optimistic scaffold. Unseparated.
- **Single seed** (42), one loader / source / `(t, r)` realisation shared by the arms; the CIs quantify
  validation-window sampling within an0 / k2s only. **No test evidence** (`kjd` / `ssx` never loaded).
- **No SOTA, no novelty claim**; a 12.8 k-parameter probe on a frozen generator, 2,200 steps.
- The gate is **not** confidence; the frozen rule says so, and the wording "adaptive modulation gate" is the
  strongest used. The mechanism by which the gate makes the scaffold usable (mask-like channel vs attention
  tokens) is **not** separated by the frozen arms.
- "Through the target stream" is a first-order, evaluation-only cancellation result (§18.2), not a mechanism.
- The ORACLE ceiling (F1 excess 0.8164) measures what the interface can carry when GT timing is supplied; it
  does not bound any PPG-only arm.
- E1–E4 are detector-dependent (frozen detector); S4 / S5 carry the structure verdict; missing = 1 − recall.
- Site-wise CIs (140) and the persistence / phase / gate diagnostics are secondary and uncorrected.
- U1 fails on magnitude with a CI that excludes zero (+0.0165 vs +0.020); the frozen rule is applied as
  written. Had U1 and U2 passed, U4 and U5 would still have failed, so A was not reachable on these numbers.

## Deviations, corrections and disclosures

1. **Evaluator variant-source bug (fixed before the final evaluation; in this commit with a static test).**
   In `7f5dc7e` the source module for the PHASE-* / NODIRECT-* variants was chosen with `arm.endswith("TF")`,
   which is also true for `PHASE-GTF` / `NODIRECT-GTF`, so both GTF variants would have been generated from
   the TF-TRUE module. It was noticed in the first evaluation run's log, where the PHASE-GTF row equalled the
   PHASE-TF row to every printed digit; the code was inspected, the run terminated, the mapping replaced by an
   explicit `VARIANT_SRC` table and pinned by `test_evaluator_variant_arms_map_to_their_own_source_modules`.
   The nine grid arms, all U / G items, the oracle reading and the decision function are independent of the
   mapping and were not changed (`git diff 7f5dc7e -- src/ppg2ecg/flow/rhythm_fusion.py` is empty). The first
   run had already printed its grid rows before termination; no rule, threshold or arm was altered afterwards.
   All evaluator outputs (`metrics_by_window.csv` … `provenance.json`, `site_metrics.csv`, `gate_diagnostics.csv`)
   come from the single completed run (UTC 2026-09-02 15:05:30 → 15:19:51); preparation / training artifacts
   (12:54–14:54 UTC) and the visual atlas (15:20 UTC) were produced by their own scripts.
2. **Site gate bootstrap indexing bug (fixed in this commit with a static test; artifacts not regenerated).**
   The (D) rows' `subject_boot_*` columns in `gate_diagnostics.csv` were computed from `G8[idx]` with `idx`
   positions of the site-masked arrays, i.e. from the first 2,048 windows of the whole cohort for every site;
   they read 0.7767 [0.7744, 0.7789] identically for all four sites and are **invalid**. The per-site `mean` /
   `p10` / `p90` in the same rows index the masked array and are valid (table above). The (D) statistic is
   not part of the §20 wording rule (which uses (A), (C) and the global std). The fix (`G8m = G8[m]`) is pinned
   by `test_evaluator_site_gate_bootstrap_indexes_the_site_masked_gate_array`; the CSV is left as produced
   because regenerating it means re-running the whole evaluator under non-deterministic cuDNN and re-collecting
   every table for four secondary columns.
3. **Dirty tree at evaluation.** `provenance.json` records `dirty_files: 2` at `7f5dc7e`: the variant-source fix
   and its test (item 1), both committed here.
4. **`tests_ran.summary` is empty** in `provenance.json` (R2 recorded pytest's trailing "-- Docs:" line instead
   of the count line, from the same cause). Cause found now: the evaluator's pytest subprocess passes `-q` on top
   of the project's `addopts = "-q"`, which suppresses the count line. The evaluator's own subprocess call is
   left unchanged in this commit (it is not re-run for R3), so a future run would record the same empty field
   unless `-o addopts=""` is added. The
   recorded exit code 0 and empty skip list are the meaningful fields. The full suite run before this commit
   (count line restored by `-o addopts=""`): **313 passed, 0 failed, 0 skipped** (22 in `tests/test_r3_rhythm_fusion.py`).
5. **Determinism flags.** Training (preflight and six arms) ran with `cudnn.deterministic = true`
   (`seed_everything(42, deterministic=True)`, recorded in `train_provenance_*.json`). Evaluator and atlas ran with
   cuDNN flags at framework defaults, as C0 / C1 / V1 (flags not recorded there) and R2's evaluator
   (`cudnn_deterministic: false` recorded, as here); step-0 parity and the R2 regression (0 flags) hold under them.
6. **Loss columns.** `loss_weighted` is normalised by the adaptive weighting and is ≈ 0.9999 throughout; the raw
   `mse` column is the informative one and is what the runtime table reports. No training decision depended on
   either.
7. **Operational.** The evaluation chain was launched three times: the first run was terminated (item 1) — the
   termination command matched its own shell and had to be repeated by PID; the second launch started from the
   wrong working directory and exited immediately (exit 2) before any computation; the third completed. The
   preflight (100 GTF-TRUE steps) was discarded per §26 step 9; only its runtime numbers are reported.
8. **GTF-ORACLE peak memory** (7,638 MiB) exceeds the other arms because the oracle field cache is resident on
   the GPU. §12 has no memory budget; its only STOP condition is the 6 GPU-hour projection (1.44 GPU-h here).

## Recommended next step (recommendation only — NOT IMPLEMENTED)

The decisive R3 facts are (i) the ungated cross-attention module extracts little timing from the scaffold
(TF-TRUE ≈ TF-SHUFFLE on F1 excess / missing; roll-insensitive on the event metrics — all-window contrasts
unresolved and ≤ 0.001, rest stratum −0.0017 resolved), (ii) adding an 81-parameter
pointwise gate whose input is the scaffold makes the scaffold usable (+0.0241 over TF-TRUE, phase-locked,
present at NFE 1), and (iii) the gate is anti-correlated with scaffold extraction quality (ρ_A −0.49), closes
around matched scaffold peaks (0.13 vs 0.47) and, on the VIZ cohort, is ≈ 1 where s < 0.35 and ≈ 0.4 elsewhere
— consistent with a scaffold-dependent multiplicative mask rather than a reliability weight. The next
preregistered probe should **separate the two channels** through
which the scaffold enters GTF before any architectural escalation:

- arms with the same 12,849 parameters, frozen generator and 2,200-step protocol: GATE-TRUE / TOKENS-SHUFFLE
  (gate sees the window's own scaffold, attention tokens see the deranged partner's), GATE-SHUFFLE / TOKENS-TRUE,
  and a MASK-ONLY arm (fusion output replaced by a learned constant channel vector so that only `g(s)` can
  carry timing; ≤ 209 parameters);
- GTF-TRUE replicated over three seeds to bound the seed variance that the present CIs do not contain;
- the same U / G-style gates with S4 / S5 as the structure criterion, the same shuffle and phase controls,
  the site-wise secondary kept because the ankle F1-excess result is where R3 reverses R2's site-wise regression
  (R3 also reverses R2's S5 degradation for TF-TRUE on the primary population).

If the MASK-ONLY arm reproduces most of the GTF-TRUE event gain, the cross-attention is unnecessary and the
structure cost should be attacked at the mask (e.g. a preregistered smoothness or magnitude constraint on
`g`), not by a larger fusion module. No larger Transformer, no generator retraining, no test subjects, no C2.

## Artifacts (`artifacts/r3_rhythm_fusion/`, never in git)

`hook_audit.json`, `frozen_checkpoint_manifest.json`, `parameter_manifest.json`, `initialization_hashes.json`,
`shuffle_manifest.csv`, `prepare_provenance.json`, `runtime_preflight.json`, `training_log_{arm}.csv`,
`train_provenance_{arm}.json`, `metrics_by_window.csv`, `event_metrics.csv`, `structure_metrics.csv`,
`paired_bootstrap.csv`, `oracle_diagnostic.csv`, `decision.json`, `phase_ablation.csv`,
`phase_ablation_summary.csv`, `direct_route_attribution.csv`, `nfe_event_persistence.csv`,
`nfe_event_persistence_summary.json`, `gate_diagnostics.csv`, `site_metrics.csv`, `provenance.json`,
`visual_atlas/` (128 PNGs, `atlas_index.csv`, `atlas_summary.json`). Modules:
`outputs/r3_{tf_true,tf_shuffle,gtf_true,gtf_shuffle,gtf_const,gtf_oracle}_seed42/module_step{0,550,1100,2200}.pt`.

## STOP

R3 ends here. C2 training remains deferred. Nothing beyond this report is implemented.
