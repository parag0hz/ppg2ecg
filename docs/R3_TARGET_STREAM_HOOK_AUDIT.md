# R3 — Target-Stream Hook Audit

**Purpose.** Before any R3 architecture is written, identify the exact tensor in the frozen generator that
carries the noisy / generated ECG stream, so that rhythm conditioning can be fused on the **target side**
without touching the PPG morphology stem (the R2 path). Read-only inspection of `src/ppg2ecg/flow/imeanflow.py`
(`MeanFlowS5.u`) and of the frozen upstream `external/PENGUIN/src/models/PENGUIN.py` (pinned `6cd70cd`,
never modified). Machine-readable record: `artifacts/r3_rhythm_fusion/hook_audit.json` (records HEAD `5e064ef`, PENGUIN `6cd70cd`, iMeanFlow `bf60cd7`, generator file sha256 `557c7054…`).

## 1. The frozen forward pass (`MeanFlowS5.u`, src/ppg2ecg/flow/imeanflow.py L45-60)

```
L48  ppg_e = bb.pre_conv_ppg(ppg)          # PPG morphology stem            [B,128,1024]   <- R2 injected here
L49  z_e   = bb.pre_conv_target(z)         # noisy/generated waveform stem  [B,128,1024]   <- R3 hook
L51  cond  = bb.timestep_embedder(h)       # adaLN conditioning              [B,128]
L56  all_dx = zeros_like(z_e)
L57  for blk in bb.flow_ssm_list:          # 4 Flow_SSM blocks
L58      ppg_e, dx = blk(ppg_e, z_e, cond) #   PPG stream is chained; the TARGET stream is NOT (same z_e to every block)
L59      all_dx = all_dx + dx
L60  return bb.final_layer(all_dx, cond)   # decode to the waveform            [B,1,1024]
```

Inside every block (PENGUIN.py L111-120) the target stream is `x_t = z_e` → LayerNorm over channels →
adaLN modulation → S5 → `pre_attn_target`; the PPG stream reaches it **only** at L115
`target_cond = target_cond + ppg_cond`. There is no attention (`cross_attn` is dead code) and no other
PPG → target path.

## 2. Selected hook

| field | value |
|---|---|
| module / tensor | `MeanFlowS5.u`, `z_e = backbone.pre_conv_target(z)` (src, L49) |
| producer | `pre_conv_target = Sequential(Conv1d(1→128, k=32, same), SiLU, Conv1d(128→128, k=32, same))` (PENGUIN.py L170-174; 528,640 frozen params; no stride, no pooling) |
| shape | `[B, 128, 1024]` (h_dim = 128; verified numerically on the frozen checkpoint: `(2, 128, 1024)`) |
| temporal length | **1024 = the waveform length**, sample-for-sample aligned (same-padded convolutions only) |
| channel dimension | 128 |
| PPG already merged here? | **No** — within one evaluation of `u`. PPG first meets the target stream inside each block at PENGUIN.py L115. (`z` itself carries the PPG-conditioned sampler trajectory at NFE > 1 and `(1−t)·GT ECG + t·noise` during training, which is what makes it the target stream.) |
| exists before final decoding? | Yes: L49 precedes the four blocks (L57-59) and `final_layer` (L60) |
| residual path without touching the PPG encoder? | Yes: `z_e' = z_e + fusion(z_e, scaffold)` inserted between L49 and L56 in a src/ subclass of `MeanFlowS5`; `pre_conv_ppg` (L48) and every `backbone.*` tensor untouched; `external/PENGUIN` untouched |
| reach | because the target stream is not chained, a residual on `z_e` reaches **all four** blocks identically — and, through each block's raw residual (§2.1), the decoder input directly |
| receptive field | `z_e[t]` depends on `z[t−30 … t+32]` (two same-padded k = 32 convolutions, 63 samples ≈ 0.49 s); index alignment is one-to-one; the even kernel gives a ≈ 1-sample asymmetry, below the 4-sample token pitch and the 50 ms tolerance |

### Why it satisfies the five criteria

1. **Downstream of the noisy/generated waveform projection** — it *is* the output of that projection.
2. **Not the PPG morphology stem** — `pre_conv_ppg` is a separate module whose output `ppg_e` never enters
   `pre_conv_target`; the two stems share nothing.
3. **Waveform-aligned temporal coordinates** — length 1024, one-to-one with the sample axis, so sample `t` of
   `z_e` is sample `t` of the ECG being generated; no interpolation is needed (spec §2 satisfied).
4. **Before final waveform decoding** — the blocks and `final_layer` follow.
5. **Residual conditioning without changing the frozen PPG encoder** — the subclass only re-states the
   frozen thirteen lines with one insertion; the R2 subclass already established this pattern for the PPG
   side and its arm-B parity was bit-exact.

### 2.1 Consumption routes of `z_e` — including a direct route to the decoder

`z_e` reaches the output through **three** routes inside every block (PENGUIN.py L111-120):

1. `norm1_target` → adaLN → S5 → `pre_attn_target` (+ `ppg_cond` at L115) → `post_attn_target`;
2. `norm2_target` → adaLN → `mlp_target`;
3. **the raw residual**: `res_target = x_t` (L112) and `dx_t = res_target + target_cond` (L120), so `z_e` is
   forwarded unchanged into `dx_t`, summed over the four blocks (imeanflow.py L59) and decoded by
   `final_layer` (`norm_final` LayerNorm over channels → adaLN → `Linear(128 → 1)`).

Route 3 bypasses the S5 / MLP paths. Measured on the frozen checkpoint (16 primary an0 windows, seed-0
source rows; rank-32 channel-varying perturbations of `z_e` at 5 / 20 / 50 % of its std, h ∈ {1, 0.25, 0.1};
cancelling the direct route in src/ by `all_dx −= 4·δ`): the direct route carries **0.21–0.70** of the
first-order output response (`1 − |Δu without route 3| / |Δu full|`, `hook_audit.json`
`direct_route_share_measured`). **Consequence:** a fusion output added to `z_e` is, to first order, also a
direct per-time-step write into the decoder input — closer to the rejected `all_dx` hook than the block
diagram suggests, and able to shape output morphology without passing through the backbone's S5 / MLP
paths. The spec-literal `H_z' = H_z + (g ⊙) out` is kept (cancelling the residual would alter the frozen
forward structure); the preregistration discloses this (§0) and pre-registers a retrain-free attribution
diagnostic that evaluates the trained fusion with route 3 cancelled (§18.2).

### Property that constrains the fusion output

Every route passes a `LayerNorm(128, elementwise_affine=False)` over the **channel** axis at each time step
(`norm1_target`, `norm2_target`, `norm_final`), so a channel-uniform addition to `z_e` is annihilated
exactly (measured max |Δu| = 7.2e-07 on the frozen checkpoint); channel-varying and per-time-step-scaled
additions are **not** (this is not scale invariance). The R3 output projection is a `Conv1d(32 → 128, k=1)`,
i.e. channel-varying by construction (the same property R2 relied on).

## 3. Alternatives considered and rejected

| candidate | why not |
|---|---|
| the per-block target input `x_t` inside `Flow_SSM_Layer` | per-block or subset residuals are reachable from src/ (imeanflow.py L58 passes `z_e` to each block) without touching `external/`, but the spec (§6) specifies **one** fusion on the shared target hidden; per-block variants add un-preregistered capacity / asymmetry and are not considered |
| `all_dx` before `final_layer` | PPG has already been merged in every block; fusing there would act on a PPG-contaminated representation, defeating the WHEN / WHAT separation |
| `pre_conv_ppg` output (`ppg_e`) | the R2 path this stage is designed to avoid |

## 4. Verdict of the audit

A clean target-side tensor exists (`z_e`, `[B,128,1024]`, PPG-free within one evaluation, waveform-aligned,
pre-decoding, hookable from src/). **R3 may proceed** with the fusion inserted at this tensor, with the §2.1
caveat carried into the preregistration: the direct residual route means the hook is not a pure "WHEN" port, and
the WHEN / WHAT reading of any result is conditioned on the attribution diagnostic. Temporal resolution is exactly 1024,
so no mapping between waveform coordinates and a hidden resolution is required (spec §2).
