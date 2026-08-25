# Improved MeanFlow (iMF) — paper / code audit for the one-step PPG→ECG experiment

Prepared 2026-08-25 while A0-b was training (read-only research; **implementation was gated on the A0-b result**).
Sources were located and every equation below was checked against the paper text and the official code by a
multi-agent audit with adversarial verification (65 agents, 3 lenses, 33 verified claims, 0 refuted on substance);
the JAX loss file was additionally read line by line by the author of this document.

## 1. Sources (verified)
| Item | Value |
|---|---|
| MeanFlow paper | Geng, Deng, Bai, Kolter, He. *Mean Flows for One-step Generative Modeling*. arXiv:2505.13447 (v1, 19 May 2025). **NeurIPS 2025 oral**. Official code: `github.com/Gsunshine/meanflow` (JAX) |
| Improved MeanFlow paper | Geng\*, Lu\*, Wu, Shechtman, Kolter, He. *Improved Mean Flows: On the Challenges of Fastforward Generative Models*. arXiv:2512.02012 (v1 1 Dec 2025, **v2 9 May 2026**, used here). **CVPR 2026 highlight** |
| Official iMF code | `https://github.com/Lyy-iiis/imeanflow` — **official** (arXiv comment "Code at …", README "official JAX implementation", owner = co-first author Yiyang Lu (IIIS Tsinghua / MIT), contributors He, Zhao, Lu, Geng; `torch` branch = inference-only re-implementation) |
| Pinned clone | `external/iMeanFlow` @ **`bf60cd7cb653f6628e59d48034b333c5eba445e2`** (main, 2026-02-19), clean, never modified. Key files: `imf.py` (objective/sampler), `models/imfDiT.py` (network, conditioning), `models/embedder.py`, `configs/default.py` |

## 2. Definitions (paper notation; t = 1 is noise, t = 0 is data)
- Interpolant / instantaneous velocity (MF §2, iMF §3): `z_t = (1 − t) x + t e`, `e ~ N(0, I)`, conditional velocity `v_t = e − x` (`imf.py:350-351`).
- **Average velocity** (MF Eq. 3, iMF Eq. 3): `u(z_t, r, t) ≜ 1/(t − r) ∫_r^t v(z_τ, τ) dτ`, with `r ≤ t`; boundary `u(z_t, t, t) = v(z_t, t)`.
- **MeanFlow identity** (MF Eq. 6, iMF Eq. 4): `u(z_t, r, t) = v(z_t, t) − (t − r) · d/dt u(z_t, r, t)`, where the total derivative (MF Eq. 7–8) is
  `d/dt u = v(z_t,t) · ∂_z u + ∂_t u` (r held fixed), i.e. the **JVP of u with tangent (v, 0, 1) on (z, r, t)**.
- Original MF objective (MF Eq. 9–11, Alg. 1): `‖u_θ(z_t, r, t) − sg(u_tgt)‖²`, `u_tgt = v_t − (t − r)·JVP(u_θ; (v_t, 0, 1))` with `v_t = e − x` as tangent; stop-gradient on the whole target.
- **iMF re-parameterisation, "MeanFlow as v-loss"** (iMF §4.1, Eq. 9–12, Alg. 1):
  `V_θ(z_t) ≜ u_θ(z_t, r, t) + (t − r) · JVP_sg(u_θ; v_θ)`, loss `E ‖V_θ − (e − x)‖²`,
  where `JVP_sg` = the directional derivative `d/dt u_θ` computed with **the network's own velocity prediction `v_θ` as z-tangent**, with **stop-gradient on the JVP outcome only** (`imf.py:373-386`: `u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, 1, 0))`; `V = u + (t − r) * stop_gradient(du_dt)`; `loss_u = Σ(V − sg(v_g))²`). With `v_θ ≡ e − x` this is algebraically the original MF loss (iMF Eq. 9–10); the change to `v_θ` (Eq. 12) makes `V_θ` a function of `z_t` alone.
- `v_θ`: iMF Alg. 1 uses the **boundary condition `v_θ(z_t, t) = u_θ(z_t, t, t)`** (h = 0). The official code instead adds an **auxiliary v-head** (8 extra transformer blocks, `imfDiT.py:149,254-267`) trained with an extra FM loss `Σ(v − sg(v_g))²` (`imf.py:389-392`; paper App. A). Both are the paper's; the aux head is the ImageNet-scale choice.
- **Adaptive weighting** (MF Eq. 22, §4.3; iMF code `imf.py:380-382`): per-sample `L / sg((L + c)^p)` with `p = 1.0`, `c = 0.01` (`configs/default.py:65-66`), `L = Σ_dims Δ²`. Applied to both loss terms in the code; the iMF paper's Table 4 has no row for it but App. A states it is used.
- **(r, t) sampling** (iMF Tab. 4; `imf.py:120-139`): `t, r` i.i.d. **logit-normal(μ = −0.4, σ = 1.0)**, then `t = max, r = min`; **50 % of the batch forced `r = t`** (`data_proportion = 0.5`, "ratio r ≠ t 50 %") — those samples are plain flow matching.
- **Time conditioning**: MF §4.3 "`u_θ(·, r, t) ≜ net(·, t, t − r)`" (t and h = t − r, each via sinusoidal embedding → 2-layer MLP, summed). The official iMF DiT conditions **on `h = t − r` only** (`imfDiT.py:342-344`, following arXiv:2502.13129 "noise conditioning is not necessary"); `t` is passed but unused. The JVP w.r.t. `t` therefore acts through `h` (dh/dt = 1).
- **Sampling** (MF Eq. 12, Alg. 2; iMF §3; `imf.py:42,112-114`): `z_r = z_t − (t − r) · u_θ(z_t, r, t)` on `t_steps = linspace(1, 0, N+1)`; **1-NFE: `x̂ = z_1 − u_θ(z_1, r = 0, t = 1)`** (h = 1), `z_1 ~ N(0, I)`. No EMA-free path: inference uses EMA weights (0.9999) in the official code.
- **Guidance / labels (ImageNet-specific, not used here)**: CFG with `ω ∈ [1, 8]` sampled from `ω^{−β}`, guidance interval `[t_min, t_max]` sampled and provided to the network as tokens, guided target `v_tgt = (e − x) + (1 − 1/ω)(v_c − v_u)` (iMF Eq. 17, Alg. 2), class dropout 0.1, in-context condition tokens (class 8, time 4, cfg 4, interval 2×2), RoPE attention, VAE latents 32×32×4, EMA 0.9999, Adam(0.9, 0.95) lr 1e-4 constant + 10-epoch warm-up, wd 0, batch 256/1024, 240–800 epochs, adaptive-weight sum over 4096 latent dims.

## 3. Mapping onto our conditional 1-D task (decisions, with reasons)
| Aspect | Official iMF | Our A2 implementation (`src/ppg2ecg/flow/imeanflow.py`) | Reason |
|---|---|---|---|
| Backbone | iMF-DiT (Transformer, RoPE, in-context tokens) | **upstream PENGUIN Flow-SSM/S5, unmodified, same 4,568,707 parameters** | controlled comparison: objective is the only change |
| Condition | class label + CFG tokens | **PPG signal through the unchanged PPG stream** (no CFG, no dropout, no guidance interval) | our condition is a same-length waveform; CFG is ImageNet-specific |
| Time inputs | `h` only (code) / `(t, h)` (MF paper) | **`cond = E(t) + E(h)`, `h = t − r`, using the backbone's single existing `TimestepEmbedder` for both (shared weights → parameter count unchanged)** | keeps the baseline's `t`-conditioning path; `E(h)` adds the interval information without new parameters. h-only (official) noted as an alternative, not adopted |
| `v_θ` | auxiliary v-head (+8 blocks) | **boundary condition `v_θ = u_θ(z, t, t)` (h = 0), paper Alg. 1** | an aux head would add parameters and change the backbone |
| JVP | `jax.jvp(u_fn, (z, t, r), (v_c, 1, 0))` | `torch.func.jvp(u_fn, (z, t, r), (v_θ.detach(), 1, 0))` — forward-mode AD verified through the S5 scan (§4); double-vjp `torch.autograd.functional.jvp` as verified fallback | exact same tangents |
| Stop-gradient | on `du/dt` and on the target `v_g` | identical: `V = u + (t − r)·du_dt.detach()`, target `(e − x)` (constant) | — |
| Loss | adaptive-weighted `Σ_dims (V − v_g)²` (+ aux v loss) | adaptive-weighted `Σ_T (V − (e − x))²` with `p = 1`, `c = 0.01`; no aux loss (no aux head; the 50 % `r = t` samples train `v_θ` through `V = u`) | as published; c is negligible relative to `Σ_T Δ²` either way |
| (r, t) sampling | logit-normal(−0.4, 1), 50 % `r = t` | identical | — |
| Time convention | `t = 1` noise | **kept as iMF (t = 1 noise, z_t = (1−t)x + t e)** inside `imeanflow.py`; the backbone only sees scalars `t`, `h` | avoids sign errors; the baseline's `τ = 1 − t` convention is irrelevant to the sample-level comparison |
| Sampling | `z_1 − u(z_1, 0, 1)` | identical 1-NFE; multi-step `z_r = z_t − (t − r)u` available for diagnostics | — |
| Optimiser / schedule | Adam(0.9,0.95) 1e-4 + warm-up, EMA 0.9999, wd 0 | **AdamW 1e-3, wd 0.01, batch 64, no EMA — the A0/A0-b baseline settings** (pre-registered rule: keep baseline optimiser; change only with a literature-based, logged reason if training is unstable) | isolate the objective |
| Selection | FID on EMA weights | deterministic fixed-bank **iMF loss** on the validation set (same 4 banks: `z` reused as `e`, `t` reused, `r` drawn once per bank with seed) | mirrors A0-b's criterion |

## 4. JVP feasibility on the frozen backbone (measured 2026-08-25, tiny config h_dim 16 / 2 blocks)
- `torch.func.jvp` (forward-mode AD) runs through the upstream S5 stack (`torch.vmap` + associative scan + `torch.jit.script`
  binary operator, complex64) without modification; `torch.autograd.functional.jvp` (double-VJP) gives **the same directional
  derivative** (max |Δ| = 0 to float precision) and the backward pass through the primal is finite.
- CPU float32 check against central finite differences on the same tiny backbone (deterministic on CPU): max |JVP − FD| =
  1.2e-2 (ε = 0.1) → 1.1e-3 (0.03) → **1.1e-4 (0.01)** → 6e-5 (0.003, float noise floor), i.e. the expected O(ε²) convergence;
  `torch.func.jvp` and the double-VJP agree to 2.4e-7. (On GPU the FD itself is unreliable because the scan is not bit-deterministic
  run-to-run; the two AD paths still agree exactly there.) Exact JVP/identity correctness is additionally unit-tested on an analytic
  toy field (`tests/test_imeanflow.py`) where the average velocity is known in closed form.
- Memory/time at the full config (batch 64, T = 1024) is measured in the A2 smoke test after A0-b releases the GPU.

## 5. What is deliberately NOT implemented
CFG / guidance interval / ω conditioning, class dropout, auxiliary v-head, in-context tokens, EMA, mixed precision,
learning-rate warm-up, VAE latents — all ImageNet-specific or architecture-changing (docs/PREREGISTRATION_V0.md §7).
