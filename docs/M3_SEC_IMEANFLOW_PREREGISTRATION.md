# M3 — SEC-iMF: Structural Endpoint Consistency for Improved MeanFlow

**Status: PREREGISTRATION. Frozen on commit. Never edited post-hoc.**
Written 2026-09-07, BEFORE any M3 training run, checkpoint, or validation metric.

Method name, chosen here and never renamed: **SEC-iMF**. Primary arm label **E**.

## 1. Provenance

| | value |
|---|---|
| **M3_START_SHA** | `1cf0d361e739ca9f26d7cd9b988cbbd1e85e8e29` |
| M2 result commit | `1cf0d36` — "M2 RESULT: VERDICT D — STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED" |
| M2 preregistration | `e8a80843b4e663887c283270eaaaa6dee4b504d1`, untouched since (0 later commits) |
| M2 implementation | `f84a923`; `M3_START_SHA` is its descendant |
| PENGUIN | `6cd70cdefb91f10efeb8dce34019b5067cb25344` |
| iMeanFlow | `bf60cd7cb653f6628e59d48034b333c5eba445e2` |
| A4 checkpoint md5 | `31c042d291052fbb6dc15263ad316be2` |
| A4 / M2-U state sha256 | `20ba7234f25e0fe30960ba0e441d4b9f751c20003c85fee80ec3c08367aa095e` |
| E2 contract sha256 | `06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115` |
| C2 | deferred; no `outputs/c2*` |
| M2 arms X and Q | never trained (M2 §13 stop rule honoured) |

The frozen M2 preregistration, report and artifacts are **not edited by M3**. M2's verdict stands as
**D — STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED**.

## 2. Question, and what M3 is not

M2 asked **where** the ordinary iMeanFlow regression error should receive more weight, and the answer
was that emphasising target-ECG high-gradient locations makes `qrs_deriv_rmse` and `qrs_curvature_err`
**worse**. M3 asks a different question: spatially reweighting the original regression does not
directly constrain differential morphology, so does a **direct consistency loss on the model's clean
endpoint estimate** improve ECG slope and curvature while leaving the original objective intact?

M3 changes **what is constrained**, not where the original loss is weighted. No QRS mask, no R
annotation, no M2 `structure_weight`, no new inference module.

**M2's post-hoc event/F1 improvements are not evidence for M3 and are not used here.**

## 3. Endpoint algebra — verified against the production sampler, not taken on trust

`imeanflow.sample_meanflow` (L157-166) and `event_reliability.sample_meanflow_schedule` (L68-84) both
compute, identically:

```
z_r = z_t - (t - r) * net.u(z_t, ppg, t, t - r)
```

so the fourth argument of `net.u` is `h = t - r`. The direct jump to the data endpoint `r = 0`
therefore uses `h0 = t`:

```
z_t    = (1 - t) x + t e                      (t = 0 data, t = 1 noise)
u0     = net.u(z_t, ppg, t, t)
x0_hat = z_t - t.reshape(-1,1,1) * u0
```

`sample_meanflow(n_steps=1)` computes `e - 1·u(e, ppg, 1, 1)`, i.e. exactly `x0_hat` at `t = 1`.

**Algebraic gate (§5), already run and PASSED before this document.** With the oracle `u0 = e − x`,
`z_t − t·u0 == x`. Measured over B ∈ {1, 7, 64, 3}, T ∈ {8, 1024, 257}, random t, float32:
generic max error **4.77e-07**, `t = 0` identity **0**, `t = 1` identity **0**; gate ≤ 1e-6 → **PASS**.
Recorded in `artifacts/m3_sec_imeanflow/endpoint_algebra_audit.json`.

## 4. Frozen finite-difference operators (§6 audit, already run)

Read from `src/ppg2ecg/evaluation/m1_structural.py` L39-47, not inferred from metric names:

| | formula | frozen impl | length | padding | smoothing | fs scaling | R-dependent |
|---|---|---|---|---|---|---|---|
| **D1** | `x[n+1] − x[n]` | `np.diff(x.astype(float64))` | T−1 | none | none | none | no |
| **D2** | `x[n+2] − 2x[n+1] + x[n]` | `x[2:] − 2·x[1:-1] + x[:-2]` | T−2 | none | none | none | no |

Torch equivalents in `src/ppg2ecg/flow/endpoint_structure.py`: `d1(x) = x[...,1:] − x[...,:-1]`,
`d2(x) = x[...,2:] − 2·x[...,1:-1] + x[...,:-2]`.

Equivalence measured: float64 max |diff| **0.000e+00** for both (criterion ≤ 1e-12); matched-dtype
float32 **bit-identical** (exactly 0.0), which is the operator-identity test. **Deviation recorded:**
the specification asked for float32 max |diff| ≤ 1e-6 against the frozen operators, but the frozen
numpy casts to float64 internally, so a float32 tensor cannot meet 1e-6 absolute when |d2| ≈ 11 —
float32 eps × 11 = 1.3e-6, and the measured gap is 9.5e-7. The criterion is therefore stated
**relatively**: ≤ 2 float32 ulp of max |d2| (measured 0.77). This is a precision artefact, not an
operator mismatch, and is recorded rather than fudged.

**Frozen sentence, required verbatim:**

> M3 reuses the frozen finite-difference operators, but not the GT-R-dependent QRS support used by the
> evaluation metrics.

The evaluation metrics apply D1/D2 only inside `pred[r−CORE−1 : r+CORE+2]` with `CORE = 10` (80 ms)
around **GT R peaks** (`m1_structural` L101, L108, L112). The M3 auxiliary is computed over the
**whole** clean-endpoint waveform. No R peak, QRS mask, event mask, predicted peak, or other
target-derived support may enter it.

## 5. The t-dependent geometry, derived and frozen (§C)

From `x0_hat = z_t − t·u0` and `z_t = (1−t)x + t·e`:

```
x0_hat − x = t · (e − x − u0)
```

`t` is a per-sample scalar constant along the waveform axis, so it commutes with D1 and D2:

```
D1(x0_hat) − D1(x) = t · D1(e − x − u0)
D2(x0_hat) − D2(x) = t · D2(e − x − u0)
```

The squared SEC numerators therefore inherit a **t²** factor. Verified numerically in float64: the
identity holds to max error **5.33e-15**, and halving `t` scales the per-sample L1 term by
**0.250000 ± 6.3e-16**, i.e. exactly k².

**Frozen design decision, taken before any training:** NO 1/t weighting, NO 1/t² weighting, NO
clipping by t, NO minimum-t threshold, NO time-dependent lambda, NO rebalanced t sampler.

> The endpoint structural auxiliary inherits the natural t-dependent scaling of the direct t→0
> endpoint construction. No inverse-t normalization is applied.

This is **not** claimed to be theoretically optimal. It is the frozen M3 definition.

**These diagnostics may never become tuning.** After this document is pushed, the TRAIN-only audit may
report L1, L2, L_SEC and gradient norm by fixed t bin `[0,0.2) [0.2,0.4) [0.4,0.6) [0.6,0.8) [0.8,1.0]`.
No gate depends on monotonicity, and no lambda, t reweighting, sampler change or low-t exclusion may
follow from them.

## 6. The loss — frozen exactly

```
z_t    = (1 - t) x + t e
u0     = net.u(z_t, ppg, t, t)
x0_hat = z_t - t.reshape(-1,1,1) * u0

d1_x, d1_hat = D1(x), D1(x0_hat)
d2_x, d2_hat = D2(x), D2(x0_hat)

s1 = mean(d1_x^2)   per sample, DETACHED
s2 = mean(d2_x^2)   per sample, DETACHED

L1 = mean((d1_hat - d1_x)^2) / (detach(s1) + 1e-6)
L2 = mean((d2_hat - d2_x)^2) / (detach(s2) + 1e-6)

L_SEC   = mean_batch( 0.5*L1 + 0.5*L2 )
L_total = L_iMF + 0.10 * L_SEC
```

Frozen: `eps_struct = 1e-6`, D1/D2 mixture `0.5 / 0.5`, `lambda_SEC = 0.10`.

**`L_iMF` must remain bit-identical to the historical baseline calculation.** No change to V, the JVP,
the adaptive weight `w`, `norm_p`, `norm_eps`, the t/r distribution, the noise, or the architecture.
No M2 `structure_weight` enters M3.

**Gradient policy (§9).** Gradients from `L_SEC` flow through `x0_hat → u0 →` parameters. The target
`x` is data and carries no gradient. Only the denominators `s1`, `s2` are detached; `u0`, `x0_hat`,
`d1_hat`, `d2_hat` are **not**.

**Why endpoint, not a velocity-derivative loss (§8).** M3 does **not** penalise `D(V − v_target)`,
because `v_target = e − x` contains the sampled Gaussian noise realisation and a derivative penalty on
it would emphasise the spatial derivative structure of the *noise*. M3 builds `x0_hat` and compares it
with the clean ECG `x`, so the auxiliary lives in target-waveform space.

**Extra forward pass (§10).** SEC-iMF adds one training-time call `net.u(z_t, ppg, t, t)`. Extra
**inference** compute is **zero**: at inference SEC-iMF uses the unmodified iMeanFlow sampler. Before
training, the model is audited for dropout, stochastic layers and running-stat mutation; if the extra
forward perturbs RNG or mutable state in a way that makes the U/E comparison ambiguous, **STOP**.

## 7. Baseline reference — no retrain

M2 established that fresh arm U reproduces A4 bit-exactly, so **arm U is reused, not retrained**:

| | |
|---|---|
| path | `outputs/m2_arm_U_seed42/checkpoint_best.pt` |
| file sha256 | `8beaff6cd9cb6db4b3bdf2023ae2676df439a9e94c0d1a34658627ec76d8a6d7` |
| state sha256 | `20ba7234f25e0fe30960ba0e441d4b9f751c20003c85fee80ec3c08367aa095e` |
| selected epoch | 45 |
| selection metric | `0.11945885431656277` |
| parameters | 4,568,707 |

**Baseline-path regression (§14), a hard gate.** M3 code with `lambda_SEC = 0` must reproduce the
untouched baseline path exactly, and must **bypass the SEC branch entirely** — no extra forward, no
endpoint construction. If an unavoidable extra stochastic forward remains at λ = 0, **STOP: verdict F,
BASELINE PATH INVALID**.

## 8. Data, budget and selection

TRAIN12 `e61 fex l38 n31 ngh p5d p9p qm9 trh tz8 u7y w4p`; VAL `an0 k2s`;
**TEST `kjd ssx` FORBIDDEN — never loaded at any M3 stage.** Preprocessing unchanged.

Seed **42**. Same initialization construction (init state sha256 `a79e527d…`), architecture, optimizer,
lr 1e-3, weight decay 0.01, t/r sampler, data order, noise RNG, batch 64, micro-batch 32, precision.

**Compute budget — the corrected figure.** M2's preregistration asserted 302,478 optimizer steps; that
was wrong, because this trainer's "epoch" is a `--val-every-steps 220` **validation round**, not a pass
over the data. The verified realised budget of A4 and of M2 arm U is:

| | |
|---|---|
| validation rounds | **66** |
| **optimizer steps** | **14,409** |

Arm E is frozen to **exactly 14,409 optimizer steps** under the same scheduling semantics, with early
stopping disabled, so E and U are step-for-step comparable. "66 rounds" is **not** 66 dataset epochs.

**Checkpoint selection** uses the same frozen historical rule — `fixed_imf_mse`, 4 banks,
`bank_seed 1000`, `min_delta 1e-4`. Selection may **never** use a derivative metric, a curvature
metric, event F1, or any M3 gate. U's selected epoch (45) and E's selected epoch are both recorded;
the primary comparison uses each arm's checkpoint under that same rule. Post-verdict descriptive
diagnostics may compare other pairings (M2 showed this matters), but **cannot change the M3 verdict**.

## 9. Evaluation and bootstrap

Development validation only (`an0`, `k2s`). Primary **NFE 4, source seed 0**. Frozen M2 evaluation
definitions, unmodified. Reported: RMSE, correlation; raw F1@50, floor F1@50, F1_excess@50, F1@100,
F1@150, F1@200, precision, recall, missing, spurious, `beats_ratio_dev`; `qrs_ptp_dev`,
`qrs_energy_dev`, `qrs_deriv_rmse`, `qrs_curvature_err`; HF error (`F4__ratio_dev`). No new metric may
be created after results.

Bootstrap: the frozen clustered procedure — all site rows sharing one target ECG move together,
subject-stratified, equal subject weight, **2,000 replicates, `default_rng(20260904)`**. Effect
convention: errors `U − E`, higher-is-better `E − U`; **positive always means SEC-iMF better**.

## 10. Gates

**G1** `qrs_deriv_rmse`: relative improvement ≥ **5 %** AND 95 % paired CI entirely > 0.
**G2** `qrs_curvature_err`: relative improvement ≥ **5 %** AND 95 % paired CI entirely > 0.
**G3** neither `an0` nor `k2s` may show **> 2 % worsening on both** `qrs_deriv_rmse` and
`qrs_curvature_err` simultaneously.

Non-inferiority: **N1** `f1_excess@50` no worse by > 0.020 absolute · **N2** `beats_ratio_dev` no worse
by > 0.020 absolute · **N3** `qrs_ptp_dev` relative worsening ≤ 5 % · **N4** `qrs_energy_dev` relative
worsening ≤ 5 % · **N5** `corr` absolute degradation ≤ 0.02 · **N6** RMSE relative worsening ≤ 5 % ·
**N7** HF error relative worsening ≤ 10 %.

Primary support requires **G1, G2, G3 and N1–N6**. N7 is a reported guardrail; if it alone fails the
verdict must explicitly say *spectral trade-off*.

## 11. Stop tree

- **G1 or G2 fails → verdict D, SEC-IMEANFLOW NOT SUPPORTED, STOP.** No lambda tuning, no dropping
  curvature, no derivative-only or curvature-only variant, no endpoint smoothing, no STFT, no QRS
  masking, no arm V, no multi-seed, no test.
- **G1/G2 pass but N1–N6 fail → verdict B**, STOP.
- Otherwise train **arm V** (value-endpoint control, §22): identical extra forward, `x0_hat`, seed,
  initialization, budget, architecture and checkpoint rule, but
  `L_VALUE = mean_b[ mean((x0_hat−x)²) / (detach(mean(x²)) + 1e-6) ]`, no derivative terms,
  `L = L_iMF + 0.10 · L_VALUE`. Then **S1** E beats V on `qrs_deriv_rmse` with CI > 0, **S2** on
  `qrs_curvature_err` with CI > 0, **S3** at least one with ≥ 3 % relative advantage. Failure →
  **verdict C, ENDPOINT SUPERVISION SUPPORTED, STRUCTURAL SPECIFICITY NOT SUPPORTED.**
- NFE 1/2/4 and source seeds 0–3 are evaluated **only after** the primary gates pass.

Verdicts: **A** supported · **B** unacceptable trade-off · **C** endpoint supervision supported but
specificity not · **D** not supported · **E** endpoint construction invalid · **F** baseline path invalid.

Exactly **three** arms may exist in M3: U, E, V. No λ = 0.05/0.2, no D1-only, no D2-only, no QRS-only —
each would require a new preregistration.

## 12. Claim boundary

M3 claims none of: that R timing is solved; that PPG determines QRS morphology; that derivative or
Sobolev losses are novel in general; that ECG structure is causally determined by PPG; that target
morphology is fully observable from PPG; SOTA; or that M2 was successful. **M2 remains
STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED, unchanged.**

M3 uses development subjects only, seed 42 only, no fresh test, no C2, no external dataset. The target
ECG is used only to build the **training** auxiliary. Two validation subjects means no population claim.

## 13. Deviations

Recorded in a dated section of the M3 **report**, never by editing this document. One is already
recorded above: the float32 operator-equivalence criterion (§4).
