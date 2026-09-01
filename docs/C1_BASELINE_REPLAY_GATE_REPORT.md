# C1 Stage 1 — Baseline Replay Gate — REPORT

## **GATE: PASS** — proceed to Stage 2 (train `C1-H25`, `C1-H50`)

Protocol: `docs/C1_INTERVAL_EXPOSURE_CONTROL_PREREGISTRATION.md` (`b32c952`).
Implementation-only commit, pushed before training: `38eaf45`.

**No test access. No architecture, loss or optimiser change. Only the (t, r) sampler may differ, and in
arm B it does not differ at all.**

---

## 1. The replay is bit-exact, and that is the first result

`C1-B` was trained from scratch under the recovered A4 recipe with `--c1-arm B` (the no-op path). It did
not merely land close to the historical run — it reproduced it exactly:

| | A4 original | C1-B replay |
|---|---|---|
| rounds run | 66 | **66** |
| best round | 45 | **45** |
| best selection metric | 0.11945885431656277 | **0.11945885431656277** |
| early stopped | True | True |
| peak VRAM (MiB) | 19216.22998046875 | 19216.23681640625 |
| wall clock (s) | 12982.77 | 12911.09 |
| selection bank hash | `2cd144684498f279…` | `2cd144684498f279…` (identical) |
| **`state_dict` sha256** | `47d7ccb94e5dbf7190d777f852b18f10…` | **identical** |

The trained weights are byte-identical to the frozen A4 checkpoint. This confirms three things at once:
the `--c1-arm B` path is a genuine no-op rather than a lookalike; the trainer is deterministic end to end
on this machine; and the intervention arms will differ from a *provably* correct control.

**What it does NOT establish, stated plainly.** Because the replay is deterministic, Stage 1 demonstrates
**pipeline reproducibility, not stability across training-run variation.** The preregistered wording
("does the target gap survive a fresh, controlled training replay") is satisfied, but the gate is
correspondingly weaker than a seed-variation study would be, and no such study is preregistered here.
Training-seed variance remains unestimated for the whole of C1.

## 2. Development metrics — frozen C0 population, oracle-free, GT-fixed

2,048 windows (`an0` 1,024 + `k2s` 1,024), 19,834 GT beats, evaluation source bank seed 0, **the same
source tensor at NFE 2 and NFE 4**. Realised step counts asserted equal to the requested NFE.

| NFE | M1 \|QRS-E−1\| | M2 \|p2p−1\| | M3 QRS RMSE | M4 RMSE | M5 raw corr | M6 \|slope−1\| | F1 excess | beats dev |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.6148 | 0.2616 | 0.5580 | 0.4395 | 0.1036 | 0.2722 | +0.3066 | 0.1058 |
| 4 | **0.6056** | **0.2425** | **0.5462** | **0.4233** | 0.1040 | 0.2733 | **+0.3176** | 0.1067 |

These reproduce the C0 values exactly, as they must given identical weights.

## 3. Paired evidence, NFE 2 → NFE 4

Subject-stratified paired bootstrap, 2,000 resamples, `default_rng(20260901)`, equal subject weight.
Positive = NFE 4 better.

| metric | oriented Δ | 95 % CI | verdict |
|---|---:|---|---|
| **M1** \|QRS-E−1\| | +0.00921 | [+0.00329, +0.01492] | **improves** |
| **M2** \|p2p−1\| | +0.01914 | [+0.01507, +0.02317] | **improves** |
| **M3** QRS RMSE | +0.01175 | [+0.01046, +0.01304] | **improves** |
| **M4** RMSE | +0.01620 | [+0.01470, +0.01768] | **improves** |
| M5 raw corr | +0.00042 | [−0.00101, +0.00186] | unresolved |
| M6 \|slope−1\| | −0.00106 | [−0.00559, +0.00351] | unresolved |
| *F1 excess* | +0.01101 | [+0.00715, +0.01473] | improves |
| *beats-ratio dev* | −0.00088 | [−0.00481, +0.00301] | unresolved |

## 4. Gate evaluation (preregistration §9)

1. at least **three** of M1–M4 with CI entirely > 0 → **4 of 4** ✓
2. **none** of M1–M4 with CI entirely < 0 → **0** ✓
3. event F1 excess does not clearly collapse → it *improves* ✓

**PASS.** Stage 2 is unblocked: train `C1-H25` and `C1-H50` under the frozen protocol, with no tuning
after observing C1-B.

## 5. Replay gap `G_m`, the Stage-2 denominator

Gap closure in Stage 2 is measured against this replay, not against the old C0 numbers. With Q oriented so
higher = better, `G_m = Q(B, NFE4) − Q(B, NFE2)`:

| metric | `G_m` |
|---|---:|
| M1 \|QRS-E−1\| | +0.00921 |
| M2 \|p2p−1\| | +0.01914 |
| M3 QRS RMSE | +0.01175 |
| M4 RMSE | +0.01620 |

All four are positive, so `C_m` is computable for all four.

## 6. Confirmations

- `kjd`/`ssx` never loaded; `assert_no_test_subjects` called before first data access.
- `outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt` md5 `31c042d291052fbb6dc15263ad316be2` —
  unchanged; C1 writes only to `outputs/c1_imf_*`.
- Architecture, loss, optimiser, schedule and checkpoint-selection rule unchanged; selection bank hash
  identical to A4's.
- No oracle metric appears in any table or decision above.
- Single training seed (42); the bootstrap quantifies development-window uncertainty only.

## Artifacts

`artifacts/c1_interval_exposure/`: `sampler_exposure.csv`, `rng_control.json`, `stage1_metrics.csv`,
`stage1_paired.csv`, `stage1_result.json`.
